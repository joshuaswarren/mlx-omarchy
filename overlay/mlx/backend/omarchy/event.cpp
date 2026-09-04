// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Event implementation on Vulkan timeline semaphores. GPU-stream events are
// real device synchronization objects whose waits and signals are carried in
// stream submissions. Events on CPU streams use a host condition counter,
// the same device model the upstream no-GPU and CUDA backends use for host
// streams (CPU streams only schedule host work in Omarchy builds).

#include "mlx/event.h"

#include <cassert>

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/vulkan.h"
#include "mlx/scheduler.h"

namespace mlx::core {

namespace {

// Host-side counter for events on CPU streams.
struct HostCounter {
  uint64_t value{0};
  std::mutex mtx;
  std::condition_variable cv;

  void wait(uint64_t v) {
    std::unique_lock<std::mutex> lk(mtx);
    if (value >= v) {
      return;
    }
    cv.wait(lk, [this, v] { return value >= v; });
  }

  void signal(uint64_t v) {
    {
      std::lock_guard<std::mutex> lk(mtx);
      value = std::max(value, v);
    }
    cv.notify_all();
  }

  bool signaled(uint64_t v) {
    std::lock_guard<std::mutex> lk(mtx);
    return value >= v;
  }
};

// Device-side timeline semaphore for events on GPU streams.
struct TimelineSemaphore {
  explicit TimelineSemaphore(omarchy::Device& device) : device(device) {
    VkSemaphoreTypeCreateInfo type{
        VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    type.initialValue = 0;
    VkSemaphoreCreateInfo ci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    ci.pNext = &type;
    VKX_CHECK(
        omarchy::vk::device_table().CreateSemaphore(
            device.handle(), &ci, nullptr, &semaphore));
  }

  ~TimelineSemaphore() {
    if (semaphore != VK_NULL_HANDLE) {
      omarchy::vk::device_table().DestroySemaphore(
          device.handle(), semaphore, nullptr);
    }
  }

  uint64_t counter() const {
    uint64_t counter_value = 0;
    VKX_CHECK(
        omarchy::vk::device_table().GetSemaphoreCounterValue(
            device.handle(), semaphore, &counter_value));
    return counter_value;
  }

  omarchy::Device& device;
  VkSemaphore semaphore{VK_NULL_HANDLE};
};

struct EventImpl {
  std::atomic<Error*> error{nullptr};
  std::unique_ptr<TimelineSemaphore> gpu;
  std::unique_ptr<HostCounter> host;
  // Completion timeline generation that carried the latest GPU-stream
  // signal (0 = none captured, e.g. CPU-stream producer), and the event
  // value that generation had signaled. Waiting threads join this
  // generation so the signal's completion handlers have run before the
  // waiter proceeds; a device waiter may wait on the completion
  // semaphore at this generation instead of the event semaphore when
  // this value covers what it needs.
  std::atomic<uint64_t> signaled_completion{0};
  std::atomic<uint64_t> signaled_value{0};
  // Set once a signal is queued on the device instead of issued from the
  // host. Queued and host signals of one event must not mix: a host
  // signal can pass an in-flight queued value and corrupt the timeline.
  std::atomic<bool> queued_signal{false};

  bool is_created() const {
    return gpu || host;
  }

  void ensure_created(Stream s) {
    if (is_created()) {
      return;
    }
    if (s.device == Device::gpu) {
      gpu =
          std::make_unique<TimelineSemaphore>(omarchy::device(s.device.index));
    } else {
      host = std::make_unique<HostCounter>();
    }
  }
};

} // namespace

Event::Event(Stream s) : stream_(s) {
  event_ = std::make_shared<EventImpl>();
}

void Event::wait() {
  check_error();
  if (value() == 0) {
    // Timeline counters start satisfied at zero.
    return;
  }
  auto& event = cast<EventImpl>();
  if (event.gpu) {
    // No-progress watchdog: blocks until the timeline counter reaches
    // |value_| OR throws if the counter fails to advance. Slow work
    // advances the counter steadily and is not misdiagnosed as a hang.
    omarchy::wait_for_timeline_progress(
        event.gpu->device.handle(), event.gpu->semaphore, value_);
    // Host reads may follow immediately: join the dispatcher handlers of
    // every already-completed submission (handler-written bytes, task
    // accounting). Later queued work is not awaited.
    //
    // The generation that carried this event's signal is joined exactly:
    // completions().wait() publishes drained_value_ only after that
    // generation's handlers ran, so handler-written state is visible
    // here. The signaler records the generation right after QueueSubmit
    // returns, which can trail the driver's semaphore signal this wait
    // just observed, so re-check the recorded generation after each
    // drain and join again if it moved.
    for (int join_attempt = 0; join_attempt < 8; ++join_attempt) {
      uint64_t generation =
          event.signaled_completion.load(std::memory_order_acquire);
      if (generation != 0) {
        event.gpu->device.completions().wait(generation);
      }
      if (event.signaled_completion.load(std::memory_order_acquire) ==
          generation) {
        break;
      }
      std::this_thread::yield();
    }
    event.gpu->device.join_completed_handlers();
    omarchy::allocator().invalidate_noncoherent(event.gpu->device.handle());
  } else if (event.host) {
    event.host->wait(value());
  }
  check_error();
}

void Event::wait(Stream s) {
  auto& event = cast<EventImpl>();
  // A wait may be recorded before any signal exists (consumer stream ahead
  // of producer). Create the backing semaphore here or the wait would be
  // dropped and cross-stream order silently lost.
  event.ensure_created(s);
  if (value() == 0) {
    return;
  }
  if (event.gpu) {
    if (s.device == Device::gpu) {
      auto& encoder = omarchy::get_command_encoder(s);
      // Retain this event's shared state in the encoder: the queued wait
      // must outlive any Event handle the caller drops before commit.
      encoder.add_semaphore_wait(event.gpu->semaphore, value(), event_);
    } else {
      uint64_t target_value = value();
      scheduler::wait_event(s, *this, [target_value](Event& self) {
        auto& impl = self.cast<EventImpl>();
        // Same no-progress watchdog as the direct host wait above.
        omarchy::wait_for_timeline_progress(
            impl.gpu->device.handle(), impl.gpu->semaphore, target_value);
        // Same handler-boundary join as Event::wait().
        impl.gpu->device.join_completed_handlers();
      });
    }
  } else if (event.host) {
    if (s.device == Device::gpu) {
      // GPU streams cannot wait on a host counter: that needs
      // device-visible synchronization, which one-queue timeline
      // semaphores cannot express for host-produced values. Unreachable in
      // release-equivalent builds: the CPU backend is compiled out there,
      // so no CPU stream exists to create a host-type event.
      throw std::runtime_error(
          "[omarchy] GPU streams cannot wait on a host counter event;"
          " only host-to-GPU signal bridging is supported.");
    }
    uint64_t target_value = value();
    scheduler::wait_event(s, *this, [target_value](Event& self) {
      self.cast<EventImpl>().host->wait(target_value);
    });
  }
}

void Event::signal(Stream s) {
  auto& event = cast<EventImpl>();
  event.ensure_created(s);
  if (event.gpu) {
    if (value() == 0) {
      // Signaling zero is invalid for a timeline semaphore; the counter is
      // already satisfied at zero.
      return;
    }
    if (s.device == Device::gpu) {
      auto& encoder = omarchy::get_command_encoder(s);
      // An idle encoder has no submission this signal could ride and no
      // queue work it could be ordered against, so the timeline counter
      // moves from the host. A lone signal in its own QueueSubmit costs
      // a driver submit, a dispatcher entry, and an in-flight dependency
      // for zero compute: the decode loop measured ~1,645 such
      // signal-only submissions per token against ~95 compute
      // dispatches. Order matters here: the generation is published
      // BEFORE the host signal, so a waiter that observes the signaled
      // counter always also observes a generation to join, and a device
      // waiter that misses both falls back to the event semaphore, which
      // this or a later signal still fires.
      if (!event.queued_signal.load(std::memory_order_acquire) &&
          encoder.idle()) {
        uint64_t generation = encoder.last_submitted_completion();
        uint64_t prior_gen =
            event.signaled_completion.load(std::memory_order_relaxed);
        while (generation > prior_gen &&
               !event.signaled_completion.compare_exchange_weak(
                   prior_gen, generation, std::memory_order_release)) {
        }
        uint64_t prior_val =
            event.signaled_value.load(std::memory_order_relaxed);
        while (value() > prior_val &&
               !event.signaled_value.compare_exchange_weak(
                   prior_val, value(), std::memory_order_release)) {
        }
        event.gpu->device.signal_timeline(event.gpu->semaphore, value());
      } else {
        event.queued_signal.store(true, std::memory_order_release);
        // Same ownership rule as waits: the signal is owned by the
        // encoder until its submission completes.
        encoder.add_semaphore_signal(event.gpu->semaphore, value(), event_);
        // Signal ordering is a flush contract: the semaphore must be on
        // the queue now, submitted immediately (flush contract).
        encoder.commit();
        // Record the generation that carries this signal so waiters can
        // join its handler boundary (see Event::wait).
        uint64_t generation = encoder.last_submitted_completion();
        uint64_t prior_gen =
            event.signaled_completion.load(std::memory_order_relaxed);
        while (generation > prior_gen &&
               !event.signaled_completion.compare_exchange_weak(
                   prior_gen, generation, std::memory_order_release)) {
        }
        uint64_t prior_val =
            event.signaled_value.load(std::memory_order_relaxed);
        while (value() > prior_val &&
               !event.signaled_value.compare_exchange_weak(
                   prior_val, value(), std::memory_order_release)) {
        }
      }
    } else {
      // Signal from a CPU stream by submitting the signal on the device
      // queue (mirrors the CUDA backend's dedicated signal stream).
      event.gpu->device.signal_timeline(event.gpu->semaphore, value());
    }
  } else if (event.host) {
    if (s.device == Device::gpu) {
      // Host-to-GPU bridge: the host counter advances when this stream's
      // submitted work completes, because the dispatcher runs this handler
      // at the completion boundary. The handler retains the event state.
      uint64_t target_value = value();
      HostCounter* host = event.host.get();
      std::shared_ptr<void> keepalive = event_;
      auto& encoder = omarchy::get_command_encoder(s);
      encoder.add_completed_handler(
          [host, target_value, keepalive]() { host->signal(target_value); });
      // Host-counter advance is a flush contract: a waiter may block on
      // this handler, so the submission must be enqueued immediately.
      encoder.commit();
      return;
    }
    // Enqueue the signal on the producer stream's queue. CPU primitives
    // dispatch their work as scheduler tasks on this same queue (cpu
    // encoder dispatch), so a signal issued inline on the caller's thread
    // can overtake work that is still queued and let a waiter read
    // half-finished buffers. The queue ordering mirrors the Metal
    // backend's cpu-stream signal path.
    uint64_t target_value = value();
    scheduler::signal_event(s, *this, [target_value](Event& self) {
      self.cast<EventImpl>().host->signal(target_value);
    });
  }
}

bool Event::is_signaled() const {
  auto& event = cast<EventImpl>();
  if (!event.is_created()) {
    return false;
  }
  if (event.gpu) {
    return event.gpu->counter() >= value();
  }
  return event.host->signaled(value());
}

std::atomic<Error*>& Event::error() {
  return cast<EventImpl>().error;
}

} // namespace mlx::core
