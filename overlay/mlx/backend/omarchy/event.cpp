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
    VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    info.semaphoreCount = 1;
    info.pSemaphores = &event.gpu->semaphore;
    info.pValues = &value_;
    VkResult res = omarchy::vk::device_table().WaitSemaphores(
        event.gpu->device.handle(), &info, omarchy::kSubmitTimeoutNs);
    if (res != VK_SUCCESS) {
      throw std::runtime_error(
          std::string("[omarchy] Vulkan event wait did not complete within ") +
          std::to_string(omarchy::kSubmitTimeoutNs / 1000000ull) + " ms (" +
          omarchy::vk::result_string(res) +
          "). The device may be hung; no CPU fallback is available.");
    }
    // Host reads may follow immediately: join the dispatcher handlers of
    // every already-completed submission (handler-written bytes, task
    // accounting). Later queued work is not awaited.
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
      // CPU stream waits on the host until the device counter advances.
      // The scheduler holds a copy of this event, so the semaphore stays
      // alive through the task; no raw pointers cross the boundary.
      uint64_t target_value = value();
      scheduler::wait_event(s, *this, [target_value](Event& self) {
        auto& impl = self.cast<EventImpl>();
        VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        info.semaphoreCount = 1;
        info.pSemaphores = &impl.gpu->semaphore;
        info.pValues = &target_value;
        VkResult res = omarchy::vk::device_table().WaitSemaphores(
            impl.gpu->device.handle(), &info, omarchy::kSubmitTimeoutNs);
        if (res != VK_SUCCESS) {
          throw std::runtime_error(
              std::string(
                  "[omarchy] Vulkan event wait did not complete within ") +
              std::to_string(omarchy::kSubmitTimeoutNs / 1000000ull) + " ms (" +
              omarchy::vk::result_string(res) +
              "). The device may be hung; no CPU fallback is available.");
        }
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
      // Same ownership rule as waits: the signal is owned by the encoder
      // until its submission completes.
      encoder.add_semaphore_signal(event.gpu->semaphore, value(), event_);
      encoder.commit();
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
      encoder.commit();
      return;
    }
    event.host->signal(value());
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
