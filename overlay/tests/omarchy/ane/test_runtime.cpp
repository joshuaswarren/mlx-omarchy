// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/ane/runtime_detail.h"
#include "mlx/backend/omarchy/ane/runtime_ownership.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/wait.h>

using namespace mlx::core::omarchy::ane;

namespace {

AneProgramBinding lane_binding() {
  AneProgramBinding binding;
  binding.tensor = "x";
  binding.channel = 5;
  binding.dtype = "float16";
  binding.shape = {1, 64, 1, 1};
  binding.nchw = {1, 64, 1, 1, 64, 64};
  binding.logical_bytes = 128;
  binding.allocation_bytes = 0x4000;
  binding.element_count = 64;
  binding.physical_elements = 64;
  return binding;
}

std::vector<uint8_t> dense_values(size_t elements) {
  std::vector<uint8_t> dense(elements * 2);
  for (size_t i = 0; i < elements; ++i) {
    dense[i * 2] = static_cast<uint8_t>(i);
    dense[i * 2 + 1] = static_cast<uint8_t>(0x80 + i);
  }
  return dense;
}

} // namespace

TEST_CASE("ANE runtime packs and unpacks qualified 64-byte lanes") {
  const auto binding = lane_binding();
  const auto dense = dense_values(64);

  const auto packed = detail::pack_binding(binding, dense);
  REQUIRE(packed.size() == binding.allocation_bytes);
  for (size_t i = 0; i < 64; ++i) {
    CHECK(packed[i * 64] == dense[i * 2]);
    CHECK(packed[i * 64 + 1] == dense[i * 2 + 1]);
    CHECK(std::all_of(
        packed.begin() + i * 64 + 2,
        packed.begin() + (i + 1) * 64,
        [](uint8_t value) { return value == 0; }));
  }

  std::vector<uint8_t> round_trip(dense.size(), 0xff);
  detail::unpack_binding(binding, packed, round_trip);
  CHECK(round_trip == dense);
}

TEST_CASE("ANE runtime preserves tensor regions outside a sliced binding") {
  auto binding = lane_binding();
  binding.shape = {1, 4, 1, 1};
  binding.logical_bytes = 8;
  binding.element_offset = 2;
  binding.element_count = 4;
  binding.physical_elements = 4;
  binding.nchw = {1, 4, 1, 1, 64, 64};
  const auto dense = dense_values(8);

  const auto packed = detail::pack_binding(binding, dense);
  std::vector<uint8_t> restored(dense.size(), 0x5a);
  detail::unpack_binding(binding, packed, restored);

  CHECK(restored[0] == 0x5a);
  CHECK(restored[3] == 0x5a);
  CHECK(std::equal(restored.begin() + 4, restored.begin() + 12, dense.begin() + 4));
  CHECK(restored[12] == 0x5a);
  CHECK(restored.back() == 0x5a);
}

TEST_CASE("ANE runtime honors plane stride for multiple channels") {
  AneProgramBinding binding;
  binding.tensor = "x";
  binding.dtype = "float16";
  binding.shape = {1, 2, 2, 2};
  binding.nchw = {1, 2, 2, 2, 32, 8};
  binding.logical_bytes = 16;
  binding.allocation_bytes = 0x4000;
  binding.element_count = 8;
  binding.physical_elements = 8;
  const auto dense = dense_values(8);

  const auto packed = detail::pack_binding(binding, dense);
  const std::array<size_t, 8> offsets = {0, 2, 8, 10, 32, 34, 40, 42};
  for (size_t i = 0; i < offsets.size(); ++i) {
    CHECK(packed[offsets[i]] == dense[i * 2]);
    CHECK(packed[offsets[i] + 1] == dense[i * 2 + 1]);
  }
  CHECK(packed[16] == 0);

  std::vector<uint8_t> restored(dense.size());
  detail::unpack_binding(binding, packed, restored);
  CHECK(restored == dense);
}

TEST_CASE("ANE runtime rejects invalid staging and deadlines before device work") {
  const auto binding = lane_binding();
  CHECK_THROWS_WITH_AS(
      detail::pack_binding(binding, std::vector<uint8_t>(126)),
      "[omarchy-ane] runtime: tensor 'x' dense staging byte count is 126, expected at least 128.",
      std::runtime_error);
  CHECK_THROWS_WITH_AS(
      detail::checked_deadline(std::chrono::milliseconds(0)),
      "[omarchy-ane] runtime: deadline must be positive.",
      std::invalid_argument);
  CHECK_THROWS_WITH_AS(
      detail::checked_deadline(std::chrono::milliseconds::max()),
      "[omarchy-ane] runtime: deadline exceeds the monotonic clock range.",
      std::invalid_argument);
}

TEST_CASE("ANE worker descriptors stay above every fixed child destination") {
  CHECK(detail::worker_source_fd_floor(0, 64) == 7);
  CHECK(detail::worker_source_fd_floor(3, 64) == 19);
  CHECK_THROWS_WITH_AS(
      detail::worker_source_fd_floor(3, 24),
      "[omarchy-ane] runtime: bundle payload count exceeds worker descriptor limit.",
      std::runtime_error);
}

TEST_CASE("ANE child receives each payload through a collision-free descriptor") {
  const auto root = std::filesystem::temp_directory_path() /
      ("mlx-omarchy-ane-descriptors-" + std::to_string(::getpid()));
  std::filesystem::remove_all(root);
  std::filesystem::create_directory(root);
  constexpr size_t payload_count = 3;
  const int floor = detail::worker_source_fd_floor(payload_count, 64);
  std::array<int, payload_count> sources{};
  for (size_t i = 0; i < payload_count; ++i) {
    const auto path = root / std::to_string(i);
    std::ofstream(path) << static_cast<char>('A' + i);
    const int original = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    REQUIRE(original >= 0);
    sources[i] = ::fcntl(original, F_DUPFD_CLOEXEC, floor);
    ::close(original);
    REQUIRE(sources[i] >= floor);
  }

  const pid_t child = ::fork();
  REQUIRE(child >= 0);
  if (child == 0) {
    for (size_t i = 0; i < payload_count; ++i) {
      if (::dup2(
              sources[i],
              detail::kWorkerPayloadFdBase + static_cast<int>(i)) < 0) {
        ::_exit(10);
      }
    }
    for (size_t i = 0; i < payload_count; ++i) {
      char value = 0;
      const int fd = detail::kWorkerPayloadFdBase + static_cast<int>(i);
      if (::lseek(fd, 0, SEEK_SET) < 0 || ::read(fd, &value, 1) != 1 ||
          value != static_cast<char>('A' + i)) {
        ::_exit(20 + static_cast<int>(i));
      }
    }
    ::_exit(0);
  }
  int status = 0;
  REQUIRE(::waitpid(child, &status, 0) == child);
  for (int fd : sources) {
    ::close(fd);
  }
  CHECK(WIFEXITED(status));
  CHECK(WEXITSTATUS(status) == 0);
  std::filesystem::remove_all(root);
}

TEST_CASE("ANE worker rejects a deadline at the hardware submission boundary") {
  CHECK_THROWS_WITH_AS(
      detail::ensure_worker_before(
          detail::RuntimeClock::now() - std::chrono::milliseconds(1),
          "ANE program execution submission"),
      "[omarchy-ane] runtime: deadline expired before ANE program execution submission.",
      std::runtime_error);
}
TEST_CASE("ANE ownership excludes peers and preserves a same-boot quarantine") {
  const auto root = std::filesystem::temp_directory_path() /
      ("mlx-omarchy-ane-ownership-" + std::to_string(::getpid()));
  std::filesystem::remove_all(root);
  std::filesystem::create_directory(root);
  const auto lock = root / "runtime.lock";
  const auto state = root / "quarantine";
  const std::string boot_a = "11111111-1111-1111-1111-111111111111";
  const std::string boot_b = "22222222-2222-2222-2222-222222222222";
  struct stat first_state {};

  {
    auto owner = detail::RuntimeOwnership::acquire_at(lock, state, boot_a);
    CHECK_THROWS_WITH_AS(
        detail::RuntimeOwnership::acquire_at(lock, state, boot_a),
        "[omarchy-ane] runtime: another ANE runtime owns the host device.",
        std::runtime_error);
    const pid_t peer = ::fork();
    REQUIRE(peer >= 0);
    if (peer == 0) {
      try {
        auto second = detail::RuntimeOwnership::acquire_at(lock, state, boot_a);
        (void)second;
        ::_exit(2);
      } catch (const std::runtime_error& error) {
        ::_exit(std::string(error.what()) ==
                "[omarchy-ane] runtime: another ANE runtime owns the host device."
            ? 0
            : 3);
      }
    }
    int peer_status = 0;
    REQUIRE(::waitpid(peer, &peer_status, 0) == peer);
    CHECK(WIFEXITED(peer_status));
    CHECK(WEXITSTATUS(peer_status) == 0);
    REQUIRE(::stat(state.c_str(), &first_state) == 0);
    owner.release_cleanly();
  }
  struct stat clean_state {};
  REQUIRE(::stat(state.c_str(), &clean_state) == 0);
  CHECK(clean_state.st_dev == first_state.st_dev);
  CHECK(clean_state.st_ino == first_state.st_ino);
  CHECK(std::filesystem::exists(state));
  CHECK(std::filesystem::file_size(state) == 0);
  {
    auto owner = detail::RuntimeOwnership::acquire_at(lock, state, boot_a);
    owner.arm();
    owner.quarantine();
  }
  CHECK_THROWS_WITH_AS(
      detail::RuntimeOwnership::acquire_at(lock, state, boot_a),
      "[omarchy-ane] runtime: ANE runtime is quarantined for this boot; reboot is required.",
      std::runtime_error);
  {
    auto owner = detail::RuntimeOwnership::acquire_at(lock, state, boot_b);
    owner.release_cleanly();
  }
  CHECK(std::filesystem::exists(state));
  CHECK(std::filesystem::file_size(state) == 0);

  std::ofstream(state) << std::string(129, 'x');
  CHECK_THROWS_WITH_AS(
      detail::RuntimeOwnership::acquire_at(lock, state, boot_a),
      "[omarchy-ane] runtime: ANE ownership state is invalid.",
      std::runtime_error);
  CHECK(std::filesystem::exists(state));
  CHECK_THROWS_WITH_AS(
      detail::RuntimeOwnership::acquire_at(lock, state, boot_a),
      "[omarchy-ane] runtime: ANE ownership state is invalid.",
      std::runtime_error);
  std::filesystem::remove_all(root);
}

TEST_CASE("ANE shared ownership paths require frozen provisioned identities") {
  const auto root = std::filesystem::temp_directory_path() /
      ("mlx-omarchy-ane-shared-ownership-" + std::to_string(::getpid()));
  std::filesystem::remove_all(root);
  std::filesystem::create_directory(root);
  REQUIRE(::chmod(root.c_str(), 0750) == 0);
  const auto lock = root / "device.lock";
  const auto state = root / "quarantine";
  std::ofstream{lock};
  std::ofstream{state};
  REQUIRE(::chmod(lock.c_str(), 0660) == 0);
  REQUIRE(::chmod(state.c_str(), 0660) == 0);

  detail::RuntimeOwnership::validate_shared_paths_at(
      root, lock, state, ::geteuid(), ::getegid());
  REQUIRE(::chmod(root.c_str(), 02750) == 0);
  CHECK_THROWS_AS(
      detail::RuntimeOwnership::validate_shared_paths_at(
          root, lock, state, ::geteuid(), ::getegid()),
      std::runtime_error);
  REQUIRE(::chmod(root.c_str(), 0750) == 0);
  REQUIRE(::chmod(lock.c_str(), 0640) == 0);
  CHECK_THROWS_WITH_AS(
      detail::RuntimeOwnership::validate_shared_paths_at(
          root, lock, state, ::geteuid(), ::getegid()),
      doctest::String{
          ("[omarchy-ane] runtime: ANE ownership state is not a provisioned shared regular file: " +
           lock.string() + ".")
              .c_str()},
      std::runtime_error);
  REQUIRE(::chmod(lock.c_str(), 0660) == 0);
  std::filesystem::remove(state);
  std::filesystem::create_symlink(lock, state);
  CHECK_THROWS_WITH_AS(
      detail::RuntimeOwnership::validate_shared_paths_at(
          root, lock, state, ::geteuid(), ::getegid()),
      doctest::String{
          ("[omarchy-ane] runtime: ANE ownership state is not a provisioned shared regular file: " +
           state.string() + ".")
              .c_str()},
      std::runtime_error);
  std::filesystem::remove_all(root);
}
