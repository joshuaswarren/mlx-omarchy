// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "mlx/backend/omarchy/ane/runtime_detail.h"

#include <cerrno>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <string>
#include <utility>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace mlx::core::omarchy::ane::detail {

class RuntimeOwnership {
 public:
  RuntimeOwnership() = default;

  ~RuntimeOwnership() {
    if (lock_fd_ >= 0 && !preserve_quarantine_) {
      clear_state_noexcept();
    }
    close_fds();
  }

  RuntimeOwnership(const RuntimeOwnership&) = delete;
  RuntimeOwnership& operator=(const RuntimeOwnership&) = delete;

  RuntimeOwnership(RuntimeOwnership&& other) noexcept
      : lock_fd_(other.lock_fd_),
        state_fd_(other.state_fd_),
        state_path_(std::move(other.state_path_)),
        preserve_quarantine_(other.preserve_quarantine_) {
    other.lock_fd_ = -1;
    other.state_fd_ = -1;
  }

  RuntimeOwnership& operator=(RuntimeOwnership&& other) noexcept {
    if (this != &other) {
      if (lock_fd_ >= 0 && !preserve_quarantine_) {
        clear_state_noexcept();
      }
      close_fds();
      lock_fd_ = other.lock_fd_;
      state_fd_ = other.state_fd_;
      state_path_ = std::move(other.state_path_);
      preserve_quarantine_ = other.preserve_quarantine_;
      other.lock_fd_ = -1;
      other.state_fd_ = -1;
    }
    return *this;
  }

  static RuntimeOwnership acquire() {
    return acquire_at(
        kRuntimeOwnershipLockPath,
        kRuntimeQuarantinePath,
        read_text_file("/proc/sys/kernel/random/boot_id"));
  }

  static RuntimeOwnership acquire_at(
      const std::filesystem::path& lock_path,
      const std::filesystem::path& state_path,
      std::string boot_id) {
    trim_line_end(boot_id);
    if (!valid_boot_id(boot_id)) {
      throw runtime_error("current boot identity is invalid");
    }

    RuntimeOwnership ownership;
    ownership.lock_fd_ = open_regular(lock_path);
    if (::flock(ownership.lock_fd_, LOCK_EX | LOCK_NB) != 0) {
      const int error = errno;
      if (error == EWOULDBLOCK || error == EAGAIN) {
        throw runtime_error("another ANE runtime owns the host device");
      }
      throw runtime_error(
          "ANE ownership lock failed: " + std::string(std::strerror(error)));
    }
    ownership.state_path_ = state_path;
    ownership.preserve_quarantine_ = true;
    ownership.state_fd_ = open_regular(state_path);
    std::string current_state = read_fd(ownership.state_fd_);
    trim_line_end(current_state);
    if (!current_state.empty() && !valid_boot_id(current_state)) {
      ownership.preserve_quarantine_ = true;
      throw runtime_error("ANE runtime quarantine state is invalid");
    }
    if (current_state == boot_id) {
      ownership.preserve_quarantine_ = true;
      throw runtime_error(
          "ANE runtime is quarantined for this boot; reboot is required");
    }
    write_fd(ownership.state_fd_, boot_id + "\n");
    ownership.preserve_quarantine_ = false;
    return ownership;
  }

  int lock_fd() const {
    if (lock_fd_ < 0) {
      throw runtime_error("ANE runtime ownership is not active");
    }
    return lock_fd_;
  }

  void arm() {
    preserve_quarantine_ = true;
  }

  void quarantine() noexcept {
    preserve_quarantine_ = true;
  }

  void release_cleanly() {
    if (!state_path_.empty() && ::unlink(state_path_.c_str()) != 0 &&
        errno != ENOENT) {
      throw runtime_error(
          "ANE ownership state removal failed: " +
          std::string(std::strerror(errno)));
    }
    preserve_quarantine_ = false;
    close_fds();
  }

 private:
  static void trim_line_end(std::string& value) {
    while (!value.empty() && (value.back() == '\n' || value.back() == '\r')) {
      value.pop_back();
    }
  }

  static bool valid_boot_id(const std::string& value) {
    if (value.size() != 36) {
      return false;
    }
    for (size_t i = 0; i < value.size(); ++i) {
      const bool separator = i == 8 || i == 13 || i == 18 || i == 23;
      if (separator ? value[i] != '-' : !std::isxdigit(
              static_cast<unsigned char>(value[i]))) {
        return false;
      }
    }
    return true;
  }

  static std::string read_text_file(const std::filesystem::path& path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
      throw runtime_error(
          "cannot read boot identity: " + std::string(std::strerror(errno)));
    }
    std::string value;
    try {
      value = read_fd(fd);
    } catch (...) {
      ::close(fd);
      throw;
    }
    ::close(fd);
    return value;
  }

  static int open_regular(const std::filesystem::path& path) {
    int fd = ::open(
        path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0666);
    if (fd < 0) {
      throw runtime_error(
          "cannot open ANE ownership state " + path.string() + ": " +
          std::strerror(errno));
    }
    struct stat status {};
    if (::fstat(fd, &status) != 0 || !S_ISREG(status.st_mode) ||
        status.st_nlink != 1) {
      ::close(fd);
      throw runtime_error("ANE ownership state is not a private regular file: " + path.string());
    }
    return fd;
  }

  static std::string read_fd(int fd) {
    if (::lseek(fd, 0, SEEK_SET) < 0) {
      throw runtime_error(
          "ANE ownership state seek failed: " + std::string(std::strerror(errno)));
    }
    std::string value;
    char buffer[128];
    while (true) {
      ssize_t count = ::read(fd, buffer, sizeof(buffer));
      if (count == 0) {
        return value;
      }
      if (count < 0) {
        if (errno == EINTR) {
          continue;
        }
        throw runtime_error(
            "ANE ownership state read failed: " + std::string(std::strerror(errno)));
      }
      value.append(buffer, static_cast<size_t>(count));
      if (value.size() > 128) {
        throw runtime_error("ANE ownership state is invalid");
      }
    }
  }

  static void write_fd(int fd, const std::string& value) {
    if (::ftruncate(fd, 0) != 0 || ::lseek(fd, 0, SEEK_SET) < 0) {
      throw runtime_error(
          "ANE ownership state reset failed: " + std::string(std::strerror(errno)));
    }
    size_t offset = 0;
    while (offset < value.size()) {
      ssize_t count = ::write(fd, value.data() + offset, value.size() - offset);
      if (count < 0 && errno == EINTR) {
        continue;
      }
      if (count <= 0) {
        throw runtime_error(
            "ANE ownership state write failed: " + std::string(std::strerror(errno)));
      }
      offset += static_cast<size_t>(count);
    }
    if (::fsync(fd) != 0) {
      throw runtime_error(
          "ANE ownership state sync failed: " + std::string(std::strerror(errno)));
    }
  }

  void clear_state_noexcept() noexcept {
    if (!state_path_.empty()) {
      ::unlink(state_path_.c_str());
    }
  }

  void close_fds() noexcept {
    if (state_fd_ >= 0) {
      ::close(state_fd_);
      state_fd_ = -1;
    }
    if (lock_fd_ >= 0) {
      ::close(lock_fd_);
      lock_fd_ = -1;
    }
  }

  int lock_fd_{-1};
  int state_fd_{-1};
  std::filesystem::path state_path_;
  bool preserve_quarantine_{false};
};

} // namespace mlx::core::omarchy::ane::detail
