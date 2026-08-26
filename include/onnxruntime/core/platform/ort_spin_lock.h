// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/common/spin_pause.h"
#include <atomic>

namespace onnxruntime {
/*
OrtSpinLock implemented mutex semantic "lock-freely",
calling thread will not be put to sleep on blocked,
which reduces cpu usage on context switching.
*/
struct OrtSpinLock {
  using LockState = enum { Locked = 0,
                           Unlocked };

  void lock() noexcept {
    // Test-and-test-and-set: attempt the acquiring RMW, and while it is
    // contended spin on a plain (shared, read-only) load rather than hammering
    // the cacheline with repeated read-for-ownership traffic. Only once the
    // lock appears free do we retry the compare-exchange.
    for (;;) {
      LockState state = Unlocked;
      if (state_.compare_exchange_weak(state, Locked, std::memory_order_acq_rel, std::memory_order_relaxed)) {
        return;
      }
      // Contended: wait for the holder to release, reading the line in the
      // shared state so other spinners can share the cacheline. The acquiring
      // ordering is supplied by the compare_exchange above on the next retry.
      while (state_.load(std::memory_order_relaxed) == Locked) {
        concurrency::SpinPause();  // pause and retry
      }
    }
  }
  bool try_lock() noexcept {
    LockState state = Unlocked;
    return state_.compare_exchange_weak(state, Locked, std::memory_order_acq_rel, std::memory_order_relaxed);
  }
  void unlock() noexcept {
    state_.store(Unlocked, std::memory_order_release);
  }

 private:
  std::atomic<LockState> state_{Unlocked};
};
}  // namespace onnxruntime