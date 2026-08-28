// Copyright 2026 Cloudflare, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Where a runtime thread's stack starts, and how big it is.
//!
//! A stack overflow aborts the process; it does not unwind, and it does
//! not say what was on the stack. The only way to see one coming is to
//! measure how deep the work actually goes, and that needs a fixed point
//! to measure from. This crate spawns the threads, so it is the only
//! place that can record one honestly.
//!
//! Every thread a [`Runtime`](crate::Runtime) starts records the address
//! of a local in its entry frame. An application reads it back with
//! [`base`] and compares it against the address of a local at whatever
//! depth it wants to measure. The difference is the number of bytes of
//! stack in use between the two, which is the number a stack budget is
//! written against.
//!
//! Nothing here is `unsafe`. Taking the address of a local, casting a
//! reference to a raw pointer and casting that pointer to an integer are
//! all safe operations; only dereferencing would not be, and nothing
//! here dereferences.
//!
//! # Example
//!
//! ```
//! # use pingora_runtime::worker_stack;
//! // Inside a Pingora runtime thread, at whatever depth you care about:
//! let used = worker_stack::used_here(worker_stack::here());
//! if let Some(bytes) = used {
//!     println!("{bytes} bytes of stack in use");
//! }
//! ```

use std::cell::Cell;

thread_local! {
    /// The address of a local in this thread's entry frame, or 0 when
    /// this thread was not started by a Pingora runtime.
    ///
    /// `const` initialized so a read compiles to a plain thread-local
    /// load with no lazy-initialization branch: this is read on a proxy's
    /// request path.
    static BASE: Cell<usize> = const { Cell::new(0) };

    /// The stack size this thread was given, or 0 when unknown.
    static SIZE: Cell<usize> = const { Cell::new(0) };
}

/// Record the current frame as this thread's stack base.
///
/// Called once per runtime thread, as the first thing that thread does.
/// Calling it again from a deeper frame would move the base and
/// under-report every later measurement, so it is deliberately not
/// public: only this crate, which owns thread startup, may set it.
///
/// `size` is the stack the thread was given, or 0 when the caller does
/// not know.
///
/// `#[inline(always)]` so the anchor lands in the caller's frame rather
/// than in a callee's. The callee frame sits below the caller's, so a
/// base recorded there is lower than a probe taken from the caller's own
/// frame, and the first measurement of a shallow call would come out
/// negative. Inlining puts the base at the top where it belongs.
#[inline(always)]
pub(crate) fn mark_base(size: usize) {
    let anchor: u8 = 0;
    // A named local, so it cannot be promoted to a constant: its
    // address is a stack address.
    BASE.with(|base| base.set(&anchor as *const u8 as usize));
    SIZE.with(|cell| cell.set(size));
}

/// The address of a local in this thread's entry frame.
///
/// Returns 0 when the current thread was not started by a Pingora
/// runtime, which is the case for the application's main thread and for
/// any thread the application spawns itself.
#[inline]
pub fn base() -> usize {
    BASE.with(Cell::get)
}

/// The stack size, in bytes, this thread was given.
///
/// Returns 0 when the current thread was not started by a Pingora
/// runtime.
#[inline]
pub fn size() -> usize {
    SIZE.with(Cell::get)
}

/// The address of a local in the calling frame.
///
/// `#[inline(never)]` on purpose: an inlined probe measures the frame it
/// was folded into rather than the frame that called it, which is a
/// quiet way to measure the wrong thing. One call is the cost.
#[inline(never)]
pub fn here() -> usize {
    let anchor: u8 = 0;
    &anchor as *const u8 as usize
}

/// Bytes of stack in use between this thread's base and `here`.
///
/// `here` comes from [`here`], called in the frame being measured.
/// Returns `None` only when the current thread has no recorded base,
/// which is every thread a Pingora runtime did not start.
///
/// Stacks grow down on every target Pingora supports, so a deeper frame
/// has a numerically smaller address. A probe taken in a frame beside
/// the one that recorded the base, rather than below it, can land a
/// little above the base; that saturates to zero rather than wrapping,
/// because "shallower than the base" and "no depth to report" are the
/// same answer.
#[inline]
pub fn used_here(here: usize) -> Option<usize> {
    let base = base();
    if base == 0 {
        return None;
    }
    Some(base.saturating_sub(here))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unmarked_thread_reports_no_base() {
        // The test harness thread was not started by a Pingora runtime.
        assert_eq!(base(), 0);
        assert_eq!(size(), 0);
        assert_eq!(used_here(here()), None);
    }

    #[test]
    fn a_marked_thread_reports_the_depth_below_its_base() {
        let handle = std::thread::spawn(|| {
            mark_base(1234);
            assert_eq!(size(), 1234);
            let shallow = used_here(here()).expect("a marked thread has a base");

            fn deeper(depth: u32) -> usize {
                // Recurse so the measured frame is genuinely below the
                // one `shallow` was taken in, whatever the optimizer
                // does to a single call.
                if depth == 0 {
                    used_here(here()).expect("a marked thread has a base")
                } else {
                    let used = deeper(depth - 1);
                    // Keep the frame alive past the recursive call so
                    // this is not a tail call.
                    std::hint::black_box(used)
                }
            }
            let deep = deeper(8);
            assert!(
                deep > shallow,
                "a deeper frame has to report more stack in use: {deep} vs {shallow}"
            );
        });
        handle.join().expect("the probe thread runs to completion");
    }

    #[test]
    fn an_address_above_the_base_reports_no_depth() {
        let handle = std::thread::spawn(|| {
            mark_base(0);
            let above = base() + 4096;
            assert_eq!(
                used_here(above),
                Some(0),
                "a frame above the base has no depth to report, and must not wrap"
            );
        });
        handle.join().expect("the probe thread runs to completion");
    }
}
