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
//! Threads started by this crate, and by Pingora's offload pools, record
//! the address of a local near the top of their stack. An application
//! reads it back with [`base`] and compares it against [`here`], taken at
//! whatever depth it wants to measure; [`used_here`] is the difference,
//! in bytes.
//!
//! # What is marked, and from where
//!
//! Not every thread's mark comes from the same place, and the difference
//! is worth knowing before trusting a number to a few hundred bytes:
//!
//! | thread | marked from |
//! |---|---|
//! | no-steal worker | the thread body, above everything it runs |
//! | offload pool thread | the thread body, above everything it runs |
//! | work-stealing worker | tokio's `on_thread_start` callback |
//! | blocking pool thread | tokio's `on_thread_start` callback |
//!
//! The first two are true tops. The last two are not: tokio invokes the
//! callback through a boxed `Fn` from the same frame that later runs the
//! scheduler loop, so the mark lands *beside* the work rather than above
//! it, and a measurement taken under that frame under-reports by roughly
//! one closure frame. That is tens of bytes against budgets in the
//! megabytes, and it is the direction that reports less headroom used
//! rather than more, but it is a real difference between the two
//! flavors and this crate cannot close it: tokio exposes no hook that
//! runs at the true top of a worker thread.
//!
//! # Safety and cost
//!
//! Nothing here is `unsafe`. Taking the address of a local, casting a
//! reference to a raw pointer and casting that pointer to an integer are
//! all safe operations; only dereferencing would not be, and nothing
//! here dereferences.
//!
//! In a release build a [`StackMark`] is one `usize` and [`used_here`]
//! is a thread-local load, a compare and a subtract. In a debug build a
//! mark also carries the [`ThreadId`] that made it, and `used_here`
//! asserts the two agree, so comparing a mark against a stack it does
//! not belong to fails loudly in tests and costs nothing in production.
//!
//! # Example
//!
//! ```
//! # use pingora_runtime::worker_stack;
//! // Inside a Pingora runtime thread, at whatever depth you care about:
//! if let Some(bytes) = worker_stack::used_here(worker_stack::here()) {
//!     println!("{bytes} bytes of stack in use");
//! }
//! ```

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(debug_assertions)]
use std::thread::ThreadId;

/// Process-wide stack size for threads Pingora starts, or 0 for the
/// crate default.
///
/// Not every thread Pingora runs belongs to a [`Runtime`](crate::Runtime).
/// The offload pools in `pingora-core` build their own threads, outlive
/// any one runtime, and are constructed from four different call sites
/// none of which carry a `RuntimeOpts`. Threading the size through all
/// of them would mean widening four option structs; a process-wide
/// default is one value, set once at startup from the same
/// configuration key, that every spawn site can read.
static PROCESS_DEFAULT: AtomicUsize = AtomicUsize::new(0);

/// The stack size Pingora gives a thread when nothing more specific
/// applies.
///
/// [`crate::DEFAULT_THREAD_STACK_SIZE`] until
/// [`set_process_default_stack_size`] says otherwise.
pub fn process_default_stack_size() -> usize {
    match PROCESS_DEFAULT.load(Ordering::Relaxed) {
        0 => crate::DEFAULT_THREAD_STACK_SIZE,
        size => size,
    }
}

/// Set the process-wide default stack size.
///
/// Call once, during startup, before any runtime or offload pool is
/// built. `Server::run` does this from `ServerConf`. A zero restores the
/// crate default.
///
/// Threads already started keep the stack they were given: a stack size
/// is fixed when the thread is created and this cannot reach back.
pub fn set_process_default_stack_size(bytes: usize) {
    PROCESS_DEFAULT.store(bytes, Ordering::Relaxed);
}

/// A point on a thread's stack.
///
/// Produced by [`here`], and by the marking this crate does when it
/// starts a thread. In a debug build it remembers which thread it came
/// from so that [`used_here`] can refuse to compare marks across
/// threads; in a release build it is one `usize` and nothing else.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StackMark {
    addr: usize,
    #[cfg(debug_assertions)]
    thread: ThreadId,
}

impl StackMark {
    /// The address this mark records.
    ///
    /// Exposed for logging and for tests. Two marks are only
    /// meaningfully comparable when they came from the same thread,
    /// which is what [`used_here`] checks.
    pub fn addr(&self) -> usize {
        self.addr
    }

    #[inline]
    fn at(addr: usize) -> Self {
        StackMark {
            addr,
            #[cfg(debug_assertions)]
            thread: std::thread::current().id(),
        }
    }
}

thread_local! {
    /// This thread's recorded base, or 0 when it has none.
    ///
    /// Stored as a bare address rather than a [`StackMark`] so the
    /// `const` initializer below works: that is what keeps a read to a
    /// plain thread-local load with no lazy-initialization branch, which
    /// is what makes this cheap enough to sit on a proxy's request path.
    static BASE: Cell<usize> = const { Cell::new(0) };

    /// The stack size this thread was given, or 0 when unknown.
    static SIZE: Cell<usize> = const { Cell::new(0) };
}

/// Record the calling frame as this thread's stack base.
///
/// Call it once, as the first thing a thread does, from the thread body
/// itself. A second call is ignored: a base recorded from a deeper frame
/// would move the origin down and under-report every later measurement,
/// so the first mark wins and the rest are no-ops.
///
/// `stack_size` is the stack the thread was given, or 0 when the caller
/// does not know.
///
/// `#[inline(always)]` so the anchor lands in the caller's frame rather
/// than in a callee's, which sits below it.
#[inline(always)]
pub fn mark_thread_start(stack_size: usize) {
    if BASE.get() != 0 {
        return;
    }
    let anchor: u8 = 0;
    // A named local, so it cannot be promoted to a constant: its
    // address is a stack address.
    BASE.set(&anchor as *const u8 as usize);
    SIZE.set(stack_size);
}

/// This thread's stack base.
///
/// `None` on a thread no Pingora runtime or offload pool started, which
/// is the application's main thread and any thread it spawns itself.
#[inline]
pub fn base() -> Option<StackMark> {
    let addr = BASE.get();
    (addr != 0).then(|| StackMark::at(addr))
}

/// The stack size, in bytes, this thread was given.
///
/// Zero on a thread with no recorded base, and on one whose starter did
/// not know the size.
#[inline]
pub fn size() -> usize {
    SIZE.get()
}

/// A mark for the frame that calls this.
///
/// `#[inline(never)]` on purpose: an inlined probe would measure
/// whichever frame it was folded into rather than a frame of its own.
/// The mark it returns is therefore taken in `here`'s own frame, one
/// call below the caller, so a measurement built from it counts that
/// call's frame as well. That is a fixed overcount of a few tens of
/// bytes, in the direction that reports more stack used rather than
/// less.
#[inline(never)]
pub fn here() -> StackMark {
    let anchor: u8 = 0;
    StackMark::at(&anchor as *const u8 as usize)
}

/// Bytes of stack in use between this thread's base and `here`.
///
/// `None` when the current thread has no recorded base.
///
/// Stacks grow down on every target Pingora supports, so a deeper frame
/// has a numerically smaller address. A mark taken beside the frame that
/// recorded the base, rather than below it, can land a little above it;
/// that saturates to zero rather than wrapping, because "shallower than
/// the base" and "no depth to report" are the same answer.
///
/// In a debug build, comparing a mark against a thread it did not come
/// from trips a `debug_assert`. In a release build the check is gone and
/// so is the thread id it would have needed.
#[inline]
pub fn used_here(here: StackMark) -> Option<usize> {
    let base = base()?;
    #[cfg(debug_assertions)]
    debug_assert_eq!(
        here.thread, base.thread,
        "a stack mark was compared against a different thread's stack; \
         the difference between two threads' addresses is not a depth"
    );
    Some(base.addr.saturating_sub(here.addr))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_process_default_is_the_crate_default_until_it_is_set() {
        assert_eq!(
            process_default_stack_size(),
            crate::DEFAULT_THREAD_STACK_SIZE
        );
        set_process_default_stack_size(3 * 1024 * 1024);
        assert_eq!(process_default_stack_size(), 3 * 1024 * 1024);
        // A zero restores the default rather than producing a thread
        // with no stack.
        set_process_default_stack_size(0);
        assert_eq!(
            process_default_stack_size(),
            crate::DEFAULT_THREAD_STACK_SIZE
        );
    }

    #[test]
    fn an_unmarked_thread_reports_no_base() {
        // The test harness thread was not started by a Pingora runtime.
        assert_eq!(base(), None);
        assert_eq!(size(), 0);
        assert_eq!(used_here(here()), None);
    }

    #[test]
    fn a_marked_thread_reports_the_depth_below_its_base() {
        let handle = std::thread::spawn(|| {
            mark_thread_start(1234);
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

    /// A second mark must not move the origin.
    ///
    /// The offload pools and the runtime both mark their threads, and a
    /// future caller could add a third. If a later, deeper call won, the
    /// base would slide down the stack and every measurement after it
    /// would report less than the truth, which is the direction that
    /// hides an overflow rather than showing it.
    #[test]
    fn the_first_mark_wins() {
        let handle = std::thread::spawn(|| {
            mark_thread_start(4096);
            let first = base().expect("marked");

            fn remark_from_a_deeper_frame() {
                let pad = [0u8; 4096];
                std::hint::black_box(&pad);
                mark_thread_start(9999);
            }
            remark_from_a_deeper_frame();

            assert_eq!(base(), Some(first), "a later mark must not move the base");
            assert_eq!(size(), 4096, "nor the recorded size");
        });
        handle.join().expect("the probe thread runs to completion");
    }

    #[test]
    fn an_address_above_the_base_reports_no_depth() {
        let handle = std::thread::spawn(|| {
            mark_thread_start(0);
            let above = StackMark::at(base().expect("marked").addr() + 4096);
            assert_eq!(
                used_here(above),
                Some(0),
                "a frame above the base has no depth to report, and must not wrap"
            );
        });
        handle.join().expect("the probe thread runs to completion");
    }

    /// The debug-only cross-thread check actually fires.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "different thread's stack")]
    fn a_mark_from_another_thread_is_refused() {
        let elsewhere = std::thread::spawn(here)
            .join()
            .expect("the other thread produces a mark");
        let handle = std::thread::spawn(move || {
            mark_thread_start(4096);
            let _ = used_here(elsewhere);
        });
        // Re-raise the inner panic so `should_panic` sees it.
        if let Err(payload) = handle.join() {
            std::panic::resume_unwind(payload);
        }
    }
}
