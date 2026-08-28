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

//! Pingora tokio runtime.
//!
//! Tokio runtime comes in two flavors: a single-threaded runtime
//! and a multi-threaded one which provides work stealing.
//! Benchmark shows that, compared to the single-threaded runtime, the multi-threaded one
//! has some overhead due to its more sophisticated work steal scheduling.
//!
//! This crate provides a third flavor: a multi-threaded runtime without work stealing.
//! This flavor is as efficient as the single-threaded runtime while allows the async
//! program to use multiple cores.

use once_cell::sync::{Lazy, OnceCell};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
#[cfg(feature = "dial9")]
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::{self, JoinHandle, ThreadId};
use std::time::Duration;
use thread_local::ThreadLocal;
use tokio::runtime::{Builder, Handle};
use tokio::sync::oneshot::{channel, Sender};

/// Default maximum size of a dial9 trace segment file.
#[cfg(feature = "dial9")]
pub const DEFAULT_DIAL9_MAX_FILE_SIZE: u64 = 100 * 1024 * 1024;
/// Default maximum bytes retained locally by dial9.
#[cfg(feature = "dial9")]
pub const DEFAULT_DIAL9_MAX_TOTAL_SIZE: u64 = 512 * 1024 * 1024;

pub mod worker_stack;

/// Default stack size, in bytes, for threads this crate's runtimes spawn.
///
/// Tokio's own default is 2 MiB, and that is not enough. A Pingora
/// worker polls the whole application future chain on this stack, and a
/// debug build gives every async state machine far larger frames than a
/// release build does, so an application that fits comfortably in
/// production can abort in CI. The application this fork serves
/// measured its request path at just over 1 MiB of the 2 MiB default on
/// a macOS debug build with none of its optional filters wired, and
/// overflowed the same 2 MiB outright on a Linux debug build with them
/// wired. Half the budget was already spent before anyone looked.
///
/// 8 MiB is the `RLIMIT_STACK` default Linux gives a process's main
/// thread, which is the size the platform already considers normal for
/// a thread running arbitrary code. It is four times tokio's default,
/// which leaves room for years of ordinary growth rather than one more
/// release: a stack overflow cannot be refactored away in small steps,
/// because the frames belong to the whole call chain and not to any one
/// function in it.
///
/// The cost of raising it on a 64-bit target is address space, not
/// memory. A thread stack is an anonymous mapping committed page by page
/// as it is touched, so resident memory tracks the depth actually
/// reached and not the size reserved. Sixteen workers plus a server
/// thread reserve 136 MiB of a 128 TiB address space at this size and
/// resident nothing extra.
///
/// Override per runtime with the `thread_stack_size` field of
/// [`RuntimeOpts`], or per server with the `runtime_thread_stack_size`
/// configuration key.
pub const DEFAULT_THREAD_STACK_SIZE: usize = 8 * 1024 * 1024;

/// Configuration options for the blocking thread pool used by the runtime.
///
/// These options control the behavior of the blocking thread pool that handles
/// [`tokio::task::spawn_blocking`] tasks.
#[derive(Debug, Clone, Default)]
pub struct BlockingPoolOpts {
    /// The maximum number of threads in the blocking thread pool.
    ///
    /// When not set, the tokio default (512) is used.
    pub max_threads: Option<usize>,
    /// The duration that idle blocking threads are kept alive before being shut down.
    ///
    /// When not set, the tokio default (10 seconds) is used.
    pub thread_keep_alive: Option<Duration>,
}

/// Configuration options for runtime metrics collection.
#[derive(Debug, Clone, Default)]
pub struct RuntimeMetricsOpts {
    /// Enable Tokio's poll-time histogram on the runtime.
    ///
    /// This must be configured before the runtime is built. Enabling it adds
    /// two timestamp reads to every task poll.
    pub poll_time_histogram: bool,
    /// Histogram bucket scale for Tokio's poll-time histogram.
    pub poll_time_histogram_scale: Option<RuntimeMetricsPollTimeHistogramScale>,
    /// Width of the first histogram bucket.
    pub poll_time_histogram_resolution: Option<Duration>,
    /// Number of histogram buckets. Memory usage scales with runtimes × workers × buckets.
    pub poll_time_histogram_buckets: Option<usize>,
}

/// Configuration options for a Tokio runtime.
#[derive(Debug, Clone, Default)]
pub struct RuntimeOpts {
    /// Options for runtime metrics collection.
    pub metrics: RuntimeMetricsOpts,
    /// Enable Tokio's experimental alternative timer.
    ///
    /// This requires building with `--cfg tokio_unstable` and only applies to
    /// Tokio's multi-threaded runtime.
    pub enable_alt_timer: bool,
    /// Stack size, in bytes, for every thread this runtime spawns.
    ///
    /// When not set, [`DEFAULT_THREAD_STACK_SIZE`] is used. It applies to
    /// work-stealing worker threads, to the per-core threads of a
    /// no-steal runtime, and to the blocking pool.
    pub thread_stack_size: Option<usize>,
    /// Options for dial9 Tokio telemetry.
    #[cfg(feature = "dial9")]
    pub dial9: Option<Dial9RuntimeOpts>,
}

impl RuntimeOpts {
    /// The stack size these options ask for, in bytes.
    ///
    /// The `thread_stack_size` field when set, otherwise
    /// [`DEFAULT_THREAD_STACK_SIZE`]. A zero is treated as unset rather
    /// than passed on: tokio panics on a zero stack size, and a runtime
    /// that refuses to start is a worse answer to a bad config value
    /// than the default is.
    pub fn resolved_thread_stack_size(&self) -> usize {
        self.thread_stack_size
            .filter(|size| *size > 0)
            .unwrap_or(DEFAULT_THREAD_STACK_SIZE)
    }
}

/// Configuration options for dial9 Tokio telemetry.
#[cfg(feature = "dial9")]
#[derive(Debug, Clone)]
pub struct Dial9RuntimeOpts {
    /// Trace output path after server configuration defaults are applied.
    pub trace_path: PathBuf,
    /// Rotate trace segments after this many bytes.
    pub max_file_size: u64,
    /// Maximum bytes retained on local disk.
    pub max_total_size: u64,
    /// Wall-clock trace rotation period.
    pub rotation_period: Option<Duration>,
    /// Enable dial9 task spawn/terminate tracking.
    pub task_tracking: bool,
    /// How often the background worker checks for sealed trace segments.
    pub worker_poll_interval: Option<Duration>,
    /// Upload sealed trace segments to S3-compatible storage.
    #[cfg(feature = "dial9-worker-s3")]
    pub s3_upload: Option<Dial9S3UploadOpts>,
}

#[cfg(feature = "dial9")]
impl Dial9RuntimeOpts {
    /// Create dial9 runtime options using Pingora's dial9 defaults.
    pub fn new(trace_path: impl Into<PathBuf>) -> Self {
        Self {
            trace_path: trace_path.into(),
            max_file_size: DEFAULT_DIAL9_MAX_FILE_SIZE,
            max_total_size: DEFAULT_DIAL9_MAX_TOTAL_SIZE,
            rotation_period: None,
            task_tracking: true,
            worker_poll_interval: None,
            #[cfg(feature = "dial9-worker-s3")]
            s3_upload: None,
        }
    }

    /// Set the maximum size of each trace segment file.
    pub fn with_max_file_size(mut self, max_file_size: u64) -> Self {
        self.max_file_size = max_file_size;
        self
    }

    /// Set the maximum bytes retained on local disk.
    pub fn with_max_total_size(mut self, max_total_size: u64) -> Self {
        self.max_total_size = max_total_size;
        self
    }

    /// Set the wall-clock trace rotation period.
    pub fn with_rotation_period(mut self, rotation_period: Duration) -> Self {
        self.rotation_period = Some(rotation_period);
        self
    }

    /// Enable or disable dial9 task spawn/terminate tracking.
    pub fn with_task_tracking(mut self, task_tracking: bool) -> Self {
        self.task_tracking = task_tracking;
        self
    }

    /// Set how often the background worker checks for sealed trace segments.
    pub fn with_worker_poll_interval(mut self, worker_poll_interval: Duration) -> Self {
        self.worker_poll_interval = Some(worker_poll_interval);
        self
    }

    /// Set S3-compatible upload options for sealed trace segments.
    #[cfg(feature = "dial9-worker-s3")]
    pub fn with_s3_upload(mut self, s3_upload: Dial9S3UploadOpts) -> Self {
        self.s3_upload = Some(s3_upload);
        self
    }
}

/// Configuration options for dial9 S3-compatible trace uploads.
#[cfg(all(feature = "dial9", feature = "dial9-worker-s3"))]
#[derive(Debug, Clone)]
pub struct Dial9S3UploadOpts {
    /// S3 bucket that receives sealed trace segments.
    pub bucket: String,
    /// Service name included in uploaded object keys.
    pub service_name: String,
    /// Optional key prefix.
    pub prefix: Option<String>,
    /// Optional region override.
    pub region: Option<String>,
    /// Optional instance identifier included in uploaded object keys.
    pub instance_path: Option<String>,
    /// Optional pre-built S3 client for custom credentials or endpoints.
    pub client: Option<aws_sdk_s3::Client>,
}

#[cfg(all(feature = "dial9", feature = "dial9-worker-s3"))]
impl Dial9S3UploadOpts {
    /// Create S3-compatible upload options.
    pub fn new(bucket: impl Into<String>, service_name: impl Into<String>) -> Self {
        Self {
            bucket: bucket.into(),
            service_name: service_name.into(),
            prefix: None,
            region: None,
            instance_path: None,
            client: None,
        }
    }

    /// Set the object key prefix.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    /// Set the AWS region override.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Set the instance identifier included in uploaded object keys.
    pub fn with_instance_path(mut self, instance_path: impl Into<String>) -> Self {
        self.instance_path = Some(instance_path.into());
        self
    }

    /// Set a pre-built S3 client for custom credentials or endpoints.
    pub fn with_client(mut self, client: aws_sdk_s3::Client) -> Self {
        self.client = Some(client);
        self
    }
}

/// Bucket scale for Tokio's poll-time histogram.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeMetricsPollTimeHistogramScale {
    /// Equal-width buckets.
    Linear,
    /// Buckets double in width at each step.
    Log,
}

/// Pingora async multi-threaded runtime
///
/// The `Steal` flavor is effectively tokio multi-threaded runtime.
///
/// The `NoSteal` flavor is backed by multiple tokio single-threaded runtime.
pub enum Runtime {
    Steal {
        runtime: tokio::runtime::Runtime,
        #[cfg(feature = "dial9")]
        dial9_guard: Option<dial9_tokio_telemetry::telemetry::TelemetryGuard>,
    },
    NoSteal(NoStealRuntime),
}

/// Apply [`BlockingPoolOpts`] to a tokio [`Builder`].
fn apply_blocking_opts(builder: &mut Builder, opts: &BlockingPoolOpts) {
    if let Some(max) = opts.max_threads {
        builder.max_blocking_threads(max);
    }
    if let Some(ttl) = opts.thread_keep_alive {
        builder.thread_keep_alive(ttl);
    }
}

/// Apply [`RuntimeMetricsOpts`] to a tokio [`Builder`].
// The replacement `metrics_poll_time_histogram_configuration` API is not used
// here so this crate can continue to compile against older Tokio 1.x versions
// selected by downstream applications while still honoring these knobs in
// tokio-unstable builds.
#[allow(deprecated)]
fn apply_metrics_opts(builder: &mut Builder, opts: &RuntimeMetricsOpts) {
    #[cfg(tokio_unstable)]
    if opts.poll_time_histogram {
        builder.enable_metrics_poll_time_histogram();

        if let Some(scale) = opts.poll_time_histogram_scale {
            builder.metrics_poll_count_histogram_scale(match scale {
                RuntimeMetricsPollTimeHistogramScale::Linear => {
                    tokio::runtime::HistogramScale::Linear
                }
                RuntimeMetricsPollTimeHistogramScale::Log => tokio::runtime::HistogramScale::Log,
            });
        }
        if let Some(resolution) = opts
            .poll_time_histogram_resolution
            .filter(|resolution| !resolution.is_zero())
        {
            builder.metrics_poll_count_histogram_resolution(resolution);
        }
        if let Some(buckets) = opts
            .poll_time_histogram_buckets
            .filter(|buckets| *buckets > 0)
        {
            builder.metrics_poll_count_histogram_buckets(buckets);
        }
    }

    #[cfg(not(tokio_unstable))]
    let _ = (builder, opts);
}

/// Apply timer options from [`RuntimeOpts`] to a tokio [`Builder`].
fn apply_timer_opts(builder: &mut Builder, opts: &RuntimeOpts) {
    #[cfg(tokio_unstable)]
    if opts.enable_alt_timer {
        builder.enable_alt_timer();
    }

    #[cfg(not(tokio_unstable))]
    let _ = (builder, opts);
}

#[cfg(feature = "dial9")]
fn build_dial9_runtime(
    builder: Builder,
    runtime_name: &str,
    opts: &Dial9RuntimeOpts,
) -> std::io::Result<(
    tokio::runtime::Runtime,
    dial9_tokio_telemetry::telemetry::TelemetryGuard,
)> {
    use dial9_tokio_telemetry::telemetry::{RotatingWriter, TracedRuntime};
    use std::io::{Error, ErrorKind};

    if opts.max_file_size == 0 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "dial9 max_file_size must be greater than zero",
        ));
    }
    if opts.max_total_size == 0 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "dial9 max_total_size must be greater than zero",
        ));
    }
    if opts.max_file_size > opts.max_total_size {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "dial9 max_file_size must be less than or equal to max_total_size",
        ));
    }
    if opts.worker_poll_interval == Some(Duration::ZERO) {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "dial9 worker_poll_interval must be greater than zero",
        ));
    }

    if let Some(parent) = opts.trace_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let writer = RotatingWriter::builder()
        .base_path(opts.trace_path.clone())
        .max_file_size(opts.max_file_size)
        .max_total_size(opts.max_total_size)
        .maybe_rotation_period(opts.rotation_period)
        .build()?;

    let mut traced = TracedRuntime::builder()
        .with_trace_path(opts.trace_path.clone())
        .with_runtime_name(runtime_name)
        .with_task_tracking(opts.task_tracking);
    if let Some(worker_poll_interval) = opts.worker_poll_interval {
        traced = traced.with_worker_poll_interval(worker_poll_interval);
    }

    #[cfg(feature = "dial9-worker-s3")]
    if let Some(s3_upload) = &opts.s3_upload {
        if s3_upload.bucket.trim().is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "dial9 s3 bucket must not be empty",
            ));
        }
        if s3_upload.service_name.trim().is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "dial9 s3 service_name must not be empty",
            ));
        }
        let s3_config = dial9_tokio_telemetry::background_task::s3::S3Config::builder()
            .bucket(s3_upload.bucket.clone())
            .service_name(s3_upload.service_name.clone())
            .maybe_prefix(s3_upload.prefix.clone())
            .maybe_region(s3_upload.region.clone())
            .maybe_instance_path(s3_upload.instance_path.clone());
        let traced = traced.with_s3_uploader(s3_config.build());
        if let Some(client) = s3_upload.client.clone() {
            return traced
                .with_s3_client(client)
                .build_and_start(builder, writer);
        }
        return traced.build_and_start(builder, writer);
    }

    traced.build_and_start(builder, writer)
}

/// Builder for constructing a [`Runtime`].
///
/// # Example
///
/// ```
/// use pingora_runtime::{RuntimeBuilder, BlockingPoolOpts};
/// use std::time::Duration;
///
/// let rt = RuntimeBuilder::new(4, "my-service")
///     .blocking_pool_opts(BlockingPoolOpts {
///         max_threads: Some(64),
///         thread_keep_alive: Some(Duration::from_secs(30)),
///     })
///     .build();
/// ```
pub struct RuntimeBuilder {
    threads: usize,
    name: String,
    work_steal: bool,
    blocking_pool_opts: BlockingPoolOpts,
    runtime_opts: RuntimeOpts,
}

impl RuntimeBuilder {
    /// Create a new builder with the given number of worker threads and runtime name.
    ///
    /// Work stealing is enabled by default.
    pub fn new(threads: usize, name: &str) -> Self {
        Self {
            threads,
            name: name.to_string(),
            work_steal: true,
            blocking_pool_opts: BlockingPoolOpts::default(),
            runtime_opts: RuntimeOpts::default(),
        }
    }

    /// Set whether work stealing is enabled.
    ///
    /// When `true` (the default), a tokio multi-thread runtime is used.
    /// When `false`, a pool of single-threaded tokio runtimes is used instead.
    pub fn work_steal(mut self, enabled: bool) -> Self {
        self.work_steal = enabled;
        self
    }

    /// Set the [`BlockingPoolOpts`] for the runtime's blocking thread pool.
    pub fn blocking_pool_opts(mut self, opts: BlockingPoolOpts) -> Self {
        self.blocking_pool_opts = opts;
        self
    }

    /// Set the [`RuntimeMetricsOpts`] for the runtime.
    pub fn metrics_opts(mut self, opts: RuntimeMetricsOpts) -> Self {
        self.runtime_opts.metrics = opts;
        self
    }

    /// Set the [`RuntimeOpts`] for the runtime.
    pub fn runtime_opts(mut self, opts: RuntimeOpts) -> Self {
        self.runtime_opts = opts;
        self
    }

    /// Set the stack size, in bytes, for every thread this runtime spawns.
    ///
    /// Defaults to [`DEFAULT_THREAD_STACK_SIZE`].
    pub fn thread_stack_size(mut self, bytes: usize) -> Self {
        self.runtime_opts.thread_stack_size = Some(bytes);
        self
    }

    /// Set whether Tokio's experimental alternative timer is enabled.
    ///
    /// This requires building with `--cfg tokio_unstable` and only applies to
    /// work-stealing runtimes.
    pub fn enable_alt_timer(mut self, enabled: bool) -> Self {
        self.runtime_opts.enable_alt_timer = enabled;
        self
    }

    fn build_work_stealing_tokio_builder(&self) -> Builder {
        let stack_size = self.runtime_opts.resolved_thread_stack_size();
        let mut builder = Builder::new_multi_thread();
        builder
            .enable_all()
            .worker_threads(self.threads)
            .thread_name(&self.name)
            .thread_stack_size(stack_size)
            // Records where this thread's stack starts, so the
            // application can measure how much of it a request actually
            // uses. One thread-local store per thread, once.
            .on_thread_start(move || worker_stack::mark_base(stack_size));
        apply_blocking_opts(&mut builder, &self.blocking_pool_opts);
        apply_metrics_opts(&mut builder, &self.runtime_opts.metrics);
        apply_timer_opts(&mut builder, &self.runtime_opts);
        builder
    }

    /// Build the [`Runtime`].
    pub fn build(self) -> Runtime {
        if self.work_steal {
            let mut builder = self.build_work_stealing_tokio_builder();
            #[cfg(feature = "dial9")]
            let dial9_guard = if let Some(dial9_opts) = &self.runtime_opts.dial9 {
                let runtime_name = self.name.clone();
                match build_dial9_runtime(builder, &runtime_name, dial9_opts) {
                    Ok((runtime, guard)) => {
                        return Runtime::Steal {
                            runtime,
                            dial9_guard: Some(guard),
                        };
                    }
                    Err(e) => {
                        log::warn!(
                            "failed to initialize dial9 runtime telemetry for {runtime_name}: {e}"
                        );
                        builder = self.build_work_stealing_tokio_builder();
                        None
                    }
                }
            } else {
                None
            };
            let runtime = builder
                .build()
                .expect("failed to build work-stealing Tokio runtime");
            Runtime::Steal {
                runtime,
                #[cfg(feature = "dial9")]
                dial9_guard,
            }
        } else {
            #[cfg(feature = "dial9")]
            if self.runtime_opts.dial9.is_some() {
                log::warn!("dial9 runtime telemetry is ignored when work stealing is disabled");
            }
            Runtime::NoSteal(NoStealRuntime::new(
                self.threads,
                &self.name,
                self.blocking_pool_opts,
                self.runtime_opts,
            ))
        }
    }
}

impl Runtime {
    /// Create a `Steal` flavor runtime. This just a regular tokio runtime
    pub fn new_steal(threads: usize, name: &str) -> Self {
        RuntimeBuilder::new(threads, name).build()
    }

    /// Create a `NoSteal` flavor runtime. This is backed by multiple tokio current-thread runtime
    pub fn new_no_steal(threads: usize, name: &str) -> Self {
        RuntimeBuilder::new(threads, name).work_steal(false).build()
    }

    /// Return the &[Handle] of the [Runtime].
    /// For `Steal` flavor, it will just return the &[Handle].
    /// For `NoSteal` flavor, it will return the &[Handle] of a random thread in its pool.
    /// So if we want tasks to spawn on all the threads, call this function to get a fresh [Handle]
    /// for each async task.
    pub fn get_handle(&self) -> &Handle {
        match self {
            Self::Steal { runtime, .. } => runtime.handle(),
            Self::NoSteal(r) => r.get_runtime(),
        }
    }

    /// Call tokio's `shutdown_timeout` of all the runtimes. This function is blocking until
    /// all runtimes exit.
    pub fn shutdown_timeout(self, timeout: Duration) {
        match self {
            Self::Steal {
                runtime,
                #[cfg(feature = "dial9")]
                dial9_guard,
            } => {
                #[cfg(feature = "dial9")]
                drop(dial9_guard);
                runtime.shutdown_timeout(timeout);
            }
            Self::NoSteal(r) => r.shutdown_timeout(timeout),
        }
    }
}

// only NoStealRuntime set the pools in thread threads
/// The no-steal pool a thread should spawn onto, per thread.
///
/// Keyed by a thread id the `thread_local` crate hands out and recycles
/// when a thread exits, so a slot can outlive the thread that filled it
/// and reappear under an unrelated one. The owning [`ThreadId`] is
/// stored with the pools and checked on read for exactly that reason:
/// without it, a thread that never joined a no-steal runtime can be
/// handed a handle to one that has already shut down, and every task it
/// spawns is cancelled the moment it is spawned. The slot is a
/// `RefCell` so a thread that inherits a recycled one can claim it.
static CURRENT_HANDLE: Lazy<ThreadLocal<Registration>> = Lazy::new(ThreadLocal::new);

/// Return the [Handle] of current runtime.
/// If the current thread is under a `Steal` runtime, the current [Handle] is returned.
/// If the current thread is under a `NoSteal` runtime, the [Handle] of a random thread
/// under this runtime is returned. This function will panic if called outside any runtime.
pub fn current_handle() -> Handle {
    if let Some(slot) = CURRENT_HANDLE.get() {
        let registered = slot.borrow();
        if let Some((owner, pools)) = registered.as_ref() {
            if *owner == thread::current().id() {
                // The pools are set in init_pools() before the thread
                // that registers them runs anything else.
                let pools = pools.get().unwrap();
                let mut rng = rand::thread_rng();
                let index = rng.gen_range(0..pools.len());
                return pools[index].clone();
            }
        }
    }
    // Not a NoStealRuntime thread, or a slot left behind by one that has
    // exited. Either way the current tokio runtime is the answer.
    Handle::current()
}

/// A thread's no-steal pool registration, with the thread that made it.
type Registration = RefCell<Option<(ThreadId, Pools)>>;

type Control = (Sender<Duration>, JoinHandle<()>);
type Pools = Arc<OnceCell<Box<[Handle]>>>;

/// Multi-threaded runtime backed by a pool of single threaded tokio runtime
pub struct NoStealRuntime {
    threads: usize,
    name: String,
    blocking_opts: BlockingPoolOpts,
    runtime_opts: RuntimeOpts,
    // Lazily init the runtimes so that they are created after pingora
    // daemonize itself. Otherwise the runtime threads are lost.
    pools: Pools,
    controls: OnceCell<Vec<Control>>,
}

impl NoStealRuntime {
    /// Create a new [`NoStealRuntime`] with blocking pool options. Panic if `threads` is 0.
    pub fn new(
        threads: usize,
        name: &str,
        blocking_opts: BlockingPoolOpts,
        runtime_opts: RuntimeOpts,
    ) -> Self {
        assert!(threads != 0);
        NoStealRuntime {
            threads,
            name: name.to_string(),
            blocking_opts,
            runtime_opts,
            pools: Arc::new(OnceCell::new()),
            controls: OnceCell::new(),
        }
    }

    fn init_pools(&self) -> (Box<[Handle]>, Vec<Control>) {
        let mut pools = Vec::with_capacity(self.threads);
        let mut controls = Vec::with_capacity(self.threads);
        let stack_size = self.runtime_opts.resolved_thread_stack_size();
        for _ in 0..self.threads {
            let mut builder = Builder::new_current_thread();
            builder.enable_all();
            // The blocking pool this runtime spawns gets the same stack
            // as the thread below drives the runtime on.
            builder.thread_stack_size(stack_size);
            apply_blocking_opts(&mut builder, &self.blocking_opts);
            apply_metrics_opts(&mut builder, &self.runtime_opts.metrics);
            let rt = builder
                .build()
                .expect("failed to build no-steal Tokio runtime worker");
            let handler = rt.handle().clone();
            let (tx, rx) = channel::<Duration>();
            let pools_ref = self.pools.clone();
            let join = std::thread::Builder::new()
                .name(self.name.clone())
                // A no-steal worker is a plain std thread that blocks on
                // its own current-thread runtime, so the tokio builder
                // above cannot size it: this is the one that matters.
                .stack_size(stack_size)
                .spawn(move || {
                    worker_stack::mark_base(stack_size);
                    // Claim the slot rather than `get_or`, which would
                    // leave a recycled one holding a dead runtime's
                    // pools and hand them to this thread.
                    *CURRENT_HANDLE.get_or_default().borrow_mut() =
                        Some((thread::current().id(), pools_ref));
                    if let Ok(timeout) = rt.block_on(rx) {
                        rt.shutdown_timeout(timeout);
                    } // else Err(_): tx is dropped, just exit
                })
                .unwrap();
            pools.push(handler);
            controls.push((tx, join));
        }

        (pools.into_boxed_slice(), controls)
    }

    /// Return the &[Handle] of a random thread of this runtime
    pub fn get_runtime(&self) -> &Handle {
        let mut rng = rand::thread_rng();

        let index = rng.gen_range(0..self.threads);
        self.get_runtime_at(index)
    }

    /// Return the number of threads of this runtime
    pub fn threads(&self) -> usize {
        self.threads
    }

    fn get_pools(&self) -> &[Handle] {
        if let Some(p) = self.pools.get() {
            p
        } else {
            // TODO: use a mutex to avoid creating a lot threads only to drop them
            let (pools, controls) = self.init_pools();
            // there could be another thread racing with this one to init the pools
            match self.pools.try_insert(pools) {
                Ok(p) => {
                    // unwrap to make sure that this is the one that init both pools and controls
                    self.controls.set(controls).unwrap();
                    p
                }
                // another thread already set it, just return it
                Err((p, _my_pools)) => p,
            }
        }
    }

    /// Return the &[Handle] of a given thread of this runtime
    pub fn get_runtime_at(&self, index: usize) -> &Handle {
        let pools = self.get_pools();
        &pools[index]
    }

    /// Call tokio's `shutdown_timeout` of all the runtimes. This function is blocking until
    /// all runtimes exit.
    pub fn shutdown_timeout(mut self, timeout: Duration) {
        if let Some(controls) = self.controls.take() {
            let (txs, joins): (Vec<Sender<_>>, Vec<JoinHandle<()>>) = controls.into_iter().unzip();
            for tx in txs {
                let _ = tx.send(timeout); // Err() when rx is dropped
            }
            for join in joins {
                let _ = join.join(); // ignore thread error
            }
        } // else, the controls and the runtimes are not even init yet, just return;
    }

    // TODO: runtime metrics
}

#[test]
fn test_steal_runtime() {
    use tokio::time::{sleep, Duration};
    let threads = 2;
    let rt = Runtime::new_steal(threads, "test");
    let handle = rt.get_handle();
    let ret = handle.block_on(async {
        sleep(Duration::from_secs(1)).await;
        let handle = current_handle();
        let join = handle.spawn(async {
            sleep(Duration::from_secs(1)).await;
        });
        join.await.unwrap();
        1
    });

    #[cfg(target_os = "linux")]
    assert_eq!(handle.metrics().num_workers(), threads);
    assert_eq!(ret, 1);
}

#[test]
fn test_no_steal_runtime() {
    use tokio::time::{sleep, Duration};

    let rt = Runtime::new_no_steal(2, "test");
    let handle = rt.get_handle();
    let ret = handle.block_on(async {
        sleep(Duration::from_secs(1)).await;
        let handle = current_handle();
        let join = handle.spawn(async {
            sleep(Duration::from_secs(1)).await;
        });
        join.await.unwrap();
        1
    });

    assert_eq!(ret, 1);
}

#[test]
fn test_no_steal_shutdown() {
    use tokio::time::{sleep, Duration};

    let rt = Runtime::new_no_steal(2, "test");
    let handle = rt.get_handle();
    let ret = handle.block_on(async {
        sleep(Duration::from_secs(1)).await;
        let handle = current_handle();
        let join = handle.spawn(async {
            sleep(Duration::from_secs(1)).await;
        });
        join.await.unwrap();
        1
    });
    assert_eq!(ret, 1);

    rt.shutdown_timeout(Duration::from_secs(1));
}

#[cfg(feature = "dial9")]
#[test]
fn test_dial9_zero_worker_poll_interval_is_rejected() {
    let mut opts = Dial9RuntimeOpts::new("trace");
    opts.worker_poll_interval = Some(Duration::ZERO);
    let err = match build_dial9_runtime(Builder::new_multi_thread(), "test", &opts) {
        Ok(_) => panic!("zero worker poll interval should be rejected"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert_eq!(
        err.to_string(),
        "dial9 worker_poll_interval must be greater than zero"
    );
}

/// How much stack the caller has left to prove it can reach.
///
/// Recurses with a frame large enough to consume `target` bytes quickly,
/// touching every frame it makes, and returns the deepest measurement
/// [`worker_stack::used_here`] reported. A runtime whose threads did not
/// get the configured stack overflows here instead of returning, which
/// is a process abort and the loudest failure this crate can produce.
#[cfg(test)]
fn burn_stack_to(target: usize) -> usize {
    // 64 KiB per frame: ~40 frames to pass 2.5 MiB, which keeps the
    // recursion shallow enough to stay honest about what it measured.
    let mut frame = [0u8; 64 * 1024];
    // Touch it, so the pages are faulted in rather than merely reserved.
    frame[0] = 1;
    frame[frame.len() - 1] = 1;
    std::hint::black_box(&frame);
    let used = worker_stack::used_here(worker_stack::here()).unwrap_or(0);
    if used >= target {
        used
    } else {
        burn_stack_to(target)
    }
}

#[test]
fn a_work_stealing_worker_knows_where_its_stack_starts() {
    let rt = RuntimeBuilder::new(1, "stack-probe")
        .thread_stack_size(4 * 1024 * 1024)
        .build();
    let (tx, rx) = std::sync::mpsc::channel();
    rt.get_handle().spawn(async move {
        tx.send((worker_stack::base(), worker_stack::size())).ok();
    });
    let (base, size) = rx
        .recv_timeout(Duration::from_secs(10))
        .expect("the worker thread runs the probe");
    assert_ne!(base, 0, "a runtime worker has to record its stack base");
    assert_eq!(
        size,
        4 * 1024 * 1024,
        "the worker reports the stack it was configured with"
    );
}

#[test]
fn a_no_steal_worker_knows_where_its_stack_starts() {
    let rt = RuntimeBuilder::new(1, "stack-probe-no-steal")
        .work_steal(false)
        .thread_stack_size(4 * 1024 * 1024)
        .build();
    let (tx, rx) = std::sync::mpsc::channel();
    rt.get_handle().spawn(async move {
        tx.send((worker_stack::base(), worker_stack::size())).ok();
    });
    let (base, size) = rx
        .recv_timeout(Duration::from_secs(10))
        .expect("the worker thread runs the probe");
    assert_ne!(base, 0, "a no-steal worker has to record its stack base");
    assert_eq!(size, 4 * 1024 * 1024);
    // Keep the runtime, and so its worker thread, alive for the rest of
    // the process, the way a server's runtime lives for the life of the
    // server. `init_pools` registers the pool in the process-global
    // `CURRENT_HANDLE`, which is a `thread_local::ThreadLocal` keyed by
    // a thread id the crate recycles when a thread exits. Letting this
    // runtime's thread exit here frees an id that a later thread picks
    // up along with the registration, and `current_handle()` then hands
    // that thread a handle to a runtime that is already shut down.
    std::mem::forget(rt);
}

/// The option is load-bearing, not decorative.
///
/// Tokio's default worker stack is 2 MiB. This asks for 4 MiB and then
/// actually uses more than 2 MiB of it. If `thread_stack_size` were
/// dropped on the floor the recursion would run off the end of the
/// default stack and abort the test binary, so this cannot pass by
/// accident.
#[test]
fn a_worker_gets_the_stack_it_was_configured_with() {
    const TOKIO_DEFAULT_STACK: usize = 2 * 1024 * 1024;
    let rt = RuntimeBuilder::new(1, "stack-burn")
        .thread_stack_size(4 * 1024 * 1024)
        .build();
    let (tx, rx) = std::sync::mpsc::channel();
    rt.get_handle().spawn(async move {
        tx.send(burn_stack_to(TOKIO_DEFAULT_STACK + 256 * 1024))
            .ok();
    });
    let used = rx
        .recv_timeout(Duration::from_secs(30))
        .expect("the worker thread survives past tokio's default stack");
    assert!(
        used > TOKIO_DEFAULT_STACK,
        "the worker only reached {used} bytes, which fits tokio's default stack"
    );
}

#[test]
fn a_zero_stack_size_falls_back_to_the_default() {
    let opts = RuntimeOpts {
        thread_stack_size: Some(0),
        ..RuntimeOpts::default()
    };
    assert_eq!(
        opts.resolved_thread_stack_size(),
        DEFAULT_THREAD_STACK_SIZE,
        "a zero would make tokio panic on build; the default is the better answer"
    );
    assert_eq!(
        RuntimeOpts::default().resolved_thread_stack_size(),
        DEFAULT_THREAD_STACK_SIZE,
        "unset means the default"
    );
}

/// A thread that never joined a no-steal runtime must not inherit one.
///
/// `CURRENT_HANDLE` is keyed by a thread id the `thread_local` crate
/// recycles, so the slot a no-steal worker filled outlives it and turns
/// up under whichever later thread is given the same id. Before the
/// owner check, `current_handle()` on that thread returned a handle to
/// the shut-down runtime and every task spawned through it was cancelled
/// on arrival. This drops a no-steal runtime, waits for its threads to
/// exit so their ids are free, and then asks a series of fresh threads
/// for a handle while a live work-stealing runtime is current.
#[test]
fn a_thread_that_never_joined_a_no_steal_runtime_is_not_handed_a_dead_one() {
    let recycled = Runtime::new_no_steal(2, "recycled");
    // Touch the handle so the pools, and the registrations, are built.
    let _ = recycled.get_handle();
    // Joins the worker threads, which is what frees their ids.
    recycled.shutdown_timeout(Duration::from_secs(5));

    let live = Runtime::new_steal(2, "live");
    for attempt in 0..16 {
        let handle = live.get_handle().clone();
        std::thread::spawn(move || {
            handle.block_on(async {
                current_handle()
                    .spawn(async { 7 })
                    .await
                    .unwrap_or_else(|e| {
                        panic!("attempt {attempt} spawned onto a dead runtime: {e}")
                    })
            })
        })
        .join()
        .expect("the probe thread runs to completion");
    }
}
