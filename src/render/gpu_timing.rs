//! GPU profiling infrastructure using wgpu timestamp queries.
//!
//! Measures per-pass GPU execution time and shows it in the egui overlay.
//! Slots exist for drainage, caustic, branched flow and render passes; only
//! the branched-flow passes are instrumented so far.

use std::cell::Cell;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// Named GPU timing passes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPass {
    Drainage = 0,
    Caustic = 1,
    BranchedFlowClear = 2,
    BranchedFlowTrace = 3,
    Render = 4,
}

impl GpuPass {
    const ALL: [GpuPass; 5] = [
        GpuPass::Drainage,
        GpuPass::Caustic,
        GpuPass::BranchedFlowClear,
        GpuPass::BranchedFlowTrace,
        GpuPass::Render,
    ];

    /// Timestamp query indices: (begin, end). Also the u64 indices of the
    /// pass's two timestamps in the readback buffer.
    fn indices(self) -> (u32, u32) {
        let base = (self as u32) * 2;
        (base, base + 1)
    }

    fn slot(self) -> u64 {
        self as u64
    }

    fn bit(self) -> u32 {
        1 << (self as u32)
    }
}

/// Number of timed passes.
const PASS_COUNT: u64 = GpuPass::ALL.len() as u64;
/// Begin + end timestamps, one u64 each.
const PASS_TIMESTAMP_BYTES: u64 = 16;
/// `resolve_query_set` destinations must be aligned to this, so every pass
/// resolves into its own slot of the resolve buffer.
const RESOLVE_SLOT_BYTES: u64 = wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT;

// Buffer layout invariants required by wgpu, checked at compile time.
const _: () = assert!(RESOLVE_SLOT_BYTES.is_multiple_of(wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT));
const _: () = assert!(PASS_TIMESTAMP_BYTES.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT));
const _: () = assert!(PASS_TIMESTAMP_BYTES <= RESOLVE_SLOT_BYTES);

/// Per-frame GPU timing results in milliseconds
#[derive(Debug, Clone, Default)]
pub struct GpuTimingResults {
    pub drainage_ms: f64,
    pub caustic_ms: f64,
    pub branched_flow_clear_ms: f64,
    pub branched_flow_trace_ms: f64,
    pub render_ms: f64,
}

impl GpuTimingResults {
    pub fn total_ms(&self) -> f64 {
        self.drainage_ms
            + self.caustic_ms
            + self.branched_flow_clear_ms
            + self.branched_flow_trace_ms
            + self.render_ms
    }
}

/// GPU profiler using timestamp queries.
///
/// Requires `wgpu::Features::TIMESTAMP_QUERY` to be enabled on the device.
/// When the feature is unavailable, all methods are no-ops and results are zero.
///
/// Per frame: request timestamp writes for the passes you record, call
/// [`resolve`](Self::resolve) before `encoder.finish()`, submit, then call
/// [`begin_readback`](Self::begin_readback). Call
/// [`poll_results`](Self::poll_results) at the start of each frame. Only the
/// passes that requested timestamps are resolved, and nothing is copied into
/// the readback buffer while a previous readback is still mapping; that
/// frame's timings are dropped instead.
pub struct GpuProfiler {
    /// Whether timestamp queries are supported
    enabled: bool,
    /// Query set for timestamps
    query_set: Option<wgpu::QuerySet>,
    /// Buffer to resolve query results into (GPU-side), one aligned slot per pass
    resolve_buffer: Option<wgpu::Buffer>,
    /// Staging buffer for CPU readback, 16 bytes per pass
    readback_buffer: Option<wgpu::Buffer>,
    /// Nanoseconds per timestamp tick (from adapter)
    timestamp_period: f32,
    /// Latest resolved timing results
    pub results: GpuTimingResults,
    /// Passes that requested timestamp writes this frame (bit per `GpuPass`)
    written_passes: Cell<u32>,
    /// Whether this frame's `resolve` copied timestamps into the readback buffer
    copied_this_frame: bool,
    /// Whether a readback is pending (avoid overlapping map operations)
    readback_pending: bool,
    /// Signal from map_async callback that data is ready
    map_ready: Arc<AtomicBool>,
}

impl GpuProfiler {
    /// Create a new GPU profiler. Pass `timestamp_period` from `queue.get_timestamp_period()`.
    /// If `enabled` is false (feature not supported), all operations are no-ops.
    pub fn new(device: &wgpu::Device, enabled: bool, timestamp_period: f32) -> Self {
        let mut profiler = Self {
            enabled,
            query_set: None,
            resolve_buffer: None,
            readback_buffer: None,
            timestamp_period: if enabled { timestamp_period } else { 1.0 },
            results: GpuTimingResults::default(),
            written_passes: Cell::new(0),
            copied_this_frame: false,
            readback_pending: false,
            map_ready: Arc::new(AtomicBool::new(false)),
        };
        if !enabled {
            return profiler;
        }

        profiler.query_set = Some(device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("GPU Timing Query Set"),
            ty: wgpu::QueryType::Timestamp,
            count: (PASS_COUNT * 2) as u32,
        }));

        profiler.resolve_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Timing Resolve Buffer"),
            size: PASS_COUNT * RESOLVE_SLOT_BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        profiler.readback_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Timing Readback Buffer"),
            size: PASS_COUNT * PASS_TIMESTAMP_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));

        profiler
    }

    /// Get timestamp writes descriptor for a compute pass.
    /// Returns None if profiling is disabled.
    ///
    /// Only request timestamps for a pass that will actually be recorded this
    /// frame: the pass is marked as written, and `resolve` reads its queries.
    pub fn compute_pass_timestamps(
        &self,
        pass: GpuPass,
    ) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let query_set = self.query_set.as_ref()?;
        self.written_passes
            .set(self.written_passes.get() | pass.bit());
        let (begin, end) = pass.indices();
        Some(wgpu::ComputePassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        })
    }

    /// Get timestamp writes descriptor for a render pass.
    /// Returns None if profiling is disabled. Same contract as
    /// [`compute_pass_timestamps`](Self::compute_pass_timestamps).
    pub fn render_pass_timestamps(
        &self,
        pass: GpuPass,
    ) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let query_set = self.query_set.as_ref()?;
        self.written_passes
            .set(self.written_passes.get() | pass.bit());
        let (begin, end) = pass.indices();
        Some(wgpu::RenderPassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        })
    }

    /// Resolve this frame's timestamps and copy them to the readback buffer.
    /// Call after all passes are recorded but before encoder.finish().
    ///
    /// Skips everything while a previous readback is still mapping (copying
    /// into a buffer with a pending map fails queue submission) and when no
    /// pass requested timestamps (resolving never-written queries is invalid
    /// on Vulkan).
    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let written = self.written_passes.replace(0);
        self.copied_this_frame = false;
        if !self.enabled || self.readback_pending || written == 0 {
            return;
        }
        let query_set = self.query_set.as_ref().unwrap();
        let resolve_buf = self.resolve_buffer.as_ref().unwrap();
        let readback_buf = self.readback_buffer.as_ref().unwrap();

        // Passes not timed this frame read back as zero (reported as 0 ms).
        encoder.clear_buffer(readback_buf, 0, None);
        for pass in GpuPass::ALL {
            if written & pass.bit() == 0 {
                continue;
            }
            let (begin, end) = pass.indices();
            let resolve_offset = pass.slot() * RESOLVE_SLOT_BYTES;
            encoder.resolve_query_set(query_set, begin..end + 1, resolve_buf, resolve_offset);
            encoder.copy_buffer_to_buffer(
                resolve_buf,
                resolve_offset,
                readback_buf,
                pass.slot() * PASS_TIMESTAMP_BYTES,
                PASS_TIMESTAMP_BYTES,
            );
        }
        self.copied_this_frame = true;
    }

    /// Initiate async readback of timing results.
    /// Call after queue.submit(). Does nothing unless this frame's `resolve`
    /// copied timestamps.
    pub fn begin_readback(&mut self) {
        if !self.enabled || self.readback_pending || !self.copied_this_frame {
            return;
        }
        let readback_buf = self.readback_buffer.as_ref().unwrap();
        let slice = readback_buf.slice(..);
        let ready = self.map_ready.clone();
        ready.store(false, Ordering::Release);
        slice.map_async(wgpu::MapMode::Read, move |_| {
            ready.store(true, Ordering::Release);
        });
        self.readback_pending = true;
    }

    /// Poll for readback completion and update results.
    /// Call at the start of each frame (before recording new passes).
    pub fn poll_results(&mut self, device: &wgpu::Device) {
        if !self.enabled || !self.readback_pending {
            return;
        }

        // Non-blocking poll to progress async operations
        device.poll(wgpu::Maintain::Poll);

        // Check if map_async callback has fired
        if !self.map_ready.load(Ordering::Acquire) {
            return; // Not ready yet, try next frame
        }

        let readback_buf = self.readback_buffer.as_ref().unwrap();

        // Read the mapped data
        {
            let data = readback_buf.slice(..).get_mapped_range();
            let timestamps: &[u64] = bytemuck::cast_slice(&data);

            let period_ms = self.timestamp_period as f64 / 1_000_000.0;

            let delta = |pass: GpuPass| -> f64 {
                let (begin, end) = pass.indices();
                let t_begin = timestamps.get(begin as usize).copied().unwrap_or(0);
                let t_end = timestamps.get(end as usize).copied().unwrap_or(0);
                if t_end > t_begin {
                    (t_end - t_begin) as f64 * period_ms
                } else {
                    0.0
                }
            };

            self.results = GpuTimingResults {
                drainage_ms: delta(GpuPass::Drainage),
                caustic_ms: delta(GpuPass::Caustic),
                branched_flow_clear_ms: delta(GpuPass::BranchedFlowClear),
                branched_flow_trace_ms: delta(GpuPass::BranchedFlowTrace),
                render_ms: delta(GpuPass::Render),
            };
        }

        readback_buf.unmap();
        self.readback_pending = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_readback_indices_match_query_indices() {
        // Alignment invariants are compile-time asserts next to the constants.
        // Readback u64 indices of a pass coincide with its query indices.
        for pass in GpuPass::ALL {
            let (begin, end) = pass.indices();
            assert_eq!(begin as u64 * 8, pass.slot() * PASS_TIMESTAMP_BYTES);
            assert_eq!(end, begin + 1);
        }
    }

    #[test]
    fn test_pass_bits_are_distinct() {
        let combined = GpuPass::ALL.iter().fold(0u32, |mask, pass| {
            assert_eq!(mask & pass.bit(), 0, "{pass:?} bit overlaps");
            mask | pass.bit()
        });
        assert_eq!(combined.count_ones() as u64, PASS_COUNT);
    }

    /// A device with TIMESTAMP_QUERY, or None when the adapter lacks it.
    fn timestamp_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
        if !adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::TIMESTAMP_QUERY,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .ok()
    }

    /// Record one frame with a timed (empty) compute pass, as pipeline.rs does.
    fn record_timed_frame(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        profiler: &mut GpuProfiler,
        timed: bool,
    ) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        if timed {
            let timestamp_writes = profiler.compute_pass_timestamps(GpuPass::BranchedFlowTrace);
            let pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("timed test pass"),
                timestamp_writes,
            });
            drop(pass);
        }
        profiler.resolve(&mut encoder);
        queue.submit(std::iter::once(encoder.finish()));
        profiler.begin_readback();
    }

    #[test]
    #[ignore] // Requires GPU with TIMESTAMP_QUERY (lavapipe works: scripts/test-local.sh -- --ignored)
    fn test_back_to_back_frames_do_not_touch_a_pending_readback() {
        let Some((device, queue)) = timestamp_device() else {
            return;
        };
        let mut profiler = GpuProfiler::new(&device, true, queue.get_timestamp_period());

        record_timed_frame(&device, &queue, &mut profiler, true);
        assert!(profiler.readback_pending);

        // Second frame before the first readback has been polled: the old code
        // copied into the still-mapping readback buffer and submission failed.
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        record_timed_frame(&device, &queue, &mut profiler, true);
        let error = pollster::block_on(device.pop_error_scope());
        assert!(
            error.is_none(),
            "submit touched a pending readback: {error:?}"
        );

        device.poll(wgpu::Maintain::Wait);
        profiler.poll_results(&device);
        assert!(!profiler.readback_pending);
        assert!(profiler.results.branched_flow_trace_ms >= 0.0);
        assert_eq!(profiler.results.drainage_ms, 0.0);
        assert_eq!(profiler.results.render_ms, 0.0);
    }

    #[test]
    #[ignore] // Requires GPU with TIMESTAMP_QUERY (lavapipe works: scripts/test-local.sh -- --ignored)
    fn test_untimed_frame_resolves_and_maps_nothing() {
        let Some((device, queue)) = timestamp_device() else {
            return;
        };
        let mut profiler = GpuProfiler::new(&device, true, queue.get_timestamp_period());

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        record_timed_frame(&device, &queue, &mut profiler, false);
        let error = pollster::block_on(device.pop_error_scope());
        assert!(error.is_none(), "{error:?}");
        assert!(
            !profiler.readback_pending,
            "nothing was resolved, so nothing should be mapped"
        );
    }
}
