//! GPU profiling infrastructure using wgpu timestamp queries.
//!
//! Measures per-pass GPU execution time for drainage, caustic, branched flow,
//! and render passes. Results are displayed in the egui overlay.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

/// Number of timestamp slots (2 per pass: begin + end)
const MAX_TIMESTAMPS: u32 = 16;

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
    /// Timestamp query indices: (begin, end)
    fn indices(self) -> (u32, u32) {
        let base = (self as u32) * 2;
        (base, base + 1)
    }
}

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
pub struct GpuProfiler {
    /// Whether timestamp queries are supported
    enabled: bool,
    /// Query set for timestamps
    query_set: Option<wgpu::QuerySet>,
    /// Buffer to resolve query results into (GPU-side)
    resolve_buffer: Option<wgpu::Buffer>,
    /// Staging buffer for CPU readback
    readback_buffer: Option<wgpu::Buffer>,
    /// Nanoseconds per timestamp tick (from adapter)
    timestamp_period: f32,
    /// Latest resolved timing results
    pub results: GpuTimingResults,
    /// Whether a readback is pending (avoid overlapping map operations)
    readback_pending: bool,
    /// Signal from map_async callback that data is ready
    map_ready: Arc<AtomicBool>,
}

impl GpuProfiler {
    /// Create a new GPU profiler. Pass `timestamp_period` from `adapter.get_info()`.
    /// If `enabled` is false (feature not supported), all operations are no-ops.
    pub fn new(device: &wgpu::Device, enabled: bool, timestamp_period: f32) -> Self {
        if !enabled {
            return Self {
                enabled: false,
                query_set: None,
                resolve_buffer: None,
                readback_buffer: None,
                timestamp_period: 1.0,
                results: GpuTimingResults::default(),
                readback_pending: false,
                map_ready: Arc::new(AtomicBool::new(false)),
            };
        }

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("GPU Timing Query Set"),
            ty: wgpu::QueryType::Timestamp,
            count: MAX_TIMESTAMPS,
        });

        let buffer_size = (MAX_TIMESTAMPS as u64) * 8; // u64 per timestamp
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Timing Resolve Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Timing Readback Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            enabled: true,
            query_set: Some(query_set),
            resolve_buffer: Some(resolve_buffer),
            readback_buffer: Some(readback_buffer),
            timestamp_period,
            results: GpuTimingResults::default(),
            readback_pending: false,
            map_ready: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Get timestamp writes descriptor for a compute pass.
    /// Returns None if profiling is disabled.
    pub fn compute_pass_timestamps(
        &self,
        pass: GpuPass,
    ) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let query_set = self.query_set.as_ref()?;
        let (begin, end) = pass.indices();
        Some(wgpu::ComputePassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        })
    }

    /// Get timestamp writes descriptor for a render pass.
    /// Returns None if profiling is disabled.
    pub fn render_pass_timestamps(
        &self,
        pass: GpuPass,
    ) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let query_set = self.query_set.as_ref()?;
        let (begin, end) = pass.indices();
        Some(wgpu::RenderPassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        })
    }

    /// Resolve timestamps and copy to readback buffer.
    /// Call after all passes are recorded but before encoder.finish().
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        if !self.enabled {
            return;
        }
        let query_set = self.query_set.as_ref().unwrap();
        let resolve_buf = self.resolve_buffer.as_ref().unwrap();
        let readback_buf = self.readback_buffer.as_ref().unwrap();

        encoder.resolve_query_set(query_set, 0..MAX_TIMESTAMPS, resolve_buf, 0);
        encoder.copy_buffer_to_buffer(
            resolve_buf,
            0,
            readback_buf,
            0,
            (MAX_TIMESTAMPS as u64) * 8,
        );
    }

    /// Initiate async readback of timing results.
    /// Call after queue.submit().
    pub fn begin_readback(&mut self) {
        if !self.enabled || self.readback_pending {
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
