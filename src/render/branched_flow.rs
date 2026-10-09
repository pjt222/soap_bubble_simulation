//! Branched flow simulation using GPU compute
//! Traces light rays through the soap film, bending based on thickness gradients
//!
//! Uses a hybrid model combining:
//! - GRIN optics: Rays bend toward thicker regions (smooth gradients)
//! - Particle scattering: Discrete scatterers (micelles) create local deflections
//!
//! The particle scattering is key for creating tree-like branches (caustics)
//! rather than parallel bands from smooth GRIN alone.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

use crate::physics::geometry::uv_to_unit_sphere;

/// Maximum number of scatterers supported
pub const MAX_SCATTERERS: u32 = 2048;

/// Spatial hash grid dimensions (must match WGSL constants)
const GRID_SIZE_U: u32 = 10;
const GRID_SIZE_V: u32 = 10;
const GRID_CELL_SIZE: f32 = 0.1;
/// Total cells + 1 for prefix-sum array
const CELL_OFFSETS_LEN: usize = (GRID_SIZE_U * GRID_SIZE_V + 1) as usize;

/// GPU-compatible scatterer data for particle-based ray deflection
/// Represents micelle clusters that scatter light within the soap film
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ScattererGPU {
    /// U position in UV space (0-1)
    pub pos_u: f32,
    /// V position in UV space (0-1)
    pub pos_v: f32,
    /// Scattering strength (signed: positive repels, negative attracts)
    pub strength: f32,
    /// Precomputed 1/(2σ²) for efficient Gaussian evaluation
    pub inv_sigma_sq: f32,
}

impl Default for ScattererGPU {
    fn default() -> Self {
        Self {
            pos_u: 0.5,
            pos_v: 0.5,
            strength: 0.0,
            inv_sigma_sq: 1.0,
        }
    }
}

/// Parameters for branched flow simulation
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BranchedFlowParams {
    /// Light entry point on sphere (normalized direction)
    pub entry_point: [f32; 3],
    /// Initial beam direction (tangent to sphere)
    pub beam_dir: [f32; 3],
    /// Number of rays to trace
    pub num_rays: u32,
    /// Steps per ray
    pub ray_steps: u32,
    /// Step size for ray marching
    pub step_size: f32,
    /// How much thickness gradient bends rays
    pub bend_strength: f32,
    /// Initial beam spread angle (radians)
    pub spread_angle: f32,
    /// Intensity falloff per step
    pub intensity_falloff: f32,
    /// Output texture dimensions
    pub tex_width: u32,
    pub tex_height: u32,
    /// Time for animation
    pub time: f32,
    /// Scale factor for thickness values (meters -> micrometers = 1e6)
    pub thickness_scale: f32,
    /// Base film thickness in nanometers (synced from BubbleUniform)
    pub base_thickness_nm: f32,
    /// Amplitude for noise/swirl patterns (synced from BubbleUniform)
    pub swirl_intensity: f32,
    /// Gravity drainage rate (synced from BubbleUniform)
    pub drainage_speed: f32,
    /// Noise coordinate scaling (synced from BubbleUniform)
    pub pattern_scale: f32,
    // === Particle scattering parameters ===
    /// Number of scatterers (micelle clusters) active
    pub num_scatterers: u32,
    /// Base scattering strength (V0 magnitude)
    pub scatterer_strength: f32,
    /// Scatterer radius σ in UV space (correlation length)
    pub scatterer_radius: f32,
    /// Blend factor: 0 = pure GRIN, 1 = pure particle scattering
    pub particle_weight: f32,
    // === Patch view mode parameters ===
    /// Whether patch view mode is enabled (0 = full sphere, 1 = patch only)
    pub patch_enabled: u32,
    /// Patch center U coordinate (0-1)
    pub patch_center_u: f32,
    /// Patch center V coordinate (0-1)
    pub patch_center_v: f32,
    /// Patch half-size in UV space (same for both axes)
    pub patch_half_size: f32,
}

impl Default for BranchedFlowParams {
    fn default() -> Self {
        let mut params = Self {
            // Entry point: front of bubble
            entry_point: [0.0, 0.0, 1.0],
            // Set from DEFAULT_BEAM_ANGLE_DEG at the chart origin below
            beam_dir: [0.0; 3],
            // Rays per frame. With 200 steps this is 8x fewer ray-steps than the earlier
            // 32768 x 400, chosen for the WSL CPU rasteriser (llvmpipe); the frame-time gain was
            // not measured. Ray seeds depend only on ray_idx, so every frame restarts from the
            // same positions (only scatterer drift varies the paths) and the 0.85 fade adds few
            // new samples. Per-frame seeds: #47; ray count per adapter type: #37.
            num_rays: 8192,
            // Steps per ray. The adaptive step factor sits at its 0.3 floor for almost every step
            // with the default scatterer field, so range is about ray_steps * step_size * 0.3 (#47).
            ray_steps: 200,
            // Small steps for smooth ray paths
            step_size: 0.005,
            // Moderate GRIN bending (particle scattering now creates branching)
            bend_strength: 5.0,
            // Beam width (now controls position spread, not angle spread)
            spread_angle: 0.4,
            // Low falloff so rays travel far
            intensity_falloff: 0.001,
            // Higher resolution for smoother branches
            tex_width: 512,
            tex_height: 256,
            time: 0.0,
            thickness_scale: 1e6,
            base_thickness_nm: 500.0,
            swirl_intensity: 1.0,
            drainage_speed: 1.0,
            pattern_scale: 1.0,
            // Particle scattering defaults
            num_scatterers: 400,
            scatterer_strength: 0.5,
            scatterer_radius: 0.03,
            particle_weight: 0.1,
            // Patch view mode defaults (enabled, centred at +z facing the default camera)
            patch_enabled: 1,
            patch_center_u: 0.75,
            patch_center_v: 0.5,
            patch_half_size: 0.158,
        };
        params.beam_dir = beam_direction(params.chart_origin(), DEFAULT_BEAM_ANGLE_DEG).into();
        params
    }
}

impl BranchedFlowParams {
    /// The point whose gnomonic chart the rays move in: the patch centre in patch mode,
    /// the laser entry point otherwise. Mirrors `entry_point` in the `main` kernel of
    /// `branched_flow_compute.wgsl` (#46).
    pub fn chart_origin(&self) -> Vec3 {
        if self.patch_enabled != 0 {
            uv_to_unit_sphere(self.patch_center_u, self.patch_center_v)
        } else {
            Vec3::from(self.entry_point).normalize()
        }
    }
}

/// Default beam angle. At the default laser entry (0, 0, 1) it reproduces the beam
/// direction (-0.5, -0.866, 0) that was hard-coded before #46.
pub const DEFAULT_BEAM_ANGLE_DEG: f32 = 60.0;

/// Tangent basis at a chart origin, built exactly as `branched_flow_compute.wgsl` builds
/// it: `tangent1 = normalize(origin x up)`, `tangent2 = normalize(origin x tangent1)`,
/// with `up = +Y` unless the origin is within about 8 degrees of a pole, then `+X`.
/// Away from the poles `tangent1` points east (increasing u) and `tangent2` south
/// (increasing v).
pub fn chart_tangents(origin: Vec3) -> (Vec3, Vec3) {
    let up = if origin.y.abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let tangent1 = origin.cross(up).normalize();
    let tangent2 = origin.cross(tangent1).normalize();
    (tangent1, tangent2)
}

/// Beam direction at `origin`, `angle_deg` measured from `tangent1` (east) toward
/// `tangent2` (south). Tangent to the sphere and of unit length at every origin, unlike
/// the fixed world vector it replaces, whose tangent projection vanished at two entry
/// points (#46).
pub fn beam_direction(origin: Vec3, angle_deg: f32) -> Vec3 {
    let (tangent1, tangent2) = chart_tangents(origin);
    let angle = angle_deg.to_radians();
    tangent1 * angle.cos() + tangent2 * angle.sin()
}

/// Optional patch bounds for confining scatterers
#[derive(Debug, Clone, Copy)]
pub struct PatchBounds {
    pub center_u: f32,
    pub center_v: f32,
    pub half_size: f32,
}

impl PatchBounds {
    /// Map a 0-1 coordinate to within the patch bounds
    fn map_to_patch(&self, u: f32, v: f32) -> (f32, f32) {
        let min_u = (self.center_u - self.half_size).max(0.0);
        let min_v = (self.center_v - self.half_size).max(0.0);
        let max_u = (self.center_u + self.half_size).min(1.0);
        let max_v = (self.center_v + self.half_size).min(1.0);
        let mapped_u = min_u + u * (max_u - min_u);
        let mapped_v = min_v + v * (max_v - min_v);
        (mapped_u, mapped_v)
    }
}

/// Generate scatterers using quasi-random distribution with jitter
/// Uses Halton sequence for good spatial coverage
/// If patch_bounds is provided, scatterers are confined within the patch
fn generate_scatterers(
    num: u32,
    time: f32,
    base_strength: f32,
    base_radius: f32,
    patch_bounds: Option<PatchBounds>,
) -> Vec<ScattererGPU> {
    let mut scatterers = Vec::with_capacity(num as usize);

    // Halton sequence bases (coprime for 2D low-discrepancy)
    let base_u = 2;
    let base_v = 3;

    for i in 0..num {
        // Halton sequence for quasi-random position
        let u = halton(i + 1, base_u);
        let v = halton(i + 1, base_v);

        // Add time-based jitter for animation
        let jitter_scale = 0.02;
        let hash_seed = (i as f32) * 0.1031 + time * 0.1;
        let jitter_u = (hash_seed.sin() * 43758.547).fract() * jitter_scale;
        let jitter_v = (hash_seed.cos() * 43758.547).fract() * jitter_scale;

        let (pos_u, pos_v) = if let Some(bounds) = patch_bounds {
            // Map to patch bounds, then apply jitter within patch
            let (mapped_u, mapped_v) = bounds.map_to_patch(u, v);
            (
                (mapped_u + jitter_u * bounds.half_size * 2.0).clamp(0.0, 1.0),
                (mapped_v + jitter_v * bounds.half_size * 2.0).clamp(0.0, 1.0),
            )
        } else {
            // Full sphere - use modular wrapping
            (
                (u + jitter_u).rem_euclid(1.0),
                (v + jitter_v).rem_euclid(1.0),
            )
        };

        // Randomize sign of strength (attractive vs repulsive)
        // Use deterministic hash based on index
        let sign_hash = ((i as f32 * 0.7531 + 0.3).sin() * 43758.547).fract();
        let sign = if sign_hash > 0.5 { 1.0 } else { -1.0 };

        // Slight variation in strength magnitude
        let strength_var = 0.8 + 0.4 * ((i as f32 * 0.9371).sin() * 43758.547).fract();
        let strength = sign * base_strength * strength_var;

        // Slight variation in radius
        let radius_var = 0.8 + 0.4 * ((i as f32 * 0.5791).cos() * 43758.547).fract();
        let sigma = (base_radius * radius_var).max(1e-6);
        let inv_sigma_sq = 1.0 / (2.0 * sigma * sigma);

        scatterers.push(ScattererGPU {
            pos_u,
            pos_v,
            strength,
            inv_sigma_sq,
        });
    }

    scatterers
}

/// Halton sequence for quasi-random number generation
/// Returns value in [0, 1) for given index and base
fn halton(index: u32, base: u32) -> f32 {
    let mut result = 0.0f32;
    let mut f = 1.0 / base as f32;
    let mut i = index;

    while i > 0 {
        result += f * (i % base) as f32;
        i /= base;
        f /= base as f32;
    }

    result
}

/// Manages GPU-based branched flow simulation
pub struct BranchedFlowSimulator {
    /// Compute pipeline for ray tracing
    trace_pipeline: wgpu::ComputePipeline,
    /// Compute pipeline for clearing/fading
    clear_pipeline: wgpu::ComputePipeline,
    /// Bind group layout (stored for rebinding when thickness buffer swaps)
    bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group
    bind_group: wgpu::BindGroup,
    /// Parameters buffer
    params_buffer: wgpu::Buffer,
    /// Scatterer buffer (storage buffer for particle positions/strengths, sorted by grid cell)
    scatterer_buffer: wgpu::Buffer,
    /// Cell offsets prefix-sum buffer for spatial hash (GRID_SIZE_U * GRID_SIZE_V + 1 entries)
    cell_offsets_buffer: wgpu::Buffer,
    /// Current parameters
    pub params: BranchedFlowParams,
    /// Whether simulation is enabled
    pub enabled: bool,
    /// Texture dimensions
    tex_width: u32,
    tex_height: u32,
    /// Dirty flag: whether scatterers need full regeneration
    /// Set when structural parameters change (count, radius, patch), cleared after upload
    scatterers_dirty: bool,
    /// Cached scatterer parameters for dirty check
    last_num_scatterers: u32,
    last_scatterer_strength: f32,
    last_scatterer_radius: f32,
    last_patch_enabled: u32,
    last_patch_center_u: f32,
    last_patch_center_v: f32,
    last_patch_half_size: f32,
    /// Stored scatterers for temporal coherence (Brownian perturbation between regenerations)
    current_scatterers: Vec<ScattererGPU>,
    /// Beam angle in degrees (see [`beam_direction`]); `params.beam_dir` is derived from it
    /// at the current chart origin before every upload
    beam_angle_deg: f32,
}

/// Create a branched flow texture buffer (called early in pipeline init)
pub fn create_branched_flow_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    // Higher resolution for smoother branches
    let tex_width = 512u32;
    let tex_height = 256u32;
    let tex_size = (tex_width * tex_height) as usize;
    let caustic_data = vec![0u32; tex_size];

    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Branched Flow Texture Buffer"),
        contents: bytemuck::cast_slice(&caustic_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

impl BranchedFlowSimulator {
    /// Create a new branched flow simulator
    /// Takes external buffers: thickness_buffer from GPU drainage, caustic_buffer from early init
    // put id:'gpu_compute_branched_init', label:'Init branched flow', input:'gpu_device.internal', output:'branched_flow_texture_gpu.internal'
    pub fn new(
        device: &wgpu::Device,
        thickness_buffer: &wgpu::Buffer,
        caustic_buffer: &wgpu::Buffer,
    ) -> Self {
        let params = BranchedFlowParams::default();
        let tex_width = params.tex_width;
        let tex_height = params.tex_height;

        // Verify dimensions match the hardcoded constants in bubble.wgsl
        // (BRANCHED_TEX_WIDTH = 512, BRANCHED_TEX_HEIGHT = 256)
        assert_eq!(
            tex_width, 512,
            "BranchedFlowParams tex_width must match BRANCHED_TEX_WIDTH in bubble.wgsl"
        );
        assert_eq!(
            tex_height, 256,
            "BranchedFlowParams tex_height must match BRANCHED_TEX_HEIGHT in bubble.wgsl"
        );

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Branched Flow Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/branched_flow_compute.wgsl").into(),
            ),
        });

        // Create bind group layout (stored for rebinding when thickness buffer swaps)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Branched Flow Bind Group Layout"),
            entries: &[
                // Params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Thickness field (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Caustic texture (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Scatterers array (read-only, sorted by grid cell)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Cell offsets prefix-sum array for spatial hash (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Branched Flow Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create trace compute pipeline
        let trace_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Branched Flow Trace Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create clear compute pipeline
        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Branched Flow Clear Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("clear"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create params buffer
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Branched Flow Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create scatterer buffer (max 2048 scatterers × 16 bytes = 32KB)
        // Initialize with default scatterers
        let initial_scatterers = generate_scatterers(
            params.num_scatterers,
            0.0,
            params.scatterer_strength,
            params.scatterer_radius,
            None, // No patch bounds on initial creation
        );
        // Pad to MAX_SCATTERERS to avoid buffer resizing
        let mut scatterer_data = initial_scatterers;
        scatterer_data.resize(MAX_SCATTERERS as usize, ScattererGPU::default());

        let scatterer_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Branched Flow Scatterer Buffer"),
            contents: bytemuck::cast_slice(&scatterer_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Create cell offsets prefix-sum buffer for spatial hash
        let initial_offsets = vec![0u32; CELL_OFFSETS_LEN];
        let cell_offsets_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Branched Flow Cell Offsets Buffer"),
            contents: bytemuck::cast_slice(&initial_offsets),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group using the external caustic buffer
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Branched Flow Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: thickness_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: caustic_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scatterer_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cell_offsets_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            trace_pipeline,
            clear_pipeline,
            bind_group_layout,
            bind_group,
            params_buffer,
            scatterer_buffer,
            cell_offsets_buffer,
            params,
            enabled: false,
            tex_width,
            tex_height,
            // Initialize dirty flag to true so first frame uploads scatterers
            scatterers_dirty: true,
            last_num_scatterers: params.num_scatterers,
            last_scatterer_strength: params.scatterer_strength,
            last_scatterer_radius: params.scatterer_radius,
            last_patch_enabled: params.patch_enabled,
            last_patch_center_u: params.patch_center_u,
            last_patch_center_v: params.patch_center_v,
            last_patch_half_size: params.patch_half_size,
            current_scatterers: Vec::new(),
            beam_angle_deg: DEFAULT_BEAM_ANGLE_DEG,
        }
    }

    /// Update parameters buffer. Recomputes `beam_dir` first, because the chart origin it
    /// is tangent to moves with the laser entry and the patch.
    pub fn update_params(&mut self, queue: &wgpu::Queue) {
        self.update_beam_direction();
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
    }

    fn update_beam_direction(&mut self) {
        self.params.beam_dir =
            beam_direction(self.params.chart_origin(), self.beam_angle_deg).into();
    }

    /// Rebuild bind group with the current thickness buffer.
    /// Must be called each frame when GPU drainage is active, because the drainage
    /// simulator double-buffers and the "current" buffer alternates after each step.
    pub fn rebuild_bind_group(
        &mut self,
        device: &wgpu::Device,
        thickness_buffer: &wgpu::Buffer,
        caustic_buffer: &wgpu::Buffer,
    ) {
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Branched Flow Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: thickness_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: caustic_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.scatterer_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.cell_offsets_buffer.as_entire_binding(),
                },
            ],
        });
    }

    /// Check if scatterer parameters have changed and mark dirty if so
    pub fn check_scatterer_params_changed(&mut self) {
        let changed = self.params.num_scatterers != self.last_num_scatterers
            || (self.params.scatterer_strength - self.last_scatterer_strength).abs() > 1e-6
            || (self.params.scatterer_radius - self.last_scatterer_radius).abs() > 1e-6
            || self.params.patch_enabled != self.last_patch_enabled
            || (self.params.patch_center_u - self.last_patch_center_u).abs() > 1e-6
            || (self.params.patch_center_v - self.last_patch_center_v).abs() > 1e-6
            || (self.params.patch_half_size - self.last_patch_half_size).abs() > 1e-6;

        if changed {
            self.scatterers_dirty = true;
            self.last_num_scatterers = self.params.num_scatterers;
            self.last_scatterer_strength = self.params.scatterer_strength;
            self.last_scatterer_radius = self.params.scatterer_radius;
            self.last_patch_enabled = self.params.patch_enabled;
            self.last_patch_center_u = self.params.patch_center_u;
            self.last_patch_center_v = self.params.patch_center_v;
            self.last_patch_half_size = self.params.patch_half_size;
        }
    }

    /// Update scatterer positions with temporal coherence.
    ///
    /// When structural parameters change (count, radius, patch bounds), scatterers are
    /// fully regenerated. Otherwise, small Brownian perturbations are applied each frame
    /// so the branching pattern evolves smoothly like real micelle clusters drifting in
    /// the film, rather than jumping discontinuously.
    ///
    /// Sorts scatterers by grid cell and builds prefix-sum cell_offsets for O(k) GPU lookup.
    pub fn update_scatterers(&mut self, queue: &wgpu::Queue, time: f32) {
        // Check if parameters changed since last upload
        self.check_scatterer_params_changed();

        if self.scatterers_dirty {
            // Full regeneration: structural parameters changed
            let patch_bounds = if self.params.patch_enabled != 0 {
                Some(PatchBounds {
                    center_u: self.params.patch_center_u,
                    center_v: self.params.patch_center_v,
                    half_size: self.params.patch_half_size,
                })
            } else {
                None
            };

            self.current_scatterers = generate_scatterers(
                self.params.num_scatterers.min(MAX_SCATTERERS),
                time,
                self.params.scatterer_strength,
                self.params.scatterer_radius,
                patch_bounds,
            );
            self.scatterers_dirty = false;
        } else if !self.current_scatterers.is_empty() {
            // Brownian perturbation: smooth temporal evolution
            // Each scatterer drifts ~0.001 UV units per frame (~3% of σ per frame).
            // Over ~30 frames the pattern shifts noticeably but continuously.
            let perturbation_scale = 0.001f32;

            let (min_u, max_u, min_v, max_v) = if self.params.patch_enabled != 0 {
                let hs = self.params.patch_half_size;
                (
                    (self.params.patch_center_u - hs).max(0.0),
                    (self.params.patch_center_u + hs).min(1.0),
                    (self.params.patch_center_v - hs).max(0.0),
                    (self.params.patch_center_v + hs).min(1.0),
                )
            } else {
                (0.0, 1.0, 0.0, 1.0)
            };

            for (i, s) in self.current_scatterers.iter_mut().enumerate() {
                // Pseudo-random perturbation using time × frequency mixing
                // Different frequencies per scatterer prevent correlated drift
                let seed_u = (i as f32 * 0.7531 + time * 31.37).sin() * 43758.547;
                let seed_v = (i as f32 * 0.9371 + time * 17.53).cos() * 43758.547;
                s.pos_u =
                    (s.pos_u + (seed_u.fract() - 0.5) * perturbation_scale).clamp(min_u, max_u);
                s.pos_v =
                    (s.pos_v + (seed_v.fract() - 0.5) * perturbation_scale).clamp(min_v, max_v);
            }
        } else {
            return; // No scatterers to update
        }

        // Sort scatterers by grid cell for true spatial hash
        self.current_scatterers.sort_by_key(|s| {
            let u_cell = (s.pos_u / GRID_CELL_SIZE).clamp(0.0, (GRID_SIZE_U - 1) as f32) as u32;
            let v_cell = (s.pos_v / GRID_CELL_SIZE).clamp(0.0, (GRID_SIZE_V - 1) as f32) as u32;
            v_cell * GRID_SIZE_U + u_cell
        });

        // Build prefix-sum cell_offsets: offsets[i] = start index of cell i in sorted array
        let total_cells = (GRID_SIZE_U * GRID_SIZE_V) as usize;
        let mut cell_offsets = vec![0u32; total_cells + 1];
        for s in &self.current_scatterers {
            let u_cell = (s.pos_u / GRID_CELL_SIZE).clamp(0.0, (GRID_SIZE_U - 1) as f32) as u32;
            let v_cell = (s.pos_v / GRID_CELL_SIZE).clamp(0.0, (GRID_SIZE_V - 1) as f32) as u32;
            let cell_idx = (v_cell * GRID_SIZE_U + u_cell) as usize;
            cell_offsets[cell_idx + 1] += 1;
        }
        for i in 1..=total_cells {
            cell_offsets[i] += cell_offsets[i - 1];
        }

        // Upload sorted scatterers and cell offsets
        queue.write_buffer(
            &self.scatterer_buffer,
            0,
            bytemuck::cast_slice(&self.current_scatterers),
        );
        queue.write_buffer(
            &self.cell_offsets_buffer,
            0,
            bytemuck::cast_slice(&cell_offsets),
        );
    }

    /// Force scatterer regeneration on next update (e.g., for animation)
    pub fn mark_scatterers_dirty(&mut self) {
        self.scatterers_dirty = true;
    }

    /// Set laser entry point (spherical coordinates: azimuth, elevation in degrees)
    pub fn set_entry_point(&mut self, azimuth_deg: f32, elevation_deg: f32) {
        let azimuth = azimuth_deg.to_radians();
        let elevation = elevation_deg.to_radians();
        self.params.entry_point = [
            elevation.cos() * azimuth.cos(),
            elevation.sin(),
            elevation.cos() * azimuth.sin(),
        ];
    }

    /// Set the beam angle in degrees, measured from east toward south at the chart origin
    /// (see [`beam_direction`])
    pub fn set_beam_angle(&mut self, angle_deg: f32) {
        self.beam_angle_deg = angle_deg;
        self.update_beam_direction();
    }

    /// Current beam angle in degrees
    pub fn beam_angle_deg(&self) -> f32 {
        self.beam_angle_deg
    }

    /// Run simulation step with optional GPU timestamp profiling.
    // put id:'gpu_compute_branched_step', label:'Dispatch branched flow rays', input:'uniform_buffers_gpu.internal', output:'branched_flow_texture_gpu.internal'
    pub fn step(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        time: f32,
        clear_timestamps: Option<wgpu::ComputePassTimestampWrites<'_>>,
        trace_timestamps: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        if !self.enabled {
            return;
        }

        self.params.time = time;

        // Clear/fade pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Branched Flow Clear Pass"),
                timestamp_writes: clear_timestamps,
            });
            pass.set_pipeline(&self.clear_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let workgroups_x = self.tex_width.div_ceil(16);
            let workgroups_y = self.tex_height.div_ceil(16);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Ray trace pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Branched Flow Trace Pass"),
                timestamp_writes: trace_timestamps,
            });
            pass.set_pipeline(&self.trace_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let workgroups = self.params.num_rays.div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }

    /// Get texture dimensions
    pub fn texture_size(&self) -> (u32, u32) {
        (self.tex_width, self.tex_height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_struct_size_matches_wgsl_layout() {
        // BranchedFlowParams must be 112 bytes (28 f32/u32 fields)
        // to match the WGSL uniform struct layout
        assert_eq!(
            std::mem::size_of::<BranchedFlowParams>(),
            28 * 4, // 28 fields * 4 bytes each
            "BranchedFlowParams size must match WGSL uniform layout (112 bytes)"
        );
    }

    #[test]
    fn params_field_offsets_match_wgsl() {
        // Verify critical field offsets match the WGSL struct declaration.
        // BranchedFlowParams uses [f32; 3] arrays for entry_point and beam_dir
        // which map to 6 scalar fields in WGSL — offsets must stay in sync.
        use std::mem::offset_of;

        assert_eq!(offset_of!(BranchedFlowParams, entry_point), 0);
        assert_eq!(offset_of!(BranchedFlowParams, beam_dir), 12);
        assert_eq!(offset_of!(BranchedFlowParams, num_rays), 24);
        assert_eq!(offset_of!(BranchedFlowParams, ray_steps), 28);
        assert_eq!(offset_of!(BranchedFlowParams, step_size), 32);
        assert_eq!(offset_of!(BranchedFlowParams, bend_strength), 36);
        assert_eq!(offset_of!(BranchedFlowParams, tex_width), 48);
        assert_eq!(offset_of!(BranchedFlowParams, tex_height), 52);
        assert_eq!(offset_of!(BranchedFlowParams, time), 56);
        assert_eq!(offset_of!(BranchedFlowParams, thickness_scale), 60);
        assert_eq!(offset_of!(BranchedFlowParams, num_scatterers), 80);
        assert_eq!(offset_of!(BranchedFlowParams, patch_enabled), 96);
    }

    #[test]
    fn set_entry_point_produces_normalized_vectors() {
        // Test various angles
        let test_angles: [(f32, f32); 4] =
            [(0.0, 0.0), (45.0, 30.0), (180.0, -45.0), (270.0, 89.0)];
        for (azimuth, elevation) in test_angles {
            let az_rad = azimuth.to_radians();
            let el_rad = elevation.to_radians();
            let entry = [
                el_rad.cos() * az_rad.cos(),
                el_rad.sin(),
                el_rad.cos() * az_rad.sin(),
            ];
            let length = (entry[0] * entry[0] + entry[1] * entry[1] + entry[2] * entry[2]).sqrt();
            assert!(
                (length - 1.0).abs() < 1e-5,
                "Entry point not normalized for azimuth={azimuth}, elevation={elevation}: length={length}"
            );
        }
    }

    /// Laser entry point for azimuth/elevation in degrees, as set_entry_point computes it
    fn entry_from_angles(azimuth_deg: f32, elevation_deg: f32) -> Vec3 {
        let mut params = BranchedFlowParams::default();
        let (azimuth, elevation) = (azimuth_deg.to_radians(), elevation_deg.to_radians());
        params.entry_point = [
            elevation.cos() * azimuth.cos(),
            elevation.sin(),
            elevation.cos() * azimuth.sin(),
        ];
        params.patch_enabled = 0;
        params.chart_origin()
    }

    #[test]
    fn beam_direction_is_a_unit_tangent_at_every_origin() {
        let origins = [
            // The two entries where the former fixed beam (-0.5, -0.866, 0) had no
            // tangent component (#46)
            entry_from_angles(0.0, 60.0),
            entry_from_angles(180.0, -60.0),
            entry_from_angles(90.0, 0.0),
            entry_from_angles(-135.0, 89.0),
            Vec3::Y,
            Vec3::NEG_Y,
            uv_to_unit_sphere(0.5, 0.5),
            uv_to_unit_sphere(0.1, 0.2),
        ];
        for origin in origins {
            for angle_deg in [0.0f32, 45.0, 60.0, 90.0, 180.0, 270.0] {
                let beam = beam_direction(origin, angle_deg);
                assert!(
                    beam.dot(origin).abs() < 1e-5 && (beam.length() - 1.0).abs() < 1e-5,
                    "beam {beam:?} at origin {origin:?}, angle {angle_deg}"
                );
            }
        }
    }

    #[test]
    fn default_beam_angle_reproduces_the_former_beam_at_the_default_entry() {
        let beam = beam_direction(Vec3::Z, DEFAULT_BEAM_ANGLE_DEG);
        assert!(
            (beam - Vec3::new(-0.5, -0.866_025_4, 0.0)).length() < 1e-5,
            "{beam:?}"
        );
    }

    #[test]
    fn chart_tangents_point_east_and_south() {
        use crate::physics::geometry::unit_sphere_to_uv;
        for (u, v) in [(0.5, 0.5), (0.3, 0.3), (0.8, 0.7)] {
            let origin = uv_to_unit_sphere(u, v);
            let (tangent1, tangent2) = chart_tangents(origin);
            let [u_east, v_east] = unit_sphere_to_uv((origin + tangent1 * 1e-3).normalize());
            let [u_south, v_south] = unit_sphere_to_uv((origin + tangent2 * 1e-3).normalize());
            assert!(
                u_east > u && (v_east - v).abs() < 1e-4,
                "tangent1 not east at ({u}, {v})"
            );
            assert!(
                v_south > v && (u_south - u).abs() < 1e-4,
                "tangent2 not south at ({u}, {v})"
            );
        }
    }

    #[test]
    fn chart_origin_is_the_patch_centre_in_patch_mode() {
        let mut params = BranchedFlowParams {
            patch_center_u: 0.3,
            patch_center_v: 0.6,
            ..BranchedFlowParams::default()
        };
        assert!((params.chart_origin() - uv_to_unit_sphere(0.3, 0.6)).length() < 1e-6);
        params.patch_enabled = 0;
        assert!((params.chart_origin() - Vec3::from(params.entry_point)).length() < 1e-6);
    }

    #[test]
    fn default_beam_is_tangent_at_the_default_chart_origin() {
        let params = BranchedFlowParams::default();
        let beam = Vec3::from(params.beam_dir);
        assert!(beam.dot(params.chart_origin()).abs() < 1e-6);
        assert!((beam.length() - 1.0).abs() < 1e-6);
    }

    fn test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()
    }

    /// Share of lit texels in the deposit texture, overall and per half
    #[derive(Debug)]
    struct DepositCoverage {
        lit: f64,
        left_half: f64,
        right_half: f64,
        top_half: f64,
        bottom_half: f64,
    }

    impl DepositCoverage {
        fn of(texels: &[u32], width: usize, height: usize) -> Self {
            let lit_share = |columns: std::ops::Range<usize>, rows: std::ops::Range<usize>| {
                let total = columns.len() * rows.len();
                let lit = rows
                    .flat_map(|row| columns.clone().map(move |column| row * width + column))
                    .filter(|&index| texels[index] > 0)
                    .count();
                lit as f64 / total as f64
            };
            Self {
                lit: lit_share(0..width, 0..height),
                left_half: lit_share(0..width / 2, 0..height),
                right_half: lit_share(width / 2..width, 0..height),
                top_half: lit_share(0..width, 0..height / 2),
                bottom_half: lit_share(0..width, height / 2..height),
            }
        }
    }

    /// Deposit texture after one compute frame of a simulator set up by `configure`.
    /// The drainage thickness is uniform, so only the scatterers bend the rays.
    fn one_frame_deposits(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        configure: impl FnOnce(&mut BranchedFlowSimulator),
    ) -> Vec<u32> {
        let thickness = vec![500e-9f32; 256 * 128];
        let thickness_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&thickness),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let texture_bytes = 512 * 256 * 4;
        let deposit_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: texture_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: texture_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut simulator = BranchedFlowSimulator::new(device, &thickness_buffer, &deposit_buffer);
        simulator.enabled = true;
        configure(&mut simulator);
        simulator.update_params(queue);
        simulator.update_scatterers(queue, 0.0);
        let mut encoder = device.create_command_encoder(&Default::default());
        simulator.step(&mut encoder, 0.0, None, None);
        encoder.copy_buffer_to_buffer(&deposit_buffer, 0, &readback_buffer, 0, texture_bytes);
        queue.submit(Some(encoder.finish()));
        if let Some(error) = pollster::block_on(device.pop_error_scope()) {
            panic!("branched flow frame failed validation: {error}");
        }

        readback_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |result| result.expect("map deposits"));
        device.poll(wgpu::Maintain::Wait);
        bytemuck::cast_slice::<u8, u32>(&readback_buffer.slice(..).get_mapped_range()).to_vec()
    }

    #[test]
    #[ignore] // Requires GPU (lavapipe works: scripts/test-local.sh -- --ignored)
    fn patch_mode_rays_cover_the_patch() {
        // In patch mode the whole deposit texture maps onto the visible patch. The patch sits
        // at u = 0.5 (the pre-#46 default), 90 degrees from the default laser entry: at the
        // current default (u = 0.75) the patch centre IS the laser entry, so a chart at the
        // wrong origin would go unnoticed. Measured on lavapipe, one frame: before #46 4.8%
        // of texels lit and none in the left half; chart centred on the patch but the old
        // beam-line spawn, 14%; rays starting over the whole patch, 90% (halves 87-93%).
        let Some((device, queue)) = test_device() else {
            panic!("no GPU adapter (run via scripts/test-local.sh for lavapipe)");
        };
        let texels = one_frame_deposits(&device, &queue, |simulator| {
            simulator.params.patch_center_u = 0.5;
        });
        let coverage = DepositCoverage::of(&texels, 512, 256);
        println!("patch at u = 0.5, one frame: {coverage:?}");
        assert!(coverage.lit > 0.6, "{coverage:?}");
        for half in [
            coverage.left_half,
            coverage.right_half,
            coverage.top_half,
            coverage.bottom_half,
        ] {
            assert!(
                half > 0.5,
                "a half of the patch is mostly dark: {coverage:?}"
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU (lavapipe works: scripts/test-local.sh -- --ignored)
    fn full_sphere_mode_deposits_around_the_laser_entry() {
        let Some((device, queue)) = test_device() else {
            panic!("no GPU adapter (run via scripts/test-local.sh for lavapipe)");
        };
        // Default laser entry (0, 0, 1) is at u = 0.75, v = 0.5
        // The patch centre (u = 0.5) differs from the laser entry, so this fails if the
        // full-sphere chart is centred on the patch instead
        let texels = one_frame_deposits(&device, &queue, |simulator| {
            simulator.params.patch_enabled = 0;
            simulator.params.patch_center_u = 0.5;
        });
        let (mut weight, mut weighted_u, mut weighted_v) = (0.0f64, 0.0f64, 0.0f64);
        for (index, &texel) in texels.iter().enumerate() {
            let (column, row) = (index % 512, index / 512);
            weight += texel as f64;
            weighted_u += texel as f64 * column as f64 / 511.0;
            weighted_v += texel as f64 * row as f64 / 255.0;
        }
        assert!(weight > 0.0, "no deposits in full-sphere mode");
        let (centroid_u, centroid_v) = (weighted_u / weight, weighted_v / weight);
        println!("full-sphere deposit centroid: ({centroid_u:.3}, {centroid_v:.3})");
        assert!(
            (centroid_u - 0.75).abs() < 0.05 && (centroid_v - 0.5).abs() < 0.1,
            "deposits centred at ({centroid_u}, {centroid_v}), not near the entry (0.75, 0.5)"
        );
    }

    #[test]
    fn default_params_are_physically_reasonable() {
        let params = BranchedFlowParams::default();

        assert!(
            params.num_rays >= 1024,
            "Need enough rays for visible pattern"
        );
        assert!(
            params.ray_steps >= 100,
            "Need enough steps for ray propagation"
        );
        assert!(
            params.step_size > 0.0 && params.step_size < 0.1,
            "Step size should be small"
        );
        assert!(params.bend_strength > 0.0, "Bend strength must be positive");
        assert!(
            params.spread_angle > 0.0 && params.spread_angle < std::f32::consts::PI,
            "Spread angle should be a reasonable radian value"
        );
        assert!(
            params.intensity_falloff > 0.0 && params.intensity_falloff < 0.1,
            "Intensity falloff should be small per step"
        );
        assert!(
            params.thickness_scale > 0.0,
            "Thickness scale must be positive"
        );
        assert_eq!(params.tex_width, 512);
        assert_eq!(params.tex_height, 256);
        // Particle scattering defaults
        assert!(
            params.num_scatterers >= 100 && params.num_scatterers <= MAX_SCATTERERS,
            "Num scatterers should be reasonable"
        );
        assert!(
            params.scatterer_strength > 0.0,
            "Scatterer strength must be positive"
        );
        assert!(
            params.scatterer_radius > 0.0 && params.scatterer_radius < 1.0,
            "Scatterer radius should be in UV space (0-1)"
        );
        assert!(
            params.particle_weight >= 0.0 && params.particle_weight <= 1.0,
            "Particle weight should be a blend factor (0-1)"
        );
    }

    #[test]
    fn scatterer_gpu_struct_size() {
        // ScattererGPU must be 16 bytes (4 f32 fields) for GPU alignment
        assert_eq!(
            std::mem::size_of::<ScattererGPU>(),
            16,
            "ScattererGPU must be 16 bytes for GPU alignment"
        );
    }

    #[test]
    fn halton_sequence_produces_values_in_range() {
        for i in 1..100u32 {
            let u = halton(i, 2);
            let v = halton(i, 3);
            assert!((0.0..1.0).contains(&u), "Halton base 2 out of range: {u}");
            assert!((0.0..1.0).contains(&v), "Halton base 3 out of range: {v}");
        }
    }

    #[test]
    fn halton_sequence_is_quasi_random() {
        // Test that halton produces low-discrepancy sequence
        // Points should be spread out, not clustered
        let n = 100;
        let mut points: Vec<(f32, f32)> = Vec::new();
        for i in 1..=n {
            points.push((halton(i as u32, 2), halton(i as u32, 3)));
        }

        // Check that no two points are too close (min spacing)
        let min_dist_sq = 0.0001f32; // Allow some clustering for quasi-random
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let dx = points[i].0 - points[j].0;
                let dy = points[i].1 - points[j].1;
                let dist_sq = dx * dx + dy * dy;
                // At least some pairs should be well separated
                if dist_sq > min_dist_sq {
                    return; // Test passes if we find separated points
                }
            }
        }
        // If we get here, all points are too close (unlikely for Halton)
        panic!("Halton sequence points are too clustered");
    }

    #[test]
    fn generate_scatterers_produces_correct_count() {
        let scatterers = generate_scatterers(100, 0.0, 0.5, 0.03, None);
        assert_eq!(scatterers.len(), 100);

        let scatterers = generate_scatterers(500, 0.0, 0.5, 0.03, None);
        assert_eq!(scatterers.len(), 500);
    }

    #[test]
    fn generate_scatterers_positions_in_range() {
        let scatterers = generate_scatterers(200, 0.0, 0.5, 0.03, None);
        for s in &scatterers {
            assert!(
                s.pos_u >= 0.0 && s.pos_u <= 1.0,
                "Scatterer pos_u out of range: {}",
                s.pos_u
            );
            assert!(
                s.pos_v >= 0.0 && s.pos_v <= 1.0,
                "Scatterer pos_v out of range: {}",
                s.pos_v
            );
        }
    }

    #[test]
    fn generate_scatterers_has_mixed_signs() {
        let scatterers = generate_scatterers(100, 0.0, 0.5, 0.03, None);
        let positive = scatterers.iter().filter(|s| s.strength > 0.0).count();
        let negative = scatterers.iter().filter(|s| s.strength < 0.0).count();

        // Should have a mix of attractive and repulsive scatterers
        assert!(positive > 20, "Too few attractive scatterers: {positive}");
        assert!(negative > 20, "Too few repulsive scatterers: {negative}");
    }

    #[test]
    fn generate_scatterers_inv_sigma_sq_is_positive() {
        let scatterers = generate_scatterers(100, 0.0, 0.5, 0.03, None);
        for s in &scatterers {
            assert!(
                s.inv_sigma_sq > 0.0,
                "inv_sigma_sq must be positive: {}",
                s.inv_sigma_sq
            );
        }
    }

    #[test]
    fn generate_scatterers_time_affects_positions() {
        let scatterers_t0 = generate_scatterers(50, 0.0, 0.5, 0.03, None);
        let scatterers_t1 = generate_scatterers(50, 1.0, 0.5, 0.03, None);

        // At least some positions should differ due to time-based jitter
        let mut different = 0;
        for i in 0..50 {
            if (scatterers_t0[i].pos_u - scatterers_t1[i].pos_u).abs() > 0.001
                || (scatterers_t0[i].pos_v - scatterers_t1[i].pos_v).abs() > 0.001
            {
                different += 1;
            }
        }
        assert!(different > 0, "Time should affect scatterer positions");
    }

    #[test]
    fn generate_scatterers_with_patch_bounds() {
        let bounds = PatchBounds {
            center_u: 0.5,
            center_v: 0.5,
            half_size: 0.2,
        };
        let scatterers = generate_scatterers(100, 0.0, 0.5, 0.03, Some(bounds));

        // All scatterers should be within patch bounds (with small margin for jitter)
        let min_u = 0.3 - 0.1; // Allow some margin for jitter
        let max_u = 0.7 + 0.1;
        let min_v = 0.3 - 0.1;
        let max_v = 0.7 + 0.1;

        for s in &scatterers {
            assert!(
                s.pos_u >= min_u && s.pos_u <= max_u,
                "Scatterer pos_u {} outside patch bounds [{}, {}]",
                s.pos_u,
                min_u,
                max_u
            );
            assert!(
                s.pos_v >= min_v && s.pos_v <= max_v,
                "Scatterer pos_v {} outside patch bounds [{}, {}]",
                s.pos_v,
                min_v,
                max_v
            );
        }
    }

    #[test]
    fn patch_bounds_mapping() {
        let bounds = PatchBounds {
            center_u: 0.5,
            center_v: 0.5,
            half_size: 0.25,
        };

        // Test corner mappings
        let (u, v) = bounds.map_to_patch(0.0, 0.0);
        assert!((u - 0.25).abs() < 1e-6);
        assert!((v - 0.25).abs() < 1e-6);

        let (u, v) = bounds.map_to_patch(1.0, 1.0);
        assert!((u - 0.75).abs() < 1e-6);
        assert!((v - 0.75).abs() < 1e-6);

        let (u, v) = bounds.map_to_patch(0.5, 0.5);
        assert!((u - 0.5).abs() < 1e-6);
        assert!((v - 0.5).abs() < 1e-6);
    }
}
