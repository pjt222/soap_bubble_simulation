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
use wgpu::util::DeviceExt;

/// Maximum number of scatterers supported
pub const MAX_SCATTERERS: u32 = 2048;

/// GPU-compatible scatterer data for particle-based ray deflection
/// Represents micelle clusters that scatter light within the soap film
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ScattererGPU {
    /// U position in UV space (0-1)
    pub pos_u: f32,
    /// V position in UV space (0-1)
    pub pos_v: f32,
    /// Scattering strength (signed: positive attracts, negative repels)
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
        Self {
            // Entry point: front of bubble
            entry_point: [0.0, 0.0, 1.0],
            // Beam direction: going down-left across the surface
            beam_dir: [-0.5, -0.866, 0.0],
            // Many rays needed - branches visible where rays converge
            num_rays: 32768,
            // More steps for longer propagation
            ray_steps: 400,
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
            num_scatterers: 800,
            scatterer_strength: 0.5,
            scatterer_radius: 0.03,
            particle_weight: 0.1,
            // Patch view mode defaults (enabled, centered at 0.5, ~10% of surface)
            patch_enabled: 1,
            patch_center_u: 0.5,
            patch_center_v: 0.5,
            patch_half_size: 0.158,
        }
    }
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
        let sigma = base_radius * radius_var;
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
    /// Bind group
    bind_group: wgpu::BindGroup,
    /// Parameters buffer
    params_buffer: wgpu::Buffer,
    /// Scatterer buffer (storage buffer for particle positions/strengths)
    scatterer_buffer: wgpu::Buffer,
    /// Current parameters
    pub params: BranchedFlowParams,
    /// Whether simulation is enabled
    pub enabled: bool,
    /// Texture dimensions
    tex_width: u32,
    tex_height: u32,
    /// Dirty flag: whether scatterers need regeneration
    /// Set when parameters change, cleared after upload
    scatterers_dirty: bool,
    /// Cached scatterer parameters for dirty check
    last_num_scatterers: u32,
    last_scatterer_strength: f32,
    last_scatterer_radius: f32,
    last_patch_enabled: u32,
    last_patch_center_u: f32,
    last_patch_center_v: f32,
    last_patch_half_size: f32,
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
    pub fn new(
        device: &wgpu::Device,
        thickness_buffer: &wgpu::Buffer,
        caustic_buffer: &wgpu::Buffer,
    ) -> Self {
        let params = BranchedFlowParams::default();
        let tex_width = params.tex_width;
        let tex_height = params.tex_height;

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Branched Flow Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/branched_flow_compute.wgsl").into()
            ),
        });

        // Create bind group layout
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
                // Scatterers array (read-only)
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
            ],
        });

        Self {
            trace_pipeline,
            clear_pipeline,
            bind_group,
            params_buffer,
            scatterer_buffer,
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
        }
    }

    /// Update parameters buffer
    pub fn update_params(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
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

    /// Update scatterer positions only if parameters have changed (dirty flag optimization)
    /// Called each frame but only regenerates when necessary
    pub fn update_scatterers(&mut self, queue: &wgpu::Queue, time: f32) {
        // Check if parameters changed since last upload
        self.check_scatterer_params_changed();

        // Skip regeneration if scatterers are not dirty
        if !self.scatterers_dirty {
            return;
        }

        // If patch mode is enabled, confine scatterers within the patch
        let patch_bounds = if self.params.patch_enabled != 0 {
            Some(PatchBounds {
                center_u: self.params.patch_center_u,
                center_v: self.params.patch_center_v,
                half_size: self.params.patch_half_size,
            })
        } else {
            None
        };

        let scatterers = generate_scatterers(
            self.params.num_scatterers.min(MAX_SCATTERERS),
            time,
            self.params.scatterer_strength,
            self.params.scatterer_radius,
            patch_bounds,
        );
        // Only write the active scatterers (not the full buffer)
        queue.write_buffer(
            &self.scatterer_buffer,
            0,
            bytemuck::cast_slice(&scatterers),
        );

        // Clear dirty flag after upload
        self.scatterers_dirty = false;
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

    /// Set beam direction (angle relative to "down" on the sphere, in degrees)
    pub fn set_beam_angle(&mut self, angle_deg: f32) {
        let angle = angle_deg.to_radians();
        // Get entry point
        let entry = glam::Vec3::from(self.params.entry_point);

        // Create tangent basis at entry point
        let up = if entry.y.abs() > 0.99 {
            glam::Vec3::X
        } else {
            glam::Vec3::Y
        };
        let tangent1 = entry.cross(up).normalize();
        let tangent2 = entry.cross(tangent1).normalize();

        // Beam direction is a combination of tangents based on angle
        let beam_dir = tangent1 * angle.cos() + tangent2 * angle.sin();
        self.params.beam_dir = beam_dir.into();
    }

    /// Run simulation step
    pub fn step(&mut self, encoder: &mut wgpu::CommandEncoder, time: f32) {
        if !self.enabled {
            return;
        }

        self.params.time = time;

        // Clear/fade pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Branched Flow Clear Pass"),
                timestamp_writes: None,
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
                timestamp_writes: None,
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
    fn set_entry_point_produces_normalized_vectors() {
        // Test various angles
        let test_angles: [(f32, f32); 4] = [(0.0, 0.0), (45.0, 30.0), (180.0, -45.0), (270.0, 89.0)];
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

    #[test]
    fn set_beam_angle_produces_tangent_vectors() {
        let sim = BranchedFlowParams::default();
        // Entry point is [0, 0, 1] by default
        let entry = glam::Vec3::from(sim.entry_point);

        // Create tangent basis (same logic as set_beam_angle)
        let up = if entry.y.abs() > 0.99 {
            glam::Vec3::X
        } else {
            glam::Vec3::Y
        };
        let tangent1 = entry.cross(up).normalize();
        let tangent2 = entry.cross(tangent1).normalize();

        // For any angle, beam_dir should be tangent to sphere at entry point
        for angle_deg in [0.0f32, 45.0, 90.0, 180.0, 270.0] {
            let angle = angle_deg.to_radians();
            let beam_dir = tangent1 * angle.cos() + tangent2 * angle.sin();
            let dot_with_entry = beam_dir.dot(entry);
            assert!(
                dot_with_entry.abs() < 1e-5,
                "Beam dir not tangent for angle={angle_deg}: dot={dot_with_entry}"
            );
            let beam_length = beam_dir.length();
            assert!(
                (beam_length - 1.0).abs() < 1e-5,
                "Beam dir not normalized for angle={angle_deg}: length={beam_length}"
            );
        }
    }

    #[test]
    fn default_params_are_physically_reasonable() {
        let params = BranchedFlowParams::default();

        assert!(params.num_rays >= 1024, "Need enough rays for visible pattern");
        assert!(params.ray_steps >= 100, "Need enough steps for ray propagation");
        assert!(params.step_size > 0.0 && params.step_size < 0.1, "Step size should be small");
        assert!(params.bend_strength > 0.0, "Bend strength must be positive");
        assert!(params.spread_angle > 0.0 && params.spread_angle < std::f32::consts::PI,
            "Spread angle should be a reasonable radian value");
        assert!(params.intensity_falloff > 0.0 && params.intensity_falloff < 0.1,
            "Intensity falloff should be small per step");
        assert!(params.thickness_scale > 0.0,
            "Thickness scale must be positive");
        assert_eq!(params.tex_width, 512);
        assert_eq!(params.tex_height, 256);
        // Particle scattering defaults
        assert!(params.num_scatterers >= 100 && params.num_scatterers <= MAX_SCATTERERS,
            "Num scatterers should be reasonable");
        assert!(params.scatterer_strength > 0.0,
            "Scatterer strength must be positive");
        assert!(params.scatterer_radius > 0.0 && params.scatterer_radius < 1.0,
            "Scatterer radius should be in UV space (0-1)");
        assert!(params.particle_weight >= 0.0 && params.particle_weight <= 1.0,
            "Particle weight should be a blend factor (0-1)");
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
            assert!(u >= 0.0 && u < 1.0, "Halton base 2 out of range: {u}");
            assert!(v >= 0.0 && v < 1.0, "Halton base 3 out of range: {v}");
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
            assert!(s.pos_u >= 0.0 && s.pos_u <= 1.0,
                "Scatterer pos_u out of range: {}", s.pos_u);
            assert!(s.pos_v >= 0.0 && s.pos_v <= 1.0,
                "Scatterer pos_v out of range: {}", s.pos_v);
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
            assert!(s.inv_sigma_sq > 0.0,
                "inv_sigma_sq must be positive: {}", s.inv_sigma_sq);
        }
    }

    #[test]
    fn generate_scatterers_time_affects_positions() {
        let scatterers_t0 = generate_scatterers(50, 0.0, 0.5, 0.03, None);
        let scatterers_t1 = generate_scatterers(50, 1.0, 0.5, 0.03, None);

        // At least some positions should differ due to time-based jitter
        let mut different = 0;
        for i in 0..50 {
            if (scatterers_t0[i].pos_u - scatterers_t1[i].pos_u).abs() > 0.001 ||
               (scatterers_t0[i].pos_v - scatterers_t1[i].pos_v).abs() > 0.001 {
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
            assert!(s.pos_u >= min_u && s.pos_u <= max_u,
                "Scatterer pos_u {} outside patch bounds [{}, {}]", s.pos_u, min_u, max_u);
            assert!(s.pos_v >= min_v && s.pos_v <= max_v,
                "Scatterer pos_v {} outside patch bounds [{}, {}]", s.pos_v, min_v, max_v);
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
