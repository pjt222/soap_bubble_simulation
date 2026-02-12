//! GPU-based drainage simulation using compute shaders.
//!
//! Moves the drainage PDE solver from CPU to GPU for real-time physics.
//! Uses double-buffering for safe parallel updates.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Uniform parameters for the drainage compute shader with Marangoni effect
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DrainageParams {
    /// Time step in seconds
    pub dt: f32,
    /// Gravitational acceleration (m/s²)
    pub gravity: f32,
    /// Dynamic viscosity (Pa·s)
    pub viscosity: f32,
    /// Fluid density (kg/m³)
    pub density: f32,
    /// Thickness diffusion coefficient (m²/s)
    pub diffusion_coeff: f32,
    /// Bubble radius for surface Laplacian (m)
    pub bubble_radius: f32,
    /// Critical thickness below which bubble bursts (m)
    pub critical_thickness: f32,
    /// Number of phi (azimuthal) grid points
    pub grid_width: u32,
    /// Number of theta (polar) grid points
    pub grid_height: u32,
    /// Whether Marangoni effect is enabled (0 or 1)
    pub marangoni_enabled: u32,
    /// Surface tension of clean interface (N/m), typically 0.072 for water
    pub gamma_air: f32,
    /// Surface tension reduction rate (N/m per concentration)
    pub gamma_reduction: f32,
    /// Surfactant diffusion coefficient (m²/s)
    pub surfactant_diffusion: f32,
    /// Marangoni stress coefficient M/η
    pub marangoni_coeff: f32,
    /// Padding for 16-byte alignment
    pub _padding1: u32,
    pub _padding2: u32,
}

impl Default for DrainageParams {
    fn default() -> Self {
        Self {
            dt: 0.001,                    // 1ms time step
            gravity: 9.81,                // Earth gravity
            viscosity: 0.001,             // Water-like (Pa·s)
            density: 1000.0,              // Water density (kg/m³)
            diffusion_coeff: 1e-9,        // Thickness diffusion
            bubble_radius: 0.025,         // 2.5cm radius
            critical_thickness: 30e-9,    // 30nm critical thickness
            grid_width: 128,
            grid_height: 64,
            // Marangoni parameters
            marangoni_enabled: 0,         // Disabled by default
            gamma_air: 0.072,             // N/m (clean water-air interface)
            gamma_reduction: 0.045,       // How much soap reduces tension
            surfactant_diffusion: 1e-9,   // m²/s
            marangoni_coeff: 0.01,        // Stress coefficient
            _padding1: 0,
            _padding2: 0,
        }
    }
}

/// GPU-based drainage simulator with Marangoni effect support.
///
/// Runs the coupled drainage + surfactant PDEs on the GPU using compute shaders.
/// Uses double-buffering for safe parallel updates of both thickness and concentration.
///
/// # Physics
/// - Drainage: dh/dt = -ρgh³/(3η)sin(θ) + D_h∇²h + Marangoni coupling
/// - Surfactant: DΓ/Dt = D_s∇²Γ + advection
/// - Surface tension: γ(Γ) = γ_air - γ_r × Γ
/// - Marangoni stress: τ = -γ_r × ∇Γ
// put id:'gpu_compute_drainage', label:'GPU drainage compute', input:'uniform_buffers_gpu.internal', output:'compute_results_gpu.internal'
pub struct GPUDrainageSimulator {
    /// Double-buffered storage for thickness field
    thickness_buffers: [wgpu::Buffer; 2],
    /// Double-buffered storage for surfactant concentration
    concentration_buffers: [wgpu::Buffer; 2],
    /// Current buffer index (0 or 1)
    current_buffer: usize,
    /// Uniform buffer for drainage parameters
    params_buffer: wgpu::Buffer,
    /// Compute pipeline for drainage step
    compute_pipeline: wgpu::ComputePipeline,
    /// Bind groups for alternating buffer access
    bind_groups: [wgpu::BindGroup; 2],
    /// Bind group layout (for potential rebinding)
    _bind_group_layout: wgpu::BindGroupLayout,
    /// Current drainage parameters
    params: DrainageParams,
    /// Current simulation time (seconds)
    current_time: f64,
    /// Whether simulation is running
    pub enabled: bool,
    /// Time scale multiplier for visual effect
    pub time_scale: f32,
    /// Number of compute steps per frame
    pub steps_per_frame: u32,
    /// Whether Marangoni effect is enabled
    pub marangoni_enabled: bool,
}

impl GPUDrainageSimulator {
    /// Create a new GPU drainage simulator with Marangoni effect support.
    ///
    /// # Arguments
    /// * `device` - wgpu device
    /// * `initial_thickness` - Initial uniform thickness in meters
    /// * `grid_width` - Number of phi (azimuthal) grid points
    /// * `grid_height` - Number of theta (polar) grid points
    pub fn new(
        device: &wgpu::Device,
        initial_thickness: f32,
        grid_width: u32,
        grid_height: u32,
    ) -> Self {
        let total_cells = (grid_width * grid_height) as usize;

        // Initialize thickness field with uniform values
        let initial_thickness_data: Vec<f32> = vec![initial_thickness; total_cells];

        // Initialize surfactant concentration with uniform values (normalized 0-1)
        let initial_concentration: f32 = 0.5;  // 50% coverage
        let initial_conc_data: Vec<f32> = vec![initial_concentration; total_cells];

        // Create double-buffered storage for thickness
        let thickness_buffers = [
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Thickness Buffer A"),
                contents: bytemuck::cast_slice(&initial_thickness_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            }),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Thickness Buffer B"),
                contents: bytemuck::cast_slice(&initial_thickness_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            }),
        ];

        // Create double-buffered storage for surfactant concentration
        let concentration_buffers = [
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Concentration Buffer A"),
                contents: bytemuck::cast_slice(&initial_conc_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            }),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Concentration Buffer B"),
                contents: bytemuck::cast_slice(&initial_conc_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            }),
        ];

        // Initialize parameters
        let params = DrainageParams {
            grid_width,
            grid_height,
            ..DrainageParams::default()
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Drainage Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout with thickness and concentration buffers
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Drainage Bind Group Layout"),
            entries: &[
                // thickness_in (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // thickness_out (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // concentration_in (read-only storage)
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
                // concentration_out (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create bind groups for both buffer configurations
        let bind_groups = [
            // A -> B (read from A, write to B)
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Drainage Bind Group A->B"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: thickness_buffers[0].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: thickness_buffers[1].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: concentration_buffers[0].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: concentration_buffers[1].as_entire_binding(),
                    },
                ],
            }),
            // B -> A (read from B, write to A)
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Drainage Bind Group B->A"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: thickness_buffers[1].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: thickness_buffers[0].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: concentration_buffers[1].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: concentration_buffers[0].as_entire_binding(),
                    },
                ],
            }),
        ];

        // Create compute shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Drainage Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/drainage.wgsl").into()),
        });

        // Create compute pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Drainage Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Drainage Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("drainage_step"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            thickness_buffers,
            concentration_buffers,
            current_buffer: 0,
            params_buffer,
            compute_pipeline,
            bind_groups,
            _bind_group_layout: bind_group_layout,
            params,
            current_time: 0.0,
            enabled: false,
            time_scale: 100.0,  // Speed up for visible effect
            steps_per_frame: 10,
            marangoni_enabled: false,
        }
    }

    /// Run one or more drainage simulation steps.
    ///
    /// # Arguments
    /// * `encoder` - Command encoder to record compute passes
    /// * `dt` - Frame delta time in seconds
    pub fn step(&mut self, encoder: &mut wgpu::CommandEncoder, dt: f32) {
        if !self.enabled {
            return;
        }

        // Scale time for visible effect
        let scaled_dt = dt * self.time_scale;

        // Compute per-step dt
        let step_dt = scaled_dt / self.steps_per_frame as f32;

        // Calculate workgroup dispatch sizes
        let workgroups_x = self.params.grid_width.div_ceil(16);
        let workgroups_y = self.params.grid_height.div_ceil(16);

        // Run multiple steps per frame for stability and speed
        for _ in 0..self.steps_per_frame {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Drainage Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.bind_groups[self.current_buffer], &[]);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);

            drop(compute_pass);

            // Swap buffers for next iteration
            self.current_buffer = 1 - self.current_buffer;
            self.current_time += step_dt as f64;
        }
    }

    /// Update drainage parameters.
    ///
    /// # Arguments
    /// * `queue` - wgpu queue for buffer updates
    /// * `params` - New drainage parameters
    pub fn update_params(&mut self, queue: &wgpu::Queue, params: DrainageParams) {
        self.params = params;
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));
    }

    /// Set the simulation time step.
    pub fn set_dt(&mut self, queue: &wgpu::Queue, dt: f32) {
        self.params.dt = dt;
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
    }

    /// Set physical parameters.
    pub fn set_physics(
        &mut self,
        queue: &wgpu::Queue,
        viscosity: f32,
        density: f32,
        gravity: f32,
        diffusion: f32,
    ) {
        self.params.viscosity = viscosity;
        self.params.density = density;
        self.params.gravity = gravity;
        self.params.diffusion_coeff = diffusion;
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
    }

    /// Set bubble radius (affects Laplacian calculation).
    pub fn set_bubble_radius(&mut self, queue: &wgpu::Queue, radius: f32) {
        self.params.bubble_radius = radius;
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
    }

    /// Reset the simulation to uniform thickness and concentration.
    ///
    /// # Arguments
    /// * `queue` - wgpu queue for buffer updates
    /// * `initial_thickness` - Thickness in meters
    pub fn reset(&mut self, queue: &wgpu::Queue, initial_thickness: f32) {
        let total_cells = (self.params.grid_width * self.params.grid_height) as usize;

        // Reset thickness buffers
        let thickness_data: Vec<f32> = vec![initial_thickness; total_cells];
        queue.write_buffer(&self.thickness_buffers[0], 0, bytemuck::cast_slice(&thickness_data));
        queue.write_buffer(&self.thickness_buffers[1], 0, bytemuck::cast_slice(&thickness_data));

        // Reset concentration buffers to uniform 50%
        let conc_data: Vec<f32> = vec![0.5; total_cells];
        queue.write_buffer(&self.concentration_buffers[0], 0, bytemuck::cast_slice(&conc_data));
        queue.write_buffer(&self.concentration_buffers[1], 0, bytemuck::cast_slice(&conc_data));

        self.current_buffer = 0;
        self.current_time = 0.0;

        log::info!("GPU drainage simulation reset to {} nm", initial_thickness * 1e9);
    }

    /// Enable or disable Marangoni effect.
    pub fn set_marangoni_enabled(&mut self, queue: &wgpu::Queue, enabled: bool) {
        self.marangoni_enabled = enabled;
        self.params.marangoni_enabled = if enabled { 1 } else { 0 };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
        log::info!("Marangoni effect {}", if enabled { "enabled" } else { "disabled" });
    }

    /// Set Marangoni physical parameters.
    pub fn set_marangoni_params(
        &mut self,
        queue: &wgpu::Queue,
        gamma_air: f32,
        gamma_reduction: f32,
        surfactant_diffusion: f32,
        marangoni_coeff: f32,
    ) {
        self.params.gamma_air = gamma_air;
        self.params.gamma_reduction = gamma_reduction;
        self.params.surfactant_diffusion = surfactant_diffusion;
        self.params.marangoni_coeff = marangoni_coeff;
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
    }

    /// Get the current concentration buffer for reading.
    pub fn concentration_buffer(&self) -> &wgpu::Buffer {
        &self.concentration_buffers[1 - self.current_buffer]
    }

    /// Get current simulation time in seconds.
    pub fn current_time(&self) -> f64 {
        self.current_time
    }

    /// Get the current (output) thickness buffer for reading.
    ///
    /// This buffer contains the most recent simulation results.
    pub fn current_buffer(&self) -> &wgpu::Buffer {
        // After step(), current_buffer points to the next input buffer,
        // so the output is the other one
        &self.thickness_buffers[1 - self.current_buffer]
    }

    /// Alias for current_buffer() - returns thickness buffer for external use.
    pub fn current_thickness_buffer(&self) -> &wgpu::Buffer {
        self.current_buffer()
    }

    /// Get grid dimensions.
    pub fn grid_size(&self) -> (u32, u32) {
        (self.params.grid_width, self.params.grid_height)
    }

    /// Get current drainage parameters.
    pub fn params(&self) -> &DrainageParams {
        &self.params
    }
}
