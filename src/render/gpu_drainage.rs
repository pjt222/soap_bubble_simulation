//! GPU-based drainage simulation using compute shaders.
//!
//! Moves the drainage PDE solver from CPU to GPU for real-time physics.
//! Uses double-buffering for safe parallel updates.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Uniform parameters for the drainage compute shader
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
    /// Diffusion coefficient (m²/s)
    pub diffusion_coeff: f32,
    /// Bubble radius for surface Laplacian (m)
    pub bubble_radius: f32,
    /// Critical thickness below which bubble bursts (m)
    pub critical_thickness: f32,
    /// Number of phi (azimuthal) grid points
    pub grid_width: u32,
    /// Number of theta (polar) grid points
    pub grid_height: u32,
    /// Padding for 16-byte alignment
    pub _padding: [u32; 3],
}

impl Default for DrainageParams {
    fn default() -> Self {
        Self {
            dt: 0.001,                    // 1ms time step
            gravity: 9.81,                // Earth gravity
            viscosity: 0.001,             // Water-like (Pa·s)
            density: 1000.0,              // Water density (kg/m³)
            diffusion_coeff: 1e-9,        // Surfactant diffusion
            bubble_radius: 0.025,         // 2.5cm radius
            critical_thickness: 30e-9,    // 30nm critical thickness
            grid_width: 128,
            grid_height: 64,
            _padding: [0; 3],
        }
    }
}

/// GPU-based drainage simulator with double-buffered storage.
///
/// Runs the drainage PDE on the GPU using compute shaders.
/// The thickness field is stored in two alternating buffers
/// for safe parallel read/write operations.
pub struct GPUDrainageSimulator {
    /// Double-buffered storage for thickness field
    thickness_buffers: [wgpu::Buffer; 2],
    /// Current buffer index (0 or 1)
    current_buffer: usize,
    /// Uniform buffer for drainage parameters
    params_buffer: wgpu::Buffer,
    /// Compute pipeline for drainage step
    compute_pipeline: wgpu::ComputePipeline,
    /// Bind groups for alternating buffer access
    bind_groups: [wgpu::BindGroup; 2],
    /// Bind group layout (for potential rebinding)
    bind_group_layout: wgpu::BindGroupLayout,
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
}

impl GPUDrainageSimulator {
    /// Create a new GPU drainage simulator.
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
        let initial_data: Vec<f32> = vec![initial_thickness; total_cells];

        // Create double-buffered storage
        let thickness_buffers = [
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Thickness Buffer A"),
                contents: bytemuck::cast_slice(&initial_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            }),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Thickness Buffer B"),
                contents: bytemuck::cast_slice(&initial_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            }),
        ];

        // Initialize parameters
        let mut params = DrainageParams::default();
        params.grid_width = grid_width;
        params.grid_height = grid_height;

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Drainage Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout
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
            current_buffer: 0,
            params_buffer,
            compute_pipeline,
            bind_groups,
            bind_group_layout,
            params,
            current_time: 0.0,
            enabled: false,
            time_scale: 100.0,  // Speed up for visible effect
            steps_per_frame: 10,
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
        let workgroups_x = (self.params.grid_width + 15) / 16;
        let workgroups_y = (self.params.grid_height + 15) / 16;

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

    /// Reset the simulation to uniform thickness.
    ///
    /// # Arguments
    /// * `queue` - wgpu queue for buffer updates
    /// * `initial_thickness` - Thickness in meters
    pub fn reset(&mut self, queue: &wgpu::Queue, initial_thickness: f32) {
        let total_cells = (self.params.grid_width * self.params.grid_height) as usize;
        let data: Vec<f32> = vec![initial_thickness; total_cells];

        // Reset both buffers
        queue.write_buffer(&self.thickness_buffers[0], 0, bytemuck::cast_slice(&data));
        queue.write_buffer(&self.thickness_buffers[1], 0, bytemuck::cast_slice(&data));

        self.current_buffer = 0;
        self.current_time = 0.0;

        log::info!("GPU drainage simulation reset to {} nm", initial_thickness * 1e9);
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

    /// Get grid dimensions.
    pub fn grid_size(&self) -> (u32, u32) {
        (self.params.grid_width, self.params.grid_height)
    }

    /// Get current drainage parameters.
    pub fn params(&self) -> &DrainageParams {
        &self.params
    }
}
