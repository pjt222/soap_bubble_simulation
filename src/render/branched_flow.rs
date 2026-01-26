//! Branched flow simulation using GPU compute
//! Traces light rays through the soap film, bending based on thickness gradients

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

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
    pub _padding: u32,
}

impl Default for BranchedFlowParams {
    fn default() -> Self {
        Self {
            // Entry point: front of bubble
            entry_point: [0.0, 0.0, 1.0],
            // Beam direction: going down-left across the surface
            beam_dir: [-0.5, -0.866, 0.0],
            num_rays: 2048,
            ray_steps: 200,
            step_size: 0.02,
            bend_strength: 5.0,
            spread_angle: 0.3,
            intensity_falloff: 0.005,
            tex_width: 256,
            tex_height: 128,
            time: 0.0,
            _padding: 0,
        }
    }
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
    /// Current parameters
    pub params: BranchedFlowParams,
    /// Whether simulation is enabled
    pub enabled: bool,
    /// Texture dimensions
    tex_width: u32,
    tex_height: u32,
}

/// Create a branched flow texture buffer (called early in pipeline init)
pub fn create_branched_flow_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    let tex_width = 256u32;
    let tex_height = 128u32;
    let tex_size = (tex_width * tex_height) as usize;
    let caustic_data = vec![0.0f32; tex_size];

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
            ],
        });

        Self {
            trace_pipeline,
            clear_pipeline,
            bind_group,
            params_buffer,
            params,
            enabled: false,
            tex_width,
            tex_height,
        }
    }

    /// Update parameters buffer
    pub fn update_params(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
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
            let workgroups_x = (self.tex_width + 15) / 16;
            let workgroups_y = (self.tex_height + 15) / 16;
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
            let workgroups = (self.params.num_rays + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }

    /// Get texture dimensions
    pub fn texture_size(&self) -> (u32, u32) {
        (self.tex_width, self.tex_height)
    }
}
