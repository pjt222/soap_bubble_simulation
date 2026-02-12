//! Caustic / Branched Flow Renderer
//!
//! Simulates light focusing through soap film thickness variations,
//! creating caustic-like branched patterns on surfaces below the bubble.
//!
//! # Physics
//! When light passes through a medium with varying thickness, rays are refracted
//! by different amounts. Where rays converge, bright caustic patterns form.
//! The characteristic "branched flow" pattern emerges when the correlation length
//! of thickness variations exceeds the wavelength of light.
//!
//! # Reference
//! Patsyk et al. (2020) - Observation of branched flow of light

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Uniform parameters for caustic computation
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CausticParams {
    /// Grid width (phi direction)
    pub grid_width: u32,
    /// Grid height (theta direction)
    pub grid_height: u32,
    /// Refractive index of soap film
    pub refractive_index: f32,
    /// Scale factor for thickness to optical path
    pub film_thickness_scale: f32,
    /// Light direction X
    pub light_dir_x: f32,
    /// Light direction Y
    pub light_dir_y: f32,
    /// Light direction Z
    pub light_dir_z: f32,
    /// Light intensity
    pub light_intensity: f32,
    /// Effective focal length for curvature-based focusing
    pub focal_length: f32,
    /// Overall caustic intensity multiplier
    pub caustic_intensity: f32,
    /// Sharpness of caustic patterns (power exponent)
    pub caustic_sharpness: f32,
    /// Threshold for branch detection
    pub branch_threshold: f32,
    /// Y position of ground plane
    pub ground_y: f32,
    /// Bubble radius for projection
    pub bubble_radius: f32,
    /// Animation time
    pub time: f32,
    /// Padding for alignment
    pub _padding: u32,
}

impl Default for CausticParams {
    fn default() -> Self {
        Self {
            grid_width: 128,
            grid_height: 64,
            refractive_index: 1.33,
            film_thickness_scale: 1e6,
            light_dir_x: 0.0,
            light_dir_y: -1.0,
            light_dir_z: 0.0,
            light_intensity: 1.0,
            focal_length: 0.1,
            caustic_intensity: 2.0,
            caustic_sharpness: 1.5,
            branch_threshold: 0.001,
            ground_y: -0.1,
            bubble_radius: 0.025,
            time: 0.0,
            _padding: 0,
        }
    }
}

/// Vertex for ground plane
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GroundVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

impl GroundVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x2,
    ];

    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// Caustic renderer that computes and displays caustic patterns
// put id:'gpu_compute_caustics', label:'Caustic compute + render', input:'compute_results_gpu.internal', output:'framebuffer_gpu.internal'
pub struct CausticRenderer {
    /// Compute pipeline for caustic calculation
    compute_pipeline: wgpu::ComputePipeline,
    /// Render pipeline for ground plane caustics
    render_pipeline: wgpu::RenderPipeline,
    /// Caustic parameters buffer
    params_buffer: wgpu::Buffer,
    /// Caustic map storage buffer
    _caustic_buffer: wgpu::Buffer,
    /// Compute bind group
    compute_bind_group: wgpu::BindGroup,
    /// Render bind group
    render_bind_group: wgpu::BindGroup,
    /// Ground plane vertex buffer
    ground_vertices: wgpu::Buffer,
    /// Ground plane index buffer
    ground_indices: wgpu::Buffer,
    /// Number of ground indices
    num_ground_indices: u32,
    /// Current parameters
    pub params: CausticParams,
    /// Whether caustics are enabled
    pub enabled: bool,
}

impl CausticRenderer {
    /// Create a new caustic renderer
    pub fn new(
        device: &wgpu::Device,
        camera_buffer: &wgpu::Buffer,
        _camera_bind_group_layout: &wgpu::BindGroupLayout,
        thickness_buffer: &wgpu::Buffer,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        msaa_samples: u32,
    ) -> Self {
        let params = CausticParams::default();
        let grid_size = (params.grid_width * params.grid_height) as usize;

        // Create params buffer
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Caustic Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create caustic map buffer
        let caustic_data: Vec<f32> = vec![0.0; grid_size];
        let caustic_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Caustic Map Buffer"),
            contents: bytemuck::cast_slice(&caustic_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Load compute shader
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Caustics Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/caustics_compute.wgsl").into()),
        });

        // Load render shader
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Caustics Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/caustics.wgsl").into()),
        });

        // Compute bind group layout: params, thickness_field (read), caustic_map (read_write)
        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Caustic Compute Bind Group Layout"),
                entries: &[
                    // params (uniform)
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
                    // thickness_field (read-only storage)
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
                    // caustic_map (read-write storage)
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

        // Compute bind group
        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Caustic Compute Bind Group"),
            layout: &compute_bind_group_layout,
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

        // Compute pipeline
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Caustic Compute Pipeline Layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Caustic Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Render bind group layout: params, camera, caustic_map (read-only)
        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Caustic Render Bind Group Layout"),
                entries: &[
                    // params (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // camera (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // caustic_map (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Render bind group
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Caustic Render Bind Group"),
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: caustic_buffer.as_entire_binding(),
                },
            ],
        });

        // Render pipeline
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Caustic Render Pipeline Layout"),
                bind_group_layouts: &[&render_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Caustic Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                buffers: &[GroundVertex::buffer_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Max,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Create ground plane geometry
        let (ground_vertices, ground_indices, num_ground_indices) =
            Self::create_ground_plane(device, params.bubble_radius * 3.0, params.ground_y);

        Self {
            compute_pipeline,
            render_pipeline,
            params_buffer,
            _caustic_buffer: caustic_buffer,
            compute_bind_group,
            render_bind_group,
            ground_vertices,
            ground_indices,
            num_ground_indices,
            params,
            enabled: false,
        }
    }

    /// Create ground plane mesh
    fn create_ground_plane(
        device: &wgpu::Device,
        size: f32,
        y: f32,
    ) -> (wgpu::Buffer, wgpu::Buffer, u32) {
        let half = size;

        let vertices = [
            GroundVertex {
                position: [-half, y, -half],
                uv: [0.0, 0.0],
            },
            GroundVertex {
                position: [half, y, -half],
                uv: [1.0, 0.0],
            },
            GroundVertex {
                position: [half, y, half],
                uv: [1.0, 1.0],
            },
            GroundVertex {
                position: [-half, y, half],
                uv: [0.0, 1.0],
            },
        ];

        let indices: [u32; 6] = [0, 1, 2, 0, 2, 3];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Ground Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Ground Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        (vertex_buffer, index_buffer, indices.len() as u32)
    }

    /// Update parameters
    pub fn update_params(&mut self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
    }

    /// Run caustic compute pass
    pub fn compute(&self, encoder: &mut wgpu::CommandEncoder) {
        if !self.enabled {
            return;
        }

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Caustic Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.compute_pipeline);
        compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);

        let workgroups_x = self.params.grid_width.div_ceil(16);
        let workgroups_y = self.params.grid_height.div_ceil(16);
        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    /// Render caustics on ground plane
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if !self.enabled {
            return;
        }

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.render_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.ground_vertices.slice(..));
        render_pass.set_index_buffer(self.ground_indices.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.num_ground_indices, 0, 0..1);
    }

    /// Update ground plane position
    pub fn set_ground_y(&mut self, device: &wgpu::Device, y: f32) {
        self.params.ground_y = y;
        let (vertices, indices, count) =
            Self::create_ground_plane(device, self.params.bubble_radius * 3.0, y);
        self.ground_vertices = vertices;
        self.ground_indices = indices;
        self.num_ground_indices = count;
    }
}
