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
//! # What this module computes
//! A stylised effect, not that physics: `caustics_compute.wgsl` maps the curvature and
//! slope of the film thickness (micrometres, differences per grid texel) to a brightness,
//! and `caustics.wgsl` maps ground (x, z) linearly onto the (phi, theta) grid, which is
//! not a projection of any light path (audit branched-02, #45). A uniform film gives
//! `caustic_intensity` everywhere. The audit's smooth drained profile (30 nm at the top to
//! 1000 nm at the bottom) spans 0.9% of it; how much the field the drainage solver will
//! produce (#43) varies the map is not measured. On screen the ground adds
//! `min(light * value, 1) * min(0.5 * value, 1)` per channel (`SrcAlpha`, `One`; an
//! 8-bit target clamps the colour to 1 before blending), so inside 80% of the disc radius
//! every value of 2.0 or more renders the same white, whatever is behind it. A one-off
//! render on lavapipe over black gave 0.72 (linear) at 1.45 and 1.0 at 2.0 and 3.0. The
//! compute bind group is built once and is not rebuilt when the GPU drainage swaps
//! buffers (#43).
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
    /// Converts the drainage thickness (metres) to micrometres: 1e6. The compute shader's
    /// `focal_length` and `branch_threshold` act on derivatives in micrometres per texel.
    pub film_thickness_scale: f32,
    /// Light direction X
    pub light_dir_x: f32,
    /// Light direction Y
    pub light_dir_y: f32,
    /// Light direction Z
    pub light_dir_z: f32,
    /// Light intensity
    pub light_intensity: f32,
    /// Effective focal length for curvature-based focusing, acting on the Laplacian of the
    /// thickness in micrometres per texel². Not retuned in #45: before it, the map was
    /// constant and this had no effect.
    pub focal_length: f32,
    /// Overall caustic intensity multiplier
    pub caustic_intensity: f32,
    /// Sharpness of caustic patterns (power exponent)
    pub caustic_sharpness: f32,
    /// Threshold on the Hessian determinant (micrometres per texel², squared) below which
    /// the branch factor applies. Not retuned in #45 (see `focal_length`).
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
            grid_width: 256,
            grid_height: 128,
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
    /// Inputs kept to rebuild `render_pipeline` when the MSAA sample count changes
    render_pipeline_layout: wgpu::PipelineLayout,
    render_shader: wgpu::ShaderModule,
    surface_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
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
            // COPY_SRC lets the tests read the map back
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
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

        let render_pipeline = Self::create_render_pipeline(
            device,
            &render_pipeline_layout,
            &render_shader,
            surface_format,
            depth_format,
            msaa_samples,
        );

        // Create ground plane geometry
        let (ground_vertices, ground_indices, num_ground_indices) =
            Self::create_ground_plane(device, params.bubble_radius * 3.0, params.ground_y);

        Self {
            compute_pipeline,
            render_pipeline,
            render_pipeline_layout,
            render_shader,
            surface_format,
            depth_format,
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

    /// Rebuild the ground-plane render pipeline for a new MSAA sample count.
    /// It draws into the same multisampled pass as the bubbles, so its count
    /// must match the pass.
    pub fn set_sample_count(&mut self, device: &wgpu::Device, sample_count: u32) {
        self.render_pipeline = Self::create_render_pipeline(
            device,
            &self.render_pipeline_layout,
            &self.render_shader,
            self.surface_format,
            self.depth_format,
            sample_count,
        );
    }

    fn create_render_pipeline(
        device: &wgpu::Device,
        render_pipeline_layout: &wgpu::PipelineLayout,
        render_shader: &wgpu::ShaderModule,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Caustic Render Pipeline"),
            layout: Some(render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: render_shader,
                entry_point: Some("vs_main"),
                buffers: &[GroundVertex::buffer_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: render_shader,
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
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
    const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    fn test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()
    }

    fn attachment(device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::TextureView {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 64,
                    height: 64,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Drainage grid size (`CausticParams::default()` and the GPU drainage buffer)
    const GRID_WIDTH: usize = 256;
    const GRID_HEIGHT: usize = 128;

    /// Caustic map after one compute pass over `thickness` (metres, row-major, rows = theta)
    fn caustic_map(thickness: &[f32]) -> Vec<f32> {
        assert_eq!(thickness.len(), GRID_WIDTH * GRID_HEIGHT);
        let Some((device, queue)) = test_device() else {
            panic!("no GPU adapter (run via scripts/test-local.sh for lavapipe)");
        };
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<crate::render::camera::CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let thickness_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(thickness),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let unused_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[],
        });
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut renderer = CausticRenderer::new(
            &device,
            &camera_buffer,
            &unused_layout,
            &thickness_buffer,
            COLOR_FORMAT,
            DEPTH_FORMAT,
            1,
        );
        renderer.enabled = true;
        let map_bytes = std::mem::size_of_val(thickness) as u64;
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: map_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        renderer.compute(&mut encoder);
        encoder.copy_buffer_to_buffer(&renderer._caustic_buffer, 0, &readback_buffer, 0, map_bytes);
        queue.submit(std::iter::once(encoder.finish()));
        if let Some(error) = pollster::block_on(device.pop_error_scope()) {
            panic!("caustic compute failed validation: {error}");
        }
        readback_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |result| {
                result.expect("map caustic map")
            });
        device.poll(wgpu::Maintain::Wait);
        bytemuck::cast_slice::<u8, f32>(&readback_buffer.slice(..).get_mapped_range()).to_vec()
    }

    /// CPU port of `compute_caustic_intensity` (caustics_compute.wgsl) at texel (x, y),
    /// on a thickness field already in micrometres. Also returns the Hessian determinant,
    /// so callers can skip texels where f32 and f64 may disagree about `branch_threshold`.
    fn reference_intensity(
        thickness_um: &[f64],
        params: &CausticParams,
        x: i32,
        y: i32,
    ) -> (f64, f64) {
        let (width, height) = (params.grid_width as i32, params.grid_height as i32);
        let sample = |column: i32, row: i32| {
            thickness_um[(row.clamp(0, height - 1) * width + column.rem_euclid(width)) as usize]
        };
        let center = sample(x, y);
        let (left, right, up, down) = (
            sample(x - 1, y),
            sample(x + 1, y),
            sample(x, y - 1),
            sample(x, y + 1),
        );
        let gradient = ((right - left) / 2.0, (down - up) / 2.0);
        let laplacian = left + right + up + down - 4.0 * center;
        let second_x = left + right - 2.0 * center;
        let second_y = up + down - 2.0 * center;
        let mixed = (sample(x + 1, y + 1) - sample(x - 1, y + 1) - sample(x + 1, y - 1)
            + sample(x - 1, y - 1))
            / 4.0;
        let hessian_det = second_x * second_y - mixed * mixed;
        let focusing = 1.0
            / (1.0 + laplacian * params.focal_length as f64)
                .abs()
                .max(0.01);
        let branch = if hessian_det < -(params.branch_threshold as f64) {
            1.5
        } else {
            1.0
        };
        let refraction = 1.0 + (gradient.0 * gradient.0 + gradient.1 * gradient.1).sqrt() * 0.5;
        let intensity = (focusing * branch * refraction).powf(params.caustic_sharpness as f64)
            * params.caustic_intensity as f64;
        (intensity.clamp(0.0, 10.0), hessian_det)
    }

    /// A 500 nm film (metres) with Gaussian bumps of `amplitude_nm` and width
    /// `sigma_texels` centred on the given texels; distances wrap in x (phi is periodic)
    fn thickness_with_bumps(
        amplitude_nm: f64,
        sigma_texels: f64,
        centres: &[(f64, f64)],
    ) -> Vec<f32> {
        (0..GRID_HEIGHT)
            .flat_map(|row| {
                (0..GRID_WIDTH).map(move |column| {
                    let bumps: f64 = centres
                        .iter()
                        .map(|&(centre_x, centre_y)| {
                            let dx = (column as f64 - centre_x).abs();
                            let dx = dx.min(GRID_WIDTH as f64 - dx);
                            let r_sq = dx * dx + (row as f64 - centre_y).powi(2);
                            amplitude_nm * (-r_sq / (2.0 * sigma_texels * sigma_texels)).exp()
                        })
                        .sum();
                    ((500.0 + bumps) * 1e-9) as f32
                })
            })
            .collect()
    }

    /// Compares a GPU caustic map with `reference_intensity` at every texel, converting the
    /// field with a literal 1e6 so a wrong `film_thickness_scale` cannot cancel out.
    /// Returns (compared, texels away from the uniform value, with the branch factor, skipped).
    fn assert_matches_reference(
        gpu_map: &[f32],
        thickness_m: &[f32],
        params: &CausticParams,
    ) -> (usize, usize, usize, usize) {
        let thickness_um: Vec<f64> = thickness_m
            .iter()
            .map(|&metres| metres as f64 * 1e6)
            .collect();
        let (mut compared, mut informative, mut branched, mut skipped) = (0, 0, 0, 0);
        for y in 0..GRID_HEIGHT as i32 {
            for x in 0..GRID_WIDTH as i32 {
                let (expected, hessian_det) = reference_intensity(&thickness_um, params, x, y);
                if (hessian_det + params.branch_threshold as f64).abs() < 1e-6 {
                    skipped += 1; // f32 and f64 may fall on different sides of the threshold
                    continue;
                }
                let actual = gpu_map[y as usize * GRID_WIDTH + x as usize] as f64;
                assert!(
                    (actual - expected).abs() < 1e-4 * expected,
                    "texel ({x}, {y}): GPU {actual}, micrometre reference {expected}"
                );
                compared += 1;
                if (expected - params.caustic_intensity as f64).abs() > 1e-4 * expected {
                    informative += 1;
                }
                if hessian_det < -(params.branch_threshold as f64) {
                    branched += 1;
                }
            }
        }
        assert!(
            skipped < 10,
            "{skipped} texels too close to branch_threshold to compare"
        );
        (compared, informative, branched, skipped)
    }

    #[test]
    #[ignore] // Requires GPU (lavapipe works: scripts/test-local.sh -- --ignored)
    fn caustic_map_follows_the_thickness_field_in_micrometres() {
        // #45: the compute shader differentiated the drainage thickness in metres, so every
        // derivative was ~1e-6 of its intended size and the map was caustic_intensity (2.0)
        // everywhere. The map must now match a CPU port of the shader formula evaluated in
        // micrometres (reference_intensity), which also pins the port's wrap in x and clamp
        // in y: one bump sits mid-grid, one on the phi seam (column 0), and the drained
        // profile reaches the pole rows.
        let params = CausticParams::default();
        assert_eq!(params.film_thickness_scale, 1e6, "metres to micrometres");
        let uniform = caustic_map(&vec![500e-9f32; GRID_WIDTH * GRID_HEIGHT]);
        assert!(
            uniform
                .iter()
                .all(|&value| (value - params.caustic_intensity).abs() < 1e-6),
            "a uniform film must give caustic_intensity everywhere"
        );

        // 500 nm bumps, sigma 2 texels: sharp enough for the branch factor
        let bumps = thickness_with_bumps(500.0, 2.0, &[(128.0, 64.0), (0.0, 64.0)]);
        let bump_map = caustic_map(&bumps);
        let (compared, informative, branched, skipped) =
            assert_matches_reference(&bump_map, &bumps, &params);
        let (minimum, maximum) = bump_map
            .iter()
            .fold((f32::MAX, f32::MIN), |(low, high), &value| {
                (low.min(value), high.max(value))
            });
        println!(
            "bumps: {compared} texels match the reference, {informative} of them away from \
             the uniform value, {branched} with the branch factor, {skipped} skipped; map \
             range {minimum:.4} to {maximum:.4}"
        );
        assert!(branched > 0, "the bumps never trigger the branch factor");
        assert!(maximum - minimum > 1.0, "map range {minimum} to {maximum}");

        // The audit's smooth drained profile, 30 nm at the top to 1000 nm at the bottom. Its
        // derivatives per texel are small, so the map varies little; the range is printed
        // as a diagnostic, not asserted. The effect is stylised; see the module documentation.
        let drained: Vec<f32> = (0..GRID_HEIGHT)
            .flat_map(|row| {
                let theta = row as f64 * std::f64::consts::PI / (GRID_HEIGHT - 1) as f64;
                let thickness_nm = 30.0 + 970.0 * (1.0 - theta.cos()) / 2.0;
                std::iter::repeat_n((thickness_nm * 1e-9) as f32, GRID_WIDTH)
            })
            .collect();
        let drained_map = caustic_map(&drained);
        let (drained_compared, drained_informative, _, _) =
            assert_matches_reference(&drained_map, &drained, &params);
        let (low, high) = drained_map
            .iter()
            .fold((f32::MAX, f32::MIN), |(low, high), &value| {
                (low.min(value), high.max(value))
            });
        println!(
            "drained profile: {drained_compared} texels match the reference \
             ({drained_informative} away from the uniform value); map range {low:.5} to \
             {high:.5}, a span of {:.2}% of caustic_intensity",
            (high - low) / params.caustic_intensity * 100.0
        );
    }

    #[test]
    #[ignore] // Requires GPU (lavapipe works: scripts/test-local.sh -- --ignored)
    fn test_set_sample_count_rebuilds_render_pipeline() {
        let Some((device, queue)) = test_device() else {
            return;
        };
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<crate::render::camera::CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let thickness_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256 * 128 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let unused_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[],
        });

        // Built for the default 4x MSAA pass, then switched to 1x as set_msaa_samples does.
        let mut renderer = CausticRenderer::new(
            &device,
            &camera_buffer,
            &unused_layout,
            &thickness_buffer,
            COLOR_FORMAT,
            DEPTH_FORMAT,
            4,
        );
        renderer.enabled = true;
        renderer.set_sample_count(&device, 1);

        let color_view = attachment(&device, COLOR_FORMAT);
        let depth_view = attachment(&device, DEPTH_FORMAT);
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("1x pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            renderer.render(&mut render_pass);
        }
        queue.submit(std::iter::once(encoder.finish()));
        let error = pollster::block_on(device.pop_error_scope());
        assert!(
            error.is_none(),
            "caustic pipeline does not match a 1x pass: {error:?}"
        );
    }
}
