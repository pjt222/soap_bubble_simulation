//! Headless rendering pipeline for automated testing
//!
//! Provides GPU rendering without requiring a window or display,
//! enabling integration tests and screenshot comparison tests.

use wgpu::util::DeviceExt;

use crate::physics::geometry::{SphereMesh, Vertex};
use crate::render::camera::Camera;
use crate::render::pipeline::BubbleUniform;
use crate::render::branched_flow::create_branched_flow_buffer;
use crate::render::interference_lut::{
    generate_interference_lut, LUT_THICKNESS_SAMPLES, LUT_ANGLE_SAMPLES,
};

/// Headless render pipeline for testing without a window
///
/// This pipeline renders to an offscreen texture and can extract
/// pixel data for verification or comparison.
pub struct HeadlessRenderPipeline {
    device: wgpu::Device,
    queue: wgpu::Queue,
    render_texture: wgpu::Texture,
    render_texture_view: wgpu::TextureView,
    depth_texture_view: wgpu::TextureView,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    camera_buffer: wgpu::Buffer,
    bubble_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
    _format: wgpu::TextureFormat,
    pub camera: Camera,
    pub bubble_uniform: BubbleUniform,
}

impl HeadlessRenderPipeline {
    /// Create a new headless render pipeline
    ///
    /// # Arguments
    /// * `width` - Render target width in pixels
    /// * `height` - Render target height in pixels
    ///
    /// # Returns
    /// A new HeadlessRenderPipeline or None if GPU initialization fails
    pub async fn new(width: u32, height: u32) -> Option<Self> {
        // Create wgpu instance
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Request adapter without surface requirement (headless)
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        // Request device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: Some("Headless Device"),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .ok()?;

        // Use a common format for offscreen rendering
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;

        // Create offscreen render texture
        let render_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Headless Render Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let render_texture_view = render_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create depth texture
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Headless Depth Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create camera
        let camera = Camera::new(width as f32 / height as f32);
        let camera_uniform = camera.uniform();

        // Create bubble uniform
        let bubble_uniform = BubbleUniform::default();

        // Create uniform buffers
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bubble_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bubble Buffer"),
            contents: bytemuck::cast_slice(&[bubble_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create branched flow buffer (needed for shader binding 2)
        let branched_flow_buffer = create_branched_flow_buffer(&device);

        // Create interference LUT texture (pre-computed thin-film colors)
        let interference_lut_data = generate_interference_lut(
            bubble_uniform.refractive_index,
            1.0,
        );
        let interference_lut_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Headless Interference LUT Texture"),
            size: wgpu::Extent3d {
                width: LUT_THICKNESS_SAMPLES,
                height: LUT_ANGLE_SAMPLES,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &interference_lut_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &interference_lut_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(LUT_THICKNESS_SAMPLES * 4),
                rows_per_image: Some(LUT_ANGLE_SAMPLES),
            },
            wgpu::Extent3d {
                width: LUT_THICKNESS_SAMPLES,
                height: LUT_ANGLE_SAMPLES,
                depth_or_array_layers: 1,
            },
        );
        let interference_lut_view = interference_lut_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let interference_lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Headless Interference LUT Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                // Interference LUT texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Interference LUT sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("headless_bind_group_layout"),
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bubble_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: branched_flow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&interference_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&interference_lut_sampler),
                },
            ],
            label: Some("headless_bind_group"),
        });

        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Headless Bubble Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bubble.wgsl").into()),
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Headless Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Headless Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Create sphere mesh
        let radius = 0.025; // 5cm diameter bubble
        let mesh = SphereMesh::new(radius, 3);

        // Create vertex buffer
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: mesh.vertex_bytes(),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Create index buffer
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: mesh.index_bytes(),
            usage: wgpu::BufferUsages::INDEX,
        });

        let num_indices = mesh.indices.len() as u32;

        Some(Self {
            device,
            queue,
            render_texture,
            render_texture_view,
            depth_texture_view,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            camera_buffer,
            bubble_buffer,
            bind_group,
            width,
            height,
            _format: format,
            camera,
            bubble_uniform,
        })
    }

    /// Update uniform buffers with current camera and bubble state
    fn update_uniforms(&self) {
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera.uniform()]),
        );
        self.queue.write_buffer(
            &self.bubble_buffer,
            0,
            bytemuck::cast_slice(&[self.bubble_uniform]),
        );
    }

    /// Render a frame to the offscreen texture
    pub fn render(&mut self) {
        self.update_uniforms();

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Headless Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Headless Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.render_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.bubble_uniform.background_r as f64,
                            g: self.bubble_uniform.background_g as f64,
                            b: self.bubble_uniform.background_b as f64,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Render a frame and return the pixel data as RGBA bytes
    ///
    /// Returns a Vec<u8> with length width * height * 4
    pub fn render_to_buffer(&mut self) -> Vec<u8> {
        self.render();

        // Calculate buffer size with proper alignment
        let bytes_per_pixel = 4u32;
        let unpadded_bytes_per_row = self.width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let buffer_size = (padded_bytes_per_row * self.height) as u64;

        // Create staging buffer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Headless Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Headless Copy Encoder"),
        });

        // Copy texture to buffer
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.render_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map buffer and read data
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        self.device.poll(wgpu::Maintain::Wait);

        if rx.recv().unwrap().is_err() {
            return Vec::new();
        }

        // Read data and remove padding
        let data = buffer_slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((self.width * self.height * 4) as usize);

        for row in 0..self.height {
            let start = (row * padded_bytes_per_row) as usize;
            let end = start + (self.width * bytes_per_pixel) as usize;
            pixels.extend_from_slice(&data[start..end]);
        }

        drop(data);
        staging_buffer.unmap();

        pixels
    }

    /// Get render dimensions
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Set film thickness in nanometers
    pub fn set_thickness(&mut self, thickness_nm: f32) {
        self.bubble_uniform.base_thickness_nm = thickness_nm;
    }

    /// Set refractive index
    pub fn set_refractive_index(&mut self, n: f32) {
        self.bubble_uniform.refractive_index = n;
    }

    /// Orbit the camera
    pub fn orbit_camera(&mut self, delta_x: f32, delta_y: f32) {
        self.camera.orbit(delta_x, delta_y);
    }

    /// Zoom the camera
    pub fn zoom_camera(&mut self, delta: f32) {
        self.camera.zoom(delta);
    }

    /// Set camera distance directly
    pub fn set_camera_distance(&mut self, distance: f32) {
        self.camera.distance = distance.clamp(0.05, 10.0);
    }

    /// Update simulation time
    pub fn set_time(&mut self, time: f32) {
        self.bubble_uniform.time = time;
        self.bubble_uniform.film_time = time;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_headless_pipeline_creation() {
        // Use pollster to run async test
        let result = pollster::block_on(HeadlessRenderPipeline::new(256, 256));
        // May fail on systems without GPU, which is acceptable for unit tests
        if result.is_some() {
            let pipeline = result.unwrap();
            assert_eq!(pipeline.size(), (256, 256));
        }
    }

    #[test]
    fn test_bubble_uniform_defaults() {
        let uniform = BubbleUniform::default();
        assert!((uniform.refractive_index - 1.33).abs() < 0.01);
        assert!((uniform.base_thickness_nm - 500.0).abs() < 1.0);
    }
}
