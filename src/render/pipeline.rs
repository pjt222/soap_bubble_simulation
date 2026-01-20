//! wgpu render pipeline for soap bubble visualization

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use std::path::Path;

use crate::physics::geometry::{SphereMesh, Vertex};
use crate::render::camera::Camera;
use crate::export::image_export;

/// Bubble-specific uniform data
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BubbleUniform {
    // Visual properties (9 floats)
    pub refractive_index: f32,
    pub base_thickness_nm: f32,
    pub time: f32,
    pub interference_intensity: f32,
    pub base_alpha: f32,
    pub edge_alpha: f32,
    pub background_r: f32,
    pub background_g: f32,
    pub background_b: f32,

    // Film dynamics parameters (4 floats)
    pub film_time: f32,
    pub swirl_intensity: f32,
    pub drainage_speed: f32,
    pub pattern_scale: f32,

    // Padding for 16-byte alignment (3 floats)
    pub _padding: [f32; 3],
}

impl Default for BubbleUniform {
    fn default() -> Self {
        Self {
            refractive_index: 1.33,
            base_thickness_nm: 500.0,
            time: 0.0,
            interference_intensity: 4.0,
            base_alpha: 0.3,
            edge_alpha: 0.6,
            background_r: 0.1,
            background_g: 0.1,
            background_b: 0.15,
            // Film dynamics defaults
            film_time: 0.0,
            swirl_intensity: 1.0,
            drainage_speed: 0.5,
            pattern_scale: 1.0,
            _padding: [0.0; 3],
        }
    }
}

/// Main render pipeline for the soap bubble
pub struct RenderPipeline {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    camera_buffer: wgpu::Buffer,
    bubble_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    depth_texture: wgpu::TextureView,
    pub camera: Camera,
    pub bubble_uniform: BubbleUniform,
    // Mesh settings
    pub subdivision_level: u32,
    radius: f32,
    // egui integration
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    // FPS tracking
    frame_times: Vec<f32>,
    fps: f32,
    // Animation state
    rotation_playing: bool,
    rotation_speed: f32,  // radians per second for camera yaw
    film_playing: bool,
    film_speed: f32,
    // Export state
    pub recording: bool,
    pub frame_counter: u32,
    pub screenshot_requested: bool,
}

impl RenderPipeline {
    /// Create a new render pipeline
    pub async fn new(window: std::sync::Arc<winit::window::Window>) -> Self {
        let size = window.inner_size();

        // Create wgpu instance
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Create surface
        let surface = instance.create_surface(window.clone()).unwrap();

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        // Request device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None,
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .unwrap();

        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Create depth texture
        let depth_texture = Self::create_depth_texture(&device, &config);

        // Create camera
        let camera = Camera::new(size.width as f32 / size.height as f32);
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
                    visibility: wgpu::ShaderStages::FRAGMENT, // Only used in fragment shader for film dynamics
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("bind_group_layout"),
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
            ],
            label: Some("bind_group"),
        });

        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bubble Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bubble.wgsl").into()),
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
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
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Draw both sides of the bubble
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

        // Create UV sphere mesh (5cm diameter, level 3 = 128 segments, ~32k triangles)
        let radius = 0.025;
        let subdivision_level = 3_u32;
        let mesh = SphereMesh::new(radius, subdivision_level);

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

        // Initialize egui
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, surface_format, Some(wgpu::TextureFormat::Depth32Float), 1, false);

        Self {
            surface,
            device,
            queue,
            config,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            camera_buffer,
            bubble_buffer,
            bind_group,
            depth_texture,
            camera,
            bubble_uniform,
            subdivision_level,
            radius,
            egui_ctx,
            egui_state,
            egui_renderer,
            frame_times: Vec::with_capacity(60),
            fps: 0.0,
            // Animation state
            rotation_playing: false,
            rotation_speed: 0.5,  // radians per second
            film_playing: true,
            film_speed: 1.0,
            // Export state
            recording: false,
            frame_counter: 0,
            screenshot_requested: false,
        }
    }

    /// Regenerate mesh with new subdivision level
    pub fn set_subdivision_level(&mut self, level: u32) {
        if level == self.subdivision_level || level > 5 {
            return; // No change or too high
        }
        self.subdivision_level = level;

        let mesh = SphereMesh::new(self.radius, level);

        self.vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: mesh.vertex_bytes(),
            usage: wgpu::BufferUsages::VERTEX,
        });

        self.index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: mesh.index_bytes(),
            usage: wgpu::BufferUsages::INDEX,
        });

        self.num_indices = mesh.indices.len() as u32;
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> wgpu::TextureView {
        let size = wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Handle window resize
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture = Self::create_depth_texture(&self.device, &self.config);
            self.camera.set_aspect(new_size.width as f32 / new_size.height as f32);
        }
    }

    /// Handle window events for egui
    pub fn handle_event(&mut self, window: &winit::window::Window, event: &winit::event::WindowEvent) -> bool {
        let response = self.egui_state.on_window_event(window, event);
        response.consumed
    }

    /// Update time for animation
    pub fn update(&mut self, dt: f32) {
        self.bubble_uniform.time += dt;

        // Camera rotation animation (orbits around Y axis)
        if self.rotation_playing {
            self.camera.yaw += dt * self.rotation_speed;
            // Keep yaw in valid range
            if self.camera.yaw > std::f32::consts::TAU {
                self.camera.yaw -= std::f32::consts::TAU;
            } else if self.camera.yaw < 0.0 {
                self.camera.yaw += std::f32::consts::TAU;
            }
        }

        // Film animation
        if self.film_playing {
            self.bubble_uniform.film_time += dt * self.film_speed;
        }

        // Track FPS
        self.frame_times.push(dt);
        if self.frame_times.len() > 60 {
            self.frame_times.remove(0);
        }
        if !self.frame_times.is_empty() {
            let avg_dt: f32 = self.frame_times.iter().sum::<f32>() / self.frame_times.len() as f32;
            self.fps = 1.0 / avg_dt;
        }

        // Update uniform buffers
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

    /// Render a frame with egui overlay
    pub fn render(&mut self, window: &winit::window::Window) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Build egui UI - extract mutable values to avoid borrow conflicts
        let raw_input = self.egui_state.take_egui_input(window);
        let mut thickness = self.bubble_uniform.base_thickness_nm;
        let mut refractive_index = self.bubble_uniform.refractive_index;
        let mut interference_intensity = self.bubble_uniform.interference_intensity;
        let mut base_alpha = self.bubble_uniform.base_alpha;
        let mut edge_alpha = self.bubble_uniform.edge_alpha;
        let mut bg_r = self.bubble_uniform.background_r;
        let mut bg_g = self.bubble_uniform.background_g;
        let mut bg_b = self.bubble_uniform.background_b;
        let mut subdivision = self.subdivision_level;
        let camera_distance = self.camera.distance;
        let camera_yaw = self.camera.yaw;
        let camera_pitch = self.camera.pitch;
        let fps = self.fps;
        let width = self.config.width;
        let height = self.config.height;
        let num_triangles = self.num_indices / 3;
        let time = self.bubble_uniform.time;

        // Animation state extraction
        let mut rotation_playing = self.rotation_playing;
        let mut rotation_speed = self.rotation_speed;
        let mut film_playing = self.film_playing;
        let mut film_speed = self.film_speed;
        let mut swirl_intensity = self.bubble_uniform.swirl_intensity;
        let mut drainage_speed = self.bubble_uniform.drainage_speed;
        let mut pattern_scale = self.bubble_uniform.pattern_scale;

        // Export state extraction
        let mut screenshot_requested = self.screenshot_requested;
        let mut recording = self.recording;
        let frame_counter = self.frame_counter;

        let egui_output = self.egui_ctx.run(raw_input, |ctx| {
            Self::build_ui_inner(
                ctx,
                &mut thickness,
                &mut refractive_index,
                &mut interference_intensity,
                &mut base_alpha,
                &mut edge_alpha,
                &mut bg_r,
                &mut bg_g,
                &mut bg_b,
                &mut subdivision,
                camera_distance,
                camera_yaw,
                camera_pitch,
                fps,
                width,
                height,
                num_triangles,
                time,
                // Animation parameters
                &mut rotation_playing,
                &mut rotation_speed,
                &mut film_playing,
                &mut film_speed,
                &mut swirl_intensity,
                &mut drainage_speed,
                &mut pattern_scale,
                // Export parameters
                &mut screenshot_requested,
                &mut recording,
                frame_counter,
            );
        });

        // Write back modified values
        self.bubble_uniform.base_thickness_nm = thickness;
        self.bubble_uniform.refractive_index = refractive_index;
        self.bubble_uniform.interference_intensity = interference_intensity;
        self.bubble_uniform.base_alpha = base_alpha;
        self.bubble_uniform.edge_alpha = edge_alpha;
        self.bubble_uniform.background_r = bg_r;
        self.bubble_uniform.background_g = bg_g;
        self.bubble_uniform.background_b = bg_b;
        if subdivision != self.subdivision_level {
            self.set_subdivision_level(subdivision);
        }

        // Write back animation state
        self.rotation_playing = rotation_playing;
        self.rotation_speed = rotation_speed;
        self.film_playing = film_playing;
        self.film_speed = film_speed;
        self.bubble_uniform.swirl_intensity = swirl_intensity;
        self.bubble_uniform.drainage_speed = drainage_speed;
        self.bubble_uniform.pattern_scale = pattern_scale;

        // Write back export state
        self.screenshot_requested = screenshot_requested;
        // Handle recording toggle from UI
        if recording != self.recording {
            if recording {
                self.frame_counter = 0;
                let _ = std::fs::create_dir_all("screenshots");
                log::info!("Recording started");
            } else {
                log::info!("Recording stopped after {} frames", self.frame_counter);
            }
            self.recording = recording;
        }

        // Handle egui platform output
        self.egui_state.handle_platform_output(window, egui_output.platform_output);

        // Tessellate egui
        let clipped_primitives = self.egui_ctx.tessellate(egui_output.shapes, egui_output.pixels_per_point);

        // Update egui textures
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: egui_output.pixels_per_point,
        };

        for (id, image_delta) in &egui_output.textures_delta.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, *id, image_delta);
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Update egui buffers
        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &clipped_primitives,
            &screen_descriptor,
        );

        // Render bubble
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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
                    view: &self.depth_texture,
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

        // Render egui
        // Safety: The render pass is used immediately and dropped before encoder.finish()
        // The 'static lifetime is a limitation of the egui-wgpu API
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // Transmute lifetime - safe because render_pass is dropped before encoder.finish()
            let render_pass: &mut wgpu::RenderPass<'static> = unsafe {
                std::mem::transmute(&mut render_pass)
            };

            self.egui_renderer.render(render_pass, &clipped_primitives, &screen_descriptor);
        }

        // Free egui textures
        for id in &egui_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        // Check if we need to capture a frame
        let should_capture = self.screenshot_requested || self.recording;
        let staging_buffer = if should_capture {
            // Calculate buffer size with proper alignment
            let bytes_per_pixel = 4u32;
            let unpadded_bytes_per_row = self.config.width * bytes_per_pixel;
            let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) / align * align;
            let buffer_size = (padded_bytes_per_row * self.config.height) as u64;

            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Screenshot Staging Buffer"),
                size: buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            // Copy texture to buffer
            encoder.copy_texture_to_buffer(
                wgpu::ImageCopyTexture {
                    texture: &output.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyBuffer {
                    buffer: &buffer,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_bytes_per_row),
                        rows_per_image: Some(self.config.height),
                    },
                },
                wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
            );

            Some((buffer, padded_bytes_per_row))
        } else {
            None
        };

        self.queue.submit(std::iter::once(encoder.finish()));

        // Process screenshot if requested
        if let Some((staging_buffer, padded_bytes_per_row)) = staging_buffer {
            let buffer_slice = staging_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

            self.device.poll(wgpu::Maintain::Wait);

            if let Ok(Ok(())) = rx.recv() {
                let data = buffer_slice.get_mapped_range();
                let width = self.config.width;
                let height = self.config.height;
                let bytes_per_pixel = 4u32;

                // Remove row padding
                let mut pixels = Vec::with_capacity((width * height * 4) as usize);
                for row in 0..height {
                    let start = (row * padded_bytes_per_row) as usize;
                    let end = start + (width * bytes_per_pixel) as usize;
                    pixels.extend_from_slice(&data[start..end]);
                }

                drop(data);
                staging_buffer.unmap();

                // Convert BGRA to RGBA
                for chunk in pixels.chunks_exact_mut(4) {
                    chunk.swap(0, 2);
                }

                // Determine filename
                let _ = std::fs::create_dir_all("screenshots");
                let path = if self.screenshot_requested && !self.recording {
                    format!("screenshots/screenshot_{:04}.png", self.frame_counter)
                } else {
                    format!("screenshots/frame_{:04}.png", self.frame_counter)
                };

                // Export
                if let Err(e) = image_export::export_frame(&path, width, height, &pixels) {
                    log::error!("Failed to export frame: {}", e);
                } else {
                    log::info!("Saved: {}", path);
                }

                self.frame_counter += 1;
            }

            self.screenshot_requested = false;
        }

        output.present();

        Ok(())
    }

    /// Build the egui UI with interactive controls (static to avoid borrow issues)
    #[allow(clippy::too_many_arguments)]
    fn build_ui_inner(
        ctx: &egui::Context,
        thickness: &mut f32,
        refractive_index: &mut f32,
        interference_intensity: &mut f32,
        base_alpha: &mut f32,
        edge_alpha: &mut f32,
        bg_r: &mut f32,
        bg_g: &mut f32,
        bg_b: &mut f32,
        subdivision: &mut u32,
        camera_distance: f32,
        camera_yaw: f32,
        camera_pitch: f32,
        fps: f32,
        width: u32,
        height: u32,
        num_triangles: u32,
        time: f32,
        // Animation parameters
        rotation_playing: &mut bool,
        rotation_speed: &mut f32,
        film_playing: &mut bool,
        film_speed: &mut f32,
        swirl_intensity: &mut f32,
        drainage_speed: &mut f32,
        pattern_scale: &mut f32,
        // Export parameters
        screenshot_requested: &mut bool,
        recording: &mut bool,
        frame_counter: u32,
    ) {
        egui::Window::new("Soap Bubble")
            .default_pos([10.0, 10.0])
            .default_width(280.0)
            .resizable(false)
            .show(ctx, |ui| {
                ui.heading("Film Properties");
                ui.separator();

                ui.add(egui::Slider::new(thickness, 100.0..=1500.0)
                    .text("Thickness")
                    .suffix(" nm"));

                ui.add(egui::Slider::new(refractive_index, 1.0..=2.0)
                    .text("Refractive index")
                    .fixed_decimals(2));

                // Preset buttons
                ui.horizontal(|ui| {
                    if ui.button("Soap").clicked() {
                        *refractive_index = 1.33;
                        *thickness = 500.0;
                    }
                    if ui.button("Oil").clicked() {
                        *refractive_index = 1.47;
                        *thickness = 300.0;
                    }
                    if ui.button("Thin").clicked() {
                        *thickness = 150.0;
                    }
                    if ui.button("Thick").clicked() {
                        *thickness = 1200.0;
                    }
                });

                ui.separator();
                ui.heading("Render Settings");
                ui.separator();

                ui.add(egui::Slider::new(interference_intensity, 0.5..=10.0)
                    .text("Color intensity")
                    .fixed_decimals(1));

                ui.add(egui::Slider::new(base_alpha, 0.0..=1.0)
                    .text("Base opacity")
                    .fixed_decimals(2));

                ui.add(egui::Slider::new(edge_alpha, 0.0..=1.0)
                    .text("Edge opacity")
                    .fixed_decimals(2));

                ui.separator();
                ui.heading("Mesh & Background");
                ui.separator();

                ui.add(egui::Slider::new(subdivision, 1..=5)
                    .text("Mesh detail"));

                ui.horizontal(|ui| {
                    ui.label("Background:");
                    let mut color = [*bg_r, *bg_g, *bg_b];
                    if ui.color_edit_button_rgb(&mut color).changed() {
                        *bg_r = color[0];
                        *bg_g = color[1];
                        *bg_b = color[2];
                    }
                });

                ui.separator();
                ui.heading("Animation");
                ui.separator();

                ui.collapsing("Camera Orbit", |ui| {
                    ui.horizontal(|ui| {
                        let play_text = if *rotation_playing { "\u{23F8} Pause" } else { "\u{25B6} Play" };
                        if ui.button(play_text).clicked() {
                            *rotation_playing = !*rotation_playing;
                        }
                    });

                    ui.add(egui::Slider::new(rotation_speed, 0.1..=2.0)
                        .text("Speed")
                        .suffix(" rad/s")
                        .fixed_decimals(2));
                });

                ui.collapsing("Film Dynamics", |ui| {
                    ui.horizontal(|ui| {
                        let play_text = if *film_playing { "\u{23F8} Pause" } else { "\u{25B6} Play" };
                        if ui.button(play_text).clicked() {
                            *film_playing = !*film_playing;
                        }
                    });

                    ui.add(egui::Slider::new(film_speed, 0.1..=3.0)
                        .text("Speed")
                        .fixed_decimals(1));

                    ui.add(egui::Slider::new(swirl_intensity, 0.0..=2.0)
                        .text("Swirl")
                        .fixed_decimals(2));

                    ui.add(egui::Slider::new(drainage_speed, 0.0..=2.0)
                        .text("Drainage")
                        .fixed_decimals(2));

                    ui.add(egui::Slider::new(pattern_scale, 0.5..=3.0)
                        .text("Pattern scale")
                        .fixed_decimals(1));
                });

                ui.separator();
                ui.collapsing("Camera Info", |ui| {
                    egui::Grid::new("camera_grid")
                        .num_columns(2)
                        .spacing([20.0, 4.0])
                        .show(ui, |ui| {
                            ui.label("Distance:");
                            ui.label(format!("{:.3} m", camera_distance));
                            ui.end_row();

                            ui.label("Yaw:");
                            ui.label(format!("{:.1}°", camera_yaw.to_degrees()));
                            ui.end_row();

                            ui.label("Pitch:");
                            ui.label(format!("{:.1}°", camera_pitch.to_degrees()));
                            ui.end_row();
                        });
                });

                ui.collapsing("Performance", |ui| {
                    egui::Grid::new("perf_grid")
                        .num_columns(2)
                        .spacing([20.0, 4.0])
                        .show(ui, |ui| {
                            ui.label("FPS:");
                            ui.label(format!("{:.0}", fps));
                            ui.end_row();

                            ui.label("Resolution:");
                            ui.label(format!("{}x{}", width, height));
                            ui.end_row();

                            ui.label("Triangles:");
                            ui.label(format!("{}", num_triangles));
                            ui.end_row();

                            ui.label("Time:");
                            ui.label(format!("{:.1} s", time));
                            ui.end_row();
                        });
                });

                ui.separator();
                ui.heading("Export");
                ui.separator();

                ui.horizontal(|ui| {
                    if ui.button("\u{1F4F7} Screenshot").clicked() {
                        *screenshot_requested = true;
                    }

                    let record_text = if *recording {
                        "\u{23F9} Stop Recording"
                    } else {
                        "\u{23FA} Record"
                    };
                    if ui.button(record_text).clicked() {
                        *recording = !*recording;
                    }
                });

                if *recording {
                    ui.colored_label(egui::Color32::RED, format!("\u{1F534} Recording... Frame {}", frame_counter));
                }

                ui.small("F12: Screenshot | F11: Toggle Recording");

                ui.separator();
                ui.small("Drag to rotate | Scroll to zoom | ESC to exit");
            });
    }

    /// Capture current frame to a PNG file
    pub fn capture_frame<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let width = self.config.width;
        let height = self.config.height;

        // Calculate buffer size with proper alignment
        // wgpu requires rows to be aligned to 256 bytes
        let bytes_per_pixel = 4u32;
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) / align * align;
        let buffer_size = (padded_bytes_per_row * height) as u64;

        // Create staging buffer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Screenshot Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Get current frame texture
        let output = self.surface.get_current_texture()
            .map_err(|e| format!("Failed to get surface texture: {}", e))?;

        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Screenshot Encoder"),
        });

        // Copy texture to buffer
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &output.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &staging_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map buffer and read data
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().map_err(|e| format!("Failed to map buffer: {:?}", e))?;

        // Read data and remove padding
        let data = buffer_slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);

        for row in 0..height {
            let start = (row * padded_bytes_per_row) as usize;
            let end = start + (width * bytes_per_pixel) as usize;
            pixels.extend_from_slice(&data[start..end]);
        }

        drop(data);
        staging_buffer.unmap();

        // The texture format is BGRA, convert to RGBA
        for chunk in pixels.chunks_exact_mut(4) {
            chunk.swap(0, 2); // Swap B and R
        }

        // Export to PNG
        image_export::export_frame(path, width, height, &pixels)
            .map_err(|e| format!("Failed to export frame: {}", e))?;

        // Don't present this frame (it was consumed by screenshot)
        // The next render() call will present a new frame

        Ok(())
    }

    /// Request a screenshot on the next frame
    pub fn request_screenshot(&mut self) {
        self.screenshot_requested = true;
    }

    /// Toggle recording mode
    pub fn toggle_recording(&mut self) {
        self.recording = !self.recording;
        if self.recording {
            self.frame_counter = 0;
            // Create screenshots directory
            let _ = std::fs::create_dir_all("screenshots");
            log::info!("Recording started");
        } else {
            log::info!("Recording stopped after {} frames", self.frame_counter);
        }
    }

    /// Get window size
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }
}
