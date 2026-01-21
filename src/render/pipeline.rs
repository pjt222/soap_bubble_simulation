//! wgpu render pipeline for soap bubble visualization

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use std::path::Path;

use crate::config::SimulationConfig;
use crate::physics::drainage::DrainageSimulator;
use crate::physics::geometry::{LodMeshCache, SphereMesh, Vertex};
use crate::physics::foam_dynamics::FoamSimulator;
use crate::render::camera::Camera;
use crate::render::gpu_drainage::GPUDrainageSimulator;
use crate::render::foam_renderer::FoamRenderer;
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

    // Bubble position (3 floats) - replaces padding
    pub position_x: f32,
    pub position_y: f32,
    pub position_z: f32,

    // Edge smoothing mode (0 = linear, 1 = smoothstep, 2 = power)
    pub edge_smoothing_mode: u32,
    // Padding for 16-byte alignment
    pub _padding1: u32,
    pub _padding2: u32,
    pub _padding3: u32,
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
            // Bubble position (starts at origin)
            position_x: 0.0,
            position_y: 0.0,
            position_z: 0.0,
            // Edge smoothing (default to smoothstep for smooth edges)
            edge_smoothing_mode: 1,
            _padding1: 0,
            _padding2: 0,
            _padding3: 0,
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
    msaa_texture: wgpu::TextureView,
    msaa_samples: u32,
    bind_group_layout: wgpu::BindGroupLayout,
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
    // External forces
    pub bubble_velocity: [f32; 3],
    pub wind_strength: f32,
    pub wind_direction: [f32; 3],  // Normalized direction
    pub buoyancy_strength: f32,
    pub forces_enabled: bool,
    // Drainage simulation
    drainage_simulator: Option<DrainageSimulator>,
    pub physics_drainage_enabled: bool,
    drainage_time_scale: f32,
    // Gravity deformation
    pub deformation_enabled: bool,
    pub aspect_ratio: f32,  // 1.0 = sphere, <1.0 = oblate (flattened)
    // LOD system
    lod_cache: LodMeshCache,
    current_lod_level: u32,
    lod_enabled: bool,
    lod_thresholds: [f32; 4],  // Distance thresholds for LOD transitions [5→4, 4→3, 3→2, 2→1]
    // GPU drainage simulation
    gpu_drainage: GPUDrainageSimulator,
    pub gpu_drainage_enabled: bool,
    // Multi-bubble foam system
    foam_simulator: Option<FoamSimulator>,
    foam_renderer: FoamRenderer,
    pub foam_enabled: bool,
    foam_time_scale: f32,
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

        // Default MSAA sample count
        let msaa_samples = 4_u32;

        // Create depth texture (MSAA)
        let depth_texture = Self::create_depth_texture(&device, &config, msaa_samples);

        // Create MSAA render target texture
        let msaa_texture = Self::create_msaa_texture(&device, &config, msaa_samples);

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
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
                count: msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Create UV sphere mesh with LOD support (5cm diameter)
        let radius = 0.025;
        let subdivision_level = 3_u32;
        let mut lod_cache = LodMeshCache::new(radius, 1.0);

        // Get initial mesh from LOD cache
        let mesh = lod_cache.get_mesh(subdivision_level);

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
        let egui_renderer = egui_wgpu::Renderer::new(&device, surface_format, None, 1, false);

        // Initialize GPU drainage simulator
        let gpu_drainage = GPUDrainageSimulator::new(
            &device,
            500e-9,  // Initial thickness: 500nm
            128,     // Grid width (phi)
            64,      // Grid height (theta)
        );

        // Initialize foam renderer
        let foam_renderer = FoamRenderer::new(&device, 64);

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
            msaa_texture,
            msaa_samples,
            bind_group_layout,
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
            // External forces
            bubble_velocity: [0.0, 0.0, 0.0],
            wind_strength: 0.0,
            wind_direction: [1.0, 0.0, 0.0],  // Default: blowing in +X direction
            buoyancy_strength: 0.02,  // Light upward force
            forces_enabled: false,
            // Drainage simulation (initialized lazily or with default config)
            drainage_simulator: None,
            physics_drainage_enabled: false,
            drainage_time_scale: 100.0,  // Speed up simulation for visible effect
            // Gravity deformation (disabled by default)
            deformation_enabled: false,
            aspect_ratio: 1.0,  // Perfect sphere
            // LOD system
            lod_cache,
            current_lod_level: subdivision_level,
            lod_enabled: false,  // Disabled by default, user can enable
            lod_thresholds: [0.08, 0.15, 0.30, 0.60],  // Distance thresholds in meters
            // GPU drainage simulation
            gpu_drainage,
            gpu_drainage_enabled: false,
            // Foam system (disabled by default)
            foam_simulator: None,
            foam_renderer,
            foam_enabled: false,
            foam_time_scale: 1.0,
        }
    }

    /// Initialize the drainage simulator with the given configuration.
    pub fn init_drainage_simulator(&mut self, config: &SimulationConfig) {
        self.drainage_simulator = Some(DrainageSimulator::new(config));
        log::info!("Drainage simulator initialized");
    }

    /// Reset the drainage simulator to initial thickness.
    pub fn reset_drainage(&mut self, initial_thickness_nm: f32) {
        if let Some(ref mut simulator) = self.drainage_simulator {
            simulator.reset((initial_thickness_nm * 1e-9) as f64);
            log::info!("Drainage simulation reset to {} nm", initial_thickness_nm);
        }
    }

    /// Get current drainage simulation time (if running).
    pub fn drainage_time(&self) -> Option<f64> {
        self.drainage_simulator.as_ref().map(|s| s.current_time())
    }

    /// Regenerate mesh with new subdivision level
    pub fn set_subdivision_level(&mut self, level: u32) {
        if level == self.subdivision_level || level > 5 {
            return; // No change or too high
        }
        self.subdivision_level = level;
        self.regenerate_mesh();
    }

    /// Set gravity deformation (aspect ratio)
    /// aspect_ratio: 1.0 = sphere, <1.0 = oblate (flattened at poles)
    pub fn set_deformation(&mut self, enabled: bool, aspect_ratio: f32) {
        let new_ratio = if enabled { aspect_ratio.clamp(0.7, 1.0) } else { 1.0 };
        if (self.aspect_ratio - new_ratio).abs() < 0.001 && self.deformation_enabled == enabled {
            return; // No significant change
        }
        self.deformation_enabled = enabled;
        self.aspect_ratio = new_ratio;

        // Update LOD cache with new aspect ratio (invalidates cached meshes)
        self.lod_cache.update(self.radius, new_ratio);

        // Regenerate current mesh
        self.regenerate_mesh();
    }

    /// Initialize the foam simulator for multi-bubble mode.
    pub fn init_foam_simulator(&mut self) {
        if self.foam_simulator.is_none() {
            let simulator = FoamSimulator::with_default_cluster();
            self.foam_simulator = Some(simulator);
            log::info!("Foam simulator initialized with default cluster");
        }
    }

    /// Enable or disable foam mode.
    pub fn set_foam_enabled(&mut self, enabled: bool) {
        self.foam_enabled = enabled;
        if enabled && self.foam_simulator.is_none() {
            self.init_foam_simulator();
        }
    }

    /// Add a bubble to the foam simulation.
    pub fn add_foam_bubble(&mut self, radius: f32) {
        if let Some(ref mut sim) = self.foam_simulator {
            sim.add_random_bubble((radius * 0.8, radius * 1.2));
        }
    }

    /// Reset the foam simulation.
    pub fn reset_foam(&mut self) {
        if let Some(ref mut sim) = self.foam_simulator {
            sim.reset();
        }
    }

    /// Get foam statistics (bubble count, connections).
    pub fn foam_stats(&self) -> (usize, usize) {
        if let Some(ref sim) = self.foam_simulator {
            (sim.bubble_count(), sim.connection_count())
        } else {
            (0, 0)
        }
    }

    /// Regenerate mesh with current parameters (subdivision level, aspect ratio)
    fn regenerate_mesh(&mut self) {
        // Use LOD cache for mesh generation
        let mesh = self.lod_cache.get_mesh(self.subdivision_level);

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

    /// Select appropriate LOD level based on camera distance
    fn select_lod_level(&self) -> u32 {
        let distance = self.camera.distance;

        if distance < self.lod_thresholds[0] {
            5 // Closest: highest detail
        } else if distance < self.lod_thresholds[1] {
            4
        } else if distance < self.lod_thresholds[2] {
            3
        } else if distance < self.lod_thresholds[3] {
            2
        } else {
            1 // Farthest: lowest detail
        }
    }

    /// Update LOD based on current camera distance (call each frame when LOD enabled)
    fn update_lod(&mut self) {
        if !self.lod_enabled {
            return;
        }

        let new_level = self.select_lod_level();
        if new_level != self.current_lod_level {
            self.switch_lod(new_level);
        }
    }

    /// Switch to a different LOD level
    fn switch_lod(&mut self, level: u32) {
        let level = level.clamp(1, 5);
        if level == self.current_lod_level {
            return;
        }

        self.current_lod_level = level;
        self.subdivision_level = level;

        // Get mesh from cache and create new GPU buffers
        let mesh = self.lod_cache.get_mesh(level);

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

        log::debug!(
            "LOD switched to level {} ({} triangles) at distance {:.3}m",
            level,
            self.num_indices / 3,
            self.camera.distance
        );
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        sample_count: u32,
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
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn create_msaa_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        sample_count: u32,
    ) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MSAA Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
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
            self.depth_texture = Self::create_depth_texture(&self.device, &self.config, self.msaa_samples);
            self.msaa_texture = Self::create_msaa_texture(&self.device, &self.config, self.msaa_samples);
            self.camera.set_aspect(new_size.width as f32 / new_size.height as f32);
        }
    }

    /// Set MSAA sample count (1 or 4)
    /// Recreates render pipeline and textures as needed
    pub fn set_msaa_samples(&mut self, samples: u32) {
        let samples = match samples {
            1 | 4 => samples,
            _ => 4, // Default to 4 for invalid values
        };

        if samples == self.msaa_samples {
            return; // No change needed
        }

        self.msaa_samples = samples;

        // Recreate textures with new sample count
        self.depth_texture = Self::create_depth_texture(&self.device, &self.config, samples);
        self.msaa_texture = Self::create_msaa_texture(&self.device, &self.config, samples);

        // Recreate render pipeline with new multisample state
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bubble Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bubble.wgsl").into()),
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&self.bind_group_layout],
            push_constant_ranges: &[],
        });

        self.render_pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                    format: self.config.format,
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
                count: samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        log::info!("MSAA changed to {}x", samples);
    }

    /// Handle window events for egui
    pub fn handle_event(&mut self, window: &winit::window::Window, event: &winit::event::WindowEvent) -> bool {
        let response = self.egui_state.on_window_event(window, event);
        response.consumed
    }

    /// Update time for animation
    pub fn update(&mut self, dt: f32) {
        self.bubble_uniform.time += dt;

        // LOD update based on camera distance
        self.update_lod();

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

        // Apply external forces (wind and buoyancy)
        if self.forces_enabled {
            // Wind force: F = wind_strength * direction
            let wind_force = [
                self.wind_strength * self.wind_direction[0],
                self.wind_strength * self.wind_direction[1],
                self.wind_strength * self.wind_direction[2],
            ];

            // Buoyancy force: light soap bubble rises (upward in +Y)
            let buoyancy_force = [0.0, self.buoyancy_strength, 0.0];

            // Simple drag to prevent runaway velocity (air resistance)
            let drag = 0.5;

            // Update velocity: v += (F - drag*v) * dt
            for i in 0..3 {
                let total_force = wind_force[i] + buoyancy_force[i] - drag * self.bubble_velocity[i];
                self.bubble_velocity[i] += total_force * dt;
            }

            // Update position: p += v * dt
            self.bubble_uniform.position_x += self.bubble_velocity[0] * dt;
            self.bubble_uniform.position_y += self.bubble_velocity[1] * dt;
            self.bubble_uniform.position_z += self.bubble_velocity[2] * dt;

            // Soft boundary: gradually push bubble back toward center if too far
            let max_distance = 0.15; // 15 cm from center
            let pos = [
                self.bubble_uniform.position_x,
                self.bubble_uniform.position_y,
                self.bubble_uniform.position_z,
            ];
            let dist_sq = pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2];
            if dist_sq > max_distance * max_distance {
                let dist = dist_sq.sqrt();
                let return_strength = 0.5 * (dist - max_distance);
                for i in 0..3 {
                    self.bubble_velocity[i] -= return_strength * pos[i] / dist * dt;
                }
            }
        }

        // Physics-based drainage simulation
        if self.physics_drainage_enabled {
            if let Some(ref mut simulator) = self.drainage_simulator {
                // Step the drainage simulation (with time scaling for visible effect)
                let scaled_dt = (dt * self.drainage_time_scale) as f64;
                simulator.step(scaled_dt);

                // Get thickness statistics from the simulation
                let field = simulator.thickness_field();
                let min_thickness = field.min_thickness() as f32 * 1e9;  // Convert to nm
                let max_thickness = field.max_thickness() as f32 * 1e9;

                // Sample thickness at equator (theta = PI/2) to get representative value
                let equator_thickness = simulator.get_thickness(
                    std::f64::consts::FRAC_PI_2,
                    0.0
                ) as f32 * 1e9;

                // Update the base thickness to reflect drainage
                // Use the equator thickness as it's a good representative
                self.bubble_uniform.base_thickness_nm = equator_thickness;

                // Adjust drainage_speed based on actual simulation progress
                // This affects the procedural overlay in the shader
                let drain_ratio = min_thickness / max_thickness;
                self.bubble_uniform.drainage_speed = (1.0 - drain_ratio).clamp(0.0, 2.0);

                // Check for burst condition
                if simulator.has_critical_region() {
                    log::info!("Bubble reached critical thickness - would burst at t={:.2}s",
                        simulator.current_time());
                }
            }
        }

        // Multi-bubble foam simulation
        if self.foam_enabled {
            if let Some(ref mut sim) = self.foam_simulator {
                let scaled_dt = dt * self.foam_time_scale;
                sim.step(scaled_dt);

                // Update foam renderer with current bubble positions
                self.foam_renderer.update_from_cluster(&sim.cluster);
                self.foam_renderer.upload(&self.queue);
            }
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

        // External forces extraction
        let mut forces_enabled = self.forces_enabled;
        let mut wind_strength = self.wind_strength;
        let mut buoyancy_strength = self.buoyancy_strength;
        let bubble_pos = [
            self.bubble_uniform.position_x,
            self.bubble_uniform.position_y,
            self.bubble_uniform.position_z,
        ];

        // Physics drainage extraction
        let mut physics_drainage_enabled = self.physics_drainage_enabled;
        let mut drainage_time_scale = self.drainage_time_scale;
        let drainage_sim_time = self.drainage_time().unwrap_or(0.0);
        let mut reset_drainage = false;
        let has_drainage_sim = self.drainage_simulator.is_some();

        // Deformation extraction
        let mut deformation_enabled = self.deformation_enabled;
        let mut aspect_ratio = self.aspect_ratio;

        // Edge smoothing extraction
        let mut edge_smoothing_mode = self.bubble_uniform.edge_smoothing_mode;

        // MSAA extraction
        let mut msaa_samples = self.msaa_samples;

        // LOD extraction
        let mut lod_enabled = self.lod_enabled;
        let current_lod_level = self.current_lod_level;

        // GPU drainage extraction
        let mut gpu_drainage_enabled = self.gpu_drainage_enabled;
        let mut gpu_drainage_time_scale = self.gpu_drainage.time_scale;
        let mut gpu_drainage_steps = self.gpu_drainage.steps_per_frame;
        let gpu_drainage_time = self.gpu_drainage.current_time();
        let mut reset_gpu_drainage = false;

        // Marangoni effect extraction
        let mut marangoni_enabled = self.gpu_drainage.marangoni_enabled;
        let mut marangoni_coeff = self.gpu_drainage.params().marangoni_coeff;

        // Foam system extraction
        let mut foam_enabled = self.foam_enabled;
        let mut foam_time_scale = self.foam_time_scale;
        let foam_stats = self.foam_stats();
        let mut add_bubble_requested = false;
        let mut reset_foam_requested = false;

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
                // External forces parameters
                &mut forces_enabled,
                &mut wind_strength,
                &mut buoyancy_strength,
                bubble_pos,
                // Physics drainage parameters
                &mut physics_drainage_enabled,
                &mut drainage_time_scale,
                drainage_sim_time,
                &mut reset_drainage,
                has_drainage_sim,
                // Deformation parameters
                &mut deformation_enabled,
                &mut aspect_ratio,
                // Edge smoothing parameter
                &mut edge_smoothing_mode,
                // MSAA parameter
                &mut msaa_samples,
                // LOD parameters
                &mut lod_enabled,
                current_lod_level,
                // GPU drainage parameters
                &mut gpu_drainage_enabled,
                &mut gpu_drainage_time_scale,
                &mut gpu_drainage_steps,
                gpu_drainage_time,
                &mut reset_gpu_drainage,
                // Marangoni parameters
                &mut marangoni_enabled,
                &mut marangoni_coeff,
                // Foam parameters
                &mut foam_enabled,
                &mut foam_time_scale,
                foam_stats,
                &mut add_bubble_requested,
                &mut reset_foam_requested,
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

        // Write back external forces state
        self.forces_enabled = forces_enabled;
        self.wind_strength = wind_strength;
        self.buoyancy_strength = buoyancy_strength;

        // Handle forces enable/disable - reset position when disabled
        if !forces_enabled {
            self.bubble_uniform.position_x = 0.0;
            self.bubble_uniform.position_y = 0.0;
            self.bubble_uniform.position_z = 0.0;
            self.bubble_velocity = [0.0, 0.0, 0.0];
        }

        // Write back physics drainage state
        self.drainage_time_scale = drainage_time_scale;

        // Handle physics drainage enable/disable
        if physics_drainage_enabled != self.physics_drainage_enabled {
            self.physics_drainage_enabled = physics_drainage_enabled;
            if physics_drainage_enabled && self.drainage_simulator.is_none() {
                // Initialize with default config if not already initialized
                let config = SimulationConfig::default();
                self.init_drainage_simulator(&config);
            }
        }

        // Handle drainage reset request
        if reset_drainage {
            self.reset_drainage(500.0);  // Reset to 500nm
        }

        // Handle deformation changes
        if deformation_enabled != self.deformation_enabled || (aspect_ratio - self.aspect_ratio).abs() > 0.001 {
            self.set_deformation(deformation_enabled, aspect_ratio);
        }

        // Write back edge smoothing mode
        self.bubble_uniform.edge_smoothing_mode = edge_smoothing_mode;

        // Handle MSAA changes
        if msaa_samples != self.msaa_samples {
            self.set_msaa_samples(msaa_samples);
        }

        // Handle LOD enable/disable
        self.lod_enabled = lod_enabled;

        // Handle GPU drainage changes
        self.gpu_drainage_enabled = gpu_drainage_enabled;
        self.gpu_drainage.enabled = gpu_drainage_enabled;
        self.gpu_drainage.time_scale = gpu_drainage_time_scale;
        self.gpu_drainage.steps_per_frame = gpu_drainage_steps;
        if reset_gpu_drainage {
            self.gpu_drainage.reset(&self.queue, 500e-9);  // Reset to 500nm
        }

        // Handle Marangoni changes
        if marangoni_enabled != self.gpu_drainage.marangoni_enabled {
            self.gpu_drainage.set_marangoni_enabled(&self.queue, marangoni_enabled);
        }
        if (marangoni_coeff - self.gpu_drainage.params().marangoni_coeff).abs() > 0.0001 {
            let params = self.gpu_drainage.params();
            self.gpu_drainage.set_marangoni_params(
                &self.queue,
                params.gamma_air,
                params.gamma_reduction,
                params.surfactant_diffusion,
                marangoni_coeff,
            );
        }

        // Apply foam parameter changes
        if foam_enabled != self.foam_enabled {
            self.set_foam_enabled(foam_enabled);
        }
        self.foam_time_scale = foam_time_scale;
        if add_bubble_requested {
            self.add_foam_bubble(self.radius * 0.8);
        }
        if reset_foam_requested {
            self.reset_foam();
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

        // GPU drainage simulation compute pass (before render pass)
        if self.gpu_drainage_enabled {
            // Get frame dt from fps tracking
            let dt = if self.fps > 0.0 { 1.0 / self.fps } else { 1.0 / 60.0 };
            self.gpu_drainage.step(&mut encoder, dt);
        }

        // Update egui buffers
        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &clipped_primitives,
            &screen_descriptor,
        );

        // Render bubble (with MSAA if enabled)
        {
            // When MSAA is enabled (samples > 1), render to msaa_texture and resolve to swap chain
            // When MSAA is disabled (samples = 1), render directly to swap chain
            let (color_view, resolve_target) = if self.msaa_samples > 1 {
                (&self.msaa_texture, Some(&view))
            } else {
                (&view, None)
            };

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target,
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

        // Render egui (2D overlay - no depth testing needed, renders to resolved swap chain)
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
                depth_stencil_attachment: None, // No depth needed for 2D UI overlay
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
        // External forces parameters
        forces_enabled: &mut bool,
        wind_strength: &mut f32,
        buoyancy_strength: &mut f32,
        bubble_pos: [f32; 3],
        // Physics drainage parameters
        physics_drainage_enabled: &mut bool,
        drainage_time_scale: &mut f32,
        drainage_sim_time: f64,
        reset_drainage: &mut bool,
        has_drainage_sim: bool,
        // Deformation parameters
        deformation_enabled: &mut bool,
        aspect_ratio: &mut f32,
        // Edge smoothing parameter
        edge_smoothing_mode: &mut u32,
        // MSAA parameter
        msaa_samples: &mut u32,
        // LOD parameters
        lod_enabled: &mut bool,
        current_lod_level: u32,
        // GPU drainage parameters
        gpu_drainage_enabled: &mut bool,
        gpu_drainage_time_scale: &mut f32,
        gpu_drainage_steps: &mut u32,
        gpu_drainage_time: f64,
        reset_gpu_drainage: &mut bool,
        // Marangoni parameters
        marangoni_enabled: &mut bool,
        marangoni_coeff: &mut f32,
        // Foam parameters
        foam_enabled: &mut bool,
        foam_time_scale: &mut f32,
        foam_stats: (usize, usize),
        add_bubble_requested: &mut bool,
        reset_foam_requested: &mut bool,
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

                ui.horizontal(|ui| {
                    ui.label("Edge blend:");
                    egui::ComboBox::from_id_salt("edge_blend")
                        .selected_text(match *edge_smoothing_mode {
                            1 => "Smoothstep",
                            2 => "Power",
                            _ => "Linear",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(edge_smoothing_mode, 0, "Linear");
                            ui.selectable_value(edge_smoothing_mode, 1, "Smoothstep");
                            ui.selectable_value(edge_smoothing_mode, 2, "Power");
                        });
                });

                ui.horizontal(|ui| {
                    ui.label("Anti-aliasing:");
                    egui::ComboBox::from_id_salt("msaa")
                        .selected_text(match *msaa_samples {
                            1 => "Off",
                            _ => "4x MSAA",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(msaa_samples, 1, "Off");
                            ui.selectable_value(msaa_samples, 4, "4x MSAA");
                        });
                });

                ui.separator();
                ui.heading("Mesh & Background");
                ui.separator();

                ui.checkbox(lod_enabled, "Auto LOD");

                if *lod_enabled {
                    // Show current LOD level when auto-LOD is enabled
                    ui.label(format!("LOD Level: {} ({} tri)", current_lod_level, num_triangles));
                } else {
                    // Manual subdivision control when LOD is disabled
                    ui.add(egui::Slider::new(subdivision, 1..=5)
                        .text("Mesh detail"));
                }

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

                ui.collapsing("External Forces", |ui| {
                    ui.checkbox(forces_enabled, "Enable forces");

                    if *forces_enabled {
                        ui.add(egui::Slider::new(wind_strength, 0.0..=0.5)
                            .text("Wind")
                            .suffix(" m/s²")
                            .fixed_decimals(2));

                        ui.add(egui::Slider::new(buoyancy_strength, 0.0..=0.1)
                            .text("Buoyancy")
                            .suffix(" m/s²")
                            .fixed_decimals(3));

                        ui.separator();
                        ui.label(format!("Position: ({:.3}, {:.3}, {:.3})",
                            bubble_pos[0], bubble_pos[1], bubble_pos[2]));
                    }
                });

                ui.collapsing("Physics Drainage (CPU)", |ui| {
                    ui.checkbox(physics_drainage_enabled, "Enable CPU simulation");

                    if *physics_drainage_enabled {
                        ui.add(egui::Slider::new(drainage_time_scale, 1.0..=500.0)
                            .text("Time scale")
                            .logarithmic(true)
                            .fixed_decimals(0));

                        ui.separator();
                        if has_drainage_sim {
                            ui.label(format!("Sim time: {:.2} s", drainage_sim_time));
                            ui.label(format!("Thickness: {:.0} nm", *thickness));

                            if ui.button("Reset").clicked() {
                                *reset_drainage = true;
                            }
                        } else {
                            ui.label("Initializing...");
                        }
                    }
                });

                ui.collapsing("GPU Drainage", |ui| {
                    ui.checkbox(gpu_drainage_enabled, "Enable GPU simulation");

                    if *gpu_drainage_enabled {
                        ui.add(egui::Slider::new(gpu_drainage_time_scale, 10.0..=500.0)
                            .text("Time scale")
                            .logarithmic(true)
                            .fixed_decimals(0));

                        ui.add(egui::Slider::new(gpu_drainage_steps, 1..=50)
                            .text("Steps/frame"));

                        ui.separator();
                        ui.checkbox(marangoni_enabled, "Marangoni effect");
                        if *marangoni_enabled {
                            ui.add(egui::Slider::new(marangoni_coeff, 0.001..=0.1)
                                .text("Strength")
                                .logarithmic(true)
                                .fixed_decimals(3));
                            ui.label("Surfactant-driven flow");
                        }

                        ui.separator();
                        ui.label(format!("Sim time: {:.2} s", gpu_drainage_time));
                        ui.label("Grid: 128×64 (8k cells)");

                        if ui.button("Reset").clicked() {
                            *reset_gpu_drainage = true;
                        }
                    }

                    ui.label("⚡ Real-time PDE solver");
                });

                ui.collapsing("Gravity Deformation", |ui| {
                    ui.checkbox(deformation_enabled, "Enable deformation");

                    if *deformation_enabled {
                        ui.add(egui::Slider::new(aspect_ratio, 0.7..=1.0)
                            .text("Aspect ratio")
                            .fixed_decimals(2));

                        ui.separator();
                        let deform_percent = (1.0 - *aspect_ratio) * 100.0;
                        ui.label(format!("Flattening: {:.1}%", deform_percent));
                        ui.label("(1.0 = sphere, <1.0 = oblate)");
                    }
                });

                ui.collapsing("Multi-Bubble Foam", |ui| {
                    ui.checkbox(foam_enabled, "Enable foam mode");

                    if *foam_enabled {
                        ui.add(egui::Slider::new(foam_time_scale, 0.1..=5.0)
                            .text("Time scale")
                            .fixed_decimals(1));

                        ui.separator();
                        ui.label(format!("Bubbles: {}", foam_stats.0));
                        ui.label(format!("Connections: {}", foam_stats.1));

                        ui.horizontal(|ui| {
                            if ui.button("Add Bubble").clicked() {
                                *add_bubble_requested = true;
                            }
                            if ui.button("Reset").clicked() {
                                *reset_foam_requested = true;
                            }
                        });

                        ui.separator();
                        ui.label("N-body bubble dynamics");
                        ui.label("with Van der Waals forces");
                    }
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
