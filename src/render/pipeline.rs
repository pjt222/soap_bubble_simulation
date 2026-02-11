//! wgpu render pipeline for soap bubble visualization

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use std::path::Path;

use crate::config::SimulationConfig;
use crate::physics::drainage::DrainageSimulator;
use crate::physics::geometry::{LodMeshCache, SpherePatch, Vertex};
use crate::physics::foam_dynamics::FoamSimulator;
use crate::render::camera::Camera;
use crate::render::gpu_drainage::GPUDrainageSimulator;
use crate::render::foam_renderer::{FoamRenderer, SharedWallRenderer, WallInstance, WallVertex};
use crate::render::branched_flow::{BranchedFlowSimulator, create_branched_flow_buffer};
use crate::render::interference_lut::{
    generate_interference_lut, LUT_THICKNESS_SAMPLES, LUT_ANGLE_SAMPLES,
};
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
    // Branched flow parameters (light focusing through film thickness variations)
    pub branched_flow_enabled: u32,
    pub branched_flow_intensity: f32,
    pub branched_flow_scale: f32,
    pub branched_flow_sharpness: f32,
    // Light direction for branched flow (normalized)
    pub light_dir_x: f32,
    pub light_dir_y: f32,
    pub light_dir_z: f32,
    // Patch view mode parameters
    pub patch_enabled: u32,
    pub patch_center_u: f32,
    pub patch_center_v: f32,
    pub patch_half_size: f32,
    // Padding for 16-byte alignment (28 actual fields + 4 padding = 32 fields = 128 bytes)
    pub _padding1: u32,
    pub _padding2: u32,
    pub _padding3: u32,
    pub _padding4: u32,
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
            // Branched flow (disabled by default)
            branched_flow_enabled: 0,
            branched_flow_intensity: 1.0,
            branched_flow_scale: 5.0,
            branched_flow_sharpness: 2.0,
            // Light direction (default: from upper-right, normalized)
            light_dir_x: 0.577,  // 1/sqrt(3)
            light_dir_y: 0.577,
            light_dir_z: 0.577,
            // Patch view mode (disabled by default)
            patch_enabled: 0,
            patch_center_u: 0.5,
            patch_center_v: 0.5,
            patch_half_size: 0.158,
            _padding1: 0,
            _padding2: 0,
            _padding3: 0,
            _padding4: 0,
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
    // Unit sphere mesh for foam instanced rendering (radius 1.0)
    foam_vertex_buffer: wgpu::Buffer,
    foam_index_buffer: wgpu::Buffer,
    foam_num_indices: u32,
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
    // FPS tracking (circular buffer for O(1) operations)
    frame_times: [f32; 60],
    frame_times_head: usize,
    frame_times_count: usize,
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
    pub foam_paused: bool,
    foam_time_scale: f32,
    // Foam generation parameters
    foam_generation_params: crate::physics::foam_generation::GenerationParams,
    // Instanced rendering pipeline for multi-bubble foam
    instanced_pipeline: wgpu::RenderPipeline,
    // Wall rendering for Plateau borders between bubbles
    wall_pipeline: wgpu::RenderPipeline,
    shared_wall_renderer: SharedWallRenderer,
    // Caustic / branched flow rendering
    caustic_renderer: crate::render::caustics::CausticRenderer,
    // Ray-traced branched flow simulation
    branched_flow_simulator: BranchedFlowSimulator,
    _branched_flow_buffer: wgpu::Buffer,
    // Interference color lookup table texture (pre-computed thin-film colors)
    interference_lut_texture: wgpu::Texture,
    interference_lut_sampler: wgpu::Sampler,
    // Track refractive index for LUT regeneration when it changes
    last_refractive_index: f32,
    // Patch view mode for focused branched flow viewing
    patch_view_enabled: bool,
    patch_center_u: f32,
    patch_center_v: f32,
    patch_half_size: f32,
    // Patch mesh buffers (separate from full sphere mesh)
    patch_vertex_buffer: wgpu::Buffer,
    patch_index_buffer: wgpu::Buffer,
    patch_num_indices: u32,
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
            alpha_mode: surface_caps
                .alpha_modes
                .iter()
                .copied()
                .find(|m| *m == wgpu::CompositeAlphaMode::Opaque)
                .unwrap_or(surface_caps.alpha_modes[0]),
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Default MSAA sample count
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

        // Create branched flow texture buffer (needed early for bind group)
        let branched_flow_buffer = create_branched_flow_buffer(&device);

        // Create interference LUT texture (pre-computed thin-film colors)
        let interference_lut_data = generate_interference_lut(
            bubble_uniform.refractive_index,
            1.0, // Intensity applied at runtime
        );
        let interference_lut_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Interference LUT Texture"),
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
            label: Some("Interference LUT Sampler"),
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
                // Branched flow texture (storage buffer, read-only in fragment shader)
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

        // Load instanced shader for multi-bubble foam rendering
        let instanced_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bubble Instanced Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bubble_instanced.wgsl").into()),
        });

        // Create instanced render pipeline with vertex + instance buffers
        let instanced_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Instanced Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &instanced_shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    Vertex::buffer_layout(),
                    crate::render::foam_renderer::BubbleInstance::buffer_layout(),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &instanced_shader,
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

        // Load wall shader for Plateau border rendering
        let wall_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Wall Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/wall.wgsl").into()),
        });

        // Create wall render pipeline (double-sided, no culling)
        let wall_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Wall Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &wall_shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    WallVertex::buffer_layout(),
                    WallInstance::buffer_layout(),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &wall_shader,
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
                cull_mode: None, // Double-sided rendering for walls
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

        // Initialize shared wall renderer
        let shared_wall_renderer = SharedWallRenderer::new(&device, 128);

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

        // Create unit sphere mesh for foam instanced rendering
        // Using radius 1.0 so the instance model matrix can scale to correct size
        use crate::physics::geometry::SphereMesh;
        let foam_mesh = SphereMesh::new(1.0, 3);

        let foam_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Foam Vertex Buffer"),
            contents: foam_mesh.vertex_bytes(),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let foam_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Foam Index Buffer"),
            contents: foam_mesh.index_bytes(),
            usage: wgpu::BufferUsages::INDEX,
        });

        let foam_num_indices = foam_mesh.indices.len() as u32;

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

        // Initialize caustic renderer
        let caustic_renderer = crate::render::caustics::CausticRenderer::new(
            &device,
            &camera_buffer,
            &bind_group_layout,
            gpu_drainage.current_thickness_buffer(),
            surface_format,
            wgpu::TextureFormat::Depth32Float,
            msaa_samples,
        );

        // Initialize branched flow simulator (ray-traced light propagation)
        let branched_flow_simulator = BranchedFlowSimulator::new(
            &device,
            gpu_drainage.current_thickness_buffer(),
            &branched_flow_buffer,
        );

        // Create patch mesh for focused branched flow viewing
        let patch_center_u = 0.5;
        let patch_center_v = 0.5;
        let patch_half_size = 0.158; // ~10% of sphere surface
        let patch = SpherePatch::new(patch_center_u, patch_center_v, patch_half_size, 32);
        let (patch_vertices, patch_indices) = patch.generate_mesh_indexed(radius, 1.0);

        let patch_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Patch Vertex Buffer"),
            contents: bytemuck::cast_slice(&patch_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let patch_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Patch Index Buffer"),
            contents: bytemuck::cast_slice(&patch_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let patch_num_indices = patch_indices.len() as u32;

        Self {
            surface,
            device,
            queue,
            config,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            foam_vertex_buffer,
            foam_index_buffer,
            foam_num_indices,
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
            frame_times: [0.0; 60],
            frame_times_head: 0,
            frame_times_count: 0,
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
            // Foam system (disabled by default, paused by default)
            foam_simulator: None,
            foam_renderer,
            foam_enabled: false,
            foam_paused: true,
            foam_time_scale: 1.0,
            foam_generation_params: crate::physics::foam_generation::GenerationParams::default(),
            instanced_pipeline,
            wall_pipeline,
            shared_wall_renderer,
            caustic_renderer,
            branched_flow_simulator,
            _branched_flow_buffer: branched_flow_buffer,
            interference_lut_texture,
            interference_lut_sampler,
            last_refractive_index: bubble_uniform.refractive_index,
            // Patch view mode (disabled by default)
            patch_view_enabled: false,
            patch_center_u,
            patch_center_v,
            patch_half_size,
            patch_vertex_buffer,
            patch_index_buffer,
            patch_num_indices,
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

    /// Regenerate the interference LUT texture when refractive index changes.
    /// This ensures the pre-computed thin-film colors match the current physics.
    fn regenerate_interference_lut_if_needed(&mut self) {
        let current_n = self.bubble_uniform.refractive_index;
        if (current_n - self.last_refractive_index).abs() < 1e-6 {
            return; // No significant change
        }

        // Regenerate LUT data with new refractive index
        let lut_data = generate_interference_lut(current_n, 1.0);

        // Upload new data to existing texture
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.interference_lut_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &lut_data,
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

        self.last_refractive_index = current_n;
        log::debug!("Regenerated interference LUT for n={:.3}", current_n);
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
            use crate::physics::foam_generation::FoamGenerator;
            let generator = FoamGenerator::new(self.foam_generation_params.clone());
            let cluster = generator.generate(0.025);
            let mut simulator = FoamSimulator::new(cluster);
            simulator.cluster.update_connections();
            self.foam_simulator = Some(simulator);
            log::info!(
                "Foam simulator initialized with {:?} positioning, {:?} sizes",
                self.foam_generation_params.positioning_mode,
                self.foam_generation_params.size_distribution
            );
        }
    }

    /// Enable or disable foam mode.
    pub fn set_foam_enabled(&mut self, enabled: bool) {
        log::info!("set_foam_enabled({})", enabled);
        self.foam_enabled = enabled;
        if enabled && self.foam_simulator.is_none() {
            self.init_foam_simulator();
        }
    }

    /// Add a bubble to the foam simulation.
    pub fn add_foam_bubble(&mut self, radius: f32) {
        if let Some(ref mut sim) = self.foam_simulator {
            let before = sim.bubble_count();
            sim.add_random_bubble((radius * 0.8, radius * 1.2));
            log::info!("Added bubble: {} -> {} bubbles", before, sim.bubble_count());
        } else {
            log::warn!("add_foam_bubble called but foam_simulator is None");
        }
    }

    /// Reset the foam simulation.
    pub fn reset_foam(&mut self) {
        if let Some(ref mut sim) = self.foam_simulator {
            sim.reset();
        }
    }

    /// Regenerate foam with current generation parameters.
    pub fn regenerate_foam(&mut self) {
        if let Some(ref mut sim) = self.foam_simulator {
            sim.reset_with_params(&self.foam_generation_params);
            log::info!(
                "Regenerated foam: {} bubbles with {:?} positioning, {:?} sizes",
                sim.bubble_count(),
                self.foam_generation_params.positioning_mode,
                self.foam_generation_params.size_distribution
            );
        } else {
            // Initialize with generation parameters
            use crate::physics::foam_generation::FoamGenerator;
            let generator = FoamGenerator::new(self.foam_generation_params.clone());
            let cluster = generator.generate(0.025);
            let mut simulator = FoamSimulator::new(cluster);
            simulator.cluster.update_connections();
            self.foam_simulator = Some(simulator);
            log::info!("Foam simulator created with generation parameters");
        }
    }

    /// Get foam statistics (bubble count, connections, walls).
    pub fn foam_stats(&self) -> (usize, usize, usize) {
        if let Some(ref sim) = self.foam_simulator {
            (sim.bubble_count(), sim.connection_count(), self.shared_wall_renderer.instance_count() as usize)
        } else {
            (0, 0, 0)
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

    /// Regenerate patch mesh when patch parameters change
    fn regenerate_patch_mesh(&mut self) {
        let patch = SpherePatch::new(
            self.patch_center_u,
            self.patch_center_v,
            self.patch_half_size,
            32,
        );
        let (patch_vertices, patch_indices) = patch.generate_mesh_indexed(self.radius, self.aspect_ratio);

        self.patch_vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Patch Vertex Buffer"),
            contents: bytemuck::cast_slice(&patch_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        self.patch_index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Patch Index Buffer"),
            contents: bytemuck::cast_slice(&patch_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        self.patch_num_indices = patch_indices.len() as u32;

        log::debug!(
            "Regenerated patch mesh: center=({:.2}, {:.2}), size={:.3}, {} triangles",
            self.patch_center_u,
            self.patch_center_v,
            self.patch_half_size,
            self.patch_num_indices / 3
        );
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
        if new_size.width > 0 && new_size.height > 0 &&
           (new_size.width != self.config.width || new_size.height != self.config.height) {
            // Wait for GPU to finish any pending work before recreating resources
            self.device.poll(wgpu::Maintain::Wait);

            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture = Self::create_depth_texture(&self.device, &self.config, self.msaa_samples);
            self.msaa_texture = Self::create_msaa_texture(&self.device, &self.config, self.msaa_samples);
            self.camera.set_aspect(new_size.width as f32 / new_size.height as f32);
        }
    }

    /// Set MSAA sample count (1, 2, or 4)
    /// Recreates render pipeline and textures as needed
    pub fn set_msaa_samples(&mut self, samples: u32) {
        let samples = match samples {
            1 | 2 | 4 => samples,
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
                for (velocity, &position) in self.bubble_velocity.iter_mut().zip(pos.iter()) {
                    *velocity -= return_strength * position / dist * dt;
                }
            }
        }

        // Physics-based drainage simulation
        if self.physics_drainage_enabled
            && let Some(ref mut simulator) = self.drainage_simulator
        {
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

        // Multi-bubble foam simulation
        if self.foam_enabled
            && let Some(ref mut sim) = self.foam_simulator
        {
            // Only step physics when not paused
            if !self.foam_paused {
                let scaled_dt = dt * self.foam_time_scale;
                sim.step(scaled_dt);
            }

            // Always update renderer to show current state
            self.foam_renderer.update_from_cluster(&sim.cluster);
            self.foam_renderer.upload(&self.queue);

            // Generate and upload shared wall (Plateau border) instances
            self.shared_wall_renderer.generate_walls(&sim.cluster);
            self.shared_wall_renderer.upload(&self.queue);
        }

        // Track FPS using circular buffer (O(1) operations)
        self.frame_times[self.frame_times_head] = dt;
        self.frame_times_head = (self.frame_times_head + 1) % 60;
        if self.frame_times_count < 60 {
            self.frame_times_count += 1;
        }
        if self.frame_times_count > 0 {
            let sum: f32 = self.frame_times[..self.frame_times_count].iter().sum();
            let avg_dt = sum / self.frame_times_count as f32;
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

        // Caustic parameters extraction
        let mut caustics_enabled = self.caustic_renderer.enabled;
        let mut caustic_intensity = self.caustic_renderer.params.caustic_intensity;
        let mut caustic_sharpness = self.caustic_renderer.params.caustic_sharpness;
        let mut ground_y = self.caustic_renderer.params.ground_y;

        // Branched flow parameters extraction (ray-traced laser through film)
        let mut branched_flow_enabled = self.branched_flow_simulator.enabled;
        let mut branched_flow_intensity = self.bubble_uniform.branched_flow_intensity;
        let mut branched_flow_sharpness = self.bubble_uniform.branched_flow_sharpness;
        // Laser entry point (spherical coordinates)
        let entry = self.branched_flow_simulator.params.entry_point;
        let mut laser_azimuth = entry[2].atan2(entry[0]).to_degrees();
        let mut laser_elevation = entry[1].asin().to_degrees();
        // Beam parameters
        let mut beam_spread = self.branched_flow_simulator.params.spread_angle.to_degrees();
        let mut bend_strength = self.branched_flow_simulator.params.bend_strength;
        let mut num_rays = self.branched_flow_simulator.params.num_rays;
        // Particle scattering parameters
        let mut num_scatterers = self.branched_flow_simulator.params.num_scatterers;
        let mut scatterer_strength = self.branched_flow_simulator.params.scatterer_strength;
        let mut scatterer_radius = self.branched_flow_simulator.params.scatterer_radius;
        let mut particle_weight = self.branched_flow_simulator.params.particle_weight;
        // Patch view mode parameters
        let mut patch_view_enabled = self.patch_view_enabled;
        let mut patch_center_u = self.patch_center_u;
        let mut patch_center_v = self.patch_center_v;
        let mut patch_half_size = self.patch_half_size;

        // Foam system extraction
        let mut foam_enabled = self.foam_enabled;
        let mut foam_paused = self.foam_paused;
        let mut foam_time_scale = self.foam_time_scale;
        let foam_stats = self.foam_stats();  // (bubbles, connections, walls)
        let mut add_bubble_requested = false;
        let mut reset_foam_requested = false;
        let mut foam_gen_params = self.foam_generation_params.clone();
        let mut regenerate_foam_requested = false;

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
                // Caustic parameters
                &mut caustics_enabled,
                &mut caustic_intensity,
                &mut caustic_sharpness,
                &mut ground_y,
                // Branched flow parameters (ray-traced laser)
                &mut branched_flow_enabled,
                &mut branched_flow_intensity,
                &mut branched_flow_sharpness,
                &mut laser_azimuth,
                &mut laser_elevation,
                &mut beam_spread,
                &mut bend_strength,
                &mut num_rays,
                // Particle scattering parameters
                &mut num_scatterers,
                &mut scatterer_strength,
                &mut scatterer_radius,
                &mut particle_weight,
                // Patch view mode parameters
                &mut patch_view_enabled,
                &mut patch_center_u,
                &mut patch_center_v,
                &mut patch_half_size,
                // Foam parameters
                &mut foam_enabled,
                &mut foam_paused,
                &mut foam_time_scale,
                foam_stats,
                &mut add_bubble_requested,
                &mut reset_foam_requested,
                // Foam generation parameters
                &mut foam_gen_params,
                &mut regenerate_foam_requested,
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

        // Regenerate interference LUT if refractive index changed
        self.regenerate_interference_lut_if_needed();

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

        // Handle caustic parameter changes
        self.caustic_renderer.enabled = caustics_enabled;
        let caustic_params_changed =
            (caustic_intensity - self.caustic_renderer.params.caustic_intensity).abs() > 0.001
            || (caustic_sharpness - self.caustic_renderer.params.caustic_sharpness).abs() > 0.001
            || (ground_y - self.caustic_renderer.params.ground_y).abs() > 0.001;
        if caustic_params_changed {
            self.caustic_renderer.params.caustic_intensity = caustic_intensity;
            self.caustic_renderer.params.caustic_sharpness = caustic_sharpness;
            if (ground_y - self.caustic_renderer.params.ground_y).abs() > 0.001 {
                self.caustic_renderer.set_ground_y(&self.device, ground_y);
            }
            self.caustic_renderer.update_params(&self.queue);
        }

        // Handle branched flow parameter changes (ray-traced laser)
        self.branched_flow_simulator.enabled = branched_flow_enabled;
        self.bubble_uniform.branched_flow_enabled = if branched_flow_enabled { 1 } else { 0 };
        self.bubble_uniform.branched_flow_intensity = branched_flow_intensity;
        self.bubble_uniform.branched_flow_sharpness = branched_flow_sharpness;
        // Update laser entry point
        self.branched_flow_simulator.set_entry_point(laser_azimuth, laser_elevation);
        // Update beam parameters
        self.branched_flow_simulator.params.spread_angle = beam_spread.to_radians();
        self.branched_flow_simulator.params.bend_strength = bend_strength;
        self.branched_flow_simulator.params.num_rays = num_rays;
        // Update particle scattering parameters
        self.branched_flow_simulator.params.num_scatterers = num_scatterers;
        self.branched_flow_simulator.params.scatterer_strength = scatterer_strength;
        self.branched_flow_simulator.params.scatterer_radius = scatterer_radius;
        self.branched_flow_simulator.params.particle_weight = particle_weight;
        // Sync film dynamics so branched flow rays bend through the same dynamic landscape
        self.branched_flow_simulator.params.base_thickness_nm = self.bubble_uniform.base_thickness_nm;
        self.branched_flow_simulator.params.swirl_intensity = self.bubble_uniform.swirl_intensity;
        self.branched_flow_simulator.params.drainage_speed = self.bubble_uniform.drainage_speed;
        self.branched_flow_simulator.params.pattern_scale = self.bubble_uniform.pattern_scale;

        // Handle patch view mode changes
        self.patch_view_enabled = patch_view_enabled;
        self.branched_flow_simulator.params.patch_enabled = if patch_view_enabled { 1 } else { 0 };
        // Check if patch parameters changed (regenerate mesh if so)
        let patch_params_changed =
            (patch_center_u - self.patch_center_u).abs() > 0.001
            || (patch_center_v - self.patch_center_v).abs() > 0.001
            || (patch_half_size - self.patch_half_size).abs() > 0.001;
        if patch_params_changed {
            self.patch_center_u = patch_center_u;
            self.patch_center_v = patch_center_v;
            self.patch_half_size = patch_half_size;
            self.regenerate_patch_mesh();
        }
        // Sync patch bounds to branched flow simulator
        self.branched_flow_simulator.params.patch_center_u = self.patch_center_u;
        self.branched_flow_simulator.params.patch_center_v = self.patch_center_v;
        self.branched_flow_simulator.params.patch_half_size = self.patch_half_size;
        // Sync patch bounds to bubble uniform (for fragment shader)
        self.bubble_uniform.patch_enabled = if patch_view_enabled { 1 } else { 0 };
        self.bubble_uniform.patch_center_u = self.patch_center_u;
        self.bubble_uniform.patch_center_v = self.patch_center_v;
        self.bubble_uniform.patch_half_size = self.patch_half_size;

        // Apply foam parameter changes
        if foam_enabled != self.foam_enabled {
            self.set_foam_enabled(foam_enabled);
        }
        self.foam_paused = foam_paused;
        self.foam_time_scale = foam_time_scale;
        // Write back foam generation parameters (may have been modified by UI)
        self.foam_generation_params = foam_gen_params;
        if add_bubble_requested {
            self.add_foam_bubble(self.radius * 0.8);
        }
        if reset_foam_requested {
            self.reset_foam();
        }
        if regenerate_foam_requested {
            self.regenerate_foam();
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

        // Caustic compute pass (after drainage, before render)
        if self.caustic_renderer.enabled && self.gpu_drainage_enabled {
            self.caustic_renderer.compute(&mut encoder);
        }

        // Branched flow compute pass (ray tracing through film)
        if self.branched_flow_simulator.enabled && self.gpu_drainage_enabled {
            // Update scatterer positions (creates animated particle distribution)
            self.branched_flow_simulator.update_scatterers(&self.queue, self.bubble_uniform.film_time);
            self.branched_flow_simulator.step(&mut encoder, self.bubble_uniform.film_time);
            self.branched_flow_simulator.update_params(&self.queue);
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

            // Use instanced pipeline for multi-bubble foam, regular pipeline for single bubble
            if self.foam_enabled && !self.foam_renderer.is_empty() {
                // Render bubbles using unit sphere mesh (radius 1.0)
                // Instance model matrix scales to correct bubble size
                render_pass.set_pipeline(&self.instanced_pipeline);
                render_pass.set_bind_group(0, &self.bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.foam_vertex_buffer.slice(..));
                render_pass.set_vertex_buffer(1, self.foam_renderer.instance_buffer().slice(..));
                render_pass.set_index_buffer(self.foam_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..self.foam_num_indices, 0, 0..self.foam_renderer.instance_count());

                // Render shared walls (Plateau borders) between touching bubbles
                if self.shared_wall_renderer.has_walls() {
                    render_pass.set_pipeline(&self.wall_pipeline);
                    render_pass.set_bind_group(0, &self.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, self.shared_wall_renderer.vertex_buffer().slice(..));
                    render_pass.set_vertex_buffer(1, self.shared_wall_renderer.instance_buffer().slice(..));
                    render_pass.set_index_buffer(self.shared_wall_renderer.index_buffer().slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(
                        0..self.shared_wall_renderer.num_mesh_indices(),
                        0,
                        0..self.shared_wall_renderer.instance_count()
                    );
                }
            } else if self.patch_view_enabled && self.branched_flow_simulator.enabled {
                // Patch view mode: render only the focused patch mesh
                render_pass.set_pipeline(&self.render_pipeline);
                render_pass.set_bind_group(0, &self.bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.patch_vertex_buffer.slice(..));
                render_pass.set_index_buffer(self.patch_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..self.patch_num_indices, 0, 0..1);
            } else {
                // Full sphere view
                render_pass.set_pipeline(&self.render_pipeline);
                render_pass.set_bind_group(0, &self.bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
            }

            // Render caustics on ground plane (after bubble, uses additive blending)
            if self.caustic_renderer.enabled && self.gpu_drainage_enabled {
                self.caustic_renderer.render(&mut render_pass);
            }
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
            let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
            let buffer_size = (padded_bytes_per_row * self.config.height) as u64;

            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Screenshot Staging Buffer"),
                size: buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            // Copy texture to buffer
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &output.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
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
        // Caustic parameters
        caustics_enabled: &mut bool,
        caustic_intensity: &mut f32,
        caustic_sharpness: &mut f32,
        ground_y: &mut f32,
        // Branched flow parameters (ray-traced laser)
        branched_flow_enabled: &mut bool,
        branched_flow_intensity: &mut f32,
        branched_flow_sharpness: &mut f32,
        laser_azimuth: &mut f32,
        laser_elevation: &mut f32,
        beam_spread: &mut f32,
        bend_strength: &mut f32,
        num_rays: &mut u32,
        // Particle scattering parameters
        num_scatterers: &mut u32,
        scatterer_strength: &mut f32,
        scatterer_radius: &mut f32,
        particle_weight: &mut f32,
        // Patch view mode parameters
        patch_view_enabled: &mut bool,
        patch_center_u: &mut f32,
        patch_center_v: &mut f32,
        patch_half_size: &mut f32,
        // Foam parameters
        foam_enabled: &mut bool,
        foam_paused: &mut bool,
        foam_time_scale: &mut f32,
        foam_stats: (usize, usize, usize),  // (bubbles, connections, walls)
        add_bubble_requested: &mut bool,
        reset_foam_requested: &mut bool,
        // Foam generation parameters
        foam_gen_params: &mut crate::physics::foam_generation::GenerationParams,
        regenerate_foam_requested: &mut bool,
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
                            2 => "2x MSAA",
                            _ => "4x MSAA",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(msaa_samples, 1, "Off");
                            ui.selectable_value(msaa_samples, 2, "2x MSAA");
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

                ui.collapsing("Caustics / Branched Flow", |ui| {
                    // Ground-plane caustics (projected below bubble)
                    ui.label("Ground Caustics");
                    ui.checkbox(caustics_enabled, "Enable ground caustics");

                    if *caustics_enabled {
                        if !*gpu_drainage_enabled {
                            ui.colored_label(egui::Color32::YELLOW, "⚠ Requires GPU Drainage");
                        }

                        ui.add(egui::Slider::new(caustic_intensity, 0.5..=5.0)
                            .text("Intensity")
                            .fixed_decimals(1));

                        ui.add(egui::Slider::new(caustic_sharpness, 1.0..=3.0)
                            .text("Sharpness")
                            .fixed_decimals(1));

                        ui.add(egui::Slider::new(ground_y, -0.15..=-0.05)
                            .text("Ground height")
                            .suffix(" m")
                            .fixed_decimals(2));
                    }

                    ui.separator();

                    // Ray-traced branched flow (laser propagating WITHIN film)
                    ui.label("In-Film Laser");
                    ui.checkbox(branched_flow_enabled, "Enable laser in film");

                    if *branched_flow_enabled {
                        if !*gpu_drainage_enabled {
                            ui.colored_label(egui::Color32::YELLOW, "⚠ Requires GPU Drainage");
                        }

                        ui.label("Injection Point");
                        ui.add(egui::Slider::new(laser_azimuth, -180.0..=180.0)
                            .text("Azimuth")
                            .suffix("°")
                            .fixed_decimals(0));

                        ui.add(egui::Slider::new(laser_elevation, -90.0..=90.0)
                            .text("Elevation")
                            .suffix("°")
                            .fixed_decimals(0));

                        ui.separator();
                        ui.label("Beam Properties");

                        ui.add(egui::Slider::new(beam_spread, 1.0..=45.0)
                            .text("Spread")
                            .suffix("°")
                            .fixed_decimals(0));

                        ui.add(egui::Slider::new(bend_strength, 0.01..=50.0)
                            .text("GRIN bending")
                            .logarithmic(true)
                            .fixed_decimals(3));

                        ui.add(egui::Slider::new(num_rays, 256..=65536)
                            .text("Ray count")
                            .logarithmic(true));

                        ui.separator();
                        ui.label("Display");

                        ui.add(egui::Slider::new(branched_flow_intensity, 0.1..=20.0)
                            .text("Brightness")
                            .fixed_decimals(2));

                        ui.add(egui::Slider::new(branched_flow_sharpness, 0.5..=3.0)
                            .text("Contrast")
                            .fixed_decimals(1));

                        ui.separator();
                        ui.label("Particle Scattering");

                        ui.add(egui::Slider::new(particle_weight, 0.0..=1.0)
                            .text("Weight")
                            .fixed_decimals(2));

                        ui.add(egui::Slider::new(num_scatterers, 100..=2000)
                            .text("Scatterers")
                            .logarithmic(true));

                        ui.add(egui::Slider::new(scatterer_strength, 0.1..=2.0)
                            .text("Strength")
                            .logarithmic(true)
                            .fixed_decimals(2));

                        ui.add(egui::Slider::new(scatterer_radius, 0.01..=0.1)
                            .text("Radius")
                            .fixed_decimals(3));

                        ui.separator();
                        ui.label("Hybrid model: GRIN + particles");
                        ui.label("Weight 0 = smooth GRIN only");
                        ui.label("Weight 1 = particle scatter only");

                        ui.separator();
                        ui.label("Patch View Mode");
                        ui.checkbox(patch_view_enabled, "Focus on patch");

                        if *patch_view_enabled {
                            ui.add(egui::Slider::new(patch_center_u, 0.1..=0.9)
                                .text("Center U")
                                .fixed_decimals(2));

                            ui.add(egui::Slider::new(patch_center_v, 0.1..=0.9)
                                .text("Center V")
                                .fixed_decimals(2));

                            ui.add(egui::Slider::new(patch_half_size, 0.05..=0.3)
                                .text("Patch size")
                                .fixed_decimals(3));

                            let area_percent = (*patch_half_size * 2.0).powi(2) * 100.0;
                            ui.label(format!("~{:.1}% of sphere", area_percent));
                        }
                    }
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
                        ui.horizontal(|ui| {
                            let pause_text = if *foam_paused { "\u{25B6} Start" } else { "\u{23F8} Pause" };
                            if ui.button(pause_text).clicked() {
                                *foam_paused = !*foam_paused;
                            }
                        });

                        ui.add(egui::Slider::new(foam_time_scale, 0.1..=5.0)
                            .text("Time scale")
                            .fixed_decimals(1));

                        ui.separator();
                        ui.label(format!("Bubbles: {}", foam_stats.0));
                        ui.label(format!("Connections: {}", foam_stats.1));
                        ui.label(format!("Walls: {}", foam_stats.2));

                        ui.horizontal(|ui| {
                            if ui.button("Add Bubble").clicked() {
                                *add_bubble_requested = true;
                            }
                            if ui.button("Reset").clicked() {
                                *reset_foam_requested = true;
                            }
                        });

                        ui.separator();
                        ui.heading("Generation");

                        // Bubble count
                        let mut bubble_count = foam_gen_params.bubble_count as i32;
                        if ui.add(egui::Slider::new(&mut bubble_count, 2..=30)
                            .text("Bubble count")).changed() {
                            foam_gen_params.bubble_count = bubble_count as u32;
                        }

                        // Positioning mode dropdown
                        ui.horizontal(|ui| {
                            ui.label("Positioning:");
                            egui::ComboBox::from_id_salt("positioning_mode")
                                .selected_text(foam_gen_params.positioning_mode.name())
                                .show_ui(ui, |ui| {
                                    use crate::physics::foam_generation::PositioningMode;
                                    for mode in PositioningMode::all() {
                                        ui.selectable_value(
                                            &mut foam_gen_params.positioning_mode,
                                            *mode,
                                            mode.name()
                                        );
                                    }
                                });
                        });

                        // Show spacing/jitter for grid modes
                        use crate::physics::foam_generation::PositioningMode;
                        let is_grid_mode = matches!(
                            foam_gen_params.positioning_mode,
                            PositioningMode::SimpleCubic |
                            PositioningMode::BodyCenteredCubic |
                            PositioningMode::FaceCenteredCubic |
                            PositioningMode::HexagonalClosePacked |
                            PositioningMode::PoissonDisk
                        );

                        if is_grid_mode {
                            ui.add(egui::Slider::new(&mut foam_gen_params.spacing, 0.03..=0.10)
                                .text("Spacing")
                                .suffix(" m")
                                .fixed_decimals(3));

                            ui.add(egui::Slider::new(&mut foam_gen_params.jitter, 0.0..=0.5)
                                .text("Jitter")
                                .fixed_decimals(2));
                        }

                        ui.separator();

                        // Size distribution dropdown
                        ui.horizontal(|ui| {
                            ui.label("Size dist:");
                            egui::ComboBox::from_id_salt("size_distribution")
                                .selected_text(foam_gen_params.size_distribution.name())
                                .show_ui(ui, |ui| {
                                    use crate::physics::foam_generation::SizeDistribution;
                                    for dist in SizeDistribution::all() {
                                        ui.selectable_value(
                                            &mut foam_gen_params.size_distribution,
                                            *dist,
                                            dist.name()
                                        );
                                    }
                                });
                        });

                        // Radius range (always shown)
                        ui.add(egui::Slider::new(&mut foam_gen_params.min_radius, 0.005..=0.03)
                            .text("Min radius")
                            .suffix(" m")
                            .fixed_decimals(3));

                        ui.add(egui::Slider::new(&mut foam_gen_params.max_radius, 0.02..=0.06)
                            .text("Max radius")
                            .suffix(" m")
                            .fixed_decimals(3));

                        // Context-sensitive sliders based on distribution type
                        use crate::physics::foam_generation::SizeDistribution;
                        match foam_gen_params.size_distribution {
                            SizeDistribution::Normal | SizeDistribution::LogNormal => {
                                ui.add(egui::Slider::new(&mut foam_gen_params.mean_radius, 0.01..=0.04)
                                    .text("Mean radius")
                                    .suffix(" m")
                                    .fixed_decimals(3));

                                if foam_gen_params.size_distribution == SizeDistribution::Normal {
                                    ui.add(egui::Slider::new(&mut foam_gen_params.std_dev, 0.001..=0.015)
                                        .text("Std dev")
                                        .suffix(" m")
                                        .fixed_decimals(3));
                                } else {
                                    ui.add(egui::Slider::new(&mut foam_gen_params.sigma, 0.1..=0.8)
                                        .text("Sigma")
                                        .fixed_decimals(2));
                                }
                            }
                            SizeDistribution::SchulzFlory => {
                                ui.add(egui::Slider::new(&mut foam_gen_params.mean_radius, 0.01..=0.04)
                                    .text("Mean radius")
                                    .suffix(" m")
                                    .fixed_decimals(3));

                                ui.add(egui::Slider::new(&mut foam_gen_params.pdi, 1.1..=3.0)
                                    .text("PDI (Mw/Mn)")
                                    .fixed_decimals(2));
                            }
                            SizeDistribution::Bimodal => {
                                ui.add(egui::Slider::new(&mut foam_gen_params.mean_radius, 0.01..=0.03)
                                    .text("Mean 1")
                                    .suffix(" m")
                                    .fixed_decimals(3));

                                ui.add(egui::Slider::new(&mut foam_gen_params.std_dev, 0.001..=0.01)
                                    .text("Std 1")
                                    .suffix(" m")
                                    .fixed_decimals(3));

                                ui.add(egui::Slider::new(&mut foam_gen_params.bimodal_ratio, 0.1..=0.9)
                                    .text("Ratio")
                                    .fixed_decimals(2));

                                ui.add(egui::Slider::new(&mut foam_gen_params.bimodal_mean2, 0.02..=0.05)
                                    .text("Mean 2")
                                    .suffix(" m")
                                    .fixed_decimals(3));

                                ui.add(egui::Slider::new(&mut foam_gen_params.bimodal_std2, 0.001..=0.01)
                                    .text("Std 2")
                                    .suffix(" m")
                                    .fixed_decimals(3));
                            }
                            SizeDistribution::Uniform => {
                                // Uniform just uses min/max, already shown above
                            }
                        }

                        ui.separator();

                        if ui.button("Regenerate Foam").clicked() {
                            *regenerate_foam_requested = true;
                        }

                        ui.separator();
                        ui.label("N-body bubble dynamics");
                        ui.label("with Plateau borders");
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
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
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
            wgpu::TexelCopyTextureInfo {
                texture: &output.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bubble_uniform_default_values() {
        let uniform = BubbleUniform::default();

        // Visual properties
        assert!((uniform.refractive_index - 1.33).abs() < 1e-6, "refractive_index");
        assert!((uniform.base_thickness_nm - 500.0).abs() < 1e-6, "base_thickness_nm");
        assert!((uniform.interference_intensity - 4.0).abs() < 1e-6, "interference_intensity");
        assert!((uniform.base_alpha - 0.3).abs() < 1e-6, "base_alpha");
        assert!((uniform.edge_alpha - 0.6).abs() < 1e-6, "edge_alpha");

        // Film dynamics - these are critical for animation
        assert!((uniform.film_time - 0.0).abs() < 1e-6, "film_time should start at 0");
        assert!((uniform.swirl_intensity - 1.0).abs() < 1e-6, "swirl_intensity");
        assert!((uniform.drainage_speed - 0.5).abs() < 1e-6, "drainage_speed");
        assert!((uniform.pattern_scale - 1.0).abs() < 1e-6, "pattern_scale");
    }

    #[test]
    fn test_bubble_uniform_film_dynamics_present() {
        // Verify all film dynamics fields exist and are accessible
        let mut uniform = BubbleUniform::default();

        // These should compile and be modifiable
        uniform.film_time = 10.0;
        uniform.swirl_intensity = 2.0;
        uniform.drainage_speed = 0.5;
        uniform.pattern_scale = 3.0;

        assert!((uniform.film_time - 10.0).abs() < 1e-6);
        assert!((uniform.swirl_intensity - 2.0).abs() < 1e-6);
        assert!((uniform.drainage_speed - 0.5).abs() < 1e-6);
        assert!((uniform.pattern_scale - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_bubble_uniform_size_alignment() {
        // Verify struct is properly aligned for GPU
        // Total size: 9 visual + 4 film + 3 position + 1 edge_mode + 1 bf_enabled
        //   + 3 bf_params + 3 light_dir + 4 patch_params + 3 padding = 32 values * 4 bytes = 128 bytes
        assert_eq!(
            std::mem::size_of::<BubbleUniform>(),
            128,
            "BubbleUniform should be 128 bytes for GPU 16-byte alignment"
        );
    }

    #[test]
    fn test_bubble_uniform_field_offsets() {
        // Verify critical fields are at expected offsets for shader compatibility
        use std::mem::offset_of;

        // Visual properties (first 9 floats = 36 bytes)
        assert_eq!(offset_of!(BubbleUniform, refractive_index), 0);
        assert_eq!(offset_of!(BubbleUniform, base_thickness_nm), 4);
        assert_eq!(offset_of!(BubbleUniform, time), 8);
        assert_eq!(offset_of!(BubbleUniform, interference_intensity), 12);
        assert_eq!(offset_of!(BubbleUniform, base_alpha), 16);
        assert_eq!(offset_of!(BubbleUniform, edge_alpha), 20);

        // Film dynamics (floats 10-13, bytes 36-52)
        assert_eq!(offset_of!(BubbleUniform, film_time), 36);
        assert_eq!(offset_of!(BubbleUniform, swirl_intensity), 40);
        assert_eq!(offset_of!(BubbleUniform, drainage_speed), 44);
        assert_eq!(offset_of!(BubbleUniform, pattern_scale), 48);
    }
}
