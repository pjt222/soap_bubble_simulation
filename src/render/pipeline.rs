//! wgpu render pipeline for soap bubble visualization

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::config::SimulationConfig;
use crate::physics::drainage::DrainageSimulator;
use crate::physics::foam_dynamics::FoamSimulator;
use crate::physics::geometry::{LodMeshCache, SpherePatch, Vertex};
use crate::render::animation::AnimationController;
use crate::render::branched_flow::{BranchedFlowSimulator, create_branched_flow_buffer};
use crate::render::camera::Camera;
use crate::render::foam_renderer::{FoamRenderer, SharedWallRenderer, WallInstance, WallVertex};
use crate::render::frame_exporter::FrameExporter;
use crate::render::gpu_drainage::GPUDrainageSimulator;
use crate::render::interference_lut::{
    LUT_ANGLE_SAMPLES, LUT_THICKNESS_SAMPLES, generate_interference_lut,
};
use crate::render::ui_state::{UiDisplayInfo, UiState};

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
            light_dir_x: 0.577, // 1/sqrt(3)
            light_dir_y: 0.577,
            light_dir_z: 0.577,
            // Patch view mode (enabled by default for focused visualization)
            patch_enabled: 1,
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

/// Main render pipeline for the soap bubble simulation.
///
/// Owns all wgpu state (device, queue, surface), GPU buffers, compute pipelines
/// (drainage, branched flow, caustics), the egui integration layer, and delegates
/// animation/export/UI to extracted subsystems.
/// Created via [`RenderPipeline::new()`] which initializes the GPU and all sub-systems.
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
    camera: Camera,
    bubble_uniform: BubbleUniform,
    // Mesh settings
    subdivision_level: u32,
    radius: f32,
    // egui integration
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    // Extracted subsystems
    animation: AnimationController,
    frame_exporter: FrameExporter,
    // Drainage simulation
    drainage_simulator: Option<DrainageSimulator>,
    physics_drainage_enabled: bool,
    drainage_time_scale: f32,
    // Gravity deformation
    deformation_enabled: bool,
    aspect_ratio: f32, // 1.0 = sphere, <1.0 = oblate (flattened)
    // LOD system
    lod_cache: LodMeshCache,
    current_lod_level: u32,
    lod_enabled: bool,
    lod_thresholds: [f32; 4], // Distance thresholds for LOD transitions [5→4, 4→3, 3→2, 2→1]
    // GPU drainage simulation
    gpu_drainage: GPUDrainageSimulator,
    gpu_drainage_enabled: bool,
    // Multi-bubble foam system
    foam_simulator: Option<FoamSimulator>,
    foam_renderer: FoamRenderer,
    foam_enabled: bool,
    foam_paused: bool,
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
    _interference_lut_sampler: wgpu::Sampler,
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
    /// Create a new render pipeline.
    ///
    /// Returns an error if GPU initialization fails (no compatible adapter,
    /// surface creation error, or device request denied).
    // put id:'gpu_init_device', label:'Initialize GPU device', input:'final_config.internal', output:'gpu_device.internal'
    pub async fn new(window: std::sync::Arc<winit::window::Window>) -> Result<Self, String> {
        let size = window.inner_size();

        // Create wgpu instance
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Create surface
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| format!("Failed to create GPU surface: {e}"))?;

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                "No compatible GPU adapter found. Ensure your GPU drivers are up to date."
                    .to_string()
            })?;

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
            .map_err(|e| format!("Failed to initialize GPU device: {e}"))?;

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

        // put id:'gpu_init_uniforms', label:'Upload uniforms to GPU', input:'gpu_device.internal', output:'uniform_buffers_gpu.internal'
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

        // put id:'gpu_init_lut_upload', label:'Upload interference LUT', input:'gpu_device.internal', output:'lut_texture_gpu.internal'
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
        let interference_lut_view =
            interference_lut_texture.create_view(&wgpu::TextureViewDescriptor::default());
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
                buffers: &[WallVertex::buffer_layout(), WallInstance::buffer_layout()],
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

        // Pre-allocate GPU buffers for the maximum LOD level to avoid allocation churn
        // when switching LOD levels at runtime. Use COPY_DST so we can update via write_buffer.
        let max_mesh = lod_cache.get_mesh(5); // Level 5 = highest detail, largest buffers
        let max_vertex_bytes = max_mesh.vertex_bytes().len();
        let max_index_bytes = max_mesh.index_bytes().len();

        // put id:'gpu_init_mesh_upload', label:'Upload mesh to GPU', input:'gpu_device.internal', output:'vertex_buffer_gpu.internal'
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: max_vertex_bytes as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Index Buffer"),
            size: max_index_bytes as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write initial mesh data (current LOD level)
        let mesh = lod_cache.get_mesh(subdivision_level);
        queue.write_buffer(&vertex_buffer, 0, mesh.vertex_bytes());
        queue.write_buffer(&index_buffer, 0, mesh.index_bytes());
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
            &device, 500e-9, // Initial thickness: 500nm
            128,    // Grid width (phi)
            64,     // Grid height (theta)
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

        // Pre-allocate patch buffers with COPY_DST to avoid allocation churn on slider changes
        let patch_vertex_bytes = bytemuck::cast_slice::<Vertex, u8>(&patch_vertices);
        let patch_index_bytes = bytemuck::cast_slice::<u32, u8>(&patch_indices);

        let patch_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Patch Vertex Buffer"),
            size: patch_vertex_bytes.len() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let patch_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Patch Index Buffer"),
            size: patch_index_bytes.len() as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.write_buffer(&patch_vertex_buffer, 0, patch_vertex_bytes);
        queue.write_buffer(&patch_index_buffer, 0, patch_index_bytes);
        let patch_num_indices = patch_indices.len() as u32;

        Ok(Self {
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
            // Extracted subsystems
            animation: AnimationController::new(),
            frame_exporter: FrameExporter::new(),
            // Drainage simulation (initialized lazily or with default config)
            drainage_simulator: None,
            physics_drainage_enabled: false,
            drainage_time_scale: 100.0, // Speed up simulation for visible effect
            // Gravity deformation (disabled by default)
            deformation_enabled: false,
            aspect_ratio: 1.0, // Perfect sphere
            // LOD system
            lod_cache,
            current_lod_level: subdivision_level,
            lod_enabled: false, // Disabled by default, user can enable
            lod_thresholds: [0.08, 0.15, 0.30, 0.60], // Distance thresholds in meters
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
            _interference_lut_sampler: interference_lut_sampler,
            last_refractive_index: bubble_uniform.refractive_index,
            // Patch view mode (enabled by default for focused visualization)
            patch_view_enabled: true,
            patch_center_u,
            patch_center_v,
            patch_half_size,
            patch_vertex_buffer,
            patch_index_buffer,
            patch_num_indices,
        })
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
        let new_ratio = if enabled {
            aspect_ratio.clamp(0.7, 1.0)
        } else {
            1.0
        };
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
            (
                sim.bubble_count(),
                sim.connection_count(),
                self.shared_wall_renderer.instance_count() as usize,
            )
        } else {
            (0, 0, 0)
        }
    }

    /// Regenerate mesh with current parameters (subdivision level, aspect ratio).
    /// Updates the pre-allocated GPU buffers via write_buffer (no allocation).
    fn regenerate_mesh(&mut self) {
        let mesh = self.lod_cache.get_mesh(self.subdivision_level);
        self.queue
            .write_buffer(&self.vertex_buffer, 0, mesh.vertex_bytes());
        self.queue
            .write_buffer(&self.index_buffer, 0, mesh.index_bytes());
        self.num_indices = mesh.indices.len() as u32;
    }

    /// Regenerate patch mesh when patch parameters change.
    /// Updates the pre-allocated GPU buffers via write_buffer (no allocation).
    fn regenerate_patch_mesh(&mut self) {
        let patch = SpherePatch::new(
            self.patch_center_u,
            self.patch_center_v,
            self.patch_half_size,
            32,
        );
        let (patch_vertices, patch_indices) =
            patch.generate_mesh_indexed(self.radius, self.aspect_ratio);

        self.queue.write_buffer(
            &self.patch_vertex_buffer,
            0,
            bytemuck::cast_slice(&patch_vertices),
        );
        self.queue.write_buffer(
            &self.patch_index_buffer,
            0,
            bytemuck::cast_slice(&patch_indices),
        );
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
    fn select_lod_level(distance: f32, thresholds: &[f32; 4]) -> u32 {
        if distance < thresholds[0] {
            5 // Closest: highest detail
        } else if distance < thresholds[1] {
            4
        } else if distance < thresholds[2] {
            3
        } else if distance < thresholds[3] {
            2
        } else {
            1 // Farthest: lowest detail
        }
    }

    /// Update LOD based on current camera distance (call each frame when LOD enabled).
    /// Applies 10% hysteresis to prevent oscillation near thresholds:
    /// switching to higher detail (closer) uses exact thresholds,
    /// switching to lower detail (farther) requires 10% more distance.
    fn update_lod(&mut self) {
        if !self.lod_enabled {
            return;
        }

        let distance = self.camera.distance;
        let candidate = Self::select_lod_level(distance, &self.lod_thresholds);

        // Hysteresis: resist switching to lower detail (higher distance)
        let new_level = if candidate < self.current_lod_level {
            // Switching to lower detail — require 10% beyond threshold
            let expanded: [f32; 4] = std::array::from_fn(|i| self.lod_thresholds[i] * 1.1);
            Self::select_lod_level(distance, &expanded)
        } else {
            candidate
        };

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

        // Update pre-allocated GPU buffers (no allocation, just data upload)
        let mesh = self.lod_cache.get_mesh(level);
        self.queue
            .write_buffer(&self.vertex_buffer, 0, mesh.vertex_bytes());
        self.queue
            .write_buffer(&self.index_buffer, 0, mesh.index_bytes());
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
        if new_size.width > 0
            && new_size.height > 0
            && (new_size.width != self.config.width || new_size.height != self.config.height)
        {
            // Wait for GPU to finish any pending work before recreating resources
            self.device.poll(wgpu::Maintain::Wait);

            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture =
                Self::create_depth_texture(&self.device, &self.config, self.msaa_samples);
            self.msaa_texture =
                Self::create_msaa_texture(&self.device, &self.config, self.msaa_samples);
            self.camera
                .set_aspect(new_size.width as f32 / new_size.height as f32);
            // Invalidate staging buffer — will be re-allocated at new size on next capture
            self.frame_exporter.invalidate_staging_buffer();
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
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Bubble Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bubble.wgsl").into()),
            });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&self.bind_group_layout],
                push_constant_ranges: &[],
            });

        self.render_pipeline =
            self.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
    pub fn handle_event(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::WindowEvent,
    ) -> bool {
        let response = self.egui_state.on_window_event(window, event);
        response.consumed
    }

    /// Update time for animation
    pub fn update(&mut self, dt: f32) {
        self.animation.update_fps(dt);
        self.bubble_uniform.time += dt;

        // LOD update based on camera distance
        self.update_lod();

        // Delegate to animation controller
        self.animation.update_rotation(&mut self.camera, dt);
        self.animation.update_film_time(&mut self.bubble_uniform, dt);
        self.animation.update_forces(&mut self.bubble_uniform, dt);

        // Physics-based drainage simulation
        if self.physics_drainage_enabled
            && let Some(ref mut simulator) = self.drainage_simulator
        {
            // Step the drainage simulation (with time scaling for visible effect)
            let scaled_dt = (dt * self.drainage_time_scale) as f64;
            simulator.step(scaled_dt);

            // Get thickness statistics from the simulation
            let field = simulator.thickness_field();
            let min_thickness = field.min_thickness() as f32 * 1e9; // Convert to nm
            let max_thickness = field.max_thickness() as f32 * 1e9;

            // Sample thickness at equator (theta = PI/2) to get representative value
            let equator_thickness =
                simulator.get_thickness(std::f64::consts::FRAC_PI_2, 0.0) as f32 * 1e9;

            // Update the base thickness to reflect drainage
            // Use the equator thickness as it's a good representative
            self.bubble_uniform.base_thickness_nm = equator_thickness;

            // Adjust drainage_speed based on actual simulation progress
            // This affects the procedural overlay in the shader
            let drain_ratio = min_thickness / max_thickness;
            self.bubble_uniform.drainage_speed = (1.0 - drain_ratio).clamp(0.0, 2.0);

            // Check for burst condition
            if simulator.has_critical_region() {
                log::info!(
                    "Bubble reached critical thickness - would burst at t={:.2}s",
                    simulator.current_time()
                );
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

        // Build egui UI using extracted UiState (avoids 95-parameter function)
        let raw_input = self.egui_state.take_egui_input(window);
        let mut ui_state = self.snapshot_ui_state();
        let display_info = self.display_info();

        let egui_output = self.egui_ctx.run(raw_input, |ctx| {
            ui_state.build_ui(ctx, &display_info);
        });

        // Apply UI changes back to pipeline state
        self.apply_ui_changes(&ui_state);

        // Handle egui platform output
        self.egui_state
            .handle_platform_output(window, egui_output.platform_output);

        // Tessellate egui
        let clipped_primitives = self
            .egui_ctx
            .tessellate(egui_output.shapes, egui_output.pixels_per_point);

        // Update egui textures
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: egui_output.pixels_per_point,
        };

        for (id, image_delta) in &egui_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, image_delta);
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // put id:'gpu_compute_dispatch', label:'Dispatch compute shaders', input:'uniform_buffers_gpu.internal', output:'compute_results_gpu.internal'
        if self.gpu_drainage_enabled {
            self.gpu_drainage.step(&mut encoder, self.animation.last_dt());
        }

        // Caustic compute pass (after drainage, before render)
        if self.caustic_renderer.enabled && self.gpu_drainage_enabled {
            self.caustic_renderer.compute(&mut encoder);
        }

        // Branched flow compute pass (ray tracing through film)
        if self.branched_flow_simulator.enabled && self.gpu_drainage_enabled {
            // Update scatterer positions (creates animated particle distribution)
            self.branched_flow_simulator
                .update_scatterers(&self.queue, self.bubble_uniform.film_time);
            self.branched_flow_simulator
                .step(&mut encoder, self.bubble_uniform.film_time);
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

        // put id:'gpu_render_pass', label:'Render bubble pass', input:'compute_results_gpu.internal', output:'framebuffer_gpu.internal'
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
                render_pass
                    .set_index_buffer(self.foam_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(
                    0..self.foam_num_indices,
                    0,
                    0..self.foam_renderer.instance_count(),
                );

                // Render shared walls (Plateau borders) between touching bubbles
                if self.shared_wall_renderer.has_walls() {
                    render_pass.set_pipeline(&self.wall_pipeline);
                    render_pass.set_bind_group(0, &self.bind_group, &[]);
                    render_pass
                        .set_vertex_buffer(0, self.shared_wall_renderer.vertex_buffer().slice(..));
                    render_pass.set_vertex_buffer(
                        1,
                        self.shared_wall_renderer.instance_buffer().slice(..),
                    );
                    render_pass.set_index_buffer(
                        self.shared_wall_renderer.index_buffer().slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(
                        0..self.shared_wall_renderer.num_mesh_indices(),
                        0,
                        0..self.shared_wall_renderer.instance_count(),
                    );
                }
            } else if self.patch_view_enabled && self.branched_flow_simulator.enabled {
                // Patch view mode: render only the focused patch mesh
                render_pass.set_pipeline(&self.render_pipeline);
                render_pass.set_bind_group(0, &self.bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.patch_vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(self.patch_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..self.patch_num_indices, 0, 0..1);
            } else {
                // Full sphere view
                render_pass.set_pipeline(&self.render_pipeline);
                render_pass.set_bind_group(0, &self.bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
            }

            // Render caustics on ground plane (after bubble, uses additive blending)
            if self.caustic_renderer.enabled && self.gpu_drainage_enabled {
                self.caustic_renderer.render(&mut render_pass);
            }
        }

        // put id:'gpu_render_egui', label:'Render egui overlay', input:'framebuffer_gpu.internal', output:'final_frame_gpu.internal'
        // Safety: The render pass is used immediately and dropped before encoder.finish()
        // The 'static lifetime is a limitation of the egui-wgpu API
        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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

            // egui-wgpu 0.31 requires `&mut RenderPass<'static>`. Use wgpu's official
            // `forget_lifetime()` to erase the borrow-checker's encoder lifetime tracking.
            // This is safe: the render pass is used only in this block and dropped before
            // encoder.finish(). Operations on the parent encoder will error at runtime
            // instead of compile-time, but we don't touch it while the pass is alive.
            let mut render_pass = render_pass.forget_lifetime();

            self.egui_renderer
                .render(&mut render_pass, &clipped_primitives, &screen_descriptor);
        }

        // Free egui textures
        for id in &egui_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        // Delegate frame capture to FrameExporter
        if self.frame_exporter.should_capture() {
            self.frame_exporter
                .prepare_capture(&self.device, &self.config, &mut encoder, &output.texture);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        if self.frame_exporter.should_capture() {
            self.frame_exporter
                .process_capture(&self.device, self.config.width, self.config.height);
        }

        output.present();

        Ok(())
    }
    /// Create a snapshot of current pipeline state as UiState for the egui panel.
    fn snapshot_ui_state(&self) -> UiState {
        let entry = self.branched_flow_simulator.params.entry_point;
        UiState {
            thickness: self.bubble_uniform.base_thickness_nm,
            refractive_index: self.bubble_uniform.refractive_index,
            interference_intensity: self.bubble_uniform.interference_intensity,
            base_alpha: self.bubble_uniform.base_alpha,
            edge_alpha: self.bubble_uniform.edge_alpha,
            bg_r: self.bubble_uniform.background_r,
            bg_g: self.bubble_uniform.background_g,
            bg_b: self.bubble_uniform.background_b,
            subdivision: self.subdivision_level,
            msaa_samples: self.msaa_samples,
            lod_enabled: self.lod_enabled,
            edge_smoothing_mode: self.bubble_uniform.edge_smoothing_mode,
            rotation_playing: self.animation.rotation_playing,
            rotation_speed: self.animation.rotation_speed,
            film_playing: self.animation.film_playing,
            film_speed: self.animation.film_speed,
            swirl_intensity: self.bubble_uniform.swirl_intensity,
            drainage_speed: self.bubble_uniform.drainage_speed,
            pattern_scale: self.bubble_uniform.pattern_scale,
            screenshot_requested: self.frame_exporter.screenshot_requested(),
            recording: self.frame_exporter.is_recording(),
            forces_enabled: self.animation.forces_enabled,
            wind_strength: self.animation.wind_strength,
            buoyancy_strength: self.animation.buoyancy_strength,
            physics_drainage_enabled: self.physics_drainage_enabled,
            drainage_time_scale: self.drainage_time_scale,
            reset_drainage_requested: false,
            gpu_drainage_enabled: self.gpu_drainage_enabled,
            gpu_drainage_time_scale: self.gpu_drainage.time_scale,
            gpu_drainage_steps: self.gpu_drainage.steps_per_frame,
            reset_gpu_drainage_requested: false,
            marangoni_enabled: self.gpu_drainage.marangoni_enabled,
            marangoni_coeff: self.gpu_drainage.params().marangoni_coeff,
            deformation_enabled: self.deformation_enabled,
            aspect_ratio: self.aspect_ratio,
            caustics_enabled: self.caustic_renderer.enabled,
            caustic_intensity: self.caustic_renderer.params.caustic_intensity,
            caustic_sharpness: self.caustic_renderer.params.caustic_sharpness,
            ground_y: self.caustic_renderer.params.ground_y,
            branched_flow_enabled: self.branched_flow_simulator.enabled,
            branched_flow_intensity: self.bubble_uniform.branched_flow_intensity,
            branched_flow_sharpness: self.bubble_uniform.branched_flow_sharpness,
            laser_azimuth: entry[2].atan2(entry[0]).to_degrees(),
            laser_elevation: entry[1].asin().to_degrees(),
            beam_spread: self.branched_flow_simulator.params.spread_angle.to_degrees(),
            bend_strength: self.branched_flow_simulator.params.bend_strength,
            num_rays: self.branched_flow_simulator.params.num_rays,
            num_scatterers: self.branched_flow_simulator.params.num_scatterers,
            scatterer_strength: self.branched_flow_simulator.params.scatterer_strength,
            scatterer_radius: self.branched_flow_simulator.params.scatterer_radius,
            particle_weight: self.branched_flow_simulator.params.particle_weight,
            patch_view_enabled: self.patch_view_enabled,
            patch_center_u: self.patch_center_u,
            patch_center_v: self.patch_center_v,
            patch_half_size: self.patch_half_size,
            foam_enabled: self.foam_enabled,
            foam_paused: self.foam_paused,
            foam_time_scale: self.foam_time_scale,
            add_bubble_requested: false,
            reset_foam_requested: false,
            regenerate_foam_requested: false,
            foam_gen_params: self.foam_generation_params.clone(),
        }
    }

    /// Create read-only display info for the UI panel.
    fn display_info(&self) -> UiDisplayInfo {
        UiDisplayInfo {
            camera_distance: self.camera.distance,
            camera_yaw: self.camera.yaw,
            camera_pitch: self.camera.pitch,
            fps: self.animation.fps(),
            width: self.config.width,
            height: self.config.height,
            num_triangles: self.num_indices / 3,
            time: self.bubble_uniform.time,
            current_lod_level: self.current_lod_level,
            frame_counter: self.frame_exporter.frame_counter(),
            drainage_sim_time: self.drainage_time().unwrap_or(0.0),
            gpu_drainage_time: self.gpu_drainage.current_time(),
            bubble_pos: [
                self.bubble_uniform.position_x,
                self.bubble_uniform.position_y,
                self.bubble_uniform.position_z,
            ],
            has_drainage_sim: self.drainage_simulator.is_some(),
            foam_stats: self.foam_stats(),
        }
    }

    /// Apply UI state changes back to the pipeline after egui has modified them.
    fn apply_ui_changes(&mut self, ui: &UiState) {
        // Film properties
        self.bubble_uniform.base_thickness_nm = ui.thickness;
        self.bubble_uniform.refractive_index = ui.refractive_index;
        self.bubble_uniform.interference_intensity = ui.interference_intensity;
        self.bubble_uniform.base_alpha = ui.base_alpha;
        self.bubble_uniform.edge_alpha = ui.edge_alpha;
        self.bubble_uniform.background_r = ui.bg_r;
        self.bubble_uniform.background_g = ui.bg_g;
        self.bubble_uniform.background_b = ui.bg_b;
        if ui.subdivision != self.subdivision_level {
            self.set_subdivision_level(ui.subdivision);
        }
        self.regenerate_interference_lut_if_needed();

        // Animation state
        self.animation.rotation_playing = ui.rotation_playing;
        self.animation.rotation_speed = ui.rotation_speed;
        self.animation.film_playing = ui.film_playing;
        self.animation.film_speed = ui.film_speed;
        self.bubble_uniform.swirl_intensity = ui.swirl_intensity;
        self.bubble_uniform.drainage_speed = ui.drainage_speed;
        self.bubble_uniform.pattern_scale = ui.pattern_scale;

        // Export state
        self.frame_exporter.set_screenshot_requested(ui.screenshot_requested);
        self.frame_exporter.set_recording(ui.recording);

        // External forces
        self.animation.forces_enabled = ui.forces_enabled;
        self.animation.wind_strength = ui.wind_strength;
        self.animation.buoyancy_strength = ui.buoyancy_strength;
        if !ui.forces_enabled {
            self.animation.reset_position(&mut self.bubble_uniform);
        }

        // Physics drainage
        self.drainage_time_scale = ui.drainage_time_scale;
        if ui.physics_drainage_enabled != self.physics_drainage_enabled {
            self.physics_drainage_enabled = ui.physics_drainage_enabled;
            if ui.physics_drainage_enabled && self.drainage_simulator.is_none() {
                let config = SimulationConfig::default();
                self.init_drainage_simulator(&config);
            }
        }
        if ui.reset_drainage_requested {
            self.reset_drainage(500.0);
        }

        // Deformation
        if ui.deformation_enabled != self.deformation_enabled
            || (ui.aspect_ratio - self.aspect_ratio).abs() > 0.001
        {
            self.set_deformation(ui.deformation_enabled, ui.aspect_ratio);
        }

        // Edge smoothing
        self.bubble_uniform.edge_smoothing_mode = ui.edge_smoothing_mode;

        // MSAA
        if ui.msaa_samples != self.msaa_samples {
            self.set_msaa_samples(ui.msaa_samples);
        }

        // LOD
        self.lod_enabled = ui.lod_enabled;

        // GPU drainage
        self.gpu_drainage_enabled = ui.gpu_drainage_enabled;
        self.gpu_drainage.enabled = ui.gpu_drainage_enabled;
        self.gpu_drainage.time_scale = ui.gpu_drainage_time_scale;
        self.gpu_drainage.steps_per_frame = ui.gpu_drainage_steps;
        if ui.reset_gpu_drainage_requested {
            self.gpu_drainage.reset(&self.queue, 500e-9);
        }

        // Marangoni
        if ui.marangoni_enabled != self.gpu_drainage.marangoni_enabled {
            self.gpu_drainage
                .set_marangoni_enabled(&self.queue, ui.marangoni_enabled);
        }
        if (ui.marangoni_coeff - self.gpu_drainage.params().marangoni_coeff).abs() > 0.0001 {
            let params = self.gpu_drainage.params();
            self.gpu_drainage.set_marangoni_params(
                &self.queue,
                params.gamma_air,
                params.gamma_reduction,
                params.surfactant_diffusion,
                ui.marangoni_coeff,
            );
        }

        // Caustics
        self.caustic_renderer.enabled = ui.caustics_enabled;
        let caustic_params_changed =
            (ui.caustic_intensity - self.caustic_renderer.params.caustic_intensity).abs() > 0.001
                || (ui.caustic_sharpness - self.caustic_renderer.params.caustic_sharpness).abs()
                    > 0.001
                || (ui.ground_y - self.caustic_renderer.params.ground_y).abs() > 0.001;
        if caustic_params_changed {
            self.caustic_renderer.params.caustic_intensity = ui.caustic_intensity;
            self.caustic_renderer.params.caustic_sharpness = ui.caustic_sharpness;
            if (ui.ground_y - self.caustic_renderer.params.ground_y).abs() > 0.001 {
                self.caustic_renderer.set_ground_y(&self.device, ui.ground_y);
            }
            self.caustic_renderer.update_params(&self.queue);
        }

        // Branched flow
        self.branched_flow_simulator.enabled = ui.branched_flow_enabled;
        self.bubble_uniform.branched_flow_enabled = u32::from(ui.branched_flow_enabled);
        self.bubble_uniform.branched_flow_intensity = ui.branched_flow_intensity;
        self.bubble_uniform.branched_flow_sharpness = ui.branched_flow_sharpness;
        self.branched_flow_simulator
            .set_entry_point(ui.laser_azimuth, ui.laser_elevation);
        self.branched_flow_simulator.params.spread_angle = ui.beam_spread.to_radians();
        self.branched_flow_simulator.params.bend_strength = ui.bend_strength;
        self.branched_flow_simulator.params.num_rays = ui.num_rays;
        self.branched_flow_simulator.params.num_scatterers = ui.num_scatterers;
        self.branched_flow_simulator.params.scatterer_strength = ui.scatterer_strength;
        self.branched_flow_simulator.params.scatterer_radius = ui.scatterer_radius;
        self.branched_flow_simulator.params.particle_weight = ui.particle_weight;
        // Sync film dynamics
        self.branched_flow_simulator.params.base_thickness_nm =
            self.bubble_uniform.base_thickness_nm;
        self.branched_flow_simulator.params.swirl_intensity = self.bubble_uniform.swirl_intensity;
        self.branched_flow_simulator.params.drainage_speed = self.bubble_uniform.drainage_speed;
        self.branched_flow_simulator.params.pattern_scale = self.bubble_uniform.pattern_scale;

        // Patch view
        self.patch_view_enabled = ui.patch_view_enabled;
        self.branched_flow_simulator.params.patch_enabled = u32::from(ui.patch_view_enabled);
        let patch_params_changed = (ui.patch_center_u - self.patch_center_u).abs() > 0.001
            || (ui.patch_center_v - self.patch_center_v).abs() > 0.001
            || (ui.patch_half_size - self.patch_half_size).abs() > 0.001;
        if patch_params_changed {
            self.patch_center_u = ui.patch_center_u;
            self.patch_center_v = ui.patch_center_v;
            self.patch_half_size = ui.patch_half_size;
            self.regenerate_patch_mesh();
        }
        self.branched_flow_simulator.params.patch_center_u = self.patch_center_u;
        self.branched_flow_simulator.params.patch_center_v = self.patch_center_v;
        self.branched_flow_simulator.params.patch_half_size = self.patch_half_size;
        self.bubble_uniform.patch_enabled = u32::from(ui.patch_view_enabled);
        self.bubble_uniform.patch_center_u = self.patch_center_u;
        self.bubble_uniform.patch_center_v = self.patch_center_v;
        self.bubble_uniform.patch_half_size = self.patch_half_size;

        // Foam
        if ui.foam_enabled != self.foam_enabled {
            self.set_foam_enabled(ui.foam_enabled);
        }
        self.foam_paused = ui.foam_paused;
        self.foam_time_scale = ui.foam_time_scale;
        self.foam_generation_params = ui.foam_gen_params.clone();
        if ui.add_bubble_requested {
            self.add_foam_bubble(self.radius * 0.8);
        }
        if ui.reset_foam_requested {
            self.reset_foam();
        }
        if ui.regenerate_foam_requested {
            self.regenerate_foam();
        }
    }

    /// Request a screenshot on the next frame.
    pub fn request_screenshot(&mut self) {
        self.frame_exporter.request_screenshot();
    }

    /// Toggle recording mode.
    pub fn toggle_recording(&mut self) {
        self.frame_exporter.toggle_recording();
    }

    /// Capture current frame to a PNG file.
    pub fn capture_frame<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), String> {
        FrameExporter::capture_frame(&self.device, &self.queue, &self.surface, &self.config, path)
    }

    /// Mutable access to the camera (for orbit/zoom from main.rs input handlers).
    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }

    /// Set the base film thickness in nanometers.
    pub fn set_thickness_nm(&mut self, nm: f32) {
        self.bubble_uniform.base_thickness_nm = nm;
    }

    /// Set the refractive index of the film.
    pub fn set_refractive_index(&mut self, n: f32) {
        self.bubble_uniform.refractive_index = n;
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
        assert!(
            (uniform.refractive_index - 1.33).abs() < 1e-6,
            "refractive_index"
        );
        assert!(
            (uniform.base_thickness_nm - 500.0).abs() < 1e-6,
            "base_thickness_nm"
        );
        assert!(
            (uniform.interference_intensity - 4.0).abs() < 1e-6,
            "interference_intensity"
        );
        assert!((uniform.base_alpha - 0.3).abs() < 1e-6, "base_alpha");
        assert!((uniform.edge_alpha - 0.6).abs() < 1e-6, "edge_alpha");

        // Film dynamics - these are critical for animation
        assert!(
            (uniform.film_time - 0.0).abs() < 1e-6,
            "film_time should start at 0"
        );
        assert!(
            (uniform.swirl_intensity - 1.0).abs() < 1e-6,
            "swirl_intensity"
        );
        assert!(
            (uniform.drainage_speed - 0.5).abs() < 1e-6,
            "drainage_speed"
        );
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
        // Verify ALL fields in the shared region (bytes 0-80) that wall.wgsl and
        // bubble_instanced.wgsl depend on. Any reorder here silently breaks those shaders.
        use std::mem::offset_of;

        // Visual properties (9 floats, bytes 0-35)
        assert_eq!(offset_of!(BubbleUniform, refractive_index), 0);
        assert_eq!(offset_of!(BubbleUniform, base_thickness_nm), 4);
        assert_eq!(offset_of!(BubbleUniform, time), 8);
        assert_eq!(offset_of!(BubbleUniform, interference_intensity), 12);
        assert_eq!(offset_of!(BubbleUniform, base_alpha), 16);
        assert_eq!(offset_of!(BubbleUniform, edge_alpha), 20);
        assert_eq!(offset_of!(BubbleUniform, background_r), 24);
        assert_eq!(offset_of!(BubbleUniform, background_g), 28);
        assert_eq!(offset_of!(BubbleUniform, background_b), 32);

        // Film dynamics (4 floats, bytes 36-51)
        assert_eq!(offset_of!(BubbleUniform, film_time), 36);
        assert_eq!(offset_of!(BubbleUniform, swirl_intensity), 40);
        assert_eq!(offset_of!(BubbleUniform, drainage_speed), 44);
        assert_eq!(offset_of!(BubbleUniform, pattern_scale), 48);

        // Position (3 floats, bytes 52-63)
        assert_eq!(offset_of!(BubbleUniform, position_x), 52);
        assert_eq!(offset_of!(BubbleUniform, position_y), 56);
        assert_eq!(offset_of!(BubbleUniform, position_z), 60);

        // Edge smoothing (u32 at byte 64) — end of the shared 80-byte region
        assert_eq!(offset_of!(BubbleUniform, edge_smoothing_mode), 64);

        // Branched flow fields (bytes 68-79 overlap wall/instanced _reserved fields)
        assert_eq!(offset_of!(BubbleUniform, branched_flow_enabled), 68);
        assert_eq!(offset_of!(BubbleUniform, branched_flow_intensity), 72);
        assert_eq!(offset_of!(BubbleUniform, branched_flow_scale), 76);
    }
}
