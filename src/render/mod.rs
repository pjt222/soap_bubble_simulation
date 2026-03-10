//! GPU rendering modules
//!
//! Contains wgpu-based rendering infrastructure:
//! - Pipeline: Render pipeline setup and management
//! - Camera: Orbit camera controls
//! - Shaders: WGSL shaders for thin-film interference
//! - GPU Drainage: Compute shader-based drainage simulation
//! - Foam Renderer: Multi-bubble instanced rendering
//! - Caustics: Branched flow / caustic pattern rendering
//! - Branched Flow: Ray-traced light propagation through film
//! - Headless: Headless rendering for automated testing
//! - Interference LUT: Pre-computed interference color lookup table
//! - Animation: Camera orbit, film dynamics, force simulation
//! - Frame Exporter: Screenshot and recording frame capture
//! - UI State: egui control panel state management

pub mod animation;
pub mod branched_flow;
pub mod camera;
pub mod caustics;
pub mod foam_renderer;
pub mod frame_exporter;
pub mod gpu_drainage;
pub mod headless;
pub mod interference_lut;
pub mod pipeline;
pub mod ui_state;

pub use branched_flow::BranchedFlowSimulator;
pub use camera::Camera;
pub use caustics::CausticRenderer;
pub use foam_renderer::{BubbleInstance, FoamRenderer};
pub use gpu_drainage::GPUDrainageSimulator;
pub use headless::HeadlessRenderPipeline;
pub use pipeline::RenderPipeline;
