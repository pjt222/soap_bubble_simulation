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

pub mod pipeline;
pub mod camera;
pub mod gpu_drainage;
pub mod foam_renderer;
pub mod caustics;
pub mod branched_flow;
pub mod headless;
pub mod interference_lut;

pub use pipeline::RenderPipeline;
pub use camera::Camera;
pub use gpu_drainage::GPUDrainageSimulator;
pub use foam_renderer::{FoamRenderer, BubbleInstance};
pub use caustics::CausticRenderer;
pub use branched_flow::BranchedFlowSimulator;
pub use headless::HeadlessRenderPipeline;
