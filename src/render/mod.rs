//! GPU rendering modules
//!
//! Contains wgpu-based rendering infrastructure:
//! - Pipeline: Render pipeline setup and management
//! - Camera: Orbit camera controls
//! - Shaders: WGSL shaders for thin-film interference
//! - GPU Drainage: Compute shader-based drainage simulation
//! - Foam Renderer: Multi-bubble instanced rendering
//! - Headless: Headless rendering for automated testing

pub mod pipeline;
pub mod camera;
pub mod gpu_drainage;
pub mod foam_renderer;
pub mod headless;

pub use pipeline::RenderPipeline;
pub use camera::Camera;
pub use gpu_drainage::GPUDrainageSimulator;
pub use foam_renderer::{FoamRenderer, BubbleInstance};
pub use headless::HeadlessRenderPipeline;
