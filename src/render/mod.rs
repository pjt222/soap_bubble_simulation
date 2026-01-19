//! GPU rendering modules
//!
//! Contains wgpu-based rendering infrastructure:
//! - Pipeline: Render pipeline setup and management
//! - Camera: Orbit camera controls
//! - Shaders: WGSL shaders for thin-film interference

pub mod pipeline;
pub mod camera;

pub use pipeline::RenderPipeline;
pub use camera::Camera;
