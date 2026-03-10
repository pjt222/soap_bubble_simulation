//! Soap Bubble Simulation Library
//!
//! Physically accurate 3D simulation of soap bubbles with:
//! - Thin-film interference colors
//! - Drainage dynamics
//! - GPU-accelerated rendering

pub mod config;
pub mod export;
pub mod physics;
pub mod render;

pub use config::SimulationConfig;
