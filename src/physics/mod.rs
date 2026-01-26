//! Physics simulation modules
//!
//! Contains the physical models for soap bubble behavior:
//! - Geometry: Spherical coordinates and mesh generation
//! - Drainage: Film thickness evolution under gravity
//! - Interference: Thin-film optical calculations
//! - Fluid: Surface flow and Marangoni effect
//! - Foam: Multi-bubble foam system with inter-bubble physics

pub mod geometry;
pub mod drainage;
pub mod interference;
pub mod foam;
pub mod foam_dynamics;
pub mod foam_generation;

pub use geometry::SphereMesh;
pub use drainage::DrainageSimulator;
pub use interference::InterferenceCalculator;
pub use foam::{Bubble, BubbleCluster, BubbleConnection};
pub use foam_dynamics::FoamSimulator;
pub use foam_generation::{FoamGenerator, GenerationParams, PositioningMode, SizeDistribution};
