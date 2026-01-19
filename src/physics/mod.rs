//! Physics simulation modules
//!
//! Contains the physical models for soap bubble behavior:
//! - Geometry: Spherical coordinates and mesh generation
//! - Drainage: Film thickness evolution under gravity
//! - Interference: Thin-film optical calculations
//! - Fluid: Surface flow and Marangoni effect

pub mod geometry;
pub mod drainage;
pub mod interference;

pub use geometry::SphereMesh;
pub use drainage::DrainageSimulator;
pub use interference::InterferenceCalculator;
