//! Export modules
//!
//! Handles image and video export:
//! - Image: PNG export of rendered frames

pub mod image_export;

pub use image_export::export_frame;
