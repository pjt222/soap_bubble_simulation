//! Validates all WGSL shader files parse and pass naga validation.
//!
//! This catches struct mismatches, type errors, and syntax issues at
//! test time rather than at GPU initialization.

use naga::front::wgsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use std::path::Path;

const SHADER_DIR: &str = "src/render/shaders";

fn validate_wgsl(path: &Path) {
    let source = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));

    let module = wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("Validation failed for {}: {e}", path.display()));
}

#[test]
fn validate_bubble_wgsl() {
    validate_wgsl(Path::new(SHADER_DIR).join("bubble.wgsl").as_path());
}

#[test]
fn validate_wall_wgsl() {
    validate_wgsl(Path::new(SHADER_DIR).join("wall.wgsl").as_path());
}

#[test]
fn validate_bubble_instanced_wgsl() {
    validate_wgsl(
        Path::new(SHADER_DIR)
            .join("bubble_instanced.wgsl")
            .as_path(),
    );
}

#[test]
fn validate_drainage_wgsl() {
    validate_wgsl(Path::new(SHADER_DIR).join("drainage.wgsl").as_path());
}

#[test]
fn validate_branched_flow_compute_wgsl() {
    validate_wgsl(
        Path::new(SHADER_DIR)
            .join("branched_flow_compute.wgsl")
            .as_path(),
    );
}

#[test]
fn validate_caustics_wgsl() {
    validate_wgsl(Path::new(SHADER_DIR).join("caustics.wgsl").as_path());
}

#[test]
fn validate_caustics_compute_wgsl() {
    validate_wgsl(
        Path::new(SHADER_DIR)
            .join("caustics_compute.wgsl")
            .as_path(),
    );
}
