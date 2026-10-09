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

/// Shaders that still add pi to an Airy phase, each with the issue that tracks it.
/// wall.wgsl renders the Airy transmission term; its pi and its transmission form
/// partly cancel and must be fixed together (issue #51).
const KNOWN_EXTRA_PI_PHASE: &[&str] = &["wall.wgsl"];

/// Source-level guard for issue #42: the Airy reflectance already contains the
/// half-wave reflection flip (r21 = -r12), so a phase built from a wavelength must
/// not add pi. The CPU paths are covered numerically in `interference.rs` and
/// `interference_lut.rs`; WGSL cannot be evaluated in unit tests, and the foam
/// shader is not covered by any golden image, so this catches the copy-paste
/// regression that spread the bug to three shaders.
#[test]
fn no_shader_adds_pi_to_an_airy_phase() {
    let mut violations = Vec::new();
    for entry in std::fs::read_dir(SHADER_DIR).expect("shader directory exists") {
        let path = entry.expect("readable directory entry").path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("wgsl") {
            continue;
        }
        let file_name = path.file_name().unwrap().to_string_lossy().into_owned();
        if KNOWN_EXTRA_PI_PHASE.contains(&file_name.as_str()) {
            continue;
        }
        let source = std::fs::read_to_string(&path).expect("readable shader");
        for (line_index, line) in source.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            if code.contains("wavelength") && (code.contains("+ PI") || code.contains("+ pi")) {
                violations.push(format!("{file_name}:{}: {}", line_index + 1, line.trim()));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "Airy phase must be the geometric phase only (issue #42):\n{}",
        violations.join("\n")
    );
}

#[test]
fn known_extra_pi_phase_entries_are_still_needed() {
    // Fails once wall.wgsl is fixed, so the allow-list cannot go stale.
    for file_name in KNOWN_EXTRA_PI_PHASE {
        let source = std::fs::read_to_string(Path::new(SHADER_DIR).join(file_name))
            .expect("allow-listed shader exists");
        assert!(
            source.lines().any(|line| {
                let code = line.split("//").next().unwrap_or("");
                code.contains("wavelength") && code.contains("+ PI")
            }),
            "{file_name} no longer adds pi; remove it from KNOWN_EXTRA_PI_PHASE"
        );
    }
}
