# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Physically accurate 3D soap bubble simulation in Rust with GPU-accelerated visualization. The simulation models thin-film interference colors, drainage dynamics, and optional deformation.

## Build Commands

```bash
cargo build                    # Debug build
cargo build --release          # Optimized build
cargo run                      # Run with default parameters
cargo run -- --config path.json  # Run with custom config
cargo run -- --thickness 600   # Override film thickness (nm)
cargo run -- --diameter 0.08   # Override diameter (meters)
cargo test                     # Run tests
cargo clippy                   # Lint
```

## Project Structure

```
soap-bubble-sim/
├── src/
│   ├── main.rs           # Entry point, winit event loop, CLI (clap)
│   ├── lib.rs            # Library root, re-exports
│   ├── config.rs         # SimulationConfig, BubbleParameters, FluidParameters
│   ├── physics/
│   │   ├── geometry.rs   # SphereMesh, Vertex (icosphere generation)
│   │   ├── drainage.rs   # DrainageSimulator, ThicknessField
│   │   └── interference.rs # InterferenceCalculator, color computation
│   ├── render/
│   │   ├── pipeline.rs   # RenderPipeline (wgpu setup, buffers, rendering)
│   │   ├── camera.rs     # Camera, CameraUniform (orbit controls)
│   │   └── shaders/
│   │       └── bubble.wgsl # Thin-film interference fragment shader
│   └── export/
│       └── image_export.rs # PNG export
└── config/default.json   # Default simulation parameters
```

## Core Physics

**Thin-film interference**: Colors from light interfering at film surfaces.
- Optical path: `δ = 2 n_film d cos(θ_t) + λ/2`
- Wavelengths: R=650nm, G=532nm, B=450nm
- Fresnel reflection via Schlick approximation

**Drainage**: Film thins under gravity (simplified in shader via UV mapping).

## Key Modules

- `physics::geometry::SphereMesh` - Generates icosphere with configurable subdivision
- `physics::interference::InterferenceCalculator` - CPU-side color computation (also has GLSL reference)
- `render::pipeline::RenderPipeline` - Owns wgpu state, renders bubble
- `render::camera::Camera` - Orbit camera with zoom/pan

## Controls

- **Left mouse drag**: Orbit camera around bubble
- **Mouse wheel**: Zoom in/out
- **Escape**: Exit

## Architecture: Branched Flow & Film Dynamics Sync

The branched flow system traces light rays through the soap film as a 2D waveguide
(GRIN optics). Rays bend toward thicker regions via `thickness_gradient()`.

**Key insight — dual thickness sources must stay in sync:**
- The **fragment shader** (`bubble.wgsl`) computes film thickness procedurally:
  `base * (1 - drainage + fbm_noise + swirl + gravity_ripples)`. This drives the
  visible iridescent colors.
- The **compute shader** (`branched_flow_compute.wgsl`) reads a GPU drainage buffer
  for physical thickness, then applies the *same* noise modulations so ray bending
  matches the visible surface patterns.
- Film dynamics parameters (`base_thickness_nm`, `swirl_intensity`, `drainage_speed`,
  `pattern_scale`) are synced from `BubbleUniform` → `BranchedFlowParams` each frame
  in `pipeline.rs`.

**Compute shader uses 3 FBM octaves** (vs fragment shader's 4) for performance —
sufficient for gradient-based ray bending where fine detail averages out.

**Unit consistency:** The drainage buffer stores thickness in meters, scaled by
`thickness_scale` (1e6) to micrometers. Noise modulations are fractional multipliers
on the buffer value, so units cancel naturally.

**Struct alignment:** `BranchedFlowParams` is 112 bytes (28 × f32), `BubbleUniform`
is 128 bytes (32 × f32), both padded for 16-byte GPU alignment. The Rust structs
and WGSL structs must match exactly — verified by size alignment tests.

## Patch View Mode

The patch view mode renders a small curved rectangular patch (~10% of sphere surface)
instead of the full bubble. This provides a focused view of branched flow effects.

**Key insight — rays must intersect the patch region:**
- The `SpherePatch` struct generates a curved mesh from UV bounds on the sphere
- Rays trace from the laser entry point (Azimuth/Elevation UI controls)
- In patch mode, deposits only occur when ray UV position is within patch bounds
- The fragment shader remaps patch-local UVs when sampling the branched flow texture

**Attempted optimizations that caused GPU freezes:**
- Moving ray entry point to patch center caused synchronization issues
- Scaling beam spread to patch size created invalid ray distributions
- Breaking ray loops early when outside patch caused incomplete traces

**Working approach:** Keep original ray tracing, filter deposits by patch bounds.
To see branched flow in patch mode, position the laser to aim through the patch region.

**Performance note:** Patch mode doesn't reduce ray computation (all rays still trace).
The benefit is focused visualization, not GPU savings. For true performance gains,
reduce `num_rays` parameter when in patch mode.

## Configuration

Parameters in `config/default.json`:
- `bubble.diameter`: Bubble size in meters (default: 0.05 = 5cm)
- `bubble.film_thickness_nm`: Initial film thickness (default: 500nm)
- `bubble.refractive_index`: Soap film index (default: 1.33)
- `fluid.*`: Viscosity, surface tension, density (for future drainage sim)
- `resolution`: Grid resolution for thickness field
