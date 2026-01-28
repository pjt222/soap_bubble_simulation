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

**Struct alignment:** `BranchedFlowParams` is 96 bytes (24 × f32), padded for
16-byte GPU alignment. The Rust struct and WGSL struct must match exactly — verified
by `params_struct_size_matches_wgsl_layout` test.

## Configuration

Parameters in `config/default.json`:
- `bubble.diameter`: Bubble size in meters (default: 0.05 = 5cm)
- `bubble.film_thickness_nm`: Initial film thickness (default: 500nm)
- `bubble.refractive_index`: Soap film index (default: 1.33)
- `fluid.*`: Viscosity, surface tension, density (for future drainage sim)
- `resolution`: Grid resolution for thickness field
