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

## Configuration

Parameters in `config/default.json`:
- `bubble.diameter`: Bubble size in meters (default: 0.05 = 5cm)
- `bubble.film_thickness_nm`: Initial film thickness (default: 500nm)
- `bubble.refractive_index`: Soap film index (default: 1.33)
- `fluid.*`: Viscosity, surface tension, density (for future drainage sim)
- `resolution`: Grid resolution for thickness field
