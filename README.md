# Soap Bubble Simulation

Physically accurate 3D soap bubble simulation in Rust with GPU-accelerated visualization. The simulation models thin-film interference colors, drainage dynamics, and real-time parameter adjustment.

![Rust](https://img.shields.io/badge/Rust-2024_Edition-orange)
![License](https://img.shields.io/badge/License-MIT-blue)
![wgpu](https://img.shields.io/badge/wgpu-24.0-green)

![Soap Bubble Simulation](showcase.gif)

## Features

- **Thin-film interference rendering** - Realistic iridescent colors based on optical physics
- **Real-time GPU rendering** - wgpu-based pipeline with Vulkan/Metal/DX12 support
- **Interactive controls** - egui-based UI for adjusting simulation parameters
- **Orbit camera** - Mouse-controlled camera with zoom and rotation
- **Configurable parameters** - Film thickness, refractive index, opacity, mesh detail

## Physics Background

### Thin-Film Interference

The iridescent colors of soap bubbles arise from light interference at the thin film surfaces. When light hits the soap film:

1. Part reflects from the outer surface
2. Part transmits, reflects from the inner surface, and exits
3. These two waves interfere constructively or destructively depending on:
   - Film thickness `d`
   - Viewing angle `θ`
   - Light wavelength `λ`

**Optical path difference:**

```
δ = 2 n_film d cos(θ_t) + λ/2
```

Where:
- `n_film` = Refractive index of soap film (~1.33)
- `d` = Film thickness (typically 100-1000 nm)
- `θ_t` = Transmission angle in film (from Snell's law)
- `λ/2` = Phase shift from reflection at denser medium

The simulation computes interference intensity for RGB wavelengths (650nm, 532nm, 450nm) and combines with Fresnel reflection using Schlick's approximation.

### Drainage

The film thickness varies across the bubble surface due to gravity. The top drains faster, becoming thinner, while fluid accumulates at the bottom. This is modeled using the normal vector's Y-component to create realistic thickness gradients.

## Installation

### Prerequisites

- Rust 1.75+ (2024 edition)
- GPU with Vulkan, Metal, or DX12 support
- For WSL2: X11 server (VcXsrv, X410, or WSLg)

### Build

```bash
# Clone the repository
git clone https://github.com/pjt222/soap_bubble_simulation.git
cd soap_bubble_simulation

# Debug build
cargo build

# Release build (recommended for performance)
cargo build --release
```

## Usage

### Running the Simulation

```bash
# Default parameters
cargo run --release

# With custom config file
cargo run --release -- --config path/to/config.json

# Override specific parameters
cargo run --release -- --thickness 600 --diameter 0.08
```

### WSL2 Users

```bash
# Set X11 backend explicitly
WAYLAND_DISPLAY= WINIT_UNIX_BACKEND=x11 ./target/release/soap-bubble-sim
```

### Controls

| Input | Action |
|-------|--------|
| Left mouse drag | Orbit camera around bubble |
| Mouse wheel | Zoom in/out |
| Escape | Exit application |

### UI Controls

The egui panel provides real-time adjustment of:

- **Film Thickness** (100-1000 nm) - Controls base interference colors
- **Refractive Index** (1.0-2.0) - Affects color intensity and shift
- **Color Intensity** (0.5-10.0) - Amplifies interference visibility
- **Base Opacity** (0.0-1.0) - Transparency when viewing straight-on
- **Edge Opacity** (0.0-1.0) - Transparency at glancing angles (Fresnel effect)
- **Mesh Detail** (1-5) - Subdivision level of sphere mesh
- **Background Color** - RGB color picker

## Configuration

Parameters can be set via JSON configuration file:

```json
{
  "bubble": {
    "diameter": 0.05,
    "film_thickness_nm": 500.0,
    "critical_thickness_nm": 10.0,
    "refractive_index": 1.33
  },
  "fluid": {
    "viscosity": 0.001,
    "surface_tension": 0.025,
    "density": 1000.0,
    "surfactant_diffusion": 1e-9,
    "surfactant_concentration": 0.5
  },
  "resolution": 128
}
```

## Project Structure

```
soap-bubble-sim/
├── src/
│   ├── main.rs              # Entry point, winit event loop, CLI
│   ├── lib.rs               # Library exports
│   ├── config.rs            # SimulationConfig, parameter structs
│   ├── physics/
│   │   ├── geometry.rs      # UV sphere mesh generation
│   │   ├── drainage.rs      # Film thickness dynamics
│   │   └── interference.rs  # Color computation (CPU reference)
│   ├── render/
│   │   ├── pipeline.rs      # wgpu setup, buffers, egui integration
│   │   ├── camera.rs        # Orbit camera with zoom/pan
│   │   └── shaders/
│   │       └── bubble.wgsl  # Thin-film interference shader
│   └── export/
│       └── image_export.rs  # PNG export functionality
└── config/
    └── default.json         # Default parameters
```

## Key Development Insights

### UV Sphere vs Icosphere

Initially, an icosphere mesh was used, but this caused visible **zigzag seam artifacts** at the equator due to UV coordinate discontinuities. The solution was twofold:

1. **Switch to UV sphere** - Regular latitude/longitude grid with proper UV wrapping
2. **Use normal vectors for thickness** - Instead of UV coordinates, the shader uses the interpolated normal vector to compute film thickness, which is continuous across the entire sphere

### WGSL Struct Alignment

A critical issue arose with uniform buffer size mismatch between Rust and WGSL:

- WGSL's `vec3<f32>` has **16-byte alignment**, inflating struct size
- Rust's `[f32; 3]` has only **4-byte alignment**

**Solution:** Use separate `f32` padding fields instead of `vec3<f32>`:

```wgsl
// Instead of: _padding: vec3<f32>
_padding1: f32,
_padding2: f32,
_padding3: f32,
```

### egui Depth Buffer Configuration

When integrating egui with a 3D scene using depth testing:

- The egui renderer's depth format must match the render pass configuration
- Either both use `Some(Depth32Float)` or both use `None`
- Mismatch causes validation errors at runtime

## References

### Thin-Film Optics

- Born, M. & Wolf, E. (1999). *Principles of Optics* (7th ed.). Cambridge University Press.
  - Chapter 7: Interference and diffraction with partially coherent light
  - Chapter 13: Optics of metals and thin films

- Macleod, H. A. (2010). *Thin-Film Optical Filters* (4th ed.). CRC Press.
  - Comprehensive treatment of interference in thin films

### Soap Bubble Physics

- Isenberg, C. (1992). *The Science of Soap Films and Soap Bubbles*. Dover Publications.
  - Classic reference for soap film geometry and physics

- de Gennes, P.-G., Brochard-Wyart, F., & Quéré, D. (2004). *Capillarity and Wetting Phenomena*. Springer.
  - Chapter on thin liquid films and drainage

### Fresnel Equations

- Schlick, C. (1994). "An Inexpensive BRDF Model for Physically-based Rendering." *Computer Graphics Forum*, 13(3), 233-246.
  - The Schlick approximation used in this simulation

### GPU Rendering

- wgpu Documentation: https://wgpu.rs/
- Learn WGPU Tutorial: https://sotrh.github.io/learn-wgpu/

### Related Research

- Glassner, A. (2000). "Soap Bubbles: Part 1 & 2." *IEEE Computer Graphics and Applications*, 20(5-6).
  - Rendering techniques for soap bubbles

- Patsyk, A., et al. (2020). "Observation of branched flow of light." *Nature*, 583, 60-65.
  - DOI: [10.1038/s41586-020-2376-8](https://doi.org/10.1038/s41586-020-2376-8)
  - Experimental observation of light patterns in soap films

- Huang, W., et al. (2020). "Chemomechanical simulation of soap film flow on spherical bubbles." *ACM Transactions on Graphics*, 39(4).
  - Advanced simulation of soap film dynamics

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

Philipp Thoss (ph.thoss@gmx.de)

## Acknowledgments

- [wgpu](https://wgpu.rs/) - Cross-platform GPU abstraction
- [egui](https://github.com/emilk/egui) - Immediate mode GUI
- [winit](https://github.com/rust-windowing/winit) - Window creation and event handling
