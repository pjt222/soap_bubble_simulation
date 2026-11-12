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
│   │   ├── film_waveguide.rs # TE0 effective index n_eff(d), ln n_eff table (GRIN)
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
- Optical path (two-beam convention): `δ = 2 n_film d cos(θ_t) + λ/2`. The code uses the Airy
  form `R = F sin²(φ/2) / (1 + F sin²(φ/2))` with the **geometric** phase `φ = 4π n_film d cos(θ_t) / λ`
  and no extra π, because `r21 = −r12` already carries the half-wave flip; adding π inverts every
  fringe (issue #42). Guarded by the interference tests in `src/physics/interference.rs`,
  `src/render/interference_lut.rs` and `tests/wgsl_validation.rs`; `scripts/physics/thin_film_reference.py`
  tabulates the math (it does not run the project code)
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

## Architecture: Branched Flow & Film Dynamics

The branched flow system traces light rays through the soap film as a 2D waveguide
(GRIN optics). Guided light sees the effective index `n_eff(d)` of the film's TE0 slab
mode, so rays bend toward thicker film along `grad ln n_eff` (#45):
- `physics::film_waveguide` solves for `n_eff` on the CPU and uploads `ln n_eff` as a
  table (binding 5, 5 nm steps up to 4 µm, 532 nm light). `thickness_gradient_uv` in
  `branched_flow_compute.wgsl` converts the drainage thickness from metres with
  `thickness_scale` (1e6), looks it up, and takes the surface gradient per radian on
  the unit sphere (`surface_gradient`: d/du ÷ 2π sinθ, d/dv ÷ π).
- `bend_strength` multiplies the GRIN force and the scatterer force alike; with
  `particle_weight = 0`, 1.0 is the physical curvature near the chart origin. The default
  5.0 was tuned for the scatterers.
- The live drainage field stays uniform until #43, so the GRIN force is zero in the app
  today. `grin_force_along_theta_matches_the_chart_integrator` and
  `grin_force_along_phi_across_the_seam_matches_the_chart_integrator` check it on synthetic
  fields (`#[ignore]`d GPU tests: lavapipe via `scripts/test-local.sh`, not in CI, #58).
  They pin the loop's gnomonic-chart integrator, which differs from geodesic propagation
  away from the chart origin (#47).

**Dual thickness sources (intentionally independent):**
- The **fragment shader** (`bubble.wgsl`) computes film thickness procedurally:
  `base * (1 - drainage + fbm_noise + swirl + gravity_ripples)`. This drives the
  visible iridescent colors.
- The **compute shader** (`branched_flow_compute.wgsl`) reads a GPU drainage buffer
  for physical thickness via bilinear-interpolated sampling. It does NOT apply
  the fragment shader's noise modulations — ray bending is driven purely by the
  physical drainage buffer. The `base_thickness_nm`, `swirl_intensity`,
  `drainage_speed`, and `pattern_scale` fields in `BranchedFlowParams` are
  reserved but currently unused in the shader.

**Stale buffer prevention:** The GPU drainage simulator double-buffers thickness.
The branched flow bind group is rebuilt each frame with the current thickness
buffer via `rebuild_bind_group()` to prevent stale reads.

**Performance features:**
- True spatial hash: scatterers sorted by grid cell on CPU, prefix-sum
  `cell_offsets` buffer enables O(k) GPU lookup per ray step
- Adaptive step size: `dt` scales by `1/(gradient*10)` clamped to [0.3, 3.0]
- Patch mode ray spawning: rays start within patch UV bounds for higher density
- Bilinear thickness sampling for smooth gradients
- Pole singularity: `smoothstep(0, 0.15, sin_theta)` taper avoids artifacts
- GPU timestamp profiling via `GpuProfiler` (requires TIMESTAMP_QUERY feature)

**Struct alignment:** `BranchedFlowParams` is 112 bytes (28 × f32), `BubbleUniform`
is 128 bytes (32 × f32), both padded for 16-byte GPU alignment. The Rust structs
and WGSL structs must match exactly — verified by size alignment tests.

## Patch View Mode

The patch view mode renders a small curved rectangular patch (~15% of the sphere at the
default size; `SpherePatch::sphere_fraction`)
instead of the full bubble. This provides a focused view of branched flow effects.

**One UV convention everywhere:** `u = (atan2(z, x) + π) / 2π` (u = 0.5 at +x),
`v = acos(y) / π`. In Rust it is `unit_sphere_to_uv` / `uv_to_unit_sphere` in
`physics/geometry.rs`, used by `SpherePatch` and the sphere mesh; in WGSL it is `normal_to_uv`
/ `uv_to_sphere` (`branched_flow_compute.wgsl`), which deposits branched flow by UV. The
fragment shader samples that texture by the mesh UV (`in.uv`), so a mesh built with another
convention is drawn where the deposits are not: the patch mesh used `φ = 2πu` and rendered
half a turn away (#46). It does not re-derive UV from the normal, which on the deformed
(oblate) mesh is not the vertex's UV. The geometry tests tie mesh UV to the convention (CI);
`tests/wgsl_uv_convention.rs` runs the compute shader's functions on the GPU against the Rust
helpers (`#[ignore]`, lavapipe via `scripts/test-local.sh`; CI does not run it).

**Key insight — rays are spawned within the patch region:**
- The `SpherePatch` struct generates a curved mesh from UV bounds on the sphere
- Rays move in the gnomonic chart of a chart origin: the laser entry in full-sphere view, the
  patch centre in patch view (`BranchedFlowParams::chart_origin`, mirrored in the shader)
- In patch mode every ray starts at a point of the patch UV rectangle (`map_to_patch`),
  converted to its gnomonic chart coordinate. The injection point and beam spread apply only
  to the full-sphere view; the beam angle (`set_beam_angle`, from east toward south at the
  chart origin) applies to both. Measured on lavapipe, one frame, patch at u = 0.5: 90% of
  the patch texture lit, against 4.8% (and none of the left half) before #46. The GPU tests
  keep the patch at u = 0.5 on purpose: at the default u = 0.75 the patch centre is the
  laser entry, so a wrong chart origin would not show. The test checks that rays reach the
  whole patch, not that branching is visible (straight, unscattered rays score 94.5%)
- Larger patches lose their outer columns: rays starting 84° or more from the patch centre
  are not traced, and the loop stops rays beyond chart radius 2.5 (68°). Lit share, one frame:
  half size 0.158 → 90%, 0.20 → 80%, 0.25 → 61%, 0.30 → 47% (the slider goes to 0.3).
  Exponential-map propagation (#47) would remove both limits
- The default patch centre is u = 0.75, v = 0.5 (+z): it faces the default camera and holds
  the default laser entry. Visual check on Dozen (`probe-adapters.sh --app 45 --compute
  --screenshot-after 15`): filaments show on the upstream part of the patch, the downstream
  part renders white. Whether that white is thick drained film or clipped deposits is not
  settled yet (#47 owns the deposit scale and intensity mapping)
- Deposits only occur when ray UV position is within patch bounds
- The fragment shader remaps patch-local UVs when sampling the branched flow texture

**Attempted optimizations that caused GPU freezes (DO NOT RETRY):**
- Modifying the propagation loop body (early break, entry point change mid-loop)
- The patch ray spawning is safe because it only changes initial conditions (#46 changed only
  code before the loop; the loop body stayed byte-identical)

**Scatterers:** When patch mode is active, scatterers are confined within the patch
bounds for higher density coverage. Each scatterer drifts with its own zero-mean
Ornstein-Uhlenbeck velocity (RMS 0.021 UV per film-second per axis, correlation time 3 s),
advanced by elapsed `film_time`, so it stops while the film is paused. Scatterers reflect
at the patch edges and the poles; u wraps in full-sphere mode. Their randomness comes
from an integer hash (`unit_random`), because Rust's `f32::fract` keeps the sign. Before
#47, `fract` made every scatterer drift -0.0005 UV per frame toward the patch's
(min u, min v) corner, made only 24% repulsive, and spread strength and radius over
0.4-1.2 instead of 0.8-1.2.

## Workflow Diagram (putior)

The project uses [putior](https://github.com/pjt222/putior) annotations (`// put id:...`)
in source files to generate a Mermaid data-flow diagram. 40 annotations across 24 files
(Rust + WGSL) describe the config→CPU→GPU→render→export pipeline.

**Regenerate after changing annotations:**
```bash
Rscript scripts/generate_workflow.R
```

Then copy the updated Mermaid block from `workflow_diagram.md` into `README.md`
between the `<!-- PUTIOR-WORKFLOW-START -->` / `<!-- PUTIOR-WORKFLOW-END -->` sentinels.

**Note:** Requires putior >= 0.2.0.9000 for native `.wgsl` file support.

## Configuration

Parameters in `config/default.json`:
- `bubble.diameter`: Bubble size in meters (default: 0.05 = 5cm)
- `bubble.film_thickness_nm`: Initial film thickness (default: 500nm)
- `bubble.refractive_index`: Soap film index (default: 1.33)
- `fluid.*`: Viscosity, surface tension, density (for future drainage sim)
- `resolution`: Grid resolution for thickness field
