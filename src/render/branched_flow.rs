//! Branched flow simulation using GPU compute
//! Traces light rays through the soap film, bending based on thickness gradients
//!
//! Uses a hybrid model combining:
//! - GRIN optics: Rays bend toward thicker regions along the gradient of ln n_eff, the
//!   effective index of light guided in the film (#45, [`crate::physics::film_waveguide`])
//! - Particle scattering: Discrete scatterers (micelles) create local deflections
//!
//! The particle scattering is key for creating tree-like branches (caustics)
//! rather than parallel bands from smooth GRIN alone.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

use crate::physics::film_waveguide::ln_effective_index_table;
use crate::physics::geometry::uv_to_unit_sphere;

/// Maximum number of scatterers supported
pub const MAX_SCATTERERS: u32 = 2048;

/// Spatial hash grid dimensions (must match WGSL constants)
const GRID_SIZE_U: u32 = 10;
const GRID_SIZE_V: u32 = 10;
const GRID_CELL_SIZE: f32 = 0.1;
/// Total cells + 1 for prefix-sum array
const CELL_OFFSETS_LEN: usize = (GRID_SIZE_U * GRID_SIZE_V + 1) as usize;

/// GPU-compatible scatterer data for particle-based ray deflection
/// Represents micelle clusters that scatter light within the soap film
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ScattererGPU {
    /// U position in UV space (0-1)
    pub pos_u: f32,
    /// V position in UV space (0-1)
    pub pos_v: f32,
    /// Scattering strength (signed: positive repels, negative attracts)
    pub strength: f32,
    /// Precomputed 1/(2σ²) for efficient Gaussian evaluation
    pub inv_sigma_sq: f32,
}

impl Default for ScattererGPU {
    fn default() -> Self {
        Self {
            pos_u: 0.5,
            pos_v: 0.5,
            strength: 0.0,
            inv_sigma_sq: 1.0,
        }
    }
}

/// Parameters for branched flow simulation
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BranchedFlowParams {
    /// Light entry point on sphere (normalized direction)
    pub entry_point: [f32; 3],
    /// Initial beam direction (tangent to sphere)
    pub beam_dir: [f32; 3],
    /// Number of rays to trace
    pub num_rays: u32,
    /// Steps per ray
    pub ray_steps: u32,
    /// Step size for ray marching
    pub step_size: f32,
    /// Multiplier on the ray kick, applied to both forces: the GRIN force (the surface
    /// gradient of ln n_eff, weighted by `1 - particle_weight`) and the scatterer force
    /// (weighted by `particle_weight`). With `particle_weight = 0`, 1.0 is the physical
    /// GRIN curvature near the chart origin (#45).
    pub bend_strength: f32,
    /// Initial beam spread angle (radians)
    pub spread_angle: f32,
    /// Intensity falloff per step
    pub intensity_falloff: f32,
    /// Output texture dimensions
    pub tex_width: u32,
    pub tex_height: u32,
    /// Seeds the ray start positions ([`ray_seeds`]). It advances on every frame on which
    /// film time moved, so a paused film re-traces the same rays ([`RaySeedClock`]).
    /// [`BranchedFlowSimulator::update_params`] sets it; it takes the slot of the former
    /// `time` field, which the shader never read.
    pub ray_seed_index: u32,
    /// Converts the drainage thickness (metres) to micrometres, the unit of the shader's
    /// ln n_eff table: 1e6. Unused before #45, which left the GRIN force in metres.
    pub thickness_scale: f32,
    /// Base film thickness in nanometers (synced from BubbleUniform)
    pub base_thickness_nm: f32,
    /// Amplitude for noise/swirl patterns (synced from BubbleUniform)
    pub swirl_intensity: f32,
    /// Gravity drainage rate (synced from BubbleUniform)
    pub drainage_speed: f32,
    /// Noise coordinate scaling (synced from BubbleUniform)
    pub pattern_scale: f32,
    // === Particle scattering parameters ===
    /// Number of scatterers (micelle clusters) active
    pub num_scatterers: u32,
    /// Base scattering strength (V0 magnitude)
    pub scatterer_strength: f32,
    /// Scatterer radius σ in UV space (correlation length)
    pub scatterer_radius: f32,
    /// Blend factor: 0 = pure GRIN, 1 = pure particle scattering
    pub particle_weight: f32,
    // === Patch view mode parameters ===
    /// Whether patch view mode is enabled (0 = full sphere, 1 = patch only)
    pub patch_enabled: u32,
    /// Patch center U coordinate (0-1)
    pub patch_center_u: f32,
    /// Patch center V coordinate (0-1)
    pub patch_center_v: f32,
    /// Patch half-size in UV space (same for both axes)
    pub patch_half_size: f32,
}

impl Default for BranchedFlowParams {
    fn default() -> Self {
        let mut params = Self {
            // Entry point: front of bubble
            entry_point: [0.0, 0.0, 1.0],
            // Set from DEFAULT_BEAM_ANGLE_DEG at the chart origin below
            beam_dir: [0.0; 3],
            // Rays per frame. With 200 steps this is 8x fewer ray-steps than the earlier
            // 32768 x 400, chosen for the WSL CPU rasteriser (llvmpipe); the frame-time gain was
            // not measured. Ray start positions change every frame (ray_seeds, #47), so the
            // 0.85 fade accumulates rays from earlier frames' start positions too; how much
            // that makes up for the cut was not measured. Ray count per adapter type: #37.
            num_rays: 8192,
            // Steps per ray. The adaptive step factor sits at its 0.3 floor for almost every step
            // with the default scatterer field, so range is about ray_steps * step_size * 0.3 (#47).
            ray_steps: 200,
            // Small steps for smooth ray paths
            step_size: 0.005,
            // Also scales the scatterer kick, which this value was tuned for. With the default
            // particle_weight the GRIN kick is 5.0 * 0.9 = 4.5x the physical one (#45). The
            // live thickness field stays uniform until #43, so GRIN is zero at runtime now.
            bend_strength: 5.0,
            // Beam width (now controls position spread, not angle spread)
            spread_angle: 0.4,
            // Low falloff so rays travel far
            intensity_falloff: 0.001,
            // Higher resolution for smoother branches
            tex_width: 512,
            tex_height: 256,
            ray_seed_index: 0,
            thickness_scale: 1e6,
            base_thickness_nm: 500.0,
            swirl_intensity: 1.0,
            drainage_speed: 1.0,
            pattern_scale: 1.0,
            // Particle scattering defaults
            num_scatterers: 400,
            scatterer_strength: 0.5,
            scatterer_radius: 0.03,
            particle_weight: 0.1,
            // Patch view mode defaults (enabled, centred at +z facing the default camera)
            patch_enabled: 1,
            patch_center_u: 0.75,
            patch_center_v: 0.5,
            patch_half_size: 0.158,
        };
        params.beam_dir = beam_direction(params.chart_origin(), DEFAULT_BEAM_ANGLE_DEG).into();
        params
    }
}

impl BranchedFlowParams {
    /// The point whose gnomonic chart the rays move in: the patch centre in patch mode,
    /// the laser entry point otherwise. Mirrors `entry_point` in the `main` kernel of
    /// `branched_flow_compute.wgsl` (#46).
    pub fn chart_origin(&self) -> Vec3 {
        if self.patch_enabled != 0 {
            uv_to_unit_sphere(self.patch_center_u, self.patch_center_v)
        } else {
            Vec3::from(self.entry_point).normalize()
        }
    }
}

/// Default beam angle. At the default laser entry (0, 0, 1) it reproduces the beam
/// direction (-0.5, -0.866, 0) that was hard-coded before #46.
pub const DEFAULT_BEAM_ANGLE_DEG: f32 = 60.0;

/// Tangent basis at a chart origin, built exactly as `branched_flow_compute.wgsl` builds
/// it: `tangent1 = normalize(origin x up)`, `tangent2 = normalize(origin x tangent1)`,
/// with `up = +Y` unless the origin is within about 8 degrees of a pole, then `+X`.
/// Away from the poles `tangent1` points east (increasing u) and `tangent2` south
/// (increasing v).
pub fn chart_tangents(origin: Vec3) -> (Vec3, Vec3) {
    let up = if origin.y.abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let tangent1 = origin.cross(up).normalize();
    let tangent2 = origin.cross(tangent1).normalize();
    (tangent1, tangent2)
}

/// Beam direction at `origin`, `angle_deg` measured from `tangent1` (east) toward
/// `tangent2` (south). Tangent to the sphere and of unit length at every origin, unlike
/// the fixed world vector it replaces, whose tangent projection vanished at two entry
/// points (#46).
pub fn beam_direction(origin: Vec3, angle_deg: f32) -> Vec3 {
    let (tangent1, tangent2) = chart_tangents(origin);
    let angle = angle_deg.to_radians();
    tangent1 * angle.cos() + tangent2 * angle.sin()
}

/// Optional patch bounds for confining scatterers
#[derive(Debug, Clone, Copy)]
pub struct PatchBounds {
    pub center_u: f32,
    pub center_v: f32,
    pub half_size: f32,
}

impl PatchBounds {
    /// Map a 0-1 coordinate to within the patch bounds
    fn map_to_patch(&self, u: f32, v: f32) -> (f32, f32) {
        let min_u = (self.center_u - self.half_size).max(0.0);
        let min_v = (self.center_v - self.half_size).max(0.0);
        let max_u = (self.center_u + self.half_size).min(1.0);
        let max_v = (self.center_v + self.half_size).min(1.0);
        let mapped_u = min_u + u * (max_u - min_u);
        let mapped_v = min_v + v * (max_v - min_v);
        (mapped_u, mapped_v)
    }
}

/// PCG-style 32-bit integer hash (Jarzynski and Olano 2020, "Hash Functions for GPU
/// Rendering"), used for all scatterer randomness. The `sin(x) * 43758.547` hashes it
/// replaces went through Rust's `f32::fract`, which keeps the sign, so they returned
/// values in (-1, 1) instead of [0, 1) (#47).
fn pcg_hash(input: u32) -> u32 {
    let state = input.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// Uniform value in [0, 1) (never 1.0: 24 bits over 2^24) for scatterer `index` in an
/// independent random `stream` at time `step`
fn unit_random(index: u32, stream: u32, step: u32) -> f32 {
    let hash = pcg_hash(index ^ pcg_hash(stream.wrapping_add(pcg_hash(step))));
    (hash >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// Random streams for scatterer generation and drift
const STREAM_JITTER_U: u32 = 1;
const STREAM_JITTER_V: u32 = 2;
const STREAM_SIGN: u32 = 3;
const STREAM_STRENGTH: u32 = 4;
const STREAM_RADIUS: u32 = 5;
const STREAM_DRIFT_RADIUS: u32 = 6;
const STREAM_DRIFT_ANGLE: u32 = 7;
/// Random streams for the ray start positions. `ray_seeds` in branched_flow_compute.wgsl
/// writes them as the literals 8u and 9u.
const STREAM_RAY_START_X: u32 = 8;
const STREAM_RAY_START_Y: u32 = 9;

/// The two start-position random numbers, each in [0, 1), of ray `ray_index` under seed
/// `ray_seed_index`: across and along the beam in full-sphere mode, u and v in the patch.
/// The compute shader's `ray_seeds` is a port of this function. The seed advances on
/// every frame on which film time moved ([`RaySeedClock`]), so the 0.85 fade adds rays
/// from new start positions while the film plays. Before #47 the shader seeded from
/// `ray_index` alone, so every frame re-traced the same rays.
pub fn ray_seeds(ray_index: u32, ray_seed_index: u32) -> [f32; 2] {
    [
        unit_random(ray_index, STREAM_RAY_START_X, ray_seed_index),
        unit_random(ray_index, STREAM_RAY_START_Y, ray_seed_index),
    ]
}

/// Picks the ray seed for each frame: it advances when film time differs from the
/// previous frame's, so a paused film re-traces the same rays over the same (still)
/// scatterers and the bubble stands still. A reset, film time going backwards, also
/// advances it. The index wraps at u32::MAX.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct RaySeedClock {
    index: u32,
    /// Film time of the previous frame; `None` before the first
    film_time: Option<f32>,
}

impl RaySeedClock {
    /// The seed for a frame at `film_time`
    pub fn seed_for(&mut self, film_time: f32) -> u32 {
        if self.film_time.is_some_and(|previous| previous != film_time) {
            self.index = self.index.wrapping_add(1);
        }
        self.film_time = Some(film_time);
        self.index
    }
}

/// Scatterer drift (#47). Each scatterer's velocity follows an Ornstein-Uhlenbeck process:
/// zero mean, exponentially correlated, so it moves smoothly and changes direction over
/// a few seconds without a net drift. RMS velocity per axis in UV per film-second;
/// sqrt(2) * 0.021 = 0.03, about one default scatterer radius per second.
const DRIFT_SPEED_PER_AXIS: f32 = 0.021;
/// Correlation time of the drift velocity in film-seconds
const DRIFT_CORRELATION_TIME: f32 = 3.0;
/// Longest film-time step applied at once. The velocity update is exact for any step and
/// positions move in a straight line within one, so the cap only bounds the jump after a
/// long hitch. At 1 s the drift keeps up with film time down to one frame per
/// film-second, so at film speed 1 it covers the CPU adapter with `--compute` (about 1 fps
/// when last measured).
const MAX_DRIFT_STEP: f32 = 1.0;

/// Two independent standard normal values for scatterer `index` at drift `step`
/// (Box-Muller on two hash streams)
fn gaussian_pair(index: u32, step: u32) -> (f32, f32) {
    let radius = (-2.0 * (1.0 - unit_random(index, STREAM_DRIFT_RADIUS, step)).ln()).sqrt();
    let angle = std::f32::consts::TAU * unit_random(index, STREAM_DRIFT_ANGLE, step);
    (radius * angle.cos(), radius * angle.sin())
}

/// Where scatterers may drift: a UV rectangle, with u wrapping around the sphere in
/// full-sphere mode and reflecting at the edges in patch mode. v always reflects (poles).
#[derive(Debug, Clone, Copy)]
struct DriftBounds {
    min_u: f32,
    max_u: f32,
    min_v: f32,
    max_v: f32,
    wrap_u: bool,
}

/// Folds `value` into [low, high] by reflection at both ends. Returns the folded value
/// and whether it was reflected an odd number of times (the velocity then flips).
fn reflect_into(value: f32, low: f32, high: f32) -> (f32, bool) {
    let span = high - low;
    if span <= 0.0 {
        return (low, false);
    }
    let folded = ((value - low) / span).rem_euclid(2.0);
    if folded > 1.0 {
        (low + (2.0 - folded) * span, true)
    } else {
        (low + folded * span, false)
    }
}

/// Film time to drift by, given the film time of the previous update. Zero while the
/// film is paused (time unchanged) or after it went backwards (a reset); capped at
/// MAX_DRIFT_STEP.
fn drift_time_step(previous_time: f32, time: f32) -> f32 {
    (time - previous_time).clamp(0.0, MAX_DRIFT_STEP)
}

/// Advances each scatterer's Ornstein-Uhlenbeck velocity by `dt` film-seconds (exact
/// update for that step) and moves the scatterer with it inside `bounds`. `step` seeds
/// the noise, so successive calls must pass different steps. Before #47 the motion was
/// `(fract(sin(...) * 43758.547) - 0.5) * 0.001` per frame, whose sign-preserving fract
/// gave every scatterer a mean step of -0.0005 UV per frame on both axes.
fn drift_scatterers(
    scatterers: &mut [ScattererGPU],
    velocities: &mut [[f32; 2]],
    dt: f32,
    step: u32,
    bounds: DriftBounds,
) {
    let decay = (-dt / DRIFT_CORRELATION_TIME).exp();
    let kick = DRIFT_SPEED_PER_AXIS * (1.0 - decay * decay).sqrt();
    for (index, (scatterer, velocity)) in
        scatterers.iter_mut().zip(velocities.iter_mut()).enumerate()
    {
        let (noise_u, noise_v) = gaussian_pair(index as u32, step);
        velocity[0] = velocity[0] * decay + kick * noise_u;
        velocity[1] = velocity[1] * decay + kick * noise_v;

        let moved_u = scatterer.pos_u + velocity[0] * dt;
        if bounds.wrap_u {
            scatterer.pos_u = moved_u.rem_euclid(1.0);
        } else {
            let (folded, reflected) = reflect_into(moved_u, bounds.min_u, bounds.max_u);
            scatterer.pos_u = folded;
            if reflected {
                velocity[0] = -velocity[0];
            }
        }
        let (folded, reflected) = reflect_into(
            scatterer.pos_v + velocity[1] * dt,
            bounds.min_v,
            bounds.max_v,
        );
        scatterer.pos_v = folded;
        if reflected {
            velocity[1] = -velocity[1];
        }
    }
}

/// Velocities drawn from the drift's stationary distribution, so motion starts at once
fn initial_drift_velocities(count: usize, step: u32) -> Vec<[f32; 2]> {
    (0..count as u32)
        .map(|index| {
            let (noise_u, noise_v) = gaussian_pair(index, step);
            [
                DRIFT_SPEED_PER_AXIS * noise_u,
                DRIFT_SPEED_PER_AXIS * noise_v,
            ]
        })
        .collect()
}

/// What scatterer generation needs from the simulation parameters
#[derive(Debug, Clone, Copy)]
struct ScattererGeneration {
    count: u32,
    strength: f32,
    radius: f32,
    patch_bounds: Option<PatchBounds>,
}

/// Scatterer positions and drift state, separate from the GPU buffers so the CPU update
/// can be tested without a device
#[derive(Debug, Default)]
struct ScattererField {
    /// In generation order: a scatterer's index is its identity for the drift
    scatterers: Vec<ScattererGPU>,
    /// Drift velocity of each scatterer in UV per film-second, same order
    velocities: Vec<[f32; 2]>,
    /// Film time of the last update, for the drift step
    last_time: f32,
    /// Seeds the drift noise; advances with every drift step and regeneration
    drift_step: u32,
}

impl ScattererField {
    /// New scatterers at film time `time`, inside `bounds`, with stationary velocities
    fn regenerate(&mut self, time: f32, generation: ScattererGeneration, bounds: DriftBounds) {
        self.scatterers = generate_scatterers(
            generation.count,
            time,
            generation.strength,
            generation.radius,
            generation.patch_bounds,
        );
        // Generation jitter can place a scatterer slightly outside the patch (up to about
        // 0.006 UV at the default size); fold it in now instead of with a jump on the
        // first drift step
        for scatterer in &mut self.scatterers {
            if !bounds.wrap_u {
                scatterer.pos_u = reflect_into(scatterer.pos_u, bounds.min_u, bounds.max_u).0;
            }
            scatterer.pos_v = reflect_into(scatterer.pos_v, bounds.min_v, bounds.max_v).0;
        }
        self.velocities = initial_drift_velocities(self.scatterers.len(), self.drift_step);
        self.drift_step = self.drift_step.wrapping_add(1);
        self.last_time = time;
    }

    /// Drifts every scatterer by the film time elapsed since the last update: nothing
    /// while the film is paused (see [`drift_time_step`])
    fn advance(&mut self, time: f32, bounds: DriftBounds) {
        let dt = drift_time_step(self.last_time, time);
        self.last_time = time;
        if dt > 0.0 && !self.scatterers.is_empty() {
            drift_scatterers(
                &mut self.scatterers,
                &mut self.velocities,
                dt,
                self.drift_step,
                bounds,
            );
            self.drift_step = self.drift_step.wrapping_add(1);
        }
    }

    /// A copy sorted by grid cell for the spatial hash, and the prefix-sum cell offsets:
    /// `offsets[i]` is the index of the first scatterer in cell i, `offsets[i + 1]` one
    /// past the last. The stored order stays as generated.
    fn sorted_for_upload(&self) -> (Vec<ScattererGPU>, Vec<u32>) {
        let cell_of = |s: &ScattererGPU| {
            let u_cell = (s.pos_u / GRID_CELL_SIZE).clamp(0.0, (GRID_SIZE_U - 1) as f32) as u32;
            let v_cell = (s.pos_v / GRID_CELL_SIZE).clamp(0.0, (GRID_SIZE_V - 1) as f32) as u32;
            (v_cell * GRID_SIZE_U + u_cell) as usize
        };
        let mut sorted = self.scatterers.clone();
        sorted.sort_by_key(cell_of);
        let total_cells = (GRID_SIZE_U * GRID_SIZE_V) as usize;
        let mut cell_offsets = vec![0u32; total_cells + 1];
        for scatterer in &sorted {
            cell_offsets[cell_of(scatterer) + 1] += 1;
        }
        for cell in 1..=total_cells {
            cell_offsets[cell] += cell_offsets[cell - 1];
        }
        (sorted, cell_offsets)
    }
}

/// Generate scatterers using quasi-random distribution with jitter
/// Uses Halton sequence for good spatial coverage
/// If patch_bounds is provided, scatterers are confined within the patch
fn generate_scatterers(
    num: u32,
    time: f32,
    base_strength: f32,
    base_radius: f32,
    patch_bounds: Option<PatchBounds>,
) -> Vec<ScattererGPU> {
    let mut scatterers = Vec::with_capacity(num as usize);

    // Halton sequence bases (coprime for 2D low-discrepancy)
    let base_u = 2;
    let base_v = 3;

    for i in 0..num {
        // Halton sequence for quasi-random position
        let u = halton(i + 1, base_u);
        let v = halton(i + 1, base_v);

        // Time-dependent jitter in (-0.02, 0.02), centred
        let jitter_scale = 0.02;
        let jitter_u = (unit_random(i, STREAM_JITTER_U, time.to_bits()) - 0.5) * 2.0 * jitter_scale;
        let jitter_v = (unit_random(i, STREAM_JITTER_V, time.to_bits()) - 0.5) * 2.0 * jitter_scale;

        let (pos_u, pos_v) = if let Some(bounds) = patch_bounds {
            // Map to patch bounds, then apply jitter within patch
            let (mapped_u, mapped_v) = bounds.map_to_patch(u, v);
            (
                (mapped_u + jitter_u * bounds.half_size * 2.0).clamp(0.0, 1.0),
                (mapped_v + jitter_v * bounds.half_size * 2.0).clamp(0.0, 1.0),
            )
        } else {
            // Full sphere - use modular wrapping
            (
                (u + jitter_u).rem_euclid(1.0),
                (v + jitter_v).rem_euclid(1.0),
            )
        };

        // Sign of strength: positive repels, negative attracts, half each.
        // Before #47 the sign-preserving fract made only about 24% repulsive.
        let sign = if unit_random(i, STREAM_SIGN, 0) < 0.5 {
            1.0
        } else {
            -1.0
        };

        // Strength and radius vary by a factor in [0.8, 1.2) (before #47: 0.4 to 1.2)
        let strength_var = 0.8 + 0.4 * unit_random(i, STREAM_STRENGTH, 0);
        let strength = sign * base_strength * strength_var;
        let radius_var = 0.8 + 0.4 * unit_random(i, STREAM_RADIUS, 0);
        let sigma = (base_radius * radius_var).max(1e-6);
        let inv_sigma_sq = 1.0 / (2.0 * sigma * sigma);

        scatterers.push(ScattererGPU {
            pos_u,
            pos_v,
            strength,
            inv_sigma_sq,
        });
    }

    scatterers
}

/// Halton sequence for quasi-random number generation
/// Returns value in [0, 1) for given index and base
fn halton(index: u32, base: u32) -> f32 {
    let mut result = 0.0f32;
    let mut f = 1.0 / base as f32;
    let mut i = index;

    while i > 0 {
        result += f * (i % base) as f32;
        i /= base;
        f /= base as f32;
    }

    result
}

/// Manages GPU-based branched flow simulation
pub struct BranchedFlowSimulator {
    /// Compute pipeline for ray tracing
    trace_pipeline: wgpu::ComputePipeline,
    /// Compute pipeline for clearing/fading
    clear_pipeline: wgpu::ComputePipeline,
    /// Bind group layout (stored for rebinding when thickness buffer swaps)
    bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group
    bind_group: wgpu::BindGroup,
    /// Parameters buffer
    params_buffer: wgpu::Buffer,
    /// Scatterer buffer (storage buffer for particle positions/strengths, sorted by grid cell)
    scatterer_buffer: wgpu::Buffer,
    /// Cell offsets prefix-sum buffer for spatial hash (GRID_SIZE_U * GRID_SIZE_V + 1 entries)
    cell_offsets_buffer: wgpu::Buffer,
    /// ln n_eff of the film's TE0 mode against thickness, for the GRIN force (#45; see
    /// [`crate::physics::film_waveguide`]). Written once at creation.
    ln_neff_table_buffer: wgpu::Buffer,
    /// Current parameters
    pub params: BranchedFlowParams,
    /// Whether simulation is enabled
    pub enabled: bool,
    /// Texture dimensions
    tex_width: u32,
    tex_height: u32,
    /// Dirty flag: whether scatterers need full regeneration
    /// Set when structural parameters change (count, radius, patch), cleared after upload
    scatterers_dirty: bool,
    /// Cached scatterer parameters for dirty check
    last_num_scatterers: u32,
    last_scatterer_strength: f32,
    last_scatterer_radius: f32,
    last_patch_enabled: u32,
    last_patch_center_u: f32,
    last_patch_center_v: f32,
    last_patch_half_size: f32,
    /// Scatterer positions and drift state; a sorted copy is uploaded each frame
    scatterer_field: ScattererField,
    /// Beam angle in degrees (see [`beam_direction`]); `params.beam_dir` is derived from it
    /// at the current chart origin before every upload
    beam_angle_deg: f32,
    /// Seed of the ray start positions, uploaded as `params.ray_seed_index`
    ray_seed_clock: RaySeedClock,
}

/// Create a branched flow texture buffer (called early in pipeline init)
pub fn create_branched_flow_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    // Higher resolution for smoother branches
    let tex_width = 512u32;
    let tex_height = 256u32;
    let tex_size = (tex_width * tex_height) as usize;
    let caustic_data = vec![0u32; tex_size];

    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Branched Flow Texture Buffer"),
        contents: bytemuck::cast_slice(&caustic_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

impl BranchedFlowSimulator {
    /// Create a new branched flow simulator
    /// Takes external buffers: thickness_buffer from GPU drainage, caustic_buffer from early init
    // put id:'gpu_compute_branched_init', label:'Init branched flow', input:'gpu_device.internal', output:'branched_flow_texture_gpu.internal'
    pub fn new(
        device: &wgpu::Device,
        thickness_buffer: &wgpu::Buffer,
        caustic_buffer: &wgpu::Buffer,
    ) -> Self {
        let params = BranchedFlowParams::default();
        let tex_width = params.tex_width;
        let tex_height = params.tex_height;

        // Verify dimensions match the hardcoded constants in bubble.wgsl
        // (BRANCHED_TEX_WIDTH = 512, BRANCHED_TEX_HEIGHT = 256)
        assert_eq!(
            tex_width, 512,
            "BranchedFlowParams tex_width must match BRANCHED_TEX_WIDTH in bubble.wgsl"
        );
        assert_eq!(
            tex_height, 256,
            "BranchedFlowParams tex_height must match BRANCHED_TEX_HEIGHT in bubble.wgsl"
        );

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Branched Flow Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/branched_flow_compute.wgsl").into(),
            ),
        });

        // Create bind group layout (stored for rebinding when thickness buffer swaps)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Branched Flow Bind Group Layout"),
            entries: &[
                // Params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Thickness field (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Caustic texture (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Scatterers array (read-only, sorted by grid cell)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Cell offsets prefix-sum array for spatial hash (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // ln n_eff table for the GRIN force (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Branched Flow Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create trace compute pipeline
        let trace_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Branched Flow Trace Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create clear compute pipeline
        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Branched Flow Clear Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("clear"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create params buffer
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Branched Flow Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create scatterer buffer (max 2048 scatterers × 16 bytes = 32KB)
        // Initialize with default scatterers
        let initial_scatterers = generate_scatterers(
            params.num_scatterers,
            0.0,
            params.scatterer_strength,
            params.scatterer_radius,
            None, // No patch bounds on initial creation
        );
        // Pad to MAX_SCATTERERS to avoid buffer resizing
        let mut scatterer_data = initial_scatterers;
        scatterer_data.resize(MAX_SCATTERERS as usize, ScattererGPU::default());

        let scatterer_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Branched Flow Scatterer Buffer"),
            contents: bytemuck::cast_slice(&scatterer_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Create cell offsets prefix-sum buffer for spatial hash
        let initial_offsets = vec![0u32; CELL_OFFSETS_LEN];
        let cell_offsets_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Branched Flow Cell Offsets Buffer"),
            contents: bytemuck::cast_slice(&initial_offsets),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let ln_neff_table_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Branched Flow ln n_eff Table Buffer"),
            contents: bytemuck::cast_slice(&ln_effective_index_table()),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Create bind group using the external caustic buffer
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Branched Flow Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: thickness_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: caustic_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scatterer_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cell_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: ln_neff_table_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            trace_pipeline,
            clear_pipeline,
            bind_group_layout,
            bind_group,
            params_buffer,
            scatterer_buffer,
            cell_offsets_buffer,
            ln_neff_table_buffer,
            params,
            enabled: false,
            tex_width,
            tex_height,
            // Initialize dirty flag to true so first frame uploads scatterers
            scatterers_dirty: true,
            last_num_scatterers: params.num_scatterers,
            last_scatterer_strength: params.scatterer_strength,
            last_scatterer_radius: params.scatterer_radius,
            last_patch_enabled: params.patch_enabled,
            last_patch_center_u: params.patch_center_u,
            last_patch_center_v: params.patch_center_v,
            last_patch_half_size: params.patch_half_size,
            scatterer_field: ScattererField::default(),
            beam_angle_deg: DEFAULT_BEAM_ANGLE_DEG,
            ray_seed_clock: RaySeedClock::default(),
        }
    }

    /// Update parameters buffer for a frame at `film_time`. Recomputes `beam_dir` first,
    /// because the chart origin it is tangent to moves with the laser entry and the patch,
    /// and picks the ray seed, which advances only if film time moved ([`RaySeedClock`]).
    /// The write lands at the next queue submission, so call this once per frame before
    /// [`Self::step`] and submit each frame on its own: two frames recorded into one
    /// submission would both trace with the last value written.
    pub fn update_params(&mut self, queue: &wgpu::Queue, film_time: f32) {
        self.update_beam_direction();
        self.params.ray_seed_index = self.ray_seed_clock.seed_for(film_time);
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
    }

    fn update_beam_direction(&mut self) {
        self.params.beam_dir =
            beam_direction(self.params.chart_origin(), self.beam_angle_deg).into();
    }

    /// Rebuild bind group with the current thickness buffer.
    /// Must be called each frame when GPU drainage is active, because the drainage
    /// simulator double-buffers and the "current" buffer alternates after each step.
    pub fn rebuild_bind_group(
        &mut self,
        device: &wgpu::Device,
        thickness_buffer: &wgpu::Buffer,
        caustic_buffer: &wgpu::Buffer,
    ) {
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Branched Flow Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: thickness_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: caustic_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.scatterer_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.cell_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.ln_neff_table_buffer.as_entire_binding(),
                },
            ],
        });
    }

    /// Check if scatterer parameters have changed and mark dirty if so
    pub fn check_scatterer_params_changed(&mut self) {
        let changed = self.params.num_scatterers != self.last_num_scatterers
            || (self.params.scatterer_strength - self.last_scatterer_strength).abs() > 1e-6
            || (self.params.scatterer_radius - self.last_scatterer_radius).abs() > 1e-6
            || self.params.patch_enabled != self.last_patch_enabled
            || (self.params.patch_center_u - self.last_patch_center_u).abs() > 1e-6
            || (self.params.patch_center_v - self.last_patch_center_v).abs() > 1e-6
            || (self.params.patch_half_size - self.last_patch_half_size).abs() > 1e-6;

        if changed {
            self.scatterers_dirty = true;
            self.last_num_scatterers = self.params.num_scatterers;
            self.last_scatterer_strength = self.params.scatterer_strength;
            self.last_scatterer_radius = self.params.scatterer_radius;
            self.last_patch_enabled = self.params.patch_enabled;
            self.last_patch_center_u = self.params.patch_center_u;
            self.last_patch_center_v = self.params.patch_center_v;
            self.last_patch_half_size = self.params.patch_half_size;
        }
    }

    /// The region scatterers drift in: the patch rectangle in patch mode, the whole sphere
    /// (u wrapping) otherwise
    fn drift_bounds(&self) -> DriftBounds {
        if self.params.patch_enabled != 0 {
            let half_size = self.params.patch_half_size;
            DriftBounds {
                min_u: (self.params.patch_center_u - half_size).max(0.0),
                max_u: (self.params.patch_center_u + half_size).min(1.0),
                min_v: (self.params.patch_center_v - half_size).max(0.0),
                max_v: (self.params.patch_center_v + half_size).min(1.0),
                wrap_u: false,
            }
        } else {
            DriftBounds {
                min_u: 0.0,
                max_u: 1.0,
                min_v: 0.0,
                max_v: 1.0,
                wrap_u: true,
            }
        }
    }

    /// Update scatterer positions with temporal coherence.
    ///
    /// When structural parameters change (count, radius, patch bounds), scatterers are
    /// fully regenerated. Otherwise each scatterer drifts with its own smooth, zero-mean
    /// velocity (see [`drift_scatterers`]) by the film time elapsed since the last update,
    /// so the pattern evolves continuously and stops while the film is paused. Scatterers
    /// reflect at the patch edges and the poles; in full-sphere mode u wraps.
    ///
    /// Uploads a copy sorted by grid cell, with prefix-sum cell_offsets for O(k) GPU lookup.
    pub fn update_scatterers(&mut self, queue: &wgpu::Queue, time: f32) {
        // Check if parameters changed since last upload
        self.check_scatterer_params_changed();

        let bounds = self.drift_bounds();
        if self.scatterers_dirty {
            // Full regeneration: structural parameters changed
            let patch_bounds = (self.params.patch_enabled != 0).then_some(PatchBounds {
                center_u: self.params.patch_center_u,
                center_v: self.params.patch_center_v,
                half_size: self.params.patch_half_size,
            });
            let generation = ScattererGeneration {
                count: self.params.num_scatterers.min(MAX_SCATTERERS),
                strength: self.params.scatterer_strength,
                radius: self.params.scatterer_radius,
                patch_bounds,
            };
            self.scatterer_field.regenerate(time, generation, bounds);
            self.scatterers_dirty = false;
        } else if !self.scatterer_field.scatterers.is_empty() {
            self.scatterer_field.advance(time, bounds);
        } else {
            return; // No scatterers to update
        }
        let (sorted_scatterers, cell_offsets) = self.scatterer_field.sorted_for_upload();

        // Upload sorted scatterers and cell offsets
        queue.write_buffer(
            &self.scatterer_buffer,
            0,
            bytemuck::cast_slice(&sorted_scatterers),
        );
        queue.write_buffer(
            &self.cell_offsets_buffer,
            0,
            bytemuck::cast_slice(&cell_offsets),
        );
    }

    /// Force scatterer regeneration on next update (e.g., for animation)
    pub fn mark_scatterers_dirty(&mut self) {
        self.scatterers_dirty = true;
    }

    /// Set laser entry point (spherical coordinates: azimuth, elevation in degrees)
    pub fn set_entry_point(&mut self, azimuth_deg: f32, elevation_deg: f32) {
        let azimuth = azimuth_deg.to_radians();
        let elevation = elevation_deg.to_radians();
        self.params.entry_point = [
            elevation.cos() * azimuth.cos(),
            elevation.sin(),
            elevation.cos() * azimuth.sin(),
        ];
    }

    /// Set the beam angle in degrees, measured from east toward south at the chart origin
    /// (see [`beam_direction`])
    pub fn set_beam_angle(&mut self, angle_deg: f32) {
        self.beam_angle_deg = angle_deg;
        self.update_beam_direction();
    }

    /// Current beam angle in degrees
    pub fn beam_angle_deg(&self) -> f32 {
        self.beam_angle_deg
    }

    /// Run simulation step with optional GPU timestamp profiling.
    // put id:'gpu_compute_branched_step', label:'Dispatch branched flow rays', input:'uniform_buffers_gpu.internal', output:'branched_flow_texture_gpu.internal'
    pub fn step(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        clear_timestamps: Option<wgpu::ComputePassTimestampWrites<'_>>,
        trace_timestamps: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        if !self.enabled {
            return;
        }

        // Clear/fade pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Branched Flow Clear Pass"),
                timestamp_writes: clear_timestamps,
            });
            pass.set_pipeline(&self.clear_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let workgroups_x = self.tex_width.div_ceil(16);
            let workgroups_y = self.tex_height.div_ceil(16);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Ray trace pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Branched Flow Trace Pass"),
                timestamp_writes: trace_timestamps,
            });
            pass.set_pipeline(&self.trace_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let workgroups = self.params.num_rays.div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }

    /// Get texture dimensions
    pub fn texture_size(&self) -> (u32, u32) {
        (self.tex_width, self.tex_height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::film_waveguide;

    #[test]
    fn params_struct_size_matches_wgsl_layout() {
        // BranchedFlowParams must be 112 bytes (28 f32/u32 fields)
        // to match the WGSL uniform struct layout
        assert_eq!(
            std::mem::size_of::<BranchedFlowParams>(),
            28 * 4, // 28 fields * 4 bytes each
            "BranchedFlowParams size must match WGSL uniform layout (112 bytes)"
        );
    }

    #[test]
    fn params_field_offsets_match_wgsl() {
        // Verify critical field offsets match the WGSL struct declaration.
        // BranchedFlowParams uses [f32; 3] arrays for entry_point and beam_dir
        // which map to 6 scalar fields in WGSL — offsets must stay in sync.
        use std::mem::offset_of;

        assert_eq!(offset_of!(BranchedFlowParams, entry_point), 0);
        assert_eq!(offset_of!(BranchedFlowParams, beam_dir), 12);
        assert_eq!(offset_of!(BranchedFlowParams, num_rays), 24);
        assert_eq!(offset_of!(BranchedFlowParams, ray_steps), 28);
        assert_eq!(offset_of!(BranchedFlowParams, step_size), 32);
        assert_eq!(offset_of!(BranchedFlowParams, bend_strength), 36);
        assert_eq!(offset_of!(BranchedFlowParams, tex_width), 48);
        assert_eq!(offset_of!(BranchedFlowParams, tex_height), 52);
        assert_eq!(offset_of!(BranchedFlowParams, ray_seed_index), 56);
        assert_eq!(offset_of!(BranchedFlowParams, thickness_scale), 60);
        assert_eq!(offset_of!(BranchedFlowParams, num_scatterers), 80);
        assert_eq!(offset_of!(BranchedFlowParams, patch_enabled), 96);
    }

    #[test]
    fn set_entry_point_produces_normalized_vectors() {
        // Test various angles
        let test_angles: [(f32, f32); 4] =
            [(0.0, 0.0), (45.0, 30.0), (180.0, -45.0), (270.0, 89.0)];
        for (azimuth, elevation) in test_angles {
            let az_rad = azimuth.to_radians();
            let el_rad = elevation.to_radians();
            let entry = [
                el_rad.cos() * az_rad.cos(),
                el_rad.sin(),
                el_rad.cos() * az_rad.sin(),
            ];
            let length = (entry[0] * entry[0] + entry[1] * entry[1] + entry[2] * entry[2]).sqrt();
            assert!(
                (length - 1.0).abs() < 1e-5,
                "Entry point not normalized for azimuth={azimuth}, elevation={elevation}: length={length}"
            );
        }
    }

    /// Laser entry point for azimuth/elevation in degrees, as set_entry_point computes it
    fn entry_from_angles(azimuth_deg: f32, elevation_deg: f32) -> Vec3 {
        let mut params = BranchedFlowParams::default();
        let (azimuth, elevation) = (azimuth_deg.to_radians(), elevation_deg.to_radians());
        params.entry_point = [
            elevation.cos() * azimuth.cos(),
            elevation.sin(),
            elevation.cos() * azimuth.sin(),
        ];
        params.patch_enabled = 0;
        params.chart_origin()
    }

    #[test]
    fn beam_direction_is_a_unit_tangent_at_every_origin() {
        let origins = [
            // The two entries where the former fixed beam (-0.5, -0.866, 0) had no
            // tangent component (#46)
            entry_from_angles(0.0, 60.0),
            entry_from_angles(180.0, -60.0),
            entry_from_angles(90.0, 0.0),
            entry_from_angles(-135.0, 89.0),
            Vec3::Y,
            Vec3::NEG_Y,
            uv_to_unit_sphere(0.5, 0.5),
            uv_to_unit_sphere(0.1, 0.2),
        ];
        for origin in origins {
            for angle_deg in [0.0f32, 45.0, 60.0, 90.0, 180.0, 270.0] {
                let beam = beam_direction(origin, angle_deg);
                assert!(
                    beam.dot(origin).abs() < 1e-5 && (beam.length() - 1.0).abs() < 1e-5,
                    "beam {beam:?} at origin {origin:?}, angle {angle_deg}"
                );
            }
        }
    }

    #[test]
    fn default_beam_angle_reproduces_the_former_beam_at_the_default_entry() {
        let beam = beam_direction(Vec3::Z, DEFAULT_BEAM_ANGLE_DEG);
        assert!(
            (beam - Vec3::new(-0.5, -0.866_025_4, 0.0)).length() < 1e-5,
            "{beam:?}"
        );
    }

    #[test]
    fn chart_tangents_point_east_and_south() {
        use crate::physics::geometry::unit_sphere_to_uv;
        for (u, v) in [(0.5, 0.5), (0.3, 0.3), (0.8, 0.7)] {
            let origin = uv_to_unit_sphere(u, v);
            let (tangent1, tangent2) = chart_tangents(origin);
            let [u_east, v_east] = unit_sphere_to_uv((origin + tangent1 * 1e-3).normalize());
            let [u_south, v_south] = unit_sphere_to_uv((origin + tangent2 * 1e-3).normalize());
            assert!(
                u_east > u && (v_east - v).abs() < 1e-4,
                "tangent1 not east at ({u}, {v})"
            );
            assert!(
                v_south > v && (u_south - u).abs() < 1e-4,
                "tangent2 not south at ({u}, {v})"
            );
        }
    }

    #[test]
    fn chart_origin_is_the_patch_centre_in_patch_mode() {
        let mut params = BranchedFlowParams {
            patch_center_u: 0.3,
            patch_center_v: 0.6,
            ..BranchedFlowParams::default()
        };
        assert!((params.chart_origin() - uv_to_unit_sphere(0.3, 0.6)).length() < 1e-6);
        params.patch_enabled = 0;
        assert!((params.chart_origin() - Vec3::from(params.entry_point)).length() < 1e-6);
    }

    #[test]
    fn default_beam_is_tangent_at_the_default_chart_origin() {
        let params = BranchedFlowParams::default();
        let beam = Vec3::from(params.beam_dir);
        assert!(beam.dot(params.chart_origin()).abs() < 1e-6);
        assert!((beam.length() - 1.0).abs() < 1e-6);
    }

    fn test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()
    }

    /// Share of lit texels in the deposit texture, overall and per half
    #[derive(Debug)]
    struct DepositCoverage {
        lit: f64,
        left_half: f64,
        right_half: f64,
        top_half: f64,
        bottom_half: f64,
    }

    impl DepositCoverage {
        fn of(texels: &[u32], width: usize, height: usize) -> Self {
            let lit_share = |columns: std::ops::Range<usize>, rows: std::ops::Range<usize>| {
                let total = columns.len() * rows.len();
                let lit = rows
                    .flat_map(|row| columns.clone().map(move |column| row * width + column))
                    .filter(|&index| texels[index] > 0)
                    .count();
                lit as f64 / total as f64
            };
            Self {
                lit: lit_share(0..width, 0..height),
                left_half: lit_share(0..width / 2, 0..height),
                right_half: lit_share(width / 2..width, 0..height),
                top_half: lit_share(0..width, 0..height / 2),
                bottom_half: lit_share(0..width, height / 2..height),
            }
        }
    }

    /// Drainage grid size, as in `branched_flow_compute.wgsl` (THICKNESS_WIDTH, _HEIGHT)
    const THICKNESS_COLUMNS: usize = 256;
    const THICKNESS_ROWS: usize = 128;

    /// A uniform 500 nm film in metres, the drainage solver's initial state: no GRIN force,
    /// so only the scatterers bend the rays
    fn uniform_thickness() -> Vec<f32> {
        vec![500e-9f32; THICKNESS_COLUMNS * THICKNESS_ROWS]
    }

    /// Deposit texture after one compute frame of a simulator set up by `configure`,
    /// over the drainage thickness field `thickness` (metres, row-major, rows = theta)
    fn one_frame_deposits(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        thickness: &[f32],
        configure: impl FnOnce(&mut BranchedFlowSimulator),
    ) -> Vec<u32> {
        let run = FrameRun {
            film_times: vec![0.0],
            clear_each_frame: true,
        };
        frame_deposits(device, queue, thickness, configure, run)
            .pop()
            .expect("one frame")
    }

    /// How [`frame_deposits`] runs its frames
    struct FrameRun {
        /// Film time of each frame; a repeated value is a paused film
        film_times: Vec<f32>,
        /// Zero the deposit texture before each frame, so each texture holds that frame's
        /// rays only. Otherwise the 0.85 fade accumulates them, as in the app.
        clear_each_frame: bool,
    }

    /// Deposit texture after each frame of a simulator set up by `configure`, over the
    /// drainage thickness field `thickness` (metres, row-major, rows = theta). Each frame
    /// runs in the app's order (update_params, update_scatterers, step) and is submitted on
    /// its own, because the params write lands at submission.
    fn frame_deposits(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        thickness: &[f32],
        configure: impl FnOnce(&mut BranchedFlowSimulator),
        run: FrameRun,
    ) -> Vec<Vec<u32>> {
        assert_eq!(thickness.len(), THICKNESS_COLUMNS * THICKNESS_ROWS);
        let thickness_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(thickness),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let texture_bytes = 512 * 256 * 4;
        let deposit_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: texture_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: texture_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut simulator = BranchedFlowSimulator::new(device, &thickness_buffer, &deposit_buffer);
        simulator.enabled = true;
        configure(&mut simulator);
        let mut textures = Vec::with_capacity(run.film_times.len());
        for &film_time in &run.film_times {
            simulator.update_params(queue, film_time);
            simulator.update_scatterers(queue, film_time);
            let mut encoder = device.create_command_encoder(&Default::default());
            if run.clear_each_frame {
                encoder.clear_buffer(&deposit_buffer, 0, None);
            }
            simulator.step(&mut encoder, None, None);
            encoder.copy_buffer_to_buffer(&deposit_buffer, 0, &readback_buffer, 0, texture_bytes);
            queue.submit(Some(encoder.finish()));

            readback_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, |result| result.expect("map deposits"));
            device.poll(wgpu::Maintain::Wait);
            textures.push(
                bytemuck::cast_slice::<u8, u32>(&readback_buffer.slice(..).get_mapped_range())
                    .to_vec(),
            );
            readback_buffer.unmap();
        }
        if let Some(error) = pollster::block_on(device.pop_error_scope()) {
            panic!("branched flow frame failed validation: {error}");
        }
        textures
    }

    #[test]
    #[ignore] // Requires GPU (lavapipe works: scripts/test-local.sh -- --ignored)
    fn patch_mode_rays_cover_the_patch() {
        // In patch mode the whole deposit texture maps onto the visible patch. The patch sits
        // at u = 0.5 (the pre-#46 default), 90 degrees from the default laser entry: at the
        // current default (u = 0.75) the patch centre IS the laser entry, so a chart at the
        // wrong origin would go unnoticed. Measured on lavapipe, one frame: before #46 4.8%
        // of texels lit and none in the left half; chart centred on the patch but the old
        // beam-line spawn, 14%; rays starting over the whole patch, 90% (halves 87-93%);
        // with #47's ray seeds 89% (halves 86-92%). This checks that rays REACH the whole
        // patch, not that branching is visible: straight, unscattered rays of the same
        // length scored 94.5% with the earlier seeds.
        let Some((device, queue)) = test_device() else {
            panic!("no GPU adapter (run via scripts/test-local.sh for lavapipe)");
        };
        let texels = one_frame_deposits(&device, &queue, &uniform_thickness(), |simulator| {
            simulator.params.patch_center_u = 0.5;
        });
        let coverage = DepositCoverage::of(&texels, 512, 256);
        println!("patch at u = 0.5, one frame: {coverage:?}");
        assert!(coverage.lit > 0.6, "{coverage:?}");
        for half in [
            coverage.left_half,
            coverage.right_half,
            coverage.top_half,
            coverage.bottom_half,
        ] {
            assert!(
                half > 0.5,
                "a half of the patch is mostly dark: {coverage:?}"
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU (lavapipe works: scripts/test-local.sh -- --ignored)
    fn ray_start_positions_advance_with_film_time() {
        // #47: the start positions used to depend on the ray index alone, so every frame
        // re-traced the same rays. Straight rays (no scatterer force, uniform film) isolate
        // the start positions: their deposits change only if the seeds do. Runs the app's
        // call order, one submission per frame.
        let Some((device, queue)) = test_device() else {
            panic!("no GPU adapter (run via scripts/test-local.sh for lavapipe)");
        };
        let run = FrameRun {
            film_times: vec![0.0, 1.0 / 60.0],
            clear_each_frame: true,
        };
        let frames = frame_deposits(
            &device,
            &queue,
            &uniform_thickness(),
            |simulator| {
                simulator.params.patch_center_u = 0.5;
                simulator.params.particle_weight = 0.0;
            },
            run,
        );
        let lit = frames[0].iter().filter(|&&texel| texel > 0).count();
        let changed = frames[0]
            .iter()
            .zip(&frames[1])
            .filter(|(first, second)| first != second)
            .count();
        println!("straight rays: {lit} texels lit, {changed} changed between frames");
        assert!(lit > 0, "no deposits");
        assert!(
            changed * 2 > lit,
            "only {changed} of {lit} lit texels changed although film time moved"
        );
    }

    #[test]
    #[ignore] // Requires GPU (lavapipe works: scripts/test-local.sh -- --ignored)
    fn a_paused_film_traces_the_same_rays() {
        // A paused bubble stands still: the scatterers stop and the ray seed does not
        // advance, so a frame at the same film time deposits exactly the same texture
        // (integer atomic adds do not depend on their order). The first frame plays, so
        // this also covers the first paused frame after playback.
        let Some((device, queue)) = test_device() else {
            panic!("no GPU adapter (run via scripts/test-local.sh for lavapipe)");
        };
        let run = FrameRun {
            film_times: vec![0.0, 1.0 / 60.0, 1.0 / 60.0],
            clear_each_frame: true,
        };
        let frames = frame_deposits(
            &device,
            &queue,
            &uniform_thickness(),
            |simulator| simulator.params.patch_center_u = 0.5,
            run,
        );
        assert!(frames[1].iter().any(|&texel| texel > 0), "no deposits");
        assert_ne!(frames[0], frames[1], "playing frames should differ");
        assert_eq!(frames[1], frames[2], "the paused frame changed");
    }

    #[test]
    #[ignore] // Requires GPU (lavapipe works: scripts/test-local.sh -- --ignored)
    fn full_sphere_mode_deposits_around_the_laser_entry() {
        let Some((device, queue)) = test_device() else {
            panic!("no GPU adapter (run via scripts/test-local.sh for lavapipe)");
        };
        // Default laser entry (0, 0, 1) is at u = 0.75, v = 0.5
        // The patch centre (u = 0.5) differs from the laser entry, so this fails if the
        // full-sphere chart is centred on the patch instead
        let texels = one_frame_deposits(&device, &queue, &uniform_thickness(), |simulator| {
            simulator.params.patch_enabled = 0;
            simulator.params.patch_center_u = 0.5;
        });
        let (mut weight, mut weighted_u, mut weighted_v) = (0.0f64, 0.0f64, 0.0f64);
        for (index, &texel) in texels.iter().enumerate() {
            let (column, row) = (index % 512, index / 512);
            weight += texel as f64;
            weighted_u += texel as f64 * column as f64 / 511.0;
            weighted_v += texel as f64 * row as f64 / 255.0;
        }
        assert!(weight > 0.0, "no deposits in full-sphere mode");
        let (centroid_u, centroid_v) = (weighted_u / weight, weighted_v / weight);
        println!("full-sphere deposit centroid: ({centroid_u:.3}, {centroid_v:.3})");
        assert!(
            (centroid_u - 0.75).abs() < 0.05 && (centroid_v - 0.5).abs() < 0.1,
            "deposits centred at ({centroid_u}, {centroid_v}), not near the entry (0.75, 0.5)"
        );
    }

    /// Thickness in metres at which the film's ln n_eff equals `ln_effective_index`
    fn thickness_m_for_ln_index(ln_effective_index: f64) -> f32 {
        (film_waveguide::thickness_for_ln_effective_index(ln_effective_index) * 1e-9) as f32
    }

    /// ln n_eff at 400 nm, the centre of the GRIN test ramps
    fn ln_index_at_400_nm() -> f64 {
        film_waveguide::te0_effective_index(
            400.0,
            film_waveguide::GUIDED_FILM_INDEX,
            film_waveguide::GUIDED_WAVELENGTH_NM,
        )
        .ln()
    }

    /// Thickness field (metres) whose ln n_eff rises linearly with theta at `slope` per
    /// radian within 0.5 rad of the equator, constant beyond, around `ln_at_equator`
    fn ln_index_ramp_in_theta(ln_at_equator: f64, slope: f64) -> Vec<f32> {
        (0..THICKNESS_ROWS)
            .flat_map(|row| {
                let theta = row as f64 * std::f64::consts::PI / (THICKNESS_ROWS - 1) as f64;
                let offset = (theta - std::f64::consts::FRAC_PI_2).clamp(-0.5, 0.5);
                let thickness = thickness_m_for_ln_index(ln_at_equator + slope * offset);
                std::iter::repeat_n(thickness, THICKNESS_COLUMNS)
            })
            .collect()
    }

    /// Thickness field (metres) whose ln n_eff rises linearly with phi at `slope` per
    /// radian within 0.5 rad of the u = 0/1 seam (phi = -pi), constant beyond, around
    /// `ln_at_seam`. Column j sits at phi = -pi + j 2 pi / W, the drainage grid's convention.
    fn ln_index_ramp_in_phi_across_the_seam(ln_at_seam: f64, slope: f64) -> Vec<f32> {
        let row: Vec<f32> = (0..THICKNESS_COLUMNS)
            .map(|column| {
                let wrapped = if column < THICKNESS_COLUMNS / 2 {
                    column as f64
                } else {
                    column as f64 - THICKNESS_COLUMNS as f64
                };
                let offset =
                    (wrapped * std::f64::consts::TAU / THICKNESS_COLUMNS as f64).clamp(-0.5, 0.5);
                thickness_m_for_ln_index(ln_at_seam + slope * offset)
            })
            .collect();
        row.repeat(THICKNESS_ROWS)
    }

    /// Deposit-weighted mean v (0 to 1) of one deposit texture column
    fn column_centroid_v(texels: &[u32], column: usize) -> f64 {
        let (mut weight, mut weighted_v) = (0.0f64, 0.0f64);
        for row in 0..256 {
            let texel = texels[row * 512 + column] as f64;
            weight += texel;
            weighted_v += texel * row as f64 / 255.0;
        }
        assert!(weight > 0.0, "no deposits in column {column}");
        weighted_v / weight
    }

    /// Deposit-weighted mean u offset from the seam in one deposit texture row. Columns
    /// map to u = column / 511 (`deposit_bilinear`), so columns 0 and 511 both lie on the
    /// seam; offsets are wrapped into [-0.5, 0.5).
    fn row_centroid_u_from_seam(texels: &[u32], row: usize) -> f64 {
        let (mut weight, mut weighted_offset) = (0.0f64, 0.0f64);
        for column in 0..512 {
            let texel = texels[row * 512 + column] as f64;
            let u = column as f64 / 511.0;
            weight += texel;
            weighted_offset += texel * if u < 0.5 { u } else { u - 1.0 };
        }
        assert!(weight > 0.0, "no deposits in row {row}");
        weighted_offset / weight
    }

    /// Deposit textures of one frame of pure-GRIN rays at physical strength
    /// (particle_weight = 0, bend_strength = 1) from the full-sphere entry `entry_point`
    /// at `beam_angle_deg`: over a uniform film, and over `field(+slope)` and `field(-slope)`
    fn grin_deflection_frames(
        entry_point: [f32; 3],
        beam_angle_deg: f32,
        field: impl Fn(f64) -> Vec<f32>,
        slope: f64,
    ) -> [Vec<u32>; 3] {
        let Some((device, queue)) = test_device() else {
            panic!("no GPU adapter (run via scripts/test-local.sh for lavapipe)");
        };
        let configure = |simulator: &mut BranchedFlowSimulator| {
            simulator.params.patch_enabled = 0;
            simulator.params.entry_point = entry_point;
            simulator.params.particle_weight = 0.0;
            simulator.params.bend_strength = 1.0;
            // |force| ~ 0.1 to 0.2 sets the step factor to about 1 to 0.5, so 400 steps
            // still reach the measured chart distance of 1
            simulator.params.ray_steps = 400;
            // A narrow beam (rays within 0.05 of the central one) keeps the measured
            // centroid on the central ray the predictions follow. At the default width of
            // 0.4 the phi test read +0.00886 / -0.00884, 4.6-4.9% above its prediction. A
            // numpy replica of the loop and the deposit pass with the exact force gives
            // 0.00885, while a per-ray replica puts the rays' mean crossing at 0.00847: the
            // excess is in the deposit-weighted centroid of a wide beam, not in the force
            // (both replicas in docs/evidence/2026-11-12-grin-deflection/). The theta test
            // read 0.4% below its prediction at 0.4 and reads 0.8-0.9% above at 0.1.
            simulator.params.spread_angle = 0.1;
            simulator.set_beam_angle(beam_angle_deg);
        };
        [
            one_frame_deposits(&device, &queue, &uniform_thickness(), configure),
            one_frame_deposits(&device, &queue, &field(slope), configure),
            one_frame_deposits(&device, &queue, &field(-slope), configure),
        ]
    }

    /// The measured shifts must have opposite signs (toward the thicker film) and lie
    /// within 5% of the closed-form prediction in the loop's gnomonic chart. Unmutated,
    /// the tests read 0.8-0.9% (theta) and 1.2-1.3% (phi) above it; uploading n_eff instead
    /// of ln n_eff reads about 27% above and fails.
    fn assert_shifts_match(toward_thicker: f64, away: f64, predicted: f64) {
        assert!(
            toward_thicker > 0.0 && away < 0.0,
            "rays not bent toward thicker film: {toward_thicker:+.5}, {away:+.5}"
        );
        for shift in [toward_thicker, -away] {
            assert!(
                (0.95..1.05).contains(&(shift / predicted)),
                "shift {shift:.5} vs predicted {predicted:.5}"
            );
        }
    }

    #[test]
    #[ignore] // Requires GPU (lavapipe works: scripts/test-local.sh -- --ignored)
    fn grin_force_along_theta_matches_the_chart_integrator() {
        // #45: before it, the GRIN force differentiated thickness in metres and bent a ray
        // by about 2e-4 degrees. Here a beam runs east along the equator from the laser
        // entry (0, 0, 1) through a film whose ln n_eff rises by `slope` per radian toward
        // the south, with only the GRIN force at bend_strength = 1. In the gnomonic chart
        // the loop integrates y'' = slope, so at chart distance x the beam has moved
        // y = slope x^2 / 2 south, which is v = atan(y / sqrt(1 + x^2)) / pi.
        //
        // This pins the loop's chart integrator, not geodesic physics: on the sphere the
        // path is a small circle of geodesic curvature `slope`, which moves
        // slope (1 - cos 0.7839) / pi = 0.00929 in v at this column, 0.83x the chart value.
        // Exponential-map propagation (#47) would change the prediction to that.
        let slope = 0.1;
        let [straight, thicker_south, thicker_north] = grin_deflection_frames(
            [0.0, 0.0, 1.0],
            0.0,
            |slope| ln_index_ramp_in_theta(ln_index_at_400_nm(), slope),
            slope,
        );

        // Column 447 is u = 0.8748, i.e. 0.7839 rad east of the entry (u = 0.75)
        let column = 447;
        let chart_distance = ((column as f64 / 511.0 - 0.75) * std::f64::consts::TAU).tan();
        let chart_shift = slope * chart_distance * chart_distance / 2.0;
        let predicted = (chart_shift / (1.0 + chart_distance * chart_distance).sqrt()).atan()
            / std::f64::consts::PI;
        let baseline = column_centroid_v(&straight, column);
        let south_shift = column_centroid_v(&thicker_south, column) - baseline;
        let north_shift = column_centroid_v(&thicker_north, column) - baseline;
        println!(
            "column {column} (chart x = {chart_distance:.4}): baseline v {baseline:.5}, \
             shift with thicker south {south_shift:+.5}, thicker north {north_shift:+.5}, \
             predicted +-{predicted:.5}"
        );
        assert_shifts_match(south_shift, north_shift, predicted);
    }

    #[test]
    #[ignore] // Requires GPU (lavapipe works: scripts/test-local.sh -- --ignored)
    fn grin_force_along_phi_across_the_seam_matches_the_chart_integrator() {
        // The east component of the force and the periodic thickness sampler, where the
        // shader uses them. The beam runs south along the u = 0/1 seam from (-1, 0, 0),
        // through a film whose ln n_eff rises by `slope` per radian of phi toward the east,
        // so every thickness sample straddles the seam. The east force on the sphere is
        // slope / sin(theta); along the meridian theta = pi/2 + atan(x), so the loop
        // integrates y'' = slope sqrt(1 + x^2) in the chart, giving
        // y = slope ((((1 + x^2)^1.5 - 1) / 3 + x asinh(x) - sqrt(1 + x^2) + 1) / 2.
        // The chart point (y east, x south) has phi = -pi + atan(y): u moves by atan(y) / 2 pi.
        let slope = 0.1;
        let [straight, thicker_east, thicker_west] = grin_deflection_frames(
            [-1.0, 0.0, 0.0],
            90.0,
            |slope| ln_index_ramp_in_phi_across_the_seam(ln_index_at_400_nm(), slope),
            slope,
        );

        // Row 191 is v = 0.7490, i.e. 0.7823 rad south of the equator
        let row = 191;
        let chart_distance = ((row as f64 / 255.0 - 0.5) * std::f64::consts::PI).tan();
        let stretch = (1.0 + chart_distance * chart_distance).sqrt();
        let chart_shift = slope
            * ((stretch.powi(3) - 1.0) / 3.0 + chart_distance * chart_distance.asinh() - stretch
                + 1.0)
            / 2.0;
        let predicted = chart_shift.atan() / std::f64::consts::TAU;
        let baseline = row_centroid_u_from_seam(&straight, row);
        let east_shift = row_centroid_u_from_seam(&thicker_east, row) - baseline;
        let west_shift = row_centroid_u_from_seam(&thicker_west, row) - baseline;
        println!(
            "row {row} (chart x = {chart_distance:.4}): baseline u offset {baseline:+.5}, \
             shift with thicker east {east_shift:+.5}, thicker west {west_shift:+.5}, \
             predicted +-{predicted:.5}"
        );
        assert_shifts_match(east_shift, west_shift, predicted);
    }

    #[test]
    fn default_params_are_physically_reasonable() {
        let params = BranchedFlowParams::default();

        assert!(
            params.num_rays >= 1024,
            "Need enough rays for visible pattern"
        );
        assert!(
            params.ray_steps >= 100,
            "Need enough steps for ray propagation"
        );
        assert!(
            params.step_size > 0.0 && params.step_size < 0.1,
            "Step size should be small"
        );
        assert!(params.bend_strength > 0.0, "Bend strength must be positive");
        assert!(
            params.spread_angle > 0.0 && params.spread_angle < std::f32::consts::PI,
            "Spread angle should be a reasonable radian value"
        );
        assert!(
            params.intensity_falloff > 0.0 && params.intensity_falloff < 0.1,
            "Intensity falloff should be small per step"
        );
        // The shader's ln n_eff table is in micrometres and the drainage buffer in metres
        assert_eq!(
            params.thickness_scale, 1e6,
            "thickness_scale must convert metres to micrometres"
        );
        assert_eq!(params.tex_width, 512);
        assert_eq!(params.tex_height, 256);
        // Particle scattering defaults
        assert!(
            params.num_scatterers >= 100 && params.num_scatterers <= MAX_SCATTERERS,
            "Num scatterers should be reasonable"
        );
        assert!(
            params.scatterer_strength > 0.0,
            "Scatterer strength must be positive"
        );
        assert!(
            params.scatterer_radius > 0.0 && params.scatterer_radius < 1.0,
            "Scatterer radius should be in UV space (0-1)"
        );
        assert!(
            params.particle_weight >= 0.0 && params.particle_weight <= 1.0,
            "Particle weight should be a blend factor (0-1)"
        );
    }

    #[test]
    fn scatterer_gpu_struct_size() {
        // ScattererGPU must be 16 bytes (4 f32 fields) for GPU alignment
        assert_eq!(
            std::mem::size_of::<ScattererGPU>(),
            16,
            "ScattererGPU must be 16 bytes for GPU alignment"
        );
    }

    #[test]
    fn halton_sequence_produces_values_in_range() {
        for i in 1..100u32 {
            let u = halton(i, 2);
            let v = halton(i, 3);
            assert!((0.0..1.0).contains(&u), "Halton base 2 out of range: {u}");
            assert!((0.0..1.0).contains(&v), "Halton base 3 out of range: {v}");
        }
    }

    #[test]
    fn halton_sequence_is_quasi_random() {
        // Test that halton produces low-discrepancy sequence
        // Points should be spread out, not clustered
        let n = 100;
        let mut points: Vec<(f32, f32)> = Vec::new();
        for i in 1..=n {
            points.push((halton(i as u32, 2), halton(i as u32, 3)));
        }

        // Check that no two points are too close (min spacing)
        let min_dist_sq = 0.0001f32; // Allow some clustering for quasi-random
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let dx = points[i].0 - points[j].0;
                let dy = points[i].1 - points[j].1;
                let dist_sq = dx * dx + dy * dy;
                // At least some pairs should be well separated
                if dist_sq > min_dist_sq {
                    return; // Test passes if we find separated points
                }
            }
        }
        // If we get here, all points are too close (unlikely for Halton)
        panic!("Halton sequence points are too clustered");
    }

    #[test]
    fn generate_scatterers_produces_correct_count() {
        let scatterers = generate_scatterers(100, 0.0, 0.5, 0.03, None);
        assert_eq!(scatterers.len(), 100);

        let scatterers = generate_scatterers(500, 0.0, 0.5, 0.03, None);
        assert_eq!(scatterers.len(), 500);
    }

    #[test]
    fn generate_scatterers_positions_in_range() {
        let scatterers = generate_scatterers(200, 0.0, 0.5, 0.03, None);
        for s in &scatterers {
            assert!(
                s.pos_u >= 0.0 && s.pos_u <= 1.0,
                "Scatterer pos_u out of range: {}",
                s.pos_u
            );
            assert!(
                s.pos_v >= 0.0 && s.pos_v <= 1.0,
                "Scatterer pos_v out of range: {}",
                s.pos_v
            );
        }
    }

    #[test]
    fn generate_scatterers_has_mixed_signs() {
        // Positive strength repels, negative attracts; about half each. Before #47 only
        // about 24% were repulsive (sign-preserving fract in the hash).
        let scatterers = generate_scatterers(400, 0.0, 0.5, 0.03, None);
        let repulsive = scatterers.iter().filter(|s| s.strength > 0.0).count();
        assert!(
            (160..=240).contains(&repulsive),
            "{repulsive} of 400 scatterers repel; expected about half"
        );
    }

    #[test]
    fn generate_scatterers_vary_strength_and_radius_by_20_percent() {
        // Factors in [0.8, 1.2); before #47 they spanned 0.4 to 1.2 with mean 0.79
        let scatterers = generate_scatterers(400, 0.0, 0.5, 0.03, None);
        let strength_factors: Vec<f32> =
            scatterers.iter().map(|s| s.strength.abs() / 0.5).collect();
        let radius_factors: Vec<f32> = scatterers
            .iter()
            .map(|s| (1.0 / (2.0 * s.inv_sigma_sq)).sqrt() / 0.03)
            .collect();
        for (name, factors) in [("strength", &strength_factors), ("radius", &radius_factors)] {
            assert!(
                factors
                    .iter()
                    .all(|&factor| (0.8 - 1e-5..1.2 + 1e-5).contains(&factor)),
                "{name} factor outside [0.8, 1.2)"
            );
            let mean = factors.iter().sum::<f32>() / factors.len() as f32;
            assert!((mean - 1.0).abs() < 0.03, "{name} factor mean {mean}");
        }
    }

    /// `count` scatterers on a jittered grid inside the given UV rectangle
    fn scatterers_in(
        count: usize,
        min_u: f32,
        max_u: f32,
        min_v: f32,
        max_v: f32,
    ) -> Vec<ScattererGPU> {
        (0..count as u32)
            .map(|index| ScattererGPU {
                pos_u: min_u + (max_u - min_u) * halton(index + 1, 2),
                pos_v: min_v + (max_v - min_v) * halton(index + 1, 3),
                strength: 0.5,
                inv_sigma_sq: 1.0 / (2.0 * 0.03 * 0.03),
            })
            .collect()
    }

    /// Default patch rectangle (centre 0.75, 0.5, half size 0.158)
    const DEFAULT_PATCH: DriftBounds = DriftBounds {
        min_u: 0.592,
        max_u: 0.908,
        min_v: 0.342,
        max_v: 0.658,
        wrap_u: false,
    };

    #[test]
    fn drift_velocity_is_zero_mean_at_the_set_speed() {
        // The increments alone, far from any boundary: 400 scatterers for 50 film-seconds
        // at 60 updates per second. Before #47 every scatterer moved by a mean of
        // -0.0005 UV per frame on both axes, i.e. -0.03 UV per second.
        let mut scatterers = scatterers_in(400, -0.5, 0.5, -0.5, 0.5);
        let mut velocities = initial_drift_velocities(400, 0);
        let unbounded = DriftBounds {
            min_u: -1e6,
            max_u: 1e6,
            min_v: -1e6,
            max_v: 1e6,
            wrap_u: false,
        };
        let start: Vec<(f32, f32)> = scatterers.iter().map(|s| (s.pos_u, s.pos_v)).collect();
        let (mut sum, mut sum_sq, mut samples) = ([0.0f64; 2], [0.0f64; 2], 0usize);
        for step in 1..=3000 {
            drift_scatterers(
                &mut scatterers,
                &mut velocities,
                1.0 / 60.0,
                step,
                unbounded,
            );
            for velocity in &velocities {
                for axis in 0..2 {
                    sum[axis] += velocity[axis] as f64;
                    sum_sq[axis] += (velocity[axis] as f64).powi(2);
                }
                samples += 1;
            }
        }
        for axis in 0..2 {
            let mean = sum[axis] / samples as f64;
            let rms = (sum_sq[axis] / samples as f64).sqrt();
            assert!(mean.abs() < 0.002, "axis {axis}: mean velocity {mean} UV/s");
            assert!(
                (rms / DRIFT_SPEED_PER_AXIS as f64 - 1.0).abs() < 0.1,
                "axis {axis}: RMS velocity {rms} UV/s"
            );
        }
        let mean_displacement = scatterers
            .iter()
            .zip(&start)
            .map(|(s, &(u, v))| [(s.pos_u - u) as f64, (s.pos_v - v) as f64])
            .fold([0.0; 2], |total, d| {
                [total[0] + d[0] / 400.0, total[1] + d[1] / 400.0]
            });
        // The old bias would have moved every scatterer by -1.5 UV in 50 s. The mean of
        // 400 independent displacements has a standard deviation of about 0.018 UV here.
        assert!(
            mean_displacement.iter().all(|d| d.abs() < 0.08),
            "mean displacement {mean_displacement:?} UV"
        );
    }

    #[test]
    fn drift_stops_while_the_film_is_paused() {
        assert_eq!(
            drift_time_step(4.0, 4.0),
            0.0,
            "paused: film time unchanged"
        );
        assert_eq!(drift_time_step(4.0, 1.0), 0.0, "film time reset");
        assert_eq!(
            drift_time_step(4.0, 9.0),
            MAX_DRIFT_STEP,
            "a long gap is capped"
        );
        assert!((drift_time_step(4.0, 4.0 + 1.0 / 60.0) - 1.0 / 60.0).abs() < 1e-6);
        // A zero step leaves positions and velocities as they are
        let mut scatterers = scatterers_in(50, 0.6, 0.9, 0.35, 0.65);
        let mut velocities = initial_drift_velocities(50, 3);
        let (before, velocities_before) = (scatterers.clone(), velocities.clone());
        drift_scatterers(&mut scatterers, &mut velocities, 0.0, 7, DEFAULT_PATCH);
        for (after, before) in scatterers.iter().zip(&before) {
            assert_eq!((after.pos_u, after.pos_v), (before.pos_u, before.pos_v));
        }
        assert_eq!(velocities, velocities_before);
    }

    #[test]
    fn drift_keeps_scatterers_spread_over_the_patch() {
        // 60 film-seconds in the default patch. The old drift collected every scatterer
        // in the patch's (min u, min v) corner within about 15 s; reflection at the edges
        // keeps them spread, with no pile-up at an edge.
        let patch = DEFAULT_PATCH;
        let mut scatterers = scatterers_in(400, patch.min_u, patch.max_u, patch.min_v, patch.max_v);
        let mut velocities = initial_drift_velocities(400, 0);
        for step in 1..=3600 {
            drift_scatterers(&mut scatterers, &mut velocities, 1.0 / 60.0, step, patch);
        }
        let (span_u, span_v) = (patch.max_u - patch.min_u, patch.max_v - patch.min_v);
        let mut near_edge = 0;
        for s in &scatterers {
            assert!(
                (patch.min_u..=patch.max_u).contains(&s.pos_u),
                "u {} left the patch",
                s.pos_u
            );
            assert!(
                (patch.min_v..=patch.max_v).contains(&s.pos_v),
                "v {} left the patch",
                s.pos_v
            );
            let edge_distance_u = (s.pos_u - patch.min_u).min(patch.max_u - s.pos_u) / span_u;
            let edge_distance_v = (s.pos_v - patch.min_v).min(patch.max_v - s.pos_v) / span_v;
            if edge_distance_u < 0.02 || edge_distance_v < 0.02 {
                near_edge += 1;
            }
        }
        // A uniform spread puts about 7.8% within 2% of the span from an edge
        assert!(near_edge < 60, "{near_edge} of 400 scatterers at an edge");
        // The old drift emptied the lower half (larger v) and the larger-u half, which is
        // the left half on screen at the default camera
        let lower_half = scatterers.iter().filter(|s| s.pos_v > 0.5).count();
        let larger_u_half = scatterers.iter().filter(|s| s.pos_u > 0.75).count();
        assert!(
            (140..=260).contains(&lower_half),
            "{lower_half} of 400 in the lower half"
        );
        assert!(
            (140..=260).contains(&larger_u_half),
            "{larger_u_half} of 400 in the larger-u half"
        );
    }

    #[test]
    fn drift_wraps_u_across_the_seam_in_full_sphere_mode() {
        let full_sphere = DriftBounds {
            min_u: 0.0,
            max_u: 1.0,
            min_v: 0.0,
            max_v: 1.0,
            wrap_u: true,
        };
        let mut scatterers = vec![ScattererGPU {
            pos_u: 0.999,
            pos_v: 0.5,
            strength: 0.5,
            inv_sigma_sq: 555.0,
        }];
        let mut velocities = vec![[0.06f32, 0.0]];
        for step in 1..=60 {
            drift_scatterers(
                &mut scatterers,
                &mut velocities,
                1.0 / 60.0,
                step,
                full_sphere,
            );
        }
        // About 0.05 UV east of the seam after one second, not held at u = 1
        assert!(
            scatterers[0].pos_u > 0.0 && scatterers[0].pos_u < 0.3,
            "u = {} after crossing the seam",
            scatterers[0].pos_u
        );
    }

    /// The default generation for the default patch
    fn default_patch_generation() -> ScattererGeneration {
        ScattererGeneration {
            count: 400,
            strength: 0.5,
            radius: 0.03,
            patch_bounds: Some(PatchBounds {
                center_u: 0.75,
                center_v: 0.5,
                half_size: 0.158,
            }),
        }
    }

    #[test]
    fn scatterer_field_stands_still_while_the_film_is_paused() {
        // The update path the app runs every frame: regenerate, then advance by film time.
        // A paused film repeats the same film time; it must not move anything.
        let mut field = ScattererField::default();
        field.regenerate(0.0, default_patch_generation(), DEFAULT_PATCH);
        for frame in 1..=30 {
            field.advance(frame as f32 / 60.0, DEFAULT_PATCH);
        }
        let positions = |field: &ScattererField| -> Vec<(f32, f32)> {
            field
                .scatterers
                .iter()
                .map(|s| (s.pos_u, s.pos_v))
                .collect()
        };
        let paused_at = positions(&field);
        for _ in 0..30 {
            field.advance(0.5, DEFAULT_PATCH);
        }
        assert_eq!(
            positions(&field),
            paused_at,
            "scatterers moved while paused"
        );
        field.advance(0.2, DEFAULT_PATCH);
        assert_eq!(
            positions(&field),
            paused_at,
            "film time going backwards moved scatterers"
        );
        field.advance(0.2 + 1.0 / 60.0, DEFAULT_PATCH);
        assert_ne!(positions(&field), paused_at, "playing again must move them");
    }

    #[test]
    fn scatterer_field_keeps_each_scatterer_with_its_velocity() {
        // Each velocity belongs to the scatterer at the same index, so the stored order
        // must stay the generation order however the upload copy is sorted
        let mut field = ScattererField::default();
        field.regenerate(0.0, default_patch_generation(), DEFAULT_PATCH);
        let generated_strengths: Vec<f32> = field.scatterers.iter().map(|s| s.strength).collect();
        for frame in 1..=120 {
            field.advance(frame as f32 / 60.0, DEFAULT_PATCH);
            let (sorted, _) = field.sorted_for_upload();
            assert_eq!(sorted.len(), field.scatterers.len());
        }
        let strengths: Vec<f32> = field.scatterers.iter().map(|s| s.strength).collect();
        assert_eq!(strengths, generated_strengths, "scatterers were reordered");
        assert_eq!(field.velocities.len(), field.scatterers.len());
    }

    #[test]
    fn scatterer_field_starts_inside_the_patch() {
        // Generation jitter puts a few scatterers just outside the patch; regenerate folds
        // them in, so the first drift step does not jump
        let mut field = ScattererField::default();
        field.regenerate(0.0, default_patch_generation(), DEFAULT_PATCH);
        for s in &field.scatterers {
            assert!(
                (DEFAULT_PATCH.min_u..=DEFAULT_PATCH.max_u).contains(&s.pos_u),
                "u {}",
                s.pos_u
            );
            assert!(
                (DEFAULT_PATCH.min_v..=DEFAULT_PATCH.max_v).contains(&s.pos_v),
                "v {}",
                s.pos_v
            );
        }
    }

    #[test]
    fn sorted_for_upload_groups_scatterers_by_cell() {
        let mut field = ScattererField::default();
        field.regenerate(0.0, default_patch_generation(), DEFAULT_PATCH);
        let (sorted, offsets) = field.sorted_for_upload();
        assert_eq!(offsets.len(), (GRID_SIZE_U * GRID_SIZE_V) as usize + 1);
        assert_eq!(offsets[offsets.len() - 1] as usize, sorted.len());
        for cell in 0..(GRID_SIZE_U * GRID_SIZE_V) as usize {
            for s in &sorted[offsets[cell] as usize..offsets[cell + 1] as usize] {
                let u_cell =
                    (s.pos_u / GRID_CELL_SIZE).clamp(0.0, (GRID_SIZE_U - 1) as f32) as usize;
                let v_cell =
                    (s.pos_v / GRID_CELL_SIZE).clamp(0.0, (GRID_SIZE_V - 1) as f32) as usize;
                assert_eq!(v_cell * GRID_SIZE_U as usize + u_cell, cell);
            }
        }
    }

    #[test]
    fn reflect_into_folds_values_back_into_the_range() {
        assert_eq!(reflect_into(0.5, 0.0, 1.0), (0.5, false));
        let (value, reflected) = reflect_into(1.1, 0.0, 1.0);
        assert!((value - 0.9).abs() < 1e-6 && reflected);
        let (value, reflected) = reflect_into(-0.2, 0.0, 1.0);
        assert!((value - 0.2).abs() < 1e-6 && reflected);
        let (value, reflected) = reflect_into(2.3, 0.0, 1.0);
        assert!((value - 0.3).abs() < 1e-5 && !reflected, "two reflections");
        assert_eq!(reflect_into(0.7, 0.4, 0.4), (0.4, false), "empty range");
    }

    #[test]
    fn unit_random_is_uniform_on_the_unit_interval() {
        assert!(((u32::MAX >> 8) as f32 * (1.0 / 16_777_216.0)) < 1.0);
        let samples: Vec<f32> = (0..100_000u32)
            .map(|index| unit_random(index, 7, 3))
            .collect();
        assert!(samples.iter().all(|&value| (0.0..1.0).contains(&value)));
        let mean = samples.iter().map(|&value| value as f64).sum::<f64>() / samples.len() as f64;
        assert!((mean - 0.5).abs() < 0.005, "mean {mean}");
        // Each decile holds about a tenth of the samples
        let mut deciles = [0usize; 10];
        for &value in &samples {
            deciles[(value * 10.0) as usize] += 1;
        }
        assert!(
            deciles
                .iter()
                .all(|&count| (9_500..10_500).contains(&count)),
            "{deciles:?}"
        );
        // Streams and steps give different values
        assert_ne!(unit_random(5, 1, 0), unit_random(5, 2, 0));
        assert_ne!(unit_random(5, 1, 0), unit_random(5, 1, 1));
    }

    #[test]
    fn ray_seeds_give_every_ray_its_own_start_at_the_slider_maximum() {
        // #47: the old float hash of the ray index repeated about a quarter of the start
        // pairs at 65536 rays, the ray-count slider's maximum (ui_state.rs). A single
        // 24-bit coordinate repeats by chance (about 128 times at 65536 rays), so this
        // checks the (across, along) pairs, which decide the start position.
        const SLIDER_MAX_RAYS: u32 = 65_536;
        for seed in [0, 1, 1_000_000, u32::MAX] {
            let pairs: std::collections::HashSet<(u32, u32)> = (0..SLIDER_MAX_RAYS)
                .map(|ray| {
                    let [across, along] = ray_seeds(ray, seed);
                    (across.to_bits(), along.to_bits())
                })
                .collect();
            assert_eq!(pairs.len(), SLIDER_MAX_RAYS as usize, "seed {seed}");
        }
        // The next seed moves every ray
        let kept = (0..SLIDER_MAX_RAYS)
            .filter(|&ray| ray_seeds(ray, 7) == ray_seeds(ray, 8))
            .count();
        assert_eq!(kept, 0, "rays that kept their start position");
        // The two coordinates are uncorrelated (standard error about 0.004 here)
        let samples: Vec<[f64; 2]> = (0..SLIDER_MAX_RAYS)
            .map(|ray| ray_seeds(ray, 3).map(f64::from))
            .collect();
        let count = samples.len() as f64;
        let mean = |axis: usize| samples.iter().map(|pair| pair[axis]).sum::<f64>() / count;
        let (mean_across, mean_along) = (mean(0), mean(1));
        let covariance = samples
            .iter()
            .map(|pair| (pair[0] - mean_across) * (pair[1] - mean_along))
            .sum::<f64>()
            / count;
        let correlation = covariance / (1.0 / 12.0);
        assert!(correlation.abs() < 0.02, "correlation {correlation}");
    }

    #[test]
    fn ray_seed_clock_advances_only_when_film_time_moves() {
        let mut clock = RaySeedClock::default();
        assert_eq!(clock.seed_for(0.0), 0, "first frame");
        assert_eq!(clock.seed_for(0.0), 0, "paused from the start");
        assert_eq!(clock.seed_for(1.0 / 60.0), 1, "playing");
        assert_eq!(clock.seed_for(1.0 / 60.0), 1, "paused");
        assert_eq!(clock.seed_for(1.0 / 60.0), 1, "still paused");
        assert_eq!(clock.seed_for(2.0 / 60.0), 2, "playing again");
        assert_eq!(clock.seed_for(0.0), 3, "reset: film time went backwards");
        let mut wrapping = RaySeedClock {
            index: u32::MAX,
            film_time: Some(0.0),
        };
        assert_eq!(wrapping.seed_for(1.0), 0, "wraps at u32::MAX");
    }

    #[test]
    fn generate_scatterers_inv_sigma_sq_is_positive() {
        let scatterers = generate_scatterers(100, 0.0, 0.5, 0.03, None);
        for s in &scatterers {
            assert!(
                s.inv_sigma_sq > 0.0,
                "inv_sigma_sq must be positive: {}",
                s.inv_sigma_sq
            );
        }
    }

    #[test]
    fn generate_scatterers_time_affects_positions() {
        let scatterers_t0 = generate_scatterers(50, 0.0, 0.5, 0.03, None);
        let scatterers_t1 = generate_scatterers(50, 1.0, 0.5, 0.03, None);

        // At least some positions should differ due to time-based jitter
        let mut different = 0;
        for i in 0..50 {
            if (scatterers_t0[i].pos_u - scatterers_t1[i].pos_u).abs() > 0.001
                || (scatterers_t0[i].pos_v - scatterers_t1[i].pos_v).abs() > 0.001
            {
                different += 1;
            }
        }
        assert!(different > 0, "Time should affect scatterer positions");
    }

    #[test]
    fn generate_scatterers_with_patch_bounds() {
        let bounds = PatchBounds {
            center_u: 0.5,
            center_v: 0.5,
            half_size: 0.2,
        };
        let scatterers = generate_scatterers(100, 0.0, 0.5, 0.03, Some(bounds));

        // All scatterers should be within patch bounds (with small margin for jitter)
        let min_u = 0.3 - 0.1; // Allow some margin for jitter
        let max_u = 0.7 + 0.1;
        let min_v = 0.3 - 0.1;
        let max_v = 0.7 + 0.1;

        for s in &scatterers {
            assert!(
                s.pos_u >= min_u && s.pos_u <= max_u,
                "Scatterer pos_u {} outside patch bounds [{}, {}]",
                s.pos_u,
                min_u,
                max_u
            );
            assert!(
                s.pos_v >= min_v && s.pos_v <= max_v,
                "Scatterer pos_v {} outside patch bounds [{}, {}]",
                s.pos_v,
                min_v,
                max_v
            );
        }
    }

    #[test]
    fn patch_bounds_mapping() {
        let bounds = PatchBounds {
            center_u: 0.5,
            center_v: 0.5,
            half_size: 0.25,
        };

        // Test corner mappings
        let (u, v) = bounds.map_to_patch(0.0, 0.0);
        assert!((u - 0.25).abs() < 1e-6);
        assert!((v - 0.25).abs() < 1e-6);

        let (u, v) = bounds.map_to_patch(1.0, 1.0);
        assert!((u - 0.75).abs() < 1e-6);
        assert!((v - 0.75).abs() < 1e-6);

        let (u, v) = bounds.map_to_patch(0.5, 0.5);
        assert!((u - 0.5).abs() < 1e-6);
        assert!((v - 0.5).abs() < 1e-6);
    }
}
