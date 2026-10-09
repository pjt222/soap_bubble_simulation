// Branched Flow Compute Shader
// Simulates branched flow of light through correlated random potential
// Based on Patsyk et al. 2020 "Observation of branched flow of light"
// put id:'gpu_compute_branched_shader', label:'Branched flow ray trace', input:'uniform_buffers_gpu.internal', output:'branched_flow_texture_gpu.internal'
//
// Key physics:
// - Light propagates through medium with smooth random refractive index variations
// - Correlation length > wavelength creates branching (not random scattering)
// - Caustics form where rays converge -> bright branch lines
// - Pattern is tree-like with successive bifurcations
//
// Performance: True spatial hash with prefix-sum cell_offsets for O(k) scatterer lookups.
// Adaptive step size scales dt by inverse gradient magnitude.
// Patch mode concentrates ray spawning within visible patch region.

struct BranchedFlowParams {
    // Laser injection point on bubble surface (normalized direction from center)
    entry_point_x: f32,
    entry_point_y: f32,
    entry_point_z: f32,
    // Initial beam direction (tangent to sphere, within film plane)
    beam_dir_x: f32,
    beam_dir_y: f32,
    beam_dir_z: f32,
    // Simulation parameters
    num_rays: u32,
    ray_steps: u32,
    step_size: f32,
    bend_strength: f32,      // How much potential gradient bends rays
    spread_angle: f32,       // Initial beam spread (radians)
    intensity_falloff: f32,  // Absorption/scattering loss along ray path
    // Output texture size
    tex_width: u32,
    tex_height: u32,
    // Time for animation
    time: f32,
    // Scale factor for thickness values (meters -> micrometers = 1e6)
    thickness_scale: f32,
    // Film dynamics parameters (synced from BubbleUniform but currently UNUSED).
    // Reserved for future: implementing FBM noise modulations in the compute shader
    // so ray bending matches the fragment shader's procedural thickness patterns.
    // See CLAUDE.md "Architecture: Branched Flow" for details.
    base_thickness_nm: f32,   // unused — reserved for compute-side noise
    swirl_intensity: f32,     // unused — reserved for compute-side noise
    drainage_speed: f32,      // unused — reserved for compute-side noise
    pattern_scale: f32,       // unused — reserved for compute-side noise
    // Particle scattering parameters
    num_scatterers: u32,      // Number of active scatterers
    scatterer_strength: f32,  // Base scattering strength
    scatterer_radius: f32,    // σ in UV space (correlation length)
    particle_weight: f32,     // Blend: 0=pure GRIN, 1=pure particle
    // Patch view mode parameters
    patch_enabled: u32,       // 0 = full sphere, 1 = patch only
    patch_center_u: f32,      // Center U coordinate (0-1)
    patch_center_v: f32,      // Center V coordinate (0-1)
    patch_half_size: f32,     // Half-width in UV space
};

// Scatterer data - represents micelle clusters that deflect light
struct Scatterer {
    pos_u: f32,           // UV position (0-1)
    pos_v: f32,
    strength: f32,        // Signed: positive repels, negative attracts
    inv_sigma_sq: f32,    // Precomputed 1/(2σ²)
};

@group(0) @binding(0) var<uniform> params: BranchedFlowParams;
@group(0) @binding(1) var<storage, read> thickness_field: array<f32>;
@group(0) @binding(2) var<storage, read_write> caustic_texture: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> scatterers: array<Scatterer>;
@group(0) @binding(4) var<storage, read> cell_offsets: array<u32>;

// Thickness field dimensions (matches GPU drainage grid)
const THICKNESS_WIDTH: u32 = 256u;
const THICKNESS_HEIGHT: u32 = 128u;

// Number of cosine modes for random potential (more = finer detail)
const NUM_POTENTIAL_MODES: i32 = 12;

// Pi constant
const PI: f32 = 3.14159265359;

// ============================================================================
// Spatial Hash Grid for Scatterer Lookups
// Scatterers are pre-sorted by grid cell on the CPU. The cell_offsets buffer
// (binding 4) is a prefix-sum array: cell_offsets[i] is the index of the first
// scatterer in cell i, cell_offsets[i+1] is one past the last.
// Each ray only checks scatterers in the 3x3 cell neighborhood → O(k) per step.
// ============================================================================

// Grid dimensions - chosen so each cell is ~3σ (scatterer radius)
// With σ ≈ 0.03, cell size ≈ 0.1, so 10×10 grid covers UV space
const GRID_SIZE_U: u32 = 10u;
const GRID_SIZE_V: u32 = 10u;
const GRID_CELL_SIZE: f32 = 0.1;

// Maximum scatterers per cell (most cells will have far fewer)
const MAX_PER_CELL: u32 = 32u;

// Get grid cell index from UV coordinates
fn uv_to_grid_cell(uv: vec2<f32>) -> vec2<u32> {
    let u_cell = u32(clamp(uv.x / GRID_CELL_SIZE, 0.0, f32(GRID_SIZE_U - 1u)));
    let v_cell = u32(clamp(uv.y / GRID_CELL_SIZE, 0.0, f32(GRID_SIZE_V - 1u)));
    return vec2<u32>(u_cell, v_cell);
}

// Convert normal direction to UV coordinates for thickness sampling
fn normal_to_uv(n: vec3<f32>) -> vec2<f32> {
    let phi = atan2(n.z, n.x);  // -PI to PI
    let theta = acos(clamp(n.y, -1.0, 1.0));  // 0 to PI
    let u = (phi + PI) / (2.0 * PI);  // 0 to 1
    let v = theta / PI;  // 0 to 1
    return vec2<f32>(u, v);
}

// ============================================================================
// UV ↔ Tangent Frame Coordinate Transformation
//
// The thickness gradient and scatterer forces are computed in UV space (phi, theta
// directions on the sphere). But vel_2d lives in the tangent frame defined at the
// laser entry point (tangent1, tangent2). These frames diverge as rays propagate
// away from the entry point — applying UV forces directly to vel_2d causes rays
// to bend in increasingly wrong directions at >1 radian from entry.
//
// Fix: compute the local phi-hat and theta-hat unit vectors at the ray's current
// 3D position, convert the UV force to a 3D vector on the sphere surface, then
// project back into the entry-point tangent frame.
// ============================================================================

// Transform a force in UV space (phi, theta directions) at the given sphere
// position into the entry-point tangent frame (tangent1, tangent2).
fn uv_force_to_tangent_frame(
    uv_force: vec2<f32>,
    pos_3d: vec3<f32>,
    tangent1: vec3<f32>,
    tangent2: vec3<f32>,
) -> vec2<f32> {
    let n = pos_3d; // Already normalized (point on unit sphere)

    // Compute local spherical coordinate unit vectors at current position:
    //   phi_hat: tangent to latitude circle (east, direction of increasing phi)
    //   theta_hat: along meridian toward south pole (direction of increasing theta)
    let xz_len = sqrt(n.x * n.x + n.z * n.z);

    var local_phi_hat: vec3<f32>;
    var local_theta_hat: vec3<f32>;

    if (xz_len > 0.001) {
        // phi_hat = (-sin(phi), 0, cos(phi)) = (-n.z/|xz|, 0, n.x/|xz|)
        local_phi_hat = vec3<f32>(-n.z / xz_len, 0.0, n.x / xz_len);
        // theta_hat = cross(normal, phi_hat) = dP/dtheta (normalized)
        //           = (n.y*n.x/|xz|, -|xz|, n.y*n.z/|xz|)
        local_theta_hat = cross(n, local_phi_hat);
    } else {
        // At poles (sin(theta) ≈ 0): phi is undefined, use arbitrary tangent frame.
        // Forces are already tapered to zero near poles by smoothstep in
        // thickness_gradient_uv(), so this branch has negligible effect.
        local_phi_hat = vec3<f32>(1.0, 0.0, 0.0);
        local_theta_hat = vec3<f32>(0.0, 0.0, 1.0);
    }

    // Convert 2D UV-space force to 3D force on sphere surface
    let force_3d = local_phi_hat * uv_force.x + local_theta_hat * uv_force.y;

    // Project 3D force into entry-point tangent frame
    return vec2<f32>(
        dot(force_3d, tangent1),
        dot(force_3d, tangent2)
    );
}

// ============================================================================
// Patch Mode Utilities
// When patch mode is enabled, rays are confined to a rectangular UV region
// ============================================================================

// Check if UV position is within patch bounds
fn is_in_patch(uv: vec2<f32>) -> bool {
    if (params.patch_enabled == 0u) {
        return true; // Full sphere mode - always in bounds
    }
    let du = abs(uv.x - params.patch_center_u);
    let dv = abs(uv.y - params.patch_center_v);
    return du <= params.patch_half_size && dv <= params.patch_half_size;
}

// Get patch UV bounds (min_u, max_u, min_v, max_v)
fn get_patch_bounds() -> vec4<f32> {
    let min_u = max(params.patch_center_u - params.patch_half_size, 0.0);
    let max_u = min(params.patch_center_u + params.patch_half_size, 1.0);
    let min_v = max(params.patch_center_v - params.patch_half_size, 0.0);
    let max_v = min(params.patch_center_v + params.patch_half_size, 1.0);
    return vec4<f32>(min_u, max_u, min_v, max_v);
}

// Map a 0-1 coordinate to within patch bounds
fn map_to_patch(t_u: f32, t_v: f32) -> vec2<f32> {
    let bounds = get_patch_bounds();
    let u = bounds.x + t_u * (bounds.y - bounds.x);
    let v = bounds.z + t_v * (bounds.w - bounds.z);
    return vec2<f32>(u, v);
}

// Map world UV to patch-local UV (0-1 within patch)
fn uv_to_patch_local(uv: vec2<f32>) -> vec2<f32> {
    let bounds = get_patch_bounds();
    let local_u = (uv.x - bounds.x) / max(bounds.y - bounds.x, 0.001);
    let local_v = (uv.y - bounds.z) / max(bounds.w - bounds.z, 0.001);
    return vec2<f32>(clamp(local_u, 0.0, 1.0), clamp(local_v, 0.0, 1.0));
}

// ============================================================================
// Hash functions for deterministic pseudo-random numbers
// ============================================================================

fn hash11(p: f32) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += p3 * (p3 + 33.33);
    return fract((p3 + p3) * p3);
}

fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

// ============================================================================
// Film Thickness as Optical Potential
// The actual soap film thickness creates the correlated disorder for branching
// Thicker regions = higher refractive index = rays bend toward them (GRIN optics)
// ============================================================================

// Sample film thickness at UV position (uses the GPU drainage buffer)
fn sample_thickness_at_uv(uv: vec2<f32>) -> f32 {
    let fx = clamp(uv.x, 0.0, 1.0) * f32(THICKNESS_WIDTH - 1u);
    let fy = clamp(uv.y, 0.0, 1.0) * f32(THICKNESS_HEIGHT - 1u);

    let x0 = u32(floor(fx));
    let y0 = u32(floor(fy));
    let x1 = min(x0 + 1u, THICKNESS_WIDTH - 1u);
    let y1 = min(y0 + 1u, THICKNESS_HEIGHT - 1u);

    let sx = fx - floor(fx);
    let sy = fy - floor(fy);

    let h00 = thickness_field[y0 * THICKNESS_WIDTH + x0];
    let h10 = thickness_field[y0 * THICKNESS_WIDTH + x1];
    let h01 = thickness_field[y1 * THICKNESS_WIDTH + x0];
    let h11 = thickness_field[y1 * THICKNESS_WIDTH + x1];

    let h0 = mix(h00, h10, sx);
    let h1 = mix(h01, h11, sx);
    return mix(h0, h1, sy);
}

// Compute thickness gradient at UV position with spherical metric correction.
// UV maps to (phi, theta) on the sphere. The physical gradient on a sphere requires
// a 1/sin(theta) factor for the phi component to account for the metric:
//   grad(h) = (1/R) * (dh/dtheta, (1/sin(theta)) * dh/dphi)
// Without this correction, ray bending is artificially exaggerated near the poles.
fn thickness_gradient_uv(uv: vec2<f32>) -> vec2<f32> {
    let eps = 0.01;  // Sampling distance in UV space

    let h_right = sample_thickness_at_uv(uv + vec2<f32>(eps, 0.0));
    let h_left = sample_thickness_at_uv(uv - vec2<f32>(eps, 0.0));
    let h_up = sample_thickness_at_uv(uv + vec2<f32>(0.0, eps));
    let h_down = sample_thickness_at_uv(uv - vec2<f32>(0.0, eps));

    // UV.y maps to theta: theta = UV.y * PI (0 at north pole, PI at south pole)
    let theta = uv.y * 3.14159265;
    let sin_theta = sin(theta);
    let clamped_sin_theta = max(sin_theta, 0.1);
    // Smoothly taper gradient to zero near poles to avoid singularity artifacts
    let pole_taper = smoothstep(0.0, 0.15, sin_theta);

    // phi gradient (UV.x direction) needs 1/sin(theta) spherical metric correction
    let grad_x = (h_right - h_left) / (2.0 * eps * clamped_sin_theta) * pole_taper;
    // theta gradient (UV.y direction) — no correction needed
    let grad_y = (h_up - h_down) / (2.0 * eps);

    return vec2<f32>(grad_x, grad_y);
}

// ============================================================================
// Particle Scattering Forces
// Discrete scatterers (micelle clusters) create local deflections
// This causes rays to diverge, cross, and form tree-like caustic branches
//
// OPTIMIZATION: Uses spatial locality - only check scatterers in nearby grid cells
// This reduces from O(n) to O(k) where k is scatterers per cell (~10-30)
// ============================================================================

// Compute force from a single scatterer using Gaussian soft potential
// V(r) = V0 * exp(-r²/2σ²)
// F = -∇V = V0 * (r/σ²) * exp(-r²/2σ²) (points toward/away from scatterer)
fn scatterer_force(ray_uv: vec2<f32>, s: Scatterer) -> vec2<f32> {
    let delta = ray_uv - vec2<f32>(s.pos_u, s.pos_v);
    let r_sq = dot(delta, delta);

    // Cutoff at ~3σ for efficiency (exp(-4.5) ≈ 0.01)
    // inv_sigma_sq = 1/(2σ²), so cutoff is when r² > 4.5 / inv_sigma_sq = 9σ²
    if (r_sq > 4.5 / s.inv_sigma_sq) {
        return vec2<f32>(0.0);
    }

    let exp_term = exp(-r_sq * s.inv_sigma_sq);
    // Force direction: delta points FROM scatterer TO ray.
    // Positive strength: force along delta (repels ray from scatterer).
    // Negative strength: force against delta (attracts ray toward scatterer).
    return delta * s.strength * s.inv_sigma_sq * 2.0 * exp_term;
}

// Check if a scatterer at given position could affect ray at ray_uv
// Returns true if within interaction range (3σ ≈ 0.09 for default radius)
fn scatterer_in_range(ray_uv: vec2<f32>, scatterer_uv: vec2<f32>, inv_sigma_sq: f32) -> bool {
    let delta = ray_uv - scatterer_uv;
    let r_sq = dot(delta, delta);
    // Cutoff radius: 3σ means r² < 9σ² = 4.5 / inv_sigma_sq
    return r_sq < 4.5 / inv_sigma_sq;
}

// Sum forces from scatterers in nearby grid cells (true spatial hash)
// Uses pre-sorted scatterer buffer + prefix-sum cell_offsets for O(k) lookup
// where k is the number of scatterers in the 3x3 neighborhood (~10-30 typical)
fn total_scatterer_force(ray_uv: vec2<f32>) -> vec2<f32> {
    var force = vec2<f32>(0.0);

    // Get ray's grid cell
    let ray_cell = uv_to_grid_cell(ray_uv);

    // Search 3x3 neighborhood
    let min_cu = select(0u, ray_cell.x - 1u, ray_cell.x >= 1u);
    let max_cu = min(ray_cell.x + 1u, GRID_SIZE_U - 1u);
    let min_cv = select(0u, ray_cell.y - 1u, ray_cell.y >= 1u);
    let max_cv = min(ray_cell.y + 1u, GRID_SIZE_V - 1u);

    // Only check scatterers indexed by the prefix-sum cell_offsets array
    for (var cv = min_cv; cv <= max_cv; cv++) {
        for (var cu = min_cu; cu <= max_cu; cu++) {
            let cell_idx = cv * GRID_SIZE_U + cu;
            let start = cell_offsets[cell_idx];
            let end = cell_offsets[cell_idx + 1u];
            for (var i = start; i < end; i++) {
                force += scatterer_force(ray_uv, scatterers[i]);
            }
        }
    }

    return force;
}

// Convert UV to output texture index
fn uv_to_tex_idx(uv: vec2<f32>) -> u32 {
    let x = u32(clamp(uv.x, 0.0, 1.0) * f32(params.tex_width - 1u));
    let y = u32(clamp(uv.y, 0.0, 1.0) * f32(params.tex_height - 1u));
    return y * params.tex_width + x;
}

// Bilinear splatting - distribute deposit across 4 neighboring pixels
// This creates smooth, anti-aliased branches instead of pixelated steps
fn deposit_bilinear(uv: vec2<f32>, intensity: f32) {
    let fx = clamp(uv.x, 0.0, 1.0) * f32(params.tex_width - 1u);
    let fy = clamp(uv.y, 0.0, 1.0) * f32(params.tex_height - 1u);

    let x0 = u32(floor(fx));
    let y0 = u32(floor(fy));
    let x1 = min(x0 + 1u, params.tex_width - 1u);
    let y1 = min(y0 + 1u, params.tex_height - 1u);

    // Fractional position within pixel
    let sx = fx - floor(fx);
    let sy = fy - floor(fy);

    // Bilinear weights
    let w00 = (1.0 - sx) * (1.0 - sy);
    let w10 = sx * (1.0 - sy);
    let w01 = (1.0 - sx) * sy;
    let w11 = sx * sy;

    // Deposit to all 4 neighboring pixels with appropriate weights
    let base_deposit = intensity * 64.0;

    let idx00 = y0 * params.tex_width + x0;
    let idx10 = y0 * params.tex_width + x1;
    let idx01 = y1 * params.tex_width + x0;
    let idx11 = y1 * params.tex_width + x1;

    atomicAdd(&caustic_texture[idx00], u32(base_deposit * w00));
    atomicAdd(&caustic_texture[idx10], u32(base_deposit * w10));
    atomicAdd(&caustic_texture[idx01], u32(base_deposit * w01));
    atomicAdd(&caustic_texture[idx11], u32(base_deposit * w11));
}

// ============================================================================
// Main ray tracing kernel - Kick-Drift model for branched flow
// ============================================================================

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_idx = global_id.x;
    if (ray_idx >= params.num_rays) {
        return;
    }

    // Entry point on sphere (laser injection point)
    let entry_point = normalize(vec3<f32>(
        params.entry_point_x,
        params.entry_point_y,
        params.entry_point_z
    ));

    // Initial beam direction (tangent to sphere)
    let beam_dir_raw = vec3<f32>(
        params.beam_dir_x,
        params.beam_dir_y,
        params.beam_dir_z
    );
    let beam_dir = normalize(beam_dir_raw - entry_point * dot(beam_dir_raw, entry_point));

    // Get tangent basis at entry point for local 2D coordinates
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(dot(entry_point, up)) > 0.99) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent1 = normalize(cross(entry_point, up));
    let tangent2 = normalize(cross(entry_point, tangent1));

    // COLLIMATED BEAM: All rays go in the SAME direction
    // Only vary the STARTING POSITION (not direction)
    // This is how real branched flow experiments work - laser beam, not point source

    let rand1 = hash21(vec2<f32>(f32(ray_idx) * 0.1, 0.0));
    let rand2 = hash21(vec2<f32>(f32(ray_idx) * 0.1 + 100.0, 1.0));

    // All rays have the same direction (collimated beam)
    var vel_2d = vec2<f32>(
        dot(beam_dir, tangent1),
        dot(beam_dir, tangent2)
    );
    vel_2d = normalize(vel_2d);

    // Vary starting POSITION perpendicular to beam direction
    // This creates a "line" of rays that then branch as they propagate
    let perp_dir = vec2<f32>(-vel_2d.y, vel_2d.x);  // Perpendicular to beam
    let pos_offset = (rand1 - 0.5) * params.spread_angle;  // spread_angle now controls beam WIDTH
    let along_offset = (rand2 - 0.5) * params.spread_angle * 0.1;  // Tiny variation along beam

    // Starting position: spread perpendicular to beam direction
    var pos_2d = perp_dir * pos_offset + vel_2d * along_offset;

    // Patch mode: concentrate rays within visible patch for higher deposit density
    if (params.patch_enabled != 0u) {
        let patch_phi = (params.patch_center_u * 2.0 - 1.0) * PI;
        let patch_theta = params.patch_center_v * PI;
        let patch_center_3d = normalize(vec3<f32>(
            sin(patch_theta) * cos(patch_phi),
            cos(patch_theta),
            sin(patch_theta) * sin(patch_phi)
        ));
        let to_patch = patch_center_3d - entry_point;
        let patch_offset = vec2<f32>(
            dot(to_patch, tangent1),
            dot(to_patch, tangent2)
        );
        let patch_spread = params.patch_half_size * PI;
        pos_2d = patch_offset
            + perp_dir * (rand1 - 0.5) * patch_spread
            + vel_2d * (rand2 - 0.5) * patch_spread * 0.5;
    }

    var intensity = 1.0;
    let dt = params.step_size;

    // ========================================================================
    // Kick-Drift ray propagation for branched flow
    //
    // HYBRID MODEL combining two deflection mechanisms:
    // 1. GRIN optics: Smooth bending toward thicker regions (like gradient-index lens)
    // 2. Particle scattering: Discrete deflections from micelle clusters
    //
    // The particle scattering is KEY for tree-like branches (caustics).
    // Pure GRIN creates parallel bands; particles cause rays to cross and diverge.
    // ========================================================================

    for (var step = 0u; step < params.ray_steps; step = step + 1u) {
        // Current 3D position and UV
        let pos_3d = normalize(entry_point + tangent1 * pos_2d.x + tangent2 * pos_2d.y);
        let uv = normal_to_uv(pos_3d);

        // === KICK: Compute forces first for adaptive stepping ===
        // 1. GRIN force: rays bend toward thicker regions (smooth, correlated)
        let grin_force_uv = thickness_gradient_uv(uv) * (1.0 - params.particle_weight);

        // 2. Particle force: discrete scatterers create local deflections (uncorrelated)
        let particle_force_uv = total_scatterer_force(uv) * params.particle_weight;

        // Combined force in UV space (phi, theta directions at current position)
        let total_force_uv = grin_force_uv + particle_force_uv;

        // Transform from UV space to entry-point tangent frame.
        // This corrects for the fact that phi-hat and theta-hat directions on the
        // sphere rotate relative to the entry-point tangent frame as rays propagate.
        // Without this transform, ray bending is increasingly wrong at >1 radian.
        let total_force = uv_force_to_tangent_frame(total_force_uv, pos_3d, tangent1, tangent2);

        // Adaptive step: larger in flat regions, smaller where gradient is steep
        let gradient_mag = length(total_force);
        let step_factor = clamp(1.0 / max(gradient_mag * 10.0, 0.333), 0.3, 3.0);
        let adaptive_dt = dt * step_factor;

        // === DEPOSIT: Scale by step factor for energy conservation ===
        // Larger steps deposit more per step to maintain constant energy per distance
        if (params.patch_enabled != 0u) {
            if (is_in_patch(uv)) {
                let local_uv = uv_to_patch_local(uv);
                deposit_bilinear(local_uv, intensity * 0.15 * step_factor);
            }
        } else {
            deposit_bilinear(uv, intensity * 0.15 * step_factor);
        }

        // Apply force with adaptive time step
        vel_2d = vel_2d + total_force * params.bend_strength * adaptive_dt;

        // Normalize velocity (constant speed, direction changes).
        // NOTE: This renormalization breaks the symplectic property of the kick-drift
        // integrator, but is physically correct for ray optics where speed is constant
        // and only direction changes (GRIN waveguide model, Patsyk et al. 2020).
        let vel_mag = length(vel_2d);
        if (vel_mag > 0.001) {
            vel_2d = vel_2d / vel_mag;
        }

        // === DRIFT: Move forward with adaptive step ===
        pos_2d = pos_2d + vel_2d * adaptive_dt;

        // Gradual intensity falloff
        intensity = intensity * (1.0 - params.intensity_falloff);

        // Stop conditions
        let dist = length(pos_2d);
        if (dist > 2.5 || intensity < 0.01) {
            break;
        }
    }
}

// ============================================================================
// Clear pass - fade existing values for smooth animation
// ============================================================================

@compute @workgroup_size(16, 16)
fn clear(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.tex_width || global_id.y >= params.tex_height) {
        return;
    }
    let idx = global_id.y * params.tex_width + global_id.x;

    // Fade existing values (creates motion blur / persistence)
    let current = atomicLoad(&caustic_texture[idx]);
    let faded = u32(f32(current) * 0.85);  // Faster fade for clearer branches
    atomicStore(&caustic_texture[idx], faded);
}
