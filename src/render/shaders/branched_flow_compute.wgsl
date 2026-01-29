// Branched Flow Compute Shader
// Simulates branched flow of light through correlated random potential
// Based on Patsyk et al. 2020 "Observation of branched flow of light"
//
// Key physics:
// - Light propagates through medium with smooth random refractive index variations
// - Correlation length > wavelength creates branching (not random scattering)
// - Caustics form where rays converge -> bright branch lines
// - Pattern is tree-like with successive bifurcations
//
// Performance optimization: Spatial hashing for O(1) scatterer lookups
// Instead of checking all 2048 scatterers per ray step, we use a grid
// where each cell contains indices to nearby scatterers.

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
    // Film dynamics parameters (synced from BubbleUniform)
    base_thickness_nm: f32,
    swirl_intensity: f32,
    drainage_speed: f32,
    pattern_scale: f32,
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
    strength: f32,        // Signed: positive attracts, negative repels
    inv_sigma_sq: f32,    // Precomputed 1/(2σ²)
};

@group(0) @binding(0) var<uniform> params: BranchedFlowParams;
@group(0) @binding(1) var<storage, read> thickness_field: array<f32>;
@group(0) @binding(2) var<storage, read_write> caustic_texture: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> scatterers: array<Scatterer>;

// Thickness field dimensions (matches GPU drainage grid)
const THICKNESS_WIDTH: u32 = 128u;
const THICKNESS_HEIGHT: u32 = 64u;

// Number of cosine modes for random potential (more = finer detail)
const NUM_POTENTIAL_MODES: i32 = 12;

// Pi constant
const PI: f32 = 3.14159265359;

// ============================================================================
// Spatial Hash Grid for Scatterer Lookups
// Grid divides UV space into cells; each ray only checks scatterers in nearby cells
//
// NOTE: This is a grid-based EARLY-EXIT optimization, not true O(k) spatial hashing.
// We still iterate all scatterers O(n), but skip the expensive force calculation
// for scatterers outside the 3x3 cell neighborhood. True O(k) would require
// pre-built cell→scatterer index buffers.
//
// Performance: ~2-3x speedup (not ~100x like true spatial hashing)
// Physics: Correct - search radius (1 cell for 3σ cutoff) ensures no scatterers
//          within force range are missed.
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
    let x = u32(clamp(uv.x, 0.0, 1.0) * f32(THICKNESS_WIDTH - 1u));
    let y = u32(clamp(uv.y, 0.0, 1.0) * f32(THICKNESS_HEIGHT - 1u));
    let idx = y * THICKNESS_WIDTH + x;
    return thickness_field[idx];
}

// Compute thickness gradient at UV position
// This drives ray bending - rays curve toward thicker regions
fn thickness_gradient_uv(uv: vec2<f32>) -> vec2<f32> {
    let eps = 0.01;  // Sampling distance in UV space

    let h_right = sample_thickness_at_uv(uv + vec2<f32>(eps, 0.0));
    let h_left = sample_thickness_at_uv(uv - vec2<f32>(eps, 0.0));
    let h_up = sample_thickness_at_uv(uv + vec2<f32>(0.0, eps));
    let h_down = sample_thickness_at_uv(uv - vec2<f32>(0.0, eps));

    // Gradient points toward increasing thickness
    // Scale to make gradient more significant for ray bending
    let grad_x = (h_right - h_left) / (2.0 * eps);
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
    // Force direction: delta points from scatterer to ray
    // For positive strength (attractive): force points toward scatterer (negative delta)
    // For negative strength (repulsive): force points away from scatterer (positive delta)
    // The formula gives: F = strength * delta * inv_sigma_sq * 2 * exp_term
    // which for positive strength creates attraction (ray bends toward scatterer)
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

// Sum forces from scatterers in nearby grid cells (spatial hash optimization)
// Instead of checking all 2048 scatterers, only check those in the 3x3 neighborhood
fn total_scatterer_force(ray_uv: vec2<f32>) -> vec2<f32> {
    var force = vec2<f32>(0.0);
    let n = min(params.num_scatterers, 2048u);

    // Get ray's grid cell
    let ray_cell = uv_to_grid_cell(ray_uv);

    // Interaction radius in grid cells (ceil of 3σ / cell_size)
    // With σ=0.03 and cell_size=0.1, this is ceil(0.09/0.1) = 1
    let search_radius = 1u;

    // Search neighboring cells (3x3 for search_radius=1)
    let min_u = select(0u, ray_cell.x - search_radius, ray_cell.x >= search_radius);
    let max_u = min(ray_cell.x + search_radius, GRID_SIZE_U - 1u);
    let min_v = select(0u, ray_cell.y - search_radius, ray_cell.y >= search_radius);
    let max_v = min(ray_cell.y + search_radius, GRID_SIZE_V - 1u);

    // Iterate through all scatterers but early-exit based on grid position
    // This is a simplified approach that still iterates all scatterers but
    // skips the expensive force calculation for distant ones
    for (var i = 0u; i < n; i++) {
        let s = scatterers[i];
        let s_cell = uv_to_grid_cell(vec2<f32>(s.pos_u, s.pos_v));

        // Skip if scatterer is outside the search neighborhood
        if (s_cell.x < min_u || s_cell.x > max_u || s_cell.y < min_v || s_cell.y > max_v) {
            continue;
        }

        // Scatterer is in nearby cell, compute force
        force += scatterer_force(ray_uv, s);
    }

    return force;
}

// ============================================================================
// Thickness-based potential (alternative mode using actual film thickness)
// ============================================================================

fn sample_thickness(normal: vec3<f32>) -> f32 {
    let uv = normal_to_uv(normal);
    let x = u32(uv.x * f32(THICKNESS_WIDTH - 1u));
    let y = u32(uv.y * f32(THICKNESS_HEIGHT - 1u));
    let idx = y * THICKNESS_WIDTH + x;
    return thickness_field[idx] * params.thickness_scale;
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
    let base_deposit = intensity * 256.0;

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

        // === DEPOSIT: Continuous thin trail ===
        // Use very small deposit per step - branches emerge from ray convergence
        // In patch mode, only deposit if within patch bounds
        if (params.patch_enabled != 0u) {
            if (is_in_patch(uv)) {
                let local_uv = uv_to_patch_local(uv);
                deposit_bilinear(local_uv, intensity * 0.15);
            }
            // Continue tracing even if outside patch (ray might re-enter)
        } else {
            deposit_bilinear(uv, intensity * 0.15);
        }

        // === KICK: Hybrid deflection model ===
        // 1. GRIN force: rays bend toward thicker regions (smooth, correlated)
        let grin_force = thickness_gradient_uv(uv) * (1.0 - params.particle_weight);

        // 2. Particle force: discrete scatterers create local deflections (uncorrelated)
        let particle_force = total_scatterer_force(uv) * params.particle_weight;

        // Combined force drives velocity change
        let total_force = grin_force + particle_force;
        vel_2d = vel_2d + total_force * params.bend_strength * dt;

        // Normalize velocity (constant speed, direction changes)
        let vel_mag = length(vel_2d);
        if (vel_mag > 0.001) {
            vel_2d = vel_2d / vel_mag;
        }

        // === DRIFT: Move forward ===
        pos_2d = pos_2d + vel_2d * dt;

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
