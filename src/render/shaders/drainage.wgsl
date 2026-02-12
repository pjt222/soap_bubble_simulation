// put id:'gpu_compute_drainage_shader', label:'Drainage PDE solver', input:'uniform_buffers_gpu.internal', output:'compute_results_gpu.internal'
// GPU Compute Shader for Soap Bubble Drainage with Marangoni Effect
//
// Implements coupled PDEs for thin-film drainage and surfactant transport:
//
// Drainage equation: dh/dt = -ρgh³/(3η) * sin(θ) + D_h∇²h + Marangoni coupling
// Surfactant: DΓ/Dt = D_s∇²Γ (advection-diffusion, simplified)
// Surface tension: γ(Γ) = γ_air - γ_r × Γ
// Marangoni stress: τ = -γ_r × ∇Γ
//
// The Marangoni effect drives flow from low surface tension (high surfactant)
// to high surface tension (low surfactant) regions.

struct DrainageParams {
    dt: f32,                    // Time step (seconds)
    gravity: f32,               // Gravitational acceleration (m/s²)
    viscosity: f32,             // Dynamic viscosity (Pa·s)
    density: f32,               // Fluid density (kg/m³)
    diffusion_coeff: f32,       // Thickness diffusion coefficient (m²/s)
    bubble_radius: f32,         // Bubble radius for Laplacian (m)
    critical_thickness: f32,    // Minimum thickness (m)
    grid_width: u32,            // Number of phi grid points
    grid_height: u32,           // Number of theta grid points
    marangoni_enabled: u32,     // Whether Marangoni is active (0 or 1)
    gamma_air: f32,             // Surface tension of clean interface (N/m)
    gamma_reduction: f32,       // Surface tension reduction rate
    surfactant_diffusion: f32,  // Surfactant diffusion coefficient (m²/s)
    marangoni_coeff: f32,       // Marangoni stress coefficient
    _padding1: u32,             // Padding for 16-byte alignment
    _padding2: u32,
};

@group(0) @binding(0) var<storage, read> thickness_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> thickness_out: array<f32>;
@group(0) @binding(2) var<uniform> params: DrainageParams;
@group(0) @binding(3) var<storage, read> concentration_in: array<f32>;
@group(0) @binding(4) var<storage, read_write> concentration_out: array<f32>;

const PI: f32 = 3.14159265358979323846;

// Get grid index from 2D coordinates
fn get_index(theta_idx: u32, phi_idx: u32) -> u32 {
    return theta_idx * params.grid_width + phi_idx;
}

// Get thickness with boundary handling
fn get_thickness(theta_idx: i32, phi_idx: i32) -> f32 {
    let t_idx = clamp(theta_idx, 0i, i32(params.grid_height) - 1i);
    var p_idx = phi_idx % i32(params.grid_width);
    if (p_idx < 0i) {
        p_idx += i32(params.grid_width);
    }
    return thickness_in[get_index(u32(t_idx), u32(p_idx))];
}

// Get concentration with boundary handling
fn get_concentration(theta_idx: i32, phi_idx: i32) -> f32 {
    let t_idx = clamp(theta_idx, 0i, i32(params.grid_height) - 1i);
    var p_idx = phi_idx % i32(params.grid_width);
    if (p_idx < 0i) {
        p_idx += i32(params.grid_width);
    }
    return concentration_in[get_index(u32(t_idx), u32(p_idx))];
}

// Compute surface tension from surfactant concentration
// γ(Γ) = γ_air - γ_reduction × Γ
fn surface_tension(concentration: f32) -> f32 {
    return params.gamma_air - params.gamma_reduction * concentration;
}

@compute @workgroup_size(16, 16)
fn drainage_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let theta_idx = global_id.y;
    let phi_idx = global_id.x;

    // Check bounds
    if (theta_idx >= params.grid_height || phi_idx >= params.grid_width) {
        return;
    }

    let idx = get_index(theta_idx, phi_idx);

    // Handle poles: copy current value (averaging done separately)
    if (theta_idx == 0u || theta_idx == params.grid_height - 1u) {
        thickness_out[idx] = thickness_in[idx];
        concentration_out[idx] = concentration_in[idx];
        return;
    }

    // Grid spacing
    let delta_theta = PI / f32(params.grid_height - 1u);
    let delta_phi = 2.0 * PI / f32(params.grid_width);

    // Compute theta angle (0 at top, PI at bottom)
    let theta = f32(theta_idx) * delta_theta;
    let sin_theta = sin(theta);
    let cos_theta = cos(theta);

    // Avoid division by zero near poles
    let sin_theta_safe = select(sin_theta, sign(sin_theta) * 0.0001, abs(sin_theta) < 0.0001);

    // Get current values and neighbors
    let h = thickness_in[idx];
    let conc = concentration_in[idx];

    // Skip if already below critical thickness
    if (h < params.critical_thickness) {
        thickness_out[idx] = h;
        concentration_out[idx] = conc;
        return;
    }

    let i_theta = i32(theta_idx);
    let i_phi = i32(phi_idx);

    // Thickness neighbors
    let h_theta_minus = get_thickness(i_theta - 1i, i_phi);
    let h_theta_plus = get_thickness(i_theta + 1i, i_phi);
    let h_phi_minus = get_thickness(i_theta, i_phi - 1i);
    let h_phi_plus = get_thickness(i_theta, i_phi + 1i);

    // Concentration neighbors
    let c_theta_minus = get_concentration(i_theta - 1i, i_phi);
    let c_theta_plus = get_concentration(i_theta + 1i, i_phi);
    let c_phi_minus = get_concentration(i_theta, i_phi - 1i);
    let c_phi_plus = get_concentration(i_theta, i_phi + 1i);

    // === THICKNESS EVOLUTION ===

    // Precompute drainage coefficient: ρg / (3η)
    let drainage_coeff = params.density * params.gravity / (3.0 * params.viscosity);

    // Gravitational drainage term: -ρgh³/(3η) * sin(θ)
    let h_cubed = h * h * h;
    let drainage_term = -drainage_coeff * h_cubed * sin_theta;

    // Surface Laplacian for thickness diffusion
    let radius_squared = params.bubble_radius * params.bubble_radius;
    let d2h_dtheta2 = (h_theta_plus - 2.0 * h + h_theta_minus) / (delta_theta * delta_theta);
    let dh_dtheta = (h_theta_plus - h_theta_minus) / (2.0 * delta_theta);
    let d2h_dphi2 = (h_phi_plus - 2.0 * h + h_phi_minus) / (delta_phi * delta_phi);

    let h_laplacian = (d2h_dtheta2 + cos_theta / sin_theta_safe * dh_dtheta
                      + d2h_dphi2 / (sin_theta_safe * sin_theta_safe)) / radius_squared;

    let diffusion_term = params.diffusion_coeff * h_laplacian;

    // Marangoni stress contribution to thickness
    var marangoni_term = 0.0;
    if (params.marangoni_enabled != 0u) {
        // Marangoni stress: τ = -γ_r × ∇Γ
        // This drives fluid from high concentration (low γ) to low concentration (high γ)
        // Gradient of concentration
        let dc_dtheta = (c_theta_plus - c_theta_minus) / (2.0 * delta_theta);
        let dc_dphi = (c_phi_plus - c_phi_minus) / (2.0 * delta_phi);

        // Magnitude of concentration gradient (on sphere surface)
        let grad_c_theta = dc_dtheta / params.bubble_radius;
        let grad_c_phi = dc_dphi / (params.bubble_radius * sin_theta_safe);

        // Marangoni stress magnitude (drives flow from high to low concentration)
        // The stress causes thickness redistribution: thin areas with high surfactant
        // push fluid toward thick areas with low surfactant
        let grad_c_mag = sqrt(grad_c_theta * grad_c_theta + grad_c_phi * grad_c_phi);
        let marangoni_stress = params.gamma_reduction * grad_c_mag;

        // Coupling: Marangoni stress affects thickness evolution
        // Higher stress in thin regions helps counter drainage (self-healing)
        marangoni_term = params.marangoni_coeff * marangoni_stress * h * h;
    }

    // Time evolution for thickness
    let dh_dt = drainage_term + diffusion_term + marangoni_term;
    let new_h = max(h + params.dt * dh_dt, 0.0);
    thickness_out[idx] = new_h;

    // === SURFACTANT CONCENTRATION EVOLUTION ===

    // Concentration diffusion (surfactant spreads on surface)
    let d2c_dtheta2 = (c_theta_plus - 2.0 * conc + c_theta_minus) / (delta_theta * delta_theta);
    let dc_dtheta = (c_theta_plus - c_theta_minus) / (2.0 * delta_theta);
    let d2c_dphi2 = (c_phi_plus - 2.0 * conc + c_phi_minus) / (delta_phi * delta_phi);

    let c_laplacian = (d2c_dtheta2 + cos_theta / sin_theta_safe * dc_dtheta
                      + d2c_dphi2 / (sin_theta_safe * sin_theta_safe)) / radius_squared;

    // Concentration evolution: diffusion + conservation coupling
    var dc_dt = params.surfactant_diffusion * c_laplacian;

    // If Marangoni is enabled, concentration also responds to thickness gradients
    // (simplified coupling: surfactant accumulates in thicker regions due to reduced drainage)
    if (params.marangoni_enabled != 0u) {
        // Coupling term: concentration tends to follow fluid accumulation
        // Thicker regions accumulate more surfactant
        let thickness_gradient_coupling = 0.001 * (h_laplacian * params.bubble_radius * params.bubble_radius);
        dc_dt += thickness_gradient_coupling * conc;
    }

    let new_c = clamp(conc + params.dt * dc_dt, 0.0, 1.0);
    concentration_out[idx] = new_c;
}

// Second pass: average pole values from neighboring ring
@compute @workgroup_size(64)
fn average_poles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let phi_idx = global_id.x;

    if (phi_idx >= params.grid_width) {
        return;
    }

    // Top pole: copy from adjacent ring
    let top_idx = get_index(0u, phi_idx);
    let top_neighbor_idx = get_index(1u, phi_idx);
    thickness_out[top_idx] = thickness_in[top_neighbor_idx];
    concentration_out[top_idx] = concentration_in[top_neighbor_idx];

    // Bottom pole: copy from adjacent ring
    let bottom_idx = get_index(params.grid_height - 1u, phi_idx);
    let bottom_neighbor_idx = get_index(params.grid_height - 2u, phi_idx);
    thickness_out[bottom_idx] = thickness_in[bottom_neighbor_idx];
    concentration_out[bottom_idx] = concentration_in[bottom_neighbor_idx];
}
