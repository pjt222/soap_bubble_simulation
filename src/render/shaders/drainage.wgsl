// GPU Compute Shader for Soap Bubble Drainage Simulation
// Implements the thin-film drainage PDE on a spherical grid
//
// Drainage equation: dh/dt = -rho*g*h³/(3*eta) * sin(theta) + D*∇²h
// where:
//   h = film thickness
//   rho = fluid density
//   g = gravity
//   eta = viscosity
//   theta = polar angle (0 at top, PI at bottom)
//   D = diffusion coefficient

struct DrainageParams {
    dt: f32,                    // Time step (seconds)
    gravity: f32,               // Gravitational acceleration (m/s²)
    viscosity: f32,             // Dynamic viscosity (Pa·s)
    density: f32,               // Fluid density (kg/m³)
    diffusion_coeff: f32,       // Diffusion coefficient (m²/s)
    bubble_radius: f32,         // Bubble radius for Laplacian (m)
    critical_thickness: f32,    // Minimum thickness (m)
    grid_width: u32,            // Number of phi grid points
    grid_height: u32,           // Number of theta grid points
    _padding: vec3<u32>,        // Padding for 16-byte alignment
};

@group(0) @binding(0) var<storage, read> thickness_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> thickness_out: array<f32>;
@group(0) @binding(2) var<uniform> params: DrainageParams;

const PI: f32 = 3.14159265358979323846;

// Get grid index from 2D coordinates
fn get_index(theta_idx: u32, phi_idx: u32) -> u32 {
    return theta_idx * params.grid_width + phi_idx;
}

// Get thickness with boundary handling
fn get_thickness(theta_idx: i32, phi_idx: i32) -> f32 {
    // Clamp theta to valid range
    let t_idx = clamp(theta_idx, 0i, i32(params.grid_height) - 1i);

    // Periodic boundary for phi
    var p_idx = phi_idx % i32(params.grid_width);
    if (p_idx < 0i) {
        p_idx += i32(params.grid_width);
    }

    return thickness_in[get_index(u32(t_idx), u32(p_idx))];
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

    // Handle poles: set to average of neighboring ring
    if (theta_idx == 0u || theta_idx == params.grid_height - 1u) {
        // For poles, just copy the current value (pole averaging done separately)
        thickness_out[idx] = thickness_in[idx];
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

    // Get current thickness and neighbors
    let h = thickness_in[idx];

    // Skip if already below critical thickness
    if (h < params.critical_thickness) {
        thickness_out[idx] = h;
        return;
    }

    let i_theta = i32(theta_idx);
    let i_phi = i32(phi_idx);

    // Theta neighbors
    let h_theta_minus = get_thickness(i_theta - 1i, i_phi);
    let h_theta_plus = get_thickness(i_theta + 1i, i_phi);

    // Phi neighbors (periodic)
    let h_phi_minus = get_thickness(i_theta, i_phi - 1i);
    let h_phi_plus = get_thickness(i_theta, i_phi + 1i);

    // Precompute drainage coefficient: rho * g / (3 * eta)
    let drainage_coeff = params.density * params.gravity / (3.0 * params.viscosity);

    // Gravitational drainage term: -rho*g*h³/(3*eta) * sin(theta)
    let h_cubed = h * h * h;
    let drainage_term = -drainage_coeff * h_cubed * sin_theta;

    // Surface Laplacian on sphere: ∇²h = (1/R²) * [∂²h/∂θ² + cot(θ)*∂h/∂θ + (1/sin²θ)*∂²h/∂φ²]
    let radius_squared = params.bubble_radius * params.bubble_radius;

    // Second derivative in theta
    let d2h_dtheta2 = (h_theta_plus - 2.0 * h + h_theta_minus) / (delta_theta * delta_theta);

    // First derivative in theta (for cot(theta) term)
    let dh_dtheta = (h_theta_plus - h_theta_minus) / (2.0 * delta_theta);

    // Second derivative in phi
    let d2h_dphi2 = (h_phi_plus - 2.0 * h + h_phi_minus) / (delta_phi * delta_phi);

    // Surface Laplacian
    let laplacian = (d2h_dtheta2 + cos_theta / sin_theta_safe * dh_dtheta
                    + d2h_dphi2 / (sin_theta_safe * sin_theta_safe)) / radius_squared;

    // Diffusion term
    let diffusion_term = params.diffusion_coeff * laplacian;

    // Time evolution: dh/dt = drainage_term + diffusion_term
    let dh_dt = drainage_term + diffusion_term;

    // Update thickness using forward Euler
    let new_thickness = max(h + params.dt * dh_dt, 0.0);

    thickness_out[idx] = new_thickness;
}

// Second pass: average pole values from neighboring ring
@compute @workgroup_size(64)
fn average_poles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let phi_idx = global_id.x;

    if (phi_idx >= params.grid_width) {
        return;
    }

    // This shader averages thickness at poles - called after drainage_step
    // Top pole (theta = 0): average from theta = 1 ring
    // Bottom pole (theta = height-1): average from theta = height-2 ring

    // For simplicity, we use a storage buffer barrier and workgroup synchronization
    // The actual averaging is done via atomic adds in a follow-up dispatch
    // But for now, we keep pole values stable by copying from adjacent ring

    let top_idx = get_index(0u, phi_idx);
    let top_neighbor_idx = get_index(1u, phi_idx);
    thickness_out[top_idx] = thickness_in[top_neighbor_idx];

    let bottom_idx = get_index(params.grid_height - 1u, phi_idx);
    let bottom_neighbor_idx = get_index(params.grid_height - 2u, phi_idx);
    thickness_out[bottom_idx] = thickness_in[bottom_neighbor_idx];
}
