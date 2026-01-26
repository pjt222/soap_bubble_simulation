// Caustic Compute Shader
// Computes caustic intensity from thickness field variations

struct CausticParams {
    grid_width: u32,
    grid_height: u32,
    refractive_index: f32,
    film_thickness_scale: f32,
    light_dir_x: f32,
    light_dir_y: f32,
    light_dir_z: f32,
    light_intensity: f32,
    focal_length: f32,
    caustic_intensity: f32,
    caustic_sharpness: f32,
    branch_threshold: f32,
    ground_y: f32,
    bubble_radius: f32,
    time: f32,
    _padding: u32,
};

@group(0) @binding(0) var<uniform> params: CausticParams;
@group(0) @binding(1) var<storage, read> thickness_field: array<f32>;
@group(0) @binding(2) var<storage, read_write> caustic_map: array<f32>;

// Sample thickness at grid position with boundary handling
fn sample_thickness(x: i32, y: i32) -> f32 {
    let width = i32(params.grid_width);
    let height = i32(params.grid_height);

    // Wrap x (phi is periodic)
    var sx = x % width;
    if (sx < 0) { sx += width; }

    // Clamp y (theta has poles)
    let sy = clamp(y, 0, height - 1);

    let idx = u32(sy * width + sx);
    return thickness_field[idx];
}

// Compute thickness gradient using central differences
fn thickness_gradient(x: i32, y: i32) -> vec2<f32> {
    let h = 1.0;
    let dx = (sample_thickness(x + 1, y) - sample_thickness(x - 1, y)) / (2.0 * h);
    let dy = (sample_thickness(x, y + 1) - sample_thickness(x, y - 1)) / (2.0 * h);
    return vec2<f32>(dx, dy);
}

// Compute thickness Laplacian (curvature)
fn thickness_laplacian(x: i32, y: i32) -> f32 {
    let h = 1.0;
    let h2 = h * h;

    let center = sample_thickness(x, y);
    let left = sample_thickness(x - 1, y);
    let right = sample_thickness(x + 1, y);
    let up = sample_thickness(x, y - 1);
    let down = sample_thickness(x, y + 1);

    return (left + right + up + down - 4.0 * center) / h2;
}

// Compute Hessian determinant
fn thickness_hessian_det(x: i32, y: i32) -> f32 {
    let h = 1.0;
    let h2 = h * h;

    let center = sample_thickness(x, y);
    let left = sample_thickness(x - 1, y);
    let right = sample_thickness(x + 1, y);
    let up = sample_thickness(x, y - 1);
    let down = sample_thickness(x, y + 1);

    let d2x = (left + right - 2.0 * center) / h2;
    let d2y = (up + down - 2.0 * center) / h2;

    let ul = sample_thickness(x - 1, y - 1);
    let ur = sample_thickness(x + 1, y - 1);
    let dl = sample_thickness(x - 1, y + 1);
    let dr = sample_thickness(x + 1, y + 1);
    let dxy = (dr - dl - ur + ul) / (4.0 * h2);

    return d2x * d2y - dxy * dxy;
}

// Compute caustic intensity at a grid point
fn compute_caustic_intensity(x: i32, y: i32) -> f32 {
    let grad = thickness_gradient(x, y);
    let laplacian = thickness_laplacian(x, y);
    let hessian_det = thickness_hessian_det(x, y);

    // Focusing factor from curvature
    let curvature_factor = -laplacian * params.focal_length;
    let focusing = 1.0 / max(abs(1.0 - curvature_factor), 0.01);

    // Branch factor from Hessian
    let branch_factor = select(1.0, 1.5, hessian_det < -params.branch_threshold);

    // Gradient magnitude affects refraction
    let grad_mag = length(grad);
    let refraction_factor = 1.0 + grad_mag * 0.5;

    // Combine factors
    var intensity = focusing * branch_factor * refraction_factor;
    intensity = pow(intensity, params.caustic_sharpness);
    intensity *= params.caustic_intensity;

    return clamp(intensity, 0.0, 10.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);

    if (global_id.x >= params.grid_width || global_id.y >= params.grid_height) {
        return;
    }

    let intensity = compute_caustic_intensity(x, y);
    let idx = global_id.y * params.grid_width + global_id.x;
    caustic_map[idx] = intensity;
}
