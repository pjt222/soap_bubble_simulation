// Branched Flow Compute Shader
// Simulates light rays propagating WITHIN the soap film layer
// Like a 2D waveguide - light travels laterally through the film,
// bending based on thickness gradients (gradient-index optics)

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
    bend_strength: f32,      // How much thickness gradient bends rays (GRIN effect)
    spread_angle: f32,       // Initial beam spread (radians)
    intensity_falloff: f32,  // Absorption/scattering loss along ray path
    // Output texture size
    tex_width: u32,
    tex_height: u32,
    // Time for animation
    time: f32,
    _padding: u32,
};

@group(0) @binding(0) var<uniform> params: BranchedFlowParams;
@group(0) @binding(1) var<storage, read> thickness_field: array<f32>;
@group(0) @binding(2) var<storage, read_write> caustic_texture: array<f32>;

// Thickness field dimensions (matches GPU drainage grid)
const THICKNESS_WIDTH: u32 = 128u;
const THICKNESS_HEIGHT: u32 = 64u;

// Convert normal direction to UV coordinates for thickness sampling
fn normal_to_uv(n: vec3<f32>) -> vec2<f32> {
    let phi = atan2(n.z, n.x);  // -PI to PI
    let theta = acos(clamp(n.y, -1.0, 1.0));  // 0 to PI
    let u = (phi + 3.14159265) / (2.0 * 3.14159265);  // 0 to 1
    let v = theta / 3.14159265;  // 0 to 1
    return vec2<f32>(u, v);
}

// Sample thickness at a point on the sphere
fn sample_thickness(normal: vec3<f32>) -> f32 {
    let uv = normal_to_uv(normal);
    let x = u32(uv.x * f32(THICKNESS_WIDTH - 1u));
    let y = u32(uv.y * f32(THICKNESS_HEIGHT - 1u));
    let idx = y * THICKNESS_WIDTH + x;
    return thickness_field[idx];
}

// Compute thickness gradient at a point (returns gradient in tangent space)
fn thickness_gradient(normal: vec3<f32>) -> vec2<f32> {
    let eps = 0.02;

    // Get tangent basis
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(dot(normal, up)) > 0.99) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent1 = normalize(cross(normal, up));
    let tangent2 = normalize(cross(normal, tangent1));

    // Sample neighbors
    let n_right = normalize(normal + tangent1 * eps);
    let n_left = normalize(normal - tangent1 * eps);
    let n_up = normalize(normal + tangent2 * eps);
    let n_down = normalize(normal - tangent2 * eps);

    let h_right = sample_thickness(n_right);
    let h_left = sample_thickness(n_left);
    let h_up = sample_thickness(n_up);
    let h_down = sample_thickness(n_down);

    let grad_x = (h_right - h_left) / (2.0 * eps);
    let grad_y = (h_up - h_down) / (2.0 * eps);

    return vec2<f32>(grad_x, grad_y);
}

// Simple hash for random numbers
fn hash(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Convert UV to output texture index
fn uv_to_tex_idx(uv: vec2<f32>) -> u32 {
    let x = u32(clamp(uv.x, 0.0, 1.0) * f32(params.tex_width - 1u));
    let y = u32(clamp(uv.y, 0.0, 1.0) * f32(params.tex_height - 1u));
    return y * params.tex_width + x;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_idx = global_id.x;
    if (ray_idx >= params.num_rays) {
        return;
    }

    // Entry point on sphere
    let entry_point = normalize(vec3<f32>(
        params.entry_point_x,
        params.entry_point_y,
        params.entry_point_z
    ));

    // Initial beam direction (should be tangent to sphere)
    let beam_dir_raw = vec3<f32>(
        params.beam_dir_x,
        params.beam_dir_y,
        params.beam_dir_z
    );
    // Project to tangent plane and normalize
    let beam_dir = normalize(beam_dir_raw - entry_point * dot(beam_dir_raw, entry_point));

    // Get tangent basis at entry point
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(dot(entry_point, up)) > 0.99) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent1 = normalize(cross(entry_point, up));
    let tangent2 = normalize(cross(entry_point, tangent1));

    // Random offset for this ray within the beam spread
    let rand1 = hash(vec2<f32>(f32(ray_idx), params.time));
    let rand2 = hash(vec2<f32>(f32(ray_idx) + 100.0, params.time + 1.0));
    let angle_offset = (rand1 - 0.5) * params.spread_angle;
    let perp_offset = (rand2 - 0.5) * params.spread_angle * 0.5;

    // Rotate beam direction by random angle
    let cos_a = cos(angle_offset);
    let sin_a = sin(angle_offset);
    var ray_dir = beam_dir * cos_a + cross(entry_point, beam_dir) * sin_a;
    ray_dir = normalize(ray_dir + tangent2 * perp_offset);

    // Starting position and intensity
    var pos = entry_point;
    var dir = ray_dir;
    var intensity = 1.0;

    // March the ray through the film (along sphere surface)
    for (var step = 0u; step < params.ray_steps; step++) {
        // Deposit intensity at current position - light leaves a trail as it propagates
        let uv = normal_to_uv(pos);
        let tex_idx = uv_to_tex_idx(uv);

        // Accumulate light intensity where ray passes through
        // Higher deposit = more visible ray paths
        let current = caustic_texture[tex_idx];
        let deposit = intensity * 0.05;  // Increased visibility
        caustic_texture[tex_idx] = current + deposit;

        // Get thickness gradient at current position
        let grad = thickness_gradient(pos);
        let grad_mag = length(grad);

        // Convert gradient to world space deflection
        // Get tangent basis at current position
        var local_up = vec3<f32>(0.0, 1.0, 0.0);
        if (abs(dot(pos, local_up)) > 0.99) {
            local_up = vec3<f32>(1.0, 0.0, 0.0);
        }
        let local_t1 = normalize(cross(pos, local_up));
        let local_t2 = normalize(cross(pos, local_t1));

        // GRIN optics: rays bend TOWARD thicker regions (higher refractive index)
        // gradient points toward increasing thickness, so rays follow the gradient
        let deflection = (local_t1 * grad.x + local_t2 * grad.y) * params.bend_strength;

        // Update direction with deflection
        dir = normalize(dir + deflection * params.step_size);

        // Project direction back to tangent plane (stay within film)
        dir = normalize(dir - pos * dot(dir, pos));

        // Move along the film (geodesic motion on sphere surface)
        let new_pos_raw = pos + dir * params.step_size;
        pos = normalize(new_pos_raw);  // Stay on sphere

        // Keep direction tangent to sphere
        dir = normalize(dir - pos * dot(dir, pos));

        // Intensity falloff (absorption/scattering in the film)
        intensity *= (1.0 - params.intensity_falloff);

        // Rays may focus where thickness creates "lensing" - boost intensity there
        // High gradient = strong bending = potential caustic
        if (grad_mag > 0.5) {
            intensity *= 1.02;  // Slight boost in high-gradient regions
        }

        // Stop if intensity too low
        if (intensity < 0.005) {
            break;
        }
    }
}

// Clear pass - run before ray tracing to reset texture
@compute @workgroup_size(16, 16)
fn clear(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.tex_width || global_id.y >= params.tex_height) {
        return;
    }
    let idx = global_id.y * params.tex_width + global_id.x;
    // Fade existing values instead of clearing completely (for smooth animation)
    caustic_texture[idx] = caustic_texture[idx] * 0.95;
}
