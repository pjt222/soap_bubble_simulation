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
    // Scale factor for thickness values (meters -> micrometers = 1e6)
    thickness_scale: f32,
    // Film dynamics parameters (synced from BubbleUniform)
    base_thickness_nm: f32,
    swirl_intensity: f32,
    drainage_speed: f32,
    pattern_scale: f32,
    // Padding to 96 bytes (16-byte aligned)
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
};

@group(0) @binding(0) var<uniform> params: BranchedFlowParams;
@group(0) @binding(1) var<storage, read> thickness_field: array<f32>;
@group(0) @binding(2) var<storage, read_write> caustic_texture: array<atomic<u32>>;

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

// ============================================================================
// Simplex Noise Implementation (3D) - ported from bubble.wgsl
// ============================================================================

fn permute(x: vec4<f32>) -> vec4<f32> {
    return ((x * 34.0 + 1.0) * x) % 289.0;
}

fn taylor_inv_sqrt(r: vec4<f32>) -> vec4<f32> {
    return 1.79284291400159 - 0.85373472095314 * r;
}

fn simplex_noise_3d(v: vec3<f32>) -> f32 {
    let C = vec2<f32>(1.0 / 6.0, 1.0 / 3.0);
    let D = vec4<f32>(0.0, 0.5, 1.0, 2.0);

    var i = floor(v + dot(v, vec3<f32>(C.y)));
    let x0 = v - i + dot(i, vec3<f32>(C.x));

    let g = step(x0.yzx, x0.xyz);
    let l = 1.0 - g;
    let i1 = min(g.xyz, l.zxy);
    let i2 = max(g.xyz, l.zxy);

    let x1 = x0 - i1 + C.x;
    let x2 = x0 - i2 + C.y;
    let x3 = x0 - D.yyy;

    i = i % 289.0;
    let p = permute(permute(permute(
        i.z + vec4<f32>(0.0, i1.z, i2.z, 1.0))
      + i.y + vec4<f32>(0.0, i1.y, i2.y, 1.0))
      + i.x + vec4<f32>(0.0, i1.x, i2.x, 1.0));

    let n_ = 0.142857142857;
    let ns = n_ * D.wyz - D.xzx;

    let j = p - 49.0 * floor(p * ns.z * ns.z);

    let x_ = floor(j * ns.z);
    let y_ = floor(j - 7.0 * x_);

    let x = x_ * ns.x + ns.yyyy;
    let y = y_ * ns.x + ns.yyyy;
    let h = 1.0 - abs(x) - abs(y);

    let b0 = vec4<f32>(x.xy, y.xy);
    let b1 = vec4<f32>(x.zw, y.zw);

    let s0 = floor(b0) * 2.0 + 1.0;
    let s1 = floor(b1) * 2.0 + 1.0;
    let sh = -step(h, vec4<f32>(0.0));

    let a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    let a1 = b1.xzyw + s1.xzyw * sh.zzww;

    var p0 = vec3<f32>(a0.xy, h.x);
    var p1 = vec3<f32>(a0.zw, h.y);
    var p2 = vec3<f32>(a1.xy, h.z);
    var p3 = vec3<f32>(a1.zw, h.w);

    let norm = taylor_inv_sqrt(vec4<f32>(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    var m = max(0.6 - vec4<f32>(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), vec4<f32>(0.0));
    m = m * m;
    return 42.0 * dot(m * m, vec4<f32>(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

// Fractal Brownian Motion - 3 octaves (cheaper than fragment shader's 4)
fn fbm_noise(p: vec3<f32>, octaves: i32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var pos = p;

    for (var i = 0; i < octaves; i = i + 1) {
        value += amplitude * simplex_noise_3d(pos * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return value;
}

// ============================================================================
// Thickness Sampling
// ============================================================================

// Sample thickness at a point on the sphere (scaled from meters to working units)
fn sample_thickness(normal: vec3<f32>) -> f32 {
    let uv = normal_to_uv(normal);
    let x = u32(uv.x * f32(THICKNESS_WIDTH - 1u));
    let y = u32(uv.y * f32(THICKNESS_HEIGHT - 1u));
    let idx = y * THICKNESS_WIDTH + x;
    return thickness_field[idx] * params.thickness_scale;
}

// Dynamic thickness combining drainage buffer with animated noise modulations
// Mirrors get_film_thickness() from bubble.wgsl so rays bend through the same landscape
fn get_dynamic_thickness(normal: vec3<f32>, time: f32) -> f32 {
    // Base thickness from GPU drainage buffer (physical drainage simulation)
    let buffer_base = sample_thickness(normal);

    let t = time;
    let scale = params.pattern_scale;
    let swirl = params.swirl_intensity;

    // Animated drainage - film collects at bottom, progresses over time
    let drain_progress = min(t * params.drainage_speed * 0.1, 1.0);
    let drainage = 0.4 * (1.0 - normal.y) * 0.5 * (1.0 + drain_progress);

    // Organic noise pattern using simplex FBM (3 octaves for compute efficiency)
    let noise_time = t * 0.08;
    let noise_coord = normal * scale * 3.0 + vec3<f32>(noise_time, noise_time * 0.7, noise_time * 0.5);
    let organic_noise = fbm_noise(noise_coord, 3) * swirl * 0.12;

    // Secondary swirling pattern
    let swirl_angle = t * 0.5;
    let sx = normal.x * cos(swirl_angle) - normal.z * sin(swirl_angle);
    let sz = normal.x * sin(swirl_angle) + normal.z * cos(swirl_angle);
    let swirl_noise = simplex_noise_3d(vec3<f32>(sx, normal.y, sz) * 5.0 + vec3<f32>(t * 0.2)) * swirl * 0.05;

    // Gravity ripples
    let wave = 0.02 * swirl * sin(normal.y * scale * 8.0 - t * 2.0) * (1.0 - abs(normal.y));

    // Apply same multiplicative modulations as fragment shader:
    // fragment: base * (1.0 - drainage + organic_noise + swirl_noise - wave)
    // Here buffer_base plays the role of "base" — it carries physical drainage from the GPU sim
    return buffer_base * (1.0 - drainage + organic_noise + swirl_noise - wave);
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

    let h_right = get_dynamic_thickness(n_right, params.time);
    let h_left = get_dynamic_thickness(n_left, params.time);
    let h_up = get_dynamic_thickness(n_up, params.time);
    let h_down = get_dynamic_thickness(n_down, params.time);

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

        // Accumulate light intensity where ray passes through (atomic to avoid race conditions)
        let deposit = intensity * 0.05;
        let deposit_fixed = u32(deposit * 65536.0);
        atomicAdd(&caustic_texture[tex_idx], deposit_fixed);

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
        // Threshold is scale-aware: with thickness_scale=1e6, gradients are ~10+
        if (grad_mag > 5.0) {
            intensity *= 1.02;  // Slight boost in high-gradient regions
        }

        // Stop if intensity too low
        if (intensity < 0.005) {
            break;
        }
    }
}

// Clear pass - run before ray tracing to fade texture
@compute @workgroup_size(16, 16)
fn clear(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.tex_width || global_id.y >= params.tex_height) {
        return;
    }
    let idx = global_id.y * params.tex_width + global_id.x;
    // Fade existing values instead of clearing completely (for smooth animation)
    // Load current value, scale down, store back
    let current = atomicLoad(&caustic_texture[idx]);
    let faded = u32(f32(current) * 0.95);
    atomicStore(&caustic_texture[idx], faded);
}
