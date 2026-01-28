// Soap Bubble Thin-Film Interference Shader
// Renders iridescent colors based on film thickness and viewing angle

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
};

struct BubbleUniform {
    // Visual properties (9 floats)
    refractive_index: f32,
    base_thickness_nm: f32,
    time: f32,
    interference_intensity: f32,
    base_alpha: f32,
    edge_alpha: f32,
    background_r: f32,
    background_g: f32,
    background_b: f32,

    // Film dynamics parameters (4 floats)
    film_time: f32,
    swirl_intensity: f32,
    drainage_speed: f32,
    pattern_scale: f32,

    // Bubble position in world space (3 floats)
    position_x: f32,
    position_y: f32,
    position_z: f32,

    // Edge smoothing mode (0 = linear, 1 = smoothstep, 2 = power)
    edge_smoothing_mode: u32,

    // Branched flow parameters (light focusing through film thickness variations)
    branched_flow_enabled: u32,
    branched_flow_intensity: f32,
    branched_flow_scale: f32,
    branched_flow_sharpness: f32,
    // Light direction for branched flow (normalized)
    light_dir_x: f32,
    light_dir_y: f32,
    light_dir_z: f32,
    // Padding for 16-byte alignment
    _padding1: u32,
};

// Branched flow texture dimensions
const BRANCHED_TEX_WIDTH: u32 = 256u;
const BRANCHED_TEX_HEIGHT: u32 = 128u;

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> bubble: BubbleUniform;
@group(0) @binding(2) var<storage, read> branched_flow_texture: array<u32>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // Apply bubble position offset
    let bubble_pos = vec3<f32>(bubble.position_x, bubble.position_y, bubble.position_z);
    let world_pos = in.position + bubble_pos;
    out.world_pos = world_pos;
    out.normal = normalize(in.normal);
    out.uv = in.uv;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    return out;
}

// ============================================================================
// Simplex Noise Implementation (3D)
// Based on Stefan Gustavson's implementation, adapted for WGSL
// ============================================================================

// Permutation polynomial hash
fn permute(x: vec4<f32>) -> vec4<f32> {
    return ((x * 34.0 + 1.0) * x) % 289.0;
}

fn taylor_inv_sqrt(r: vec4<f32>) -> vec4<f32> {
    return 1.79284291400159 - 0.85373472095314 * r;
}

// 3D Simplex noise
fn simplex_noise_3d(v: vec3<f32>) -> f32 {
    let C = vec2<f32>(1.0 / 6.0, 1.0 / 3.0);
    let D = vec4<f32>(0.0, 0.5, 1.0, 2.0);

    // First corner
    var i = floor(v + dot(v, vec3<f32>(C.y)));
    let x0 = v - i + dot(i, vec3<f32>(C.x));

    // Other corners
    let g = step(x0.yzx, x0.xyz);
    let l = 1.0 - g;
    let i1 = min(g.xyz, l.zxy);
    let i2 = max(g.xyz, l.zxy);

    let x1 = x0 - i1 + C.x;
    let x2 = x0 - i2 + C.y;
    let x3 = x0 - D.yyy;

    // Permutations
    i = i % 289.0;
    let p = permute(permute(permute(
        i.z + vec4<f32>(0.0, i1.z, i2.z, 1.0))
      + i.y + vec4<f32>(0.0, i1.y, i2.y, 1.0))
      + i.x + vec4<f32>(0.0, i1.x, i2.x, 1.0));

    // Gradients: 7x7 points over a square, mapped onto an octahedron
    let n_ = 0.142857142857; // 1.0/7.0
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

    // Normalise gradients
    let norm = taylor_inv_sqrt(vec4<f32>(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix final noise value
    var m = max(0.6 - vec4<f32>(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), vec4<f32>(0.0));
    m = m * m;
    return 42.0 * dot(m * m, vec4<f32>(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

// Fractal Brownian Motion - layered noise for natural patterns
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
// Film Thickness Calculation
// ============================================================================

// Calculate film thickness with organic noise patterns
// Uses normal vector directly to avoid UV seam artifacts
fn get_film_thickness(normal: vec3<f32>, time: f32) -> f32 {
    let base = bubble.base_thickness_nm;
    let t = bubble.film_time;
    let scale = bubble.pattern_scale;
    let swirl = bubble.swirl_intensity;

    // Animated drainage - film collects at bottom, progresses over time
    let drain_progress = min(t * bubble.drainage_speed * 0.1, 1.0);
    let drainage = 0.4 * (1.0 - normal.y) * 0.5 * (1.0 + drain_progress);

    // Organic noise pattern using simplex FBM
    // Slowly animate the noise field for flowing effect
    let noise_time = t * 0.08;
    let noise_coord = normal * scale * 3.0 + vec3<f32>(noise_time, noise_time * 0.7, noise_time * 0.5);
    let organic_noise = fbm_noise(noise_coord, 4) * swirl * 0.12;

    // Secondary swirling pattern - faster, smaller scale
    let swirl_angle = t * 0.5;
    let sx = normal.x * cos(swirl_angle) - normal.z * sin(swirl_angle);
    let sz = normal.x * sin(swirl_angle) + normal.z * cos(swirl_angle);
    let swirl_noise = simplex_noise_3d(vec3<f32>(sx, normal.y, sz) * 5.0 + vec3<f32>(t * 0.2)) * swirl * 0.05;

    // Gravity ripples - waves propagating downward (reduced, organic noise dominates)
    let wave = 0.02 * swirl * sin(normal.y * scale * 8.0 - t * 2.0) * (1.0 - abs(normal.y));

    return base * (1.0 - drainage + organic_noise + swirl_noise - wave);
}

// ============================================================================
// Branched Flow / Caustic Computation
// ============================================================================

// Convert normal direction to UV coordinates for branched flow texture sampling
fn normal_to_branched_uv(n: vec3<f32>) -> vec2<f32> {
    let phi = atan2(n.z, n.x);  // -PI to PI
    let theta = acos(clamp(n.y, -1.0, 1.0));  // 0 to PI
    let u = (phi + 3.14159265) / (2.0 * 3.14159265);  // 0 to 1
    let v = theta / 3.14159265;  // 0 to 1
    return vec2<f32>(u, v);
}

// Sample branched flow texture with bilinear interpolation
fn sample_branched_flow_texture(uv: vec2<f32>) -> f32 {
    let fx = uv.x * f32(BRANCHED_TEX_WIDTH - 1u);
    let fy = uv.y * f32(BRANCHED_TEX_HEIGHT - 1u);

    let x0 = i32(floor(fx));
    let y0 = i32(floor(fy));
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let tx = fx - f32(x0);
    let ty = fy - f32(y0);

    let width = i32(BRANCHED_TEX_WIDTH);
    let height = i32(BRANCHED_TEX_HEIGHT);

    // Wrap x (phi is periodic), clamp y
    let sx0 = ((x0 % width) + width) % width;
    let sx1 = ((x1 % width) + width) % width;
    let sy0 = clamp(y0, 0, height - 1);
    let sy1 = clamp(y1, 0, height - 1);

    let idx00 = u32(sy0 * width + sx0);
    let idx10 = u32(sy0 * width + sx1);
    let idx01 = u32(sy1 * width + sx0);
    let idx11 = u32(sy1 * width + sx1);

    // Decode from fixed-point u32 (encoded as value * 65536.0)
    let c00 = f32(branched_flow_texture[idx00]) / 65536.0;
    let c10 = f32(branched_flow_texture[idx10]) / 65536.0;
    let c01 = f32(branched_flow_texture[idx01]) / 65536.0;
    let c11 = f32(branched_flow_texture[idx11]) / 65536.0;

    let c0 = mix(c00, c10, tx);
    let c1 = mix(c01, c11, tx);
    return mix(c0, c1, ty);
}

// Get branched flow intensity from ray-traced texture
// The compute shader traces rays from a laser entry point through the film,
// bending based on thickness gradients, and accumulates intensity in a texture
fn compute_branched_flow(normal: vec3<f32>, time: f32) -> f32 {
    if (bubble.branched_flow_enabled == 0u) {
        return 0.0;
    }

    // Sample the pre-computed branched flow texture
    let uv = normal_to_branched_uv(normal);
    var intensity = sample_branched_flow_texture(uv);

    // Apply intensity scaling
    intensity *= bubble.branched_flow_intensity;

    // Apply sharpness (power function for contrast)
    intensity = pow(intensity, bubble.branched_flow_sharpness);

    return clamp(intensity, 0.0, 3.0);
}

// Snell's law: calculate transmission angle cosine
fn snells_law(cos_theta_i: f32, n_film: f32) -> f32 {
    let n_air = 1.0;
    let sin_theta_i = sqrt(max(0.0, 1.0 - cos_theta_i * cos_theta_i));
    let sin_theta_t = sin_theta_i * n_air / n_film;
    return sqrt(max(0.0, 1.0 - sin_theta_t * sin_theta_t));
}

// Full Fresnel equations with s and p polarization
// Returns vec2(R_s, R_p) - reflectances for s and p polarized light
// More accurate than Schlick approximation, especially at grazing angles
fn fresnel_full(cos_theta_i: f32, cos_theta_t: f32, n1: f32, n2: f32) -> vec2<f32> {
    // s-polarization (perpendicular to plane of incidence)
    // r_s = (n₁ cos θ_i - n₂ cos θ_t) / (n₁ cos θ_i + n₂ cos θ_t)
    let n1_cos_i = n1 * cos_theta_i;
    let n2_cos_t = n2 * cos_theta_t;
    let r_s_num = n1_cos_i - n2_cos_t;
    let r_s_den = n1_cos_i + n2_cos_t;
    let r_s = r_s_num / max(r_s_den, 0.0001);

    // p-polarization (parallel to plane of incidence)
    // r_p = (n₂ cos θ_i - n₁ cos θ_t) / (n₂ cos θ_i + n₁ cos θ_t)
    let n2_cos_i = n2 * cos_theta_i;
    let n1_cos_t = n1 * cos_theta_t;
    let r_p_num = n2_cos_i - n1_cos_t;
    let r_p_den = n2_cos_i + n1_cos_t;
    let r_p = r_p_num / max(r_p_den, 0.0001);

    // Return reflectances (squared amplitude coefficients)
    return vec2<f32>(r_s * r_s, r_p * r_p);
}

// Unpolarized Fresnel reflectance (average of s and p)
fn fresnel_unpolarized(cos_theta_i: f32, cos_theta_t: f32, n1: f32, n2: f32) -> f32 {
    let R = fresnel_full(cos_theta_i, cos_theta_t, n1, n2);
    return (R.x + R.y) * 0.5;
}

// Calculate finesse coefficient F from reflectance
// Higher F = sharper interference fringes (thicker films, grazing angles)
// Lower F = broader, softer fringes (thinner films, normal incidence)
fn finesse_coefficient(reflectance: f32) -> f32 {
    // F = 4R / (1-R)² - coefficient of finesse
    let one_minus_r = max(1.0 - reflectance, 0.001);
    return 4.0 * reflectance / (one_minus_r * one_minus_r);
}

// Airy formula for multi-bounce interference with finesse-based sharpness
// Returns reflected intensity accounting for multiple internal reflections
// Thicker films and higher reflectance produce sharper color bands
fn airy_interference(phase: f32, reflectance: f32) -> f32 {
    // Finesse coefficient determines fringe sharpness
    let F = finesse_coefficient(reflectance);

    // Airy formula: I_reflected = F * sin²(δ/2) / (1 + F * sin²(δ/2))
    let sin_half_phase = sin(phase * 0.5);
    let sin2 = sin_half_phase * sin_half_phase;
    let numerator = F * sin2;
    let denominator = 1.0 + F * sin2;

    return numerator / max(denominator, 0.001);
}

// CIE 1931 color matching functions (simplified 7-point sampling)
// Returns XYZ tristimulus values for a given wavelength
fn cie_color_matching(wavelength: f32) -> vec3<f32> {
    // Approximate CIE curves with Gaussian fits
    let x = 1.056 * exp(-0.5 * pow((wavelength - 599.8) / 37.9, 2.0))
          + 0.362 * exp(-0.5 * pow((wavelength - 442.0) / 16.0, 2.0))
          - 0.065 * exp(-0.5 * pow((wavelength - 501.1) / 20.4, 2.0));
    let y = 0.821 * exp(-0.5 * pow((wavelength - 568.8) / 46.9, 2.0))
          + 0.286 * exp(-0.5 * pow((wavelength - 530.9) / 31.1, 2.0));
    let z = 1.217 * exp(-0.5 * pow((wavelength - 437.0) / 11.8, 2.0))
          + 0.681 * exp(-0.5 * pow((wavelength - 459.0) / 26.0, 2.0));
    return vec3<f32>(max(x, 0.0), max(y, 0.0), max(z, 0.0));
}

// Convert XYZ to linear sRGB
fn xyz_to_rgb(xyz: vec3<f32>) -> vec3<f32> {
    // sRGB matrix (D65 white point)
    let r =  3.2404542 * xyz.x - 1.5371385 * xyz.y - 0.4985314 * xyz.z;
    let g = -0.9692660 * xyz.x + 1.8760108 * xyz.y + 0.0415560 * xyz.z;
    let b =  0.0556434 * xyz.x - 0.2040259 * xyz.y + 1.0572252 * xyz.z;
    return vec3<f32>(r, g, b);
}

// Calculate thin-film interference color using spectral sampling
fn thin_film_interference(thickness_nm: f32, cos_theta: f32, n_film: f32) -> vec3<f32> {
    // 7-point spectral sampling across visible range (380-700nm)
    let wavelengths = array<f32, 7>(400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0);

    // Transmission angle from Snell's law
    let cos_theta_t = snells_law(cos_theta, n_film);

    // Full Fresnel reflection with polarization (air → film interface)
    let fresnel = fresnel_unpolarized(cos_theta, cos_theta_t, 1.0, n_film);

    // Optical path (same for all wavelengths, phase varies)
    let optical_path = 2.0 * n_film * thickness_nm * cos_theta_t;

    // Accumulate XYZ tristimulus values
    var xyz = vec3<f32>(0.0);
    let pi = 3.14159265;

    // Sample each wavelength and accumulate color
    for (var i = 0u; i < 7u; i = i + 1u) {
        let wavelength = wavelengths[i];

        // Phase difference (including π phase shift from reflection at denser medium)
        let phase = 2.0 * pi * optical_path / wavelength + pi;

        // Use Airy formula for accurate multi-bounce interference
        let intensity = airy_interference(phase, fresnel);

        // Weight by CIE color matching functions
        let cie = cie_color_matching(wavelength);
        xyz = xyz + cie * intensity;
    }

    // Normalize by number of samples and convert to RGB
    xyz = xyz / 7.0;
    var rgb = xyz_to_rgb(xyz);

    // Apply intensity scaling
    rgb = rgb * bubble.interference_intensity;

    // Clamp to valid range
    return clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Renormalize interpolated normal (GPU interpolation denormalizes)
    let normal = normalize(in.normal);

    // View direction
    let view_dir = normalize(camera.camera_pos - in.world_pos);

    // Angle between view and surface normal
    let cos_theta = abs(dot(view_dir, normal));

    // Get film thickness at this point (using normal to avoid UV seam)
    let thickness = get_film_thickness(normal, bubble.time);

    // Alpha: more transparent when looking straight on, more visible at edges
    // Apply edge smoothing based on mode
    let edge_factor = 1.0 - cos_theta;
    var smooth_edge: f32;
    if (bubble.edge_smoothing_mode == 1u) {
        // Smoothstep: S-curve easing for gradual transition
        smooth_edge = edge_factor * edge_factor * (3.0 - 2.0 * edge_factor);
    } else if (bubble.edge_smoothing_mode == 2u) {
        // Power falloff: softer edges with pow(x, 1.5)
        smooth_edge = pow(edge_factor, 1.5);
    } else {
        // Linear (original behavior)
        smooth_edge = edge_factor;
    }
    let alpha = bubble.base_alpha + bubble.edge_alpha * smooth_edge;

    // Black film detection: Newton's black appears when film thins below ~30nm
    // At this thickness, destructive interference eliminates most reflection
    if (thickness < 30.0) {
        return vec4<f32>(0.02, 0.02, 0.02, alpha * 0.5);
    }

    // Calculate interference color
    let interference_color = thin_film_interference(
        thickness,
        cos_theta,
        bubble.refractive_index
    );

    // Base bubble color (slight bluish tint)
    let base_color = vec3<f32>(0.95, 0.97, 1.0);

    // Combine base with interference
    var final_color = base_color * 0.1 + interference_color;

    // Add branched flow effect - bright light filaments within the film
    let branched_flow = compute_branched_flow(normal, bubble.film_time);
    if (branched_flow > 0.0) {
        // Branched flow appears as bright white/gold filaments
        // Color shifts slightly based on thickness (chromatic caustics)
        let caustic_hue = vec3<f32>(1.0, 0.95, 0.85); // Warm white
        let caustic_color = caustic_hue * branched_flow;
        final_color = final_color + caustic_color;
    }

    return vec4<f32>(final_color, alpha);
}
