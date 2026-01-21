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
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> bubble: BubbleUniform;

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

// Calculate film thickness with enhanced dynamics
// Uses normal vector directly to avoid UV seam artifacts
fn get_film_thickness(normal: vec3<f32>, time: f32) -> f32 {
    let base = bubble.base_thickness_nm;
    let t = bubble.film_time;
    let scale = bubble.pattern_scale;
    let swirl = bubble.swirl_intensity;

    // Animated drainage - film collects at bottom, progresses over time
    let drain_progress = min(t * bubble.drainage_speed * 0.1, 1.0);
    let drainage = 0.4 * (1.0 - normal.y) * 0.5 * (1.0 + drain_progress);

    // Swirling pattern - rotates around Y axis
    let swirl_angle = t * 0.5;
    let sx = normal.x * cos(swirl_angle) - normal.z * sin(swirl_angle);
    let sz = normal.x * sin(swirl_angle) + normal.z * cos(swirl_angle);
    let n = normal * scale;

    // Multi-frequency noise for organic look
    let noise = swirl * 0.1 * (
        sin(4.0 * sx + t * 0.3) * sin(4.0 * n.y + t * 0.2) +
        0.6 * sin(6.0 * n.y + t * 0.4) * sin(6.0 * sz + t * 0.25) +
        0.3 * sin(8.0 * sz + t * 0.35) * sin(8.0 * sx + t * 0.15)
    ) / 1.9;

    // Gravity ripples - waves propagating downward
    let wave = 0.03 * swirl * sin(n.y * 10.0 - t * 2.0) * (1.0 - abs(n.y));

    return base * (1.0 - drainage - noise - wave);
}

// Snell's law: calculate transmission angle cosine
fn snells_law(cos_theta_i: f32, n_film: f32) -> f32 {
    let n_air = 1.0;
    let sin_theta_i = sqrt(max(0.0, 1.0 - cos_theta_i * cos_theta_i));
    let sin_theta_t = sin_theta_i * n_air / n_film;
    return sqrt(max(0.0, 1.0 - sin_theta_t * sin_theta_t));
}

// Fresnel reflectance (Schlick approximation)
fn fresnel_schlick(cos_theta: f32, n_film: f32) -> f32 {
    let n_air = 1.0;
    let r0 = pow((n_film - n_air) / (n_film + n_air), 2.0);
    return r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);
}

// Airy formula for multi-bounce interference (more accurate than simple cos)
// Returns intensity accounting for multiple internal reflections
fn airy_interference(phase: f32, reflectance: f32) -> f32 {
    let r2 = reflectance * reflectance;
    let cos_delta = cos(phase);
    // Airy formula: I = 2R(1 - cos δ) / (1 + R² - 2R cos δ)
    let numerator = 2.0 * reflectance * (1.0 - cos_delta);
    let denominator = 1.0 + r2 - 2.0 * reflectance * cos_delta;
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

    // Fresnel reflection coefficient
    let fresnel = fresnel_schlick(cos_theta, n_film);

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
    let alpha = bubble.base_alpha + bubble.edge_alpha * (1.0 - cos_theta);

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
    let final_color = base_color * 0.1 + interference_color;

    return vec4<f32>(final_color, alpha);
}
