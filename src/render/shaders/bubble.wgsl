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

    // Padding for 16-byte alignment (3 floats)
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
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
    out.world_pos = in.position;
    out.normal = normalize(in.normal);
    out.uv = in.uv;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
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

// Calculate thin-film interference color
fn thin_film_interference(thickness_nm: f32, cos_theta: f32, n_film: f32) -> vec3<f32> {
    // RGB wavelengths in nanometers
    let wavelengths = vec3<f32>(650.0, 532.0, 450.0);

    // Transmission angle from Snell's law
    let cos_theta_t = snells_law(cos_theta, n_film);

    // Optical path difference for each wavelength
    // δ = 2 * n * d * cos(θt) + λ/2 (phase shift at interface)
    let optical_path = 2.0 * n_film * thickness_nm * cos_theta_t;

    // Phase difference (including π phase shift from reflection)
    let phase = 2.0 * 3.14159265 * optical_path / wavelengths + 3.14159265;

    // Interference intensity: I = (1 + cos(phase)) / 2
    let intensity = 0.5 * (vec3<f32>(1.0) + cos(phase));

    // Fresnel reflection coefficient
    let fresnel = fresnel_schlick(cos_theta, n_film);

    // Combine interference with Fresnel reflection
    // Scale for visibility (thin films are weakly reflecting)
    return intensity * fresnel * bubble.interference_intensity;
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

    // Alpha: more transparent when looking straight on, more visible at edges
    let alpha = bubble.base_alpha + bubble.edge_alpha * (1.0 - cos_theta);

    return vec4<f32>(final_color, alpha);
}
