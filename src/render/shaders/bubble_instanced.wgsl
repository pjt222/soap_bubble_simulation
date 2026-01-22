// Instanced Soap Bubble Shader for Multi-Bubble Foam Rendering
// Supports rendering multiple bubbles with a single draw call using hardware instancing

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
};

// Must match BubbleUniform from bubble.wgsl (shared bind group)
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

    // Bubble position in world space (3 floats) - ignored for instanced rendering
    position_x: f32,
    position_y: f32,
    position_z: f32,

    // Edge smoothing mode (0 = linear, 1 = smoothstep, 2 = power)
    edge_smoothing_mode: u32,
    // Padding for 16-byte alignment
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> bubble: BubbleUniform;

// Per-vertex data (from mesh)
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

// Per-instance data (from instance buffer)
struct InstanceInput {
    @location(3) model_0: vec4<f32>,
    @location(4) model_1: vec4<f32>,
    @location(5) model_2: vec4<f32>,
    @location(6) model_3: vec4<f32>,
    @location(7) radius: f32,
    @location(8) aspect_ratio: f32,
    @location(9) thickness_nm: f32,
    @location(10) refractive_index: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) thickness_nm: f32,
    @location(4) refractive_index: f32,
};

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    // Reconstruct model matrix from instance data
    let model_matrix = mat4x4<f32>(
        instance.model_0,
        instance.model_1,
        instance.model_2,
        instance.model_3,
    );

    // Transform position
    let world_pos = model_matrix * vec4<f32>(vertex.position, 1.0);

    // Transform normal (using inverse transpose approximation for uniform scale)
    // For non-uniform scale (aspect_ratio != 1.0), we need proper normal transformation
    let normal_matrix = mat3x3<f32>(
        model_matrix[0].xyz,
        model_matrix[1].xyz,
        model_matrix[2].xyz,
    );

    // Adjust for aspect ratio in normal transformation
    var adjusted_normal = vertex.normal;
    if (instance.aspect_ratio < 0.999) {
        // Scale Y component inversely for oblate spheroid normal correction
        adjusted_normal.y = adjusted_normal.y / instance.aspect_ratio;
    }

    var out: VertexOutput;
    out.world_pos = world_pos.xyz;
    out.normal = normalize(normal_matrix * adjusted_normal);
    out.uv = vertex.uv;
    out.thickness_nm = instance.thickness_nm;
    out.refractive_index = instance.refractive_index;
    out.clip_position = camera.view_proj * world_pos;
    return out;
}

// ============================================================================
// Constants
// ============================================================================

const PI: f32 = 3.14159265358979323846;

// Standard RGB wavelengths (nm)
const WAVELENGTH_R: f32 = 650.0;
const WAVELENGTH_G: f32 = 532.0;
const WAVELENGTH_B: f32 = 450.0;

// ============================================================================
// Thin-Film Interference (Airy Formula)
// ============================================================================

// Snell's law: compute transmission angle cosine
fn snells_law_cos_theta_t(cos_theta_i: f32, n_ratio: f32) -> f32 {
    let sin_theta_i_sq = 1.0 - cos_theta_i * cos_theta_i;
    let sin_theta_t_sq = sin_theta_i_sq * n_ratio * n_ratio;

    if (sin_theta_t_sq > 1.0) {
        return 0.0; // Total internal reflection
    }

    return sqrt(1.0 - sin_theta_t_sq);
}

// Fresnel reflectance (Schlick approximation)
fn fresnel_schlick(cos_theta: f32, n_film: f32) -> f32 {
    let n_air = 1.0;
    let r0 = ((n_air - n_film) / (n_air + n_film)) * ((n_air - n_film) / (n_air + n_film));
    return r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);
}

// Airy formula for thin-film interference intensity
fn airy_intensity(delta: f32, reflectance: f32) -> f32 {
    let F = 4.0 * reflectance / ((1.0 - reflectance) * (1.0 - reflectance));
    let sin_half_delta = sin(delta / 2.0);
    return 1.0 / (1.0 + F * sin_half_delta * sin_half_delta);
}

// Full interference calculation for one wavelength
fn interference_for_wavelength(
    thickness_nm: f32,
    wavelength: f32,
    cos_theta_i: f32,
    n_film: f32
) -> f32 {
    let n_ratio = 1.0 / n_film;
    let cos_theta_t = snells_law_cos_theta_t(cos_theta_i, n_ratio);

    // Optical path difference
    let optical_path = 2.0 * n_film * thickness_nm * cos_theta_t;

    // Phase difference (including π shift from reflection at denser medium)
    let delta = 2.0 * PI * optical_path / wavelength + PI;

    // Fresnel reflectance
    let R = fresnel_schlick(cos_theta_i, n_film);

    return airy_intensity(delta, R);
}

// ============================================================================
// Simplex Noise (simplified 3D version)
// ============================================================================

fn hash(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn noise_3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    return mix(
        mix(mix(hash(i + vec3<f32>(0.0, 0.0, 0.0)),
                hash(i + vec3<f32>(1.0, 0.0, 0.0)), u.x),
            mix(hash(i + vec3<f32>(0.0, 1.0, 0.0)),
                hash(i + vec3<f32>(1.0, 1.0, 0.0)), u.x), u.y),
        mix(mix(hash(i + vec3<f32>(0.0, 0.0, 1.0)),
                hash(i + vec3<f32>(1.0, 0.0, 1.0)), u.x),
            mix(hash(i + vec3<f32>(0.0, 1.0, 1.0)),
                hash(i + vec3<f32>(1.0, 1.0, 1.0)), u.x), u.y),
        u.z
    );
}

fn fbm(p: vec3<f32>, octaves: i32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var pos = p;

    for (var i = 0; i < octaves; i++) {
        value += amplitude * noise_3d(pos * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return value;
}

// ============================================================================
// Film Thickness Variation
// ============================================================================

fn get_film_thickness(normal: vec3<f32>, base_thickness: f32, time: f32) -> f32 {
    // Drainage: film thins at top (normal.y > 0) due to gravity
    let drainage_factor = 0.3 * (1.0 - normal.y) * 0.5;

    // Swirling patterns using noise
    let noise_pos = normal * bubble.pattern_scale + vec3<f32>(time * 0.1, 0.0, 0.0);
    let swirl = fbm(noise_pos, 3) * bubble.swirl_intensity * 0.2;

    // Combine factors
    return base_thickness * (1.0 - drainage_factor + swirl);
}

// ============================================================================
// Fragment Shader
// ============================================================================

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.normal);
    let view_dir = normalize(camera.camera_pos - in.world_pos);

    // Viewing angle (angle from normal)
    let cos_theta = max(dot(normal, view_dir), 0.001);

    // Get film thickness with variation
    let thickness = get_film_thickness(normal, in.thickness_nm, bubble.time);

    // Calculate interference for RGB wavelengths
    let r = interference_for_wavelength(thickness, WAVELENGTH_R, cos_theta, in.refractive_index);
    let g = interference_for_wavelength(thickness, WAVELENGTH_G, cos_theta, in.refractive_index);
    let b = interference_for_wavelength(thickness, WAVELENGTH_B, cos_theta, in.refractive_index);

    var color = vec3<f32>(r, g, b);

    // Blend with background color
    let background = vec3<f32>(bubble.background_r, bubble.background_g, bubble.background_b);
    color = mix(background, color, bubble.interference_intensity);

    // Edge smoothing for alpha
    let edge_factor = 1.0 - cos_theta;
    var smooth_edge: f32;
    if (bubble.edge_smoothing_mode == 1u) {
        smooth_edge = edge_factor * edge_factor * (3.0 - 2.0 * edge_factor);
    } else if (bubble.edge_smoothing_mode == 2u) {
        smooth_edge = pow(edge_factor, 1.5);
    } else {
        smooth_edge = edge_factor;
    }
    let alpha = bubble.base_alpha + bubble.edge_alpha * smooth_edge;

    return vec4<f32>(color, alpha);
}
