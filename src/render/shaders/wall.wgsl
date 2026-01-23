// Shared Wall (Plateau Border) Shader for Foam Rendering
// Renders curved disk geometry (spherical caps) at contact points between touching bubbles

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
};

// Shared uniform data (same as bubble shader)
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

    // Unused for walls
    position_x: f32,
    position_y: f32,
    position_z: f32,

    // Edge smoothing mode
    edge_smoothing_mode: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> bubble: BubbleUniform;

// Per-vertex data (from disk mesh)
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
    @location(7) disk_radius: f32,
    @location(8) curvature_radius: f32,
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
    @location(5) radial_distance: f32,  // For edge fade
};

// ============================================================================
// Vertex Shader - Apply spherical cap displacement
// ============================================================================

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    // Reconstruct model matrix from instance data
    let model_matrix = mat4x4<f32>(
        instance.model_0,
        instance.model_1,
        instance.model_2,
        instance.model_3,
    );

    // Scale vertex position by disk radius
    var local_pos = vertex.position * instance.disk_radius;

    // Calculate radial distance from center (0 at center, 1 at edge)
    let radial_dist = length(vertex.position.xy);

    // Apply spherical cap curvature displacement in local Z
    // z = R - sqrt(R^2 - r^2) where r is distance from center in world units
    let r_world = radial_dist * instance.disk_radius;
    let R = abs(instance.curvature_radius);

    if (R > 0.001 && R < 1000.0 && r_world < R) {
        var z_displacement = R - sqrt(R * R - r_world * r_world);

        // Sign of curvature_radius determines direction
        // Positive: curves toward bubble A (in +Z local direction)
        // Negative: curves toward bubble B (in -Z local direction)
        if (instance.curvature_radius < 0.0) {
            z_displacement = -z_displacement;
        }

        local_pos.z = z_displacement;
    }

    // Calculate normal for curved surface
    var local_normal = vec3<f32>(0.0, 0.0, 1.0);
    if (R > 0.001 && R < 1000.0 && r_world > 0.001 && r_world < R) {
        // Normal on spherical cap points radially outward from sphere center
        let nx = vertex.position.x * instance.disk_radius / R;
        let ny = vertex.position.y * instance.disk_radius / R;
        let nz = sqrt(max(0.0, 1.0 - nx*nx - ny*ny));

        // Flip normal based on curvature direction
        if (instance.curvature_radius < 0.0) {
            local_normal = normalize(vec3<f32>(nx, ny, -nz));
        } else {
            local_normal = normalize(vec3<f32>(-nx, -ny, nz));
        }
    }

    // Transform to world space
    let world_pos = model_matrix * vec4<f32>(local_pos, 1.0);

    // Transform normal (using upper 3x3 of model matrix)
    let normal_matrix = mat3x3<f32>(
        model_matrix[0].xyz,
        model_matrix[1].xyz,
        model_matrix[2].xyz,
    );
    let world_normal = normalize(normal_matrix * local_normal);

    var out: VertexOutput;
    out.world_pos = world_pos.xyz;
    out.normal = world_normal;
    out.uv = vertex.uv;
    out.thickness_nm = instance.thickness_nm;
    out.refractive_index = instance.refractive_index;
    out.radial_distance = radial_dist;
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
// Thin-Film Interference (shared with bubble shader)
// ============================================================================

fn snells_law_cos_theta_t(cos_theta_i: f32, n_ratio: f32) -> f32 {
    let sin_theta_i_sq = 1.0 - cos_theta_i * cos_theta_i;
    let sin_theta_t_sq = sin_theta_i_sq * n_ratio * n_ratio;

    if (sin_theta_t_sq > 1.0) {
        return 0.0; // Total internal reflection
    }

    return sqrt(1.0 - sin_theta_t_sq);
}

fn fresnel_schlick(cos_theta: f32, n_film: f32) -> f32 {
    let n_air = 1.0;
    let r0 = ((n_air - n_film) / (n_air + n_film)) * ((n_air - n_film) / (n_air + n_film));
    return r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);
}

fn airy_intensity(delta: f32, reflectance: f32) -> f32 {
    let F = 4.0 * reflectance / ((1.0 - reflectance) * (1.0 - reflectance));
    let sin_half_delta = sin(delta / 2.0);
    return 1.0 / (1.0 + F * sin_half_delta * sin_half_delta);
}

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
// Noise functions for thickness variation
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

// ============================================================================
// Fragment Shader
// ============================================================================

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Use absolute value for double-sided rendering
    var normal = normalize(in.normal);
    let view_dir = normalize(camera.camera_pos - in.world_pos);

    // Flip normal if facing away from viewer (for backface)
    let facing = dot(normal, view_dir);
    if (facing < 0.0) {
        normal = -normal;
    }

    let cos_theta = abs(dot(normal, view_dir));

    // Thickness variation - walls tend to be thinner at edges due to drainage
    let edge_thinning = 1.0 - in.radial_distance * 0.2;
    let noise_pos = in.world_pos * bubble.pattern_scale * 50.0 + vec3<f32>(bubble.time * 0.1, 0.0, 0.0);
    let thickness_variation = 1.0 + (noise_3d(noise_pos) - 0.5) * 0.15 * bubble.swirl_intensity;
    let thickness = in.thickness_nm * edge_thinning * thickness_variation;

    // Calculate interference for RGB wavelengths
    let r = interference_for_wavelength(thickness, WAVELENGTH_R, cos_theta, in.refractive_index);
    let g = interference_for_wavelength(thickness, WAVELENGTH_G, cos_theta, in.refractive_index);
    let b = interference_for_wavelength(thickness, WAVELENGTH_B, cos_theta, in.refractive_index);

    var color = vec3<f32>(r, g, b);

    // Blend with background color
    let background = vec3<f32>(bubble.background_r, bubble.background_g, bubble.background_b);
    color = mix(background, color, bubble.interference_intensity);

    // Edge fade - walls fade out at edges for softer blending with bubbles
    let edge_fade = 1.0 - smoothstep(0.7, 1.0, in.radial_distance);

    // Viewing angle effect on alpha
    let edge_factor = 1.0 - cos_theta;
    var smooth_edge: f32;
    if (bubble.edge_smoothing_mode == 1u) {
        smooth_edge = edge_factor * edge_factor * (3.0 - 2.0 * edge_factor);
    } else if (bubble.edge_smoothing_mode == 2u) {
        smooth_edge = pow(edge_factor, 1.5);
    } else {
        smooth_edge = edge_factor;
    }

    let base_alpha = bubble.base_alpha + bubble.edge_alpha * smooth_edge;
    let alpha = base_alpha * edge_fade;

    return vec4<f32>(color, alpha);
}
