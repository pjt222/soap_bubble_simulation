// Caustic Render Shader
// Renders caustic patterns on ground plane

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

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
};

@group(0) @binding(0) var<uniform> params: CausticParams;
@group(0) @binding(1) var<uniform> camera: CameraUniform;
@group(0) @binding(2) var<storage, read> caustic_map: array<f32>;

struct GroundVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct GroundVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs_main(in: GroundVertexInput) -> GroundVertexOutput {
    var out: GroundVertexOutput;
    out.world_pos = in.position;
    out.uv = in.uv;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    return out;
}

// Sample caustic map with bilinear interpolation
fn sample_caustic(uv: vec2<f32>) -> f32 {
    let fx = uv.x * f32(params.grid_width - 1u);
    let fy = uv.y * f32(params.grid_height - 1u);

    let x0 = i32(floor(fx));
    let y0 = i32(floor(fy));
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let tx = fx - f32(x0);
    let ty = fy - f32(y0);

    let width = i32(params.grid_width);
    let height = i32(params.grid_height);

    let idx00 = u32(clamp(y0, 0, height - 1) * width + clamp(x0, 0, width - 1));
    let idx10 = u32(clamp(y0, 0, height - 1) * width + clamp(x1, 0, width - 1));
    let idx01 = u32(clamp(y1, 0, height - 1) * width + clamp(x0, 0, width - 1));
    let idx11 = u32(clamp(y1, 0, height - 1) * width + clamp(x1, 0, width - 1));

    let c00 = caustic_map[idx00];
    let c10 = caustic_map[idx10];
    let c01 = caustic_map[idx01];
    let c11 = caustic_map[idx11];

    let c0 = mix(c00, c10, tx);
    let c1 = mix(c01, c11, tx);
    return mix(c0, c1, ty);
}

@fragment
fn fs_main(in: GroundVertexOutput) -> @location(0) vec4<f32> {
    let bubble_center = vec3<f32>(0.0, 0.0, 0.0);
    let relative_pos = in.world_pos - bubble_center;

    let dist = length(relative_pos.xz);
    let max_dist = params.bubble_radius * 2.0;

    if (dist > max_dist) {
        discard;
    }

    // Radial mapping from ground to bubble surface
    let u = (relative_pos.x / max_dist + 1.0) * 0.5;
    let v = (relative_pos.z / max_dist + 1.0) * 0.5;

    let caustic_value = sample_caustic(vec2<f32>(u, v));

    // Fade out at edges
    let edge_fade = 1.0 - smoothstep(0.8, 1.0, dist / max_dist);

    // Light color with slight warm tint
    let light_color = vec3<f32>(1.0, 0.98, 0.95) * params.light_intensity;

    // Final color with caustic pattern
    let color = light_color * caustic_value * edge_fade;

    // Alpha based on intensity for additive blending
    let alpha = min(caustic_value * edge_fade * 0.5, 1.0);

    return vec4<f32>(color, alpha);
}
