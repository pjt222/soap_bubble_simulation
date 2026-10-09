//! Runs the shaders' own sphere-UV functions on the GPU and compares them with the
//! Rust helpers the meshes use (`unit_sphere_to_uv` / `uv_to_unit_sphere`).
//!
//! The patch view broke because the patch mesh and the shaders used longitude
//! conventions half a turn apart (#46). A Rust port of the WGSL formula would not catch
//! the WGSL changing, so this test extracts each function's source text from its shader
//! file, wraps it in a one-line compute kernel and evaluates it on the device.
//!
//! GPU tests are `#[ignore]`d; run them with `scripts/test-local.sh -- --ignored`
//! (lavapipe works).

use soap_bubble_sim::physics::geometry::{SpherePatch, unit_sphere_to_uv, uv_to_unit_sphere};
use wgpu::util::DeviceExt;

const SHADER_DIR: &str = "src/render/shaders";

/// Source text of `fn name(...) { ... }` in `source`, found by brace matching.
fn extract_wgsl_function(source: &str, name: &str) -> String {
    let start = source
        .find(&format!("fn {name}("))
        .unwrap_or_else(|| panic!("fn {name} not found"));
    let body_start = start + source[start..].find('{').expect("function body");
    let mut depth = 0;
    for (offset, character) in source[body_start..].char_indices() {
        match character {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return source[start..=body_start + offset].to_string();
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced braces in fn {name}");
}

/// Evaluate `function` from `shader_file` on the GPU once per input.
///
/// `call` is the WGSL expression that calls the function on `input: vec4<f32>` and
/// returns a `vec4<f32>`, e.g. `vec4<f32>(normal_to_uv(input.xyz), 0.0, 0.0)`. The
/// shader file's own `const PI` line is included when it has one.
fn run_wgsl_function(
    shader_file: &str,
    function: &str,
    call: &str,
    inputs: &[[f32; 4]],
) -> Vec<[f32; 4]> {
    let source = std::fs::read_to_string(format!("{SHADER_DIR}/{shader_file}"))
        .unwrap_or_else(|error| panic!("read {shader_file}: {error}"));
    let pi_line = source
        .lines()
        .find(|line| line.starts_with("const PI"))
        .unwrap_or("const PI: f32 = 3.14159265358979323846;");
    let kernel = format!(
        "{pi_line}\n\
         @group(0) @binding(0) var<storage, read> inputs: array<vec4<f32>>;\n\
         @group(0) @binding(1) var<storage, read_write> outputs: array<vec4<f32>>;\n\
         {function_source}\n\
         @compute @workgroup_size(64)\n\
         fn harness_main(@builtin(global_invocation_id) id: vec3<u32>) {{\n\
             if (id.x >= arrayLength(&inputs)) {{ return; }}\n\
             let input = inputs[id.x];\n\
             outputs[id.x] = {call};\n\
         }}\n",
        function_source = extract_wgsl_function(&source, function),
    );

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("no GPU adapter (run via scripts/test-local.sh for lavapipe)");
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
            .expect("request_device");

    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wgsl function harness"),
        source: wgpu::ShaderSource::Wgsl(kernel.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("wgsl function harness"),
        layout: None,
        module: &module,
        entry_point: Some("harness_main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let output_size = std::mem::size_of_val(inputs) as wgpu::BufferAddress;
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(inputs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((inputs.len() as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_size);
    queue.submit(Some(encoder.finish()));
    if let Some(error) = pollster::block_on(device.pop_error_scope()) {
        panic!("{shader_file}::{function} harness failed validation: {error}");
    }

    readback_buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |result| {
            result.expect("map readback buffer")
        });
    device.poll(wgpu::Maintain::Wait);
    bytemuck::cast_slice::<u8, [f32; 4]>(&readback_buffer.slice(..).get_mapped_range()).to_vec()
}

/// Normals of the patch meshes the UI can produce, plus a coarse grid over the sphere
fn sample_normals() -> Vec<[f32; 4]> {
    let mut normals = Vec::new();
    for (center_u, center_v, half_size) in [(0.5, 0.5, 0.158), (0.2, 0.5, 0.158), (0.8, 0.3, 0.3)] {
        let (vertices, _) =
            SpherePatch::new(center_u, center_v, half_size, 8).generate_mesh_indexed(1.0, 1.0);
        normals.extend(vertices.iter().map(|vertex| {
            let [x, y, z] = vertex.normal;
            [x, y, z, 0.0]
        }));
    }
    for i in 1..16 {
        for j in 1..8 {
            let direction = uv_to_unit_sphere(i as f32 / 16.0 + 0.01, j as f32 / 8.0);
            normals.push([direction.x, direction.y, direction.z, 0.0]);
        }
    }
    normals
}

fn assert_uv_matches_rust(shader_file: &str, function: &str) {
    let normals = sample_normals();
    let call = format!("vec4<f32>({function}(input.xyz), 0.0, 0.0)");
    let gpu_uvs = run_wgsl_function(shader_file, function, &call, &normals);
    for (normal, gpu_uv) in normals.iter().zip(&gpu_uvs) {
        if normal[1].abs() > 0.9999 {
            continue; // longitude is undefined at a pole: atan2(0, 0)
        }
        let [u, v] = unit_sphere_to_uv(glam::Vec3::new(normal[0], normal[1], normal[2]));
        // u is periodic: 0 and 1 are the same seam, and f32 rounding may pick either
        let u_distance = (gpu_uv[0] - u).abs();
        assert!(
            u_distance.min(1.0 - u_distance) < 1e-4 && (gpu_uv[1] - v).abs() < 1e-4,
            "{shader_file}::{function}({normal:?}) = ({}, {}), Rust unit_sphere_to_uv = ({u}, {v})",
            gpu_uv[0],
            gpu_uv[1]
        );
    }
}

#[test]
#[ignore] // Requires GPU (lavapipe works: scripts/test-local.sh -- --ignored)
fn bubble_fragment_uv_matches_the_mesh_convention() {
    assert_uv_matches_rust("bubble.wgsl", "normal_to_branched_uv");
}

#[test]
#[ignore] // Requires GPU (lavapipe works: scripts/test-local.sh -- --ignored)
fn branched_flow_compute_uv_matches_the_mesh_convention() {
    assert_uv_matches_rust("branched_flow_compute.wgsl", "normal_to_uv");
}

#[test]
#[ignore] // Requires GPU (lavapipe works: scripts/test-local.sh -- --ignored)
fn branched_flow_compute_uv_to_sphere_matches_rust() {
    let uvs: Vec<[f32; 4]> = (0..=20)
        .flat_map(|i| (0..=10).map(move |j| [i as f32 / 20.0, j as f32 / 10.0, 0.0, 0.0]))
        .collect();
    let call = "vec4<f32>(uv_to_sphere(input.xy), 0.0)";
    let gpu_directions =
        run_wgsl_function("branched_flow_compute.wgsl", "uv_to_sphere", call, &uvs);
    for (uv, gpu_direction) in uvs.iter().zip(&gpu_directions) {
        let expected = uv_to_unit_sphere(uv[0], uv[1]);
        let actual = glam::Vec3::new(gpu_direction[0], gpu_direction[1], gpu_direction[2]);
        assert!(
            (actual - expected).length() < 1e-4,
            "uv_to_sphere({}, {}) = {actual:?}, Rust uv_to_unit_sphere = {expected:?}",
            uv[0],
            uv[1]
        );
    }
}

#[test]
fn extract_wgsl_function_returns_the_whole_body() {
    let source = "fn a() -> f32 { return 1.0; }\nfn b(x: f32) -> f32 {\n    if (x > 0.0) { return x; }\n    return -x;\n}\nfn c() {}";
    assert_eq!(
        extract_wgsl_function(source, "b"),
        "fn b(x: f32) -> f32 {\n    if (x > 0.0) { return x; }\n    return -x;\n}"
    );
}
