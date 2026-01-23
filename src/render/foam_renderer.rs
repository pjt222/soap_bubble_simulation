//! Multi-bubble foam renderer using GPU instancing.
//!
//! This module provides efficient rendering of multiple bubbles using
//! hardware instancing to minimize draw calls.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;

use crate::physics::foam::{Bubble, BubbleCluster, BubbleConnection};

/// Maximum number of bubbles that can be rendered.
pub const MAX_BUBBLES: u32 = 64;

/// Per-instance data sent to the GPU for each bubble.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BubbleInstance {
    /// Model matrix row 0
    pub model_0: [f32; 4],
    /// Model matrix row 1
    pub model_1: [f32; 4],
    /// Model matrix row 2
    pub model_2: [f32; 4],
    /// Model matrix row 3
    pub model_3: [f32; 4],
    /// Bubble radius (for thickness calculations)
    pub radius: f32,
    /// Aspect ratio (1.0 = sphere, <1.0 = oblate)
    pub aspect_ratio: f32,
    /// Base film thickness in nanometers
    pub thickness_nm: f32,
    /// Refractive index
    pub refractive_index: f32,
}

impl BubbleInstance {
    /// Create instance data from a Bubble.
    pub fn from_bubble(bubble: &Bubble) -> Self {
        // Create model matrix: translation + scale
        let scale = Mat4::from_scale(Vec3::new(
            bubble.radius,
            bubble.radius * bubble.aspect_ratio,
            bubble.radius,
        ));
        let translation = Mat4::from_translation(bubble.position);
        let model = translation * scale;

        // WGSL mat4x4 constructor takes COLUMNS, so we pass columns not rows
        Self {
            model_0: model.col(0).to_array(),
            model_1: model.col(1).to_array(),
            model_2: model.col(2).to_array(),
            model_3: model.col(3).to_array(),
            radius: bubble.radius,
            aspect_ratio: bubble.aspect_ratio,
            thickness_nm: bubble.thickness_nm,
            refractive_index: bubble.refractive_index,
        }
    }

    /// Returns the vertex buffer layout for instance data.
    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<BubbleInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // Model matrix row 0
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Model matrix row 1
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Model matrix row 2
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Model matrix row 3
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Radius
                wgpu::VertexAttribute {
                    offset: 64,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32,
                },
                // Aspect ratio
                wgpu::VertexAttribute {
                    offset: 68,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32,
                },
                // Thickness
                wgpu::VertexAttribute {
                    offset: 72,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32,
                },
                // Refractive index
                wgpu::VertexAttribute {
                    offset: 76,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

/// Renderer for multi-bubble foam using GPU instancing.
pub struct FoamRenderer {
    /// Instance data buffer
    instance_buffer: wgpu::Buffer,
    /// Current instance data (CPU-side)
    instances: Vec<BubbleInstance>,
    /// Maximum supported instances
    max_instances: u32,
    /// Whether the instance buffer needs updating
    dirty: bool,
}

impl FoamRenderer {
    /// Create a new foam renderer.
    pub fn new(device: &wgpu::Device, max_instances: u32) -> Self {
        let max_instances = max_instances.min(MAX_BUBBLES);

        // Create instance buffer with max capacity
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Foam Instance Buffer"),
            size: (max_instances as usize * std::mem::size_of::<BubbleInstance>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            instance_buffer,
            instances: Vec::with_capacity(max_instances as usize),
            max_instances,
            dirty: true,
        }
    }

    /// Update instance data from a bubble cluster.
    pub fn update_from_cluster(&mut self, cluster: &BubbleCluster) {
        self.instances.clear();

        for bubble in cluster.bubbles().iter().take(self.max_instances as usize) {
            self.instances.push(BubbleInstance::from_bubble(bubble));
        }

        self.dirty = true;
    }

    /// Upload instance data to GPU if dirty.
    pub fn upload(&mut self, queue: &wgpu::Queue) {
        if self.dirty && !self.instances.is_empty() {
            queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&self.instances),
            );
            self.dirty = false;
        }
    }

    /// Get the instance buffer for rendering.
    pub fn instance_buffer(&self) -> &wgpu::Buffer {
        &self.instance_buffer
    }

    /// Get the number of active instances.
    pub fn instance_count(&self) -> u32 {
        self.instances.len() as u32
    }

    /// Check if there are any instances to render.
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Add a single bubble instance.
    pub fn add_instance(&mut self, bubble: &Bubble) {
        if self.instances.len() < self.max_instances as usize {
            self.instances.push(BubbleInstance::from_bubble(bubble));
            self.dirty = true;
        }
    }

    /// Clear all instances.
    pub fn clear(&mut self) {
        self.instances.clear();
        self.dirty = true;
    }

    /// Get instance data for debugging.
    pub fn instances(&self) -> &[BubbleInstance] {
        &self.instances
    }
}

/// Maximum number of shared walls that can be rendered.
pub const MAX_WALLS: u32 = 128;

/// Per-instance data sent to the GPU for each shared wall (Plateau border).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct WallInstance {
    /// Model matrix column 0
    pub model_0: [f32; 4],
    /// Model matrix column 1
    pub model_1: [f32; 4],
    /// Model matrix column 2
    pub model_2: [f32; 4],
    /// Model matrix column 3
    pub model_3: [f32; 4],
    /// Intersection circle radius
    pub disk_radius: f32,
    /// Spherical cap curvature radius (from Young-Laplace)
    pub curvature_radius: f32,
    /// Film thickness in nanometers
    pub thickness_nm: f32,
    /// Refractive index (average of both bubbles)
    pub refractive_index: f32,
}

impl WallInstance {
    /// Create instance data from a BubbleConnection and bubble data.
    pub fn from_connection(
        connection: &BubbleConnection,
        bubble_a: &Bubble,
        bubble_b: &Bubble,
    ) -> Self {
        // Calculate intersection circle radius using sphere-sphere intersection formula
        let distance = (bubble_a.position - bubble_b.position).length();
        let disk_radius = intersection_radius(bubble_a.radius, bubble_b.radius, distance);

        // Build model matrix: translate to contact_point, rotate to align with contact_normal
        let translation = Mat4::from_translation(connection.contact_point);

        // Rotate so local +Z aligns with contact_normal
        let rotation = rotation_from_z_to_direction(connection.contact_normal);
        let model = translation * rotation;

        // Average refractive index
        let refractive_index = (bubble_a.refractive_index + bubble_b.refractive_index) / 2.0;

        Self {
            model_0: model.col(0).to_array(),
            model_1: model.col(1).to_array(),
            model_2: model.col(2).to_array(),
            model_3: model.col(3).to_array(),
            disk_radius,
            curvature_radius: connection.wall_curvature_radius,
            thickness_nm: connection.wall_thickness_nm,
            refractive_index,
        }
    }

    /// Returns the vertex buffer layout for wall instance data.
    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<WallInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // Model matrix column 0
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Model matrix column 1
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Model matrix column 2
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Model matrix column 3
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Disk radius
                wgpu::VertexAttribute {
                    offset: 64,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32,
                },
                // Curvature radius
                wgpu::VertexAttribute {
                    offset: 68,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32,
                },
                // Thickness
                wgpu::VertexAttribute {
                    offset: 72,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32,
                },
                // Refractive index
                wgpu::VertexAttribute {
                    offset: 76,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

/// Calculate the radius of the intersection circle between two overlapping spheres.
///
/// Uses the sphere-sphere intersection formula:
/// r = sqrt((R1+R2+d)(R1+R2-d)(d+R1-R2)(d-R1+R2)) / (2*d)
///
/// Returns 0.0 if spheres don't overlap.
pub fn intersection_radius(radius_a: f32, radius_b: f32, distance: f32) -> f32 {
    let sum = radius_a + radius_b;
    let diff_ab = radius_a - radius_b;
    let diff_ba = radius_b - radius_a;

    // Check if spheres overlap
    if distance >= sum || distance <= diff_ab.abs() {
        return 0.0;
    }

    // Sphere-sphere intersection formula
    let term1 = sum + distance;
    let term2 = sum - distance;
    let term3 = distance + diff_ab;
    let term4 = distance + diff_ba;

    let product = term1 * term2 * term3 * term4;
    if product <= 0.0 {
        return 0.0;
    }

    product.sqrt() / (2.0 * distance)
}

/// Create a rotation matrix that rotates the +Z axis to the given direction.
fn rotation_from_z_to_direction(direction: Vec3) -> Mat4 {
    let dir = direction.normalize();

    // Handle edge cases where direction is parallel to Z
    if (dir.z - 1.0).abs() < 1e-6 {
        return Mat4::IDENTITY;
    }
    if (dir.z + 1.0).abs() < 1e-6 {
        return Mat4::from_rotation_x(std::f32::consts::PI);
    }

    // Calculate rotation from +Z to direction
    let z_axis = Vec3::Z;
    let rotation_axis = z_axis.cross(dir).normalize();
    let angle = z_axis.dot(dir).acos();

    Mat4::from_quat(Quat::from_axis_angle(rotation_axis, angle))
}

/// Vertex for the disk mesh used in wall rendering.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct WallVertex {
    /// Position in local space (unit disk in XY plane)
    pub position: [f32; 3],
    /// Normal (starts as +Z, shader applies curvature)
    pub normal: [f32; 3],
    /// UV coordinates (radial: u = distance from center, v = angle)
    pub uv: [f32; 2],
}

impl WallVertex {
    /// Returns the vertex buffer layout for wall vertices.
    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<WallVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // Position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Normal
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // UV
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

/// Generate a unit disk mesh with radial segments and rings.
///
/// The disk lies in the XY plane, centered at origin, with radius 1.0.
/// Shader will scale by disk_radius and apply spherical cap curvature.
fn generate_disk_mesh(radial_segments: u32, rings: u32) -> (Vec<WallVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Center vertex
    vertices.push(WallVertex {
        position: [0.0, 0.0, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 0.0],
    });

    // Generate ring vertices
    for ring in 1..=rings {
        let r = ring as f32 / rings as f32;
        for seg in 0..radial_segments {
            let theta = (seg as f32 / radial_segments as f32) * std::f32::consts::TAU;
            let x = r * theta.cos();
            let y = r * theta.sin();

            vertices.push(WallVertex {
                position: [x, y, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [r, theta / std::f32::consts::TAU],
            });
        }
    }

    // Generate indices for center triangles (first ring)
    for seg in 0..radial_segments {
        let next_seg = (seg + 1) % radial_segments;
        indices.push(0); // center
        indices.push(1 + seg);
        indices.push(1 + next_seg);
    }

    // Generate indices for ring strips
    for ring in 1..rings {
        let inner_start = 1 + (ring - 1) * radial_segments;
        let outer_start = 1 + ring * radial_segments;

        for seg in 0..radial_segments {
            let next_seg = (seg + 1) % radial_segments;

            let inner_current = inner_start + seg;
            let inner_next = inner_start + next_seg;
            let outer_current = outer_start + seg;
            let outer_next = outer_start + next_seg;

            // Two triangles per quad
            indices.push(inner_current);
            indices.push(outer_current);
            indices.push(outer_next);

            indices.push(inner_current);
            indices.push(outer_next);
            indices.push(inner_next);
        }
    }

    (vertices, indices)
}

/// Renderer for shared walls (Plateau borders) between bubbles.
///
/// Uses instanced rendering with curved disk geometry.
pub struct SharedWallRenderer {
    /// Vertex buffer for unit disk mesh
    vertex_buffer: wgpu::Buffer,
    /// Index buffer for unit disk mesh
    index_buffer: wgpu::Buffer,
    /// Number of indices in the disk mesh
    num_mesh_indices: u32,
    /// Instance data buffer
    instance_buffer: wgpu::Buffer,
    /// Current instance data (CPU-side)
    instances: Vec<WallInstance>,
    /// Maximum supported instances
    max_instances: u32,
    /// Whether the instance buffer needs updating
    dirty: bool,
}

impl SharedWallRenderer {
    /// Create a new shared wall renderer.
    pub fn new(device: &wgpu::Device, max_instances: u32) -> Self {
        let max_instances = max_instances.min(MAX_WALLS);

        // Generate unit disk mesh (16 radial segments, 6 rings = ~176 triangles)
        let (vertices, indices) = generate_disk_mesh(16, 6);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Wall Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Wall Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Wall Instance Buffer"),
            size: (max_instances as usize * std::mem::size_of::<WallInstance>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            vertex_buffer,
            index_buffer,
            num_mesh_indices: indices.len() as u32,
            instance_buffer,
            instances: Vec::with_capacity(max_instances as usize),
            max_instances,
            dirty: true,
        }
    }

    /// Generate wall instances from bubble connections.
    pub fn generate_walls(&mut self, cluster: &BubbleCluster) {
        self.instances.clear();

        for connection in cluster.connections() {
            if self.instances.len() >= self.max_instances as usize {
                break;
            }

            // Get the two bubbles involved in this connection
            let bubble_a = cluster.get_bubble(connection.bubble_a);
            let bubble_b = cluster.get_bubble(connection.bubble_b);

            if let (Some(a), Some(b)) = (bubble_a, bubble_b) {
                let instance = WallInstance::from_connection(connection, a, b);

                // Only add walls with meaningful intersection radius
                if instance.disk_radius > 0.001 {
                    self.instances.push(instance);
                }
            }
        }

        self.dirty = true;
    }

    /// Upload instance data to GPU if dirty.
    pub fn upload(&mut self, queue: &wgpu::Queue) {
        if self.dirty && !self.instances.is_empty() {
            queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&self.instances),
            );
            self.dirty = false;
        }
    }

    /// Check if there are any walls to render.
    pub fn has_walls(&self) -> bool {
        !self.instances.is_empty()
    }

    /// Get the vertex buffer.
    pub fn vertex_buffer(&self) -> &wgpu::Buffer {
        &self.vertex_buffer
    }

    /// Get the index buffer.
    pub fn index_buffer(&self) -> &wgpu::Buffer {
        &self.index_buffer
    }

    /// Get the instance buffer.
    pub fn instance_buffer(&self) -> &wgpu::Buffer {
        &self.instance_buffer
    }

    /// Get the number of mesh indices.
    pub fn num_mesh_indices(&self) -> u32 {
        self.num_mesh_indices
    }

    /// Get the number of wall instances.
    pub fn instance_count(&self) -> u32 {
        self.instances.len() as u32
    }

    /// Get instance data for debugging.
    pub fn instances(&self) -> &[WallInstance] {
        &self.instances
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn test_bubble_instance_size() {
        // Ensure instance struct is properly sized for GPU
        assert_eq!(std::mem::size_of::<BubbleInstance>(), 80); // 16*4 + 4*4 = 80 bytes
    }

    #[test]
    fn test_wall_instance_size() {
        // Ensure wall instance struct is properly sized for GPU
        assert_eq!(std::mem::size_of::<WallInstance>(), 80); // 16*4 + 4*4 = 80 bytes
    }

    #[test]
    fn test_instance_from_bubble() {
        let bubble = Bubble::new(0, Vec3::new(1.0, 2.0, 3.0), 0.5);
        let instance = BubbleInstance::from_bubble(&bubble);

        assert!((instance.radius - 0.5).abs() < 1e-6);
        assert!((instance.aspect_ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_model_matrix_translation() {
        let bubble = Bubble::new(0, Vec3::new(1.0, 2.0, 3.0), 1.0);
        let instance = BubbleInstance::from_bubble(&bubble);

        // WGSL expects columns, so we pass columns:
        // col(0) = [sx, 0, 0, 0]   → model_0
        // col(1) = [0, sy, 0, 0]   → model_1
        // col(2) = [0, 0, sz, 0]   → model_2
        // col(3) = [tx, ty, tz, 1] → model_3
        // Translation is in model_3 (the 4th column)
        assert!((instance.model_3[0] - 1.0).abs() < 1e-6, "tx should be 1.0");
        assert!((instance.model_3[1] - 2.0).abs() < 1e-6, "ty should be 2.0");
        assert!((instance.model_3[2] - 3.0).abs() < 1e-6, "tz should be 3.0");
        assert!((instance.model_3[3] - 1.0).abs() < 1e-6, "w should be 1.0");
    }

    #[test]
    fn test_intersection_radius() {
        // Two identical spheres of radius 1.0, overlapping with distance 1.5
        // Expected intersection radius: sqrt((2+1.5)(2-1.5)(1.5+0)(1.5-0)) / (2*1.5)
        // = sqrt(3.5 * 0.5 * 1.5 * 1.5) / 3.0 = sqrt(3.9375) / 3.0 ≈ 0.661
        let r = intersection_radius(1.0, 1.0, 1.5);
        assert!((r - 0.661).abs() < 0.01, "Expected ~0.661, got {}", r);

        // Non-overlapping spheres
        let r_no_overlap = intersection_radius(1.0, 1.0, 3.0);
        assert!(r_no_overlap < 1e-6, "Non-overlapping should return 0");

        // Fully contained sphere
        let r_contained = intersection_radius(2.0, 0.5, 0.5);
        assert!(r_contained < 1e-6, "Contained sphere should return 0");
    }

    #[test]
    fn test_disk_mesh_generation() {
        let (vertices, indices) = generate_disk_mesh(8, 3);

        // Center vertex + 8 segments * 3 rings = 1 + 24 = 25 vertices
        assert_eq!(vertices.len(), 25);

        // Center triangles: 8
        // Ring strips: 2 rings * 8 segments * 2 triangles = 32
        // Total triangles: 40, indices: 120
        assert_eq!(indices.len(), 120);

        // Check center vertex
        assert_eq!(vertices[0].position, [0.0, 0.0, 0.0]);
        assert_eq!(vertices[0].normal, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_rotation_from_z_to_direction() {
        // Identity case (already pointing +Z)
        let rot = rotation_from_z_to_direction(Vec3::Z);
        let result = rot.transform_vector3(Vec3::Z);
        assert!((result - Vec3::Z).length() < 1e-5);

        // Rotate to +X
        let rot = rotation_from_z_to_direction(Vec3::X);
        let result = rot.transform_vector3(Vec3::Z);
        assert!((result - Vec3::X).length() < 1e-5);

        // Rotate to -Z
        let rot = rotation_from_z_to_direction(-Vec3::Z);
        let result = rot.transform_vector3(Vec3::Z);
        assert!((result - (-Vec3::Z)).length() < 1e-5);
    }
}
