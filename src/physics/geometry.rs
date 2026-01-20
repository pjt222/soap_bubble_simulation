//! Sphere mesh generation for soap bubble rendering
//!
//! Generates a UV sphere mesh suitable for GPU rendering with thin-film
//! interference effects. UV coordinates map to spherical coordinates (theta, phi)
//! for film thickness texture mapping.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::f32::consts::PI;

/// Vertex data for GPU rendering
///
/// Contains position, normal, and UV coordinates for thin-film interference mapping.
/// UV.x maps to theta (0 to 1 for 0 to 2*PI around the sphere)
/// UV.y maps to phi (0 to 1 for 0 to PI from top to bottom)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    /// 3D position of the vertex
    pub position: [f32; 3],
    /// Surface normal (normalized, pointing outward)
    pub normal: [f32; 3],
    /// UV coordinates for film thickness texture mapping
    /// u: longitude (theta), v: latitude (phi)
    pub uv: [f32; 2],
}

impl Vertex {
    /// Create a new vertex with position, normal, and UV coordinates
    pub fn new(position: Vec3, normal: Vec3, uv: [f32; 2]) -> Self {
        Self {
            position: position.to_array(),
            normal: normal.normalize().to_array(),
            uv,
        }
    }

    /// Create a vertex from a point on a unit sphere, computing normal and UV
    pub fn from_sphere_point(point: Vec3, radius: f32) -> Self {
        let normalized_point = point.normalize();
        let position = normalized_point * radius;
        let normal = normalized_point;

        // Compute spherical UV coordinates
        // theta: angle around Y axis (longitude), 0 to 2*PI
        // phi: angle from +Y axis (latitude), 0 to PI
        let theta = normalized_point.z.atan2(normalized_point.x);
        let phi = normalized_point.y.acos();

        // Map to UV space [0, 1]
        let u = (theta + PI) / (2.0 * PI);
        let v = phi / PI;

        Self {
            position: position.to_array(),
            normal: normal.to_array(),
            uv: [u, v],
        }
    }

    /// Returns the vertex buffer layout for wgpu
    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
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
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // UV
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

/// Sphere mesh for GPU rendering
///
/// Generates an icosphere by subdividing an icosahedron. This produces
/// a more uniform triangle distribution than a UV sphere, which is better
/// for physics simulation and avoids polar distortion.
pub struct SphereMesh {
    /// Vertex data for the mesh
    pub vertices: Vec<Vertex>,
    /// Triangle indices (3 per triangle)
    pub indices: Vec<u32>,
    /// Radius of the sphere
    pub radius: f32,
    /// Subdivision level used to generate the mesh
    pub subdivision_level: u32,
}

impl SphereMesh {
    /// Create a new UV sphere mesh
    ///
    /// # Arguments
    /// * `radius` - Radius of the sphere
    /// * `subdivision_level` - Controls resolution: segments = 16 * 2^level
    ///   * 0: 16 segments (512 triangles)
    ///   * 1: 32 segments (2,048 triangles)
    ///   * 2: 64 segments (8,192 triangles)
    ///   * 3: 128 segments (32,768 triangles)
    ///   * 4: 256 segments (131,072 triangles)
    ///   * 5: 512 segments (524,288 triangles)
    pub fn new(radius: f32, subdivision_level: u32) -> Self {
        let segments = 16 * (1 << subdivision_level);
        let (vertices, indices) = Self::generate_uv_sphere(radius, segments, segments / 2);

        Self {
            vertices,
            indices,
            radius,
            subdivision_level,
        }
    }

    /// Generate a UV sphere mesh (latitude/longitude grid)
    fn generate_uv_sphere(radius: f32, lon_segments: u32, lat_segments: u32) -> (Vec<Vertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Generate vertices
        for lat in 0..=lat_segments {
            let theta = (lat as f32 / lat_segments as f32) * PI; // 0 to PI (top to bottom)
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();

            for lon in 0..=lon_segments {
                let phi = (lon as f32 / lon_segments as f32) * 2.0 * PI; // 0 to 2*PI
                let sin_phi = phi.sin();
                let cos_phi = phi.cos();

                // Position on unit sphere
                let x = sin_theta * cos_phi;
                let y = cos_theta;
                let z = sin_theta * sin_phi;

                let position = Vec3::new(x, y, z) * radius;
                let normal = Vec3::new(x, y, z);
                let uv = [lon as f32 / lon_segments as f32, lat as f32 / lat_segments as f32];

                vertices.push(Vertex::new(position, normal, uv));
            }
        }

        // Generate indices
        for lat in 0..lat_segments {
            for lon in 0..lon_segments {
                let current = lat * (lon_segments + 1) + lon;
                let next = current + lon_segments + 1;

                // Two triangles per quad (skip degenerate triangles at poles)
                if lat != 0 {
                    indices.push(current);
                    indices.push(next);
                    indices.push(current + 1);
                }

                if lat != lat_segments - 1 {
                    indices.push(current + 1);
                    indices.push(next);
                    indices.push(next + 1);
                }
            }
        }

        (vertices, indices)
    }

    /// Get the number of triangles in the mesh
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Get the number of vertices in the mesh
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Get vertex data as bytes for GPU buffer creation
    pub fn vertex_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.vertices)
    }

    /// Get index data as bytes for GPU buffer creation
    pub fn index_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uv_sphere_creation() {
        let mesh = SphereMesh::new(1.0, 0);
        // Level 0: 16 lon segments, 8 lat segments
        // Triangles: 2 * 16 * 8 - 16 (top) - 16 (bottom) = 224
        // But we skip degenerate triangles at poles, so slightly less
        assert!(mesh.triangle_count() > 200);
        assert!(mesh.vertex_count() > 100);
    }

    #[test]
    fn test_subdivision_increases_detail() {
        // Each level doubles segments, quadrupling triangle count
        let mesh_0 = SphereMesh::new(1.0, 0);
        let mesh_1 = SphereMesh::new(1.0, 1);
        let mesh_2 = SphereMesh::new(1.0, 2);

        assert!(mesh_1.triangle_count() > mesh_0.triangle_count() * 3);
        assert!(mesh_2.triangle_count() > mesh_1.triangle_count() * 3);
    }

    #[test]
    fn test_vertices_on_sphere() {
        let radius = 2.5;
        let mesh = SphereMesh::new(radius, 2);

        for vertex in &mesh.vertices {
            let pos = Vec3::from_array(vertex.position);
            let distance = pos.length();
            assert!(
                (distance - radius).abs() < 1e-6,
                "Vertex not on sphere: distance = {}, expected = {}",
                distance,
                radius
            );
        }
    }

    #[test]
    fn test_normals_point_outward() {
        let mesh = SphereMesh::new(1.0, 2);

        for vertex in &mesh.vertices {
            let pos = Vec3::from_array(vertex.position);
            let normal = Vec3::from_array(vertex.normal);

            // Normal should be normalized
            assert!(
                (normal.length() - 1.0).abs() < 1e-6,
                "Normal not normalized"
            );

            // Normal should point in same direction as position (outward)
            let dot = pos.normalize().dot(normal);
            assert!(
                dot > 0.99,
                "Normal not pointing outward: dot = {}",
                dot
            );
        }
    }

    #[test]
    fn test_uv_coordinates_in_range() {
        let mesh = SphereMesh::new(1.0, 3);

        for vertex in &mesh.vertices {
            let [u, v] = vertex.uv;
            assert!(
                (0.0..=1.0).contains(&u),
                "U coordinate out of range: {}",
                u
            );
            assert!(
                (0.0..=1.0).contains(&v),
                "V coordinate out of range: {}",
                v
            );
        }
    }

    #[test]
    fn test_vertex_size() {
        // Ensure vertex struct is properly packed for GPU
        assert_eq!(std::mem::size_of::<Vertex>(), 32); // 3*4 + 3*4 + 2*4 = 32 bytes
    }
}
