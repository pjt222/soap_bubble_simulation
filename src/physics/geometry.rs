//! Sphere mesh generation for soap bubble rendering
//!
//! Generates an icosphere mesh suitable for GPU rendering with thin-film
//! interference effects. UV coordinates map to spherical coordinates (theta, phi)
//! for film thickness texture mapping.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::collections::HashMap;
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
    /// Create a new icosphere mesh
    ///
    /// # Arguments
    /// * `radius` - Radius of the sphere
    /// * `subdivision_level` - Number of subdivision iterations (0 = icosahedron,
    ///   each level quadruples the triangle count)
    ///
    /// # Triangle counts by subdivision level
    /// * 0: 20 triangles (icosahedron)
    /// * 1: 80 triangles
    /// * 2: 320 triangles
    /// * 3: 1,280 triangles
    /// * 4: 5,120 triangles
    /// * 5: 20,480 triangles
    pub fn new(radius: f32, subdivision_level: u32) -> Self {
        let (vertices, indices) = Self::generate_icosphere(radius, subdivision_level);

        Self {
            vertices,
            indices,
            radius,
            subdivision_level,
        }
    }

    /// Generate an icosphere mesh
    fn generate_icosphere(radius: f32, subdivision_level: u32) -> (Vec<Vertex>, Vec<u32>) {
        // Start with a unit icosahedron
        let (mut positions, mut indices) = Self::create_icosahedron();

        // Subdivide the icosahedron
        for _ in 0..subdivision_level {
            (positions, indices) = Self::subdivide(&positions, &indices);
        }

        // Create vertices with proper normals and UVs
        let vertices: Vec<Vertex> = positions
            .iter()
            .map(|&pos| Vertex::from_sphere_point(pos, radius))
            .collect();

        (vertices, indices)
    }

    /// Create the initial icosahedron vertices and indices
    fn create_icosahedron() -> (Vec<Vec3>, Vec<u32>) {
        // Golden ratio
        let phi = (1.0 + 5.0_f32.sqrt()) / 2.0;

        // Normalize so all vertices are on unit sphere
        let scale = 1.0 / (1.0 + phi * phi).sqrt();
        let a = scale;
        let b = phi * scale;

        // 12 vertices of an icosahedron
        let positions = vec![
            Vec3::new(-a, b, 0.0),
            Vec3::new(a, b, 0.0),
            Vec3::new(-a, -b, 0.0),
            Vec3::new(a, -b, 0.0),
            Vec3::new(0.0, -a, b),
            Vec3::new(0.0, a, b),
            Vec3::new(0.0, -a, -b),
            Vec3::new(0.0, a, -b),
            Vec3::new(b, 0.0, -a),
            Vec3::new(b, 0.0, a),
            Vec3::new(-b, 0.0, -a),
            Vec3::new(-b, 0.0, a),
        ];

        // 20 triangular faces
        let indices = vec![
            0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7,
            6, 7, 1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10,
            8, 6, 7, 9, 8, 1,
        ];

        (positions, indices)
    }

    /// Subdivide the mesh by splitting each triangle into 4 triangles
    fn subdivide(positions: &[Vec3], indices: &[u32]) -> (Vec<Vec3>, Vec<u32>) {
        let mut new_positions = positions.to_vec();
        let mut new_indices = Vec::with_capacity(indices.len() * 4);
        let mut midpoint_cache: HashMap<(u32, u32), u32> = HashMap::new();

        // Helper to get or create midpoint vertex
        let mut get_midpoint = |p1_idx: u32, p2_idx: u32| -> u32 {
            // Use ordered pair as key to avoid duplicates
            let key = if p1_idx < p2_idx {
                (p1_idx, p2_idx)
            } else {
                (p2_idx, p1_idx)
            };

            if let Some(&idx) = midpoint_cache.get(&key) {
                return idx;
            }

            // Create new midpoint vertex on unit sphere
            let p1 = new_positions[p1_idx as usize];
            let p2 = new_positions[p2_idx as usize];
            let midpoint = ((p1 + p2) / 2.0).normalize();

            let new_idx = new_positions.len() as u32;
            new_positions.push(midpoint);
            midpoint_cache.insert(key, new_idx);

            new_idx
        };

        // Process each triangle
        for triangle in indices.chunks(3) {
            let v0 = triangle[0];
            let v1 = triangle[1];
            let v2 = triangle[2];

            // Get midpoints of each edge
            let m01 = get_midpoint(v0, v1);
            let m12 = get_midpoint(v1, v2);
            let m20 = get_midpoint(v2, v0);

            // Create 4 new triangles
            // Corner triangles
            new_indices.extend_from_slice(&[v0, m01, m20]);
            new_indices.extend_from_slice(&[v1, m12, m01]);
            new_indices.extend_from_slice(&[v2, m20, m12]);
            // Center triangle
            new_indices.extend_from_slice(&[m01, m12, m20]);
        }

        (new_positions, new_indices)
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
    fn test_icosahedron_creation() {
        let mesh = SphereMesh::new(1.0, 0);
        assert_eq!(mesh.triangle_count(), 20);
        assert_eq!(mesh.vertex_count(), 12);
    }

    #[test]
    fn test_subdivision_levels() {
        // Each subdivision quadruples triangle count
        let mesh_0 = SphereMesh::new(1.0, 0);
        let mesh_1 = SphereMesh::new(1.0, 1);
        let mesh_2 = SphereMesh::new(1.0, 2);

        assert_eq!(mesh_0.triangle_count(), 20);
        assert_eq!(mesh_1.triangle_count(), 80);
        assert_eq!(mesh_2.triangle_count(), 320);
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
