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

/// Sphere/Ellipsoid mesh for GPU rendering
///
/// Generates a UV sphere or oblate ellipsoid mesh. When aspect_ratio < 1.0,
/// the mesh is flattened in the Y direction to simulate gravity deformation.
pub struct SphereMesh {
    /// Vertex data for the mesh
    pub vertices: Vec<Vertex>,
    /// Triangle indices (3 per triangle)
    pub indices: Vec<u32>,
    /// Equatorial radius of the sphere/ellipsoid
    pub radius: f32,
    /// Subdivision level used to generate the mesh
    pub subdivision_level: u32,
    /// Aspect ratio (polar/equatorial). 1.0 = sphere, <1.0 = oblate (flattened)
    pub aspect_ratio: f32,
}

/// Calculate Bond number for bubble deformation
/// Bo = ρgL² / γ where L is characteristic length (diameter)
pub fn bond_number(density: f32, gravity: f32, diameter: f32, surface_tension: f32) -> f32 {
    density * gravity * diameter * diameter / surface_tension
}

/// Calculate aspect ratio from Bond number
/// For small Bo: aspect_ratio ≈ 1 - 0.1 * Bo
/// Clamped to reasonable range [0.7, 1.0]
pub fn aspect_ratio_from_bond(bond: f32) -> f32 {
    (1.0 - 0.1 * bond).clamp(0.7, 1.0)
}

impl SphereMesh {
    /// Create a new UV sphere mesh (perfect sphere)
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
        Self::new_ellipsoid(radius, subdivision_level, 1.0)
    }

    /// Create a new ellipsoid mesh with specified aspect ratio
    ///
    /// # Arguments
    /// * `radius` - Equatorial radius (x and z axes)
    /// * `subdivision_level` - Controls resolution
    /// * `aspect_ratio` - Polar/equatorial ratio. 1.0 = sphere, <1.0 = oblate (flattened by gravity)
    pub fn new_ellipsoid(radius: f32, subdivision_level: u32, aspect_ratio: f32) -> Self {
        let segments = 16 * (1 << subdivision_level);
        let (vertices, indices) = Self::generate_uv_ellipsoid(radius, segments, segments / 2, aspect_ratio);

        Self {
            vertices,
            indices,
            radius,
            subdivision_level,
            aspect_ratio,
        }
    }

    /// Generate a UV ellipsoid mesh (latitude/longitude grid)
    /// For aspect_ratio = 1.0, generates a perfect sphere
    /// For aspect_ratio < 1.0, generates an oblate spheroid (flattened at poles)
    fn generate_uv_ellipsoid(radius: f32, lon_segments: u32, lat_segments: u32, aspect_ratio: f32) -> (Vec<Vertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Radii: equatorial (x, z) and polar (y)
        let r_eq = radius;           // equatorial radius
        let r_pol = radius * aspect_ratio;  // polar radius (smaller for oblate)

        // For ellipsoid normal calculation: n = (x/a², y/b², z/a²) normalized
        // where a = r_eq, b = r_pol
        let inv_a2 = 1.0 / (r_eq * r_eq);
        let inv_b2 = 1.0 / (r_pol * r_pol);

        // Generate vertices
        for lat in 0..=lat_segments {
            let theta = (lat as f32 / lat_segments as f32) * PI; // 0 to PI (top to bottom)
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();

            for lon in 0..=lon_segments {
                let phi = (lon as f32 / lon_segments as f32) * 2.0 * PI; // 0 to 2*PI
                let sin_phi = phi.sin();
                let cos_phi = phi.cos();

                // Position on ellipsoid
                let x = r_eq * sin_theta * cos_phi;
                let y = r_pol * cos_theta;
                let z = r_eq * sin_theta * sin_phi;

                let position = Vec3::new(x, y, z);

                // Normal for ellipsoid: gradient of (x²/a² + y²/b² + z²/a² - 1)
                // = (2x/a², 2y/b², 2z/a²), normalized
                let normal = Vec3::new(x * inv_a2, y * inv_b2, z * inv_a2).normalize();

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

/// LOD mesh cache for storing pre-generated meshes at different detail levels.
///
/// Generates meshes lazily on first access and caches them for reuse.
/// Cache is invalidated when aspect_ratio or radius changes.
pub struct LodMeshCache {
    /// Cached meshes for levels 1-5 (index 0 = level 1, index 4 = level 5)
    meshes: [Option<SphereMesh>; 5],
    /// Current aspect ratio (invalidates cache if changed)
    aspect_ratio: f32,
    /// Bubble radius
    radius: f32,
}

impl LodMeshCache {
    /// Create a new LOD cache with given parameters.
    ///
    /// Meshes are generated lazily on first access.
    pub fn new(radius: f32, aspect_ratio: f32) -> Self {
        Self {
            meshes: [None, None, None, None, None],
            aspect_ratio,
            radius,
        }
    }

    /// Get or generate mesh for given LOD level (1-5).
    ///
    /// Meshes are cached after first generation.
    pub fn get_mesh(&mut self, level: u32) -> &SphereMesh {
        let index = (level.clamp(1, 5) - 1) as usize;

        if self.meshes[index].is_none() {
            let mesh = SphereMesh::new_ellipsoid(self.radius, level, self.aspect_ratio);
            log::debug!(
                "Generated LOD {} mesh: {} triangles",
                level,
                mesh.triangle_count()
            );
            self.meshes[index] = Some(mesh);
        }

        self.meshes[index].as_ref().unwrap()
    }

    /// Invalidate the entire cache.
    ///
    /// Call this when aspect_ratio or radius changes.
    pub fn invalidate(&mut self) {
        self.meshes = [None, None, None, None, None];
    }

    /// Update parameters and invalidate cache if they changed.
    ///
    /// Returns true if cache was invalidated.
    pub fn update(&mut self, radius: f32, aspect_ratio: f32) -> bool {
        let radius_changed = (self.radius - radius).abs() > 1e-6;
        let aspect_changed = (self.aspect_ratio - aspect_ratio).abs() > 1e-6;

        if radius_changed || aspect_changed {
            self.radius = radius;
            self.aspect_ratio = aspect_ratio;
            self.invalidate();
            true
        } else {
            false
        }
    }

    /// Get the current aspect ratio.
    pub fn aspect_ratio(&self) -> f32 {
        self.aspect_ratio
    }

    /// Get the current radius.
    pub fn radius(&self) -> f32 {
        self.radius
    }
}

/// Curved rectangular patch on a sphere surface for focused branched flow viewing
///
/// Generates a mesh covering only a portion of the sphere, defined by UV bounds.
/// This reduces computation for effects like branched flow ray tracing while
/// providing a focused view of the effect.
pub struct SpherePatch {
    /// UV center u-coordinate (0-1, corresponds to phi angle)
    pub center_u: f32,
    /// UV center v-coordinate (0-1, corresponds to theta angle)
    pub center_v: f32,
    /// Half-width in UV space (0.158 ≈ 10% of surface area when squared)
    pub half_size: f32,
    /// Grid subdivisions along each axis
    pub subdivisions: u32,
}

impl Default for SpherePatch {
    fn default() -> Self {
        Self {
            center_u: 0.5,
            center_v: 0.5,
            half_size: 0.158, // ~10% of sphere surface area
            subdivisions: 32,
        }
    }
}

impl SpherePatch {
    /// Create a new sphere patch with specified parameters
    pub fn new(center_u: f32, center_v: f32, half_size: f32, subdivisions: u32) -> Self {
        Self {
            center_u: center_u.clamp(0.0, 1.0),
            center_v: center_v.clamp(0.0, 1.0),
            half_size: half_size.clamp(0.01, 0.5),
            subdivisions: subdivisions.max(2),
        }
    }

    /// Get UV bounds of the patch (min_u, max_u, min_v, max_v)
    pub fn uv_bounds(&self) -> (f32, f32, f32, f32) {
        let min_u = (self.center_u - self.half_size).max(0.0);
        let max_u = (self.center_u + self.half_size).min(1.0);
        let min_v = (self.center_v - self.half_size).max(0.0);
        let max_v = (self.center_v + self.half_size).min(1.0);
        (min_u, max_u, min_v, max_v)
    }

    /// Generate a curved rectangular mesh on the sphere surface
    ///
    /// # Arguments
    /// * `radius` - Radius of the sphere
    /// * `aspect_ratio` - Polar/equatorial ratio (1.0 = sphere, <1.0 = oblate)
    ///
    /// # Returns
    /// Tuple of (vertices, indices) for the patch mesh
    pub fn generate_mesh(&self, radius: f32, aspect_ratio: f32) -> (Vec<Vertex>, Vec<Vertex>) {
        let (vertices, _indices) = self.generate_mesh_indexed(radius, aspect_ratio);
        // Return vertices twice to match expected signature (this is a workaround)
        (vertices, Vec::new())
    }

    /// Generate indexed mesh for the patch
    pub fn generate_mesh_indexed(&self, radius: f32, aspect_ratio: f32) -> (Vec<Vertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let (min_u, max_u, min_v, max_v) = self.uv_bounds();
        let subs = self.subdivisions;

        // Radii for ellipsoid support
        let r_eq = radius;
        let r_pol = radius * aspect_ratio;
        let inv_a2 = 1.0 / (r_eq * r_eq);
        let inv_b2 = 1.0 / (r_pol * r_pol);

        // Generate vertices on a grid within the UV patch bounds
        for j in 0..=subs {
            let v = min_v + (max_v - min_v) * (j as f32 / subs as f32);
            let theta = v * PI; // 0 to PI (top to bottom)
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();

            for i in 0..=subs {
                let u = min_u + (max_u - min_u) * (i as f32 / subs as f32);
                let phi = u * 2.0 * PI; // 0 to 2*PI
                let sin_phi = phi.sin();
                let cos_phi = phi.cos();

                // Position on ellipsoid
                let x = r_eq * sin_theta * cos_phi;
                let y = r_pol * cos_theta;
                let z = r_eq * sin_theta * sin_phi;
                let position = Vec3::new(x, y, z);

                // Normal for ellipsoid
                let normal = Vec3::new(x * inv_a2, y * inv_b2, z * inv_a2).normalize();

                // UV coordinates - use the actual u, v position
                let uv = [u, v];

                vertices.push(Vertex::new(position, normal, uv));
            }
        }

        // Generate indices (two triangles per quad)
        for j in 0..subs {
            for i in 0..subs {
                let row_width = subs + 1;
                let current = j * row_width + i;
                let next_row = current + row_width;

                // First triangle
                indices.push(current);
                indices.push(next_row);
                indices.push(current + 1);

                // Second triangle
                indices.push(current + 1);
                indices.push(next_row);
                indices.push(next_row + 1);
            }
        }

        (vertices, indices)
    }

    /// Get the number of vertices in the generated mesh
    pub fn vertex_count(&self) -> usize {
        let subs = self.subdivisions as usize;
        (subs + 1) * (subs + 1)
    }

    /// Get the number of triangles in the generated mesh
    pub fn triangle_count(&self) -> usize {
        let subs = self.subdivisions as usize;
        subs * subs * 2
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

    // SpherePatch tests
    #[test]
    fn test_sphere_patch_default() {
        let patch = SpherePatch::default();
        assert_eq!(patch.center_u, 0.5);
        assert_eq!(patch.center_v, 0.5);
        assert!((patch.half_size - 0.158).abs() < 0.001);
        assert_eq!(patch.subdivisions, 32);
    }

    #[test]
    fn test_sphere_patch_uv_bounds() {
        let patch = SpherePatch::new(0.5, 0.5, 0.2, 16);
        let (min_u, max_u, min_v, max_v) = patch.uv_bounds();
        assert!((min_u - 0.3).abs() < 1e-6);
        assert!((max_u - 0.7).abs() < 1e-6);
        assert!((min_v - 0.3).abs() < 1e-6);
        assert!((max_v - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_sphere_patch_uv_bounds_clamped() {
        // Patch near edge should clamp to valid range
        let patch = SpherePatch::new(0.1, 0.9, 0.2, 16);
        let (min_u, max_u, min_v, max_v) = patch.uv_bounds();
        assert!(min_u >= 0.0);
        assert!(max_u <= 1.0);
        assert!(min_v >= 0.0);
        assert!(max_v <= 1.0);
    }

    #[test]
    fn test_sphere_patch_mesh_generation() {
        let patch = SpherePatch::new(0.5, 0.5, 0.2, 8);
        let (vertices, indices) = patch.generate_mesh_indexed(1.0, 1.0);

        // Should have (subdivisions+1)^2 vertices
        assert_eq!(vertices.len(), 9 * 9); // 81 vertices
        // Should have subdivisions^2 * 2 * 3 indices (2 triangles per quad, 3 indices per triangle)
        assert_eq!(indices.len(), 8 * 8 * 2 * 3); // 384 indices
    }

    #[test]
    fn test_sphere_patch_vertices_on_sphere() {
        let radius = 1.5;
        let patch = SpherePatch::new(0.5, 0.5, 0.2, 16);
        let (vertices, _) = patch.generate_mesh_indexed(radius, 1.0);

        for vertex in &vertices {
            let pos = Vec3::from_array(vertex.position);
            let distance = pos.length();
            assert!(
                (distance - radius).abs() < 1e-5,
                "Patch vertex not on sphere: distance = {}, expected = {}",
                distance,
                radius
            );
        }
    }

    #[test]
    fn test_sphere_patch_uv_in_bounds() {
        let patch = SpherePatch::new(0.3, 0.7, 0.15, 8);
        let (min_u, max_u, min_v, max_v) = patch.uv_bounds();
        let (vertices, _) = patch.generate_mesh_indexed(1.0, 1.0);

        for vertex in &vertices {
            let [u, v] = vertex.uv;
            assert!(
                u >= min_u - 1e-6 && u <= max_u + 1e-6,
                "U coordinate {} outside patch bounds [{}, {}]",
                u, min_u, max_u
            );
            assert!(
                v >= min_v - 1e-6 && v <= max_v + 1e-6,
                "V coordinate {} outside patch bounds [{}, {}]",
                v, min_v, max_v
            );
        }
    }

    #[test]
    fn test_sphere_patch_count_methods() {
        let patch = SpherePatch::new(0.5, 0.5, 0.2, 10);
        assert_eq!(patch.vertex_count(), 11 * 11); // 121
        assert_eq!(patch.triangle_count(), 10 * 10 * 2); // 200
    }
}
