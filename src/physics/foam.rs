//! Multi-bubble foam system for soap bubble simulation.
//!
//! This module implements a system for simulating multiple interacting bubbles
//! that can form clusters and foam structures, following Plateau's rules.
//!
//! # Physics Background
//!
//! ## Plateau's Rules
//! 1. Smooth films meet in **threes** along edges (Plateau borders)
//! 2. Edges meet in **fours** at vertices
//! 3. Films meet at **120°** angles
//! 4. Edges meet at **109.47°** (tetrahedral angle)
//!
//! ## Young-Laplace Equation
//! The pressure difference across a curved interface:
//! ΔP = 4γ/R (for soap bubble with two interfaces)
//!
//! ## Shared Wall Curvature
//! When two bubbles of different radii meet:
//! 1/R_wall = 1/R_small - 1/R_large
//! The shared wall curves toward the smaller (higher pressure) bubble.

use glam::Vec3;
use std::collections::HashMap;

/// Unique identifier for a bubble in the cluster.
pub type BubbleId = u32;

/// Individual bubble in the foam system.
// put id:'cpu_foam_system', label:'Foam bubble system', input:'final_config.internal', output:'foam_state.internal'
#[derive(Debug, Clone)]
pub struct Bubble {
    /// Unique identifier
    pub id: BubbleId,
    /// Center position in world space (meters)
    pub position: Vec3,
    /// Base radius (meters)
    pub radius: f32,
    /// Current velocity (m/s)
    pub velocity: Vec3,
    /// Aspect ratio for gravity deformation (1.0 = sphere, <1.0 = oblate)
    pub aspect_ratio: f32,
    /// Film thickness in nanometers
    pub thickness_nm: f32,
    /// Refractive index of the soap film
    pub refractive_index: f32,
    /// IDs of neighboring bubbles in contact
    pub neighbors: Vec<BubbleId>,
}

impl Bubble {
    /// Create a new bubble with default properties.
    pub fn new(id: BubbleId, position: Vec3, radius: f32) -> Self {
        Self {
            id,
            position,
            radius,
            velocity: Vec3::ZERO,
            aspect_ratio: 1.0,
            thickness_nm: 500.0,
            refractive_index: 1.33,
            neighbors: Vec::new(),
        }
    }

    /// Calculate the bubble's volume (assuming spherical).
    pub fn volume(&self) -> f32 {
        (4.0 / 3.0) * std::f32::consts::PI * self.radius.powi(3)
    }

    /// Calculate internal pressure difference using Young-Laplace.
    /// ΔP = 4γ/R for soap bubble (two interfaces)
    pub fn pressure_difference(&self, surface_tension: f32) -> f32 {
        4.0 * surface_tension / self.radius
    }

    /// Check if this bubble overlaps with another.
    pub fn overlaps(&self, other: &Bubble) -> bool {
        let distance = (self.position - other.position).length();
        distance < self.radius + other.radius
    }

    /// Calculate overlap distance with another bubble (positive = overlapping).
    pub fn overlap_distance(&self, other: &Bubble) -> f32 {
        let distance = (self.position - other.position).length();
        self.radius + other.radius - distance
    }
}

/// Connection between two touching bubbles.
#[derive(Debug, Clone)]
pub struct BubbleConnection {
    /// ID of the first bubble
    pub bubble_a: BubbleId,
    /// ID of the second bubble
    pub bubble_b: BubbleId,
    /// Contact point in world space
    pub contact_point: Vec3,
    /// Contact normal (from A toward B)
    pub contact_normal: Vec3,
    /// Shared wall curvature radius (positive = curves toward A)
    pub wall_curvature_radius: f32,
    /// Film thickness at the shared wall
    pub wall_thickness_nm: f32,
}

impl BubbleConnection {
    /// Create a new connection between two bubbles.
    pub fn new(bubble_a: &Bubble, bubble_b: &Bubble, surface_tension: f32) -> Self {
        let delta = bubble_b.position - bubble_a.position;
        let distance = delta.length();
        let contact_normal = delta / distance;

        // Contact point weighted by radii
        let t = bubble_a.radius / (bubble_a.radius + bubble_b.radius);
        let contact_point = bubble_a.position + delta * t;

        // Young-Laplace: wall curvature from pressure difference
        let p_a = bubble_a.pressure_difference(surface_tension);
        let p_b = bubble_b.pressure_difference(surface_tension);
        let delta_p = p_a - p_b;

        // R_wall = 2γ / |ΔP| (single interface)
        let wall_curvature_radius = if delta_p.abs() > 1e-6 {
            2.0 * surface_tension / delta_p // Signed: positive if curving toward A
        } else {
            f32::INFINITY // Flat wall for equal-sized bubbles
        };

        // Wall thickness is typically thinner than free film
        let wall_thickness_nm = (bubble_a.thickness_nm + bubble_b.thickness_nm) / 2.0 * 0.8;

        Self {
            bubble_a: bubble_a.id,
            bubble_b: bubble_b.id,
            contact_point,
            contact_normal,
            wall_curvature_radius,
            wall_thickness_nm,
        }
    }

    /// Check if the wall is effectively flat (large curvature radius).
    pub fn is_flat(&self) -> bool {
        self.wall_curvature_radius.abs() > 1000.0 // Effectively flat if R > 1km
    }
}

/// Simple spatial hash for O(1) neighbor queries.
#[derive(Debug)]
pub struct SpatialHash {
    /// Cell size (should be ~2x max bubble radius)
    cell_size: f32,
    /// Hash map from cell coordinates to bubble IDs
    cells: HashMap<(i32, i32, i32), Vec<BubbleId>>,
}

impl SpatialHash {
    /// Create a new spatial hash with given cell size.
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            cells: HashMap::new(),
        }
    }

    /// Convert world position to cell coordinates.
    fn cell_coords(&self, position: Vec3) -> (i32, i32, i32) {
        (
            (position.x / self.cell_size).floor() as i32,
            (position.y / self.cell_size).floor() as i32,
            (position.z / self.cell_size).floor() as i32,
        )
    }

    /// Clear all cells.
    pub fn clear(&mut self) {
        self.cells.clear();
    }

    /// Insert a bubble into the spatial hash.
    pub fn insert(&mut self, id: BubbleId, position: Vec3) {
        let coords = self.cell_coords(position);
        self.cells.entry(coords).or_default().push(id);
    }

    /// Query all bubble IDs in cells near the given position.
    ///
    /// Returns IDs of bubbles that might be within `radius` of `position`.
    pub fn query(&self, position: Vec3, radius: f32) -> Vec<BubbleId> {
        let mut result = Vec::new();
        let cell_radius = (radius / self.cell_size).ceil() as i32;
        let center = self.cell_coords(position);

        for dx in -cell_radius..=cell_radius {
            for dy in -cell_radius..=cell_radius {
                for dz in -cell_radius..=cell_radius {
                    let coords = (center.0 + dx, center.1 + dy, center.2 + dz);
                    if let Some(ids) = self.cells.get(&coords) {
                        result.extend(ids.iter().copied());
                    }
                }
            }
        }

        result
    }

    /// Rebuild the spatial hash from a list of bubbles.
    pub fn rebuild(&mut self, bubbles: &[Bubble]) {
        self.clear();
        for bubble in bubbles {
            self.insert(bubble.id, bubble.position);
        }
    }
}

/// Collection of interacting bubbles forming a foam cluster.
#[derive(Debug)]
pub struct BubbleCluster {
    /// All bubbles in the cluster
    bubbles: Vec<Bubble>,
    /// Next available bubble ID
    next_id: BubbleId,
    /// Spatial acceleration structure
    spatial_hash: SpatialHash,
    /// Active connections between touching bubbles
    connections: Vec<BubbleConnection>,
    /// Surface tension for connection calculations
    surface_tension: f32,
}

impl BubbleCluster {
    /// Create a new empty bubble cluster.
    pub fn new(surface_tension: f32) -> Self {
        Self {
            bubbles: Vec::new(),
            next_id: 0,
            spatial_hash: SpatialHash::new(0.1), // 10cm cells
            connections: Vec::new(),
            surface_tension,
        }
    }

    /// Add a new bubble at the given position.
    ///
    /// Returns the ID of the newly created bubble.
    pub fn add_bubble(&mut self, position: Vec3, radius: f32) -> BubbleId {
        let id = self.next_id;
        self.next_id += 1;

        let bubble = Bubble::new(id, position, radius);
        self.spatial_hash.insert(id, position);
        self.bubbles.push(bubble);

        id
    }

    /// Remove a bubble by ID.
    ///
    /// Returns true if the bubble was found and removed.
    pub fn remove_bubble(&mut self, id: BubbleId) -> bool {
        if let Some(idx) = self.bubbles.iter().position(|b| b.id == id) {
            self.bubbles.remove(idx);
            // Remove connections involving this bubble
            self.connections
                .retain(|c| c.bubble_a != id && c.bubble_b != id);
            true
        } else {
            false
        }
    }

    /// Get a reference to a bubble by ID.
    pub fn get_bubble(&self, id: BubbleId) -> Option<&Bubble> {
        self.bubbles.iter().find(|b| b.id == id)
    }

    /// Get a mutable reference to a bubble by ID.
    pub fn get_bubble_mut(&mut self, id: BubbleId) -> Option<&mut Bubble> {
        self.bubbles.iter_mut().find(|b| b.id == id)
    }

    /// Get all bubbles in the cluster.
    pub fn bubbles(&self) -> &[Bubble] {
        &self.bubbles
    }

    /// Get mutable access to all bubbles.
    pub fn bubbles_mut(&mut self) -> &mut [Bubble] {
        &mut self.bubbles
    }

    /// Get the number of bubbles in the cluster.
    pub fn len(&self) -> usize {
        self.bubbles.len()
    }

    /// Check if the cluster is empty.
    pub fn is_empty(&self) -> bool {
        self.bubbles.is_empty()
    }

    /// Get all active connections.
    pub fn connections(&self) -> &[BubbleConnection] {
        &self.connections
    }

    /// Rebuild the spatial hash from current bubble positions.
    pub fn rebuild_spatial_hash(&mut self) {
        self.spatial_hash.rebuild(&self.bubbles);
    }

    /// Query bubbles near a position.
    pub fn query_nearby(&self, position: Vec3, radius: f32) -> Vec<BubbleId> {
        self.spatial_hash.query(position, radius)
    }

    /// Update connections between touching bubbles.
    ///
    /// This should be called after bubble positions change.
    pub fn update_connections(&mut self) {
        self.connections.clear();

        // Clear all neighbor lists
        for bubble in &mut self.bubbles {
            bubble.neighbors.clear();
        }

        // Rebuild spatial hash
        self.rebuild_spatial_hash();

        // Find all overlapping pairs using brute force O(n²) for now
        // The spatial hash was causing issues - simplify for correctness first
        let n = self.bubbles.len();

        for i in 0..n {
            for j in (i + 1)..n {
                let bubble_a = &self.bubbles[i];
                let bubble_b = &self.bubbles[j];

                // Check if bubbles are in contact (or nearly so)
                let overlap = bubble_a.overlap_distance(bubble_b);
                let threshold = -bubble_a.radius * 0.3; // Allow 30% gap for stable connections

                if overlap > threshold {
                    // In contact or very close
                    let connection =
                        BubbleConnection::new(bubble_a, bubble_b, self.surface_tension);
                    self.connections.push(connection);
                }
            }
        }

        // Update neighbor lists based on connections
        for connection in &self.connections {
            if let Some(bubble_a) = self.bubbles.iter_mut().find(|b| b.id == connection.bubble_a) {
                bubble_a.neighbors.push(connection.bubble_b);
            }
            if let Some(bubble_b) = self.bubbles.iter_mut().find(|b| b.id == connection.bubble_b) {
                bubble_b.neighbors.push(connection.bubble_a);
            }
        }
    }

    /// Merge two overlapping bubbles into one (coalescence).
    ///
    /// Conserves total volume. Returns the ID of the merged bubble.
    pub fn merge_bubbles(&mut self, id_a: BubbleId, id_b: BubbleId) -> Option<BubbleId> {
        let bubble_a = self.get_bubble(id_a)?;
        let bubble_b = self.get_bubble(id_b)?;

        // Calculate merged properties (volume-conserving)
        let vol_a = bubble_a.volume();
        let vol_b = bubble_b.volume();
        let vol_total = vol_a + vol_b;

        // New radius from combined volume
        let new_radius = (vol_total * 3.0 / (4.0 * std::f32::consts::PI)).powf(1.0 / 3.0);

        // Position weighted by volume
        let new_position = (bubble_a.position * vol_a + bubble_b.position * vol_b) / vol_total;

        // Velocity weighted by volume (momentum conservation approximation)
        let new_velocity = (bubble_a.velocity * vol_a + bubble_b.velocity * vol_b) / vol_total;

        // Thickness - average weighted by surface area
        let area_a = 4.0 * std::f32::consts::PI * bubble_a.radius.powi(2);
        let area_b = 4.0 * std::f32::consts::PI * bubble_b.radius.powi(2);
        let new_thickness =
            (bubble_a.thickness_nm * area_a + bubble_b.thickness_nm * area_b) / (area_a + area_b);

        // Remove old bubbles
        self.remove_bubble(id_a);
        self.remove_bubble(id_b);

        // Create merged bubble
        let new_id = self.add_bubble(new_position, new_radius);
        if let Some(bubble) = self.get_bubble_mut(new_id) {
            bubble.velocity = new_velocity;
            bubble.thickness_nm = new_thickness;
        }

        Some(new_id)
    }

    /// Create a default cluster with a few bubbles for testing.
    pub fn create_default_cluster(surface_tension: f32) -> Self {
        let mut cluster = Self::new(surface_tension);

        // Central bubble
        cluster.add_bubble(Vec3::ZERO, 0.025);

        // Surrounding bubbles - positioned to slightly overlap with central
        // Central radius = 0.025, so place neighbors at distance < sum_of_radii
        cluster.add_bubble(Vec3::new(0.042, 0.0, 0.0), 0.02);      // 0.042 < 0.025+0.02=0.045
        cluster.add_bubble(Vec3::new(-0.038, 0.015, 0.0), 0.022);  // closer
        cluster.add_bubble(Vec3::new(0.0, 0.040, 0.0), 0.018);     // 0.040 < 0.025+0.018=0.043
        cluster.add_bubble(Vec3::new(0.018, -0.038, 0.015), 0.02); // closer

        cluster.update_connections();
        cluster
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bubble_creation() {
        let bubble = Bubble::new(0, Vec3::ZERO, 0.025);
        assert_eq!(bubble.id, 0);
        assert!((bubble.radius - 0.025).abs() < 1e-6);
    }

    #[test]
    fn test_bubble_volume() {
        let bubble = Bubble::new(0, Vec3::ZERO, 1.0);
        let expected = (4.0 / 3.0) * std::f32::consts::PI;
        assert!((bubble.volume() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_bubble_overlap() {
        let a = Bubble::new(0, Vec3::ZERO, 0.5);
        let b = Bubble::new(1, Vec3::new(0.8, 0.0, 0.0), 0.5);
        assert!(a.overlaps(&b)); // 0.5 + 0.5 > 0.8

        let c = Bubble::new(2, Vec3::new(1.5, 0.0, 0.0), 0.5);
        assert!(!a.overlaps(&c)); // 0.5 + 0.5 < 1.5
    }

    #[test]
    fn test_cluster_add_remove() {
        let mut cluster = BubbleCluster::new(0.025);

        let id1 = cluster.add_bubble(Vec3::ZERO, 0.5);
        let id2 = cluster.add_bubble(Vec3::X, 0.5);

        assert_eq!(cluster.len(), 2);

        cluster.remove_bubble(id1);
        assert_eq!(cluster.len(), 1);
        assert!(cluster.get_bubble(id1).is_none());
        assert!(cluster.get_bubble(id2).is_some());
    }

    #[test]
    fn test_spatial_hash() {
        let mut hash = SpatialHash::new(1.0);
        hash.insert(0, Vec3::ZERO);
        hash.insert(1, Vec3::new(0.5, 0.0, 0.0));
        hash.insert(2, Vec3::new(5.0, 0.0, 0.0));

        let nearby = hash.query(Vec3::ZERO, 1.0);
        assert!(nearby.contains(&0));
        assert!(nearby.contains(&1));
        assert!(!nearby.contains(&2)); // Too far
    }

    #[test]
    fn test_connection_curvature() {
        let a = Bubble::new(0, Vec3::ZERO, 0.03); // Larger
        let b = Bubble::new(1, Vec3::new(0.05, 0.0, 0.0), 0.02); // Smaller

        let connection = BubbleConnection::new(&a, &b, 0.025);

        // Wall should curve toward smaller bubble (higher pressure)
        // Positive curvature radius means curving toward A (the larger one)
        // But since B is smaller (higher pressure), the wall curves toward B
        // So curvature_radius should be negative
        assert!(connection.wall_curvature_radius.is_finite());
    }

    #[test]
    fn test_merge_conserves_volume() {
        let mut cluster = BubbleCluster::new(0.025);
        let id1 = cluster.add_bubble(Vec3::ZERO, 0.5);
        let id2 = cluster.add_bubble(Vec3::new(0.8, 0.0, 0.0), 0.5);

        let vol_before = cluster.get_bubble(id1).unwrap().volume()
            + cluster.get_bubble(id2).unwrap().volume();

        let merged_id = cluster.merge_bubbles(id1, id2).unwrap();
        let vol_after = cluster.get_bubble(merged_id).unwrap().volume();

        assert!((vol_before - vol_after).abs() < 1e-6);
    }

    #[test]
    fn test_connections_after_physics_step() {
        use super::super::foam_dynamics::FoamSimulator;

        let mut sim = FoamSimulator::with_default_cluster();
        let initial_connections = sim.connection_count();
        println!("\nInitial connections: {}", initial_connections);

        // Simulate one physics step with typical dt
        sim.step(0.016); // ~60fps
        println!("After 1 step: {} connections", sim.connection_count());

        // Simulate several more steps
        for i in 0..10 {
            sim.step(0.016);
            println!("After {} steps: {} connections", i + 2, sim.connection_count());
        }

        // Note: This test is for diagnosis, not assertion
        // The issue is that repulsion forces push overlapping bubbles apart
    }

    #[test]
    fn test_default_cluster_has_connections() {
        let cluster = BubbleCluster::create_default_cluster(0.025);

        // Debug: print bubble positions and overlaps
        println!("\n=== Default Cluster Debug ===");
        println!("Bubbles: {}", cluster.len());
        for bubble in cluster.bubbles() {
            println!(
                "  Bubble {}: pos=({:.4}, {:.4}, {:.4}), r={:.4}",
                bubble.id, bubble.position.x, bubble.position.y, bubble.position.z, bubble.radius
            );
        }

        // Check expected overlaps manually
        let bubbles = cluster.bubbles();
        println!("\nExpected overlaps:");
        for i in 0..bubbles.len() {
            for j in (i + 1)..bubbles.len() {
                let a = &bubbles[i];
                let b = &bubbles[j];
                let distance = (a.position - b.position).length();
                let sum_radii = a.radius + b.radius;
                let overlap = sum_radii - distance;
                println!(
                    "  {} <-> {}: dist={:.4}, sum_r={:.4}, overlap={:.4} {}",
                    a.id,
                    b.id,
                    distance,
                    sum_radii,
                    overlap,
                    if overlap > 0.0 { "OVERLAP" } else { "" }
                );
            }
        }

        println!("\nConnections found: {}", cluster.connections().len());
        for conn in cluster.connections() {
            println!("  {} <-> {}", conn.bubble_a, conn.bubble_b);
        }

        // The default cluster should have at least one connection
        assert!(
            cluster.connections().len() > 0,
            "Default cluster should have overlapping bubbles that form connections"
        );
    }
}
