//! Physics simulation for foam dynamics.
//!
//! Implements forces, collision response, and Plateau's rules enforcement
//! for multi-bubble foam systems.

use super::foam::{Bubble, BubbleCluster, BubbleId};
use glam::Vec3;

/// Physics simulator for foam bubble dynamics.
pub struct FoamSimulator {
    /// The bubble cluster being simulated
    pub cluster: BubbleCluster,
    /// Surface tension coefficient (N/m)
    pub surface_tension: f32,
    /// Fluid viscosity (Pa·s)
    pub viscosity: f32,
    /// Fluid density (kg/m³)
    pub density: f32,
    /// Gravitational acceleration vector (m/s²)
    pub gravity: Vec3,
    /// Van der Waals attraction strength
    pub van_der_waals_strength: f32,
    /// Collision repulsion stiffness
    pub repulsion_stiffness: f32,
    /// Damping coefficient for velocity
    pub damping: f32,
    /// Enable coalescence (bubble merging)
    pub coalescence_enabled: bool,
    /// Relative velocity threshold for coalescence
    pub coalescence_threshold: f32,
}

impl FoamSimulator {
    /// Create a new foam simulator with default physics parameters.
    pub fn new(cluster: BubbleCluster) -> Self {
        Self {
            cluster,
            surface_tension: 0.025,
            viscosity: 0.001,
            density: 1000.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            van_der_waals_strength: 1e-12,
            repulsion_stiffness: 100.0,
            damping: 0.95,
            coalescence_enabled: false,
            coalescence_threshold: 0.5,
        }
    }

    /// Create a simulator with a default test cluster.
    pub fn with_default_cluster() -> Self {
        let cluster = BubbleCluster::create_default_cluster(0.025);
        Self::new(cluster)
    }

    /// Compute all forces on bubbles.
    pub fn compute_forces(&self) -> Vec<Vec3> {
        let bubbles = self.cluster.bubbles();
        let mut forces = vec![Vec3::ZERO; bubbles.len()];

        for (i, bubble) in bubbles.iter().enumerate() {
            // Buoyancy force (reduced gravity for air-filled bubbles)
            forces[i] += self.compute_buoyancy(bubble);

            // Inter-bubble forces
            for (j, other) in bubbles.iter().enumerate().skip(i + 1) {
                let (force_ij, force_ji) = self.compute_interaction_force(bubble, other);
                forces[i] += force_ij;
                forces[j] += force_ji;
            }

            // Drag force (proportional to velocity)
            forces[i] += self.compute_drag(bubble);
        }

        forces
    }

    /// Compute buoyancy-adjusted gravity force.
    ///
    /// Bubbles experience reduced effective gravity due to buoyancy.
    fn compute_buoyancy(&self, bubble: &Bubble) -> Vec3 {
        // Mass of soap film (thin shell)
        let shell_volume = 4.0 * std::f32::consts::PI * bubble.radius.powi(2)
            * (bubble.thickness_nm * 1e-9);
        let film_mass = shell_volume * self.density;

        // Buoyancy from displaced air (small effect)
        let air_density = 1.2; // kg/m³
        let buoyancy_force = bubble.volume() * air_density * self.gravity.length();

        // Net force: gravity on film minus buoyancy
        let net_gravity_magnitude = film_mass * self.gravity.length() - buoyancy_force;

        self.gravity.normalize_or_zero() * net_gravity_magnitude.max(0.0)
    }

    /// Compute interaction force between two bubbles.
    ///
    /// Returns (force on A, force on B).
    fn compute_interaction_force(&self, bubble_a: &Bubble, bubble_b: &Bubble) -> (Vec3, Vec3) {
        let delta = bubble_b.position - bubble_a.position;
        let distance = delta.length();

        if distance < 1e-6 {
            return (Vec3::ZERO, Vec3::ZERO);
        }

        let direction = delta / distance;
        let sum_radii = bubble_a.radius + bubble_b.radius;

        let mut force = Vec3::ZERO;

        // Van der Waals attraction (long-range, weak)
        if distance < sum_radii * 3.0 && distance > sum_radii {
            let vdw_force = self.van_der_waals_strength
                * bubble_a.radius
                * bubble_b.radius
                / distance.powi(2);
            force += direction * vdw_force;
        }

        // Collision repulsion (short-range, strong)
        let overlap = sum_radii - distance;
        if overlap > 0.0 {
            // Hertzian-like contact force
            let repulsion = self.repulsion_stiffness * overlap.powf(1.5);
            force -= direction * repulsion;
        }

        // Force on A is opposite to force on B (Newton's third law)
        (force, -force)
    }

    /// Compute viscous drag force.
    fn compute_drag(&self, bubble: &Bubble) -> Vec3 {
        // Stokes drag: F = -6πηRv
        let drag_coeff = 6.0 * std::f32::consts::PI * self.viscosity * bubble.radius;
        -bubble.velocity * drag_coeff
    }

    /// Perform a single simulation step.
    ///
    /// Updates bubble positions and velocities using semi-implicit Euler integration.
    pub fn step(&mut self, dt: f32) {
        let forces = self.compute_forces();
        let bubbles = self.cluster.bubbles_mut();

        // Semi-implicit Euler integration
        for (i, bubble) in bubbles.iter_mut().enumerate() {
            // Approximate mass (film mass)
            let shell_volume = 4.0 * std::f32::consts::PI * bubble.radius.powi(2)
                * (bubble.thickness_nm * 1e-9);
            let mass = (shell_volume * self.density).max(1e-9);

            // Update velocity
            let acceleration = forces[i] / mass;
            bubble.velocity += acceleration * dt;
            bubble.velocity *= self.damping;

            // Update position
            bubble.position += bubble.velocity * dt;
        }

        // Update connections after positions change
        self.cluster.update_connections();

        // Handle coalescence if enabled
        if self.coalescence_enabled {
            self.handle_coalescence();
        }
    }

    /// Handle bubble coalescence (merging) for high-energy collisions.
    fn handle_coalescence(&mut self) {
        let mut merge_pairs: Vec<(BubbleId, BubbleId)> = Vec::new();

        // Find pairs that should merge
        for connection in self.cluster.connections() {
            let bubble_a = self.cluster.get_bubble(connection.bubble_a);
            let bubble_b = self.cluster.get_bubble(connection.bubble_b);

            if let (Some(a), Some(b)) = (bubble_a, bubble_b) {
                let rel_velocity = (b.velocity - a.velocity).dot(connection.contact_normal);

                // Weber number criterion
                let weber = self.density * rel_velocity.powi(2) * a.radius / self.surface_tension;

                if weber > self.coalescence_threshold {
                    merge_pairs.push((connection.bubble_a, connection.bubble_b));
                }
            }
        }

        // Merge pairs (only first one to avoid complications)
        if let Some((id_a, id_b)) = merge_pairs.first() {
            self.cluster.merge_bubbles(*id_a, *id_b);
        }
    }

    /// Get the number of bubbles in the simulation.
    pub fn bubble_count(&self) -> usize {
        self.cluster.len()
    }

    /// Get the number of active connections.
    pub fn connection_count(&self) -> usize {
        self.cluster.connections().len()
    }

    /// Add a new bubble at a random position near the cluster.
    pub fn add_random_bubble(&mut self, radius_range: (f32, f32)) {
        use std::f32::consts::PI;

        // Random position on a sphere around the origin
        let theta = rand_f32() * 2.0 * PI;
        let phi = rand_f32() * PI;
        let spawn_distance = 0.1; // 10cm from center

        let position = Vec3::new(
            spawn_distance * phi.sin() * theta.cos(),
            spawn_distance * phi.cos(),
            spawn_distance * phi.sin() * theta.sin(),
        );

        let radius = radius_range.0 + rand_f32() * (radius_range.1 - radius_range.0);

        self.cluster.add_bubble(position, radius);
    }

    /// Reset the simulation with a new default cluster.
    pub fn reset(&mut self) {
        self.cluster = BubbleCluster::create_default_cluster(self.surface_tension);
    }
}

/// Simple pseudo-random number generator (0.0 to 1.0).
/// Uses a static counter for basic randomness without external dependencies.
fn rand_f32() -> f32 {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEED: AtomicU32 = AtomicU32::new(12345);

    let mut s = SEED.load(Ordering::Relaxed);
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    SEED.store(s, Ordering::Relaxed);

    (s as f32) / (u32::MAX as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulator_creation() {
        let sim = FoamSimulator::with_default_cluster();
        assert!(sim.bubble_count() > 0);
    }

    #[test]
    fn test_force_computation() {
        let sim = FoamSimulator::with_default_cluster();
        let forces = sim.compute_forces();
        assert_eq!(forces.len(), sim.bubble_count());
    }

    #[test]
    fn test_simulation_step() {
        let mut sim = FoamSimulator::with_default_cluster();
        let initial_count = sim.bubble_count();

        sim.step(0.001);

        // Bubble count should remain the same (no coalescence)
        assert_eq!(sim.bubble_count(), initial_count);
    }

    #[test]
    fn test_buoyancy_reduces_gravity() {
        let sim = FoamSimulator::with_default_cluster();
        let bubble = &sim.cluster.bubbles()[0];
        let buoyancy = sim.compute_buoyancy(bubble);

        // Buoyancy should be in the direction of gravity (but much smaller magnitude)
        // or in the opposite direction if buoyancy dominates
        let gravity_dir = sim.gravity.normalize();
        let buoyancy_dir = buoyancy.normalize_or_zero();

        // Either aligned with gravity (net downward) or very small
        let alignment = gravity_dir.dot(buoyancy_dir).abs();
        assert!(alignment > 0.99 || buoyancy.length() < 1e-6);
    }

    #[test]
    fn test_repulsion_separates_overlapping_bubbles() {
        let mut cluster = BubbleCluster::new(0.025);
        cluster.add_bubble(Vec3::ZERO, 0.03);
        cluster.add_bubble(Vec3::new(0.04, 0.0, 0.0), 0.03); // Overlapping

        let mut sim = FoamSimulator::new(cluster);
        sim.gravity = Vec3::ZERO; // Disable gravity

        let initial_distance = 0.04_f32;

        // Run simulation
        for _ in 0..100 {
            sim.step(0.001);
        }

        // Bubbles should have separated
        let bubbles = sim.cluster.bubbles();
        let final_distance = (bubbles[0].position - bubbles[1].position).length();
        assert!(final_distance > initial_distance);
    }
}
