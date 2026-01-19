//! Drainage simulation for soap bubble film thickness evolution.
//!
//! This module implements the physics of gravitational drainage in soap films.
//! The film thickness evolves according to a simplified drainage equation:
//!
//! ```text
//! dh/dt = -rho * g * h^3 / (3 * eta) * sin(theta) + D * nabla^2(h)
//! ```
//!
//! Where:
//! - rho = fluid density (kg/m^3)
//! - g = gravitational acceleration (m/s^2)
//! - h = film thickness (m)
//! - eta = dynamic viscosity (Pa*s)
//! - theta = polar angle (0 at top, pi at bottom)
//! - D = diffusion coefficient (m^2/s)

use crate::config::SimulationConfig;

/// A 2D grid storing film thickness values in spherical coordinates.
///
/// The grid uses (theta, phi) coordinates where:
/// - theta: polar angle from 0 (top) to PI (bottom)
/// - phi: azimuthal angle from 0 to 2*PI
#[derive(Debug, Clone)]
pub struct ThicknessField {
    /// Number of grid points in the theta direction
    num_theta: usize,

    /// Number of grid points in the phi direction
    num_phi: usize,

    /// Thickness values stored in row-major order [theta][phi]
    /// Units: meters
    data: Vec<f64>,

    /// Grid spacing in theta direction (radians)
    delta_theta: f64,

    /// Grid spacing in phi direction (radians)
    delta_phi: f64,
}

impl ThicknessField {
    /// Create a new thickness field with uniform initial thickness.
    ///
    /// # Arguments
    /// * `num_theta` - Number of grid points in polar direction
    /// * `num_phi` - Number of grid points in azimuthal direction
    /// * `initial_thickness` - Initial uniform thickness in meters
    pub fn new(num_theta: usize, num_phi: usize, initial_thickness: f64) -> Self {
        let data = vec![initial_thickness; num_theta * num_phi];
        let delta_theta = std::f64::consts::PI / (num_theta - 1) as f64;
        let delta_phi = 2.0 * std::f64::consts::PI / num_phi as f64;

        Self {
            num_theta,
            num_phi,
            data,
            delta_theta,
            delta_phi,
        }
    }

    /// Get the thickness at grid indices (i, j).
    #[inline]
    pub fn get(&self, theta_index: usize, phi_index: usize) -> f64 {
        self.data[theta_index * self.num_phi + phi_index]
    }

    /// Set the thickness at grid indices (i, j).
    #[inline]
    pub fn set(&mut self, theta_index: usize, phi_index: usize, value: f64) {
        self.data[theta_index * self.num_phi + phi_index] = value;
    }

    /// Get the theta angle for a given index.
    #[inline]
    pub fn theta_at(&self, theta_index: usize) -> f64 {
        theta_index as f64 * self.delta_theta
    }

    /// Get the phi angle for a given index.
    #[inline]
    pub fn phi_at(&self, phi_index: usize) -> f64 {
        phi_index as f64 * self.delta_phi
    }

    /// Get the number of theta grid points.
    pub fn num_theta(&self) -> usize {
        self.num_theta
    }

    /// Get the number of phi grid points.
    pub fn num_phi(&self) -> usize {
        self.num_phi
    }

    /// Get grid spacing in theta direction.
    pub fn delta_theta(&self) -> f64 {
        self.delta_theta
    }

    /// Get grid spacing in phi direction.
    pub fn delta_phi(&self) -> f64 {
        self.delta_phi
    }

    /// Get the minimum thickness in the field.
    pub fn min_thickness(&self) -> f64 {
        self.data
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    /// Get the maximum thickness in the field.
    pub fn max_thickness(&self) -> f64 {
        self.data
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get raw data slice for direct access.
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Get mutable raw data slice.
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }
}

/// Simulator for soap film drainage under gravity.
///
/// Evolves the film thickness field over time using a finite difference
/// discretization of the drainage equation.
#[derive(Debug)]
pub struct DrainageSimulator {
    /// Current thickness field
    thickness: ThicknessField,

    /// Scratch buffer for time stepping
    scratch: ThicknessField,

    /// Fluid density (kg/m^3)
    density: f64,

    /// Gravitational acceleration (m/s^2)
    gravity: f64,

    /// Dynamic viscosity (Pa*s)
    viscosity: f64,

    /// Diffusion coefficient (m^2/s)
    diffusion_coefficient: f64,

    /// Critical thickness below which the bubble bursts (m)
    critical_thickness: f64,

    /// Bubble radius for computing surface Laplacian (m)
    bubble_radius: f64,

    /// Current simulation time (s)
    current_time: f64,
}

impl DrainageSimulator {
    /// Create a new drainage simulator from configuration.
    ///
    /// # Arguments
    /// * `config` - Simulation configuration containing physical parameters
    pub fn new(config: &SimulationConfig) -> Self {
        let num_theta = config.resolution as usize;
        let num_phi = config.resolution as usize * 2; // 2x resolution for phi (0 to 2*PI)
        let initial_thickness = config.film_thickness_meters();

        let thickness = ThicknessField::new(num_theta, num_phi, initial_thickness);
        let scratch = thickness.clone();

        Self {
            thickness,
            scratch,
            density: config.fluid.density,
            gravity: config.environment.gravity,
            viscosity: config.fluid.viscosity,
            diffusion_coefficient: config.fluid.surfactant_diffusion,
            critical_thickness: config.bubble.critical_thickness_nm * 1e-9,
            bubble_radius: config.bubble_radius(),
            current_time: 0.0,
        }
    }

    /// Advance the simulation by one time step.
    ///
    /// Uses forward Euler integration with finite differences for the
    /// spatial derivatives.
    ///
    /// # Arguments
    /// * `dt` - Time step in seconds
    pub fn step(&mut self, dt: f64) {
        let num_theta = self.thickness.num_theta();
        let num_phi = self.thickness.num_phi();
        let delta_theta = self.thickness.delta_theta();
        let delta_phi = self.thickness.delta_phi();

        // Precompute drainage coefficient: rho * g / (3 * eta)
        let drainage_coefficient = self.density * self.gravity / (3.0 * self.viscosity);

        // Precompute diffusion coefficient scaled for the surface Laplacian
        let diffusion_scaled = self.diffusion_coefficient;

        // Copy current state to scratch for reading while writing updates
        self.scratch.data_mut().copy_from_slice(self.thickness.data());

        // Update interior points (skip poles at theta=0 and theta=PI)
        for theta_index in 1..(num_theta - 1) {
            let theta = self.thickness.theta_at(theta_index);
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();

            // Avoid division by zero near poles
            let sin_theta_safe = if sin_theta.abs() < 1e-10 {
                1e-10_f64.copysign(sin_theta)
            } else {
                sin_theta
            };

            for phi_index in 0..num_phi {
                // Get current thickness and neighbors
                let thickness_current = self.scratch.get(theta_index, phi_index);

                // Skip if already below critical thickness
                if thickness_current < self.critical_thickness {
                    continue;
                }

                // Theta neighbors (with boundary handling)
                let thickness_theta_minus = self.scratch.get(theta_index - 1, phi_index);
                let thickness_theta_plus = self.scratch.get(theta_index + 1, phi_index);

                // Phi neighbors (periodic boundary)
                let phi_minus = (phi_index + num_phi - 1) % num_phi;
                let phi_plus = (phi_index + 1) % num_phi;
                let thickness_phi_minus = self.scratch.get(theta_index, phi_minus);
                let thickness_phi_plus = self.scratch.get(theta_index, phi_plus);

                // Compute gravitational drainage term: -rho*g*h^3/(3*eta) * sin(theta)
                // The sin(theta) factor accounts for the vertical component of drainage
                let thickness_cubed = thickness_current * thickness_current * thickness_current;
                let drainage_term = -drainage_coefficient * thickness_cubed * sin_theta;

                // Compute surface Laplacian on sphere: nabla^2 h
                // On a sphere of radius R:
                // nabla^2 h = (1/R^2) * [d^2h/dtheta^2 + cot(theta)*dh/dtheta + (1/sin^2(theta))*d^2h/dphi^2]
                let radius_squared = self.bubble_radius * self.bubble_radius;

                // Second derivative in theta
                let d2h_dtheta2 = (thickness_theta_plus - 2.0 * thickness_current + thickness_theta_minus)
                    / (delta_theta * delta_theta);

                // First derivative in theta (for cot(theta) term)
                let dh_dtheta = (thickness_theta_plus - thickness_theta_minus) / (2.0 * delta_theta);

                // Second derivative in phi
                let d2h_dphi2 = (thickness_phi_plus - 2.0 * thickness_current + thickness_phi_minus)
                    / (delta_phi * delta_phi);

                // Surface Laplacian
                let laplacian = (d2h_dtheta2 + cos_theta / sin_theta_safe * dh_dtheta
                    + d2h_dphi2 / (sin_theta_safe * sin_theta_safe))
                    / radius_squared;

                // Diffusion term
                let diffusion_term = diffusion_scaled * laplacian;

                // Time evolution: dh/dt = drainage_term + diffusion_term
                let dh_dt = drainage_term + diffusion_term;

                // Update thickness using forward Euler
                let new_thickness = (thickness_current + dt * dh_dt).max(0.0);
                self.thickness.set(theta_index, phi_index, new_thickness);
            }
        }

        // Handle poles: average neighboring values
        self.update_poles();

        self.current_time += dt;
    }

    /// Update pole values by averaging neighboring ring.
    fn update_poles(&mut self) {
        let num_phi = self.thickness.num_phi();

        // Top pole (theta = 0): average first ring
        let mut top_sum = 0.0;
        for phi_index in 0..num_phi {
            top_sum += self.thickness.get(1, phi_index);
        }
        let top_average = top_sum / num_phi as f64;
        for phi_index in 0..num_phi {
            self.thickness.set(0, phi_index, top_average);
        }

        // Bottom pole (theta = PI): average last ring
        let last_theta = self.thickness.num_theta() - 1;
        let mut bottom_sum = 0.0;
        for phi_index in 0..num_phi {
            bottom_sum += self.thickness.get(last_theta - 1, phi_index);
        }
        let bottom_average = bottom_sum / num_phi as f64;
        for phi_index in 0..num_phi {
            self.thickness.set(last_theta, phi_index, bottom_average);
        }
    }

    /// Get the film thickness at a specific spherical coordinate.
    ///
    /// Uses bilinear interpolation between grid points.
    ///
    /// # Arguments
    /// * `theta` - Polar angle in radians (0 at top, PI at bottom)
    /// * `phi` - Azimuthal angle in radians (0 to 2*PI)
    ///
    /// # Returns
    /// Film thickness in meters
    pub fn get_thickness(&self, theta: f64, phi: f64) -> f64 {
        let num_theta = self.thickness.num_theta();
        let num_phi = self.thickness.num_phi();
        let delta_theta = self.thickness.delta_theta();
        let delta_phi = self.thickness.delta_phi();

        // Clamp theta to valid range
        let theta_clamped = theta.clamp(0.0, std::f64::consts::PI);

        // Normalize phi to [0, 2*PI)
        let phi_normalized = phi.rem_euclid(2.0 * std::f64::consts::PI);

        // Find grid cell
        let theta_continuous = theta_clamped / delta_theta;
        let phi_continuous = phi_normalized / delta_phi;

        let theta_index_low = (theta_continuous.floor() as usize).min(num_theta - 2);
        let phi_index_low = phi_continuous.floor() as usize % num_phi;

        let theta_index_high = theta_index_low + 1;
        let phi_index_high = (phi_index_low + 1) % num_phi;

        // Interpolation weights
        let theta_fraction = theta_continuous - theta_index_low as f64;
        let phi_fraction = phi_continuous - phi_index_low as f64;

        // Bilinear interpolation
        let value_00 = self.thickness.get(theta_index_low, phi_index_low);
        let value_01 = self.thickness.get(theta_index_low, phi_index_high);
        let value_10 = self.thickness.get(theta_index_high, phi_index_low);
        let value_11 = self.thickness.get(theta_index_high, phi_index_high);

        let value_0 = value_00 * (1.0 - phi_fraction) + value_01 * phi_fraction;
        let value_1 = value_10 * (1.0 - phi_fraction) + value_11 * phi_fraction;

        value_0 * (1.0 - theta_fraction) + value_1 * theta_fraction
    }

    /// Get reference to the thickness field.
    pub fn thickness_field(&self) -> &ThicknessField {
        &self.thickness
    }

    /// Get the current simulation time in seconds.
    pub fn current_time(&self) -> f64 {
        self.current_time
    }

    /// Check if any part of the film has reached critical thickness.
    pub fn has_critical_region(&self) -> bool {
        self.thickness.min_thickness() < self.critical_thickness
    }

    /// Get the critical thickness threshold in meters.
    pub fn critical_thickness(&self) -> f64 {
        self.critical_thickness
    }

    /// Reset the simulation to initial uniform thickness.
    pub fn reset(&mut self, initial_thickness: f64) {
        for value in self.thickness.data_mut() {
            *value = initial_thickness;
        }
        self.current_time = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thickness_field_creation() {
        let field = ThicknessField::new(64, 128, 500e-9);
        assert_eq!(field.num_theta(), 64);
        assert_eq!(field.num_phi(), 128);
        assert!((field.get(0, 0) - 500e-9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_thickness_field_get_set() {
        let mut field = ThicknessField::new(64, 128, 500e-9);
        field.set(10, 20, 400e-9);
        assert!((field.get(10, 20) - 400e-9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_drainage_simulator_creation() {
        let config = SimulationConfig::default();
        let simulator = DrainageSimulator::new(&config);

        assert!((simulator.current_time() - 0.0).abs() < f64::EPSILON);
        assert!(!simulator.has_critical_region());
    }

    #[test]
    fn test_drainage_step() {
        let config = SimulationConfig::default();
        let mut simulator = DrainageSimulator::new(&config);

        let initial_time = simulator.current_time();
        simulator.step(0.001);

        assert!((simulator.current_time() - initial_time - 0.001).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_thickness_interpolation() {
        let config = SimulationConfig::default();
        let simulator = DrainageSimulator::new(&config);

        // At initial state, thickness should be uniform
        let thickness_1 = simulator.get_thickness(0.5, 1.0);
        let thickness_2 = simulator.get_thickness(1.5, 3.0);

        assert!((thickness_1 - thickness_2).abs() < 1e-15);
    }

    #[test]
    fn test_drainage_causes_thinning_at_top() {
        let config = SimulationConfig::default();
        let mut simulator = DrainageSimulator::new(&config);

        let initial_top_thickness = simulator.get_thickness(0.1, 0.0);

        // Run several steps
        for _ in 0..100 {
            simulator.step(config.dt);
        }

        let final_top_thickness = simulator.get_thickness(0.1, 0.0);

        // Top of bubble should thin due to drainage
        assert!(
            final_top_thickness < initial_top_thickness,
            "Top should thin: initial={}, final={}",
            initial_top_thickness,
            final_top_thickness
        );
    }

    #[test]
    fn test_phi_periodicity() {
        let config = SimulationConfig::default();
        let simulator = DrainageSimulator::new(&config);

        let thickness_at_0 = simulator.get_thickness(1.0, 0.0);
        let thickness_at_2pi = simulator.get_thickness(1.0, 2.0 * std::f64::consts::PI);

        assert!(
            (thickness_at_0 - thickness_at_2pi).abs() < 1e-12,
            "Phi should be periodic"
        );
    }

    #[test]
    fn test_reset() {
        let config = SimulationConfig::default();
        let mut simulator = DrainageSimulator::new(&config);

        // Run some steps
        for _ in 0..10 {
            simulator.step(config.dt);
        }

        // Reset
        simulator.reset(600e-9);

        assert!((simulator.current_time() - 0.0).abs() < f64::EPSILON);
        assert!((simulator.get_thickness(0.5, 0.5) - 600e-9).abs() < 1e-15);
    }
}
