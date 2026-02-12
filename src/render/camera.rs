//! Orbit camera for viewing the soap bubble

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use std::f32::consts::PI;

/// Camera uniform data for GPU
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub _padding: f32,
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0, 0.0, 1.0],
            _padding: 0.0,
        }
    }
}

/// Orbit camera that rotates around a target point
// put id:'cpu_camera_state', label:'Camera view-projection', input:'loop_iteration.internal', output:'uniform_buffers_gpu.internal'
pub struct Camera {
    /// Target point to look at
    pub target: Vec3,
    /// Distance from target
    pub distance: f32,
    /// Horizontal angle (radians)
    pub yaw: f32,
    /// Vertical angle (radians, clamped)
    pub pitch: f32,
    /// Field of view (radians)
    pub fov: f32,
    /// Aspect ratio (width / height)
    pub aspect: f32,
    /// Near clipping plane
    pub near: f32,
    /// Far clipping plane
    pub far: f32,
}

impl Camera {
    /// Create a new camera with given aspect ratio
    pub fn new(aspect: f32) -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 0.2,  // 20cm from center, good for 5cm bubble
            yaw: 0.0,
            pitch: 0.3,     // Slightly above horizontal
            fov: 45.0_f32.to_radians(),
            aspect,
            near: 0.001,
            far: 100.0,
        }
    }

    /// Calculate camera position from orbit parameters
    pub fn position(&self) -> Vec3 {
        let x = self.distance * self.pitch.cos() * self.yaw.sin();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.pitch.cos() * self.yaw.cos();
        self.target + Vec3::new(x, y, z)
    }

    /// Rotate camera around target
    pub fn orbit(&mut self, delta_x: f32, delta_y: f32) {
        self.yaw += delta_x * 0.01;
        self.pitch += delta_y * 0.01;

        // Clamp pitch to avoid gimbal lock
        let max_pitch = PI / 2.0 - 0.01;
        self.pitch = self.pitch.clamp(-max_pitch, max_pitch);
    }

    /// Zoom in/out
    pub fn zoom(&mut self, delta: f32) {
        self.distance *= 1.0 - delta * 0.1;
        self.distance = self.distance.clamp(0.05, 10.0);
    }

    /// Set aspect ratio (call on window resize)
    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
    }

    /// Get the view matrix
    pub fn view_matrix(&self) -> Mat4 {
        let position = self.position();
        Mat4::look_at_rh(position, self.target, Vec3::Y)
    }

    /// Get the projection matrix
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov, self.aspect, self.near, self.far)
    }

    /// Get combined view-projection matrix
    pub fn view_projection_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    /// Get camera uniform for GPU
    pub fn uniform(&self) -> CameraUniform {
        let position = self.position();
        CameraUniform {
            view_proj: self.view_projection_matrix().to_cols_array_2d(),
            camera_pos: position.to_array(),
            _padding: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_creation() {
        let camera = Camera::new(16.0 / 9.0);
        assert!(camera.distance > 0.0);
        assert!(camera.fov > 0.0);
    }

    #[test]
    fn test_camera_position() {
        let camera = Camera::new(1.0);
        let pos = camera.position();
        // Should be at distance from target
        assert!((pos.length() - camera.distance).abs() < 0.001);
    }

    #[test]
    fn test_camera_uniform_size() {
        // CameraUniform should be properly aligned for GPU
        assert_eq!(std::mem::size_of::<CameraUniform>(), 80);
    }

    // === Input Simulation Tests (Phase 2) ===

    #[test]
    fn test_orbit_increases_yaw() {
        let mut camera = Camera::new(1.0);
        let initial_yaw = camera.yaw;
        camera.orbit(100.0, 0.0);
        assert!(
            camera.yaw > initial_yaw,
            "Yaw should increase with positive delta_x"
        );
    }

    #[test]
    fn test_orbit_decreases_yaw() {
        let mut camera = Camera::new(1.0);
        let initial_yaw = camera.yaw;
        camera.orbit(-100.0, 0.0);
        assert!(
            camera.yaw < initial_yaw,
            "Yaw should decrease with negative delta_x"
        );
    }

    #[test]
    fn test_orbit_changes_position() {
        let mut camera = Camera::new(1.0);
        let initial_pos = camera.position();
        camera.orbit(50.0, 50.0);
        let final_pos = camera.position();
        assert!(
            (initial_pos - final_pos).length() > 0.001,
            "Camera position should change after orbit"
        );
    }

    #[test]
    fn test_pitch_clamping_upper() {
        let mut camera = Camera::new(1.0);
        camera.orbit(0.0, 10000.0); // Large upward rotation
        let max_pitch = PI / 2.0 - 0.01;
        assert!(
            camera.pitch <= max_pitch,
            "Pitch should be clamped below PI/2: got {}",
            camera.pitch
        );
    }

    #[test]
    fn test_pitch_clamping_lower() {
        let mut camera = Camera::new(1.0);
        camera.orbit(0.0, -10000.0); // Large downward rotation
        let min_pitch = -(PI / 2.0 - 0.01);
        assert!(
            camera.pitch >= min_pitch,
            "Pitch should be clamped above -PI/2: got {}",
            camera.pitch
        );
    }

    #[test]
    fn test_zoom_in_decreases_distance() {
        let mut camera = Camera::new(1.0);
        let initial_distance = camera.distance;
        camera.zoom(2.0); // Positive delta zooms in
        assert!(
            camera.distance < initial_distance,
            "Distance should decrease when zooming in"
        );
    }

    #[test]
    fn test_zoom_out_increases_distance() {
        let mut camera = Camera::new(1.0);
        let initial_distance = camera.distance;
        camera.zoom(-2.0); // Negative delta zooms out
        assert!(
            camera.distance > initial_distance,
            "Distance should increase when zooming out"
        );
    }

    #[test]
    fn test_zoom_min_bound() {
        let mut camera = Camera::new(1.0);
        camera.zoom(1000.0); // Extreme zoom in
        assert!(
            camera.distance >= 0.05,
            "Distance should not go below 0.05: got {}",
            camera.distance
        );
    }

    #[test]
    fn test_zoom_max_bound() {
        let mut camera = Camera::new(1.0);
        camera.zoom(-1000.0); // Extreme zoom out
        assert!(
            camera.distance <= 10.0,
            "Distance should not exceed 10.0: got {}",
            camera.distance
        );
    }

    #[test]
    fn test_aspect_ratio_update() {
        let mut camera = Camera::new(16.0 / 9.0);
        assert!((camera.aspect - 16.0 / 9.0).abs() < 0.001);

        camera.set_aspect(4.0 / 3.0);
        assert!((camera.aspect - 4.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_view_matrix_is_valid() {
        let camera = Camera::new(1.0);
        let view = camera.view_matrix();
        // View matrix should be invertible (determinant != 0)
        let det = view.determinant();
        assert!(det.abs() > 0.001, "View matrix should be invertible");
    }

    #[test]
    fn test_projection_matrix_is_valid() {
        let camera = Camera::new(1.0);
        let proj = camera.projection_matrix();
        // Projection matrix should be invertible
        let det = proj.determinant();
        assert!(det.abs() > 0.001, "Projection matrix should be invertible");
    }

    #[test]
    fn test_input_sequence_simulation() {
        let mut camera = Camera::new(1.0);
        let initial_pos = camera.position();

        // Simulate a drag sequence: move right 100px in 10 steps
        for _ in 0..10 {
            camera.orbit(10.0, 5.0);
        }

        // Simulate zoom
        camera.zoom(2.0);

        // Verify final state
        let final_pos = camera.position();

        // Position should have changed
        assert!(
            (initial_pos - final_pos).length() > 0.01,
            "Camera should have moved significantly"
        );

        // Camera position should still be at camera.distance from target
        let distance_from_target = (final_pos - camera.target).length();
        assert!(
            (distance_from_target - camera.distance).abs() < 0.001,
            "Camera should remain at correct distance from target"
        );
    }

    #[test]
    fn test_orbit_360_degrees() {
        let mut camera = Camera::new(1.0);

        // Full 360 degree rotation should bring camera back near original position
        let initial_pos = camera.position();

        // 360 degrees = 2*PI radians, orbit sensitivity is 0.01
        // So we need delta_x such that delta_x * 0.01 = 2*PI
        // delta_x = 2*PI / 0.01 = ~628
        for _ in 0..628 {
            camera.orbit(1.0, 0.0);
        }

        let final_pos = camera.position();

        // Should be approximately back to start
        assert!(
            (initial_pos - final_pos).length() < 0.1,
            "360 degree rotation should return to approximate start position"
        );
    }

    #[test]
    fn test_camera_uniform_contents() {
        let camera = Camera::new(1.0);
        let uniform = camera.uniform();

        // Camera position should match
        let pos = camera.position();
        assert!((uniform.camera_pos[0] - pos.x).abs() < 0.001);
        assert!((uniform.camera_pos[1] - pos.y).abs() < 0.001);
        assert!((uniform.camera_pos[2] - pos.z).abs() < 0.001);

        // View-projection matrix should match
        let vp = camera.view_projection_matrix();
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (uniform.view_proj[i][j] - vp.col(i)[j]).abs() < 0.001,
                    "View-projection matrix mismatch at [{},{}]",
                    i,
                    j
                );
            }
        }
    }
}
