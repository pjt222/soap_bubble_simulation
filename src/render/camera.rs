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
}
