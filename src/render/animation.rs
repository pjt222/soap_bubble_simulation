//! Animation controller for camera orbit, film dynamics, and external forces.
//!
//! Extracted from `pipeline.rs` to separate animation/simulation tick from GPU rendering.

use super::pipeline::BubbleUniform;
use crate::render::camera::Camera;

/// Manages animation state: camera rotation, film time, external forces, and FPS tracking.
pub(crate) struct AnimationController {
    // Camera orbit
    pub rotation_playing: bool,
    pub rotation_speed: f32,
    // Film animation
    pub film_playing: bool,
    pub film_speed: f32,
    // External forces
    pub bubble_velocity: [f32; 3],
    pub wind_strength: f32,
    pub wind_direction: [f32; 3],
    pub buoyancy_strength: f32,
    pub forces_enabled: bool,
    // FPS tracking (circular buffer for O(1) operations)
    frame_times: [f32; 60],
    frame_times_head: usize,
    frame_times_count: usize,
    fps: f32,
    last_dt: f32,
    seconds_since_fps_report: f32,
}

/// How often `update_fps` reports the frame rate for logging.
const FPS_REPORT_INTERVAL_SECONDS: f32 = 5.0;

impl AnimationController {
    pub fn new() -> Self {
        Self {
            rotation_playing: false,
            rotation_speed: 0.5,
            film_playing: true,
            film_speed: 1.0,
            bubble_velocity: [0.0; 3],
            wind_strength: 0.1,
            wind_direction: [1.0, 0.0, 0.0],
            buoyancy_strength: 0.02,
            forces_enabled: false,
            frame_times: [0.0; 60],
            frame_times_head: 0,
            frame_times_count: 0,
            fps: 0.0,
            last_dt: 0.0,
            seconds_since_fps_report: 0.0,
        }
    }

    pub fn fps(&self) -> f32 {
        self.fps
    }

    pub fn last_dt(&self) -> f32 {
        self.last_dt
    }

    /// Update camera rotation animation.
    pub fn update_rotation(&self, camera: &mut Camera, dt: f32) {
        if self.rotation_playing {
            camera.yaw += dt * self.rotation_speed;
            if camera.yaw > std::f32::consts::TAU {
                camera.yaw -= std::f32::consts::TAU;
            } else if camera.yaw < 0.0 {
                camera.yaw += std::f32::consts::TAU;
            }
        }
    }

    /// Update film animation time.
    pub fn update_film_time(&self, bubble_uniform: &mut BubbleUniform, dt: f32) {
        if self.film_playing {
            bubble_uniform.film_time += dt * self.film_speed;
        }
    }

    /// Apply external forces (wind and buoyancy) to bubble position.
    pub fn update_forces(&mut self, bubble_uniform: &mut BubbleUniform, dt: f32) {
        if !self.forces_enabled {
            return;
        }

        // Wind force: F = wind_strength * direction
        let wind_force = [
            self.wind_strength * self.wind_direction[0],
            self.wind_strength * self.wind_direction[1],
            self.wind_strength * self.wind_direction[2],
        ];

        // Buoyancy force: light soap bubble rises (upward in +Y)
        let buoyancy_force = [0.0, self.buoyancy_strength, 0.0];

        // Simple drag to prevent runaway velocity (air resistance)
        let drag = 0.5;

        // Update velocity: v += (F - drag*v) * dt
        for i in 0..3 {
            let total_force = wind_force[i] + buoyancy_force[i] - drag * self.bubble_velocity[i];
            self.bubble_velocity[i] += total_force * dt;
        }

        // Update position: p += v * dt
        bubble_uniform.position_x += self.bubble_velocity[0] * dt;
        bubble_uniform.position_y += self.bubble_velocity[1] * dt;
        bubble_uniform.position_z += self.bubble_velocity[2] * dt;

        // Soft boundary: gradually push bubble back toward center if too far
        let max_distance = 0.15;
        let pos = [
            bubble_uniform.position_x,
            bubble_uniform.position_y,
            bubble_uniform.position_z,
        ];
        let dist_sq = pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2];
        if dist_sq > max_distance * max_distance {
            let dist = dist_sq.sqrt();
            let return_strength = 0.5 * (dist - max_distance);
            for (velocity, &position) in self.bubble_velocity.iter_mut().zip(pos.iter()) {
                *velocity -= return_strength * position / dist * dt;
            }
        }
    }

    /// Reset bubble position and velocity (called when forces are disabled).
    pub fn reset_position(&mut self, bubble_uniform: &mut BubbleUniform) {
        bubble_uniform.position_x = 0.0;
        bubble_uniform.position_y = 0.0;
        bubble_uniform.position_z = 0.0;
        self.bubble_velocity = [0.0, 0.0, 0.0];
    }

    /// Update FPS tracking using circular buffer.
    /// Record a frame time. Every `FPS_REPORT_INTERVAL_SECONDS` of frame time,
    /// returns the current 60-frame average FPS so the caller can log it
    /// (`scripts/gpu/probe-adapters.sh --app` compares adapters with it).
    pub fn update_fps(&mut self, dt: f32) -> Option<f32> {
        self.last_dt = dt;
        self.frame_times[self.frame_times_head] = dt;
        self.frame_times_head = (self.frame_times_head + 1) % 60;
        if self.frame_times_count < 60 {
            self.frame_times_count += 1;
        }
        if self.frame_times_count > 0 {
            let sum: f32 = self.frame_times[..self.frame_times_count].iter().sum();
            let avg_dt = sum / self.frame_times_count as f32;
            self.fps = 1.0 / avg_dt;
        }
        self.seconds_since_fps_report += dt;
        if self.seconds_since_fps_report >= FPS_REPORT_INTERVAL_SECONDS {
            self.seconds_since_fps_report = 0.0;
            Some(self.fps)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // dt = 0.125 s is exact in binary floating point, so 40 frames sum to exactly 5 s.
    const DT: f32 = 0.125;

    #[test]
    fn test_update_fps_reports_once_per_interval() {
        let mut animation = AnimationController::new();
        let reports: Vec<f32> = (0..120).filter_map(|_| animation.update_fps(DT)).collect();
        assert_eq!(reports.len(), 3, "120 frames x 0.125 s = 15 s -> 3 reports");
        for fps in reports {
            assert!((fps - 8.0).abs() < 1e-3, "reported {fps}");
        }
    }

    #[test]
    fn test_update_fps_does_not_report_before_interval() {
        let mut animation = AnimationController::new();
        let silent = (0..39).filter_map(|_| animation.update_fps(DT)).count();
        assert_eq!(silent, 0);
        assert!(
            animation.update_fps(DT).is_some(),
            "40th frame completes 5 s"
        );
    }
}
