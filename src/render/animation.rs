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
    frames_since_fps_report: u32,
}

/// Frame rate over one report interval: every frame in it counts once, so
/// consecutive reports are independent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FpsReport {
    /// Mean frames per second over the interval
    pub fps: f32,
    /// Length of the interval in seconds (at least `FPS_REPORT_INTERVAL_SECONDS`)
    pub seconds: f32,
    pub frames: u32,
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
            frames_since_fps_report: 0,
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

    /// Record a frame time. Updates the 60-frame average shown in the UI and,
    /// once at least `FPS_REPORT_INTERVAL_SECONDS` of frame time has passed,
    /// returns the mean frame rate over that interval for logging
    /// (`scripts/gpu/probe-adapters.sh --app` compares adapters with it).
    pub fn update_fps(&mut self, dt: f32) -> Option<FpsReport> {
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
        self.frames_since_fps_report += 1;
        if self.seconds_since_fps_report < FPS_REPORT_INTERVAL_SECONDS {
            return None;
        }
        let report = FpsReport {
            fps: self.frames_since_fps_report as f32 / self.seconds_since_fps_report,
            seconds: self.seconds_since_fps_report,
            frames: self.frames_since_fps_report,
        };
        self.seconds_since_fps_report = 0.0;
        self.frames_since_fps_report = 0;
        Some(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // dt values that are exact in binary floating point, so interval sums are exact.
    const DT: f32 = 0.125;

    #[test]
    fn test_update_fps_reports_once_per_interval() {
        let mut animation = AnimationController::new();
        let reports: Vec<FpsReport> = (0..120).filter_map(|_| animation.update_fps(DT)).collect();
        assert_eq!(reports.len(), 3, "120 frames x 0.125 s = 15 s -> 3 reports");
        for report in reports {
            assert_eq!(report.frames, 40);
            assert_eq!(report.seconds, 5.0);
            assert_eq!(report.fps, 8.0);
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

    #[test]
    fn test_fps_report_is_the_interval_mean() {
        let mut animation = AnimationController::new();
        // Interval 1: 40 frames x 0.125 s.
        let first = (0..40).filter_map(|_| animation.update_fps(DT)).last();
        assert_eq!(first.map(|report| report.fps), Some(8.0));
        // Interval 2: alternating 0.125 / 0.375 s, 20 frames in exactly 5 s -> 4 FPS.
        // The last frame alone would give 1 / 0.375 = 2.67 FPS, and the 60-frame
        // UI average (40 x 0.125 + 20 frames over 10 s) 6 FPS; only the interval
        // mean is 4.
        let second = (0..20)
            .filter_map(|frame| animation.update_fps(if frame % 2 == 0 { 0.125 } else { 0.375 }))
            .last()
            .expect("second interval reported");
        assert_eq!(second.frames, 20);
        assert_eq!(second.seconds, 5.0);
        assert_eq!(second.fps, 4.0);
        assert!(
            (animation.fps() - 6.0).abs() < 1e-4,
            "UI average {}",
            animation.fps()
        );
    }
}
