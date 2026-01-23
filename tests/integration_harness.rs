//! Integration test harness for soap bubble simulation
//!
//! Uses headless rendering to test the full render pipeline without
//! requiring a display or window.

use soap_bubble_sim::render::HeadlessRenderPipeline;

/// Test harness for integration testing
pub struct TestHarness {
    pipeline: HeadlessRenderPipeline,
    frames: Vec<Vec<u8>>,
}

/// Steps that can be executed in a test scenario
#[derive(Debug, Clone)]
pub enum TestStep {
    /// Render a frame and capture the pixels
    RenderFrame,
    /// Orbit the camera by (delta_x, delta_y)
    OrbitCamera(f32, f32),
    /// Zoom the camera by delta
    ZoomCamera(f32),
    /// Set the film thickness in nanometers
    SetThickness(f32),
    /// Set simulation time
    SetTime(f32),
    /// Set camera distance directly
    SetCameraDistance(f32),
}

impl TestHarness {
    /// Create a new test harness with specified render dimensions
    pub async fn new(width: u32, height: u32) -> Option<Self> {
        let pipeline = HeadlessRenderPipeline::new(width, height).await?;
        Some(Self {
            pipeline,
            frames: Vec::new(),
        })
    }

    /// Render a frame and return the pixel data
    pub fn render_frame(&mut self) -> &[u8] {
        let frame = self.pipeline.render_to_buffer();
        self.frames.push(frame);
        self.frames.last().unwrap()
    }

    /// Get the last rendered frame
    pub fn last_frame(&self) -> Option<&[u8]> {
        self.frames.last().map(|v| v.as_slice())
    }

    /// Get all captured frames
    pub fn frames(&self) -> &[Vec<u8>] {
        &self.frames
    }

    /// Clear captured frames
    pub fn clear_frames(&mut self) {
        self.frames.clear();
    }

    /// Orbit the camera
    pub fn orbit_camera(&mut self, dx: f32, dy: f32) {
        self.pipeline.orbit_camera(dx, dy);
    }

    /// Zoom the camera
    pub fn zoom_camera(&mut self, delta: f32) {
        self.pipeline.zoom_camera(delta);
    }

    /// Set film thickness in nanometers
    pub fn set_thickness(&mut self, nm: f32) {
        self.pipeline.set_thickness(nm);
    }

    /// Set simulation time
    pub fn set_time(&mut self, time: f32) {
        self.pipeline.set_time(time);
    }

    /// Set camera distance directly
    pub fn set_camera_distance(&mut self, distance: f32) {
        self.pipeline.set_camera_distance(distance);
    }

    /// Get render dimensions
    pub fn size(&self) -> (u32, u32) {
        self.pipeline.size()
    }

    /// Run a sequence of test steps and return captured frames
    pub fn run_scenario(&mut self, steps: &[TestStep]) -> Vec<Vec<u8>> {
        let mut captured = Vec::new();

        for step in steps {
            match step {
                TestStep::RenderFrame => {
                    let frame = self.pipeline.render_to_buffer();
                    captured.push(frame.clone());
                    self.frames.push(frame);
                }
                TestStep::OrbitCamera(dx, dy) => {
                    self.pipeline.orbit_camera(*dx, *dy);
                }
                TestStep::ZoomCamera(delta) => {
                    self.pipeline.zoom_camera(*delta);
                }
                TestStep::SetThickness(nm) => {
                    self.pipeline.set_thickness(*nm);
                }
                TestStep::SetTime(t) => {
                    self.pipeline.set_time(*t);
                }
                TestStep::SetCameraDistance(d) => {
                    self.pipeline.set_camera_distance(*d);
                }
            }
        }

        captured
    }

    /// Get access to the underlying pipeline for advanced testing
    pub fn pipeline(&self) -> &HeadlessRenderPipeline {
        &self.pipeline
    }

    /// Get mutable access to the underlying pipeline
    pub fn pipeline_mut(&mut self) -> &mut HeadlessRenderPipeline {
        &mut self.pipeline
    }
}

/// Helper to compute the average color of a frame
pub fn average_color(pixels: &[u8]) -> (f64, f64, f64, f64) {
    if pixels.is_empty() || pixels.len() % 4 != 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let pixel_count = pixels.len() / 4;
    let mut r_sum = 0u64;
    let mut g_sum = 0u64;
    let mut b_sum = 0u64;
    let mut a_sum = 0u64;

    for chunk in pixels.chunks_exact(4) {
        r_sum += chunk[0] as u64;
        g_sum += chunk[1] as u64;
        b_sum += chunk[2] as u64;
        a_sum += chunk[3] as u64;
    }

    (
        r_sum as f64 / pixel_count as f64,
        g_sum as f64 / pixel_count as f64,
        b_sum as f64 / pixel_count as f64,
        a_sum as f64 / pixel_count as f64,
    )
}

/// Helper to check if two frames are identical
pub fn frames_identical(a: &[u8], b: &[u8]) -> bool {
    a == b
}

/// Helper to compute the percentage of pixels that differ
pub fn frame_diff_ratio(a: &[u8], b: &[u8], tolerance: u8) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 1.0;
    }

    let pixel_count = a.len() / 4;
    let mut diff_count = 0;

    for i in (0..a.len()).step_by(4) {
        let dr = (a[i] as i32 - b[i] as i32).abs();
        let dg = (a[i + 1] as i32 - b[i + 1] as i32).abs();
        let db = (a[i + 2] as i32 - b[i + 2] as i32).abs();

        if dr > tolerance as i32 || dg > tolerance as i32 || db > tolerance as i32 {
            diff_count += 1;
        }
    }

    diff_count as f64 / pixel_count as f64
}

// ============================================================================
// Integration Tests
// ============================================================================

#[tokio::test]
async fn test_harness_creation() {
    let harness = TestHarness::new(256, 256).await;
    // May fail on systems without GPU
    if let Some(h) = harness {
        assert_eq!(h.size(), (256, 256));
    }
}

#[tokio::test]
async fn test_full_render_cycle() {
    let harness = TestHarness::new(256, 256).await;
    if harness.is_none() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    let mut harness = harness.unwrap();
    let frame = harness.render_frame();

    assert!(!frame.is_empty(), "Frame should not be empty");
    assert_eq!(
        frame.len(),
        256 * 256 * 4,
        "Frame should be 256x256 RGBA"
    );
}

#[tokio::test]
async fn test_frame_has_content() {
    let harness = TestHarness::new(256, 256).await;
    if harness.is_none() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    let mut harness = harness.unwrap();
    let frame = harness.render_frame();

    // Frame should not be all zeros (completely black)
    let all_zero = frame.iter().all(|&b| b == 0);
    assert!(!all_zero, "Frame should have some content (not all black)");
}

#[tokio::test]
async fn test_camera_orbit_changes_view() {
    let harness = TestHarness::new(256, 256).await;
    if harness.is_none() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    let mut harness = harness.unwrap();

    // Render initial frame
    let frame1 = harness.render_frame().to_vec();

    // Orbit camera
    harness.orbit_camera(100.0, 0.0);

    // Render new frame
    let frame2 = harness.render_frame().to_vec();

    // Frames should be different
    assert_ne!(frame1, frame2, "Camera orbit should change the rendered view");
}

#[tokio::test]
async fn test_zoom_changes_view() {
    let harness = TestHarness::new(256, 256).await;
    if harness.is_none() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    let mut harness = harness.unwrap();

    // Render initial frame
    let frame1 = harness.render_frame().to_vec();

    // Zoom in
    harness.zoom_camera(3.0);

    // Render new frame
    let frame2 = harness.render_frame().to_vec();

    // Frames should be different
    assert_ne!(frame1, frame2, "Zoom should change the rendered view");
}

#[tokio::test]
async fn test_thickness_changes_appearance() {
    let harness = TestHarness::new(256, 256).await;
    if harness.is_none() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    let mut harness = harness.unwrap();

    // Render at default thickness
    harness.set_thickness(500.0);
    let frame1 = harness.render_frame().to_vec();

    // Change thickness significantly
    harness.set_thickness(200.0);
    let frame2 = harness.render_frame().to_vec();

    // Frames should be different (different interference colors)
    assert_ne!(
        frame1, frame2,
        "Different film thickness should produce different colors"
    );
}

#[tokio::test]
async fn test_scenario_execution() {
    let harness = TestHarness::new(256, 256).await;
    if harness.is_none() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    let mut harness = harness.unwrap();

    let scenario = vec![
        TestStep::SetThickness(500.0),
        TestStep::RenderFrame,
        TestStep::OrbitCamera(50.0, 0.0),
        TestStep::RenderFrame,
        TestStep::ZoomCamera(2.0),
        TestStep::RenderFrame,
        TestStep::SetThickness(300.0),
        TestStep::RenderFrame,
    ];

    let frames = harness.run_scenario(&scenario);

    assert_eq!(frames.len(), 4, "Should have captured 4 frames");

    // All frames should have content
    for (i, frame) in frames.iter().enumerate() {
        assert!(!frame.is_empty(), "Frame {} should not be empty", i);
        assert_eq!(
            frame.len(),
            256 * 256 * 4,
            "Frame {} should be correct size",
            i
        );
    }

    // Consecutive frames should be different (due to camera/thickness changes)
    assert_ne!(frames[0], frames[1], "Frame 0 and 1 should differ (orbit)");
    assert_ne!(frames[1], frames[2], "Frame 1 and 2 should differ (zoom)");
    assert_ne!(frames[2], frames[3], "Frame 2 and 3 should differ (thickness)");
}

#[tokio::test]
async fn test_multiple_renders_deterministic() {
    let harness = TestHarness::new(128, 128).await;
    if harness.is_none() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    let mut harness = harness.unwrap();

    // Set specific state
    harness.set_thickness(500.0);
    harness.set_time(0.0);
    harness.set_camera_distance(0.2);

    // Render twice with same state
    let frame1 = harness.render_frame().to_vec();
    let frame2 = harness.render_frame().to_vec();

    // Should be identical (deterministic rendering)
    assert_eq!(
        frame1, frame2,
        "Rendering same state twice should be deterministic"
    );
}

#[tokio::test]
async fn test_frame_diff_utility() {
    let a = vec![255, 0, 0, 255, 0, 255, 0, 255]; // Red, Green pixels
    let b = vec![255, 0, 0, 255, 0, 255, 0, 255]; // Same
    let c = vec![0, 0, 255, 255, 255, 255, 0, 255]; // Blue, Yellow pixels

    assert_eq!(frame_diff_ratio(&a, &b, 0), 0.0, "Identical frames");
    assert_eq!(frame_diff_ratio(&a, &c, 0), 1.0, "Completely different frames");
}

#[tokio::test]
async fn test_average_color_utility() {
    // All red pixels
    let red = vec![255, 0, 0, 255, 255, 0, 0, 255];
    let (r, g, b, a) = average_color(&red);
    assert!((r - 255.0).abs() < 0.001);
    assert!(g.abs() < 0.001);
    assert!(b.abs() < 0.001);
    assert!((a - 255.0).abs() < 0.001);
}
