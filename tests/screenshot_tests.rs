//! Screenshot comparison tests for visual regression testing
//!
//! Uses golden image comparison to detect visual changes in rendering.

use image::{ImageBuffer, RgbaImage};
use std::fs;
use std::path::{Path, PathBuf};

use soap_bubble_sim::render::HeadlessRenderPipeline;

/// Golden image directory (relative to crate root)
const GOLDEN_DIR: &str = "tests/golden";

/// Errors that can occur during screenshot comparison
#[derive(Debug)]
pub enum ComparisonError {
    /// Golden image was created (first run)
    NewGoldenCreated(PathBuf),
    /// Images differ by more than tolerance
    ImagesDiffer {
        name: String,
        diff_ratio: f64,
        tolerance: f64,
    },
    /// Failed to load golden image
    LoadError(String),
    /// Failed to save image
    SaveError(String),
    /// Dimension mismatch
    DimensionMismatch {
        expected: (u32, u32),
        actual: (u32, u32),
    },
}

impl std::fmt::Display for ComparisonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComparisonError::NewGoldenCreated(path) => {
                write!(f, "Created new golden image: {}", path.display())
            }
            ComparisonError::ImagesDiffer {
                name,
                diff_ratio,
                tolerance,
            } => {
                write!(
                    f,
                    "Image '{}' differs by {:.2}% (tolerance: {:.2}%)",
                    name,
                    diff_ratio * 100.0,
                    tolerance * 100.0
                )
            }
            ComparisonError::LoadError(msg) => write!(f, "Failed to load image: {}", msg),
            ComparisonError::SaveError(msg) => write!(f, "Failed to save image: {}", msg),
            ComparisonError::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "Dimension mismatch: expected {}x{}, got {}x{}",
                    expected.0, expected.1, actual.0, actual.1
                )
            }
        }
    }
}

impl std::error::Error for ComparisonError {}

/// Screenshot comparator for golden image testing
pub struct ScreenshotComparator {
    golden_dir: PathBuf,
    /// Tolerance as fraction of pixels that can differ (0.0 - 1.0)
    tolerance: f64,
    /// Per-channel color tolerance (allows for GPU precision differences)
    color_tolerance: u8,
}

impl ScreenshotComparator {
    /// Create a new comparator with specified golden directory and tolerance
    pub fn new<P: AsRef<Path>>(golden_dir: P, tolerance: f64) -> Self {
        Self {
            golden_dir: golden_dir.as_ref().to_path_buf(),
            tolerance,
            color_tolerance: 5, // Allow small color differences
        }
    }

    /// Create golden directory if it doesn't exist
    fn ensure_golden_dir(&self) -> Result<(), ComparisonError> {
        fs::create_dir_all(&self.golden_dir)
            .map_err(|e| ComparisonError::SaveError(format!("Failed to create directory: {}", e)))
    }

    /// Get the path to a golden image
    fn golden_path(&self, name: &str) -> PathBuf {
        self.golden_dir.join(format!("{}.png", name))
    }

    /// Get the path to a diff image (for debugging)
    fn diff_path(&self, name: &str) -> PathBuf {
        self.golden_dir.join(format!("{}_diff.png", name))
    }

    /// Save image from raw RGBA data
    fn save_image(
        &self,
        path: &Path,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(), ComparisonError> {
        let buffer: RgbaImage = ImageBuffer::from_raw(width, height, data.to_vec())
            .ok_or_else(|| ComparisonError::SaveError("Failed to create image buffer".into()))?;

        buffer
            .save(path)
            .map_err(|e| ComparisonError::SaveError(format!("Failed to save: {}", e)))
    }

    /// Compute the difference ratio between two images
    fn compute_diff(&self, expected: &[u8], actual: &[u8]) -> f64 {
        if expected.len() != actual.len() || expected.is_empty() {
            return 1.0;
        }

        let pixel_count = expected.len() / 4;
        let mut diff_count = 0;

        for i in (0..expected.len()).step_by(4) {
            let dr = (expected[i] as i32 - actual[i] as i32).abs();
            let dg = (expected[i + 1] as i32 - actual[i + 1] as i32).abs();
            let db = (expected[i + 2] as i32 - actual[i + 2] as i32).abs();

            if dr > self.color_tolerance as i32
                || dg > self.color_tolerance as i32
                || db > self.color_tolerance as i32
            {
                diff_count += 1;
            }
        }

        diff_count as f64 / pixel_count as f64
    }

    /// Generate a visual diff image highlighting differences
    fn generate_diff_image(
        &self,
        expected: &[u8],
        actual: &[u8],
        _width: u32,
        _height: u32,
    ) -> Vec<u8> {
        let mut diff = vec![0u8; expected.len()];

        for i in (0..expected.len()).step_by(4) {
            let dr = (expected[i] as i32 - actual[i] as i32).abs();
            let dg = (expected[i + 1] as i32 - actual[i + 1] as i32).abs();
            let db = (expected[i + 2] as i32 - actual[i + 2] as i32).abs();

            if dr > self.color_tolerance as i32
                || dg > self.color_tolerance as i32
                || db > self.color_tolerance as i32
            {
                // Highlight differences in red
                diff[i] = 255;
                diff[i + 1] = 0;
                diff[i + 2] = 0;
                diff[i + 3] = 255;
            } else {
                // Dim the matching pixels
                diff[i] = expected[i] / 3;
                diff[i + 1] = expected[i + 1] / 3;
                diff[i + 2] = expected[i + 2] / 3;
                diff[i + 3] = 255;
            }
        }

        diff
    }

    /// Compare actual render against golden image
    ///
    /// If no golden image exists, creates one and returns NewGoldenCreated.
    /// If images differ beyond tolerance, returns ImagesDiffer and saves a diff image.
    pub fn compare(
        &self,
        name: &str,
        actual: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(), ComparisonError> {
        self.ensure_golden_dir()?;

        let golden_path = self.golden_path(name);

        if !golden_path.exists() {
            // First run: create golden image
            self.save_image(&golden_path, actual, width, height)?;
            return Err(ComparisonError::NewGoldenCreated(golden_path));
        }

        // Load golden image
        let golden_image = image::open(&golden_path)
            .map_err(|e| ComparisonError::LoadError(format!("Failed to load golden: {}", e)))?;

        let golden_rgba = golden_image.to_rgba8();

        // Check dimensions
        if golden_rgba.width() != width || golden_rgba.height() != height {
            return Err(ComparisonError::DimensionMismatch {
                expected: (golden_rgba.width(), golden_rgba.height()),
                actual: (width, height),
            });
        }

        let golden_data = golden_rgba.as_raw();

        // Compute difference
        let diff_ratio = self.compute_diff(golden_data, actual);

        if diff_ratio > self.tolerance {
            // Save diff image for debugging
            let diff_image = self.generate_diff_image(golden_data, actual, width, height);
            let diff_path = self.diff_path(name);
            let _ = self.save_image(&diff_path, &diff_image, width, height);

            // Also save the actual image for comparison
            let actual_path = self.golden_dir.join(format!("{}_actual.png", name));
            let _ = self.save_image(&actual_path, actual, width, height);

            return Err(ComparisonError::ImagesDiffer {
                name: name.to_string(),
                diff_ratio,
                tolerance: self.tolerance,
            });
        }

        Ok(())
    }

    /// Update the golden image (for intentional visual changes)
    pub fn update_golden(
        &self,
        name: &str,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(), ComparisonError> {
        self.ensure_golden_dir()?;
        let golden_path = self.golden_path(name);
        self.save_image(&golden_path, data, width, height)
    }
}

/// Test fixture that provides a headless pipeline and comparator
pub struct ScreenshotTestFixture {
    pub pipeline: HeadlessRenderPipeline,
    pub comparator: ScreenshotComparator,
}

impl ScreenshotTestFixture {
    /// Create a new test fixture
    pub async fn new(width: u32, height: u32, tolerance: f64) -> Option<Self> {
        let pipeline = HeadlessRenderPipeline::new(width, height).await?;
        let comparator = ScreenshotComparator::new(GOLDEN_DIR, tolerance);
        Some(Self {
            pipeline,
            comparator,
        })
    }

    /// Render and compare against golden image
    pub fn render_and_compare(&mut self, name: &str) -> Result<(), ComparisonError> {
        let (width, height) = self.pipeline.size();
        let frame = self.pipeline.render_to_buffer();
        self.comparator.compare(name, &frame, width, height)
    }
}

// ============================================================================
// Screenshot Comparison Tests
// ============================================================================

#[tokio::test]
async fn test_default_bubble_appearance() {
    let fixture = ScreenshotTestFixture::new(256, 256, 0.05).await;
    if fixture.is_none() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    let mut fixture = fixture.unwrap();

    // Set deterministic state
    fixture.pipeline.set_thickness(500.0);
    fixture.pipeline.set_time(0.0);
    fixture.pipeline.set_camera_distance(0.2);

    match fixture.render_and_compare("default_bubble") {
        Ok(()) => {} // Match!
        Err(ComparisonError::NewGoldenCreated(path)) => {
            println!("Created golden image: {}", path.display());
        }
        Err(e) => panic!("Screenshot comparison failed: {}", e),
    }
}

#[tokio::test]
async fn test_thin_film_200nm() {
    let fixture = ScreenshotTestFixture::new(256, 256, 0.05).await;
    if fixture.is_none() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    let mut fixture = fixture.unwrap();

    fixture.pipeline.set_thickness(200.0);
    fixture.pipeline.set_time(0.0);
    fixture.pipeline.set_camera_distance(0.2);

    match fixture.render_and_compare("thickness_200nm") {
        Ok(()) => {}
        Err(ComparisonError::NewGoldenCreated(path)) => {
            println!("Created golden image: {}", path.display());
        }
        Err(e) => panic!("Screenshot comparison failed: {}", e),
    }
}

#[tokio::test]
async fn test_thin_film_400nm() {
    let fixture = ScreenshotTestFixture::new(256, 256, 0.05).await;
    if fixture.is_none() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    let mut fixture = fixture.unwrap();

    fixture.pipeline.set_thickness(400.0);
    fixture.pipeline.set_time(0.0);
    fixture.pipeline.set_camera_distance(0.2);

    match fixture.render_and_compare("thickness_400nm") {
        Ok(()) => {}
        Err(ComparisonError::NewGoldenCreated(path)) => {
            println!("Created golden image: {}", path.display());
        }
        Err(e) => panic!("Screenshot comparison failed: {}", e),
    }
}

#[tokio::test]
async fn test_thin_film_600nm() {
    let fixture = ScreenshotTestFixture::new(256, 256, 0.05).await;
    if fixture.is_none() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    let mut fixture = fixture.unwrap();

    fixture.pipeline.set_thickness(600.0);
    fixture.pipeline.set_time(0.0);
    fixture.pipeline.set_camera_distance(0.2);

    match fixture.render_and_compare("thickness_600nm") {
        Ok(()) => {}
        Err(ComparisonError::NewGoldenCreated(path)) => {
            println!("Created golden image: {}", path.display());
        }
        Err(e) => panic!("Screenshot comparison failed: {}", e),
    }
}

#[tokio::test]
async fn test_thin_film_800nm() {
    let fixture = ScreenshotTestFixture::new(256, 256, 0.05).await;
    if fixture.is_none() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    let mut fixture = fixture.unwrap();

    fixture.pipeline.set_thickness(800.0);
    fixture.pipeline.set_time(0.0);
    fixture.pipeline.set_camera_distance(0.2);

    match fixture.render_and_compare("thickness_800nm") {
        Ok(()) => {}
        Err(ComparisonError::NewGoldenCreated(path)) => {
            println!("Created golden image: {}", path.display());
        }
        Err(e) => panic!("Screenshot comparison failed: {}", e),
    }
}

#[tokio::test]
async fn test_zoomed_in_view() {
    let fixture = ScreenshotTestFixture::new(256, 256, 0.05).await;
    if fixture.is_none() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    let mut fixture = fixture.unwrap();

    fixture.pipeline.set_thickness(500.0);
    fixture.pipeline.set_time(0.0);
    fixture.pipeline.set_camera_distance(0.1); // Closer

    match fixture.render_and_compare("zoomed_in") {
        Ok(()) => {}
        Err(ComparisonError::NewGoldenCreated(path)) => {
            println!("Created golden image: {}", path.display());
        }
        Err(e) => panic!("Screenshot comparison failed: {}", e),
    }
}

#[tokio::test]
async fn test_zoomed_out_view() {
    let fixture = ScreenshotTestFixture::new(256, 256, 0.05).await;
    if fixture.is_none() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    let mut fixture = fixture.unwrap();

    fixture.pipeline.set_thickness(500.0);
    fixture.pipeline.set_time(0.0);
    fixture.pipeline.set_camera_distance(0.5); // Farther

    match fixture.render_and_compare("zoomed_out") {
        Ok(()) => {}
        Err(ComparisonError::NewGoldenCreated(path)) => {
            println!("Created golden image: {}", path.display());
        }
        Err(e) => panic!("Screenshot comparison failed: {}", e),
    }
}

#[tokio::test]
async fn test_rotated_view() {
    let fixture = ScreenshotTestFixture::new(256, 256, 0.05).await;
    if fixture.is_none() {
        eprintln!("Skipping test: no GPU available");
        return;
    }

    let mut fixture = fixture.unwrap();

    fixture.pipeline.set_thickness(500.0);
    fixture.pipeline.set_time(0.0);
    fixture.pipeline.set_camera_distance(0.2);
    fixture.pipeline.orbit_camera(100.0, 50.0); // Rotate camera

    match fixture.render_and_compare("rotated_view") {
        Ok(()) => {}
        Err(ComparisonError::NewGoldenCreated(path)) => {
            println!("Created golden image: {}", path.display());
        }
        Err(e) => panic!("Screenshot comparison failed: {}", e),
    }
}

// ============================================================================
// Utility Tests
// ============================================================================

#[test]
fn test_comparator_creation() {
    let comparator = ScreenshotComparator::new("tests/golden", 0.01);
    assert!((comparator.tolerance - 0.01).abs() < 0.0001);
}

#[test]
fn test_diff_computation() {
    let comparator = ScreenshotComparator::new("tests/golden", 0.01);

    // Identical images
    let a = vec![255, 0, 0, 255, 0, 255, 0, 255];
    let b = vec![255, 0, 0, 255, 0, 255, 0, 255];
    assert_eq!(comparator.compute_diff(&a, &b), 0.0);

    // Completely different
    let c = vec![0, 0, 255, 255, 255, 255, 0, 255];
    assert_eq!(comparator.compute_diff(&a, &c), 1.0);

    // Half different
    let d = vec![255, 0, 0, 255, 255, 255, 0, 255]; // First pixel same, second different
    assert!((comparator.compute_diff(&a, &d) - 0.5).abs() < 0.01);
}

#[test]
fn test_diff_image_generation() {
    let comparator = ScreenshotComparator::new("tests/golden", 0.01);

    let a = vec![255, 0, 0, 255, 0, 255, 0, 255]; // Red, Green
    let b = vec![255, 0, 0, 255, 255, 255, 0, 255]; // Red, Yellow (different)

    let diff = comparator.generate_diff_image(&a, &b, 2, 1);

    // First pixel matches (dimmed)
    assert!(diff[0] < 128); // Dimmed red
    assert_eq!(diff[3], 255); // Full alpha

    // Second pixel differs (red highlight)
    assert_eq!(diff[4], 255); // Red
    assert_eq!(diff[5], 0); // No green
    assert_eq!(diff[6], 0); // No blue
}
