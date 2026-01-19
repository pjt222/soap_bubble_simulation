//! Image export functionality

use std::path::Path;

/// Errors that can occur during export
#[derive(Debug)]
pub enum ExportError {
    /// Failed to create image buffer
    BufferCreation(String),
    /// Failed to save image file
    SaveError(String),
    /// Invalid dimensions
    InvalidDimensions { width: u32, height: u32 },
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExportError::BufferCreation(msg) => write!(f, "Failed to create image buffer: {}", msg),
            ExportError::SaveError(msg) => write!(f, "Failed to save image: {}", msg),
            ExportError::InvalidDimensions { width, height } => {
                write!(f, "Invalid dimensions: {}x{}", width, height)
            }
        }
    }
}

impl std::error::Error for ExportError {}

/// Export raw RGBA pixel data to a PNG file
///
/// # Arguments
/// * `path` - Output file path
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `data` - RGBA u8 pixel data (length must be width * height * 4)
///
/// # Returns
/// * `Ok(())` on success
/// * `Err(ExportError)` on failure
pub fn export_frame<P: AsRef<Path>>(
    path: P,
    width: u32,
    height: u32,
    data: &[u8],
) -> Result<(), ExportError> {
    // Validate dimensions
    if width == 0 || height == 0 {
        return Err(ExportError::InvalidDimensions { width, height });
    }

    let expected_len = (width * height * 4) as usize;
    if data.len() != expected_len {
        return Err(ExportError::BufferCreation(format!(
            "Data length {} doesn't match expected {} ({}x{}x4)",
            data.len(),
            expected_len,
            width,
            height
        )));
    }

    // Create image buffer from raw data
    let image_buffer: image::ImageBuffer<image::Rgba<u8>, _> =
        image::ImageBuffer::from_raw(width, height, data.to_vec()).ok_or_else(|| {
            ExportError::BufferCreation("Failed to create image buffer from raw data".to_string())
        })?;

    // Save as PNG
    image_buffer
        .save(path.as_ref())
        .map_err(|e| ExportError::SaveError(e.to_string()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_export_frame() {
        // Create a small test image (2x2 red pixels)
        let width = 2;
        let height = 2;
        let red_pixel = [255u8, 0, 0, 255];
        let data: Vec<u8> = red_pixel.iter().cycle().take(16).copied().collect();

        let path = "/tmp/test_export.png";
        let result = export_frame(path, width, height, &data);
        assert!(result.is_ok());

        // Clean up
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_invalid_dimensions() {
        let result = export_frame("/tmp/test.png", 0, 100, &[]);
        assert!(matches!(result, Err(ExportError::InvalidDimensions { .. })));
    }

    #[test]
    fn test_wrong_data_length() {
        let result = export_frame("/tmp/test.png", 10, 10, &[0u8; 100]);
        assert!(matches!(result, Err(ExportError::BufferCreation(_))));
    }
}
