//! Frame capture and PNG export for screenshots and recording.
//!
//! Extracted from `pipeline.rs` to separate export logic from the render pipeline.

use std::path::Path;

use crate::export::image_export;

/// Handles screenshot and recording frame capture with pre-allocated staging buffers
/// and background PNG encoding.
pub(crate) struct FrameExporter {
    recording: bool,
    frame_counter: u32,
    screenshot_requested: bool,
    /// Pre-allocated staging buffer for GPU readback (avoids per-frame allocation).
    staging_buffer: Option<(wgpu::Buffer, u32)>, // (buffer, padded_bytes_per_row)
    /// Background PNG encoding thread handle (waited on before next capture).
    png_thread: Option<std::thread::JoinHandle<()>>,
}

impl FrameExporter {
    pub fn new() -> Self {
        Self {
            recording: false,
            frame_counter: 0,
            screenshot_requested: false,
            staging_buffer: None,
            png_thread: None,
        }
    }

    pub fn is_recording(&self) -> bool {
        self.recording
    }

    pub fn frame_counter(&self) -> u32 {
        self.frame_counter
    }

    pub fn request_screenshot(&mut self) {
        self.screenshot_requested = true;
    }

    pub fn screenshot_requested(&self) -> bool {
        self.screenshot_requested
    }

    /// Returns true if a frame capture is needed (screenshot or recording).
    pub fn should_capture(&self) -> bool {
        self.screenshot_requested || self.recording
    }

    /// Toggle recording mode on/off.
    pub fn toggle_recording(&mut self) {
        self.recording = !self.recording;
        if self.recording {
            self.frame_counter = 0;
            if let Err(e) = std::fs::create_dir_all("screenshots") {
                log::error!("Cannot create screenshots directory: {}", e);
            }
            log::info!("Recording started");
        } else {
            log::info!("Recording stopped after {} frames", self.frame_counter);
        }
    }

    /// Set recording state from UI. Handles the start/stop transitions.
    pub fn set_recording(&mut self, recording: bool) {
        if recording != self.recording {
            if recording {
                self.frame_counter = 0;
                if let Err(e) = std::fs::create_dir_all("screenshots") {
                    log::error!("Cannot create screenshots directory: {}", e);
                }
                log::info!("Recording started");
            } else {
                log::info!("Recording stopped after {} frames", self.frame_counter);
            }
            self.recording = recording;
        }
    }

    pub fn set_screenshot_requested(&mut self, requested: bool) {
        self.screenshot_requested = requested;
    }

    /// Invalidate the staging buffer (called on resize).
    pub fn invalidate_staging_buffer(&mut self) {
        self.staging_buffer = None;
    }

    /// Ensure the staging buffer exists and copy the frame texture into it.
    /// Must be called before `encoder.finish()` / `queue.submit()`.
    pub fn prepare_capture(
        &mut self,
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        encoder: &mut wgpu::CommandEncoder,
        output_texture: &wgpu::Texture,
    ) {
        let bytes_per_pixel = 4u32;
        let unpadded_bytes_per_row = config.width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let buffer_size = padded_bytes_per_row as u64 * config.height as u64;

        // Re-use existing staging buffer if size matches
        if self
            .staging_buffer
            .as_ref()
            .is_none_or(|(buf, _)| buf.size() != buffer_size)
        {
            self.staging_buffer = Some((
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Screenshot Staging Buffer"),
                    size: buffer_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                padded_bytes_per_row,
            ));
        }

        let (staging, _) = self.staging_buffer.as_ref().unwrap();

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(config.height),
                },
            },
            wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Read back the staging buffer, convert BGRA to RGBA, and export as PNG.
    /// Must be called after `queue.submit()`.
    pub fn process_capture(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) {
        // Wait for any previous PNG encoding thread to finish before reading the buffer
        if let Some(handle) = self.png_thread.take() {
            let _ = handle.join();
        }

        let (ref staging, padded_bytes_per_row) = *self.staging_buffer.as_ref().unwrap();
        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let data = buffer_slice.get_mapped_range();
            let bytes_per_pixel = 4u32;

            // Remove row padding and convert BGRA to RGBA
            let mut pixels = Vec::with_capacity((width * height * 4) as usize);
            for row in 0..height {
                let start = (row * padded_bytes_per_row) as usize;
                let end = start + (width * bytes_per_pixel) as usize;
                pixels.extend_from_slice(&data[start..end]);
            }

            drop(data);
            staging.unmap();

            for chunk in pixels.chunks_exact_mut(4) {
                chunk.swap(0, 2);
            }

            // Determine filename
            let path = if self.screenshot_requested && !self.recording {
                format!("screenshots/screenshot_{:04}.png", self.frame_counter)
            } else {
                format!("screenshots/frame_{:04}.png", self.frame_counter)
            };

            self.frame_counter = self.frame_counter.saturating_add(1);

            // put id:'io_export_frame', label:'Export frame to PNG', input:'final_frame_gpu.internal', output:'screenshots/*.png'
            // Offload PNG encoding to a background thread to avoid blocking the render loop
            self.png_thread = Some(std::thread::spawn(move || {
                if let Err(e) = std::fs::create_dir_all("screenshots") {
                    log::error!("Cannot create screenshots directory: {}", e);
                    return;
                }
                if let Err(e) = image_export::export_frame(&path, width, height, &pixels) {
                    log::error!("Failed to export frame: {}", e);
                } else {
                    log::info!("Saved: {}", path);
                }
            }));
        }

        self.screenshot_requested = false;
    }

    /// Capture a single frame to a PNG file (standalone, not using the pre-allocated buffer).
    pub fn capture_frame(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface: &wgpu::Surface<'_>,
        config: &wgpu::SurfaceConfiguration,
        path: impl AsRef<Path>,
    ) -> Result<(), String> {
        let width = config.width;
        let height = config.height;

        let bytes_per_pixel = 4u32;
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let buffer_size = padded_bytes_per_row as u64 * height as u64;

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Screenshot Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let output = surface
            .get_current_texture()
            .map_err(|e| format!("Failed to get surface texture: {}", e))?;

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Screenshot Encoder"),
            });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &output.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| format!("Channel disconnected: {e}"))?
            .map_err(|e| format!("Failed to map buffer: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);

        for row in 0..height {
            let start = (row * padded_bytes_per_row) as usize;
            let end = start + (width * bytes_per_pixel) as usize;
            pixels.extend_from_slice(&data[start..end]);
        }

        drop(data);
        staging_buffer.unmap();

        for chunk in pixels.chunks_exact_mut(4) {
            chunk.swap(0, 2);
        }

        image_export::export_frame(path, width, height, &pixels)
            .map_err(|e| format!("Failed to export frame: {}", e))?;

        Ok(())
    }
}
