//! Multi-bubble foam renderer using GPU instancing.
//!
//! This module provides efficient rendering of multiple bubbles using
//! hardware instancing to minimize draw calls.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::physics::foam::{Bubble, BubbleCluster};

/// Maximum number of bubbles that can be rendered.
pub const MAX_BUBBLES: u32 = 64;

/// Per-instance data sent to the GPU for each bubble.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BubbleInstance {
    /// Model matrix row 0
    pub model_0: [f32; 4],
    /// Model matrix row 1
    pub model_1: [f32; 4],
    /// Model matrix row 2
    pub model_2: [f32; 4],
    /// Model matrix row 3
    pub model_3: [f32; 4],
    /// Bubble radius (for thickness calculations)
    pub radius: f32,
    /// Aspect ratio (1.0 = sphere, <1.0 = oblate)
    pub aspect_ratio: f32,
    /// Base film thickness in nanometers
    pub thickness_nm: f32,
    /// Refractive index
    pub refractive_index: f32,
}

impl BubbleInstance {
    /// Create instance data from a Bubble.
    pub fn from_bubble(bubble: &Bubble) -> Self {
        // Create model matrix: translation + scale
        let scale = Mat4::from_scale(Vec3::new(
            bubble.radius,
            bubble.radius * bubble.aspect_ratio,
            bubble.radius,
        ));
        let translation = Mat4::from_translation(bubble.position);
        let model = translation * scale;

        // WGSL mat4x4 constructor takes COLUMNS, so we pass columns not rows
        Self {
            model_0: model.col(0).to_array(),
            model_1: model.col(1).to_array(),
            model_2: model.col(2).to_array(),
            model_3: model.col(3).to_array(),
            radius: bubble.radius,
            aspect_ratio: bubble.aspect_ratio,
            thickness_nm: bubble.thickness_nm,
            refractive_index: bubble.refractive_index,
        }
    }

    /// Returns the vertex buffer layout for instance data.
    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<BubbleInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // Model matrix row 0
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Model matrix row 1
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Model matrix row 2
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Model matrix row 3
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Radius
                wgpu::VertexAttribute {
                    offset: 64,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32,
                },
                // Aspect ratio
                wgpu::VertexAttribute {
                    offset: 68,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32,
                },
                // Thickness
                wgpu::VertexAttribute {
                    offset: 72,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32,
                },
                // Refractive index
                wgpu::VertexAttribute {
                    offset: 76,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

/// Renderer for multi-bubble foam using GPU instancing.
pub struct FoamRenderer {
    /// Instance data buffer
    instance_buffer: wgpu::Buffer,
    /// Current instance data (CPU-side)
    instances: Vec<BubbleInstance>,
    /// Maximum supported instances
    max_instances: u32,
    /// Whether the instance buffer needs updating
    dirty: bool,
}

impl FoamRenderer {
    /// Create a new foam renderer.
    pub fn new(device: &wgpu::Device, max_instances: u32) -> Self {
        let max_instances = max_instances.min(MAX_BUBBLES);

        // Create instance buffer with max capacity
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Foam Instance Buffer"),
            size: (max_instances as usize * std::mem::size_of::<BubbleInstance>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            instance_buffer,
            instances: Vec::with_capacity(max_instances as usize),
            max_instances,
            dirty: true,
        }
    }

    /// Update instance data from a bubble cluster.
    pub fn update_from_cluster(&mut self, cluster: &BubbleCluster) {
        self.instances.clear();

        for bubble in cluster.bubbles().iter().take(self.max_instances as usize) {
            self.instances.push(BubbleInstance::from_bubble(bubble));
        }

        self.dirty = true;
    }

    /// Upload instance data to GPU if dirty.
    pub fn upload(&mut self, queue: &wgpu::Queue) {
        if self.dirty && !self.instances.is_empty() {
            queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&self.instances),
            );
            self.dirty = false;
        }
    }

    /// Get the instance buffer for rendering.
    pub fn instance_buffer(&self) -> &wgpu::Buffer {
        &self.instance_buffer
    }

    /// Get the number of active instances.
    pub fn instance_count(&self) -> u32 {
        self.instances.len() as u32
    }

    /// Check if there are any instances to render.
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Add a single bubble instance.
    pub fn add_instance(&mut self, bubble: &Bubble) {
        if self.instances.len() < self.max_instances as usize {
            self.instances.push(BubbleInstance::from_bubble(bubble));
            self.dirty = true;
        }
    }

    /// Clear all instances.
    pub fn clear(&mut self) {
        self.instances.clear();
        self.dirty = true;
    }

    /// Get instance data for debugging.
    pub fn instances(&self) -> &[BubbleInstance] {
        &self.instances
    }
}

/// Renderer for shared walls (Plateau borders) between bubbles.
///
/// This generates curved interface geometry where bubbles touch.
pub struct SharedWallRenderer {
    /// Vertex buffer for wall geometry
    vertex_buffer: Option<wgpu::Buffer>,
    /// Index buffer for wall geometry
    index_buffer: Option<wgpu::Buffer>,
    /// Number of indices
    num_indices: u32,
    /// Whether geometry needs regeneration
    dirty: bool,
}

impl SharedWallRenderer {
    /// Create a new shared wall renderer.
    pub fn new() -> Self {
        Self {
            vertex_buffer: None,
            index_buffer: None,
            num_indices: 0,
            dirty: true,
        }
    }

    /// Generate wall geometry for bubble connections.
    ///
    /// This creates curved surfaces between touching bubbles based on
    /// Young-Laplace curvature calculations.
    pub fn generate_walls(
        &mut self,
        _device: &wgpu::Device,
        _cluster: &BubbleCluster,
    ) {
        // TODO: Implement shared wall geometry generation
        // This is complex and will be implemented in Phase 4
        // For now, we render bubbles without explicit shared walls

        self.dirty = false;
    }

    /// Check if there are any walls to render.
    pub fn has_walls(&self) -> bool {
        self.num_indices > 0
    }

    /// Get the vertex buffer.
    pub fn vertex_buffer(&self) -> Option<&wgpu::Buffer> {
        self.vertex_buffer.as_ref()
    }

    /// Get the index buffer.
    pub fn index_buffer(&self) -> Option<&wgpu::Buffer> {
        self.index_buffer.as_ref()
    }

    /// Get the number of indices.
    pub fn num_indices(&self) -> u32 {
        self.num_indices
    }
}

impl Default for SharedWallRenderer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn test_bubble_instance_size() {
        // Ensure instance struct is properly sized for GPU
        assert_eq!(std::mem::size_of::<BubbleInstance>(), 80); // 16*4 + 4*4 = 80 bytes
    }

    #[test]
    fn test_instance_from_bubble() {
        let bubble = Bubble::new(0, Vec3::new(1.0, 2.0, 3.0), 0.5);
        let instance = BubbleInstance::from_bubble(&bubble);

        assert!((instance.radius - 0.5).abs() < 1e-6);
        assert!((instance.aspect_ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_model_matrix_translation() {
        let bubble = Bubble::new(0, Vec3::new(1.0, 2.0, 3.0), 1.0);
        let instance = BubbleInstance::from_bubble(&bubble);

        // WGSL expects columns, so we pass columns:
        // col(0) = [sx, 0, 0, 0]   → model_0
        // col(1) = [0, sy, 0, 0]   → model_1
        // col(2) = [0, 0, sz, 0]   → model_2
        // col(3) = [tx, ty, tz, 1] → model_3
        // Translation is in model_3 (the 4th column)
        assert!((instance.model_3[0] - 1.0).abs() < 1e-6, "tx should be 1.0");
        assert!((instance.model_3[1] - 2.0).abs() < 1e-6, "ty should be 2.0");
        assert!((instance.model_3[2] - 3.0).abs() < 1e-6, "tz should be 3.0");
        assert!((instance.model_3[3] - 1.0).abs() < 1e-6, "w should be 1.0");
    }
}
