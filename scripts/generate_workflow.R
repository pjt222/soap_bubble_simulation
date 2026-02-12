#!/usr/bin/env Rscript
# Generate putior workflow diagram for the soap bubble simulation
# Usage: Rscript scripts/generate_workflow.R
#
# Requires: putior >= 0.2.0 (install_github("pjt222/putior"))

library(putior)

# --- 1. Parse annotations from Rust source files ---
rust_workflow <- put("D:/dev/p/soap_bubble_simulation/src", recursive = TRUE)
cat("Parsed", nrow(rust_workflow), "annotations from Rust files\n")

# --- 2. Manually add WGSL shader annotations (unsupported extension) ---
# Note: putior does not yet support .wgsl files (GitHub issue #TBD)
wgsl_nodes <- data.frame(
  file_name = c(
    "bubble.wgsl", "bubble.wgsl",
    "branched_flow_compute.wgsl",
    "drainage.wgsl",
    "bubble_instanced.wgsl",
    "caustics.wgsl",
    "caustics_compute.wgsl",
    "wall.wgsl"
  ),
  file_path = paste0(
    "D:/dev/p/soap_bubble_simulation/src/render/shaders/",
    c("bubble.wgsl", "bubble.wgsl",
      "branched_flow_compute.wgsl",
      "drainage.wgsl",
      "bubble_instanced.wgsl",
      "caustics.wgsl",
      "caustics_compute.wgsl",
      "wall.wgsl")
  ),
  file_type = "wgsl",
  id = c(
    "gpu_render_bubble_vs", "gpu_render_bubble_fs",
    "gpu_compute_branched_shader",
    "gpu_compute_drainage_shader",
    "gpu_render_instanced",
    "gpu_render_caustics",
    "gpu_compute_caustics_shader",
    "gpu_render_wall"
  ),
  input = c(
    "vertex_buffer_gpu.internal", "lut_texture_gpu.internal",
    "uniform_buffers_gpu.internal",
    "uniform_buffers_gpu.internal",
    "vertex_buffer_gpu.internal",
    "compute_results_gpu.internal",
    "compute_results_gpu.internal",
    "vertex_buffer_gpu.internal"
  ),
  label = c(
    "Bubble vertex shader", "Bubble fragment shader",
    "Branched flow ray trace",
    "Drainage PDE solver",
    "Instanced foam shader",
    "Caustic render shader",
    "Caustic intensity compute",
    "Plateau border shader"
  ),
  output = c(
    "framebuffer_gpu.internal", "framebuffer_gpu.internal",
    "branched_flow_texture_gpu.internal",
    "compute_results_gpu.internal",
    "framebuffer_gpu.internal",
    "framebuffer_gpu.internal",
    "compute_results_gpu.internal",
    "framebuffer_gpu.internal"
  ),
  stringsAsFactors = FALSE
)

# --- 3. Combine into unified workflow ---
workflow <- rbind(rust_workflow, wgsl_nodes)
class(workflow) <- c("putior_workflow", "data.frame")
cat("Total workflow nodes:", nrow(workflow), "\n")

# --- 4. Generate compact diagram for README (no artifacts) ---
cat("\nGenerating README diagram...\n")
put_diagram(
  workflow,
  output = "file",
  file = "D:/dev/p/soap_bubble_simulation/workflow_diagram.md",
  title = "Soap Bubble Simulation Data Flow",
  direction = "TD",
  theme = "plasma",
  show_files = FALSE,
  show_artifacts = FALSE,
  style_nodes = TRUE,
  show_workflow_boundaries = TRUE
)
cat("Diagram written to workflow_diagram.md\n")

# To generate a detailed version with artifact nodes, set show_artifacts = TRUE:
# put_diagram(workflow, output = "file", file = "workflow_diagram_detailed.md",
#   title = "Detailed", direction = "TD", theme = "plasma", show_artifacts = TRUE)
