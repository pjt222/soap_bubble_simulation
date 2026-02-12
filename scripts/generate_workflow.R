#!/usr/bin/env Rscript
# Generate putior workflow diagram for the soap bubble simulation
# Usage: Rscript scripts/generate_workflow.R
#
# Requires: putior >= 0.2.0.9000 (install_github("pjt222/putior"))

library(putior)

# --- 1. Parse annotations from all source files (Rust + WGSL) ---
workflow <- put("D:/dev/p/soap_bubble_simulation/src", recursive = TRUE)
cat("Parsed", nrow(workflow), "annotations from", length(unique(workflow$file_path)), "files\n")

# --- 2. Generate compact diagram for README (no artifacts) ---
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
