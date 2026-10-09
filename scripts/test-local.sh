#!/usr/bin/env bash
# Run `cargo test` with the lavapipe (CPU) Vulkan ICD forced.
#
# On WSL with Mesa >= 26 the Dozen (Vulkan-on-D3D12) ICD is installed, and
# wgpu 24's debug-build Vulkan path segfaults in
# render::headless::tests::test_headless_pipeline_creation whenever Dozen is
# visible, which aborts the whole lib test binary. Restricting the loader to
# lavapipe makes local runs deterministic. All arguments go to `cargo test`,
# e.g.  scripts/test-local.sh -- --include-ignored
#
# Diagnose the driver situation with scripts/gpu/probe-adapters.sh.
set -euo pipefail
[[ -f Cargo.toml && -d src/render ]] || { echo "error: run from the project root" >&2; exit 1; }

lavapipe_icd=/usr/share/vulkan/icd.d/lvp_icd.json
if [[ -f "$lavapipe_icd" ]]; then
    export VK_DRIVER_FILES="$lavapipe_icd"
    echo "test-local: VK_DRIVER_FILES=$lavapipe_icd" >&2
else
    echo "test-local: warning: $lavapipe_icd not found, using system ICDs" >&2
fi
exec cargo test "$@"
