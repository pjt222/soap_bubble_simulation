#!/usr/bin/env bash
# Regenerate the golden images in tests/golden after an intentional rendering change.
#
#   scripts/regen-goldens.sh            # all goldens
#   scripts/regen-goldens.sh zoomed_in  # only goldens whose name contains "zoomed_in"
#
# Goldens are rendered headless on the lavapipe (CPU) Vulkan ICD so that they
# are reproducible on any machine with Mesa, independent of the GPU and of the
# Dozen driver crash (see scripts/test-local.sh). Previous images are moved to
# target/golden-previous/<UTC timestamp>/ for before/after review, never deleted.
# The screenshot tests re-create any missing golden, then the suite is re-run to
# confirm the new goldens pass.
set -euo pipefail
[[ -f Cargo.toml && -d tests/golden ]] || { echo "error: run from the project root" >&2; exit 1; }

name_filter="${1:-}"
backup_dir="target/golden-previous/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$backup_dir"

shopt -s nullglob
moved=0
for golden in tests/golden/*.png; do
    # *_diff.png / *_actual.png are comparison artifacts from failed runs: keep them with the backup.
    case "$golden" in *_diff.png | *_actual.png) mv -- "$golden" "$backup_dir/"; continue ;; esac
    if [[ -z "$name_filter" || "$golden" == *"$name_filter"* ]]; then
        mv -- "$golden" "$backup_dir/"
        moved=$((moved + 1))
    fi
done
echo "regen-goldens: moved $moved previous golden(s) to $backup_dir"

scripts/test-local.sh --test screenshot_tests -- --ignored --test-threads=1 >/dev/null 2>&1 || true
echo "regen-goldens: verifying regenerated goldens"
scripts/test-local.sh --test screenshot_tests -- --ignored --test-threads=1 2>&1 | sed -n '/^test result/p'
git status --short -- tests/golden
