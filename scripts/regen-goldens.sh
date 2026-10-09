#!/usr/bin/env bash
# Regenerate the golden images in tests/golden after an intentional rendering change.
#
#   scripts/regen-goldens.sh            # all goldens
#   scripts/regen-goldens.sh zoomed_in  # only goldens whose name contains "zoomed_in"
#
# Goldens are rendered headless on the lavapipe (CPU) Vulkan ICD, never on a
# hardware adapter: the script refuses to run without lavapipe. Lavapipe output
# depends on the Mesa/LLVM version, so the driver string is recorded in
# <backup>/renderer.txt and printed; compare it with the version that produced
# the committed goldens before trusting a diff.
#
# Order of operations, so a failure never leaves tests/golden half-deleted:
#   1. build the screenshot tests (goldens untouched if this fails),
#   2. move previous goldens and *_diff.png / *_actual.png artifacts to
#      target/golden-previous/<UTC timestamp>/,
#   3. render (the screenshot tests re-create missing goldens),
#   4. re-run the suite against the new goldens,
#   5. on any failure in 3-4, restore the moved goldens.
# Full cargo output goes to <backup>/regen.log.
set -uo pipefail
[[ -f Cargo.toml && -d tests/golden ]] || { echo "error: run from the project root" >&2; exit 1; }

lavapipe_icd=/usr/share/vulkan/icd.d/lvp_icd.json
if [[ ! -f "$lavapipe_icd" ]]; then
    echo "error: lavapipe ICD not found at $lavapipe_icd (install mesa-vulkan-drivers);" \
         "refusing to render goldens on another adapter" >&2
    exit 1
fi
export VK_DRIVER_FILES="$lavapipe_icd"

name_filter="${1:-}"
backup_dir="target/golden-previous/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$backup_dir"
log_file="$backup_dir/regen.log"

driver_info=$(vulkaninfo --summary 2>/dev/null | sed -n 's/^[[:space:]]*driverInfo[[:space:]]*=[[:space:]]*//p' | head -1)
echo "${driver_info:-unknown (vulkaninfo not installed)}" > "$backup_dir/renderer.txt"
echo "regen-goldens: renderer: $(cat "$backup_dir/renderer.txt")"

echo "regen-goldens: building screenshot tests"
if ! cargo test --test screenshot_tests --no-run >>"$log_file" 2>&1; then
    echo "error: build failed, goldens untouched; see $log_file" >&2
    exit 1
fi

shopt -s nullglob
moved_goldens=()
for golden in tests/golden/*.png; do
    # *_diff.png / *_actual.png are comparison artifacts from failed runs: keep them with the backup.
    case "$golden" in *_diff.png | *_actual.png) mv -- "$golden" "$backup_dir/"; continue ;; esac
    if [[ -z "$name_filter" || "$golden" == *"$name_filter"* ]]; then
        mv -- "$golden" "$backup_dir/"
        moved_goldens+=("$(basename -- "$golden")")
    fi
done
echo "regen-goldens: moved ${#moved_goldens[@]} previous golden(s) to $backup_dir"

restore_previous_goldens() {
    for name in "${moved_goldens[@]}"; do
        cp -- "$backup_dir/$name" "tests/golden/$name"
    done
    echo "regen-goldens: restored ${#moved_goldens[@]} previous golden(s); see $log_file" >&2
}

echo "regen-goldens: rendering"
if ! cargo test --test screenshot_tests -- --ignored --test-threads=1 >>"$log_file" 2>&1; then
    echo "error: rendering run failed" >&2
    restore_previous_goldens
    exit 1
fi

echo "regen-goldens: verifying regenerated goldens"
if ! cargo test --test screenshot_tests -- --ignored --test-threads=1 >>"$log_file" 2>&1; then
    echo "error: regenerated goldens do not pass their own comparison" >&2
    restore_previous_goldens
    exit 1
fi
sed -n '/^test result/p' "$log_file" | tail -1
git status --short -- tests/golden
