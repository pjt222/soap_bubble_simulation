#!/usr/bin/env bash
# Probe which GPU adapters wgpu can use on this machine (issue #38).
#
#   scripts/gpu/probe-adapters.sh                        # vulkaninfo summary + per-ICD headless test
#   scripts/gpu/probe-adapters.sh --app 30               # additionally run the release app for 30 s
#   scripts/gpu/probe-adapters.sh --app 30 --compute     # ... with GPU compute effects on (app args follow)
#   WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER=1 scripts/gpu/probe-adapters.sh --app 30
#                                                        # ... on the Mesa Dozen GPU instead of llvmpipe
#
# Step 1 lists the Vulkan devices the loader sees (Dozen exposes D3D12 GPUs on WSL).
# Step 2 runs the (normally ignored) headless pipeline test once per Vulkan ICD
# (Dozen only, lavapipe only, no ICD), so a driver crash is attributed to one ICD;
# exit code 139 is SIGSEGV. "pass" requires the test to have actually run.
# Step 3 (--app) launches target/release/soap-bubble-sim under X11 and prints the
# adapter the app logs ("GPU adapter: ...") and its frame-rate reports (every 5 s).
# The release binary is built only if missing: rebuild after code changes.
# Logs go to target/probe-adapters/ (gitignored via /target).
set -uo pipefail
[[ -f Cargo.toml && -d src/render ]] || { echo "error: run from the project root" >&2; exit 1; }

app_seconds=0
app_args=()
if [[ "${1:-}" == "--app" ]]; then
    app_seconds="${2:-20}"
    shift 2 2>/dev/null || shift $#
    app_args=("$@")
fi

log_dir=target/probe-adapters
mkdir -p "$log_dir"
icd_dir=/usr/share/vulkan/icd.d
test_name=render::headless::tests::test_headless_pipeline_creation

echo "--- 1. Vulkan devices (vulkaninfo --summary)"
if command -v vulkaninfo >/dev/null; then
    vulkaninfo --summary 2>/dev/null | sed -n 's/^[[:space:]]*\(deviceName\|deviceType\|driverName\|driverInfo\|apiVersion\|conformanceVersion\)[[:space:]]*=/  \1 =/p'
else
    echo "  vulkaninfo not installed (apt install vulkan-tools)"
fi

echo "--- 2. Headless test per Vulkan ICD"
test_binary=$(cargo test --lib --no-run --message-format=json 2>/dev/null | python3 -c '
import json, sys
for line in sys.stdin:
    message = json.loads(line)
    if message.get("reason") == "compiler-artifact" and message.get("executable") and message["target"]["kind"] == ["lib"]:
        print(message["executable"])
')
if [[ -z "$test_binary" ]]; then
    echo "  could not locate the lib test binary (cargo test --lib --no-run failed?)"
    exit 1
fi
echo "  binary: $test_binary"
for icd in dzn lvp none; do
    if [[ "$icd" == none ]]; then icd_file=/nonexistent.json; else icd_file="$icd_dir/${icd}_icd.json"; fi
    if [[ "$icd" != none && ! -f "$icd_file" ]]; then echo "  $icd: ICD not installed, skipped"; continue; fi
    log_file="$log_dir/headless-$icd.log"
    VK_DRIVER_FILES="$icd_file" "$test_binary" "$test_name" --exact --ignored --test-threads=1 --nocapture >"$log_file" 2>&1
    exit_code=$?
    if [[ "$exit_code" == 139 ]]; then
        verdict="SIGSEGV"
    elif [[ "$exit_code" == 0 ]] && grep -q "1 passed" "$log_file"; then
        verdict="pass"
    elif [[ "$exit_code" == 0 ]]; then
        verdict="test did not run"
    else
        verdict="fail"
    fi
    echo "  $icd: exit $exit_code ($verdict)  log: $log_file"
done

if (( app_seconds > 0 )); then
    echo "--- 3. Release app for ${app_seconds}s (X11)${app_args[*]:+, args: ${app_args[*]}}"
    if [[ ! -x target/release/soap-bubble-sim ]]; then
        echo "  building release binary (fat LTO, several minutes)"
        cargo build --release >/dev/null 2>&1 || { echo "  release build failed"; exit 1; }
    fi
    log_file="$log_dir/app.log"
    WAYLAND_DISPLAY="" DISPLAY="${DISPLAY:-:0}" RUST_LOG="${RUST_LOG:-info}" \
        timeout "$app_seconds" target/release/soap-bubble-sim "${app_args[@]}" >"$log_file" 2>&1
    exit_code=$?
    echo "  exit $exit_code (124 = stopped by timeout, i.e. ran fine)  log: $log_file"
    sed -n '/GPU adapter:\|hiding adapter\|not Vulkan compliant\|Compute effects enabled\|panicked\|Validation Error/p' "$log_file" | head -8 | sed 's/^/  /'
    echo "  frame rate (one line per >= 5 s of frames):"
    if grep -q 'FPS ' "$log_file"; then
        sed -n 's/.*\(FPS [0-9.]* (.*\)$/    \1/p' "$log_file"
    else
        echo "    no FPS lines: run longer than 5 s, or rebuild (cargo build --release) if the binary predates the FPS log"
    fi
fi
