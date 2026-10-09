#!/bin/bash
# Convenience script to run the soap bubble simulation under WSL
# Forces X11 backend since WSLg Wayland is unreliable from some terminals

WAYLAND_DISPLAY="" DISPLAY="${DISPLAY:-:0}" exec cargo run --release "$@"
