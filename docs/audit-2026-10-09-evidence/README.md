# Audit 2026-10-09: evidence scripts

Numeric replicas and checks that the audit's finding agents wrote and ran on 2026-10-09:
Python ports of the WGSL kernels, f32/stability probes, CIE colour checks, and Rust timing
replicas. They are **as-run snapshots**, not maintained tools; the maintained tools live in
`scripts/` (see `scripts/README.md`).

## Mapping from the findings JSON

`docs/audit-2026-10-09-findings.json` cites these files under the session scratchpad, e.g.
`/tmp/claude-1000/<session>/scratchpad/optics/airy_check.py`. That prefix corresponds to this
directory: `scratchpad/<dimension>/<file>` is `docs/audit-2026-10-09-evidence/<dimension>/<file>`.

| Directory | Audit dimension |
|---|---|
| `optics/` | thin-film interference optics, colour pipeline |
| `drainage/` | drainage PDE, stability, f32 resolution |
| `branched/` | branched-flow ray optics, caustics, deposits |
| `geomfoam/` | geometry, patch mesh, foam dynamics |
| `cpuperf/` | CPU-side per-frame cost (Rust replicas) |
| `gpuperf/` | GPU kernel replicas (divergence, step factor, deposits) |
| `sota/` | state-of-the-art comparison checks |

## Re-running

- The scripts still contain absolute scratchpad paths; adjust them before running.
- Many need `numpy`.
- Third-party inputs were not committed: the CIE 1931 2° colour-matching functions and the D65
  illuminant (from http://www.cvrl.org), papers (Patsyk 2020/2022, Wyman 2013), and the wgpu
  changelog. Their URLs are listed in the `sources` field of the findings JSON.
- Compiled binaries and caches were not committed either, nor were the upstream library
  sources the research agent fetched for reading (wgpu-hal / wgpu-types v30, CubeCL); the
  findings cite them by URL.
- `sys.path.append(...)` lines that pointed at the auditor's per-user site-packages directory
  were removed; otherwise the scripts are unchanged.
