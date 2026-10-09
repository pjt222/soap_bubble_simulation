# Project toolbox

The toolbox scripts below run from the project root and refuse to run anywhere else. The older
`scripts/generate_workflow.R` (hard-coded Windows paths) and the root `run.sh` predate this
convention. Output that is not meant to be committed goes under `target/` (gitignored). The
Python tools use only the standard library.

## Testing

| Tool | What it does |
|---|---|
| `scripts/test-local.sh [cargo test args]` | Runs `cargo test` with the lavapipe Vulkan ICD forced. On WSL with Mesa ≥ 26 the Dozen ICD segfaults wgpu 24's debug-build Vulkan path (`render::headless::tests::test_headless_pipeline_creation`), and that kills the whole lib test binary (#48). Example: `scripts/test-local.sh -- --include-ignored`. |
| `scripts/regen-goldens.sh [name-filter]` | Re-renders `tests/golden/*.png` headless after an intentional rendering change. It requires lavapipe and refuses other adapters. It records the Mesa/LLVM driver string, because lavapipe output depends on it. It builds before touching anything, moves previous images and `*_diff.png` / `*_actual.png` artifacts to `target/golden-previous/<UTC timestamp>/`, re-runs the suite against the new goldens, and restores the previous ones on any failure. Cargo output goes to `regen.log` in that folder. |

## GPU

| Tool | What it does |
|---|---|
| `scripts/gpu/probe-adapters.sh [--app SECONDS]` | Lists the Vulkan devices the loader sees (`vulkaninfo --summary`), then runs the headless pipeline test once per ICD (Dozen only, lavapipe only, none) so a driver crash is attributed to one ICD (exit 139 = SIGSEGV). With `--app`, it also runs the release app under X11 and prints the wgpu adapter log lines, because the app does not log which adapter it chose. Logs go to `target/probe-adapters/`. See #38. |

## Physics checks

| Tool | What it does |
|---|---|
| `scripts/physics/thin_film_reference.py [--wavelength NM] [--cos-theta C] [--n N] [--check]` | Documents the math: prints the exact complex-amplitude reflectance of a free-standing film next to Airy per polarisation, the convention the renderer ships (s/p averaged *before* Airy, exact only at normal incidence, #51), and the inverted `+π` form (#42). `--check` is a math sanity check (Airy vs exact). It does not run the project's code, so it cannot detect a regression there. The Rust tests in `src/physics/interference.rs`, `src/render/interference_lut.rs` and `tests/wgsl_validation.rs` guard the code. |

## Audits and issues

| Tool | What it does |
|---|---|
| `scripts/audit/findings.py table FINDINGS [--groups GROUPS --filed FILED]` | Prints a Markdown table of all findings in an audit findings JSON (e.g. `docs/audit-2026-10-09-findings.json`), ordered by severity after verification. With `--groups`/`--filed`, the last column shows where each finding was filed (`#42`, `#35 (comment)`) instead of the auditor's guess. |
| `scripts/audit/findings.py filed GROUPS FILED` | Prints the filed issues and comments with the finding ids each one covers. |
| `scripts/audit/findings.py check FINDINGS GROUPS` | Validates an issue-grouping spec (e.g. `docs/audit-2026-10-09-issues.json`). Every finding must be in exactly one group, issue groups need titles and acceptance criteria, and `{#key}` placeholders (in title, summary, acceptance, related, lead findings) may only reference earlier *issue* groups. |
| `scripts/audit/findings.py draft FINDINGS GROUPS --out DIR` | Renders one Markdown body per issue or comment into `DIR` for review. |
| `scripts/audit/findings.py file FINDINGS GROUPS --out DIR [--yes]` | Dry run by default. With `--yes` it creates the issues and posts the comments through `gh`, in spec order, resolving `{#key}` to the filed issue numbers and recording each URL in `DIR/filed.json` as soon as `gh` returns it. It stops at the first failure, so later groups never get unresolved references, and a re-run skips what is already filed. |

## Other

| Tool | What it does |
|---|---|
| `scripts/generate_workflow.R` | Regenerates the putior Mermaid workflow diagram from `// put` annotations (see `CLAUDE.md`). |
| `run.sh` (project root) | Runs the app under WSL with the X11 backend forced. |
