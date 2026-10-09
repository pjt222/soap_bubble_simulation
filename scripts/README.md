# Project toolbox

Every script runs from the project root and refuses to run anywhere else. Output that is not
meant to be committed goes under `target/` (gitignored). The Python tools use only the standard
library.

## Testing

| Tool | What it does |
|---|---|
| `scripts/test-local.sh [cargo test args]` | Runs `cargo test` with the lavapipe Vulkan ICD forced. On WSL with Mesa ≥ 26 the Dozen ICD segfaults wgpu 24's debug-build Vulkan path (`render::headless::tests::test_headless_pipeline_creation`), and that kills the whole lib test binary (#48). Example: `scripts/test-local.sh -- --include-ignored`. |
| `scripts/regen-goldens.sh [name-filter]` | Re-renders `tests/golden/*.png` headless on lavapipe after an intentional rendering change. Previous images and any `*_diff.png` / `*_actual.png` comparison artifacts move to `target/golden-previous/<UTC timestamp>/` for before/after review. The script then re-runs the screenshot suite. |

## GPU

| Tool | What it does |
|---|---|
| `scripts/gpu/probe-adapters.sh [--app SECONDS]` | Lists the Vulkan devices the loader sees (`vulkaninfo --summary`), then runs the headless pipeline test once per ICD (Dozen only, lavapipe only, none) so a driver crash is attributed to one ICD (exit 139 = SIGSEGV). With `--app`, it also runs the release app under X11 and prints the wgpu adapter log lines, because the app does not log which adapter it chose. Logs go to `target/probe-adapters/`. See #38. |

## Physics checks

| Tool | What it does |
|---|---|
| `scripts/physics/thin_film_reference.py [--wavelength NM] [--cos-theta C] [--n N] [--check]` | Prints the exact complex-amplitude reflectance of a free-standing film next to the Airy expression with and without the extra π phase, per polarisation. `--check` exits 1 unless Airy with the geometric phase matches the exact slab formula. Use it to sanity-check any change to the interference code (#42). |

## Audits and issues

| Tool | What it does |
|---|---|
| `scripts/audit/findings.py table FINDINGS` | Prints a Markdown table of all findings in an audit findings JSON (e.g. `docs/audit-2026-10-09-findings.json`), ordered by severity after verification. |
| `scripts/audit/findings.py check FINDINGS GROUPS` | Validates an issue-grouping spec (e.g. `docs/audit-2026-10-09-issues.json`). Every finding must be in exactly one group, issue groups need titles and acceptance criteria, and `{#key}` placeholders may only reference earlier groups. |
| `scripts/audit/findings.py draft FINDINGS GROUPS --out DIR` | Renders one Markdown body per issue or comment into `DIR` for review. |
| `scripts/audit/findings.py file FINDINGS GROUPS --out DIR [--yes]` | Dry run by default. With `--yes` it creates the issues and posts the comments through `gh`, in spec order, resolving `{#key}` to the filed issue numbers and recording URLs in `DIR/filed.json`, so a re-run skips what is already filed. |

## Other

| Tool | What it does |
|---|---|
| `scripts/generate_workflow.R` | Regenerates the putior Mermaid workflow diagram from `// put` annotations (see `CLAUDE.md`). |
| `run.sh` (project root) | Runs the app under WSL with the X11 backend forced. |
