# Ray seed evidence (#47 criterion 2)

This folder holds two measurements from the PR that made the ray start positions change
with film time: the frame-to-frame figures below, and the patch coverage re-measured with the
new seeds (last section). Both `.rs` files are test functions, not maintained tools.
`frame_to_frame_change.rs` is the first. To re-run it, paste it into the `tests` module of `src/render/branched_flow.rs`
at the commit that adds this folder, then run
`scripts/test-local.sh --lib -- --ignored --nocapture measure_frame_to_frame_change`. On
lavapipe it needed most of a 302 s run that also held three short tests.
`frame_to_frame_change.log` holds its output.

**Setup.**
- **Simulator.** The default `BranchedFlowSimulator`: patch view centred at u = 0.75, v = 0.5,
  8192 rays, 400 scatterers.
- **Film.** A uniform 500 nm film, so there is no GRIN force.
- **Frames.** Frames run in the app's order (update_params, update_scatterers, step), one
  submission per frame. The deposits accumulate under the 0.85 fade, as in the app.
- **Metric.** The run lasts 60 frames. For frames 40–59 it measures the change from the previous
  frame, Σ|T_k − T_(k−1)| / Σ T_k, in two ways:
  - over the raw texels;
  - over the displayed value min((texel / 64)², 3). That is the fragment shader's mapping
    (`compute_branched_flow` in `bubble.wgsl`) at the default intensity 1.0 and sharpness
    2.0, taken per texel, before bilinear sampling and compositing.
- **Cases.**
  - *Playing* advances film time by 1/60 s per frame, so the scatterers drift. *Paused* repeats
    film time 0.
  - *Frozen seeds* resets the seed clock every frame, so every frame uses seed 0. It stands in
    for the behaviour before #47, which seeded by the ray index alone, with a different hash.

| Case | Texels | Displayed |
|---|---|---|
| Paused, seeds follow film time (this change) | 0.0000 | 0.0000 |
| Playing at 60 fps, seeds follow film time (this change) | 0.0741 | 0.0983 |
| Playing at 60 fps, frozen seeds (stand-in for before) | 0.0494 | 0.0522 |
| Paused, frozen seeds | 0.0000 | 0.0000 |

The values are printed to four decimal places.

**What it shows.** While the film plays, the displayed branched-flow layer changes about 1.9×
as much from frame to frame as with frozen seeds: 0.0983 against 0.0522. The extra part is ray
sampling noise; the rest is scatterer drift. A paused film reads 0.0000 in both cases once the
fade has settled. Separately, `a_paused_film_traces_the_same_rays` checks that the deposits of
two paused frames are identical.

**Not measured.**
- Frame rate. On Dozen the compute view ran at 13–17.5 fps on 2026-11-12, so each frame there
  covers more film time and more drift.
- The full-sphere view.
- How much closer the accumulated image comes to a many-ray average now that the fade
  accumulates rays from different start positions.

## Patch coverage with the new seeds

`patch_size_coverage.rs` re-measures the one-frame lit share that `CLAUDE.md`, the
pre-loop comment in `branched_flow_compute.wgsl` and `patch_mode_rays_cover_the_patch`
quote. It uses the same setup as that test: patch centre u = 0.5, default scatterers, a
uniform film and one frame. Paste it into the same `tests` module, then run
`scripts/test-local.sh --lib -- --ignored --nocapture measure_patch_size_coverage`. Its
output is in `patch_size_coverage.log`. The earlier figures come from commit `01230b8`
(review of #57), measured with the seeds before #47.

| Half size | 0.158 | 0.18 | 0.20 | 0.25 | 0.30 |
|---|---|---|---|---|---|
| Seeds before #47 (`01230b8`) | 90% | 88% | 80% | 61% | 47% |
| Seeds after #47 | 89.3% | 86.0% | 77.9% | 58.9% | 45.7% |

Every row is 1–2 points lower. A plausible reason is that independent random start points
clump more than the old float hash's points did, so one frame leaves a few more texels
unlit. This is an inference; it was not tested. The "straight, unscattered rays of the same
length score 94.5%" figure from the same review was not re-measured.
