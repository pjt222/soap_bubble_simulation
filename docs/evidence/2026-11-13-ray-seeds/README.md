# Ray seed evidence (#47 criterion 2)

These are the measurements behind the PR that seeds the branched-flow ray start positions per
traced frame and freezes the picture while the film is paused. The `.rs` files are test
functions, not maintained tools.

**Running them.**
- Paste a file into the `tests` module of `src/render/branched_flow.rs` at the commit named
  in the table, then run
  `scripts/test-local.sh --lib -- --ignored --nocapture <test name>` (lavapipe).
- Most of them run at `f8d0998`, the branch before the trace schedule. There a paused film
  still traced and faded every frame, and the seed was picked by `RaySeedClock`, which the
  schedule replaced. They do not compile at later commits.

| File | Test | Commit | Output | Time on lavapipe |
|---|---|---|---|---|
| `frame_to_frame_change.rs` | `measure_frame_to_frame_change` | `f8d0998` | `frame_to_frame_change.log` | most of a 302 s run |
| `accumulation_error.rs` | `measure_accumulation_error` | `f8d0998` | `accumulation_error.log` | 247 s |
| `patch_size_coverage.rs` | `measure_patch_size_coverage` | `f8d0998`, and `main` at `4216e34` | `patch_size_coverage.log`, `patch_size_coverage_main.log` | a few seconds |
| `review_measurements.rs` | `review_measure_pause_transition`, `review_measure_coverage_seed_band` | `f8d0998` | `review_measurements.log` | 155 s for both |

**Common setup.**
- **Simulator.** The default `BranchedFlowSimulator`: 8192 rays, 400 scatterers, patch view
  centred at u = 0.75, v = 0.5 unless stated.
- **Film.** A uniform 500 nm film, so there is no GRIN force.
- **Frames.** Frames run in the app's order (update_params, update_scatterers, step), one
  submission per frame.
- **Displayed value.** min((texel / 64)², 3). That is the fragment shader's mapping
  (`compute_branched_flow` in `bubble.wgsl`) at the default intensity 1.0 and sharpness 2.0,
  taken per texel, before bilinear sampling and compositing.

## Frame-to-frame change while playing

The run lasts 60 frames under the 0.85 fade. For frames 40–59 it measures the change from the
previous frame, Σ|T_k − T_(k−1)| / Σ T_k.
- *Playing* advances film time by 1/60 s per frame, so the scatterers drift. *Paused* repeats
  film time 0 from the first frame.
- *Seed 0 held fixed* resets the seed clock every frame. It stands in for `main` before #47,
  which seeded by the ray index alone with a different hash.

| Case | Raw texels | Displayed |
|---|---|---|
| Playing at 60 fps, new seed per frame | 0.0741 | 0.0983 |
| Playing at 60 fps, seed 0 held fixed | 0.0494 | 0.0522 |
| Paused from the first frame, either | 0.0000 | 0.0000 |

Values are printed to four decimal places.

**What it shows.**
- While the film plays, the displayed layer changes about 1.9× as much from frame to frame as
  with seed 0 held fixed (1.5× over raw texels). The extra part is ray sampling noise; the rest
  is scatterer drift.
- The paused row is a film paused from the start, measured after it settled. The next sections
  show what happens when a playing film is paused.

**Not measured.**
- Frame rate. On Dozen the compute view ran at 13–17.5 fps on 2026-11-12, so each frame there
  covers more film time and more drift.
- The full-sphere view.

## Accuracy of the faded image

How close the faded deposit texture comes to the expected one.

**Setup.**
- **Scatterers.** They stay still (`update_scatterers` always gets film time 0), so the
  expected texture is fixed and only the ray sampling differs.
- **Reference.** The mean of 128 single frames, with the texture cleared before each. They use
  seeds 41–168, which neither faded run uses.
- **Faded runs.** Each runs 40 frames under the 0.85 fade, and frames 30–39 are measured.
  - *Seed advancing* passes film time = frame number to `update_params`, which advanced the
    seed every frame at `f8d0998`.
  - *Seed frozen* passes film time 0, so every frame uses seed 0. It stands in for `main`
    before #47.
- **Distance.** Σ|T − R·ΣT/ΣR| / ΣT over raw texels (T is the texture, R the reference). The
  same is computed over displayed values, relative to the displayed reference.

| Image | Raw texels | Displayed |
|---|---|---|
| Faded, seed advancing | 0.1344 | 0.1806 |
| Faded, seed frozen | 0.4147 | 0.4746 |
| For scale: a single frame, cleared (4 frames) | 0.4218 | – |

Only the raw value of the single frame compares with the faded rows. Its displayed value, 0.6385
in the log, is taken at about 1/6.7 of the faded brightness, and the display mapping squares
and clips at 3.

**What it shows.**
- With the seed advancing, the faded texture is 3.1× closer to the reference over raw texels,
  and 2.6× closer over displayed values.
- With the seed frozen, the fade adds no accuracy over a single frame (raw texels: 0.4147
  against 0.4218), because it repeats the same rays.
- A rough cross-check [derived, assuming the distance shrinks as 1/√N]. An exponential average
  with λ = 0.85 has the variance of (1 + λ) / (1 − λ) ≈ 12.3 independent frames, so the
  expected distance is about 0.42 / √12.3 ≈ 0.12. The reference's own noise, about
  0.42 / √128 ≈ 0.04, comes on top. The measured 0.134 lies in that range.

**Not measured.** While the film plays, the scatterers also drift within the fade window. This
measurement covers the sampling part only.

## Pausing a playing film, before the trace schedule

The `review_measure_pause_transition` run covers 40 playing frames, then 60 paused ones, under
the fade. It ran at `f8d0998`, where the paused frames kept fading and re-tracing seed 39.

**What happened.**
- **Change per frame.** The displayed layer changed by 0.0812 on the first paused frame,
  against 0.098 for a playing frame. It fell below 0.001 from the 18th paused frame, and no
  texel changed after the 36th.
- **Final picture.** The picture settled onto a single ray set: displayed distance 0.4183 from
  the picture at the pause instant, and 129,810 lit texels down to 118,476 (8.7% fewer).
  Single ray sets are the noisy case of the previous section.
- **`main` before #47** settles the same way, since its seeds never change; its settling time
  was not measured.

That is why a paused film now traces nothing:
- `TraceSchedule` skips the fade and trace passes while film time stands still, so the picture
  freezes at once, at the accumulated image.
- A change to a traced input re-traces `PAUSED_SETTLE_FRAMES` frames, then freezes again.
- `a_paused_film_freezes_the_picture` and
  `an_input_change_while_paused_retraces_until_the_picture_settles` check this on the GPU.
- The second review round found that the UI's per-frame laser-angle round trip never settled
  at azimuth ±180° or elevation ±90°, so the picture did not freeze there. The simulator now
  keeps the angles in degrees.
  `paused_picture_freezes_under_the_ui_round_trip_at_the_laser_end_stops` covers it.

## Patch coverage per patch size

This re-measures the one-frame lit share that `CLAUDE.md`, the pre-loop comment in
`branched_flow_compute.wgsl` and `patch_mode_rays_cover_the_patch` quote. It uses the setup of
that test: patch centre u = 0.5, default scatterers, a uniform film and one frame.
- `patch_size_coverage.rs` ran on this branch, where the first frame uses seed 0, and on `main`.
- `review_measure_coverage_seed_band` repeats the branch run for seeds 0–7.

| Half size | 0.158 | 0.18 | 0.20 | 0.25 | 0.30 |
|---|---|---|---|---|---|
| `main` at `4216e34` | 89.7% | 86.8% | 78.9% | 59.8% | 46.2% |
| This branch, seed 0 | 89.3% | 86.0% | 77.9% | 58.9% | 45.7% |
| This branch, seeds 0–7 | 89.1–89.8% | 86.0–86.8% | 77.9–79.0% | 58.9–59.7% | 45.7–46.5% |

**How `main` compares with the branch.**
- `main` lies inside this branch's seed range at three sizes, and up to 0.15 points above it
  at the other two.
- It is 0.2–0.4 points above the eight-seed mean at every size.
- The five sizes share one ray set, so this is a single comparison: a shift of a few tenths of
  a point is neither shown nor ruled out.

**The older figures.** The figures recorded at `01230b8` (90%, 88%, 80%, 61%, 47%) lie 0.3–1.2
points above `main` itself. They predate #63's scatterer changes; that this is the cause is
inferred, not measured. The figure for straight rays of the same length (94.5%) from the same
review was not re-measured.
