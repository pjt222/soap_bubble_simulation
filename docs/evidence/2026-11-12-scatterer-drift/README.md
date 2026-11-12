# Scatterer drift evidence (#47)

## Replica of the old drift

`old_drift_replica.py` replicates the scatterer perturbation that `update_scatterers` in
`src/render/branched_flow.rs` applied before #47. Each frame it added
`(fract(sin(...) * 43758.547) - 0.5) * 0.001` UV per axis and clamped to the patch. It uses Rust's
sign-preserving `f32::fract` (here `x - trunc(x)` in numpy float32). The question: where are the
400 scatterers of the default patch (centre 0.75, 0.5, half size 0.158) after 15 s, the delay of
the PR #57 screenshots, at the 25 to 36 fps measured on Dozen?

Output (re-run 2026-11-12; needs numpy, so not `python3 -I`):

```
25 fps, 375 frames: share in lower half 0.00, lower third 0.00, right half 0.01; v range 0.342..0.475 (patch 0.342..0.658)
30 fps, 450 frames: share in lower half 0.00, lower third 0.00, right half 0.00; v range 0.342..0.453 (patch 0.342..0.658)
36 fps, 540 frames: share in lower half 0.00, lower third 0.00, right half 0.00; v range 0.342..0.394 (patch 0.342..0.658)
```

(The script's "right half" is u > 0.75, which is the left half on screen.)

The fraction's mean is 0, so every step has a mean of -0.0005 UV. The scatterers collect toward
the patch's (min u, min v) corner, which is the top right on screen at the default camera (u is
0.5 at +x, 0.75 at +z, facing the camera, so larger u is further left). None is left in the lower
half (v > 0.5). In the larger-u half (u > 0.75, the left half on screen) at most 2 of 400 remain
at 25 fps, and none at 30 or 36 fps; the review counted these exactly over several seeds and start
layouts.

Limits:
- numpy's float32 `sin` differs from Rust's in the last bits, which the factor 43758 amplifies.
  So only the statistics carry over, not per-scatterer values.
- The replica assumes scatterers start uniform over the patch and that film time advances every
  frame. Any advancing time gives the same mean step.

## Captures on Dozen

Two release builds, both run with `--compute --screenshot-after 15` and default settings on the
RTX 3070 Ti through Dozen (`WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER=1`), 2026-11-12:

| File | Build | Frame rate |
|---|---|---|
| `capture-main-2c3603d.png` | `main` at `2c3603d`, the old drift and generation | 13.2 to 17.2 fps |
| `capture-drift-a0ee3db.png` | the tree of `a0ee3db` on `fix/47-scatterer-drift`: integer hash and smooth drift, before the 1 s step cap and the fold after generation (neither matters at this frame rate) | 15.6 to 17.5 fps |

What they show (pixel counts on a crop of the patch, made during review):
- **Film colour.** In both, the whitish colour of the film shows between the filaments: the
  median film pixel is (172, 172, 177) in both. The filaments render darker than the film (median
  about 125), not white.
- **`main`:** the filaments are denser toward the right, with a fan at the top right, and reach
  across the lower centre and right. The lower left shows plain film.
- **The branch:** the filaments are spread over the whole patch, the lower left included.

The whitish colour is the film itself: it shows between the filaments everywhere in both captures,
and the filaments render darker than it. The "white lower part" of the PR #57 screenshots (a
temporary build at intensity 0.2) was probably the same film without filaments over it, rather
than clipped deposits; that is an inference, not a re-capture of that build.

About 230 frames ran before this run's screenshot (log: 66 + 84 + 78), so the old drift had moved
scatterers about 0.11 UV. At that point the replica leaves none in the patch's bottom third or its
larger-u third (left on screen); the plain lower left of the `main` capture is where those meet.
Elsewhere in the bottom third filaments still appear, presumably from rays scattered higher up.

Caveats:
- There is one capture of each build.
- The branch also changes how scatterers are generated (sign, strength and radius), so the
  comparison does not isolate the drift.
- Filaments rendering darker than the film is unexplained: the fragment shader adds them
  (`bubble.wgsl`). It belongs to #47's intensity mapping.
