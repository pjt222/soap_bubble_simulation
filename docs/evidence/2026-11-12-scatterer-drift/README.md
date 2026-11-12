# Scatterer drift evidence (#47)

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

The fraction's mean is 0, so every step has a mean of -0.0005 UV, and all scatterers end up in the
(min u, min v) corner. The patch's lower and right halves are left without scatterers. That may
explain the white lower part of the PR #57 screenshots, but this is a replica, not a render. The
replica assumes scatterers start uniform over the patch and that film time advances every frame;
any advancing time gives the same mean.
