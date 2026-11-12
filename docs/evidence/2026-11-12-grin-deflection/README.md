# GRIN deflection evidence (PR #60, #45)

Numeric replicas behind statements in PR #60 and in the code it added. These are snapshots, not
maintained tools; the maintained tools live in `scripts/` (see `scripts/README.md`). They need
`numpy`, so run them with `python3` but not in isolated mode (`-I`). Each runs in under 20 s.

| Script | Backs | Output (re-run 2026-11-12) |
|---|---|---|
| `deposit_replica.py` | The narrow-beam comment in `grin_deflection_frames` (`src/render/branched_flow.rs`). At the default beam spread 0.4, the phi test's deposit-weighted centroid reads about 5% above its chart prediction even with the exact analytic force; at 0.1 it agrees within about 1%. | spread 0.4: phi +0.00885 (u32 truncation) / +0.00881 (without), theta +0.01112 / +0.01115. Spread 0.1: phi +0.00855 / +0.00854, theta +0.01128 / +0.01127. Chart predictions: 0.00845 (phi), 0.01120 (theta). The GPU reads +0.00886 (phi, 0.4) and +0.00856 / +0.01129 (0.1). |
| `east_beam_replica.py` | That the wide-beam excess is not a displacement of the rays: their mean crossing of row 191 is close to the central ray. | Central ray +0.00852, beam mean +0.00847, closed form 0.00845. |
| `neff_sensitivity.py` | The mode and wavelength sensitivity in `src/physics/film_waveguide.rs`, the 2000 nm reference value in its tests, and the geodesic shift quoted in the theta test. | n_eff(2000 nm) = 1.32446. d ln n_eff / dd at 500 nm: TE0 0.1245, TM0 0.1781 per µm (1.43×); 450 nm 0.1013, 650 nm 0.1553 per µm. Geodesic shift at column 447: 0.00929. |

The replicas model the shader's propagation loop (kick, renormalise, drift in the gnomonic chart,
adaptive step factor) and, in `deposit_replica.py`, `deposit_bilinear`, with the GRIN force given
analytically rather than sampled from a thickness grid. Rays are spread uniformly across the beam;
the shader uses a hash of the ray index.
