import sys
import numpy as np

rng = np.random.default_rng(1)
N = 200000
sx = rng.random(N); sy = rng.random(N)
W = np.stack([(1 - sx) * (1 - sy), sx * (1 - sy), (1 - sx) * sy, sx * sy], 1)

print("--- deposit_bilinear truncation (wgsl:388-398): u32(base*w) per pixel ---")
for intensity, sf in [(1.0, 0.3), (0.82, 0.3), (1.0, 1.0), (1.0, 3.0), (0.5, 0.3), (0.2, 3.0)]:
    base = intensity * 0.15 * sf * 64.0
    kept = np.floor(base * W).sum(1)
    print(f"intensity={intensity:4.2f} step_factor={sf:3.1f}: base={base:6.3f}  mean kept/ideal = {kept.mean()/base:.3f}; "
          f"kept=0 for {np.mean(kept==0)*100:5.1f}% of sub-pixel positions")

# position dependence at sf=0.3 (the 99% case): pixel-centre vs pixel-corner
base = 1.0 * 0.15 * 0.3 * 64
for (a, b) in [(0.0, 0.0), (0.25, 0.25), (0.5, 0.5)]:
    w = np.array([(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b])
    print(f"  deposit at sub-pixel ({a},{b}): kept {np.floor(base*w).sum():.0f} of {base:.2f}")

print("--- clear pass fade truncation (wgsl:567-569): x <- u32(0.85 x) + d ---")
for d in [1, 2, 3, 5, 10, 28]:
    x = 0
    for _ in range(500):
        x = int(np.float32(x) * np.float32(0.85)) + d
    print(f"  per-frame deposit {d:2d}: steady state {x:4d}  vs ideal d/0.15 = {d/0.15:7.1f}  ratio {x/(d/0.15):.2f}")

print("--- u32 overflow headroom ---")
max_per_step = 1.0 * 0.15 * 3.0 * 64.0
for rays, steps in [(8192, 200), (32768, 400), (65536, 200), (65536, 400)]:
    worst = rays * steps * max_per_step / 0.15  # every ray-step into one pixel, steady state 1/(1-0.85)
    print(f"  {rays} rays x {steps} steps: worst-case pixel {worst:.3e} vs u32 max {2**32-1:.3e} -> {'OK' if worst < 2**32 else 'OVERFLOW possible'}")
