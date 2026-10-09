"""caustics_compute.wgsl replica on metre-scale thickness (drainage buffer units)."""
import sys
sys.path.append('/home/phtho/.local/lib/python3.12/site-packages')
import numpy as np
W, H = 256, 128
rng = np.random.default_rng(0)
th = (np.arange(H) / (H - 1))[:, None] * np.pi * np.ones((1, W))
for name, h in [("smooth drained 30->1000nm", 30e-9 + 970e-9 * (1 - np.cos(th)) / 2),
                ("plus 200nm white noise", 30e-9 + 970e-9 * (1 - np.cos(th)) / 2 + 200e-9 * rng.standard_normal((H, W)))]:
    S = lambda dx, dy: np.roll(np.roll(np.pad(h, ((1, 1), (0, 0)), mode='edge'), -dy, 0), -dx, 1)[1:-1]
    gx = (S(1, 0) - S(-1, 0)) / 2; gy = (S(0, 1) - S(0, -1)) / 2
    lap = S(-1, 0) + S(1, 0) + S(0, -1) + S(0, 1) - 4 * h
    d2x = S(-1, 0) + S(1, 0) - 2 * h; d2y = S(0, -1) + S(0, 1) - 2 * h
    dxy = (S(1, 1) - S(-1, 1) - S(1, -1) + S(-1, -1)) / 4
    hdet = d2x * d2y - dxy ** 2
    focal, thr, sharp, inten = 0.1, 0.001, 1.5, 2.0
    focusing = 1 / np.maximum(np.abs(1 - (-lap * focal)), 0.01)
    branch = np.where(hdet < -thr, 1.5, 1.0)
    refr = 1 + np.hypot(gx, gy) * 0.5
    I = np.clip((focusing * branch * refr) ** sharp * inten, 0, 10)
    print(f"{name}: caustic_map min {I.min():.9f} max {I.max():.9f} (expected const = caustic_intensity = {inten}); "
          f"max|lap| {np.abs(lap).max():.2e}, min hessian_det {hdet.min():.2e} vs -threshold {-thr}")
