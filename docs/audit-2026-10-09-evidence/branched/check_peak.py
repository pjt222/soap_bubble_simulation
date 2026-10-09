"""Estimate realistic peak per-texel deposit (float, before x64 encode) for default patch mode."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bf_model import *
from check_patch_scatter import gen_scatterers, make_force_fn
params = dict(DEFAULTS); hs = 0.158
S = gen_scatterers(400, 0.0, 0.5, 0.03, (0.5, 0.5, hs)); ff = make_force_fn(S)
tex = np.zeros((256, 512))
stride = 16
for ray in range(0, 8192, stride):
    steps, _ = trace_ray(ray, params, force_fn=ff)
    for s in steps:
        u, v = s['uv']
        if abs(u - 0.5) <= hs and abs(v - 0.5) <= hs:
            lu = np.clip((u - (0.5 - hs)) / (2 * hs), 0, 1); lv = np.clip((v - (0.5 - hs)) / (2 * hs), 0, 1)
            fx, fy = lu * 511, lv * 255; x0, y0 = int(fx), int(fy)
            x1, y1 = min(x0 + 1, 511), min(y0 + 1, 255); sx, sy = fx - x0, fy - y0
            d = s['deposit']
            tex[y0, x0] += d * (1 - sx) * (1 - sy); tex[y0, x1] += d * sx * (1 - sy)
            tex[y1, x0] += d * (1 - sx) * sy; tex[y1, x1] += d * sx * sy
tex *= stride  # scale to 8192 rays
peak = tex.max()
print(f"per-frame peak texel deposit (float) ~ {peak:.2f}; nonzero texels {np.mean(tex>0)*100:.1f}%")
for scale in [64, 1024, 4096, 65536]:
    ss = peak * scale / 0.15
    print(f"  scale {scale:6d}: steady-state peak u32 ~ {ss:.3e}  ({ss/2**32*100:.4f}% of u32 range)")
print("  (with 65536 rays: x8)")
