import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bf_model import *

# ---------- 1. zero-force: are rays great circles? ----------
params = dict(DEFAULTS); params['patch_enabled'] = 0; params['ray_steps'] = 2000
worst_plane_dev = 0.0
for ray in [0, 1, 17, 4095, 8191]:
    steps, (E, t1, t2) = trace_ray(ray, params)
    P = np.array([s['p3'] for s in steps])
    # plane through origin spanned by first two points
    nrm = normalize(np.cross(P[0], P[5]))
    dev = np.max(np.abs(P @ nrm))
    worst_plane_dev = max(worst_plane_dev, dev)
print(f"[geodesic] max |p.n_plane| over 5 zero-force rays: {worst_plane_dev:.2e} (0 => exact great circle)")

# ---------- 2. arc length per step vs chart radius ----------
steps, (E, t1, t2) = trace_ray(0, params)
P = np.array([s['p3'] for s in steps]); R = np.array([np.linalg.norm(s['pos']) for s in steps])
arc = np.arccos(np.clip(np.sum(P[1:] * P[:-1], axis=1), -1, 1))
chart = np.array([np.linalg.norm(steps[i+1]['pos'] - steps[i]['pos']) for i in range(len(steps)-1)])
for target in [0.0, 0.5, 1.0, 1.5, 2.0, 2.45]:
    i = np.argmin(np.abs(R[:-1] - target))
    a = np.arctan(R[i])
    print(f"[arc] r={R[i]:.3f} (angle {np.degrees(a):5.1f} deg): arc/chart step = {arc[i]/chart[i]:.4f}  "
          f"cos^2(alpha)={np.cos(a)**2:.4f}  deposit-per-arc rel. = {chart[i]/arc[i]:.3f}")
print(f"[range] max angular distance from entry reachable (r=2.5): {np.degrees(np.arctan(2.5)):.1f} deg")

# ---------- 3. default patch mode: where do rays spawn / deposit? ----------
params = dict(DEFAULTS)
E = np.array([0, 0, 1.0]); print("[patch] entry uv:", normal_to_uv(E))
P = uv_to_normal(0.5, 0.5); print("[patch] patch centre 3D:", np.round(P, 3),
                                  " angle from entry: %.1f deg" % np.degrees(np.arccos(P @ E)))
t1, t2 = frame(E)
off = np.array([(P - E) @ t1, (P - E) @ t2])
spawn3 = normalize(E + t1 * off[0] + t2 * off[1])
print("[patch] spawn centre chart offset", off, "-> 3D", np.round(spawn3, 3), "uv", np.round(normal_to_uv(spawn3), 4),
      " angle from entry %.1f deg" % np.degrees(np.arccos(spawn3 @ E)))

hs = 0.158
def in_patch(uv):
    return abs(uv[0] - 0.5) <= hs and abs(uv[1] - 0.5) <= hs
nr = 1024
in_cnt = tot = 0
spawn_in = 0
cover = np.zeros((64, 64), bool)
for ray in range(0, 8192, 8192 // nr):
    steps, _ = trace_ray(ray, params)
    tot += len(steps)
    if in_patch(steps[0]['uv']):
        spawn_in += 1
    for s in steps:
        if in_patch(s['uv']):
            in_cnt += 1
            lu = (s['uv'][0] - (0.5 - hs)) / (2 * hs); lv = (s['uv'][1] - (0.5 - hs)) / (2 * hs)
            cover[min(int(lv * 64), 63), min(int(lu * 64), 63)] = True
print(f"[patch] rays spawning inside patch: {spawn_in}/{nr}")
print(f"[patch] fraction of steps depositing inside patch: {in_cnt/tot:.3f}  (steps traced {tot})")
print(f"[patch] fraction of patch area (64x64 bins) receiving any zero-force deposit: {cover.mean():.3f}")
cols = cover.any(axis=0); rows = cover.any(axis=1)
print(f"[patch] covered local-u bins: {cols.sum()}/64 (first {np.argmax(cols)}, last {63-np.argmax(cols[::-1])}); "
      f"local-v bins: {rows.sum()}/64")

# correct gnomonic coordinate would be (P.t1, P.t2)/(P.E) -> infinite for 90deg
print("[patch] P.E =", P @ E, "-> gnomonic coordinate of patch centre undefined (>=90 deg away)")
