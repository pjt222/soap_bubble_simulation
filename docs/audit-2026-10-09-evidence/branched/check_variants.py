"""Row-by-row: committed vs working-tree defaults, patch vs full-sphere.
Reports step_factor saturation, ray travel distance, patch coverage."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bf_model import *
from check_patch_scatter import gen_scatterers, make_force_fn

rows = [
    ("working-tree (8192 rays, 200 steps, 400 scat), patch", 200, 400, 1),
    ("committed   (32768 rays, 400 steps, 800 scat), patch", 400, 800, 1),
    ("working-tree, full sphere", 200, 400, 0),
    ("committed, full sphere", 400, 800, 0),
]
hs = 0.158
for name, nsteps, nscat, patch in rows:
    params = dict(DEFAULTS); params['ray_steps'] = nsteps; params['patch_enabled'] = patch
    S = gen_scatterers(nscat, 0.0, 0.5, 0.03, (0.5, 0.5, hs) if patch else None)
    ff = make_force_fn(S)
    sfs, travel_chart, travel_arc = [], [], []
    cover = np.zeros((64, 64), bool)
    for ray in range(0, 8192, 64):
        steps, _ = trace_ray(ray, params, force_fn=ff)
        sfs += [s['sf'] for s in steps]
        P = np.array([s['p3'] for s in steps])
        travel_arc.append(np.sum(np.arccos(np.clip(np.sum(P[1:] * P[:-1], 1), -1, 1))))
        travel_chart.append(sum(s['sf'] for s in steps) * params['step_size'])
        if patch:
            for s in steps:
                if abs(s['uv'][0] - 0.5) <= hs and abs(s['uv'][1] - 0.5) <= hs:
                    lu = (s['uv'][0] - (0.5 - hs)) / (2 * hs); lv = (s['uv'][1] - (0.5 - hs)) / (2 * hs)
                    cover[min(int(lv * 64), 63), min(int(lu * 64), 63)] = True
    sfs = np.array(sfs)
    msg = (f"{name}: step_factor==0.3 in {np.mean(sfs < 0.3001)*100:.1f}% of steps, mean sf {sfs.mean():.3f}; "
           f"mean travel chart {np.mean(travel_chart):.3f}, arc {np.degrees(np.mean(travel_arc)):.1f} deg")
    if patch:
        msg += f"; patch bins touched (128 rays) {cover.mean()*100:.1f}%"
    print(msg)
