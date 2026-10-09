"""Is the GRIN term alive when thickness_field is in metres (pipeline.rs:709 init 500e-9)?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bf_model import *

v = (np.arange(TH) / (TH - 1))[:, None] * np.ones((1, TW))
theta = v * np.pi
# strongly drained film: 30 nm at top -> 1000 nm at bottom (metres, as stored by drainage.wgsl)
field_smooth = 30e-9 + (1000e-9 - 30e-9) * (1 - np.cos(theta)) / 2
# add a sharp 500 nm step at the equator (pathological gradient)
field_step = field_smooth + 500e-9 * (v > 0.5)

for name, field in [("smooth drained profile", field_smooth), ("plus 500nm step front", field_step)]:
    for bend in [5.0, 50.0]:
        params = dict(DEFAULTS); params['patch_enabled'] = 0; params['particle_weight'] = 0.0
        params['bend_strength'] = bend
        st_f, _ = trace_ray(100, params, field=field)
        st_0, _ = trace_ray(100, params, field=None)
        n = min(len(st_f), len(st_0))
        dev = np.degrees(np.arccos(np.clip(st_f[n-1]['p3'] @ st_0[n-1]['p3'], -1, 1)))
        gmax = max(np.linalg.norm(s['f2']) for s in st_f)
        sf = np.mean([s['sf'] for s in st_f])
        print(f"{name:24s} bend={bend:4.1f}: max |grad| {gmax:.2e} /UV, mean step_factor {sf:.2f}, "
              f"endpoint deviation vs straight ray {dev:.2e} deg over {n} steps")

# Same field in micrometres (what thickness_scale=1e6 was meant to do)
params = dict(DEFAULTS); params['patch_enabled'] = 0; params['particle_weight'] = 0.0
st_f, _ = trace_ray(100, params, field=field_smooth * 1e6)
st_0, _ = trace_ray(100, params, field=None)
n = min(len(st_f), len(st_0))
print("if scaled to micrometres (x1e6), bend=5: endpoint deviation %.2f deg" %
      np.degrees(np.arccos(np.clip(st_f[n-1]['p3'] @ st_0[n-1]['p3'], -1, 1))))
