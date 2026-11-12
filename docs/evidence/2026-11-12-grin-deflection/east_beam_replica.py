"""Per-ray replica of the branched-flow loop for the phi deflection test of PR #60: entry
(-1, 0, 0), beam south, ln n_eff = slope * phi offset, rays offset y0 in [-0.2, 0.2]. Each ray's
u offset where it crosses row 191 (v = 191/255); the beam mean of these crossings is close to the
central ray, so the wide-beam excess in deposit_replica.py is not a displacement of the rays."""
import numpy as np
entry = np.array([-1.0, 0.0, 0.0]); up = np.array([0.0, 1.0, 0.0])
t1 = np.cross(entry, up); t1 /= np.linalg.norm(t1)
t2 = np.cross(entry, t1); t2 /= np.linalg.norm(t2)
def frames(p):
    xz = np.hypot(p[0], p[2]); ph = np.array([-p[2]/xz, 0.0, p[0]/xz]); return ph, np.cross(p, ph)
def uv(p):
    return (np.arctan2(p[2], p[0])+np.pi)/(2*np.pi), np.arccos(np.clip(p[1], -1, 1))/np.pi
target_v = 191/255
def ray(slope, y0, dt=0.005, steps=400):
    pos = np.array([y0, 0.0]); vel = np.array([0.0, 1.0]); prev = None
    for _ in range(steps):
        p = entry + t1*pos[0] + t2*pos[1]; p /= np.linalg.norm(p)
        ph, th = frames(p); sin_t = np.hypot(p[0], p[2])
        f3 = ph*slope/max(sin_t, 0.1)
        f = np.array([f3 @ t1, f3 @ t2])
        sf = np.clip(1/max(np.linalg.norm(f)*10, 0.333), 0.3, 3.0); adt = dt*sf
        u, v = uv(p); du = u if u < 0.5 else u-1
        if prev is not None and prev[1] < target_v <= v:
            w = (target_v-prev[1])/(v-prev[1]); return prev[0]+w*(du-prev[0])
        prev = (du, v)
        vel = vel + f*adt; vel /= np.linalg.norm(vel); pos = pos + vel*adt
    return None
y0s = np.linspace(-0.2, 0.2, 81)
for slope in (0.1, -0.1):
    shifts = [ray(slope, y)-ray(0.0, y) for y in y0s]
    central = ray(slope, 0.0)-ray(0.0, 0.0)
    print(f"slope {slope:+}: central-ray shift {central:+.5f}, beam-mean shift {np.mean(shifts):+.5f}")
x = np.tan((target_v-0.5)*np.pi); st = np.sqrt(1+x*x)
y = 0.1*(((st**3-1)/3 + x*np.arcsinh(x) - st + 1)/2)
print(f"test's closed-form central prediction {np.arctan(y)/(2*np.pi):.5f}")
