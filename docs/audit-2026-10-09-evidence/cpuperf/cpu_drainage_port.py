# Faithful numpy port of DrainageSimulator::step (src/physics/drainage.rs:291-506)
# driven the way RenderPipeline::update drives it (pipeline.rs:1292-1315): one step per frame,
# dt = frame_dt * drainage_time_scale (default 100.0, pipeline.rs:802).
import numpy as np, sys
np.seterr(all='ignore')
PI = np.pi
def rust_max0(x):              # f64::max(x, 0.0): NaN -> 0.0
    y = np.where(np.isnan(x), 0.0, x); return np.maximum(y, 0.0)

class Sim:
    def __init__(s, res=128, h0=500e-9, R=0.025, D=1e-9, rho=1000., g=9.81, eta=1e-3, hcrit=10e-9, tfe=True):
        s.nt, s.nphi = res, 2*res
        s.h = np.full((s.nt, s.nphi), h0)
        s.dth = PI/(s.nt-1); s.dph = 2*PI/s.nphi
        s.R, s.D, s.hc = R, D, hcrit
        s.kdrain = rho*g/(3*eta)
        s.t = 0.0; s.fronts = []; s.next_spawn = 0.5; s.interval = 2.0; s.ratio = 0.85; s.tfe = tfe
        th = np.arange(s.nt)*s.dth
        s.sin = np.sin(th); s.cos = np.cos(th)
        s.sin_safe = np.where(np.abs(s.sin) < 1e-10, np.copysign(1e-10, s.sin), s.sin)
    def step(s, dt):
        sc = s.h.copy(); nt = s.nt
        hc = sc[1:nt-1]; hm = sc[0:nt-2]; hp = sc[2:nt]
        hpm = np.roll(hc, 1, axis=1); hpp = np.roll(hc, -1, axis=1)
        sn = s.sin[1:nt-1, None]; cs = s.cos[1:nt-1, None]; ss = s.sin_safe[1:nt-1, None]
        drain = -s.kdrain * hc**3 * sn
        d2t = (hp - 2*hc + hm)/s.dth**2; d1t = (hp - hm)/(2*s.dth); d2p = (hpp - 2*hc + hpm)/s.dph**2
        lap = (d2t + cs/ss*d1t + d2p/(ss*ss))/(s.R*s.R)
        new = rust_max0(hc + dt*(drain + s.D*lap))
        skip = hc < s.hc                        # `continue` keeps old value
        s.h[1:nt-1] = np.where(skip, s.h[1:nt-1], new)
        s.h[0, :] = s.h[1, :].sum()/s.nphi; s.h[-1, :] = s.h[-2, :].sum()/s.nphi
        if s.tfe: s.tfe_update(dt)
        s.t += dt
    def tfe_update(s, dt):
        s.next_spawn -= dt
        if s.next_spawn <= 0:
            phi = abs(np.sin(s.t*7.89+1.23))*2*PI; w = 0.3+abs(np.sin(s.t*3.21))*0.4; v = 0.05+abs(np.cos(s.t*5.67))*0.05
            s.fronts.append([phi, 0.0, w, v])
            if len(s.fronts) > 8: s.fronts.pop(0)
            s.next_spawn = s.interval*(0.5+abs(np.sin(s.t*12.345)))
        for f in s.fronts: f[1] = min(f[1]+f[3]*dt, PI/2)
        s.fronts = [f for f in s.fronts if f[1] < PI/2*0.95]
        if not s.fronts: return
        th = np.arange(s.nt)[:, None]*s.dth; ph = np.arange(s.nphi)[None, :]*s.dph
        done = np.zeros_like(s.h, dtype=bool)
        lo = min(int(np.floor((PI/2-f[1])/s.dth)) for f in s.fronts); hi = max(min(int(np.ceil((PI/2+f[1])/s.dth)), s.nt-1) for f in s.fronts)
        rows = np.zeros((s.nt, 1), bool); rows[min(lo, s.nt-1):hi+1] = True
        for f in s.fronts:
            inside = (np.abs(th-PI/2) <= f[1])
            pd = np.abs(ph-f[0]); pd = np.where(pd > PI, 2*PI-pd, pd)
            m = inside & (pd < f[2]/2) & rows & ~done
            s.h = np.where(m, s.h*(1-(1-s.ratio)*0.5*dt), s.h); done |= m
    def equator(s):                         # get_thickness(PI/2, 0) (drainage.rs:519-555)
        tc = (PI/2)/s.dth; i = min(int(np.floor(tc)), s.nt-2); f = tc-i
        return s.h[i, 0]*(1-f) + s.h[i+1, 0]*f

def run(fps, frames=600, scale=100.0, tfe=True):
    s = Sim(tfe=tfe); dt = scale/fps; first_bad = None
    for k in range(frames):
        s.step(dt)
        mx = np.nanmax(np.where(np.isfinite(s.h), s.h, np.nan)) if np.isfinite(s.h).any() else np.inf
        bad = (not np.isfinite(s.h).all()) or mx > 1e-5   # >10 um (20x initial) = blow-up
        if bad and first_bad is None: first_bad = k
    eq = s.equator()
    rows_bad = np.where(~np.isfinite(s.h).all(axis=1) | (np.abs(s.h).max(axis=1) > 1e-5))[0]
    print(f"fps={fps:>4} dt_step={dt:7.3f}s tfe={tfe}: first blow-up frame={first_bad}, "
          f"max|h|={np.nanmax(np.abs(s.h)):.3e} m, equator={eq*1e9:.4g} nm, "
          f"bad rows={rows_bad[:6].tolist()}{'...' if len(rows_bad)>6 else ''} (n={len(rows_bad)})")

# Stability bound for phi-diffusion at row k: dt_max = R^2 sin^2(k dth) dphi^2 / (2 D)
R, D, nt = 0.025, 1e-9, 128; dth = PI/(nt-1); dph = 2*PI/(2*nt)
for k in (1, 2, 3, 4, 6, 10):
    print(f"row {k}: explicit phi-diffusion dt_max = {R**2*np.sin(k*dth)**2*dph**2/(2*D):8.3f} s")
for fps in (144, 60, 30, 10):
    run(fps)
run(60, tfe=False)
