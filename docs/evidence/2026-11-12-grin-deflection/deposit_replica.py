"""Vectorised replica of the branched-flow propagation loop and deposit_bilinear (u32
truncation), with an exact analytic GRIN force, for the two deflection tests of PR #60
(grin_force_along_theta / _phi_across_the_seam_matches_the_chart_integrator).

Shows that at the default beam spread (0.4) the phi test's deposit-weighted centroid reads
about 5% above the central-ray prediction with the exact force, i.e. the excess is in the
observable, and that a narrow beam (0.1) removes it. Rays are spread uniformly (the shader
uses a hash); deposits follow the shader (bilinear, truncated to u32 at scale 64)."""
import numpy as np, sys
W, H = 512, 256
def run(entry, beam, force_fn, n_rays=8192, steps=400, dt=0.005, truncate=True, shuffle=0, spread=0.4):
    entry = np.asarray(entry, float); up = np.array([0, 1.0, 0])
    t1 = np.cross(entry, up); t1 /= np.linalg.norm(t1); t2 = np.cross(entry, t1); t2 /= np.linalg.norm(t2)
    rng = np.random.default_rng(shuffle)
    r1 = (np.arange(n_rays) + 0.5) / n_rays if shuffle == 0 else rng.random(n_rays)
    vel = np.tile(beam, (n_rays, 1)).astype(float); perp = np.array([-beam[1], beam[0]])
    pos = np.outer(r1 - 0.5, perp) * spread
    tex = np.zeros(W * H, np.uint64); inten = np.ones(n_rays); alive = np.ones(n_rays, bool)
    for _ in range(steps):
        p = entry + np.outer(pos[:, 0], t1) + np.outer(pos[:, 1], t2); p /= np.linalg.norm(p, axis=1)[:, None]
        u = (np.arctan2(p[:, 2], p[:, 0]) + np.pi) / (2 * np.pi); v = np.arccos(np.clip(p[:, 1], -1, 1)) / np.pi
        xz = np.hypot(p[:, 0], p[:, 2]); ph = np.stack([-p[:, 2] / xz, 0 * xz, p[:, 0] / xz], 1); th = np.cross(p, ph)
        east, south = force_fn(u, v)
        f3 = ph * east[:, None] + th * south[:, None]
        f = np.stack([f3 @ t1, f3 @ t2], 1)
        sf = np.clip(1 / np.maximum(np.linalg.norm(f, axis=1) * 10, 0.333), 0.3, 3.0); adt = dt * sf
        # deposit_bilinear
        fx = np.clip(u, 0, 1) * (W - 1); fy = np.clip(v, 0, 1) * (H - 1)
        x0 = np.floor(fx).astype(int); y0 = np.floor(fy).astype(int)
        x1 = np.minimum(x0 + 1, W - 1); y1 = np.minimum(y0 + 1, H - 1); sx = fx - x0; sy = fy - y0
        base = inten * 0.15 * sf * 64.0
        for xi, yi, w in ((x0, y0, (1 - sx) * (1 - sy)), (x1, y0, sx * (1 - sy)), (x0, y1, (1 - sx) * sy), (x1, y1, sx * sy)):
            amount = np.floor(base * w) if truncate else base * w
            np.add.at(tex, (yi * W + xi)[alive], (amount[alive] * (1 if truncate else 1000)).astype(np.uint64))
        vel = vel + f * adt[:, None]; vel /= np.linalg.norm(vel, axis=1)[:, None]
        pos = pos + vel * adt[:, None]; inten *= 0.999
        alive &= ~((np.linalg.norm(pos, axis=1) > 2.5) | (inten < 0.01))
    return tex.reshape(H, W).astype(float)

def east_field(slope):
    def f(u, v):
        s = np.maximum(np.sin(v * np.pi), 0.1)
        return slope / s * np.ones_like(u), 0 * u
    return f
def south_field(slope):
    return lambda u, v: (0 * u, slope * np.ones_like(u))

def row_centroid_seam(tex, row):
    cols = np.arange(W) / 511.0; off = np.where(cols < 0.5, cols, cols - 1)
    return (tex[row] * off).sum() / tex[row].sum()
def col_centroid(tex, col):
    rows = np.arange(H) / 255.0; return (tex[:, col] * rows).sum() / tex[:, col].sum()

if __name__ == "__main__":
    for spread in (0.4, 0.1):
        for truncate in (True, False):
            a = run([-1, 0, 0], np.array([0, 1.0]), east_field(0), truncate=truncate, spread=spread)
            b = run([-1, 0, 0], np.array([0, 1.0]), east_field(0.1), truncate=truncate, spread=spread)
            c = run([0, 0, 1], np.array([1.0, 0]), south_field(0), truncate=truncate, spread=spread)
            d = run([0, 0, 1], np.array([1.0, 0]), south_field(0.1), truncate=truncate, spread=spread)
            east_shift = row_centroid_seam(b, 191) - row_centroid_seam(a, 191)
            south_shift = col_centroid(d, 447) - col_centroid(c, 447)
            print(f"spread {spread} truncate={truncate}: phi test shift {east_shift:+.5f} "
                  f"(chart prediction 0.00845); theta test shift {south_shift:+.5f} (0.01120)")
