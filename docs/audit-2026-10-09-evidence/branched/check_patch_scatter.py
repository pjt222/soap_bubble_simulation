"""Default patch config WITH the default scatterer field (branched_flow.rs generate_scatterers)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bf_model import *


def halton(i, b):
    r, f = 0.0, 1.0 / b
    while i > 0:
        r += f * (i % b); i //= b; f /= b
    return r


def fract(x):
    return x - np.floor(x)


def gen_scatterers(num, time, base_strength, base_radius, patch):
    out = []
    for i in range(num):
        u, v = halton(i + 1, 2), halton(i + 1, 3)
        hs_ = np.float32(i) * np.float32(0.1031) + time * 0.1
        ju = fract(np.sin(hs_) * 43758.547) * 0.02
        jv = fract(np.cos(hs_) * 43758.547) * 0.02
        if patch:
            cu, cv, h = patch
            mnu, mxu = max(cu - h, 0), min(cu + h, 1); mnv, mxv = max(cv - h, 0), min(cv + h, 1)
            mu, mv = mnu + u * (mxu - mnu), mnv + v * (mxv - mnv)
            pu, pv = np.clip(mu + ju * h * 2, 0, 1), np.clip(mv + jv * h * 2, 0, 1)
        else:
            pu, pv = (u + ju) % 1.0, (v + jv) % 1.0
        sh = fract(np.sin(i * 0.7531 + 0.3) * 43758.547)
        sign = 1.0 if sh > 0.5 else -1.0
        sv = 0.8 + 0.4 * fract(np.sin(i * 0.9371) * 43758.547)
        rv = 0.8 + 0.4 * fract(np.cos(i * 0.5791) * 43758.547)
        sigma = max(base_radius * rv, 1e-6)
        out.append((pu, pv, sign * base_strength * sv, 1.0 / (2 * sigma * sigma)))
    return np.array(out)


def make_force_fn(S):
    # brute force over all scatterers (the hash only truncates; checked separately)
    def f(uv):
        d = uv[None, :] - S[:, :2]
        r2 = np.sum(d * d, axis=1)
        m = r2 <= 4.5 / S[:, 3]
        e = np.exp(-r2[m] * S[m, 3])
        return np.sum(d[m] * (S[m, 2] * S[m, 3] * 2.0 * e)[:, None], axis=0)
    return f


if __name__ == '__main__':
    params = dict(DEFAULTS)
    S = gen_scatterers(400, 0.0, 0.5, 0.03, (0.5, 0.5, 0.158))
    ff = make_force_fn(S)
    hs = 0.158
    cover = np.zeros((64, 64), float)
    tot = inp = 0
    sfs = []
    fmag = []
    for ray in range(0, 8192, 16):
        steps, _ = trace_ray(ray, params, field=None, force_fn=ff)
        tot += len(steps)
        for s in steps:
            sfs.append(s['sf']); fmag.append(np.linalg.norm(s['f2']))
            if abs(s['uv'][0] - 0.5) <= hs and abs(s['uv'][1] - 0.5) <= hs:
                inp += 1
                lu = (s['uv'][0] - (0.5 - hs)) / (2 * hs); lv = (s['uv'][1] - (0.5 - hs)) / (2 * hs)
                cover[min(int(lv * 64), 63), min(int(lu * 64), 63)] += s['deposit']
    print(f"[patch+scatter] fraction of steps inside patch: {inp/tot:.3f}")
    print(f"[patch+scatter] fraction of patch bins with any deposit: {(cover>0).mean():.3f}")
    cols = (cover > 0).any(axis=0)
    print(f"[patch+scatter] local-u columns touched: {cols.sum()}/64, first col {np.argmax(cols)}")
    sfs = np.array(sfs); fmag = np.array(fmag)
    print(f"[adaptive] step_factor distribution: min {sfs.min():.3f} median {np.median(sfs):.3f} "
          f"frac==3.0 {np.mean(sfs>2.999):.3f} frac==0.3 {np.mean(sfs<0.3001):.3f}")
    print(f"[force] |F_tangent| (particle-weighted) median {np.median(fmag):.3f}, p95 {np.percentile(fmag,95):.3f}, max {fmag.max():.3f}")
    # column profile of deposit (sum over v), to show where light ends up
    prof = cover.sum(axis=0); prof /= prof.max()
    print("[patch+scatter] deposit vs local-u (8 bins):", np.round([prof[i*8:(i+1)*8].sum() for i in range(8)], 2))
