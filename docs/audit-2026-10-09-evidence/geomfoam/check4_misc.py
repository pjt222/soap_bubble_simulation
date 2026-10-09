"""Check 5-8: instanced normal transform, sampling math, HCP, winding, Bond-number docstring."""
import numpy as np
from math import erf, sqrt, log, exp

rng = np.random.default_rng(1)

# 5. instanced normal: model upper3x3 = diag(r, r a, r); code: M * (nx, ny/a, nz)
r, a = 0.03, 0.8
M = np.diag([r, r * a, r])
th = np.radians(50); ph = np.radians(20)
# ellipsoid surface point param and its true normal
p = np.array([np.sin(th) * np.cos(ph), np.cos(th), np.sin(th) * np.sin(ph)])  # unit-sphere vertex (foam mesh is SphereMesh::new(1.0,3))
n_vertex = p.copy()
n_code = M @ np.array([n_vertex[0], n_vertex[1] / a, n_vertex[2]]); n_code /= np.linalg.norm(n_code)
n_true = np.linalg.inv(M).T @ n_vertex; n_true /= np.linalg.norm(n_true)
print("instanced normal: code=", np.round(n_code, 4), " unit-sphere n=", np.round(n_vertex, 4), " true ellipsoid n=", np.round(n_true, 4),
      " angle err (deg)=", np.degrees(np.arccos(np.clip(n_code @ n_true, -1, 1))).round(2))

# 6a. random positions phi = rand*pi (foam_generation.rs:233) — polar clustering
N = 200000
u1, u2, u3 = rng.random(N), rng.random(N), rng.random(N)
theta = u1 * 2 * np.pi; phi = u2 * np.pi; rr = np.cbrt(u3)
y = rr * np.cos(phi)
cap = np.abs(np.cos(phi)) > np.cos(np.radians(30))
print(f"phi~U(0,pi): fraction within 30deg of +/-y axis = {cap.mean():.3f} (uniform-solid-angle expectation {1-np.cos(np.radians(30)):.3f})")
# density near axis: fraction with sin(phi)*r < 0.2 (cylinder of radius 0.2 R around y)
cyl = (rr * np.sin(phi) < 0.2).mean()
# uniform-in-ball expectation
pts = rng.normal(size=(N, 3)); pts /= np.linalg.norm(pts, axis=1)[:, None]; pts *= np.cbrt(rng.random(N))[:, None]
cyl_u = (np.hypot(pts[:, 0], pts[:, 2]) < 0.2).mean()
print(f"fraction within cylinder rho<0.2R around y: code={cyl:.3f}, uniform ball={cyl_u:.3f}, ratio={cyl/cyl_u:.2f}")

# 6b. HCP generator (foam_generation.rs:347-388), jitter=0
def hcp(count, s):
    n = max(int(np.ceil((count / 2) ** (1 / 3))), 2)
    ox = (n - 1) * s / 2; oy = (n - 1) * s * np.sqrt(2 / 3) / 2; oz = (n - 1) * s / 2
    lh = s * np.sqrt(2 / 3)
    P = []
    for layer in range(n):
        yy = layer * lh - oy; b = layer % 2 == 1
        for i in range(n):
            for j in range(n):
                if len(P) >= count: return np.array(P)
                x = i * s - ox; z = j * s - oz
                if j % 2 == 1: x += s / 2
                if b: x += s / 2; z += s / (2 * np.sqrt(3))
                P.append((x, yy, z))
    return np.array(P)

def nn_stats(P):
    D = np.linalg.norm(P[:, None] - P[None], axis=-1); np.fill_diagonal(D, np.inf)
    return D.min(axis=1)

s = 0.05
P = hcp(10_000, s)  # big lattice -> interior statistics
D = np.linalg.norm(P[:, None] - P[None], axis=-1); np.fill_diagonal(D, np.inf)
nn = D.min()
# within-layer neighbour distances for one interior point of layer 0
layer0 = P[np.isclose(P[:, 1], P[:, 1].min())]
c = layer0[len(layer0) // 2 + 3]
dl = np.sort(np.linalg.norm(layer0 - c, axis=1))[1:7]
print("HCP in-plane 6 nearest distances / spacing:", np.round(dl / s, 3), "(ideal hexagonal: six 1.0)")
cnt = np.sum((D < 1.01 * s), axis=1)
interior = np.all(np.abs(P - P.mean(0)) < 0.3 * np.ptp(P, axis=0), axis=1)
print("HCP coordination (neighbours within 1.01*spacing), interior points: unique counts", np.unique(cnt[interior], return_counts=True), "(ideal 12)")
print("min interlayer distance / s:", end=" ")
l0 = P[np.isclose(P[:, 1], np.unique(P[:, 1])[2])]; l1 = P[np.isclose(P[:, 1], np.unique(P[:, 1])[3])]
print(np.round(np.min(np.linalg.norm(l0[:, None] - l1[None], axis=-1)) / s, 3))

# FCC / BCC nearest-neighbour distance relative to 'spacing'
print("nn distance / spacing: SC=1, BCC=sqrt(3)/2=%.3f, FCC=1/sqrt(2)=%.3f" % (np.sqrt(3) / 2, 1 / np.sqrt(2)))

# 6c. clamping distorts distributions (sample_radius clamps to [min,max])
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
mn, mx, mean, sd, sig = 0.015, 0.030, 0.022, 0.005, 0.3
pl = Phi((mn - mean) / sd); ph = 1 - Phi((mx - mean) / sd)
print(f"Normal(mean=.022,sd=.005) clamped to [.015,.030]: {pl*100:.1f}% at min, {ph*100:.1f}% at max")
mu = log(mean) - sig**2 / 2
pl = Phi((log(mn) - mu) / sig); ph = 1 - Phi((log(mx) - mu) / sig)
print(f"LogNormal(mean=.022,sigma=.3): {pl*100:.1f}% at min, {ph*100:.1f}% at max")
k = 1 / (1.5 - 1); thetaS = mean * (1.5 - 1)
gs = rng.gamma(k, thetaS, 1_000_000)
print(f"Schulz-Flory PDI=1.5 (gamma k={k}, theta={thetaS}): {np.mean(gs<mn)*100:.1f}% at min, {np.mean(gs>mx)*100:.1f}% at max; sample mean {gs.mean():.4f}, PDI check <r^2>/<r>^2={np.mean(gs**2)/gs.mean()**2:.3f}")
ln = np.exp(mu + sig * rng.normal(size=1_000_000))
print(f"lognormal mean check: {ln.mean():.5f} (target {mean})")

# Marsaglia-Tsang re-implementation check
def mt(shape, scale, n):
    out = []
    for _ in range(n):
        a = shape; boost = 1.0
        if a < 1:
            boost = rng.random() ** (1 / a); a += 1
        d = a - 1 / 3; c = 1 / np.sqrt(9 * d)
        while True:
            x = rng.normal(); v = 1 + c * x
            if v <= 0: continue
            v = v**3; u = rng.random()
            if u < 1 - 0.0331 * x**4 or np.log(u) < 0.5 * x * x + d * (1 - v + np.log(v)):
                out.append(d * v * scale * boost); break
    return np.array(out)
for sh in (0.5, 2.0, 5.0):
    s_ = mt(sh, 1.0, 40000)
    print(f"Marsaglia-Tsang shape={sh}: mean={s_.mean():.3f} (exp {sh}), var={s_.var():.3f} (exp {sh})")

# 8. winding of SphereMesh triangles (geometry.rs:216-233) at equator
lon_seg, lat_seg = 16, 8
def vpos(lat, lon):
    t = lat / lat_seg * np.pi; p = lon / lon_seg * 2 * np.pi
    return np.array([np.sin(t) * np.cos(p), np.cos(t), np.sin(t) * np.sin(p)])
lat, lon = 4, 3
cur = (lat, lon); nxt = (lat + 1, lon); cur1 = (lat, lon + 1); nxt1 = (lat + 1, lon + 1)
for tri in [(cur, nxt, cur1), (cur1, nxt, nxt1)]:
    A, B, C = [vpos(*v) for v in tri]
    nrm = np.cross(B - A, C - A); cen = (A + B + C) / 3
    print("tri normal . outward =", np.sign(nrm @ cen), "(+1 = CCW seen from outside)")
# disk mesh winding (foam_renderer.rs:455-483): +z?
A = np.array([0, 0, 0.]); B = np.array([1, 0, 0.]); C = np.array([np.cos(2*np.pi/16), np.sin(2*np.pi/16), 0])
print("wall disk centre tri normal z:", np.cross(B - A, C - A)[2])

# 7. Bond number docstring geometry.rs:118-122 (rho=1000, gamma=0.025, g=9.81, R = D/2)
for D in (0.01, 0.05, 0.10):
    Bo = 1000 * 9.81 * (D / 2) ** 2 / 0.025
    print(f"D={D*100:.0f} cm: Bo(radius, rho=1000)={Bo:.3f}, aspect=clamp(1-0.4Bo)={min(max(1-0.4*Bo,0.7),1):.3f}")
for rho in (1.2, 1.0):
    print(" rho=", rho, [round(rho * 9.81 * (D / 2) ** 2 / 0.025, 4) for D in (0.01, 0.05, 0.10)])
