"""Geodesic curvature produced by the chart kick (wgsl:511,530-542) vs the target kappa = k*|F_perp|.
The code's discrete update implies dT/ds = k F_perp in the *chart*; we measure what that is on the sphere."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bf_model import *

E = np.array([0, 0, 1.0]); t1, t2 = frame(E)
k = 1.0; dt = 1e-4; Fmag = 1.0


def to3(pos):
    return normalize(E + t1 * pos[0] + t2 * pos[1])


def geodesic_curvature(pts):
    # finite-difference curve on unit sphere, arc-length reparam
    p0, p1, p2 = pts
    s1 = np.arccos(np.clip(p0 @ p1, -1, 1)); s2 = np.arccos(np.clip(p1 @ p2, -1, 1))
    d2 = 2 * ((p2 - p1) / s2 - (p1 - p0) / s1) / (s1 + s2)
    tang = d2 - (d2 @ p1) * p1       # remove normal (-p) part
    return np.linalg.norm(tang)


print("alpha(deg) | motion     | force dir  | measured kappa / k|F|")
for r in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
    alpha = np.degrees(np.arctan(r))
    for motion in ['radial', 'transverse']:
        pos = np.array([r, 0.0])
        vel = np.array([1.0, 0.0]) if motion == 'radial' else np.array([0.0, 1.0])
        p = to3(pos)
        # 3D tangent of motion, then a force perpendicular to it (in tangent plane at p)
        J = (np.array([t1, t2]).T - np.outer(p, p @ np.array([t1, t2]).T))
        T3 = normalize(J @ vel)
        F3 = Fmag * normalize(np.cross(p, T3))        # perpendicular, tangent
        pts = []
        for _ in range(3):
            pts.append(to3(pos))
            f2 = np.array([F3 @ t1, F3 @ t2])
            vel = vel + f2 * k * dt
            vel = vel / np.linalg.norm(vel)
            pos = pos + vel * dt
        kap = geodesic_curvature(pts)
        pred = (1 / np.cos(np.radians(alpha)) ** 3) if motion == 'radial' else np.cos(np.radians(alpha))
        print(f"{alpha:9.1f} | {motion:10s} | perp       | {kap/(k*Fmag):7.3f}   (analytic {pred:.3f})")
