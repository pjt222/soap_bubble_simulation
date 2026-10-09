"""Check 2: shared-wall curvature (foam.rs:118-128) magnitude and rendered bulge direction
(foam_renderer.rs:241-270 + wall.wgsl:90-131), contact point placement, Plateau angle."""
import numpy as np

gamma = 0.025


def code_wall_R(Ra, Rb):
    pa = 4 * gamma / Ra
    pb = 4 * gamma / Rb
    dp = pa - pb
    return 2 * gamma / dp  # foam.rs:125


def physical_wall_R(Ra, Rb):
    # wall is a film (two surfaces): dp = 4 gamma / Rw -> 1/Rw = 1/Ra - 1/Rb
    return 1.0 / (1.0 / Ra - 1.0 / Rb)


for Ra, Rb in [(0.025, 0.02), (0.02, 0.025), (0.03, 0.015)]:
    print(f"Ra={Ra} Rb={Rb}: code R_wall={code_wall_R(Ra,Rb):+.4f} m, physical={physical_wall_R(Ra,Rb):+.4f} m, ratio={code_wall_R(Ra,Rb)/physical_wall_R(Ra,Rb):.3f}")


def rot_z_to(d):
    d = d / np.linalg.norm(d)
    z = np.array([0, 0, 1.0])
    if abs(d[2] - 1) < 1e-6:
        return np.eye(3)
    if abs(d[2] + 1) < 1e-6:
        return np.diag([1, -1, -1.0])
    axis = np.cross(z, d)
    axis /= np.linalg.norm(axis)
    ang = np.arccos(z @ d)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def intersection_radius(ra, rb, d):
    s = ra + rb
    if d >= s or d <= abs(ra - rb):
        return 0.0
    p = (s + d) * (s - d) * (d + ra - rb) * (d - ra + rb)
    return np.sqrt(p) / (2 * d)


# Case: A smaller (higher pressure) at origin, B larger along +x
for (Ra, Rb) in [(0.02, 0.025), (0.025, 0.02)]:
    A = np.zeros(3)
    d = 0.95 * (Ra + Rb)
    B = np.array([d, 0, 0])
    n = (B - A) / d
    t = Ra / (Ra + Rb)
    contact = A + (B - A) * t  # foam.rs:115-116
    Rcurv = code_wall_R(Ra, Rb)
    a = intersection_radius(Ra, Rb, d)
    Rot = rot_z_to(n)
    # wall.wgsl: centre vertex (r=0) and rim vertex (r=1)
    def vert(local_xy):
        lp = np.array([local_xy[0], local_xy[1], 0.0]) * a
        rw = np.linalg.norm(local_xy) * a
        R = abs(Rcurv)
        if 0.001 < R < 1000 and rw < R:
            zd = R - np.sqrt(R * R - rw * rw)
            if Rcurv < 0:
                zd = -zd
            lp[2] = zd
        return contact + Rot @ lp
    c = vert([0, 0]); r = vert([1, 0])
    smaller = 'A' if Ra < Rb else 'B'
    # The cap's centre relative to its rim: if centre is displaced toward bubble X, the wall bulges into X.
    along = (c - r) @ n  # >0: centre is further toward B than rim -> bulges into B
    bulge_into = 'B' if along > 0 else 'A'
    larger = 'B' if Rb > Ra else 'A'
    print(f"Ra={Ra}, Rb={Rb}: smaller={smaller}, code curvature_radius={Rcurv:+.4f}, "
          f"rendered cap bulges into {bulge_into}; physics: bulges into larger={larger} -> {'OK' if bulge_into==larger else 'WRONG DIRECTION'}")
    # contact point vs true intersection plane
    x_true = (d * d + Ra * Ra - Rb * Rb) / (2 * d)
    x_code = d * t
    sag = abs(Rcurv) - np.sqrt(Rcurv**2 - a * a)
    print(f"    intersection plane x_true={x_true*1e3:.3f} mm, code contact x={x_code*1e3:.3f} mm, offset={abs(x_true-x_code)*1e3:.3f} mm; "
          f"disk radius a={a*1e3:.3f} mm; rim displaced by sagitta {sag*1e3:.3f} mm (rim should stay on intersection circle)")

# Plateau angle: equilibrium separation the dynamics settles to (~0.95*(Ra+Rb)) vs Plateau separation
for R in [0.0225]:
    for frac in [0.95, 0.933]:
        d = frac * 2 * R
        x = d / 2
        alpha = np.degrees(np.arccos(x / R))  # angle between outer film tangent and wall plane
        print(f"equal R={R}: d={frac:.3f}*(2R): outer-film/outer-film exterior angle = {2*alpha:.1f} deg, film-wall angle = {180-alpha:.1f} deg (Plateau: 120/120)")
        print(f"   wall disk radius = {np.sqrt(R*R - x*x)/R:.3f} R (Plateau double bubble: sin60 = {np.sin(np.pi/3):.3f} R)")
Ra, Rb = 0.025, 0.02
dP = np.sqrt(Ra**2 + Rb**2 - Ra * Rb)
print(f"Plateau-consistent separation for Ra={Ra},Rb={Rb}: d={dP*1e3:.2f} mm vs sum of radii {(Ra+Rb)*1e3:.1f} mm (ratio {dP/(Ra+Rb):.3f})")
# verify 120 deg at junction for Plateau separation with Rw from physical formula
x = (dP * dP + Ra * Ra - Rb * Rb) / (2 * dP)
J = np.array([x, np.sqrt(Ra * Ra - x * x)])
nA = J / Ra
nB = (J - np.array([dP, 0])) / Rb
print("angle between sphere normals at junction (expect 60):", np.degrees(np.arccos(nA @ nB)))
