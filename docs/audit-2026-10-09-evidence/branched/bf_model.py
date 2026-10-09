"""Python replica of src/render/shaders/branched_flow_compute.wgsl (main kernel).

Mirrors the WGSL line-by-line where it matters for geometry/math checks.
Float64 for geometry, float32 emulation where quantization matters.
"""
import sys
import numpy as np

PI = 3.14159265359
TW, TH = 256, 128           # THICKNESS_WIDTH/HEIGHT (wgsl:74-75)
GRID_N, CELL = 10, 0.1      # spatial hash (wgsl:93-95)


def normalize(v):
    return v / np.linalg.norm(v)


def normal_to_uv(n):        # wgsl:108-114
    phi = np.arctan2(n[2], n[0])
    theta = np.arccos(np.clip(n[1], -1.0, 1.0))
    return np.array([(phi + PI) / (2 * PI), theta / PI])


def uv_to_normal(u, v):     # inverse of normal_to_uv (same as wgsl:460-466)
    phi = (u * 2 - 1) * PI
    th = v * PI
    return np.array([np.sin(th) * np.cos(phi), np.cos(th), np.sin(th) * np.sin(phi)])


def sample_thickness(field, uv):   # wgsl:241-261
    fx = np.clip(uv[0], 0, 1) * (TW - 1)
    fy = np.clip(uv[1], 0, 1) * (TH - 1)
    x0, y0 = int(np.floor(fx)), int(np.floor(fy))
    x1, y1 = min(x0 + 1, TW - 1), min(y0 + 1, TH - 1)
    sx, sy = fx - np.floor(fx), fy - np.floor(fy)
    h00, h10 = field[y0, x0], field[y0, x1]
    h01, h11 = field[y1, x0], field[y1, x1]
    h0 = h00 + (h10 - h00) * sx
    h1 = h01 + (h11 - h01) * sx
    return h0 + (h1 - h0) * sy


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def thickness_gradient_uv(field, uv):  # wgsl:268-289
    eps = 0.01
    hr = sample_thickness(field, uv + np.array([eps, 0]))
    hl = sample_thickness(field, uv - np.array([eps, 0]))
    hu = sample_thickness(field, uv + np.array([0, eps]))
    hd = sample_thickness(field, uv - np.array([0, eps]))
    theta = uv[1] * 3.14159265
    st = np.sin(theta)
    cst = max(st, 0.1)
    taper = smoothstep(0.0, 0.15, st)
    gx = (hr - hl) / (2 * eps * cst) * taper
    gy = (hu - hd) / (2 * eps)
    return np.array([gx, gy])


def uv_force_to_tangent_frame(f_uv, p, t1, t2):  # wgsl:132-170
    xz = np.sqrt(p[0] ** 2 + p[2] ** 2)
    if xz > 0.001:
        phat = np.array([-p[2] / xz, 0.0, p[0] / xz])
        that = np.cross(p, phat)
    else:
        phat = np.array([1.0, 0, 0]); that = np.array([0, 0, 1.0])
    f3 = phat * f_uv[0] + that * f_uv[1]
    return np.array([f3 @ t1, f3 @ t2]), f3


def frame(entry):           # wgsl:428-433
    up = np.array([0.0, 1.0, 0.0])
    if abs(entry @ up) > 0.99:
        up = np.array([1.0, 0, 0])
    t1 = normalize(np.cross(entry, up))
    t2 = normalize(np.cross(entry, t1))
    return t1, t2


def hash21_f32(px, py):     # wgsl:222-226 emulated in float32
    f = np.float32
    p3 = np.array([px, py, px], dtype=f) * f(0.1031)
    p3 = p3 - np.floor(p3)
    d = f(np.dot(p3, (p3[[1, 2, 0]] + f(33.33)).astype(f)))
    p3 = (p3 + d).astype(f)
    r = (p3[0] + p3[1]) * p3[2]
    return f(r - np.floor(r))


def trace_ray(ray_idx, params, field=None, force_fn=None, record=True):
    """Replicates main() for one ray. force_fn(uv)->uv-space force (optional extra).
    Returns list of dicts per step."""
    entry = normalize(np.array(params['entry'], float))
    bd = np.array(params['beam_dir'], float)
    beam = normalize(bd - entry * (bd @ entry))
    t1, t2 = frame(entry)
    r1 = float(hash21_f32(np.float32(ray_idx) * np.float32(0.1), 0.0))
    r2 = float(hash21_f32(np.float32(ray_idx) * np.float32(0.1) + np.float32(100.0), 1.0))
    vel = normalize(np.array([beam @ t1, beam @ t2]))
    perp = np.array([-vel[1], vel[0]])
    spread = params['spread_angle']
    pos = perp * (r1 - 0.5) * spread + vel * (r2 - 0.5) * spread * 0.1
    if params['patch_enabled']:
        P = normalize(uv_to_normal(params['patch_u'], params['patch_v']))
        to_patch = P - entry
        off = np.array([to_patch @ t1, to_patch @ t2])
        ps = params['patch_half'] * PI
        pos = off + perp * (r1 - 0.5) * ps + vel * (r2 - 0.5) * ps * 0.5
    intensity = 1.0
    dt = params['step_size']
    out = []
    for step in range(params['ray_steps']):
        p3 = normalize(entry + t1 * pos[0] + t2 * pos[1])
        uv = normal_to_uv(p3)
        f_uv = np.zeros(2)
        if field is not None:
            f_uv += thickness_gradient_uv(field, uv) * (1 - params['particle_weight'])
        if force_fn is not None:
            f_uv += force_fn(uv) * params['particle_weight']
        f2, f3 = uv_force_to_tangent_frame(f_uv, p3, t1, t2)
        gm = np.linalg.norm(f2)
        sf = np.clip(1.0 / max(gm * 10.0, 0.333), 0.3, 3.0)
        adt = dt * sf
        if record:
            out.append(dict(p3=p3, uv=uv, pos=pos.copy(), sf=sf, intensity=intensity,
                            deposit=intensity * 0.15 * sf, f2=f2, f3=f3))
        vel = vel + f2 * params['bend_strength'] * adt
        vm = np.linalg.norm(vel)
        if vm > 0.001:
            vel = vel / vm
        pos = pos + vel * adt
        intensity *= (1 - params['intensity_falloff'])
        if np.linalg.norm(pos) > 2.5 or intensity < 0.01:
            break
    return out, (entry, t1, t2)


DEFAULTS = dict(entry=[0.0, 0.0, 1.0], beam_dir=[-0.5, -0.866, 0.0], num_rays=8192,
                ray_steps=200, step_size=0.005, bend_strength=5.0, spread_angle=0.4,
                intensity_falloff=0.001, particle_weight=0.1, patch_enabled=1,
                patch_u=0.5, patch_v=0.5, patch_half=0.158)
