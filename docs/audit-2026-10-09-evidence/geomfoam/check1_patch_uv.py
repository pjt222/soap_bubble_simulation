"""Check 1: SpherePatch mesh placement (geometry.rs:403-427) vs shader UV convention
(bubble.wgsl normal_to_branched_uv, branched_flow_compute.wgsl normal_to_uv / patch_center_3d).
Also: patch area fraction vs claimed ~10% and UI (2*hs)^2."""
import numpy as np

PI = np.float32(np.pi)


def patch_mesh_normals(center_u, center_v, half_size, subs, radius=1.0, aspect=1.0):
    # geometry.rs uv_bounds + generate_mesh_indexed, verbatim math
    min_u = max(center_u - half_size, 0.0)
    max_u = min(center_u + half_size, 1.0)
    min_v = max(center_v - half_size, 0.0)
    max_v = min(center_v + half_size, 1.0)
    r_eq = radius
    r_pol = radius * aspect
    normals = []
    mesh_uv = []
    for j in range(subs + 1):
        v = min_v + (max_v - min_v) * (j / subs)
        theta = np.float32(v) * PI
        for i in range(subs + 1):
            u = min_u + (max_u - min_u) * (i / subs)
            phi = np.float32(u) * np.float32(2.0) * PI
            x = r_eq * np.sin(theta) * np.cos(phi)
            y = r_pol * np.cos(theta)
            z = r_eq * np.sin(theta) * np.sin(phi)
            n = np.array([x / r_eq**2, y / r_pol**2, z / r_eq**2], dtype=np.float32)
            n /= np.linalg.norm(n)
            normals.append(n)
            mesh_uv.append((u, v))
    return np.array(normals), np.array(mesh_uv)


def shader_uv(n):
    # bubble.wgsl:237-242 and branched_flow_compute.wgsl:108-111
    phi = np.arctan2(n[..., 2], n[..., 0])
    theta = np.arccos(np.clip(n[..., 1], -1, 1))
    return np.stack([(phi + np.pi) / (2 * np.pi), theta / np.pi], axis=-1)


def is_in_patch(uv, cu, cv, hs):
    return (np.abs(uv[..., 0] - cu) <= hs) & (np.abs(uv[..., 1] - cv) <= hs)


for (cu, cv, hs) in [(0.5, 0.5, 0.158), (0.1, 0.5, 0.158), (0.9, 0.3, 0.158), (0.5, 0.5, 0.3), (0.5, 0.5, 0.05)]:
    normals, muv = patch_mesh_normals(cu, cv, hs, 32)
    suv = shader_uv(normals)
    inside = is_in_patch(suv, cu, cv, hs)
    print(f"center_u={cu} center_v={cv} hs={hs}: patch-mesh vertices whose shader-uv is_in_patch: "
          f"{inside.sum()}/{len(inside)} ({100*inside.mean():.1f}%)")
    print(f"   shader u range over mesh: min={suv[:,0].min():.3f} max={suv[:,0].max():.3f}; mesh u range [{muv[:,0].min():.3f},{muv[:,0].max():.3f}]")

# Center vertex and compute shader patch_center_3d
cu, cv = 0.5, 0.5
phi_mesh = cu * 2 * np.pi
theta = cv * np.pi
mesh_center = np.array([np.sin(theta) * np.cos(phi_mesh), np.cos(theta), np.sin(theta) * np.sin(phi_mesh)])
phi_comp = (cu * 2 - 1) * np.pi  # branched_flow_compute.wgsl:460
comp_center = np.array([np.sin(theta) * np.cos(phi_comp), np.cos(theta), np.sin(theta) * np.sin(phi_comp)])
print("mesh patch centre (geometry.rs)      :", np.round(mesh_center, 4))
print("compute patch_center_3d (wgsl:460-466):", np.round(comp_center, 4))
print("angle between them (deg):", np.degrees(np.arccos(np.clip(mesh_center @ comp_center, -1, 1))))

# Area fraction of UV square patch on unit sphere centred at v=0.5
for hs in [0.05, 0.158, 0.3]:
    frac = (2 * hs) * (np.cos(np.pi * (0.5 - hs)) - np.cos(np.pi * (0.5 + hs))) / 2
    print(f"hs={hs}: true sphere area fraction={frac*100:.2f}%  UI (2hs)^2={((2*hs)**2)*100:.2f}%  2hs*sin(pi hs)={(2*hs*np.sin(np.pi*hs))*100:.2f}%")
