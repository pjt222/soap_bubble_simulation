import math
# Defaults from src/render/gpu_drainage.rs:49-63
radius = 0.025; diffusion = 1e-9; viscosity = 1e-3; density = 1000.0; gravity = 9.81
grid_width, grid_height = 256, 128
delta_theta = math.pi / (grid_height - 1)
delta_phi = 2 * math.pi / grid_width
theta_rows = [i * delta_theta for i in range(1, grid_height - 1)]
dx_theta = radius * delta_theta
dx_phi = [radius * math.sin(t) * delta_phi for t in theta_rows]
dt_limit = [1.0 / (2 * diffusion * (1 / dx_theta**2 + 1 / d**2)) for d in dx_phi]
print(f"dx_theta = {dx_theta:.3e} m, dx_phi equator = {max(dx_phi):.3e} m, dx_phi row1 = {min(dx_phi):.3e} m, ratio = {max(dx_phi)/min(dx_phi):.1f}")
print(f"explicit diffusion dt limit: equator {max(dt_limit):.3e} s, near-pole row {min(dt_limit):.3e} s, ratio {max(dt_limit)/min(dt_limit):.0f}")
for dt in (1e-3, (1/60)*100/10, (1/60)*500/1):
    print(f"dt={dt:.4g}s  stable everywhere? {dt <= min(dt_limit)}  dt/limit_pole = {dt/min(dt_limit):.3g}")
k = density * gravity / (3 * viscosity)
for h_nm in (500, 1000):
    h = h_nm * 1e-9
    coded = k * h**3
    conservative = k * h**3 * 2 / radius
    print(f"h={h_nm} nm: coded |dh/dt|max={coded:.3e} (units m^2/s), conservative-form scale 2k h^3/R={conservative:.3e} m/s = {conservative*1e9:.4f} nm/s; ratio={conservative/coded:.0f}")
gamma = 0.03
h = 500e-9
mobility = gamma * h**3 / (3*viscosity)
for name, dx in (("equator", max(dx_phi)), ("row1", min(dx_phi))):
    print(f"capillary 4th-order explicit dt limit at {name}: {dx**4/(32*mobility):.3e} s")
