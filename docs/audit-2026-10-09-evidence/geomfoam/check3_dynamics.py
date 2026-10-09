"""Check 3/4: buoyancy/mass (foam_dynamics.rs:99-113, 181-183) and integrator behaviour
(foam_dynamics.rs:118-200). Re-implements the force law verbatim and runs 2 bubbles."""
import numpy as np

rho_w = 1000.0
g = 9.81
rho_air = 1.2
h = 500e-9
for R in [0.0012, 0.015, 0.025, 0.03]:
    A = 4 * np.pi * R**2
    V = 4 / 3 * np.pi * R**3
    m_film = A * h * rho_w
    W_film = m_film * g
    B = V * rho_air * g
    net_code = max(W_film - B, 0.0)
    m_gas = V * rho_air  # enclosed air ~ ambient density (overpressure 4 gamma/R ~ 4 Pa << 1e5 Pa)
    m_added = 0.5 * rho_air * V
    print(f"R={R*1e3:.1f} mm: film weight={W_film:.3e} N, displaced-air buoyancy={B:.3e} N, code net={net_code:.3e} N; "
          f"physical net (film+gas-buoyancy)={W_film + m_gas*g - B:.3e} N; m_film={m_film:.3e} kg, m_gas={m_gas:.3e}, m_added={m_added:.3e} -> inertia underestimated x{(m_film+m_gas+m_added)/m_film:.1f}")
print("crossover radius where code gravity becomes nonzero: R < 3 h rho_w/rho_air =", 3 * h * rho_w / rho_air * 1e3, "mm")

# Stability estimate of contact springs
R = 0.0225
m = 4 * np.pi * R**2 * h * rho_w
k_rep = 5.0
for ov in [0.001, 0.003]:
    keff = 1.5 * k_rep * ov**0.5
    w = np.sqrt(keff / m)
    print(f"Hertz linearised at overlap {ov*1e3:.0f} mm: k_eff={keff:.3f} N/m, omega={w:.0f} rad/s, omega*dt(0.016)={w*0.016:.2f} (symplectic Euler bound 2)")
k_adh = 2.0
w = np.sqrt(k_adh / m)
print(f"adhesion spring k=2 N/m: omega={w:.0f} rad/s, omega*dt={w*0.016:.2f}")
S = 2 * R
F_adh = 2.0 * (1.2 * S - 0.95 * S)
print(f"adhesion force just outside 0.95*S: {F_adh:.4f} N -> dv per 16ms step = {F_adh/m*0.016:.1f} m/s (clamp 0.1 m/s)")
F_rep = k_rep * (0.05 * S) ** 1.5
print(f"repulsion just inside 0.95*S (overlap {0.05*S*1e3:.2f} mm): {F_rep:.2e} N; net force jumps from +{F_adh:.4f} (attract) to -{F_rep:.1e} (repel) across d=0.95 S")


def simulate(dt, steps, damping=0.8, d0=None):
    Ra = Rb = R
    xa, xb = 0.0, d0 if d0 else 0.042
    va = vb = 0.0
    hist = []
    for _ in range(steps):
        d = xb - xa
        S = Ra + Rb
        f = 0.0
        if S * 3 > d > S:
            f += 0.001 * Ra * Rb / d**2
        ov = S - d
        if ov > 0:
            f -= k_rep * ov**1.5
        if S * 1.2 > d > S * 0.95:
            f += 2.0 * (S * 1.2 - d)
        # force on A along +x is f (direction A->B), on B is -f
        for which in (0, 1):
            F = f if which == 0 else -f
            if which == 0:
                va += F / m * dt; va *= damping; va = np.clip(va, -0.1, 0.1)
            else:
                vb += F / m * dt; vb *= damping; vb = np.clip(vb, -0.1, 0.1)
        xa += va * dt
        xb += vb * dt
        hist.append((xb - xa, va, vb))
    return np.array(hist)


for dt in [0.016, 0.008, 0.001]:
    hist = simulate(dt, int(2.0 / dt))
    last = hist[-int(0.5 / dt):]
    clamp_frac = np.mean(np.abs(hist[:, 1]) >= 0.0999)
    print(f"dt={dt}: final d/S={last[-1,0]/(2*R):.4f}, d/S range over last 0.5s=[{last[:,0].min()/(2*R):.4f},{last[:,0].max()/(2*R):.4f}], "
          f"|v|=0.1 clamp active in {clamp_frac*100:.0f}% of steps, mean|v| last 0.5s={np.mean(np.abs(last[:,1])):.4f}")

print("per-second velocity retention from damping=0.8/step:", {fps: 0.8**fps for fps in (30, 60, 144)})
