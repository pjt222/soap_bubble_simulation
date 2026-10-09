"""Thin-film optics audit: phase convention, two-beam vs Airy, polarization order.

Run: python3 -I airy_check.py
Pure math; no external data.
"""
import sys
import numpy as np

N_FILM = 1.33


def snell_cos_t(cos_i, n=N_FILM):
    sin_i = np.sqrt(np.maximum(0.0, 1.0 - cos_i**2))
    sin_t = sin_i / n
    return np.sqrt(np.maximum(0.0, 1.0 - sin_t**2))


def fresnel_amp(cos_i, n=N_FILM):
    """Amplitude coefficients air(1) -> film(n)."""
    cos_t = snell_cos_t(cos_i, n)
    r_s = (cos_i - n * cos_t) / (cos_i + n * cos_t)
    r_p = (n * cos_i - cos_t) / (n * cos_i + cos_t)
    return r_s, r_p, cos_t


def exact_slab_reflectance(d_nm, lam_nm, cos_i, n=N_FILM):
    """Exact single-layer (air/film/air) reflectance from the multiple-beam sum.

    r_tot = (r12 + r23 e^{i delta}) / (1 + r12 r23 e^{i delta}),  r23 = -r12,
    delta = 4 pi n d cos_t / lambda  (round-trip phase, NO extra pi).
    Done separately per polarization, then averaged (unpolarized light).
    """
    r_s, r_p, cos_t = fresnel_amp(cos_i, n)
    delta = 4.0 * np.pi * n * d_nm * cos_t / lam_nm
    e = np.exp(1j * delta)
    out = []
    for r12 in (r_s, r_p):
        r23 = -r12
        r_tot = (r12 + r23 * e) / (1.0 + r12 * r23 * e)
        out.append(np.abs(r_tot) ** 2)
    return 0.5 * (out[0] + out[1]), out[0], out[1]


def two_beam_reflectance(d_nm, lam_nm, cos_i, n=N_FILM):
    """Two-beam (first two reflected beams only), amplitude-coherent, per pol."""
    r_s, r_p, cos_t = fresnel_amp(cos_i, n)
    delta = 4.0 * np.pi * n * d_nm * cos_t / lam_nm
    e = np.exp(1j * delta)
    out = []
    for r12 in (r_s, r_p):
        r21 = -r12
        t12t21 = 1.0 - r12**2
        r_tot = r12 + t12t21 * r21 * e
        out.append(np.abs(r_tot) ** 2)
    return 0.5 * (out[0] + out[1])


def code_airy(d_nm, lam_nm, cos_i, n=N_FILM, add_pi=True):
    """Replicates interference_lut.rs:55-71 / bubble.wgsl:447-467.

    R_avg = (Rs+Rp)/2 first, then F sin^2(phase/2)/(1+F sin^2(phase/2)),
    phase = 2 pi (2 n d cos_t)/lambda + pi.
    """
    r_s, r_p, cos_t = fresnel_amp(cos_i, n)
    R = 0.5 * (r_s**2 + r_p**2)
    one_minus_r = np.maximum(1.0 - R, 0.001)
    F = 4.0 * R / one_minus_r**2
    phase = 2.0 * np.pi * (2.0 * n * d_nm * cos_t) / lam_nm + (np.pi if add_pi else 0.0)
    s2 = np.sin(0.5 * phase) ** 2
    return F * s2 / np.maximum(1.0 + F * s2, 0.001)


def cpu_airy(d_nm, lam_nm, cos_i, n=N_FILM):
    """Replicates interference.rs:324-366 (OPD + lambda/2, then 2R(1-cos)/(1+R^2-2Rcos))."""
    if d_nm <= 0:
        return 0.0
    r_s, r_p, cos_t = fresnel_amp(cos_i, n)
    R = 0.5 * (r_s**2 + r_p**2)
    opd = 2.0 * n * d_nm * cos_t + lam_nm / 2.0
    phi = 2.0 * np.pi * opd / lam_nm
    c = np.cos(phi)
    return 2.0 * R * (1.0 - c) / (1.0 + R**2 - 2.0 * R * c)


def main():
    lam = 550.0
    R0 = ((N_FILM - 1) / (N_FILM + 1)) ** 2
    print(f"R0 = ((n-1)/(n+1))^2 = {R0:.6f}; 4R0/(1+R0)^2 = {4*R0/(1+R0)**2:.6f}")

    print("\n[1] d -> 0 limit at normal incidence, lambda=550")
    for d in [0.0, 0.5, 1.0, 5.0, 10.0, 29.9, 30.0, 50.0, 100.0]:
        ex, _, _ = exact_slab_reflectance(d, lam, 1.0)
        print(f"  d={d:6.1f}  exact={ex:.5f}  code(+pi)={code_airy(d, lam, 1.0):.5f}  "
              f"code(no pi)={code_airy(d, lam, 1.0, add_pi=False):.5f}  cpu(+lambda/2)={cpu_airy(d, lam, 1.0):.5f}")

    print("\n[2] Max abs error over d in [0,1500] (step 0.5 nm) for several lambdas, cos_i in {1, 0.5, 0.3}")
    d = np.arange(0.0, 1500.0001, 0.5)
    for cos_i in (1.0, 0.5, 0.3):
        for lam in (400.0, 450.0, 550.0, 650.0, 700.0):
            ex, _, _ = exact_slab_reflectance(d, lam, cos_i)
            cd = code_airy(d, lam, cos_i)
            cd_fix = code_airy(d, lam, cos_i, add_pi=False)
            tb = two_beam_reflectance(d, lam, cos_i)
            corr = np.corrcoef(ex, cd)[0, 1]
            print(f"  cos_i={cos_i:.1f} lam={lam:.0f}: max|code-exact|={np.max(np.abs(cd-ex)):.5f} "
                  f"(max exact={ex.max():.5f}), corr(code,exact)={corr:+.4f}; "
                  f"max|codeNoPi-exact|={np.max(np.abs(cd_fix-ex)):.2e}; "
                  f"max|twobeam-exact|={np.max(np.abs(tb-ex)):.2e} ({100*np.max(np.abs(tb-ex))/ex.max():.2f}% of peak)")

    print("\n[3] Shift identity: code(d) == exact(d + lambda/(4 n cos_t)) at normal incidence")
    lam = 550.0
    shift = lam / (4 * N_FILM)
    ex_shift, _, _ = exact_slab_reflectance(d + shift, lam, 1.0)
    print(f"  shift = {shift:.2f} nm; max|code(d) - exact(d+shift)| = {np.max(np.abs(code_airy(d, lam, 1.0) - ex_shift)):.2e}")
    for lam in (450.0, 650.0):
        print(f"  (shift is wavelength dependent: lambda={lam:.0f} -> {lam/(4*N_FILM):.2f} nm)")

    print("\n[4] Complement identity: code ~= F - exact for small F (normal incidence)")
    R = R0
    F = 4 * R / (1 - R) ** 2
    ex, _, _ = exact_slab_reflectance(d, 550.0, 1.0)
    print(f"  F={F:.5f}; max|code - (F/(1+F)) + exact*...| check: max|code + exact - F/(1+F)*(...)|")
    print(f"  max|code + exact - F| = {np.max(np.abs(code_airy(d,550.0,1.0) + ex - F)):.2e} (F={F:.5f})")

    print("\n[5] Polarization order: Airy(mean R) [code] vs mean(Airy(Rs), Airy(Rp)) [exact], no-pi phase")
    d = np.arange(0.0, 1500.0001, 0.5)
    for cos_i in (1.0, 0.6, 0.3, 0.15, 0.05):
        ex, rs, rp = exact_slab_reflectance(d, 550.0, cos_i)
        cd_fix = code_airy(d, 550.0, cos_i, add_pi=False)
        r_s, r_p, cos_t = fresnel_amp(cos_i)
        print(f"  cos_i={cos_i:.2f}: Rs={r_s**2:.4f} Rp={r_p**2:.4f}; max exact={ex.max():.4f}; "
              f"max|Airy(Ravg)-exact|={np.max(np.abs(cd_fix-ex)):.4f} "
              f"({100*np.max(np.abs(cd_fix-ex))/ex.max():.1f}% of peak)")

    print("\n[6] Fringe period in wavelength ~ lambda^2/(2 n d): Nyquist for 50 nm sampling needs period > 100 nm")
    for dd in (300, 500, 800, 1000, 1137, 1500, 2000):
        print(f"  d={dd:5d}: period@450={450**2/(2*N_FILM*dd):6.1f} nm, @550={550**2/(2*N_FILM*dd):6.1f} nm, @650={650**2/(2*N_FILM*dd):6.1f} nm")


if __name__ == '__main__':
    main()
