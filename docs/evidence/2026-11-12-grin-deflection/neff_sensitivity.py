"""TE0 vs TM0 and wavelength sensitivity of d ln n_eff / dd at 500 nm (cited in
src/physics/film_waveguide.rs), the 2000 nm n_eff, and the geodesic (small-circle) shift of the
theta deflection test at column 447. Bisection on the slab dispersion relation, as in
docs/audit-2026-10-09-evidence/branched/check_neff.py."""
import math
def neff(d, n1=1.33, lam=532.0, tm=False):
    k0 = 2*math.pi/lam
    lo, hi = 1.0+1e-12, n1-1e-12
    for _ in range(200):
        m = 0.5*(lo+hi)
        kap = k0*math.sqrt(n1*n1-m*m); gam = k0*math.sqrt(m*m-1)
        if kap*d/2 >= math.pi/2: f = 1.0
        else: f = math.tan(kap*d/2) - gam/kap*((n1*n1) if tm else 1.0)
        if f > 0: lo = m
        else: hi = m
    return 0.5*(lo+hi)
g = lambda d, **k: (math.log(neff(d+0.5,**k))-math.log(neff(d-0.5,**k)))*1000
print(f"n_eff(2000) TE0 = {neff(2000):.5f}")
print(f"d ln n/dd at 500 nm: TE0 {g(500):.4f}, TM0 {g(500,tm=True):.4f} per um, ratio {g(500,tm=True)/g(500):.3f}")
print(f"TE0 at 450 nm light {g(500,lam=450):.4f}, 650 nm {g(500,lam=650):.4f}")
dphi = (447/511 - 0.75)*2*math.pi
print(f"geodesic small-circle shift at column 447: dv = {0.1*(1-math.cos(dphi))/math.pi:.5f}")
