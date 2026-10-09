"""TE0 / TM0 effective index of a free-standing soap film (air | n=1.33 | air) vs thickness, lambda=532nm.
Gives the physically-motivated GRIN coefficient (1/n_eff) dn_eff/dd for the ray eq. dT/ds = grad_perp ln n_eff."""
import sys
sys.path.append('/home/phtho/.local/lib/python3.12/site-packages')
import numpy as np
n1, n0, lam = 1.33, 1.0, 532e-9
k0 = 2*np.pi/lam
def neff(d, tm=False):
    lo, hi = n0 + 1e-9, n1 - 1e-12
    for _ in range(200):
        m = 0.5*(lo+hi)
        kap = k0*np.sqrt(n1**2 - m**2); gam = k0*np.sqrt(m**2 - n0**2)
        rhs = gam/kap * ((n1/n0)**2 if tm else 1.0)
        f = np.tan(kap*d/2) - rhs   # fundamental mode: kap*d/2 in (0, pi/2)
        if kap*d/2 >= np.pi/2: f = 1.0  # beyond branch -> n_eff too small
        if f > 0: lo = m
        else: hi = m
    return 0.5*(lo+hi)
print(" d(nm)  n_eff(TE0)  (1/n)dn/dd [1/um]   n_eff(TM0)")
for d in [30, 100, 200, 300, 500, 800, 1000, 2000]:
    dd = 1e-9
    ne = neff(d*1e-9); g = (neff(d*1e-9+dd)-neff(d*1e-9-dd))/(2*dd)/ne
    print(f"{d:6d}  {ne:.4f}      {g*1e-6:8.4f}            {neff(d*1e-9, True):.4f}")
# implied bend_strength if thickness were in micrometres and gradient per UV (u: 2pi rad, v: pi rad):
g500 = (neff(501e-9)-neff(499e-9))/(2e-9)/neff(500e-9)
print(f"\nphysical k at 500nm: {g500:.3e} 1/m = {g500*1e-6:.3f} per um")
print(f"=> with h in um and grad per UV-unit, equivalent bend_strength ~ {g500*1e-6/(2*np.pi):.3f} (u) .. {g500*1e-6/np.pi:.3f} (v)  [code default 5.0]")
print(f"=> with h in metres (as now): ~{g500/(2*np.pi):.2e} .. {g500/np.pi:.2e}")
