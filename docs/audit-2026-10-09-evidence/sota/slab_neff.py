import math
n_film, n_clad, lam = 1.33, 1.0, 532.0
k0 = 2*math.pi/lam
def te0_neff(d):
    # symmetric slab TE0: kappa*tan(kappa*d/2) = gamma
    f = lambda ne: math.sqrt(n_film**2-ne**2)*k0*math.tan(math.sqrt(n_film**2-ne**2)*k0*d/2) - math.sqrt(ne**2-n_clad**2)*k0
    lo, hi = n_clad+1e-9, n_film-1e-9
    # ensure kappa*d/2 < pi/2 branch: restrict lo so that tan argument < pi/2
    kappa_max = math.pi/d  # kappa*d/2 < pi/2
    ne_min = math.sqrt(max(n_clad**2, n_film**2 - (kappa_max/k0)**2)) + 1e-9
    lo = max(lo, ne_min)
    for _ in range(200):
        mid = 0.5*(lo+hi)
        if f(mid) > 0: lo = mid
        else: hi = mid
    return 0.5*(lo+hi)
prev = None
print(" d_nm   n_eff   dn_eff/dd [1/um]")
for d in (50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000):
    ne = te0_neff(d); h = 1.0
    deriv = (te0_neff(d+h)-te0_neff(d-h))/(2*h)*1000
    print(f"{d:5d}  {ne:.4f}  {deriv:.4f}")
