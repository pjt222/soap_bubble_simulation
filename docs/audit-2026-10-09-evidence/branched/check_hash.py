import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bf_model import *
f = np.float32
for N in [8192, 32768, 65536]:
    r1 = np.array([hash21_f32(f(i) * f(0.1), 0.0) for i in range(N)])
    r2 = np.array([hash21_f32(f(i) * f(0.1) + f(100.0), 1.0) for i in range(N)])
    h, _ = np.histogram(r1, bins=32, range=(0, 1))
    uniq = len(np.unique(np.round(r1 * 2**20)))
    print(f"N={N}: rand1 histogram min/max per 32 bins {h.min()}/{h.max()} (ideal {N/32:.0f}); "
          f"unique(2^-20) {uniq}; corr(r1,r2) {np.corrcoef(r1, r2)[0,1]:+.3f}; "
          f"chi2/dof {np.sum((h - N/32)**2/(N/32))/31:.2f}")
