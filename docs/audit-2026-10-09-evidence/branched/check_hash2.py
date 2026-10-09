import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bf_model import *
f = np.float32
M = 2**20
for N in [8192, 32768, 65536]:
    r1 = np.array([hash21_f32(f(i) * f(0.1), 0.0) for i in range(N)], dtype=f)
    r2 = np.array([hash21_f32(f(i) * f(0.1) + f(100.0), 1.0) for i in range(N)], dtype=f)
    u1 = len(np.unique(r1)); up = len(set(zip(r1.tolist(), r2.tolist())))
    # perpendicular offset resolution: spread 0.4 chart units -> distinct offsets closer than 1 texel?
    off = np.sort((r1 - 0.5) * 0.4)
    gaps = np.diff(off)
    print(f"N={N}: exact-unique rand1 {u1} ({u1/N*100:.0f}%), unique (rand1,rand2) pairs {up} ({up/N*100:.0f}%); "
          f"birthday expectation at 2^-20 bins {M*(1-np.exp(-N/M)):.0f}; zero-gap neighbours {np.sum(gaps==0)}")
    # input precision: f32 ulp of i*0.1*0.1031 at max i
    x = f(N-1) * f(0.1) * f(0.1031)
    print(f"   max hash input p*0.1031 = {x:.2f}, f32 ulp there = {np.spacing(x):.2e}")
