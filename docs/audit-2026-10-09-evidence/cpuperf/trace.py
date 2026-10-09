import importlib.util, numpy as np
spec = importlib.util.spec_from_file_location("port", "./cpu_drainage_port.py")
src = open("./cpu_drainage_port.py").read().split("# Stability bound")[0]
ns = {}; exec(src, ns); Sim = ns["Sim"]
for fps in (60, 10):
    s = Sim(); dt = 100.0/fps
    print(f"--- fps={fps} dt={dt:.3f}s")
    for k in range(80):
        s.step(dt)
        a = np.abs(s.h); r, c = np.unravel_index(np.argmax(a), a.shape)
        zero_rows = [i for i in range(1, 8) if (s.h[i] == 0).any()]
        # phi-checkerboard amplitude per near-pole row (Nyquist mode)
        alt = (-1.0)**np.arange(s.nphi)
        ck = [abs((s.h[i]*alt).mean()) for i in (1, 2, 3, 4)]
        if k < 40 and (k % 3 == 0 or a.max() > 1e-6):
            print(f"f{k:3d} max={a.max():.2e} at row {r:3d}; nyquist amp rows1-4={['%.1e'%x for x in ck]}; rows with zeros(1..7)={zero_rows}; min={s.h.min():.2e}")
    print("frozen(<10nm) cells near pole rows 1-6:", [(i, int((s.h[i] < 1e-8).sum())) for i in range(1, 7)], "of", s.nphi)
    print("cells < 10nm total:", int((s.h < 1e-8).sum()), "/", s.h.size)
