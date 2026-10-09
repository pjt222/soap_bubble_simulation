exec(open("cpu_replica.py").read().split("# analytic")[0])
for dt in (8.333, 15.0, 20.0):
    s=Sim(True)
    for k in range(6):
        s.step(dt)
        if (s.h<0).any():
            print(f"dt={dt}: negative thickness at frame {k}: min={s.h.min()*1e9:.4g} nm, count={(s.h<0).sum()}, fronts={[round(f.ext,3) for f in s.fronts]}")
            break
    else:
        print(f"dt={dt}: no negatives in 6 frames; min={s.h.min()*1e9:.4g} nm")
