exec(open("cpu_replica.py").read().split("# analytic")[0])
s=Sim(True); dt=100/60
first=None
for k in range(400):
    s.step(dt)
    r=[(s.h[i].min()*1e9, s.h[i].max()*1e9) for i in (1,2,3,4)]
    if first is None and (r[0][1]-r[0][0])>1e-6: first=k
    if k in (0,5,10,15,20,25,30,40,60,80,100,150,200,300,399) or (first is not None and k<first+12):
        print(k, f"t={s.t:.0f}s", "ring1..4 [min,max]nm:", [("%.4g"%a,"%.4g"%b) for a,b in r], "pole", "%.4g"%(s.h[0,0]*1e9), "zeros", int((s.h==0).sum()))
print("first ring-1 phi asymmetry at frame", first)
# single hitch test: a uniform-but-TFE-perturbed field hit by one 0.5 s wall hitch (dt=50 s)
s=Sim(True)
for k in range(30): s.step(0.05)   # gently build some TFE structure
before=s.h.copy()
s.step(0.5*100)
print("hitch dt=50s: min %.4g nm max %.4g nm, zeros %d, negatives %d" % (s.h.min()*1e9, s.h.max()*1e9, (s.h==0).sum(), (s.h<0).sum()))
