import sys; sys.path.append("/home/phtho/.local/lib/python3.12/site-packages")
import numpy as np
f=np.float32
# GPU DrainageParams::default (gpu_drainage.rs:46-67)
dt=f(0.001); g=f(9.81); eta=f(0.001); rho=f(1000.0); R=f(0.025)
H,W=128,256
h=f(500e-9)
print("ulp(500nm f32) =", np.spacing(h))
coeff = rho*g/(f(3.0)*eta)
dth = f(np.pi)/f(H-1)
worst=0
changed=0
for ti in range(1,H-1):
    th=f(ti)*dth
    s=np.sin(th, dtype=np.float32)
    dterm = -coeff*h*h*h*s
    new = np.maximum(h+dt*dterm, f(0))
    if new!=h: changed+=1
    worst=max(worst, abs(float(dt*dterm)))
print("drainage_coeff f32 =", coeff, " max |dt*dh/dt| per step =", worst, " ratio to ulp =", worst/float(np.spacing(h)))
print("rows whose value changes after 1 step (dt=1ms):", changed, "of", H-2)
# with the 'intended' per-step dt = frame_dt*time_scale/steps = (1/60)*100/10
for step_dt in [f(1/60*100/10), f(1/60*500/10), f(1/10*100/10)]:
    ch=0; incs=[]
    for ti in range(1,H-1):
        th=f(ti)*dth; s=np.sin(th,dtype=np.float32)
        new=h+step_dt*(-coeff*h*h*h*s)
        if new!=h: ch+=1; incs.append(float(h-new)/float(np.spacing(h)))
    print(f"step_dt={float(step_dt):.4f}s rows changed {ch}/{H-2}; decrement in ulps: min {min(incs) if incs else 0:.2f} max {max(incs) if incs else 0:.2f}")
# real-time scale: physical drainage rate magnitude
print("code rate at equator (m/s, dimensionally m^2/s):", float(coeff*h**3))
