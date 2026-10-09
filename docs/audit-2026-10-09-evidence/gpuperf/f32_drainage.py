import sys
import numpy as np
f=np.float32
# drainage.wgsl:142-147,170-171 ; gpu_drainage.rs:49-55 defaults, pipeline.rs:708 initial 500e-9
rho,g,eta=f(1000.0),f(9.81),f(0.001)
h=f(500e-9)
coeff=rho*g/(f(3.0)*eta)
for theta_deg in [10,45,90]:
    s=f(np.sin(np.radians(theta_deg)))
    dhdt=-coeff*h*h*h*s
    for dt in [f(0.001), f(1/60*100/10)]:
        new=np.maximum(h+dt*dhdt,f(0))
        print(f"theta={theta_deg:3d} dt={float(dt):.4f}s  dh/dt={float(dhdt):.3e} m/s  incr={float(dt*dhdt):.3e} m  ulp(h)={float(np.spacing(h)):.3e}  changed={new!=h}  rel_incr={float(dt*dhdt/h):.2e}")
# uniform field => laplacian exactly 0 in f32?
hn=np.full(5,h,dtype=f)
print("uniform laplacian numerator:", float(hn[0]-f(2)*hn[1]+hn[2]))
# GRIN force scale with meters vs micrometers (branched_flow_compute.wgsl:268-288, 499, 530)
dh_dv=f(200e-9)  # e.g. 500nm top -> 300nm bottom over v in [0,1]
for unit,scale in [("meters (as coded)",1.0),("micrometers (thickness_scale=1e6 applied)",1e6)]:
    grad=dh_dv*scale
    force=grad*(1-0.1)
    dv_step=force*5.0*(0.005*3.0)
    print(f"{unit:45s} |grad|={grad:.3e} force={force:.3e} dvel/step={dv_step:.3e} rad, over 200 steps={dv_step*200:.3e} rad")
