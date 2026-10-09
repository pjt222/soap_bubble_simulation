import sys
import numpy as np, math
# SimulationConfig::default (config.rs) -> DrainageSimulator::new (drainage.rs:256-281)
NT, NP = 128, 256
R=0.025; D=1e-9; rho=1000.; g=9.81; eta=1e-3; hcrit=10e-9; h0=500e-9
dth=math.pi/(NT-1); dph=2*math.pi/NP
theta=np.arange(NT)*dth; phi=np.arange(NP)*dph

class Front:
    def __init__(s,pc,w,v): s.pc=pc; s.ext=0.0; s.w=w; s.v=v
    def advance(s,dt): s.ext=min(s.ext+s.v*dt, math.pi/2)

class Sim:
    def __init__(s, tfe=True):
        s.h=np.full((NT,NP),h0); s.t=0.0; s.fronts=[]; s.next=0.5; s.tfe=tfe
    def step(s,dt):
        sc=s.h.copy(); C=rho*g/(3*eta)
        for i in range(1,NT-1):
            th=theta[i]; st=math.sin(th); ct=math.cos(th)
            sts = st if abs(st)>=1e-10 else math.copysign(1e-10,st)
            hc=sc[i]; hm=sc[i-1]; hp=sc[i+1]
            pm=np.roll(hc,1); pp=np.roll(hc,-1)
            drain=-C*hc**3*st
            lap=((hp-2*hc+hm)/dth**2 + ct/sts*(hp-hm)/(2*dth) + (pp-2*hc+pm)/dph**2/(sts*sts))/R**2
            new=np.maximum(hc+dt*(drain+D*lap),0.0)
            s.h[i]=np.where(hc<hcrit, s.h[i], new)
        s.h[0]=s.h[1].mean(); s.h[NT-1]=s.h[NT-2].mean()
        if s.tfe: s.mr(dt)
        s.t+=dt
    def mr(s,dt):
        s.next-=dt
        if s.next<=0:
            t=s.t
            pc=abs(math.sin(t*7.89+1.23))*2*math.pi; w=0.3+abs(math.sin(t*3.21))*0.4; v=0.05+abs(math.cos(t*5.67))*0.05
            s.fronts.append(Front(pc,w,v))
            if len(s.fronts)>8: s.fronts.pop(0)
            s.next=2.0*(0.5+abs(math.sin(s.t*12.345)))
        for f in s.fronts: f.advance(dt)
        s.fronts=[f for f in s.fronts if f.ext<math.pi/2*0.95]
        if not s.fronts: return
        lo=NT; hi=0
        for f in s.fronts:
            a,b=math.pi/2-f.ext, math.pi/2+f.ext
            lo=min(lo,int(math.floor(a/dth))); hi=max(hi,min(int(math.ceil(b/dth)),NT-1))
        lo=min(lo,NT-1)
        rate=(1-0.85)*0.5
        for i in range(lo,hi+1):
            th=i*dth
            done=np.zeros(NP,bool)
            for f in s.fronts:
                if abs(th-math.pi/2)>f.ext: continue
                d=np.abs(phi-f.pc); d=np.where(d>math.pi,2*math.pi-d,d)
                m=(d<f.w/2)&~done
                s.h[i,m]*=(1-rate*dt); done|=m

def run(dt, frames, tfe=True, label=""):
    s=Sim(tfe)
    for k in range(frames):
        s.step(dt)
        if not np.all(np.isfinite(s.h)): print(label,"non-finite at frame",k); break
    ring=[float(np.std(s.h[i])/h0) for i in (1,2,3,4)]
    print(f"{label} dt={dt:.4g}s frames={frames} simT={s.t:.1f}s  max={s.h.max()*1e9:.4g}nm min={s.h.min()*1e9:.4g}nm  zeros={(s.h==0).sum()} phi-std/h0 rings1-4={['%.2e'%r for r in ring]}  pole={s.h[0,0]*1e9:.4g}nm eq={s.h[NT//2,0]*1e9:.4g}nm")
    return s

# analytic explicit-Euler limit for the Laplacian (worst eigenvalue)
lam=[]
for i in range(1,NT-1):
    st=math.sin(theta[i])
    lam.append(4*D/(R*R*dth*dth)+4*D/(R*R*st*st*dph*dph))
print("explicit diffusion limit dt_max = 2/lambda_max =", 2/max(lam), "s at ring", 1+int(np.argmax(lam)))
for i in (1,2,3,4,5,8): print("  ring",i,"dt_max",2/lam[i-1])
frame=1/60
run(frame*100, 600, True, "default ts=100 @60fps (TFE on)")
run(frame*100, 600, False, "default ts=100 @60fps (TFE off)")
run(0.1, 600, True, "dt=0.1 (stable ref)")
