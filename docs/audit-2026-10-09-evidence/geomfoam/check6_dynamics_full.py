"""Faithful re-implementation of FoamSimulator::step (foam_dynamics.rs:62-200) incl. Stokes drag
(mu=0.001, line 163), buoyancy clamp (line 112), damping 0.8/step (line 188), |v|<=0.1 clamp (192-196)."""
import numpy as np
rho=1000.0; h=500e-9; g=np.array([0,-9.81,0.]); mu=0.001; kr=5.0; vdw=0.001
def step(P,V,Rs,dt):
    n=len(P); F=np.zeros_like(P)
    for i in range(n):
        R=Rs[i]; A=4*np.pi*R*R; m_f=A*h*rho; Vol=4/3*np.pi*R**3
        net=m_f*9.81-Vol*1.2*9.81
        F[i]+=g/np.linalg.norm(g)*max(net,0.0)
        for j in range(i+1,n):
            dlt=P[j]-P[i]; d=np.linalg.norm(dlt); u=dlt/d; S=Rs[i]+Rs[j]; f=np.zeros(3)
            if S*3>d>S: f+=u*vdw*Rs[i]*Rs[j]/d**2
            if S-d>0: f-=u*kr*(S-d)**1.5
            if S*1.2>d>S*0.95: f+=u*2.0*(S*1.2-d)
            F[i]+=f; F[j]-=f
        F[i]+=-V[i]*6*np.pi*mu*R
    for i in range(n):
        R=Rs[i]; m=max(4*np.pi*R*R*h*rho,1e-9)
        V[i]=(V[i]+F[i]/m*dt)*0.8
        s=np.linalg.norm(V[i])
        if s>0.1: V[i]*=0.1/s
        P[i]+=V[i]*dt
# 1) isolated bubble, tiny perturbation
for R in (0.015,0.025):
    for fps in (60,30):
        P=np.zeros((1,3)); V=np.array([[1e-6,0,0]]); dt=1/fps
        sp=[]
        for k in range(120): step(P,V,[R],dt); sp.append(np.linalg.norm(V[0]))
        k_over_m=6*np.pi*mu*R/(4*np.pi*R*R*h*rho)
        print(f"isolated R={R*1e3:.0f}mm @{fps}fps: drag k/m={k_over_m:.0f}/s, per-step factor (1-k dt)*0.8={(1-k_over_m*dt)*0.8:+.2f}; |v| after 0.5s={sp[int(0.5*fps)-1]:.2e}, after 2s={sp[-1]:.3f} m/s")
# 2) default 5-bubble cluster (foam.rs:417-424), 60 fps, 5 s
P=np.array([[0,0,0],[0.042,0,0],[-0.038,0.015,0],[0,0.040,0],[0.018,-0.038,0.015]],float)
Rs=[0.025,0.02,0.022,0.018,0.02]; V=np.zeros_like(P); dt=1/60
hist=[]
for k in range(300):
    step(P,V,Rs,dt); hist.append(P.copy())
H=np.array(hist); speeds=np.linalg.norm(np.diff(H,axis=0),axis=2)/dt
print("default cluster @60fps: mean |v| over last 1s per bubble (m/s):", np.round(speeds[-60:].mean(0),3))
print("  frame-to-frame displacement over last 1s per bubble (mm):", np.round(np.linalg.norm(np.diff(H[-61:],axis=0),axis=2).mean(0)*1e3,2))
print("  centroid drift over 5 s (mm):", np.round((H[-1].mean(0)-H[0].mean(0))*1e3,1))
