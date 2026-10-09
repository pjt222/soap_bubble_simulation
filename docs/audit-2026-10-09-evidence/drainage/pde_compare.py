import sys; sys.path.append("/home/phtho/.local/lib/python3.12/site-packages")
import numpy as np
rho,g,eta,R,h=1000.,9.81,1e-3,0.025,500e-9
th=np.linspace(0,np.pi,2001)
# code (drainage.rs:346 / drainage.wgsl:136): dh/dt = -(rho g/(3 eta)) h^3 sin(theta)
code=-(rho*g/(3*eta))*h**3*np.sin(th)
# conservative lubrication on sphere, rigid (no-slip) interfaces: q = rho g sin(th) h^3/(12 eta)
# dh/dt = -(1/(R sin th)) d/dth (sin th q) ; for uniform h -> -(rho g h^3/(6 eta R)) cos th
cons=-(rho*g*h**3/(6*eta*R))*np.cos(th)
# numeric check of the divergence for uniform h
q=rho*g*np.sin(th)*h**3/(12*eta)
div=np.gradient(np.sin(th)*q, th)/(R*np.maximum(np.sin(th),1e-12))
print("max |numeric div - analytic|/max:", np.max(np.abs(-div[5:-5]-cons[5:-5]))/np.max(np.abs(cons)))
w=2*np.pi*R**2*np.sin(th)   # area element per dtheta
I=lambda f: np.trapezoid(f*w, th)
print("net volume rate code  [m^3/s]: %.3e  (analytic -(rho g h^3/3eta) pi^2 R^2 = %.3e)" % (I(code), -(rho*g*h**3/(3*eta))*np.pi**2*R**2))
print("net volume rate conservative: %.3e" % I(cons))
for t in (0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi):
    i=np.argmin(abs(th-t))
    print(f"theta={t:.3f}: code {code[i]*1e9:+.4e} nm/s   conservative {cons[i]*1e9:+.4e} nm/s")
print("ratio |cons|_max/|code|_max = 1/(2R) =", np.max(abs(cons))/np.max(abs(code)))
# drainage timescale estimates (rigid film): tau = 6 eta R/(rho g h^2)
print("rigid-film drainage time 6 eta R/(rho g h^2) at 500nm: %.3g s" % (6*eta*R/(rho*g*h**2)))
# TFE ratio misuse: thickness after T seconds inside a front
for T in (2,10,30,60): print(f"TFE: after {T}s inside front h/h0 = {np.exp(-0.075*T):.3f} (Monier target 0.8-0.9)")
