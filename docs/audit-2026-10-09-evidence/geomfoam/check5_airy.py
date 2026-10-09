"""Check 9: Airy forms used in wall.wgsl:184-209 vs bubble_instanced.wgsl:161-168, against an
explicit thin-film (air|film|air) characteristic-matrix / multi-beam reflectance at normal incidence."""
import numpy as np
n=1.33; lam=532.0
def exact_R(d):
    # air|film|air, normal incidence, amplitude multi-beam sum (Stokes: r21=-r12, t12 t21 = 1-r12^2)
    r12=(1-n)/(1+n); r21=-r12; t=1-r12**2
    delta=4*np.pi*n*d/lam
    r = r12 + t*r21*np.exp(1j*delta)/(1 - r21**2*np.exp(1j*delta))
    return abs(r)**2
R=((1-n)/(1+n))**2; F=4*R/(1-R)**2
def airy_refl(phase): s=np.sin(phase/2)**2; return F*s/(1+F*s)
def airy_trans(phase): s=np.sin(phase/2)**2; return 1/(1+F*s)
print(f"R={R:.5f} F={F:.5f}")
print(" d(nm) exact_R  refl(delta)  refl(delta+pi)[bubble_instanced]  trans(delta+pi)[wall.wgsl]")
for d in [0,10,30,50,100,150,200,300,500]:
    dl=4*np.pi*n*d/lam
    print(f"{d:5d}  {exact_R(d):.4f}   {airy_refl(dl):.4f}       {airy_refl(dl+np.pi):.4f}                         {airy_trans(dl+np.pi):.4f}")
ds=np.linspace(0,2000,4001)
w=[airy_trans(4*np.pi*n*d/lam+np.pi) for d in ds]
print("wall.wgsl value range over 0-2000nm:", round(min(w),4), round(max(w),4))
