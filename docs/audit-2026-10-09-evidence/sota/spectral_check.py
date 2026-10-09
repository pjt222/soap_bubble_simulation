import math
n_film = 1.33
def gauss_sym(x, mu, sigma):
    return math.exp(-0.5*((x-mu)/sigma)**2)
def cie_code(lam):  # src/render/interference_lut.rs:113-120 (symmetric sigmas)
    x = 1.056*gauss_sym(lam,599.8,37.9) + 0.362*gauss_sym(lam,442.0,16.0) - 0.065*gauss_sym(lam,501.1,20.4)
    y = 0.821*gauss_sym(lam,568.8,46.9) + 0.286*gauss_sym(lam,530.9,31.1)
    z = 1.217*gauss_sym(lam,437.0,11.8) + 0.681*gauss_sym(lam,459.0,26.0)
    return (x,y,z)
def lobe(lam, mu, inv_left, inv_right):
    t = (lam-mu)*(inv_left if lam<mu else inv_right)
    return math.exp(-0.5*t*t)
def cie_wyman(lam):  # Wyman, Sloan, Shirley 2013 JCGT 2(2) Listing 1
    x = 0.362*lobe(lam,442.0,0.0624,0.0374) + 1.056*lobe(lam,599.8,0.0264,0.0323) - 0.065*lobe(lam,501.1,0.0490,0.0382)
    y = 0.821*lobe(lam,568.8,0.0213,0.0247) + 0.286*lobe(lam,530.9,0.0613,0.0322)
    z = 1.217*lobe(lam,437.0,0.0845,0.0278) + 0.681*lobe(lam,459.0,0.0385,0.0725)
    return (x,y,z)
def reflectance(d, lam, add_pi):
    R = ((1-n_film)/(1+n_film))**2  # normal incidence
    F = 4*R/(1-R)**2
    phase = 4*math.pi*n_film*d/lam + (math.pi if add_pi else 0.0)
    s2 = math.sin(phase/2)**2
    return F*s2/(1+F*s2)
def xyz_of(d, wavelengths, cie, add_pi):
    X=Y=Z=0.0
    for lam in wavelengths:
        r = reflectance(d, lam, add_pi); c = cie(lam)
        X += c[0]*r; Y += c[1]*r; Z += c[2]*r
    return X,Y,Z
def chroma(xyz):
    s = sum(xyz)
    return (xyz[0]/s, xyz[1]/s) if s > 1e-12 else (1/3,1/3)
seven = [400.0,450.0,500.0,550.0,600.0,650.0,700.0]
dense = [380.0 + i for i in range(401)]
print("CIE fit at sample wavelengths (code vs Wyman):")
for lam in seven:
    a=cie_code(lam); b=cie_wyman(lam)
    print(f"  {lam:.0f}: code=({a[0]:.3f},{a[1]:.3f},{a[2]:.3f}) wyman=({b[0]:.3f},{b[1]:.3f},{b[2]:.3f})")
# white point of equal-energy spectrum through each 7-sample CIE
for name, cie in (("code", cie_code), ("wyman", cie_wyman)):
    X=sum(cie(l)[0] for l in seven); Y=sum(cie(l)[1] for l in seven); Z=sum(cie(l)[2] for l in seven)
    print(f"equal-energy white chromaticity via 7 samples, {name}: x={X/(X+Y+Z):.4f} y={Y/(X+Y+Z):.4f}  (ideal E: 0.3333,0.3333)")
X=sum(cie_wyman(l)[0] for l in dense); Y=sum(cie_wyman(l)[1] for l in dense); Z=sum(cie_wyman(l)[2] for l in dense)
print(f"equal-energy white via dense 1nm Wyman: x={X/(X+Y+Z):.4f} y={Y/(X+Y+Z):.4f}")
# chromaticity error bands
def band_errors(lo, hi, wl, cie, add_pi):
    errs=[]
    for i in range(lo, hi, 5):
        ref = chroma(xyz_of(i, dense, cie_wyman, False))
        test = chroma(xyz_of(i, wl, cie, add_pi))
        errs.append(math.hypot(ref[0]-test[0], ref[1]-test[1]))
    return sum(errs)/len(errs), max(errs)
print("mean/max chromaticity error |dxy| vs dense-1nm Wyman reference (no pi):")
for lo, hi in ((30,600),(600,1200),(1200,2000)):
    a = band_errors(lo,hi,seven,cie_code,True)
    b = band_errors(lo,hi,seven,cie_code,False)
    c = band_errors(lo,hi,seven,cie_wyman,False)
    print(f"  d in [{lo},{hi}) nm: as-coded(7,sym,+pi) mean={a[0]:.3f} max={a[1]:.3f} | 7,sym,no-pi mean={b[0]:.3f} max={b[1]:.3f} | 7,wyman,no-pi mean={c[0]:.3f} max={c[1]:.3f}")
# Nyquist: phase step between adjacent samples in wavenumber
for lam1, lam2 in ((400,450),(650,700)):
    dk = 1/lam1 - 1/lam2
    print(f"aliasing onset (phase step between {lam1}/{lam2} nm exceeds pi): d > {1/(4*n_film*dk):.0f} nm")
