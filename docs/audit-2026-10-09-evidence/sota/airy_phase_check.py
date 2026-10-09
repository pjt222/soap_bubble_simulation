import cmath, math
n_film = 1.33
def exact_reflectance(d_nm, lam_nm, cos_i=1.0):
    # Born & Wolf thin film: air|film|air, average s/p
    sin_i = math.sqrt(max(0.0, 1 - cos_i**2)); sin_t = sin_i / n_film; cos_t = math.sqrt(1 - sin_t**2)
    out = 0.0
    for pol in ("s", "p"):
        if pol == "s":
            r12 = (cos_i - n_film*cos_t) / (cos_i + n_film*cos_t)
        else:
            r12 = (n_film*cos_i - cos_t) / (n_film*cos_i + cos_t)
        r23 = -r12
        delta = 4*math.pi*n_film*d_nm*cos_t/lam_nm
        r = (r12 + r23*cmath.exp(1j*delta)) / (1 + r12*r23*cmath.exp(1j*delta))
        out += abs(r)**2 / 2
    return out
def code_reflectance(d_nm, lam_nm, cos_i=1.0, add_pi=True):
    # src/render/interference_lut.rs:63-71 and 112-122
    sin_i = math.sqrt(max(0.0, 1 - cos_i**2)); sin_t = sin_i / n_film; cos_t = math.sqrt(1 - sin_t**2)
    rs = (cos_i - n_film*cos_t)/(cos_i + n_film*cos_t); rp = (n_film*cos_i - cos_t)/(n_film*cos_i + cos_t)
    R = (rs*rs + rp*rp)/2
    phase = 2*math.pi*(2*n_film*d_nm*cos_t)/lam_nm + (math.pi if add_pi else 0.0)
    F = 4*R/(1-R)**2
    s2 = math.sin(phase/2)**2
    return F*s2/(1+F*s2)
print("d_nm  exact(550)  code(+pi)  code(no pi)")
for d in (0, 10, 25, 50, 103.4, 150, 206.8, 300, 500):
    print(f"{d:6.1f}  {exact_reflectance(d,550):.5f}    {code_reflectance(d,550):.5f}    {code_reflectance(d,550,add_pi=False):.5f}")
