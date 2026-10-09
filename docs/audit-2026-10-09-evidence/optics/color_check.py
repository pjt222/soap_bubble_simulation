"""Color pipeline audit: replicate interference_lut.rs + bubble.wgsl LUT path, compare to
1 nm spectral reference (CIE 1931 2 deg CMFs x D65 x exact Airy slab reflectance).

Usage: python3 -I color_check.py <ciexyz31_1.csv> <Illuminantd65.csv>
"""
import sys
import csv
import numpy as np

N_FILM = 1.33
INTENSITY = 4.0  # BubbleUniform::default interference_intensity (pipeline.rs:77)

M_XYZ_TO_RGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                         [-0.9692660, 1.8760108, 0.0415560],
                         [0.0556434, -0.2040259, 1.0572252]])
M_RGB_TO_XYZ = np.linalg.inv(M_XYZ_TO_RGB)
D65_WHITE = M_RGB_TO_XYZ @ np.ones(3)  # (0.9505, 1.0, 1.089)


def load_table(path):
    rows = []
    with open(path, newline='') as fh:
        for rec in csv.reader(fh):
            if not rec or not rec[0].strip():
                continue
            rows.append([float(x) for x in rec])
    return np.array(rows)


def snell_cos_t(cos_i, n=N_FILM):
    sin_i = np.sqrt(np.maximum(0.0, 1.0 - cos_i**2))
    return np.sqrt(np.maximum(0.0, 1.0 - (sin_i / n) ** 2))


def fresnel_amp(cos_i, n=N_FILM):
    cos_t = snell_cos_t(cos_i, n)
    r_s = (cos_i - n * cos_t) / (cos_i + n * cos_t)
    r_p = (n * cos_i - cos_t) / (n * cos_i + cos_t)
    return r_s, r_p, cos_t


def exact_R(d, lam, cos_i, n=N_FILM):
    r_s, r_p, cos_t = fresnel_amp(cos_i, n)
    e = np.exp(1j * 4.0 * np.pi * n * d * cos_t / lam)
    acc = 0.0
    for r in (r_s, r_p):
        acc = acc + np.abs(r * (1 - e) / (1 - r * r * e)) ** 2
    return 0.5 * acc


def code_R(d, lam, cos_i, n=N_FILM, add_pi=True):
    r_s, r_p, cos_t = fresnel_amp(cos_i, n)
    R = 0.5 * (r_s**2 + r_p**2)
    F = 4 * R / max(1 - R, 0.001) ** 2
    phase = 2 * np.pi * (2 * n * d * cos_t) / lam + (np.pi if add_pi else 0.0)
    s2 = np.sin(0.5 * phase) ** 2
    return F * s2 / np.maximum(1 + F * s2, 0.001)


def g(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


def cmf_code(lam):
    """interference_lut.rs:136-145 (single-sigma Gaussians)."""
    x = 1.056 * g(lam, 599.8, 37.9) + 0.362 * g(lam, 442.0, 16.0) - 0.065 * g(lam, 501.1, 20.4)
    y = 0.821 * g(lam, 568.8, 46.9) + 0.286 * g(lam, 530.9, 31.1)
    z = 1.217 * g(lam, 437.0, 11.8) + 0.681 * g(lam, 459.0, 26.0)
    return np.array([max(x, 0), max(y, 0), max(z, 0)])


def g2(x, mu, s1, s2):
    return np.exp(-0.5 * ((x - mu) / (s1 if x < mu else s2)) ** 2)


def cmf_wyman(lam):
    """Wyman, Sloan, Shirley 2013 multi-lobe piecewise Gaussian (JCGT 2(2))."""
    x = 1.056 * g2(lam, 599.8, 37.9, 31.0) + 0.362 * g2(lam, 442.0, 16.0, 26.7) - 0.065 * g2(lam, 501.1, 20.4, 26.2)
    y = 0.821 * g2(lam, 568.8, 46.9, 40.5) + 0.286 * g2(lam, 530.9, 16.3, 31.1)
    z = 1.217 * g2(lam, 437.0, 11.8, 36.0) + 0.681 * g2(lam, 459.0, 26.0, 13.8)
    return np.array([x, y, z])


WL7 = [400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0]


def code_lut_rgb(d, cos_i, add_pi=True, quantize=True, cmf=cmf_code):
    """interference_lut.rs:45-93 then bubble.wgsl:425-439."""
    xyz = np.zeros(3)
    for lam in WL7:
        xyz += cmf(lam) * code_R(d, lam, cos_i, add_pi=add_pi)
    xyz /= 7.0
    rgb = M_XYZ_TO_RGB @ xyz
    if quantize:
        rgb = np.floor(np.clip(rgb, 0, 1) * 255.0) / 255.0  # `as u8` truncates
    rgb = rgb * INTENSITY
    return np.clip(rgb, 0, 1), xyz


def reference_rgb(d, cos_i, tab, d65, exposure):
    lam = tab[:, 0]
    R = exact_R(d, lam, cos_i)
    w = d65 * 1.0
    xyz = (tab[:, 1:4] * (w * R)[:, None]).sum(0) / (tab[:, 2] * w).sum()  # Y_white = 1
    xyz *= exposure
    rgb = M_XYZ_TO_RGB @ xyz
    return np.clip(rgb, 0, 1), xyz


def srgb_encode(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def lab(rgb_lin):
    xyz = M_RGB_TO_XYZ @ rgb_lin
    t = xyz / D65_WHITE
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def de76(a, b):
    return float(np.linalg.norm(lab(a) - lab(b)))


def to8(rgb_lin):
    return tuple(int(round(v)) for v in srgb_encode(rgb_lin) * 255)


def main():
    cie = load_table(sys.argv[1])
    d65t = load_table(sys.argv[2])
    lam = cie[:, 0]
    d65 = np.interp(lam, d65t[:, 0], d65t[:, 1])
    mask = (lam >= 380) & (lam <= 780)
    cie = cie[mask]
    d65 = d65[mask]

    # exposure: the code maps a flat reflectance R to Y = INTENSITY * mean(ybar_fit(WL7)) * R
    y_code_white = INTENSITY * np.mean([cmf_code(l)[1] for l in WL7])
    print(f"Code effective exposure: flat R -> Y = {y_code_white:.4f} * R")

    print("\n[A] CMF fit used by code vs CIE 1931 table at the 7 sample wavelengths")
    for l in WL7:
        row = cie[cie[:, 0] == l][0, 1:4]
        c = cmf_code(l)
        w = cmf_wyman(l)
        print(f"  {l:.0f}: table=({row[0]:.4f},{row[1]:.4f},{row[2]:.4f}) code=({c[0]:.4f},{c[1]:.4f},{c[2]:.4f}) "
              f"wyman2013=({w[0]:.4f},{w[1]:.4f},{w[2]:.4f})")

    print("\n[B] White point: flat reflectance R=1 through code path (no quantization, before x4)")
    xyz = sum(cmf_code(l) for l in WL7) / 7
    rgb = M_XYZ_TO_RGB @ xyz
    print(f"  code CMFs: XYZ={np.round(xyz,4)} linear sRGB={np.round(rgb,4)} normalized={np.round(rgb/rgb.max(),3)}")
    xyz_t = sum(cie[cie[:, 0] == l][0, 1:4] for l in WL7) / 7
    rgb_t = M_XYZ_TO_RGB @ xyz_t
    print(f"  table CMFs, equal-energy: XYZ={np.round(xyz_t,4)} linear sRGB normalized={np.round(rgb_t/rgb_t.max(),3)}")
    xyzw = (cie[:, 1:4] * d65[:, None]).sum(0) / (cie[:, 2] * d65).sum()
    print(f"  1nm D65 reference: XYZ={np.round(xyzw,4)} linear sRGB={np.round(M_XYZ_TO_RGB @ xyzw,4)}")

    print("\n[C] LUT dynamic range (intensity=1 as generated): max channel over d in [0,2000], cos in [0,1]")
    mx = 0.0
    mx_normal = 0.0
    for ci in np.linspace(0, 1, 64):
        for dd in np.linspace(0, 2000, 256):
            _, xyz = code_lut_rgb(dd, ci, quantize=False)
            v = (M_XYZ_TO_RGB @ xyz).max()
            mx = max(mx, v)
            if ci == 1.0:
                mx_normal = max(mx_normal, v)
    print(f"  max linear value stored: all angles={mx:.4f} ({int(mx*255)} of 255 levels); "
          f"normal incidence row={mx_normal:.4f} ({int(mx_normal*255)} levels)")
    print(f"  after x{INTENSITY:.0f}: quantization step = {INTENSITY/255:.4f} linear; first non-zero level "
          f"in sRGB8 = {to8(np.array([INTENSITY/255]*3))[0]}")

    print("\n[D] Normal-incidence color vs thickness (sRGB8 as displayed, interference term only; no +0.1 base, no alpha)")
    print("     d    ref(1nm,D65,exact)   code-as-is      dE76   code-pi-fixed    dE76   fixed+tableCMF+1nm dE")
    for dd in [5, 20, 29.9, 30, 50, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 1000, 1200, 1500]:
        ref, _ = reference_rgb(dd, 1.0, cie, d65, y_code_white)
        a, _ = code_lut_rgb(dd, 1.0, add_pi=True)
        if dd < 30.0:
            a = np.array([0.02, 0.02, 0.02]) - 0.1 * np.array([0.95, 0.97, 1.0])  # hard black branch, remove base later
            a = np.clip(a + 0.1 * np.array([0.95, 0.97, 1.0]), 0, 1)  # = 0.02 grey (bubble.wgsl:518)
        b, _ = code_lut_rgb(dd, 1.0, add_pi=False)
        print(f"  {dd:6.1f}  {str(to8(ref)):>18}  {str(to8(a)):>16}  {de76(a, ref):6.1f}  {str(to8(b)):>16}  {de76(b, ref):6.1f}")

    print("\n[E] Isolate spectral sampling (aliasing) error: correct phase, TABLE CMFs, D65; 7-sample vs 1 nm")
    for dd in [100, 300, 500, 800, 1000, 1200, 1500, 2000]:
        ref, _ = reference_rgb(dd, 1.0, cie, d65, y_code_white)
        xyz = np.zeros(3)
        wsum = 0.0
        for l in WL7:
            row = cie[cie[:, 0] == l][0, 1:4]
            dw = d65[cie[:, 0] == l][0]
            xyz += row * dw * exact_R(dd, l, 1.0)
            wsum += row[1] * dw
        xyz = xyz / wsum * y_code_white
        s7 = np.clip(M_XYZ_TO_RGB @ xyz, 0, 1)
        print(f"  d={dd:5d}: ref={to8(ref)}  7-sample={to8(s7)}  dE76={de76(s7, ref):5.1f}")

    print("\n[F] Isolate CMF-fit + equal-energy error: correct phase, no quantization, 7 samples; code CMF/E vs table CMF/D65")
    for dd in [100, 200, 300, 400, 500, 700]:
        a, _ = code_lut_rgb(dd, 1.0, add_pi=False, quantize=False)
        xyz = np.zeros(3)
        wsum = 0.0
        for l in WL7:
            row = cie[cie[:, 0] == l][0, 1:4]
            dw = d65[cie[:, 0] == l][0]
            xyz += row * dw * exact_R(dd, l, 1.0)
            wsum += row[1] * dw
        xyz = xyz / wsum * y_code_white
        s7 = np.clip(M_XYZ_TO_RGB @ xyz, 0, 1)
        print(f"  d={dd:5d}: codeCMF/E={to8(a)}  tableCMF/D65={to8(s7)}  dE76={de76(a, s7):5.1f}")

    print("\n[G] CPU InterferenceCalculator (3 wavelengths, sRGB-encoded, no exposure) at normal incidence")
    for dd in [100, 300, 500]:
        vals = []
        for l in (650.0, 532.0, 450.0):
            r_s, r_p, cos_t = fresnel_amp(1.0)
            R = 0.5 * (r_s**2 + r_p**2)
            phi = 2 * np.pi * (2 * N_FILM * dd + l / 2) / l
            vals.append(2 * R * (1 - np.cos(phi)) / (1 + R**2 - 2 * R * np.cos(phi)))
        vals = np.array(vals)
        print(f"  d={dd}: linear={np.round(vals,4)} -> encoded={np.round(srgb_encode(vals),3)}; "
              f"GPU LUT (pre x4, linear)={np.round(code_lut_rgb(dd,1.0,quantize=False)[0]/INTENSITY,4)}")


if __name__ == '__main__':
    main()
