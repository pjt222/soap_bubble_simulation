#!/usr/bin/env python3
"""Exact thin-film reflectance reference for checking the interference code.

Computes the reflectance of a free-standing film (air | film | air) from the
exact complex-amplitude slab formula and compares it with the Airy expression
the renderer uses, both with the correct geometric phase and with the extra
pi that issue #42 removed. Standard library only.

    python3 scripts/physics/thin_film_reference.py                 # table, 532 nm, normal incidence
    python3 scripts/physics/thin_film_reference.py --wavelength 450 --cos-theta 0.6
    python3 scripts/physics/thin_film_reference.py --check         # exit 1 unless Airy(no pi) == exact

Physics: with r12 = -r21 (Stokes), the slab amplitude is
    r = (r12 + r23 e^{i delta}) / (1 + r12 r23 e^{i delta}),  r23 = -r12,
    delta = 4 pi n d cos(theta_t) / lambda,
so |r|^2 = F sin^2(delta/2) / (1 + F sin^2(delta/2)) with F = 4R / (1 - R)^2.
The half-wave reflection flip is already inside r23 = -r12; adding pi to delta
double-counts it and inverts every fringe (a zero-thickness film turns bright).
Each polarisation (s, p) is evaluated separately and averaged afterwards.
"""

import argparse
import cmath
import math
import sys
from pathlib import Path


def require_project_root():
    if not (Path("Cargo.toml").is_file() and Path("src/render").is_dir()):
        sys.exit("error: run from the soap_bubble_simulation project root")


def cos_transmitted(cos_theta_incident, refractive_index):
    sin_theta_incident = math.sqrt(max(0.0, 1.0 - cos_theta_incident**2))
    sin_theta_transmitted = sin_theta_incident / refractive_index
    return math.sqrt(max(0.0, 1.0 - sin_theta_transmitted**2))


def interface_amplitudes(cos_theta_incident, cos_theta_film, refractive_index):
    """Fresnel amplitude reflection coefficients air -> film for s and p."""
    n_air, n_film = 1.0, refractive_index
    r_s = (n_air * cos_theta_incident - n_film * cos_theta_film) / (
        n_air * cos_theta_incident + n_film * cos_theta_film
    )
    r_p = (n_film * cos_theta_incident - n_air * cos_theta_film) / (
        n_film * cos_theta_incident + n_air * cos_theta_film
    )
    return r_s, r_p


def geometric_phase(thickness_nm, cos_theta_film, refractive_index, wavelength_nm):
    return 4.0 * math.pi * refractive_index * thickness_nm * cos_theta_film / wavelength_nm


def exact_reflectance(thickness_nm, cos_theta_incident, refractive_index, wavelength_nm):
    cos_theta_film = cos_transmitted(cos_theta_incident, refractive_index)
    delta = geometric_phase(thickness_nm, cos_theta_film, refractive_index, wavelength_nm)
    phasor = cmath.exp(1j * delta)
    total = 0.0
    for r12 in interface_amplitudes(cos_theta_incident, cos_theta_film, refractive_index):
        r23 = -r12
        amplitude = (r12 + r23 * phasor) / (1.0 + r12 * r23 * phasor)
        total += abs(amplitude) ** 2
    return total / 2.0


def airy_reflectance(phase, single_surface_reflectance):
    finesse = 4.0 * single_surface_reflectance / (1.0 - single_surface_reflectance) ** 2
    sin_squared = math.sin(phase / 2.0) ** 2
    return finesse * sin_squared / (1.0 + finesse * sin_squared)


def airy_per_polarisation(thickness_nm, cos_theta_incident, refractive_index, wavelength_nm, extra_phase):
    cos_theta_film = cos_transmitted(cos_theta_incident, refractive_index)
    delta = geometric_phase(thickness_nm, cos_theta_film, refractive_index, wavelength_nm) + extra_phase
    r_s, r_p = interface_amplitudes(cos_theta_incident, cos_theta_film, refractive_index)
    return 0.5 * (airy_reflectance(delta, r_s**2) + airy_reflectance(delta, r_p**2))


def main():
    require_project_root()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wavelength", type=float, default=532.0, help="nm (default 532)")
    parser.add_argument("--cos-theta", type=float, default=1.0, help="cosine of incidence angle in air")
    parser.add_argument("--n", type=float, default=1.33, help="film refractive index")
    parser.add_argument("--max-thickness", type=float, default=1000.0, help="nm")
    parser.add_argument("--step", type=float, default=25.0, help="nm between table rows")
    parser.add_argument("--check", action="store_true", help="fail unless Airy without pi matches exact to 1e-12")
    arguments = parser.parse_args()

    cos_theta_film = cos_transmitted(arguments.cos_theta, arguments.n)
    quarter_wave = arguments.wavelength / (4.0 * arguments.n * cos_theta_film)
    print(
        f"n={arguments.n}  lambda={arguments.wavelength} nm  cos(theta_i)={arguments.cos_theta}  "
        f"cos(theta_t)={cos_theta_film:.6f}  first maximum at d={quarter_wave:.2f} nm"
    )
    print(f"{'d (nm)':>8} {'exact':>10} {'airy':>10} {'airy+pi':>10}")
    worst_error = 0.0
    thickness = 0.0
    while thickness <= arguments.max_thickness + 1e-9:
        exact = exact_reflectance(thickness, arguments.cos_theta, arguments.n, arguments.wavelength)
        airy = airy_per_polarisation(thickness, arguments.cos_theta, arguments.n, arguments.wavelength, 0.0)
        airy_pi = airy_per_polarisation(thickness, arguments.cos_theta, arguments.n, arguments.wavelength, math.pi)
        worst_error = max(worst_error, abs(airy - exact))
        print(f"{thickness:8.1f} {exact:10.6f} {airy:10.6f} {airy_pi:10.6f}")
        thickness += arguments.step
    print(f"max |airy - exact| = {worst_error:.3e}")
    if arguments.check and worst_error > 1e-12:
        print("CHECK FAILED: Airy (geometric phase) disagrees with the exact slab formula")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
