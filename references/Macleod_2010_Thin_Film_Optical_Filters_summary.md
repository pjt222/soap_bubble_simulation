# Macleod_2010_Thin_Film_Optical_Filters - Extracted Summary

_Generated: Tue Jan 21 2026_

---

## Macleod (2010) - Thin-Film Optical Filters, 4th Edition

**Citation:** Macleod, H. A. (2010). *Thin-Film Optical Filters* (4th ed.). CRC Press (Taylor & Francis). Series in Optics and Optoelectronics.

**ISBN:** 978-1-4200-7302-7

**Pages:** 800 pages

**Author:** H. Angus Macleod, President of Thin Film Center Inc., Tucson, Arizona; Professor Emeritus of Optical Sciences, University of Arizona

## Overview

Authoritative reference on design, manufacture, and application of thin-film optical coatings. Covers theory through practical implementation with sufficient mathematics for calculations.

**Review:** "An indispensable text for every filter manufacturer and user and an excellent guide for students." —Mario Bertolotti, *Contemporary Physics*

## Table of Contents (Key Chapters)

### Fundamentals
1. **Basic Theory** - Wave optics of thin films
2. **Antireflection Coatings** - Single and multilayer AR
3. **Neutral Mirrors and Beam Splitters**
4. **Multilayer High-Reflectance Coatings**

### Advanced Topics
5. **Edge Filters** - Longwave/shortwave pass
6. **Band-Pass Filters** - Narrow and broad
7. **Tilted Coatings** - Polarization effects
8. **Color** (NEW in 4th ed.) - Color theory in coatings
9. **Gain in Optical Coatings** (NEW in 4th ed.)
10. **Production and Testing**

## Key Concepts

### Characteristic Matrix Method
For a single layer:
$$M = \begin{pmatrix} \cos\delta & \frac{i\sin\delta}{\eta} \\ i\eta\sin\delta & \cos\delta \end{pmatrix}$$

where:
- $\delta = \frac{2\pi n d \cos\theta}{\lambda}$ (phase thickness)
- $\eta = n$ for s-polarization, $\eta = n/\cos\theta$ for p-polarization

### Multilayer Stack
$$M_{total} = M_1 \cdot M_2 \cdot ... \cdot M_n$$

Reflectance:
$$R = \left|\frac{(M_{11} + M_{12}\eta_s)\eta_0 - (M_{21} + M_{22}\eta_s)}{(M_{11} + M_{12}\eta_s)\eta_0 + (M_{21} + M_{22}\eta_s)}\right|^2$$

### Quarter-Wave Stack
High reflector: alternating high/low index layers
$$R = \left(\frac{n_0(n_L/n_H)^{2N} - n_s}{n_0(n_L/n_H)^{2N} + n_s}\right)^2$$

### Fabry-Perot Interferometer
Finesse:
$$\mathcal{F} = \frac{\Delta\lambda_{FSR}}{\Delta\lambda_{FWHM}} = \frac{\pi\sqrt{R}}{1-R}$$

## Filter Types

| Type | Structure | Application |
|------|-----------|-------------|
| **Antireflection** | Single/multilayer | Reduce surface reflections |
| **High reflector** | Quarter-wave stack | Mirrors, laser cavities |
| **Edge filter** | Multilayer | Cut-off wavelengths |
| **Band-pass** | Fabry-Perot cavity | Select narrow band |
| **Neutral density** | Metal/dielectric | Attenuate uniformly |
| **Dichroic** | Multilayer | Wavelength-selective mirrors |

## Relevance to Simulation

| Topic | Application |
|-------|-------------|
| **Matrix method** | Exact multilayer calculation |
| **Interference theory** | Understanding color physics |
| **Angle dependence** | View-angle color shift |
| **Material properties** | Refractive indices |
| **Color chapter** | CIE color space, perception |
| **Dispersion** | Wavelength-dependent n(λ) |

### Soap Film as Optical Filter

A soap film is essentially a simple thin-film interference system:
1. **Single dielectric layer** (water + soap)
2. **Surrounded by air** (n = 1.0)
3. **Film index** ≈ 1.33 (water-like)
4. **Variable thickness** → variable phase → colors

### Key Differences from Engineered Filters

| Engineered Filter | Soap Film |
|-------------------|-----------|
| Precise thickness control | Continuous drainage |
| Uniform layers | Thickness gradient |
| Stable | Dynamic, evolving |
| Designed spectrum | Natural interference |

## Equations for Soap Film Implementation

### Simple Soap Film (air-film-air)
$$R = \frac{2r^2(1 - \cos\delta)}{1 + r^4 - 2r^2\cos\delta}$$

where $r$ is Fresnel coefficient at air-film interface.

### Phase Thickness
$$\delta = \frac{4\pi n_{film} d \cos\theta_t}{\lambda}$$

### Color Calculation
Integrate over visible spectrum:
$$X = \int_\lambda R(\lambda) \bar{x}(\lambda) I(\lambda) d\lambda$$

(Similar for Y, Z tristimulus values)

## Citations

- Heavens, O. S. (1955). *Optical Properties of Thin Solid Films*.
- Baumeister, P. (2004). *Optical Coating Technology*.
- Rancourt, J. D. (1996). *Optical Thin Films: User Handbook*.
- Born, M., & Wolf, E. (1999). *Principles of Optics*.

---

_Source: Taylor & Francis, Amazon, Contemporary Physics review_
