# Born_Wolf_1999_Principles_of_Optics - Extracted Summary

_Generated: Tue Jan 21 2026_

---

## Born & Wolf (1999) - Principles of Optics, 7th Edition

**Citation:** Born, M., & Wolf, E. (1999). *Principles of Optics: Electromagnetic Theory of Propagation, Interference and Diffraction of Light* (7th expanded ed.). Cambridge University Press.

**ISBN:** 978-0-521-64222-4 (7th ed.), 978-1-108-47743-7 (60th Anniversary ed.)

**Pages:** 992 pages, 393 illustrations, 30 tables

## Overview

The definitive reference on classical optics, first published 1959. The 7th edition (1999) includes updates for laser optics while maintaining comprehensive coverage of electromagnetic wave theory.

**Authors:**
- Max Born (Georg-August-Universität Göttingen, University of Edinburgh)
- Emil Wolf (University of Rochester)

**Contributors:** A. B. Bhatia, P. C. Clemmow, D. Gabor, A. R. Stokes, A. M. Taylor, P. A. Wayman, W. L. Wilcock

## Relevant Chapters for Thin-Film Simulation

### Chapter 1: Basic Properties of the Electromagnetic Field
- Maxwell's equations
- Energy and momentum of light
- Scalar wave approximation

### Chapter 7: Interference and Interferometers

#### 7.5: Two-Beam Interference
- **7.5.1**: Fringes with plane parallel plate
- **7.5.2**: Fringes with thin films; Fizeau interferometer

#### 7.6: Multiple-Beam Interference
- **7.6.1**: Multiple-beam fringes with plane parallel plate
- **7.6.6**: Interference filters
- **7.6.7**: Multiple-beam fringes with thin films
- **7.6.8**: Multiple-beam fringes with two parallel plates

### Chapter 13: Optics of Metals
- Complex refractive index
- Reflection from metallic surfaces
- Thin metallic films

### Chapter 14: Optics of Crystals
- Anisotropic media
- Birefringence effects

## Key Equations for Thin-Film Interference

### Optical Path Difference (Two-Beam)
$$\Delta = 2nd\cos\theta_t + \frac{\lambda}{2}$$

where:
- $n$ = refractive index of film
- $d$ = film thickness
- $\theta_t$ = angle of refraction
- $\lambda/2$ = phase shift at reflection (if applicable)

### Constructive Interference
$$2nd\cos\theta_t = m\lambda \quad (m = 0, 1, 2, ...)$$

### Destructive Interference
$$2nd\cos\theta_t = (m + \frac{1}{2})\lambda$$

### Fresnel Reflection Coefficients

**s-polarization (TE):**
$$r_s = \frac{n_1\cos\theta_i - n_2\cos\theta_t}{n_1\cos\theta_i + n_2\cos\theta_t}$$

**p-polarization (TM):**
$$r_p = \frac{n_2\cos\theta_i - n_1\cos\theta_t}{n_2\cos\theta_i + n_1\cos\theta_t}$$

### Airy Function (Multiple-Beam)
$$I = \frac{I_0}{1 + F\sin^2(\delta/2)}$$

where $F = 4R/(1-R)^2$ is the coefficient of finesse.

### Finesse
$$\mathcal{F} = \frac{\pi\sqrt{R}}{1-R}$$

## Relevance to Simulation

| Topic | Application |
|-------|-------------|
| **Two-beam interference** | Basic thin-film color calculation |
| **Multiple-beam** | Accurate intensity for thick films |
| **Fresnel equations** | Reflection/transmission coefficients |
| **Airy function** | Spectral response of Fabry-Perot |
| **Coherence** | When interference is observable |
| **Polarization** | Angle-dependent reflection |

## Key Physical Concepts

### Coherence Length
$$l_c = \frac{\lambda^2}{\Delta\lambda}$$

Films thicker than coherence length show no interference.

### Stokes Relations
Phase relationships at interfaces:
$$r_{12} = -r_{21}, \quad t_{12}t_{21} = 1 - r_{12}^2$$

### Newton's Rings
Classical thin-film interference demonstration:
$$r_m = \sqrt{m\lambda R}$$

## Implementation Notes for Soap Bubbles

1. **Soap film**: Two air-film interfaces → two reflections
2. **Phase shift**: Reflection from higher-n medium adds $\lambda/2$
3. **Thickness variation**: Non-uniform film → color patterns
4. **Viewing angle**: Colors shift with observation angle
5. **White light**: Sum over visible spectrum for color

## Citations

- Fabry, C., & Perot, A. (1899). Théorie et applications d'une nouvelle méthode de spectroscopie interférentielle.
- Airy, G. B. (1833). On the phænomena of Newton's rings.
- Michelson, A. A. (1927). *Studies in Optics*.

---

_Source: Cambridge University Press, Library of Congress catalog, Wikipedia_
