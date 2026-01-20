# Schlick_1994_BRDF_Model - Extracted Summary

_Generated: Tue Jan 20 13:17:59 CET 2026_

---

## Part 1

# Schlick 1994: An Inexpensive BRDF Model for Physically-based Rendering

## Key Concepts

**BRDF (Bidirectional Reflectance Distribution Function)**: Describes how light interacts with a surface by relating incoming and outgoing radiances at a point $P$.

**Surface Types**:
- **Diffuse (Lambertian)**: Light reflected equally in every direction
- **Specular (Fresnel)**: Light reflected only around mirror direction

**Microfacet Theory**: Surfaces modeled as collections of small smooth planar elements (microfacets) with varying orientations.

**Key Properties**:
1. **Helmholtz Reciprocity**: $R_\lambda(P, V, V') = R_\lambda(P, V', V)$
2. **Energy Conservation**: $\int R_\lambda(P, V, V') (N \cdot V') \, dV' \leq 1$

## Equations/Formulas

**Reflected Radiance** (rendering equation):
$$L_\lambda(P, V) = \int_{V' \in \mathcal{V}} R_\lambda(P, V, V') \, L_\lambda(P, -V') \, (N \cdot V') \, dV'$$

**Cook-Torrance Model**:
$$R_\lambda(\alpha, \beta, \theta, \theta', \varphi) = \frac{d}{\pi} C_\lambda + \frac{s}{4\pi v v'} F_\lambda(\beta) \, G(\theta, \theta') \, D(\alpha, \varphi)$$

Where:
- $d, s \in [0,1]$: diffuse/specular reflector ratios ($d + s = 1$)
- $C_\lambda$: diffuse reflector color
- $F_\lambda(\beta)$: **Fresnel factor** (ratio of light reflected at wavelength $\lambda$)
- $G(\theta, \theta')$: geometrical attenuation (self-shadowing)
- $D(\alpha, \varphi)$: microfacet slope distribution

**Normalization Condition**:
$$\int_0^{\pi/2} \int_0^{2\pi} D(\alpha, \varphi) \cos\alpha \sin\alpha \, d\alpha \, d\varphi = \pi$$

For isotropic surfaces:
$$\int_0^{\pi/2} D(\alpha) \, 2\cos\alpha \sin\alpha \, d\alpha = 1$$

## Relevance to Soap Bubble Simulation

| Aspect | Relevance |
|--------|-----------|
| **Fresnel Factor** | Directly applicable—your simulation already uses Schlick approximation for thin-film interference |
| **Angle-dependent reflection** | Critical for accurate bubble appearance at grazing angles |
| **Energy conservation** | Ensures physically plausible rendering |
| **Specular vs diffuse** | Bubbles are primarily specular; model provides framework for balancing components |

**Direct Application**: The paper establishes the theoretical foundation for the Fresnel approximation you're using in `physics/interference.rs`. The cosines $v = (V \cdot N)$ and $v' = (V' \cdot N)$ map directly to your incident angle calculations.

## Notable Citations

- **[2]** Beckmann & Spizzichino: Electromagnetic wave scattering on rough surfaces
- **[5]** Cook & Torrance: Original microfacet BRDF model
- **[9]** He et al.: Comprehensive physical model (polarization, diffraction, interference)
- **[19]** Sparrow: Electromagnetic reflection theory
- **[15]** Shirley: Notes on linear combination weight issues

---

## Part 2

# Schlick 1994 BRDF Model - Chunk 2 Summary

## Key Concepts

### Rational Fraction Approximation (Section 4.1)
- **Method**: Approximate functions using rational fractions instead of Taylor expansions
- **Kernel conditions**: Intrinsic function characteristics (value at points, derivatives) used to derive coefficients
- Provides better accuracy than Taylor expansions over larger ranges

### Fresnel Factor (Section 4.2)
- $F_\lambda(u)$ expresses light reflection on microfacets where $u = \cos\theta$ (angle with normal)
- $n_\lambda$ = refractive index ratio, $k_\lambda$ = extinction coefficient

**Schlick's Fresnel Approximation** (Equation 15):
$$F_\lambda(u) = f_\lambda + (1 - f_\lambda)(1 - u)^5$$

Where $f_\lambda = F_\lambda(1)$ is the reflectance at normal incidence. This requires only **4 multiplications and 2 additions**.

### Geometrical Attenuation (Section 4.3)
- $G(v, v')$ models self-shadowing/masking on rough surfaces
- Cook-Torrance approximation (Equation 16): $G(t, u, v, v') = \min\left[1, 2\frac{tv}{u}, 2\frac{tv'}{u}\right]$

**Schlick's G approximation** (Equation 19):
$$G(v) = \frac{v}{v - kv + k} \quad \text{with} \quad k = \sqrt{\frac{2m^2}{\pi}}$$

Where $m$ = RMS slope of microfacets.

### Slope Distribution Function (Section 4.4)
**Beckmann distribution** (Equation 20):
$$D(t) = \frac{1}{m^2 t^4} e^{\frac{t^2-1}{m^2 t^2}}$$

**Schlick's D approximation** (Equation 21):
$$D(t) = \frac{m^3 x}{t(mx^2 - x^2 + m^2)^2} \quad \text{with} \quad x = t + m - 1$$

## Relevance to Soap Bubble Simulation

| Aspect | Application |
|--------|-------------|
| **Fresnel approximation** | Already used in your `interference.rs` - Schlick's $(1-u)^5$ term is the standard for real-time rendering |
| **Thin-film interference** | Fresnel reflectance at both air-film and film-air interfaces determines interference amplitude |
| **Performance** | Rational fraction approximations give ~20-32x speedup with <3% error - ideal for real-time GPU shaders |
| **Refractive index** | $f_\lambda$ depends on $n_\lambda$ which varies with wavelength (dispersion) - important for soap film color accuracy |

## Notable Citations
- **[5]** Cook & Torrance - Original microfacet BRDF model
- **[11]** Experimental $n_\lambda$, $k_\lambda$ values for materials
- **[14, 18, 19]** Geometrical attenuation formulations
- **[2, 3, 19]** Slope distribution functions (Beckmann)

---

## Part 3

# Schlick 1994 BRDF Model - Chunk 3 Summary

## Key Concepts

### Material Parameters
Materials characterized by parameter sets:
- **SINGLE material**: $(C_\lambda, r, p)$
- **DOUBLE material**: $(C_\lambda, r, p)$ and $(C'_\lambda, r', p')$ for layered surfaces

| Parameter | Range | Meaning |
|-----------|-------|---------|
| $C_\lambda$ | $[0,1]$ | Reflection factor at wavelength $\lambda$ |
| $r$ | $[0,1]$ | Roughness ($r=0$: specular, $r=1$: diffuse) |
| $p$ | $[0,1]$ | Isotropy ($p=0$: anisotropic, $p=1$: isotropic) |

### BRDF Formulation (Section 5.1)
Separates spectral and directional behavior:

**Single layer:**
$$R_\lambda(t, u, v, v', w) = S_\lambda(u) \cdot D(t, v, v', w)$$

**Double layer (for thin films):**
$$R_\lambda = S_\lambda(u) \cdot D + [1 - S_\lambda(u)] \cdot S'_\lambda(u) \cdot D'$$

## Core Equations

### Schlick Fresnel Approximation
$$S_\lambda(u) = C_\lambda + (1 - C_\lambda)(1 - u)^5$$

where $u = \cos\theta$ (angle from normal). This is the famous "Schlick approximation" widely used in real-time rendering.

### Directional Factor
$$D(t, v, v', w) = \frac{1}{4\pi v v'} Z(t) \cdot A(w)$$

With zenith factor:
$$Z(t) = \frac{r}{(1 + rt^2 - t^2)^2}$$

And azimuth factor:
$$A(w) = \sqrt{\frac{p}{p^2 - p^2w^2 + w^2}}$$

### Geometric Attenuation (Smith factor)
$$G(v) = \frac{v}{r - rv + v}$$

### Self-Shadowing with Fresnel
$$D(t, v, v', w) = \frac{G(v)G(v')}{4\pi vv'} Z(t) A(w) + \frac{1 - G(v)G(v')}{4\pi vv'}$$

## Relevance to Soap Bubble Simulation

### Direct Applications
1. **Fresnel reflection**: The Schlick approximation (Eq. 24) is already used in your shader - this confirms the physics basis
2. **Double-layer model**: Directly applicable to soap films with air-soap-air interfaces
3. **Wavelength-dependent $C_\lambda$**: Enables spectral rendering for thin-film interference

### Implementation Notes
- Your current code uses: `F = F0 + (1.0 - F0) * pow(1.0 - cos_theta, 5.0)` - this matches Eq. 24
- The double-layer formulation could model both surfaces of the soap film
- Roughness parameter $r$ could simulate surface micro-ripples from drainage

## Notable Citations

| Ref | Authors | Topic |
|-----|---------|-------|
| [5] | Cook, Torrance | Reflectance model (1981) |
| [8] | Hanrahan, Krueger | Subsurface scattering from layered surfaces |
| [9] | He, Torrance, Sillion, Greenberg | Comprehensive physical model |
| [11] | Palik | Handbook of Optical Constants |
| [19] | Torrance, Sparrow | Off-specular reflection theory |

---

_Total parts: 3_
