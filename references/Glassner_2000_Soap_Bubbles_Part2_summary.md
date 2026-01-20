# Glassner_2000_Soap_Bubbles_Part2 - Extracted Summary

_Generated: Tue Jan 20 14:17:57 CET 2026_

---

## Part 1

# Glassner (2000) Soap Bubbles Part 2 - Extraction

## Key Concepts

### Wave Interference Fundamentals
- **Superposition**: When two waves occupy the same space, they add together
- **Constructive interference**: Waves in phase reinforce (amplitude doubles)
- **Destructive interference**: Waves out of phase cancel (amplitude → 0)
- **Phase**: Position along a wave cycle, measured in radians (full cycle = 2π)
- **Wavelength (λ)**: Distance between wave crests; visible light spans 380-780 nm

### Thin-Film Interference at Soap Films
Light striking a soap film splits into:
- **R**: Light reflected from outer surface
- **T₃**: Light that enters film, reflects off inner surface, exits parallel to R

These two beams interfere, creating colors based on their phase difference.

---

## Equations/Formulas

### Optical Path Difference
The geometric path difference through the film:
$$d = 2w\eta\cos\theta_t$$

Where:
- $w$ = film thickness
- $\eta$ = refractive index (~1.4 for soap)
- $\theta_t$ = transmitted angle (from Snell's Law)

### Effective Optical Path Length (EOPL)
Including the λ/2 phase shift at air→film interface:
$$\text{EOPL} = 2w\eta\cos\theta_t + \frac{\lambda}{2}$$

### Phase Shift (δ)
$$\delta = \frac{2\pi}{\lambda}\left(2w\eta\cos\theta_t + \frac{\lambda}{2}\right)$$

### Reflected Intensity
$$I_r = 4I_iR_f\sin^2\left(\frac{2\pi w\eta\cos\theta_t}{\lambda}\right)$$

Where:
- $I_i$ = incident intensity
- $R_f$ = Fresnel reflectivity

### Bubble Pair Geometry (Common Wall Radius)
When two bubbles of radii $r_A$ and $r_B$ merge ($r_A > r_B$):
$$\frac{1}{r_C} = \frac{1}{r_B} - \frac{1}{r_A}$$

Or equivalently:
$$r_C = \frac{r_A r_B}{r_A - r_B}$$

### Distance Between Bubble Centers
$$AB = \sqrt{r_A^2 + r_B^2 - r_A r_B}$$

### Distance to Common Wall Center
$$AC = \sqrt{r_A^2 + r_C^2 + r_A r_C}$$

---

## Relevance to Simulation

### Thin-Film Interference (Direct Application)
- The EOPL formula is the **core of interference color computation**
- Confirms the λ/2 phase shift occurs only when light enters denser medium (air→soap)
- Colors arise from wavelength-dependent reinforcement/cancellation
- For each wavelength, compute intensity separately, then convert spectrum to RGB

### Drainage Effects
- Gravity creates a **wedge-shaped thickness profile** (thicker at bottom)
- Very thin regions at top appear **black** (complete destructive interference)
- Thickness profile is **non-linear** (Figure 9b)

### Rendering Implementation
- Use sampled spectrum (81 values, 5nm intervals, 380-780nm)
- Fresnel reflectivity can be approximated by scaling with incident angle cosine
- Convert final spectrum to XYZ then RGB via CIE color-matching functions
- Higher-order bounces contribute negligibly (can ignore R₃, etc.)

### Bubble Cluster Geometry
- Triple junctions always meet at **120° angles**
- Common walls between bubbles are **spherical** (curved into larger bubble)
- CSG operations can construct clusters from sphere primitives

---

## Citations

### Books
- Boys, C.V. (1959). *Soap Bubbles: Their Colors and Forces Which Mold Them*. Dover.
- Isenberg, C. (1978). *The Science of Soap Films and Soap Bubbles*. Dover.
- Lovett, D. (1994). *Demonstrating Science with Soap Films*. Institute of Physics Publishing.

### Papers
- Smits & Meyer (1990). "Newton's Colors: Simulating Interference Phenomena in Realistic Image Synthesis". *Eurographics Workshop*.
- Dias, M.L. (1991). "Ray Tracing Interference Color". *IEEE CG&A*.
- Icart & Arquès (1999). "An Approach to Geometrical and Optical Simulation of Soap Froth". *Computers & Graphics*.
- Sun et al. (1999). "Deriving Spectra from Colors and Rendering Light Interference". *IEEE CG&A*.

### Rendering References
- Watt & Watt (1992). *Advanced Animation and Rendering Techniques*. Addison-Wesley.
- Glassner, A. (1995). *Principles of Digital Image Synthesis*. Morgan-Kaufmann.

---

## Part 2

# Glassner 2000 - Soap Bubbles Part 2, Chunk 002

## Key Concepts

- **Drainage modeling**: Liquid drains to bottom, modeled by "moving the air-filled sphere upward" — creates thickness gradient (thinner at top, thicker at bottom)
- **Realistic thickness variation**: Real bubbles are "liquids sloshing around in the air" — perfect spherical shells don't match reality
- **Noise-based thickness perturbation**: Adding noise to thickness computation simulates fluid dynamics without full simulation
- **Fresnel term**: Applied to interference calculations (shown in Figure 19)

## Thickness Values (from drainage example)

- Top of bubble: ~3,000 nm
- Bottom of bubble: ~7,000 nm

## Rendering Shortcut

Glassner notes a practical insight: since noise "scrambles" the careful interference calculations anyway, a **valid shortcut** is:
1. Take a gradient of bright colors
2. Swirl them with noise
3. Map onto the surface

This produces visually indistinguishable results from proper physical simulation, though a real fluid-dynamics simulation would be more accurate.

## Relevance to Project

| Aspect | Connection |
|--------|------------|
| **Drainage** | Confirms vertical thickness gradient approach; your `DrainageSimulator` models this |
| **Interference** | Fresnel term is essential for realism (you have Schlick approximation) |
| **Rendering** | Noise perturbation of thickness field adds realism without fluid sim complexity |
| **Optimization** | Suggests noise-swirled color gradients as fast approximation if needed |

## Implementation Notes

- Current project uses UV-based thickness variation in shader — aligns with Glassner's simplified approach
- Adding procedural noise to `ThicknessField` would improve realism
- The "shortcut" method could be useful for performance-critical scenarios

## Citation

Glassner, A. (2000). Soap Bubbles: Part 2. *IEEE Computer Graphics and Applications*, p. 109.

---

_Total parts: 2_
