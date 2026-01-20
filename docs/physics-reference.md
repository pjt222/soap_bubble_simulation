# Physics Reference

Mathematical and physical foundations for the soap bubble simulation.

## Thin-Film Interference

### Optical Path Difference

When light hits a thin film (thickness `d`, refractive index `n`), it partially reflects at both surfaces. The optical path difference between these reflections determines interference:

```
δ = 2 n d cos(θ_t) + λ/2
```

| Symbol | Description | Typical Value |
|--------|-------------|---------------|
| `δ` | Optical path difference | varies |
| `n` | Refractive index of film | 1.33 (soap) |
| `d` | Film thickness | 100-1000 nm |
| `θ_t` | Transmission angle in film | from Snell's law |
| `λ/2` | Phase shift at denser interface | half wavelength |

### Interference Conditions

**Constructive interference** (bright):
```
δ = m λ,  where m = 1, 2, 3, ...
```

**Destructive interference** (dark):
```
δ = (m + 1/2) λ
```

### Snell's Law

Relates incident and transmitted angles:

```
n₁ sin(θ₁) = n₂ sin(θ₂)
```

For air (n₁ = 1) to soap film (n₂ ≈ 1.33):
```
sin(θ_t) = sin(θ_i) / n_film
cos(θ_t) = √(1 - sin²(θ_t))
```

### Fresnel Equations

The fraction of light reflected depends on the angle and polarization. For unpolarized light, the reflectance is approximated by Schlick's formula:

```
R(θ) = R₀ + (1 - R₀)(1 - cos(θ))⁵
```

Where:
```
R₀ = ((n₁ - n₂) / (n₁ + n₂))²
```

For soap film (n = 1.33):
```
R₀ = ((1 - 1.33) / (1 + 1.33))² ≈ 0.02 (2%)
```

---

## Soap Film Geometry

### Young-Laplace Equation

Pressure difference across a curved interface:

```
ΔP = γ (1/R₁ + 1/R₂)
```

For a spherical soap bubble with two surfaces:
```
ΔP = 4γ/R
```

| Symbol | Description | Typical Value |
|--------|-------------|---------------|
| `ΔP` | Pressure difference | ~10 Pa |
| `γ` | Surface tension | 0.025 N/m |
| `R` | Bubble radius | 0.025 m (5cm diameter) |

### Film Thickness Distribution

Under gravity, the film drains from top to bottom. Simplified model:

```
d(θ) = d₀ (1 - α(1 - cos(θ)))
```

Where:
- `d₀` = base thickness at equator
- `θ` = polar angle (0 at top, π at bottom)
- `α` = drainage factor (typically 0.2-0.4)

---

## Drainage Dynamics

### Drainage Equation

The full drainage equation for a thin film under gravity:

```
∂d/∂t = -ρgd³/(3η) + D∇²d + Marangoni terms
```

| Symbol | Description | Typical Value |
|--------|-------------|---------------|
| `ρ` | Fluid density | 1000 kg/m³ |
| `g` | Gravitational acceleration | 9.81 m/s² |
| `η` | Dynamic viscosity | 0.001 Pa·s |
| `D` | Diffusion coefficient | 10⁻⁹ m²/s |

### Simplified Drainage Model (Used in Simulation)

For real-time rendering, we use a simplified steady-state model based on the normal vector:

```
d(n) = d_base * (1 - drainage_factor * (1 - n_y) / 2)
```

This gives:
- Maximum thickness at bottom (n_y = -1)
- Minimum thickness at top (n_y = +1)

---

## Color Calculation

### RGB Wavelengths

| Color | Wavelength | Frequency |
|-------|------------|-----------|
| Red | 650 nm | 461 THz |
| Green | 532 nm | 564 THz |
| Blue | 450 nm | 666 THz |

### Interference Intensity

For each color channel:

```
I = (1 + cos(φ)) / 2
```

Where phase:
```
φ = 2π * 2nd*cos(θ_t) / λ + π
```

The `+π` accounts for phase shift at the first (outer) interface.

### Color Mapping

Final color calculation:
```
color.rgb = base_color * 0.1 + interference_rgb * fresnel * intensity_scale
color.a = base_alpha + edge_alpha * (1 - cos(θ))
```

---

## Physical Constants

| Constant | Value | Description |
|----------|-------|-------------|
| n_soap | 1.33 | Refractive index of soap solution |
| n_air | 1.0 | Refractive index of air |
| γ_soap | 0.025-0.030 N/m | Surface tension of soap solution |
| η_soap | 0.001-0.002 Pa·s | Viscosity of soap solution |
| ρ_soap | 1000-1020 kg/m³ | Density of soap solution |

---

## References

1. Born, M. & Wolf, E. (1999). *Principles of Optics* (7th ed.). Cambridge University Press.

2. Hecht, E. (2017). *Optics* (5th ed.). Pearson.

3. de Gennes, P.-G., Brochard-Wyart, F., & Quéré, D. (2004). *Capillarity and Wetting Phenomena*. Springer.

4. Isenberg, C. (1992). *The Science of Soap Films and Soap Bubbles*. Dover Publications.

5. Schlick, C. (1994). "An Inexpensive BRDF Model for Physically-based Rendering." *Computer Graphics Forum*, 13(3), 233-246.
