# Huang_2020_Chemomechanical_soap_film - Extracted Summary

_Generated: Tue Jan 20 14:15:49 CET 2026_

---

## Part 1

# Huang et al. 2020 - Chemomechanical Simulation of Soap Film Flow on Spherical Bubbles

**Citation**: Huang, W., Iseringhausen, J., Kneiphof, T., Qu, Z., Jiang, C., & Hullin, M. B. (2020). Chemomechanical Simulation of Soap Film Flow on Spherical Bubbles. *ACM Trans. Graph.*, 39(4). https://doi.org/10.1145/3386569.3392094

---

## Key Concepts

- **Chemomechanical coupling**: Surface tension depends on surfactant (soap) concentration via Marangoni effect
- **Lubrication theory**: Reduces 3D Navier-Stokes to 2D flow on sphere surface; thickness becomes a variable, not a dimension
- **Three-layer film structure**: Two water-air interfaces with soap molecules + thin bulk fluid layer (~1μm thick)
- **Surfactant behavior**: Soap molecules concentrate at surfaces (hydrophobic ends avoid water), reducing surface tension
- **Compressible-like flow**: Unlike 3D incompressible flow, 2D thin film behaves as compressible elastic medium

---

## Governing Equations

### 3D Navier-Stokes (starting point)
$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = \frac{1}{\rho}\nabla \cdot \sigma + \mathbf{f}$$
$$\nabla \cdot \mathbf{u} = 0$$

### Surface stress condition
$$\sigma \cdot \mathbf{n} = (2\mathcal{C}\gamma - p_a)\mathbf{n} + \nabla_s \gamma$$

### Surface tension (Marangoni)
$$\gamma = \gamma_a - \gamma_r \Gamma$$
where $\gamma_a$ = surface tension of pure water, $\gamma_r$ = elasticity coefficient, $\Gamma$ = surfactant concentration

### Reduced 2D equations on sphere (main simulation model)
$$\frac{D\mathbf{u}}{Dt} = -\frac{M}{\eta}\nabla\Gamma + \frac{C_r}{\eta}(\mathbf{u}_{air} - \mathbf{u}) + \mathbf{g}$$
$$\frac{D\Gamma}{Dt} = -\Gamma \nabla \cdot \mathbf{u}$$
$$\frac{D\eta}{Dt} = -\eta \nabla \cdot \mathbf{u}$$

### Dimensionless numbers
| Parameter | Definition | Typical Value |
|-----------|------------|---------------|
| Marangoni number | $M = \frac{\Gamma_0 \bar{R} T}{\rho \eta_0 U^2}$ | 0.83 |
| Reynolds number | $Re = \frac{UR\rho}{\mu}$ | $5.6 \times 10^4$ |
| Drag coefficient | $C_r = \frac{\rho_a \sqrt{\nu_a R}}{\rho \eta_0 \sqrt{U}}$ | 2.1 |

### Equilibrium thickness profile (under gravity)
$$\eta = \frac{\pi}{\int_0^\pi e^{-\frac{g\cos\theta}{M}}d\theta} e^{-\frac{g\cos\theta}{M}}$$

---

## Thin-Film Interference (Rendering)

### Optical path difference
$$\mathcal{D} = 4\eta n_s \cos\theta_s$$

### Phase shift
$$\Delta\phi = 2\pi \frac{\mathcal{D}}{\lambda}$$

### Thin-film reflectance
$$R(\lambda) = \left| r_{as} + \frac{t_{as}r_{sa}t_{sa}e^{i\Delta\phi}}{1 - r_{sa}^2 e^{i\Delta\phi}} \right|^2$$

### Transmittance
$$T(\lambda) = \left| \frac{t_{as}t_{sa}}{1 - r_{sa}^2 e^{i\Delta\phi}} \right|^2$$

### Multi-bounce light transport (order n)
$$\mathcal{R}^{(0)} = R^{(0)}, \quad \mathcal{R}^{(k)} = T^{(0)} \prod_{i=1}^{k-1} R^{(i)} T^{(k)}$$

---

## Relevance to Your Simulation

| Paper Component | Your Project Relevance |
|-----------------|------------------------|
| **Thickness field evolution** | Enhances `DrainageSimulator` - use coupled $\eta$, $\Gamma$ PDEs instead of simplified drainage |
| **Marangoni flow** | Adds surface-tension-driven convection patterns missing from current model |
| **Equilibrium wedge profile** | Validates gravity-induced thickness gradient (thinner top, thicker bottom) |
| **Interference equations** | Matches your `InterferenceCalculator` approach; confirms optical path formula |
| **Spherical advection** | Novel stable advection scheme for flow across poles (useful if adding flow simulation) |
| **BiMocq² integration** | Preserves sharp thickness features during advection |
| **Real-time spectral rendering** | 5nm wavelength sampling, polarization-aware - could enhance `bubble.wgsl` |

---

## Physical Parameters

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Bubble radius | $R$ | 0.02–0.1 | m |
| Mean half thickness | $\eta_0$ | 400–1000 | nm |
| Refractive index (soap water) | $n_s$ | 1.33 | - |
| Surface tension (water-air) | $\gamma_a$ | $7.275 \times 10^{-2}$ | N/m |
| Water density | $\rho$ | 997 | kg/m³ |

---

## Notable References

- **Chomaz (2001)** - Lubrication theory for flat soap films
- **Ida & Miksis (1998a,b)** - 3D soap film dynamics model
- **Belcour & Barla (2017)** - Thin-film interference rendering
- **Qu et al. (2019)** - BiMocq² advection scheme
- **Couder et al. (1989)** - Soap film structure (three-layer model)

---

## Part 2

# Huang et al. (2020) - Chemomechanical Soap Film Simulation (Chunk 2)

**Source**: ACM Trans. Graph., Vol. 39, No. 4, Article 1 (SIGGRAPH 2020)

---

## Key Concepts

### Evaporation Model
- Film thickness decreases at constant rate due to exposed surface area
- Modeled by subtracting constant $\eta$ each timestep
- Simulation terminates when any point reaches zero thickness
- Top of bubble thins first → gray appearance before bursting

### Black Film
- Extremely thin film (~5–30 nm) exhibiting **destructive interference** → appears black
- At this scale, molecular forces dominate:
  - Van der Waals attraction
  - Electrostatic repulsion  
  - Born repulsion
- Forms stable "islands" within colorful film regions

### Marginal Regeneration
- Boundary phenomenon causing film to thin at edges
- Produces erratic upward-flowing regions (see Nierstrasz & Frens 1999)

### Viscosity Effects
- Higher viscosity (e.g., glycerin-added solutions) → more stable, longer-lasting bubbles
- Viscous films retain texture longer, resist fractal breakup
- Example: commercial soap ~$1.2 \times 10^{-1}$ Pa·s (~100× water viscosity)
- Viscosity term: $Re^{-1}\mathbf{V}$ (Reynolds number inverse × viscous term)

---

## Equations/Formulas

### Material Derivative (Spherical Coordinates)

**Scalar quantity** $\Phi(\theta, \phi, t)$:
$$\frac{D\Phi}{Dt} = \frac{\partial\Phi}{\partial t} + u_\theta\frac{\partial\Phi}{\partial\theta} + \frac{u_\phi}{\sin\theta}\frac{\partial\Phi}{\partial\phi}$$

**Dimensionless velocities**:
$$u_\theta = \frac{d\theta}{dt}, \quad u_\phi = \sin\theta\frac{d\phi}{dt}$$

**Vector quantity** (velocity):
$$\frac{D\mathbf{u}}{Dt} = \left(\frac{\partial u_\theta}{\partial t} + u_\theta\frac{\partial u_\theta}{\partial\theta} + \frac{u_\phi}{\sin\theta}\frac{\partial u_\theta}{\partial\phi} - \frac{u_\phi^2}{\tan\theta}\right)\mathbf{e}_\theta + \left(\frac{\partial u_\phi}{\partial t} + u_\theta\frac{\partial u_\phi}{\partial\theta} + \frac{u_\phi}{\sin\theta}\frac{\partial u_\phi}{\partial\phi} + \frac{u_\theta u_\phi}{\tan\theta}\right)\mathbf{e}_\phi$$

### Unit Vector Derivatives
$$\frac{\partial\mathbf{e}_\theta}{\partial\theta} = 0, \quad \frac{\partial\mathbf{e}_\theta}{\partial\phi} = \cos\theta\,\mathbf{e}_\phi, \quad \frac{\partial\mathbf{e}_\phi}{\partial\theta} = 0, \quad \frac{\partial\mathbf{e}_\phi}{\partial\phi} = -\cos\theta\,\mathbf{e}_\theta$$

### Linear System (Appendix C)
Solved on $m \times n$ grid: $A\mathbf{\Gamma} = \mathbf{b}$
- $A \in \mathbb{R}^{mn \times mn}$ is sparse, block diagonal, 5 elements per row
- Made symmetric positive definite by multiplying by $\sin\theta$

---

## Relevance to Soap Bubble Simulation

| Aspect | Relevance |
|--------|-----------|
| **Thin-film interference** | Black film threshold (5–30 nm) defines destructive interference regime; color bands move downward as drainage progresses |
| **Drainage** | Evaporation model ($\eta$ subtraction), gravity-driven thinning at top |
| **Rendering** | Gray/black appearance at extreme thinness; viscosity affects texture persistence |
| **Spherical solver** | Material derivative formulas directly applicable to icosphere mesh advection |

---

## Notable Citations

- **Ida & Miksis (1998a,b)**: Thin film dynamics theory (general manifolds)
- **Isenberg (1978)**: Black film physics, marginal regeneration
- **Born & Wolf (1970)**: Optical interference fundamentals
- **Belcour & Barla (2017)**: Microfacet iridescence rendering
- **Iwasaki et al. (2004)**: Real-time soap bubble interference rendering
- **Ishida et al. (2020)**: Soap film dynamics with evolving thickness
- **Hill & Henderson (2016)**: Efficient fluid simulation on sphere
- **Cabral & Leedom (1993)**: Line integral convolution (flow visualization)

---

_Total parts: 2_
