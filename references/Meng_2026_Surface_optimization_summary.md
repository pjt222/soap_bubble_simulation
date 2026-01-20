# Meng_2026_Surface_optimization - Extracted Summary

_Generated: Tue Jan 20 14:14:00 CET 2026_

---

## Part 1

# Meng et al. (2026) - Surface Optimisation in Physical Networks

## Key Concepts

- **Physical networks**: Tangible networks (brain connectomes, vascular systems, plant roots) where material resources constrain layout
- **Optimal wiring hypothesis**: Networks minimize total link length (1D approximation)
- **Steiner graph**: Classic solution predicting shortest connections via intermediate branching points
- **Surface minimisation**: Proposed alternative accounting for full 3D geometry of network links as smooth manifolds
- **Manifold representation**: Networks modeled as 2D surfaces embedded in 3D space (charts/atlas formalism)

### Steiner Rules (classical predictions):
1. **Bifurcation only**: All branching nodes have degree $k=3$
2. **Planarity**: Three links at bifurcation lie in same plane ($\Omega = 2\pi$)
3. **Angle symmetry**: All branches form $\theta = 2\pi/3$ (120°) angles

### Key findings:
- Real networks violate all three Steiner rules
- **Trifurcations** ($k=4$ nodes) are common and stable under surface minimisation
- **Orthogonal sprouts** (90° branches) emerge naturally when branch diameter ratio $\rho < \rho_{th}$

---

## Equations/Formulas

**Total surface area cost function:**
$$S_{\mathcal{M}(\mathcal{G})} = \sum_{i=1}^{L} \int d^2\sigma_i \sqrt{\det \gamma_i}$$

where $\gamma_{i,\alpha\beta} \equiv \frac{\partial \mathbf{X}_i}{\partial \sigma_i^\alpha} \cdot \frac{\partial \mathbf{X}_i}{\partial \sigma_i^\beta}$

**Functional constraint (minimum circumference):**
$$\oint_{\text{circumference}} dl_i \geq w$$

where $(dl_i)^2 = \sum_{\alpha,\beta} \gamma_{i,\alpha\beta} \, d\sigma_i^\alpha \, d\sigma_i^\beta$

**Dimensionless parameters:**
- Weight parameter: $\chi = w/r$ (circumference/node distance)
- Separation: $\lambda = l/w$ (inter-node length/circumference)
- Diameter ratio: $\rho = w'/w$

**Phase transition**: At $\chi \approx 0.83$, bifurcations ($k=3$) merge into trifurcations ($k=4$)

---

## Relevance to Soap Bubble Simulation

| Paper Concept | Bubble Simulation Application |
|---------------|------------------------------|
| Surface minimisation | Soap films naturally minimize surface area (surface tension) |
| Nambu-Goto action | Equivalent to soap film energy functional |
| Smooth manifold junctions | Plateau borders where bubble films meet |
| Minimum circumference constraint | Film thickness constraints in drainage |
| Orthogonal sprouts | Could inform bubble cluster geometry |

**Direct connections:**
- The Nambu-Goto action (Eq. 1) is formally identical to soap film energy minimization
- Smooth junction conditions mirror Plateau's laws for soap bubble intersections
- The $\chi$ parameter relates thickness to geometry—analogous to film thickness in drainage simulation

---

## Notable Citations

| Ref | Author/Topic |
|-----|-------------|
| 2 | Murray (1926) - Murray's law for vascular branching |
| 17-20 | Steiner graph/problem |
| 28-30 | String theory worldsheets, Feynman diagrams |
| 40 | Manifold/atlas formalism |
| 44 | Non-contractable closed curves |
| 45 | Nambu-Goto action |

**arXiv**: 2509.23431v1 [physics.bio-ph] 27 Sep 2025

---

## Part 2

# Meng et al. (2026) - Surface Optimization Chunk 002

## Key Concepts

- **Surface minimization framework**: Physical networks in 3D Euclidean space described as 2D manifolds $\mathcal{M}(\mathcal{G})$ subject to surface area minimization
- **Volume optimization**: Alternative approach treating networks as 3D objects; existing literature assumes cylindrical links but fails at non-trivial junction topologies
- **min-surf-netw algorithm**: Uses string-theoretic solutions for surface minimization
- **Trifurcation junctions**: Consistently smooth with symmetric morphology, as predicted by surface minimization
- **Physical network features beyond scope**: Varying link thickness/curvature, loops (absent in biological networks but common in engineered grids)

## Terminology

| Term | Definition |
|------|------------|
| Manifold $\mathcal{M}(\mathcal{G})$ | 2D surface representation of physical network graph |
| Steiner graph | 1D cost-minimized network topology |
| Charts | 2D link representations |
| Atlas | Unified coordinate system via quadratic differentials |
| Catenoid sleeve | Example minimal surface for tubular links |

## Key Finding

Minimal surfaces correspond to close-to-optimal volumes:
> "Sub-optimal surfaces also increase the volume, suggesting that the minimal surfaces correspond to close-to-optimal volumes as well."

## Relevance to Soap Bubble Simulation

**Low direct relevance** - This paper focuses on physical network topology (neurons, blood vessels, trees) rather than thin-film physics. However:

- **Minimal surfaces**: The mathematical framework for surface minimization relates to how soap films naturally minimize surface area (same variational principle)
- **Smooth junctions**: Trifurcation junction smoothness mirrors how soap bubble intersections form at 120° angles (Plateau's laws)
- **Surface vs volume**: The relationship between surface and volume optimization parallels soap bubble physics where surface tension minimizes area for a given enclosed volume

## Notable Citations

- Murray (1926) - Physiological principle of minimum work (vascular systems)
- Dehmamy et al. (2018) - Structural transition in physical networks, *Nature*
- Bobenko et al. (2008) - *Discrete Differential Geometry* (mesh optimization)
- Witten (1986) - Non-commutative geometry and string field theory (theoretical basis)

## Resources

- **Dataset**: https://physical.network
- **Code**: https://github.com/Barabasi-Lab/min-surf-netw

---

## Part 3

# Meng et al. (2026) - Chunk 3: Physical Network Datasets & Surface Optimization

## Key Concepts

### Physical Network Datasets
Six real-world tree-like networks spanning 10 orders of magnitude in volume:
1. **Human neurons** - ~10⁴ neurons, ~10⁶ axons/dendrites
2. **Fruit fly neurons** - ~10³ dendritic/axonal pathways per neuron
3. **Blood vessels** - Human pulmonary arterial network, ~10³ vessels
4. **Tropical trees** - Branch structures from 29 trees
5. **Corals** - Tubular inner structure of 28 coral specimens
6. **Arabidopsis** - 3D scans of plant branches

### Data Representations
- **Volumetric images** (voxels) - easiest to construct
- **Point clouds** - sampled surface coordinates
- **Triangular meshes** - discretized surface description
- **Tetrahedral meshes** - volume triangulation with internal structure

### Skeletonization
**Kimimaro algorithm** extracts graph skeleton from volumetric data via:
1. Foreground/background classification
2. Distance transform (medial axis detection)
3. Source node selection
4. Path construction through sleeve centers
5. Recurrence until complete

### Charts (2D Links)
Physical network links modeled as **surfaces/sleeves** rather than 1D paths, parametrized by:
- $\sigma_i^0$ - longitudinal coordinate
- $\sigma_i^1$ - azimuthal coordinate (periodic)

---

## Equations/Formulas

**Penalty field** (guides paths to sleeve centers):
$$P(\mathbf{r}) = K \left(1 - \frac{D(\mathbf{r})}{\max\{D(\mathbf{r})\}}\right)^\alpha$$

**Invalidation sphere radius**:
$$\rho(\mathbf{r}) = s \times D(\mathbf{r}) + C$$

**Link length** (1D):
$$l_i = \int_0^1 d\eta_i \sqrt{\frac{d\mathbf{X}_i}{d\eta_i} \cdot \frac{d\mathbf{X}_i}{d\eta_i}}$$

**Graph cost** (Steiner optimization):
$$S_\mathcal{G} = \sum_{i=1}^{L} \int_0^1 d\eta_i \sqrt{\frac{d\mathbf{X}_i}{d\eta_i} \cdot \frac{d\mathbf{X}_i}{d\eta_i}}$$

**Manifold sewing conditions** (sleeve intersection):
$$\mathbf{X}_i(l_i, \sigma_i^1) = \mathbf{X}_j(0, \sigma_j^1)$$
$$\left.\frac{\partial \mathbf{X}_i}{\partial \sigma_i^0}\right|_{\sigma_i^0=l_i} = \left.\frac{\partial \mathbf{X}_j}{\partial \sigma_j^0}\right|_{\sigma_j^0=0}$$

**Dimensionless measures**:
- Weight parameter: $\chi = w/r$
- Circumference ratio: $\rho = w'/w$

---

## Relevance to Soap Bubble Simulation

| Concept | Application |
|---------|-------------|
| **Triangular meshes** | Current icosphere representation in `geometry.rs` |
| **Surface parametrization** (σ⁰, σ¹) | Could inform UV mapping for thickness field |
| **Manifold smoothness conditions** (Eq. 7) | Relevant for bubble deformation/merging |
| **Distance transform** | Potential use in thickness field computation |
| **Steiner graph 120° angles** | Analogous to Plateau's laws for soap film junctions |

**Steiner graph characteristics** (bifurcations at 120°) directly parallel **Plateau's laws** governing soap bubble/film junctions.

---

## Citations

- [1] Human brain electron microscopy dataset
- [2] FlyEM/Hemibrain connectome project
- [7] Skeletonization in biological systems
- [8] Kimimaro algorithm (https://github.com/seung-lab/kimimaro)
- [12] Trimesh library
- [13] Skeletor algorithm
- [14-15] Steiner graph theory

---

## Part 4

# Meng et al. (2026) - Surface Optimization (Chunk 4)

## Key Concepts

### Network Manifolds
- **Two-dimensional manifold** $\mathcal{M}(\mathcal{G})$: A physical network with graph structure $\mathcal{G}$ (nodes/links) "dressed" with 2D **sleeves** around each link
- Sleeves must be **continuous**, **smooth**, and **non-self-intersecting** at boundaries
- **Charts**: Local coordinate patches $(\sigma^0_i, \sigma^1_i)$ for each link $i$

### Riemannian Metric Tensor
$$\gamma_{i,\alpha\beta} = \frac{\partial \mathbf{X}_i}{\partial \sigma^\alpha_i} \cdot \frac{\partial \mathbf{X}_i}{\partial \sigma^\beta_i}$$

Describes local surface geometry; symmetric $2 \times 2$ tensor at each point.

### Isothermal Coordinates
A reparametrization yielding diagonal metric:
$$\gamma_{i,\alpha\beta}(\boldsymbol{\sigma}_i) = f_i(\boldsymbol{\sigma}_i)\delta_{\alpha\beta}$$

Reduces 3 d.o.f. to a single scalar function $f_i(\sigma_i)$.

---

## Core Equations

**Link surface area** (general form):
$$\int_0^{l_i} \int_0^{w_i} d^2\sigma_i \sqrt{\left(\frac{\partial \mathbf{X}_i}{\partial \sigma^0_i} \cdot \frac{\partial \mathbf{X}_i}{\partial \sigma^0_i}\right)\left(\frac{\partial \mathbf{X}_i}{\partial \sigma^1_i} \cdot \frac{\partial \mathbf{X}_i}{\partial \sigma^1_i}\right) - \left(\frac{\partial \mathbf{X}_i}{\partial \sigma^0_i} \cdot \frac{\partial \mathbf{X}_i}{\partial \sigma^1_i}\right)^2}$$

**Manifold cost** (total surface area):
$$S_{\mathcal{M}(\mathcal{G})} = \sum_{i=1}^{L} \int_{\sigma_i} d^2\sigma_i \sqrt{\det \gamma_i(\boldsymbol{\sigma}_i)}$$

**Simplified form** (isothermal coords with $f_i = e^{2\phi}$):
$$S_{\mathcal{M}(\mathcal{G})} = \int d^2\sigma \, e^{2\phi(\sigma^0, \sigma^1)}$$

**Catenoid surface** embedding:
$$\mathbf{X}(\boldsymbol{\sigma}) = \frac{w}{2\pi}\begin{pmatrix} \cosh\sigma^0 \cos\sigma^1 \\ \cosh\sigma^0 \sin\sigma^1 \\ \sigma^0 \end{pmatrix}$$

**Systolic constraint** (minimum circumference for flow):
$$\oint d\sigma^1 \|\partial \mathbf{X}/\partial \sigma^1\| \geq w$$

---

## Relevance to Soap Bubble Simulation

| Topic | Connection |
|-------|------------|
| **Thin-film geometry** | Metric tensor formalism could model film thickness variations across curved surfaces |
| **Minimal surfaces** | Surface minimization (Eq. 30) directly relates to soap film equilibrium shapes (mean curvature = 0) |
| **Drainage** | Catenoid/sleeve geometry provides parametric surfaces for thickness field mapping |
| **Rendering** | Isothermal coordinates simplify UV mapping for interference calculations |
| **Mesh generation** | Quad-mesh from quadratic differentials could improve icosphere parameterization |

---

## Notable References

- [16] Riemannian metric tensor formalism
- [17] Nambu–Goto action (string theory connection to minimal surfaces)
- [18] Isothermal coordinate systems on Riemann surfaces
- [19, 20] String field theory for global atlas construction
- [21] Jenkins–Strebel quadratic differentials
- [22] Quadratic differentials and Riemann surfaces

---

## Part 5

# Meng et al. (2026) - Surface Optimization for Network Manifolds

## Key Concepts

### Surface Minimisation
- **Metric tensor** $\gamma_{\alpha\beta}$: Captures infinitesimal surface area elements; surface area = $\int \sqrt{\det \gamma}$
- **Plateau's soap film transition**: Critical ratio $h/w_0 \approx 0.168$ where connected film becomes two disconnected planes
- **Systolic constraint**: Non-contractible loops must maintain minimum circumference $w$ (functional constraint for structural integrity)

### Strebel's Theorem
- For tree-graph networks: flat metric tensor ($e^{2\phi_i(\sigma)} \equiv 1$) yields minimal surface area
- Results in perfect cylindrical sleeves
- Limitation: Creates conical singularities at intersections (non-physical)

### Relaxed Formulation
$$e^{2\phi_i(\sigma)} \geq 1 \quad \text{(equivalently } \phi_i(\sigma) \geq 0\text{)}$$
- Allows local expansion at sleeve intersections while maintaining systolic constraint
- Produces locally area-minimising surfaces with realistic geometries

## Equations/Formulas

### Quad-Mesh Metric Tensor
$$\gamma^{\text{quad}}_{i,00} = \gamma^{\text{quad}}_{i,11} = e^{2\phi^{\text{quad}}_i(\sigma_i)}$$

### Tile Area
$$S_r = Z_r(1 + \Delta\lambda_r^2)$$

### Optimization Target Function
$$E = w_{\text{iso}}E_{\text{iso}} + w_{\text{glue}}E_{\text{glue}} + w_{\text{terminal}}E_{\text{terminal}} + w_{\text{fair}}E_{\text{fair}} + w_{\text{surface}}E_{\text{surface}}$$

**Cost terms:**
- $E_{\text{iso}}$: Isometric mapping accuracy (Varignon parallelogram constraint)
- $E_{\text{glue}}$: Boundary continuity between adjacent links
- $E_{\text{fair}}$: Curvature smoothness (minimizes second derivative)
- $E_{\text{surface}} = \sum_{r=1}^{R} Z(1 + \Delta\lambda_r^2)$: Total surface area

### Fairness Cost (Smoothness)
$$E_{\text{fair}} = \sum_{i,j,k} \frac{(x_i - 2x_j + x_k)^2 + (y_i - 2y_j + y_k)^2 + (z_i - 2z_j + z_k)^2}{(x_i - x_j)^2 + \ldots}$$

### Trifurcation Transition
$$\lambda \approx \frac{0.212}{\chi} f\left(\frac{0.212 - 0.26\chi}{\chi\sigma}\right)$$
where $\chi = w/r$ (circumference/length scale), $f(x) = (1 + e^{-x})^{-1}$ (sigmoid)

## Relevance to Soap Bubble Simulation

| Aspect | Connection |
|--------|------------|
| **Thin-film interference** | Plateau's transition models soap film instability at critical thickness ratios |
| **Drainage** | Systolic constraint ($w \geq w_{\text{min}}$) analogous to minimum film thickness before rupture |
| **Mesh generation** | Quad-mesh tiling with Varignon constraints applicable to bubble surface discretization |
| **Rendering** | Fairness cost ensures smooth surfaces without excessive curvature artifacts |
| **Deformation** | $\phi(\sigma)$ field captures local stretching—relevant for non-spherical bubble shapes |

## Notable Citations

- **[20, 21]** Strebel's theorem on minimal surfaces with systolic constraints
- **[27]** Systolic surface minimisation problem
- **[28]** Quad-mesh tiling framework
- **[29, 30]** Fairness cost and optimization implementation
- **[31]** Extrinsic trifurcations in skeletonisation

## Implementation Notes

- Algorithm: `min-surf-netw` (GitHub: Barabasi-Lab/min-surf-netw)
- Computational complexity: $O(R)$ linear in tile count
- Weight defaults: $w_{\text{iso}} = 1$, $w_{\text{glue}} = w_{\text{terminal}} = 10^3$
- Fairness annealing: Exponentially decay $w_{\text{fair}}$ from $10^0 \to 10^{-5}$ during optimization

---

## Part 6

# Meng et al. (2026) - Surface Optimization: Chunk 6 Summary

## Key Concepts

### Terminology
- **Internal vs. External legs**: Network segments connecting nodes (internal links join branch points; external legs extend to terminals)
- **Trifurcation**: Junction where 4 links meet (k=4 motif)
- **Bifurcation**: Junction where 3 links meet (k=3 motif)
- **Steiner problem**: Minimizing total link length in networks
- **Surface minimization**: Alternative optimization principle minimizing surface area of tubular manifolds
- **Sprouting vs. Branching**: Sharp angle transition where thick branches exert perpendicular sprouts
- **Gaussian curvature (κ)**: Measure of surface smoothness

### Core Finding
Real physical networks (neurons, blood vessels, trees, coral) follow **surface minimization** principles rather than Steiner's length minimization. This manifests in smooth manifold-like surfaces with near-zero Gaussian curvature.

## Equations/Formulas

### Generalised Gamma Distribution (leg lengths)
$$P(x) = \begin{cases} \frac{\gamma e^{-\left(\frac{x-\mu}{\beta}\right)^\gamma} \left(\frac{x-\mu}{\beta}\right)^{\alpha\gamma-1}}{\beta\Gamma(\alpha)} & x > \mu \\ 0 & x \leq \mu \end{cases}$$

### Internal-External Leg Relationship
$$l_{\text{int}} \approx A(l_{\text{ext}} - Bw)$$
- Internal leg vanishes when $l_{\text{ext}}/w \leq B$

### Three-dimensional Steering Angle
$$\Omega_{1\to 2} = 4\pi \sin^2\left[\frac{\pi - \theta}{4}\right]$$
- Relates solid angle Ω to branching angle θ

### Trifurcation Junction Constraints (stitch lines)
$$a + b + f = w_1, \quad a + c + e = w_2$$
$$b + c + d = w_3, \quad d + e + f = w_4$$

Solution: $a = d, \quad b = e, \quad c = f = w - a - b$

### Excess Length Ratio
$$\eta = L/L_S \approx 1.25$$
- Real networks ~25% longer than Steiner optimal

## Relevance to Soap Bubble Simulation

| Aspect | Connection |
|--------|------------|
| **Surface minimization** | Soap films naturally minimize surface area - same principle governs physical network geometry |
| **Gaussian curvature** | Soap bubbles exhibit smooth surfaces with κ ≈ 0 away from Plateau borders |
| **Junction geometry** | Trifurcation analysis (Fig. 23) parallels Plateau's laws for soap film junctions |
| **Manifold smoothness** | Requirement for continuous, differentiable patching mirrors soap film physics |

**Direct applications**:
- The smooth manifold construction requiring differentiable patching at intersections is analogous to how soap films meet at Plateau borders
- Surface area minimization under circumference constraints relates to film thickness variation during drainage

## Citations

- **[20]**: String field theory reference for manifold configurations
- **[33, 34]**: Zamir - cylindrical link geometric construction, sprouting/branching analysis
- **[35]**: Brain network loops reference
- **[36]**: Murray, Cherniak et al. - geometric framework for physical networks

---

## Part 7

# Meng et al. (2026) - Surface Optimization in Physical Networks

## Key Concepts

### Trifurcation Node Geometry
- **Barycentric coordinates** $(a, b, c)$ with constraint $a + b + c = w$ define a triangular parameter space for trifurcation manifolds
- **Surface-minimized configuration**: symmetric solution at $a = b = c = w/3$ (triangle center)
- Real-world networks cluster near symmetric configurations, avoiding corners (highly asymmetric)

### Surface vs Volume Minimisation
- Physical networks can be modeled as 2D manifolds $\mathcal{M}(\mathcal{G})$ subject to surface minimization
- **Key finding**: Surface and volume optimization yield similar optimal configurations
- Intermediate link length $\lambda = l_{int}/w \to 0$ minimizes both surface area and volume simultaneously

### Physical Network Manifold
- Links represented as **charts** with local coordinates $\sigma_i = (\sigma_i^0, \sigma_i^1)$ (longitudinal, azimuthal)
- Minimum circumference $w$ measured along azimuthal direction
- Connection to **string theory worldsheets**: Feynman diagrams map to smooth manifolds

## Equations/Formulas

### Transformed Barycentric Coordinates (Eq. 51)
$$\tilde{a} = a + (-w_1 - w_2 + w_3 + w_4)/4$$
$$\tilde{b} = b + (-w_1 + w_2 - w_3 + w_4)/4$$
$$\tilde{c} = c + (+w_1 - w_2 - w_3 + w_4)/4$$

### Key Parameters
- **Thickness ratio**: $\chi = w/r$ (circumference/spatial scale)
- **Separation parameter**: $\lambda = l/w$ (link length/circumference)
- **Bifurcation-to-trifurcation transition**: occurs at $\chi \approx 0.83$
- **Branching ratio**: $\rho = w'/w$ with sprouting-to-branching transition at $\rho \approx 0.6$

### Steiner Graph Rules
1. All branchings are bifurcations ($k = 3$)
2. Planar bifurcations with solid angle $\Omega = 2\pi$
3. Adjacent link angles $\theta = 2\pi/3$

## Relevance to Soap Bubble Simulation

| Concept | Application |
|---------|-------------|
| Surface minimization | Soap films naturally minimize surface area (same principle) |
| Manifold parametrization | UV mapping for thickness/drainage simulation |
| Barycentric coordinates | Potential for complex bubble junction geometry |
| Bifurcation/trifurcation transitions | Bubble merging/splitting dynamics |

**Limited direct relevance** to thin-film interference or drainage - this paper focuses on network topology optimization rather than optical or fluid properties.

## Notable Citations

- [16] Bobenko et al. *Discrete Differential Geometry* (2008) - mesh/surface algorithms
- [17] Tong, *Lectures on String Theory* (2009) - worldsheet mathematics
- [21] Strebel, *Quadratic Differentials* (1984) - theoretical foundation
- [27] Headrick & Zwiebach, *Convex programs for minimal-area problems* (2019)
- [35] Liu et al. *Isotopy and energy of physical networks*, Nature Physics (2021)

---

## Part 8

# Meng et al. (2026) - Chunk 008 Summary

## Key Concepts

- **Sprouting vs Branching Regimes**: Two distinct modes of network formation in physical systems
- **Branching Angle Distribution $P(\Omega)$**: Probability distribution of angles at network bifurcation points
- **Steiner Predictions**: Classical optimal network theory (120° angles) - shown to be violated in real systems
- **Manifold Theory**: The paper's proposed model that better predicts observed branching patterns
- **Parameter $\rho$**: Controls the transition between sprouting and branching regimes (relates to $\Omega_{1\to2}$)

## Key Relationships

The branching angle $\Omega$ distribution depends on the regime parameter $\rho$:
- **Sprouting regime** (dashed curves): Smaller angles, narrower distribution
- **Branching regime** (solid curves): Larger angles approaching but not reaching Steiner optimum

## Systems Analyzed

Six physical network types validated against manifold predictions:
1. Human neurons
2. Fruit fly neurons  
3. Blood vessels
4. Tropical trees
5. Coral
6. Arabidopsis (plant)

## Relevance to Soap Bubble Simulation

**Limited direct relevance** - This chunk focuses on biological/physical network branching patterns rather than thin-film physics. However:

- **Foam/Plateau border networks**: The branching angle analysis could inform modeling of soap foam structures where bubble films meet
- **Surface energy minimization**: The deviation from Steiner angles (120°) in manifold theory parallels how real soap film junctions deviate from ideal predictions under dynamic conditions
- **Network topology**: Relevant if extending simulation to multi-bubble foam systems

## Notable Observations

- Empirical data (colored dots) consistently matches manifold predictions (green curves)
- All six systems violate classical Steiner predictions (grey curves)
- Universal behavior across vastly different scales (neurons to trees)

## Citations

This is Figure 5 from the paper, referencing Figure 4 for the $\Omega_{1\to2}$ vs $\rho$ relationship.

---

_Total parts: 8_
