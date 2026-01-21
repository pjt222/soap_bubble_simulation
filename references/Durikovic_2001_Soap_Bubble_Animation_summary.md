# Durikovič_2001_Soap_Bubble_Animation - Extracted Summary

_Generated: Tue Jan 21 2026_

---

## Durikovič (2001) - Animation of Soap Bubble Dynamics, Cluster Formation and Collision

**Citation:** Durikovič, R. (2001). Animation of Soap Bubble Dynamics, Cluster Formation and Collision. *Computer Graphics Forum*, 20(3), 67-76.

**DOI:** 10.1111/1467-8659.00499

**URLs:**
- https://diglib.eg.org/handle/10.2312/8850
- https://onlinelibrary.wiley.com/doi/pdf/10.1111/1467-8659.00499

## Key Concepts

### Physics-Based Simulation
- **Focus**: Complete dynamic simulation of soap bubbles
- Addresses: "What happens when a soap bubble floats in air?"
- Models: Dynamic formation of irregular bubble clusters
- Handles: Bubble collision and coalescence

### Spring Mesh Model
- Bubble surface represented as triangular mesh
- Vertices connected by spring elements
- Springs model surface tension forces
- Enables deformation under external forces

### Van der Waals Forces
- **Definition**: Intermolecular attraction between bubble surfaces
- Approximates interaction forces between nearby bubbles
- Drives bubble adhesion and cluster formation
- Distance-dependent: strong at close range, negligible far away

### External Forces
| Force | Effect |
|-------|--------|
| **Gravity** | Vertical acceleration, shape distortion |
| **Wind** | Horizontal displacement, surface ripples |
| **Buoyancy** | Upward force (bubble lighter than air) |
| **Drag** | Air resistance during motion |

## Physical Principles

### Surface Tension
$$F_{tension} = \gamma \cdot \kappa$$
where $\gamma$ is surface tension coefficient and $\kappa$ is local curvature.

### Principal Curvatures
- $R_1$ and $R_2$: Principal radii of curvature
- Occur in perpendicular planes
- Both perpendicular to surface tangent plane

### Laplace Pressure
$$\Delta P = \gamma \left(\frac{1}{R_1} + \frac{1}{R_2}\right)$$
Pressure difference across curved interface.

### Plateau's Rules (for clusters)
- Films meet in **threes** at 120° angles
- Edges meet in **fours** at tetrahedral angles (~109.47°)

## Relevance to Simulation

| Topic | Application |
|-------|-------------|
| **Spring mesh** | Alternative to thickness-based deformation |
| **Van der Waals** | Multi-bubble interaction model |
| **Cluster formation** | Extending beyond single bubble |
| **Collision handling** | Bubble merging physics |
| **External forces** | Wind and gravity effects on shape |
| **Dynamic animation** | Time-stepping for bubble motion |

## Implementation Details

### Mesh Representation
- Triangular mesh approximates spherical surface
- Vertex positions updated each timestep
- Spring forces computed from edge lengths
- Damping prevents oscillation

### Interaction Model
- Proximity detection for nearby bubbles
- Van der Waals potential for adhesion
- Collision response when surfaces contact
- Topology changes for merging

### Animation Pipeline
1. Compute internal spring forces
2. Apply external forces (gravity, wind)
3. Detect bubble-bubble interactions
4. Update vertex positions (integration)
5. Handle topology changes (merge/split)
6. Render with interference colors

## Limitations (noted in later papers)

- Does not model film thickness evolution
- Simplified interference (no drainage effects)
- No Marangoni flows on surface
- Static optical properties

## Citations

- Boys, C. V. (1959). *Soap Bubbles: Their Colors and Forces Which Mold Them*. Dover.
- Plateau, J. (1873). *Statique Expérimentale et Théorique des Liquides*.
- Glassner, A. (2000). Soap Bubbles: Parts 1 & 2. *IEEE Computer Graphics and Applications*.

---

_Source: Eurographics Digital Library, ResearchGate, CGF abstracts_
