# Söderlund_2022_SDF_Ray_Tracing - Extracted Summary

_Generated: Tue Jan 21 2026_

---

## Hansson-Söderlund, Evans, Akenine-Möller (2022) - Ray Tracing of Signed Distance Function Grids

**Citation:** Hansson-Söderlund, H., Evans, A., & Akenine-Möller, T. (2022). Ray Tracing of Signed Distance Function Grids. *Journal of Computer Graphics Techniques (JCGT)*, 11(3), 94-113.

**DOI/URL:** https://jcgt.org/published/0011/03/06/

## Key Concepts

### Grid Sphere Tracing (GST)
- **Definition**: Sphere tracing through a dense grid of SDF values
- Grid stores all voxels including those not containing the SDF surface
- Values formatted and clamped to [-1, 1], where 1 encodes 2.5 interior voxel diagonals

### Storage Strategies
- **SVS Storage**: Stores 4×4×4 signed distance values around a voxel (vs 2×2×2 per voxel)
- Allows easy access to neighboring voxel values for normal evaluation
- Custom intersection shader invoked when voxel is reached

### Sparse Voxel Octree (SVO)
- Follows work by Laine and Karras (2010)
- All traversal occurs in shader code
- Memory-efficient for sparse surfaces

### Trilinear Interpolation
- Optimized intersection computation between ray and trilinearly-interpolated surface
- Constants grouped for fewer computations
- Better use of fused-multiply-and-add (FMA) operations

## Equations/Formulas

### Ray-Surface Intersection
For trilinear interpolation in a voxel:
$$f(t) = at^3 + bt^2 + ct + d = 0$$
where coefficients derive from the 8 corner SDF values and ray parameters.

### Continuous Normals
For hit point, compute normals for each of 4 voxels at same hit point:
$$\mathbf{n} = \sum_{i} w_i \mathbf{n}_i$$
where weights $w_i$ based on position within voxel (barycentric-like weighting).

## Relevance to Simulation

| Topic | Application |
|-------|-------------|
| **SDF rendering** | Future bubble deformation using signed distance fields |
| **Ray marching** | Alternative to rasterization for complex surfaces |
| **Trilinear interpolation** | Smooth surface reconstruction from discrete grid |
| **Normal computation** | Continuous normals for correct shading across voxel boundaries |
| **Performance** | Path tracing evaluation shows feasibility for real-time |

## Implementation Details

### Test Configuration
- In-house path tracer running in Falcor framework
- Four test scenes: Cheese, Goblin, Heads, Ladies
- Full path tracing with up to 3 bounces
- Single square light source

### Key Findings
- GST effective for dense grids
- SVS storage enables accurate normal computation
- Trilinear intersection more accurate than sphere tracing alone

## Citations

- Laine, S., & Karras, T. (2010). Efficient sparse voxel octrees. *ACM SIGGRAPH Symposium on Interactive 3D Graphics and Games*.
- Parker, S., et al. (1998). Interactive ray tracing for isosurface rendering.
- Hart, J. C. (1996). Sphere tracing: A geometric method for the antialiased ray tracing of implicit surfaces.

---

_Source: Web-based summary from JCGT, NVIDIA Research, and search results_
