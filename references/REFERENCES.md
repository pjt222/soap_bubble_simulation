# References Tracking

Status tracking for academic references used in the soap bubble simulation project.

**Last Updated:** 2026-01-20

## Status Legend

| Field | Values |
|-------|--------|
| **Collected** | ✓ PDF obtained / ✗ Not yet / — N/A (online/book) |
| **Summarized** | ✓ Summary exists / ✗ Not yet / — N/A |
| **In App** | ✓ Concepts implemented / ◐ Partial / ✗ Not yet |

---

## Core Papers

### Thin-Film Interference & BRDF

| Reference | Type | Collected | Summarized | In App | Source |
|-----------|------|:---------:|:----------:|:------:|--------|
| Schlick (1994) "An Inexpensive BRDF Model for Physically-based Rendering" | Paper | ✓ | ✓ | ✓ | [DOI](https://doi.org/10.1111/1467-8659.1330233) |
| Glassner (2000) "Soap Bubbles: Part 1" *IEEE CG&A* | Paper | ✓ | ✓ | ◐ | IEEE |
| Glassner (2000) "Soap Bubbles: Part 2" *IEEE CG&A* | Paper | ✓ | ✓ | ◐ | IEEE |

### Soap Film Dynamics

| Reference | Type | Collected | Summarized | In App | Source |
|-----------|------|:---------:|:----------:|:------:|--------|
| Huang et al. (2020) "Chemomechanical simulation of soap film flow on spherical bubbles" *ACM TOG* | Paper | ✓ | ✓ | ✗ | [DOI](https://doi.org/10.1145/3386569.3392094) |
| Durikovič (2001) "Animation of Soap Bubble Dynamics, Cluster Formation and Collision" | Paper | ✗ | ✗ | ✗ | — |

### Surface Optimization & Network Physics

| Reference | Type | Collected | Summarized | In App | Source |
|-----------|------|:---------:|:----------:|:------:|--------|
| Meng et al. (2026) "Surface optimization governs the local design of physical networks" *Nature* | Paper | ✓ | ✓ | ✗ | [DOI](https://doi.org/10.1038/s41586-025-09784-4) |

### Branched Flow

| Reference | Type | Collected | Summarized | In App | Source |
|-----------|------|:---------:|:----------:|:------:|--------|
| Patsyk et al. (2020) "Observation of branched flow of light" *Nature* | Paper | ✗ | ✗ | ✗ | [DOI](https://doi.org/10.1038/s41586-020-2376-8) |
| Jura et al. (2007) "Unexpected features of branched flow..." *Nature Physics* | Paper | ✗ | ✗ | ✗ | [DOI](https://doi.org/10.1038/nphys756) |

---

## SDF & Rendering Techniques

| Reference | Type | Collected | Summarized | In App | Source |
|-----------|------|:---------:|:----------:|:------:|--------|
| Söderlund et al. (2022) "Ray Tracing of Signed Distance Function Grids" *JCGT* | Paper | ✗ | ✗ | ✗ | [JCGT](https://jcgt.org/published/0011/03/06/) |
| Losasso & Hoppe (2004) "Geometry Clipmaps: Terrain Rendering Using Nested Regular Grids" | Paper | ✗ | ✗ | ✗ | [PDF](https://hhoppe.com/geomclipmap.pdf) |
| Green (2007) "Improved Alpha-Tested Magnification for Vector Textures" *SIGGRAPH* | Paper | ✗ | ✗ | ✗ | [PDF](https://steamcdn-a.akamaihd.net/apps/valve/2007/SIGGRAPH2007_AlphaTestedMagnification.pdf) |

---

## String Theory (Background)

| Reference | Type | Collected | Summarized | In App | Source |
|-----------|------|:---------:|:----------:|:------:|--------|
| Witten (1986) "Non-commutative geometry and string field theory" *Nucl. Phys. B* | Paper | ✗ | ✗ | ✗ | — |
| Carlip (1988) "Quadratic differentials and closed string vertices" *Phys. Lett. B* | Paper | ✗ | ✗ | ✗ | — |
| Saadi & Zwiebach (1989) "Closed string field theory from polyhedra" *Ann. Phys.* | Paper | ✗ | ✗ | ✗ | — |
| Tong "Lectures on String Theory" | Online | — | ✗ | ✗ | [URL](http://www.damtp.cam.ac.uk/user/tong/string.html) |

---

## Books

| Reference | Type | Collected | Summarized | In App | Notes |
|-----------|------|:---------:|:----------:|:------:|-------|
| Born & Wolf (1999) *Principles of Optics* 7th ed. | Book | ✗ | ✗ | ◐ | Ch. 7, 13 relevant |
| Macleod (2010) *Thin-Film Optical Filters* 4th ed. | Book | ✗ | ✗ | ✗ | — |
| Isenberg (1992) *The Science of Soap Films and Soap Bubbles* | Book | ✗ | ✗ | ✗ | Dover, classic reference |
| de Gennes et al. (2004) *Capillarity and Wetting Phenomena* | Book | ✗ | ✗ | ✗ | Thin films chapter |

---

## Video Resources

| Reference | Type | Watched | Notes | Source |
|-----------|------|:-------:|-------|--------|
| Turitzin "SDF-based Game Engine" | Video | ✗ | Brick maps, clipmaps, physics | [YouTube](https://www.youtube.com/watch?v=il-TXbn5iMA) |
| Lague "Coding Adventure: Ray Marching" | Video | ✗ | Beginner SDF intro | [YouTube](https://www.youtube.com/watch?v=Cp5WWtMoeKg) |
| Nature "Branched Flow of Light" | Video | ✗ | Experimental footage | [Nature](https://www.nature.com/articles/d41586-020-01991-5) |

---

## Online Resources & Tools

| Resource | Type | Status | Source |
|----------|------|:------:|--------|
| wgpu Documentation | Docs | ✓ In use | [wgpu.rs](https://wgpu.rs/) |
| Learn WGPU Tutorial | Tutorial | ✓ Referenced | [URL](https://sotrh.github.io/learn-wgpu/) |
| Inigo Quilez SDF Articles | Articles | ◐ Referenced | [iquilezles.org](https://iquilezles.org/articles/) |
| Ray Optics Simulation | Tool | ✗ | [phydemo.app](https://phydemo.app/ray-optics/) |
| min-surf-netw (Barabási Lab) | Code | ✗ | [GitHub](https://github.com/Barabasi-Lab/min-surf-netw) |
| Physical Network Dataset | Data | ✗ | [physical.network](https://physical.network) |
| Surface Evolver | Tool | ✗ | [URL](https://kenbrakke.com/evolver/) |
| tmm (Python) | Library | ✗ | [PyPI](https://pypi.org/project/tmm/) |

---

## Summary Statistics

| Category | Total | Collected | Summarized | In App |
|----------|:-----:|:---------:|:----------:|:------:|
| Core Papers | 7 | 5 | 5 | 2 |
| SDF/Rendering Papers | 3 | 0 | 0 | 0 |
| String Theory Papers | 4 | 0 | 0 | 0 |
| Books | 4 | 0 | 0 | 0 |
| Videos | 3 | 0 | — | — |

---

## Priority Queue

Papers to collect next (by relevance):

1. **Patsyk et al. (2020)** - Branched flow of light ⭐ High priority
2. **Söderlund et al. (2022)** - SDF ray tracing
3. **Losasso & Hoppe (2004)** - Geometry clipmaps
4. **Green (2007)** - SDF magnification
5. **Durikovič (2001)** - Bubble animation

---

## File Locations

```
references/
├── REFERENCES.md                              # This file
├── Schlick_1994_BRDF_Model.pdf
├── Schlick_1994_BRDF_Model_summary.md
├── Glassner_2000_Soap_Bubbles_Part1.pdf
├── Glassner_2000_Soap_Bubbles_Part1_summary.md
├── Glassner_2000_Soap_Bubbles_Part2.pdf
├── Glassner_2000_Soap_Bubbles_Part2_summary.md
├── Huang_2020_Chemomechanical_soap_film.pdf
├── Huang_2020_Chemomechanical_soap_film_summary.md
├── Meng_2026_Surface_optimization.pdf
├── Meng_2026_Surface_optimization_summary.md
├── chunks_*/                                  # PDF chunks for extraction
├── summaries_*/                               # Individual chunk summaries
├── split_pdf.sh                               # PDF splitting script
├── extract_pdf.sh                             # Extraction workflow script
└── merge_summaries.sh                         # Summary merging script
```

---

## Extraction Workflow

To add a new reference:

```bash
cd references/

# 1. Add PDF file (naming convention: Author_Year_ShortTitle.pdf)

# 2. Run extraction
./extract_pdf.sh NewPaper_2024_Title.pdf 10

# 3. Update this file (REFERENCES.md) with status
```
