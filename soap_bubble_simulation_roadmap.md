# Seifenblasen-Simulation in Rust

## Projektübersicht

Physikalisch korrekte 3D-Simulation einer Seifenblase mit parametrisierbaren Eigenschaften und Echtzeit-Visualisierung. Die Simulation soll die charakteristischen Interferenzfarben, Drainage-Dynamik und optionale Deformation korrekt abbilden.

### Ziele

- Parametrisierbare Simulation (Durchmesser, Schichtdicke, Viskosität, Surfactant-Konzentration)
- Echtzeit-Rendering mit GPU-beschleunigter Visualisierung
- Export von Einzelbildern und Animationen
- Integration mit Claude Code via Dateisystem/MCP

---

## Physikalische Grundlagen

### 1. Geometrie der Seifenblase

Eine ideale Seifenblase ist eine Sphäre mit Radius $R$. Die Oberfläche besteht aus einem dünnen Flüssigkeitsfilm der Dicke $d$ (typisch 10 nm – 10 µm).

**Young-Laplace-Gleichung** für den Druckunterschied:

$$\Delta P = P_{\text{innen}} - P_{\text{außen}} = \frac{4\gamma}{R}$$

wobei $\gamma$ die Oberflächenspannung ist (Faktor 4 wegen zwei Grenzflächen).

### 2. Dünnfilm-Interferenz

Die irisierenden Farben entstehen durch Interferenz von Licht, das an der äußeren und inneren Grenzfläche des Films reflektiert wird.

**Optische Wegdifferenz:**

$$\delta = 2 n_{\text{film}} d \cos(\theta_t) + \frac{\lambda}{2}$$

wobei:
- $n_{\text{film}}$ = Brechungsindex des Seifenfilms (~1.33)
- $d$ = Filmdicke
- $\theta_t$ = Transmissionswinkel im Film
- $\lambda/2$ = Phasensprung bei Reflexion am dichteren Medium

**Interferenzbedingung:**
- Konstruktiv: $\delta = m \lambda$ (m = 1, 2, 3, ...)
- Destruktiv: $\delta = (m + \frac{1}{2}) \lambda$

### 3. Drainage und Filmverdünnung

Der Seifenfilm wird unter Schwerkraft dünner. Die Drainage-Gleichung (vereinfacht):

$$\frac{\partial d}{\partial t} = -\frac{\rho g d^3}{3\eta} + D \nabla^2 d + \text{Marangoni-Term}$$

wobei:
- $\rho$ = Dichte der Seifenlösung
- $g$ = Erdbeschleunigung
- $\eta$ = dynamische Viskosität
- $D$ = Diffusionskoeffizient

### 4. Marangoni-Effekt

Lokale Unterschiede in der Surfactant-Konzentration $\Gamma$ erzeugen Gradienten in der Oberflächenspannung:

$$\vec{F}_{\text{Marangoni}} = \nabla_s \gamma = \frac{\partial \gamma}{\partial \Gamma} \nabla_s \Gamma$$

Dies führt zu charakteristischen Strömungsmustern auf der Blasenoberfläche.

### 5. Oberflächenminimierung und Netzwerk-Physik

Ein fundamentales Prinzip, das Seifenblasen mit biologischen und physikalischen Netzwerken (Neuronen, Blutgefäße, Korallen) verbindet, ist die **Oberflächenminimierung**. Meng et al. (2026) zeigten, dass diese Systeme nicht der klassischen Längenminimierung (Steiner-Graphen) folgen, sondern ihre Oberfläche minimieren.

**Nambu-Goto-Wirkung** (aus der String-Theorie):

$$S_{\mathcal{M}(\mathcal{G})} = \sum_{i=1}^{L} \int d^2\sigma_i \sqrt{\det \gamma_i}$$

wobei $\gamma_{i,\alpha\beta} \equiv \frac{\partial \mathbf{X}_i}{\partial \sigma_i^\alpha} \cdot \frac{\partial \mathbf{X}_i}{\partial \sigma_i^\beta}$ die induzierte Metrik auf der Oberfläche beschreibt.

Diese Formulierung ist formal identisch mit der Oberflächenminimierung bei Seifenblasen! Die Verbindung zur String-Theorie ermöglicht analytische Lösungen für komplexe Verzweigungsgeometrien.

**Relevante Vorhersagen für Seifenblasen:**
- **Phasenübergang**: Bei dimensionslosem Parameter $\chi = w/r \approx 0.83$ (Umfang/Radius) wechselt die optimale Struktur von Bifurkation zu Trifurkation
- **Orthogonale Sprossen**: Dünnere Zweige sprießen bevorzugt im 90°-Winkel – ein Prinzip, das auch für Blasen-Cluster relevant sein könnte
- **Smooth Manifolds**: Die Übergänge zwischen Strukturen müssen glatt und singularitätsfrei sein

---

---

## Architektur-Entscheidung: SDF vs. Mesh

### Option A: Traditionelles Mesh (aktueller Plan)

**Vorteile:**
- Etablierte Rendering-Pipeline
- Einfache Textur-Koordinaten für Filmdicke
- Gute Integration mit Standard-Tools

**Nachteile:**
- Tessellierung erforderlich
- Artefakte bei starker Verformung
- Komplexere Kollisionserkennung

### Option B: SDF-basierter Ansatz

**Vorteile:**
- Perfekt glatte Oberflächen
- Natürliche Darstellung des Dünnfilms
- Effiziente Boolesche Operationen (Blasen-Cluster!)
- Einfache Kollisionserkennung
- Dynamische Verformung ohne Remeshing

**Nachteile:**
- Höherer initialer Implementierungsaufwand
- Ray-Marching kann teuer sein (aber optimierbar)
- Weniger etablierte Tools

### Empfehlung: Hybrid-Ansatz

Für maximale Flexibilität könnte ein **Hybrid-Ansatz** sinnvoll sein:

1. **SDF für Geometrie-Definition** – Blase als Hohlkugel-SDF
2. **Mesh-Extraktion für Standard-Rendering** – Marching Cubes für Kompatibilität
3. **Direktes SDF Ray-Marching** – Für hochqualitative Renderings

```rust
pub enum RenderMode {
    /// Mesh via Marching Cubes (kompatibel, schnell)
    MeshBased,
    /// Direktes Ray-Marching (höchste Qualität)
    SDFRayMarch,
    /// Hybrid: Mesh für Szene, SDF für Nahaufnahmen
    Hybrid { lod_threshold: f32 },
}
```

---

## Architektur

```
soap-bubble-sim/
├── Cargo.toml
├── config/
│   └── default.json          # Standard-Parameter
├── src/
│   ├── main.rs               # Entry Point
│   ├── lib.rs                # Library Root
│   ├── config.rs             # Parameter-Handling
│   ├── physics/
│   │   ├── mod.rs
│   │   ├── geometry.rs       # Sphärische Geometrie
│   │   ├── drainage.rs       # Filmdicken-Dynamik
│   │   ├── interference.rs   # Optische Berechnungen
│   │   ├── fluid.rs          # Oberflächenströmung
│   │   └── branched_flow.rs  # Branched Flow Simulation
│   ├── sdf/
│   │   ├── mod.rs
│   │   ├── primitives.rs     # Basis-SDFs (Kugel, Hohlkugel)
│   │   ├── operations.rs     # Union, Intersection, Smooth-Min
│   │   ├── brick_map.rs      # Spärliches SDF-Caching
│   │   └── marching_cubes.rs # Mesh-Extraktion aus SDF
│   ├── render/
│   │   ├── mod.rs
│   │   ├── pipeline.rs       # wgpu Render Pipeline
│   │   ├── shaders/
│   │   │   ├── bubble.wgsl   # Vertex + Fragment Shader
│   │   │   ├── compute.wgsl  # Compute Shader für Physik
│   │   │   ├── raycast.wgsl  # Ray-Tracing für Branched Flow
│   │   │   └── raymarch.wgsl # SDF Ray-Marching Shader
│   │   └── camera.rs         # Kamerasteuerung
│   └── export/
│       ├── mod.rs
│       ├── image.rs          # PNG/EXR Export
│       └── video.rs          # FFmpeg Integration
├── assets/
│   └── environment.hdr       # Environment Map für Reflexionen
└── README.md
```

---

## Abhängigkeiten (Cargo.toml)

```toml
[package]
name = "soap-bubble-sim"
version = "0.1.0"
edition = "2021"

[dependencies]
# GPU & Rendering
wgpu = "0.19"
winit = "0.29"
pollster = "0.3"              # Async Runtime für wgpu
bytemuck = { version = "1.14", features = ["derive"] }

# Mathematik
nalgebra = "0.32"             # Lineare Algebra
glam = "0.25"                 # GPU-freundliche Vektoren

# Konfiguration & Serialisierung
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
toml = "0.8"

# Bildexport
image = "0.24"
exr = "1.7"                   # HDR Export

# Logging & CLI
env_logger = "0.11"
log = "0.4"
clap = { version = "4.4", features = ["derive"] }

# Optional: GUI
egui = "0.26"
egui-wgpu = "0.26"
egui-winit = "0.26"

[profile.release]
opt-level = 3
lto = true
```

---

## Parameter-Struktur

```rust
// src/config.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BubbleParameters {
    /// Durchmesser der Blase in Metern
    pub diameter: f64,
    
    /// Initiale Filmdicke in Nanometern
    pub film_thickness_nm: f64,
    
    /// Minimale Filmdicke bevor die Blase platzt (nm)
    pub critical_thickness_nm: f64,
    
    /// Brechungsindex des Seifenfilms
    pub refractive_index: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluidParameters {
    /// Dynamische Viskosität in Pa·s
    pub viscosity: f64,
    
    /// Oberflächenspannung in N/m
    pub surface_tension: f64,
    
    /// Dichte in kg/m³
    pub density: f64,
    
    /// Surfactant-Diffusionskoeffizient in m²/s
    pub surfactant_diffusion: f64,
    
    /// Surfactant-Konzentration (relativ, 0-1)
    pub surfactant_concentration: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentParameters {
    /// Gravitationsbeschleunigung in m/s²
    pub gravity: f64,
    
    /// Umgebungstemperatur in Kelvin
    pub temperature: f64,
    
    /// Luftdruck in Pascal
    pub pressure: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub bubble: BubbleParameters,
    pub fluid: FluidParameters,
    pub environment: EnvironmentParameters,
    
    /// Zeitschritt in Sekunden
    pub dt: f64,
    
    /// Auflösung der Simulation (Vertices auf der Sphäre)
    pub resolution: u32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            bubble: BubbleParameters {
                diameter: 0.05,              // 5 cm
                film_thickness_nm: 500.0,    // 500 nm
                critical_thickness_nm: 10.0, // 10 nm
                refractive_index: 1.33,
            },
            fluid: FluidParameters {
                viscosity: 0.001,            // ~Wasser
                surface_tension: 0.025,      // Seifenlösung
                density: 1000.0,             // kg/m³
                surfactant_diffusion: 1e-9,  // m²/s
                surfactant_concentration: 0.5,
            },
            environment: EnvironmentParameters {
                gravity: 9.81,
                temperature: 293.15,         // 20°C
                pressure: 101325.0,          // 1 atm
            },
            dt: 0.001,                       // 1 ms
            resolution: 128,                 // 128x256 Grid
        }
    }
}
```

---

## Beispiel-Konfiguration (JSON)

```json
{
  "bubble": {
    "diameter": 0.08,
    "film_thickness_nm": 600.0,
    "critical_thickness_nm": 15.0,
    "refractive_index": 1.34
  },
  "fluid": {
    "viscosity": 0.0015,
    "surface_tension": 0.028,
    "density": 1020.0,
    "surfactant_diffusion": 8e-10,
    "surfactant_concentration": 0.6
  },
  "environment": {
    "gravity": 9.81,
    "temperature": 298.15,
    "pressure": 101325.0
  },
  "dt": 0.0005,
  "resolution": 256
}
```

---

## Kern-Algorithmen

### 1. Dünnfilm-Interferenz (Fragment Shader)

```wgsl
// src/render/shaders/bubble.wgsl

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) film_thickness: f32,  // in nm
};

struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    refractive_index: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// Berechnet Interferenzfarbe basierend auf Filmdicke und Blickwinkel
fn thin_film_interference(
    thickness_nm: f32,
    cos_theta: f32,
    n_film: f32
) -> vec3<f32> {
    // Wellenlängen für RGB (nm)
    let wavelengths = vec3<f32>(650.0, 532.0, 450.0);
    
    // Transmissionswinkel im Film (Snellius)
    let n_air = 1.0;
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    let sin_theta_t = sin_theta * n_air / n_film;
    let cos_theta_t = sqrt(1.0 - sin_theta_t * sin_theta_t);
    
    // Optische Wegdifferenz
    let optical_path = 2.0 * n_film * thickness_nm * cos_theta_t;
    
    // Phasendifferenz für jede Wellenlänge
    let phase = 2.0 * 3.14159265 * optical_path / wavelengths + 3.14159265;
    
    // Interferenz-Intensität (vereinfacht)
    let intensity = 0.5 * (1.0 + cos(phase));
    
    // Fresnel-Reflexion (Näherung für schwache Reflexion)
    let r0 = pow((n_film - n_air) / (n_film + n_air), 2.0);
    let fresnel = r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);
    
    return intensity * fresnel * 4.0;  // Verstärkung für Sichtbarkeit
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let view_dir = normalize(uniforms.camera_pos - in.world_pos);
    let cos_theta = abs(dot(view_dir, in.normal));
    
    let interference_color = thin_film_interference(
        in.film_thickness,
        cos_theta,
        uniforms.refractive_index
    );
    
    // Basis-Transparenz
    let base_color = vec3<f32>(0.95, 0.95, 0.98);
    let final_color = base_color * 0.1 + interference_color;
    
    // Alpha basierend auf Fresnel (Rand sichtbarer)
    let alpha = 0.3 + 0.7 * (1.0 - cos_theta);
    
    return vec4<f32>(final_color, alpha);
}
```

### 2. Drainage-Simulation (Compute Shader)

```wgsl
// src/render/shaders/compute.wgsl

struct SimParams {
    dt: f32,
    gravity: f32,
    viscosity: f32,
    density: f32,
    diffusion: f32,
    grid_size: u32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> thickness_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> thickness_out: array<f32>;

// Index-Konvertierung für sphärisches Grid (θ, φ)
fn get_index(theta_idx: u32, phi_idx: u32) -> u32 {
    return theta_idx * params.grid_size * 2u + phi_idx;
}

@compute @workgroup_size(16, 16)
fn drainage_step(@builtin(global_invocation_id) id: vec3<u32>) {
    let theta_idx = id.x;
    let phi_idx = id.y;
    
    if (theta_idx >= params.grid_size || phi_idx >= params.grid_size * 2u) {
        return;
    }
    
    let idx = get_index(theta_idx, phi_idx);
    let d = thickness_in[idx];
    
    // Sphärische Koordinaten
    let theta = f32(theta_idx) / f32(params.grid_size) * 3.14159265;
    let sin_theta = sin(theta);
    let cos_theta = cos(theta);
    
    // Drainage-Term: Schwerkraft treibt Film nach unten
    // dd/dt = -ρgd³/(3η) * sin(θ) (vereinfacht)
    let drainage = params.density * params.gravity * d * d * d 
                   / (3.0 * params.viscosity) * sin_theta;
    
    // Diffusions-Term (Laplacian auf Sphäre, vereinfacht)
    let d_north = thickness_in[get_index(max(theta_idx, 1u) - 1u, phi_idx)];
    let d_south = thickness_in[get_index(min(theta_idx + 1u, params.grid_size - 1u), phi_idx)];
    let phi_left = (phi_idx + params.grid_size * 2u - 1u) % (params.grid_size * 2u);
    let phi_right = (phi_idx + 1u) % (params.grid_size * 2u);
    let d_west = thickness_in[get_index(theta_idx, phi_left)];
    let d_east = thickness_in[get_index(theta_idx, phi_right)];
    
    let laplacian = d_north + d_south + d_west + d_east - 4.0 * d;
    let diffusion = params.diffusion * laplacian;
    
    // Update
    let new_thickness = d - drainage * params.dt + diffusion * params.dt;
    thickness_out[idx] = max(new_thickness, 1.0);  // Minimum 1 nm
}
```

---

## Implementierungs-Roadmap

### Phase 1: Grundgerüst (Woche 1)
- [ ] Projekt-Setup mit wgpu und winit
- [ ] Sphären-Mesh-Generierung mit UV-Koordinaten
- [ ] Basis-Render-Pipeline (einfarbige Sphäre)
- [ ] Kamera mit Orbit-Controls
- [ ] JSON-Config laden

### Phase 2: Optik (Woche 2)
- [ ] Dünnfilm-Interferenz im Fragment Shader
- [ ] Fresnel-Reflexion
- [ ] Uniforme Filmdicke als Parameter
- [ ] Environment Mapping für Reflexionen

### Phase 3: Dynamik (Woche 3)
- [ ] Filmdicken-Textur (Storage Buffer)
- [ ] Compute Shader für Drainage
- [ ] Marangoni-Term implementieren
- [ ] Zeitliche Animation

### Phase 4: Erweiterungen (Woche 4)
- [ ] Deformation via Spherical Harmonics
- [ ] Bildexport (PNG, EXR)
- [ ] Video-Export via FFmpeg
- [ ] Optional: GUI mit egui

### Phase 5: Branched Flow (Woche 5)
- [ ] Korrelierte Dickenvariation (Perlin/Simplex Noise)
- [ ] Ray-Tracing durch Dickenfeld
- [ ] Kaustik-Berechnung und Visualisierung
- [ ] Vergleich mit Patsyk et al. (2020) Experimenten
- [ ] Laser-Einkopplung als optionaler Modus

### Phase 6: Integration (Woche 6)
- [ ] CLI-Interface mit clap
- [ ] Batch-Modus für Parameter-Sweeps
- [ ] MCP-Server-Integration (optional)
- [ ] Dokumentation

---

## CLI-Interface

```bash
# Echtzeit-Visualisierung mit Default-Parametern
soap-bubble-sim

# Mit Custom-Config
soap-bubble-sim --config my_bubble.json

# Parameter direkt überschreiben
soap-bubble-sim --diameter 0.1 --thickness 800

# Bildsequenz exportieren
soap-bubble-sim --export frames/ --frames 300 --fps 60

# Video exportieren
soap-bubble-sim --export-video bubble.mp4 --duration 10.0

# Headless-Modus (ohne Fenster)
soap-bubble-sim --headless --export snapshot.png
```

---

## MCP/Claude Code Integration

Für die Integration mit Claude Code gibt es zwei Hauptansätze:

### Option A: Dateisystem-basiert

```
input/
├── config.json      # Parameter
└── commands.json    # Steuerungsbefehle

output/
├── frames/          # Exportierte Bilder
├── state.json       # Aktueller Simulationszustand
└── log.txt          # Ausgaben
```

Claude Code kann:
1. `config.json` schreiben/modifizieren
2. Simulation starten via Shell-Befehl
3. Output-Dateien lesen und analysieren

### Option B: MCP-Server (fortgeschritten)

Ein MCP-Server könnte folgende Tools exponieren:

```json
{
  "tools": [
    {
      "name": "set_bubble_parameters",
      "description": "Setzt Simulationsparameter",
      "parameters": {
        "diameter": "number",
        "film_thickness_nm": "number",
        "viscosity": "number"
      }
    },
    {
      "name": "run_simulation",
      "description": "Startet die Simulation",
      "parameters": {
        "duration_seconds": "number",
        "export_frames": "boolean"
      }
    },
    {
      "name": "get_state",
      "description": "Liefert aktuellen Simulationszustand",
      "returns": {
        "time": "number",
        "avg_thickness": "number",
        "min_thickness": "number"
      }
    },
    {
      "name": "export_image",
      "description": "Exportiert aktuelles Frame",
      "parameters": {
        "filename": "string",
        "format": "png|exr"
      }
    }
  ]
}
```

---

### Experimenteller Aufbau (aus Patsyk et al.)

Basierend auf dem Nature-Paper und dem zugehörigen Video lässt sich der experimentelle Aufbau wie folgt zusammenfassen:

**Flacher Seifenfilm:**
```
┌─────────────────────────────────────────┐
│                                         │
│   Laser ──► Linse ──► Faser ──┐        │
│                               │        │
│            ┌──────────────────┼────┐   │
│            │   Seifenfilm     │    │   │
│            │   (vertikal)     ●────┼──►│ Kamera
│            │                       │   │
│            └───────────────────────┘   │
│                                         │
└─────────────────────────────────────────┘
```

**Sphärische Blase:**
```
┌─────────────────────────────────────────┐
│                                         │
│                    ╭───────╮            │
│   Faser ──────────●│       │            │
│    (in Film       ╰───────╯            │
│     eingekoppelt)    Blase             │
│                                         │
│              Beobachtung von oben       │
│                      ▼                  │
│                   Kamera                │
└─────────────────────────────────────────┘
```

**Typische Parameter aus dem Experiment:**
- Filmdicke: 100 nm – 2 µm (variiert räumlich)
- Korrelationslänge: ~50-100 µm
- Laser-Wellenlänge: 532 nm (grün) oder 633 nm (rot)
- Blasendurchmesser: 5-20 mm

---

## Branched Flow von Licht

Ein besonders faszinierendes Phänomen, das in Seifenblasen beobachtet wurde, ist der **Branched Flow** – die Verzweigung von Lichtwellen durch die unregelmäßige Dickenvariation des Seifenfilms.

### Physikalischer Hintergrund

Wenn Wellen durch ein Medium mit schwachen, räumlich korrelierten Störungen propagieren, bilden sie charakteristische verzweigte Muster. Dies wurde erstmals 2001 für Elektronen beobachtet (Jura et al., Nature Physics 2007) und 2020 erstmals für sichtbares Licht in Seifenblasen nachgewiesen (Patsyk et al., Nature 2020).

**Bedingungen für Branched Flow:**
- Korrelationslänge der Störungen > Wellenlänge
- Sanfte (nicht abrupte) Variationen im Medium
- Schwache Streuung (kleine Winkeländerungen)

In Seifenblasen wirken die Dickenvariationen des Films (5 nm – mehrere µm) als korreliertes Störpotential für das Licht.

### Mathematische Beschreibung

Die Lichtausbreitung im Film kann durch eine effektive Brechungsindex-Variation beschrieben werden:

$$n_{\text{eff}}(x, y) = n_{\text{film}} \cdot \frac{d(x, y)}{d_0}$$

wobei $d(x,y)$ die lokale Filmdicke ist.

Die Wellengleichung mit diesem Potential führt zu Kaustiken (Fokussierungspunkten), die sich verzweigen:

$$\nabla^2 \psi + k^2 n_{\text{eff}}^2(x,y) \psi = 0$$

**Charakteristische Längen:**
- Erste Verzweigung bei: $L_b \approx \left(\frac{\lambda \cdot l_c^2}{\sigma^2}\right)^{1/3}$
- $l_c$ = Korrelationslänge der Dickenvariation
- $\sigma$ = Standardabweichung der Dicke

### Implementierung für die Simulation

Für eine vollständige Simulation sollten wir Branched Flow als optionales Feature implementieren:

```rust
// src/physics/branched_flow.rs

#[derive(Debug, Clone)]
pub struct BranchedFlowParams {
    /// Korrelationslänge der Dickenvariation (m)
    pub correlation_length: f64,
    
    /// RMS der Dickenvariation (nm)
    pub thickness_rms: f64,
    
    /// Anzahl der Lichtstrahlen für Raytracing
    pub ray_count: u32,
    
    /// Wellenlänge des Lichts (nm)
    pub wavelength: f64,
}

impl Default for BranchedFlowParams {
    fn default() -> Self {
        Self {
            correlation_length: 50e-6,  // 50 µm
            thickness_rms: 50.0,         // 50 nm
            ray_count: 10000,
            wavelength: 532.0,           // Grüner Laser
        }
    }
}
```

### Visualisierung

Branched Flow kann auf zwei Arten visualisiert werden:

1. **Ray Density Map** – Dichteverteilung der Strahlen
2. **Intensity Field** – Wellenoptische Berechnung der Intensität

```wgsl
// Branched flow intensity visualization (Fragment Shader Erweiterung)
fn branched_flow_intensity(uv: vec2<f32>, thickness_map: texture_2d<f32>) -> f32 {
    // Effektiver Brechungsindex aus Dickenvariation
    let thickness = textureSample(thickness_map, sampler, uv).r;
    let n_eff = 1.33 * thickness / 500.0;  // Normiert auf 500nm
    
    // Gradient des effektiven Index (für Strahlablenkung)
    let dx = dpdx(n_eff);
    let dy = dpdy(n_eff);
    
    // Kaustik-Detektion (hohe Gradienten = Fokussierung)
    let caustic_strength = length(vec2(dx, dy)) * 1000.0;
    
    return 1.0 + caustic_strength;
}
```

---

## Signed Distance Functions (SDFs)

SDFs bieten einen alternativen Ansatz zur klassischen Mesh-basierten Darstellung und sind besonders geeignet für:
- Glatte, dynamisch verformbare Oberflächen
- Boolesche Operationen (Vereinigung, Schnitt, Differenz)
- Effizientes Ray-Marching
- Dünnfilm-Effekte auf impliziten Oberflächen

### Grundkonzept

Eine SDF ist eine Funktion $f: \mathbb{R}^3 \to \mathbb{R}$, die für jeden Punkt den vorzeichenbehafteten Abstand zur nächsten Oberfläche liefert:
- $f(p) < 0$: Punkt liegt innerhalb des Objekts
- $f(p) = 0$: Punkt liegt auf der Oberfläche
- $f(p) > 0$: Punkt liegt außerhalb

**SDF einer Kugel (Seifenblase):**
$$f_{\text{sphere}}(p) = |p - c| - r$$

**SDF eines Dünnfilms (Hohlkugel):**
$$f_{\text{shell}}(p) = \left| |p - c| - r \right| - \frac{d}{2}$$

wobei $d$ die Filmdicke ist.

### Vorteile für Seifenblasen-Simulation

1. **Glatte Oberflächen** – Keine Tessellierungs-Artefakte
2. **Dynamische Verformung** – Einfache Modifikation der SDF
3. **Effiziente Kollisionserkennung** – Abstand direkt aus SDF
4. **Dünnfilm-Darstellung** – Natürliche Repräsentation als Hohlkugel-SDF

### Ray-Marching für SDFs

```rust
// Sphere-tracing / Ray-marching
fn ray_march(origin: Vec3, direction: Vec3, sdf: &impl Fn(Vec3) -> f32) -> Option<f32> {
    let mut t = 0.0;
    for _ in 0..MAX_STEPS {
        let p = origin + direction * t;
        let d = sdf(p);
        if d < EPSILON {
            return Some(t);
        }
        if t > MAX_DIST {
            return None;
        }
        t += d;  // Sicherer Schritt basierend auf SDF-Wert
    }
    None
}
```

### Performance-Optimierungen (nach Turitzin)

Das Video von Mike Turitzin zeigt fortgeschrittene Techniken für performante SDF-Engines:

**1. Brick-Map und Brick-Atlas**
- SDF-Werte werden in einem spärlich besetzten 3D-Grid gespeichert
- Nur Regionen nahe der Oberfläche werden gecacht
- Dramatische Speicherreduktion

**2. Geometry Clipmaps**
- Hierarchische LOD-Struktur für große Welten
- Höhere Auflösung in der Nähe der Kamera
- Ermöglicht effiziente inkrementelle Updates

**3. Interpolation gecachter Werte**
- Trilineare Interpolation zwischen Grid-Punkten
- Glatte Oberflächen trotz diskreter Speicherung

### WGSL Compute Shader für SDF-Evaluation

```wgsl
// SDF für Seifenblase mit Dickenvariation
fn soap_bubble_sdf(p: vec3<f32>, params: BubbleParams) -> f32 {
    let center = vec3<f32>(0.0, 0.0, 0.0);
    let dist_to_center = length(p - center);
    
    // Basisform: Kugel
    let sphere_dist = dist_to_center - params.radius;
    
    // Filmdicke (kann räumlich variieren)
    let theta = acos(p.y / dist_to_center);
    let phi = atan2(p.z, p.x);
    let thickness = get_thickness(theta, phi, params.time);
    
    // Hohlkugel-SDF
    return abs(sphere_dist) - thickness * 0.5;
}

fn get_thickness(theta: f32, phi: f32, time: f32) -> f32 {
    // Basis-Dicke + Drainage + Noise für Variation
    let base = 500.0;  // nm
    let drainage = 200.0 * (1.0 - cos(theta));  // Dünner oben
    let noise = fbm_noise(vec2(theta, phi) * 10.0 + time) * 100.0;
    return (base - drainage + noise) * 1e-9;  // Konvertierung zu Metern
}
```

---

## Referenz-Tools

### Ray Optics Simulation (phydemo.app)

Ein exzellentes Open-Source-Tool für 2D-Strahlenoptik:

- **URL**: https://phydemo.app/ray-optics/simulator/
- **GitHub**: https://github.com/ricktu288/ray-optics
- **Lizenz**: Apache 2.0

**Features:**
- Interaktive 2D-Szenen für geometrische Optik
- Unterstützt Reflexion, Brechung, Beugung
- Farbsimulation für wellenlängenabhängige Effekte
- Export als JSON (für programmatischen Zugriff)

**Relevanz für unser Projekt:**
- Kann als Referenz für Ray-Tracing-Algorithmen dienen
- Die Node.js-Version (`dist-node`) könnte für Vorberechnungen genutzt werden
- Gute Inspirationsquelle für UI/UX von Optik-Simulatoren

### Integration mit unserem Projekt

```bash
# Ray Optics als Node-Modul nutzen (optional)
npm install ray-optics-web
```

```javascript
// Beispiel: Szene exportieren für Vergleich
const scene = {
  objects: [
    { type: "laser", position: [0, 0], angle: 0 },
    { type: "glass", shape: "sphere", radius: 0.05, n: 1.33 }
  ]
};
```

---

## Referenzen

### Primärliteratur: Oberflächenminimierung & Netzwerk-Physik

- **Meng, Piazza, Both, Barzel & Barabási (2026)**: "Surface optimization governs the local design of physical networks", *Nature* 649, 315-322
  - DOI: [10.1038/s41586-025-09784-4](https://doi.org/10.1038/s41586-025-09784-4)
  - **Schlüsselpaper**: Zeigt exakte Abbildung zwischen Oberflächenminimierung und String-Theorie (Nambu-Goto-Wirkung)
  - Erklärt warum reale Netzwerke (Neuronen, Blutgefäße, Korallen) von Steiner-Vorhersagen abweichen
  - **Code**: https://github.com/Barabasi-Lab/min-surf-netw
  - **Daten**: https://physical.network
  - Kernaussagen:
    - Phasenübergang bei χ ≈ 0.83: Bifurkation → Trifurkation
    - ~15% der Knoten in realen Netzwerken sind Trifurkationen (k=4)
    - Orthogonale Sprossen (90°) sind stabile Lösungen der Oberflächenminimierung
    - 98% der neuronalen Sprossen enden in Synapsen → funktionale Optimierung

### Primärliteratur: Branched Flow

- **Jura et al. (2007)**: "Unexpected features of branched flow through high-mobility two-dimensional electron gases", *Nature Physics* 3, 841-845
  - DOI: [10.1038/nphys756](https://www.nature.com/articles/nphys756)
  - Grundlagenarbeit zu Branched Flow in Elektronengasen
  
- **Patsyk et al. (2020)**: "Observation of branched flow of light", *Nature* 583, 60-65
  - DOI: [10.1038/s41586-020-2376-8](https://www.nature.com/articles/s41586-020-2376-8)
  - **Schlüsselpaper**: Erstmaliger Nachweis von Branched Flow in Seifenblasen mit Laserlicht
  - Enthält experimentelle Daten zu Dickenvariationen und Intensitätsverteilungen

### Seifenblasen-Physik & Simulation

- Huang et al. (2020): "Chemomechanical simulation of soap film flow on spherical bubbles", *ACM TOG*
- Glassner (2000): "Soap Bubbles: Part 1 & 2", *IEEE CG&A*
- Durikovič (2001): "Animation of Soap Bubble Dynamics, Cluster Formation and Collision"

### String-Theorie & Minimal-Oberflächen

Die Verbindung zwischen Netzwerk-Optimierung und Seifenblasen läuft über die String-Theorie:

- Witten (1986): "Non-commutative geometry and string field theory", *Nucl. Phys. B* 268, 253-294
- Carlip (1988): "Quadratic differentials and closed string vertices", *Phys. Lett. B* 214, 187-192
- Saadi & Zwiebach (1989): "Closed string field theory from polyhedra", *Ann. Phys.* 192, 213-227
- Tong: "Lectures on String Theory" – http://www.damtp.cam.ac.uk/user/tong/string.html
  - Herleitung der Nambu-Goto-Wirkung und ihrer klassischen Lösungen

### Dünnfilm-Optik

- Born & Wolf: "Principles of Optics" (Kapitel zu Interferenz)
- Macleod: "Thin-Film Optical Filters"

### Bücher

- de Gennes et al.: "Capillarity and Wetting Phenomena"
- Isenberg: "The Science of Soap Films and Soap Bubbles"

### Video-Ressourcen

- **Mike Turitzin: SDF-basierte Game Engine**
  - YouTube: https://www.youtube.com/watch?v=il-TXbn5iMA
  - Behandelt Performance-Optimierungen für SDF-Rendering:
    - Brick-Maps und spärliche Speicherung
    - Geometry Clipmaps für große Welten
    - Inkrementelle Updates für dynamische Geometrie
    - Physik und Kollisionserkennung mit SDFs
  - Kapitelübersicht:
    - 0:00 - Intro und Motivation
    - 1:03 - Kurze Einführung in SDFs
    - 1:56 - Ray-Marching Performance-Probleme
    - 4:15 - Caching von SDF-Distanzen
    - 8:43 - Spärliches Caching
    - 9:16 - Brick-Map und Brick-Atlas
    - 11:44 - Geometry Clipmaps
    - 14:29 - Physik und Kollision

- **Sebastian Lague: Coding Adventure - Ray Marching**
  - YouTube: https://www.youtube.com/watch?v=Cp5WWtMoeKg
  - Einsteiger-freundliche Einführung in SDFs und Ray-Marching

- **Nature Video: Branched Flow of Light**
  - Nature News: https://www.nature.com/articles/d41586-020-01991-5
  - Experimentelle Aufnahmen von Branched Flow in Seifenblasen

### Code-Ressourcen

- **min-surf-netw (Barabási Lab)**: https://github.com/Barabasi-Lab/min-surf-netw
  - Python-Package für Oberflächenminimierung in Netzwerken
  - Basiert auf String-Theorie (Strebel-Theorem)
  - Nützlich für Blasen-Cluster-Geometrie
- **Physical Network Dataset**: https://physical.network
  - 3D-Daten von Neuronen, Blutgefäßen, Korallen, Bäumen
  - Validierungsdaten für Verzweigungswinkel und Trifurkationen
- **wgpu-rs Examples**: https://github.com/gfx-rs/wgpu/tree/trunk/examples
- **Learn WGPU**: https://sotrh.github.io/learn-wgpu/
- **Surface Evolver**: https://kenbrakke.com/evolver/
- **Ray Optics Simulation**: https://phydemo.app/ray-optics/
  - GitHub: https://github.com/ricktu288/ray-optics
  - DOI: 10.5281/zenodo.6386611
- **tmm (Python)**: Transfer Matrix Method für Dünnfilm-Optik
  - https://pypi.org/project/tmm/
- **Jolt Physics**: https://github.com/jrouwe/JoltPhysics
  - Hochperformante Physik-Engine, kompatibel mit SDF-Kollisionen

### SDF-Ressourcen

- **Inigo Quilez's Artikel**: https://iquilezles.org/articles/
  - Umfassende Sammlung zu SDFs, Ray-Marching, Shading
  - Besonders relevant: "Distance Functions", "Smooth Minimum"
  
- **SDF Tracing Visualization (Shadertoy)**: https://www.shadertoy.com/view/lslXD8
  - Interaktive Visualisierung von Sphere-Tracing
  
- **"Improved Alpha-Tested Magnification for Vector Textures and Special Effects"**
  - Chris Green (Valve), 2007
  - https://steamcdn-a.akamaihd.net/apps/valve/2007/SIGGRAPH2007_AlphaTestedMagnification.pdf
  
- **"Ray Tracing of Signed Distance Function Grids"**
  - Söderlund et al. (NVIDIA), 2022
  - https://jcgt.org/published/0011/03/06/
  - Optimiertes Ray-Tracing für gecachte SDF-Grids
  
- **"Geometry Clipmaps: Terrain Rendering Using Nested Regular Grids"**
  - Losasso & Hoppe, 2004
  - https://hhoppe.com/geomclipmap.pdf
  - Hierarchische LOD-Struktur für große Welten

---

## Nächste Schritte

1. **Projektinitialisierung**: `cargo new soap-bubble-sim`
2. **Dependencies hinzufügen**: Cargo.toml wie oben
3. **Basis-Fenster**: wgpu + winit Setup
4. **Sphären-Mesh**: Icosphere oder UV-Sphere generieren
5. **Erster Shader**: Statische Interferenzfarben testen

Bei Fragen oder für die Implementierung einzelner Module einfach anfragen!
