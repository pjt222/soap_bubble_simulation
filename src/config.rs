//! Configuration module for soap bubble simulation parameters.
//!
//! This module defines the parameter structures for the soap bubble simulation,
//! including bubble geometry, fluid properties, and environmental conditions.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Parameters defining the physical properties of the soap bubble.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BubbleParameters {
    /// Diameter of the bubble in meters
    pub diameter: f64,

    /// Initial film thickness in nanometers
    pub film_thickness_nm: f64,

    /// Critical film thickness before bursting (nm)
    pub critical_thickness_nm: f64,

    /// Refractive index of the soap film (~1.33 for water-based)
    pub refractive_index: f64,
}

impl Default for BubbleParameters {
    fn default() -> Self {
        Self {
            diameter: 0.05,              // 5 cm
            film_thickness_nm: 500.0,    // 500 nm
            critical_thickness_nm: 10.0, // 10 nm
            refractive_index: 1.33,
        }
    }
}

/// Parameters defining the fluid properties of the soap solution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluidParameters {
    /// Dynamic viscosity in Pa*s (Pascal-seconds)
    pub viscosity: f64,

    /// Surface tension in N/m (Newtons per meter)
    pub surface_tension: f64,

    /// Density in kg/m^3
    pub density: f64,

    /// Surfactant diffusion coefficient in m^2/s
    pub surfactant_diffusion: f64,

    /// Surfactant concentration (relative, 0.0 to 1.0)
    pub surfactant_concentration: f64,
}

impl Default for FluidParameters {
    fn default() -> Self {
        Self {
            viscosity: 0.001,             // ~water viscosity
            surface_tension: 0.025,       // soap solution
            density: 1000.0,              // kg/m^3
            surfactant_diffusion: 1e-9,   // m^2/s
            surfactant_concentration: 0.5,
        }
    }
}

/// Parameters defining the environmental conditions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentParameters {
    /// Gravitational acceleration in m/s^2
    pub gravity: f64,

    /// Ambient temperature in Kelvin
    pub temperature: f64,

    /// Atmospheric pressure in Pascal
    pub pressure: f64,

    /// Wind velocity in m/s (x, y, z components)
    #[serde(default)]
    pub wind: [f64; 3],

    /// Buoyancy strength multiplier (default 1.0)
    #[serde(default = "default_buoyancy")]
    pub buoyancy: f64,
}

fn default_buoyancy() -> f64 {
    1.0
}

impl Default for EnvironmentParameters {
    fn default() -> Self {
        Self {
            gravity: 9.81,       // Earth gravity
            temperature: 293.15, // 20 degrees Celsius
            pressure: 101325.0,  // 1 atm
            wind: [0.0, 0.0, 0.0], // No wind by default
            buoyancy: 1.0,       // Normal buoyancy
        }
    }
}

/// Parameters for multi-bubble foam simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoamParameters {
    /// Enable multi-bubble foam mode
    #[serde(default)]
    pub enabled: bool,

    /// Maximum number of bubbles
    #[serde(default = "default_max_bubbles")]
    pub max_bubbles: u32,

    /// Initial number of bubbles
    #[serde(default = "default_initial_bubbles")]
    pub initial_count: u32,

    /// Van der Waals attraction strength
    #[serde(default = "default_vdw_strength")]
    pub van_der_waals_strength: f64,

    /// Enable bubble coalescence (merging)
    #[serde(default)]
    pub coalescence_enabled: bool,

    /// Plateau's rules enforcement strength (0.0 to 1.0)
    #[serde(default = "default_plateau_stiffness")]
    pub plateau_stiffness: f64,

    /// Minimum bubble radius before removal (meters)
    #[serde(default = "default_min_radius")]
    pub min_radius: f64,

    /// Maximum bubble radius (meters)
    #[serde(default = "default_max_radius")]
    pub max_radius: f64,

    /// Physics simulation time scale multiplier
    #[serde(default = "default_time_scale")]
    pub time_scale: f64,

    // Generation parameters
    /// Positioning mode (0=Random, 1=SimpleCubic, 2=BCC, 3=FCC, 4=HCP, 5=PoissonDisk)
    #[serde(default)]
    pub positioning_mode: u8,

    /// Spacing between grid positions (meters)
    #[serde(default = "default_spacing")]
    pub spacing: f64,

    /// Jitter amount for lattice positions (0.0 to 1.0)
    #[serde(default = "default_jitter")]
    pub jitter: f64,

    /// Size distribution (0=Uniform, 1=Normal, 2=LogNormal, 3=SchulzFlory, 4=Bimodal)
    #[serde(default)]
    pub size_distribution: u8,

    /// Mean radius for normal/log-normal distributions (meters)
    #[serde(default = "default_mean_radius")]
    pub mean_radius: f64,

    /// Standard deviation for normal distribution (meters)
    #[serde(default = "default_std_dev")]
    pub std_dev: f64,

    /// Sigma parameter for log-normal distribution
    #[serde(default = "default_sigma")]
    pub sigma: f64,

    /// Polydispersity index for Schulz-Flory (PDI = Mw/Mn, must be > 1)
    #[serde(default = "default_pdi")]
    pub pdi: f64,

    /// Ratio of first peak for bimodal distribution (0.0 to 1.0)
    #[serde(default = "default_bimodal_ratio")]
    pub bimodal_ratio: f64,

    /// Mean of second peak for bimodal distribution (meters)
    #[serde(default = "default_bimodal_mean2")]
    pub bimodal_mean2: f64,

    /// Standard deviation of second peak for bimodal distribution (meters)
    #[serde(default = "default_bimodal_std2")]
    pub bimodal_std2: f64,
}

fn default_max_bubbles() -> u32 {
    64
}

fn default_initial_bubbles() -> u32 {
    5
}

fn default_vdw_strength() -> f64 {
    1e-12
}

fn default_plateau_stiffness() -> f64 {
    1.0
}

fn default_min_radius() -> f64 {
    0.005 // 5mm
}

fn default_max_radius() -> f64 {
    0.1 // 10cm
}

fn default_time_scale() -> f64 {
    1.0
}

fn default_spacing() -> f64 {
    0.05 // 5cm
}

fn default_jitter() -> f64 {
    0.1
}

fn default_mean_radius() -> f64 {
    0.022 // 2.2cm
}

fn default_std_dev() -> f64 {
    0.005 // 5mm
}

fn default_sigma() -> f64 {
    0.3
}

fn default_pdi() -> f64 {
    1.5
}

fn default_bimodal_ratio() -> f64 {
    0.5
}

fn default_bimodal_mean2() -> f64 {
    0.028 // 2.8cm
}

fn default_bimodal_std2() -> f64 {
    0.003 // 3mm
}

impl Default for FoamParameters {
    fn default() -> Self {
        Self {
            enabled: false,
            max_bubbles: default_max_bubbles(),
            initial_count: default_initial_bubbles(),
            van_der_waals_strength: default_vdw_strength(),
            coalescence_enabled: false,
            plateau_stiffness: default_plateau_stiffness(),
            min_radius: default_min_radius(),
            max_radius: default_max_radius(),
            time_scale: default_time_scale(),
            positioning_mode: 0,
            spacing: default_spacing(),
            jitter: default_jitter(),
            size_distribution: 0,
            mean_radius: default_mean_radius(),
            std_dev: default_std_dev(),
            sigma: default_sigma(),
            pdi: default_pdi(),
            bimodal_ratio: default_bimodal_ratio(),
            bimodal_mean2: default_bimodal_mean2(),
            bimodal_std2: default_bimodal_std2(),
        }
    }
}

/// Complete simulation configuration combining all parameter groups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Bubble geometry and optical properties
    pub bubble: BubbleParameters,

    /// Fluid dynamics parameters
    pub fluid: FluidParameters,

    /// Environmental conditions
    pub environment: EnvironmentParameters,

    /// Multi-bubble foam parameters
    #[serde(default)]
    pub foam: FoamParameters,

    /// Simulation time step in seconds
    pub dt: f64,

    /// Simulation grid resolution (vertices on sphere)
    pub resolution: u32,
}

// put id:'cfg_default', label:'Default config', output:'final_config.internal'
impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            bubble: BubbleParameters::default(),
            fluid: FluidParameters::default(),
            environment: EnvironmentParameters::default(),
            foam: FoamParameters::default(),
            dt: 0.001,       // 1 ms time step
            resolution: 128, // 128x256 grid
        }
    }
}

impl SimulationConfig {
    /// Load configuration from a JSON file.
    ///
    /// # Arguments
    /// * `path` - Path to the JSON configuration file
    ///
    /// # Returns
    /// * `Ok(SimulationConfig)` - Parsed configuration
    /// * `Err` - If file cannot be read or parsed
    // put id:'cfg_from_file', label:'Parse JSON config', input:'config.json', output:'final_config.internal'
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let contents = fs::read_to_string(path.as_ref()).map_err(|error| ConfigError::Io {
            path: path.as_ref().to_path_buf(),
            error,
        })?;
        serde_json::from_str(&contents).map_err(|error| ConfigError::Parse {
            path: path.as_ref().to_path_buf(),
            error,
        })
    }

    /// Save configuration to a JSON file.
    ///
    /// # Arguments
    /// * `path` - Path to write the JSON configuration file
    ///
    /// # Returns
    /// * `Ok(())` - If successfully written
    /// * `Err` - If file cannot be written
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), ConfigError> {
        let contents =
            serde_json::to_string_pretty(self).map_err(|error| ConfigError::Serialize { error })?;
        fs::write(path.as_ref(), contents).map_err(|error| ConfigError::Io {
            path: path.as_ref().to_path_buf(),
            error,
        })
    }

    /// Calculate the bubble radius in meters.
    pub fn bubble_radius(&self) -> f64 {
        self.bubble.diameter / 2.0
    }

    /// Calculate the internal pressure difference using Young-Laplace equation.
    /// Delta P = 4 * gamma / R (factor 4 because of two interfaces)
    pub fn internal_pressure_difference(&self) -> f64 {
        4.0 * self.fluid.surface_tension / self.bubble_radius()
    }

    /// Convert film thickness from nanometers to meters.
    pub fn film_thickness_meters(&self) -> f64 {
        self.bubble.film_thickness_nm * 1e-9
    }
}

/// Error types for configuration operations.
#[derive(Debug)]
pub enum ConfigError {
    /// IO error when reading or writing configuration files
    Io {
        path: std::path::PathBuf,
        error: std::io::Error,
    },
    /// JSON parsing error
    Parse {
        path: std::path::PathBuf,
        error: serde_json::Error,
    },
    /// JSON serialization error
    Serialize { error: serde_json::Error },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Io { path, error } => {
                write!(
                    formatter,
                    "Failed to read/write config file '{}': {}",
                    path.display(),
                    error
                )
            }
            ConfigError::Parse { path, error } => {
                write!(
                    formatter,
                    "Failed to parse config file '{}': {}",
                    path.display(),
                    error
                )
            }
            ConfigError::Serialize { error } => {
                write!(formatter, "Failed to serialize config: {}", error)
            }
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ConfigError::Io { error, .. } => Some(error),
            ConfigError::Parse { error, .. } => Some(error),
            ConfigError::Serialize { error } => Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SimulationConfig::default();
        assert!((config.bubble.diameter - 0.05).abs() < f64::EPSILON);
        assert!((config.bubble.film_thickness_nm - 500.0).abs() < f64::EPSILON);
        assert!((config.fluid.viscosity - 0.001).abs() < f64::EPSILON);
        assert!((config.environment.gravity - 9.81).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bubble_radius() {
        let config = SimulationConfig::default();
        assert!((config.bubble_radius() - 0.025).abs() < f64::EPSILON);
    }

    #[test]
    fn test_internal_pressure() {
        let config = SimulationConfig::default();
        let expected_pressure = 4.0 * 0.025 / 0.025; // 4 Pa
        assert!((config.internal_pressure_difference() - expected_pressure).abs() < 1e-10);
    }

    #[test]
    fn test_film_thickness_conversion() {
        let config = SimulationConfig::default();
        assert!((config.film_thickness_meters() - 500e-9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let config = SimulationConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: SimulationConfig = serde_json::from_str(&json).unwrap();
        assert!((config.bubble.diameter - deserialized.bubble.diameter).abs() < f64::EPSILON);
    }
}
