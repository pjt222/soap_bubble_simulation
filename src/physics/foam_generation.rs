//! Foam bubble generation with structured positioning and size distributions.
//!
//! Provides algorithms for generating bubble clusters with various spatial
//! arrangements (lattices, Poisson disk) and size distributions (normal,
//! log-normal, Schulz-Flory, bimodal).

use super::foam::{BubbleCluster};
use glam::Vec3;
use std::sync::atomic::{AtomicU64, Ordering};

/// Positioning mode for foam bubble generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum PositioningMode {
    /// Random positioning within bounding volume
    #[default]
    Random = 0,
    /// Simple cubic lattice
    SimpleCubic = 1,
    /// Body-centered cubic lattice
    BodyCenteredCubic = 2,
    /// Face-centered cubic lattice
    FaceCenteredCubic = 3,
    /// Hexagonal close-packed lattice
    HexagonalClosePacked = 4,
    /// Poisson disk sampling (blue noise)
    PoissonDisk = 5,
}

impl PositioningMode {
    /// Convert from u8 value
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Random,
            1 => Self::SimpleCubic,
            2 => Self::BodyCenteredCubic,
            3 => Self::FaceCenteredCubic,
            4 => Self::HexagonalClosePacked,
            5 => Self::PoissonDisk,
            _ => Self::Random,
        }
    }

    /// Get display name for UI
    pub fn name(&self) -> &'static str {
        match self {
            Self::Random => "Random",
            Self::SimpleCubic => "Simple Cubic",
            Self::BodyCenteredCubic => "Body-Centered Cubic",
            Self::FaceCenteredCubic => "Face-Centered Cubic",
            Self::HexagonalClosePacked => "Hexagonal Close-Packed",
            Self::PoissonDisk => "Poisson Disk",
        }
    }

    /// Get all positioning modes
    pub fn all() -> &'static [Self] {
        &[
            Self::Random,
            Self::SimpleCubic,
            Self::BodyCenteredCubic,
            Self::FaceCenteredCubic,
            Self::HexagonalClosePacked,
            Self::PoissonDisk,
        ]
    }
}

/// Size distribution for bubble radii.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum SizeDistribution {
    /// Uniform distribution between min and max
    #[default]
    Uniform = 0,
    /// Normal (Gaussian) distribution
    Normal = 1,
    /// Log-normal distribution
    LogNormal = 2,
    /// Schulz-Flory distribution (polymer-like polydispersity)
    SchulzFlory = 3,
    /// Bimodal distribution (mixture of two normals)
    Bimodal = 4,
}

impl SizeDistribution {
    /// Convert from u8 value
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Uniform,
            1 => Self::Normal,
            2 => Self::LogNormal,
            3 => Self::SchulzFlory,
            4 => Self::Bimodal,
            _ => Self::Uniform,
        }
    }

    /// Get display name for UI
    pub fn name(&self) -> &'static str {
        match self {
            Self::Uniform => "Uniform",
            Self::Normal => "Normal",
            Self::LogNormal => "Log-Normal",
            Self::SchulzFlory => "Schulz-Flory",
            Self::Bimodal => "Bimodal",
        }
    }

    /// Get all size distributions
    pub fn all() -> &'static [Self] {
        &[
            Self::Uniform,
            Self::Normal,
            Self::LogNormal,
            Self::SchulzFlory,
            Self::Bimodal,
        ]
    }
}

/// Parameters for foam generation.
// put id:'cpu_foam_gen', label:'Generate foam cluster', input:'final_config.internal', output:'foam_state.internal'
#[derive(Debug, Clone)]
pub struct GenerationParams {
    // Positioning parameters
    /// Positioning mode (lattice type or random)
    pub positioning_mode: PositioningMode,
    /// Number of bubbles to generate
    pub bubble_count: u32,
    /// Spacing between grid points (for lattice modes)
    pub spacing: f32,
    /// Random jitter applied to lattice positions (0.0 to 1.0)
    pub jitter: f32,

    // Size distribution parameters
    /// Size distribution type
    pub size_distribution: SizeDistribution,
    /// Minimum bubble radius (meters)
    pub min_radius: f32,
    /// Maximum bubble radius (meters)
    pub max_radius: f32,
    /// Mean radius for normal/log-normal distributions
    pub mean_radius: f32,
    /// Standard deviation for normal distribution
    pub std_dev: f32,
    /// Sigma parameter for log-normal distribution
    pub sigma: f32,
    /// Polydispersity index for Schulz-Flory (PDI = Mw/Mn)
    pub pdi: f32,

    // Bimodal parameters
    /// Ratio of first peak (0.0 to 1.0)
    pub bimodal_ratio: f32,
    /// Mean of second peak for bimodal
    pub bimodal_mean2: f32,
    /// Std dev of second peak for bimodal
    pub bimodal_std2: f32,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            positioning_mode: PositioningMode::Random,
            bubble_count: 5,
            spacing: 0.05,
            jitter: 0.1,

            size_distribution: SizeDistribution::Uniform,
            min_radius: 0.015,
            max_radius: 0.030,
            mean_radius: 0.022,
            std_dev: 0.005,
            sigma: 0.3,
            pdi: 1.5,

            bimodal_ratio: 0.5,
            bimodal_mean2: 0.028,
            bimodal_std2: 0.003,
        }
    }
}

/// Foam generator with configurable positioning and size distributions.
pub struct FoamGenerator {
    params: GenerationParams,
}

impl FoamGenerator {
    /// Create a new foam generator with given parameters.
    pub fn new(params: GenerationParams) -> Self {
        Self { params }
    }

    /// Generate a bubble cluster using current parameters.
    pub fn generate(&self, surface_tension: f32) -> BubbleCluster {
        let mut cluster = BubbleCluster::new(surface_tension);

        // Generate positions based on positioning mode
        let positions = self.generate_positions();

        // Add bubbles at generated positions
        for position in positions {
            let radius = self.sample_radius();
            cluster.add_bubble(position, radius);
        }

        cluster.update_connections();
        cluster
    }

    /// Generate positions based on positioning mode.
    fn generate_positions(&self) -> Vec<Vec3> {
        match self.params.positioning_mode {
            PositioningMode::Random => self.generate_random_positions(),
            PositioningMode::SimpleCubic => self.generate_simple_cubic(),
            PositioningMode::BodyCenteredCubic => self.generate_bcc(),
            PositioningMode::FaceCenteredCubic => self.generate_fcc(),
            PositioningMode::HexagonalClosePacked => self.generate_hcp(),
            PositioningMode::PoissonDisk => self.generate_poisson_disk(),
        }
    }

    /// Generate random positions within a sphere.
    fn generate_random_positions(&self) -> Vec<Vec3> {
        use std::f32::consts::PI;
        let mut positions = Vec::with_capacity(self.params.bubble_count as usize);
        let spawn_radius = self.params.spacing * 2.0;

        for _ in 0..self.params.bubble_count {
            // Random spherical coordinates
            let theta = rand_f32() * 2.0 * PI;
            let phi = rand_f32() * PI;
            let r = rand_f32().cbrt() * spawn_radius; // Cube root for uniform volume distribution

            let position = Vec3::new(
                r * phi.sin() * theta.cos(),
                r * phi.cos(),
                r * phi.sin() * theta.sin(),
            );
            positions.push(position);
        }
        positions
    }

    /// Generate simple cubic lattice positions.
    fn generate_simple_cubic(&self) -> Vec<Vec3> {
        let mut positions = Vec::new();
        let n = (self.params.bubble_count as f32).cbrt().ceil() as i32;
        let spacing = self.params.spacing;
        let offset = (n as f32 - 1.0) * spacing / 2.0;

        'outer: for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    if positions.len() >= self.params.bubble_count as usize {
                        break 'outer;
                    }
                    let base = Vec3::new(
                        i as f32 * spacing - offset,
                        j as f32 * spacing - offset,
                        k as f32 * spacing - offset,
                    );
                    let jitter = self.apply_jitter();
                    positions.push(base + jitter);
                }
            }
        }
        positions
    }

    /// Generate body-centered cubic lattice positions.
    fn generate_bcc(&self) -> Vec<Vec3> {
        let mut positions = Vec::new();
        // BCC: simple cubic + center of each cube
        let n = ((self.params.bubble_count as f32 / 2.0).cbrt().ceil() as i32).max(2);
        let spacing = self.params.spacing;
        let offset = (n as f32 - 1.0) * spacing / 2.0;

        'outer: for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    if positions.len() >= self.params.bubble_count as usize {
                        break 'outer;
                    }
                    // Corner atom
                    let corner = Vec3::new(
                        i as f32 * spacing - offset,
                        j as f32 * spacing - offset,
                        k as f32 * spacing - offset,
                    );
                    positions.push(corner + self.apply_jitter());

                    if positions.len() >= self.params.bubble_count as usize {
                        break 'outer;
                    }
                    // Body center
                    let center = corner + Vec3::splat(spacing / 2.0);
                    positions.push(center + self.apply_jitter());
                }
            }
        }
        positions.truncate(self.params.bubble_count as usize);
        positions
    }

    /// Generate face-centered cubic lattice positions.
    fn generate_fcc(&self) -> Vec<Vec3> {
        let mut positions = Vec::new();
        // FCC: 4 atoms per unit cell
        let n = ((self.params.bubble_count as f32 / 4.0).cbrt().ceil() as i32).max(2);
        let spacing = self.params.spacing;
        let offset = (n as f32 - 1.0) * spacing / 2.0;
        let half = spacing / 2.0;

        'outer: for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let base = Vec3::new(
                        i as f32 * spacing - offset,
                        j as f32 * spacing - offset,
                        k as f32 * spacing - offset,
                    );

                    // FCC basis: corner + 3 face centers
                    let fcc_offsets = [
                        Vec3::ZERO,
                        Vec3::new(half, half, 0.0),
                        Vec3::new(half, 0.0, half),
                        Vec3::new(0.0, half, half),
                    ];

                    for fcc_offset in &fcc_offsets {
                        if positions.len() >= self.params.bubble_count as usize {
                            break 'outer;
                        }
                        positions.push(base + *fcc_offset + self.apply_jitter());
                    }
                }
            }
        }
        positions.truncate(self.params.bubble_count as usize);
        positions
    }

    /// Generate hexagonal close-packed lattice positions.
    fn generate_hcp(&self) -> Vec<Vec3> {
        let mut positions = Vec::new();
        // HCP: ABAB stacking
        let n = ((self.params.bubble_count as f32 / 2.0).cbrt().ceil() as i32).max(2);
        let spacing = self.params.spacing;
        let offset_x = (n as f32 - 1.0) * spacing / 2.0;
        let offset_y = (n as f32 - 1.0) * spacing * (2.0_f32 / 3.0).sqrt() / 2.0;
        let offset_z = (n as f32 - 1.0) * spacing / 2.0;

        let layer_height = spacing * (2.0_f32 / 3.0).sqrt();

        'outer: for layer in 0..n {
            let y = layer as f32 * layer_height - offset_y;
            let is_b_layer = layer % 2 == 1;

            for i in 0..n {
                for j in 0..n {
                    if positions.len() >= self.params.bubble_count as usize {
                        break 'outer;
                    }

                    let mut x = i as f32 * spacing - offset_x;
                    let mut z = j as f32 * spacing - offset_z;

                    // Offset every other row in x
                    if j % 2 == 1 {
                        x += spacing / 2.0;
                    }

                    // B layers are offset
                    if is_b_layer {
                        x += spacing / 2.0;
                        z += spacing / (2.0 * 3.0_f32.sqrt());
                    }

                    positions.push(Vec3::new(x, y, z) + self.apply_jitter());
                }
            }
        }
        positions.truncate(self.params.bubble_count as usize);
        positions
    }

    /// Generate Poisson disk sampling positions (Bridson's algorithm).
    fn generate_poisson_disk(&self) -> Vec<Vec3> {
        let min_distance = self.params.spacing;
        let max_attempts = 30;
        let bounds = min_distance * 3.0; // Approximate bounding box

        let mut positions: Vec<Vec3> = Vec::new();
        let mut active: Vec<usize> = Vec::new();

        // Start with a random point at center
        let initial = Vec3::ZERO;
        positions.push(initial);
        active.push(0);

        while !active.is_empty() && positions.len() < self.params.bubble_count as usize {
            // Pick a random active point
            let active_idx = (rand_f32() * active.len() as f32) as usize % active.len();
            let point_idx = active[active_idx];
            let point = positions[point_idx];

            let mut found = false;

            for _ in 0..max_attempts {
                // Generate random point in annulus [r, 2r] around the point
                let theta = rand_f32() * std::f32::consts::TAU;
                let phi = (rand_f32() * 2.0 - 1.0).acos();
                let r = min_distance * (1.0 + rand_f32());

                let candidate = point + Vec3::new(
                    r * phi.sin() * theta.cos(),
                    r * phi.cos(),
                    r * phi.sin() * theta.sin(),
                );

                // Check bounds
                if candidate.x.abs() > bounds
                    || candidate.y.abs() > bounds
                    || candidate.z.abs() > bounds
                {
                    continue;
                }

                // Check distance to all existing points
                let mut valid = true;
                for existing in &positions {
                    if (*existing - candidate).length() < min_distance {
                        valid = false;
                        break;
                    }
                }

                if valid {
                    positions.push(candidate + self.apply_jitter() * 0.1); // Smaller jitter for Poisson
                    active.push(positions.len() - 1);
                    found = true;
                    break;
                }
            }

            if !found {
                active.swap_remove(active_idx);
            }
        }

        positions.truncate(self.params.bubble_count as usize);
        positions
    }

    /// Apply random jitter to a position.
    fn apply_jitter(&self) -> Vec3 {
        if self.params.jitter <= 0.0 {
            return Vec3::ZERO;
        }
        let scale = self.params.jitter * self.params.spacing;
        Vec3::new(
            (rand_f32() - 0.5) * scale,
            (rand_f32() - 0.5) * scale,
            (rand_f32() - 0.5) * scale,
        )
    }

    /// Sample a radius from the configured distribution.
    fn sample_radius(&self) -> f32 {
        let radius = match self.params.size_distribution {
            SizeDistribution::Uniform => self.sample_uniform(),
            SizeDistribution::Normal => self.sample_normal(),
            SizeDistribution::LogNormal => self.sample_log_normal(),
            SizeDistribution::SchulzFlory => self.sample_schulz_flory(),
            SizeDistribution::Bimodal => self.sample_bimodal(),
        };
        // Clamp to valid range
        radius.clamp(self.params.min_radius, self.params.max_radius)
    }

    /// Sample from uniform distribution.
    fn sample_uniform(&self) -> f32 {
        self.params.min_radius + rand_f32() * (self.params.max_radius - self.params.min_radius)
    }

    /// Sample from normal distribution using Box-Muller transform.
    fn sample_normal(&self) -> f32 {
        let (z0, _) = box_muller();
        self.params.mean_radius + z0 * self.params.std_dev
    }

    /// Sample from log-normal distribution.
    fn sample_log_normal(&self) -> f32 {
        // Log-normal: X = exp(mu + sigma * Z) where Z ~ N(0,1)
        // Mean of log-normal is exp(mu + sigma^2/2), so:
        // mu = ln(mean) - sigma^2/2
        let sigma = self.params.sigma;
        let mu = self.params.mean_radius.ln() - sigma * sigma / 2.0;
        let (z, _) = box_muller();
        (mu + sigma * z).exp()
    }

    /// Sample from Schulz-Flory distribution using gamma distribution.
    /// PDI (polydispersity index) = Mw/Mn, related to shape parameter k = 1/(PDI-1)
    fn sample_schulz_flory(&self) -> f32 {
        // Schulz distribution is a gamma distribution
        // k = 1/(PDI - 1), theta = mean * (PDI - 1)
        let pdi = self.params.pdi.max(1.01); // PDI must be > 1
        let k = 1.0 / (pdi - 1.0);
        let theta = self.params.mean_radius * (pdi - 1.0);

        sample_gamma(k, theta)
    }

    /// Sample from bimodal distribution (mixture of two normals).
    fn sample_bimodal(&self) -> f32 {
        if rand_f32() < self.params.bimodal_ratio {
            // First peak
            let (z, _) = box_muller();
            self.params.mean_radius + z * self.params.std_dev
        } else {
            // Second peak
            let (z, _) = box_muller();
            self.params.bimodal_mean2 + z * self.params.bimodal_std2
        }
    }
}

// ============================================================================
// Random number generation utilities
// ============================================================================

/// Xorshift64 PRNG state.
static SEED: AtomicU64 = AtomicU64::new(88172645463325252);

/// Generate a random f32 in [0, 1).
pub fn rand_f32() -> f32 {
    let mut s = SEED.load(Ordering::Relaxed);
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    SEED.store(s, Ordering::Relaxed);
    // Use upper bits for better distribution
    ((s >> 40) as f32) / ((1u64 << 24) as f32)
}

/// Reset the random seed.
pub fn set_seed(seed: u64) {
    SEED.store(seed.max(1), Ordering::Relaxed);
}

/// Box-Muller transform: generate two independent standard normal samples.
fn box_muller() -> (f32, f32) {
    let u1 = rand_f32().max(1e-10); // Avoid log(0)
    let u2 = rand_f32();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f32::consts::TAU * u2;
    (r * theta.cos(), r * theta.sin())
}

/// Sample from gamma distribution using Marsaglia-Tsang method.
fn sample_gamma(shape: f32, scale: f32) -> f32 {
    if shape < 1.0 {
        // For shape < 1, use: Gamma(a) = Gamma(a+1) * U^(1/a)
        let u = rand_f32().max(1e-10);
        return sample_gamma(shape + 1.0, scale) * u.powf(1.0 / shape);
    }

    // Marsaglia-Tsang method for shape >= 1
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        let (x, _) = box_muller();
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }

        let v = v * v * v;
        let u = rand_f32();

        // Quick check
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v * scale;
        }

        // Slower check
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positioning_modes() {
        for mode in PositioningMode::all() {
            let params = GenerationParams {
                positioning_mode: *mode,
                bubble_count: 10,
                ..Default::default()
            };
            let generator = FoamGenerator::new(params);
            let cluster = generator.generate(0.025);
            assert!(cluster.len() <= 10);
            assert!(cluster.len() > 0);
        }
    }

    #[test]
    fn test_size_distributions() {
        for dist in SizeDistribution::all() {
            let params = GenerationParams {
                size_distribution: *dist,
                bubble_count: 20,
                ..Default::default()
            };
            let min_radius = params.min_radius;
            let max_radius = params.max_radius;
            let generator = FoamGenerator::new(params);
            let cluster = generator.generate(0.025);

            // Check all radii are within bounds
            for bubble in cluster.bubbles() {
                assert!(bubble.radius >= min_radius * 0.99);
                assert!(bubble.radius <= max_radius * 1.01);
            }
        }
    }

    #[test]
    fn test_normal_distribution_mean() {
        set_seed(12345);
        let params = GenerationParams {
            size_distribution: SizeDistribution::Normal,
            bubble_count: 100,
            mean_radius: 0.025,
            std_dev: 0.005,
            min_radius: 0.01,
            max_radius: 0.04,
            ..Default::default()
        };
        let generator = FoamGenerator::new(params);
        let cluster = generator.generate(0.025);

        let sum: f32 = cluster.bubbles().iter().map(|b| b.radius).sum();
        let mean = sum / cluster.len() as f32;
        // Mean should be close to target (within 2 std errors)
        let std_error = 0.005 / (100.0_f32).sqrt();
        assert!((mean - 0.025).abs() < 3.0 * std_error);
    }

    #[test]
    fn test_box_muller() {
        set_seed(42);
        let mut sum = 0.0;
        let n = 1000;
        for _ in 0..n {
            let (z0, z1) = box_muller();
            sum += z0 + z1;
        }
        let mean = sum / (2 * n) as f32;
        // Mean should be close to 0
        assert!(mean.abs() < 0.1);
    }

    #[test]
    fn test_gamma_distribution() {
        set_seed(42);
        // Gamma(2, 1) has mean = 2
        let mut sum = 0.0;
        let n = 1000;
        for _ in 0..n {
            sum += sample_gamma(2.0, 1.0);
        }
        let mean = sum / n as f32;
        assert!((mean - 2.0).abs() < 0.2);
    }

    #[test]
    fn test_lattice_positions_are_distinct() {
        let params = GenerationParams {
            positioning_mode: PositioningMode::FaceCenteredCubic,
            bubble_count: 8,
            jitter: 0.0,
            ..Default::default()
        };
        let generator = FoamGenerator::new(params);
        let positions = generator.generate_positions();

        // All positions should be distinct
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let dist = (positions[i] - positions[j]).length();
                assert!(dist > 0.001, "Positions {} and {} are too close", i, j);
            }
        }
    }
}
