//! Thin-film interference calculations for soap bubble coloring.
//!
//! This module implements the physics of thin-film interference, which produces
//! the characteristic iridescent colors seen in soap bubbles. The colors arise
//! from the interference between light reflected from the outer and inner
//! surfaces of the thin soap film.
//!
//! # Physics Background
//!
//! When light hits a thin film, part of it reflects from the top surface and
//! part transmits through, reflects from the bottom surface, and exits. The
//! optical path difference between these rays causes interference:
//!
//! - **Optical Path Difference**: `delta = 2 * n_film * d * cos(theta_t) + lambda/2`
//! - **Phase shift of lambda/2**: Occurs at the air-to-film interface (low-to-high n)
//! - **No phase shift**: At the film-to-air interface (high-to-low n)
//!
//! The intensity depends on whether the path difference corresponds to
//! constructive (bright) or destructive (dark) interference for each wavelength.

use std::f64::consts::PI;

/// Standard visible light wavelengths in nanometers for RGB color calculation.
pub mod wavelengths {
    /// Red wavelength in nanometers
    pub const RED_NM: f64 = 650.0;
    /// Green wavelength in nanometers
    pub const GREEN_NM: f64 = 532.0;
    /// Blue wavelength in nanometers
    pub const BLUE_NM: f64 = 450.0;
}

/// Physical constants for interference calculations.
pub mod constants {
    /// Refractive index of air
    pub const REFRACTIVE_INDEX_AIR: f64 = 1.0;
    /// Default refractive index for soap film (water-based)
    pub const REFRACTIVE_INDEX_SOAP_FILM: f64 = 1.33;
}

/// RGB color representation with values in range [0.0, 1.0].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RgbColor {
    /// Red component (0.0 to 1.0)
    pub red: f64,
    /// Green component (0.0 to 1.0)
    pub green: f64,
    /// Blue component (0.0 to 1.0)
    pub blue: f64,
}

impl RgbColor {
    /// Create a new RGB color, clamping values to valid range.
    pub fn new(red: f64, green: f64, blue: f64) -> Self {
        Self {
            red: red.clamp(0.0, 1.0),
            green: green.clamp(0.0, 1.0),
            blue: blue.clamp(0.0, 1.0),
        }
    }

    /// Create RGB color from f32 components.
    pub fn from_f32(red: f32, green: f32, blue: f32) -> Self {
        Self::new(red as f64, green as f64, blue as f64)
    }

    /// Convert to f32 tuple (useful for graphics APIs).
    pub fn to_f32_tuple(&self) -> (f32, f32, f32) {
        (self.red as f32, self.green as f32, self.blue as f32)
    }

    /// Convert to u8 tuple for standard 8-bit color.
    pub fn to_u8_tuple(&self) -> (u8, u8, u8) {
        (
            (self.red * 255.0).round() as u8,
            (self.green * 255.0).round() as u8,
            (self.blue * 255.0).round() as u8,
        )
    }

    /// Convert to f32 array with alpha channel (RGBA).
    pub fn to_f32_array_with_alpha(&self, alpha: f32) -> [f32; 4] {
        [self.red as f32, self.green as f32, self.blue as f32, alpha]
    }
}

impl Default for RgbColor {
    fn default() -> Self {
        Self {
            red: 0.0,
            green: 0.0,
            blue: 0.0,
        }
    }
}

/// Fresnel reflection coefficients for both polarizations.
#[derive(Debug, Clone, Copy)]
pub struct FresnelCoefficients {
    /// Reflectance for s-polarized light (perpendicular)
    pub reflectance_s: f64,
    /// Reflectance for p-polarized light (parallel)
    pub reflectance_p: f64,
    /// Average reflectance (unpolarized light)
    pub reflectance_average: f64,
}

/// Calculator for thin-film interference effects in soap bubbles.
///
/// This struct encapsulates the physics calculations needed to determine
/// the interference colors produced by a thin soap film at various thicknesses
/// and viewing angles.
///
/// # Example
///
/// ```
/// use soap_bubble_sim::physics::interference::InterferenceCalculator;
///
/// let calculator = InterferenceCalculator::new(1.33);
///
/// // Calculate color for a 400nm thick film viewed head-on
/// let color = calculator.calculate_interference_color(400.0, 1.0);
/// println!("RGB: ({}, {}, {})", color.red, color.green, color.blue);
/// ```
// put id:'cpu_interference_ref', label:'CPU interference reference', input:'final_config.internal', output:'interference_color.internal'
#[derive(Debug, Clone)]
pub struct InterferenceCalculator {
    /// Refractive index of the soap film
    refractive_index_film: f64,
    /// Refractive index of the surrounding medium (typically air)
    refractive_index_medium: f64,
    /// RGB wavelengths to use for color calculation (in nm)
    wavelengths_nm: [f64; 3],
}

impl InterferenceCalculator {
    /// Create a new interference calculator with the specified film refractive index.
    ///
    /// # Arguments
    /// * `refractive_index_film` - Refractive index of the soap film (~1.33 for water-based)
    pub fn new(refractive_index_film: f64) -> Self {
        Self {
            refractive_index_film,
            refractive_index_medium: constants::REFRACTIVE_INDEX_AIR,
            wavelengths_nm: [
                wavelengths::RED_NM,
                wavelengths::GREEN_NM,
                wavelengths::BLUE_NM,
            ],
        }
    }

    /// Create a calculator with default soap film properties.
    pub fn default_soap_film() -> Self {
        Self::new(constants::REFRACTIVE_INDEX_SOAP_FILM)
    }

    /// Create a calculator with custom wavelengths for RGB channels.
    ///
    /// # Arguments
    /// * `refractive_index_film` - Refractive index of the soap film
    /// * `red_nm` - Wavelength for red channel in nanometers
    /// * `green_nm` - Wavelength for green channel in nanometers
    /// * `blue_nm` - Wavelength for blue channel in nanometers
    pub fn with_custom_wavelengths(
        refractive_index_film: f64,
        red_nm: f64,
        green_nm: f64,
        blue_nm: f64,
    ) -> Self {
        Self {
            refractive_index_film,
            refractive_index_medium: constants::REFRACTIVE_INDEX_AIR,
            wavelengths_nm: [red_nm, green_nm, blue_nm],
        }
    }

    /// Get the film refractive index.
    pub fn refractive_index_film(&self) -> f64 {
        self.refractive_index_film
    }

    /// Get the medium refractive index.
    pub fn refractive_index_medium(&self) -> f64 {
        self.refractive_index_medium
    }

    /// Calculate the transmission angle using Snell's law.
    ///
    /// Snell's law: n1 * sin(theta_i) = n2 * sin(theta_t)
    ///
    /// # Arguments
    /// * `cos_theta_incident` - Cosine of the incident angle (1.0 = normal incidence)
    ///
    /// # Returns
    /// Cosine of the transmission angle inside the film
    pub fn calculate_transmission_angle_cos(&self, cos_theta_incident: f64) -> f64 {
        let cos_theta_incident_clamped = cos_theta_incident.clamp(-1.0, 1.0);
        let sin_theta_incident = (1.0 - cos_theta_incident_clamped.powi(2)).sqrt();

        // Snell's law: n1 * sin(theta_i) = n2 * sin(theta_t)
        let sin_theta_transmitted =
            (self.refractive_index_medium / self.refractive_index_film) * sin_theta_incident;

        // Check for total internal reflection (shouldn't happen going from low to high n)
        if sin_theta_transmitted.abs() > 1.0 {
            return 0.0; // Grazing angle
        }

        (1.0 - sin_theta_transmitted.powi(2)).sqrt()
    }

    /// Calculate the optical path difference for thin-film interference.
    ///
    /// The optical path difference includes:
    /// - Path through the film: 2 * n_film * d * cos(theta_t)
    /// - Phase shift of lambda/2 from reflection at air-film interface
    ///
    /// # Arguments
    /// * `film_thickness_nm` - Film thickness in nanometers
    /// * `cos_theta_transmitted` - Cosine of the angle inside the film
    /// * `wavelength_nm` - Wavelength of light in nanometers
    ///
    /// # Returns
    /// Optical path difference in nanometers
    pub fn calculate_optical_path_difference(
        &self,
        film_thickness_nm: f64,
        cos_theta_transmitted: f64,
        wavelength_nm: f64,
    ) -> f64 {
        // Geometric path difference through film (factor of 2 for round trip)
        let geometric_path = 2.0 * self.refractive_index_film * film_thickness_nm * cos_theta_transmitted;

        // Add lambda/2 phase shift from reflection at air-to-film interface
        // (reflection from going low-n to high-n medium)
        geometric_path + wavelength_nm / 2.0
    }

    /// Calculate the phase difference in radians from the optical path difference.
    ///
    /// # Arguments
    /// * `optical_path_difference_nm` - Optical path difference in nanometers
    /// * `wavelength_nm` - Wavelength of light in nanometers
    ///
    /// # Returns
    /// Phase difference in radians
    pub fn calculate_phase_difference(
        &self,
        optical_path_difference_nm: f64,
        wavelength_nm: f64,
    ) -> f64 {
        2.0 * PI * optical_path_difference_nm / wavelength_nm
    }

    /// Calculate Fresnel reflection coefficients at the air-film interface.
    ///
    /// This implements the Fresnel equations for the amplitude reflection
    /// coefficients at the interface between two media.
    ///
    /// # Arguments
    /// * `cos_theta_incident` - Cosine of the incident angle
    ///
    /// # Returns
    /// `FresnelCoefficients` containing reflectance for both polarizations
    pub fn calculate_fresnel_reflection(&self, cos_theta_incident: f64) -> FresnelCoefficients {
        let cos_theta_incident_clamped = cos_theta_incident.clamp(-1.0, 1.0);
        let cos_theta_transmitted = self.calculate_transmission_angle_cos(cos_theta_incident_clamped);

        let n_incident = self.refractive_index_medium;
        let n_transmitted = self.refractive_index_film;

        // Fresnel equations for amplitude reflection coefficients
        // r_s = (n1*cos(theta_i) - n2*cos(theta_t)) / (n1*cos(theta_i) + n2*cos(theta_t))
        // r_p = (n2*cos(theta_i) - n1*cos(theta_t)) / (n2*cos(theta_i) + n1*cos(theta_t))

        let n1_cos_i = n_incident * cos_theta_incident_clamped;
        let n2_cos_t = n_transmitted * cos_theta_transmitted;
        let n2_cos_i = n_transmitted * cos_theta_incident_clamped;
        let n1_cos_t = n_incident * cos_theta_transmitted;

        // s-polarization (perpendicular)
        let amplitude_reflection_s = (n1_cos_i - n2_cos_t) / (n1_cos_i + n2_cos_t);
        let reflectance_s = amplitude_reflection_s.powi(2);

        // p-polarization (parallel)
        let amplitude_reflection_p = (n2_cos_i - n1_cos_t) / (n2_cos_i + n1_cos_t);
        let reflectance_p = amplitude_reflection_p.powi(2);

        // Average for unpolarized light
        let reflectance_average = (reflectance_s + reflectance_p) / 2.0;

        FresnelCoefficients {
            reflectance_s,
            reflectance_p,
            reflectance_average,
        }
    }

    /// Calculate the reflected intensity for a single wavelength using thin-film interference.
    ///
    /// The intensity is calculated using the Airy formula for thin-film interference,
    /// accounting for Fresnel reflection at both interfaces.
    ///
    /// # Arguments
    /// * `film_thickness_nm` - Film thickness in nanometers
    /// * `cos_theta_incident` - Cosine of the incident viewing angle (1.0 = normal)
    /// * `wavelength_nm` - Wavelength of light in nanometers
    ///
    /// # Returns
    /// Reflected intensity as a fraction (0.0 to 1.0)
    pub fn calculate_reflected_intensity(
        &self,
        film_thickness_nm: f64,
        cos_theta_incident: f64,
        wavelength_nm: f64,
    ) -> f64 {
        // Handle edge case of zero thickness
        if film_thickness_nm <= 0.0 {
            return 0.0;
        }

        let cos_theta_transmitted = self.calculate_transmission_angle_cos(cos_theta_incident);

        // Calculate optical path difference
        let optical_path_difference =
            self.calculate_optical_path_difference(film_thickness_nm, cos_theta_transmitted, wavelength_nm);

        // Calculate phase difference
        let phase_difference =
            self.calculate_phase_difference(optical_path_difference, wavelength_nm);

        // Get Fresnel reflectance
        let fresnel = self.calculate_fresnel_reflection(cos_theta_incident);
        let reflectance = fresnel.reflectance_average;

        // Airy formula for thin-film interference
        // I = 4*R*sin^2(delta/2) / (1 + R)^2
        // where R is the reflectance and delta is the phase difference
        //
        // For a more physically accurate model with multiple reflections:
        // I = (2*R*(1 - cos(delta))) / (1 + R^2 - 2*R*cos(delta))

        let cos_phase = phase_difference.cos();
        let numerator = 2.0 * reflectance * (1.0 - cos_phase);
        let denominator = 1.0 + reflectance.powi(2) - 2.0 * reflectance * cos_phase;

        if denominator.abs() < f64::EPSILON {
            return reflectance;
        }

        (numerator / denominator).clamp(0.0, 1.0)
    }

    /// Calculate the interference color for a given film thickness and viewing angle.
    ///
    /// This is the main method for obtaining the visible color produced by
    /// thin-film interference at a specific point on the bubble surface.
    ///
    /// # Arguments
    /// * `film_thickness_nm` - Film thickness in nanometers
    /// * `cos_theta_incident` - Cosine of the incident viewing angle (1.0 = looking straight on)
    ///
    /// # Returns
    /// `RgbColor` representing the interference color
    ///
    /// # Example
    ///
    /// ```
    /// use soap_bubble_sim::physics::interference::InterferenceCalculator;
    ///
    /// let calculator = InterferenceCalculator::default_soap_film();
    ///
    /// // Normal incidence at 300nm thickness
    /// let color = calculator.calculate_interference_color(300.0, 1.0);
    ///
    /// // Grazing angle at 500nm thickness
    /// let color_angled = calculator.calculate_interference_color(500.0, 0.5);
    /// ```
    pub fn calculate_interference_color(
        &self,
        film_thickness_nm: f64,
        cos_theta_incident: f64,
    ) -> RgbColor {
        let red_intensity =
            self.calculate_reflected_intensity(film_thickness_nm, cos_theta_incident, self.wavelengths_nm[0]);
        let green_intensity =
            self.calculate_reflected_intensity(film_thickness_nm, cos_theta_incident, self.wavelengths_nm[1]);
        let blue_intensity =
            self.calculate_reflected_intensity(film_thickness_nm, cos_theta_incident, self.wavelengths_nm[2]);

        // Apply gamma correction for perceptually uniform display
        // Using sRGB gamma of approximately 2.2
        let gamma = 2.2_f64;
        let red_corrected = red_intensity.powf(1.0 / gamma);
        let green_corrected = green_intensity.powf(1.0 / gamma);
        let blue_corrected = blue_intensity.powf(1.0 / gamma);

        RgbColor::new(red_corrected, green_corrected, blue_corrected)
    }

    /// Calculate interference color without gamma correction (linear RGB).
    ///
    /// Use this when you need linear color values for further processing
    /// or when gamma correction will be applied elsewhere (e.g., in a shader).
    ///
    /// # Arguments
    /// * `film_thickness_nm` - Film thickness in nanometers
    /// * `cos_theta_incident` - Cosine of the incident viewing angle
    ///
    /// # Returns
    /// `RgbColor` in linear color space
    pub fn calculate_interference_color_linear(
        &self,
        film_thickness_nm: f64,
        cos_theta_incident: f64,
    ) -> RgbColor {
        let red_intensity =
            self.calculate_reflected_intensity(film_thickness_nm, cos_theta_incident, self.wavelengths_nm[0]);
        let green_intensity =
            self.calculate_reflected_intensity(film_thickness_nm, cos_theta_incident, self.wavelengths_nm[1]);
        let blue_intensity =
            self.calculate_reflected_intensity(film_thickness_nm, cos_theta_incident, self.wavelengths_nm[2]);

        RgbColor::new(red_intensity, green_intensity, blue_intensity)
    }

    /// Calculate interference color with custom refractive index override.
    ///
    /// Useful for simulating different soap solutions or testing.
    ///
    /// # Arguments
    /// * `film_thickness_nm` - Film thickness in nanometers
    /// * `cos_theta_incident` - Cosine of the incident viewing angle
    /// * `refractive_index` - Custom refractive index to use
    ///
    /// # Returns
    /// `RgbColor` representing the interference color
    pub fn calculate_interference_color_with_index(
        &self,
        film_thickness_nm: f64,
        cos_theta_incident: f64,
        refractive_index: f64,
    ) -> RgbColor {
        let custom_calculator = InterferenceCalculator::new(refractive_index);
        custom_calculator.calculate_interference_color(film_thickness_nm, cos_theta_incident)
    }

    /// Generate a color lookup table for a range of thicknesses.
    ///
    /// This is useful for pre-computing colors for GPU texture lookup.
    ///
    /// # Arguments
    /// * `min_thickness_nm` - Minimum film thickness
    /// * `max_thickness_nm` - Maximum film thickness
    /// * `num_samples` - Number of samples in the table
    /// * `cos_theta` - Viewing angle cosine (typically 1.0 for the table)
    ///
    /// # Returns
    /// Vector of `RgbColor` values
    pub fn generate_color_lookup_table(
        &self,
        min_thickness_nm: f64,
        max_thickness_nm: f64,
        num_samples: usize,
        cos_theta: f64,
    ) -> Vec<RgbColor> {
        let mut table = Vec::with_capacity(num_samples);
        let thickness_step = (max_thickness_nm - min_thickness_nm) / (num_samples - 1) as f64;

        for i in 0..num_samples {
            let thickness = min_thickness_nm + i as f64 * thickness_step;
            table.push(self.calculate_interference_color(thickness, cos_theta));
        }

        table
    }

    /// Generate a 2D color lookup table for thickness and angle.
    ///
    /// Creates a 2D texture suitable for GPU shader lookup.
    ///
    /// # Arguments
    /// * `min_thickness_nm` - Minimum film thickness
    /// * `max_thickness_nm` - Maximum film thickness
    /// * `thickness_samples` - Number of thickness samples
    /// * `angle_samples` - Number of angle samples (from cos=0 to cos=1)
    ///
    /// # Returns
    /// 2D vector where `[thickness_idx][angle_idx]` gives the color
    pub fn generate_2d_color_lookup_table(
        &self,
        min_thickness_nm: f64,
        max_thickness_nm: f64,
        thickness_samples: usize,
        angle_samples: usize,
    ) -> Vec<Vec<RgbColor>> {
        let thickness_step = (max_thickness_nm - min_thickness_nm) / (thickness_samples - 1).max(1) as f64;
        let angle_step = 1.0 / (angle_samples - 1).max(1) as f64;

        (0..thickness_samples)
            .map(|thickness_index| {
                let thickness = min_thickness_nm + thickness_index as f64 * thickness_step;
                (0..angle_samples)
                    .map(|angle_index| {
                        let cos_theta = angle_index as f64 * angle_step;
                        self.calculate_interference_color(thickness, cos_theta)
                    })
                    .collect()
            })
            .collect()
    }
}

impl Default for InterferenceCalculator {
    fn default() -> Self {
        Self::default_soap_film()
    }
}

/// GLSL shader code generation for GPU-accelerated interference calculation.
///
/// This module provides GLSL code snippets that implement the same physics
/// as the CPU `InterferenceCalculator`, allowing for real-time GPU rendering.
pub mod shader_code {
    /// GLSL function for calculating transmission angle using Snell's law.
    pub const SNELLS_LAW_GLSL: &str = r#"
// Calculate transmission angle cosine using Snell's law
float calculateTransmissionAngleCos(float cosIncident, float nMedium, float nFilm) {
    float sinIncident = sqrt(1.0 - cosIncident * cosIncident);
    float sinTransmitted = (nMedium / nFilm) * sinIncident;
    if (abs(sinTransmitted) > 1.0) return 0.0;
    return sqrt(1.0 - sinTransmitted * sinTransmitted);
}
"#;

    /// GLSL function for calculating Fresnel reflectance.
    pub const FRESNEL_GLSL: &str = r#"
// Calculate average Fresnel reflectance for unpolarized light
float calculateFresnelReflectance(float cosIncident, float nMedium, float nFilm) {
    float cosTransmitted = calculateTransmissionAngleCos(cosIncident, nMedium, nFilm);

    float n1CosI = nMedium * cosIncident;
    float n2CosT = nFilm * cosTransmitted;
    float n2CosI = nFilm * cosIncident;
    float n1CosT = nMedium * cosTransmitted;

    float rs = (n1CosI - n2CosT) / (n1CosI + n2CosT);
    float rp = (n2CosI - n1CosT) / (n2CosI + n1CosT);

    return (rs * rs + rp * rp) * 0.5;
}
"#;

    /// GLSL function for calculating thin-film interference intensity.
    pub const INTERFERENCE_INTENSITY_GLSL: &str = r#"
// Calculate reflected intensity for thin-film interference
float calculateInterferenceIntensity(float thickness, float cosIncident, float wavelength, float nMedium, float nFilm) {
    if (thickness <= 0.0) return 0.0;

    float cosTransmitted = calculateTransmissionAngleCos(cosIncident, nMedium, nFilm);

    // Optical path difference with lambda/2 phase shift
    float opticalPath = 2.0 * nFilm * thickness * cosTransmitted + wavelength * 0.5;
    float phaseDiff = 2.0 * 3.14159265 * opticalPath / wavelength;

    float R = calculateFresnelReflectance(cosIncident, nMedium, nFilm);
    float cosPhi = cos(phaseDiff);

    float numerator = 2.0 * R * (1.0 - cosPhi);
    float denominator = 1.0 + R * R - 2.0 * R * cosPhi;

    return clamp(numerator / max(denominator, 0.0001), 0.0, 1.0);
}
"#;

    /// GLSL function for calculating interference RGB color.
    pub const INTERFERENCE_COLOR_GLSL: &str = r#"
// Calculate interference color (linear RGB)
vec3 calculateInterferenceColor(float thickness, float cosIncident, float nFilm) {
    const float N_AIR = 1.0;
    const float RED_WL = 650.0;
    const float GREEN_WL = 532.0;
    const float BLUE_WL = 450.0;

    float r = calculateInterferenceIntensity(thickness, cosIncident, RED_WL, N_AIR, nFilm);
    float g = calculateInterferenceIntensity(thickness, cosIncident, GREEN_WL, N_AIR, nFilm);
    float b = calculateInterferenceIntensity(thickness, cosIncident, BLUE_WL, N_AIR, nFilm);

    return vec3(r, g, b);
}

// With gamma correction for display
vec3 calculateInterferenceColorGamma(float thickness, float cosIncident, float nFilm) {
    vec3 linear = calculateInterferenceColor(thickness, cosIncident, nFilm);
    return pow(linear, vec3(1.0 / 2.2));
}
"#;

    /// Complete GLSL shader code for interference calculation.
    pub fn complete_shader_code() -> String {
        format!(
            "{}\n{}\n{}\n{}",
            SNELLS_LAW_GLSL,
            FRESNEL_GLSL,
            INTERFERENCE_INTENSITY_GLSL,
            INTERFERENCE_COLOR_GLSL
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;

    #[test]
    fn test_rgb_color_clamping() {
        let color = RgbColor::new(1.5, -0.5, 0.5);
        assert!((color.red - 1.0).abs() < EPSILON);
        assert!((color.green - 0.0).abs() < EPSILON);
        assert!((color.blue - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_rgb_color_to_u8() {
        let color = RgbColor::new(1.0, 0.5, 0.0);
        let (red, green, blue) = color.to_u8_tuple();
        assert_eq!(red, 255);
        assert_eq!(green, 128);
        assert_eq!(blue, 0);
    }

    #[test]
    fn test_calculator_creation() {
        let calculator = InterferenceCalculator::new(1.33);
        assert!((calculator.refractive_index_film() - 1.33).abs() < EPSILON);
        assert!((calculator.refractive_index_medium() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_default_calculator() {
        let calculator = InterferenceCalculator::default();
        assert!((calculator.refractive_index_film() - constants::REFRACTIVE_INDEX_SOAP_FILM).abs() < EPSILON);
    }

    #[test]
    fn test_transmission_angle_normal_incidence() {
        let calculator = InterferenceCalculator::new(1.33);
        // At normal incidence, transmission angle should also be normal
        let cos_theta_transmitted = calculator.calculate_transmission_angle_cos(1.0);
        assert!((cos_theta_transmitted - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_transmission_angle_snells_law() {
        let calculator = InterferenceCalculator::new(1.5);
        // 45 degree incidence (cos = 0.707...)
        let cos_incident = (2.0_f64).sqrt() / 2.0;
        let cos_transmitted = calculator.calculate_transmission_angle_cos(cos_incident);

        // Verify Snell's law: n1*sin(theta1) = n2*sin(theta2)
        let sin_incident = (1.0 - cos_incident.powi(2)).sqrt();
        let sin_transmitted = (1.0 - cos_transmitted.powi(2)).sqrt();
        let ratio = sin_incident / sin_transmitted;

        assert!((ratio - 1.5).abs() < 1e-4);
    }

    #[test]
    fn test_fresnel_normal_incidence() {
        let calculator = InterferenceCalculator::new(1.33);
        let fresnel = calculator.calculate_fresnel_reflection(1.0);

        // At normal incidence, R = ((n1-n2)/(n1+n2))^2
        let expected_reflectance = ((1.0_f64 - 1.33) / (1.0 + 1.33)).powi(2);
        assert!((fresnel.reflectance_average - expected_reflectance).abs() < 1e-4);
    }

    #[test]
    fn test_zero_thickness_gives_zero_intensity() {
        let calculator = InterferenceCalculator::default();
        let intensity = calculator.calculate_reflected_intensity(0.0, 1.0, 550.0);
        assert!((intensity - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_interference_produces_colors() {
        let calculator = InterferenceCalculator::default();

        // Test various thicknesses produce non-zero colors
        for thickness in [100.0, 200.0, 300.0, 400.0, 500.0] {
            let color = calculator.calculate_interference_color(thickness, 1.0);
            // At least one channel should have significant intensity
            let max_channel = color.red.max(color.green).max(color.blue);
            assert!(max_channel > 0.01, "Thickness {} produced no color", thickness);
        }
    }

    #[test]
    fn test_color_varies_with_thickness() {
        let calculator = InterferenceCalculator::default();

        let color_100nm = calculator.calculate_interference_color(100.0, 1.0);
        let color_300nm = calculator.calculate_interference_color(300.0, 1.0);

        // Colors at different thicknesses should differ
        let difference =
            (color_100nm.red - color_300nm.red).abs() +
            (color_100nm.green - color_300nm.green).abs() +
            (color_100nm.blue - color_300nm.blue).abs();

        assert!(difference > 0.01, "Colors should vary with thickness");
    }

    #[test]
    fn test_color_varies_with_angle() {
        let calculator = InterferenceCalculator::default();

        let color_normal = calculator.calculate_interference_color(300.0, 1.0);
        let color_angled = calculator.calculate_interference_color(300.0, 0.5);

        // Colors at different angles should differ
        let difference =
            (color_normal.red - color_angled.red).abs() +
            (color_normal.green - color_angled.green).abs() +
            (color_normal.blue - color_angled.blue).abs();

        assert!(difference > 0.001, "Colors should vary with viewing angle");
    }

    #[test]
    fn test_lookup_table_generation() {
        let calculator = InterferenceCalculator::default();
        let table = calculator.generate_color_lookup_table(0.0, 1000.0, 100, 1.0);

        assert_eq!(table.len(), 100);

        // Check that table contains valid colors
        for color in &table {
            assert!(color.red >= 0.0 && color.red <= 1.0);
            assert!(color.green >= 0.0 && color.green <= 1.0);
            assert!(color.blue >= 0.0 && color.blue <= 1.0);
        }
    }

    #[test]
    fn test_2d_lookup_table_generation() {
        let calculator = InterferenceCalculator::default();
        let table = calculator.generate_2d_color_lookup_table(0.0, 500.0, 10, 5);

        assert_eq!(table.len(), 10);
        assert_eq!(table[0].len(), 5);
    }

    #[test]
    fn test_linear_vs_gamma_color() {
        let calculator = InterferenceCalculator::default();

        let linear_color = calculator.calculate_interference_color_linear(300.0, 1.0);
        let gamma_color = calculator.calculate_interference_color(300.0, 1.0);

        // Gamma corrected should generally be brighter (for values < 1)
        // This is because x^(1/2.2) > x for 0 < x < 1
        if linear_color.red > 0.01 && linear_color.red < 0.99 {
            assert!(gamma_color.red >= linear_color.red);
        }
    }

    #[test]
    fn test_custom_wavelengths() {
        let calculator = InterferenceCalculator::with_custom_wavelengths(1.33, 700.0, 550.0, 400.0);
        let color = calculator.calculate_interference_color(300.0, 1.0);

        // Should still produce valid colors
        assert!(color.red >= 0.0 && color.red <= 1.0);
        assert!(color.green >= 0.0 && color.green <= 1.0);
        assert!(color.blue >= 0.0 && color.blue <= 1.0);
    }

    #[test]
    fn test_shader_code_generation() {
        let shader_code = shader_code::complete_shader_code();

        // Verify key functions are present
        assert!(shader_code.contains("calculateTransmissionAngleCos"));
        assert!(shader_code.contains("calculateFresnelReflectance"));
        assert!(shader_code.contains("calculateInterferenceIntensity"));
        assert!(shader_code.contains("calculateInterferenceColor"));
    }
}
