//! Interference color lookup table generation
//!
//! Pre-computes thin-film interference colors across the thickness/angle domain
//! to replace the expensive per-pixel spectral sampling in the fragment shader.

use std::f32::consts::PI;

/// LUT dimensions - balance between quality and memory
pub const LUT_THICKNESS_SAMPLES: u32 = 256;  // 0-2000nm range
pub const LUT_ANGLE_SAMPLES: u32 = 64;       // 0-90° (cos_theta 0-1)

/// Maximum thickness in nanometers covered by the LUT
pub const LUT_MAX_THICKNESS_NM: f32 = 2000.0;

/// Generate the interference color lookup table
/// Returns RGBA8 data as Vec<u8> for texture upload
pub fn generate_interference_lut(refractive_index: f32, intensity: f32) -> Vec<u8> {
    let size = (LUT_THICKNESS_SAMPLES * LUT_ANGLE_SAMPLES) as usize;
    let mut data = Vec::with_capacity(size * 4);

    for angle_idx in 0..LUT_ANGLE_SAMPLES {
        for thickness_idx in 0..LUT_THICKNESS_SAMPLES {
            // Map indices to physical values
            let thickness_nm = (thickness_idx as f32 / (LUT_THICKNESS_SAMPLES - 1) as f32)
                * LUT_MAX_THICKNESS_NM;
            let cos_theta = angle_idx as f32 / (LUT_ANGLE_SAMPLES - 1) as f32;

            // Compute interference color
            let rgb = thin_film_interference(thickness_nm, cos_theta, refractive_index, intensity);

            // Convert to RGBA8
            data.push((rgb[0].clamp(0.0, 1.0) * 255.0) as u8);
            data.push((rgb[1].clamp(0.0, 1.0) * 255.0) as u8);
            data.push((rgb[2].clamp(0.0, 1.0) * 255.0) as u8);
            data.push(255); // Alpha
        }
    }

    data
}

/// Compute thin-film interference color for given parameters
/// This is a CPU implementation matching the shader algorithm
fn thin_film_interference(
    thickness_nm: f32,
    cos_theta: f32,
    n_film: f32,
    intensity: f32,
) -> [f32; 3] {
    // 7-point spectral sampling
    const WAVELENGTHS: [f32; 7] = [400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0];

    // Transmission angle from Snell's law
    let cos_theta_t = snells_law(cos_theta, n_film);

    // Fresnel reflectance
    let fresnel = fresnel_unpolarized(cos_theta, cos_theta_t, 1.0, n_film);

    // Optical path
    let optical_path = 2.0 * n_film * thickness_nm * cos_theta_t;

    // Accumulate XYZ tristimulus
    let mut xyz = [0.0f32; 3];

    for wavelength in WAVELENGTHS {
        // Phase with π shift
        let phase = 2.0 * PI * optical_path / wavelength + PI;

        // Airy formula
        let airy_intensity = airy_interference(phase, fresnel);

        // CIE color matching
        let cie = cie_color_matching(wavelength);
        xyz[0] += cie[0] * airy_intensity;
        xyz[1] += cie[1] * airy_intensity;
        xyz[2] += cie[2] * airy_intensity;
    }

    // Normalize and convert to RGB
    xyz[0] /= 7.0;
    xyz[1] /= 7.0;
    xyz[2] /= 7.0;

    let mut rgb = xyz_to_rgb(xyz);

    // Apply intensity
    rgb[0] *= intensity;
    rgb[1] *= intensity;
    rgb[2] *= intensity;

    rgb
}

/// Snell's law: calculate transmission angle cosine
fn snells_law(cos_theta_i: f32, n_film: f32) -> f32 {
    let sin_theta_i = (1.0 - cos_theta_i * cos_theta_i).max(0.0).sqrt();
    let sin_theta_t = sin_theta_i / n_film;
    (1.0 - sin_theta_t * sin_theta_t).max(0.0).sqrt()
}

/// Fresnel equations for unpolarized light
fn fresnel_unpolarized(cos_theta_i: f32, cos_theta_t: f32, n1: f32, n2: f32) -> f32 {
    // s-polarization
    let n1_cos_i = n1 * cos_theta_i;
    let n2_cos_t = n2 * cos_theta_t;
    let r_s_num = n1_cos_i - n2_cos_t;
    let r_s_den = n1_cos_i + n2_cos_t;
    let r_s = r_s_num / r_s_den.max(0.0001);

    // p-polarization
    let n2_cos_i = n2 * cos_theta_i;
    let n1_cos_t = n1 * cos_theta_t;
    let r_p_num = n2_cos_i - n1_cos_t;
    let r_p_den = n2_cos_i + n1_cos_t;
    let r_p = r_p_num / r_p_den.max(0.0001);

    // Average
    (r_s * r_s + r_p * r_p) * 0.5
}

/// Airy formula for interference intensity
fn airy_interference(phase: f32, reflectance: f32) -> f32 {
    let one_minus_r = (1.0 - reflectance).max(0.001);
    let f = 4.0 * reflectance / (one_minus_r * one_minus_r);

    let sin_half_phase = (phase * 0.5).sin();
    let sin2 = sin_half_phase * sin_half_phase;
    let numerator = f * sin2;
    let denominator = 1.0 + f * sin2;

    numerator / denominator.max(0.001)
}

/// CIE 1931 color matching functions (Gaussian approximation)
fn cie_color_matching(wavelength: f32) -> [f32; 3] {
    let x = 1.056 * gaussian(wavelength, 599.8, 37.9)
        + 0.362 * gaussian(wavelength, 442.0, 16.0)
        - 0.065 * gaussian(wavelength, 501.1, 20.4);

    let y = 0.821 * gaussian(wavelength, 568.8, 46.9)
        + 0.286 * gaussian(wavelength, 530.9, 31.1);

    let z = 1.217 * gaussian(wavelength, 437.0, 11.8)
        + 0.681 * gaussian(wavelength, 459.0, 26.0);

    [x.max(0.0), y.max(0.0), z.max(0.0)]
}

/// Gaussian function for CIE approximation
#[inline]
fn gaussian(x: f32, mean: f32, sigma: f32) -> f32 {
    let t = (x - mean) / sigma;
    (-0.5 * t * t).exp()
}

/// Convert XYZ to linear sRGB
fn xyz_to_rgb(xyz: [f32; 3]) -> [f32; 3] {
    let r = 3.2404542 * xyz[0] - 1.5371385 * xyz[1] - 0.4985314 * xyz[2];
    let g = -0.9692660 * xyz[0] + 1.8760108 * xyz[1] + 0.0415560 * xyz[2];
    let b = 0.0556434 * xyz[0] - 0.2040259 * xyz[1] + 1.0572252 * xyz[2];
    [r, g, b]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lut_generation_produces_valid_data() {
        let lut = generate_interference_lut(1.33, 1.0);
        let expected_size = (LUT_THICKNESS_SAMPLES * LUT_ANGLE_SAMPLES * 4) as usize;
        assert_eq!(lut.len(), expected_size);
    }

    #[test]
    fn test_thin_film_produces_colors() {
        // At 500nm thickness, should produce some color
        let rgb = thin_film_interference(500.0, 0.9, 1.33, 1.0);
        assert!(rgb[0] >= 0.0 && rgb[0] <= 1.0);
        assert!(rgb[1] >= 0.0 && rgb[1] <= 1.0);
        assert!(rgb[2] >= 0.0 && rgb[2] <= 1.0);
        // Should have some intensity
        assert!(rgb[0] + rgb[1] + rgb[2] > 0.01);
    }

    #[test]
    fn test_snells_law() {
        // Normal incidence
        let cos_t = snells_law(1.0, 1.33);
        assert!((cos_t - 1.0).abs() < 0.01);

        // Grazing angle
        let cos_t = snells_law(0.1, 1.33);
        assert!(cos_t > 0.0 && cos_t < 1.0);
    }

    #[test]
    fn test_fresnel_at_normal_incidence() {
        // At normal incidence, Fresnel should give Schlick-like result
        let r = fresnel_unpolarized(1.0, 1.0, 1.0, 1.33);
        // For n1=1, n2=1.33, R0 ≈ ((0.33)/(2.33))² ≈ 0.02
        assert!(r > 0.01 && r < 0.05);
    }
}
