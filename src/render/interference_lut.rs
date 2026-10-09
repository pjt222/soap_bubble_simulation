//! Interference color lookup table generation
//!
//! Pre-computes thin-film interference colors across the thickness/angle domain
//! to replace the expensive per-pixel spectral sampling in the fragment shader.

use std::f32::consts::PI;

/// LUT dimensions - balance between quality and memory
pub const LUT_THICKNESS_SAMPLES: u32 = 256; // 0-2000nm range
pub const LUT_ANGLE_SAMPLES: u32 = 64; // 0-90° (cos_theta 0-1)

/// Maximum thickness in nanometers covered by the LUT
pub const LUT_MAX_THICKNESS_NM: f32 = 2000.0;

/// Generate the interference color lookup table
/// Returns RGBA8 data as Vec<u8> for texture upload
// put id:'cpu_lut_gen', label:'Generate interference LUT', input:'final_config.internal', output:'lut_texture_gpu.internal'
pub fn generate_interference_lut(refractive_index: f32, intensity: f32) -> Vec<u8> {
    let size = (LUT_THICKNESS_SAMPLES * LUT_ANGLE_SAMPLES) as usize;
    let mut data = Vec::with_capacity(size * 4);

    for angle_idx in 0..LUT_ANGLE_SAMPLES {
        for thickness_idx in 0..LUT_THICKNESS_SAMPLES {
            // Map indices to physical values
            let thickness_nm =
                (thickness_idx as f32 / (LUT_THICKNESS_SAMPLES - 1) as f32) * LUT_MAX_THICKNESS_NM;
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

    // Accumulate XYZ tristimulus
    let mut xyz = [0.0f32; 3];

    for wavelength in WAVELENGTHS {
        let airy_intensity =
            film_reflectance(thickness_nm, cos_theta_t, n_film, wavelength, fresnel);

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

/// Reflectance of a free-standing film (air | film | air) at one wavelength.
///
/// The Airy phase is the geometric round-trip phase only,
/// `delta = 4 pi n d cos(theta_t) / lambda`. The half-wave flip at the air->film
/// reflection is already contained in the Airy derivation through
/// `r21 = -r12`; adding pi here would invert every fringe (issue #42).
fn film_reflectance(
    thickness_nm: f32,
    cos_theta_t: f32,
    n_film: f32,
    wavelength_nm: f32,
    surface_reflectance: f32,
) -> f32 {
    let phase = 4.0 * PI * n_film * thickness_nm * cos_theta_t / wavelength_nm;
    airy_interference(phase, surface_reflectance)
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
    let x = 1.056 * gaussian(wavelength, 599.8, 37.9) + 0.362 * gaussian(wavelength, 442.0, 16.0)
        - 0.065 * gaussian(wavelength, 501.1, 20.4);

    let y = 0.821 * gaussian(wavelength, 568.8, 46.9) + 0.286 * gaussian(wavelength, 530.9, 31.1);

    let z = 1.217 * gaussian(wavelength, 437.0, 11.8) + 0.681 * gaussian(wavelength, 459.0, 26.0);

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
    let g = -0.969_266 * xyz[0] + 1.8760108 * xyz[1] + 0.0415560 * xyz[2];
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
        // At 500 nm and cos(theta_i) = 0.9 the reflectance maxima sit where
        // 2 n d cos(theta_t) = (m + 1/2) lambda, i.e. m = 2 at ~502 nm, so the
        // colour is green-dominant. Saturated interference colours can fall
        // slightly outside the sRGB gamut (a small negative channel); the LUT
        // packing in generate_interference_lut clamps them.
        let rgb = thin_film_interference(500.0, 0.9, 1.33, 1.0);
        assert!(
            rgb.iter()
                .all(|channel| channel.is_finite() && *channel <= 1.0),
            "{rgb:?}"
        );
        assert!(
            rgb[1] > rgb[0] && rgb[1] > rgb[2],
            "expected green-dominant colour: {rgb:?}"
        );
        let in_gamut_sum: f32 = rgb.iter().map(|channel| channel.clamp(0.0, 1.0)).sum();
        assert!(in_gamut_sum > 0.01, "{rgb:?}");
    }

    /// Exact free-standing slab reflectance at normal incidence,
    /// |(r12 + r23 e^{i delta}) / (1 + r12 r23 e^{i delta})|^2 with r23 = -r12,
    /// evaluated with explicit complex arithmetic so it does not share the
    /// Airy closed form under test.
    fn exact_slab_reflectance_normal_incidence(
        thickness_nm: f64,
        n_film: f64,
        wavelength_nm: f64,
    ) -> f64 {
        let r12 = (1.0 - n_film) / (1.0 + n_film);
        let r23 = -r12;
        let delta = 4.0 * std::f64::consts::PI * n_film * thickness_nm / wavelength_nm;
        let numerator_re = r12 + r23 * delta.cos();
        let numerator_im = r23 * delta.sin();
        let denominator_re = 1.0 + r12 * r23 * delta.cos();
        let denominator_im = r12 * r23 * delta.sin();
        (numerator_re * numerator_re + numerator_im * numerator_im)
            / (denominator_re * denominator_re + denominator_im * denominator_im)
    }

    #[test]
    fn test_film_reflectance_matches_exact_slab_at_normal_incidence() {
        let n_film = 1.33_f32;
        let surface_reflectance = fresnel_unpolarized(1.0, 1.0, 1.0, n_film);
        for wavelength_nm in [400.0_f32, 450.0, 532.0, 650.0, 700.0] {
            for thickness_step in 0..=400 {
                let thickness_nm = thickness_step as f32 * 5.0;
                let computed = film_reflectance(
                    thickness_nm,
                    1.0,
                    n_film,
                    wavelength_nm,
                    surface_reflectance,
                );
                let exact = exact_slab_reflectance_normal_incidence(
                    thickness_nm as f64,
                    n_film as f64,
                    wavelength_nm as f64,
                );
                assert!(
                    (computed as f64 - exact).abs() < 2e-5,
                    "d={thickness_nm} nm, lambda={wavelength_nm} nm: airy={computed}, exact={exact}"
                );
            }
        }
    }

    #[test]
    fn test_film_reflectance_vanishes_for_zero_thickness() {
        // A film much thinner than the wavelength reflects almost nothing (black film).
        let surface_reflectance = fresnel_unpolarized(1.0, 1.0, 1.0, 1.33);
        for wavelength_nm in [400.0_f32, 550.0, 700.0] {
            assert!(film_reflectance(0.0, 1.0, 1.33, wavelength_nm, surface_reflectance) < 1e-7);
        }
        let rgb = thin_film_interference(0.0, 1.0, 1.33, 1.0);
        assert!(
            rgb.iter().all(|channel| channel.abs() < 1e-6),
            "zero-thickness film should be black: {rgb:?}"
        );
    }

    #[test]
    fn test_film_reflectance_extremes_at_quarter_and_half_wave() {
        let n_film = 1.33_f32;
        let wavelength_nm = 532.0_f32;
        let surface_reflectance = fresnel_unpolarized(1.0, 1.0, 1.0, n_film);
        let peak = 4.0 * surface_reflectance / (1.0 + surface_reflectance).powi(2);

        let quarter_wave_nm = wavelength_nm / (4.0 * n_film);
        let at_quarter_wave = film_reflectance(
            quarter_wave_nm,
            1.0,
            n_film,
            wavelength_nm,
            surface_reflectance,
        );
        assert!(
            (at_quarter_wave - peak).abs() < 1e-5,
            "quarter wave: {at_quarter_wave} vs peak {peak}"
        );

        let at_half_wave = film_reflectance(
            2.0 * quarter_wave_nm,
            1.0,
            n_film,
            wavelength_nm,
            surface_reflectance,
        );
        assert!(
            at_half_wave < 1e-5,
            "half wave should be dark: {at_half_wave}"
        );
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
