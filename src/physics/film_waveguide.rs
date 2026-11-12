//! The soap film as a slab waveguide for light travelling in the film plane (#45).
//!
//! A free-standing film (air | film | air) guides light like a symmetric slab. Its
//! fundamental TE mode travels with an effective index `n_eff(d)` between 1 (air, for a
//! vanishing film) and `n_film` (a thick film), set by the film thickness `d`. A guided
//! ray then obeys the 2D ray equation `dT/ds = grad_perp ln n_eff(d)` (Patsyk et al.
//! 2020, "Observation of branched flow of light"), so it bends toward thicker film. The
//! branched-flow compute shader reads `ln n_eff` from the table built here and
//! differentiates it along the film (`thickness_gradient_uv` in
//! `branched_flow_compute.wgsl`).
//!
//! `n_eff` changes fastest for thin films: `(1/n_eff) dn_eff/dd` is about 1.0 per µm at
//! 100 nm, 0.125 per µm at 500 nm and 0.024 per µm at 1 µm (532 nm light). A constant
//! factor times the thickness gradient is therefore only a local approximation, which is
//! why the shader uses this table rather than a fixed coefficient.

use super::interference::wavelengths;

/// Refractive index of the film for the guided-light model (water-based soap film).
pub const GUIDED_FILM_INDEX: f64 = 1.33;

/// Vacuum wavelength of the guided light in nanometres: the green channel, near the
/// 532 nm laser of the branched-flow experiments.
pub const GUIDED_WAVELENGTH_NM: f64 = wavelengths::GREEN_NM;

/// Thickness spacing of [`ln_effective_index_table`] in micrometres (5 nm). Must match
/// `LN_NEFF_TABLE_STEP_UM` in `branched_flow_compute.wgsl` (checked by a test). At 10 nm
/// the interpolated slope was off by up to 50% below 30 nm, where `ln n_eff` curves most.
pub const LN_NEFF_TABLE_STEP_UM: f64 = 0.005;

/// Number of entries in [`ln_effective_index_table`]: thicknesses 0 to 4 µm. Beyond
/// 4 µm `n_eff` is within 0.2% of `n_film`, and the shader clamps to the last entry.
pub const LN_NEFF_TABLE_LEN: usize = 801;

/// Effective index of the fundamental TE mode (TE0) of a symmetric slab of the given
/// thickness and refractive index in air, for light of the given vacuum wavelength.
///
/// Solves the even-mode dispersion relation `kappa tan(kappa d / 2) = gamma`, with
/// `kappa = k0 sqrt(n_film^2 - n_eff^2)` and `gamma = k0 sqrt(n_eff^2 - 1)`, in the
/// normalised form `X tan X = sqrt(V^2 - X^2)`, where `X = kappa d / 2` and
/// `V = (k0 d / 2) sqrt(n_film^2 - 1)`. TE0 has no cutoff, and the root is unique in
/// `0 < X < min(V, pi/2)`, where the left side minus the right side increases
/// monotonically from `-V`; it is found by bisection. Returns 1.0 for a zero thickness.
pub fn te0_effective_index(thickness_nm: f64, film_index: f64, wavelength_nm: f64) -> f64 {
    assert!(
        thickness_nm >= 0.0,
        "negative film thickness {thickness_nm} nm"
    );
    assert!(
        film_index > 1.0,
        "film index {film_index} does not guide light in air"
    );
    if thickness_nm == 0.0 {
        return 1.0;
    }
    let free_space_wavenumber = 2.0 * std::f64::consts::PI / wavelength_nm;
    let half_phase = free_space_wavenumber * thickness_nm / 2.0;
    let normalised_frequency = half_phase * (film_index * film_index - 1.0).sqrt();
    let mismatch = |x: f64| {
        x * x.tan()
            - (normalised_frequency * normalised_frequency - x * x)
                .max(0.0)
                .sqrt()
    };
    let (mut low, mut high) = (0.0, normalised_frequency.min(std::f64::consts::FRAC_PI_2));
    for _ in 0..200 {
        let middle = 0.5 * (low + high);
        if mismatch(middle) < 0.0 {
            low = middle;
        } else {
            high = middle;
        }
    }
    let transverse_index = 0.5 * (low + high) / half_phase; // kappa / k0
    (film_index * film_index - transverse_index * transverse_index).sqrt()
}

/// `ln n_eff` of the film's TE0 mode (at [`GUIDED_FILM_INDEX`] and
/// [`GUIDED_WAVELENGTH_NM`]) at the thicknesses `i * LN_NEFF_TABLE_STEP_UM`, for
/// `i = 0..LN_NEFF_TABLE_LEN`. The branched-flow shader interpolates it linearly.
pub fn ln_effective_index_table() -> Vec<f32> {
    (0..LN_NEFF_TABLE_LEN)
        .map(|index| {
            let thickness_nm = index as f64 * LN_NEFF_TABLE_STEP_UM * 1000.0;
            te0_effective_index(thickness_nm, GUIDED_FILM_INDEX, GUIDED_WAVELENGTH_NM).ln() as f32
        })
        .collect()
}

/// The film thickness in nanometres at which the TE0 mode (at [`GUIDED_FILM_INDEX`] and
/// [`GUIDED_WAVELENGTH_NM`]) has the given `ln n_eff`: the inverse of
/// [`te0_effective_index`], found by bisection on 0 to 100 µm. Used to build test fields
/// with a prescribed GRIN force. Panics unless `0 <= ln_effective_index < ln n_film`.
pub fn thickness_for_ln_effective_index(ln_effective_index: f64) -> f64 {
    assert!(
        (0.0..GUIDED_FILM_INDEX.ln()).contains(&ln_effective_index),
        "ln n_eff {ln_effective_index} is outside [0, ln {GUIDED_FILM_INDEX})"
    );
    let (mut low, mut high) = (0.0, 100_000.0);
    for _ in 0..200 {
        let middle = 0.5 * (low + high);
        let ln_middle = te0_effective_index(middle, GUIDED_FILM_INDEX, GUIDED_WAVELENGTH_NM).ln();
        if ln_middle < ln_effective_index {
            low = middle;
        } else {
            high = middle;
        }
    }
    0.5 * (low + high)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guided_index(thickness_nm: f64) -> f64 {
        te0_effective_index(thickness_nm, GUIDED_FILM_INDEX, GUIDED_WAVELENGTH_NM)
    }

    #[test]
    fn te0_index_matches_the_audit_table() {
        // docs/audit-2026-10-09-evidence/sota/slab_neff.py and branched/check_neff.py,
        // two independent bisection solvers (findings sota-06, branched-14)
        for (thickness_nm, expected) in [
            (30.0, 1.0089),
            (100.0, 1.0740),
            (500.0, 1.2745),
            (1000.0, 1.3112),
            (2000.0, 1.3245),
        ] {
            let index = guided_index(thickness_nm);
            assert!(
                (index - expected).abs() < 1e-4,
                "n_eff({thickness_nm} nm) = {index:.5}, audit {expected}"
            );
        }
    }

    #[test]
    fn te0_index_satisfies_the_dispersion_relation() {
        let free_space_wavenumber = 2.0 * std::f64::consts::PI / GUIDED_WAVELENGTH_NM;
        for thickness_nm in [10.0, 250.0, 800.0, 3000.0] {
            let index = guided_index(thickness_nm);
            let kappa = free_space_wavenumber * (GUIDED_FILM_INDEX.powi(2) - index * index).sqrt();
            let gamma = free_space_wavenumber * (index * index - 1.0).sqrt();
            let residual = kappa * (kappa * thickness_nm / 2.0).tan() - gamma;
            assert!(
                residual.abs() < 1e-9 * gamma,
                "{thickness_nm} nm: residual {residual:e}"
            );
        }
    }

    #[test]
    fn te0_index_rises_from_air_to_the_film_index() {
        assert_eq!(guided_index(0.0), 1.0);
        let mut previous = 1.0;
        for step in 1..=400 {
            let index = guided_index(step as f64 * 10.0);
            assert!(index > previous, "n_eff not increasing at {} nm", step * 10);
            assert!(index < GUIDED_FILM_INDEX);
            previous = index;
        }
        assert!(
            GUIDED_FILM_INDEX - previous < 0.003,
            "n_eff(4 um) = {previous}"
        );
    }

    #[test]
    fn grin_coefficient_matches_the_audit_at_500_nm() {
        // (1/n_eff) dn_eff/dd = d ln n_eff / dd: 0.125 per um at 500 nm (branched-14)
        let coefficient_per_um = (guided_index(501.0).ln() - guided_index(499.0).ln()) / 2e-3;
        assert!(
            (coefficient_per_um - 0.125).abs() < 0.003,
            "d ln n_eff / dd at 500 nm = {coefficient_per_um} per um"
        );
    }

    #[test]
    fn table_samples_ln_index_on_its_grid() {
        let table = ln_effective_index_table();
        assert_eq!(table.len(), LN_NEFF_TABLE_LEN);
        assert_eq!(table[0], 0.0);
        assert!((table[100] as f64 - guided_index(500.0).ln()).abs() < 1e-6);
        assert!((table[LN_NEFF_TABLE_LEN - 1] as f64 - guided_index(4000.0).ln()).abs() < 1e-6);
    }

    #[test]
    fn table_interpolation_error_is_small() {
        // The shader interpolates linearly; compare midpoints with the exact value
        let table = ln_effective_index_table();
        let worst = (0..LN_NEFF_TABLE_LEN - 1)
            .map(|index| {
                let midpoint_nm = (index as f64 + 0.5) * LN_NEFF_TABLE_STEP_UM * 1000.0;
                let interpolated = 0.5 * (table[index] as f64 + table[index + 1] as f64);
                (interpolated - guided_index(midpoint_nm).ln()).abs()
            })
            .fold(0.0, f64::max);
        assert!(worst < 7e-5, "worst midpoint error {worst:e}");
    }

    #[test]
    fn table_slope_matches_the_grin_coefficient() {
        // The ray force is a difference of interpolated values, i.e. the slope of the
        // interval a thickness falls in. Compare it with the exact d ln n_eff / dd at
        // points across each interval. Below about 60 nm the curvature makes the error
        // exceed 3%; such films are already below the 30 nm rupture threshold of the
        // drainage model, or close to it.
        let table = ln_effective_index_table();
        let step_nm = LN_NEFF_TABLE_STEP_UM * 1000.0;
        let mut worst = (0.0f64, 0.0f64);
        for index in (60.0 / step_nm) as usize..LN_NEFF_TABLE_LEN - 1 {
            let table_slope = (table[index + 1] as f64 - table[index] as f64) / step_nm;
            for fraction in [0.0, 0.5, 1.0] {
                let thickness_nm = (index as f64 + fraction) * step_nm;
                let exact_slope = (guided_index(thickness_nm + 0.01).ln()
                    - guided_index(thickness_nm - 0.01).ln())
                    / 0.02;
                let relative_error = (table_slope / exact_slope - 1.0).abs();
                if relative_error > worst.0 {
                    worst = (relative_error, thickness_nm);
                }
            }
        }
        assert!(
            worst.0 < 0.03,
            "slope off by {:.1}% at {} nm",
            worst.0 * 100.0,
            worst.1
        );
    }

    #[test]
    fn shader_table_step_matches() {
        let shader = include_str!("../render/shaders/branched_flow_compute.wgsl");
        let line = shader
            .lines()
            .find(|line| line.starts_with("const LN_NEFF_TABLE_STEP_UM"))
            .expect("LN_NEFF_TABLE_STEP_UM in branched_flow_compute.wgsl");
        let value: f64 = line
            .split('=')
            .nth(1)
            .and_then(|value| value.trim().trim_end_matches(';').parse().ok())
            .unwrap_or_else(|| panic!("cannot parse {line:?}"));
        assert_eq!(value, LN_NEFF_TABLE_STEP_UM, "{line}");
    }

    #[test]
    fn thickness_inverse_round_trips() {
        for thickness_nm in [50.0, 300.0, 500.0, 1200.0] {
            let recovered = thickness_for_ln_effective_index(guided_index(thickness_nm).ln());
            assert!(
                (recovered - thickness_nm).abs() < 1e-6 * thickness_nm,
                "{thickness_nm} nm -> {recovered} nm"
            );
        }
    }
}
