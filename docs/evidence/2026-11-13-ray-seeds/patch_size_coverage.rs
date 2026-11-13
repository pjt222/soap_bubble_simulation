    #[test]
    #[ignore] // Measurement for docs/evidence/2026-11-13-ray-seeds (lavapipe: a few seconds)
    fn measure_patch_size_coverage() {
        let Some((device, queue)) = test_device() else {
            panic!("no GPU adapter");
        };
        for half_size in [0.158f32, 0.18, 0.20, 0.25, 0.30] {
            let texels = one_frame_deposits(&device, &queue, &uniform_thickness(), |simulator| {
                simulator.params.patch_center_u = 0.5;
                simulator.params.patch_half_size = half_size;
            });
            let coverage = DepositCoverage::of(&texels, 512, 256);
            println!("COVERAGE half size {half_size}: {coverage:?}");
        }
    }
