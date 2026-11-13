
    // Review measurements for docs/evidence/2026-11-13-ray-seeds (run at f8d0998)

    #[test]
    #[ignore]
    fn review_measure_pause_transition() {
        let Some((device, queue)) = test_device() else {
            panic!("no GPU adapter");
        };
        let display = |texel: u32| ((texel as f64 / 64.0).powi(2)).min(3.0);
        let raw = |texel: u32| texel as f64;
        let distance = |now: &[u32], then: &[u32], map: &dyn Fn(u32) -> f64| {
            let (mut diff, mut total) = (0f64, 0f64);
            for (&a, &b) in now.iter().zip(then) {
                diff += (map(a) - map(b)).abs();
                total += map(a);
            }
            diff / total
        };
        let playing_frames = 40usize;
        let paused_frames = 60usize;
        let mut film_times: Vec<f32> =
            (0..playing_frames).map(|frame| frame as f32 / 60.0).collect();
        let pause_time = *film_times.last().unwrap();
        film_times.extend(std::iter::repeat_n(pause_time, paused_frames));
        let run = FrameRun {
            film_times,
            clear_each_frame: false,
        };
        let frames = frame_deposits(&device, &queue, &uniform_thickness(), |_| {}, run);
        let mut last_changing = None;
        for frame in playing_frames..frames.len() {
            let changed_texels = frames[frame]
                .iter()
                .zip(&frames[frame - 1])
                .filter(|(a, b)| a != b)
                .count();
            if changed_texels > 0 {
                last_changing = Some(frame);
            }
            println!(
                "PAUSE frame {frame} (paused frame {}): change raw {:.4} displayed {:.4}, texels changed {changed_texels}",
                frame - playing_frames + 1,
                distance(&frames[frame], &frames[frame - 1], &raw),
                distance(&frames[frame], &frames[frame - 1], &display),
            );
        }
        let at_pause = &frames[playing_frames - 1];
        let settled = frames.last().unwrap();
        println!(
            "PAUSE image at the pause instant vs settled paused image: raw {:.4} displayed {:.4}; last frame that changed: {:?}",
            distance(settled, at_pause, &raw),
            distance(settled, at_pause, &display),
            last_changing
        );
        let lit = |texture: &[u32]| texture.iter().filter(|&&t| t > 0).count();
        println!(
            "PAUSE lit texels: at pause {}, settled {}",
            lit(at_pause),
            lit(settled)
        );
    }

    #[test]
    #[ignore]
    fn review_measure_coverage_seed_band() {
        let Some((device, queue)) = test_device() else {
            panic!("no GPU adapter");
        };
        for half_size in [0.158f32, 0.18, 0.20, 0.25, 0.30] {
            let mut lits = Vec::new();
            for seed in 0u32..8 {
                let texels = one_frame_deposits(&device, &queue, &uniform_thickness(), |simulator| {
                    simulator.params.patch_center_u = 0.5;
                    simulator.params.patch_half_size = half_size;
                    simulator.ray_seed_clock = RaySeedClock {
                        index: seed,
                        film_time: Some(0.0),
                    };
                });
                lits.push(DepositCoverage::of(&texels, 512, 256).lit);
            }
            let mean = lits.iter().sum::<f64>() / lits.len() as f64;
            let min = lits.iter().cloned().fold(f64::MAX, f64::min);
            let max = lits.iter().cloned().fold(f64::MIN, f64::max);
            println!(
                "BAND half size {half_size}: seeds 0-7 lit mean {:.4} min {:.4} max {:.4} all {:?}",
                mean, min, max, lits
            );
        }
    }
