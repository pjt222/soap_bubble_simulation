    #[test]
    #[ignore] // Measurement for docs/evidence/2026-11-13-ray-seeds (lavapipe: about 4 min)
    fn measure_accumulation_error() {
        // How close the app's faded deposit texture comes to the expected one, with the ray
        // seed advancing every frame against a frozen seed. The scatterers stay still
        // (update_scatterers always gets film time 0), so the expected texture is fixed and
        // only the ray sampling differs: update_params gets film time = frame number, which
        // advances the seed through RaySeedClock, or 0, which freezes it.
        let Some((device, queue)) = test_device() else {
            panic!("no GPU adapter");
        };
        const REFERENCE_FRAMES: u32 = 128;
        const FADED_FRAMES: usize = 40;
        const MEASURED_FROM: usize = 30;
        let thickness = uniform_thickness();
        let thickness_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&thickness),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let texture_bytes = 512 * 256 * 4;
        let deposit_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: texture_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: texture_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // One frame: params at `params_film_time`, scatterers at film time 0
        let run_frame = |simulator: &mut BranchedFlowSimulator,
                         params_film_time: f32,
                         clear: bool|
         -> Vec<u32> {
            simulator.update_params(&queue, params_film_time);
            simulator.update_scatterers(&queue, 0.0);
            let mut encoder = device.create_command_encoder(&Default::default());
            if clear {
                encoder.clear_buffer(&deposit_buffer, 0, None);
            }
            simulator.step(&mut encoder, None, None);
            encoder.copy_buffer_to_buffer(&deposit_buffer, 0, &readback_buffer, 0, texture_bytes);
            queue.submit(Some(encoder.finish()));
            readback_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, |result| result.expect("map"));
            device.poll(wgpu::Maintain::Wait);
            let texels =
                bytemuck::cast_slice::<u8, u32>(&readback_buffer.slice(..).get_mapped_range())
                    .to_vec();
            readback_buffer.unmap();
            texels
        };
        let new_simulator = || {
            let mut simulator =
                BranchedFlowSimulator::new(&device, &thickness_buffer, &deposit_buffer);
            simulator.enabled = true;
            simulator
        };

        // Reference: mean of single frames (texture cleared each frame) with seeds 41..168,
        // which neither faded run uses (they use seeds 0..39, or 0 alone)
        let mut reference_simulator = new_simulator();
        let mut reference = vec![0f64; 512 * 256];
        let mut single_frames = Vec::new();
        // Move the reference clock to seed 40; its first frame then traces with seed 41
        for skip in 0..=FADED_FRAMES as u32 {
            reference_simulator.ray_seed_clock.seed_for(-(skip as f32) - 1.0);
        }
        for frame in 0..REFERENCE_FRAMES {
            let texels = run_frame(&mut reference_simulator, 10_000.0 + frame as f32, true);
            for (sum, &texel) in reference.iter_mut().zip(&texels) {
                *sum += texel as f64 / REFERENCE_FRAMES as f64;
            }
            if frame < 4 {
                single_frames.push(texels);
            }
        }
        let reference_total: f64 = reference.iter().sum();
        let display = |value: f64| (value / 64.0).powi(2).min(3.0);
        // Normalised L1 distance between `texels` and the reference scaled to the same
        // total, over raw texels and over displayed values
        let distance = |texels: &[u32]| -> (f64, f64) {
            let total: f64 = texels.iter().map(|&texel| texel as f64).sum();
            let scale = total / reference_total;
            let (mut raw, mut shown, mut shown_total) = (0f64, 0f64, 0f64);
            for (&texel, &expected) in texels.iter().zip(&reference) {
                raw += (texel as f64 - expected * scale).abs();
                shown += (display(texel as f64) - display(expected * scale)).abs();
                shown_total += display(expected * scale);
            }
            (raw / total, shown / shown_total)
        };
        let mean = |values: &[(f64, f64)]| {
            let count = values.len() as f64;
            (
                values.iter().map(|value| value.0).sum::<f64>() / count,
                values.iter().map(|value| value.1).sum::<f64>() / count,
            )
        };
        let single: Vec<_> = single_frames.iter().map(|texels| distance(texels)).collect();
        println!(
            "ACCUMULATION single frame (cleared), {} frames: raw {:.4}, displayed {:.4}",
            single.len(),
            mean(&single).0,
            mean(&single).1
        );
        for (label, advance_seed) in [("seed advancing", true), ("seed frozen", false)] {
            let mut simulator = new_simulator();
            let mut distances = Vec::new();
            for frame in 0..FADED_FRAMES {
                let params_film_time = if advance_seed { frame as f32 } else { 0.0 };
                let texels = run_frame(&mut simulator, params_film_time, false);
                if frame >= MEASURED_FROM {
                    distances.push(distance(&texels));
                }
            }
            let (raw, shown) = mean(&distances);
            println!(
                "ACCUMULATION faded, {label}, frames {MEASURED_FROM}-{}: raw {raw:.4}, displayed {shown:.4}",
                FADED_FRAMES - 1
            );
        }
        println!(
            "ACCUMULATION reference: mean of {REFERENCE_FRAMES} cleared frames, seeds {}..{}",
            FADED_FRAMES + 1,
            FADED_FRAMES as u32 + REFERENCE_FRAMES
        );
    }
