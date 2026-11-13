    #[test]
    #[ignore] // Measurement for docs/evidence/2026-11-13-ray-seeds (lavapipe: about 5 min)
    fn measure_frame_to_frame_change() {
        let Some((device, queue)) = test_device() else {
            panic!("no GPU adapter");
        };
        let display = |texel: u32| ((texel as f64 / 64.0).powi(2)).min(3.0);
        for (label, film_seconds_per_frame, freeze_seeds) in [
            ("paused, per-frame seeds", 0.0f32, false),
            ("playing 60 fps, per-frame seeds", 1.0 / 60.0, false),
            ("playing 60 fps, frozen seeds", 1.0 / 60.0, true),
            ("paused, frozen seeds", 0.0, true),
        ] {
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
            let mut simulator =
                BranchedFlowSimulator::new(&device, &thickness_buffer, &deposit_buffer);
            simulator.enabled = true;
            let mut previous: Option<Vec<u32>> = None;
            let (mut texel_ratios, mut display_ratios) = (Vec::new(), Vec::new());
            for frame in 0..60usize {
                if freeze_seeds {
                    simulator.ray_seed_clock = RaySeedClock::default();
                }
                let film_time = frame as f32 * film_seconds_per_frame;
                simulator.update_params(&queue, film_time);
                simulator.update_scatterers(&queue, film_time);
                let mut encoder = device.create_command_encoder(&Default::default());
                simulator.step(&mut encoder, None, None);
                encoder.copy_buffer_to_buffer(
                    &deposit_buffer,
                    0,
                    &readback_buffer,
                    0,
                    texture_bytes,
                );
                queue.submit(Some(encoder.finish()));
                readback_buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, |result| result.expect("map"));
                device.poll(wgpu::Maintain::Wait);
                let texels: Vec<u32> =
                    bytemuck::cast_slice::<u8, u32>(&readback_buffer.slice(..).get_mapped_range())
                        .to_vec();
                readback_buffer.unmap();
                if frame >= 40 {
                    let before = previous.as_ref().unwrap();
                    let (mut diff, mut total, mut display_diff, mut display_total) =
                        (0f64, 0f64, 0f64, 0f64);
                    for (&now, &then) in texels.iter().zip(before) {
                        diff += (now as f64 - then as f64).abs();
                        total += now as f64;
                        display_diff += (display(now) - display(then)).abs();
                        display_total += display(now);
                    }
                    texel_ratios.push(diff / total);
                    display_ratios.push(display_diff / display_total);
                }
                previous = Some(texels);
            }
            let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
            println!(
                "MEASURE {label}: frame-to-frame change, frames 40-59: texels {:.4}, displayed {:.4}",
                mean(&texel_ratios),
                mean(&display_ratios)
            );
        }
    }
