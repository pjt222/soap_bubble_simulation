//! UI state and display info for the egui control panel.
//!
//! Extracted from `pipeline.rs` to eliminate the 95-parameter `build_ui_inner` function.
//! `UiState` holds all mutable parameters that the UI can modify, while `UiDisplayInfo`
//! holds read-only values displayed in the UI.

/// Mutable UI parameters — modified by egui widgets, then applied back to the pipeline.
pub struct UiState {
    // Film properties (synced to/from BubbleUniform)
    pub thickness: f32,
    pub refractive_index: f32,
    pub interference_intensity: f32,
    pub base_alpha: f32,
    pub edge_alpha: f32,
    pub bg_r: f32,
    pub bg_g: f32,
    pub bg_b: f32,

    // Mesh
    pub subdivision: u32,
    pub msaa_samples: u32,
    pub lod_enabled: bool,
    pub edge_smoothing_mode: u32,

    // Animation
    pub rotation_playing: bool,
    pub rotation_speed: f32,
    pub film_playing: bool,
    pub film_speed: f32,
    pub swirl_intensity: f32,
    pub drainage_speed: f32,
    pub pattern_scale: f32,

    // Export
    pub screenshot_requested: bool,
    pub recording: bool,

    // External forces
    pub forces_enabled: bool,
    pub wind_strength: f32,
    pub buoyancy_strength: f32,

    // Physics drainage (CPU)
    pub physics_drainage_enabled: bool,
    pub drainage_time_scale: f32,
    pub reset_drainage_requested: bool,

    // GPU drainage
    pub gpu_drainage_enabled: bool,
    pub gpu_drainage_time_scale: f32,
    pub gpu_drainage_steps: u32,
    pub reset_gpu_drainage_requested: bool,
    pub marangoni_enabled: bool,
    pub marangoni_coeff: f32,

    // Deformation
    pub deformation_enabled: bool,
    pub aspect_ratio: f32,

    // Caustics
    pub caustics_enabled: bool,
    pub caustic_intensity: f32,
    pub caustic_sharpness: f32,
    pub ground_y: f32,

    // Branched flow (ray-traced laser through film)
    pub branched_flow_enabled: bool,
    pub branched_flow_intensity: f32,
    pub branched_flow_sharpness: f32,
    pub laser_azimuth: f32,
    pub laser_elevation: f32,
    pub beam_spread: f32,
    pub bend_strength: f32,
    pub num_rays: u32,
    pub num_scatterers: u32,
    pub scatterer_strength: f32,
    pub scatterer_radius: f32,
    pub particle_weight: f32,

    // Patch view
    pub patch_view_enabled: bool,
    pub patch_center_u: f32,
    pub patch_center_v: f32,
    pub patch_half_size: f32,

    // Foam
    pub foam_enabled: bool,
    pub foam_paused: bool,
    pub foam_time_scale: f32,
    pub add_bubble_requested: bool,
    pub reset_foam_requested: bool,
    pub regenerate_foam_requested: bool,
    pub foam_gen_params: crate::physics::foam_generation::GenerationParams,
}

/// Read-only values displayed in the UI (not modifiable by widgets).
pub struct UiDisplayInfo {
    pub camera_distance: f32,
    pub camera_yaw: f32,
    pub camera_pitch: f32,
    pub fps: f32,
    pub width: u32,
    pub height: u32,
    pub num_triangles: u32,
    pub time: f32,
    pub current_lod_level: u32,
    pub frame_counter: u32,
    pub drainage_sim_time: f64,
    pub gpu_drainage_time: f64,
    pub bubble_pos: [f32; 3],
    pub has_drainage_sim: bool,
    pub foam_stats: (usize, usize, usize),
}

impl UiState {
    /// Build the egui UI panel. All mutable state is modified in-place via widgets.
    pub fn build_ui(&mut self, ctx: &egui::Context, info: &UiDisplayInfo) {
        egui::Window::new("Soap Bubble")
            .default_pos([10.0, 10.0])
            .default_width(280.0)
            .resizable(false)
            .show(ctx, |ui| {
                ui.heading("Film Properties");
                ui.separator();

                ui.add(
                    egui::Slider::new(&mut self.thickness, 100.0..=1500.0)
                        .text("Thickness")
                        .suffix(" nm"),
                );

                ui.add(
                    egui::Slider::new(&mut self.refractive_index, 1.0..=2.0)
                        .text("Refractive index")
                        .fixed_decimals(2),
                );

                // Preset buttons
                ui.horizontal(|ui| {
                    if ui.button("Soap").clicked() {
                        self.refractive_index = 1.33;
                        self.thickness = 500.0;
                    }
                    if ui.button("Oil").clicked() {
                        self.refractive_index = 1.47;
                        self.thickness = 300.0;
                    }
                    if ui.button("Thin").clicked() {
                        self.thickness = 150.0;
                    }
                    if ui.button("Thick").clicked() {
                        self.thickness = 1200.0;
                    }
                });

                ui.separator();
                ui.heading("Render Settings");
                ui.separator();

                ui.add(
                    egui::Slider::new(&mut self.interference_intensity, 0.5..=10.0)
                        .text("Color intensity")
                        .fixed_decimals(1),
                );

                ui.add(
                    egui::Slider::new(&mut self.base_alpha, 0.0..=1.0)
                        .text("Base opacity")
                        .fixed_decimals(2),
                );

                ui.add(
                    egui::Slider::new(&mut self.edge_alpha, 0.0..=1.0)
                        .text("Edge opacity")
                        .fixed_decimals(2),
                );

                ui.horizontal(|ui| {
                    ui.label("Edge blend:");
                    egui::ComboBox::from_id_salt("edge_blend")
                        .selected_text(match self.edge_smoothing_mode {
                            1 => "Smoothstep",
                            2 => "Power",
                            _ => "Linear",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.edge_smoothing_mode, 0, "Linear");
                            ui.selectable_value(&mut self.edge_smoothing_mode, 1, "Smoothstep");
                            ui.selectable_value(&mut self.edge_smoothing_mode, 2, "Power");
                        });
                });

                ui.horizontal(|ui| {
                    ui.label("Anti-aliasing:");
                    egui::ComboBox::from_id_salt("msaa")
                        .selected_text(match self.msaa_samples {
                            1 => "Off",
                            2 => "2x MSAA",
                            _ => "4x MSAA",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.msaa_samples, 1, "Off");
                            ui.selectable_value(&mut self.msaa_samples, 2, "2x MSAA");
                            ui.selectable_value(&mut self.msaa_samples, 4, "4x MSAA");
                        });
                });

                ui.separator();
                ui.heading("Mesh & Background");
                ui.separator();

                ui.checkbox(&mut self.lod_enabled, "Auto LOD");

                if self.lod_enabled {
                    ui.label(format!(
                        "LOD Level: {} ({} tri)",
                        info.current_lod_level, info.num_triangles
                    ));
                } else {
                    ui.add(egui::Slider::new(&mut self.subdivision, 1..=5).text("Mesh detail"));
                }

                ui.horizontal(|ui| {
                    ui.label("Background:");
                    let mut color = [self.bg_r, self.bg_g, self.bg_b];
                    if ui.color_edit_button_rgb(&mut color).changed() {
                        self.bg_r = color[0];
                        self.bg_g = color[1];
                        self.bg_b = color[2];
                    }
                });

                ui.separator();
                ui.heading("Animation");
                ui.separator();

                ui.collapsing("Camera Orbit", |ui| {
                    ui.horizontal(|ui| {
                        let play_text = if self.rotation_playing {
                            "\u{23F8} Pause"
                        } else {
                            "\u{25B6} Play"
                        };
                        if ui.button(play_text).clicked() {
                            self.rotation_playing = !self.rotation_playing;
                        }
                    });

                    ui.add(
                        egui::Slider::new(&mut self.rotation_speed, 0.1..=2.0)
                            .text("Speed")
                            .suffix(" rad/s")
                            .fixed_decimals(2),
                    );
                });

                ui.collapsing("Film Dynamics", |ui| {
                    ui.horizontal(|ui| {
                        let play_text = if self.film_playing {
                            "\u{23F8} Pause"
                        } else {
                            "\u{25B6} Play"
                        };
                        if ui.button(play_text).clicked() {
                            self.film_playing = !self.film_playing;
                        }
                    });

                    ui.add(
                        egui::Slider::new(&mut self.film_speed, 0.1..=3.0)
                            .text("Speed")
                            .fixed_decimals(1),
                    );

                    ui.add(
                        egui::Slider::new(&mut self.swirl_intensity, 0.0..=2.0)
                            .text("Swirl")
                            .fixed_decimals(2),
                    );

                    ui.add(
                        egui::Slider::new(&mut self.drainage_speed, 0.0..=2.0)
                            .text("Drainage")
                            .fixed_decimals(2),
                    );

                    ui.add(
                        egui::Slider::new(&mut self.pattern_scale, 0.5..=3.0)
                            .text("Pattern scale")
                            .fixed_decimals(1),
                    );
                });

                ui.collapsing("External Forces", |ui| {
                    ui.checkbox(&mut self.forces_enabled, "Enable forces");

                    if self.forces_enabled {
                        ui.add(
                            egui::Slider::new(&mut self.wind_strength, 0.0..=0.5)
                                .text("Wind")
                                .suffix(" m/s\u{00b2}")
                                .fixed_decimals(2),
                        );

                        ui.add(
                            egui::Slider::new(&mut self.buoyancy_strength, 0.0..=0.1)
                                .text("Buoyancy")
                                .suffix(" m/s\u{00b2}")
                                .fixed_decimals(3),
                        );

                        ui.separator();
                        ui.label(format!(
                            "Position: ({:.3}, {:.3}, {:.3})",
                            info.bubble_pos[0], info.bubble_pos[1], info.bubble_pos[2]
                        ));
                    }
                });

                ui.collapsing("Physics Drainage (CPU)", |ui| {
                    ui.checkbox(&mut self.physics_drainage_enabled, "Enable CPU simulation");

                    if self.physics_drainage_enabled {
                        ui.add(
                            egui::Slider::new(&mut self.drainage_time_scale, 1.0..=500.0)
                                .text("Time scale")
                                .logarithmic(true)
                                .fixed_decimals(0),
                        );

                        ui.separator();
                        if info.has_drainage_sim {
                            ui.label(format!("Sim time: {:.2} s", info.drainage_sim_time));
                            ui.label(format!("Thickness: {:.0} nm", self.thickness));

                            if ui.button("Reset").clicked() {
                                self.reset_drainage_requested = true;
                            }
                        } else {
                            ui.label("Initializing...");
                        }
                    }
                });

                ui.collapsing("GPU Drainage", |ui| {
                    ui.checkbox(&mut self.gpu_drainage_enabled, "Enable GPU simulation");

                    if self.gpu_drainage_enabled {
                        ui.add(
                            egui::Slider::new(&mut self.gpu_drainage_time_scale, 10.0..=500.0)
                                .text("Time scale")
                                .logarithmic(true)
                                .fixed_decimals(0),
                        );

                        ui.add(
                            egui::Slider::new(&mut self.gpu_drainage_steps, 1..=50)
                                .text("Steps/frame"),
                        );

                        ui.separator();
                        ui.checkbox(&mut self.marangoni_enabled, "Marangoni effect");
                        if self.marangoni_enabled {
                            ui.add(
                                egui::Slider::new(&mut self.marangoni_coeff, 0.001..=0.1)
                                    .text("Strength")
                                    .logarithmic(true)
                                    .fixed_decimals(3),
                            );
                            ui.label("Surfactant-driven flow");
                        }

                        ui.separator();
                        ui.label(format!("Sim time: {:.2} s", info.gpu_drainage_time));
                        ui.label("Grid: 128\u{00d7}64 (8k cells)");

                        if ui.button("Reset").clicked() {
                            self.reset_gpu_drainage_requested = true;
                        }
                    }

                    ui.label("\u{26a1} Real-time PDE solver");
                });

                ui.collapsing("Caustics / Branched Flow", |ui| {
                    // Ground-plane caustics (projected below bubble)
                    ui.label("Ground Caustics");
                    ui.checkbox(&mut self.caustics_enabled, "Enable ground caustics");

                    if self.caustics_enabled {
                        if !self.gpu_drainage_enabled {
                            ui.colored_label(
                                egui::Color32::YELLOW,
                                "\u{26a0} Requires GPU Drainage",
                            );
                        }

                        ui.add(
                            egui::Slider::new(&mut self.caustic_intensity, 0.5..=5.0)
                                .text("Intensity")
                                .fixed_decimals(1),
                        );

                        ui.add(
                            egui::Slider::new(&mut self.caustic_sharpness, 1.0..=3.0)
                                .text("Sharpness")
                                .fixed_decimals(1),
                        );

                        ui.add(
                            egui::Slider::new(&mut self.ground_y, -0.15..=-0.05)
                                .text("Ground height")
                                .suffix(" m")
                                .fixed_decimals(2),
                        );
                    }

                    ui.separator();

                    // Ray-traced branched flow (laser propagating WITHIN film)
                    ui.label("In-Film Laser");
                    ui.checkbox(&mut self.branched_flow_enabled, "Enable laser in film");

                    if self.branched_flow_enabled {
                        if !self.gpu_drainage_enabled {
                            ui.colored_label(
                                egui::Color32::YELLOW,
                                "\u{26a0} Requires GPU Drainage",
                            );
                        }

                        ui.label("Injection Point");
                        ui.add(
                            egui::Slider::new(&mut self.laser_azimuth, -180.0..=180.0)
                                .text("Azimuth")
                                .suffix("\u{00b0}")
                                .fixed_decimals(0),
                        );

                        ui.add(
                            egui::Slider::new(&mut self.laser_elevation, -90.0..=90.0)
                                .text("Elevation")
                                .suffix("\u{00b0}")
                                .fixed_decimals(0),
                        );

                        ui.separator();
                        ui.label("Beam Properties");

                        ui.add(
                            egui::Slider::new(&mut self.beam_spread, 1.0..=45.0)
                                .text("Spread")
                                .suffix("\u{00b0}")
                                .fixed_decimals(0),
                        );

                        ui.add(
                            egui::Slider::new(&mut self.bend_strength, 0.01..=50.0)
                                .text("GRIN bending")
                                .logarithmic(true)
                                .fixed_decimals(3),
                        );

                        ui.add(
                            egui::Slider::new(&mut self.num_rays, 256..=65536)
                                .text("Ray count")
                                .logarithmic(true),
                        );

                        ui.separator();
                        ui.label("Display");

                        ui.add(
                            egui::Slider::new(&mut self.branched_flow_intensity, 0.1..=20.0)
                                .text("Brightness")
                                .fixed_decimals(2),
                        );

                        ui.add(
                            egui::Slider::new(&mut self.branched_flow_sharpness, 0.5..=3.0)
                                .text("Contrast")
                                .fixed_decimals(1),
                        );

                        ui.separator();
                        ui.label("Particle Scattering");

                        ui.add(
                            egui::Slider::new(&mut self.particle_weight, 0.0..=1.0)
                                .text("Weight")
                                .fixed_decimals(2),
                        );

                        ui.add(
                            egui::Slider::new(&mut self.num_scatterers, 100..=2000)
                                .text("Scatterers")
                                .logarithmic(true),
                        );

                        ui.add(
                            egui::Slider::new(&mut self.scatterer_strength, 0.1..=2.0)
                                .text("Strength")
                                .logarithmic(true)
                                .fixed_decimals(2),
                        );

                        ui.add(
                            egui::Slider::new(&mut self.scatterer_radius, 0.01..=0.1)
                                .text("Radius")
                                .fixed_decimals(3),
                        );

                        ui.separator();
                        ui.label("Hybrid model: GRIN + particles");
                        ui.label("Weight 0 = smooth GRIN only");
                        ui.label("Weight 1 = particle scatter only");

                        ui.separator();
                        ui.label("Patch View Mode");
                        ui.checkbox(&mut self.patch_view_enabled, "Focus on patch");

                        if self.patch_view_enabled {
                            ui.add(
                                egui::Slider::new(&mut self.patch_center_u, 0.1..=0.9)
                                    .text("Center U")
                                    .fixed_decimals(2),
                            );

                            ui.add(
                                egui::Slider::new(&mut self.patch_center_v, 0.1..=0.9)
                                    .text("Center V")
                                    .fixed_decimals(2),
                            );

                            ui.add(
                                egui::Slider::new(&mut self.patch_half_size, 0.05..=0.3)
                                    .text("Patch size")
                                    .fixed_decimals(3),
                            );

                            let area_percent = (self.patch_half_size * 2.0).powi(2) * 100.0;
                            ui.label(format!("~{:.1}% of sphere", area_percent));
                        }
                    }
                });

                ui.collapsing("Gravity Deformation", |ui| {
                    ui.checkbox(&mut self.deformation_enabled, "Enable deformation");

                    if self.deformation_enabled {
                        ui.add(
                            egui::Slider::new(&mut self.aspect_ratio, 0.7..=1.0)
                                .text("Aspect ratio")
                                .fixed_decimals(2),
                        );

                        ui.separator();
                        let deform_percent = (1.0 - self.aspect_ratio) * 100.0;
                        ui.label(format!("Flattening: {:.1}%", deform_percent));
                        ui.label("(1.0 = sphere, <1.0 = oblate)");
                    }
                });

                ui.collapsing("Multi-Bubble Foam", |ui| {
                    ui.checkbox(&mut self.foam_enabled, "Enable foam mode");

                    if self.foam_enabled {
                        ui.horizontal(|ui| {
                            let pause_text = if self.foam_paused {
                                "\u{25B6} Start"
                            } else {
                                "\u{23F8} Pause"
                            };
                            if ui.button(pause_text).clicked() {
                                self.foam_paused = !self.foam_paused;
                            }
                        });

                        ui.add(
                            egui::Slider::new(&mut self.foam_time_scale, 0.1..=5.0)
                                .text("Time scale")
                                .fixed_decimals(1),
                        );

                        ui.separator();
                        ui.label(format!("Bubbles: {}", info.foam_stats.0));
                        ui.label(format!("Connections: {}", info.foam_stats.1));
                        ui.label(format!("Walls: {}", info.foam_stats.2));

                        ui.horizontal(|ui| {
                            if ui.button("Add Bubble").clicked() {
                                self.add_bubble_requested = true;
                            }
                            if ui.button("Reset").clicked() {
                                self.reset_foam_requested = true;
                            }
                        });

                        ui.separator();
                        ui.heading("Generation");

                        // Bubble count
                        let mut bubble_count = self.foam_gen_params.bubble_count as i32;
                        if ui
                            .add(
                                egui::Slider::new(&mut bubble_count, 2..=30).text("Bubble count"),
                            )
                            .changed()
                        {
                            self.foam_gen_params.bubble_count = bubble_count as u32;
                        }

                        // Positioning mode dropdown
                        ui.horizontal(|ui| {
                            ui.label("Positioning:");
                            egui::ComboBox::from_id_salt("positioning_mode")
                                .selected_text(self.foam_gen_params.positioning_mode.name())
                                .show_ui(ui, |ui| {
                                    use crate::physics::foam_generation::PositioningMode;
                                    for mode in PositioningMode::all() {
                                        ui.selectable_value(
                                            &mut self.foam_gen_params.positioning_mode,
                                            *mode,
                                            mode.name(),
                                        );
                                    }
                                });
                        });

                        // Show spacing/jitter for grid modes
                        use crate::physics::foam_generation::PositioningMode;
                        let is_grid_mode = matches!(
                            self.foam_gen_params.positioning_mode,
                            PositioningMode::SimpleCubic
                                | PositioningMode::BodyCenteredCubic
                                | PositioningMode::FaceCenteredCubic
                                | PositioningMode::HexagonalClosePacked
                                | PositioningMode::PoissonDisk
                        );

                        if is_grid_mode {
                            ui.add(
                                egui::Slider::new(&mut self.foam_gen_params.spacing, 0.03..=0.10)
                                    .text("Spacing")
                                    .suffix(" m")
                                    .fixed_decimals(3),
                            );

                            ui.add(
                                egui::Slider::new(&mut self.foam_gen_params.jitter, 0.0..=0.5)
                                    .text("Jitter")
                                    .fixed_decimals(2),
                            );
                        }

                        ui.separator();

                        // Size distribution dropdown
                        ui.horizontal(|ui| {
                            ui.label("Size dist:");
                            egui::ComboBox::from_id_salt("size_distribution")
                                .selected_text(self.foam_gen_params.size_distribution.name())
                                .show_ui(ui, |ui| {
                                    use crate::physics::foam_generation::SizeDistribution;
                                    for dist in SizeDistribution::all() {
                                        ui.selectable_value(
                                            &mut self.foam_gen_params.size_distribution,
                                            *dist,
                                            dist.name(),
                                        );
                                    }
                                });
                        });

                        // Radius range (always shown)
                        ui.add(
                            egui::Slider::new(
                                &mut self.foam_gen_params.min_radius,
                                0.005..=0.03,
                            )
                            .text("Min radius")
                            .suffix(" m")
                            .fixed_decimals(3),
                        );

                        ui.add(
                            egui::Slider::new(
                                &mut self.foam_gen_params.max_radius,
                                0.02..=0.06,
                            )
                            .text("Max radius")
                            .suffix(" m")
                            .fixed_decimals(3),
                        );

                        // Context-sensitive sliders based on distribution type
                        use crate::physics::foam_generation::SizeDistribution;
                        match self.foam_gen_params.size_distribution {
                            SizeDistribution::Normal | SizeDistribution::LogNormal => {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.foam_gen_params.mean_radius,
                                        0.01..=0.04,
                                    )
                                    .text("Mean radius")
                                    .suffix(" m")
                                    .fixed_decimals(3),
                                );

                                if self.foam_gen_params.size_distribution
                                    == SizeDistribution::Normal
                                {
                                    ui.add(
                                        egui::Slider::new(
                                            &mut self.foam_gen_params.std_dev,
                                            0.001..=0.015,
                                        )
                                        .text("Std dev")
                                        .suffix(" m")
                                        .fixed_decimals(3),
                                    );
                                } else {
                                    ui.add(
                                        egui::Slider::new(
                                            &mut self.foam_gen_params.sigma,
                                            0.1..=0.8,
                                        )
                                        .text("Sigma")
                                        .fixed_decimals(2),
                                    );
                                }
                            }
                            SizeDistribution::SchulzFlory => {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.foam_gen_params.mean_radius,
                                        0.01..=0.04,
                                    )
                                    .text("Mean radius")
                                    .suffix(" m")
                                    .fixed_decimals(3),
                                );

                                ui.add(
                                    egui::Slider::new(&mut self.foam_gen_params.pdi, 1.1..=3.0)
                                        .text("PDI (Mw/Mn)")
                                        .fixed_decimals(2),
                                );
                            }
                            SizeDistribution::Bimodal => {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.foam_gen_params.mean_radius,
                                        0.01..=0.03,
                                    )
                                    .text("Mean 1")
                                    .suffix(" m")
                                    .fixed_decimals(3),
                                );

                                ui.add(
                                    egui::Slider::new(
                                        &mut self.foam_gen_params.std_dev,
                                        0.001..=0.01,
                                    )
                                    .text("Std 1")
                                    .suffix(" m")
                                    .fixed_decimals(3),
                                );

                                ui.add(
                                    egui::Slider::new(
                                        &mut self.foam_gen_params.bimodal_ratio,
                                        0.1..=0.9,
                                    )
                                    .text("Ratio")
                                    .fixed_decimals(2),
                                );

                                ui.add(
                                    egui::Slider::new(
                                        &mut self.foam_gen_params.bimodal_mean2,
                                        0.02..=0.05,
                                    )
                                    .text("Mean 2")
                                    .suffix(" m")
                                    .fixed_decimals(3),
                                );

                                ui.add(
                                    egui::Slider::new(
                                        &mut self.foam_gen_params.bimodal_std2,
                                        0.001..=0.01,
                                    )
                                    .text("Std 2")
                                    .suffix(" m")
                                    .fixed_decimals(3),
                                );
                            }
                            SizeDistribution::Uniform => {
                                // Uniform just uses min/max, already shown above
                            }
                        }

                        ui.separator();

                        if ui.button("Regenerate Foam").clicked() {
                            self.regenerate_foam_requested = true;
                        }

                        ui.separator();
                        ui.label("N-body bubble dynamics");
                        ui.label("with Plateau borders");
                    }
                });

                ui.separator();
                ui.collapsing("Camera Info", |ui| {
                    egui::Grid::new("camera_grid")
                        .num_columns(2)
                        .spacing([20.0, 4.0])
                        .show(ui, |ui| {
                            ui.label("Distance:");
                            ui.label(format!("{:.3} m", info.camera_distance));
                            ui.end_row();

                            ui.label("Yaw:");
                            ui.label(format!("{:.1}\u{00b0}", info.camera_yaw.to_degrees()));
                            ui.end_row();

                            ui.label("Pitch:");
                            ui.label(format!("{:.1}\u{00b0}", info.camera_pitch.to_degrees()));
                            ui.end_row();
                        });
                });

                ui.collapsing("Performance", |ui| {
                    egui::Grid::new("perf_grid")
                        .num_columns(2)
                        .spacing([20.0, 4.0])
                        .show(ui, |ui| {
                            ui.label("FPS:");
                            ui.label(format!("{:.0}", info.fps));
                            ui.end_row();

                            ui.label("Resolution:");
                            ui.label(format!("{}x{}", info.width, info.height));
                            ui.end_row();

                            ui.label("Triangles:");
                            ui.label(format!("{}", info.num_triangles));
                            ui.end_row();

                            ui.label("Time:");
                            ui.label(format!("{:.1} s", info.time));
                            ui.end_row();
                        });
                });

                ui.separator();
                ui.heading("Export");
                ui.separator();

                ui.horizontal(|ui| {
                    if ui.button("\u{1F4F7} Screenshot").clicked() {
                        self.screenshot_requested = true;
                    }

                    let record_text = if self.recording {
                        "\u{23F9} Stop Recording"
                    } else {
                        "\u{23FA} Record"
                    };
                    if ui.button(record_text).clicked() {
                        self.recording = !self.recording;
                    }
                });

                if self.recording {
                    ui.colored_label(
                        egui::Color32::RED,
                        format!("\u{1F534} Recording... Frame {}", info.frame_counter),
                    );
                }

                ui.small("F12: Screenshot | F11: Toggle Recording");

                ui.separator();
                ui.small("Drag to rotate | Scroll to zoom | ESC to exit");
            });
    }

    /// Clear one-shot action flags after they have been processed.
    pub fn clear_actions(&mut self) {
        self.reset_drainage_requested = false;
        self.reset_gpu_drainage_requested = false;
        self.add_bubble_requested = false;
        self.reset_foam_requested = false;
        self.regenerate_foam_requested = false;
    }
}
