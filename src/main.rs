//! Soap Bubble Simulation
//!
//! Physically accurate 3D simulation of soap bubbles with thin-film interference colors.

use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

use soap_bubble_sim::config::SimulationConfig;
use soap_bubble_sim::render::RenderPipeline;

/// Soap bubble simulation with thin-film interference
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to configuration file
    #[arg(short, long)]
    config: Option<String>,

    /// Override bubble diameter (meters)
    #[arg(long)]
    diameter: Option<f64>,

    /// Override film thickness (nanometers)
    #[arg(long)]
    thickness: Option<f64>,
}

/// Application state
struct App {
    window: Option<Arc<Window>>,
    pipeline: Option<RenderPipeline>,
    config: SimulationConfig,
    last_frame: Instant,
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
}

impl App {
    fn new(config: SimulationConfig) -> Self {
        Self {
            window: None,
            pipeline: None,
            config,
            last_frame: Instant::now(),
            mouse_pressed: false,
            last_mouse_pos: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        // Create window (start maximized + no decorations to avoid resize artifacts on WSLg)
        let window_attributes = Window::default_attributes()
            .with_title("Soap Bubble Simulation")
            .with_inner_size(LogicalSize::new(1280, 720))
            .with_maximized(true)
            .with_decorations(false);

        let window = Arc::new(
            event_loop
                .create_window(window_attributes)
                .expect("Failed to create window"),
        );

        self.window = Some(window.clone());

        // Create render pipeline
        let pipeline = pollster::block_on(RenderPipeline::new(window));
        self.pipeline = Some(pipeline);
        self.last_frame = Instant::now();

        // Apply config to pipeline
        if let Some(ref mut pipeline) = self.pipeline {
            pipeline.bubble_uniform.base_thickness_nm = self.config.bubble.film_thickness_nm as f32;
            pipeline.bubble_uniform.refractive_index = self.config.bubble.refractive_index as f32;
        }

        log::info!("Window created, rendering started");
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // First, let egui handle the event
        let egui_consumed = if let (Some(pipeline), Some(window)) = (&mut self.pipeline, &self.window) {
            pipeline.handle_event(window, &event)
        } else {
            false
        };

        // If egui consumed the event, don't process it further (except for essential events)
        match event {
            WindowEvent::CloseRequested => {
                log::info!("Close requested, exiting");
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                if let Some(ref mut pipeline) = self.pipeline {
                    pipeline.resize(new_size);
                }
            }
            WindowEvent::MouseInput { state, button, .. } if !egui_consumed => {
                if button == MouseButton::Left {
                    self.mouse_pressed = state == ElementState::Pressed;
                }
            }
            WindowEvent::CursorMoved { position, .. } if !egui_consumed => {
                if self.mouse_pressed
                    && let Some((last_x, last_y)) = self.last_mouse_pos
                {
                    let delta_x = position.x - last_x;
                    let delta_y = position.y - last_y;
                    if let Some(ref mut pipeline) = self.pipeline {
                        pipeline.camera.orbit(delta_x as f32, delta_y as f32);
                    }
                }
                self.last_mouse_pos = Some((position.x, position.y));
            }
            WindowEvent::CursorMoved { position, .. } => {
                // Always track mouse position for smooth transitions
                self.last_mouse_pos = Some((position.x, position.y));
            }
            WindowEvent::MouseWheel { delta, .. } if !egui_consumed => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                };
                if let Some(ref mut pipeline) = self.pipeline {
                    pipeline.camera.zoom(scroll);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    use winit::keyboard::{Key, NamedKey};
                    match event.logical_key {
                        Key::Named(NamedKey::Escape) => {
                            event_loop.exit();
                        }
                        Key::Named(NamedKey::F12) => {
                            // Take screenshot
                            if let Some(ref mut pipeline) = self.pipeline {
                                pipeline.screenshot_requested = true;
                            }
                        }
                        Key::Named(NamedKey::F11) => {
                            // Toggle recording
                            if let Some(ref mut pipeline) = self.pipeline {
                                pipeline.toggle_recording();
                            }
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                // Calculate delta time
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f32();
                self.last_frame = now;

                if let (Some(pipeline), Some(window)) = (&mut self.pipeline, &self.window) {
                    // Update animation
                    pipeline.update(dt);

                    // Render
                    match pipeline.render(window) {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Lost) => {
                            let size = pipeline.size();
                            pipeline.resize(winit::dpi::PhysicalSize::new(size.0, size.1));
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            log::error!("Out of memory!");
                            event_loop.exit();
                        }
                        Err(e) => log::warn!("Render error: {:?}", e),
                    }
                }

                // Request next frame
                if let Some(ref window) = self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // put id:'cli_parse_args', label:'Parse CLI arguments', output:'cli_args.internal'
    let args = Args::parse();

    // put id:'cfg_load', label:'Load config JSON', input:'cli_args.internal', output:'config.json'
    let mut config = if let Some(ref path) = args.config {
        match SimulationConfig::from_file(path) {
            Ok(cfg) => {
                log::info!("Loaded config from {}", path);
                cfg
            }
            Err(e) => {
                log::warn!("Failed to load config: {}, using defaults", e);
                SimulationConfig::default()
            }
        }
    } else {
        SimulationConfig::default()
    };

    // put id:'cfg_merge_cli', label:'Merge CLI overrides', input:'config.json', output:'final_config.internal'
    if let Some(diameter) = args.diameter {
        config.bubble.diameter = diameter;
    }
    if let Some(thickness) = args.thickness {
        config.bubble.film_thickness_nm = thickness;
    }

    log::info!(
        "Starting simulation: {}cm bubble, {}nm film",
        config.bubble.diameter * 100.0,
        config.bubble.film_thickness_nm
    );

    // put id:'cli_event_loop', label:'Run event loop', input:'final_config.internal', output:'loop_iteration.internal'
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(config);
    event_loop.run_app(&mut app).expect("Event loop failed");
}
