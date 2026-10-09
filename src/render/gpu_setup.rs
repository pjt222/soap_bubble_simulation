//! wgpu instance and adapter selection shared by the windowed and headless renderers.

/// Create the wgpu instance.
///
/// Starts from wgpu's defaults (all backends, build-dependent debug and
/// validation flags) and applies wgpu's instance environment overrides:
/// `WGPU_BACKEND`, `WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER`,
/// `WGPU_VALIDATION`, `WGPU_DEBUG`, `WGPU_GPU_BASED_VALIDATION`,
/// `WGPU_DISCARD_HAL_LABELS`, `WGPU_GLES_MINOR_VERSION`, `WGPU_DX12_COMPILER`.
/// `WGPU_ADAPTER_NAME` is not honoured (it is read only by
/// `wgpu::util::initialize_adapter_from_env`). wgpu 29 removes
/// `InstanceDescriptor::from_env_or_default`, so this needs rework on upgrade.
///
/// On WSL, `WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER=1` exposes the Mesa
/// Dozen (Vulkan on D3D12) adapters, which wgpu otherwise hides because Dozen
/// is not a conformant Vulkan implementation (#38).
pub fn create_instance() -> wgpu::Instance {
    wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default())
}

/// Adapter power preference: `WGPU_POWER_PREF` (`low`, `high` or `none`) if
/// set, otherwise high performance, so a discrete GPU wins over an integrated
/// one or a CPU rasteriser.
pub fn power_preference() -> wgpu::PowerPreference {
    wgpu::PowerPreference::from_env().unwrap_or(wgpu::PowerPreference::HighPerformance)
}

/// Log the chosen adapter. `scripts/gpu/probe-adapters.sh --app` and bug
/// reports rely on this line.
pub fn log_adapter(adapter: &wgpu::Adapter) {
    let info = adapter.get_info();
    log::info!(
        "GPU adapter: {} ({:?}, {:?} backend, driver: {} {})",
        info.name,
        info.device_type,
        info.backend,
        info.driver,
        info.driver_info
    );
}
