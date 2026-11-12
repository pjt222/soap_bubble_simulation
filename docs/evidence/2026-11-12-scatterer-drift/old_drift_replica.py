"""Replica of the pre-fix Brownian perturbation (branched_flow.rs update_scatterers) with Rust's
sign-preserving f32::fract, for the default patch (centre 0.75, 0.5, half 0.158), 400 scatterers
spread uniformly over the patch. Where are they after 15 s at 25-36 fps?"""
import numpy as np
def rust_fract(x): return x - np.trunc(x)
rng = np.random.default_rng(1)
center_u, center_v, half = 0.75, 0.5, 0.158
min_u, max_u, min_v, max_v = center_u-half, center_u+half, center_v-half, center_v+half
for fps in (25, 30, 36):
    u = rng.uniform(min_u, max_u, 400).astype(np.float32); v = rng.uniform(min_v, max_v, 400).astype(np.float32)
    i = np.arange(400, dtype=np.float32)
    frames = 15*fps
    for frame in range(frames):
        t = np.float32(frame/fps)   # film_time advancing (any advancing sequence gives the bias)
        seed_u = np.sin(i*np.float32(0.7531) + t*np.float32(31.37))*np.float32(43758.547)
        seed_v = np.cos(i*np.float32(0.9371) + t*np.float32(17.53))*np.float32(43758.547)
        u = np.clip(u + (rust_fract(seed_u)-np.float32(0.5))*np.float32(0.001), min_u, max_u)
        v = np.clip(v + (rust_fract(seed_v)-np.float32(0.5))*np.float32(0.001), min_v, max_v)
    lower_half = np.mean(v > center_v); right_half = np.mean(u > center_u)
    lower_third = np.mean(v > min_v + 2*(max_v-min_v)/3)
    print(f"{fps} fps, {frames} frames: share in lower half {lower_half:.2f}, lower third {lower_third:.2f}, right half {right_half:.2f}; v range {v.min():.3f}..{v.max():.3f} (patch {min_v:.3f}..{max_v:.3f})")
