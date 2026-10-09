// Standalone replicas of per-frame CPU hot paths (no wgpu). Compiled with rustc -O, NOT cargo.
use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct ScattererGPU { pos_u: f32, pos_v: f32, strength: f32, inv_sigma_sq: f32 }
const GRID_SIZE_U: u32 = 10; const GRID_SIZE_V: u32 = 10; const GRID_CELL_SIZE: f32 = 0.1;

fn halton(index: u32, base: u32) -> f32 { let mut r = 0.0f32; let mut f = 1.0 / base as f32; let mut i = index;
    while i > 0 { r += f * (i % base) as f32; i /= base; f /= base as f32; } r }
fn gen(num: u32) -> Vec<ScattererGPU> { (0..num).map(|i| { let u = halton(i+1,2); let v = halton(i+1,3);
    ScattererGPU{pos_u: 0.342 + u*0.316, pos_v: 0.342 + v*0.316, strength: 0.5, inv_sigma_sq: 555.0} }).collect() }

// Replica of branched_flow.rs:608-670 (Brownian branch + sort + prefix sum); returns (sorted, offsets)
fn update_scatterers(cur: &mut Vec<ScattererGPU>, time: f32) -> Vec<u32> {
    let (min_u, max_u, min_v, max_v) = (0.342f32, 0.658f32, 0.342f32, 0.658f32);
    for (i, s) in cur.iter_mut().enumerate() {
        let seed_u = (i as f32 * 0.7531 + time * 31.37).sin() * 43758.547;
        let seed_v = (i as f32 * 0.9371 + time * 17.53).cos() * 43758.547;
        s.pos_u = (s.pos_u + (seed_u.fract() - 0.5) * 0.001).clamp(min_u, max_u);
        s.pos_v = (s.pos_v + (seed_v.fract() - 0.5) * 0.001).clamp(min_v, max_v);
    }
    cur.sort_by_key(|s| { let uc = (s.pos_u / GRID_CELL_SIZE).clamp(0.0, (GRID_SIZE_U-1) as f32) as u32;
        let vc = (s.pos_v / GRID_CELL_SIZE).clamp(0.0, (GRID_SIZE_V-1) as f32) as u32; vc * GRID_SIZE_U + uc });
    let total = (GRID_SIZE_U*GRID_SIZE_V) as usize; let mut off = vec![0u32; total+1];
    for s in cur.iter() { let uc = (s.pos_u / GRID_CELL_SIZE).clamp(0.0, 9.0) as u32; let vc = (s.pos_v / GRID_CELL_SIZE).clamp(0.0, 9.0) as u32;
        off[(vc*GRID_SIZE_U+uc) as usize + 1] += 1; }
    for i in 1..=total { off[i] += off[i-1]; }
    off
}

// Replica of DrainageSimulator::step core loop (drainage.rs:291-394, without TFE) + per-frame scans (pipeline.rs:1300-1318)
struct Field { nt: usize, np: usize, d: Vec<f64>, dth: f64, dph: f64 }
fn drain_step(h: &mut Field, scratch: &mut Vec<f64>, dt: f64) {
    let (nt, np, dth, dph) = (h.nt, h.np, h.dth, h.dph);
    let kd = 1000.0*9.81/(3.0*0.001); let dc = 1e-9; let r2 = 0.025f64*0.025; let hc = 10e-9;
    scratch.copy_from_slice(&h.d);
    for ti in 1..(nt-1) { let th = ti as f64*dth; let st = th.sin(); let ct = th.cos();
        let sts = if st.abs() < 1e-10 { 1e-10f64.copysign(st) } else { st };
        for pi in 0..np { let hcur = scratch[ti*np+pi]; if hcur < hc { continue; }
            let htm = scratch[(ti-1)*np+pi]; let htp = scratch[(ti+1)*np+pi];
            let pm = (pi + np - 1) % np; let pp = (pi + 1) % np;
            let hpm = scratch[ti*np+pm]; let hpp = scratch[ti*np+pp];
            let drain = -kd*hcur*hcur*hcur*st;
            let d2t = (htp - 2.0*hcur + htm)/(dth*dth); let d1t = (htp-htm)/(2.0*dth); let d2p = (hpp - 2.0*hcur + hpm)/(dph*dph);
            let lap = (d2t + ct/sts*d1t + d2p/(sts*sts))/r2;
            h.d[ti*np+pi] = (hcur + dt*(drain + dc*lap)).max(0.0);
        } }
    // poles
    let top: f64 = (0..np).map(|p| h.d[np+p]).sum::<f64>()/np as f64; for p in 0..np { h.d[p] = top; }
    let l = nt-1; let bot: f64 = (0..np).map(|p| h.d[(l-1)*np+p]).sum::<f64>()/np as f64; for p in 0..np { h.d[l*np+p] = bot; }
}
fn scans(h: &Field) -> (f64, f64, bool) {
    let mn = h.d.iter().copied().fold(f64::INFINITY, f64::min);
    let mx = h.d.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let crit = h.d.iter().copied().fold(f64::INFINITY, f64::min) < 10e-9;   // has_critical_region rescans
    (mn, mx, crit)
}

fn time_min<F: FnMut()>(label: &str, reps: usize, inner: usize, mut f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps { let t = Instant::now(); for _ in 0..inner { f(); } let us = t.elapsed().as_secs_f64()*1e6/inner as f64; if us < best { best = us; } }
    println!("{label:<58} min {best:9.2} us/call"); best
}


fn stability(fps: f64, seed_row: usize) {
    let (nt, np) = (128usize, 256usize);
    let mut f = Field{ nt, np, d: vec![500e-9; nt*np], dth: std::f64::consts::PI/(nt-1) as f64, dph: 2.0*std::f64::consts::PI/np as f64 };
    let mut scratch = vec![0.0; nt*np];
    f.d[seed_row*np + 17] *= 1.0 + 1e-12;          // 1e-12 relative perturbation in one cell
    let dt = 100.0 / fps;                          // pipeline.rs:1296 with drainage_time_scale=100
    let mut out = String::new();
    for k in 0..60 {
        drain_step(&mut f, &mut scratch, dt);
        let mut dev = 0.0f64; let mut row = 0;
        for t in 0..nt { let mean: f64 = f.d[t*np..(t+1)*np].iter().sum::<f64>()/np as f64;
            for p in 0..np { let d = (f.d[t*np+p]-mean).abs(); if d > dev { dev = d; row = t; } } }
        if k % 6 == 5 || k == 0 { out += &format!(" s{}: dev={:.1e}(r{})", k+1, dev, row); }
    }
    println!("fps={fps:>5} dt={dt:.3}s seed_row={seed_row}:{out}");
}
fn main() {
    if std::env::args().any(|a| a == "stab") {
        for &(fps, row) in &[(60.0, 1usize), (60.0, 5), (144.0, 1), (1000.0, 1)] { stability(fps, row); }
        return;
    }
    for &n in &[400u32, 800, 2048] {
        let mut s = gen(n); let mut t = 0.0f32;
        time_min(&format!("update_scatterers CPU part, n={n}"), 30, 200, || { t += 0.016; black_box(update_scatterers(&mut s, t)); });
    }
    for &res in &[128usize, 256] {
        let (nt, np) = (res, 2*res);
        let mut f = Field{ nt, np, d: vec![500e-9; nt*np], dth: std::f64::consts::PI/(nt-1) as f64, dph: 2.0*std::f64::consts::PI/np as f64 };
        let mut scratch = vec![0.0; nt*np];
        time_min(&format!("CPU drainage step {nt}x{np} (no TFE), dt=0.05"), 20, 20, || { drain_step(&mut f, &mut scratch, 0.05); });
        time_min(&format!("min/max/has_critical scans {nt}x{np}"), 20, 50, || { black_box(scans(&f)); });
    }
}
