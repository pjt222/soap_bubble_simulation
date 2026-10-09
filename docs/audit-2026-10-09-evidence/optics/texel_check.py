"""LUT addressing: texels generated at i/(N-1)*max, sampled with u = d/max under linear filtering
(texel i centre at (i+0.5)/N). Effective sampled thickness d' = (u*N - 0.5) * max/(N-1)."""
import math

N_T, MAX_D = 256, 2000.0
N_A = 64
N_FILM = 1.33

for d in (0.0, 30.0, 250.0, 500.0, 1000.0, 1500.0, 2000.0):
    u = d / MAX_D
    s = min(max(u * N_T - 0.5, 0.0), N_T - 1)
    d_eff = s * MAX_D / (N_T - 1)
    err = d_eff - d
    print(f"d={d:7.1f}: sampled d'={d_eff:8.2f}  err={err:+6.2f} nm  phase err @400nm={4*math.pi*N_FILM*err/400:+.3f} rad")

for c in (0.0, 0.25, 0.5, 0.9, 1.0):
    s = min(max(c * N_A - 0.5, 0.0), N_A - 1)
    print(f"cos={c:.2f}: sampled cos'={s/(N_A-1):.4f} err={s/(N_A-1)-c:+.4f}")

print(f"thickness texel pitch = {MAX_D/(N_T-1):.2f} nm; phase step per texel @400nm = {4*math.pi*N_FILM*MAX_D/(N_T-1)/400:.3f} rad")
