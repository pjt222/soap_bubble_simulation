"""thickness_gradient_uv (wgsl:268-289) vs the true surface gradient, test field h = n.x."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bf_model import *

# Field stored like drainage grid: row j -> theta=j*pi/(H-1); column i -> u=i/(W-1) (as the sampler assumes)
jj, ii = np.meshgrid(np.arange(TH), np.arange(TW), indexing='ij')
U = ii / (TW - 1); V = jj / (TH - 1)
field = np.array([[uv_to_normal(U[j, i], V[j, i])[0] for i in range(TW)] for j in range(TH)])

worst = 0; rows = []
for (u, v) in [(0.30, 0.5), (0.375, 0.5), (0.4, 0.3), (0.6, 0.25), (0.45, 0.7), (0.3, 0.2)]:
    n = uv_to_normal(u, v)
    g_uv = thickness_gradient_uv(field, np.array([u, v]))
    _, g3 = uv_force_to_tangent_frame(g_uv, n, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
    true = np.array([1.0, 0, 0]) - n[0] * n          # surface gradient of h = x
    ang = np.degrees(np.arccos(np.clip(g3 @ true / np.linalg.norm(g3) / np.linalg.norm(true), -1, 1)))
    # analytic: code = (2*pi*g_phi, pi*g_theta)
    worst = max(worst, ang)
    print(f"uv=({u:.3f},{v:.2f}): |code|/|true| = {np.linalg.norm(g3)/np.linalg.norm(true):5.2f}  direction error {ang:5.1f} deg")
print("analytic max direction error for (2a,b) vs (a,b): %.1f deg" % (np.degrees(np.arctan(np.sqrt(2)) - np.arctan(1/np.sqrt(2)))))
