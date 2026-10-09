import sys
sys.argv=[sys.argv[0],"1","64"]
src=open("ray_replica.py").read()
# instrument: collect per-lane k_checked and in-cutoff counts each step
src=src.replace("stats['steps']+=a.sum();","KC.append(inN.sum(1));KE.append(inC.sum(1));GM.append(gm.copy());stats['steps']+=a.sum();")
src="KC=[];KE=[];GM=[]\n"+src
exec(compile(src,"ray_replica","exec"))
import numpy as np
KC=np.array(KC);KE=np.array(KE);GM=np.array(GM)   # [steps, N]
def warp_eff(order,K):
    Ko=K[:,order].reshape(K.shape[0],-1,32)
    return K.mean()/Ko.max(2).mean()
hash_order=np.arange(N)
strat_order=np.argsort(r1)  # emulate rand1=(idx+0.5)/N stratification: consecutive lanes adjacent across the beam
print(f"inner-loop SIMD efficiency (mean k / mean warp-max k), warp=32:")
print(f"  hash-ordered lanes (current):      {warp_eff(hash_order,KC):.3f}")
print(f"  stratified lanes (rand1 sorted):   {warp_eff(strat_order,KC):.3f}")
print(f"cutoff-branch (exp) active fraction per evaluated scatterer: {KE.sum()/KC.sum():.3f}")
print("|F| percentiles (tangent frame, incl particle_weight): p5 %.3f p25 %.3f p50 %.3f p75 %.3f p95 %.3f"%tuple(np.percentile(GM,[5,25,50,75,95])))
print("turn angle per step = bend*|F|*0.3*dt: median %.4f rad"%(np.median(GM)*5*0.3*0.005))
