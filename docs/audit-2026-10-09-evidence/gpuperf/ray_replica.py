import sys
import numpy as np
# Faithful numpy replica of branched_flow_compute.wgsl main() (lines 405-553) with defaults
# from branched_flow.rs:105-145 (uncommitted: 8192 rays, 200 steps, 400 scatterers).
# GRIN force = 0 because the drainage field is exactly uniform (see f32_drainage.py).
f32=np.float32
PI=3.14159265359
P=dict(entry=(0,0,1.0),beam=(-0.5,-0.866,0),num_rays=8192,ray_steps=200,step=0.005,bend=5.0,
       spread=0.4,falloff=0.001,tw=512,th=256,nscat=400,sstr=0.5,srad=0.03,pw=0.1,
       patch=int(sys.argv[1]) if len(sys.argv)>1 else 1,pcu=0.5,pcv=0.5,phs=0.158,deposit_scale=float(sys.argv[2]) if len(sys.argv)>2 else 64.0)
def fract(x): return x-np.floor(x)
def hash21(px,py):
    p3=fract(np.stack([px,py,px],-1).astype(f32)*f32(0.1031))
    d=(p3*(p3[...,[1,2,0]]+f32(33.33))).sum(-1,keepdims=True)
    p3=p3+d
    return fract((p3[...,0]+p3[...,1])*p3[...,2])
def halton(i,b):
    r=0.0;fct=1.0/b
    while i>0: r+=fct*(i%b); i//=b; fct/=b
    return r
def gen_scatterers(n,time,bs,br,patch):
    out=[]
    for i in range(n):
        u=halton(i+1,2);v=halton(i+1,3)
        hs=f32(i)*f32(0.1031)+f32(time*0.1)
        ju=fract(np.sin(hs)*f32(43758.547))*0.02; jv=fract(np.cos(hs)*f32(43758.547))*0.02
        if patch:
            mnu=max(P['pcu']-P['phs'],0);mnv=max(P['pcv']-P['phs'],0);mxu=min(P['pcu']+P['phs'],1);mxv=min(P['pcv']+P['phs'],1)
            mu=mnu+u*(mxu-mnu);mv=mnv+v*(mxv-mnv)
            pu=np.clip(mu+ju*P['phs']*2,0,1);pv=np.clip(mv+jv*P['phs']*2,0,1)
        else:
            pu=(u+ju)%1.0;pv=(v+jv)%1.0
        sh=fract(np.sin(f32(i)*f32(0.7531)+f32(0.3))*f32(43758.547)); sign=1.0 if sh>0.5 else -1.0
        sv=0.8+0.4*fract(np.sin(f32(i)*f32(0.9371))*f32(43758.547))
        rv=0.8+0.4*fract(np.cos(f32(i)*f32(0.5791))*f32(43758.547))
        sig=max(br*rv,1e-6)
        out.append((pu,pv,sign*bs*sv,1.0/(2*sig*sig)))
    return np.array(out,dtype=np.float64)
S=gen_scatterers(P['nscat'],0.0,P['sstr'],P['srad'],P['patch'])
cell=lambda u,v:(np.clip(np.floor(np.clip(u/0.1,0,9)),0,9).astype(int),np.clip(np.floor(np.clip(v/0.1,0,9)),0,9).astype(int))
scu,scv=cell(S[:,0],S[:,1])
def nrm(v): return v/np.linalg.norm(v,axis=-1,keepdims=True)
entry=nrm(np.array(P['entry']));br=np.array(P['beam']);beam=nrm(br-entry*br.dot(entry))
up=np.array([0,1.0,0]);
if abs(entry.dot(up))>0.99: up=np.array([1.0,0,0])
t1=nrm(np.cross(entry,up));t2=nrm(np.cross(entry,t1))
N=P['num_rays'];idx=np.arange(N,dtype=f32)
r1=hash21(idx*f32(0.1),np.zeros(N,f32)).astype(float);r2=hash21(idx*f32(0.1)+f32(100),np.ones(N,f32)).astype(float)
vel=nrm(np.array([beam.dot(t1),beam.dot(t2)]))[None,:].repeat(N,0)
perp=np.stack([-vel[:,1],vel[:,0]],1)
pos=perp*((r1-0.5)*P['spread'])[:,None]+vel*((r2-0.5)*P['spread']*0.1)[:,None]
if P['patch']:
    pphi=(P['pcu']*2-1)*PI;pth=P['pcv']*PI
    pc=nrm(np.array([np.sin(pth)*np.cos(pphi),np.cos(pth),np.sin(pth)*np.sin(pphi)]))
    tp=pc-entry;po=np.array([tp.dot(t1),tp.dot(t2)]);ps=P['phs']*PI
    pos=po[None,:]+perp*((r1-0.5)*ps)[:,None]+vel*((r2-0.5)*ps*0.5)[:,None]
    print("patch_center_3d",pc,"patch_offset(tangent)",po,"-> spawn centre maps to",nrm(entry+t1*po[0]+t2*po[1]))
inten=np.ones(N);alive=np.ones(N,bool)
tex=np.zeros(P['tw']*P['th'],np.int64)
stats=dict(steps=0,inpatch=0,k_checked=0,k_cut=0,dep_ideal=0.0,dep_actual=0.0,sf=[],zero_corner=0,corners=0)
for step in range(P['ray_steps']):
    a=alive
    if not a.any(): break
    p3=nrm(entry[None,:]+t1[None,:]*pos[:,0:1]+t2[None,:]*pos[:,1:2])
    phi=np.arctan2(p3[:,2],p3[:,0]);th=np.arccos(np.clip(p3[:,1],-1,1))
    u=(phi+PI)/(2*PI);v=th/PI
    rcu,rcv=cell(u,v)
    mincu=np.where(rcu>=1,rcu-1,0);maxcu=np.minimum(rcu+1,9);mincv=np.where(rcv>=1,rcv-1,0);maxcv=np.minimum(rcv+1,9)
    inN=(scu[None,:]>=mincu[:,None])&(scu[None,:]<=maxcu[:,None])&(scv[None,:]>=mincv[:,None])&(scv[None,:]<=maxcv[:,None])
    du=u[:,None]-S[None,:,0];dv=v[:,None]-S[None,:,1];r2_=du*du+dv*dv
    inC=inN&(r2_<=4.5/S[None,:,3])
    w=np.where(inC,S[None,:,2]*S[None,:,3]*2.0*np.exp(-r2_*S[None,:,3]),0.0)
    Fu=(du*w).sum(1)*P['pw'];Fv=(dv*w).sum(1)*P['pw']
    xz=np.sqrt(p3[:,0]**2+p3[:,2]**2)
    ph=np.where(xz[:,None]>0.001,np.stack([-p3[:,2]/np.maximum(xz,1e-9),np.zeros(N),p3[:,0]/np.maximum(xz,1e-9)],1),np.array([1.0,0,0]))
    thh=np.where(xz[:,None]>0.001,np.cross(p3,ph),np.array([0,0,1.0]))
    F3=ph*Fu[:,None]+thh*Fv[:,None];F=np.stack([F3@t1,F3@t2],1)
    gm=np.linalg.norm(F,axis=1);sf=np.clip(1.0/np.maximum(gm*10,0.333),0.3,3.0);adt=P['step']*sf
    if P['patch']:
        ip=(np.abs(u-P['pcu'])<=P['phs'])&(np.abs(v-P['pcv'])<=P['phs'])
        mnu=max(P['pcu']-P['phs'],0);mxu=min(P['pcu']+P['phs'],1);mnv=max(P['pcv']-P['phs'],0);mxv=min(P['pcv']+P['phs'],1)
        lu=np.clip((u-mnu)/max(mxu-mnu,0.001),0,1);lv=np.clip((v-mnv)/max(mxv-mnv,0.001),0,1)
    else:
        ip=np.ones(N,bool);lu=u;lv=v
    dep=a&ip
    stats['steps']+=a.sum();stats['inpatch']+=dep.sum();stats['k_checked']+=inN[a].sum();stats['k_cut']+=inC[a].sum();stats['sf'].append(sf[a])
    if dep.any():
        fx=np.clip(lu[dep],0,1)*(P['tw']-1);fy=np.clip(lv[dep],0,1)*(P['th']-1)
        x0=np.floor(fx).astype(int);y0=np.floor(fy).astype(int);x1=np.minimum(x0+1,P['tw']-1);y1=np.minimum(y0+1,P['th']-1)
        sx=fx-np.floor(fx);sy=fy-np.floor(fy)
        base=(inten[dep]*0.15*sf[dep]*P['deposit_scale']).astype(f32)
        ws=[(1-sx)*(1-sy),sx*(1-sy),(1-sx)*sy,sx*sy];ids=[y0*P['tw']+x0,y0*P['tw']+x1,y1*P['tw']+x0,y1*P['tw']+x1]
        for wi,ii in zip(ws,ids):
            val=(base*wi.astype(f32)).astype(f32);q=np.floor(val).astype(np.int64)
            np.add.at(tex,ii,q);stats['dep_ideal']+=float(val.sum());stats['dep_actual']+=float(q.sum())
            stats['zero_corner']+=int((q==0).sum());stats['corners']+=q.size
    vel=vel+F*P['bend']*adt[:,None];vm=np.linalg.norm(vel,axis=1);vel=np.where(vm[:,None]>0.001,vel/np.maximum(vm,1e-9)[:,None],vel)
    pos=np.where(a[:,None],pos+vel*adt[:,None],pos);inten=np.where(a,inten*(1-P['falloff']),inten)
    alive=a&~((np.linalg.norm(pos,axis=1)>2.5)|(inten<0.01))
sfall=np.concatenate(stats['sf'])
print(f"mode={'patch' if P['patch'] else 'full'} deposit_scale={P['deposit_scale']}")
print(f"ray-steps executed: {stats['steps']} of {N*P['ray_steps']} ({stats['steps']/(N*P['ray_steps']):.3f}); steps that deposit (in patch): {stats['inpatch']/stats['steps']:.3f}")
print(f"scatterers examined per step (3x3 cells): mean {stats['k_checked']/stats['steps']:.1f}; within 3-sigma cutoff: {stats['k_cut']/stats['steps']:.1f}")
print(f"step_factor: mean {sfall.mean():.2f}; frac at 0.3 clamp {np.mean(sfall<=0.3001):.3f}; frac at 3.0 clamp {np.mean(sfall>=2.999):.3f}; median {np.median(sfall):.2f}")
if stats['dep_ideal']>0:
    print(f"fixed-point truncation: deposited {stats['dep_actual']/stats['dep_ideal']:.3f} of ideal; corners truncated to 0: {stats['zero_corner']/stats['corners']:.3f}")
cov=(tex>0).reshape(P['th'],P['tw'])
print(f"texels with any deposit (single frame): {cov.mean():.3f}; left half (local u<0.5): {cov[:,:256].mean():.3f}; right half: {cov[:,256:].mean():.3f}")
print(f"max texel value one frame: {tex.max()}  -> steady state x{1/0.15:.2f} = {tex.max()/0.15:.3e} (u32 max 4.29e9)")
