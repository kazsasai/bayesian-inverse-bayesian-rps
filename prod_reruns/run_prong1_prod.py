#!/usr/bin/env python3
# Production Prong1: laminar exponent across INVERSE-STEP FORM variants + controls,
# on the real reward_huge_v2 engine. Laminar phase = consecutive steps with max_h P(h)>theta.
import sys, os, json, time
import numpy as np
import os
from pathlib import Path
def _find_engine():
    env = os.environ.get("ENGINE_DIR")
    if env and (Path(env) / "rpsgame_reward.py").exists():
        return env
    here = Path(__file__).resolve()
    for base in [here.parent, *here.parents]:
        for sub in ("BIB_Levy_github/simulation/reward_huge_v2", "simulation/reward_huge_v2"):
            c = base / sub
            if (c / "rpsgame_reward.py").exists():
                return str(c)
    raise SystemExit("rpsgame_reward.py not found near this script; set ENGINE_DIR=/path/to/reward_huge_v2")
ENG = _find_engine()
sys.path.insert(0, ENG); sys.path.insert(0, os.path.dirname(__file__))
from rpsgame_reward import BayesReward, rps, HANDS
import powerlaw, warnings
warnings.filterwarnings("ignore")
THETA=0.4

def hist_of(b):
    h=np.zeros(b.d_num)
    for obs in b.history[-b.h_length:]:
        h[int(np.where(b.d_type==obs)[0][0])]+=1
    return h/max(len(b.history[-b.h_length:]),1)

def smooth(b,idx):
    if (b.likelihood[idx]<1e-6).any():
        b.likelihood[idx]=b.likelihood[idx]*0.95+0.05/b.d_num

def inverse_variant(b, qr, form):
    if len(b.history)<b.h_length: return
    if form in ('bo',): return
    h=hist_of(b); P=b.h_prov
    if form=='ref':        idx=[int(np.argmin(P))]; src=[h]
    elif form=='softmin':  w=np.exp(-(P/P.mean())); w/=w.sum(); idx=[int(qr.choice(b.h_num,p=w))]; src=[h]
    elif form=='bottom2':  idx=list(np.argsort(P)[:2]); src=[h,h]
    elif form=='blend':    i=int(np.argmin(P)); idx=[i]; src=[0.5*h+0.5*b.likelihood[i]]
    elif form=='cadence':  # ref but only with prob 0.5
        if qr.random()<0.5: idx=[int(np.argmin(P))]; src=[h]
        else: return
    elif form=='noise':    i=int(np.argmin(P)); r=qr.random(b.d_num); idx=[i]; src=[r/r.sum()]
    else: raise ValueError(form)
    for i,s in zip(idx,src):
        b.likelihood[i]=s; smooth(b,i)

def step(b,my,opp,form,qr):
    res=rps(my,opp)
    if res=='quits': return
    kan=my if res=='win' else qr.choice([x for x in HANDS if x!=my])
    b.inference(kan); b.update_history(kan)
    inverse_variant(b,qr,form)

def collect_laminar(P_is_high, lengths, run_state):
    # run_state: current laminar run length
    if P_is_high:
        run_state[0]+=1
    else:
        if run_state[0]>0: lengths.append(run_state[0])
        run_state[0]=0

def run_form(form, n_runs, T, burn, seed0):
    lengths=[]
    for r in range(n_runs):
        b1=BayesReward(h_num=10,d_num=3,h_length=50,init_mode='random',predict_mode='sample',seed=seed0+2*r)
        b2=BayesReward(h_num=10,d_num=3,h_length=50,init_mode='random',predict_mode='sample',seed=seed0+2*r+1)
        qr1=np.random.default_rng(500000+seed0+2*r); qr2=np.random.default_rng(500000+seed0+2*r+1)
        st1=[0]; st2=[0]
        for t in range(T):
            h1=b1.expect(); h2=b2.expect()
            step(b1,h1,h2,form,qr1); step(b2,h2,h1,form,qr2)
            if t>=burn:
                collect_laminar(b1.h_prov.max()>THETA, lengths, st1)
                collect_laminar(b2.h_prov.max()>THETA, lengths, st2)
        if st1[0]>0: lengths.append(st1[0])
        if st2[0]>0: lengths.append(st2[0])
    lengths=np.array([l for l in lengths if l>=1])
    fit=powerlaw.Fit(lengths, discrete=True, verbose=False)
    return dict(form=form, n_phases=int(len(lengths)), mean_len=float(lengths.mean()),
                alpha_tpl=float(fit.truncated_power_law.alpha),
                alpha_pl=float(fit.power_law.alpha), xmin=float(fit.xmin))

if __name__=="__main__":
    forms=sys.argv[1].split(",")
    n_runs=int(os.environ.get("NR","8")); T=int(os.environ.get("T","30000")); burn=int(os.environ.get("BURN","5000"))
    cache=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prong1_cache.json")
    data=json.load(open(cache)) if os.path.exists(cache) else {}
    for f in forms:
        t0=time.time(); res=run_form(f,n_runs,T,burn,seed0=2000)
        res['secs']=round(time.time()-t0,1); data[f]=res
        print("%-8s alpha_TPL=%.3f alpha_PL=%.3f xmin=%.0f  mean_len=%.1f n=%d (%.1fs)"%(
              f,res['alpha_tpl'],res['alpha_pl'],res['xmin'],res['mean_len'],res['n_phases'],res['secs']))
        json.dump(data,open(cache,"w"),indent=2)
