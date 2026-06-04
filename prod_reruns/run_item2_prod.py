#!/usr/bin/env python3
# Production Item2: laminar/dominance signature for BIB vs SIR vs prior-restart.
# BIB: directed renewal (ref). SIR: ESS-triggered resample of hypotheses ~ posterior + jitter.
# restart: reinit argmin likelihood from the prior (random) at every step.
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
sys.path.insert(0, ENG)
from rpsgame_reward import BayesReward, rps, HANDS
import powerlaw, warnings; warnings.filterwarnings("ignore")
THETA=0.4

def hist_of(b):
    h=np.zeros(b.d_num)
    for o in b.history[-b.h_length:]:
        h[int(np.where(b.d_type==o)[0][0])]+=1
    return h/max(len(b.history[-b.h_length:]),1)

def post_step(b, method, qr):
    if len(b.history) < b.h_length: return
    if method=='bib':
        h=hist_of(b); i=int(np.argmin(b.h_prov)); b.likelihood[i]=h
        if (b.likelihood[i]<1e-6).any(): b.likelihood[i]=b.likelihood[i]*0.95+0.05/b.d_num
    elif method=='restart':
        i=int(np.argmin(b.h_prov)); r=np.abs(qr.standard_normal(b.d_num)); b.likelihood[i]=r/r.sum()
    elif method=='sir':
        P=b.h_prov; ess=1.0/np.sum(P**2)
        if ess < b.h_num/2.0:
            idx=qr.choice(b.h_num, size=b.h_num, p=P)
            L=b.likelihood[idx]+0.02*qr.standard_normal(b.likelihood.shape)
            L=np.clip(L,1e-3,None); L/=L.sum(axis=1,keepdims=True)
            b.likelihood=L; b.h_prov=np.ones(b.h_num)/b.h_num

def step(b,my,opp,method,qr):
    res=rps(my,opp)
    if res=='quits': return
    kan=my if res=='win' else qr.choice([x for x in HANDS if x!=my])
    b.inference(kan); b.update_history(kan); post_step(b,method,qr)

def run(method,n_runs,T,burn,seed0):
    lam=[]
    for r in range(n_runs):
        b1=BayesReward(h_num=10,d_num=3,h_length=50,init_mode='random',predict_mode='sample',seed=seed0+2*r)
        b2=BayesReward(h_num=10,d_num=3,h_length=50,init_mode='random',predict_mode='sample',seed=seed0+2*r+1)
        qr1=np.random.default_rng(300000+seed0+2*r); qr2=np.random.default_rng(300000+seed0+2*r+1)
        c1=0;c2=0
        for t in range(T):
            h1=b1.expect(); h2=b2.expect(); step(b1,h1,h2,method,qr1); step(b2,h2,h1,method,qr2)
            if t>=burn:
                if b1.h_prov.max()>THETA: c1+=1
                else:
                    if c1>0: lam.append(c1); 
                    c1=0
                if b2.h_prov.max()>THETA: c2+=1
                else:
                    if c2>0: lam.append(c2)
                    c2=0
        if c1>0: lam.append(c1)
        if c2>0: lam.append(c2)
    lam=np.array([l for l in lam if l>=1])
    if len(lam)<50: return dict(method=method,n_phases=int(len(lam)),mean_len=float(lam.mean()) if len(lam) else 0.0,alpha_tpl=None,frac_high=None)
    fit=powerlaw.Fit(lam,discrete=True,verbose=False)
    return dict(method=method,n_phases=int(len(lam)),mean_len=float(lam.mean()),alpha_tpl=float(fit.truncated_power_law.alpha))

if __name__=="__main__":
    methods=sys.argv[1].split(",")
    n_runs=int(os.environ.get("NR","8")); T=int(os.environ.get("T","20000")); burn=int(os.environ.get("BURN","4000"))
    cache=os.path.join(os.path.dirname(os.path.abspath(__file__)), "item2_cache.json")
    data=json.load(open(cache)) if os.path.exists(cache) else {}
    for m in methods:
        t0=time.time(); res=run(m,n_runs,T,burn,2500); res['secs']=round(time.time()-t0,1); data[m]=res
        a=res['alpha_tpl']; print("%-8s mean_len=%.1f n_phases=%d alpha_TPL=%s (%.1fs)"%(m,res['mean_len'],res['n_phases'],("%.3f"%a) if a else "n/a",res['secs']))
        json.dump(data,open(cache,"w"),indent=2)
