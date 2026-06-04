#!/usr/bin/env python3
# Production Prong2: within-phase log-posterior drift <eta> vs renewal prob q,
# on the REAL reward_huge_v2 engine (windowed histogram + conditional smoothing).
# q=0 -> Bayes-only, q=1 -> full BIB. Drift = mean per-step change of
# x = log P(top) - log P(2nd) while the argmax (top) is unchanged. Pooled over both agents.
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
EPS=1e-12

def step(b, my, opp, q, qr):
    res=rps(my,opp)
    if res=='quits': return
    kan = my if res=='win' else qr.choice([h for h in HANDS if h!=my])
    b.inference(kan); b.update_history(kan)
    if qr.random() < q:
        b.inverse()

def drift_accum(P, state):
    order=np.argsort(P)[::-1]; top=int(order[0]); sec=int(order[1])
    x=float(np.log(P[top]+EPS)-np.log(P[sec]+EPS))
    if state['top']==top and state['x'] is not None:
        state['acc']+=(x-state['x']); state['cnt']+=1
    state['x']=x; state['top']=top

def run_q(q, n_runs, T, burn, seed0):
    per_run=[]
    for r in range(n_runs):
        b1=BayesReward(h_num=10,d_num=3,h_length=50,init_mode='random',predict_mode='sample',seed=seed0+2*r)
        b2=BayesReward(h_num=10,d_num=3,h_length=50,init_mode='random',predict_mode='sample',seed=seed0+2*r+1)
        qr1=np.random.default_rng(900000+seed0+2*r); qr2=np.random.default_rng(900000+seed0+2*r+1)
        s1={'x':None,'top':-1,'acc':0.0,'cnt':0}; s2={'x':None,'top':-1,'acc':0.0,'cnt':0}
        for t in range(T):
            h1=b1.expect(); h2=b2.expect()
            step(b1,h1,h2,q,qr1); step(b2,h2,h1,q,qr2)
            if t>=burn:
                drift_accum(b1.h_prov,s1); drift_accum(b2.h_prov,s2)
        if s1['cnt']>0: per_run.append(s1['acc']/s1['cnt'])
        if s2['cnt']>0: per_run.append(s2['acc']/s2['cnt'])
    per_run=np.array(per_run)
    return dict(q=q, n=len(per_run), mean=float(per_run.mean()),
                sem=float(per_run.std()/max(len(per_run),1)**0.5))

if __name__=="__main__":
    qs=[float(x) for x in sys.argv[1].split(",")]
    n_runs=int(os.environ.get("NR","16")); T=int(os.environ.get("T","12000")); burn=int(os.environ.get("BURN","3000"))
    cache=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prong2_cache.json")
    data=json.load(open(cache)) if os.path.exists(cache) else {}
    for q in qs:
        t0=time.time(); res=run_q(q,n_runs,T,burn,seed0=1000+int(q*1000))
        res['secs']=round(time.time()-t0,1); data["%.3f"%q]=res
        print("q=%.3f  <eta>=%+.5f  sem=%.5f  n=%d  (%.1fs)"%(q,res['mean'],res['sem'],res['n'],res['secs']))
        json.dump(data,open(cache,"w"),indent=2)
