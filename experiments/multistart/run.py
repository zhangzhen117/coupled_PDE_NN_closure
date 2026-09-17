"""Shared 10k IC pool; frozen checkpoints; unchanged Newton with a 20-step cap."""
import sys,json,time,hashlib,signal
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import solve_banded
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
from chan.cases import case_setup
from chan import mesh as M,operator as OP
from closure.train import build
from data import lm_channel as LM
from s12_newton_krylov import newton_krylov,residual,operators,initial_guess
import os
OUT=Path(os.environ.get('MULTISTART_OUTPUT', ROOT/'data/multistart')).resolve()
OUT.mkdir(parents=True, exist_ok=True)
SELECTION=[('force',0),('force_pen2w',0),('tau',8),('nueff',0),('hybrid',9)]
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def prepare():
 c=case_setup(1000);y=c['m'].yc;rng=np.random.default_rng(20260913)
 profiles=[np.zeros_like(y),initial_guess('laminar',c)];labels=['zero','laminar']
 modes=np.arange(1,17);basis=np.sin(np.pi*np.outer(modes,y)/2)
 for i in range(9998):
  kind=i%4;coef=rng.normal(size=16)/modes**(2 if kind!=3 else 1)
  v=coef@basis;v/=max(abs(v).max(),1e-12)
  if kind==0:u=rng.uniform(0,60)*y*(2-y)+rng.uniform(0,10)*v
  elif kind==1:u=rng.uniform(-100,100)*v
  elif kind==2:u=rng.choice([-1,1])*10**rng.uniform(2,3)*v
  else:u=rng.uniform(-50,50)*y*(2-y)+rng.uniform(0,50)*v
  profiles.append(u);labels.append(['positive_base','signed_smooth','large_smooth','rougher'][kind])
 np.save(OUT/'initial_profiles.npy',profiles)
 write(OUT/'protocol.json',dict(re=1000,n=len(y),count=10000,rng_seed=20260913,labels=labels,selection=SELECTION,max_newton_steps=20,timeout_seconds=15,refinement_timeout_seconds=30,residual_tolerance=1e-8,correction_tolerance=1e-7,cluster_tolerance=1e-4,pool_sha256=hashlib.sha256((OUT/'initial_profiles.npy').read_bytes()).hexdigest()))
class Timeout(Exception):pass
def alarm(*args):raise Timeout('per-start wall-clock cap')
def main(task):
 if not 0 <= task < 50: raise ValueError('task must be between 0 and 49')
 family,seed=SELECTION[task//10];shard=task%10;dest=OUT/f'{family}_{shard:02}'
 if (dest/'records.jsonl').exists(): raise FileExistsError('Refusing to append duplicate trials; choose a fresh MULTISTART_OUTPUT directory.')
 dest.mkdir(exist_ok=True)
 torch.set_num_threads(1);c=case_setup(1000);m=c['m'];rt=LM.load(1000)['retau'];src=OP.source(m)
 norm=lambda u:float(M.wnorm(u,m.dV))
 distance=lambda a,b:norm(a-b)/max(1,norm(a),norm(b))
 rn=lambda u:norm(residual(family,model,m,u,rt,c['A'],src)/m.dV)/norm(src/m.dV)
 frozen=ROOT/'data/multistart/checkpoints'/f'{family}.pt'
 model=build(family,seed);model.load_state_dict(torch.load(frozen,weights_only=False)['state']);model.eval()
 pool=np.load(OUT/'initial_profiles.npy',mmap_mode='r');roots=[];records=[];started=time.time()
 signal.signal(signal.SIGALRM,alarm)
 for ic in range(shard*1000,(shard+1)*1000):
  row=dict(ic=ic,root=None);t=time.time()
  try:
   signal.setitimer(signal.ITIMER_REAL,15)
   ok,steps,counts,u=newton_krylov(family,model,c,rt,precond='A',itmax=20,u0=pool[ic])
   r=rn(u);row.update(solver_ok=bool(ok),steps=steps,residual=r,status='screen_failed')
   if np.isfinite(r) and r<1e-8:
    idx=next((j for j,v in enumerate(roots) if distance(u,v)<1e-4),None)
    if idx is None:
     signal.setitimer(signal.ITIMER_REAL,30)
     _,_,_,v=newton_krylov(family,model,c,rt,precond='A',u0=u)
     if rn(v)<r:u=v
     r=rn(u);J,_=operators(family,model,c,rt,u);ab=np.zeros((3,m.n));ab[0,1:]=np.diag(J,1);ab[1]=np.diag(J);ab[2,:-1]=np.diag(J,-1)
     du=solve_banded((1,1),ab,-residual(family,model,m,u,rt,c['A'],src));correction=norm(du)/max(1,norm(u))
     row.update(residual=r,relative_newton_correction=correction)
     if r<1e-8 and correction<1e-7:
      idx=next((j for j,v in enumerate(roots) if distance(u,v)<1e-4),None)
      if idx is None:idx=len(roots);roots.append(u.copy());np.save(dest/f'root{idx}.npy',u)
    if idx is not None:row.update(root=idx,status='verified_root')
  except Timeout:row['status']='timeout'
  except (RuntimeError,ValueError,np.linalg.LinAlgError) as e:row.update(status='error',exception=str(e))
  finally:signal.setitimer(signal.ITIMER_REAL,0)
  row['seconds']=time.time()-t;records.append(row)
  with (dest/'records.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if len(records)%50==0:
   summary=dict(family=family,seed=seed,shard=shard,attempted=len(records),converged=sum(v['root'] is not None for v in records),distinct_roots=len(roots),seconds=time.time()-started,complete=len(records)==1000,checkpoint_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest())
   write(dest/'status.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 else:main(int(sys.argv[1]))
