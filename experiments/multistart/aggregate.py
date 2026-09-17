"""Merge shard-local root identities and write an auditable empirical report."""
from run import *
def aggregate():
 c=case_setup(1000);m=c['m'];norm=lambda u:float(M.wnorm(u,m.dV));results=[]
 for family,seed in SELECTION:
  roots=[];counts=[];records=[];finished=0
  for shard in range(10):
   d=OUT/f'{family}_{shard:02}';p=d/'records.jsonl'
   if not p.exists():continue
   local={}
   for line in p.read_text().splitlines():
    try:row=json.loads(line)
    except json.JSONDecodeError:continue
    idx=row['root']
    if idx is not None:
     if idx not in local:
      u=np.load(d/f'root{idx}.npy');k=next((j for j,v in enumerate(roots) if norm(u-v)/max(1,norm(u),norm(v))<1e-4),None)
      if k is None:k=len(roots);roots.append(u);counts.append(0)
      local[idx]=k
     counts[local[idx]]+=1
     row['root']=local[idx]
    records.append(row)
   if (d/'status.json').exists():finished+=int(json.loads((d/'status.json').read_text())['complete'])
  dest=OUT/family;dest.mkdir(exist_ok=True)
  for j,u in enumerate(roots):np.save(dest/f'root{j}.npy',u)
  write(dest/'merged_records.json',records)
  pairs=[dict(roots=[i,j],relative_distance=norm(roots[i]-roots[j])/max(1,norm(roots[i]),norm(roots[j]))) for i in range(len(roots)) for j in range(i)]
  result=dict(family=family,seed=seed,attempted=len(records),converged=sum(counts),distinct_roots=len(roots),root_counts=counts,root_error_pct=[100*norm(u-c['U_dns'])/norm(c['U_dns']) for u in roots],pairs=pairs,timeouts=sum(r['status']=='timeout' for r in records),complete=finished==10)
  results.append(result)
 write(OUT/'summary.json',results)
 lines=['# Common-Reynolds-number multiple-root search','', 'Fixed Re_tau = 1000; one frozen checkpoint per closure; the identical 10,000-profile pool for all five closures. The original Newton–GMRES solver uses preconditioner A, unchanged line search and early exits, with screening capped at 20 Newton steps and 15 seconds. Novel candidate roots receive up to 60 refinement steps (30 seconds). Acceptance requires relative weighted residual < 1e-8 and relative Newton correction < 1e-7 for each representative; clustering tolerance is 1e-4 in relative weighted velocity distance.','', 'The pool contains zero and laminar profiles, then 9,998 profiles cycling through four distributions: positive parabolic bases with small smooth perturbations; signed smooth profiles with amplitudes up to 100; signed smooth profiles with log-uniform amplitudes 100–1000; and rougher profiles with signed parabolic bases. Smooth Fourier coefficients decay as mode^-2; rougher coefficients decay as mode^-1. All use 16 sine modes and homogeneous wall values. RNG seed: 20260913. See protocol.json and run.py for the precise distributions.','', '| Closure | Seed | IC attempted | Root hits | Distinct roots | Timed out | Complete |','|---|---:|---:|---:|---:|---:|---|']
 for r in results:lines.append(f"| {r['family']} | {r['seed']} | {r['attempted']} | {r['converged']} | {r['distinct_roots']} | {r['timeouts']} | {r['complete']} |")
 lines.extend(['','Root counts are empirical basin hit counts under this particular IC distribution and stopping rule. Distinct verified roots support discrete numerical multiplicity. Finding only one root cannot establish uniqueness, and failed or timed-out starts do not establish absence of a root. Checkpoints and the manuscript are unchanged.'])
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(results,indent=2))
if __name__=='__main__':aggregate()
