"""Independently validate merged representatives and plot numerical evidence."""
from run import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 summary=json.loads((OUT/'summary.json').read_text());c=case_setup(1000);m=c['m'];src=OP.source(m);rt=LM.load(1000)['retau'];norm=lambda u:float(M.wnorm(u,m.dV))
 protocol=json.loads((OUT/'protocol.json').read_text())
 assert hashlib.sha256((OUT/'initial_profiles.npy').read_bytes()).hexdigest()==protocol['pool_sha256']
 labels=protocol['labels'];all_results=[]
 fig,axes=plt.subplots(2,5,figsize=(17,7),constrained_layout=True)
 for ax,curve,row in zip(axes[0],axes[1],summary):
  family=row['family'];seed=row['seed'];model=build(family,seed)
  frozen=ROOT/'data/multistart/checkpoints'/f'{family}.pt';model.load_state_dict(torch.load(frozen,weights_only=False)['state']);model.eval()
  roots=[np.load(OUT/family/f'root{i}.npy') for i in range(row['distinct_roots'])];verified=[]
  for i,u in enumerate(roots):
   r=residual(family,model,m,u,rt,c['A'],src);J,_=operators(family,model,c,rt,u)
   correction=np.linalg.solve(J,-r);s=np.linalg.svd(J/np.sqrt(m.dV[:,None]*m.dV[None,:]),compute_uv=False)
   residual_norm=norm(r/m.dV)/norm(src/m.dV);correction_norm=norm(correction)/max(1,norm(u))
   assert residual_norm<1e-8 and correction_norm<1e-7,(family,i,residual_norm,correction_norm)
   verified.append(dict(root=i,residual=residual_norm,relative_newton_correction=correction_norm,sigma_min=float(s[-1]),error_pct=100*norm(u-c['U_dns'])/norm(c['U_dns']),reflection_asymmetry=norm(u-u[::-1])/max(1,norm(u)),velocity_min=float(u.min()),velocity_max=float(u.max()),hits=row['root_counts'][i]))
   ax.plot(m.yc,u,label=(f'Root {i+1} ({row["root_counts"][i]})' if len(roots)<=8 else ('Computed roots' if i==0 else '_nolegend_')),lw=1)
  ax.plot(m.yc,c['U_dns'],'k--',lw=1,label='DNS');ax.set_title(f'{family}, seed {seed}\n{len(roots)} roots / {row["attempted"]} IC');ax.set_xlabel('y/h');ax.set_ylabel('U+');ax.legend(fontsize=7)
  records=json.loads((OUT/family/'merged_records.json').read_text());by_distribution={}
  records.sort(key=lambda v:v['ic']);assert len(set(v['ic'] for v in records))==len(records)
  if row['complete']:assert [v['ic'] for v in records]==list(range(10000))
  seen=set();discovery=[]
  for v in records:
   name=labels[v['ic']];t=by_distribution.setdefault(name,dict(attempted=0,hits=[0]*len(roots),timeouts=0));t['attempted']+=1;t['timeouts']+=int(v['status']=='timeout')
   if v['root'] is not None:seen.add(v['root']);t['hits'][v['root']]+=1
   discovery.append(len(seen))
  curve.step([v['ic']+1 for v in records],discovery,where='post');curve.set_xlabel('IC index');curve.set_ylabel('Distinct roots found');curve.set_ylim(bottom=0);curve.grid(alpha=.2)
  write(OUT/family/'merged_records.json',records)
  all_results.append(dict(family=family,seed=seed,roots=verified,by_distribution=by_distribution,checkpoint_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),complete=row['complete']))
 fig.suptitle('Common Re_tau = 1000: frozen-model root multiplicity and discovery curves')
 fig.savefig(OUT/'roots_and_discovery.pdf');fig.savefig(OUT/'roots_and_discovery.png',dpi=160);plt.close(fig)
 write(OUT/'validated_roots.json',all_results)
 p=OUT/'REPORT.md'
 with p.open('a') as f:
  f.write('\n## Independent validation\n\nEvery merged representative passed a fresh dense Jacobian Newton-correction check. The detailed residuals, smallest weighted Jacobian singular values, reflection asymmetries, and hit counts by IC distribution are in `validated_roots.json`.\n\n![Root profiles and discovery curves](roots_and_discovery.png)\n')
 print('Validated representatives:',sum(len(r['roots']) for r in all_results),flush=True)
if __name__=='__main__':main()
