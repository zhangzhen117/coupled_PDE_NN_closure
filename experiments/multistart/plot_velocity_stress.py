"""Full-channel roots, unobstructed panels, compact headings and a legend inside the penalty panel."""
import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from chan.cases import case_setup
from closure.jacobian import _cell_gradient_coeffs
import os
OUT=Path(os.environ.get('MULTISTART_OUTPUT', ROOT/'data/multistart')).resolve()
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.7,'legend.fontsize':10,'savefig.facecolor':'white'})
rows={r['family']:r for r in json.loads((OUT/'summary.json').read_text())}
c=case_setup(1000);y=c['m'].yc;dns=c['U_dns']
cl,cc,cr=_cell_gradient_coeffs(c['m'])
def stress(u):
 ue=np.concatenate([[0.],u,[0.]])
 return (1.-y)-c['nu']*(cl*ue[:-2]+cc*ue[1:-1]+cr*ue[2:])
tau_dns=stress(dns)
selection=[('force','A0 (force)'),('force_pen2w','A1 (penalty)'),('tau','B0 (stress)'),('nueff','B1 (tangent diffusion)'),('hybrid','C0 (hybrid)')]
for count in [3,5]:
 fig,panels=plt.subplots(2,count,figsize=(5.3,3.15),sharex=True)
 axs=panels[0]
 for column,(ax,(fam,name)) in enumerate(zip(axs,selection[:count])):
  axT=panels[1,column]
  r=rows[fam];assert r['complete'];roots=[np.load(OUT/fam/f'root{i}.npy') for i in range(r['distinct_roots'])]
  # Paper classification: the secondary near-DNS C0 root (3.90% error)
  # is brown; only the closest C0 root (0.476% error) stays blue.
  closest=int(np.argmin(r['root_error_pct']))
  for i,(u,err) in enumerate(zip(roots,r['root_error_pct'])):
   physical=err<5 and (fam!='hybrid' or i==closest)
   ax.plot(y,u,color='#2a78d6' if physical else '#b3411a',lw=1.0 if physical else .55,alpha=1 if physical else .85)
   axT.plot(y,stress(u),color='#2a78d6' if physical else '#b3411a',lw=1.0 if physical else .55,alpha=1 if physical else .85)
  axT.plot(y,tau_dns,'--',color='#0b0b0b',lw=1.15,zorder=5)
  tv=np.concatenate([stress(u) for u in roots]+[tau_dns]);tlo=float(tv.min());thi=float(tv.max());tp=.04*max(1,thi-tlo)
  axT.set(xlim=(0,2),ylim=(tlo-tp,thi+tp),xlabel='$y/h$',ylabel=(r'$\tau/u_\tau^2$' if column==0 else ''))
  axT.set_xticks([0,1,2]);axT.spines[['top','right']].set_visible(False);axT.grid(alpha=.25,lw=.5);axT.set_axisbelow(True)
  axT.axhline(0,color='#777777',lw=.5,zorder=0)
  ax.plot(y,dns,'--',color='#0b0b0b',lw=1.15,zorder=5)
  values=np.concatenate(roots+[dns]);lo=float(values.min());hi=float(values.max());pad=.04*max(1,hi-lo)
  ax.set(xlim=(0,2),ylim=(lo-pad,hi+pad),ylabel=('$U^+$' if column==0 else ''))
  ax.set_xticks([0,1,2]);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.25,lw=.5);ax.set_axisbelow(True)
  text=f'{name.split()[0]}: {len(roots)} '+('root' if len(roots)==1 else 'roots')
  ax.set_title(text,fontsize=7.5,pad=6)
  from matplotlib.ticker import MaxNLocator
  for panel in (ax,axT):
   panel.yaxis.set_major_locator(MaxNLocator(nbins=3,steps=[1,2,5,10]))
   panel.tick_params(length=2.5,pad=2)
 keys=[Line2D([],[],color='#0b0b0b',ls='--',lw=1.7,label='DNS'),Line2D([],[],color='#b3411a',lw=1.2,label='Non-\nphysical'),Line2D([],[],color='#2a78d6',lw=1.3,label='Physical')]
 fig.subplots_adjust(left=.105,right=.99,top=.90,bottom=.13,wspace=.52,hspace=.22)
 axs[1].legend(handles=keys,loc='center',bbox_to_anchor=(.52,.40),frameon=False,handlelength=1.4,handletextpad=.35,borderaxespad=0,fontsize=6.8)
 stem='velocity_stress_five_closures_full' if count==5 else 'velocity_stress_three_closures_full'
 fig.savefig(OUT/f'{stem}.png',dpi=220);fig.savefig(OUT/f'{stem}.pdf');plt.close(fig)
(OUT/'velocity_stress_five_closures_caption.txt').write_text('Full-channel velocity (top) and Reynolds shear stress (bottom) at Re_tau=1000, one frozen checkpoint per closure, 10,000 shared initial conditions each. One curve per distinct verified root. Stress is reconstructed using the same convention and cell-gradient stencil as the paper: tau/u_tau^2 = (1-y/h) - nu*dU/dy; DNS stress is reconstructed from the DNS velocity by the identical procedure. Dashed black: DNS; brown: non-physical solution; blue: physical solution. Colour convention: relative weighted velocity error below 5% is blue except for C0, where the user requests the secondary near-DNS root (root0, 3.9033% error) be brown; only the closest C0 root (root2, 0.4757% error) is blue. This explicit display classification is not an independent test of physical admissibility and does not change the recorded errors or root counts. Independent linear velocity ranges show the minimum and maximum of all roots and DNS, with 4% padding. Root counts: A0 7, A1 1, B0 93, B1 1, C0 4.\n')
