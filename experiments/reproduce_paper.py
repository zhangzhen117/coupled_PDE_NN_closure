"""Regenerate current paper figures and tables from distributed artifacts."""
import os, sys, subprocess, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]
import torch
torch.set_num_threads(1)
(ROOT / 'figures').mkdir(exist_ok=True)
(ROOT / 'paper/tex/tables').mkdir(parents=True, exist_ok=True)
import make_figures as f
import make_tables as t
f.fig_apriori()
f.fig_indices()
f.fig_bound()
for ext in ('pdf', 'png'):
    (ROOT / f'figures/fig3_error_bound.{ext}').rename(ROOT / f'figures/figS1_error_bound.{ext}')
S, R, P, sf = t.s20(), t.s21(), t.s12(), t.frozen_sigma()
t.tab_closures_apriori()
t.tab_indices_merged(S, sf)
t.tab_useful(S)
t.tab_ratio(R, S)
t.tab_precond(P)
env = dict(os.environ, MULTISTART_OUTPUT=str(ROOT / 'data/multistart'))
subprocess.run([sys.executable, str(ROOT / 'experiments/multistart/plot_velocity_stress.py')], check=True, env=env)
for ext in ('pdf', 'png'):
    shutil.copyfile(ROOT / f'data/multistart/velocity_stress_five_closures_full.{ext}', ROOT / f'figures/fig3_fixed_model_multistart.{ext}')
