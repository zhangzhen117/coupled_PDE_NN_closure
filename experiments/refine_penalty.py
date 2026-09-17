"""Refine an A1 checkpoint with zero and intermediate reference states."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from torch.nn.utils import parameters_to_vector, vector_to_parameters
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from closure.train import build, per_case_data, data_loss, reaction_penalty_full


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--maxiter', type=int, default=30000)
    ap.add_argument('--gtol', type=float, default=1e-5)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output path; checkpoints are not overwritten.')
    torch.set_num_threads(1)
    original = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = build('force_pen2w', args.seed)
    model.load_state_dict(original['state'])
    data = per_case_data()
    alphas = [0., .25, .5, .75]
    extra = [dict(c, U=a*c['U'], S=a*c['S']) for c in data for a in alphas]
    last = {}
    def objective(x):
        vector_to_parameters(torch.from_numpy(x.copy()), model.parameters())
        model.zero_grad()
        fit = data_loss(model, 'force_pen2w', data)
        ref = reaction_penalty_full(model, data)
        path = reaction_penalty_full(model, extra)
        loss = fit + 30*ref + 30*path
        loss.backward()
        grad = parameters_to_vector([p.grad for p in model.parameters()]).detach().numpy().copy()
        last.update(loss=float(loss.detach()), fit=float(fit.detach()),
                    reference_penalty=float(ref.detach()), path_penalty=float(path.detach()),
                    grad_norm=float(np.linalg.norm(grad)))
        return 1e4*last['loss'], 1e4*grad
    def callback(x):
        objective(x)
        if last['grad_norm'] <= args.gtol:
            raise StopIteration
    x = parameters_to_vector(model.parameters()).detach().numpy().copy()
    result = minimize(objective, x, jac=True, method='L-BFGS-B', callback=callback,
                      options=dict(maxiter=args.maxiter, maxfun=100000, maxcor=200,
                                   maxls=60, ftol=0., gtol=1e-5))
    objective(result.x)
    info = dict(original.get('info', {}), **last)
    info.update(converged=last['grad_norm'] <= args.gtol, gtol=args.gtol,
                stop_reason=str(result.message), penalty_alphas=[1.]+alphas,
                extra_penalty_weight=30., source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest())
    # Do not propagate stale optimizer state or per-case accuracy measurements.
    info.pop('per_re', None)
    info.pop('finetune_solves', None)
    info.pop('finetune', None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=model.state_dict(), info=info), args.output)
    print(json.dumps(dict(last, converged=info['converged'], stop_reason=info['stop_reason']), indent=2))

if __name__ == '__main__':
    main()
