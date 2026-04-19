"""
Run full-grid DeepReach inference on the artifact grid used by v_direct_all.npy /
v_hat_all.npy and save predictions with the same shape.

Example:
    python infer_deepreach_to_artifact_grid.py --model_name native_run

This loads:
    runs/native_run/

and saves:
    native_run_values.npy
"""

import argparse
import inspect
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from dynamics import dynamics as dynamics_module
from utils import modules


def _axis_from_range(lo, hi, n, periodic=False):
    if periodic:
        return np.linspace(float(lo), float(hi), int(n), endpoint=False, dtype=np.float32)
    return np.linspace(float(lo), float(hi), int(n), endpoint=True, dtype=np.float32)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True, help='Experiment directory name under ./runs')
parser.add_argument('--checkpoint', type=int, default=-1, help='Epoch to load (-1 = model_final.pth).')
parser.add_argument('--device', type=str, default=None, help='Inference device, e.g. cpu or cuda:0. Defaults to cuda:0 when available.')
parser.add_argument('--batch_size', type=int, default=100000)
parser.add_argument('--artifact_manifest', type=str, default='artifact_manifest.json')
parser.add_argument('--reference_array', type=str, default='v_direct_all.npy')
opt = parser.parse_args()

if opt.device is None:
    opt.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {opt.device}')

repo_root = Path(__file__).resolve().parent
experiment_dir = repo_root / 'runs' / opt.model_name
if not experiment_dir.exists():
    raise FileNotFoundError(f'Missing experiment directory: {experiment_dir}')
manifest_path = repo_root / opt.artifact_manifest
reference_path = repo_root / opt.reference_array
if not manifest_path.exists():
    raise FileNotFoundError(f'Missing artifact manifest: {manifest_path}')
if not reference_path.exists():
    raise FileNotFoundError(f'Missing reference array: {reference_path}')

with open(manifest_path, 'r', encoding='utf-8') as f:
    json.load(f)

reference = np.load(reference_path, mmap_mode='r')
target_shape = tuple(reference.shape)
if len(target_shape) != 5:
    raise ValueError(f'Expected 5D artifact array, got shape {target_shape}')

with open(experiment_dir / 'orig_opt.pickle', 'rb') as f:
    orig_opt = pickle.load(f)

dynamics_class = getattr(dynamics_module, orig_opt.dynamics_class)
dyn_kwargs = {
    k: getattr(orig_opt, k)
    for k in inspect.signature(dynamics_class).parameters.keys()
    if k != 'self' and hasattr(orig_opt, k)
}
dyn = dynamics_class(**dyn_kwargs)
dyn.deepreach_model = orig_opt.deepreach_model

model = modules.SingleBVPNet(
    in_features=dyn.input_dim,
    out_features=1,
    type=orig_opt.model,
    mode=orig_opt.model_mode,
    final_layer_factor=1.0,
    hidden_features=orig_opt.num_nl,
    num_hidden_layers=orig_opt.num_hl,
)

checkpoints_dir = experiment_dir / 'training' / 'checkpoints'
if opt.checkpoint == -1:
    epoch_100000_path = checkpoints_dir / 'model_epoch_100000.pth'
    if epoch_100000_path.exists():
        ckpt_path = epoch_100000_path
    else:
        raise FileNotFoundError(f'Could not find model_epoch_100000.pth in {checkpoints_dir}')
    state = torch.load(ckpt_path, map_location=opt.device)
    if isinstance(state, dict) and 'model' in state:
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)
else:
    ckpt_path = checkpoints_dir / f'model_epoch_{opt.checkpoint:04d}.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Missing checkpoint: {ckpt_path}')
    state = torch.load(ckpt_path, map_location=opt.device)
    if isinstance(state, dict) and 'model' in state:
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)

model.to(opt.device)
model.eval()

nx, ny, nv, nth, nt = target_shape
state_ranges = dyn.state_test_range()
x_axis = _axis_from_range(state_ranges[0][0], state_ranges[0][1], nx, periodic=False)
y_axis = _axis_from_range(state_ranges[1][0], state_ranges[1][1], ny, periodic=False)
v_axis = _axis_from_range(state_ranges[2][0], state_ranges[2][1], nv, periodic=False)
th_axis = _axis_from_range(state_ranges[3][0], state_ranges[3][1], nth, periodic=True)
t_axis = _axis_from_range(orig_opt.tMin, orig_opt.tMax, nt, periodic=False)

XX, YY, VV, THH, TT = np.meshgrid(x_axis, y_axis, v_axis, th_axis, t_axis, indexing='ij')
N = XX.size
coords_np = np.empty((N, 5), dtype=np.float32)
coords_np[:, 0] = TT.ravel()
coords_np[:, 1] = XX.ravel()
coords_np[:, 2] = YY.ravel()
coords_np[:, 3] = VV.ravel()
coords_np[:, 4] = THH.ravel()

coords_t = torch.from_numpy(coords_np)
pred_flat = np.empty(N, dtype=np.float32)

for start in range(0, N, opt.batch_size):
    end = min(start + opt.batch_size, N)
    batch = coords_t[start:end].to(opt.device)
    with torch.inference_mode():
        model_input = dyn.coord_to_input(batch)
        model_out = model({'coords': model_input})
        vals = dyn.io_to_value(
            model_out['model_in'].detach(),
            model_out['model_out'].squeeze(-1).detach(),
        )
    pred_flat[start:end] = vals.detach().cpu().numpy()
    print(f'  {end:>10,} / {N:,}', end='\r')
print()

pred = pred_flat.reshape(target_shape)
output_path = repo_root / f'{opt.model_name}_values.npy'
np.save(output_path, pred)
print(f'Saved prediction array: {output_path}')
print(f'Loaded checkpoint: {ckpt_path}')
print(f'Output shape: {pred.shape}, dtype={pred.dtype}')
