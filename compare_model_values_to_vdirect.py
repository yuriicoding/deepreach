"""
Compare a saved DeepReach prediction array against v_direct_all.npy and write
a text report with the same basename as the prediction file.

Example:
    python compare_model_values_to_vdirect.py --model_values dubins4d_classic_run_values.npy
"""

import argparse
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--model_values', type=str, required=True, help='Prediction .npy file in the repo root or an absolute path.')
opt = parser.parse_args()

repo_root = Path(__file__).resolve().parent
model_values_path = Path(opt.model_values)
if not model_values_path.is_absolute():
    model_values_path = repo_root / model_values_path
vdirect_path = repo_root / 'v_direct_all.npy'

if not model_values_path.exists():
    raise FileNotFoundError(f'Missing model values file: {model_values_path}')
if not vdirect_path.exists():
    raise FileNotFoundError(f'Missing v_direct_all.npy: {vdirect_path}')

pred = np.load(model_values_path)
vdirect = np.load(vdirect_path)
if pred.shape != vdirect.shape:
    raise ValueError(f'Shape mismatch: pred={pred.shape}, v_direct={vdirect.shape}')

diff = pred - vdirect
mae = float(np.mean(np.abs(diff)))
rmse = float(np.sqrt(np.mean(diff ** 2)))

pred_nonpos = pred <= 0.0
vdirect_nonpos = vdirect <= 0.0
sign_agreement = float(np.mean((pred >= 0.0) == (vdirect >= 0.0)) * 100.0)
unsafe_agreement = float(np.mean(pred_nonpos == vdirect_nonpos) * 100.0)
false_safe = float(np.mean((pred > 0.0) & (vdirect <= 0.0)) * 100.0)
false_unsafe = float(np.mean((pred <= 0.0) & (vdirect > 0.0)) * 100.0)

report_lines = [
    f'model_values_file: {model_values_path.name}',
    f'v_direct_file: {vdirect_path.name}',
    f'shape: {pred.shape}',
    f'mae: {mae:.10f}',
    f'rmse: {rmse:.10f}',
    f'sign_agreement_percent: {sign_agreement:.6f}',
    f'unsafe_set_agreement_percent: {unsafe_agreement:.6f}',
    f'false_safe_percent: {false_safe:.6f}',
    f'false_unsafe_percent: {false_unsafe:.6f}',
]

report_path = model_values_path.with_suffix('.txt')
report_path.write_text('\n'.join(report_lines) + '\n', encoding='utf-8')

print('\n'.join(report_lines))
print(f'report_saved_to: {report_path}')
