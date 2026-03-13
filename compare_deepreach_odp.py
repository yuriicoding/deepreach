"""
compare_deepreach_odp.py
Run from the deepreach directory:
    python compare_deepreach_odp.py \
        --checkpoint_path runs/inference/model_epoch_100000.pth

Loads odp_V.npy + odp_comparison_metadata.json (from save_odp_grid.py),
runs DeepReach inference at every ODP grid point, prints metrics, and saves plots.

Plots (saved to ./brs_comparison_plots/):
  brs_overview_theta{X}.png    -- grid: rows=time, cols=velocity, fixed theta
  brs_t{k}_tau{t}.png          -- one plot per time step, cols=velocity

Color scheme:
  Purple -- both DeepReach and ODP agree state is in BRS  (V <= 0)
  Blue   -- ODP only in BRS
  Red    -- DeepReach only in BRS
  White  -- both agree state is safe
"""

import argparse
import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from dynamics import dynamics as dynamics_module
from utils import modules

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
# checkpoint
parser.add_argument('--checkpoint_path', type=str, required=True,
                    help='Path to the .pth weights file.')
# model architecture
parser.add_argument('--num_hl', type=int, default=3,
                    help='Number of hidden layers.')
parser.add_argument('--num_nl', type=int, default=512,
                    help='Number of neurons per hidden layer.')
parser.add_argument('--model', type=str, default='sine',
                    choices=['sine', 'tanh', 'sigmoid', 'relu'])
parser.add_argument('--model_mode', type=str, default='mlp',
                    choices=['mlp', 'rbf', 'pinn'])
parser.add_argument('--deepreach_model', type=str, default='exact',
                    choices=['exact', 'diff', 'vanilla'])
# dynamics
parser.add_argument('--dynamics_class', type=str, default='Dubins4D_new')
parser.add_argument('--goalR', type=float, default=0.25)
parser.add_argument('--omega_max', type=float, default=0.1)
parser.add_argument('--accel_max', type=float, default=0.1)
parser.add_argument('--angle_alpha_factor', type=float, default=1.0)
parser.add_argument('--velocity_alpha_factor', type=float, default=0.5)
parser.add_argument('--set_mode', type=str, default='avoid')
parser.add_argument('--freeze_model', action='store_true', default=False)
# inference
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--batch_size', type=int, default=100000)
# data / output
parser.add_argument('--odp_V', type=str, default='odp_V.npy')
parser.add_argument('--odp_meta', type=str, default='odp_comparison_metadata.json')
parser.add_argument('--theta_fixed', type=float, default=0.0)
parser.add_argument('--out_dir', type=str, default='brs_comparison_plots')
opt = parser.parse_args()

os.makedirs(opt.out_dir, exist_ok=True)

# ── Load ODP data ─────────────────────────────────────────────────────────────
print("Loading ODP data ...")
V_odp = np.load(opt.odp_V)
with open(opt.odp_meta) as f:
    meta = json.load(f)

x_pts     = np.array(meta['axes']['dim_0']['values'])
y_pts     = np.array(meta['axes']['dim_1']['values'])
theta_pts = np.array(meta['axes']['dim_2']['values'])
v_pts     = np.array(meta['axes']['dim_3']['values'])
tau       = np.array(meta['axes']['dim_4']['values'])
goalR     = meta['dynamics']['goal_R']

Nx, Ny, Nth, Nv, Ntau = V_odp.shape
print(f"  ODP grid: ({Nx}, {Ny}, {Nth}, {Nv}, {Ntau})  dtype={V_odp.dtype}")

# ── Build dynamics ────────────────────────────────────────────────────────────
print("Building dynamics ...")
dynamics_class = getattr(dynamics_module, opt.dynamics_class)
dyn = dynamics_class(
    goalR=opt.goalR,
    omega_max=opt.omega_max,
    accel_max=opt.accel_max,
    angle_alpha_factor=opt.angle_alpha_factor,
    velocity_alpha_factor=opt.velocity_alpha_factor,
    set_mode=opt.set_mode,
    freeze_model=opt.freeze_model,
)
dyn.deepreach_model = opt.deepreach_model
print(f"  {opt.dynamics_class}  input_dim={dyn.input_dim}  state_dim={dyn.state_dim}")

# ── Build and load model ──────────────────────────────────────────────────────
print("Loading DeepReach model ...")
model = modules.SingleBVPNet(
    in_features=dyn.input_dim, out_features=1,
    type=opt.model, mode=opt.model_mode,
    final_layer_factor=1., hidden_features=opt.num_nl,
    num_hidden_layers=opt.num_hl
)
ckpt = torch.load(opt.checkpoint_path, map_location=opt.device)
model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
model.to(opt.device)
model.eval()
print(f"  Loaded: {opt.checkpoint_path}")

# ── Run inference on all ODP grid points ──────────────────────────────────────
print("Building coordinate array ...")
XX, YY, THH, VV, TT = np.meshgrid(x_pts, y_pts, theta_pts, v_pts, tau, indexing='ij')
N = XX.size

coords_np = np.empty((N, 5), dtype=np.float32)
coords_np[:, 0] = TT.ravel()
coords_np[:, 1] = XX.ravel()
coords_np[:, 2] = YY.ravel()
coords_np[:, 3] = THH.ravel()
coords_np[:, 4] = VV.ravel()

print(f"Running DeepReach inference on {N:,} points ...")
use_cuda = opt.device.startswith('cuda')

# Pin memory enables async CPU→GPU DMA (no effect on CPU)
coords_t  = torch.tensor(coords_np).pin_memory() if use_cuda else torch.tensor(coords_np)
V_dr_flat = np.empty(N, dtype=np.float32)


if use_cuda:
    compute_stream = torch.cuda.Stream()
    transfer_stream = torch.cuda.Stream()

for start in range(0, N, opt.batch_size):
    end = min(start + opt.batch_size, N)
    if use_cuda:
        # Overlap host→device transfer with GPU compute from previous batch
        with torch.cuda.stream(transfer_stream):
            batch = coords_t[start:end].to(opt.device, non_blocking=True)
        compute_stream.wait_stream(transfer_stream)
        with torch.cuda.stream(compute_stream):
            with torch.no_grad():
                model_input = dyn.coord_to_input(batch)
                model_out   = model({'coords': model_input})
                vals = dyn.io_to_value(
                    model_out['model_in'].detach(),
                    model_out['model_out'].squeeze(-1).detach()
                )
        torch.cuda.current_stream().wait_stream(compute_stream)
    else:
        with torch.no_grad():
            batch       = coords_t[start:end]
            model_input = dyn.coord_to_input(batch)
            model_out   = model({'coords': model_input})
            vals = dyn.io_to_value(
                model_out['model_in'].detach(),
                model_out['model_out'].squeeze(-1).detach()
            )
    V_dr_flat[start:end] = vals.cpu().numpy()
    print(f"  {end:>10,} / {N:,}", end='\r')

print()
V_dr = V_dr_flat.reshape(Nx, Ny, Nth, Nv, Ntau).astype(np.float64)

# ── Save DeepReach value function ─────────────────────────────────────────────
dr_npy = os.path.join(os.path.dirname(opt.odp_V), 'dr_V.npy')
np.save(dr_npy, V_dr)
print(f"Saved DeepReach V: {dr_npy}  shape={V_dr.shape}  dtype={V_dr.dtype}")

# ── BRS membership masks ──────────────────────────────────────────────────────
in_odp   = V_odp <= 0
in_dr    = V_dr  <= 0
both     = in_odp &  in_dr
odp_only = in_odp & ~in_dr
dr_only  = ~in_odp &  in_dr

# ── Metrics ───────────────────────────────────────────────────────────────────
total        = N
pct_both     = 100.0 * both.sum()     / total
pct_odp_only = 100.0 * odp_only.sum() / total
pct_dr_only  = 100.0 * dr_only.sum()  / total
pct_disagree = pct_odp_only + pct_dr_only
mae  = np.mean(np.abs(V_dr.ravel()  - V_odp.ravel()))
rmse = np.sqrt(np.mean((V_dr.ravel() - V_odp.ravel()) ** 2))

print("\n── Overall metrics ──────────────────────────────────────────────────────")
print(f"  MAE                        : {mae:.5f}")
print(f"  RMSE                       : {rmse:.5f}")
print(f"  Both in BRS (agree unsafe) : {pct_both:.2f}%")
print(f"  ODP only in BRS            : {pct_odp_only:.2f}%")
print(f"  DeepReach only in BRS      : {pct_dr_only:.2f}%")
print(f"  Total BRS disagreement     : {pct_disagree:.2f}%")

print("\n── Per time step ────────────────────────────────────────────────────────")
print(f"  {'t':>6}  {'MAE':>8}  {'RMSE':>8}  {'disagree%':>10}  {'ODP_only%':>10}  {'DR_only%':>9}")
print("  " + "-" * 60)
for itau, t in enumerate(tau):
    n  = Nx * Ny * Nth * Nv
    vo = V_odp[..., itau].ravel()
    vd = V_dr[...,  itau].ravel()
    print(f"  {t:6.3f}  "
          f"{np.mean(np.abs(vd-vo)):8.5f}  "
          f"{np.sqrt(np.mean((vd-vo)**2)):8.5f}  "
          f"{100.0*(odp_only[...,itau].sum()+dr_only[...,itau].sum())/n:10.2f}  "
          f"{100.0*odp_only[...,itau].sum()/n:10.2f}  "
          f"{100.0*dr_only[...,itau].sum()/n:9.2f}")

# ── Plotting ──────────────────────────────────────────────────────────────────
COLORS = {
    'both':     [0.55, 0.00, 0.75],
    'odp_only': [0.10, 0.45, 0.85],
    'dr_only':  [0.90, 0.20, 0.20],
    'safe':     [1.00, 1.00, 1.00],
}
legend_handles = [
    mpatches.Patch(color=COLORS['both'],     label='Both in BRS'),
    mpatches.Patch(color=COLORS['odp_only'], label='ODP only'),
    mpatches.Patch(color=COLORS['dr_only'],  label='DeepReach only'),
    mpatches.Patch(color=COLORS['safe'],     label='Both safe'),
]

ith_fixed = int(np.argmin(np.abs(theta_pts - opt.theta_fixed)))
theta_val = theta_pts[ith_fixed]
extent    = [x_pts[0], x_pts[-1], y_pts[0], y_pts[-1]]

def make_rgb(b, oo, dr):
    rgb = np.ones((*b.shape, 3))
    rgb[b,  :] = COLORS['both']
    rgb[oo, :] = COLORS['odp_only']
    rgb[dr, :] = COLORS['dr_only']
    return rgb

def add_target(ax):
    ax.add_patch(plt.Rectangle(
        (-goalR, -goalR), 2 * goalR, 2 * goalR,
        linewidth=1.5, edgecolor='black', facecolor='none', linestyle='--'
    ))

v_indices    = np.linspace(0, Nv   - 1, min(5, Nv),   dtype=int)
time_indices = np.linspace(0, Ntau - 1, min(4, Ntau), dtype=int)

# Overview grid: rows=time, cols=velocity
print(f"\nSaving overview plot (theta={theta_val:.2f} rad) ...")
n_rows, n_cols = len(time_indices), len(v_indices)
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(3.5 * n_cols, 3.2 * n_rows),
                         squeeze=False)
for ri, itau in enumerate(time_indices):
    for ci, iv in enumerate(v_indices):
        rgb = make_rgb(both    [:, :, ith_fixed, iv, itau],
                       odp_only[:, :, ith_fixed, iv, itau],
                       dr_only [:, :, ith_fixed, iv, itau])
        ax = axes[ri][ci]
        ax.imshow(rgb.transpose(1, 0, 2), origin='lower', extent=extent, aspect='equal')
        add_target(ax)
        ax.set_title(f't={tau[itau]:.2f}  v={v_pts[iv]:.2f}', fontsize=9)
        ax.set_xlabel('x', fontsize=8)
        ax.set_ylabel('y', fontsize=8)
        ax.tick_params(labelsize=7)

fig.legend(handles=legend_handles, loc='lower center', ncol=4,
           fontsize=9, bbox_to_anchor=(0.5, -0.01))
fig.suptitle(
    f'BRS Comparison  |  θ={theta_val:.2f} rad  |  disagree={pct_disagree:.1f}%  |  MAE={mae:.4f}',
    fontsize=12
)
plt.tight_layout(rect=[0, 0.05, 1, 1])
out_path = os.path.join(opt.out_dir, f'brs_overview_theta{theta_val:.2f}.png')
plt.savefig(out_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved: {out_path}")

# One plot per time step
print("Saving per-time-step plots ...")
for itau in range(Ntau):
    fig2, axes2 = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.5), squeeze=False)
    for ci, iv in enumerate(v_indices):
        b  = both    [:, :, ith_fixed, iv, itau]
        oo = odp_only[:, :, ith_fixed, iv, itau]
        dr = dr_only [:, :, ith_fixed, iv, itau]
        d_pct = 100.0 * (oo.sum() + dr.sum()) / (Nx * Ny)
        ax = axes2[0][ci]
        ax.imshow(make_rgb(b, oo, dr).transpose(1, 0, 2),
                  origin='lower', extent=extent, aspect='equal')
        add_target(ax)
        ax.set_title(f'v={v_pts[iv]:.2f}  diff={d_pct:.1f}%', fontsize=9)
        ax.set_xlabel('x', fontsize=8)
        ax.set_ylabel('y', fontsize=8)
        ax.tick_params(labelsize=7)

    fig2.legend(handles=legend_handles, loc='lower center', ncol=4,
                fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig2.suptitle(
        f'BRS Comparison  |  t={tau[itau]:.2f}  |  θ={theta_val:.2f} rad',
        fontsize=11
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = os.path.join(opt.out_dir, f'brs_t{itau:02d}_tau{tau[itau]:.2f}.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

print("\nDone.")
