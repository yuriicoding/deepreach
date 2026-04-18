import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class _ArtifactGuidance:
    def __init__(self, dynamics, repo_root, t_min, t_max, gt_radius, close_gap_scale):
        self.dynamics = dynamics
        self.repo_root = Path(repo_root)
        self.gt_radius = float(gt_radius)
        self.close_gap_scale = float(close_gap_scale)

        manifest_path = self.repo_root / 'artifact_manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f'Missing artifact manifest at {manifest_path}')
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        vhat_rel = manifest['values']['v_hat_all']['path']
        gap_rel = manifest['values']['close_value_gap_all']['path']
        self.vhat = np.load(self.repo_root / vhat_rel, mmap_mode='r')
        self.close_gap = np.load(self.repo_root / gap_rel, mmap_mode='r')
        if self.vhat.shape != self.close_gap.shape:
            raise ValueError(f'Artifact shape mismatch: v_hat={self.vhat.shape}, close_gap={self.close_gap.shape}')

        self.shape = self.vhat.shape
        if len(self.shape) != self.dynamics.state_dim + 1:
            raise ValueError(
                f'Expected {self.dynamics.state_dim + 1}D artifact, got shape {self.shape}'
            )

        self.axes = []
        self.step_sizes = []
        time_axis = np.linspace(float(t_min), float(t_max), self.shape[0], endpoint=True, dtype=np.float32)
        self.axes.append(time_axis)
        self.step_sizes.append(self._axis_step(time_axis))
        for dim, (lo, hi) in enumerate(self.dynamics.state_test_range()):
            axis = np.linspace(float(lo), float(hi), self.shape[dim + 1], endpoint=True, dtype=np.float32)
            self.axes.append(axis)
            self.step_sizes.append(self._axis_step(axis))

    @staticmethod
    def _axis_step(axis):
        if len(axis) <= 1:
            return 1.0
        return float(axis[1] - axis[0])

    def query(self, real_coords):
        coords_np = real_coords.detach().cpu().numpy().astype(np.float32)
        nearest_idx = np.empty_like(coords_np, dtype=np.int64)
        normalized_sq = np.zeros(coords_np.shape[0], dtype=np.float32)

        for dim, axis in enumerate(self.axes):
            step = max(self.step_sizes[dim], 1e-12)
            raw = (coords_np[:, dim] - axis[0]) / step
            clipped = np.clip(np.rint(raw), 0, len(axis) - 1).astype(np.int64)
            nearest_idx[:, dim] = clipped
            nearest_vals = axis[clipped]
            normalized_sq += ((coords_np[:, dim] - nearest_vals) / step) ** 2

        normalized_dist = np.sqrt(normalized_sq)
        guidance_mask = normalized_dist <= self.gt_radius

        matched_vhat = self.vhat[tuple(nearest_idx[:, dim] for dim in range(nearest_idx.shape[1]))].astype(np.float32)
        matched_gap = self.close_gap[tuple(nearest_idx[:, dim] for dim in range(nearest_idx.shape[1]))].astype(np.float32)

        # Smaller close-gap means stronger PDE freedom and weaker v_hat anchoring.
        guidance_weight = np.clip(
            matched_gap / max(self.close_gap_scale, 1e-12),
            a_min=0.0,
            a_max=1.0,
        ).astype(np.float32)
        guidance_weight = guidance_weight * guidance_mask.astype(np.float32)

        return {
            'guidance_mask': torch.from_numpy(guidance_mask),
            'guidance_vhat': torch.from_numpy(matched_vhat),
            'guidance_gap': torch.from_numpy(matched_gap),
            'guidance_weight': torch.from_numpy(guidance_weight),
        }

# uses model input and real boundary fn
class ReachabilityDataset(Dataset):
    def __init__(
        self,
        dynamics,
        numpoints,
        pretrain,
        pretrain_iters,
        tMin,
        tMax,
        counter_start,
        counter_end,
        num_src_samples,
        num_target_samples,
        seed=0,
        rank=0,
        world_size=1,
        use_vhat_guidance=False,
        gt_radius=0.05,
        close_gap_scale=0.1,
        repo_root=None,
    ):
        self.dynamics = dynamics
        self.numpoints = numpoints
        self.pretrain = pretrain
        self.pretrain_counter = 0
        self.pretrain_iters = pretrain_iters
        self.tMin = tMin 
        self.tMax = tMax 
        self.counter = counter_start 
        self.counter_end = counter_end 
        self.num_src_samples = num_src_samples
        self.num_target_samples = num_target_samples
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.use_vhat_guidance = bool(use_vhat_guidance)
        self.artifact_guidance = None
        if self.use_vhat_guidance:
            root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
            self.artifact_guidance = _ArtifactGuidance(
                dynamics=dynamics,
                repo_root=root,
                t_min=tMin,
                t_max=tMax,
                gt_radius=gt_radius,
                close_gap_scale=close_gap_scale,
            )

    def __len__(self):
        return 1

    def _local_count(self, count):
        base = count // self.world_size
        remainder = count % self.world_size
        return base + int(self.rank < remainder)

    def __getitem__(self, idx):
        local_numpoints = self._local_count(self.numpoints)
        local_num_src_samples = min(self._local_count(self.num_src_samples), local_numpoints)
        local_num_target_samples = min(self._local_count(self.num_target_samples), local_numpoints)

        step_seed = (
            self.seed
            + self.rank
            + 1000003 * self.counter
            + 9176 * self.pretrain_counter
        )
        generator = torch.Generator()
        generator.manual_seed(step_seed)

        # uniformly sample domain and include coordinates where source is non-zero 
        model_states = torch.empty(local_numpoints, self.dynamics.state_dim).uniform_(-1, 1, generator=generator)
        if local_num_target_samples > 0:
            with torch.random.fork_rng():
                torch.manual_seed(step_seed + 1)
                target_state_samples = self.dynamics.sample_target_state(local_num_target_samples)
            model_states[-local_num_target_samples:] = self.dynamics.coord_to_input(torch.cat((torch.zeros(local_num_target_samples, 1), target_state_samples), dim=-1))[:, 1:self.dynamics.state_dim+1]

        if self.pretrain:
            # only sample in time around the initial condition
            times = torch.full((local_numpoints, 1), self.tMin)
        else:
            # slowly grow time values from start time
            times = self.tMin + torch.empty(local_numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.counter_end), generator=generator)
            # make sure we always have training samples at the initial time
            if local_num_src_samples > 0:
                times[-local_num_src_samples:, 0] = self.tMin
        model_coords = torch.cat((times, model_states), dim=1)        
        if self.dynamics.input_dim > self.dynamics.state_dim + 1: # temporary workaround for having to deal with dynamics classes for parametrized models with extra inputs
            model_coords = torch.cat((model_coords, torch.zeros(local_numpoints, self.dynamics.input_dim - self.dynamics.state_dim - 1)), dim=1)      

        boundary_values = self.dynamics.boundary_fn(self.dynamics.input_to_coord(model_coords)[..., 1:])
        if self.dynamics.loss_type == 'brat_hjivi':
            reach_values = self.dynamics.reach_fn(self.dynamics.input_to_coord(model_coords)[..., 1:])
            avoid_values = self.dynamics.avoid_fn(self.dynamics.input_to_coord(model_coords)[..., 1:])
        
        if self.pretrain:
            dirichlet_masks = torch.ones(model_coords.shape[0]) > 0
        else:
            # only enforce initial conditions around self.tMin
            dirichlet_masks = (model_coords[:, 0] == self.tMin)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.counter_end:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        gt = {'boundary_values': boundary_values, 'dirichlet_masks': dirichlet_masks}
        if self.use_vhat_guidance:
            real_coords = self.dynamics.input_to_coord(model_coords)
            gt.update(self.artifact_guidance.query(real_coords))

        if self.dynamics.loss_type == 'brt_hjivi':
            return {'model_coords': model_coords}, gt
        elif self.dynamics.loss_type == 'brat_hjivi':
            gt.update({'reach_values': reach_values, 'avoid_values': avoid_values})
            return {'model_coords': model_coords}, gt
        else:
            raise NotImplementedError
