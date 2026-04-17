import torch
from torch.utils.data import Dataset

# uses model input and real boundary fn
class ReachabilityDataset(Dataset):
    def __init__(self, dynamics, numpoints, pretrain, pretrain_iters, tMin, tMax, counter_start, counter_end, num_src_samples, num_target_samples, seed=0, rank=0, world_size=1):
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

        if self.dynamics.loss_type == 'brt_hjivi':
            return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'dirichlet_masks': dirichlet_masks}
        elif self.dynamics.loss_type == 'brat_hjivi':
            return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'reach_values': reach_values, 'avoid_values': avoid_values, 'dirichlet_masks': dirichlet_masks}
        else:
            raise NotImplementedError
