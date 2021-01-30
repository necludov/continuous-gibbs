import torch

class DGibbs:
    def __init__(self, distribution, v_coeffs=None):
        assert hasattr(distribution, 'get_log_prob')
        assert hasattr(distribution, 'get_state')
        assert hasattr(distribution, 'set_state')
        assert hasattr(distribution, 'dims')
        assert hasattr(distribution, 'get_cond_prob')
        assert hasattr(distribution, 'get_cond_dist')
        self.dist = distribution
        self.device = distribution.device
        self.dims = self.dist.dims.flatten()
        self.state = self.dist.get_state().clone().flatten()
        self.x = self.state.clone()
        self.v_coeffs = v_coeffs
        if self.v_coeffs is None:
            self.v = torch.ones_like(self.state).to(self.device)
        else:
            self.v = self.v_coeffs.to(self.device)
        self.v = self.v/self.dist.get_cond_prob().flatten()
        self.dist_to_border = torch.ones_like(self.v)
        self.samples = []
        self.weights = []
        self.trajectory = [self.state.clone()]
        
    def iterate(self):
        # evaluate time
        time_until_border = self.dist_to_border/self.v
        iterate_time = torch.min(time_until_border)
        change_dim = torch.argmin(time_until_border)
        # update coordinates
        self.x = (self.x + self.v*iterate_time) % self.dims
        self.state[change_dim] = (self.state[change_dim]+1) % self.dims[change_dim]
        self.dist_to_border = self.dist_to_border - self.v*iterate_time
        self.dist_to_border[change_dim] = 1.0
        # put into samples
        self.samples.append(self.state.clone().cpu())
        self.weights.append(iterate_time.cpu())
        self.trajectory.append(self.x.clone().cpu())
        # update state
        self.dist.set_state(self.state.reshape(self.dist.dims.shape).clone())
        if self.v_coeffs is None:
            self.v = torch.ones_like(self.state).to(self.device)
        else:
            self.v = self.v_coeffs.to(self.device)
        self.v = self.v/self.dist.get_cond_prob().flatten()
        
class Gibbs:
    def __init__(self, distribution):
        assert hasattr(distribution, 'get_log_prob')
        assert hasattr(distribution, 'get_state')
        assert hasattr(distribution, 'set_state')
        assert hasattr(distribution, 'dims')
        assert hasattr(distribution, 'get_cond_prob')
        assert hasattr(distribution, 'get_cond_dist')
        self.dist = distribution
        self.device = self.dist.device
        self.dims = self.dist.dims.flatten()
        self.state = self.dist.get_state().clone().flatten()
        self.samples = []
        self.weights = []
        self.change_dim = 0
        
    def iterate(self):
        # update the current dimension
        cond_dist = self.dist.get_cond_dist(self.change_dim)
        cond_dist = torch.distributions.categorical.Categorical(cond_dist.flatten())
        self.state[self.change_dim] = cond_dist.sample()
        # put into samples
        self.samples.append(self.state.clone())
        self.weights.append(torch.tensor(1.0))
        # update state
        self.dist.set_state(self.state.reshape(self.dist.dims.shape).clone())
        self.change_dim = (self.change_dim + 1) % self.state.shape[0]
