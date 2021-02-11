import torch

class DGibbs:
    def __init__(self, distribution, v_coeffs=None):
        self.dist = distribution
        self.dims = self.dist.dims
        self.v_coeffs = v_coeffs
        
    def initialize(self, state0):
        self.state = state0.clone()
        self.x = self.state.clone()
        self.v = torch.ones_like(self.state)
        if self.v_coeffs is not None:
            self.v = self.v_coeffs
        self.v = torch.log(self.v)-self.dist.log_cond_probs(self.state.unsqueeze(0))
        self.v = torch.exp(self.v)
        self.dist_to_border = torch.ones_like(self.v)
        
    def iterate(self):
        # evaluate weight of the current state
        time_to_border = self.dist_to_border/self.v
        iterate_time = torch.min(time_to_border)
        change_dim = torch.argmin(time_to_border)
        output_state = self.state.clone()
        output_log_w = iterate_time
        # update coordinates
        self.x = (self.x + iterate_time*self.v) % self.dims
        self.x = (self.state[change_dim]+1) % self.dims[change_dim]
        self.dist_to_border = self.dist_to_border - iterate_time*self.v
        self.dist_to_border[change_dim] = 1.0
        # update state
        self.state[change_dim] = (self.state[change_dim]+1) % self.dims[change_dim]
        self.v = torch.ones_like(self.state)
        if self.v_coeffs is not None:
            self.v = self.v_coeffs
        self.v = torch.log(self.v)-self.dist.log_cond_probs(self.state.unsqueeze(0))
        self.v = torch.exp(self.v)
        return output_state, output_log_w
    
    def iterate_n(self, n):
        assert n > 0
        for _ in range(n-1):
            self.iterate()
        return self.iterate()
    
    
class Gibbs:
    def __init__(self, distribution):
        self.dist = distribution
        self.dims = self.dist.dims
        self.change_dim = 0
        
    def initialize(self, state0):
        self.state = state0.clone()
        self.change_dim = 0
        
    def iterate(self):
        # update the current dimension
        logits = self.dist.cond_dist(self.state.unsqueeze(0), self.change_dim)
        cond_dist = torch.distributions.categorical.Categorical(logits=logits)
        self.state[self.change_dim] = cond_dist.sample()
        self.change_dim = (self.change_dim + 1) % self.state.shape[0]
        return self.state, torch.tensor(1.0)
        
    def iterate_n(self, n):
        assert n > 0
        for _ in range(n-1):
            self.iterate()
        return self.iterate()

# class DGibbs:
#     def __init__(self, distribution, v_coeffs=None):
#         assert hasattr(distribution, 'log_prob')
#         assert hasattr(distribution, 'dims')
#         assert hasattr(distribution, 'cond_probs')
#         self.dist = distribution
#         self.device = distribution.device
#         self.dims = self.dist.dims
#         self.state = self.dist.get_state().clone().flatten()
#         self.x = self.state.clone()
#         self.v_coeffs = v_coeffs
#         if self.v_coeffs is None:
#             self.v = torch.ones_like(self.state).to(self.device)
#         else:
#             self.v = self.v_coeffs.to(self.device)
#         self.v = self.v/self.dist.get_cond_prob().flatten()
#         self.dist_to_border = torch.ones_like(self.v)
#         self.samples = []
#         self.weights = []
#         self.trajectory = [self.x.clone().cpu()]
        
#     def iterate(self):
#         # evaluate time
#         time_until_border = self.dist_to_border/self.v
#         iterate_time = torch.min(time_until_border)
#         change_dim = torch.argmin(time_until_border)
#         # update coordinates
#         self.x = (self.x + self.v*iterate_time) % self.dims
#         self.x[change_dim] = (self.state[change_dim]+1) % self.dims[change_dim]
#         self.dist_to_border = self.dist_to_border - self.v*iterate_time
#         self.dist_to_border[change_dim] = 1.0
#         # put into samples
#         self.samples.append(self.state.clone().cpu())
#         self.weights.append(iterate_time.cpu())
#         self.trajectory.append(self.x.clone().cpu())
#         # update state
#         self.state[change_dim] = (self.state[change_dim]+1) % self.dims[change_dim]
#         self.dist.set_state(self.state.reshape(self.dist.dims.shape).clone())
#         if self.v_coeffs is None:
#             self.v = torch.ones_like(self.state).to(self.device)
#         else:
#             self.v = self.v_coeffs.to(self.device)
#         self.v = self.v/self.dist.get_cond_prob().flatten()
        
# class Gibbs:
#     def __init__(self, distribution):
#         assert hasattr(distribution, 'get_log_prob')
#         assert hasattr(distribution, 'get_state')
#         assert hasattr(distribution, 'set_state')
#         assert hasattr(distribution, 'dims')
#         assert hasattr(distribution, 'get_cond_prob')
#         assert hasattr(distribution, 'get_cond_dist')
#         self.dist = distribution
#         self.device = self.dist.device
#         self.dims = self.dist.dims.flatten()
#         self.state = self.dist.get_state().clone().flatten()
#         self.samples = []
#         self.weights = []
#         self.change_dim = 0
        
#     def iterate(self):
#         # update the current dimension
#         cond_dist = self.dist.get_cond_dist(self.change_dim)
#         cond_dist = torch.distributions.categorical.Categorical(cond_dist.flatten())
#         self.state[self.change_dim] = cond_dist.sample()
#         # put into samples
#         self.samples.append(self.state.clone())
#         self.weights.append(torch.tensor(1.0))
#         # update state
#         self.dist.set_state(self.state.reshape(self.dist.dims.shape).clone())
#         self.change_dim = (self.change_dim + 1) % self.state.shape[0]


class DGibbsBatch:
    def __init__(self, distribution, v_coeffs=None):
        assert hasattr(distribution, 'get_log_prob')
        assert hasattr(distribution, 'get_state')
        assert hasattr(distribution, 'set_state')
        assert hasattr(distribution, 'dims')
        assert hasattr(distribution, 'get_cond_prob')
        assert hasattr(distribution, 'get_cond_dist')
        self.dist = distribution
        self.device = self.dist.device
        self.batch_size = self.dist.batch_size
        self.batch_ids = torch.arange(self.batch_size).long().to(self.device)
        self.dims = self.dist.dims.flatten().repeat([self.batch_size,1]).to(self.device)
        self.state = self.dist.get_state().clone().reshape([self.batch_size,-1])
        self.x = self.state.clone().double()
        self.v = torch.ones_like(self.state).to(self.device)
        self.v = self.v/self.dist.get_cond_prob().reshape([self.batch_size,-1])
        self.v_coeffs = v_coeffs
        if self.v_coeffs is not None:
            self.v_coeffs = self.v_coeffs.to(self.device)
            self.v = self.v*self.v_coeffs.reshape([1,-1])
        self.dist_to_border = torch.ones_like(self.state)
        self.samples = []
        self.weights = []
        self.trajectory = [self.x.clone().cpu()]
        self.iter_num = 1
        
    def iterate(self):
        # evaluate time and decide what to update
        time_until_border = self.dist_to_border/self.v
        iterate_times, change_dims = torch.min(time_until_border, dim=1)
        # update coordinates
        self.x = self.x + self.v*iterate_times.reshape([-1,1])
        # update coordinates to be sure that the changed coordiantes now integer
        self.x[self.batch_ids,change_dims] = self.state[self.batch_ids,change_dims].double()+1.0
        self.x[self.batch_ids,change_dims] %= self.dims[self.batch_ids,change_dims]
        self.dist_to_border = self.dist_to_border - self.v*iterate_times.reshape([-1,1])
        self.dist_to_border[self.batch_ids,change_dims] = 1.0
        # write down old state and the time we've spent in it
        self.samples.append(self.state.clone().cpu())
        self.weights.append(iterate_times.cpu())
        self.trajectory.append(self.x.clone().cpu())
        # update state along updated dims
        self.state[self.batch_ids,change_dims] += 1
        self.state[self.batch_ids,change_dims] %= self.dims[self.batch_ids,change_dims]
        self.dist.set_state(self.state.clone().reshape([self.batch_size,*self.dist.dims.shape]))
        self.v = torch.ones_like(self.state).to(self.device)
        self.v = self.v/self.dist.get_cond_prob().reshape([self.batch_size,-1])
        if self.v_coeffs is not None:
            self.v = self.v*self.v_coeffs.reshape([1,-1])
        self.iter_num += 1
        
        
class GibbsBatch:
    def __init__(self, distribution):
        assert hasattr(distribution, 'get_log_prob')
        assert hasattr(distribution, 'get_state')
        assert hasattr(distribution, 'set_state')
        assert hasattr(distribution, 'dims')
        assert hasattr(distribution, 'get_cond_prob')
        assert hasattr(distribution, 'get_cond_dist')
        self.dist = distribution
        self.device = self.dist.device
        self.batch_size = self.dist.batch_size
        self.dims = self.dist.dims.flatten().repeat([self.batch_size,1]).to(self.device)
        self.state = self.dist.get_state().clone().reshape([self.batch_size,-1])
        self.samples = []
        self.weights = []
        self.change_dim = 0
        self.iter_num = 1
        
    def iterate(self):
        # update the current dimension
        cond_dist = self.dist.get_cond_dist(self.change_dim)
        cond_dist = torch.distributions.categorical.Categorical(cond_dist)
        self.state[:,self.change_dim] = cond_dist.sample()
        # put into samples
        self.samples.append(self.state.clone())
        self.weights.append(torch.ones(self.batch_size))
        # update state
        self.dist.set_state(self.state.clone().reshape([self.batch_size,*self.dist.dims.shape]))
        self.change_dim = (self.change_dim + 1) % self.dims.shape[1]
        self.iter_num += 1
