import numpy as np

class DGibbs:
    def __init__(self, distribution, v_coeffs=None):
        self.dist = distribution
        self.dims = self.dist.dims
        self.v_coeffs = v_coeffs
        
    def initialize(self, state0):
        self.state = state0.copy()
        self.x = self.state.copy()
        self.v = np.ones_like(self.state)
        if self.v_coeffs is not None:
            self.v = self.v_coeffs
        self.v = self.v/self.dist.get_cond_probs(self.state)
        self.dist_to_border = np.ones_like(self.v)
        
    def iterate(self):
        # evaluate weight of the current state
        time_until_border = self.dist_to_border/self.v
        iterate_time = np.min(time_until_border)
        change_dim = np.argmin(time_until_border)
        output_state = self.state.copy()
        output_weight = iterate_time
        # update coordinates
        self.x = (self.x + self.v*iterate_time) % self.dims
        self.x = (self.state[change_dim]+1) % self.dims[change_dim]
        self.dist_to_border = self.dist_to_border - self.v*iterate_time
        self.dist_to_border[change_dim] = 1.0
        # update state
        self.state[change_dim] = (self.state[change_dim]+1) % self.dims[change_dim]
        self.v = np.ones_like(self.state)
        if self.v_coeffs is not None:
            self.v = self.v_coeffs
        self.v = self.v/self.dist.get_cond_probs(self.state)
        return output_state, output_weight
    
    def iterate_n(self, n):
        assert n > 0
        for _ in range(n-1):
            self.iterate()
        return self.iterate()
        
class Gibbs:
    def __init__(self, distribution):
        self.dist = distribution
        self.dims = self.dist.dims.flatten()
        
    def initialize(self, state0):
        self.state = state0.copy().flatten()
        self.change_dim = 0

    def iterate(self):
        output_state = self.state.copy()
        cond_prob = self.dist.get_cond_probs(self.state.reshape(self.dist.shape)).flatten()
        # flip the state with 1-its probability
        u = np.random.uniform()
        if u > cond_prob[self.change_dim]:
            self.state[self.change_dim] = (self.state[self.change_dim]+1) % self.dims[self.change_dim]
        # update state
        self.change_dim = (self.change_dim + 1) % len(self.dims)
        return output_state, 1.0
    
    def iterate_n(self, n):
        assert n > 0
        for _ in range(n-1):
            self.iterate()
        return self.iterate()
