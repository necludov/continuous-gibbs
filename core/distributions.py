import torch

class IsingModel2D:
    def __init__(self, state, target, beta=1.0, eta=2.1):
        self.h, self.w = target.shape
        self.device = target.device
        self.img = target*2 - 1
        self.spins = state*2 - 1
        self.dims = 2*torch.ones_like(self.spins, dtype=torch.int).to(self.device)
        self.eta = eta
        self.beta = beta
        
    def get_state(self):
        return (self.spins > 0).int()
    
    def set_state(self, state):
        self.spins = (2*state-1).to(self.device)
        
    def get_log_prob(self):
        w,h = self.w, self.h
        energy = -self.beta*torch.sum(self.spins[:,:w-1]*self.spins[:,1:])
        energy += -self.beta*torch.sum(self.spins[:h-1,:]*self.spins[1:,:])
        energy += -self.eta*torch.sum(self.spins*self.img)
        return -energy
        
    def get_cond_prob(self):
        w,h = self.w, self.h
        cond_energy = torch.zeros_like(self.spins).to(self.device)
        cond_energy[:,:w-1] += self.spins[:,1:]
        cond_energy[:,1:] += self.spins[:,:w-1]
        cond_energy[:h-1,:] += self.spins[1:,:]
        cond_energy[1:,:] += self.spins[:h-1,:]
        cond_energy = -self.beta*cond_energy - self.eta*self.img
        cond_log_prob = -cond_energy - torch.logaddexp(cond_energy,-cond_energy)
        cond_prob = torch.exp(cond_log_prob)
        cond_prob = cond_prob*(self.spins > 0) + (1-cond_prob)*(self.spins < 0)
        return cond_prob
    
    def get_cond_dist(self, dim):
        """returns p(x_dim|x_rest)"""


class Discrete2D:
    def __init__(self, probs):
        self.h, self.w = probs.shape
        self.device = probs.device
        self.dims = torch.tensor([self.h, self.w]).long()
        self.probs = probs
        self.state = torch.tensor([0, 0]).long()
        
    def get_state(self):
        return self.state
    
    def set_state(self, state):
        self.state = state
        
    def get_log_prob(self):
        return torch.log(self.probs[self.state[0], self.state[1]])
        
    def get_cond_prob(self):
        joint_prob = self.probs[self.state[0], self.state[1]]
        cond_prob = torch.zeros_like(self.state).float()
        for d in range(len(cond_prob)):
            cond_prob[d] = joint_prob/torch.sum(self.probs, dim=d)[self.state[(d+1) % 2]]
        return cond_prob
    
    def get_cond_dist(self, dim):
        """returns p(x_dim|x_rest)"""
        rest_dim = (dim+1) % 2
        cond_dist = torch.index_select(self.probs, dim=rest_dim, index=self.state[rest_dim])
        cond_dist /= torch.sum(cond_dist)
        return cond_dist
