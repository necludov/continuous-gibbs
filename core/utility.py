import numpy as np

def generate_primes(until_n):
    all_numbers = np.arange(until_n)
    prime_numbers = []
    for i in range(2,np.ceil(np.sqrt(until_n)).astype(int)):
        prime_numbers.append(i)
        all_numbers[::i] = -1.0
    prime_numbers = np.array(prime_numbers)
    return np.concatenate([prime_numbers, all_numbers[all_numbers > 0][1:]])

def batch_means_ess(x,w):
    """ Estimate the effective sample size as the ratio of the variance
    of the batch means to the variance of the chain. As explained here:
    https://arxiv.org/pdf/1011.0175.pdf. We expect the chain in the format
    Time-Steps, Num-Chains, Dimension (T, M, D) and the weights in the format
    Time-Steps, Num-Chains, Dimension (T, M)"""
    
    T, M, D = x.shape
    weights = w[:,:,np.newaxis].copy()
    weights = weights/np.sum(weights, axis=0, keepdims=True)
    num_batches = int(np.floor(T ** (1 / 3)))
    batch_size = int(np.floor(num_batches ** 2))
    batch_means = []
    for i in range(num_batches):
        batch = x[batch_size * i:batch_size * i + batch_size]
        batch_weights = weights[batch_size * i:batch_size * i + batch_size].copy()
        batch_weights /= np.sum(batch_weights, axis=0, keepdims=True)
        batch_means.append(np.sum(batch*batch_weights, axis=0))
    batch_variance = np.var(np.array(batch_means), axis=0)
    chain_mean = np.sum(x*weights, axis=0, keepdims=True)
    chain_variance = np.sum(weights*(x-chain_mean)**2, axis=0)
    act = batch_size * batch_variance / (chain_variance+1e-20)
    return 1/act
