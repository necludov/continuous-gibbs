def generate_primes(until_n):
    all_numbers = np.arange(until_n)
    prime_numbers = []
    for i in range(2,np.ceil(np.sqrt(until_n)).astype(int)):
        prime_numbers.append(i)
        all_numbers[::i] = -1.0
    prime_numbers = np.array(prime_numbers)
    return np.concatenate([prime_numbers, all_numbers[all_numbers > 0][1:]])

def batch_means_ess(x):
    """ Estimate the effective sample size as the ratio of the variance
    of the batch means to the variance of the chain. As explained here:
    https://arxiv.org/pdf/1011.0175.pdf. We expect the chain in the format
    Time-Steps, Num-Chains, Dimension (T, M, D) """

    x = np.transpose(x, [1, 0, 2])
    T, M, D = x.shape
    num_batches = int(np.floor(T ** (1 / 3)))
    batch_size = int(np.floor(num_batches ** 2))
    batch_means = []
    for i in range(num_batches):
        batch = x[batch_size * i:batch_size * i + batch_size]
        batch_means.append(np.mean(batch, axis=0))
    batch_variance = np.var(np.array(batch_means), axis=0)
    chain_variance = np.var(x, axis=0)

    act = batch_size * batch_variance / (chain_variance + 1e-20)

    return 1 / act
