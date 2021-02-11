import numpy as np
import math
import time
from copy import deepcopy
import warnings

class DHMCSampler(object):

    def __init__(self, f, f_update, n_disc, n_param, scale=None):
        if scale is None:
            scale = np.ones(n_param)
        # Set the scale of p to be inversely proportional to the scale of theta.
        self.M = 1 / np.concatenate((scale[:-n_disc] ** 2, scale[-n_disc:]))
        self.n_param = n_param
        self.n_disc = n_disc
        self.f = f
        self.f_update = f_update

    # Utility function to check that the returned values of f and f_updates are
    # all consistent.

    def test_cont_grad(self, theta0, sd=1, atol=None, rtol=.01, dx=10**-6, n_test=10):
        """
        Wrapper function for test_grad to check the returned gradient values
        (with respect to the continuous parameters). The gradients are
        evaluated at n_test randomly generated points around theta0.
        """

        if atol is None:
            atol = dx

        for i in range(n_test):
            theta = theta0.copy()
            theta[-self.n_disc:] += sd * np.random.randn(self.n_disc)
            def f_test(theta_cont):
                logp, grad, aux \
                    = self.f(np.concatenate((theta_cont, theta0[-self.n_disc:])))
                grad = grad[:-self.n_disc]
                return logp, grad

            test_pass, theta_cont, grad, grad_est \
                = self.test_grad(f_test, theta[:-self.n_disc], atol, rtol, dx)

            if not test_pass:
                warnings.warn(
                    'Test failed: the returned gradient value does not agree with ' +
                    'the centered difference approximation within the tolerance level.',
                    RuntimeWarning
                )
                break

        if test_pass:
            print('Test passed! The computed gradient seems to be correct.')

        return test_pass, theta_cont, grad, grad_est

    def test_grad(self, f, x, atol, rtol, dx):
        """Compare the computed gradient to a centered finite difference approximation. """
        x = np.array(x, ndmin=1)
        grad_est = np.zeros(len(x))
        for i in range(len(x)):
            x_minus = x.copy()
            x_minus[i] -= dx
            x_plus = x.copy()
            x_plus[i] += dx
            f_minus, _ = f(x_minus)
            f_plus, _ = f(x_plus)
            grad_est[i] = (f_plus - f_minus) / (2 * dx)

        _, grad = f(x)
        test_pass = np.allclose(grad, grad_est, atol=atol, rtol=rtol)

        return test_pass, x, grad, grad_est

    def test_update(self, theta0, sd, n_test=10, atol=10 ** -3, rtol=10 ** -3):
        """
        Check that the outputs of 'f' and 'f_update' functions are consistent
        by comparing the values logp differences computed by the both functions.
        """

        test_pass = True
        for i in range(n_test):
            index = np.random.randint(self.n_param - self.n_disc, self.n_param)
            theta = theta0 + .1 * sd * np.random.randn(len(theta0))
            dtheta = sd * np.random.randn(1)
            theta_new = theta.copy()
            theta_new[index] += dtheta
            logp_prev, _, aux = self.f(theta)
            logp_curr, _, _ = self.f(theta_new)
            logp_diff, _ = self.f_update(theta, dtheta, index, aux)
            both_inf = math.isinf(logp_diff) \
                       and math.isinf(logp_curr - logp_prev)
            if not both_inf:
                abs_err = abs(logp_diff - (logp_curr - logp_prev))
                if logp_curr == logp_prev:
                    rel_err = 0
                else:
                    rel_err = abs_err / abs(logp_curr - logp_prev)
                if abs_err > atol or rel_err > rtol:
                    test_pass = False
                    break

        if test_pass:
            print('Test passed! The logp differences agree.')
        else:
            warnings.warn(
                'Test failed: the outputs of f and f_update are not consistent.' +
                'the logp differences do not agree.',
                RuntimeWarning
            )
        return test_pass, theta, logp_diff, logp_curr - logp_prev

    def gauss_laplace_leapfrog(
            self, f, f_update, dt, theta0, p0, logp, grad, aux, n_disc=0, M=None):
        """
        One numerical integration step of the DHMC integrator for a mixed
        Gaussian and Laplace momentum.

        Params
        ------
        f: function(theta, req_grad)
          Returns the log probability and, if req_grad is True, its gradient.
          The gradient for discrete parameters should be zero.
        f_update: function(theta, dtheta, index, aux)
          Computes the difference in the log probability when theta[index] is
          modified by dtheta. The input 'aux' is whatever the quantity saved from
          the previous call to 'f' or 'f_update' that can be recycled.
        M: column vector
          Represents the diagonal mass matrix
        n_disc: int
          Number of discrete parameters. The parameters theta[:-n_disc] are
          assumed continuous.
        """

        if M is None:
            M = np.ones(len(theta0))
        theta = theta0.copy()
        p = p0.copy()

        p[:-n_disc] = p[:-n_disc] + 0.5 * dt * grad[:-n_disc]
        if n_disc == 0:
            theta = theta + dt * p / M
        else:
            # Update continuous parameters if any.
            if self.n_param != self.n_disc:
                theta[:-n_disc] = theta[:-n_disc] + dt / 2 * p[:-n_disc] / M[:-n_disc]
                logp, _, aux = f(theta, req_grad=False)

            # Update discrete parameters.
            if math.isinf(logp):
                return theta, p, grad, logp, aux
            coord_order = len(theta) - n_disc + np.random.permutation(n_disc)
#             for index in coord_order:
            theta, p, logp, aux \
                = self.update_coordwise(f_update, aux, self.current_dim, theta, p, M, dt, logp)

            theta[:-n_disc] = theta[:-n_disc] + dt / 2 * p[:-n_disc] / M[:-n_disc]

        if self.n_param != self.n_disc:
            logp, grad, aux = f(theta)
            p[:-n_disc] = p[:-n_disc] + 0.5 * dt * grad[:-n_disc]

        return theta, p, grad, logp, aux

    def update_coordwise(self, f_update, aux, index, theta, p, M, dt, logp):
        p_sign = math.copysign(1.0, p[index])
        dtheta = p_sign / M[index] * dt
        logp_diff, aux_new = f_update(theta, dtheta, index, aux)
        dU = - logp_diff
        if abs(p[index]) / M[index] > dU:
            p[index] += - p_sign * M[index] * dU
            theta[index] += dtheta
            logp += logp_diff
            aux = aux_new
        else:
            p[index] = - p[index]
        return theta, p, logp, aux

    def hmc(self, epsilon, n_step, theta0, logp0, grad0, aux0):
        """
        Proposal scheme basically identical to the standard HMC. The code is
        however written so that one can use any kinetic energy along with a
        corresponding reversible and volume-preserving integrator.
        """

        p = self.random_momentum()
        joint0 = - self.compute_hamiltonian(logp0, p)

        pathlen = 0
        theta, grad, aux = theta0.copy(), grad0.copy(), deepcopy(aux0)
        theta, p, grad, logp, aux \
            = self.integrator(epsilon, theta, p, logp0, grad, aux)
        pathlen += 1
        for i in range(1, n_step):
            if math.isinf(logp):
                break
            theta, p, grad, logp, aux \
                = self.integrator(epsilon, theta, p, logp, grad, aux)
            pathlen += 1

        if math.isinf(logp):
            acceptprob = 0.
        else:
            joint = - self.compute_hamiltonian(logp, p)
            acceptprob = min(1, np.exp(joint - joint0))

        if acceptprob < np.random.rand():
            theta = theta0
            logp = logp0
            grad = grad0
            aux = aux0
            
        return theta, logp, grad, aux, acceptprob, pathlen

    # Integrator and kinetic energy functions for the proposal scheme. The
    # class allows any reversible dynamics based samplers by changing the
    # 'integrator', 'random_momentum', and 'compute_hamiltonian' functions.
    def integrator(self, dt, theta0, p0, logp, grad, aux):
        return self.gauss_laplace_leapfrog(
            self.f, self.f_update, dt, theta0, p0, logp, grad, aux, self.n_disc, self.M
        )

    def random_momentum(self):
        p_cont = np.sqrt(self.M[:-self.n_disc]) \
            * np.random.normal(size = self.n_param - self.n_disc)
        p_disc = self.M[-self.n_disc:] * np.random.laplace(size=self.n_disc)
        return np.concatenate((p_cont, p_disc))

    def compute_hamiltonian(self, logp, p):
        return - logp \
            + np.sum(p[:-self.n_disc] ** 2 / 2 / self.M[:-self.n_disc]) \
            + np.sum(np.abs(p[-self.n_disc:]) / self.M[-self.n_disc:])

    def run_sampler(self, theta0, dt_range, nstep_range, n_burnin, n_sample, seed=None, n_update=10):
        """Run DHMC and return samples and some additional info."""

        np.random.seed(seed)
        self.current_dim = 0

        # Run HMC.
        theta = theta0
        n_per_update = math.ceil((n_burnin + n_sample) / n_update)
        pathlen_ave = 0
        samples = np.zeros((n_sample + n_burnin, len(theta)))
        logp_samples = np.zeros(n_sample + n_burnin)
        accept_prob = np.zeros(n_sample + n_burnin)

        tic = time.process_time()  # Start clock
        logp, grad, aux = self.f(theta)
        for i in range(n_sample + n_burnin):
            dt = np.random.uniform(dt_range[0], dt_range[1])
            nstep = np.random.randint(nstep_range[0], nstep_range[1] + 1)
            theta, logp, grad, aux, accept_prob[i], pathlen \
                = self.hmc(dt, nstep, theta, logp, grad, aux)
            self.current_dim = (self.current_dim+1) % self.n_param
            pathlen_ave = i / (i + 1) * pathlen_ave + 1 / (i + 1) * pathlen
            samples[i, :] = theta
            logp_samples[i] = logp
#             if (i + 1) % n_per_update == 0:
#                 print('{:d} iterations have been completed.'.format(i + 1))

        toc = time.process_time()
        time_elapsed = toc - tic

        return samples, logp_samples, time_elapsed
    
    
def f(theta, req_grad=False):
    if (np.sum(theta > 2.0) > 0.0) or (np.sum(theta < 0.0) > 0.0):
        return -float('inf'), float('nan'), None
    l = 28
    spins = 2.*np.floor(theta.reshape([l,l]).copy())-1.
    energy = 0.0
    energy += np.sum(spins[:,:l-1]*spins[:,1:])
    energy += np.sum(spins[:l-1,:]*spins[1:,:])
    return -energy, np.zeros(1), None

def f_update(theta, dtheta, k, aux):
    l = 28
    def log_prob(theta):
        spins = 2.*np.floor(theta.reshape([l,l]).copy())-1.
        energy = 0.0
        energy += np.sum(spins[:,:l-1]*spins[:,1:])
        energy += np.sum(spins[:l-1,:]*spins[1:,:])
        return -energy
    new_theta = theta.copy()
    new_theta[k] += dtheta
    if (new_theta[k] < 0.0) or (new_theta[k] > 2.0):
        return -float('inf'), np.zeros(1)
    i, j = k % l, k // l
    return log_prob(new_theta) - log_prob(theta), np.zeros(1)


def get_f_mnist(img):
    h,w = 28,28
    beta=1.0
    eta=2.1
    def f(theta, req_grad=False):
        if (np.sum(theta > 2.0) > 0.0) or (np.sum(theta < 0.0) > 0.0):
            return -float('inf'), float('nan'), None
        spins = 2.*np.floor(theta.reshape([h,w]).copy())-1.
        energy = -beta*np.sum(spins[:,:w-1]*spins[:,1:])
        energy += -beta*np.sum(spins[:h-1,:]*spins[1:,:])
        energy += -eta*np.sum(spins*img)
        return -energy, np.zeros(1), None
    return f

def get_f_update_mnist(img):
    h,w = 28,28
    beta=1.0
    eta=2.1
    def f_update(theta, dtheta, k, aux):
        def log_prob(theta):
            spins = 2.*np.floor(theta.reshape([h,w]).copy())-1.
            energy = -beta*np.sum(spins[:,:w-1]*spins[:,1:])
            energy += -beta*np.sum(spins[:h-1,:]*spins[1:,:])
            energy += -eta*np.sum(spins*img)
            return -energy
        new_theta = theta.copy()
        new_theta[k] += dtheta
        if (new_theta[k] < 0.0) or (new_theta[k] > 2.0):
            return -float('inf'), np.zeros(1)
        i, j = k % w, k // w
        return log_prob(new_theta) - log_prob(theta), np.zeros(1)
    return f_update
