import pandas as pd
import numpy as np
import warnings
from random import choices
from statsmodels.regression.linear_model import OLS
from scipy.stats import weibull_min, norm
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

def logsumexp(lp):
    y = np.max(lp)
    return np.log(np.sum(np.exp(lp-y), keepdims=True)) + y


def ppoints(n, a=None):
    if len(n) > 1:
        n = len(n)

    if a is None:
        if n <= 10.:
            a = 3/8
        else:
            a = 1/2

    if n > 0:
        return (np.linspace(1, n, n)-a) / (n + 1 - 2*a)
    else:
        return np.empty(shape=(), dtype=float)


def get_wn_startvals(x, nbins=None):
    if nbins is None:
        nbins = np.round(len(x) / 6)

    bins = np.append(np.zeros(1), np.linspace(
        np.min(x), np.max(x), num=int(nbins)))
    x_binned = pd.cut(x, bins)

    modebin_idx = np.argmax(x_binned.value_counts())
    modebin_id = x_binned.value_counts().axes[0][modebin_idx]

    thresh = (modebin_id.right - modebin_id.left) / 2 + modebin_id.left

    init_mu = np.mean(x[x > thresh])
    init_sigma = np.std(x[x > thresh], ddof=1)

    ws = np.sort(x[x <= thresh])
    Fh = ppoints(ws)

    init_shape = OLS(endog=np.log(-np.log(1-Fh)),
                     exog=np.c_[np.ones(shape=np.shape(ws)), np.log(ws)]).\
        fit().params[1]
    init_scale = np.quantile(x[x <= thresh], 0.632)
    init_prop = np.array([np.mean(x < thresh), 1 - np.mean(x < thresh)])

    return {
        'init_mu': init_mu,
        'init_sigma': init_sigma,
        'init_shape': init_shape,
        'init_scale': init_scale,
        'init_prop': init_prop
    }


def loglikelihood(x, mu, sigma, shape, scale, prop):
    ll_weib = prop[0] * np.nan_to_num(weibull_min.pdf(x,
                                      shape, loc=0, scale=scale), nan=0.0, copy=False)
    ll_norm = prop[1] * norm.pdf(x, loc=mu, scale=sigma)

    return sum(np.log(ll_weib + ll_norm))


def shape_nll(par, x, mu, sigma, scale, prop):
    shape = np.exp(par)

    return -loglikelihood(x, mu, sigma, shape, scale, prop)


def scale_nll(par, x, mu, sigma, shape, prop):
    scale = np.exp(par)
    return -loglikelihood(x, mu, sigma, shape, scale, prop)


def weibull_normal_em(
    x,
    init_mu,
    init_sigma,
    init_shape,
    init_scale,
    init_prop,
    itermax=30,
    tol=1E-6
):
    n = len(x)
    shape = init_shape
    scale = init_scale
    prop = init_prop
    mu = init_mu
    sigma = init_sigma
    ll_old = float("inf")

    # E step -----------------
    z = np.empty((n, 2), dtype=np.float64)
    z[:, 0] = np.log(prop[0]) + weibull_min.logpdf(x,
                                                   shape, loc=0, scale=scale)
    z[:, 1] = np.log(prop[1]) + norm.logpdf(x, loc=mu, scale=sigma)

    norm_factor = np.apply_along_axis(logsumexp, 1, z)
    z -= norm_factor

    # Run EM ----------------------
    for step in range(itermax):
        if step % 4 == 0:
            print(f'iteration {step + 1}')

        # M step ---------------------------------------------------------------------

        # Mixing weights---------
        expz = np.exp(z)
        prop = np.mean(expz, axis=0)

        # Shape ------------
        opt_shape = minimize_scalar(fun=shape_nll, bounds=(np.log(.00001), np.log(
            10000)), args=(x, mu, sigma, scale, prop), method='bounded', tol=1e-8)

        shape = np.exp(opt_shape.x)

        # Scale ---------
        opt_scale = minimize_scalar(fun=scale_nll, bounds=(np.log(.00001), np.log(
            10000)), args=(x, mu, sigma, shape, prop), method='bounded', tol=1e-8)

        scale = np.exp(opt_scale.x)

        # Mu -----------------------------------
        mu = sum(expz[:, 1] * x) / sum(expz[:, 1])

        # Sigma --------------------------------
        sigma = np.sqrt(sum(expz[:, 1] * pow(x - mu, 2)) / sum(expz[:, 1]))

        # Check convergence --------------------------------------------------------
        ll_new = loglikelihood(x, mu, sigma, shape, scale, prop)

        if (step == itermax) & (abs(ll_new - ll_old) > tol):
            warnings.warn(
                "EM finished without reaching convergence.", UserWarning)

        if abs(ll_new - ll_old) < tol:
            break
        else:
            ll_old = ll_new

        # E step ---------------------------------------------------------------------
        z[:, 0] = np.log(prop[0]) + weibull_min.logpdf(x,
                                                       shape, loc=0, scale=scale)

        z[:, 1] = np.log(prop[1]) + norm.logpdf(x, loc=mu, scale=sigma)

        norm_factor = np.apply_along_axis(logsumexp, 1, z)
        z -= norm_factor

    return {
        'x': x,
        'parameters': {
            'prop': prop,
            'mu': mu,
            'sigma': sigma,
            'shape': shape,
            'scale': scale
        },
        'loglikelihood': ll_old,
        'z': np.exp(z),
        'iters': step + 1
    }

def weibnormix(n: int, object):
    params = object.get('parameters')
    cl = choices([0,1],k=n, weights = params['prop'])
    cl.sort()
    
    n1 = np.count_nonzero(cl)
    n0 = n - n1
    
    sim_weib = weibull_min.rvs(params['shape'], loc = 0, scale = params['scale'], size = (n0,))
    sim_norm = norm.rvs(params['mu'], scale = params['sigma'], size = (n1,))
    
    return pd.DataFrame(
        {
            'x' : np.r_[sim_weib, sim_norm],
            'cl' : cl
        }
    )
    
def dweibnormix(x, object):
    params = object.get('parameters')
    
    dens_weib = weibull_min.pdf(x,  params['shape'], loc = 0, scale = params['scale'])
    dens_norm = norm.pdf(x, params['mu'], scale = params['sigma'])
    
    return dens_weib * params['prop'][0] + dens_norm * params['prop'][1]
    
  
def pweibnormix(q, object):
    params = object.get('parameters')
    
    cum_weib = weibull_min.cdf(q, params['shape'], loc = 0, scale = params['scale'])
    cum_norm = norm.cdf(q, params['mu'], scale = params['sigma'])
    
    return cum_weib * params['prop'][0] + cum_norm * params['prop'][1]

def qweibnormix(p, object):
    params = object.get('parameters')
    bounds = [0, np.max(object['x']) * 10]
    
    x = np.linspace(bounds[0], bounds[1], 10000)
    y = pweibnormix(x, object)
    
    myfun = interp1d(y,x,kind = 'linear')
    out = myfun(p)
    out[p==0] = -float("inf")
    out[p==1] = float("inf")
    
    return out