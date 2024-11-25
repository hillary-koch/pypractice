import numpy as np
from random import choices
from scipy.stats import norm, weibull_min, expon
from scipy.interpolate import interp1d
from pypractice.logsumexp import logsumexp


class MixtureDistribution():
    """
    Mixture distribution
    __valid_distns: distributions for which methods have been implemented.
        The keys in the `components` dictionary must be found in this array. Subclasses
        will update appropriately to be a subset of this array.        
    """

    __supported_distns = ['norm', 'weibull_min', 'expon']

    def __init__(self, components: dict[str, int]):
        """
        k: number of mixture components
        __components: dict{distribution_name: number of components for that distribution}
        __params: dict{
            prop: (k,)-dimensional array with mixing proportions for each component
        }
        """
        self._check_components(components)
        self.__k = self._count_components(components)
        self.__components = components

        # """
        #     (alternatively, consider specifying None for loc as a keyword argument)
        #     Positional arguments:
        #     weibull_min: c, loc (without specifying loc, it defaults to 0)
        #     norm: loc
        #     expon: loc
        # """
        self.__params = {
            'prop': np.ones(shape=(self.k,))/self.k,
            'positional': [1 if key == "weibull_min" else 0 for key in
                           [key for key in self.components.keys() for value in range(self.components[key])]],
            'scale': np.ones(shape=(self.k,))
        }

        # Distribution support ---------
        self.__a = min([getattr(globals()[distr], 'a')
                       for distr in self.components.keys()])
        self.__b = max([getattr(globals()[distr], 'b')
                       for distr in self.components.keys()])

        # Initialize ll as -inf; without x this is meaningless
        self.__ll = float("-inf")

    # Properties immutable to the user ------------------------------------------------
    @property
    def supported_distns(self):
        return self.__supported_distns

    @property
    def params(self):
        return self.__params
    
    @property
    def k(self):
        return self.__k

    @property
    def components(self):
        return self.__components
    
    @property
    def z(self):
        return self.__z

    @property
    def a(self):
        return self.__a

    @property
    def b(self):
        return self.__b

    @property
    def ll(self):
        return self.__ll

    # Class-specific auxiliary functions ------------------------------------------------
    @staticmethod
    def _count_components(x: dict[str, int]):
        return sum(x.values())

    @staticmethod
    def _do_call(distr, method, *args, **kwargs):
        fun = getattr(globals()[distr], method)
        return fun(*args, **kwargs)

    # Private class methods ------------------------------------------------
    def _check_components(self, x: dict[str, int]):
        if not all([dist in self.supported_distns for dist in x.keys()]):
            raise ValueError(f"Supported components keys are {
                             self.supported_distns}. You provided {[dist for dist in x.keys()]}")

    def _get_component_iterator(self) -> zip:
        return zip(
            [key for key in self.components.keys()
             for value in range(self.components[key])],
            self.params['prop'],
            self.params['positional'],
            self.params['scale']
        )

    # Public class methods ------------------------------------------------
    def add_components(self, components=dict[str, int]):
        self._check_components(components)
        components = {key: self.components.get(
            key, 0) + components.get(key, 0) for key in (self.components.keys() | components.keys())}

        # Re-initialize after changing components
        self.__init__(components)

    def logpdf(self, x):
        """
        Returns the log density of x in each component
        """
        return np.transpose([np.log(prop) + self._do_call(distr, "logpdf", x, positional, scale=scale) for
                distr, prop, positional, scale in self._get_component_iterator()])
        
    def loglikelihood(self, x):
        component_likelihoods = np.exp(self.logpdf(x))
        
        return np.log(np.sum(component_likelihoods))
        
    
    def pdf(self, x):
        """
        Returns the density of x as an array the same shape as x
        """
        pdf_component = [prop * self._do_call(distr, "pdf", x, positional, scale=scale) for
                         distr, prop, positional, scale in self._get_component_iterator()]
        return np.sum(pdf_component, axis=0)

    def cdf(self, x):
        """
        Returns the cdf of x as an array the same shape as x
        """
        cdf_component = [prop * self._do_call(distr, "cdf", x, positional, scale=scale) for
                         distr, prop, positional, scale in self._get_component_iterator()]
        return np.sum(cdf_component, axis=0)

    def rvs(self, size = 1):
        
        # Total observations to generate (in case size is a tuple)
        tot = np.prod(size)
        
        # Simulate true cluster memberships
        cl = choices(list(range(self.k)), k = tot, weights = self.params['prop'])
        
        # Simulate from each
        rle = [ sum([x == K for x in cl]) for K in range(self.k) ]

        sim = np.empty(tot)
        idx = 0
        for n, distr in zip(rle, list(self._get_component_iterator())):
            sim[idx:(idx+n)] = self._do_call(distr[0], "rvs", distr[2], scale = distr[3], size = n)
            idx += n
        
        return sim.reshape(size)

    
    def ppf(self, x, minx=None, maxx=None, n_approx_pts=10000):
        """
        Returns the approximate quantile function (inverse cdf) of x as an array the same shape as x
        """
        
        if self.a != float("-inf"):
            approx_a = self.a
        else:
            if minx is None:
                raise ValueError(f"With infinite lower bound, `minx` must be specified.")
            approx_a = minx

        if self.b != float("inf"):
            approx_b = self.b
        else:
            if maxx is None:
                raise ValueError(f"With infinite upper bound, `maxx` must be specified.")
            approx_b = maxx

        approx_pts = np.linspace(approx_a, approx_b, n_approx_pts)
        y = self.cdf(approx_pts)
        myfun = interp1d(y, approx_pts, kind='linear')

        return myfun(x)
    
    def Estep(self, x):
        self.__z = self.logpdf(x)
        self.__z -= np.apply_along_axis(logsumexp, 1, self.__z)
        self.__z = np.exp(self.__z)
        
    def Mstep(self, x, weights = False):
        """
        Default method should just do numerical optimization for all parameters
            (except mixing weights).
        """
        if weights:
            # update mixing weights
            self.params['prop'] = np.mean(self.z, axis = 0)

        else:
            # update anything else
            pass 


    def fit(self):
        pass

    # # Setters ---------------------------------------------
    # def ll(self, x):
    #     ll = sum(self.pdf(x))
    #     self.ll = ll
