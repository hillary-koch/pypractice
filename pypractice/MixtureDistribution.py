import numpy as np
from scipy.stats import norm, weibull_min, expon


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
        self.k = self._count_components(components)
        self.__components = components
        self.__params = {
            'prop': np.ones(shape=(self.k,))/self.k,
            'loc': np.empty(shape=(self.k,)),
            'scale': np.ones(shape=(self.k,))
        }

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
    def components(self):
        return self.__components

    @property
    def ll(self):
        return self.__ll

    # Class-specific auxiliary functions ------------------------------------------------
    @staticmethod
    def _count_components(x: dict[str, int]):
        return sum(x.values())

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
            self.params['loc'],
            self.params['scale']
        )

    # Public class methods ------------------------------------------------
    def add_components(self, components=dict[str, int]):
        self._check_components(components)
        components = {key: self.components.get(
            key, 0) + components.get(key, 0) for key in (self.components.keys() | components.keys())}
        
        # Re-initialize after changing components
        self.__init__(components)

    def rvs(self):
        raise NotImplementedError(f"This method not yet implemented for {
                                  self.__class__} class.")

    def pdf(self, x):
        """
        Returns the density of x in each component
        """
        return [eval(f"{prop} * {dist}.pdf({x}, loc = {loc}, scale = {scale})")
                for dist, prop, loc, scale in self._get_component_iterator()]

    def logpdf(self, x):
        """
        Returns the log density of x in each component
        """
        return [eval(f"np.log({prop}) + {dist}.logpdf({x}, loc = {loc}, scale = {scale})")
                for dist, prop, loc, scale in self._get_component_iterator()]

    def cdf(self):
        raise NotImplementedError(f"This method not yet implemented for {
                                  self.__class__} class.")

    def ppf(self):
        raise NotImplementedError(f"This method not yet implemented for {
                                  self.__class__} class.")

    def fit(self):
        pass

    # # Setters ---------------------------------------------
    # def ll(self, x):
    #     ll = sum(self.pdf(x))
    #     self.ll = ll
