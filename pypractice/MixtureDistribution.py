import numpy as np

class MixtureDistribution():
    """
    Mixture distribution
    __valid_distns: distributions for which methods have been implemented.
        The keys in the `components` dictionary must be found in this array.
    """
    
    __supported_distns = ['norm', 'weibull_min', 'exp']
    
    def __init__(self, components: dict[str, int]):
        """
        k = number of mixture components
        """
        self._check_components(components)
        self.k = self._count_components(components)
        self.__components = components
        self.__params = {
            'prop': np.empty(shape=(self.k,))
        }

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
    
    # Class-specific auxiliary functions ------------------------------------------------
    @staticmethod
    def _count_components(x: dict[str, int]):
        return sum(x.values())

    # Private class methods ------------------------------------------------
    def _check_components(self, x: dict[str, int]):
        if not all([dist in self.supported_distns for dist in x.keys()]):
            raise ValueError(f"Supported components keys are {
                             self.supported_distns}. You provided {[dist for dist in x.keys()]}")

    # Public class methods ------------------------------------------------
    def add_components(self, components=dict[str, int]):
        self._check_components(components)
        self.k += self._count_components(components)

        self.__components = components

        self.__params = {
            'prop': np.empty(shape=(self.k,))
        }
        
    def rvs(self):
        raise NotImplementedError(f"This method not yet implemented for {self.__class__} class.")
    
    def pdf(self):
        raise NotImplementedError(f"This method not yet implemented for {self.__class__} class.")
    
    def logpdf(self):
        np.log(self.pdf())
    
    def cdf(self):
        raise NotImplementedError(f"This method not yet implemented for {self.__class__} class.")
    
    def ppf(self):
        raise NotImplementedError(f"This method not yet implemented for {self.__class__} class.")
    
    def ll(self):
        pass