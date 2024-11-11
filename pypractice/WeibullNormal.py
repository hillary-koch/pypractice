from pypractice import MixtureDistribution
    
class WeibullNormalMixture(MixtureDistribution):

    def __init__(self, components: dict[str, int]):
        super().__init__(components)

    @property
    def supported_distns(self):
        return ['weibull_min', 'norm']
    
dir(__builtins__)