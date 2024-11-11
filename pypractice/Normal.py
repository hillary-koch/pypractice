from pypractice import MixtureDistribution

class NormalMixture(MixtureDistribution):

    def __init__(self, components: dict[str, int]):
        super().__init__(components)

    @property
    def supported_distns(self):
        return ['norm']