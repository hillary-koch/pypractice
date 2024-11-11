"""
(Eventually)
Runs an EM algorithm to estimate the parameters of a mixture model
Code is written using object-oriented programming
"""

from pypractice import NormalMixture
from scipy.stats import norm, weibull_min, expon

mydist = NormalMixture({"norm": 2})

[print(x) for x in zip(mydist.params['prop'],
                       mydist.params['loc'], mydist.params['scale'])]

x = 0
myiter = zip(
    [key for key in mydist.components.keys() for value in range(mydist.components[key])],
    mydist.params['prop'],
    mydist.params['loc'],
    mydist.params['scale']
)
[ eval(f"{prop} * {dist}.pdf({x}, loc = {loc}, scale = {scale})") for dist, prop, loc, scale in myiter ]

mydist._do_call('expon', 'pdf', x=1)
mydist.k
mydist.add_components(components={"norm": 3})
mydist.k
mydist.pdf(x = 1)
mydist.logpdf(x = 2)
# mydist.ll()
