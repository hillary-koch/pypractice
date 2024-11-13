"""
(Eventually)
Runs an EM algorithm to estimate the parameters of a mixture model
Code is written using object-oriented programming
"""

from pypractice import NormalMixture, WeibullNormalMixture
from scipy.stats import norm, weibull_min, expon

mydist = WeibullNormalMixture({"norm": 2})


# mydist.k
mydist.add_components(components={"norm": 1, "weibull_min": 1})
# mydist.k

mydist.components
mydist.pdf(x = [0,1,2])
mydist.params

mydist.pdf(x = [[0,2,3,1,3,-2]])
mydist.pdf(x = [[0,2,3], [1,3,-2]])
mydist.ppf(x = [0.1,.2,.3,.9], minx = -100, maxx = 100)
# mydist.logpdf(x = [1,2])
# # mydist.ll()
# help(weibull_min)
# weibull_min.pdf([1,2,3], 1, scale = 2)
# expon.pdf([1,2,3], 1)


# if i change how parameters are specified, i.e. move away from some positional arguments,
#   but specify kwargs as None for distributions that don't have/use a given kwarg,
#   I can filter the Nones in do_call like this 
def testcall(distr, method, args=[], **kwargs):
    if not hasattr(args, '__iter__'):
        args = [args]
    
    # filter Nones
    args = [ r for r in args if r is not None ]
    kwargs = {key: kwargs.get(key) for key in kwargs.keys() if kwargs.get(key) is not None}
    
    fun = getattr(globals()[distr], method)
    return fun(*args, **kwargs)

# throws an error because 2.5 positional argument is interpreted as the location parameter
testcall("norm", "pdf", [[1,2], 2.5], loc = 2.5, scale = 1)

# works
testcall("norm", "pdf", [[1,2], None], loc = 2.5, scale = 1)
