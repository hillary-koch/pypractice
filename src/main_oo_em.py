"""
(Eventually)
Runs an EM algorithm to estimate the parameters of a mixture model
Code is written using object-oriented programming
"""

from pypractice import NormalMixture

mydist = NormalMixture({"norm": 2})

mydist.k
mydist.add_components(components={"norm": 3})
mydist.k
mydist.pdf()
mydist.logpdf()
mydist.ll()