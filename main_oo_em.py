from pypractice import Normal

mydist = NormalMixture({"norm": 2})

mydist.k
mydist.add_components(components={"norm": 3})
mydist.k
mydist.pdf()
mydist.logpdf()
mydist.ll()