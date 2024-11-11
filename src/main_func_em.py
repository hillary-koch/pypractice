"""
Runs an EM algorithm to estimate the parameters of a Weibull-Normal mixture model
Code is written using functional programming
"""
import numpy as np
import pypractice 
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, norm

# generate data
weib = weibull_min.rvs(c = 10, scale = 4, loc = 0, size = (200,))
normals = norm.rvs(loc = 7, scale = 1.5, size = (100,))
d = np.r_[weib, normals]

# Plotting a basic histogram
plt.hist(d, bins=30, color='skyblue', edgecolor='black')

# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Basic Histogram')

# Display the plot
plt.show()

start_vals = pypractice.get_wn_startvals(d)
fit_wnmix = pypractice.weibull_normal_em(
    d,
    init_mu = start_vals.get('init_mu'),
    init_sigma = start_vals.get('init_sigma'),
    init_shape = start_vals.get('init_shape'),
    init_scale = start_vals.get('init_scale'),
    init_prop = start_vals.get('init_prop')
)

fit_wnmix['parameters']
p = pypractice.pweibnormix(fit_wnmix['x'], fit_wnmix)
q = pypractice.qweibnormix(p, fit_wnmix)
max(q - fit_wnmix['x']) # want to be close to 0

