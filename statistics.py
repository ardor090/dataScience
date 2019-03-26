import matplotlib
matplotlib.use(‘Agg’)
import numpy as np
import matplotlib.pyplot as plt
size = [ 1400, 2400, 1800, 1900, 1300, 1100]
cost = [ 112000, 192000, 144000, 152000, 104000, 88000]
plt.scatter(size, cost)
plt.savefig(‘values1.png’)