import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def quadratic(r, p1, p2, p3, p4, p5, p6):
    Z1 = 74
    Z2 = 74
    a = 0.46848 / (Z1 ** 0.23 + Z2 ** 0.23) 
    A = 14.399645 * Z1 * Z2
    x = r / a
    return A / r * (p1 * np.exp(-p2 * x) + p3 * np.exp(-p4 * x) + p5 * np.exp(-p6 * x)) 

r = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
y = np.array([1094.5, 696.8, 443.7, 282.5, 164.3, 91.1])
p0 = [0.3, 2.5, 0.091, 0.3, 0.58, 0.58]
bounds = ([0.25, 2.4, 0.09, 0.29, 0.56, 0.56], [0.5, 2.55, 0.1, 0.3, 0.6, 0.6])

popt, pcov = curve_fit(quadratic, r, y, p0=p0, bounds= bounds, maxfev=100000)

print(popt)

# 绘制拟合结果
plt.scatter(r, y)
plt.plot(r, quadratic(r, *popt), color='red')
plt.show()