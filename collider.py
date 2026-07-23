import matplotlib.pyplot as plt
import numpy as np
from parton import mkPDF
from scipy import optimize

pdf = mkPDF('cteq6l1')
invGeV2_to_pb = 3.89e8

ldata = np.loadtxt('dijet_low.txt').T
hdata = np.loadtxt('dijet_high.txt').T
s_low = 7e3 ** 2 # in GeV^2
A_low = 0.6
s_high = 1.3e4 ** 2 # in GeV^2
A_high = 0.5

def crosssec(m, f1, f2, lmbda, dataset):
    # m in GeV
    # flavors f1, f2: {1,2,3,4,5} = {d,u,s,c,b}
    # dataset: 'low' or 'high'
    if dataset == 'low':
        s = s_low
        A = A_low
    elif dataset == 'high':
        s = s_high
        A = A_high
    Gamma = 3 * lmbda ** 2 * m / 8 / np.pi
    xs = np.linspace(m ** 2 / s, 1, 1000)
    dx = xs[1] - xs[0]
    integrand = 1 / xs * (pdf.xfxQ(f1, xs, m)[:,0] * pdf.xfxQ(f2, m ** 2 / s / xs[::-1], m)[::-1,0] + pdf.xfxQ(f2, xs, m)[:,0] * pdf.xfxQ(f1, m ** 2 / s / xs[::-1], m)[::-1,0])
    return invGeV2_to_pb * A * lmbda ** 4 / 108 / Gamma / m * np.trapezoid(integrand, dx = dx)

def limit(f1, f2, lmbda, dataset):
    # flavors f1, f2: {1,2,3,4,5} = {d,u,s,c,b}
    # dataset: 'low' or 'high'
    if dataset == 'low': data = ldata
    if dataset == 'high': data = hdata
    crosssecs = np.array([crosssec(m, f1, f2, lmbda, dataset) for m in data[0]])
    return optimize.fsolve(lambda x: np.interp(x, data[0], np.log(crosssecs / data[1])), data[0,-1] * 0.99)[0]

# Compare to Fig. 4 in arXiv:1010.0203
##plt.plot(*ldata, color = 'k')
##crosssecs = np.array([crosssec(m, 2, 2, 1., 'low') for m in ldata[0]])
##plt.plot(ldata[0], crosssecs)
##plt.xlim(400., 2700.)
##plt.ylim(1.5e-3, 3.5e5)
##plt.yscale('log')
##plt.show()

# Compare to Fig. 6a in arXiv:1911.03947
##plt.plot(*hdata, color = 'k')
##crosssecs = np.array([crosssec(m, 2, 2, 1., 'high') for m in hdata[0]])
##plt.plot(hdata[0], crosssecs)
##plt.xlim(hdata[0,0], hdata[0,-1])
##plt.ylim(1e-5, 1e2)
##plt.yscale('log')
##plt.show()

print(limit(1, 3, 1, 'high'))
plt.plot(*hdata, color = 'k', label = 'CMS data')
crosssecs = np.array([crosssec(m, 1, 3, 1, 'high') for m in hdata[0]])
plt.plot(hdata[0], crosssecs, label = '$ds$ diquark')
plt.xlim(hdata[0,0], hdata[0,-1])
plt.ylim(3e-5, 0.3)
plt.yscale('log')
plt.legend()
plt.show()
