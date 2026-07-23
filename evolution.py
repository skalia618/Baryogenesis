from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, optimize
import time

plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})

fig, ax = plt.subplots(figsize = (8., 6.))

BLUE = np.array([81, 167, 192]) / 255
ORANGE = np.array([255, 184, 56]) / 255
GREEN = np.array([107, 171, 115]) / 255

t0 = time.time()
Mpl = 2.4e18 # in GeV
zeta3 = 1.20206

gstardata = np.loadtxt('gstar.txt')[::-1].T
gstardata[0] /= 1000 # convert MeV to GeV
gstar = lambda T: np.interp(np.log(T), np.log(gstardata[0]), gstardata[1])
# The following is d(gstar) / d(log T):
dgstar = lambda T: np.interp(np.log(T), (np.log(gstardata[0,1:]) + np.log(gstardata[0,:-1])) / 2, (gstardata[1,1:] - gstardata[1,:-1]) / (np.log(gstardata[0,1:]) - np.log(gstardata[0,:-1])))

xs = np.linspace(0, 100, 1000)
yint = lambda x: integrate.quad(lambda xi: xi ** 2 / (np.exp(np.sqrt(xi ** 2 + x ** 2)) + 1), 0, max(20, 3 * x))[0]
yintdata = np.array([yint(x) for x in xs])
def Yeq(T, m):
    # T can be array, but m must be float
    if m == 0: return 135 * zeta3 / 4 / np.pi ** 4 / gstar(T)
    x = m / T
    return np.piecewise(x, [x <= 100, x > 100],
                        [lambda temp: 45 / 2 / np.pi ** 4 / gstar(m / temp) * np.interp(temp, xs, yintdata),
                         lambda temp: 45 / 2 / np.pi ** 4 / gstar(m / temp) * np.sqrt(np.pi * temp ** 3 / 2) * np.exp(-temp)])

H = lambda T: np.sqrt(np.pi ** 2 * gstar(T) / 90) * T ** 2 / Mpl

gDelta = lambda x: x * ((1 - x ** 4) * (1 - 8 * x ** 2 + x ** 4) - 24 * x ** 4 * np.log(x))
g1 = lambda x: (1 - x) ** 4 * (1 + 4 * x + 10 * x ** 2 + 4 * x ** 3 + x ** 4)
g2 = lambda x: x * ((1 - x ** 2) * (1 + 10 * x ** 2 + x ** 4) + 12 * x ** 2 * (1 + x ** 2) * np.log(x))

def asym(m1, m2, M, kappa1, kappa2, lmbda, theta, full = False, bounds = False, scat_asym = False):
    # m1, m2, and M in GeV
    # full returns full solution (as opposed to just final asymmetry)
    # bounds returns None if EFT or BBN bound is violated
    # scat_asym includes asymmetry from scatterings (uses same epsilon as decays)
    m1 = float(m1)
    m2 = float(m2)
    M = float(M)
    kappa1 = float(kappa1)
    kappa2 = float(kappa2)
    lmbda = float(lmbda)
    theta = float(theta)
    
    GammaSB = lambda T: 45 * zeta3 * kappa1 ** 2 * lmbda ** 2 * T ** 5 / 16 / np.pi ** 3 / M ** 4 # assuming T >> m1
    GammaSB2 = lambda T: 45 * zeta3 * kappa2 ** 2 * lmbda ** 2 * T ** 5 / 16 / np.pi ** 3 / M ** 4 # assuming T >> m1
    GammaS0 = lambda T: 27 * zeta3 * kappa1 ** 2 * kappa2 ** 2 * T ** 5 / 32 / np.pi ** 3 / M ** 4 # assuming T >> m1,m2
    GammaSBV = lambda T: kappa2 ** 4 * lmbda ** 4 * T ** 11 / m2 ** 2 / M ** 8 # constant factor omitted
    GammaDB = 3 * kappa1 ** 2 * lmbda ** 2 * m1 ** 5 / 3072 / np.pi ** 3 / M ** 4
    GammaDB2 = 3 * kappa2 ** 2 * lmbda ** 2 * m2 ** 5 / 3072 / np.pi ** 3 / M ** 4
    GammaD0 = 3 * kappa1 ** 2 * kappa2 ** 2 * m1 ** 5 / 3072 / np.pi ** 3 / M ** 4 * (g1(m2 / m1) - 2 * np.cos(2 * theta) * g2(m2 / m1))

    T0 = 10 * m1 * (H(m1) / (2 * GammaSB(m1) + 2 * GammaS0(m1))) ** (1 / 3) # in GeV
    if bounds and T0 > M: return None # EFT bound
    Tf = 0.1 * min(np.sqrt((2 * GammaDB + GammaD0) / H(1.)), np.sqrt(2 * GammaDB2 / H(1.)), m2) # in GeV
    
    eps = (m1 ** 2 / 8 / np.pi / M ** 2 * (kappa1 ** 2 * kappa2 ** 2 * lmbda ** 2 * np.sin(2 * theta) * gDelta(m2 / m1))
           / (kappa1 ** 2 * kappa2 ** 2 * g1(m2 / m1) - 2 * kappa1 ** 2 * kappa2 ** 2 * np.cos(2 * theta) * g2(m2 / m1) + kappa1 ** 2 * lmbda ** 2))

    def ode(logT, Ys):
        T = np.exp(logT)
        Y1, Y2, Ydel = Ys

        dgfactor = 1 + dgstar(T) / gstar(T) / 3
        Y1eq = Yeq(T, m1)
        Y2eq = Yeq(T, m2)
        YBeq = 6 * Yeq(T, 0)
        if Y2eq !=0: rat = Y1eq / Y2eq
        else: rat = (m1 / m2) ** 1.5 * np.exp(-(m1 - m2) / T)
        
        dY1 = dgfactor / H(T) * (2 * (GammaDB + GammaSB(T)) * (Y1 - Y1eq) + (GammaD0 + 2 * GammaS0(T)) * (Y1 - Y2 * rat))
        dY2 = dgfactor / H(T) * (2 * (GammaDB2 + GammaSB2(T)) * (Y2 - Y2eq) - (GammaD0 + 2 * GammaS0(T)) * (Y1 - Y2 * rat))
        dYdel = dgfactor / H(T) * (-eps * (2 * GammaDB + GammaD0 + scat_asym * (2 * GammaSB(T) + GammaS0(T))) * (Y1 - Y1eq)
                                   + (3 * GammaDB * Y1eq + 3 * GammaDB2 * Y2eq + GammaSB(T) * (Y1 + 2 * Y1eq) + GammaSB2(T) * (Y2 + 2 * Y2eq)) * Ydel / YBeq
                                   + 6 * GammaSBV(T) * Ydel)

        return [dY1, dY2, dYdel]
    sol = integrate.solve_ivp(ode, [np.log(T0), np.log(Tf)], [Yeq(T0, m1), Yeq(T0, m2), 0.], method = 'Radau', atol = 1e-15, t_eval = np.linspace(np.log(T0), np.log(Tf), 1000))

    ind = np.argmin(np.abs(sol.t - np.log(1e-3)))
    if bounds and (sol.y[0,ind] > 1e-2 * Yeq(1e-3, 0) or sol.y[1,ind] > 1e-2 * Yeq(1e-3, 0)): return None # BBN bound

    T1 = optimize.fsolve(lambda T: 2 * GammaSB(T) + 2 * GammaS0(T) - H(T), T0)
    T2 = optimize.fsolve(lambda T: 2 * GammaDB + GammaD0 - H(T), T0)

    if full: return sol, T1, T2
    else: return sol.y[2,-1]

##sol, T1, T2 = asym(50. / 0.3, 50., 59000., 3e-4, 1., 1., np.pi / 3, full = True, bounds = True)
##xlim1 = 0.5 * np.exp(sol.t[-1])
##xlim2 = 2. * np.exp(sol.t[0])
##ax.text(T1 * 0.78, 0.002, r'$\Gamma_S=H$', ha = 'center', va = 'center', color = '0.6', rotation = 90)
##ax.text(T2 * 0.77, 0.0035, r'$\Gamma_D=H$', ha = 'center', va = 'center', color = '0.6', rotation = 90)

sol, T1, T2 = asym(50. / 0.3, 50., 9200., 3e-4, 1., 1., np.pi / 3, full = True, bounds = True)
xlim1 = 0.66 * np.exp(sol.t[-1])
xlim2 = 1.5 * np.exp(sol.t[0])
ax.text(T1 * 0.89, 0.0005, r'$\Gamma_S=H$', ha = 'center', va = 'center', color = '0.6', rotation = 90)
ax.text(T2 * 0.87, 0.0035, r'$\Gamma_D=H$', ha = 'center', va = 'center', color = '0.6', rotation = 90)

ax.axvline(T1, color = '0.7', ls = ':')
ax.axvline(T2, color = '0.7', ls = ':')

ax.plot(np.exp(sol.t), Yeq(np.exp(sol.t), 50. / 0.3), color = BLUE, ls = '--', label = r'$Y_{1,\mathrm{eq}}$')
ax.plot(np.exp(sol.t), sol.y[0], color = BLUE, label = '$Y_1$')
ax.plot(np.exp(sol.t), sol.y[1], color = ORANGE, label = '$Y_2$')
ax.plot(np.exp(sol.t), sol.y[2] * 3e7, color = GREEN, label = r'$Y_{\Delta B}\cdot3\times10^7$')

##handles, labels = ax.get_legend_handles_labels()
##handles[:2] = [handles[1], handles[0]]
##labels[:2] = [labels[1], labels[0]]
##ax.legend(handles, labels, loc = (0.7, 0.08))

ylim1 = -0.0002
ylim2 = 0.0041
ax.set_xlim(xlim1, xlim2)
ax.set_ylim(ylim1, ylim2)
ax.set_xscale('log')
ax.set_xlabel(r'$T$\,[GeV]')

class CustomTicker(ticker.LogFormatterSciNotation): 
    def __call__(self, x, pos = None): 
        if x not in [0.1, 1, 10]: 
            return ticker.LogFormatterSciNotation.__call__(self, x, pos = None) 
        else: 
            return "{x:g}".format(x = x)

ax.tick_params(which = 'both', direction = 'in')
ax.xaxis.set_major_formatter(CustomTicker())
ax.xaxis.minorticks_off()
secxax = ax.secondary_xaxis('top')
secxax.tick_params(which = 'both', direction = 'in')
secxax.xaxis.minorticks_off()
plt.setp(secxax.get_xticklabels(), visible = False)
secyax = ax.secondary_yaxis('right', zorder = 1)
secyax.tick_params(which = 'both', direction = 'in')
plt.setp(secyax.get_yticklabels(), visible = False)

fig.tight_layout()
fig.show()
#fig.savefig('lowM.pdf')
