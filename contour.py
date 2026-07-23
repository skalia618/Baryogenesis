import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, optimize
import time
import warnings
warnings.simplefilter('ignore', category = DeprecationWarning)

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
    sol = integrate.solve_ivp(ode, [np.log(T0), np.log(Tf)], [Yeq(T0, m1), Yeq(T0, m2), 0.], method = 'Radau', atol = 1e-15)

    ind = np.argmin(np.abs(sol.t - np.log(1e-3)))
    if bounds and (sol.y[0,ind] > 1e-2 * Yeq(1e-3, 0) or sol.y[1,ind] > 1e-2 * Yeq(1e-3, 0)): return None # BBN bound

    if full: return sol
    else: return sol.y[2,-1]

def contour(x, kappa1, m2i, Mi, kappa2 = 1., lmbda = 1., theta = np.pi / 3, target = 8.8e-11, dx = 1e-2, err = 1e-3):
    # target is observed Y_delB
    # dx is distance (in log space) used to calculate derivatives
    # err controls step size along contour by limiting relative error in asym (before correction)
    asym_sh = lambda m2, M: asym(m2 / x, m2, M, kappa1, kappa2, lmbda, theta)

    contour = [(m2i, optimize.fsolve(lambda M: asym_sh(m2i, M) - target, Mi, epsfcn = dx * Mi)[0])]
    while contour[-1][0] <= m2i:
        m2 = contour[-1][0]
        M = contour[-1][1]

        asym0 = asym_sh(m2, M)
        asym_m2p = asym_sh(m2 * (1 + dx), M)
        asym_m2m = asym_sh(m2 * (1 - dx), M)
        asym_Mp = asym_sh(m2, M * (1 + dx))
        asym_Mm = asym_sh(m2, M * (1 - dx))
        asym_pp = asym_sh(m2 * (1 + dx), M * (1 + dx))
        asym_mp = asym_sh(m2 * (1 - dx), M * (1 + dx))
        asym_pm = asym_sh(m2 * (1 + dx), M * (1 - dx))
        asym_mm = asym_sh(m2 * (1 - dx), M * (1 - dx))
        
        dm2 = (asym_m2p - asym_m2m) / 2 / dx
        dM = (asym_Mp - asym_Mm) / 2 / dx
        ddm2 = (asym_m2p - 2 * asym0 + asym_m2m) / dx ** 2
        ddM = (asym_Mp - 2 * asym0 + asym_Mm) / dx ** 2
        dm2dM = (asym_pp - asym_mp - asym_pm + asym_mm) / 4 / dx ** 2

        step = np.sqrt(2 * err * asym0 / np.abs(ddm2 * dM ** 2 - 2 * dm2dM * dm2 * dM + ddM * dm2 ** 2))
        # Cap step size at 0.1 (in log space)
        if np.abs(dM * step) > 0.1 or np.abs(dm2 * step) > 0.1:
            step = 0.1 / max(np.abs(dM), np.abs(dm2))
        correction = asym_sh(m2 * (1 + dM * step), M * (1 - dm2 * step)) - target
        
        contour.append((m2 * (1 + dM * step - correction * dm2 / (dm2 ** 2 + dM ** 2)), M * (1 - dm2 * step - correction * dM / (dm2 ** 2 + dM ** 2))))

    return np.array(contour)

x = 0.3
m2i = 1e5
Mi = 1.2e8
kappa1 = 3e-3
kappa2 = 1.
lmbda = 1.
theta = np.pi / 3
data = contour(x, kappa1, m2i, Mi)
asyms = np.array([asym(dat[0] / x, dat[0], dat[1], kappa1, kappa2, lmbda, theta, bounds = True) for dat in data])
np.savez('contour3',
         x = x,
         kappa1 = kappa1,
         kappa2 = kappa2,
         lmbda = lmbda,
         theta = theta,
         contour = data,
         bound_mask = np.where(asyms != None)[0])
print(time.time() - t0)
