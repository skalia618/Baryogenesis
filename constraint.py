from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 14})

fig, ax = plt.subplots(figsize = (8., 6.))

flavor = 'dsc' # 'uds', 'dsc', or 'udb'

BLUE = np.array([81, 167, 192]) / 255
ORANGE = np.array([255, 184, 56]) / 255
RED = np.array([235, 126, 110]) / 255

ax.set_xscale('log')
ax.set_yscale('log')
xlim1 = 0.5
xlim2 = 1e5
ylim1 = 30
ylim2 = 3e8
ax.set_xlim(xlim1, xlim2)
ax.set_ylim(ylim1, ylim2)
ax.set_xlabel(r'$m_2\,[\mathrm{GeV}]$')
ax.set_ylabel(r'$M\,[\mathrm{GeV}]$')

yr_to_invGeV = 4.79434e31
fm_to_invGeV = 5.06773

mp = 0.938 # in GeV
LQCD = 0.2 # in GeV
fpi = 0.093 # in GeV
ms = 0.095 # in GeV
mc = 1.27 # in GeV
mb = 4.18 # in GeV
mW = 80 # in GeV
GF = 1.166e-5 # in GeV^-2
rhoN = 0.25 / fm_to_invGeV ** 3 # in GeV^3
Vud = 0.97
Vus = 0.23
Vcd = 0.22
Vub = 0.0037

taupp = 1.7e32 * yr_to_invGeV # in GeV^-1
delnn = 1.9e-33 # in GeV
Cnn = 5.6e-4 # in GeV^6
if flavor in {'uds', 'dsc'}: Mmax = 2500 # in GeV
elif flavor == 'udb': Mmax = 770 # in GeV

data1 = np.load('contour1.npz')
data2 = np.load('contour2.npz')
data3 = np.load('contour3.npz')
kappa1_1 = data1['kappa1']
contour1 = data1['contour']
kappa2 = data1['kappa2']
lmbda = data1['lmbda']
bound_mask1 = data1['bound_mask']
kappa1_2 = data2['kappa1']
contour2 = data2['contour']
bound_mask2 = data2['bound_mask']
kappa1_3 = data3['kappa1']
contour3 = data3['contour']
bound_mask3 = data3['bound_mask']

angle = -180 / np.pi * np.arctan(np.log(xlim2 / xlim1) / np.log(ylim2 / ylim1) * 3 / 16)
masses = np.logspace(np.log10(xlim1), np.log10(xlim2), 1000)
uds_bound = (kappa2 ** 4 * lmbda ** 4 * rhoN * LQCD ** 10 * taupp / (32 * np.pi * mp ** 2 * masses ** 2)) ** (1 / 8)
dsc_bound = (Cnn / delnn / masses) ** (1 / 4) * np.sqrt(Vus * Vcd * GF / np.sqrt(2) * 4 * np.pi * fpi ** 3 / mc)
udb_bound = (Cnn / delnn / masses) ** (1 / 4) * np.sqrt(Vud * Vub * GF / np.sqrt(2) * 4 * np.pi * fpi ** 3 / mb)
if flavor == 'uds':
    ax.fill_between(masses[np.where(masses > mp)], Mmax, uds_bound[np.where(masses > mp)], color = '0.9', alpha = 0.5)
    ax.plot(masses[np.where(masses > mp)], uds_bound[np.where(masses > mp)], color = '0.7')
    ax.text(masses[300], uds_bound[300] * 0.65, r'Dinucleon Decay', color = '0.2', ha = 'center', va = 'center', rotation = angle)
ax.fill([xlim1, mp, mp, xlim2, xlim2, xlim1], [ylim2, ylim2, Mmax, Mmax, ylim1, ylim1], color = '0.75', alpha = 0.5)
ax.axvline(mp, ymin = np.log(Mmax / ylim1) / np.log(ylim2 / ylim1), color = '0.55')
ax.axhline(Mmax, xmin = np.log(mp / xlim1) / np.log(xlim2 / xlim1), color = '0.55')
ax.text(masses[800], Mmax * 0.6, r'Collider', color = '0.2', ha = 'center', va = 'center', fontsize = 16)
ax.text(mp * 0.8, np.sqrt(ylim1 * ylim2), r'Proton Decay', color = '0.2', ha = 'center', va = 'center', rotation = 'vertical')
if flavor == 'dsc':
    ax.plot(masses, dsc_bound, color = '0.6', ls = '--')
    ax.text(masses[500], dsc_bound[500] * 1.35, r'Neutron Oscillations', color = '0.2', ha = 'center', va = 'center', rotation = angle)
elif flavor == 'udb':
    ax.plot(masses, udb_bound, color = '0.6', ls = '--')
    ax.text(masses[280], udb_bound[280] * 1.35, r'Neutron Oscillations', color = '0.2', ha = 'center', va = 'center', rotation = angle)
ax.text(3e4, 1.25e8, rf'${flavor}$', ha = 'center', va = 'center', fontsize = 16, bbox = dict(boxstyle = 'round', facecolor = 'white', alpha = 0.5))

def scientific(x, n = 0):
    exp = int(np.floor(np.log10(x)))
    coeff = x / 10 ** exp
    if f'{coeff:.{n}f}' != '1': return rf'{coeff:.{n}f}\times10^{{{exp}}}'
    else: return rf'10^{{{exp}}}'

plt.loglog(*contour1[bound_mask1].T, color = BLUE, label = r'$\kappa_1=' + scientific(kappa1_1) + '$', zorder = 2.3)
plt.loglog(*contour2[bound_mask2].T, color = ORANGE, label = r'$\kappa_1=' + scientific(kappa1_2) + '$', zorder = 2.2)
plt.loglog(*contour3[bound_mask3].T, color = RED, label = r'$\kappa_1=' + scientific(kappa1_3) + '$', zorder = 2.1)

class CustomTicker(ticker.LogFormatterSciNotation): 
    def __call__(self, x, pos = None):
        if x not in np.concatenate((0.1 * np.arange(1, 10), np.arange(1, 10), 10 * np.arange(1, 10))): 
            return ticker.LogFormatterSciNotation.__call__(self, x, pos = None) 
        else:
            return "{x:g}".format(x = x)

ax.tick_params(which = 'both', direction = 'in')
ax.xaxis.set_major_formatter(CustomTicker())
ax.yaxis.set_major_formatter(CustomTicker())
secxax = ax.secondary_xaxis('top')
secxax.tick_params(which = 'both', direction = 'in')
plt.setp(secxax.get_xticklabels(), visible = False)
secyax = ax.secondary_yaxis('right')
secyax.tick_params(which = 'both', direction = 'in')
plt.setp(secyax.get_yticklabels(), visible = False)

#ax.legend(loc = 'upper left', bbox_to_anchor = (0.05, 0.99))

fig.tight_layout()
#fig.show()
fig.savefig(f'constraint_{flavor}.pdf')
