"""Generate publication-quality convergence + eigenvalue figures."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
})

PHI = (1 + np.sqrt(5)) / 2

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
labels_N = [100, 200, 500, 1000]

for i, N in enumerate(labels_N):
    fname = f'results_convergence_N{N}.json'
    try:
        d = json.load(open(fname))
    except FileNotFoundError:
        continue
    JoA = np.array(d['JoA_values'])
    ustars = np.array(d['u_stars'])
    axes[0].plot(JoA, ustars, color=colors[i], linewidth=1.2, label=f'N={N}', alpha=0.85)

axes[0].axvline(x=PHI**(-2), color='red', linestyle='--', linewidth=0.8, alpha=0.6, label=r'$\phi^{-2}$')
axes[0].set_xlabel(r'$J/A$')
axes[0].set_ylabel(r'$u^* = \log_{10}(k_S / k_T)$')
axes[0].set_title('(a) Convergence: $u^*(J/A)$')
axes[0].set_xscale('log')
axes[0].legend(loc='upper right')
axes[0].set_xlim([0.01, 10])

Ns = []
crits = []
deltas = []
for N in labels_N:
    fname = f'results_convergence_N{N}.json'
    try:
        d = json.load(open(fname))
        Ns.append(N)
        crits.append(d['JoA_crit'])
        deltas.append(d['delta_u'])
    except FileNotFoundError:
        continue

axes[1].plot(Ns, crits, 'o-', color='#2166ac', markersize=6, linewidth=1.5)
axes[1].axhline(y=PHI**(-2), color='red', linestyle='--', linewidth=0.8, alpha=0.6, label=r'$\phi^{-2} \approx 0.382$')
axes[1].set_xlabel('Number of scan points $N$')
axes[1].set_ylabel(r'$J/A_{\rm crit}$')
axes[1].set_title(r'(b) $(J/A)_{\rm crit}$ convergence')
axes[1].legend()

try:
    eig = json.load(open('results_eigenvalue_spectrum.json'))
    JoA_eig = np.array(eig['JoA_values'])
    ustars_eig = np.array(eig['u_stars'])

    for ed in eig['eigenvalue_data']:
        joa = ed['JoA']
        for r_val in ed['gap_real'][:4]:
            axes[2].plot(joa, r_val, '.', color='#2166ac', markersize=2, alpha=0.5)

    axes[2].axvline(x=0.333, color='red', linestyle='--', linewidth=0.8, alpha=0.6, label=r'$J/A_{\rm crit}$')
    axes[2].set_xlabel(r'$J/A$')
    axes[2].set_ylabel(r'Re$(\lambda)$ (dominant Liouvillian eigenvalues)')
    axes[2].set_title('(c) Eigenvalue spectrum across bifurcation')
    axes[2].legend(loc='lower left')
except FileNotFoundError:
    axes[2].text(0.5, 0.5, 'No eigenvalue data', transform=axes[2].transAxes, ha='center')

plt.tight_layout()
plt.savefig('figures/figure6_convergence_eigenvalue.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/figure6_convergence_eigenvalue.pdf', bbox_inches='tight')
print("Saved figures/figure6_convergence_eigenvalue.png/pdf")

fig2, ax2 = plt.subplots(figsize=(6, 4))
try:
    eig = json.load(open('results_eigenvalue_spectrum.json'))
    JoA_eig = np.array(eig['JoA_values'])
    ustars_eig = np.array(eig['u_stars'])
    DY_maxs = np.array(eig['DY_maxs'])

    ax2_twin = ax2.twinx()
    ax2.plot(JoA_eig, ustars_eig, 'o-', color='#2166ac', markersize=3, linewidth=1.2, label=r'$u^*$')
    ax2_twin.plot(JoA_eig, DY_maxs, 's-', color='#b2182b', markersize=3, linewidth=1.2, label=r'$\Delta Y_{\max}$')

    ax2.axvline(x=0.333, color='grey', linestyle=':', linewidth=0.8, alpha=0.6)
    ax2.set_xlabel(r'$J/A$')
    ax2.set_ylabel(r'$u^*$', color='#2166ac')
    ax2_twin.set_ylabel(r'$\Delta Y_{\max}$', color='#b2182b')
    ax2.set_title('Bifurcation: $u^*$ jump and anisotropy')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
except FileNotFoundError:
    ax2.text(0.5, 0.5, 'No eigenvalue data', transform=ax2.transAxes, ha='center')

plt.tight_layout()
plt.savefig('figures/figure7_bifurcation_detail.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/figure7_bifurcation_detail.pdf', bbox_inches='tight')
print("Saved figures/figure7_bifurcation_detail.png/pdf")

print("\nConvergence summary:")
for N in labels_N:
    fname = f'results_convergence_N{N}.json'
    try:
        d = json.load(open(fname))
        print(f"  N={N:>4d}: J/A_crit={d['JoA_crit']:.4f}, delta_u={d['delta_u']:.3f}")
    except FileNotFoundError:
        print(f"  N={N:>4d}: not available")
