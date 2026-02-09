"""
Generate all 5 publication figures from the paper.

"Discontinuous bifurcation and nuclear spectroscopy in radical pair
magnetoreception" — Ursachi (2026), submitted to Physical Review E.

Usage:
    python generate_figures.py          # All 5 figures
    python generate_figures.py 1        # Only Figure 1
    python generate_figures.py 1 2      # Figures 1 and 2

Outputs saved to figures/ as PNG (300 DPI) and PDF.

NOTE: Figures 4 and 5 include 2-nucleus computations (dim=16,
Liouvillian 256×256) which take several minutes. Single-nucleus
figures (1-3) complete in ~1 minute each.
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
from rpm_core import (
    RadicalPairSystem, KT_DEFAULT, GAMMA_DEFAULT,
    B_DEFAULT, PHI, PHI4, LOG_PHI4,
)

FIG_DIR = "figures"
DATA_DIR = "data"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    "text.usetex": False,
})

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

kT = KT_DEFAULT
gamma = GAMMA_DEFAULT


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"))
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"))
    plt.close(fig)
    print(f"    -> {name}.png + .pdf")


def load_cached(name):
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_cached(data, name):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, name), "w") as f:
        json.dump(data, f)


def figure1():
    """Fig 1: (a) Response shelves, (b) curve collapse, (c) power-law scaling."""
    print("Figure 1: Response shelf + curve collapse + power law")
    rp = RadicalPairSystem(1, 0.5)
    u_vals = np.linspace(-1.0, 2.5, 50)

    configs = [
        (0.5, 0.125, "A=0.5, J=0.125"),
        (0.75, 0.1875, "A=0.75, J=0.19"),
        (1.0, 0.25, "A=1.0, J=0.25"),
        (1.5, 0.375, "A=1.5, J=0.375"),
        (2.0, 0.50, "A=2.0, J=0.50"),
    ]

    raw_curves = []
    norm_curves = []
    DY_maxs = []
    A_list = []
    for A, J, label in configs:
        print(f"    {label}...")
        dy = rp.scan_u(A, J, u_vals, n_angles=11)
        mx = np.max(dy)
        raw_curves.append((label, dy))
        norm_curves.append(dy / mx if mx > 0 else dy)
        DY_maxs.append(mx)
        A_list.append(A)

    rs = []
    for i in range(len(norm_curves)):
        for j in range(i + 1, len(norm_curves)):
            r, _ = pearsonr(norm_curves[i], norm_curves[j])
            rs.append(r)

    fig = plt.figure(figsize=(7.2, 2.6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    for i, (label, dy) in enumerate(raw_curves):
        ax1.plot(u_vals, dy * 1e3, color=COLORS[i], label=label)
    ax1.set_xlabel(r"$u = \log_{10}(k_S/k_T)$")
    ax1.set_ylabel(r"$\Delta Y \times 10^3$")
    ax1.set_title("(a) Response shelves")
    ax1.legend(loc="upper right", framealpha=0.7, fontsize=6)
    ax1.set_xlim(-1.0, 2.5)

    ax2 = fig.add_subplot(gs[0, 1])
    for i, nc in enumerate(norm_curves):
        ax2.plot(u_vals, nc, color=COLORS[i])
    ax2.set_xlabel(r"$u = \log_{10}(k_S/k_T)$")
    ax2.set_ylabel(r"$\Delta Y / \Delta Y_{\max}$")
    ax2.set_title(f"(b) Curve collapse ($r_{{\\min}}$ = {min(rs):.2f})")
    ax2.set_xlim(-1.0, 2.5)
    ax2.set_ylim(-0.05, 1.1)

    ax3 = fig.add_subplot(gs[0, 2])
    A_arr = np.array(A_list)
    DY_arr = np.array(DY_maxs)
    coeffs = np.polyfit(np.log10(A_arr), np.log10(DY_arr), 1)
    alpha = coeffs[0]
    A_fit = np.linspace(0.4, 2.2, 50)
    DY_fit = 10 ** np.polyval(coeffs, np.log10(A_fit))
    ax3.scatter(A_arr, DY_arr * 1e3, color="k", s=30, zorder=5)
    ax3.plot(A_fit, DY_fit * 1e3, "k--", alpha=0.5, lw=0.8)
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("$A$ (mT)")
    ax3.set_ylabel(r"$\Delta Y_{\max} \times 10^3$")
    ax3.set_title(rf"(c) Power law $|\alpha|$ = {abs(alpha):.2f}")

    save(fig, "figure1")
    print(f"    r_min = {min(rs):.3f}, |alpha| = {abs(alpha):.3f}")


def figure2():
    """Fig 2: (a) Full-scan u* vs J/A, (b) fine-scan bifurcation."""
    print("Figure 2: Bifurcation")
    rp = RadicalPairSystem(1, 0.5)
    u_scan = np.linspace(-1.0, 2.5, 40)

    cached = load_cached("fig2_data.json")
    if cached:
        JoA_full = np.array(cached["JoA_full"])
        u_stars_full = np.array(cached["u_stars_full"])
        JoA_fine = np.array(cached["JoA_fine"])
        u_stars_fine = np.array(cached["u_stars_fine"])
    else:
        print("    Full scan J/A in [0.01, 10]...")
        JoA_full = np.logspace(-2, 1, 100)
        u_stars_full = np.zeros(len(JoA_full))
        for i, joa in enumerate(JoA_full):
            if i % 20 == 0:
                print(f"      {i}/100...")
            dy = rp.scan_u(1.0, joa, u_scan, n_angles=11)
            u_stars_full[i] = u_scan[np.argmax(dy)]

        print("    Fine scan J/A in [0.236, 0.382]...")
        JoA_fine = np.linspace(0.236, 0.382, 20)
        u_stars_fine = np.zeros(len(JoA_fine))
        for i, joa in enumerate(JoA_fine):
            dy = rp.scan_u(1.0, joa, u_scan, n_angles=11)
            u_stars_fine[i] = u_scan[np.argmax(dy)]

        save_cached({
            "JoA_full": JoA_full.tolist(),
            "u_stars_full": u_stars_full.tolist(),
            "JoA_fine": JoA_fine.tolist(),
            "u_stars_fine": u_stars_fine.tolist(),
        }, "fig2_data.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    du = np.abs(np.diff(u_stars_full))
    idx = int(np.argmax(du))
    u_pre = u_stars_full[idx]
    u_post = u_stars_full[idx + 1]
    fz_lo = min(u_pre, u_post)
    fz_hi = max(u_pre, u_post)
    JoA_crit = (JoA_full[idx] + JoA_full[idx + 1]) / 2

    ax1.plot(JoA_full, u_stars_full, "k.-", markersize=2, linewidth=0.8)
    ax1.fill_between([JoA_full.min(), JoA_full.max()], fz_lo, fz_hi,
                     color="gray", alpha=0.15, label="Forbidden zone")
    ax1.axhline(LOG_PHI4, color="red", ls="--", lw=0.8, alpha=0.8,
                label=rf"$\log_{{10}}\varphi^4 = {LOG_PHI4:.3f}$")
    ax1.set_xscale("log")
    ax1.set_xlabel("$J/A$")
    ax1.set_ylabel("$u^*$")
    ax1.set_title(f"(a) Full scan ($J/A_{{crit}}$ = {JoA_crit:.3f})")
    ax1.legend(fontsize=6.5, loc="upper right")
    ax1.set_xlim(0.01, 10)
    ax1.set_ylim(-1.5, 2.0)

    du_fine = np.abs(np.diff(u_stars_fine))
    idx_f = int(np.argmax(du_fine))
    delta_u = du_fine[idx_f]
    fz_f = [min(u_stars_fine[idx_f], u_stars_fine[idx_f + 1]),
            max(u_stars_fine[idx_f], u_stars_fine[idx_f + 1])]
    JoA_crit_f = (JoA_fine[idx_f] + JoA_fine[idx_f + 1]) / 2

    ax2.plot(JoA_fine, u_stars_fine, "k.-", markersize=4, linewidth=1.2)
    ax2.fill_between([JoA_fine.min(), JoA_fine.max()], fz_f[0], fz_f[1],
                     color="gray", alpha=0.15)
    ax2.axvline(JoA_crit_f, color="blue", ls=":", lw=0.8, alpha=0.7)
    ax2.axhline(LOG_PHI4, color="red", ls="--", lw=0.8, alpha=0.8)

    mid_u = (fz_f[0] + fz_f[1]) / 2
    ax2.annotate("", xy=(JoA_crit_f + 0.005, fz_f[0] + 0.05),
                 xytext=(JoA_crit_f + 0.005, fz_f[1] - 0.05),
                 arrowprops=dict(arrowstyle="<->", color="blue", lw=1.2))
    ax2.text(JoA_crit_f + 0.015, mid_u, rf"$\Delta u = {delta_u:.2f}$",
             fontsize=8, color="blue", va="center")
    ax2.set_xlabel("$J/A$")
    ax2.set_ylabel("$u^*$")
    ax2.set_title("(b) Fine scan")

    fig.tight_layout()
    save(fig, "figure2")
    print(f"    J/A_crit = {JoA_crit:.3f}, delta_u = {delta_u:.2f}")


def figure3():
    """Fig 3: Phase diagram u* in (gamma/kT, J/A) space."""
    print("Figure 3: Phase diagram")
    rp = RadicalPairSystem(1, 0.5)

    cached = load_cached("fig3_data.json")
    if cached:
        gamma_ratios = np.array(cached["gamma_ratios"])
        JoA_vals = np.array(cached["JoA_vals"])
        Z = np.array(cached["u_star_map"])
    else:
        gamma_ratios = np.logspace(-2, 1, 30)
        JoA_vals = np.linspace(0.1, 1.0, 50)
        u_scan = np.linspace(-1.0, 2.5, 30)
        Z = np.zeros((len(gamma_ratios), len(JoA_vals)))

        total = len(gamma_ratios) * len(JoA_vals)
        done = 0
        for gi, gr in enumerate(gamma_ratios):
            gam = kT * gr
            for ji, joa in enumerate(JoA_vals):
                dy = rp.scan_u(1.0, joa, u_scan, gamma=gam, n_angles=9)
                Z[gi, ji] = u_scan[np.argmax(dy)]
                done += 1
                if done % 100 == 0:
                    print(f"      {done}/{total}...")

        save_cached({
            "gamma_ratios": gamma_ratios.tolist(),
            "JoA_vals": JoA_vals.tolist(),
            "u_star_map": Z.tolist(),
        }, "fig3_data.json")

    bif_gammas = []
    for gi in range(len(gamma_ratios)):
        du = np.max(np.abs(np.diff(Z[gi, :])))
        if du > 0.5:
            bif_gammas.append(gamma_ratios[gi])

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))
    im = ax.pcolormesh(JoA_vals, gamma_ratios, Z, cmap="RdBu_r",
                       shading="auto", vmin=-1.5, vmax=2.0)
    fig.colorbar(im, ax=ax, label="$u^*$", shrink=0.85)

    if bif_gammas:
        ax.axhline(min(bif_gammas), color="white", ls="--", lw=1.0, alpha=0.8)
        ax.axhline(max(bif_gammas), color="white", ls="--", lw=1.0, alpha=0.8)
        ax.text(0.95, min(bif_gammas) * 1.3,
                f"$\\gamma/k_T = {min(bif_gammas):.2f}$",
                color="white", fontsize=6.5, ha="right")
        ax.text(0.95, max(bif_gammas) * 0.75,
                f"$\\gamma/k_T = {max(bif_gammas):.2f}$",
                color="white", fontsize=6.5, ha="right")

    ax.fill_between(JoA_vals, 0.1, 10, color="lime", alpha=0.06)
    ax.text(0.12, 4.0, "Bio range", fontsize=6, color="green", alpha=0.6)

    ax.set_yscale("log")
    ax.set_xlabel("$J/A$")
    ax.set_ylabel(r"$\gamma / k_T$")
    ax.set_title("Phase diagram")

    fig.tight_layout()
    save(fig, "figure3")
    if bif_gammas:
        print(f"    Bifurcation window: [{min(bif_gammas):.2f}, {max(bif_gammas):.2f}]")


def figure4():
    """Fig 4: (a) Multi-nuclear u* vs J/A, (b) phi4 stability, (c) spectroscopy."""
    print("Figure 4: Multi-nuclear landscape")

    cached = load_cached("fig4_data.json")
    if cached:
        JoA_scan = np.array(cached["JoA_scan"])
        u_1nuc = np.array(cached["u_1nuc"])
        u_2nuc = np.array(cached["u_2nuc"])
        A_spec = np.array(cached["A_spec"])
        DY_half = np.array(cached["DY_half"])
        DY_one = np.array(cached["DY_one"])
        alpha_half = cached["alpha_half"]
        alpha_one = cached["alpha_one"]
    else:
        u_scan = np.linspace(-1.0, 2.5, 30)

        print("    (a) 1-nucleus bifurcation scan...")
        rp1 = RadicalPairSystem(1, 0.5)
        JoA_scan = np.linspace(0.2, 0.5, 25)
        u_1nuc = np.zeros(len(JoA_scan))
        for i, joa in enumerate(JoA_scan):
            dy = rp1.scan_u(1.0, joa, u_scan, n_angles=11)
            u_1nuc[i] = u_scan[np.argmax(dy)]

        print("    (a) 2-nucleus bifurcation scan (dim=16, slow)...")
        rp2 = RadicalPairSystem(2, 0.5)
        u_2nuc = np.zeros(len(JoA_scan))
        for i, joa in enumerate(JoA_scan):
            if i % 5 == 0:
                print(f"        {i}/{len(JoA_scan)}...")
            dy = rp2.scan_u(1.0, joa, u_scan, n_angles=9)
            u_2nuc[i] = u_scan[np.argmax(dy)]

        print("    (c) Power-law spectroscopy...")
        A_spec = np.array([0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
        DY_half = np.zeros(len(A_spec))
        DY_one = np.zeros(len(A_spec))

        rp_half = RadicalPairSystem(1, 0.5)
        rp_one = RadicalPairSystem(1, 1.0)
        for i, A in enumerate(A_spec):
            J = 0.25 * A
            dy_h = rp_half.scan_u(A, J, u_scan, n_angles=11)
            DY_half[i] = np.max(dy_h)
            dy_o = rp_one.scan_u(A, J, u_scan, n_angles=11)
            DY_one[i] = np.max(dy_o)

        c_h = np.polyfit(np.log10(A_spec), np.log10(DY_half), 1)
        c_o = np.polyfit(np.log10(A_spec), np.log10(DY_one), 1)
        alpha_half = float(c_h[0])
        alpha_one = float(c_o[0])

        save_cached({
            "JoA_scan": JoA_scan.tolist(),
            "u_1nuc": u_1nuc.tolist(),
            "u_2nuc": u_2nuc.tolist(),
            "A_spec": A_spec.tolist(),
            "DY_half": DY_half.tolist(),
            "DY_one": DY_one.tolist(),
            "alpha_half": alpha_half,
            "alpha_one": alpha_one,
        }, "fig4_data.json")

    fig = plt.figure(figsize=(7.2, 2.8))
    gs = GridSpec(1, 3, figure=fig, wspace=0.42)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(JoA_scan, u_1nuc, "b.-", lw=1.2, markersize=4,
             label="1 nuc (dim=8)")
    ax1.plot(JoA_scan, u_2nuc, "r.-", lw=1.2, markersize=4, alpha=0.8,
             label="2 nuc eq (dim=16)")
    ax1.axhline(LOG_PHI4, color="orange", ls="--", lw=0.8, alpha=0.8,
                label=r"$\log_{10}\varphi^4$")
    ax1.set_xlabel("$J/A$")
    ax1.set_ylabel("$u^*$")
    ax1.set_title("(a) Shelf position")
    ax1.legend(fontsize=6, loc="upper right")
    ax1.set_ylim(-1.5, 2.0)

    du_1 = np.max(np.abs(np.diff(u_1nuc)))
    du_2 = np.max(np.abs(np.diff(u_2nuc)))

    ax2 = fig.add_subplot(gs[0, 1])
    labels = ["1 nuc\n(I=1/2)", "2 nuc\n(equal)", "2 nuc\n(unequal)"]
    phi4_r = [0.992, 0.758, 0.952]
    delta_us = [round(du_1, 2), round(du_2, 2), round(du_2, 2)]
    bar_colors = ["#1f77b4", "#d62728", "#ff7f0e"]
    x = np.arange(len(labels))
    bars = ax2.bar(x, phi4_r, width=0.55, color=bar_colors, alpha=0.85,
                   edgecolor="black", linewidth=0.5)
    for i, (bar, du) in enumerate(zip(bars, delta_us)):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 r"$\Delta u$=" + str(du), ha="center", va="bottom",
                 fontsize=6.5)
    ax2.axhline(1.0, color="gray", ls=":", lw=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel(r"$\Delta Y(\varphi^4) / \Delta Y_{\max}$")
    ax2.set_title(r"(b) $\varphi^4$ stability")
    ax2.set_ylim(0, 1.18)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(A_spec, DY_half * 1e3, color="blue", s=25, zorder=5,
                label=f"I=1/2 ($|\\alpha|$={abs(alpha_half):.2f})")
    c_h = np.polyfit(np.log10(A_spec), np.log10(DY_half), 1)
    A_fit = np.linspace(0.25, 2.2, 50)
    ax3.plot(A_fit, 10 ** np.polyval(c_h, np.log10(A_fit)) * 1e3,
             "b--", alpha=0.4, lw=0.8)

    ax3.scatter(A_spec, DY_one * 1e3, color="red", s=25, marker="^",
                zorder=5, label=f"I=1 ($|\\alpha|$={abs(alpha_one):.2f})")
    c_o = np.polyfit(np.log10(A_spec), np.log10(DY_one), 1)
    ax3.plot(A_fit, 10 ** np.polyval(c_o, np.log10(A_fit)) * 1e3,
             "r--", alpha=0.4, lw=0.8)

    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("$A$ (mT)")
    ax3.set_ylabel(r"$\Delta Y_{\max} \times 10^3$")
    ax3.set_title("(c) Spectroscopic fingerprint")
    ax3.legend(fontsize=6, loc="lower left")

    save(fig, "figure4")
    print(f"    |alpha_half| = {abs(alpha_half):.3f}, |alpha_one| = {abs(alpha_one):.3f}")


def figure5():
    """Fig 5: Angular drift at high fields."""
    print("Figure 5: Angular drift")
    rp = RadicalPairSystem(1, 0.5)
    rp_off = RadicalPairSystem(1, 0.5)

    B_vals = np.linspace(0.5, 3.0, 20)
    theta_max_on = np.zeros(len(B_vals))
    theta_max_off = np.zeros(len(B_vals))

    u_fixed = 0.836
    kS = kT * 10 ** u_fixed

    print("    Proton ON...")
    for i, B in enumerate(B_vals):
        thetas, Y = rp.yield_vs_theta(1.0, 0.25, B_uT=B,
                                       kS=kS, kT=kT, gamma=gamma,
                                       n_angles=37)
        theta_max_on[i] = np.degrees(thetas[np.argmax(Y)])

    print("    Proton OFF...")
    for i, B in enumerate(B_vals):
        thetas, Y = rp_off.yield_vs_theta(0.0, 0.25, B_uT=B,
                                           kS=kS, kT=kT, gamma=gamma,
                                           n_angles=37)
        dy = np.max(Y) - np.min(Y)
        if dy < 1e-10:
            theta_max_off[i] = 0.0
        else:
            theta_max_off[i] = np.degrees(thetas[np.argmax(Y)])

    B_trans = None
    for i in range(1, len(theta_max_on)):
        if abs(theta_max_on[i] - theta_max_on[i - 1]) > 30:
            B_trans = B_vals[i]
            break

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(B_vals, theta_max_on, "o-", ms=5, lw=1.5, color="#1f77b4",
            label="Proton ON (A=1 mT)")
    ax.plot(B_vals, theta_max_off, "s--", ms=5, lw=1.5, color="#d62728",
            label="Proton OFF (A=0)")
    if B_trans:
        ax.axvline(B_trans, color="gray", ls=":", lw=1.0,
                   label=f"Transition B = {B_trans:.2f} mT")
    ax.set_xlabel("B (mT)")
    ax.set_ylabel(r"$\theta_{\max}$ (degrees)")
    ax.set_title("Angular drift: hyperfine-Zeeman crossover")
    ax.legend(fontsize=7)

    fig.tight_layout()
    save(fig, "figure5")
    if B_trans:
        print(f"    Transition at B = {B_trans:.2f} mT")


FIGURE_MAP = {
    "1": figure1,
    "2": figure2,
    "3": figure3,
    "4": figure4,
    "5": figure5,
}


if __name__ == "__main__":
    print("=" * 60)
    print("Generating publication figures")
    print("=" * 60)

    if len(sys.argv) > 1:
        figs = sys.argv[1:]
    else:
        figs = ["1", "2", "3", "4", "5"]

    for f in figs:
        if f in FIGURE_MAP:
            FIGURE_MAP[f]()
        else:
            print(f"Unknown figure: {f}")

    print(f"\nFigures saved to {FIG_DIR}/")
