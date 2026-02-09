"""
Core RPM simulation engine.

Radical pair mechanism (RPM) model for biological magnetoreception.
Two electron spins coupled to N nuclear spins.

Haberkorn master equation with Lindblad dephasing:
    dρ/dt = -i[H, ρ] - (kS/2){QS, ρ} - (kT/2){QT, ρ}
            + γ·Σ(Siz·ρ·Siz - ρ/4)

Singlet yield via Liouvillian superoperator inversion (no time-stepping):
    YS = kS · Tr[QS · (-L⁻¹) · ρ₀]

Unit system: "natural mT" units
    B, A, J in mT (used directly in Hamiltonian)
    kT, kS in mT units: physical kT = 1e6 s⁻¹ => kT = 5.68e-3 mT
    Conversion: ω_rad = value_mT × 1e-3 × g_e × μ_B / ℏ

CRITICAL: Uses AXIAL hyperfine coupling (S1z·Iz), NOT isotropic (S1·I).
Axial symmetry breaking is essential for non-vanishing anisotropy.
"""

import numpy as np
from scipy import linalg

PHI = (1 + np.sqrt(5)) / 2
PHI4 = PHI ** 4
LOG_PHI4 = np.log10(PHI4)

g_e = 2.0023
muB = 9.2741e-24
hbar = 1.0546e-34
CONV = 1e-3 * g_e * muB / hbar

B_DEFAULT = 0.05
KT_DEFAULT = 1e6 / CONV
GAMMA_DEFAULT = KT_DEFAULT


def _spin_matrices(spin):
    d = int(2 * spin + 1)
    m_vals = np.arange(spin, -spin - 1, -1)
    sz = np.diag(m_vals).astype(complex)
    sp = np.zeros((d, d), dtype=complex)
    for i in range(d - 1):
        m = m_vals[i + 1]
        sp[i, i + 1] = np.sqrt(spin * (spin + 1) - m * (m + 1))
    sm = sp.T.copy()
    sx = (sp + sm) / 2
    sy = (sp - sm) / (2j)
    return sx, sy, sz


_sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
_sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
_sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)


def _kron_list(ops):
    r = ops[0]
    for o in ops[1:]:
        r = np.kron(r, o)
    return r


class RadicalPairSystem:
    """Radical pair spin system with 2 electrons + N nuclear spins."""

    def __init__(self, n_nuc=1, spin_I=0.5):
        if isinstance(spin_I, (int, float)):
            self.nuclear_spins = [spin_I] * n_nuc
        else:
            self.nuclear_spins = list(spin_I)
        self.n_nuc = n_nuc
        self.dims = [2, 2] + [int(2 * s + 1) for s in self.nuclear_spins]
        self.dim = int(np.prod(self.dims))
        self.d_nuc = self.dim // 4
        self._build_operators()

    def _build_operators(self):
        ns = len(self.dims)

        def embed(op, site):
            return _kron_list(
                [op if i == site else np.eye(self.dims[i], dtype=complex)
                 for i in range(ns)]
            )

        def local_ops(site):
            if self.dims[site] == 2:
                return [_sx, _sy, _sz]
            spin_val = self.nuclear_spins[site - 2]
            return list(_spin_matrices(spin_val))

        self.S1 = [embed(_sx, 0), embed(_sy, 0), embed(_sz, 0)]
        self.S2 = [embed(_sx, 1), embed(_sy, 1), embed(_sz, 1)]

        self.I_ops = []
        for i in range(self.n_nuc):
            site = 2 + i
            lo = local_ops(site)
            self.I_ops.append([embed(lo[k], site) for k in range(3)])

        self.S1dS2 = sum(self.S1[k] @ self.S2[k] for k in range(3))
        self.Id = np.eye(self.dim, dtype=complex)

        singlet_e = np.zeros((4, 1), dtype=complex)
        singlet_e[1, 0] = 1 / np.sqrt(2)
        singlet_e[2, 0] = -1 / np.sqrt(2)
        QS_e = singlet_e @ singlet_e.conj().T
        I_nuc = np.eye(self.d_nuc, dtype=complex)
        self.QS = np.kron(QS_e, I_nuc)
        self.QT = self.Id - self.QS
        self.rho0 = np.kron(QS_e, I_nuc / self.d_nuc)

    def build_hamiltonian(self, theta, A_vals, J, B=B_DEFAULT):
        """Build spin Hamiltonian at field angle theta.

        H = H_Z + H_HF + H_ex
          = g·μB·B·(cosθ·Sz + sinθ·Sx) + Σ A_i·S1z·Iz_i + 2J·S1·S2
        """
        S1x, S1y, S1z = self.S1
        S2x, S2y, S2z = self.S2
        H = B * (np.cos(theta) * (S1z + S2z) + np.sin(theta) * (S1x + S2x))
        for i, A in enumerate(A_vals):
            Iz = self.I_ops[i][2]
            H += A * (S1z @ Iz)
        H += 2 * J * self.S1dS2
        return H

    def build_liouvillian(self, H, kS, kT, gamma=0.0):
        """Build Liouvillian superoperator.

        L·vec(ρ) = -i(I⊗H - Hᵀ⊗I)·vec(ρ)
                   - (kS/2)(I⊗QS + QSᵀ⊗I)·vec(ρ)
                   - (kT/2)(I⊗QT + QTᵀ⊗I)·vec(ρ)
                   + γ·Σ(Szᵀ⊗Sz - I/4)·vec(ρ)
        """
        Id = self.Id
        L = -1j * (np.kron(Id, H) - np.kron(H.T, Id))
        L -= (kS / 2) * (np.kron(Id, self.QS) + np.kron(self.QS.T, Id))
        L -= (kT / 2) * (np.kron(Id, self.QT) + np.kron(self.QT.T, Id))
        if gamma > 0:
            I_super = np.eye(self.dim ** 2, dtype=complex)
            for S in [self.S1, self.S2]:
                Sz = S[2]
                L += gamma * (np.kron(Sz.T, Sz) - 0.25 * I_super)
        return L

    def singlet_yield(self, L, kS):
        """Compute singlet yield via Liouvillian inversion.

        YS = kS · Tr[QS · (-L⁻¹) · ρ₀]
        """
        rho_vec = self.rho0.flatten(order="F")
        QS_vec = self.QS.flatten(order="F")
        try:
            x = linalg.solve(L, -rho_vec)
            return kS * np.real(QS_vec.conj() @ x)
        except linalg.LinAlgError:
            return 0.0

    def compute_anisotropy(self, A_mT, J_mT, B_uT=B_DEFAULT,
                           kS=None, kT=KT_DEFAULT, gamma=GAMMA_DEFAULT,
                           n_angles=18):
        """Compute magnetic anisotropy ΔY = max(YS) - min(YS) over angles.

        Parameters
        ----------
        A_mT : float or list
            Hyperfine coupling(s) in mT. If float, applied to all nuclei.
        J_mT : float
            Exchange coupling in mT.
        B_uT : float
            External field in mT (default 0.05 mT = 50 μT).
        kS : float
            Singlet recombination rate (mT units). If None, equals kT.
        kT : float
            Triplet recombination rate (mT units).
        gamma : float
            Lindblad dephasing rate (mT units).
        n_angles : int
            Number of angles in [0, π].

        Returns
        -------
        float : ΔY = max_θ(YS) - min_θ(YS)
        """
        if kS is None:
            kS = kT
        if isinstance(A_mT, (int, float)):
            A_vals = [A_mT] * self.n_nuc
        else:
            A_vals = list(A_mT)

        thetas = np.linspace(0, np.pi, n_angles)
        Y = np.zeros(n_angles)
        for i, th in enumerate(thetas):
            H = self.build_hamiltonian(th, A_vals, J_mT, B_uT)
            L = self.build_liouvillian(H, kS, kT, gamma)
            Y[i] = self.singlet_yield(L, kS)
        return np.max(Y) - np.min(Y)

    def scan_u(self, A_mT, J_mT, u_range, B_uT=B_DEFAULT,
               kT=KT_DEFAULT, gamma=GAMMA_DEFAULT, n_angles=18):
        """Scan asymmetry parameter u = log10(kS/kT).

        Returns
        -------
        dy : ndarray, shape (len(u_range),)
            Anisotropy ΔY at each u value.
        """
        if isinstance(A_mT, (int, float)):
            A_vals = [A_mT] * self.n_nuc
        else:
            A_vals = list(A_mT)

        dy = np.zeros(len(u_range))
        for j, u in enumerate(u_range):
            kS = kT * 10 ** u
            dy[j] = self.compute_anisotropy(A_vals, J_mT, B_uT, kS, kT,
                                            gamma, n_angles)
        return dy

    def find_shelf_position(self, dy_array, u_range):
        """Find u* = argmax(ΔY) — the shelf position."""
        idx = int(np.argmax(dy_array))
        return u_range[idx], dy_array[idx]

    def yield_vs_theta(self, A_mT, J_mT, B_uT=B_DEFAULT,
                       kS=None, kT=KT_DEFAULT, gamma=GAMMA_DEFAULT,
                       n_angles=37):
        """Compute YS(θ) for angles in [0, π].

        Returns
        -------
        thetas : ndarray
        Y : ndarray
        """
        if kS is None:
            kS = kT
        if isinstance(A_mT, (int, float)):
            A_vals = [A_mT] * self.n_nuc
        else:
            A_vals = list(A_mT)

        thetas = np.linspace(0, np.pi, n_angles)
        Y = np.zeros(n_angles)
        for i, th in enumerate(thetas):
            H = self.build_hamiltonian(th, A_vals, J_mT, B_uT)
            L = self.build_liouvillian(H, kS, kT, gamma)
            Y[i] = self.singlet_yield(L, kS)
        return thetas, Y


if __name__ == "__main__":
    print("Smoke test: 1-nucleus system (dim=8)")
    rp = RadicalPairSystem(n_nuc=1, spin_I=0.5)
    print(f"  Hilbert dim = {rp.dim}")

    u_test = np.linspace(-1.0, 2.5, 20)
    dy = rp.scan_u(1.0, 0.25, u_test, n_angles=11)
    u_star, dy_max = rp.find_shelf_position(dy, u_test)
    print(f"  u* = {u_star:.3f}, ΔY_max = {dy_max:.6f}")
    print(f"  log10(φ⁴) = {LOG_PHI4:.3f}")
    print("OK")
