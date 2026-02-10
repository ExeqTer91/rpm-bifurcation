"""Modal: Just the N=1000 convergence scan."""
import modal
import json
import numpy as np

app = modal.App("rpm-conv-1000")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "scipy")
vol = modal.Volume.from_name("rpm-shelves-results", create_if_missing=True)

SOLVER_CODE = r'''
import numpy as np
from scipy import linalg

PHI = (1 + np.sqrt(5)) / 2
B_DEFAULT = 0.05
g_e = 2.0023; muB = 9.2741e-24; hbar = 1.0546e-34
CONV = 1e-3 * g_e * muB / hbar
KT_DEFAULT = 1e6 / CONV
GAMMA_BIFURCATION = PHI**(-2) * KT_DEFAULT

_sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
_sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
_sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)

def _kron_list(ops):
    r = ops[0]
    for o in ops[1:]: r = np.kron(r, o)
    return r

def build_operators(n_nuclei, nuclear_spins=None):
    if nuclear_spins is None: nuclear_spins = [0.5] * n_nuclei
    dims = [2, 2] + [int(2 * s + 1) for s in nuclear_spins]
    td = int(np.prod(dims)); ns = len(dims)
    def embed(op, site):
        return _kron_list([op if i == site else np.eye(dims[i], dtype=complex) for i in range(ns)])
    S1 = [embed(_sx, 0), embed(_sy, 0), embed(_sz, 0)]
    S2 = [embed(_sx, 1), embed(_sy, 1), embed(_sz, 1)]
    I_ops = []
    for i in range(n_nuclei):
        site = 2 + i
        I_ops.append([embed(_sx, site), embed(_sy, site), embed(_sz, site)])
    S1dS2 = sum(S1[k] @ S2[k] for k in range(3))
    Id = np.eye(td, dtype=complex)
    singlet_e = np.zeros((4, 1), dtype=complex)
    singlet_e[1, 0] = 1 / np.sqrt(2); singlet_e[2, 0] = -1 / np.sqrt(2)
    QS_e = singlet_e @ singlet_e.conj().T
    d_nuc = td // 4; I_nuc = np.eye(d_nuc, dtype=complex)
    QS = np.kron(QS_e, I_nuc)
    return {"dim": td, "S1": S1, "S2": S2, "I": I_ops, "S1dS2": S1dS2,
            "QS": QS, "QT": Id - QS, "Id": Id, "nuclear_spins": nuclear_spins, "d_nuc": d_nuc}

def build_hamiltonian(ops, theta, A_vals, J, B=B_DEFAULT):
    H = B * (np.cos(theta) * (ops["S1"][2] + ops["S2"][2]) + np.sin(theta) * (ops["S1"][0] + ops["S2"][0]))
    for i, A in enumerate(A_vals): H += A * (ops["S1"][2] @ ops["I"][i][2])
    H += 2 * J * ops["S1dS2"]
    return H

def build_liouvillian(ops, H, kS, kT, gamma=0.0):
    dim = ops["dim"]; Id = ops["Id"]; QS = ops["QS"]; QT = ops["QT"]
    L = -1j * (np.kron(Id, H) - np.kron(H.T, Id))
    L -= (kS / 2) * (np.kron(Id, QS) + np.kron(QS.T, Id))
    L -= (kT / 2) * (np.kron(Id, QT) + np.kron(QT.T, Id))
    if gamma > 0:
        I_super = np.eye(dim * dim, dtype=complex)
        for S in [ops["S1"], ops["S2"]]:
            Sz = S[2]; L += gamma * (np.kron(Sz.T, Sz) - 0.25 * I_super)
    return L

def singlet_yield(ops, L, kS):
    QS = ops["QS"]; d_nuc = ops["d_nuc"]
    singlet_e = np.zeros((4, 1), dtype=complex)
    singlet_e[1, 0] = 1 / np.sqrt(2); singlet_e[2, 0] = -1 / np.sqrt(2)
    QS_e = singlet_e @ singlet_e.conj().T
    I_nuc = np.eye(d_nuc, dtype=complex)
    rho0 = np.kron(QS_e, I_nuc / d_nuc)
    rho_vec = rho0.flatten(order="F"); QS_vec = QS.flatten(order="F")
    try:
        x = linalg.solve(L, -rho_vec)
        return kS * np.real(QS_vec.conj() @ x)
    except linalg.LinAlgError: return 0.0

def compute_anisotropy(ops, A_vals, J, kS, kT, gamma=0.0, n_theta=11, B=B_DEFAULT):
    thetas = np.linspace(0, np.pi, n_theta)
    Y = np.zeros(n_theta)
    for i, th in enumerate(thetas):
        H = build_hamiltonian(ops, th, A_vals, J, B)
        L = build_liouvillian(ops, H, kS, kT, gamma)
        Y[i] = singlet_yield(ops, L, kS)
    return np.max(Y) - np.min(Y)

def find_u_star(ops, A_vals, J, kT, gamma, u_values, n_theta=11, B=B_DEFAULT):
    DY = np.zeros(len(u_values))
    for j, u in enumerate(u_values):
        kS = kT * 10**u
        DY[j] = compute_anisotropy(ops, A_vals, J, kS, kT, gamma, n_theta, B)
    idx = np.argmax(DY)
    return u_values[idx], DY[idx], DY
'''


@app.function(image=image, timeout=14400, memory=8192, cpu=2.0, volumes={"/results": vol})
def scan_1000():
    exec(SOLVER_CODE, globals())
    import time

    N = 1000
    kT = KT_DEFAULT
    gamma = GAMMA_BIFURCATION
    ops = build_operators(1, [0.5])
    u_values = np.linspace(-1.0, 2.5, 60)

    print(f"Scanning J/A with N={N} points...")
    t0 = time.time()
    JoA_vals = np.logspace(-2, 1, N)
    u_stars = np.zeros(N)

    for i, joa in enumerate(JoA_vals):
        if i % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i}/{N} ({elapsed:.0f}s elapsed)")
        _, _, DY = find_u_star(ops, [1.0], joa, kT, gamma, u_values, n_theta=11)
        u_stars[i] = u_values[np.argmax(DY)]

    du = np.abs(np.diff(u_stars))
    idx = int(np.argmax(du))
    delta_u = float(du[idx])
    JoA_crit = float((JoA_vals[idx] + JoA_vals[idx + 1]) / 2)

    dt = time.time() - t0
    result = {
        "N_points": N,
        "JoA_values": JoA_vals.tolist(),
        "u_stars": u_stars.tolist(),
        "JoA_crit": JoA_crit,
        "delta_u": delta_u,
        "u_pre": float(u_stars[idx]),
        "u_post": float(u_stars[idx + 1]),
        "time_seconds": round(dt, 1),
    }

    with open("/results/convergence_N1000.json", "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()

    print(f"\nN={N}: J/A_crit={JoA_crit:.4f}, delta_u={delta_u:.3f} ({dt:.0f}s)")
    print("Saved to /results/convergence_N1000.json")
    return result


@app.local_entrypoint()
def main():
    print("Running N=1000 convergence scan...")
    r = scan_1000.remote()
    print(f"\nResult: J/A_crit={r['JoA_crit']:.4f}, delta_u={r['delta_u']:.3f}")
    print(f"Time: {r['time_seconds']:.0f}s")

    with open("results_convergence_N1000.json", "w") as f:
        json.dump(r, f, indent=2)
    print("Saved locally: results_convergence_N1000.json")
