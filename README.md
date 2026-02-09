# radical-pair-shelves

Computational exploration of the radical pair mechanism sensitivity landscape.

## Paper
"Discontinuous bifurcation and nuclear spectroscopy in radical pair magnetoreception"
Andrei Ursachi (2026), submitted to Physical Review E.

## Results
This code reproduces all computational results reported in the paper:
1. Response shelves with universal curve collapse
2. Discontinuous bifurcation at J/A ≈ 0.328
3. Phase diagram in (γ/k_T, J/A) space
4. Multi-nuclear landscape reorganization
5. Angular drift at high fields
6. Nuclear spectroscopy through recombination asymmetry

## Requirements
- Python 3.8+
- numpy, scipy, matplotlib

## Usage
```bash
pip install -r requirements.txt
python generate_figures.py      # Generate all 5 paper figures
python run_all_experiments.py   # Run all 12 experiments (Table S1)
```

## Model
Haberkorn master equation with Lindblad dephasing for a radical pair
(2 electrons + N nuclear spins). Singlet yield computed via Liouvillian
superoperator inversion (no time-stepping).

### Spin Hamiltonian
```
H = H_Z + H_HF + H_ex
```
- **Zeeman**: H_Z = g·μ_B·B·(cosθ·S_z + sinθ·S_x) on both electrons
- **Hyperfine** (axial): H_HF = A·S_{1z}·I_z
- **Exchange**: H_ex = 2J·S₁·S₂

### Dynamics
Haberkorn master equation with Lindblad dephasing:
```
dρ/dt = -i[H, ρ] - (k_S/2){Q_S, ρ} - (k_T/2){Q_T, ρ} + γ·Σᵢ(S_{iz}·ρ·S_{iz} - ρ/4)
```

Singlet yield via Liouvillian inversion: Y_S = k_S · Tr[Q_S · (-L⁻¹) · ρ₀]

## License
MIT

## Contact
Andrei Ursachi — ORCID: 0009-0002-6114-5011
