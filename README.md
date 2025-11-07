# MC Structure Explorer

A Python package for exploring crystalline structure configurations using Monte Carlo (MC) sampling with Metropolis acceptance criteria. Designed for investigating dopant distributions, substitutional defects, and interstitial atoms in materials using either Ewald electrostatic or MACE machine learning potentials.

## Features

- **Metropolis Monte Carlo Sampling**: Efficient exploration of configurational space for doped materials
- **Dual Energy Models**:
  - Ewald electrostatic model for fast screening
  - MACE (Machine Learning Accelerated Computational Engine) for accurate energy predictions
- **Flexible Structure Modifications**:
  - Substitutional dopants (e.g., Al/Si, O/N)
  - Interstitial atoms (e.g., rare earth dopants like Eu)
  - Multiple dopant types simultaneously
- **Smart Move Proposals**: Biased sampling to avoid unfavorable Al-O coordination
- **Structure Relaxation**: Optional BFGS relaxation with ASE
- **Parallel Execution**: Multi-chain runs with multiprocessing support
- **Automated Visualization**: Integration with Chemiscope for interactive 3D structure viewing
- **Temperature Annealing**: Linear or custom temperature schedules

## Installation

### Prerequisites

- Python ≥ 3.10
- CUDA-compatible GPU (optional, for MACE acceleration)

### Install from source

```bash
git clone https://github.com/yourusername/mc_structure_explorer.git
cd mc_structure_explorer
pip install .
```

### Install for development

```bash
pip install -e .[dev]
```

## Quick Start

### 1. Prepare a configuration file

Create a YAML configuration file (e.g., `config.yaml`):

```yaml
# Structure setup
base_structure_path: "path/to/primitive_cell.cif"
supercell_matrix: [2, 2, 2]

# Dopant configuration
n_Al: 4          # Number of Al substitutions (Si → Al)
n_O: 4           # Number of O substitutions (N → O)
n_Eu: 1          # Number of Eu interstitials
Eu_position: [0.25, 0.25, 0.25]  # Fractional coordinates

# MC parameters
calculation_type: "mace_small"  # or "ewald"
n_steps: 10000
n_runs: 4        # Number of parallel chains
seed_global: 42

# Temperature schedule
T_start: 0.5     # Initial effective temperature (eV)
T_end: 0.01      # Final effective temperature (eV)

# Sampling parameters
thin: 10         # Save every nth structure
burn_frac: 0.1   # Fraction of initial steps to discard

# Bias parameters
bias_strength: 1.0      # Penalty for Al-O coordination
eu_bias_strength: 1.0   # Penalty for Eu proximity to other Eu
bias_on_plane: 0        # Penalty for atoms on specific planes

# Interstitial move probability
eu_move_prob: 0.2       # Probability of moving interstitial vs swapping

# Structure relaxation (optional)
relax: true
relax_fmax: 0.1         # Force convergence criterion (eV/Å)
relax_steps: 200        # Maximum relaxation steps
follow_relaxed: false   # Use relaxed or unrelaxed structure for next move

# Output directories
result_dir: "results"
chemiscope_dir: "chemiscope"

# Misc
verbose: true
same_initial_structure: false  # All chains start from same initial config
```

### 2. Run Monte Carlo exploration

```bash
run-mc --config config.yaml
```

This will:
- Run multiple MC chains in parallel
- Save trajectory data to `results/results_*.json`
- Generate interactive Chemiscope visualizations in `chemiscope/`
- Save the configuration for reproducibility

### 3. Analyze results

```python
import json

# Load results
results = json.load(open('results/results_0.json'))

# Access trajectory data
for step in results["trajectory"]:
    energy = step["energy_relaxed"]  # or "energy_unrelaxed"
    structure = step["structure"]    # pymatgen Structure dict
    accepted = step["accepted"]      # Was this move accepted?

# Get final structures (after burn-in)
final_structures = results["final_structures"]

# Access run parameters
params = results["run_params"]
```

## Python API

You can also use the package programmatically:

```python
from pymatgen.core import Structure
from mc_structure_explorer.mc_engine import MonteCarloEngine
import numpy as np

# Load base structure
structure = Structure.from_file("primitive.cif")
structure.make_supercell([2, 2, 2])

# Define substitutions
substitutions = [
    {"replaced_atom": "Si", "dopant_atom": "Al"},
    {"replaced_atom": "Si", "dopant_atom": "Al"},
    {"replaced_atom": "N", "dopant_atom": "O"},
    {"replaced_atom": "N", "dopant_atom": "O"}
]

# Define interstitial (optional)
interstitial = {"Eu": [0.25, 0.25, 0.25]}

# Create MC engine
mc = MonteCarloEngine(
    base_structure=structure,
    substitution_list=substitutions,
    interstitial_site=interstitial,
    calculation_type="mace_small",
    seed=42
)

# Run MC chain
T_schedule = np.linspace(0.5, 0.01, 10000)
mc.run_chain(
    n_steps=10000,
    T_schedule=T_schedule,
    thin=10,
    burn_frac=0.1,
    relax=True,
    follow_relaxed=False
)

# Save results
mc.save_results("results.json")
```

## Advanced Features

### Assessing Relaxation Importance

Determine whether structure relaxation significantly affects your MC sampling:

```python
import json
from scipy.stats import pearsonr, spearmanr

# Load results from a run with relax=True
results = json.load(open('results/results_0.json'))

# Extract energies
e_unrelaxed = [step["energy_unrelaxed"] for step in results["trajectory"]]
e_relaxed = [step["energy_relaxed"] for step in results["trajectory"]]

# Analyze correlation
corr_pearson, _ = pearsonr(e_unrelaxed, e_relaxed)
corr_spearman, _ = spearmanr(e_unrelaxed, e_relaxed)

print(f"Pearson correlation: {corr_pearson:.3f}")
print(f"Spearman correlation: {corr_spearman:.3f}")

# High correlation (>0.95) means relaxation doesn't change structure rankings
# Low correlation (<0.8) means relaxation is important
```

### Custom Temperature Schedules

```python
# Exponential cooling
T_schedule = 0.5 * np.exp(-np.linspace(0, 5, 10000))

# Step-wise annealing
T_schedule = np.concatenate([
    np.full(2000, 0.5),
    np.full(3000, 0.1),
    np.full(5000, 0.01)
])

# Simulated annealing with reheating
T_schedule = 0.5 * (1 + np.cos(np.linspace(0, 4*np.pi, 10000))) / 2 + 0.01
```

### Chemiscope Visualization

Interactive 3D visualization files are automatically generated. Open `chemiscope/dataset_0.json` in [Chemiscope](https://chemiscope.org/) to:

- Visualize structure trajectories in 3D
- Color by energy, acceptance, or custom properties
- Explore structure-property relationships
- Export structures for further analysis

## Configuration Reference

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `base_structure_path` | str | Path to primitive cell structure file | Required |
| `supercell_matrix` | list | Supercell expansion (e.g., [2,2,2]) | Required |
| `n_Al`, `n_O`, `n_Eu` | int | Number of dopant atoms | Required |
| `calculation_type` | str | "ewald" or "mace_small" | Required |
| `n_steps` | int | Number of MC steps | Required |
| `T_start`, `T_end` | float | Temperature schedule endpoints (eV) | Required |
| `thin` | int | Save every nth structure | 10 |
| `burn_frac` | float | Fraction of burn-in | 0.1 |
| `bias_strength` | float | Al-O aggregation penalty | 1.0 |
| `eu_bias_strength` | float | Eu-Eu proximity penalty | 1.0 |
| `relax` | bool | Enable structure relaxation | false |
| `relax_fmax` | float | Relaxation force threshold (eV/Å) | 0.1 |
| `follow_relaxed` | bool | Propose moves from relaxed structures | false |

## How It Works

### Metropolis Algorithm

1. **Initialization**: Start from a random configuration with specified dopants
2. **Move Proposal**:
   - For substitutional: Swap two atoms of different elements
   - For interstitial: Translate atom by random displacement
   - Biased selection to avoid high-energy Al-O bonds
3. **Energy Calculation**: Compute energy using Ewald or MACE
4. **Acceptance**: Accept if ΔE < 0 or with probability exp(-ΔE/T)
5. **Repeat**: Continue for n_steps with temperature annealing

### Energy Models

**Ewald Electrostatic**:
- Fast screening tool (~ms per structure)
- Good for initial exploration
- Limited accuracy for covalent systems

**MACE (Machine Learning)**:
- High accuracy (~DFT-level)
- Slower (~100ms per structure)
- Better for final refinement
- Requires GPU for best performance

## Examples

See the `tests/` directory for complete examples:

- `tests/config.yaml`: Example configuration
- `tests/custom_analysis.ipynb`: Analysis notebook with relaxation assessment
- `tests/prim.cif`: Example β-SiAlON primitive cell

## Citation

If you use this package, please cite:

```bibtex
@software{mc_structure_explorer,
  title = {MC Structure Explorer: Monte Carlo Sampling for Doped Materials},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/mc_structure_explorer}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Email: you@example.com

## Acknowledgments

- Built with [Pymatgen](https://pymatgen.org/) for structure manipulation
- Energy predictions with [MACE](https://github.com/ACEsuit/mace)
- Visualization with [Chemiscope](https://chemiscope.org/)
