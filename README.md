# mc_structure_explorer

> **Archival notice**: This repository is not actively maintained. It was developed to support a specific research publication and is provided as-is for reproducibility purposes.

Python package for exploring dopant configurations in crystalline structures using Metropolis Monte Carlo sampling. Supports Ewald electrostatic and MACE machine learning potentials, optional BFGS relaxation, parallel chain execution, and [Chemiscope](https://chemiscope.org) visualization.

## Installation

```bash
git clone https://github.com/jbouquiaux/mc_structure_explorer.git
cd mc_structure_explorer
pip install .
```

## Usage

```bash
run-mc --config config.yaml
```

Results are written to `results/results_*.json` and interactive [Chemiscope](https://chemiscope.org) files to `chemiscope/`. See [tests/config.yaml](tests/config.yaml) for a fully annotated configuration example.

## Citation

If you use this code, please cite the associated paper:

Julien Bouquiaux, Samuel Poncé, Yongchao Jia, Masayoshi Mikami, Xavier Gonze; Microenvironment of the Eu Interstitial in β-SiAlON:Eu2+ Green Phosphor. Chem. Mater. 28 July 2026; 38 (14): 7447–7456. https://doi.org/10.1021/acs.chemmater.6c01369
## License

MIT
