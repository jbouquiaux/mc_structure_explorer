import sys
import os
import json
from pymatgen.core.structure import Structure
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
import chemiscope

def compute_beta_sialon_z(structure):
    """
    Compute the beta-SiAlON composition parameter z for a supercell,
    including Eu interstitials and their associated O removal.

    β-SiAlON formula: Si_{6-z}Al_z O_z N_{8-z} (+ Eu interstitials)

    Parameters:
        structure : pymatgen Structure object

    Returns:
        dict with:
            - z_per_FU : float, number of Al/O substitutions per formula unit
            - n_FU     : int, number of formula units in the supercell
            - n_Al     : int, total number of Al atoms
            - n_O      : int, total number of O atoms
            - n_Si     : int, total number of Si atoms
            - n_N      : int, total number of N atoms
            - n_Eu     : int, total number of Eu interstitials
            - O_deficit_per_Eu: expected number of O removed per Eu (should be ~2)
    """
    n_Al = sum(1 for s in structure if s.specie.symbol == "Al")
    n_O  = sum(1 for s in structure if s.specie.symbol == "O")
    n_Si = sum(1 for s in structure if s.specie.symbol == "Si")
    n_N  = sum(1 for s in structure if s.specie.symbol == "N")
    n_Eu = sum(1 for s in structure if s.specie.symbol == "Eu")


    z = n_Al/(n_Al+n_Si) * 6  # z per formula unit
    y = n_Eu/(n_Al+n_Si) * 6  # Eu per formula unit

    # Check Eu charge compensation (expect 2 O removed per Eu)
    if n_Eu != 0:
        expected_O = n_Al - 2 * n_Eu
        if n_O != expected_O:
            print(f"Warning: n_O={n_O}, expected {expected_O} based on {n_Eu} Eu interstitial(s)")

    return {
        "z": z,
        "y": y,
        "n_Al": n_Al,
        "n_O": n_O,
        "n_Si": n_Si,
        "n_N": n_N,
        "n_Eu": n_Eu,
    }


# -------------------------
# Utility functions
# -------------------------
def save_config(config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)

def generate_chemiscope_files(path, result):
    list_energies = [frame["energy"] for frame in result["trajectory"]]
    list_strus = [Structure.from_dict(frame["structure"]) for frame in result["trajectory"]]
    agg_scores = [frame["Al_O_aggregation_score"] for frame in result["trajectory"]]
    eu_scores = [frame["Eu_proximity_score"] for frame in result["trajectory"]]

    e = (np.array(list_energies))
    list_ase = [AseAtomsAdaptor.get_atoms(stru) for stru in list_strus]

    properties = {
        "index": {"target": "structure", "values": np.arange(len(list_ase)), "description": "structure index"},
        "energy per atom": {"target": "structure", "values": e, "units": "eV", "description": "potential energy (eV/atom)"},
        "Al-O aggregation score": {"target": "structure", "values": np.array(agg_scores), "description": "average number of O neighbors per Al"},
#        "Eu proximity score": {"target": "structure", "values": np.array(eu_scores), "description": "average proximity of Al/O to Eu"}
    }

    widget = chemiscope.show(frames=list_ase, properties=properties)
    widget.save(path)