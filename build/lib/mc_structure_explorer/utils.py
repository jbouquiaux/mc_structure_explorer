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

    # β-Si3N4 formula unit has 6 Si + 8 N = 14 atoms per FU
    n_atoms_FU = 14
    n_FU = len(structure) / n_atoms_FU

    # z is Al/O substitutions per FU
    z_per_FU = n_Al / n_FU

    # Check Eu charge compensation (expect 2 O removed per Eu)
    expected_O = n_Al - 2 * n_Eu
    if n_O != expected_O:
        print(f"Warning: n_O={n_O}, expected {expected_O} based on {n_Eu} Eu interstitial(s)")

    return {
        "z_per_FU": z_per_FU,
        "n_FU": n_FU,
        "n_Al": n_Al,
        "n_O": n_O,
        "n_Si": n_Si,
        "n_N": n_N,
        "n_Eu": n_Eu,
        "O_deficit_per_Eu": n_Al - n_O
    }
