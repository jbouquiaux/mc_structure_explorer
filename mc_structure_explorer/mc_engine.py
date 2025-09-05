import random
import time
import numpy as np
from pymatgen.analysis.energy_models import EwaldElectrostaticModel
from pymatgen.io.ase import AseAtomsAdaptor
from mace.calculators import mace_mp
import json
from mc_structure_explorer.utils import compute_beta_sialon_z

class MonteCarloEngine:
    """
    Unified Monte Carlo engine for structure exploration using Metropolis sampling.

    This class encapsulates initialization, move proposals, energy calculation, and the Metropolis algorithm.
    Supports both Ewald and MACE energy models, and single or pair moves for dopant atoms.

    Attributes:
        base_structure (Structure): The starting pymatgen Structure object.
        substitution_list (list): List of dicts, each with 'replaced_atom' and 'dopant_atom'.
        calculation_type (str): Either 'ewald' or 'mace_small'.
        seed (int or None): Random seed for reproducibility.
        target_counts (dict): Target counts for each element after substitution.
        replacement_map (dict): Maps dopant element to replaced host element.
        ewald_model (EwaldElectrostaticModel): Ewald energy model instance.
        calc_mace (mace_mp): MACE energy model instance.
        results (dict or None): Stores results after running MC chain.
    """

    def __init__(self, base_structure, substitution_list, interstitial_site=None,
                 calculation_type="ewald", seed=None):
        """
        Initialize the MonteCarloEngine.

        Args:
            base_structure (Structure): The starting pymatgen Structure object.
            substitution_list (list): List of dicts, each with 'replaced_atom' and 'dopant_atom'.
            interstitial_sites (list of dicts or None): Optional list of dicts mapping element symbols to lists of fractional coordinates for interstitial atoms.
            calculation_type (str): Either 'ewald' or 'mace_small'.
            seed (int or None): Random seed for reproducibility.
        """
        self.base_structure = base_structure
        self.substitution_list = substitution_list
        self.interstitial = interstitial_site  # store interstitials for later use
        self.calculation_type = calculation_type
        self.seed = seed

        # Derive target counts from substitution_list for conservation checks
        self.target_counts = {}
        self.replacement_map = {}  # Maps dopant -> replaced host element (first matching)
        for sub in self.substitution_list:
            repl = sub["replaced_atom"]
            dop = sub["dopant_atom"]
            # Count dopants and replaced atoms we expect in the system
            self.target_counts[dop] = self.target_counts.get(dop, 0) + 1
            self.target_counts[repl] = self.target_counts.get(repl, 0)  # Ensure key exists
            # Store mapping (assumes each dopant type maps to single replaced type)
            self.replacement_map[dop] = repl

        self.ewald_model = EwaldElectrostaticModel()
        self.calc_mace = mace_mp(model="small")
        if self.seed is not None:
            random.seed(self.seed)
        self.results = None

    def generate_initial_structure(self):
        """
        Generate a random initial structure with the specified dopants.

        Returns:
            Structure: pymatgen Structure object with substitutions applied.
        """
        stru = self.base_structure.copy()
        indices_by_element = {}
        # Map each element to its site indices
        for i, site in enumerate(stru.sites):
            el = site.specie.symbol
            indices_by_element.setdefault(el, []).append(i)
        chosen_indices = set()
        # Apply substitutions
        for sub in self.substitution_list:
            replaced, dopant = sub["replaced_atom"], sub["dopant_atom"]
            avail = [idx for idx in indices_by_element.get(replaced, []) if idx not in chosen_indices]
            if not avail:
                raise RuntimeError(f"No available site for {replaced}")
            idx = random.choice(avail)
            chosen_indices.add(idx)
            stru.replace(idx, dopant)

        # Add interstitial atoms if defined at init
        if self.interstitial:
            element = list(self.interstitial.keys())[0]
            coords = self.interstitial[element]
            stru.append(element, coords, coords_are_cartesian=False)
        return stru

    def _count_element(self, structure, symbol):
        """
        Count the number of sites with a given element symbol in the structure.

        Args:
            structure (Structure): pymatgen Structure object.
            symbol (str): Element symbol to count.

        Returns:
            int: Number of sites with the specified element.
        """
        return sum(1 for s in structure if s.specie.symbol == symbol)

    def _sites_by_element(self, structure, symbol):
        """
        Get list of site indices containing a given element symbol.

        Args:
            structure (Structure): pymatgen Structure object.
            symbol (str): Element symbol to search for.

        Returns:
            list: List of site indices containing the element.
        """
        return [i for i, s in enumerate(structure) if s.specie.symbol == symbol]

    def _swap_sites(self, structure, i, j):
        """
        Swap elements at site indices i and j in a copy of the structure.

        Args:
            structure (Structure): pymatgen Structure object.
            i (int): Index of first site.
            j (int): Index of second site.

        Returns:
            Structure: New Structure instance with swapped elements.
        """
        st = structure.copy()
        el_i = st[i].specie.symbol
        el_j = st[j].specie.symbol
        st.replace(i, el_j)
        st.replace(j, el_i)
        return st

    def propose_move(self, structure, rcut=3.0, bias_strength=1, eu_bias_strength=1,
                    return_info=False, interstitial_move_prob=0.2):
        """
        Propose a single move: either displace Eu (interstitial move) or swap dopant ↔ host.
        Adds bias for Al–O aggregation AND for O/Al proximity to Eu.

        Args:
            structure (Structure): Current pymatgen Structure object.
            rcut (float): Cutoff radius for neighbor counting (Å).
            bias_strength (float): Bias strength toward Al–O aggregation.
            eu_bias_strength (float): Bias strength toward placing O/Al near Eu.
            return_info (bool): If True, also returns move details for verbose output.
            interstitial_move_prob (float): Probability of choosing Eu displacement.

        Returns:
            Structure or tuple: New structure (and move info if return_info=True).
        """

        st = structure.copy()

        # ---- Interstitial move (Eu displacement) ----
        if self.interstitial and random.random() < interstitial_move_prob:
            element = list(self.interstitial.keys())[0]  # assume only Eu for now
            indices = [i for i, site in enumerate(st) if site.specie.symbol == element]
            if indices:
                idx = random.choice(indices)
                old_frac = st[idx].frac_coords.copy()
                new_frac = old_frac.copy()
                dz = (np.random.rand() - 0.5) * 0.1  # ±0.05 random displacement
                new_frac[2] = (new_frac[2] + dz) % 1.0
                st.replace(idx, element, coords=new_frac, coords_are_cartesian=False)
                move_info = {
                    "type": "interstitial_move",
                    "element": element,
                    "site_idx": idx,
                    "old_frac": old_frac.tolist(),
                    "new_frac": new_frac.tolist()
                }
                return (st, move_info) if return_info else st

        # ---- Dopant swap ----
        dopant_sites = []
        for sub in self.substitution_list:
            dopant = sub["dopant_atom"]
            host = sub["replaced_atom"]
            for i, site in enumerate(st):
                if site.specie.symbol == dopant:
                    dopant_sites.append((i, dopant, host))
        if not dopant_sites:
            return (st, None) if return_info else st

        idx, dopant_symbol, host_symbol = random.choice(dopant_sites)
        candidate_sites = [i for i, s in enumerate(st) if s.specie.symbol == host_symbol and i != idx]
        if not candidate_sites:
            return (st, None) if return_info else st

        # Helper: count Al–O pairs around a site
        def al_o_neighbors(site_idx, structure):
            s = structure[site_idx]
            neighbors = 0
            for j, other in enumerate(structure):
                if j == site_idx:
                    continue
                if (s.specie.symbol == "Al" and other.specie.symbol == "O") or \
                (s.specie.symbol == "O" and other.specie.symbol == "Al"):
                    if structure.get_distance(site_idx, j) <= rcut:
                        neighbors += 1
            return neighbors

        # Helper: Eu bias — inverse distance of site to nearest Eu
        def eu_proximity(site_idx, structure, cutoff=5.0):
            eu_sites = [i for i, s in enumerate(structure) if s.specie.symbol == "Eu"]
            if not eu_sites:
                return 0.0
            min_dist = min(structure.get_distance(site_idx, eu_idx) for eu_idx in eu_sites)
            return 1.0 / (min_dist + 1e-6) if min_dist < cutoff else 0.0

        weights = []
        neighbor_counts = []
        eu_scores = []
        for target_idx in candidate_sites:
            temp_st = self._swap_sites(st, idx, target_idx)
            n_neighbors = al_o_neighbors(target_idx, temp_st)
            eu_score = eu_proximity(target_idx, temp_st)

            neighbor_counts.append(n_neighbors)
            eu_scores.append(eu_score)

            # combined weight: Al–O bias * Eu bias
            weight = np.exp(bias_strength * n_neighbors + eu_bias_strength * eu_score)
            weights.append(weight)

        weights = np.array(weights)
        weights /= weights.sum()

        chosen_idx = np.random.choice(candidate_sites, p=weights)
        bias_weight = weights[candidate_sites.index(chosen_idx)]
        n_neighbors = neighbor_counts[candidate_sites.index(chosen_idx)]
        eu_score = eu_scores[candidate_sites.index(chosen_idx)]

        new_st = self._swap_sites(st, idx, chosen_idx)
        move_info = {
            "type": "dopant_swap",
            "from_idx": idx,
            "to_idx": chosen_idx,
            "from_element": dopant_symbol,
            "to_element": host_symbol,
            "bias_weight": bias_weight,
            "al_o_neighbors": n_neighbors,
            "eu_proximity": eu_score
        }
        return (new_st, move_info) if return_info else new_st


    def get_energy(self, struct):
        """
        Compute the energy per atom for the given structure.

        Args:
            struct (Structure): pymatgen Structure object.

        Returns:
            float: Energy per atom (eV).
        """
        if self.calculation_type == "ewald":
            # Assign oxidation states for Ewald calculation
            struct.add_oxidation_state_by_element({"Eu": 2, "Si": 4, "Al": 3, "N": -3, "O": -2})
            return self.ewald_model.get_energy(struct) / len(struct)
        elif self.calculation_type == "mace_small":
            atoms = AseAtomsAdaptor.get_atoms(struct)
            atoms.set_calculator(self.calc_mace)
            return atoms.get_potential_energy() / len(atoms)
        else:
            raise ValueError(f"Unknown calculation_type {self.calculation_type}")

    def al_o_aggregation_score(self, structure, rcut=3.0, max_coord=4):
        """
        Fraction of possible Al–O bonds that exist.

        0.0 → Al and O far apart
        1.0 → Every Al has max_coord O neighbors within rcut
        """
        al_indices = [i for i, s in enumerate(structure) if s.specie.symbol == "Al"]
        o_indices  = [i for i, s in enumerate(structure) if s.specie.symbol == "O"]

        if not al_indices or not o_indices:
            return 0.0

        total_neighbors = 0
        for i in al_indices:
            neighbors = structure.get_neighbors(structure[i], rcut)
            total_neighbors += sum(1 for n in neighbors if n.specie.symbol == "O")

        # Normalize: average per Al / max coordination
        score = (total_neighbors / len(al_indices)) / max_coord
        return min(score, 1.0)

    def eu_proximity_score(self, structure, cutoff=5.0):
        """
        Average inverse-distance of Al/O atoms to the single Eu in the structure.
        0 → all Al/O far from Eu
        Higher → more Al/O close to Eu
        """
        eu_idx = [i for i, s in enumerate(structure) if s.specie.symbol == "Eu"]
        if not eu_idx:
            return 0.0
        ao_indices = [i for i, s in enumerate(structure) if s.specie.symbol in ("Al", "O")]
        if not ao_indices:
            return 0.0

        score_sum = 0.0
        for i in ao_indices:
            d = structure.get_distance(i, eu_idx[0])
            if d < cutoff:
                score_sum += 1.0 / (d + 1e-6)

        return score_sum / len(ao_indices)
    
    def run_chain(
        self,
        n_steps,
        T_schedule,
        thin=10,
        burn_frac=0.1,
        bias_strength=1.0,
        eu_bias_strength=1.0,
        interstitial_move_prob=0.2,
        progress_callback=None,
        verbose=False,
        initial_structure=None  
    ):
        """
        Run a Metropolis Monte Carlo chain.

        Args:
            n_steps (int): Number of MC steps.
            T_schedule (list): List of effective temperatures for each step.
            thin (int): Thinning interval for trajectory storage.
            burn_frac (float): Fraction of trajectory to discard as burn-in.
            bias_strength (float): Strength of bias towards Al-O aggregation.
            eu_bias_strength (float): Strength of bias towards placing O/Al near Eu.
            verbose (bool): If True, print detailed info about each move.
            initial_structure (Structure or None): If provided, use this as the starting structure.

        Returns:
            dict: Results including trajectory, acceptance ratio, and timing.
        """
        if verbose:
            print(f"Starting Metropolis chain with n_steps={n_steps}")

        start_time = time.time()

        # Use provided initial structure, else generate one
        if initial_structure is not None:
            current = initial_structure.copy()
        else:
            current = self.generate_initial_structure()

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        # Nomenclature infos
        z_info = compute_beta_sialon_z(current)

        if verbose:
            print(f"z per formula unit: {z_info['z_per_FU']:.2f}")
            print(f"Number of formula units in supercell: {z_info['n_FU']:.1f}")
            print(
                f"Counts: "
                f"Si={z_info['n_Si']}, "
                f"N={z_info['n_N']}, "
                f"Al={z_info['n_Al']}, "
                f"O={z_info['n_O']}, "
                f"Eu={z_info['n_Eu']}"
            )
            print(f"Total number of atoms in supercell: {len(current)}")

            if z_info["n_Eu"] > 0:
                print(
                    f"Eu charge compensation check: "
                    f"O deficit per Eu = {z_info['O_deficit_per_Eu'] / z_info['n_Eu']:.2f} "
                    f"(expected ~2)"
                )
            else:
                print("No Eu interstitials in this structure.")

        E_current = self.get_energy(current)
        trajectory = []
        accepted = 0
        proposed = 0

        for step in range(n_steps):
            if progress_callback is not None:
                progress_callback(step)
            T_eff = T_schedule[step]
            proposed += 1
            t_prop_start = time.time()

            # Propose move with info
            proposal, move_info = self.propose_move(
                current, rcut=3.0, bias_strength=bias_strength,
                eu_bias_strength=eu_bias_strength,
                interstitial_move_prob=interstitial_move_prob,
                return_info=True
            )
            E_prop = self.get_energy(proposal)
            Al_O_aggregation_score = self.al_o_aggregation_score(current, rcut=3.0)
            Eu_proximity_score = self.eu_proximity_score(current, cutoff=5.0)
            t_prop_end = time.time()
            dE = E_prop - E_current
            accepted_move = False

            # Metropolis acceptance criterion
            if dE <= 0.0 or random.random() < np.exp(-dE / T_eff):
                current = proposal
                E_current = E_prop
                accepted += 1
                accepted_move = True

            if step % thin == 0:
                # store instead as dict.
                trajectory.append({
                    "energy": E_current,
                    "structure": current.copy().as_dict(),
                    "Al_O_aggregation_score": Al_O_aggregation_score,
                    "Eu_proximity_score": Eu_proximity_score
                })

            # Verbose output
            if verbose and step % max(1, n_steps // 100) == 0:
                print(f"\n--- Step {step} ---")
                print(f"Temperature (T_eff):      {T_eff:.4f} eV")
                if move_info is not None:
                    if move_info["type"] == "dopant_swap":
                        print(f"Proposed move:            Swap {move_info['from_element']} (site {move_info['from_idx']}) "
                              f"<-> {move_info['to_element']} (site {move_info['to_idx']})")
                        print(f"Al-O neighbors (post):    {move_info['al_o_neighbors']}")
                        print(f"Bias weight:              {move_info['bias_weight']:.4f}")
                    elif move_info["type"] == "interstitial_move":
                        print(f"Proposed move:            Interstitial move of {move_info['element']} (site {move_info['site_idx']})")
                        print(f"Old fractional coords:    {move_info['old_frac']}")
                        print(f"New fractional coords:    {move_info['new_frac']}")
                else:
                    print("Proposed move:            None")
                print(f"Proposed energy:          {E_prop:.4f} eV")
                print(f"Current energy:           {E_current:.4f} eV")
                print(f"ΔE:                       {dE:.4f} eV")
                print(f"Move accepted:            {accepted_move}")
                print(f"Acceptance ratio:         {accepted/proposed:.3f}")
                print(f"Move time:                {t_prop_end-t_prop_start:.4f} s")
                print(f"Summary:                  E={E_current:.4f}, accepted={accepted}, proposed={proposed}, "
                      f"prop_time={t_prop_end-t_prop_start:.4f} s, T_eff={T_eff:.4f} eV\n")

        acc_ratio = accepted / proposed
        burn = int(len(trajectory) * burn_frac)
        elapsed = time.time() - start_time
        if verbose:
            print(f"Metropolis chain finished in {elapsed:.2f} s, acceptance ratio={acc_ratio:.3f}")

        run_params = {
            "n_steps": n_steps,
            "thin": thin,
            "burn_frac": burn_frac,
            "bias_strength": bias_strength,
            "calculation_type": self.calculation_type,
            "verbose": verbose,
        }

        results = {
            "trajectory": trajectory[burn:],
            "acceptance": acc_ratio,
            "time": elapsed,
            "run_params": run_params
        }
        self.results = results
        return results

    def save_results(self, path):
        """
        Save the results of the MC chain to a JSON file.

        Args:
            path (str): File path to save results.
        """
        if self.results is not None:
            with open(path, "w") as f:
                json.dump(self.results, f, indent=2)
        else:
            print("Run the MC chain first.")