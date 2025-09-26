#!/usr/bin/env python
import argparse
import os
import yaml
import multiprocessing
import random
import numpy as np
from tqdm import tqdm
from pymatgen.core import Structure
from mc_structure_explorer.mc_engine import MonteCarloEngine
from mc_structure_explorer.utils import save_config, generate_chemiscope_files

# -------------------------
# Argument parsing
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run Monte Carlo chains for dopant exploration.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    return parser.parse_args()


# -------------------------
# Single chain runner
# -------------------------
def run_single_chain(seed, config, save_path=None, progress_callback=None, initial_structure=None):
    random.seed(seed)
    np.random.seed(seed)

    # Load structure and make supercell
    stru = Structure.from_file(config["base_structure_path"])
    stru.make_supercell(config["supercell_matrix"])

    # Prepare substitution list
    sub_list = [{"replaced_atom": "Si", "dopant_atom": "Al"} for _ in range(config["n_Al"])] + \
               [{"replaced_atom": "N", "dopant_atom": "O"} for _ in range(config["n_O"])]

    # Interstitial site (Eu)
    inter_site = {"Eu": config["Eu_position"]} if config["n_Eu"] > 0 else None

    mc_engine = MonteCarloEngine(
        base_structure=stru,
        substitution_list=sub_list,
        calculation_type=config["calculation_type"],
        interstitial_site=inter_site,
        seed=seed
    )

    # Temperature schedule
    T_schedule = np.linspace(config["T_start"], config["T_end"], config["n_steps"])

    # Run MC chain
    mc_engine.run_chain(
        n_steps=config["n_steps"],
        T_schedule=T_schedule,
        thin=config["thin"],
        burn_frac=config["burn_frac"],
        bias_strength=config["bias_strength"],
        eu_bias_strength=config["eu_bias_strength"],
        bias_on_plane=config["bias_on_plane"],
        interstitial_move_prob=config["eu_move_prob"],
        verbose=config["verbose"],
        progress_callback=progress_callback,
        initial_structure=initial_structure
    )

    # Save results
    if save_path:
        mc_engine.save_results(save_path)

    return mc_engine.results


# -------------------------
# Multiprocessing wrapper
# -------------------------
def run_chain_wrapper(seed, idx, queue, config, initial_structure=None):
    os.makedirs(config["result_dir"], exist_ok=True)
    os.makedirs(config["chemiscope_dir"], exist_ok=True)

    json_path = os.path.join(config["result_dir"], f"results_{idx}.json")
    chemiscope_path = os.path.join(config["chemiscope_dir"], f"dataset_{idx}.json")

    # Progress bar
    pbar = tqdm(total=config["n_steps"], desc=f"Seed {seed}", position=idx, leave=True)
    def progress_callback(step):
        pbar.update(1)

    results = run_single_chain(
        seed=seed,
        config=config,
        save_path=json_path,
        progress_callback=progress_callback,
        initial_structure=initial_structure
    )
    pbar.close()

    generate_chemiscope_files(path=chemiscope_path, result=results)
    queue.put(f"DONE_{idx}")


# -------------------------
# Main function
# -------------------------
def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Define a base directory for relative paths (optional)
    # For example, use the directory where the YAML file resides
    base_dir = os.path.dirname(os.path.abspath(args.config))

    # Convert relevant paths to absolute
    config["base_structure_path"] = os.path.join(base_dir, config["base_structure_path"])
    config["result_dir"] = os.path.join(base_dir, config["result_dir"])
    config["chemiscope_dir"] = os.path.join(base_dir, config["chemiscope_dir"])

    # Ensure directories
    for d in [config["result_dir"], config["chemiscope_dir"]]:
        os.makedirs(d, exist_ok=True)

    # Set global seeds
    random.seed(config["seed_global"])
    np.random.seed(config["seed_global"])

    # Determine seeds for each run
    if config.get("seeds") is None:
        seeds = [config["seed_global"] + i for i in range(config["n_runs"])]
    else:
        seeds = config["seeds"]

    # Optional: same initial structure for all chains
    initial_struct = None
    if config.get("same_initial_structure"):
        base_struct = Structure.from_file(config["base_structure_path"])
        base_struct.make_supercell(config["supercell_matrix"])
        sub_list = [{"replaced_atom": "Si", "dopant_atom": "Al"} for _ in range(config["n_Al"])] + \
                   [{"replaced_atom": "N", "dopant_atom": "O"} for _ in range(config["n_O"])]
        inter_site = {"Eu": config["Eu_position"]} if config["n_Eu"] > 0 else None
        mc_tmp = MonteCarloEngine(base_structure=base_struct, substitution_list=sub_list,
                                  interstitial_site=inter_site, seed=config["seed_global"])
        initial_struct = mc_tmp.generate_initial_structure()
    
    if config.get("initial_structure"):
        initial_struct = Structure.from_file(config["initial_structure"])

    # Start multiprocessing chains
    queue = multiprocessing.Queue()
    processes = []
    for idx, seed in enumerate(seeds):
        p = multiprocessing.Process(target=run_chain_wrapper,
                                    args=(seed, idx, queue, config, initial_struct))
        p.start()
        processes.append(p)

    # Monitor progress
    finished_runs = 0
    while finished_runs < config["n_runs"]:
        msg = queue.get()
        if msg.startswith("DONE_"):
            finished_runs += 1
        else:
            print(msg)

    for p in processes:
        p.join()

    # Save config for reproducibility
    save_config(config, os.path.join(config["result_dir"], "mc_config.yaml"))
    print("All MC chains finished.")


if __name__ == "__main__":
    main()
