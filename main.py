from __future__ import annotations

import gc
import os
import traceback

from .common import ExperimentConfig, inspect_graph, load_graph, save_results, sync_random_seeds, comm, rank
from .estimation import fit_bootstrap_models, fit_point_estimator, mpi_generate_propagation_samples
from .graph_ops import initialize_ground_truth_probabilities
from .pipeline import run_darim_pipeline


def main() -> None:
    config = ExperimentConfig()
    sync_random_seeds(config.global_seed)
    graph = load_graph(config.graph_path)
    inspect_graph(graph)

    if rank == 0:
        os.makedirs("result", exist_ok=True)
        print("[Rank 0] >>> Starting Robust Ratio Comparison <<<", flush=True)

    real_prob_map = initialize_ground_truth_probabilities(graph, config.theta_dimension, config.sigmoid_bias)
    comm.Barrier()

    if rank == 0:
        print("[Rank 0] Training Point Estimate...", flush=True)
    samples = mpi_generate_propagation_samples(
        graph,
        num_seeds=config.point_sample_num_seeds,
        max_steps=config.point_sample_max_steps,
        num_samples=config.data_size,
        base_seed=config.global_seed,
    )

    point_prob_map, _ = fit_point_estimator(graph, samples, config)
    samples = comm.bcast(samples, root=0)

    if rank == 0:
        print(f"[Rank 0] Training {config.bootstrap_num} Bootstrap Models...", flush=True)
    bootstrap_states = fit_bootstrap_models(samples or [], config)

    if rank == 0:
        del samples
    gc.collect()

    records = run_darim_pipeline(
        graph=graph,
        real_prob_map=real_prob_map,
        point_prob_map=point_prob_map,
        bootstrap_states=bootstrap_states,
        config=config,
    )
    save_results(records, config.output_csv, config.output_plot_png)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        comm.Abort(1)
