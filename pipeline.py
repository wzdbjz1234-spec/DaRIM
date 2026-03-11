from __future__ import annotations

import gc
from typing import Dict, List, Tuple

import networkx as nx
from tqdm import tqdm

from .common import Edge, EvaluationRecord, ExperimentConfig, GraphAttributeProxy, comm, rank
from .estimation import precompute_bootstrap_intervals
from .graph_ops import evaluate_robust_ratio, lugreedy
from .intervals import (
    cleanup_interval_attributes,
    compute_avg_width_proxy,
    compute_interval_coverage_proxy,
    find_delta_for_target_coverage,
    find_delta_for_target_width,
    update_graph_delta_intervals,
    write_interval_attributes_for_alpha,
)


def run_global_delta_baseline(
    graph: nx.DiGraph,
    point_prob_map: Dict[Edge, float],
    delta: float,
    k: int,
    config: ExperimentConfig,
) -> Tuple[List[int], float]:
    update_graph_delta_intervals(graph, point_prob_map, delta)
    lower_proxy = GraphAttributeProxy(graph, "p_lower_delta")
    upper_proxy = GraphAttributeProxy(graph, "p_upper_delta")
    seeds = lugreedy(
        graph=graph,
        lower_proxy=lower_proxy,
        upper_proxy=upper_proxy,
        k=k,
        rrset_num=config.rrset_num,
        influence_simul=config.influence_simul,
        global_seed=config.global_seed,
    )
    ratio = evaluate_robust_ratio(
        graph=graph,
        lower_proxy=lower_proxy,
        upper_proxy=upper_proxy,
        seeds=seeds,
        k=k,
        rrset_num=config.rrset_num,
        influence_simul=config.influence_simul,
        global_seed=config.global_seed,
    )
    return seeds, ratio


def run_darim_pipeline(
    graph: nx.DiGraph,
    real_prob_map: Dict[Edge, float],
    point_prob_map: Dict[Edge, float],
    bootstrap_states,
    config: ExperimentConfig,
) -> List[EvaluationRecord]:
    sync_intervals_data = precompute_bootstrap_intervals(
        graph=graph,
        model_states=bootstrap_states,
        alpha_list=config.alpha_list,
        batch_size=config.bootstrap_batch_size,
    )
    comm.Barrier()

    records: List[EvaluationRecord] = []
    for alpha in config.alpha_list:
        if rank == 0:
            tqdm.write(f"\n>>> Processing alpha = {alpha} ...")

        current_alpha_vals = sync_intervals_data.get(alpha) if rank == 0 else None
        current_alpha_vals = comm.bcast(current_alpha_vals, root=0)

        attr_l, attr_h = write_interval_attributes_for_alpha(graph, alpha, current_alpha_vals)
        lower_proxy_boot = GraphAttributeProxy(graph, attr_l)
        upper_proxy_boot = GraphAttributeProxy(graph, attr_h)

        cov_boot = compute_interval_coverage_proxy(graph, lower_proxy_boot, upper_proxy_boot, real_prob_map)
        width_boot = compute_avg_width_proxy(graph, lower_proxy_boot, upper_proxy_boot)
        cov_boot = comm.bcast(cov_boot, root=0)
        width_boot = comm.bcast(width_boot, root=0)

        seed_boot = lugreedy(
            graph=graph,
            lower_proxy=lower_proxy_boot,
            upper_proxy=upper_proxy_boot,
            k=config.fixed_seed_size,
            rrset_num=config.rrset_num,
            influence_simul=config.influence_simul,
            global_seed=config.global_seed,
        )
        ratio_boot = evaluate_robust_ratio(
            graph=graph,
            lower_proxy=lower_proxy_boot,
            upper_proxy=upper_proxy_boot,
            seeds=seed_boot,
            k=config.fixed_seed_size,
            rrset_num=config.rrset_num,
            influence_simul=config.influence_simul,
            global_seed=config.global_seed,
        )

        delta_risk = find_delta_for_target_coverage(graph, point_prob_map, real_prob_map, cov_boot)
        seed_risk, ratio_risk = run_global_delta_baseline(
            graph=graph,
            point_prob_map=point_prob_map,
            delta=delta_risk,
            k=config.fixed_seed_size,
            config=config,
        )

        delta_width = find_delta_for_target_width(graph, point_prob_map, width_boot)
        seed_width, ratio_width = run_global_delta_baseline(
            graph=graph,
            point_prob_map=point_prob_map,
            delta=delta_width,
            k=config.fixed_seed_size,
            config=config,
        )

        if rank == 0:
            s_boot = set(seed_boot)
            s_risk = set(seed_risk)
            s_width = set(seed_width)
            cons_risk = len(s_risk & s_boot) / len(s_risk | s_boot) if (s_risk | s_boot) else 1.0
            cons_width = len(s_width & s_boot) / len(s_width | s_boot) if (s_width | s_boot) else 1.0
            tqdm.write(f"  > Boot : ratio={ratio_boot:.4f}, cov={cov_boot:.4f}, width={width_boot:.4f}")
            tqdm.write(f"  > Risk : ratio={ratio_risk:.4f}, cons={cons_risk:.4f}, delta={delta_risk:.4f}")
            tqdm.write(f"  > Width: ratio={ratio_width:.4f}, cons={cons_width:.4f}, delta={delta_width:.4f}")
            records.append(
                EvaluationRecord(
                    alpha=alpha,
                    boot_ratio=ratio_boot,
                    risk_ratio=ratio_risk,
                    width_ratio=ratio_width,
                    risk_consistency=cons_risk,
                    width_consistency=cons_width,
                    boot_coverage=cov_boot,
                    boot_width=width_boot,
                    delta_risk=delta_risk,
                    delta_width=delta_width,
                )
            )

        cleanup_interval_attributes(graph, attr_l, attr_h)
        gc.collect()

    return records
