from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm

from .common import GraphAttributeProxy, Edge, comm, rank, size, compute_prob
from .common import MPI  # type: ignore[attr-defined]


# MPI is re-exported by mpi4py import side effect on common.py


def _generate_rr_set(
    graph: nx.DiGraph,
    edge_prob: GraphAttributeProxy,
    start_node: int,
    predecessors_dict: Dict[int, List[int]],
    rng: random.Random,
) -> set[int]:
    rr_set_nodes = {start_node}
    queue = deque([start_node])
    while queue:
        v = queue.popleft()
        for u in predecessors_dict.get(v, []):
            if u in rr_set_nodes:
                continue
            prob = edge_prob.get((u, v), 0.0)
            if prob > 0.0 and rng.random() < prob:
                rr_set_nodes.add(u)
                queue.append(u)
    return rr_set_nodes


def compute_influence_from_seed(
    graph: nx.DiGraph,
    edge_prob: GraphAttributeProxy,
    seedset: Sequence[int],
    simul: int = 1000,
    rng: Optional[random.Random] = None,
) -> float:
    if not seedset:
        return 0.0
    if rng is None:
        rng = random.Random()

    successors_dict = {u: list(graph.successors(u)) for u in graph.nodes()}
    total_activated = 0.0
    for _ in range(simul):
        activated = set(seedset)
        frontier = deque(seedset)
        while frontier:
            u = frontier.popleft()
            for v in successors_dict[u]:
                if v in activated:
                    continue
                prob = edge_prob.get((u, v), 0.0)
                if prob > 0.0 and rng.random() < prob:
                    activated.add(v)
                    frontier.append(v)
        total_activated += len(activated)
    return total_activated / float(simul)


def mpi_build_rr_sets_distributed(
    graph: nx.DiGraph,
    edge_prob: GraphAttributeProxy,
    num_rr: int,
    global_seed: int,
    desc: Optional[str] = None,
) -> Tuple[List[List[int]], List[int], int]:
    num_rr = comm.bcast(num_rr, root=0)
    nodes = list(graph.nodes())
    sorted_nodes = sorted(nodes)
    node_to_int = {n: i for i, n in enumerate(sorted_nodes)}
    rr_sets_per_rank = num_rr // size
    remainder = num_rr % size
    local_num_rr = rr_sets_per_rank + (1 if rank < remainder else 0)

    rng = random.Random(global_seed + rank * 10000)
    predecessors_dict = {n: list(graph.predecessors(n)) for n in nodes}
    local_rr_sets: List[List[int]] = []

    pbar = None
    if rank == 0:
        pbar = tqdm(total=num_rr, desc=desc or "Generating RR Sets", unit="RR", ncols=100)

    for _ in range(local_num_rr):
        v = rng.choice(nodes)
        rr = _generate_rr_set(graph, edge_prob, v, predecessors_dict, rng)
        local_rr_sets.append([node_to_int[u] for u in rr])
        if pbar is not None:
            pbar.update(size)

    if pbar is not None:
        pbar.close()

    return local_rr_sets, sorted_nodes, num_rr


def distributed_greedy(local_rr_sets: List[List[int]], sorted_nodes: List[int], num_seed: int) -> Optional[List[int]]:
    num_nodes = len(sorted_nodes)
    local_num = len(local_rr_sets)
    is_covered = np.zeros(local_num, dtype=bool)
    local_inverted_index: List[List[int]] = [[] for _ in range(num_nodes)]
    for rr_idx, rr in enumerate(local_rr_sets):
        for u_idx in rr:
            local_inverted_index[u_idx].append(rr_idx)

    seeds_indices: List[int] = []
    for _ in range(num_seed):
        local_gains = np.zeros(num_nodes, dtype=np.int32)
        for u_idx in range(num_nodes):
            gain = 0
            for rr_idx in local_inverted_index[u_idx]:
                if not is_covered[rr_idx]:
                    gain += 1
            local_gains[u_idx] = gain

        global_gains = np.zeros(num_nodes, dtype=np.int32)
        comm.Reduce([local_gains, MPI.INT], [global_gains, MPI.INT], op=MPI.SUM, root=0)

        best_node_idx = -1
        if rank == 0:
            for selected_idx in seeds_indices:
                global_gains[selected_idx] = -1
            best_node_idx = int(np.argmax(global_gains))
            if global_gains[best_node_idx] <= 0:
                best_node_idx = -1

        best_node_idx = comm.bcast(best_node_idx, root=0)
        if best_node_idx == -1:
            break

        seeds_indices.append(best_node_idx)
        for rr_idx in local_inverted_index[best_node_idx]:
            is_covered[rr_idx] = True

    if rank == 0:
        return [sorted_nodes[i] for i in seeds_indices]
    return None


def ris_greedy(
    graph: nx.DiGraph,
    edge_prob: GraphAttributeProxy,
    k: int,
    rrset_num: int,
    rr_seed: int,
    desc: str,
) -> List[int]:
    local_rr, sorted_nodes, _ = mpi_build_rr_sets_distributed(
        graph=graph,
        edge_prob=edge_prob,
        num_rr=rrset_num,
        global_seed=rr_seed,
        desc=desc,
    )
    seeds = distributed_greedy(local_rr, sorted_nodes, k)
    return comm.bcast(seeds, root=0)


def lugreedy(
    graph: nx.DiGraph,
    lower_proxy: GraphAttributeProxy,
    upper_proxy: GraphAttributeProxy,
    k: int,
    rrset_num: int,
    influence_simul: int,
    global_seed: int,
) -> List[int]:
    seed_l = ris_greedy(graph, lower_proxy, k, rrset_num, global_seed, "RR Sets (Lower)")
    seed_u = ris_greedy(graph, upper_proxy, k, rrset_num, global_seed + 999, "RR Sets (Upper)")

    local_simul = influence_simul // size + (1 if rank < influence_simul % size else 0)
    rng_l = random.Random(global_seed + 30000 + rank)
    rng_u = random.Random(global_seed + 40000 + rank)

    local_score_l = compute_influence_from_seed(graph, lower_proxy, seed_l, simul=local_simul, rng=rng_l)
    local_score_u = compute_influence_from_seed(graph, lower_proxy, seed_u, simul=local_simul, rng=rng_u)

    score_l = comm.reduce(local_score_l * local_simul, op=MPI.SUM, root=0)
    score_u = comm.reduce(local_score_u * local_simul, op=MPI.SUM, root=0)

    chosen = seed_l
    if rank == 0:
        mean_l = score_l / float(influence_simul)
        mean_u = score_u / float(influence_simul)
        chosen = seed_l if mean_l >= mean_u else seed_u
    return comm.bcast(chosen, root=0)


def evaluate_robust_ratio(
    graph: nx.DiGraph,
    lower_proxy: GraphAttributeProxy,
    upper_proxy: GraphAttributeProxy,
    seeds: Sequence[int],
    k: int,
    rrset_num: int,
    influence_simul: int,
    global_seed: int,
) -> float:
    local_simul = influence_simul // size + (1 if rank < influence_simul % size else 0)

    influence_rng = random.Random(global_seed + rank * 30000)
    local_num = compute_influence_from_seed(graph, lower_proxy, seeds, simul=local_simul, rng=influence_rng)
    numerator = comm.reduce(local_num * local_simul, op=MPI.SUM, root=0)

    oracle_seeds = ris_greedy(graph, upper_proxy, k, rrset_num, global_seed + 8888, "RR Sets (Upper Oracle)")
    denom_rng = random.Random(global_seed + rank * 30000 + 50000)
    local_denom = compute_influence_from_seed(graph, upper_proxy, oracle_seeds, simul=local_simul, rng=denom_rng)
    denominator = comm.reduce(local_denom * local_simul, op=MPI.SUM, root=0)

    ratio = 0.0
    if rank == 0 and denominator and denominator > 1e-9:
        ratio = float(numerator / denominator)
    return comm.bcast(ratio, root=0)


def initialize_ground_truth_probabilities(
    graph: nx.DiGraph,
    theta_dimension: int,
    default_bias: float,
) -> Dict[Edge, float]:
    real_prob_map: Dict[Edge, float] = {}
    if rank == 0:
        default_theta = np.array([-0.31, -0.32, -0.57, -0.21, -0.21, -0.96, -0.10, -0.24, -0.78, -0.33], dtype=float)
        if len(default_theta) != theta_dimension:
            default_theta = np.resize(default_theta, theta_dimension)

        for u, v, data in graph.edges(data=True):
            feat = np.asarray(data.get("feature", np.zeros(theta_dimension)), dtype=float)
            data["feature"] = feat
            if "probability" not in data:
                data["probability"] = compute_prob(default_theta, feat, default_bias)
            real_prob_map[(u, v)] = float(data["probability"])

    real_prob_map = comm.bcast(real_prob_map, root=0)
    if rank != 0:
        for u, v in graph.edges():
            graph[u][v]["probability"] = real_prob_map[(u, v)]
    return real_prob_map
