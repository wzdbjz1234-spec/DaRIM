from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.special import expit as sigmoid
from tqdm import tqdm

from .common import Edge, ExperimentConfig, FittedModelState, Sample, comm, compute_prob, rank, size


def _generate_sample_batch(
    graph: nx.DiGraph,
    num_seeds: int,
    max_steps: int,
    num_samples: int,
    rng_np: np.random.RandomState,
) -> List[Sample]:
    samples: List[Sample] = []
    nodes = list(graph.nodes())

    pbar = tqdm(total=num_samples, desc=f"[Rank {rank}] Samples", disable=(rank != 0), ncols=80)
    while len(samples) < num_samples:
        seeds = rng_np.choice(nodes, num_seeds, replace=False)
        current_active_nodes = list(seeds)
        global_activated = set(seeds)
        step = 0

        while current_active_nodes and step < max_steps and len(samples) < num_samples:
            attempts: Dict[int, List[int]] = {}
            for u in current_active_nodes:
                for v in graph.successors(u):
                    if v not in global_activated:
                        attempts.setdefault(v, []).append(u)

            next_active_nodes: List[int] = []
            for v, parents in attempts.items():
                if len(samples) >= num_samples:
                    break

                features_list: List[np.ndarray] = []
                valid_parents: List[int] = []
                for u in parents:
                    if "feature" in graph[u][v]:
                        features_list.append(np.asarray(graph[u][v]["feature"], dtype=float))
                        valid_parents.append(u)
                if not features_list:
                    continue

                fail_prob = 1.0
                for u in valid_parents:
                    fail_prob *= 1.0 - float(graph[u][v].get("probability", 0.0))
                prob_activation = 1.0 - fail_prob

                y = 1.0 if rng_np.random_sample() < prob_activation else 0.0
                if y > 0.5:
                    next_active_nodes.append(v)
                    global_activated.add(v)

                samples.append((features_list, y))
                if rank == 0:
                    pbar.update(1)

            current_active_nodes = next_active_nodes
            step += 1

    pbar.close()
    return samples[:num_samples]


def mpi_generate_propagation_samples(
    graph: nx.DiGraph,
    num_seeds: int,
    max_steps: int,
    num_samples: int,
    base_seed: int,
) -> Optional[List[Sample]]:
    num_samples = comm.bcast(num_samples, root=0)
    samples_per_rank = num_samples // size + (1 if rank < num_samples % size else 0)
    rng_np = np.random.RandomState(base_seed + rank * 1000)
    local_samples = (
        _generate_sample_batch(graph, num_seeds, max_steps, samples_per_rank, rng_np) if samples_per_rank > 0 else []
    )
    all_samples = comm.gather(local_samples, root=0)
    if rank == 0:
        return [s for sub in all_samples for s in sub][:num_samples]
    return None


class HyperparametricModel:
    def __init__(self, d: int, B: float = 1.0, bias: float = -2.0):
        self.d = d
        self.B = B
        self.bias = bias
        self.theta = np.random.normal(0.0, 0.01, d)

    def predict_edge_prob(self, x: np.ndarray) -> float:
        return float(sigmoid(np.dot(self.theta, x) + self.bias))

    def sgd_step(self, sample: Sample, lr: float) -> None:
        feat_list, y = sample
        if not feat_list:
            return

        pos_weight = 5.0 if y > 0.5 else 1.0
        p_list: List[float] = []
        one_minus_p_list: List[float] = []
        total_fail_prob = 1.0
        for feat in feat_list:
            p = self.predict_edge_prob(feat)
            p_list.append(p)
            omp = 1.0 - p
            one_minus_p_list.append(omp)
            total_fail_prob *= omp

        eps = 1e-8
        f_theta = float(np.clip(1.0 - total_fail_prob, eps, 1.0 - eps))
        grad_factor = ((y / f_theta) - ((1.0 - y) / (1.0 - f_theta))) * pos_weight

        grad_theta = np.zeros(self.d)
        grad_bias = 0.0
        for i, feat in enumerate(feat_list):
            omp = one_minus_p_list[i]
            term = (total_fail_prob / omp) if omp > eps else 0.0
            common_grad = p_list[i] * omp * term
            grad_theta += common_grad * feat
            grad_bias += common_grad

        self.theta = np.clip(self.theta + lr * grad_factor * grad_theta, -self.B, self.B)
        self.bias = float(np.clip(self.bias + lr * grad_factor * grad_bias, -8.0, 2.0))

    def train(self, samples: List[Sample], num_epochs: int, rng: Optional[random.Random] = None) -> None:
        if rng is None:
            rng = random.Random()
        base_lr = 0.5 / math.sqrt(len(samples)) if samples else 0.001
        for _ in range(num_epochs):
            rng.shuffle(samples)
            for sample in samples:
                self.sgd_step(sample, base_lr)

    def export_state(self) -> FittedModelState:
        return FittedModelState(theta=self.theta.copy(), bias=float(self.bias))


def fit_point_estimator(
    graph: nx.DiGraph,
    samples: Optional[List[Sample]],
    config: ExperimentConfig,
) -> Tuple[Dict[Edge, float], Optional[FittedModelState]]:
    model_state = None
    point_prob_map: Optional[Dict[Edge, float]] = None
    if rank == 0:
        model = HyperparametricModel(config.theta_dimension, bias=config.sigmoid_bias)
        model.train(samples or [], config.num_epochs, rng=random.Random(config.global_seed))
        model_state = model.export_state()
        point_prob_map = {
            (u, v): compute_prob(model_state.theta, np.asarray(d.get("feature", np.zeros(config.theta_dimension))), model_state.bias)
            for u, v, d in graph.edges(data=True)
        }

    point_prob_map = comm.bcast(point_prob_map, root=0)
    model_state = comm.bcast(model_state, root=0)
    return point_prob_map, model_state


def fit_bootstrap_models(
    samples: List[Sample],
    config: ExperimentConfig,
) -> List[FittedModelState]:
    local_states: List[FittedModelState] = []
    if samples:
        iterator = range(rank, config.bootstrap_num, size)
        if rank == 0:
            iterator = tqdm(iterator, desc="Bootstrap", ncols=100)
        bootstrap_rng = random.Random(config.global_seed + rank * 20000)
        for _ in iterator:
            resample = [bootstrap_rng.choice(samples) for _ in range(len(samples))]
            model = HyperparametricModel(config.theta_dimension, bias=config.sigmoid_bias)
            model.train(resample, config.num_epochs, rng=bootstrap_rng)
            local_states.append(model.export_state())

    all_states = comm.gather(local_states, root=0)
    flat_states: List[FittedModelState] = []
    if rank == 0:
        flat_states = [state for sub in all_states for state in sub]
    return comm.bcast(flat_states, root=0)


def precompute_bootstrap_intervals(
    graph: nx.DiGraph,
    model_states: List[FittedModelState],
    alpha_list: Sequence[float],
    batch_size: int,
) -> Dict[float, List[float]]:
    sync_intervals_data: Dict[float, List[float]] = {}
    if rank != 0 or not model_states:
        return sync_intervals_data

    theta_matrix = np.stack([state.theta for state in model_states], axis=1)
    bias_vec = np.array([state.bias for state in model_states], dtype=float)
    percentiles_map = {alpha: ((alpha / 2.0) * 100.0, (1.0 - alpha / 2.0) * 100.0) for alpha in alpha_list}
    sync_intervals_data = {alpha: [] for alpha in alpha_list}

    all_edges = list(graph.edges(data=True))
    for i in tqdm(range(0, len(all_edges), batch_size), desc="Pre-calc Intervals", ncols=100):
        batch = all_edges[i : i + batch_size]
        batch_features: List[np.ndarray] = []
        valid_indices: List[int] = []
        for idx, (_, _, data) in enumerate(batch):
            if "feature" in data:
                batch_features.append(np.asarray(data["feature"], dtype=float))
                valid_indices.append(idx)

        if not batch_features:
            continue

        feat_mat = np.stack(batch_features, axis=0)
        probs = sigmoid(np.dot(feat_mat, theta_matrix) + bias_vec[np.newaxis, :])
        for alpha in alpha_list:
            low_p, high_p = percentiles_map[alpha]
            lows = np.percentile(probs, low_p, axis=1)
            highs = np.percentile(probs, high_p, axis=1)
            for idx_feat in range(len(valid_indices)):
                sync_intervals_data[alpha].append(float(lows[idx_feat]))
                sync_intervals_data[alpha].append(float(highs[idx_feat]))

    return sync_intervals_data
