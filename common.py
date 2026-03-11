from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mpi4py import MPI
from scipy.special import expit as sigmoid


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Edge = Tuple[int, int]
Sample = Tuple[List[np.ndarray], float]


@dataclass
class ExperimentConfig:
    theta_dimension: int = 10
    global_seed: int = 51251234
    num_epochs: int = 50
    bootstrap_num: int = 400
    data_size: int = 1000
    rrset_num: int = 40000
    fixed_seed_size: int = 10
    influence_simul: int = 100
    influence_max_time: int = 50000
    sigmoid_bias: float = 0.0
    alpha_list: Sequence[float] = field(
        default_factory=lambda: [
            0.01,
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
            1.00,
        ]
    )
    graph_path: str = "LFR_network_5000_nodes.graphml"
    output_csv: str = "result/epinions1_15wRRset_refactored.csv"
    output_plot_png: str = "result/epinions1_15wRRset_refactored.png"
    point_sample_num_seeds: int = 40
    point_sample_max_steps: int = 1
    bootstrap_batch_size: int = 5000


@dataclass
class FittedModelState:
    theta: np.ndarray
    bias: float


@dataclass
class EvaluationRecord:
    alpha: float
    boot_ratio: float
    risk_ratio: float
    width_ratio: float
    risk_consistency: float
    width_consistency: float
    boot_coverage: float
    boot_width: float
    delta_risk: float
    delta_width: float


class GraphAttributeProxy:
    def __init__(self, graph: nx.DiGraph, attr_name: str, default: float = 0.0):
        self.graph = graph
        self.attr_name = attr_name
        self.default = default

    def get(self, edge: Edge, default_val: Optional[float] = None) -> float:
        u, v = edge
        final_default = self.default if default_val is None else default_val
        if self.graph.has_edge(u, v):
            return float(self.graph[u][v].get(self.attr_name, final_default))
        return float(final_default)


def sync_random_seeds(global_seed: int) -> None:
    random.seed(global_seed + rank * 12345)
    np.random.seed(global_seed + rank * 12345)


def compute_prob(theta: np.ndarray, feature: np.ndarray, bias: float) -> float:
    return float(sigmoid(np.dot(theta, feature) + bias))


def load_graph(filename: str) -> nx.DiGraph:
    if filename.endswith(".graphml"):
        graph = nx.read_graphml(filename)
        if isinstance(graph, nx.MultiDiGraph):
            graph = nx.DiGraph(graph)
    elif filename.endswith(".txt"):
        graph = nx.DiGraph()
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    graph.add_edge(parts[0], parts[1])
    else:
        raise ValueError("不支持的图文件格式")

    try:
        mapping = {n: int(n) for n in graph.nodes()}
        graph = nx.relabel_nodes(graph, mapping)
    except ValueError:
        pass

    for _, _, data in graph.edges(data=True):
        if "feature" in data and isinstance(data["feature"], str):
            raw = data["feature"].replace(",", " ")
            data["feature"] = np.array([float(x) for x in raw.split()], dtype=float)
        if "probability" in data:
            data["probability"] = float(data["probability"])
    return graph


def inspect_graph(graph: nx.DiGraph) -> None:
    if rank != 0:
        return
    print("\n[DEBUG] --- Inspecting Data Quality ---")
    for u, v, data in list(graph.edges(data=True))[:3]:
        print(f"Edge ({u}->{v}):")
        if "feature" in data:
            print(f"  Feature Check: OK (Mean={np.mean(data['feature']):.4f})")
        else:
            print("  [CRITICAL] 'feature' key MISSING!")
        if "probability" in data:
            print(f"  Probability: {data['probability']}")
        print(f"  Raw Keys: {list(data.keys())}")
    print("[DEBUG] -----------------------------\n")
    sys.stdout.flush()


def save_results(records: List[EvaluationRecord], output_csv: str, output_plot_png: str) -> None:
    if rank != 0:
        return
    import pandas as pd

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df = pd.DataFrame([record.__dict__ for record in records])
    df.to_csv(output_csv, index=False)
    print(f"\n[Done] Results saved to {output_csv}", flush=True)
    print(df, flush=True)

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_xlabel("Alpha")
    ax1.set_ylabel("Robust Ratio")
    ax1.plot(df["alpha"], df["boot_ratio"], marker="o", label="Bootstrap")
    ax1.plot(df["alpha"], df["risk_ratio"], marker="^", label="Risk-Match")
    ax1.plot(df["alpha"], df["width_ratio"], marker="v", label="Width-Match")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Consistency")
    ax2.plot(df["alpha"], df["risk_consistency"], linestyle="--", label="Risk Cons")
    ax2.plot(df["alpha"], df["width_consistency"], linestyle="--", label="Width Cons")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.savefig(output_plot_png)
    print(f"[Done] Plot saved to {output_plot_png}", flush=True)
