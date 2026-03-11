"""Build a GraphML dataset with edge features and diffusion probabilities.

Workflow:
1. Read an edge list from a real topology file.
2. Sample node features.
3. Construct edge features by concatenating source/target node features.
4. Sample a ground-truth theta vector.
5. Compute edge probabilities via sigmoid(theta · x + bias).
6. Save the graph as GraphML.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import networkx as nx
import numpy as np
from scipy.special import expit as sigmoid

import config_refactored as config


@dataclass
class GraphBuildConfig:
    input_path: str = "NetHEPT.txt"
    output_path: str = "graph_with_features.graphml"
    feature_dimension: int = config.FEATURE_DIMENSION
    theta_dimension: int = config.THETA_DIMENSION
    node_embedding_dimension: int = config.NODE_EMBEDDING_DIM
    random_seed: int = 114514
    edge_delimiter: str | None = None
    add_reverse_edges: bool = False
    theta_low: float = -1.0
    theta_high: float = 0.0
    sigmoid_bias: float = 0.0

    def validate(self) -> None:
        if self.theta_dimension != self.feature_dimension:
            raise ValueError("theta_dimension must equal feature_dimension.")
        if self.feature_dimension != 2 * self.node_embedding_dimension:
            raise ValueError(
                "Current builder assumes edge_feature = src_embedding ⊕ dst_embedding, "
                "so feature_dimension must equal 2 * node_embedding_dimension."
            )
        if self.theta_low >= self.theta_high:
            raise ValueError("theta_low must be smaller than theta_high.")


def parse_edge_line(line: str, delimiter: str | None = None) -> Tuple[str, str] | None:
    parts = line.strip().split(delimiter)
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def load_directed_graph_from_edgelist(
    path: str,
    delimiter: str | None = None,
    add_reverse_edges: bool = False,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            edge = parse_edge_line(raw_line, delimiter=delimiter)
            if edge is None:
                continue
            u, v = edge
            graph.add_edge(u, v)
            if add_reverse_edges:
                graph.add_edge(v, u)
    return graph


def generate_node_features(
    graph: nx.DiGraph,
    node_dim: int,
    rng: np.random.Generator,
) -> tuple[list[str], dict[str, int], np.ndarray]:
    node_list = list(graph.nodes())
    node_index = {node: idx for idx, node in enumerate(node_list)}
    node_features = rng.uniform(0.0, 1.0, size=(len(node_list), node_dim))
    return node_list, node_index, node_features


def attach_edge_features(
    graph: nx.DiGraph,
    node_index: dict[str, int],
    node_features: np.ndarray,
) -> None:
    for u, v in graph.edges():
        src_feat = node_features[node_index[u]]
        dst_feat = node_features[node_index[v]]
        edge_feature = np.concatenate([src_feat, dst_feat])
        graph.edges[u, v]["feature"] = edge_feature


def sample_theta(cfg: GraphBuildConfig, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(cfg.theta_low, cfg.theta_high, size=cfg.theta_dimension)


def attach_probabilities(graph: nx.DiGraph, theta: np.ndarray, bias: float = 0.0) -> None:
    for _, _, data in graph.edges(data=True):
        x = data["feature"]
        prob = sigmoid(float(np.dot(theta, x)) + bias)
        data["probability"] = float(prob)


def to_graphml_safe_graph(graph: nx.DiGraph) -> nx.DiGraph:
    safe_graph = nx.DiGraph()
    safe_graph.add_nodes_from(graph.nodes(data=True))
    for u, v, data in graph.edges(data=True):
        serialized = dict(data)
        if "feature" in serialized and isinstance(serialized["feature"], np.ndarray):
            serialized["feature"] = ",".join(map(str, serialized["feature"].tolist()))
        if "probability" in serialized:
            serialized["probability"] = float(serialized["probability"])
        safe_graph.add_edge(u, v, **serialized)
    return safe_graph


def build_graph_dataset(cfg: GraphBuildConfig) -> tuple[nx.DiGraph, np.ndarray]:
    cfg.validate()
    rng = np.random.default_rng(cfg.random_seed)

    graph = load_directed_graph_from_edgelist(
        path=cfg.input_path,
        delimiter=cfg.edge_delimiter,
        add_reverse_edges=cfg.add_reverse_edges,
    )
    _, node_index, node_features = generate_node_features(
        graph=graph,
        node_dim=cfg.node_embedding_dimension,
        rng=rng,
    )
    attach_edge_features(graph, node_index, node_features)
    theta_true = sample_theta(cfg, rng)
    attach_probabilities(graph, theta_true, bias=cfg.sigmoid_bias)
    return graph, theta_true


def save_graphml(graph: nx.DiGraph, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(to_graphml_safe_graph(graph), output)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build graph dataset with edge features and probabilities.")
    parser.add_argument("--input-path", default="NetHEPT.txt")
    parser.add_argument("--output-path", default="graph_with_features.graphml")
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--delimiter", default=None)
    parser.add_argument("--add-reverse-edges", action="store_true")
    parser.add_argument("--theta-low", type=float, default=-1.0)
    parser.add_argument("--theta-high", type=float, default=0.0)
    parser.add_argument("--sigmoid-bias", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    cfg = GraphBuildConfig(
        input_path=args.input_path,
        output_path=args.output_path,
        random_seed=args.seed,
        edge_delimiter=args.delimiter,
        add_reverse_edges=args.add_reverse_edges,
        theta_low=args.theta_low,
        theta_high=args.theta_high,
        sigmoid_bias=args.sigmoid_bias,
    )
    graph, theta_true = build_graph_dataset(cfg)
    save_graphml(graph, cfg.output_path)
    print(
        f"Built graph from {cfg.input_path}: |V|={graph.number_of_nodes()}, "
        f"|E|={graph.number_of_edges()}"
    )
    print(f"Saved GraphML to {cfg.output_path}")
    print(f"theta_true[:5] = {theta_true[:5]}")


if __name__ == "__main__":
    main()
