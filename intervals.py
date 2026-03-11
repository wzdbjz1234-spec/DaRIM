from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx

from .common import Edge, GraphAttributeProxy


def write_interval_attributes_for_alpha(
    graph: nx.DiGraph,
    alpha: float,
    current_alpha_vals: Optional[List[float]],
) -> Tuple[str, str]:
    attr_l = f"p_lower_{alpha}"
    attr_h = f"p_upper_{alpha}"
    if current_alpha_vals is None:
        return attr_l, attr_h

    val_idx = 0
    for _, _, d in graph.edges(data=True):
        if "feature" in d:
            d[attr_l] = float(current_alpha_vals[val_idx])
            d[attr_h] = float(current_alpha_vals[val_idx + 1])
            val_idx += 2
    return attr_l, attr_h


def cleanup_interval_attributes(graph: nx.DiGraph, attr_l: str, attr_h: str) -> None:
    for _, _, d in graph.edges(data=True):
        d.pop(attr_l, None)
        d.pop(attr_h, None)


def compute_interval_coverage_proxy(
    graph: nx.DiGraph,
    lower_proxy: GraphAttributeProxy,
    upper_proxy: GraphAttributeProxy,
    real_prob_map: Dict[Edge, float],
) -> float:
    covered, total = 0, 0
    for u, v in graph.edges():
        if (u, v) not in real_prob_map:
            continue
        p_true = real_prob_map[(u, v)]
        l = lower_proxy.get((u, v), 0.0)
        r = upper_proxy.get((u, v), 1.0)
        covered += int(l <= p_true <= r)
        total += 1
    return covered / total if total > 0 else 0.0


def compute_avg_width_proxy(graph: nx.DiGraph, lower_proxy: GraphAttributeProxy, upper_proxy: GraphAttributeProxy) -> float:
    total_width = 0.0
    count = 0
    for u, v in graph.edges():
        total_width += upper_proxy.get((u, v), 1.0) - lower_proxy.get((u, v), 0.0)
        count += 1
    return total_width / count if count > 0 else 0.0


def update_graph_delta_intervals(graph: nx.DiGraph, point_prob_map: Dict[Edge, float], delta: float) -> None:
    for u, v in graph.edges():
        p = point_prob_map.get((u, v), 0.0)
        graph[u][v]["p_lower_delta"] = max(0.0, p * (1.0 - delta))
        graph[u][v]["p_upper_delta"] = min(1.0, p * (1.0 + delta))


def find_delta_for_target_coverage(
    graph: nx.DiGraph,
    point_prob_map: Dict[Edge, float],
    real_prob_map: Dict[Edge, float],
    target_coverage: float,
    steps: int = 20,
) -> float:
    low, high = 0.0, 1.0
    best = high
    for _ in range(steps):
        mid = (low + high) / 2.0
        covered = 0
        total = 0
        for u, v in graph.edges():
            if (u, v) not in real_prob_map:
                continue
            p = point_prob_map.get((u, v), 0.0)
            l = max(0.0, p * (1.0 - mid))
            r = min(1.0, p * (1.0 + mid))
            covered += int(l <= real_prob_map[(u, v)] <= r)
            total += 1
        curr_cov = covered / total if total > 0 else 0.0
        if curr_cov < target_coverage:
            low = mid
        else:
            high = mid
            best = mid
    return best


def find_delta_for_target_width(
    graph: nx.DiGraph,
    point_prob_map: Dict[Edge, float],
    target_width: float,
    steps: int = 20,
) -> float:
    low, high = 0.0, 1.0
    best = high
    for _ in range(steps):
        mid = (low + high) / 2.0
        wd_sum = 0.0
        cnt = 0
        for u, v in graph.edges():
            p = point_prob_map.get((u, v), 0.0)
            wd_sum += min(1.0, p * (1.0 + mid)) - max(0.0, p * (1.0 - mid))
            cnt += 1
        curr_wd = wd_sum / cnt if cnt > 0 else 0.0
        if curr_wd < target_width:
            low = mid
        else:
            high = mid
            best = mid
    return best
