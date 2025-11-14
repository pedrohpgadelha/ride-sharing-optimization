"""
Compute and save an instance's OD matrix, a compact matrix of travel times and
distances between relevant nodes in the grid graph.
"""

from pathlib import Path
import networkx as nx
import numpy as np

from src.graph import build_grid_graph, build_grid_graph_spec, snap_to_index
from src.io import read_yaml, ensure_dir, save_npz, write_manifest, write_json


def compute_and_save_od(instance_yaml: str | Path, out_dir: str | Path) -> None:
    """
    Compute and save an instance's OD matrix into a .npz file.
    """
    instance = read_yaml(instance_yaml)
    G = build_grid_graph(instance["grid"], instance["speeds"], instance["zones"])
    grid_graph_spec = build_grid_graph_spec(instance["grid"])

    garages = [tuple(v["garage_xy_km"]) for v in instance["vehicles"]["list"]]
    reqs = instance["requests"]["list"]
    pickups = [tuple(r["pickup_xy_km"]) for r in reqs]
    dropoffs = [tuple(r["dropoff_xy_km"]) for r in reqs]

    nodes_ij = []
    seen = set()
    for x, y in garages + pickups + dropoffs:
        i, j = snap_to_index(float(x), float(y), grid_graph_spec)
        if (i, j) not in seen:
            nodes_ij.append((i, j))
            seen.add((i, j))

    n = len(nodes_ij)
    T_min = np.zeros((n, n), dtype=np.float64)
    D_km = np.zeros((n, n), dtype=np.float64)

    for si, s in enumerate(nodes_ij):
        lengths, paths = nx.single_source_dijkstra(G, s, weight="time_h")
        for tj, t in enumerate(nodes_ij):
            if s == t:
                T_min[si, tj] = 0.0
                D_km[si, tj] = 0.0
                continue
            time_h = float(lengths.get(t, np.inf))
            T_min[si, tj] = 60.0 * time_h
            path = paths.get(t, None)
            if path is None:
                D_km[si, tj] = np.inf
            else:
                D_km[si, tj] = _path_distance_km(G, path)

    out_dir = ensure_dir(out_dir)
    nodes_i = np.array([ij[0] for ij in nodes_ij], dtype=np.int32)
    nodes_j = np.array([ij[1] for ij in nodes_ij], dtype=np.int32)
    save_npz(
        Path(out_dir) / "od.npz",
        {"T_min": T_min, "D_km": D_km, "nodes_i": nodes_i, "nodes_j": nodes_j}
    )

    write_json(
        Path(out_dir) / "od_meta.json",
        {"n_nodes": int(n), "source": str(instance_yaml)},
    )
    write_manifest(
        Path(out_dir) / "manifest.json",
        {"built_from": Path(instance_yaml).name}
    )


def _path_distance_km(G: nx.Graph, path: list[tuple[int, int]]) -> float:
    """
    Compute the distance in km of a path in the grid graph.
    """
    if len(path) <= 1:
        return 0.0
    acc = 0.0
    for a, b in zip(path[:-1], path[1:]):
        acc += float(G.edges[a, b]["distance_km"])
    return acc
