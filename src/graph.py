"""
Graph utilities.
"""

from dataclasses import dataclass
from typing import Any
import networkx as nx

from src.zones import build_zones, edge_speed_kmph


@dataclass
class GridGraphSpecification:
    """
    Specification of a grid graph. The grid graph is a square grid with `size`
    times `size` nodes, with edges of length `step_km` between adjacent nodes.
    The grid is bounded by `x_min`, `x_max`, `y_min`, and `y_max`.
    """
    size: int
    step_km: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float


def build_grid_graph_spec(grid_cfg: dict[str, Any]) -> GridGraphSpecification:
    """
    Build a grid graph specification from a YAML extracted grid config.
    """
    size = int(grid_cfg["size"])
    step = float(grid_cfg["step_km"])
    x_min, x_max, y_min, y_max = map(float, grid_cfg["bounds_km"])
    return GridGraphSpecification(
        size=size,
        step_km=step,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )


def node_xy(
    i: int,
    j: int,
    spec: GridGraphSpecification,
) -> tuple[float, float]:
    """
    Get the (x, y) coordinates of a node in the grid graph.
    """
    return spec.x_min + i * spec.step_km, spec.y_min + j * spec.step_km


def snap_to_index(
    x_km: float,
    y_km: float,
    spec: GridGraphSpecification,
) -> tuple[int, int]:
    """
    Snap a point to the nearest node in the grid.
    """
    i = round((x_km - spec.x_min) / spec.step_km)
    j = round((y_km - spec.y_min) / spec.step_km)
    i = max(0, min(spec.size - 1, i))
    j = max(0, min(spec.size - 1, j))
    return i, j


def build_grid_graph(
    grid_cfg: dict[str, Any],
    speeds_cfg: dict[str, float],
    zones_cfg: list[dict[str, Any]],
) -> nx.Graph:
    """
    Build a grid graph in networkx format from a YAML extracted grid, speeds and
    zones config.
    """
    spec = build_grid_graph_spec(grid_cfg)
    zones = build_zones(zones_cfg)
    G = nx.Graph()
    for i in range(spec.size):
        for j in range(spec.size):
            x, y = node_xy(i, j, spec)
            G.add_node((i, j), x_km=x, y_km=y)
    for i in range(spec.size):
        for j in range(spec.size):
            u = (i, j)
            x_u, y_u = G.nodes[u]["x_km"], G.nodes[u]["y_km"]
            if i + 1 < spec.size:
                v = (i + 1, j)
                x_v, y_v = G.nodes[v]["x_km"], G.nodes[v]["y_km"]
                dist_km = spec.step_km
                speed_kmph = edge_speed_kmph(
                    x_u, y_u,
                    x_v, y_v,
                    speeds_cfg,
                    zones,
                )
                time_h = dist_km / speed_kmph
                G.add_edge(u, v, distance_km=dist_km, time_h=time_h)
            if j + 1 < spec.size:
                v = (i, j + 1)
                x_v, y_v = G.nodes[v]["x_km"], G.nodes[v]["y_km"]
                dist_km = spec.step_km
                speed_kmph = edge_speed_kmph(
                    x_u, y_u,
                    x_v, y_v,
                    speeds_cfg,
                    zones,
                )
                time_h = dist_km / speed_kmph
                G.add_edge(u, v, distance_km=dist_km, time_h=time_h)
    return G


def edge_time_h(G: nx.Graph, u: tuple[int, int], v: tuple[int, int]) -> float:
    """
    Get the displacement time of an edge in the grid graph.
    """
    return float(G.edges[u, v]["time_h"])


def edge_distance_km(
    G: nx.Graph,
    u: tuple[int, int],
    v: tuple[int, int],
) -> float:
    """
    Get the displacement distance of an edge in the grid graph.
    """
    return float(G.edges[u, v]["distance_km"])
