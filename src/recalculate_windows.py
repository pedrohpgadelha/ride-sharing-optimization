from pathlib import Path
from typing import Iterable
import numpy as np

from src.graph import build_grid_graph_spec, snap_to_index
from src.io import read_yaml, write_yaml


def recompute_dropoff_windows_with_od(
    instance_yaml: str | Path,
    od_npz: str | Path,
    phi: float,
    kappa: float,
) -> None:
    """
    Recompute l_max for each request using network travel time from OD matrix.
    """
    instance_path = Path(instance_yaml)
    od_path = Path(od_npz)
    instance = read_yaml(instance_path)
    od = np.load(od_path)
    T_min = od["T_min"]
    nodes_i = od["nodes_i"]
    nodes_j = od["nodes_j"]

    spec = build_grid_graph_spec(instance["grid"])
    node_idx_by_ij = {
        (int(i), int(j)): idx for idx, (i, j) in enumerate(zip(nodes_i, nodes_j))
    }

    def idx_of_xy(xy: Iterable[float]) -> int:
        i, j = snap_to_index(float(xy[0]), float(xy[1]), spec)
        return int(node_idx_by_ij[(i, j)])

    for r in instance["requests"]["list"]:
        p_idx = idx_of_xy(r["pickup_xy_km"])
        d_idx = idx_of_xy(r["dropoff_xy_km"])
        t_pd = float(T_min[p_idx, d_idx])
        e_min = float(r["e_min"])
        new_l_max = e_min + float(phi) * t_pd + float(kappa)
        r["l_max"] = new_l_max

    write_yaml(instance_path, instance)
