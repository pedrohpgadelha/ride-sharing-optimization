"""
Functions to compute the metrics of a solution.
"""

from pathlib import Path
import numpy as np

from src.feas import evaluate_solution
from src.io import read_yaml
from src.models import Request, Route


def build_requests_by_id(instance: dict) -> dict[int, Request]:
    """
    Build a dictionary of requests by their id from an instance.
    """
    out: dict[int, Request] = {}
    for r in instance["requests"]["list"]:
        out[int(r["id"])] = Request(
            id=int(r["id"]),
            pickup_xy_km=tuple(r["pickup_xy_km"]),
            dropoff_xy_km=tuple(r["dropoff_xy_km"]),
            e_min=float(r["e_min"]),
            l_max=float(r["l_max"]),
        )
    return out


def summarize_solution(
    instance_yaml: str | Path,
    T_min: np.ndarray,
    D_km: np.ndarray,
    routes: dict[int, Route],
) -> dict[str, float]:
    """
    Summarize the global metrics of a solution.
    """
    instance = read_yaml(instance_yaml)
    cap = int(instance["vehicles"]["capacity"])
    reqs = build_requests_by_id(instance)
    res = evaluate_solution(routes, reqs, cap, T_min, D_km)
    return {
        "served": float(len(res.served_requests)),
        "unserved": float(len(res.unserved_requests)),
        "wait_mean_min": float(res.wait_mean_min),
        "ride_time_extra_mean_min": float(res.ride_time_extra_mean_min),
        "time_total_min": float(res.time_total_min),
        "dist_total_km": float(res.dist_total_km),
    }
