"""
Functions to compute the objective value of a solution.
"""

from os import PathLike
from pathlib import Path
from typing import Any

from src.io import read_json


def compute_objective(
    time_total_min: float,
    dist_total_km: float,
    lambda_value: float,
    mean_time_total_min: float,
    mean_dist_total_km: float,
    penalty_M: float,
    n_unserved: int,
) -> tuple[float, dict[str, float]]:
    """
    Compute the objective value of a solution.
    """
    t_den = max(1e-6, float(mean_time_total_min))
    d_den = max(1e-6, float(mean_dist_total_km))
    t_comp = float(time_total_min) / t_den
    d_comp = float(dist_total_km) / d_den
    pen = penalty_M * float(n_unserved)
    cost = lambda_value * t_comp + (1.0 - lambda_value) * d_comp + pen
    return cost, {
        "time_norm": t_comp,
        "dist_norm": d_comp,
        "penalty": pen,
        "lambda": lambda_value,
        "cost": cost,
    }


def load_baseline_means(path: str | Path | PathLike[str]) -> dict[str, Any]:
    """
    Load the baseline mean time and distance for a scenario from a JSON file.
    """
    data = read_json(path)
    return data


def means_for_scenario(
    baseline_means: dict[str, Any],
    scenario_name: str,
) -> tuple[float, float]:
    """
    Compute the mean time and distance for a scenario.
    """
    entry = baseline_means.get(scenario_name)
    if not entry:
        return 1.0, 1.0
    m_t = float(entry.get("time_total_min", 1.0))
    m_d = float(entry.get("dist_total_km", 1.0))
    m_t = m_t if m_t > 0 else 1.0
    m_d = m_d if m_d > 0 else 1.0
    return m_t, m_d
