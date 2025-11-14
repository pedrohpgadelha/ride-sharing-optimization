"""
Functions to generate instances for a given scenario.
"""

from math import sqrt
from pathlib import Path
import numpy as np

from src.io import ensure_dir, write_yaml, write_manifest


def generate_instances_for_scenario(
    scenario_cfg: dict,
    seeds: list[int],
    data_dir: str | Path,
) -> None:
    """
    Generate instances for a given scenario with a list of seeds. For each seed,
    writes in the data directory the complete instance in instance.yaml and
    useful metadata in manifest.json.

    data/
        instances/
            <scenario_name>/
                <seed0>/
                    instance.yaml
                    manifest.json
                <seed1>/
                    ...
    """
    scenario_name = scenario_cfg.get("name", "scenario")
    base_dir = ensure_dir(Path(data_dir) / "instances" / scenario_name)
    for seed in seeds:
        instance = generate_instance(scenario_cfg, int(seed))
        out_dir = ensure_dir(base_dir / f"seed{seed}")
        write_yaml(out_dir / "instance.yaml", instance)
        write_manifest(
            out_dir / "manifest.json",
            {
                "scenario": scenario_name,
                "seed": int(seed),
                "counts": {
                    "n_requests": len(instance["requests"]["list"]),
                    "n_vehicles": len(instance["vehicles"]["list"]),
                },
            },
        )


def generate_instance(scenario_cfg: dict, seed: int) -> dict:
    """
    Generate an instance for a given scenario with a given seed.
    """
    rng = np.random.default_rng(seed)

    grid = scenario_cfg["grid"]
    speeds = scenario_cfg["speeds"]
    zones = scenario_cfg["zones"]
    vehicles_cfg = scenario_cfg["vehicles"]
    requests_cfg = scenario_cfg["requests"]

    x_min, x_max, y_min, y_max = map(float, grid["bounds_km"])
    garages_by_zone: dict[str, int] = vehicles_cfg.get("garages_by_zone")
    n_vehicles = int(vehicles_cfg["n_vehicles"])
    capacity = int(vehicles_cfg["capacity"])

    if sum(garages_by_zone.values()) != n_vehicles:
        raise ValueError(
            "vehicles.garages_by_zone must sum to vehicles.n_vehicles"
        )

    zone_centers = {
        zone.get("id", f"Z{idx}"): tuple(zone["center"])
        for idx, zone in enumerate(zones)
    }

    vehicles_list = []
    vehicle_id = 0
    for zone_id, garage_count in garages_by_zone.items():
        center_x, center_y = zone_centers[zone_id]
        for _ in range(int(garage_count)):
            vehicles_list.append(
                {
                    "id": vehicle_id,
                    "garage_xy_km": [float(center_x), float(center_y)],
                    "capacity": capacity,
                }
            )
            vehicle_id += 1

    n_requests = int(requests_cfg["n_requests"])
    weights = requests_cfg["policentric_weights"]
    uniform_w = float(requests_cfg["uniform_weight"])
    e_min_min = float(requests_cfg["e_min_min"])
    e_max_min = float(requests_cfg["e_max_min"])
    phi = float(requests_cfg["phi"])
    kappa_min = float(requests_cfg["kappa_min"])

    centers = [
        (
            zone.get("id", f"Z{idx}"),
            tuple(map(float, zone["center"]))
        ) for idx, zone in enumerate(zones)
    ]
    center_ids = [c[0] for c in centers]
    center_lookup = {c[0]: c[1] for c in centers}
    center_w = np.array(
        [float(weights.get(cid, 0.0)) for cid in center_ids], dtype=float
    )

    requests_list = []
    for rid in range(n_requests):
        # Pickup point
        px, py = _sample_point_km(
            rng,
            x_min, x_max,
            y_min, y_max,
            center_ids,
            center_w,
            uniform_w,
            center_lookup,
        )
        # Dropoff point
        dx, dy = _sample_point_km(
            rng,
            x_min, x_max,
            y_min, y_max,
            center_ids,
            center_w,
            uniform_w,
            center_lookup,
        )

        # Earliest pickup time
        e_min = float(rng.uniform(e_min_min, e_max_min))
        # Estimate travel time
        v_avg_kmph = 0.5 * (
            float(speeds["inside_zone_kmph"]) +
            float(speeds["outside_zone_kmph"])
        )
        dist_km = sqrt((dx - px) ** 2 + (dy - py) ** 2)
        t_est_min = 60.0 * (dist_km / max(v_avg_kmph, 1e-6))
        # Latest dropoff time
        l_max = e_min + phi * t_est_min + kappa_min

        requests_list.append(
            {
                "id": rid,
                "pickup_xy_km": [float(px), float(py)],
                "dropoff_xy_km": [float(dx), float(dy)],
                "e_min": float(e_min),
                "l_max": float(l_max),
            }
        )

    instance = {
        "name": scenario_cfg.get("name", "scenario"),
        "seed": seed,
        "grid": grid,
        "speeds": speeds,
        "zones": zones,
        "vehicles": {
            "capacity": capacity,
            "list": vehicles_list,
        },
        "requests": {
            "list": requests_list,
        },
    }
    return instance


def _sample_point_km(
    rng: np.random.Generator,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    center_ids: list[str],
    center_w: np.ndarray,
    uniform_w: float,
    center_lookup: dict[str, tuple[float, float]],
    sigma_km: float = 1.5,
) -> tuple[float, float]:
    """
    Sample a random point from the grid graph.
    """
    u = max(0.0, min(1.0, float(uniform_w)))
    total_centers = float(center_w.sum())

    if total_centers <= 0.0:
        u = 1.0

    if rng.uniform() < u:
        ux = float(rng.uniform(x_min, x_max))
        uy = float(rng.uniform(y_min, y_max))
        return ux, uy

    p = center_w / total_centers
    idx = int(rng.choice(len(center_ids), p=p))
    cx, cy = center_lookup[center_ids[idx]]
    for _ in range(100):
        x = float(rng.normal(cx, sigma_km))
        y = float(rng.normal(cy, sigma_km))
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return x, y

    x = float(min(max(x_min, x), x_max))
    y = float(min(max(y_min, y), y_max))
    return x, y
