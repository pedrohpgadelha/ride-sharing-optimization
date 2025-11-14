"""
Experiment and scenario configuration loading and validation utilities.
"""

from pathlib import Path
from typing import Any

from src.io import read_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Project root directory.
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"       # Default configuration directory.


def resolve_config_path(name_or_path: str | Path) -> Path:
    """
    Resolve a YAML config file path.
    """
    p = Path(name_or_path)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p
    if not p.suffix:
        candidate = DEFAULT_CONFIG_DIR / f"{p.name}.yaml"
        if candidate.exists():
            return candidate
    candidate = DEFAULT_CONFIG_DIR / p.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Config not found: {name_or_path}")


def load_yaml_config(name_or_path: str | Path) -> dict:
    """
    Load a YAML config file content.
    """
    path = resolve_config_path(name_or_path)
    cfg = read_yaml(path)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML at {path}")
    return cfg


def load_scenario(name_or_path: str | Path) -> dict:
    """
    Load a scenario YAML config file content.
    """
    cfg = load_yaml_config(name_or_path)
    _validate_scenario(cfg)
    return cfg


def load_experiment(name_or_path: str | Path = "experiment_main.yaml") -> dict:
    """
    Load an experiment YAML config file content.
    """
    cfg = load_yaml_config(name_or_path)
    _validate_experiment(cfg)
    return cfg


def _validate_scenario(cfg: dict[str, Any]) -> None:
    """
    Validate a scenario YAML config file content. Requires:
        - grid: positive size; bounds_km with 4 elements.
        - speeds: inside_zone_kmph and outside_zone_kmph.
        - zones: non-empty list of zones.
        - vehicles: positive n_vehicles and capacity.
        - requests: positive n_requests; phi, kappa_min, e_min_min, e_max_min.
    """
    required_top = ["grid", "speeds", "zones", "vehicles", "requests"]
    for k in required_top:
        if k not in cfg:
            raise ValueError(f"scenario missing key: {k}")
    g = cfg["grid"]
    if not isinstance(g.get("size"), int) or g["size"] <= 0:
        raise ValueError("grid.size must be a positive int")
    if "bounds_km" not in g or len(g["bounds_km"]) != 4:
        raise ValueError("grid.bounds_km must be [x_min, x_max, y_min, y_max]")
    s = cfg["speeds"]
    if "inside_zone_kmph" not in s or "outside_zone_kmph" not in s:
        raise ValueError("speeds must define inside_zone_kmph and outside_zone_kmph")
    if not cfg["zones"]:
        raise ValueError("zones must be non-empty")
    v = cfg["vehicles"]
    if not isinstance(v.get("n_vehicles"), int) or v["n_vehicles"] <= 0:
        raise ValueError("vehicles.n_vehicles must be a positive int")
    if not isinstance(v.get("capacity"), int) or v["capacity"] <= 0:
        raise ValueError("vehicles.capacity must be a positive int")
    r = cfg["requests"]
    if not isinstance(r.get("n_requests"), int) or r["n_requests"] <= 0:
        raise ValueError("requests.n_requests must be a positive int")
    for kk in ["phi", "kappa_min", "e_min_min", "e_max_min"]:
        if kk not in r:
            raise ValueError(f"requests missing key: {kk}")


def _validate_experiment(cfg: dict[str, Any]) -> None:
    """
    Validate an experiment YAML config file content. Requires:
        - scenarios: non-empty list of scenario paths.
        - seeds: non-empty list of seeds.
        - objective: lambda and penalty_M.
        - runtime: grasp_time_limit_sec and vnd_time_limit_sec.
    """
    if "scenarios" not in cfg or not cfg["scenarios"]:
        raise ValueError("experiment.scenarios must be non-empty")
    if "seeds" not in cfg or not cfg["seeds"]:
        raise ValueError("experiment.seeds must be non-empty")
    obj = cfg.get("objective", {})
    if "lambda" not in obj or "penalty_M" not in obj:
        raise ValueError("experiment.objective must define lambda and penalty_M")
    rt = cfg.get("runtime", {})
    if "grasp_time_limit_sec" not in rt or "vnd_time_limit_sec" not in rt:
        raise ValueError(
            "experiment.runtime must define grasp_time_limit_sec and vnd_time_limit_sec"
        )
