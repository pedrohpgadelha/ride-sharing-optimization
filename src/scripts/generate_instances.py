"""
Generate all experiment instances.
"""

from pathlib import Path

from src.config import load_experiment, load_scenario, PROJECT_ROOT
from src.instances import generate_instances_for_scenario
from src.io import ensure_dir


def generate_instances() -> None:
    experiment_cfg = load_experiment(
        PROJECT_ROOT / "configs" / "experiment_main.yaml"
    )
    data_dir = ensure_dir(
        Path(experiment_cfg.get("output", {}).get("data_dir", "data"))
    )
    scenarios = experiment_cfg["scenarios"]
    seeds = [int(s) for s in experiment_cfg["seeds"]]
    for scenario in scenarios:
        print(f"    - {scenario}")
        scenario_cfg = load_scenario(PROJECT_ROOT / "configs" / scenario)
        generate_instances_for_scenario(scenario_cfg, seeds, data_dir)


def main() -> None:
    generate_instances()


if __name__ == "__main__":
    main()
