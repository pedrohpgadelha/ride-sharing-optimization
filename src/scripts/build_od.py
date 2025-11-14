"""
Build the OD matrix for all experiment instances.
"""

from pathlib import Path

from src.config import load_experiment, PROJECT_ROOT, load_scenario
from src.io import ensure_dir
from src.od import compute_and_save_od
from src.recalculate_windows import recompute_dropoff_windows_with_od


def build_od() -> None:
    exp = load_experiment(PROJECT_ROOT / "configs" / "experiment_main.yaml")
    data_dir = Path(exp.get("output", {}).get("data_dir", "data"))
    scenarios = exp["scenarios"]
    seeds = [int(s) for s in exp["seeds"]]

    for scenario in scenarios:
        print(f"    - scenario {scenario}")
        scen_cfg = load_scenario(PROJECT_ROOT / "configs" / scenario)
        scen_name = scen_cfg.get("name", Path(scenario).stem)
        phi = float(scen_cfg["requests"]["phi"])
        kappa = float(scen_cfg["requests"]["kappa_min"])
        for seed in seeds:
            print(f"        - seed {seed}")
            inst_dir = (PROJECT_ROOT / data_dir / "instances" / scen_name / f"seed{seed}")
            od_dir = PROJECT_ROOT / data_dir / "od" / scen_name / f"seed{seed}"
            ensure_dir(od_dir)
            compute_and_save_od(inst_dir / "instance.yaml", od_dir)

            recompute_dropoff_windows_with_od(
                instance_yaml=inst_dir / "instance.yaml",
                od_npz=od_dir / "od.npz",
                phi=phi,
                kappa=kappa,
            )


def main() -> None:
    build_od()


if __name__ == "__main__":
    main()
