"""
Run the greedy insertion baseline heuristic for all experiment instances.
"""

from dataclasses import asdict
from pathlib import Path
import time

from src.baseline import run_greedy_insertion
from src.config import PROJECT_ROOT, load_experiment, load_scenario
from src.io import (
    ensure_dir,
    make_run_id,
    write_csv_rows,
    write_json,
)


def run_baseline() -> None:
    exp = load_experiment(PROJECT_ROOT / "configs" / "experiment_main.yaml")
    data_dir = Path(exp.get("output", {}).get("data_dir", "data"))

    scenarios = [s for s in exp["scenarios"]]
    seeds = [int(s) for s in exp["seeds"]]

    rows: list[dict[str, object]] = []
    sums: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}

    ensure_dir(data_dir / "solutions")
    ensure_dir(data_dir / "metrics")

    for scenario in scenarios:
        print(f"    - scenario {scenario}")
        scen_cfg = load_scenario(PROJECT_ROOT / "configs" / scenario)
        scen_name = scen_cfg.get("name", Path(scenario).stem)
        sums.setdefault(
            scen_name,
            {"time_total_min": 0.0, "dist_total_km": 0.0}
        )
        counts.setdefault(scen_name, 0)

        for seed in seeds:
            print(f"        - seed {seed}")
            inst_dir = data_dir / "instances" / scen_name / f"seed{seed}"
            od_dir = data_dir / "od" / scen_name / f"seed{seed}"
            instance_yaml = inst_dir / "instance.yaml"
            od_npz = od_dir / "od.npz"

            run_id = make_run_id(scen_name, seed, {"algo": "baseline"})
            out_sol_dir = ensure_dir(data_dir / "solutions" / run_id)

            start_time = time.time()
            solution, metrics = run_greedy_insertion(
                instance_yaml,
                od_npz,
            )
            elapsed_time = time.time() - start_time
            metrics["cpu_time_sec"] = float(elapsed_time)

            sol_dict = {
                "routes": {
                    str(k): asdict(v) for k, v in solution.routes.items()
                },
                "served_requests": solution.served_requests,
                "unserved_requests": solution.unserved_requests,
                "time_total_min": solution.time_total_min,
                "dist_total_km": solution.dist_total_km,
                "cost": solution.cost,
                "feasible": solution.feasible,
                "meta": solution.meta,
                "scenario": scen_name,
                "seed": int(seed),
                "run_id": run_id,
            }
            write_json(out_sol_dir / "solution.json", sol_dict)
            write_json(
                out_sol_dir / "metrics.json",
                {
                    "run_id": run_id,
                    "scenario": scen_name,
                    "seed": int(seed),
                    "algo": "baseline",
                    **metrics,
                },
            )

            rows.append(
                {
                    "run_id": run_id,
                    "scenario": scen_name,
                    "seed": int(seed),
                    "algo": "baseline",
                    **metrics,
                }
            )

            sums[scen_name]["time_total_min"] += float(metrics["time_total_min"])
            sums[scen_name]["dist_total_km"] += float(metrics["dist_total_km"])
            counts[scen_name] += 1

    write_csv_rows(
        data_dir / "metrics" / "baseline_runs.csv",
        rows,
        fieldnames=[
            "run_id",
            "scenario",
            "seed",
            "algo",
            "served",
            "unserved",
            "wait_mean_min",
            "ride_time_extra_mean_min",
            "time_total_min",
            "dist_total_km",
            "cpu_time_sec",
        ],
    )

    means = {}
    for scen_name, s in sums.items():
        c = max(1, counts.get(scen_name, 0))
        means[scen_name] = {
            "time_total_min": s["time_total_min"] / c,
            "dist_total_km": s["dist_total_km"] / c,
        }
    write_json(data_dir / "metrics" / "baseline_means.json", means)


def main() -> None:
    run_baseline()


if __name__ == "__main__":
    main()
