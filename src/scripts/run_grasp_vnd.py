"""
Run the GRASP+VND heuristics for all experiment instances.
"""

from dataclasses import asdict
from pathlib import Path
import time

from src.config import PROJECT_ROOT, load_experiment, load_scenario
from src.grasp import run_grasp
from src.io import (
    ensure_dir,
    make_run_id,
    write_csv_rows,
    write_json,
)
from src.vnd import vnd_improve


def run_grasp_vnd() -> None:
    exp = load_experiment(PROJECT_ROOT / "configs" / "experiment_main.yaml")
    data_dir = Path(exp.get("output", {}).get("data_dir", "data"))
    baseline_means_json = data_dir / "metrics" / "baseline_means.json"

    alpha = float(exp.get("grasp", {}).get("alpha", 0.3))
    multistart = int(exp.get("grasp", {}).get("multistart", 20))
    grasp_time_limit_sec = float(exp["runtime"]["grasp_time_limit_sec"])
    vnd_time_limit_sec = float(exp["runtime"]["vnd_time_limit_sec"])

    lambda_value = float(exp.get("objective", {}).get("lambda", 0.5))
    penalty_M = float(exp.get("objective", {}).get("penalty_M", 20))

    vnd_cfg = exp.get("vnd", {}) or {}
    vnd_enable = bool(vnd_cfg.get("enable", True))
    use_inter = bool(vnd_cfg.get("use_inter_route", True))
    nb_order = vnd_cfg.get("neighborhoods_order")

    scenarios = [s for s in exp["scenarios"]]
    seeds = [int(s) for s in exp["seeds"]]

    ensure_dir(data_dir / "solutions")
    ensure_dir(data_dir / "metrics")

    rows: list[dict[str, object]] = []

    for scenario in scenarios:
        print(f"    - scenario {scenario}")
        scen_cfg = load_scenario(PROJECT_ROOT / "configs" / scenario)
        scen_name = scen_cfg.get("name", Path(scenario).stem)

        for seed in seeds:
            print(f"        - seed {seed}")
            inst_dir = data_dir / "instances" / scen_name / f"seed{seed}"
            od_dir = data_dir / "od" / scen_name / f"seed{seed}"
            instance_yaml = inst_dir / "instance.yaml"
            od_npz = od_dir / "od.npz"

            run_id_grasp = make_run_id(
                scen_name,
                seed,
                {
                    "algo": "grasp",
                    "alpha": alpha,
                    "ms": multistart,
                    "tl": int(grasp_time_limit_sec)
                },
            )
            out_sol_dir = ensure_dir(data_dir / "solutions" / run_id_grasp)
            start_time = time.time()
            sol_g, met_g = run_grasp(
                instance_yaml,
                od_npz,
                baseline_means_json,
                lambda_value=lambda_value,
                penalty_M=penalty_M,
                alpha=alpha,
                multistart=multistart,
                time_limit_sec=grasp_time_limit_sec,
                random_seed=seed,
            )
            elapsed_grasp = time.time() - start_time

            cpu_time_grasp_sec = float(elapsed_grasp)
            cpu_time_vnd_sec = 0.0
            cpu_time_total_sec = cpu_time_grasp_sec + cpu_time_vnd_sec

            met_g_with_time = {
                **met_g,
                "cost": met_g.get("cost", float("nan")),
                "cpu_time_sec": cpu_time_total_sec,
                "cpu_time_grasp_sec": cpu_time_grasp_sec,
                "cpu_time_vnd_sec": cpu_time_vnd_sec,
            }

            write_json(
                out_sol_dir / "solution.json",
                {
                    "routes": {str(k): asdict(v) for k, v in sol_g.routes.items()},
                    "served_requests": sol_g.served_requests,
                    "unserved_requests": sol_g.unserved_requests,
                    "time_total_min": sol_g.time_total_min,
                    "dist_total_km": sol_g.dist_total_km,
                    "cost": sol_g.cost,
                    "feasible": sol_g.feasible,
                    "meta": sol_g.meta,
                    "scenario": scen_name,
                    "seed": int(seed),
                    "run_id": run_id_grasp,
                },
            )
            write_json(
                out_sol_dir / "metrics.json",
                {
                    "run_id": run_id_grasp,
                    "scenario": scen_name,
                    "seed": int(seed),
                    "algo": "grasp",
                    **met_g_with_time,
                },
            )
            rows.append(
                {
                    "run_id": run_id_grasp,
                    "scenario": scen_name,
                    "seed": int(seed),
                    "algo": "grasp",
                    **met_g_with_time,
                }
            )

            if vnd_enable:
                run_id_vnd = make_run_id(
                    scen_name,
                    seed,
                    {
                        "algo": "grasp_vnd",
                        "alpha": alpha,
                        "ms": multistart,
                        "tl": int(vnd_time_limit_sec),
                        "vnd": 1,
                    },
                )
                out_vnd_dir = ensure_dir(data_dir / "solutions" / run_id_vnd)
                vnd_start = time.time()
                sol_v, met_v = vnd_improve(
                    instance_yaml,
                    od_npz,
                    baseline_means_json,
                    routes=sol_g.routes,
                    lambda_value=lambda_value,
                    penalty_M=penalty_M,
                    neighborhoods_order=nb_order,
                    use_inter_route=use_inter,
                    time_limit_sec=vnd_time_limit_sec,
                )
                elapsed_vnd = time.time() - vnd_start

                cpu_time_vnd_sec = float(elapsed_vnd)
                cpu_time_total_sec = cpu_time_grasp_sec + cpu_time_vnd_sec

                met_v_with_time = {
                    **met_v,
                    "cost": met_v.get("cost", float("nan")),
                    "cpu_time_sec": cpu_time_total_sec,
                    "cpu_time_grasp_sec": cpu_time_grasp_sec,
                    "cpu_time_vnd_sec": cpu_time_vnd_sec,
                }

                write_json(
                    out_vnd_dir / "solution.json",
                    {
                        "routes": {str(k): asdict(v) for k, v in sol_v.routes.items()},
                        "served_requests": sol_v.served_requests,
                        "unserved_requests": sol_v.unserved_requests,
                        "time_total_min": sol_v.time_total_min,
                        "dist_total_km": sol_v.dist_total_km,
                        "cost": sol_v.cost,
                        "feasible": sol_v.feasible,
                        "meta": sol_v.meta,
                        "scenario": scen_name,
                        "seed": int(seed),
                        "run_id": run_id_vnd,
                    },
                )
                write_json(
                    out_vnd_dir / "metrics.json",
                    {
                        "run_id": run_id_vnd,
                        "scenario": scen_name,
                        "seed": int(seed),
                        "algo": "grasp_vnd",
                        **met_v_with_time,
                    },
                )
                rows.append(
                    {
                        "run_id": run_id_vnd,
                        "scenario": scen_name,
                        "seed": int(seed),
                        "algo": "grasp_vnd",
                        **met_v_with_time,
                    }
                )

    write_csv_rows(
        data_dir / "metrics" / "grasp_vnd_runs.csv",
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
            "cost",
            "cpu_time_sec",
            "cpu_time_grasp_sec",
            "cpu_time_vnd_sec",
        ],
    )


def main() -> None:
    run_grasp_vnd()


if __name__ == "__main__":
    main()
