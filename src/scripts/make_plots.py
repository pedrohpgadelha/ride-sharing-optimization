"""
Make plots for the experiment results.
"""

from pathlib import Path

from src.config import PROJECT_ROOT, load_experiment
from src.io import ensure_dir, read_json
from src.plotting import plot_comparison_boxplots, plot_routes_on_grid


def make_plots() -> None:
    exp = load_experiment(PROJECT_ROOT / "configs" / "experiment_main.yaml")
    data_dir = Path(exp.get("output", {}).get("data_dir", "data"))
    plots_dir = ensure_dir(data_dir / "plots")
    routes_dir = ensure_dir(plots_dir / "routes")

    sols_root = data_dir / "solutions"
    if sols_root.exists():
        for sol_dir in sorted(sols_root.iterdir()):
            sol_json = sol_dir / "solution.json"
            if not sol_json.exists():
                continue
            meta = read_json(sol_json)
            scen = meta.get("scenario")
            seed = meta.get("seed")
            if scen is None or seed is None:
                continue
            inst_yaml = data_dir / "instances" / str(scen) / f"seed{seed}" / "instance.yaml"
            od_npz = data_dir / "od" / str(scen) / f"seed{seed}" / "od.npz"
            if not inst_yaml.exists() or not od_npz.exists():
                continue
            out_png = routes_dir / f"{meta.get('run_id', sol_dir.name)}.png"
            plot_routes_on_grid(inst_yaml, od_npz, sol_json, out_png)

    metrics_csvs = []
    m1 = data_dir / "metrics" / "baseline_runs.csv"
    m2 = data_dir / "metrics" / "grasp_vnd_runs.csv"
    if m1.exists():
        metrics_csvs.append(m1)
    if m2.exists():
        metrics_csvs.append(m2)
    if metrics_csvs:
        plot_comparison_boxplots(
            metrics_csv_paths=metrics_csvs,
            out_png=plots_dir / "comparison_boxplots.png",
            value_columns=[
                "served",
                "unserved",
                "wait_mean_min",
                "ride_time_extra_mean_min",
                "time_total_min",
                "dist_total_km",
                "cost",
                "cpu_time_sec",
            ],
        )


def main() -> None:
    make_plots()


if __name__ == "__main__":
    main()
