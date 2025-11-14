"""
Plotting utilities.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.graph import build_grid_graph_spec, node_xy
from src.io import read_json, read_yaml


def plot_routes_on_grid(
    instance_yaml: str | Path,
    od_npz: str | Path,
    solution_json: str | Path,
    out_png: str | Path,
) -> None:
    """
    Plot the routes on the grid.
    """
    instance = read_yaml(instance_yaml)
    spec = build_grid_graph_spec(instance["grid"])
    zones = instance.get("zones", [])

    od = np.load(od_npz)
    nodes_i = od["nodes_i"]
    nodes_j = od["nodes_j"]
    idx_to_xy = {}
    for idx, (i, j) in enumerate(zip(nodes_i, nodes_j)):
        x, y = node_xy(int(i), int(j), spec)
        idx_to_xy[int(idx)] = (x, y)

    sol = read_json(solution_json)
    routes = sol["routes"]

    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(spec.size):
        xs = [spec.x_min, spec.x_max]
        y = spec.y_min + i * spec.step_km
        ax.plot(xs, [y, y], linewidth=0.5, alpha=0.2)
        ys = [spec.y_min, spec.y_max]
        x = spec.x_min + i * spec.step_km
        ax.plot([x, x], ys, linewidth=0.5, alpha=0.2)

    for z in zones:
        from matplotlib.patches import Circle

        cx, cy = z["center"]
        r = z["radius_km"]
        circle = Circle(
            (cx, cy),
            r,
            fill=False,
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
        )
        ax.add_patch(circle)

    for vid_str, route_dict in routes.items():
        evs = route_dict["events"]
        xs, ys = [], []
        for ev in evs:
            idx = int(ev["node_idx"])
            x, y = idx_to_xy[idx]
            xs.append(x)
            ys.append(y)
        if len(xs) >= 2:
            ax.plot(xs, ys, linewidth=2.0, alpha=0.8, marker="o", markersize=3)

    ax.set_xlim(spec.x_min, spec.x_max)
    ax.set_ylim(spec.y_min, spec.y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Rotas por veículo")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_comparison_boxplots(
    metrics_csv_paths: list[str | Path],
    out_png: str | Path,
    value_columns: list[str] | None = None,
) -> None:
    """
    Plot the comparison boxplots.
    """
    frames = []
    for p in metrics_csv_paths:
        df = pd.read_csv(p)
        frames.append(df)
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    if value_columns is None:
        value_columns = ["served", "unserved", "time_total_min", "dist_total_km"]
    fig, axes = plt.subplots(
        1,
        len(value_columns),
        figsize=(4 * len(value_columns), 4),
    )
    if len(value_columns) == 1:
        axes = [axes]
    for ax, col in zip(axes, value_columns):
        df.boxplot(column=col, by="algo", ax=ax)
        ax.set_title(col)
        ax.set_xlabel("algo")
        ax.set_ylabel(col)
    fig.suptitle("")
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
