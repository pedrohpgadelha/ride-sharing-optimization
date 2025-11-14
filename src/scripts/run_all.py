"""
Run the entire experiment pipeline.

Steps:
01. Generate instances
02. Build OD matrix
03. Run baseline
04. Run GRASP+VND
05. Make plots
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts.generate_instances import generate_instances
from src.scripts.build_od import build_od
from src.scripts.run_baseline import run_baseline
from src.scripts.run_grasp_vnd import run_grasp_vnd
from src.scripts.make_plots import make_plots


def run_all() -> None:
    steps = [
        ("01. Generate instances", generate_instances),
        ("02. Build OD matrix", build_od),
        ("03. Run baseline", run_baseline),
        ("04. Run GRASP+VND", run_grasp_vnd),
        ("05. Make plots", make_plots),
    ]

    print("----------------------------------------------------------------------")
    print("Running the entire experiment pipeline...\n\n")

    for title, func in steps:
        print("----------------------------------------------------------------------")
        print(f"Running {title}...")
        
        func()

        print(f"{title} completed successfully!")

    print("\n\n----------------------------------------------------------------------")
    print("Pipeline completed successfully!")
    print("----------------------------------------------------------------------")


def main() -> None:
    run_all()


if __name__ == "__main__":
    main()
