# Ride-Sharing Optimization

This repository contains the implementation of a static, batch, multi-vehicle ride-sharing optimization problem, developed as a project for the **MO824 – Topics in Combinatorial Optimization** course at **UNICAMP**.

The goal is to compare heuristic algorithms (a greedy baseline, GRASP, and GRASP + VND) on a family of synthetic ride-sharing instances defined on an explicit road grid with heterogeneous speeds and time windows.

---

## Problem Overview

We consider a **static batch** of ride requests and a fleet of vehicles:

- Each **request** has:
  - pickup and dropoff coordinates (in km),
  - an earliest pickup time `e_min`,
  - a latest dropoff time `l_max` (time window).

- Each **vehicle**:
  - starts and ends at a **garage** location,
  - has a fixed **capacity** (max number of passengers on board),
  - must serve a subsequence of requests (pickup then dropoff) without violating:
    - time windows,
    - capacity constraints,
    - precedence (pickup before dropoff).

- The road network is represented as a **grid graph**:
  - nodes are regularly spaced positions in km,
  - edges connect horizontal/vertical neighbors,
  - travel time depends on whether the edge lies inside or outside **zones**.

- **Zones**:
  - are circular regions with different `inside_zone_kmph` vs `outside_zone_kmph`,
  - induce heterogeneous speeds, modeling, e.g., CBD, uptown, suburbs.

### Objective

For a given solution (set of routes), we measure:

- total service time (sum of route durations),
- total distance traveled,
- number of unserved requests,
- average waiting time at pickup,
- average “extra” ride time vs direct travel,
- CPU time of the algorithm.

The main scalar **objective function** used in GRASP and VND is:

- Normalize time and distance by **baseline means** for the same scenario,
- Combine them with a weight `lambda` and a penalty `M` per unserved request:

> cost = λ · (time / mean_time) + (1 − λ) · (dist / mean_dist) + M · (#unserved)

The greedy baseline is evaluated with the same metrics but **does not** use this cost during construction.

---

## Repository Structure
```
configs/
  experiment_main.yaml       # Global experiment configuration
  scenario_1.yaml … scenario_10.yaml
                            # 10 scenario definitions (grid, vehicles, requests)

src/
  scripts/
    generate_instances.py    # Generate instances from scenario configs
    build_od.py              # Build OD matrices and recompute time windows
    run_baseline.py          # Run greedy insertion baseline over all instances
    run_grasp_vnd.py         # Run GRASP and GRASP+VND
    make_plots.py            # Generate visualization plots
    run_all.py               # Full pipeline (all steps in sequence)
  
  baseline.py                # Greedy insertion heuristic
  config.py                  # Config loading & validation
  feas.py                    # Feasibility checking & metrics aggregation
  graph.py                   # Grid graph construction and utilities
  grasp.py                   # GRASP implementation
  instances.py               # Instance generation from scenario configs
  io.py                      # I/O utilities (YAML/JSON/CSV/NPZ, manifests)
  metrics.py                 # Global metrics for solutions
  models.py                  # Core data models (Request, Vehicle, Route, Solution)
  objective.py               # Objective computation and baseline means
  od.py                      # OD matrix computation (time & distance)
  plotting.py                # Plot routes and algorithm comparison boxplots
  recalculate_windows.py     # Recompute `l_max` using OD network times
  vnd.py                     # VND improvement heuristic
  zones.py                   # Zone handling (speeds, membership)
  moves.py                   # Neighborhood move generators (relocate/swap)

data/
  instances/                 # Generated instances per scenario & seed
  od/                        # OD matrices per instance
  solutions/                 # JSON solutions and metrics per run
  metrics/                   # Summary CSVs and baseline means
  plots/                     # Route plots and comparison boxplots
```

> Note: the `data/` directory is created by the scripts and will be populated after running the pipeline.

---

## Installation

1. **Clone** this repository:

   ```bash
   git clone <your-repo-url>.git
   cd ride-sharing-optimization
   ```

2. Create and activate a **virtual environment** (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Linux/macOS
   # or on Windows:
   # .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

The code assumes **Python 3.10+** (for modern typing syntax).

---

## Configuration

### Experiment configuration

`configs/experiment_main.yaml` controls the global experiment:

* `scenarios`: list of scenario config files (`scenario_1.yaml` … `scenario_10.yaml`)
* `seeds`: list of random seeds (e.g., `[1, 2, 3, 4, 5]`)
* `objective`:

  * `lambda`: weight of normalized time vs distance
  * `penalty_M`: penalty per unserved request
* `runtime`:

  * `grasp_time_limit_sec`: time limit per instance for GRASP
  * `vnd_time_limit_sec`: time limit per instance for VND
* `grasp`:

  * `alpha`: RCL parameter (0 = greedy, 1 = fully random among candidates)
  * `multistart`: number of GRASP iterations
* `vnd`:

  * `enable`: whether to run the VND improvement step
  * `use_inter_route`: whether to consider inter-route moves
  * `neighborhoods_order`: order of neighborhoods
* `output.data_dir`: root directory for experiment outputs (defaults to `data/`)

### Scenario configuration

Each `configs/scenario_k.yaml` defines:

* the **grid**: size, step, bounds,
* **zones**: centers and radii,
* **speeds**: inside/outside zone km/h,
* **vehicles**: number, capacity, distribution of garages by zone,
* **requests**:

  * number of requests,
  * spatial distribution (`policentric_weights` + `uniform_weight`),
  * earliest pickup range `[e_min_min, e_max_min]`,
  * provisional `phi` and `kappa_min` used to approximate `l_max` (later recomputed with OD).

---

## Running the Pipeline

You can run the full experiment pipeline in one go:

```bash
python3 src/scripts/run_all.py
```

This executes the following steps:

1. **generate_instances**
   Reads all scenario configs and seeds, then generates `instance.yaml` files under:

   ```text
   data/instances/<scenario_name>/seed<seed>/instance.yaml
   ```

2. **build_od**
   For each instance:

   * builds the grid graph and computes shortest paths between all relevant nodes,
   * stores travel times (minutes) and distances (km) in `od.npz`,
   * recomputes each request’s `l_max` using the OD travel time and the parameters `phi` and `kappa_min`.

3. **run_baseline**
   Runs the greedy insertion heuristic:

   * constructs routes incrementally,
   * chooses the insertion that minimizes increase in total time (breaking ties by distance),
   * computes feasibility and metrics,
   * writes per-run solution + metrics and a summary CSV `baseline_runs.csv`,
   * also writes `baseline_means.json` with mean time and distance per scenario.

4. **run_grasp_vnd**
   For each instance:

   * runs **GRASP** with:

     * constructive phase using a cost aligned with the global objective (normalized time/distance),
     * `multistart` restarts,
     * per-instance time limit from `runtime.grasp_time_limit_sec`.
   * optionally runs **VND** starting from the best GRASP solution:

     * neighborhoods: relocate-intra, swap-intra, relocate-inter,
     * strategy: **first-improving** (stop scanning a neighborhood as soon as an improving move is found),
     * per-instance time limit from `runtime.vnd_time_limit_sec`.
   * saves solutions and metrics for both “grasp” and “grasp_vnd” runs into `grasp_vnd_runs.csv`.

5. **make_plots**

   * Generates **route plots** for every solution under `data/plots/routes/`.
   * Generates **comparison boxplots** over all algorithms and instances:

     * served / unserved requests,
     * mean waiting time,
     * mean extra ride time,
     * total time and distance,
     * objective cost,
     * CPU time (seconds).

---

## Outputs and Metrics

### Per-run metrics (JSON and CSV)

For each run (baseline, GRASP, GRASP+VND) we record:

* `served` – number of served requests,
* `unserved` – number of unserved requests,
* `wait_mean_min` – average waiting time at pickup (in minutes),
* `ride_time_extra_mean_min` – average extra in-vehicle time vs direct OD travel,
* `time_total_min` – total service time (sum over route durations, in minutes),
* `dist_total_km` – total distance traveled (km),
* `cost` – objective value (for GRASP/GRASP+VND),
* `cpu_time_sec` – total CPU time (per instance, per algorithm),
* `cpu_time_grasp_sec`, `cpu_time_vnd_sec` – decomposition for GRASP+VND runs.

These are aggregated across scenarios and seeds in:

* `data/metrics/baseline_runs.csv`
* `data/metrics/grasp_vnd_runs.csv`

### Route plots

Located under:

```text
data/plots/routes/<run_id>.png
```

Each plot shows:

* the grid,
* the zones (circles),
* the vehicle routes (polylines) over the OD nodes used in the solution.

### Boxplots

`data/plots/comparison_boxplots.png` summarizes the distribution of metrics across:

* all scenarios,
* all seeds,
* algorithms: baseline, grasp, grasp_vnd.

This provides a compact visual comparison of feasibility, quality, and CPU time.

---

## Reproducibility

* Randomness is controlled via:

  * the `seeds` list in `experiment_main.yaml` for instance generation,
  * per-instance `random_seed` passed to GRASP (equal to the instance seed).
* The OD construction and feasibility checks are deterministic given the instance.
* Results can be reproduced by:

  * keeping the same `configs/`,
  * re-running the pipeline with the same environment and Python version.

---

## Acknowledgements

This project was implemented as part of the **MO824 – Topics in Combinatorial Optimization** course at UNICAMP, under the guidance of the course instructors.

---

## License

This project is licensed under the MIT License.

Copyright (c) 2025 Pedro Henrique Pinheiro Gadêlha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in  
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN  
THE SOFTWARE.
