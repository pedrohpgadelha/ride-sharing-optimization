"""
Greedy randomized adaptive search procedure (GRASP) heuristic.
"""

import random
import time
from pathlib import Path
from typing import Any
import numpy as np

from src.feas import FeasRouteResult, evaluate_route, evaluate_solution
from src.graph import build_grid_graph_spec, snap_to_index
from src.io import read_json, read_yaml
from src.models import Event, Request, Route, Solution
from src.objective import compute_objective, means_for_scenario
from src.metrics import summarize_solution


def run_grasp(
    instance_yaml: str | Path,
    od_npz: str | Path,
    baseline_means_json: str | Path,
    *,
    lambda_value: float,
    penalty_M: float,
    alpha: float = 0.3,
    multistart: int = 20,
    time_limit_sec: float | None = None,
    random_seed: int | None = None,
) -> tuple[Solution, dict[str, Any]]:
    """
    Run the GRASP heuristic for an instance.
    """
    if random_seed is not None:
        random.seed(random_seed)

    instance = read_yaml(instance_yaml)
    od = np.load(Path(od_npz))
    T_min = od["T_min"]
    D_km = od["D_km"]
    nodes_i = od["nodes_i"]
    nodes_j = od["nodes_j"]
    node_idx_by_ij = {
        (int(i), int(j)): idx for idx, (i, j) in enumerate(zip(nodes_i, nodes_j))
    }
    spec = build_grid_graph_spec(instance["grid"])

    def idx_of_xy(xy: tuple[float, float]) -> int:
        i, j = snap_to_index(float(xy[0]), float(xy[1]), spec)
        return int(node_idx_by_ij[(i, j)])

    scenario_name = instance.get("name", "scenario")
    baseline_means = read_json(baseline_means_json)
    mean_time, mean_dist = means_for_scenario(baseline_means, scenario_name)

    vehicles = instance["vehicles"]["list"]
    capacity = int(instance["vehicles"]["capacity"])

    requests_list = instance["requests"]["list"]
    requests_by_id: dict[int, Request] = {}
    for r in requests_list:
        requests_by_id[int(r["id"])] = Request(
            id=int(r["id"]),
            pickup_xy_km=tuple(r["pickup_xy_km"]),
            dropoff_xy_km=tuple(r["dropoff_xy_km"]),
            e_min=float(r["e_min"]),
            l_max=float(r["l_max"]),
        )

    best_solution: Solution | None = None
    best_cost = float("inf")
    deadline = (time.time() + time_limit_sec) if time_limit_sec else None

    for _ in range(multistart):
        routes: dict[int, Route] = {}
        route_eval: dict[int, FeasRouteResult] = {}
        for v in vehicles:
            vid = int(v["id"])
            garage_idx = idx_of_xy(tuple(v["garage_xy_km"]))
            evs = [
                Event(
                    kind="garage",
                    request_id=-1,
                    node_idx=garage_idx,
                    arrival_min=0.0,
                ),
                Event(kind="garage", request_id=-1, node_idx=garage_idx),
            ]
            r = Route(vehicle_id=vid, events=evs)
            routes[vid] = r
            route_eval[vid] = evaluate_route(
                r,
                requests_by_id,
                capacity,
                T_min,
                D_km,
            )

        unplaced = set(int(r["id"]) for r in requests_list)
        while unplaced:
            if deadline and time.time() >= deadline:
                break
            cands: list[tuple[float, int, int, int, int, Route, FeasRouteResult]] = []
            for rid in list(unplaced):
                rq = requests_by_id[rid]
                for vid, route in routes.items():
                    cur = route_eval[vid]
                    n = len(route.events)
                    for p in range(1, n):
                        for d in range(p + 1, n + 1):
                            cand_route = _insert_pair(
                                route,
                                rq,
                                idx_of_xy,
                                p,
                                d,
                            )
                            rr = evaluate_route(
                                cand_route,
                                requests_by_id,
                                capacity,
                                T_min,
                                D_km,
                            )
                            if not rr.feasible:
                                continue
                            dt = rr.time_total_min - cur.time_total_min
                            dd = rr.dist_total_km - cur.dist_total_km
                            delta_norm = (
                                lambda_value * (dt / max(mean_time, 1e-6)) + 
                                (1.0 - lambda_value) * (dd / max(mean_dist, 1e-6))
                            )
                            cands.append((
                                float(delta_norm),
                                rid,
                                vid,
                                p,
                                d,
                                cand_route,
                                rr,
                            ))
            if not cands:
                break
            cands.sort(key=lambda x: x[0])
            min_delta = cands[0][0]
            max_delta = cands[-1][0]
            thr = min_delta + alpha * (max_delta - min_delta)
            rcl = [c for c in cands if c[0] <= thr]
            chosen = random.choice(rcl)
            _, rid, vid, p, d, cand_route, rr = chosen
            routes[vid] = cand_route
            route_eval[vid] = rr
            unplaced.remove(rid)

        sol_eval = evaluate_solution(
            routes,
            requests_by_id,
            capacity,
            T_min,
            D_km,
        )
        cost, breakdown = compute_objective(
            sol_eval.time_total_min,
            sol_eval.dist_total_km,
            lambda_value,
            mean_time,
            mean_dist,
            penalty_M,
            n_unserved=len(sol_eval.unserved_requests),
        )
        solution = Solution(
            routes=routes,
            served_requests=sorted(list(sol_eval.served_requests)),
            unserved_requests=sorted(list(sol_eval.unserved_requests)),
            time_total_min=sol_eval.time_total_min,
            dist_total_km=sol_eval.dist_total_km,
            cost=cost,
            feasible=sol_eval.feasible,
            meta={
                "lambda": lambda_value,
                "alpha": alpha,
                "strategy": "grasp",
                "objective": breakdown,
            },
        )

        if cost < best_cost:
            best_cost = cost
            best_solution = solution

        if deadline and time.time() >= deadline:
            break

    metrics = summarize_solution(
        instance_yaml,
        T_min,
        D_km,
        best_solution.routes,
    )
    metrics["cost"] = (
        best_solution.cost if best_solution.cost is not None else float("nan")
    )
    return best_solution, metrics


def _insert_pair(
    route: Route,
    req: Request,
    idx_of_xy,
    pick_pos: int,
    drop_pos: int,
) -> Route:
    """
    Insert a request into a route at the given pickup and dropoff positions.
    """
    p_idx = idx_of_xy(req.pickup_xy_km)
    d_idx = idx_of_xy(req.dropoff_xy_km)
    evs = list(route.events)
    evs.insert(pick_pos, Event(kind="pickup", request_id=req.id, node_idx=p_idx))
    evs.insert(drop_pos, Event(kind="dropoff", request_id=req.id, node_idx=d_idx))
    return Route(vehicle_id=route.vehicle_id, events=evs)
