"""
Greedy constructive heuristic baseline function.
"""

from pathlib import Path
from typing import Any
import numpy as np

from src.feas import FeasRouteResult, evaluate_route, evaluate_solution
from src.graph import build_grid_graph_spec, snap_to_index
from src.io import read_yaml
from src.models import Event, Request, Route, Solution


def run_greedy_insertion(
    instance_yaml: str | Path,
    od_npz: str | Path,
) -> tuple[Solution, dict[str, Any]]:
    """
    Run the greedy insertion baseline solution for an instance by iteratively
    inserting requests into all routes by choosing the best time increment (ties
    are solved by choosing the best distance increment).
    """
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

    vehicles = instance["vehicles"]["list"]
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

    garage_idx_by_vehicle: dict[int, int] = {}
    for v in vehicles:
        vid = int(v["id"])
        garage_idx_by_vehicle[vid] = idx_of_xy(tuple(v["garage_xy_km"]))

    routes: dict[int, Route] = {}
    route_eval: dict[int, FeasRouteResult] = {}
    for v in vehicles:
        vid = int(v["id"])
        garage_idx = garage_idx_by_vehicle[vid]
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
            int(instance["vehicles"]["capacity"]),
            T_min,
            D_km,
        )

    order = _order_requests_by_slack(requests_by_id, T_min, idx_of_xy)
    served: set[int] = set()
    unserved: set[int] = set()

    for rid in order:
        best = None
        best_vid = None
        best_route = None
        for vid, route in routes.items():
            current = route_eval[vid]
            for pick_pos in range(1, len(route.events)):
                for drop_pos in range(pick_pos + 1, len(route.events) + 1):
                    candidate = _insert_pair(
                        route,
                        requests_by_id[rid],
                        idx_of_xy,
                        pick_pos,
                        drop_pos,
                    )
                    rr = evaluate_route(
                        candidate,
                        requests_by_id,
                        int(instance["vehicles"]["capacity"]),
                        T_min,
                        D_km,
                    )
                    if not rr.feasible:
                        continue
                    dt = rr.time_total_min - current.time_total_min
                    dd = rr.dist_total_km - current.dist_total_km
                    key = (round(dt, 6), round(dd, 6))
                    if best is None or key < best[0]:
                        best = (key, rr)
                        best_vid = vid
                        best_route = candidate
        if best is None:
            unserved.add(rid)
            continue
        routes[best_vid] = best_route
        route_eval[best_vid] = best[1]
        served.add(rid)

    sol_eval = evaluate_solution(
        routes,
        requests_by_id,
        int(instance["vehicles"]["capacity"]),
        T_min,
        D_km,
    )
    solution = Solution(
        routes=routes,
        served_requests=sorted(list(sol_eval.served_requests)),
        unserved_requests=sorted(list(sol_eval.unserved_requests)),
        time_total_min=sol_eval.time_total_min,
        dist_total_km=sol_eval.dist_total_km,
        cost=None,
        feasible=sol_eval.feasible,
        meta={
            "strategy": "baseline_greedy_insertion",
        },
    )

    metrics = {
        "served": len(sol_eval.served_requests),
        "unserved": len(sol_eval.unserved_requests),
        "wait_mean_min": sol_eval.wait_mean_min,
        "ride_time_extra_mean_min": sol_eval.ride_time_extra_mean_min,
        "time_total_min": sol_eval.time_total_min,
        "dist_total_km": sol_eval.dist_total_km,
    }
    return solution, metrics


def _order_requests_by_slack(
    requests_by_id: dict[int, Request],
    T_min: np.ndarray,
    idx_of_xy,
) -> list[int]:
    """
    Order requests by slack time, the time difference between the latest
    possible arrival and the earliest possible departure.
    """
    order: list[tuple[float, int]] = []
    for rid, r in requests_by_id.items():
        p_idx = idx_of_xy(r.pickup_xy_km)
        d_idx = idx_of_xy(r.dropoff_xy_km)
        direct = float(T_min[p_idx, d_idx])
        slack = (r.l_max - r.e_min) - direct
        order.append((slack, rid))
    order.sort(key=lambda x: (x[0], x[1]))
    return [rid for _, rid in order]


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
