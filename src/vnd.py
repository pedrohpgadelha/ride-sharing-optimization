"""
Variable neighborhood descent (VND) heuristic.
"""

import time
from copy import deepcopy
from typing import Any
from os import PathLike
import numpy as np

from src.feas import evaluate_route, evaluate_solution
from src.models import Event, Route, Solution, Request
from src.moves import (
    RelocateIntra,
    SwapIntra,
    RelocateInter,
    generate_relocate_intra,
    generate_relocate_inter,
    generate_swap_intra,
)
from src.objective import (
    compute_objective,
    load_baseline_means,
    means_for_scenario,
)
from src.io import read_yaml
from src.metrics import summarize_solution


def vnd_improve(
    instance_yaml: str | PathLike[str],
    od_npz: str | PathLike[str],
    baseline_means_json: str | PathLike[str],
    routes: dict[int, Route],
    lambda_value: float,
    penalty_M: float,
    neighborhoods_order: list[str] | None = None,
    use_inter_route: bool = True,
    time_limit_sec: float | None = None,
) -> tuple[Solution, dict[str, float]]:
    """
    Run the VND heuristic for an instance starting from given routes.
    Uses first-improving strategy within each neighborhood.
    """
    instance = read_yaml(instance_yaml)
    cap = int(instance["vehicles"]["capacity"])
    reqs = _build_requests_by_id(instance)

    od = np.load(od_npz)
    T_min = od["T_min"]
    D_km = od["D_km"]

    baseline_means = load_baseline_means(baseline_means_json)
    mean_time, mean_dist = means_for_scenario(
        baseline_means,
        instance.get("name", "scenario"),
    )

    nb_order = neighborhoods_order or [
        "relocate_intra",
        "swap_intra",
        "relocate_inter",
    ]
    deadline = (time.time() + time_limit_sec) if time_limit_sec else None

    cur_routes = deepcopy(routes)

    improved = True
    while improved:
        improved = False
        if deadline and time.time() >= deadline:
            break
        for nb in nb_order:
            if deadline and time.time() >= deadline:
                break
            if nb == "relocate_intra":
                ok = _try_relocate_intra(
                    cur_routes,
                    reqs,
                    cap,
                    T_min,
                    D_km,
                    lambda_value,
                    mean_time,
                    mean_dist,
                    deadline,
                )
            elif nb == "swap_intra":
                ok = _try_swap_intra(
                    cur_routes,
                    reqs,
                    cap,
                    T_min,
                    D_km,
                    lambda_value,
                    mean_time,
                    mean_dist,
                    deadline,
                )
            elif nb == "relocate_inter" and use_inter_route:
                ok = _try_relocate_inter(
                    cur_routes,
                    reqs,
                    cap,
                    T_min,
                    D_km,
                    lambda_value,
                    mean_time,
                    mean_dist,
                    deadline,
                )
            else:
                ok = False

            if ok:
                improved = True
                break

    final_eval = evaluate_solution(cur_routes, reqs, cap, T_min, D_km)
    final_cost, breakdown = compute_objective(
        final_eval.time_total_min,
        final_eval.dist_total_km,
        lambda_value,
        mean_time,
        mean_dist,
        penalty_M,
        n_unserved=len(final_eval.unserved_requests),
    )
    sol = Solution(
        routes=cur_routes,
        served_requests=sorted(list(final_eval.served_requests)),
        unserved_requests=sorted(list(final_eval.unserved_requests)),
        time_total_min=final_eval.time_total_min,
        dist_total_km=final_eval.dist_total_km,
        cost=final_cost,
        feasible=final_eval.feasible,
        meta={
            "lambda": lambda_value,
            "strategy": "grasp+vnd",
            "objective": breakdown
        },
    )
    metrics = summarize_solution(instance_yaml, T_min, D_km, sol.routes)
    metrics["cost"] = sol.cost if sol.cost is not None else float("nan")
    return sol, metrics


def _try_relocate_intra(
    routes: dict[int, Route],
    requests_by_id: dict[int, Any],
    cap: int,
    T_min: np.ndarray,
    D_km: np.ndarray,
    lambda_value: float,
    mean_time: float,
    mean_dist: float,
    deadline: float | None = None,
) -> bool:
    """
    Try to relocate a request within the same route with first-improving.
    """
    for vid, route in routes.items():
        if deadline and time.time() >= deadline:
            return False

        cur = evaluate_route(route, requests_by_id, cap, T_min, D_km)
        for mv in generate_relocate_intra(route):
            if deadline and time.time() >= deadline:
                return False
            cand = _apply_relocate_intra(route, mv)
            rr = evaluate_route(cand, requests_by_id, cap, T_min, D_km)
            if not rr.feasible:
                continue
            dt = rr.time_total_min - cur.time_total_min
            dd = rr.dist_total_km - cur.dist_total_km
            delta_norm = (
                lambda_value * (dt / max(mean_time, 1e-6)) +
                (1.0 - lambda_value) * (dd / max(mean_dist, 1e-6))
            )
            if float(delta_norm) < -1e-9:
                routes[vid] = cand
                return True
    return False


def _try_swap_intra(
    routes: dict[int, Route],
    requests_by_id: dict[int, Any],
    cap: int,
    T_min: np.ndarray,
    D_km: np.ndarray,
    lambda_value: float,
    mean_time: float,
    mean_dist: float,
    deadline: float | None = None,
) -> bool:
    """
    Try to swap two requests within the same route with first-improving.
    """
    for vid, route in routes.items():
        if deadline and time.time() >= deadline:
            return False

        cur = evaluate_route(route, requests_by_id, cap, T_min, D_km)
        for mv in generate_swap_intra(route):
            if deadline and time.time() >= deadline:
                return False
            cand = _apply_swap_intra(route, mv)
            rr = evaluate_route(cand, requests_by_id, cap, T_min, D_km)
            if not rr.feasible:
                continue
            dt = rr.time_total_min - cur.time_total_min
            dd = rr.dist_total_km - cur.dist_total_km
            delta_norm = (
                lambda_value * (dt / max(mean_time, 1e-6)) +
                (1.0 - lambda_value) * (dd / max(mean_dist, 1e-6))
            )
            if float(delta_norm) < -1e-9:
                routes[vid] = cand
                return True
    return False


def _try_relocate_inter(
    routes: dict[int, Route],
    requests_by_id: dict[int, Any],
    cap: int,
    T_min: np.ndarray,
    D_km: np.ndarray,
    lambda_value: float,
    mean_time: float,
    mean_dist: float,
    deadline: float | None = None,
) -> bool:
    """
    Try to relocate a request between two routes with first-improving.
    """
    vids = list(routes.keys())

    for i in range(len(vids)):
        if deadline and time.time() >= deadline:
            return False
        for j in range(len(vids)):
            if deadline and time.time() >= deadline:
                return False
            if i == j:
                continue

            va, vb = vids[i], vids[j]
            ra, rb = routes[va], routes[vb]
            cur_a = evaluate_route(ra, requests_by_id, cap, T_min, D_km)
            cur_b = evaluate_route(rb, requests_by_id, cap, T_min, D_km)

            for mv in generate_relocate_inter(ra, rb):
                if deadline and time.time() >= deadline:
                    return False
                cand_a, cand_b = _apply_relocate_inter(ra, rb, mv)
                rra = evaluate_route(cand_a, requests_by_id, cap, T_min, D_km)
                rrb = evaluate_route(cand_b, requests_by_id, cap, T_min, D_km)
                if not (rra.feasible and rrb.feasible):
                    continue
                dt = (
                    (rra.time_total_min + rrb.time_total_min) -
                    (cur_a.time_total_min + cur_b.time_total_min)
                )
                dd = (
                    (rra.dist_total_km + rrb.dist_total_km) -
                    (cur_a.dist_total_km + cur_b.dist_total_km)
                )
                delta_norm = (
                    lambda_value * (dt / max(mean_time, 1e-6)) +
                    (1.0 - lambda_value) * (dd / max(mean_dist, 1e-6))
                )
                if float(delta_norm) < -1e-9:
                    routes[va] = cand_a
                    routes[vb] = cand_b
                    return True
    return False


def _apply_relocate_intra(route: Route, mv: RelocateIntra) -> Route:
    """
    Apply a relocate intra move to a route.
    """
    evs = list(route.events)
    pick = evs.pop(mv.from_pick_idx)
    drop = evs.pop(
        mv.from_drop_idx - 1
        if mv.from_drop_idx > mv.from_pick_idx
        else mv.from_drop_idx
    )
    evs.insert(mv.to_pick_pos, pick)
    evs.insert(mv.to_drop_pos, drop)
    return Route(vehicle_id=route.vehicle_id, events=evs)


def _apply_swap_intra(route: Route, mv: SwapIntra) -> Route:
    """
    Apply a swap intra move to a route.
    """
    evs = list(route.events)
    pa, da = _find_pair_positions(evs, mv.request_a)
    pb, db = _find_pair_positions(evs, mv.request_b)
    if pa is None or pb is None or da is None or db is None:
        return route
    pa, da = min(pa, da), max(pa, da)
    pb, db = min(pb, db), max(pb, db)

    block_a = [evs[pa], evs[da]]
    block_b = [evs[pb], evs[db]]

    remove_idxs = sorted([pa, da, pb, db], reverse=True)
    for idx in remove_idxs:
        evs.pop(idx)

    pos_a = pa - sum(1 for r_idx in remove_idxs if r_idx < pa)
    pos_b = pb - sum(1 for r_idx in remove_idxs if r_idx < pb)

    evs[pos_a:pos_a] = block_b
    pos_b_new = pos_b + (2 if pos_b >= pos_a else 0)
    evs[pos_b_new:pos_b_new] = block_a

    return Route(vehicle_id=route.vehicle_id, events=evs)


def _apply_relocate_inter(
    route_from: Route,
    route_to: Route,
    mv: RelocateInter,
) -> tuple[Route, Route]:
    """
    Apply a relocate inter move to two routes.
    """
    evs_from = [e for e in route_from.events if e.request_id != mv.request_id]
    p_idx, d_idx = _find_pair_indices(route_from.events, mv.request_id)
    if p_idx is None or d_idx is None:
        return route_from, route_to
    pick = route_from.events[p_idx]
    drop = route_from.events[d_idx]
    evs_to = list(route_to.events)
    evs_to.insert(
        mv.to_pick_pos,
        Event(kind="pickup", request_id=pick.request_id, node_idx=pick.node_idx)
    )
    evs_to.insert(
        mv.to_drop_pos,
        Event(
            kind="dropoff",
            request_id=drop.request_id,
            node_idx=drop.node_idx,
        ),
    )
    return (
        Route(vehicle_id=route_from.vehicle_id, events=evs_from),
        Route(vehicle_id=route_to.vehicle_id, events=evs_to),
    )


def _find_pair_positions(
    evs: list[Event],
    request_id: int
) -> tuple[int | None, int | None]:
    """
    Find the pickup and dropoff positions of a request in a list of events.
    """
    p = d = None
    for i, e in enumerate(evs):
        if e.request_id == request_id:
            if e.kind == "pickup" and p is None:
                p = i
            elif e.kind == "dropoff" and d is None:
                d = i
    return p, d


def _find_pair_indices(
    evs: list[Event],
    request_id: int
) -> tuple[int | None, int | None]:
    """
    Find the pickup and dropoff indices of a request in a list of events.
    """
    p = d = None
    for i, e in enumerate(evs):
        if e.request_id == request_id and e.kind == "pickup":
            p = i
            break
    for i, e in enumerate(evs):
        if e.request_id == request_id and e.kind == "dropoff":
            d = i
            break
    return p, d


def _build_requests_by_id(instance: dict) -> dict[int, Any]:
    """
    Build a dictionary of requests by their id from an instance.
    """
    out: dict[int, Request] = {}
    for r in instance["requests"]["list"]:
        out[int(r["id"])] = Request(
            id=int(r["id"]),
            pickup_xy_km=tuple(r["pickup_xy_km"]),
            dropoff_xy_km=tuple(r["dropoff_xy_km"]),
            e_min=float(r["e_min"]),
            l_max=float(r["l_max"]),
        )
    return out
