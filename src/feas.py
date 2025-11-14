"""
Functions and data models to evaluate the feasibility of a route and a solution.
"""

from dataclasses import dataclass, field
import numpy as np

from src.models import Request, Route


@dataclass
class FeasRouteResult:
    """
    The result of evaluating the feasibility of a route.
    """
    feasible: bool
    time_total_min: float
    dist_total_km: float
    arrivals_min: list[float]
    departs_min: list[float]
    waits_min: list[float]
    loads: list[int]
    ride_time_extra_by_request: dict[int, float] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)


@dataclass
class FeasSolutionResult:
    """
    The result of evaluating the feasibility of a solution.
    """
    feasible: bool
    time_total_min: float
    dist_total_km: float
    served_requests: set[int]
    unserved_requests: set[int]
    wait_mean_min: float
    ride_time_extra_mean_min: float
    per_route: dict[int, FeasRouteResult]


def _delta_load(kind: str) -> int:
    if kind == "pickup":
        return 1
    if kind == "dropoff":
        return -1
    return 0


def evaluate_route(
    route: Route,
    requests_by_id: dict[int, Request],
    vehicle_capacity: int,
    T_min: np.ndarray,
    D_km: np.ndarray,
) -> FeasRouteResult:
    """
    Evaluate the feasibility of a route.
    """
    if not route.events:
        return FeasRouteResult(
            feasible=True,
            time_total_min=0.0,
            dist_total_km=0.0,
            arrivals_min=[],
            departs_min=[],
            waits_min=[],
            loads=[],
        )

    n = len(route.events)
    arrivals: list[float] = [0.0] * n
    departs: list[float] = [0.0] * n
    waits: list[float] = [0.0] * n
    loads: list[int] = [0] * n
    violations: list[str] = []

    t_cur = 0.0
    load = 0
    picked: set[int] = set()
    pickup_time_by_req: dict[int, float] = {}

    for idx, ev in enumerate(route.events):
        if idx == 0:
            arrivals[idx] = t_cur
            departs[idx] = t_cur
            loads[idx] = load
            continue

        prev = route.events[idx - 1]
        if ev.node_idx >= T_min.shape[0] or prev.node_idx >= T_min.shape[0]:
            violations.append("node_idx out of bounds")
            return FeasRouteResult(
                feasible=False,
                time_total_min=np.inf,
                dist_total_km=np.inf,
                arrivals_min=arrivals,
                departs_min=departs,
                waits_min=waits,
                loads=loads,
                violations=violations,
            )

        travel_min = float(T_min[prev.node_idx, ev.node_idx])
        dist_km = float(D_km[prev.node_idx, ev.node_idx])
        if not np.isfinite(travel_min) or not np.isfinite(dist_km):
            violations.append("disconnected nodes")
            return FeasRouteResult(
                feasible=False,
                time_total_min=np.inf,
                dist_total_km=np.inf,
                arrivals_min=arrivals,
                departs_min=departs,
                waits_min=waits,
                loads=loads,
                violations=violations,
            )

        t_cur = departs[idx - 1] + travel_min
        arrivals[idx] = t_cur

        if ev.kind == "pickup" or ev.kind == "dropoff":
            req = requests_by_id[ev.request_id]
            if ev.kind == "pickup":
                if t_cur < req.e_min:
                    wait = req.e_min - t_cur
                    waits[idx] = wait
                    t_cur += wait
                else:
                    waits[idx] = 0.0
                picked.add(req.id)
                pickup_time_by_req[req.id] = t_cur
                departs[idx] = t_cur
                load += _delta_load(ev.kind)
            else:
                if req.id not in picked:
                    violations.append(
                        f"precedence violation for request {req.id}"
                    )
                if t_cur > req.l_max + 1e-9:
                    violations.append(
                        f"time window violation at dropoff for request {req.id}"
                    )
                departs[idx] = t_cur
                load += _delta_load(ev.kind)
            loads[idx] = load
        else:
            waits[idx] = 0.0
            departs[idx] = t_cur
            loads[idx] = load

        if load < 0 or load > vehicle_capacity:
            violations.append("capacity violation")

    total_time = max(0.0, departs[-1] - arrivals[0])
    total_dist = 0.0
    for idx in range(1, n):
        a = route.events[idx - 1].node_idx
        b = route.events[idx].node_idx
        total_dist += float(D_km[a, b])

    ride_extra: dict[int, float] = {}
    seen_drop: set[int] = set()
    for idx, ev in enumerate(route.events):
        if ev.kind == "dropoff":
            rid = ev.request_id
            if rid in pickup_time_by_req:
                t_pick = pickup_time_by_req[rid]
                t_drop = arrivals[idx]
                direct = float(
                    T_min[_safe_idx(route, rid, "pickup"), ev.node_idx]
                )
                extra = (t_drop - t_pick) - direct
                ride_extra[rid] = max(0.0, extra)
                seen_drop.add(rid)

    feasible = len(violations) == 0
    return FeasRouteResult(
        feasible=feasible,
        time_total_min=total_time,
        dist_total_km=total_dist,
        arrivals_min=arrivals,
        departs_min=departs,
        waits_min=waits,
        loads=loads,
        ride_time_extra_by_request=ride_extra,
        violations=violations,
    )


def evaluate_solution(
    routes: dict[int, Route],
    requests_by_id: dict[int, Request],
    vehicle_capacity: int,
    T_min: np.ndarray,
    D_km: np.ndarray,
) -> FeasSolutionResult:
    """
    Evaluate the feasibility of a solution.
    """
    per_route: dict[int, FeasRouteResult] = {}
    time_sum = 0.0
    dist_sum = 0.0
    served: set[int] = set()
    waits_all: list[float] = []
    ride_extra_all: list[float] = []
    feasible_all = True

    for vid, route in routes.items():
        rr = evaluate_route(
            route,
            requests_by_id,
            vehicle_capacity,
            T_min,
            D_km,
        )
        per_route[vid] = rr
        time_sum += rr.time_total_min if np.isfinite(rr.time_total_min) else 0.0
        dist_sum += rr.dist_total_km if np.isfinite(rr.dist_total_km) else 0.0
        feasible_all = feasible_all and rr.feasible
        for idx, ev in enumerate(route.events):
            if ev.kind == "pickup":
                waits_all.append(rr.waits_min[idx])
        for rid, val in rr.ride_time_extra_by_request.items():
            ride_extra_all.append(val)
            served.add(rid)

    all_req_ids = set(requests_by_id.keys())
    unserved = all_req_ids - served
    wait_mean = float(np.mean(waits_all)) if waits_all else 0.0
    ride_extra_mean = float(np.mean(ride_extra_all)) if ride_extra_all else 0.0

    return FeasSolutionResult(
        feasible=feasible_all,
        time_total_min=time_sum,
        dist_total_km=dist_sum,
        served_requests=served,
        unserved_requests=unserved,
        wait_mean_min=wait_mean,
        ride_time_extra_mean_min=ride_extra_mean,
        per_route=per_route,
    )


def _safe_idx(route: Route, request_id: int, kind: str) -> int:
    """
    Get the index of the event with the given request id and kind.
    """
    for ev in route.events:
        if ev.request_id == request_id and ev.kind == kind:
            return ev.node_idx
    return route.events[0].node_idx
