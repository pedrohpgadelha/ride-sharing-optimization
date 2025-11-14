"""
Data models and functions to generate and apply moves to a route set of events.
"""

from dataclasses import dataclass
from typing import Iterable

from src.models import Route


@dataclass(frozen=True)
class RelocateIntra:
    """
    A move to relocate a request within the same route.
    """
    request_id: int
    from_pick_idx: int
    from_drop_idx: int
    to_pick_pos: int
    to_drop_pos: int


@dataclass(frozen=True)
class SwapIntra:
    """
    A move to swap two requests within the same route.
    """
    request_a: int
    request_b: int


@dataclass(frozen=True)
class RelocateInter:
    """
    A move to relocate a request from one route to another.
    """
    request_id: int
    from_vehicle_id: int
    to_vehicle_id: int
    to_pick_pos: int
    to_drop_pos: int


def _request_positions(route: Route) -> dict[int, tuple[int, int]]:
    """
    Get the positions of the pickup and dropoff events for each request in the
    route.
    """
    pos: dict[int, list[int]] = {}
    for idx, ev in enumerate(route.events):
        if ev.kind in ("pickup", "dropoff"):
            pos.setdefault(ev.request_id, []).append(idx)
    out: dict[int, tuple[int, int]] = {}
    for rid, lst in pos.items():
        if len(lst) == 2:
            i, j = sorted(lst)
            out[rid] = (i, j)
    return out


def generate_relocate_intra(route: Route) -> Iterable[RelocateIntra]:
    """
    Generate all possible relocate intra moves for each request of a route.
    """
    pos = _request_positions(route)
    n = len(route.events)
    for rid, (i_pick, i_drop) in pos.items():
        for p in range(1, n):  # keep index 0 for garage/start
            for d in range(p + 1, n + 1):
                if p == i_pick and d == i_drop:
                    continue
                yield RelocateIntra(rid, i_pick, i_drop, p, d)


def generate_swap_intra(route: Route) -> Iterable[SwapIntra]:
    """
    Generate all possible swap intra moves for each request of a route.
    """
    pos = list(_request_positions(route).keys())
    m = len(pos)
    for a in range(m):
        for b in range(a + 1, m):
            yield SwapIntra(pos[a], pos[b])


def generate_relocate_inter(
    route_from: Route,
    route_to: Route
) -> Iterable[RelocateInter]:
    """
    Generate all possible relocate inter moves for each request of one route to
    another.
    """
    pos_from = _request_positions(route_from)
    n_to = len(route_to.events)
    for rid in pos_from.keys():
        for p in range(1, n_to):
            for d in range(p + 1, n_to + 1):
                yield RelocateInter(
                    request_id=rid,
                    from_vehicle_id=route_from.vehicle_id,
                    to_vehicle_id=route_to.vehicle_id,
                    to_pick_pos=p,
                    to_drop_pos=d,
                )
