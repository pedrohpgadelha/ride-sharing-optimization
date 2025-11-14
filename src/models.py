"""
Data models for the problem entities.
"""

from dataclasses import dataclass, field
from typing import Any, Literal


# Tuple of (x, y) coordinates in kilometers.
Coord = tuple[float, float]


@dataclass
class Request:
    """
    A request with a pickup and dropoff location with an earliest arrival time
    and a latest arrival time.
    """
    id: int
    pickup_xy_km: Coord
    dropoff_xy_km: Coord
    e_min: float
    l_max: float


@dataclass
class Vehicle:
    """
    A vehicle with a garage location (where it starts and ends its
    route) and a maximum number of passengers it can carry.
    """
    id: int
    garage_xy_km: Coord
    capacity: int


@dataclass
class Event:
    """
    Represents a node in the route. It can be a pickup, dropoff, or garage (end
    of route). It also has an arrival time, a wait time, and a load after the
    event. THe node index is the snapped index of the node in the grid graph.
    """
    kind: Literal["pickup", "dropoff", "garage"]
    request_id: int
    node_idx: int
    arrival_min: float | None = None
    wait_min: float = 0.0
    load_after: int | None = None


@dataclass
class Route:
    """
    A route is a sequence of events for a vehicle, always beggining and ending
    with a garage kind event.
    """
    vehicle_id: int
    events: list[Event] = field(default_factory=list)


@dataclass
class Solution:
    """
    A solution is a set of routes for a set of vehicles, with the served and
    unserved requests. It also has a total time, a total distance, a cost, and a
    feasibility flag. The meta field stores additional information about the
    solution, such as the lambda value, the strategy used, and the objective
    breakdown, etc.
    """
    routes: dict[int, Route]
    served_requests: list[int]
    unserved_requests: list[int]
    time_total_min: float
    dist_total_km: float
    cost: float | None = None
    feasible: bool = True
    meta: dict[str, Any] = field(default_factory=dict)
