"""
Utilities for working with zones.

A zone is a circular area around a center point with a given radius where
the speed is different than the speed outside the zone.
"""

from math import hypot
from typing import Any


def build_zones(zones_cfg: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalize the YAML extracted zones config.
    """
    zones = []
    for z in zones_cfg:
        cx, cy = float(z["center"][0]), float(z["center"][1])
        r = float(z["radius_km"])
        zones.append(
            {
                "id": str(z["id"]),
                "center": (cx, cy),
                "radius_km": r,
            }
        )
    return zones


def in_zone(x_km: float, y_km: float, zone: dict[str, Any]) -> bool:
    """
    Check if a point is inside a zone.
    """
    cx, cy = zone["center"]
    r = zone["radius_km"]
    return hypot(x_km - cx, y_km - cy) <= r


def in_any_zone(x_km: float, y_km: float, zones: list[dict[str, Any]]) -> bool:
    """
    Check if a point is inside any of the zones.
    """
    for z in zones:
        if in_zone(x_km, y_km, z):
            return True
    return False


def point_speed_kmph(
    x_km: float,
    y_km: float,
    speeds_cfg: dict[str, float],
    zones: list[dict[str, Any]]
) -> float:
    """
    Get the speed of a point in the grid based on if it is inside any of the
    zones.
    """
    inside = in_any_zone(x_km, y_km, zones)
    if inside:
        return float(speeds_cfg["inside_zone_kmph"])
    else:
        return float(speeds_cfg["outside_zone_kmph"])


def edge_speed_kmph(
    x1_km: float,
    y1_km: float,
    x2_km: float,
    y2_km: float,
    speeds_cfg: dict[str, float],
    zones: list[dict[str, Any]],
) -> float:
    """
    Get the speed of an edge in the grid based on the average speed of its two
    vertices.
    """
    speed1 = point_speed_kmph(x1_km, y1_km, speeds_cfg, zones)
    speed2 = point_speed_kmph(x2_km, y2_km, speeds_cfg, zones)
    return (speed1 + speed2) / 2.0
