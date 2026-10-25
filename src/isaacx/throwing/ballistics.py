"""Ballistic release planning and a point-mass flight simulator.

World frame: z up, metres, seconds. The planner ignores drag on purpose; the gap between it and
`simulate(..., drag=k)` is what the learned residual in a later slice corrects.
"""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

GRAVITY = 9.81
AIR_DENSITY = 1.2

Vec = NDArray[np.float64]


def as_point(value: ArrayLike) -> Vec:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (3,):
        raise ValueError(f"expected a 3-vector, got shape {point.shape}")
    return point


def release_speed(
    release: ArrayLike, target: ArrayLike, angle: float, g: float = GRAVITY
) -> float | None:
    """Drag-free release speed that passes through `target` when launched at `angle` radians.

    Returns None when the target is unreachable at this angle: straight above or below the
    release point, or at or above the line of the launch direction (d * tan(angle) <= h).
    """
    if not 0.0 < angle < math.pi / 2:
        raise ValueError("angle must be strictly between 0 and pi/2 radians")
    p0, p1 = as_point(release), as_point(target)
    d = float(np.hypot(*(p1[:2] - p0[:2])))
    h = float(p1[2] - p0[2])
    rise = d * math.tan(angle) - h
    if d == 0.0 or rise <= 0.0:
        return None
    # z(d) = d tan(a) - g d^2 / (2 v^2 cos^2(a)) = h, solved for v.
    return math.sqrt(g * d * d / (2.0 * math.cos(angle) ** 2 * rise))


def release_velocity(
    release: ArrayLike, target: ArrayLike, angle: float, g: float = GRAVITY
) -> Vec | None:
    """Release velocity vector toward `target`, or None when it is unreachable."""
    speed = release_speed(release, target, angle, g)
    if speed is None:
        return None
    p0, p1 = as_point(release), as_point(target)
    offset = p1[:2] - p0[:2]
    horizontal = speed * math.cos(angle) * offset / np.hypot(*offset)
    return np.array([horizontal[0], horizontal[1], speed * math.sin(angle)])


def drag_coefficient(
    mass: float, radius: float, cd: float = 0.5, air_density: float = AIR_DENSITY
) -> float:
    """Quadratic drag constant k in a = -k |v| v, for a sphere: k = rho * Cd * pi r^2 / (2 m)."""
    return air_density * cd * math.pi * radius * radius / (2.0 * mass)


def simulate(
    release: ArrayLike,
    velocity: ArrayLike,
    drag: float = 0.0,
    g: float = GRAVITY,
    dt: float = 1e-3,
    t_max: float = 5.0,
) -> Vec:
    """Integrate point-mass flight with RK4. Returns positions, shape (steps + 1, 3)."""
    gravity = np.array([0.0, 0.0, -g])

    def accel(v: Vec) -> Vec:
        return gravity - drag * float(np.linalg.norm(v)) * v

    p, v = as_point(release), as_point(velocity)
    positions = [p]
    for _ in range(int(round(t_max / dt))):
        k1v, k1p = accel(v), v
        k2v, k2p = accel(v + 0.5 * dt * k1v), v + 0.5 * dt * k1v
        k3v, k3p = accel(v + 0.5 * dt * k2v), v + 0.5 * dt * k2v
        k4v, k4p = accel(v + dt * k3v), v + dt * k3v
        p = p + dt / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
        v = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        positions.append(p)
    return np.array(positions)


def descending_crossing(trajectory: ArrayLike, z: float) -> Vec | None:
    """First point where the trajectory passes down through the plane at height `z`."""
    path = np.asarray(trajectory, dtype=np.float64)
    for a, b in zip(path[:-1], path[1:], strict=True):
        if a[2] >= z > b[2]:
            t = (a[2] - z) / (a[2] - b[2])
            return a + t * (b - a)
    return None


def lands_in_bin(
    release: ArrayLike,
    velocity: ArrayLike,
    bin_center: ArrayLike,
    rim_radius: float,
    drag: float = 0.0,
) -> bool:
    """True when the throw comes down through the bin's rim plane inside its radius.

    `bin_center` is the centre of the rim opening, so its z is the rim height.
    """
    centre = as_point(bin_center)
    hit = descending_crossing(simulate(release, velocity, drag=drag), float(centre[2]))
    if hit is None:
        return False
    return float(np.hypot(*(hit[:2] - centre[:2]))) <= rim_radius
