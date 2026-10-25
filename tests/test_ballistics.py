import math

import numpy as np
import pytest

from isaacx.throwing.ballistics import (
    descending_crossing,
    drag_coefficient,
    lands_in_bin,
    release_speed,
    release_velocity,
    simulate,
)

RELEASE = np.array([0.0, 0.0, 0.6])
# YCB 056 tennis ball: about 57 g and 6.7 cm across.
TENNIS_DRAG = drag_coefficient(mass=0.057, radius=0.0335, cd=0.55)

TARGETS = [
    np.array([1.0, 0.0, 0.2]),
    np.array([0.8, 0.6, 0.0]),
    np.array([-1.5, 0.4, 0.3]),
    np.array([0.3, -2.0, 0.1]),
]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("angle_deg", [30.0, 45.0, 60.0])
def test_solved_release_reaches_target_without_drag(target: np.ndarray, angle_deg: float) -> None:
    velocity = release_velocity(RELEASE, target, math.radians(angle_deg))
    assert velocity is not None
    hit = descending_crossing(simulate(RELEASE, velocity), float(target[2]))
    assert hit is not None
    np.testing.assert_allclose(hit, target, atol=1e-3)


def test_velocity_matches_speed_and_angle() -> None:
    target = np.array([0.8, 0.6, 0.0])
    angle = math.radians(45.0)
    speed = release_speed(RELEASE, target, angle)
    velocity = release_velocity(RELEASE, target, angle)
    assert speed is not None and velocity is not None
    assert np.linalg.norm(velocity) == pytest.approx(speed)
    assert math.atan2(velocity[2], np.hypot(velocity[0], velocity[1])) == pytest.approx(angle)
    assert math.atan2(velocity[1], velocity[0]) == pytest.approx(math.atan2(0.6, 0.8))


def test_target_above_launch_line_is_unreachable() -> None:
    # At 45 degrees the launch line rises 1 m per metre, so 1.5 m up at 1 m out cannot be hit.
    assert release_speed(RELEASE, RELEASE + [1.0, 0.0, 1.5], math.radians(45.0)) is None
    assert release_velocity(RELEASE, RELEASE + [1.0, 0.0, 1.0], math.radians(45.0)) is None


def test_target_straight_below_is_unreachable() -> None:
    assert release_speed(RELEASE, RELEASE - [0.0, 0.0, 0.5], math.radians(45.0)) is None


@pytest.mark.parametrize("angle", [0.0, math.pi / 2, -0.1])
def test_angle_outside_open_range_is_rejected(angle: float) -> None:
    with pytest.raises(ValueError, match="angle"):
        release_speed(RELEASE, TARGETS[0], angle)


def test_point_must_be_three_vector() -> None:
    with pytest.raises(ValueError, match="3-vector"):
        release_speed([0.0, 0.0], TARGETS[0], 0.5)


def test_drag_makes_the_physics_estimate_fall_short() -> None:
    target = np.array([1.5, 0.0, 0.1])
    velocity = release_velocity(RELEASE, target, math.radians(45.0))
    assert velocity is not None
    hit = descending_crossing(simulate(RELEASE, velocity, drag=TENNIS_DRAG), float(target[2]))
    assert hit is not None
    shortfall = target[0] - hit[0]
    # Short, but by centimetres rather than metres: that gap is the residual's job.
    assert 0.01 < shortfall < 0.3


def test_lands_in_bin_scores_hits_and_misses() -> None:
    bin_center = np.array([1.2, 0.3, 0.15])
    velocity = release_velocity(RELEASE, bin_center, math.radians(45.0))
    assert velocity is not None
    assert lands_in_bin(RELEASE, velocity, bin_center, rim_radius=0.08)
    assert not lands_in_bin(RELEASE, 0.8 * velocity, bin_center, rim_radius=0.08)


def test_throw_that_never_comes_down_through_rim_misses() -> None:
    bin_center = np.array([1.0, 0.0, 2.0])
    assert not lands_in_bin(RELEASE, np.array([1.0, 0.0, 1.0]), bin_center, rim_radius=0.1)


def test_drag_coefficient_for_tennis_ball() -> None:
    # 1.2 * 0.55 * pi * 0.0335^2 / (2 * 0.057)
    assert TENNIS_DRAG == pytest.approx(0.0204, rel=0.01)
