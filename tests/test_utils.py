import pytest

from isomorphic_yagis.utils import clip_antenna_to_limits, PARAMETER_LIMITS


ANTENNA = {
    "driven_length_80": 0.9,
    "reflector_length_80": 0.9,
    "driven_length_40": 0.5,
    "reflector_length_40": 0.5,
    "driven_length_20": 0.25,
    "reflector_length_20": 0.25,
}


@pytest.mark.parametrize(
    "antenna, limits, override_values, expected",
    [
        (ANTENNA, PARAMETER_LIMITS, None, ANTENNA),
        (ANTENNA, PARAMETER_LIMITS, {}, ANTENNA),
        (
            ANTENNA,
            PARAMETER_LIMITS,
            {"reflector_length_20": "reflector_length_40"},
            ANTENNA | {"reflector_length_20": 0.5},
        ),
        (
            ANTENNA,
            PARAMETER_LIMITS | {"reflector_length_40": (0.001, 0.001)},
            {"reflector_length_20": "reflector_length_40"},
            ANTENNA | {"reflector_length_20": 0.001, "reflector_length_40": 0.001},
        ),
    ],
)
def test_clip_antenna_to_limits(antenna, limits, override_values, expected):
    assert (
        clip_antenna_to_limits(antenna, limits=limits, override_values=override_values) == expected
    )
