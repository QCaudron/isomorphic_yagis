import numpy as np
import pytest

from isomorphic_yagis.algorithm import (
    generate_valid_element_lengths,
    initialise,
    mutate,
    population_variance,
)
from isomorphic_yagis.utils import PARAMETER_LIMITS, BANDS


@pytest.mark.parametrize(
    "element, bands, limits",
    [
        ("driven", BANDS, PARAMETER_LIMITS),
        ("driven", ["80", "40"], PARAMETER_LIMITS),
        ("reflector", BANDS, PARAMETER_LIMITS),
    ],
)
def test_generate_valid_element_lengths(element, bands, limits):
    element_lengths = generate_valid_element_lengths(element, bands, limits)

    assert len(element_lengths) == len(bands)
    assert np.diff(list(element_lengths.values())).all() >= 0

    for band, element_length in element_lengths.items():
        assert (
            limits[f"{element}_length_{band}"][0]
            <= element_length
            <= limits[f"{element}_length_{band}"][1]
        )


@pytest.mark.parametrize(
    "n_antennas, bands, limits, override_values",
    [
        (10, BANDS, PARAMETER_LIMITS, None),
        (20, BANDS, PARAMETER_LIMITS, None),
        (10, ["80", "40"], PARAMETER_LIMITS, None),
        (
            10,
            BANDS,
            PARAMETER_LIMITS | {"driven_length_40": (0, 1)},
            {"driven_length_40": "driven_length_20"},
        ),
    ],
)
def test_initialise(n_antennas, bands, limits, override_values):
    antennas = initialise(n_antennas, bands, limits, override_values)
    if override_values is None:
        override_values = {}

    assert len(antennas) == n_antennas

    for common_param in ["pole_distance", "anchor_offset", "height"]:
        assert common_param in antennas[0].keys()

    for band in bands:
        assert f"driven_length_{band}" in antennas[0].keys()

    for antenna in antennas:
        for key, val in antenna.items():
            assert limits[key][0] <= val <= limits[key][1]

        for override_key, override_val in override_values.items():
            assert antennas[0][override_key] == antennas[0][override_val]


@pytest.mark.parametrize(
    "n_antennas, bands, limits, override_values, crossover_prob, diff_weight",
    [
        (10, BANDS, PARAMETER_LIMITS, None, 0.7, 0.8),
        (20, ["20", "15"], PARAMETER_LIMITS, None, 0.7, 0.8),
        (10, BANDS, PARAMETER_LIMITS, {"reflector_length_20": "reflector_length_15"}, 0.7, 0.8),
    ],
)
def test_mutate(n_antennas, bands, limits, override_values, crossover_prob, diff_weight):
    antennas = initialise(
        n_antennas=n_antennas, bands=bands, limits=limits, override_values=override_values
    )
    mutated_antennas = mutate(
        antennas=antennas,
        crossover_prob=crossover_prob,
        differential_weight=diff_weight,
        limits=limits,
        override_values=override_values,
    )

    assert len(antennas) == len(mutated_antennas)

    for antenna in mutated_antennas:
        for band in bands:
            assert (
                limits[f"driven_length_{band}"][0]
                <= antenna[f"driven_length_{band}"]
                <= limits[f"driven_length_{band}"][1]
            )


@pytest.mark.parametrize(
    "antennas, expected",
    [
        ([{"driven_length_20": 1}, {"driven_length_20": 2}], 0.25),
        ([{"driven_length_20": 1}, {"driven_length_20": 1}], 0),
    ],
)
def test_population_variance(antennas, expected):
    assert population_variance(antennas) == expected
