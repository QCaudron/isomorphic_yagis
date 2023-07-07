import matplotlib.pyplot as plt
import numpy as np


BANDS = ["80", "40", "20", "15", "10"]


BAND_CENTRAL_FREQUENCIES = {
    "80": 3.6,
    "40": 7.1,
    "20": 14.1,
    "15": 21.2,
    "10": 28.5,
}


PARAMETER_LIMITS = {
    "driven_length_80": (0.6, 0.99),
    "reflector_length_80": (0.3, 0.9),
    "driven_length_40": (0.3, 0.8),
    "reflector_length_40": (0.1, 0.6),
    "driven_length_20": (0.1, 0.4),
    "reflector_length_20": (0.05, 0.4),
    "driven_length_15": (0.05, 0.4),
    "reflector_length_15": (0.05, 0.3),
    "driven_length_10": (0.05, 0.3),
    "reflector_length_10": (0.05, 0.2),
    "common_reflector_length": (0.05, 0.9),
    "anchor_offset": (6, 20),
    "pole_distance": (42, 42),
    "height": (10, 10),
}


def clip_antenna_to_limits(
    antenna: dict[str, float], limits: dict[str, tuple[float, float]] = PARAMETER_LIMITS
) -> dict[str, float]:
    """
    Clip an antenna's parameters to its reasonable limits.
    """
    for key in antenna.keys():
        antenna[key] = np.clip(antenna[key], limits[key][0], limits[key][1])
    return antenna


def plot_results(output: dict):
    antennas = output["antennas"]
    best_antenna = antennas[np.argmax(output["fitness"])]

    plt.figure(figsize=(14, 3))

    plt.subplot(151)
    plt.plot(output["history"])
    plt.title("Mean Fitness")

    plt.subplot(152)
    plt.hist(output["fitness"], bins=50)
    plt.title("Final Fitness")

    plt.subplot(153)
    plt.plot(output["accepted"])
    plt.title("Acceptance rate")

    plt.subplot(154)
    plt.plot(output["variance"])
    plt.title("Variance")

    plt.subplot(155)
    plt.plot(output["generation_time"])
    plt.title("Generation time")

    plt.tight_layout()

    return antennas, best_antenna
