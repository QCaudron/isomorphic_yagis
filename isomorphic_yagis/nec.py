"""Functions related to running NEC simulations and evaluating antennas."""

import os
from subprocess import PIPE, Popen
from uuid import uuid4

import numpy as np
from scipy.stats import hmean

from isomorphic_yagis.utils import BAND_CENTRAL_FREQUENCIES

RP_QUICK_CALC = "RP 0 10 1 1500 45.0 0.0 3.0 3.0 5.0E+03"
RP_FULL_CALC = "RP 0 91 181 1500 -90.0 0.0 2.0 2.0 5.0E+03"

CARD_DECK = """CM V-Yagi
CE
GW 1 51 0 -{driven_element_length} {height} 0 {driven_element_length} {height} 0.001
GW 2 51 -{anchor_offset} 0 {height} {reflector_dx} {reflector_dy} {height} 0.001
GW 3 51 -{anchor_offset} 0 {height} {reflector_dx} -{reflector_dy} {height} 0.001
GE -1
GN 2 0 0 0 13 0.005
EK
EX 0 1 26 0 1 0 0 0
FR 0 0 0 0 {frequency} 0
{rp}
XQ 0
EN"""


def find_impedance(output: str) -> complex:
    """
    Find the impedance at the feedpoint given the output from the simulation.

    Parameters
    ----------
    output : str
        The output from the NEC simulation.


    Returns
    -------
    complex
        The complex impedance at the feedpoint.
    """
    lines = output.splitlines()

    impedance_title_line = np.where(["IMPEDANCE (OHMS)" in line for line in lines])[0][0]
    impedance_data_line = impedance_title_line + 2
    impedance_data = lines[impedance_data_line].split()[6:8]

    return complex(float(impedance_data[0]), float(impedance_data[1]))


def find_gain(output: str) -> float:
    """
    From the first line in the output from stdout, grab the gain as a float.

    Parameters
    ----------
    output : str
        The output from the NEC simulation.


    Returns
    -------
    float
        The gain, in dBi.
    """
    line = output.splitlines()[0]
    return float(line.split(": ")[1])


def calculate_swr(impedance: complex, Z0: float = 50) -> float:
    """
    Compute the standing wave ratio for a coax impedance of 50 ohms.

    Parameters
    ----------
    impedance : complex
        The complex impedance of the antenna.
    Z0 : float, optional
        The characteristic impedance at the feedpoint, in ohms. By default, 50.

    Returns
    -------
    float
        The standing wave ratio.
    """
    # Compute the standing wave radio from the complex impedances
    gamma = (impedance - Z0) / (impedance + Z0)
    swr = (1 + np.abs(gamma)) / (1 - np.abs(gamma))
    return swr


def evaluate_antenna(antenna: dict[str, float], write: bool = False) -> float | dict[str, float]:
    """
    Evaluate the antenna by running the NEC simulation and computing the gain and SWR.

    Parameters
    ----------
    antenna : dict[str, float]
        The antenna parameters.
    write : bool, optional
        Whether to write the antenna files to disk for visualization. If False, files will
        be temporary, just for the simulation, before being deleted, and will only contain
        radiation pattern requirements to calculate SWR and gain; if True, files will be
        saved to "antenna_setup_{band}.nec" and will contain the full radiation pattern to
        allow a better visualization of the antenna, but this calculation will take longer.
        By default, False.

    Returns
    -------
    float
        The antenna's fitness -- the harmonic mean of its gain-to-SWR ratio across bands.
    """
    # Radiation pattern fixed strings for the quicker and full calculations

    # Unpack params common to all bands
    pole_distance = antenna["pole_distance"]
    height = antenna["height"]
    anchor_offset = antenna["anchor_offset"]

    # Unpack band-specific params
    bands = [key.split("_")[-1] if "driven" in key else None for key in antenna]
    bands = [band for band in bands if band is not None]
    frequencies = [BAND_CENTRAL_FREQUENCIES[band] for band in bands]
    driven_lengths = [antenna[f"driven_length_{band}"] for band in bands]
    reflector_lengths = [antenna[f"reflector_length_{band}"] for band in bands]

    # Throw away antennas whose band lengths are not in order
    if not all(np.diff(driven_lengths) <= 0) or not all(np.diff(reflector_lengths) <= 0):
        return 0

    # Compute the frequencies for each band
    results = []
    for band, frequency, driven_length, reflector_length in zip(
        bands, frequencies, driven_lengths, reflector_lengths, strict=True
    ):
        identifier = str(uuid4())
        contents = CARD_DECK.format(
            height=height,
            anchor_offset=anchor_offset,
            driven_element_length=pole_distance * driven_length / 2,
            reflector_dx=-anchor_offset * (1 - reflector_length),
            reflector_dy=pole_distance * reflector_length / 2,
            frequency=frequency,
            rp=RP_QUICK_CALC if not write else RP_FULL_CALC,
        )

        filename = f"antenna_setup_{band}_{identifier}.nec"
        with open(filename, "w") as f:
            f.write(contents)

        if write:
            with open(f"antenna_setup_{band}.nec", "w") as f:
                f.write(contents)

        command = [
            "./nec2++",
            "-i",
            f"antenna_setup_{band}_{identifier}.nec",
            "-s",
            "-g",
        ]

        process = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output_bytes, _ = process.communicate()
        os.remove(filename)
        os.remove(filename.replace(".nec", ".out"))

        # Attempt to decode the output; if the simulation fails anywhere along the line,
        # return a fitness of 0 for this entire antenna
        try:
            output: str = output_bytes.decode()
            swr = calculate_swr(find_impedance(output))
            gain = find_gain(output)
        except Exception:
            return 0

        # Any weirdnesses, return 0 too
        if np.isnan(swr) or np.isnan(gain) or np.isinf(swr) or np.isinf(gain):
            return 0

        # If we're writing the files, return the gain and SWR individually for each band
        if write:
            results.append({"gain": gain, "swr": swr})
        # Otherwise, calculate the fitness for this band and add it to the list
        else:
            results.append(max(gain / swr, 0))

    if write:
        return {band: result for band, result in zip(bands, results, strict=True)}

    return hmean(results)
