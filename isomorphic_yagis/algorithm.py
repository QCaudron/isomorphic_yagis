"""Algorithm pieces for differential evolution of antenna configurations."""

import json
from time import time

import numpy as np
import ray
from tqdm import trange

from isomorphic_yagis.nec import evaluate_antenna
from isomorphic_yagis.utils import (
    BANDS,
    PARAMETER_LIMITS,
    clip_antenna_to_limits,
)

rng = np.random.default_rng()


@ray.remote
def evaluate_antenna_with_ray(antenna: dict[str, float]) -> float:
    """
    Evaluate an antenna's fitness using NEC, using Ray to parallelise the process.

    Parameters
    ----------
    antenna : dict[str, float]
        A dictionary of antenna parameters.

    Returns
    -------
    float
        The antenna's fitness.
    """
    return evaluate_antenna(antenna)  # type: ignore


def generate_valid_element_lengths(
    element: str, bands: list[str] = BANDS, limits: dict[str, tuple[float, float]] | None = None
) -> dict[str, float]:
    """
    Generate a set of element lengths within parameter limits, and are in order of length by band.

    Parameters
    ----------
    element : str
        The element to generate lengths for, in {"driven", "reflector"}.
    bands : list[str], optional
        A list of bands to generate element lengths for.
        Defaults to ["80", "40", "20", "15", "10"].
    limits : dict[str, tuple[float, float]], optional
        A dictionary of parameter limits to use.

    Returns
    -------
    dict[str, float]
        A mapping between bands and element lengths.
    """
    if limits is None:
        limits = PARAMETER_LIMITS

    lower_limits = [limits[f"{element}_length_{band}"][0] for band in bands]
    upper_limits = [limits[f"{element}_length_{band}"][1] for band in bands]

    lengths = rng.uniform(lower_limits, upper_limits)
    while not (np.diff(lengths) <= 0).all():
        lengths = rng.uniform(lower_limits, upper_limits)

    return {band: length for band, length in zip(bands, lengths, strict=True)}


def initialise(
    n_antennas: int,
    bands: list[str] = BANDS,
    limits: dict[str, tuple[float, float]] | None = None,
    override_values: dict[str, str] | None = None,
) -> list[dict[str, float]]:
    """
    Create an initial, random population of antennas for the specified bands.

    Parameters
    ----------
    n_antennas : int
        The number of antennas to create.
    bands : list[str], optional
        A list of bands to create antennas for.
        Defaults to ["80", "40", "20", "15", "10"].
    limits : dict[str, tuple[float, float]]
        A dictionary of parameter limits to use.
    override_values : dict[str, str] | None, optional
        A dictionary of antenna parameters to override with other values.

    Returns
    -------
    list[dict[str, float]]
        A list of antenna parameter dictionaries.
    """
    if limits is None:
        limits = PARAMETER_LIMITS

    initial_pop = []
    while len(initial_pop) < n_antennas:
        # Band-specific parameters -- the driven elements and reflector(s)
        driven_lengths = generate_valid_element_lengths("driven", bands=bands, limits=limits)
        reflector_lengths = generate_valid_element_lengths("reflector", bands=bands, limits=limits)

        band_params = {}
        for band in bands:
            band_params[f"driven_length_{band}"] = driven_lengths[band]
            band_params[f"reflector_length_{band}"] = reflector_lengths[band]

        # Parameters common to all antennas
        common_params = {
            "anchor_offset": rng.uniform(
                PARAMETER_LIMITS["anchor_offset"][0], PARAMETER_LIMITS["anchor_offset"][1]
            ),
            "pole_distance": rng.uniform(
                PARAMETER_LIMITS["pole_distance"][0], PARAMETER_LIMITS["pole_distance"][1]
            ),
            "height": rng.uniform(PARAMETER_LIMITS["height"][0], PARAMETER_LIMITS["height"][1]),
        }

        initial_pop.append(
            clip_antenna_to_limits(
                band_params | common_params, limits=limits, override_values=override_values
            )
        )

    return initial_pop


def evaluate_generation(antennas: list[dict[str, float]]) -> np.ndarray:
    """
    Return an array of fitness values for the given antennas.

    Parameters
    ----------
    antennas : list[dict[str, float]]
        A list of antenna parameter dictionaries.

    Returns
    -------
    np.ndarray
        An array of fitness values.
    """
    results = [evaluate_antenna_with_ray.remote(antenna) for antenna in antennas]
    return np.array(ray.get(results))


def mutate(
    antennas: list[dict[str, float]],
    crossover_prob: float = 0.7,
    differential_weight: float = 0.8,
    limits: dict[str, tuple[float, float]] | None = None,
    override_values: dict[str, str] | None = None,
) -> list[dict[str, float]]:
    """
    Mutate a population of antennas using differential evolution.

    Parameters
    ----------
    antennas : list[dict[str, float]]
        A list of antenna parameter dictionaries.
    crossover_prob : float, optional
        The probability of a given parameter crossing over, by default 0.7.
    differential_weight : float, optional
        The mutation strength, by default 0.8.
    limits : dict[str, tuple[float, float]] | None, optional
        A dictionary of parameter limits to use, by default None. If None,
        defaults to isomorphic_yagis.utils.PARAMETER_LIMITS.
    override_values : dict[str, str] | None, optional
        A dictionary of antenna parameters to override with other values.

    Returns
    -------
    list[dict[str, float]]
        A list of mutated antenna parameter dictionaries.
    """
    if limits is None:
        limits = PARAMETER_LIMITS

    mutated_antennas = []

    for ant_idx, antenna in enumerate(antennas):
        # Select three others to crossover with -- ensure they are all different
        others = [ant_idx, ant_idx, ant_idx]
        while ant_idx in others and len(set(others)) != 3:
            others = rng.choice(range(len(antennas)), 3)

        # Determine which parameters will crossover
        crossover = rng.uniform(0, 1, size=len(antenna)) < crossover_prob

        # Create the new antenna
        new_antenna = {}
        for dim_idx, (key, val) in enumerate(antenna.items()):
            if not crossover[dim_idx]:
                new_antenna[key] = val
            else:
                new_antenna[key] = antennas[others[0]][key] + differential_weight * (
                    antennas[others[1]][key] - antennas[others[2]][key]
                )

        mutated_antennas.append(
            clip_antenna_to_limits(new_antenna, limits=limits, override_values=override_values)
        )

    return mutated_antennas


def population_variance(antennas: list[dict[str, float]]) -> float:
    """
    Return the variance of the given antennas.

    This is a measure of difference across an antenna population. It is not necessarily
    comparable across experiments or different antenna types, but can be used to gauge
    the diversity of a population within a single experiment.

    Parameters
    ----------
    antennas : list[dict[str, float]]
        A list of antenna parameter dictionaries.

    Returns
    -------
    float
        The variance of the given antennas.
    """
    array = np.array([list(antenna.values()) for antenna in antennas])
    return array.var(0).mean()


def differential_evolution(
    n_population: int = 100,
    n_generations: int = 50,
    init: dict | None = None,
    crossover_prob: float = 0.7,
    differential_weight: float = 0.8,
    checkpoint: int = 50,
    bands: list[str] = BANDS,
    limits: dict[str, tuple[float, float]] | None = None,
    override_values: dict[str, str] | None = None,
) -> dict:
    """
    Perform differential evolution to optimise a population of antennas.

    Parameters
    ----------
    n_population : int, optional
        The size of the population, by default 100
    n_generations : int, optional
        The number of generations to evolve, by default 50
    init : dict | None, optional
        A dictionary containing antennas, fitness, history, accepted rates, variance rates, and
        generation times, from a previous experiment. This is used to continue evolving a previous
        experiment. If None, initialise a new population and start from scratch. By default None.
    crossover_prob : float, optional
        The probability of a given parameter crossing over, by default 0.7.
    differential_weight : float, optional
        The mutation strength, by default 0.8.
    checkpoint : int, optional
        An interval, in number of generations, to save a checkpoint at, by default 50.
    bands : list[str], optional
        A list of bands, used in initialising a new antenna population.
        By default, ["80", "40", "20", "15", "10"].
    limits : dict[str, tuple[float, float]] | None, optional
        A dictionary of parameter limits to use, by default None. If None,
        defaults to isomorphic_yagis.utils.PARAMETER_LIMITS.
    override_values : dict[str, str] | None, optional
        A dictionary of antenna parameters to override with other values.

    Returns
    -------
    dict
        A dictionary of results, containing antennas, fitness, history, accepted rates,
        variance rates, and generation times.
    """
    if init is not None:
        antennas = init["antennas"]
        fitness = init["fitness"]
        history = init["history"]
        accepted = init["accepted"]
        variance = init["variance"]
        generation_time = init["generation_time"]

    else:
        antennas = initialise(
            n_population, bands=bands, limits=limits, override_values=override_values
        )
        fitness = evaluate_generation(antennas)
        history = [fitness.mean()]
        accepted = []
        variance = []
        generation_time = []

    for i in trange(n_generations):
        tic = time()

        # Mutate antennas and reevaluate fitness
        mutated_antennas = mutate(
            antennas,
            crossover_prob=crossover_prob,
            differential_weight=differential_weight,
            limits=limits,
            override_values=override_values,
        )
        new_fitness = evaluate_generation(mutated_antennas)

        # Compute some metrics
        accepted.append((fitness < new_fitness).mean())
        variance.append(population_variance(antennas))

        # Select the best antennas
        antennas = [
            antennas[idx] if fitness[idx] > new_fitness[idx] else mutated_antennas[idx]
            for idx in range(len(antennas))
        ]
        fitness = np.array(
            [
                fitness[idx] if fitness[idx] > new_fitness[idx] else new_fitness[idx]
                for idx in range(len(antennas))
            ]
        )

        # More metrics
        history.append(np.mean(fitness))
        generation_time.append(time() - tic)

        # Write a checkpoint backup if required
        if (checkpoint > 0) and (((i % checkpoint) == 0) or (i == (n_generations - 1))):
            with open(f"checkpoint_{i+1}.json", "w") as f:
                json.dump(antennas, f)

    return {
        "antennas": antennas,
        "fitness": fitness,
        "history": history,
        "accepted": accepted,
        "variance": variance,
        "generation_time": generation_time,
    }
