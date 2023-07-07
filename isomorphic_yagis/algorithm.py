import json
from time import time

import numpy as np
import ray

from tqdm import trange

from isomorphic_yagis.nec import evaluate_antenna
from isomorphic_yagis.utils import (
    BANDS,
    BAND_CENTRAL_FREQUENCIES,
    PARAMETER_LIMITS,
    clip_antenna_to_limits,
)


@ray.remote
def evaluate_antenna_with_ray(antenna: dict[str, float]):
    return evaluate_antenna(antenna)


def generate_valid_element_lengths(element: str, bands: list[str] = BANDS):
    lower_limits = [PARAMETER_LIMITS[f"{element}_length_{band}"][0] for band in bands]
    upper_limits = [PARAMETER_LIMITS[f"{element}_length_{band}"][1] for band in bands]

    lengths = np.random.uniform(lower_limits, upper_limits)
    while not (np.diff(lengths) <= 0).all():
        lengths = np.random.uniform(lower_limits, upper_limits)

    return {band: length for band, length in zip(bands, lengths)}


def initialise(
    n_antennas: int, bands: list[str] = BANDS, common_reflector: bool = False
) -> list[dict[str, float]]:
    """
    Create an initial, random population of antennas for the specified bands.
    """

    initial_pop = []
    while len(initial_pop) < n_antennas:
        # Band-specific parameters -- the driven elements and reflector(s)
        driven_lengths = generate_valid_element_lengths("driven", bands)
        reflector_lengths = generate_valid_element_lengths("reflector", bands)

        band_params = {}
        for band in bands:
            band_params[f"driven_length_{band}"] = driven_lengths[band]
            band_params[f"reflector_length_{band}"] = reflector_lengths[band]

        # Parameters common to all antennas
        common_params = {
            "anchor_offset": np.random.uniform(
                PARAMETER_LIMITS["anchor_offset"][0], PARAMETER_LIMITS["anchor_offset"][1]
            ),
            "pole_distance": np.random.uniform(
                PARAMETER_LIMITS["pole_distance"][0], PARAMETER_LIMITS["pole_distance"][1]
            ),
            "height": np.random.uniform(
                PARAMETER_LIMITS["height"][0], PARAMETER_LIMITS["height"][1]
            ),
        }

        initial_pop.append(clip_antenna_to_limits(band_params | common_params))

    return initial_pop


def evaluate_generation(antennas: list) -> np.ndarray:
    """
    Return an array of fitness values for the given antennas.
    """
    results = [evaluate_antenna_with_ray.remote(antenna) for antenna in antennas]
    return np.array(ray.get(results))


def mutate(antennas, crossover_prob=0.7, differential_weight=0.8):
    mutated_antennas = []

    for ant_idx, antenna in enumerate(antennas):
        others = [ant_idx, ant_idx, ant_idx]
        while ant_idx in others and len(set(others)) != 3:
            others = np.random.choice(range(len(antennas)), 3)

        crossover = np.random.uniform(0, 1, size=len(antenna)) < crossover_prob

        new_antenna = {}
        for dim_idx, (key, val) in enumerate(antenna.items()):
            if not crossover[dim_idx]:
                new_antenna[key] = val
            else:
                new_antenna[key] = antennas[others[0]][key] + differential_weight * (
                    antennas[others[1]][key] - antennas[others[2]][key]
                )

        mutated_antennas.append(clip_antenna_to_limits(new_antenna))

    return mutated_antennas


def population_variance(antennas):
    array = np.array([list(antenna.values()) for antenna in antennas])
    return array.var(0).mean()


def differential_evolution(
    n_population: int = 100,
    n_generations: int = 50,
    init: dict | None = None,
    checkpoint: int = 50,
    bands=BANDS,
):
    if init is not None:
        antennas = init["antennas"]
        fitness = init["fitness"]
        history = init["history"]
        accepted = init["accepted"]
        variance = init["variance"]
        generation_time = init["generation_time"]

    else:
        antennas = initialise(n_population, bands=bands)
        fitness = evaluate_generation(antennas)
        history = [fitness.mean()]
        accepted = []
        variance = []
        generation_time = []

    fitness = evaluate_generation(antennas)

    for i in trange(n_generations):
        tic = time()

        # Mutate antennas and reevaluate fitness
        mutated_antennas = mutate(antennas)
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
