import json
from time import time

import matplotlib.pyplot as plt
import numpy as np
import ray
from scipy.stats import hmean
from tqdm import trange

from isomorphic_yagis.nec import evaluate_antenna
from isomorphic_yagis.utils import BANDS, PARAMETER_LIMITS, clip_antenna_to_limits


@ray.remote
def evaluate_antenna_with_ray(antenna: dict[str, float]):
    return evaluate_antenna(antenna)


def initialise(
    n_antennas: int, bands: list[str] = BANDS, common_reflector: bool = False
) -> list[dict[str, float]]:
    """
    Create an initial, random population of antennas for the specified bands.
    """

    initial_pop = []
    while len(initial_pop) < n_antennas:
        # Parameters common to all antennas
        antenna = {
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

        # Band-specific parameters -- the driven elements and reflector(s)
        for band in bands:
            antenna[f"driven_length_{band}"] = np.random.uniform(
                PARAMETER_LIMITS[f"driven_length_{band}"][0],
                PARAMETER_LIMITS[f"driven_length_{band}"][1],
            )
            if not common_reflector:
                antenna[f"reflector_length_{band}"] = np.random.uniform(
                    PARAMETER_LIMITS[f"reflector_length_{band}"][0],
                    PARAMETER_LIMITS[f"reflector_length_{band}"][1],
                )
            else:
                antenna["common_reflector_length"] = np.random.uniform(
                    PARAMETER_LIMITS["common_reflector_length"][0],
                    PARAMETER_LIMITS["common_reflector_length"][1],
                )

        initial_pop.append(clip_antenna_to_limits(antenna))

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
        if (checkpoint > 0) and ((i % checkpoint == 0) or (i == n_generations - 1)):
            with open(f"checkpoint_{i}.json", "w") as f:
                json.dump(antennas, f)

    return {
        "antennas": antennas,
        "fitness": fitness,
        "history": history,
        "accepted": accepted,
        "variance": variance,
        "generation_time": generation_time,
    }


def plot_results(output: dict):
    antennas = output["antennas"]
    fitness = output["fitness"]
    history = output["history"]
    accepted = output["accepted"]
    variance = output["variance"]
    generation_time = output["generation_time"]
    best_antenna = antennas[np.argmax(fitness)]

    plt.figure(figsize=(14, 3))

    plt.subplot(151)
    plt.plot(history)
    plt.title("Mean Fitness")

    plt.subplot(152)
    plt.hist(fitness, bins=50)
    plt.title("Final Fitness")

    plt.subplot(153)
    plt.plot(accepted)
    plt.title("Acceptance rate")

    plt.subplot(154)
    plt.plot(variance)
    plt.title("Variance")

    plt.subplot(155)
    plt.plot(generation_time)
    plt.title("Generation time")

    plt.tight_layout()

    return antennas, fitness, history, accepted, variance, generation_time, best_antenna
