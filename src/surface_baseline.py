import csv
from pathlib import Path

import stim
import pymatching
import sinter
import numpy as np
import scipy
import matplotlib.pyplot as plt

"""Surface-code memory simulation under circuit-level depolarizing noise.

This script builds a set of surface-code memory experiments, runs them with a
noise-aware decoder, collects the resulting logical error statistics, and plots
those results for both X and Z bases. The main parameters are grouped at the top
so they can be adjusted quickly before execution.
"""

# User-adjustable parameters.
# Change these values at the top before running the script.
distances = [3, 5, 7]  # Surface-code distances to simulate.
noise_values = np.logspace(-4, -2, 15)  # Physical noise values to sweep.
num_workers = 1  # Use 1 worker on Windows to avoid multiprocessing shutdown issues.
max_shots = 100000  # Maximum number of shots per task.
max_errors = 500  # Stop a task once this many errors have been seen.
figure_size = (12, 5)  # Size of the side-by-side plot.

# Output locations for saved results and figures.
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data"
figures_dir = project_root / "figures"
data_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)
data_file = data_dir / "surface_code_results.csv"
figure_file = figures_dir / "surface_code_error_rates.png"


def extract_logical_error_rates(collected_stats):
    """Extract logical error rates and organise them by distance and noise level.

    Parameters
    ----------
    collected_stats : list
        The sinter results returned by a collection run.

    Returns
    -------
    dict
        A dictionary mapping each distance to another dictionary of
        {physical_noise: logical_error_rate} entries.
    """
    rates_by_distance = {}
    for stats in collected_stats:
        d = stats.json_metadata["d"]
        p = stats.json_metadata["p"]
        rate = stats.errors / stats.shots
        rates_by_distance.setdefault(d, {})[p] = rate
    return rates_by_distance


def fit_scaling_curve(noise_values, rates, d):
    """Fit the error-rate data to the expected scaling form p_L ≈ a p^{(d+1)/2}.

    Parameters
    ----------
    noise_values : array-like
        Physical error probabilities used in the simulation.
    rates : array-like
        Corresponding logical error rates observed from the data.
    d : int
        Surface-code distance.

    Returns
    -------
    tuple
        The fitted coefficient a, along with the transformed x and y arrays used
        in the fit.
    """
    exponent = (d + 1) // 2
    x = np.asarray(noise_values) ** exponent
    y = np.asarray(rates)
    valid = np.isfinite(y)

    if np.count_nonzero(valid) < 2:
        return np.nan, np.array([]), np.array([])

    x_valid = x[valid]
    y_valid = y[valid]
    a = np.sum(x_valid * y_valid) / np.sum(x_valid * x_valid)
    return a, x_valid, y_valid


def collect_stats(tasks):
    """Run sinter collection using a single worker to avoid multiprocessing issues on Windows."""
    return sinter.collect(
        num_workers=1,
        tasks=tasks,
        decoders=["pymatching"],
        max_shots=max_shots,
        max_errors=max_errors,
    )


if __name__ == "__main__":
    # Build the list of simulation tasks for the X-basis memory experiment.
    # Each task uses a rotated surface-code circuit with a specific distance and
    # phenomenological noise level.
    tasks_x = [ sinter.Task(
        circuit=stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            rounds = d,
            distance = d,
            after_clifford_depolarization=noise,
            after_reset_flip_probability=noise,
            before_measure_flip_probability=noise,
            before_round_data_depolarization=noise,
        ), 
        json_metadata = {'d':d, 'p':noise}
        )
        for d in distances
        for noise in noise_values
    ]
    # Build the corresponding list of simulation tasks for the Z-basis experiment.
    tasks_z = [ sinter.Task(
        circuit=stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds = d,
            distance = d,
            after_clifford_depolarization=noise,
            after_reset_flip_probability=noise,
            before_measure_flip_probability=noise,
            before_round_data_depolarization=noise,
        ), 
        json_metadata = {'d':d, 'p':noise}
        )
        for d in distances
        for noise in noise_values
    ]

    # Run the X-basis simulations using the configured decoder and sampling limits.
    collected_stats_x = collect_stats(tasks_x)

    # Run the Z-basis simulations with the same settings for comparison.
    collected_stats_z = collect_stats(tasks_z)

    # Create a side-by-side figure for the X-basis and Z-basis results.
    fig, axes = plt.subplots(1, 2, figsize=figure_size, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    results = []

    # Plot the collected results and overplot the fitted scaling curves.
    for ax, (stats_name, stats) in zip(axes, [("X basis", collected_stats_x), ("Z basis", collected_stats_z)]):
        sinter.plot_error_rate(
            ax=ax,
            stats=stats,
            x_func=lambda stats: stats.json_metadata['p'],
            group_func=lambda stats: stats.json_metadata['d'],
        )

        # Convert the collected statistics into a dictionary that can be indexed by
        # distance and physical noise level.
        rate_by_distance = extract_logical_error_rates(stats)
        for d in distances:
            rates = np.array([rate_by_distance[d].get(p, np.nan) for p in noise_values], dtype=float)
            valid = np.isfinite(rates)
            if np.count_nonzero(valid) > 0:
                a, _, _ = fit_scaling_curve(noise_values[valid], rates[valid], d)
                exponent = (d + 1) // 2
                fit_curve = a * noise_values[valid] ** exponent
                ax.plot(noise_values[valid], fit_curve, linestyle='--', label=f'Fit d={d}')
                print(f"Estimated a for d={d} ({stats_name}): {a}")
                for p, rate in zip(noise_values[valid], rates[valid]):
                    results.append(
                        {
                            "basis": stats_name,
                            "distance": d,
                            "physical_error_probability": p,
                            "logical_error_rate": rate,
                            "fit_coefficient": a,
                        }
                    )

        ax.loglog()
        ax.set_title(f"Surface Code Error Rates ({stats_name})")
        ax.set_xlabel("Physical Error Rate")
        ax.set_ylabel("Logical Error Rate per Shot")
        ax.grid(which='major')
        ax.grid(which='minor')
        ax.legend()

    fig.suptitle("Surface Code Error Rates (Phenomenological Noise)")
    fig.tight_layout()
    fig.set_dpi(120)

    with data_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "basis",
                "distance",
                "physical_error_probability",
                "logical_error_rate",
                "fit_coefficient",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    fig.savefig(figure_file, dpi=300, bbox_inches="tight")