"""Plotting helpers for simulation results."""

import matplotlib.pyplot as plt


def plot_error_rates(x, y, xlabel="x", ylabel="y", title="Error rates"):
    fig, ax = plt.subplots()
    ax.plot(x, y, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    return fig
