#!/usr/bin/env python3
"""
Unmix one element with Monte Carlo uncertainties: how well constrained is each Mg source?

The data
--------
River sediment (the fine-grained, <150 micron fraction) was collected at 63 sites on five
rivers draining the Cairngorms mountains of NE Scotland: the Spey, Dee, Don, Deveron and Tay.
Each sample was analysed for ~50 elements, reported in `data/sample_data.csv` in mg/kg. The
sediment at a site is a mixture of material eroded from everywhere upstream of it, so the
downstream samples blur together the geochemistry of very different rocks (granites, mafic
intrusions, sandstones and metasediments). `data/d8.tif` is a 50 m flow-direction grid that
says which way water flows out of each cell. Both are in British National Grid (EPSG:27700)
coordinates, so x and y are eastings and northings in metres. See Barnes & Lipp (2024),
Water Resources Research, doi:10.1029/2023WR036159, section 4.

This example
------------
Every measurement carries analytical error, and that error propagates into the recovered
sources: a sub-basin that contributes little to any downstream sample is poorly constrained.
Here the observations are perturbed with a 10% relative error and re-unmixed many times; the
spread of the answers is the uncertainty. The unmixing itself is the same as in
`unmix_single_element.py`, and so is the regularization strength.

Run from the repository root:

    python examples/unmix_with_uncertainties.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import funmixer

FLOWDIRS_FILENAME = "data/d8.tif"
ELEMENT = "Mg"
# Display only: cells draining more than this area are drawn as rivers on the maps.
RIVER_THRESHOLD_KM2 = 100.0
# Display only: shared by both uncertainty examples so their maps compare directly.
UNCERTAINTY_MAX = 10_000.0
SAMPLE_DATA_FILENAME = "data/sample_data.csv"
# Method choice: matches `unmix_single_element.py`, which shows how it was picked.
REGULARIZATION_STRENGTH = 1e-3
# Assumption: a typical analytical error for sediment geochemistry.
RELATIVE_ERROR_PERCENT = 10.0
# Enough repeats for a stable standard deviation; increase for smoother maps.
NUM_REPEATS = 50
# Fixed so results are reproducible; the uncertainty should not depend on its value.
RANDOM_SEED = 8128


def load_basemap(flowdirs_filename: str, samples: pd.DataFrame) -> dict:
    """
    Gather what every map needs: the raster extent, the river cells and the sample sites.

    Parameters
    ----------
    flowdirs_filename : str
        D8 flow-direction raster.
    samples : pd.DataFrame
        Sample table; columns 2 and 3 are x and y.

    Returns
    -------
    dict
        Keys `extent` (left, right, bottom, top), `rivers` (boolean raster), `x` and `y`.
    """
    accumulator = funmixer.D8Accumulator(flowdirs_filename)
    cell_area_km2 = accumulator.dx * abs(accumulator.dy) / 1e6
    drainage_area_km2 = accumulator.accumulate() * cell_area_km2
    return {
        "extent": accumulator.extent,
        "rivers": drainage_area_km2 > RIVER_THRESHOLD_KM2,
        "x": pd.to_numeric(samples.iloc[:, 1]),
        "y": pd.to_numeric(samples.iloc[:, 2]),
    }


def plot_map(ax: plt.Axes, values: np.ndarray, basemap: dict, label: str, **imshow_kwargs) -> None:
    """
    Draw a sub-basin map in British National Grid coordinates, zoomed to the sampled area.

    Parameters
    ----------
    ax : plt.Axes
        Axes to draw on.
    values : np.ndarray
        Raster from `get_upstream_concentration_map`; NaN outside sampled sub-basins.
    basemap : dict
        Output of `load_basemap`.
    label : str
        Colour-bar label.
    **imshow_kwargs
        Passed to `imshow`, e.g. `cmap` and `norm`.
    """
    left, right, _, top = basemap["extent"]
    image = ax.imshow(values, extent=basemap["extent"], **imshow_kwargs)
    # Rivers are one cell wide, too thin to survive imshow's downsampling, so outline them.
    rivers = basemap["rivers"].astype(float)
    ax.contour(
        rivers,
        levels=[0.5],
        extent=basemap["extent"],
        origin="upper",
        colors="deepskyblue",
        linewidths=0.6,
    )
    ax.scatter(
        basemap["x"], basemap["y"], s=14, c="red", edgecolors="black", linewidths=0.5, zorder=3
    )

    rows, cols = np.nonzero(np.isfinite(values))
    cell = (right - left) / values.shape[1]
    ax.set_xlim(left + cols.min() * cell, left + (cols.max() + 1) * cell)
    ax.set_ylim(top - (rows.max() + 1) * cell, top - rows.min() * cell)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    # Arrow-ended colour bar wherever values fall outside the colour limits and are clipped.
    below = np.nanmin(values) < image.norm.vmin
    above = np.nanmax(values) > image.norm.vmax
    extend = {(False, False): "neither", (True, False): "min", (False, True): "max"}.get(
        (below, above), "both"
    )
    plt.colorbar(image, ax=ax, label=label, shrink=0.8, extend=extend)


def plot_fit(ax: plt.Axes, observed: dict, predicted: dict) -> None:
    """
    Plot predicted against observed downstream concentrations on log axes with a 1:1 line.

    Parameters
    ----------
    ax : plt.Axes
        Axes to draw on.
    observed, predicted : dict
        Concentration at each sample site, keyed by site name.
    """
    names = list(observed)
    obs = np.array([observed[name] for name in names])
    pred = np.array([predicted[name] for name in names])
    # Axis limits rounded out to whole decades.
    low = 10 ** np.floor(np.log10(min(obs.min(), pred.min())))
    high = 10 ** np.ceil(np.log10(max(obs.max(), pred.max())))
    ax.plot([low, high], [low, high], color="grey", linewidth=1)
    ax.scatter(obs, pred, s=15, c="black")
    ax.set(xscale="log", yscale="log", xlim=(low, high), ylim=(low, high), aspect="equal")
    ax.set_xlabel(f"Observed {ELEMENT} (mg/kg)")
    ax.set_ylabel(f"Predicted {ELEMENT} (mg/kg)")


def main() -> None:
    samples = pd.read_csv(SAMPLE_DATA_FILENAME)
    sample_network, labels = funmixer.get_sample_graph(FLOWDIRS_FILENAME, samples)

    # The sample network: each arrow points from a site to the next site downstream.
    plt.figure(figsize=(15, 10))
    funmixer.plot_network(sample_network)
    observed = funmixer.get_element_obs(ELEMENT, samples)

    problem = funmixer.SampleNetworkUnmixer(sample_network=sample_network)
    # solve_montecarlo draws from numpy's global random state, so seed that (NPY002).
    np.random.seed(RANDOM_SEED)  # noqa: NPY002
    downstream_mc, upstream_mc = problem.solve_montecarlo(
        observed,
        relative_error=RELATIVE_ERROR_PERCENT,
        num_repeats=NUM_REPEATS,
        regularization_strength=REGULARIZATION_STRENGTH,
    )
    upstream_mean = {site: np.mean(values) for site, values in upstream_mc.items()}
    upstream_std = {site: np.std(values) for site, values in upstream_mc.items()}
    downstream_mean = {site: np.mean(values) for site, values in downstream_mc.items()}

    areas = funmixer.get_unique_upstream_areas(sample_network, labels)
    basemap = load_basemap(FLOWDIRS_FILENAME, samples)
    source_map = funmixer.get_upstream_concentration_map(areas, upstream_mean)
    uncertainty_map = funmixer.get_upstream_concentration_map(areas, upstream_std)
    obs_values = list(observed.values())
    # Colour limits are the range of the observations, so every example's map shares one scale.
    norm = plt.matplotlib.colors.LogNorm(vmin=min(obs_values), vmax=max(obs_values))
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), width_ratios=[1.4, 1.4, 1])
    plot_map(axes[0], source_map, basemap, f"Source {ELEMENT} (mg/kg)", cmap="cividis", norm=norm)
    plot_map(
        axes[1],
        uncertainty_map,
        basemap,
        f"1-sigma uncertainty in {ELEMENT} (mg/kg)",
        cmap="magma",
        vmin=0,
        vmax=UNCERTAINTY_MAX,
    )
    plot_fit(axes[2], observed, downstream_mean)
    for ax, letter in zip(axes, "abc", strict=True):
        ax.text(-0.15, 1.02, letter, transform=ax.transAxes, fontweight="bold")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
