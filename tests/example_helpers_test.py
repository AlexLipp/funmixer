"""Tests for the library behaviour the example scripts rely on."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest

import funmixer

FLOWDIRS = "data/synthetic_topo_d8.asc"
SAMPLES = "data/synthetic_samples.csv"


@pytest.fixture
def network_and_labels() -> tuple[nx.DiGraph, np.ndarray]:
    return funmixer.get_sample_graph(FLOWDIRS, SAMPLES)


def test_sample_graph_from_dataframe_matches_filename(
    network_and_labels: tuple[nx.DiGraph, np.ndarray],
) -> None:
    from_file, labels_from_file = network_and_labels
    from_frame, labels_from_frame = funmixer.get_sample_graph(FLOWDIRS, pd.read_csv(SAMPLES))
    assert set(from_frame.edges) == set(from_file.edges)
    assert np.array_equal(labels_from_frame, labels_from_file)


def test_sample_graph_accepts_string_coordinates(
    network_and_labels: tuple[nx.DiGraph, np.ndarray],
) -> None:
    # Below-detection-limit tables are read with dtype=str, which makes the coordinates strings.
    as_strings = pd.read_csv(SAMPLES, dtype=str)
    network, _ = funmixer.get_sample_graph(FLOWDIRS, as_strings)
    assert set(network.edges) == set(network_and_labels[0].edges)


def test_sample_graph_rejects_non_numeric_coordinates() -> None:
    samples = pd.read_csv(SAMPLES, dtype=str)
    samples.iloc[0, 1] = "not a number"
    with pytest.raises(ValueError, match="must be numeric"):
        funmixer.get_sample_graph(FLOWDIRS, samples)


def test_concentration_map_is_nan_outside_sampled_basins(
    network_and_labels: tuple[nx.DiGraph, np.ndarray],
) -> None:
    network, labels = network_and_labels
    areas = funmixer.get_unique_upstream_areas(network, labels)
    values = {name: float(i + 1) for i, name in enumerate(areas)}
    conc_map = funmixer.get_upstream_concentration_map(areas, values)

    sampled = np.any(np.stack(list(areas.values())), axis=0)
    assert np.all(np.isnan(conc_map[~sampled]))
    for name, mask in areas.items():
        assert np.all(conc_map[mask] == values[name])


def test_regularizer_sweep_returns_one_row_per_strength(
    network_and_labels: tuple[nx.DiGraph, np.ndarray],
) -> None:
    network, _ = network_and_labels
    upstream = {name: 10.0 * (i + 1) for i, name in enumerate(network.nodes)}
    downstream = funmixer.forward_model(network, upstream)
    problem = funmixer.SampleNetworkUnmixer(network)

    sweep = funmixer.plot_sweep_of_regularizer_strength(problem, downstream, -3, -1, 3)

    assert list(sweep.columns) == ["regularization_strength", "roughness", "misfit"]
    assert np.allclose(sweep["regularization_strength"], [1e-3, 1e-2, 1e-1])


def test_regularizer_sweep_marks_chosen_strength(
    network_and_labels: tuple[nx.DiGraph, np.ndarray],
) -> None:
    network, _ = network_and_labels
    upstream = {name: 10.0 * (i + 1) for i, name in enumerate(network.nodes)}
    downstream = funmixer.forward_model(network, upstream)
    problem = funmixer.SampleNetworkUnmixer(network)

    funmixer.plot_sweep_of_regularizer_strength(
        problem, downstream, -3, -1, 3, chosen_strength=1e-2
    )

    labels = [handle.get_label() for handle in plt.gca().get_legend().legend_handles]
    assert labels == ["chosen strength"]
