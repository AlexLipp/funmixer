#!/usr/bin/env python3

"""
How well do the two solvers recover known sources as the source range widens?

Setup
-----
A single synthetic network of 100 sample sites, with sub-basin areas varying by only +/-10%
and a uniform export rate, so the inversion is as well conditioned as it realistically gets.
Source concentrations are drawn log-uniformly spanning a controlled number of orders of
magnitude (0.5 up to 6), forward-modelled to observations, and then corrupted with relative
Gaussian noise. The error model is assumed known, so every solver that can use it is
handed `C_d = diag((f d)^2)` for the relative error f.

Why the range matters
---------------------
The two solvers differ in what they call misfit. The non-linear solver penalises *relative*
differences, which is scale-free. The linear solver penalises *absolute* differences, which is
not -- unweighted, it will spend all its effort on the largest observations and ignore the
small ones. That should not matter at 0.5 orders of magnitude and should matter enormously at
6.

Weighting by the inverse data covariance is the fix. With proportional errors, sigma_i = f*d_i,
so the whitened residual is (Mc - d)_i / (f d_i) = (1/f)((Mc)_i/d_i - 1): a *relative* misfit.
The weighted linear solver is therefore the linearisation of the non-linear one, and this
script is largely a test of how far that linearisation stretches.

Run from the repository root:
    python tests/linear_vs_nonlinear_benchmark.py
"""

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

# pyre-fixme[21]: Could not find module `matplotlib.pyplot`.
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import funmixer
from funmixer.linear_unmixer import LinearSampleNetworkUnmixer

N_SITES = 100
AREA_VARIATION = 0.10  # sub-basin areas drawn uniformly from 1 +/- this fraction
RELATIVE_ERROR = 0.20  # relative error on the observations, assumed known
GEOMETRIC_MEAN_CONC = 1000.0  # mg/kg
ORDERS_OF_MAGNITUDE = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
N_REPEATS = 5
LAMBDA_GRID = np.logspace(-6, 5, 23)


def build_network(n_sites: int, area_variation: float, seed: int) -> nx.DiGraph:
    """A random tree of sample sites with near-uniform sub-basin areas, flowing to node 0."""
    rng = np.random.default_rng(seed)
    undirected = nx.random_labeled_tree(n_sites, seed=seed)
    # Orient every edge towards the root, so edges point downstream.
    network = nx.DiGraph((child, parent) for parent, child in nx.bfs_edges(undirected, 0))
    for upstream, downstream in network.edges:
        network[upstream][downstream]["length"] = 1.0
    for node in network.nodes:
        network.nodes[node]["data"] = funmixer.SampleNode(
            name=node,
            area=float(rng.uniform(1 - area_variation, 1 + area_variation)),
            downstream_node=funmixer.nx_get_downstream_node(network, node),
            x=-1,
            y=-1,
            total_upstream_area=0,
            label=0,
            upstream_nodes=[],
            distance_downstream=1.0,
        )
    return network


def draw_sources(network: nx.DiGraph, orders: float, seed: int) -> funmixer.ElementData:
    """Log-uniform source concentrations spanning `orders` decades about a fixed geometric mean."""
    rng = np.random.default_rng(seed)
    half = orders / 2.0
    exponents = rng.uniform(-half, half, size=network.number_of_nodes())
    return {
        node: float(GEOMETRIC_MEAN_CONC * 10**exponent)
        for node, exponent in zip(network.nodes, exponents)
    }


@dataclass
class Score:
    """How well a set of predictions matches the truth, measured in log space."""

    median_log_error: float  # median |log10(pred/true)|, i.e. typical factor-of error
    rms_log_error: float  # RMS of the same, so bad sites are not hidden by good ones
    interior_median_log_error: float  # median over non-leaf sites only
    fraction_within_2x: float
    n_nonpositive: int
    chosen_lambda: float = 0.0

    @classmethod
    def compare(
        cls,
        predicted: funmixer.ElementData,
        truth: funmixer.ElementData,
        interior: Optional[np.ndarray] = None,
    ) -> "Score":
        keys = list(truth)
        true_values = np.array([truth[k] for k in keys])
        pred_values = np.array([predicted[k] for k in keys])
        n_nonpositive = int(np.sum(pred_values <= 0))
        # Clamped sites come back as exactly zero, where a log ratio is undefined. Floor them
        # far below any plausible source so they register as a large, finite error rather
        # than being silently dropped from the average.
        floor = 1e-6 * float(np.exp(np.mean(np.log(true_values))))
        log_error = np.abs(np.log10(np.maximum(pred_values, floor) / true_values))
        # A leaf site is recovered exactly (c_i = d_i), so its error is just the data error
        # whatever the solver does. Reporting the interior separately stops those sites from
        # masking the differences between methods.
        interior_errors = log_error if interior is None else log_error[interior]
        return cls(
            median_log_error=float(np.median(log_error)),
            rms_log_error=float(np.sqrt(np.mean(log_error**2))),
            interior_median_log_error=float(
                np.median(interior_errors) if interior_errors.size else np.nan
            ),
            fraction_within_2x=float(np.mean(log_error < np.log10(2.0))),
            n_nonpositive=n_nonpositive,
        )


def best_over_lambda(
    solve_for_lambda: Callable[[float], Optional[funmixer.ElementData]],
    truth: funmixer.ElementData,
    lambdas: np.ndarray,
    interior: np.ndarray,
) -> Score:
    """
    Score a solver at its *oracle-best* regularization strength.

    Choosing lambda against the known truth is of course cheating, but it is cheating equally
    for every solver, and it isolates what we actually want to measure: the best each method
    can possibly do, rather than how well some particular lambda-selection heuristic works.
    """
    best: Optional[Score] = None
    for lam in lambdas:
        predictions = solve_for_lambda(float(lam))
        if predictions is None:
            continue
        score = Score.compare(predictions, truth, interior)
        score.chosen_lambda = float(lam)
        if best is None or score.rms_log_error < best.rms_log_error:
            best = score
    assert best is not None, "every lambda failed to solve"
    return best


def run_trial(orders: float, seed: int) -> Dict[str, Score]:
    """Generate one synthetic dataset and invert it every way we know how."""
    network = build_network(N_SITES, AREA_VARIATION, seed)
    truth = draw_sources(network, orders, seed)
    clean = funmixer.forward_model(network, truth)

    rng = np.random.default_rng(seed + 10_000)
    observed = {
        node: value * max(1.0 + RELATIVE_ERROR * rng.normal(), 1e-3)
        for node, value in clean.items()
    }
    # The assumed-known error model.
    sigmas = {node: RELATIVE_ERROR * value for node, value in observed.items()}
    interior = np.array([network.in_degree(node) > 0 for node in truth], dtype=bool)

    scores: Dict[str, Score] = {}

    linear_unreg = LinearSampleNetworkUnmixer(network, use_regularization=False)
    scores["linear, lambda=0"] = Score.compare(
        linear_unreg.solve(observed, data_covariance=sigmas).upstream_preds, truth, interior
    )

    linear = LinearSampleNetworkUnmixer(network, use_regularization=True)
    scores["linear, weighted"] = best_over_lambda(
        lambda lam: linear.solve(
            observed, regularization_strength=lam, data_covariance=sigmas, weighted=True
        ).upstream_preds,
        truth,
        LAMBDA_GRID,
        interior,
    )
    scores["linear, unweighted"] = best_over_lambda(
        lambda lam: linear.solve(
            observed, regularization_strength=lam, data_covariance=sigmas, weighted=False
        ).upstream_preds,
        truth,
        LAMBDA_GRID,
        interior,
    )

    nonlinear_unreg = funmixer.SampleNetworkUnmixer(network, use_regularization=False)
    scores["nonlinear, lambda=0"] = Score.compare(
        nonlinear_unreg.solve(observed).upstream_preds, truth, interior
    )

    nonlinear = funmixer.SampleNetworkUnmixer(network, use_regularization=True)

    def solve_nonlinear(lam: float) -> Optional[funmixer.ElementData]:
        try:
            return nonlinear.solve(observed, regularization_strength=lam).upstream_preds
        except Exception:
            return None

    scores["nonlinear, regularized"] = best_over_lambda(
        solve_nonlinear, truth, LAMBDA_GRID, interior
    )
    return scores


METHODS: List[str] = [
    "linear, lambda=0",
    "linear, unweighted",
    "linear, weighted",
    "nonlinear, lambda=0",
    "nonlinear, regularized",
]


def main() -> None:
    print(
        f"{N_SITES} sites, areas +/-{AREA_VARIATION:.0%}, uniform export rate, "
        f"{RELATIVE_ERROR:.0%} relative error on observations ({N_REPEATS} repeats).\n"
        "Regularized methods are shown at their oracle-best lambda.\n"
        "Score is the median factor-of error: 10^median|log10(pred/true)|.\n"
    )

    results: Dict[str, Dict[float, List[Score]]] = {
        m: {o: [] for o in ORDERS_OF_MAGNITUDE} for m in METHODS
    }
    for orders in ORDERS_OF_MAGNITUDE:
        for repeat in range(N_REPEATS):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trial = run_trial(orders, seed=repeat)
            for method, score in trial.items():
                results[method][orders].append(score)
        print(f"  ...done {orders} orders of magnitude")

    header = f"\n{'method':<24}" + "".join(f"{o:>12.1f}" for o in ORDERS_OF_MAGNITUDE)
    for label, extract, fmt in [
        ("median factor-of error (all sites)", lambda s: 10**s.median_log_error, "{:>12.2f}"),
        (
            "median factor-of error (interior sites only)",
            lambda s: 10**s.interior_median_log_error,
            "{:>12.2f}",
        ),
        ("RMS factor-of error", lambda s: 10**s.rms_log_error, "{:>12.2f}"),
        ("fraction within 2x", lambda s: s.fraction_within_2x, "{:>11.0%} "),
        ("sites hitting zero", lambda s: s.n_nonpositive, "{:>12.1f}"),
        ("oracle-best lambda", lambda s: s.chosen_lambda, "{:>12.2g}"),
    ]:
        print(f"\n=== {label} (source range, orders of magnitude) ===")
        print(header.strip("\n"))
        for method in METHODS:
            row = f"{method:<24}"
            for orders in ORDERS_OF_MAGNITUDE:
                row += fmt.format(float(np.mean([extract(s) for s in results[method][orders]])))
            print(row)

    _, axis = plt.subplots(figsize=(8, 5))
    for method in METHODS:
        axis.plot(
            ORDERS_OF_MAGNITUDE,
            [
                float(np.mean([10**s.rms_log_error for s in results[method][o]]))
                for o in ORDERS_OF_MAGNITUDE
            ],
            marker="o",
            label=method,
        )
    axis.axhline(1.0, color="grey", linestyle=":", label="perfect recovery")
    axis.set_yscale("log")
    axis.set_xlabel("Source range (orders of magnitude)")
    axis.set_ylabel("RMS factor-of error in recovered concentration")
    axis.set_title(
        f"Source recovery vs. source range\n({N_SITES} sites, {RELATIVE_ERROR:.0%} data error)"
    )
    axis.legend()
    plt.tight_layout()
    plt.savefig("linear_vs_nonlinear_benchmark.png", dpi=150)
    print("\nSaved plot to linear_vs_nonlinear_benchmark.png")


if __name__ == "__main__":
    main()
