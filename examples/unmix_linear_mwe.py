#!/usr/bin/env python3

"""
Minimum working example for the *linear* unmixing solver.

This is the counterpart to `unmix_mwe.py`. Where that script penalises relative (log-ratio)
misfit, this one penalises absolute misfit, which makes the forward model an exactly invertible
matrix and gives closed-form uncertainties: no Monte Carlo required.

The script loads a sample network, inspects the diagnostics that say whether the linear
approach is appropriate for this network at all, sweeps the regularization strength to find the
elbow of the L-curve, then solves and maps both the recovered concentrations and their
propagated standard deviations.

Run from the repository root:
    python examples/unmix_linear_mwe.py
"""

import logging
import warnings

# pyre-fixme[21]: Could not find module `matplotlib.pyplot`.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import funmixer

logging.getLogger().addHandler(logging.StreamHandler())

ELEMENT = "Mg"
RELATIVE_ERROR_PERCENT = 10.0
REGULARIZATION_STRENGTH = 1.0


def main() -> None:
    sample_network, labels = funmixer.get_sample_graph(
        flowdirs_filename="data/d8.asc",
        sample_data_filename="data/sample_data.csv",
    )

    obs_data = pd.read_csv("data/sample_data.csv").drop(columns=["Bi", "S"])
    element_data = funmixer.get_element_obs(ELEMENT, obs_data)

    # ------------------------------------------------------------------ diagnostics
    # Solve once with no regularization. This is the exact matrix inversion, and it is the
    # honest test of whether the network supports a linear solution at all: if sites clamp at
    # zero, or the amplification factors are large, the raw inversion is amplifying noise
    # rather than resolving sources.
    unregularized = funmixer.LinearSampleNetworkUnmixer(sample_network, use_regularization=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = unregularized.solve(element_data, data_covariance=RELATIVE_ERROR_PERCENT)

    amplification = np.array(list(raw.amplification.values()))
    unconstrained = np.array([raw.unconstrained_preds[n] for n in raw.node_order])
    print(f"Sites                       : {len(raw.node_order)}")
    print(f"Condition number of M       : {raw.condition_number:.4g}")
    print(
        "Noise amplification Q/q     : "
        f"median {np.median(amplification):.2f}, max {amplification.max():.1f}"
    )
    print(f"Sites clamped at zero       : {len(raw.clamped_nodes)}")
    print(
        "Unconstrained estimate range: "
        f"{unconstrained.min():.0f} to {unconstrained.max():.0f} mg/kg"
    )
    print(
        "  (values outside [0, 1e6] mg/kg mean the exact inversion is physically impossible,\n"
        "   which is the signal that regularization is needed)"
    )

    # ---------------------------------------------------------------------- L-curve
    # `plot_sweep_of_regularizer_strength` is duck-typed on solve/get_misfit/get_roughness,
    # so it works on the linear solver unchanged.
    problem = funmixer.LinearSampleNetworkUnmixer(sample_network, use_regularization=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        funmixer.plot_sweep_of_regularizer_strength(problem, element_data, -3, 2, 11)

    # ------------------------------------------------------------------------ solve
    solution = problem.solve(
        element_data,
        regularization_strength=REGULARIZATION_STRENGTH,
        data_covariance=RELATIVE_ERROR_PERCENT,
    )
    print(f"\nlambda                      : {solution.regularization_strength}")
    print(f"Effective degrees of freedom: {solution.effective_dof:.1f}")
    print(f"Sites clamped at zero       : {len(solution.clamped_nodes)}")
    relative_sigma = np.array(
        [solution.upstream_std[n] / abs(solution.upstream_preds[n]) for n in solution.node_order]
    )
    print(f"Median relative uncertainty : {100 * np.median(relative_sigma):.1f}%")

    funmixer.visualise_downstream(
        pred_dict=solution.downstream_preds, obs_dict=element_data, element=ELEMENT
    )

    # ------------------------------------------------------------------------- maps
    area_dict = funmixer.get_unique_upstream_areas(sample_network, labels)
    concentration_map = funmixer.get_upstream_concentration_map(area_dict, solution.upstream_preds)
    uncertainty_map = funmixer.get_upstream_concentration_map(area_dict, solution.upstream_std)

    _, axes = plt.subplots(1, 2, figsize=(15, 6))
    concentrations = axes[0].imshow(concentration_map)
    axes[0].set_title(f"Recovered {ELEMENT} source concentration")
    plt.colorbar(concentrations, ax=axes[0], label="mg/kg")

    uncertainties = axes[1].imshow(uncertainty_map)
    axes[1].set_title(f"Propagated 1-sigma uncertainty ({RELATIVE_ERROR_PERCENT:.0f}% data error)")
    plt.colorbar(uncertainties, ax=axes[1], label="mg/kg")
    plt.show()


if __name__ == "__main__":
    main()
