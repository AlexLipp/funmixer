#!/usr/bin/env python3
"""
Minimal working example of unmixing data containing below-detection-limit (BDL) values.

Uses `data/BDL_sample_data.csv`, a censored copy of the Cairngorms (NE Scotland) drainage
geochemistry survey in which the lowest decile of each element has been replaced by entries of
the form "<X" (see `data/make_bdl_sample_data.py`). Sample sites and the D8 flow-direction
raster are unchanged from `examples/unmix_mwe.py`, so the two scripts can be compared directly.

A censored observation says only that the concentration lies somewhere at or below the
detection limit. `BDLSampleNetworkUnmixer` models this with a one-sided misfit that is flat
below the limit, rather than pretending the concentration equals the limit.

Regularization is essential here: below a detection limit the misfit cannot distinguish between
candidate source concentrations, and the regularizer is what makes the solution well-posed.

Run from the repository root:

    python examples/unmix_BDL.py
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd

import funmixer

logging.getLogger().addHandler(logging.StreamHandler())

# Chosen because it is censored in the example dataset and has no missing measurements. Every
# site in the network needs an observation, so an element with gaps needs those sites removing.
ELEMENT = "Mg"

# Balances fit against the pull towards the geometric mean. Matches `examples/unmix_mwe.py`.
REGULARIZATION_STRENGTH = 10 ** (-3.0)

FLOWDIRS_FILENAME = "data/d8.tif"
SAMPLE_DATA_FILENAME = "data/BDL_sample_data.csv"


def main() -> None:
    """Unmix one censored element and plot the recovered sources and the data fit."""
    sample_network, labels = funmixer.get_sample_graph(
        flowdirs_filename=FLOWDIRS_FILENAME,
        sample_data_filename=SAMPLE_DATA_FILENAME,
    )

    # Read as strings so that censored entries such as "<3821.8" survive; the parser in
    # `get_bdl_element_obs` turns them into BDLObservation objects.
    obs_data = pd.read_csv(SAMPLE_DATA_FILENAME, dtype=str, keep_default_na=False)
    element_data = funmixer.get_bdl_element_obs(ELEMENT, obs_data)

    n_censored = sum(obs.is_bdl for obs in element_data.values())
    limits = sorted({obs.value for obs in element_data.values() if obs.is_bdl})
    print(f"{ELEMENT}: {n_censored} of {len(element_data)} observations are below detection.")
    print(f"Detection limits present: {limits}")

    problem = funmixer.BDLSampleNetworkUnmixer(
        sample_network=sample_network,
        observation_data=element_data,
    )

    solution = problem.solve(
        element_data,
        solver="clarabel",
        regularization_strength=REGULARIZATION_STRENGTH,
    )

    # Map the recovered source concentrations back onto their sub-basins
    area_dict = funmixer.get_unique_upstream_areas(sample_network, labels)
    upstream_map = funmixer.get_upstream_concentration_map(area_dict, solution.upstream_preds)

    # Censored observations are drawn as bars spanning the concentrations they could have taken
    funmixer.visualise_downstream_bdl(
        pred_dict=solution.downstream_preds, obs_dict=element_data, element=ELEMENT
    )
    plt.title(f"Observed vs predicted downstream {ELEMENT}")
    plt.show()

    plt.imshow(upstream_map)
    cb = plt.colorbar()
    cb.set_label(ELEMENT + " concentration mg/kg")
    plt.title("Upstream Concentration Map")
    plt.show()


if __name__ == "__main__":
    main()
