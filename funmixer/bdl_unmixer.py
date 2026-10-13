#!/usr/bin/env python3
"""
Unmixing of river-network tracer data containing below-detection-limit (BDL) observations.

Geochemical surveys routinely report a concentration as ``"<X"``, meaning only that the
concentration is somewhere at or below the instrument detection limit ``X``. Passing ``X``
into the standard unmixer asserts that the concentration *is* ``X``, which over-states what
the measurement says and biases the recovered sources upwards.

`BDLSampleNetworkUnmixer` behaves exactly like `SampleNetworkUnmixer` except that censored
observations use a one-sided misfit::

    detected observation:  max(c_pred / c_obs, c_obs / c_pred)   (`cp_log_ratio`)
    censored observation:  max(1, c_pred / DL)                   (`cp_bdl_log_ratio`)

The censored form is flat for any prediction at or below the detection limit, encoding
"any value below the detection limit is equally plausible", and grows in the usual relative
way above it.

Convexity
---------
The one-sided misfit is the pointwise maximum of a constant and a linear function of the
prediction, so it is DCP-convex and DPP-compliant; the whole problem remains convex. What it
loses is *strictness*: inside the flat region every prediction scores identically, so the
minimiser is not unique and the solver may return an arbitrary point there. Regularisation
pulls each source concentration towards the geometric mean of the observations and restores
a well-posed answer, which is why `use_regularization=False` raises a warning here.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cvxpy as cp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .cvxpy_extensions import ReciprocalParameter
from .network_unmixer import (
    ElementData,
    RateConstantData,
    SampleNetworkUnmixer,
    geo_mean,
)

# Character that marks a censored value in an input table, e.g. "<0.5".
BDL_PREFIX = "<"

# Censored values are replaced by (detection limit x this factor) when computing the
# geometric mean used to normalise the problem. One half is the standard - if imperfect -
# substitution for censored geochemical data. It affects only the normalisation constant,
# never the misfit itself, which always compares against the full detection limit.
BDL_SUBSTITUTION_FACTOR = 0.5


@dataclass(frozen=True)
class BDLObservation:
    """
    A single tracer observation that may be below the detection limit.

    Attributes:
        value: The measured concentration, or the detection limit when `is_bdl` is True.
        is_bdl: True if the observation was reported as below the detection limit.
    """

    value: float
    is_bdl: bool

    def __post_init__(self) -> None:
        if not np.isfinite(self.value) or self.value <= 0:
            kind = "detection limit" if self.is_bdl else "concentration"
            raise ValueError(
                f"Invalid {kind} '{self.value}': must be finite and strictly positive. "
                "The misfit function is undefined at zero, so remove or replace "
                "non-positive entries before unmixing."
            )

    @property
    def substituted_value(self) -> float:
        """
        The value used in place of this observation when computing the geometric mean.

        Returns:
            Half the detection limit if censored, otherwise the measured concentration.
        """
        return self.value * BDL_SUBSTITUTION_FACTOR if self.is_bdl else self.value


# Mapping of sample name to a possibly-censored observation.
BDLElementData = Dict[str, BDLObservation]


def cp_bdl_log_ratio(a: cp.Variable, b: ReciprocalParameter) -> cp.Expression:
    """
    Returns the one-sided convex misfit for an observation below the detection limit.

    Evaluates to `max(1, a / b)`: flat (at the perfect-fit value of 1) for any prediction at
    or below the detection limit, and equal to `cp_log_ratio` above it.

    Args:
        a: The CVXPY variable holding the predicted concentration.
        b: The ReciprocalParameter holding the detection limit.

    Returns:
        A convex, DPP-compliant expression penalising predictions above the detection limit.
    """
    return cp.maximum(1.0, a * b.rp)


def parse_bdl_value(raw: Any, sample_name: str = "<unknown>") -> Optional[BDLObservation]:
    """
    Parses one cell of a tracer column into a `BDLObservation`.

    Accepts a number, or a string holding either a number or a censored value such as
    "<0.5" or "< 0.5". Blank and missing cells are treated as absent data.

    Args:
        raw: The raw cell value, as read from a table.
        sample_name: Name of the sample, used only to make error messages actionable.

    Returns:
        The parsed observation, or None if the cell is blank or missing.

    Raises:
        ValueError: If the cell is neither blank, a number, nor a valid "<number" string.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        is_bdl = text.startswith(BDL_PREFIX)
        number = text[len(BDL_PREFIX) :].strip() if is_bdl else text
        try:
            value = float(number)
        except ValueError:
            raise ValueError(
                f"Could not parse the value '{raw}' for sample '{sample_name}'. Expected a "
                f"number, or a detection limit written as '{BDL_PREFIX}number' (e.g. "
                f"'{BDL_PREFIX}0.5')."
            ) from None
        return BDLObservation(value=value, is_bdl=is_bdl)

    value = float(raw)
    if np.isnan(value):
        return None
    return BDLObservation(value=value, is_bdl=False)


def get_bdl_element_obs(element: str, obs_data: pd.DataFrame) -> BDLElementData:
    """
    Extracts observed element data, including censored values, from a pandas DataFrame.

    The counterpart of `get_element_obs` for tables that mix numbers with censored entries
    such as "<0.5". Assumes the first column contains sample names. Samples with a blank or
    missing entry for this element are omitted.

    Args:
        element: The symbol of the element, matching a column name in `obs_data`.
        obs_data: The table of observations.

    Returns:
        A mapping of sample name to `BDLObservation`.

    Raises:
        ValueError: If `element` is not a column of `obs_data`, or a cell cannot be parsed.
    """
    if element not in obs_data.columns:
        raise ValueError(
            f"Element '{element}' is not a column of the observation table. "
            f"Available columns: {', '.join(map(str, obs_data.columns))}."
        )

    element_data: BDLElementData = {}
    for sample_name, raw in zip(obs_data.iloc[:, 0].tolist(), obs_data[element].tolist()):
        observation = parse_bdl_value(raw, sample_name=str(sample_name))
        if observation is not None:
            element_data[sample_name] = observation
    return element_data


class BDLSampleNetworkUnmixer(SampleNetworkUnmixer):
    """
    Unmixes a sample network whose observations may be below the detection limit.

    Identical to `SampleNetworkUnmixer` in every respect except the misfit used at censored
    sites, the geometric mean used to normalise the problem, and the treatment of censored
    values under Monte Carlo resampling.

    Which sites are censored is baked into the CVXPY problem when it is built, so the
    censoring pattern is fixed for the lifetime of the object. Observation *values* can still
    be changed between solves without recanonicalising, as in the base class.
    """

    def __init__(
        self,
        sample_network: nx.DiGraph,
        observation_data: BDLElementData,
        use_regularization: bool = True,
        rate_constants: Optional[RateConstantData] = None,
    ) -> None:
        """
        Initialize the BDLSampleNetworkUnmixer class.

        Args:
            sample_network: The sample network.
            observation_data: Observations for every site in the network. Only the censoring
                pattern is used here; values are supplied again at solve time.
            use_regularization: Flag indicating whether to use regularization. Strongly
                recommended - see the module docstring for why unregularized censored
                problems are degenerate.
            rate_constants: Per-site first-order decay constants. See the base class.

        Raises:
            ValueError: If any site in the network has no observation.
        """
        missing = sorted(str(node) for node in sample_network.nodes if node not in observation_data)
        if missing:
            raise ValueError(
                f"No observation supplied for sample site(s): {', '.join(missing)}. Every site "
                "in the network needs an entry in `observation_data`; either add the missing "
                "observations or remove those sites from the network."
            )

        # NOTE: this must be assigned *before* `super().__init__()`. The base constructor
        # calls `_build_primary_terms`, which calls `_misfit_term`, which reads this mapping.
        self._is_bdl: Dict[str, bool] = {site: obs.is_bdl for site, obs in observation_data.items()}

        if not use_regularization and any(self._is_bdl.values()):
            warnings.warn(
                "Unmixing censored (below-detection-limit) data without regularization gives a "
                "degenerate problem: the misfit is flat below each detection limit, so many "
                "source concentrations fit equally well and the solver may return an arbitrary "
                "value from that flat region. The problem stays convex, so it will usually still "
                "report 'optimal'. Pass use_regularization=True and a regularization_strength to "
                "obtain a well-posed solution.",
                UserWarning,
                stacklevel=2,
            )

        super().__init__(
            sample_network=sample_network,
            use_regularization=use_regularization,
            rate_constants=rate_constants,
        )

    def _misfit_term(
        self, site_name: str, prediction: cp.Variable, observed: ReciprocalParameter
    ) -> cp.Expression:
        """
        Build the misfit at one site, one-sided if that site's observation is censored.

        Args:
            site_name: Name of the sample site.
            prediction: Parameter-free variable holding the normalised predicted concentration.
            observed: The observation (or detection limit) at this site.

        Returns:
            A convex expression penalising disagreement between prediction and observation.
        """
        if self._is_bdl[site_name]:
            return cp_bdl_log_ratio(prediction, observed)
        return super()._misfit_term(site_name, prediction, observed)

    def _check_censoring_pattern(self, observation_data: BDLElementData) -> None:
        """
        Check that the supplied censoring pattern matches the one the problem was built with.

        Args:
            observation_data: The observation data for this solve.

        Raises:
            ValueError: If any site's censoring status differs from that used at construction.
        """
        changed = sorted(
            str(site)
            for site, obs in observation_data.items()
            if self._is_bdl.get(site, obs.is_bdl) != obs.is_bdl
        )
        if changed:
            raise ValueError(
                f"The censoring status of sample site(s) {', '.join(changed)} differs from the "
                "data this problem was built with. Which sites are below the detection limit is "
                "fixed when the problem is constructed; build a new BDLSampleNetworkUnmixer for "
                "a different censoring pattern."
            )

    # pyre-fixme[14]: Overrides a method with a narrower observation type.
    def _set_observation_parameters(self, observation_data: BDLElementData) -> None:
        """
        Reset and set the observation parameters according to input observations.

        Censored sites contribute half their detection limit to the geometric mean used for
        normalisation, but the *full* detection limit to the misfit, which is what the
        one-sided penalty `max(1, c / DL)` compares against.

        Args:
            observation_data: The observation data, which may include censored values.
        """
        self._check_censoring_pattern(observation_data)

        obs_mean: float = geo_mean([obs.substituted_value for obs in observation_data.values()])
        self._obs_geo_mean = obs_mean

        # Reset all sites' observations
        for x in self._site_to_observation.values():
            x.value = None
        # Assign each observed value to a site, making sure that the site exists
        for site, obs in observation_data.items():
            assert site in self._site_to_observation
            # Normalise observation by mean
            self._site_to_observation[site].value = obs.value / obs_mean

        # Ensure that all sites in the problem were assigned
        for x in self._site_to_observation.values():
            assert x.value is not None

    # pyre-fixme[14]: Overrides a method with a narrower observation type.
    def _resample_observations(
        self, observation_data: BDLElementData, relative_error: float
    ) -> BDLElementData:
        """
        Draw one noisy realisation of the observations, holding detection limits fixed.

        A detection limit is a property of the instrument rather than a measurement, so it
        carries no analytical error to propagate. Keeping it fixed also keeps the censoring
        pattern constant across Monte Carlo repeats, so the problem is never rebuilt.

        Args:
            observation_data: The observed data for each site.
            relative_error: The *relative* error as a percentage.

        Returns:
            A resampled copy of the observation data.
        """
        return {
            sample: (
                obs
                if obs.is_bdl
                else BDLObservation(
                    value=obs.value * np.random.normal(loc=1, scale=relative_error / 100),
                    is_bdl=False,
                )
            )
            for sample, obs in observation_data.items()
        }


def visualise_downstream_bdl(
    pred_dict: ElementData,
    obs_dict: BDLElementData,
    element: str,
    decades_below_limit: float = 1.0,
) -> None:
    """
    Plot predicted against observed downstream concentrations, showing censored data as ranges.

    Detected observations are drawn as points. A censored observation is drawn as a horizontal
    bar spanning every concentration it could plausibly have taken, from a plotting floor up to
    its detection limit, because the datum constrains the observation only from above.

    The true lower bound is zero, which cannot be drawn on a log axis, so the bars are clipped
    to `decades_below_limit` decades below the smallest detection limit and the x-axis is
    limited to match.

    Args:
        pred_dict: Predicted downstream concentrations.
        obs_dict: Observed downstream concentrations, which may be censored.
        element: The symbol of the element.
        decades_below_limit: How many decades below the smallest detection limit to extend the
            censored bars and the x-axis.

    Raises:
        ValueError: If `obs_dict` is empty.
    """
    if not obs_dict:
        raise ValueError("No observations to plot: `obs_dict` is empty.")

    detected_obs: List[float] = []
    detected_pred: List[float] = []
    censored_limits: List[float] = []
    censored_pred: List[float] = []
    for sample, obs in obs_dict.items():
        if obs.is_bdl:
            censored_limits.append(obs.value)
            censored_pred.append(pred_dict[sample])
        else:
            detected_obs.append(obs.value)
            detected_pred.append(pred_dict[sample])

    all_obs = np.array(detected_obs + censored_limits)
    all_pred = np.array(detected_pred + censored_pred)
    # Floor the censored bars a fixed number of decades below the smallest limit, since the
    # true lower bound of zero is at minus infinity on a log axis.
    floor = float(np.amin(all_obs)) / 10**decades_below_limit

    ax = plt.gca()
    ax.plot([0, 1e6], [0, 1e6], alpha=0.5, color="grey")
    if censored_limits:
        ax.hlines(
            y=censored_pred,
            xmin=floor,
            xmax=censored_limits,
            color="tab:red",
            linewidth=1.5,
            alpha=0.8,
            label="Below detection limit",
        )
        ax.scatter(
            x=censored_limits,
            y=censored_pred,
            facecolors="none",
            edgecolors="tab:red",
            marker="o",
            label="Detection limit",
        )
    ax.scatter(x=detected_obs, y=detected_pred, color="tab:blue", label="Detected")

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Observed " + element + " concentration mg/kg")
    ax.set_ylabel("Predicted " + element + " concentration mg/kg")
    ax.set_xlim((floor * 0.9, float(np.amax(all_obs)) * 1.1))
    ax.set_ylim((float(np.amin(all_pred)) * 0.9, float(np.amax(all_pred)) * 1.1))
    ax.set_aspect(1)
    ax.legend(loc="lower right", fontsize="small")
