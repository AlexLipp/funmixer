"""
Tests for below-detection-limit (BDL) unmixing.

Mirrors the property-based template of `random_networks_test.py` for the cases where exact
recovery is still expected, and falls back to deterministic structural properties for censored
data - censoring destroys information, so no amount of regularisation recovers a unique truth
below a detection limit (see `funmixer/bdl_unmixer.py` for the full argument).
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import funmixer
from funmixer.bdl_unmixer import (
    BDLObservation,
    BDLSampleNetworkUnmixer,
    cp_bdl_log_ratio,
    get_bdl_element_obs,
    parse_bdl_value,
)
from funmixer.network_unmixer import geo_mean
from random_networks_test import (
    MAXIMUM_AREA,
    MAXIMUM_BRANCHING_FACTOR,
    MAXIMUM_CONC,
    MAXIMUM_HEIGHT,
    MAXIMUM_NUMBER_OF_NODES,
    MINIMUM_AREA,
    MINIMUM_CONC,
    TARGET_TOLERANCE,
    conc_list_to_dict,
    draw_random_log_uniform,
    generate_balanced_sample_network,
    generate_r_ary_sample_network,
)

# Regularization strength used where a well-posed censored solution is needed. Small enough
# that the data misfit dominates, matching the value used in the example scripts.
REGULARIZATION_STRENGTH = 1e-3

# Slack allowed on top of solver tolerance when asserting a one-sided bound is respected.
SOLVER_SLACK = 1e-4


# Fixed seed for the deterministic test network. Chosen arbitrarily (a prime, deliberately not
# the ubiquitous 42) and not tuned: every assertion below is structural and holds for any seed.
NETWORK_SEED = 7919


@pytest.fixture
def simple_network_and_data():
    """A small deterministic network with its forward-modelled downstream observations."""
    rng = np.random.default_rng(NETWORK_SEED)
    network = generate_r_ary_sample_network(
        N=7, branching_factor=2, areas=lambda: float(rng.uniform(1.0, 10.0))
    )
    upstream = {node: float(rng.uniform(10.0, 100.0)) for node in network.nodes}
    downstream = funmixer.forward_model(sample_network=network, upstream_concentrations=upstream)
    return network, upstream, downstream


def as_detected(observations: funmixer.ElementData) -> funmixer.BDLElementData:
    """Wrap plain concentrations as uncensored BDL observations."""
    return {site: BDLObservation(value=value, is_bdl=False) for site, value in observations.items()}


def censor_all(observations: funmixer.ElementData, factor: float) -> funmixer.BDLElementData:
    """Censor every observation at `factor` times its true value."""
    return {
        site: BDLObservation(value=value * factor, is_bdl=True)
        for site, value in observations.items()
    }


# ---------------------------------------------------------------------------------------
# Uncensored data: the existing exact-recovery contract must still hold
# ---------------------------------------------------------------------------------------


@given(
    branching_factor=st.integers(min_value=1, max_value=MAXIMUM_BRANCHING_FACTOR),
    height=st.integers(min_value=1, max_value=MAXIMUM_HEIGHT),
    min_area=st.floats(min_value=MINIMUM_AREA, max_value=MAXIMUM_AREA),
    max_area=st.floats(min_value=MINIMUM_AREA, max_value=MAXIMUM_AREA),
    min_conc=st.floats(min_value=MINIMUM_CONC, max_value=MAXIMUM_CONC),
    max_conc=st.floats(min_value=MINIMUM_CONC, max_value=MAXIMUM_CONC),
)
@settings(deadline=None, max_examples=25)
def test_balanced_network_uncensored_recovery(
    branching_factor: int,
    height: int,
    min_area: float,
    max_area: float,
    min_conc: float,
    max_conc: float,
) -> None:
    """With nothing censored, the BDL unmixer recovers sources as accurately as the base class."""
    if max_area < min_area:
        max_area, min_area = min_area, max_area
    if max_conc < min_conc:
        max_conc, min_conc = min_conc, max_conc

    areas = lambda: draw_random_log_uniform(min_area, max_area)  # noqa: E731
    concentrations = lambda: draw_random_log_uniform(min_conc, max_conc)  # noqa: E731

    network = generate_balanced_sample_network(
        branching_factor=branching_factor, height=height, areas=areas
    )
    upstream = conc_list_to_dict(network, concentrations)
    downstream = funmixer.forward_model(sample_network=network, upstream_concentrations=upstream)

    problem = BDLSampleNetworkUnmixer(
        sample_network=network, observation_data=as_detected(downstream), use_regularization=False
    )
    solution = problem.solve(as_detected(downstream))

    for node in network.nodes:
        assert np.isclose(solution.upstream_preds[node], upstream[node], rtol=TARGET_TOLERANCE)


@given(
    branching_factor=st.integers(min_value=1, max_value=MAXIMUM_BRANCHING_FACTOR),
    N=st.integers(min_value=2, max_value=MAXIMUM_NUMBER_OF_NODES),
    min_area=st.floats(min_value=MINIMUM_AREA, max_value=MAXIMUM_AREA),
    max_area=st.floats(min_value=MINIMUM_AREA, max_value=MAXIMUM_AREA),
    min_conc=st.floats(min_value=MINIMUM_CONC, max_value=MAXIMUM_CONC),
    max_conc=st.floats(min_value=MINIMUM_CONC, max_value=MAXIMUM_CONC),
)
@settings(deadline=None, max_examples=25)
def test_rary_network_uncensored_recovery(
    branching_factor: int,
    N: int,
    min_area: float,
    max_area: float,
    min_conc: float,
    max_conc: float,
) -> None:
    """The same exact-recovery contract, on full r-ary networks."""
    if max_area < min_area:
        max_area, min_area = min_area, max_area
    if max_conc < min_conc:
        max_conc, min_conc = min_conc, max_conc

    areas = lambda: draw_random_log_uniform(min_area, max_area)  # noqa: E731
    concentrations = lambda: draw_random_log_uniform(min_conc, max_conc)  # noqa: E731

    network = generate_r_ary_sample_network(N=N, branching_factor=branching_factor, areas=areas)
    upstream = conc_list_to_dict(network, concentrations)
    downstream = funmixer.forward_model(sample_network=network, upstream_concentrations=upstream)

    problem = BDLSampleNetworkUnmixer(
        sample_network=network, observation_data=as_detected(downstream), use_regularization=False
    )
    solution = problem.solve(as_detected(downstream))

    for node in network.nodes:
        assert np.isclose(solution.upstream_preds[node], upstream[node], rtol=TARGET_TOLERANCE)


def test_uncensored_matches_standard_unmixer(simple_network_and_data) -> None:
    """With nothing censored, results must agree with `SampleNetworkUnmixer`."""
    network, upstream, downstream = simple_network_and_data

    # Solved sequentially: `SampleNode` objects are mutated in place when a problem is built,
    # so results must be read out of one problem before the next is constructed.
    standard = funmixer.SampleNetworkUnmixer(sample_network=network, use_regularization=False)
    standard_preds = standard.solve(downstream).upstream_preds

    bdl = BDLSampleNetworkUnmixer(
        sample_network=network, observation_data=as_detected(downstream), use_regularization=False
    )
    bdl_preds = bdl.solve(as_detected(downstream)).upstream_preds

    for node in network.nodes:
        assert np.isclose(bdl_preds[node], standard_preds[node], rtol=TARGET_TOLERANCE)


# ---------------------------------------------------------------------------------------
# Censored data: deterministic structural properties
# ---------------------------------------------------------------------------------------


def test_censoring_above_truth_does_not_increase_misfit(simple_network_and_data) -> None:
    """
    Censoring at a limit above the true value can only lower the misfit.

    The limit is set to exactly twice each observation, so the DL/2 substitution leaves the
    normalising geometric mean identical to the uncensored case and the misfits are directly
    comparable. Termwise, max(1, c/2a) <= max(c/a, a/c) for every c, so the minimum must fall.
    """
    network, _, downstream = simple_network_and_data

    standard = funmixer.SampleNetworkUnmixer(sample_network=network, use_regularization=False)
    standard.solve(downstream)
    standard_misfit = standard.get_misfit()

    censored = censor_all(downstream, factor=2.0)
    with pytest.warns(UserWarning, match="degenerate"):
        bdl = BDLSampleNetworkUnmixer(
            sample_network=network, observation_data=censored, use_regularization=False
        )
    bdl.solve(censored)

    assert bdl.get_misfit() <= standard_misfit * (1 + SOLVER_SLACK)


def test_censoring_below_truth_pulls_predictions_down(simple_network_and_data) -> None:
    """
    A detection limit below the true value binds: predictions are pushed to at most the limit.

    Every site is censored at half its true observation. Halving every source concentration
    reproduces exactly half of every downstream value, so a solution meeting every limit exists
    and any optimum must therefore respect them all.
    """
    network, _, downstream = simple_network_and_data

    censored = censor_all(downstream, factor=0.5)
    with pytest.warns(UserWarning, match="degenerate"):
        bdl = BDLSampleNetworkUnmixer(
            sample_network=network, observation_data=censored, use_regularization=False
        )
    solution = bdl.solve(censored)

    for node in network.nodes:
        limit = censored[node].value
        assert solution.downstream_preds[node] <= limit * (1 + SOLVER_SLACK)


def test_problem_stays_convex_and_dpp_with_mixed_censoring(simple_network_and_data) -> None:
    """
    The one-sided misfit is convex and DPP, so mixed censored/detected problems canonicalize once.

    This is the executable record of the convexity finding: the flat region below a detection
    limit does not make the problem non-convex, it only makes the minimiser non-unique.
    """
    network, _, downstream = simple_network_and_data

    mixed = {
        site: BDLObservation(value=value, is_bdl=(index % 2 == 0))
        for index, (site, value) in enumerate(downstream.items())
    }
    problem = BDLSampleNetworkUnmixer(sample_network=network, observation_data=mixed)

    assert problem._problem is not None
    assert problem._problem.is_dcp(dpp=True)
    solution = problem.solve(mixed, regularization_strength=REGULARIZATION_STRENGTH)
    assert np.isfinite(solution.objective_value)


def test_bdl_misfit_expression_is_convex_and_dpp() -> None:
    """`cp_bdl_log_ratio` itself is DCP-convex and DPP-compliant."""
    import cvxpy as cp

    from funmixer.cvxpy_extensions import ReciprocalParameter

    prediction = cp.Variable(pos=True)
    limit = ReciprocalParameter(pos=True)
    limit.value = 2.0

    expression = cp_bdl_log_ratio(prediction, limit)
    assert expression.is_convex()
    assert expression.is_dcp(dpp=True)


def test_regularization_resolves_degeneracy(simple_network_and_data) -> None:
    """
    Regularization tightens an otherwise under-determined censored solution.

    With every site censored above its true value, much of the misfit sits in its flat region
    and cannot distinguish between candidate sources. The regularizer breaks the tie by pulling
    each source towards the geometric mean of the observations, so a stronger regularizer must
    produce a narrower spread of recovered source concentrations.
    """
    network, _, downstream = simple_network_and_data
    censored = censor_all(downstream, factor=2.0)

    problem = BDLSampleNetworkUnmixer(sample_network=network, observation_data=censored)

    def log_spread(regularization_strength: float) -> float:
        solution = problem.solve(censored, regularization_strength=regularization_strength)
        return float(np.std(np.log([solution.upstream_preds[n] for n in network.nodes])))

    assert log_spread(1.0) < log_spread(1e-4)


# ---------------------------------------------------------------------------------------
# Normalisation, warnings and validation
# ---------------------------------------------------------------------------------------


def test_geometric_mean_uses_half_the_detection_limit(simple_network_and_data) -> None:
    """Censored observations enter the normalising geometric mean at half their limit."""
    network, _, downstream = simple_network_and_data

    censored = {
        site: BDLObservation(value=value, is_bdl=(index % 2 == 0))
        for index, (site, value) in enumerate(downstream.items())
    }
    problem = BDLSampleNetworkUnmixer(sample_network=network, observation_data=censored)
    problem.solve(censored, regularization_strength=REGULARIZATION_STRENGTH)

    expected = geo_mean([obs.value / 2 if obs.is_bdl else obs.value for obs in censored.values()])
    assert np.isclose(problem._obs_geo_mean, expected)


def test_warns_when_censored_data_used_without_regularization(simple_network_and_data) -> None:
    """Unregularized censored problems warn about degeneracy."""
    network, _, downstream = simple_network_and_data
    with pytest.warns(UserWarning, match="degenerate"):
        BDLSampleNetworkUnmixer(
            sample_network=network,
            observation_data=censor_all(downstream, factor=2.0),
            use_regularization=False,
        )


def test_no_warning_when_nothing_is_censored(simple_network_and_data) -> None:
    """Uncensored data is exactly the base-class problem, so it must not warn."""
    network, _, downstream = simple_network_and_data
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        BDLSampleNetworkUnmixer(
            sample_network=network,
            observation_data=as_detected(downstream),
            use_regularization=False,
        )


def test_missing_observation_raises(simple_network_and_data) -> None:
    """Every site in the network needs an observation to fix its censoring status."""
    network, _, downstream = simple_network_and_data
    incomplete = as_detected(downstream)
    dropped = next(iter(incomplete))
    del incomplete[dropped]

    with pytest.raises(ValueError, match="No observation supplied"):
        BDLSampleNetworkUnmixer(sample_network=network, observation_data=incomplete)


def test_changing_censoring_pattern_between_solves_raises(simple_network_and_data) -> None:
    """The censoring pattern is baked into the problem when it is built."""
    network, _, downstream = simple_network_and_data
    observations = as_detected(downstream)
    problem = BDLSampleNetworkUnmixer(sample_network=network, observation_data=observations)

    changed = dict(observations)
    site = next(iter(changed))
    changed[site] = BDLObservation(value=changed[site].value, is_bdl=True)

    with pytest.raises(ValueError, match="censoring status"):
        problem.solve(changed, regularization_strength=REGULARIZATION_STRENGTH)


def test_montecarlo_holds_detection_limits_fixed(simple_network_and_data) -> None:
    """Monte Carlo perturbs detected values but leaves detection limits untouched."""
    network, _, downstream = simple_network_and_data
    observations = {
        site: BDLObservation(value=value, is_bdl=(index % 2 == 0))
        for index, (site, value) in enumerate(downstream.items())
    }
    problem = BDLSampleNetworkUnmixer(sample_network=network, observation_data=observations)

    resampled = problem._resample_observations(observations, relative_error=10.0)

    for site, original in observations.items():
        assert resampled[site].is_bdl == original.is_bdl
        if original.is_bdl:
            assert resampled[site].value == original.value
        else:
            assert resampled[site].value != original.value


# ---------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------


@pytest.fixture
def mixed_observation_table() -> pd.DataFrame:
    """A small table mixing detected values, censored strings, whitespace and blanks."""
    return pd.DataFrame(
        {
            "sample_name": ["A", "B", "C", "D", "E"],
            "Mg": ["12.5", "<0.5", "< 0.75", "", "3"],
            "Ni": [1.0, 2.0, np.nan, 4.0, 5.0],
        }
    )


def test_get_bdl_element_obs_parses_mixed_column(mixed_observation_table) -> None:
    """Censored strings, whitespace and blanks are handled in one pass."""
    observations = get_bdl_element_obs("Mg", mixed_observation_table)

    assert set(observations) == {"A", "B", "C", "E"}  # blank cell dropped
    assert observations["A"] == BDLObservation(value=12.5, is_bdl=False)
    assert observations["B"] == BDLObservation(value=0.5, is_bdl=True)
    assert observations["C"] == BDLObservation(value=0.75, is_bdl=True)
    assert observations["E"] == BDLObservation(value=3.0, is_bdl=False)


def test_get_bdl_element_obs_drops_missing_numeric_values(mixed_observation_table) -> None:
    """A NaN in a purely numeric column is missing data, not a censored value."""
    observations = get_bdl_element_obs("Ni", mixed_observation_table)

    assert set(observations) == {"A", "B", "D", "E"}
    assert all(not obs.is_bdl for obs in observations.values())


def test_get_bdl_element_obs_rejects_unknown_element(mixed_observation_table) -> None:
    """An element that is not a column gets a message naming what is available."""
    with pytest.raises(ValueError, match="not a column"):
        get_bdl_element_obs("Pb", mixed_observation_table)


@pytest.mark.parametrize(
    "raw, expected",
    [
        (1.5, BDLObservation(value=1.5, is_bdl=False)),
        ("1.5", BDLObservation(value=1.5, is_bdl=False)),
        ("<1.5", BDLObservation(value=1.5, is_bdl=True)),
        ("  < 1.5  ", BDLObservation(value=1.5, is_bdl=True)),
        ("1e-3", BDLObservation(value=1e-3, is_bdl=False)),
    ],
)
def test_parse_bdl_value_accepts_valid_forms(raw, expected) -> None:
    """Numbers and '<number' strings parse, with surrounding whitespace tolerated."""
    assert parse_bdl_value(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", None, np.nan])
def test_parse_bdl_value_treats_blanks_as_absent(raw) -> None:
    """Blank and missing cells are absent data rather than an error."""
    assert parse_bdl_value(raw) is None


@pytest.mark.parametrize("raw", ["<0", "0", "-1", "<-2"])
def test_parse_bdl_value_rejects_non_positive(raw) -> None:
    """The misfit is undefined at zero, so non-positive values must be rejected loudly."""
    with pytest.raises(ValueError, match="finite and strictly positive"):
        parse_bdl_value(raw, sample_name="A")


@pytest.mark.parametrize("raw", ["not a number", "<", "<abc", "1.2.3"])
def test_parse_bdl_value_rejects_unparseable(raw) -> None:
    """An unparseable cell names the offending sample and the accepted format."""
    with pytest.raises(ValueError, match="Could not parse"):
        parse_bdl_value(raw, sample_name="A")
