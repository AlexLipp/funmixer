# pyre-ignore-all-errors[56]
"""
Tests for the linear (least-squares) unmixing solver.

The network generators mirror those in `random_networks_test.py`. Tolerances here are far
tighter than for the non-linear solver because, at lambda = 0, the inversion is exact rather
than iterative.
"""

import warnings
from typing import Callable, Optional

import networkx as nx
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import funmixer
from funmixer.linear_unmixer import LinearSampleNetworkUnmixer

MINIMUM_CONC = 1.0
MAXIMUM_CONC = 1e2
MINIMUM_AREA = 1.0
MAXIMUM_AREA = 1e2
MAX_RATE_PARAMETER = 3.0

# The inversion is exact, so we can demand far more than the 1% used for the non-linear solver.
EXACT_TOLERANCE = 1e-8


def draw_random_log_uniform(min_val: float, max_val: float) -> float:
    return float(np.exp(np.random.uniform(np.log(min_val), np.log(max_val), 1))[0])


def generate_balanced_sample_network(
    branching_factor: int, height: int, areas: Callable[[], float]
) -> nx.DiGraph:
    """A balanced tree of sample sites, with flow directed towards the root."""
    G = nx.balanced_tree(branching_factor, height, create_using=nx.DiGraph)
    G = nx.reverse(G)
    for u, v in G.edges:
        G[u][v]["length"] = 1.0
    for node in G.nodes:
        G.nodes[node]["data"] = funmixer.SampleNode(
            name=node,
            area=areas(),
            downstream_node=funmixer.nx_get_downstream_node(G, node),
            x=-1,
            y=-1,
            total_upstream_area=0,
            label=0,
            upstream_nodes=[],
            distance_downstream=1.0,
        )
    return G


def default_network(branching_factor: int = 2, height: int = 3, seed: int = 0) -> nx.DiGraph:
    np.random.seed(seed)
    return generate_balanced_sample_network(
        branching_factor, height, lambda: draw_random_log_uniform(MINIMUM_AREA, MAXIMUM_AREA)
    )


def random_concentrations(network: nx.DiGraph) -> funmixer.ElementData:
    return {node: draw_random_log_uniform(MINIMUM_CONC, MAXIMUM_CONC) for node in network.nodes}


# --------------------------------------------------------------- forward operator


@given(
    branching_factor=st.integers(min_value=1, max_value=4),
    height=st.integers(min_value=1, max_value=4),
    rate_constant=st.floats(min_value=0.0, max_value=MAX_RATE_PARAMETER),
    conservative=st.booleans(),
    use_export_rates=st.booleans(),
)
@settings(deadline=None, max_examples=25)
def test_mixing_matrix_matches_forward_model(
    branching_factor: int,
    height: int,
    rate_constant: float,
    conservative: bool,
    use_export_rates: bool,
) -> None:
    """`M @ c` must reproduce `forward_model` exactly: it is the same sweep, vectorised."""
    network = default_network(branching_factor, height)
    rate_constants = None if conservative else {n: rate_constant for n in network.nodes}
    export_rates = (
        {n: draw_random_log_uniform(0.1, 10.0) for n in network.nodes} if use_export_rates else None
    )

    problem = LinearSampleNetworkUnmixer(network, use_regularization=False)
    M, node_order = (
        problem._build_mixing_matrix(export_rates, rate_constants)[0],
        problem.node_order,
    )

    upstream = random_concentrations(network)
    expected = funmixer.forward_model(
        sample_network=network,
        upstream_concentrations=upstream,
        export_rates=export_rates,
        rate_constants=rate_constants,
    )

    c = np.array([upstream[n] for n in node_order])
    predicted = M @ c
    for i, node in enumerate(node_order):
        assert np.isclose(predicted[i], expected[node], rtol=1e-12)


@given(
    branching_factor=st.integers(min_value=1, max_value=4),
    height=st.integers(min_value=1, max_value=4),
    conservative=st.booleans(),
)
@settings(deadline=None, max_examples=25)
def test_mixing_matrix_structure(branching_factor: int, height: int, conservative: bool) -> None:
    """M is lower triangular in topological order, with a positive diagonal."""
    network = default_network(branching_factor, height)
    rate_constants = None if conservative else {n: 1.0 for n in network.nodes}
    problem = LinearSampleNetworkUnmixer(network, use_regularization=False)
    M, _, _ = problem._build_mixing_matrix(None, rate_constants)

    assert np.allclose(M, np.tril(M)), "M must be lower triangular in topological order"
    assert np.all(np.diag(M) > 0), "M must have a strictly positive diagonal"
    if conservative:
        # Conservative mixing weights are a convex combination, so rows sum to one.
        assert np.allclose(M.sum(axis=1), 1.0)
    else:
        # Decay removes tracer from the numerator only, so rows sum to less than one.
        assert np.all(M.sum(axis=1) <= 1.0 + 1e-12)


def test_closed_form_inverse_matches_dense_inverse() -> None:
    """The sparse O(n) closed form must agree with an explicit dense inverse."""
    network = default_network(branching_factor=3, height=3)
    rate_constants = {n: 0.7 for n in network.nodes}
    problem = LinearSampleNetworkUnmixer(network, use_regularization=False)
    M, total_flux, own_flux = problem._build_mixing_matrix(None, rate_constants)

    rng = np.random.default_rng(0)
    d = rng.uniform(1.0, 10.0, size=M.shape[0])

    closed_form = problem._solve_unregularized(d, total_flux, own_flux, rate_constants)
    assert np.allclose(closed_form, np.linalg.solve(M, d), rtol=1e-10)
    assert np.allclose(problem._build_estimator(M, 0.0, None), np.linalg.inv(M), rtol=1e-10)


# ------------------------------------------------------------------- round trip


@given(
    branching_factor=st.integers(min_value=1, max_value=4),
    height=st.integers(min_value=1, max_value=4),
    rate_constant=st.floats(min_value=0.0, max_value=MAX_RATE_PARAMETER),
    conservative=st.booleans(),
)
@settings(deadline=None, max_examples=25)
def test_exact_round_trip(
    branching_factor: int, height: int, rate_constant: float, conservative: bool
) -> None:
    """forward_model -> solve must recover the source concentrations exactly at lambda = 0."""
    network = default_network(branching_factor, height)
    rate_constants = None if conservative else {n: rate_constant for n in network.nodes}
    upstream = random_concentrations(network)
    downstream = funmixer.forward_model(
        sample_network=network, upstream_concentrations=upstream, rate_constants=rate_constants
    )

    problem = LinearSampleNetworkUnmixer(
        network, use_regularization=False, rate_constants=rate_constants
    )
    solution = problem.solve(downstream)

    assert solution.clamped_nodes == []
    assert solution.residual_norm < 1e-10
    for node in network.nodes:
        assert np.isclose(solution.upstream_preds[node], upstream[node], rtol=EXACT_TOLERANCE)
        assert np.isclose(solution.downstream_preds[node], downstream[node], rtol=EXACT_TOLERANCE)


def test_returned_estimate_is_the_analytical_one() -> None:
    """
    On interior problems the returned estimate must be the analytical estimator exactly, and
    CVXPY must independently agree with it. This is what makes the covariance meaningful.
    """
    network = default_network(branching_factor=2, height=3)
    upstream = random_concentrations(network)
    downstream = funmixer.forward_model(network, upstream)

    for lam, use_reg in [(None, False), (1e-3, True), (1.0, True)]:
        problem = LinearSampleNetworkUnmixer(network, use_regularization=use_reg)
        solution = problem.solve(downstream, regularization_strength=lam)
        assert solution.clamped_nodes == []
        # `unconstrained_preds` is the analytical R @ d; the returned estimate must equal it.
        for node in network.nodes:
            assert solution.upstream_preds[node] == solution.unconstrained_preds[node]
        # And CVXPY, solving the same problem numerically, must land in the same place.
        R, node_order = problem.get_estimator()
        d = np.array([downstream[n] for n in node_order])
        assert np.allclose(R @ d, [solution.upstream_preds[n] for n in node_order], rtol=1e-6)


# ---------------------------------------------------------------- regularization


def test_regularized_solve_matches_augmented_least_squares() -> None:
    """
    The Cholesky/normal-equations path must agree with the numerically safer augmented
    least-squares form, which never forms M^T M.
    """
    network = default_network(branching_factor=3, height=2)
    upstream = random_concentrations(network)
    downstream = funmixer.forward_model(network, upstream)

    problem = LinearSampleNetworkUnmixer(network, use_regularization=True)
    for lam in [1e-4, 1e-2, 1.0, 10.0]:
        solution = problem.solve(downstream, regularization_strength=lam)
        M, node_order = problem.get_mixing_matrix()
        n = len(node_order)
        d = np.array([downstream[node] for node in node_order])
        P = np.eye(n) - np.ones((n, n)) / n

        augmented_operator = np.vstack([M, np.sqrt(lam) * P])
        augmented_data = np.concatenate([d, np.zeros(n)])
        expected, *_ = np.linalg.lstsq(augmented_operator, augmented_data, rcond=None)

        got = np.array([solution.upstream_preds[node] for node in node_order])
        assert np.allclose(got, expected, rtol=1e-7)


def test_small_lambda_reproduces_unregularized_solution() -> None:
    network = default_network(branching_factor=2, height=3)
    upstream = random_concentrations(network)
    downstream = funmixer.forward_model(network, upstream)

    unregularized = LinearSampleNetworkUnmixer(network, use_regularization=False).solve(downstream)
    barely = LinearSampleNetworkUnmixer(network, use_regularization=True).solve(
        downstream, regularization_strength=1e-12
    )
    for node in network.nodes:
        assert np.isclose(
            barely.upstream_preds[node], unregularized.upstream_preds[node], rtol=1e-5
        )


def test_large_lambda_drives_model_to_a_constant() -> None:
    """
    As lambda grows the variance penalty dominates and every c_i must converge to a common
    value. For a conservative network M is row-stochastic, so that value is the mean
    observation.
    """
    network = default_network(branching_factor=2, height=3)
    upstream = random_concentrations(network)
    downstream = funmixer.forward_model(network, upstream)

    problem = LinearSampleNetworkUnmixer(network, use_regularization=True)
    solution = problem.solve(downstream, regularization_strength=1e10)

    values = np.array(list(solution.upstream_preds.values()))
    assert np.allclose(values, values[0], rtol=1e-4), "all c_i should collapse to one value"
    assert np.isclose(values[0], np.mean(list(downstream.values())), rtol=1e-4)


def test_effective_dof_decreases_with_lambda() -> None:
    """Resolution degrades monotonically with lambda, starting from the identity at lambda=0."""
    network = default_network(branching_factor=2, height=3)
    upstream = random_concentrations(network)
    downstream = funmixer.forward_model(network, upstream)
    n = network.number_of_nodes()

    unregularized = LinearSampleNetworkUnmixer(network, use_regularization=False).solve(downstream)
    assert np.allclose(unregularized.resolution_matrix, np.eye(n), atol=1e-8)
    assert np.isclose(unregularized.effective_dof, n)

    problem = LinearSampleNetworkUnmixer(network, use_regularization=True)
    dofs = [
        problem.solve(downstream, regularization_strength=lam).effective_dof
        for lam in [1e-6, 1e-3, 1e-1, 1.0, 1e2]
    ]
    assert all(a > b for a, b in zip(dofs, dofs[1:])), f"not monotonically decreasing: {dofs}"
    # The penalty has a one-dimensional null space (constant vectors), which is never damped.
    assert dofs[-1] > 1.0


# --------------------------------------------------------------------- weighting


def test_weighting_cannot_change_an_unregularized_solution() -> None:
    """
    At lambda = 0 the fit is exact and passes through every observation, so there is nothing
    for a weighting to trade off: (M^T W M)^-1 M^T W = M^-1 for any invertible W. This is a
    mathematical identity, not an approximation, and is worth pinning down.
    """
    network = default_network(branching_factor=2, height=3)
    upstream = random_concentrations(network)
    downstream = funmixer.forward_model(network, upstream)
    problem = LinearSampleNetworkUnmixer(network, use_regularization=False)

    # A deliberately extreme weighting: five orders of magnitude between the best- and
    # worst-known sites.
    rng = np.random.default_rng(3)
    sigmas = {node: 10 ** rng.uniform(-2, 3) for node in network.nodes}

    unweighted = problem.solve(downstream, data_covariance=sigmas, weighted=False)
    weighted = problem.solve(downstream, data_covariance=sigmas, weighted=True)
    for node in network.nodes:
        assert np.isclose(weighted.upstream_preds[node], unweighted.upstream_preds[node], rtol=1e-9)
    # The covariance is C_c = R C_d R^T with the same R, so it too must be unchanged.
    assert np.allclose(weighted.upstream_covariance, unweighted.upstream_covariance, rtol=1e-9)


def test_weighting_changes_a_regularized_solution() -> None:
    """Once lambda > 0 the weighting decides which observations the penalty is allowed to
    sacrifice, so it must change the answer."""
    network = default_network(branching_factor=2, height=3)
    upstream = random_concentrations(network)
    downstream = funmixer.forward_model(network, upstream)
    problem = LinearSampleNetworkUnmixer(network, use_regularization=True)

    unweighted = problem.solve(
        downstream, regularization_strength=0.1, data_covariance=10.0, weighted=False
    )
    weighted = problem.solve(
        downstream, regularization_strength=0.1, data_covariance=10.0, weighted=True
    )
    assert weighted.weighted and not unweighted.weighted
    difference = max(
        abs(weighted.upstream_preds[n] - unweighted.upstream_preds[n]) for n in network.nodes
    )
    scale = max(abs(v) for v in unweighted.upstream_preds.values())
    assert difference / scale > 1e-3, "weighting should materially change the solution"


def test_weighted_estimator_matches_explicit_gls_formula() -> None:
    """The whitening transform must reproduce the textbook GLS normal equations."""
    network = default_network(branching_factor=3, height=2)
    upstream = random_concentrations(network)
    downstream = funmixer.forward_model(network, upstream)
    problem = LinearSampleNetworkUnmixer(network, use_regularization=True)

    rng = np.random.default_rng(11)
    sigmas = {node: 10 ** rng.uniform(-1, 2) for node in network.nodes}
    lam = 0.37
    solution = problem.solve(
        downstream, regularization_strength=lam, data_covariance=sigmas, weighted=True
    )

    M, node_order = problem.get_mixing_matrix()
    n = len(node_order)
    d = np.array([downstream[node] for node in node_order])
    d_bar = float(np.mean(d))
    # The estimator is built in normalised space, so the covariance must be scaled to match.
    C_d = np.diag(np.array([sigmas[node] for node in node_order]) ** 2) / d_bar**2
    W = np.linalg.inv(C_d)
    P = np.eye(n) - np.ones((n, n)) / n

    expected_R = np.linalg.solve(M.T @ W @ M + lam * P, M.T @ W)
    got_R, _ = problem.get_estimator()
    assert np.allclose(got_R, expected_R, rtol=1e-8)
    assert np.allclose(
        [solution.upstream_preds[node] for node in node_order], expected_R @ d, rtol=1e-7
    )


def test_unregularized_weighted_covariance_is_the_cramer_rao_bound() -> None:
    """
    At lambda = 0 the estimator is `M^-1`, so `C_c = M^-1 C_d M^-T = (M^T C_d^-1 M)^-1` -- the
    inverse Fisher information. The unregularized inversion is therefore an efficient
    estimator, attaining the Cramer-Rao lower bound, and no weighting can improve on it.
    """
    network = default_network(branching_factor=3, height=2)
    upstream = random_concentrations(network)
    downstream = funmixer.forward_model(network, upstream)
    problem = LinearSampleNetworkUnmixer(network, use_regularization=False)

    rng = np.random.default_rng(2)
    sigmas = {node: 10 ** rng.uniform(-1, 2) for node in network.nodes}
    solution = problem.solve(downstream, data_covariance=sigmas)

    M, node_order = problem.get_mixing_matrix()
    C_d = np.diag(np.array([sigmas[node] for node in node_order]) ** 2)
    cramer_rao = np.linalg.inv(M.T @ np.linalg.inv(C_d) @ M)
    assert np.allclose(solution.upstream_covariance, cramer_rao, rtol=1e-8)


def test_posterior_and_propagated_covariance_agree_at_zero_lambda() -> None:
    """At lambda = 0 the penalty contributes nothing, so both covariances must coincide."""
    network = default_network(branching_factor=3, height=2)
    upstream = random_concentrations(network)
    downstream = funmixer.forward_model(network, upstream)
    problem = LinearSampleNetworkUnmixer(network, use_regularization=True)
    solution = problem.solve(downstream, regularization_strength=0.0, data_covariance=7.0)
    assert np.allclose(
        solution.upstream_posterior_covariance, solution.upstream_covariance, rtol=1e-7
    )


def test_posterior_covariance_exceeds_propagated_when_regularized() -> None:
    """
    The propagated covariance measures only how the estimate scatters; the posterior also
    counts what the damping costs. They differ by exactly `lambda A^-1 P A^-1`, so the
    posterior is always the larger -- and quoting the propagated one alone at large lambda
    understates the error.
    """
    network = default_network(branching_factor=3, height=2)
    upstream = random_concentrations(network)
    downstream = funmixer.forward_model(network, upstream)
    problem = LinearSampleNetworkUnmixer(network, use_regularization=True)

    previous_ratio = 1.0
    for lam in [1e-3, 1e-1, 1.0, 10.0]:
        solution = problem.solve(
            downstream, regularization_strength=lam, data_covariance=7.0, weighted=True
        )
        propagated = np.diag(solution.upstream_covariance)
        posterior = np.diag(solution.upstream_posterior_covariance)
        assert np.all(posterior >= propagated - 1e-12)

        # Check the exact identity relating the two.
        M, node_order = problem.get_mixing_matrix()
        n = len(node_order)
        d_bar = float(np.mean([downstream[node] for node in node_order]))
        C_d = np.diag(np.array([downstream[node] for node in node_order]) * 0.07) ** 2
        A = M.T @ np.linalg.inv(C_d / d_bar**2) @ M + lam * (np.eye(n) - np.ones((n, n)) / n)
        A_inv = np.linalg.inv(A)
        expected = (A_inv - lam * A_inv @ (np.eye(n) - np.ones((n, n)) / n) @ A_inv) * d_bar**2
        assert np.allclose(solution.upstream_covariance, expected, rtol=1e-6)

        # The gap widens with lambda: damping buys stability at the cost of accuracy.
        ratio = float(np.mean(np.sqrt(posterior / propagated)))
        assert ratio >= previous_ratio - 1e-9
        previous_ratio = ratio
    assert previous_ratio > 1.5, "the two covariances should diverge substantially by lambda=10"


def test_posterior_covariance_absent_when_unweighted() -> None:
    """The posterior form needs C_d^-1, so it is only defined for a weighted solve."""
    network = default_network(branching_factor=2, height=2)
    upstream = random_concentrations(network)
    downstream = funmixer.forward_model(network, upstream)
    problem = LinearSampleNetworkUnmixer(network, use_regularization=True)
    solution = problem.solve(
        downstream, regularization_strength=0.1, data_covariance=5.0, weighted=False
    )
    assert solution.upstream_covariance is not None
    assert solution.upstream_posterior_covariance is None


def test_weighted_misfit_is_a_chi_distance() -> None:
    """With a correct error model, a weighted solve at small lambda should give a misfit of
    order sqrt(n) -- one standard deviation per observation."""
    network = default_network(branching_factor=2, height=4)
    n = network.number_of_nodes()
    upstream = random_concentrations(network)
    truth = funmixer.forward_model(network, upstream)

    rng = np.random.default_rng(5)
    relative_error = 0.1
    noisy = {node: value * (1 + relative_error * rng.normal()) for node, value in truth.items()}
    sigmas = {node: relative_error * value for node, value in noisy.items()}

    problem = LinearSampleNetworkUnmixer(network, use_regularization=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem.solve(noisy, regularization_strength=1e-6, data_covariance=sigmas)
    # Essentially unregularized, so the fit is near-exact and the chi distance near zero.
    assert problem.get_misfit() < np.sqrt(n)


# ------------------------------------------------------------ error propagation


@pytest.mark.parametrize("lam, use_reg", [(None, False), (1e-2, True)])
def test_covariance_matches_monte_carlo(lam: Optional[float], use_reg: bool) -> None:
    """
    The load-bearing test: the closed-form covariance must reproduce what you get by actually
    perturbing the data and re-solving.

    The network and data are chosen so the solution stays comfortably interior, so that the
    estimator really is linear and the analytical result is exact.
    """
    np.random.seed(7)
    network = generate_balanced_sample_network(2, 2, lambda: 1.0)
    # A near-uniform source field keeps the differenced solution well away from zero.
    upstream = {node: 100.0 + np.random.uniform(-5, 5) for node in network.nodes}
    downstream = funmixer.forward_model(network, upstream)

    relative_error = 2.0  # percent
    problem = LinearSampleNetworkUnmixer(network, use_regularization=use_reg)
    solution = problem.solve(
        downstream, regularization_strength=lam, data_covariance=relative_error
    )
    assert solution.clamped_nodes == []

    R, node_order = problem.get_estimator()
    M, _ = problem.get_mixing_matrix()
    d = np.array([downstream[node] for node in node_order])
    sigma = d * relative_error / 100.0

    rng = np.random.default_rng(42)
    draws = 20000
    perturbed = d[None, :] + rng.normal(0.0, sigma, size=(draws, len(d)))
    sampled_c = perturbed @ R.T
    sampled_d = sampled_c @ M.T

    empirical_c = np.cov(sampled_c, rowvar=False)
    empirical_d = np.cov(sampled_d, rowvar=False)

    # Compare standard deviations: with 20k draws the standard error on a std is ~0.5%.
    assert np.allclose(
        np.sqrt(np.diag(empirical_c)), np.sqrt(np.diag(solution.upstream_covariance)), rtol=0.05
    )
    assert np.allclose(
        np.sqrt(np.diag(empirical_d)),
        np.sqrt(np.diag(solution.downstream_covariance)),
        rtol=0.05,
    )
    # And the full matrices, scaled by the diagonal to make the comparison dimensionless.
    scale = np.outer(
        np.sqrt(np.diag(solution.upstream_covariance)),
        np.sqrt(np.diag(solution.upstream_covariance)),
    )
    assert np.allclose(empirical_c / scale, solution.upstream_covariance / scale, atol=0.05)


def test_unregularized_downstream_covariance_equals_data_covariance() -> None:
    """
    At lambda = 0 the fit is exact, so the modelled observations are the observations and their
    covariance must be the input covariance unchanged.
    """
    network = default_network(branching_factor=2, height=3)
    upstream = random_concentrations(network)
    downstream = funmixer.forward_model(network, upstream)

    problem = LinearSampleNetworkUnmixer(network, use_regularization=False)
    solution = problem.solve(downstream, data_covariance=5.0)

    d = np.array([downstream[node] for node in solution.node_order])
    expected = np.diag((d * 5.0 / 100.0) ** 2)
    assert np.allclose(solution.downstream_covariance, expected, rtol=1e-6, atol=1e-12)


def test_covariance_input_forms_agree() -> None:
    """A scalar relative error, a dict of sigmas and an explicit matrix must agree."""
    network = default_network(branching_factor=2, height=2)
    upstream = random_concentrations(network)
    downstream = funmixer.forward_model(network, upstream)
    problem = LinearSampleNetworkUnmixer(network, use_regularization=False)

    scalar = problem.solve(downstream, data_covariance=10.0)
    sigmas = {node: value * 0.1 for node, value in downstream.items()}
    as_dict = problem.solve(downstream, data_covariance=sigmas)
    as_array = problem.solve(
        downstream,
        data_covariance=np.array([sigmas[node] for node in problem.node_order]),
    )
    as_matrix = problem.solve(
        downstream,
        data_covariance=np.diag(np.array([sigmas[node] for node in problem.node_order]) ** 2),
    )
    for other in [as_dict, as_array, as_matrix]:
        assert np.allclose(scalar.upstream_covariance, other.upstream_covariance, rtol=1e-10)


def test_covariance_scales_quadratically_with_units() -> None:
    """
    The problem is homogeneous, so rescaling the observations must rescale concentrations
    linearly and covariances quadratically. This checks the mean-normalisation cancels exactly.
    """
    network = default_network(branching_factor=2, height=3)
    upstream = random_concentrations(network)
    downstream = funmixer.forward_model(network, upstream)
    problem = LinearSampleNetworkUnmixer(network, use_regularization=False)

    base = problem.solve(downstream, data_covariance=10.0)
    factor = 1e4  # e.g. converting fractions to mg/kg
    scaled = problem.solve(
        {node: value * factor for node, value in downstream.items()}, data_covariance=10.0
    )
    for node in network.nodes:
        assert np.isclose(
            scaled.upstream_preds[node], base.upstream_preds[node] * factor, rtol=1e-9
        )
    assert np.allclose(
        scaled.upstream_covariance, base.upstream_covariance * factor**2, rtol=1e-9
    )


# ------------------------------------------------------------------ constraints


def test_clamping_warns_and_respects_the_lower_bound() -> None:
    """
    Data that violate the mixing model -- a downstream site carrying less tracer flux than its
    tributaries deliver -- must clamp at zero, warn, and still return a feasible solution.
    """
    network = generate_balanced_sample_network(2, 1, lambda: 1.0)
    node_order = [
        name for name, _ in funmixer.network_unmixer.nx_topological_sort_with_data(network)
    ]
    # Leaves are rich, the outlet is poor: impossible under conservative mixing.
    root = node_order[-1]
    observations = {node: (100.0 if node != root else 1.0) for node in network.nodes}

    problem = LinearSampleNetworkUnmixer(network, use_regularization=False)
    with pytest.warns(UserWarning, match="clamped at the lower bound"):
        solution = problem.solve(observations)

    assert solution.clamped_nodes, "expected at least one site to clamp"
    assert all(value >= -1e-8 for value in solution.upstream_preds.values())
    assert min(solution.unconstrained_preds.values()) < 0, "the raw estimate should go negative"
    assert solution.residual_norm > 0, "a clamped fit cannot be exact"


def test_clamping_warning_mentions_covariance_caveat() -> None:
    network = generate_balanced_sample_network(2, 1, lambda: 1.0)
    node_order = [
        name for name, _ in funmixer.network_unmixer.nx_topological_sort_with_data(network)
    ]
    observations = {node: (100.0 if node != node_order[-1] else 1.0) for node in network.nodes}
    problem = LinearSampleNetworkUnmixer(network, use_regularization=False)
    with pytest.warns(UserWarning, match="overstate uncertainty"):
        problem.solve(observations, data_covariance=5.0)


# ------------------------------------------------------------------- diagnostics


def test_amplification_is_total_over_own_flux() -> None:
    """Equal-area sub-basins in a balanced binary tree give amplification = subtree size."""
    network = generate_balanced_sample_network(2, 2, lambda: 1.0)
    upstream = random_concentrations(network)
    downstream = funmixer.forward_model(network, upstream)
    solution = LinearSampleNetworkUnmixer(network, use_regularization=False).solve(downstream)

    for node in network.nodes:
        subtree_size = len(nx.ancestors(network, node)) + 1
        assert np.isclose(solution.amplification[node], subtree_size, rtol=1e-10)


def test_misfit_and_roughness_are_available_for_the_l_curve() -> None:
    """
    `plot_sweep_of_regularizer_strength` is duck-typed on solve/get_misfit/get_roughness, so
    those three must work together on this class without any change to that helper.
    """
    network = default_network(branching_factor=2, height=3)
    upstream = random_concentrations(network)
    downstream = funmixer.forward_model(network, upstream)
    problem = LinearSampleNetworkUnmixer(network, use_regularization=True)

    misfits, roughnesses = [], []
    for lam in np.logspace(-4, 2, 7):
        problem.solve(downstream, regularization_strength=lam)
        misfits.append(problem.get_misfit())
        roughnesses.append(problem.get_roughness())

    # The classic L-curve trade-off: misfit rises, roughness falls, both monotonically.
    assert all(a <= b + 1e-9 for a, b in zip(misfits, misfits[1:])), misfits
    assert all(a >= b - 1e-9 for a, b in zip(roughnesses, roughnesses[1:])), roughnesses


def test_requires_regularization_strength_when_enabled() -> None:
    network = default_network(branching_factor=2, height=2)
    problem = LinearSampleNetworkUnmixer(network, use_regularization=True)
    upstream = random_concentrations(network)
    downstream = funmixer.forward_model(network, upstream)
    with pytest.raises(Exception, match="no strength assigned"):
        problem.solve(downstream)


def test_rejects_mismatched_observations() -> None:
    network = default_network(branching_factor=2, height=2)
    problem = LinearSampleNetworkUnmixer(network, use_regularization=False)
    upstream = random_concentrations(network)
    downstream = funmixer.forward_model(network, upstream)
    incomplete = {k: v for k, v in list(downstream.items())[:-1]}
    with pytest.raises(ValueError, match="No observation supplied"):
        problem.solve(incomplete)


def test_does_not_disturb_a_live_nonlinear_problem() -> None:
    """
    `SampleNode` fields are shared scratch space. Building and solving a linear problem on a
    graph must not corrupt a `SampleNetworkUnmixer` built on the same graph.
    """
    network = default_network(branching_factor=2, height=2)
    upstream = random_concentrations(network)
    downstream = funmixer.forward_model(network, upstream)

    nonlinear = funmixer.SampleNetworkUnmixer(network, use_regularization=False)
    before = nonlinear.solve(downstream).upstream_preds

    linear = LinearSampleNetworkUnmixer(network, use_regularization=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        linear.solve(downstream)

    after = nonlinear.solve(downstream).upstream_preds
    for node in network.nodes:
        assert np.isclose(before[node], after[node], rtol=1e-6)
