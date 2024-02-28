# Test script: Run TICC on sample trajectory, check output labels
#
# This is a "have I broken it?" test for TICC.  We pull an example
# trajectory from Tracktable's sample data, define four mundane
# feature functions, run TICC, and compare its label output with
# what we expect.
#
# We make no claims that these are good feature functions or a
# good trajectory to test on, just that it runs and the output
# hasn't changed.


from fast_ticc import front_end
import numpy as np
import pytest
import os
# Disable Numba JIT compilation
os.environ["NUMBA_DISABLE_JIT"] = "1"


@pytest.fixture(scope="module")
def multiple_trajectory_features(load_test_data):
    return load_test_data("multiple_trajectory_features")


@pytest.fixture
def ground_truth_result(load_test_data):
    return load_test_data("single_trajectory_ticc_result")


@pytest.fixture(scope="module")
def computed_ticc_result(multiple_trajectory_features,
                         num_clusters,
                         window_size,
                         label_switching_cost,
                         random_seed):

    np.random.seed(random_seed)
    ticc_multi_result = front_end.ticc_joint_labels(
        multiple_trajectory_features,
        window_size=window_size,
        num_clusters=num_clusters,
        num_processors=num_clusters,
        label_switching_cost=label_switching_cost
    )
    return ticc_multi_result


def test_ticc_multiple_trajectory_labels(computed_ticc_result, num_regression):
    result_dict = {}
    for cluster_id in range(computed_ticc_result.num_clusters):
        # For some reason, pytest-regressions will only check multiple arrays with
        # different shapes if they're floats.
        # It's safe to convert these to floats because labels are always small enough
        # to be represented exactly.
        labels = computed_ticc_result.point_labels[cluster_id]
        result_dict[f"cluster_{cluster_id}_labels"] = np.array(
            labels).astype(np.float32)

    num_regression.check(result_dict)


def test_ticc_multiple_trajectory_mrf(computed_ticc_result, ndarrays_regression):
    result_dict = {}
    for cluster_id in range(computed_ticc_result.num_clusters):
        result_dict[f"cluster_{cluster_id}_mrf"] = computed_ticc_result.markov_random_fields[cluster_id]
    ndarrays_regression.check(result_dict)


def test_ticc_multiple_trajectory_bayesian_information_criterion(computed_ticc_result, num_regression):
    result_dict = {"BIC": computed_ticc_result.bayesian_information_criterion}
    num_regression.check(result_dict)


def test_ticc_multiple_trajectory_calinski_harabasz_index(computed_ticc_result, num_regression):
    result_dict = {
        "CHI": computed_ticc_result.calinski_harabasz_index}
    num_regression.check(result_dict)


def test_ticc_multiple_trajectory_overall_log_likelihood(computed_ticc_result, num_regression):
    result_dict = {
        "overall_log_likelihood": computed_ticc_result.overall_log_likelihood,
        "overall_log_likelihood_mean": computed_ticc_result.overall_log_likelihood_mean,
        "overall_log_likelihood_median": computed_ticc_result.overall_log_likelihood_median
    }
    num_regression.check(result_dict)


def test_ticc_multiple_trajectory_cluster_log_likelihood(computed_ticc_result, num_regression):
    result_dict = {
        "per_cluster_log_likelihood_mean": computed_ticc_result.cluster_log_likelihood_mean,
        "per_cluster_log_likelihood_median": computed_ticc_result.cluster_log_likelihood_median
    }
    num_regression.check(result_dict)
