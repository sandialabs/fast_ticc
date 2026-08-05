# Copyright 2023-2026 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
###
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
###
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
###
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in
# the documentation and/or other materials provided with the
# distribution.
###
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
###
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Test script: Run TICC on a single trajectory, grab trained model,
# use to label new data (actually the same input data)
#

from fast_ticc import front_end as ticc_front_end
from fast_ticc.containers import results as ft_results
import fast_ticc
import numpy as np
import pytest
import os
# Disable Numba JIT compilation
#os.environ["NUMBA_DISABLE_JIT"] = "1"


@pytest.fixture(scope="module")
def single_trajectory_features(load_test_data):
    return load_test_data("single_trajectory_features")


@pytest.fixture(scope="module")
def initial_ticc_result(single_trajectory_features,
                         min_meaningful_covariance,
                         random_seed,
                         label_switching_cost,
                         num_clusters,
                         window_size) -> fast_ticc.SingleDataSeriesResult:

    np.random.seed(random_seed)
    ticc_result = ticc_front_end.ticc_labels(
        single_trajectory_features,
        window_size=window_size,
        num_clusters=num_clusters,
        min_meaningful_covariance=min_meaningful_covariance,
        num_processors=num_clusters,
        label_switching_cost=label_switching_cost
    )

    return ticc_result


@pytest.fixture
def new_labels_from_old_model_single(single_trajectory_features,
                                     min_meaningful_covariance,
                                     random_seed,
                                     label_switching_cost,
                                     num_clusters,
                                     window_size) -> fast_ticc.SingleDataSeriesResult:

    np.random.seed(random_seed)
    first_ticc_result = ticc_front_end.ticc_labels(
            single_trajectory_features,
            window_size=window_size,
            num_clusters=num_clusters,
            min_meaningful_covariance=min_meaningful_covariance,
            num_processors=num_clusters,
            label_switching_cost=label_switching_cost
        )

    trained_model = first_ticc_result.trained_model

    relabel_result = ticc_front_end.ticc_labels(
            single_trajectory_features,
            window_size=window_size,
            num_clusters=num_clusters,
            min_meaningful_covariance=min_meaningful_covariance,
            num_processors=num_clusters,
            label_switching_cost=label_switching_cost,
            initial_model=trained_model,
            allow_model_updates=False
    )

    return relabel_result


def test_ticc_relabel_single_trajectory_labels(new_labels_from_old_model_single, num_regression):
    result_dict = {
        "point_labels": new_labels_from_old_model_single.point_labels
    }
    num_regression.check(result_dict)


def test_ticc_relabel_single_trajectory_mrf(new_labels_from_old_model_single, ndarrays_regression):
    result_dict = {}
    for i in range(new_labels_from_old_model_single.num_clusters):
        result_dict[f"cluster_{i}_mrf"] = new_labels_from_old_model_single.markov_random_fields[i]
    ndarrays_regression.check(result_dict)


def test_ticc_relabel_single_trajectory_label_cost(new_labels_from_old_model_single, num_regression):
    result_dict = {
        "label_cost": new_labels_from_old_model_single.label_assignment_cost
    }
    num_regression.check(result_dict)


def test_ticc_relabel_single_trajectory_bayesian_information_criterion(new_labels_from_old_model_single, num_regression):
    result_dict = {
        "BIC": new_labels_from_old_model_single.bayesian_information_criterion
    }
    num_regression.check(result_dict)


def test_ticc_relabel_single_trajectory_calinski_harabasz_index(new_labels_from_old_model_single, num_regression):
    result_dict = {
        "CHI": new_labels_from_old_model_single.calinski_harabasz_index}
    num_regression.check(result_dict)


def test_ticc_relabel_single_trajectory_overall_log_likelihood(new_labels_from_old_model_single, num_regression):
    result_dict = {
        "overall_log_likelihood": new_labels_from_old_model_single.overall_log_likelihood,
        "overall_log_likelihood_mean": new_labels_from_old_model_single.overall_log_likelihood_mean,
        "overall_log_likelihood_median": new_labels_from_old_model_single.overall_log_likelihood_median
    }
    num_regression.check(result_dict)


def test_ticc_relabel_single_trajectory_cluster_log_likelihood(new_labels_from_old_model_single, num_regression):
    result_dict = {
        "cluster_log_likelihood_mean": new_labels_from_old_model_single.cluster_log_likelihood_mean,
        "cluster_log_likelihood_median": new_labels_from_old_model_single.cluster_log_likelihood_median
    }
    num_regression.check(result_dict)


def test_relabeled_ticc_model_fully_populated(
        new_labels_from_old_model_single: fast_ticc.SingleDataSeriesResult,
        window_size: int,
        num_clusters: int,
        single_trajectory_features: np.ndarray):
    trained_model = new_labels_from_old_model_single.trained_model

    num_points = single_trajectory_features.shape[0]
    num_sensors = single_trajectory_features.shape[1]

    nw = num_sensors * window_size

    assert len(trained_model.per_cluster_mean) == num_clusters
    assert len(trained_model.inverse_covariance) == num_clusters
    assert trained_model.window_size == window_size
    assert trained_model.num_clusters == num_clusters

    for mean in trained_model.per_cluster_mean:
        assert mean.shape == (nw,)

    for inv_covariance in trained_model.inverse_covariance:
        assert inv_covariance.shape == (nw, nw)


# ----------------------------------------------------------
# Below here are the tests for multiple trajectories at once
# ----------------------------------------------------------

@pytest.fixture(scope="module")
def multiple_trajectory_features(load_test_data):
    return load_test_data("multiple_trajectory_features")

@pytest.fixture(scope="module")
def new_labels_from_old_model_multiple(
                         multiple_trajectory_features,
                         num_clusters,
                         window_size,
                         label_switching_cost,
                         random_seed):

    np.random.seed(random_seed)
    first_ticc_result = ticc_front_end.ticc_joint_labels(
        multiple_trajectory_features,
        window_size=window_size,
        num_clusters=num_clusters,
        num_processors=num_clusters,
        label_switching_cost=label_switching_cost
    )

    trained_model = first_ticc_result.trained_model

    relabel_result = ticc_front_end.ticc_joint_labels(
            multiple_trajectory_features,
            window_size=window_size,
            num_clusters=num_clusters,
            num_processors=num_clusters,
            label_switching_cost=label_switching_cost,
            initial_model=trained_model,
            allow_model_updates=False
    )

    return relabel_result


def test_ticc_multiple_trajectory_labels(new_labels_from_old_model_multiple, num_regression):
    result_dict = {}
    for cluster_id in range(new_labels_from_old_model_multiple.num_clusters):
        # For some reason, pytest-regressions will only check multiple arrays with
        # different shapes if they're floats.
        # It's safe to convert these to floats because labels are always small enough
        # to be represented exactly.
        labels = new_labels_from_old_model_multiple.point_labels[cluster_id]
        result_dict[f"cluster_{cluster_id}_labels"] = np.array(
            labels).astype(np.float32)

    num_regression.check(result_dict)


def test_ticc_multiple_trajectory_mrf(new_labels_from_old_model_multiple, ndarrays_regression):
    result_dict = {}
    for cluster_id in range(new_labels_from_old_model_multiple.num_clusters):
        result_dict[f"cluster_{cluster_id}_mrf"] = new_labels_from_old_model_multiple.markov_random_fields[cluster_id]
    ndarrays_regression.check(result_dict)


def test_ticc_multiple_trajectory_bayesian_information_criterion(new_labels_from_old_model_multiple, num_regression):
    result_dict = {"BIC": new_labels_from_old_model_multiple.bayesian_information_criterion}
    num_regression.check(result_dict)


def test_ticc_multiple_trajectory_calinski_harabasz_index(new_labels_from_old_model_multiple, num_regression):
    result_dict = {
        "CHI": new_labels_from_old_model_multiple.calinski_harabasz_index}
    num_regression.check(result_dict)


def test_ticc_multiple_trajectory_overall_log_likelihood(new_labels_from_old_model_multiple, num_regression):
    result_dict = {
        "overall_log_likelihood": new_labels_from_old_model_multiple.overall_log_likelihood,
        "overall_log_likelihood_mean": new_labels_from_old_model_multiple.overall_log_likelihood_mean,
        "overall_log_likelihood_median": new_labels_from_old_model_multiple.overall_log_likelihood_median
    }
    num_regression.check(result_dict)


def test_ticc_multiple_trajectory_cluster_log_likelihood(new_labels_from_old_model_multiple, num_regression):
    result_dict = {
        "per_cluster_log_likelihood_mean": new_labels_from_old_model_multiple.cluster_log_likelihood_mean,
        "per_cluster_log_likelihood_median": new_labels_from_old_model_multiple.cluster_log_likelihood_median
    }
    num_regression.check(result_dict)


