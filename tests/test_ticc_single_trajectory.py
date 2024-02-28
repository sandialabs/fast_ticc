# Copyright 2023 National Technology & Engineering Solutions of Sandia,
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

# Test script: Run TICC on a single trajectory, check output labels
# against precomputed version
#

from fast_ticc import front_end as ticc_front_end
import numpy as np
import pytest
import os
# Disable Numba JIT compilation
os.environ["NUMBA_DISABLE_JIT"] = "1"


@pytest.fixture(scope="module")
def single_trajectory_features(load_test_data):
    return load_test_data("single_trajectory_features")


@pytest.fixture(scope="module")
def computed_ticc_result(single_trajectory_features,
                         min_meaningful_covariance,
                         random_seed,
                         label_switching_cost,
                         num_clusters,
                         window_size):

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


def test_ticc_single_trajectory_labels(computed_ticc_result, num_regression):
    result_dict = {
        "point_labels": computed_ticc_result.point_labels
    }
    num_regression.check(result_dict)


def test_ticc_single_trajectory_mrf(computed_ticc_result, ndarrays_regression):
    result_dict = {}
    for i in range(computed_ticc_result.num_clusters):
        result_dict[f"cluster_{i}_mrf"] = computed_ticc_result.markov_random_fields[i]
    ndarrays_regression.check(result_dict)


def test_ticc_single_trajectory_label_cost(computed_ticc_result, num_regression):
    result_dict = {
        "label_cost": computed_ticc_result.label_assignment_cost
    }
    num_regression.check(result_dict)


def test_ticc_single_trajectory_bayesian_information_criterion(computed_ticc_result, num_regression):
    result_dict = {
        "BIC": computed_ticc_result.bayesian_information_criterion
    }
    num_regression.check(result_dict)


def test_ticc_single_trajectory_calinski_harabasz_index(computed_ticc_result, num_regression):
    result_dict = {
        "CHI": computed_ticc_result.calinski_harabasz_index}
    num_regression.check(result_dict)


def test_ticc_single_trajectory_overall_log_likelihood(computed_ticc_result, num_regression):
    result_dict = {
        "overall_log_likelihood": computed_ticc_result.overall_log_likelihood,
        "overall_log_likelihood_mean": computed_ticc_result.overall_log_likelihood_mean,
        "overall_log_likelihood_median": computed_ticc_result.overall_log_likelihood_median
    }
    num_regression.check(result_dict)


def test_ticc_single_trajectory_cluster_log_likelihood(computed_ticc_result, num_regression):
    result_dict = {
        "cluster_log_likelihood_mean": computed_ticc_result.cluster_log_likelihood_mean,
        "cluster_log_likelihood_median": computed_ticc_result.cluster_log_likelihood_median
    }
    num_regression.check(result_dict)
