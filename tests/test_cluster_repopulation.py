### Copyright 2023 National Technology & Engineering Solutions of Sandia,
### LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
### U.S. Government retains certain rights in this software.
###
### Redistribution and use in source and binary forms, with or without
### modification, are permitted provided that the following conditions are
### met:
###
### 1. Redistributions of source code must retain the above copyright
###    notice, this list of conditions and the following disclaimer.
###
### 2. Redistributions in binary form must reproduce the above copyright
###    notice, this list of conditions and the following disclaimer in
###    the documentation and/or other materials provided with the
###    distribution.
###
### 3. Neither the name of the copyright holder nor the names of its
###    contributors may be used to endorse or promote products derived
###    from this software without specific prior written permission.
###
### THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
### “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
### LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
### A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
### HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
### SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
### LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
### DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
### THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
### (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
### OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Test cases for detecting and repairing empty clusters.

import os
# Disable Numba JIT compilation
os.environ["NUMBA_DISABLE_JIT"] = "1"

import itertools
import logging
from typing import List

import pytest

import numpy as np

from fast_ticc.containers import arguments
from fast_ticc.containers import model_state
from fast_ticc import cluster_maintenance


@pytest.fixture
def local_num_data_points() -> int:
    return 100

# This should divide num_data_points evenly
@pytest.fixture
def local_num_clusters() -> int:
    return 5

@pytest.fixture
def local_num_time_series() -> int:
    return 3

@pytest.fixture
def local_window_size() -> int:
    return 2

@pytest.fixture
def local_min_cluster_size() -> int:
    return 20


@pytest.fixture
def user_params(local_num_clusters, local_window_size, local_min_cluster_size) -> arguments.UserArguments:
    # local_num_data_points 100, local_num_time_series 3, window_size 2
    params = arguments.UserArguments(
        sparsity_weight=None,
        iteration_limit=None,
        label_switching_cost=0,
        min_cluster_size=local_min_cluster_size,
        min_meaningful_covariance=0,
        num_clusters=local_num_clusters,
        num_processors=-1,
        biased_covariance=False,
        window_size=local_window_size
    )
    return params


@pytest.fixture
def dummy_stacked_data(local_num_data_points, local_num_time_series, local_window_size) -> np.ndarray:
    """Random numbers, won't get used

    This is just so that we have an array of the proper size to
    pass into the function being tested.  The values don't matter.
    """
    # local_num_data_points 100, local_num_time_series 3, window_size 2
    return np.random.rand(local_num_data_points, local_num_time_series * local_window_size)


@pytest.fixture
def uniform_cluster_assignments(local_num_data_points, local_num_clusters) -> List[int]:
    """Assign an even number of points to each cluster

    If we have 10 points and 5 clusters, the cluster assignments look
    like [0, 0, 1, 1, 2, 2, 3, 3, 4, 4].  We don't worry about the
    case where local_num_points mod local_num_clusters != 0 because we control
    both values during test setup.
    """
    assert (local_num_data_points / local_num_clusters) == int(local_num_data_points / local_num_clusters)
    points_per_cluster = int(local_num_data_points / local_num_clusters)
    labels = []
    for cluster_id in range(local_num_clusters):
        labels.extend([cluster_id] * points_per_cluster)
    return labels


@pytest.fixture
def model_state_no_empty_clusters(user_params, dummy_stacked_data, uniform_cluster_assignments):
    state = model_state.ModelState.empty_model(user_params, dummy_stacked_data)
    state.point_labels = uniform_cluster_assignments
    for cluster in state.clusters:
        cluster.computed_covariance = np.array([1])

    return state


@pytest.fixture
def model_state_one_empty_one_donor(model_state_no_empty_clusters):
    # Assign all points in cluster 1 to cluster 2
    original_state = model_state_no_empty_clusters
    new_labels = [2 if old_label == 1 else old_label
                  for old_label in original_state.point_labels]
    original_state.point_labels = new_labels
    return original_state

@pytest.fixture
def model_state_two_empty_one_donor(model_state_no_empty_clusters):
    # Assign all points in clusters 1 and 4 to cluster 2
    original_state = model_state_no_empty_clusters
    new_labels = [2 if (old_label == 1 or old_label == 2) else old_label
                  for old_label in original_state.point_labels]
    original_state.point_labels = new_labels
    return original_state


@pytest.fixture
def model_state_two_empty_two_donors(model_state_no_empty_clusters):
    # Assign all points in cluster 1 to cluster 2 and cluster 4 to cluster 3
    original_state = model_state_no_empty_clusters
    new_labels = [2 if old_label == 1 else old_label
                  for old_label in original_state.point_labels]
    new_labels = [4 if old_label == 3 else old_label
                  for old_label in new_labels]
    original_state.point_labels = new_labels
    return original_state


def contains_all_points(state: model_state.ModelState, num_points: int) -> bool:
    all_member_lists = [cluster.member_points for cluster in state.clusters]
    concatenated_members = list(sorted(itertools.chain(*all_member_lists)))
    ground_truth_point_ids = list(range(num_points))

    return concatenated_members == ground_truth_point_ids

def all_clusters_large_enough(state: model_state.ModelState) -> bool:
    return all([
        cluster.size >= state.arguments.min_cluster_size
        for cluster in state.clusters
    ])


def test_no_changes_necessary(model_state_no_empty_clusters, local_num_data_points):
    new_state = cluster_maintenance.repopulate_empty_clusters(model_state_no_empty_clusters)
    assert new_state.point_labels == model_state_no_empty_clusters.point_labels
    assert contains_all_points(new_state, local_num_data_points)
    assert all_clusters_large_enough(new_state)


def test_one_empty_one_donor(model_state_one_empty_one_donor, dummy_stacked_data, local_min_cluster_size):
    assert model_state_one_empty_one_donor.clusters[2].size == 2 * local_min_cluster_size
    assert model_state_one_empty_one_donor.clusters[1].size == 0

    repopulated_state = cluster_maintenance.repopulate_empty_clusters(model_state_one_empty_one_donor)

    assert repopulated_state.clusters[2].size == local_min_cluster_size
    assert repopulated_state.clusters[1].size == local_min_cluster_size

    all_member_lists = [cluster.member_points for cluster in repopulated_state.clusters]
    all_point_ids = set(itertools.chain(*all_member_lists))

    # Make sure we haven't lost any points
    assert contains_all_points(repopulated_state, dummy_stacked_data.shape[0])
    assert all_clusters_large_enough(repopulated_state)


def test_no_donors(model_state_one_empty_one_donor, dummy_stacked_data):
    model_state_one_empty_one_donor.arguments.min_cluster_size = 100

    with pytest.raises(RuntimeError, match=r".*Unable to find a donor.*"):
        repopulated_state = cluster_maintenance.repopulate_empty_clusters(model_state_one_empty_one_donor)


def test_two_empty_one_donor(model_state_two_empty_one_donor, dummy_stacked_data):
    assert not all_clusters_large_enough(model_state_two_empty_one_donor)
    new_state = cluster_maintenance.repopulate_empty_clusters(model_state_two_empty_one_donor)

    assert contains_all_points(new_state, dummy_stacked_data.shape[0])
    assert all_clusters_large_enough(new_state)


def test_two_empty_two_donors(model_state_two_empty_two_donors, dummy_stacked_data):
    logging.basicConfig(level=logging.DEBUG)
    assert not all_clusters_large_enough(model_state_two_empty_two_donors)
    new_state = cluster_maintenance.repopulate_empty_clusters(model_state_two_empty_two_donors)

    assert contains_all_points(new_state, dummy_stacked_data.shape[0])
    assert all_clusters_large_enough(new_state)
