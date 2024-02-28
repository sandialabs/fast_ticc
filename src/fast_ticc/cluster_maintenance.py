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

"""Detect and repopulate empty clusters.

Like many cluster optimization algorithms, TICC can fail if one or more
clusters become empty.  We call repopulate_empty_clusters() during each
iteration to identify and fix that situation before the algorithm can
crash.
"""

import logging
import random
from typing import List, Tuple

import numpy as np

from fast_ticc.containers import model_state

LOGGER = logging.getLogger(__name__)


def repopulate_empty_clusters(
    model: model_state.ModelState
) -> model_state.ModelState:
    """Repopulate any nearly-empty clusters.

    Empty or nearly-empty clusters (those that contain fewer than
    2 points) lead to degenerate behavior in the solver.  When we
    detect one, we reconstruct it by taking a number of points
    from larger clusters that can afford to spare them.

    The size of the repopulated cluster is controlled by the
    user and is stored in model.arguments.min_cluster_size.

    Note that this function does not update the empirical
    covariance or stacked data mean for clusters.  That happens
    just before the optimization step.

    Arguments:
        model (ModelState): Current TICC model
        training_data (NumPy array): T x NW array of (stacked)
            data points

    Reads:
        Point labels
        Training data

    Writes:
        Cluster covariance
        Cluster data mean
        Point labels

    Returns:
        New ModelState.  If all clusters were adequately populated,
        this will be the same as the input.
    """

    clusters_to_repopulate = set()
    for (cluster_id, cluster) in enumerate(model.clusters):
        if cluster.size < 2:
            clusters_to_repopulate.add(cluster_id)

    if len(clusters_to_repopulate) == 0:
        LOGGER.debug("Cluster repopulation not needed at this iteration.")
        return model

    LOGGER.info("Need to repopulate %d cluster(s): %s",
                len(clusters_to_repopulate), clusters_to_repopulate)

    new_model = model.shallow_copy()
    new_model.clusters = [cluster.deep_copy() for cluster in model.clusters]

    # Draw from clusters with the largest spread (highest
    # covariance) to encourage diversity in the clustering
    # results
    donor_cluster_ids = _find_ranked_donor_cluster_ids(new_model)

    remaining_donors = donor_cluster_ids
    for empty_cluster_id in clusters_to_repopulate:
        (donor_cluster_id, remaining_donors) = _find_point_donor(
            new_model, remaining_donors)
        LOGGER.info("Repopulating cluster %d with %d points from cluster %d.",
                    empty_cluster_id, model.arguments.min_cluster_size, donor_cluster_id)
        updated_point_labels = _move_random_points(
            new_model, donor_cluster_id, empty_cluster_id)
        new_model.point_labels = updated_point_labels

    # We don't need to recompute stacked_data_mean or log_determinant for the
    # clusters we change.  Those will be updated when we hit train_clusters().
    return new_model


def _find_point_donor(model: model_state.ModelState,
                      potential_donor_ids: List[int]) -> Tuple[int, List[int]]:
    """Find the first cluster with enough points to spare

    Reads from model:
        Minimum cluster size

    Arguments:
        model (ModelState): Model to draw from
        potential_donor_ids (list of int): Clusters to consider as donors

    Returns:
        Tuple of (donor_cluster_id, remaining_donors) where
            remaining_donors contains all the clusters that still
            have capacity to donate points

    Raises:
        RuntimeError: No suitable donor cluster could be found
    """

    min_cluster_size = model.arguments.min_cluster_size
    remaining_donors = list(potential_donor_ids)
    while len(remaining_donors) > 0:
        potential_donor_id = remaining_donors[0]
        potential_donor_size = model.clusters[potential_donor_id].size
        if potential_donor_size >= 2 * min_cluster_size:
            # We've found a winner.  Will it still be able
            # to donate more points afterward?
            if potential_donor_size < 3 * min_cluster_size:
                # No, it won't be big enough.
                remaining_donors.pop(0)
            return (potential_donor_id, remaining_donors)

        # This cluster isn't big enough to donate points.
        # Eliminate it from consideration and move on.
        remaining_donors.pop()

    raise RuntimeError((
        f"Unable to find a donor cluster with at least "
        f"{2 * min_cluster_size} points.  You may have set "
        f"min_cluster_size too high."
    ))


def _find_ranked_donor_cluster_ids(model: model_state.ModelState) -> List[int]:
    """Find and rank potential donor clusters

    When repopulating an empty cluster, we want to take points
    from the existing cluster with the largest spread (covariance
    magnitude) in the hope of generating more diversity in the
    cluster population.

    To be a donor, a cluster must contain at least twice the
    minimum cluster population according to the model arguments.

    Arguments:
        model (ModelState): Current cluster parameters

    Reads from model:
        Minimum cluster size, cluster computed covariance,
        cluster sizes

    Returns:
        List of available donor cluster IDs
    """

    potential_donor_ids = [i
                           for i in range(len(model.clusters))
                           if model.clusters[i].size >= 2 * model.arguments.min_cluster_size]

    cluster_spread = [np.linalg.norm(cluster.computed_covariance)
                      for cluster in model.clusters]

    def get_cluster_spread(i):
        return cluster_spread[i]

    ranked_donor_ids = sorted(
        potential_donor_ids, key=get_cluster_spread, reverse=True)
    return ranked_donor_ids


def _move_random_points(model: model_state.ModelState,
                        donor_cluster_id: int,
                        recipient_cluster_id: int) -> List[int]:
    """Draw points to repopulate a cluster

    Arguments:
        model (ModelState): TICC model in progress
        donor_cluster_id (int): Index of cluster that will donate points
        recipient_cluster_id (int): Index of cluster that will receive points
    Reads from model:
        Minimum cluster size

    Returns:
        New list of point labels for all clusters
    """

    available_point_ids = model.clusters[donor_cluster_id].member_points
    donated_point_indices = random.sample(range(len(available_point_ids)),
                                          model.arguments.min_cluster_size)
    donated_point_ids = [available_point_ids[i] for i in donated_point_indices]
    new_point_labels = list(model.point_labels)
    for point_id in donated_point_ids:
        assert new_point_labels[point_id] == donor_cluster_id, \
            "Point to be reassigned is not found in the donor cluster. "
        new_point_labels[point_id] = recipient_cluster_id

    return new_point_labels


def update_cluster_member_data_statistics(cluster: model_state.ClusterParameters,
                                          training_data: np.ndarray,
                                          use_biased_covariance: bool
                                          ) -> model_state.ClusterParameters:
    """Update mean, covariance for a cluster

    We use the mean and covariance of the data points in a cluster
    as inputs to the ADMM optimization and log likelihood computation.
    We update those values every time we build a new assignment of
    points to clusters.

    Arguments:
        cluster (fast_ticc.containers.model_state.ClusterParameters):
            Cluster being updated
        member_points (list of int): Points assigned to that cluster.
            These must be valid row indices for ``training_data``
            below.
        training_data (NumPy array): Input data for TICC.
        use_biased_covariance: If True, use a biased estimator for the
            population covariance.  If False, try to correct that bias.

    Returns:
        New ClusterParameters with empirical_covariance and stacked_data_mean
        members filled in.
    """

    assert cluster.size > 0, "Cluster needs at least one point assigned to it."
    updated_cluster = cluster.shallow_copy()
    training_data_this_cluster = training_data[cluster.member_points, :]

    updated_cluster.empirical_covariance = np.cov(
        np.transpose(training_data_this_cluster),
        bias=use_biased_covariance
    )
    updated_cluster.stacked_data_mean = np.mean(
        training_data_this_cluster, axis=0)

    return updated_cluster


def update_all_cluster_statistics(model: model_state.ModelState,
                                  training_data: np.ndarray) -> model_state.ModelState:
    """Update mean, covariance for all clusters

    Each cluster has a mean and covariance calculated from the data points
    that compose it.  Call this function after updating cluster membership
    to update those statistics.

    Since we update all the statistics in one go, build all your new
    clusters before you update the statistics.

    Arguments:
        model_state (fast_ticc.containers.ModelState):
            Current state of model in training.  We get the clusters,
            current point labels, and which covariance estimator to use
            from the model.
        training_data (NumPy array): Stacked training data used as
            TICC input

    Returns:
        New model state (shallow copy of previous state) with updated
        clusters
    """

    num_clusters = model.arguments.num_clusters
    cluster_members = {cluster_id: [] for cluster_id in range(num_clusters)}
    for (point_id, cluster_id) in enumerate(model.point_labels):
        cluster_members[cluster_id].append(point_id)

    updated_model = model.shallow_copy()
    for cluster_id in range(num_clusters):
        updated_model.clusters[cluster_id] = update_cluster_member_data_statistics(
            updated_model.clusters[cluster_id],
            training_data,
            model.arguments.biased_covariance
        )

    return updated_model
