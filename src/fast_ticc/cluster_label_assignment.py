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

"""All math related to computing cluster labels in TICC."""

import logging
from typing import Tuple, List
import numpy as np
import sklearn.mixture

from fast_ticc.containers import model_state
from fast_ticc import likelihood
from fast_ticc import numba_guard

LOGGER = logging.getLogger(__name__)


def predict_cluster_labels(model: model_state.ModelState,
                           test_data: np.ndarray) -> model_state.ModelState:
    """Predict point labels given the current trained model.

    In this process, we start by computing the (negative) log likelihood
    for each point with respect to the Markov random field for each
    cluster.  We use those values as a cost function.  We want to find
    an assignment that minimizes the total cost.  We also introduce a
    regularization parameter (beta in the TICC paper) to encourage
    the solver to assign the same point to successive clusters.

    Arguments:
        model (containers.ModelState): Current trained model
        test_data (NumPy array, T x NW): Stacked training data

    Returns:
        Updated model state with new labels and assignment cost
    """

    log_likelihood = likelihood.all_points_all_clusters_log_likelihood(
        model, test_data
    )

    # this is important enough to call out on its own line --
    # the more unlikely a point is for a given cluster,
    # the more it costs to assign to that cluster
    label_assignment_cost = - log_likelihood

    (new_labels, cost) = assign_point_cluster_labels(
        label_assignment_cost=label_assignment_cost,
        label_switching_cost=model.arguments.label_switching_cost
    )

    new_model = model.shallow_copy()
    new_model.clusters = [cluster.deep_copy() for cluster in new_model.clusters]
    new_model.point_labels = new_labels
    new_model.label_assignment_cost = cost

    return new_model


@numba_guard.njit(parallel=False)
def assign_point_cluster_labels(
        label_assignment_cost: np.ndarray,
        label_switching_cost: float
        ) -> Tuple[List[int], float]:
    """Compute point cluster labels given cluster parameters.

    Given an array containing a cost for allocating each point to
    each cluster and a label switching cost that imposes a penalty
    every time the label changes from one point to the next, compute
    a minimum-cost assignment of points to clusters.

    This is Algorithm 1 in the paper.  I believe this is equivalent
    to the Viterbi algorithm.

    You have two options when specifying the label switching cost.

    1.  Supply a single value.  This will be used as the label-
        switching cost throughout the data set.

    2.  Supply a NumPy array with as many entries as there are data
        points (rows).  This allows you to vary the switching cost
        however you like.  Our main use for this is when we are
        computing labels jointly for several separate data sequences.
        There is no relationship between the state of the system
        at the end of one sequence and the beginning of the next,
        so the label switching cost between those two points
        should be zero.

        Each element ``label_switching_cost[i]`` is the cost
        of using different cluster labels at points ``i`` and
        ``i+1``.

    TODO: Verify that this works correctly.

    Arguments:
        label_assignment_cost (NumPy array): Cost for assigning
            each point (row) to each cluster (column).
        label_switching_cost (float or NumPy array): Regularization
            parameter.  See above for details.

    Returns:
        ``(labels, final_cost)``, where ``labels`` is an array of
        integers containing the cluster label for each point and
        ``final_cost`` is the best cost achieved by the optimizer.
    """

    (num_points, num_clusters) = label_assignment_cost.shape
    future_cost_vals = np.zeros(label_assignment_cost.shape)
    path_matrix = np.zeros(label_assignment_cost.shape, dtype=np.uint16)


    label_switching_cost = np.zeros(shape=(num_points,)) + label_switching_cost

    for i in range(num_points-2, -1, -1):
        # we're computing "What state should I step into, going from observation
        # i to j, to get the best cost in the future?"
        total_vals = future_cost_vals[i+1] + label_assignment_cost[i+1] + label_switching_cost[i]
        arg_general_min = np.argmin(total_vals)

        for cluster in range(num_clusters):
            # keeping the cluster the same does NOT incur a label switching cost
            # The optimal is either the previous minimum OR the discounted option
            if total_vals[arg_general_min] < total_vals[cluster] - label_switching_cost[i]:
                path_matrix[i, cluster] = arg_general_min
                future_cost_vals[i, cluster] = total_vals[arg_general_min]
            else:
                path_matrix[i, cluster] = cluster
                future_cost_vals[i, cluster] = total_vals[cluster] - label_switching_cost[i]


    # Read off the best path from the path matrix
    path = [-1] * num_points

    # the first location
    curr_location = np.argmin(future_cost_vals[0, :] + label_assignment_cost[0, :])
    path[0] = curr_location
    true_cost = future_cost_vals[0, path[0]] + label_assignment_cost[0, path[0]]

    # compute the path
    for i in range(num_points-1):
        path[i+1] = path_matrix[i, path[i]]

    # return the computed path
    return (path, true_cost)



def build_initial_clusters(num_clusters: int, training_data: np.ndarray) -> List[int]:
    """Compute an initial assignment of points to clusters

    We use a Gaussian mixture model to compute a first approximation
    to what our cluster labeling might look like.  It doesn't have
    any of the Toeplitz block structure we require from the ultimate
    result, but it's better than a random starting point.

    Arguments:
        num_clusters (int): How many components to include in the mixture model
        training_data (np.ndarray): num_points x (num_time_series * window_size)
            data array

    Returns:
        List of cluster IDs for each point
    """

    LOGGER.debug("Building Gaussian mixture model for initial point labels")
    gmm = sklearn.mixture.GaussianMixture(n_components=num_clusters,
                                          covariance_type="full")
    gmm.fit(training_data)
    point_labels = gmm.predict(training_data)

    labels_as_pylist = [point_labels[i] for i in range(point_labels.size)]
    return labels_as_pylist
