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

"""Functions for solving the TICC graphical lasso problem.
   Supports parallelism where available."""

import logging
import multiprocessing

from typing import List, Union

import numpy as np

from fast_ticc import admm
from fast_ticc import matrix_compression
from fast_ticc.containers import model_state


TaskList = List[multiprocessing.pool.AsyncResult]

LOGGER = logging.getLogger(__name__)


def optimize_markov_random_fields(model: model_state.ModelState,
                                  stacked_training_data: np.ndarray,
                                  pool: multiprocessing.Pool
                                  ) -> model_state.ModelState:
    """Solve the TICC Graphical Lasso problem

    This is Section 4.2 in the TICC paper.  Here we treat the
    cluster labels as fixed and optimize the values in the Markov
    random field for each cluster.

    This is also where parallelism enters the implementation.
    Since clusters are independent of one another, we can
    run each optimization task in its own process.  Note that
    this will only take advantage of a single core for each
    cluster being created.  NumPy may also parallelize some of
    the work behind your back.

    Arguments:
        model (fast_ticc.containers.ModelState): Current model state
        pool (multprocessing.Pool): Already-initialized task pool

    Returns:
        New containers.modelState instance.  The clusters will
        contain the updated MRF values.
    """

    num_time_series = int(stacked_training_data.shape[1] / model.arguments.window_size)

    # Fire off the task to compute a new optimal Markov random field
    #
    optimization_tasks = [None] * model.arguments.num_clusters
    for cluster_id in range(model.arguments.num_clusters):
        optimization_tasks[cluster_id] = _setup_optimization_task(model.clusters[cluster_id],
                                                                  num_time_series,
                                                                  model.arguments.window_size,
                                                                  model.arguments.sparsity_weight,
                                                                  pool)


    #
    # Collect the task results and save the new Markov random field
    # values in the clusters
    #
    return _retrieve_optimization_results(model, optimization_tasks)


def _zero_small_elements(array: np.ndarray, epsilon: float, copy: bool=True) -> np.ndarray:
    """Zero out small-magnitude elements of an array

    Arguments:
        array (NumPy array): Array object to filter
        epsilon (float): "big enough" threshold

    Keyword Arguments:
        copy (bool): If True (the default), operates on a copy
            of the input.  If False, operates on the input array
            in-place.

    Returns:
        NumPy array where every element whose absolute value was
        smaller than epsilon is set to zero.  If copy is True,
        this array will be freshly allocated.  If copy is False,
        this array will be the same object that was passed in.
    """

    if copy:
        filtered = np.copy(array)
    else:
        filtered = array
    small_element_indices = (filtered < epsilon) & (filtered > -epsilon)
    filtered[small_element_indices] = 0
    return filtered


def _retrieve_optimization_results(model: model_state.ModelState,
                                   optimization_tasks: TaskList) -> model_state.ModelState:
    """Uncompress ADMM results and store in model

    This is the gather method corresponding to the broadcast/
    scatter that happens in setup_optimization_tasks().  We
    pull back the compressed, optimized inverse covariance
    matrix (aka Markov random field) from each task, turn it
    back into usable matrix form, and store it in its
    corresponding cluster.

    Arguments:
        model (fast_ticc.containers.ModelState):
            Model in progress.
        optimization_tasks (list of multiprocessing.pool.AsyncResult):
            Tasks spawned in setup_optimization_tasks()

    Reads from model:
        min_meaningful_covariance parameter
        cluster containers

    Returns:
        New model state with updated clusters
    """

    updated_clusters = []
    for (cluster, optimization_task) in zip(model.clusters, optimization_tasks):
        assert optimization_task is not None
        admm_result = optimization_task.get()
        updated_clusters.append(
            _update_cluster_covariances(model, cluster, admm_result.theta)
        )

    model = model.shallow_copy()
    model.clusters = updated_clusters
    return model


def _update_cluster_covariances(model: model_state.ModelState,
                                cluster: model_state.ClusterParameters,
                                admm_result: np.ndarray) -> model_state.ClusterParameters:
    """Update a cluster's contents with optimizer results

    This function starts with the result that comes back
    from the ADMM solver and updates the (trained) inverse
    covariance and (computed) covariance matrices in the
    appropriate cluster.

    Arguments:
        model (fast_ticc.containers.ModelState):
            Current model (only used to retrieve filter threshold for
            small elements)
        cluster (fast_ticc.containers.ClusterParameters):
            Cluster to which the optimization results belong
        admm_result (NumPy array): Raw result from ADMM solver task

    Returns:
        New ClusterParameters with updated train_inverse,
        computed_covariance, log_determinant
    """

    optimized_inverse_covariance = _reconstruct_optimized_matrix(model, admm_result)
    computed_covariance = np.linalg.inv(optimized_inverse_covariance)

    updated_cluster = cluster.shallow_copy()
    updated_cluster.computed_covariance = computed_covariance
    updated_cluster.train_inverse = optimized_inverse_covariance
    updated_cluster.log_determinant = np.log(
        np.linalg.det(optimized_inverse_covariance)
    )
    return updated_cluster


def _reconstruct_optimized_matrix(model: model_state.ModelState,
                                  compressed_result: np.ndarray) -> np.ndarray:
    """Reconstruct the NW x NW inverse covariance matrix from the solver result

    The ADMM solver returns its results in compressed form.  This
    function reconstructs the full symmetric matrix from that
    form and zeros out any entries with magnitudes too small to
    be useful.

    Arguments:
        model (containers.ModelState): Current model state
            in progress (used to get at epsilon threshold for
            filtering)
        compressed_result (NumPy array): Raw result from ADMM
            solver containing ((NW)^2 + NW) / 2 elements

    Returns:
        Symmetric NW x NW matrix
    """

    return _zero_small_elements(
        matrix_compression.reinflate_matrix(compressed_result),
        model.arguments.min_meaningful_covariance,
        copy=False
    )


def _setup_optimization_task(cluster: model_state.ClusterParameters,
                             num_data_series: int,
                             window_size: int,
                             density_penalty: Union[float, np.ndarray],
                             pool: multiprocessing.pool.Pool) -> multiprocessing.pool.AsyncResult:
    """Dispatch an optimization task to our worker pool

    We run all of the ADMM solvers that optimize the values for
    each cluster's MRFs in subprocesses.  This function fires off
    the task for one of those.

    Arguments:
        cluster (containers.ClusterParameters): Cluster to optimize for
        num_data_series (int): Count of data series in the original data
        window_size (int): Size of subsequence window for TICC
        density_penalty (float or NumPy array): Lambda parameter from
            TICC paper - this is the regularization term to drive the
            covariance matrix toward sparsity.  If you want to use a
            constant penalty for all elements in the matrix, just pass
            in the constant value. If you want to vary it for different
            elements, assemble the NW x NW array yourself and pass that in.
        pool (multiprocessing.pool.Pool): Task pool for parallel task

    Returns:
        multiprocessing.pool.AsyncResult for newly created task
    """

    admm_args = [
        cluster.empirical_covariance,
        density_penalty,
        window_size,
        num_data_series
    ]

    admm_kwargs = {
        "rho": 1,
        "rho_update": None,
        "max_iterations": 1000,
        "relative_tolerance": 1e-6,
        "absolute_tolerance": 1e-6,
        "verbose": False
    }

    return pool.apply_async(admm.admm_optimize_theta,
                            admm_args,
                            admm_kwargs)


def _update_cluster_statistics(cluster: model_state.ClusterParameters,
                               training_data: np.ndarray,
                               biased_covariance: bool) -> model_state.ClusterParameters:
    """Recompute covariances and data mean for a cluster

    Arguments:
        cluster (model.ClusterParameters): State container for the
            cluster we're updating
        training_data (NumPy array): Stacked data array, T x NW elements

    Returns:
        New containers.ClusterParameters with member_points, stacked_data_mean,
        and computed_covariance filled in
    """

    if cluster.size == 0:
        assert RuntimeError((
                "Empty cluster detected in _update_cluster_statistics. "
                "This shouldn't happen."
                ))

    updated_cluster = cluster.shallow_copy()
    training_data_this_cluster = training_data[cluster.member_points, :]
    updated_cluster.stacked_data_mean = np.mean(training_data_this_cluster, axis=0)
    updated_cluster.empirical_covariance = np.cov(
        np.transpose(training_data_this_cluster),
        bias=biased_covariance
    )
    return updated_cluster
