### Copyright 2023-2026 National Technology & Engineering Solutions of Sandia,
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

"""Main loop for TICC solver.

You probably don't want to call these functions directly.  Instead, use
the wrappers in fast_ticc, which in turn are imported from
front_end.py.
"""

import copy
import itertools
import logging
import multiprocessing
import os

from typing import List, Optional

import numpy as np

from fast_ticc import cluster_label_assignment
from fast_ticc import cluster_maintenance
from fast_ticc import graphical_lasso
from fast_ticc import cluster_metrics
from fast_ticc import likelihood

from fast_ticc.containers import arguments
from fast_ticc.containers import model_state
from fast_ticc.containers import results

# We need this to correctly type hint multiprocessing.Pool
from multiprocessing.pool import Pool as mp_Pool

LOGGER = logging.getLogger(__name__)


def fit_stacked_data(user_args: arguments.UserArguments,
                     stacked_training_data: np.ndarray
                     ) -> results.SingleDataSeriesResult:
    """Fit cluster labels to already-stacked data series.

    This is the TICC main loop.

    We expect the functions in the front end to handle data stacking
    and result splitting.

    Arguments:
        stacked_training_data (NumPy array): Data to fit.  Must have
            one row for each data point and T x W columns.  The
            first W columns at each time t are the original time series.
            Columns [W, 2*W) are the data values at time t+1.
            Columns [2*W, 3*W) are the data values at time t+2.
            Continue in this fashion all the way up to time
            t+(W-1).

    Returns:
        ticc.containers.results.SingleDataSeriesResult containing assigned
        labels and Markov random fields
    """

    model = model_state.ModelState.empty_model(
        user_args, stacked_training_data)


    model.point_labels = cluster_label_assignment.build_initial_clusters(
        user_args.num_clusters, stacked_training_data
    )
    model = _optimize_model(model, stacked_training_data)

    # Extract all the model statistics and build the derived results
    return _package_results(model, user_args, stacked_training_data)


def _optimize_model(model: model_state.ModelState,
                    stacked_training_data: np.ndarray) -> model_state.ModelState:


    assert model.arguments.iteration_limit > 0  # must have at least one iteration
    model.arguments.print()

    # We will use this to detect convergence
    previous_iteration_point_labels = None

    # The context manager syntax for multiprocessing pools is convenient
    # but causes problems with computing test coverage.  We initialize
    # and close it manually to let pycov/coverage do their thing.
    task_pool = _init_task_pool(model.arguments.num_processors)

    for current_iteration in range(model.arguments.iteration_limit):
        LOGGER.info("TICC: Beginning iteration %d", current_iteration)

        if current_iteration > 0:
            model = cluster_maintenance.repopulate_empty_clusters(
                model)

        model = cluster_maintenance.update_all_cluster_statistics(
            model, stacked_training_data
        )

        model = graphical_lasso.optimize_markov_random_fields(
            model, stacked_training_data, task_pool
        )

        model = cluster_label_assignment.predict_cluster_labels(
            model, stacked_training_data
        )

        if (previous_iteration_point_labels ==
                model.point_labels):
            LOGGER.info((
                "Cluster assignments have converged. Optimization "
                "complete."))
            break
        previous_iteration_point_labels = copy.copy(
            model.point_labels)
        # end of main loop

    # This will trigger pycov/coverage's handlers to record test coverage
    task_pool.close()
    task_pool.join()

    return model


def _init_task_pool(num_processes: int) -> mp_Pool:
    """Instantiate a task pool for multiprocessing

    When benchmarking or debugging, we might want to force the
    ADMM solver to use a single process.  This function allows
    us to control that at run time by setting the
    CUPCAKE_ENABLE_MULTIPROCESSING environment variable.

    If that environment variable is present and not empty,
    we will use multiprocessing.  If it is either absent
    or present and empty, we will default to a single process
    (actually a multiprocessing.Pool initialized with processes=1).

    The strange part is that in most cases, enabling multiprocessing
    results in slower results than if you leave num_processes
    set to 1.  This is because most modern BLAS implementations
    do parallelism on their own and are very, very heavily
    optimized to make efficient use of CPU cache.  If you have
    several of those running at once they will fight over the
    cache lines and registers and slow things down.

    Arguments:
        num_processes (int): Maximum number of processes to allow
            if we spawn a multiprocessing pool

    Returns:
        Either a multiprocessing.Pool or something that looks just
        like it but runs in a single process.  Note that the
        single-process version only supports apply_async()
        for now.
    """

    enable_multiprocessing = os.environ.get('CUPCAKE_ENABLE_MULTIPROCESSING', None)
    if enable_multiprocessing is not None and len(enable_multiprocessing) > 0:
        LOGGER.debug(
            "Initializing multiprocessing with %d processes",
            num_processes
        )
    else:
        num_processes = 1
        LOGGER.debug(
            "Running optimization in a single process"
        )
    return multiprocessing.Pool(processes=num_processes)


def _compute_log_likelihood_by_cluster(stacked_training_data: np.ndarray,
                                       model: model_state.ModelState) -> list[list[float]]:
    """Compute final log likelihood values for each cluster's data

    For each cluster, build a list containing the log likelihood values for
    that cluster's constituent points.

    This supports computing per-cluster statistics about log likelihood.  This
    data structure is NOT intended as a user-facing index for log likelihood per
    cluster.

    Arguments:
        stacked_training_data (NumPy array): Training data for TICC
        model (TICC model state): Final trained model

    Returns:
        List of list of floats, one per cluster.  Each list contains
        the log likelihood values for its corresponding cluster's points.
    """

    num_data_series = stacked_training_data.shape[1] / model.arguments.window_size

    cluster_log_likelihood = [[] for i in range(model.arguments.num_clusters)]

    for (point_id, cluster_id) in enumerate(model.point_labels):
        if cluster_id == -1:
            # these points did not participate in clustering
            continue
        ll = likelihood.point_log_likelihood(
            stacked_training_data[point_id],
            model.clusters[cluster_id],
            model.arguments.window_size,
            num_data_series
        )
        cluster_log_likelihood[cluster_id].append(ll)

    for next_cluster_array in cluster_log_likelihood:
        if len(next_cluster_array) == 0:
            next_cluster_array.append(0)

    return cluster_log_likelihood

def _package_results(model: model_state.ModelState,
                     user_args: arguments.UserArguments,
                     stacked_training_data: np.ndarray
                     ) -> results.SingleDataSeriesResult:
    """Bundle up trained model and statistics for return to the user

    This is an internal function not meant to be called from outside the
    library.

    Here we pull out all of the model components that we'll need to initialize a
    new model later on.  We also compute and package diagnostic information
    about the clusters so that the user can analyze goodness of fit.  See
    results.SingleDataSeriesResult for details of what's included.

    Arguments:
        model (model_state.ModelState): Trained model from TICC main loop
        user_args (arguments.UserArguments): Initial arguments to TICC
        stacked_training_data (numpy.ndarray): Main data array for training

    Returns:
        results.SingleDataSeriesResult: Container for trained model and
        info statistics
    """


    bayesian_ic = cluster_metrics.bayesian_information_criterion(model)
    calinski_harabasz = cluster_metrics.calinski_harabasz_index(stacked_training_data,
                                                                model)

    num_data_points = stacked_training_data.shape[0]
    labels = [-1] * num_data_points
    for i in range(stacked_training_data.shape[0]):
        labels[i] = model.point_labels[i]


    markov_random_fields = [
        model.clusters[cluster_id].train_inverse
        for cluster_id in range(model.arguments.num_clusters)
    ]

    cluster_log_likelihood = _compute_log_likelihood_by_cluster(
        stacked_training_data, model
    )

    # make this forward-facing
    all_log_likelihood = list(itertools.chain(*cluster_log_likelihood))

    overall_log_likelihood = np.sum(all_log_likelihood)
    overall_log_likelihood_mean = float(np.mean(all_log_likelihood))
    overall_log_likelihood_median = np.median(all_log_likelihood)
    cluster_log_likelihood_mean = np.array([
        np.mean(single_cluster_log_likelihood)
        for single_cluster_log_likelihood in cluster_log_likelihood
    ])
    cluster_log_likelihood_median = np.array([
        np.median(single_cluster_log_likelihood)
        for single_cluster_log_likelihood in cluster_log_likelihood
    ])

    final_model = results.TiccModel(
        window_size=model.arguments.window_size,
        inverse_covariance=markov_random_fields,
        per_cluster_mean=[cluster.stacked_data_mean for cluster in model.clusters]
    )

    return results.SingleDataSeriesResult(
        bayesian_information_criterion=bayesian_ic,
        calinski_harabasz_index=calinski_harabasz,
        label_assignment_cost=model.label_assignment_cost,
        overall_log_likelihood=overall_log_likelihood,
        overall_log_likelihood_mean=overall_log_likelihood_mean,
        overall_log_likelihood_median=overall_log_likelihood_median,
        cluster_log_likelihood_mean=cluster_log_likelihood_mean,
        cluster_log_likelihood_median=cluster_log_likelihood_median,
        all_log_likelihood=all_log_likelihood,
        markov_random_fields=markov_random_fields,
        num_clusters=model.arguments.num_clusters,
        point_labels=labels,
        window_size=model.arguments.window_size,
        trained_model=final_model)
