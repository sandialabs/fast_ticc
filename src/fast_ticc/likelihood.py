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

"""Compute (log) likelihood for data points in TICC."""

import math

import numpy as np

from fast_ticc.containers import model_state
from fast_ticc import numba_guard


@numba_guard.njit()
def point_log_likelihood_fast(point: np.ndarray,
                              mu_i: np.ndarray,
                              theta_i: np.ndarray,
                              log_det_theta: float,
                              window_size: int,
                              num_data_series: int) -> float:
    """Numba-able computation of per-point log likelihood

    Compute point log likelihood with respect to a cluster
    This is Equation 2 from the TICC paper (page 3, left column, bottom).

    LL(X_t, Theta_i) = -(1/2) (X_t - mu_i)^T Theta_i (X_t - mu_i)
                        + (1/2) log det Theta_i - (nw/2) log (2 pi)

    Here, Theta_i is the sparse inverse covariance matrix for cluster `i`,
    `n` is the number of time series, `w` is the window size, and mu_i is
    the mean of all the data points assigned to cluster `i`.

    This formula looks a little bit different from the likelihood function
    for covariance matrices as presented in the Wikipedia article.  A full
    derivation is in doc/ticc/log_likelihood.tex.

    Note: You can calculate likelihood for any point at all.  It doesn't
          have to be a point currently assigned to the cluster in question
          or even a point from the current data set.

    Arguments:
        point (NumPy array): 1 x (window_size * num_data_series) array. This
            is the data point whose likelihood we will calculate.
        mu_i (NumPy array): Data mean for current point.
        theta_i (NumPy array): Inverse covariance matrix for cluster that
            currently contains point i.
        log_det_theta (float): Log of the determinant of theta_i.  Expensive
            enough to compute that we compute it once and pass it in.
        window_size (int): TICC window size.
        num_data_series (int): Number of data series in input.

    Returns:
        Floating-point log likelihood value.
    """

    nw = window_size * num_data_series
    nw_log_2pi = nw * np.log(2 * math.pi)
    x_t = point
    x_minus_mu = x_t - mu_i

    lle = 0.5 * (log_det_theta
                 - (x_minus_mu.T @ theta_i @ x_minus_mu)
                 - nw_log_2pi)
    return lle


def point_log_likelihood(point: np.ndarray,
                         cluster: model_state.ClusterParameters,
                         window_size: int,
                         num_data_series: int) -> float:
    """Compute point log likelihood with respect to a cluster
    This is Equation 2 from the TICC paper (page 3, left column, bottom).

    LL(X_t, Theta_i) = -(1/2) (X_t - mu_i)^T Theta_i (X_t - mu_i)
                        + (1/2) log det Theta_i - (nw/2) log (2 pi)

    Here, Theta_i is the sparse inverse covariance matrix for cluster `i`,
    `n` is the number of time series, `w` is the window size, and mu_i is
    the mean of all the data points assigned to cluster `i`.

    This formula looks a little bit different from the likelihood function
    for covariance matrices as presented in the Wikipedia article.  A full
    derivation is in doc/ticc/log_likelihood.tex.

    Note: You can calculate likelihood for any point at all.  It doesn't
          have to be a point currently assigned to the cluster in question
          or even a point from the current data set.

    Arguments:
        point (NumPy array): 1 x (window_size * num_data_series) array. This
            is the data point whose likelihood we will calculate.
        cluster (fast_ticc.containers.ClusterParameters):
            Parameters for the cluster in question.  We need the cluster's
            Markov random field (AKA precision matrix AKA inverse covariance
            matrix) and the mean of all data points currently assigned
            to that cluster.
        window_size (int): TICC window size.
        num_data_series (int): Number of data series in input.

    Returns:
        Floating-point log likelihood value.
    """
    mu_i = cluster.stacked_data_mean
    theta_i = cluster.inverse_covariance
    log_det_theta = cluster.log_determinant

    return point_log_likelihood_fast(point,
                                     mu_i, theta_i, log_det_theta,
                                     window_size, num_data_series)


@numba_guard.njit(parallel=True)
def all_points_all_clusters_log_likelihood_fast(window_size: int,
                                                num_clusters: int,
                                                mus: np.ndarray,
                                                thetas: np.ndarray,
                                                log_det_thetas: np.ndarray,
                                                stacked_training_data: np.ndarray) -> np.ndarray:
    """Numba-friendly computation of log likelihood

    Compute the log likelihood of each point with respect to each cluster.
    This function is separate from all_points_all_clusters_log_likelihood so
    that Numba can parallelize the loop.

    Arguments:
       window_size (int): How many points are in a data window
       num_clusters (int): How many clusters TICC is fitting
       mus (NumPy array): Per-cluster data means
       thetas (NumPy array): Per-cluster inverse covariance matrices
       log_det_thetas (NumPy array): Per-cluster log(det(theta))
       stacked_training_data (NumPy array): Data for which we're fitting
           covariance matrices

    Returns:
        (num_points x num_clusters) array of log likelihood values

    TODO: Improve this to deal with incomplete data windows.  Numba
          can't handle masking per its documentation.
    """
    num_input_points = len(stacked_training_data)
    num_data_series = int(stacked_training_data.shape[1] / window_size)

    result = np.zeros(shape=(num_input_points, num_clusters), dtype=np.float64)

    # Pylint does not recognize that prange is in fact an iterable.
    # pylint: disable-next=not-an-iterable
    for point in numba_guard.prange(num_input_points):
        for cluster in range(num_clusters):
            result[point, cluster] = point_log_likelihood_fast(
                stacked_training_data[point, :],
                mus[cluster], thetas[cluster], log_det_thetas[cluster],
                window_size,
                num_data_series
            )
    return result


def all_points_all_clusters_log_likelihood(model: model_state.ModelState,
                                           stacked_training_data: np.ndarray) -> np.ndarray:
    """Compute log likelihood for each point wrt each cluster

    Arguments:
        model (ticc.containers.ModelState): Current trained-up model.
            We need the computed covariance matrix for each cluster and
            the window size.
        stacked_training_data (NumPy array): Data for which we're fitting
            a model.
    Returns:
        New array containing log likelihood values, one row for
        each data point, one column for each cluster

    """

    for cluster in range(model.arguments.num_clusters):
        inverse_covariance = model.clusters[cluster].train_inverse
        model.clusters[cluster].inverse_covariance = inverse_covariance
        model.clusters[cluster].log_determinant = np.log(np.linalg.det(inverse_covariance))

    #Unwrap arguments from custom objects to let numba optimize them
    window_size = model.arguments.window_size
    num_clusters = model.arguments.num_clusters
    mus = np.asarray([x.stacked_data_mean for x in model.clusters])
    thetas = np.asarray([x.inverse_covariance for x in model.clusters])
    log_det_thetas = np.asarray([x.log_determinant for x in model.clusters])

    return all_points_all_clusters_log_likelihood_fast(window_size,
                                                       num_clusters,
                                                       mus,
                                                       thetas,
                                                       log_det_thetas,
                                                       stacked_training_data)
