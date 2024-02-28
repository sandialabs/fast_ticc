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

"""Measures that tell us about the quality of TICC results

We have two measures we can use to evaluate results: the Bayesian
information criterion and Calinski-Harabasz index.

For the Bayesian information criterion, lower values are generally better.
For the Calinski-Harabasz index, higher values are generally better.

A full description of either measure is beyond the scope of this
docstring.  Please consult the literature -- Wikipedia is a good
place to start.
"""

import numpy as np

from fast_ticc.containers import model_state

def bayesian_information_criterion(model: model_state.ModelState) -> float:
    """Compute Bayesian information criterion

    TODO: Understand this code, compare it to the paper and the textbook
    definition of BIC, and make sure it's right.

    Arguments:
        model (TICC ModelState): Fully trained model

    Returns:
        Floating-point value for the Bayesian information criterion
    """

    mod_lle = 0

    threshold = 2e-5
    cluster_params = {}
    non_zero_params = 0
    for cluster_id in range(model.arguments.num_clusters):
        trained_inverse_covariance = model.clusters[cluster_id].train_inverse
        empirical_covariance = model.clusters[cluster_id].empirical_covariance

        log_det_inverse = np.log(np.linalg.det(trained_inverse_covariance))
        trace_dot = np.trace(np.dot(trained_inverse_covariance,
                                    empirical_covariance))
        mod_lle += log_det_inverse - trace_dot
        cluster_params[cluster_id] = np.sum(np.abs(trained_inverse_covariance) > threshold)

    last_point_label = -1
    for point_label in model.point_labels:
        if point_label != last_point_label:
            non_zero_params += cluster_params[point_label]
            last_point_label = point_label

    num_data_points = len(model.point_labels)
    return non_zero_params * np.log(num_data_points) - 2*mod_lle


def calinski_harabasz_index(stacked_training_data: np.ndarray,
                            model: model_state.ModelState)-> float:
    """Compute Calinski-Harabasz index for trained model

    The Calinski-Harabasz index is the ratio of between-cluster
    separation to within-cluster dispersion.  It's used to evaluate
    the quality of a clustering result.

    Arguments:
        stacked_training_data (numpy.ndarray): the whole of the dataset
            against whichthe metric is calculated
        model (fast_ticc.containers.model_state.ModelState): Trained
            TICC model.  Cluster parameters and assignments are used to
            calculate within-cluster dispersion.

    Returns:
        Floating point value for the Calinski-Harabasz index
    """

    numerator = 0
    denominator = 0
    global_center = np.mean(stacked_training_data)

    for cluster in model.clusters:
        recentered_data_mean = (cluster.stacked_data_mean - global_center).reshape(-1, 1)
        group_dispersion = cluster.size * (recentered_data_mean @ recentered_data_mean.T)
        numerator += group_dispersion

        for point_id in cluster.member_points:
            recentered_point = (stacked_training_data[point_id] -
                                cluster.stacked_data_mean).reshape(-1, 1)
            within_cluster_dispersion = recentered_point @ recentered_point.T
            denominator += within_cluster_dispersion

    dispersion_ratio = np.trace(numerator) / np.trace(denominator)
    degree_of_freedom_normalization = (
        (len(stacked_training_data) - len(model.clusters)) /
         (len(model.clusters) - 1)
    )

    return dispersion_ratio * degree_of_freedom_normalization
