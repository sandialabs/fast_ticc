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

"""Containers for TICC parameters, model state, and result"""


import dataclasses
from typing import List
import numpy as np


@dataclasses.dataclass
class SingleDataSeriesResult:
    """Labels computed for a single data series with TICC

    The main results from TICC are in ``point_labels`` and
    ``markov_random_fields``.  The `num_clusters` and `window_size` fields
    are diagnostic information.  The other fields are useful for
    evaluating how well the model fits the data.

    You may notice that there is no total log likelihood for each cluster.
    This is deliberate.  Since the overall log likelihood is the sum of the
    log likelihood at each point, it is inextricably dependent on the
    number of points.  We expect that to change from one run to the next.
    However, likelihood values are only informative when compared across
    multiple runs.  As a result, the total log likelihood for each cluster
    doesn't do us any good, so we don't compute it.

    Attributes:
        bayesian_information_criterion (float): The Bayesian information
            criterion is a "quality" score that attempts to balance how
            well a model fits the data with the model's complexity.  Please
            see Wikipedia or a good statistics text for more information.
        calinski_harabasz_index (float): This number talks about the
            "tightness" of a clustering in terms of the ratio of
            between-cluster separation and within-cluster dispersion.
        label_assignment_cost (float): Optimization result for the
            cluster labels.  This is the value of Equation 3 in the
            TICC paper.
        cluster_log_likelihood_mean (array of float): Arithmetic mean of
            per-point log likelihood values within each separate cluster.
            List indices are cluster IDs.
        cluster_log_likelihood_median (array of float): Median of per-point
            log likelihood values within each separate cluster.  List indices
            are cluster IDs.
        overall_log_likelihood (float): Log likelihood of entire training data
            set with respect to the final trained model.
        overall_log_likelihood_mean (float): Arithmetic mean of per-point log
            likelihood values with respect to the final trained model.
        overall_log_likelihood_median (float): Median of per-point log likelihood
            values for points in training data with respect to final trained
            model.
        markov_random_fields (list -> NW x NW NumPy array): Values for
            the matrices containing the Markov random fields that define each
            cluster.  This list is indexed by cluster number.  These
            can be interpreted as precision matrices or as edge weights
            in a Markov random field.
        num_clusters (int): Number of clusters constructed by TICC.
        point_labels (list of int): Computed label for each symbol in the input.
            Labels 0 through ``num_clusters - 1`` are valid IDs.  Label -1
            indicates that a point from the input was not labeled by TICC.
            This happens with the last ``window_size - 1`` points.
        window_size (int): Length of each subsequence (measured in
            time steps) used in training.
    """

    bayesian_information_criterion: float
    calinski_harabasz_index: float
    label_assignment_cost: float
    all_log_likelihood: List[float]
    overall_log_likelihood: float
    overall_log_likelihood_mean: float
    overall_log_likelihood_median: float
    cluster_log_likelihood_mean: List[float]
    cluster_log_likelihood_median: List[float]
    markov_random_fields: List[np.ndarray]
    num_clusters: int
    point_labels: List[int]
    window_size: int


@dataclasses.dataclass
class MultipleDataSeriesResult:
    """Labels computed for multiple data series with TICC

    The main results from TICC are in ``point_labels`` and
    ``markov_random_fields``.  The `num_clusters` and `window_size`
    fields are diagnostic information.  The other fields are useful for
    evaluating how well the model fits the data.

    The Bayesian information criterion and label assignment cost
    are relative to the data set as a whole.  We do not yet provide
    goodness-of-fit information for individual data series.

    You may notice that there is no total log likelihood for each cluster.
    This is deliberate.  Since the overall log likelihood is the sum of the
    log likelihood at each point, it is inextricably dependent on the
    number of points. We expect that to change from one run to the next.
    However, likelihood values are only informative when compared across
    multiple runs.  As a result, the total log likelihood for each cluster
    doesn't do us any good, so we don't compute it.

    Also, at present, per-cluster log likelihood values are computed
    using points from all input data series.  We'll break this out
    further (by cluster and data series) if someone asks for it.

    Attributes:
        bayesian_information_criterion (float): The Bayesian information
            criterion is a "quality" score that attempts to balance how
            well a model fits the data with the model's complexity.  Please
            see Wikipedia or a good statistics text for more information.
        calinski_harabasz_index (float): This number talks about the
            "tightness" of a clustering in terms of the ratio of
            between-cluster separation and within-cluster dispersion.
        label_assignment_cost (float): Optimization result for the
            cluster labels.  This is the value of Equation 3 in the
            TICC paper.
        markov_random_fields (list -> NW x NW NumPy array): Values for
            the matrices containing the Markov random fields that define each
            cluster.  This list is indexed by cluster number.  These
            can be interpreted as precision matrices or as edge weights
            in a Markov random field.  NOTE: The Markov random fields are
            shared across all the data series.
        num_clusters (int): Number of clusters constructed by TICC.
        cluster_log_likelihood_mean (array of float): Arithmetic mean of
            per-point log likelihood values within each separate cluster.
            List indices are cluster IDs.
        cluster_log_likelihood_median (array of float): Median of per-point
            log likelihood values within each separate cluster.  List indices
            are cluster IDs.
        overall_log_likelihood (float): Log likelihood of entire training data
            set with respect to the final trained model.
        overall_log_likelihood_mean (float): Arithmetic mean of per-point log
            likelihood values with respect to the final trained model.
        overall_log_likelihood_median (float): Median of per-point log likelihood
            values for points in training data with respect to final trained
            model.
        point_labels (list of list of int): Computed cluster ID for each
            data point (symbol) in the input.  Indexed first by data series,
            then by point ID within each series.  Labels 0 through
            ``num_clusters - 1`` are valid IDs.  Label -1 indicates that a
            point from the input was not labeled by TICC. This happens with
            the last ``window_size - 1`` points in each data series.
        window_size (int): Length of each subsequence (measured in
            time steps) used in training.
    """

    bayesian_information_criterion: float
    calinski_harabasz_index: float
    label_assignment_cost: float
    point_labels: List[List[int]]
    markov_random_fields: List[np.ndarray]
    overall_log_likelihood: float
    all_log_likelihood: List[float]
    overall_log_likelihood_mean: float
    overall_log_likelihood_median: float
    cluster_log_likelihood_mean: List[float]
    cluster_log_likelihood_median: List[float]
    num_clusters: int
    window_size: int



@dataclasses.dataclass
class ADMMResult:
    """Optimization results from ADMM in graphical lasso

    This is an internal class used for passing results back from
    ADMM to the cluster labeling code.  You will not need to instantiate
    it yourself.

    Attributes:
        theta (NumPy array): Optimized Theta array in compressed upper
            triangular form.  Use
            fast_ticc.matrix_compression.reinflate_matrix()
            to restore to full square matrix.
    """

    theta: np.ndarray
