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

"""Wrappers to run the whole TICC process.

You'll want to call one of the following two functions:

ticc_labels() - Compute cluster labels for a single data sequence

ticc_joint_labels() - Compute cluster labels for several data sequences
    at the same time

Be aware that the current implementation will not label the last
(window_size - 1) points in the each data sequence.  We're working on
ways to fix that.
"""

from typing import List, Sequence, Union

import numpy as np

from fast_ticc.containers import arguments
from fast_ticc.containers import results
from fast_ticc import data_preparation
from fast_ticc import main_loop


def ticc_labels(data_series: np.ndarray,
                window_size: int = 10,
                num_clusters: int = 5,
                sparsity_weight: Union[float, np.ndarray] = 11e-2,
                label_switching_cost: Union[float, np.ndarray] = 400,
                iteration_limit: int = 1000,
                min_meaningful_covariance: float = 0,
                num_processors: int = 1,
                min_cluster_size: int = 20,
                biased_covariance: bool = False) -> results.SingleDataSeriesResult:
    """Compute TICC labels for a data series.

    This is the front end for running TICC.  This function computes
    labels and inverse covariance matrices for a single set of
    time-series data.

    For implementation reasons, the first and last (window_size-1)/2
    points in each data series will get the label -1 (no label).  If
    you really need labels for those, you have two choices:

    1.  Use the first and last real labels (those with value 0 or higher)
        for those points, or
    2.  Extend your data by at least (window_size-1)/2 points at
        both beginning and end

    About the label-switching cost:

        One of the assumptions underlying TICC is that data series can be
        divided into contiguous chunks that all exhibit similar behavior.
        To encourage this, we penalize the solver whenever it switches labels
        from one data point to the next.  The lower the lambda parameter, the
        less expensive it is to change labels.

        By default, we assume that this parameter (known as beta or "label
        switching cost/penalty") is constant throughout the input data.  However,
        if you are clustering multiple data series at once, you might want to
        supply an array instead.  If the data series are independent, there's
        no reason to expect that the label at the end of data series D would be
        the same as the label at the beginning of data series D+1.  At that
        specific point, beta should be zero.

        Here's how you do that.  Let's suppose that all of your already-stacked
        data series are in the list `data_series`.  Each data series is a
        NumPy array with dimensions (num_data_points, window_size * num_features).

        total_data_points = sum([data.shape[0] for data in data_series])
        label_switching_penalty = (
            np.zeros(shape=(total_data_points,))
            + constant_label_switching_penalty
            )
        end_of_latest_series = 0
        for data in data_series[0:-1]:
            end_of_latest_series += data.shape[0]
            label_switching_penalty[end_of_latest_series] = 0

    About the covariance sparsity parameter:

        In the paper, this parameter (called lambda) is described as a
        constant that gets turned into an NW x NW matrix.  However, like
        the label-switching cost, you are free to supply something more
        complex if you know more about which entries in the Markov random
        field should be more or less expensive.

        If you want to use the default (constant value), pass in a float.
        If you want to roll your own, pass in an NW x NW NumPy array.

    Arguments:
        data_series (NumPy array): Data points for which you want labels.
            Each row in each series is a separate point.  Each column is a
            feature for that data point.

    Keyword Arguments:
        window_size (int): Number of data points to stack together
            for window.  Defaults to 10.
        num_clusters (int): How many different labels to construct.
            Defaults to 5.
        sparsity_weight (float or NumPy array): Regularization term
            to encourage the solver to build sparse covariance matrices.
            Higher values of this parameter bias us toward inverse
            covariance matrices with fewer non-zero entries.  This parameter
            is named lambda in the TICC paper.  That's a reserved word in
            Python, hence the different name.  Defaults to 0.11.  See above
            for special cases.
        label_switching_cost (float): Cost for switching labels
            between one data point and the next.  Defaults to 400.  You may
            also supply a NumPy array with as many entries as data points
            in order to vary this from one point to the next.
        iteration_limit (int): Maximum number of iterations before we
            give up and declare victory.  Defaults to 1000.
        min_meaningful_covariance (float): Any entries in the covariance
            matrix that are smaller in magnitude than this value will be
            zeroed out.  Defaults to 0 (no filtering).
        num_processors (int): Number of processors to use in ADMM solver.
            Leave this at 1 unless you know for certain that your BLAS
            implementation doesn't do parallelism on its own.  See below.
        min_cluster_size (int): How many points to assign when we need
            to reinitialize an empty cluster.  Defaults to 20.
        biased (bool): Whether to compute biased or unbiased covariance
            (i.e. divide by N or N-1).  Defaults to False (unbiased).

    Returns:
        TICC results, including labels, Markov random fields, and information
        about the optimizer results.  See
        fast_ticc.containers.Result for details.

    About num_processors:

    Most of our parallel execution happens automatically.  Modern BLAS
    implementations do their linear algebra work in parallel.  We also
    use Numba to parallelize our log likelihood computation.  You
    do not need to intervene in either of these cases.  In fact, setting
    num_processors > 1 here will likely cause TICC to run more slowly
    because of cache and register contention.

    If you are certain that your BLAS implementation is single-threaded,
    you may be able to get faster execution by setting num_processors > 1.
    """

    params = arguments.UserArguments(
        window_size=window_size,
        num_clusters=num_clusters,
        sparsity_weight=sparsity_weight,
        label_switching_cost=label_switching_cost,
        iteration_limit=iteration_limit,
        min_meaningful_covariance=min_meaningful_covariance,
        num_processors=num_processors,
        min_cluster_size=min_cluster_size,
        biased_covariance=biased_covariance)

    try:
        stacked_data = data_preparation.stack_training_data(
            data_series, window_size)
    except AttributeError as not_a_numpy_array:
        raise TypeError((
            "The data series supplied to ticc_labels must be a 2D NumPy "
            "array.  Did you mean to call ticc_joint_labels instead?"
        )) from not_a_numpy_array

    ticc_result = main_loop.fit_stacked_data(params, stacked_data)
    ticc_result.point_labels = data_preparation.pad_missing_labels(ticc_result.point_labels,
                                                                   window_size)
    return ticc_result


def ticc_joint_labels(data_series: Sequence[np.ndarray],
                      window_size: int = 10,
                      num_clusters: int = 5,
                      sparsity_weight: float = 11e-2,
                      label_switching_cost: Union[float, np.ndarray] = 400,
                      iteration_limit: int = 1000,
                      min_meaningful_covariance: float = 0,
                      num_processors: int = 1,
                      min_cluster_size: int = 20,
                      biased_covariance: bool = False) -> results.MultipleDataSeriesResult:
    """Compute TICC labels over several data series at once.

    This function is almost exactly like regular TICC, but instead of
    computing labels for a single series of data points, it computes
    them over several series at the same time.

    This is the front end for running TICC.  This function computes
    labels and inverse covariance matrices for a single set of
    time-series data.

    For implementation reasons, the last (window_size-1)/2 points in each
    data series will get the label -1 (no label).  If you really need labels
    for those, you have two choices:

    1.  Use the first and last real labels (those with value 0 or higher)
        for those points, or
    2.  Extend your data by at least (window_size-1)/2 points at
        both beginning and end

    About the label-switching cost:

        One of the assumptions underlying TICC is that data series can be
        divided into contiguous chunks that all exhibit similar behavior.
        To encourage this, we penalize the solver whenever it switches labels
        from one data point to the next.  The lower the lambda parameter, the
        less expensive it is to change labels.

        By default, we assume that this parameter (known as beta or "label
        switching cost/penalty") is constant throughout the input data.  However,
        if you are clustering multiple data series at once, you might want to
        supply an array instead.  If the data series are independent, there's
        no reason to expect that the label at the end of data series D would be
        the same as the label at the beginning of data series D+1.  At that
        specific point, beta should be zero.

        Here's how you do that.  Let's suppose that all of your already-stacked
        data series are in the list `data_series`.  Each data series is a
        NumPy array with dimensions (num_data_points, window_size * num_features).

        total_data_points = sum([data.shape[0] for data in data_series])
        label_switching_penalty = (
            np.zeros(shape=(total_data_points,))
            + constant_label_switching_penalty
            )
        end_of_latest_series = 0
        for data in data_series[0:-1]:
            end_of_latest_series += data.shape[0]
            label_switching_penalty[end_of_latest_series] = 0

    About the covariance sparsity parameter:

        In the paper, this parameter (called lambda) is described as a
        constant that gets turned into an NW x NW matrix.  However, like
        the label-switching cost, you are free to supply something more
        complex if you know more about which entries in the Markov random
        field should be more or less expensive.

        If you want to use the default (constant value), pass in a float.
        If you want to roll your own, pass in an NW x NW NumPy array.

    Arguments:
        data_series (sequence of NumPy arrays): Data points for which you
            want labels.  Each element in the sequence is a separate data
            series.  Each row in each data series is a single point.  Each
            column is a variable/sensor within that data point.

    Keyword Arguments:
        window_size (int): Number of data points to stack together
            for window.  Defaults to 10.
        num_clusters (int): How many different labels to construct.
            Defaults to 5.
        sparsity_weight (float or NumPy array): Regularization term
            to encourage the solver to build sparse covariance matrices.
            Higher values of this parameter bias us toward inverse
            covariance matrices with fewer non-zero entries.  This parameter
            is named lambda in the TICC paper.  That's a reserved word in
            Python, hence the different name.  Defaults to 0.11.  See above
            for special cases.
        label_switching_cost (float): Cost for switching labels
            between one data point and the next.  Defaults to 400.  You may
            also supply a NumPy array with as many entries as data points
            in order to vary this from one point to the next.
        iteration_limit (int): Maximum number of iterations before we
            give up and declare victory.  Defaults to 1000.
        min_meaningful_covariance (float): Any entries in the covariance
            matrix that are smaller in magnitude than this value will be
            zeroed out.  Defaults to 0 (no filtering).
        num_proc (int): Number of processors to use in parallel ADMM
            solver.  Defaults to 1.
        min_cluster_size (int): How many points to assign when we need
            to reinitialize an empty cluster.  Defaults to 20.
        biased (bool): Whether to compute biased or unbiased covariance
            (i.e. divide by N or N-1).  Defaults to False (unbiased).

    Returns:
        A list of TICC results, one for each input data series.  Point labels
        will be different across each individual result but all other fields
        will be the same.
    """

    # The user may have provided a forward-only iterable.  We need to
    # traverse it multiple times, so make it a list.
    data_series = list(data_series)

    try:
        combined_data_series = data_preparation.stack_training_data_multiple_series(
            data_series, window_size
        )
    except IndexError as not_a_list_of_numpy_arrays:
        raise TypeError((
            "The data_series argument to ticc_joint_labels must be a list "
            "(or other iterable) of 2D NumPy arrays.  Did you mean to call "
            "ticc_labels instead?"
        )) from not_a_list_of_numpy_arrays

    stacked_data_sizes = [
        len(series) - window_size + 1
        for series in data_series
    ]

    args = arguments.UserArguments(
        window_size=window_size,
        num_clusters=num_clusters,
        sparsity_weight=sparsity_weight,
        label_switching_cost=label_switching_cost,
        iteration_limit=iteration_limit,
        min_meaningful_covariance=min_meaningful_covariance,
        num_processors=num_processors,
        min_cluster_size=min_cluster_size,
        biased_covariance=biased_covariance)

    lsc_template = data_preparation.label_switching_cost_template(
        stacked_data_sizes)
    label_switching_cost = label_switching_cost * lsc_template
    assert label_switching_cost.shape[0] == sum(stacked_data_sizes), \
        "Length of modified label switching cost array is not equal to length" + \
        "of inputted data series."

    # Here we go!  This will take a while.
    master_result = main_loop.fit_stacked_data(args, combined_data_series)

    ticc_multi_result = _split_combined_result(master_result, stacked_data_sizes,
                                               data_series)
    return ticc_multi_result


def _split_combined_result(master_result: results.SingleDataSeriesResult,
                           stacked_data_sizes: List[int],
                           data_series: List[np.ndarray]) -> results.MultipleDataSeriesResult:
    """Split TICC result for multiple combined data series

    Internally, TICC processes multple data series as a single big
    brick of data and generates one very long list of labels.  We
    need to split that back apart into lists corresponding to the
    original data series for the user's consumption.  That happens
    here.

    Arguments:
        master_result (results.SingleDataSeriesResult): TICC
            optimization result for everything at once
        stacked_data_sizes (list): List of (potentially trimmed)
            data series lengths after stacking -- this will help us
            find the label array boundaries
        data_series (list of NumPy array): Original input data series.
            Used for consistency checking.

    Returns:
        results.MultipleDataSeriesResult ready to go back to the user.
    """

    # Disassemble the single master result into results for each input
    # data series
    individual_label_sets = data_preparation.split_joint_labels(master_result.point_labels,
                                                                stacked_data_sizes)
    padded_label_sets = []
    for (i, label_set) in enumerate(individual_label_sets):
        padded_label_sets.append(
            data_preparation.pad_missing_labels(
                label_set, master_result.window_size)
        )
        assert len(padded_label_sets[-1]) == data_series[i].shape[0], \
            "Padded label set not matching length of original data series"

    ticc_multi_result = results.MultipleDataSeriesResult(
        bayesian_information_criterion=master_result.bayesian_information_criterion,
        calinski_harabasz_index=master_result.calinski_harabasz_index,
        label_assignment_cost=master_result.label_assignment_cost,
        point_labels=padded_label_sets,
        markov_random_fields=master_result.markov_random_fields,
        num_clusters=master_result.num_clusters,
        window_size=master_result.window_size,
        all_log_likelihood=master_result.all_log_likelihood,
        overall_log_likelihood=master_result.overall_log_likelihood,
        overall_log_likelihood_mean=master_result.overall_log_likelihood_mean,
        overall_log_likelihood_median=master_result.overall_log_likelihood_median,
        cluster_log_likelihood_mean=master_result.cluster_log_likelihood_mean,
        cluster_log_likelihood_median=master_result.cluster_log_likelihood_median,
    )

    return ticc_multi_result
