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

"""Methods to stack data sets for use with TICC and unstack the results"""

import itertools
from typing import List

import numpy as np


def stack_training_data(data: np.ndarray, window_size: int) -> np.ndarray:
    """Stack data points into time windows.

    This method will take the original data (one observation per row, one
    data series per column) and stack it into the expanded matrix that
    TICC will use for training.  If you run TICC using fit(), this will be
    done for you.  If you run TICC using fit_stacked_data(), you must
    call this function yourself.

    Note that this function trims off the last few data points where we
    do not have enough input data to fill the time window.  The TICC
    paper says that we just compute on smaller windows in these cases,
    but this does not match the authors' reference implementation.

    Arguments:
        data (NumPy array): Input data: one data point per row, one
            data series per column.
        window_size (int): How many data points to put in each window

    Returns:
        If input array has shape (rows, columns), the result has shape
        (rows - window_size + 1, columns * window_size).
    """

    num_data_points = data.shape[0]
    num_full_windows = num_data_points - window_size + 1
    num_features = data.shape[1]
    stacked_training_data = np.zeros([num_full_windows,
                                      num_features * window_size])

    for i in range(num_full_windows):
        for j in range(window_size):
            start_column = j * num_features
            end_column = (j+1) * num_features
            stacked_training_data[i, start_column:end_column] = data[i+j, :]

    return stacked_training_data


def stack_training_data_multiple_series(all_series: List[np.ndarray],
                                        window_size: int) -> np.ndarray:
    """Stack data points from multiple input series into time windows.

    This method takes a list of data series with the same sensors (variables
    per point) but potentially a different number of points and stacks
    them into the windows used by TICC.

    Note that for implementation reasons, the last window_size - 1
    points are trimmed from each input data series.

    If you call this function you will probably also want to use
    label_switching_cost_template() to build your label switching
    cost array.

    Arguments:
        data (NumPy array): Input data: one data point per row, one
            data series per column.
        window_size (int): How many data points to put in each window

    Returns:
        Single stacked data array with input points from all samples
    """

    # Stack all the input data sets separately
    stacked_data = [
        stack_training_data(data, window_size) for data in all_series
    ]

    # Concatenate the individual stacks into our one big brick.
    combined_data_series = np.vstack(stacked_data)
    return combined_data_series


def label_switching_cost_template(stacked_series_lengths: List[int]) -> np.ndarray:
    """Build mask for zeroing out label switching cost between time series

    The label switching cost between data points in a sequence
    does not apply when switching from the end of one sequence
    to the beginning of the next.  This function builds a mask
    that has zeros at the places where the cost array should
    be zero and ones everywhere else.

    Arguments:
        stacked_series_lengths (list of int): Lengths of stacked
            subsequence data points.  Stacking trims off the last
            (window_size - 1) points from each series.

    Returns:
        1D NumPy array with as many elements as data points in
        all the stacked data series combined
    """

    num_points_total = sum(stacked_series_lengths)
    endpoints = list(itertools.accumulate(stacked_series_lengths))
    # Drop the last one -- the last sequence ends at the end of
    # the stacked data array and there's no label switching cost
    # to zero out after that
    endpoints.pop()

    template = np.ones(shape=(num_points_total,))
    template[endpoints] = 0
    return template


def pad_missing_labels(original_labels: List[int], window_size: int) -> List[int]:
    """Add 'point not labeled' entries to the end of a list

    For implementation reasons, TICC doesn't compute labels for
    data points with incomplete windows.  These are the first and
    last (window_size / 2) entries in a data series.  We assign
    the label -1 to these points in order to pass back some label
    for every input point.

    Arguments:
        original_labels (list of int): Labels that came from TICC
        window_size: How big the window size was during stacking

    Returns:
        Label array with (window_size - 1)/2 values at the
        beginning and end
    """

    front_length = int((window_size - 1)/2)
    back_length = (window_size - 1) - front_length
    front_labels = [-1] * front_length
    back_labels = [-1] * back_length

    final_labels = front_labels + original_labels + back_labels
    assert len(final_labels) == len(original_labels) + window_size - 1
    return final_labels


def split_joint_labels(joint_labels: List[int],
                       stacked_series_lengths: List[int]) -> List[List[int]]:
    """Break out TICC labels for each input series

    When we compute labels for several data series jointly, we get
    back a single list of labels corresponding to the concatenated
    input data.  This function undoes that concatenation.

    Arguments:
        joint_labels (list of int): Concatenated block of labels
            from TICC
        stacked_series_lengths (list of int): Lengths of the
            individual data series that went into the concatenated
            block

    Returns:
        Original list of labels partitioned into chunks corresponding
        to each input data series
    """

    assert len(joint_labels) == sum(stacked_series_lengths), \
        "Joint labels size must equate to size of all individual data series" + \
        " that have been concatenated."
    sequence_end_indices = list(itertools.accumulate(stacked_series_lengths))

    label_lists = []

    for i in range(len(stacked_series_lengths)):
        # Start and end indices for each original data series in the
        # list of joint labels
        if i == 0:
            start = 0
        else:
            start = sequence_end_indices[i-1]
        end = sequence_end_indices[i]

        label_lists.append(joint_labels[start:end])

    return label_lists
