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

# Test script: make sure data stacking works as expected

import os
# Disable Numba JIT compilation
os.environ["NUMBA_DISABLE_JIT"] = "1"

import pytest

import numpy as np

from fast_ticc import data_preparation




# @pytest.fixture
# def single_trajectory_stacked_data(single_trajectory_features, ticc_window_size):
#     stacked_data = data_preparation.stack_training_data(single_trajectory_features, ticc_window_size)
#     return stacked_data


# @pytest.fixture
# def multiple_trajectory_stacked_data(multiple_trajectory_features, ticc_window_size):
#     stacked_data = data_preparation.stack_training_data_multiple_series(multiple_trajectory_features, ticc_window_size)
#     return stacked_data


def test_ticc_data_stacking_single(load_test_data, window_size, ndarrays_regression):
    single_trajectory_features = load_test_data("single_trajectory_features")
    stacked_data = data_preparation.stack_training_data(single_trajectory_features, window_size)

    result_dict = {"stacked_data": stacked_data}
    ndarrays_regression.check(result_dict)


# def test_ticc_data_stacking_multiple(multiple_trajectory_stacked_data, oracle):
#     oracle_gt = oracle.get("multiple_trajectory_stacked_data")
#     np.testing.assert_array_almost_equal(oracle_gt, multiple_trajectory_stacked_data)