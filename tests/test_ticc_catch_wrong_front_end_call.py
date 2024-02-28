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


# This script tests the TICC front ends by calling them with the
# wrong input data (multiple data series for ticc_labels, a single
# data series for ticc_joint_labels) and watching to make sure the
# proper exception gets generated.


import os
# Disable Numba JIT compilation
os.environ["NUMBA_DISABLE_JIT"] = "1"

import pytest

import numpy as np

from fast_ticc import front_end

# We get the load_test_data fixture from tests/fixtures/ via Pytest's
# conftest.py.

def test_ticc_single_series_front_end_multiple_inputs(load_test_data):
    single_data_series = load_test_data("single_trajectory_features")
    with pytest.raises(TypeError):
        dummy_result = front_end.ticc_joint_labels(single_data_series)


def test_ticc_multiple_series_front_end_single_input(load_test_data):
    multiple_data_series = load_test_data("multiple_trajectory_features")
    with pytest.raises(TypeError):
        dummy_result = front_end.ticc_labels(multiple_data_series)

