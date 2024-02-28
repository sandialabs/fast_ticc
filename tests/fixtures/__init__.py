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

# PyTest fixtures for TICC tests

import os.path
import pickle
import pytest

from typing import Any, List

import numpy as np

def test_data_filename(filename: str) -> str:
    # We are in the directory tests/fixtures.  We want to get from here
    # to tests/test_data.
    here = os.path.dirname(__file__)
    data_dir = os.path.abspath(
        os.path.join(here, "..", "test_data")
    )
    return os.path.join(data_dir, filename)

def test_data(basename: str) -> Any:
    full_filename = f"{basename}.pkl"
    with open(test_data_filename(full_filename), "rb") as infile:
        return pickle.load(infile)
    
@pytest.fixture(scope="module")
def random_seed():
    """This is the random seed we will use for all of our test computation"""
    return 12345

@pytest.fixture(scope="module")
def window_size() -> int:
    return 10

@pytest.fixture(scope="module")
def num_clusters() -> int:
    return 5

@pytest.fixture(scope="module")
def label_switching_cost() -> float:
    return 200

@pytest.fixture(scope="module")
def min_meaningful_covariance() -> float:
    # the value of 1e-4 is causing the TICC solver to get stuck
    # in a loop of repopulating clusters
    #return 1e-4
    return 0

# Below here we have fixtures that contain input data

@pytest.fixture(scope="module")
def load_test_data():
    def retrieve_test_data_file(basename):
        return test_data(basename)
    return retrieve_test_data_file


# def expected_single_trajectory_stacked_data() -> np.ndarray:
#     return test_data("single_trajectory_stacked_data")

# @pytest.fixture
# def expected_multiple_trajectory_stacked_data() -> np.ndarray:
#     return test_data("multiple_trajectory_stacked_data")

# @pytest.fixture
# def single_trajectory_features() -> np.ndarray:
#     return test_data("single_trajectory_features")

# @pytest.fixture
# def multiple_trajectory_features() -> List[np.ndarray]:
#     return test_data("multiple_trajectory_features")
