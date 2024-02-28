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

# Test script: Check initial labels computed by TICC
#
# This test makes sure we still get consistent initial cluster labels
# from the scikit-learn Gaussian mixture model.  A change in the labels
# does not necessarily indicate an error in the code -- it could, for
# example, be caused by a change in the random seed -- but it
# definitely requires notice and investigation.

import os
# Disable Numba JIT compilation
os.environ["NUMBA_DISABLE_JIT"] = "1"

import pytest
import numpy as np

from fast_ticc import cluster_label_assignment


def test_ticc_single_trajectory_initial_labels(load_test_data, num_clusters, random_seed, num_regression):
    single_trajectory_stacked_data = load_test_data("single_trajectory_stacked_data")
    np.random.seed(random_seed)
    computed_labels = cluster_label_assignment.build_initial_clusters(num_clusters,
                                                                      single_trajectory_stacked_data)
    test_dict = {"initial_labels": computed_labels}
    num_regression.check(test_dict)


def test_ticc_multiple_trajectory_initial_labels(load_test_data, num_clusters, random_seed, num_regression):
    stacked_data = load_test_data("multiple_trajectory_stacked_data")
    np.random.seed(random_seed)
    computed_labels = cluster_label_assignment.build_initial_clusters(num_clusters,
                                                                      stacked_data)
    test_dict = {"initial_labels": computed_labels}
    num_regression.check(test_dict)
