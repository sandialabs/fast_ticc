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

"""Containers for TICC parameters

UserArguments is for user-specified parameters that run the whole process.
ADMMArguments is for the parameters specific to ADMM optimization.
"""

import dataclasses
import sys
from typing import Callable, Optional, TextIO, Union

import numpy as np


@dataclasses.dataclass
class UserArguments:
    """User-specified parameters to the TICC algorithm

    You don't need to instantiate this class directly.  It's
    for internal bookkeeping.
    """
    # How large is the optimization penalty for each non-zero element
    # in a covariance matrix?
    sparsity_weight: Union[float, np.ndarray]
    # How many iterations will we put up with before we give up and
    # declare victory?
    iteration_limit: int
    # How large is the optimization penalty for switching labels
    # between one point and its successor?
    label_switching_cost: Union[float, np.ndarray]
    # How small will we allow a cluster to get before we pull points
    # from other clusters to repopulate it?
    min_cluster_size: int
    # How large must an entry in the covariance matrix be to be
    # kept?  Values smaller than this will be zeroed out.
    min_meaningful_covariance: float
    # How many clusters will we build?
    num_clusters: int
    # How many processors can we use for covariance optimization
    num_processors: int
    # How many data points are in the window TICC scans?
    window_size: int
    # Use biased or unbiased covariance estimate?
    biased_covariance: bool

    def print(self, out: Optional[TextIO]=None) -> None:
        """Print parameters to some convenient output

        This prints a conveniently-formatted list of parameters
        to a user-specified output stream.

        No positional arguments.

        Keyword arguments:
            out (text IO sink): File-like object for output.
                Use this argument if you want to write parameters
                to something other than sys.stdout (the default).

        Returns:
            None
        """

        if out is None:
            out = sys.stdout

        def my_log(description, value):
            print(f"    {description}: {value}", file=out)

        print("TICC arguments:", file=out)
        my_log("Sparsity weight parameter (lambda)",
               self.sparsity_weight)
        my_log("Iteration limit",
               self.iteration_limit)
        my_log("Label switching cost (beta)",
               self.label_switching_cost)
        my_log("Minimum cluster size",
               self.min_cluster_size)
        my_log("Minimum meaningful covariance",
               self.min_meaningful_covariance)
        my_log("Number of clusters",
               self.num_clusters)
        my_log("Number of tasks for parallel optimization",
               self.num_processors)
        my_log("Use biased covariance estimator",
               self.biased_covariance)
        my_log("TICC window size",
               self.window_size)


    def shallow_copy(self) -> "UserArguments":
        """Return a new copy of the user arguments

        Although this is a shallow copy, all the members are scalars,
        so there's no upstream modification possible.

        Returns:
            New UserArguments instance with same contents
        """
        return UserArguments(
            sparsity_weight=self.sparsity_weight,
            iteration_limit=self.iteration_limit,
            label_switching_cost=self.label_switching_cost,
            min_cluster_size=self.min_cluster_size,
            min_meaningful_covariance=self.min_meaningful_covariance,
            num_clusters=self.num_clusters,
            num_processors=self.num_processors,
            biased_covariance=self.biased_covariance,
            window_size=self.window_size)


    def deep_copy(self):
        """Return a new copy of the user arguments

        Returns:
            New UserArguments instance with same contents
        """
        # Since all the arguments are scalars, there's no difference
        # between a shallow and a deep copy.
        return self.shallow_copy()


@dataclasses.dataclass
class ADMMArguments:
    """Convenience container for ADMM arguments

    Like TICCArguments, you don't need to instantiate this class directly.
    It's for internal bookkeeping.

    Attributes:
        window_size (int): TICC window length
        num_data_series (int): Number of variables/sensors in input data
        rho (float): ADMM feasibility/optimization parameter
        rho_update (function): Function to compute new tradeoff parameter
        sparsity (float or NumPy array): Sparsity regularization parameter
            ('lambda' in the paper)
        absolute_tolerance (float): Absolute convergence threshold
        relative_tolerance (float): Relative convergence threshold
        max_iterations (int): Maximum number of iterations for solver
        verbose (bool): Whether to log copious status information
    """

    window_size: int
    num_data_series: int
    rho: float
    rho_update: Callable[[float, float, float, float, float], float]
    sparsity_weight: Union[float, np.ndarray]
    absolute_tolerance: float
    relative_tolerance: float
    max_iterations: int
    verbose: bool

    def shallow_copy(self) -> "ADMMArguments":
        """Return a new copy of the ADMM arguments

        Although this is a shallow copy, all the members are scalars or
        pointers to opaque objects, so there's no upstream modification
        possible.

        Returns:
            New ADMMArguments instance with same contents
        """
        return ADMMArguments(
            window_size=self.window_size,
            num_data_series=self.num_data_series,
            rho=self.rho,
            rho_update=self.rho_update,
            sparsity_weight=self.sparsity_weight,
            absolute_tolerance=self.absolute_tolerance,
            relative_tolerance=self.relative_tolerance,
            max_iterations=self.max_iterations,
            verbose=self.verbose
        )

    def deep_copy(self) -> "ADMMArguments":
        """Return a new copy of the ADMM arguments

        Since all the members of this class are scalars or pointers to opaque
        objects, this is the same as a shallow copy.

        Returns:
            New ADMMArguments instance with same contents
        """
        # Since all the arguments are scalars, there's no difference
        # between a shallow and a deep copy.
        return self.shallow_copy()
