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

"""Callable front end for ADMM optimization

This module defines the ``admm_optimize_theta`` function that drives the
optimization of the matrices defining each cluster.
"""

from typing import Optional, Union

import numpy as np

from fast_ticc.admm import solver
from fast_ticc.containers import arguments
from fast_ticc.containers import results
from fast_ticc.ticc_types import RhoUpdateFunction

def admm_optimize_theta(empirical_covariance: np.ndarray,
                        sparsity_weight: Union[float, np.ndarray],
                        window_size: int,
                        num_data_series: int,
                        rho: float=1,
                        rho_update: Optional[RhoUpdateFunction] = None,
                        max_iterations: int=1000,
                        absolute_tolerance: float=1e-6,
                        relative_tolerance: float=1e-6,
                        verbose: bool=False
                        ) -> results.ADMMResult:
    """Run the ADMM solver to optimize a precision matrix

    This is the mechanism by which we find the best value for the inverse
    precision matrix (Markov random field) that characterizes one of the
    clusters in TICC.  Section 4.2 in the paper is mostly devoted to how
    we use ADMM to do so.

    Arguments:
        empirical_covariance (NumPy arrray): Empirical covariance matrix for
            points assigned to this cluster.  This is S in the early
            part of section 4.2.
        sparsity_weight (float or NumPy array): Regularization penalty to
            encourage the solver to find covariance matrices with fewer
            non-zero terms.  This is called lambda in the paper.
        window_size (int): Number of data points stacked to form a
            temporal subsequence.
        num_data_series (int): Number of values/sensors in the input time
            series.

    Keyword Arguments:
        rho (float): ADMM penalty parameter.  This controls the tradeoff
            between minimizing the objective function and ensuring a
            feasible solution.  See Boyd et al. 2010 for suggestions on
            how to choose a rho value.  Defaults to 1.
        rho_update (callable): Function to compute a new value of rho
            in much the same way an adaptive-step-size solver chooses the
            step size for its next iteration.  Arguments are (current_rho,
            primal_residual, primal_epsilon, dual_residual, dual_epsilon)
            and the function must return a new value for rho.  The default
            is to leave rho unchanged between iterations.
        max_iterations (int): Maximum number of iterations before we give
            up and declare victory.  Defaults to 1000.
        absolute_tolerance (float): Convergence parameter.  Once the
            residuals from one step to the next change by less than
            this amount in absolute terms and relative_tolerance in
            relative terms, we stop and return results.  Defaults
            to 1e-6.
        relative_tolerance (float): Convergence parameter.  Once the
            residuals from one step to the next change by less than
            this amount in relative terms and absolute_tolerance in
            absolute terms, we stop and return results.  Defaults
            to 1e-6.
        verbose (bool): If true, the solver will emit lots of debug log
            messages about what it's doing.  If you need to use this,
            consider running with just one process so the messages from
            different ADMM solvers running simultaneously don't get mixed
            up.  Defaults to False.

    Returns:
        ADMM result container
        (fast_ticc.containers.results.ADMMResult) with new estimate
        for Theta matrix and information about optimizer state
    """

    args = arguments.ADMMArguments(window_size=window_size,
                                    num_data_series=num_data_series,
                                    rho=rho,
                                    rho_update=rho_update,
                                    sparsity_weight=sparsity_weight,
                                    absolute_tolerance=absolute_tolerance,
                                    relative_tolerance=relative_tolerance,
                                    max_iterations=max_iterations,
                                    verbose=verbose)

    compressed_theta = solver.run_admm_optimization(args, empirical_covariance)

    return results.ADMMResult(theta=compressed_theta)
