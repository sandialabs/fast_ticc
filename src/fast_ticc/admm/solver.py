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

"""Solver for graphical lasso problem in TICC paper.

This module contains an implementation of the Alternating
Direction Method of Multipliers (ADMM) specialized for the
structure of the optimization in TICC.  It implements the
process on pages 5 and 6 of the TICC paper.

The solver itself is not multithreaded.  We rely on the
caller to run instances in parallel if desired.  Since
optimization happens on a per-cluster basis, we can
take advantage of up to one core per cluster.

If your NumPy installation relies on a back end that does
its own multiprocessing such as the Intel Math Kernel
Libraries or Apple's Accelerate.framework, you may see
benefits from that.

If you're looking for a general-purpose ADMM solver that
you can use with your own problems, check out CVXPY at
https://www.cvxpy.org/index.html.

"""

import logging
import math
from typing import Tuple, Union

import numpy as np

from fast_ticc import matrix_compression
from fast_ticc.admm import unique_values
from fast_ticc.containers import arguments


LOGGER = logging.getLogger(__name__)


def run_admm_optimization(args: arguments.ADMMArguments,
                          empirical_covariance: np.ndarray) -> np.ndarray:
    """Run main loop for ADMM solver

    This is the driver function that makes everything happen.  We
    assume that the arguments to the solver itself are passed in
    when the class is instantiated.

    Arguments:
        args (fast_ticc.containers.arguments.ADMMArguments): Control
            parameters for optimization
        empirical_covariance (numpy.ndarray): Computed covariance matrix for
            data points in the cluster under optimization

    Returns:
        Optimized inverse covariance matrix in compressed upper
        triangular form
    """

    # This is the quantity that appears in the paper as NW.
    matrix_size = args.window_size * args.num_data_series
    compressed_array_size = int((matrix_size * (matrix_size + 1)) / 2)

    # These are the three state arrays.  The single-character variable
    # names have meaning in the context of the ADMM solver and the
    # notation in the paper.
    x = np.zeros(compressed_array_size)
    z = np.zeros(compressed_array_size)
    u = np.zeros(compressed_array_size)

    z_old = None
    for iteration in range(args.max_iterations):
        z_old = z
        x = admm_update_x(args, u, z, empirical_covariance)
        z = admm_update_z(args, u, x)
        u = admm_update_u(u, x, z)

        if iteration > 0:
            (converged,
             residual_primal, tolerance_primal,
             residual_dual, tolerance_dual) = check_convergence(args, u, x, z,
                                                                z_old)
            if converged:
                LOGGER.debug("ADMM converged after %d iteration(s).", iteration)
                break
            if args.rho_update:
                new_rho = args.rho_update(args.rho,
                                          residual_primal, tolerance_primal,
                                          residual_dual, tolerance_dual)
                scale = args.rho / new_rho
                args.rho = new_rho
                u = scale * u

        if args.verbose:
            # Debugging information prints current iteration #
            LOGGER.debug("ADMM: Finished iteration %d", iteration)

    return x


def admm_update_u(u: np.ndarray, x: np.ndarray, z: np.ndarray):
    """ADMM U update

    This is the scaled dual variable step.

    Arguments:
        u (NumPy array): Latest estimate for U
        x (NumPy array): Latest estimate for X / Theta
        z (NumPy array): Latest estimate for consensus variable Z

    Returns:
        New estimate for U matrix
    """
    return u + x - z


def admm_update_x(args: arguments.ADMMArguments,
                  u: np.ndarray,
                  z: np.ndarray,
                  empirical_covariance: np.ndarray) -> np.ndarray:
    """X update step for ADMM: compute new Theta matrix

    Arguments:
        args (containers.ADMMArguments): bundled control parameters
        z (NumPy array): Latest estimate for Z
        u (NumPy array): Latest estimate for U
        empirical_covariance (NumPy array): Covariance estimate for data
            points assigned to the MRF being optimized

    Returns:
        New estimate for X/Theta matrix
    """
    z_minus_u_compressed = z - u
    z_minus_u = matrix_compression.reinflate_matrix(z_minus_u_compressed)
    x_update = x_update_prox(empirical_covariance, z_minus_u, args.rho)
    return x_update


def admm_update_z(args: arguments.ADMMArguments,
                  u: np.ndarray,
                  x: np.ndarray) -> np.ndarray:
    """Z update step for ADMM: update consensus variable

    This is Equation (7) from page 5 of the TICC paper.  The
    nested loops take care of the B^{m}_{ij,l} at the top of
    page 6.

    Arguments:
        args (containers.ADMMArguments): bundled control parameters
        x (NumPy array): Latest X/Theta matrix, compressed
        u (NumPy array): Latest U matrix, compressed

    Returns:
        New estimate for Z matrix
    """

    theta_plus_u = x + u
    z_update = np.zeros(x.size)

    # Since these values are all independent, we could theoretically
    # parallelize this function.  My guess is that the overhead of
    # coordinating the tasks would far outweigh any benefit we'd get,
    # especially since this is not an expensive process overall.

    num_blocks = args.window_size
    block_size = args.num_data_series
    # this mess is all the indexing on page 6, left column
    for block_id in range(num_blocks):
        # How many times does this block show up in the
        # upper triangle of the matrix?
        num_occurrences = num_blocks - block_id
        for row in range(block_size):
            # block_id 0 is on the diagonal of the big matrix; we know that
            # that block is symmetric, so we only have to iterate over its
            # upper triangle.  For all other blocks we have to iterate through
            # all block_size x block_size entries.
            start_column = row if block_id == 0 else 0
            for col in range(start_column, block_size):
                # We want to operate on all the instances of the unique
                # value at position (r, c) within block block_id.
                # The calls to unique_variable_locations_* give us
                # lists of indices we can pass to NumPy to address
                # all those instances quickly.

                # Add up the elements of the Lambda (sparsity parameter)
                # matrix at these locations.  Note: lambda_sum is the
                # variable called Q in equation 9.
                lambda_sum = compute_lambda_sum(args.sparsity_weight,
                                                block_id, row, col,
                                                block_size,
                                                num_blocks)

                # Here's where to find those same elements in the compressed upper triangle
                indices = unique_values.locations_compressed(block_id,
                                                             row, col,
                                                             block_size,
                                                             num_blocks)
                # NOTE:
                # Computing this sum with np.sum instead of Python's sum changed the
                # algorithm output verrrrrrry slightly.  I'm guessing that this is
                # because np.sum uses a slightly different algorithm (pairwise summation)
                # than Python's sum.
                #
                # Also note: theta_plus_u called S in equation (9).
                scaled_point_sum = args.rho * np.sum(theta_plus_u[indices])

                z_update[indices] = soft_threshold_prox(scaled_point_sum,
                                                        lambda_sum,
                                                        args.rho * num_occurrences)
            # done iterating over columns in block
        # done iterating over rows in block
    # done iterating over blocks
    return z_update


def check_convergence(args: arguments.ADMMArguments,
                      u: np.ndarray,
                      x: np.ndarray,
                      z: np.ndarray,
                      z_old: np.ndarray) -> Tuple[bool, float, float, float, float]:
    """Look at X and Z matrices to decide whether optimization has converged

    Arguments:
        args (containers.ADMMArguments): Bundled control parameters
        u (NumPy array): Latest ADMM estimate for U
        x (NumPy array): Latest ADMM estimate for X/Theta
        z (NumPy array): Latest ADMM estimate for Z
        z_old (NumPy array): Previous ADMM estimate for Z

    Returns:
        Tuple with the following values:
            should_stop (bool): Optimization has converged
            residual_primal (float): Norm of difference between X and Z
            tolerance_primal (float): Convergence threshold for primal
                problem
            residual_dual (float): Norm of change in Z since last
                iteration scaled by rho
            tolerance_dual (float): Convergence threshold for dual
                problem
    """
    norm = np.linalg.norm
    # Primal and dual thresholds. Add .0001 to prevent the case of 0.
    absolute_term = (math.sqrt(x.size)
                     * args.absolute_tolerance
                     + 0.0001)
    tolerance_primal = (absolute_term
                        + args.relative_tolerance * max(norm(x), norm(z)))
    tolerance_dual = (absolute_term
                      + args.relative_tolerance * norm(args.rho * u))
    # Primal and dual residuals
    residual_primal = norm(x - z)
    residual_dual = norm(args.rho * (z - z_old))
    should_stop = ((residual_primal <= tolerance_primal) and
                   (residual_dual <= tolerance_dual))

    if args.verbose:
        LOGGER.debug("ADMM convergence check:")
        LOGGER.debug("    Solver converged:  %s", should_stop)
        LOGGER.debug("    Primary residual:  %f", residual_primal)
        LOGGER.debug("    Primary tolerance: %f", tolerance_primal)
        LOGGER.debug("    Dual residual:     %f", residual_dual)
        LOGGER.debug("    Dual tolerance:    %f", tolerance_dual)

    return (should_stop,
            residual_primal, tolerance_primal,
            residual_dual, tolerance_dual)


def soft_threshold_prox(scaled_point_sum: float,
                        lambda_sum: float,
                        rho_times_r: float) -> float:
    """Soft-threshold proximal operator for ADMM Z update

    This is the right-hand side of Equation (9) in the TICC paper.
    It computes one of the new unique values for the updated Z matrix
    that will then be broadcast to all the locations where that value
    occurs.

    Arguments:
        scaled_point_sum (float): rho * sum[l] S_l (S = Theta^{k+1} - U^k)
        lambda_sum (float): sum[l] Lambda_l
        rho_times_r (float): rho times the number of times this specific
            value occurs

    Returns:
        New value for several entries in Z matrix.  See Equation 9 for
        full explanation.
    """

    if scaled_point_sum > lambda_sum:
        updated_z_value = (scaled_point_sum - lambda_sum) / rho_times_r
        updated_z_value = max(updated_z_value, 0)
    elif scaled_point_sum < -1 * lambda_sum:
        updated_z_value = (scaled_point_sum + lambda_sum) / rho_times_r
        updated_z_value = min(updated_z_value, 0)
    else:
        updated_z_value = 0

    return updated_z_value


def compute_lambda_sum(lambda_parameter: Union[float, np.ndarray],
                       block_id: int,
                       row: int,
                       column: int,
                       block_size: int,
                       num_blocks: int) -> float:
    """Compute Q term from Equation 9

    When computing an update for one of the terms in the Z matrix,
    we need to add up the elements of the lambda matrix (AKA the
    MRF sparsity parameter) for all the places where that term occurs.

    There are two ways this can go.  If the user passed in a float value
    for lambda (meaning that the matrix is uniform), we can just
    multiply by the number of occurrences and call it a day.  If the
    user passed in an array for lambda (meaning they know more about
    what the structure of the MRF should be), we have to find the instances
    and add them up ourselves.

    Arguments:
        lambda_parameter (float or NumPy array): Value of sparsity parameter
            for TICC solver
        block_id (int): Block ID (from 0 to num_blocks - 1) containing the
             value we're updating
        row (int): Row of value within that block
        column (int): Column of value within that block
        block_size (int): Number of rows/columns in a block
        num_blocks (int): How many blocks are in one row/column of the big
            Z or Theta matrix

    Returns:
        Sum of lambda values for all of the places where (block_id, row,
        column) occurs

    Raises:
        ValueError: lambda_parameter is neither a float nor a NumPy array
    """

    if isinstance(lambda_parameter, float):
        num_occurrences = num_blocks - block_id
        return lambda_parameter * num_occurrences

    if isinstance(lambda_parameter, np.ndarray):
        # OK, fine, we'll do it the long way.
        (rows, cols) = unique_values.locations_index_slices(
            block_id, row, column, block_size, num_blocks
        )
        # NumPy slicing lets us avoid slow Python iteration
        return np.sum(lambda_parameter[rows, cols])

    raise ValueError((
        f"Lambda parameter (a {type(lambda_parameter)}) must be "
        f"either a float or a NumPy array."
    ))


def x_update_prox(empirical_covariance: np.ndarray,
                  z_minus_u: np.ndarray,
                  rho: float) -> np.ndarray:
    """Proximal operator for ADMM X update

    This is equation (6) from page 5 of the TICC paper.  This is
    the proximal operator that is part of the ADMM X update step.

    Arguments:
        empirical_covariance (numpy.ndarray): Empirical covariance matrix
        for all data points assigned to the current cluster
        z_minus_u (numpy.ndarray): Z - U (Z and U are ADMM variables)
        rho (float): ADMM hyperparameter.  Properly adjusting the
            value of rho can lead to faster convergence.

    Returns:
        Compressed version of new value for Theta matrix
    """
    # Equation 6 from the paper
    # Is rho * A an error here?  The paper suggests that it should
    # be (1/rho) * A.  The reference they got it from suggests that
    # rho * A is correct.
    d, q = np.linalg.eigh(rho * z_minus_u - empirical_covariance)
    # Is the 1 / (2 rho) an error here?  The paper suggests that it should
    # be rho/2, but again, the reference they got it from says that (1 / 2 rho)
    # is right.
    rho_scale = 1 / (2*rho)
    determinant = np.square(d) + (4*rho) * np.ones(d.shape)
    inner_term = np.diag(d + np.sqrt(determinant))
    theta_new = rho_scale * (q @ inner_term @ q.T)
    return matrix_compression.compress_matrix(theta_new)
