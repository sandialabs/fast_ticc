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

"""Convert a 2D symmetric matrix to and from a compressed 1D representation

For efficiency, we store and operate on just the half of the inverse covariance
matrix that has unique values.  This module contains functions to go from
2D to 1D and back.
"""

import functools
from typing import List, Tuple

import numpy as np

__all__ = ["compress_matrix", "reinflate_matrix"]


def _full_matrix_size(flattened_size: int) -> int:
    """Given (n^2 + n) / 2, find n

    This function is part of reinflating a compressed matrix.  The
    compressed form is a row-major layout of the upper triangle
    including the diagonal.  For an n x n matrix, that yields
    (n^2 + n) / 2 elements.  The formula herein just inverts that.

    It looks opaque, I know.  If you'd like to verify it, start with

    flattened_size = (n^2 + n) / 2

    ...and solve for n using the quadratic equation.

    TODO:  Combine this function with the equivalent in
        graphical_lasso.py so we only have to make each mistake once

    Arguments:
        flattened_size (int): Size of compressed upper triangle (number
            of elements)

    Returns:
        Size in rows/columns (both are equal) of the uncompressed matrix
    """
    n = np.sqrt(8 * flattened_size + 1)
    return int((n-1) / 2)


def _uncompress_upper_triangle(compressed_tri: np.ndarray) -> np.ndarray:
    """Convert 1D compressed upper triangle to 2D matrix

    This is the inverse of the compression scheme we use
    for the matrices in the ADMM state.

    Arguments:
        compressed_tri (NumPy array): 1D array with (n^2 + n) / 2 elements

    Returns:
        Upper triangular matrix, n x n, with the elements from compressed_tri
    """
    size = _full_matrix_size(compressed_tri.shape[0])
    square = np.zeros(shape=(size, size))
    square[_upper_triangle_indices(size)] = compressed_tri
    return square


def _upper_to_full(upper_tri: np.ndarray) -> np.ndarray:
    """Rebuild a symmetric matrix from its upper triangle

    Here's what this does:
        full = (upper + upper.T) - diag(upper)

    We subtract off the diagonal because we wind up with
    two copies of it after (upper + upper.T).

    Arguments:
        upper_tri (NumPy array): Square matrix with values
            in the upper triangle and diagonal

    Returns:
        Symmetric matrix with that upper triangle

    Note:
        We don't verify that the lower triangle is empty.
        That's on the caller.
    """

    diag_temp = upper_tri.diagonal()
    full_matrix = (upper_tri + upper_tri.T) - np.diag(diag_temp)
    return full_matrix


def compress_matrix(full_matrix: np.ndarray) -> np.ndarray:
    """Compress a square symmetric matrix to its flattened upper triangle

    Starting with an N x N square matrix, we extract its upper triangle
    (including the diagonal) and then lay that out in row-major order.

    Arguments:
        full_matrix (NumPy array): Uncompressed square symmetric matrix

    Returns:
        Compressed upper triangle

    Raises:
        RuntimeError: what you passed is not a square matrix
    """

    if full_matrix.shape[0] != full_matrix.shape[1]:
        raise RuntimeError("compress_matrix: Input must be a square matrix")

    matrix_size = full_matrix.shape[0]
    return full_matrix[_upper_triangle_indices(matrix_size)]


def reinflate_matrix(compressed_utri: np.ndarray) -> np.ndarray:
    """Reconstruct a full symmetric matrix from its compressed upper triangle

    This is the inverse of the compression we use for the matrices inside
    the ADMM solver.

    Arguments:
        compressed_utri (NumPy array): 1D array with (n^2 + n) / 2 elements

    Returns:
        NumPy array containing n x n symmetric matrix
    """

    utri = _uncompress_upper_triangle(compressed_utri)
    full = _upper_to_full(utri)
    return full


@functools.cache
def _upper_triangle_indices(size: int) -> Tuple[List[int], List[int]]:
    """Caching version of NumPy triu_indices()

    Nothing fancy here, just a wrapper to let functools
    do its thing.

    Arguments:
        Size of square matrix to compress or uncompress

    Returns:
        (row_indices, column_indices), where zip(row_indices, column_indices)
        gives the locations of all the elements in the matrix's upper triangle
    """

    return np.triu_indices(size)
