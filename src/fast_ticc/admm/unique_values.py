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

"""Helper functions to locate occurrences of unique values within
compressed and uncompressed matrices

Because of the block Toeplitz structure of our Theta matrix,
we know that there are only approximately N * N * W unique
values in the whole N * W x N * W matrix.  Moreover, we know
where they are.

By using that information, we can do a lot less work in updating
the Z matrix than we would need to if we computed each value
from scratch.

These indexing shenanigans are what the B^{(m)}_{ij,l} expression
on page 6 of the TICC paper are talking about.
"""

import functools
from typing import List, Tuple

from fast_ticc.ticc_types import ArrayPositionList, IndexList


def _size_including_this_row(r: int, uncompressed_size: int) -> int:
    """Size of compressed representation up to the end of row r

    You should not need to call this method yourself.

    Given a matrix in compressed format, return the number of
    elements in all rows up to and including row r.

    Arguments:
        r (int): Row in matrix
        uncompressed_size (int): Rows/cols in big matrix

    Returns:
        Number of elements in all rows up to and including r
    """
    return uncompressed_size*(r+1) - r*(r+1)/2


def _elements_in_row_after_target(c: int,
                                  full_row_length: int) -> int:
    """How many elements are left in the row after the one we want?"""
    return (full_row_length-1) - c


# This function gets called a lot.  In the original version, we spent
# 7% of our wall-clock time just in here.  Having functools memoize it
# speeds it up by about a factor of three.
#
# It seems as though we should be able to make it run even faster by
# turning it into an array lookup.  I've tried several different ways
# of doing that and have been unable to do more than marginally better
# than functools.cache does on its own.  I'm going to stick with the
# simplest thing that works well.

@functools.cache
def _compressed_index(row: int, column: int, uncompressed_size: int) -> int:
    """2D (i, j) coordinates -> index in compressed representation

    This function is an adapter from coordinates in a square 2D matrix
    to an index in our compressed 1D representation.

    Arguments:
        row (int): Row of desired element
        column (int): Column of desired element
        uncompressed_size (int): Rank of 2D matrix

    Returns:
        Index into compressed 1D representation

    Raises:
        IndexError: you've asked for coordinates outside the upper triangle
    """

    if column < row:
        raise IndexError((f"Coordinates ({row}, {column}) are outside "
                         "the matrix's upper triangle."))

    return int(
        _size_including_this_row(row, uncompressed_size)
        - (_elements_in_row_after_target(column, uncompressed_size) + 1)
        )


def _block_start_coordinates(block_id: int,
                            block_size: int,
                            window_size: int) -> List[Tuple[int, int]]:
    """Return the start coordinates for all instances of a given block

    The matrices we're building are assembled from a total of W unique
    N x N matrices (W is window_size, N is num_data_series) arranged
    in a block Toeplitz structure.  In the paper, these blocks
    are referred to as A^(i).

    This function computes the coordinates of the upper left corner of
    all the occurrences of one of those unique blocks.

    Arguments:
        block_id (int): Block index.  Legal values are from 0 to
            window_size - 1.
        block_size (int): Number of rows and columns in each block.
            This comes from num_data_series originally.
        window_size (int): Number of data points stacked together
            to form the temporal subsequences we process.

    Returns:
        List of (row, column) tuples.  Each entry is the location of
        the upper left corner of one instance of the given block.

    Raises:
        IndexError: block_id < 0 or block_id >= window_size
        ValueError: block_size <= 0 or window_size <= 0
    """

    if block_id < 0 or block_id >= window_size:
        raise IndexError((
           f"Block ID ({block_id}) must be between 0 and {window_size-1}."
        ))

    if block_size <= 0:
        raise ValueError((
            f"Block size ({block_size}) cannot be negative."
        ))

    if window_size <= 0:
        raise ValueError((
            f"Window size ({window_size}) cannot be negative."
        ))

    num_occurrences = window_size - block_id
    # All blocks are present in the top row
    start_row = 0
    start_column = block_id * block_size
    corner_coordinates = []
    for i in range(num_occurrences):
        corner_coordinates.append((
            start_row + i * block_size,
            start_column + i * block_size
        ))
        assert corner_coordinates[-1][0] >= 0
        assert corner_coordinates[-1][0] < window_size * block_size
        assert corner_coordinates[-1][1] >= 0
        assert corner_coordinates[-1][1] < window_size * block_size
    return corner_coordinates


def _unique_variable_locations(block_id: int,
                               row_in_block: int,
                               col_in_block: int,
                               block_size: int,
                               num_blocks: int) -> ArrayPositionList:
    """Find all the locations of a given unique element in Theta

    This function encapsulates the index-wrangling that goes on
    inside ADMM_z() where we compute values for the N^2 W unique
    values in our big matrix.

    You will probably not need to call this directly.  Instead,
    call one of the versions that gives you back the index type
    you need.

    Arguments:
        block_id (int): Block ID from 0 to num_blocks - 1
        row_in_block (int): Row of desired element within block
        col_in_block (int): Column of desired element within block
        block_size (int): Number of rows and columns in block (also known
            as num_data_series)
        num_blocks (int): Number of blocks in a row of the big matrix
            (also known as window_size)

    Returns:
        List of (r, c) positions in array (upper triangle only)
    """

    block_corners = _block_start_coordinates(block_id, block_size, num_blocks)
    element_positions = [(r + row_in_block, c + col_in_block) for (r, c) in block_corners]
    return element_positions


@functools.cache
def locations_compressed(block_id: int,
                         row_in_block: int,
                         col_in_block: int,
                         block_size: int,
                         num_blocks: int) -> IndexList:
    """Get all locations for a unique value from a block

    Find all the positions in the full Theta/Z matrix (upper triangle only)
    where a particular value from one block occurs.  Return as indices
    into the compressed upper triangle representation.

    Arguments:
        block_id (int): Block ID from 0 to num_blocks - 1
        row_in_block (int): Row of desired element within block
        col_in_block (int): Column of desired element within block
        block_size (int): Number of rows and columns in block (also known
            as num_data_series)
        num_blocks (int): Number of blocks in a row of the big matrix
            (also known as window_size)

    Returns:
        List of indices into compressed representation
    """

    positions_as_coordinates = _unique_variable_locations(block_id,
                                                          row_in_block,
                                                          col_in_block,
                                                          block_size,
                                                          num_blocks)
    full_matrix_size = block_size * num_blocks
    indices = [_compressed_index(r, c, full_matrix_size)
               for (r, c) in positions_as_coordinates]
    return indices


@functools.cache
def locations_index_slices(block_id: int,
                           row_in_block: int,
                           col_in_block: int,
                           block_size: int,
                           num_blocks: int) -> Tuple[IndexList, IndexList]:
    """Get all locations for a unique value from a block

    Find all the positions in the full Theta/Z matrix (upper triangle only)
    where a particular value from one block occurs.  Return as a list of
    row indices and a list of column indices.  This can be passed to a 2D
    NumPy array to return all the selected elements quickly.

    Arguments:
        block_id (int): Block ID from 0 to num_blocks - 1
        row_in_block (int): Row of desired element within block
        col_in_block (int): Column of desired element within block
        block_size (int): Number of rows and columns in block (also known
            as num_data_series)
        num_blocks (int): Number of blocks in a row of the big matrix
            (also known as window_size)

    Returns:
        Tuple of (row_indices, column_indices)
    """

    positions_as_coordinates = _unique_variable_locations(block_id,
                                                          row_in_block,
                                                          col_in_block,
                                                          block_size,
                                                          num_blocks)

    row_indices = [r for (r, _) in positions_as_coordinates]
    column_indices = [c for (_, c) in positions_as_coordinates]
    return (row_indices, column_indices)
