# Changes to Fast TICC by release

This document will contain a reference to all the changes we make as
we improve Fast TICC over time.  We intend to track our work using
issues on Github so we'll include links to those.

## Release 1.0.0: February 20, 2024

Initial release.

## Release 1.0.1: March 26, 2024

Fixed two bugs:

- #5: Numba was missing type information.  This was causing weird compile-like errors.
- #6: Argument order was flipped between `num_clusters` and `window_size` in the user guide.

