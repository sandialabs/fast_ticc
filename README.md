# fast_ticc: Toeplitz Inverse Covariance Clustering

This package contains a reimplementation of the algorithm described in
"Toeplitz Inverse Covariance-Based Clustering" (hereinafter abbreviated
TICC) by D. Hallac, S. Vare, S. Boyd, and J. Leskovec.  It improves on
the authors' reference implementation mainly by adding more documentation,
more test cases, and Numba-based parallelism / JIT compilation where
appropriate.

TICC's purpose is to segment multivariate time-series data into regions
of similar behavior.  Here, "behavior" is defined by the covariance of the
different components of the time-series data in a window around the point
being labeled.

## Installing the Library

Our implementation of TICC is available from [PyPI](https://pypi.org)
and [conda-forge](https://conda-forge.org).  You can install it from
there with `pip install fast_ticc` and
`conda install -c conda-forge fast_ticc` (for Anaconda users),
respectively.

You can install directly from a copy of this repository with the
following two commands:

```bash
cd src
pip install .
```


## Using the Library

Your best resource for learning to use the library is
[the documentation](https://fast-ticc.readthedocs.io).
There are also examples in this repository under the
`src/examples/` directory.

### The Briefest of Quick-Start Instructions

Start with your data in an N x D NumPy array: one row per data point, one
column per variable.

Call TICC to compute labels:

```python
ticc_result = fast_ticc.ticc_compute_labels(my_data)
labels = ticc_result.point_labels
```

The main function (`ticc_compute_labels`) returns a structure with the
computed labels (in the `point_labels` field) and lots of information
describing the clusters and how well they describe the data.  You are
probably most interested in the labels themselves, which are a list of
integers with one entry for each input data point.

### Learning More

For further information, consult
[the documentation](https://fast-ticc.readthedocs.io)
or the original
[TICC paper](https://web.stanford.edu/~boyd/papers/ticc.html).

## Contributing

We welcome contributions!  Open a discussion or a pull request in this
repository and we'll talk.

## Authors

Andy Wilson, Daniel DeLayo, Renee Gooding, Jessica Jones, Kanad Khanna,
Nitin Sharan, and Jon Whetzel worked on this implementation.  Andy
Wilson was the chief author and is the maintainer.

## License

This library is distributed under a 3-clause BSD license.  Full text is
available in LICENSE at the top level of the repository along with the
license under which we use the original authors' implementation.