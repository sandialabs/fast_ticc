Quirks, Tips, and Caveats
=========================

This page is about the corner cases, the quirks, and the subtleties involved in TICC.

Invalid Cluster Labels at Beginning and End
-------------------------------------------

When you run TICC with a window size of ``W``, the first ``W/2`` and the last ``W/2`` cluster labels for each data series will always be -1.   This is not an error.  The value -1 indicates "this point was not labeled".  Here's why.

TICC does all of its math on windows of data points.  For each window of length ``W``, it assigns a label to the data point in the middle of the window.  At the beginning and end of the data series, though, there aren't enough data points to fill the window.

The TICC paper says in section 2 (Problem Setup) that "the first *w* observations of *x:sub:`orig`* simply map to a shorter subsequence, since the time series does not start until *x:sub:`1`*."  It's not quite that simple in practice.  An incomplete data window (which is what they're describing) means that we are unable to compute the full inverse covariance matrix.  Similarly, we are unable to compute a correct log likelihood value to decide which cluster the points should belong to.

There are a few ways to deal with this:

1.  Acknowledge the shortfall and generate obviously-incorrect labels for the points we can't assign.  This is what we do.

2.  Fill in the missing entries in the first and last ``W/2`` data windows ourselves.  We decided against this -- see below for discussion.

3.  You, the user, can pad or otherwise extend your data by at least ``W/2`` points at beginning and end so that all the points you care about are guaranteed to have full windows.  This is the best choice if you really, truly need labels for those points.

4.  Re-use the first and last valid label for the incomplete windows.  That is, instead of labels that look like [-1, -1, -1, 0, 1, 2, 3, -1, -1, -1], produce [0, 0, 0, 0, 1, 2, 3, 3, 3, 3].  We decided against this because we want you as the user to be aware of which labels are real and which ones aren't.

5.  Add special-case code for the math for incomplete windows.  This ends up being much the same as option 2 (fill in the missing data) but much harder to implement.

6.  Rely on NumPy's support for masked arrays.  This is incompatible with using Numba for acceleration.

We welcome discussion and suggestions and especially contributions for dealing with this.



