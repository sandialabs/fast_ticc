How To Use This Library
=======================

This page contains instructions for getting your data into Fast TICC, running the algorithm, and interpreting the results.


Labeling a Single Data Series
-----------------------------

Here's how to run TICC on a single data series:

1. Store your data in a NumPy array.  Each data point is a single row.  Each value within a data point is in its own column.  Don't include the timestamps -- just the data values.

2. Choose a number of clusters.  This should be the number of behaviors you expect to find in your data, more or less.

3. Choose a window size.  This should be the smallest number of data points in a row that you believe will be enough to identify each behavior you want TICC to identify.

4. Invoke TICC:

.. code-block:: python

    import fast_ticc
    ticc_result = fast_ticc.ticc_labels(my_data,
                                        window_size,
                                        num_clusters)


5. Cluster labels are in ``ticc_result.point_labels``.  This is a list of integers with the same length as the number of data points in your input.  Note that if your window size is W, the first W/2 and last W-2 points will all be labeled -1, meaning "no label was computed for this point".  See the "Beginning and Ending Labels" section of the :doc:`Quirks and Caveats <quirks>` page for an explanation of why this happens and what you can do about it.

Jointly Labeling Multiple Data Series
-------------------------------------

We also support labeling several data sets at the same time with a common set of labels.  Each data series can have a different number of points but must have the same number of values at each data point.

1. Store your data in NumPy arrays -- one array for each data set.  Each data point is a single row.  Each value within a data point is in its own column.  Don't include the timestamps -- just the data values.

2. Make a list (``my_data_arrays``) containing all the data sets you want to label.

3. Choose a number of clusters.  This should be the number of behaviors you expect to find in your data, more or less.

4. Choose a window size.  This should be the smallest number of data points in a row that you believe will be enough to identify each behavior you want TICC to identify.

5. Invoke TICC:

.. code-block:: python

    import fast_ticc
    ticc_result = fast_ticc.ticc_joint_labels(my_data_arrays,
                                              window_size,
                                              num_clusters)


5. Cluster labels are in ``ticc_result.point_labels``.  Unlike before, this is a list of lists.  The value of ``ticc_result.point_labels[i]`` is a list of labels for the ``i`` th data series.  As before, the first W/2 labels for each data series will be -1, as will the last W/2 labels.  See the "Beginning and Ending Labels" section of the :doc:`Quirks and Caveats <quirks>` page for an explanation of why this happens and what you can do about it.


