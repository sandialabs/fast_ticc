An Overview of TICC
===================

This page is a very-high-level guided tour of the TICC algorithm.  We're going to gloss over a lot of the technical details and instead focus on a general understanding of what's going on with attention to how to choose values for the algorithm's input parameters.  In particular, we're not going to talk about TICC's cross-time covariance tracking. Please refer to sections 1 and 2 of the TICC paper for a detailed description of how that works.


TICC is an Unsupervised Clustering Algorithm
--------------------------------------------

TICC is an unsupervised clustering algorithm.  All unsupervised clustering algorithms take as input a collection of data points and a few parameters, then assign points to groups based on some notion of similarity.  Each of these groups is called a *cluster*.

Input: Multivariate Time-Series Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TICC's input is an ordered sequence of points that is assumed to come from measurements of some system over time.  Each measurement has to include at least two quantities.  For the sake of discussion, we'll say that each component of a data point comes from a different *sensor*.   For TICC to work well, the sensors need to be measuring quantities that influence one another somehow.

Let's make this more concrete with an example.

The Car Example
***************

We'll use the car telemetry example from the TICC paper for illustration.  Modern cars are constantly recording information about all aspects of their operation, including...

* Driver controls

  * Steering wheel angle
  * Accelerator pedal position
  * Brake pedal position
  * Selected gear
  * Selected windshield wiper state
  * Turn signal status

* Engine data

  * Engine RPM
  * Engine temperature
  * Oil level

* Other vehicle data

  * Wheel rotation speed
  * Brake status
  * Exterior and interior temperature
  * Car acceleration

For simplicity's sake, we're going to focus on a small subset of these that are most relevant to the car's motion:

* Steering wheel angle
* Accelerator pedal position
* Brake pedal position
* Wheel rotation speed
* Car acceleration in X and Y [#f1]_

That gives us six sensors for each data point.

A single data point comprises measurements from all six sensors at a single instant.  Let's suppose that our sensors record at a rate of 60 measurements per second.



Required Parameters: Number of Clusters and Window Size
-------------------------------------------------------

TICC requires two parameters apart from the input data itself: the number of clusters and the window size.  Choosing good values for both of these depends on what you want to learn from the data.

Choosing a Number of Clusters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TICC is good at picking out different regimes of behavior in time-series data.  To choose a number of clusters, think about how many different kinds of behavior you expect to find.  The number of clusters needs to be much smaller than the number of data points.

Absent any further guidance, pick a number somewhere between four and twenty.  A smaller number will tend to yield larger clusters whose members are similar in broad senses but differ in details.  A larger number will usually yield smaller clusters dominated by some unique characteristic.  You can and should experiment with this: exploring clustering behavior is a great way to get to know your data.

For the car example, we might expect our clusters to capture phenomena like accelerating/decelerating on a straight road, stopping, accelerating from a stop, slowing down into a turn, accelerating out of a turn, and maintaining a constant speed during a turn.  Seven or eight clusters would be a good place to start.

Choosing a Window Size
^^^^^^^^^^^^^^^^^^^^^^

TICC builds clusters based on the covariance of the different sensor values over a short time window.  This window should be long enough to show meaningful covariance in the data but otherwise as short as possible.  It should be significantly smaller than the length of time any given behavior will last in the data.

When a human is driving a car, behaviors typically last a minimum of 1-2 seconds.  A window duration between 0.25 and 0.5 seconds might work well.

Note that TICC expects its window size to be specified in terms of the number of data points.  Since we decided that our sensors sample at 60 Hz, this gives us a window size of 15-30 data points.  As with the number of clusters, this is a good parameter to experiment with to see what results you get.


How TICC Works
--------------

TICC's operation follows a pattern common to most unsupervised clustering algorithms:

0. Arrange the data in whatever form we need.
1. Initialize the allocation of points to clusters with some not-totally-unreasonable guess.
2. Update the description of each cluster to better represent the points it contains.
3. Compute the cost of assigning each point to each separate cluster. [#f2]_
4. Given those costs, assign points to clusters in a way that minimizes overall cost.
5. Repeat steps 2-4 until the cluster assignments stop changing.

In this section we'll give a summary of how TICC implements each of these tasks.  We'll call out parameters you can change and then list them all later on.


Phase 0: Data Preparation
^^^^^^^^^^^^^^^^^^^^^^^^^

Unlike many covariance-based clustering methods, TICC computes a precision matrix separately for each time delay within its time window. In other words, for a window size of W, we compute W-1 separate precision matrices: one for points at time T and T-1, another for points at time T and T-2, and so on, up to points at times T and (T-W+1).

To make this easier to compute, we stack W copies of the input data on top of one another (where W is the window size parameter).  Each successive copy is shifted forward in time by one point.


Phase 1: Initialization
^^^^^^^^^^^^^^^^^^^^^^^

We initialize the cluster assignments by computing a `Gaussian mixture model <https://scikit-learn.org/stable/modules/mixture.html>`_ with the full covariance matrix of the stacked data.  We use the same number of components for the mixture model as the number of clusters we intend to create with TICC.


Phase 2: Update Cluster Description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In TICC, each cluster is described by a matrix that (loosely speaking) describes the influence between different sensor values at different times.  In the paper and in our code it's called the Markov Random Field.  "Inverse covariance matrix" and "precision matrix" are equally valid names.

We optimize the description of each cluster separately.  We start with the empirical covariance matrix of the data assigned to a cluster, then use the `alternating direction method of multipliers (ADMM) <https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf>`_ to encourage TICC to produce sparse, easily-explained results.  This process is described in sections 3.2 and 4.2 of the TICC paper.  The weight parameter that governs how strongly we bias the solver toward generating sparse matrices is denoted :math:`\lambda` in the TICC paper and is called ``sparsity_weight`` in our code.

This phase is implemented in ``graphical_lasso.py``.  Our ADMM solver is in the ``fast_ticc.admm`` package.

Optimizing the Markov random field describing each cluster is the most computationally expensive part of TICC.  Fortunately, most of the expense is in linear algebra operations ultimately handled by a `BLAS library <https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms>`_ like OpenBLAS.  These usually do their own multithreading and are heavily optimized.


Phase 3: Compute Labeling Costs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We compute the (log) likelihood of each point with respect to each cluster.  We assume that the points are produced from a multivariate normal distribution.  See section 2 of the TICC paper or ``likelihood.py`` in our source tree for the details.

This is a moderately expensive thing to compute.  Where available, we call out to `Numba <https://numba.pydata.org>`_ to accelerate and parallelize this step.


Phase 4: Assign Points to Clusters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once we have the cost of assigning each point P to each separate cluster C, we can compute the lowest-cost assignment of all points to clusters.  This is where the *label switching cost* parameter comes in [#f3]_.  We want TICC to create clusters that are coherent in time instead of constantly switching back and forth.  To accomplish this, we impose a cost penalty whenever the label changes.

We compute the actual assignment using the `Viterbi algorithm <https://en.wikipedia.org/wiki/Viterbi_algorithm>`_.  The implementation is in ``cluster_label_assignment.py``.  This phase takes very little time to execute.


Phase 5: Test For Convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a simple comparison: have any points moved to different clusters since the last iteration?

There's a fair argument to make that this isn't a powerful enough test; that we should instead watch the cluster MRFs to make sure they also aren't changing.  If you would like to work with this and send us a pull request on Github we would be delighted to see it.

We also keep track of how many iterations of the main loop we've done.  By default, we bail out and declare victory after 1000 iterations whether or not assignments have converged.  In our experiments we have rarely needed more than 100 iterations to get a stable assignment.

Finished!
^^^^^^^^^

We return the following information once we're done:

* Integer cluster label for each point
* Markov random field describing each cluster
* Total assignment cost for each cluster
* Statistics on log likelihood for points within each cluster
* `Bayesian information criterion <https://en.wikipedia.org/wiki/Bayesian_information_criterion>`_ for the overall model
* `Calinski-Harabasz index <https://en.wikipedia.org/wiki/Calinski%E2%80%93Harabasz_index>`_ for the overall model

The Bayesian information criterion and Calinski-Harabasz index can be useful in evaluating the quality of a clustering model and in comparing one model against another.


All The Parameters
------------------

Here's a full list of parameters you can specify when running Fast TICC.  These are also spelled out in the examples [FIXME: URL] and API documentation [FIXME: URL].

**Required**: Input data.  Each row in your input data is a set of measurements from one point in time.  Each column contains the measurements from one sensor.  This array must be dense.  If there are missing values in your original data, you must come up with a plausible way to impute those values.

**Required**: Number of clusters.  See above for advice on choosing a value for this parameter.  Intuition: this should be about the same as the number of different behaviors you expect to observe in your data.

**Required**: Window size.  See above for advice on choosing a value for this parameter.  Intuition: large enough to show meaningful covariance between different sensors, but much shorter than the length of time you expect a behavior to last.

**Optional**: Label switching cost.  The higher this cost, the more TICC is encouraged to keep the same label from one point to the next.  If this cost is too low, cluster labels will change more frequently than they should.  We've found that a reasonable value is about 5 times the average log likelihood for each point.

**Optional**: Sparsity weight.  This parameter encourages the ADMM solver to produce simpler Markov random fields.  We have no reliable intuition yet on how to set this value, so we use the value from the TICC reference implementation.

**Optional**: Iteration limit.  How many times are you willing to go around the main loop before giving up and declaring victory?  The default value of 1000 should be fine.

**Optional**: Minimum cluster size.  Like many iterative clustering algorithms, TICC can break if clusters get too small.  At each iteration, we check to see if clusters have fallen below the threshold number of points specified by this parameter.  If so, we pull points from other clusters to repopulate them.

**Optional**: Minimum meaningful covariance.  For real-world applications, there is often some floor beneath which covariance might as well be zero.  In order to promote numerical stability in the solver, we go through the Markov random fields after each iteration and zero out any entry with magnitude smaller than this threshold.

**Optional**: Number of processors.  You will almost never need to change this.  Please refer to the API documentation [FIXME] for an explanation.

**Optional**: Biased vs. unbiased covariance.  If you care about sample mean versus population mean, this parameter is for you.


Quirks and Caveats
------------------

Now it's time for full disclosure.  This is where we talk about all the quirks we've observed with TICC in general and our implementation in particular.

...I'm not going to write this tonight.


.. rubric:: Footnotes

.. [#f1] If your car is accelerating significantly in the Z direction, you may have other concerns.  Buckle up.

.. [#f2] Think of this cost as a measure of goodness of fit.  Assigning a point to a cluster that matches it well doesn't cost much.  The worse the fit, the higher the cost.

.. [#f3] In the TICC paper and the reference implementation, the label-switching cost is denoted :math:`\beta`.  In our code, it is called ``label_switching_cost``.
