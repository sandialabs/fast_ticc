.. Fast TICC documentation master file, created by
   sphinx-quickstart on Fri Dec 15 15:44:02 2023.


Welcome to the documentation for Fast TICC.
===========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   tutorial
   user_guide
   changelog
   how_ticc_works
   quirks
   plans

This library implements the clustering algorithm described in "Toeplitz Inverse Covariance-Based Clustering of Multivariate Time Series Data" by D. Hallac, S. Vare, S. Boyd, and J. Leskovec.  Its purpose is to take multivariate time-series data where each data point has a timestamp and two or more data values, then assign labels to those data points based on how their values change (or don't) together.

Here are a few examples of the kinds of data you could use with Fast TICC.

* Car telemetry.  Cars are continuously collecting data about the state of the vehicle, including the positions of the steering wheel, brake pedal, accelerator pedal; vehicle speed; and engine statistics such as RPM, temperature, and fluid levels.  You can use TICC to automatically identify segments of data where the car is driving straight, braking to come to a stop, braking into a turn, or accelerating out of a turn.

* Stock prices.  TICC can identify periods of behavior when collections of stocks exhibit coordinated behavior -- rising or falling together, being sold as a group or not.

* Health/fitness telemetry.  Suppose that a fitness tracker collects heart rate and velocity and acceleration data -- the kind of data used to count steps, for example.  TICC can label segments of data that may correspond to low- or high-intensity exercise, walking, running, or climbing steps.

Like any clustering algorithm, TICC is inherently unaware of the meaning of the data -- it's all just numbers.  Understanding the meaning of the clusters it creates is up to you.

Our library was originally based on the `reference TICC implementation <https://github.com/davidhallac/TICC>`_ by David Hallac and colleagues.  We thank them for making their work available to the community.

If you'd like to contribute, please visit `our GitHub repository <https://github.com/sandialabs/fast_ticc>`_ and jump in.  We welcome discussions, feature requests, and pull requests.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
