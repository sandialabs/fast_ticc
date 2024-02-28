Installing Fast TICC
====================

If you're reading this message, congratulations, you're an early adopter!  The code is still hot from being uploaded and we're working on the PyPI and conda-forge recipes in real time.

From PyPI
---------

Once this code is officially released, we'll upload wheels to `PyPI <https://pypi.org>`_ so that you can install with ``pip install fast_ticc``.

From conda-forge
----------------

We're going to contribute a recipe to `conda-forge <https://conda-forge.org>`_ so that you can install with ``conda install -c conda-forge fast_ticc``.

From source
-----------

First, get a copy of the source.  Our Github repository is at https://github.com/sandialabs/fast_ticc and you can either clone or download the repository or download the release package for the latest version.

Second, if you haven't done so already, create a Python virtual environment to hold the ``fast_ticc`` installation and its dependencies.  This helps avoid version conflicts between different packages.

Third, go to the directory containing the package (whether you've cloned it or unpacked it from a download) and run ``python -m pip install .``.  This will build and install Fast TICC and its dependencies.

**Developer Mode**: If you want to work on the TICC code to add features or fix bugs, you can install it with ``python -m pip install -e .``.  The "-e" argument tells Python to use the source tree you downloaded instead of copying the library into its package collection.

PyPy, CPython, and Fast TICC
----------------------------

We support `PyPy <https://www.pypy.org>`_ as well as the `CPython interpreter <https://en.wikipedia.org/wiki/CPython>`_.  We depend on `scikit-learn <https://scikit-learn.org/stable>`_ and `NumPy <https://numpy.org>`_ for most of our math operations.  Other relatively recent Python interpreters (version 3.8 or newer) will probably work as long as they support those two packages.

If you're using CPython, Fast TICC will try to use `Numba <https://numba.pydata.org>`_ to accelerate some of its operations.  If Numba isn't available for whatever reason the code will still work fine.