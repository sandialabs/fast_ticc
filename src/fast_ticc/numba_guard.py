
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

"""JIT decorator safe to use whether or not Numba is installed

We want our code to run on Python interpreters where Numba isn't available
(Jython, PyPy) as well as in CPython installations where it just isn't
installed for whatever reason.

This file provides njit(), a decorator that will invoke Numba njit()
if available and do nothing if not; and prange(), a function
that falls through to numba.prange() or range() as appropriate.
"""

import logging

try:
    import numba
    logging.info("Likelihood computation will be accelerated with Numba.")
    NUMBA_AVAILABLE = True
    # Uncomment this next line if you want to make sure the code runs
    # without Numba.  Remember to comment it out again before committing.
    # NUMBA_AVAILABLE = False
except ImportError:
    logging.info("Numba not available.  Likelihood computations will run in Python.")
    NUMBA_AVAILABLE = False


__all__ = ["njit", "prange"]


def noop_decorator(func):
    """Decorator that does nothing

    This decorator is the fallback path for Numba JIT compilation.  If
    Numba cannot be imported, likelihood computations will be wrapped
    with this decorator.

    Arguments:
        func (function): Any function

    Returns:
        Wrapped version of func that just calls func
    """

    def wrapped(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapped


def fake_njit(*args, **kwargs):
    """Decorator that invoks numba.njit() if available

    This decorator falls through to numba.njit if available and the
    no-op decorator defined above if not.

    All arguments are passed along to the Numba wrapper.

    Arguments:
        func (function): Any function

    Returns:
        function: New function wrapped with Numba if available
    """

    if NUMBA_AVAILABLE:
        wrapper = numba.njit(*args, **kwargs)
    else:
        wrapper = noop_decorator

    return wrapper


def fake_prange(*args, **kwargs):
    """numba.prange() if available; range() if not

    All arguments will be passed through to range() or numba.prange()
    as appropriate.

    Returns:
        Range or parallel range object
    """

    if NUMBA_AVAILABLE:
        return numba.prange(*args, **kwargs)
    return range(*args, **kwargs)

if NUMBA_AVAILABLE:
    prange = numba.prange
    njit = numba.njit
else:
    prange = fake_prange
    njit = fake_njit
