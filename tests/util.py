# bartz/tests/util.py
#
# Copyright (c) 2024, Giacomo Petrillo
#
# This file is part of bartz.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from scipy import linalg

def assert_close_matrices(actual, desired, *, rtol=0, atol=0, tozero=False):
    """
    Check if two matrices are similar.

    Scalars and vectors are intepreted as 1x1 and Nx1 matrices, but the two
    arrays must have the same shape beforehand.

    The closeness condition is:

        ||actual - desired|| <= atol + rtol * ||desired||,

    where the norm is the matrix 2-norm, i.e., the maximum (in absolute value)
    singular value.

    Parameters
    ----------
    actual, desired : array_like
        The two matrices to be compared. Must be scalars, vectors, or 2d arrays.
    rtol, atol : scalar, default 0
        Relative and absolute tolerances for the comparison.
    tozero : bool, default False
        If True, use the following codition instead:

            ||actual|| <= atol + rtol * ||desired||

    Raises
    ------
    AssertionError :
        If the condition is not satisfied.
    """

    actual = np.asarray(actual)
    desired = np.asarray(desired)
    assert actual.shape == desired.shape
    if actual.size == 0:
        return
    actual = np.atleast_1d(actual)
    desired = np.atleast_1d(desired)
    
    if tozero:
        expr = 'actual'
        ref = 'zero'
    else:
        expr = 'actual - desired'
        ref = 'desired'

    dnorm = linalg.norm(desired, 2)
    adnorm = linalg.norm(eval(expr), 2)
    ratio = adnorm / dnorm if dnorm else np.nan

    msg = f"""\
matrices actual and {ref} are not close in 2-norm
matrix shape: {desired.shape}
norm(desired) = {dnorm:.2g}
norm({expr}) = {adnorm:.2g}  (atol = {atol:.2g})
ratio = {ratio:.2g}  (rtol = {rtol:.2g})"""

    assert adnorm <= atol + rtol * dnorm, msg
