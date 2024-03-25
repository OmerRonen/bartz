# bartz/tests/test_prepcovars.py
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

import pytest
from jax import numpy as jnp
from jax import random
import numpy

import bartz

from .rbartpackages import BART
from . import util

@pytest.mark.parametrize('fill_value', [jnp.inf, 2 ** 31 - 1])
def test_splits_fill(fill_value):
    fill_value = jnp.array(fill_value)
    x = jnp.array([[1, 3, 3, 5], [1, 3, 5, 7]], fill_value.dtype)
    splits, _ = bartz.prepcovars.quantilized_splits_from_matrix(x, 100)
    expected_splits = [[2, 4, fill_value], [2, 4, 6]]
    numpy.testing.assert_array_equal(splits, expected_splits)

def test_integer_splits_overflow():
    x = jnp.array([[-2 ** 31, 2 ** 31 - 2]])
    splits, _ = bartz.prepcovars.quantilized_splits_from_matrix(x, 100)
    expected_splits = [[-1]]
    numpy.testing.assert_array_equal(splits, expected_splits)

@pytest.mark.parametrize('dtype', [int, float])
def test_splits_type(dtype):
    x = jnp.arange(10, dtype=dtype)[None, :]
    splits, _ = bartz.prepcovars.quantilized_splits_from_matrix(x, 100)
    assert splits.dtype == x.dtype

def test_splits_length():
    x = jnp.linspace(0, 1, 10)[None, :]
    
    short_splits, _ = bartz.prepcovars.quantilized_splits_from_matrix(x, 2)
    assert short_splits.shape == (1, 1)
    
    long_splits, _ = bartz.prepcovars.quantilized_splits_from_matrix(x, 100)
    assert long_splits.shape == (1, 9)
    
    just_right_splits, _ = bartz.prepcovars.quantilized_splits_from_matrix(x, 10)
    assert just_right_splits.shape == (1, 9)
    
    no_splits, _ = bartz.prepcovars.quantilized_splits_from_matrix(x, 1)
    assert no_splits.shape == (1, 0)

def test_binner_left_boundary():
    splits = jnp.array([[1, 2, 3]])

    x = jnp.array([[0, 1]])
    b = bartz.prepcovars.bin_predictors(x, splits)
    numpy.testing.assert_array_equal(b, [[0, 0]])

def test_binner_right_boundary():
    splits = jnp.array([[1, 2, 3, 2 ** 31 - 1]])

    x = jnp.array([[2 ** 31 - 1]])
    b = bartz.prepcovars.bin_predictors(x, splits)
    numpy.testing.assert_array_equal(b, [[3]])

def test_quantilize_round_trip():
    x = jnp.arange(10)[None, :]
    splits, _ = bartz.prepcovars.quantilized_splits_from_matrix(x, 100)
    b = bartz.prepcovars.bin_predictors(x, splits)
    numpy.testing.assert_array_equal(x, b)
