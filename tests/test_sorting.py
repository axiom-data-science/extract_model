#!/usr/bin/env pytest
# -*- coding: utf-8 -*-
"""Tests the sorting algorithm implementation."""
import numpy as np
import pytest

from extract_model.sorting import HAS_NUMBA, _numpy_index_of_sorted


@pytest.fixture(scope="session")
def deterministic_haystack():
    """Return a predictable sorted haystack to search through."""
    big_haystack = 2 * np.arange(10_000_000, dtype=np.int32)
    needle = 3 * np.arange(1_000_000, dtype=np.int32)
    return big_haystack, needle


@pytest.fixture(scope="session")
def stochastic_haystack():
    """Return a random sorted haystack to search through."""
    big_haystack = np.sort(np.random.randint(0, 2147483647, 10_000_000))
    needle = np.random.randint(0, 2147483647, 1_000_000)
    return big_haystack, needle


def test_numpy_deterministic_sorting(benchmark, deterministic_haystack):
    """Benchmark deterministic haystack using numpy implementation of index_of_sorted."""
    i = benchmark(_numpy_index_of_sorted, *deterministic_haystack)
    count = np.sum(i >= 0)
    assert count == 500000


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_numba_deterministic_sorting(benchmark, deterministic_haystack):
    """Benchmark deterministic haystack using numba implementation of index_of_sorted."""
    from extract_model.sorting import _numba_index_of_sorted

    i = benchmark(_numba_index_of_sorted, *deterministic_haystack)
    count = np.sum(i >= 0)
    assert count == 500000


def test_numpy_stochastic_sorting(benchmark, stochastic_haystack):
    """Benchmark stochastic haystack using numpy implementation of index_of_sorted."""
    benchmark(_numpy_index_of_sorted, *stochastic_haystack)


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_numba_stochastic_sorting(benchmark, stochastic_haystack):
    """Benchmark stochastic haystack using numba implementation of index_of_sorted."""
    from extract_model.sorting import _numba_index_of_sorted

    benchmark(_numba_index_of_sorted, *stochastic_haystack)
