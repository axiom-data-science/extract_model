#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Solutions for sorting functions."""
import numpy as np


try:
    from numba import njit

    HAS_NUMBA = True

    @njit
    def _numba_index_of_sorted(
        haystack: np.array, values: np.array
    ) -> np.array:  # pragma: no cover
        """Numba's JIT implementation of index_of_sorted.

        This function is O(lg n) for resolving an array of indices.
        """
        # pylint: disable=invalid-name
        out = np.full_like(values, -1, dtype=np.int32)
        n = haystack.shape[0]

        for i, search_val in np.ndenumerate(values):
            left = 0
            right = n - 1
            if search_val < haystack[0] or search_val > haystack[-1]:
                out[i] = -1
                continue
            while left <= right:
                m = (left + right) // 2
                if haystack[m] < search_val:
                    left = m + 1
                elif haystack[m] > search_val:
                    right = m - 1
                else:
                    out[i] = m
                    break
        return out

except ImportError:
    HAS_NUMBA = False


def _numpy_index_of_sorted(haystack: np.array, needle: np.array) -> np.array:
    """Pure numpy implementation of index_of_sorted.

    This function is O(n lg n) for resolving an array of indices.
    """
    i = np.searchsorted(haystack, needle)
    i[needle != np.take(haystack, i, mode="clip")] = -1
    return i


def index_of_sorted(haystack: np.array, needle: np.array) -> np.array:
    """Return an array of indexes for each value in values found in haystack.

    This function uses binary search on haystack to find each value in values and returns an array
    of indices or -1 if an exact value is not identified. This function behaves similarly to
    np.searchsorted but will return -1 if there is no exact value.

    Parameters
    ----------
    haystack: np.ndarray
        A _sorted_ array of values from which each value in values array is matched to.
    values: np.ndarray
        An array of values to search for.

    Returns
    -------
    np.ndarray
        An array indices which such that
    """
    if HAS_NUMBA:
        return _numba_index_of_sorted(haystack, needle)
    return _numpy_index_of_sorted(haystack, needle)
