import numpy as np
from helical.models.scgpt.binning import _digitize, binning
import torch
import pytest


def test_digitize_basic():
    np.random.seed(42)
    x = np.array([-99, 1, 2, 3, 4, 5, 6, 9])
    bins = np.array([2, 4, 9])
    result = _digitize(x, bins)
    # -99 and 1 are below the first bin, 2 and 3 are in the first bin
    # 4, 5 and 6 are in the second bin, 9 in the third
    expected = np.array([0, 0, 1, 1, 2, 2, 2, 3])
    assert np.array_equal(result, expected)


def test_digitize_side_one():
    np.random.seed(42)
    x = np.array([1, 2, 3, 4, 5])
    bins = np.array([2, 4])
    result = _digitize(x, bins, side="one")
    expected = np.array([0, 1, 1, 2, 2])
    assert np.array_equal(result, expected)


def test_digitize_empty_array():
    np.random.seed(42)
    x = np.array([])
    bins = np.array([2, 4])
    result = _digitize(x, bins)
    expected = np.array([])
    assert np.array_equal(result, expected)


def test_digitize_identical_bins():
    np.random.seed(42)
    x = np.array([1, 2, 3, 4, 5])
    bins = np.array([2, 2, 4, 4])
    result = _digitize(x, bins)
    expected = np.array([0, 1, 2, 3, 4])
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "row, expected",
    (
        (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([1, 1, 1, 2, 2, 2, 3, 3, 4])),
        (
            torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            np.array([1, 1, 1, 2, 2, 2, 3, 3, 4]),
        ),
        # distrubution of the bins depends on the distribution of the data
        (np.array([1, 1, 1, 1, 1, 1, 1, 8, 9]), np.array([2, 1, 1, 2, 3, 3, 3, 3, 4])),
        (np.array([1, 2, 1, 1, 9, 6, 7, 8, 9]), np.array([1, 2, 1, 1, 4, 2, 2, 3, 4])),
    ),
)
def test_binning_basic(row, expected):
    np.random.seed(42)
    n_bins = 5
    result = binning(row, n_bins)
    assert np.array_equal(result, expected)


def test_binning_with_zeros():
    np.random.seed(42)
    row = np.array([0, 0, 0, 0, 0])
    n_bins = 5
    result = binning(row, n_bins)
    expected = np.zeros_like(row)
    assert np.array_equal(result, expected)


def test_binning_with_negative_values():
    np.random.seed(42)
    row = np.array([-1, -2, -3, -4, -5, -6, -7, -8, -9])
    n_bins = 5
    result = binning(row, n_bins)
    expected = np.array([4, 3, 3, 2, 2, 2, 1, 1, 1])
    assert np.array_equal(result, expected)
