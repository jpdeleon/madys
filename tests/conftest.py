import pytest
import numpy as np
import os
import sys

madys_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, madys_path)


@pytest.fixture
def sample_arrays():
    return {
        'sorted_array': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        'unsorted_array': np.array([5.0, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0, 4.0, 6.0, 10.0]),
        'values': [3.5, 5.0, 10.0, 0.5],
        'values_list': [3.5, 5.0, 10.0],
    }


@pytest.fixture
def array_with_nans():
    return np.array([1.0, np.nan, 3.0, np.nan, 5.0, 6.0, np.nan, 8.0, 9.0, np.nan])


@pytest.fixture
def magnitude_pairs():
    return [
        (1.0, 2.0),
        (5.0, 5.0),
        (10.0, 15.0),
    ]


@pytest.fixture
def isochrone_grid_params():
    return {
        'mass': np.linspace(0.1, 10.0, 100),
        'age': np.logspace(1, 3, 50),
        'feh': 0.0,
        'afe': 0.0,
    }