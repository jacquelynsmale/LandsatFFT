import pytest
import numpy as np
import filter
import scipy.ndimage as nd


@pytest.fixture(scope='module')
def box():
    arr = np.zeros((7, 7))
    arr[2:5, 2:5] = 1
    arr[6, 6] = 1
    return arr


@pytest.fixture(scope='module')
def box2():
    arr = np.zeros((7, 7))
    arr[2:5, 2:5] = 1
    arr[0, 0] = 1
    arr[6, 6] = 1
    return arr


@pytest.fixture(scope='module')
def distance(box):
    m, n = box.shape
    true_array = np.full((m, n), True)
    true_array[4, 4] = False
    dist_from_centroid = nd.distance_transform_edt(true_array)
    dist_from_centroid[box == 0] = 0
    return dist_from_centroid


@pytest.fixture(scope='module')
def regions(box):
    m, n = box.shape
    center_m = int(np.floor(m / 2))
    center_n = int(np.floor(n / 2))

    params = {'top_left': (0, center_m, 0, center_n),
              'top_right': (0, center_m, center_n + 1, n),
              'bottom_left': (center_m + 1, m, 0, center_n),
              'bottom_right': (center_m + 1, m, center_n + 1, n)}
    return params


def test_find_largest(box, box2):
    result1 = filter.find_largest_region(box)
    assert result1 == 1

    result2 = filter.find_largest_region(box2)
    assert result2 == 2

