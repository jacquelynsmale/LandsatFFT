import pytest
import numpy as np
import filter
import scipy.ndimage as nd


@pytest.fixture(scope='module')
def box():
    arr = np.zeros((9, 9))
    arr[3:6, 3:6] = 1
    arr[8, 8] = 1
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


def test_centroid(box):
    result = filter.find_centroid(box)
    assert result == (4, 4)


def test_find_corners(regions, distance):
    result = filter.locate_max_in_subset(regions, distance)
    assert result['top_left'] == (3, 3)
    assert result['top_right'] == (3, 5)
    assert result['bottom_left'] == (5, 3)
    assert result['bottom_right'] == (8, 8)


def test_slope():
    result = filter.calculate_slope((0, 0), (1, 1))
    assert np.rad2deg(result) == 45
