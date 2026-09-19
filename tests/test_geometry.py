import numpy as np

from visionpipe.geometry import (
    iou_matrix,
    point_in_polygon,
    segments_properly_intersect,
    side_of_line,
)

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_point_in_polygon_basic():
    assert point_in_polygon((5, 5), SQUARE)
    assert not point_in_polygon((15, 5), SQUARE)
    assert not point_in_polygon((-1, -1), SQUARE)


def test_point_in_concave_polygon():
    # "L" shape: the notch at (7, 7) must be outside
    L = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]
    assert point_in_polygon((2, 8), L)
    assert point_in_polygon((8, 2), L)
    assert not point_in_polygon((7, 7), L)


def test_side_of_line_and_intersection():
    a, b = (0, 5), (10, 5)
    assert side_of_line((5, 0), a, b) != side_of_line((5, 9), a, b)
    assert side_of_line((5, 5), a, b) == 0
    assert segments_properly_intersect((5, 0), (5, 9), a, b)
    assert not segments_properly_intersect((15, 0), (15, 9), a, b)  # crosses the infinite line only


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    iou = iou_matrix(a, b)
    assert np.allclose(iou[0], [1.0, 1 / 3, 0.0])
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)
