import numpy

from informed_sampling.generate import rotate


def test_rotate():
    x = numpy.array([1, 2.5, 3])
    y = numpy.array([0, 2.5, 6])
    # First rotate around 0
    bounds = [-2, 2, -4, 4]
    rx, ry = rotate(x, y, bounds, angle=numpy.deg2rad(90))
    assert numpy.allclose(rx, [0, -2.5, -6])
    assert numpy.allclose(ry, [1, 2.5, 3])

    # Then around another point
    bounds = [0, 6, -2, 2]
    rx, ry = rotate(x, y, bounds, angle=numpy.deg2rad(-90))
    assert numpy.allclose(rx, [3, 5.5, 9])
    assert numpy.allclose(ry, [2, 0.5, 0])
