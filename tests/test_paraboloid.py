import numpy
import pytest

from informed_sampling.paraboloid import Paraboloid, vertex2std


def test_generate():

    x_mean = 10
    y_mean = 30
    bounds = numpy.array([0, 40, 0, 50])
    max_slope = 20
    integral_values = (1000, 9000)

    peaks = []
    for _ in range(1000):
        pboid = Paraboloid.generate(
            x_mean=x_mean,
            x_std=10,
            y_mean=y_mean,
            y_std=10,
            bounds=bounds,
            slope=max_slope,
            values=integral_values,
        )

        # Check that it is sloping down
        assert pboid.coeff[0] <= 0

        # The gradient is less than the max
        for x, y in ([0, 0], [40, 0], [0, 50], [40, 50]):
            assert numpy.linalg.norm(pboid.gradient(x, y)) <= max_slope

        # The integral is within bounds
        integral = pboid.integrate(bounds)
        assert integral_values[0] <= integral
        assert integral_values[1] >= integral

        # The peaks are (on average) at the mean
        peaks.append(pboid.peak)

    peaks = numpy.array(peaks)
    assert numpy.isclose(peaks[:, 0].mean(), x_mean, atol=5)
    assert numpy.isclose(peaks[:, 1].mean(), y_mean, atol=5)


class TestGradient:

    def test_null(self):
        pboid = Paraboloid(coeff=(0, 0, 0, 0, 0, 0))

        # Test point
        grad = pboid.gradient(x=10, y=100)
        assert grad.shape == (2,)
        assert numpy.allclose(grad, 0)

        # Test array
        grad = pboid.gradient(
            x=numpy.array([1, 1, 2, 3]), y=numpy.array([5, 8, 13, 21])
        )
        assert grad.shape == (2, 4)
        assert numpy.allclose(grad, 0)

    def test_mismatch(self):
        pboid = Paraboloid(coeff=(0, 0, 0, 0, 0, 0))
        with pytest.raises(ValueError):
            pboid.gradient(x=numpy.array([1, 2, 3, 4]), y=numpy.array([1, 2]))

    def test_simple(self):
        pboid = Paraboloid(coeff=(1, 2, 0, 0, 0, 0))

        # Test point
        assert numpy.allclose(pboid.gradient(x=1, y=0), [2, 0])
        assert numpy.allclose(pboid.gradient(x=0, y=1), [0, 4])
        assert numpy.allclose(pboid.gradient(x=1, y=1), [2, 4])

        # Test array
        assert numpy.allclose(
            pboid.gradient(x=numpy.array([-1, 0, -1]), y=numpy.array([0, -1, -1])),
            numpy.array([[-2, 0, -2], [0, -4, -4]]),
        )


class TestIntegrate:

    def test_null(self):
        pboid = Paraboloid(coeff=(0, 0, 0, 0, 0, 0))
        assert numpy.isclose(pboid.integrate(bounds=(0, 10, 0, 10)), 0)

    def test_empty(self):
        pboid = Paraboloid(coeff=(1, 1, 0, 0, 0, 0))
        assert numpy.isclose(pboid.integrate(bounds=(0, 0, 0, 0)), 0)

    def test_simple(self):
        pboid1 = Paraboloid(coeff=(1, 1, 0, 0, 0, 0))
        result1 = pboid1.integrate(bounds=(0, 1, 0, 1))
        assert result1 < 1

        pboid2 = Paraboloid(coeff=(2, 2, 0, 0, 0, 0))
        result2 = pboid2.integrate(bounds=(0, 1, 0, 1))
        assert result2 < 4
        assert numpy.isclose(result1 * 2, result2)

    def test_flat(self):
        pboid = Paraboloid(coeff=(0, 0, 0, 0, 0, 0.25))
        result = pboid.integrate(bounds=(-2, 0, -5, 0))
        assert numpy.isclose(result, 2.5)

    def test_plane(self):
        pboid = Paraboloid(coeff=(0, 0, 0, 3, 0, 0))
        result = pboid.integrate(bounds=(1, 2, -2, 0))
        assert numpy.isclose(result, 9)


def test_r2():
    # Test a perfect fit
    pboid = Paraboloid(coeff=(-1, -2, 3, -4, 5, 100))
    pboid.data = numpy.vstack(pboid.predict_grid([-2, 2, -5, 5], spacing=0.1))
    assert numpy.isclose(pboid.r2_score(), 1.0)

    # Make the fit a little less good
    pboid.data[2, 10] += 3
    pboid.data[2, 20] -= 3
    assert pboid.r2_score() < 1.0
    assert pboid.r2_score() > 0.95


class TestErrors:
    @pytest.mark.parametrize("error_fn", ["mae", "rmse"])
    def test_nudge(self, error_fn):
        # Test a perfect fit
        pboid = Paraboloid(coeff=(-1, -2, 3, -4, 5, -6))
        x, y, z = pboid.predict_grid([-2, 2, -5, 5], spacing=0.1)
        assert numpy.isclose(getattr(pboid, error_fn)(x, y, z), 0.0)

        # Make the fit a little less good
        z[10] += 3
        z[20] -= 3
        assert getattr(pboid, error_fn)(x, y, z) > 0

    @pytest.mark.parametrize("error_fn", ["mae", "rmse"])
    @pytest.mark.parametrize("offset", [-1, 1])
    def test_offset(self, error_fn, offset):
        pboid1 = Paraboloid(coeff=(-1, -1, 0, -4, 5, 100))
        x, y, z = pboid1.predict_grid([-2, 2, -5, 5], spacing=0.1)

        # The last coefficient (constant term) is +-1
        pboid2 = Paraboloid(coeff=(-1, -1, 0, -4, 5, 100 + offset))
        assert numpy.isclose(getattr(pboid2, error_fn)(x, y, z), 1.0)


def test_peak():
    pboid = Paraboloid(vertex2std(A=-1, B=-1, C=0, xv=3.5, yv=-0.2))
    assert numpy.allclose(pboid.peak, [3.5, -0.2, 0])


@pytest.mark.parametrize(
    "coeff, bounds, expected",
    (
        (
            vertex2std(A=-1, B=-1, C=0, xv=10, yv=5),
            [0, 10, 0, 10],
            0.0,
        ),
        (
            vertex2std(A=-1, B=-1, C=0, xv=10, yv=5),
            [0, 10, -5, 5],
            numpy.deg2rad(45),
        ),
        (
            vertex2std(A=-1, B=-1, C=0, xv=0, yv=0),
            [0, 60, 0, 60],
            numpy.deg2rad(-135),
        ),
    ),
)
def test_angle(coeff, bounds, expected):
    pboid = Paraboloid(coeff)
    assert numpy.isclose(pboid.angle(bounds), expected)


def test_vertex2std():

    A, B, C, xv, yv = (-0.3, -1.5, 54.2, 10.5, -60)
    a, b, c, d, e, f = vertex2std(A, B, C, xv, yv)

    # Test at a few random points
    X = [-1.5, 10, 3.1]
    Y = [7.3, 1.6, 42.9]

    for x, y in zip(X, Y):
        z_vert = A * (x - xv) ** 2 + B * (y - yv) ** 2 + C
        z_std = a * x**2 + b * y**2 + c * x * y + d * x + e * y + f
        assert numpy.isclose(z_vert, z_std)
