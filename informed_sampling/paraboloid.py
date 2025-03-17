import numpy
from scipy import integrate
from sklearn.metrics import r2_score


class Paraboloid:
    """
    Class with some basic functions to generate and assess 3D paraboloids.
    """

    def __init__(self, coeff, data=None):
        """
        Coefficients should come in the form
            a*x^2 + b*y^2 + c*xy + d*x + e*y + f

        If given, data should come in the form (x, y, z) = (N, 3)
        """
        assert len(coeff) == 6, f"6 elements required, got {len(coeff)}"
        for value in coeff:
            assert isinstance(value, (int, float)), f"Number required, got {value}"
        self.coeff = coeff
        self.data = data

    def __repr__(self):
        a, b, c, d, e, f = self.coeff
        return f"{a:.3f}x² + {b:.3f}y² + {c:.3f}xy + {d:.3f}x + {e:.3f}y + {c:.3f}"

    @classmethod
    def generate(cls, x_mean, x_std, y_mean, y_std, bounds, slope, values):
        """
        Generate a downward facing paraboloid with limits on slope and integral
        within a defined area.

        The vertex form of a paraboloid is useful because you know the (X, Y)
        position of the peak at the offset.
            z = A(x - xv)^2 + B(y - yv)^2 + C
        A, B, C, xv, and xy are the parameters, x and y are the independent
        variables.

        The gradient of vertex form is
            [
                2A(x - xv),
                2B(y - yv),
            ]

        NOTE: Assumes both squared components will be negative

        Arguments:
            x_mean: Mean for a normal distribution that will be used to pick
                the X position of the peak
            x_std: Std seviation for a normal distribution that will be used to
                pick the X position of the peak
            y_mean: Mean for a normal distribution that will be used to pick
                the Y position of the peak
            y_std: Std seviation for a normal distribution that will be used to
                pick the Y position of the peak
            bounds: Numpy array of (xmin, ymax, ymin, ymax) defining an area
                where the slope will be limited and the integral will be
                calculated
            slope: Maximum gradient magnitude allowed within the bounds (given
                positive, though the slope will be negative)
            values: Tuple of (min, max) integral over the bounds that will
                be allowed
        """

        # TODO: Add in rotation

        # Pick the vertex locations
        xv = numpy.random.normal(loc=x_mean, scale=x_std)
        yv = numpy.random.normal(loc=y_mean, scale=y_std)

        # Pick the X slope first (arbitrary), bounded only by the max slope
        #   2A(x - xv) ≤ slope
        xΔ = max(numpy.abs(bounds[:2] - xv))
        maxA = slope / (2 * xΔ)
        A = numpy.random.uniform(low=-maxA, high=0)

        # Now pick the Y slope, knowing that the X value will constrain the
        # Y value because we have a cap on the total gradient
        #   sqrt([2A(x - xv)]^2 + [2B(y - yv)]^2) ≤ slope
        #   ...
        #   B ≤ sqrt(slope^2 - [2A(x - xv)]^2) / 2(y - yv)
        yΔ = max(numpy.abs(bounds[2:] - yv))
        maxB = numpy.sqrt(slope**2 - (2 * A * xΔ) ** 2) / (2 * yΔ)
        B = numpy.random.uniform(low=-maxB, high=0)

        # Get the base integral and scale off that
        base = Paraboloid(coeff=vertex2std(A, B, 0, xv, yv)).integrate(bounds)
        area = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])
        # The integral should fall between values[0] and values[1]
        intΔ = numpy.random.uniform(low=values[0] - base, high=values[1] - base)
        # The offset is scaled by the area
        C = intΔ / area

        return cls(coeff=vertex2std(A, B, C, xv, yv))

    # TODO: Are there ways to bake in extra real-world knowledge here, like
    # limits on the slope or symmetry?
    @classmethod
    def fit(cls, x, y, z):
        """Inputs should be (N,) arrays, at least 6"""

        assert len(x) >= 6, "At least 6 points needed for full fit"

        # Construct the design matrix for least squares fitting
        A = numpy.column_stack([x**2, y**2, x * y, x, y, numpy.ones_like(x)])

        # Solve the least squares problem
        coefficients, _, _, _ = numpy.linalg.lstsq(A, z, rcond=None)

        return cls(coefficients, numpy.vstack([x, y, z]))

    def predict(self, x, y):
        """
        Take numpy arrays for x and y, shape must match. Apply the paraboloid.
        """
        a, b, c, d, e, f = self.coeff
        return paraboloid(x=x, y=y, a=a, b=b, c=c, d=d, e=e, f=f)

    def predict_grid(self, bounds, spacing=1):
        """
        Given bounds and spacing, return prediction values on a grid.

        x, y, and z will be returned each as (M,) arrays, where M = L * N
        where L and N are the grid size.
        """
        # Simple range
        x = numpy.arange(bounds[0], bounds[1], spacing, dtype=float)
        y = numpy.arange(bounds[2], bounds[3], spacing, dtype=float)
        # Gridify
        X, Y = numpy.meshgrid(x, y)
        # Flatten
        X = X.flatten()
        Y = Y.flatten()
        return X, Y, self.predict(X, Y)

    def gradient(self, x, y):
        """
        Returns [dz/dx, dz/dy] at an (x, y) point or a set of (x, y) points.

        If you give x and y as (N,) arrays, you will recieve a (2, N) array.
        """
        a, b, c, d, e, f = self.coeff
        return numpy.array(
            [
                2 * a * x + c * y + d,
                2 * b * y + c * x + e,
            ]
        )

    def integrate(self, bounds):
        """
        Numerically integrate over the bounds. I know there's a closed form
        paraboloid solution, but this is accurate enough and simpler to write.
        """
        x_min, x_max, y_min, y_max = bounds
        result, error = integrate.dblquad(
            func=lambda x, y: paraboloid(x, y, *self.coeff),
            a=y_min,
            b=y_max,
            gfun=lambda y: x_min,
            hfun=lambda y: x_max,
        )
        return result

    def peak(self):
        """Return the (x, y, z) peak of this paraboloid."""
        a, b, c, d, e, f = self.coeff
        denom = 4 * a * b - c**2
        x = ((-d * 2 * b) + (e * c)) / denom
        y = ((-e * 2 * a) + (d * c)) / denom
        z = self.predict(x, y)
        return numpy.array([x, y, z])

    def r2_score(self):
        """
        Return the R^2 value for this paraboloid if there is corresponding
        data to compare to.
        """
        if self.data is None:
            return None
        else:
            # Actual
            x, y, z = self.data
            # Predicted
            z_predicted = self.predict(x, y)
            # Calculate
            return r2_score(z, z_predicted)

    def mae(self, x, y, z):
        """
        Mean absolute error of predicted on (x, y) vs. given z if data
        exists.
        """
        z_predicted = self.predict(x, y)
        return numpy.mean(numpy.abs(z - z_predicted))

    def rmse(self, x, y, z):
        """
        Root mean swuared error of predicted on (x, y) vs. given z if data
        exists.
        """
        z_predicted = self.predict(x, y)
        return numpy.sqrt(numpy.mean((z - z_predicted) ** 2))

    def plot2d(self, xmin, xmax, ymin, ymax, save_path=None):
        from matplotlib import pyplot

        x = numpy.linspace(xmin, xmax, 50)
        y = numpy.linspace(ymin, ymax, 50)
        X, Y = numpy.meshgrid(x, y)
        Z = self.predict(X, Y)

        # Create contour plot
        figure = pyplot.figure(figsize=(8, 6))
        contour = pyplot.contourf(X, Y, Z, levels=30, cmap="jet")
        pyplot.colorbar(contour, label="Z value")

        # Plot original data points
        if self.data is not None:
            pyplot.scatter(
                self.data[0],
                self.data[1],
                color="red",
                edgecolors="black",
                label="Data Points",
            )
            pyplot.legend()

        # Labels and title
        pyplot.xlabel("X")
        pyplot.ylabel("Y")
        pyplot.title("Contour Plot of Fitted 3D Paraboloid")
        pyplot.grid(True)
        figure.tight_layout()

        if save_path is None:
            pyplot.show()
        else:
            figure.savefig(save_path)

    def plot3d(self, xmin, xmax, ymin, ymax, save_path=None):
        from matplotlib import pyplot
        from mpl_toolkits.mplot3d import Axes3D

        # Create 3D plot
        figure = pyplot.figure(figsize=(10, 7))
        axis = figure.add_subplot(111, projection="3d")

        x = numpy.linspace(xmin, xmax, 50)
        y = numpy.linspace(ymin, ymax, 50)
        X, Y = numpy.meshgrid(x, y)
        Z = self.predict(X, Y)

        # Scatter plot of the original data
        if self.data is not None:
            axis.scatter(
                self.data[0],
                self.data[1],
                self.data[2],
                color="red",
                label="Data Points",
            )

        # Plot the fitted paraboloid
        axis.plot_surface(X, Y, Z, color="blue", alpha=0.5, edgecolor="none")

        # Labels and legend
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
        axis.set_title("3D Paraboloid Fit")
        axis.legend()

        if save_path is None:
            pyplot.show()
        else:
            figure.savefig(save_path)


def vertex2std(A, B, C, xv, yv):
    """
    Convert vertex form coefficients of a paraboloid to standard form.

    Vertex form: A(x - xv)^2 + B(y - yv)^2 + C = z
    Standard form: ax^2 + by^2 + cxy + dx + ey + f = z
    """
    a = A
    b = B
    c = 0
    d = -2 * A * xv
    e = -2 * B * yv
    f = A * xv**2 + B * yv**2 + C
    return a, b, c, d, e, f


def paraboloid(x, y, a, b, c, d, e, f):
    """Define out of class for SciPy integration use."""
    return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f
