import numpy
from scipy import integrate
from sklearn.metrics import r2_score
from scipy.optimize import bisect


class Paraboloid:
    """
    Class with some basic functions to generate and assess 3D paraboloids.

    Importantly, any values below 0 are clipped to be 0.
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

        # Get the base integral and scale off that. Check this in a loop
        # because of the non-linearities clipping at 0 lead to some
        # underestimations
        low, high = search_C_bounds(
            integral=lambda C: Paraboloid(coeff=vertex2std(A, B, C, xv, yv)).integrate(
                bounds
            ),
            integral_lims=values,
            bounds=bounds,
        )
        C = numpy.random.uniform(low=low, high=high)

        return cls(coeff=vertex2std(A, B, C, xv, yv))

    # TODO: Are there ways to bake in extra real-world knowledge here, like
    # limits on the slope or symmetry?
    @classmethod
    def fit(cls, x, y, z, strategy="all"):
        """Inputs should be (N,) arrays, at least 6"""

        required = 4
        if strategy == "all":
            pass
        elif strategy == "nonzero":
            mask = z > 0
            if sum(mask) >= required:
                x = x[mask]
                y = y[mask]
                z = z[mask]
        else:
            raise NotImplementedError(f"Unknown: {strategy}")

        assert len(x) >= required, f"At least {required} points needed for full fit"

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

    # # Deprecating due to the non-linearity (zeroing) added to paraboloid.
    # # Keeping for a bit even though it's bad form just in case
    # def integrate(self, bounds):
    #     """
    #     Numerically integrate over the bounds. I know there's a closed form
    #     paraboloid solution, but this is accurate enough and simpler to write.
    #     """
    #     x_min, x_max, y_min, y_max = bounds
    #     result, error = integrate.dblquad(
    #         func=lambda x, y: paraboloid(x, y, *self.coeff),
    #         # X limits (why do these need to be flipped to test correctly?)
    #         a=y_min,
    #         b=y_max,
    #         # Y limits as functions of X
    #         gfun=lambda y: x_min,
    #         hfun=lambda y: x_max,
    #     )
    #     return result

    def integrate(self, bounds, N=50):
        """Do grid-based numerical integration b/c of non-linearities"""
        # Create a grid of x and y values
        x = numpy.linspace(bounds[0], bounds[1], N)
        y = numpy.linspace(bounds[2], bounds[3], N)
        # Evaluate z on the grid
        Z = paraboloid(*numpy.meshgrid(x, y, indexing="ij"), *self.coeff)
        # Integrate along both x and y axes
        return integrate.simpson(integrate.simpson(Z, x=y, axis=1), x=x, axis=0)

    @property
    def peak(self):
        """Return the (x, y, z) peak of this paraboloid."""
        a, b, c, d, e, f = self.coeff
        denom = 4 * a * b - c**2
        x = ((-d * 2 * b) + (e * c)) / denom
        y = ((-e * 2 * a) + (d * c)) / denom
        z = self.predict(x, y)
        return numpy.array([x, y, z])

    def angle(self, bounds):
        """
        Return the angle (rad) of the peak from the center of the bounds,
        where angle=0 is when the peak is +X from the center.
        """
        center = numpy.array(
            [
                numpy.mean([bounds[0], bounds[1]]),
                numpy.mean([bounds[2], bounds[3]]),
            ]
        )
        vector = self.peak[:2] - center

        x_axis = numpy.array([1, 0])
        y_axis = numpy.array([0, 1])

        return numpy.arctan2(vector.dot(y_axis), vector.dot(x_axis))

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

    def plot2d(
        self,
        xmin,
        xmax,
        ymin,
        ymax,
        figure=None,
        axis=None,
        save_path=None,
        show=True,
        title="Contour Plot of Fitted 3D Paraboloid",
    ):
        from matplotlib import pyplot

        x = numpy.linspace(xmin, xmax, 50)
        y = numpy.linspace(ymin, ymax, 50)
        X, Y = numpy.meshgrid(x, y)
        Z = self.predict(X, Y)

        if figure is None or axis is None:
            figure, axis = pyplot.subplots(figsize=(8, 6))

        # Create contour plot
        contour = axis.contourf(X, Y, Z, levels=30, cmap="jet")
        figure.colorbar(contour, ax=axis, label="Z value")

        # Plot original data points
        if self.data is not None:
            axis.scatter(
                self.data[0],
                self.data[1],
                color="red",
                edgecolors="black",
                label="Data Points",
            )
            axis.legend()

        # Labels and title
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_title(title)
        axis.grid(True)
        figure.tight_layout()

        if save_path is None:
            if show:
                pyplot.show()
        else:
            figure.savefig(save_path)

        return figure, axis

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


def search_C_bounds(integral, integral_lims, bounds):

    # Use the area to make a reasonable step size
    area = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])
    step = 0.5 * (integral_lims[1] - integral_lims[0]) / area

    # Start off in the middle our guess for a valid C value
    value = (numpy.mean(integral_lims) - integral(0)) / area
    integrated = integral(value)

    args = (integral, value, integrated)
    if integrated > integral_lims[1]:
        hi_x = value
        lo_x = search_C_edge(*args, integral_lims[0], -step)
    elif integrated < integral_lims[0]:
        lo_x = value
        hi_x = search_C_edge(*args, integral_lims[1], step)
    else:
        lo_x = search_C_edge(*args, integral_lims[0], -step)
        hi_x = search_C_edge(*args, integral_lims[1], step)

    # Helper functions for bisect to find zeros
    def find_x_lo(x):
        return integral(x) - integral_lims[0]

    def find_x_hi(x):
        return integral(x) - integral_lims[1]

    # Use bisect to search for the point where x (C value) causes the integral
    # to cross the given limit
    bound_x_lo = bisect(find_x_lo, lo_x, hi_x, xtol=1e-3)
    bound_x_hi = bisect(find_x_hi, lo_x, hi_x, xtol=1e-3)

    return (bound_x_lo, bound_x_hi)


def search_C_edge(integral, init_x, init_y, cross, step, root=1.5):

    # Set some sort of limit for the search
    for i in range(20):
        check_x = init_x + root**i * step
        check_y = integral(check_x)
        if (init_y < cross < check_y) or (init_y > cross > check_y):
            return check_x
    else:
        raise RuntimeError("Couldn't find a solution")


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
    return numpy.clip(
        a * x**2 + b * y**2 + c * x * y + d * x + e * y + f,
        0,
        numpy.inf,
    )
