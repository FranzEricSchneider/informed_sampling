"""
Provide a tool to produce data on sample spots for these paraboloids.
"""

import argparse
import numpy
import pandas
from scipy.stats import qmc
import time
from tqdm import tqdm
import yaml

from informed_sampling.paraboloid import Paraboloid


# Make some pre-made grids as fractions of the available space. The points are
# in no particular order
GRIDS = {
    4: numpy.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]),
    5: numpy.array([[0.5, 0], [0, 0.5], [0.5, 0.5], [1, 0.5], [0.5, 1]]),
    6: numpy.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.3, 0.5], [0.7, 0.5]]),
    7: numpy.array(
        [[0.33, 0], [0.66, 0], [0, 0.5], [0.5, 0.5], [1, 0.5], [0.33, 1], [0.66, 1]]
    ),
    8: numpy.array(
        [[0, 0], [0.5, 0], [1, 0], [0.25, 0.5], [0.75, 0.5], [0, 1], [0.5, 1], [1, 1]]
    ),
    9: numpy.array(
        [
            [0, 0],
            [0.5, 0],
            [1, 0],
            [0, 0.5],
            [0.5, 0.5],
            [1, 0.5],
            [0, 1],
            [0.5, 1],
            [1, 1],
        ]
    ),
    10: numpy.array(
        [
            [0, 0],
            [0.5, 0],
            [1, 0],
            [0.25, 0.33],
            [0.75, 0.33],
            [0.25, 0.66],
            [0.75, 0.66],
            [0, 1],
            [0.5, 1],
            [1, 1],
        ]
    ),
}


def random_sample(bounds, buffer, N):
    x = numpy.random.uniform(
        low=bounds[0] + buffer,
        high=bounds[1] - buffer,
        size=N,
    )
    y = numpy.random.uniform(
        low=bounds[2] + buffer,
        high=bounds[3] - buffer,
        size=N,
    )
    return x, y


def poisson_sample(bounds, buffer, N):
    points = qmc.scale(
        # 2-dimensional sampler
        qmc.LatinHypercube(d=2).random(n=N),
        [bounds[0] + buffer, bounds[2] + buffer],
        [bounds[1] - buffer, bounds[3] - buffer],
    )
    return points[:, 0], points[:, 1]


def grid_sample(bounds, buffer, N):

    base = GRIDS[N].T
    x_range = bounds[1] - bounds[0] - 4 * buffer
    y_range = bounds[3] - bounds[2] - 4 * buffer
    x = base[0] * x_range + bounds[0] + 2 * buffer
    y = base[1] * y_range + bounds[2] + 2 * buffer

    # Add in some small (capped) local noise
    x += numpy.random.normal(loc=0, scale=0.5 * buffer, size=len(x))
    x = numpy.clip(x, bounds[0] + buffer, bounds[1] - buffer)
    y += numpy.random.normal(loc=0, scale=0.5 * buffer, size=len(y))
    y = numpy.clip(y, bounds[0] + buffer, bounds[1] - buffer)

    return x, y


def circle_sample(bounds, buffer, N):
    """
    Like Poisson, but we reject points that aren't in a circle around the
    center of the bounds.
    """
    center = numpy.array([numpy.mean(bounds[:2]), numpy.mean(bounds[2:])])
    radius = min([bounds[1], bounds[3]]) / 2

    done = False
    while not done:
        x, y = poisson_sample(bounds, buffer, N * 2)
        dist = numpy.linalg.norm(numpy.vstack([x, y]).T - center, axis=1)
        mask = dist < radius
        if sum(mask) >= N:
            done = True
            where = numpy.where(mask)[0]
            x = numpy.array([x[i] for i in where[:N]])
            y = numpy.array([y[i] for i in where[:N]])

    return x, y


def main(
    x_mean,
    x_std,
    y_mean,
    y_std,
    bounds,
    max_slope,
    integral_values,
    gen_i,
    assess_j,
    sample_buffer,
    num_samples,
    strategy,
    sampler,
    fit_strategy,
    augmentation,
    noise_std,
):

    # Store the data as dictionaries in a list with shared keys, as this is
    # easy to turn into a DataFrame
    patterns = {}
    results = []
    metadata = {
        "bounds": bounds.tolist(),
        "max_slope": max_slope,
        "integral_values": integral_values,
        "num_samples": num_samples,
        "noise_std": noise_std,
        "strategy": strategy,
        "fit_strategy": fit_strategy,
        "augmentation": augmentation,
    }

    for pattern_idx in tqdm(range(assess_j)):
        # Sample X and Y locations that we want to sample the measurements
        # from on the known distribution
        x, y = sampler(bounds, sample_buffer, num_samples)
        pattern = {}
        for i, xy in enumerate(zip(x, y)):
            pattern[f"sample_{i:02}"] = numpy.array(xy).tolist()
        patterns[pattern_idx] = pattern

        for _ in range(gen_i):

            pboid = Paraboloid.generate(
                x_mean=x_mean,
                x_std=x_std,
                y_mean=y_mean,
                y_std=y_std,
                bounds=bounds,
                slope=max_slope,
                values=integral_values,
            )
            # Load the assessment points on a grid
            true_xyz = pboid.predict_grid(
                bounds=bounds, spacing=(bounds[1] - bounds[0]) / 10
            )

            if augmentation == "none":
                aug_x = x.copy()
                aug_y = y.copy()
            elif augmentation == "face-peak":
                aug_x, aug_y = rotate(x, y, bounds, pboid.angle(bounds))
            else:
                raise NotImplementedError(augmentation)

            # Calculate noise on the measurement as a fraction around 1.0
            noise = numpy.random.normal(
                loc=1.0,
                scale=noise_std,
                size=num_samples,
            )
            # Add the noise to the measurements
            z = pboid.predict(aug_x, aug_y) * noise
            fit_pboid = Paraboloid.fit(aug_x, aug_y, z, fit_strategy)

            # Build up our assessment data
            result = {
                "pattern": pattern_idx,
                "mae": fit_pboid.mae(*true_xyz),
                "rmse": fit_pboid.rmse(*true_xyz),
            }
            for name, coeff in zip("abcdef", map(float, pboid.coeff)):
                result[f"true_{name}"] = coeff
            for name, coeff in zip("abcdef", map(float, fit_pboid.coeff)):
                result[f"fit_{name}"] = coeff
            for i, sz in enumerate(map(float, z)):
                result[f"sample_{i:02}"] = sz
            results.append(result)

    timestamp = int(time.time() * 1e6)
    pandas.DataFrame(results).to_csv(
        f"/tmp/parabola_results_{timestamp}.csv",
        index=False,
    )
    yaml.dump(patterns, open(f"/tmp/parabola_patterns_{timestamp}.yaml", "w"))
    yaml.dump(metadata, open(f"/tmp/parabola_metadata_{timestamp}.yaml", "w"))
    print(f"Saved run to /tmp/parabola_*_{timestamp}.csv/yaml")


def rotate(x, y, bounds, angle):

    # Center xy, temporarily
    center = numpy.array(
        [
            numpy.mean([bounds[0], bounds[1]]),
            numpy.mean([bounds[2], bounds[3]]),
        ]
    ).reshape(2, 1)
    xy = numpy.vstack([x, y])
    centered = xy - center

    # Make a rotation matrix and apply to the centered points
    R = numpy.array(
        [[numpy.cos(angle), -numpy.sin(angle)], [numpy.sin(angle), numpy.cos(angle)]]
    )
    rotated = R @ centered

    # Add the center offset back in and return
    reset = rotated + center
    return reset[0], reset[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "xmin",
        help="X min for the bounded area we care about",
        type=float,
    )
    parser.add_argument(
        "xmax",
        help="X max for the bounded area we care about",
        type=float,
    )
    parser.add_argument(
        "ymin",
        help="Y min for the bounded area we care about",
        type=float,
    )
    parser.add_argument(
        "ymax",
        help="Y max for the bounded area we care about",
        type=float,
    )
    parser.add_argument(
        "intmin",
        help="Integral min for the bounded area we care about",
        type=float,
    )
    parser.add_argument(
        "intmax",
        help="Integral max for the bounded area we care about",
        type=float,
    )
    parser.add_argument(
        "-i",
        "--generation-iterations",
        help="How many paraboloids to generate",
        default=100,
        type=int,
    )
    parser.add_argument(
        "-j",
        "--assess-iterations",
        help="How many sampling points to try",
        default=100,
        type=int,
    )
    parser.add_argument(
        "-s",
        "--max-slope",
        help="Maximum gradient we will allow in the generated paraboloids",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "-b",
        "--sample-buffer",
        help="Buffer at the edge bounds that we want to stay within",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        help="Number of samples to take when fitting another paraboloid",
        default=10,
        type=int,
    )
    parser.add_argument(
        "-N",
        "--noise-std",
        help="Std dev of noise added onto the measurements, as a fraction. So"
        " 0.1 means that the measurement will get multiplied by a number"
        " sampled around 1.0, with 0.1 as the std dev.",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "-S",
        "--sampler-strategy",
        help="Which sampling strategy to use",
        choices=["random", "poisson", "grid", "circle"],
        default="random",
    )
    parser.add_argument(
        "-F",
        "--fit-strategy",
        help="Which point fitting strategy to use",
        choices=["all", "nonzero"],
        default="all",
    )
    parser.add_argument(
        "-A",
        "--augmentation",
        help="Possible augmentation options",
        choices=["none", "face-peak"],
        default="none",
    )
    args = parser.parse_args()

    assert args.xmax > args.xmin
    assert args.ymax > args.ymin
    assert args.intmax > args.intmin
    assert args.max_slope > 0
    assert args.sample_buffer >= 0

    sampler_map = {
        "random": random_sample,
        "poisson": poisson_sample,
        "grid": grid_sample,
        "circle": circle_sample,
    }

    main(
        x_mean=numpy.mean([args.xmin, args.xmax]),
        x_std=args.xmax - args.xmin,
        y_mean=numpy.mean([args.ymin, args.ymax]),
        y_std=args.ymax - args.ymin,
        bounds=numpy.array([args.xmin, args.xmax, args.ymin, args.ymax]),
        max_slope=args.max_slope,
        integral_values=[args.intmin, args.intmax],
        gen_i=args.generation_iterations,
        assess_j=args.assess_iterations,
        sample_buffer=args.sample_buffer,
        num_samples=args.num_samples,
        strategy=args.sampler_strategy,
        sampler=sampler_map[args.sampler_strategy],
        fit_strategy=args.fit_strategy,
        augmentation=args.augmentation,
        noise_std=args.noise_std,
    )
