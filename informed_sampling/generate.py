"""
Provide a tool to produce data on sample spots for these paraboloids.
"""

import argparse
import numpy
import pandas
import time
from tqdm import tqdm

from informed_sampling.paraboloid import Paraboloid


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
    noise_std,
):

    # Store the data as dictionaries in a list with shared keys, as this is
    # easy to turn into a DataFrame
    patterns = []
    results = []

    for pattern_idx in tqdm(range(assess_j)):
        # Sample X and Y locations that we want to sample the measurements
        # from on the known distribution
        x = numpy.random.uniform(
            low=bounds[0] + sample_buffer,
            high=bounds[1] + sample_buffer,
            size=num_samples,
        )
        y = numpy.random.uniform(
            low=bounds[2] + sample_buffer,
            high=bounds[3] + sample_buffer,
            size=num_samples,
        )
        pattern = {"pattern": pattern_idx}
        for i, xy in enumerate(zip(x, y)):
            pattern[f"sample_{i:02}"] = numpy.array(xy).tolist()
        patterns.append(pattern)

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

            # Calculate noise on the measurement as a fraction around 1.0
            noise = numpy.random.normal(
                loc=1.0,
                scale=noise_std,
                size=num_samples,
            )
            # Add the noise to the measurements
            z = pboid.predict(x, y) * noise
            fit_pboid = Paraboloid.fit(x, y, z)

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
        f"/tmp/parabola_results_{timestamp}.csv", index=False
    )
    pandas.DataFrame(patterns).to_csv(
        f"/tmp/parabola_patterns_{timestamp}.csv", index=False
    )
    print(f"Saved run to /tmp/parabola_*_{timestamp}.csv")


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
    args = parser.parse_args()

    assert args.xmax > args.xmin
    assert args.ymax > args.ymin
    assert args.intmax > args.intmin
    assert args.max_slope > 0
    assert args.sample_buffer >= 0

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
        noise_std=args.noise_std,
    )
