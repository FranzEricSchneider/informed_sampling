"""
Provide a tool to produce data on sample spots for these paraboloids.
"""

import ast
import argparse
from matplotlib import pyplot
import numpy
import pandas
from pathlib import Path
import yaml

from informed_sampling.paraboloid import Paraboloid


def get_top_n(df, N, column):
    return df.groupby(["pattern", "file_index"])[column].mean().nlargest(N)


def get_last_n(df, N, column):
    return df.groupby(["pattern", "file_index"])[column].mean().nsmallest(N)


def cdf(axis, vector, label="", percentiles=None):
    x = numpy.sort(vector)
    y = numpy.linspace(0, 1, len(vector))
    axis.plot(x, y, lw=2, label=label)
    if percentiles is not None:
        for percentile in percentiles:
            axis.axvline(numpy.percentile(vector, percentile), color="r", linestyle="dashed")


def full_mae_cdf(
    dfs,
    show=True,
    figure=None,
    axis=None,
    label="",
    title="Histogram of the average MAE value over all patterns",
    percentile=95,
):
    """
    Take in results dfs, group by pattern, and histogram the average MAE.
    """

    errors = []
    for df in dfs:
        errors.extend(df.groupby("pattern")["mae"].mean().tolist())
    errors = numpy.array(errors)

    # Remove outliers
    cutoff = numpy.percentile(errors, percentile)
    filtered = errors[errors <= cutoff]

    if figure is None or axis is None:
        figure, axis = pyplot.subplots(figsize=(8, 6))

    cdf(axis, filtered, label)
    axis.set_xlabel("MAE")
    axis.set_ylabel("Counts")
    axis.set_title(title)
    axis.grid()
    figure.tight_layout()
    if show:
        pyplot.show()


def plot_distributions(row, bounds):
    """
    Plot the true and fit data distributions
    """
    figure, axes = pyplot.subplots(1, 2, figsize=(8, 6))
    true = Paraboloid(
        coeff=[row.true_a, row.true_b, row.true_c, row.true_d, row.true_e, row.true_f]
    )
    fit = Paraboloid(
        coeff=[row.fit_a, row.fit_b, row.fit_c, row.fit_d, row.fit_e, row.fit_f]
    )
    true.plot2d(
        *bounds,
        figure=figure,
        axis=axes[0],
        show=False,
        title=f"True Distribution (integral: {true.integrate(bounds):.1f})",
    )
    fit.plot2d(
        *bounds,
        figure=figure,
        axis=axes[1],
        show=False,
        title=f"Fit Distribution (integral: {fit.integrate(bounds):.1f}, MAE: {row.mae:.1f})",
    )
    figure.tight_layout()
    pyplot.show()


def full_sample_hist(dfs):

    samples = []
    for df in dfs:
        for col in df.columns:
            if "sample" in col:
                samples.extend(df[col].tolist())

    figure, axis = pyplot.subplots()

    mean = numpy.mean(samples)

    pyplot.hist(samples, bins=50)
    axis.axvline(mean, color="r", linestyle="dashed")
    pyplot.xlabel("Sample Values")
    pyplot.ylabel("Counts")
    pyplot.title(f"Histogram of the raw sample values (mean: {mean:.2f})")
    pyplot.tight_layout()
    pyplot.show()


def n_dists(metadatas, result_dfs, N):
    i = 0
    for meta, df in zip(metadatas, result_dfs):
        for row in df.itertuples(index=False):
            plot_distributions(row, meta["bounds"])
            i += 1
            if i >= N:
                return


def sample_patterns(metadatas, result_dfs, patterns, N):

    all_results = pandas.concat(result_dfs)
    worst = get_top_n(all_results, N, "mae")
    best = get_last_n(all_results, N, "mae")

    figure = pyplot.figure(figsize=(6, 6))

    # Build up what we need to plot (points and colors)
    plot = []

    def get_points(cmap, source, vmin=0, vmax=1, edge="k"):
        for ((pattern, file_index), value), rgba in zip(
            source.items(),
            cmap_iterator(cmap, N, vmin, vmax),
        ):
            num_samples = metadatas[file_index]["num_samples"]
            pattern_dict = patterns[file_index]
            plot.append(
                (
                    numpy.array(
                        [
                            pattern_dict[pattern][f"sample_{i:02}"]
                            for i in range(num_samples)
                        ]
                    ),
                    rgba,
                    edge,
                    f"File index: {file_index}",
                )
            )

    get_points(source=worst, cmap="autumn", vmax=0.75, edge="r")
    get_points(source=best, cmap="Greens", vmin=0.4, vmax=0.8, edge="g")

    for points, color, edge, label in plot:
        color = color[:3] + (0.9,)
        pyplot.scatter(points[:, 0], points[:, 1], color=color, edgecolor=edge, s=55)
        # Sort by closeness to the center and do a weird splat line plot to
        # show connectivity
        points = numpy.array(
            sorted(
                points, key=lambda x: numpy.linalg.norm(x - numpy.mean(points, axis=0))
            )
        )
        pyplot.plot(
            sum(
                [[points[0, 0], points[i, 0]] for i in range(1, len(points))], start=[]
            ),
            sum(
                [[points[0, 1], points[i, 1]] for i in range(1, len(points))], start=[]
            ),
            color=color,
            label=label,
        )
    pyplot.legend()
    pyplot.show()


def cmap_iterator(name, N, vmin=0, vmax=1):
    """Yields RGBA"""
    cmap = pyplot.get_cmap(name)
    for i in numpy.linspace(vmin, vmax, N):
        yield cmap(i)


def by_sample_num(metadatas, result_dfs):
    unique = sorted(numpy.unique([m["num_samples"] for m in metadatas]))
    y = []
    for value in unique:
        all_results = pandas.concat(
            [
                df
                for i, df in enumerate(result_dfs)
                if metadatas[i]["num_samples"] == value
            ]
        )
        for mae in get_last_n(all_results, 1, "mae"):
            y.append(mae)

    figure, axis = pyplot.subplots(1, 1, figsize=(5, 5))
    axis.bar(unique, y)
    axis.set_xlabel("Number of samples")
    axis.set_ylabel("Best average MAE for a pattern")
    axis.set_xticks(unique)
    pyplot.show()


def by_strategy(metadatas, result_dfs):

    unique = sorted(numpy.unique([m["strategy"] for m in metadatas]))
    figure, axis = pyplot.subplots(1, 1, figsize=(6, 6))

    for value in unique:
        full_mae_cdf(
            dfs=[
                df
                for i, df in enumerate(result_dfs)
                if metadatas[i]["strategy"] == value
            ],
            figure=figure,
            axis=axis,
            show=False,
            label=value,
            title=f"All MAE for sampling strategy {value}",
        )
    figure.legend()
    pyplot.show()


def metamasks(metadatas, include_flag, exclude_flag):

    # List the allowable filter candidates
    datatypes = {
        "num_samples": int,
        "strategy": str,
    }

    if include_flag is None:
        include = list(range(len(metadatas)))
    else:
        options = [opt.split(":") for opt in include_flag]
        keys = [opt[0] for opt in options]
        values = [datatypes[opt[0]](opt[1]) for opt in options]
        include = [
            i
            for i, meta in enumerate(metadatas)
            if any([meta[k] == v for k, v in zip(keys, values)])
        ]

    if exclude_flag is None:
        exclude = []
    else:
        options = [opt.split(":") for opt in exclude_flag]
        keys = [opt[0] for opt in options]
        values = [datatypes[opt[0]](opt[1]) for opt in options]
        exclude = [
            i
            for i, meta in enumerate(metadatas)
            if any([meta[k] == v for k, v in zip(keys, values)])
        ]

    return [i for i in include if i not in exclude]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "results_paths",
        help="Space separated paths to results files",
        type=Path,
        nargs="+",
    )
    parser.add_argument(
        "-N",
        "--top-N",
        help="Take the top N results and display them, or -1 for all",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--full-mae-cdf",
        help="Histogram of all error values",
        action="store_true",
    )
    parser.add_argument(
        "--full-sample-hist",
        help="Histogram of the raw sample values",
        action="store_true",
    )
    parser.add_argument(
        "--n-dists",
        help="Plot N of the data distributions",
        action="store_true",
    )
    parser.add_argument(
        "--sample-patterns",
        help="Plot N best and worst sample patterns",
        action="store_true",
    )
    parser.add_argument(
        "--by-sample-num",
        help="Plot error of best pattern by number of samples",
        action="store_true",
    )
    parser.add_argument(
        "--by-strategy",
        help="Plot error histogram by sampling strategy",
        action="store_true",
    )
    parser.add_argument(
        "--exclude",
        help="Metadata values we will avoid, written as key:value",
        nargs="+",
    )
    parser.add_argument(
        "--include",
        help="Metadata values we will include and avoid all else, written as key:value",
        nargs="+",
    )
    args = parser.parse_args()

    # Check that all results exist and have corresponding patterns
    pattern_paths = [
        Path(str(rpath).replace("results", "patterns").replace("csv", "yaml"))
        for rpath in args.results_paths
    ]
    metadata_paths = [
        Path(str(rpath).replace("results", "metadata").replace("csv", "yaml"))
        for rpath in args.results_paths
    ]
    for path in args.results_paths + pattern_paths + metadata_paths:
        assert path.is_file(), f"{path} not found"

    result_dfs = [pandas.read_csv(path) for path in args.results_paths]
    for i, df in enumerate(result_dfs):
        df["file_index"] = i
    patterns = [yaml.safe_load(path.open("r")) for path in pattern_paths]
    metadatas = [yaml.safe_load(path.open("r")) for path in metadata_paths]

    # Filter by include and exclude
    include = metamasks(metadatas, args.include, args.exclude)
    result_dfs = [result_dfs[i] for i in include]
    patterns = [patterns[i] for i in include]
    metadatas = [metadatas[i] for i in include]

    for i, path in enumerate(metadata_paths):
        if i in include:
            print(f"File index: {i} = {path}")

    if args.full_mae_cdf:
        full_mae_cdf(result_dfs)
    if args.full_sample_hist:
        full_sample_hist(result_dfs)
    if args.n_dists:
        n_dists(metadatas, result_dfs, args.top_N)
    if args.sample_patterns:
        sample_patterns(metadatas, result_dfs, patterns, args.top_N)
    if args.by_sample_num:
        by_sample_num(metadatas, result_dfs)
    if args.by_strategy:
        by_strategy(metadatas, result_dfs)
