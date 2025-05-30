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
            axis.axvline(
                numpy.percentile(vector, percentile), color="r", linestyle="dashed"
            )


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
    for df in dfs.values():
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
    axis.grid("on")
    figure.tight_layout()
    if show:
        pyplot.show()


def plot_distributions(row, bounds):
    """
    Plot the true and fit data distributions
    """
    figure, axes = pyplot.subplots(1, 2, figsize=(12, 6))
    true = Paraboloid(
        coeff=[row.true_a, row.true_b, row.true_c, row.true_d, row.true_e, row.true_f]
    )
    fit = Paraboloid(
        coeff=[row.fit_a, row.fit_b, row.fit_c, row.fit_d, row.fit_e, row.fit_f]
    )
    gradient = true.gradient(
        x=numpy.array([bounds[0]] * 2 + [bounds[1]] * 2),
        y=numpy.array([bounds[2], bounds[3]] * 2),
    )
    max_gradient = max(numpy.linalg.norm(gradient, axis=1))
    true.plot2d(
        *bounds,
        figure=figure,
        axis=axes[0],
        show=False,
        title=f"True Distribution (integral: {true.integrate(bounds):.1f}, max ∇ {max_gradient:.1f})",
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
    for df in dfs.values():
        for col in df.columns:
            if "sample" in col:
                samples.extend(df[col].tolist())

    figure, axis = pyplot.subplots()

    mean = numpy.mean(samples)

    pyplot.hist(samples, bins=100)
    axis.axvline(mean, color="r", linestyle="dashed")
    pyplot.xlabel("Sample Values")
    pyplot.ylabel("Counts")
    pyplot.title(f"Histogram of the raw sample values (mean: {mean:.2f})")
    pyplot.tight_layout()
    pyplot.show()


def n_dists(metadatas, result_dfs, N):
    i = 0
    for file_index in metadatas.keys():
        meta = metadatas[file_index]
        df = result_dfs[file_index]
        for row in df.itertuples(index=False):
            plot_distributions(row, meta["bounds"])
            i += 1
            if i >= N:
                return


def sample_patterns(metadatas, result_dfs, patterns, N):

    all_results = pandas.concat(result_dfs.values())
    worst = get_top_n(all_results, N, "mae")
    best = get_last_n(all_results, N, "mae")
    bounds = [v for v in metadatas.values()][0]["bounds"]

    figure, axes = pyplot.subplots(2, N, figsize=(15, 8))

    # Build up what we need to plot (points and colors)
    plot = []

    def get_points(source, cmap, row, vmin=0, vmax=1, edge="k"):
        for ((pattern, file_index), value), rgba, axis in zip(
            source.items(),
            cmap_iterator(cmap, N, vmin, vmax),
            row,
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
                    axis,
                )
            )

    get_points(source=worst, cmap="autumn", row=axes[0], vmax=0.75, edge="r")
    get_points(source=best, cmap="Greens", row=axes[1], vmin=0.4, vmax=0.8, edge="g")

    for points, color, edge, label, axis in plot:
        color = color[:3] + (0.9,)
        axis.scatter(points[:, 0], points[:, 1], color=color, edgecolor=edge, s=55)
        # Sort by closeness to the center and do a weird splat line plot to
        # show connectivity
        points = numpy.array(
            sorted(
                points, key=lambda x: numpy.linalg.norm(x - numpy.mean(points, axis=0))
            )
        )
        axis.plot(
            sum(
                [[points[0, 0], points[i, 0]] for i in range(1, len(points))], start=[]
            ),
            sum(
                [[points[0, 1], points[i, 1]] for i in range(1, len(points))], start=[]
            ),
            color=color,
        )
        axis.set_title(label)
        axis.set_xlim(bounds[:2])
        axis.set_ylim(bounds[2:])

    figure.tight_layout()
    pyplot.show()


def bar_patterns(metas, result_dfs, N):

    all_results = pandas.concat(result_dfs.values())
    worst = get_top_n(all_results, N, "mae")
    best = get_last_n(all_results, N, "mae")

    def make_key(meta):
        key = f"{meta['strategy']}-S#{meta['num_samples']}"
        if meta["augmentation"] != "none":
            key += f"-{meta['augmentation']}"
        return key

    x = list(range(2 * N + 1))
    y = numpy.hstack([worst.values, [0], best[::-1].values])
    labels = [
        make_key(metas[fi]) for series in [worst, best[::-1]] for _, fi in series.keys()
    ]
    labels.insert(N, "...")

    figure = pyplot.figure(figsize=(15, 5))
    pyplot.bar(x, y)
    pyplot.xticks(x, labels, rotation=90)
    pyplot.ylim(min(best) * 0.95, max(best) * 1.05)
    pyplot.ylabel("MAE")
    figure.tight_layout()
    pyplot.show()


def cmap_iterator(name, N, vmin=0, vmax=1):
    """Yields RGBA"""
    cmap = pyplot.get_cmap(name)
    for i in numpy.linspace(vmin, vmax, N):
        yield cmap(i)


def by_sample_num(metadatas, result_dfs, N=10):
    unique = sorted(numpy.unique([m["num_samples"] for m in metadatas.values()]))
    y = []
    for value in unique:
        all_results = pandas.concat(
            [df for i, df in result_dfs.items() if metadatas[i]["num_samples"] == value]
        )
        y.append(
            numpy.mean(
                [mae for mae in get_last_n(all_results, N, "mae")]
            )
        )

    figure, axis = pyplot.subplots(1, 1, figsize=(8, 6))
    axis.bar(unique, y)
    axis.set_xlabel("Number of samples")
    axis.set_ylabel(f"Best average MAE for the top {N} patterns")
    axis.set_title(f"Best average MAE for the top {N} patterns, by sample number")
    axis.set_xticks(unique)
    figure.tight_layout()
    pyplot.show()


def by_strategy(metadatas, result_dfs):

    unique = sorted(numpy.unique([m["strategy"] for m in metadatas.values()]))
    figure, axis = pyplot.subplots(1, 1, figsize=(8, 6))

    for value in unique:
        full_mae_cdf(
            dfs={
                i: df
                for i, df in result_dfs.items()
                if metadatas[i]["strategy"] == value
            },
            figure=figure,
            axis=axis,
            show=False,
            label=value,
        )
    axis.set_title(f"All MAE for different strategies")
    figure.legend()
    pyplot.show()


def by_fit_strategy(metadatas, result_dfs):

    unique = sorted(
        numpy.unique([m.get("fit_strategy", "all") for m in metadatas.values()])
    )
    figure, axis = pyplot.subplots(1, 1, figsize=(8, 6))

    for value in unique:
        full_mae_cdf(
            dfs={
                i: df
                for i, df in result_dfs.items()
                if metadatas[i].get("fit_strategy", "all") == value
            },
            figure=figure,
            axis=axis,
            show=False,
            label=value,
        )
    axis.set_title(f"All MAE for different fit strategies")
    figure.legend()
    pyplot.show()


def by_augmentation(metadatas, result_dfs):

    unique = sorted(
        numpy.unique([m.get("augmentation", "none") for m in metadatas.values()])
    )
    figure, axis = pyplot.subplots(1, 1, figsize=(8, 6))

    for value in unique:
        full_mae_cdf(
            dfs={
                i: df
                for i, df in result_dfs.items()
                if metadatas[i].get("augmentation", "none") == value
            },
            figure=figure,
            axis=axis,
            show=False,
            label=value,
        )
    axis.set_title(f"All MAE for different sampling augmentations")
    figure.legend()
    pyplot.show()


def metamasks(metadatas, include_flag, exclude_flag):

    # List the allowable filter candidates
    datatypes = {
        "num_samples": int,
        "strategy": str,
        "fit_strategy": str,
        "augmentation": str,
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
            if any([meta.get(k, None) == v for k, v in zip(keys, values)])
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
            if any([meta.get(k, None) == v for k, v in zip(keys, values)])
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
        default=5,
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
        "--bar-patterns",
        help="Plot N best and worst patterns by metadata key",
        action="store_true",
    )
    parser.add_argument(
        "--by-sample-num",
        help="Plot error of best pattern by number of samples",
        action="store_true",
    )
    parser.add_argument(
        "--by-strategy",
        help="Plot error CDF by sampling strategy",
        action="store_true",
    )
    parser.add_argument(
        "--by-fit-strategy",
        help="Plot error CDF by fitting strategy",
        action="store_true",
    )
    parser.add_argument(
        "--by-augmentation",
        help="Plot error CDF by augmentation",
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
    metadatas = [
        yaml.safe_load(path.open("r")) for i, path in enumerate(metadata_paths)
    ]

    # Filter by include and exclude
    include = metamasks(metadatas, args.include, args.exclude)
    result_dfs = {i: result_dfs[i] for i in include}
    patterns = {i: patterns[i] for i in include}
    metadatas = {i: metadatas[i] for i in include}

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
    if args.bar_patterns:
        bar_patterns(metadatas, result_dfs, args.top_N)
    if args.by_sample_num:
        by_sample_num(metadatas, result_dfs)
    if args.by_strategy:
        by_strategy(metadatas, result_dfs)
    if args.by_fit_strategy:
        by_fit_strategy(metadatas, result_dfs)
    if args.by_augmentation:
        by_augmentation(metadatas, result_dfs)
