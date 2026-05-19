#!/usr/bin/env python3

import argparse
import csv
import re
import shlex
import statistics
import sys
from collections import defaultdict
from pathlib import Path


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
PERF_RE = re.compile(
    r"DTD_GEMM\s+PxQxg:\s+"
    r"(?P<P>\d+)\s+(?P<Q>\d+)\s+(?P<gpus>\d+)\s+"
    r"M:\s+(?P<M>\d+)\s+N:\s+(?P<N>\d+)\s+K:\s+(?P<K>\d+)\s+"
    r"mb:\s+(?P<mb>\d+)\s+nb:\s+(?P<nb>\d+)\s+kb:\s+(?P<kb>\d+)"
    r"(?:\s+batch_mode:\s+(?P<mode>\S+))?"
    r".*?\btime:\s+(?P<time>{float_re})\s+"
    r"gflops:\s+(?P<gflops>{float_re})".format(float_re=FLOAT_RE)
)

BEGIN_PREFIX = "PARSEC_SIMPLE_GEMM_RUN_BEGIN "
END_PREFIX = "PARSEC_SIMPLE_GEMM_RUN_END "


def parse_kv_payload(payload):
    values = {}
    for token in shlex.split(payload):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        values[key] = value
    return values


def read_records(log_path):
    records = []
    current = {}

    with open(log_path, "r", encoding="utf-8", errors="replace") as log:
        for line_no, line in enumerate(log, 1):
            line = line.rstrip("\n")
            if line.startswith(BEGIN_PREFIX):
                current = parse_kv_payload(line[len(BEGIN_PREFIX):])
                continue
            if line.startswith(END_PREFIX):
                current = {}
                continue

            match = PERF_RE.search(line)
            if not match:
                continue

            item = dict(current)
            item["line_no"] = line_no
            for key, value in match.groupdict().items():
                if value is not None:
                    item[key] = value
            item["mode"] = item.get("mode", "unknown")
            records.append(normalize_record(item))

    return records


def normalize_record(item):
    int_fields = ["P", "Q", "gpus", "M", "N", "K", "mb", "nb", "kb"]
    for field in int_fields:
        item[field] = int(item[field])
    item["time"] = float(item["time"])
    item["gflops"] = float(item["gflops"])
    item["tile"] = item["mb"] if item["mb"] == item["nb"] == item["kb"] else item["mb"]
    item["tile_label"] = (
        str(item["mb"])
        if item["mb"] == item["nb"] == item["kb"]
        else f"{item['mb']}x{item['nb']}x{item['kb']}"
    )
    item["matrix_key"] = (item["M"], item["N"], item["K"])
    item["matrix_label"] = (
        f"M=N=K={item['M']}"
        if item["M"] == item["N"] == item["K"]
        else f"M={item['M']} N={item['N']} K={item['K']}"
    )
    return item


def summarize(records, metric):
    grouped = defaultdict(list)
    examples = {}
    for record in records:
        key = (
            record["matrix_key"],
            record["gpus"],
            record["mode"],
            record["tile"],
            record["tile_label"],
        )
        grouped[key].append(record["gflops"])
        examples[key] = record

    rows = []
    for key, values in grouped.items():
        matrix_key, gpus, mode, tile, tile_label = key
        example = examples[key]
        stats = {
            "avg": statistics.mean(values),
            "best": max(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
        rows.append({
            "matrix_key": matrix_key,
            "matrix_label": example["matrix_label"],
            "M": matrix_key[0],
            "N": matrix_key[1],
            "K": matrix_key[2],
            "gpus": gpus,
            "mode": mode,
            "tile": tile,
            "tile_label": tile_label,
            "count": len(values),
            "avg": stats["avg"],
            "best": stats["best"],
            "median": stats["median"],
            "min": stats["min"],
            "max": stats["max"],
            "selected": stats[metric],
        })

    return sorted(rows, key=lambda r: (r["matrix_key"], r["gpus"], r["mode"], r["tile"]))


def write_summary(rows, summary_path):
    fields = [
        "M", "N", "K", "gpus", "mode", "tile", "tile_label", "count",
        "avg", "best", "median", "min", "max", "selected",
    ]
    with open(summary_path, "w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def safe_filename(label):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")


def plot_rows(rows, output_dir, image_format, metric, dpi, show):
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required to plot the sweep results. "
            "Install it, or use the generated CSV summary with another plotting tool."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    by_matrix = defaultdict(list)
    for row in rows:
        by_matrix[row["matrix_key"]].append(row)

    written = []
    for matrix_key, matrix_rows in sorted(by_matrix.items()):
        matrix_label = matrix_rows[0]["matrix_label"]
        series = defaultdict(list)
        for row in matrix_rows:
            label = f"{row['mode']}, {row['gpus']} GPU"
            if row["gpus"] != 1:
                label += "s"
            series[label].append(row)

        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        for label, points in sorted(series.items()):
            points = sorted(points, key=lambda r: r["tile"])
            ax.plot(
                [p["tile"] for p in points],
                [p["selected"] for p in points],
                marker="o",
                linewidth=1.8,
                label=label,
            )

        ax.set_title(f"DTD simple_gemm performance, {matrix_label}")
        ax.set_xlabel("Tile size (mb=nb=kb)")
        ax.set_ylabel(f"Performance ({metric} GF/s)")
        ax.grid(True, linestyle=":", linewidth=0.8)
        ax.legend()
        fig.tight_layout()

        filename = output_dir / f"simple_gemm_{safe_filename(matrix_label)}.{image_format}"
        fig.savefig(filename, dpi=dpi)
        written.append(filename)
        if show:
            plt.show()
        plt.close(fig)

    return written


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot DTD simple_gemm batch sweep results."
    )
    parser.add_argument("log", type=Path, help="Log produced by run_simple_gemm_batch_sweep.sh")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path("simple_gemm_plots"),
        help="Directory for generated figures and summary CSV."
    )
    parser.add_argument(
        "--summary", type=Path,
        help="CSV summary path (default: OUTPUT_DIR/simple_gemm_summary.csv)."
    )
    parser.add_argument(
        "--metric", choices=("avg", "best", "median", "min", "max"), default="avg",
        help="Metric to plot when several measured runs exist for one configuration."
    )
    parser.add_argument("--format", default="png", help="Image format passed to matplotlib.")
    parser.add_argument("--dpi", type=int, default=150, help="Figure resolution.")
    parser.add_argument("--show", action="store_true", help="Display figures interactively.")
    args = parser.parse_args(argv)

    records = read_records(args.log)
    if not records:
        raise SystemExit(
            f"No simple_gemm performance records found in {args.log}. "
            "The log must contain DTD_GEMM lines with 'time:' and 'gflops:'."
        )

    rows = summarize(records, args.metric)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary or (args.output_dir / "simple_gemm_summary.csv")
    write_summary(rows, summary_path)
    figures = plot_rows(rows, args.output_dir, args.format, args.metric, args.dpi, args.show)

    print(f"Read {len(records)} performance samples")
    print(f"Wrote summary: {summary_path}")
    for figure in figures:
        print(f"Wrote figure: {figure}")


if __name__ == "__main__":
    main()
