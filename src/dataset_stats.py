"""CLI to count images per class in a dataset directory.

The tool can optionally save counts as JSON, CSV or a PNG bar chart when
``matplotlib`` is installed. Results are alphabetically ordered by default and
can be sorted by count.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable


def count_images_per_class(
    input_dir: str, extensions: Iterable[str] | None = None
) -> Dict[str, int]:
    """Return a mapping of class name to image count.

    Parameters
    ----------
    input_dir:
        Directory containing one subfolder per class.
    extensions:
        Iterable of allowed file extensions. Defaults to ``{'.jpg', '.jpeg', '.png'}``.

    Raises
    ------
    FileNotFoundError
        If ``input_dir`` does not exist.
    NotADirectoryError
        If ``input_dir`` exists but is not a directory.
    ValueError
        If no class folders are found in ``input_dir``.
    """
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png"}
    else:
        extensions = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in extensions
        }
    path = Path(input_dir)
    if not path.exists():
        raise FileNotFoundError(f"Input directory '{input_dir}' not found")
    if not path.is_dir():
        raise NotADirectoryError(f"Input path '{input_dir}' is not a directory")
    counts: Dict[str, int] = {}
    for cls_dir in path.iterdir():
        if cls_dir.is_dir():
            num_files = sum(
                1
                for f in cls_dir.rglob("*")
                if f.is_file() and f.suffix.lower() in extensions
            )
            counts[cls_dir.name] = num_files
    if not counts:
        raise ValueError(f"No class folders found in '{input_dir}'")
    return counts


def _sort_items(counts: Dict[str, int], sort_by: str) -> list[tuple[str, int]]:
    """Return ``counts`` items sorted by ``sort_by``.

    Parameters
    ----------
    counts:
        Mapping of class name to count.
    sort_by:
        ``"name"`` for alphabetical order or ``"count"`` for descending count.
    """

    if sort_by == "count":
        return sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return sorted(counts.items())


def print_stats(counts: Dict[str, int], *, sort_by: str = "name") -> None:
    """Print formatted dataset statistics."""
    total = sum(counts.values())
    for cls, num in _sort_items(counts, sort_by):
        pct = (num / total) * 100 if total else 0
        print(f"{cls}: {num} images ({pct:.1f}% of total)")
    print(f"Total images: {total}")


def plot_bar(counts: Dict[str, int], output_path: str, *, sort_by: str = "name") -> None:
    """Save a horizontal bar chart of ``counts`` to ``output_path``.

    Parameters
    ----------
    counts:
        Mapping of class name to number of images.
    output_path:
        Destination PNG file. Parent directories must exist.
    sort_by:
        ``"name"`` for alphabetical order or ``"count"`` for descending count.

    Raises
    ------
    RuntimeError
        If ``matplotlib`` is not available.
    FileNotFoundError
        If ``output_path`` is in a non-existent directory.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError("matplotlib is required for plotting") from exc

    output_file = Path(output_path)
    if not output_file.parent.exists():
        raise FileNotFoundError(
            f"Output directory '{output_file.parent}' does not exist"
        )

    items = _sort_items(counts, sort_by)
    labels = [cls for cls, _ in items]
    values = [num for _, num in items]
    plt.figure(figsize=(max(6, len(labels) * 1.2), 4))
    plt.barh(labels, values)
    plt.xlabel("Number of images")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def main(argv: Iterable[str] | None = None) -> None:
    """Run the dataset statistics CLI.

    Parameters
    ----------
    argv:
        Optional list of command-line arguments. Defaults to ``sys.argv[1:]``.

    The command prints class counts and can optionally save them as JSON, CSV
    or a PNG bar chart when ``matplotlib`` is installed. Results are sorted
    alphabetically by default; pass ``--sort_by count`` to order by descending
    count instead. When using ``--plot_png`` the destination directory must
    already exist.
    """
    parser = argparse.ArgumentParser(
        description="Print number of images per class in a dataset"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Dataset root directory with one subfolder per class",
    )
    parser.add_argument(
        "--json_output",
        help="Optional path to save counts as JSON",
    )
    parser.add_argument(
        "--csv_output",
        help="Optional path to save counts as CSV",
    )
    parser.add_argument(
        "--plot_png",
        help="Optional path to save a bar chart of the counts",
    )
    parser.add_argument(
        "--sort_by",
        choices=["name", "count"],
        default="name",
        help="Sort output by class name or count",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        help="File extensions to count (default: .jpg .jpeg .png)",
    )
    args = parser.parse_args(argv)

    try:
        counts = count_images_per_class(args.input_dir, args.extensions)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:  # pragma: no cover - CLI exit
        parser.error(str(exc))

    print_stats(counts, sort_by=args.sort_by)
    if args.json_output:
        ordered = {cls: num for cls, num in _sort_items(counts, args.sort_by)}
        Path(args.json_output).write_text(
            json.dumps(ordered, indent=2)
        )
    if args.csv_output:
        csv_lines = ["class,count"]
        for cls, count in _sort_items(counts, args.sort_by):
            csv_lines.append(f"{cls},{count}")
        Path(args.csv_output).write_text("\n".join(csv_lines))
    if args.plot_png:
        try:
            plot_bar(counts, args.plot_png, sort_by=args.sort_by)
        except RuntimeError as exc:  # pragma: no cover - optional dep
            parser.error(str(exc))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
