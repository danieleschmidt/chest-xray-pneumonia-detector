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


def print_stats(counts: Dict[str, int]) -> None:
    """Print formatted dataset statistics."""
    total = sum(counts.values())
    for cls, num in sorted(counts.items()):
        pct = (num / total) * 100 if total else 0
        print(f"{cls}: {num} images ({pct:.1f}% of total)")
    print(f"Total images: {total}")


def main() -> None:
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
        "--extensions",
        nargs="*",
        help="File extensions to count (default: .jpg .jpeg .png)",
    )
    args = parser.parse_args()

    try:
        counts = count_images_per_class(args.input_dir, args.extensions)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:  # pragma: no cover - CLI exit
        parser.error(str(exc))

    print_stats(counts)
    if args.json_output:
        Path(args.json_output).write_text(
            json.dumps(counts, indent=2, sort_keys=True)
        )
    if args.csv_output:
        csv_lines = ["class,count"]
        for cls, count in sorted(counts.items()):
            csv_lines.append(f"{cls},{count}")
        Path(args.csv_output).write_text("\n".join(csv_lines))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
