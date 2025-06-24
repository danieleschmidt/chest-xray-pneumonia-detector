"""Utilities to generate a basic architecture overview."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ModuleInfo:
    """Information extracted from a Python module."""

    name: str
    path: Path
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    docstring: Optional[str] = None


def parse_module(path: Path) -> ModuleInfo:
    """Parse a module file and return its high-level structure."""

    tree = ast.parse(path.read_text())
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    docstring = ast.get_docstring(tree)
    return ModuleInfo(
        name=path.stem,
        path=path,
        functions=functions,
        classes=classes,
        docstring=docstring,
    )


def scan_project(src_dir: Path) -> List[ModuleInfo]:
    """Scan all Python modules in the given directory."""

    modules = []
    for file_path in src_dir.glob("*.py"):
        if file_path.name == "__init__.py":
            continue
        modules.append(parse_module(file_path))
    return modules


def generate_markdown(modules: List[ModuleInfo]) -> str:
    """Create a Markdown summary of the given modules."""

    lines = ["# Architecture Overview", ""]
    for module in sorted(modules, key=lambda m: m.name):
        lines.append(f"## {module.name}")
        if module.docstring:
            lines.append(module.docstring.splitlines()[0])
            lines.append("")
        if module.classes:
            lines.append("### Classes")
            for cls in module.classes:
                lines.append(f"- {cls}")
        if module.functions:
            lines.append("### Functions")
            for func in module.functions:
                lines.append(f"- {func}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a simple architecture overview."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Source directory to scan",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ARCHITECTURE.md"),
        help="Output markdown file",
    )
    args = parser.parse_args()

    modules = scan_project(args.src)
    markdown = generate_markdown(modules)
    args.output.write_text(markdown)
    print(f"Architecture overview written to {args.output}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
