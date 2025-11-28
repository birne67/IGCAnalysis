#!/usr/bin/env python3
"""Set version in pyproject.toml and in package __init__.py

Usage:
    python tools/set_version.py 1.3.0

This script is intentionally small and uses stdlib tomllib (Python>=3.11).
It writes the given version into [project] version in pyproject.toml and
updates __version__ in IGCAnalysis/__init__.py.
"""
import sys
from pathlib import Path
import tomllib


def set_pyproject_version(pyproject_path: Path, version: str):
    text = pyproject_path.read_text(encoding="utf-8")
    # naive replace for simple pyproject with a single version line
    new_text = []
    for line in text.splitlines():
        if line.strip().startswith("version") and "=" in line:
            new_text.append(f'version = "{version}"')
        else:
            new_text.append(line)
    pyproject_path.write_text("\n".join(new_text), encoding="utf-8")


def set_pkg_init_version(init_path: Path, version: str):
    txt = init_path.read_text(encoding="utf-8")
    lines = txt.splitlines()
    found = False
    for i, l in enumerate(lines):
        if l.strip().startswith("__version__"):
            lines[i] = f'__version__ = "{version}"'
            found = True
            break
    if not found:
        # insert at top
        lines.insert(0, f'__version__ = "{version}"')
    init_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    if len(sys.argv) < 2:
        print("Usage: set_version.py X.Y.Z")
        sys.exit(1)
    version = sys.argv[1]
    root = Path(__file__).resolve().parents[1]
    pyproject = root / "pyproject.toml"
    pkg_init = root / "IGCAnalysis" / "__init__.py"
    if not pyproject.exists():
        print("pyproject.toml not found; aborting")
        sys.exit(2)
    if not pkg_init.exists():
        print("package __init__.py not found - creating new file")
        pkg_init.write_text("")
    set_pyproject_version(pyproject, version)
    set_pkg_init_version(pkg_init, version)
    print(f"Set version to {version} in {pyproject} and {pkg_init}")


if __name__ == "__main__":
    main()
