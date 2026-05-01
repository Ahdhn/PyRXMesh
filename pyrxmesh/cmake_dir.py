"""Print the installed PyRXMesh CMake package directory."""

from __future__ import annotations

from pathlib import Path


def get_cmake_dir() -> Path:
    return Path(__file__).resolve().parent / "cmake"


def main() -> None:
    print(get_cmake_dir())


if __name__ == "__main__":
    main()
