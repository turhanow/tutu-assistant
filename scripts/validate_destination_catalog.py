"""Validate the immutable destination catalog used by discovery flow."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.adapters.file_catalog import FileCatalogRepository
from app.domain.errors import CatalogContractError

DEFAULT_CATALOG = Path("data/destinations/v1/catalog.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_CATALOG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        repository = FileCatalogRepository.from_path(args.path)
    except CatalogContractError as error:
        print(f"Catalog validation failed: {error}")
        return 1
    print(
        f"Catalog {repository.version} is valid: "
        f"{repository.destination_count} destinations, "
        f"{repository.activity_count} activities"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
