from __future__ import annotations

"""Record Figure 6(a) reference availability without inventing coordinates."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def build_reference(asset_root: Path, coordinates: list[dict[str, float]] | None = None) -> dict[str, Any]:
    candidates = sorted(
        path
        for path in Path(asset_root).rglob("*")
        if path.is_file() and any(token in path.name.lower() for token in ("fig6", "figure6"))
    )
    result: dict[str, Any] = {
        "asset_root": str(Path(asset_root)),
        "candidate_assets": [],
        "status": "unavailable",
        "coordinates": [],
        "uncertainty": None,
        "method": "no automatic digitization performed",
    }
    for path in candidates:
        item: dict[str, Any] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        try:
            from PIL import Image

            with Image.open(path) as image:
                item["width"] = image.width
                item["height"] = image.height
        except Exception:
            item["dimensions"] = "unreadable-by-Pillow"
        result["candidate_assets"].append(item)
    if coordinates is not None:
        if len(coordinates) != 10:
            raise ValueError("Figure 6(a) requires exactly ten marker coordinates")
        result.update(
            {
                "status": "provided_coordinates",
                "coordinates": coordinates,
                "uncertainty": "user-provided; no automatic uncertainty estimate",
                "method": "explicit coordinate input",
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--coordinates-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    coordinates = None
    if args.coordinates_json is not None:
        coordinates = json.loads(args.coordinates_json.read_text(encoding="utf-8"))
    result = build_reference(args.asset_root, coordinates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
