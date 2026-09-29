"""Package a human-approved A/B/V release for the cloud without recoding images.

Only image files named by the native manifests are copied. Existing City images
stay relative to the cloud City data root (or at its known cloud path).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil


DEFAULT_CITY_ROOT = "/root/autodl-tmp/rematch_20260922/data/city/train"
ARMS = ("A", "B", "V")
METADATA = ("gt_manifest", "scene_map", "class_map")
WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _source_file(value: str, release_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute() and not PureWindowsPath(value).is_absolute():
        path = release_dir / path
    return path.resolve(strict=True)


def _city_relative(value: str, *, old_root: PurePosixPath,
                   cloud_root: PurePosixPath) -> str | None:
    path = PurePosixPath(value.replace("\\", "/"))
    if not path.is_absolute():
        if ".." in path.parts or not str(path):
            raise ValueError(f"invalid relative City image path: {value}")
        return str(path)
    try:
        relative = path.relative_to(old_root)
    except ValueError:
        return None
    return str(cloud_root / relative)


def _no_windows_paths(value) -> bool:
    if isinstance(value, str):
        return not WINDOWS_PATH.match(value)
    if isinstance(value, list):
        return all(_no_windows_paths(item) for item in value)
    if isinstance(value, dict):
        return all(_no_windows_paths(item) for item in value.values())
    return True


def export_release(release_path: Path, cloud_root: str, output_dir: Path,
                   cloud_city_data_root: str = DEFAULT_CITY_ROOT) -> dict:
    release_path = release_path.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"deployment directory already exists: {output_dir}")
    release = read_json(release_path)
    if release.get("status") != "ready":
        raise ValueError("only a human-approved ready release may be exported")
    cloud = PurePosixPath(cloud_root)
    city_cloud = PurePosixPath(cloud_city_data_root)
    if not cloud.is_absolute() or not city_cloud.is_absolute():
        raise ValueError("cloud root and City data root must be absolute POSIX paths")
    old_city = PurePosixPath(release.get("path_roots", {}).get("city_old_cloud", DEFAULT_CITY_ROOT))

    # Resolve every referenced file before creating the deployment. The source
    # path is the deduplication key; no image is decoded, resized, or re-encoded.
    assets: dict[Path, str] = {}
    rewritten: dict[str, list[dict]] = {}
    counts = {"native_rows": {}, "city_image_references": 0,
              "copied_image_references": 0, "unique_copied_images": 0}

    def cloud_image(value: str, *, native_reference: bool) -> str:
        city = _city_relative(value, old_root=old_city, cloud_root=city_cloud)
        if city is not None and not PureWindowsPath(value).is_absolute():
            if native_reference:
                counts["city_image_references"] += 1
            return city
        local = _source_file(value, release_path.parent)
        if not local.is_file():
            raise FileNotFoundError(f"release image is not a file: {local}")
        if local not in assets:
            assets[local] = f"asset_{len(assets) + 1:06d}{local.suffix}"
        if native_reference:
            counts["copied_image_references"] += 1
        return str(cloud / "assets" / assets[local])

    manifest_sources = {arm: release["manifests"][arm] for arm in ARMS}
    manifest_sources.update({f"diagnostics/{condition}": value
                             for condition, value in release["diagnostics"].items()})
    for name, location in manifest_sources.items():
        source = _source_file(location, release_path.parent)
        rows = read_json(source)
        if not isinstance(rows, list):
            raise ValueError(f"native manifest must be a JSON list: {source}")
        converted = []
        for row in rows:
            images = row["image"]
            if not isinstance(images, list) or not images or not all(isinstance(p, str) for p in images):
                raise ValueError(f"{source}: {row.get('id')} has invalid image list")
            new_images = [cloud_image(value, native_reference=True) for value in images]
            converted.append({**row, "image": new_images})
        rewritten[name] = converted
        counts["native_rows"][name] = len(converted)

    metadata_sources = {key: _source_file(release[key], release_path.parent) for key in METADATA}
    if not all(path.is_file() for path in metadata_sources.values()):
        raise FileNotFoundError("a release metadata path is not a file")
    sidecars = {
        path.name: path for path in release_path.parent.iterdir()
        if path.is_file() and path.suffix in {".json", ".jsonl"}
        and path.name not in {release_path.name, "A.json", "B.json", "V.json"}
    }
    audit_sources = {
        name: _source_file(value, release_path.parent)
        for name, value in release.get("audit_files", {}).items()
    }
    for path in audit_sources.values():
        if not path.is_file():
            raise FileNotFoundError(f"release audit file is not a file: {path}")
        sidecars[path.name] = path
    gt = read_json(metadata_sources["gt_manifest"])
    gt_rows = gt.values() if isinstance(gt, dict) else gt
    for row in gt_rows:
        for modality in ("visible", "infrared", "depth"):
            if row.get(modality):
                row[modality] = cloud_image(row[modality], native_reference=False)
    counts["unique_copied_images"] = len(assets)
    exported = dict(release)
    exported["manifests"] = {arm: str(cloud / "manifests" / f"{arm}.json") for arm in ARMS}
    exported["diagnostics"] = {
        condition: str(cloud / "manifests" / "diagnostics" / f"{condition}.json")
        for condition in release["diagnostics"]
    }
    for key in METADATA:
        exported[key] = str(cloud / "manifests" / "diagnostics" / f"{key}.json")
    exported["path_roots"] = {"city_old_cloud": str(city_cloud),
                              "exported_assets_cloud": str(cloud / "assets")}
    exported["evidence"] = {path.stem: str(cloud / "evidence" / path.name)
                            for path in sidecars.values()}
    if audit_sources:
        exported["audit_files"] = {name: str(cloud / "evidence" / path.name)
                                   for name, path in audit_sources.items()}
    for field in ("accepted_reviews", "scene_exposure", "overlap_audit"):
        if field in release and field not in exported["evidence"]:
            raise FileNotFoundError(f"release sidecar is missing: {field}")
        if field in exported["evidence"]:
            exported[field] = exported["evidence"][field]
    if (not _no_windows_paths(exported) or not _no_windows_paths(gt)
            or any(not _no_windows_paths(rows) for rows in rewritten.values())):
        raise ValueError("local Windows path remained in the exported release")

    output_dir.mkdir(parents=True)
    for source, filename in assets.items():
        destination = output_dir / "assets" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for name, rows in rewritten.items():
        destination = output_dir / "manifests" / f"{name}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    for key, source in metadata_sources.items():
        destination = output_dir / "manifests" / "diagnostics" / f"{key}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if key == "gt_manifest":
            destination.write_text(json.dumps(gt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            shutil.copy2(source, destination)
    for source in sidecars.values():
        destination = output_dir / "evidence" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (output_dir / "release.json").write_text(
        json.dumps(exported, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {"cloud_root": str(cloud), "cloud_city_data_root": str(city_cloud),
               "evidence_files": sorted(sidecars), **counts}
    (output_dir / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--cloud-root", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cloud-city-data-root", default=DEFAULT_CITY_ROOT)
    args = parser.parse_args()
    summary = export_release(args.release, args.cloud_root, args.output_dir,
                             args.cloud_city_data_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
