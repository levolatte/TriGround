from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


PATH_FIELDS = ("rgb", "visible", "ir", "infrared", "thermal", "depth", "aux")


def load_json(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object keyed by sample id")
    return payload


def sequence_key(sample_id: str) -> str:
    parts = sample_id.split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and len(parts[-1]) <= 3:
        parts = parts[:-1]
    if len(parts) >= 3 and parts[-1].isdigit() and len(parts[-1]) >= 6:
        parts = parts[:-1]
    return "_".join(parts)


def rebase_record(record: dict, source: Path, output: Path) -> dict:
    revised = dict(record)
    for field in PATH_FIELDS:
        value = revised.get(field)
        if not value:
            continue
        path = Path(value)
        resolved = path if path.is_absolute() else (source.parent / path).resolve()
        revised[field] = Path(os.path.relpath(resolved, output.parent)).as_posix()
    return revised


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove source rows sharing a sequence with an exclusion manifest"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    source_path = args.source.resolve()
    exclude_path = args.exclude.resolve()
    output_path = args.output.resolve()
    source = load_json(source_path)
    excluded = load_json(exclude_path)
    excluded_sequences = {sequence_key(sample_id) for sample_id in excluded}
    kept = {
        sample_id: rebase_record(record, source_path, output_path)
        for sample_id, record in source.items()
        if sequence_key(sample_id) not in excluded_sequences
    }
    remaining_overlap = sorted(
        {sequence_key(sample_id) for sample_id in kept} & excluded_sequences
    )
    if remaining_overlap:
        raise RuntimeError(f"sequence filtering failed: {remaining_overlap[:5]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(kept, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "exclude": str(exclude_path),
        "exclude_sha256": sha256(exclude_path),
        "source_samples": len(source),
        "kept_samples": len(kept),
        "removed_samples": len(source) - len(kept),
        "excluded_sequences": len(excluded_sequences),
        "remaining_sequence_overlap": 0,
        "output": str(output_path),
        "output_sha256": sha256(output_path),
    }
    args.report.resolve().write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
