"""Small, fail-fast checks for the bounded G/U diagnostic queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line]


def manifest_rows(path: Path, *, allow_repeated_ids: bool = False) -> list[dict]:
    rows = read_json(path)
    assert isinstance(rows, list), path
    ids = [str(row["id"]) for row in rows]
    if not allow_repeated_ids:
        assert len(ids) == len(set(ids)), f"duplicate IDs in {path}"
    return rows


def check_manifests(root: Path) -> None:
    counts = {
        "train_b": 1600,
        "train_gstar": 1600,
        "train_ustar": 1600,
        "fit_legacy181": 181,
        "fit_canonical_aux83": 83,
        "pressure16": 16,
        "pressure2city": 2,
        "env8": 8,
        **{f"city96_{mode}": 96 for mode in ("rgb", "rgb_ir", "rgb_depth", "trimodal")},
    }
    rows = {}
    for name, expected in counts.items():
        rows[name] = manifest_rows(root / f"{name}.json", allow_repeated_ids=name == "pressure16")
        assert len(rows[name]) == expected, (name, len(rows[name]), expected)
    for name in ("fit_legacy181_gt", "fit_canonical_aux83_gt", "city96_gt"):
        gt = read_json(root / f"{name}.json")
        assert isinstance(gt, dict), name
        assert len(gt) == {"fit_legacy181_gt": 181, "fit_canonical_aux83_gt": 83, "city96_gt": 96}[name]
    legacy_ids = {str(row["id"]) for row in rows["fit_legacy181"]}
    canonical_ids = {str(row["id"]) for row in rows["fit_canonical_aux83"]}
    assert canonical_ids <= legacy_ids
    city_ids = [str(row["id"]) for row in rows["city96_rgb"]]
    for mode in ("rgb_ir", "rgb_depth", "trimodal"):
        assert [str(row["id"]) for row in rows[f"city96_{mode}"]] == city_ids
    for name in ("train_gstar", "train_ustar"):
        assert [str(row["id"]) for row in rows[name]] == [str(row["id"]) for row in rows["train_b"]]
    assert read_json(root / "summary.json")


def check_predictions(manifest: Path, path: Path, limit: int) -> None:
    expected = [str(row["id"]) for row in manifest_rows(manifest)]
    if limit:
        expected = expected[:limit]
    actual = [str(row["id"]) for row in read_jsonl(path)]
    assert len(actual) == len(expected), (path, len(actual), len(expected))
    assert actual == expected, f"prediction IDs/order mismatch: {path}"


def check_reproduce8(reference: Path, actual: Path) -> None:
    baseline = read_jsonl(reference)[:8]
    repeated = read_jsonl(actual)
    assert len(baseline) == len(repeated) == 8
    for old, new in zip(baseline, repeated, strict=True):
        for field in ("id", "prompt", "raw_text", "prediction", "image_grid_thw"):
            assert old[field] == new[field], f"M2 old-eight mismatch: {old['id']} {field}"


def check_checkpoint(path: Path, step: int) -> None:
    assert path.is_dir(), path
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_config.json"):
        assert (path / name).is_file(), path / name
    assert list(path.glob("rng_state*.pth")), path
    assert int(read_json(path / "trainer_state.json")["global_step"]) == step


def check_trace(trace_path: Path, manifest: Path, steps: int) -> None:
    ids = [str(row["id"]) for row in manifest_rows(manifest, allow_repeated_ids=True)]
    trace = [str(row["id"]) for row in read_jsonl(trace_path)]
    expected = [ids[index % len(ids)] for index in range(steps * 8)]
    assert trace == expected, f"committed sample order differs: {trace_path}"


def check_optimizer_progress(path: Path, step: int) -> None:
    import torch

    optimizer = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False)
    values = set()
    for state in optimizer["state"].values():
        value = state.get("step")
        if value is not None:
            values.add(int(value.item() if hasattr(value, "item") else value))
    assert values == {step}, f"optimizer step mismatch: {values} != {step}"
    scheduler = torch.load(path / "scheduler.pt", map_location="cpu", weights_only=False)
    assert int(scheduler["last_epoch"]) == step, (path, scheduler["last_epoch"])


def check_preflight(train: Path, manifest: Path, first: Path, second: Path) -> None:
    for step in (1, 2):
        checkpoint = train / f"checkpoint-{step}"
        check_checkpoint(checkpoint, step)
        check_optimizer_progress(checkpoint, step)
    check_trace(train / "consumed_samples.jsonl", manifest, 2)
    a, b = read_jsonl(first), read_jsonl(second)
    assert len(a) == len(b) == 2
    for left, right in zip(a, b, strict=True):
        for field in ("id", "prompt", "raw_text", "prediction", "image_grid_thw"):
            assert left[field] == right[field], f"preflight reload mismatch: {left['id']} {field}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("manifests", "predictions", "reproduce8", "checkpoint", "trace", "preflight"))
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    p = [Path(value) for value in args.paths]
    if args.action == "manifests":
        check_manifests(p[0])
    elif args.action == "predictions":
        check_predictions(p[0], p[1], int(args.paths[2]))
    elif args.action == "reproduce8":
        check_reproduce8(p[0], p[1])
    elif args.action == "checkpoint":
        check_checkpoint(p[0], int(args.paths[1]))
    elif args.action == "trace":
        check_trace(p[0], p[1], int(args.paths[2]))
    elif args.action == "preflight":
        check_preflight(p[0], p[1], p[2], p[3])
    print(f"PASS {args.action}")


if __name__ == "__main__":
    main()
