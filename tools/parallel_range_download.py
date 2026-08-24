from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import subprocess
from pathlib import Path


def _download_part(url: str, path: Path, start: int, end: int, retries: int) -> Path:
    expected = end - start + 1
    if path.is_file() and path.stat().st_size == expected:
        return path
    path.unlink(missing_ok=True)
    subprocess.run(
        [
            "curl", "--fail", "--location", "--retry", str(retries),
            "--retry-delay", "3", "--range", f"{start}-{end}",
            "--output", str(path), url,
        ],
        check=True,
    )
    if path.stat().st_size != expected:
        raise RuntimeError(
            f"range {start}-{end} size mismatch: {path.stat().st_size} != {expected}"
        )
    return path


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume one file using validated parallel ranges")
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--total-size", type=int, required=True)
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--md5", default=None)
    parser.add_argument("--retries", type=int, default=10)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    current = output.stat().st_size if output.exists() else 0
    if current > args.total_size:
        raise RuntimeError(f"existing output is too large: {current} > {args.total_size}")
    if current < args.total_size:
        remaining = args.total_size - current
        chunk = (remaining + args.connections - 1) // args.connections
        ranges = []
        for index in range(args.connections):
            start = current + index * chunk
            if start >= args.total_size:
                break
            end = min(start + chunk - 1, args.total_size - 1)
            ranges.append((index, start, end))
        part_dir = output.parent / f".{output.name}.parts"
        part_dir.mkdir(exist_ok=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(ranges)) as executor:
            futures = [
                executor.submit(
                    _download_part, args.url, part_dir / f"{index:03d}.part", start, end,
                    args.retries,
                )
                for index, start, end in ranges
            ]
            parts = [future.result() for future in futures]
        with output.open("ab") as destination:
            for part in parts:
                with part.open("rb") as source:
                    while block := source.read(8 * 1024 * 1024):
                        destination.write(block)
                part.unlink()
                destination.flush()
                os.fsync(destination.fileno())
        part_dir.rmdir()

    if output.stat().st_size != args.total_size:
        raise RuntimeError(f"final size mismatch: {output.stat().st_size} != {args.total_size}")
    if args.md5:
        actual = _md5(output)
        if actual.lower() != args.md5.lower():
            raise RuntimeError(f"MD5 mismatch: {actual} != {args.md5}")
    print({"output": str(output), "bytes": output.stat().st_size, "md5": args.md5})


if __name__ == "__main__":
    main()
