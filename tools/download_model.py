from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a complete Hugging Face model snapshot")
    parser.add_argument("--repo-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--token", default=None, help="HF token; normally use the HF_TOKEN env variable")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        cache_dir=cache_dir,
        token=args.token,
    )
    files = [path for path in Path(snapshot).rglob("*") if path.is_file()]
    size = sum(path.stat().st_size for path in files)
    print(json.dumps({
        "repo_id": args.repo_id,
        "revision": args.revision,
        "snapshot": str(Path(snapshot).resolve()),
        "files": len(files),
        "size_gib": round(size / 2**30, 3),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
