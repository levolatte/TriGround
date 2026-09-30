"""Run a GT-free teacher episode on the dedicated host and fetch its visible inputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex

from .model import prepare_image_messages


ROOT = "/root/autodl-tmp/rematch_20260922"
B = ROOT + "/results/visual_agent/capability_rebuild_20260929"
PYTHON = ROOT + "/.venvs/aux_selection/bin/python"
DEFAULT_CODE_DIR = B + "/training_code"


def _prepare_visible_images(messages: list[dict], local_dir: Path) -> tuple[list[dict], list[dict]]:
    """Save the exact resized views that the student processor would pass to Qwen."""
    prepared, metadata = prepare_image_messages(messages)
    image_dir = local_dir / "prepared_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_index = 0
    metadata_index = 0
    visible_metadata = []
    for message_index, message in enumerate(prepared):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part_index, part in enumerate(content):
            if part.get("type") != "image":
                continue
            item = metadata[metadata_index]
            metadata_index += 1
            remote_path = part.pop("remote_image", item.get("path"))
            image_path = image_dir / f"image_{image_index:04d}.png"
            image_index += 1
            part["image"].save(image_path, format="PNG")
            part["remote_image"] = remote_path
            part["source_size"] = item["source_size"]
            part["prepared_size"] = item["prepared_size"]
            part["max_pixels"] = item["max_pixels"]
            visible_metadata.append({
                "message_index": message_index,
                "part_index": part_index,
                "remote_image": remote_path,
                "prepared_image": str(image_path.resolve()),
                "modality": item.get("modality"),
                "view": item.get("view"),
                "source_size": item["source_size"],
                "prepared_size": item["prepared_size"],
                "max_pixels": item["max_pixels"],
            })
            part["image"] = str(image_path.resolve())
    return prepared, visible_metadata


def main() -> None:
    import paramiko

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["init", "step", "export", "view"])
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--code-dir", default=DEFAULT_CODE_DIR,
                        help="remote checkout containing the synchronized teacher code")
    parser.add_argument("--manifest")
    parser.add_argument("--candidate-cache")
    parser.add_argument("--sample-id")
    parser.add_argument("--action-file", type=Path)
    parser.add_argument("--resume-prefix")
    parser.add_argument("--prefix-events", type=int)
    args = parser.parse_args()

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("connect.westc.seetacloud.com", port=27135, username="root",
                   password=os.environ["TRIGROUND_SSH_PASSWORD"], timeout=20)
    episode_dir = args.episode_dir.rstrip("/")
    remote_episode = episode_dir + "/episode.json"
    command = [PYTHON, "-m", "experiments.visual_agent.object_teacher", args.operation]
    if args.operation == "init":
        command += ["--manifest", args.manifest, "--candidate-cache", args.candidate_cache,
                    "--sample-id", args.sample_id, "--output-dir", args.episode_dir,
                    "--dino-model", ROOT + "/models/grounding-dino-tiny",
                    "--sam-model", ROOT + "/models/sam2.1-hiera-tiny"]
        if args.resume_prefix:
            command += ["--resume-prefix", args.resume_prefix]
        if args.prefix_events is not None:
            command += ["--prefix-events", str(args.prefix_events)]
    elif args.operation == "step":
        action = json.loads(args.action_file.read_text(encoding="utf-8-sig"))
        command += ["--episode", remote_episode]
        if "raw_output" in action:
            command += ["--raw-output", action["raw_output"]]
        else:
            command += ["--note", action.get("note", ""), "--name", action["name"],
                        "--arguments", json.dumps(action["arguments"], ensure_ascii=False)]
    elif args.operation == "export":
        command += ["--episode", remote_episode]

    if args.operation != "view":
        shell = ("cd " + shlex.quote(args.code_dir.rstrip("/")) +
                 " && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 " + shlex.join(command))
        _, stdout, stderr = client.exec_command(shell, timeout=180)
        output, error = stdout.read().decode(), stderr.read().decode()
        code = stdout.channel.recv_exit_status()
        print(output)
        if code:
            raise RuntimeError(error)

    sftp = client.open_sftp()
    sftp.get_channel().settimeout(120)
    local_dir = args.local_dir.resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    cache_path = local_dir / "image_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    messages = json.loads(sftp.open(episode_dir + "/current_messages.json").read().decode())
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "image":
                continue
            remote_path = part["image"]
            if remote_path in cache and Path(cache[remote_path]).exists():
                local_path = Path(cache[remote_path])
            else:
                local_path = local_dir / f"source_{len(cache):04d}{Path(remote_path).suffix}"
                sftp.get(remote_path, str(local_path), max_concurrent_prefetch_requests=16)
                cache[remote_path] = str(local_path.resolve())
                cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            part["remote_image"] = remote_path
            part["image"] = str(local_path.resolve())

    visible_messages, image_metadata = _prepare_visible_images(messages, local_dir)
    messages_path = local_dir / "current_messages.json"
    messages_path.write_text(json.dumps(visible_messages, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    metadata_path = local_dir / "image_preparation.json"
    metadata_path.write_text(json.dumps(image_metadata, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    tools_path = local_dir / "current_tools.json"
    with sftp.open(episode_dir + "/current_tools.json") as source:
        tools_path.write_bytes(source.read())
    sftp.get(remote_episode, str(local_dir / "episode.json"))
    print(json.dumps({
        "messages": str(messages_path),
        "tools": str(tools_path),
        "episode": str(local_dir / "episode.json"),
        "prepared_images": len(image_metadata),
        "image_preparation": str(metadata_path),
    }, ensure_ascii=False))
    client.close()


if __name__ == "__main__":
    main()
