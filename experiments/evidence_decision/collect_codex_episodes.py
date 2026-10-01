"""Run GT-blind Codex action and blind-review loops on real teacher episodes.

Example API use from an authenticated REPL:

    collect_jobs(jobs, Path("results/codex_collection"),
                 run_task_module, fetch_visible_state, fetch_episode, batch_size=8)

Each job supplies ``manifest``, ``candidate``, ``sample_id``, and remote
``episode_dir`` paths. The helper callables own remote execution and file
transfer; this module contains no credentials or scheduler.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from . import codex_teacher
from .object_teacher import STATE_SCHEMA


OBJECT_TEACHER_MODULE = "object_teacher"
MAX_NEW_TEACHER_STEPS = 8


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _read_episode(path: Path, sample_id: str) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != STATE_SCHEMA:
        raise ValueError(f"{path}: unsupported teacher episode state")
    if str(state.get("sample_id")) != sample_id:
        raise ValueError(f"{path}: episode sample ID differs from its job")
    return state


def _is_terminal(state: dict[str, Any]) -> bool:
    return bool(state.get("complete") or state.get("terminal_status"))


def _new_teacher_steps(state: dict[str, Any]) -> int:
    # object_teacher restores the student's events into turns/prefix_events;
    # steps contains only actions executed by this teacher, indexed from zero.
    return len(state.get("steps", []))


def _review_rows(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = {}
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["id"]), row["step"])
        if key in rows:
            raise ValueError(f"{path}:{line_number}: duplicate review for {key[0]} step {key[1]}")
        rows[key] = row
    return rows


def _append_review(path: Path, row: dict[str, Any], rows: dict[tuple[str, int], dict[str, Any]]) -> None:
    key = (str(row["id"]), row["step"])
    if key in rows:
        return
    if not isinstance(row.get("use_action"), bool) or not isinstance(row.get("note_supported"), bool):
        raise ValueError(f"{row['id']} step {row['step']}: reviewer must return boolean decisions")
    if not isinstance(row.get("reason"), str) or not row["reason"].strip():
        raise ValueError(f"{row['id']} step {row['step']}: reviewer reason must be nonempty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
    rows[key] = row


def _action_path(local_episode_dir: Path, step_index: int) -> Path:
    return local_episode_dir / "actions" / f"step_{step_index:04d}.action.json"


def _before_state_path(local_episode_dir: Path, step_index: int) -> Path:
    return local_episode_dir / "states" / f"step_{step_index:04d}.before.json"


def _save_before_state(local_episode_dir: Path, step_index: int,
                       state: dict[str, Any]) -> dict[str, Any]:
    path = _before_state_path(local_episode_dir, step_index)
    if not path.exists():
        _write_json(path, state)
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("id") != state.get("id") or not saved.get("messages") or not saved.get("tools"):
        raise ValueError(f"{path}: invalid cached before-state")
    return saved


def _load_action(path: Path, sample_id: str) -> dict[str, Any]:
    action = json.loads(path.read_text(encoding="utf-8"))
    if str(action.get("id")) != sample_id or action.get("name") is None:
        raise ValueError(f"{path}: saved action does not match {sample_id}")
    if not isinstance(action.get("arguments"), dict):
        raise ValueError(f"{path}: saved action arguments must be an object")
    return action


def _save_action(path: Path, action: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(action, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _next_log_path(root: Path, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    suffix = 1
    while path.exists():
        path = root / f"{Path(name).stem}_{suffix:03d}{Path(name).suffix}"
        suffix += 1
    return path


def _remote_episode_file(remote_episode_dir: str) -> str:
    return remote_episode_dir.rstrip("/") + "/episode.json"


def _fetch_remote_episode(fetch_episode: Callable, remote_episode_dir: str,
                          local_episode_dir: Path) -> bool:
    exists = fetch_episode(remote_episode_dir, local_episode_dir)
    if not isinstance(exists, bool):
        raise TypeError("fetch_episode must return True when fetched or False when the remote episode is absent")
    return exists


def _init_remote(job: dict[str, Any], run_task_module: Callable) -> None:
    args = [
        "init", "--manifest", str(job["manifest"]),
        "--candidate-cache", str(job["candidate"]),
        "--sample-id", str(job["sample_id"]),
        "--output-dir", str(job["episode_dir"]),
    ]
    if job.get("trace_path"):
        args.extend(["--resume-prefix", str(job["trace_path"])])
    if job.get("prefix_events") is not None:
        args.extend(["--prefix-events", str(job["prefix_events"])])
    if job.get("migrate_legacy_prefix"):
        args.append("--migrate-legacy-prefix")
    run_task_module(OBJECT_TEACHER_MODULE, *args)


def _initialise_job(job: dict[str, Any], local_episode_dir: Path,
                    run_task_module: Callable, fetch_visible_state: Callable,
                    fetch_episode: Callable) -> tuple[dict[str, Any], bool]:
    local_episode_path = local_episode_dir / "episode.json"
    if local_episode_path.exists():
        local_state = _read_episode(local_episode_path, str(job["sample_id"]))
        if _is_terminal(local_state):
            return local_state, True
        if not _fetch_remote_episode(fetch_episode, str(job["episode_dir"]), local_episode_dir):
            raise FileNotFoundError(
                f"remote episode missing for unfinished local episode {local_episode_path}; refusing to reinitialize"
            )
    elif _fetch_remote_episode(fetch_episode, str(job["episode_dir"]), local_episode_dir):
        pass
    else:
        if local_episode_dir.exists() and any(local_episode_dir.iterdir()):
            raise FileExistsError(
                f"{local_episode_dir} has collection artifacts but no remote episode; refusing to overwrite them"
            )
        _init_remote(job, run_task_module)

    episode_path = local_episode_dir / "episode.json"
    if not episode_path.exists():
        if not _fetch_remote_episode(fetch_episode, str(job["episode_dir"]), local_episode_dir):
            raise FileNotFoundError(f"ObjectTeacher init did not create {_remote_episode_file(str(job['episode_dir']))}")
    fetch_visible_state(str(job["episode_dir"]), local_episode_dir)
    state = _read_episode(episode_path, str(job["sample_id"]))
    return state, False


def _refresh_visible_state(item: dict[str, Any]) -> dict[str, Any]:
    state = codex_teacher.read_state(item["local_episode_dir"])
    if state["id"] != item["sample_id"]:
        raise ValueError(f"visible state ID differs for {item['sample_id']}")
    return state


def _review_pending_steps(active: list[dict[str, Any]], review_root: Path,
                          local_root: Path, log_root: Path, batch_index: int,
                          round_index: int) -> bool:
    """Review already-executed steps from their cached pre-action messages only."""
    pending = []
    for item in active:
        state = _read_episode(item["local_episode_dir"] / "episode.json", item["sample_id"])
        review_path = review_root / f"{item['bucket']}.jsonl"
        review_index = _review_rows(review_path)
        item["review_path"] = review_path
        item["review_index"] = review_index
        base_steps = 0
        for step_index in range(base_steps, len(state.get("steps", []))):
            key = (item["sample_id"], step_index)
            if key in review_index:
                continue
            action_path = _action_path(item["local_episode_dir"], step_index)
            if not action_path.exists():
                raise FileNotFoundError(
                    f"{item['sample_id']} step {step_index} exists without a saved teacher action"
                )
            before_path = _before_state_path(item["local_episode_dir"], step_index)
            if not before_path.exists():
                raise FileNotFoundError(
                    f"{item['sample_id']} step {step_index} exists without its cached local before-state"
                )
            before_state = json.loads(before_path.read_text(encoding="utf-8"))
            if before_state.get("id") != item["sample_id"]:
                raise ValueError(f"{before_path}: cached before-state ID differs")
            action = _load_action(action_path, item["sample_id"])
            step = state["steps"][step_index]
            actual_call = {"name": (step.get("call") or {}).get("name"),
                           "arguments": (step.get("call") or {}).get("arguments")}
            if actual_call != {"name": action["name"], "arguments": action["arguments"]}:
                raise ValueError(f"{item['sample_id']} step {step_index}: saved action differs from real step")
            pending.append({"item": item, "step_index": step_index, "action": action,
                            "state": before_state})
            # Keep IDs unique in this blind review batch. A repeated pass handles
            # any later unreviewed action from the same episode.
            break

    if not pending:
        return False
    with tempfile.TemporaryDirectory(prefix="codex_review_", dir=local_root) as staging_name:
        staging = Path(staging_name)
        for row in pending:
            _write_json(staging / f"{row['item']['sample_id']}.action.json", row["action"])
        review_log = _next_log_path(
            log_root, _batch_log_name("review", batch_index, round_index,
                                      [row["item"]["sample_id"] for row in pending]))
        reviews = codex_teacher.review_actions([row["state"] for row in pending], staging, review_log)
    review_by_id = {row["id"]: row for row in reviews}
    pending_ids = {row["item"]["sample_id"] for row in pending}
    if set(review_by_id) != pending_ids or len(reviews) != len(pending):
        raise ValueError("reviewer must return exactly one row for every before-state")
    for row in pending:
        item = row["item"]
        review = review_by_id[item["sample_id"]]
        review_row = {"id": item["sample_id"], "step": row["step_index"],
                      "use_action": review["use_action"],
                      "note_supported": review["note_supported"],
                      "reason": review["reason"]}
        _append_review(item["review_path"], review_row, item["review_index"])
    return True


def _batch_log_name(kind: str, batch_index: int, round_index: int, ids: list[str]) -> str:
    label = "_".join(ids[:2]).replace("/", "_").replace("\\", "_")
    if len(ids) > 2:
        label += f"_plus{len(ids) - 2}"
    return f"{kind}_batch_{batch_index:03d}_round_{round_index:03d}_{label}.log.json"


def collect_jobs(
    jobs: list[dict[str, Any]],
    local_root: Path,
    run_task_module: Callable,
    fetch_visible_state: Callable,
    fetch_episode: Callable,
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    """Run up to eight new teacher actions per job, in unique-ID batches.

    ``fetch_episode(remote_episode_dir, local_episode_dir)`` must return True
    after copying the remote ``episode.json`` or False when it is absent.
    Exceptions from the remote helpers or Codex CLI fail fast; completed
    local/remote artifacts are retained for the next invocation.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    local_root = Path(local_root).resolve()
    local_root.mkdir(parents=True, exist_ok=True)
    review_root = local_root / "reviews"
    episode_root = local_root / "episodes"
    log_root = local_root / "logs"
    prepared_jobs = []
    remote_dirs = set()
    local_dirs = set()
    for source in jobs:
        job = dict(source)
        sample_id = str(job["sample_id"])
        job["sample_id"] = sample_id
        job["bucket"] = str(job.get("bucket") or "recovery")
        for key in ("manifest", "candidate", "episode_dir"):
            if key not in job:
                raise KeyError(f"job {sample_id}: missing {key}")
        remote_dir = str(job["episode_dir"]).rstrip("/")
        if remote_dir in remote_dirs:
            raise ValueError(f"duplicate remote episode directory: {remote_dir}")
        remote_dirs.add(remote_dir)
        job["episode_dir"] = remote_dir
        local_dir = episode_root / job["bucket"] / sample_id
        local_key = str(local_dir.resolve())
        if local_key in local_dirs:
            raise ValueError(f"duplicate local episode directory: {local_dir}")
        local_dirs.add(local_key)
        prepared_jobs.append({"job": job, "sample_id": sample_id, "bucket": job["bucket"],
                              "remote_episode_dir": remote_dir, "local_episode_dir": local_dir})

    results = []
    for batch_index, start in enumerate(range(0, len(prepared_jobs), batch_size), 1):
        batch = prepared_jobs[start:start + batch_size]
        ids = [item["sample_id"] for item in batch]
        if len(set(ids)) != len(ids):
            raise ValueError(f"batch {batch_index} contains duplicate sample IDs")
        active = []
        for item in batch:
            state, skip_local_terminal = _initialise_job(
                item["job"], item["local_episode_dir"], run_task_module,
                fetch_visible_state, fetch_episode)
            base_steps = 0
            if skip_local_terminal:
                saved_reviews = _review_rows(review_root / f"{item['bucket']}.jsonl")
                skip_local_terminal = not any(
                    (item["sample_id"], step_index) not in saved_reviews
                    for step_index in range(base_steps, len(state.get("steps", [])))
                )
            status = "already_terminal" if skip_local_terminal else "running"
            if not skip_local_terminal:
                active.append({**item, "state": state, "base_steps": base_steps})
            results.append({"id": item["sample_id"], "bucket": item["bucket"],
                            "episode": str(item["local_episode_dir"] / "episode.json"),
                            "status": status})

        round_index = 0
        while active:
            round_index += 1
            for item in active:
                state = _read_episode(item["local_episode_dir"] / "episode.json", item["sample_id"])
                item["state"] = state
            if _review_pending_steps(active, review_root, local_root, log_root,
                                     batch_index, round_index):
                continue

            active = [item for item in active if not _is_terminal(item["state"])]
            if not active:
                break

            eligible = []
            for item in active:
                state = item["state"]
                new_steps = _new_teacher_steps(state)
                if new_steps >= MAX_NEW_TEACHER_STEPS:
                    continue
                step_index = len(state["steps"])
                visible = _refresh_visible_state(item)
                visible = _save_before_state(item["local_episode_dir"], step_index, visible)
                item.update(visible=visible, step_index=step_index, new_steps=new_steps)
                eligible.append(item)
            active = eligible
            if not active:
                break

            review_indexes = {}
            actions = {}
            teacher_states = []
            for item in active:
                action_path = _action_path(item["local_episode_dir"], item["step_index"])
                item["action_path"] = action_path
                review_path = review_root / f"{item['bucket']}.jsonl"
                item["review_path"] = review_path
                if review_path not in review_indexes:
                    review_indexes[review_path] = _review_rows(review_path)
                key = (item["sample_id"], item["step_index"])
                item["saved_review"] = review_indexes[review_path].get(key)
                if action_path.exists():
                    action = _load_action(action_path, item["sample_id"])
                else:
                    action = None
                    teacher_states.append(item["visible"])
                if action is None and item["saved_review"] is not None:
                    raise ValueError(f"{review_path} has a review without a saved action for {key}")
                if action is not None:
                    actions[item["sample_id"]] = action

            if teacher_states:
                teacher_log = _next_log_path(
                    log_root, _batch_log_name("teacher", batch_index, round_index,
                                              [state["id"] for state in teacher_states]))
                for action in codex_teacher.collect_actions(teacher_states, teacher_log):
                    item = next(item for item in active if item["sample_id"] == action["id"])
                    _save_action(item["action_path"], action)
                    actions[action["id"]] = action

            # Cache each BEFORE state and action before executing anything. Reviews
            # run after real steps, but receive only these frozen inputs.
            for item in active:
                action = actions[item["sample_id"]]
                if action.get("name") not in {tool["function"]["name"]
                                               for tool in item["visible"]["tools"]}:
                    raise ValueError(f"{item['sample_id']}: action is no longer available in the visible state")
                arguments = json.dumps(action["arguments"], ensure_ascii=False, allow_nan=False)
                run_task_module(OBJECT_TEACHER_MODULE, "step", "--episode",
                                _remote_episode_file(item["remote_episode_dir"]),
                                "--note", action.get("note", ""), "--name", action["name"],
                                "--arguments", arguments)

            review_states = [item["visible"] for item in active if item["saved_review"] is None]
            if review_states:
                with tempfile.TemporaryDirectory(prefix="codex_review_", dir=local_root) as staging_name:
                    staging = Path(staging_name)
                    for state in review_states:
                        _write_json(staging / f"{state['id']}.action.json", actions[state["id"]])
                    review_log = _next_log_path(
                        log_root, _batch_log_name("review", batch_index, round_index,
                                                  [state["id"] for state in review_states]))
                    reviews = codex_teacher.review_actions(review_states, staging, review_log)
                review_by_id = {row["id"]: row for row in reviews}
                if set(review_by_id) != {state["id"] for state in review_states} or len(reviews) != len(review_states):
                    raise ValueError("reviewer must return exactly one row for every before-state")
                for item in active:
                    if item["saved_review"] is not None:
                        continue
                    review = review_by_id[item["sample_id"]]
                    review_row = {"id": item["sample_id"], "step": item["step_index"],
                                  "use_action": review["use_action"],
                                  "note_supported": review["note_supported"],
                                  "reason": review["reason"]}
                    _append_review(item["review_path"], review_row,
                                   review_indexes[item["review_path"]])

            # Only now fetch the actual outcomes; the reviewer saw cached BEFORE states.
            for item in active:
                if not _fetch_remote_episode(fetch_episode, item["remote_episode_dir"],
                                             item["local_episode_dir"]):
                    raise FileNotFoundError(
                        f"remote episode disappeared after step for {item['sample_id']}"
                    )
                after = _read_episode(item["local_episode_dir"] / "episode.json", item["sample_id"])
                if len(after.get("steps", [])) != item["step_index"] + 1:
                    raise ValueError(f"{item['sample_id']}: real step did not append exactly one episode event")
                if _is_terminal(after):
                    continue
                if _new_teacher_steps(after) < MAX_NEW_TEACHER_STEPS:
                    fetch_visible_state(item["remote_episode_dir"], item["local_episode_dir"])

        # Capture terminal/max-step status from the actual remote episode after the batch settles.
        for row in results[-len(batch):]:
            state = _read_episode(Path(row["episode"]), row["id"])
            if state.get("complete"):
                row["status"] = "complete"
            elif state.get("terminal_status"):
                row["status"] = str(state["terminal_status"])
            elif _new_teacher_steps(state) >= MAX_NEW_TEACHER_STEPS:
                row["status"] = "max_new_steps"
            elif row["status"] == "running":
                row["status"] = "stopped"
    return results
