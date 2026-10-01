import json

from experiments.evidence_decision import collect_codex_episodes as loop
from experiments.evidence_decision.object_teacher import STATE_SCHEMA


def test_student_prefix_does_not_consume_new_teacher_steps():
    state = {"sample_id": "recovery", "prefix_events": [{}, {}, {}], "steps": []}
    assert loop._new_teacher_steps(state) == 0
    state["steps"].append({"index": 0})
    assert loop._new_teacher_steps(state) == 1


def test_recovery_first_teacher_action_is_blind_reviewed(tmp_path, monkeypatch):
    episode = tmp_path / "episodes" / "recovery" / "sample"
    episode.mkdir(parents=True)
    action = {"id": "sample", "name": "inspect", "arguments": {"ids": ["C"]}, "note": "Compare."}
    before = {"id": "sample", "messages": [{"role": "user", "content": "real prefix"}],
              "tools": [{"function": {"name": "inspect"}}]}
    loop._write_json(episode / "episode.json", {
        "schema_version": STATE_SCHEMA, "sample_id": "sample", "prefix_events": [{}, {}, {}],
        "steps": [{"index": 0, "call": {"name": "inspect", "arguments": {"ids": ["C"]}}}],
    })
    loop._write_json(loop._action_path(episode, 0), action)
    loop._write_json(loop._before_state_path(episode, 0), before)
    seen = []

    def review(states, staging, log):
        seen.extend(states)
        assert json.loads((staging / "sample.action.json").read_text()) == action
        return [{"id": "sample", "use_action": True, "note_supported": True, "reason": "Visible comparison."}]

    monkeypatch.setattr(loop.codex_teacher, "review_actions", review)
    item = {"sample_id": "sample", "bucket": "recovery", "local_episode_dir": episode}
    assert loop._review_pending_steps([item], tmp_path / "reviews", tmp_path, tmp_path / "logs", 1, 1)
    assert seen == [before]
    saved = loop._review_rows(tmp_path / "reviews" / "recovery.jsonl")
    assert ("sample", 0) in saved
