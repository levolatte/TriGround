"""CPU checks for the optional IR two-pass gate and its isolated queue."""
import json
from pathlib import Path
import sys

import pytest

from tools import run_triground_tm_ir_two_pass as ir


def test_ir_gate_requires_net_two_distinct_scenes_and_limited_rgb_harm():
    classes = {"i1": "ir", "i2": "ir", "i3": "ir", "r1": "rgb_sufficient",
               "r2": "rgb_sufficient", "r3": "rgb_sufficient", "d1": "depth"}
    scenes = {"i1": "one", "i2": "two", "i3": "three", "r1": "x",
              "r2": "y", "r3": "z", "d1": "depth"}
    classes.update({f"r{i}": "rgb_sufficient" for i in range(4,33)})
    scenes.update({f"r{i}": f"rgb{i}" for i in range(4,33)})
    available = set(classes)
    direct = {key: False for key in classes}
    direct.update(i3=True, r1=True, r2=True, r3=True)
    second = dict(direct, i1=True, i2=True, r1=False)
    result = ir.gate_result(direct, second, classes, scenes, ir_present_ids=available)
    assert result["passed"] and result["ir_net"] == 2
    assert result["ir_rescued_scene_groups"] == ["one", "two"]
    assert result["rgb_new_harm"] == 1
    assert not ir.gate_result(direct, dict(second, i3=False), classes, scenes, ir_present_ids=available)["passed"]
    assert not ir.gate_result(direct, dict(second, r2=False), classes, scenes, ir_present_ids=available)["passed"]
    same_scene = dict(scenes, i2="one")
    assert not ir.gate_result(direct, second, classes, same_scene, ir_present_ids=available)["passed"]
    with pytest.raises(ValueError, match="IDs"):
        ir.gate_result(direct, second, classes, {"i1": "one"}, ir_present_ids=available)


def test_city_copy_adds_verified_order_and_preserves_original(tmp_path):
    source = tmp_path / "city.json"
    original = [{"id": str(index), "image": [
        f"visible/{index}.png", f"infrared/{index}.png",
        f"target_v2/qwen3vl_native_sft/depth_rgb/{index}.png"],
        "conversations": [{"from": "human", "value": "<image>\n<image>\n<image>"},
                          {"from": "gpt", "value": '{"bbox_2d":[1,2,3,4]}'}]}
        for index in range(412)]
    ir.write_json(source, original)
    copy = tmp_path / "inference.json"
    assert ir.city_inference_manifest(source, copy) == 412
    assert ir.read_json(copy)[0]["image_order"] == ["rgb", "ir", "depth"]
    assert "image_order" not in ir.read_json(source)[0]
    assert ir.city_inference_manifest(source, copy) == 412
    original[0]["image"][1] = "visible/wrong.png"
    ir.write_json(source, original)
    with pytest.raises(ValueError, match="do not prove"):
        ir.city_inference_manifest(source, tmp_path / "bad.json")


def test_two_pass_stage_reuses_grounding_command_with_resume(monkeypatch, tmp_path):
    monkeypatch.setattr(ir, "grounding_stage", lambda *args, **kwargs: {
        "name": args[1], "command": ["python", "--inference-mode", "direct", "--resume"],
        "queries": 103, "expected_artifact": str(args[5]), "gpu": True})
    stage = ir.two_pass_stage({}, "A_ir_two_pass_diagnostic_normal", "/adapter",
                              "/normal", "/gt", tmp_path / "output")
    assert stage["command"] == ["python", "--inference-mode", "ir_then_ground", "--resume"]


def test_diagnosed_retries_max_two_and_budget_guard(tmp_path, monkeypatch):
    direct = tmp_path / "direct"
    direct.mkdir()
    (direct / "predictions.jsonl").write_text(
        json.dumps({"id": "one", "latency_seconds": 2.0}) + "\n", encoding="utf-8")
    stage = {"name": "A_ir_two_pass_diagnostic_normal", "queries": 1,
             "expected_artifact": str(tmp_path / "two")}
    state = {"limit_seconds": ir.BUDGET_SECONDS, "spent_seconds": 0,
             "stages": [{"name": stage["name"], "status": "failed", "run_dir": str(tmp_path)}]}
    monkeypatch.setattr(ir, "execute_stage", lambda *args: None)
    monkeypatch.setattr(ir, "assert_evaluation", lambda *args: None)
    with pytest.raises(RuntimeError, match="--retry-reason"):
        ir.execute_once(stage, tmp_path, tmp_path, state, resume=True,
                        retry_reason="", direct_output=direct)
    ir.execute_once(stage, tmp_path, tmp_path, state, resume=True,
                    retry_reason="fixed cause", direct_output=direct)
    state["stages"] *= 3
    with pytest.raises(RuntimeError, match="exhausted"):
        ir.execute_once(stage, tmp_path, tmp_path, state, resume=True,
                        retry_reason="fixed cause", direct_output=direct)
    state["stages"] = []
    state["spent_seconds"] = ir.BUDGET_SECONDS - 100
    with pytest.raises(RuntimeError, match="90%"):
        ir.execute_once(stage, tmp_path, tmp_path, state, resume=True,
                        retry_reason="", direct_output=direct)


def test_a_city_requires_explicit_historical_timing_output(tmp_path):
    with pytest.raises(ValueError, match="--baseline-a-city-output"):
        ir.city_latency_output(tmp_path, "A", None)
    historical = tmp_path / "historical_A_city412"
    assert ir.city_latency_output(tmp_path, "A", historical) == historical
    assert ir.city_latency_output(tmp_path, "M", None) == tmp_path / "evaluation/M/city412"


def test_city_timing_reference_must_contain_all_412_predictions(tmp_path, monkeypatch):
    historical = tmp_path / "historical_A_city412"
    historical.mkdir()
    (historical / "predictions.jsonl").write_text(
        json.dumps({"id": "one", "latency_seconds": 2.0}) + "\n", encoding="utf-8")
    stage = {"name": "A_ir_two_pass_city412", "queries": 412,
             "expected_artifact": str(tmp_path / "two")}
    state = {"limit_seconds": ir.BUDGET_SECONDS, "spent_seconds": 0, "stages": []}
    monkeypatch.setattr(ir, "execute_stage", lambda *args: pytest.fail("GPU stage should not start"))
    with pytest.raises(ValueError, match="1 of 412"):
        ir.execute_once(stage, tmp_path, tmp_path, state, resume=False,
                        retry_reason="", direct_output=historical)


def test_prediction_ids_are_exact_and_unique(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({"id": "x", "acc_0.5": True}) + "\n", encoding="utf-8")
    assert ir.prediction_hits(path, {"x"}) == {"x": True}
    path.write_text(path.read_text() * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="IDs"):
        ir.prediction_hits(path, {"x"})


def test_dry_run_does_not_need_or_create_city_copy(tmp_path, monkeypatch, capsys):
    from tools import run_triground_tm as tm

    city = tmp_path / "city.json"
    normal = tmp_path / "normal.json"
    ir.write_json(city, [{"id": str(i), "image": ["visible/a", "infrared/a", "depth_rgb/a"]}
                         for i in range(412)])
    ir.write_json(normal, [{"id": str(i)} for i in range(103)])
    classes = {str(i): "ir" if i < 48 else "rgb_sufficient" if i < 80 else "depth"
               for i in range(103)}
    scenes = {key: key for key in classes}
    ir.write_json(tmp_path / "classes.json", classes)
    ir.write_json(tmp_path / "scenes.json", scenes)
    setup = {"city_manifest": str(city), "city_gt": "/gt.json",
             "baseline_adapters": {"A": "/A"}}
    release = {"steps": 600, "diagnostics": {"normal": str(normal)},
               "gt_manifest": "/gt.json", "class_map": str(tmp_path / "classes.json"),
               "scene_map": str(tmp_path / "scenes.json")}
    ir.write_json(tmp_path / "config.json", setup)
    ir.write_json(tmp_path / "release.json", release)
    monkeypatch.setattr(ir, "validate_config", lambda value: None)
    monkeypatch.setattr(ir, "validate_release", lambda value: None)
    # The stage builder is real, so it must count the source City412 before the copy exists.
    setup.update({"python": "python", "model": "/model", "data_root": "/data", "repo": "/repo"})
    ir.write_json(tmp_path / "config.json", setup)
    monkeypatch.setattr(sys, "argv", ["run_triground_tm_ir_two_pass", "--config",
                                         str(tmp_path / "config.json"), "--release",
                                         str(tmp_path / "release.json"), "--output-dir",
                                         str(tmp_path / "run")])
    ir.main()
    preview = json.loads(capsys.readouterr().out)
    assert [stage["queries"] for stage in preview["conditional_city"]] == [412, 412]
    assert all(stage["command"][stage["command"].index("--manifest") + 1] ==
               str(tmp_path / "run/inputs/city412_ir_two_pass.json")
               for stage in preview["conditional_city"])
    assert not (tmp_path / "run").exists()


def test_no_ir_rgb_controls_cannot_pass_even_with_positive_ir_gain():
    classes={"i1":"ir","i2":"ir",**{f"r{i}":"rgb_sufficient" for i in range(32)}}
    scenes={key:key for key in classes}
    direct={key:key.startswith("r") for key in classes}
    revised={key:True for key in classes}
    result=ir.gate_result(direct,revised,classes,scenes,ir_present_ids={"i1","i2"})
    assert result["ir_net"]==2 and result["rgb_new_harm"]==0
    assert not result["passed"] and result["rgb_sufficient_samples"]==0
    assert result["blocked_reason"]=="fewer_than_32_rgb_sufficient_cases_with_real_ir"
