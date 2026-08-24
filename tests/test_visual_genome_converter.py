import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "convert_visual_genome.py"
SPEC = importlib.util.spec_from_file_location("convert_visual_genome", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_normalize_visual_genome_xywh_box():
    assert MODULE.normalize_bbox([10, 20, 30, 40], 100, 200) == [0.1, 0.1, 0.4, 0.3]


def test_invalid_box_is_rejected():
    assert MODULE.normalize_bbox([10, 20, 0, 40], 100, 200) is None


def test_split_is_deterministic_by_image():
    first = MODULE.split_for_image("000108", 0.1, 2026)
    assert first == MODULE.split_for_image("000108", 0.1, 2026)
