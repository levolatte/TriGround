import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "enrich_city_queries.py"
SPEC = importlib.util.spec_from_file_location("enrich_city_queries", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_extracts_class_from_all_weak_templates():
    assert MODULE.object_name("The car") == "car"
    assert MODULE.object_name("The leftmost street light") == "street light"
    assert MODULE.object_name("The 3rd person from the left") == "person"


def test_region_phrase_uses_normalized_box_center():
    assert MODULE.region_phrase([0.0, 0.0, 0.2, 0.2]) == "in the upper-left part of the image"
