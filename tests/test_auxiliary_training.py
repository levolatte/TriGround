from types import SimpleNamespace

from mm_grounding.engine import _geometry_gradient_scale


def _config(enabled=True, head_only=100, warmup=400):
    return SimpleNamespace(
        model=SimpleNamespace(auxiliary_bbox_enabled=enabled),
        train=SimpleNamespace(
            auxiliary_head_only_steps=head_only,
            auxiliary_geometry_warmup_steps=warmup,
        ),
    )


def test_geometry_gradient_warmup_keeps_head_learning_first():
    config = _config()
    assert _geometry_gradient_scale(config, "a", 99) == 0.0
    assert 0.0 < _geometry_gradient_scale(config, "a", 100) < 1.0
    assert _geometry_gradient_scale(config, "a", 499) == 1.0
    assert _geometry_gradient_scale(config, "b", 0) == 1.0


def test_geometry_gradient_warmup_is_disabled_for_ce_baseline():
    assert _geometry_gradient_scale(_config(enabled=False), "a", 0) == 1.0
