from copy import deepcopy
import inspect

import numpy as np
import pytest
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from tools.grounding_structure import (StructureConfig, StructureBundle, ModalStructure,
    GeometryHead, depth_features, geometry_loss, object_losses)
from tools.structured_grounder import QwenStructureRuntime, numeric_for_row, finalized_revision


torch.set_num_threads(1)


def config():
    return StructureConfig(32, width=16, heads=4, dropout=0.)


def fixture(modalities=("rgb", "ir", "depth")):
    grids = torch.tensor([[1, 4, 4]]*len(modalities))
    exits = [torch.randn(len(modalities)*4, 32) for _ in range(4)]
    return exits, grids, list(modalities), torch.randn(32), [.1, .2, .6, .8]


@pytest.mark.parametrize("modalities", [("rgb", "ir"), ("rgb", "depth"), ("rgb", "ir", "depth")])
def test_zero_initialization_all_visual_routes_and_evidence(modalities):
    model = ModalStructure(config()).eval()
    args = fixture(modalities)
    out, evidence, readout = model(*args)
    assert all(torch.equal(a, b) for a, b in zip(args[0], out))
    assert torch.count_nonzero(evidence) == 0
    assert all(torch.equal(g, torch.full_like(g, .5)) for g in readout["gates"].values())


def test_numeric_physical_scale_invalid_and_grid_order():
    raw = np.array([[0, 100], [1000, 19999]], dtype=np.uint16)
    out = depth_features(raw, [1, 4, 4])
    assert out.shape == (4, 7)
    assert out[:, 2].tolist() == [0, 1, 1, 0]
    assert out[:, 3].tolist() == [0, 1, 0, 0]
    assert out[:, 4].tolist() == [0, 0, 0, 1]
    assert out[1, 0] < out[2, 0]
    assert torch.equal(out[:, -2:], torch.tensor([[.25,.25],[.75,.25],[.25,.75],[.75,.75]]))


def test_missing_and_unknown_depth_never_reads_raw_path():
    row = {"depth_encoding": "millimeter", "depth_raw": "DOES_NOT_EXIST", "modalities": ["rgb", "depth"]}
    assert numeric_for_row({**row, "missing_modalities_actual": ["depth"]}, [[1,4,4]]*2, 2) is None
    assert numeric_for_row({**row, "degraded_modality": "depth"}, [[1,4,4]]*2, 2) is None
    assert numeric_for_row({**row, "depth_encoding": "unknown"}, [[1,4,4]]*2, 2) is None


def test_private_routes_and_numeric_have_real_gradients_and_effects():
    model = ModalStructure(config()).eval()
    args = fixture()
    numeric = depth_features(np.full((8,8), 2000, dtype=np.uint16), [1,4,4])
    out, evidence, _ = model(*args, numeric=numeric)
    loss = sum(x.square().mean() for x in out) + (evidence-torch.ones_like(evidence)).square().mean()
    loss.backward()
    assert all(m.up.weight.grad.abs().sum() > 0 for layers in model.adapters.values() for m in layers)
    assert all(m.weight.grad.abs().sum() > 0 for m in model.numeric_out)
    assert all(m.weight.grad.abs().sum() > 0 for m in model.evidence_out.values())
    torch.optim.SGD(model.parameters(), lr=.01).step()
    changed, evidence, _ = model(*args, numeric=numeric)
    disabled, _, _ = model(*args, numeric=numeric, disabled=("numeric",))
    assert all(torch.equal(x[:4], original[:4]) for x, original in zip(changed, args[0]))
    assert all(not torch.equal(x[8:], y[8:]) for x, y in zip(changed, disabled))
    assert evidence.abs().sum() > 0


def test_binding_loss_uses_all_reference_permutations():
    args = fixture()
    _, _, readout = ModalStructure(config())(*args)
    sup = {"rgb_objects": [[0,0,.5,.5], [.5,.5,1,1]],
           "bindings": {"depth": {"target": [0,0,.5,.5], "references": [[.5,0,1,.5],[0,.5,.5,1]]}}}
    a = object_losses(readout, sup, args[1], args[2])
    swapped = deepcopy(sup)
    swapped["bindings"]["depth"]["references"].reverse()
    b = object_losses(readout, swapped, args[1], args[2])
    assert torch.equal(a[1], b[1])
    assert a[0] > 0 and a[1] > 0


def test_geometry_loss_has_no_gradient_to_language_or_visual_input():
    model = GeometryHead(config())
    context = torch.randn(32, requires_grad=True)
    exits = [torch.randn(4,32, requires_grad=True) for _ in range(4)]
    box = model(context, exits, [1,4,4], [.1,.1,.5,.5])
    torch.testing.assert_close(box, torch.tensor([.1,.1,.5,.5]))
    geometry_loss(box, [.2,.2,.8,.8]).backward()
    assert context.grad is None and all(x.grad is None for x in exits)
    assert model.out.weight.grad.abs().sum() > 0


def tiny_native():
    cfg = Qwen3VLConfig(
        vision_config=dict(depth=4, hidden_size=32, intermediate_size=64, num_heads=4,
                           patch_size=2, spatial_merge_size=2, temporal_patch_size=2,
                           out_hidden_size=32, deepstack_visual_indexes=[0,1,2], num_position_embeddings=16),
        text_config=dict(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         rope_scaling={"rope_type":"default", "mrope_section":[1,1,2], "mrope_interleaved":True}),
        image_token_id=100, video_token_id=101, vision_start_token_id=102, vision_end_token_id=103)
    model = Qwen3VLForConditionalGeneration(cfg).eval().requires_grad_(False)
    ids = torch.tensor([[1,102,100,100,100,100,103,2,102,100,100,100,100,103,20,21,22]])
    inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                  pixel_values=torch.randn(32,24), image_grid_thw=torch.tensor([[1,4,4],[1,4,4]]))
    if 'mm_token_type_ids' in inspect.signature(model.forward).parameters:
        inputs['mm_token_type_ids']=(ids==100).long()
    return model, inputs


def test_real_qwen_hooks_zero_equivalence_and_no_answer_leakage(tmp_path):
    torch.manual_seed(2030)
    model, inputs = tiny_native()
    bundle = StructureBundle(config()).eval()
    runtime = QwenStructureRuntime(model, bundle)
    row = {"modalities": ["rgb", "ir"]}
    runtime.begin(row, inputs["image_grid_thw"], 15)
    with torch.no_grad():
        original = model(**inputs, use_cache=False).logits
    context = runtime.hidden.clone()
    runtime.begin(row, inputs["image_grid_thw"], 15, revision=True, context=context, proposal=[.1,.2,.6,.8])
    corrected = model(**inputs, use_cache=False).logits
    assert torch.equal(original, corrected)
    hidden, box = runtime.hidden.clone(), runtime.box.clone()
    altered = {**inputs, "input_ids": inputs["input_ids"].clone()}
    altered["input_ids"][0,15:] = torch.tensor([40,41])
    runtime.begin(row, inputs["image_grid_thw"], 15, revision=True, context=context, proposal=[.1,.2,.6,.8])
    model(**altered, use_cache=False)
    assert torch.equal(hidden, runtime.hidden) and torch.equal(box, runtime.box)
    geometry_loss(runtime.box, [.2,.2,.7,.9]).backward()
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is None for p in bundle.modal.parameters())
    bundle.save(tmp_path)
    loaded = StructureBundle.load(tmp_path).eval()
    assert all(torch.equal(value, loaded.state_dict()[key]) for key,value in bundle.state_dict().items())
    runtime.close()


def test_keep_literal_and_parse_failure_full_denominator():
    first = {"prediction": [.1,.2,.6,.8], "raw_text": '{"bbox_2d": [100, 200, 600, 800]}'}
    assert finalized_revision(first, {"raw_text": '{"action":"keep"}'}, continuous=True)[:2] == (first["raw_text"], first["prediction"])
    assert finalized_revision(first, {"raw_text": "bad"}) == ("",None,"parse_failure")
