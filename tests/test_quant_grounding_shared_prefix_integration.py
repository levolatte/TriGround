"""CPU contracts for the actual Swift installer, without a Swift/GPU runtime."""

from __future__ import annotations

from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch
import torch.nn.functional as F

from tools.quant_grounding_shared_prefix import install_shared_prefix_training
from tools.train_quant_grounding import install_dapo_loss_normalizer


@pytest.fixture
def group_factory(monkeypatch):
    swift_grpo = ModuleType("swift.rlhf_trainers.grpo_trainer")
    swift_grpo.selective_log_softmax = lambda logits, labels: F.log_softmax(logits, -1).gather(
        -1, labels.unsqueeze(-1)).squeeze(-1)
    swift_grpo.entropy_from_logits = lambda logits: -(F.softmax(logits, -1) * F.log_softmax(logits, -1)).sum(-1)
    monkeypatch.setitem(sys.modules, "swift.rlhf_trainers.grpo_trainer", swift_grpo)
    import tools.qwen35_shared_prefix as shared

    def create(generations=2):
        prefix, suffix, left = 5, 3, 2
        ids = torch.zeros((generations, left + prefix + suffix), dtype=torch.long)
        ids[:, left:left + prefix] = torch.arange(1, prefix + 1)
        for index in range(generations):
            ids[index, -suffix:] = torch.arange(6 + index, 6 + index + suffix)
        mask = torch.ones_like(ids)
        mask[:, :left] = 0
        positions = torch.zeros((4, *ids.shape), dtype=torch.long)
        positions[:, :, left:] = torch.arange(prefix + suffix)
        unique_grid = torch.ones((3, 3), dtype=torch.long)
        unique_pixels = torch.arange(6.0).reshape(3, 2)
        inputs = {"input_ids": ids, "attention_mask": mask, "position_ids": positions,
                  "image_grid_thw": unique_grid.repeat(generations, 1),
                  "pixel_values": unique_pixels.repeat(generations, 1)}
        grpo_batch = SimpleNamespace(completion_mask=torch.ones((generations, suffix)),
                                     num_items_in_batch=torch.tensor(float(generations * suffix)))
        samples = [SimpleNamespace(extra={"id": "question-1", "occurrence_id": "occ:1"},
                                   images=["rgb.png", "ir.png", "depth.png"])
                   for _ in range(generations)]
        visual_calls, forward_calls = [], []

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.visual = torch.nn.Linear(2, 2).requires_grad_(False)
                self.base = SimpleNamespace(model=SimpleNamespace(
                    visual=self.visual, get_image_features=self.get_image_features))

            def get_base_model(self):
                return self.base

            def get_image_features(self, pixels, grid, return_dict):
                visual_calls.append({"pixels": pixels.clone(), "grid": grid.clone(),
                                     "grad_enabled": torch.is_grad_enabled(), "return_dict": return_dict})
                return SimpleNamespace(pooler_output=[pixels + 10])

        model = Model().train()

        def shared_forward(actual_model, **kwargs):
            forward_calls.append({**kwargs, "grad_enabled": torch.is_grad_enabled()})
            # Each token's value identifies its own logit row, making the first
            # answer's shift from the final prompt token directly observable.
            logits = (kwargs["input_ids"].float() * actual_model.weight).unsqueeze(-1) * F.one_hot(
                torch.tensor(0), 32).float()
            return SimpleNamespace(logits=logits[:, -kwargs["logits_to_keep"]:])

        monkeypatch.setattr(shared, "shared_prefix_forward", shared_forward)

        class Trainer:
            def __init__(self):
                self.model = model
                self.template = SimpleNamespace(padding_free=False, sequence_parallel_size=1)
                self.accelerator = SimpleNamespace(num_processes=1, autocast=nullcontext)
                self.dynamic_num_samples = False
                self.args = SimpleNamespace(per_device_train_batch_size=generations)
                self.temperature = 2.0
                self.prepared_calls = 0
                self.old_ref = []

            def _prepare_batch_inputs(self, _samples):
                self.prepared_calls += 1
                with torch.no_grad():
                    # Mirrors the native old-policy and disabled-adapter reference paths.
                    for _ in ("old", "reference"):
                        self.old_ref.append(self._get_per_token_logps_and_entropies(
                            model, inputs, grpo_batch, compute_entropy=True))
                return [{"model_inputs": inputs, "grpo_batch": grpo_batch}]

            def _get_per_token_logps_and_entropies(self, actual_model, model_inputs, batch,
                                                  compute_entropy=False, origin_data=None):
                return self._get_logps_via_local_forward(actual_model, model_inputs,
                                                        batch.completion_mask.shape[-1],
                                                        model_inputs["input_ids"], compute_entropy)

            def _get_logps_via_local_forward(self, *_args, **_kwargs):
                return "original-local-forward"

        trainer = Trainer()
        originals = {name: getattr(trainer, name) for name in (
            "_prepare_batch_inputs", "_get_per_token_logps_and_entropies", "_get_logps_via_local_forward")}
        return SimpleNamespace(trainer=trainer, model=model, inputs=inputs, batch=grpo_batch,
                               samples=samples, visual_calls=visual_calls, forward_calls=forward_calls,
                               originals=originals, prefix=prefix, suffix=suffix, generations=generations)

    return create


@pytest.mark.parametrize("generations", [2, 8])
def test_one_visual_encode_reused_by_old_reference_and_differentiable_current(group_factory, generations):
    group = group_factory(generations)
    audit, restore = install_shared_prefix_training(group.trainer, {"num_generations": generations})
    group.trainer._prepare_batch_inputs(group.samples)
    logps, entropies = group.trainer._get_per_token_logps_and_entropies(
        group.model, group.inputs, group.batch, compute_entropy=True)
    assert len(group.visual_calls) == 1
    assert group.visual_calls[0]["grad_enabled"] is False
    assert group.visual_calls[0]["grid"].shape == (3, 3)
    torch.testing.assert_close(group.visual_calls[0]["pixels"], torch.arange(6.0).reshape(3, 2))
    assert len(group.forward_calls) == 3
    assert [call["grad_enabled"] for call in group.forward_calls] == [False, False, True]
    features = group.forward_calls[0]["frozen_image_features"]
    assert all(call["frozen_image_features"] is features for call in group.forward_calls)
    assert features.requires_grad is False
    assert logps.shape == entropies.shape == (generations, 3)
    for prior_logps, prior_entropy in group.trainer.old_ref:
        torch.testing.assert_close(prior_logps, logps)
        torch.testing.assert_close(prior_entropy, entropies)
    logps.sum().backward()
    assert group.model.weight.grad is not None and group.model.weight.grad.abs() > 0
    assert group.model.visual.weight.grad is None
    assert audit["groups"] == audit["visual_encodes"] == audit["with_grad_calls"] == 1
    assert audit["without_grad_calls"] == 2
    restore()


def test_first_answer_uses_final_prompt_row_and_original_multimodal_positions(group_factory):
    group = group_factory()
    audit, restore = install_shared_prefix_training(group.trainer, {"num_generations": 2})
    group.trainer._prepare_batch_inputs(group.samples)
    logps, entropy = group.trainer._get_per_token_logps_and_entropies(
        group.model, group.inputs, group.batch, compute_entropy=False)
    assert entropy is None
    call = group.forward_calls[-1]
    assert call["logits_to_keep"] == group.suffix + 1
    assert call["plan"].prefix_length == group.prefix
    assert call["plan"].starts == (2, 2)
    assert call["position_ids"] is group.inputs["position_ids"]
    expected_first = -torch.log(torch.exp(torch.tensor(5.0 / group.trainer.temperature)) + 31)
    torch.testing.assert_close(logps[:, 0], expected_first.expand(2))
    assert audit["first_group"] == {
        "id": "question-1", "image_count": 3, "image_grid_thw": [[1, 1, 1]] * 3,
        "prefix_length": 5, "suffix_lengths": [3, 3], "compact_tokens": 11, "expanded_tokens": 16,
    }
    restore()


@pytest.mark.parametrize("generations", [2, 8])
def test_full_group_microbatch_keeps_native_token_denominator_compensation(group_factory, generations):
    group = group_factory(generations)
    native_tokens = group.batch.num_items_in_batch.clone()
    _, restore_normalizer = install_dapo_loss_normalizer(
        group.trainer, arm="B", train_micro_batch_size=generations, runtime_optimization=True)
    _, restore_shared = install_shared_prefix_training(group.trainer, {"num_generations": generations})
    batches = group.trainer._prepare_batch_inputs(group.samples)
    assert len(batches) == 1
    torch.testing.assert_close(batches[0]["grpo_batch"].num_items_in_batch, native_tokens * generations)
    torch.testing.assert_close(native_tokens, torch.tensor(float(3 * generations)))
    restore_shared()
    restore_normalizer()
    assert group.trainer._prepare_batch_inputs == group.originals["_prepare_batch_inputs"]


@pytest.mark.parametrize("field,value", [("id", "question-2"), ("occurrence_id", "occ:2")])
def test_distinct_dataset_id_or_occurrence_cannot_share(group_factory, field, value):
    group = group_factory()
    group.samples[1].extra[field] = value
    _, restore = install_shared_prefix_training(group.trainer, {"num_generations": 2})
    with pytest.raises(ValueError, match="distinct question occurrences"):
        group.trainer._prepare_batch_inputs(group.samples)
    assert group.trainer.prepared_calls == 0 and not group.visual_calls
    restore()


@pytest.mark.parametrize("media", [[], ["different.png", "ir.png", "depth.png"]])
def test_missing_or_different_source_media_cannot_share(group_factory, media):
    group = group_factory()
    group.samples[1].images = media
    _, restore = install_shared_prefix_training(group.trainer, {"num_generations": 2})
    with pytest.raises(ValueError, match="source images"):
        group.trainer._prepare_batch_inputs(group.samples)
    assert group.trainer.prepared_calls == 0 and not group.visual_calls
    restore()


@pytest.mark.parametrize("tensor", ["image_grid_thw", "pixel_values"])
def test_same_paths_with_different_encoded_image_tensors_cannot_share(group_factory, tensor):
    group = group_factory()
    group.inputs[tensor][-1, -1] += 1
    _, restore = install_shared_prefix_training(group.trainer, {"num_generations": 2})
    with pytest.raises(ValueError, match="image tensors differ"):
        group.trainer._prepare_batch_inputs(group.samples)
    assert not group.visual_calls
    restore()


def test_unequal_prompt_lengths_rejected_before_shared_plan(group_factory):
    group = group_factory()
    group.batch.completion_mask[1, -1] = 0
    _, restore = install_shared_prefix_training(group.trainer, {"num_generations": 2})
    with pytest.raises(ValueError, match="unequal prompt lengths"):
        group.trainer._prepare_batch_inputs(group.samples)
    assert not group.visual_calls
    restore()


def test_restore_and_explicit_group_state_boundary(group_factory):
    group = group_factory()
    _, restore = install_shared_prefix_training(group.trainer, {"num_generations": 2})
    with pytest.raises(RuntimeError, match="first be built during rollout preparation"):
        group.trainer._get_per_token_logps_and_entropies(group.model, group.inputs, group.batch)
    with pytest.raises(RuntimeError, match="explicit prepared question state"):
        group.trainer._get_logps_via_local_forward(group.model, group.inputs, 3, group.inputs["input_ids"])
    restore()
    for name, original in group.originals.items():
        assert getattr(group.trainer, name) == original
    assert group.trainer._get_logps_via_local_forward() == "original-local-forward"


def test_group_size_frozen_visual_and_runtime_contract(group_factory):
    group = group_factory()
    _, restore = install_shared_prefix_training(group.trainer, {"num_generations": 2})
    with pytest.raises(ValueError, match="Expected one G2 group"):
        group.trainer._prepare_batch_inputs(group.samples[:1])
    group.model.visual.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen visual"):
        group.trainer._prepare_batch_inputs(group.samples)
    restore()
    group.trainer.args.per_device_train_batch_size = 1
    with pytest.raises(ValueError, match="one full question per microbatch"):
        install_shared_prefix_training(group.trainer, {"num_generations": 2})
