"""Opt-in Swift integration for one explicit question's differentiable G answers."""

from __future__ import annotations


def install_shared_prefix_training(trainer, config):
    import torch
    from swift.rlhf_trainers.grpo_trainer import selective_log_softmax, entropy_from_logits
    from tools.qwen35_shared_prefix import build_shared_prefix_plan, shared_prefix_forward

    generations = int(config.get("num_generations", 8))
    if (trainer.template.padding_free or trainer.template.sequence_parallel_size != 1
            or trainer.accelerator.num_processes != 1 or trainer.dynamic_num_samples
            or trainer.args.per_device_train_batch_size != generations):
        raise ValueError("shared prefix requires one device, dense padding, and one full question per microbatch")
    original_prepare = trainer._prepare_batch_inputs
    original_logps = trainer._get_logps_via_local_forward
    original_per_token = trainer._get_per_token_logps_and_entropies
    active_samples = None
    active_state = None
    audit = {"enabled": True, "groups": 0, "with_grad_calls": 0, "without_grad_calls": 0,
             "visual_encodes": 0, "expanded_tokens": 0, "compact_tokens": 0,
             "num_generations": generations, "backend": "fla", "first_group": None}

    def prepare(samples):
        nonlocal active_samples
        if len(samples) != generations:
            raise ValueError(f"Expected one G{generations} group, got {len(samples)}")
        identities = [(sample.extra["occurrence_id"], sample.extra["id"]) for sample in samples]
        if len(set(identities)) != 1:
            raise ValueError("Shared-prefix group mixes distinct question occurrences")
        # Dataset identity, plus media paths, is authoritative; image tokens alone are not.
        images = tuple(str(image) for image in samples[0].images)
        if not images or any(tuple(str(image) for image in sample.images) != images for sample in samples[1:]):
            raise ValueError("Shared-prefix group has missing or unequal source images")
        active_samples = samples
        try:
            return original_prepare(samples)
        finally:
            active_samples = None

    def make_state(model, inputs, batch):
        if active_samples is None:
            raise RuntimeError("Shared-prefix group must first be built during rollout preparation")
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        positions = inputs["position_ids"]
        prefixes = mask.sum(-1) - batch.completion_mask.sum(-1)
        if not torch.all(prefixes == prefixes[0]):
            raise ValueError("Same-question group has unequal prompt lengths")
        plan = build_shared_prefix_plan(ids, mask, positions, int(prefixes[0]))
        image_count = len(active_samples[0].images)
        grid = inputs["image_grid_thw"]
        pixels = inputs["pixel_values"]
        if grid.shape[0] != generations * image_count:
            raise ValueError("Repeated image-grid count differs from explicit question media")
        unique_grid = grid[:image_count]
        pixel_count = int(unique_grid.prod(-1).sum())
        unique_pixels = pixels[:pixel_count]
        if pixels.shape[0] != pixel_count * generations:
            raise ValueError("Pixel rows differ from the original image grids")
        for index in range(1, generations):
            if (not torch.equal(unique_grid, grid[index * image_count:(index + 1) * image_count])
                    or not torch.equal(unique_pixels, pixels[index * pixel_count:(index + 1) * pixel_count])):
                raise ValueError("Same-question image tensors differ; sharing is invalid")
        base = model.get_base_model()
        if any(parameter.requires_grad for parameter in base.model.visual.parameters()):
            raise ValueError("Only frozen visual features can be reused")
        with torch.no_grad(), trainer.accelerator.autocast():
            output = base.model.get_image_features(unique_pixels, unique_grid, return_dict=True)
            features = torch.cat(output.pooler_output, dim=0).detach()
        audit["groups"] += 1
        audit["visual_encodes"] += 1
        if audit["first_group"] is None:
            audit["first_group"] = {
                "id": active_samples[0].extra["id"], "image_count": image_count,
                "image_grid_thw": unique_grid.cpu().tolist(), "prefix_length": plan.prefix_length,
                "suffix_lengths": list(plan.suffix_lengths), "compact_tokens": plan.compact_length,
                "expanded_tokens": plan.total_tokens,
            }
        return {"plan": plan, "frozen_image_features": features}

    def per_token(model, model_inputs, grpo_batch, compute_entropy=False, origin_data=None):
        nonlocal active_state
        if not hasattr(grpo_batch, "_quant_shared_prefix"):
            grpo_batch._quant_shared_prefix = make_state(model, model_inputs, grpo_batch)
        active_state = grpo_batch._quant_shared_prefix
        try:
            return original_per_token(model, model_inputs, grpo_batch,
                                      compute_entropy=compute_entropy, origin_data=origin_data)
        finally:
            active_state = None

    def local_forward(model, model_inputs, logits_to_keep, input_ids, compute_entropy=False):
        if active_state is None:
            raise RuntimeError("Shared-prefix logps require an explicit prepared question state")
        plan = active_state["plan"]
        with trainer.accelerator.autocast():
            logits = shared_prefix_forward(
                model, plan=plan, input_ids=input_ids, attention_mask=model_inputs["attention_mask"],
                position_ids=model_inputs["position_ids"],
                frozen_image_features=active_state["frozen_image_features"],
                logits_to_keep=logits_to_keep + 1, checkpoint_layers=True,
            ).logits[:, :-1].float()
        logits = logits / trainer.temperature
        logps = selective_log_softmax(logits, input_ids[:, -logits_to_keep:])
        entropies = entropy_from_logits(logits) if compute_entropy else None
        audit["with_grad_calls" if torch.is_grad_enabled() else "without_grad_calls"] += 1
        audit["expanded_tokens"] += plan.total_tokens
        audit["compact_tokens"] += plan.compact_length
        return logps, entropies

    trainer._prepare_batch_inputs = prepare
    trainer._get_per_token_logps_and_entropies = per_token
    trainer._get_logps_via_local_forward = local_forward

    def restore():
        trainer._prepare_batch_inputs = original_prepare
        trainer._get_per_token_logps_and_entropies = original_per_token
        trainer._get_logps_via_local_forward = original_logps

    return audit, restore
