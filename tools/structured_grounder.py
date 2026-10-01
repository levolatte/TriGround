"""Native Qwen integration for two-call structural grounding (one example at a time)."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import json
import time

import numpy as np
from PIL import Image
import torch

from tools.grounding_structure import depth_features, object_losses, geometry_loss
from tools.grounding_revision import revision_prompt, parse_revision


def image_content(prompt, paths):
    chunks = prompt.split("<image>")
    if len(chunks) != len(paths)+1:
        raise ValueError("image slots and prompt do not agree")
    content = []
    for index, text in enumerate(chunks):
        if text:
            content.append({"type": "text", "text": text})
        if index < len(paths):
            with Image.open(paths[index]) as im:
                content.append({"type": "image", "image": im.convert("RGB").copy()})
    return content


def prepare_inputs(processor, row, prompt, answer=None, device="cpu"):
    messages = [{"role": "user", "content": image_content(prompt, row["image"])}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt")
    inputs.pop("token_type_ids", None)
    length = inputs["input_ids"].shape[1]
    if answer is not None:
        tail = processor.tokenizer(answer + "<|im_end|>", add_special_tokens=False, return_tensors="pt")["input_ids"]
        inputs["input_ids"] = torch.cat((inputs["input_ids"], tail), 1)
        inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(tail)), 1)
        if 'mm_token_type_ids' in inputs:
            # Transformers 5.14 returns modality IDs for M-RoPE; the answer is text.
            inputs['mm_token_type_ids']=torch.cat((inputs['mm_token_type_ids'],torch.zeros_like(tail)),1)
    if inputs["input_ids"].shape[1] > 4096:
        raise ValueError(f"{row['id']}: full answer would exceed 4096 tokens")
    return inputs.to(device), length


def numeric_for_row(row, grids, merge_size):
    # All quality interventions are represented here as well as in visible images.
    if row.get("depth_encoding") != "millimeter" or not row.get("depth_raw"):
        return None
    if "depth" in row.get("missing_modalities_actual", []) or row.get("degraded_modality") == "depth":
        return None
    if "depth" not in row["modalities"]:
        raise ValueError("raw depth supplied without a depth image slot")
    with Image.open(row["depth_raw"]) as image:
        raw = np.array(image)
    return depth_features(raw, grids[row["modalities"].index("depth")], merge_size)


class QwenStructureRuntime:
    """Hooks at real native interfaces; causal hidden states are captured before answers.

    The model owns its normal visual/rotary-position/cache path. We change neither
    token count nor its image masks. Hooks are disabled on the retained A call.
    """
    def __init__(self, model, bundle=None):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.owner = base.model
        self.bundle = bundle
        self.merge_size = self.owner.visual.spatial_merge_size
        self.handles = [self.owner.visual.register_forward_hook(self._visual),
                        self.owner.language_model.register_forward_pre_hook(self._language_input, with_kwargs=True),
                        self.owner.language_model.register_forward_hook(self._language_output)]
        self.active = False

    def begin(self, row, grids, prompt_length, *, context=None, proposal=None, revision=False, disabled=()):
        self.row, self.grids = row, grids
        self.prompt_length = prompt_length
        self.context, self.proposal = context, proposal
        self.revision, self.disabled = revision, disabled
        self.evidence = self.readout = self.box = self.hidden = self.rgb_exits = None
        self.prefill_seen = False
        self.numeric = numeric_for_row(row, grids, self.merge_size) if revision and self.bundle and self.bundle.modal else None
        if revision and self.bundle and self.bundle.modal and context is None:
            raise ValueError("modal correction needs the actual retained-grounder context")
        self.active = True

    def _float_context(self, tensor):
        return torch.autocast(tensor.device.type, enabled=False) if tensor.device.type in ("cpu", "cuda") else nullcontext()

    def _visual(self, module, args, output):
        if not self.active:
            return output
        # The project has 4.57 locally and 5.14 on its actual GPU host.
        # 5.14 returns a ModelOutput; its pooler output is the merged-token stream.
        if isinstance(output, tuple):
            main, deepstack = output
        else:
            main, deepstack = output.pooler_output, output.deepstack_features
        exits = [main, *deepstack]
        sizes = [int(g.prod())//self.merge_size**2 for g in self.grids]
        rgb_index = self.row["modalities"].index("rgb")
        self.rgb_exits = [x.split(sizes)[rgb_index].detach() for x in exits]
        if not self.revision or not self.bundle or self.bundle.modal is None:
            return output
        with self._float_context(main):
            exits, self.evidence, self.readout = self.bundle.modal(
                exits, self.grids, self.row["modalities"], self.context, self.proposal,
                self.numeric, self.disabled, self.merge_size)
        if isinstance(output, tuple):
            return exits[0], exits[1:]
        output.pooler_output = exits[0]
        output.deepstack_features = exits[1:]
        return output

    def _language_input(self, module, args, kwargs):
        if not self.active or self.evidence is None or self.prefill_seen:
            return None
        embeds = kwargs.get("inputs_embeds")
        if embeds is None or embeds.shape[0] != 1 or embeds.shape[1] < self.prompt_length:
            raise ValueError("structural correction requires an unpacked single-example prefill")
        embeds = embeds.clone()
        embeds[0, self.prompt_length-1] = embeds[0, self.prompt_length-1] + self.evidence.to(embeds.dtype)
        return args, {**kwargs, "inputs_embeds": embeds}

    def _language_output(self, module, args, output):
        if not self.active or self.prefill_seen:
            return None
        states = output.last_hidden_state
        if states.shape[0] != 1 or states.shape[1] < self.prompt_length:
            raise ValueError("expected full prompt before answer generation")
        self.hidden = states[0, self.prompt_length-1].detach()
        self.prefill_seen = True
        if self.revision and self.bundle and self.bundle.geometry is not None:
            if self.rgb_exits is None:
                raise RuntimeError("geometry head was not connected to the visual forward")
            with self._float_context(states):
                self.box = self.bundle.geometry(self.hidden, self.rgb_exits,
                    self.grids[self.row["modalities"].index("rgb")], self.proposal, self.merge_size)

    def auxiliary_losses(self, supervision):
        if self.readout is None:
            return None
        return object_losses(self.readout, supervision, self.grids, self.row["modalities"], self.merge_size)

    def close(self):
        self.active = False
        for handle in self.handles:
            handle.remove()


def answer_loss(logits, input_ids, prompt_length):
    # Only answer logits need an FP32 CE buffer, not every visual/prompt token.
    return torch.nn.functional.cross_entropy(logits[0, prompt_length-1:-1].float(), input_ids[0, prompt_length:])


def training_forward(model, processor, bundle, runtime, row, proposal_row, device):
    proposal = proposal_row["prediction"]
    prompt = revision_prompt(row["prompt"], proposal)
    from tools.prepare_grounding_revision import box_iou
    from tools.prepare_qwen3vl_native_sft import bbox_to_qwen1000
    keep = box_iou(proposal, row["rgb_gt_bbox"]) >= .5
    answer = {"action": "keep"} if keep else {"action": "replace", "bbox_2d": bbox_to_qwen1000(row["rgb_gt_bbox"])}
    inputs, length = prepare_inputs(processor, row, prompt, json.dumps(answer, separators=(",", ":")), device)
    runtime.begin(row, inputs["image_grid_thw"], length, context=proposal_row["context"], proposal=proposal, revision=True)
    outputs = model(**inputs, use_cache=False)
    loss = answer_loss(outputs.logits, inputs["input_ids"], length)
    stats = {"answer_loss": float(loss.detach()), "action": answer["action"]}
    auxiliary = runtime.auxiliary_losses(row.get("supervision", {}))
    if auxiliary is not None:
        competition, binding = auxiliary
        loss = loss + .1*competition + .1*binding
        stats.update(competition_loss=float(competition.detach()), binding_loss=float(binding.detach()))
    box_loss = geometry_loss(runtime.box, row["rgb_gt_bbox"]) if runtime.box is not None else None
    if box_loss is not None:
        stats["geometry_loss"] = float(box_loss.detach())
    return loss, box_loss, stats


@torch.no_grad()
def generate(model, processor, runtime, row, *, context=None, proposal=None, revision=False):
    prompt = revision_prompt(row["prompt"], proposal) if revision else row["prompt"]
    device = next(model.parameters()).device
    inputs, length = prepare_inputs(processor, row, prompt, device=device)
    runtime.begin(row, inputs["image_grid_thw"], length, context=context, proposal=proposal, revision=revision)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    tokens = output[0, length:]
    raw = processor.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    eos = processor.tokenizer.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    result = {"raw_text": raw, "context": runtime.hidden.cpu(), "input_tokens": length,
              "image_grid_thw": inputs["image_grid_thw"].cpu().tolist(), "generated_tokens": len(tokens),
              "generation_cap_hit": len(tokens) >= 128 and not any(int(t) in eos_ids for t in tokens),
              "latency_seconds": time.perf_counter()-start}
    if runtime.box is not None:
        result["continuous_box"] = runtime.box.cpu().tolist()
    if runtime.readout is not None:
        result["readout"] = {key: {m: tensor.detach().cpu() for m, tensor in value.items()}
                             for key, value in runtime.readout.items()}
    return result


def finalized_revision(first, second, *, continuous=False):
    action, text_box = parse_revision(second["raw_text"])
    if action == "keep" and first["prediction"] is not None:
        return first["raw_text"], first["prediction"], action
    if action == "replace":
        box = second["continuous_box"] if continuous else [x/1000 for x in text_box]
        if len(box) == 4 and all(np.isfinite(x) and 0 <= x <= 1 for x in box) and box[0] < box[2] and box[1] < box[3]:
            return json.dumps({"bbox_2d": [x*1000 for x in box]}, separators=(",", ":")), box, action
    return "", None, "parse_failure"


@torch.no_grad()
def predict_two_calls(model, processor, runtime, row):
    """Live A + correction inference, without an external proposal cache or GT."""
    from tools.evaluate_pretrained_grounder import parse_generated_bbox
    if runtime.bundle is None:
        raise ValueError('load the correction structure and the default/revision adapters first')
    runtime.bundle.eval().requires_grad_(False)
    model.set_adapter('default')
    model.eval().requires_grad_(False)
    first=generate(model,processor,runtime,row)
    first['prediction']=parse_generated_bbox(first['raw_text'])
    model.set_adapter('revision')
    model.eval().requires_grad_(False)
    second=generate(model,processor,runtime,row,context=first['context'],proposal=first['prediction'],revision=True)
    arm='S' if runtime.bundle.modal is not None else 'R'
    finals={}
    for name in [arm]+(['S+G'] if runtime.bundle.geometry is not None else []):
        raw,box,action=finalized_revision(first,second,continuous=name=='S+G')
        finals[name]={'raw_text':raw,'prediction':box,'revision_action':action}
    return {'first':first,'second':second,'variants':finals,'calls':2}
