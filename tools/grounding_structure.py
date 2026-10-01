"""Small trainable structures on frozen Qwen visual exits; no GT enters forward."""
from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class StructureConfig:
    hidden_size: int
    revision_structure: str = "modal"
    width: int = 256
    heads: int = 4
    layers: int = 2
    rank: int = 8
    alpha: int = 16
    dropout: float = .05
    numeric_width: int = 128
    box_decoder: str = "continuous"
    version: int = 1


def token_xy(grid, merge_size=2, *, device=None):
    t, h, w = map(int, grid)
    if h % merge_size or w % merge_size:
        raise ValueError("visual grid is not divisible by the spatial merge size")
    h, w = h // merge_size, w // merge_size
    yy, xx = torch.meshgrid((torch.arange(h, device=device)+.5)/h,
                            (torch.arange(w, device=device)+.5)/w, indexing="ij")
    return torch.stack((xx, yy), -1).reshape(-1, 2).repeat(t, 1)


def depth_features(raw, grid, merge_size=2):
    """Area statistics at merged-token resolution, preserving one physical scale."""
    if np.asarray(raw).ndim != 2:
        raise ValueError("millimeter depth must be a single-channel image")
    t, h, w = map(int, grid)
    shape = (h // merge_size, w // merge_size)
    depth = torch.as_tensor(np.asarray(raw).astype(np.float32, copy=True))[None, None]
    valid = (depth > 0) & (depth < 19999)
    logdepth = torch.log1p(depth.clamp(0, 19999)) / math.log(20000)
    pool = lambda x: F.adaptive_avg_pool2d(x.float(), shape)
    fraction = pool(valid)
    mean = pool(logdepth * valid) / fraction.clamp_min(1e-8)
    variance = (pool(logdepth.square() * valid)/fraction.clamp_min(1e-8)-mean.square()).clamp_min(0)
    features = torch.cat((mean, variance.sqrt(), fraction,
                          pool((depth > 0) & (depth < 300)), pool(depth >= 19999)), 1)
    features = features[0].permute(1, 2, 0).reshape(-1, 5).repeat(t, 1)
    return torch.cat((features, token_xy(grid, merge_size)), -1)


def proposal_tensor(proposal, device):
    if proposal is None:
        return torch.tensor([.5, .5, 1., 1., 0.], device=device)
    box = torch.as_tensor(proposal, dtype=torch.float32, device=device)
    return torch.cat(((box[:2]+box[2:])/2, box[2:]-box[:2], box.new_ones(1)))


class ExitAdapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.down = nn.Linear(cfg.hidden_size, cfg.rank, bias=False)
        self.up = nn.Linear(cfg.rank, cfg.hidden_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.scale = cfg.alpha / cfg.rank
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(self.dropout(x.float()))) * self.scale


class ReadBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads = heads
        self.qnorm, self.knorm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.q, self.k, self.v, self.out = [nn.Linear(width, width) for _ in range(4)]
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width*4),
                                nn.GELU(), nn.Dropout(dropout), nn.Linear(width*4, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory):
        n, width = query.shape
        q = self.q(self.qnorm(query)).view(n, self.heads, -1).transpose(0, 1)
        key = self.knorm(memory)
        k = self.k(key).view(len(key), self.heads, -1).transpose(0, 1)
        v = self.v(key).view(len(key), self.heads, -1).transpose(0, 1)
        logits = q @ k.transpose(-1, -2) / math.sqrt(width // self.heads)
        weights = logits.softmax(-1)
        value = (self.dropout(weights) @ v).transpose(0, 1).reshape(n, width)
        query = query + self.out(value)
        return query + self.ff(query), logits.mean(0), weights.mean(0)


class ModalStructure(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.adapters = nn.ModuleDict({m: nn.ModuleList([ExitAdapter(cfg) for _ in range(4)])
                                       for m in ("ir", "depth")})
        self.numeric = nn.Sequential(nn.Linear(7, cfg.numeric_width), nn.GELU(),
                                     nn.Linear(cfg.numeric_width, cfg.numeric_width), nn.GELU())
        self.numeric_out = nn.ModuleList([nn.Linear(cfg.numeric_width, cfg.hidden_size) for _ in range(4)])
        self.context = nn.Linear(cfg.hidden_size, cfg.width)
        self.proposal = nn.Linear(5, cfg.width)
        self.roles = nn.Parameter(torch.randn(3, cfg.width)*.02)
        self.memory = nn.ModuleDict({m: nn.Linear(cfg.hidden_size, cfg.width) for m in ("rgb", "ir", "depth")})
        self.position = nn.Linear(2, cfg.width)
        self.readers = nn.ModuleDict({m: nn.ModuleList([
            ReadBlock(cfg.width, cfg.heads, cfg.dropout) for _ in range(cfg.layers)])
            for m in ("rgb", "ir", "depth")})
        self.gates = nn.ModuleDict({m: nn.Sequential(nn.Linear(cfg.width*2, 64), nn.GELU(), nn.Linear(64, 1))
                                   for m in ("ir", "depth")})
        self.evidence_out = nn.ModuleDict({m: nn.Linear(3*cfg.width, cfg.hidden_size, bias=False)
                                           for m in ("ir", "depth")})
        for module in (*self.numeric_out, *self.evidence_out.values()):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        for gate in self.gates.values():
            nn.init.zeros_(gate[-1].weight)
            nn.init.zeros_(gate[-1].bias)

    def forward(self, exits, grids, modalities, context, proposal, numeric=None,
                disabled=(), merge_size=2):
        if len(exits) != 4 or len(modalities) != len(grids) or len(set(modalities)) != len(modalities):
            raise ValueError("one RGB and at most one image per auxiliary modality are required")
        device = exits[0].device
        sizes = [int(torch.as_tensor(g).prod()) // merge_size**2 for g in grids]
        if any(len(x) != sum(sizes) for x in exits):
            raise ValueError("visual output/token grid mismatch")
        transformed = []
        numeric_encoded = self.numeric(numeric.to(device).float()) if numeric is not None and "numeric" not in disabled else None
        for level, output in enumerate(exits):
            parts = list(output.split(sizes))
            for i, modality in enumerate(modalities):
                if modality in self.adapters and modality not in disabled:
                    delta = self.adapters[modality][level](parts[i])
                    if modality == "depth" and numeric_encoded is not None:
                        if len(numeric_encoded) != len(parts[i]):
                            raise ValueError("numeric depth/grid mismatch")
                        delta = delta + self.numeric_out[level](numeric_encoded)
                    parts[i] = parts[i] + delta.to(parts[i].dtype)
            transformed.append(torch.cat(parts))
        parts = transformed[0].split(sizes)
        memories = {m: self.memory[m](part.float()) + self.position(token_xy(g, merge_size, device=device))
                    for m, part, g in zip(modalities, parts, grids, strict=True)}
        query = self.roles + self.context(context.detach().float().to(device)) + self.proposal(proposal_tensor(proposal, device))
        maps, scores, gates = {}, {}, {}
        for block in self.readers["rgb"]:
            query, scores["rgb"], maps["rgb"] = block(query, memories["rgb"])
        evidence = query.new_zeros(self.cfg.hidden_size)
        for modality in ("ir", "depth"):
            if modality not in memories or modality in disabled:
                continue
            auxiliary = query
            for block in self.readers[modality]:
                auxiliary, scores[modality], maps[modality] = block(auxiliary, memories[modality])
            gates[modality] = self.gates[modality](torch.cat((query, auxiliary), -1)).sigmoid()
            evidence = evidence + self.evidence_out[modality]((gates[modality]*auxiliary).flatten())
        return transformed, evidence, {"maps": maps, "scores": scores, "gates": gates}


def region_weights(box, grid, merge_size=2, *, device=None):
    """Fractional patch/box intersection: also covers objects smaller than a token."""
    xy = token_xy(grid, merge_size, device=device)
    _, h, w = map(int, grid)
    half = xy.new_tensor([merge_size/w, merge_size/h]) / 2
    box = xy.new_tensor(box)
    return (torch.minimum(xy+half, box[2:])-torch.maximum(xy-half, box[:2])).clamp_min(0).prod(-1)


def object_losses(readout, supervision, grids, modalities, merge_size=2):
    """Only this loss function receives annotated regions/competitors."""
    zero = readout["maps"]["rgb"].sum()*0
    competition, binding = [], []
    def weights(box, modality):
        return region_weights(box, grids[modalities.index(modality)], merge_size,
                              device=zero.device)
    boxes = supervision.get("rgb_objects", [])
    if len(boxes) > 1:
        logits = readout["scores"]["rgb"][0]
        values = []
        for box in boxes:
            area = weights(box, "rgb")
            values.append((logits*area).sum()/area.sum().clamp_min(1e-12))
        competition.append(F.cross_entropy(torch.stack(values)[None], torch.zeros(1, dtype=torch.long, device=zero.device)))
    for modality, regions in supervision.get("bindings", {}).items():
        if modality not in readout["maps"]:
            continue
        maps = readout["maps"][modality]
        def cost(slot, box):
            area = weights(box, modality)
            # Attention mass in a partially covered cell is weighted by its occupancy.
            _, h, w = map(int, grids[modalities.index(modality)])
            occupancy = area * (h//merge_size) * (w//merge_size)
            return -(maps[slot]*occupancy).sum().clamp_min(1e-8).log()
        if regions.get("target") is not None:
            binding.append(cost(0, regions["target"]))
        references = regions.get("references", [])
        if references:
            if len(references) > 2:
                raise ValueError("the reader supports at most two reference objects")
            assignments = list(itertools.permutations((1, 2), len(references))) if regions.get("unordered", True) else [(1, 2)[:len(references)]]
            costs = [torch.stack([cost(slot, box) for slot, box in zip(order, references)]).mean() for order in assignments]
            binding.append(torch.stack(costs).min())
    return (torch.stack(competition).mean() if competition else zero,
            torch.stack(binding).mean() if binding else zero)


class GeometryHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.context = nn.Linear(cfg.hidden_size, cfg.width)
        self.proposal = nn.Linear(5, cfg.width)
        self.memory = nn.ModuleList([nn.Linear(cfg.hidden_size, cfg.width) for _ in range(4)])
        self.position = nn.Linear(2, cfg.width)
        # No dropout: G must not consume the correction model's random stream.
        self.blocks = nn.ModuleList([ReadBlock(cfg.width, cfg.heads, 0.) for _ in range(cfg.layers)])
        self.out = nn.Linear(cfg.width, 4)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, context, rgb_exits, grid, proposal, merge_size=2):
        context = context.detach().float()
        memory = sum(project(x.detach().float()) for project, x in zip(self.memory, rgb_exits, strict=True))/4
        memory = memory + self.position(token_xy(grid, merge_size, device=memory.device))
        base = proposal_tensor(proposal, memory.device)
        query = (self.context(context)+self.proposal(base))[None]
        for block in self.blocks:
            query, _, _ = block(query, memory)
        cxcywh = (torch.logit(base[:4].clamp(1e-6, 1-1e-6)) + self.out(query)[0]).sigmoid()
        return torch.cat((cxcywh[:2]-cxcywh[2:]/2, cxcywh[:2]+cxcywh[2:]/2)).clamp(0, 1)


def geometry_loss(prediction, target):
    target = prediction.new_tensor(target)
    inter = (torch.minimum(prediction[2:], target[2:])-torch.maximum(prediction[:2], target[:2])).clamp_min(0).prod()
    area = (prediction[2:]-prediction[:2]).clamp_min(0).prod()
    target_area = (target[2:]-target[:2]).prod()
    union = area+target_area-inter
    cover = (torch.maximum(prediction[2:], target[2:])-torch.minimum(prediction[:2], target[:2])).clamp_min(0).prod()
    giou = inter/union.clamp_min(1e-8)-(cover-union)/cover.clamp_min(1e-8)
    return 5*F.l1_loss(prediction, target)+2*(1-giou)


class StructureBundle(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.revision_structure not in ("plain", "modal"):
            raise ValueError(config.revision_structure)
        self.modal = ModalStructure(config) if config.revision_structure == "modal" else None
        self.geometry = GeometryHead(config) if config.box_decoder == "continuous" else None

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory/"structure_config.json").write_text(json.dumps(asdict(self.config), indent=2)+"\n", encoding="utf-8")
        torch.save(self.state_dict(), directory/"structure.pt")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        config = StructureConfig(**json.loads((directory/"structure_config.json").read_text()))
        model = cls(config)
        model.load_state_dict(torch.load(directory/"structure.pt", map_location="cpu", weights_only=True), strict=True)
        return model
