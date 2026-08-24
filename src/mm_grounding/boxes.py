from __future__ import annotations

import torch


def cxcywh_to_xyxy(boxes: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    center_x, center_y, width, height = boxes.unbind(dim=-1)
    converted = torch.stack(
        (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ),
        dim=-1,
    )
    return converted.clamp(0.0, 1.0) if clamp else converted


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    left, top, right, bottom = boxes.unbind(dim=-1)
    return torch.stack(
        (
            (left + right) / 2,
            (top + bottom) / 2,
            right - left,
            bottom - top,
        ),
        dim=-1,
    )


def box_iou_aligned(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(first[..., :2], second[..., :2])
    bottom_right = torch.minimum(first[..., 2:], second[..., 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
    first_area = (first[..., 2:] - first[..., :2]).clamp(min=0).prod(dim=-1)
    second_area = (second[..., 2:] - second[..., :2]).clamp(min=0).prod(dim=-1)
    union = first_area + second_area - intersection
    return intersection / union.clamp(min=1e-7)


def generalized_iou_aligned(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    iou = box_iou_aligned(first, second)
    enclosure_tl = torch.minimum(first[..., :2], second[..., :2])
    enclosure_br = torch.maximum(first[..., 2:], second[..., 2:])
    enclosure = (enclosure_br - enclosure_tl).clamp(min=0).prod(dim=-1)
    first_area = (first[..., 2:] - first[..., :2]).clamp(min=0).prod(dim=-1)
    second_area = (second[..., 2:] - second[..., :2]).clamp(min=0).prod(dim=-1)
    top_left = torch.maximum(first[..., :2], second[..., :2])
    bottom_right = torch.minimum(first[..., 2:], second[..., 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
    union = first_area + second_area - intersection
    return iou - (enclosure - union) / enclosure.clamp(min=1e-7)
