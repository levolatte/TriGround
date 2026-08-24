import torch
from torch import nn

from mm_grounding.lora import LoRALinear, inject_vision_lora, vision_lora_parameters


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(8, 24)
        self.proj = nn.Linear(8, 8)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()


class Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block(), Block()])


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = Vision()


def test_injects_only_last_vision_blocks_and_preserves_output():
    torch.manual_seed(2)
    model = Backbone()
    inputs = torch.randn(3, 8)
    expected = model.visual.blocks[-1].attn.qkv(inputs)
    count = inject_vision_lora(model, rank=2, alpha=4, dropout=0, last_n_blocks=1)
    actual = model.visual.blocks[-1].attn.qkv(inputs)
    assert count == 2
    assert not isinstance(model.visual.blocks[0].attn.qkv, LoRALinear)
    assert isinstance(model.visual.blocks[-1].attn.qkv, LoRALinear)
    assert torch.equal(expected, actual)
    assert len(list(vision_lora_parameters(model))) == 4
