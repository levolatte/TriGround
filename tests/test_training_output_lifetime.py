from dataclasses import replace
from pathlib import Path
import weakref

import torch
from torch import nn
from torch.utils.data import DataLoader

from mm_grounding import engine
from mm_grounding.config import load_config


def test_accumulation_releases_logits_before_next_forward(monkeypatch, tmp_path):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))
            self.previous_logits = None
            self.calls = 0

        def forward(self, **_):
            if self.previous_logits is not None:
                assert self.previous_logits() is None, "previous batch logits still retained"
            logits = self.weight.expand(8, 2).clone()
            self.previous_logits = weakref.ref(logits)
            self.calls += 1
            loss = logits.square().mean()
            return {"loss": loss, "token_loss": loss, "logits": logits}

    model = Model()
    config = load_config(Path(__file__).resolve().parents[1] / "configs/qwen3_vl_8b_stage2_joint.yaml")
    config = replace(config, output_dir=str(tmp_path), train=replace(
        config.train, epochs=1, phase_a_epochs=1, grad_accumulation=2,
        compact_checkpoints=True,
    ))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    monkeypatch.setattr(engine, "_optimizer", lambda *_: optimizer)
    monkeypatch.setattr(engine, "_scheduler", lambda *_: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1))
    monkeypatch.setattr(engine, "evaluate", lambda *_: {
        "acc_0.5": 0.5, "mean_iou": 0.4, "acc_0.7": 0.3, "parse_rate": 1.0,
    })
    saved = []
    monkeypatch.setattr(engine, "save_checkpoint", lambda path, *_, **kw: saved.append((Path(path).name, kw["compact"])))
    monkeypatch.setattr(engine, "load_model_checkpoint", lambda *_: None)
    loader = DataLoader([{"input_ids": torch.tensor([1])}] * 3, batch_size=1)
    engine._run_phase(
        model, loader, loader, loader, None, None, config, torch.device("cpu"),
        torch.amp.GradScaler("cpu", enabled=False), torch.float32, "a", 1, 0,
    )
    assert model.calls == 3
    assert not torch.equal(model.weight, torch.ones(2))
    assert saved == [("last_phase_a.pt", False), ("best_phase_a.pt", True)]
    assert '"event": "best_mean_iou"' in (tmp_path / "metrics.jsonl").read_text()
