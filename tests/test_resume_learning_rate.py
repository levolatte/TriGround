import torch

from mm_grounding.engine import _override_optimizer_scheduler_lrs


def test_resume_learning_rate_override_survives_scheduler_step():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=4e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    _override_optimizer_scheduler_lrs(optimizer, scheduler, [2e-5])
    optimizer.step()
    scheduler.step()

    assert optimizer.param_groups[0]["lr"] == 2e-5
    assert scheduler.get_last_lr() == [2e-5]
    assert scheduler.base_lrs == [2e-5]
