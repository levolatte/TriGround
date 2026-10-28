"""ms-swift 4.5.3 SFT loss: equal weight per example, not per answer token."""

import torch


def sample_mean_from_token_losses(token_losses: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Reduce ms-swift's flat, next-token cross-entropies over each sample.

    The template has already replaced non-training tokens (including empty-think
    tokens under the binary default+ignore_empty_think mask) with -100. Its EOS
    label remains supervised. ms-swift's per_token_loss_func rolls labels left
    once; the ignored first label must therefore be present in each sequence.
    """
    assert token_losses.numel() == labels.numel()
    shifted_labels = torch.roll(labels, shifts=-1, dims=-1)
    valid = shifted_labels.ne(-100)
    counts = valid.sum(dim=-1)
    if (counts == 0).any():
        raise ValueError("sample_mean requires a supervised answer token in every sample")
    losses = token_losses.reshape_as(labels)
    return ((losses * valid).sum(dim=-1) / counts).mean()


def register_swift_loss() -> None:
    from swift.loss import BaseLoss, loss_map
    from swift.trainers import per_token_loss_func

    class GroundingSampleMeanLoss(BaseLoss):
        def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs):
            if loss_scale is not None:
                raise ValueError("grounding_sample_mean requires a binary loss mask represented by labels")
            token_losses = per_token_loss_func(outputs, labels)
            loss = sample_mean_from_token_losses(token_losses, labels)
            if self.trainer.model.training:
                # Transformers 5 does not divide a custom compute_loss_func again.
                # This value also handles a short final accumulation group.
                loss = loss / self.trainer.current_gradient_accumulation_steps
            return loss

    loss_map["grounding_sample_mean"] = GroundingSampleMeanLoss


# runpy.run_path(..., run_name="__main__") exposes the torch-only helper to CPU
# tests without importing ms-swift. --external_plugins imports this as a module.
if __name__ != "__main__":
    register_swift_loss()
