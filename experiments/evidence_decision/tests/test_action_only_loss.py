"""Numerical equivalence checks for sparse current-action causal loss."""

import importlib.util
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.evidence_decision.train import action_only_loss


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is available in the training environment")
class ActionOnlyLossTest(unittest.TestCase):
    def test_balanced_loss_weights_action_value_and_remaining_tokens_equally(self):
        from types import SimpleNamespace
        import torch
        from torch import nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self, values):
                super().__init__()
                self.values = nn.Parameter(values.clone())

            def forward(self, input_ids, logits_to_keep):
                return SimpleNamespace(logits=self.values[:, logits_to_keep])

        torch.manual_seed(2026)
        initial = torch.randn(1, 5, 7)
        model = Model(initial)
        reference = initial.clone().requires_grad_(True)
        labels = torch.tensor([[-100, 1, 2, 3, 4]])
        inputs = {'input_ids': torch.tensor([[0, 1, 2, 3, 4]]), 'labels': labels,
                  'action_value_mask': torch.tensor([[False, False, True, False, False]])}
        loss, _ = action_only_loss(model, inputs)
        expected = (0.5 * F.cross_entropy(reference[:, 1], labels[:, 2]) +
                    0.5 * F.cross_entropy(reference[:, [0, 2, 3]].reshape(-1, 7),
                                          labels[:, [1, 3, 4]].reshape(-1)))
        loss.backward()
        expected.backward()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(model.values.grad, reference.grad)
        self.assertEqual(int(model.values.grad[:, 4].count_nonzero()), 0)

    def test_sparse_loss_matches_full_masked_causal_ce_and_parameter_gradients(self):
        import copy
        from types import SimpleNamespace

        import torch
        from torch import nn
        import torch.nn.functional as F

        class TinyCausalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(11, 5)
                self.hidden = nn.Linear(5, 7)
                self.lm_head = nn.Linear(7, 11)
                self.seen_input = None
                self.seen_keep = None
                self.projected_hidden_length = None

            def forward(self, input_ids, attention_mask=None, logits_to_keep=None):
                self.seen_input = input_ids.detach().clone()
                hidden = torch.tanh(self.hidden(self.embedding(input_ids)))
                if logits_to_keep is not None:
                    self.seen_keep = logits_to_keep.detach().clone()
                    hidden = hidden.index_select(1, logits_to_keep)
                self.projected_hidden_length = hidden.shape[1]
                return SimpleNamespace(logits=self.lm_head(hidden))

        torch.manual_seed(2026)
        sparse_model = TinyCausalModel()
        reference_model = copy.deepcopy(sparse_model)
        input_ids = torch.tensor([[1, 3, 4, 5, 6, 2, 10, 0, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.long)
        labels = torch.full_like(input_ids, -100)
        # The supervised action ends in token 2 (<|im_end|>); the following
        # newline and padding remain masked, as does the user/prompt prefix.
        labels[0, 3:6] = input_ids[0, 3:6]
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        sparse_loss, sparse_outputs = action_only_loss(sparse_model, inputs)
        sparse_loss.backward()

        reference_outputs = reference_model(input_ids, attention_mask=attention_mask)
        reference_loss = F.cross_entropy(
            reference_outputs.logits[:, :-1, :].float().reshape(-1, reference_outputs.logits.shape[-1]),
            labels[:, 1:].reshape(-1), ignore_index=-100,
        )
        reference_loss.backward()

        torch.testing.assert_close(sparse_loss, reference_loss, rtol=1e-6, atol=1e-7)
        self.assertEqual(sparse_outputs.logits.shape, (1, 3, 11))
        self.assertEqual(sparse_model.projected_hidden_length, 3)
        self.assertEqual(sparse_model.seen_keep.tolist(), [2, 3, 4])
        self.assertTrue(torch.equal(sparse_model.seen_input, input_ids))
        self.assertEqual(int(labels[0, 5]), 2)
        self.assertEqual(int(labels[0, 6]), -100)
        for (sparse_name, sparse_param), (reference_name, reference_param) in zip(
                sparse_model.named_parameters(), reference_model.named_parameters()):
            self.assertEqual(sparse_name, reference_name)
            torch.testing.assert_close(sparse_param.grad, reference_param.grad, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
