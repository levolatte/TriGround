"""The pre-Trainer capacity check must exercise the allocated CUDA device."""

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.evidence_decision.train import _place_capacity_probe


class FakeModel:
    def __init__(self):
        self.device = "cpu"
        self.moves = []

    def to(self, device):
        self.moves.append(device)
        self.device = "cuda:0"
        return self


class FakeProbe:
    def __init__(self):
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self


class CapacityProbeDeviceTest(unittest.TestCase):
    def test_model_moves_to_cuda_before_probe_uses_model_device(self):
        model = FakeModel()
        probe = FakeProbe()

        placed = _place_capacity_probe(model, probe)

        self.assertEqual(model.moves, ["cuda"])
        self.assertEqual(model.device, "cuda:0")
        self.assertEqual(placed.device, model.device)


if __name__ == "__main__":
    unittest.main()
