"""Both holdout controller arms consume the same frozen-C candidate cache."""

from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.evidence_decision.run_gpu_phase import command_for


class RunGpuPhaseTest(unittest.TestCase):
    def test_untrained_and_trained_holdout_share_frozen_c_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = str(root / "data/holdout120_candidates_frozenC.jsonl")
            for phase in ("untrained_holdout", "trained_holdout"):
                _cwd, command, _seconds = command_for(phase, 8192, root)
                option = command.index("--candidate-cache")
                self.assertEqual(command[option + 1], expected)


if __name__ == "__main__":
    unittest.main()
