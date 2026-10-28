"""ms-swift 4.5.3 的真实 ORM 插件，通过 --external_plugins 加载。"""

from __future__ import annotations

import sys
from pathlib import Path

from swift.rewards import ORM, orms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.grounding_reward import grounding_rewards  # noqa: E402


class Acc05BBoxReward(ORM):
    def __call__(self, completions, solution, **kwargs) -> list[float]:
        return grounding_rewards(completions, solution)


class IoUBBoxReward(ORM):
    def __call__(self, completions, solution, **kwargs) -> list[float]:
        return grounding_rewards(completions, solution, mode="iou")


class BinaryBBoxReward(ORM):
    def __call__(self, completions, solution, **kwargs) -> list[float]:
        return grounding_rewards(completions, solution, mode="binary")


orms["acc05_bbox"] = Acc05BBoxReward
orms["iou_bbox"] = IoUBBoxReward
orms["binary_bbox"] = BinaryBBoxReward
