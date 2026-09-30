"""Frozen starting budgets; actual tokens are measured after processing."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    evidence_calls: int
    output_tokens: int
    finish_tokens: int
    action_tokens: int
    global_pixels: int
    tool_pixels: int
    visual_tokens: int
    cumulative_visual_tokens: int
    context_tokens: int
    search_calls: int

    def to_dict(self):
        return asdict(self)


PROFILES = {
    "fast": Profile("fast", 2, 512, 128, 192, 602112, 602112, 3072, 7200, 8192, 1),
    "capacity24": Profile("capacity24", 6, 2048, 512, 256, 602112, 602112, 6144, 28000, 12288, 3),
    "capacity48": Profile("capacity48", 6, 2048, 512, 256, 602112, 1204224, 9216, 40000, 16384, 3),
    "sft": Profile("sft", 6, 2048, 512, 256, 200704, 602112, 1600, 11200, 4096, 3),
}
