"""把代码与配置打成交付包（不含数据与虚拟环境）。

排除项的理由：
* `.venv`（3.7 GB）—— 评委应按 requirements 自建，打包无意义且超大。
* `docs`（152 MB）—— 内部研究记录，与复现无关；只在技术方案里引用结论。
* `experiments` / `output` —— 历史产物。
* 各类缓存（`__pycache__` / `.pytest_cache` / `.ruff_cache`）。

包含 `src/mm_grounding` 是有意的：它是早期 RDT 架构的实现，虽然当前方案不用它，
但保留能让评审看到方案演进，也避免"代码里缺一块"的疑问。
"""

from __future__ import annotations

import shutil
from pathlib import Path
import sys

CODE = Path(r"F:\AIC\code")
DEST = Path(r"F:\AIC\deliverables\半决赛作品材料\代码与数据")

INCLUDE_DIRS = ("tools", "configs", "scripts", "src", "tests")
INCLUDE_FILES = ("README.md", "pyproject.toml", "requirements-quant-train.txt",
                 "requirements-experiment.txt", "MODEL_REGISTRY.md", "LICENSE")
EXCLUDE_NAMES = {"__pycache__", ".pytest_cache", ".ruff_cache", ".venv", ".git"}
EXCLUDE_SUFFIX = {".pyc", ".pyo"}


def ignored(directory: str, names: list[str]) -> set[str]:
    return {name for name in names
            if name in EXCLUDE_NAMES or Path(name).suffix in EXCLUDE_SUFFIX}


def main() -> int:
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)

    copied = 0
    total = 0
    for name in INCLUDE_DIRS:
        source = CODE / name
        if not source.is_dir():
            print(f"  跳过（不存在）: {name}")
            continue
        shutil.copytree(source, DEST / name, ignore=ignored)
        count = sum(1 for _ in (DEST / name).rglob("*") if _.is_file())
        size = sum(_.stat().st_size for _ in (DEST / name).rglob("*") if _.is_file())
        copied += count
        total += size
        print(f"  {name:<10} {count:>4} 文件  {size / 1e6:8.2f} MB")

    for name in INCLUDE_FILES:
        source = CODE / name
        if source.is_file():
            shutil.copy2(source, DEST / name)
            copied += 1
            total += source.stat().st_size
            print(f"  {name:<10}   1 文件  {source.stat().st_size / 1e3:8.1f} KB")

    print(f"\n合计 {copied} 文件，{total / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
