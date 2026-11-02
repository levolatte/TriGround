"""把技术报告 Markdown 转成 DOCX，供 LibreOffice 转 PDF。

只处理这份报告实际用到的 Markdown 构造：标题（#/##/###）、段落、粗体、行内代码、
无序列表、管道表格、围栏代码块、水平线。不追求通用性——报告是我们自己写的，
构造集合是封闭的，写一个通用的 Markdown 解析器反而会引入不必要的不确定性。

**中文字体必须显式设 w:eastAsia**：python-docx 的 run.font.name 只设西文字体，
中文会回退到渲染器默认字体，可能出现缺字形或字重不一致。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

BODY_FONT = "Microsoft YaHei"
CODE_FONT = "Consolas"


def set_font(run, name: str = BODY_FONT, size: float | None = None,
             bold: bool | None = None, mono: bool = False) -> None:
    """同时设西文与东亚字体，并可选设字号/加粗。"""
    font = run.font
    font.name = CODE_FONT if mono else name
    if size is not None:
        font.size = Pt(size)
    if bold is not None:
        font.bold = bold
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    target = CODE_FONT if mono else name
    rfonts.set(qn("w:ascii"), target)
    rfonts.set(qn("w:hAnsi"), target)
    rfonts.set(qn("w:eastAsia"), name if not mono else BODY_FONT)


def join_lines(lines: list[str]) -> str:
    """把硬换行合并成一个段落。

    中文之间直接相接；两侧都是 ASCII 字母数字时补一个空格，避免英文单词粘连。
    """
    out = ""
    for line in lines:
        if out and out[-1].isascii() and out[-1].isalnum() and line[:1].isascii() and line[:1].isalnum():
            out += " "
        out += line
    return out


INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")


def add_inline(paragraph, text: str, size: float = 10.5) -> None:
    """按 **粗体** 与 `代码` 切分，其余按普通文本写入。"""
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            run = paragraph.add_run(piece[2:-2])
            set_font(run, size=size, bold=True)
        elif piece.startswith("`") and piece.endswith("`") and len(piece) > 2:
            run = paragraph.add_run(piece[1:-1])
            set_font(run, size=size - 0.5, mono=True)
            run.font.color.rgb = RGBColor(0xA0, 0x30, 0x30)
        else:
            run = paragraph.add_run(piece)
            set_font(run, size=size)


def is_table_row(line: str) -> bool:
    return line.strip().startswith("|") and line.strip().endswith("|")


def is_separator_row(line: str) -> bool:
    return bool(re.fullmatch(r"\|[\s:|-]+\|", line.strip()))


def split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def convert(markdown: Path, output: Path) -> None:
    document = Document()
    # 页边距收到 2 cm，报告里的表格较宽，需要更多正文宽度。
    for section in document.sections:
        section.left_margin = section.right_margin = Pt(56)
        section.top_margin = section.bottom_margin = Pt(56)

    lines = markdown.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        # 围栏代码块
        if stripped.startswith("```"):
            index += 1
            block: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            index += 1
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.space_before = Pt(3)
            paragraph.paragraph_format.space_after = Pt(6)
            paragraph.paragraph_format.left_indent = Pt(12)
            run = paragraph.add_run("\n".join(block))
            set_font(run, size=9, mono=True)
            continue

        # 表格
        if is_table_row(line):
            rows: list[list[str]] = []
            while index < len(lines) and is_table_row(lines[index]):
                if not is_separator_row(lines[index]):
                    rows.append(split_row(lines[index]))
                index += 1
            if rows:
                width = max(len(r) for r in rows)
                table = document.add_table(rows=len(rows), cols=width)
                table.style = "Table Grid"
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                for r, cells in enumerate(rows):
                    for c in range(width):
                        cell = table.cell(r, c)
                        cell.text = ""
                        paragraph = cell.paragraphs[0]
                        add_inline(paragraph, cells[c] if c < len(cells) else "", size=9)
                        if r == 0:
                            for run in paragraph.runs:
                                run.font.bold = True
            document.add_paragraph()
            continue

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped[level:].strip()
            if level == 1:
                paragraph = document.add_heading(level=0)
                run = paragraph.add_run(text)
                set_font(run, size=18, bold=True)
            else:
                paragraph = document.add_heading(level=min(level - 1, 4))
                run = paragraph.add_run(text)
                set_font(run, size=15 if level == 2 else 12.5, bold=True)
            index += 1
            continue

        if stripped in ("---", "***"):
            paragraph = document.add_paragraph()
            rpr = paragraph.add_run()._element.get_or_add_rPr()
            from docx.oxml import OxmlElement
            border = OxmlElement("w:pBdr")
            bottom = OxmlElement("w:bottom")
            bottom.set(qn("w:val"), "single")
            bottom.set(qn("w:sz"), "6")
            bottom.set(qn("w:color"), "999999")
            border.append(bottom)
            rpr.append(border)
            index += 1
            continue

        if re.match(r"^[*\-]\s+", stripped):
            text = re.sub(r"^[*\-]\s+", "", stripped)
            paragraph = document.add_paragraph(style="List Bullet")
            add_inline(paragraph, text)
            index += 1
            continue

        if stripped:
            # Markdown 语义：连续的非空行属于同一段落，直到空行为止。
            # 必须先合并再解析行内标记——否则跨行的 **粗体** 会各留一个孤立的 **
            # 匹配不上，原样漏进正文（技术报告 L21-22 就有一处跨行粗体）。
            buffer = [stripped]
            index += 1
            while index < len(lines):
                nxt = lines[index]
                probe = nxt.strip()
                if (not probe or probe.startswith("#") or probe.startswith("```")
                        or is_table_row(nxt) or probe in ("---", "***")
                        or re.match(r"^[*\-]\s+", probe)):
                    break
                buffer.append(probe)
                index += 1
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.space_after = Pt(4)
            paragraph.paragraph_format.first_line_indent = Pt(0)
            add_inline(paragraph, join_lines(buffer))
            continue

        index += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)
    print(f"已写出 {output}（{output.stat().st_size:,} 字节）", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.markdown, args.output)
