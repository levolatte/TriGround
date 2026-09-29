"""Assemble the September ABV audit evidence without weights or raw image batches."""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

TEXT = {'.py', '.sh', '.ps1', '.md', '.txt', '.json', '.jsonl', '.csv',
        '.toml', '.yaml', '.yml', '.html', '.css', '.js', '.log', '.ini'}
MEDIA = {'.png', '.jpg', '.jpeg', '.svg', '.pdf', '.docx'}
SKIP = {'.git', '.venv', '__pycache__', '.pytest_cache', 'node_modules'}


def build(root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    records = []

    def copy(src: Path, relative: Path) -> None:
        dest = output / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        records.append((relative.as_posix(), str(src), src.stat().st_size, 'original'))

    def tree(src: Path, relative: Path, excluded: set[str] = frozenset()) -> None:
        for p in sorted(src.rglob('*')):
            rel = p.relative_to(src)
            if not p.is_file() or any(x in SKIP | excluded for x in rel.parts):
                continue
            if p.name in {'AGENTS.md', 'tokenizer.json'} or p.suffix.lower() not in TEXT | MEDIA:
                continue
            copy(p, relative / rel)

    repo = root / 'code'
    paths = subprocess.check_output(['git', 'ls-files', '--cached', '--others',
                                     '--exclude-standard', '-z'], cwd=repo).decode().split('\0')
    for name in sorted(set(filter(None, paths))):
        p = repo / name
        if p.is_file() and p.name != 'AGENTS.md' and not any(x in SKIP for x in p.parts):
            copy(p, Path('code') / name)
    tree(root / 'contest', Path('contest'))
    tree(root / 'docs/research', Path('docs/research'))
    run = root / 'results/triground_abv_execution_20260928'
    tree(run, Path('results/triground_abv_execution_20260928'), {'assets'})
    # Original deployment snapshot is retained, explicitly distinct from final cloud code.
    if (run / 'code_snapshot.tar').exists():
        copy(run / 'code_snapshot.tar', Path('results/triground_abv_execution_20260928/code_snapshot.tar'))

    # These reduced previews support review; they are never substituted into manifests.
    from PIL import Image
    previews = []
    for p in sorted((run / 'deployment_600_seed2026/assets').glob('*')):
        if p.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
            continue
        rel = Path('asset_previews') / (p.stem + '.jpg')
        dest = output / rel
        dest.parent.mkdir(exist_ok=True)
        with Image.open(p) as im:
            im = im.convert('RGB')
            size = im.size
            im.thumbnail((960, 720))
            im.save(dest, quality=82)
        previews.append({'asset': p.name, 'preview': rel.as_posix(), 'original_size': size})
        records.append((rel.as_posix(), str(p), dest.stat().st_size, 'resized_preview'))
    (output / 'asset_previews/index.json').write_text(json.dumps(previews, ensure_ascii=False, indent=2), encoding='utf-8')
    for name in ['README.md', 'AUDIT_REQUEST.md']:
        copy(repo / 'docs/audit-20260929' / name, Path(name))
    with (output / 'FILE_INDEX.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bundle_path', 'source_path', 'bytes', 'kind'])
        w.writerows(records)
    suspicious = []
    secret = re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY|\b(?:hf_|ghp_)[A-Za-z0-9]{20,}|\bsk-[A-Za-z0-9]{25,}')
    for p in output.rglob('*'):
        if p.is_file() and p.suffix.lower() in TEXT:
            if secret.search(p.read_text(encoding='utf-8', errors='replace')):
                suspicious.append(str(p.relative_to(output)))
    if suspicious:
        raise ValueError(f'Credential-like content requires review: {suspicious}')
    archive = output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(output))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        count = len(z.namelist())
    print(json.dumps({'archive': str(archive), 'files': count,
                      'bytes': archive.stat().st_size, 'preview_count': len(previews)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('F:/AIC'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.output)
