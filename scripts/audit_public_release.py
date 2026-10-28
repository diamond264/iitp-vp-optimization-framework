"""Check the code-only release and, if present, every reachable Git commit.

This allowlist complements manual review; it is not a comprehensive secret scanner.
Private inputs, binaries and experiment outputs are not allowed in a release.
Run before copying private inputs into the working tree.
"""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    '.gitignore', 'README.md', 'requirements.txt',
    'configs/teacher_labels.yaml', 'configs/distil_default.yaml',
    'configs/distil_hard_only.yaml', 'scripts/run_distill.sh',
    'scripts/extract_teacher_labels.sh', 'scripts/audit_public_release.py',
    *{'src/distil_vp/' + n + '.py' for n in (
        '__init__', 'model', 'dataset', 'train', 'teacher_labels',
        'evaluate', 'compare', 'llm_compare', 'latency_bench')},
}
PATTERNS = [
    re.compile(r'/(?:home|Users)/[A-Za-z0-9_.-]+/'),
    re.compile(r'gh[pousr]_' + r'[A-Za-z0-9]{30,}'),
    re.compile(r'github_pat_' + r'[A-Za-z0-9_]{30,}'),
    re.compile(r'AKIA' + r'[0-9A-Z]{16}'),
    re.compile(r'-----BEGIN ' + r'(?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
]
issues = []

def check(name, data, context):
    if name not in ALLOWED:
        issues.append(f'{context}: file is not on the release allowlist: {name}')
        return
    if len(data) > 200_000:
        issues.append(f'{context}: oversized source file: {name}')
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        issues.append(f'{context}: non-UTF-8 or binary file: {name}')
        return
    if '\0' in text or any(p.search(text) for p in PATTERNS):
        issues.append(f'{context}: private path, potential secret, or binary content: {name}')

found = set()
for p in ROOT.rglob('*'):
    relative = p.relative_to(ROOT)
    if '.git' in relative.parts:
        continue
    if p.is_symlink():
        issues.append(f'working tree: symbolic link is not allowed: {relative}')
    elif p.is_file():
        found.add(relative.as_posix())
        check(relative.as_posix(), p.read_bytes(), 'working tree')
for missing in sorted(ALLOWED - found):
    issues.append('working tree: required file missing: ' + missing)

def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])

if (ROOT / '.git').exists():
    for commit in git('rev-list', '--all').decode().splitlines():
        for raw in git('ls-tree', '-r', '-z', commit).split(b'\0'):
            if not raw:
                continue
            metadata, filename = raw.split(b'\t', 1)
            mode, kind, oid = metadata.decode().split()
            name = filename.decode()
            if mode not in {'100644', '100755'} or kind != 'blob':
                issues.append(f'{commit[:12]}: unsupported file type: {name}')
            else:
                check(name, git('cat-file', 'blob', oid), commit[:12])

if issues:
    print('\n'.join(issues), file=sys.stderr)
    sys.exit(1)
print(f'PASS: {len(found)} code/config/documentation files; no disallowed artifacts found.')
