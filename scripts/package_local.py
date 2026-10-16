"""Create a ZIP from a clean checkout and verify its contents."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def package(output):
    project_root = Path(__file__).resolve().parents[1]
    archive_path = Path(output).resolve()
    if archive_path.exists() or archive_path.with_suffix('.receipt.json').exists():
        raise FileExistsError('Choose a new destination to preserve earlier packages')
    working_tree_status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=project_root).decode()
    if working_tree_status.strip():
        raise ValueError('Commit and verify the deliverable before packaging it')
    commit_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=project_root).decode().strip()
    tracked_paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=project_root).decode().rstrip('\0').split('\0')
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    member_hashes = {}
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative_path in tracked_paths:
            file_bytes = (project_root / relative_path).read_bytes()
            archive.writestr('ring-placement/' + relative_path, file_bytes)
            member_hashes[relative_path] = digest(file_bytes)
        archive.writestr('ring-placement/LOCAL_COMMIT.txt', commit_sha + '\n')
    with zipfile.ZipFile(archive_path) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise ValueError('Corrupt archive member: ' + corrupt_member)
        for relative_path, expected in member_hashes.items():
            if digest(archive.read('ring-placement/' + relative_path)) != expected:
                raise ValueError('Archive byte mismatch: ' + relative_path)
    receipt = dict(
        commit=commit_sha,
        archive=str(archive_path),
        archive_sha256=digest(archive_path.read_bytes()),
        bytes=archive_path.stat().st_size,
        tracked_files=len(tracked_paths),
        crc_and_all_member_hashes_verified=True,
        publication='Local archive only',
        member_sha256=member_hashes,
    )
    archive_path.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'member_sha256'}, indent=2))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='../ring-placement.zip')
    package(parser.parse_args().output)
