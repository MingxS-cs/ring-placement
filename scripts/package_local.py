"""Package exactly the committed local deliverable and verify every member."""
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
    root = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    if output.exists() or output.with_suffix('.receipt.json').exists():
        raise FileExistsError('Choose a new destination to preserve earlier packages')
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=root).decode()
    if status.strip():
        raise ValueError('Commit and verify the deliverable before packaging it')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip()
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().rstrip('\0').split('\0')
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in names:
            data = (root/name).read_bytes()
            archive.writestr('ring-placement/'+name, data)
            hashes[name] = digest(data)
        archive.writestr('ring-placement/LOCAL_COMMIT.txt', commit+'\n')
    with zipfile.ZipFile(output) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError('Corrupt archive member: '+corrupt)
        for name, expected in hashes.items():
            if digest(archive.read('ring-placement/'+name)) != expected:
                raise ValueError('Archive byte mismatch: '+name)
    result = dict(commit=commit, archive=str(output), archive_sha256=digest(output.read_bytes()),
                  bytes=output.stat().st_size, tracked_files=len(names), crc_and_all_member_hashes_verified=True,
                  publication='Local archive only', member_sha256=hashes)
    output.with_suffix('.receipt.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key:value for key,value in result.items() if key != 'member_sha256'}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='../ring-placement.zip')
    package(parser.parse_args().output)
