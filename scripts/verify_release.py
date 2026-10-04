"""Verify an untouched release tree, including ordinary and Git LFS asset bytes."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def main():
    manifest=ROOT/'release_manifest.json'
    if not manifest.exists():
        raise SystemExit('No release manifest: use this check on the exported release tree.')
    files=json.loads(manifest.read_text())['files']
    for item in files:
        path=ROOT/item['path']
        if not path.resolve().is_relative_to(ROOT): raise ValueError('Manifest path escapes release root')
        if not path.is_file(): raise ValueError(f'Missing release file: {item["path"]}')
        with path.open('rb') as stream:
            if stream.read(43).startswith(b'version https://git-lfs.github.com/spec/'):
                raise ValueError(f'Unresolved Git LFS pointer: {item["path"]}')
        if path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
            raise ValueError(f'Release byte mismatch: {item["path"]}')
    print(f'All {len(files)} released files verified, including unsplit binary assets.')

if __name__=='__main__': main()
