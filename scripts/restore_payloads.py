"""Restore split release payloads locally, validating every part and final file."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    manifest = ROOT/'payload_manifest.json'
    if not manifest.exists():
        print('No split payload manifest: this checkout already contains original files.')
        return
    restored = 0
    for item in json.loads(manifest.read_text())['files']:
        target = ROOT/item['path']
        if not target.resolve().is_relative_to(ROOT):
            raise ValueError('Payload destination escapes the repository')
        if target.exists():
            if target.stat().st_size != item['bytes'] or digest(target) != item['sha256']:
                raise ValueError(f'Refusing to overwrite conflicting file: {item["path"]}')
            continue
        for part in item['parts']:
            source = ROOT/part['path']
            if not source.resolve().is_relative_to(ROOT):
                raise ValueError('Payload part escapes the repository')
            if not source.is_file() or source.stat().st_size != part['bytes'] or digest(source) != part['sha256']:
                raise ValueError(f'Missing/incorrect payload part: {part["path"]}. Check Git LFS checkout or archive completeness.')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name+'.restoring')
        try:
            with temporary.open('xb') as handle:
                for part in item['parts']:
                    with (ROOT/part['path']).open('rb') as source:
                        for block in iter(lambda: source.read(1024*1024), b''):
                            handle.write(block)
            if digest(temporary) != item['sha256']:
                raise ValueError(f'Reconstructed hash mismatch: {item["path"]}')
            os.replace(temporary, target)
            restored += 1
        finally:
            if temporary.exists():
                temporary.unlink()
    print(f'Restored and verified {restored} files; all released payloads are intact.')

if __name__ == '__main__': main()
