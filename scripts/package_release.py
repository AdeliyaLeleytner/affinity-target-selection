"""Create a portable release tree with small Git LFS payload parts."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ['study', 'tests', 'scripts', 'configs', 'data', 'results', 'figs']
FILES = ['README.md', 'LICENSE', 'requirements.txt', 'paper.tex', 'paper.pdf',
         'supplementary.tex', 'supplementary.pdf', 'references.tex', 'sn-jnl.cls',
         'sn-nature.bst', 'study_config.json']
PART_BYTES = 6_000_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination == ROOT or destination.is_relative_to(ROOT):
        raise ValueError('Release destination must be outside the working source tree')
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Release destination must be empty')
    destination.mkdir(parents=True, exist_ok=True)
    files = [ROOT/p for p in FILES if (ROOT/p).exists()]
    files += [p for directory in DIRECTORIES for p in (ROOT/directory).rglob('*')
              if p.is_file() and p.suffix != '.pyc'
              and not any(part.startswith('.') or part in
                          {'__pycache__', 'runtime', 'build', 'rejected_evaluation'}
                          for part in p.relative_to(ROOT).parts)]
    manifest = {'format': 1, 'part_size_limit_bytes': PART_BYTES, 'files': []}
    ignored = ['__pycache__/', '.venv/', '.DS_Store', 'build/', '*.aux', '*.log', '*.out', '.mplconfig/', 'runtime/']
    for path in sorted(files):
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError('Release selection contains an external or symbolic path')
        relative = path.relative_to(ROOT)
        if any(term in str(relative).lower() for term in ['private_manifest', 'agent_log', '.env']):
            raise ValueError(f'Non-release file in selected tree: {relative}')
        if path.stat().st_size <= PART_BYTES:
            target = destination/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            continue
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        item = {'path': str(relative), 'bytes': path.stat().st_size, 'sha256': checksum, 'parts': []}
        with path.open('rb') as handle:
            index = 0
            while True:
                block = handle.read(PART_BYTES)
                if not block: break
                relative_part = Path('payloads')/(checksum+f'.{index:03d}.bin')
                target = destination/relative_part; target.parent.mkdir(exist_ok=True)
                if not target.exists(): target.write_bytes(block)
                item['parts'].append({'path': str(relative_part), 'bytes': len(block),
                                      'sha256': hashlib.sha256(block).hexdigest()})
                index += 1
        manifest['files'].append(item); ignored.append('/'+str(relative))
    (destination/'payload_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (destination/'.gitignore').write_text('\n'.join(ignored)+'\n')
    patterns = ['*.npz','*.npy','*.parquet','*.gz','*.zip','*.pt','*.ckpt','*.tsv','*.bin','*.png','*.pdf','*.cif']
    attribute_lines = [p+' filter=lfs diff=lfs merge=lfs -text\n' for p in patterns]
    for path in sorted(destination.rglob('*')):
        if path.is_file() and path.stat().st_size >= 1_000_000:
            attribute_lines.append(json.dumps(str(path.relative_to(destination)))+' filter=lfs diff=lfs merge=lfs -text\n')
    (destination/'.gitattributes').write_text(''.join(attribute_lines))
    release_files = []
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            payload = path.read_bytes()
            release_files.append({'path': str(path.relative_to(destination)),
                                  'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()})
    (destination/'release_manifest.json').write_text(json.dumps({'files':release_files},indent=2)+'\n')
    print(json.dumps({'destination':str(destination), 'source_files':len(files),
                      'split_files':len(manifest['files']), 'maximum_part_bytes':PART_BYTES}, indent=2))

if __name__ == '__main__': main()
