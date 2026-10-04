"""Release restoration preserves bytes and rejects unsafe or conflicting inputs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('restore_payloads', Path(__file__).resolve().parents[1]/'scripts/restore_payloads.py')
restore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore)


class PayloadRestorationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.old_root = restore.ROOT
        restore.ROOT = self.root
        payload = b'input\x00data\xff' * 100
        (self.root/'payloads').mkdir()
        parts = []
        for i, block in enumerate([payload[:123], payload[123:]]):
            name = f'payloads/part{i}.bin'
            (self.root/name).write_bytes(block)
            parts.append({'path':name,'bytes':len(block),'sha256':hashlib.sha256(block).hexdigest()})
        self.payload = payload
        self.item = {'path':'data/source.bin','bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'parts':parts}
        self.write_manifest()

    def tearDown(self):
        restore.ROOT = self.old_root
        self.temp.cleanup()

    def write_manifest(self):
        (self.root/'payload_manifest.json').write_text(json.dumps({'files':[self.item]}))

    def test_roundtrip_and_existing_identical_file(self):
        restore.main()
        self.assertEqual((self.root/'data/source.bin').read_bytes(),self.payload)
        restore.main()
        self.assertEqual((self.root/'data/source.bin').read_bytes(),self.payload)

    def test_corrupted_part_is_rejected_without_destination(self):
        (self.root/'payloads/part0.bin').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'payload part'):
            restore.main()
        self.assertFalse((self.root/'data/source.bin').exists())

    def test_existing_different_file_is_preserved(self):
        (self.root/'data').mkdir()
        (self.root/'data/source.bin').write_bytes(b'unique local data')
        with self.assertRaisesRegex(ValueError,'overwrite'):
            restore.main()
        self.assertEqual((self.root/'data/source.bin').read_bytes(),b'unique local data')

    def test_part_path_cannot_escape_repository(self):
        self.item['parts'][0]['path']='../outside.bin'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError,'part escapes'):
            restore.main()

    def test_destination_cannot_escape_repository(self):
        self.item['path']='../outside.bin'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError,'destination escapes'):
            restore.main()

if __name__=='__main__': unittest.main()
