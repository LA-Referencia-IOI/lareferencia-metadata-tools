"""Regression tests for lossless, independently published vector editions."""
import gzip
import json
from pathlib import Path
import sqlite3
import struct
from tempfile import TemporaryDirectory
import unittest

from enrichment_package import lookup_vector, publish_vectors
from enrichment_package.store import digest


class VectorPackageTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.destination = self.root / 'edition'
        self.row = {'network': 'AR', 'catalogId': 'item', 'vector': [-0.0, 1.0, 2**-149]}

    def write_package(self, rows):
        name = 'part.embeddings.jsonl.gz'
        path = self.source / name
        with gzip.open(path, 'wt', encoding='utf-8') as output:
            for row in rows:
                output.write(json.dumps(row) + '\n')
        manifest = {
            'formatVersion': 1, 'type': 'embedding-vectors', 'status': 'COMPLETE',
            'format': 'jsonl', 'compression': 'gzip',
            'source': {'packageId': 'emb-inputs-test', 'manifestSha256': 'sha256:' + 'a' * 64},
            'embeddings': [{'id': 'recipe', 'model': 'gemma', 'revision': 'rev',
                            'dimensions': 3, 'outputDtype': 'float32', 'normalization': 'l2'}],
            'counts': {'records': len(rows)},
            'parts': [{'file': name, 'bytes': path.stat().st_size, 'sha256': digest(path),
                       'sourceFile': 'part.jsonl.gz', 'sourceSha256': 'sha256:' + 'b' * 64,
                       'records': len(rows)}],
        }
        (self.source / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        return path

    def test_lookup_and_binary_round_trip(self):
        self.write_package([self.row, dict(self.row, network='UY', vector=[1.0, 0.0, 0.0])])
        manifest = publish_vectors(self.source, self.destination)
        database = self.destination / 'embeddings.sqlite'
        self.assertEqual(manifest['files'][0]['sha256'], digest(database))
        self.assertEqual(manifest['source']['vectorManifestSha256'], digest(self.source / 'manifest.json'))
        result = lookup_vector(database, network='AR', catalog_id='item')
        self.assertEqual(result['sourceRef'], {'file': 'part.embeddings.jsonl.gz', 'line': 1})
        expected = struct.pack('<3f', *self.row['vector'])
        self.assertEqual(expected, struct.pack('<3f', *result['vector']))
        with sqlite3.connect(database) as db:
            self.assertEqual(db.execute('SELECT vector FROM vectors WHERE item=1').fetchone(), (expected,))
        with self.assertRaises(FileExistsError):
            publish_vectors(self.source, self.destination)

    def test_invalid_vectors_leave_no_published_edition(self):
        cases = [
            ([self.row, self.row], 'UNIQUE'),
            ([dict(self.row, vector=[1.0])], 'dimension'),
            ([dict(self.row, vector=[True, 0.0, 0.0])], 'finite numbers'),
            ([dict(self.row, vector=[float('inf'), 0.0, 0.0])], 'finite numbers'),
        ]
        for rows, reason in cases:
            with self.subTest(reason=reason):
                self.write_package(rows)
                with self.assertRaisesRegex(ValueError, reason):
                    publish_vectors(self.source, self.destination)
                self.assertFalse(self.destination.exists())
                self.assertFalse((self.root / '.edition.lock').exists())

    def test_modified_gzip_part_is_rejected(self):
        path = self.write_package([self.row])
        raw = path.read_bytes()
        path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        with self.assertRaisesRegex(ValueError, 'checksum'):
            publish_vectors(self.source, self.destination)
        self.assertFalse(self.destination.exists())
