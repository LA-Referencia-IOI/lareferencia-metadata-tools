"""Publish standalone SQLite editions from batch-calculated embedding packages."""
import gzip
from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sqlite3
import struct
import tempfile

from .store import canonical, digest, readonly, string

APPLICATION_ID = 0x4C524556  # LREV
VERSION = 1
DATABASE = 'embeddings.sqlite'
NO_LABELS_FIELD = object()


def reference_labels(value, mode):
    if value is None and mode == 'predict':
        return None
    if (not isinstance(value, list) or not value
            or any(not isinstance(label, str) or not label.strip() for label in value)
            or len(set(value)) != len(value)):
        raise ValueError('Expected distinct reference labels or null in predict mode')
    return value


def count(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f'Invalid {label}: expected a nonnegative integer')
    return value


def checksum(entry, key):
    value = string(entry, key)
    if not re.fullmatch(r'sha256:[a-f0-9]{64}', value):
        raise ValueError(f'Invalid SHA-256 for {key}')
    return value


def load_source(directory):
    raw = (directory / 'manifest.json').read_bytes()
    source = json.loads(raw)
    if not isinstance(source, dict) or (
        type(source.get('formatVersion')) is not int
        or source['formatVersion'] != 1
        or (source.get('type'), source.get('status'), source.get('format'), source.get('compression'))
        != ('embedding-vectors', 'COMPLETE', 'jsonl', 'gzip')
    ):
        raise ValueError('Expected a COMPLETE embedding-vectors v1 JSONL gzip package')
    string(source['source'], 'packageId')
    checksum(source['source'], 'manifestSha256')
    profiles = source['embeddings']
    if not isinstance(profiles, list) or len(profiles) != 1:
        raise ValueError('Expected exactly one embedding profile')
    profile = profiles[0]
    for key in ('id', 'model', 'revision'):
        string(profile, key)
    dimensions = count(profile['dimensions'], 'embedding dimension')
    if not dimensions or profile['outputDtype'] != 'float32':
        raise ValueError('Expected a positive dimension and float32 vectors')
    if not isinstance(source['parts'], list):
        raise ValueError('Expected a list of vector parts')
    return source, 'sha256:' + sha256(raw).hexdigest()


def source_parts(directory, source):
    seen_files = set()
    seen_inputs = set()
    for part in source['parts']:
        name = string(part, 'file')
        input_name = string(part, 'sourceFile')
        checksum(part, 'sourceSha256')
        relative = Path(name)
        path = (directory / relative).resolve(strict=True)
        if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(directory):
            raise ValueError(f'Part outside the package: {name}')
        if path in seen_files or input_name in seen_inputs:
            raise ValueError(f'Duplicate part: {name}')
        seen_files.add(path)
        seen_inputs.add(input_name)
        records = count(part['records'], f'record count of {name}')
        yield part, path, records


class VectorDatabase:
    """Write and seal one SQLite edition in a staging directory."""

    def __init__(self, db, dimensions, *, classification=None):
        self.db = db
        self.dimensions = dimensions
        self.classification = classification
        self.codec = struct.Struct(f'<{dimensions}f')
        db.executescript(f'''
        PRAGMA page_size=16384;
        PRAGMA application_id={APPLICATION_ID};
        PRAGMA user_version={VERSION};
        PRAGMA foreign_keys=ON;
        CREATE TABLE package (
            singleton INTEGER PRIMARY KEY CHECK(singleton = 1), metadata TEXT NOT NULL
        );
        CREATE TABLE parts (
            id INTEGER PRIMARY KEY, metadata TEXT NOT NULL
        );
        CREATE TABLE items (
            id INTEGER PRIMARY KEY, network TEXT NOT NULL, catalog_id TEXT NOT NULL,
            part INTEGER NOT NULL REFERENCES parts(id), line INTEGER NOT NULL CHECK(line > 0),
            UNIQUE(network, catalog_id)
        );
        CREATE TABLE vectors (
            item INTEGER PRIMARY KEY REFERENCES items(id),
            vector BLOB NOT NULL CHECK(typeof(vector) = 'blob' AND length(vector) = {self.codec.size})
        );
        ''')
        if classification is not None:
            db.execute('CREATE TABLE labels (item INTEGER PRIMARY KEY REFERENCES items(id), labels_json TEXT)')

    def start_part(self, metadata):
        return self.db.execute('INSERT INTO parts(metadata) VALUES (?)', (canonical(metadata),)).lastrowid

    def finish_part(self, part, metadata):
        self.db.execute('UPDATE parts SET metadata = ? WHERE id = ?', (canonical(metadata), part))

    def write_vector(self, part, line, identity, vector, *, labels=NO_LABELS_FIELD):
        if self.classification is not None:
            labels = reference_labels(labels, self.classification['mode'])
        if len(vector) != self.dimensions:
            raise ValueError('Vector dimension mismatch')
        if any(type(value) not in (float, int) or not math.isfinite(value) for value in vector):
            raise ValueError('Vector components must be finite numbers')
        try:
            payload = self.codec.pack(*vector)
        except (OverflowError, struct.error) as error:
            raise ValueError('Vector component cannot be stored as float32') from error
        item = self.db.execute(
            'INSERT INTO items(network,catalog_id,part,line) VALUES (?,?,?,?)',
            (*identity, part, line),
        ).lastrowid
        self.db.execute('INSERT INTO vectors VALUES (?,?)', (item, payload))
        if self.classification is not None:
            payload = None if labels is None else canonical(labels)
            self.db.execute('INSERT INTO labels VALUES (?,?)', (item, payload))

    def seal(self, manifest):
        with self.db:
            self.db.execute('INSERT INTO package VALUES (1,?)', (canonical(manifest),))
        if self.db.execute('PRAGMA integrity_check').fetchall() != [('ok',)] or self.db.execute(
            'PRAGMA foreign_key_check'
        ).fetchone():
            raise ValueError('SQLite integrity check failed')


def populate(writer, directory, source):
    expected = count(source['counts']['records'], 'total record count')
    total = 0
    for part, path, records in source_parts(directory, source):
        if path.stat().st_size != count(part['bytes'], f'byte size of {part["file"]}') or (
            digest(path) != checksum(part, 'sha256')
        ):
            raise ValueError(f'Part checksum or size mismatch: {part["file"]}')
        with writer.db, gzip.open(path, 'rt', encoding='utf-8') as stream:
            part_id = writer.start_part(part)
            read = 0
            for line, text in enumerate(stream, 1):
                try:
                    row = json.loads(text)
                    network, catalog = string(row, 'network'), string(row, 'catalogId')
                    vector = row['vector']
                    if not isinstance(vector, list):
                        raise ValueError('Vector dimension mismatch')
                    labels = row['labels'] if writer.classification is not None else NO_LABELS_FIELD
                    writer.write_vector(part_id, line, (network, catalog), vector, labels=labels)
                    read += 1
                except (ValueError, KeyError, TypeError, OverflowError, struct.error, sqlite3.IntegrityError) as error:
                    raise ValueError(f'{part["file"]}:{line}: {error}') from error
            if read != records:
                raise ValueError(f'Part record count mismatch: {part["file"]}')
        total += read
    if total != expected:
        raise ValueError(f'Total record count mismatch: expected {expected}, read {total}')
    return total


def publish_vectors(source, destination):
    """Validate the JSONL gzip package and publish an independent, lossless SQLite edition."""
    directory = Path(source).resolve(strict=True)
    destination = Path(destination)
    original, original_hash = load_source(directory)
    profile = original['embeddings'][0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_name(f'.{destination.name}.lock')
    lock.mkdir()
    try:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        with tempfile.TemporaryDirectory(prefix=f'.{destination.name}.', dir=destination.parent) as temporary:
            edition = Path(temporary) / 'edition'
            edition.mkdir()
            database = edition / DATABASE
            with closing(sqlite3.connect(database)) as db:
                writer = VectorDatabase(db, profile['dimensions'], classification=original.get('classification'))
                total = populate(writer, directory, original)
                manifest = {
                    'type': 'embedding-vectors', 'formatVersion': VERSION, 'format': 'sqlite',
                    'status': 'COMPLETE', 'createdAt': datetime.now(timezone.utc).isoformat(),
                    'source': {'vectorManifestSha256': original_hash, 'input': original['source']},
                    'embeddings': [profile], 'encoding': 'float32le',
                    'counts': {**original['counts'], 'records': total},
                    'parts': original['parts'],
                }
                for key in ('runtime', 'truncatedInputs', 'classification'):
                    if key in original:
                        manifest[key] = original[key]
                writer.seal(manifest)
            manifest['files'] = [{'path': DATABASE, 'bytes': database.stat().st_size, 'sha256': digest(database)}]
            (edition / 'manifest.json').write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
            )
            edition.rename(destination)
            return manifest
    finally:
        lock.rmdir()


def lookup_vector(database, *, network, catalog_id):
    """Read a vector and its provenance without modifying the published database."""
    with closing(readonly(database)) as db:
        if db.execute('PRAGMA application_id').fetchone() != (APPLICATION_ID,) or db.execute(
            'PRAGMA user_version'
        ).fetchone() != (VERSION,):
            raise ValueError('Expected an embeddings SQLite v1 database')
        (metadata,) = db.execute('SELECT metadata FROM package WHERE singleton=1').fetchone()
        package = json.loads(metadata)
        row = db.execute('''
            SELECT v.vector, p.metadata, i.line, i.id
            FROM items i JOIN vectors v ON v.item=i.id JOIN parts p ON p.id=i.part
            WHERE i.network=? AND i.catalog_id=?
        ''', (network, catalog_id)).fetchone()
        if row is None:
            raise LookupError(f'Unknown embedding: {network}/{catalog_id}')
        payload, part_json, line, item = row
        (profile,) = package['embeddings']
        part = json.loads(part_json)
        result = {
            'network': network, 'catalogId': catalog_id, 'embedding': profile,
            'source': package['source'],
            'sourceRef': {'file': part.get('file', part['sourceFile']), 'line': line},
            'vector': list(struct.unpack(f'<{profile["dimensions"]}f', payload)),
        }
        if 'classification' in package:
            (labels,) = db.execute('SELECT labels_json FROM labels WHERE item=?', (item,)).fetchone()
            result['labels'] = None if labels is None else json.loads(labels)
        return result
