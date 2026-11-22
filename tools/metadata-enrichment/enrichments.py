#!/usr/bin/env python3
"""Publish and query metadata enrichment and vector packages."""
import argparse
import gzip
import json
from pathlib import Path
import sqlite3

from enrichment_package import lookup, lookup_vector, publish, publish_vectors, upsert_batch


def read_returns(path, run):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                required = {'exportId', 'network', 'catalogId', 'identifier', 'metadataHash', 'enrichment'}
                if not isinstance(row, dict) or set(row) != required:
                    raise ValueError('Invalid v2 envelope')
                entry = row['enrichment']
                if not isinstance(entry, dict) or set(entry) != {'id','type','schemaVersion','runId','generatedAt','values'}:
                    raise ValueError('Invalid v2 enrichment')
                if any(row[k] != run[k] for k in ('exportId', 'network')) or entry['id'] != run['producer']:
                    raise ValueError('Source or producer mismatch')
                if any(entry[k] != run[k] for k in ('type', 'schemaVersion', 'runId')):
                    raise ValueError('Run mismatch')
                yield {**row, 'fields': entry['values']['fields'], 'generatedAt': entry['generatedAt'],
                       'sourceRef': f'{path.name}:{number}'}
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f'{path}:{number}: {error}') from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    ingest = commands.add_parser('ingest')
    ingest.add_argument('database', type=Path)
    ingest.add_argument('jsonl', type=Path)
    ingest.add_argument('--run', type=Path, required=True)
    query = commands.add_parser('lookup')
    query.add_argument('database', type=Path)
    keys = query.add_mutually_exclusive_group(required=True)
    keys.add_argument('--catalog-id')
    keys.add_argument('--identifier')
    edition = commands.add_parser('publish')
    edition.add_argument('database', type=Path)
    edition.add_argument('destination', type=Path)
    vectors = commands.add_parser('pack-vectors', help='Convert a vector JSONL gzip package to SQLite')
    vectors.add_argument('source', type=Path)
    vectors.add_argument('destination', type=Path)
    vector = commands.add_parser('lookup-vector', help='Read one vector by network and catalog ID')
    vector.add_argument('database', type=Path)
    vector.add_argument('--network', required=True)
    vector.add_argument('--catalog-id', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'ingest':
            run = json.loads(args.run.read_text())
            result = {'evaluated': upsert_batch(args.database, run, read_returns(args.jsonl, run))}
        elif args.command == 'lookup':
            result = lookup(args.database, catalog_id=args.catalog_id, identifier=args.identifier)
        elif args.command == 'pack-vectors':
            result = publish_vectors(args.source, args.destination)
        elif args.command == 'lookup-vector':
            result = lookup_vector(args.database, network=args.network, catalog_id=args.catalog_id)
        else:
            result = publish(args.database, args.destination)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, EOFError, ValueError, LookupError, TypeError, sqlite3.Error) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
