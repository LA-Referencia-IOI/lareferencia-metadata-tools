import argparse
import gzip
import json
from pathlib import Path
import sqlite3
import struct
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from enrichment_package import lookup_vector, publish_vectors
from enrichment_package.store import digest
from lareferencia_metadata_tools import compute_metadata_embeddings as compute


class PreparedInputTest(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.root = Path(work.name) / "prepared"
        self.output = self.root.with_name("vectors")
        self.root.mkdir()
        self.part = "part-000001.jsonl.gz"
        self.inputs = [
            {"network": "UY", "catalogId": "1", "text": "title: Un título | text: Resumen\n\ntema"},
            {"network": "AR", "catalogId": "1", "text": "title: Title | text: Mixed\r\nFunding note"},
            {"network": "UY", "catalogId": "2", "text": "title: Other | text: Other"},
        ]
        with gzip.open(self.root / self.part, "wt", encoding="utf-8") as out:
            for row in self.inputs:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        with gzip.open(self.root / "provenance.jsonl.gz", "wt", encoding="utf-8") as audit:
            audit.write('{"network":"UY","catalogId":"1","identifier":"oai:sample:1"}\n')
        self.manifest = {
            "packageType": "embedding-inputs", "contractVersion": 1, "status": "COMPLETE",
            "packageId": "test", "recipe": "combined-v1", "parts": [self.part],
            "counts": {"emitted": len(self.inputs)},
            "files": [{"path": name, **compute.file_entry(self.root / name)}
                      for name in (self.part, "provenance.jsonl.gz")],
        }
        self.args = argparse.Namespace(source=self.root, output=self.output, revision="f" * 40,
                                       on_overflow=None, recipe_id=None, chunk_size=2,
                                       dimensions=768, max_tokens=32, batch_size=2, dtype="float32", device="cpu",
                                       format="jsonl")
        self.manifest_file = self.root / "manifest.json"

    def test_sends_literal_text_and_keeps_vector_order(self):
        self.manifest_file.write_text(json.dumps(self.manifest))
        encoder = Mock(versions={})
        # Record the submitted strings before the writer clears its working batch.
        submitted = []

        def encode(texts):
            submitted.extend(texts)
            vectors = [Mock(tolist=lambda: [1.0, 0.0]) for _ in texts]
            return vectors, [(0, 40, 32)] if len(texts) == 2 else []

        encoder.encode.side_effect = encode
        with patch.object(compute, "Encoder", return_value=encoder):
            compute.build(self.args)
        self.assertEqual(submitted, [row["text"] for row in self.inputs])
        result = json.loads((self.output / "manifest.json").read_bytes())
        self.assertEqual(result["type"], "embedding-vectors")
        self.assertEqual(result["embeddings"][0]["onOverflow"], "truncate")
        self.assertEqual(result["truncatedInputs"], [
            {"sourceFile": self.part, "sourceLine": 1, "sourceTokens": 40, "encodedTokens": 32},
        ])
        with gzip.open(self.output / result["parts"][0]["file"], "rt", encoding="utf-8") as source:
            vectors = [json.loads(line) for line in source]
        self.assertEqual(vectors, [{"network": row["network"], "catalogId": row["catalogId"], "vector": [1.0, 0.0]}
                                   for row in self.inputs])

    def test_reduced_dimensions_are_recorded_and_suffix_recipe(self):
        self.args.dimensions = 256
        self.manifest_file.write_text(json.dumps(self.manifest))
        encoder = Mock(versions={})
        encoder.encode.side_effect = lambda texts: (
            [Mock(tolist=lambda: [1.0] + [0.0] * 255) for _ in texts], []
        )
        with patch.object(compute, "Encoder", return_value=encoder):
            compute.build(self.args)
        result = json.loads((self.output / "manifest.json").read_bytes())
        profile = result["embeddings"][0]
        self.assertEqual(profile["id"], "combined-v1-256d")
        self.assertEqual(profile["dimensions"], 256)
        self.assertEqual(profile["nativeDimensions"], 768)
        self.assertEqual(profile["dimensionReduction"], "matryoshka-prefix")

    def test_invalid_data_checksum_prevents_publication(self):
        self.manifest_file.write_text(json.dumps(self.manifest))
        (self.root / self.part).write_bytes(b"invalid")
        with patch.object(compute, "Encoder"):
            with self.assertRaisesRegex(ValueError, self.part):
                compute.build(self.args)
        self.assertFalse(self.output.exists())

    def test_partial_package_is_rejected_before_loading_model(self):
        self.manifest["status"] = "PARTIAL"
        self.manifest_file.write_text(json.dumps(self.manifest))
        with patch.object(compute, "Encoder") as encoder:
            with self.assertRaisesRegex(ValueError, "COMPLETE"):
                compute.build(self.args)
        encoder.assert_not_called()

    def test_cli_selects_sqlite_and_preserves_vector_bytes(self):
        self.manifest_file.write_text(json.dumps(self.manifest))
        vector = [-0.0, 1.0, 2**-149] + [0.0] * 509
        encoder = Mock(versions={})
        encoder.encode.side_effect = lambda texts: ([Mock(tolist=lambda: vector) for _ in texts], [(0, 40, 32)])
        options = ["--revision", "f" * 40, "--dimensions", "512", "--chunk-size", "2", "--max-tokens", "32"]
        direct = self.output.with_name("direct")
        with patch.object(compute, "Encoder", return_value=encoder):
            with patch.object(sys, "argv", ["compute-metadata-embeddings", str(self.root), str(self.output), *options]):
                self.assertEqual(compute.main(), 0)
            with patch.object(sys, "argv", ["compute-metadata-embeddings", str(self.root), str(direct),
                                            *options, "--format", "sqlite"]):
                self.assertEqual(compute.main(), 0)
        packed = direct.with_name("packed")
        publish_vectors(self.output, packed)
        self.assertEqual({path.name for path in direct.iterdir()}, {"embeddings.sqlite", "manifest.json"})
        database = direct / "embeddings.sqlite"
        manifest = json.loads((direct / "manifest.json").read_bytes())
        self.assertEqual(manifest["format"], "sqlite")
        self.assertEqual(manifest["source"], {"input": {"packageId": "test",
                                                       "manifestSha256": digest(self.manifest_file)}})
        self.assertEqual(manifest["parts"], [{"sourceFile": self.part, "sourceSha256": digest(self.root / self.part),
                                             "sourceBytes": (self.root / self.part).stat().st_size, "records": 3}])
        self.assertEqual(manifest["files"][0]["sha256"], digest(database))
        self.assertEqual([item["sourceLine"] for item in manifest["truncatedInputs"]], [1, 3])
        for line, row in enumerate(self.inputs, 1):
            query = {"network": row["network"], "catalog_id": row["catalogId"]}
            result = lookup_vector(database, **query)
            converted = lookup_vector(packed / "embeddings.sqlite", **query)
            self.assertEqual(result["sourceRef"], {"file": self.part, "line": line})
            self.assertEqual(struct.pack("<512f", *result["vector"]), struct.pack("<512f", *converted["vector"]))
            self.assertEqual(struct.pack("<512f", *result["vector"]), struct.pack("<512f", *vector))

    def test_sqlite_rejects_duplicate_identities_without_publishing(self):
        self.args.format = "sqlite"
        self.inputs[-1] = self.inputs[0]
        part = self.root / self.part
        with gzip.open(part, "wt", encoding="utf-8") as out:
            for row in self.inputs:
                out.write(json.dumps(row) + "\n")
        self.manifest["files"][0].update(compute.file_entry(part))
        self.manifest_file.write_text(json.dumps(self.manifest))
        encoder = Mock(versions={})
        encoder.encode.side_effect = lambda texts: ([Mock(tolist=lambda: [1.0] + [0.0] * 767) for _ in texts], [])
        with patch.object(compute, "Encoder", return_value=encoder):
            with self.assertRaisesRegex(ValueError, r"part-000001.jsonl.gz:3: UNIQUE"):
                compute.build(self.args)
        self.assertEqual(set(self.root.parent.iterdir()), {self.root})

    def test_classification_labels_survive_jsonl_and_both_sqlite_paths(self):
        labels = [["5.4", "2.11"], None, ["6.1"]]
        self.manifest.update(
            packageType="classification-dataset", mode="predict", labelProfile="ford-level2",
            recipe={"id": compute.RECIPE, "textFormat": "gemma-classification", "includeFallback": False},
        )
        submitted = []
        part = self.root / self.part
        with gzip.open(part, "wt", encoding="utf-8") as out:
            for row, reference in zip(self.inputs, labels, strict=True):
                row["text"] = compute.PREFIX["gemma-classification"] + row["catalogId"]
                out.write(json.dumps({**row, "labels": reference}) + "\n")
        self.manifest["files"][0].update(compute.file_entry(part))
        self.manifest_file.write_text(json.dumps(self.manifest))

        def encode(texts):
            submitted.extend(texts)
            return [Mock(tolist=lambda: [1.0] + [0.0] * 255) for _ in texts], []

        direct = self.output.with_name("direct")
        self.args.dimensions = 256
        encoder = Mock(versions={})
        encoder.encode.side_effect = encode
        with patch.object(compute, "Encoder", return_value=encoder):
            compute.build(self.args)
            self.args.output = direct
            self.args.format = "sqlite"
            compute.build(self.args)

        self.assertEqual(submitted, [row["text"] for row in self.inputs] * 2)
        packed = self.output.with_name("packed")
        converted = publish_vectors(self.output, packed)
        manifest = json.loads((self.output / "manifest.json").read_bytes())
        self.assertEqual(converted["classification"], {"mode": "predict", "labelProfile": "ford-level2"})
        with gzip.open(self.output / manifest["parts"][0]["file"], "rt", encoding="utf-8") as stream:
            for expected, line in zip(labels, stream, strict=True):
                row = json.loads(line)
                self.assertEqual(row["labels"], expected)
                for edition in (direct, packed):
                    found = lookup_vector(edition / "embeddings.sqlite", network=row["network"],
                                          catalog_id=row["catalogId"])
                    self.assertEqual(found["labels"], expected)
                    self.assertEqual(found["vector"], row["vector"])

        self.manifest["mode"] = "train"
        self.manifest_file.write_text(json.dumps(self.manifest))
        self.args.output = self.output.with_name("invalid-train")
        with patch.object(compute, "Encoder", return_value=encoder):
            with self.assertRaisesRegex(ValueError, "reference labels"):
                compute.build(self.args)
        self.assertFalse(self.args.output.exists())

    def test_sqlite_accepts_legacy_export_with_verified_record_identity(self):
        self.args.format = "sqlite"
        self.args.title_field = "titles"
        self.args.abstract_field = "descriptions"
        self.args.abstract_qualifier = "abstract"
        row = {"network": "AR", "catalogId": "1", "identifier": "oai:1", "metadataHash": "abc",
               "data": {"titles": ["First"], "descriptions": [{"value": "Abstract", "qualifier": "abstract"}]}}
        part = self.root / self.part
        with gzip.open(part, "wt", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        index = self.root / "records.sqlite"
        with sqlite3.connect(index) as db:
            db.execute("CREATE TABLE records(export_id, variant, status, file, position, network, catalog_id, identifier, metadata_hash)")
            db.execute("INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       ("exp-test", "original", "EXPORTED", self.part, 1, "AR", "1", "oai:1", "abc"))
        self.manifest_file.write_text(json.dumps({
            "status": "COMPLETE", "exportId": "exp-test", "options": {"format": "jsonl", "variant": "original"},
            "counts": {"exported": 1},
            "files": [{"path": path.name, **compute.file_entry(path)} for path in (part, index)],
        }))
        encoder = Mock(versions={})
        encoder.encode.return_value = ([Mock(tolist=lambda: [1.0] + [0.0] * 767)], [])
        with patch.object(compute, "Encoder", return_value=encoder):
            compute.build(self.args)
        encoder.encode.assert_called_once_with(["title: First | text: Abstract"])
        result = lookup_vector(self.output / "embeddings.sqlite", network="AR", catalog_id="1")
        self.assertEqual(result["source"]["input"]["exportId"], "exp-test")
        self.assertEqual(result["sourceRef"], {"file": self.part, "line": 1})
        self.args.format = "jsonl"
        self.args.output = self.output.with_name("legacy-jsonl")
        with patch.object(compute, "Encoder", return_value=encoder):
            compute.build(self.args)
        manifest = json.loads((self.args.output / "manifest.json").read_bytes())
        self.assertEqual(manifest["type"], "metadata-embeddings")
        self.assertEqual(result["source"]["input"], manifest["source"])
        with gzip.open(self.args.output / manifest["parts"][0]["file"], "rt", encoding="utf-8") as stream:
            self.assertEqual(json.load(stream), [result["vector"]])

    def test_raw_export_keeps_legacy_text_and_vector_shape(self):
        row = {"network": "UY", "catalogId": "1", "identifier": "oai:1", "metadataHash": "abc",
               "data": {"titles": [{"value": "First"}, {"value": "Second"}],
                        "descriptions": [{"value": "Description"}, {"value": "Abstract", "qualifier": "abstract"}]}}
        part = self.root / "legacy.jsonl.gz"
        with gzip.open(part, "wt", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        index = iter([(part.name, 1, "UY", "1", "oai:1", "abc")])
        destination = self.root / "legacy.embeddings.jsonl.gz"
        self.args.title_field = "titles"
        self.args.abstract_field = "descriptions"
        self.args.abstract_qualifier = "abstract"
        encoder = Mock()
        encoder.encode.return_value = ([Mock(tolist=lambda: [1.0])], [])
        self.assertEqual(compute.write_part(part, part.name, destination, index, encoder, self.args, []), (1, 1))
        encoder.encode.assert_called_once_with(["title: First | text: Abstract"])
        with gzip.open(destination, "rt", encoding="utf-8") as stream:
            self.assertEqual(json.load(stream), [[1.0]])
