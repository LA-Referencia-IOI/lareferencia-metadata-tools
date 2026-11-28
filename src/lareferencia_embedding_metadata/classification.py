"""Prepare one classification example per bibliographic record."""
import argparse
from collections import defaultdict
from contextlib import closing
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from uuid import uuid4

from enrichment_package.store import digest, string
from .annotation_profiles import FORD_LEVEL2


PREFIX = {"plain": "", "gemma-classification": "task: classification | query: "}
LABEL_PROFILES = ("ford-level1", "ford-level2", "ods")
FORD_AREAS = frozenset(str(i) for i in range(1, 7))
ODS_GOALS = frozenset(str(i) for i in range(1, 18))
RECIPE = "titles-abstracts-by-language-v1"


def classification_labels(terms, profile):
    if profile not in LABEL_PROFILES:
        raise ValueError(f"Unknown classification label profile: {profile}")
    labels = {}
    for term in terms:
        scheme = term["scheme"]
        if profile == "ods":
            if scheme != "ods":
                continue
            code = string(term, "code")
            if code not in ODS_GOALS:
                raise ValueError(f"Invalid ODS goal: {code}")
        else:
            if scheme != "becyt-ford":
                continue
            code = string(term, "code")
            specific = FORD_LEVEL2.fullmatch("https://purl.org/becyt/ford/" + code)
            if not specific and not (profile == "ford-level1" and code in FORD_AREAS):
                raise ValueError(f"Invalid FORD code for {profile}: {code}")
            if profile == "ford-level1":
                code = code.partition(".")[0]
        labels[code] = None
    return list(labels) or None


def example(row, text_format, include_fallback, label_profile="ford-level2"):
    data = row["data"]
    groups = defaultdict(list)
    for field in ("titles", "abstracts"):
        for item in data[field]:
            if field == "abstracts" and not include_fallback and item["selection"] == "uncertain_fallback":
                continue
            value = item["value"].strip()
            if value:
                groups[string(item, "language")].append(value)
    labels = classification_labels(data.get("subjectTerms", []), label_profile)
    if not groups:
        return None
    content = "\n\n".join(value for values in groups.values() for value in values)
    return {"network": string(row, "network"), "catalogId": string(row, "catalogId"),
            "languages": list(groups), "text": PREFIX[text_format] + content, "labels": labels}


def verified_file(root, name, inventory):
    path = (root / name).resolve(strict=True)
    if Path(name).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"File outside package: {name}")
    entry = inventory[name]
    if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
        raise ValueError(f"Invalid inventory entry: {name}")
    return path


def prepare(root, output, mode, text_format, include_fallback=False, label_profile="ford-level2"):
    root = root.resolve()
    if mode not in ("train", "predict") or text_format not in PREFIX or label_profile not in LABEL_PROFILES:
        raise ValueError("Unknown mode, text format or label profile")
    if output.resolve().is_relative_to(root):
        raise ValueError("Choose an output directory outside the input package")
    raw = (root / "manifest.json").read_bytes()
    parent = json.loads(raw)
    if (parent["packageType"], parent["contractVersion"], parent["status"]) != ("annotated-metadata", 2, "COMPLETE"):
        raise ValueError("Expected a COMPLETE annotated-metadata v2 package")
    inventory = {entry["path"]: entry for entry in parent["files"]}
    index = verified_file(root, "records.sqlite", inventory)
    parts = parent["parts"]
    if len(set(parts)) != len(parts):
        raise ValueError("Duplicate annotated part")
    output.mkdir(parents=True, exist_ok=True)
    final = output / ("cls-" + str(uuid4()))
    counts = dict(input=0, emitted=0, missingText=0, omittedUnlabeled=0, emittedUnlabeled=0)
    with tempfile.TemporaryDirectory(dir=output, prefix=".classification-") as work:
        dest = Path(work) / final.name
        dest.mkdir()
        part = dest / "records.jsonl.gz"
        with closing(sqlite3.connect(index.as_uri() + "?mode=ro&immutable=1", uri=True)) as db, gzip.open(
            part, "wt", encoding="utf-8", compresslevel=1
        ) as out:
            for name in parts:
                path = verified_file(root, name, inventory)
                cursor = iter(db.execute("SELECT position,network,catalog_id FROM records WHERE file=? "
                                         "AND status='ANNOTATED' ORDER BY position", (name,)))
                with gzip.open(path, "rt", encoding="utf-8") as lines:
                    for pos, line in enumerate(lines, 1):
                        row = json.loads(line)
                        if next(cursor, None) != (pos, row["network"], row["catalogId"]):
                            raise ValueError(f"Index mismatch at {name}:{pos}")
                        counts["input"] += 1
                        item = example(row, text_format, include_fallback, label_profile)
                        if item is None:
                            counts["missingText"] += 1
                        elif mode == "train" and item["labels"] is None:
                            counts["omittedUnlabeled"] += 1
                        else:
                            out.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
                            counts["emitted"] += 1
                            counts["emittedUnlabeled"] += item["labels"] is None
                if next(cursor, None) is not None:
                    raise ValueError(f"Missing rows in {name}")
            indexed = db.execute("SELECT count(*) FROM records WHERE status='ANNOTATED'").fetchone()[0]
            if counts["input"] != indexed or indexed != parent["counts"]["annotated"]:
                raise ValueError("Annotated counts do not reconcile")
        manifest = {"packageType": "classification-dataset", "contractVersion": 1, "packageId": final.name,
                    "status": "COMPLETE", "mode": mode, "parent": {"packageId": parent["packageId"],
                    "manifestHash": "sha256:" + hashlib.sha256(raw).hexdigest()},
                    "recipe": {"id": RECIPE, "textFormat": text_format, "includeFallback": include_fallback},
                    "labelProfile": label_profile, "counts": counts, "parts": [part.name],
                    "files": [{"path": part.name, "bytes": part.stat().st_size, "sha256": digest(part)}]}
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        dest.rename(final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("train", "predict"), required=True)
    parser.add_argument("--text-format", choices=PREFIX)
    parser.add_argument("--label-profile", choices=LABEL_PROFILES)
    parser.add_argument("--include-fallback", action="store_true")
    parser.add_argument("--recipe-from", type=Path, help="Reuse the recipe of a classification package directory")
    args = parser.parse_args()
    try:
        fmt, fallback = args.text_format or "plain", args.include_fallback
        profile = args.label_profile or "ford-level2"
        if args.recipe_from:
            if args.text_format or args.include_fallback:
                parser.error("--recipe-from replaces --text-format and --include-fallback")
            source = json.loads((args.recipe_from / "manifest.json").read_bytes())
            recipe = source["recipe"]
            if ((source["packageType"], source["contractVersion"], source["status"], recipe["id"]) !=
                    ("classification-dataset", 1, "COMPLETE", RECIPE) or type(recipe["includeFallback"]) is not bool):
                raise ValueError("Unsupported classification recipe")
            profile = args.label_profile or source["labelProfile"]
            fmt, fallback = recipe["textFormat"], recipe["includeFallback"]
        dest = prepare(args.input, args.output, args.mode, fmt, fallback, profile)
    except (ValueError, OSError, EOFError, KeyError, TypeError, sqlite3.Error) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(dest.resolve())
    return 0
