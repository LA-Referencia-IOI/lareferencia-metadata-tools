# LA Referencia Metadata Tools

Imported from lareferencia-shell at 7a80682bccbd9ed092d13679777f901150bee1cb and lareferencia-embedding-service at eb251cca4a3b694bb802261dbc496f1448bec6a5.

## Installation

Create a virtual environment with Python 3.12 for the complete workflow. The
rights and SQLite commands also support Python 3.9 and use the standard library.
Install the extras required on each machine.

```sh
python -m pip install .
python -m pip install '.[bibliographic]'
python -m pip install '.[inference]'
```

Install the appropriate PyTorch build for the target CPU or CUDA environment.
Provision fastText and Gemma models separately. See the
[bibliographic classifier guide](docs/BIBLIOGRAPHIC_CLASSIFIER.md) and
[embedding metadata guide](docs/EMBEDDING_METADATA.md).

## Commands

- Export metadata with the Java shell and its maintained profiles.
- Run `run-rights-pipeline --network AR --export-dir /path/to/export` to
  publish a rights edition. To export a fresh LGK, supply `--shell-dir` or
  `LR_SHELL_DIR`. From a platform checkout the sibling shell is discovered.
  See the [enrichment package guide](docs/METADATA_ENRICHMENT_PACKAGE.md).
- Run `prepare-embedding-metadata --combine` to produce a portable text package.
- Run `compute-metadata-embeddings INPUT OUTPUT --revision REVISION` locally
  on CPU or GPU. The command accepts prepared text packages and the historical
  Java JSONL export mode.
- Run `enrichments pack-vectors VECTORS EDITION` to publish `embeddings.sqlite`.
  Use `enrichments lookup-vector` to resolve a vector by network and catalog ID.

Run the tests with `python -m unittest discover -s tests` in an environment
with the bibliographic dependencies.
