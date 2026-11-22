# LA Referencia Metadata Tools

Imported from lareferencia-shell at 7a80682bccbd9ed092d13679777f901150bee1cb and lareferencia-embedding-service at eb251cca4a3b694bb802261dbc496f1448bec6a5.

The package version follows the LA Referencia platform release. Python uses
PEP 440, so platform version `5.0.0-rc4` is wheel version `5.0.0rc4`.

## Installation

Use the module revision pinned in the platform's `workspace.lock`. In a Git
checkout, `githelper init` prepares the workspace. The platform release ZIP
already contains the sources. Use the same revision on the harvester, the
metadata processing host and the inference host.

From the root of this module, create and activate a virtual environment with
Python 3.12 or later.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install .
```

Use the base install for rights and SQLite. For bibliographic preparation,
install `python -m pip install '.[bibliographic]'`. For batch inference,
install `python -m pip install '.[inference]'` on the inference host. Run the
commands from the activated environment or use their paths under `.venv/bin/`.

Install the appropriate PyTorch build for the target CPU or CUDA environment.
Provision fastText and Gemma models separately. See the
[bibliographic classifier guide](docs/BIBLIOGRAPHIC_CLASSIFIER.md) and
[embedding metadata guide](docs/EMBEDDING_METADATA.md).

## Commands

- Export metadata with the Java shell and its maintained profiles. Pass the
  complete `exp-…` directory, including its manifest, mapping, data parts and
  index, to the Python commands.
- Use `count-export-licenses` to produce an inventory of the original values
  and `normalize-rights` to normalize it.
  [License inventory guide](docs/LICENSE_METADATA_EXTRACTION.md).
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
