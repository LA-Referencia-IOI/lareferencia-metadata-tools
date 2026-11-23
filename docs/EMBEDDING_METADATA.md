# Prepare metadata for embeddings

The pipeline has two preparation steps. The Java shell exports metadata from a snapshot. The Python command in this repository selects descriptions and builds the text for the batch embedding calculator. Use `--combine` for one vector per record with a title.

## Export and prepare

From the LA Referencia shell, export the original metadata for the required network. Add `--valid-only` when the campaign is limited to validated records.

```text
export-metadata --network UY --variant original --format jsonl --mapping config/exports/bibliographic-roles-v1.json --compression gzip --output /srv/metadata-exports
```

Install the Python command from this repository and provision the approved `lid.176.ftz` and `role.bin` files in the same models directory. The [pilot guide](BIBLIOGRAPHIC_CLASSIFIER.md) documents how to obtain the language model and train the role model. The preparer also accepts the `classification-embeddings` 2.0 profile.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install ".[bibliographic]"

.venv/bin/prepare-embedding-metadata \
  --input "$EXPORT_DIR" --models-dir "$MODELS_DIR" \
  --output /srv/metadata-exports/prepared --combine
```

Use the exact `exp-…` directory reported by Java as `EXPORT_DIR`. You can set `LR_EMBEDDING_MODELS` in place of `--models-dir`. The Python command validates the export inventory, checks each input against its index and writes a new package directory. It publishes by renaming a temporary directory. Exit code 0 means `COMPLETE`, 2 means `PARTIAL` with row errors in `errors.jsonl`, and 1 signals a fatal failure. Automated consumers must require `COMPLETE`.

## Combined text

fastText and the role rules evaluate every description. The preparer retains all accepted descriptions. When none is accepted, it picks the longest nonblank uncertain description after cleaning, breaking ties by source order. Each selected occurrence keeps its original literal, including any languages, rights statements or acknowledgments it contains.

The title and abstract blocks each put detected English occurrences first and preserve source order within the English and remaining groups. Two newlines separate occurrences. The body uses the title block when no description was selected. The preparer appends all nonblank keywords as the last paragraph of the body, in source order and separated by commas. It performs language detection only on titles and descriptions. The template is `title: {titles} | text: {body}`.

The `embedding-inputs` v1 manifest lists the recipe, completion status, ordered GZIP JSONL parts, file sizes, SHA-256 checksums and counts. The main data rows contain only `network`, `catalogId` and `text`. Counts reconcile input rows with emitted rows, omissions for missing titles and errors. The abstract counts describe the emitted records and distinguish direct classification, uncertain fallback and missing descriptions.

Add `--include-provenance` to produce `provenance.jsonl.gz`. The sidecar uses `(network, catalogId)` to join to the data rows. It includes the OAI identifier, metadata hash, snapshot reference, export part and 1-based line, and the 0-based occurrence indexes in text order. Description references record `classified` or `uncertain_fallback`. The manifest resolves snapshot references and identifies the parent export, profile and fastText model hashes. Keep the parent export to inspect original projected occurrences and the snapshot store to access historical XML.

## Calculate vectors

Install the `inference` extra on the machine that runs the encoder. Use the
same module revision for preparation and inference. Set `PREPARED_DIR` to the
complete `embedding-inputs` directory reported by the preparer. Set
`MODEL_REVISION` to the 40-character Hugging Face commit of
`google/embeddinggemma-2` you want to run.

```sh
python3.12 -m venv .venv-inference
. .venv-inference/bin/activate
python -m pip install '.[inference]'
```

Choose a new output directory and set `--format jsonl` or `--format sqlite`.
The default is `jsonl`. Both formats use the same text, model, dimensions and
normalization.

```sh
# Calculate and publish GZIP JSONL.
compute-metadata-embeddings "$PREPARED_DIR" "$VECTOR_DIR" \
  --revision "$MODEL_REVISION" --dimensions 512 --format jsonl

# Calculate and publish SQLite directly.
compute-metadata-embeddings "$PREPARED_DIR" "$SQLITE_EDITION_DIR" \
  --revision "$MODEL_REVISION" --dimensions 512 --format sqlite
```

| Option | Behavior |
| --- | --- |
| `--device`, `--dtype` | Default `cuda` and `bfloat16`. On CPU use `--device cpu --dtype float32`. Both choices produce `float32` vectors. |
| `--offline` | Load the pinned model revision from the local cache. |
| `--dimensions` | 768 by default. At 512 or 256, take the first components and L2-normalize the resulting vector in float32. The default recipe ID gets a dimension suffix for reduced vectors. |
| `--max-tokens`, `--on-overflow` | Limit of 8,192 tokens by default. Prepared texts default to `truncate`. Use `error` to reject oversize inputs. |
| `--batch-size`, `--chunk-size` | Inference batch cap of 16 and read buffer of 512 records by default. Long documents use smaller inference batches. |
| `--recipe-id` | Overrides the recipe identifier recorded in the manifest. |

The calculator checks the input status, file sizes, SHA-256 hashes and row
counts. It submits the prepared `text` as supplied, with its document template
already applied. Truncation affects the text sent to the model. The manifest
records the input part, its one-based line and the before/after token counts.
The directory is published by renaming a staging directory after successful
verification. The command exits with 0 on success and 1 on a processing error.

## GZIP JSONL output

With prepared input, the result is an `embedding-vectors` v1 package. Each
input part produces one corresponding `*.embeddings.jsonl.gz` file. GZIP
contains UTF-8 JSONL with one vector per row, in input order.

```json
{"network":"AR","catalogId":"0219761202bbe88b2cfa665aa5b29466","vector":[0.12,-0.04,0.09]}
```

The example shows three components. The actual length is
`embeddings[0].dimensions`. Read only the parts listed in the manifest, in
that order. Within each part, line numbers start at 1 and match the input
part. The key for joins is `(network, catalogId)`.

| Manifest field | Meaning |
| --- | --- |
| `type`, `formatVersion`, `format`, `compression`, `status` | `embedding-vectors`, `1`, `jsonl`, `gzip`, `COMPLETE`. |
| `source` | Input `packageId` and `manifestSha256`. |
| `embeddings[0]` | Recipe ID, model and revision, native and output dimensions, reduction, precision, normalization, token policy and input recipe. |
| `runtime`, `counts` | Device, runtime versions, batch cap and totals. |
| `parts[]` | Output `file`, `bytes`, `sha256` and `records`, with `sourceFile` and `sourceSha256` for the corresponding input part. |
| `truncatedInputs[]` | `sourceFile`, `sourceLine`, `sourceTokens` and `encodedTokens` when the policy is `truncate`. |

### Compatibility with an original Java export

The calculator also accepts a complete Java JSONL export and checks its
`records.sqlite`. It builds the text from the first nonblank title and the
nonblank descriptions under the configured abstract qualifier. On missing
abstracts it uses the title as the body. Select fields with `--title-field`,
`--abstract-field` and `--abstract-qualifier`. Pass `--abstract-qualifier ''`
to read every occurrence of the selected field. The overflow default for
this input is `error`.

With this input and `--format jsonl`, the calculator preserves the legacy
`metadata-embeddings` v1 format. Each line contains an array with one vector
(`[[0.12,-0.04,0.09]]`). Identity comes from the matching row in the Java
export. Keep both packages together for the PostgreSQL importer in semantic
service v2. If you select SQLite with this input, the calculator stores the
verified network and catalog ID in the vector database and references the
original export in its manifest.

## SQLite output and conversion

With `--format sqlite`, the calculator writes the vectors directly to the
database while it runs. The edition contains `embeddings.sqlite` and
`manifest.json`. To keep both formats from one inference run on prepared
texts, generate JSONL first and convert that package.

```sh
enrichments pack-vectors "$VECTOR_DIR" "$SQLITE_EDITION_DIR"
enrichments lookup-vector "$SQLITE_EDITION_DIR/embeddings.sqlite" \
  --network AR --catalog-id "$CATALOG_ID"
```

The converter accepts complete `embedding-vectors` v1 GZIP JSONL packages
and preserves the float32 values. With either route, keep the input package
and the original export for provenance. The [SQLite vector contract](METADATA_ENRICHMENT_PACKAGE.md#pack-calculated-embeddings-in-sqlite)
describes the storage schema and how to resolve references from direct and
converted editions.

## Annotated metadata and per-language texts

Omit `--combine` to produce an `annotated-metadata` v2 package. It includes a local SQLite index and arrays for titles, abstracts, raw keywords, affiliations, funders, document types and other descriptions. Titles and abstracts include their detected languages. Abstracts record their selection. Records without a title remain available for inspection. Add `--include-provenance` for the export locators and occurrence indexes.

### Annotation profiles and FORD subjects

`--annotation-profile auto|bibliographic|ford-level2` chooses the annotated
view independently of the export mapping. Its default is `auto`:

| Export mapping | `auto` resolves to | Source subject array |
| --- | --- | --- |
| `bibliographic-roles` 1.0 | `bibliographic` | `keywords` |
| `classification-embeddings` 2.0 | `bibliographic` | `keywords` |
| `subjects-ford` 1.0 | `ford-level2` | `subjects` |

Existing commands, classification thresholds, role rules, abstract fallback,
language detection, keyword preservation and combined recipes retain their
behavior for the two previous mappings. The manifest now includes the additive
`annotationProfile` object with `name`, `version` (`1.0`) and `requested`.
Input mappings still must pass the inventory, version and identity checks.

```sh
prepare-embedding-metadata --input "$FORD_EXPORT_DIR" \
  --output "$OUTPUT_DIR" --models-dir "$MODELS_DIR" \
  --annotation-profile ford-level2 --include-provenance
```

The `ford-level2` view keeps all the existing annotated families and adds:

- `subjects`: the complete original subject occurrences, unchanged and in
  source order, including free text, general FORD codes, language and qualifiers.
- `subjectTerms`: second-level BECyT FORD identifiers only, deduplicated by
  scheme/code in first-occurrence order. Each is an object such as
  `{"scheme":"becyt-ford","code":"5.4","uri":"https://purl.org/becyt/ford/5.4"}`.

Matching trims surrounding whitespace and requires the full literal
`http://purl.org/becyt/ford/<code>` or its HTTPS equivalent. The code must have
two numeric components, a first component from 1 to 6 and a positive second
component without leading zeros. This is structural selection, not validation
against a complete vocabulary release. `5`, `5.4.1`, trailing path segments,
queries, labels and unrelated URLs are excluded. Peruvian `pe-repo/ocde/ford`
identifiers are preserved in `subjects` but are not translated to BECyT codes.
HTTP and HTTPS versions of a code produce one canonical HTTPS term.

Records without second-level codes remain in the output with `subjectTerms: []`.
This view does not infer a child from a parent code or classify new disciplines.
For `subjects-ford`, original occurrences live in `subjects`; `keywords` is empty
unless the input also supplies that field. Explicit `ford-level2` also works with
the previous mappings, taking subjects from their `keywords` and preserving
those keywords unchanged. Explicit `bibliographic` on a `subjects-ford` export
instead copies all subjects to `keywords`, without adding the FORD arrays.

With `--include-provenance`, `origins.subjects[i]` references the original
`[field, index]`; `origins.subjectTerms[i]` is a list of **all** such references
that contributed to that deduplicated term. Indexes are zero-based. Manifest
counts `subjectOccurrences`, `subjectTerms` and `recordsWithSubjectTerms` count
original occurrences, retained distinct terms per record, and records having at
least one term, respectively; they are also reported per network.

`ford-level2` produces annotated metadata and rejects `--combine` to avoid
discarding the selected labels. For the historical combined recipe, select
`--annotation-profile bibliographic --combine` explicitly. The per-language
consumer below can read either annotated view; subjects and subjectTerms do not
enter its embedding texts. Keep the annotated package to join labels by
`(network, catalogId)`. Abstracts selected by uncertain fallback remain marked
`selection: "uncertain_fallback"`; they are candidates, not confirmed abstracts.

```sh
build-embedding-inputs "$ANNOTATED_DIR" > entries.jsonl
```

This example creates texts per language for records with a title. Unknown-language occurrences get individual entries. New annotated packages keep the keywords in metadata and omit them from these texts. The example also reads historical v1 packages and respects their recorded keyword assignments. Use `--classified-only` to exclude uncertain fallbacks from this per-language view.

## Verification with Uruguay

The reference export for UY snapshot 4186 contains 30,752 records with usable titles. The combined recipe produces 30,752 inputs and preserves the 2,040 uncertain description selections from the historical annotation. The historical per-language pilot produced 53,737 inputs.

Run the preparer tests from this repository after installing the package.

```sh
.venv/bin/python -m unittest discover -s tests
```

## Prepare a classification dataset

Run `prepare-classification-dataset --input "$ANNOTATED_DIR" --output "$OUTPUT_DIR" --mode train --text-format gemma-classification`.
It reads a complete v2 annotated package and writes a new `cls-…` directory with a manifest and UTF-8 `records.jsonl.gz`.
Each row contains `network`, `catalogId`, ordered `languages`, one `text`, and `labels` (distinct second-level BECyT codes from `subjectTerms`).
Join the row to the annotated package by network and catalog ID. Its manifest is identified by package ID and SHA-256.

The v1 recipe reads titles then abstracts, trims surrounding whitespace, skips blank values and groups occurrences by language in first-seen order. Paragraphs are separated by two newlines. By default it skips uncertain fallback abstracts. Use `--include-fallback` to include them. Rows with no usable text are counted and omitted in both modes.

The default `plain` format writes the content directly. `gemma-classification` prepends the literal `task: classification | query: ` once. Encode preformatted text with an empty additional prompt. FORD labels stay outside the text.

For prediction, use `prepare-classification-dataset --input "$ANNOTATED_DIR" --output "$OUTPUT_DIR" --mode predict --recipe-from "$TRAIN_PACKAGE"`. Prediction keeps the available reference labels and writes JSON null when they are unknown. Training omits rows without labels. Only pass `text` to the classifier. The manifest counts omitted and unlabeled rows.

Pass the classification package directory directly to `compute-metadata-embeddings` with `--format jsonl` or `--format sqlite`. The encoder receives the prepared `text` with an empty additional prompt. Choose `gemma-classification` during preparation to include the Gemma classification prefix in that text.

The vectors manifest records `classification.mode` and `classification.labelProfile`, together with the source package and text recipe. Each JSONL vector includes `labels` alongside `network`, `catalogId` and `vector`. A train vector requires a nonempty list of reference labels. A predict vector carries the known labels or JSON null when the reference is unknown.

In SQLite, join `labels.item` to `items.id` and `vectors.item`. The `labels.labels_json` column stores the ordered JSON array or SQL NULL. Both `enrichments lookup-vector` and `enrichments pack-vectors` preserve the labels and that distinction.
