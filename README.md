# A3: frozen BERT constituent probes

Classify the label of a **given gold constituent span** using frozen BERT representations.
Download `train.jsonl`, `dev.jsonl`, `labels.json`, and `manifest.json` from [compling/cs489-f26-a3](https://huggingface.co/datasets/compling/cs489-f26-a3) into `data/`.
The dataset card provides the download command and describes the constituent-label data format.
The tests use handwritten English examples.

## Fixed files and student work

**Keep fixed:** `protocol.py`, `config.py`, `data.py`, `probe_model.py`, `artifacts.py`, `evaluator.py`, `pyproject.toml`, and `uv.lock`.

**Implement:** `features.py`, `run_part1.py`, and `plot_results.py`. They are intentional stubs.
Your training entry point and the starter evaluator both use your `features.py` implementation.
The evaluator loads checkpoints with the fixed model loader.
The TA uses separate course feature extraction for final grading.

## Setup and commands

Copy the starter contents into your project root and place the authorized data under `data/`.
Install `uv`, then run:

```sh
uv sync --locked
uv run python run_part1.py --config config.example.yaml
uv run python evaluator.py --artifacts artifacts/part1 --data data/dev.jsonl --output results/runs.csv --device cpu
uv run python plot_results.py --input results/runs.csv --table results/layers.csv --output results/layer_accuracy.pdf
```

The training, evaluation, and plotting commands require your implementations.
Python is pinned to 3.12.12; the lockfile pins transitive dependencies.
Set `runtime.device` explicitly to `cpu`, `cuda:0`, another available CUDA index, or `mps` when supported.
Unavailable devices are errors, never a CPU fallback. PyTorch operators may still reject unsupported device operations.
The evaluator has its own `--device` setting and does not inherit the training device.
Feature batch size, evaluation batch size, CPU thread count, deterministic algorithms, and offline model loading are explicit runtime options.
GPU nondeterminism and platform-dependent floating-point differences are possible; cross-platform bitwise reproduction is not required.

The model and fast tokenizer are `google-bert/bert-base-cased`, pinned to revision `cd5ef92a9fb2f889e972770a36d4ed042daf221e`, resolved using the unauthenticated Hugging Face model API.
Normal extraction downloads that exact revision if not already cached. Set `local_files_only: true` after caching it to require offline loading.
There is no fallback to another revision or randomly initialized encoder.
The automated tests use a local toy vocabulary and do not download pretrained weights.

## Data and feature contract

Each JSONL line has exactly `id`, `tokens`, and `constituents`.
Each constituent has `start`, `end`, and `label`; boundaries are zero-based, half-open **word** indices, unique and sorted by `(start, end)`.
`labels.json` is a sorted list of unique strings and fixes the classifier output order.
`data.verify_public_manifest` checks `files["train.jsonl"]`, `files["dev.jsonl"]`, and `files["labels.json"]`; each entry is a SHA-256 string or an object containing `sha256`.
It also accepts the preparation tool's equivalent `artifacts` inventory key (but rejects ambiguous manifests containing both keys).
The prep manifest, not your own split or label policy, defines the released data.
All source sentence occurrences are retained in their original splits, including repeated text and conflicting annotations across rows; do not deduplicate them or merge their labels.

Encode each full sentence with the pinned fast tokenizer and `word_ids()`, right padding, special tokens, zero token-type IDs, and an attention mask passed to BERT.
Reject inputs longer than 512 WordPieces including special tokens; never truncate.
BERT is frozen, in evaluation mode, and run without gradients.
Layer 0 is the embedding module output (including position/type embeddings and LayerNorm); layers 1 through 12 are block outputs.
Average all WordPieces inside the gold span directly, excluding special tokens and padding. Do not first average pieces within each word.

`features.build_cache(data_path, labels, cache_dir, *, device="cpu", batch_size=8, local_files_only=False, tokenizer=None, encoder=None)` returns an object with:

- `layer(k)`: read-only float32 NumPy memory map of shape `(num_spans, 768)`.
- `targets`: int64 memory map of label IDs in sentence/span order.
- `metadata["identity"]`: data hash, protocol hash, labels, sizes, dtype, and cache version.

The tokenizer/encoder injection arguments are solely an offline testing seam, never an alternative model option for experiments.
The reference cache writes `layer_00.npy` through `layer_12.npy`, `targets.npy`, `rows.jsonl`, and fingerprinted `metadata.json`.
It streams sentence batches and saves pooled vectors, not token states or computation graphs.
One split cache is shared across all seeds; a mismatching existing cache is rejected.
Use a new directory after a protocol/data change, rather than ignoring a mismatch.
Do not submit caches.

## Training and artifact contract

For every layer 0 through 12 and seed 0 through 4, independently initialize:

```text
Linear(768, 1024, bias=True) -> ReLU -> Linear(1024, C, bias=True)
```

Use cross-entropy on logits, Adam (learning rate 0.001, betas 0.9/0.999, epsilon 1e-8, weight decay 0), batch size 256, and all 10 epochs.
Reset Python, NumPy, and PyTorch random states for each run; shuffle training spans reproducibly.
Never update BERT or warm-start from another probe.
Select the highest accuracy on the entire dev split, keeping the earliest epoch on ties; reload the best checkpoint for final scores.

Use `artifacts.initialize_artifacts`, `save_checkpoint`, and `load_checkpoint` to maintain:

```text
artifacts/part1/
  config.yaml
  labels.json
  manifest.json
  checkpoints/layer_00/seed_0.pt
  ... 65 checkpoints through layer_12/seed_4.pt
```

Checkpoints contain only CPU float32 tensor state dictionaries with the fixed parameter names.
Loading uses `torch.load(weights_only=True, map_location="cpu")`, finite-tensor checks, and strict fixed-architecture loading.
Do not save model objects, optimizer states, or BERT weights.
The manifest binds checkpoint hashes, `(layer, seed)`, best epoch, full-dev counts, runtime, dataset train/dev/labels hashes, and the fixed protocol hash.
Metadata is provenance, not trusted evaluation output.
All config paths are relative to the YAML, not the working directory; `config.export_config` rewrites paths relative to exported `artifacts/part1/config.yaml`.
Keep the whole project directory structure when moving it. Supply the same authorized data again for reproduction:

```sh
uv run python run_part1.py --config artifacts/part1/config.yaml --output-root rerun-01
```

Training refuses nonempty artifact/results directories and does not resume an incomplete run.
Implement `--output-root` to direct `artifacts/part1/` and `results/` into a new run directory while retaining the configured data and feature cache.
This CLI root is relative to the current working directory; exported YAML paths are still relative to their YAML file.
Input files, the feature cache, artifacts, and results must not overlap in a way that overwrites inputs or mixes outputs.
Checkpoints alone are insufficient to prove training provenance. The instructor may spot-check executable code and released-data reproduction.

## Development evaluation and reporting

Public Python API:

```python
from evaluator import evaluate_artifacts
rows = evaluate_artifacts(
    artifact_dir, data_path, output_path,
    labels_path=None, split="dev", device="cpu", cache_dir=None, batch_size=32,
)
```

`labels_path` is the **trusted course label file**, defaulting to `labels.json` beside the evaluation data, never beside artifacts.
All 65 checkpoints are required. Train/dev input hashes must match the artifact pins; held-out `split="test"` accepts the explicitly supplied test file without changing checkpoint selection.
The evaluator recomputes predictions and counts; it never reads submitted scores or predictions as answers.
By default it uses a temporary feature cache. `--cache-dir` may point to a **caller-owned trusted cache**, never a submitted cache; datasets and splits receive separate fingerprinted subdirectories.
The CLI also accepts `--labels`, `--split`, `--batch-size`, and `--device`.
The TA uses a separate trusted extractor and grading entry point for final evaluation.
Tensor-only loading is not a resource sandbox; isolate untrusted submissions and apply normal file/resource limits.

`results/runs.csv` has exactly 65 best-checkpoint dev rows:

```text
layer,seed,split,best_epoch,num_correct,num_spans,accuracy,data_sha256,protocol_sha256
```

Accuracy is total correct spans divided by total evaluated spans, not an average of batch accuracies.
`results/layers.csv` has exactly 13 rows:

```text
layer,split,n_seeds,mean_accuracy,std_accuracy,se_accuracy,data_sha256,protocol_sha256
```

Use sample SD (`ddof=1`) and `SE = SD / sqrt(5)` across seeds, not spans.
Require all layer/seed combinations and identical split/data/protocol fingerprints before aggregating.
Save CSV accuracy in `[0, 1]`. Plot mean plus/minus one SE; scale both mean and SE by 100 for percentages.
`results/training.csv` records `layer,seed,epoch,train_loss,train_num_spans,dev_loss,dev_num_correct,dev_num_spans,dev_accuracy` for all 650 epochs.

## Submission and limitations

Submit the executable project, `README.md` with exact reproduction commands, `pyproject.toml`, `uv.lock`, completed stubs, unchanged fixed modules, portable artifacts, the three CSV files, `results/layer_accuracy.pdf`, and `writeup.pdf` (at most two pages).
Keep course data, predictions, feature/model caches, `.venv`, and optimizer states outside the submission.
There is no W&B requirement, Kaggle submission, retraining competition, leaderboard, or accuracy threshold inferred from synthetic tests.
The complete real-data reference experiment has passed independent evaluation with this 10-epoch protocol.
The laptop run took about 74 minutes for feature preparation and training, with 27.3 GiB of feature caches and 19.2 GiB observed process maximum RSS (including resident memory-mapped pages).
Those are local RTX 4090 Laptop measurements, not a T4 or lower-RAM guarantee; supported-device resource guidance remains to be confirmed.

