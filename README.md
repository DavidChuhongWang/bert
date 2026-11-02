# A3: Probing BERT for constituent labels

## Implemented

- `features.py`: extracts frozen BERT representations and averages the WordPieces in each span, caching features for all 13 layers.
- `run_part1.py`: trains five independent probes per layer for 10 epochs and saves the best development checkpoint for each run.
- `plot_results.py`: summarizes development accuracy across seeds and plots the mean with standard-error bars.

## Run

Place the course files (`train.jsonl`, `dev.jsonl`, `labels.json`, and `manifest.json`) in `data/`. Set the device in your YAML configuration to match your hardware; the example uses `cuda:0`. Feature extraction needs about 28 GiB of disk space, plus room for checkpoints.

Install the environment and train from scratch:

```sh
uv sync --locked
uv run python run_part1.py --config config.example.yaml
```

Training writes checkpoints to `artifacts/part1/` and scores and the plot to `results/`.

To reproduce the full submission using its saved configuration, use a fresh output directory:

```sh
uv run python run_part1.py --config artifacts/part1/config.yaml --output-root rerun-01
```

To evaluate saved checkpoints and regenerate the plot:

```sh
uv run python evaluator.py --artifacts artifacts/part1 --data data/dev.jsonl --output results/runs.csv
uv run python plot_results.py --input results/runs.csv --table results/layers.csv --output results/layer_accuracy.pdf
```


## Verification notes

The fixed course modules are unchanged. Checks passed for span pooling, cache validation, five-seed statistics, checkpoint selection, and all 65 saved development scores. Full real-data training was not repeated during the final review. Private-test results are not verified.

YAML layer, seed, and Adam-beta lists use compact inline formatting. Optional checks for tokenizer type, data changing during extraction, reloaded-score equality, and plotting best-epoch bounds were removed. Assignment-required validation remains.
