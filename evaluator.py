"""Evaluate saved probes using the project's feature extractor."""

import argparse
import csv
import sys
import tempfile
from pathlib import Path

# Python -I omits the script directory. Add only this project directory,
# never the working directory or any path inside an artifact bundle.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from artifacts import load_checkpoint, validate_artifacts
from config import resolve_device
from data import load_labels
from probe_model import evaluate_probe
from protocol import protocol_sha256, sha256_file

RUN_FIELDS = ["layer", "seed", "split", "best_epoch", "num_correct", "num_spans", "accuracy", "data_sha256", "protocol_sha256"]


def write_runs(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RUN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_artifacts(artifact_dir, data_path, output_path, *, labels_path=None,
                       split="dev", device="cpu", cache_dir=None, batch_size=32,
                       feature_builder=None):
    """Recompute 65 accuracies from saved probes and full-sentence features.

    The default extractor is features.build_cache from the caller's project.
    The TA supplies a trusted private feature_builder for independent grading.
    labels_path defaults to data_path.parent / 'labels.json'. cache_dir must be
    caller-owned; None uses a fresh temporary cache. batch_size controls sentence
    extraction and probe evaluation. Train/dev inputs must match artifact pins.
    """
    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be train, dev, or test")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = resolve_device(device)
    data_path = Path(data_path)
    labels = load_labels(labels_path if labels_path is not None else data_path.parent / "labels.json")
    manifest, _ = validate_artifacts(artifact_dir, labels)
    data_hash = sha256_file(data_path)
    if split in {"train", "dev"} and data_hash != manifest["dataset_sha256"][split]:
        raise ValueError(f"{split} dataset fingerprint mismatch")

    if feature_builder is None:
        from features import build_cache

        feature_builder = build_cache

    def run(cache_path):
        cache = feature_builder(data_path, labels, cache_path, device=str(device), batch_size=batch_size)
        rows = []
        for entry in sorted(manifest["runs"], key=lambda row: (row["layer"], row["seed"])):
            probe = load_checkpoint(artifact_dir, entry, len(labels), device=device)
            metrics = evaluate_probe(probe, cache.layer(entry["layer"]), cache.targets, device=device, batch_size=batch_size)
            rows.append({
                "layer": entry["layer"], "seed": entry["seed"], "split": split,
                "best_epoch": entry["best_epoch"],
                "num_correct": metrics["num_correct"], "num_spans": metrics["num_spans"],
                "accuracy": metrics["accuracy"], "data_sha256": data_hash,
                "protocol_sha256": protocol_sha256(labels),
            })
            del probe
        if sha256_file(data_path) != data_hash:
            raise ValueError("Evaluation data changed during evaluation")
        write_runs(output_path, rows)
        return rows

    if cache_dir is not None:
        # Separate splits and dataset versions; never discover caches in artifacts.
        return run(Path(cache_dir) / f"{split}-{data_hash}-{protocol_sha256(labels)}")
    with tempfile.TemporaryDirectory(prefix="cs489-a3-eval-") as temporary:
        return run(Path(temporary) / "features")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--labels", type=Path, help="Trusted course labels; default: labels.json beside data")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", type=Path, help="Caller-owned cache only, never a submitted cache")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    evaluate_artifacts(args.artifacts, args.data, args.output, labels_path=args.labels,
                       split=args.split, device=args.device, cache_dir=args.cache_dir, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
