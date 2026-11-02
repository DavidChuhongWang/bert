"""STUDENT IMPLEMENTATION: train independent probes for every layer and seed.

Use config.load_config, data.verify_public_manifest, your features.build_cache,
probe_model.Probe, and artifacts helpers. Train only probes with the fixed
protocol. Save best full-dev accuracy, keeping the earlier epoch on ties.
Reload each best checkpoint before reporting. Export portable config.yaml,
labels.json, manifest.json, all 65 checkpoints, and results/training.csv.
Use evaluator.py to produce final results/runs.csv.
Refuse nonempty artifact/results directories; --output-root selects a fresh
root for artifacts/part1 and results while reusing the input data and cache.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path,
                        help="Fresh run root relative to the current directory; reuse input data and cache")
    parser.parse_args()
    raise NotImplementedError("Implement the complete fixed-protocol experiment using features.py")


if __name__ == "__main__":
    main()
