"""STUDENT IMPLEMENTATION: validate 65 runs, aggregate five seeds, and plot.

Require exactly layers 0..12 and seeds 0..4, without duplicates, mixed splits,
nonfinite scores, inconsistent counts, or mixed data/protocol fingerprints.
Write 13 rows with fields:
layer,split,n_seeds,mean_accuracy,std_accuracy,se_accuracy,data_sha256,protocol_sha256
Use std(ddof=1) / sqrt(5), not population SD or binomial standard error.
Plot mean dev accuracy and +/- 1 SE. Label layer 0 as embedding-module output.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from protocol import LAYERS, SEEDS, validate_sha256


LAYER_FIELDS = ['layer', 'split', 'n_seeds', 'mean_accuracy', 'std_accuracy',
                'se_accuracy', 'data_sha256', 'protocol_sha256']


def plot_results(input_path, table_path, output_path):
    with Path(input_path).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    identities = set()
    counts = set()
    for row in rows:
        layer, seed = int(row['layer']), int(row['seed'])
        if (layer, seed) in seen or layer not in LAYERS or seed not in SEEDS:
            raise ValueError('Duplicate or invalid layer/seed')
        seen.add((layer, seed))
        correct, count = int(row['num_correct']), int(row['num_spans'])
        accuracy = float(row['accuracy'])
        if not 0 <= correct <= count or count == 0 or not math.isfinite(accuracy) or not math.isclose(
                accuracy, correct / count, rel_tol=0, abs_tol=1e-12):
            raise ValueError('Invalid accuracy or counts')
        for key in ('data_sha256', 'protocol_sha256'):
            validate_sha256(row[key])
        identities.add((row['split'], row['data_sha256'], row['protocol_sha256']))
        counts.add(count)
    if seen != {(k, s) for k in LAYERS for s in SEEDS} or len(identities) != 1 or len(counts) != 1:
        raise ValueError('Require all 65 runs with matching split, counts and fingerprints')
    split, data_hash, protocol_hash = identities.pop()
    if split != 'dev':
        raise ValueError('The layer curve must report development performance')
    summaries = []
    for layer in LAYERS:
        values = np.array([float(r['accuracy']) for r in rows if int(r['layer']) == layer])
        sd = values.std(ddof=1)
        summaries.append(dict(layer=layer, split=split, n_seeds=5,
            mean_accuracy=values.mean(), std_accuracy=sd, se_accuracy=sd / math.sqrt(5),
            data_sha256=data_hash, protocol_sha256=protocol_hash))
    Path(table_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(table_path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=LAYER_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)
    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout='constrained')
    ax.errorbar(LAYERS, [r['mean_accuracy'] * 100 for r in summaries],
                yerr=[r['se_accuracy'] * 100 for r in summaries], fmt='o-', capsize=3)
    ax.set(xticks=LAYERS, xlabel='BERT layer (0 = embedding module output)',
           ylabel='Development span accuracy (%)', title='Frozen BERT probes: mean +/- 1 SE across five seeds')
    ax.grid(alpha=0.25)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plot_results(args.input, args.table, args.output)


if __name__ == "__main__":
    main()
