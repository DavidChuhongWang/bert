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
import csv
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from torch import nn

from artifacts import initialize_artifacts, load_checkpoint, save_checkpoint
from config import configure_runtime, load_config
from data import load_labels, verify_public_manifest
from evaluator import evaluate_artifacts
from features import build_cache
from probe_model import Probe, evaluate_probe, feature_batches
from protocol import LAYERS, SEEDS, protocol_sha256, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path,
                        help="Fresh run root relative to the current directory; reuse input data and cache")
    args = parser.parse_args()
    config, paths = load_config(args.config)
    if args.output_root is not None:
        root = args.output_root.resolve()
        paths['artifacts'] = root / 'artifacts/part1'
        paths['results'] = root / 'results'
    outputs = [paths['artifacts'], paths['results']]
    inputs = [paths[key] for key in ('train', 'dev', 'labels', 'data_manifest', 'cache')]
    for i, output in enumerate(outputs):
        for other in inputs + outputs[:i]:
            if output == other or output in other.parents or other in output.parents:
                raise ValueError('Input, cache, artifact and result paths must not overlap')
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise FileExistsError('Artifact/results directories must be empty; use --output-root')
    runtime = config['runtime']
    device = configure_runtime(runtime)
    labels = load_labels(paths['labels'])
    pins = verify_public_manifest(paths['data_manifest'], paths)
    started = time.perf_counter()
    info = {'device': str(device), 'platform': platform.platform(),
            'python': platform.python_version(), 'torch': torch.__version__,
            'numpy': np.__version__, 'transformers': transformers.__version__,
            'num_threads': runtime['num_threads']}
    if device.type == 'cuda':
        info['device_name'] = torch.cuda.get_device_name(device)
    manifest = initialize_artifacts(paths['artifacts'], config, paths, pins, info)
    paths['results'].mkdir(parents=True, exist_ok=True)
    caches = {}
    for split in ('train', 'dev'):
        directory = paths['cache'] / f'{split}-{pins[split]}-{protocol_sha256(labels)}'
        caches[split] = build_cache(paths[split], labels, directory, device=str(device),
            batch_size=runtime['feature_batch_size'], local_files_only=runtime['local_files_only'])
    info['feature_seconds'] = time.perf_counter() - started
    fields = ['layer', 'seed', 'epoch', 'train_loss', 'train_num_spans',
              'dev_loss', 'dev_num_correct', 'dev_num_spans', 'dev_accuracy']
    protocol = config['protocol']
    with (paths['results'] / 'training.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for layer in LAYERS:
            train_x, dev_x = caches['train'].layer(layer), caches['dev'].layer(layer)
            for seed in SEEDS:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if device.type == 'cuda':
                    torch.cuda.manual_seed_all(seed)
                if device.type == 'mps':
                    torch.mps.manual_seed(seed)
                shuffle = np.random.default_rng(seed)
                probe = Probe(len(labels)).to(device)
                optimizer = torch.optim.Adam(probe.parameters(), lr=protocol['learning_rate'],
                    betas=tuple(protocol['adam_betas']), eps=protocol['adam_eps'],
                    weight_decay=protocol['weight_decay'])
                best_correct = -1
                for epoch in range(1, protocol['epochs'] + 1):
                    probe.train()
                    total_loss = 0.0
                    count = 0
                    order = shuffle.permutation(len(caches['train'].targets))
                    for x, y in feature_batches(train_x, caches['train'].targets,
                            protocol['batch_size'], device, order):
                        optimizer.zero_grad()
                        loss = nn.functional.cross_entropy(probe(x), y)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * len(y)
                        count += len(y)
                    metrics = evaluate_probe(probe, dev_x, caches['dev'].targets,
                        device=device, batch_size=runtime['eval_batch_size'])
                    if metrics['num_correct'] > best_correct:
                        best_correct = metrics['num_correct']
                        entry = save_checkpoint(paths['artifacts'], probe, layer=layer,
                            seed=seed, best_epoch=epoch, dev_metrics=metrics)
                    writer.writerow(dict(layer=layer, seed=seed, epoch=epoch,
                        train_loss=total_loss / count, train_num_spans=count,
                        dev_loss=metrics['loss'], dev_num_correct=metrics['num_correct'],
                        dev_num_spans=metrics['num_spans'], dev_accuracy=metrics['accuracy']))
                    stream.flush()
                    print(f"Layer {layer}, seed {seed}, epoch {epoch}: dev {metrics['accuracy']:.6f}", flush=True)
                best = load_checkpoint(paths['artifacts'], entry, len(labels), device=device)
                evaluate_probe(best, dev_x, caches['dev'].targets,
                    device=device, batch_size=runtime['eval_batch_size'])
                del probe, best, optimizer
    info['training_and_features_seconds'] = time.perf_counter() - started
    evaluate_artifacts(paths['artifacts'], paths['dev'], paths['results'] / 'runs.csv',
        device=str(device), cache_dir=paths['cache'], batch_size=runtime['eval_batch_size'])
    info['total_seconds'] = time.perf_counter() - started
    manifest = read_json(paths['artifacts'] / 'manifest.json')
    manifest['runtime'] = info
    write_json(paths['artifacts'] / 'manifest.json', manifest)
    from plot_results import plot_results
    plot_results(paths['results'] / 'runs.csv', paths['results'] / 'layers.csv',
                 paths['results'] / 'layer_accuracy.pdf')


if __name__ == "__main__":
    main()
