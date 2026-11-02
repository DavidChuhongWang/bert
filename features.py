"""STUDENT IMPLEMENTATION: frozen BERT extraction and reusable span caches.

Implement build_cache with the signature below and this returned interface:
cache.layer(k) is a read-only float32 (num_spans, 768) memory map;
cache.targets is an int64 label-ID memory map; cache.metadata binds data/protocol.
Use full sentences, word_ids(), all 13 hidden states, right padding, masks,
no gradients, and WordPiece mean pooling. Reuse disk shards across seeds.
"""

import json
import shutil
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from data import inspect_data, iter_sentences, wordpiece_bounds
from protocol import LAYERS, MODEL_ID, REVISION, protocol_sha256, read_json, write_json


class FeatureCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata = read_json(self.directory / 'metadata.json')
        self.targets = np.load(self.directory / 'targets.npy', mmap_mode='r')

    def layer(self, k):
        return np.load(self.directory / f'layer_{k:02d}.npy', mmap_mode='r')


def build_cache(data_path, labels, cache_dir, *, device="cpu", batch_size=8,
                local_files_only=False, tokenizer=None, encoder=None):
    directory = Path(cache_dir)
    identity = {
        **inspect_data(data_path, labels),
        'protocol_sha256': protocol_sha256(labels),
        'labels': labels, 'hidden_size': 768, 'dtype': 'float32',
        'cache_version': 'wordpiece-mean-v1',
    }
    if directory.exists() and any(directory.iterdir()):
        cache = FeatureCache(directory)
        if cache.metadata['identity'] != identity:
            raise ValueError('Feature cache mismatch; use a fresh cache directory')
        if cache.targets.shape != (identity['num_spans'],) or cache.targets.dtype != np.int64:
            raise ValueError('Invalid cached targets')
        for k in LAYERS:
            layer = cache.layer(k)
            if layer.shape != (identity['num_spans'], 768) or layer.dtype != np.float32:
                raise ValueError('Invalid cached layer')
        return cache

    parent = directory
    while not parent.exists():
        parent = parent.parent
    required = identity['num_spans'] * (13 * 768 * 4 + 8)
    if shutil.disk_usage(parent).free < required + 1024 ** 3:
        raise OSError('Insufficient storage for the required float32 span cache')
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=REVISION, use_fast=True, local_files_only=local_files_only)
    tokenizer.padding_side = 'right'
    if encoder is None:
        encoder = AutoModel.from_pretrained(
            MODEL_ID, revision=REVISION, local_files_only=local_files_only)
    encoder.to(device).eval()
    encoder.requires_grad_(False)
    directory.mkdir(parents=True, exist_ok=True)
    count = identity['num_spans']
    layers = [np.lib.format.open_memmap(directory / f'layer_{k:02d}.npy',
              mode='w+', dtype=np.float32, shape=(count, 768)) for k in LAYERS]
    targets = np.lib.format.open_memmap(directory / 'targets.npy', mode='w+',
                                      dtype=np.int64, shape=(count,))
    label_ids = {label: i for i, label in enumerate(labels)}
    sentences = iter(iter_sentences(data_path, labels))
    offset = processed = 0
    with (directory / 'rows.jsonl').open('w', encoding='utf-8') as rows:
        while batch := list(islice(sentences, batch_size)):
            encoded = tokenizer([s['tokens'] for s in batch], is_split_into_words=True,
                                add_special_tokens=True, padding=True, truncation=False,
                                return_attention_mask=True, return_tensors='pt')
            if encoded['input_ids'].shape[1] > 512:
                raise ValueError('Sentence exceeds 512 WordPieces; truncation is forbidden')
            bounds = [wordpiece_bounds(encoded.word_ids(batch_index=i), len(s['tokens']),
                      encoded['attention_mask'][i].tolist()) for i, s in enumerate(batch)]
            inputs = {key: value.to(device) for key, value in encoded.items()}
            inputs['token_type_ids'] = torch.zeros_like(inputs['input_ids'])
            with torch.no_grad():
                hidden = encoder(**inputs, output_hidden_states=True).hidden_states
                locations = []
                for i, sentence in enumerate(batch):
                    for span in sentence['constituents']:
                        left = bounds[i][span['start']][0]
                        right = bounds[i][span['end'] - 1][1]
                        locations.append((i, left, right))
                        targets[offset + len(locations) - 1] = label_ids[span['label']]
                        rows.write(json.dumps({'id': sentence['id'], **span}) + '\n')
                for k in LAYERS:
                    pooled = torch.stack([hidden[k][i, left:right].mean(0)
                                          for i, left, right in locations])
                    layers[k][offset:offset + len(locations)] = pooled.float().cpu().numpy()
                offset += len(locations)
            processed += len(batch)
            if processed % (batch_size * 100) == 0:
                print(f"Features: {processed}/{identity['num_sentences']} sentences", flush=True)
    for layer in layers:
        layer.flush()
    targets.flush()
    write_json(directory / 'metadata.json', {'identity': identity})
    return FeatureCache(directory)
