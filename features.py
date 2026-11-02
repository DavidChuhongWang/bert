"""STUDENT IMPLEMENTATION: frozen BERT extraction and reusable span caches.

Implement build_cache with the signature below and this returned interface:
cache.layer(k) is a read-only float32 (num_spans, 768) memory map;
cache.targets is an int64 label-ID memory map; cache.metadata binds data/protocol.
Use full sentences, word_ids(), all 13 hidden states, right padding, masks,
no gradients, and WordPiece mean pooling. Reuse disk shards across seeds.
"""


def build_cache(data_path, labels, cache_dir, *, device="cpu", batch_size=8,
                local_files_only=False, tokenizer=None, encoder=None):
    raise NotImplementedError("Implement frozen extraction, span pooling, and fingerprinted disk caches")
