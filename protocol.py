"""Fixed course protocol. Do not change these values for the main experiment."""

import hashlib
import json
import re
from pathlib import Path

MODEL_ID = "google-bert/bert-base-cased"
# Resolved with the unauthenticated Hugging Face model API during implementation.
REVISION = "cd5ef92a9fb2f889e972770a36d4ed042daf221e"
LAYERS = tuple(range(13))
SEEDS = tuple(range(5))
PROTOCOL = {
    "version": "cs489-a3-v1",
    "data_schema": "sentence-word-spans-v1",
    "model_id": MODEL_ID,
    "model_revision": REVISION,
    "tokenizer_id": MODEL_ID,
    "tokenizer_revision": REVISION,
    "layers": list(LAYERS),
    "seeds": list(SEEDS),
    "hidden_size": 768,
    "probe_hidden_size": 1024,
    "probe": "Linear(768,1024,bias)-ReLU-Linear(1024,C,bias)",
    "pooling": "full-sentence-wordpiece-mean",
    "feature_dtype": "float32",
    "padding_side": "right",
    "max_length": 512,
    "frozen_encoder": True,
    "encoder_eval": True,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "adam_betas": [0.9, 0.999],
    "adam_eps": 1e-8,
    "weight_decay": 0.0,
    "batch_size": 256,
    "epochs": 10,
    "loss": "cross_entropy",
    "selection": "highest-full-dev-accuracy-earliest-tie",
}


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_sha256(labels):
    return canonical_hash({"protocol": PROTOCOL, "labels": labels})


def validate_sha256(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected a lowercase SHA-256 fingerprint")
    return value


def read_json(path):
    # Reject duplicate object keys instead of silently taking the last value.
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
