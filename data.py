"""Fixed streaming sentence schema, label IDs, data pins, and WordPiece alignment."""

import json
from pathlib import Path

from protocol import read_json, sha256_file, validate_sha256


def load_labels(path):
    labels = read_json(path)
    if (
        not isinstance(labels, list)
        or not labels
        or any(not isinstance(label, str) or not label for label in labels)
        or labels != sorted(set(labels))
    ):
        raise ValueError("labels.json must be a nonempty sorted list of unique strings")
    return labels


def iter_sentences(path, labels):
    allowed = set(labels)
    seen_ids = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                sentence = json.loads(line)
                if not isinstance(sentence, dict) or set(sentence) != {"id", "tokens", "constituents"}:
                    raise ValueError("Expected id, tokens, constituents")
                identity = sentence["id"]
                if not isinstance(identity, str) or not identity or identity in seen_ids:
                    raise ValueError("Sentence IDs must be nonempty and unique")
                seen_ids.add(identity)
                tokens = sentence["tokens"]
                if not isinstance(tokens, list) or not tokens or any(
                    not isinstance(token, str) or not token or token.isspace() for token in tokens
                ):
                    raise ValueError("Expected a nonempty list of word tokens")
                spans = sentence["constituents"]
                if not isinstance(spans, list) or not spans:
                    raise ValueError("Expected at least one constituent")
                previous = (-1, -1)
                for span in spans:
                    if not isinstance(span, dict) or set(span) != {"start", "end", "label"}:
                        raise ValueError("Expected start, end, label")
                    start, end = span["start"], span["end"]
                    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(tokens):
                        raise ValueError("Invalid half-open word span")
                    if (start, end) <= previous:
                        raise ValueError("Constituents must have unique boundaries sorted by (start, end)")
                    previous = (start, end)
                    if not isinstance(span["label"], str) or span["label"] not in allowed:
                        raise ValueError("Unknown constituent label")
                yield sentence
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"Invalid sentence at {path}:{line_number}: {error}") from error


def inspect_data(path, labels):
    sentences = spans = 0
    for sentence in iter_sentences(path, labels):
        sentences += 1
        spans += len(sentence["constituents"])
    if not sentences:
        raise ValueError(f"Empty dataset: {path}")
    return {"data_sha256": sha256_file(path), "num_sentences": sentences, "num_spans": spans}


def verify_public_manifest(manifest_path, paths):
    """Accept files/artifacts[name] as a SHA string or a sha256-bearing object."""
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or ("files" in manifest and "artifacts" in manifest):
        raise ValueError("Expected one unambiguous public manifest file inventory")
    files = manifest.get("files", manifest.get("artifacts", {}))
    if not isinstance(files, dict):
        raise TypeError("Public manifest file inventory must be a mapping")
    pins = {}
    for name in ("train", "dev", "labels"):
        filename = "labels.json" if name == "labels" else f"{name}.jsonl"
        entry = files.get(filename)
        expected = entry.get("sha256") if isinstance(entry, dict) else entry
        validate_sha256(expected)
        actual = sha256_file(paths[name])
        if actual != expected:
            raise ValueError(f"Public manifest fingerprint mismatch: {filename}")
        pins[name] = actual
    return pins


def wordpiece_bounds(word_ids, num_words, attention_mask):
    """Return word-to-piece half-open ranges, including the CLS input offset."""
    if len(word_ids) != len(attention_mask):
        raise ValueError("Alignment and attention mask lengths differ")
    active = sum(attention_mask)
    if list(attention_mask) != [1] * active + [0] * (len(word_ids) - active):
        raise ValueError("Only right padding with a binary attention mask is permitted")
    if active < 3 or word_ids[0] is not None or word_ids[active - 1] is not None:
        raise ValueError("Expected CLS and SEP outside the word alignment")
    if any(item is not None for item in word_ids[active:]):
        raise ValueError("Padding must not align to words")
    middle = word_ids[1 : active - 1]
    if any(type(item) is not int or not 0 <= item < num_words for item in middle):
        raise ValueError("Missing or invalid word alignment")
    if list(middle) != sorted(middle) or set(middle) != set(range(num_words)):
        raise ValueError("Every word must have contiguous ordered WordPieces")
    bounds = []
    cursor = 1
    for word in range(num_words):
        start = cursor
        while cursor < active - 1 and word_ids[cursor] == word:
            cursor += 1
        bounds.append((start, cursor))
    return bounds
