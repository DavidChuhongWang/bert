"""Fixed portable artifact contract and safe tensor-only checkpoint loading."""

import math
import shutil
from pathlib import Path

import torch
from config import export_config, load_config
from data import load_labels
from probe_model import Probe
from protocol import (
    LAYERS,
    SEEDS,
    protocol_sha256,
    read_json,
    sha256_file,
    validate_sha256,
    write_json,
)


def checkpoint_name(layer, seed):
    if type(layer) is not int or layer not in LAYERS or type(seed) is not int or seed not in SEEDS:
        raise ValueError("Invalid layer or seed")
    return f"checkpoints/layer_{layer:02d}/seed_{seed}.pt"


def initialize_artifacts(directory, config, paths, dataset_pins, runtime_info):
    directory = Path(directory)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError("Artifact directory must be empty; choose a fresh run directory")
    directory.mkdir(parents=True, exist_ok=True)
    labels = load_labels(paths["labels"])
    for key in ("train", "dev", "labels"):
        validate_sha256(dataset_pins[key])
    if sha256_file(paths["labels"]) != dataset_pins["labels"]:
        raise ValueError("Labels changed since dataset verification")
    if Path(paths["labels"]).resolve() != (directory / "labels.json").resolve():
        shutil.copyfile(paths["labels"], directory / "labels.json")
    export_config(config, paths, directory / "config.yaml")
    manifest = {
        "schema_version": 1,
        "protocol_sha256": protocol_sha256(labels),
        "dataset_sha256": dict(dataset_pins),
        "runtime": runtime_info,
        "runs": [],
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


def save_checkpoint(directory, probe, *, layer, seed, best_epoch, dev_metrics):
    directory = Path(directory)
    relative = checkpoint_name(layer, seed)
    if type(best_epoch) is not int or not 1 <= best_epoch <= 10:
        raise ValueError("best_epoch must be in 1..10")
    manifest = read_json(directory / "manifest.json")
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().clone() for key, value in probe.state_dict().items()}
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    entry = {
        "layer": layer, "seed": seed, "best_epoch": best_epoch,
        "num_correct": dev_metrics["num_correct"], "num_spans": dev_metrics["num_spans"],
        "accuracy": dev_metrics["accuracy"], "checkpoint": relative,
        "checkpoint_sha256": sha256_file(path),
    }
    validate_run(entry)
    manifest["runs"] = [row for row in manifest["runs"] if (row["layer"], row["seed"]) != (layer, seed)] + [entry]
    manifest["runs"].sort(key=lambda row: (row["layer"], row["seed"]))
    write_json(directory / "manifest.json", manifest)
    return entry


def validate_run(row):
    required = {"layer", "seed", "best_epoch", "num_correct", "num_spans", "accuracy", "checkpoint", "checkpoint_sha256"}
    if not isinstance(row, dict) or set(row) != required:
        raise ValueError("Invalid checkpoint metadata fields")
    if row["checkpoint"] != checkpoint_name(row["layer"], row["seed"]):
        raise ValueError("Checkpoint path does not match layer/seed")
    if type(row["best_epoch"]) is not int or not 1 <= row["best_epoch"] <= 10:
        raise ValueError("Invalid best epoch")
    correct, count = row["num_correct"], row["num_spans"]
    if type(correct) is not int or type(count) is not int or not 0 <= correct <= count or count == 0:
        raise ValueError("Invalid development counts")
    accuracy = row["accuracy"]
    if type(accuracy) not in (int, float) or not math.isfinite(accuracy) or not math.isclose(accuracy, correct / count, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Invalid development accuracy metadata")
    validate_sha256(row["checkpoint_sha256"])


def validate_artifacts(directory, expected_labels, *, require_complete=True):
    directory = Path(directory)
    if load_labels(directory / "labels.json") != expected_labels:
        raise ValueError("Artifact label mapping differs from trusted course labels")
    config, _ = load_config(directory / "config.yaml")
    manifest = read_json(directory / "manifest.json")
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "protocol_sha256", "dataset_sha256", "runtime", "runs"}:
        raise ValueError("Invalid artifact manifest")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("Unsupported artifact schema")
    if manifest["protocol_sha256"] != protocol_sha256(expected_labels):
        raise ValueError("Artifact protocol fingerprint mismatch")
    pins = manifest["dataset_sha256"]
    if not isinstance(pins, dict) or set(pins) != {"train", "dev", "labels"}:
        raise ValueError("Expected train/dev/labels dataset pins")
    for digest in pins.values():
        validate_sha256(digest)
    if sha256_file(directory / "labels.json") != pins["labels"]:
        raise ValueError("Artifact labels file fingerprint mismatch")
    if not isinstance(manifest["runtime"], dict) or not isinstance(manifest["runs"], list):
        raise TypeError("Invalid artifact runtime/runs")
    seen = set()
    for row in manifest["runs"]:
        validate_run(row)
        key = (row["layer"], row["seed"])
        if key in seen:
            raise ValueError("Duplicate layer/seed checkpoint")
        seen.add(key)
        if sha256_file(directory / row["checkpoint"]) != row["checkpoint_sha256"]:
            raise ValueError("Checkpoint fingerprint mismatch")
    if require_complete and seen != {(layer, seed) for layer in LAYERS for seed in SEEDS}:
        raise ValueError("Artifacts must contain all 13 layers and all 5 seeds exactly once")
    return manifest, config


def load_checkpoint(directory, entry, num_labels, *, device="cpu"):
    validate_run(entry)
    path = Path(directory) / checkpoint_name(entry["layer"], entry["seed"])
    if sha256_file(path) != entry["checkpoint_sha256"]:
        raise ValueError("Checkpoint fingerprint mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    probe = Probe(num_labels)
    expected = probe.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("Checkpoint must be the exact Probe state_dict")
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError("Checkpoint must contain ordinary dense tensors only")
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32 or tensor.shape != expected[name].shape:
            raise ValueError(f"Invalid checkpoint tensor: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite checkpoint tensor: {name}")
    probe.load_state_dict(state, strict=True)
    return probe.to(device).eval()
