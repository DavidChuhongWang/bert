"""Fixed YAML configuration and explicit runtime validation."""

import copy
import os
from pathlib import Path

import torch
import yaml
from protocol import PROTOCOL, canonical_hash

PATH_KEYS = {"train", "dev", "labels", "data_manifest", "artifacts", "cache", "results"}
RUNTIME_DEFAULTS = {
    "device": "cpu",
    "feature_batch_size": 8,
    "eval_batch_size": 32,
    "num_threads": 1,
    "deterministic": True,
    "local_files_only": False,
}


def validate_config(config):
    if not isinstance(config, dict) or set(config) != {"schema_version", "protocol", "paths", "runtime"}:
        raise ValueError("Config requires schema_version, protocol, paths, runtime only")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Unsupported config schema")
    if canonical_hash(config["protocol"]) != canonical_hash(PROTOCOL):
        raise ValueError("Config does not match the fixed course protocol")
    if not isinstance(config["paths"], dict) or set(config["paths"]) != PATH_KEYS:
        raise ValueError(f"Config paths must be {sorted(PATH_KEYS)}")
    for value in config["paths"].values():
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            raise ValueError("All config paths must be nonempty and relative to the YAML")
    runtime = config["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != set(RUNTIME_DEFAULTS):
        raise ValueError("Invalid runtime fields")
    for name in ("feature_batch_size", "eval_batch_size", "num_threads"):
        if type(runtime[name]) is not int or runtime[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("deterministic", "local_files_only"):
        if type(runtime[name]) is not bool:
            raise ValueError(f"{name} must be a boolean")
    if not isinstance(runtime["device"], str):
        raise TypeError("device must be an explicit string")
    return config


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        config = validate_config(yaml.safe_load(stream))
    paths = {key: (path.parent / value).resolve() for key, value in config["paths"].items()}
    return config, paths


def export_config(config, paths, destination):
    destination = Path(destination).resolve()
    exported = copy.deepcopy(config)
    exported["paths"] = {
        key: os.path.relpath(Path(value).resolve(), destination.parent)
        for key, value in paths.items()
    }
    validate_config(exported)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(exported, sort_keys=False), encoding="utf-8")


def resolve_device(name):
    try:
        device = torch.device(name)
    except (RuntimeError, ValueError, TypeError) as error:
        raise ValueError(f"Invalid device: {name}") from error
    if device.type == "cpu" and device.index is None:
        return device
    if device.type == "cuda":
        index = 0 if device.index is None else device.index
        if torch.cuda.is_available() and 0 <= index < torch.cuda.device_count():
            return torch.device("cuda", index)
    if device.type == "mps" and device.index is None and torch.backends.mps.is_available():
        return device
    raise ValueError(f"Requested device is unavailable or unsupported: {name}; no fallback is allowed")


def configure_runtime(runtime):
    # This must precede the first CUDA operation in a fresh training process.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(runtime["num_threads"])
    torch.use_deterministic_algorithms(runtime["deterministic"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return resolve_device(runtime["device"])
