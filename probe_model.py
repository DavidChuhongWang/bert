"""Fixed architecture and minibatch metrics, independent of student code."""

import numpy as np
import torch
from torch import nn


class Probe(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        if type(num_labels) is not int or num_labels < 1:
            raise ValueError("num_labels must be a positive integer")
        self.linear1 = nn.Linear(768, 1024, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(1024, num_labels, bias=True)

    def forward(self, features):
        return self.linear2(self.relu(self.linear1(features)))


def feature_batches(features, targets, batch_size, device, order=None):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    if len(features) != len(targets) or len(targets) == 0:
        raise ValueError("Features and targets must have equal nonzero lengths")
    for start in range(0, len(targets), batch_size):
        indices = slice(start, start + batch_size) if order is None else order[start : start + batch_size]
        # Copy only one minibatch from the read-only memory map.
        x = torch.from_numpy(np.array(features[indices], dtype=np.float32, copy=True)).to(device)
        y = torch.from_numpy(np.array(targets[indices], dtype=np.int64, copy=True)).to(device)
        yield x, y


def evaluate_probe(probe, features, targets, *, device="cpu", batch_size=256):
    probe.eval()
    total_loss = 0.0
    correct = count = 0
    with torch.no_grad():
        for x, y in feature_batches(features, targets, batch_size, device):
            logits = probe(x)
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite probe logits")
            total_loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
            correct += int((logits.argmax(dim=-1) == y).sum().item())
            count += len(y)
    return {"loss": total_loss / count, "num_correct": correct, "num_spans": count, "accuracy": correct / count}
