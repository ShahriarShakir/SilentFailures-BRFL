#!/usr/bin/env python3
"""
Build the detector at the task's class count so the classification head federates.

The released checkpoint has 80 output classes. A client that loads it directly is constructed with an
80-class head; the library rewrites the head to the task's class count only when training first reads
the data YAML. In between, an aggregated single-class parameter vector does not match the head's
shape, and a tolerant loader skips those tensors, so the classifier is re-initialised at every client
in every round. This module produces one shared checkpoint at the task's class count, from which every
client and the server evaluator are built.
"""
from __future__ import annotations
from ultralytics import YOLO


def build_nc_model(weights_path: str, nc: int = 1):
    y = YOLO(weights_path)
    cls = y.model.__class__
    new = cls(cfg=y.model.yaml, nc=nc, verbose=False)
    old_sd, new_sd = y.model.state_dict(), new.state_dict()
    carried = 0
    for k, v in old_sd.items():
        if k in new_sd and new_sd[k].shape == v.shape:
            new_sd[k] = v
            carried += 1
    new.load_state_dict(new_sd)
    y.model = new
    return y, carried


def n_mismatched(weights_path: str, nc: int = 1) -> int:
    """How many tensors the un-fixed path silently skips."""
    a = YOLO(weights_path)
    b = a.model.__class__(cfg=a.model.yaml, nc=nc, verbose=False)
    return sum(1 for (n1, p1), (n2, p2) in zip(a.model.named_parameters(), b.named_parameters())
               if p1.shape != p2.shape)
