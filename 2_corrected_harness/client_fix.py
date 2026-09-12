#!/usr/bin/env python3
"""
Client wrappers for the corrected pipeline.

`FixedClient` builds the model at the task's class count before any parameter exchange, so the
classification head takes part in aggregation. `FullStateClient` exchanges the full state dictionary
(parameters and normalisation buffers) rather than the parameter list. `TolerantFullStateClient`
exchanges the full state but silently skips shape-mismatched tensors; it exists only to reproduce the
original behaviour for the controlled comparison in the paper. The photometric statistic a client
reports is measured on its training split and raises on failure instead of returning a constant.
"""
import sys
from pathlib import Path
ROOT = Path("/ANON/fl_hod")
sys.path.insert(0, str(ROOT))
from fl.client_yolo10s import YOLOv10ClientWithAttack


class FixedClient(YOLOv10ClientWithAttack):
    """the model_path handed in is the SHARED nc=1 checkpoint, so Ultralytics never rewrites the
 head mid-run and all 322 tensors take part in aggregation."""

    def __init__(self, *a, nc: int = 1, **kw):
        super().__init__(*a, **kw)
        got = self.model.model.yaml.get("nc")
        if got != nc:
            raise RuntimeError(f"client {self.client_id}: expected nc={nc} checkpoint, got nc={got} "
                               f"- check")
        print(f"[Client {self.client_id}] check: nc={got}, {len(list(self.model.model.parameters()))} tensors")


def create_fixed_client_fn(client_configs, fedbn=False, extract_bn_stats=True,
                           attack=None, malicious_clients=None, seed=0, nc=1):
    malicious_clients = malicious_clients or []

    def client_fn(cid: str) -> FixedClient:
        i = int(cid); c = client_configs[i]
        return FixedClient(
            client_id=i, data_yaml=c["data_yaml"], model_path=c.get("model_path", "yolov10s.pt"),
            local_epochs=c.get("local_epochs", 3), batch_size=c.get("batch_size", 4),
            imgsz=c.get("imgsz", 640), device=c.get("device", "0"),
            fedbn=fedbn, extract_bn_stats=extract_bn_stats,
            attack=attack if i in malicious_clients else None, seed=seed, nc=nc)
    return client_fn


# ---------------------------------------------------------------------------
# exchange the full state, not just `named_parameters`.
#
# `get_parameters` in the frozen client iterates `named_parameters`, which returns 322 tensors.
# The model's `state_dict` has 619 entries: the other 297 are buffers, including every BN
# `running_mean` / `running_var` (42,304 elements). Buffers are never sent, never aggregated and
# never returned, so the server's "global model" is averaged weights bolted onto whatever
# normalisation statistics its own checkpoint happened to hold, not a coherent model.
# Clients hide this because they re-adapt BN during their own local epochs before self-scoring.
# ---------------------------------------------------------------------------
import numpy as _np
import torch as _torch


class FullStateClient(FixedClient):
    """Client that exchanges the complete state_dict."""

    def _keys(self):
        return [k for k in self.model.model.state_dict().keys()]

    def get_parameters(self, config):
        sd = self.model.model.state_dict()
        return [sd[k].detach().cpu().float().numpy() for k in self._keys()]

    def set_parameters(self, parameters):
        if not parameters:
            return
        sd = self.model.model.state_dict()
        keys = self._keys()
        if len(parameters) != len(keys):
            raise RuntimeError(f"client {self.client_id}: got {len(parameters)} tensors, "
                               f"expected {len(keys)} - full-state protocol violated")
        new = {}
        for k, arr in zip(keys, parameters):
            t = sd[k]
            v = _torch.from_numpy(_np.asarray(arr)).reshape(t.shape)
            new[k] = v.to(dtype=t.dtype)          # int buffers (num_batches_tracked) cast back
        self.model.model.load_state_dict(new)


def create_fullstate_client_fn(client_configs, extract_bn_stats=True, attack=None,
                               malicious_clients=None, seed=0, nc=1):
    malicious_clients = malicious_clients or []

    def client_fn(cid: str) -> FullStateClient:
        i = int(cid); c = client_configs[i]
        return FullStateClient(
            client_id=i, data_yaml=c["data_yaml"], model_path=c.get("model_path"),
            local_epochs=c.get("local_epochs", 3), batch_size=c.get("batch_size", 4),
            imgsz=c.get("imgsz", 640), device=c.get("device", "0"),
            fedbn=False, extract_bn_stats=extract_bn_stats,
            attack=attack if i in malicious_clients else None, seed=seed, nc=nc)
    return client_fn


# ---------------------------------------------------------------------------
# the photometric statistic must actually be measured.
# The frozen `_extract_bn_stats` treats the client YAML's `val`/`train` entries as directories.
# Here they are.txt list files, so Image.open raises and a bare except returns the constant
# {'mu': 0.29, 'sigma': 0.19} for every client, every round. We resolve list files properly and
# raise on failure, so a silent constant cannot recur.
# ---------------------------------------------------------------------------
import yaml as _yaml
from pathlib import Path as _Path
from PIL import Image as _Image


def _measure_photometric(data_yaml: str, max_images: int = 32) -> dict:
    """measure the training split only.

 The frozen convention reads `val` first. Our poisoned datasets keep val clean so that evaluation
 stays honest, so a val-first statistic reports a diluted value for a poisoned client
 (mu = 0.4516 instead of 0.7060 for leipzig under Brightness Flood) and the screen misses it.
 Semantically the reliability signal should describe the data the client actually trains on.
 """
    cfg = _yaml.safe_load(open(data_yaml))
    root = _Path(cfg["path"])
    paths = []
    for key in ("train",):
        rel = cfg.get(key)
        if not rel:
            continue
        p = _Path(rel) if _Path(rel).is_absolute() else root / rel
        if p.is_file() and p.suffix.lower() == ".txt":
            paths += [_Path(l.strip()) for l in p.read_text().splitlines() if l.strip()]
        elif p.is_dir():
            for pat in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                paths += sorted(p.rglob(pat))
    paths = [q for q in paths[:max_images] if q.exists()]
    if not paths:
        raise RuntimeError(f"no images resolved from {data_yaml} - check")
    ms, ss = [], []
    for q in paths:
        with _Image.open(q) as im:
            arr = _np.asarray(im.convert("L"), dtype=_np.float32) / 255.0
        ms.append(float(arr.mean())); ss.append(float(arr.std()))
    return {"mu": float(_np.mean(ms)), "sigma": float(_np.mean(ss))}


def _patched_extract(self):
    return _measure_photometric(self.data_yaml)


FullStateClient._extract_bn_stats = _patched_extract
FixedClient._extract_bn_stats = _patched_extract


class TolerantFullStateClient(FullStateClient):
    """ factorial only: exchange the full state but SKIP shape-mismatched tensors, reproducing the
 original silent-skip behaviour while still transmitting buffers. This isolates the head fault
 from the buffer fault; the repaired client raises instead, which is why the strict client cannot
 run this arm."""

    def __init__(self, *a, **kw):
        kw['nc'] = kw.get('nc', 80)
        super(FixedClient, self).__init__(*a, **{k: v for k, v in kw.items() if k != 'nc'})
        self._skipped = 0

    def set_parameters(self, parameters):
        if not parameters:
            return
        sd = self.model.model.state_dict(); keys = list(sd.keys())
        new, skipped = {}, 0
        for k, arr in zip(keys, parameters):
            t = sd[k]; a = _np.asarray(arr)
            if int(a.size) != int(_np.prod(t.shape)):
                new[k] = t; skipped += 1; continue
            new[k] = _torch.from_numpy(a).reshape(t.shape).to(dtype=t.dtype)
        self._skipped = skipped
        if skipped:
            print(f"[Client {self.client_id}] skipped {skipped} shape-mismatched tensors (fault arm)")
        self.model.model.load_state_dict(new)


def create_tolerant_client_fn(client_configs, extract_bn_stats=True, attack=None,
                              malicious_clients=None, seed=0, nc=80):
    malicious_clients = malicious_clients or []
    def client_fn(cid: str):
        i = int(cid); c = client_configs[i]
        return TolerantFullStateClient(
            client_id=i, data_yaml=c["data_yaml"], model_path=c.get("model_path"),
            local_epochs=c.get("local_epochs", 3), batch_size=c.get("batch_size", 4),
            imgsz=c.get("imgsz", 640), device=c.get("device", "0"),
            fedbn=False, extract_bn_stats=extract_bn_stats,
            attack=attack if i in malicious_clients else None, seed=seed, nc=nc)
    return client_fn


# ---------------------------------------------------------------------------------------------
# Ultralytics seeds its dataloader generator from a hard-coded constant
# (ultralytics/data/build.py: generator.manual_seed(6148914691236517205 + RANK)), so the caller's
# `seed` argument never reaches the data order and every seed value produces a bit-identical run.
# InfiniteDataLoader consumes the generator in __init__, so it must be reseeded before that call,
# from torch.initial_seed, which the client sets to eff_seed immediately before training.
import ultralytics.data.build as _ub
if not getattr(_ub.InfiniteDataLoader, "_seed_patched", False):
    _orig_idl_init = _ub.InfiniteDataLoader.__init__
    def _seeded_idl_init(self, *a, **kw):
        g = kw.get("generator")
        if g is not None:
            g.manual_seed(int(_torch.initial_seed()) % (2**63 - 1))
        _orig_idl_init(self, *a, **kw)
    _ub.InfiniteDataLoader.__init__ = _seeded_idl_init
    _ub.InfiniteDataLoader._seed_patched = True
