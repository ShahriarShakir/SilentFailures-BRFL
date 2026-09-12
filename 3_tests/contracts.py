#!/usr/bin/env python3
"""
Evaluation and state contracts.

The paper argues that a widely used measurement is structurally unable to see a fault. The obvious
reply is to ask how the reader knows this measurement is sound. These are the checks that answer it.
Nothing here produces a scientific result; it constrains what the rest of the code is allowed to
claim.

 1. metric identity four distinct quantities are all called "mAP50" in common code. Name them.
 2. state hashes prove a tensor set actually travelled, and that two runs really differ.
 3. round timeline the paper's claim is about the order of steps, so record the order.
 4. buffer policy batch-norm carries float statistics and one integer counter. Averaging the
 counter is meaningless. Declare which is which.
 5. split integrity no training image may appear, or near-appear, in an evaluation split.
"""
import hashlib, json
from pathlib import Path

ROOT = Path("/ANON/fl_hod")
CFG = ROOT / "data/yolo/client_configs"
CITIES = ["budapest", "koeln", "leipzig", "lyon", "prague", "roma", "zagreb"]

# --- 1. metric identity --------------------------------------------------------
METRICS = {
    "G_pool": "one model, all held-out images pooled into a single evaluation set, scored once",
    "G_client_macro": "one model, scored on each client's held-out split, unweighted mean over clients",
    "G_client_weighted": "as G_client_macro, weighted by each client's held-out image count",
    "L_client_macro": "each client's own post-training model on its own held-out split, unweighted "
                      "mean over clients. This is the reported quantity under audit.",
}

# --- 2. state hashes -----------------------------------------------------------
def state_hash(sd, keys=None):
    """Order-independent SHA-256 over a tensor mapping. Ties a number to the bytes that produced it."""
    h = hashlib.sha256()
    for k in sorted(sd.keys() if keys is None else keys):
        v = sd[k]
        h.update(k.encode())
        arr = v.detach().cpu().numpy() if hasattr(v, "detach") else v
        h.update(str(getattr(arr, "shape", ())).encode())
        h.update(getattr(arr, "tobytes", lambda: str(arr).encode())())
    return h.hexdigest()[:16]

# --- 4. buffer policy ----------------------------------------------------------
FLOAT_BUFFERS = ("running_mean", "running_var")
INTEGER_COUNTERS = ("num_batches_tracked",)

def buffer_policy(sd):
    """Partition a state dict into what may be averaged and what may not."""
    out = {"parameters": [], "float_buffers": [], "integer_counters": [], "other_buffers": []}
    for k, v in sd.items():
        if any(k.endswith(s) for s in INTEGER_COUNTERS):   out["integer_counters"].append(k)
        elif any(k.endswith(s) for s in FLOAT_BUFFERS):    out["float_buffers"].append(k)
        elif hasattr(v, "dtype") and "float" in str(v.dtype): out["parameters"].append(k)
        else: out["other_buffers"].append(k)
    return out

# --- 5. split integrity --------------------------------------------------------
def _read(p):
    return [l.strip() for l in open(p) if l.strip()]

def _dhash(path, s=8):
    """64-bit difference hash. Robust to compression and small shifts, sensitive to scene change."""
    from PIL import Image
    import numpy as _np
    a = _np.asarray(Image.open(path).convert("L").resize((s + 1, s)), dtype=_np.int16)
    bits = (a[:, 1:] > a[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)

PIXEL_R = 0.95   # confirmation threshold for a hash candidate to count as a duplicate

def split_integrity(near_dup=True, ham_thresh=6):
    """Exact overlap between every train split and every evaluation split, plus a content-based
 near-duplicate check.

 An earlier version of this check flagged pairs whose filenames differed by one trailing index,
 on the assumption that ECP night frames come from driven sequences. All four pairs it returned
 were visually distinct (maximum pixel correlation +0.53), so filename adjacency does not imply
 scene adjacency in this corpus.

 A 64-bit difference hash alone is also unreliable here: night frames are largely dark, so at
 Hamming <= 6 it returns 82 pairs of which 60 are between different cities and therefore cannot
 be duplicates. The hash is used only as a cheap prefilter; a pair counts as a leak when it also
 exceeds PIXEL_R on downsampled pixel correlation."""
    splits = {}
    for c in CITIES:
        splits[f"{c}/train"] = _read(CFG / f"client_{c}_train.txt")
        splits[f"{c}/val"] = _read(CFG / f"client_{c}_val.txt")
    splits["global/train"] = _read(ROOT / "data/yolo/ecp_train.txt")
    splits["global/val"] = _read(ROOT / "data/yolo/ecp_val.txt")

    trains = {k: set(v) for k, v in splits.items() if k.endswith("/train")}
    vals = {k: set(v) for k, v in splits.items() if k.endswith("/val")}
    res = {"sizes": {k: len(v) for k, v in splits.items()}, "exact_overlaps": {},
           "near_duplicates": {}, "ham_thresh": ham_thresh}
    for tk, tv in trains.items():
        for vk, vvv in vals.items():
            n = len(tv & vvv)
            if n: res["exact_overlaps"][f"{tk} vs {vk}"] = n
    if near_dup:
        allp = sorted(set().union(*trains.values(), *vals.values()))
        H = {p: _dhash(p) for p in allp}
        T = sorted(set().union(*trains.values())); V = sorted(set().union(*vals.values()))
        hits, cand = [], []
        for v in V:
            hv = H[v]
            for t in T:
                if v == t: continue
                if bin(hv ^ H[t]).count("1") <= ham_thresh:
                    cand.append({"val": v, "train": t, "hamming": bin(hv ^ H[t]).count("1")})
        # stage 2: confirm each candidate on pixel content
        import numpy as _np
        from PIL import Image as _Im
        _cache = {}
        def _px(p, s=256):
            if p not in _cache:
                _cache[p] = _np.asarray(_Im.open(p).convert("L").resize((s, s)),
                                        dtype=_np.float32) / 255.
            return _cache[p]
        for h in cand:
            A, B = _px(h["val"]), _px(h["train"])
            h["pixel_r"] = float(_np.corrcoef(A.ravel(), B.ravel())[0, 1])
            if h["pixel_r"] > PIXEL_R:
                hits.append(h)
        res["hash_candidates"] = len(cand)
        res["max_pixel_r"] = max((h["pixel_r"] for h in cand), default=0.0)
        res["near_duplicates"] = hits
        res["n_val"] = len(V); res["n_train"] = len(T)
    res["pass"] = not res["exact_overlaps"] and not res["near_duplicates"]
    return res

if __name__ == "__main__":
    r = split_integrity()
    print("5. split integrity")
    for k, v in sorted(r["sizes"].items()): print(f"     {k:18s} {v:5d} images")
    print(f"   exact train/val overlaps : {r['exact_overlaps'] or 'none'}")
    nd = r["near_duplicates"]
    print(f"   hash prefilter candidates: {r.get('hash_candidates',0)} "
          f"(Hamming <= {r['ham_thresh']} of 64, over {r.get('n_train',0)}x{r.get('n_val',0)} pairs)")
    print(f"   max pixel correlation    : {r.get('max_pixel_r',0):+.3f} (threshold {PIXEL_R})")
    print(f"   confirmed duplicates     : {len(nd) if nd else 'none'}")
    for h in nd[:10]:
        print(f"       {Path(h['val']).name} ~ {Path(h['train']).name}  r={h['pixel_r']:+.3f}")
    print(f"   PASS                     : {r['pass']}")
    json.dump(r, open(ROOT / "project/work/results/contract_splits.json", "w"), indent=2)
