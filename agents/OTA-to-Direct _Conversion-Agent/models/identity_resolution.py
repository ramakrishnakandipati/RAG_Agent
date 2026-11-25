"""Identity resolution: masked OTA guest -> CDP profile.

Pipeline: (1) deterministic real-email match; (2) FAISS nearest-neighbour candidate blocking on hashed
char-trigram name embeddings; (3) GPU (torch/CUDA) cosine + Jaro-Winkler rerank with phone-last4 / country evidence;
(4) conservative acceptance rules so false-positive merges stay <2% (Gate D2).
"""
from __future__ import annotations
import sys, time, zlib
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import get_settings, detect_hardware
from data.synthetic_generator import NICK

DIM = 512
K = 20
NICK_INV = {v.lower(): k.lower() for k, v in NICK.items()}


def _norm(s) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalpha())


def _embed(first, last) -> np.ndarray:
    """Hashed char-trigram embedding; last name weighted 2x. L2-normalised float32."""
    out = np.zeros((len(first), DIM), dtype=np.float32)
    for i, (f, l) in enumerate(zip(first, last)):
        for token, w in ((f"^{_norm(l)}$", 2.0), (f"^{_norm(NICK_INV.get(_norm(f), _norm(f)))}$", 1.0)):
            for j in range(len(token) - 2):
                out[i, zlib.crc32(token[j:j + 3].encode()) % DIM] += w
    n = np.linalg.norm(out, axis=1, keepdims=True); n[n == 0] = 1
    return out / n


def jaro_winkler(a: str, b: str, p: float = 0.1) -> float:
    a, b = _norm(a), _norm(b)
    if a == b: return 1.0 if a else 0.0
    la, lb = len(a), len(b)
    if not la or not lb: return 0.0
    md = max(la, lb) // 2 - 1
    ma, mb = [False] * la, [False] * lb; m = 0
    for i in range(la):
        for j in range(max(0, i - md), min(lb, i + md + 1)):
            if not mb[j] and a[i] == b[j]:
                ma[i] = mb[j] = True; m += 1; break
    if not m: return 0.0
    t = 0; k = 0
    for i in range(la):
        if ma[i]:
            while not mb[k]: k += 1
            if a[i] != b[k]: t += 1
            k += 1
    jaro = (m / la + m / lb + (m - t / 2) / m) / 3
    pre = 0
    for x, y in zip(a[:4], b[:4]):
        if x != y: break
        pre += 1
    return jaro + pre * p * (1 - jaro)


def _first_sim(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    a, b = NICK_INV.get(a, a), NICK_INV.get(b, b)
    return jaro_winkler(a, b)


def _build_index(emb: np.ndarray, use_gpu: bool):
    import faiss
    idx = faiss.IndexFlatIP(emb.shape[1])
    backend = "faiss-cpu"
    if use_gpu and hasattr(faiss, "StandardGpuResources"):
        try:
            idx = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, idx); backend = "faiss-gpu"
        except Exception:
            pass
    idx.add(emb)
    return idx, backend


def prepare(cdp: pd.DataFrame, dev: str | None = None) -> dict:
    """Pre-compute CDP embeddings + FAISS index once (reused by the online agent)."""
    dev = dev or ("cuda" if detect_hardware()["cuda"] else "cpu")
    e = _embed(cdp.first_name, cdp.last_name); idx, backend = _build_index(e, dev == "cuda")
    return {"e_cdp": e, "index": idx, "backend": backend, "dev": dev}


def resolve(bookings: pd.DataFrame, cdp: pd.DataFrame, force_device: str | None = None, cache: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Returns (bookings + resolved_profile_id/method/confidence, telemetry)."""
    import torch
    hw = detect_hardware(); dev = force_device or (cache["dev"] if cache else ("cuda" if hw["cuda"] else "cpu"))
    t0 = time.perf_counter()
    cdp = cdp.reset_index(drop=True)
    if cache is None: cache = prepare(cdp, dev)
    e_cdp, index, backend = cache["e_cdp"], cache["index"], cache["backend"]
    e_q = _embed(bookings.first_name, bookings.last_name)
    t_embed = time.perf_counter() - t0

    t1 = time.perf_counter()
    _, cand = index.search(e_q, K)                                # (n, K) blocking
    t_block = time.perf_counter() - t1

    t2 = time.perf_counter()
    td = torch.device(dev)
    Ec = torch.from_numpy(e_cdp).to(td); Eq = torch.from_numpy(e_q).to(td)
    cand_t = torch.from_numpy(cand).to(td)
    cos = (Eq[:, None, :] * Ec[cand_t]).sum(-1)                  # (n, K) GPU cosine
    if dev == "cuda": torch.cuda.synchronize()
    cos = cos.cpu().numpy(); t_cos = time.perf_counter() - t2

    n = len(bookings)
    res_id = np.full(n, None, dtype=object); method = np.array(["unmatched"] * n, dtype=object); conf = np.zeros(n)
    prof_by_email = dict(zip(cdp.email.str.lower(), cdp.profile_id))
    cdp_ids = cdp.profile_id.values; cdp_l4 = cdp.phone.astype(str).str[-4:].values
    cdp_last = cdp.last_name.values; cdp_first = cdp.first_name.values; cdp_ctry = cdp.country.values
    b_email = bookings.masked_email.str.lower().values; b_l4 = bookings.phone_last4.values
    b_last = bookings.last_name.values; b_first = bookings.first_name.values; b_ctry = bookings.country.values

    t3 = time.perf_counter()
    for i in range(n):
        pid = prof_by_email.get(b_email[i])
        if pid is not None:                                       # deterministic
            res_id[i], method[i], conf[i] = pid, "deterministic_email", 1.0; continue
        has_l4 = isinstance(b_l4[i], str) and b_l4[i] != ""
        best, best_s, second_s = None, 0.0, 0.0
        for k in range(K):
            c = cand[i, k]
            if c < 0 or cos[i, k] < 0.45: continue
            if has_l4 and cdp_l4[c] != b_l4[i]: continue
            jl = jaro_winkler(b_last[i], cdp_last[c]); jf = _first_sim(b_first[i], cdp_first[c])
            if jl < 0.88 or jf < 0.80: continue
            score = 0.35 * cos[i, k] + 0.30 * jl + 0.15 * jf + 0.10 * (b_ctry[i] == cdp_ctry[c]) + (0.10 if has_l4 else 0.0)
            if score > best_s: best, second_s, best_s = c, best_s, score
            elif score > second_s: second_s = score
        if best is None: continue
        if has_l4:
            res_id[i], method[i], conf[i] = cdp_ids[best], "fuzzy_name+phone4", best_s
        elif (_norm(b_last[i]) == _norm(cdp_last[best]) and b_ctry[i] == cdp_ctry[best]
              and _first_sim(b_first[i], cdp_first[best]) >= 0.99 and best_s - second_s > 0.15):
            res_id[i], method[i], conf[i] = cdp_ids[best], "fuzzy_name_only_unique", best_s   # no phone: very conservative
    t_rerank = time.perf_counter() - t3
    out = bookings.copy()
    out["resolved_profile_id"], out["resolution_method"], out["resolution_confidence"] = res_id, method, conf.round(4)
    tel = {"device": dev, "faiss_backend": backend, "n_queries": n, "n_cdp": len(cdp),
           "t_embed_s": round(t_embed, 3), "t_block_s": round(t_block, 3), "t_gpu_cosine_s": round(t_cos, 4), "t_rerank_s": round(t_rerank, 3),
           "t_total_s": round(time.perf_counter() - t0, 3)}
    return out, tel


def evaluate(resolved: pd.DataFrame) -> dict:
    truth = resolved.true_profile_id; pred = resolved.resolved_profile_id
    linked = truth.notna(); merged = pred.notna()
    correct = merged & linked & (pred == truth)
    recall = float(correct.sum() / max(linked.sum(), 1))
    wrong = merged & ~correct
    fp_rate = float(wrong.sum() / max(merged.sum(), 1))
    return {"recall": round(recall, 4), "false_positive_merge_rate": round(fp_rate, 4), "match_rate": round(float(merged.mean()), 4),
            "n_linked_truth": int(linked.sum()), "n_merged": int(merged.sum()), "n_wrong_merges": int(wrong.sum()),
            "by_method": resolved.resolution_method.value_counts().to_dict()}


def gate_d2(resolved: pd.DataFrame) -> dict:
    m = evaluate(resolved)
    checks = [{"check": "recall_gt_90pct", "passed": m["recall"] > 0.90, "detail": m["recall"]},
              {"check": "false_positive_merge_lt_2pct", "passed": m["false_positive_merge_rate"] < 0.02, "detail": m["false_positive_merge_rate"]}]
    return {"gate": "D2_identity_resolution", "passed": all(c["passed"] for c in checks), "metrics": m, "checks": checks}


if __name__ == "__main__":
    from data.synthetic_generator import load
    b, c = load(); r, tel = resolve(b, c); print(tel); print(gate_d2(r))
