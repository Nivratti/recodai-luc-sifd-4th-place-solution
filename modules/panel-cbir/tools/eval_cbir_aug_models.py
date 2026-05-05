#!/usr/bin/env python3
"""
eval_cbir_aug_models.py

Evaluate multiple CBIR models on augmentation-positive dataset:
- For each augmented variant (query), search among all ORIG images (candidates)
- Check whether the correct orig appears in Top-K => Recall@K
- Detailed report:
  - overall Recall@K, MRR, mean/median rank
  - per-transform breakdown (crop/rotate/jpeg/noise/...)
  - per-rotation degree, JPEG quality bins, noise sigma bins
  - common wrong top1 hits
  - failure visualization grids (optional)
- Resumable:
  - caches candidate embeddings per model
  - results.jsonl can be resumed (skip already processed queries)
  - per-model outputs stored under out_dir/<model_name>__<hash>/

Requires:
  - panel_cbir.py (your existing script) accessible as a module
  - pyyaml
  - numpy, torch, pillow, loguru, tqdm

Example:
  python eval_cbir_aug_models.py --config configs/eval_models.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from loguru import logger

# Reuse your existing CBIR code
import panel_cbir as cbir


# -----------------------------
# Utils
# -----------------------------
def _sha1_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha1_jsonable(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(s).hexdigest()


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _safe_rel_to(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _normalize_transform_name(name: str) -> str:
    # e.g. "noise(forced)" -> "noise"
    return name.split("(")[0].strip().lower()


def _bin_value(x: float, bins: Sequence[Tuple[str, float, float]]) -> str:
    for label, lo, hi in bins:
        if lo <= x < hi:
            return label
    return "other"


# -----------------------------
# Manifest parsing
# -----------------------------
def load_aug_manifest(aug_root: Path, manifest_name: str) -> Tuple[List[Dict[str, Any]], List[Path], List[Path]]:
    """
    Returns:
      records: list of dict per variant (query)
      orig_paths: unique candidate orig paths (absolute)
      variant_paths: query paths (absolute) aligned with records
    """
    manifest_path = Path(manifest_name)
    if not manifest_path.is_absolute():
        manifest_path = aug_root / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    records: List[Dict[str, Any]] = []
    orig_set: Dict[str, Path] = {}
    variant_paths: List[Path] = []

    n_total = 0
    n_ok = 0
    n_missing = 0

    for rec in _read_jsonl(manifest_path):
        n_total += 1
        if rec.get("kind") != "aug_positive":
            continue
        orig_rel = rec.get("orig_file")
        var_rel = rec.get("variant_file")
        if not orig_rel or not var_rel:
            continue

        orig_abs = aug_root / orig_rel
        var_abs = aug_root / var_rel

        if not orig_abs.exists() or not var_abs.exists():
            n_missing += 1
            continue

        # store
        orig_set[str(orig_rel)] = orig_abs
        records.append(rec)
        variant_paths.append(var_abs)
        n_ok += 1

    orig_paths = list(orig_set.values())
    orig_paths.sort()

    logger.info(f"Loaded manifest: {manifest_path}")
    logger.info(f"Records total={n_total}, usable_variants={n_ok}, missing_skipped={n_missing}")
    logger.info(f"Candidates (unique orig): {len(orig_paths)}")

    return records, orig_paths, variant_paths


# -----------------------------
# Embedding (reusing embedder internals)
# -----------------------------
def embed_paths_strict(
    embedder: cbir.BaseEmbedder,
    paths: Sequence[Path],
    batch_size: int,
    fp16: bool,
    desc: str,
) -> Tuple[List[Path], np.ndarray]:
    """
    Strict embed: returns (good_paths, embs) aligned.
    Reuses panel_cbir embedder._preprocess_one + model forward.
    """
    model = embedder.model
    assert model is not None
    model.eval()

    device = embedder.device
    bs = max(1, int(batch_size))

    good_paths: List[Path] = []
    all_embs: List[np.ndarray] = []

    # tqdm from panel_cbir is already available; reuse
    pbar = cbir.tqdm(range(0, len(paths), bs), desc=desc, unit="batch")
    for i in pbar:
        batch_paths = paths[i : i + bs]
        batch_imgs: List[cbir.torch.Tensor] = []
        batch_ok_paths: List[Path] = []

        for p in batch_paths:
            im = cbir._safe_load_image(p)
            if im is None:
                continue
            batch_imgs.append(embedder._preprocess_one(im))
            batch_ok_paths.append(p)

        if not batch_imgs:
            continue

        x = cbir.torch.stack(batch_imgs, dim=0).to(device, non_blocking=True)
        use_amp = bool(fp16 and device.type == "cuda")

        with cbir.torch.autocast(device_type="cuda", dtype=cbir.torch.float16, enabled=use_amp):
            with cbir.torch.inference_mode():
                out = model(x)
                feat = cbir._normalize_model_output(out)

        feat_np = feat.float().cpu().numpy().astype(np.float32)
        feat_np = cbir._l2_normalize(feat_np)

        good_paths.extend(batch_ok_paths)
        all_embs.append(feat_np)

    if not good_paths:
        return [], np.zeros((0, embedder.output_dim()), dtype=np.float32)

    embs = np.concatenate(all_embs, axis=0).astype(np.float32)
    return good_paths, embs


def load_or_build_candidate_cache(
    model_out: Path,
    dataset_id: str,
    model_id: str,
    cfg_resolved: cbir.CBIRConfig,
    candidate_paths: List[Path],
    resume: bool,
) -> Tuple[List[Path], np.ndarray]:
    """
    Cache files:
      candidates.npz  (paths, embs)
      candidates_meta.json (dataset_id, model_id, cfg_resolved hash)
    """
    npz_path = model_out / "candidates.npz"
    meta_path = model_out / "candidates_meta.json"

    cfg_hash = _sha1_jsonable(asdict(cfg_resolved))

    if resume and npz_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("dataset_id") == dataset_id and meta.get("model_id") == model_id and meta.get("cfg_hash") == cfg_hash:
            logger.info("Loading cached candidate embeddings...")
            data = np.load(npz_path, allow_pickle=True)
            paths = [Path(p) for p in data["paths"].tolist()]
            embs = data["embs"].astype(np.float32)
            logger.info(f"Loaded candidates cache: N={len(paths)}, D={embs.shape[1]}")
            return paths, embs
        else:
            logger.warning("Candidate cache exists but meta mismatch -> rebuilding cache")

    logger.info("Building candidate embeddings cache...")
    embedder = cbir.build_embedder(cfg_resolved)
    good_paths, embs = embed_paths_strict(
        embedder=embedder,
        paths=candidate_paths,
        batch_size=cfg_resolved.batch_size,
        fp16=cfg_resolved.fp16,
        desc="Embed candidates",
    )

    np.savez_compressed(npz_path, paths=np.array([str(p) for p in good_paths], dtype=object), embs=embs)
    _write_json(
        meta_path,
        {
            "dataset_id": dataset_id,
            "model_id": model_id,
            "cfg_hash": cfg_hash,
            "n_candidates": len(good_paths),
            "dim": int(embs.shape[1]) if embs.size else 0,
        },
    )
    logger.info(f"Saved candidates cache: {npz_path}")
    return good_paths, embs


# -----------------------------
# Evaluation
# -----------------------------
def eval_one_model(
    aug_root: Path,
    records: List[Dict[str, Any]],
    variant_paths: List[Path],
    candidate_paths: List[Path],
    candidate_embs: np.ndarray,
    cfg_resolved: cbir.CBIRConfig,
    model_out: Path,
    topk_list: List[int],
    store_topk: int,
    resume: bool,
    make_fail_viz: bool,
    fail_viz_k: int,
    fail_viz_thumb: int,
    max_fail_viz: int,
) -> Dict[str, Any]:
    """
    Writes:
      results.jsonl  (per query)
      summary.json
      breakdown_transforms.json
      breakdown_bins.json
      failures_viz/  (optional)
    """
    _ensure_dir(model_out)

    # Build candidate index mapping (path string -> idx)
    cand_idx: Dict[str, int] = {str(p): i for i, p in enumerate(candidate_paths)}

    # Prepare resume set
    results_path = model_out / "results.jsonl"
    done_queries: set[str] = set()
    if resume and results_path.exists():
        for r in _read_jsonl(results_path):
            q = r.get("query_abs")
            if q:
                done_queries.add(q)
        logger.info(f"Resume enabled: already have {len(done_queries)} queries in {results_path.name}")

    # Build embedder for queries
    embedder = cbir.build_embedder(cfg_resolved)

    max_k = max(topk_list) if topk_list else 10
    store_k = max(int(store_topk), max_k)

    failures_dir = model_out / "failures_viz"
    if make_fail_viz:
        _ensure_dir(failures_dir)

    # For failure viz, reuse PanelCBIR.save_viz_grid (doesn't require indexing)
    cbir_viz = cbir.PanelCBIR(cfg_resolved)

    n_processed = 0
    n_skipped = 0
    n_missing_gt = 0
    n_fail_viz_saved = 0

    t0 = time.time()

    # iterate variants in batches and append results
    bs = max(1, int(cfg_resolved.batch_size))
    pbar = cbir.tqdm(range(0, len(variant_paths), bs), desc="Eval queries", unit="batch")
    with open(results_path, "a", encoding="utf-8") as out_f:
        for bi in pbar:
            batch_vars = variant_paths[bi : bi + bs]
            batch_recs = records[bi : bi + bs]

            # skip already done
            todo_vars: List[Path] = []
            todo_recs: List[Dict[str, Any]] = []
            for vp, rc in zip(batch_vars, batch_recs):
                if str(vp) in done_queries:
                    n_skipped += 1
                    continue
                todo_vars.append(vp)
                todo_recs.append(rc)

            if not todo_vars:
                continue

            # embed this batch
            good_q_paths, q_embs = embed_paths_strict(
                embedder=embedder,
                paths=todo_vars,
                batch_size=len(todo_vars),  # embed in one mini-batch
                fp16=cfg_resolved.fp16,
                desc="",
            )
            if len(good_q_paths) == 0:
                continue

            # mapping from path->embedding row
            q_map = {str(p): q_embs[i] for i, p in enumerate(good_q_paths)}

            for vp, rc in zip(todo_vars, todo_recs):
                q_abs = str(vp)
                if q_abs not in q_map:
                    # unreadable query
                    continue

                orig_rel = rc["orig_file"]
                orig_abs = str(aug_root / orig_rel)
                if orig_abs not in cand_idx:
                    n_missing_gt += 1
                    continue

                qi = q_map[q_abs]  # (D,)
                scores = (candidate_embs @ qi).astype(np.float32)  # (N,)

                gt_i = cand_idx[orig_abs]
                gt_score = float(scores[gt_i])

                # exact rank without sorting everything:
                # rank = 1 + count(scores > gt_score)
                rank = int(1 + np.sum(scores > gt_score))

                # top1
                top1_i = int(np.argmax(scores))
                top1_path = str(candidate_paths[top1_i])
                top1_score = float(scores[top1_i])

                # store topK list (paths+scores) for report/viz
                k = min(store_k, scores.shape[0])
                idx = np.argpartition(-scores, kth=k - 1)[:k]
                idx = idx[np.argsort(-scores[idx])]
                topk_paths = [str(candidate_paths[int(j)]) for j in idx]
                topk_scores = [float(scores[int(j)]) for j in idx]

                # transform info
                transforms = rc.get("transforms", [])
                t_names = [_normalize_transform_name(t.get("name", "")) for t in transforms if t.get("name")]
                t_names = [t for t in t_names if t]

                # rotation degree / jpeg quality / noise sigma if present
                rot_deg = None
                jpeg_q = None
                noise_sigma = None
                for t in transforms:
                    name = _normalize_transform_name(t.get("name", ""))
                    params = (t.get("params") or {})
                    if name == "rotate" and "deg" in params:
                        rot_deg = params.get("deg")
                    if name == "jpeg" and "quality" in params:
                        jpeg_q = params.get("quality")
                    if name == "noise" and "sigma" in params:
                        noise_sigma = params.get("sigma")

                rec_out = {
                    "query_abs": q_abs,
                    "query_rel": _safe_rel_to(vp, aug_root),
                    "orig_abs": orig_abs,
                    "orig_rel": orig_rel,
                    "rank": rank,
                    "gt_score": gt_score,
                    "top1_abs": top1_path,
                    "top1_score": top1_score,
                    "is_top1": bool(top1_path == orig_abs),
                    "transform_names": t_names,
                    "rot_deg": rot_deg,
                    "jpeg_q": jpeg_q,
                    "noise_sigma": noise_sigma,
                    "topk_abs": topk_paths[:store_k],
                    "topk_scores": topk_scores[:store_k],
                }
                out_f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                n_processed += 1

                # optional fail viz: save only for failures (gt not in top fail_viz_k)
                if make_fail_viz and n_fail_viz_saved < max_fail_viz:
                    if rank > fail_viz_k:
                        # build Match list for viz
                        matches = []
                        for rr, (pp, sc) in enumerate(zip(topk_paths[:fail_viz_k], topk_scores[:fail_viz_k]), start=1):
                            matches.append(cbir.Match(path=pp, score=float(sc), rank=rr))
                        out_img = failures_dir / f"{Path(q_abs).stem}__rank{rank}.png"
                        cbir_viz.save_viz_grid(
                            query_path=q_abs,
                            matches=matches,
                            out_path=out_img,
                            thumb=fail_viz_thumb,
                            max_k=fail_viz_k,
                            title=f"rank={rank}  top1={Path(top1_path).name}",
                        )
                        n_fail_viz_saved += 1

    t1 = time.time()

    logger.info(f"Eval done: processed={n_processed}, skipped={n_skipped}, missing_gt={n_missing_gt}")
    logger.info(f"Results: {results_path}")
    logger.info(f"Time: {t1 - t0:.1f}s")

    # build summary + breakdown from results.jsonl (single source of truth, supports resume)
    all_rows = list(_read_jsonl(results_path))
    summary, breakdown_transforms, breakdown_bins = analyze_results(all_rows, topk_list)

    summary["n_processed_this_run"] = n_processed
    summary["n_skipped_this_run"] = n_skipped
    summary["n_missing_gt_this_run"] = n_missing_gt
    summary["runtime_seconds"] = round(t1 - t0, 3)
    summary["cfg_resolved"] = asdict(cfg_resolved)

    _write_json(model_out / "summary.json", summary)
    _write_json(model_out / "breakdown_transforms.json", breakdown_transforms)
    _write_json(model_out / "breakdown_bins.json", breakdown_bins)

    return summary


def analyze_results(rows: List[Dict[str, Any]], topk_list: List[int]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if not rows:
        return {"n": 0}, {}, {}

    ranks = np.array([int(r["rank"]) for r in rows], dtype=np.int32)

    def recall_at(k: int) -> float:
        if k <= 0:
            return 0.0
        return float(np.mean(ranks <= k))

    # MRR
    mrr = float(np.mean(1.0 / np.maximum(ranks, 1)))

    summary = {
        "n": int(len(rows)),
        "mrr": mrr,
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "p95_rank": float(np.percentile(ranks, 95)),
    }
    for k in topk_list:
        summary[f"recall@{k}"] = recall_at(int(k))

    # common wrong top1s
    top1_wrong = [r["top1_abs"] for r in rows if not r.get("is_top1", False)]
    if top1_wrong:
        counts: Dict[str, int] = {}
        for p in top1_wrong:
            counts[p] = counts.get(p, 0) + 1
        top10 = sorted(counts.items(), key=lambda x: -x[1])[:10]
        summary["top1_wrong_top10"] = [{"path": p, "count": c} for p, c in top10]
    else:
        summary["top1_wrong_top10"] = []

    # per-transform breakdown
    # A query can have multiple transforms; we count it in each transform bucket it contains
    buckets: Dict[str, List[int]] = {}
    for r in rows:
        tnames = r.get("transform_names") or []
        tset = set(tnames)
        for t in tset:
            buckets.setdefault(t, []).append(int(r["rank"]))
        # also bucket by number of transforms
        buckets.setdefault(f"n_transforms={len(tset)}", []).append(int(r["rank"]))

    breakdown_transforms: Dict[str, Any] = {}
    for name, branks in sorted(buckets.items(), key=lambda x: x[0]):
        br = np.array(branks, dtype=np.int32)
        bsum = {"n": int(len(br)), "mrr": float(np.mean(1.0 / np.maximum(br, 1))), "mean_rank": float(np.mean(br))}
        for k in topk_list:
            bsum[f"recall@{k}"] = float(np.mean(br <= int(k)))
        breakdown_transforms[name] = bsum

    # bins for rotation / jpeg / noise
    rot_bins: Dict[str, List[int]] = {}
    jpeg_bins: Dict[str, List[int]] = {}
    noise_bins: Dict[str, List[int]] = {}

    jpeg_def = [("q<50", 0, 50), ("q50-70", 50, 70), ("q70-90", 70, 90), ("q>=90", 90, 10_000)]
    noise_def = [("s<2", 0, 2), ("s2-4", 2, 4), ("s4-8", 4, 8), ("s>=8", 8, 10_000)]

    for r in rows:
        rk = int(r["rank"])
        deg = r.get("rot_deg", None)
        if deg is not None:
            rot_bins.setdefault(f"rot={deg}", []).append(rk)
        q = r.get("jpeg_q", None)
        if q is not None:
            jpeg_bins.setdefault(_bin_value(float(q), jpeg_def), []).append(rk)
        s = r.get("noise_sigma", None)
        if s is not None:
            noise_bins.setdefault(_bin_value(float(s), noise_def), []).append(rk)

    def pack_bins(d: Dict[str, List[int]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, rr in sorted(d.items(), key=lambda x: x[0]):
            br = np.array(rr, dtype=np.int32)
            out[k] = {"n": int(len(br)), "mean_rank": float(np.mean(br))}
            for tk in topk_list:
                out[k][f"recall@{tk}"] = float(np.mean(br <= int(tk)))
        return out

    breakdown_bins = {
        "rotation": pack_bins(rot_bins),
        "jpeg_quality": pack_bins(jpeg_bins),
        "noise_sigma": pack_bins(noise_bins),
    }

    return summary, breakdown_transforms, breakdown_bins


# -----------------------------
# Report writing
# -----------------------------
def write_global_report(out_dir: Path, model_summaries: List[Dict[str, Any]]) -> None:
    # comparison markdown
    lines: List[str] = []
    lines.append("# CBIR Augmentation-Positive Evaluation Report\n")
    lines.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # build table
    keys = ["name", "n", "mrr", "mean_rank", "median_rank", "p95_rank"]
    # add recall columns found in first summary
    recall_keys = sorted([k for k in model_summaries[0].keys() if k.startswith("recall@")], key=lambda x: int(x.split("@")[1])) if model_summaries else []
    keys += recall_keys

    # header
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("|" + "|".join(["---"] * len(keys)) + "|")

    for s in model_summaries:
        row = []
        for k in keys:
            v = s.get(k, "")
            if isinstance(v, float):
                row.append(f"{v:.4f}")
            else:
                row.append(str(v))
        lines.append("| " + " | ".join(row) + " |")

    # highlight wrong-top1 patterns
    lines.append("\n## Common wrong Top-1 predictions per model\n")
    for s in model_summaries:
        lines.append(f"### {s.get('name')}\n")
        top10 = s.get("top1_wrong_top10") or []
        if not top10:
            lines.append("- No wrong top-1 cases (or none recorded).\n")
            continue
        for item in top10:
            lines.append(f"- {item['count']}×  {Path(item['path']).name}\n")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote global report: {out_dir / 'report.md'}")


# -----------------------------
# Config loading / CLI
# -----------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="eval-cbir-aug-models",
        description="Evaluate multiple CBIR models on augmentation-positive dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, required=True, help="YAML config path.")
    return p.parse_args(argv)


def build_cfg_from_yaml(defaults: Dict[str, Any], model: Dict[str, Any]) -> cbir.CBIRConfig:
    """
    Only timm + sscd are intended (torchhub is allowed by panel_cbir but you can avoid it in YAML).
    """
    d = dict(defaults)
    d.update(model)

    # Map YAML keys to CBIRConfig fields
    cfg = cbir.CBIRConfig(
        img_size=d.get("img_size", None),
        resize_mode=d.get("resize_mode", "letterpad"),
        grayscale=bool(d.get("grayscale", False)),
        device=d.get("device", "cpu"),
        batch_size=int(d.get("batch_size", 32)),
        fp16=bool(d.get("fp16", False)),
        seed=int(d.get("seed", 0)) if d.get("seed", 0) is not None else 0,
        deterministic=bool(d.get("deterministic", True)),
        mean=tuple(d["mean"]) if d.get("mean") is not None else None,
        std=tuple(d["std"]) if d.get("std") is not None else None,
        backend=d.get("backend", "timm"),
        model_name=d.get("model_name", d.get("model", "resnet50")),
        hub_repo=d.get("hub_repo", "pytorch/vision"),
        hub_source=d.get("hub_source", "github"),
        sscd_torchscript_path=d.get("sscd_torchscript_path", d.get("sscd_torchscript_path", None)),
    )
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg_path = Path(args.config)
    cfg_y = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    dataset_cfg = cfg_y.get("dataset", {})
    eval_cfg = cfg_y.get("eval", {})
    defaults = cfg_y.get("defaults", {})
    out_cfg = cfg_y.get("outputs", {})
    models = cfg_y.get("models", [])

    if not models:
        logger.error("No models defined in YAML under 'models:'")
        return 2

    aug_root = Path(dataset_cfg.get("aug_root", "")).expanduser()
    if not aug_root.exists():
        logger.error(f"aug_root not found: {aug_root}")
        return 2

    manifest_name = dataset_cfg.get("manifest", "manifest.jsonl")
    records, orig_paths, var_paths = load_aug_manifest(aug_root, manifest_name)

    # dataset id = sha1(manifest file)
    manifest_path = Path(manifest_name)
    if not manifest_path.is_absolute():
        manifest_path = aug_root / manifest_name
    dataset_id = _sha1_file(manifest_path)

    # eval params
    topk_list = [int(x) for x in eval_cfg.get("topk_list", [1, 5, 10])]
    store_topk = int(eval_cfg.get("store_topk", max(topk_list)))
    resume = bool(eval_cfg.get("resume", True))
    make_fail_viz = bool(eval_cfg.get("make_fail_viz", False))
    fail_viz_k = int(eval_cfg.get("fail_viz_k", 6))
    fail_viz_thumb = int(eval_cfg.get("fail_viz_thumb", 224))
    max_fail_viz = int(eval_cfg.get("max_fail_viz", 200))

    out_dir = Path(out_cfg.get("out_dir", "out/cbir_aug_eval")).expanduser()
    _ensure_dir(out_dir)
    _write_json(out_dir / "run_config_resolved.json", cfg_y)

    logger.info(f"Output dir: {out_dir}")
    logger.info(f"Dataset id: {dataset_id[:12]} (sha1(manifest))")

    model_summaries: List[Dict[str, Any]] = []

    for m in models:
        name = m.get("name") or f"{m.get('backend','timm')}_{m.get('model_name', m.get('model','model'))}"
        cfg_resolved = build_cfg_from_yaml(defaults, m)

        # model id hash for output folder
        model_id = _sha1_jsonable({"name": name, "cfg": asdict(cfg_resolved)})
        model_out = out_dir / f"{name}__{model_id[:10]}"
        _ensure_dir(model_out)

        # save resolved config
        _write_json(model_out / "config_resolved.json", {"name": name, "model_id": model_id, "dataset_id": dataset_id, "cfg": asdict(cfg_resolved)})

        logger.info(f"\n=== Model: {name} ===")
        logger.info(f"backend={cfg_resolved.backend}, model_name={cfg_resolved.model_name}, device={cfg_resolved.device}, bs={cfg_resolved.batch_size}, fp16={cfg_resolved.fp16}")

        # Candidate embeddings cache (orig only)
        cand_paths_good, cand_embs = load_or_build_candidate_cache(
            model_out=model_out,
            dataset_id=dataset_id,
            model_id=model_id,
            cfg_resolved=cfg_resolved,
            candidate_paths=orig_paths,
            resume=resume,
        )

        # Evaluate (resumable results.jsonl)
        summary = eval_one_model(
            aug_root=aug_root,
            records=records,
            variant_paths=var_paths,
            candidate_paths=cand_paths_good,
            candidate_embs=cand_embs,
            cfg_resolved=cfg_resolved,
            model_out=model_out,
            topk_list=topk_list,
            store_topk=store_topk,
            resume=resume,
            make_fail_viz=make_fail_viz,
            fail_viz_k=fail_viz_k,
            fail_viz_thumb=fail_viz_thumb,
            max_fail_viz=max_fail_viz,
        )
        summary["name"] = name
        summary["model_out"] = str(model_out)
        model_summaries.append(summary)

    # Global report
    _write_json(out_dir / "all_model_summaries.json", model_summaries)
    if model_summaries:
        write_global_report(out_dir, model_summaries)

    logger.info("All done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
