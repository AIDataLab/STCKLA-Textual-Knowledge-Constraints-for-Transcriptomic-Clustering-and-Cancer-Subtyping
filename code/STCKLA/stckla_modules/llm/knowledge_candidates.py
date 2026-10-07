"""llm.py definitions moved here without algorithm changes."""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
from ..logging_utils import diagnostic_print
from .expression import (
    make_expr_z,
)
from .utils import (
    gene_family_key,
    norm_gene,
    safe_float,
)


def collect_genes_from_obj(obj: Any, gene_set: Optional[set] = None) -> set:
    if gene_set is None:
        gene_set = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if lk in {"gene", "genes", "target", "targets", "regulator", "top_genes", "marker_genes", "members_uniprot", "members", "gene_symbols"}:
                if isinstance(v, (list, tuple)):
                    for x in v:
                        gx = norm_gene(x)
                        if gx:
                            gene_set.add(gx)
                else:
                    gx = norm_gene(v)
                    if gx:
                        gene_set.add(gx)
            else:
                collect_genes_from_obj(v, gene_set)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            collect_genes_from_obj(x, gene_set)
    return gene_set


def iter_jsonl(path: Path, max_records: int) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def source_bucket(ch: Dict[str, Any]) -> str:
    s = str(ch.get("source", "")).lower()
    if "trrust" in s:
        return "TRRUST"
    if "panglao" in s:
        return "PanglaoDB"
    if "gtex" in s:
        return "GTEx"
    if "reactome" in s:
        return "Reactome"
    if "msig" in s:
        return "MSigDB"
    if "chea" in s:
        return "ChEA"
    return str(ch.get("source", "Other")) or "Other"


def load_local_chunks(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.knowledge_path:
        path = Path(args.knowledge_path)
    else:
        path = Path(args.prior_root) / "qwen_rag" / "knowledge_chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"knowledge_chunks.jsonl not found: {path}")

    allowed = [x.strip().lower() for x in str(args.allowed_sources).split(",") if x.strip()]
    chunks = []
    for ch in iter_jsonl(path, int(args.max_knowledge_records)):
        src = str(ch.get("source", ""))
        if allowed:
            low_src = src.lower()
            if not any(a in low_src for a in allowed):
                continue
        genes = sorted(collect_genes_from_obj(ch.get("metadata", {})))
        if not genes:
            genes = sorted(collect_genes_from_obj(ch))
        ch["_genes"] = genes
        ch["_source_bucket"] = source_bucket(ch)
        ch["_search_text"] = f"{src} {ch.get('type','')} {ch.get('title','')} {ch.get('text','')}".upper()
        chunks.append(ch)
    print(f"[RETRIEVE] loaded local chunks={len(chunks)} from {path}", flush=True)
    return chunks


def compress_gene_evidence(candidate_genes: List[str], chunks: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    cset = {norm_gene(g) for g in candidate_genes if norm_gene(g)}
    evidence = {
        g: {
            "gene": g,
            "source_counts": Counter(),
            "titles": [],
            "total_hits": 0,
            "reactome_hits": 0,
            "trrust_hits": 0,
            "panglao_hits": 0,
            "gtex_hits": 0,
        } for g in cset
    }
    scored_chunks = []
    disease_terms = [t for t in re.split(r"[^A-Za-z0-9]+", str(args.disease_context).upper()) if len(t) >= 3]

    for ch in chunks:
        genes = {norm_gene(g) for g in ch.get("_genes", []) or []}
        overlap = sorted(cset & genes)
        if not overlap:
            continue
        bucket = source_bucket(ch)
        text = str(ch.get("_search_text", ""))
        disease_hit = sum(1 for t in disease_terms if t in text)
        density = len(overlap) / max(1, len(genes))
        score = 2.5 * len(overlap) + 1.0 * min(5, disease_hit) + 2.0 * density
        scored_chunks.append((score, bucket, overlap, ch))
        title = str(ch.get("title", "") or ch.get("type", ""))[:120]
        for g in overlap:
            ev = evidence[g]
            ev["source_counts"][bucket] += 1
            ev["total_hits"] += 1
            if bucket == "Reactome":
                ev["reactome_hits"] += 1
            elif bucket == "TRRUST":
                ev["trrust_hits"] += 1
            elif bucket == "PanglaoDB":
                ev["panglao_hits"] += 1
            elif bucket == "GTEx":
                ev["gtex_hits"] += 1
            if title and len(ev["titles"]) < 3:
                ev["titles"].append(title)

    for ev in evidence.values():
        ev["source_counts"] = dict(ev["source_counts"])
        ev["evidence_score"] = float(
            1.30 * ev["trrust_hits"]
            + 1.15 * ev["panglao_hits"]
            + 1.10 * ev["reactome_hits"]
            + 1.05 * ev["gtex_hits"]
            + 0.20 * ev["total_hits"]
        )

    # Balanced selected chunks for audit/debug only.
    scored_chunks.sort(key=lambda x: -x[0])
    selected = []
    bucket_counter = Counter()
    max_per_bucket = max(1, int(args.max_chunks_per_source))
    used_chars = 0
    for score, bucket, overlap, ch in scored_chunks:
        if bucket_counter[bucket] >= max_per_bucket:
            continue
        txt = str(ch.get("text", ""))
        add_chars = len(txt[: int(args.max_chars_per_chunk)])
        if selected and used_chars + add_chars > int(args.max_context_chars):
            break
        ch2 = dict(ch)
        ch2["_score"] = float(score)
        ch2["_source_bucket"] = bucket
        ch2["_overlap_genes"] = overlap[:30]
        selected.append(ch2)
        bucket_counter[bucket] += 1
        used_chars += add_chars
        if len(selected) >= int(args.max_chunks):
            break
    diagnostic_print(f"[RETRIEVE][QCMCE] selected_chunks={len(selected)} sources={dict(bucket_counter)}", flush=True)
    return evidence, selected


BROAD_OR_TECHNICAL_GENES = {
    "MKI67", "TOP2A", "PCNA", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7",
    "CCNB1", "CCNB2", "CDK1", "AURKA", "AURKB", "UBE2C", "BIRC5",
    "FOS", "JUN", "JUNB", "DUSP1", "ATF3", "EGR1", "IER2",
    "HSPA1A", "HSPA1B", "HSP90AA1", "HSP90AB1",
    "ACTB", "GAPDH", "B2M", "MALAT1", "TPT1", "EEF1A1",
}


def is_broad_or_technical(g: str) -> bool:
    g = norm_gene(g)
    return (
        g in BROAD_OR_TECHNICAL_GENES
        or re.match(r"^(RPL|RPS)\d", g) is not None
        or g.startswith("MT-")
        or re.match(r"^MT[A-Z0-9]+", g) is not None
    )


def select_top_genes_by_score(genes: List[str], scores: np.ndarray, n: int, positive: bool = True, family_cap: int = 5) -> List[str]:
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-scores if positive else scores)
    out = []
    fam_counter = Counter()
    for j in order:
        val = float(scores[j])
        if positive and val <= 0:
            continue
        if (not positive) and val >= 0:
            continue
        g = norm_gene(genes[j])
        if not g:
            continue
        fam = gene_family_key(g)
        if fam_counter[fam] >= int(family_cap):
            continue
        out.append(g)
        fam_counter[fam] += 1
        if len(out) >= int(n):
            break
    return out


def build_marker_candidate_bank(expr: pd.DataFrame, labels: pd.Series, exposed_mask: np.ndarray, subtype_map: Dict[int, str], chunks: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    ybase, z, mu, sd, transform_used = make_expr_z(expr, input_clip=float(args.factor_z_clip), transform=str(args.expr_transform))
    genes = [norm_gene(g) for g in expr.columns]
    gene_to_idx = {g: j for j, g in enumerate(genes)}
    y = labels.to_numpy(dtype=int)
    ex = np.asarray(exposed_mask, dtype=bool)
    var = z.var(axis=0)
    keep = var > 1e-8

    subtype_candidates = {}
    all_candidate_genes = set()
    class_effect_lookup = {}

    for c in sorted(set(y[ex].tolist())):
        cls = ex & (y == c)
        rest = ex & (y != c)
        min_exposed_per_class = int(getattr(args, "min_exposed_per_class", 3))
        if cls.sum() < min_exposed_per_class or rest.sum() < min_exposed_per_class:
            continue

        mean_c = z[cls].mean(axis=0)
        mean_r = z[rest].mean(axis=0)
        var_c = z[cls].var(axis=0)
        var_r = z[rest].var(axis=0)
        effect = mean_c - mean_r
        pooled = np.sqrt(0.5 * (var_c + var_r) + 1e-6)
        z_effect = (effect / pooled).astype(np.float32)
        z_effect = np.where(keep, z_effect, 0.0)
        class_effect_lookup[str(int(c))] = {genes[j]: float(z_effect[j]) for j in np.where(np.abs(z_effect) > 0)[0]}

        up = select_top_genes_by_score(genes, z_effect, int(args.candidate_genes_per_subtype), positive=True, family_cap=int(args.max_genes_per_family))
        down = select_top_genes_by_score(genes, z_effect, int(args.down_candidate_genes_per_subtype), positive=False, family_cap=int(args.max_genes_per_family))

        up_rows, down_rows = [], []
        for g in up:
            j = gene_to_idx[g]
            up_rows.append({
                "gene": g,
                "z_effect": float(z_effect[j]),
                "effect": float(effect[j]),
                "class_mean_z": float(mean_c[j]),
                "rest_mean_z": float(mean_r[j]),
                "broad_or_technical": bool(is_broad_or_technical(g)),
            })
            all_candidate_genes.add(g)
        for g in down:
            j = gene_to_idx[g]
            down_rows.append({
                "gene": g,
                "z_effect": float(z_effect[j]),
                "effect": float(effect[j]),
                "class_mean_z": float(mean_c[j]),
                "rest_mean_z": float(mean_r[j]),
                "broad_or_technical": bool(is_broad_or_technical(g)),
            })
            all_candidate_genes.add(g)

        subtype_candidates[str(int(c))] = {
            "class_id": int(c),
            "class_name": str(subtype_map.get(int(c), f"class_{int(c)}")),
            "exposed_class_n": int(cls.sum()),
            "exposed_rest_n": int(rest.sum()),
            "up_candidates": up_rows,
            "down_candidates": down_rows,
        }

    evidence, selected_chunks = compress_gene_evidence(sorted(all_candidate_genes), chunks, args)
    # Add evidence and redundancy.
    gene_class_hits = defaultdict(list)
    for cid, p in subtype_candidates.items():
        for r in p["up_candidates"]:
            gene_class_hits[r["gene"]].append(cid)

    for cid, p in subtype_candidates.items():
        for key in ["up_candidates", "down_candidates"]:
            for r in p[key]:
                ev = evidence.get(r["gene"], {})
                r["evidence_score"] = safe_float(ev.get("evidence_score", 0.0), 0.0)
                r["evidence_sources"] = ev.get("source_counts", {})
                r["evidence_titles"] = ev.get("titles", [])
                r["multi_class_up_count"] = len(set(gene_class_hits.get(r["gene"], [])))

    return {
        "z": z,
        "ybase": ybase,
        "mu": mu,
        "sd": sd,
        "expr_transform": transform_used,
        "gene_list": genes,
        "subtype_candidates": subtype_candidates,
        "gene_evidence": evidence,
        "selected_chunks": selected_chunks,
        "class_effect_lookup": class_effect_lookup,
        "label_protocol": {"exposed_n": int(ex.sum()), "use_exposed_labels": True},
    }
