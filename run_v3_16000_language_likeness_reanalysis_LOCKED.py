#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Locked language-likeness reanalysis for V3 16000 language-acquisition runs.

Purpose
-------
This script does NOT treat cross-population L7/L7b transfer as a gate.
It asks whether the acquired within-population signal system is language-like:
  1. sender grounding
  2. receiver action effect
  3. non-reflexive / context-dependent response
  4. internal grounding beyond external labels
  5. grammar-like sequence/order/ngram structure beyond unigram/shuffle nulls
  6. functional relevance to action/outcome
  7. within-population sharing / codebook convergence
  8. temporal development / stabilization over 0-1000, 1000-2000, 2000-4000, 4000-8000, 8000-16000

Design choices made to avoid prior failures
-------------------------------------------
- Always creates outdir/desktop dirs before any write.
- Discovers input files recursively; does not assume one exact folder layout.
- Supports column aliases: seed/seed_id/target_seed_id, comm/communication_condition, etc.
- Handles receiver_signal_action_log.csv and receiver_action_log.csv.
- Never calls pd.concat on an empty list.
- Writes every expected CSV even when rows=0, with explicit status/reason columns.
- Writes a final Markdown report unconditionally.
- Uses fixed bins, fixed metrics, fixed contrasts, and fixed interpretation classes.
- Does not use language_like_level, maxL, summed level score, or cross-pop transfer as primary evidence.
- Uses deterministic null/permutation with a fixed seed for reproducibility.

Tested modes
------------
  python3 -m py_compile <this_file>
  python3 <this_file> --help
  python3 <this_file> --estimate-only
  python3 <this_file> --mock-self-test --mock-outdir /tmp/mock_langlike
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# -----------------------------
# Locked design constants
# -----------------------------
SCENARIOS = ["baseline_3d", "unknown_x1p50", "danger_x1p25", "danger_x1p50", "vertical_x1p50"]
TASKS = ["exploration_recovery", "physics_adaptation", "social_reappraisal"]
COMMS = ["PRIVATE", "RANDOM_MATCHED", "SHUFFLED", "RECEIVER_LEARN", "FULL_INTERACTIVE"]
ACTIVE_COMMS = ["RECEIVER_LEARN", "FULL_INTERACTIVE"]
CONTROL_COMMS = ["PRIVATE", "RANDOM_MATCHED", "SHUFFLED"]
TIME_BINS = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]
TIME_BIN_LABELS = [f"{a}_{b}" for a, b in TIME_BINS]
RNG_SEED = 20260619

EXPECTED_OUTPUTS = [
    "00_LANGUAGE_ACQUISITION_AUDIT.csv",
    "01_development_by_time_bin.csv",
    "02_sender_grounding_by_condition.csv",
    "03_receiver_action_effects_by_action.csv",
    "04_nonreflexive_context_dependence.csv",
    "05_internal_grounding_vs_external_label.csv",
    "06_sequence_order_ngram_null_test.csv",
    "07_bigram_trigram_action_prediction.csv",
    "08_sequence_to_outcome_relevance.csv",
    "09_shared_codebook_within_population.csv",
    "10_language_likeness_judgement.csv",
    "11_language_acquisition_final_report.md",
]

# File families. More names are included than necessary because older code variants used slightly different names.
FAMILY_PATTERNS = {
    "run_index": ["run_index.csv"],
    "population_episode": ["language_population_episode_summary.csv", "population_episode_summary.csv"],
    "agent_episode": ["language_agent_episode_summary.csv", "agent_episode_summary.csv"],
    "signal_event": ["signal_event_log.csv"],
    "signal_delivery": ["signal_delivery_log.csv"],
    "receiver_action": ["receiver_signal_action_log.csv", "receiver_action_log.csv"],
    "utterance_sequence": ["utterance_sequence_log.csv"],
    "state_perturbation": ["state_perturbation_reflex_assay.csv", "state_perturbation_reflex_log.csv"],
    "memory": ["signal_memory_snapshot.csv", "memory_log.csv", "channel_memory_log.csv"],
    "primary_criteria": ["primary_language_criteria_summary.csv", "10_PRIMARY_LANGUAGE_CRITERIA__L1_to_L7.csv"],
}

ALIASES = {
    "scenario": ["scenario", "scenario_name", "env", "environment", "condition_scenario"],
    "task": ["task", "task_name", "condition_task"],
    "comm": ["comm", "communication_condition", "communication", "communication_mode", "target_comm", "source_comm"],
    "seed": ["seed", "seed_id", "target_seed_id", "source_seed_id", "replicate", "run_seed"],
    "agent": ["agent", "agent_id", "sender_id", "receiver_id"],
    "sender_id": ["sender_id", "sender", "source_agent_id"],
    "receiver_id": ["receiver_id", "receiver", "target_agent_id"],
    "step": ["step", "emitted_step", "delivery_step", "action_step", "receiver_step", "time", "t", "tick"],
    "emitted_step": ["emitted_step", "step", "time", "t"],
    "delivery_step": ["delivery_step", "step", "time", "t"],
    "channel": ["channel", "raw_channel", "signal_channel", "dominant_channel", "last_channel"],
    "raw_channel": ["raw_channel", "channel"],
    "action": ["receiver_action", "action", "sender_action", "base_action_after_signal_bias", "assay_action"],
    "action_class": ["receiver_action_class", "action_class", "assay_action_class", "receiver_event"],
    "sequence": ["sequence", "utterance", "channel_sequence", "ngram_sequence"],
    "sequence_length": ["sequence_length", "seq_len", "length"],
}

SENDER_GROUNDING_METRICS = [
    "sender_h", "sender_Q", "sender_Q_R", "sender_Q_G", "sender_physics_score",
    "sender_relief", "sender_safe_surprise", "sender_self_appraisal_gap", "sender_q_relief",
    "sender_danger_pressure", "sender_resource_pressure", "sender_unknown_pressure", "sender_vertical_pressure", "sender_friction_pressure",
    "composite_danger_unknown", "composite_danger_vertical", "composite_resource_unknown", "composite_safe_recovery",
]
RECEIVER_CONTEXT_METRICS = [
    "receiver_danger_pressure", "receiver_resource_pressure", "receiver_unknown_pressure", "receiver_vertical_pressure",
    "sender_danger_pressure", "sender_resource_pressure", "sender_unknown_pressure", "sender_vertical_pressure",
    "receiver_Q", "receiver_body_h", "sender_h", "sender_Q",
]
OUTCOME_METRICS = [
    "receiver_delta_h", "receiver_damage", "receiver_resource_gain", "receiver_recovery_gain",
    "receiver_entered_unknown", "receiver_vertical", "receiver_embodied_value", "receiver_Q", "receiver_body_h",
    "agent_mean_final_body_h", "agent_mean_total_damage", "mean_receiver_embodied_value",
]
POSITIVE_ORIENTATION = {
    "receiver_damage": -1.0,
    "agent_mean_total_damage": -1.0,
    "damage": -1.0,
    "mean_damage": -1.0,
}

# -----------------------------
# General utilities
# -----------------------------

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_unlink_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    ensure_dir(path)


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def normalize_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def entropy_norm(values: Sequence[Any]) -> float:
    vals = [v for v in values if not pd.isna(v)]
    if not vals:
        return np.nan
    c = Counter(vals)
    n = sum(c.values())
    if n <= 0 or len(c) <= 1:
        return 0.0
    ent = -sum((v / n) * math.log(v / n + 1e-15) for v in c.values())
    return ent / math.log(len(c))


def top_share(values: Sequence[Any]) -> float:
    vals = [v for v in values if not pd.isna(v)]
    if not vals:
        return np.nan
    c = Counter(vals)
    return max(c.values()) / max(1, sum(c.values()))


def unique_count(values: Sequence[Any]) -> int:
    return len(set([v for v in values if not pd.isna(v)]))


def bh_fdr(p_values: Sequence[float]) -> List[float]:
    p = np.array([np.nan if v is None else float(v) for v in p_values], dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return q.tolist()
    order = valid[np.argsort(p[valid])]
    m = len(order)
    prev = 1.0
    for rank_from_end, idx in enumerate(order[::-1], start=1):
        rank = m - rank_from_end + 1
        val = min(prev, p[idx] * m / max(rank, 1))
        q[idx] = min(val, 1.0)
        prev = q[idx]
    return q.tolist()


def spearman_safe(x: Sequence[float], y: Sequence[float]) -> float:
    try:
        a = pd.to_numeric(pd.Series(x), errors="coerce")
        b = pd.to_numeric(pd.Series(y), errors="coerce")
        mask = a.notna() & b.notna()
        if int(mask.sum()) < 3:
            return np.nan
        if a[mask].nunique() < 2 or b[mask].nunique() < 2:
            return np.nan
        return float(a[mask].rank().corr(b[mask].rank()))
    except Exception:
        return np.nan


def pearson_safe(x: Sequence[float], y: Sequence[float]) -> float:
    try:
        a = pd.to_numeric(pd.Series(x), errors="coerce")
        b = pd.to_numeric(pd.Series(y), errors="coerce")
        mask = a.notna() & b.notna()
        if int(mask.sum()) < 3:
            return np.nan
        if a[mask].nunique() < 2 or b[mask].nunique() < 2:
            return np.nan
        return float(a[mask].corr(b[mask]))
    except Exception:
        return np.nan


def eta_squared_categorical(groups: Sequence[Any], values: Sequence[float]) -> float:
    df = pd.DataFrame({"g": groups, "v": pd.to_numeric(pd.Series(values), errors="coerce")}).dropna()
    if df.empty or df["g"].nunique() < 2 or df["v"].nunique() < 2:
        return np.nan
    grand = df["v"].mean()
    ss_total = float(((df["v"] - grand) ** 2).sum())
    if ss_total <= 0:
        return np.nan
    ss_between = 0.0
    for _, sub in df.groupby("g", dropna=False):
        ss_between += len(sub) * float((sub["v"].mean() - grand) ** 2)
    return float(max(0.0, min(1.0, ss_between / ss_total)))


def fisher_p_from_r(r: float, n: int) -> float:
    if not np.isfinite(r) or n < 4:
        return np.nan
    r = max(-0.999999, min(0.999999, float(r)))
    z = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(max(1, n - 3))
    # two-sided normal approximation without scipy
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return max(0.0, min(1.0, p))


def binomial_sign_p(pos: int, n: int, p0: float = 0.5) -> float:
    if n <= 0:
        return np.nan
    # two-sided exact-ish by summing tails for p0=.5, adequate for small seed n.
    from math import comb
    prob = 0.0
    k = pos
    # probability of outcomes as or more extreme than observed around n*p0
    mean = n * p0
    observed_dev = abs(k - mean)
    for i in range(n + 1):
        if abs(i - mean) >= observed_dev - 1e-12:
            prob += comb(n, i) * (p0 ** i) * ((1 - p0) ** (n - i))
    return max(0.0, min(1.0, prob))


def time_bin_for_step(step: Any) -> str:
    s = safe_float(step)
    if not np.isfinite(s):
        return "unknown"
    for a, b in TIME_BINS:
        if s >= a and s < b:
            return f"{a}_{b}"
    if s >= TIME_BINS[-1][1]:
        return f">={TIME_BINS[-1][1]}"
    return "unknown"


def find_alias(df: pd.DataFrame, canonical: str) -> Optional[str]:
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for cand in ALIASES.get(canonical, [canonical]):
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def add_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    out = df.copy()
    for canon in ["scenario", "task", "comm", "seed", "step", "emitted_step", "delivery_step", "channel", "raw_channel", "sender_id", "receiver_id", "action", "action_class", "sequence", "sequence_length"]:
        col = find_alias(out, canon)
        if col is not None and canon not in out.columns:
            out[canon] = out[col]
    if "comm" not in out.columns:
        out["comm"] = "unknown"
    if "seed" not in out.columns:
        out["seed"] = np.nan
    if "scenario" not in out.columns:
        out["scenario"] = "unknown"
    if "task" not in out.columns:
        out["task"] = "unknown"
    if "step" not in out.columns:
        st = find_alias(out, "emitted_step") or find_alias(out, "delivery_step")
        out["step"] = out[st] if st else np.nan
    if "time_bin" not in out.columns:
        out["time_bin"] = out["step"].map(time_bin_for_step)
    if "channel" in out.columns:
        out["channel"] = pd.to_numeric(out["channel"], errors="coerce")
    if "seed" in out.columns:
        out["seed"] = pd.to_numeric(out["seed"], errors="coerce")
    return out


def relevant_usecols(cols: List[str], family: str) -> List[str]:
    keep = set()
    always_patterns = [
        "scenario", "task", "communication", "comm", "seed", "step", "time", "tick", "channel", "raw_channel",
        "sender", "receiver", "agent", "action", "event", "sequence", "lag", "distance", "intensity",
        "gradient", "population", "world", "z_size", "arm",
    ]
    metric_patterns = [
        "pressure", "hidden_ref", "asym_", "known", "danger", "resource", "unknown", "vertical", "friction",
        "relief", "surprise", "appraisal", "physics", "body", "damage", "recovery", "embodied", "delta", "gain",
        "entropy", "count", "value", "h", "q",
    ]
    for c in cols:
        cl = c.lower()
        if any(p in cl for p in always_patterns) or any(p in cl for p in metric_patterns):
            keep.add(c)
    # For episode summaries, keep all columns: they are not event-level huge.
    if family in {"run_index", "population_episode", "agent_episode", "primary_criteria"}:
        return cols
    return [c for c in cols if c in keep]


def read_csv_header(path: Path) -> List[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def count_csv_rows(path: Path) -> int:
    try:
        with open(path, "rb") as f:
            n = sum(1 for _ in f)
        return max(0, n - 1)
    except Exception:
        return 0


def read_csv_safe(path: Path, family: str, chunksize: Optional[int] = None) -> pd.DataFrame:
    try:
        cols = read_csv_header(path)
        if not cols:
            return pd.DataFrame()
        usecols = relevant_usecols(cols, family)
        if not usecols:
            usecols = cols
        if chunksize:
            chunks = []
            for ch in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
                ch["__source_file"] = str(path)
                chunks.append(add_canonical_columns(ch))
            return safe_concat(chunks)
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        df["__source_file"] = str(path)
        return add_canonical_columns(df)
    except Exception as e:
        return pd.DataFrame({"__read_error": [str(e)], "__source_file": [str(path)]})


def safe_concat(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    good = [x for x in frames if isinstance(x, pd.DataFrame) and not x.empty]
    if not good:
        return pd.DataFrame()
    try:
        return pd.concat(good, ignore_index=True, sort=False)
    except Exception:
        # Last-resort robust concat: align columns manually.
        cols = sorted(set().union(*[set(x.columns) for x in good]))
        return pd.concat([x.reindex(columns=cols) for x in good], ignore_index=True, sort=False)


def write_csv(path: Path, df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
    ensure_dir(path.parent)
    if df is None:
        df = pd.DataFrame()
    if columns is not None:
        for c in columns:
            if c not in df.columns:
                df[c] = np.nan
        df = df[columns + [c for c in df.columns if c not in columns]]
    df.to_csv(path, index=False)


def parse_sequence(x: Any) -> List[int]:
    if pd.isna(x):
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        vals = x
    else:
        s = str(x)
        # Accept "1,2,3", "[1 2 3]", "1-2-3", "1|2|3".
        vals = re.findall(r"-?\d+", s)
    out = []
    for v in vals:
        try:
            out.append(int(v))
        except Exception:
            pass
    return out


def ngrams(seq: List[int], n: int) -> List[Tuple[int, ...]]:
    if len(seq) < n:
        return []
    return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]


def mutual_information_bigram(seqs: List[List[int]]) -> float:
    pairs = []
    for s in seqs:
        pairs.extend(ngrams(s, 2))
    if not pairs:
        return np.nan
    a = Counter(x for x, _ in pairs)
    b = Counter(y for _, y in pairs)
    ab = Counter(pairs)
    n = sum(ab.values())
    mi = 0.0
    for (x, y), c in ab.items():
        pxy = c / n
        px = a[x] / n
        py = b[y] / n
        mi += pxy * math.log((pxy + 1e-15) / (px * py + 1e-15))
    return float(mi)


def g2_ngram_independence(seqs: List[List[int]], ngram_n: int = 2) -> Tuple[float, int, int]:
    grams = []
    tokens = []
    for s in seqs:
        tokens.extend(s)
        grams.extend(ngrams(s, ngram_n))
    if not grams or not tokens:
        return np.nan, 0, 0
    tok = Counter(tokens)
    total_tok = sum(tok.values())
    gram_count = Counter(grams)
    total_grams = sum(gram_count.values())
    g2 = 0.0
    for gram, obs in gram_count.items():
        expected_prob = 1.0
        for t in gram:
            expected_prob *= tok[t] / total_tok
        exp = max(1e-12, expected_prob * total_grams)
        g2 += 2.0 * obs * math.log((obs + 1e-12) / exp)
    df_approx = max(1, len(gram_count) - len(tok))
    return float(g2), int(total_grams), int(df_approx)


def empirical_sequence_null(seqs: List[List[int]], n_perm: int, rng: np.random.Generator) -> Tuple[float, float, float, int]:
    obs = mutual_information_bigram(seqs)
    if not np.isfinite(obs):
        return np.nan, np.nan, np.nan, 0
    null = []
    for _ in range(max(1, n_perm)):
        permuted = []
        for s in seqs:
            ss = list(s)
            rng.shuffle(ss)
            permuted.append(ss)
        null.append(mutual_information_bigram(permuted))
    null = np.array([x for x in null if np.isfinite(x)], dtype=float)
    if len(null) == 0:
        return obs, np.nan, np.nan, 0
    p = (1.0 + float(np.sum(null >= obs))) / (len(null) + 1.0)
    z = (obs - float(np.mean(null))) / (float(np.std(null)) + 1e-12)
    return obs, p, z, len(null)


def js_similarity_from_counts(a: Counter, b: Counter) -> float:
    keys = sorted(set(a.keys()) | set(b.keys()))
    if not keys:
        return np.nan
    pa = np.array([a.get(k, 0) for k in keys], dtype=float)
    pb = np.array([b.get(k, 0) for k in keys], dtype=float)
    if pa.sum() <= 0 or pb.sum() <= 0:
        return np.nan
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    m = 0.5 * (pa + pb)
    def kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log((p[mask] + 1e-15) / (q[mask] + 1e-15))))
    js = 0.5 * kl(pa, m) + 0.5 * kl(pb, m)
    max_js = math.log(2.0)
    return float(max(0.0, 1.0 - js / max_js))


@dataclass
class LoadedData:
    source_root: Path
    files: Dict[str, List[Path]]
    audit: pd.DataFrame
    dfs: Dict[str, pd.DataFrame]


# -----------------------------
# Discovery and loading
# -----------------------------

def discover_files(source_root: Path) -> Dict[str, List[Path]]:
    files = {k: [] for k in FAMILY_PATTERNS}
    if not source_root.exists():
        return files
    name_to_family = {}
    for fam, pats in FAMILY_PATTERNS.items():
        for p in pats:
            name_to_family[p.lower()] = fam
    for dirpath, _, filenames in os.walk(source_root):
        d = Path(dirpath)
        for fn in filenames:
            if fn.lower() in name_to_family:
                files[name_to_family[fn.lower()]].append(d / fn)
    for fam in files:
        files[fam] = sorted(set(files[fam]))
    return files


def load_data(source_root: Path, families: Optional[List[str]] = None, chunksize: Optional[int] = None) -> LoadedData:
    files = discover_files(source_root)
    rows = []
    dfs: Dict[str, pd.DataFrame] = {}
    target_fams = families if families else list(FAMILY_PATTERNS.keys())
    for fam in target_fams:
        paths = files.get(fam, [])
        f_frames = []
        for p in paths:
            cols = read_csv_header(p)
            nrows = count_csv_rows(p)
            status = "ok" if nrows > 0 else "empty_or_header_only"
            rows.append({
                "family": fam,
                "path": str(p),
                "rows": nrows,
                "n_columns": len(cols),
                "columns": "|".join(cols[:80]),
                "status": status,
            })
            if nrows > 0:
                f_frames.append(read_csv_safe(p, fam, chunksize=chunksize))
        dfs[fam] = safe_concat(f_frames)
    # Add missing families to audit.
    for fam in FAMILY_PATTERNS:
        if fam not in [r.get("family") for r in rows]:
            rows.append({"family": fam, "path": "", "rows": 0, "n_columns": 0, "columns": "", "status": "missing"})
    audit = pd.DataFrame(rows)
    return LoadedData(source_root=source_root, files=files, audit=audit, dfs=dfs)


# -----------------------------
# Analysis functions
# -----------------------------

def analysis_00_audit(db: LoadedData) -> pd.DataFrame:
    audit = db.audit.copy()
    # Add condition coverage if run index / population episode exists.
    coverage_rows = []
    for fam in ["run_index", "population_episode", "agent_episode", "signal_event", "signal_delivery", "receiver_action", "utterance_sequence"]:
        df = db.dfs.get(fam, pd.DataFrame())
        if df.empty:
            coverage_rows.append({"family": fam, "coverage_scope": "condition_seed", "condition_seed_rows": 0, "scenario_task_comm_seed_unique": 0})
            continue
        needed = ["scenario", "task", "comm", "seed"]
        for c in needed:
            if c not in df.columns:
                df[c] = np.nan
        uniq = df[needed].drop_duplicates()
        coverage_rows.append({
            "family": fam,
            "coverage_scope": "condition_seed",
            "condition_seed_rows": int(len(uniq)),
            "scenario_task_comm_seed_unique": int(len(uniq)),
            "scenario_unique": int(df["scenario"].nunique(dropna=True)),
            "task_unique": int(df["task"].nunique(dropna=True)),
            "comm_unique": int(df["comm"].nunique(dropna=True)),
            "seed_unique": int(df["seed"].nunique(dropna=True)),
        })
    cov = pd.DataFrame(coverage_rows)
    audit["audit_type"] = "file"
    cov["audit_type"] = "coverage"
    return pd.concat([audit, cov], ignore_index=True, sort=False)


def summarize_event_like(df: pd.DataFrame, family: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame([{"source_log": family, "status": "no_rows"}])
    x = add_canonical_columns(df)
    keys = ["scenario", "task", "comm", "seed", "time_bin"]
    for k in keys:
        if k not in x.columns:
            x[k] = "unknown" if k != "seed" else np.nan
    rows = []
    value_cols = [c for c in x.columns if c in SENDER_GROUNDING_METRICS + RECEIVER_CONTEXT_METRICS + OUTCOME_METRICS]
    has_channel = "channel" in x.columns
    has_action = "action_class" in x.columns or "action" in x.columns
    action_col = "action_class" if "action_class" in x.columns else ("action" if "action" in x.columns else None)
    for keys_vals, sub in x.groupby(keys, dropna=False):
        row = dict(zip(keys, keys_vals))
        row["source_log"] = family
        row["row_count"] = int(len(sub))
        if has_channel:
            row["channel_entropy_norm"] = entropy_norm(sub["channel"].tolist())
            row["top_channel_share"] = top_share(sub["channel"].tolist())
            row["channel_unique_count"] = unique_count(sub["channel"].tolist())
        else:
            row["channel_entropy_norm"] = np.nan
            row["top_channel_share"] = np.nan
            row["channel_unique_count"] = 0
        if has_action and action_col:
            row["action_entropy_norm"] = entropy_norm(sub[action_col].tolist())
            row["top_action_share"] = top_share(sub[action_col].tolist())
            row["action_unique_count"] = unique_count(sub[action_col].tolist())
        else:
            row["action_entropy_norm"] = np.nan
            row["top_action_share"] = np.nan
            row["action_unique_count"] = 0
        for c in value_cols:
            row[f"mean_{c}"] = pd.to_numeric(sub[c], errors="coerce").mean()
        if "sequence" in sub.columns:
            lens = [len(parse_sequence(v)) for v in sub["sequence"].tolist()]
            row["mean_sequence_length"] = float(np.mean(lens)) if lens else np.nan
            row["sequence_unique_count"] = int(len(set(map(str, sub["sequence"].tolist()))))
        rows.append(row)
    return pd.DataFrame(rows)


def analysis_01_development(db: LoadedData) -> pd.DataFrame:
    frames = []
    for fam in ["signal_event", "signal_delivery", "receiver_action", "utterance_sequence"]:
        frames.append(summarize_event_like(db.dfs.get(fam, pd.DataFrame()), fam))
    out = safe_concat(frames)
    if out.empty:
        return pd.DataFrame([{"status": "no_development_rows", "reason": "no signal/delivery/receiver/utterance logs found"}])
    # Add early-late change rows as another view.
    change_rows = []
    metrics = [c for c in out.columns if c.startswith("mean_") or c in ["row_count", "channel_entropy_norm", "top_channel_share", "action_entropy_norm", "mean_sequence_length", "sequence_unique_count"]]
    group_cols = ["scenario", "task", "comm", "seed", "source_log"]
    for keys, sub in out.groupby(group_cols, dropna=False):
        for metric in metrics:
            vals = sub[["time_bin", metric]].copy()
            if vals.empty:
                continue
            early = pd.to_numeric(vals.loc[vals["time_bin"].astype(str) == "0_1000", metric], errors="coerce").mean()
            late = pd.to_numeric(vals.loc[vals["time_bin"].astype(str) == "8000_16000", metric], errors="coerce").mean()
            if np.isfinite(early) or np.isfinite(late):
                row = dict(zip(group_cols, keys))
                row.update({"time_bin": "early_late_delta", "temporal_metric": metric, "early_0_1000": early, "late_8000_16000": late, "late_minus_early": late - early if np.isfinite(early) and np.isfinite(late) else np.nan})
                change_rows.append(row)
    if change_rows:
        ch = pd.DataFrame(change_rows)
        out["temporal_metric"] = np.nan
        out = pd.concat([out, ch], ignore_index=True, sort=False)
    return out


def analysis_02_sender_grounding(db: LoadedData) -> pd.DataFrame:
    df = db.dfs.get("signal_event", pd.DataFrame())
    if df.empty:
        return pd.DataFrame([{"status": "no_signal_event_rows", "reason": "signal_event_log.csv missing or empty"}])
    x = add_canonical_columns(df)
    if "channel" not in x.columns:
        return pd.DataFrame([{"status": "missing_channel", "reason": "signal event file has no channel/raw_channel column"}])
    rows = []
    metrics = [m for m in SENDER_GROUNDING_METRICS if m in x.columns]
    if not metrics:
        return pd.DataFrame([{"status": "missing_sender_metrics", "reason": "no sender internal/environment metrics available"}])
    for keys, sub in x.groupby(["scenario", "task", "comm"], dropna=False):
        for m in metrics:
            eta = eta_squared_categorical(sub["channel"], sub[m])
            rho = spearman_safe(sub["channel"], sub[m])
            rows.append({
                "scenario": keys[0], "task": keys[1], "comm": keys[2],
                "grounding_metric": m,
                "n": int(pd.to_numeric(sub[m], errors="coerce").notna().sum()),
                "channel_eta2": eta,
                "channel_spearman_r": rho,
                "abs_spearman_r": abs(rho) if np.isfinite(rho) else np.nan,
                "grounding_family": "sender_internal_or_environment",
                "status": "ok",
            })
    return pd.DataFrame(rows)


def action_counts_by_channel(sub: pd.DataFrame, action_col: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "channel" not in sub.columns or action_col not in sub.columns:
        return out
    # Action distribution difference by channel: mean total variation from global distribution.
    actions = sorted([a for a in sub[action_col].dropna().unique()])
    if not actions:
        return out
    global_counts = sub[action_col].value_counts(normalize=True).reindex(actions).fillna(0.0)
    tvs = []
    for _, ss in sub.groupby("channel", dropna=True):
        p = ss[action_col].value_counts(normalize=True).reindex(actions).fillna(0.0)
        tvs.append(0.5 * float(np.abs(p.values - global_counts.values).sum()))
    out["mean_channel_action_total_variation"] = float(np.nanmean(tvs)) if tvs else np.nan
    out["action_entropy_norm"] = entropy_norm(sub[action_col].tolist())
    out["top_action_share"] = top_share(sub[action_col].tolist())
    return out


def analysis_03_receiver_action_effects(db: LoadedData) -> pd.DataFrame:
    df = db.dfs.get("receiver_action", pd.DataFrame())
    if df.empty:
        return pd.DataFrame([{"status": "no_receiver_action_rows", "reason": "receiver_signal_action_log.csv/receiver_action_log.csv missing or empty"}])
    x = add_canonical_columns(df)
    action_col = "action_class" if "action_class" in x.columns else ("action" if "action" in x.columns else None)
    if action_col is None:
        return pd.DataFrame([{"status": "missing_receiver_action_column", "reason": "no receiver_action/receiver_action_class/action column"}])
    rows = []
    outcome_cols = [c for c in OUTCOME_METRICS if c in x.columns]
    for keys, sub in x.groupby(["scenario", "task", "comm", "time_bin"], dropna=False):
        base = {"scenario": keys[0], "task": keys[1], "comm": keys[2], "time_bin": keys[3], "n": int(len(sub)), "status": "ok"}
        base.update(action_counts_by_channel(sub, action_col))
        if "channel" in sub.columns:
            base["channel_entropy_norm"] = entropy_norm(sub["channel"].tolist())
            base["top_channel_share"] = top_share(sub["channel"].tolist())
        for oc in outcome_cols:
            base[f"mean_{oc}"] = pd.to_numeric(sub[oc], errors="coerce").mean()
        rows.append(base)
        # Action-specific rows.
        for action, ss in sub.groupby(action_col, dropna=False):
            r = {"scenario": keys[0], "task": keys[1], "comm": keys[2], "time_bin": keys[3], "receiver_action_class": action, "n": int(len(ss)), "row_type": "action_specific", "status": "ok"}
            if "channel" in ss.columns:
                r["channel_entropy_norm"] = entropy_norm(ss["channel"].tolist())
                r["top_channel_share"] = top_share(ss["channel"].tolist())
            for oc in outcome_cols:
                r[f"mean_{oc}"] = pd.to_numeric(ss[oc], errors="coerce").mean()
            rows.append(r)
    out = pd.DataFrame(rows)
    # Active-control contrast by condition/time/action summary.
    contrast_rows = []
    if not out.empty:
        metric_cols = [c for c in out.columns if c.startswith("mean_") or c in ["mean_channel_action_total_variation", "action_entropy_norm", "channel_entropy_norm"]]
        for (sc, task, tb), sub in out[out.get("row_type", "summary").fillna("summary") != "action_specific"].groupby(["scenario", "task", "time_bin"], dropna=False):
            for metric in metric_cols:
                act = pd.to_numeric(sub.loc[sub["comm"].isin(ACTIVE_COMMS), metric], errors="coerce").mean()
                ctl = pd.to_numeric(sub.loc[sub["comm"].isin(CONTROL_COMMS), metric], errors="coerce").mean()
                contrast_rows.append({"scenario": sc, "task": task, "time_bin": tb, "row_type": "active_vs_control_contrast", "metric": metric, "active_mean": act, "control_mean": ctl, "active_minus_control": act - ctl if np.isfinite(act) and np.isfinite(ctl) else np.nan})
    if contrast_rows:
        out = pd.concat([out, pd.DataFrame(contrast_rows)], ignore_index=True, sort=False)
    return out


def analysis_04_nonreflexive_context(db: LoadedData) -> pd.DataFrame:
    df = db.dfs.get("receiver_action", pd.DataFrame())
    if df.empty:
        return pd.DataFrame([{"status": "no_receiver_action_rows", "reason": "cannot test context dependence without receiver action rows"}])
    x = add_canonical_columns(df)
    if "channel" not in x.columns:
        return pd.DataFrame([{"status": "missing_channel", "reason": "no channel column"}])
    action_col = "action_class" if "action_class" in x.columns else ("action" if "action" in x.columns else None)
    context_cols = [c for c in RECEIVER_CONTEXT_METRICS if c in x.columns]
    outcome_cols = [c for c in OUTCOME_METRICS if c in x.columns]
    if not context_cols:
        return pd.DataFrame([{"status": "missing_context_metrics", "reason": "no sender/receiver context metrics"}])
    rows = []
    for keys, sub in x.groupby(["scenario", "task", "comm"], dropna=False):
        for ctx in context_cols:
            vals = pd.to_numeric(sub[ctx], errors="coerce")
            if vals.notna().sum() < 10 or vals.nunique(dropna=True) < 2:
                continue
            med = vals.median()
            lo = sub.loc[vals <= med]
            hi = sub.loc[vals > med]
            if len(lo) == 0 or len(hi) == 0:
                continue
            for ch, s_ch in sub.groupby("channel", dropna=True):
                vals_ch = pd.to_numeric(s_ch[ctx], errors="coerce")
                if vals_ch.notna().sum() < 4:
                    continue
                med_ch = vals_ch.median()
                lo_ch = s_ch.loc[vals_ch <= med_ch]
                hi_ch = s_ch.loc[vals_ch > med_ch]
                row = {"scenario": keys[0], "task": keys[1], "comm": keys[2], "context_metric": ctx, "channel": ch, "n_low": int(len(lo_ch)), "n_high": int(len(hi_ch)), "status": "ok"}
                if action_col:
                    row["action_entropy_high_minus_low"] = entropy_norm(hi_ch[action_col].tolist()) - entropy_norm(lo_ch[action_col].tolist()) if len(lo_ch) and len(hi_ch) else np.nan
                    row["top_action_share_high_minus_low"] = top_share(hi_ch[action_col].tolist()) - top_share(lo_ch[action_col].tolist()) if len(lo_ch) and len(hi_ch) else np.nan
                for oc in outcome_cols:
                    h = pd.to_numeric(hi_ch[oc], errors="coerce").mean()
                    l = pd.to_numeric(lo_ch[oc], errors="coerce").mean()
                    row[f"{oc}_high_minus_low"] = h - l if np.isfinite(h) and np.isfinite(l) else np.nan
                # Overall interaction score from available deltas.
                deltas = [abs(v) for k, v in row.items() if k.endswith("_high_minus_low") and np.isfinite(safe_float(v))]
                row["context_interaction_strength"] = float(np.mean(deltas)) if deltas else np.nan
                rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame([{"status": "no_context_rows", "reason": "insufficient context variation"}])


def analysis_05_internal_grounding(db: LoadedData, sender_grounding: pd.DataFrame) -> pd.DataFrame:
    df = db.dfs.get("signal_event", pd.DataFrame())
    if df.empty:
        return pd.DataFrame([{"status": "no_signal_event_rows"}])
    x = add_canonical_columns(df)
    if "channel" not in x.columns:
        return pd.DataFrame([{"status": "missing_channel"}])
    internal_metrics = [m for m in SENDER_GROUNDING_METRICS if m in x.columns]
    rows = []
    # Condition-level internal score from channel eta2 over internal metrics.
    sg = sender_grounding.copy() if sender_grounding is not None and not sender_grounding.empty else pd.DataFrame()
    for keys, sub in x.groupby(["scenario", "task", "comm"], dropna=False):
        sc, task, comm = keys
        internal_scores = []
        for m in internal_metrics:
            internal_scores.append(eta_squared_categorical(sub["channel"], sub[m]))
        internal_score = float(np.nanmean(internal_scores)) if internal_scores else np.nan
        # External label predictability proxy: channel eta by scenario/task is not within condition, so use global separately below.
        rows.append({"scenario": sc, "task": task, "comm": comm, "internal_grounding_eta2_mean": internal_score, "n_internal_metrics": len(internal_scores), "status": "ok"})
    # Global external label effects.
    for label in ["scenario", "task", "comm"]:
        if label in x.columns:
            rows.append({
                "scenario": "ALL", "task": "ALL", "comm": "ALL", "external_label": label,
                "external_label_channel_eta2": eta_squared_categorical(x[label], x["channel"]),
                "status": "global_external_label_proxy",
            })
    out = pd.DataFrame(rows)
    # Add global comparison scalar.
    internal_mean = pd.to_numeric(out["internal_grounding_eta2_mean"], errors="coerce").mean() if "internal_grounding_eta2_mean" in out.columns else np.nan
    external_mean = pd.to_numeric(out["external_label_channel_eta2"], errors="coerce").mean() if "external_label_channel_eta2" in out.columns else np.nan
    out["global_internal_mean_eta2"] = internal_mean
    out["global_external_label_mean_eta2"] = external_mean
    out["internal_minus_external_label_eta2"] = internal_mean - external_mean if np.isfinite(internal_mean) and np.isfinite(external_mean) else np.nan
    return out


def analysis_06_sequence_ngram(db: LoadedData, n_null: int = 50) -> pd.DataFrame:
    df = db.dfs.get("utterance_sequence", pd.DataFrame())
    if df.empty:
        return pd.DataFrame([{"status": "no_utterance_sequence_rows", "reason": "utterance_sequence_log.csv missing or empty"}])
    x = add_canonical_columns(df)
    if "sequence" not in x.columns:
        return pd.DataFrame([{"status": "missing_sequence_column", "reason": "no sequence column in utterance log"}])
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for keys, sub in x.groupby(["scenario", "task", "comm", "time_bin"], dropna=False):
        seqs = [parse_sequence(v) for v in sub["sequence"].tolist()]
        seqs = [s for s in seqs if len(s) >= 2]
        obs_mi, p_emp, z_emp, n_perm = empirical_sequence_null(seqs, n_null, rng)
        g2_2, n2, df2 = g2_ngram_independence(seqs, 2)
        g2_3, n3, df3 = g2_ngram_independence(seqs, 3)
        rows.append({
            "scenario": keys[0], "task": keys[1], "comm": keys[2], "time_bin": keys[3],
            "n_sequences": len(seqs),
            "mean_sequence_length": float(np.mean([len(s) for s in seqs])) if seqs else np.nan,
            "bigram_mi_observed": obs_mi,
            "bigram_order_shuffle_p": p_emp,
            "bigram_order_shuffle_z": z_emp,
            "n_null_permutations": n_perm,
            "bigram_g2_vs_unigram_independence": g2_2,
            "bigram_count": n2,
            "trigram_g2_vs_unigram_independence": g2_3,
            "trigram_count": n3,
            "status": "ok" if seqs else "insufficient_sequences",
        })
    out = pd.DataFrame(rows)
    if "bigram_order_shuffle_p" in out.columns:
        out["bigram_order_shuffle_q_bh"] = bh_fdr(out["bigram_order_shuffle_p"].tolist())
        out["grammar_like_order_survives_fdr"] = (pd.to_numeric(out["bigram_order_shuffle_q_bh"], errors="coerce") <= 0.05) & (pd.to_numeric(out["bigram_order_shuffle_z"], errors="coerce") > 0)
    return out


def analysis_07_bigram_trigram_action_prediction(db: LoadedData) -> pd.DataFrame:
    utt = db.dfs.get("utterance_sequence", pd.DataFrame())
    recv = db.dfs.get("receiver_action", pd.DataFrame())
    if utt.empty or recv.empty:
        return pd.DataFrame([{"status": "missing_required_rows", "reason": "requires utterance_sequence and receiver_action logs"}])
    u = add_canonical_columns(utt)
    r = add_canonical_columns(recv)
    if "sequence" not in u.columns:
        return pd.DataFrame([{"status": "missing_sequence_column"}])
    action_col = "action_class" if "action_class" in r.columns else ("action" if "action" in r.columns else None)
    if action_col is None:
        return pd.DataFrame([{"status": "missing_action_column"}])
    # Aggregate by condition/seed/time_bin, then correlate sequence features with receiver action features.
    u_rows = []
    for keys, sub in u.groupby(["scenario", "task", "comm", "seed", "time_bin"], dropna=False):
        seqs = [parse_sequence(v) for v in sub["sequence"].tolist()]
        seqs = [s for s in seqs if s]
        bigrams = []
        trigrams = []
        for s in seqs:
            bigrams.extend(ngrams(s, 2)); trigrams.extend(ngrams(s, 3))
        u_rows.append({
            "scenario": keys[0], "task": keys[1], "comm": keys[2], "seed": keys[3], "time_bin": keys[4],
            "n_sequences": len(seqs), "bigram_entropy_norm": entropy_norm(bigrams), "trigram_entropy_norm": entropy_norm(trigrams),
            "top_bigram_share": top_share(bigrams), "top_trigram_share": top_share(trigrams),
            "mean_sequence_length": float(np.mean([len(s) for s in seqs])) if seqs else np.nan,
        })
    r_rows = []
    outcome_cols = [c for c in OUTCOME_METRICS if c in r.columns]
    for keys, sub in r.groupby(["scenario", "task", "comm", "seed", "time_bin"], dropna=False):
        row = {"scenario": keys[0], "task": keys[1], "comm": keys[2], "seed": keys[3], "time_bin": keys[4], "receiver_action_entropy_norm": entropy_norm(sub[action_col].tolist()), "receiver_top_action_share": top_share(sub[action_col].tolist()), "n_receiver_actions": int(len(sub))}
        for oc in outcome_cols:
            row[f"mean_{oc}"] = pd.to_numeric(sub[oc], errors="coerce").mean()
        r_rows.append(row)
    uu = pd.DataFrame(u_rows)
    rr = pd.DataFrame(r_rows)
    if uu.empty or rr.empty:
        return pd.DataFrame([{"status": "insufficient_aggregated_rows"}])
    m = uu.merge(rr, on=["scenario", "task", "comm", "seed", "time_bin"], how="inner")
    if m.empty:
        return pd.DataFrame([{"status": "no_matched_time_bin_rows"}])
    rows = []
    predictor_cols = ["bigram_entropy_norm", "trigram_entropy_norm", "top_bigram_share", "top_trigram_share", "mean_sequence_length", "n_sequences"]
    target_cols = ["receiver_action_entropy_norm", "receiver_top_action_share", "n_receiver_actions"] + [c for c in m.columns if c.startswith("mean_")]
    for keys, sub in m.groupby(["scenario", "task", "comm"], dropna=False):
        for p in predictor_cols:
            for t in target_cols:
                rho = spearman_safe(sub[p], sub[t])
                n = int(pd.DataFrame({p: sub[p], t: sub[t]}).dropna().shape[0])
                rows.append({"scenario": keys[0], "task": keys[1], "comm": keys[2], "sequence_predictor": p, "receiver_or_outcome_target": t, "n_time_seed_bins": n, "spearman_r": rho, "abs_spearman_r": abs(rho) if np.isfinite(rho) else np.nan, "p_value_approx": fisher_p_from_r(rho, n), "status": "ok"})
    out = pd.DataFrame(rows)
    out["q_value_bh"] = bh_fdr(out["p_value_approx"].tolist()) if "p_value_approx" in out.columns else np.nan
    return out


def analysis_08_sequence_to_outcome(db: LoadedData, seq_pred: pd.DataFrame) -> pd.DataFrame:
    if seq_pred is None or seq_pred.empty:
        return pd.DataFrame([{"status": "no_sequence_prediction_rows"}])
    df = seq_pred.copy()
    if "receiver_or_outcome_target" not in df.columns:
        return pd.DataFrame([{"status": "missing_target_column"}])
    outcome_mask = df["receiver_or_outcome_target"].astype(str).str.contains("damage|body|embodied|recovery|resource|entered_unknown|vertical|delta_h", case=False, regex=True)
    out = df.loc[outcome_mask].copy()
    if out.empty:
        return pd.DataFrame([{"status": "no_outcome_relevance_rows", "reason": "no sequence predictor correlated with outcome target columns"}])
    # Orientation-aware functional sign.
    def orient(row):
        target = str(row.get("receiver_or_outcome_target", ""))
        sign = -1.0 if any(k in target.lower() for k in ["damage"]) else 1.0
        r = safe_float(row.get("spearman_r"))
        return sign * r if np.isfinite(r) else np.nan
    out["oriented_functional_r"] = out.apply(orient, axis=1)
    out["functional_positive"] = pd.to_numeric(out["oriented_functional_r"], errors="coerce") > 0
    return out


def analysis_09_shared_codebook(db: LoadedData) -> pd.DataFrame:
    df = db.dfs.get("signal_event", pd.DataFrame())
    if df.empty:
        return pd.DataFrame([{"status": "no_signal_event_rows"}])
    x = add_canonical_columns(df)
    if "sender_id" not in x.columns or "channel" not in x.columns:
        return pd.DataFrame([{"status": "missing_sender_or_channel"}])
    rows = []
    for keys, sub in x.groupby(["scenario", "task", "comm", "seed", "time_bin"], dropna=False):
        agent_counts = {}
        for aid, ss in sub.groupby("sender_id", dropna=False):
            agent_counts[aid] = Counter(pd.to_numeric(ss["channel"], errors="coerce").dropna().astype(int).tolist())
        sims = []
        agents = list(agent_counts.keys())
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                sims.append(js_similarity_from_counts(agent_counts[agents[i]], agent_counts[agents[j]]))
        rows.append({
            "scenario": keys[0], "task": keys[1], "comm": keys[2], "seed": keys[3], "time_bin": keys[4],
            "n_sender_agents": len(agents),
            "mean_pairwise_channel_distribution_similarity": float(np.nanmean(sims)) if sims else np.nan,
            "channel_entropy_norm_population": entropy_norm(sub["channel"].tolist()),
            "top_channel_share_population": top_share(sub["channel"].tolist()),
            "status": "ok" if sims else "insufficient_agents",
        })
    return pd.DataFrame(rows)


def aggregate_condition_scores(dev: pd.DataFrame, sender: pd.DataFrame, receiver: pd.DataFrame, nonref: pd.DataFrame, internal: pd.DataFrame, seq: pd.DataFrame, seq_outcome: pd.DataFrame, shared: pd.DataFrame) -> pd.DataFrame:
    rows = []
    conditions = [(s, t) for s in SCENARIOS for t in TASKS]
    # Include discovered conditions not in locked set.
    for df in [dev, sender, receiver, nonref, internal, seq, seq_outcome, shared]:
        if isinstance(df, pd.DataFrame) and not df.empty and {"scenario", "task"}.issubset(df.columns):
            for s, t in df[["scenario", "task"]].drop_duplicates().itertuples(index=False):
                if (s, t) not in conditions and str(s) != "ALL":
                    conditions.append((s, t))
    for sc, task in conditions:
        row = {"scenario": sc, "task": task}
        # Sender grounding: mean top eta among active comms.
        if isinstance(sender, pd.DataFrame) and not sender.empty and "channel_eta2" in sender.columns:
            ss = sender[(sender["scenario"] == sc) & (sender["task"] == task) & (sender["comm"].isin(ACTIVE_COMMS))]
            row["sender_grounding_score"] = pd.to_numeric(ss["channel_eta2"], errors="coerce").quantile(0.75) if not ss.empty else np.nan
            row["sender_grounding_n_metrics"] = int(ss["grounding_metric"].nunique()) if "grounding_metric" in ss.columns and not ss.empty else 0
        else:
            row["sender_grounding_score"] = np.nan
            row["sender_grounding_n_metrics"] = 0
        # Receiver effect: active-control contrast mean action TV / outcome.
        if isinstance(receiver, pd.DataFrame) and not receiver.empty and "row_type" in receiver.columns:
            rr = receiver[(receiver["scenario"] == sc) & (receiver["task"] == task) & (receiver["row_type"] == "active_vs_control_contrast")]
            if not rr.empty and "active_minus_control" in rr.columns:
                vals = pd.to_numeric(rr["active_minus_control"], errors="coerce")
                row["receiver_action_effect_score"] = float(vals.abs().mean()) if vals.notna().any() else np.nan
                row["receiver_action_effect_positive_fraction"] = float((vals > 0).mean()) if vals.notna().any() else np.nan
            else:
                row["receiver_action_effect_score"] = np.nan
                row["receiver_action_effect_positive_fraction"] = np.nan
        else:
            row["receiver_action_effect_score"] = np.nan
            row["receiver_action_effect_positive_fraction"] = np.nan
        # Nonreflexivity.
        if isinstance(nonref, pd.DataFrame) and not nonref.empty and "context_interaction_strength" in nonref.columns:
            nn = nonref[(nonref["scenario"] == sc) & (nonref["task"] == task) & (nonref["comm"].isin(ACTIVE_COMMS))]
            row["nonreflexive_context_score"] = pd.to_numeric(nn["context_interaction_strength"], errors="coerce").mean() if not nn.empty else np.nan
        else:
            row["nonreflexive_context_score"] = np.nan
        # Internal grounding.
        if isinstance(internal, pd.DataFrame) and not internal.empty and "internal_grounding_eta2_mean" in internal.columns:
            ii = internal[(internal["scenario"] == sc) & (internal["task"] == task) & (internal["comm"].isin(ACTIVE_COMMS))]
            row["internal_grounding_score"] = pd.to_numeric(ii["internal_grounding_eta2_mean"], errors="coerce").mean() if not ii.empty else np.nan
        else:
            row["internal_grounding_score"] = np.nan
        # Grammar/order.
        if isinstance(seq, pd.DataFrame) and not seq.empty and "grammar_like_order_survives_fdr" in seq.columns:
            qq = seq[(seq["scenario"] == sc) & (seq["task"] == task) & (seq["comm"].isin(ACTIVE_COMMS))]
            row["grammar_order_fdr_fraction"] = float(qq["grammar_like_order_survives_fdr"].fillna(False).mean()) if not qq.empty else 0.0
            row["grammar_order_z_mean"] = pd.to_numeric(qq.get("bigram_order_shuffle_z", pd.Series(dtype=float)), errors="coerce").mean() if not qq.empty else np.nan
        else:
            row["grammar_order_fdr_fraction"] = 0.0
            row["grammar_order_z_mean"] = np.nan
        # Sequence to outcome.
        if isinstance(seq_outcome, pd.DataFrame) and not seq_outcome.empty and "oriented_functional_r" in seq_outcome.columns:
            so = seq_outcome[(seq_outcome["scenario"] == sc) & (seq_outcome["task"] == task) & (seq_outcome["comm"].isin(ACTIVE_COMMS) if "comm" in seq_outcome.columns else True)]
            row["sequence_outcome_functional_score"] = pd.to_numeric(so["oriented_functional_r"], errors="coerce").mean() if not so.empty else np.nan
            row["sequence_outcome_positive_fraction"] = float(so["functional_positive"].fillna(False).mean()) if not so.empty and "functional_positive" in so.columns else np.nan
        else:
            row["sequence_outcome_functional_score"] = np.nan
            row["sequence_outcome_positive_fraction"] = np.nan
        # Sharing.
        if isinstance(shared, pd.DataFrame) and not shared.empty and "mean_pairwise_channel_distribution_similarity" in shared.columns:
            sh = shared[(shared["scenario"] == sc) & (shared["task"] == task) & (shared["comm"].isin(ACTIVE_COMMS))]
            row["within_population_sharing_score"] = pd.to_numeric(sh["mean_pairwise_channel_distribution_similarity"], errors="coerce").mean() if not sh.empty else np.nan
        else:
            row["within_population_sharing_score"] = np.nan
        # Temporal development: late minus early for sequence length/channel entropy/receiver action effect.
        if isinstance(dev, pd.DataFrame) and not dev.empty and "time_bin" in dev.columns:
            dd = dev[(dev["scenario"] == sc) & (dev["task"] == task) & (dev["comm"].isin(ACTIVE_COMMS))]
            delta_cols = ["late_minus_early"] if "late_minus_early" in dd.columns else []
            if delta_cols:
                row["temporal_development_score"] = pd.to_numeric(dd["late_minus_early"], errors="coerce").mean()
            else:
                row["temporal_development_score"] = np.nan
        else:
            row["temporal_development_score"] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    # Conservative fixed classification. These are deliberately descriptive and not L7/cross-pop gates.
    def present(v, weak=0.03):
        return np.isfinite(safe_float(v)) and abs(safe_float(v)) >= weak
    classes = []
    for _, r in out.iterrows():
        grounded = present(r.get("sender_grounding_score"), 0.03) or present(r.get("internal_grounding_score"), 0.03)
        receiver_eff = present(r.get("receiver_action_effect_score"), 0.03)
        nonref_ok = present(r.get("nonreflexive_context_score"), 0.01)
        grammar_ok = safe_float(r.get("grammar_order_fdr_fraction"), 0.0) > 0 or safe_float(r.get("grammar_order_z_mean"), 0.0) > 1.5
        functional_ok = present(r.get("sequence_outcome_functional_score"), 0.05) or safe_float(r.get("sequence_outcome_positive_fraction"), 0.0) >= 0.60
        shared_ok = safe_float(r.get("within_population_sharing_score"), 0.0) >= 0.50
        temporal_ok = present(r.get("temporal_development_score"), 0.01)
        n_core = sum([grounded, receiver_eff, nonref_ok, shared_ok, temporal_ok])
        if grounded and receiver_eff and nonref_ok and grammar_ok and functional_ok and shared_ok:
            cls = "A_language_like_proto_communication"
        elif grammar_ok and (receiver_eff or functional_ok):
            cls = "B_grammar_like_precursor"
        elif grounded and receiver_eff and nonref_ok:
            cls = "C_grounded_signal_system"
        elif n_core >= 2:
            cls = "D_signaling_with_partial_language_like_organization"
        else:
            cls = "E_no_reliable_language_like_organization"
        classes.append({
            "grounded_present": grounded,
            "receiver_effect_present": receiver_eff,
            "nonreflexive_context_present": nonref_ok,
            "grammar_like_sequence_present": grammar_ok,
            "functional_outcome_present": functional_ok,
            "within_population_shared_present": shared_ok,
            "temporal_development_present": temporal_ok,
            "language_likeness_class": cls,
            "core_positive_count_excluding_cross_pop_transfer": n_core + int(grammar_ok) + int(functional_ok),
        })
    cls_df = pd.DataFrame(classes)
    return pd.concat([out, cls_df], axis=1)


def write_report(outdir: Path, audit: pd.DataFrame, outputs: Dict[str, pd.DataFrame], judgement: pd.DataFrame, errors: List[str]) -> None:
    lines = []
    lines.append("# Language-likeness Reanalysis Report")
    lines.append("")
    lines.append(f"Generated: {now()}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This reanalysis asks whether the acquired within-population signal system can be treated as language-like proto-communication. Cross-population L7/L7b transfer is not used as a primary gate.")
    lines.append("")
    lines.append("## Output availability")
    lines.append("")
    for name in EXPECTED_OUTPUTS:
        if name.endswith(".md"):
            continue
        df = outputs.get(name, pd.DataFrame())
        lines.append(f"- `{name}`: rows={0 if df is None else len(df)}")
    lines.append("")
    if errors:
        lines.append("## Non-fatal errors / warnings")
        lines.append("")
        for e in errors[:50]:
            lines.append(f"- {e}")
        lines.append("")
    lines.append("## File audit summary")
    lines.append("")
    if audit is not None and not audit.empty and "family" in audit.columns:
        fam = audit.groupby("family", dropna=False).agg(paths=("path", lambda x: int(sum(bool(str(v)) for v in x))), rows=("rows", "sum")).reset_index()
        for _, r in fam.iterrows():
            lines.append(f"- {r['family']}: paths={int(r['paths'])}, rows={int(r['rows'])}")
    lines.append("")
    lines.append("## Language-likeness judgement")
    lines.append("")
    if judgement is not None and not judgement.empty and "language_likeness_class" in judgement.columns:
        counts = judgement["language_likeness_class"].value_counts(dropna=False)
        for k, v in counts.items():
            lines.append(f"- {k}: {int(v)}")
        lines.append("")
        top_cols = ["scenario", "task", "language_likeness_class", "core_positive_count_excluding_cross_pop_transfer", "sender_grounding_score", "receiver_action_effect_score", "grammar_order_fdr_fraction", "sequence_outcome_functional_score", "within_population_sharing_score"]
        top_cols = [c for c in top_cols if c in judgement.columns]
        if top_cols:
            lines.append("### Condition table")
            lines.append("")
            lines.append(judgement[top_cols].to_markdown(index=False))
            lines.append("")
    else:
        lines.append("No judgement rows were generated. Check audit CSV for missing input logs.")
        lines.append("")
    lines.append("## Conservative interpretation rule")
    lines.append("")
    lines.append("A condition is treated as language-like only when within-population grounding, receiver action effect, non-reflexive/context-dependent response, grammar-like sequence structure, functional outcome relevance, and shared use are jointly supported. Signal frequency alone, language_like_level, and cross-population transfer are not treated as sufficient evidence.")
    lines.append("")
    (outdir / "11_language_acquisition_final_report.md").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Mock data for self-test
# -----------------------------

def create_mock_source(root: Path) -> None:
    safe_unlink_dir(root)
    rng = np.random.default_rng(123)
    scenarios = ["baseline_3d", "unknown_x1p50"]
    tasks = ["exploration_recovery", "physics_adaptation"]
    comms = ["PRIVATE", "RECEIVER_LEARN", "FULL_INTERACTIVE"]
    seeds = [0, 1]
    for sc in scenarios:
        shard = root / "shards" / sc
        ensure_dir(shard)
        sig_rows = []
        del_rows = []
        rec_rows = []
        utt_rows = []
        pop_rows = []
        run_rows = []
        for task in tasks:
            for comm in comms:
                for seed in seeds:
                    run_rows.append({"scenario": sc, "task": task, "communication_condition": comm, "seed_id": seed, "status": "complete"})
                    pop_rows.append({"scenario": sc, "task": task, "communication_condition": comm, "seed_id": seed, "steps": 16000, "final_body_h_mean": 0.7 + 0.1 * rng.random()})
                    for step in range(0, 16000, 200):
                        unknown_p = rng.random() + (0.5 if sc == "unknown_x1p50" else 0)
                        vertical_p = rng.random() + (0.4 if sc == "baseline_3d" else 0)
                        channel = int((unknown_p > 0.8) * 2 + (vertical_p > 0.8) * 1 + rng.integers(0, 2)) % 5
                        if comm == "PRIVATE":
                            channel = int(rng.integers(0, 5))
                        sig_rows.append({"scenario": sc, "task": task, "communication_condition": comm, "seed_id": seed, "sender_id": int(rng.integers(0, 8)), "channel": channel, "raw_channel": channel, "emitted_step": step, "sender_h": rng.random(), "sender_Q": rng.random(), "sender_unknown_pressure": unknown_p, "sender_vertical_pressure": vertical_p, "sender_danger_pressure": rng.random(), "sender_resource_pressure": rng.random()})
                        receiver = int(rng.integers(0, 8))
                        del_rows.append({"scenario": sc, "task": task, "communication_condition": comm, "seed_id": seed, "sender_id": 0, "receiver_id": receiver, "channel": channel, "delivery_step": step + 1, "emitted_step": step, "distance": rng.random(), "receiver_unknown_pressure": unknown_p + 0.1 * rng.normal(), "receiver_vertical_pressure": vertical_p + 0.1 * rng.normal(), "sender_unknown_pressure": unknown_p, "sender_vertical_pressure": vertical_p})
                        action = "SCAN" if channel in [2, 3] and comm != "PRIVATE" else rng.choice(["MOVE", "REST", "SCAN", "VERTICAL"])
                        rec_rows.append({"scenario": sc, "task": task, "communication_condition": comm, "seed_id": seed, "sender_id": 0, "receiver_id": receiver, "channel": channel, "delivery_step": step + 2, "emitted_step": step, "lag": 2, "receiver_action": action, "receiver_action_class": action, "receiver_delta_h": 0.01 * rng.normal(), "receiver_damage": max(0, 0.02 * rng.random() - (0.005 if action == "SCAN" else 0)), "receiver_recovery_gain": 0.01 * (action == "SCAN") + 0.01 * rng.random(), "receiver_entered_unknown": int(action == "MOVE" and sc == "unknown_x1p50"), "receiver_vertical": int(action == "VERTICAL"), "receiver_embodied_value": 0.05 * (action == "SCAN") + 0.01 * rng.normal(), "receiver_Q": rng.random(), "receiver_body_h": 0.6 + 0.2 * rng.random(), "receiver_unknown_pressure": unknown_p, "receiver_vertical_pressure": vertical_p})
                        seq = [channel, (channel + 1) % 5] if comm != "PRIVATE" else [int(rng.integers(0, 5)), int(rng.integers(0, 5))]
                        if step > 8000 and comm == "FULL_INTERACTIVE":
                            seq = [2, 3, 2]
                        utt_rows.append({"scenario": sc, "task": task, "communication_condition": comm, "seed_id": seed, "sender_id": 0, "step": step, "sequence_length": len(seq), "sequence": "-".join(map(str, seq)), "last_channel": seq[-1], "sender_unknown_pressure": unknown_p, "sender_vertical_pressure": vertical_p, "sender_h": rng.random(), "sender_Q": rng.random()})
        pd.DataFrame(sig_rows).to_csv(shard / "signal_event_log.csv", index=False)
        pd.DataFrame(del_rows).to_csv(shard / "signal_delivery_log.csv", index=False)
        pd.DataFrame(rec_rows).to_csv(shard / "receiver_signal_action_log.csv", index=False)
        pd.DataFrame(utt_rows).to_csv(shard / "utterance_sequence_log.csv", index=False)
        pd.DataFrame(pop_rows).to_csv(shard / "language_population_episode_summary.csv", index=False)
        pd.DataFrame(run_rows).to_csv(shard / "run_index.csv", index=False)


# -----------------------------
# Main runner
# -----------------------------

def run_reanalysis(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir).expanduser().resolve()
    if args.clean:
        safe_unlink_dir(outdir)
    else:
        ensure_dir(outdir)
    if args.desktop_outdir:
        ensure_dir(Path(args.desktop_outdir).expanduser().resolve())
    errors: List[str] = []
    try:
        db = load_data(Path(args.source_root).expanduser().resolve(), chunksize=args.chunksize)
        outputs: Dict[str, pd.DataFrame] = {}
        eprint("[analysis] 00 audit")
        audit = analysis_00_audit(db)
        outputs["00_LANGUAGE_ACQUISITION_AUDIT.csv"] = audit
        eprint("[analysis] 01 development")
        outputs["01_development_by_time_bin.csv"] = analysis_01_development(db)
        eprint("[analysis] 02 sender grounding")
        outputs["02_sender_grounding_by_condition.csv"] = analysis_02_sender_grounding(db)
        eprint("[analysis] 03 receiver action effects")
        outputs["03_receiver_action_effects_by_action.csv"] = analysis_03_receiver_action_effects(db)
        eprint("[analysis] 04 nonreflexive context")
        outputs["04_nonreflexive_context_dependence.csv"] = analysis_04_nonreflexive_context(db)
        eprint("[analysis] 05 internal grounding")
        outputs["05_internal_grounding_vs_external_label.csv"] = analysis_05_internal_grounding(db, outputs["02_sender_grounding_by_condition.csv"])
        eprint("[analysis] 06 sequence/order/ngram null")
        outputs["06_sequence_order_ngram_null_test.csv"] = analysis_06_sequence_ngram(db, n_null=args.n_null)
        eprint("[analysis] 07 bigram/trigram action prediction")
        outputs["07_bigram_trigram_action_prediction.csv"] = analysis_07_bigram_trigram_action_prediction(db)
        eprint("[analysis] 08 sequence to outcome")
        outputs["08_sequence_to_outcome_relevance.csv"] = analysis_08_sequence_to_outcome(db, outputs["07_bigram_trigram_action_prediction.csv"])
        eprint("[analysis] 09 shared codebook")
        outputs["09_shared_codebook_within_population.csv"] = analysis_09_shared_codebook(db)
        eprint("[analysis] 10 language-likeness judgement")
        outputs["10_language_likeness_judgement.csv"] = aggregate_condition_scores(
            outputs["01_development_by_time_bin.csv"],
            outputs["02_sender_grounding_by_condition.csv"],
            outputs["03_receiver_action_effects_by_action.csv"],
            outputs["04_nonreflexive_context_dependence.csv"],
            outputs["05_internal_grounding_vs_external_label.csv"],
            outputs["06_sequence_order_ngram_null_test.csv"],
            outputs["08_sequence_to_outcome_relevance.csv"],
            outputs["09_shared_codebook_within_population.csv"],
        )
        # Write outputs before report.
        for name, df in outputs.items():
            write_csv(outdir / name, df)
        write_report(outdir, audit, outputs, outputs["10_language_likeness_judgement.csv"], errors)
        # Desktop copy/symlink. Avoid raw copies of huge data; only copy small summary files.
        if args.desktop_outdir:
            dd = Path(args.desktop_outdir).expanduser().resolve()
            ensure_dir(dd)
            for name in EXPECTED_OUTPUTS:
                src = outdir / name
                if src.exists():
                    shutil.copy2(src, dd / name)
        eprint(f"[complete] outputs written to {outdir}")
        return 0
    except Exception as e:
        errors.append(f"fatal: {type(e).__name__}: {e}")
        errors.append(traceback.format_exc())
        # Always write failure report and audit if possible.
        try:
            audit = pd.DataFrame([{"status": "fatal_error", "error": str(e), "traceback": traceback.format_exc()}])
            write_csv(outdir / "00_LANGUAGE_ACQUISITION_AUDIT.csv", audit)
            outputs = {"00_LANGUAGE_ACQUISITION_AUDIT.csv": audit}
            # Write empty expected outputs so downstream workflow never fails on missing file.
            for name in EXPECTED_OUTPUTS:
                if name.endswith(".csv") and name not in outputs:
                    write_csv(outdir / name, pd.DataFrame([{"status": "not_generated_due_to_fatal_error", "error": str(e)}]))
            write_report(outdir, audit, outputs, pd.DataFrame(), errors)
        except Exception:
            pass
        eprint("[fatal] " + str(e))
        return 2


def run_mock_self_test(args: argparse.Namespace) -> int:
    mock_out = Path(args.mock_outdir or "/tmp/v3_language_likeness_mock").expanduser().resolve()
    source = mock_out / "source"
    out = mock_out / "out"
    desk = mock_out / "desktop"
    create_mock_source(source)
    ns = argparse.Namespace(
        source_root=str(source), outdir=str(out), desktop_outdir=str(desk), clean=True,
        chunksize=None, n_null=10, mock_self_test=False, mock_outdir=None, estimate_only=False,
    )
    code = run_reanalysis(ns)
    missing = [name for name in EXPECTED_OUTPUTS if not (out / name).exists()]
    if missing:
        raise RuntimeError("mock self-test missing outputs: " + ", ".join(missing))
    # Basic sanity: report and judgement have rows.
    j = pd.read_csv(out / "10_language_likeness_judgement.csv")
    if j.empty:
        raise RuntimeError("mock judgement is empty")
    print(f"[ok] mock-self-test passed: {out}")
    return code


def print_estimate() -> int:
    info = {
        "purpose": "within-population language-likeness judgement after V3 16000 language-acquisition run",
        "expected_input_root": "~/V11_LANGUAGE_LOCAL/V11_LANGUAGE_DEV_16000_LANGUAGE_ACQUISITION_RAW",
        "expected_outputs": EXPECTED_OUTPUTS,
        "fixed_time_bins": TIME_BIN_LABELS,
        "primary_gate": "within-population grounding + receiver effect + nonreflexivity + grammar-like sequence + functional relevance + sharing",
        "not_primary_gate": ["cross-population L7/L7b immediate transfer", "language_like_level", "maxL", "summed level score"],
        "robustness_features": [
            "recursive file discovery", "column alias normalization", "empty-safe output writing", "final report always written", "receiver_signal_action_log alias support", "deterministic ngram null", "BH FDR for sequence/action prediction p-values"
        ],
    }
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Locked language-likeness reanalysis for V3 16000 language-acquisition outputs.")
    p.add_argument("--source-root", default="~/V11_LANGUAGE_LOCAL/V11_LANGUAGE_DEV_16000_LANGUAGE_ACQUISITION_RAW", help="Root containing scenario shard outputs from v5/v6 16000 language acquisition run.")
    p.add_argument("--outdir", default="~/V11_LANGUAGE_LOCAL/V11_LANGUAGE_DEV_16000_LANGUAGE_LIKENESS_REANALYSIS", help="Output directory for CSVs and report.")
    p.add_argument("--desktop-outdir", default="~/Desktop/V11_LANGUAGE_DEV_16000_LANGUAGE_LIKENESS_REANALYSIS", help="Optional desktop copy directory for output summaries.")
    p.add_argument("--clean", action="store_true", help="Delete and recreate outdir before running.")
    p.add_argument("--chunksize", type=int, default=None, help="Optional pandas chunksize for very large CSVs. Default reads selected columns directly.")
    p.add_argument("--n-null", type=int, default=50, help="Deterministic within-sequence shuffle null repetitions for sequence/order test.")
    p.add_argument("--estimate-only", action="store_true", help="Print locked analysis plan and exit.")
    p.add_argument("--mock-self-test", action="store_true", help="Create mock data and run all outputs as a self-test.")
    p.add_argument("--mock-outdir", default="/tmp/v3_language_likeness_mock", help="Mock self-test directory.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.estimate_only:
        return print_estimate()
    if args.mock_self_test:
        return run_mock_self_test(args)
    return run_reanalysis(args)


if __name__ == "__main__":
    raise SystemExit(main())
