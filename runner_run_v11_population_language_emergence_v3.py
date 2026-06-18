#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11 population language-emergence validation runner V3
===================================================

Purpose
-------
Instantiate the locked DARCA TRUE 3D integrated task battery v11 as a fixed
population of autonomous-life cores in a shared TRUE 3D world, without rewriting
v11 internals. The purpose is to test whether anonymous, non-semantic social
signals become internally grounded, receiver-effective, shared, referential,
partially compositional, and cross-play transferable language-like communication
precursors.

Core guardrail
--------------
This script imports v11 from --v11-file and DARCA from --darca-file. It does not
edit v11 classes or the DARCA core. It uses v11 classes and functions directly:
TrueWorld3D, AgentMemory, QualitativeValenceLayer, PhysicalLawLayer,
SocialSignalLayer, DarcaWrapper, signal_for_darca, darca_action, and the one-
factor 3D scenario design.

Main design
-----------
Main language-emergence battery fixes population size and density to avoid
confounding language-like signal grounding with crowding:

N = 8, density_phi = 1.0, arm = DARCA_Q_PHYSICS_SOCIAL,
21 one-factor environments, 5 tasks, 5 communication conditions,
50 seeds, 1000 steps.

Primary anti-reflex and proto-syntax analyses in V3
--------------------------------
V3 treats reflex rejection, state-perturbation response divergence, asymmetric
hidden-state reference, internal-vs-external grounding, future-outcome reference,
ordered n-gram embodied success, functional code alignment, and matched-control
counterfactual proxy tests as primary language-like criteria, not as optional
supplementary analyses.

Communication conditions
------------------------
PRIVATE: signal is generated and logged but not delivered.
RANDOM_MATCHED: delivery timing, distance, and intensity are preserved, but
    channel identity is randomized.
SHUFFLED: emitted packets are shuffled within step before delivery.
RECEIVER_LEARN: real packets are delivered and receivers learn signal-outcome
    associations from their own embodied consequences.
FULL_INTERACTIVE: RECEIVER_LEARN plus sender-side channel preference is updated
    from receiver embodied consequences.

Typical smoke run
-----------------
python3 -u run_v11_population_language_emergence_v3.py \
  --v11-file ~/Downloads/darca_true_3d_integrated_task_battery_v11.py \
  --darca-file ~/Downloads/darca_v24_direct_rewrite_source.py \
  --outdir ~/Desktop/V11_LANGUAGE_SMOKE \
  --plan smoke \
  --overwrite

Main run
--------
python3 -u run_v11_population_language_emergence_v3.py \
  --v11-file ~/Downloads/darca_true_3d_integrated_task_battery_v11.py \
  --darca-file ~/Downloads/darca_v24_direct_rewrite_source.py \
  --outdir ~/Desktop/V11_LANGUAGE_MAIN \
  --plan main_language \
  --resume

Self-test without v11
---------------------
python3 run_v11_population_language_emergence_v3.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

EPS = 1e-12
LAGS = (1, 2, 3, 5, 10)
ACTION_CLASSES = ("move", "vertical", "scan", "rest", "approach_sender", "avoid_sender", "enter_unknown", "low_risk_move")
COMMUNICATION_CONDITIONS = ("PRIVATE", "RANDOM_MATCHED", "SHUFFLED", "RECEIVER_LEARN", "FULL_INTERACTIVE")
MAIN_ARM = "DARCA_Q_PHYSICS_SOCIAL"

# =============================================================================
# Generic utilities
# =============================================================================

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(x))))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(_format_row(r, fields))


def append_rows_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(_format_row(r, fields))


def _format_row(r: Dict[str, Any], fields: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in fields:
        v = r.get(k, "")
        if isinstance(v, (float, np.floating)):
            out[k] = f"{float(v):.10g}" if math.isfinite(float(v)) else ""
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        elif isinstance(v, (list, tuple, dict)):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = v
    return out


def normalized_entropy(vals: Sequence[Any], k_total: Optional[int] = None) -> float:
    vals2 = [str(v) for v in vals if str(v) != "nan"]
    if not vals2:
        return 0.0
    s = pd.Series(vals2)
    counts = s.value_counts().to_numpy(dtype=float)
    p = counts / max(counts.sum(), EPS)
    h = float(-(p * np.log2(p + EPS)).sum())
    if k_total is None:
        hmax = math.log2(len(counts)) if len(counts) > 1 else 1.0
    else:
        hmax = math.log2(max(2, int(k_total)))
    return float(h / hmax) if hmax > 0 else 0.0


def bootstrap_ci(x: Sequence[float], seed: int, n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    arr = np.asarray([safe_float(v, np.nan) for v in x], dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0, 0.0
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(int(n_boot), len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def describe_vector(x: Sequence[float], seed: int, prefix: str = "") -> Dict[str, Any]:
    arr = np.asarray([safe_float(v, np.nan) for v in x], dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            f"{prefix}n": 0, f"{prefix}mean": 0.0, f"{prefix}sd": 0.0, f"{prefix}sem": 0.0,
            f"{prefix}ci95_low": 0.0, f"{prefix}ci95_high": 0.0, f"{prefix}prop_positive": 0.0,
            f"{prefix}cohen_dz": 0.0,
        }
    lo, hi = bootstrap_ci(arr, seed=seed)
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    mean = float(arr.mean())
    return {
        f"{prefix}n": int(len(arr)),
        f"{prefix}mean": mean,
        f"{prefix}sd": sd,
        f"{prefix}sem": sd / math.sqrt(len(arr)) if len(arr) > 1 else 0.0,
        f"{prefix}ci95_low": lo,
        f"{prefix}ci95_high": hi,
        f"{prefix}prop_positive": float((arr > 0).mean()),
        f"{prefix}cohen_dz": mean / (sd + EPS),
    }


def corr_np(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]; b = b[mask]
    if len(a) < 3 or float(np.std(a)) <= EPS or float(np.std(b)) <= EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def mutual_information_discrete(x: Sequence[Any], y: Sequence[Any]) -> float:
    xs = [str(v) for v in x]
    ys = [str(v) for v in y]
    if len(xs) == 0 or len(xs) != len(ys):
        return 0.0
    n = len(xs)
    cx: Dict[str, int] = {}
    cy: Dict[str, int] = {}
    cxy: Dict[Tuple[str, str], int] = {}
    for a, b in zip(xs, ys):
        cx[a] = cx.get(a, 0) + 1
        cy[b] = cy.get(b, 0) + 1
        cxy[(a, b)] = cxy.get((a, b), 0) + 1
    mi = 0.0
    for (a, b), c in cxy.items():
        pxy = c / n
        px = cx[a] / n
        py = cy[b] / n
        mi += pxy * math.log2((pxy + EPS) / (px * py + EPS))
    return max(0.0, float(mi))




def conditional_mutual_information_discrete(x: Sequence[Any], y: Sequence[Any], z: Sequence[Any]) -> float:
    xs = [str(v) for v in x]
    ys = [str(v) for v in y]
    zs = [str(v) for v in z]
    if len(xs) == 0 or len(xs) != len(ys) or len(xs) != len(zs):
        return 0.0
    n = len(xs)
    xyz: Dict[Tuple[str, str, str], int] = {}
    xz: Dict[Tuple[str, str], int] = {}
    yz: Dict[Tuple[str, str], int] = {}
    zc: Dict[str, int] = {}
    for a, b, c in zip(xs, ys, zs):
        xyz[(a, b, c)] = xyz.get((a, b, c), 0) + 1
        xz[(a, c)] = xz.get((a, c), 0) + 1
        yz[(b, c)] = yz.get((b, c), 0) + 1
        zc[c] = zc.get(c, 0) + 1
    out = 0.0
    for (a, b, c), cnt in xyz.items():
        p_xyz = cnt / n
        p_z = zc[c] / n
        p_xz = xz[(a, c)] / n
        p_yz = yz[(b, c)] / n
        out += p_xyz * math.log2((p_xyz * p_z + EPS) / (p_xz * p_yz + EPS))
    return max(0.0, float(out))


def joint_label(*cols: Sequence[Any]) -> List[str]:
    if not cols:
        return []
    n = len(cols[0])
    return ["|".join(str(c[i]) for c in cols) for i in range(n)]


def auc_binary_score(y_true: Sequence[Any], y_score: Sequence[float]) -> float:
    y = np.asarray([int(safe_float(v) > 0.5) for v in y_true], dtype=int)
    s = np.asarray([safe_float(v, np.nan) for v in y_score], dtype=float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    wins = 0.0
    total = 0.0
    for ps in pos:
        wins += float(np.sum(ps > neg)) + 0.5 * float(np.sum(ps == neg))
        total += len(neg)
    return float(wins / max(total, EPS))

def bin_series(values: Sequence[float], labels: Tuple[str, str, str] = ("low", "mid", "high")) -> List[str]:
    arr = np.asarray([safe_float(v, np.nan) for v in values], dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0 or float(np.std(finite)) <= EPS:
        return ["flat" for _ in arr]
    q1, q2 = np.quantile(finite, [1 / 3, 2 / 3])
    out: List[str] = []
    for v in arr:
        if not math.isfinite(float(v)):
            out.append("nan")
        elif v <= q1:
            out.append(labels[0])
        elif v <= q2:
            out.append(labels[1])
        else:
            out.append(labels[2])
    return out


def odd_round(x: float, min_value: int = 5) -> int:
    n = max(min_value, int(round(x)))
    if n % 2 == 0:
        n += 1
    return int(n)


class Logger:
    def __init__(self, outdir: Path, filename: str = "population_language_runner.log"):
        self.outdir = ensure_dir(outdir)
        self.t0 = time.time()
        self.path = outdir / filename
        self.path.write_text(f"V11 population language-emergence runner\nStarted: {now()}\n" + "=" * 92 + "\n", encoding="utf-8")

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:10.2f}s] {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class QuietV11Logger:
    def __init__(self, parent: Logger):
        self.parent = parent

    def log(self, msg: str) -> None:
        if str(msg).startswith("ERROR") or "Interrupted" in str(msg):
            self.parent.log("[v11] " + str(msg))

# =============================================================================
# One-factor environment scenarios copied from the locked gradient design
# =============================================================================

@dataclass(frozen=True)
class Scenario:
    scenario: str
    description: str
    world_size: int = 11
    z_size: int = 7
    danger_frac: float = 0.075
    resource_frac: float = 0.065
    unknown_frac: float = 0.120
    rest_count: int = 4
    false_resource_frac: float = 0.030
    hidden_rest_frac: float = 0.025
    friction_frac: float = 0.060
    crisis_interval: int = 180
    observation_radius: int = 1
    world_seed_offset: int = 0
    gradient_factor: str = "baseline"
    gradient_level: float = 1.0
    gradient_param: str = "none"
    gradient_value: float = 1.0


def _base_params() -> Dict[str, Any]:
    return dict(
        world_size=11,
        z_size=7,
        danger_frac=0.075,
        resource_frac=0.065,
        unknown_frac=0.120,
        rest_count=4,
        false_resource_frac=0.030,
        hidden_rest_frac=0.025,
        friction_frac=0.060,
        crisis_interval=180,
        observation_radius=1,
    )


def built_in_scenarios() -> List[Scenario]:
    base = _base_params()
    scenarios: List[Scenario] = [Scenario(
        "baseline_3d",
        "Reference TRUE 3D ecology; common baseline for all one-factor gradients.",
        **base,
        world_seed_offset=0,
        gradient_factor="baseline",
        gradient_level=1.0,
        gradient_param="none",
        gradient_value=1.0,
    )]
    levels = [0.50, 0.75, 1.25, 1.50]
    seed_offset = 10000

    def add(name: str, desc: str, factor: str, param: str, level: float, updates: Dict[str, Any]) -> None:
        nonlocal seed_offset
        d = dict(base)
        d.update(updates)
        scenarios.append(Scenario(
            name,
            desc,
            **d,
            world_seed_offset=seed_offset,
            gradient_factor=factor,
            gradient_level=float(level),
            gradient_param=param,
            gradient_value=float(updates.get(param, level)) if param in updates else float(level),
        ))
        seed_offset += 10000

    for lv in levels:
        add(f"resource_x{lv:.2f}".replace(".", "p"), f"Resource-density gradient {lv:.2f}× baseline.", "resource_density", "resource_frac", lv, {"resource_frac": max(0.001, base["resource_frac"] * lv)})
    for lv in levels:
        add(f"danger_x{lv:.2f}".replace(".", "p"), f"Danger-density gradient {lv:.2f}× baseline.", "danger_density", "danger_frac", lv, {"danger_frac": min(0.40, max(0.001, base["danger_frac"] * lv))})
    for lv in levels:
        add(f"friction_x{lv:.2f}".replace(".", "p"), f"Friction/slip gradient {lv:.2f}× baseline.", "friction_slip", "friction_frac", lv, {"friction_frac": min(0.40, max(0.001, base["friction_frac"] * lv))})
    for lv in levels:
        add(f"unknown_x{lv:.2f}".replace(".", "p"), f"Unknown/ambiguity gradient {lv:.2f}× baseline.", "unknown_ambiguity", "unknown_frac", lv, {"unknown_frac": min(0.45, max(0.001, base["unknown_frac"] * lv))})
    for lv, z in [(0.50, 4), (0.75, 5), (1.25, 9), (1.50, 11)]:
        add(f"vertical_x{lv:.2f}".replace(".", "p"), f"Vertical-complexity gradient {lv:.2f}× baseline z-size.", "vertical_complexity", "z_size", lv, {"z_size": int(z)})
    return scenarios


def scenario_names_for_factors(factors: Sequence[str]) -> List[str]:
    wanted = ["baseline_3d"]
    fs = set(factors)
    for sc in built_in_scenarios():
        if sc.gradient_factor in fs and sc.scenario != "baseline_3d":
            wanted.append(sc.scenario)
    out: List[str] = []
    for x in wanted:
        if x not in out:
            out.append(x)
    return out


TASKS_DEFAULT = "viability,delayed_memory,exploration_recovery,physics_adaptation,social_reappraisal"
COMM_DEFAULT = "PRIVATE,RANDOM_MATCHED,SHUFFLED,RECEIVER_LEARN,FULL_INTERACTIVE"


def plan_defaults(plan: str, factors: str = "") -> Dict[str, Any]:
    all_scenarios = [sc.scenario for sc in built_in_scenarios()]
    if plan == "smoke":
        return dict(
            seeds=2,
            steps=80,
            tasks="physics_adaptation,social_reappraisal",
            arms=MAIN_ARM,
            scenarios=["baseline_3d", "unknown_x1p50"],
            comm="PRIVATE,SHUFFLED,RECEIVER_LEARN,FULL_INTERACTIVE",
            population_sizes="4",
            density_phi=1.0,
        )
    if plan == "quick_language":
        return dict(
            seeds=5,
            steps=200,
            tasks="physics_adaptation,exploration_recovery,social_reappraisal",
            arms=MAIN_ARM,
            scenarios=["baseline_3d", "unknown_x1p50", "danger_x1p50", "vertical_x1p50"],
            comm=COMM_DEFAULT,
            population_sizes="8",
            density_phi=1.0,
        )
    if plan == "main_language":
        return dict(
            seeds=50,
            steps=1000,
            tasks=TASKS_DEFAULT,
            arms=MAIN_ARM,
            scenarios=all_scenarios,
            comm=COMM_DEFAULT,
            population_sizes="8",
            density_phi=1.0,
        )
    if plan == "module_contribution":
        return dict(
            seeds=50,
            steps=1000,
            tasks="physics_adaptation,exploration_recovery,social_reappraisal",
            arms="DARCA_ONLY,DARCA_Q,DARCA_PHYSICS,DARCA_Q_PHYSICS,DARCA_Q_SOCIAL,DARCA_Q_PHYSICS_SOCIAL",
            scenarios=["baseline_3d", "unknown_x1p50", "danger_x1p50", "vertical_x1p50", "resource_x0p50"],
            comm="SHUFFLED,RECEIVER_LEARN,FULL_INTERACTIVE",
            population_sizes="8",
            density_phi=1.0,
        )
    if plan == "size_robustness":
        return dict(
            seeds=30,
            steps=1000,
            tasks="physics_adaptation,exploration_recovery,social_reappraisal",
            arms=MAIN_ARM,
            scenarios=["baseline_3d", "unknown_x1p50", "danger_x1p50", "vertical_x1p50"],
            comm="SHUFFLED,FULL_INTERACTIVE",
            population_sizes="4,8,16",
            density_phi=1.0,
        )
    if plan == "selected_factors":
        fs = [x.strip() for x in factors.split(",") if x.strip()] or ["unknown_ambiguity", "vertical_complexity"]
        return dict(
            seeds=50,
            steps=1000,
            tasks=TASKS_DEFAULT,
            arms=MAIN_ARM,
            scenarios=scenario_names_for_factors(fs),
            comm=COMM_DEFAULT,
            population_sizes="8",
            density_phi=1.0,
        )
    raise ValueError(plan)

# =============================================================================
# Imported v11 module and runtime data structures
# =============================================================================

def import_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import file: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class SignalPacket:
    sender_id: int
    channel: int
    intensity: float
    sender_pos: Tuple[int, int, int]
    emitted_step: int
    raw_channel: int = -1
    communication_condition: str = ""
    # analysis-only source fields. Receivers never see these values.
    sender_h: float = 0.0
    sender_Q: float = 0.0
    sender_Q_R: float = 0.0
    sender_Q_G: float = 0.0
    sender_physics_score: float = 0.0
    sender_relief: float = 0.0
    sender_safe_surprise: float = 0.0
    sender_self_appraisal_gap: float = 0.0
    sender_q_relief: float = 0.0
    sender_danger_pressure: float = 0.0
    sender_resource_pressure: float = 0.0
    sender_unknown_pressure: float = 0.0
    sender_vertical_pressure: float = 0.0
    sender_friction_pressure: float = 0.0
    sender_event: str = ""
    sender_action: str = ""


@dataclass
class EmbodiedSignalMemory:
    n_channels: int = 5
    lr: float = 0.045
    decay: float = 0.002
    action_value: Dict[int, Dict[str, float]] = field(default_factory=dict)
    channel_value: Dict[int, float] = field(default_factory=dict)
    channel_risk: Dict[int, float] = field(default_factory=dict)
    channel_recovery: Dict[int, float] = field(default_factory=dict)
    channel_exploration: Dict[int, float] = field(default_factory=dict)
    channel_vertical: Dict[int, float] = field(default_factory=dict)
    channel_unknown: Dict[int, float] = field(default_factory=dict)
    channel_count: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for ch in range(int(self.n_channels)):
            self.action_value.setdefault(ch, {a: 0.0 for a in ACTION_CLASSES})
            self.channel_value.setdefault(ch, 0.0)
            self.channel_risk.setdefault(ch, 0.0)
            self.channel_recovery.setdefault(ch, 0.0)
            self.channel_exploration.setdefault(ch, 0.0)
            self.channel_vertical.setdefault(ch, 0.0)
            self.channel_unknown.setdefault(ch, 0.0)
            self.channel_count.setdefault(ch, 0)

    def action_bias(self, channels: Sequence[int], action_classes: Sequence[str], beta: float) -> float:
        if not channels:
            return 0.0
        vals: List[float] = []
        for ch in channels:
            if ch < 0:
                continue
            av = self.action_value.get(int(ch), {})
            vals.extend([safe_float(av.get(ac)) for ac in action_classes])
        return float(beta * np.mean(vals)) if vals else 0.0

    def update(self, channel: int, action_classes: Sequence[str], embodied_value: float, outcome_info: Dict[str, float]) -> None:
        ch = int(channel)
        if ch < 0:
            return
        self.__post_init__()
        # slow decay prevents early frozen assignments while preserving memory.
        for ac in ACTION_CLASSES:
            self.action_value[ch][ac] *= (1.0 - self.decay)
        self.channel_value[ch] *= (1.0 - self.decay)
        self.channel_risk[ch] *= (1.0 - self.decay)
        self.channel_recovery[ch] *= (1.0 - self.decay)
        self.channel_exploration[ch] *= (1.0 - self.decay)
        self.channel_vertical[ch] *= (1.0 - self.decay)
        self.channel_unknown[ch] *= (1.0 - self.decay)

        val = clip(float(embodied_value), -1.0, 1.0)
        for ac in action_classes:
            if ac in self.action_value[ch]:
                self.action_value[ch][ac] = (1.0 - self.lr) * self.action_value[ch][ac] + self.lr * val
        self.channel_value[ch] = (1.0 - self.lr) * self.channel_value[ch] + self.lr * val
        self.channel_risk[ch] = (1.0 - self.lr) * self.channel_risk[ch] + self.lr * safe_float(outcome_info.get("risk", 0.0))
        self.channel_recovery[ch] = (1.0 - self.lr) * self.channel_recovery[ch] + self.lr * safe_float(outcome_info.get("recovery", 0.0))
        self.channel_exploration[ch] = (1.0 - self.lr) * self.channel_exploration[ch] + self.lr * safe_float(outcome_info.get("exploration", 0.0))
        self.channel_vertical[ch] = (1.0 - self.lr) * self.channel_vertical[ch] + self.lr * safe_float(outcome_info.get("vertical", 0.0))
        self.channel_unknown[ch] = (1.0 - self.lr) * self.channel_unknown[ch] + self.lr * safe_float(outcome_info.get("unknown", 0.0))
        self.channel_count[ch] = int(self.channel_count.get(ch, 0)) + 1

    def rows(self, scenario: str, task: str, arm: str, comm: str, seed_id: int, agent_id: int, step: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for ch in range(int(self.n_channels)):
            base = {
                "scenario": scenario, "task": task, "arm": arm, "communication_condition": comm,
                "seed_id": seed_id, "agent_id": agent_id, "step": step, "channel": ch,
                "channel_count": self.channel_count.get(ch, 0),
                "channel_value": self.channel_value.get(ch, 0.0),
                "channel_risk": self.channel_risk.get(ch, 0.0),
                "channel_recovery": self.channel_recovery.get(ch, 0.0),
                "channel_exploration": self.channel_exploration.get(ch, 0.0),
                "channel_vertical": self.channel_vertical.get(ch, 0.0),
                "channel_unknown": self.channel_unknown.get(ch, 0.0),
            }
            for ac in ACTION_CLASSES:
                base[f"action_value_{ac}"] = self.action_value.get(ch, {}).get(ac, 0.0)
            out.append(base)
        return out


@dataclass
class AgentRuntime:
    agent_id: int
    mem: Any
    pos: Tuple[int, int, int]
    q_layer: Optional[Any]
    physics: Optional[Any]
    social: Optional[Any]
    darca: Optional[Any]
    terminal: bool = False
    signal_inbox: List[SignalPacket] = field(default_factory=list)
    pending_traces: List[Dict[str, Any]] = field(default_factory=list)
    signal_memory: EmbodiedSignalMemory = field(default_factory=EmbodiedSignalMemory)
    utterance_buffer: List[int] = field(default_factory=list)
    sender_channel_preference: Dict[int, float] = field(default_factory=dict)
    records: List[Dict[str, Any]] = field(default_factory=list)

# =============================================================================
# Runtime helpers
# =============================================================================

def scaled_world_geometry(base_size: int, base_z: int, population_size: int, density_phi: float) -> Tuple[int, int, float]:
    # density_phi=1.0 preserves approximate single-agent area per agent.
    effective_area_scale = max(1.0, float(population_size) * float(density_phi))
    world_size = odd_round(float(base_size) * math.sqrt(effective_area_scale), min_value=int(base_size))
    return int(world_size), int(base_z), float(effective_area_scale)


def make_base_args(args: argparse.Namespace, scenario: Scenario, seed: int, world_seed: int, world_size: int, z_size: int, rest_count: int) -> SimpleNamespace:
    return SimpleNamespace(
        outdir=str(args.outdir),
        darca_file=str(args.darca_file),
        tasks=args.tasks,
        arms=args.arms,
        episodes=args.seeds,
        steps=args.steps,
        seed=seed,
        world_seed=world_seed,
        world_size=int(world_size),
        z_size=int(z_size),
        danger_frac=float(scenario.danger_frac),
        resource_frac=float(scenario.resource_frac),
        unknown_frac=float(scenario.unknown_frac),
        rest_count=int(rest_count),
        false_resource_frac=float(scenario.false_resource_frac),
        hidden_rest_frac=float(scenario.hidden_rest_frac),
        friction_frac=float(scenario.friction_frac),
        crisis_interval=int(scenario.crisis_interval),
        observation_radius=int(scenario.observation_radius),
        terminal_h=float(args.terminal_h),
        theta=float(args.theta),
        causal_horizon=int(args.causal_horizon),
        recurrent_N=int(args.recurrent_N),
        block_consult_below_h=0.22,
        max_consecutive_scans=int(args.max_consecutive_scans),
        progress_every=10**9,
        q_probe_train_steps=0,
        q_probe_n_state=0,
        q_probe_n_history=0,
        q_agency_probe_n=0,
        overwrite=True,
    )


def choose_start_positions(v11: Any, world: Any, n_agents: int) -> List[Tuple[int, int, int]]:
    center = world.start
    candidates: List[Tuple[float, Tuple[int, int, int]]] = []
    for p, tile in world.grid.items():
        try:
            actual = world.actual_kind(p, 0)
        except Exception:
            actual = tile.kind
        if actual == v11.T_DANGER or tile.kind == v11.T_UNKNOWN:
            continue
        dist = abs(p[0] - center[0]) + abs(p[1] - center[1]) + abs(p[2] - center[2])
        z_penalty = 0.15 * abs(p[2] - center[2])
        rest_bonus = -0.25 if actual == v11.T_REST else 0.0
        candidates.append((dist + z_penalty + rest_bonus, p))
    candidates.sort(key=lambda x: (x[0], x[1][2], x[1][0], x[1][1]))
    starts: List[Tuple[int, int, int]] = []
    # Enforce a small minimum separation when possible.
    for _, p in candidates:
        if any(manhattan(p, q) < 2 for q in starts):
            continue
        starts.append(p)
        if len(starts) >= n_agents:
            return starts
    for _, p in candidates:
        if p not in starts:
            starts.append(p)
        if len(starts) >= n_agents:
            return starts
    raise RuntimeError(f"Could not allocate {n_agents} safe unique start positions.")


def make_agents(v11: Any, darca_module: Any, arm: str, task: str, task_args: SimpleNamespace, seed_id: int, starts: List[Tuple[int, int, int]]) -> List[AgentRuntime]:
    use_q, use_physics, use_social = v11.arm_modules(arm)
    agents: List[AgentRuntime] = []
    for aid, pos in enumerate(starts):
        seed = int(task_args.seed + seed_id * 1009 + v11.hash_arm(arm) * 17 + v11.task_hash(task) * 31 + aid * 100003)
        mem = v11.AgentMemory()
        mem.known[pos] = "REST"
        mem.visited.add(pos)
        q_layer = v11.QualitativeValenceLayer(
            lesion=("q_lesion" if arm == "DARCA_Q_LESION" else "memory_lesion" if arm == "DARCA_Q_MEMORY_LESION" else "agency_lesion" if arm == "DARCA_Q_AGENCY_LESION" else "none"),
            seed=seed + 101,
        ) if use_q else None
        physics = v11.PhysicalLawLayer(lesion=(arm == "DARCA_PHYSICS_LESION")) if use_physics else None
        social = v11.SocialSignalLayer(seed=seed + 4049) if use_social else None
        darca = v11.DarcaWrapper(darca_module, seed, task_args.theta, task_args.causal_horizon, task_args.recurrent_N)
        n_channels = int(getattr(social, "n_signals", 5)) if social is not None else 5
        mem_obj = EmbodiedSignalMemory(n_channels=n_channels)
        pref = {ch: 0.0 for ch in range(n_channels)}
        agents.append(AgentRuntime(aid, mem, pos, q_layer, physics, social, darca, False, [], [], mem_obj, [], pref, []))
    return agents


def manhattan(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def action_target(v11: Any, world: Any, pos: Tuple[int, int, int], action: str) -> Tuple[int, int, int]:
    if action in v11.MOVE_ACTIONS:
        di, dj, dk = v11.DIRS[action]
        return (pos[0] + di, pos[1] + dj, pos[2] + dk)
    return pos


def relative_direction_bin(sender_pos: Tuple[int, int, int], receiver_pos: Tuple[int, int, int]) -> str:
    di = sender_pos[0] - receiver_pos[0]
    dj = sender_pos[1] - receiver_pos[1]
    dk = sender_pos[2] - receiver_pos[2]
    axes = [(abs(di), "S" if di > 0 else "N" if di < 0 else ""), (abs(dj), "E" if dj > 0 else "W" if dj < 0 else ""), (abs(dk), "UP" if dk > 0 else "DOWN" if dk < 0 else "")]
    axes.sort(reverse=True)
    return axes[0][1] or "HERE"


def distance_bin(d: int) -> str:
    if d <= 1:
        return "near"
    if d <= 3:
        return "mid"
    return "far"


def action_classes_for_action(v11: Any, action: str, old_pos: Tuple[int, int, int], target: Tuple[int, int, int], signal_ctx: Dict[str, Any]) -> List[str]:
    classes: List[str] = []
    if action in getattr(v11, "MOVE_ACTIONS", []):
        classes.append("move")
    if action in ("MOVE_UP", "MOVE_DOWN"):
        classes.append("vertical")
    if action == "SCAN":
        classes.append("scan")
    if action == "REST":
        classes.append("rest")
    nearest = signal_ctx.get("nearest_sender_pos")
    if nearest is not None and action in getattr(v11, "MOVE_ACTIONS", []):
        before = manhattan(old_pos, nearest)
        after = manhattan(target, nearest)
        if after < before:
            classes.append("approach_sender")
        elif after > before:
            classes.append("avoid_sender")
    if signal_ctx.get("candidate_enters_unknown", False):
        classes.append("enter_unknown")
    if action in getattr(v11, "MOVE_ACTIONS", []) and not classes:
        classes.append("low_risk_move")
    return classes or ["move"]


def compute_signal_context(agent: AgentRuntime, step: int, v11: Any) -> Dict[str, Any]:
    packets = list(agent.signal_inbox)
    if not packets:
        return {
            "n_signals": 0, "channels": [], "dominant_channel": -1, "channel_entropy": 0.0,
            "coupling_t": 0.0, "sigma_t": 0.0, "mean_intensity": 0.0,
            "nearest_sender_pos": None, "nearest_sender_distance": 999,
        }
    channels = [int(p.channel) for p in packets if int(p.channel) >= 0]
    intensities = [safe_float(p.intensity) for p in packets]
    counts = pd.Series(channels).value_counts() if channels else pd.Series(dtype=int)
    dom = int(counts.index[0]) if len(counts) else -1
    ent = normalized_entropy(channels, k_total=max(2, int(max(channels) + 1) if channels else 5))
    distances = [manhattan(p.sender_pos, agent.pos) for p in packets]
    nearest_idx = int(np.argmin(distances)) if distances else 0
    coupling = clip(sum(i / (1.0 + d) for i, d in zip(intensities, distances)), 0.0, 1.0)
    sigma = clip(ent * min(1.0, len(channels) / 3.0), 0.0, 1.0)
    return {
        "n_signals": len(packets),
        "channels": channels,
        "dominant_channel": dom,
        "channel_entropy": ent,
        "coupling_t": coupling,
        "sigma_t": sigma,
        "mean_intensity": float(np.mean(intensities)) if intensities else 0.0,
        "nearest_sender_pos": packets[nearest_idx].sender_pos if packets else None,
        "nearest_sender_distance": distances[nearest_idx] if distances else 999,
    }


def choose_signal_biased_action(v11: Any, world: Any, agent: AgentRuntime, base_action: str, rng: random.Random, step: int, signal_ctx: Dict[str, Any], beta: float, margin: float) -> Tuple[str, str, float]:
    channels = signal_ctx.get("channels", [])
    if not channels or beta <= 0.0:
        return base_action, "no_signal_bias", 0.0
    candidates = list(v11.MOVE_ACTIONS) + ["REST", "SCAN"]
    scored: List[Tuple[float, str, float]] = []
    for a in candidates:
        try:
            base = v11.score_candidate_base(world, agent.pos, agent.mem, a, step)
        except Exception:
            base = 0.0 if a == base_action else -0.25
        tgt = action_target(v11, world, agent.pos, a)
        if a in v11.MOVE_ACTIONS and not world.in_bounds(tgt):
            continue
        pr_here = v11.local_pressures(world, agent.pos, agent.mem, step)
        if a in v11.MOVE_ACTIONS and world.in_bounds(tgt):
            tile = world.tile(tgt)
            signal_ctx["candidate_enters_unknown"] = int(agent.mem.known.get(tgt, tile.kind) == v11.T_UNKNOWN)
        else:
            signal_ctx["candidate_enters_unknown"] = 0
        ac = action_classes_for_action(v11, a, agent.pos, tgt, signal_ctx)
        sig_bias = agent.signal_memory.action_bias(channels, ac, beta=beta)
        # Keep the DARCA-chosen action slightly sticky. Signal is a weak modulator.
        sticky = 0.18 if a == base_action else 0.0
        q_cost = 0.0
        if agent.q_layer is not None:
            try:
                q_cost = 0.80 * safe_float(agent.q_layer.action_risk_modifier(a, world, agent.pos, agent.mem, step))
            except Exception:
                q_cost = 0.0
        p_cost = 0.0
        if agent.physics is not None:
            try:
                pred = agent.physics.predict(a, world, agent.pos, agent.mem, step)
                p_cost = 0.80 * safe_float(pred.get("pred_damage")) + 0.45 * safe_float(pred.get("pred_wall")) - 0.35 * safe_float(pred.get("pred_gain"))
            except Exception:
                p_cost = 0.0
        scored.append((base + sig_bias + sticky - q_cost - p_cost + rng.random() * 0.02, a, sig_bias))
    if not scored:
        return base_action, "signal_bias_no_candidates", 0.0
    scored.sort(reverse=True)
    best_score, best_action, best_bias = scored[0]
    base_score = next((s for s, a, _ in scored if a == base_action), scored[-1][0])
    if best_action != base_action and best_score > base_score + margin:
        return best_action, "receiver_signal_memory_bias", best_bias
    return base_action, "signal_bias_below_margin", best_bias



def state_perturbation_action_probe(v11: Any, world: Any, agent: AgentRuntime, base_action: str, signal_ctx: Dict[str, Any], step: int, beta: float) -> List[Dict[str, Any]]:
    """Side-effect-minimized receiver state perturbation assay.

    This does not call DARCA or update Q/physics/social learners. It uses the
    current receiver signal memory plus v11's outer shell candidate scoring while
    temporarily substituting receiver body_h. The same channel/context is thus
    tested under low, original, and high viability states without rewriting v11.
    """
    channels = list(signal_ctx.get("channels", []))
    if not channels:
        return []
    old_h = safe_float(agent.mem.body_h)
    states = [
        ("H_LOW", 0.30),
        ("H_ORIGINAL", old_h),
        ("H_HIGH", 0.75),
    ]
    rows: List[Dict[str, Any]] = []
    try:
        for label, hval in states:
            agent.mem.body_h = clip(hval, 0.0, 1.0)
            candidates: List[Tuple[float, str, float]] = []
            for a in list(v11.MOVE_ACTIONS) + ["REST", "SCAN"]:
                tgt = action_target(v11, world, agent.pos, a)
                if a in v11.MOVE_ACTIONS and not world.in_bounds(tgt):
                    continue
                try:
                    base = v11.score_candidate_base(world, agent.pos, agent.mem, a, step)
                except Exception:
                    base = 0.0 if a == base_action else -0.25
                tmp_ctx = dict(signal_ctx)
                if a in v11.MOVE_ACTIONS and world.in_bounds(tgt):
                    tile = world.tile(tgt)
                    tmp_ctx["candidate_enters_unknown"] = int(agent.mem.known.get(tgt, tile.kind) == v11.T_UNKNOWN)
                else:
                    tmp_ctx["candidate_enters_unknown"] = 0
                ac = action_classes_for_action(v11, a, agent.pos, tgt, tmp_ctx)
                sig_bias = agent.signal_memory.action_bias(channels, ac, beta=beta)
                sticky = 0.18 if a == base_action else 0.0
                # Lightweight shell terms only. No q_layer or physics calls here,
                # because those can mutate online learners.
                candidates.append((base + sig_bias + sticky, a, sig_bias))
            candidates.sort(reverse=True)
            if candidates:
                score, action, sigb = candidates[0]
            else:
                score, action, sigb = (0.0, base_action, 0.0)
            rows.append({
                "perturbation_label": label,
                "perturbed_body_h": hval,
                "assay_action": action,
                "assay_action_class": ",".join(action_classes_for_action(v11, action, agent.pos, action_target(v11, world, agent.pos, action), signal_ctx)),
                "assay_score": score,
                "assay_signal_bias": sigb,
            })
    finally:
        agent.mem.body_h = old_h
    # Add divergence relative to original-state action after all rows are known.
    original = next((r["assay_action"] for r in rows if r["perturbation_label"] == "H_ORIGINAL"), base_action)
    for r in rows:
        r["diverged_from_original_state"] = int(str(r.get("assay_action")) != str(original))
    return rows


def asymmetric_known_reference_flags(v11: Any, world: Any, sender: Optional[AgentRuntime], receiver: AgentRuntime, step: int) -> Dict[str, Any]:
    """Analysis-only sender/receiver information asymmetry flags.

    Receiver does not receive these fields. They are logged to test whether a
    channel can carry sender-known local information that is absent from the
    receiver's memory.
    """
    out = {
        "asym_known_count": 0,
        "asym_sender_known_danger": 0,
        "asym_sender_known_resource": 0,
        "asym_sender_known_rest": 0,
        "asym_sender_known_empty": 0,
    }
    if sender is None:
        return out
    positions = [sender.pos]
    for _, d in getattr(v11, "DIRS", {}).items():
        p = (sender.pos[0] + d[0], sender.pos[1] + d[1], sender.pos[2] + d[2])
        if world.in_bounds(p):
            positions.append(p)
    for p0 in positions:
        sk = sender.mem.known.get(p0, None)
        if sk is None:
            continue
        rk = receiver.mem.known.get(p0, None)
        receiver_absent = rk is None or rk == getattr(v11, "T_UNKNOWN", "UNKNOWN")
        if not receiver_absent:
            continue
        out["asym_known_count"] += 1
        if sk == getattr(v11, "T_DANGER", "DANGER"):
            out["asym_sender_known_danger"] = 1
        elif sk == getattr(v11, "T_RESOURCE", "RESOURCE"):
            out["asym_sender_known_resource"] = 1
        elif sk == getattr(v11, "T_REST", "REST"):
            out["asym_sender_known_rest"] = 1
        elif sk == getattr(v11, "T_EMPTY", "EMPTY"):
            out["asym_sender_known_empty"] = 1
    return out

def make_interference_outcome(v11: Any, world: Any, pos: Tuple[int, int, int], action: str, step: int) -> Any:
    vertical = action in ("MOVE_UP", "MOVE_DOWN")
    delta_h = -0.0035
    if world.is_crisis(step):
        delta_h -= 0.006
    if action in v11.MOVE_ACTIONS:
        delta_h -= 0.004 + (0.006 if vertical else 0.0)
    damage = 0.010 + (0.004 if vertical else 0.0)
    delta_h -= damage
    return v11.StepOutcome(
        delta_h=delta_h,
        damage=damage,
        resource_gain=0.0,
        recovery_gain=0.0,
        hit_wall=False,
        entered_unknown=False,
        revealed_type="",
        event="inter_agent_interference",
        vertical=vertical,
        friction=world.tile(pos).friction,
    )


def embodied_value_from_outcome(old_h: float, old_q: float, q_state: Dict[str, Any], outcome: Any, social_state: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    new_q = safe_float(q_state.get("Q", old_q))
    delta_h = safe_float(outcome.delta_h)
    increase_q = max(0.0, new_q - old_q)
    val = (
        1.00 * delta_h
        - 1.50 * safe_float(outcome.damage)
        + 0.80 * safe_float(outcome.recovery_gain)
        + 0.60 * safe_float(outcome.resource_gain)
        - 0.60 * float(getattr(outcome, "hit_wall", False))
        - 0.35 * increase_q
        + 0.25 * safe_float(social_state.get("q_relief"))
    )
    info = {
        "risk": clip(2.5 * safe_float(outcome.damage) + 0.4 * float(getattr(outcome, "hit_wall", False)) + 0.2 * increase_q, 0.0, 1.0),
        "recovery": clip(safe_float(outcome.recovery_gain) + safe_float(social_state.get("q_relief")), 0.0, 1.0),
        "exploration": float(getattr(outcome, "entered_unknown", False) or safe_float(outcome.resource_gain) > 0.0),
        "vertical": float(getattr(outcome, "vertical", False)),
        "unknown": float(getattr(outcome, "entered_unknown", False)),
    }
    return clip(val, -1.0, 1.0), info


def maybe_remap_channel_by_sender_preference(packet: SignalPacket, agent: AgentRuntime, rng: np.random.Generator, strength: float, n_channels: int) -> SignalPacket:
    if strength <= 0.0 or n_channels <= 1:
        return packet
    prefs = np.array([safe_float(agent.sender_channel_preference.get(ch)) for ch in range(n_channels)], dtype=float)
    if float(np.std(prefs)) <= EPS:
        return packet
    # Preferences only weakly bias channel reuse. This is an outer convention layer,
    # not semantic labeling.
    if rng.random() < strength:
        probs = np.exp(np.clip(prefs - np.max(prefs), -20, 20))
        probs = probs / max(probs.sum(), EPS)
        new_ch = int(rng.choice(np.arange(n_channels), p=probs))
        packet.raw_channel = packet.channel
        packet.channel = new_ch
    return packet


def apply_task_and_population_world(v11: Any, args: argparse.Namespace, scenario: Scenario, task: str, population_size: int, seed_id: int) -> Tuple[Any, SimpleNamespace, int, int, float]:
    world_size, z_size, effective_area_scale = scaled_world_geometry(scenario.world_size, scenario.z_size, population_size, args.density_phi)
    unscaled = make_base_args(args, scenario, args.seed + 1000003, args.world_seed + scenario.world_seed_offset, scenario.world_size, scenario.z_size, scenario.rest_count)
    task_unscaled = v11.apply_task_profile(unscaled, task)
    # Preserve environmental densities while scaling rest-site count with area.
    rest_count = max(1, int(round(float(task_unscaled.rest_count) * effective_area_scale)))
    base_args = make_base_args(args, scenario, args.seed + 1000003, args.world_seed + scenario.world_seed_offset, world_size, z_size, rest_count)
    task_args = v11.apply_task_profile(base_args, task)
    task_args.world_size = int(world_size)
    task_args.z_size = int(z_size)
    task_args.rest_count = int(rest_count)
    world_seed = int(args.world_seed + scenario.world_seed_offset + seed_id * 7919 + population_size * 31337 + int(round(args.density_phi * 1000)) * 101)
    world = v11.TrueWorld3D(
        world_seed,
        task_args.world_size,
        task_args.z_size,
        task_args.danger_frac,
        task_args.resource_frac,
        task_args.unknown_frac,
        task_args.rest_count,
        task_args.false_resource_frac,
        task_args.hidden_rest_frac,
        task_args.crisis_interval,
        task_args.observation_radius,
        task_args.friction_frac,
    )
    return world, task_args, world_size, z_size, effective_area_scale

# =============================================================================
# Population language episode
# =============================================================================

def run_population_language_episode(v11: Any, darca_module: Any, scenario: Scenario, task: str, arm: str, communication_condition: str, seed_id: int, population_size: int, args: argparse.Namespace, logger: Logger) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if communication_condition not in COMMUNICATION_CONDITIONS:
        raise ValueError(f"Unknown communication condition: {communication_condition}")
    base_seed = int(args.seed + seed_id * 1009 + stable_hash(scenario.scenario, task, arm, communication_condition, population_size) % 1000000)
    rng_np = np.random.default_rng(base_seed & 0xFFFFFFFF)
    rngs = [random.Random(base_seed + aid * 104729) for aid in range(population_size)]

    world, task_args, world_size, z_size, effective_area_scale = apply_task_and_population_world(v11, args, scenario, task, population_size, seed_id)
    starts = choose_start_positions(v11, world, population_size)
    agents = make_agents(v11, darca_module, arm, task, task_args, seed_id, starts)
    for ag in agents:
        ag.mem.known[ag.pos] = world.actual_kind(ag.pos, 0)
        ag.mem.visited.add(ag.pos)

    signal_event_rows: List[Dict[str, Any]] = []
    signal_delivery_rows: List[Dict[str, Any]] = []
    receiver_action_rows: List[Dict[str, Any]] = []
    state_perturbation_rows: List[Dict[str, Any]] = []
    utterance_rows: List[Dict[str, Any]] = []
    state_perturbation_count = 0
    total_interference = 0
    total_agent_steps = 0

    for step in range(int(args.steps)):
        active = [ag for ag in agents if not ag.mem.terminal]
        if not active:
            break

        decisions: Dict[int, Dict[str, Any]] = {}
        targets: Dict[int, Tuple[int, int, int]] = {}
        intended_target_counts: Dict[Tuple[int, int, int], int] = {}
        occupied = {ag.pos: ag.agent_id for ag in active}

        # Decision phase. All agents decide from the previous signal inbox.
        for ag in active:
            signal_ctx = compute_signal_context(ag, step, v11)
            y, shock, pr = v11.signal_for_darca(world, ag.pos, ag.mem, step, ag.q_layer)
            extra = {
                "z": pr["resource_pressure"],
                "exo": pr["unknown_pressure"],
                "d_dyn": pr["vertical_pressure"],
                "friction": pr["friction_pressure"],
                "coupling_t": signal_ctx["coupling_t"] if communication_condition in ("RECEIVER_LEARN", "FULL_INTERACTIVE") else 0.0,
                "sigma_t": signal_ctx["sigma_t"] if communication_condition in ("RECEIVER_LEARN", "FULL_INTERACTIVE") else 0.0,
            }
            darca_out = ag.darca.step(y, shock, extra) if ag.darca is not None else {}
            action = v11.darca_action(darca_out, world, ag.pos, ag.mem, rngs[ag.agent_id], step, ag.q_layer, ag.physics)
            action_source = "darca_core"
            if action not in v11.ACTIONS:
                action = "SCAN"
                action_source = "darca_core_invalid_scan"
            if action == "SCAN" and ag.mem.consecutive_scans >= task_args.max_consecutive_scans:
                if pr["danger_pressure"] < 0.50 and ag.mem.body_h > 0.28:
                    alt = v11.low_risk_non_scan_action(world, ag.pos, ag.mem, rngs[ag.agent_id], step, ag.q_layer, ag.physics)
                    if alt != "SCAN":
                        action = alt
                        action_source += "_scan_loop_escape"
            if action == "SCAN" and world.actual_kind(ag.pos, step) == v11.T_REST and ag.mem.body_h < 0.58:
                action = "REST"
                action_source += "_rest_site_restore"
            if ag.physics is not None:
                action, phys_reason = ag.physics.best_action_adjustment(action, world, ag.pos, ag.mem, rngs[ag.agent_id], step)
            sig_bias_value = 0.0
            sig_bias_reason = "none"
            if communication_condition in ("RECEIVER_LEARN", "FULL_INTERACTIVE") and signal_ctx["n_signals"] > 0:
                biased, reason, bval = choose_signal_biased_action(v11, world, ag, action, rngs[ag.agent_id], step, signal_ctx, args.signal_bias_strength, args.signal_bias_margin)
                if biased != action:
                    action_source += "_" + reason
                action = biased
                sig_bias_value = bval
                sig_bias_reason = reason

                if bool(getattr(args, "state_perturbation_assay", False)) and state_perturbation_count < int(getattr(args, "state_perturb_max_events_per_episode", 2)):
                    assay_rows = state_perturbation_action_probe(v11, world, ag, action, signal_ctx, step, args.signal_bias_strength)
                    dom_channel = int(signal_ctx.get("dominant_channel", -1))
                    for ar in assay_rows:
                        ar.update({
                            "scenario": scenario.scenario, "gradient_factor": scenario.gradient_factor, "gradient_level": scenario.gradient_level,
                            "task": task, "arm": arm, "communication_condition": communication_condition, "seed_id": seed_id,
                            "population_size": population_size, "agent_id": ag.agent_id, "step": step,
                            "dominant_channel": dom_channel, "channel_entropy": signal_ctx.get("channel_entropy", 0.0),
                            "original_body_h": safe_float(ag.mem.body_h), "base_action_after_signal_bias": action,
                            "signal_inbox_count": signal_ctx.get("n_signals", 0),
                        })
                        state_perturbation_rows.append(ar)
                    state_perturbation_count += 1
            tgt = action_target(v11, world, ag.pos, action)
            decisions[ag.agent_id] = {"action": action, "action_source": action_source, "darca_out": darca_out, "pr": pr, "old_pos": ag.pos, "old_h": ag.mem.body_h, "old_q": safe_float(getattr(ag.q_layer, "q", 0.0)), "signal_ctx": signal_ctx, "signal_bias_value": sig_bias_value, "signal_bias_reason": sig_bias_reason}
            targets[ag.agent_id] = tgt
            if action in v11.MOVE_ACTIONS and world.in_bounds(tgt):
                intended_target_counts[tgt] = intended_target_counts.get(tgt, 0) + 1

        # Conflict resolution. Exclusive occupancy, no priority advantage.
        interference_flags: Dict[int, bool] = {}
        for ag in active:
            action = decisions[ag.agent_id]["action"]
            tgt = targets[ag.agent_id]
            inter = False
            if action in v11.MOVE_ACTIONS and world.in_bounds(tgt):
                if intended_target_counts.get(tgt, 0) > 1:
                    inter = True
                if tgt in occupied and occupied[tgt] != ag.agent_id:
                    inter = True
                for other in active:
                    if other.agent_id == ag.agent_id:
                        continue
                    if targets.get(other.agent_id) == ag.pos and tgt == other.pos:
                        inter = True
                        break
            interference_flags[ag.agent_id] = inter

        emitted_packets: List[SignalPacket] = []
        emitted_by_agent: Dict[int, SignalPacket] = {}

        # Outcome and module update phase.
        for ag in active:
            d = decisions[ag.agent_id]
            action = d["action"]
            if action == "SCAN":
                ag.mem.scans += 1
            if action == "REST":
                ag.mem.rest_steps += 1
            pred = ag.physics.predict(action, world, ag.pos, ag.mem, step) if ag.physics is not None else {"pred_damage": 0.0, "pred_gain": 0.0, "pred_wall": 0.0}
            old_pos = ag.pos
            old_h = safe_float(ag.mem.body_h)
            old_q = d["old_q"]
            if interference_flags[ag.agent_id]:
                outcome = make_interference_outcome(v11, world, ag.pos, action, step)
                new_pos = ag.pos
                total_interference += 1
            else:
                new_pos, outcome = world.apply_action(ag.pos, action, ag.mem.known, ag.mem.body_h, step)
            ag.pos = new_pos
            ag.mem.previous_pos = old_pos
            ag.mem.body_h = v11.clip(ag.mem.body_h + outcome.delta_h, 0.0, 1.0)
            if outcome.resource_gain > 0:
                ag.mem.resources += 1
                ag.mem.total_resource_gain += outcome.resource_gain
            if outcome.damage > 0:
                ag.mem.total_damage += outcome.damage
            if action == "REST" and outcome.recovery_gain > 0:
                ag.mem.recovery_events += 1
            if action == "REST" and old_h > 0.62 and d["pr"]["danger_pressure"] < 0.25:
                ag.mem.unnecessary_rest_steps += 1
            if action in v11.MOVE_ACTIONS and outcome.damage > 0 and old_h < 0.45:
                ag.mem.reckless_moves += 1
            if ag.mem.body_h <= task_args.terminal_h:
                ag.mem.terminal = True
                ag.terminal = True
            ag.mem.update_history(ag.pos, action, outcome.event)
            q_state = ag.q_layer.update(ag.pos, action, outcome, d["pr"], ag.mem, d["darca_out"]) if ag.q_layer is not None else default_q_state()
            phys_state = ag.physics.update(action, outcome, pred) if ag.physics is not None else {"physics_pred_error": 0.0, "physics_score": 0.0, "physics_action_n": 0.0}
            social_state = ag.social.update(ag.mem, q_state, d["pr"], outcome, action) if ag.social is not None else default_social_state()
            total_agent_steps += 1

            # Receiver memory update from packets heard before this decision.
            action_classes = action_classes_for_action(v11, action, old_pos, ag.pos, d["signal_ctx"])
            embodied_value, outcome_info = embodied_value_from_outcome(old_h, old_q, q_state, outcome, social_state)
            if communication_condition in ("RECEIVER_LEARN", "FULL_INTERACTIVE"):
                for packet in ag.signal_inbox:
                    ag.signal_memory.update(packet.channel, action_classes, embodied_value, outcome_info)
                    if communication_condition == "FULL_INTERACTIVE":
                        # Store receiver benefit for sender update after all agents complete this step.
                        ag.pending_traces.append({
                            "type": "sender_feedback", "sender_id": packet.sender_id, "channel": packet.channel,
                            "benefit": embodied_value, "step": step,
                        })

            # Lagged receiver action logging for all pending delivered packets.
            new_pending: List[Dict[str, Any]] = []
            for tr in ag.pending_traces:
                if tr.get("type") == "sender_feedback":
                    new_pending.append(tr)
                    continue
                age = step - int(tr.get("delivery_step", step))
                if age in LAGS:
                    rr = dict(tr)
                    rr.update({
                        "lag": age,
                        "receiver_action": action,
                        "receiver_action_class": ",".join(action_classes),
                        "receiver_event": outcome.event,
                        "receiver_delta_h": safe_float(outcome.delta_h),
                        "receiver_damage": safe_float(outcome.damage),
                        "receiver_resource_gain": safe_float(outcome.resource_gain),
                        "receiver_recovery_gain": safe_float(outcome.recovery_gain),
                        "receiver_entered_unknown": int(getattr(outcome, "entered_unknown", False)),
                        "receiver_vertical": int(getattr(outcome, "vertical", False)),
                        "receiver_embodied_value": embodied_value,
                        "receiver_Q": safe_float(q_state.get("Q")),
                        "receiver_body_h": safe_float(ag.mem.body_h),
                    })
                    receiver_action_rows.append(rr)
                if age < max(LAGS):
                    new_pending.append(tr)
            ag.pending_traces = new_pending

            row = {
                "scenario": scenario.scenario,
                "gradient_factor": scenario.gradient_factor,
                "gradient_level": scenario.gradient_level,
                "task": task,
                "task_family": v11.task_family(task),
                "arm": arm,
                "communication_condition": communication_condition,
                "seed_id": seed_id,
                "population_size": population_size,
                "density_phi": args.density_phi,
                "world_size": world_size,
                "z_size": z_size,
                "agent_id": ag.agent_id,
                "step": step,
                "pos_i": ag.pos[0], "pos_j": ag.pos[1], "pos_k": ag.pos[2],
                "body_h": ag.mem.body_h,
                "terminal": int(ag.mem.terminal),
                "action": action,
                "action_source": d["action_source"],
                "event": outcome.event,
                "damage": outcome.damage,
                "resource_gain": outcome.resource_gain,
                "recovery_gain": outcome.recovery_gain,
                "hit_wall": int(outcome.hit_wall),
                "entered_unknown": int(outcome.entered_unknown),
                "outcome_friction": outcome.friction,
                "inter_agent_interference": int(interference_flags[ag.agent_id]),
                "signal_inbox_count": d["signal_ctx"].get("n_signals", 0),
                "signal_inbox_dominant_channel": d["signal_ctx"].get("dominant_channel", -1),
                "signal_inbox_entropy": d["signal_ctx"].get("channel_entropy", 0.0),
                "signal_coupling_t": d["signal_ctx"].get("coupling_t", 0.0),
                "signal_sigma_t": d["signal_ctx"].get("sigma_t", 0.0),
                "receiver_signal_bias_value": d.get("signal_bias_value", 0.0),
                "receiver_signal_bias_reason": d.get("signal_bias_reason", ""),
                "coverage": len(ag.mem.visited) / float(world.size * world.size * world.z_size),
                "resources": ag.mem.resources,
                "total_damage": ag.mem.total_damage,
                "rest_steps": ag.mem.rest_steps,
                "scans": ag.mem.scans,
                "physics_pred_damage": pred.get("pred_damage", 0.0),
                "physics_pred_gain": pred.get("pred_gain", 0.0),
                "physics_pred_wall": pred.get("pred_wall", 0.0),
                **{f"pressure_{k}": v for k, v in d["pr"].items()},
                **q_state,
                **phys_state,
                **social_state,
            }
            for k in ["h", "autonomy", "identity", "causal_confidence", "causal_engagement", "prediction_error", "agency_abs", "memory_force", "chi", "action_name"]:
                if k in d["darca_out"]:
                    row[f"darca_{k}"] = d["darca_out"][k]
            ag.records.append(row)

            # Emit a packet from the v11 anonymous social signal layer.
            if safe_float(social_state.get("social_signal")) > 0.0 and int(social_state.get("selected_signal_channel", -1)) >= 0:
                raw_ch = int(social_state.get("selected_signal_channel", -1))
                packet = SignalPacket(
                    sender_id=ag.agent_id,
                    channel=raw_ch,
                    intensity=max(0.05, safe_float(social_state.get("signal_probability_max", 0.5))),
                    sender_pos=ag.pos,
                    emitted_step=step,
                    raw_channel=raw_ch,
                    communication_condition=communication_condition,
                    sender_h=safe_float(ag.mem.body_h),
                    sender_Q=safe_float(q_state.get("Q")),
                    sender_Q_R=safe_float(q_state.get("Q_R", q_state.get("Q_avoidance_pressure"))),
                    sender_Q_G=safe_float(q_state.get("Q_G", q_state.get("Q_agency"))),
                    sender_physics_score=safe_float(phys_state.get("physics_score")),
                    sender_relief=safe_float(social_state.get("relief")),
                    sender_safe_surprise=safe_float(social_state.get("safe_surprise")),
                    sender_self_appraisal_gap=safe_float(social_state.get("self_appraisal_gap")),
                    sender_q_relief=safe_float(social_state.get("q_relief")),
                    sender_danger_pressure=safe_float(d["pr"].get("danger_pressure")),
                    sender_resource_pressure=safe_float(d["pr"].get("resource_pressure")),
                    sender_unknown_pressure=safe_float(d["pr"].get("unknown_pressure")),
                    sender_vertical_pressure=safe_float(d["pr"].get("vertical_pressure")),
                    sender_friction_pressure=safe_float(d["pr"].get("friction_pressure")),
                    sender_event=str(outcome.event),
                    sender_action=str(action),
                )
                if communication_condition == "FULL_INTERACTIVE":
                    packet = maybe_remap_channel_by_sender_preference(packet, ag, rng_np, args.sender_preference_strength, int(getattr(ag.social, "n_signals", 5)))
                emitted_packets.append(packet)
                emitted_by_agent[ag.agent_id] = packet
                ag.utterance_buffer.append(packet.channel)
                ag.utterance_buffer = ag.utterance_buffer[-3:]
                signal_event_rows.append(signal_packet_event_row(packet, scenario, task, arm, communication_condition, seed_id, population_size, world_size, z_size))
                if len(ag.utterance_buffer) >= 2:
                    utterance_rows.append(utterance_row_from_buffer(ag, packet, scenario, task, arm, communication_condition, seed_id, 2))
                if len(ag.utterance_buffer) >= 3:
                    utterance_rows.append(utterance_row_from_buffer(ag, packet, scenario, task, arm, communication_condition, seed_id, 3))

            # Clear consumed inbox. New delivery happens after all emissions.
            ag.signal_inbox = []

        # Sender feedback in FULL_INTERACTIVE.
        if communication_condition == "FULL_INTERACTIVE":
            feedback_by_sender: Dict[Tuple[int, int], List[float]] = {}
            for ag in agents:
                kept: List[Dict[str, Any]] = []
                for tr in ag.pending_traces:
                    if tr.get("type") == "sender_feedback":
                        key = (int(tr["sender_id"]), int(tr["channel"]))
                        feedback_by_sender.setdefault(key, []).append(safe_float(tr.get("benefit")))
                    else:
                        kept.append(tr)
                ag.pending_traces = kept
            for (sid, ch), vals in feedback_by_sender.items():
                if sid < len(agents):
                    cur = safe_float(agents[sid].sender_channel_preference.get(ch, 0.0))
                    agents[sid].sender_channel_preference[ch] = (1.0 - args.sender_preference_lr) * cur + args.sender_preference_lr * float(np.mean(vals))

        # Packet transformation and delivery. PRIVATE logs emissions only.
        packets_to_deliver = transform_packets(emitted_packets, communication_condition, rng_np)
        if communication_condition != "PRIVATE":
            for packet in packets_to_deliver:
                for rec in agents:
                    if rec.mem.terminal or rec.agent_id == packet.sender_id:
                        continue
                    dist = manhattan(packet.sender_pos, rec.pos)
                    if dist > int(args.signal_radius):
                        continue
                    rec.signal_inbox.append(packet)
                    rec_pr = v11.local_pressures(world, rec.pos, rec.mem, step)
                    delivery = signal_delivery_row(packet, rec, rec_pr, scenario, task, arm, communication_condition, seed_id, population_size, dist, step)
                    sender_agent = agents[packet.sender_id] if 0 <= int(packet.sender_id) < len(agents) else None
                    delivery.update(asymmetric_known_reference_flags(v11, world, sender_agent, rec, step))
                    signal_delivery_rows.append(delivery)
                    trace = dict(delivery)
                    trace["delivery_step"] = step
                    trace["type"] = "receiver_lag_trace"
                    rec.pending_traces.append(trace)

        if args.progress_every_steps > 0 and (step + 1) % args.progress_every_steps == 0:
            active_n = sum(1 for ag in agents if not ag.mem.terminal)
            logger.log(f"progress scenario={scenario.scenario} task={task} comm={communication_condition} seed={seed_id} step={step+1}/{args.steps} active={active_n} emitted={len(emitted_packets)}")

    # Summaries
    per_agent_rows: List[Dict[str, Any]] = []
    memory_rows: List[Dict[str, Any]] = []
    preference_rows: List[Dict[str, Any]] = []
    all_visited = set()
    final_positions: List[Tuple[int, int, int]] = []
    q_series: List[List[float]] = []
    s_series: List[List[float]] = []

    for ag in agents:
        records = ag.records
        all_visited.update(ag.mem.visited)
        final_positions.append(ag.pos)
        try:
            summary = v11.summarize_episode(task, arm, ag.agent_id, world, ag.mem, records)
        except Exception:
            summary = {"task": task, "arm": arm, "episode": ag.agent_id, "steps_run": len(records), "survived": int(not ag.mem.terminal), "final_body_h": ag.mem.body_h, "coverage": len(ag.mem.visited) / float(world.size * world.size * world.z_size), "resources": ag.mem.resources, "total_damage": ag.mem.total_damage, "autonomy_proper_index": 0.0}
        actions = [str(r.get("action", "")) for r in records]
        summary.update({
            "scenario": scenario.scenario,
            "gradient_factor": scenario.gradient_factor,
            "gradient_level": scenario.gradient_level,
            "task": task,
            "task_family": v11.task_family(task),
            "arm": arm,
            "communication_condition": communication_condition,
            "seed_id": seed_id,
            "population_size": population_size,
            "density_phi": args.density_phi,
            "world_size": world_size,
            "z_size": z_size,
            "agent_id": ag.agent_id,
            "action_entropy_norm": normalized_entropy(actions, k_total=len(v11.ACTIONS)),
            "vertical_action_fraction": float(np.mean([a in ("MOVE_UP", "MOVE_DOWN") for a in actions])) if actions else 0.0,
            "entered_unknown_fraction": float(np.mean([safe_float(r.get("entered_unknown")) for r in records])) if records else 0.0,
            "interference_rate": float(np.mean([safe_float(r.get("inter_agent_interference")) for r in records])) if records else 0.0,
            "mean_Q": float(np.mean([safe_float(r.get("Q")) for r in records])) if records else 0.0,
            "mean_physics_score": float(np.mean([safe_float(r.get("physics_score")) for r in records])) if records else 0.0,
            "mean_social_signal": float(np.mean([safe_float(r.get("social_signal")) for r in records])) if records else 0.0,
            "mean_signal_inbox_count": float(np.mean([safe_float(r.get("signal_inbox_count")) for r in records])) if records else 0.0,
            "mean_receiver_signal_bias": float(np.mean([safe_float(r.get("receiver_signal_bias_value")) for r in records])) if records else 0.0,
        })
        per_agent_rows.append(summary)
        memory_rows.extend(ag.signal_memory.rows(scenario.scenario, task, arm, communication_condition, seed_id, ag.agent_id, int(args.steps)))
        for ch, pref in ag.sender_channel_preference.items():
            preference_rows.append({
                "scenario": scenario.scenario, "task": task, "arm": arm, "communication_condition": communication_condition,
                "seed_id": seed_id, "agent_id": ag.agent_id, "channel": ch, "sender_channel_preference": pref,
            })
        q_series.append([safe_float(r.get("Q")) for r in records])
        s_series.append([safe_float(r.get("social_signal")) for r in records])

    dists = pairwise_distances(final_positions)
    pop_row = {
        "scenario": scenario.scenario,
        "gradient_factor": scenario.gradient_factor,
        "gradient_level": scenario.gradient_level,
        "task": task,
        "task_family": v11.task_family(task),
        "arm": arm,
        "communication_condition": communication_condition,
        "seed_id": seed_id,
        "population_size": population_size,
        "density_phi": args.density_phi,
        "world_size": world_size,
        "z_size": z_size,
        "world_volume": world.size * world.size * world.z_size,
        "steps_requested": int(args.steps),
        "total_agent_steps": int(total_agent_steps),
        "active_agents_final": int(sum(not ag.mem.terminal for ag in agents)),
        "survival_fraction": float(np.mean([not ag.mem.terminal for ag in agents])) if agents else 0.0,
        "collective_coverage": len(all_visited) / float(world.size * world.size * world.z_size),
        "per_capita_collective_coverage": (len(all_visited) / float(world.size * world.size * world.z_size)) / max(1, population_size),
        "mean_pairwise_distance_final": float(np.mean(dists)) if dists else 0.0,
        "aggregation_index_final": 1.0 / (1.0 + float(np.mean(dists))) if dists else 0.0,
        "interference_rate_population": total_interference / max(total_agent_steps, 1),
        "q_synchrony": mean_pairwise_correlation(q_series),
        "s_synchrony": mean_pairwise_correlation(s_series),
        "signal_event_count": len(signal_event_rows),
        "signal_delivery_count": len(signal_delivery_rows),
        "receiver_action_trace_count": len(receiver_action_rows),
        "state_perturbation_assay_count": len(state_perturbation_rows),
        "utterance_sequence_count": len(utterance_rows),
    }
    for f in ["autonomy_proper_index", "final_body_h", "coverage", "resources", "total_damage", "action_entropy_norm", "vertical_action_fraction", "entered_unknown_fraction", "mean_Q", "mean_physics_score", "mean_social_signal", "mean_signal_inbox_count", "mean_receiver_signal_bias"]:
        vals = [safe_float(r.get(f), np.nan) for r in per_agent_rows]
        vals = [v for v in vals if math.isfinite(v)]
        pop_row[f"agent_mean_{f}"] = float(np.mean(vals)) if vals else 0.0
        pop_row[f"agent_sd_{f}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return per_agent_rows, pop_row, signal_event_rows, signal_delivery_rows, receiver_action_rows, state_perturbation_rows, memory_rows, preference_rows, utterance_rows


def stable_hash(*parts: Any) -> int:
    text = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def pairwise_distances(positions: Sequence[Tuple[int, int, int]]) -> List[float]:
    vals: List[float] = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            vals.append(float(manhattan(positions[i], positions[j])))
    return vals


def mean_pairwise_correlation(series_by_agent: List[List[float]]) -> float:
    if len(series_by_agent) < 2:
        return 0.0
    vals: List[float] = []
    for i in range(len(series_by_agent)):
        for j in range(i + 1, len(series_by_agent)):
            a = np.asarray(series_by_agent[i], dtype=float)
            b = np.asarray(series_by_agent[j], dtype=float)
            m = min(len(a), len(b))
            if m >= 8:
                r = corr_np(a[:m], b[:m])
                if math.isfinite(r):
                    vals.append(r)
    return float(np.mean(vals)) if vals else 0.0


def default_q_state() -> Dict[str, Any]:
    return {"Q": 0.0, "Q_R": 0.0, "Q_G": 0.0, "Q_A": 0.0, "Q_Dg": 0.0, "Q_C": 0.0, "Q_P": 0.0, "Q_Mp": 0.0, "Q_Md": 0.0, "Q_Mc": 0.0, "Q_energy": 0.0, "Q_lesion_mode": "none"}


def default_social_state() -> Dict[str, Any]:
    return {"social_signal": 0.0, "selected_signal_channel": -1, "signal_probability_max": 0.0, "relief": 0.0, "safe_surprise": 0.0, "self_appraisal_gap": 0.0, "q_relief": 0.0, "safe_context": 0.0, "danger_context": 0.0, "receiver_recovery_increment": 0.0, "receiver_recovery_total": 0.0, "social_event_class": "none"}


def transform_packets(packets: List[SignalPacket], condition: str, rng: np.random.Generator) -> List[SignalPacket]:
    if condition == "PRIVATE":
        return []
    out = [SignalPacket(**asdict(p)) for p in packets]
    if not out:
        return []
    if condition == "RANDOM_MATCHED":
        n_channels = max(5, max(p.channel for p in out if p.channel >= 0) + 1)
        for p in out:
            p.raw_channel = p.channel
            p.channel = int(rng.integers(0, n_channels))
        return out
    if condition == "SHUFFLED":
        channels = [p.channel for p in out]
        positions = [p.sender_pos for p in out]
        senders = [p.sender_id for p in out]
        rng.shuffle(channels)
        rng.shuffle(positions)
        rng.shuffle(senders)
        for idx, p in enumerate(out):
            p.raw_channel = p.channel
            p.channel = int(channels[idx])
            p.sender_pos = positions[idx]
            p.sender_id = int(senders[idx])
        return out
    return out


def signal_packet_event_row(packet: SignalPacket, scenario: Scenario, task: str, arm: str, comm: str, seed_id: int, n: int, world_size: int, z_size: int) -> Dict[str, Any]:
    row = asdict(packet)
    row.update({
        "scenario": scenario.scenario, "gradient_factor": scenario.gradient_factor, "gradient_level": scenario.gradient_level,
        "task": task, "arm": arm, "communication_condition": comm, "seed_id": seed_id,
        "population_size": n, "world_size": world_size, "z_size": z_size,
        "sender_pos_i": packet.sender_pos[0], "sender_pos_j": packet.sender_pos[1], "sender_pos_k": packet.sender_pos[2],
    })
    row.pop("sender_pos", None)
    return row


def signal_delivery_row(packet: SignalPacket, receiver: AgentRuntime, receiver_pr: Dict[str, float], scenario: Scenario, task: str, arm: str, comm: str, seed_id: int, n: int, distance: int, step: int) -> Dict[str, Any]:
    return {
        "scenario": scenario.scenario, "gradient_factor": scenario.gradient_factor, "gradient_level": scenario.gradient_level,
        "task": task, "arm": arm, "communication_condition": comm, "seed_id": seed_id,
        "population_size": n, "delivery_step": step, "emitted_step": packet.emitted_step,
        "sender_id": packet.sender_id, "receiver_id": receiver.agent_id,
        "channel": packet.channel, "raw_channel": packet.raw_channel, "intensity": packet.intensity,
        "distance": distance, "distance_bin": distance_bin(distance), "relative_direction": relative_direction_bin(packet.sender_pos, receiver.pos),
        "sender_h": packet.sender_h, "sender_Q": packet.sender_Q, "sender_Q_R": packet.sender_Q_R, "sender_Q_G": packet.sender_Q_G,
        "sender_physics_score": packet.sender_physics_score,
        "sender_relief": packet.sender_relief, "sender_safe_surprise": packet.sender_safe_surprise,
        "sender_self_appraisal_gap": packet.sender_self_appraisal_gap, "sender_q_relief": packet.sender_q_relief,
        "sender_danger_pressure": packet.sender_danger_pressure,
        "sender_resource_pressure": packet.sender_resource_pressure,
        "sender_unknown_pressure": packet.sender_unknown_pressure,
        "sender_vertical_pressure": packet.sender_vertical_pressure,
        "sender_friction_pressure": packet.sender_friction_pressure,
        "sender_event": packet.sender_event, "sender_action": packet.sender_action,
        "receiver_danger_pressure": receiver_pr.get("danger_pressure", 0.0),
        "receiver_resource_pressure": receiver_pr.get("resource_pressure", 0.0),
        "receiver_unknown_pressure": receiver_pr.get("unknown_pressure", 0.0),
        "receiver_vertical_pressure": receiver_pr.get("vertical_pressure", 0.0),
        "hidden_ref_danger": int(packet.sender_danger_pressure > 0.34 and receiver_pr.get("danger_pressure", 0.0) <= 0.16),
        "hidden_ref_resource": int(packet.sender_resource_pressure > 0.18 and receiver_pr.get("resource_pressure", 0.0) <= 0.06),
        "hidden_ref_unknown": int(packet.sender_unknown_pressure > 0.24 and receiver_pr.get("unknown_pressure", 0.0) <= 0.08),
        "hidden_ref_vertical": int(packet.sender_vertical_pressure > 0.24 and receiver_pr.get("vertical_pressure", 0.0) <= 0.08),
    }


def utterance_row_from_buffer(agent: AgentRuntime, packet: SignalPacket, scenario: Scenario, task: str, arm: str, comm: str, seed_id: int, length: int) -> Dict[str, Any]:
    seq = agent.utterance_buffer[-length:]
    return {
        "scenario": scenario.scenario, "gradient_factor": scenario.gradient_factor, "gradient_level": scenario.gradient_level,
        "task": task, "arm": arm, "communication_condition": comm, "seed_id": seed_id,
        "sender_id": agent.agent_id, "step": packet.emitted_step, "sequence_length": length,
        "sequence": ">".join(str(x) for x in seq), "last_channel": packet.channel,
        "sender_h": packet.sender_h, "sender_Q": packet.sender_Q, "sender_Q_R": packet.sender_Q_R,
        "sender_physics_score": packet.sender_physics_score,
        "sender_relief": packet.sender_relief, "sender_safe_surprise": packet.sender_safe_surprise,
        "sender_self_appraisal_gap": packet.sender_self_appraisal_gap,
        "sender_danger_pressure": packet.sender_danger_pressure,
        "sender_resource_pressure": packet.sender_resource_pressure,
        "sender_unknown_pressure": packet.sender_unknown_pressure,
        "sender_vertical_pressure": packet.sender_vertical_pressure,
        "composite_danger_unknown": int(packet.sender_danger_pressure > 0.34 and packet.sender_unknown_pressure > 0.24),
        "composite_danger_vertical": int(packet.sender_danger_pressure > 0.34 and packet.sender_vertical_pressure > 0.24),
        "composite_resource_unknown": int(packet.sender_resource_pressure > 0.18 and packet.sender_unknown_pressure > 0.24),
        "composite_safe_recovery": int(packet.sender_relief > 0.05 and packet.sender_safe_surprise > 0.5),
    }

# =============================================================================
# Incremental runner
# =============================================================================

PER_AGENT_FIELDS = [
    "scenario", "gradient_factor", "gradient_level", "task", "task_family", "arm", "communication_condition", "seed_id", "population_size", "density_phi", "world_size", "z_size", "agent_id", "steps_run", "survived", "terminal", "mean_body_h", "final_body_h", "coverage", "resources", "total_damage", "recovery_events", "rest_steps", "scans", "action_entropy_norm", "vertical_action_fraction", "entered_unknown_fraction", "interference_rate", "mean_Q", "mean_physics_score", "mean_social_signal", "mean_signal_inbox_count", "mean_receiver_signal_bias", "autonomy_proper_index", "system_sovereignty", "information_theoretic_autonomy", "resilience_sacrifice", "heteronomy_index", "q_action_coupling", "q_risk_suppression", "physics_adaptation_delta", "social_signal_rate", "receiver_recovery_total"
]

POP_FIELDS = [
    "scenario", "gradient_factor", "gradient_level", "task", "task_family", "arm", "communication_condition", "seed_id", "population_size", "density_phi", "world_size", "z_size", "world_volume", "steps_requested", "total_agent_steps", "active_agents_final", "survival_fraction", "collective_coverage", "per_capita_collective_coverage", "mean_pairwise_distance_final", "aggregation_index_final", "interference_rate_population", "q_synchrony", "s_synchrony", "signal_event_count", "signal_delivery_count", "receiver_action_trace_count", "state_perturbation_assay_count", "utterance_sequence_count", "agent_mean_autonomy_proper_index", "agent_sd_autonomy_proper_index", "agent_mean_final_body_h", "agent_mean_coverage", "agent_mean_resources", "agent_mean_total_damage", "agent_mean_action_entropy_norm", "agent_mean_vertical_action_fraction", "agent_mean_entered_unknown_fraction", "agent_mean_mean_Q", "agent_mean_mean_physics_score", "agent_mean_mean_social_signal", "agent_mean_mean_signal_inbox_count", "agent_mean_mean_receiver_signal_bias"
]

RUN_FIELDS = ["run_key", "status", "error", "elapsed_sec", "created_at", "scenario", "task", "arm", "communication_condition", "seed_id", "population_size", "density_phi"]


def run_key(scenario: str, task: str, arm: str, comm: str, seed_id: int, n: int) -> str:
    return f"{scenario}|{task}|{arm}|{comm}|seed{seed_id}|N{n}"


def load_done(outdir: Path) -> set:
    path = outdir / "run_index.csv"
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(path)
        return set(df.loc[df["status"] == "ok", "run_key"].astype(str).tolist())
    except Exception:
        return set()


def run_all(v11: Any, darca_module: Any, scenarios: List[Scenario], args: argparse.Namespace, logger: Logger) -> None:
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    comms = [x.strip() for x in args.communication_conditions.split(",") if x.strip()]
    pops = [int(x.strip()) for x in args.population_sizes.split(",") if x.strip()]
    for c in comms:
        if c not in COMMUNICATION_CONDITIONS:
            raise ValueError(f"Invalid communication condition: {c}")
    done = load_done(args.outdir) if args.resume else set()
    total = len(scenarios) * len(tasks) * len(arms) * len(comms) * int(args.seeds) * len(pops)
    requested_agent_steps = len(scenarios) * len(tasks) * len(arms) * len(comms) * int(args.seeds) * int(args.steps) * sum(pops)
    logger.log(f"planned episodes={total:,}; requested agent-steps before early termination={requested_agent_steps:,}")
    completed = skipped = 0
    for sc in scenarios:
        logger.log(f"[scenario] {sc.scenario} factor={sc.gradient_factor} level={sc.gradient_level}")
        for task in tasks:
            for arm in arms:
                for comm in comms:
                    for seed_id in range(int(args.seeds)):
                        for n in pops:
                            rk = run_key(sc.scenario, task, arm, comm, seed_id, n)
                            if rk in done:
                                skipped += 1
                                continue
                            t0 = time.time()
                            meta = {"run_key": rk, "status": "ok", "error": "", "created_at": now(), "scenario": sc.scenario, "task": task, "arm": arm, "communication_condition": comm, "seed_id": seed_id, "population_size": n, "density_phi": args.density_phi}
                            try:
                                per_agent, pop_row, events, deliveries, recv_actions, state_perturb_rows, mem_rows, pref_rows, utterances = run_population_language_episode(v11, darca_module, sc, task, arm, comm, seed_id, n, args, logger)
                                append_rows_csv(args.outdir / "language_agent_episode_summary.csv", per_agent, PER_AGENT_FIELDS)
                                append_rows_csv(args.outdir / "language_population_episode_summary.csv", [pop_row], POP_FIELDS)
                                append_rows_csv(args.outdir / "signal_event_log.csv", events, SIGNAL_EVENT_FIELDS)
                                append_rows_csv(args.outdir / "signal_delivery_log.csv", deliveries, SIGNAL_DELIVERY_FIELDS)
                                append_rows_csv(args.outdir / "receiver_signal_action_log.csv", recv_actions, RECEIVER_ACTION_FIELDS)
                                append_rows_csv(args.outdir / "state_perturbation_reflex_assay.csv", state_perturb_rows, STATE_PERTURBATION_FIELDS)
                                append_rows_csv(args.outdir / "receiver_signal_memory_snapshot.csv", mem_rows, MEMORY_FIELDS)
                                append_rows_csv(args.outdir / "sender_channel_preference_snapshot.csv", pref_rows, PREF_FIELDS)
                                append_rows_csv(args.outdir / "utterance_sequence_log.csv", utterances, UTTERANCE_FIELDS)
                            except KeyboardInterrupt:
                                logger.log("Interrupted by user.")
                                raise
                            except Exception as e:
                                meta["status"] = "error"
                                meta["error"] = repr(e)
                                logger.log(f"[error] {rk}: {repr(e)}")
                                logger.log(traceback.format_exc())
                            meta["elapsed_sec"] = time.time() - t0
                            append_rows_csv(args.outdir / "run_index.csv", [meta], RUN_FIELDS)
                            completed += 1
                            if completed % max(1, int(args.progress_every_runs)) == 0:
                                logger.log(f"[progress] completed_new={completed:,} skipped_existing={skipped:,} / planned={total:,}")
    logger.log(f"run stage done: completed_new={completed:,} skipped_existing={skipped:,}")

# Field lists for compact incremental CSVs.
SIGNAL_EVENT_FIELDS = [
    "scenario", "gradient_factor", "gradient_level", "task", "arm", "communication_condition", "seed_id", "population_size", "world_size", "z_size", "sender_id", "channel", "raw_channel", "intensity", "emitted_step", "sender_pos_i", "sender_pos_j", "sender_pos_k", "sender_h", "sender_Q", "sender_Q_R", "sender_Q_G", "sender_physics_score", "sender_relief", "sender_safe_surprise", "sender_self_appraisal_gap", "sender_q_relief", "sender_danger_pressure", "sender_resource_pressure", "sender_unknown_pressure", "sender_vertical_pressure", "sender_friction_pressure", "sender_event", "sender_action"
]
SIGNAL_DELIVERY_FIELDS = [
    "scenario", "gradient_factor", "gradient_level", "task", "arm", "communication_condition", "seed_id", "population_size", "delivery_step", "emitted_step", "sender_id", "receiver_id", "channel", "raw_channel", "intensity", "distance", "distance_bin", "relative_direction", "sender_h", "sender_Q", "sender_Q_R", "sender_Q_G", "sender_physics_score", "sender_relief", "sender_safe_surprise", "sender_self_appraisal_gap", "sender_q_relief", "sender_danger_pressure", "sender_resource_pressure", "sender_unknown_pressure", "sender_vertical_pressure", "sender_friction_pressure", "sender_event", "sender_action", "receiver_danger_pressure", "receiver_resource_pressure", "receiver_unknown_pressure", "receiver_vertical_pressure", "hidden_ref_danger", "hidden_ref_resource", "hidden_ref_unknown", "hidden_ref_vertical", "asym_known_count", "asym_sender_known_danger", "asym_sender_known_resource", "asym_sender_known_rest", "asym_sender_known_empty"
]
RECEIVER_ACTION_FIELDS = SIGNAL_DELIVERY_FIELDS + ["lag", "receiver_action", "receiver_action_class", "receiver_event", "receiver_delta_h", "receiver_damage", "receiver_resource_gain", "receiver_recovery_gain", "receiver_entered_unknown", "receiver_vertical", "receiver_embodied_value", "receiver_Q", "receiver_body_h"]
STATE_PERTURBATION_FIELDS = ["scenario", "gradient_factor", "gradient_level", "task", "arm", "communication_condition", "seed_id", "population_size", "agent_id", "step", "dominant_channel", "channel_entropy", "signal_inbox_count", "original_body_h", "perturbation_label", "perturbed_body_h", "base_action_after_signal_bias", "assay_action", "assay_action_class", "assay_score", "assay_signal_bias", "diverged_from_original_state"]
MEMORY_FIELDS = ["scenario", "task", "arm", "communication_condition", "seed_id", "agent_id", "step", "channel", "channel_count", "channel_value", "channel_risk", "channel_recovery", "channel_exploration", "channel_vertical", "channel_unknown"] + [f"action_value_{a}" for a in ACTION_CLASSES]
PREF_FIELDS = ["scenario", "task", "arm", "communication_condition", "seed_id", "agent_id", "channel", "sender_channel_preference"]
UTTERANCE_FIELDS = ["scenario", "gradient_factor", "gradient_level", "task", "arm", "communication_condition", "seed_id", "sender_id", "step", "sequence_length", "sequence", "last_channel", "sender_h", "sender_Q", "sender_Q_R", "sender_physics_score", "sender_relief", "sender_safe_surprise", "sender_self_appraisal_gap", "sender_danger_pressure", "sender_resource_pressure", "sender_unknown_pressure", "sender_vertical_pressure", "composite_danger_unknown", "composite_danger_vertical", "composite_resource_unknown", "composite_safe_recovery"]

# =============================================================================
# Post hoc analysis
# =============================================================================

def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def signal_sender_grounding_summary(event_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if event_df.empty:
        return pd.DataFrame()
    metrics = [
        "sender_h", "sender_Q", "sender_Q_R", "sender_Q_G", "sender_physics_score", "sender_relief", "sender_safe_surprise",
        "sender_self_appraisal_gap", "sender_q_relief", "sender_danger_pressure", "sender_resource_pressure", "sender_unknown_pressure",
        "sender_vertical_pressure", "sender_friction_pressure",
    ]
    rows: List[Dict[str, Any]] = []
    df = event_df.copy()
    for c in metrics + ["channel"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for keys, sub in df.groupby(["scenario", "task", "arm", "communication_condition"], dropna=False):
        scenario, task, arm, comm = keys
        if sub.shape[0] < 5:
            continue
        ch = sub["channel"].astype(str).tolist()
        for m in metrics:
            if m not in sub.columns:
                continue
            bins = bin_series(sub[m].tolist())
            rows.append({
                "scenario": scenario, "task": task, "arm": arm, "communication_condition": comm,
                "grounding_target": m, "n_events": int(sub.shape[0]),
                "mi_bits_channel_target": mutual_information_discrete(ch, bins),
                "channel_entropy_norm": normalized_entropy(ch),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_control_delta(out, ["scenario", "task", "arm", "grounding_target"], "mi_bits_channel_target", controls=("SHUFFLED", "RANDOM_MATCHED", "PRIVATE"))
    return out


def receiver_action_effect_summary(action_df: pd.DataFrame) -> pd.DataFrame:
    if action_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    df = action_df.copy()
    for keys, sub in df.groupby(["scenario", "task", "arm", "communication_condition", "lag"], dropna=False):
        scenario, task, arm, comm, lag = keys
        if sub.shape[0] < 5:
            continue
        ch = sub["channel"].astype(str).tolist()
        act = sub["receiver_action"].astype(str).tolist()
        act_class = sub["receiver_action_class"].astype(str).tolist()
        rows.append({
            "scenario": scenario, "task": task, "arm": arm, "communication_condition": comm, "lag": int(lag),
            "n_traces": int(sub.shape[0]),
            "mi_bits_channel_receiver_action": mutual_information_discrete(ch, act),
            "mi_bits_channel_receiver_action_class": mutual_information_discrete(ch, act_class),
            "mean_receiver_embodied_value": float(pd.to_numeric(sub.get("receiver_embodied_value", 0.0), errors="coerce").fillna(0).mean()),
            "mean_receiver_damage": float(pd.to_numeric(sub.get("receiver_damage", 0.0), errors="coerce").fillna(0).mean()),
            "mean_receiver_recovery_gain": float(pd.to_numeric(sub.get("receiver_recovery_gain", 0.0), errors="coerce").fillna(0).mean()),
            "scan_fraction": float((sub["receiver_action"].astype(str) == "SCAN").mean()) if "receiver_action" in sub else 0.0,
            "rest_fraction": float((sub["receiver_action"].astype(str) == "REST").mean()) if "receiver_action" in sub else 0.0,
            "vertical_fraction": float(sub["receiver_vertical"].astype(float).mean()) if "receiver_vertical" in sub else 0.0,
            "unknown_entry_fraction": float(sub["receiver_entered_unknown"].astype(float).mean()) if "receiver_entered_unknown" in sub else 0.0,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_control_delta(out, ["scenario", "task", "arm", "lag"], "mi_bits_channel_receiver_action_class", controls=("SHUFFLED", "RANDOM_MATCHED", "PRIVATE"))
    return out


def embodied_signal_value_summary(memory_df: pd.DataFrame) -> pd.DataFrame:
    if memory_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for keys, sub in memory_df.groupby(["scenario", "task", "arm", "communication_condition"], dropna=False):
        scenario, task, arm, comm = keys
        rows.append({
            "scenario": scenario, "task": task, "arm": arm, "communication_condition": comm,
            "n_memory_rows": int(sub.shape[0]),
            "mean_abs_channel_value": float(pd.to_numeric(sub["channel_value"], errors="coerce").abs().mean()),
            "mean_abs_action_value": float(np.mean([pd.to_numeric(sub[f"action_value_{a}"], errors="coerce").abs().mean() for a in ACTION_CLASSES if f"action_value_{a}" in sub.columns])),
            "max_channel_count": int(pd.to_numeric(sub["channel_count"], errors="coerce").fillna(0).max()),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_control_delta(out, ["scenario", "task", "arm"], "mean_abs_channel_value", controls=("SHUFFLED", "RANDOM_MATCHED", "PRIVATE"))
    return out


def hidden_reference_summary(delivery_df: pd.DataFrame, action_df: pd.DataFrame) -> pd.DataFrame:
    if delivery_df.empty:
        return pd.DataFrame()
    flags = ["hidden_ref_danger", "hidden_ref_resource", "hidden_ref_unknown", "hidden_ref_vertical"]
    rows: List[Dict[str, Any]] = []
    df = delivery_df.copy()
    for f in flags:
        if f in df.columns:
            df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0).astype(int)
    for keys, sub in df.groupby(["scenario", "task", "arm", "communication_condition"], dropna=False):
        scenario, task, arm, comm = keys
        if sub.shape[0] < 5:
            continue
        ch = sub["channel"].astype(str).tolist()
        for f in flags:
            if f not in sub.columns:
                continue
            rows.append({
                "scenario": scenario, "task": task, "arm": arm, "communication_condition": comm,
                "hidden_reference_target": f, "n_deliveries": int(sub.shape[0]),
                "target_fraction": float(sub[f].mean()),
                "mi_bits_channel_hidden_target": mutual_information_discrete(ch, sub[f].astype(str).tolist()),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_control_delta(out, ["scenario", "task", "arm", "hidden_reference_target"], "mi_bits_channel_hidden_target", controls=("SHUFFLED", "RANDOM_MATCHED", "PRIVATE"))
    return out


def compositionality_summary(utt_df: pd.DataFrame) -> pd.DataFrame:
    if utt_df.empty:
        return pd.DataFrame()
    targets = ["composite_danger_unknown", "composite_danger_vertical", "composite_resource_unknown", "composite_safe_recovery"]
    rows: List[Dict[str, Any]] = []
    df = utt_df.copy()
    for keys, sub in df.groupby(["scenario", "task", "arm", "communication_condition", "sequence_length"], dropna=False):
        scenario, task, arm, comm, L = keys
        if int(L) < 2 or sub.shape[0] < 5:
            continue
        seq = sub["sequence"].astype(str).tolist()
        last = sub["last_channel"].astype(str).tolist()
        first = [s.split(">", 1)[0] if ">" in s else s for s in seq]
        for t in targets:
            if t not in sub.columns:
                continue
            y = sub[t].astype(str).tolist()
            mi_seq = mutual_information_discrete(seq, y)
            mi_first = mutual_information_discrete(first, y)
            mi_last = mutual_information_discrete(last, y)
            rows.append({
                "scenario": scenario, "task": task, "arm": arm, "communication_condition": comm,
                "sequence_length": int(L), "composite_target": t, "n_sequences": int(sub.shape[0]),
                "mi_bits_sequence_target": mi_seq,
                "mi_bits_first_channel_target": mi_first,
                "mi_bits_last_channel_target": mi_last,
                "pair_gain_over_best_single": mi_seq - max(mi_first, mi_last),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_control_delta(out, ["scenario", "task", "arm", "sequence_length", "composite_target"], "pair_gain_over_best_single", controls=("SHUFFLED", "RANDOM_MATCHED", "PRIVATE"))
    return out


def shared_codebook_summary(event_df: pd.DataFrame) -> pd.DataFrame:
    if event_df.empty:
        return pd.DataFrame()
    # Codebook vector: channel × sender-state bins. Similarity is cosine across agents within seed and across seeds.
    targets = ["sender_Q", "sender_physics_score", "sender_relief", "sender_danger_pressure", "sender_unknown_pressure", "sender_vertical_pressure"]
    rows: List[Dict[str, Any]] = []
    df = event_df.copy()
    for t in targets:
        if t in df.columns:
            df[t + "_bin"] = bin_series(pd.to_numeric(df[t], errors="coerce").tolist())
    for keys, sub in df.groupby(["scenario", "task", "arm", "communication_condition"], dropna=False):
        scenario, task, arm, comm = keys
        if sub.shape[0] < 20:
            continue
        vectors: List[np.ndarray] = []
        for _, ss in sub.groupby(["seed_id", "sender_id"], dropna=False):
            parts: List[float] = []
            for ch in sorted(sub["channel"].dropna().unique()):
                chs = ss[ss["channel"] == ch]
                for t in targets:
                    col = t + "_bin"
                    if col in chs.columns:
                        vc = chs[col].value_counts(normalize=True)
                        parts.extend([float(vc.get(b, 0.0)) for b in ("low", "mid", "high", "flat")])
            if parts:
                vectors.append(np.asarray(parts, dtype=float))
        sims: List[float] = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                a = vectors[i]; b = vectors[j]
                m = min(len(a), len(b))
                if m <= 0:
                    continue
                den = float(np.linalg.norm(a[:m]) * np.linalg.norm(b[:m]))
                if den > EPS:
                    sims.append(float(np.dot(a[:m], b[:m]) / den))
        rows.append({
            "scenario": scenario, "task": task, "arm": arm, "communication_condition": comm,
            "n_codebooks": len(vectors), "mean_codebook_cosine_similarity": float(np.mean(sims)) if sims else 0.0,
            "sd_codebook_cosine_similarity": float(np.std(sims, ddof=1)) if len(sims) > 1 else 0.0,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_control_delta(out, ["scenario", "task", "arm"], "mean_codebook_cosine_similarity", controls=("SHUFFLED", "RANDOM_MATCHED", "PRIVATE"))
    return out


def population_episode_summary(pop_df: pd.DataFrame) -> pd.DataFrame:
    if pop_df.empty:
        return pd.DataFrame()
    metrics = ["collective_coverage", "survival_fraction", "signal_event_count", "signal_delivery_count", "q_synchrony", "s_synchrony", "agent_mean_mean_receiver_signal_bias"]
    rows: List[Dict[str, Any]] = []
    for keys, sub in pop_df.groupby(["scenario", "task", "arm", "communication_condition", "population_size"], dropna=False):
        scenario, task, arm, comm, n = keys
        row = {"scenario": scenario, "task": task, "arm": arm, "communication_condition": comm, "population_size": int(n), "n_seeds": int(sub["seed_id"].nunique())}
        for m in metrics:
            if m in sub.columns:
                row.update(describe_vector(pd.to_numeric(sub[m], errors="coerce").to_numpy(dtype=float), seed=2026, prefix=m + "_"))
        rows.append(row)
    return pd.DataFrame(rows)




def receiver_state_signature(sub: pd.DataFrame) -> List[str]:
    n = len(sub)
    if n == 0:
        return []
    h_bin = bin_series(pd.to_numeric(sub.get("receiver_body_h", pd.Series([0.0] * n)), errors="coerce").tolist())
    q_bin = bin_series(pd.to_numeric(sub.get("receiver_Q", pd.Series([0.0] * n)), errors="coerce").tolist())
    danger_bin = bin_series(pd.to_numeric(sub.get("receiver_danger_pressure", pd.Series([0.0] * n)), errors="coerce").tolist())
    unknown_bin = bin_series(pd.to_numeric(sub.get("receiver_unknown_pressure", pd.Series([0.0] * n)), errors="coerce").tolist())
    dist = sub.get("distance_bin", pd.Series(["NA"] * n)).astype(str).tolist()
    return joint_label(h_bin, q_bin, danger_bin, unknown_bin, dist)


def reflex_rejection_summary(action_df: pd.DataFrame) -> pd.DataFrame:
    """Primary anti-reflex test.

    A fixed reflex predicts receiver action mainly from channel alone. A richer,
    language-like interpretation predicts extra information from the interaction
    between channel and receiver internal/context state.
    """
    if action_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    df = action_df.copy()
    for keys, sub in df.groupby(["scenario", "task", "arm", "communication_condition", "lag"], dropna=False):
        scenario, task, arm, comm, lag = keys
        if sub.shape[0] < 8:
            continue
        ch = sub["channel"].astype(str).tolist()
        act = sub["receiver_action_class"].astype(str).tolist()
        state = receiver_state_signature(sub)
        joint = joint_label(ch, state)
        mi_ch = mutual_information_discrete(ch, act)
        mi_state = mutual_information_discrete(state, act)
        mi_joint = mutual_information_discrete(joint, act)
        cmi_ch_given_state = conditional_mutual_information_discrete(ch, act, state)
        interaction_gain = mi_joint - max(mi_ch, mi_state)
        reflex_dominance = mi_ch / max(mi_joint, EPS)
        rows.append({
            "scenario": scenario, "task": task, "arm": arm, "communication_condition": comm, "lag": int(lag),
            "n_traces": int(sub.shape[0]),
            "mi_bits_channel_action": mi_ch,
            "mi_bits_state_action": mi_state,
            "mi_bits_channel_state_action": mi_joint,
            "cmi_bits_channel_action_given_receiver_state": cmi_ch_given_state,
            "interaction_gain_over_best_single": interaction_gain,
            "reflex_dominance_ratio": reflex_dominance,
            "nonreflex_score": interaction_gain + cmi_ch_given_state,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_control_delta(out, ["scenario", "task", "arm", "lag"], "nonreflex_score", controls=("SHUFFLED", "RANDOM_MATCHED", "PRIVATE"))
    return out


def future_reference_lag_summary(action_df: pd.DataFrame) -> pd.DataFrame:
    """Test whether signal identity predicts future receiver consequences beyond current receiver state."""
    if action_df.empty:
        return pd.DataFrame()
    targets = {
        "future_embodied_value_bin": lambda s: bin_series(pd.to_numeric(s.get("receiver_embodied_value", pd.Series([0.0] * len(s))), errors="coerce").tolist()),
        "future_damage": lambda s: ["1" if safe_float(v) > 0 else "0" for v in s.get("receiver_damage", pd.Series([0.0] * len(s))).tolist()],
        "future_recovery": lambda s: ["1" if safe_float(v) > 0 else "0" for v in s.get("receiver_recovery_gain", pd.Series([0.0] * len(s))).tolist()],
        "future_unknown_entry": lambda s: ["1" if safe_float(v) > 0 else "0" for v in s.get("receiver_entered_unknown", pd.Series([0.0] * len(s))).tolist()],
        "future_vertical_action": lambda s: ["1" if safe_float(v) > 0 else "0" for v in s.get("receiver_vertical", pd.Series([0.0] * len(s))).tolist()],
    }
    rows: List[Dict[str, Any]] = []
    for keys, sub in action_df.groupby(["scenario", "task", "arm", "communication_condition", "lag"], dropna=False):
        scenario, task, arm, comm, lag = keys
        if sub.shape[0] < 8:
            continue
        ch = sub["channel"].astype(str).tolist()
        state = receiver_state_signature(sub)
        for target_name, fn in targets.items():
            y = fn(sub)
            if len(set(y)) < 2:
                continue
            mi = mutual_information_discrete(ch, y)
            cmi = conditional_mutual_information_discrete(ch, y, state)
            rows.append({
                "scenario": scenario, "task": task, "arm": arm, "communication_condition": comm, "lag": int(lag),
                "future_target": target_name, "n_traces": int(sub.shape[0]),
                "mi_bits_channel_future_target": mi,
                "cmi_bits_channel_future_target_given_receiver_state": cmi,
                "future_reference_score": cmi,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_control_delta(out, ["scenario", "task", "arm", "lag", "future_target"], "future_reference_score", controls=("SHUFFLED", "RANDOM_MATCHED", "PRIVATE"))
    return out


def order_dependence_summary(utt_df: pd.DataFrame) -> pd.DataFrame:
    """Ordered sequence test: ordered bigram/trigram must outperform unordered bag-of-channels."""
    if utt_df.empty:
        return pd.DataFrame()
    targets = ["composite_danger_unknown", "composite_danger_vertical", "composite_resource_unknown", "composite_safe_recovery"]
    rows: List[Dict[str, Any]] = []
    df = utt_df.copy()
    for keys, sub in df.groupby(["scenario", "task", "arm", "communication_condition", "sequence_length"], dropna=False):
        scenario, task, arm, comm, L = keys
        if int(L) < 2 or sub.shape[0] < 8:
            continue
        seq = sub["sequence"].astype(str).tolist()
        bag = [">".join(sorted(x.split(">"))) for x in seq]
        first = [x.split(">")[0] for x in seq]
        last = [x.split(">")[-1] for x in seq]
        for target in targets:
            if target not in sub.columns:
                continue
            y = sub[target].astype(str).tolist()
            if len(set(y)) < 2:
                continue
            mi_ordered = mutual_information_discrete(seq, y)
            mi_bag = mutual_information_discrete(bag, y)
            mi_first = mutual_information_discrete(first, y)
            mi_last = mutual_information_discrete(last, y)
            gain_bag = mi_ordered - mi_bag
            gain_single = mi_ordered - max(mi_first, mi_last)
            rows.append({
                "scenario": scenario, "task": task, "arm": arm, "communication_condition": comm,
                "sequence_length": int(L), "order_target": target, "n_sequences": int(sub.shape[0]),
                "mi_bits_ordered_sequence_target": mi_ordered,
                "mi_bits_unordered_bag_target": mi_bag,
                "mi_bits_first_channel_target": mi_first,
                "mi_bits_last_channel_target": mi_last,
                "ordered_gain_over_bag": gain_bag,
                "ordered_gain_over_best_single": gain_single,
                "order_dependence_score": min(gain_bag, gain_single),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_control_delta(out, ["scenario", "task", "arm", "sequence_length", "order_target"], "order_dependence_score", controls=("SHUFFLED", "RANDOM_MATCHED", "PRIVATE"))
    return out


def internal_grounding_vs_external_label_summary(event_df: pd.DataFrame) -> pd.DataFrame:
    """Compare internal bodily/valuation grounding against external-state labels."""
    if event_df.empty:
        return pd.DataFrame()
    internal = ["sender_h", "sender_Q", "sender_Q_R", "sender_Q_G", "sender_physics_score", "sender_relief", "sender_safe_surprise", "sender_self_appraisal_gap", "sender_q_relief"]
    external = ["sender_danger_pressure", "sender_resource_pressure", "sender_unknown_pressure", "sender_vertical_pressure", "sender_friction_pressure"]
    rows: List[Dict[str, Any]] = []
    df = event_df.copy()
    for keys, sub in df.groupby(["scenario", "task", "arm", "communication_condition"], dropna=False):
        scenario, task, arm, comm = keys
        if sub.shape[0] < 8:
            continue
        ch = sub["channel"].astype(str).tolist()
        internal_mis: List[float] = []
        external_mis: List[float] = []
        target_rows: Dict[str, float] = {}
        for m in internal:
            if m in sub.columns:
                y = bin_series(pd.to_numeric(sub[m], errors="coerce").tolist())
                mi = mutual_information_discrete(ch, y)
                internal_mis.append(mi); target_rows[m] = mi
        for m in external:
            if m in sub.columns:
                y = bin_series(pd.to_numeric(sub[m], errors="coerce").tolist())
                mi = mutual_information_discrete(ch, y)
                external_mis.append(mi); target_rows[m] = mi
        internal_mean = float(np.mean(internal_mis)) if internal_mis else 0.0
        external_mean = float(np.mean(external_mis)) if external_mis else 0.0
        row = {
            "scenario": scenario, "task": task, "arm": arm, "communication_condition": comm,
            "n_events": int(sub.shape[0]),
            "internal_grounding_mi_mean": internal_mean,
            "external_label_mi_mean": external_mean,
            "internal_minus_external_mi": internal_mean - external_mean,
            "internal_grounding_score": internal_mean - external_mean,
        }
        for k, v in target_rows.items():
            row[f"mi_{k}"] = v
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_control_delta(out, ["scenario", "task", "arm"], "internal_grounding_score", controls=("SHUFFLED", "RANDOM_MATCHED", "PRIVATE"))
    return out


def functional_code_alignment_summary(event_df: pd.DataFrame) -> pd.DataFrame:
    """Channel-number-free alignment of functional roles across agents/seeds."""
    if event_df.empty:
        return pd.DataFrame()
    functions = ["sender_Q", "sender_physics_score", "sender_relief", "sender_danger_pressure", "sender_unknown_pressure", "sender_vertical_pressure"]
    rows: List[Dict[str, Any]] = []
    df = event_df.copy()
    for keys, sub in df.groupby(["scenario", "task", "arm", "communication_condition"], dropna=False):
        scenario, task, arm, comm = keys
        if sub.shape[0] < 20:
            continue
        profiles: List[np.ndarray] = []
        role_maps: List[str] = []
        for _, ss in sub.groupby(["seed_id", "sender_id"], dropna=False):
            if ss.shape[0] < 5:
                continue
            prof: List[float] = []
            roles: List[str] = []
            for fn in functions:
                if fn not in ss.columns:
                    prof.append(0.0); roles.append("NA"); continue
                best_ch = None
                best_val = -1e9
                for ch, chs in ss.groupby("channel", dropna=False):
                    val = float(pd.to_numeric(chs[fn], errors="coerce").fillna(0).mean())
                    if val > best_val:
                        best_val = val; best_ch = ch
                prof.append(best_val if math.isfinite(best_val) else 0.0)
                roles.append(f"{fn}:ch{best_ch}")
            profiles.append(np.asarray(prof, dtype=float))
            role_maps.append(";".join(roles))
        sims: List[float] = []
        for i in range(len(profiles)):
            for j in range(i + 1, len(profiles)):
                a = profiles[i]; b = profiles[j]
                den = float(np.linalg.norm(a) * np.linalg.norm(b))
                if den > EPS:
                    sims.append(float(np.dot(a, b) / den))
        rows.append({
            "scenario": scenario, "task": task, "arm": arm, "communication_condition": comm,
            "n_functional_codebooks": len(profiles),
            "mean_function_profile_cosine_similarity": float(np.mean(sims)) if sims else 0.0,
            "sd_function_profile_cosine_similarity": float(np.std(sims, ddof=1)) if len(sims) > 1 else 0.0,
            "example_role_map": role_maps[0] if role_maps else "",
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_control_delta(out, ["scenario", "task", "arm"], "mean_function_profile_cosine_similarity", controls=("SHUFFLED", "RANDOM_MATCHED", "PRIVATE"))
    return out


def signal_counterfactual_replay_summary(action_df: pd.DataFrame) -> pd.DataFrame:
    """Matched-control counterfactual proxy.

    This is a primary lightweight causal-control analysis that compares FULL_INTERACTIVE
    receiver consequences against matched non-interactive controls at the same
    scenario/task/arm/lag. It is not a full state clone/replay; the output field
    states this explicitly so the manuscript can avoid overclaiming.
    """
    if action_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    df = action_df.copy()
    if "receiver_embodied_value" not in df.columns:
        return pd.DataFrame()
    df["receiver_embodied_value"] = pd.to_numeric(df["receiver_embodied_value"], errors="coerce").fillna(0.0)
    df["receiver_damage"] = pd.to_numeric(df.get("receiver_damage", 0.0), errors="coerce").fillna(0.0)
    df["receiver_recovery_gain"] = pd.to_numeric(df.get("receiver_recovery_gain", 0.0), errors="coerce").fillna(0.0)
    for keys, sub in df.groupby(["scenario", "task", "arm", "lag"], dropna=False):
        scenario, task, arm, lag = keys
        full = sub[sub["communication_condition"] == "FULL_INTERACTIVE"]
        if full.empty:
            continue
        for ctrl in ["PRIVATE", "RANDOM_MATCHED", "SHUFFLED", "RECEIVER_LEARN"]:
            c = sub[sub["communication_condition"] == ctrl]
            if c.empty:
                continue
            rows.append({
                "scenario": scenario, "task": task, "arm": arm, "lag": int(lag),
                "comparison": f"FULL_INTERACTIVE_vs_{ctrl}",
                "analysis_type": "matched_control_counterfactual_proxy_not_state_clone_replay",
                "n_full": int(full.shape[0]), "n_control": int(c.shape[0]),
                "full_mean_embodied_value": float(full["receiver_embodied_value"].mean()),
                "control_mean_embodied_value": float(c["receiver_embodied_value"].mean()),
                "delta_embodied_value": float(full["receiver_embodied_value"].mean() - c["receiver_embodied_value"].mean()),
                "full_mean_damage": float(full["receiver_damage"].mean()),
                "control_mean_damage": float(c["receiver_damage"].mean()),
                "delta_damage_control_minus_full": float(c["receiver_damage"].mean() - full["receiver_damage"].mean()),
                "full_mean_recovery": float(full["receiver_recovery_gain"].mean()),
                "control_mean_recovery": float(c["receiver_recovery_gain"].mean()),
                "delta_recovery_full_minus_control": float(full["receiver_recovery_gain"].mean() - c["receiver_recovery_gain"].mean()),
            })
    return pd.DataFrame(rows)



def state_perturbation_reflex_summary(assay_df: pd.DataFrame) -> pd.DataFrame:
    """Primary L3 state-perturbation assay.

    Tests whether the same received channel/context maps to different receiver
    actions when only the receiver's body_h state is perturbed. This directly
    targets the reflex concern: a fixed reflex should be invariant across H_LOW,
    H_ORIGINAL, and H_HIGH.
    """
    if assay_df.empty or "perturbation_label" not in assay_df.columns:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    key_cols = ["scenario", "task", "arm", "communication_condition", "seed_id", "population_size", "agent_id", "step", "dominant_channel"]
    df = assay_df.copy()
    for key, sub in df.groupby(key_cols, dropna=False):
        if sub.shape[0] < 2:
            continue
        acts = {str(r["perturbation_label"]): str(r["assay_action"]) for _, r in sub.iterrows()}
        low = acts.get("H_LOW", "")
        orig = acts.get("H_ORIGINAL", "")
        high = acts.get("H_HIGH", "")
        if not orig:
            continue
        rows.append({
            "scenario": key[0], "task": key[1], "arm": key[2], "communication_condition": key[3],
            "seed_id": key[4], "population_size": key[5], "agent_id": key[6], "step": key[7], "dominant_channel": key[8],
            "low_action": low, "original_action": orig, "high_action": high,
            "low_differs_from_original": int(low != "" and low != orig),
            "high_differs_from_original": int(high != "" and high != orig),
            "low_high_divergence": int(low != "" and high != "" and low != high),
            "state_conditioned_switch": int(len(set([x for x in [low, orig, high] if x])) > 1),
        })
    event_df = pd.DataFrame(rows)
    if event_df.empty:
        return pd.DataFrame()
    out_rows: List[Dict[str, Any]] = []
    for keys, sub in event_df.groupby(["scenario", "task", "arm", "communication_condition"], dropna=False):
        sc, task, arm, comm = keys
        out_rows.append({
            "scenario": sc, "task": task, "arm": arm, "communication_condition": comm,
            "n_assay_events": int(sub.shape[0]),
            "state_conditioned_switch_rate": float(pd.to_numeric(sub["state_conditioned_switch"], errors="coerce").mean()),
            "low_high_divergence_rate": float(pd.to_numeric(sub["low_high_divergence"], errors="coerce").mean()),
            "low_differs_from_original_rate": float(pd.to_numeric(sub["low_differs_from_original"], errors="coerce").mean()),
            "high_differs_from_original_rate": float(pd.to_numeric(sub["high_differs_from_original"], errors="coerce").mean()),
        })
    out = pd.DataFrame(out_rows)
    if not out.empty:
        out["state_perturbation_score"] = pd.to_numeric(out["state_conditioned_switch_rate"], errors="coerce").fillna(0.0)
        out = add_control_delta(out, ["scenario", "task", "arm"], "state_perturbation_score", controls=("SHUFFLED", "RANDOM_MATCHED", "PRIVATE"))
    return out


def asymmetric_hidden_reference_summary(delivery_df: pd.DataFrame, action_df: pd.DataFrame) -> pd.DataFrame:
    """Primary L4 partial-observability hidden-reference assay.

    Uses sender.mem.known vs receiver.mem.known asymmetry logged at delivery.
    It is stricter than pressure-only hidden_ref_* because it requires that the
    sender has explicit known local information while the receiver lacks it.
    """
    if delivery_df.empty:
        return pd.DataFrame()
    flags = ["asym_sender_known_danger", "asym_sender_known_resource", "asym_sender_known_rest", "asym_sender_known_empty"]
    rows: List[Dict[str, Any]] = []
    for keys, sub in delivery_df.groupby(["scenario", "task", "arm", "communication_condition"], dropna=False):
        sc, task, arm, comm = keys
        for f in flags:
            if f not in sub.columns:
                continue
            y = pd.to_numeric(sub[f], errors="coerce").fillna(0).astype(int)
            if int(y.sum()) < 2 or int((1 - y).sum()) < 2:
                mi = auc = 0.0
            else:
                mi = mutual_information_discrete(sub["channel"].astype(str).tolist(), y.astype(str).tolist())
                auc = auc_binary_score(y.tolist(), pd.to_numeric(sub["channel"], errors="coerce").fillna(-1).tolist())
            rows.append({
                "scenario": sc, "task": task, "arm": arm, "communication_condition": comm,
                "asym_hidden_reference_target": f,
                "n_deliveries": int(sub.shape[0]),
                "positive_fraction": float(y.mean()) if len(y) else 0.0,
                "mi_bits_channel_asym_target": mi,
                "auc_channel_asym_target_numeric": auc,
                "asymmetric_hidden_reference_score": mi,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_control_delta(out, ["scenario", "task", "arm", "asym_hidden_reference_target"], "asymmetric_hidden_reference_score", controls=("SHUFFLED", "RANDOM_MATCHED", "PRIVATE"))
    return out


def ngram_embodied_success_summary(action_df: pd.DataFrame, max_n: int = 3) -> pd.DataFrame:
    """Primary L6 ordered n-gram embodied-success analysis.

    Builds ordered delivery-channel n-grams within sender-receiver streams and
    asks whether ordered n-grams explain post-signal embodied value better than
    unordered bags of the same channels.
    """
    if action_df.empty or "channel" not in action_df.columns:
        return pd.DataFrame()
    df = action_df.copy()
    if "lag" in df.columns:
        df = df[pd.to_numeric(df["lag"], errors="coerce") == 1].copy()
    if df.empty:
        return pd.DataFrame()
    sort_cols = [c for c in ["scenario", "task", "arm", "communication_condition", "seed_id", "sender_id", "receiver_id", "delivery_step"] if c in df.columns]
    df = df.sort_values(sort_cols)
    seq_rows: List[Dict[str, Any]] = []
    group_cols = [c for c in ["scenario", "task", "arm", "communication_condition", "seed_id", "sender_id", "receiver_id"] if c in df.columns]
    for _, sub in df.groupby(group_cols, dropna=False):
        chans = [str(int(safe_float(x, -1))) for x in sub["channel"].tolist()]
        vals = pd.to_numeric(sub.get("receiver_embodied_value", 0.0), errors="coerce").fillna(0.0).tolist()
        dmg = pd.to_numeric(sub.get("receiver_damage", 0.0), errors="coerce").fillna(0.0).tolist()
        rec = pd.to_numeric(sub.get("receiver_recovery_gain", 0.0), errors="coerce").fillna(0.0).tolist()
        rows_orig = sub.to_dict("records")
        for i in range(len(chans)):
            for n in range(2, int(max_n) + 1):
                if i - n + 1 < 0:
                    continue
                seq = chans[i - n + 1:i + 1]
                base = {k: rows_orig[i].get(k, "") for k in ["scenario", "task", "arm", "communication_condition", "seed_id", "sender_id", "receiver_id"]}
                base.update({
                    "sequence_length": n,
                    "ordered_sequence": ">".join(seq),
                    "unordered_bag": "+".join(sorted(seq)),
                    "post_sequence_embodied_value": vals[i],
                    "post_sequence_damage": dmg[i],
                    "post_sequence_recovery_gain": rec[i],
                })
                seq_rows.append(base)
    seq_df = pd.DataFrame(seq_rows)
    if seq_df.empty:
        return pd.DataFrame()
    out_rows: List[Dict[str, Any]] = []
    for keys, sub in seq_df.groupby(["scenario", "task", "arm", "communication_condition", "sequence_length"], dropna=False):
        sc, task, arm, comm, n = keys
        if sub.shape[0] < 8:
            continue
        y = pd.to_numeric(sub["post_sequence_embodied_value"], errors="coerce").fillna(0.0)
        ordered_means = sub.groupby("ordered_sequence")["post_sequence_embodied_value"].mean()
        bag_means = sub.groupby("unordered_bag")["post_sequence_embodied_value"].mean()
        ordered_pred = sub["ordered_sequence"].map(ordered_means).astype(float)
        bag_pred = sub["unordered_bag"].map(bag_means).astype(float)
        ordered_var = float(np.var(ordered_pred))
        bag_var = float(np.var(bag_pred))
        ordered_corr = corr_np(ordered_pred.tolist(), y.tolist())
        bag_corr = corr_np(bag_pred.tolist(), y.tolist())
        top_seq = ordered_means.sort_values(ascending=False).index[0] if len(ordered_means) else ""
        top_val = float(ordered_means.max()) if len(ordered_means) else 0.0
        out_rows.append({
            "scenario": sc, "task": task, "arm": arm, "communication_condition": comm, "sequence_length": int(n),
            "n_sequence_events": int(sub.shape[0]),
            "n_ordered_sequences": int(sub["ordered_sequence"].nunique()),
            "n_unordered_bags": int(sub["unordered_bag"].nunique()),
            "ordered_sequence_variance": ordered_var,
            "unordered_bag_variance": bag_var,
            "ordered_minus_bag_variance": ordered_var - bag_var,
            "ordered_embodied_corr": ordered_corr,
            "bag_embodied_corr": bag_corr,
            "ordered_minus_bag_corr": ordered_corr - bag_corr,
            "top_ordered_sequence": top_seq,
            "top_ordered_sequence_embodied_value": top_val,
            "mean_post_sequence_embodied_value": float(y.mean()),
            "mean_post_sequence_damage": float(pd.to_numeric(sub["post_sequence_damage"], errors="coerce").fillna(0.0).mean()),
            "mean_post_sequence_recovery_gain": float(pd.to_numeric(sub["post_sequence_recovery_gain"], errors="coerce").fillna(0.0).mean()),
        })
    out = pd.DataFrame(out_rows)
    if not out.empty:
        out["ngram_embodied_success_score"] = pd.to_numeric(out["ordered_minus_bag_corr"], errors="coerce").fillna(0.0) + pd.to_numeric(out["ordered_minus_bag_variance"], errors="coerce").fillna(0.0)
        out = add_control_delta(out, ["scenario", "task", "arm", "sequence_length"], "ngram_embodied_success_score", controls=("SHUFFLED", "RANDOM_MATCHED", "PRIVATE"))
    return out

def primary_language_criteria_summary(levels: pd.DataFrame, reflex: pd.DataFrame, hidden: pd.DataFrame, internal: pd.DataFrame, order: pd.DataFrame, future: pd.DataFrame, counter: pd.DataFrame, functional: pd.DataFrame, statep: pd.DataFrame, asym_hidden: pd.DataFrame, ngram: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    contexts = set()
    for df in [levels, reflex, hidden, internal, order, future, counter, functional, statep, asym_hidden, ngram]:
        if not df.empty and all(c in df.columns for c in ["scenario", "task", "arm"]):
            contexts.update(tuple(x) for x in df[["scenario", "task", "arm"]].drop_duplicates().to_numpy())
    for scenario, task, arm in sorted(contexts):
        lev = levels[(levels["scenario"] == scenario) & (levels["task"] == task) & (levels["arm"] == arm)] if not levels.empty else pd.DataFrame()
        ref = reflex[(reflex["scenario"] == scenario) & (reflex["task"] == task) & (reflex["arm"] == arm) & (reflex["communication_condition"] == "FULL_INTERACTIVE")] if not reflex.empty else pd.DataFrame()
        hid = hidden[(hidden["scenario"] == scenario) & (hidden["task"] == task) & (hidden["arm"] == arm) & (hidden["communication_condition"] == "FULL_INTERACTIVE")] if not hidden.empty else pd.DataFrame()
        inte = internal[(internal["scenario"] == scenario) & (internal["task"] == task) & (internal["arm"] == arm) & (internal["communication_condition"] == "FULL_INTERACTIVE")] if not internal.empty else pd.DataFrame()
        ords = order[(order["scenario"] == scenario) & (order["task"] == task) & (order["arm"] == arm) & (order["communication_condition"] == "FULL_INTERACTIVE")] if not order.empty else pd.DataFrame()
        fut = future[(future["scenario"] == scenario) & (future["task"] == task) & (future["arm"] == arm) & (future["communication_condition"] == "FULL_INTERACTIVE")] if not future.empty else pd.DataFrame()
        cnt = counter[(counter["scenario"] == scenario) & (counter["task"] == task) & (counter["arm"] == arm)] if not counter.empty else pd.DataFrame()
        fun = functional[(functional["scenario"] == scenario) & (functional["task"] == task) & (functional["arm"] == arm) & (functional["communication_condition"] == "FULL_INTERACTIVE")] if not functional.empty else pd.DataFrame()
        stp = statep[(statep["scenario"] == scenario) & (statep["task"] == task) & (statep["arm"] == arm) & (statep["communication_condition"] == "FULL_INTERACTIVE")] if not statep.empty else pd.DataFrame()
        asym = asym_hidden[(asym_hidden["scenario"] == scenario) & (asym_hidden["task"] == task) & (asym_hidden["arm"] == arm) & (asym_hidden["communication_condition"] == "FULL_INTERACTIVE")] if not asym_hidden.empty else pd.DataFrame()
        ngr = ngram[(ngram["scenario"] == scenario) & (ngram["task"] == task) & (ngram["arm"] == arm) & (ngram["communication_condition"] == "FULL_INTERACTIVE")] if not ngram.empty else pd.DataFrame()
        language_level = int(safe_max(lev, "language_like_level")) if not lev.empty else 0
        reflex_score = safe_max(ref, "delta_vs_best_control")
        hidden_score = safe_max(hid, "delta_vs_best_control")
        internal_score = safe_max(inte, "delta_vs_best_control")
        order_score = safe_max(ords, "delta_vs_best_control")
        future_score = safe_max(fut, "delta_vs_best_control")
        functional_score = safe_max(fun, "delta_vs_best_control")
        statep_score = safe_max(stp, "delta_vs_best_control")
        asym_score = safe_max(asym, "delta_vs_best_control")
        ngram_score = safe_max(ngr, "delta_vs_best_control")
        counter_score = safe_max(cnt, "delta_embodied_value")
        criteria = {
            "receiver_effective_level_ge_2": language_level >= 2,
            "nonreflexive_interpretation": reflex_score > 0.002,
            "state_perturbation_response_divergence": statep_score > 0.05,
            "pressure_hidden_reference": hidden_score > 0.002,
            "asymmetric_hidden_reference": asym_score > 0.001,
            "internal_grounding_over_external_label": internal_score > 0.000,
            "future_outcome_reference": future_score > 0.001,
            "order_dependence": order_score > 0.0005,
            "ngram_embodied_success": ngram_score > 0.000,
            "functional_code_alignment": functional_score > 0.000,
            "matched_counterfactual_proxy_positive": counter_score > 0.0,
        }
        n_pass = sum(1 for v in criteria.values() if v)
        rows.append({
            "scenario": scenario, "task": task, "arm": arm,
            "language_like_level": language_level,
            "primary_criteria_passed": n_pass,
            "primary_criteria_total": len(criteria),
            "primary_criteria_pass_rate": n_pass / max(1, len(criteria)),
            "reflex_rejection_delta": reflex_score,
            "hidden_reference_delta": hidden_score,
            "internal_grounding_delta": internal_score,
            "future_reference_delta": future_score,
            "order_dependence_delta": order_score,
            "functional_alignment_delta": functional_score,
            "state_perturbation_delta": statep_score,
            "asymmetric_hidden_reference_delta": asym_score,
            "ngram_embodied_success_delta": ngram_score,
            "counterfactual_proxy_delta": counter_score,
            **{k: int(v) for k, v in criteria.items()},
        })
    return pd.DataFrame(rows)

def language_like_level_summary(sender_df: pd.DataFrame, receiver_df: pd.DataFrame, emb_df: pd.DataFrame, hidden_df: pd.DataFrame, comp_df: pd.DataFrame, shared_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    contexts = set()
    for df in [sender_df, receiver_df, emb_df, hidden_df, comp_df, shared_df]:
        if not df.empty:
            contexts.update(tuple(x) for x in df[["scenario", "task", "arm"]].drop_duplicates().to_numpy())
    for scenario, task, arm in sorted(contexts):
        sfull = subset_comm(sender_df, scenario, task, arm, "FULL_INTERACTIVE")
        rfull = subset_comm(receiver_df, scenario, task, arm, "FULL_INTERACTIVE")
        efull = subset_comm(emb_df, scenario, task, arm, "FULL_INTERACTIVE")
        hfull = subset_comm(hidden_df, scenario, task, arm, "FULL_INTERACTIVE")
        cfull = subset_comm(comp_df, scenario, task, arm, "FULL_INTERACTIVE")
        shfull = subset_comm(shared_df, scenario, task, arm, "FULL_INTERACTIVE")
        sender_score = safe_max(sfull, "delta_vs_best_control")
        receiver_score = safe_max(rfull, "delta_vs_best_control")
        embodied_score = safe_max(efull, "delta_vs_best_control")
        hidden_score = safe_max(hfull, "delta_vs_best_control")
        compositional_score = safe_max(cfull, "delta_vs_best_control")
        shared_score = safe_max(shfull, "delta_vs_best_control")
        level = 0
        if sender_score > 0.010:
            level = 1
        if level >= 1 and receiver_score > 0.006:
            level = 2
        if level >= 2 and embodied_score > 0.003:
            level = 3
        if level >= 3 and shared_score > 0.010:
            level = 4
        if level >= 4 and hidden_score > 0.006:
            level = 5
        if level >= 5 and compositional_score > 0.003:
            level = 6
        # Level 7 requires cross-play, which is not run in this main runner yet.
        rows.append({
            "scenario": scenario, "task": task, "arm": arm,
            "language_like_level": level,
            "sender_grounding_delta": sender_score,
            "receiver_action_delta": receiver_score,
            "embodied_value_delta": embodied_score,
            "shared_code_delta": shared_score,
            "hidden_reference_delta": hidden_score,
            "compositionality_delta": compositional_score,
            "interpretation": level_interpretation(level),
        })
    return pd.DataFrame(rows)


def subset_comm(df: pd.DataFrame, scenario: str, task: str, arm: str, comm: str) -> pd.DataFrame:
    if df.empty:
        return df
    return df[(df["scenario"] == scenario) & (df["task"] == task) & (df["arm"] == arm) & (df["communication_condition"] == comm)]


def safe_max(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return 0.0
    arr = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.max()) if len(arr) else 0.0


def add_control_delta(df: pd.DataFrame, key_cols: List[str], value_col: str, controls: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    out["best_control_value"] = 0.0
    out["delta_vs_best_control"] = 0.0
    if out.empty or value_col not in out.columns:
        return out
    for key, sub in out.groupby(key_cols, dropna=False):
        idx = sub.index
        ctrl = sub[sub["communication_condition"].isin(controls)]
        best = float(pd.to_numeric(ctrl[value_col], errors="coerce").max()) if not ctrl.empty else 0.0
        out.loc[idx, "best_control_value"] = best
        out.loc[idx, "delta_vs_best_control"] = pd.to_numeric(out.loc[idx, value_col], errors="coerce") - best
    return out


def level_interpretation(level: int) -> str:
    return {
        0: "no detectable language-like communication",
        1: "expressive sender-state signal",
        2: "receiver-effective social signal",
        3: "embodied grounded signal",
        4: "partially shared code",
        5: "referential signal under receiver-hidden states",
        6: "partially compositional language-like precursor",
        7: "cross-play transferable language-like precursor",
    }.get(int(level), "unknown")


def posthoc(outdir: Path, args: argparse.Namespace, logger: Logger) -> None:
    logger.log("[posthoc] reading logs")
    event_df = read_csv_if_exists(outdir / "signal_event_log.csv")
    delivery_df = read_csv_if_exists(outdir / "signal_delivery_log.csv")
    action_df = read_csv_if_exists(outdir / "receiver_signal_action_log.csv")
    state_assay_df = read_csv_if_exists(outdir / "state_perturbation_reflex_assay.csv")
    mem_df = read_csv_if_exists(outdir / "receiver_signal_memory_snapshot.csv")
    utt_df = read_csv_if_exists(outdir / "utterance_sequence_log.csv")
    pop_df = read_csv_if_exists(outdir / "language_population_episode_summary.csv")

    logger.log("[posthoc] signal sender grounding")
    sender = signal_sender_grounding_summary(event_df, seed=args.seed + 1)
    logger.log("[posthoc] receiver action effects")
    receiver = receiver_action_effect_summary(action_df)
    logger.log("[posthoc] embodied signal value")
    embodied = embodied_signal_value_summary(mem_df)
    logger.log("[posthoc] hidden reference")
    hidden = hidden_reference_summary(delivery_df, action_df)
    logger.log("[posthoc] asymmetric hidden reference")
    asym_hidden = asymmetric_hidden_reference_summary(delivery_df, action_df)
    logger.log("[posthoc] compositionality")
    comp = compositionality_summary(utt_df)
    logger.log("[posthoc] shared codebook")
    shared = shared_codebook_summary(event_df)
    logger.log("[posthoc] primary reflex rejection")
    reflex = reflex_rejection_summary(action_df)
    logger.log("[posthoc] primary state-perturbation reflex assay")
    statep = state_perturbation_reflex_summary(state_assay_df)
    logger.log("[posthoc] primary future reference")
    future = future_reference_lag_summary(action_df)
    logger.log("[posthoc] primary order dependence")
    order = order_dependence_summary(utt_df)
    logger.log("[posthoc] primary n-gram embodied success")
    ngram = ngram_embodied_success_summary(action_df, max_n=int(getattr(args, "ngram_max_n", 3)))
    logger.log("[posthoc] primary internal grounding vs external labels")
    internal = internal_grounding_vs_external_label_summary(event_df)
    logger.log("[posthoc] primary functional code alignment")
    functional = functional_code_alignment_summary(event_df)
    logger.log("[posthoc] primary matched-control counterfactual proxy")
    counter = signal_counterfactual_replay_summary(action_df)
    logger.log("[posthoc] population summary")
    popsum = population_episode_summary(pop_df)
    logger.log("[posthoc] level summary")
    levels = language_like_level_summary(sender, receiver, embodied, hidden, comp, shared)
    logger.log("[posthoc] primary language criteria")
    primary = primary_language_criteria_summary(levels, reflex, hidden, internal, order, future, counter, functional, statep, asym_hidden, ngram)

    sender.to_csv(outdir / "signal_sender_grounding_summary.csv", index=False)
    receiver.to_csv(outdir / "receiver_action_effect_summary.csv", index=False)
    embodied.to_csv(outdir / "embodied_signal_value_summary.csv", index=False)
    hidden.to_csv(outdir / "hidden_reference_summary.csv", index=False)
    asym_hidden.to_csv(outdir / "asymmetric_hidden_reference_summary.csv", index=False)
    comp.to_csv(outdir / "compositionality_summary.csv", index=False)
    shared.to_csv(outdir / "shared_codebook_summary.csv", index=False)
    reflex.to_csv(outdir / "reflex_rejection_summary.csv", index=False)
    statep.to_csv(outdir / "state_perturbation_reflex_summary.csv", index=False)
    future.to_csv(outdir / "future_reference_lag_summary.csv", index=False)
    order.to_csv(outdir / "order_dependence_summary.csv", index=False)
    ngram.to_csv(outdir / "ngram_embodied_success_summary.csv", index=False)
    internal.to_csv(outdir / "internal_grounding_vs_external_label_summary.csv", index=False)
    functional.to_csv(outdir / "functional_code_alignment_summary.csv", index=False)
    counter.to_csv(outdir / "signal_counterfactual_replay_summary.csv", index=False)
    primary.to_csv(outdir / "primary_language_criteria_summary.csv", index=False)
    popsum.to_csv(outdir / "population_language_episode_summary.csv", index=False)
    levels.to_csv(outdir / "language_like_level_summary.csv", index=False)
    write_language_report(outdir, args, sender, receiver, embodied, hidden, comp, shared, levels, reflex, future, order, internal, functional, counter, primary, statep, asym_hidden, ngram)
    make_language_figures(outdir, sender, receiver, hidden, comp, levels, reflex, order, internal, primary)
    logger.log("[posthoc] complete")


def write_language_report(outdir: Path, args: argparse.Namespace, sender: pd.DataFrame, receiver: pd.DataFrame, embodied: pd.DataFrame, hidden: pd.DataFrame, comp: pd.DataFrame, shared: pd.DataFrame, levels: pd.DataFrame, reflex: pd.DataFrame, future: pd.DataFrame, order: pd.DataFrame, internal: pd.DataFrame, functional: pd.DataFrame, counter: pd.DataFrame, primary: pd.DataFrame, statep: pd.DataFrame, asym_hidden: pd.DataFrame, ngram: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("V11 population language-emergence validation report")
    lines.append("=" * 84)
    lines.append(f"Created: {now()}")
    lines.append("")
    lines.append("Design")
    lines.append("------")
    lines.append(f"Plan: {args.plan}")
    lines.append(f"Tasks: {args.tasks}")
    lines.append(f"Arms: {args.arms}")
    lines.append(f"Communication conditions: {args.communication_conditions}")
    lines.append(f"Population sizes: {args.population_sizes}")
    lines.append(f"Density phi: {args.density_phi}")
    lines.append(f"Seeds: {args.seeds}")
    lines.append(f"Steps: {args.steps}")
    lines.append("")
    lines.append("Core guardrail")
    lines.append("--------------")
    lines.append("The v11 core is imported. This runner adds only a shared-world population wrapper, anonymous signal delivery, receiver embodied signal memory, and post hoc language-like communication analyses.")
    lines.append("Receivers see channel, intensity, direction, and distance only. Sender Q, physical state, relief, safe surprise, and hidden-reference fields are analysis-only variables.")
    lines.append("")
    lines.append("Primary language-like criteria")
    lines.append("------------------------------")
    if primary.empty:
        lines.append("(none)")
    else:
        show = primary.sort_values(["primary_criteria_passed", "language_like_level"], ascending=[False, False]).head(35)
        cols = ["scenario", "task", "arm", "language_like_level", "primary_criteria_passed", "primary_criteria_total", "primary_criteria_pass_rate", "receiver_effective_level_ge_2", "nonreflexive_interpretation", "state_perturbation_response_divergence", "pressure_hidden_reference", "asymmetric_hidden_reference", "internal_grounding_over_external_label", "future_outcome_reference", "order_dependence", "ngram_embodied_success", "functional_code_alignment", "matched_counterfactual_proxy_positive"]
        lines.append(show[[c for c in cols if c in show.columns]].to_string(index=False))
    lines.append("")
    lines.append("Highest language-like levels")
    lines.append("----------------------------")
    if levels.empty:
        lines.append("(none)")
    else:
        show = levels.sort_values(["language_like_level", "sender_grounding_delta", "receiver_action_delta", "compositionality_delta"], ascending=[False, False, False, False]).head(30)
        cols = ["scenario", "task", "arm", "language_like_level", "interpretation", "sender_grounding_delta", "receiver_action_delta", "embodied_value_delta", "shared_code_delta", "hidden_reference_delta", "compositionality_delta"]
        lines.append(show[[c for c in cols if c in show.columns]].to_string(index=False))
    lines.append("")
    for title, df, col in [
        ("Strongest sender grounding", sender, "delta_vs_best_control"),
        ("Strongest receiver action effects", receiver, "delta_vs_best_control"),
        ("Strongest hidden-reference effects", hidden, "delta_vs_best_control"),
        ("Strongest compositionality effects", comp, "delta_vs_best_control"),
        ("Strongest shared-code effects", shared, "delta_vs_best_control"),
        ("Primary reflex rejection", reflex, "delta_vs_best_control"),
        ("Primary state-perturbation reflex assay", statep, "delta_vs_best_control"),
        ("Primary asymmetric hidden-reference assay", asym_hidden, "delta_vs_best_control"),
        ("Primary future reference", future, "delta_vs_best_control"),
        ("Primary order dependence", order, "delta_vs_best_control"),
        ("Primary n-gram embodied success", ngram, "delta_vs_best_control"),
        ("Primary internal grounding over external labels", internal, "delta_vs_best_control"),
        ("Primary functional code alignment", functional, "delta_vs_best_control"),
        ("Matched-control counterfactual proxy", counter, "delta_embodied_value"),
    ]:
        lines.append(title)
        lines.append("-" * len(title))
        if df.empty or col not in df.columns:
            lines.append("(none)")
        else:
            show = df.sort_values(col, ascending=False).head(20)
            lines.append(show.to_string(index=False))
        lines.append("")
    lines.append("Interpretive constraint")
    lines.append("-----------------------")
    lines.append("The primary endpoint is not signal emission alone. V3 treats reflex rejection, explicit receiver-state perturbation, asymmetric hidden-state reference, internal-state grounding, future-outcome reference, order dependence, n-gram embodied success, and functional alignment as main criteria for internally grounded language-like communication.")
    lines.append("The counterfactual file is a matched-control proxy, not a full cloned-state replay. It is used as a lightweight main-control analysis and should not be described as exact causal replay unless cloned-state snapshots are added in a later runner.")
    lines.append("Level 1-2 supports expressive and receiver-effective signaling. Level 3-5 supports embodied grounding, shared code, and referentiality. Level 6 supports a compositional precursor. Level 7 requires a separate cross-play battery and is not assigned by this main runner.")
    (outdir / "population_language_emergence_report.txt").write_text("\n".join(lines), encoding="utf-8")


def make_language_figures(outdir: Path, sender: pd.DataFrame, receiver: pd.DataFrame, hidden: pd.DataFrame, comp: pd.DataFrame, levels: pd.DataFrame, reflex: pd.DataFrame, order: pd.DataFrame, internal: pd.DataFrame, primary: pd.DataFrame) -> None:
    if plt is None:
        return
    figdir = ensure_dir(outdir / "figures")
    def barh_top(df: pd.DataFrame, value_col: str, label_cols: Sequence[str], title: str, fname: str, n: int = 25) -> None:
        if df.empty or value_col not in df.columns:
            return
        sub = df.sort_values(value_col, ascending=False).head(n).copy()
        labels = ["|".join(str(r.get(c, "")) for c in label_cols) for _, r in sub.iterrows()]
        y = np.arange(len(sub))
        plt.figure(figsize=(12, max(5, 0.34 * len(sub))))
        plt.barh(y, pd.to_numeric(sub[value_col], errors="coerce").to_numpy(dtype=float))
        plt.yticks(y, labels, fontsize=7)
        plt.xlabel(value_col)
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(figdir / fname, dpi=180)
        plt.close()
    barh_top(sender, "delta_vs_best_control", ["scenario", "task", "grounding_target"], "Sender grounding above controls", "Fig1_sender_grounding.png")
    barh_top(receiver, "delta_vs_best_control", ["scenario", "task", "lag"], "Receiver action effect above controls", "Fig2_receiver_action_effect.png")
    barh_top(hidden, "delta_vs_best_control", ["scenario", "task", "hidden_reference_target"], "Hidden-reference effect above controls", "Fig3_hidden_reference.png")
    barh_top(comp, "delta_vs_best_control", ["scenario", "task", "composite_target"], "Compositionality above controls", "Fig4_compositionality.png")
    barh_top(reflex, "delta_vs_best_control", ["scenario", "task", "lag"], "Primary reflex rejection above controls", "Fig5_reflex_rejection.png")
    barh_top(order, "delta_vs_best_control", ["scenario", "task", "order_target"], "Primary order dependence above controls", "Fig6_order_dependence.png")
    barh_top(internal, "delta_vs_best_control", ["scenario", "task"], "Internal grounding over external labels", "Fig7_internal_grounding.png")
    if not primary.empty and "primary_criteria_passed" in primary.columns:
        sub = primary.sort_values(["primary_criteria_passed", "language_like_level"], ascending=False).head(30)
        labels = [f"{r.scenario}|{r.task}" for _, r in sub.iterrows()]
        y = np.arange(len(sub))
        plt.figure(figsize=(12, max(5, 0.34 * len(sub))))
        plt.barh(y, pd.to_numeric(sub["primary_criteria_passed"], errors="coerce").to_numpy(dtype=float))
        plt.yticks(y, labels, fontsize=7)
        plt.xlabel("primary criteria passed")
        plt.title("Primary language-like criteria summary")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(figdir / "Fig8_primary_language_criteria.png", dpi=180)
        plt.close()
    if not levels.empty:
        sub = levels.sort_values("language_like_level", ascending=False).head(30)
        labels = [f"{r.scenario}|{r.task}" for _, r in sub.iterrows()]
        y = np.arange(len(sub))
        plt.figure(figsize=(12, max(5, 0.34 * len(sub))))
        plt.barh(y, pd.to_numeric(sub["language_like_level"], errors="coerce").to_numpy(dtype=float))
        plt.yticks(y, labels, fontsize=7)
        plt.xlabel("language-like level")
        plt.title("Language-like communication level summary")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(figdir / "Fig9_language_like_level.png", dpi=180)
        plt.close()

# =============================================================================
# Self-test
# =============================================================================

def run_self_test() -> None:
    with tempfile.TemporaryDirectory() as td:
        outdir = Path(td)
        rng = np.random.default_rng(123)
        event_rows: List[Dict[str, Any]] = []
        delivery_rows: List[Dict[str, Any]] = []
        action_rows: List[Dict[str, Any]] = []
        state_assay_rows: List[Dict[str, Any]] = []
        memory_rows: List[Dict[str, Any]] = []
        utterance_rows: List[Dict[str, Any]] = []
        for comm in ["SHUFFLED", "FULL_INTERACTIVE"]:
            for seed_id in range(4):
                for step in range(60):
                    high = int(step % 3 == 0)
                    ch = high if comm == "FULL_INTERACTIVE" else int(rng.integers(0, 2))
                    base = {
                        "scenario": "baseline_3d", "gradient_factor": "baseline", "gradient_level": 1.0,
                        "task": "social_reappraisal", "arm": MAIN_ARM, "communication_condition": comm,
                        "seed_id": seed_id, "population_size": 4,
                    }
                    event_rows.append({**base, "sender_id": seed_id % 4, "channel": ch, "raw_channel": ch, "intensity": 0.8, "emitted_step": step, "sender_h": 0.7, "sender_Q": 0.8 if high else 0.1, "sender_Q_R": 0.7 if high else 0.1, "sender_Q_G": 0.5, "sender_physics_score": 0.5, "sender_relief": 0.2 if high else 0.0, "sender_safe_surprise": float(high), "sender_self_appraisal_gap": 0.2 if high else 0.0, "sender_q_relief": 0.1 if high else 0.0, "sender_danger_pressure": 0.6 if high else 0.0, "sender_resource_pressure": 0.0, "sender_unknown_pressure": 0.5 if high else 0.0, "sender_vertical_pressure": 0.0, "sender_friction_pressure": 0.0, "sender_event": "synthetic", "sender_action": "SCAN"})
                    delivery = {**base, "delivery_step": step, "emitted_step": step, "sender_id": seed_id % 4, "receiver_id": (seed_id + 1) % 4, "channel": ch, "raw_channel": ch, "intensity": 0.8, "distance": 2, "distance_bin": "mid", "relative_direction": "N", "sender_Q": 0.8 if high else 0.1, "sender_danger_pressure": 0.6 if high else 0.0, "sender_unknown_pressure": 0.5 if high else 0.0, "sender_resource_pressure": 0.0, "sender_vertical_pressure": 0.0, "sender_h": 0.7, "sender_Q_R": 0.0, "sender_Q_G": 0.0, "sender_physics_score": 0.0, "sender_relief": 0.0, "sender_safe_surprise": 0.0, "sender_self_appraisal_gap": 0.0, "sender_q_relief": 0.0, "sender_friction_pressure": 0.0, "sender_event": "", "sender_action": "", "receiver_danger_pressure": 0.0, "receiver_resource_pressure": 0.0, "receiver_unknown_pressure": 0.0, "receiver_vertical_pressure": 0.0, "hidden_ref_danger": high, "hidden_ref_resource": 0, "hidden_ref_unknown": high, "hidden_ref_vertical": 0}
                    delivery_rows.append(delivery)
                    action_rows.append({**delivery, "lag": 1, "receiver_action": "SCAN" if high else "MOVE_N", "receiver_action_class": "scan" if high else "move", "receiver_event": "synthetic", "receiver_delta_h": 0.01 if high else -0.01, "receiver_damage": 0.0 if high else 0.01, "receiver_resource_gain": 0.0, "receiver_recovery_gain": 0.02 if high else 0.0, "receiver_entered_unknown": 0, "receiver_vertical": 0, "receiver_embodied_value": 0.1 if high else -0.1, "receiver_Q": 0.1, "receiver_body_h": 0.7})
                    # Synthetic V3 state-perturbation assay: FULL_INTERACTIVE changes interpretation across receiver h; SHUFFLED is mostly invariant.
                    for plabel, ph, pact in [("H_LOW", 0.30, "REST" if comm == "FULL_INTERACTIVE" else "SCAN"), ("H_ORIGINAL", 0.55, "SCAN"), ("H_HIGH", 0.75, "MOVE_N" if comm == "FULL_INTERACTIVE" else "SCAN")]:
                        state_assay_rows.append({**base, "agent_id": (seed_id + 1) % 4, "step": step, "dominant_channel": ch, "channel_entropy": 0.0, "signal_inbox_count": 1, "original_body_h": 0.55, "perturbation_label": plabel, "perturbed_body_h": ph, "base_action_after_signal_bias": "SCAN", "assay_action": pact, "assay_action_class": "rest" if pact == "REST" else "scan" if pact == "SCAN" else "move", "assay_score": 1.0, "assay_signal_bias": 0.1, "diverged_from_original_state": int(pact != "SCAN")})
                    utterance_rows.append({**base, "sender_id": seed_id % 4, "step": step, "sequence_length": 2, "sequence": f"{ch}>{ch}", "last_channel": ch, "sender_h": 0.7, "sender_Q": 0.8 if high else 0.1, "sender_Q_R": 0.0, "sender_physics_score": 0.0, "sender_relief": 0.0, "sender_safe_surprise": 0.0, "sender_self_appraisal_gap": 0.0, "sender_danger_pressure": 0.6 if high else 0.0, "sender_resource_pressure": 0.0, "sender_unknown_pressure": 0.5 if high else 0.0, "sender_vertical_pressure": 0.0, "composite_danger_unknown": high, "composite_danger_vertical": 0, "composite_resource_unknown": 0, "composite_safe_recovery": 0})
        write_csv(outdir / "signal_event_log.csv", event_rows)
        write_csv(outdir / "signal_delivery_log.csv", delivery_rows)
        write_csv(outdir / "receiver_signal_action_log.csv", action_rows)
        write_csv(outdir / "state_perturbation_reflex_assay.csv", state_assay_rows)
        write_csv(outdir / "receiver_signal_memory_snapshot.csv", memory_rows)
        write_csv(outdir / "utterance_sequence_log.csv", utterance_rows)
        args = SimpleNamespace(seed=2026, plan="self_test", tasks="social_reappraisal", arms=MAIN_ARM, communication_conditions="SHUFFLED,FULL_INTERACTIVE", population_sizes="4", density_phi=1.0, seeds=4, steps=60)
        logger = Logger(outdir, filename="self_test.log")
        posthoc(outdir, args, logger)
        levels = read_csv_if_exists(outdir / "language_like_level_summary.csv")
        if levels.empty or levels["language_like_level"].max() < 2:
            raise RuntimeError("Self-test failed: language_like_level did not detect synthetic receiver-effective signal.")
        print("SELF_TEST_OK", outdir)

# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Population language-emergence validation for locked v11 autonomous-life cores.")
    p.add_argument("--v11-file", default="", help="Path to DARCA TRUE 3D integrated task battery v11 file.")
    p.add_argument("--darca-file", default="", help="Path to DARCA core file.")
    p.add_argument("--outdir", default="V11_POPULATION_LANGUAGE_EMERGENCE")
    p.add_argument("--plan", choices=["smoke", "quick_language", "main_language", "module_contribution", "size_robustness", "selected_factors"], default="smoke")
    p.add_argument("--factors", default="", help="For selected_factors: resource_density,danger_density,friction_slip,unknown_ambiguity,vertical_complexity")
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--episodes", type=int, default=None, help="Alias for --seeds")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--tasks", default=None)
    p.add_argument("--arms", default=None)
    p.add_argument("--communication-conditions", default=None)
    p.add_argument("--scenario-subset", default="")
    p.add_argument("--population-size", type=int, default=None, help="Fixed population size for main run. Overrides plan population_sizes if provided.")
    p.add_argument("--population-sizes", default=None, help="Comma-separated N values for robustness runs.")
    p.add_argument("--density-phi", type=float, default=None)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--world-seed", type=int, default=9001)
    p.add_argument("--theta", type=float, default=0.70)
    p.add_argument("--causal-horizon", type=int, default=12)
    p.add_argument("--recurrent-N", type=int, default=96)
    p.add_argument("--terminal-h", type=float, default=0.05)
    p.add_argument("--max-consecutive-scans", type=int, default=3)
    p.add_argument("--signal-radius", type=int, default=4)
    p.add_argument("--signal-bias-strength", type=float, default=0.20)
    p.add_argument("--signal-bias-margin", type=float, default=0.04)
    p.add_argument("--sender-preference-strength", type=float, default=0.20)
    p.add_argument("--sender-preference-lr", type=float, default=0.055)
    p.add_argument("--state-perturbation-assay", action=argparse.BooleanOptionalAction, default=True, help="Log bounded receiver state-perturbation assay for L3 reflex rejection.")
    p.add_argument("--state-perturb-max-events-per-episode", type=int, default=2, help="Maximum signal contexts per episode used for state perturbation assay.")
    p.add_argument("--ngram-max-n", type=int, default=3, help="Maximum ordered signal n-gram length for embodied-success analysis.")
    p.add_argument("--ngram-outcome-window", type=int, default=10, help="Reserved metadata value; current implementation uses lagged receiver traces.")
    p.add_argument("--progress-every-runs", type=int, default=10)
    p.add_argument("--progress-every-steps", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--posthoc-only", action="store_true")
    p.add_argument("--self-test", action="store_true", help="Run internal synthetic smoke test without v11 or DARCA.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    defaults = plan_defaults(args.plan, args.factors)
    if args.episodes is not None and args.seeds is None:
        args.seeds = args.episodes
    args.seeds = int(args.seeds if args.seeds is not None else defaults["seeds"])
    args.steps = int(args.steps if args.steps is not None else defaults["steps"])
    args.tasks = args.tasks or defaults["tasks"]
    args.arms = args.arms or defaults["arms"]
    args.communication_conditions = args.communication_conditions or defaults["comm"]
    if args.population_size is not None:
        args.population_sizes = str(int(args.population_size))
    else:
        args.population_sizes = args.population_sizes or defaults["population_sizes"]
    args.density_phi = float(args.density_phi if args.density_phi is not None else defaults["density_phi"])
    args.outdir = Path(args.outdir).expanduser()

    if args.outdir.exists() and any(args.outdir.iterdir()) and args.overwrite and not args.posthoc_only:
        shutil.rmtree(args.outdir)
    ensure_dir(args.outdir)
    logger = Logger(args.outdir)

    scenarios_all = built_in_scenarios()
    if args.scenario_subset.strip():
        wanted = [x.strip() for x in args.scenario_subset.split(",") if x.strip()]
    else:
        wanted = defaults["scenarios"]
    scenarios = [s for s in scenarios_all if s.scenario in set(wanted)]
    missing = sorted(set(wanted) - {s.scenario for s in scenarios})
    if missing:
        raise ValueError(f"Unknown scenarios: {missing}")
    write_csv(args.outdir / "scenario_manifest.csv", [asdict(s) for s in scenarios])

    if not args.posthoc_only:
        if not args.v11_file or not args.darca_file:
            raise RuntimeError("--v11-file and --darca-file are required unless --self-test or --posthoc-only is used.")
        v11_file = Path(args.v11_file).expanduser().resolve()
        darca_file = Path(args.darca_file).expanduser().resolve()
        args.darca_file = darca_file
        if not v11_file.exists():
            raise FileNotFoundError(v11_file)
        if not darca_file.exists():
            raise FileNotFoundError(darca_file)
        lock = {
            "created_at": now(),
            "purpose": "locked-v11 population language-like communication emergence validation",
            "plan": args.plan,
            "v11_file": str(v11_file),
            "v11_sha256": sha256_file(v11_file),
            "darca_file": str(darca_file),
            "darca_sha256": sha256_file(darca_file),
            "seeds": args.seeds,
            "steps": args.steps,
            "tasks": args.tasks,
            "arms": args.arms,
            "communication_conditions": args.communication_conditions,
            "population_sizes": args.population_sizes,
            "density_phi": args.density_phi,
            "signal_radius": args.signal_radius,
            "state_perturbation_assay": args.state_perturbation_assay,
            "state_perturb_max_events_per_episode": args.state_perturb_max_events_per_episode,
            "ngram_max_n": args.ngram_max_n,
            "ngram_outcome_window": args.ngram_outcome_window,
            "guardrail": "v11 internals are imported, not rewritten; receivers only receive channel/intensity/direction/distance; sender states are analysis-only.",
            "primary_language_like_criteria": [
                "receiver-effective signal",
                "reflex rejection",
                "receiver state-perturbation response divergence",
                "pressure hidden-state reference",
                "asymmetric partial-observability hidden-state reference",
                "internal grounding over external labels",
                "future-outcome reference",
                "order dependence",
                "ordered n-gram embodied success",
                "functional code alignment",
                "matched-control counterfactual proxy"
            ],
            "interpretive_guardrail": "This runner tests internally grounded language-like / proto-symbolic communication, not human language or full grammar.",
        }
        (args.outdir / "language_model_lock.json").write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.log(f"locked v11={lock['v11_sha256'][:16]}... darca={lock['darca_sha256'][:16]}...")
        logger.log(f"plan={args.plan} scenarios={len(scenarios)} tasks={args.tasks} arms={args.arms} comm={args.communication_conditions} seeds={args.seeds} steps={args.steps} N={args.population_sizes}")
        v11 = import_module_from_path("locked_v11_language_runtime", v11_file)
        darca_module = v11.load_darca_module(str(darca_file))
        run_all(v11, darca_module, scenarios, args, logger)
    posthoc(args.outdir, args, logger)
    logger.log(f"[complete] outputs written to {args.outdir}")


if __name__ == "__main__":
    main()
