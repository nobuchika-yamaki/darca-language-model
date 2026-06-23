#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal-pathway ablation causal validation for the V11 embodied multi-agent signal system.

This script runs live intervention simulations, not post hoc shuffles. It imports the
existing population runner as a library and reuses the locked agent/environment code,
while replacing the communication condition set with targeted signal-pathway ablations.

Default design:
  scenarios: baseline_3d, danger_x1p25, danger_x1p50, unknown_x1p50, vertical_x1p50
  tasks: exploration_recovery, physics_adaptation, social_reappraisal
  conditions: FULL_INTERACTIVE, NO_SIGNAL_BIAS, NO_RECEIVER_MEMORY_UPDATE,
              ONLINE_CHANNEL_SHUFFLED, NO_SENDER_PREFERENCE
  seeds: 20
  steps: 16000
  population: 8 agents

Key additional outputs:
  01_action_decision_trace.csv
  02_no_signal_counterfactual_trace.csv
  03_receiver_memory_update_trace.csv
  04_signal_packet_lifecycle.csv
  05_sender_preference_update_trace.csv
  06_local_environment_exposure_by_agent_timebin.csv
  07_communication_network_by_timebin.csv
  08_agent_role_by_timebin.csv
  09_sequence_context_trace.csv
  10_safety_override_trace.csv
  12_ablation_primary_causal_contrasts.csv
  13_ablation_mechanism_decision_table.csv

The script intentionally avoids post hoc selection of a main condition. All conditions,
scenarios, tasks, seeds, and time bins are handled by the same fixed metrics.
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
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12
ACTION_CLASSES = ("move", "vertical", "scan", "rest", "approach_sender", "avoid_sender", "enter_unknown", "low_risk_move")
ABLATION_CONDITIONS = (
    "FULL_INTERACTIVE",
    "NO_SIGNAL_BIAS",
    "NO_RECEIVER_MEMORY_UPDATE",
    "ONLINE_CHANNEL_SHUFFLED",
    "NO_SENDER_PREFERENCE",
    "NO_SIGNAL_EMISSION",
)
DEFAULT_SCENARIOS = "baseline_3d,danger_x1p25,danger_x1p50,unknown_x1p50,vertical_x1p50"
DEFAULT_TASKS = "exploration_recovery,physics_adaptation,social_reappraisal"
DEFAULT_CONDITIONS = "FULL_INTERACTIVE,NO_SIGNAL_BIAS,NO_RECEIVER_MEMORY_UPDATE,ONLINE_CHANNEL_SHUFFLED,NO_SENDER_PREFERENCE"
DEFAULT_ARM = "DARCA_Q_PHYSICS_SOCIAL"
TIME_BINS = ((0, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000))
LAGS = (1, 2, 3, 5, 10)

# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(x))))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(*parts: Any) -> int:
    s = "||".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:12], 16)


def normalized_entropy(vals: Sequence[Any], k_total: Optional[int] = None) -> float:
    if not vals:
        return 0.0
    s = pd.Series(list(vals)).value_counts().to_numpy(dtype=float)
    p = s / max(EPS, float(s.sum()))
    h = -float(np.sum(p * np.log2(np.maximum(p, EPS))))
    k = int(k_total) if k_total is not None else int(len(s))
    return float(h / max(EPS, math.log2(max(2, k))))


def mutual_information_discrete(x: Sequence[Any], y: Sequence[Any]) -> float:
    n = min(len(x), len(y))
    if n <= 1:
        return 0.0
    xs = [str(v) for v in list(x)[:n]]
    ys = [str(v) for v in list(y)[:n]]
    x_vals = sorted(set(xs))
    y_vals = sorted(set(ys))
    xi = {v: i for i, v in enumerate(x_vals)}
    yi = {v: i for i, v in enumerate(y_vals)}
    tab = np.zeros((len(x_vals), len(y_vals)), dtype=float)
    for a, b in zip(xs, ys):
        tab[xi[a], yi[b]] += 1.0
    pxy = tab / max(EPS, tab.sum())
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    expected = px @ py
    mask = pxy > 0
    return float(np.sum(pxy[mask] * np.log2(pxy[mask] / np.maximum(expected[mask], EPS))))


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
            out = {k: r.get(k, "") for k in fields}
            w.writerow(out)


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
            w.writerow({k: r.get(k, "") for k in fields})


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def time_bin_label(step: Any) -> str:
    s = safe_int(step, 0)
    for lo, hi in TIME_BINS:
        if lo <= s < hi:
            return f"{lo:05d}_{hi:05d}"
    if s >= TIME_BINS[-1][1]:
        return f"{TIME_BINS[-1][0]:05d}_{TIME_BINS[-1][1]:05d}"
    return "unknown"


def import_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import file: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Logger:
    def __init__(self, outdir: Path, filename: str = "ablation_causal_validation.log"):
        self.path = outdir / filename
        ensure_dir(outdir)
        self.path.write_text("", encoding="utf-8")

    def log(self, msg: str) -> None:
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

# -----------------------------------------------------------------------------
# Ablation condition semantics
# -----------------------------------------------------------------------------

def delivers_signal(condition: str) -> bool:
    return condition != "NO_SIGNAL_EMISSION"


def signal_bias_enabled(condition: str) -> bool:
    return condition not in ("NO_SIGNAL_BIAS", "NO_SIGNAL_EMISSION")


def receiver_memory_update_enabled(condition: str) -> bool:
    return condition not in ("NO_RECEIVER_MEMORY_UPDATE", "NO_SIGNAL_EMISSION")


def sender_preference_update_enabled(condition: str) -> bool:
    return condition not in ("NO_SENDER_PREFERENCE", "NO_SIGNAL_EMISSION")


def sender_preference_remap_enabled(condition: str) -> bool:
    return condition not in ("NO_SENDER_PREFERENCE", "NO_SIGNAL_EMISSION")


def online_channel_shuffled(condition: str) -> bool:
    return condition == "ONLINE_CHANNEL_SHUFFLED"


def causal_condition_description(condition: str) -> str:
    return {
        "FULL_INTERACTIVE": "All signal-pathway mechanisms enabled.",
        "NO_SIGNAL_BIAS": "Signal packets are received and memory is updated, but receiver signal-memory bias is removed from action selection.",
        "NO_RECEIVER_MEMORY_UPDATE": "Signal packets are received, but receiver channel-action memory is not updated.",
        "ONLINE_CHANNEL_SHUFFLED": "Signal timing, delivery, distance, and intensity are preserved, but delivered channel identity is shuffled online.",
        "NO_SENDER_PREFERENCE": "Receiver learning is retained, but sender-side preference updating and preference-based channel remapping are removed.",
        "NO_SIGNAL_EMISSION": "Signal emission is suppressed; negative control for the whole signal pathway.",
    }.get(condition, "Unknown condition")

# -----------------------------------------------------------------------------
# Action scoring with same-state counterfactual trace
# -----------------------------------------------------------------------------

def score_signal_candidates(
    B: Any,
    v11: Any,
    world: Any,
    agent: Any,
    base_action: str,
    rng: random.Random,
    step: int,
    signal_ctx: Dict[str, Any],
    beta: float,
) -> Dict[str, Any]:
    """Score candidate actions with and without receiver-memory signal bias.

    The no-signal counterfactual uses the same candidate set, q/physical costs, sticky
    term, and tie-breaking noise, but sets signal bias to zero. The actual simulator still
    keeps the base v11 action unless a signal-biased candidate exceeds the base-action
    score by the fixed margin.
    """
    channels = list(signal_ctx.get("channels", []))
    candidates = list(v11.MOVE_ACTIONS) + ["REST", "SCAN"]
    rows: List[Dict[str, Any]] = []
    for a in candidates:
        tgt = B.action_target(v11, world, agent.pos, a)
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
        action_classes = B.action_classes_for_action(v11, a, agent.pos, tgt, tmp_ctx)
        sig_bias = agent.signal_memory.action_bias(channels, action_classes, beta=beta) if channels else 0.0
        sticky = 0.18 if a == base_action else 0.0
        q_cost = 0.0
        if agent.q_layer is not None:
            try:
                q_cost = 0.80 * safe_float(agent.q_layer.action_risk_modifier(a, world, agent.pos, agent.mem, step))
            except Exception:
                q_cost = 0.0
        p_cost = 0.0
        pred_damage = pred_gain = pred_wall = 0.0
        if agent.physics is not None:
            try:
                pred = agent.physics.predict(a, world, agent.pos, agent.mem, step)
                pred_damage = safe_float(pred.get("pred_damage"))
                pred_gain = safe_float(pred.get("pred_gain"))
                pred_wall = safe_float(pred.get("pred_wall"))
                p_cost = 0.80 * pred_damage + 0.45 * pred_wall - 0.35 * pred_gain
            except Exception:
                p_cost = 0.0
        noise = rng.random() * 0.02
        score_no_signal = base + sticky - q_cost - p_cost + noise
        score_with_signal = score_no_signal + sig_bias
        rows.append({
            "action": a,
            "action_classes": ",".join(action_classes),
            "base_score": base,
            "sticky": sticky,
            "q_cost": q_cost,
            "p_cost": p_cost,
            "pred_damage": pred_damage,
            "pred_gain": pred_gain,
            "pred_wall": pred_wall,
            "signal_bias": sig_bias,
            "score_no_signal": score_no_signal,
            "score_with_signal": score_with_signal,
        })
    if not rows:
        return {
            "candidate_rows": [],
            "best_action_with_signal": base_action,
            "best_score_with_signal": 0.0,
            "best_signal_bias": 0.0,
            "base_action_score_with_signal": 0.0,
            "base_action_score_no_signal": 0.0,
            "best_action_no_signal": base_action,
            "best_score_no_signal": 0.0,
            "score_delta_best_vs_base": 0.0,
        }
    rows_sorted_signal = sorted(rows, key=lambda r: safe_float(r["score_with_signal"]), reverse=True)
    rows_sorted_no = sorted(rows, key=lambda r: safe_float(r["score_no_signal"]), reverse=True)
    best_sig = rows_sorted_signal[0]
    best_no = rows_sorted_no[0]
    base_row = next((r for r in rows if r["action"] == base_action), rows_sorted_signal[-1])
    return {
        "candidate_rows": rows,
        "best_action_with_signal": best_sig["action"],
        "best_action_with_signal_classes": best_sig["action_classes"],
        "best_score_with_signal": safe_float(best_sig["score_with_signal"]),
        "best_signal_bias": safe_float(best_sig["signal_bias"]),
        "best_q_cost": safe_float(best_sig["q_cost"]),
        "best_p_cost": safe_float(best_sig["p_cost"]),
        "best_pred_damage": safe_float(best_sig["pred_damage"]),
        "best_pred_wall": safe_float(best_sig["pred_wall"]),
        "best_pred_gain": safe_float(best_sig["pred_gain"]),
        "base_action_score_with_signal": safe_float(base_row["score_with_signal"]),
        "base_action_score_no_signal": safe_float(base_row["score_no_signal"]),
        "best_action_no_signal": best_no["action"],
        "best_score_no_signal": safe_float(best_no["score_no_signal"]),
        "score_delta_best_vs_base": safe_float(best_sig["score_with_signal"]) - safe_float(base_row["score_with_signal"]),
    }


def choose_action_from_score_trace(base_action: str, trace: Dict[str, Any], margin: float, enabled: bool) -> Tuple[str, str, float]:
    if not enabled:
        return base_action, "signal_bias_disabled", 0.0
    if not trace.get("candidate_rows"):
        return base_action, "signal_bias_no_candidates", 0.0
    best_action = str(trace.get("best_action_with_signal", base_action))
    best_score = safe_float(trace.get("best_score_with_signal"))
    base_score = safe_float(trace.get("base_action_score_with_signal"))
    best_bias = safe_float(trace.get("best_signal_bias"))
    if best_action != base_action and best_score > base_score + margin:
        return best_action, "receiver_signal_memory_bias", best_bias
    return base_action, "signal_bias_below_margin", best_bias

# -----------------------------------------------------------------------------
# Packet handling
# -----------------------------------------------------------------------------

def copy_packet(B: Any, packet: Any) -> Any:
    p = B.SignalPacket(**asdict(packet))
    if hasattr(packet, "packet_id"):
        setattr(p, "packet_id", getattr(packet, "packet_id"))
    if hasattr(packet, "emitted_channel"):
        setattr(p, "emitted_channel", getattr(packet, "emitted_channel"))
    return p


def transform_packets_ablation(B: Any, packets: List[Any], condition: str, rng: np.random.Generator, n_channels: int = 5) -> List[Any]:
    if condition == "NO_SIGNAL_EMISSION":
        return []
    out = [copy_packet(B, p) for p in packets]
    if not out:
        return []
    if condition == "ONLINE_CHANNEL_SHUFFLED":
        for p in out:
            emitted_channel = int(getattr(p, "channel", -1))
            p.raw_channel = emitted_channel
            setattr(p, "emitted_channel", emitted_channel)
            p.channel = int(rng.integers(0, max(2, int(n_channels))))
            setattr(p, "was_channel_shuffled", 1)
        return out
    for p in out:
        setattr(p, "emitted_channel", int(getattr(p, "channel", -1)))
        setattr(p, "was_channel_shuffled", 0)
    return out

# -----------------------------------------------------------------------------
# Episode runner
# -----------------------------------------------------------------------------

def run_ablation_episode(
    B: Any,
    v11: Any,
    autonomous_core_module: Any,
    scenario: Any,
    task: str,
    arm: str,
    condition: str,
    seed_id: int,
    population_size: int,
    args: argparse.Namespace,
    logger: Logger,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    if condition not in ABLATION_CONDITIONS:
        raise ValueError(f"Unknown ablation condition: {condition}")

    # Paired stochastic seed: the random stream is independent of ablation label.
    # Mechanism changes still alter trajectories, but all conditions start from the same paired seed.
    base_seed = int(args.seed + seed_id * 1009 + stable_hash(scenario.scenario, task, arm, "SIGNAL_PATHWAY_ABLATION_SHARED", population_size) % 1000000)
    rng_np = np.random.default_rng(base_seed & 0xFFFFFFFF)
    rngs = [random.Random(base_seed + aid * 104729) for aid in range(population_size)]

    world, task_args, world_size, z_size, effective_area_scale = B.apply_task_and_population_world(v11, args, scenario, task, population_size, seed_id)
    starts = B.choose_start_positions(v11, world, population_size)
    agents = B.make_agents(v11, autonomous_core_module, arm, task, task_args, seed_id, starts)
    for ag in agents:
        ag.mem.known[ag.pos] = world.actual_kind(ag.pos, 0)
        ag.mem.visited.add(ag.pos)
        setattr(ag, "received_channel_buffer", [])
        setattr(ag, "last_received_signal_step", -1)

    logs: Dict[str, List[Dict[str, Any]]] = {
        "signal_event_rows": [],
        "signal_delivery_rows": [],
        "receiver_action_rows": [],
        "state_perturbation_rows": [],
        "memory_rows": [],
        "preference_rows": [],
        "utterance_rows": [],
        "action_decision_rows": [],
        "no_signal_counterfactual_rows": [],
        "receiver_memory_update_rows": [],
        "signal_packet_lifecycle_rows": [],
        "sender_preference_update_rows": [],
        "sequence_context_rows": [],
        "safety_override_rows": [],
    }
    state_perturbation_count = 0
    total_interference = 0
    total_agent_steps = 0
    packet_counter = 0

    for step in range(int(args.steps)):
        active = [ag for ag in agents if not ag.mem.terminal]
        if not active:
            break
        decisions: Dict[int, Dict[str, Any]] = {}
        targets: Dict[int, Tuple[int, int, int]] = {}
        intended_target_counts: Dict[Tuple[int, int, int], int] = {}
        occupied = {ag.pos: ag.agent_id for ag in active}

        # Decision phase: agents decide from previous-step signal inbox.
        for ag in active:
            signal_ctx = B.compute_signal_context(ag, step, v11)
            if condition == "NO_SIGNAL_EMISSION":
                signal_ctx = B.compute_signal_context(ag, step, v11)
            # Signal coupling to the abstract regulatory core is retained for all signal-delivery ablations.
            y, shock, pr = v11.signal_for_darca(world, ag.pos, ag.mem, step, ag.q_layer)
            has_signal_context = delivers_signal(condition) and signal_ctx.get("n_signals", 0) > 0
            extra = {
                "z": pr["resource_pressure"],
                "exo": pr["unknown_pressure"],
                "d_dyn": pr["vertical_pressure"],
                "friction": pr["friction_pressure"],
                "coupling_t": signal_ctx["coupling_t"] if has_signal_context else 0.0,
                "sigma_t": signal_ctx["sigma_t"] if has_signal_context else 0.0,
            }
            darca_out = ag.darca.step(y, shock, extra) if ag.darca is not None else {}
            action_before_physics = v11.darca_action(darca_out, world, ag.pos, ag.mem, rngs[ag.agent_id], step, ag.q_layer, ag.physics)
            action_source = "autonomous_core"
            if action_before_physics not in v11.ACTIONS:
                action_before_physics = "SCAN"
                action_source += "_invalid_scan"
            action = action_before_physics
            if action == "SCAN" and ag.mem.consecutive_scans >= task_args.max_consecutive_scans:
                if pr["danger_pressure"] < 0.50 and ag.mem.body_h > 0.28:
                    alt = v11.low_risk_non_scan_action(world, ag.pos, ag.mem, rngs[ag.agent_id], step, ag.q_layer, ag.physics)
                    if alt != "SCAN":
                        action = alt
                        action_source += "_scan_loop_escape"
            if action == "SCAN" and world.actual_kind(ag.pos, step) == v11.T_REST and ag.mem.body_h < 0.58:
                action = "REST"
                action_source += "_rest_site_restore"
            action_before_physics_adjustment = action
            phys_reason = "none"
            if ag.physics is not None:
                adjusted_action, phys_reason = ag.physics.best_action_adjustment(action, world, ag.pos, ag.mem, rngs[ag.agent_id], step)
                action = adjusted_action
            base_action_after_physics = action
            phys_adjusted = int(action_before_physics_adjustment != base_action_after_physics)

            sig_bias_value = 0.0
            sig_bias_reason = "none"
            score_trace = score_signal_candidates(B, v11, world, ag, base_action_after_physics, rngs[ag.agent_id], step, signal_ctx, args.signal_bias_strength) if has_signal_context else {}
            if has_signal_context:
                action_candidate, reason, bval = choose_action_from_score_trace(
                    base_action_after_physics,
                    score_trace,
                    args.signal_bias_margin,
                    enabled=signal_bias_enabled(condition),
                )
                if action_candidate != action:
                    action_source += "_" + reason
                action = action_candidate
                sig_bias_value = bval
                sig_bias_reason = reason

                if bool(getattr(args, "state_perturbation_assay", False)) and state_perturbation_count < int(getattr(args, "state_perturb_max_events_per_episode", 2)):
                    assay_rows = B.state_perturbation_action_probe(v11, world, ag, action, signal_ctx, step, args.signal_bias_strength)
                    dom_channel = int(signal_ctx.get("dominant_channel", -1))
                    for ar in assay_rows:
                        ar.update({
                            "scenario": scenario.scenario, "gradient_factor": scenario.gradient_factor, "gradient_level": scenario.gradient_level,
                            "task": task, "arm": arm, "communication_condition": condition, "seed_id": seed_id,
                            "population_size": population_size, "agent_id": ag.agent_id, "step": step,
                            "dominant_channel": dom_channel, "channel_entropy": signal_ctx.get("channel_entropy", 0.0),
                            "original_body_h": safe_float(ag.mem.body_h), "base_action_after_signal_bias": action,
                            "signal_inbox_count": signal_ctx.get("n_signals", 0),
                        })
                        logs["state_perturbation_rows"].append(ar)
                    state_perturbation_count += 1

            final_tgt = B.action_target(v11, world, ag.pos, action)
            final_classes = B.action_classes_for_action(v11, action, ag.pos, final_tgt, signal_ctx)
            action_changed_by_signal = int(has_signal_context and action != base_action_after_physics)
            action_decision_row = {
                "scenario": scenario.scenario,
                "gradient_factor": scenario.gradient_factor,
                "gradient_level": scenario.gradient_level,
                "task": task,
                "arm": arm,
                "communication_condition": condition,
                "ablation_condition": condition,
                "seed_id": seed_id,
                "population_size": population_size,
                "step": step,
                "time_bin": time_bin_label(step),
                "agent_id": ag.agent_id,
                "pos_i": ag.pos[0], "pos_j": ag.pos[1], "pos_k": ag.pos[2],
                "n_received_signals": signal_ctx.get("n_signals", 0),
                "received_channels": "|".join(str(c) for c in signal_ctx.get("channels", [])),
                "dominant_channel": signal_ctx.get("dominant_channel", -1),
                "channel_entropy": signal_ctx.get("channel_entropy", 0.0),
                "mean_signal_intensity": signal_ctx.get("mean_intensity", 0.0),
                "nearest_sender_distance": signal_ctx.get("nearest_sender_distance", 999),
                "action_before_physics": action_before_physics,
                "action_after_scan_rest_rules": action_before_physics_adjustment,
                "base_action_after_physics": base_action_after_physics,
                "final_action": action,
                "final_action_class": ",".join(final_classes),
                "action_changed_by_signal": action_changed_by_signal,
                "action_source": action_source,
                "phys_adjusted": phys_adjusted,
                "phys_adjustment_reason": phys_reason,
                "signal_bias_enabled": int(signal_bias_enabled(condition)),
                "receiver_memory_update_enabled": int(receiver_memory_update_enabled(condition)),
                "sender_preference_update_enabled": int(sender_preference_update_enabled(condition)),
                "online_channel_shuffled": int(online_channel_shuffled(condition)),
                "best_action_with_signal": score_trace.get("best_action_with_signal", ""),
                "best_action_with_signal_class": score_trace.get("best_action_with_signal_classes", ""),
                "best_action_no_signal_candidate": score_trace.get("best_action_no_signal", ""),
                "counterfactual_no_signal_action": base_action_after_physics,
                "score_with_signal_best": score_trace.get("best_score_with_signal", 0.0),
                "score_no_signal_base": score_trace.get("base_action_score_no_signal", 0.0),
                "score_with_signal_base": score_trace.get("base_action_score_with_signal", 0.0),
                "delta_score_due_to_signal": score_trace.get("score_delta_best_vs_base", 0.0),
                "signal_bias_value": sig_bias_value,
                "signal_bias_reason": sig_bias_reason,
                "best_q_cost": score_trace.get("best_q_cost", 0.0),
                "best_p_cost": score_trace.get("best_p_cost", 0.0),
                "best_pred_damage": score_trace.get("best_pred_damage", 0.0),
                "best_pred_wall": score_trace.get("best_pred_wall", 0.0),
                "best_pred_gain": score_trace.get("best_pred_gain", 0.0),
                "body_h": safe_float(ag.mem.body_h),
                "pressure_danger": safe_float(pr.get("danger_pressure")),
                "pressure_resource": safe_float(pr.get("resource_pressure")),
                "pressure_unknown": safe_float(pr.get("unknown_pressure")),
                "pressure_vertical": safe_float(pr.get("vertical_pressure")),
                "pressure_friction": safe_float(pr.get("friction_pressure")),
            }
            if (not args.trace_signal_context_only) or has_signal_context:
                logs["action_decision_rows"].append(action_decision_row)
            if has_signal_context:
                logs["no_signal_counterfactual_rows"].append({
                    "scenario": scenario.scenario,
                    "task": task,
                    "arm": arm,
                    "communication_condition": condition,
                    "ablation_condition": condition,
                    "seed_id": seed_id,
                    "population_size": population_size,
                    "step": step,
                    "time_bin": time_bin_label(step),
                    "agent_id": ag.agent_id,
                    "actual_action": action,
                    "counterfactual_no_signal_action": base_action_after_physics,
                    "actual_action_class": ",".join(final_classes),
                    "signal_changed_action": action_changed_by_signal,
                    "score_with_signal_best": score_trace.get("best_score_with_signal", 0.0),
                    "score_no_signal_base": score_trace.get("base_action_score_no_signal", 0.0),
                    "delta_score_due_to_signal": score_trace.get("score_delta_best_vs_base", 0.0),
                    "dominant_channel": signal_ctx.get("dominant_channel", -1),
                    "received_channels": "|".join(str(c) for c in signal_ctx.get("channels", [])),
                    "signal_bias_value": sig_bias_value,
                })
                recv_buf = list(getattr(ag, "received_channel_buffer", []))
                dominant_channel = int(signal_ctx.get("dominant_channel", -1))
                current_sender = ag.signal_inbox[0].sender_id if ag.signal_inbox else -1
                previous_sender = getattr(ag, "last_signal_sender", -1)
                logs["sequence_context_rows"].append({
                    "scenario": scenario.scenario,
                    "task": task,
                    "arm": arm,
                    "communication_condition": condition,
                    "ablation_condition": condition,
                    "seed_id": seed_id,
                    "population_size": population_size,
                    "step": step,
                    "time_bin": time_bin_label(step),
                    "receiver_id": ag.agent_id,
                    "prev_channel_2": recv_buf[-2] if len(recv_buf) >= 2 else -1,
                    "prev_channel_1": recv_buf[-1] if len(recv_buf) >= 1 else -1,
                    "current_channel": dominant_channel,
                    "received_channels": "|".join(str(c) for c in signal_ctx.get("channels", [])),
                    "time_since_previous_signal": step - int(getattr(ag, "last_received_signal_step", step)) if int(getattr(ag, "last_received_signal_step", -1)) >= 0 else -1,
                    "same_sender_as_previous": int(current_sender == previous_sender) if previous_sender >= 0 else 0,
                    "current_sender_id": current_sender,
                    "receiver_action": action,
                    "receiver_action_class": ",".join(final_classes),
                    "pressure_danger": safe_float(pr.get("danger_pressure")),
                    "pressure_resource": safe_float(pr.get("resource_pressure")),
                    "pressure_unknown": safe_float(pr.get("unknown_pressure")),
                    "pressure_vertical": safe_float(pr.get("vertical_pressure")),
                    "pressure_friction": safe_float(pr.get("friction_pressure")),
                })
                recv_buf.append(dominant_channel)
                setattr(ag, "received_channel_buffer", recv_buf[-3:])
                setattr(ag, "last_received_signal_step", step)
                setattr(ag, "last_signal_sender", current_sender)
            if phys_adjusted or action_changed_by_signal or safe_float(score_trace.get("best_pred_wall", 0.0)) > 0.0 or safe_float(score_trace.get("best_pred_damage", 0.0)) > 0.0:
                logs["safety_override_rows"].append({
                    "scenario": scenario.scenario,
                    "task": task,
                    "arm": arm,
                    "communication_condition": condition,
                    "ablation_condition": condition,
                    "seed_id": seed_id,
                    "population_size": population_size,
                    "step": step,
                    "time_bin": time_bin_label(step),
                    "agent_id": ag.agent_id,
                    "proposed_action": action_before_physics_adjustment,
                    "base_action_after_physics": base_action_after_physics,
                    "signal_biased_action": action,
                    "final_action": action,
                    "phys_adjusted": phys_adjusted,
                    "phys_adjustment_reason": phys_reason,
                    "signal_changed_action": action_changed_by_signal,
                    "predicted_wall_best_signal": score_trace.get("best_pred_wall", 0.0),
                    "predicted_damage_best_signal": score_trace.get("best_pred_damage", 0.0),
                    "viability": safe_float(ag.mem.body_h),
                    "was_signal_effect_blocked": int(has_signal_context and signal_bias_enabled(condition) and action == base_action_after_physics and score_trace.get("best_action_with_signal", base_action_after_physics) != base_action_after_physics),
                    "signal_bias_reason": sig_bias_reason,
                })

            decisions[ag.agent_id] = {
                "action": action,
                "action_source": action_source,
                "darca_out": darca_out,
                "pr": pr,
                "old_pos": ag.pos,
                "old_h": ag.mem.body_h,
                "old_q": safe_float(getattr(ag.q_layer, "q", 0.0)),
                "signal_ctx": signal_ctx,
                "signal_bias_value": sig_bias_value,
                "signal_bias_reason": sig_bias_reason,
            }
            targets[ag.agent_id] = final_tgt
            if action in v11.MOVE_ACTIONS and world.in_bounds(final_tgt):
                intended_target_counts[final_tgt] = intended_target_counts.get(final_tgt, 0) + 1

        # Conflict resolution.
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

        emitted_packets: List[Any] = []

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
                outcome = B.make_interference_outcome(v11, world, ag.pos, action, step)
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
            q_state = ag.q_layer.update(ag.pos, action, outcome, d["pr"], ag.mem, d["darca_out"]) if ag.q_layer is not None else B.default_q_state()
            phys_state = ag.physics.update(action, outcome, pred) if ag.physics is not None else {"physics_pred_error": 0.0, "physics_score": 0.0, "physics_action_n": 0.0}
            social_state = ag.social.update(ag.mem, q_state, d["pr"], outcome, action) if ag.social is not None else B.default_social_state()
            total_agent_steps += 1

            action_classes = B.action_classes_for_action(v11, action, old_pos, ag.pos, d["signal_ctx"])
            embodied_value, outcome_info = B.embodied_value_from_outcome(old_h, old_q, q_state, outcome, social_state)

            # Receiver memory update and sender-feedback trace are separated.
            if delivers_signal(condition):
                for packet in ag.signal_inbox:
                    ch = int(packet.channel)
                    for ac in action_classes:
                        before = safe_float(ag.signal_memory.action_value.get(ch, {}).get(ac, 0.0)) if ch >= 0 else 0.0
                        channel_value_before = safe_float(ag.signal_memory.channel_value.get(ch, 0.0)) if ch >= 0 else 0.0
                        if receiver_memory_update_enabled(condition):
                            # Update once per packet, not once per action class. We update before writing the first action-class row.
                            pass
                        logs["receiver_memory_update_rows"].append({
                            "scenario": scenario.scenario,
                            "task": task,
                            "arm": arm,
                            "communication_condition": condition,
                            "ablation_condition": condition,
                            "seed_id": seed_id,
                            "population_size": population_size,
                            "step": step,
                            "time_bin": time_bin_label(step),
                            "receiver_id": ag.agent_id,
                            "sender_id": packet.sender_id,
                            "packet_id": getattr(packet, "packet_id", ""),
                            "channel": ch,
                            "raw_channel": int(getattr(packet, "raw_channel", ch)),
                            "emitted_channel": int(getattr(packet, "emitted_channel", getattr(packet, "raw_channel", ch))),
                            "action": action,
                            "action_class": ac,
                            "M_before": before,
                            "M_after": "PENDING",
                            "channel_value_before": channel_value_before,
                            "channel_value_after": "PENDING",
                            "embodied_outcome_y": embodied_value,
                            "delta_h": safe_float(outcome.delta_h),
                            "damage": safe_float(outcome.damage),
                            "recovery": safe_float(outcome.recovery_gain),
                            "resource_gain": safe_float(outcome.resource_gain),
                            "wall_collision": int(outcome.hit_wall),
                            "delta_q": safe_float(q_state.get("Q")) - old_q,
                            "relief": safe_float(social_state.get("relief")),
                            "memory_update_enabled": int(receiver_memory_update_enabled(condition)),
                        })
                    if receiver_memory_update_enabled(condition):
                        ag.signal_memory.update(ch, action_classes, embodied_value, outcome_info)
                        # Patch pending M_after values for this packet/action classes.
                        for rr in reversed(logs["receiver_memory_update_rows"]):
                            if rr.get("packet_id") == getattr(packet, "packet_id", "") and rr.get("receiver_id") == ag.agent_id and rr.get("step") == step:
                                ac = rr.get("action_class")
                                rr["M_after"] = safe_float(ag.signal_memory.action_value.get(ch, {}).get(ac, 0.0))
                                rr["channel_value_after"] = safe_float(ag.signal_memory.channel_value.get(ch, 0.0))
                            else:
                                # recent rows for this packet are contiguous
                                if rr.get("step") == step:
                                    continue
                                break
                    else:
                        for rr in reversed(logs["receiver_memory_update_rows"]):
                            if rr.get("packet_id") == getattr(packet, "packet_id", "") and rr.get("receiver_id") == ag.agent_id and rr.get("step") == step:
                                rr["M_after"] = rr["M_before"]
                                rr["channel_value_after"] = rr["channel_value_before"]
                            else:
                                if rr.get("step") == step:
                                    continue
                                break
                    if sender_preference_update_enabled(condition):
                        ag.pending_traces.append({
                            "type": "sender_feedback", "sender_id": packet.sender_id, "channel": ch,
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
                    logs["receiver_action_rows"].append(rr)
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
                "communication_condition": condition,
                "seed_id": seed_id,
                "population_size": population_size,
                "density_phi": args.density_phi,
                "world_size": world_size,
                "z_size": z_size,
                "agent_id": ag.agent_id,
                "step": step,
                "time_bin": time_bin_label(step),
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
                    row[f"core_{k}"] = d["darca_out"][k]
            ag.records.append(row)

            # Emit anonymous signal packet unless signal emission is ablated.
            if condition != "NO_SIGNAL_EMISSION" and safe_float(social_state.get("social_signal")) > 0.0 and int(social_state.get("selected_signal_channel", -1)) >= 0:
                raw_ch = int(social_state.get("selected_signal_channel", -1))
                packet = B.SignalPacket(
                    sender_id=ag.agent_id,
                    channel=raw_ch,
                    intensity=max(0.05, safe_float(social_state.get("signal_probability_max", 0.5))),
                    sender_pos=ag.pos,
                    emitted_step=step,
                    raw_channel=raw_ch,
                    communication_condition=condition,
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
                setattr(packet, "packet_id", f"{scenario.scenario}|{task}|{condition}|seed{seed_id}|step{step}|sender{ag.agent_id}|pkt{packet_counter}")
                setattr(packet, "emitted_channel", raw_ch)
                packet_counter += 1
                if sender_preference_remap_enabled(condition):
                    packet = B.maybe_remap_channel_by_sender_preference(packet, ag, rng_np, args.sender_preference_strength, int(getattr(ag.social, "n_signals", 5)))
                    setattr(packet, "emitted_channel", raw_ch)
                emitted_packets.append(packet)
                ag.utterance_buffer.append(packet.channel)
                ag.utterance_buffer = ag.utterance_buffer[-3:]
                ev = B.signal_packet_event_row(packet, scenario, task, arm, condition, seed_id, population_size, world_size, z_size)
                ev["packet_id"] = getattr(packet, "packet_id", "")
                ev["time_bin"] = time_bin_label(step)
                logs["signal_event_rows"].append(ev)
                if len(ag.utterance_buffer) >= 2:
                    logs["utterance_rows"].append(B.utterance_row_from_buffer(ag, packet, scenario, task, arm, condition, seed_id, 2))
                if len(ag.utterance_buffer) >= 3:
                    logs["utterance_rows"].append(B.utterance_row_from_buffer(ag, packet, scenario, task, arm, condition, seed_id, 3))

            # Clear consumed inbox. New delivery happens after all agents emit.
            ag.signal_inbox = []

        # Sender feedback update.
        if sender_preference_update_enabled(condition):
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
                    newv = (1.0 - args.sender_preference_lr) * cur + args.sender_preference_lr * float(np.mean(vals))
                    agents[sid].sender_channel_preference[ch] = newv
                    logs["sender_preference_update_rows"].append({
                        "scenario": scenario.scenario,
                        "task": task,
                        "arm": arm,
                        "communication_condition": condition,
                        "ablation_condition": condition,
                        "seed_id": seed_id,
                        "population_size": population_size,
                        "step": step,
                        "time_bin": time_bin_label(step),
                        "sender_id": sid,
                        "channel": ch,
                        "preference_before": cur,
                        "mean_receiver_benefit": float(np.mean(vals)),
                        "preference_after": newv,
                        "sender_preference_update_enabled": 1,
                    })

        # Packet transformation and delivery.
        n_channels = 5
        try:
            n_channels = int(max([getattr(ag.social, "n_signals", 5) for ag in agents if ag.social is not None] + [5]))
        except Exception:
            n_channels = 5
        packets_to_deliver = transform_packets_ablation(B, emitted_packets, condition, rng_np, n_channels=n_channels)
        if delivers_signal(condition):
            for packet in packets_to_deliver:
                for rec in agents:
                    if rec.mem.terminal or rec.agent_id == packet.sender_id:
                        continue
                    dist = B.manhattan(packet.sender_pos, rec.pos)
                    if dist > int(args.signal_radius):
                        continue
                    rec.signal_inbox.append(packet)
                    rec_pr = v11.local_pressures(world, rec.pos, rec.mem, step)
                    delivery = B.signal_delivery_row(packet, rec, rec_pr, scenario, task, arm, condition, seed_id, population_size, dist, step)
                    sender_agent = agents[packet.sender_id] if 0 <= int(packet.sender_id) < len(agents) else None
                    delivery.update(B.asymmetric_known_reference_flags(v11, world, sender_agent, rec, step))
                    delivery["packet_id"] = getattr(packet, "packet_id", "")
                    delivery["emitted_channel"] = int(getattr(packet, "emitted_channel", getattr(packet, "raw_channel", packet.channel)))
                    delivery["delivered_channel"] = int(packet.channel)
                    delivery["was_channel_shuffled"] = int(getattr(packet, "was_channel_shuffled", 0))
                    delivery["time_bin"] = time_bin_label(step)
                    logs["signal_delivery_rows"].append(delivery)
                    logs["signal_packet_lifecycle_rows"].append({
                        "scenario": scenario.scenario,
                        "task": task,
                        "arm": arm,
                        "communication_condition": condition,
                        "ablation_condition": condition,
                        "seed_id": seed_id,
                        "population_size": population_size,
                        "packet_id": getattr(packet, "packet_id", ""),
                        "emission_step": int(getattr(packet, "emitted_step", step)),
                        "delivery_step": step,
                        "time_bin": time_bin_label(step),
                        "sender_id": packet.sender_id,
                        "receiver_id": rec.agent_id,
                        "emitted_channel": int(getattr(packet, "emitted_channel", getattr(packet, "raw_channel", packet.channel))),
                        "delivered_channel": int(packet.channel),
                        "raw_channel": int(getattr(packet, "raw_channel", packet.channel)),
                        "was_channel_shuffled": int(getattr(packet, "was_channel_shuffled", 0)),
                        "signal_intensity": safe_float(packet.intensity),
                        "sender_pos_i": packet.sender_pos[0], "sender_pos_j": packet.sender_pos[1], "sender_pos_k": packet.sender_pos[2],
                        "receiver_pos_i": rec.pos[0], "receiver_pos_j": rec.pos[1], "receiver_pos_k": rec.pos[2],
                        "distance": dist,
                    })
                    trace = dict(delivery)
                    trace["delivery_step"] = step
                    trace["type"] = "receiver_lag_trace"
                    rec.pending_traces.append(trace)

        if args.progress_every_steps > 0 and (step + 1) % args.progress_every_steps == 0:
            active_n = sum(1 for ag in agents if not ag.mem.terminal)
            logger.log(f"progress scenario={scenario.scenario} task={task} cond={condition} seed={seed_id} step={step+1}/{args.steps} active={active_n} emitted={len(emitted_packets)}")

    # Per-agent and population summaries.
    per_agent_rows: List[Dict[str, Any]] = []
    memory_rows: List[Dict[str, Any]] = []
    preference_rows: List[Dict[str, Any]] = []
    for ag in agents:
        recs = ag.records
        df = pd.DataFrame(recs)
        actions = df["action"].tolist() if not df.empty and "action" in df.columns else []
        action_counts = pd.Series(actions).value_counts().to_dict() if actions else {}
        row = {
            "scenario": scenario.scenario,
            "gradient_factor": scenario.gradient_factor,
            "gradient_level": scenario.gradient_level,
            "task": task,
            "task_family": v11.task_family(task),
            "arm": arm,
            "communication_condition": condition,
            "seed_id": seed_id,
            "population_size": population_size,
            "density_phi": args.density_phi,
            "world_size": world_size,
            "z_size": z_size,
            "agent_id": ag.agent_id,
            "steps_run": len(recs),
            "steps_completed": len(recs),
            "survived": int(not ag.mem.terminal),
            "final_body_h": safe_float(ag.mem.body_h),
            "mean_body_h": float(pd.to_numeric(df.get("body_h", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()) if not df.empty else safe_float(ag.mem.body_h),
            "terminal": int(ag.mem.terminal),
            "coverage": len(ag.mem.visited) / float(world.size * world.size * world.z_size),
            "resources": ag.mem.resources,
            "total_damage": ag.mem.total_damage,
            "total_resource_gain": ag.mem.total_resource_gain,
            "recovery_events": ag.mem.recovery_events,
            "rest_steps": ag.mem.rest_steps,
            "scans": ag.mem.scans,
            "action_entropy_norm": normalized_entropy(actions, k_total=len(getattr(v11, "ACTIONS", []))) if actions else 0.0,
            "vertical_action_fraction": float(np.mean([1 if str(a).endswith("UP") or str(a).endswith("DOWN") else 0 for a in actions])) if actions else 0.0,
            "entered_unknown_fraction": float(pd.to_numeric(df.get("entered_unknown", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()) if not df.empty else 0.0,
            "mean_Q": float(pd.to_numeric(df.get("Q", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()) if not df.empty else 0.0,
            "mean_physics_score": float(pd.to_numeric(df.get("physics_score", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()) if not df.empty else 0.0,
            "mean_social_signal": float(pd.to_numeric(df.get("social_signal", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()) if not df.empty else 0.0,
            "mean_signal_inbox_count": float(pd.to_numeric(df.get("signal_inbox_count", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()) if not df.empty else 0.0,
            "mean_receiver_signal_bias": float(pd.to_numeric(df.get("receiver_signal_bias_value", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()) if not df.empty else 0.0,
            "interference_rate": float(pd.to_numeric(df.get("inter_agent_interference", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()) if not df.empty else 0.0,
            "autonomy_proper_index": float(pd.to_numeric(df.get("core_autonomy", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()) if not df.empty else 0.0,
            "system_sovereignty": float(pd.to_numeric(df.get("core_identity", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()) if not df.empty else 0.0,
            "information_theoretic_autonomy": float(pd.to_numeric(df.get("core_causal_confidence", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()) if not df.empty else 0.0,
            "resilience_sacrifice": 0.0,
            "heteronomy_index": 0.0,
            "q_action_coupling": 0.0,
            "q_risk_suppression": 0.0,
            "physics_adaptation_delta": 0.0,
            "social_signal_rate": float(pd.to_numeric(df.get("social_signal", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()) if not df.empty else 0.0,
            "receiver_recovery_total": float(pd.to_numeric(df.get("receiver_recovery_total", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()) if not df.empty else 0.0,
        }
        for a in getattr(v11, "ACTIONS", []):
            row[f"action_count_{a}"] = int(action_counts.get(a, 0))
        per_agent_rows.append(row)
        memory_rows.extend(ag.signal_memory.rows(scenario.scenario, task, arm, condition, seed_id, ag.agent_id, int(args.steps)))
        for ch, val in sorted(ag.sender_channel_preference.items()):
            preference_rows.append({
                "scenario": scenario.scenario, "task": task, "arm": arm, "communication_condition": condition,
                "seed_id": seed_id, "agent_id": ag.agent_id, "channel": ch, "sender_channel_preference": val,
            })

    pop_df = pd.DataFrame(per_agent_rows)
    final_positions = [ag.pos for ag in agents]
    try:
        mpd = float(np.mean(B.pairwise_distances(final_positions))) if len(final_positions) > 1 else 0.0
    except Exception:
        mpd = 0.0
    collective_visited = set()
    for ag in agents:
        collective_visited.update(getattr(ag.mem, "visited", set()))
    world_volume = int(world.size * world.size * world.z_size)
    pop_row = {
        "scenario": scenario.scenario,
        "gradient_factor": scenario.gradient_factor,
        "gradient_level": scenario.gradient_level,
        "task": task,
        "task_family": v11.task_family(task),
        "arm": arm,
        "communication_condition": condition,
        "seed_id": seed_id,
        "population_size": population_size,
        "density_phi": args.density_phi,
        "world_size": world_size,
        "z_size": z_size,
        "world_volume": world_volume,
        "steps_requested": int(args.steps),
        "steps": int(args.steps),
        "total_agent_steps": total_agent_steps,
        "actual_agent_steps": total_agent_steps,
        "active_agents_final": int(sum(1 for ag in agents if not ag.mem.terminal)),
        "terminal_agents": int(pop_df["terminal"].sum()) if not pop_df.empty else 0,
        "survival_fraction": float((1.0 - pop_df["terminal"].mean())) if not pop_df.empty else 0.0,
        "collective_coverage": len(collective_visited) / max(1, world_volume),
        "per_capita_collective_coverage": (len(collective_visited) / max(1, world_volume)) / max(1, population_size),
        "mean_pairwise_distance_final": mpd,
        "aggregation_index_final": 1.0 / (1.0 + mpd),
        "interference_rate_population": total_interference / max(1, total_agent_steps),
        "q_synchrony": 0.0,
        "s_synchrony": 0.0,
        "mean_final_body_h": float(pop_df["final_body_h"].mean()) if not pop_df.empty else 0.0,
        "mean_coverage": float(pop_df["coverage"].mean()) if not pop_df.empty else 0.0,
        "mean_action_entropy_norm": float(pop_df["action_entropy_norm"].mean()) if not pop_df.empty else 0.0,
        "mean_vertical_action_fraction": float(pop_df["vertical_action_fraction"].mean()) if not pop_df.empty else 0.0,
        "mean_entered_unknown_fraction": float(pop_df["entered_unknown_fraction"].mean()) if not pop_df.empty else 0.0,
        "mean_Q": float(pop_df["mean_Q"].mean()) if not pop_df.empty else 0.0,
        "mean_physics_score": float(pop_df["mean_physics_score"].mean()) if not pop_df.empty else 0.0,
        "mean_social_signal": float(pop_df["mean_social_signal"].mean()) if not pop_df.empty else 0.0,
        "mean_signal_inbox_count": float(pop_df["mean_signal_inbox_count"].mean()) if not pop_df.empty else 0.0,
        "mean_receiver_signal_bias": float(pop_df["mean_receiver_signal_bias"].mean()) if not pop_df.empty else 0.0,
        "signal_event_count": len(logs["signal_event_rows"]),
        "signal_delivery_count": len(logs["signal_delivery_rows"]),
        "receiver_action_trace_count": len(logs["receiver_action_rows"]),
        "state_perturbation_assay_count": len(logs["state_perturbation_rows"]),
        "utterance_sequence_count": len(logs["utterance_rows"]),
        "action_changed_by_signal_count": int(sum(safe_int(r.get("action_changed_by_signal")) for r in logs["action_decision_rows"])),
        "total_interference": total_interference,
    }
    for col in ["autonomy_proper_index", "final_body_h", "coverage", "resources", "total_damage", "action_entropy_norm", "vertical_action_fraction", "entered_unknown_fraction", "mean_Q", "mean_physics_score", "mean_social_signal", "mean_signal_inbox_count", "mean_receiver_signal_bias"]:
        if col in pop_df.columns:
            pop_row[f"agent_mean_{col}"] = float(pd.to_numeric(pop_df[col], errors="coerce").fillna(0).mean())
            pop_row[f"agent_sd_{col}"] = float(pd.to_numeric(pop_df[col], errors="coerce").fillna(0).std(ddof=0))

    logs["memory_rows"] = memory_rows
    logs["preference_rows"] = preference_rows
    logs["local_environment_exposure_rows"] = summarize_local_exposure(per_agent_rows, agents, scenario, task, arm, condition, seed_id, population_size)
    logs["communication_network_rows"] = summarize_communication_network(logs["signal_packet_lifecycle_rows"])
    logs["agent_role_rows"] = summarize_agent_roles(agents, logs, scenario, task, arm, condition, seed_id, population_size)
    return per_agent_rows, pop_row, logs

# -----------------------------------------------------------------------------
# Summaries from within-episode traces
# -----------------------------------------------------------------------------

def summarize_local_exposure(per_agent_rows: List[Dict[str, Any]], agents: Sequence[Any], scenario: Any, task: str, arm: str, condition: str, seed_id: int, population_size: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ag in agents:
        df = pd.DataFrame(getattr(ag, "records", []))
        if df.empty:
            continue
        if "time_bin" not in df.columns:
            df["time_bin"] = df["step"].map(time_bin_label)
        for tb, sub in df.groupby("time_bin", dropna=False):
            out.append({
                "scenario": scenario.scenario,
                "task": task,
                "arm": arm,
                "communication_condition": condition,
                "ablation_condition": condition,
                "seed_id": seed_id,
                "population_size": population_size,
                "agent_id": ag.agent_id,
                "time_bin": tb,
                "n_steps": int(sub.shape[0]),
                "danger_pressure_mean": float(pd.to_numeric(sub.get("pressure_danger_pressure", sub.get("pressure_danger", 0.0)), errors="coerce").fillna(0).mean()),
                "resource_pressure_mean": float(pd.to_numeric(sub.get("pressure_resource_pressure", sub.get("pressure_resource", 0.0)), errors="coerce").fillna(0).mean()),
                "unknown_pressure_mean": float(pd.to_numeric(sub.get("pressure_unknown_pressure", sub.get("pressure_unknown", 0.0)), errors="coerce").fillna(0).mean()),
                "vertical_pressure_mean": float(pd.to_numeric(sub.get("pressure_vertical_pressure", sub.get("pressure_vertical", 0.0)), errors="coerce").fillna(0).mean()),
                "friction_pressure_mean": float(pd.to_numeric(sub.get("pressure_friction_pressure", sub.get("pressure_friction", 0.0)), errors="coerce").fillna(0).mean()),
                "unknown_entries": int(pd.to_numeric(sub.get("entered_unknown", 0), errors="coerce").fillna(0).sum()),
                "danger_contacts": int((pd.to_numeric(sub.get("damage", 0), errors="coerce").fillna(0) > 0).sum()),
                "resource_contacts": int((pd.to_numeric(sub.get("resource_gain", 0), errors="coerce").fillna(0) > 0).sum()),
                "wall_contacts": int(pd.to_numeric(sub.get("hit_wall", 0), errors="coerce").fillna(0).sum()),
                "vertical_moves": int(sub.get("action", pd.Series([], dtype=str)).astype(str).str.contains("UP|DOWN", regex=True).sum()),
                "visited_cells": int(len(getattr(ag.mem, "visited", []))),
            })
    return out


def summarize_communication_network(lifecycle_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not lifecycle_rows:
        return []
    df = pd.DataFrame(lifecycle_rows)
    rows: List[Dict[str, Any]] = []
    for keys, sub in df.groupby(["scenario", "task", "arm", "communication_condition", "ablation_condition", "seed_id", "population_size", "time_bin", "sender_id", "receiver_id"], dropna=False):
        scenario, task, arm, cond, abl, seed_id, n, tb, sid, rid = keys
        sent_ch = sub["emitted_channel"].astype(str).tolist() if "emitted_channel" in sub else []
        recv_ch = sub["delivered_channel"].astype(str).tolist() if "delivered_channel" in sub else []
        rows.append({
            "scenario": scenario,
            "task": task,
            "arm": arm,
            "communication_condition": cond,
            "ablation_condition": abl,
            "seed_id": seed_id,
            "population_size": n,
            "time_bin": tb,
            "sender_id": sid,
            "receiver_id": rid,
            "delivered_count": int(sub.shape[0]),
            "mean_distance": float(pd.to_numeric(sub.get("distance", 0), errors="coerce").fillna(0).mean()),
            "channel_entropy_sent": normalized_entropy(sent_ch, k_total=5),
            "channel_entropy_received": normalized_entropy(recv_ch, k_total=5),
            "dominant_sent_channel": pd.Series(sent_ch).mode().iloc[0] if sent_ch else -1,
            "dominant_received_channel": pd.Series(recv_ch).mode().iloc[0] if recv_ch else -1,
        })
    return rows


def summarize_agent_roles(agents: Sequence[Any], logs: Dict[str, List[Dict[str, Any]]], scenario: Any, task: str, arm: str, condition: str, seed_id: int, population_size: int) -> List[Dict[str, Any]]:
    sent_df = pd.DataFrame(logs.get("signal_event_rows", []))
    recv_df = pd.DataFrame(logs.get("signal_packet_lifecycle_rows", []))
    mem_df = pd.DataFrame(logs.get("receiver_memory_update_rows", []))
    dec_df = pd.DataFrame(logs.get("action_decision_rows", []))
    rows: List[Dict[str, Any]] = []
    all_bins = [f"{lo:05d}_{hi:05d}" for lo, hi in TIME_BINS]
    for ag in agents:
        rec_df = pd.DataFrame(getattr(ag, "records", []))
        if not rec_df.empty and "time_bin" not in rec_df.columns:
            rec_df["time_bin"] = rec_df["step"].map(time_bin_label)
        for tb in all_bins:
            sub = rec_df[rec_df["time_bin"] == tb] if not rec_df.empty else pd.DataFrame()
            sent_n = int(((sent_df.get("sender_id", pd.Series(dtype=int)) == ag.agent_id) & (sent_df.get("time_bin", pd.Series(dtype=str)) == tb)).sum()) if not sent_df.empty else 0
            recv_n = int(((recv_df.get("receiver_id", pd.Series(dtype=int)) == ag.agent_id) & (recv_df.get("time_bin", pd.Series(dtype=str)) == tb)).sum()) if not recv_df.empty else 0
            mem_n = int(((mem_df.get("receiver_id", pd.Series(dtype=int)) == ag.agent_id) & (mem_df.get("time_bin", pd.Series(dtype=str)) == tb) & (mem_df.get("memory_update_enabled", pd.Series(dtype=int)) == 1)).sum()) if not mem_df.empty else 0
            changed_n = int(((dec_df.get("agent_id", pd.Series(dtype=int)) == ag.agent_id) & (dec_df.get("time_bin", pd.Series(dtype=str)) == tb) & (dec_df.get("action_changed_by_signal", pd.Series(dtype=int)) == 1)).sum()) if not dec_df.empty else 0
            n_steps = int(sub.shape[0]) if not sub.empty else 0
            rows.append({
                "scenario": scenario.scenario,
                "task": task,
                "arm": arm,
                "communication_condition": condition,
                "ablation_condition": condition,
                "seed_id": seed_id,
                "population_size": population_size,
                "agent_id": ag.agent_id,
                "time_bin": tb,
                "n_steps": n_steps,
                "signals_sent": sent_n,
                "signals_received": recv_n,
                "receiver_memory_update_count": mem_n,
                "action_changed_by_signal_count": changed_n,
                "exploration_fraction": float((sub.get("action", pd.Series([], dtype=str)).astype(str).isin(["SCAN"]).mean())) if n_steps else 0.0,
                "vertical_fraction": float(sub.get("action", pd.Series([], dtype=str)).astype(str).str.contains("UP|DOWN", regex=True).mean()) if n_steps else 0.0,
                "unknown_entry_fraction": float(pd.to_numeric(sub.get("entered_unknown", 0), errors="coerce").fillna(0).mean()) if n_steps else 0.0,
                "damage_total": float(pd.to_numeric(sub.get("damage", 0), errors="coerce").fillna(0).sum()) if n_steps else 0.0,
                "recovery_total": float(pd.to_numeric(sub.get("recovery_gain", 0), errors="coerce").fillna(0).sum()) if n_steps else 0.0,
                "viability_mean": float(pd.to_numeric(sub.get("body_h", 0), errors="coerce").fillna(0).mean()) if n_steps else 0.0,
                "role_score_sender": sent_n / max(1, sent_n + recv_n),
                "role_score_receiver": recv_n / max(1, sent_n + recv_n),
                "role_score_explorer": float(pd.to_numeric(sub.get("entered_unknown", 0), errors="coerce").fillna(0).mean()) if n_steps else 0.0,
            })
    return rows

# -----------------------------------------------------------------------------
# Output field definitions
# -----------------------------------------------------------------------------

RUN_FIELDS = ["run_key", "status", "error", "created_at", "scenario", "task", "arm", "communication_condition", "seed_id", "population_size", "density_phi", "elapsed_sec"]

EXTRA_SIGNAL_EVENT_FIELDS = ["packet_id", "time_bin"]
ACTION_DECISION_FIELDS = [
    "scenario", "gradient_factor", "gradient_level", "task", "arm", "communication_condition", "ablation_condition", "seed_id", "population_size", "step", "time_bin", "agent_id",
    "pos_i", "pos_j", "pos_k", "n_received_signals", "received_channels", "dominant_channel", "channel_entropy", "mean_signal_intensity", "nearest_sender_distance",
    "action_before_physics", "action_after_scan_rest_rules", "base_action_after_physics", "final_action", "final_action_class", "action_changed_by_signal", "action_source",
    "phys_adjusted", "phys_adjustment_reason", "signal_bias_enabled", "receiver_memory_update_enabled", "sender_preference_update_enabled", "online_channel_shuffled",
    "best_action_with_signal", "best_action_with_signal_class", "best_action_no_signal_candidate", "counterfactual_no_signal_action",
    "score_with_signal_best", "score_no_signal_base", "score_with_signal_base", "delta_score_due_to_signal", "signal_bias_value", "signal_bias_reason",
    "best_q_cost", "best_p_cost", "best_pred_damage", "best_pred_wall", "best_pred_gain", "body_h",
    "pressure_danger", "pressure_resource", "pressure_unknown", "pressure_vertical", "pressure_friction"
]
NO_SIGNAL_COUNTERFACTUAL_FIELDS = ["scenario", "task", "arm", "communication_condition", "ablation_condition", "seed_id", "population_size", "step", "time_bin", "agent_id", "actual_action", "counterfactual_no_signal_action", "actual_action_class", "signal_changed_action", "score_with_signal_best", "score_no_signal_base", "delta_score_due_to_signal", "dominant_channel", "received_channels", "signal_bias_value"]
RECEIVER_MEMORY_UPDATE_FIELDS = ["scenario", "task", "arm", "communication_condition", "ablation_condition", "seed_id", "population_size", "step", "time_bin", "receiver_id", "sender_id", "packet_id", "channel", "raw_channel", "emitted_channel", "action", "action_class", "M_before", "M_after", "channel_value_before", "channel_value_after", "embodied_outcome_y", "delta_h", "damage", "recovery", "resource_gain", "wall_collision", "delta_q", "relief", "memory_update_enabled"]
SIGNAL_PACKET_LIFECYCLE_FIELDS = ["scenario", "task", "arm", "communication_condition", "ablation_condition", "seed_id", "population_size", "packet_id", "emission_step", "delivery_step", "time_bin", "sender_id", "receiver_id", "emitted_channel", "delivered_channel", "raw_channel", "was_channel_shuffled", "signal_intensity", "sender_pos_i", "sender_pos_j", "sender_pos_k", "receiver_pos_i", "receiver_pos_j", "receiver_pos_k", "distance"]
SENDER_PREFERENCE_UPDATE_FIELDS = ["scenario", "task", "arm", "communication_condition", "ablation_condition", "seed_id", "population_size", "step", "time_bin", "sender_id", "channel", "preference_before", "mean_receiver_benefit", "preference_after", "sender_preference_update_enabled"]
LOCAL_ENV_EXPOSURE_FIELDS = ["scenario", "task", "arm", "communication_condition", "ablation_condition", "seed_id", "population_size", "agent_id", "time_bin", "n_steps", "danger_pressure_mean", "resource_pressure_mean", "unknown_pressure_mean", "vertical_pressure_mean", "friction_pressure_mean", "unknown_entries", "danger_contacts", "resource_contacts", "wall_contacts", "vertical_moves", "visited_cells"]
COMM_NETWORK_FIELDS = ["scenario", "task", "arm", "communication_condition", "ablation_condition", "seed_id", "population_size", "time_bin", "sender_id", "receiver_id", "delivered_count", "mean_distance", "channel_entropy_sent", "channel_entropy_received", "dominant_sent_channel", "dominant_received_channel"]
AGENT_ROLE_FIELDS = ["scenario", "task", "arm", "communication_condition", "ablation_condition", "seed_id", "population_size", "agent_id", "time_bin", "n_steps", "signals_sent", "signals_received", "receiver_memory_update_count", "action_changed_by_signal_count", "exploration_fraction", "vertical_fraction", "unknown_entry_fraction", "damage_total", "recovery_total", "viability_mean", "role_score_sender", "role_score_receiver", "role_score_explorer"]
SEQUENCE_CONTEXT_FIELDS = ["scenario", "task", "arm", "communication_condition", "ablation_condition", "seed_id", "population_size", "step", "time_bin", "receiver_id", "prev_channel_2", "prev_channel_1", "current_channel", "received_channels", "time_since_previous_signal", "same_sender_as_previous", "current_sender_id", "receiver_action", "receiver_action_class", "pressure_danger", "pressure_resource", "pressure_unknown", "pressure_vertical", "pressure_friction"]
SAFETY_OVERRIDE_FIELDS = ["scenario", "task", "arm", "communication_condition", "ablation_condition", "seed_id", "population_size", "step", "time_bin", "agent_id", "proposed_action", "base_action_after_physics", "signal_biased_action", "final_action", "phys_adjusted", "phys_adjustment_reason", "signal_changed_action", "predicted_wall_best_signal", "predicted_damage_best_signal", "viability", "was_signal_effect_blocked", "signal_bias_reason"]

# -----------------------------------------------------------------------------
# Run management
# -----------------------------------------------------------------------------

def run_key(scenario: str, task: str, arm: str, condition: str, seed_id: int, n: int) -> str:
    return f"{scenario}__{task}__{arm}__{condition}__seed{seed_id}__N{n}"


def load_done(outdir: Path) -> set:
    idx = read_csv_if_exists(outdir / "run_index.csv")
    if idx.empty or "run_key" not in idx.columns:
        return set()
    ok = idx[idx.get("status", "") == "ok"] if "status" in idx.columns else idx
    return set(ok["run_key"].astype(str).tolist())


def run_all(B: Any, v11: Any, autonomous_core_module: Any, scenarios: List[Any], args: argparse.Namespace, logger: Logger) -> None:
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    conditions = [x.strip() for x in args.ablation_conditions.split(",") if x.strip()]
    pops = [int(x.strip()) for x in args.population_sizes.split(",") if x.strip()]
    for c in conditions:
        if c not in ABLATION_CONDITIONS:
            raise ValueError(f"Invalid ablation condition: {c}. Valid: {ABLATION_CONDITIONS}")
    done = load_done(args.outdir) if args.resume else set()
    total = len(scenarios) * len(tasks) * len(arms) * len(conditions) * int(args.seeds) * len(pops)
    requested_agent_steps = len(scenarios) * len(tasks) * len(arms) * len(conditions) * int(args.seeds) * int(args.steps) * sum(pops)
    logger.log(f"planned episodes={total:,}; requested agent-steps before early termination={requested_agent_steps:,}")
    completed = skipped = 0
    for sc in scenarios:
        logger.log(f"[scenario] {sc.scenario} factor={sc.gradient_factor} level={sc.gradient_level}")
        for task in tasks:
            for arm in arms:
                for condition in conditions:
                    for seed_id in range(int(args.seeds)):
                        for n in pops:
                            rk = run_key(sc.scenario, task, arm, condition, seed_id, n)
                            if rk in done:
                                skipped += 1
                                continue
                            t0 = time.time()
                            meta = {"run_key": rk, "status": "ok", "error": "", "created_at": now(), "scenario": sc.scenario, "task": task, "arm": arm, "communication_condition": condition, "seed_id": seed_id, "population_size": n, "density_phi": args.density_phi}
                            try:
                                per_agent, pop_row, logs = run_ablation_episode(B, v11, autonomous_core_module, sc, task, arm, condition, seed_id, n, args, logger)
                                append_rows_csv(args.outdir / "language_agent_episode_summary.csv", per_agent, B.PER_AGENT_FIELDS + ["time_bin"] if hasattr(B, "PER_AGENT_FIELDS") else sorted(set().union(*(r.keys() for r in per_agent))))
                                append_rows_csv(args.outdir / "language_population_episode_summary.csv", [pop_row], B.POP_FIELDS + ["action_changed_by_signal_count"] if hasattr(B, "POP_FIELDS") else sorted(pop_row.keys()))
                                append_rows_csv(args.outdir / "signal_event_log.csv", logs["signal_event_rows"], B.SIGNAL_EVENT_FIELDS + EXTRA_SIGNAL_EVENT_FIELDS)
                                append_rows_csv(args.outdir / "signal_delivery_log.csv", logs["signal_delivery_rows"], B.SIGNAL_DELIVERY_FIELDS + ["packet_id", "emitted_channel", "delivered_channel", "was_channel_shuffled", "time_bin"])
                                append_rows_csv(args.outdir / "receiver_signal_action_log.csv", logs["receiver_action_rows"], B.RECEIVER_ACTION_FIELDS + ["packet_id", "emitted_channel", "delivered_channel", "was_channel_shuffled", "time_bin"])
                                append_rows_csv(args.outdir / "state_perturbation_reflex_assay.csv", logs["state_perturbation_rows"], B.STATE_PERTURBATION_FIELDS)
                                append_rows_csv(args.outdir / "receiver_signal_memory_snapshot.csv", logs["memory_rows"], B.MEMORY_FIELDS)
                                append_rows_csv(args.outdir / "sender_channel_preference_snapshot.csv", logs["preference_rows"], B.PREF_FIELDS)
                                append_rows_csv(args.outdir / "utterance_sequence_log.csv", logs["utterance_rows"], B.UTTERANCE_FIELDS)
                                append_rows_csv(args.outdir / "01_action_decision_trace.csv", logs["action_decision_rows"], ACTION_DECISION_FIELDS)
                                append_rows_csv(args.outdir / "02_no_signal_counterfactual_trace.csv", logs["no_signal_counterfactual_rows"], NO_SIGNAL_COUNTERFACTUAL_FIELDS)
                                append_rows_csv(args.outdir / "03_receiver_memory_update_trace.csv", logs["receiver_memory_update_rows"], RECEIVER_MEMORY_UPDATE_FIELDS)
                                append_rows_csv(args.outdir / "04_signal_packet_lifecycle.csv", logs["signal_packet_lifecycle_rows"], SIGNAL_PACKET_LIFECYCLE_FIELDS)
                                append_rows_csv(args.outdir / "05_sender_preference_update_trace.csv", logs["sender_preference_update_rows"], SENDER_PREFERENCE_UPDATE_FIELDS)
                                append_rows_csv(args.outdir / "06_local_environment_exposure_by_agent_timebin.csv", logs["local_environment_exposure_rows"], LOCAL_ENV_EXPOSURE_FIELDS)
                                append_rows_csv(args.outdir / "07_communication_network_by_timebin.csv", logs["communication_network_rows"], COMM_NETWORK_FIELDS)
                                append_rows_csv(args.outdir / "08_agent_role_by_timebin.csv", logs["agent_role_rows"], AGENT_ROLE_FIELDS)
                                append_rows_csv(args.outdir / "09_sequence_context_trace.csv", logs["sequence_context_rows"], SEQUENCE_CONTEXT_FIELDS)
                                append_rows_csv(args.outdir / "10_safety_override_trace.csv", logs["safety_override_rows"], SAFETY_OVERRIDE_FIELDS)
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

# -----------------------------------------------------------------------------
# Posthoc ablation summaries
# -----------------------------------------------------------------------------

def summarize_action_decisions(decision_df: pd.DataFrame) -> pd.DataFrame:
    if decision_df.empty:
        return pd.DataFrame()
    df = decision_df.copy()
    for c in ["action_changed_by_signal", "delta_score_due_to_signal", "signal_bias_value", "n_received_signals"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    rows = []
    keys = ["scenario", "task", "arm", "communication_condition", "seed_id", "population_size", "time_bin"]
    for key, sub in df.groupby(keys, dropna=False):
        row = dict(zip(keys, key))
        row.update({
            "n_decision_signal_contexts": int(sub.shape[0]),
            "action_changed_by_signal_fraction": float(sub["action_changed_by_signal"].mean()) if "action_changed_by_signal" in sub else 0.0,
            "mean_delta_score_due_to_signal": float(sub["delta_score_due_to_signal"].mean()) if "delta_score_due_to_signal" in sub else 0.0,
            "mean_abs_signal_bias_value": float(sub["signal_bias_value"].abs().mean()) if "signal_bias_value" in sub else 0.0,
            "mean_received_signal_count": float(sub["n_received_signals"].mean()) if "n_received_signals" in sub else 0.0,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_memory_updates(mem_df: pd.DataFrame) -> pd.DataFrame:
    if mem_df.empty:
        return pd.DataFrame()
    df = mem_df.copy()
    for c in ["M_before", "M_after", "channel_value_before", "channel_value_after", "embodied_outcome_y", "memory_update_enabled"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["abs_delta_M"] = (df.get("M_after", 0) - df.get("M_before", 0)).abs()
    rows = []
    keys = ["scenario", "task", "arm", "communication_condition", "seed_id", "population_size", "time_bin"]
    for key, sub in df.groupby(keys, dropna=False):
        row = dict(zip(keys, key))
        row.update({
            "n_memory_trace_rows": int(sub.shape[0]),
            "n_memory_updates_enabled_rows": int((sub.get("memory_update_enabled", 0) == 1).sum()),
            "mean_abs_delta_M": float(sub["abs_delta_M"].mean()),
            "mean_embodied_outcome_y": float(sub.get("embodied_outcome_y", pd.Series(dtype=float)).mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_sequence_context(seq_df: pd.DataFrame) -> pd.DataFrame:
    if seq_df.empty:
        return pd.DataFrame()
    rows = []
    keys = ["scenario", "task", "arm", "communication_condition", "seed_id", "population_size", "time_bin"]
    df = seq_df.copy()
    df["prev_current"] = df.get("prev_channel_1", -1).astype(str) + "|" + df.get("current_channel", -1).astype(str)
    for key, sub in df.groupby(keys, dropna=False):
        row = dict(zip(keys, key))
        row.update({
            "n_sequence_contexts": int(sub.shape[0]),
            "mi_prev_current_to_receiver_action_class": mutual_information_discrete(sub["prev_current"].tolist(), sub.get("receiver_action_class", pd.Series(dtype=str)).astype(str).tolist()),
            "mi_current_to_receiver_action_class": mutual_information_discrete(sub.get("current_channel", pd.Series(dtype=str)).astype(str).tolist(), sub.get("receiver_action_class", pd.Series(dtype=str)).astype(str).tolist()),
        })
        row["incremental_sequence_action_mi"] = row["mi_prev_current_to_receiver_action_class"] - row["mi_current_to_receiver_action_class"]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_channel_action_mi(receiver_df: pd.DataFrame) -> pd.DataFrame:
    if receiver_df.empty:
        return pd.DataFrame()
    rows = []
    df = receiver_df.copy()
    if "time_bin" not in df.columns:
        step_col = "delivery_step" if "delivery_step" in df.columns else "step"
        df["time_bin"] = df[step_col].map(time_bin_label) if step_col in df.columns else "unknown"
    keys = ["scenario", "task", "arm", "communication_condition", "seed_id", "population_size", "time_bin"]
    for key, sub in df.groupby(keys, dropna=False):
        row = dict(zip(keys, key))
        row.update({
            "n_receiver_action_rows": int(sub.shape[0]),
            "mi_channel_to_receiver_action_class": mutual_information_discrete(sub.get("channel", pd.Series(dtype=str)).astype(str).tolist(), sub.get("receiver_action_class", pd.Series(dtype=str)).astype(str).tolist()),
            "mean_receiver_embodied_value": float(pd.to_numeric(sub.get("receiver_embodied_value", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def make_primary_causal_contrasts(outdir: Path, logger: Logger) -> None:
    logger.log("[posthoc] summarizing ablation traces")
    decision = summarize_action_decisions(read_csv_if_exists(outdir / "01_action_decision_trace.csv"))
    memory = summarize_memory_updates(read_csv_if_exists(outdir / "03_receiver_memory_update_trace.csv"))
    seq = summarize_sequence_context(read_csv_if_exists(outdir / "09_sequence_context_trace.csv"))
    recv = summarize_channel_action_mi(read_csv_if_exists(outdir / "receiver_signal_action_log.csv"))

    for name, df in [("decision", decision), ("memory", memory), ("sequence", seq), ("receiver", recv)]:
        if not df.empty:
            df.to_csv(outdir / f"11_{name}_ablation_summary_by_seed_timebin.csv", index=False)

    key_cols = ["scenario", "task", "arm", "seed_id", "population_size", "time_bin"]
    metric_frames = []
    def melt_metric(df: pd.DataFrame, metric_cols: Sequence[str], source: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        rows = []
        for _, r in df.iterrows():
            for m in metric_cols:
                if m in df.columns:
                    rows.append({**{k: r.get(k) for k in key_cols}, "communication_condition": r.get("communication_condition"), "metric": m, "value": safe_float(r.get(m)), "source": source})
        return pd.DataFrame(rows)
    metric_frames.append(melt_metric(decision, ["action_changed_by_signal_fraction", "mean_delta_score_due_to_signal", "mean_abs_signal_bias_value"], "decision"))
    metric_frames.append(melt_metric(memory, ["mean_abs_delta_M", "mean_embodied_outcome_y"], "memory"))
    metric_frames.append(melt_metric(seq, ["incremental_sequence_action_mi", "mi_prev_current_to_receiver_action_class"], "sequence"))
    metric_frames.append(melt_metric(recv, ["mi_channel_to_receiver_action_class", "mean_receiver_embodied_value"], "receiver"))
    allm = pd.concat([x for x in metric_frames if not x.empty], ignore_index=True) if any(not x.empty for x in metric_frames) else pd.DataFrame()
    contrasts = []
    if not allm.empty:
        for keys, sub in allm.groupby(key_cols + ["metric"], dropna=False):
            base = sub[sub["communication_condition"] == "FULL_INTERACTIVE"]
            if base.empty:
                continue
            base_val = safe_float(base["value"].mean())
            for _, rr in sub[sub["communication_condition"] != "FULL_INTERACTIVE"].iterrows():
                contrasts.append({
                    **dict(zip(key_cols + ["metric"], keys)),
                    "ablation_condition": rr.get("communication_condition"),
                    "full_interactive_value": base_val,
                    "ablation_value": safe_float(rr.get("value")),
                    "full_minus_ablation": base_val - safe_float(rr.get("value")),
                    "source": rr.get("source"),
                })
    cdf = pd.DataFrame(contrasts)
    if not cdf.empty:
        cdf.to_csv(outdir / "12_ablation_primary_causal_contrasts.csv", index=False)

    decision_rows = []
    if not cdf.empty:
        for (abl, metric), sub in cdf.groupby(["ablation_condition", "metric"], dropna=False):
            vals = pd.to_numeric(sub["full_minus_ablation"], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            med = float(vals.median())
            mean = float(vals.mean())
            pos_frac = float((vals > 0).mean())
            if abl == "NO_SIGNAL_BIAS" and metric in ("action_changed_by_signal_fraction", "mean_delta_score_due_to_signal", "mi_channel_to_receiver_action_class", "incremental_sequence_action_mi"):
                interp = "direct receiver signal-bias contribution" if med > 0 else "weak or absent direct signal-bias contribution"
            elif abl == "NO_RECEIVER_MEMORY_UPDATE" and metric in ("mean_abs_delta_M", "mi_channel_to_receiver_action_class", "incremental_sequence_action_mi"):
                interp = "receiver memory contribution" if med > 0 else "weak or absent receiver memory contribution"
            elif abl == "ONLINE_CHANNEL_SHUFFLED" and metric in ("mi_channel_to_receiver_action_class", "incremental_sequence_action_mi"):
                interp = "channel identity contribution" if med > 0 else "weak or absent channel identity contribution"
            elif abl == "NO_SENDER_PREFERENCE":
                interp = "sender preference contribution" if med > 0 else "weak or absent sender preference contribution"
            else:
                interp = "descriptive ablation contrast"
            decision_rows.append({
                "ablation_condition": abl,
                "metric": metric,
                "n_matched_cells": int(len(vals)),
                "mean_full_minus_ablation": mean,
                "median_full_minus_ablation": med,
                "positive_fraction": pos_frac,
                "interpretation": interp,
            })
    write_csv(outdir / "13_ablation_mechanism_decision_table.csv", decision_rows)


def write_readme(outdir: Path, args: argparse.Namespace, selected_scenarios: Sequence[Any]) -> None:
    lines = []
    lines.append("# Signal-pathway ablation causal validation")
    lines.append("")
    lines.append("First-look files:")
    lines.append("")
    lines.append("- `run_index.csv`: episode-level execution status")
    lines.append("- `01_action_decision_trace.csv`: signal contexts and immediate action-selection effects")
    lines.append("- `02_no_signal_counterfactual_trace.csv`: same-state no-signal action counterfactuals")
    lines.append("- `03_receiver_memory_update_trace.csv`: receiver memory update events")
    lines.append("- `04_signal_packet_lifecycle.csv`: emitted-to-delivered channel lifecycle")
    lines.append("- `12_ablation_primary_causal_contrasts.csv`: FULL_INTERACTIVE minus ablation matched contrasts")
    lines.append("- `13_ablation_mechanism_decision_table.csv`: compact mechanism interpretation table")
    lines.append("")
    lines.append("Fixed design:")
    lines.append(f"- scenarios: {','.join([s.scenario for s in selected_scenarios])}")
    lines.append(f"- tasks: {args.tasks}")
    lines.append(f"- ablation conditions: {args.ablation_conditions}")
    lines.append(f"- seeds: {args.seeds}")
    lines.append(f"- steps: {args.steps}")
    lines.append(f"- population sizes: {args.population_sizes}")
    lines.append("")
    lines.append("Ablation semantics:")
    for c in [x.strip() for x in args.ablation_conditions.split(",") if x.strip()]:
        lines.append(f"- {c}: {causal_condition_description(c)}")
    lines.append("")
    lines.append("Interpretive note:")
    lines.append("These are live intervention simulations. They should be described as signal-pathway ablations, not as post hoc shuffles. The primary causal contrasts compare FULL_INTERACTIVE with each ablation under the same scenario, task, seed, population size, and time bin.")
    (outdir / "00_FIRST_LOOK_README.md").write_text("\n".join(lines), encoding="utf-8")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live signal-pathway ablation causal validation for embodied multi-agent signal organization.")
    p.add_argument("--base-runner-file", default="run_v11_population_language_emergence_v3.py", help="Path to the existing population runner; used as a library for locked helper functions.")
    p.add_argument("--v11-file", required=True, help="Path to darca_true_3d_integrated_task_battery_v11.py")
    p.add_argument("--darca-file", required=True, help="Path to the autonomous core implementation file")
    p.add_argument("--outdir", default="V11_SIGNAL_PATHWAY_ABLATION_CAUSAL_VALIDATION")
    p.add_argument("--scenario-subset", default=DEFAULT_SCENARIOS)
    p.add_argument("--tasks", default=DEFAULT_TASKS)
    p.add_argument("--arms", default=DEFAULT_ARM)
    p.add_argument("--ablation-conditions", default=DEFAULT_CONDITIONS)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--steps", type=int, default=16000)
    p.add_argument("--population-size", type=int, default=8)
    p.add_argument("--population-sizes", default="", help="Optional comma-separated N values. Overrides --population-size if provided.")
    p.add_argument("--density-phi", type=float, default=1.0)
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
    p.add_argument("--state-perturbation-assay", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--state-perturb-max-events-per-episode", type=int, default=2)
    p.add_argument("--trace-signal-context-only", action=argparse.BooleanOptionalAction, default=True, help="If true, action_decision_trace only logs decision steps with received signal contexts.")
    p.add_argument("--progress-every-runs", type=int, default=5)
    p.add_argument("--progress-every-steps", type=int, default=0)
    p.add_argument("--clean", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--posthoc-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir = Path(args.outdir).expanduser()
    if args.clean and args.outdir.exists() and not args.posthoc_only:
        shutil.rmtree(args.outdir)
    ensure_dir(args.outdir)
    logger = Logger(args.outdir)
    if not args.population_sizes.strip():
        args.population_sizes = str(int(args.population_size))

    base_runner = Path(args.base_runner_file).expanduser()
    if not base_runner.exists():
        # Common Downloads fallback for macOS workflow.
        fallback = Path.home() / "Downloads" / args.base_runner_file
        if fallback.exists():
            base_runner = fallback
    if not base_runner.exists():
        raise FileNotFoundError(f"Base runner not found: {args.base_runner_file}")

    v11_file = Path(args.v11_file).expanduser()
    core_file = Path(args.darca_file).expanduser()
    if not v11_file.exists():
        raise FileNotFoundError(v11_file)
    if not core_file.exists():
        raise FileNotFoundError(core_file)

    B = import_module_from_path("population_signal_base_runner", base_runner.resolve())
    v11 = B.import_module_from_path("locked_v11_for_ablation", v11_file.resolve())
    autonomous_core_module = v11.load_darca_module(str(core_file.resolve()))

    scenarios_all = B.built_in_scenarios()
    wanted = [x.strip() for x in args.scenario_subset.split(",") if x.strip()]
    scenarios = [s for s in scenarios_all if s.scenario in set(wanted)]
    missing = sorted(set(wanted) - {s.scenario for s in scenarios})
    if missing:
        raise ValueError(f"Unknown scenarios: {missing}")

    write_csv(args.outdir / "00_ablation_condition_manifest.csv", [
        {"communication_condition": c, "ablation_condition": c, "description": causal_condition_description(c)}
        for c in [x.strip() for x in args.ablation_conditions.split(",") if x.strip()]
    ])
    write_csv(args.outdir / "00_scenario_manifest.csv", [asdict(s) for s in scenarios])
    lock = {
        "created_at": now(),
        "purpose": "live signal-pathway ablation causal validation",
        "base_runner_file": str(base_runner.resolve()),
        "base_runner_sha256": sha256_file(base_runner.resolve()),
        "v11_file": str(v11_file.resolve()),
        "v11_sha256": sha256_file(v11_file.resolve()),
        "autonomous_core_file": str(core_file.resolve()),
        "autonomous_core_sha256": sha256_file(core_file.resolve()),
        "scenarios": [s.scenario for s in scenarios],
        "tasks": args.tasks,
        "ablation_conditions": args.ablation_conditions,
        "seeds": args.seeds,
        "steps": args.steps,
        "population_sizes": args.population_sizes,
        "trace_signal_context_only": args.trace_signal_context_only,
    }
    (args.outdir / "00_ABLATION_RUN_LOCK.json").write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(args.outdir, args, scenarios)

    logger.log(f"locked base_runner={lock['base_runner_sha256'][:16]}... v11={lock['v11_sha256'][:16]}... core={lock['autonomous_core_sha256'][:16]}...")
    logger.log(f"scenarios={len(scenarios)} tasks={args.tasks} conditions={args.ablation_conditions} seeds={args.seeds} steps={args.steps} N={args.population_sizes}")

    if not args.posthoc_only:
        run_all(B, v11, autonomous_core_module, scenarios, args, logger)
    make_primary_causal_contrasts(args.outdir, logger)
    logger.log(f"[complete] outputs written to {args.outdir}")


if __name__ == "__main__":
    main()
