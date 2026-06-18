#!/usr/bin/env bash
# V11/V3 16000-step LANGUAGE-ACQUISITION locked developmental runner
#
# Purpose
#   Run the same predefined V3 developmental design as the completed 8000-step run,
#   but with 16000 steps, specifically as a language-acquisition validation.
#   The primary object is not cross-population transfer. The primary object is whether
#   a population-internal signal system is acquired over time and comes to affect
#   sender state, receiver action, internal grounding, and sequence/order structure.
#
# Design locked here
#   scenarios: baseline_3d, unknown_x1p50, danger_x1p25, danger_x1p50, vertical_x1p50
#   tasks: exploration_recovery, physics_adaptation, social_reappraisal
#   communication conditions: PRIVATE, RANDOM_MATCHED, SHUFFLED, RECEIVER_LEARN, FULL_INTERACTIVE
#   seeds: 20 by default
#   population size: 8 by default
#   steps: 16000 by default
#
# Important safety choices based on previous failures
#   - Raw shard outputs are written to ~/V11_LANGUAGE_LOCAL, not Desktop/iCloud.
#   - Desktop receives only logs, symlinks, and readable summaries.
#   - No huge raw CSV merge is attempted.
#   - Each scenario is isolated into its own shard directory to avoid write conflicts.
#   - Resume is enabled by default.
#   - Preflight smoke run checks core logs. receiver_action_log is optional in short smoke because receiver responses may not occur in 240 steps.
#   - No hard free-space threshold is enforced unless MIN_FREE_GB or --min-free-gb is explicitly set.
#   - Completion audit checks expected episode counts per scenario.
#   - All log/shard directories are re-created immediately before each write/launch;
#     this prevents FileNotFoundError from missing parent directories during parallel runs.
#   - This script does not use language_like_level as an evidence gate.
#   - Cross-population L7/L7b transfer is not treated as a required criterion here;
#     later transfer/intervention assays are separate portability/causal-function tests.
#   - L6/grammar-like acquisition is explicitly logged via utterance_sequence_log,
#     order_dependence_summary, and ngram_embodied_success_summary.
#   - The script writes a language-acquisition audit so the run is evaluated as
#     language acquisition, not as a generic behavior simulation.
#
# Typical use
#   chmod +x ~/Downloads/run_v3_16000_language_acquisition_LOCKED_v6_DIR_GUARD.sh
#   python3 -m py_compile ~/Downloads/run_v11_population_language_emergence_v3.py
#   ~/Downloads/run_v3_16000_language_acquisition_LOCKED_v6_DIR_GUARD.sh 2>&1 | tee ~/Desktop/V11_LANGUAGE_DEV_16000_V3_LAUNCH.log
#
set -Eeuo pipefail

# -----------------------------
# Defaults; override by env vars or CLI flags.
# -----------------------------
RUNNER="${RUNNER:-$HOME/Downloads/run_v11_population_language_emergence_v3.py}"
V11_FILE="${V11_FILE:-$HOME/Downloads/darca_true_3d_integrated_task_battery_v11.py}"
DARCA_FILE="${DARCA_FILE:-$HOME/Downloads/darca_v24_direct_rewrite_source.py}"

LOCAL_OUTROOT="${LOCAL_OUTROOT:-$HOME/V11_LANGUAGE_LOCAL/V11_LANGUAGE_DEV_16000_LANGUAGE_ACQUISITION_RAW}"
DESKTOP_OUTROOT="${DESKTOP_OUTROOT:-$HOME/Desktop/V11_LANGUAGE_DEV_16000_LANGUAGE_ACQUISITION}"

STEPS="${STEPS:-16000}"
SEEDS="${SEEDS:-20}"
POPULATION_SIZE="${POPULATION_SIZE:-8}"
DENSITY_PHI="${DENSITY_PHI:-1.0}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
MIN_FREE_GB="${MIN_FREE_GB:-0}"
PROGRESS_EVERY_RUNS="${PROGRESS_EVERY_RUNS:-10}"
PROGRESS_EVERY_STEPS="${PROGRESS_EVERY_STEPS:-1000}"

RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
PREFLIGHT_STEPS="${PREFLIGHT_STEPS:-240}"
PREFLIGHT_SEEDS="${PREFLIGHT_SEEDS:-1}"
NGRAM_MAX_N="${NGRAM_MAX_N:-3}"
NGRAM_OUTCOME_WINDOW="${NGRAM_OUTCOME_WINDOW:-10}"
STATE_PERTURB_MAX_EVENTS_PER_EPISODE="${STATE_PERTURB_MAX_EVENTS_PER_EPISODE:-1}"

# Phases: all, preflight, run, summary, audit, reanalysis, estimate
PHASE="${PHASE:-all}"
RESUME="${RESUME:-1}"
NO_FIGURES="${NO_FIGURES:-1}"

# Reanalysis is optional and source-only here. Cross-population transfer/intervention assays
# are intentionally kept separate from the developmental source run.
# Set RUN_REANALYSIS_AFTER=1 or use --run-reanalysis only when you want an immediate
# 16000 source-only descriptive/causal reanalysis.
RUN_REANALYSIS_AFTER="${RUN_REANALYSIS_AFTER:-0}"
REANALYSIS_SCRIPT="${REANALYSIS_SCRIPT:-$HOME/Downloads/run_v3_8000_multilevel_causal_reanalysis_LOCKED_FULL_v2.py}"
ADDITIONAL_SCRIPT="${ADDITIONAL_SCRIPT:-$HOME/Downloads/run_v3_8000_additional_mechanism_analyses_LOCKED_v3.py}"
L7B_DIR="${L7B_DIR:-$HOME/V11_LANGUAGE_LOCAL/V11_LANGUAGE_DEV_16000_LANGUAGE_ACQUISITION_L7_PORTABILITY}"
TIME_BINS="${TIME_BINS:-0:1000,1000:2000,2000:4000,4000:8000,8000:16000}"
LAG_BINS="${LAG_BINS:-1,2,5,10,20,40}"
LAG_SLOT_WIDTH="${LAG_SLOT_WIDTH:-100}"

TASKS="${TASKS:-exploration_recovery,physics_adaptation,social_reappraisal}"
COMMS="${COMMS:-PRIVATE,RANDOM_MATCHED,SHUFFLED,RECEIVER_LEARN,FULL_INTERACTIVE}"
SCENARIO_CSV="${SCENARIO_CSV:-baseline_3d,unknown_x1p50,danger_x1p25,danger_x1p50,vertical_x1p50}"
IFS=',' read -r -a SCENARIOS <<< "$SCENARIO_CSV"

usage() {
  cat <<EOF
Usage: $0 [options]

This is a language-acquisition developmental run. Cross-population transfer is not a gate.

Options:
  --phase PHASE              all|preflight|run|summary|audit|reanalysis|estimate (default: $PHASE)
  --runner PATH              population runner path
  --v11-file PATH            DARCA TRUE 3D v11 file
  --darca-file PATH          DARCA core file
  --local-outroot PATH       heavy local output root
  --desktop-outroot PATH     lightweight Desktop output root
  --steps N                  default 16000
  --seeds N                  default 20
  --population-size N        default 8
  --max-parallel N           default 3
  --min-free-gb N            default 0; when 0, warn only and do not hard stop
  --no-preflight             skip preflight smoke run
  --run-reanalysis           run 16000 source reanalysis after summary/audit
  --no-resume                do not pass --resume to runner
  --ngram-max-n N            maximum n-gram length for sequence acquisition logs (default: 3)
  --ngram-outcome-window N   outcome window metadata for n-gram analysis (default: 10)
  --help                     show this help

Examples:
  $0 --phase estimate
  MAX_PARALLEL=2 $0
  $0 --phase summary
  $0 --phase reanalysis --run-reanalysis
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --runner) RUNNER="$2"; shift 2 ;;
    --v11-file) V11_FILE="$2"; shift 2 ;;
    --darca-file) DARCA_FILE="$2"; shift 2 ;;
    --local-outroot) LOCAL_OUTROOT="$2"; shift 2 ;;
    --desktop-outroot) DESKTOP_OUTROOT="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --population-size) POPULATION_SIZE="$2"; shift 2 ;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --min-free-gb) MIN_FREE_GB="$2"; shift 2 ;;
    --no-preflight) RUN_PREFLIGHT=0; shift ;;
    --run-reanalysis) RUN_REANALYSIS_AFTER=1; shift ;;
    --no-resume) RESUME=0; shift ;;
    --ngram-max-n) NGRAM_MAX_N="$2"; shift 2 ;;
    --ngram-outcome-window) NGRAM_OUTCOME_WINDOW="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "[fatal] unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$LOCAL_OUTROOT/logs" "$LOCAL_OUTROOT/shards" "$LOCAL_OUTROOT/code_snapshot" "$DESKTOP_OUTROOT"
LOG="$LOCAL_OUTROOT/logs/run_v3_16000_extension_LOCKED.log"
DESKTOP_LOG="$DESKTOP_OUTROOT/00_RUN_LOG.txt"

ensure_base_dirs() {
  # Re-create parent directories before every log write and scenario launch.
  # This is intentionally idempotent and safe under parallel scenario workers.
  mkdir -p "$LOCAL_OUTROOT/logs" "$LOCAL_OUTROOT/shards" "$LOCAL_OUTROOT/code_snapshot" "$DESKTOP_OUTROOT"
}

ensure_base_dirs
: >> "$LOG"
: >> "$DESKTOP_LOG"

log() {
  ensure_base_dirs
  local msg="$*"
  echo "$msg" | tee -a "$LOG" "$DESKTOP_LOG"
}

run_logged() {
  ensure_base_dirs
  log "[cmd] $*"
  "$@" 2>&1 | tee -a "$LOG" "$DESKTOP_LOG"
  local code=${PIPESTATUS[0]}
  if [ "$code" -ne 0 ]; then
    log "[error] command failed code=$code: $*"
    return "$code"
  fi
}

require_file() {
  local f="$1"
  if [ ! -f "$f" ]; then
    log "[fatal] required file not found: $f"
    exit 2
  fi
}

compile_python() {
  local f="$1"
  require_file "$f"
  log "[check] py_compile: $f"
  python3 -m py_compile "$f" 2>&1 | tee -a "$LOG" "$DESKTOP_LOG"
  local code=${PIPESTATUS[0]}
  if [ "$code" -ne 0 ]; then
    log "[fatal] python compile failed: $f"
    exit "$code"
  fi
}

free_gb_for_path() {
  local p="$1"
  mkdir -p "$p"
  df -Pk "$p" | awk 'NR==2 {printf "%.2f", $4/1024/1024}'
}

assert_min_free_space() {
  local p="$1"
  local min_gb="$2"
  local free_gb
  free_gb=$(free_gb_for_path "$p")
  if python3 - "$min_gb" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) <= 0 else 1)
PY
  then
    log "[check] free space at $p: ${free_gb} GB; no hard threshold (MIN_FREE_GB=0)"
    log "[warn] disk-full risk remains; set MIN_FREE_GB or --min-free-gb if you want a hard stop"
    return 0
  fi
  log "[check] free space at $p: ${free_gb} GB; required >= ${min_gb} GB"
  python3 - "$free_gb" "$min_gb" <<'PY'
import sys
free_gb = float(sys.argv[1]); min_gb = float(sys.argv[2])
if free_gb < min_gb:
    raise SystemExit(1)
PY
  local code=$?
  if [ "$code" -ne 0 ]; then
    log "[fatal] insufficient free space at $p"
    exit 3
  fi
}

jobs_running() {
  jobs -pr | wc -l | tr -d ' '
}

wait_for_slot() {
  while true; do
    local n
    n=$(jobs_running)
    if [ "$n" -lt "$MAX_PARALLEL" ]; then
      break
    fi
    log "[wait] active_jobs=$n max_parallel=$MAX_PARALLEL"
    sleep 60
  done
}

caffeinate_prefix=()
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate_prefix=(caffeinate -dimsu)
fi

write_plan() {
  local plan="$LOCAL_OUTROOT/00_LOCKED_16000_PLAN.md"
  cat > "$plan" <<EOF
# Locked V3 16000-step Language-Acquisition Plan

created_at: $(date)

## Fixed design
- steps: $STEPS
- seeds: $SEEDS
- population_size: $POPULATION_SIZE
- density_phi: $DENSITY_PHI
- scenarios: ${SCENARIOS[*]}
- tasks: $TASKS
- communication_conditions: $COMMS
- time_bins_for_reanalysis: $TIME_BINS
- lag_slot_width: $LAG_SLOT_WIDTH
- lag_bins: $LAG_BINS
- ngram_max_n: $NGRAM_MAX_N
- ngram_outcome_window: $NGRAM_OUTCOME_WINDOW

## Guardrails
- All predefined scenario-task-communication-seed cells are run under the same settings.
- No condition is selected post hoc as primary.
- Raw data are kept local under $LOCAL_OUTROOT.
- Desktop output is a lightweight view under $DESKTOP_OUTROOT.
- No huge raw CSV merge is performed by this script.
- language_like_level is not used as a primary evidence gate.
- Cross-population L7/L7b transfer is not used as a required criterion for language-acquisition origin in this source run.
- Later transfer/intervention assays should be interpreted as portability or causal-function tests, not as the definition of language acquisition.

## Language-acquisition evidence blocks
- A. Signal production and use: signal_event_log.csv, signal_delivery_log.csv
- B. Sender grounding: signal_sender_grounding_summary.csv
- C. Receiver action effect: receiver_action_log.csv, receiver_action_effect_summary.csv
- D. Non-reflexive / state-dependent response: reflex_rejection_summary.csv, state_perturbation_reflex_summary.csv
- E. Internal grounding vs external label: internal_grounding_vs_external_label_summary.csv
- F. Sequence / grammar-like precursor: utterance_sequence_log.csv, order_dependence_summary.csv, ngram_embodied_success_summary.csv

## Interpretation guardrail
- Grammar-like acquisition is not inferred from channel frequency alone.
- Sequence/order/n-gram evidence must be interpreted against order-shuffle/unigram controls in later analysis.
- Cross-population transfer failure does not negate population-internal language-like acquisition.

## Expected run count
- scenarios: ${#SCENARIOS[@]}
- tasks: $(python3 - <<PY
print(len('$TASKS'.split(',')))
PY
)
- communication conditions: $(python3 - <<PY
print(len('$COMMS'.split(',')))
PY
)
- seeds: $SEEDS
- expected total episodes: $(python3 - <<PY
print(${#SCENARIOS[@]} * len('$TASKS'.split(',')) * len('$COMMS'.split(',')) * int('$SEEDS'))
PY
)
EOF
  cp "$plan" "$DESKTOP_OUTROOT/00_LOCKED_16000_PLAN.md"
}

prepare() {
  ensure_base_dirs
  : >> "$LOG"
  : >> "$DESKTOP_LOG"
  log "[info] started: $(date)"
  log "[info] PHASE=$PHASE"
  log "[info] RUNNER=$RUNNER"
  log "[info] V11_FILE=$V11_FILE"
  log "[info] DARCA_FILE=$DARCA_FILE"
  log "[info] LOCAL_OUTROOT=$LOCAL_OUTROOT"
  log "[info] DESKTOP_OUTROOT=$DESKTOP_OUTROOT"
  log "[info] STEPS=$STEPS SEEDS=$SEEDS POPULATION_SIZE=$POPULATION_SIZE"
  log "[info] SCENARIOS=${SCENARIOS[*]}"
  log "[info] TASKS=$TASKS"
  log "[info] COMMS=$COMMS"
  log "[info] MAX_PARALLEL=$MAX_PARALLEL"
  require_file "$RUNNER"
  require_file "$V11_FILE"
  require_file "$DARCA_FILE"
  compile_python "$RUNNER"
  if [ -f "$REANALYSIS_SCRIPT" ]; then
    compile_python "$REANALYSIS_SCRIPT"
  else
    log "[warn] reanalysis script not found yet: $REANALYSIS_SCRIPT"
  fi
  if [ -f "$ADDITIONAL_SCRIPT" ]; then
    compile_python "$ADDITIONAL_SCRIPT"
  else
    log "[warn] additional mechanism script not found yet: $ADDITIONAL_SCRIPT"
  fi
  assert_min_free_space "$LOCAL_OUTROOT" "$MIN_FREE_GB"
  cp "$RUNNER" "$LOCAL_OUTROOT/code_snapshot/runner_$(basename "$RUNNER")" || true
  [ -f "$REANALYSIS_SCRIPT" ] && cp "$REANALYSIS_SCRIPT" "$LOCAL_OUTROOT/code_snapshot/reanalysis_$(basename "$REANALYSIS_SCRIPT")" || true
  [ -f "$ADDITIONAL_SCRIPT" ] && cp "$ADDITIONAL_SCRIPT" "$LOCAL_OUTROOT/code_snapshot/additional_$(basename "$ADDITIONAL_SCRIPT")" || true
  ln -sfn "$LOCAL_OUTROOT" "$DESKTOP_OUTROOT/LOCAL_RAW_OUTPUT_LINK"
  write_plan
}

run_runner_selftest() {
  local td="$LOCAL_OUTROOT/PREFLIGHT_SELF_TEST"
  rm -rf "$td"; mkdir -p "$td"
  log "[preflight] runner --self-test"
  run_logged python3 -u "$RUNNER" --self-test --outdir "$td"
}

run_preflight_smoke() {
  local pd="$LOCAL_OUTROOT/PREFLIGHT_SMOKE_REAL_FILES"
  rm -rf "$pd"; mkdir -p "$pd"
  log "[preflight] real-file smoke scenario=baseline_3d task=exploration_recovery comm=FULL_INTERACTIVE seeds=$PREFLIGHT_SEEDS steps=$PREFLIGHT_STEPS"
  run_logged "${caffeinate_prefix[@]}" python3 -u "$RUNNER" \
    --v11-file "$V11_FILE" \
    --darca-file "$DARCA_FILE" \
    --outdir "$pd" \
    --plan main_language \
    --scenario-subset baseline_3d \
    --tasks exploration_recovery \
    --communication-conditions FULL_INTERACTIVE \
    --population-size "$POPULATION_SIZE" \
    --density-phi "$DENSITY_PHI" \
    --seeds "$PREFLIGHT_SEEDS" \
    --steps "$PREFLIGHT_STEPS" \
    --state-perturbation-assay \
    --state-perturb-max-events-per-episode "$STATE_PERTURB_MAX_EVENTS_PER_EPISODE" \
    --ngram-max-n "$NGRAM_MAX_N" \
    --ngram-outcome-window "$NGRAM_OUTCOME_WINDOW" \
    --progress-every-runs 1 \
    --progress-every-steps 60 \
    --overwrite

  # In a very short smoke test, receiver_action_log.csv may legitimately be absent/empty
  # if no receiver-side action transition occurs after a delivered signal. Do not hard-fail
  # on that file here. The full 16000 completion audit still checks real shard outputs.
  local required=(
    "language_population_episode_summary.csv"
    "language_agent_episode_summary.csv"
    "run_index.csv"
    "signal_event_log.csv"
    "signal_delivery_log.csv"
    "utterance_sequence_log.csv"
  )
  local optional=(
    "receiver_action_log.csv"
  )
  local missing=0
  for f in "${required[@]}"; do
    if [ ! -s "$pd/$f" ]; then
      log "[fatal] preflight required file missing or empty: $pd/$f"
      missing=1
    else
      local lines
      lines=$(wc -l < "$pd/$f" | tr -d ' ')
      log "[preflight-ok] $f lines=$lines"
    fi
  done
  for f in "${optional[@]}"; do
    if [ ! -s "$pd/$f" ]; then
      log "[preflight-warn] optional short-smoke file missing or empty: $pd/$f"
      log "[preflight-warn] this does not abort; full 16000 shards are audited after the run"
    else
      local lines
      lines=$(wc -l < "$pd/$f" | tr -d ' ')
      log "[preflight-ok] optional $f lines=$lines"
    fi
  done
  if [ "$missing" -ne 0 ]; then
    log "[fatal] preflight failed; aborting before 16000 run"
    exit 4
  fi
}

run_preflight() {
  prepare
  run_runner_selftest
  run_preflight_smoke
  log "[preflight] complete: $(date)"
}

run_one_scenario() {
  ensure_base_dirs
  local sc="$1"
  local shard_dir="$LOCAL_OUTROOT/shards/$sc"
  local shard_log="$LOCAL_OUTROOT/logs/${sc}.log"
  mkdir -p "$LOCAL_OUTROOT/logs" "$LOCAL_OUTROOT/shards" "$shard_dir"
  : >> "$shard_log"
  if [ ! -d "$shard_dir" ]; then
    log "[fatal] failed to create scenario shard directory: $shard_dir"
    return 11
  fi
  if [ ! -d "$LOCAL_OUTROOT/logs" ]; then
    log "[fatal] failed to create logs directory: $LOCAL_OUTROOT/logs"
    return 12
  fi
  log "[launch] scenario=$sc outdir=$shard_dir time=$(date)"

  local resume_flag=()
  if [ "$RESUME" = "1" ]; then
    resume_flag=(--resume)
  fi

  "${caffeinate_prefix[@]}" python3 -u "$RUNNER" \
    --v11-file "$V11_FILE" \
    --darca-file "$DARCA_FILE" \
    --outdir "$shard_dir" \
    --plan main_language \
    --scenario-subset "$sc" \
    --tasks "$TASKS" \
    --communication-conditions "$COMMS" \
    --population-size "$POPULATION_SIZE" \
    --density-phi "$DENSITY_PHI" \
    --seeds "$SEEDS" \
    --steps "$STEPS" \
    --state-perturbation-assay \
    --state-perturb-max-events-per-episode "$STATE_PERTURB_MAX_EVENTS_PER_EPISODE" \
    --ngram-max-n "$NGRAM_MAX_N" \
    --ngram-outcome-window "$NGRAM_OUTCOME_WINDOW" \
    --progress-every-runs "$PROGRESS_EVERY_RUNS" \
    --progress-every-steps "$PROGRESS_EVERY_STEPS" \
    "${resume_flag[@]}" \
    2>&1 | tee -a "$shard_log"
  local code=${PIPESTATUS[0]}
  if [ "$code" -ne 0 ]; then
    log "[error] scenario=$sc exited code=$code time=$(date)"
    return "$code"
  fi
  log "[done] scenario=$sc time=$(date)"
}

run_all_scenarios() {
  ensure_base_dirs
  prepare
  if [ "$RUN_PREFLIGHT" = "1" ]; then
    run_runner_selftest
    run_preflight_smoke
  else
    log "[warn] preflight skipped by user setting"
  fi

  local pids=()
  for sc in "${SCENARIOS[@]}"; do
    wait_for_slot
    run_one_scenario "$sc" &
    pids+=("$!")
    sleep 3
  done

  local fail=0
  for p in "${pids[@]}"; do
    if ! wait "$p"; then
      fail=1
    fi
  done
  if [ "$fail" -ne 0 ]; then
    log "[complete_with_errors] one or more scenario shards failed: $(date)"
    exit 1
  fi
  log "[run] all scenario shards finished: $(date)"
}

summary_collect_no_raw_merge() {
  ensure_base_dirs
  log "[summary] collecting no-raw-merge summaries"
  python3 - "$LOCAL_OUTROOT" "$DESKTOP_OUTROOT" "$TASKS" "$COMMS" "$SEEDS" <<'PY'
from __future__ import annotations
import os, sys, shutil
from pathlib import Path
import pandas as pd

local = Path(sys.argv[1]).expanduser().resolve()
desktop = Path(sys.argv[2]).expanduser().resolve()
tasks = sys.argv[3].split(',')
comms = sys.argv[4].split(',')
seeds = int(sys.argv[5])
shards_root = local / 'shards'
out = local / 'NO_RAW_MERGE_SUMMARY'
out.mkdir(parents=True, exist_ok=True)

summary_files = [
    'scenario_manifest.csv',
    'run_index.csv',
    'language_population_episode_summary.csv',
    'language_agent_episode_summary.csv',
    'primary_language_criteria_summary.csv',
    'language_like_level_summary.csv',
    'signal_sender_grounding_summary.csv',
    'receiver_action_effect_summary.csv',
    'reflex_rejection_summary.csv',
    'state_perturbation_reflex_summary.csv',
    'hidden_reference_summary.csv',
    'internal_grounding_vs_external_label_summary.csv',
    'order_dependence_summary.csv',
    'ngram_embodied_success_summary.csv',
    'functional_code_alignment_summary.csv',
    'signal_counterfactual_replay_summary.csv',
]
text_files = ['population_language_emergence_report.txt']
raw_required = ['signal_event_log.csv','signal_delivery_log.csv','receiver_action_log.csv','utterance_sequence_log.csv']

shards = [p for p in sorted(shards_root.iterdir()) if p.is_dir()] if shards_root.exists() else []
audit_rows = []
expected_per_shard = len(tasks) * len(comms) * seeds

for sd in shards:
    row = {'scenario_shard': sd.name, 'path': str(sd), 'expected_episode_rows': expected_per_shard}
    for rf in raw_required:
        p = sd / rf
        row[f'{rf}_exists'] = p.exists()
        row[f'{rf}_bytes'] = p.stat().st_size if p.exists() else 0
        row[f'{rf}_lines'] = sum(1 for _ in open(p, 'rb')) if p.exists() and p.stat().st_size > 0 else 0
    pop = sd / 'language_population_episode_summary.csv'
    if pop.exists() and pop.stat().st_size > 0:
        try:
            df = pd.read_csv(pop, low_memory=False)
            row['episode_rows'] = len(df)
            row['episode_rows_ok'] = (len(df) == expected_per_shard)
        except Exception as e:
            row['episode_rows'] = None
            row['episode_rows_ok'] = False
            row['episode_read_error'] = repr(e)
    else:
        row['episode_rows'] = 0
        row['episode_rows_ok'] = False
    audit_rows.append(row)

for name in summary_files:
    frames = []
    for sd in shards:
        p = sd / name
        if not p.exists() or p.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(p, low_memory=False)
            df.insert(0, 'scenario_shard', sd.name)
            df.insert(1, 'source_file', name)
            frames.append(df)
        except Exception as e:
            frames.append(pd.DataFrame([{'scenario_shard': sd.name, 'source_file': name, 'read_error': repr(e)}]))
    if frames:
        pd.concat(frames, ignore_index=True, sort=False).to_csv(out / name, index=False)

for name in text_files:
    dest = out / name
    with dest.open('w', encoding='utf-8') as w:
        for sd in shards:
            p = sd / name
            if p.exists() and p.stat().st_size > 0:
                w.write('\n' + '='*88 + '\n')
                w.write(f'SCENARIO_SHARD: {sd.name}\nSOURCE: {p}\n')
                w.write('='*88 + '\n\n')
                w.write(p.read_text(encoding='utf-8', errors='replace'))
                w.write('\n')

adf = pd.DataFrame(audit_rows)
adf.to_csv(out / '00_COMPLETION_AUDIT.csv', index=False)

n_shards = len(shards)
total_expected = expected_per_shard * n_shards
total_observed = int(adf.get('episode_rows', pd.Series(dtype=float)).fillna(0).sum()) if not adf.empty else 0
raw_ok = True
if not adf.empty:
    for rf in raw_required:
        raw_ok = raw_ok and bool((adf[f'{rf}_exists'] & (adf[f'{rf}_bytes'] > 0)).all())
    episodes_ok = bool(adf['episode_rows_ok'].all())
else:
    episodes_ok = False

report = out / '00_READ_ME_FIRST_16000_SUMMARY.md'
report.write_text(f'''# V3 16000 no-raw-merge summary

created_at: {pd.Timestamp.now()}

- shards_found: {n_shards}
- expected_episode_rows_per_shard: {expected_per_shard}
- total_expected_episode_rows: {total_expected}
- total_observed_episode_rows: {total_observed}
- episode_rows_all_ok: {episodes_ok}
- raw_required_logs_all_present_nonempty: {raw_ok}

Important files:
- 00_COMPLETION_AUDIT.csv
- language_population_episode_summary.csv
- language_agent_episode_summary.csv
- primary_language_criteria_summary.csv
- receiver_action_effect_summary.csv
- order_dependence_summary.csv
- ngram_embodied_success_summary.csv
- functional_code_alignment_summary.csv

No raw logs were merged here. Raw logs remain in local shard folders.
''', encoding='utf-8')

# Language-acquisition audit: schema/availability table, not a success claim.
acq_files = [
    'signal_event_log.csv',
    'signal_delivery_log.csv',
    'receiver_action_log.csv',
    'utterance_sequence_log.csv',
    'signal_sender_grounding_summary.csv',
    'receiver_action_effect_summary.csv',
    'reflex_rejection_summary.csv',
    'state_perturbation_reflex_summary.csv',
    'internal_grounding_vs_external_label_summary.csv',
    'order_dependence_summary.csv',
    'ngram_embodied_success_summary.csv',
    'functional_code_alignment_summary.csv',
]
acq_rows = []
for sd in shards:
    for name in acq_files:
        p = sd / name
        rec = {'scenario_shard': sd.name, 'file': name, 'exists': p.exists(), 'bytes': p.stat().st_size if p.exists() else 0}
        rec['rows'] = 0
        rec['columns'] = ''
        rec['read_error'] = ''
        if p.exists() and p.stat().st_size > 0:
            try:
                if name.endswith('.csv'):
                    # For raw logs, count lines cheaply but only read header.
                    rec['lines'] = sum(1 for _ in open(p, 'rb'))
                    try:
                        head = pd.read_csv(p, nrows=0)
                        rec['columns'] = '|'.join(map(str, head.columns.tolist()))
                    except Exception as e:
                        rec['read_error'] = repr(e)
                    if not name.endswith('_log.csv'):
                        dfh = pd.read_csv(p, low_memory=False)
                        rec['rows'] = len(dfh)
                    else:
                        rec['rows'] = max(0, rec.get('lines', 0)-1)
                else:
                    rec['lines'] = sum(1 for _ in open(p, 'rb'))
            except Exception as e:
                rec['read_error'] = repr(e)
        acq_rows.append(rec)
acq = pd.DataFrame(acq_rows)
acq.to_csv(out / '00_LANGUAGE_ACQUISITION_AUDIT.csv', index=False)

# Minimal grammar-readiness summary by scenario. This deliberately avoids declaring grammar success.
grammar_names = ['utterance_sequence_log.csv', 'order_dependence_summary.csv', 'ngram_embodied_success_summary.csv']
grows = []
for sd in shards:
    gr = {'scenario_shard': sd.name}
    for name in grammar_names:
        p = sd / name
        gr[name + '_exists'] = p.exists()
        gr[name + '_bytes'] = p.stat().st_size if p.exists() else 0
        gr[name + '_rows_or_logrows'] = 0
        if p.exists() and p.stat().st_size > 0:
            try:
                if name.endswith('_log.csv'):
                    gr[name + '_rows_or_logrows'] = max(0, sum(1 for _ in open(p, 'rb')) - 1)
                else:
                    gr[name + '_rows_or_logrows'] = len(pd.read_csv(p, low_memory=False))
            except Exception as e:
                gr[name + '_read_error'] = repr(e)
    gready = all(bool(gr.get(name + '_exists')) and int(gr.get(name + '_bytes', 0)) > 0 for name in grammar_names)
    gr['grammar_like_analysis_ready'] = gready
    grows.append(gr)
gram = pd.DataFrame(grows)
gram.to_csv(out / '00_GRAMMAR_LIKE_SEQUENCE_READINESS.csv', index=False)

(out / '00_LANGUAGE_ACQUISITION_READ_ME.md').write_text(f'''# V3 16000 language-acquisition summary

created_at: {pd.Timestamp.now()}

This folder is a no-raw-merge summary for the 16000-step developmental run.
The run is interpreted as a language-acquisition validation, not as a cross-population transfer gate.

Primary evidence blocks:
1. signal production and delivery
2. sender grounding
3. receiver action effect
4. non-reflexive state-dependent response
5. internal grounding
6. sequence/order/n-gram grammar-like precursor

Important audit files:
- 00_COMPLETION_AUDIT.csv
- 00_LANGUAGE_ACQUISITION_AUDIT.csv
- 00_GRAMMAR_LIKE_SEQUENCE_READINESS.csv

Guardrail:
Grammar-like acquisition must not be inferred from channel frequency alone.
It requires later sequence-order / n-gram analyses against unigram and order-shuffle controls, with seed-level replication and FDR control.
''', encoding='utf-8')

# Lightweight desktop view: replace only summary folder, not raw data.
desktop.mkdir(parents=True, exist_ok=True)
dsum = desktop / 'NO_RAW_MERGE_SUMMARY'
if dsum.exists() or dsum.is_symlink():
    if dsum.is_symlink() or dsum.is_file():
        dsum.unlink()
    else:
        shutil.rmtree(dsum)
try:
    dsum.symlink_to(out, target_is_directory=True)
except Exception:
    shutil.copytree(out, dsum)
print(f'[summary] wrote {out}')
print(f'[summary] desktop view {dsum}')
print(f'[summary] shards={n_shards} total_observed_episode_rows={total_observed} expected={total_expected}')
print(f'[summary] episode_rows_all_ok={episodes_ok} raw_required_logs_all_present_nonempty={raw_ok}')
PY
  local code=$?
  if [ "$code" -ne 0 ]; then
    log "[fatal] summary collector failed"
    exit "$code"
  fi
}

completion_audit_strict() {
  ensure_base_dirs
  log "[audit] strict completion audit"
  python3 - "$LOCAL_OUTROOT" "$SCENARIO_CSV" "$TASKS" "$COMMS" "$SEEDS" <<'PY'
from pathlib import Path
import sys, pandas as pd
root = Path(sys.argv[1]).expanduser().resolve()
scenarios = sys.argv[2].split(',')
tasks = sys.argv[3].split(',')
comms = sys.argv[4].split(',')
seeds = int(sys.argv[5])
expected = len(tasks) * len(comms) * seeds
summary = root / 'NO_RAW_MERGE_SUMMARY' / '00_COMPLETION_AUDIT.csv'
if not summary.exists():
    print('[fatal] completion audit missing:', summary)
    raise SystemExit(1)
df = pd.read_csv(summary)
fail = []
for sc in scenarios:
    sub = df[df['scenario_shard'] == sc]
    if sub.empty:
        fail.append(f'missing shard {sc}')
        continue
    r = sub.iloc[0]
    if int(r.get('episode_rows', -1)) != expected:
        fail.append(f'{sc}: episode_rows={r.get("episode_rows")} expected={expected}')
    for rf in ['signal_event_log.csv','signal_delivery_log.csv','receiver_action_log.csv','utterance_sequence_log.csv']:
        if not bool(r.get(f'{rf}_exists', False)) or int(r.get(f'{rf}_bytes', 0)) <= 0:
            fail.append(f'{sc}: missing/empty {rf}')
if fail:
    print('[audit-fail]')
    for x in fail:
        print(' -', x)
    raise SystemExit(1)
print('[audit-ok] all expected scenario shards, episode counts, and required raw logs are present')
PY
  local code=$?
  if [ "$code" -ne 0 ]; then
    log "[fatal] completion audit failed"
    exit "$code"
  fi
}

run_reanalysis_if_requested() {
  if [ "$RUN_REANALYSIS_AFTER" != "1" ] && [ "$PHASE" != "reanalysis" ]; then
    log "[info] reanalysis skipped. Use --run-reanalysis or --phase reanalysis after the 16000 run."
    return 0
  fi
  if [ ! -f "$REANALYSIS_SCRIPT" ]; then
    log "[fatal] reanalysis script not found: $REANALYSIS_SCRIPT"
    exit 2
  fi
  compile_python "$REANALYSIS_SCRIPT"
  mkdir -p "$L7B_DIR"
  local reout="$HOME/V11_LANGUAGE_LOCAL/V11_LANGUAGE_DEV_16000_REANALYSIS_FULL"
  local redesktop="$HOME/Desktop/V11_LANGUAGE_DEV_16000_REANALYSIS_FULL"
  local figflag=()
  if [ "$NO_FIGURES" = "1" ]; then
    figflag=(--no-figures)
  fi
  log "[reanalysis] running 16000 source reanalysis time_bins=$TIME_BINS"
  run_logged python3 -u "$REANALYSIS_SCRIPT" \
    --source-root "$LOCAL_OUTROOT" \
    --l7b-dir "$L7B_DIR" \
    --outdir "$reout" \
    --desktop-outdir "$redesktop" \
    --clean \
    "${figflag[@]}" \
    --time-bins "$TIME_BINS" \
    --lag-slot-width "$LAG_SLOT_WIDTH" \
    --lag-bins "$LAG_BINS"

  if [ -f "$ADDITIONAL_SCRIPT" ]; then
    compile_python "$ADDITIONAL_SCRIPT"
    local aout="$HOME/V11_LANGUAGE_LOCAL/V11_LANGUAGE_DEV_16000_ADDITIONAL_MECHANISMS"
    local adesktop="$HOME/Desktop/V11_LANGUAGE_DEV_16000_ADDITIONAL_MECHANISMS"
    log "[additional] running 16000 additional mechanisms"
    run_logged python3 -u "$ADDITIONAL_SCRIPT" \
      --source-root "$LOCAL_OUTROOT" \
      --reanalysis-dir "$reout" \
      --l7b-dir "$L7B_DIR" \
      --outdir "$aout" \
      --desktop-outdir "$adesktop" \
      --clean
  else
    log "[warn] additional mechanism script not found; skipped: $ADDITIONAL_SCRIPT"
  fi
}

estimate() {
  log "[info] started: $(date)"
  log "[info] PHASE=$PHASE"
  log "[info] language-acquisition developmental extension; no cross-population L7/L7b transfer gate"
  write_plan
  python3 - <<PY | tee -a "$LOG" "$DESKTOP_LOG"
scenarios = '$SCENARIO_CSV'.split(',')
tasks = '$TASKS'.split(',')
comms = '$COMMS'.split(',')
seeds = int('$SEEDS')
steps = int('$STEPS')
pop = int('$POPULATION_SIZE')
episodes = len(scenarios) * len(tasks) * len(comms) * seeds
agent_steps = episodes * steps * pop
print('[estimate] scenarios=', len(scenarios), scenarios)
print('[estimate] tasks=', len(tasks), tasks)
print('[estimate] comms=', len(comms), comms)
print('[estimate] seeds=', seeds)
print('[estimate] episodes=', episodes)
print('[estimate] steps_per_episode=', steps)
print('[estimate] population_size=', pop)
print('[estimate] agent_steps=', f'{agent_steps:,}')
print('[estimate] output_local=', '$LOCAL_OUTROOT')
print('[estimate] desktop_view=', '$DESKTOP_OUTROOT')
print('[estimate] ngram_max_n=', '$NGRAM_MAX_N')
print('[estimate] ngram_outcome_window=', '$NGRAM_OUTCOME_WINDOW')
print('[estimate] no_raw_merge=YES')
PY
}

case "$PHASE" in
  estimate)
    estimate
    ;;
  preflight)
    run_preflight
    ;;
  run)
    run_all_scenarios
    ;;
  summary)
    prepare
    summary_collect_no_raw_merge
    ;;
  audit)
    prepare
    summary_collect_no_raw_merge
    completion_audit_strict
    ;;
  reanalysis)
    prepare
    summary_collect_no_raw_merge
    completion_audit_strict
    RUN_REANALYSIS_AFTER=1
    run_reanalysis_if_requested
    ;;
  all)
    run_all_scenarios
    summary_collect_no_raw_merge
    completion_audit_strict
    run_reanalysis_if_requested
    ;;
  *)
    log "[fatal] unknown PHASE=$PHASE"
    usage
    exit 2
    ;;
esac

log "[complete] phase=$PHASE time=$(date)"
log "[complete] local output: $LOCAL_OUTROOT"
log "[complete] desktop view: $DESKTOP_OUTROOT"
