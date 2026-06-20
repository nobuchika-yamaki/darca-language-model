**Simulation and Reanalysis Codes for “Emergent Language-Like Communication in Embodied Multi-Agent Populations”**

1. `darca_v24_direct_rewrite_source.py`
   Core implementation of the individual autonomous agent. This file defines the agent-level regulatory variables, internal state updates, embodied valuation variables, agency-related evidence, memory dynamics, and basic closed-loop computation used by the simulation.

2. `darca_true_3d_integrated_task_battery_v11.py`
   TRUE 3D environment and integrated task framework. This file defines the three-dimensional grid world, environmental pressure encoding, movement rules, physical-risk consequences, task contexts, and the interface between the autonomous agent and the embodied environment.

3. `run_v11_population_language_emergence_v3.py`
   Main population-level simulation runner. This file executes the eight-agent population simulation across environmental scenarios, task contexts, communication conditions, seeds, and 16,000-step episodes. It generates the raw signal-event, signal-delivery, receiver-action, sequence, agent-summary, and population-summary logs used in the analyses.

4. `run_16000_language_acquisition_simulation.sh`
   Shell launcher for the full 16,000-step simulation. This script runs the population simulation in scenario-level shards, creates the required output directories, records progress logs, and supports resumable execution.

5. `reanalysis_population_signal_system_all_in_one.py`
   Integrated reanalysis script for the reported results. This file reads the raw simulation logs and reproduces the fixed Phase 1–7 analyses: environmental channel differentiation, sender-side grounding association, receiver-side action relation, temporal development, population-level sharing, sequence dependence, and null/matched-control evaluation. It also generates the manuscript-ready CSV summaries and Figures 1–6.

6. `README_REPRODUCIBILITY.md`
   Reproducibility guide. This file explains the required files, software environment, execution order, command-line examples, expected outputs, and interpretation of the main result files.

