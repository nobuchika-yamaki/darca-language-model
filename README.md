Simulation codes of "Emergent Language-Like Communication in Embodied Multi-Agent Populations”
1. `darca_v24_direct_rewrite_source.py`
   Core implementation of the individual autonomous agent. This file defines the agent-level regulatory variables, internal state update, valuation-related variables, and basic closed-loop computation used by the simulation.

2. `darca_true_3d_integrated_task_battery_v11.py`
   TRUE 3D environment and integrated task framework. This file defines the 3D grid world, environmental pressures, movement rules, physical-risk consequences, task contexts, and the interface between the agent and the embodied environment.

3. `run_v11_population_language_emergence_v3.py`
   Main population-level language-acquisition simulation runner. This file executes the eight-agent population simulation across scenario conditions, task contexts, communication conditions, seeds, and 16000-step episodes.

4. `run_16000_language_acquisition_simulation.sh`
   Shell launcher for the full 16000-step simulation. This script runs the population simulation in scenario-level shards, creates required output directories, records progress logs, and supports resumable execution.

5. `run_16000_language_likeness_reanalysis.py`
   Main reanalysis script for language-likeness judgement. This file reads the simulation logs and computes sender grounding, receiver-action effects, context dependence, internal grounding, sequence/order structure, functional relevance, within-population sharing, FDR-corrected tests, and final language-likeness categories.

6. `README_REPRODUCIBILITY.md`
   Reproducibility guide. This file explains the required files, software environment, execution order, command-line examples, expected outputs, and the interpretation of the main result files.
