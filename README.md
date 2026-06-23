This package contains the simulation and reanalysis code for:

Environment-Conditioned Organization of Anonymous Signal Channels in Embodied Artificial Agents

The code can reproduce the main population simulation, the Phase 1–7 reanalysis, the manuscript figures, and the live signal-pathway ablation analysis.

Files
darca_v24_direct_rewrite_source.py
Defines the individual autonomous agent, including internal regulation, embodied valuation, memory, and closed-loop state updates.
darca_true_3d_integrated_task_battery_v11.py
Defines the TRUE 3D environment, task contexts, environmental pressures, movement rules, and physical consequences.
run_v11_population_language_emergence_v3.py
Runs the main eight-agent population simulation across environmental scenarios, task contexts, communication conditions, seeds, and 16,000-step episodes.
run_16000_language_acquisition_simulation.sh
Shell launcher for the full 16,000-step population simulation. It creates output folders, records progress logs, and supports resumable execution.
reanalysis_population_signal_system_all_in_one.py
Reproduces the fixed Phase 1–7 analyses from the raw simulation logs and generates manuscript-ready CSV files and Figures 1–6.
run_v11_signal_pathway_ablation_causal_validation.py
Runs live signal-pathway ablation simulations. It tests NO_SIGNAL_BIAS, NO_RECEIVER_MEMORY_UPDATE, ONLINE_CHANNEL_SHUFFLED, and NO_SENDER_PREFERENCE against FULL_INTERACTIVE.
README_REPRODUCIBILITY.md
This file.
What the code does

The code tests whether anonymous non-semantic signal channels become organized in an embodied multi-agent population.

It can reproduce:

environmental channel differentiation
sender-side grounding association
receiver action-class relation
temporal development
population-level sharing
sequence dependence
null and matched-control analyses
live signal-pathway ablations
