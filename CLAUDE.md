# CLAUDE.md — Project Context Entrypoint

> This file is auto-loaded by Claude when the repository is opened. Use it as the
> first thing to read in any new session.

## What this project is

A Federated Learning (FL) simulation that parallelizes the client-training phase
using `ProcessPoolExecutor`. The course-report phase is complete (see
`README.md` for headline results: ~1.27x speedup, Amdahl-bound at parallel
fraction ~24%).

**This project is now an ongoing research effort** for Summer + Fall 2026,
extending the simulation toward HPC scale, asynchronous aggregation, and
realistic ML workloads (CIFAR-10 + ResNet).

## How to resume work (start here)

1. **Read `.notes/research_plan.md`** — the full context handoff document.
   It contains the research goal, agreed Future Work directions, open
   questions, decision log, and the next concrete actions.
2. **Skim `README.md`** — current results, methodology, and the public
   description of the project.
3. Ask the user what changed since the last session (e.g., advisor meeting,
   lab GPU access confirmation, Libra account approval) before suggesting next
   steps.

## Repo layout (high level)

- `model.py`, `data.py`, `client.py`, `server.py` — core FL components
- `serial_main.py`, `parallel_main.py` — runnable simulation entry points
- `plot_results.py`, `plot_amdahl.py` — analysis scripts
- `results/` — JSON results and generated plots
- `.notes/` — internal research notes (start with `research_plan.md`)

## Working preferences

- Prefer minimal edits to existing files; preserve the MNIST baseline as the
  reference unless explicitly told to change it.
- New research work belongs on a feature branch or in a new subdirectory, not
  by mutating the current MNIST baseline.
- When in doubt about scope or direction, check `.notes/research_plan.md` and
  ask the user before assuming.
