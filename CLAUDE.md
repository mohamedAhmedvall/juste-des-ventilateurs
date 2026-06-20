# CLAUDE.md

Guidance for AI assistants (Claude Code and others) working in this repository.

## Project overview

**Juste des Ventilateurs** is a **predictive maintenance and thermal-regulation
service** for a *simulated* datacenter. It consumes telemetry from the
**jumeaux-chauds** digital twin (a separate project), predicts machine failures
(overheating / degraded mode / thermal shutdown) with ML, and drives the cooling
fans toward an optimal trade-off between **thermal safety** and **energy
sobriety**.

- **Domain language is French.** Code comments, docstrings, commit messages, and
  documentation are written in French. Match this convention — keep new comments
  and docs in French. Identifiers (functions, variables) are mostly English.
- **Academic project** (M2 Data/IA, LaPlateforme_). Organized in numbered
  *phases*; see `documents/roadmap.md`.
- The service is **read-from-MQTT, command-via-REST**: it never runs the physics
  itself — jumeaux-chauds owns the simulation.

## Prerequisites & external dependency

This service does nothing useful on its own. **jumeaux-chauds must be running**
and reachable:
- MQTT broker on `:1883` — publishes telemetry/status/fault/summary on topics
  `dt/{cluster}/{machine}/...`
- REST API on `:8000` — `GET /cluster/status`, `PUT /machines/{id}/fan_speed`,
  `PUT /machines/{id}/fan_mode`

Python **3.11+** required. Configuration is read from environment variables
(`.env`, see `.env.example`).

## Repository layout

```
ingest/        Phase 2 — MQTT collection → normalized Parquet datasets
  mqtt_subscriber.py   async subscriber (auto-reconnect, backoff) · CLI entrypoint
  normalizer.py        payload → unified schema (4 message types)
  dataset_exporter.py  Parquet export partitioned by episode/machine
features/      Phase 3 — offline feature engineering
  temporal.py · contextual.py · energy.py   feature builders
  labeler.py           failure_60s / failure_30s / hot_30s + action_class (oracle)
  pipeline.py          raw → features+labels · CLI entrypoint
models/
  failure_prediction/  baseline_threshold, logistic_regression, random_forest,
                       gradient_boosting, splitter
  fan_control/         baseline_fixed/threshold/pid, supervised_controller,
                       score_controller
supervisor/    Real-time supervision service
  supervisor.py        decision loop (predict → decide → command) · CLI entrypoint
  online_features.py   OnlineFeatureBuffer — per-machine sliding windows
  mqtt_telemetry.py    live telemetry consumer
  decision_logger.py   JSONL decision log
evaluation/    Offline comparative benchmarks
  benchmark.py · robustness.py · fan_control_eval.py · failure_prediction_eval.py
notebooks/     01..06 Jupyter analyses (ingestion, features, prediction,
               control, comparison, MQTT supervision)
data/          datasets (git-ignored except schema.md) — raw/ and processed/
documents/     roadmap.md, specifications.md, rapport_analyse.md
tests/         pytest suite (per-phase)
*.bat          Windows workflow runners (01..05, see below)
```

> **Note on the README:** `README.md` documents some **Phase 9** items
> (`evaluation/closed_loop_eval.py`, `notebooks/07_closed_loop_evaluation.ipynb`,
> closed-loop metrics) that are **planned but not yet present** in the codebase.
> Do not assume those files exist — verify before referencing them. The roadmap
> currently tops out at Phase 7/8 in code (`tests/test_phase8_oracle.py`).

## Core data flow

1. **Ingest** — `mqtt_subscriber` collects telemetry → `normalizer` maps to the
   unified schema (see `data/schema.md`) → `dataset_exporter` writes
   `data/raw/episode=NNN/machine=mXX/part-*.parquet` + `metadata.json`.
2. **Features** — `features.pipeline` turns raw telemetry into the ML dataset in
   `data/processed/episode=NNN/`, applying temporal → contextual → energy
   features, then failure/control labels, dropping warmup rows and critical-NaN
   rows.
3. **Train/eval** — `evaluation.failure_prediction_eval` trains failure models;
   `evaluation.fan_control_eval` evaluates controllers; `evaluation.benchmark` /
   `evaluation.robustness` produce comparative metrics in
   `evaluation/results/*.json` (label suffix in filename).
4. **Supervise** — `supervisor.supervisor` runs the live loop: MQTT telemetry →
   `OnlineFeatureBuffer` → predict risk → decide RPM → `PUT fan_speed`, with a
   REST `GET /cluster/status` fallback when MQTT is unavailable.

### Failure-prediction labels

| Label         | Horizon | Use |
|---------------|---------|-----|
| `failure_60s` | 60 s | **default**, used by the supervisor |
| `failure_30s` | 30 s | more precise, shorter lead time |
| `hot_30s`     | 30 s | thermal alert (temp > 95 % of shutdown threshold) |

Most workflow scripts/tools take a `--label` argument defaulting to
`failure_60s`.

### Supervisor decision constants (`supervisor/supervisor.py`)

- `RPM_LEVELS = [800, 1500, 2500, 3500, 4500]` — discrete fan setpoints.
- `RISK_THRESHOLD = 0.60` — above this risk score, override to `RPM_HIGH` (4500).
- `HOT30S_THRESHOLD` (env, default 0.5) — overheat override.
- `RISK_LOG_THRESHOLD` (env, default 0.05) — only log machines above this risk.
- Decisions fire every `DECISION_INTERVAL_TICKS` *simulated* ticks (1 tick = 1 s
  simulated), so behavior is independent of simulation speed.

## Common commands

Run from the repo root. The `*.bat` files are convenience wrappers for Windows;
on Linux/Mac call the underlying `python -m ...` commands directly.

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env            # then edit for your jumeaux-chauds host

# Tests (markers: "slow" integration tests excluded by default)
pytest tests/ -v
pytest tests/ -v -m "not slow"
pytest tests/test_phase4_models.py -v

# Ingest one episode
python -m ingest.mqtt_subscriber --duration 600 --episode 001 --scenario nominal
python -m ingest.mqtt_subscriber --continuous --episode 001 --scenario stress

# Feature engineering
python -m features.pipeline \
  --input data/raw/episode=001 \
  --output data/processed/episode=001 \
  --config data/raw/episode=001/metadata.json

# Train / evaluate (label-parameterized)
python -m evaluation.failure_prediction_eval --label failure_60s   # 03_train_models.bat
python -m evaluation.fan_control_eval --label failure_60s --models all --output evaluation/results/fan_control_results_failure_60s.json
python -m evaluation.benchmark --label failure_60s                 # 05_benchmark_offline_metrics.bat
python -m evaluation.robustness --label failure_60s
python ingest_quick_EDA.py --processed-only                        # quick EDA

# Run the supervisor
python -m supervisor.supervisor --mode ml --duration 300 --dry-run
python -m supervisor.supervisor --mode threshold

# Docker (jumeaux-chauds must be reachable; uses host.docker.internal)
docker compose up --build supervisor
```

### Batch-script map (Windows runners)

| Script | Underlying step |
|--------|-----------------|
| `01_ingest_mqtt_simulations.bat` | collect the 6 scenarios via MQTT |
| `02_ingest_gen_features.bat [NNN]` | run `features.pipeline` per episode |
| `03_train_models.bat [label]` | failure-prediction training/eval |
| `04_train_fan_controllers.bat` | controller training/eval |
| `05_benchmark_offline_metrics.bat [label]` | benchmark + robustness |
| `03_04_05_run_all_labels.bat` | run 03–05 across all 3 labels |
| `build-clean-app.bat` | full Docker rebuild |

## Conventions & gotchas

- **Module entrypoints:** runnable modules use `python -m <pkg>.<module>` with
  `argparse`. CLI modules: `ingest.mqtt_subscriber`, `features.pipeline`,
  `supervisor.supervisor`, and the four `evaluation.*` scripts.
- **`from __future__ import annotations`** is used throughout — keep it on new
  modules.
- **Saved models** live under `models/*/saved/` (joblib). These dirs are
  git-ignored and produced by the training scripts; don't assume they're present
  in a fresh clone — run training first, or tests that need them will skip/fail.
- **xgboost stub isolation:** `tests/conftest.py` strips a fake `xgboost` stub
  that `test_phase7_supervisor.py` may inject, so real joblib unpickling works in
  other phases. Don't remove this fixture; if you touch xgboost imports in tests,
  preserve the stub-cleanup behavior.
- **Logging:** the supervisor sets `httpx`/`httpcore` to WARNING to avoid
  per-request noise. Follow that pattern rather than re-enabling verbose HTTP
  logs.
- **Data is reproducible & git-ignored:** everything under `data/` (except
  `data/schema.md`) is regenerated from ingestion + features. Never commit
  datasets or trained models.
- **Time semantics:** "ticks" are *simulated* seconds, not wall-clock. Keep
  decision cadence in tick units (`DECISION_INTERVAL_TICKS`), reserving real
  seconds (`DECISION_INTERVAL_S`) for the REST fallback path only.
- **Online vs offline features:** the supervisor can only compute a subset of
  features live (`ONLINE_FEATURES` in `supervisor.py`). When adding model
  features, ensure they're derivable from `OnlineFeatureBuffer` or the predictor
  will break at runtime even if offline metrics look fine.

## Git workflow

- Develop on the assigned feature branch; never push to `main` without explicit
  permission. Push with `git push -u origin <branch>`.
- Commit messages are in French and follow Conventional-Commits-style prefixes
  (`feat`, `fix`, `docs`, often phase-scoped, e.g. `feat(phase8): ...`). Match
  this style.
- Do **not** open a pull request unless explicitly asked.

## Where to learn more

- `documents/roadmap.md` — phase-by-phase plan and deliverables.
- `documents/specifications.md` — technical specs, interfaces, metrics.
- `documents/rapport_analyse.md` — analysis report / results.
- `data/schema.md` — unified telemetry schema and Parquet partitioning.
- `README.md` — user-facing quickstart (note the Phase 9 caveat above).
