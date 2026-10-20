# Presentation outline

1. **Title and team** — HypoGuard Twin; add team/institution placeholders.
2. **Problem** — Hypoglycaemic events can develop between static reviews; define the 30-minute recorded-event question.
3. **Digital Twin proposition** — A timestamped virtual state updates as each observation arrives.
4. **Data and provenance** — HUPA-UCM planned source, lawful access, schema verification, synthetic technical mode.
5. **Leakage prevention** — Current/past inputs only; future glucose used exclusively for labels; patient-level split.
6. **Architecture** — Local lightweight ingestion, model, API, SQLite metadata, and dashboard.
7. **Model and outputs** — Baseline plus Random Forest; uncalibrated estimates, feature importance caveat.
8. **Demonstration** — Dashboard, historical replay, API, and a constrained scenario comparison.
9. **Measured results** — Insert only saved, verified results and label synthetic metrics correctly.
10. **Limitations, safety, and future work** — calibration, external validation, fairness, privacy, governance, clinical evaluation.
