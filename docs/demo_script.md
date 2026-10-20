# Demonstration and submission checklist

## 20-minute video run-of-show

1. **0:00–1:30 — Problem and safety.** Explain the 30-minute recorded-event target and research-only disclaimer.
2. **1:30–3:30 — Why a twin.** Show state update, historical observations versus prediction, and no invented future trajectory.
3. **3:30–6:00 — Data.** Show `data/raw/` absence/presence, quality policy, HUPA-UCM attribution placeholder, and synthetic-mode banner.
4. **6:00–9:00 — Leakage safeguards.** Walk through past-only features and separately built future labels; show patient-level split metadata.
5. **9:00–12:00 — Training.** Run the synthetic command and open saved metadata. State clearly that synthetic metrics are not clinical evidence.
6. **12:00–16:30 — Live dashboard.** Select a patient, timeline, derived state, estimate, quality table, feature importance, and historical replay.
7. **16:30–18:00 — API.** Show `/docs`, health, valid prediction, malformed request, and unknown patient response.
8. **18:00–20:00 — Limits/future work.** Cover calibration, validation, privacy, fairness, and clinical governance.

## Execution checklist

- [ ] Start API on `127.0.0.1`, not a public interface.
- [ ] Confirm the dashboard data-mode banner.
- [ ] State whether the data are synthetic or real.
- [ ] Review quality warnings before showing a prediction.
- [ ] Do not present model estimates as diagnoses or treatment advice.

## Final submission checklist

- [ ] Fill team/institution placeholders.
- [ ] Verify dataset licence and attribution with its primary source.
- [ ] Remove patient data, secrets, local SQLite, and unsupported results.
- [ ] Run `python -m pytest -q` and record the actual output.
- [ ] Run the API and dashboard smoke checks.
- [ ] Include only verified evaluation results and label synthetic results.
