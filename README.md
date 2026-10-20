# HypoGuard Twin

### AI-powered digital twin prototype for early hypoglycaemia-event prediction

**Challenge:** Happiest Health Digital Twin Challenge 2026\
**Institution:** JECRC Foundation, Jaipur, Rajasthan, India\
**Repository:** https://github.com/graduate8824-123/healthcare-glycemia

> **Clinical safety notice:** This is an educational research prototype,
> not a medical device or validated clinical decision-support system. Do
> not use it to diagnose, treat, or make decisions about insulin,
> medication, food, or exercise.

## Team and institution

-   **Project title:** HypoGuard Twin --- AI-powered digital twin
    prototype for early hypoglycaemia-event prediction
-   **Team name:** `Nipun'
-   **Team leader:** Gaurav Khandelwal
    `[VERIFY spelling and required format]`

    `[ADD ALL MEMBERS AND ROLES; if solo, confirm solo participation is permitted]`
-   **College/institution:** JECRC Foundation, Jaipur, Rajasthan, India
    `[VERIFY official form wording]`
-   **Incubator/innovation centre:** `[ADD VERIFIED NAME OR N/A]`
-   **Team/contact details:**
    `[ADD ONLY DETAILS REQUIRED BY THE CHALLENGE]`

## Problem statement

Glucose time-series data can be irregular, incomplete, and difficult to
interpret. HypoGuard Twin demonstrates a technical workflow for
validating timestamped observations, building temporal features from
current and prior readings, evaluating a machine-learning model with
patient-level separation, and displaying a patient-state view.

The target is whether at least one **valid recorded glucose observation
below 70 mg/dL** occurs in `(t, t + 30 minutes]`. Missing future
measurements are not treated as proof that no event occurred.

## Healthcare use case

The prototype explores a short-horizon hypoglycaemia-event prediction
workflow for technical research and demonstration. It uses synthetic
demonstration data by default; real datasets must be obtained lawfully
and used according to their documentation and licence. This system is
not intended for patient care and must not guide diagnosis, treatment,
insulin dosing, medication, food, or exercise decisions.

## What is implemented

-   Timestamped CSV ingestion, delimiter detection, column
    normalization, validation, and data-quality summaries.
-   Past-only temporal feature construction and separate future-window
    labels for training/evaluation.
-   Synthetic demonstration data when `data/raw/` is empty, labelled as
    synthetic.
-   Scikit-learn baseline and CPU-friendly Random Forest model.
-   Patient-level train/test separation where supported by the data.
-   Model persistence and evaluation metadata.
-   Patient-state engine, SQLite metadata/prediction history, FastAPI
    service, and Streamlit/Plotly dashboard.
-   Historical replay and constrained hypothetical scenario comparison.
-   Automated tests in `tests/`.

## Technical stack

Python, pandas, NumPy, scikit-learn, joblib, FastAPI, Uvicorn, Pydantic,
Streamlit, Plotly, SQLite, pytest, and HTTPX.

## Architecture

``` mermaid
flowchart TD
  A[CSV files or synthetic demo] --> B[Validation and data-quality checks]
  B --> C[Past-only temporal features]
  B --> D[Future 30-minute labels for training only]
  C --> E[Patient-level train/test split]
  D --> E
  E --> F[Baseline and Random Forest]
  F --> G[Saved model and evaluation metadata]
  B --> H[Digital Twin state engine]
  G --> H
  H --> I[FastAPI local service]
  H --> J[Streamlit + Plotly dashboard]
  H --> K[(SQLite metadata and prediction history)]
```

**Architecture diagram PDF:** `HypoGuard_Twin_Architecture_Diagram.pdf`
--- add this file to the public repository, then link it here using the
repository path or public URL.

## Machine-learning model and evaluation

-   **Baseline:** `DummyClassifier(strategy="prior")`.
-   **Candidate model:** scikit-learn Random Forest with 120 trees,
    maximum depth 8, and one worker.
-   **Target:** positive if at least one valid measured glucose value
    below 70 mg/dL occurs in `(t, t + 30 minutes]`; negative only when
    valid future observations exist and none is below 70; rows with no
    valid future observation are excluded.
-   **Leakage controls:** features use current and prior observations
    only; patient identifiers are separated at patient level where
    possible.
-   **Metrics:** the pipeline may report precision, recall, F1, false
    positives/negatives, confusion matrix, and PR-AUC/ROC-AUC when the
    held-out set supports them.
-   **Limitations:** model estimates are uncalibrated. Synthetic metrics
    are technical smoke-test results, not clinical evidence.

See [`docs/model_evaluation.md`](docs/model_evaluation.md) and
[`docs/limitations.md`](docs/limitations.md).

## Quick start

Requires a supported Python version (project guidance: Python
3.10--3.13).

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m scripts.train --mode synthetic
```

Start the API in one terminal:

``` bash
uvicorn hypoguard_twin.api:app --host 127.0.0.1 --port 8000
```

Start the dashboard in another terminal:

``` bash
source .venv/bin/activate
streamlit run dashboard.py
```

Open the Streamlit URL printed in the terminal. API docs:
`http://127.0.0.1:8000/docs`.

## Tests

``` bash
python -m pytest -q
python -m scripts.train --mode synthetic
```

## Repository structure

``` text
hypoguard_twin/       ingestion, features, model, twin, persistence, API
scripts/train.py      reproducible local training
dashboard.py          Streamlit dashboard
tests/                automated tests
data/raw/             user-supplied real data; keep private/ignored
data/synthetic/       synthetic demonstration CSVs
artifacts/            generated model and metadata
docs/                 architecture, data dictionary, limitations, demo material
```

## Challenge deliverables

### Demo video --- mandatory

-   **Required duration:** 15--20 minutes.
-   **Hosting:** YouTube set to **Unlisted**.
-   **Video URL:** `[ADD FINAL UNLISTED YOUTUBE LINK AFTER UPLOAD]`
-   **Status:** `[NOT COMPLETE UNTIL VIDEO IS UPLOADED AND LINK TESTED]`

The link must open without requiring evaluators to request access. See
[`docs/demo_script.md`](docs/demo_script.md) for a suggested
run-of-show.

### Architecture diagram --- mandatory

-   **File:** `HypoGuard_Twin_Architecture_Diagram.pdf`
-   **Repository path/link:**
    `[AFTER UPLOADING, ADD PUBLIC REPOSITORY PATH OR URL HERE]`

### Project presentation --- mandatory

-   **File:** `HypoGuard_Twin_Project_Presentation.pptx`
-   **Repository path/link:**
    `[AFTER UPLOADING, ADD PUBLIC REPOSITORY PATH OR URL HERE]`

The presentation is a challenge-ready draft. Review it and replace the
team-name placeholder before recording/submission.

### Open-source licence

-   **Licence:** MIT
-   **Licence file:** [`LICENSE`](LICENSE)
-   **Action:** Verify the copyright holder/team details in `LICENSE`.
    Dataset licences and attribution are separate from the source-code
    licence.

## Unstop submission checklist

The following checklist mirrors the required items shown in the Digital
Twin Challenge 2026 --- Prototype & Code Submission form. Check an item
in Unstop only after the corresponding content or deliverable is
genuinely present and publicly accessible.

-   [x] **Team details section included in README** --- replace
    placeholders with confirmed team name, leader, member names/roles.
-   [x] **College/incubator information section included in README** ---
    verify the official college name and add verified incubator details
    or N/A.
-   [x] **Project title included in README.**
-   [x] **Problem statement included in README.**
-   [x] **Healthcare use case included in README.**
-   [x] **Technical stack included in README.**
-   [x] **AI/ML model or framework details included in README.**
-   [ ] **15--20 minute demo uploaded to YouTube as Unlisted and link
    added to README.** `[PENDING: ADD LINK]`
-   [x] **Open-source licence details described in README.** Verify the
    `LICENSE` file before submission.
-   [x] **Architecture diagram PDF/PPT created.** Upload the PDF to the
    public repository and add its public path/link above.
-   [x] **Project presentation PDF/PPT created.** Upload the PPTX to the
    public repository and add its public path/link above.
-   [ ] **All files and links publicly accessible without additional
    permissions.**
    `[PENDING: upload the deliverables, test every link in a private/logged-out browser, and verify repository visibility]`

### Final submission verification

-   [ ] Replace every `[ADD ...]`, `[VERIFY ...]`, and `[PENDING ...]`
    placeholder.
-   [ ] Confirm repository visibility is Public.
-   [ ] Upload the architecture PDF and presentation PPTX into the
    repository (suggested location: `docs/submission/`).
-   [ ] Upload and link the Unlisted YouTube demo, 15--20 minutes long.
-   [ ] Verify every linked file opens without signing in or requesting
    access.
-   [ ] Verify `LICENSE` details.
-   [ ] Ensure no secrets, credentials, or unauthorized private health
    data are committed.

## Data and privacy

The [HUPA-UCM Diabetes Dataset landing
page](https://data.mendeley.com/datasets/3hbcscwz44/1) is one possible
dataset source. Read its documentation and licence before use. The
project does not download it automatically. Never commit private patient
data, credentials, `.env` files, or data you are not permitted to
redistribute.

## Limitations and safety

-   This is not a medical device or validated clinical decision-support
    tool.
-   It has not been established as safe or effective for patient care.
-   Synthetic data cannot establish real-world performance.
-   Predictions can be wrong; estimates are uncalibrated.
-   Missingness, irregular observations, dataset quality, and
    representativeness affect interpretation.
-   Do not use outputs to make diagnosis, treatment, insulin,
    medication, food, or exercise decisions.

See [`docs/limitations.md`](docs/limitations.md).

## Licence

MIT. See [`LICENSE`](LICENSE). Verify copyright details and all
third-party data obligations before public submission.
