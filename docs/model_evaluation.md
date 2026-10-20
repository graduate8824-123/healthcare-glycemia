# Model evaluation

The baseline is `DummyClassifier(strategy="prior")`; the candidate model is a bounded CPU-only Random Forest (120 trees, depth 8, one worker). Median imputation is inside each fitted pipeline and therefore learned from training patients only.

Whole patient IDs are deterministically partitioned with random seed 2026. Test patients are never used for fitting or preprocessing. The final held-out test is evaluated once by the train command. With fewer than three patients, training fails rather than pretending that a random row split is valid. Metrics are persisted to `artifacts/model_metadata.json`.

Metric fields are precision, recall, F1, labels/order of the 2x2 confusion matrix, false positives, false negatives, positive prevalence, and ROC-AUC/PR-AUC when both test classes exist. No accuracy-only claim is made. Any synthetic result is labeled as synthetic demonstration only.

Newly trained metadata also records the scikit-learn, joblib, and NumPy versions used to serialize the artifact. The current local artifact was loaded successfully with scikit-learn 1.9.0 and emitted no scikit-learn compatibility warning. Under NumPy 2.5.1, joblib 1.5.3 emits a NumPy shape deprecation warning while unpickling the Random Forest arrays. This is an upstream serialization warning, not evidence of an incompatible scikit-learn artifact; it is deliberately not suppressed. Re-training under the same package versions would not remove it.
