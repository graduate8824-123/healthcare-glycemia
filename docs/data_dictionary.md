# Data dictionary and ingestion policy

Required inputs: timestamp (`timestamp`, `time`, `datetime`, `date_time`, `date`) and glucose (`glucose`, `cgm`, `sensor_glucose`, `sgv`, `blood_glucose`). A `patient_id` can be provided; otherwise the CSV filename is used. Optional aliases: heart rate (`heart_rate`, `heartrate`, `hr`), steps, carbs/carbohydrates, basal/bolus insulin, and sleep.

Column names are normalized to lowercase snake case while source columns are retained in the file-quality report. Timestamps are parsed as UTC then made timezone-naive, sorted per patient, and malformed values are flagged. Glucose must be 20–600 mg/dL in this prototype. Missing values remain missing. Glucose 0 is invalid; zero steps and zero recorded carbs can remain valid values.

Features use measured current/past data. Lag values allow a small timestamp tolerance; 30-minute trailing aggregates use actual timestamps, not a fixed row count. An irregular previous interval is represented explicitly. Records with no valid future glucose in `(t, t+30min]` are excluded from labels.
