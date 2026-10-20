"""Streamlit research dashboard; starts with `streamlit run dashboard.py`."""
import json
import pandas as pd
import plotly.express as px
import streamlit as st

from hypoguard_twin.runtime import Runtime

st.set_page_config(page_title="HypoGuard Twin", layout="wide")
@st.cache_resource
def runtime():
    return Runtime()

r = runtime()
st.title("HypoGuard Twin")
st.warning("Research prototype only — not a clinically validated medical device. It provides no diagnosis, treatment, or insulin-dose advice.")
st.info(f"Data mode: **{r.mode}**. Observations marked synthetic are only technical demonstration data.")
st.sidebar.header("Patient selection")
patient_id = st.sidebar.selectbox("Patient identifier", r.engine.patients())
history = pd.DataFrame(r.patient_history(patient_id))
history["timestamp"] = pd.to_datetime(history["timestamp"])
state = r.engine.derive_state(patient_id)
latest = state["latest_model_output"]

st.header("Patient summary")
columns = st.columns(4)
columns[0].metric("Latest recorded glucose", f"{state['latest_valid_glucose']:.0f} mg/dL" if state["latest_valid_glucose"] is not None else "Unavailable")
columns[1].metric("Observations", len(history))
columns[2].metric("History duration", str(history.timestamp.max() - history.timestamp.min()))
columns[3].metric("Data quality", f"{state['data_quality']['valid_glucose_observations']} valid glucose records")

st.header("Glucose timeline")
chart = px.line(history, x="timestamp", y="glucose", markers=True, labels={"glucose": "Recorded glucose (mg/dL)"})
chart.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="70 mg/dL reference")
st.plotly_chart(chart, width="stretch")

st.header("Hypoglycaemia event estimate")
if latest.get("available"):
    prediction = r.predict(patient_id)
    st.metric("Estimated event probability within next 30 minutes", f"{prediction['probability_low_30m']:.1%}")
    st.caption(f"Prediction timestamp: {prediction['prediction_timestamp']} · {prediction['calibration']}")
else:
    st.info(latest.get("message", "No valid prediction is available."))

st.header("Explainability")
metadata = r.metadata()
if metadata.get("feature_importance"):
    importance = pd.DataFrame(metadata["feature_importance"].items(), columns=["Feature", "Importance"])
    st.plotly_chart(px.bar(importance, x="Importance", y="Feature", orientation="h"), width="stretch")
    st.caption("Random-forest impurity importance from this fitted prototype model, not a patient-specific causal explanation.")
else:
    st.info("Feature importance is unavailable until a model has been trained.")

st.header("Historical replay")
position = st.slider("Recorded observation position", 0, max(0, len(history) - 1), 0)
if st.button("Replay selected historical point"):
    st.json(r.historical_replay(patient_id, position))
st.caption("Replay uses recorded historical observations; it is not live wearable data.")

st.header("Scenario comparison")
st.caption("Hypothetical inputs do not alter recorded observations and are not causal or treatment recommendations.")
if "steps_30m" in (r.model or {}).get("feature_columns", []):
    alternative_steps = st.number_input("Hypothetical trailing 30-minute steps", min_value=0.0, value=0.0)
    if st.button("Compare hypothetical activity"):
        st.json(r.scenario(patient_id, {"steps_30m": alternative_steps}))
else:
    st.info("No supported activity feature is available for scenario comparison.")

st.header("Model evaluation")
if metadata.get("evaluation"):
    st.json(metadata["evaluation"])
else:
    st.info("Evaluation results are unavailable because no model is trained.")
st.header("Data quality")
st.dataframe(pd.DataFrame(r.quality_summary()), width="stretch")
