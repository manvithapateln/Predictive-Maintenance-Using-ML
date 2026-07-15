import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# ---------- LOAD MODEL FILES ----------
model = joblib.load("models/failure_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# ---------- CUSTOM STYLING ----------
from ui_style import apply_global_style
apply_global_style()
st.markdown("""
<style>
.blink-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: -5px;
    margin-bottom: 10px;
    font-size: 14px;
    font-weight: 600;
    color: #16a34a;
}

.blink-dot {
    height: 10px;
    width: 10px;
    background-color: #22c55e;
    border-radius: 50%;
    animation: blink 1.2s infinite;
}

@keyframes blink {
    0% { opacity: 1; }
    50% { opacity: 0.2; }
    100% { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)


# ---------- HEADER ----------




st.title("🔮 Single Machine Failure Prediction")
st.markdown(
    '<div class="blink-wrap"><div class="blink-dot"></div>AI Model Live</div>',
    unsafe_allow_html=True
)

# ---------- INPUT PANEL ----------
st.caption("Predict the probability of machine failure using sensor inputs and receive AI-based explanations using SHAP.")

col1, col2 = st.columns(2)

with col1:
    air_temp = st.number_input("Air Temperature (K)", 250.0, 350.0, 300.0)
    process_temp = st.number_input("Process Temperature (K)", 250.0, 350.0, 310.0)
    rot_speed = st.number_input("Rotational Speed (rpm)", 1000.0, 3000.0, 1500.0)

with col2:
    torque = st.number_input("Torque (Nm)", 10.0, 100.0, 40.0)
    tool_wear = st.number_input("Tool Wear (min)", 0.0, 300.0, 50.0)
    machine_type = st.selectbox("Machine Type", ["L", "M", "H"])



# ---------- PREDICTION ----------
if st.button("🚀 Predict Failure Risk"):

    type_L = 1 if machine_type == "L" else 0
    type_M = 1 if machine_type == "M" else 0

    input_dict = {
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Type_L": type_L,
        "Type_M": type_M
    }

    input_df = pd.DataFrame([input_dict])[feature_names]
    scaled = scaler.transform(input_df)
    prob = model.predict_proba(scaled)[0][1]

    # ---------- RESULT CARD ----------
   
    st.markdown('<p class="section-title">Prediction Result</p>', unsafe_allow_html=True)

    

    

    if prob < 0.3:
        st.success("🟢 Low Risk — Routine Maintenance")
    elif prob < 0.7:
        st.warning("🟡 Medium Risk — Plan Inspection")
    else:
        st.error("🔴 High Risk — Immediate Maintenance Required")

    
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Failure Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#0B3B5B"},
            'steps': [
                {'range': [0, 30], 'color': "#97DFB9"},
                {'range': [30, 70], 'color': "#E4D38B"},
                {'range': [70, 100], 'color': "#E28585"},
            ],
        }
    ))
        # ---- Animated Risk Progress Bar ----
    st.markdown("### Risk Level")

    progress_placeholder = st.empty()

    for i in range(int(prob * 100) + 1):
        progress_placeholder.progress(i)

    st.plotly_chart(gauge, use_container_width=True)


    

    # ---------- SHAP EXPLANATION ----------
   
    st.markdown('<p class="section-title">🔍 AI Explanation (SHAP)</p>', unsafe_allow_html=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(scaled)

    # ---- Compact Feature Impact Chart ----
    import plotly.express as px

    # Get SHAP values for this prediction
    values = shap_values[0][0] if isinstance(shap_values, list) else shap_values[0]

    impact_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": values
    })

    # Take top 6 most important features
    impact_df["AbsImpact"] = impact_df["Impact"].abs()
    impact_df = impact_df.sort_values("AbsImpact", ascending=False).head(6)
    impact_df = impact_df.sort_values("Impact")

    fig = px.bar(
        impact_df,
        x="Impact",
        y="Feature",
        orientation="h",
        color="Impact",
        color_continuous_scale=["#F94D4D", "#837e7e", "#6BB890"],
        title="Top Factors Influencing Prediction"
    )

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        coloraxis_showscale=False
    )

    st.plotly_chart(fig, use_container_width=True)


    st.markdown("These features contributed most to the predicted failure risk.")
