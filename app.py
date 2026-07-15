import streamlit as st
from ui_style import apply_global_style

st.set_page_config(page_title="Predictive Maintenance System", layout="wide")
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
st.title("🏭 AI Predictive Maintenance Dashboard")
st.caption("Explainable Machine Learning for Smart Industrial Decision Support")
st.markdown(
    '<div class="blink-wrap"><div class="blink-dot"></div>AI Model Live</div>',
    unsafe_allow_html=True
)


# ---------- FEATURE SECTIONS ----------

st.markdown("### Single Machine Prediction")
st.write("Predict the probability of machine failure using sensor inputs and receive AI-based explanations using SHAP.")



st.markdown("### Fleet Monitoring")
st.write("Monitor multiple machines in real time with dynamic failure risk evaluation and live status updates.")



st.markdown("### Dataset Insights")
st.write("Analyze system-wide statistics, failure distribution, and operational sensor trends.")




# ---------- ABOUT SECTION ----------

st.markdown("### About This System")
st.write("""
This AI-powered system uses **XGBoost Machine Learning** and **SHAP Explainable AI**
to help industries predict equipment failures before breakdowns occur.

The platform supports:
- Preventive maintenance planning  
- Reduced machine downtime  
- Transparent AI decision-making  
- Real-time monitoring capability  
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown(
    "<div style='text-align:center; color:gray; margin-top:30px;'>"
    "Final Year Major Project — Explainable AI for Predictive Maintenance"
    "</div>",
    unsafe_allow_html=True
)
