import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

# ---------- LOAD DATA ----------
data = pd.read_csv("data/ai4i2020.csv")

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



st.title("📊 Dataset Insights & System Overview")

# ---------- BASIC METRICS ----------
total = len(data)
failures = data["Machine failure"].sum()
failure_rate = (failures / total) * 100
avg_temp = data["Air temperature [K]"].mean()
avg_wear = data["Tool wear [min]"].mean()


st.markdown('<p class="section-title">Overall System Statistics</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Machines", total)
col2.metric("Failure Cases", failures)
col3.metric("Failure Rate", f"{failure_rate:.2f}%")
col4.metric("Avg Tool Wear", f"{avg_wear:.1f} min")



# ---------- FAILURE DISTRIBUTION ----------

st.markdown('<p class="section-title">Failure vs Normal Distribution</p>', unsafe_allow_html=True)

pie_fig = px.pie(
    names=["Normal", "Failure"],
    values=[total - failures, failures],
    color_discrete_sequence=["#74a1d5", "#ef4444"]
)
st.plotly_chart(pie_fig, use_container_width=True)


# ---------- SENSOR DISTRIBUTION ----------

st.markdown('<p class="section-title">Sensor Value Distributions</p>', unsafe_allow_html=True)

colA, colB = st.columns(2)

with colA:
    fig_temp = px.histogram(data, x="Air temperature [K]", nbins=30, color_discrete_sequence=["#200966"])
    st.plotly_chart(fig_temp, use_container_width=True)

with colB:
    fig_wear = px.histogram(data, x="Tool wear [min]", nbins=30, color_discrete_sequence=["#155D68"])
    st.plotly_chart(fig_wear, use_container_width=True)



# ---------- CORRELATION HEATMAP ----------

st.markdown('<p class="section-title">Feature Correlation Heatmap</p>', unsafe_allow_html=True)

corr = data.corr(numeric_only=True)
fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
st.plotly_chart(fig_corr, use_container_width=True)
