import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide")
st_autorefresh(interval=2000, key="fleetrefresh")


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
from streamlit_autorefresh import st_autorefresh

# Refresh every 2 seconds (2000 ms)





st.title("🏭 Real-Time Fleet Monitoring")
st.markdown(
    '<div class="blink-wrap"><div class="blink-dot"></div>AI Model Live</div>',
    unsafe_allow_html=True
)

from datetime import datetime
st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")



# ---------- MACHINE COUNT ----------

st.markdown('<p class="section-title">Fleet Size</p>', unsafe_allow_html=True)
num_machines = st.slider("Number of Machines", 3, 12, 5)


# ---------- GENERATE MACHINE DATA ----------
def generate_machine_data(n):
    machines = []
    for i in range(n):
        air_temp = np.random.uniform(290, 340)
        process_temp = np.random.uniform(300, 350)
        rot_speed = np.random.uniform(1200, 3000)
        torque = np.random.uniform(20, 100)
        tool_wear = np.random.uniform(0, 250)

        machine_type = np.random.choice(["L", "M", "H"])
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

        df_input = pd.DataFrame([input_dict])[feature_names]
        scaled = scaler.transform(df_input)
        prob = model.predict_proba(scaled)[0][1]

        if prob < 0.3:
            status = "Low"
        elif prob < 0.7:
            status = "Medium"
        else:
            status = "High"

        machines.append([
            f"M-{100+i}",
            round(air_temp, 1),
            round(rot_speed, 0),
            round(torque, 1),
            round(tool_wear, 1),
            f"{prob*100:.1f}%",
            status
        ])

    return pd.DataFrame(machines, columns=[
        "Machine ID", "Air Temp", "Speed", "Torque", "Tool Wear", "Risk", "Status"
    ])



df_live = generate_machine_data(num_machines)
# ---------- STATUS TABLE ----------

st.markdown('<p class="section-title">Live Machine Status</p>', unsafe_allow_html=True)

def color_status(val):
    if val == "High":
        return "background-color:#7f1d1d; color:white;"
    elif val == "Medium":
        return "background-color:#78350f; color:white;"
    else:
        return "background-color:#14532d; color:white;"

styled_df = df_live.style.applymap(color_status, subset=["Status"])
st.dataframe(styled_df, use_container_width=True)



# ---------- RISK DISTRIBUTION ----------

st.markdown('<p class="section-title">Risk Distribution</p>', unsafe_allow_html=True)

risk_counts = df_live["Status"].value_counts().reset_index()
risk_counts.columns = ["Risk Level", "Count"]

fig = px.bar(
    risk_counts,
    x="Risk Level",
    y="Count",
    color="Risk Level",
    color_discrete_map={
        "Low": "#22c55e",
        "Medium": "#f59e0b",
        "High": "#ef4444"
    }
)

st.plotly_chart(fig, use_container_width=True)


# ---------- ALERT SECTION ----------
high_count = (df_live["Status"] == "High").sum()

if high_count > 0:
    st.error(f"⚠ ALERT: {high_count} machine(s) at HIGH RISK! Immediate inspection recommended.")
else:
    st.success("✅ All machines operating within safe limits")
