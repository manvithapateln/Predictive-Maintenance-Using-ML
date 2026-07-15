import streamlit as st
import plotly.graph_objects as go
import random

from ui_style import apply_global_style

st.set_page_config(layout="wide")
apply_global_style()

st.title("🔮 Single Machine Failure Prediction")
st.caption("Predict machine failure risk using sensor values.")

col1, col2 = st.columns(2)

with col1:
    air_temp = st.number_input("Air Temperature (K)", 250.0, 350.0, 300.0)
    process_temp = st.number_input("Process Temperature (K)", 250.0, 350.0, 310.0)
    rot_speed = st.number_input("Rotational Speed (RPM)", 1000.0, 3000.0, 1500.0)

with col2:
    torque = st.number_input("Torque (Nm)", 10.0, 100.0, 40.0)
    tool_wear = st.number_input("Tool Wear (min)", 0.0, 300.0, 50.0)
    machine_type = st.selectbox("Machine Type", ["L", "M", "H"])

if st.button("🚀 Predict Failure Risk"):

    risk = random.randint(15, 95)

    if risk < 30:
        st.success("🟢 Low Risk - Routine Maintenance")
    elif risk < 70:
        st.warning("🟡 Medium Risk - Schedule Inspection")
    else:
        st.error("🔴 High Risk - Immediate Maintenance Required")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        title={"text": "Failure Risk (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 30], "color": "lightgreen"},
                {"range": [30, 70], "color": "gold"},
                {"range": [70, 100], "color": "red"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("AI Explanation")

    st.write("Top factors influencing prediction:")

    st.progress(min(int(tool_wear / 3), 100))
    st.write("• Tool Wear")

    st.progress(min(int((torque / 100) * 100), 100))
    st.write("• Torque")

    st.progress(min(int((rot_speed - 1000) / 20), 100))
    st.write("• Rotational Speed")

    st.info("This is a demonstration interface for the project.")
