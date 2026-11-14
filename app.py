"""Agricultural Robot Localization Dashboard — entry point.

Run:  streamlit run app.py
"""
import streamlit as st

nav = st.navigation([
    st.Page("pages/0_Mission_Control.py", title="Mission Control", icon="🛰️", default=True),
    st.Page("pages/1_GPS_Outage_Lab.py", title="GPS Outage Lab", icon="📡"),
    st.Page("pages/2_Sensor_Intelligence.py", title="Sensor Intelligence", icon="🧭"),
    st.Page("pages/3_Orchard_Risk_Map.py", title="Orchard Risk Map", icon="🌳"),
    st.Page("pages/4_Worker_Safety.py", title="Worker Safety", icon="👷"),
    st.Page("pages/5_Results.py", title="Results and Evaluation", icon="📊"),
    st.Page("pages/6_Data_Import.py", title="Data Import", icon="📥"),
])
nav.run()
