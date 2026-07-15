import streamlit as st

def apply_global_style():
    st.markdown("""
    <style>
    .stApp{
        background-color:#0E1117;
        color:white;
    }

    section[data-testid="stSidebar"]{
        background-color:#161B22;
    }

    .stButton>button{
        background:#238636;
        color:white;
        border:none;
        border-radius:10px;
        padding:10px 20px;
        font-weight:bold;
    }

    .stButton>button:hover{
        background:#2EA043;
        color:white;
    }

    h1,h2,h3{
        color:#58A6FF;
    }
    </style>
    """, unsafe_allow_html=True)
