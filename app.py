import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import traceback
import sklearn
import sys
from pathlib import Path

st.set_page_config(
    page_title="Prediksi Pembatalan Reservasi Hotel",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}

h1, h2, h3 {
    color: #0f172a;
}

div[data-testid="stMetricValue"] {
    font-size: 28px;
    font-weight: 700;
}

div[data-testid="stMetricLabel"] {
    font-size: 14px;
    font-weight: 600;
}

.stButton>button {
    width: 100%;
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px;
}

.stButton>button:hover {
    background-color: #1d4ed8;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_files():
    try:
        model_path = BASE_DIR / "model_rf.pkl"
        preprocessor_path = BASE_DIR / "preprocessor.pkl"

        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)

        return model, preprocessor

    except Exception:
        st.error("❌ Gagal memuat model atau preprocessor")
        st.code(traceback.format_exc())
        return None, None

# Debug info
st.sidebar.markdown("### Informasi Sistem")
st.sidebar.write(f"Python: {sys.version.split()[0]}")
st.sidebar.write(f"Scikit-Learn: {sklearn.__version__}")

model, preprocessor = load_files()
