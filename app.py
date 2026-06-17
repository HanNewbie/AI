import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import traceback
import sklearn
import sys
from pathlib import Path

# =========================
# KONFIGURASI HALAMAN
# =========================

st.set_page_config(
    page_title="Prediksi Pembatalan Reservasi Hotel",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CSS
# =========================

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

.stButton > button {
    width: 100%;
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px;
}

.stButton > button:hover {
    background-color: #1d4ed8;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================

BASE_DIR = Path(__file__).parent

st.sidebar.markdown("### Informasi Sistem")
st.sidebar.write(f"Python: {sys.version.split()[0]}")
st.sidebar.write(f"Scikit-Learn: {sklearn.__version__}")

try:
    model = joblib.load(BASE_DIR / "model_rf.pkl")
    st.sidebar.success("✅ Model berhasil dimuat")
except Exception:
    st.error("❌ Gagal memuat model_rf.pkl")
    st.code(traceback.format_exc())
    st.stop()

try:
    preprocessor = joblib.load(BASE_DIR / "preprocessor.pkl")
    st.sidebar.success("✅ Preprocessor berhasil dimuat")
except Exception:
    st.error("❌ Gagal memuat preprocessor.pkl")
    st.code(traceback.format_exc())
    st.stop()

# =========================
# SIDEBAR
# =========================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/139/139899.png",
    width=100
)

st.sidebar.title("🏨 Menu Sistem")

menu = st.sidebar.selectbox(
    "Pilih Menu",
    [
        "Dashboard Analisis",
        "Sistem Prediksi"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Hotel Booking Demand Prediction")

# =========================
# DASHBOARD
# =========================

if menu == "Dashboard Analisis":

    st.title("📊 Dashboard Analisis Reservasi Hotel")

    st.markdown(
        """
        Dashboard ini menampilkan ringkasan dataset Hotel Booking Demand
        serta faktor-faktor yang memengaruhi pembatalan reservasi hotel.
        """
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Total Reservasi", "119.390")

    with m2:
        st.metric("Reservasi Batal", "44.224")

    with m3:
        st.metric("Reservasi Tidak Batal", "75.166")

    with m4:
        st.metric("ROC-AUC", "0.88")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Proporsi Status Reservasi")

        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=["Tidak Dibatalkan", "Dibatalkan"],
                    values=[75166, 44224],
                    hole=0.45
                )
            ]
        )

        fig_pie.update_layout(height=400)

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

    with col2:

        st.subheader("Feature Importance Random Forest")

        fitur = [
            "Lead Time",
            "Deposit Type",
            "ADR",
            "Market Segment",
            "Customer Type",
            "Previous Cancellations",
            "Previous Bookings Not Canceled",
            "Total Special Requests"
        ]

        importance = [
            0.32,
            0.24,
            0.14,
            0.10,
            0.08,
            0.05,
            0.04,
            0.03
        ]

        fig_imp = px.bar(
            x=importance,
            y=fitur,
            orientation="h"
        )

        fig_imp.update_layout(
            height=400,
            yaxis={"categoryorder": "total ascending"}
        )

        st.plotly_chart(
            fig_imp,
            use_container_width=True
        )

    st.subheader("📈 Hasil Evaluasi Model")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Accuracy", "81%")
    c2.metric("Precision", "79%")
    c3.metric("Recall", "68%")
    c4.metric("F1-Score", "73%")
    c5.metric("ROC-AUC", "0.88")

# =========================
# SISTEM PREDIKSI
# =========================

elif menu == "Sistem Prediksi":

    st.title("🔮 Sistem Prediksi Pembatalan Reservasi Hotel")

    st.markdown("""
    Halaman ini digunakan untuk melakukan prediksi kemungkinan
    pembatalan reservasi hotel berdasarkan karakteristik pemesanan pelanggan.
    """)

    with st.form("form_prediksi"):

        col1, col2 = st.columns(2)

        with col1:

            lead_time = st.number_input(
                "Jarak Waktu Pemesanan (Hari)",
                min_value=0,
                max_value=700,
                value=30
            )

            deposit_type = st.selectbox(
                "Jenis Deposit",
                ["No Deposit", "Non Refund", "Refundable"]
            )

            market_segment = st.selectbox(
                "Segmen Pasar",
                [
                    "Online TA",
                    "Offline TA/TO",
                    "Direct",
                    "Corporate",
                    "Groups"
                ]
            )

            customer_type = st.selectbox(
                "Tipe Pelanggan",
                [
                    "Transient",
                    "Transient-Party",
                    "Contract",
                    "Group"
                ]
            )

        with col2:

            adr = st.number_input(
                "Rata-rata Tarif Kamar (ADR)",
                min_value=0.0,
                value=100.0
            )

            total_special_requests = st.slider(
                "Jumlah Permintaan Khusus",
                0,
                5,
                1
            )

        submit_button = st.form_submit_button(
            "Lakukan Prediksi"
        )

    if submit_button:

        try:

            input_data = pd.DataFrame({
                "lead_time": [lead_time],
                "deposit_type": [deposit_type],
                "market_segment": [market_segment],
                "customer_type": [customer_type],
                "adr": [adr],
                "total_of_special_requests": [total_special_requests]
            })

            input_encoded = preprocessor.transform(input_data)

            prediction = model.predict(input_encoded)[0]

            prob_batal = model.predict_proba(input_encoded)[0][1]
            probabilitas = prob_batal * 100

            if prediction == 1:
                st.error(
                    "🔴 Reservasi Diprediksi Mungkin Dibatalkan"
                )
            else:
                st.success(
                    "🟢 Reservasi Diprediksi Mungkin Tidak Dibatalkan"
                )

            st.metric(
                "Probabilitas Pembatalan",
                f"{probabilitas:.2f}%"
            )

        except Exception:
            st.error("❌ Terjadi kesalahan saat prediksi")
            st.code(traceback.format_exc())
