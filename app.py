import streamlit as st
from PIL import Image

# ======================
# KONFIGURASI HALAMAN
# ======================
st.set_page_config(
    page_title="Deteksi Kanker Kulit",
    layout="centered"
)

# ======================
# STYLE SEDERHANA
# ======================
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #1f4e79;
    }
    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #555;
        margin-bottom: 20px;
    }
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

# ======================
# SIDEBAR
# ======================
menu = st.sidebar.radio(
    "Menu",
    ["Home", "Demo AI", "Tentang Model"]
)

# ======================
# HALAMAN HOME
# ======================
if menu == "Home":
    st.markdown('<div class="title">Deteksi Dini Kanker Kulit</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">MobileNetV2 + Explainable AI (LIME)</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("""
    Aplikasi ini digunakan untuk membantu proses deteksi dini kanker kulit 
    berdasarkan citra menggunakan model Deep Learning.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Tujuan Penelitian")
    st.write("""
    - Mengklasifikasikan kanker kulit (benign dan malignant)
    - Menggunakan model MobileNetV2
    - Menampilkan interpretasi model menggunakan LIME
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ======================
# HALAMAN DEMO AI
# ======================
elif menu == "Demo AI":
    st.markdown('<div class="title">Demo Deteksi Kanker Kulit</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload Gambar Kulit",
        type=["jpg", "png", "jpeg"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(image, caption="Gambar Input", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("Hasil Prediksi akan ditampilkan di sini")
            st.write("Confidence: -")
            st.write("Visualisasi LIME akan muncul di sini")
            st.markdown('</div>', unsafe_allow_html=True)

# ======================
# HALAMAN TENTANG MODEL
# ======================
elif menu == "Tentang Model":
    st.markdown('<div class="title">Tentang Model</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("MobileNetV2")
    st.write("Model CNN ringan yang digunakan untuk klasifikasi citra.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Dataset")
    st.write("""
    Dataset terdiri dari dua kelas:
    - Benign
    - Malignant
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Preprocessing")
    st.write("""
    - Resize (224x224)
    - Normalisasi (0–1)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Explainable AI (LIME)")
    st.write("Digunakan untuk menunjukkan area penting pada citra.")
    st.markdown('</div>', unsafe_allow_html=True)
