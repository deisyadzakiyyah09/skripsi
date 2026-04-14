import streamlit as st
from PIL import Image

# ======================
# CONFIG
# ======================
st.set_page_config(
    page_title="Deteksi Kanker Kulit",
    layout="centered"
)

# ======================
# STYLE MINIMAL (AMAN)
# ======================
st.markdown("""
<style>
h1, h2, h3 {
    color: #1f4e79;
}
</style>
""", unsafe_allow_html=True)

# ======================
# SIDEBAR
# ======================
menu = st.sidebar.radio(
    "Menu",
    ["🏠 Home", "🤖 Demo AI", "📊 Tentang Model"]
)

# ======================
# HOME
# ======================
if menu == "🏠 Home":
    st.title("🧠 Deteksi Dini Kanker Kulit")
    st.caption("MobileNetV2 + Explainable AI (LIME)")

    st.write("""
    Aplikasi ini digunakan untuk membantu proses deteksi dini kanker kulit 
    berdasarkan citra menggunakan model Deep Learning MobileNetV2 
    serta Explainable AI (LIME).
    """)

    st.subheader("🎯 Tujuan Penelitian")
    st.markdown("""
    - Mengklasifikasikan kanker kulit (Benign dan Malignant)  
    - Menggunakan arsitektur MobileNetV2  
    - Menampilkan interpretasi model menggunakan LIME  
    """)

# ======================
# DEMO AI
# ======================
elif menu == "🤖 Demo AI":
    st.title("🔍 Demo Deteksi Kanker Kulit")

    uploaded_file = st.file_uploader(
        "Upload Gambar Kulit",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Gambar Input", use_column_width=True)

        with col2:
            st.subheader("Hasil Prediksi")

            # Placeholder UI (AMAN untuk skripsi)
            st.info("Label: Benign / Malignant")
            st.write("Confidence: -")

            st.subheader("Penjelasan (LIME)")
            st.warning("Visualisasi LIME akan ditampilkan di sini")

# ======================
# TENTANG MODEL
# ======================
elif menu == "📊 Tentang Model":
    st.title("📊 Tentang Model")

    st.subheader("🤖 MobileNetV2")
    st.write("Model CNN ringan yang digunakan untuk klasifikasi citra.")

    st.subheader("📂 Dataset")
    st.markdown("""
    Dataset berasal dari Mendeley dengan dua kelas:
    - Benign
    - Malignant
    """)

    st.subheader("⚙️ Preprocessing")
    st.markdown("""
    - Resize (224x224)
    - Normalisasi (0–1)
    """)

    st.subheader("🔍 Explainable AI (LIME)")
    st.write("Digunakan untuk menunjukkan area penting pada citra.")
