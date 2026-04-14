import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Deteksi Kanker Kulit", layout="centered")

# ======================
# STYLE (BIAR LEBIH BAGUS)
# ======================
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #1f4e79;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ======================
# SIDEBAR
# ======================
menu = st.sidebar.radio("Menu", ["🏠 Home", "🤖 Demo AI", "📊 Tentang Model"])

# ======================
# HOME
# ======================
if menu == "🏠 Home":
    st.markdown('<div class="title">🧠 Deteksi Dini Kanker Kulit</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">MobileNetV2 + Explainable AI (LIME)</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("""
    Aplikasi ini digunakan untuk mendeteksi kanker kulit berdasarkan citra 
    menggunakan model Deep Learning MobileNetV2 serta Explainable AI (LIME).
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎯 Tujuan Penelitian")
    st.write("""
    ✔ Klasifikasi kanker kulit (Benign vs Malignant)  
    ✔ Menggunakan MobileNetV2  
    ✔ Menampilkan interpretasi model dengan LIME  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ======================
# DEMO AI
# ======================
elif menu == "🤖 Demo AI":
    st.markdown('<div class="title">🔍 Demo Deteksi Kanker Kulit</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Gambar Kulit", type=["jpg", "png", "jpeg"])
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
            if st.button("🔍 Prediksi"):
                
                with st.spinner("Model sedang menganalisis..."):
                    # dummy prediction
                    pred = np.random.rand()

                if pred > 0.5:
                    st.error("⚠️ Malignant (Berpotensi Kanker)")
                else:
                    st.success("✅ Benign (Tidak Berbahaya)")

                st.write(f"Confidence: {pred:.2f}")

                st.subheader("📌 Penjelasan (LIME)")
                st.info("Visualisasi LIME akan muncul di sini")
            st.markdown('</div>', unsafe_allow_html=True)

# ======================
# TENTANG MODEL
# ======================
elif menu == "📊 Tentang Model":
    st.markdown('<div class="title">📊 Tentang Model</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🤖 MobileNetV2")
    st.write("Model CNN ringan yang digunakan untuk klasifikasi citra.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📂 Dataset")
    st.write("""
    Dataset berasal dari Mendeley dengan 2 kelas:
    - Benign
    - Malignant
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⚙️ Preprocessing")
    st.write("""
    - Resize (224x224)
    - Normalisasi (0–1)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 Explainable AI (LIME)")
    st.write("Digunakan untuk menunjukkan area penting pada citra.")
    st.markdown('</div>', unsafe_allow_html=True)
