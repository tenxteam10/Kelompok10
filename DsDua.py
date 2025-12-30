import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from streamlit_option_menu import option_menu
import pandas as pd
from streamlit_lottie import st_lottie
import requests

# Fungsi untuk mendeteksi plat nomor menggunakan edge detection, morfologi, dan contour filtering
def detect_license_plate(image, canny_min=100, canny_max=200, kernel_size=5, min_area=500, max_area=50000, aspect_ratio_min=2.0, aspect_ratio_max=5.0):
    # Konversi ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Edge detection menggunakan Canny
    edges = cv2.Canny(gray, canny_min, canny_max)
    
    # Transformasi morfologi: Closing untuk mengisi celah
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Temukan kontur
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter kontur berdasarkan area dan aspect ratio
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if aspect_ratio_min < aspect_ratio < aspect_ratio_max:
                filtered_contours.append(cnt)
    
    # Gambar bounding box pada gambar asli
    result_image = image.copy()
    cropped_plates = []
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped = image[y:y+h, x:x+w]
        cropped_plates.append(cropped)
    
    return result_image, cropped_plates

# Fungsi untuk mengkonversi gambar ke bytes untuk download
def image_to_bytes(image):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    img_pil.save(buf, format='PNG')
    buf.seek(0)
    return buf

# Fungsi untuk load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Streamlit App
st.set_page_config(page_title="Deteksi Plat Nomor", layout="wide")

# Aplikasi langsung berjalan tanpa autentikasi
st.write("Welcome to the License Plate Detection App!")

# Navigasi menu di atas dengan posisi center
col1, col2, col3 = st.columns([1, 2, 1])  # Kolom untuk center menu
with col2:
    selected = option_menu(
        menu_title=None,  # Tidak ada judul menu
        options=["Home", "Edge Detection", "Morphological Transformation", "Contour Filtering"],
        icons=["house", "edge", "morph", "filter"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",  # Horizontal di atas
    )

# Parameter default
canny_min = 100
canny_max = 200
kernel_size = 5
min_area = 500
max_area = 50000
aspect_ratio_min = 2.0
aspect_ratio_max = 5.0

# Konten berdasarkan pilihan menu
if selected == "Home":
    st.title("Home - Deteksi Plat Nomor")
    st.markdown("Selamat datang di aplikasi deteksi plat nomor. Gunakan navigasi untuk mengakses fitur.")

    # Animasi Lottie
    lottie_url = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"  # Contoh URL Lottie
    lottie_json = load_lottieurl(lottie_url)
    st_lottie(lottie_json, height=300)

    # Tabel interaktif menggunakan st.dataframe (pengganti streamlit-aggrid)
    st.subheader("Contoh Tabel Data Plat Nomor")
    df = pd.DataFrame({
        'ID': [1, 2, 3],
        'Plat Nomor': ['ABC123', 'DEF456', 'GHI789'],
        'Status': ['Terdeteksi', 'Tidak Terdeteksi', 'Terdeteksi']
    })
    st.dataframe(df, use_container_width=True)  # Menggunakan st.dataframe untuk tabel interaktif

    # Diganti dari st_extras.write ke st.write
    st.write("Ini adalah contoh penggunaan streamlit-extras untuk elemen UI tambahan.")

elif selected == "Edge Detection":
    st.title("Edge Detection")
    st.markdown("Atur parameter untuk deteksi tepi menggunakan algoritma Canny.")

    # Layout: Kiri untuk parameter dan download, kanan untuk gambar
    col_left, col_right = st.columns([1, 2])  # Kolom kiri lebih kecil untuk parameter

    with col_left:
        st.subheader("Pengaturan Parameter")
        canny_min = st.slider("Canny Min Threshold", 0, 255, 100, help="Threshold minimum untuk deteksi tepi.")
        canny_max = st.slider("Canny Max Threshold", 0, 255, 200, help="Threshold maksimum untuk deteksi tepi.")

        # Upload gambar
        uploaded_file = st.file_uploader("Upload Gambar Kendaraan", type=["jpg", "jpeg", "png"])

        # Button proses
        if st.button("Proses"):
            if uploaded_file is not None:
                with st.spinner("Memproses..."):
                    image = Image.open(uploaded_file)
                    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    result_image, _ = detect_license_plate(image_cv, canny_min=canny_min, canny_max=canny_max)
                    st.session_state['result_image'] = result_image  # Simpan hasil untuk download
            else:
                st.error("Silakan upload gambar terlebih dahulu.")

        # Button download (jika ada hasil)
        if 'result_image' in st.session_state:
            buf = image_to_bytes(st.session_state['result_image'])
            st.download_button(
                label="Download Hasil Gambar",
                data=buf,
                file_name="hasil_deteksi_plat.png",
                mime="image/png"
            )

    with col_right:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar Asli", use_container_width=True)
        if 'result_image' in st.session_state:
            st.image(cv2.cvtColor(st.session_state['result_image'], cv2.COLOR_BGR2RGB), caption="Hasil Deteksi", use_container_width=True)

elif selected == "Morphological Transformation":
    st.title("Morphological Transformation")
    st.markdown("Atur parameter untuk operasi morfologi (closing) untuk mengisi celah pada tepi.")

    # Layout: Kiri untuk parameter dan download, kanan untuk gambar
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Pengaturan Parameter")
        kernel_size = st.slider("Kernel Size", 1, 15, 5, step=2, help="Ukuran kernel untuk operasi morfologi.")

        # Upload gambar
        uploaded_file = st.file_uploader("Upload Gambar Kendaraan", type=["jpg", "jpeg", "png"])

        # Button proses
        if st.button("Proses"):
            if uploaded_file is not None:
                with st.spinner("Memproses..."):
                    image = Image.open(uploaded_file)
                    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    result_image, _ = detect_license_plate(image_cv, kernel_size=kernel_size)
                    st.session_state['result_image'] = result_image
            else:
                st.error("Silakan upload gambar terlebih dahulu.")

        # Button download
        if 'result_image' in st.session_state:
            buf = image_to_bytes(st.session_state['result_image'])
            st.download_button(
                label="Download Hasil Gambar",
                data=buf,
                file_name="hasil_deteksi_plat.png",
                mime="image/png"
            )

    with col_right:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar Asli", use_container_width=True)
        if 'result_image' in st.session_state:
            st.image(cv2.cvtColor(st.session_state['result_image'], cv2.COLOR_BGR2RGB), caption="Hasil Deteksi", use_container_width=True)

elif selected == "Contour Filtering":
    st.title("Contour Filtering")
    st.markdown("Atur parameter untuk filtering kontur berdasarkan area dan rasio aspek.")

    # Layout: Kiri untuk parameter dan download, kanan untuk gambar
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Pengaturan Parameter")
        min_area = st.slider("Min Area", 100, 10000, 500, help="Area minimum kontur yang dianggap sebagai plat.")
        max_area = st.slider("Max Area", 1000, 100000, 50000, help="Area maksimum kontur yang dianggap sebagai plat.")
        aspect_ratio_min = st.slider("Aspect Ratio Min", 1.0, 10.0, 2.0, step=0.1, help="Rasio aspek minimum (lebar/tinggi).")
        aspect_ratio_max = st.slider("Aspect Ratio Max", 1.0, 10.0, 5.0, step=0.1, help="Rasio aspek maksimum (lebar/tinggi).")

        # Upload gambar
        uploaded_file = st.file_uploader("Upload Gambar Kendaraan", type=["jpg", "jpeg", "png"])

        # Button proses
        if st.button("Proses"):
            if uploaded_file is not None:
                with st.spinner("Memproses..."):
                    image = Image.open(uploaded_file)
                    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    result_image, _ = detect_license_plate(image_cv, min_area=min_area, max_area=max_area, aspect_ratio_min=aspect_ratio_min, aspect_ratio_max=aspect_ratio_max)
                    st.session_state['result_image'] = result_image
            else:
                st.error("Silakan upload gambar terlebih dahulu.")

        # Button download
        if 'result_image' in st.session_state:
            buf = image_to_bytes(st.session_state['result_image'])
            st.download_button(
                label="Download Hasil Gambar",
                data=buf,
                file_name="hasil_deteksi_plat.png",
                mime="image/png"
            )

    with col_right:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar Asli", use_container_width=True)
        if 'result_image' in st.session_state:
            st.image(cv2.cvtColor(st.session_state['result_image'], cv2.COLOR_BGR2RGB), caption="Hasil Deteksi", use_container_width=True)
