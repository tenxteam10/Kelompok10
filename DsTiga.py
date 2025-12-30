import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from streamlit_option_menu import option_menu

# =============================
# FUNGSI UTAMA
# =============================

def detect_license_plate(image, canny_min=100, canny_max=200, kernel_size=5,
                         min_area=500, max_area=50000, aspect_ratio_min=2.0, aspect_ratio_max=5.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_min, canny_max)
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if aspect_ratio_min < aspect_ratio < aspect_ratio_max:
                filtered_contours.append(cnt)

    result_image = image.copy()
    cropped_images = []
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped = image[y:y+h, x:x+w]
        cropped_images.append(cropped)

    return result_image, filtered_contours, cropped_images, edges, morph


def image_to_bytes(image):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    buf.seek(0)
    return buf


# =============================
# KONFIGURASI HALAMAN
# =============================

st.set_page_config(page_title="Deteksi Plat Nomor", layout="wide", page_icon="🚗")


# =============================
# CSS UNTUK STYLING
# =============================

st.markdown("""
    <style>
        .main-header {
            text-align: center;
            font-size: 2.5em;
            color: #333;
            margin-bottom: 20px;
        }
        .sidebar .sidebar-content {
            background-color: #f0f8ff;
            padding: 20px;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .image-container {
            display: flex;
            justify-content: space-around;
            align-items: center;
            margin-top: 20px;
        }
        .coming-soon {
            text-align: center;
            font-size: 1.5em;
            color: #666;
            margin-top: 50px;
        }
        .step-header {
            font-size: 1.5em;
            color: #007bff;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)


# =============================
# MENU NAVIGASI ATAS
# =============================

col_menu = st.columns([1, 5, 1])[1]

with col_menu:
    selected = option_menu(
        menu_title=None,
        options=["Home", "Edge Detection", "Morfologi", "Contour Filter"],
        orientation="horizontal",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f8f9fa"},
            "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "color": "black", "--hover-color": "#e9ecef"},
            "nav-link-selected": {"background-color": "#007bff", "color": "white"},
        }
    )


# =============================
# HOME PAGE
# =============================

if selected == "Home":
    st.markdown("<h1 class='main-header'>Selamat Datang di Aplikasi Deteksi Plat Nomor!</h1>", unsafe_allow_html=True)
    st.write("""
    Aplikasi ini dirancang untuk mendeteksi dan mengekstraksi area plat nomor motor/mobil secara otomatis 
    menggunakan kombinasi metode edge detection, transformasi morfologi, dan contour filtering tanpa OCR. 
    Aplikasi akan memisahkan area plat dari latar belakang yang kompleks, kemudian menampilkan bounding box 
    dan hasil cropping secara interaktif.
    
    **Langkah-langkah penggunaan:**
    1. Upload gambar kendaraan.
    2. Sesuaikan parameter di setiap tahap (Edge Detection, Morfologi, Contour Filter).
    3. Lihat hasil bounding box dan cropping.
    4. Download hasil jika diperlukan.
    """)
    
    # Upload gambar di atas
    uploaded_file = st.file_uploader("Upload Gambar Kendaraan (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], key="upload_home")
    if uploaded_file:
        st.session_state["uploaded_image"] = uploaded_file
        st.success("Gambar berhasil diupload! Lanjutkan ke tahap berikutnya.")


# =============================
# EDGE DETECTION
# =============================

elif selected == "Edge Detection":
    st.markdown("<h1 class='main-header'>Tahap 1: Edge Detection</h1>", unsafe_allow_html=True)
    st.write("Tahap ini mendeteksi tepi pada gambar menggunakan algoritma Canny.")
    
    # Upload gambar di atas
    uploaded_file = st.file_uploader("Upload Gambar Kendaraan (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], key="upload_edge")
    if uploaded_file:
        st.session_state["uploaded_image"] = uploaded_file
    
    # Sidebar untuk parameter
    with st.sidebar:
        st.header("Parameter Edge Detection")
        canny_min = st.slider("Canny Min Threshold", 0, 255, 50, help="Nilai minimum untuk deteksi tepi.")
        canny_max = st.slider("Canny Max Threshold", 0, 255, 150, help="Nilai maksimum untuk deteksi tepi.")
        
        if st.button("Proses Edge Detection", key="process_edge"):
            if "uploaded_image" in st.session_state:
                image = Image.open(st.session_state["uploaded_image"])
                image_np = np.array(image)
                image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                result, contours, cropped, edges, morph = detect_license_plate(image_cv, canny_min, canny_max)
                st.session_state["original_image"] = image_np
                st.session_state["edges"] = edges
                st.session_state["morph"] = morph
                st.session_state["result_image"] = result
                st.session_state["cropped_images"] = cropped
                st.success("Edge Detection berhasil diproses!")
            else:
                st.error("Silakan upload gambar terlebih dahulu.")
    
    # Panel utama
    if "original_image" in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Gambar Asli")
            st.image(st.session_state["original_image"], use_container_width=True)
        
        with col2:
            if "edges" in st.session_state:
                st.subheader("Hasil Edge Detection")
                st.image(st.session_state["edges"], use_container_width=True)
    else:
        st.info("Upload gambar dan klik 'Proses Edge Detection' untuk melihat hasil.")


# =============================
# MORFOLOGI
# =============================

elif selected == "Morfologi":
    st.markdown("<h1 class='main-header'>Tahap 2: Transformasi Morfologi</h1>", unsafe_allow_html=True)
    st.write("Tahap ini menerapkan operasi morfologi (closing) untuk menutup celah pada tepi.")
    
    # Upload gambar di atas
    uploaded_file = st.file_uploader("Upload Gambar Kendaraan (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], key="upload_morph")
    if uploaded_file:
        st.session_state["uploaded_image"] = uploaded_file
    
    # Sidebar untuk parameter
    with st.sidebar:
        st.header("Parameter Morfologi")
        kernel_size = st.slider("Kernel Size", 1, 15, 5, step=2, help="Ukuran kernel untuk operasi morfologi.")
        
        if st.button("Proses Morfologi", key="process_morph"):
            if "edges" in st.session_state:
                edges = st.session_state["edges"]
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
                st.session_state["morph"] = morph
                st.success("Morfologi berhasil diproses!")
            else:
                st.error("Lakukan Edge Detection terlebih dahulu.")
    
    # Panel utama
    if "morph" in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            if "edges" in st.session_state:
                st.subheader("Hasil Edge Detection")
                st.image(st.session_state["edges"], use_container_width=True)
        
        with col2:
            st.subheader("Hasil Morfologi")
            st.image(st.session_state["morph"], use_container_width=True)
    else:
        st.info("Lakukan Edge Detection dan klik 'Proses Morfologi'.")


# =============================
# CONTOUR FILTERING
# =============================

elif selected == "Contour Filter":
    st.markdown("<h1 class='main-header'>Tahap 3: Contour Filtering & Cropping</h1>", unsafe_allow_html=True)
    st.write("Tahap ini memfilter kontur berdasarkan area dan rasio aspek, menampilkan bounding box, dan cropping area plat.")
    
    # Upload gambar di atas
    uploaded_file = st.file_uploader("Upload Gambar Kendaraan (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], key="upload_contour")
    if uploaded_file:
        st.session_state["uploaded_image"] = uploaded_file
    
    # Sidebar untuk parameter
    with st.sidebar:
        st.header("Parameter Contour Filter")
        min_area = st.slider("Min Area", 100, 10000, 500, help="Area minimum kontur.")
        max_area = st.slider("Max Area", 1000, 100000, 50000, help="Area maksimum kontur.")
        aspect_ratio_min = st.slider("Aspect Ratio Min", 1.0, 10.0, 2.0, help="Rasio aspek minimum.")
        aspect_ratio_max = st.slider("Aspect Ratio Max", 1.0, 10.0, 5.0, help="Rasio aspek maksimum.")
        
        if st.button("Proses Contour Filter", key="process_contour"):
            if "morph" in st.session_state and "original_image" in st.session_state:
                image_cv = cv2.cvtColor(st.session_state["original_image"], cv2.COLOR_RGB2BGR)
                result, contours, cropped, _, _ = detect_license_plate(image_cv, min_area=min_area, max_area=max_area, aspect_ratio_min=aspect_ratio_min, aspect_ratio_max=aspect_ratio_max)
                st.session_state["result_image"] = result
                st.session_state["cropped_images"] = cropped
                st.success("Contour Filtering berhasil diproses!")
            else:
                st.error("Lakukan tahap sebelumnya terlebih dahulu.")
        
        if "result_image" in st.session_state:
            buf = image_to_bytes(st.session_state["result_image"])
            st.download_button(
                label="Download Bounding Box",
                data=buf,
                file_name="bounding_box.png",
                mime="image/png",
                key="download_bbox"
            )
            if st.session_state["cropped_images"]:
                for i, crop in enumerate(st.session_state["cropped_images"]):
                    buf_crop = image_to_bytes(crop)
                    st.download_button(
                        label=f"Download Cropped {i+1}",
                        data=buf_crop,
                        file_name=f"cropped_{i+1}.png",
                        mime="image/png",
                        key=f"download_crop_{i}"
                    )
    
    # Panel utama
    if "result_image" in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Gambar dengan Bounding Box")
            st.image(cv2.cvtColor(st.session_state["result_image"], cv2.COLOR_BGR2RGB), use_container_width=True)
        
        with col2:
            if st.session_state["cropped_images"]:
                st.subheader("Hasil Cropping")
                for i, crop in enumerate(st.session_state["cropped_images"]):
                    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True, caption=f"Cropped Area {i+1}")
            else:
                st.info("Tidak ada area plat yang terdeteksi.")
    else:
        st.info("Lakukan tahap sebelumnya dan klik 'Proses Contour Filter'.")
