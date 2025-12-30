import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from streamlit_option_menu import option_menu  # Tambahkan import ini

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

# Streamlit App
st.set_page_config(page_title="Deteksi Plat Nomor", layout="wide")

# Sidebar dengan navigasi menu menggunakan streamlit-option-menu
st.sidebar.title("Navigasi Menu Metode")

# Gunakan option_menu untuk memilih metode
selected = option_menu(
    menu_title=None,
    options=["Edge Detection", "Transformasi Morfologi", "Contour Filtering"],
    icons=["edge", "morph", "filter"],  # Ikon opsional, bisa dihapus jika tidak diinginkan
    menu_icon="cast",
    default_index=0,
    orientation="vertical",
)

# Parameter default
canny_min = 100
canny_max = 200
kernel_size = 5
min_area = 500
max_area = 50000
aspect_ratio_min = 2.0
aspect_ratio_max = 5.0

# Tampilkan parameter berdasarkan pilihan menu
if selected == "Edge Detection":
    st.sidebar.markdown("Atur parameter untuk deteksi tepi menggunakan algoritma Canny.")
    canny_min = st.sidebar.slider("Canny Min Threshold", 0, 255, 100, help="Threshold minimum untuk deteksi tepi.")
    canny_max = st.sidebar.slider("Canny Max Threshold", 0, 255, 200, help="Threshold maksimum untuk deteksi tepi.")

elif selected == "Transformasi Morfologi":
    st.sidebar.markdown("Atur parameter untuk operasi morfologi (closing) untuk mengisi celah pada tepi.")
    kernel_size = st.sidebar.slider("Kernel Size", 1, 15, 5, step=2, help="Ukuran kernel untuk operasi morfologi.")

elif selected == "Contour Filtering":
    st.sidebar.markdown("Atur parameter untuk filtering kontur berdasarkan area dan rasio aspek.")
    min_area = st.sidebar.slider("Min Area", 100, 10000, 500, help="Area minimum kontur yang dianggap sebagai plat.")
    max_area = st.sidebar.slider("Max Area", 1000, 100000, 50000, help="Area maksimum kontur yang dianggap sebagai plat.")
    aspect_ratio_min = st.sidebar.slider("Aspect Ratio Min", 1.0, 10.0, 2.0, step=0.1, help="Rasio aspek minimum (lebar/tinggi).")
    aspect_ratio_max = st.sidebar.slider("Aspect Ratio Max", 1.0, 10.0, 5.0, step=0.1, help="Rasio aspek maksimum (lebar/tinggi).")

st.sidebar.markdown("---")

# Tombol download (akan aktif setelah proses)
download_placeholder = st.sidebar.empty()

# Navigasi utama
st.title("Deteksi dan Ekstraksi Area Plat Nomor Kendaraan")
st.markdown("Aplikasi ini menggunakan kombinasi metode **Edge Detection**, **Transformasi Morfologi**, dan **Contour Filtering** untuk mendeteksi plat nomor tanpa OCR.")

# Upload gambar
uploaded_file = st.file_uploader("Upload Gambar Kendaraan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca gambar
    image = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Tampilkan gambar asli
    st.subheader("Gambar Asli")
    st.image(image, use_container_width=True)
    
    # Tombol proses
    if st.button("Proses Deteksi Plat Nomor"):
        with st.spinner("Memproses..."):
            # Deteksi plat nomor
            result_image, cropped_plates = detect_license_plate(
                image_cv, canny_min, canny_max, kernel_size, min_area, max_area, aspect_ratio_min, aspect_ratio_max
            )
        
        # Tampilkan hasil dengan bounding box
        st.subheader("Hasil Deteksi (dengan Bounding Box)")
        st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # Tampilkan cropped plat nomor
        if cropped_plates:
            st.subheader("Plat Nomor yang Diekstraksi")
            cols = st.columns(len(cropped_plates))
            for i, cropped in enumerate(cropped_plates):
                with cols[i]:
                    st.image(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), caption=f"Plat {i+1}", use_container_width=True)
                    
                    # Tombol download untuk setiap cropped
                    buf = image_to_bytes(cropped)
                    st.download_button(
                        label=f"Download Plat {i+1}",
                        data=buf,
                        file_name=f"plat_{i+1}.png",
                        mime="image/png"
                    )
        else:
            st.warning("Tidak ada plat nomor yang terdeteksi. Coba sesuaikan parameter di menu navigasi.")
        
        # Tombol download untuk gambar hasil deteksi
        result_buf = image_to_bytes(result_image)
        download_placeholder.download_button(
            label="Download Gambar Hasil Deteksi",
            data=result_buf,
            file_name="hasil_deteksi.png",
            mime="image/png"
        )
else:
    st.info("Silakan upload gambar untuk memulai.")
