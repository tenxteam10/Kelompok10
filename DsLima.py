import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
import pandas as pd
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set page config
st.set_page_config(page_title="Dashboard Deteksi Plat", layout="wide")

# Custom CSS to mimic the template style and add responsiveness
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Ubuntu:wght@300;400;500;700&display=swap');

* {
    font-family: 'Ubuntu', sans-serif;
}

.sidebar .sidebar-content {
    background-color: #2a2185;
    color: white;
}

.main {
    background-color: #f5f5f5;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 7px 25px rgba(0, 0, 0, 0.08);
    margin: 10px;
    color: black;  /* Ensure text is black */
}

.card:hover {
    background: #87ceeb;  /* Changed to light blue */
    color: black;  /* Keep text black on hover */
}

.table {
    width: 100%;
    border-collapse: collapse;
}

.table th, .table td {
    padding: 10px;
    text-align: left;
    border-bottom: 1px solid #ddd;
}

.status {
    padding: 2px 4px;
    border-radius: 4px;
    color: white;
    font-weight: 500;
}

.delivered { background: #8de02c; }
.pending { background: #e9b10a; }
.return { background: #f00; }
.inProgress { background: #1795ce; }

/* Responsiveness */
@media (max-width: 768px) {
    .card {
        margin: 5px;
        padding: 15px;
    }
    .stColumns > div {
        flex: 1 1 100% !important;
        margin-bottom: 10px;
    }
    .stImage {
        width: 100% !important;
    }
    .table th, .table td {
        padding: 5px;
        font-size: 12px;
    }
}

.footer {
    text-align: center;
    padding: 10px;
    background-color: #87ceeb;  /* Changed to light blue */
    color: black;  /* Changed to black for better contrast */
    position: fixed;
    bottom: 0;
    width: 100%;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigasi")
menu = ["Beranda", "Langkah Deteksi", "Hasil", "Penjelasan", "Unduh Hasil"]
choice = st.sidebar.selectbox("Menu", menu)

# Initialize session state for results
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = []

# Default parameters (improved for better plate detection, with max_plates added and adjusted for better detection)
default_params = {
    'canny_min': 50,
    'canny_max': 200,
    'kernel_w': 20,
    'kernel_h': 8,
    'min_area': 1500,  # Lowered for smaller plates
    'min_aspect': 2.0,  # Lowered for more flexibility
    'max_aspect': 6.0,  # Increased for more flexibility
    'min_solidity': 0.6,  # Lowered for more tolerance
    'max_plates': 1  # New parameter to limit to top N plates
}

# Function to preprocess image for OCR
def preprocess_for_ocr(img):
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Resize if too small (minimum height 50 pixels)
    if gray.shape[0] < 50:
        scale = 50 / gray.shape[0]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply thresholding to get binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

# Main content
if choice == "Beranda":
    st.title("Dashboard Deteksi Plat")
    
    # Cards
    st.subheader("Statistik")
    total_images = len(st.session_state.detection_results)
    detected_plates = sum(1 for r in st.session_state.detection_results if r['plates_found'] > 0)
    success_rate = (detected_plates / total_images * 100) if total_images > 0 else 0
    efficiency = success_rate * 10  # Dummy calculation
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="card"><div class="numbers">{total_images}</div><div class="cardName">Gambar Diproses</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="card"><div class="numbers">{detected_plates}</div><div class="cardName">Plat Terdeteksi</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="card"><div class="numbers">{success_rate:.1f}%</div><div class="cardName">Tingkat Keberhasilan</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="card"><div class="numbers">{efficiency:.1f}</div><div class="cardName">Skor Efisiensi</div></div>', unsafe_allow_html=True)
    
    # Tables
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Deteksi Terbaru")
        if st.session_state.detection_results:
            # Buat DataFrame dengan header yang diinginkan
            table_data = {
                'Nama': [r['filename'] for r in st.session_state.detection_results[-5:]],
                'Jumlah': [f"{r['plates_found']} plat" for r in st.session_state.detection_results[-5:]],
                'Hasil': ["Terdeteksi" if r['plates_found'] > 0 else "Tidak Terdeteksi" for r in st.session_state.detection_results[-5:]],
                'Output': ["Selesai" for _ in st.session_state.detection_results[-5:]]
            }
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.write("Belum ada deteksi.")
    
    with col2:
        st.subheader("Gambar Terbaru")
        if st.session_state.detection_results:
            for r in st.session_state.detection_results[-2:]:  # Last 2
                st.image(r['original_image'], caption=r['filename'], width=100)
        else:
            st.image("https://via.placeholder.com/100", caption="Gambar Contoh")
    
    # Footer
    st.markdown('<div class="footer">xteam 2025 image processing</div>', unsafe_allow_html=True)

elif choice == "Langkah Deteksi":
    st.title("Langkah Deteksi")
    st.write("Unggah gambar untuk melihat langkah-langkah individu: Deteksi Tepi, Transformasi Morfologi, dan Penyaringan Kontur. Bandingkan sebelum (pengaturan default) dan sesudah (pengaturan kustom).")
    
    # Sliders for custom parameters
    st.subheader("Sesuaikan Parameter Kustom")
    col1, col2, col3 = st.columns(3)
    with col1:
        canny_min = st.slider("Ambang Batas Canny Minimum", 0, 255, 50)
        canny_max = st.slider("Ambang Batas Canny Maksimum", 0, 255, 200)
    with col2:
        kernel_w = st.slider("Lebar Kernel", 1, 50, 20)
        kernel_h = st.slider("Tinggi Kernel", 1, 50, 8)
    with col3:
        min_area = st.slider("Luas Kontur Minimum", 100, 10000, 1500)  # Adjusted default
        min_aspect = st.slider("Rasio Aspek Minimum", 1.0, 10.0, 2.0)  # Adjusted
        max_aspect = st.slider("Rasio Aspek Maksimum", 1.0, 10.0, 6.0)  # Adjusted
        min_solidity = st.slider("Soliditas Minimum", 0.0, 1.0, 0.6)  # Adjusted
        max_plates = st.slider("Maksimal Plat Terdeteksi", 1, 10, 1)  # New slider
    
    custom_params = {
        'canny_min': canny_min,
        'canny_max': canny_max,
        'kernel_w': kernel_w,
        'kernel_h': kernel_h,
        'min_area': min_area,
        'min_aspect': min_aspect,
        'max_aspect': max_aspect,
        'min_solidity': min_solidity,
        'max_plates': max_plates
    }
    
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"], key="steps")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Function to process image with given params
        def process_steps(img, params):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, params['canny_min'], params['canny_max'])
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (params['kernel_w'], params['kernel_h']))
            morph = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours and sort by area descending
            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > params['min_area']:
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area > 0 else 0
                    if solidity > params['min_solidity']:
                        rect = cv2.minAreaRect(contour)
                        # Fix aspect ratio calculation: ensure it's always >=1
                        w, h = rect[1]
                        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                        if params['min_aspect'] < aspect_ratio < params['max_aspect']:
                            filtered_contours.append((contour, area, rect))
            
            # Sort by area descending and take top max_plates
            filtered_contours.sort(key=lambda x: x[1], reverse=True)
            top_contours = filtered_contours[:params['max_plates']]
            
            img_with_boxes = img.copy()
            cropped_plates = []
            plate_texts = []  # List untuk menyimpan teks OCR
            for i, (contour, _, rect) in enumerate(top_contours):
                # Draw rotated rectangle
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(img_with_boxes, [box], 0, (0, 255, 0), 2)
                # Add green text overlay above the plate
                text_x = int(np.min(box[:, 0]))
                text_y = int(np.min(box[:, 1])) - 10
                cv2.putText(img_with_boxes, f"Plat {i+1}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Crop using bounding box with padding
                x, y, w, h = cv2.boundingRect(contour)
                padding = 30  # Increased padding
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img.shape[1] - x, w + 2 * padding)
                h = min(img.shape[0] - y, h + 2 * padding)
                cropped = img[y:y+h, x:x+w]
                cropped_plates.append(cropped)
                
                # Preprocess and OCR pada cropped image
                try:
                    preprocessed = preprocess_for_ocr(cropped)
                    text = pytesseract.image_to_string(preprocessed, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    cleaned_text = ''.join(c for c in text if c.isalnum()).strip()  # Clean non-alphanumeric
                    plate_texts.append(cleaned_text if cleaned_text else "Tidak Ditemukan")
                except Exception as e:
                    st.warning(f"OCR gagal untuk Plat {i+1}: {str(e)}. Pastikan Tesseract terinstal dengan benar.")
                    plate_texts.append("OCR Gagal")
            
            return edged, morph, img_with_boxes, cropped_plates, plate_texts
        
        # Process with default and custom
        edged_default, morph_default, detected_default, crops_default, texts_default = process_steps(img_cv, default_params)
        edged_custom, morph_custom, detected_custom, crops_custom, texts_custom = process_steps(img_cv, custom_params)
        
        # Display comparison
        st.subheader("Perbandingan: Pengaturan Default vs. Pengaturan Kustom")
        
        # Edge Detection
        st.subheader("1. Deteksi Tepi")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Default**")
            st.image(cv2.cvtColor(edged_default, cv2.COLOR_GRAY2RGB), caption="Tepi Default", use_container_width=True)
        with col2:
            st.write("**Kustom**")
            st.image(cv2.cvtColor(edged_custom, cv2.COLOR_GRAY2RGB), caption="Tepi Kustom", use_container_width=True)
        
        # Morphological Transformation
        st.subheader("2. Transformasi Morfologi")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Default**")
            st.image(cv2.cvtColor(morph_default, cv2.COLOR_GRAY2RGB), caption="Morfo Default", use_container_width=True)
        with col2:
            st.write("**Kustom**")
            st.image(cv2.cvtColor(morph_custom, cv2.COLOR_GRAY2RGB), caption="Morfo Kustom", use_container_width=True)
        
        # Contour Filtering
        st.subheader("3. Penyaringan Kontur & Deteksi")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Default**")
            detected_default_rgb = cv2.cvtColor(detected_default, cv2.COLOR_BGR2RGB)
            st.image(detected_default_rgb, caption="Deteksi Default", use_container_width=True)
            if crops_default:
                st.write("Plat yang Dipotong (Teks OCR):")
                for i, text in enumerate(texts_default):
                    st.write(f"Plat {i+1}: {text}")
            else:
                st.write("Tidak ada plat terdeteksi. Coba sesuaikan parameter atau unggah gambar yang berbeda.")
        with col2:
            st.write("**Kustom**")
            detected_custom_rgb = cv2.cvtColor(detected_custom, cv2.COLOR_BGR2RGB)
            st.image(detected_custom_rgb, caption="Deteksi Kustom", use_container_width=True)
            if crops_custom:
                st.write("Plat yang Dipotong (Teks OCR):")
                for i, text in enumerate(texts_custom):
                    st.write(f"Plat {i+1}: {text}")
            else:
                st.write("Tidak ada plat terdeteksi. Coba sesuaikan parameter atau unggah gambar yang berbeda.")
        
        # Save result to session state (using custom)
        result = {
            'filename': uploaded_file.name,
            'original_image': image,
            'detected_image': detected_custom_rgb,
            'plates_found': len(crops_custom),
            'cropped_plates': [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in crops_custom],
            'plate_texts': texts_custom
        }
        st.session_state.detection_results.append(result)
    
    # Footer
    st.markdown('<div class="footer">xteam 2025 image processing</div>', unsafe_allow_html=True)

elif choice == "Unduh Hasil":
    st.title("Unduh Hasil")
    if st.session_state.detection_results:
        st.write("Pilih hasil untuk diunduh:")
        result_options = [f"{r['filename']} - {r['plates_found']} plat" for r in st.session_state.detection_results]
        selected_result = st.selectbox("Pilih hasil", result_options)
        
        if selected_result:
            index = result_options.index(selected_result)
            result = st.session_state.detection_results[index]
            
            # Create ZIP file
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                # Add detected image
                detected_img = Image.fromarray(result['detected_image'])
                img_buffer = io.BytesIO()
                detected_img.save(img_buffer, format='PNG')
                zip_file.writestr(f"{result['filename']}_terdeteksi.png", img_buffer.getvalue())
                
                # Add cropped plates
                for i, crop in enumerate(result['cropped_plates']):
                    crop_img = Image.fromarray(crop)
                    crop_buffer = io.BytesIO()
                    crop_img.save(crop_buffer, format='PNG')
                    zip_file.writestr(f"{result['filename']}_plat_{i+1}.png", crop_buffer.getvalue())
            
            zip_buffer.seek(0)
            st.download_button(
                label="Unduh ZIP",
                data=zip_buffer,
                file_name=f"{result['filename']}_hasil.zip",
                mime="application/zip"
            )
    else:
        st.write("Tidak ada hasil untuk diunduh")