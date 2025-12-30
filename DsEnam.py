import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import pandas as pd
import pytesseract
import re

# ================= KONFIG =================
st.set_page_config(page_title="Deteksi Plat Nomor", layout="wide")

try:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    OCR_READY = True
except:
    OCR_READY = False

# ================= SIDEBAR =================
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Menu", ["Deteksi", "Hasil", "Parameter", "Penjelasan"])

# ================= SESSION =================
if "results" not in st.session_state:
    st.session_state.results = []

if "canny_min" not in st.session_state:
    st.session_state.canny_min = 50
if "canny_max" not in st.session_state:
    st.session_state.canny_max = 200
if "kernel_w" not in st.session_state:
    st.session_state.kernel_w = 20
if "kernel_h" not in st.session_state:
    st.session_state.kernel_h = 8
if "min_area" not in st.session_state:
    st.session_state.min_area = 1500
if "min_ratio" not in st.session_state:
    st.session_state.min_ratio = 2.0
if "max_ratio" not in st.session_state:
    st.session_state.max_ratio = 6.0

# ================= FUNGSI LOKASI =================

def get_region(text):
    if not text:
        return "teks plat kosong"

    # Normalisasi keras
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)

    # Koreksi OCR umum
    text = text.replace("8E", "BE")

    if not text.startswith("BE"):
        return "plat ini bukan dari lampung"

    # Cari huruf pertama SETELAH angka
    match = re.search(r"BE\d+([A-Z])", text)

    if not match:
        return "kode wilayah Lampung tidak dikenali"

    code = match.group(1)

    wilayah = {
        "ABC": "Kota Bandar Lampung",
        "EF": "Kabupaten Lampung Selatan",
        "GH": "Kabupaten Lampung Tengah",
        "JK": "Kabupaten Lampung Utara",
        "LM": "Kabupaten Tanggamus",
        "NP": "Kabupaten Tulang Bawang",
        "QR": "Kabupaten Lampung Timur",
        "ST": "Kabupaten Way Kanan",
        "UV": "Kabupaten Pesawaran",
        "WX": "Kabupaten Mesuji",
        "YZ": "Kabupaten Pesisir Barat & Tulang Bawang Barat"
    }

    for k, v in wilayah.items():
        if code in k:
            return v

    return "kode wilayah Lampung tidak dikenali"

# ================= DETEKSI =================
def detect_plate(img):
    h_img, w_img = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edge = cv2.Canny(blur, st.session_state.canny_min, st.session_state.canny_max)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (st.session_state.kernel_w, st.session_state.kernel_h))
    morph = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    box_img = img.copy()
    plates, texts, locations = [], [], []

    for c in contours:
        if cv2.contourArea(c) > st.session_state.min_area:
            x,y,w,h = cv2.boundingRect(c)
            ratio = w/h if h != 0 else 0

            if st.session_state.min_ratio < ratio < st.session_state.max_ratio:
                cv2.rectangle(box_img, (x,y), (x+w,y+h), (0,255,0), 2)
                crop = img[y:y+h, x:x+w]

                plates.append(crop)

                if OCR_READY:
                    txt = pytesseract.image_to_string(
                        crop,
                        config="--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    )
                    text = "".join(t for t in txt if t.isalnum())
                    texts.append(text)
                    locations.append(get_region(text))
                else:
                    texts.append("-")
                    locations.append("plat ini bukan dari lampung")

    return box_img, plates, edge, morph, texts, locations

# ================= MENU DETEKSI =================
if menu == "Deteksi":
    st.title("Deteksi Plat Nomor")

    files = st.file_uploader("Upload gambar", type=["jpg","png","jpeg"], accept_multiple_files=True)

    if st.button("Jalankan Deteksi") and files:
        st.session_state.results = []

        for f in files:
            img = np.array(Image.open(f).convert("RGB"))
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            box, plates, edge, morph, texts, locations = detect_plate(bgr)

            st.session_state.results.append({
                "name": f.name,
                "box": cv2.cvtColor(box, cv2.COLOR_BGR2RGB),
                "edge": edge,
                "morph": morph,
                "plates": plates,
                "texts": texts,
                "locations": locations
            })

        st.success("Deteksi selesai")

# ================= MENU HASIL =================
elif menu == "Hasil":
    st.title("Hasil Deteksi")

    if not st.session_state.results:
        st.info("Belum ada hasil")
    else:
        for r in st.session_state.results:
            st.subheader(r["name"])
            c1, c2, c3 = st.columns(3)
            c1.image(r["box"], caption="Bounding Box", use_container_width=True)
            c2.image(r["edge"], caption="Edge", clamp=True)
            c3.image(r["morph"], caption="Morph", clamp=True)

        st.markdown("## Tabel Lokasi Plat")

        rows = []
        for r in st.session_state.results:
            for i in range(len(r["plates"])):
                rows.append({
                    "Nama Gambar": r["name"],
                    "Plat Ke-": i+1,
                    "Lokasi Plat": r["locations"][i],
                    "Hasil OCR": r["texts"][i]
                })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

# ================= MENU PARAMETER =================
elif menu == "Parameter":
    st.title("Pengaturan Parameter")

    st.markdown("### Parameter Deteksi")
    st.session_state.canny_min = st.slider("Canny Min", 0, 150, st.session_state.canny_min)
    st.session_state.canny_max = st.slider("Canny Max", 150, 300, st.session_state.canny_max)
    st.session_state.kernel_w = st.slider("Kernel Width", 5, 30, st.session_state.kernel_w)
    st.session_state.kernel_h = st.slider("Kernel Height", 3, 15, st.session_state.kernel_h)
    st.session_state.min_area = st.slider("Min Area", 500, 5000, st.session_state.min_area)
    st.session_state.min_ratio = st.slider("Min Ratio", 1.0, 4.0, st.session_state.min_ratio)
    st.session_state.max_ratio = st.slider("Max Ratio", 4.0, 8.0, st.session_state.max_ratio)

    st.success("Parameter telah disimpan")

# ================= PENJELASAN =================
else:
    st.markdown("""
    Sistem menentukan **lokasi plat nomor** berdasarkan kode daerah
    yang terdeteksi dari hasil OCR plat nomor motor.

    Daerah ditentukan berdasarkan kode plat:
    - Kota Bandar Lampung: BE A, B, C
    - Kabupaten Lampung Selatan: BE E, F
    - Kabupaten Lampung Tengah: BE G, H
    - Kabupaten Lampung Utara: BE J, K
    - Kabupaten Tanggamus: BE L, M
    - Kabupaten Tulang Bawang: BE N, P
    - Kabupaten Lampung Timur: BE Q, R
    - Kabupaten Way Kanan: BE S, T
    - Kabupaten Pesawaran: BE U, V
    - Kabupaten Mesuji: BE W, X
    - Kabupaten Pesisir Barat & Tulang Bawang Barat: BE Y, Z

    Jika kode plat tidak cocok dengan daftar di atas atau tidak dimulai dengan "BE",
    maka akan ditampilkan "plat ini bukan dari lampung".

    Informasi lokasi ini ditampilkan dalam tabel
    agar jelas plat berasal dari daerah mana berdasarkan kode plat.
    """)