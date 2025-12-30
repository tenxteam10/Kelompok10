import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import pytesseract
import re
import uuid
import sqlite3
import hashlib

# ================= OCR PATH (WINDOWS) =================
# SESUAIKAN JIKA LOKASI BERBEDA
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ================= KONFIG =================
st.set_page_config(page_title="Deteksi Plat Nomor", layout="wide")

# ================= DATABASE =================
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")
conn.commit()

# ================= AUTH =================
def hash_password(p): 
    return hashlib.sha256(p.encode()).hexdigest()

def register_user(u, p):
    try:
        cursor.execute("INSERT INTO users (username,password) VALUES (?,?)", (u, hash_password(p)))
        conn.commit()
        return True
    except:
        return False

def login_user(u, p):
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (u, hash_password(p)))
    return cursor.fetchone()

# ================= SESSION =================
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "username" not in st.session_state: st.session_state.username = ""
if "user_id" not in st.session_state: st.session_state.user_id = str(uuid.uuid4())
if "results" not in st.session_state: st.session_state.results = []

# ================= LOGIN =================
if not st.session_state.logged_in:
    st.title("Login Sistem Deteksi Plat Nomor")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if login_user(u, p):
                st.session_state.logged_in = True
                st.session_state.username = u
                st.rerun()
            else:
                st.error("Username / Password salah")
    
    with tab2:
        ru = st.text_input("Username Baru")
        rp = st.text_input("Password Baru", type="password")

        if st.button("Register"):
             if register_user(ru, rp):
                 st.success("Registrasi berhasil, silakan login")
             else:
                 st.error("Username sudah terdaftar")


    st.stop()

# ================= OCR CHECK =================
try:
    pytesseract.get_tesseract_version()
    OCR_READY = True
except:
    OCR_READY = False

# ================= SIDEBAR =================
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Menu", ["Deteksi", "Hasil", "Parameter", "Penjelasan"])
st.sidebar.write("User:", st.session_state.username)
st.sidebar.write("OCR Ready:", OCR_READY)
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

# ================= PARAMETER =================
defaults = {
    "canny_min": 50, "canny_max": 200,
    "kernel_w": 20, "kernel_h": 8,
    "min_area": 1500, "min_ratio": 2.0, "max_ratio": 6.0
}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# ================= LOKASI LAMPUNG =================
def get_region(text):
    if not text: return "teks plat kosong"
    text = re.sub(r"[^A-Z0-9]", "", text.upper()).replace("8E","BE")
    if not text.startswith("BE"): return "plat ini bukan dari lampung"
    m = re.search(r"BE\d+([A-Z])", text)
    if not m: return "kode wilayah Lampung tidak dikenali"
    code = m.group(1)
    wilayah = {
        "ABC":"Kota Bandar Lampung","EF":"Kabupaten Lampung Selatan","GH":"Kabupaten Lampung Tengah",
        "JK":"Kabupaten Lampung Utara","LM":"Kabupaten Tanggamus","NP":"Kabupaten Tulang Bawang",
        "QR":"Kabupaten Lampung Timur","ST":"Kabupaten Way Kanan","UV":"Kabupaten Pesawaran",
        "WX":"Kabupaten Mesuji","YZ":"Kabupaten Pesisir Barat & Tulang Bawang Barat"
    }
    for k,v in wilayah.items():
        if code in k: return v
    return "kode wilayah Lampung tidak dikenali"

# ================= DETEKSI =================
def detect_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edge = cv2.Canny(blur, st.session_state.canny_min, st.session_state.canny_max)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(st.session_state.kernel_w, st.session_state.kernel_h))
    morph = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)

    cnts,_ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box_img = img.copy()
    plates,texts,locs = [],[],[]

    for c in cnts:
        if cv2.contourArea(c) > st.session_state.min_area:
            x,y,w,h = cv2.boundingRect(c)
            r = w/h if h else 0
            if st.session_state.min_ratio < r < st.session_state.max_ratio:
                cv2.rectangle(box_img,(x,y),(x+w,y+h),(0,255,0),2)
                crop = img[y:y+h, x:x+w]
                plates.append(crop)

                if OCR_READY:
                    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    _,th = cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    txt = pytesseract.image_to_string(
                        th, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    )
                    text = "".join(ch for ch in txt if ch.isalnum())
                    texts.append(text)
                    locs.append(get_region(text))
                else:
                    texts.append("-")
                    locs.append("OCR tidak tersedia")

    return box_img, plates, edge, morph, texts, locs

# ================= MENU =================
if menu == "Deteksi":
    st.title("Deteksi Plat Nomor")
    files = st.file_uploader("Upload gambar", type=["jpg","png","jpeg"], accept_multiple_files=True)
    if st.button("Jalankan Deteksi") and files:
        st.session_state.results = []
        for f in files:
            img = np.array(Image.open(f).convert("RGB"))
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            box,plates,edge,morph,texts,locs = detect_plate(bgr)
            st.session_state.results.append({
                "name":f.name,"box":cv2.cvtColor(box,cv2.COLOR_BGR2RGB),
                "edge":edge,"morph":morph,"plates":plates,"texts":texts,"locations":locs
            })
        st.success("Deteksi selesai")

elif menu == "Hasil":
    st.title("Hasil Deteksi")
    if not st.session_state.results:
        st.info("Belum ada hasil")
    else:
        for r in st.session_state.results:
            st.subheader(r["name"])
            c1,c2,c3 = st.columns(3)
            c1.image(r["box"],caption="Bounding Box",use_container_width=True)
            c2.image(r["edge"],caption="Edge",clamp=True)
            c3.image(r["morph"],caption="Morph",clamp=True)

        rows=[]
        for r in st.session_state.results:
            for i in range(len(r["plates"])):
                rows.append({
                    "Nama Gambar":r["name"],"Plat Ke-":i+1,
                    "Hasil OCR":r["texts"][i],"Lokasi Plat":r["locations"][i]
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

elif menu == "Parameter":
    st.title("Pengaturan Parameter")
    st.session_state.canny_min = st.slider("Canny Min",0,150,st.session_state.canny_min)
    st.session_state.canny_max = st.slider("Canny Max",150,300,st.session_state.canny_max)
    st.session_state.kernel_w = st.slider("Kernel Width",5,30,st.session_state.kernel_w)
    st.session_state.kernel_h = st.slider("Kernel Height",3,15,st.session_state.kernel_h)
    st.session_state.min_area = st.slider("Min Area",500,5000,st.session_state.min_area)
    st.session_state.min_ratio = st.slider("Min Ratio",1.0,4.0,st.session_state.min_ratio)
    st.session_state.max_ratio = st.slider("Max Ratio",4.0,8.0,st.session_state.max_ratio)

else:
    st.markdown("""
    Aplikasi deteksi plat nomor berbasis image processing dan OCR.
    Mendukung autentikasi pengguna (SQLite) dan multi-user session.
    """)
