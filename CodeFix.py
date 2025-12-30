import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import pytesseract
import re
import sqlite3
import hashlib
import uuid

# ================= OCR PATH =================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Sistem Deteksi Plat Nomor", page_icon="🚗", layout="wide")

# ================= DATABASE =================
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")
conn.commit()

# ================= FUNCTIONS =================
def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def register_user(username, password):
    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hash_pw(password))
        )
        conn.commit()
        return True
    except:
        return False

def login_user(username, password):
    cursor.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, hash_pw(password))
    )
    return cursor.fetchone()

def wilayah(text):
    if not text: return "Tidak dikenali"
    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    peta = {
        "BE":"Lampung","D":"Bandung","E":"Cirebon","B":"Jakarta & Sekitar",
        "F":"Bogor","BG":"Palembang","L":"Surabaya","H":"Semarang","AB":"Yogyakarta"
    }
    for k,v in peta.items():
        if text.startswith(k): return v
    return "Wilayah tidak terdaftar"

def detect(img, cmin, cmax, kw, kh, min_area, min_r, max_r):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edge = cv2.Canny(blur,cmin,cmax)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kw, kh))
    morph = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)

    cnts,_ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box = img.copy()
    plates,texts,locs = [],[],[]
    for c in cnts:
        if cv2.contourArea(c) > min_area:
            x,y,w,h = cv2.boundingRect(c)
            r = w/h if h else 0
            if min_r < r < max_r:
                cv2.rectangle(box,(x,y),(x+w,y+h),(0,255,0),2)
                crop = img[y:y+h, x:x+w]
                plates.append(crop)
                txt = pytesseract.image_to_string(crop, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                clean = "".join(t for t in txt if t.isalnum())
                texts.append(clean)
                locs.append(wilayah(clean))
    return box,edge,morph,plates,texts,locs

# ================= SESSION =================
if "login" not in st.session_state: st.session_state.login=False
if "user" not in st.session_state: st.session_state.user=""
if "results" not in st.session_state: st.session_state.results=[]
if "uid" not in st.session_state: st.session_state.uid=str(uuid.uuid4())
if "cmin" not in st.session_state:
    st.session_state.cmin=50
    st.session_state.cmax=200
    st.session_state.kw=20
    st.session_state.kh=8
    st.session_state.min_area=1500
    st.session_state.min_r=2.0
    st.session_state.max_r=6.0

# ================= CSS =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');

body, .stApp {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(-45deg, #ff9a9e, #fad0c4, #a1c4fd, #c2e9fb);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}
@keyframes gradientBG {0% {background-position:0% 50%;}50% {background-position:100% 50%;}100% {background-position:0% 50%;}}

.header {
    background: linear-gradient(135deg,#f97316,#facc15);
    padding: 30px;
    border-radius: 25px;
    color: white;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
}

.card {
    background: linear-gradient(135deg,#ffffff,#e0f7fa);
    padding: 25px;
    border-radius: 25px;
    margin-bottom: 25px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
    transform: scale(1.03);
    box-shadow: 0 25px 50px rgba(0,0,0,0.3);
}

.stButton > button {
    background: linear-gradient(90deg,#f472b6,#3b82f6);
    color: white;
    font-weight: 700;
    border-radius: 20px;
    padding: 12px 28px;
    border: none;
    transition: transform 0.3s, opacity 0.3s;
}
.stButton > button:hover {
    opacity: 0.85;
    transform: scale(1.05);
}

input, textarea { 
    background: #ffffff !important; 
    color: #0f172a !important; 
    border-radius: 12px;
    padding: 8px;
}
[data-testid="stDataFrame"] { 
    background: #ffffff; 
    border-radius: 20px; 
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("<div class='header'><h1>🚗 Sistem Deteksi Plat Nomor</h1><p>Login & Register Terhubung Database</p></div>", unsafe_allow_html=True)

# ================= LOGIN & REGISTER =================
if not st.session_state.login:
    tab_login, tab_reg = st.tabs(["🔐 Login", "📝 Register"])
    with tab_login:
        st.markdown("<div class='card'><h3>Login</h3></div>", unsafe_allow_html=True)
        lu = st.text_input("Username", key="login_user")
        lp = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            user = login_user(lu, lp)
            if user:
                st.session_state.login=True
                st.session_state.user=lu
                st.stop()  # Refresh page setelah login
            else:
                st.error("Username atau password salah")
    with tab_reg:
        st.markdown("<div class='card'><h3>Register</h3></div>", unsafe_allow_html=True)
        ru = st.text_input("Username Baru", key="reg_user")
        rp = st.text_input("Password", type="password", key="reg_pass")
        cp = st.text_input("Konfirmasi Password", type="password", key="reg_confirm")
        if st.button("Daftar"):
            if ru=="" or rp=="": st.warning("Semua field wajib diisi")
            elif rp!=cp: st.error("Password tidak sama")
            elif register_user(ru,rp): 
                st.success("Registrasi berhasil, silakan login")
                st.stop()  # Refresh page setelah register
            else: st.error("Username sudah terdaftar")
    st.stop()

# ================= SIDEBAR =================
st.sidebar.markdown(f"<div class='card'><h3>👤 User</h3><b>{st.session_state.user}</b></div>", unsafe_allow_html=True)
menu = st.sidebar.radio("Menu", ["Deteksi","Hasil","Parameter","Penjelasan"])
if st.sidebar.button("Logout"):
    st.session_state.login=False
    st.session_state.user=""
    st.stop()  # Refresh page setelah logout

# ================= MENU DETEKSI =================
if menu=="Deteksi":
    st.markdown("<div class='card'><h1>🚘 Deteksi Plat Nomor</h1></div>",unsafe_allow_html=True)
    files = st.file_uploader("Upload gambar", type=["jpg","png","jpeg"], accept_multiple_files=True)
    if files and st.button("🚀 Jalankan Deteksi"):
        st.session_state.results=[]
        for f in files:
            img = np.array(Image.open(f).convert("RGB"))
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            box,edge,morph,plates,texts,locs = detect(
                bgr,
                st.session_state.cmin,
                st.session_state.cmax,
                st.session_state.kw,
                st.session_state.kh,
                st.session_state.min_area,
                st.session_state.min_r,
                st.session_state.max_r
            )
            st.session_state.results.append({
                "nama": f.name,
                "box": cv2.cvtColor(box,cv2.COLOR_BGR2RGB),
                "edge": edge,
                "morph": morph,
                "plates": plates,
                "texts": texts,
                "locs": locs
            })
        st.success("Deteksi selesai")

# ================= MENU HASIL =================
elif menu=="Hasil":
    st.markdown("<div class='card'><h1>📊 Hasil Deteksi</h1></div>",unsafe_allow_html=True)
    rows=[]
    for r in st.session_state.results:
        st.subheader(r["nama"])
        c1,c2,c3=st.columns(3)
        c1.image(r["box"],use_container_width=True)
        c2.image(r["edge"],caption="Edge",clamp=True)
        c3.image(r["morph"],caption="Morph",clamp=True)
        for i in range(len(r["plates"])):
            rows.append({
                "Nama Gambar":r["nama"],
                "Plat Ke":i+1,
                "Hasil OCR":r["texts"][i],
                "Wilayah":r["locs"][i]
            })
    if rows:
        df=pd.DataFrame(rows)
        st.dataframe(df,use_container_width=True)
        csv=df.to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV",csv,"hasil_deteksi_plat.csv","text/csv")

# ================= MENU PARAMETER =================
elif menu=="Parameter":
    st.markdown("<div class='card'><h1>⚙️ Parameter Deteksi</h1></div>",unsafe_allow_html=True)
    st.session_state.cmin=st.slider("Canny Min",0,150,st.session_state.cmin)
    st.session_state.cmax=st.slider("Canny Max",150,300,st.session_state.cmax)
    st.session_state.kw=st.slider("Kernel Width",5,30,st.session_state.kw)
    st.session_state.kh=st.slider("Kernel Height",3,15,st.session_state.kh)
    st.session_state.min_area=st.slider("Min Area",500,5000,st.session_state.min_area)
    st.session_state.min_r=st.slider("Min Ratio",1.0,4.0,st.session_state.min_r)
    st.session_state.max_r=st.slider("Max Ratio",4.0,8.0,st.session_state.max_r)

# ================= MENU PENJELASAN =================
else:
    st.markdown("""
    <div class='card'>
    <h2>📌 Penjelasan Sistem</h2>
    <ul>
    <li>Deteksi plat menggunakan image processing klasik</li>
    <li>OCR Tesseract untuk membaca teks plat</li>
    <li>Wilayah dikenali berdasarkan kode plat nasional</li>
    <li>Hasil ditampilkan dalam tabel dan bisa diunduh</li>
    </ul>
    </div>
    """,unsafe_allow_html=True)