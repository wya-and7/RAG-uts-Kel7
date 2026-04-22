"""
=============================================================
ANTARMUKA STREAMLIT — RAG UTS Data Engineering
=============================================================

Jalankan dengan: streamlit run ui/app.py
=============================================================
"""

import sys
import os
from pathlib import Path

# Agar bisa import dari folder src/
sys.path.append(str(Path(__file__).parent.parent / "src"))

import streamlit as st
from dotenv import load_dotenv
import requests
from streamlit_lottie import st_lottie
import json
import uuid
from datetime import datetime

load_dotenv()

# ─── Konfigurasi Halaman (Ekstrem) ────────────────────────────────────────────
st.set_page_config(
    page_title="AgriBot - Premium AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Injeksi Extreme Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
    
    /* Global Typography */
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }
    
    /* 1. HIDE DEFAULT STREAMLIT ELEMENTS (BREAKING THE TEMPLATE) */
    footer {visibility: hidden;}
    /* header visibility restored to support native menu/options */
    
    /* Remove default layout paddings */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1200px;
    }

    /* 4. PREMIUM HERO BANNER HTML CUSTOM */
    .hero-banner {
        background: linear-gradient(120deg, #10b981 0%, #059669 100%);
        border-radius: 30px;
        padding: 40px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 20px 40px rgba(5, 150, 105, 0.2);
        position: relative;
        overflow: hidden;
        border: 2px solid rgba(255,255,255,0.2);
    }
    
    .hero-banner::after {
        content: '';
        position: absolute;
        top: -50px;
        right: -50px;
        width: 200px;
        height: 200px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
    }

    .hero-banner h1 {
        font-size: 3.5rem;
        margin: 0;
        font-weight: 800;
        color: white !important;
        line-height: 1.1;
        letter-spacing: -1px;
    }
    
    .hero-banner p {
        font-size: 1.1rem;
        margin-top: 15px;
        opacity: 0.95;
        font-weight: 400;
    }

    /* 5. MODERN CHAT BUBBLES WITH SHADOWS */
    [data-testid="stChatMessage"] {
        background-color: var(--secondary-background-color);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        border: 1px solid rgba(16, 185, 129, 0.15);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    [data-testid="stChatMessage"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(16, 185, 129, 0.1);
    }
    
    /* Text in chat */
    div[data-testid="stChatMessageContent"] {
        color: var(--text-color);
        font-size: 1.05rem;
    }

    /* 6. FLOATING NEON CHAT INPUT */
    [data-testid="stChatInput"] {
        background: var(--background-color) !important;
        border: 2px solid rgba(16, 185, 129, 0.3) !important;
        border-radius: 30px !important;
        box-shadow: 0 10px 40px rgba(5, 150, 105, 0.08) !important;
        transition: all 0.3s ease;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #10b981 !important;
        box-shadow: 0 15px 50px rgba(16, 185, 129, 0.2) !important;
    }

    /* 7. QUICK ACTION BUTTONS AND PRIMARY BUTTONS */
    button[kind="primary"] {
        background: linear-gradient(to right, #10b981, #059669) !important;
        border-radius: 25px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        color: white !important;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 25px rgba(16, 185, 129, 0.45) !important;
    }
    
    /* General transparent/quick action buttons */
    div.stButton > button {
        border-radius: 18px;
        background: var(--background-color);
        border: 2px solid transparent;
        color: #10b981;
        font-weight: 500;
        transition: all 0.2s cubic-bezier(0.4, 0.0, 0.2, 1);
        height: auto;
        padding: 18px 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    div.stButton > button:hover {
        background: var(--secondary-background-color);
        border: 2px solid #10b981;
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(16, 185, 129, 0.15);
    }
    
    /* 8. CONTEXT CARD (EXPANDER UPGRADE) */
    .context-card {
        background-color: var(--background-color);
        border-left: 5px solid #10b981;
        padding: 1.5rem;
        border-radius: 8px 12px 12px 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        color: var(--text-color);
    }
    .context-card b { color: #10b981; font-size: 1.1rem; }
    
    hr {
        border-color: rgba(16, 185, 129, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ─── Lottie Fetcher ───────────────────────────────────────────────────────────
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# URL lottie animasi tanaman yang estetik
lottie_plant = load_lottieurl("https://lottie.host/80998f48-356a-4c28-be94-e3db78a571da/1A3oG5Fp7Y.json")

# ─── Load Vector Store (Backend logic intact) ──────────────────────────────────
@st.cache_resource
def load_vs():
    """Load vector store sekali saja, di-cache untuk performa."""
    try:
        from query import load_vectorstore
        return load_vectorstore(), None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Error: {e}"

# ─── Eksekusi Load ─────────────────────────────────────────────────────────────
vectorstore, error = load_vs()

if error:
    st.error(f" {error}")
    st.info("Jalankan terlebih dahulu: `python src/indexing.py`")
    st.stop()

# ─── Start UI Layout ──────────────────────────────────────────────────────────
if "app_loaded" not in st.session_state:
    st.toast("AgriBot siap membantu! 🌱", icon="✨")
    st.session_state.app_loaded = True

# ─── KONTROL STATE CHAT & RIWAYAT ─────────────────────────────────────────────
history_file = Path(__file__).parent / "chat_history.json"

if "chat_history" not in st.session_state:
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                st.session_state.chat_history = json.load(f)
        except Exception:
            st.session_state.chat_history = {}
    else:
        st.session_state.chat_history = {}

def save_chat_history():
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=2)

def create_new_session():
    new_id = str(uuid.uuid4())
    st.session_state.current_session_id = new_id
    st.session_state.chat_history[new_id] = {
        "title": "Percakapan Baru",
        "messages": [],
        "timestamp": datetime.now().isoformat()
    }
    save_chat_history()

if "current_session_id" not in st.session_state:
    create_new_session()

curr_id = st.session_state.current_session_id
if curr_id not in st.session_state.chat_history:
    create_new_session()
    curr_id = st.session_state.current_session_id

# ─── AUTO-PRUNE EMPTY SESSIONS ───
# Hapus semua sesi yang kosong (belum ada pesan) dan bukan sesi yang sedang aktif
keys_to_delete = [sid for sid, chat_data in st.session_state.chat_history.items() 
                  if not chat_data.get("messages") and sid != curr_id]

if keys_to_delete:
    for sid in keys_to_delete:
        del st.session_state.chat_history[sid]
    save_chat_history()

current_chat = st.session_state.chat_history[curr_id]
messages_list = current_chat["messages"]

# ─── SIDEBAR FLOATING KUSTOM ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("➕ Percakapan Baru", use_container_width=True, type="primary"):
        # Jangan buat baru terus-menerus jika chat saat ini masih kosong
        if current_chat.get("messages"):
            create_new_session()
            st.rerun()
        else:
            st.toast("Anda sudah berada di percakapan baru yang belum terisi!", icon="ℹ️")

    st.markdown("### 💬 Riwayat Percakapan")
    
    # Sort history by descending timestamp
    history_items = list(st.session_state.chat_history.items())
    history_items.sort(key=lambda x: x[1].get('timestamp', ''), reverse=True)
    
    for sid, chat_data in history_items:
        title = chat_data.get("title", "Percakapan")
        if sid == curr_id:
            btn_label = f"📍 {title} (Aktif)"
        else:
            btn_label = f"💬 {title}"
            
        if st.button(btn_label, key=f"hist_{sid}", use_container_width=True):
            st.session_state.current_session_id = sid
            st.rerun()

    st.divider()

    st.markdown("### ⚙️ Engine RAG")
    
    top_k = st.slider(
        "Jumlah dokumen relevan",
        min_value=1, max_value=10, value=3
    )
    
    show_context = st.checkbox("Tampilkan rujukan konteks", value=True)
    show_prompt = st.checkbox("Tampilkan prompt orisinal", value=False)
    
    st.divider()
    st.markdown("### 🌿 Info Sistem")
    
    st.markdown("""
    **Kelompok:** *(nama)*  
    **Framework RAG:** LangChain  
    **Vector DB:** ChromaDB  
    **Embedding:** MiniLM 
    """)
    st.divider()
    if lottie_plant:
        st_lottie(lottie_plant, height=150, key="sidebar_plant")

# ─── HERO BANNER ────────────────────────────────────────────────────────────
# Membuat layout container banner di atas dengan flex
st.markdown("""
<div class="hero-banner">
    <h1>AgriBot</h1>
    <p>Asisten AI Pertanian Premium — Tingkatkan panen dan rawat tanaman Anda dengan panduan instan dari sistem cerdas RAG.</p>
</div>
""", unsafe_allow_html=True)

# Tampilkan riwayat chat dengan custom styling & avatars
for msg in messages_list:
    avatar = "🧑‍🌾" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])
        
        # Format Konteks Card
        if msg["role"] == "assistant" and show_context and "contexts" in msg:
            with st.expander("📚 Rujukan Dokumen"):
                for i, ctx in enumerate(msg["contexts"], 1):
                    st.markdown(f"""
                    <div class="context-card">
                        <b>[{i}] Skor Relevansi: {ctx['score']}/5</b><br/>
                        <small style="color: var(--text-color); opacity: 0.7; font-family: monospace;">{ctx['source']}</small><br/><br/>
                        <span style="font-size: 0.95rem;">{ctx['content'][:300]}...</span>
                    </div>
                    """, unsafe_allow_html=True)

# ─── EMPTY STATE & QUICK ACTIONS ──────────────────────────────────────────────
quick_action_clicked = None
if len(messages_list) == 0:
    st.markdown("<h3 style='text-align: center; color: #059669; font-weight: 700; margin-bottom: 2rem;'>Coba tanyakan sesuatu:</h3>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    if c1.button("🪴 Cara budidaya selada hidroponik", use_container_width=True):
        quick_action_clicked = "Bagaimana tahapan menanam selada menggunakan sistem hidroponik?"
    if c2.button("🐛 Solusi penyakit wereng coklat", use_container_width=True):
        quick_action_clicked = "Apa obat atau cara ampuh untuk membasmi hama wereng coklat pada tanaman?"
    if c3.button("💧 Panduan penyiraman bunga hias", use_container_width=True):
        quick_action_clicked = "Kapan jadwal dan cara menyiram tanaman hias yang benar agar tidak layu?"
        
    st.markdown("<br><br>", unsafe_allow_html=True)

# ─── FRONTEND INPUT & BACKEND CALL ──────────────────────────────────────────────
user_input = st.chat_input("Ketik pertanyaan seputar botani, tanaman basah, hidroponik...")
question = quick_action_clicked or user_input

if question:
    # Set chat title first if it's new
    if len(messages_list) == 0:
        current_chat["title"] = question[:25] + ("..." if len(question) > 25 else "")
        current_chat["timestamp"] = datetime.now().isoformat()
        
    # 1. Catat input user dan render
    messages_list.append({"role": "user", "content": question})
    save_chat_history()
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.write(question)
    
    # 2. Render dan generate respon Assistant
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Merumuskan jawaban terbaik..."):
            try:
                from query import answer_question
                # Pastikan backend tetap berkerja persis seperti versi awal
                result = answer_question(question, vectorstore, top_k=top_k)
                
                # Tampilkan hasil
                st.write(result["answer"])
                
                # Tampilkan Expandable Konteks Card
                if show_context:
                    with st.expander("📚 Rujukan Dokumen"):
                        for i, ctx in enumerate(result["contexts"], 1):
                            st.markdown(f"""
                            <div class="context-card">
                                <b>[{i}] Skor Relevansi: {ctx['score']}/5</b><br/>
                                <small style="color: var(--text-color); opacity: 0.7; font-family: monospace;">{ctx['source']}</small><br/><br/>
                                <span style="font-size: 0.95rem;">{ctx['content'][:300]}...</span>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Tampilkan prompt orisinal jika di-toggle
                if show_prompt:
                    with st.expander("🔧 Internal LLM Prompt"):
                        st.code(result["prompt"], language="text")
                
                # Simpan metadata ke log percakapan
                messages_list.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "contexts": result["contexts"]
                })
                save_chat_history()
                
            except Exception as e:
                error_msg = f"Gangguan Sistem: {e}\n\nPastikan API Key LLM yang digunakan sudah tervalidasi."
                st.error(error_msg)
                messages_list.append({"role": "assistant", "content": error_msg})
                save_chat_history()

# ─── TOMBOL RESET CHAT ────────────────────────────────────────────────────────
if messages_list:
    st.markdown("<hr style='margin-top: 40px;'>", unsafe_allow_html=True)
    colA, colB, colC = st.columns([1,2,1])
    with colB:
        if st.button("🗑️ Hapus Percakapan", type="primary", use_container_width=True):
            if curr_id in st.session_state.chat_history:
                del st.session_state.chat_history[curr_id]
                save_chat_history()
            create_new_session()
            st.rerun()
