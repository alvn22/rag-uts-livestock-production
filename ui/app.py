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

load_dotenv()

# ─── Konfigurasi Halaman ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG System — UTS Data Engineering",
    page_icon="🤖",
    layout="wide"
)

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("Sistem Tanya-Jawab RAG")
st.caption("UTS Data Engineering — Retrieval-Augmented Generation")
st.divider()

# ─── Sidebar: Info & Konfigurasi ─────────────────────────────────────────────
with st.sidebar:
    st.header("Konfigurasi")
    
    top_k = st.slider(
        "Jumlah dokumen relevan (top-k)",
        min_value=1, max_value=10, value=3,
        help="Berapa banyak chunk yang diambil dari vector database"
    )
    
    show_context = st.checkbox("Tampilkan konteks yang digunakan", value=True)
    show_prompt = st.checkbox("Tampilkan prompt ke LLM", value=False)
    
    st.divider()
    st.header("Info Sistem")
    
    # TODO: Isi informasi kelompok kalian di sini
    st.markdown("""
    **Kelompok:** *Kelompok 7*  
    **Domain:** *Livestock Production*  
    **LLM:** *Gemini 3 Flash*  
    **Vector DB:** FAISS  
    **Embedding:** multilingual-MiniLM
    """)
    
    st.divider()
    st.info("💡 Tip: Mulai dengan pertanyaan spesifik yang jawabannya ada di dalam dokumen Anda.")


# ─── Load Vector Store (cached agar tidak reload setiap query) ───────────────
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


# ─── Main Content ──────────────────────────────────────────────────────────────
vectorstore, error = load_vs()

if error:
    st.error(f" {error}")
    st.info("Jalankan terlebih dahulu: `python src/indexing.py`")
    st.stop()

st.success("Vector database berhasil dimuat dan siap digunakan!")

# ─── Chat Interface ───────────────────────────────────────────────────────────
# Simpan riwayat chat di session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan riwayat chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and show_context and "contexts" in msg:
            with st.expander("Konteks yang digunakan"):
                for i, ctx in enumerate(msg["contexts"], 1):
                    st.markdown(f"**[{i}] Skor: {ctx['score']:.4f}** | `{ctx['source']}`")
                    st.text(ctx["content"][:300] + "...")
                    st.divider()

# Input pertanyaan baru
if question := st.chat_input("Ketik pertanyaan Anda di sini..."):
    
    # Tampilkan pertanyaan user
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    
    # Generate jawaban
    with st.chat_message("assistant"):
        with st.spinner("Mencari informasi relevan dan menghasilkan jawaban..."):
            try:
                from query import answer_question
                result = answer_question(question, vectorstore, top_k=top_k)
                
                st.write(result["answer"])
                
                # Tampilkan konteks jika diaktifkan
                if show_context:
                    with st.expander("📚 Konteks yang digunakan"):
                        for i, ctx in enumerate(result["contexts"], 1):
                            st.markdown(f"**[{i}] Skor relevansi: {ctx['score']:.4f}** | `{ctx['source']}`")
                            st.text(ctx["content"][:300] + "...")
                            st.divider()
                
                # Tampilkan prompt jika diaktifkan
                if show_prompt:
                    with st.expander("🔧 Prompt yang dikirim ke LLM"):
                        st.code(result["prompt"], language="text")
                
                # Simpan ke riwayat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "contexts": result["contexts"]
                })
                
            except Exception as e:
                error_msg = f"Error: {e}\n\nPastikan API key sudah diatur di file .env"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ─── Tombol Reset Chat ────────────────────────────────────────────────────────
if st.session_state.messages:
    if st.button("Hapus Riwayat Chat"):
        st.session_state.messages = []
        st.rerun()
