"""
=============================================================
PIPELINE QUERY — RAG UTS Data Engineering
=============================================================

Pipeline ini dijalankan setiap kali user mengajukan pertanyaan:
1. Ubah pertanyaan user ke vektor (query embedding)
2. Cari chunk paling relevan dari vector database (retrieval)
3. Gabungkan konteks + pertanyaan ke dalam prompt
4. Kirim ke LLM untuk mendapatkan jawaban

Jalankan CLI dengan: python src/query.py
=============================================================
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K         = int(os.getenv("TOP_K", 3))
VS_DIR        = Path(os.getenv("VECTORSTORE_DIR", "./vectorstore"))
LLM_MODEL     = os.getenv("LLM_MODEL_NAME", "llama3-8b-8192")


# =============================================================
# TODO MAHASISWA:
# Pilih implementasi yang sesuai dengan pilihan LLM kalian
# =============================================================


def load_vectorstore():
    """Memuat vector database yang sudah dibuat oleh indexing.py"""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    if not VS_DIR.exists():
        raise FileNotFoundError(
            f"Vector store tidak ditemukan di '{VS_DIR}'.\n"
            "Jalankan dulu: python src/indexing.py"
        )

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = Chroma(
        persist_directory=str(VS_DIR),
        embedding_function=embedding_model
    )
    return vectorstore


def retrieve_context(vectorstore, question: str, top_k: int = TOP_K) -> list:
    """
    LANGKAH 1 & 2: Query embedding + Similarity search.
    
    Fungsi ini:
    - Mengubah pertanyaan ke vektor
    - Mencari top_k chunk paling relevan
    - Mengembalikan list dokumen relevan
    """
    results = vectorstore.similarity_search_with_score(question, k=top_k)
    
    contexts = []
    for doc, score in results:
        contexts.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": round(float(score), 4)
        })
    
    return contexts


def build_prompt(question: str, contexts: list) -> str:
    """
    LANGKAH 3: Membangun prompt untuk LLM.
    
    Prompt yang baik untuk RAG harus:
    - Memberikan instruksi jelas ke LLM
    - Menyertakan konteks yang sudah diambil
    - Menanyakan pertanyaan user
    - Meminta LLM untuk jujur jika tidak tahu
    
    TODO: Modifikasi prompt ini sesuai domain dan bahasa proyek kalian!
    """
    context_text = "\n\n---\n\n".join(
        [f"[Sumber: {c['source']}]\n{c['content']}" for c in contexts]
    )

    prompt = f"""Role: Anda adalah Asisten Ahli Produksi Peternakan yang bertugas menganalisis hasil ternak dan harga pakan.

Task: Jawablah pertanyaan pengguna hanya dengan menggunakan informasi dari konteks dokumen yang diberikan.

Guidelines             :
Strict Fidelity        : Gunakan hanya informasi dari dokumen yang disediakan. Jangan menggunakan pengetahuan eksternal, asumsi, atau data di luar dokumen.
Handling Unknowns      : Jika jawaban tidak ditemukan secara eksplisit maupun implisit dalam dokumen, Anda wajib menjawab: "Saya tidak tahu". Jangan mencoba mengarang jawaban.
Linguistic Flexibility : Anda diperbolehkan memahami variasi bahasa, sinonim, atau maksud tersirat dari pertanyaan pengguna selama jawaban akhirnya tetap bersumber dari dokumen. (Contoh: Jika user bertanya "Berapa modal makan sapi?" dan dokumen menyebutkan "Harga pakan ternak potong", Anda harus mampu menghubungkannya).

Tone: Berikan jawaban yang informatif, ringkas, dan profesional.

KONTEKS:
{context_text}

PERTANYAAN:
{question}

JAWABAN:"""
    
    return prompt


# ─────────────────────────────────────────────────────────────
# OPSI LLM A: Groq (gratis, cepat) — REKOMENDASI
# ─────────────────────────────────────────────────────────────
def get_answer_groq(prompt: str) -> str:
    """Menggunakan Groq API (gratis, sangat cepat)."""
    from groq import Groq
    
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=LLM_MODEL,  # "llama3-8b-8192" atau "mixtral-8x7b-32768"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,   # Rendah = jawaban lebih konsisten/faktual
        max_tokens=1024
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────
# OPSI LLM B: Google Gemini (gratis tier)
# ─────────────────────────────────────────────────────────────
# def get_answer_gemini(prompt: str) -> str:
#     import google.generativeai as genai
#     genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#     model = genai.GenerativeModel("gemini-1.5-flash")
#     response = model.generate_content(prompt)
#     return response.text


# ─────────────────────────────────────────────────────────────
# OPSI LLM C: Ollama (100% offline, gratis)
# Pastikan Ollama sudah diinstall dan model sudah di-pull:
# ollama pull llama3
# ─────────────────────────────────────────────────────────────
# def get_answer_ollama(prompt: str) -> str:
#     import requests
#     response = requests.post(
#         "http://localhost:11434/api/generate",
#         json={"model": "llama3", "prompt": prompt, "stream": False}
#     )
#     return response.json()["response"]


def answer_question(question: str, vectorstore=None) -> dict:
    """
    Fungsi utama: menerima pertanyaan, mengembalikan jawaban + konteks.
    
    Returns:
        dict dengan keys: answer, contexts, prompt
    """
    if vectorstore is None:
        vectorstore = load_vectorstore()
    
    # Retrieve
    print(f"🔍 Mencari konteks relevan untuk: '{question}'")
    contexts = retrieve_context(vectorstore, question)
    print(f"   ✅ {len(contexts)} chunk relevan ditemukan")
    
    # Build prompt
    prompt = build_prompt(question, contexts)
    
    # Generate answer
    print("🤖 Mengirim ke LLM...")
    
    # TODO: Ganti sesuai LLM yang kalian pilih
    answer = get_answer_groq(prompt)
    # answer = get_answer_gemini(prompt)
    # answer = get_answer_ollama(prompt)
    
    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "prompt": prompt
    }


# ─── CLI Interface ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  🤖 RAG System — UTS Data Engineering")
    print("  Ketik 'keluar' untuk mengakhiri")
    print("=" * 55)

    try:
        vs = load_vectorstore()
        print("✅ Vector database berhasil dimuat\n")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        exit(1)

    while True:
        print()
        question = input("❓ Pertanyaan Anda: ").strip()
        
        if question.lower() in ["keluar", "exit", "quit", "q"]:
            print("👋 Sampai jumpa!")
            break
        
        if not question:
            print("⚠️  Pertanyaan tidak boleh kosong.")
            continue
        
        try:
            result = answer_question(question, vs)
            
            print("\n" + "─" * 55)
            print("💬 JAWABAN:")
            print(result["answer"])
            
            print("\n📚 SUMBER KONTEKS:")
            for i, ctx in enumerate(result["contexts"], 1):
                print(f"  [{i}] Skor: {ctx['score']:.4f} | {ctx['source']}")
                print(f"      {ctx['content'][:100]}...")
            print("─" * 55)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Pastikan API key sudah diatur di file .env")
