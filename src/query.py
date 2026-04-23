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

TOP_K         = int(os.getenv("TOP_K", 6))
VS_DIR        = Path(os.getenv("VECTORSTORE_DIR", "./vectorstore"))
LLM_MODEL     = os.getenv("LLM_MODEL_NAME", "gemini-3-flash-preview")

def load_vectorstore():
    """Memuat vector database yang sudah dibuat oleh indexing.py"""
    import json
    import faiss

    if not VS_DIR.exists():
        raise FileNotFoundError(
            f"Vector store tidak ditemukan di '{VS_DIR}'.\n"
            "Jalankan dulu: python src/indexing.py"
        )

    index = faiss.read_index(str(VS_DIR / "index.faiss"))

    with open(VS_DIR / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return {
        "index": index,
        "chunks": chunks
    }


def retrieve_context(vectorstore, question: str, top_k: int = TOP_K) -> list:
    """
    LANGKAH 1 & 2: Query embedding + Similarity search.
    
    Fungsi ini:
    - Mengubah pertanyaan ke vektor
    - Mencari top_k chunk paling relevan
    - Mengembalikan list dokumen relevan
    """
    import numpy as np
    import faiss
    from embeddings import embed_query

    index = vectorstore["index"]
    chunks = vectorstore["chunks"]

    # embed query
    query_vec = embed_query(question).astype("float32")
    query_vec = np.array([query_vec])

    # cosine similarity
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, top_k)

    contexts = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = chunks[idx]

        contexts.append({
            "content": chunk["text"],
            "source": chunk["source"],
            "score": float(score)
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

    prompt = f"""Anda adalah Asisten Ahli Produksi Peternakan Indonesia yang bertugas menjawab pertanyaan berdasarkan dokumen yang diberikan (hasil retrieval RAG).

TUJUAN:
Menjawab pertanyaan pengguna tentang:
- Produksi peternakan
- Pendapatan peternakan
- Harga pakan
- Populasi ternak
- Data provinsi 


SUMBER JAWABAN:
Gunakan HANYA informasi yang terdapat pada konteks dokumen yang diberikan.
Jangan gunakan pengetahuan luar, asumsi, atau tebakan.

ATURAN UTAMA:

1. PRIORITASKAN PENCOCOKAN MAKNA, BUKAN KATA PERSIS
Jika pengguna bertanya dengan istilah berbeda, cocokkan dengan data yang mirip di dokumen.

Contoh:
- "Pendapatan ternak Jawa Timur" dapat cocok dengan:
  "nilai produksi", "nilai usaha ternak", "penerimaan peternakan", "omzet peternakan", "revenue peternakan"

- "Produksi sapi Jatim" dapat cocok dengan:
  "hasil ternak sapi", "jumlah produksi sapi", "output sapi potong"

- "Modal makan sapi" dapat cocok dengan:
  "harga pakan", "biaya pakan", "biaya konsentrat"

2. PERTANYAAN WILAYAH
Jika user menyebut provinsi, cari semua variasi penulisannya.

Contoh:
- Jawa Timur = Jatim
- Jawa Barat = Jabar
- DI Yogyakarta = DIY
- Sumatera Utara = Sumut

3. JIKA DATA ADA SEBAGIAN, JAWAB SEBAGIAN
Jangan langsung jawab "Saya tidak tahu" jika ada data yang relevan sebagian.

Contoh:
Jika user bertanya:
"Pendapatan peternakan Jawa Timur tahun 2023"

dan dokumen hanya punya:
"Pendapatan peternakan Jawa Timur = ..."

maka jawab:
"Dokumen memuat data pendapatan peternakan Jawa Timur sebesar ... , namun tahun tidak disebutkan."

4. GABUNGKAN INFORMASI TERSEBAR
Jika informasi tersebar di beberapa potongan konteks, gabungkan menjadi jawaban ringkas.

5. JIKA ADA TABEL
Utamakan membaca kolom:
Provinsi, Tahun, Produksi, Pendapatan, Harga, Populasi, Komoditas.

6. JANGAN TERLALU CEPAT MENOLAK
Sebelum menjawab "Saya tidak tahu", periksa apakah ada:
- sinonim
- singkatan wilayah
- angka dalam tabel
- data relevan sebagian
- kalimat implisit

7. JIKA BENAR-BENAR TIDAK ADA DATA
Jawab tepat:

"Saya tidak tahu" dan sertakan penjelasan singkat bahwa data tidak ada

FORMAT JAWABAN:

- Ringkas
- Langsung ke inti
- Sertakan angka jika tersedia
- Sebut wilayah/tahun jika ada

CONTOH:

Pertanyaan:
Berapa produksi telur Jawa Timur?

Jawaban:
Produksi telur di Jawa Timur tercatat sebesar 125.000 ton.

Pertanyaan:
Pendapatan peternakan Sulawesi Selatan?

Jawaban:
Nilai usaha peternakan Sulawesi Selatan tercatat Rp2,3 triliun.

Pertanyaan:
Harga pakan kambing Papua?

Jawaban:
Saya tidak tahu

KONTEKS:
{context_text}

PERTANYAAN:
{question}

JAWABAN:"""
    
    return prompt

# ─────────────────────────────────────────────────────────────
# OPSI LLM B: Google Gemini (gratis tier)
# ─────────────────────────────────────────────────────────────
def get_answer_gemini(prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt)
    return response.text

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
    answer = get_answer_gemini(prompt)
    
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
