"""
=============================================================
PIPELINE INDEXING — RAG UTS Data Engineering
=============================================================

Pipeline ini dijalankan SEKALI untuk:
1. Memuat dokumen dari folder data/
2. Memecah dokumen menjadi chunk-chunk kecil
3. Mengubah setiap chunk menjadi vektor (embedding)
4. Menyimpan vektor ke dalam vector database

Jalankan dengan: python src/indexing.py
=============================================================
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ─── LANGKAH 0: Load konfigurasi dari .env ───────────────────────────────────
load_dotenv()

# Konfigurasi — bisa diubah sesuai kebutuhan
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
DATA_DIR      = Path(os.getenv("DATA_DIR", "./data"))
VS_DIR        = Path(os.getenv("VECTORSTORE_DIR", "./vectorstore"))

# ─────────────────────────────────────────────────────────────
# IMPLEMENTASI B: From Scratch (tanpa LangChain)
# Uncomment blok ini jika memilih opsi from scratch
# ─────────────────────────────────────────────────────────────

def build_index_scratch():
    """Implementasi RAG dari scratch menggunakan sentence-transformers + FAISS."""
    import json
    import pandas as pd
    import numpy as np
    import faiss
    from embeddings import embed_texts
    from utils import clean_filename

    print(" Memulai Pipeline Indexing (From Scratch)")

    # Load dokumen
    documents = []
    for file_path in DATA_DIR.glob("**/*.csv"):
        df = pd.read_csv(file_path, sep=";")
        # ambil nama file tanpa extension
        jenis_produksi = clean_filename(file_path.stem)
        content = ""
        for _, row in df.iterrows():
            if row["2021"] != "-" and row["2022"] != "-":
                content += (
                    f"Produksi {jenis_produksi} di provinsi {row['Provinsi']} "
                    f"pada tahun 2021 adalah {row['2021']} ton, "
                    f"dan pada tahun 2022 adalah {row['2022']} ton.\n"
                )
        documents.append({
            "source": str(file_path),
            "content": content
        })
    print(f" {len(documents)} dokumen dimuat")

    # Chunking manual
    chunks = []
    for doc in documents:
        text = doc["content"]
        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_text = text[i:i + CHUNK_SIZE]
            if len(chunk_text.strip()) > 50:
                chunks.append({"source": doc["source"], "text": chunk_text, "id": len(chunks)})
    print(f" {len(chunks)} chunk dibuat")

    # Embedding
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)
    print(f" Embedding selesai, dimensi: {embeddings.shape}")

    # Simpan ke FAISS
    VS_DIR.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, str(VS_DIR / "index.faiss"))

    # Simpan metadata
    with open(VS_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f" Index FAISS tersimpan di {VS_DIR}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_index_scratch()
