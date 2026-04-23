# 🤖 RAG Starter Pack — UTS Data Engineering

> **Retrieval-Augmented Generation** — Sistem Tanya-Jawab Cerdas Berbasis Dokumen

Starter pack ini adalah **kerangka awal** proyek RAG untuk UTS Data Engineering D4.
Mahasiswa mengisi, memodifikasi, dan mengembangkan kode ini sesuai topik kelompok masing-masing.

---

## 👥 Identitas Kelompok

| Nama | NIM | Tugas Utama |
|------|-----|-------------|
| Alvian Dwiky P.S  | 244311002 | Data Engineer         |
| Davin Rafael S.  | 244311007 | Project Manager         |
| Nadia Tifara S. | 244311022 | Data Analyst         |

**Topik Domain:** *Pertanian & Lingkungan*  
**Stack yang Dipilih:** *From Scratch*  
**LLM yang Digunakan:** *Gemini*  
**Vector DB yang Digunakan:** *FAISS*

---

## 🗂️ Struktur Proyek

```
rag-uts-livestock-production/
├── data/                    # Dokumen sumber
│   └── sample.txt           # Contoh dokumen
├── src/
│   ├── indexing.py          # 🔧 Pipeline indexing
│   ├── query.py             # 🔧 Pipeline query & retrieval
│   ├── embeddings.py        # 🔧 Konfigurasi embedding
│   └── utils.py             # Helper functions
├── ui/
│   └── app.py               # 🔧 Antarmuka Streamlit
├── docs/
│   └── arsitektur.png       # 📌 Diagram arsitektur
├── evaluation/
│   └── hasil_evaluasi.xlsx  # 📌 Tabel evaluasi 10 pertanyaan
├── notebooks/
│   └── 01_demo_rag.ipynb    # Notebook demo dari hands-on session
├── .env.example             # Template environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚡ Cara Memulai (Quickstart)

### 1. Clone & Setup

```bash
# Clone repository ini
git clone https://github.com/[username]/rag-uts-[kelompok].git
cd rag-uts-[kelompok]

# Buat virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# atau: venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Konfigurasi API Key

```bash
# Salin template env
cp .env.example .env

# Edit .env dan isi API key Anda
# JANGAN commit file .env ke GitHub!
```

### 3. Siapkan Dokumen

Letakkan dokumen sumber Anda di folder `data/`:
```bash
# Contoh: salin PDF atau TXT ke folder data
cp dokumen-saya.pdf data/
```

### 4. Jalankan Indexing (sekali saja)

```bash
python src/indexing.py
```

### 5. Jalankan Sistem RAG

```bash
# Dengan Streamlit UI
streamlit run ui/app.py

# Atau via CLI
python src/query.py
```

---

## 🔧 Konfigurasi

Semua konfigurasi utama ada di `src/config.py` (atau langsung di setiap file):

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `CHUNK_SIZE` | 500 | Ukuran setiap chunk teks (karakter) |
| `CHUNK_OVERLAP` | 50 | Overlap antar chunk |
| `TOP_K` | 6 | Jumlah dokumen relevan yang diambil |
| `MODEL_NAME` | *gemini-3-flash-preview* | Nama model LLM yang digunakan |

---

## 📊 Hasil Evaluasi

*(Isi setelah pengujian selesai)*

| # | Pertanyaan | Jawaban Sistem | Jawaban Ideal | Skor (1-5) |
|---|-----------|----------------|---------------|-----------|
| 1 | Hasil produksi telur provinsi Aceh | Produksi telur ayam petelur di Provinsi Aceh tercatat sebesar 774,4 ton pada tahun 2021 dan 1.406,31 ton pada tahun 2022. | Jawaban sudah relevan dan lengkap | 5 |
| 2 | Hasil produksi susu | Total produksi susu segar di Indonesia tercatat sebesar 946.388,17 ton pada tahun 2021 dan meningkat menjadi 968.980,14 ton pada tahun 2022. | Jawaban sudah relevan dan lengkap | 5 |
| 3 | Telur tahun 2022 | Berikut adalah data produksi telur ayam petelur pada tahun 2022 di berbagai provinsi berdasarkan dokumen yang tersedia: | Jawaban sudah relevan dan lengkap | 5 |
| 4 | Hasil produksi daging kambing provinsi jawa timur | Saya tidak tahu. Informasi mengenai hasil produksi daging kambing untuk Provinsi Jawa Timur tidak ditemukan dalam dokumen yang tersedia. | Daripada berhenti di kata "tidak tahu", AI bisa memberikan gambaran tentang apa yang sebenarnya ada di dokumen. | 1 |
| 5 | Hasil produksi daging sapi provinsi maluku | Saya tidak tahu karena data mengenai hasil produksi daging sapi di Provinsi Maluku tidak tersedia dalam dokumen yang diberikan. | Sistem disarankan memberikan respon yang menyatakan keterbatasan data namun tetap menyebutkan cakupan informasi yang tersedia (misalnya data ayam) agar tetap membantu bagi | 1 |
| 6 | Hasil produksi daging kerbau | Saya tidak tahu. Dokumen yang tersedia hanya memuat data mengenai produksi daging ayam ras pedaging dan produksi daging sapi, sedangkan data untuk produksi daging kerbau tidak ditemukan. | sistem harus bisa menawarkan bantuan pada data yang tersedia (Ayam/Sapi) alih-alih hanya menolak pertanyaan. | 1 |
| 7 | Hasil produksi daging ayam provinsi sumatera | Berdasarkan dokumen yang tersedia, berikut adalah data produksi daging ayam ras pedaging untuk wilayah di Sumatera: | Jawaban sudah relevan dan lengkap | 5 |
| 8 | Daging tahun 2020 | Saya tidak tahu. Dokumen yang tersedia hanya memuat informasi mengenai data produksi daging (kerbau dan ayam ras pedaging) untuk tahun 2021 dan 2022. | Sistem seharusnya tetap jujur(mencegah halusinasi), transparan mengenai isi dokumen, dan memberikan Call to Action (CTA) kepada pengguna. | 1 |
| 9 | Susu tahun 2019 | Saya tidak tahu. Dokumen yang tersedia hanya memuat data produksi susu segar untuk tahun 2021 dan 2022. | Tidak ada jawaban ideal, karena kesalahan dari prompter atau user karena data yang digunakan hanya data pada tahun 2021 dan 2022 | 1 |
| 10 | Produksi daging kambing provinsi NTB | Produksi daging kambing di provinsi Nusa tenggara Barat tercatat sebesar 392,82 ton | Jawaban sudah relevan dan lengkap | 5 |







**Rata-rata Skor:** 3  
**Analisis:** 

•	Jika prompt tidak spesifik, LLM tidak dapat menemukan jawaban

•	Sistem belum bisa memberi alternatif jawaban 

•	Respon sistem cenderung memberikan jawaban yang masih kaku, belum natural seperti percakapan manusia.


---

## 🏗️ Arsitektur Sistem

*(Masukkan gambar diagram arsitektur di sini)*


<img width="1920" height="1080" alt="Dokumen" src="https://github.com/user-attachments/assets/8f587c15-367a-40c9-9179-000b346a014e" />


---

## 📚 Referensi & Sumber

- Framework: *From Scratch*
- LLM: *Gemini*
- Vector DB: *FAISS docs*
- Tutorial yang digunakan: *(cantumkan URL)*

---

## 👨‍🏫 Informasi UTS

- **Mata Kuliah:** Data Engineering
- **Program Studi:** D4 Teknologi Rekayasa Perangkat Lunak
- **Deadline:** *23 April 2026*
