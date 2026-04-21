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

# Model embedding lokal (berjalan offline, tidak perlu API key)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# =============================================================
# HELPER: Load dokumen CSV
# =============================================================

def load_csv_documents(data_dir: Path) -> list:
    """
    Memuat semua file CSV dari folder data/ menggunakan CSVLoader.
    Setiap baris CSV diubah menjadi satu dokumen LangChain.
    """
    from langchain_community.document_loaders import CSVLoader

    csv_docs = []
    csv_files = list(data_dir.glob("**/*.csv"))

    if not csv_files:
        print("   [!] Tidak ada file CSV ditemukan.")
        return csv_docs

    for csv_path in csv_files:
        print(f"   📊 Memuat CSV: {csv_path.name}")
        try:
            loader = CSVLoader(
                file_path=str(csv_path),
                encoding="utf-8",
                csv_args={"delimiter": ","},
            )
            docs = loader.load()
            # Tambahkan metadata sumber
            for doc in docs:
                doc.metadata["source_type"] = "csv"
                doc.metadata["file_name"] = csv_path.name
            csv_docs.extend(docs)
            print(f"      ✓ {len(docs)} baris dimuat dari {csv_path.name}")
        except Exception as e:
            print(f"      ✗ Gagal memuat {csv_path.name}: {e}")

    return csv_docs


# =============================================================
# HELPER: Load dokumen PDF
# =============================================================

def load_pdf_documents(data_dir: Path) -> list:
    """
    Memuat semua file PDF dari folder data/ menggunakan PyPDFLoader.
    Setiap halaman PDF diubah menjadi satu dokumen LangChain.
    """
    from langchain_community.document_loaders import PyPDFLoader

    pdf_docs = []
    pdf_files = list(data_dir.glob("**/*.pdf"))

    if not pdf_files:
        print("   [!] Tidak ada file PDF ditemukan.")
        return pdf_docs

    for pdf_path in pdf_files:
        print(f"   📕 Memuat PDF: {pdf_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            # Tambahkan metadata sumber
            for doc in docs:
                doc.metadata["source_type"] = "pdf"
                doc.metadata["file_name"] = pdf_path.name
            pdf_docs.extend(docs)
            print(f"      ✓ {len(docs)} halaman dimuat dari {pdf_path.name}")
        except Exception as e:
            print(f"      ✗ Gagal memuat {pdf_path.name}: {e}")

    return pdf_docs


# =============================================================
# IMPLEMENTASI UTAMA: LangChain + ChromaDB + Model Lokal
# =============================================================

def build_index_langchain():
    """
    Membangun index menggunakan LangChain dan ChromaDB.

    Komponen yang digunakan:
    - CSVLoader        : memuat file CSV (tiap baris = 1 dokumen)
    - PyPDFLoader      : memuat file PDF (tiap halaman = 1 dokumen)
    - RecursiveCharacterTextSplitter : memecah dokumen jadi chunk
    - HuggingFaceEmbeddings          : model lokal (GRATIS, offline)
    - Chroma           : vector database lokal
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    print("=" * 60)
    print("  🚀  Memulai Pipeline Indexing (LangChain + Model Lokal)")
    print("=" * 60)
    print(f"  Data dir   : {DATA_DIR.absolute()}")
    print(f"  Vector dir : {VS_DIR.absolute()}")
    print(f"  Chunk size : {CHUNK_SIZE} | Overlap: {CHUNK_OVERLAP}")
    print(f"  Model      : {EMBEDDING_MODEL}")

    # ─── LANGKAH 1: Load Dokumen ─────────────────────────────
    print("\n📄 Langkah 1: Memuat dokumen (CSV + PDF)...")

    all_documents = []

    # Load CSV
    csv_docs = load_csv_documents(DATA_DIR)
    all_documents.extend(csv_docs)

    # Load PDF
    pdf_docs = load_pdf_documents(DATA_DIR)
    all_documents.extend(pdf_docs)

    if not all_documents:
        print("\n❌ Tidak ada dokumen yang berhasil dimuat. Periksa folder data/.")
        return None

    print(f"\n   ✅ Total dokumen dimuat  : {len(all_documents)}")
    print(f"      - Dari CSV           : {len(csv_docs)}")
    print(f"      - Dari PDF           : {len(pdf_docs)}")
    print(f"   Total karakter          : {sum(len(d.page_content) for d in all_documents):,}")

    # ─── LANGKAH 2: Chunking ─────────────────────────────────
    print(f"\n✂️  Langkah 2: Memecah dokumen "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(all_documents)

    print(f"   ✅ {len(chunks)} chunk berhasil dibuat")
    if chunks:
        avg_size = sum(len(c.page_content) for c in chunks) // len(chunks)
        print(f"   Rata-rata ukuran chunk: {avg_size} karakter")

        # Tampilkan contoh chunk dari masing-masing tipe
        csv_chunks = [c for c in chunks if c.metadata.get("source_type") == "csv"]
        pdf_chunks = [c for c in chunks if c.metadata.get("source_type") == "pdf"]

        if csv_chunks:
            print(f"\n   📊 Contoh chunk CSV (pertama):")
            print(f"   {'-'*45}")
            print(f"   {csv_chunks[0].page_content[:300]}")

        if pdf_chunks:
            print(f"\n   📕 Contoh chunk PDF (pertama):")
            print(f"   {'-'*45}")
            print(f"   {pdf_chunks[0].page_content[:300]}")

    # ─── LANGKAH 3: Embedding (Model Lokal) ──────────────────
    print(f"\n🧠 Langkah 3: Membuat embedding menggunakan model lokal...")
    print(f"   Model: {EMBEDDING_MODEL}")
    print("   (Pertama kali dijalankan akan mengunduh model ~90MB)")

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("   ✅ Embedding model siap (multilingual, mendukung Bahasa Indonesia)")

    # ─── LANGKAH 4: Simpan ke Vector DB ──────────────────────
    print(f"\n🗄️  Langkah 4: Menyimpan ke ChromaDB ({VS_DIR})...")

    VS_DIR.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(VS_DIR)
    )

    print(f"   ✅ {len(chunks)} chunk tersimpan di vector database")
    print("\n" + "=" * 60)
    print("  ✅  Indexing selesai! Vector database siap digunakan.")
    print(f"      Lokasi: {VS_DIR.absolute()}")
    print("=" * 60)

    return vectorstore


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_index_langchain()
