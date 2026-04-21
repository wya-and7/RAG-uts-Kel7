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
import sys
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

TOP_K         = int(os.getenv("TOP_K", 3))
VS_DIR        = Path(os.getenv("VECTORSTORE_DIR", "./vectorstore"))

# Model embedding HARUS sama persis dengan yang dipakai saat indexing.py
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Model LLM lokal (Ollama) — default: llama3
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "llama3")
# OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://localhost:11434")

# (Opsional) Groq API sebagai fallback
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = os.getenv("LLM_MODEL_NAME", "llama-3.1-8b-instant")


# =============================================================
# LANGKAH 1: Load Vector Database
# =============================================================

def load_vectorstore():
    """
    Memuat vector database yang sudah dibuat oleh indexing.py.
    Menggunakan embedding model lokal yang SAMA dengan saat indexing.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    if not VS_DIR.exists():
        raise FileNotFoundError(
            f"Vector store tidak ditemukan di '{VS_DIR}'.\n"
            "Jalankan dulu: python src/indexing.py"
        )

    print(f"   Model embedding : {EMBEDDING_MODEL}")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma(
        persist_directory=str(VS_DIR),
        embedding_function=embedding_model
    )
    return vectorstore


# =============================================================
# LANGKAH 2: Retrieval — Cari chunk paling relevan
# =============================================================

def retrieve_context(vectorstore, question: str, top_k: int = TOP_K) -> list:
    """
    Mengubah pertanyaan ke vektor lalu mencari top_k chunk paling relevan.
    Mengembalikan list dict berisi konten, sumber, tipe, dan skor relevansi.
    """
    results = vectorstore.similarity_search_with_score(question, k=top_k)

    contexts = []
    for doc, score in results:
        contexts.append({
            "content"     : doc.page_content,
            "source"      : doc.metadata.get("source", "unknown"),
            "file_name"   : doc.metadata.get("file_name", ""),
            "source_type" : doc.metadata.get("source_type", "unknown"),
            "score"       : round(float(score), 4)
        })

    return contexts


# =============================================================
# LANGKAH 3: Build Prompt
# =============================================================

def build_prompt(question: str, contexts: list) -> str:
    """
    Membangun prompt RAG untuk LLM.

    Prompt menyertakan:
    - Instruksi jelas (jawab berdasarkan konteks saja)
    - Konteks yang diambil (dari CSV dan/atau PDF)
    - Pertanyaan user
    """
    context_blocks = []
    for i, c in enumerate(contexts, 1):
        tipe = c["source_type"].upper()
        nama = c["file_name"] or c["source"]
        context_blocks.append(
            f"[{i}] Sumber ({tipe}): {nama}\n{c['content']}"
        )

    context_text = "\n\n---\n\n".join(context_blocks)

    prompt = f"""Kamu adalah asisten cerdas yang menjawab pertanyaan berdasarkan dokumen yang diberikan.

INSTRUKSI:
- Jawab HANYA berdasarkan konteks di bawah ini
- Jika jawaban tidak ada dalam konteks, katakan "Saya tidak menemukan informasi tersebut dalam dokumen yang tersedia"
- Jawab dalam Bahasa Indonesia yang jelas dan ringkas
- Jangan mengarang informasi yang tidak ada di konteks
- Jika konteks berasal dari CSV (data tabular), analisis datanya secara ringkas

KONTEKS DARI DOKUMEN:
{context_text}

PERTANYAAN:
{question}

JAWABAN:"""

    return prompt


# =============================================================
# LANGKAH 4 — OPSI LLM
# =============================================================

def get_answer_ollama(prompt: str) -> str:
    """
    LLM LOKAL via Ollama (100% offline, gratis).

    Cara setup:
      1. Install Ollama: https://ollama.com/download
      2. Pull model  : ollama pull llama3
      3. Jalankan    : ollama serve  (biasanya auto-start)
    """
    import requests

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model"  : OLLAMA_MODEL,
                "prompt" : prompt,
                "stream" : False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 1024,
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Ollama tidak berjalan di {OLLAMA_URL}.\n"
            "Pastikan Ollama sudah diinstall dan jalankan: ollama serve"
        )


def get_answer_groq(prompt: str) -> str:
    """
    LLM via Groq API (gratis tier, butuh internet + API key).
    Atur GROQ_API_KEY di file .env untuk menggunakan opsi ini.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY belum diatur di file .env")

    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024
    )
    return response.choices[0].message.content


# def get_answer_gemini(prompt: str) -> str:
#     """LLM via Google Gemini API (gratis tier)."""
#     import google.generativeai as genai
#     genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#     model = genai.GenerativeModel("gemini-1.5-flash")
#     response = model.generate_content(prompt)
#     return response.text


# =============================================================
# FUNGSI UTAMA: answer_question
# =============================================================

def answer_question(question: str, vectorstore=None) -> dict:
    """
    Fungsi utama: menerima pertanyaan, mengembalikan jawaban + konteks.

    Urutan LLM yang dicoba:
      1. Ollama (lokal, offline) — default
      2. Groq (fallback jika GROQ_API_KEY tersedia)

    Returns:
        dict dengan keys: question, answer, contexts, prompt, llm_used
    """
    if vectorstore is None:
        vectorstore = load_vectorstore()

    # ── Retrieval ──
    print(f"🔍 Mencari konteks relevan untuk: '{question}'")
    contexts = retrieve_context(vectorstore, question)
    print(f"   ✅ {len(contexts)} chunk relevan ditemukan:")
    for i, ctx in enumerate(contexts, 1):
        tipe = ctx["source_type"].upper()
        nama = ctx["file_name"] or ctx["source"]
        print(f"      [{i}] ({tipe}) {nama} | skor: {ctx['score']:.4f}")

    # ── Build prompt ──
    prompt = build_prompt(question, contexts)

    # ── Generate answer (coba Ollama dulu, fallback ke Groq) ──
    llm_used = ""
    answer   = ""

    print(f"🤖 Mengirim ke LLM lokal (Ollama/{OLLAMA_MODEL})...")
    try:
        answer   = get_answer_ollama(prompt)
        llm_used = f"Ollama ({OLLAMA_MODEL})"
    except Exception as ollama_err:
        print(f"   ⚠️  Ollama gagal: {ollama_err}")
        if GROQ_API_KEY:
            print(f"   🔄 Fallback ke Groq ({GROQ_MODEL})...")
            try:
                answer   = get_answer_groq(prompt)
                llm_used = f"Groq ({GROQ_MODEL})"
            except Exception as groq_err:
                answer   = f"Semua LLM gagal.\nOllama: {ollama_err}\nGroq: {groq_err}"
                llm_used = "none"
        else:
            answer   = (
                f"LLM gagal: {ollama_err}\n"
                "Atur GROQ_API_KEY di .env untuk fallback ke Groq."
            )
            llm_used = "none"

    return {
        "question" : question,
        "answer"   : answer,
        "contexts" : contexts,
        "prompt"   : prompt,
        "llm_used" : llm_used,
    }


# ─── CLI Interface ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  🤖  RAG System — UTS Data Engineering")
    print(f"  📦  Embedding : {EMBEDDING_MODEL.split('/')[-1]}")
    print(f"  🧠  LLM       : Ollama/{OLLAMA_MODEL} (lokal)")
    print("  Ketik 'keluar' untuk mengakhiri")
    print("=" * 60)

    try:
        print("\n⏳ Memuat vector database...")
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

            print("\n" + "─" * 60)
            print(f"💬 JAWABAN  [{result['llm_used']}]:")
            print(result["answer"])

            print("\n📚 SUMBER KONTEKS:")
            for i, ctx in enumerate(result["contexts"], 1):
                tipe = ctx["source_type"].upper()
                nama = ctx["file_name"] or ctx["source"]
                print(f"  [{i}] Skor: {ctx['score']:.4f} | ({tipe}) {nama}")
                print(f"      {ctx['content'][:120]}...")
            print("─" * 60)

        except Exception as e:
            print(f"❌ Error: {e}")
