import os
import pickle
import re
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Thread
from tkinter import filedialog
from typing import List

import faiss
import fitz  # PyMuPDF
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

chat_history = []  # Chat memory


def clean_citations(text: str) -> str:
    text = re.sub(r'\[[^\]]*\]', '', text)  # Square brackets
    text = re.sub(r'\((?:[A-Za-z]+,?\s*\d{4}[;,\s]*)+\)', '', text)  # Author-year
    return re.sub(r'\s+', ' ', text).strip()


def clean_excerpt(text: str) -> str:
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\((?:[A-Za-z]+(?: et al\.)?,?\s*\d{4}[;,\s]*)+\)', '', text)
    text = re.sub(r'\b(?:PII|doi|DOI):?\s*[\w\-\./\(\)]+', '', text, flags=re.I)
    text = re.sub(r'\b[Pp]age\s*\d+\b', '', text)
    text = re.sub(r'\b[\w\-. ]+\.(indd|pdf|docx|txt)\b', '', text, flags=re.I)
    text = re.sub(r'\b(Fig\.?|Figure|Table)\s*\d+\b', '', text)
    text = re.sub(r'\(\s*(ref|Ref)?\s*\d+\s*\)', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_single_pdf(file_path: Path, chunk_size: int, overlap: int) -> List[dict]:
    try:
        doc = fitz.open(file_path)
        title = doc.metadata.get("title") or file_path.stem
        chunks = []

        for i, page in enumerate(doc):
            text = clean_excerpt(page.get_text().strip())
            if text:
                # Split long pages into chunks
                for j in range(0, len(text), chunk_size):
                    chunk = text[j:j + chunk_size]
                    chunks.append({"text": chunk, "source": f"{title} - Page {i + 1}"})

        return chunks
    except Exception as e:
        print(f"❌ Error reading {file_path.name}: {e}")
        return []


def load_documents(directory: str, chunk_size: int = 1000, overlap: int = 200) -> List[dict]:
    pdf_paths = list(Path(directory).glob("*.pdf"))
    documents = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(load_single_pdf, path, chunk_size, overlap): path for path in pdf_paths}
        for future in tqdm(futures, desc="📄 Loading PDFs", unit="file"):
            try:
                documents.extend(future.result())
            except Exception as e:
                print(f"❌ Error processing {futures[future].name}: {e}")

    return documents


def build_vector_store(documents: List[dict], cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_emb_path = os.path.join(cache_dir, "embeddings.npy")
    cache_docs_path = os.path.join(cache_dir, "documents.pkl")

    if os.path.exists(cache_emb_path) and os.path.exists(cache_docs_path):
        print("💾 Loading cached embeddings and documents...")
        embeddings = np.load(cache_emb_path)
        with open(cache_docs_path, "rb") as f:
            documents_cached = pickle.load(f)

        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index, encoder, embeddings, documents_cached

    print("🔄 Computing embeddings...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [doc["text"] for doc in documents]
    embeddings = encoder.encode(texts, batch_size=16, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save to cache
    np.save(cache_emb_path, embeddings)
    with open(cache_docs_path, "wb") as f:
        pickle.dump(documents, f)

    return index, encoder, embeddings, documents


def retrieve(query: str, index, encoder, documents: List[dict], k=10) -> List[dict]:
    query_vec = encoder.encode(query).reshape(1, -1).astype("float32")
    _, indices = index.search(query_vec, k)
    return [documents[i] for i in indices[0]]


def extract_top_sentences(text: str, query: str, encoder, top_n=2, similarity_threshold=0.4) -> str:
    clean_text = clean_citations(text)
    sentences = [s.strip() for s in clean_text.replace("\n", " ").split(".") if len(s.strip()) > 20]

    if not sentences:
        return ""

    sentence_embeddings = encoder.encode(sentences)
    query_embedding = encoder.encode([query])
    scores = cosine_similarity(query_embedding, sentence_embeddings)[0]

    filtered = [(i, score) for i, score in enumerate(scores) if score >= similarity_threshold]
    if not filtered:
        return ""

    top_indices = [i for i, _ in sorted(filtered, key=lambda x: x[1], reverse=True)[:top_n]]
    return ". ".join(sentences[i] for i in top_indices) + "."


def detect_short_answer_request(question: str) -> bool:
    short_keywords = ["recap", "summary", "summarize", "short answer", "in brief", "tl;dr", "briefly", "quick overview"]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in short_keywords)


def generate_answer(docs: List[dict], question: str, model, tokenizer, device, encoder,
                    chat_history: List[tuple] = None, max_length=500) -> tuple:
    chat_history = chat_history or []

    cleaned_chunks = []
    used_sources = set()

    for doc in docs:
        summary = extract_top_sentences(doc["text"], question, encoder, top_n=2)
        if summary:
            cleaned_chunks.append(summary)
            used_sources.add(doc["source"])

    if not cleaned_chunks:
        return None, set()

    context = "\n".join(f"- {chunk}" for chunk in cleaned_chunks)

    history_str = ""
    for q, a in chat_history[-3:]:
        history_str += f"User: {q}\nAssistant: {a}\n"

    short_answer = detect_short_answer_request(question)

    # Core prompt template
    instructions = f"""You are a marine science assistant. Use the research excerpts below to answer the **final user question only**.

    **Instructions:**
    - Base your answer only on the excerpts below.
    - Do not answer earlier questions again.
    {"- This is a request for a SHORT answer. Limit your response to 2–3 sentences." if short_answer else ""}
    - Do not repeat the same phrase or idea.
    - Write in clear, plain English.
    - Be concise, factual, and avoid repetition.
    - Support claims with up to 3 source citations in the format: [Source 1].
    - Only place citations after specific factual claims.
    - Do **not** include file names, page numbers, or citation metadata in your answer.
    """

    prompt = f"""{instructions}
    
    Chat history (for reference only):
    {history_str}
    
    Excerpts:
    {context}
    
    Question:
    {question}
    
    Answer:
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_length,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
    )

    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
    answer_only = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return answer_only, used_sources


def create_app(documents, index, encoder, model, tokenizer, device):
    root = tk.Tk()
    root.title("Coral Reef Assistant")

    # Frame for chat + scrollbar
    chat_frame = tk.Frame(root)
    chat_frame.pack(padx=10, pady=10)

    scrollbar = tk.Scrollbar(chat_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    chat = tk.Text(chat_frame, wrap="word", height=25, width=90, yscrollcommand=scrollbar.set)
    chat.pack(side=tk.LEFT, fill=tk.BOTH)
    scrollbar.config(command=chat.yview)

    # Text tag styles
    chat.tag_configure("user", foreground="blue")
    chat.tag_configure("bot", foreground="green")
    chat.tag_configure("source", foreground="darkorange")

    # Entry + Ask button + Thinking label
    input_frame = tk.Frame(root)
    input_frame.pack(pady=5)

    entry = tk.Entry(input_frame, width=70)
    entry.grid(row=0, column=0, padx=5)
    entry.focus()

    thinking_label = tk.Label(input_frame, text="Thinking...", fg="gray")
    thinking_label.grid(row=0, column=2, padx=5)
    thinking_label.grid_remove()  # Hide initially

    chat_history = []

    def ask():
        question = entry.get()
        if not question.strip():
            return
        entry.delete(0, tk.END)

        def process():
            thinking_label.grid()  # Show thinking label

            # Insert user question
            chat.insert(tk.END, "\nYou asked:\n", "user")
            chat.insert(tk.END, question + "\n\n", "user")
            chat.insert(tk.END, "Bot answered:\n", "bot")
            chat.see(tk.END)

            # RAG logic to get answer
            docs = retrieve(question, index, encoder, documents, k=5)
            raw_answer, used_sources = generate_answer(docs, question, model, tokenizer, device, encoder, chat_history)

            if not raw_answer or not raw_answer.strip():
                fallback_query = question + " threats examples consequences"
                docs = retrieve(fallback_query, index, encoder, documents, k=10)
                raw_answer, used_sources = generate_answer(docs, fallback_query, model, tokenizer, device, encoder,
                                                           chat_history)

                if not raw_answer or not raw_answer.strip():
                    raw_answer = "I'm sorry, I couldn't find relevant information to answer that question."

            cleaned_answer = clean_citations(raw_answer)
            chat_history.append((question, cleaned_answer))

            # Insert final answer
            chat.insert(tk.END, cleaned_answer + "\n\n", "bot")

            # Insert sources
            sources = sorted(used_sources)
            if sources:
                formatted_sources = "\n".join(f"– {s}" for s in sources)
                chat.insert(tk.END, "Sources:\n", "source")
                chat.insert(tk.END, formatted_sources + "\n", "source")

            chat.insert(tk.END, "-" * 50 + "\n")
            chat.see(tk.END)

            thinking_label.grid_remove()  # Hide thinking label

        Thread(target=process, daemon=True).start()

    entry.bind("<Return>", lambda e: ask())
    button = tk.Button(input_frame, text="Ask", command=ask)
    button.grid(row=0, column=1, padx=5)

    root.mainloop()


def main():
    print("🔄 Loading PDFs...")
    documents = load_documents("papers", chunk_size=1000, overlap=200)
    if not documents:
        print("❌ No documents found.")
        return

    print("🔄 Building FAISS index...")
    index, encoder, embeddings, documents = build_vector_store(documents)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/phi-2"
    print(f"🔄 Loading {model_path} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    create_app(documents, index, encoder, model, tokenizer, device)


if __name__ == "__main__":
    main()
