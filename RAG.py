import os
import re
import torch
import fitz  # PyMuPDF
import numpy as np
import faiss
import tkinter as tk
from pathlib import Path
from tqdm import tqdm
from typing import List
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

chat_history = []  # Chat memory


def clean_citations(text: str) -> str:
    text = re.sub(r'\[[^\]]*\]', '', text)  # Square brackets
    text = re.sub(r'\((?:[A-Za-z]+,?\s*\d{4}[;,\s]*)+\)', '', text)  # Author-year
    return re.sub(r'\s+', ' ', text).strip()


def chunk_text_by_paragraphs(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    chunks = []
    current_chunk_parts = []
    current_length = 0
    i = 0

    while i < len(paragraphs):
        para = paragraphs[i]
        if current_length + len(para) + 2 <= chunk_size:
            current_chunk_parts.append(para)
            current_length += len(para) + 2
            i += 1
        else:
            chunks.append("\n\n".join(current_chunk_parts).strip())
            # handle overlap
            overlap_parts = []
            overlap_len = 0
            j = i - 1
            while j >= 0 and overlap_len < overlap:
                overlap_parts.insert(0, paragraphs[j])
                overlap_len += len(paragraphs[j])
                j -= 1
            current_chunk_parts = overlap_parts
            current_length = sum(len(p) + 2 for p in current_chunk_parts)

    if current_chunk_parts:
        chunks.append("\n\n".join(current_chunk_parts).strip())
    return chunks

def clean_excerpt(text: str) -> str:
    # Remove square bracket citations like [12], [Smith et al., 2020]
    text = re.sub(r'\[[^\]]*\]', '', text)

    # Remove author-year citations like (Smith, 2020; Jones, 2019)
    text = re.sub(r'\((?:[A-Za-z]+(?: et al\.)?,?\s*\d{4}[;,\s]*)+\)', '', text)

    # Remove citations with doi or PII references (e.g. PII: 0022-0981(86)90245-5)
    text = re.sub(r'\b(?:PII|doi|DOI):?\s*[\w\-\./\(\)]+', '', text, flags=re.I)

    # Remove page numbers "Page 12" or "page 12"
    text = re.sub(r'\b[Pp]age\s*\d+\b', '', text)

    # Remove file names like "05 Masuda FB 108(2).indd"
    # A rough heuristic: strings with file extensions
    text = re.sub(r'\b[\w\-. ]+\.(indd|pdf|docx|txt)\b', '', text, flags=re.I)

    # Remove figure/table references "Fig. 1", "Table 2"
    text = re.sub(r'\b(Fig\.?|Figure|Table)\s*\d+\b', '', text)

    # Remove parentheses with only numbers or identifiers inside e.g., "(12345)", "(ref 12)"
    text = re.sub(r'\(\s*(ref|Ref)?\s*\d+\s*\)', '', text)

    # Remove multiple spaces, newlines, tabs etc.
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def clean_source_name(source: str) -> str:
    source = re.sub(r'\b[Pp]age\s*\d+\b', '', source)
    source = re.sub(r'\b[\w\-. ]+\.(indd|pdf|docx|txt)\b', '', source, flags=re.I)
    source = re.sub(r'\s+', ' ', source).strip()
    return source


def load_single_pdf(file_path: Path, chunk_size: int, overlap: int) -> List[dict]:
    try:
        doc = fitz.open(file_path)
        title = doc.metadata.get("title") or file_path.stem
        chunks = []

        for i, page in enumerate(doc):
            text = clean_excerpt(page.get_text().strip())  # <-- use clean_excerpt here
            if text:
                # Split long pages into chunks
                for j in range(0, len(text), chunk_size):
                    chunk = text[j:j + chunk_size]
                    chunks.append({"text": chunk, "source": f"{title} - Page {i+1}"})

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


def build_vector_store(documents: List[dict]):
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [doc["text"] for doc in documents]
    embeddings = encoder.encode(texts, batch_size=16, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, encoder, embeddings


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

    # Filter out weak matches
    filtered = [(i, score) for i, score in enumerate(scores) if score >= similarity_threshold]
    if not filtered:
        return ""

    # Sort by relevance and return top sentences
    top_indices = [i for i, _ in sorted(filtered, key=lambda x: x[1], reverse=True)[:top_n]]
    return ". ".join(sentences[i] for i in top_indices) + "."



def generate_answer(docs: List[dict], question: str, model, tokenizer, device, encoder,
                    chat_history: List[tuple] = None, max_length=256) -> tuple:
    chat_history = chat_history or []

    cleaned_chunks = []
    used_sources = set()

    for doc in docs:
        summary = extract_top_sentences(doc["text"], question, encoder, top_n=2)
        if summary:
            cleaned_chunks.append(summary)
            used_sources.add(doc["source"])

    if not cleaned_chunks:
        return None, set()  # We'll handle fallback outside

    context = "\n".join(f"- {chunk}" for chunk in cleaned_chunks)

    # Build chat history string (last 3 turns)
    history_str = ""
    for q, a in chat_history[-3:]:
        history_str += f"User: {q}\nAssistant: {a}\n"

    prompt = f"""You are an expert marine biologist assistant. When answering, ignore file names, page numbers, 
    and citations, and focus on the scientific findings described. Summarize facts clearly. Provide direct, clear, 
    and well-supported answers without step-by-step reasoning unless explicitly asked.
    For each claim in your answer, cite the relevant document (e.g. '[Source 1]') and list only up to 3 unique sources at the end.

Chat history:
{history_str}

Excerpts:
{context}

Question: {question}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_length,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
    )

    # The model output contains input tokens + generated tokens.
    # So slice output to get only newly generated tokens after input length
    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]

    answer_only = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return answer_only, used_sources



def create_app(documents, index, encoder, model, tokenizer, device):
    def ask():
        question = entry.get()
        entry.delete(0, tk.END)

        def process():
            # First try normal retrieval
            docs = retrieve(question, index, encoder, documents, k=5)
            raw_answer, used_sources = generate_answer(docs, question, model, tokenizer, device, encoder, chat_history)

            if raw_answer is None or raw_answer.strip() == "":
                # Fallback: Expand query with some keywords and increase k
                expanded_query = question + " threats examples consequences"
                docs = retrieve(expanded_query, index, encoder, documents, k=10)
                raw_answer, used_sources = generate_answer(docs, expanded_query, model, tokenizer, device, encoder,
                                                           chat_history)

                if raw_answer is None or raw_answer.strip() == "":
                    raw_answer = "I'm sorry, I couldn't find relevant information to answer that question."

            cleaned_answer = clean_citations(raw_answer)

            # Save to chat history
            chat_history.append((question, cleaned_answer))

            # Format sources
            sources = sorted(used_sources)
            if sources:
                formatted_sources = "\n".join(f"– {s}" for s in sources)
                full_response = f"\nYou: {question}\nBot: {raw_answer}\n\nSources:\n{formatted_sources}\n\n"
            else:
                full_response = f"\nYou: {question}\nBot: {raw_answer}\n\n"

            chat.insert(tk.END, full_response)
            chat.see(tk.END)

        Thread(target=process).start()



    root = tk.Tk()
    root.title("Coral Reef Assistant")

    chat = tk.Text(root, wrap="word", height=25, width=90)
    chat.pack(padx=10, pady=10)

    entry = tk.Entry(root, width=80)
    entry.pack(side=tk.LEFT, padx=10)
    entry.bind("<Return>", lambda e: ask())

    button = tk.Button(root, text="Ask", command=ask)
    button.pack(side=tk.LEFT, padx=5)

    root.mainloop()



def main():
    print("🔄 Loading PDFs...")
    documents = load_documents("papers", chunk_size=1000, overlap=200)
    if not documents:
        print("❌ No documents found.")
        return

    print("🔄 Building FAISS index...")
    index, encoder, _ = build_vector_store(documents)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/phi-2"
    print(f"🔄 Loading {model_path} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)

    # Set pad_token_id if missing
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    create_app(documents, index, encoder, model, tokenizer, device)


if __name__ == "__main__":
    main()
