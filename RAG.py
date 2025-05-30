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
    short_keywords = [
        "recap", "summary", "summarize", "short answer",
        "in brief", "tl;dr", "briefly", "quick overview",
        "quick summary", "simplify", "just the key points"
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in short_keywords)


def detect_bullet_request(question: str) -> bool:
    bullet_keywords = [
        "list", "types of", "examples of", "show me", "enumerate",
        "identify", "bullet points", "categories of", "name some",
        "advantages and disadvantages", "pros and cons"
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in bullet_keywords)


def generate_answer(docs: List[dict], question: str, model, tokenizer, device, encoder,
                    chat_history: List[tuple] = None, max_length=500) -> tuple:
    chat_history = chat_history or []
    print(f"[DEBUG] Inside generate_answer, got question: '{question}'")
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
    bullet_list = detect_bullet_request(question)

    # Instruction customization
    if bullet_list:
        list_format_hint = ""
        if "pros and cons" in question.lower():
            list_format_hint = (
                "- Begin with '**Pros:**' followed by a bulleted list.\n"
                "- Then include '**Cons:**' followed by another bulleted list.\n"
            )
        answer_instruction = (
            "- This is a request for a BULLET LIST.\n"
            "- Use line breaks between items.\n"
            "- List distinct items with 1–2 line explanations.\n"
            "- Begin each item with a dash (-).\n"
            f"{list_format_hint}"
        )
        output_length = 180
    elif short_answer:
        answer_instruction = (
            "- This is a request for a SHORT answer.\n"
            "- Respond in 2–3 short sentences MAXIMUM.\n"
            "- Use plain language and focus only on the main ideas.\n"
        )
        output_length = 120
    else:
        answer_instruction = ""
        output_length = max_length

    instructions = f"""You are a marine science assistant. Use the research excerpts below to answer the **final user question only**.

        **Instructions:**
        - Base your answer only on the excerpts below.
        - Do not answer earlier questions again.
        {answer_instruction}- Do not repeat the same phrase or idea.
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

    print(f"[DEBUG] Final prompt:\n{prompt[:500]}")  # First 500 chars

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=output_length,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
    )

    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
    answer_only = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return answer_only, used_sources




def create_app(documents, index, encoder, model, tokenizer, device):
    root = tk.Tk()
    root.title("Coral Reef Assistant")

    # Default colors for light mode
    LIGHT_BG = "white"
    LIGHT_FG = "black"
    LIGHT_USER = "blue"
    LIGHT_BOT = "green"
    LIGHT_SOURCE = "darkorange"

    # Colors for dark mode
    DARK_BG = "#2e2e2e"
    DARK_FG = "#e0e0e0"
    DARK_USER = "#539bf5"
    DARK_BOT = "#7cfc00"
    DARK_SOURCE = "#ffa500"

    current_theme = {
        "bg": LIGHT_BG,
        "fg": LIGHT_FG,
        "user": LIGHT_USER,
        "bot": LIGHT_BOT,
        "source": LIGHT_SOURCE,
    }

    # Theme toggling
    dark_mode_enabled = [False]  # use list to make mutable in nested scope

    def apply_theme():
        root.config(bg=current_theme["bg"])
        chat.config(bg=current_theme["bg"], fg=current_theme["fg"], insertbackground=current_theme["fg"])
        input_frame.config(bg=current_theme["bg"])
        entry.config(bg=current_theme["bg"], fg=current_theme["fg"], insertbackground=current_theme["fg"])
        thinking_label.config(bg=current_theme["bg"], fg="gray")
        button.config(bg=current_theme["bg"], fg=current_theme["fg"], activebackground=current_theme["bg"])
        toggle_button.config(bg=current_theme["bg"], fg=current_theme["fg"], activebackground=current_theme["bg"])
        chat.tag_configure("user", foreground=current_theme["user"])
        chat.tag_configure("bot", foreground=current_theme["bot"])
        chat.tag_configure("source", foreground=current_theme["source"])

    def toggle_dark_mode():
        if dark_mode_enabled[0]:
            # Switch to light mode
            current_theme.update(
                bg=LIGHT_BG, fg=LIGHT_FG,
                user=LIGHT_USER, bot=LIGHT_BOT, source=LIGHT_SOURCE
            )
            toggle_button.config(text="Enable Dark Mode")
            dark_mode_enabled[0] = False
        else:
            # Switch to dark mode
            current_theme.update(
                bg=DARK_BG, fg=DARK_FG,
                user=DARK_USER, bot=DARK_BOT, source=DARK_SOURCE
            )
            toggle_button.config(text="Disable Dark Mode")
            dark_mode_enabled[0] = True
        apply_theme()

    # Chat frame
    chat_frame = tk.Frame(root, bg=current_theme["bg"])
    chat_frame.pack(padx=10, pady=10)

    scrollbar = tk.Scrollbar(chat_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    chat = tk.Text(chat_frame, wrap="word", height=25, width=90, yscrollcommand=scrollbar.set,
                   bg=current_theme["bg"], fg=current_theme["fg"], insertbackground=current_theme["fg"])
    chat.pack(side=tk.LEFT, fill=tk.BOTH)
    scrollbar.config(command=chat.yview)

    chat.tag_configure("user", foreground=current_theme["user"])
    chat.tag_configure("bot", foreground=current_theme["bot"])
    chat.tag_configure("source", foreground=current_theme["source"])

    # Entry area
    input_frame = tk.Frame(root, bg=current_theme["bg"])
    input_frame.pack(pady=5)

    entry = tk.Entry(input_frame, width=70,
                     bg=current_theme["bg"], fg=current_theme["fg"], insertbackground=current_theme["fg"])
    entry.grid(row=0, column=0, padx=5)
    entry.focus()

    thinking_label = tk.Label(input_frame, text="Thinking...", fg="gray", bg=current_theme["bg"])
    thinking_label.grid(row=0, column=2, padx=5)
    thinking_label.grid_remove()

    chat_history = []

    def ask():
        question = entry.get().strip()
        if not question:
            return
        entry.delete(0, tk.END)

        def process(user_question):
            thinking_label.grid()
            print(f"[DEBUG] question: '{user_question}'")

            chat.insert(tk.END, "\nYou asked:\n", "user")
            chat.insert(tk.END, user_question + "\n\n", "user")
            chat.insert(tk.END, "Bot answered:\n", "bot")
            chat.see(tk.END)

            docs = retrieve(user_question, index, encoder, documents, k=5)
            print(f"[DEBUG] Calling generate_answer with question: '{user_question}'")
            raw_answer, used_sources = generate_answer(docs, user_question, model, tokenizer, device, encoder,
                                                       chat_history)

            # Fallback if needed
            if not raw_answer or not raw_answer.strip():
                fallback_query = user_question + " threats examples consequences"
                docs = retrieve(fallback_query, index, encoder, documents, k=10)
                raw_answer, used_sources = generate_answer(docs, fallback_query, model, tokenizer, device, encoder,
                                                           chat_history)

                if not raw_answer or not raw_answer.strip():
                    raw_answer = "I'm sorry, I couldn't find relevant information to answer that question."

            cleaned_answer = clean_citations(raw_answer)
            chat_history.append((user_question, cleaned_answer))

            words = cleaned_answer.split()
            idx = 0

            def stream_words():
                nonlocal idx
                if idx < len(words):
                    chat.insert(tk.END, words[idx] + " ", "bot")
                    idx += 1
                    chat.see(tk.END)
                    chat.after(50, stream_words)
                else:
                    chat.insert(tk.END, "\n\n", "bot")
                    if used_sources:
                        sources = sorted(used_sources)
                        formatted_sources = "\n".join(f"– {s}" for s in sources)
                        chat.insert(tk.END, "Sources:\n", "source")
                        chat.insert(tk.END, formatted_sources + "\n", "source")
                    chat.insert(tk.END, "-" * 50 + "\n")
                    chat.see(tk.END)
                    thinking_label.grid_remove()

            stream_words()

        # ✅ Pass the captured question to the thread
        Thread(target=lambda: process(question), daemon=True).start()

    entry.bind("<Return>", lambda e: ask())
    button = tk.Button(input_frame, text="Ask", command=ask,
                       bg=current_theme["bg"], fg=current_theme["fg"], activebackground=current_theme["bg"])
    button.grid(row=0, column=1, padx=5)

    # Dark mode toggle button
    toggle_button = tk.Button(root, text="Enable Dark Mode", command=toggle_dark_mode,
                              bg=current_theme["bg"], fg=current_theme["fg"], activebackground=current_theme["bg"])
    toggle_button.pack(anchor="ne", padx=10, pady=5)

    apply_theme()
    root.mainloop()


def main():
    folders = ["papers", "noaa_reports"]  # Add more folders here as needed

    print("🔄 Loading PDFs...")
    all_documents = []
    for folder in folders:
        print(f"📂 Loading from: {folder}")
        docs = load_documents(folder, chunk_size=1000, overlap=200)
        all_documents.extend(docs)

    if not all_documents:
        print("❌ No documents found.")
        return

    print("🔄 Building FAISS index...")
    index, encoder, embeddings, all_documents = build_vector_store(all_documents)

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

    create_app(all_documents, index, encoder, model, tokenizer, device)



if __name__ == "__main__":
    main()
