import os
from pathlib import Path
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
from typing import List, Tuple
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# Load documents from PDFs with chunking and extract titles
def extract_title_from_text(text: str) -> str:
    lines = text.strip().splitlines()
    for line in lines[:5]:
        clean_line = line.strip()
        if len(clean_line.split()) >= 3 and len(clean_line) > 10:
            return clean_line
    return "Untitled Document"


def load_documents(directory: str) -> Tuple[List[Document], List[str]]:
    documents = []
    sources = []

    for file_path in Path(directory).glob("*.pdf"):
        try:
            doc = fitz.open(file_path)
            title = None

            # Try PDF metadata title first
            if doc.metadata and doc.metadata.get("title"):
                title = doc.metadata["title"].strip()

            # Else first line of first page
            if not title and len(doc) > 0:
                first_page_text = doc[0].get_text("text")
                title = extract_title_from_text(first_page_text)

            if not title:
                title = file_path.stem  # fallback to filename

            full_text = "\n".join(page.get_text("text") for page in doc)
            chunk_size = 1000
            for i in range(0, len(full_text), chunk_size):
                chunk = full_text[i:i + chunk_size]
                documents.append(Document(page_content=chunk, metadata={"source": title}))
            sources.append(title)

        except Exception as e:
            print(f"❌ Error reading {file_path.name}: {e}")

    return documents, sources


# Build vector store with sentence-transformers and Faiss
def build_vector_store(documents: List[Document]):
    print("🔄 Building vector store...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = []
    for doc in tqdm(documents, desc="Encoding"):
        text = doc.page_content.strip()
        if text:
            emb = encoder.encode(text, convert_to_tensor=False)
            embeddings.append(emb)

    if not embeddings:
        raise ValueError("No valid text chunks found. Please check your documents.")

    import numpy as np
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, encoder, embeddings


# Retrieve top k docs
def retrieve(query: str, index, encoder, documents: List[Document], k=3) -> List[Document]:
    query_embedding = encoder.encode(query, convert_to_tensor=False).reshape(1, -1).astype("float32")
    D, I = index.search(query_embedding, k)
    retrieved = [documents[i] for i in I[0]]
    return retrieved


# Relevance filter: pick top sentences from context relevant to query using TF-IDF
def find_best_sentences(context: str, query: str, top_n=3) -> str:
    sentences = [s.strip() for s in context.replace("\n", " ").split(".") if len(s.strip()) > 10]
    if not sentences:
        return context
    vect = TfidfVectorizer().fit([query] + sentences)
    query_vec = vect.transform([query])
    sents_vec = vect.transform(sentences)
    scores = cosine_similarity(query_vec, sents_vec).flatten()
    ranked_sentences = [s for _, s in sorted(zip(scores, sentences), reverse=True)]
    return ". ".join(ranked_sentences[:top_n])


# Generate answer using Hugging Face QA pipeline
def generate_answer(context: str, question: str, qa_pipeline) -> str:
    try:
        result = qa_pipeline(question=question, context=context)
        return result.get("answer", "Sorry, I couldn't find an answer.")
    except Exception as e:
        return f"Error during answer generation: {e}"


def main():
    print("🔄 Loading documents...")
    documents, sources = load_documents("papers")

    if not documents:
        print("❌ No valid documents loaded. Exiting.")
        return

    index, encoder, _ = build_vector_store(documents)

    print("✅ Vector store ready.")

    # Load QA pipeline model
    model_path = r"models/bert-qa"  # Your local path
    qa_pipeline = pipeline("question-answering", model=model_path, tokenizer=model_path)

    while True:
        question = input("\n❓ Ask a question (or 'exit'): ")
        if question.lower() == "exit":
            break

        # Retrieve top 3 chunks
        retrieved_docs = retrieve(question, index, encoder, documents, k=3)

        # Build combined context from retrieved chunks
        combined_context = "\n---\n".join(doc.page_content for doc in retrieved_docs)

        # Filter context to most relevant sentences
        filtered_context = find_best_sentences(combined_context, question, top_n=5)

        # Generate answer with filtered context
        answer = generate_answer(filtered_context, question, qa_pipeline)

        print("\n🧠 Answer:")
        print(answer)

        # Show sources for retrieved docs
        print("\n📄 Sources:")
        unique_sources = list(dict.fromkeys(doc.metadata.get("source", "Unknown") for doc in retrieved_docs))
        for src in unique_sources:
            print("-", src)


if __name__ == "__main__":
    main()
