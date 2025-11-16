# =============================================
#  ADVANCED MULTI-DOCUMENT RAG WITH FUSION
#  For Laser-Microstructure Interaction Studies
#  Date: November 16, 2025
# =============================================

import streamlit as st
import os
import tempfile
import time
import json
import requests
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from io import BytesIO

# --- LangChain & RAG ---
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.schema import Document
from langchain.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI  # For Arena Model (Grok)

# --- PDF Parsing: marker-pdf (best for scientific PDFs) ---
try:
    from marker.convert import convert_single_pdf
    from marker.models import load_all_models
    MARKER_AVAILABLE = True
except ImportError:
    MARKER_AVAILABLE = False
    st.warning("Install `marker-pdf` for best equation/table/caption support: `pip install marker-pdf`")

# --- Clustering ---
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
SCIBERT_MODEL = "allenai/scibert_scivocab_uncased"  # Domain-specific
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
DEFAULT_LLM = "llama3.1:8b"

# --- Initialize Marker Models (once) ---
@st.cache_resource
def load_marker_models():
    if MARKER_AVAILABLE:
        return load_all_models()
    return None

marker_models = load_marker_models()

# =============================================
#  1. PDF LOADING WITH MARKER (Equations + Tables + Captions)
# =============================================
def load_pdf_with_marker(uploaded_file) -> str:
    """Convert PDF to Markdown using marker-pdf (preserves LaTeX, tables, captions)."""
    if not MARKER_AVAILABLE:
        st.error("marker-pdf not installed. Falling back to PyPDFLoader (no equations/tables).")
        return load_pdf_fallback(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.getbuffer())
        filepath = f.name

    try:
        full_text, images, out_meta = convert_single_pdf(
            filepath, marker_models, batch_multiplier=2, langs=["English"]
        )
        os.remove(filepath)
        return full_text
    except Exception as e:
        st.error(f"Marker failed: {e}. Using fallback.")
        os.remove(filepath)
        return load_pdf_fallback(uploaded_file)

def load_pdf_fallback(uploaded_file) -> str:
    """Fallback: PyPDFLoader (text only)."""
    from langchain_community.document_loaders import PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.getbuffer())
        filepath = f.name
    loader = PyPDFLoader(filepath)
    pages = loader.load()
    os.remove(filepath)
    return "\n".join([p.page_content for p in pages])

# =============================================
#  2. SMART CHUNKING (Markdown + LaTeX Aware)
# =============================================
def chunk_markdown_text(text: str, source_name: str) -> List[Document]:
    """Split Markdown with section awareness, then recursive fallback."""
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    # First: split by headers
    chunks = markdown_splitter.split_text(text)
    final_chunks = []

    for i, chunk in enumerate(chunks):
        subchunks = recursive_splitter.split_text(chunk.page_content)
        for j, sc in enumerate(subchunks):
            metadata = {
                "source_document": source_name,
                "chunk_id": f"{source_name}_chunk_{i}_{j}",
                "header_1": chunk.metadata.get("Header 1"),
                "header_2": chunk.metadata.get("Header 2"),
                "chunk_type": "markdown",
            }
            final_chunks.append(Document(page_content=sc, metadata=metadata))

    return final_chunks

# =============================================
#  3. EMBEDDINGS & VECTOR STORE
# =============================================
@st.cache_resource
def get_embeddings(use_scibert: bool = True):
    if use_scibert:
        return HuggingFaceEmbeddings(model_name=SCIBERT_MODEL)
    return None  # Fallback to Ollama later

@st.cache_resource
def create_vector_store(_chunks, _embeddings):
    try:
        return FAISS.from_documents(_chunks, _embeddings)
    except Exception as e:
        st.error(f"FAISS error: {e}")
        return None

# =============================================
#  4. ENHANCED RETRIEVER (MultiQuery + Compression)
# =============================================
def create_enhanced_retriever(vectorstore, llm, k=6):
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    return MultiQueryRetriever.from_llm(
        retriever=compression_retriever, llm=llm
    )

# =============================================
#  5. FUSION PROMPT (Laser-Microstructure Specific)
# =============================================
FUSION_TEMPLATE = """
You are a materials scientist analyzing laser-microstructure interactions in multicomponent alloys.

CONTEXT (from multiple papers):
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Extract: laser power, scan speed, spot size, energy density
2. Identify: alloy system (HEA, MEA, etc.), phases, composition
3. Describe: microstructure (grain size, dendrites, precipitates, texture)
4. Compare across papers: same alloy? different lasers? conflicting results?
5. Note contradictions, agreements, or complementary data
6. Cite sources: [Source: paper_name.pdf, Chunk ID]

OUTPUT FORMAT:
### Key Findings
- ...

### Cross-Paper Comparison
- ...

### Open Questions
- ...

SYNTHESIZED ANSWER:
"""

FUSION_PROMPT = PromptTemplate(
    template=FUSION_TEMPLATE, input_variables=["context", "question"]
)

# =============================================
#  6. RAG CHAIN (map_reduce for multi-doc synthesis)
# =============================================
def create_fusion_rag_chain(vectorstore, llm, k=6):
    retriever = create_enhanced_retriever(vectorstore, llm, k=k)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": FUSION_PROMPT}
    )

# =============================================
#  7. SEMANTIC CLUSTERING
# =============================================
@st.cache_data
def compute_semantic_clusters(_chunks, _embeddings, n_clusters=5):
    if len(_chunks) < 2:
        return {}
    vectors = _embeddings.embed_documents([c.page_content for c in _chunks])
    k = min(n_clusters, len(vectors))
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(vectors)

    clusters = {}
    for label in range(k):
        idxs = [i for i, l in enumerate(labels) if l == label]
        chunks_in_cluster = [_chunks[i] for i in idxs]
        sources = list(set([c.metadata["source_document"] for c in chunks_in_cluster]))
        sample = chunks_in_cluster[0].page_content[:200]
        clusters[f"Cluster {label}"] = {
            "size": len(idxs),
            "sources": sources,
            "sample": sample,
            "keywords": extract_keywords(sample)  # simple
        }
    return clusters

def extract_keywords(text: str) -> str:
    words = ["laser", "microstructure", "alloy", "grain", "phase", "power", "speed", "HEA"]
    return ", ".join([w for w in words if w in text.lower()])

# =============================================
#  8. RESPONSE FORMATTING WITH CLICKABLE SOURCES
# =============================================
def format_response_with_sources(response, question):
    answer = response.get("result", "No answer.")
    docs = response.get("source_documents", [])

    formatted = f"**Question:** {question}\n\n"
    formatted += f"**Synthesized Answer:**\n{answer}\n\n"

    if docs:
        formatted += "**Retrieved Sources (click to expand):**\n"
        for i, doc in enumerate(docs):
            src = doc.metadata.get("source_document", "Unknown")
            chunk_id = doc.metadata.get("chunk_id", "N/A")
            header = doc.metadata.get("header_1") or doc.metadata.get("header_2") or "No Header"
            with st.expander(f"Source {i+1}: {src} | {header} | Chunk {chunk_id}"):
                st.code(doc.page_content[:1500] + ("..." if len(doc.page_content) > 1500 else ""), language="markdown")

    return formatted

# =============================================
#  9. OLLAMA MODEL DETECTION
# =============================================
def get_ollama_models():
    try:
        r = requests.get(f"{st.session_state.ollama_base_url}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except:
        pass
    return ["llama3.1:8b", "gemma3:latest", "mistral:7b"]

# =============================================
#  10. STREAMLIT UI
# =============================================
def main():
    st.set_page_config(page_title="Laser-Microstructure RAG Fusion", layout="wide")
    st.title("Laser-Microstructure Interaction RAG")
    st.markdown("*Multi-document synthesis with equations, tables, and domain-aware embeddings.*")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")
        st.session_state.ollama_base_url = st.text_input("Ollama URL", OLLAMA_BASE_URL)

        # Model Selection
        models = get_ollama_models()
        models += ["Arena Model (Grok)"]
        default_idx = 0 if DEFAULT_LLM in models else 0
        st.session_state.llm_model = st.selectbox("LLM", models, index=default_idx)

        st.session_state.use_scibert = st.checkbox("Use SciBERT Embeddings (Recommended)", True)
        st.session_state.retrieval_k = st.slider("Chunks to Retrieve", 3, 15, 6)

        if st.button("Clear Cache & Reset"):
            st.cache_resource.clear()
            st.success("Cache cleared!")

    # --- Upload ---
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.get("processed_files", set())]
        if new_files:
            st.session_state.processed_files = st.session_state.get("processed_files", set())
            st.session_state.existing_chunks = []
            st.session_state.messages = []
            with st.spinner(f"Processing {len(new_files)} PDFs..."):
                all_chunks = []
                for f in new_files:
                    with st.status(f"Processing {f.name}..."):
                        md_text = load_pdf_with_marker(f)
                        chunks = chunk_markdown_text(md_text, f.name)
                        all_chunks.extend(chunks)
                        st.write(f"→ {len(chunks)} chunks")
                    st.session_state.processed_files.add(f.name)
                st.session_state.existing_chunks = all_chunks

                # Embeddings
                embeddings = get_embeddings(st.session_state.use_scibert)
                if not embeddings and MARKER_AVAILABLE:
                    st.warning("Using Ollama embeddings as fallback.")
                    from langchain_community.embeddings import OllamaEmbeddings
                    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=st.session_state.ollama_base_url)

                st.session_state.vectorstore = create_vector_store(all_chunks, embeddings)
                st.success("Ingestion complete!")

                # Semantic Clusters
                if len(all_chunks) > 5:
                    clusters = compute_semantic_clusters(all_chunks, embeddings)
                    with st.expander("Semantic Clusters (Themes)"):
                        st.json(clusters, expanded=False)

    # --- Chat ---
    if st.session_state.get("vectorstore"):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "query_log" not in st.session_state:
            st.session_state.query_log = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # LLM
        if st.session_state.llm_model == "Arena Model (Grok)":
            llm = ChatOpenAI(model="grok-beta", base_url="https://api.x.ai/v1", api_key=os.getenv("XAI_API_KEY"))
        else:
            llm = Ollama(model=st.session_state.llm_model, base_url=st.session_state.ollama_base_url)

        if "qa_chain" not in st.session_state:
            st.session_state.qa_chain = create_fusion_rag_chain(
                st.session_state.vectorstore, llm, k=st.session_state.retrieval_k
            )

        if prompt := st.chat_input("Ask about laser parameters, microstructure, or cross-paper comparisons..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                with st.spinner("Retrieving + synthesizing..."):
                    try:
                        response = st.session_state.qa_chain.invoke({"query": prompt})
                        formatted = format_response_with_sources(response, prompt)
                        placeholder.markdown(formatted)

                        # Log
                        log_entry = {
                            "timestamp": datetime.now().isoformat(),
                            "question": prompt,
                            "answer": response["result"],
                            "sources": [d.metadata["source_document"] for d in response.get("source_documents", [])]
                        }
                        st.session_state.query_log.append(log_entry)

                    except Exception as e:
                        st.error(f"Error: {e}")

            st.session_state.messages.append({"role": "assistant", "content": formatted})

        # Export Log
        if st.session_state.query_log:
            st.download_button(
                "Export Query Log (JSON)",
                data=json.dumps(st.session_state.query_log, indent=2),
                file_name=f"rag_log_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )

    else:
        st.info("Upload PDFs to begin.")

if __name__ == "__main__":
    main()
