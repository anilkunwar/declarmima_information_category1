# =============================================
# LASER-MICROSTRUCTURE RAG + FUSION (FINAL v2.1)
# LangChain 0.2+ | CPU-SAFE | Pydantic-FIXED
# Updated: November 16, 2025 | PL | 03:40 AM CET
# =============================================

import streamlit as st
import os
import tempfile
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
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader

# --- Optional: Grok (xAI) ---
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

# --- PDF Parsers: Marker Alternatives ---
try:
    import pymupdf4llm
    PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    PYMUPDF4LLM_AVAILABLE = False
    st.warning("Install PyMuPDF4LLM: `pip install pymupdf4llm`")

try:
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    st.warning("Install Unstructured: `pip install \"unstructured[pdf]\"`")

try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False
    st.warning("Install MarkItDown: `pip install markitdown[all]`")

# --- Clustering ---
from sklearn.cluster import KMeans

# --- Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434"
SCIBERT_MODEL = "allenai/scibert_scivocab_uncased"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
DEFAULT_LLM = "llama3.1:8b"

# =============================================
# 1. PDF → MARKDOWN
# =============================================
def load_pdf_with_pymupdf4llm(uploaded_file) -> str:
    if not PYMUPDF4LLM_AVAILABLE: return load_pdf_fallback(uploaded_file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.getbuffer())
        filepath = f.name
    try:
        md = pymupdf4llm.to_markdown(filepath)
        os.remove(filepath)
        return md
    except Exception as e:
        st.error(f"PyMuPDF4LLM failed: {e}")
        os.remove(filepath)
        return load_pdf_fallback(uploaded_file)

def load_pdf_with_unstructured(uploaded_file) -> str:
    if not UNSTRUCTURED_AVAILABLE: return load_pdf_fallback(uploaded_file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.getbuffer())
        filepath = f.name
    try:
        elements = partition_pdf(filepath, strategy="hi_res")
        md = "\n\n".join([el.text for el in elements if hasattr(el, "text")])
        os.remove(filepath)
        return md
    except Exception as e:
        st.error(f"Unstructured failed: {e}")
        os.remove(filepath)
        return load_pdf_fallback(uploaded_file)

def load_pdf_with_markitdown(uploaded_file) -> str:
    if not MARKITDOWN_AVAILABLE: return load_pdf_fallback(uploaded_file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.getbuffer())
        filepath = f.name
    try:
        md = MarkItDown()
        result = md.convert(filepath)
        os.remove(filepath)
        return result
    except Exception as e:
        st.error(f"MarkItDown failed: {e}")
        os.remove(filepath)
        return load_pdf_fallback(uploaded_file)

def load_pdf_fallback(uploaded_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.getbuffer())
        filepath = f.name
    loader = PyPDFLoader(filepath)
    pages = loader.load()
    os.remove(filepath)
    return "\n".join([p.page_content for p in pages])

def load_pdf_with_alternatives(uploaded_file) -> str:
    parsers = [
        ("PyMuPDF4LLM", load_pdf_with_pymupdf4llm),
        ("Unstructured", load_pdf_with_unstructured),
        ("MarkItDown", load_pdf_with_markitdown),
    ]
    for name, parser in parsers:
        try:
            result = parser(uploaded_file)
            st.info(f"Used {name} for parsing.")
            return result
        except: continue
    st.warning("All parsers failed. Using text.")
    return load_pdf_fallback(uploaded_file)

# =============================================
# 2. CHUNKING
# =============================================
def chunk_markdown_text(text: str, source_name: str) -> List[Document]:
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = markdown_splitter.split_text(text)
    final_chunks = []
    for i, chunk in enumerate(chunks):
        subchunks = recursive_splitter.split_text(chunk.page_content)
        for j, sc in enumerate(subchunks):
            metadata = {
                "source_document": source_name,
                "chunk_id": f"{source_name}_c{i}_s{j}",
                "header_1": chunk.metadata.get("Header 1"),
                "header_2": chunk.metadata.get("Header 2"),
                "chunk_type": "markdown",
            }
            final_chunks.append(Document(page_content=sc, metadata=metadata))
    return final_chunks

# =============================================
# 3. EMBEDDINGS
# =============================================
@st.cache_resource
def get_embeddings(use_scibert: bool = True):
    if use_scibert:
        return HuggingFaceEmbeddings(
            model_name=SCIBERT_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    return None

def get_safe_embeddings(use_scibert: bool, ollama_url: str):
    embeddings = get_embeddings(use_scibert)
    if embeddings is None:
        return OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_url)
    return embeddings

# =============================================
# 4. VECTOR STORE
# =============================================
@st.cache_resource
def create_vector_store(_chunks, _embeddings):
    try:
        return FAISS.from_documents(_chunks, _embeddings)
    except Exception as e:
        st.error(f"FAISS error: {e}")
        return None

# =============================================
# 5. RAG CHAIN (LangChain 0.2+ FINAL)
# =============================================
def create_enhanced_retriever(vectorstore, llm, k=6):
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
    return MultiQueryRetriever.from_llm(retriever=compression_retriever, llm=llm)

QUESTION_PROMPT = PromptTemplate.from_template(
    "Summarize this chunk in 2-3 sentences focusing on laser parameters, alloy composition, and microstructure.\n\n"
    "Context: {context}\nQuestion: {question}\nSummary:"
)

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

FUSION_COMBINE_PROMPT = PromptTemplate.from_template(FUSION_TEMPLATE)

def create_fusion_rag_chain(vectorstore, llm, k=6):
    retriever = create_enhanced_retriever(vectorstore, llm, k=k)
    
    from langchain.chains import StuffDocumentsChain, MapReduceDocumentsChain
    from langchain.chains.llm import LLMChain

    map_chain = LLMChain(llm=llm, prompt=QUESTION_PROMPT)
    reduce_chain = LLMChain(llm=llm, prompt=FUSION_COMBINE_PROMPT)
    reduce_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="context")

    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,  # ← FIXED
        document_variable_name="context",
        return_intermediate_steps=False,
    )

    return RetrievalQA(
        retriever=retriever,
        combine_documents_chain=map_reduce_chain,
        return_source_documents=True
    )

# =============================================
# 6. CLUSTERING
# =============================================
@st.cache_data
def compute_semantic_clusters(_chunks, _embeddings, n_clusters=5):
    if len(_chunks) < 2: return {}
    try:
        vectors = _embeddings.embed_documents([c.page_content for c in _chunks])
        k = min(n_clusters, len(vectors))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)
        clusters = {}
        for label in range(k):
            idxs = [i for i, l in enumerate(labels) if l == label]
            chunks_in_cluster = [_chunks[i] for i in idxs]
            sources = list(set([c.metadata["source_document"] for c in chunks_in_cluster]))
            sample = chunks_in_cluster[0].page_content[:200]
            clusters[f"Cluster {label}"] = {"size": len(idxs), "sources": sources, "sample": sample}
        return clusters
    except Exception as e:
        st.warning(f"Clustering failed: {e}")
        return {}

# =============================================
# 7. FORMATTING
# =============================================
def format_response_with_sources(response, question):
    answer = response.get("result", "No answer.")
    docs = response.get("source_documents", [])
    formatted = f"**Question:** {question}\n\n**Synthesized Answer:**\n{answer}\n\n"
    if docs:
        formatted += "**Sources (click to expand):**\n"
        for i, doc in enumerate(docs):
            src = doc.metadata.get("source_document", "Unknown")
            chunk_id = doc.metadata.get("chunk_id", "N/A")
            header = doc.metadata.get("header_1") or doc.metadata.get("header_2") or "No Header"
            with st.expander(f"Source {i+1}: {src} | {header} | {chunk_id}"):
                st.code(doc.page_content[:1500] + ("..." if len(doc.page_content) > 1500 else ""), language="markdown")
    return formatted

# =============================================
# 8. OLLAMA
# =============================================
def get_ollama_models():
    try:
        r = requests.get(f"{st.session_state.ollama_base_url}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except: pass
    return ["llama3.1:8b", "gemma3:latest", "mistral:7b"]

# =============================================
# 9. UI
# =============================================
def main():
    st.set_page_config(page_title="Laser-Microstructure RAG", layout="wide")
    st.title("Laser-Microstructure Interaction RAG")
    st.markdown("*Multi-document synthesis with equations, tables, and CPU-safe embeddings.*")

    with st.sidebar:
        st.header("Configuration")
        st.session_state.ollama_base_url = st.text_input("Ollama URL", OLLAMA_BASE_URL)
        models = get_ollama_models()
        if ChatOpenAI: models += ["Arena Model (Grok)"]
        st.session_state.llm_model = st.selectbox("LLM", models, index=0)
        st.session_state.use_scibert = st.checkbox("Use SciBERT (CPU)", True)
        st.session_state.retrieval_k = st.slider("Chunks", 3, 15, 6)
        if st.button("Clear Cache"): st.cache_resource.clear(); st.success("Cache cleared!")

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
                    with st.status(f"Parsing {f.name}..."):
                        md_text = load_pdf_with_alternatives(f)
                        chunks = chunk_markdown_text(md_text, f.name)
                        all_chunks.extend(chunks)
                        st.write(f"→ {len(chunks)} chunks")
                    st.session_state.processed_files.add(f.name)
                st.session_state.existing_chunks = all_chunks
                embeddings = get_safe_embeddings(st.session_state.use_scibert, st.session_state.ollama_base_url)
                st.session_state.vectorstore = create_vector_store(all_chunks, embeddings)
                st.success("Ingestion complete!")
                if len(all_chunks) > 5:
                    clusters = compute_semantic_clusters(all_chunks, embeddings)
                    with st.expander("Semantic Themes"): st.json(clusters, expanded=False)

    if st.session_state.get("vectorstore"):
        if "messages" not in st.session_state: st.session_state.messages = []
        if "query_log" not in st.session_state: st.session_state.query_log = []
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        llm = ChatOpenAI(model="grok-beta", base_url="https://api.x.ai/v1", api_key=os.getenv("XAI_API_KEY")) \
            if st.session_state.llm_model == "Arena Model (Grok)" and ChatOpenAI and os.getenv("XAI_API_KEY") \
            else Ollama(model=st.session_state.llm_model, base_url=st.session_state.ollama_base_url)

        if "qa_chain" not in st.session_state:
            st.session_state.qa_chain = create_fusion_rag_chain(st.session_state.vectorstore, llm, k=st.session_state.retrieval_k)

        if prompt := st.chat_input("Ask about laser parameters, microstructure, or cross-paper insights..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                placeholder = st.empty()
                with st.spinner("Synthesizing..."):
                    try:
                        response = st.session_state.qa_chain.invoke({"query": prompt})
                        formatted = format_response_with_sources(response, prompt)
                        placeholder.markdown(formatted)
                        st.session_state.query_log.append({
                            "time": datetime.now().isoformat(),
                            "q": prompt,
                            "a": response["result"],
                            "sources": [d.metadata["source_document"] for d in response.get("source_documents", [])]
                        })
                    except Exception as e:
                        st.error(f"Error: {e}")
            st.session_state.messages.append({"role": "assistant", "content": formatted})

        if st.session_state.query_log:
            st.download_button("Export Log", data=json.dumps(st.session_state.query_log, indent=2),
                               file_name=f"rag_log_{datetime.now().strftime('%Y%m%d_%H%M')}.json", mime="application/json")
    else:
        st.info("Upload PDFs to start.")

if __name__ == "__main__":
    main()
