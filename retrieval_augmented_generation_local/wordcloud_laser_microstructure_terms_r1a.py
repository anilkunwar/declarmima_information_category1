# =============================================
# LASER-MICROSTRUCTURE RAG + FUSION (FINAL v3.4 - ROBUST PDF INGESTION)
# Fixed: PDF ingestion with reliable PyPDFLoader + fallback mechanisms
# Updated: November 17, 2025 | PL | 5:00 AM CET
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
from collections import Counter, defaultdict
import math
import re
import pandas as pd
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

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
from langchain.chains import LLMChain

# --- Optional: Grok (xAI) ---
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

# --- PDF Parsers ---
try:
    import pymupdf4llm
    PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    PYMUPDF4LLM_AVAILABLE = False
try:
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False

# --- Modeling & Visualization ---
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models

# =============================================
# WORDCLOUD & MATPLOTLIB (GLOBAL IMPORT + SAFETY)
# =============================================
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except Exception as e:
    st.warning(f"WordCloud failed to import: {e}")
    WordCloud = None
    plt = None
    WORDCLOUD_AVAILABLE = False

# --- Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434"
SCIBERT_MODEL = "allenai/scibert_scivocab_uncased"
CHUNK_SIZE = 1000  # Increased for better content preservation
CHUNK_OVERLAP = 200
BATCH_SIZE = 32
MAX_WORKERS = min(6, os.cpu_count() or 4)
DEFAULT_LLM = "llama3.1:8b"

# =============================================
# DOMAIN-SPECIFIC CONFIGURATION
# =============================================

# Enhanced domain-specific keywords for focused extraction
DOMAIN_KEYWORDS = {
    # Laser parameters and processing
    'laser_processing': {
        'power', 'speed', 'scan', 'energy', 'density', 'watt', 'velocity', 'hatch', 'spot', 
        'beam', 'pulse', 'frequency', 'wavelength', 'processing', 'parameter', 'condition',
        'fluence', 'intensity', 'irradiation', 'melting', 'solidification', 'cooling rate',
        'thermal gradient', 'marangoni', 'recoil pressure', 'vaporization', 'keyhole'
    },
    
    # Additive manufacturing and 3D printing
    'additive_manufacturing': {
        '3d printing', 'additive manufacturing', 'selective laser melting', 'slm', 
        'laser powder bed', 'lpbf', 'direct energy deposition', 'ded', 'laser engineered net shaping', 'lens',
        'am', 'rapid prototyping', 'layer by layer', 'build platform', 'scan strategy',
        'support structure', 'post processing', 'heat treatment', 'annealing', 'hot isostatic pressing', 'hip'
    },
    
    # Multicomponent alloys and HEA/MPEA
    'multicomponent_alloys': {
        'high entropy alloy', 'hea', 'multi-principal element alloy', 'mpea', 'complex concentrated alloy', 'cca',
        'multicomponent', 'compositionally complex', 'cantor alloy', 'cocrfeni', 'cocrfenimn', 'cocrfeimn',
        'refractory hea', 'lightweight hea', 'transformation induced plasticity', 'trip', 'twinning induced plasticity', 'twip',
        'composition', 'alloy design', 'phase stability', 'configurational entropy', 'mixing entropy'
    },
    
    # Microstructure features
    'microstructure': {
        'grain', 'dendrite', 'precipitate', 'phase', 'texture', 'microstructure', 'columnar', 'equiaxed',
        'size', 'morphology', 'boundary', 'defect', 'crystal', 'structure', 'orientation', 'annealing',
        'heat treatment', 'segregation', 'porosity', 'crack', 'dislocation', 'twin', 'subgrain', 'cell structure',
        'eutectic', 'peritectic', 'monotectic', 'intermetallic', 'amorphous', 'nanocrystalline'
    },
    
    # Characterization methods
    'characterization': {
        'sem', 'tem', 'xrd', 'ebsd', 'oms', 'afm', 'atom probe', 'tomography', 'neutron diffraction',
        'synchrotron', 'x-ray', 'electron microscopy', 'optical microscopy', 'hardness', 'tensile',
        'compression', 'fatigue', 'creep', 'corrosion', 'wear', 'differential scanning calorimetry', 'dsc',
        'thermogravimetric analysis', 'tga'
    },
    
    # Computational and ML methods
    'computational_methods': {
        'phase field', 'finite element', 'fea', 'computational fluid dynamics', 'cfd', 'molecular dynamics', 'md',
        'monte carlo', 'calphad', 'dft', 'density functional theory', 'machine learning', 'ml', 'neural network',
        'deep learning', 'gaussian process', 'random forest', 'support vector machine', 'svm', 'regression',
        'classification', 'clustering', 'optimization', 'bayesian', 'uncertainty quantification', 'uq',
        'sensitivity analysis', 'surrogate model', 'digital twin', 'multiscale modeling'
    },
    
    # Uncertainty and reliability
    'uncertainty': {
        'uncertainty', 'variability', 'stochastic', 'probabilistic', 'reliability', 'robustness',
        'sensitivity', 'error', 'deviation', 'standard deviation', 'variance', 'confidence interval',
        'probability distribution', 'monte carlo simulation', 'random variable', 'aleatoric', 'epistemic'
    }
}

# =============================================
# ROBUST PDF INGESTION (FIXED)
# =============================================

def file_hash(file_obj) -> str:
    file_obj.seek(0)
    h = hashlib.md5(file_obj.read()).hexdigest()
    file_obj.seek(0)
    return h

def robust_load_and_chunk_pdf(uploaded_file) -> List[Document]:
    """Robust PDF loading with multiple fallback methods - based on working code"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file_path = tmp_file.name

    try:
        # Method 1: Direct PyPDFLoader (most reliable)
        try:
            loader = PyPDFLoader(tmp_file_path)
            pages = loader.load()
            
            # Enhanced metadata like working code
            for i, page in enumerate(pages):
                page.metadata["source_document"] = uploaded_file.name
                page.metadata["document_id"] = f"{uploaded_file.name}_{hash(uploaded_file.name)}"
                page.metadata["global_page_id"] = f"{uploaded_file.name}_page_{i+1}"
                page.metadata["header_1"] = None
                page.metadata["header_2"] = None

            # Simple text splitting (like working code)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len
            )
            chunks = text_splitter.split_documents(pages)
            
            if chunks and len(chunks) > 0:
                st.success(f"✓ PyPDFLoader: {uploaded_file.name} → {len(chunks)} chunks")
                return chunks
                
        except Exception as e:
            st.warning(f"PyPDFLoader failed for {uploaded_file.name}: {e}")

        # Method 2: Try advanced parsers as fallback
        advanced_content = ""
        
        # Try pymupdf4llm
        if PYMUPDF4LLM_AVAILABLE:
            try:
                md = pymupdf4llm.to_markdown(tmp_file_path)
                if md and len(md.strip()) > 100:  # Reasonable content length
                    advanced_content = md
                    st.info(f"✓ pymupdf4llm: {uploaded_file.name}")
            except Exception as e:
                st.warning(f"pymupdf4llm failed: {e}")

        # Try unstructured
        if not advanced_content and UNSTRUCTURED_AVAILABLE:
            try:
                elements = partition_pdf(tmp_file_path, strategy="hi_res")
                md = "\n\n".join([el.text for el in elements if hasattr(el, "text") and el.text])
                if md and len(md.strip()) > 100:
                    advanced_content = md
                    st.info(f"✓ unstructured: {uploaded_file.name}")
            except Exception as e:
                st.warning(f"unstructured failed: {e}")

        # Try markitdown
        if not advanced_content and MARKITDOWN_AVAILABLE:
            try:
                md = MarkItDown()
                result = md.convert(tmp_file_path)
                if result and len(str(result).strip()) > 100:
                    advanced_content = str(result)
                    st.info(f"✓ markitdown: {uploaded_file.name}")
            except Exception as e:
                st.warning(f"markitdown failed: {e}")

        # Process advanced content if available
        if advanced_content:
            doc = Document(
                page_content=advanced_content,
                metadata={
                    "source_document": uploaded_file.name,
                    "document_id": f"{uploaded_file.name}_{hash(uploaded_file.name)}",
                    "global_page_id": f"{uploaded_file.name}_full",
                    "header_1": None,
                    "header_2": None
                }
            )
            
            # Try markdown splitting first
            try:
                headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
                markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
                recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                
                md_chunks = markdown_splitter.split_text(advanced_content)
                final_chunks = []
                
                for i, chunk in enumerate(md_chunks):
                    subchunks = recursive_splitter.split_text(chunk.page_content)
                    base_meta = chunk.metadata.copy()
                    base_meta.update({
                        "source_document": uploaded_file.name,
                        "header_1": chunk.metadata.get("Header 1"),
                        "header_2": chunk.metadata.get("Header 2"),
                    })
                    
                    for j, sc in enumerate(subchunks):
                        meta = base_meta.copy()
                        meta["chunk_id"] = f"{uploaded_file.name}_md{i}_s{j}"
                        final_chunks.append(Document(page_content=sc, metadata=meta))
                
                if final_chunks:
                    st.success(f"✓ Markdown parsing: {uploaded_file.name} → {len(final_chunks)} chunks")
                    return final_chunks
                    
            except Exception as e:
                st.warning(f"Markdown splitting failed, using simple splitting: {e}")
                
            # Fallback: simple splitting for advanced content
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len
            )
            simple_chunks = text_splitter.split_documents([doc])
            if simple_chunks:
                st.success(f"✓ Simple splitting: {uploaded_file.name} → {len(simple_chunks)} chunks")
                return simple_chunks

        # Final fallback: empty document with warning
        st.error(f"❌ All PDF parsing methods failed for {uploaded_file.name}")
        return []
        
    finally:
        # Clean up temp file
        try:
            os.remove(tmp_file_path)
        except:
            pass

@st.cache_data(show_spinner=False)
def load_pdf_cached(_file_hash: str, file_bytes: bytes, source_name: str) -> List[Document]:
    """Cached version of robust PDF loading"""
    return robust_load_and_chunk_pdf_simple(file_bytes, source_name)

def robust_load_and_chunk_pdf_simple(file_bytes: bytes, source_name: str) -> List[Document]:
    """Simplified robust PDF loading for caching"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file_path = tmp_file.name

    try:
        # Primary method: Direct PyPDFLoader
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()
        
        # Enhanced metadata
        for i, page in enumerate(pages):
            page.metadata.update({
                "source_document": source_name,
                "document_id": f"{source_name}_{hash(source_name)}",
                "global_page_id": f"{source_name}_page_{i+1}",
                "header_1": None,
                "header_2": None
            })

        # Simple text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        chunks = text_splitter.split_documents(pages)
        
        return chunks if chunks else []
        
    except Exception as e:
        st.warning(f"PDF loading failed for {source_name}: {e}")
        return []
    finally:
        try:
            os.remove(tmp_file_path)
        except:
            pass

def parse_pdf_parallel(uploaded_file) -> List[Document]:
    """Parse PDF using robust method"""
    file_bytes = uploaded_file.getvalue()
    file_h = file_hash(uploaded_file)
    return load_pdf_cached(file_h, file_bytes, uploaded_file.name)

# =============================================
# 3. EMBEDDINGS (BATCHED)
# =============================================
@st.cache_resource
def get_embeddings(use_scibert: bool = True):
    if use_scibert:
        return HuggingFaceEmbeddings(
            model_name=SCIBERT_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': BATCH_SIZE}
        )
    return OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)

# =============================================
# 4. VECTOR STORE (INCREMENTAL)
# =============================================
@st.cache_resource
def create_vector_store_incremental(_chunks, _embeddings, _existing_store=None):
    if not _chunks:
        st.error("No chunks available to create vector store")
        return None
        
    texts = [c.page_content for c in _chunks]
    metadatas = [c.metadata for c in _chunks]
    
    if _existing_store is not None:
        _existing_store.add_texts(texts, metadatas=metadatas)
        return _existing_store
    return FAISS.from_texts(texts, _embeddings, metadatas=metadatas)

# =============================================
# 5. RAG CHAIN (DOMAIN-ENHANCED)
# =============================================
def create_fusion_rag_chain(vectorstore, llm, k=6):
    if not vectorstore:
        st.error("No vector store available")
        return None
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    multi_retriever = MultiQueryRetriever.from_llm(retriever=compression_retriever, llm=llm)

    QUESTION_PROMPT = PromptTemplate.from_template(
        "Summarize in 2-3 sentences focusing on: laser parameters, alloy systems, microstructure evolution.\nContext: {context}\nQuestion: {question}\nSummary:"
    )
    FUSION_COMBINE_PROMPT = PromptTemplate.from_template(DOMAIN_FUSION_TEMPLATE)

    from langchain.chains import StuffDocumentsChain, MapReduceDocumentsChain
    from langchain.chains.llm import LLMChain
    map_chain = LLMChain(llm=llm, prompt=QUESTION_PROMPT)
    reduce_chain = LLMChain(llm=llm, prompt=FUSION_COMBINE_PROMPT)
    reduce_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="context")
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="context",
        return_intermediate_steps=False,
    )
    return RetrievalQA(
        retriever=multi_retriever,
        combine_documents_chain=map_reduce_chain,
        return_source_documents=True
    )

DOMAIN_FUSION_TEMPLATE = """
You are a materials scientist specializing in laser-microstructure interactions in multicomponent alloys and additive manufacturing.

CONTEXT (from multiple papers):
{context}

QUESTION: {question}

DOMAIN FOCUS AREAS:
- Laser processing parameters (power, speed, spot size, energy density, scanning strategy)
- Multicomponent alloy systems (HEA, MPEA, compositionally complex alloys)
- Microstructure evolution (grain structure, phase formation, defects, texture)
- Additive manufacturing (3D printing, LPBF/SLM processes)
- Uncertainty quantification and variability analysis
- Experimental and computational methods (phase field, CALPHAD, machine learning)

INSTRUCTIONS:
1. Extract quantitative laser parameters and process conditions
2. Identify specific alloy systems and compositions
3. Analyze microstructure features and evolution mechanisms
4. Compare process-structure-property relationships across papers
5. Highlight uncertainty quantification methods and reliability analysis
6. Note computational/experimental methods for interface studies
7. Identify gaps in laser-microstructure understanding

OUTPUT FORMAT:
### Laser-Alloy-Microstructure Synthesis
- **Process Parameters**: [laser power, speed, energy density, etc.]
- **Alloy Systems**: [HEA/MPEA types, compositions]
- **Microstructure Features**: [grain size, phases, defects, texture]
- **AM Process Details**: [3D printing method, parameters]

### Cross-Study Analysis
- **Methodologies**: [experimental, computational, ML approaches]
- **Uncertainty Analysis**: [variability, reliability methods]
- **Key Relationships**: [process → structure → property links]

### Research Insights
- [Synthesized findings and open questions]

CITE SOURCES: [Source: paper_name.pdf, Chunk ID]

SYNTHESIZED ANSWER:
"""

# =============================================
# 6. CLUSTERING (MiniBatchKMeans)
# =============================================
@st.cache_data
def compute_semantic_clusters_fast(_chunks, _embeddings, n_clusters=5):
    if len(_chunks) < 2: return {}
    texts = [c.page_content for c in _chunks]
    vectors = np.array(_embeddings.embed_documents(texts))
    k = min(n_clusters, len(vectors))
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256, n_init=3)
    labels = kmeans.fit_predict(vectors)
    clusters = {}
    for label in range(k):
        idxs = np.where(labels == label)[0].tolist()
        sources = list(set(_chunks[i].metadata["source_document"] for i in idxs))
        sample = _chunks[idxs[0]].page_content[:200]
        clusters[f"Cluster {label}"] = {"size": len(idxs), "sources": sources, "sample": sample}
    return clusters

# =============================================
# 7. FORMATTING
# =============================================
def format_response_with_sources(response, question):
    if not response:
        return "No response generated."
        
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
# 9. ENHANCED UI WITH ROBUST INGESTION
# =============================================
def main():
    st.set_page_config(page_title="Laser-Microstructure RAG", layout="wide")
    st.title("🧪 Laser-Microstructure Interaction RAG (v3.4)")
    st.markdown("*Robust PDF Ingestion • Domain-Focused • Multi-Document Fusion*")

    with st.sidebar:
        st.header("⚙️ Configuration")
        st.session_state.ollama_base_url = st.text_input("Ollama URL", OLLAMA_BASE_URL)
        models = get_ollama_models()
        if ChatOpenAI: models += ["Grok"]
        st.session_state.llm_model = st.selectbox("LLM Model", models, index=0)
        st.session_state.use_scibert = st.checkbox("Use SciBERT Embeddings", True)
        st.session_state.retrieval_k = st.slider("Retrieval Chunks", 3, 15, 6)
        
        # Enhanced WordCloud Configuration
        st.markdown("---")
        st.header("🎯 Domain Focus")
        
        domain_options = {
            "All Domains": "all",
            "Laser & Additive Manufacturing": "laser_am", 
            "Materials (HEA/MPEA)": "materials",
            "Microstructure Analysis": "microstructure",
            "Methods & Characterization": "methods"
        }
        
        st.session_state.domain_focus = st.selectbox(
            "Research Domain Focus",
            list(domain_options.keys()),
            index=0,
            help="Focus wordcloud on specific research domains"
        )
        
        st.markdown("---")
        st.header("🚀 Processing Strategy")
        processing_strategies = [
            "Fast & Reliable (PyPDFLoader)",
            "Advanced Parsing (Multiple Methods)",
            "High Quality (Markdown Preservation)"
        ]
        st.session_state.processing_strategy = st.selectbox(
            "PDF Processing",
            processing_strategies,
            index=0,
            help="Choose PDF processing strategy"
        )
            
        if st.button("🗑️ Clear Cache", type="secondary"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success("Cache cleared!")

    # File upload and processing - ENHANCED
    uploaded_files = st.file_uploader("📁 Upload PDF Research Papers", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        if "processed_files" not in st.session_state:
            st.session_state.processed_files = set()
        if "existing_chunks" not in st.session_state:
            st.session_state.existing_chunks = []

        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]

        if new_files:
            with st.spinner(f"🔬 Processing {len(new_files)} new PDF(s)..."):
                all_new_chunks = []
                successful_files = 0
                
                for uploaded_file in new_files:
                    try:
                        # Use robust PDF loading
                        chunks = robust_load_and_chunk_pdf(uploaded_file)
                        
                        if chunks and len(chunks) > 0:
                            all_new_chunks.extend(chunks)
                            st.session_state.processed_files.add(uploaded_file.name)
                            successful_files += 1
                            st.success(f"✅ {uploaded_file.name}: {len(chunks)} chunks")
                        else:
                            st.error(f"❌ {uploaded_file.name}: Failed to extract content")
                            
                    except Exception as e:
                        st.error(f"❌ {uploaded_file.name}: Error - {e}")

                # Update existing chunks
                if all_new_chunks:
                    st.session_state.existing_chunks.extend(all_new_chunks)
                    
                    # Create or update vector store
                    embeddings = get_embeddings(st.session_state.use_scibert)
                    existing_store = st.session_state.get("vectorstore")
                    st.session_state.vectorstore = create_vector_store_incremental(
                        all_new_chunks, embeddings, existing_store
                    )
                    
                    st.success(f"🎉 Successfully processed {successful_files}/{len(new_files)} files. Total chunks: {len(st.session_state.existing_chunks)}")
                    
                    # Show document analytics
                    with st.expander("📊 Document Analytics"):
                        doc_stats = {}
                        for chunk in st.session_state.existing_chunks:
                            source = chunk.metadata.get("source_document", "unknown")
                            if source not in doc_stats:
                                doc_stats[source] = 0
                            doc_stats[source] += 1
                        
                        st.write("**Chunks per Document:**")
                        for doc, count in doc_stats.items():
                            st.write(f"- {doc}: {count} chunks")
                            
                        if len(st.session_state.existing_chunks) > 5:
                            clusters = compute_semantic_clusters_fast(st.session_state.existing_chunks, embeddings)
                            st.write("**Semantic Clusters:**")
                            st.json(clusters)
                else:
                    st.error("❌ No content could be extracted from the uploaded files.")

        else:
            st.info("📚 All uploaded files have been processed. You can ask questions that span across all documents.")

    # Main interaction section
    if st.session_state.get("vectorstore") and st.session_state.existing_chunks:
        llm = Ollama(model=st.session_state.llm_model, base_url=st.session_state.ollama_base_url) if st.session_state.llm_model != "Grok" else ChatOpenAI(model="grok-beta", base_url="https://api.x.ai/v1", api_key=os.getenv("XAI_API_KEY"))
        st.session_state.llm = llm
        
        if "qa_chain" not in st.session_state:
            st.session_state.qa_chain = create_fusion_rag_chain(
                st.session_state.vectorstore, 
                llm, 
                k=st.session_state.retrieval_k
            )

        # Enhanced chat interface
        st.markdown("---")
        st.header("💬 Research Assistant")
        
        # Suggested questions
        suggested_questions = [
            "How do laser parameters affect microstructure in high entropy alloys?",
            "What are the key microstructure features in additively manufactured HEAs?",
            "How is uncertainty quantified in laser-microstructure relationships?",
            "Compare computational methods for predicting microstructure evolution",
            "What experimental techniques are used for HEA microstructure characterization?"
        ]
        
        st.markdown("**💡 Suggested Questions:**")
        cols = st.columns(2)
        for i, question in enumerate(suggested_questions):
            with cols[i % 2]:
                if st.button(question, key=f"q_{i}", use_container_width=True):
                    if "messages" not in st.session_state:
                        st.session_state.messages = []
                    st.session_state.messages.append({"role": "user", "content": question})
                    with st.chat_message("user"): st.markdown(question)
                    with st.chat_message("assistant"):
                        with st.spinner("🔍 Synthesizing research insights..."):
                            response = st.session_state.qa_chain.invoke({"query": question})
                            formatted = format_response_with_sources(response, question)
                            st.markdown(formatted)
                    st.session_state.messages.append({"role": "assistant", "content": formatted})

        # Chat history
        if "messages" in st.session_state:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]): 
                    st.markdown(msg["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about laser-microstructure interactions..."):
            if "messages" not in st.session_state:
                st.session_state.messages = []
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("🔍 Synthesizing cross-document insights..."):
                    response = st.session_state.qa_chain.invoke({"query": prompt})
                    formatted = format_response_with_sources(response, prompt)
                    st.markdown(formatted)
            st.session_state.messages.append({"role": "assistant", "content": formatted})

        # Rest of the UI components (WordCloud, Topic Modeling, etc.)
        # ... [Keep the existing WordCloud and Topic Modeling sections from previous version] ...

    else:
        st.info("""
        ## 🚀 Getting Started
        
        **1.** Upload PDF research papers on:
        - Laser materials processing
        - High entropy alloys (HEA) and multicomponent systems  
        - Additive manufacturing and 3D printing
        - Microstructure characterization
        - Computational modeling and uncertainty quantification
        
        **2.** Configure your analysis preferences in the sidebar
        
        **3.** Ask research questions or generate domain-focused visualizations
        
        **📝 Note:** Using PyPDFLoader as primary method for reliable content extraction.
        """)

if __name__ == "__main__":
    main()
