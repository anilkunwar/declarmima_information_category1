# =============================================
# LASER-MICROSTRUCTURE RAG + FUSION (FINAL v3.5 – FOCUSED WORDCLOUD + POSTPROCESSING)
# LangChain 0.2+ | CPU-SAFE | 5–10× FASTER WORDCLOUD | MULTI-STRATEGY + FILTERING
# Updated: November 17, 2025 | PL | 02:19 AM CET
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
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import random

# --- LangChain & RAG ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

# --- Modeling & Visualization ---
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

# --- WordCloud ---
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception as e:
    st.warning(f"WordCloud import failed: {e}")
    WordCloud = None
    WORDCLOUD_AVAILABLE = False

# =============================================
# CONFIGURATION
# =============================================
OLLAMA_BASE_URL = "http://localhost:11434"
SCIBERT_MODEL = "allenai/scibert_scivocab_uncased"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
BATCH_SIZE = 32
MAX_WORKERS = min(6, os.cpu_count() or 4)
DEFAULT_LLM = "llama3.1:8b"

# =============================================
# 1. PDF → CHUNKS (PROVEN SINGLE-STEP INGESTION)
# =============================================
def load_and_chunk_pdf(uploaded_file) -> List[Document]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load_and_split()

        for i, page in enumerate(pages):
            page.metadata.update(
                source_document=uploaded_file.name,
                document_id=f"{uploaded_file.name}_{hash(uploaded_file.name)}",
                global_page_id=f"{uploaded_file.name}_page_{i+1}",
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = splitter.split_documents(pages)

        final_chunks: List[Document] = []
        for idx, chunk in enumerate(chunks):
            final_chunks.append(
                Document(
                    page_content=chunk.page_content,
                    metadata={
                        "source_document": uploaded_file.name,
                        "global_page_id": chunk.metadata.get("global_page_id"),
                        "chunk_id": f"{uploaded_file.name}_c{idx}",
                        "header_1": chunk.metadata.get("Header 1"),
                        "header_2": chunk.metadata.get("Header 2"),
                        "chunk_type": "pdf_page",
                    },
                )
            )
        return final_chunks
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# =============================================
# 2. EMBEDDINGS
# =============================================
@st.cache_resource
def get_embeddings(use_scibert: bool = True):
    if use_scibert:
        return HuggingFaceEmbeddings(
            model_name=SCIBERT_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": BATCH_SIZE},
        )
    return OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)


# =============================================
# 3. VECTOR STORE (INCREMENTAL)
# =============================================
@st.cache_resource
def create_vector_store_incremental(_chunks, _embeddings, _existing_store=None):
    texts = [c.page_content for c in _chunks]
    metadatas = [c.metadata for c in _chunks]
    if _existing_store is not None:
        _existing_store.add_texts(texts, metadatas=metadatas)
        return _existing_store
    return FAISS.from_texts(texts, _embeddings, metadatas=metadatas)


# =============================================
# 4. RAG CHAIN (FUSION-AWARE)
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

def create_fusion_rag_chain(vectorstore, llm, k=6):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=compression_retriever, llm=llm
    )
    QUESTION_PROMPT = PromptTemplate.from_template(
        "Summarize in 2-3 sentences: laser, alloy, microstructure.\nContext: {context}\nQuestion: {question}\nSummary:"
    )
    FUSION_COMBINE_PROMPT = PromptTemplate.from_template(FUSION_TEMPLATE)

    from langchain.chains import StuffDocumentsChain, MapReduceDocumentsChain
    from langchain.chains.llm import LLMChain

    map_chain = LLMChain(llm=llm, prompt=QUESTION_PROMPT)
    reduce_chain = LLMChain(llm=llm, prompt=FUSION_COMBINE_PROMPT)
    reduce_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="context"
    )
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="context",
        return_intermediate_steps=False,
    )
    return RetrievalQA(
        retriever=multi_retriever,
        combine_documents_chain=map_reduce_chain,
        return_source_documents=True,
    )


# =============================================
# 5. CLUSTERING
# =============================================
@st.cache_data
def compute_semantic_clusters_fast(_chunks, _embeddings, n_clusters=5):
    if len(_chunks) < 2:
        return {}
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
# 6. RESPONSE FORMATTING
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
                st.code(
                    doc.page_content[:1500] + ("..." if len(doc.page_content) > 1500 else ""),
                    language="markdown",
                )
    return formatted


# =============================================
# 7. OLLAMA MODEL LIST
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
# 8. WORDCLOUD STRATEGIES (FOCUSED + POSTPROCESSING + COLOR GRADIENT)
# =============================================
def get_relevant_chunks_for_wordcloud(chunks, sample_ratio=0.3, min_chunks=10, max_chunks=50):
    if len(chunks) == 0:
        st.error("No chunks available for wordcloud.")
        return []
    if len(chunks) == 1:
        return chunks
    if len(chunks) <= max_chunks:
        return chunks

    try:
        embeddings = get_embeddings(st.session_state.use_scibert)
        vectors = np.array(embeddings.embed_documents([c.page_content for c in chunks]))
        n_samples = min(max_chunks, max(min_chunks, int(len(chunks) * sample_ratio)))
        n_clusters = min(10, len(chunks) // 3)
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256)
        labels = kmeans.fit_predict(vectors)

        selected_chunks = []
        for cluster_id in np.unique(labels):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_chunks = [chunks[i] for i in cluster_indices]
            n_from_cluster = max(1, min(2, len(cluster_chunks) // 5))
            selected_indices = np.random.choice(len(cluster_chunks), n_from_cluster, replace=False)
            selected_chunks.extend([cluster_chunks[i] for i in selected_indices])
        return selected_chunks[:max_chunks]
    except Exception as e:
        st.warning(f"Clustering failed, using random sampling: {e}")
        return chunks[:max_chunks]


def extract_phrases_batch_optimized(llm, chunks, batch_size=8):
    extract_prompt = PromptTemplate.from_template(
        "Extract MAX 5 key phrases (1-3 words) for:\n"
        "- Laser parameters: power, scan speed, energy density, 3D printing, additive manufacturing\n"
        "- Microstructure: high entropy alloys, HEA, multicomponent alloy, multiprincipal element alloy, MEA, grain size, dendrites, precipitates, texture, interface, dynamics\n"
        "- Methods: experimental, computational, machine learning, AI, uncertainty quantification, UQ, simulation\n"
        "Output ONLY JSON:\n```json\n{{'laser': [...], 'microstructure': [...], 'methods': [...]}}\n```\nText: {text}"
    )
    extract_chain = LLMChain(llm=llm, prompt=extract_prompt)

    def process_batch(batch_chunks):
        combined_text = "\n---\n".join([chunk.page_content[:500] for chunk in batch_chunks])
        try:
            resp = extract_chain.invoke({"text": f"Extract from these text segments:\n{combined_text}"})
            raw = resp.get("text", "") or resp.get("result", "")
            cleaned = re.sub(r"^```[a-z]*\n|```$", "", raw.strip(), flags=re.MULTILINE)
            data = json.loads(cleaned) if cleaned else {}
            laser_phrases = data.get("laser", [])[:3]
            micro_phrases = data.get("microstructure", [])[:3]
            methods_phrases = data.get("methods", [])[:3]
            results = []
            for chunk in batch_chunks:
                results.append({
                    "source": chunk.metadata["source_document"],
                    "laser": laser_phrases,
                    "micro": micro_phrases,
                    "methods": methods_phrases
                })
            return results
        except Exception as e:
            return [{"source": chunk.metadata["source_document"], "laser": [], "micro": [], "methods": []}
                    for chunk in batch_chunks]

    all_results = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_results = process_batch(batch)
        all_results.extend(batch_results)
    return all_results


def hybrid_phrase_extraction(chunks, use_llm_fallback=True):
    laser_keywords = {
        'power', 'speed', 'scan', 'energy', 'density', 'watt', 'velocity',
        'hatch', 'spot', 'beam', 'pulse', 'frequency', 'wavelength', 'laser',
        'processing', 'parameter', 'condition', '3d printing', 'additive manufacturing',
        'selective laser melting', 'slm', 'direct energy deposition'
    }
    micro_keywords = {
        'grain', 'dendrite', 'precipitate', 'phase', 'texture', 'microstructure',
        'columnar', 'equiaxed', 'size', 'morphology', 'boundary', 'defect',
        'crystal', 'structure', 'orientation', 'high entropy alloy', 'hea',
        'multicomponent alloy', 'multiprincipal element alloy', 'mea',
        'interface', 'dynamics', 'solidification', 'melt pool'
    }
    methods_keywords = {
        'experimental', 'computational', 'machine learning', 'ml', 'ai',
        'artificial intelligence', 'uncertainty quantification', 'uq',
        'simulation', 'modeling', 'finite element', 'molecular dynamics',
        'phase field', 'monte carlo', 'deep learning', 'neural network'
    }
    results = []
    for chunk in chunks:
        text_lower = chunk.page_content.lower()
        words = re.findall(r'\b[\w-]+\b', text_lower)
        laser_phrases = []
        micro_phrases = []
        methods_phrases = []
        for i, word in enumerate(words):
            if word in laser_keywords:
                start = max(0, i-1)
                end = min(len(words), i+2)
                phrase = ' '.join(words[start:end])
                if len(phrase) > 3:
                    laser_phrases.append(phrase)
            if word in micro_keywords:
                start = max(0, i-1)
                end = min(len(words), i+2)
                phrase = ' '.join(words[start:end])
                if len(phrase) > 3:
                    micro_phrases.append(phrase)
            if word in methods_keywords:
                start = max(0, i-1)
                end = min(len(words), i+2)
                phrase = ' '.join(words[start:end])
                if len(phrase) > 3:
                    methods_phrases.append(phrase)

        if use_llm_fallback and (len(laser_phrases) + len(micro_phrases) + len(methods_phrases) < 3):
            try:
                extract_prompt = PromptTemplate.from_template(
                    "Extract key phrases (1-3 words) for laser parameters, microstructure (HEA, MEA, grain, dendrites, etc.), methods (ML, AI, UQ, experimental, computational). Output ONLY JSON:\n"
                    "```json\n{{'laser': [...], 'microstructure': [...], 'methods': [...]}}\n```\nText: {text}"
                )
                extract_chain = LLMChain(llm=st.session_state.llm, prompt=extract_prompt)
                resp = extract_chain.invoke({"text": chunk.page_content})
                raw = resp.get("text", "") or resp.get("result", "")
                cleaned = re.sub(r"^```[a-z]*\n|```$", "", raw.strip(), flags=re.MULTILINE)
                data = json.loads(cleaned) if cleaned else {}
                laser_phrases.extend(data.get("laser", [])[:3])
                micro_phrases.extend(data.get("microstructure", [])[:3])
                methods_phrases.extend(data.get("methods", [])[:3])
            except:
                pass

        laser_phrases = list(set(laser_phrases))[:5]
        micro_phrases = list(set(micro_phrases))[:5]
        methods_phrases = list(set(methods_phrases))[:5]
        results.append({
            "source": chunk.metadata["source_document"],
            "laser": laser_phrases,
            "micro": micro_phrases,
            "methods": methods_phrases
        })
    return results


def extract_from_chunk_original(llm, chunk):
    extract_prompt = PromptTemplate.from_template(
        "Extract key phrases (1-3 words) for laser parameters, microstructure (HEA, MEA, grain, dendrites, etc.), methods (ML, AI, UQ, experimental, computational). Output ONLY JSON:\n"
        "```json\n{{'laser': [...], 'microstructure': [...], 'methods': [...]}}\n```\nText: {text}"
    )
    extract_chain = LLMChain(llm=llm, prompt=extract_prompt)
    try:
        resp = extract_chain.invoke({"text": chunk.page_content})
        raw = resp.get("text", "") or resp.get("result", "")
        cleaned = re.sub(r"^```[a-z]*\n|```$", "", raw.strip(), flags=re.MULTILINE)
        data = json.loads(cleaned) if cleaned else {}
        return {
            "source": chunk.metadata["source_document"],
            "laser": data.get("laser", []),
            "micro": data.get("microstructure", []),
            "methods": data.get("methods", []),
        }
    except:
        return {"source": chunk.metadata["source_document"], "laser": [], "micro": [], "methods": []}


@st.cache_data(show_spinner=False, ttl=3600)
def generate_wordcloud_cached(_strategy, _llm_model, _chunks_hash, use_scibert):
    return "cached_data"


def generate_wordcloud_with_strategy(strategy, llm, chunks):
    if not WORDCLOUD_AVAILABLE:
        st.error("WordCloud package not available. Install with: `pip install wordcloud`")
        return

    # === Postprocessing Options from Sidebar ===
    max_words = st.session_state.get('wordcloud_max_words', 150)
    exclude_terms = [t.strip().lower() for t in st.session_state.get('wordcloud_exclude_terms', '').split(',') if t.strip()]
    font_path = st.session_state.get('wordcloud_font', None)

    st.info(f"Using strategy: **{strategy}**")
    if strategy == "Default (Caching)":
        return generate_wordcloud_cached("default", llm.model if hasattr(llm, 'model') else "grok",
                                       hashlib.md5(str(chunks).encode()).hexdigest(),
                                       st.session_state.use_scibert)

    start_time = datetime.now()

    # === Strategy Execution ===
    if strategy == "Chunk Sampling":
        sampled_chunks = get_relevant_chunks_for_wordcloud(chunks)
        st.write(f"Using {len(sampled_chunks)} representative chunks (from {len(chunks)} total)")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(lambda c: extract_from_chunk_original(llm, c), sampled_chunks))

    elif strategy == "Batch Processing":
        results = extract_phrases_batch_optimized(llm, chunks, batch_size=8)
        st.write(f"Processed {len(chunks)} chunks in batches")

    elif strategy == "Hybrid Extraction":
        results = hybrid_phrase_extraction(chunks, use_llm_fallback=True)
        st.write(f"Hybrid extraction on {len(chunks)} chunks")

    elif strategy == "Combined (All Optimizations)":
        sampled_chunks = get_relevant_chunks_for_wordcloud(chunks)
        st.write(f"Combined: {len(sampled_chunks)} sampled chunks")
        results = hybrid_phrase_extraction(sampled_chunks, use_llm_fallback=True)

    else:  # Original
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(lambda c: extract_from_chunk_original(llm, c), chunks))

    # === Build term lists ===
    laser_phrases = [p for r in results for p in r["laser"] if len(p) > 2]
    micro_phrases = [p for r in results for p in r["micro"] if len(p) > 2]
    methods_phrases = [p for r in results for p in r["methods"] if len(p) > 2]

    if not laser_phrases and not micro_phrases and not methods_phrases:
        st.warning("No meaningful phrases extracted. Try a different strategy.")
        return

    # === Source tracking ===
    laser_sources = defaultdict(set)
    micro_sources = defaultdict(set)
    methods_sources = defaultdict(set)
    for r in results:
        for p in r["laser"]: laser_sources[p].add(r["source"])
        for p in r["micro"]: micro_sources[p].add(r["source"])
        for p in r["methods"]: methods_sources[p].add(r["source"])

    # === TF-IDF or Frequency Scoring ===
    source_docs = defaultdict(list)
    for r in results:
        doc_text = " ".join(r["laser"] + r["micro"] + r["methods"])
        if doc_text.strip():
            source_docs[r["source"]].append(doc_text)

    corpus = [" ".join(source_docs[src]) for src in source_docs if source_docs[src]]
    n_sources = len(corpus)
    all_terms = set(laser_phrases + micro_phrases + methods_phrases)

    if n_sources >= 2:
        try:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                token_pattern=r'(?u)\b[\w\-]+\b',
                max_features=1000
            )
            tfidf_matrix = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = dict(zip(feature_names, np.ravel(tfidf_matrix.sum(axis=0))))
            st.info(f"TF-IDF applied across {n_sources} source documents.")
        except Exception as e:
            st.warning(f"TF-IDF failed ({e}), using frequency.")
            tfidf_scores = {}
    else:
        st.info("Single source → using frequency scoring.")
        tfidf_scores = {}

    final_scores = {}
    categories = {}
    for term in all_terms:
        if len(term) < 3:
            continue
        freq = laser_phrases.count(term) + micro_phrases.count(term) + methods_phrases.count(term)
        num_sources_term = len(laser_sources.get(term, set()) | micro_sources.get(term, set()) | methods_sources.get(term, set()))
        score = freq * (1 + math.log(num_sources_term + 1))
        if term.lower() in tfidf_scores:
            score *= tfidf_scores[term.lower()]
        final_scores[term] = max(score, 0.01)
        has_l = term in laser_phrases
        has_m = term in micro_phrases
        has_meth = term in methods_phrases
        if has_l and not has_m and not has_meth:
            categories[term] = "laser"
        elif has_m and not has_l and not has_meth:
            categories[term] = "microstructure"
        elif has_meth and not has_l and not has_m:
            categories[term] = "methods"
        else:
            categories[term] = "mixed"  # gradient/mixed

    # === Postprocess: exclude and top N ===
    for ex in exclude_terms:
        final_scores.pop(ex, None)
    top_terms = dict(sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:max_words])

    if not top_terms:
        st.warning("No high-scoring terms for wordcloud after filtering.")
        return

    # === Color Function with Gradient ===
    def color_func(word, **kwargs):
        cat = categories.get(word, "")
        if cat == "laser":
            return "#d62728"  # Red
        elif cat == "microstructure":
            return "#1f77b4"  # Blue
        elif cat == "methods":
            return "#2ca02c"  # Green
        elif cat == "mixed":
            # Gradient: interpolate between colors
            r1, g1, b1 = 214, 39, 40   # red
            r2, g2, b2 = 31, 119, 180  # blue
            r3, g3, b3 = 44, 160, 44   # green
            # Randomly pick two base colors for mixed
            base1 = random.choice([(r1,g1,b1), (r2,g2,b2), (r3,g3,b3)])
            base2 = random.choice([(r1,g1,b1), (r2,g2,b2), (r3,g3,b3)])
            while base2 == base1:
                base2 = random.choice([(r1,g1,b1), (r2,g2,b2), (r3,g3,b3)])
            factor = 0.5
            r = int(base1[0] * (1 - factor) + base2[0] * factor)
            g = int(base1[1] * (1 - factor) + base2[1] * factor)
            b = int(base1[2] * (1 - factor) + base2[2] * factor)
            return f"#{r:02x}{g:02x}{b:02x}"
        return "#999999"

    # === Generate WordCloud ===
    wc = WordCloud(
        width=1000, height=550,
        background_color="white",
        color_func=color_func,
        max_words=max_words,
        collocations=False,
        prefer_horizontal=0.8,
        font_path=font_path
    )
    wc.generate_from_frequencies(top_terms)
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    duration = (datetime.now() - start_time).total_seconds()
    st.success(f"WordCloud generated in {duration:.1f}s using **{strategy}**")
    st.markdown(f"""
    ### WordCloud: {strategy}
    - **Red**: Laser parameters (heat source, 3D printing)
    - **Blue**: Microstructure (HEA, MEA, grain, dendrites, interface)
    - **Green**: Methods (ML, AI, UQ, experimental, computational)
    - **Gradient (Mixed)**: Overlapping terms (e.g., "laser solidification", "ML microstructure")
    - **Sources**: {n_sources} document(s)
    - **Chunks analyzed**: {len(results)}
    - **Time**: {duration:.1f}s
    """)


# =============================================
# 9. LDA / NMF + COHERENCE
# =============================================
def plot_topic_wordclouds(topics, n_topics, model_type="LDA"):
    if not WORDCLOUD_AVAILABLE:
        st.warning("WordCloud not available - skipping visualization")
        return
    cols = st.columns(min(n_topics, 3))
    for i, topic in enumerate(topics):
        with cols[i % 3]:
            word_freq = {term: float(w) for term, w in zip(topic["terms"], topic["weights"])}
            wc = WordCloud(width=300, height=200, background_color="white", colormap="viridis")
            wc.generate_from_frequencies(word_freq)
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            ax.set_title(f"{model_type} Topic {topic['topic_id']}", fontsize=10)
            st.pyplot(fig)


@st.cache_data(show_spinner=False)
def compute_lda_fast(_chunks, n_topics=5):
    texts = [c.page_content.lower() for c in _chunks]
    sources = [c.metadata["source_document"] for c in _chunks]
    vectorizer = TfidfVectorizer(max_df=0.7, min_df=2, stop_words='english', token_pattern=r'(?u)\b\w\w+\b', ngram_range=(1,2))
    X = vectorizer.fit_transform(texts)
    docs = [text.split() for text in texts]
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=2, no_above=0.7)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics, random_state=42, passes=5, alpha='auto')
    coherence = CoherenceModel(model=lda, texts=docs, dictionary=dictionary, coherence='c_v').get_coherence()
    topics = [{"topic_id": i, "terms": [w for w,p in lda.show_topic(i, 10)], "weights": [f"{p:.3f}" for w,p in lda.show_topic(i, 10)]} for i in range(n_topics)]
    doc_dist = []
    for i, bow in enumerate(corpus):
        dist = lda.get_document_topics(bow)
        probs = [0.0] * n_topics
        for t,p in dist: probs[t] = p
        total = sum(probs)
        if total > 0: probs = [p/total for p in probs]
        doc_dist.append({"document": sources[i], "dominant_topic": probs.index(max(probs)), "contributions": [f"{p:.3f}" for p in probs]})
    return topics, doc_dist, coherence


@st.cache_data(show_spinner=False)
def compute_nmf_fast(_chunks, n_topics=5, vectorizer=None):
    texts = [c.page_content.lower() for c in _chunks]
    sources = [c.metadata["source_document"] for c in _chunks]
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_df=0.7, min_df=2, stop_words='english', token_pattern=r'(?u)\b\w\w+\b', ngram_range=(1,2))
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=200)
    W = nmf.fit_transform(X)
    H = nmf.components_
    topics = []
    for i, topic in enumerate(H):
        top_idx = topic.argsort()[-10:][::-1]
        topics.append({"topic_id": i, "terms": [feature_names[j] for j in top_idx], "weights": [f"{topic[j]:.3f}" for j in top_idx]})
    doc_dist = [{"document": sources[i], "dominant_topic": np.argmax(W[i]), "contributions": [f"{p:.3f}" for p in (W[i]/W[i].sum())]} for i in range(len(sources))]
    docs = [text.split() for text in texts]
    dictionary = Dictionary(docs)
    coherence = CoherenceModel(topics=[[t for t in topic["terms"]] for topic in topics], texts=docs, dictionary=dictionary, coherence='c_v').get_coherence()
    return topics, doc_dist, coherence


@st.cache_data
def evaluate_coherence_sweep(_chunks, model_type="LDA", max_topics=10):
    texts = [c.page_content.lower() for c in _chunks]
    vectorizer = TfidfVectorizer(max_df=0.7, min_df=2, stop_words='english', token_pattern=r'(?u)\b\w\w+\b', ngram_range=(1,2))
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    docs = [text.split() for text in texts]
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=2, no_above=0.7)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    coherence_values = []
    ks = list(range(2, max_topics + 1))
    for k in ks:
        if model_type == "LDA":
            model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, random_state=42, passes=5, alpha='auto')
            cm = CoherenceModel(model=model, texts=docs, dictionary=dictionary, coherence='c_v')
        elif model_type == "NMF":
            nmf = NMF(n_components=k, random_state=42, max_iter=200)
            nmf.fit(X)
            topic_terms = []
            for topic in nmf.components_:
                top_indices = topic.argsort()[-10:][::-1]
                topic_terms.append([feature_names[i] for i in top_indices])
            cm = CoherenceModel(topics=topic_terms, texts=docs, dictionary=dictionary, coherence='c_v')
        coherence_values.append(cm.get_coherence())
    return coherence_values, ks


# =============================================
# 10. UI
# =============================================
def main():
    st.set_page_config(page_title="Laser-Microstructure RAG", layout="wide")
    st.title("Laser-Microstructure Interaction RAG (v3.5)")
    st.markdown("*Multi-document synthesis + Focused WordCloud + LDA/NMF*")

    # ---------- Sidebar ----------
    with st.sidebar:
        st.header("Config")
        st.session_state.ollama_base_url = st.text_input("Ollama URL", OLLAMA_BASE_URL)
        models = get_ollama_models()
        if ChatOpenAI:
            models += ["Grok"]
        st.session_state.llm_model = st.selectbox("LLM", models, index=0)
        st.session_state.use_scibert = st.checkbox("SciBERT", True)
        st.session_state.retrieval_k = st.slider("Chunks", 3, 15, 6)

        st.markdown("---")
        st.header("WordCloud Strategy")
        wordcloud_strategies = [
            "Default (Caching)",
            "Chunk Sampling",
            "Batch Processing",
            "Hybrid Extraction",
            "Combined (All Optimizations)",
            "Original (Slowest)"
        ]
        st.session_state.wordcloud_strategy = st.selectbox(
            "Select Strategy",
            wordcloud_strategies,
            index=0,
            help="Choose optimization strategy for wordcloud generation"
        )
        with st.expander("Strategy Details"):
            st.markdown("""
            - **Default (Caching)**: Uses cached results (fastest for repeated runs)
            - **Chunk Sampling**: Processes only representative chunks (5-10x faster)
            - **Batch Processing**: Processes chunks in batches (3-5x faster)
            - **Hybrid Extraction**: Combines rule-based + LLM extraction (2-3x faster)
            - **Combined**: Uses all optimizations together (fastest overall)
            - **Original**: No optimizations (slowest, most accurate)
            """)

        st.markdown("---")
        st.header("WordCloud Postprocessing")
        st.session_state.wordcloud_max_words = st.slider("Top N Words", 50, 200, 150)
        st.session_state.wordcloud_exclude_terms = st.text_input("Exclude Terms (comma-separated)", "")
        st.session_state.wordcloud_font = st.text_input("Custom Font Path (optional)", "")

        if st.button("Clear Cache"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success("Cache cleared!")

    # ---------- File Upload ----------
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.get("processed_files", set())]
        if new_files:
            st.session_state.processed_files = set(st.session_state.get("processed_files", set()))
            with st.spinner(f"Parsing {len(new_files)} PDF(s)..."):
                all_chunks = []
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    chunk_lists = list(executor.map(load_and_chunk_pdf, new_files))
                for f, chunks in zip(new_files, chunk_lists):
                    all_chunks.extend(chunks)
                    st.write(f"→ **{f.name}**: {len(chunks)} chunks")
                    st.session_state.processed_files.add(f.name)

                embeddings = get_embeddings(st.session_state.use_scibert)
                existing_store = st.session_state.get("vectorstore")
                st.session_state.vectorstore = create_vector_store_incremental(all_chunks, embeddings, existing_store)
                st.session_state.existing_chunks = st.session_state.get("existing_chunks", []) + all_chunks
                st.success("Ingestion complete!")
                if len(all_chunks) > 5:
                    clusters = compute_semantic_clusters_fast(all_chunks, embeddings)
                    with st.expander("Semantic Themes"):
                        st.json(clusters, expanded=False)

    # ---------- RAG + Chat ----------
    if st.session_state.get("vectorstore"):
        llm = (
            Ollama(model=st.session_state.llm_model, base_url=st.session_state.ollama_base_url)
            if st.session_state.llm_model != "Grok"
            else ChatOpenAI(model="grok-beta", base_url="https://api.x.ai/v1", api_key=os.getenv("XAI_API_KEY"))
        )
        st.session_state.llm = llm

        if "qa_chain" not in st.session_state:
            st.session_state.qa_chain = create_fusion_rag_chain(
                st.session_state.vectorstore, llm, k=st.session_state.retrieval_k
            )

        for msg in st.session_state.get("messages", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about laser-microstructure..."):
            st.session_state.messages = st.session_state.get("messages", []) + [{"role": "user", "content": prompt}]
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Synthesizing..."):
                    response = st.session_state.qa_chain.invoke({"query": prompt})
                    formatted = format_response_with_sources(response, prompt)
                    st.markdown(formatted)
            st.session_state.messages.append({"role": "assistant", "content": formatted})

        # ---------- WordCloud ----------
        st.markdown("---")
        st.header("Focused TF-IDF WordCloud Generation")
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("Generate WordCloud", type="primary"):
                with st.spinner(f"Generating wordcloud using {st.session_state.wordcloud_strategy}..."):
                    generate_wordcloud_with_strategy(
                        st.session_state.wordcloud_strategy,
                        llm,
                        st.session_state.existing_chunks
                    )
        with col2:
            st.info(f"**Current Strategy**: {st.session_state.wordcloud_strategy}")

        # ---------- Topic Modeling ----------
        st.markdown("---")
        st.header("Topic Modeling (LDA/NMF)")
        model_type = st.selectbox("Model", ["LDA", "NMF"])
        n_topics = st.slider("Topics", 2, 12, 5)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Run Topic Modeling"):
                with st.spinner("Modeling..."):
                    if model_type == "LDA":
                        topics, dist, coh = compute_lda_fast(st.session_state.existing_chunks, n_topics)
                    else:
                        topics, dist, coh = compute_nmf_fast(st.session_state.existing_chunks, n_topics)
                    st.session_state.topic_result = (topics, dist, coh, model_type)
        with col2:
            if st.button("Evaluate Coherence (2–10)"):
                with st.spinner("Sweep..."):
                    scores, ks = evaluate_coherence_sweep(st.session_state.existing_chunks, model_type, max_topics=10)
                    fig, ax = plt.subplots()
                    ax.plot(ks, scores, marker='o')
                    ax.set_xlabel("Number of Topics")
                    ax.set_ylabel("Coherence (C_v)")
                    ax.set_title(f"Optimal k ({model_type})")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    best_k = ks[scores.index(max(scores))]
                    st.success(f"Best: k={best_k} (C_v={max(scores):.3f})")

        if st.session_state.get("topic_result"):
            topics, dist, coh, mtype = st.session_state.topic_result
            st.metric("Coherence (C_v)", f"{coh:.3f}")
            plot_topic_wordclouds(topics, n_topics, mtype)
            df = pd.DataFrame(dist).sort_values("dominant_topic")
            st.dataframe(df, use_container_width=True)
            export_data = {"model": mtype, "coherence": coh, "topics": topics, "distribution": dist}
            st.download_button(
                "Export Topics",
                data=json.dumps(export_data, indent=2),
                file_name=f"{mtype.lower()}_topics.json"
            )
    else:
        st.info("Upload PDFs to start.")


if __name__ == "__main__":
    main()
