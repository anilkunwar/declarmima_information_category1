# =============================================
# LASER-MICROSTRUCTURE RAG + FUSION (FINAL v3.0 - FULLY OPTIMIZED)
# LangChain 0.2+ | CPU-SAFE | 3–5× FASTER | LDA/NMF + COHERENCE + TF-IDF WORDCLOUD
# Updated: November 16, 2025 | PL | 11:53 PM CET
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

#try:
#    from wordcloud import WordCloud
#    import matplotlib.pyplot as plt
#    WORDCLOUD_AVAILABLE = True
#except ImportError:
#    WORDCLOUD_AVAILABLE = False
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
#------------------------------
# --- Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434"
SCIBERT_MODEL = "allenai/scibert_scivocab_uncased"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
BATCH_SIZE = 32
MAX_WORKERS = min(6, os.cpu_count() or 4)
DEFAULT_LLM = "llama3.1:8b"

# =============================================
# 1. PDF → MARKDOWN (CACHED + PARALLEL)
# =============================================
def file_hash(file_obj) -> str:
    file_obj.seek(0)
    h = hashlib.md5(file_obj.read()).hexdigest()
    file_obj.seek(0)
    return h

@st.cache_data(show_spinner=False)
def load_pdf_cached(_file_hash: str, file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        filepath = f.name
    try:
        if PYMUPDF4LLM_AVAILABLE:
            md = pymupdf4llm.to_markdown(filepath)
            os.remove(filepath)
            return md
        elif UNSTRUCTURED_AVAILABLE:
            elements = partition_pdf(filepath, strategy="hi_res")
            md = "\n\n".join([el.text for el in elements if hasattr(el, "text")])
            os.remove(filepath)
            return md
        elif MARKITDOWN_AVAILABLE:
            md = MarkItDown()
            result = md.convert(filepath)
            os.remove(filepath)
            return result
        else:
            loader = PyPDFLoader(filepath)
            pages = loader.load()
            os.remove(filepath)
            return "\n".join([p.page_content for p in pages])
    except Exception as e:
        os.remove(filepath)
        return ""

def parse_pdf_parallel(uploaded_file) -> str:
    file_bytes = uploaded_file.getvalue()
    file_h = file_hash(uploaded_file)
    return load_pdf_cached(file_h, file_bytes)

# =============================================
# 2. CHUNKING (OPTIMIZED)
# =============================================
def chunk_markdown_text_optimized(text: str, source_name: str) -> List[Document]:
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = markdown_splitter.split_text(text)
    final_chunks = []
    for i, chunk in enumerate(chunks):
        subchunks = recursive_splitter.split_text(chunk.page_content)
        base_meta = {
            "source_document": source_name,
            "header_1": chunk.metadata.get("Header 1"),
            "header_2": chunk.metadata.get("Header 2"),
        }
        for j, sc in enumerate(subchunks):
            meta = base_meta.copy()
            meta["chunk_id"] = f"{source_name}_c{i}_s{j}"
            final_chunks.append(Document(page_content=sc, metadata=meta))
    return final_chunks

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
    texts = [c.page_content for c in _chunks]
    metadatas = [c.metadata for c in _chunks]
    if _existing_store is not None:
        _existing_store.add_texts(texts, metadatas=metadatas)
        return _existing_store
    return FAISS.from_texts(texts, _embeddings, metadatas=metadatas)

# =============================================
# 5. RAG CHAIN
# =============================================
def create_fusion_rag_chain(vectorstore, llm, k=6):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    multi_retriever = MultiQueryRetriever.from_llm(retriever=compression_retriever, llm=llm)

    QUESTION_PROMPT = PromptTemplate.from_template(
        "Summarize in 2-3 sentences: laser, alloy, microstructure.\nContext: {context}\nQuestion: {question}\nSummary:"
    )
    FUSION_COMBINE_PROMPT = PromptTemplate.from_template(FUSION_TEMPLATE)

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
# 9. WORDCLOUD (TF-IDF + BATCHED EXTRACTION + SOURCE BOOST)
# =============================================
def generate_wordcloud_tfidf_fast(llm, chunks):
    if not WORDCLOUD_AVAILABLE:
        st.error("Install wordcloud")
        return

    extract_prompt = PromptTemplate.from_template(
        "Extract key phrases (1-3 words) for laser and microstructure. Output ONLY JSON:\n"
        "```json\n{{'laser': [...], 'microstructure': [...]}}\n```\nText: {text}"
    )
    extract_chain = LLMChain(llm=llm, prompt=extract_prompt)

    def extract_from_chunk(chunk):
        try:
            resp = extract_chain.invoke({"text": chunk.page_content})
            raw = resp.get("text", "") or resp.get("result", "")
            cleaned = re.sub(r"^```[a-z]*\n|```$", "", raw.strip(), flags=re.MULTILINE)
            data = json.loads(cleaned) if cleaned else {}
            return {
                "source": chunk.metadata["source_document"],
                "laser": data.get("laser", []),
                "micro": data.get("microstructure", [])
            }
        except:
            return {"source": chunk.metadata["source_document"], "laser": [], "micro": []}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(extract_from_chunk, chunks))

    laser_phrases = [p for r in results for p in r["laser"]]
    micro_phrases = [p for r in results for p in r["micro"]]
    laser_sources = defaultdict(set)
    micro_sources = defaultdict(set)
    for r in results:
        for p in r["laser"]: laser_sources[p].add(r["source"])
        for p in r["micro"]: micro_sources[p].add(r["source"])

    docs_by_source = defaultdict(list)
    for r in results:
        docs_by_source[r["source"]].extend(r["laser"] + r["micro"])
    corpus = [" ".join(docs_by_source[src]) for src in docs_by_source]

    vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r'(?u)\b[\w\-]+\b')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = dict(zip(feature_names, np.ravel(tfidf_matrix.sum(axis=0))))

    all_terms = set(laser_phrases + micro_phrases)
    final_scores = {}
    categories = {}
    for term in all_terms:
        freq = laser_phrases.count(term) + micro_phrases.count(term)
        tfidf = tfidf_scores.get(term.lower(), 0)
        num_sources = len(laser_sources[term] | micro_sources[term])
        score = (tfidf * freq * (1 + math.log(num_sources + 1))) if tfidf > 0 else (freq * (1 + math.log(num_sources + 1)))
        final_scores[term] = max(score, 0.01)
        has_l = term in laser_phrases
        has_m = term in micro_phrases
        categories[term] = "both" if has_l and has_m else ("laser" if has_l else "microstructure")

    def color_func(word, **kwargs):
        cat = categories.get(word, "")
        return {"laser": "#d62728", "microstructure": "#1f77b4", "both": "#9467bd"}.get(cat, "#999999")

    wc = WordCloud(width=1000, height=550, background_color="white", color_func=color_func, max_words=300)
    wc.generate_from_frequencies(final_scores)
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    st.markdown("""
### **Wordcloud: TF-IDF + Cross-Paper Scoring**
- **Red**: Laser parameters  
- **Blue**: Microstructure  
- **Purple**: Both  
- **Size**: `TF-IDF × frequency × (1 + log(#PDFs))`
""")

# =============================================
# 11. LDA + NMF + COHERENCE (OPTIMIZED)
# =============================================
def plot_topic_wordclouds(topics, n_topics, model_type="LDA"):
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
    feature_names = vectorizer.get_feature_names_out()
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
        vectorizer = TfidfVectorizer(
            max_df=0.7, min_df=2, stop_words='english',
            token_pattern=r'(?u)\b\w\w+\b', ngram_range=(1,2)
        )
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    nmf = NMF(n_components=n_topics, random_state=42, max_iter=200)
    W = nmf.fit_transform(X)
    H = nmf.components_

    # === Extract top terms ===
    topics = []
    for i, topic in enumerate(H):
        top_idx = topic.argsort()[-10:][::-1]
        terms = [feature_names[j] for j in top_idx]
        weights = [f"{topic[j]:.3f}" for j in top_idx]
        topics.append({"topic_id": i, "terms": terms, "weights": weights})

    # === Document-topic distribution ===
    doc_dist = []
    for i in range(len(sources)):
        probs = W[i]
        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            probs = np.zeros(n_topics)
        dominant = int(np.argmax(probs))
        doc_dist.append({
            "document": sources[i],
            "dominant_topic": dominant,
            "contributions": [f"{p:.3f}" for p in probs]
        })

    # === COHERENCE: Tokenize NMF terms using dictionary ===
    # Build dictionary from raw texts (same as LDA)
    raw_docs = [text.split() for text in texts]
    dictionary = Dictionary(raw_docs)
    dictionary.filter_extremes(no_below=2, no_above=0.7)

    # Split NMF phrases into tokens that exist in dictionary
    tokenized_topics = []
    for topic in topics:
        topic_tokens = []
        for phrase in topic["terms"]:
            # Split phrase: "laser power" → ["laser", "power"]
            tokens = phrase.split()
            # Keep only tokens that are in dictionary
            valid_tokens = [t for t in tokens if t in dictionary.token2id]
            topic_tokens.extend(valid_tokens)
        # Dedupe and limit
        topic_tokens = list(dict.fromkeys(topic_tokens))[:10]
        tokenized_topics.append(topic_tokens)

    # Only compute coherence if we have valid tokens
    if any(tokenized_topics):
        coherence_model = CoherenceModel(
            topics=tokenized_topics,
            texts=raw_docs,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence = coherence_model.get_coherence()
    else:
        coherence = 0.0
        st.warning("NMF: No valid tokens for coherence. Using 0.0")

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
    st.title("Laser-Microstructure Interaction RAG (v3.0)")
    st.markdown("*Multi-document synthesis + TF-IDF Wordcloud + LDA/NMF + Coherence*")

    with st.sidebar:
        st.header("Config")
        st.session_state.ollama_base_url = st.text_input("Ollama URL", OLLAMA_BASE_URL)
        models = get_ollama_models()
        if ChatOpenAI: models += ["Grok"]
        st.session_state.llm_model = st.selectbox("LLM", models, index=0)
        st.session_state.use_scibert = st.checkbox("SciBERT", True)
        st.session_state.retrieval_k = st.slider("Chunks", 3, 15, 6)
        if st.button("Clear Cache"): st.cache_resource.clear(); st.cache_data.clear(); st.success("Cache cleared!")

    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.get("processed_files", set())]
        if new_files:
            st.session_state.processed_files = set(st.session_state.get("processed_files", set()))
            with st.spinner(f"Parsing {len(new_files)} PDFs (parallel)..."):
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    md_texts = list(executor.map(parse_pdf_parallel, new_files))
                all_chunks = []
                for f, md in zip(new_files, md_texts):
                    chunks = chunk_markdown_text_optimized(md, f.name)
                    all_chunks.extend(chunks)
                    st.write(f"→ {f.name}: {len(chunks)} chunks")
                    st.session_state.processed_files.add(f.name)
                st.session_state.existing_chunks = all_chunks

                embeddings = get_embeddings(st.session_state.use_scibert)
                existing_store = st.session_state.get("vectorstore")
                st.session_state.vectorstore = create_vector_store_incremental(all_chunks, embeddings, existing_store)
                st.success("Ingestion complete!")

                if len(all_chunks) > 5:
                    clusters = compute_semantic_clusters_fast(all_chunks, embeddings)
                    with st.expander("Semantic Themes"): st.json(clusters, expanded=False)

    if st.session_state.get("vectorstore"):
        llm = Ollama(model=st.session_state.llm_model, base_url=st.session_state.ollama_base_url) if st.session_state.llm_model != "Grok" else ChatOpenAI(model="grok-beta", base_url="https://api.x.ai/v1", api_key=os.getenv("XAI_API_KEY"))
        if "qa_chain" not in st.session_state:
            st.session_state.qa_chain = create_fusion_rag_chain(st.session_state.vectorstore, llm, k=st.session_state.retrieval_k)

        # Chat
        for msg in st.session_state.get("messages", []):
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        if prompt := st.chat_input("Ask about laser-microstructure..."):
            st.session_state.messages = st.session_state.get("messages", []) + [{"role": "user", "content": prompt}]
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Synthesizing..."):
                    response = st.session_state.qa_chain.invoke({"query": prompt})
                    formatted = format_response_with_sources(response, prompt)
                    st.markdown(formatted)
            st.session_state.messages.append({"role": "assistant", "content": formatted})

        # Wordcloud
        if st.button("Generate TF-IDF Wordcloud"):
            with st.spinner("Fast wordcloud..."):
                generate_wordcloud_tfidf_fast(llm, st.session_state.existing_chunks)

        # Topic Modeling
        st.markdown("---")
        st.header("Topic Modeling (LDA/NMF)")
        model_type = st.selectbox("Model", ["LDA", "NMF"])
        n_topics = st.slider("Topics", 2, 12, 5)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Run"):
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
            st.download_button("Export Topics", data=json.dumps(export_data, indent=2), file_name=f"{mtype.lower()}_topics.json")

    else:
        st.info("Upload PDFs to start.")

if __name__ == "__main__":
    main()
