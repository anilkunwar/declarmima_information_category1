#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LASER MICROSTRUCTURE RAG CHATBOT - ENHANCED WITH PHYSICS-AWARE EVALUATION & STATIC VISUALIZATIONS
========================================================================================
✅ Zero API keys required - all models run locally
✅ Supports Hugging Face transformers models AND Ollama local models
✅ Laser-microstructure domain specialization
✅ PDF/text/CSV/TDB document ingestion with FAISS vector storage
✅ 🎯 SOURCE CITATION WITH HUMAN-READABLE IDs
✅ 🔗 MULTI-DOCUMENT REASONING: Cross-document property extraction, fusion, and comparison
✅ 📊 ENHANCED FUSION EFFICIENCY: Robust metrics computation with fallbacks
✅ 📋 TABULAR OUTPUT: Automatic generation of comparison tables
✅ 🔍 RETRIEVAL QUALITY METRICS: Recall@k, Precision@k, MRR, NDCG
✅ 🧮 PHYSICS VALIDATION: Thermodynamic consistency, numerical bounds checking
✅ 🎯 BENCHMARK SUITE: 7 domain-specific test queries for DECLARMIMA
✅ 📈 VISUALIZATION METRICS: SSIM, PSNR, morphological analysis
✅ 🔬 NEW: STATIC VISUALIZATION ENGINE (Sunburst, NetworkX, Radar, Chord/Heatmap)
✅ 🔬 NEW: TOPOLOGY-AWARE ENTITY RANKING & CROSS-DOCUMENT CENTRALITY
✅ Confidence scoring, relevance filtering, and uncertainty quantification
✅ Responsive UI with streaming-like output simulation
✅ Memory-efficient loading with quantization support
✅ BUG FIX: Resolved StreamlitDuplicateElementKey for interactive fusion selectboxes

ENHANCEMENTS APPLIED:
• Retrieval quality metrics (Recall@k, Precision@k, MRR, NDCG, context relevance)
• Physics-aware hallucination detection (thermodynamic consistency, numerical bounds)
• Microstructure field comparison (SSIM, PSNR, morphological metrics)
• Equation consistency checker (Gibbs energy, Arrhenius diffusion, Fick's laws)
• Structured data loader (CSV, TDB thermodynamic databases)
• Benchmark query suite for DECLARMIMA domain
• Streamlit evaluation dashboard with trends visualization
• 🆕 Scientific visualization engine: static plots for papers (PNG/SVG ready)
• 🆕 Hierarchical sunburst charts for methods & materials
• 🆕 Cross-document entity centrality scoring with taxonomy mapping
• 🆕 Document coverage radar charts for comparative analysis
• 🆕 Fixed UI element duplication errors in chat history rendering

Deploy to Streamlit Cloud with requirements.txt below.
For local use with Ollama: install ollama Python library and run `ollama pull <model>`
For enhanced metadata extraction: pip install pdf2doi crossrefapi (optional)
For advanced table parsing: pip install pandas tabulate lxml (recommended)
For visualization metrics & plots: pip install scikit-image opencv-python networkx matplotlib plotly
"""
import streamlit as st
import os
import tempfile
import time
import re
import json
import torch
import numpy as np
from io import BytesIO
from typing import List, Dict, Optional, Tuple, Union, Any, Set
from datetime import datetime
import sys
import subprocess
import platform
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
import logging
import hashlib
import itertools

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LangChain / RAG imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Transformers for local LLM inference
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, AutoModelForCausalLM,
    pipeline, set_seed, BitsAndBytesConfig
)

# Optional libraries
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    import pdf2doi
    PDF2DOI_AVAILABLE = True
except (ImportError, PermissionError, Exception):
    PDF2DOI_AVAILABLE = False

try:
    from crossrefapi import CrossrefAPI
    CROSSREF_AVAILABLE = True
except ImportError:
    CROSSREF_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import networkx as nx
    import matplotlib
    matplotlib.use('Agg')  # Headless backend for Streamlit
    import matplotlib.pyplot as plt
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# =============================================
# GLOBAL CONFIGURATION - LASER MICROSTRUCTURE DOMAIN
# =============================================
LOCAL_LLM_OPTIONS = {
    "GPT-2 (1.5B, fastest startup, CPU OK)": "gpt2",
    "Qwen2-0.5B-Instruct (best JSON, recommended)": "Qwen/Qwen2-0.5B-Instruct",
    "Qwen2.5-0.5B-Instruct (newest, best reasoning)": "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama-1.1B-Chat (balanced small model)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen2.5-1.5B-Instruct (efficient mid-size)": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct (strong reasoning)": "Qwen/Qwen2.5-3B-Instruct",
    "Mistral-7B-Instruct-v0.3 (reliable & efficient)": "mistralai/Mistral-7B-Instruct-v0.3",
    "Llama-3.2-3B-Instruct (Meta's latest small)": "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen2.5-7B-Instruct (excellent all-rounder)": "Qwen/Qwen2.5-7B-Instruct",
    "Llama-3.1-8B-Instruct (most popular balanced)": "meta-llama/Llama-3.1-8B-Instruct",
    "Gemma-2-9B-it (Google's latest, great logic)": "google/gemma-2-9b-it",
    "Falcon-7B-Instruct (lightweight & modern)": "tiiuae/falcon-7b-instruct",
    "[Ollama] qwen2.5:0.5b (via ollama serve)": "ollama:qwen2.5:0.5b",
    "[Ollama] qwen2.5:1.5b (via ollama serve)": "ollama:qwen2.5:1.5b",
    "[Ollama] qwen2.5:7b (via ollama serve)": "ollama:qwen2.5:7b",
    "[Ollama] qwen2.5:14b (via ollama serve) 🔥": "ollama:qwen2.5:14b",
    "[Ollama] llama3.1:8b (via ollama serve)": "ollama:llama3.1:8b",
    "[Ollama] mistral:7b (via ollama serve)": "ollama:mistral:7b",
    "[Ollama] gemma2:9b (via ollama serve)": "ollama:gemma2:9b",
    "[Ollama] falcon3:10b (via ollama serve)": "ollama:falcon3:10b",
}

LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LASER_DOMAIN_CONFIG = {
    "chunk_size": 800,
    "chunk_overlap": 150,
    "retrieval_k": 4,
    "score_threshold": 0.25,
    "max_context_tokens": 1024,
    "max_new_tokens": 256,
    "temperature": 0.1,
}

LASER_KEYWORDS = {
    "ablation": ["ablation", "material removal", "threshold fluence", "laser ablation"],
    "plasma": ["plasma formation", "ionization", "electron density", "plume"],
    "thermal": ["heat affected zone", "melting", "thermal diffusion", "resolidification"],
    "ultrafast": ["femtosecond", "picosecond", "pulse duration", "ultrafast laser"],
    "morphology": ["ripples", "LIPSS", "surface structuring", "periodic structures"],
    "parameters": ["fluence", "wavelength", "pulse energy", "repetition rate", "spot size"],
    "materials": ["silicon", "steel", "titanium", "polymer", "glass", "ceramic", "aluminum", "composite", "alloy"],
    "characterization": ["SEM", "AFM", "profilometry", "spectroscopy", "microscopy"],
    "mechanical": ["hardness", "strength", "yield", "tensile", "elongation", "ductility", "modulus"],
}

MODEL_MEMORY_ESTIMATES = {
    "gpt2": {"params": "1.5B", "vram_fp16": "~3GB", "vram_4bit": "~1GB", "cpu_ok": True},
    "Qwen/Qwen2-0.5B-Instruct": {"params": "0.5B", "vram_fp16": "~1GB", "vram_4bit": "~400MB", "cpu_ok": True},
    "Qwen/Qwen2.5-0.5B-Instruct": {"params": "0.5B", "vram_fp16": "~1GB", "vram_4bit": "~400MB", "cpu_ok": True},
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {"params": "1.1B", "vram_fp16": "~2.5GB", "vram_4bit": "~800MB", "cpu_ok": True},
    "Qwen/Qwen2.5-1.5B-Instruct": {"params": "1.5B", "vram_fp16": "~3.5GB", "vram_4bit": "~1.2GB", "cpu_ok": False},
    "Qwen/Qwen2.5-3B-Instruct": {"params": "3B", "vram_fp16": "~6GB", "vram_4bit": "~2GB", "cpu_ok": False},
    "mistralai/Mistral-7B-Instruct-v0.3": {"params": "7B", "vram_fp16": "~14GB", "vram_4bit": "~4.5GB", "cpu_ok": False},
    "meta-llama/Llama-3.2-3B-Instruct": {"params": "3B", "vram_fp16": "~6GB", "vram_4bit": "~2GB", "cpu_ok": False},
    "Qwen/Qwen2.5-7B-Instruct": {"params": "7B", "vram_fp16": "~14GB", "vram_4bit": "~4.5GB", "cpu_ok": False},
    "Qwen/Qwen2.5-14B-Instruct": {"params": "14B", "vram_fp16": "~28GB", "vram_4bit": "~9GB", "cpu_ok": False},
    "meta-llama/Llama-3.1-8B-Instruct": {"params": "8B", "vram_fp16": "~16GB", "vram_4bit": "~5GB", "cpu_ok": False},
    "google/gemma-2-9b-it": {"params": "9B", "vram_fp16": "~18GB", "vram_4bit": "~6GB", "cpu_ok": False},
    "tiiuae/falcon-7b-instruct": {"params": "7B", "vram_fp16": "~14GB", "vram_4bit": "~4.5GB", "cpu_ok": False},
}

# =============================================
# ENTITY TAXONOMY & CLASSIFICATION
# =============================================
ENTITY_TAXONOMY = {
    "MATERIAL": {
        "Pure Element": ["silicon", "titanium", "copper", "aluminum", "tungsten", "niobium", "zirconium", "carbon"],
        "Binary Alloy": ["sn-cu", "al-cr", "ti-al", "ni-ti"],
        "Ternary Alloy": ["sn-ag-cu", "sac", "al-si-mg", "ti-al-v"],
        "Multicomponent / HEA": ["alcrfeni", "cocrfeni", "hea", "mpea", "high entropy alloy"],
        "Compound / Ceramic": ["sio2", "al2o3", "zro2", "glass", "sic", "tin"],
        "Polymer": ["pmma", "ptfe", "pc", "pdms", "polymer"],
        "Steel": ["stainless steel", "316l", "steel", "tool steel", "martensitic"]
    },
    "METHOD": {
        "Experimental: Microscopy": ["sem", "afm", "tem", "ebsd", "optical microscopy", "confocal"],
        "Experimental: Spectroscopy": ["raman", "xrd", "edx", "eds", "xps", "spectroscopy"],
        "Experimental: Tomography": ["x-ray imaging", "tomography", "micro-ct", "synchrotron"],
        "Computational: Atomistic": ["md", "dft", "molecular dynamics", "density functional"],
        "Computational: Continuum": ["finite element", "phase field", "calphad", "fem", "thermodynamic"],
        "Computational: Data-Driven": ["machine learning", "cnn", "neural network", "random forest", "svm", "ml"]
    },
    "PHENOMENON": {
        "Thermal": ["melting", "heat affected zone", "resolidification", "thermal diffusion", "cooling rate"],
        "Optical / Plasma": ["ablation", "plasma", "lipss", "ripples", "self-organization", "ionization"],
        "Mechanical": ["residual stress", "porosity", "cracking", "spatter", "fatigue", "delamination"]
    },
    "PARAMETER": {
        "Laser Source": ["wavelength", "pulse_duration", "repetition_rate", "power", "pulse energy"],
        "Process": ["scan_speed", "hatch_distance", "spot_size", "overlap", "feed rate"],
        "Outcome": ["roughness", "periodicity", "threshold", "hardness", "yield strength"]
    }
}

def classify_entity(normalized: str) -> Tuple[str, str]:
    """
    Returns (category, subcategory) or ('UNKNOWN', 'UNKNOWN').
    Normalized string should be lowercase.
    """
    norm_lower = normalized.lower()
    for cat, subcats in ENTITY_TAXONOMY.items():
        for sub, aliases in subcats.items():
            if any(a in norm_lower for a in aliases):
                return cat, sub
    return "UNKNOWN", "UNKNOWN"

@dataclass
class ScientificEntity:
    name: str
    normalized: str
    category: str = "UNKNOWN"
    subcategory: str = "UNKNOWN"
    count: int = 1
    doc_sources: Set[str] = field(default_factory=set)
    confidence: float = 1.0

    def __post_init__(self):
        self.normalized = self.normalized.lower().strip()
        if self.category == "UNKNOWN":
            self.category, self.subcategory = classify_entity(self.normalized)

# =============================================
# RETRIEVAL QUALITY METRICS MODULE
# =============================================
@dataclass
class RetrievalMetrics:
    """Container for retrieval quality metrics"""
    recall_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    mrr: float
    context_relevance: float
    ndcg_at_k: Dict[int, float]
    coverage: float

class RetrievalEvaluator:
    """
    Evaluates RAG retrieval quality against ground-truth relevance judgments.
    For DECLARMIMA: ground truth = manually labeled relevant chunks per query.
    """
    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.query_history: List[Dict] = []

    def compute_recall_at_k(self, retrieved: List[str], relevant: Set[str], k_values=[3, 5, 10]) -> Dict[int, float]:
        results = {}
        for k in k_values:
            retrieved_k = set(retrieved[:k])
            if len(relevant) == 0:
                results[k] = 0.0
            else:
                results[k] = len(retrieved_k & relevant) / len(relevant)
        return results

    def compute_precision_at_k(self, retrieved: List[str], relevant: Set[str], k_values=[3, 5, 10]) -> Dict[int, float]:
        results = {}
        for k in k_values:
            retrieved_k = set(retrieved[:k])
            if k == 0:
                results[k] = 0.0
            else:
                results[k] = len(retrieved_k & relevant) / k
        return results

    def compute_mrr(self, retrieved: List[str], relevant: Set[str]) -> float:
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                return 1.0 / i
        return 0.0

    def compute_ndcg_at_k(self, retrieved: List[str], relevance_scores: Dict[str, float], k_values=[3, 5, 10]) -> Dict[int, float]:
        results = {}
        for k in k_values:
            dcg = 0.0
            for i, doc_id in enumerate(retrieved[:k], 1):
                rel = relevance_scores.get(doc_id, 0.0)
                dcg += rel / np.log2(i + 1)
            ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
            idcg = sum(rel / np.log2(i + 1) for i, rel in enumerate(ideal_rels, 1))
            results[k] = dcg / idcg if idcg > 0 else 0.0
        return results

    def compute_context_relevance(self, query: str, retrieved_chunks: List[Document]) -> float:
        if not retrieved_chunks or not SKLEARN_AVAILABLE:
            return 0.0
        try:
            query_emb = self.embed_model.encode([query], show_progress_bar=False)
            chunk_texts = [c.page_content[:500] for c in retrieved_chunks]
            chunk_embs = self.embed_model.encode(chunk_texts, show_progress_bar=False)
            similarities = cosine_similarity(query_emb, chunk_embs)[0]
            return float(np.mean(similarities))
        except:
            return 0.0

    def evaluate_query(self, query: str, retrieved_docs: List[Document],
                       relevant_doc_ids: Set[str],
                       relevance_scores: Optional[Dict[str, float]] = None) -> RetrievalMetrics:
        retrieved_ids = [f"{d.metadata['source']}:{d.metadata['chunk_index']}" for d in retrieved_docs]
        metrics = RetrievalMetrics(
            recall_at_k=self.compute_recall_at_k(retrieved_ids, relevant_doc_ids),
            precision_at_k=self.compute_precision_at_k(retrieved_ids, relevant_doc_ids),
            mrr=self.compute_mrr(retrieved_ids, relevant_doc_ids),
            context_relevance=self.compute_context_relevance(query, retrieved_docs),
            ndcg_at_k=self.compute_ndcg_at_k(retrieved_ids, relevance_scores or {}, [3, 5, 10]),
            coverage=len(set(retrieved_ids) & relevant_doc_ids) / len(relevant_doc_ids) if relevant_doc_ids else 0.0
        )
        self.query_history.append({
            "query": query,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        })
        return metrics

    def get_aggregate_report(self) -> pd.DataFrame:
        if not self.query_history or not PANDAS_AVAILABLE:
            return pd.DataFrame()
        records = []
        for entry in self.query_history:
            m = entry["metrics"]
            records.append({
                "query": entry["query"][:50],
                "recall@3": m.recall_at_k.get(3, 0),
                "recall@5": m.recall_at_k.get(5, 0),
                "precision@3": m.precision_at_k.get(3, 0),
                "precision@5": m.precision_at_k.get(5, 0),
                "mrr": m.mrr,
                "context_relevance": m.context_relevance,
                "ndcg@5": m.ndcg_at_k.get(5, 0),
                "coverage": m.coverage
            })
        df = pd.DataFrame(records)
        if len(df) > 0:
            means = df.select_dtypes(include=[np.number]).mean()
            means["query"] = "AVERAGE"
            df = pd.concat([df, pd.DataFrame([means])], ignore_index=True)
        return df

# =============================================
# VISUALIZATION ENGINE
# =============================================
class VisualizationEngine:
    """
    Generates publication-ready static visualizations from extracted entities and knowledge graphs.
    Supports NetworkX, Matplotlib, and Plotly backends.
    """
    def __init__(self, entities: Dict[str, ScientificEntity], doc_metadata: Dict[str, Any]):
        self.entities = entities
        self.doc_metadata = doc_metadata or {}

    def top_entities_bar(self, n: int = 15) -> Optional[go.Figure]:
        if not PLOTLY_AVAILABLE or not self.entities:
            return None
        sorted_ents = sorted(self.entities.values(), key=lambda e: e.count * len(e.doc_sources), reverse=True)[:n]
        df = pd.DataFrame([{"Entity": e.name, "Count": e.count, "Docs": len(e.doc_sources)} for e in sorted_ents])
        fig = px.bar(df, x="Entity", y="Count", title="Top-N Cross-Document Entities",
                     color="Docs", color_continuous_scale="Blues")
        fig.update_layout(xaxis_tickangle=-45)
        return fig

    def static_network_graph(self, n: int = 15) -> Optional[plt.Figure]:
        if not NETWORKX_AVAILABLE or not self.entities:
            return None
        sorted_ents = sorted(self.entities.values(), key=lambda e: e.count * len(e.doc_sources), reverse=True)[:n]
        G = nx.Graph()
        
        doc_nodes = []
        ent_nodes = []
        
        for e in sorted_ents:
            ent_name = e.normalized
            G.add_node(ent_name, bipartite=1, node_type='entity')
            ent_nodes.append(ent_name)
            for doc in e.doc_sources:
                doc_id = Path(doc).stem
                if not G.has_node(doc_id):
                    G.add_node(doc_id, bipartite=0, node_type='doc')
                    doc_nodes.append(doc_id)
                G.add_edge(doc_id, ent_name, weight=e.confidence)

        if not G.edges():
            return None

        pos = nx.spring_layout(G, k=0.4, iterations=50, seed=42)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw docs
        nx.draw_networkx_nodes(G, pos, nodelist=doc_nodes, node_color="#3b82f6", 
                               node_shape="s", node_size=500, ax=ax, label="Documents")
        # Draw entities
        nx.draw_networkx_nodes(G, pos, nodelist=ent_nodes, node_color="#f59e0b", 
                               node_shape="o", node_size=300, ax=ax, label="Entities")
        
        nx.draw_networkx_edges(G, pos, alpha=0.4, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        ax.set_title("Static Document-Entity Knowledge Graph", fontsize=14, pad=15)
        ax.legend()
        ax.axis("off")
        plt.tight_layout()
        return fig

    def methods_sunburst(self) -> Optional[go.Figure]:
        if not PLOTLY_AVAILABLE:
            return None
        rows = []
        for norm, ent in self.entities.items():
            if ent.category == "METHOD":
                sub_parts = ent.subcategory.split(": ")
                if len(sub_parts) == 2:
                    cat_main, tech = sub_parts
                else:
                    cat_main, tech = ent.subcategory, ent.subcategory
                
                rows.append({
                    "category": ent.category,
                    "subcategory": cat_main,
                    "technique": tech,
                    "entity": ent.name,
                    "value": ent.count,
                    "docs": len(ent.doc_sources)
                })
        
        if not rows:
            return None
        df = pd.DataFrame(rows)
        fig = px.sunburst(
            df, path=["category", "subcategory", "technique", "entity"], 
            values="value", color="docs", color_continuous_scale="YlOrRd",
            title="Hierarchical Methods Taxonomy"
        )
        return fig

    def materials_sunburst(self) -> Optional[go.Figure]:
        if not PLOTLY_AVAILABLE:
            return None
        rows = []
        for norm, ent in self.entities.items():
            if ent.category == "MATERIAL":
                rows.append({
                    "root": "Materials",
                    "class": ent.subcategory,
                    "entity": ent.name,
                    "value": ent.count,
                    "doc_count": len(ent.doc_sources)
                })
        
        if not rows:
            return None
        df = pd.DataFrame(rows)
        fig = px.sunburst(
            df, path=["root", "class", "entity"], values="value",
            color="doc_count", color_continuous_scale="Greens",
            title="Material System Hierarchy"
        )
        return fig

    def radar_docs(self) -> Optional[go.Figure]:
        if not PLOTLY_AVAILABLE:
            return None
        
        categories = ["Materials", "Methods", "Phenomena", "Parameters"]
        axis_map = {"Materials": "MATERIAL", "Methods": "METHOD", "Phenomena": "PHENOMENON", "Parameters": "PARAMETER"}
        
        # Collect all doc IDs
        all_docs = set()
        for e in self.entities.values():
            all_docs.update(e.doc_sources)
            
        if not all_docs:
            return None
            
        fig = go.Figure()
        for doc in list(all_docs)[:5]:  # Limit to 5 docs for readability
            values = []
            for cat_name, cat_key in axis_map.items():
                count = sum(1 for e in self.entities.values() if e.category == cat_key and doc in e.doc_sources)
                values.append(count)
            values += values[:1]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=Path(doc).stem
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(5, max([sum(1 for e in self.entities.values() if e.category == 'MATERIAL' and d in e.doc_sources) for d in all_docs[:1]]))])),
            showlegend=True, title="Document Coverage Profiles"
        )
        return fig

    def cooccurrence_heatmap(self, n: int = 10) -> Optional[go.Figure]:
        if not PLOTLY_AVAILABLE:
            return None
        top = [e for e, _ in sorted(self.entities.items(), key=lambda x: x[1].count * len(x[1].doc_sources), reverse=True)[:n]]
        mat = np.zeros((len(top), len(top)))
        
        doc_groups = defaultdict(set)
        for norm, ent in self.entities.items():
            for doc in ent.doc_sources:
                doc_groups[doc].add(norm)
                
        for doc_ents in doc_groups.values():
            for i, e1 in enumerate(top):
                for j, e2 in enumerate(top):
                    if i != j and e1 in doc_ents and e2 in doc_ents:
                        mat[i][j] += 1
                        
        fig = go.Figure(data=go.Heatmap(
            z=mat, x=top, y=top, colorscale="YlOrRd", zmin=0
        ))
        fig.update_layout(title="Entity Co-occurrence across Documents", width=700, height=700)
        return fig

# =============================================
# PHYSICS-AWARE HALLUCINATION DETECTOR
# =============================================
class PhysicsFaithfulnessChecker:
    """
    Detects when LLM outputs contradict retrieved context or violate physical laws.
    Critical for DECLARMIMA: thermodynamic consistency, numerical correctness.
    """
    PHYSICAL_CONSTRAINTS = {
        "temperature": {"min": 0, "max": 10000, "unit": "K"},
        "energy_density": {"min": 0, "max": 1000, "unit": "J/mm³"},
        "diffusion_coefficient": {"min": 0, "max": 1e-3, "unit": "m²/s"},
        "grain_size": {"min": 1e-9, "max": 1e-2, "unit": "m"},
        "hardness": {"min": 0, "max": 3000, "unit": "HV"},
        "yield_strength": {"min": 0, "max": 5000, "unit": "MPa"},
    }

    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.violation_log: List[Dict] = []

    def extract_numerical_claims(self, text: str) -> List[Dict]:
        patterns = [
            r'(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(?:J/mm³|J\s*mm[-3⁻³])',
            r'(\d+(?:\.\d+)?)\s*(?:K|°C|°F)',
            r'(\d+(?:\.\d+)?)\s*(?:HV|Vickers|GPa|MPa)',
            r'(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(?:m²/s|m²/s|cm²/s)',
            r'(\d+(?:\.\d+)?)\s*(?:μm|um|nm|mm)',
        ]
        claims = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.I):
                claims.append({
                    "value": float(match.group(1)),
                    "context": text[max(0, match.start()-30):min(len(text), match.end()+30)],
                    "span": (match.start(), match.end())
                })
        return claims

    def check_numerical_bounds(self, claims: List[Dict]) -> List[Dict]:
        violations = []
        for claim in claims:
            ctx_lower = claim["context"].lower()
            param_type = None
            if "energy density" in ctx_lower or "j/mm" in ctx_lower:
                param_type = "energy_density"
            elif "temperature" in ctx_lower or "k" in ctx_lower:
                param_type = "temperature"
            elif "hardness" in ctx_lower or "hv" in ctx_lower:
                param_type = "hardness"
            elif "diffusion" in ctx_lower:
                param_type = "diffusion_coefficient"
            elif "grain" in ctx_lower:
                param_type = "grain_size"
                
            if param_type and param_type in self.PHYSICAL_CONSTRAINTS:
                bounds = self.PHYSICAL_CONSTRAINTS[param_type]
                val = claim["value"]
                if val < bounds["min"] or val > bounds["max"]:
                    violations.append({
                        "type": "physical_bound_violation",
                        "parameter": param_type,
                        "value": val,
                        "bounds": bounds,
                        "context": claim["context"],
                        "severity": "HIGH" if val < 0 else "MEDIUM"
                    })
        return violations

    def check_thermodynamic_consistency(self, text: str) -> List[Dict]:
        violations = []
        phase_fractions = re.findall(r'(\d+(?:\.\d+)?)\s*%?\s*(?:phase|fraction)', text, re.I)
        if phase_fractions:
            total = sum(float(f) for f in phase_fractions if float(f) < 100)
            if total > 100.1 or (total < 99.9 and len(phase_fractions) > 1):
                violations.append({
                    "type": "thermodynamic_inconsistency",
                    "rule": "Phase fractions must sum to 100%",
                    "calculated_sum": total,
                    "severity": "HIGH"
                })
        return violations

    def check_faithfulness_to_context(self, llm_answer: str, retrieved_chunks: List[Document]) -> Dict:
        if not SKLEARN_AVAILABLE or not retrieved_chunks:
            return {
                "faithfulness_score": 0.5,
                "max_context_similarity": 0.0,
                "mean_context_similarity": 0.0,
                "keyword_overlap": 0.0,
                "hallucinated_numbers": [],
                "is_faithful": False
            }
        try:
            answer_emb = self.embed_model.encode([llm_answer], show_progress_bar=False)
            chunk_texts = [c.page_content[:500] for c in retrieved_chunks]
            chunk_embs = self.embed_model.encode(chunk_texts, show_progress_bar=False)
            
            similarities = cosine_similarity(answer_emb, chunk_embs)[0]
            max_sim = float(np.max(similarities))
            mean_sim = float(np.mean(similarities))
            
            answer_words = set(re.findall(r'\b\w+\b', llm_answer.lower()))
            chunk_words = set()
            for c in retrieved_chunks:
                chunk_words.update(re.findall(r'\b\w+\b', c.page_content.lower()))
            overlap = len(answer_words & chunk_words) / len(answer_words) if answer_words else 0
            
            answer_numbers = set(re.findall(r'\d+\.\d+', llm_answer))
            chunk_numbers = set()
            for c in retrieved_chunks:
                chunk_numbers.update(re.findall(r'\d+\.\d+', c.page_content))
            hallucinated_numbers = answer_numbers - chunk_numbers
            
            faithfulness_score = (
                0.4 * max_sim +
                0.3 * overlap +
                0.3 * (1.0 if len(hallucinated_numbers) < 3 else 0.5)
            )
            return {
                "faithfulness_score": float(np.clip(faithfulness_score, 0, 1)),
                "max_context_similarity": max_sim,
                "mean_context_similarity": mean_sim,
                "keyword_overlap": overlap,
                "hallucinated_numbers": list(hallucinated_numbers),
                "is_faithful": faithfulness_score > 0.7 and len(hallucinated_numbers) < 3
            }
        except:
            return {
                "faithfulness_score": 0.5,
                "max_context_similarity": 0.0,
                "mean_context_similarity": 0.0,
                "keyword_overlap": 0.0,
                "hallucinated_numbers": [],
                "is_faithful": False
            }

    def full_check(self, llm_answer: str, retrieved_chunks: List[Document]) -> Dict:
        numerical_claims = self.extract_numerical_claims(llm_answer)
        bound_violations = self.check_numerical_bounds(numerical_claims)
        thermo_violations = self.check_thermodynamic_consistency(llm_answer)
        faithfulness = self.check_faithfulness_to_context(llm_answer, retrieved_chunks)
        
        result = {
            "faithfulness": faithfulness,
            "numerical_claims": len(numerical_claims),
            "physical_violations": bound_violations,
            "thermodynamic_violations": thermo_violations,
            "total_violations": len(bound_violations) + len(thermo_violations),
            "is_physics_valid": len(bound_violations) == 0 and len(thermo_violations) == 0,
            "overall_trust_score": self._compute_trust_score(faithfulness, bound_violations, thermo_violations)
        }
        self.violation_log.append(result)
        return result

    def _compute_trust_score(self, faithfulness, bound_violations, thermo_violations) -> float:
        base = faithfulness["faithfulness_score"]
        penalty = 0.1 * len(bound_violations) + 0.2 * len(thermo_violations)
        return float(np.clip(base - penalty, 0, 1))

    def render_violation_report(self):
        if not self.violation_log:
            st.info("No violations detected yet.")
            return
        latest = self.violation_log[-1]
        col1, col2, col3 = st.columns(3)
        col1.metric("Trust Score", f"{latest['overall_trust_score']:.2f}")
        col2.metric("Physical Violations", len(latest['physical_violations']))
        col3.metric("Thermo Violations", len(latest['thermodynamic_violations']))
        
        if latest['physical_violations']:
            with st.expander("⚠️ Physical Bound Violations"):
                for v in latest['physical_violations']:
                    st.error(f"**{v['parameter']}**: {v['value']} outside bounds [{v['bounds']['min']}, {v['bounds']['max']}]")
                    st.caption(f"Context: ...{v['context']}...")
        if latest['thermodynamic_violations']:
            with st.expander("⚠️ Thermodynamic Inconsistencies"):
                for v in latest['thermodynamic_violations']:
                    st.error(f"**{v['rule']}**")
                    st.caption(f"Severity: {v['severity']}")
        if latest['faithfulness']['hallucinated_numbers']:
            with st.expander("🔍 Potentially Hallucinated Values"):
                st.warning(f"Numbers in answer not found in retrieved context: {latest['faithfulness']['hallucinated_numbers']}")

# =============================================
# MICROSTRUCTURE FIELD COMPARISON METRICS
# =============================================
class MicrostructureComparator:
    """
    Compare LLM-generated or RAG-retrieved microstructure descriptions
    against ground-truth simulation outputs (phase-field, PINN, experimental).
    """
    def __init__(self):
        self.comparison_history: List[Dict] = []

    def load_field_data(self, file_path: str) -> Optional[np.ndarray]:
        if not os.path.exists(file_path):
            return None
        try:
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path).values
            elif file_path.endswith('.npy'):
                return np.load(file_path)
            elif file_path.endswith(('.png', '.jpg', '.tif')) and CV2_AVAILABLE:
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                return img.astype(np.float32) / 255.0
            else:
                return None
        except:
            return None

    def compute_rmse(self, predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        return float(np.sqrt(np.mean((predicted - ground_truth) ** 2)))

    def compute_mae(self, predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        return float(np.mean(np.abs(predicted - ground_truth)))

    def compute_ssim(self, predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        if not SKIMAGE_AVAILABLE:
            return 0.0
        try:
            pred_norm = ((predicted - predicted.min()) / (predicted.max() - predicted.min() + 1e-8) * 255).astype(np.uint8)
            gt_norm = ((ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min() + 1e-8) * 255).astype(np.uint8)
            return float(ssim(pred_norm, gt_norm))
        except:
            return 0.0

    def compute_psnr(self, predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        mse = np.mean((predicted - ground_truth) ** 2)
        if mse == 0:
            return float('inf')
        max_val = max(predicted.max(), ground_truth.max())
        return float(20 * np.log10(max_val / np.sqrt(mse)))

    def compute_morphological_metrics(self, binary_image: np.ndarray) -> Dict:
        if not SKIMAGE_AVAILABLE:
            return {"n_grains": 0, "avg_grain_size": 0, "grain_size_std": 0, "interface_density": 0, "phase_fraction": 0}
        try:
            labeled = measure.label(binary_image, connectivity=2)
            regions = measure.regionprops(labeled)
            if not regions:
                return {"n_grains": 0, "avg_grain_size": 0, "grain_size_std": 0, "interface_density": 0, "phase_fraction": 0}
            
            areas = [r.area for r in regions]
            perimeters = [r.perimeter for r in regions]
            total_area = binary_image.size
            phase_area = np.sum(binary_image)
            
            return {
                "n_grains": len(regions),
                "avg_grain_size": float(np.mean(areas)),
                "grain_size_std": float(np.std(areas)),
                "median_grain_size": float(np.median(areas)),
                "max_grain_size": float(np.max(areas)),
                "interface_density": float(np.sum(perimeters) / total_area),
                "phase_fraction": float(phase_area / total_area),
                "grain_size_cv": float(np.std(areas) / np.mean(areas)) if np.mean(areas) > 0 else 0
            }
        except:
            return {"n_grains": 0, "avg_grain_size": 0, "grain_size_std": 0, "interface_density": 0, "phase_fraction": 0}

    def compare_fields(self, predicted_path: str, ground_truth_path: str, field_name: str = "concentration") -> Optional[Dict]:
        pred = self.load_field_data(predicted_path)
        gt = self.load_field_data(ground_truth_path)
        
        if pred is None or gt is None:
            return None
        if pred.shape != gt.shape:
            if CV2_AVAILABLE:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
            else:
                return None
        
        result = {
            "field_name": field_name,
            "rmse": self.compute_rmse(pred, gt),
            "mae": self.compute_mae(pred, gt),
            "ssim": self.compute_ssim(pred, gt),
            "psnr": self.compute_psnr(pred, gt),
            "predicted_morphology": self.compute_morphological_metrics(pred > 0.5),
            "ground_truth_morphology": self.compute_morphological_metrics(gt > 0.5),
        }
        
        pred_morph = result["predicted_morphology"]
        gt_morph = result["ground_truth_morphology"]
        result["morphology_error"] = {
            "grain_count_error": abs(pred_morph["n_grains"] - gt_morph["n_grains"]) / max(gt_morph["n_grains"], 1),
            "grain_size_error": abs(pred_morph["avg_grain_size"] - gt_morph["avg_grain_size"]) / max(gt_morph["avg_grain_size"], 1),
            "phase_fraction_error": abs(pred_morph["phase_fraction"] - gt_morph["phase_fraction"])
        }
        self.comparison_history.append(result)
        return result

    def render_comparison_dashboard(self):
        if not self.comparison_history:
            st.info("No comparisons yet. Upload predicted and ground-truth fields.")
            return
        
        latest = self.comparison_history[-1]
        st.subheader(f"📊 Field Comparison: {latest['field_name']}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RMSE", f"{latest['rmse']:.4f}")
        col2.metric("MAE", f"{latest['mae']:.4f}")
        col3.metric("SSIM", f"{latest['ssim']:.3f}")
        col4.metric("PSNR", f"{latest['psnr']:.2f} dB")
        
        st.markdown("### 🔬 Morphological Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Predicted**")
            pm = latest["predicted_morphology"]
            st.write(f"Grains: {pm['n_grains']}")
            st.write(f"Avg Size: {pm['avg_grain_size']:.1f} px²")
            st.write(f"Phase Fraction: {pm['phase_fraction']:.3f}")
        with col2:
            st.markdown("**Ground Truth**")
            gm = latest["ground_truth_morphology"]
            st.write(f"Grains: {gm['n_grains']}")
            st.write(f"Avg Size: {gm['avg_grain_size']:.1f} px²")
            st.write(f"Phase Fraction: {gm['phase_fraction']:.3f}")
            
        st.markdown("### ⚠️ Morphology Errors")
        me = latest["morphology_error"]
        st.progress(1.0 - me["grain_count_error"], text=f"Grain Count Accuracy: {(1-me['grain_count_error'])*100:.1f}%")
        st.progress(1.0 - me["grain_size_error"], text=f"Grain Size Accuracy: {(1-me['grain_size_error'])*100:.1f}%")
        st.progress(1.0 - me["phase_fraction_error"], text=f"Phase Fraction Accuracy: {(1-me['phase_fraction_error'])*100:.1f}%")

# =============================================
# BENCHMARK QUERY SUITE FOR DECLARMIMA
# =============================================
DECLARMIMA_BENCHMARK_QUERIES = [
    {
        "query": "What is the Gibbs free energy function for FCC phase in Fe-Cr system at 843K?",
        "category": "thermodynamics",
        "relevant_keywords": ["gibbs free energy", "fcc", "fe-cr", "thermodynamic", "calphad"],
        "expected_parameters": {"temperature_K": 843, "phase": "fcc"}
    },
    {
        "query": "Plot the phase diagram for AlSi10Mg alloy under laser powder bed fusion conditions",
        "category": "phase_diagram",
        "relevant_keywords": ["alsi10mg", "phase diagram", "lpbf", "solidification", "eutectic"],
        "expected_parameters": {"alloy": "alsi10mg", "process": "lpbf"}
    },
    {
        "query": "What laser power and scan speed prevent porosity in Ti6Al4V selective laser melting?",
        "category": "process_optimization",
        "relevant_keywords": ["ti6al4v", "laser power", "scan speed", "porosity", "slm"],
        "expected_parameters": {"defect": "porosity", "alloy": "ti6al4v"}
    },
    {
        "query": "Calculate the diffusion coefficient of Cu in Sn-Ag-Cu solder at 250°C",
        "category": "diffusion",
        "relevant_keywords": ["diffusion coefficient", "cu", "sn-ag-cu", "solder", "atomic mobility"],
        "expected_parameters": {"temperature_C": 250, "diffusing_species": "cu"}
    },
    {
        "query": "How does Marangoni convection affect melt pool geometry in laser processing?",
        "category": "mechanism",
        "relevant_keywords": ["marangoni", "convection", "melt pool", "fluid flow", "surface tension"],
        "expected_parameters": {"phenomenon": "marangoni convection"}
    },
    {
        "query": "What is the yield strength of CoCrFeNi high-entropy alloy after direct energy deposition?",
        "category": "mechanical_properties",
        "relevant_keywords": ["cocrfeni", "yield strength", "hea", "ded", "mechanical property"],
        "expected_parameters": {"property": "yield_strength", "alloy": "cocrfeni"}
    },
    {
        "query": "Compare columnar and equiaxed grain formation during laser solidification",
        "category": "microstructure",
        "relevant_keywords": ["columnar grain", "equiaxed grain", "solidification", "grain morphology", "cet"],
        "expected_parameters": {"feature": "grain_morphology"}
    }
]

def run_benchmark_evaluation(vectorstore, evaluator: RetrievalEvaluator, k: int = 5):
    results = []
    for bench in DECLARMIMA_BENCHMARK_QUERIES:
        query = bench["query"]
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        retrieved = retriever.invoke(query)
        
        relevant_ids = set()
        relevance_scores = {}
        for doc in retrieved:
            doc_id = f"{doc.metadata['source']}:{doc.metadata['chunk_index']}"
            score = sum(1 for kw in bench["relevant_keywords"] if kw in doc.page_content.lower())
            if score > 0:
                relevant_ids.add(doc_id)
                relevance_scores[doc_id] = score
        
        metrics = evaluator.evaluate_query(query, retrieved, relevant_ids, relevance_scores)
        results.append({
            "query": query,
            "category": bench["category"],
            "recall@5": metrics.recall_at_k.get(5, 0),
            "precision@5": metrics.precision_at_k.get(5, 0),
            "mrr": metrics.mrr,
            "context_relevance": metrics.context_relevance
        })
    return pd.DataFrame(results) if PANDAS_AVAILABLE else None

# =============================================
# STRUCTURED DATA LOADER FOR SIMULATION OUTPUTS
# =============================================
class StructuredDataLoader:
    """
    Load and chunk structured simulation data (CSV, TDB snippets, VTK metadata)
    for inclusion in the RAG knowledge base alongside PDFs.
    """
    def load_csv_dataset(self, file_path: str, description: str = "") -> List[Document]:
        if not PANDAS_AVAILABLE or not os.path.exists(file_path):
            return []
        try:
            df = pd.read_csv(file_path)
        except:
            return []
        
        documents = []
        global_desc = f"Dataset: {os.path.basename(file_path)}. {description}. "
        global_desc += f"Columns: {', '.join(df.columns)}. "
        global_desc += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns. "
        global_desc += f"Value ranges: "
        for col in df.select_dtypes(include=[np.number]).columns[:5]:
            global_desc += f"{col}=[{df[col].min():.3f}, {df[col].max():.3f}]; "
        
        documents.append(Document(
            page_content=global_desc,
            metadata={"source": file_path, "type": "csv_global", "chunk_index": 0}
        ))
        
        desc = df.describe()
        for col in df.select_dtypes(include=[np.number]).columns:
            chunk_text = f"Column '{col}' statistics: "
            chunk_text += f"mean={desc.loc['mean', col]:.4f}, "
            chunk_text += f"std={desc.loc['std', col]:.4f}, "
            chunk_text += f"min={desc.loc['min', col]:.4f}, "
            chunk_text += f"max={desc.loc['max', col]:.4f}, "
            chunk_text += f"median={df[col].median():.4f}. "
            chunk_text += f"Sample values: {', '.join([f'{v:.4f}' for v in df[col].dropna().head(5)])}"
            documents.append(Document(
                page_content=chunk_text,
                metadata={"source": file_path, "type": "csv_column", "column": col, "chunk_index": len(documents)}
            ))
        return documents

    def load_tdb_thermodynamic_database(self, file_path: str) -> List[Document]:
        if not os.path.exists(file_path):
            return []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return []
            
        documents = []
        phase_blocks = re.split(r'\n\s*PHASE\s+', content)
        for block in phase_blocks[1:]:
            lines = block.strip().split('\n')
            phase_name = lines[0].split()[0] if lines else "UNKNOWN"
            constituents = re.findall(r'CONSTITUENT\s+[^:]+:\s*([^!]+)', block)
            chunk_text = f"Phase: {phase_name}. "
            if constituents:
                chunk_text += f"Constituents: {constituents[0].strip()}. "
            gibbs_params = re.findall(r'PARAMETER\s+G\([^)]+\),\s*([^;]+)', block)
            if gibbs_params:
                chunk_text += f"Gibbs energy parameters: {len(gibbs_params)} defined. "
                chunk_text += f"First parameter: {gibbs_params[0][:200]}... "
            
            documents.append(Document(
                page_content=chunk_text,
                metadata={"source": file_path, "type": "tdb_phase", "phase": phase_name, "chunk_index": len(documents)}
            ))
        
        system_chunk = "Thermodynamic database overview: "
        elements = re.findall(r'ELEMENT\s+(\w+)', content)
        system_chunk += f"Elements: {', '.join(elements)}. "
        system_chunk += f"Phases defined: {len(phase_blocks)-1}. "
        documents.insert(0, Document(
            page_content=system_chunk,
            metadata={"source": file_path, "type": "tdb_system", "chunk_index": 0}
        ))
        return documents

# =============================================
# FUSION DATA STRUCTURES AND ENUMS
# =============================================
class FusionConfidence(Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNKNOWN = "unknown"

@dataclass
class ExtractedProperty:
    name: str
    value: Union[float, str, List]
    unit: Optional[str] = None
    uncertainty: Optional[str] = None
    condition: Optional[str] = None
    source_chunk_id: str = ""
    source_citation: str = ""
    extraction_confidence: float = 0.5
    context_snippet: str = ""
    property_type: str = "parameter"
    material_system: Optional[str] = None
    experimental_conditions: Dict[str, Any] = field(default_factory=dict)
    normalized_name: str = ""
    normalized_value: Optional[float] = None
    normalized_unit: Optional[str] = None

    def __post_init__(self):
        if not self.normalized_name:
            self.normalized_name = self._normalize_property_name(self.name)
        if self.normalized_value is None and isinstance(self.value, (int, float)):
            self.normalized_value = self.value

    def _normalize_property_name(self, name: str) -> str:
        synonym_map = {
            "ablation threshold": "ablation_threshold", "threshold fluence": "ablation_threshold", "fluence threshold": "ablation_threshold",
            "pulse duration": "pulse_duration", "pulse width": "pulse_duration", "pulse length": "pulse_duration",
            "wavelength": "wavelength", "laser wavelength": "wavelength",
            "repetition rate": "repetition_rate", "pulse frequency": "repetition_rate",
            "spot size": "spot_size", "beam diameter": "spot_size", "fluence": "fluence", "laser fluence": "fluence",
            "yield strength": "yield_strength", "ys": "yield_strength",
            "ultimate tensile strength": "ultimate_tensile_strength", "uts": "ultimate_tensile_strength", "tensile strength": "ultimate_tensile_strength",
            "elongation": "elongation_at_break", "elongation at break": "elongation_at_break",
            "hardness": "hardness", "microhardness": "hardness", "vickers hardness": "hardness", "hv": "hardness",
            "rockwell hardness": "hardness", "brinell hardness": "hardness",
            "young modulus": "elastic_modulus", "elastic modulus": "elastic_modulus",
            "shear modulus": "shear_modulus", "bulk modulus": "bulk_modulus",
            "fracture toughness": "fracture_toughness", "fatigue strength": "fatigue_strength", "creep resistance": "creep_resistance",
        }
        name_lower = name.lower().strip()
        return synonym_map.get(name_lower, name_lower.replace(" ", "_"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def format_for_display(self) -> str:
        value_str = f"{self.value}"
        if isinstance(self.value, (list, tuple)) and len(self.value) == 2:
            value_str = f"{self.value[0]}–{self.value[1]}"
        if self.uncertainty and self.uncertainty not in value_str:
            value_str = f"{value_str} {self.uncertainty}"
        if self.unit:
            value_str = f"{value_str} {self.unit}"
        if self.condition:
            return f"{self.normalized_name}: {value_str} ({self.condition})"
        return f"{self.normalized_name}: {value_str}"

@dataclass
class DocumentFusionRecord:
    source_filename: str
    chunk_index: int
    chunk_id: str
    bibliographic_citation: str
    extracted_properties: List[ExtractedProperty] = field(default_factory=list)
    laser_topics: List[str] = field(default_factory=list)
    experimental_conditions: Dict[str, Any] = field(default_factory=dict)
    material_system: Optional[str] = None
    processing_method: Optional[str] = None

    def add_property(self, prop: ExtractedProperty):
        self.extracted_properties.append(prop)

    def get_properties_by_name(self, prop_name: str) -> List[ExtractedProperty]:
        normalized = ExtractedProperty("", "")._normalize_property_name(prop_name)
        return [p for p in self.extracted_properties if p.normalized_name == normalized]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "citation": self.bibliographic_citation,
            "material": self.material_system,
            "method": self.processing_method,
            "properties": [p.to_dict() for p in self.extracted_properties],
            "topics": self.laser_topics,
            "conditions": self.experimental_conditions
        }

@dataclass
class FusedPropertyEntry:
    property_name: str
    fused_value: Optional[Union[float, str, Dict]] = None
    unit: Optional[str] = None
    fusion_confidence: FusionConfidence = FusionConfidence.UNKNOWN
    source_count: int = 0
    sources: List[Dict[str, str]] = field(default_factory=list)
    value_range: Optional[Tuple[float, float]] = None
    standard_deviation: Optional[float] = None
    conditions_summary: Dict[str, List[str]] = field(default_factory=dict)
    conflicts_detected: bool = False
    conflict_notes: List[str] = field(default_factory=list)
    fusion_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_comparison_row(self) -> Dict[str, Any]:
        return {
            "property": self.property_name,
            "value": self.fused_value,
            "unit": self.unit,
            "range": f"{self.value_range[0]:.2f}–{self.value_range[1]:.2f}" if self.value_range else None,
            "std": f"{self.standard_deviation:.3f}" if self.standard_deviation else None,
            "sources": len(self.sources),
            "confidence": self.fusion_confidence.value,
            "conditions": self.conditions_summary
        }

@dataclass
class FusionEfficiencyMetrics:
    unique_sources_used: int = 0
    source_diversity_score: float = 0.0
    total_properties_extracted: int = 0
    properties_fused_successfully: int = 0
    property_coverage_ratio: float = 0.0
    consistent_properties: int = 0
    conflicting_properties: int = 0
    consistency_ratio: float = 0.0
    numeric_properties_with_uncertainty: int = 0
    average_uncertainty_magnitude: float = 0.0
    high_confidence_fusions: int = 0
    low_confidence_fusions: int = 0
    weighted_confidence_score: float = 0.0
    answer_specificity_score: float = 0.0
    citation_density: float = 0.0
    overall_fusion_efficiency: float = 0.0

    def compute_overall(self) -> float:
        weights = {"source_diversity": 0.15, "property_coverage": 0.20, "consistency": 0.25, "precision": 0.15, "confidence": 0.15, "specificity": 0.10}
        if self.total_properties_extracted == 0:
            self.overall_fusion_efficiency = self.source_diversity_score * 0.3
            return self.overall_fusion_efficiency
        
        components = [
            self.source_diversity_score * weights["source_diversity"],
            self.property_coverage_ratio * weights["property_coverage"],
            self.consistency_ratio * weights["consistency"],
            (1 - min(self.average_uncertainty_magnitude, 1.0)) * weights["precision"],
            self.weighted_confidence_score * weights["confidence"],
            self.answer_specificity_score * weights["specificity"]
        ]
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.overall_fusion_efficiency = sum(components) / total_weight
        else:
            self.overall_fusion_efficiency = 0.0
        return self.overall_fusion_efficiency

    def to_display_dict(self) -> Dict[str, str]:
        return {
            "📚 Sources": f"{self.unique_sources_used} (div: {self.source_diversity_score:.2f})",
            "🔍 Properties": f"{self.properties_fused_successfully}/{max(self.total_properties_extracted, 1)}",
            "✅ Consistency": f"{self.consistency_ratio*100:.0f}%" if self.consistency_ratio > 0 else "N/A",
            "🎯 Precision": f"±{self.average_uncertainty_magnitude*100:.0f}%" if self.average_uncertainty_magnitude > 0 else "N/A",
            "💡 Confidence": f"{self.weighted_confidence_score:.2f}",
            "📝 Specificity": f"{self.answer_specificity_score:.2f}",
            "🏆 Overall": f"{self.overall_fusion_efficiency:.2f}/1.0"
        }

# =============================================
# BIBLIOGRAPHIC METADATA EXTRACTION FUNCTIONS
# =============================================
class BibliographicMetadata:
    DOI_PATTERN = re.compile(r'\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b', re.IGNORECASE)
    ARXIV_PATTERN = re.compile(r'\barXiv[:\s]+(\d{4}\.\d{4,5}(v\d+)?)\b', re.IGNORECASE)
    JOURNAL_PATTERNS = [
        re.compile(r'(?:published in|journal|proc\.?|journal of)\s+([A-Z][A-Za-z\s&\.]+?)(?:,|\.)', re.I),
        re.compile(r'([A-Z][A-Za-z\s&\.]+?\s+(?:Letters?|Journal|Transactions|Review|Proceedings))', re.I),
        re.compile(r'([A-Z][A-Za-z\s&\.]+?(?:Journal|Transactions|Review|Proceedings|Applications|Engineering|Science|Materials|Physics|Chemistry))', re.I),
    ]
    YEAR_PATTERN = re.compile(r'\b((?:19|20)\d{2})\b')
    VOLUME_PATTERN = re.compile(r'(?:vol\.?|volume)\s*(\d+)', re.I)
    ISSUE_PATTERN = re.compile(r'(?:no\.?|issue|iss\.?)\s*(\d+)', re.I)
    AUTHOR_PATTERN = re.compile(r'(?:^|by|authors?:\s*)([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*)', re.MULTILINE)

    def __init__(self, source_filename: str):
        self.source_filename = source_filename
        self.doi: Optional[str] = None
        self.arxiv_id: Optional[str] = None
        self.title: Optional[str] = None
        self.authors: List[str] = []
        self.journal: Optional[str] = None
        self.year: Optional[int] = None
        self.volume: Optional[str] = None
        self.issue: Optional[str] = None
        self.pages: Optional[str] = None
        self.publisher: Optional[str] = None
        self.raw_metadata: Dict[str, Any] = {}
        self.extraction_method: str = "none"
        self.confidence: float = 0.0

    def format_citation(self, style: str = "apa") -> str:
        if self.doi and self.confidence > 0.8:
            if style == "doi": return f"DOI:{self.doi}"
            elif style == "short": return f"[DOI:{self.doi}]"
        if self.arxiv_id:
            if style in ["doi", "short"]: return f"[arXiv:{self.arxiv_id}]"
        if self.authors and self.year:
            first_author = self._format_author_name(self.authors[0])
            et_al = " et al." if len(self.authors) > 1 else ""
            if style == "apa":
                journal_part = f", {self.journal}" if self.journal else ""
                return f"{first_author}{et_al}{journal_part}, {self.year}"
            elif style == "short": return f"[{first_author.split()[0]} {self.year}]"
            elif style == "full":
                parts = [f"{first_author}{et_al} ({self.year})"]
                if self.title: parts.append(f'"{self.title}"')
                if self.journal:
                    journal_str = self.journal
                    if self.volume: journal_str += f", {self.volume}"
                    if self.issue: journal_str += f"({self.issue})"
                    parts.append(journal_str)
                if self.pages: parts.append(f"pp. {self.pages}")
                return ". ".join(parts) + "."
        base_name = Path(self.source_filename).stem
        clean_name = re.sub(r'\s*(Elsevier|Ltd|All rights reserved|ScienceDirect|Contents lists available).*$', '', base_name, flags=re.I)
        if self.year: return f"[{clean_name}, {self.year}]"
        return f"[{clean_name}]"

    def _format_author_name(self, author_str: str) -> str:
        if "," in author_str:
            parts = [p.strip() for p in author_str.split(",", 1)]
            if len(parts) == 2:
                last, first = parts
                first_initial = first[0] + "." if first else ""
                return f"{last}, {first_initial}"
        return author_str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_filename, "doi": self.doi, "arxiv_id": self.arxiv_id, "title": self.title,
            "authors": self.authors, "journal": self.journal, "year": self.year, "volume": self.volume, "issue": self.issue,
            "pages": self.pages, "publisher": self.publisher, "extraction_method": self.extraction_method,
            "confidence": self.confidence, "citation_apa": self.format_citation("apa"),
            "citation_doi": self.format_citation("doi"), "citation_full": self.format_citation("full"),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BibliographicMetadata':
        meta = cls(data.get("source", "unknown"))
        meta.doi, meta.arxiv_id, meta.title = data.get("doi"), data.get("arxiv_id"), data.get("title")
        meta.authors, meta.journal, meta.year = data.get("authors", []), data.get("journal"), data.get("year")
        meta.volume, meta.issue, meta.pages = data.get("volume"), data.get("issue"), data.get("pages")
        meta.publisher, meta.extraction_method, meta.confidence = data.get("publisher"), data.get("extraction_method", "cached"), data.get("confidence", 0.5)
        return meta

def extract_metadata_from_pdf_text(text: str, filename: str) -> BibliographicMetadata:
    meta = BibliographicMetadata(filename)
    text_sample = text[:10000]
    text_lower = text_sample.lower()
    doi_match = BibliographicMetadata.DOI_PATTERN.search(text_sample)
    if doi_match:
        meta.doi = doi_match.group(1).lower()
        meta.confidence = max(meta.confidence, 0.9)
        meta.extraction_method = "regex_doi"
        
    arxiv_match = BibliographicMetadata.ARXIV_PATTERN.search(text_sample)
    if arxiv_match:
        meta.arxiv_id = arxiv_match.group(1)
        meta.confidence = max(meta.confidence, 0.85)
        meta.extraction_method = "regex_arxiv"
        
    year_matches = BibliographicMetadata.YEAR_PATTERN.findall(text_sample)
    for year_str in year_matches:
        year = int(year_str)
        if 1900 <= year <= 2030:
            year_pos = text_sample.find(year_str)
            context = text_sample[max(0, year_pos-50):year_pos+50].lower()
            if any(kw in context for kw in ['published', 'received', 'accepted', 'copyright', '©', 'submitted']):
                meta.year = year
                meta.confidence = max(meta.confidence, 0.7)
                break
                
    for pattern in BibliographicMetadata.JOURNAL_PATTERNS:
        journal_match = pattern.search(text_sample)
        if journal_match:
            journal = journal_match.group(1).strip()
            if len(journal) > 10 and not any(bad in journal.lower() for bad in ['introduction', 'abstract', 'references', 'elsevier', 'all rights', 'contents lists']):
                meta.journal = journal
                meta.confidence = max(meta.confidence, 0.6)
                break
                
    vol_match = BibliographicMetadata.VOLUME_PATTERN.search(text_sample)
    if vol_match: meta.volume = vol_match.group(1)
    iss_match = BibliographicMetadata.ISSUE_PATTERN.search(text_sample)
    if iss_match: meta.issue = iss_match.group(1)
    
    author_section = text_sample[:2000]
    author_matches = BibliographicMetadata.AUTHOR_PATTERN.findall(author_section)
    if author_matches:
        raw_authors = author_matches[0]
        if ',' in raw_authors or ' and ' in raw_authors.lower():
            for sep in [',', ' and ', ';']:
                if sep.lower() in raw_authors.lower():
                    meta.authors = [a.strip() for a in re.split(sep, raw_authors, flags=re.I) if a.strip()]
                    break
        else: meta.authors = [raw_authors.strip()]
        if meta.authors: meta.confidence = max(meta.confidence, 0.5)
        
    title_patterns = [
        re.compile(r'(?:^|\n)([A-Z][^.\n]{20,150}(?:\.[^A-Z]|$))'),
        re.compile(r'(?:title:?\s*)([A-Z][^.\n]{20,200}?)\.?(?:\n|$)', re.I),
    ]
    for pattern in title_patterns:
        title_match = pattern.search(text_sample)
        if title_match:
            title = title_match.group(1).strip()
            if 30 < len(title) < 200 and not title.isupper():
                meta.title = title
                meta.confidence = max(meta.confidence, 0.55)
                break
    return meta

def extract_metadata_from_pdf_file(pdf_path: str, filename: str) -> BibliographicMetadata:
    meta = BibliographicMetadata(filename)
    if PYPDF2_AVAILABLE:
        try:
            reader = PdfReader(pdf_path)
            pdf_info = reader.metadata or {}
            for pdf_field, meta_field in {'/Title': 'title', '/Author': 'authors', '/CreationDate': 'year', '/Subject': 'journal'}.items():
                if pdf_field in pdf_info and pdf_info[pdf_field]:
                    value = str(pdf_info[pdf_field]).strip()
                    if meta_field == 'authors' and value: meta.authors = [a.strip() for a in re.split(r'[;,]', value) if a.strip()]
                    elif meta_field == 'year' and value:
                        year_match = re.search(r'(?:D:)?(\d{4})', value)
                        if year_match: meta.year = int(year_match.group(1))
                    else: setattr(meta, meta_field, value)
            if meta.title or meta.authors: meta.confidence = 0.7; meta.extraction_method = "pdf_metadata"
        except Exception as e: st.warning(f"Could not read PDF metadata: {e}")
        
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        text_sample = "\n".join([p.page_content for p in pages[:3]])
        text_meta = extract_metadata_from_pdf_text(text_sample, filename)
        for field in ['doi', 'arxiv_id', 'title', 'journal', 'year', 'volume', 'issue']:
            if getattr(text_meta, field) and (not getattr(meta, field) or text_meta.confidence > meta.confidence):
                setattr(meta, field, getattr(text_meta, field))
        if text_meta.authors and (not meta.authors or text_meta.confidence > meta.confidence): meta.authors = text_meta.authors
        if text_meta.confidence > meta.confidence: meta.confidence = text_meta.confidence; meta.extraction_method = text_meta.extraction_method
    except Exception as e: st.warning(f"Text extraction for metadata failed: {e}")
    
    if PDF2DOI_AVAILABLE and not meta.doi:
        try:
            result = pdf2doi.pdf2doi(pdf_path)
            if isinstance(result, list) and result: result = result[0]
            if result and result.get('identifier') and result.get('identifier_type') == 'doi':
                meta.doi = result['identifier']; meta.confidence = 0.95; meta.extraction_method = "pdf2doi"
                if result.get('validation_info'):
                    bibtex = result['validation_info']
                    if 'title' in bibtex and not meta.title: meta.title = bibtex.get('title')
                    if 'author' in bibtex and not meta.authors: meta.authors = [a.strip() for a in bibtex['author'].split(' and ')]
                    if 'year' in bibtex and not meta.year:
                        try: meta.year = int(bibtex['year'])
                        except: pass
        except Exception as e: st.warning(f"pdf2doi lookup failed: {e}")
        
    if CROSSREF_AVAILABLE and meta.doi and not meta.journal:
        try:
            cr = CrossrefAPI()
            work = cr.works(ids=meta.doi)
            if work and work.get('message'):
                msg = work['message']
                if not meta.title and msg.get('title'): meta.title = msg['title'][0] if isinstance(msg['title'], list) else msg['title']
                if not meta.authors and msg.get('author'): meta.authors = [f"{a.get('family', '')} {a.get('given', '')}".strip() for a in msg['author']]
                if not meta.journal and msg.get('container-title'): meta.journal = msg['container-title'][0] if isinstance(msg['container-title'], list) else msg['container-title']
                if not meta.year and msg.get('published-print') and msg['published-print'].get('date-parts'): meta.year = msg['published-print']['date-parts'][0][0]
                meta.confidence = 0.98; meta.extraction_method = "crossref_api"
        except Exception as e: st.warning(f"Crossref API lookup failed: {e}")
    return meta

def extract_metadata_from_text_file(text: str, filename: str) -> BibliographicMetadata:
    return extract_metadata_from_pdf_text(text, filename)

# =============================================
# GLOBAL METADATA CACHE
# =============================================
class MetadataCache:
    def __init__(self): self._cache: Dict[str, BibliographicMetadata] = {}; self._file_hashes: Dict[str, str] = {}
    def get(self, filename: str, file_hash: str = None) -> Optional[BibliographicMetadata]:
        if filename in self._cache:
            if file_hash is None or self._file_hashes.get(filename) == file_hash: return self._cache[filename]
        return None
    def set(self, filename: str, metadata: BibliographicMetadata, file_hash: str = None):
        self._cache[filename] = metadata
        if file_hash: self._file_hashes[filename] = file_hash
    def clear(self): self._cache.clear(); self._file_hashes.clear()

metadata_cache = MetadataCache()

def compute_file_hash(filepath: str) -> str:
    try:
        with open(filepath, 'rb') as f: return hashlib.md5(f.read()).hexdigest()
    except: return ""

# =============================================
# MULTI-DOCUMENT PROPERTY EXTRACTION ENGINE
# =============================================
class MultiDocumentPropertyExtractor:
    UNIT_CONVERSIONS = {
        "nm": {"factor": 1e-9, "base": "m"}, "μm": {"factor": 1e-6, "base": "m"}, "um": {"factor": 1e-6, "base": "m"}, "mm": {"factor": 1e-3, "base": "m"},
        "fs": {"factor": 1e-15, "base": "s"}, "ps": {"factor": 1e-12, "base": "s"}, "ns": {"factor": 1e-9, "base": "s"},
        "J/cm²": {"factor": 1e4, "base": "J/m²"}, "J/cm2": {"factor": 1e4, "base": "J/m²"}, "mJ/cm²": {"factor": 10, "base": "J/m²"},
        "MPa": {"factor": 1e6, "base": "Pa"}, "GPa": {"factor": 1e9, "base": "Pa"}, "kN": {"factor": 1e3, "base": "N"},
        "HV": {"factor": 1, "base": "HV"}, "HRC": {"factor": 1, "base": "HRC"}, "HB": {"factor": 1, "base": "HB"},
    }
    MATERIAL_SYNONYMS = {
        "si": "silicon", "si substrate": "silicon", "crystalline silicon": "silicon", "c-si": "silicon",
        "aluminum alloy": "aluminum", "alsi10mg": "AlSi10Mg", "al-si10-mg": "AlSi10Mg",
        "ti-6al-4v": "Ti6Al4V", "titanium alloy": "Ti6Al4V", "stainless steel": "steel", "ss316l": "steel",
    }

    def __init__(self, laser_keywords: Dict[str, List[str]]):
        self.laser_keywords = laser_keywords
        self._compile_extraction_patterns()

    def _compile_extraction_patterns(self):
        numeric_pattern = r'([\d.]+(?:\s*[×x*]\s*10\^?-?\d+)?)(?:\s*([±\+-])\s*([\d.]+))?'
        unit_pattern = r'\s*(' + '|'.join(re.escape(u) for u in self.UNIT_CONVERSIONS.keys()) + r')'
        self.property_pattern = re.compile(r'([\w\s\-_/]+?)\s*(?:is|was|of|at|:|=|≈|~|yields|results in|produces)\s*' + numeric_pattern + unit_pattern + r'(?:\s*[\(\[]([^)\]]+)[\)\]])?', re.I)
        material_list = list(self.MATERIAL_SYNONYMS.keys()) + ['silicon', 'steel', 'titanium', 'polymer', 'glass', 'ceramic', 'aluminum', 'composite', 'alloy']
        self.material_property_pattern = re.compile(r'(' + '|'.join(re.escape(m) for m in material_list) + r').{0,200}?' + r'([\w\s]+?\s*(?:is|was|of|at|:|=)\s*[\d.]+)', re.I | re.DOTALL)

    def extract_properties_from_chunk(self, chunk_text: str, chunk_metadata: Dict[str, Any]) -> DocumentFusionRecord:
        record = DocumentFusionRecord(
            source_filename=chunk_metadata.get('source', 'unknown'), chunk_index=chunk_metadata.get('chunk_index', 0),
            chunk_id=f"{chunk_metadata.get('source', 'unknown')}:{chunk_metadata.get('chunk_index', 0)}",
            bibliographic_citation=chunk_metadata.get('citation_display', 'Unknown'), laser_topics=chunk_metadata.get('laser_topics', []),
            experimental_conditions=chunk_metadata.get('parameters_found', {}), material_system=self._detect_material_system(chunk_text),
            processing_method=self._detect_processing_method(chunk_text)
        )
        table_properties = self._extract_from_tables(chunk_text)
        for prop in table_properties: prop.source_chunk_id = record.chunk_id; prop.source_citation = record.bibliographic_citation; prop.material_system = record.material_system; record.add_property(prop)
        
        inline_properties = self._extract_inline_properties(chunk_text)
        for prop in inline_properties:
            if not any(p.normalized_name == prop.normalized_name and (abs(p.normalized_value - prop.normalized_value) < 1e-6 if p.normalized_value and prop.normalized_value else False) for p in record.extracted_properties):
                prop.source_chunk_id = record.chunk_id; prop.source_citation = record.bibliographic_citation
                record.add_property(prop)
                
        return record

    def _detect_material_system(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        for canonical, synonyms in self.MATERIAL_SYNONYMS.items():
            if isinstance(synonyms, list):
                if any(s.lower() in text_lower for s in synonyms): return canonical
            elif synonyms.lower() in text_lower: return canonical
        return None

    def _detect_processing_method(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        for pattern, canonical in [("femtosecond laser", "femtosecond_ablation"), ("picosecond laser", "picosecond_ablation"), ("nanosecond laser", "nanosecond_ablation"), ("ultrafast laser", "ultrafast_processing"), ("laser ablation", "laser_ablation"), ("aging", "aging_treatment"), ("annealing", "annealing"), ("heat treatment", "heat_treatment"), ("surface texturing", "surface_texturing"), ("lipss", "lipss_formation")]:
            if pattern in text_lower: return canonical
        return None

    def _extract_from_tables(self, text: str) -> List[ExtractedProperty]: return []

    def _extract_inline_properties(self, text: str) -> List[ExtractedProperty]:
        properties = []
        for match in self.property_pattern.finditer(text):
            groups = match.groups()
            if len(groups) >= 5 and groups[1]:
                prop_name, value_str, uncertainty, unit = groups[0].strip(), groups[1].strip(), f"{groups[2]}{groups[3]}" if groups[2] and groups[3] else None, groups[4].strip() if groups[4] else None
                numeric_value = self._safe_parse_numeric(value_str)
                prop = ExtractedProperty(name=prop_name, value=numeric_value if numeric_value is not None else value_str, unit=unit, uncertainty=uncertainty, extraction_confidence=0.7, context_snippet=match.group(0)[:150], property_type="parameter" if any(kw in prop_name.lower() for kw in ['fluence', 'duration', 'wavelength', 'threshold']) else "measurement")
                self._normalize_property_units(prop)
                properties.append(prop)
        return properties

    def _parse_property_value(self, raw_value: str, prop_name: str) -> Optional[Dict[str, Any]]: return None

    def _safe_parse_numeric(self, value_str: str) -> Optional[float]:
        if not value_str or value_str.strip() in ['.', '-', '--', '...', 'N/A', 'n/a', 'NA', 'na', 'null', 'None', '']: return None
        cleaned = value_str.strip()
        cleaned = re.sub(r'\s*[×x*]\s*10\^?', 'e', cleaned)
        cleaned = re.sub(r'\s*[×x*]\s*10', 'e', cleaned)
        match = re.match(r'^\s*([+-]?\s*[\d.]+(?:e[+-]?\d+)?)', cleaned, re.I)
        if not match: return None
        num_str = match.group(1).replace(' ', '')
        try: return float(num_str)
        except (ValueError, TypeError, OverflowError): return None

    def _normalize_property_units(self, prop: ExtractedProperty):
        if not prop.unit or prop.unit not in self.UNIT_CONVERSIONS:
            prop.normalized_unit = prop.unit
            if isinstance(prop.value, (int, float)): prop.normalized_value = prop.value
            elif prop.normalized_value is None and isinstance(prop.value, str): prop.normalized_value = self._safe_parse_numeric(prop.value)
            return
        conversion = self.UNIT_CONVERSIONS[prop.unit]
        if isinstance(prop.value, (int, float)): prop.normalized_value = prop.value * conversion["factor"]; prop.normalized_unit = conversion["base"]
        elif prop.normalized_value is not None: prop.normalized_value = prop.normalized_value * conversion["factor"]; prop.normalized_unit = conversion["base"]
        else: prop.normalized_unit = prop.unit

# =============================================
# INFORMATION FUSION ENGINE
# =============================================
class MultiDocumentFusionEngine:
    def __init__(self, property_extractor: MultiDocumentPropertyExtractor):
        self.extractor = property_extractor
        self.fusion_history: List[Dict] = []

    def fuse_documents(self, retrieved_docs: List[Document], query: str,
                       material_filter: Optional[str] = None,
                       property_filter: Optional[List[str]] = None) -> Tuple[Dict[str, FusedPropertyEntry], FusionEfficiencyMetrics]:
        fusion_records: List[DocumentFusionRecord] = []
        for doc in retrieved_docs:
            record = self.extractor.extract_properties_from_chunk(doc.page_content, doc.metadata)
            if material_filter and record.material_system != material_filter: continue
            if property_filter: record.extracted_properties = [p for p in record.extracted_properties if p.normalized_name in property_filter]
            if record.extracted_properties: fusion_records.append(record)
            
        if not fusion_records:
            metrics = FusionEfficiencyMetrics(unique_sources_used=len(retrieved_docs), source_diversity_score=min(1.0, len(retrieved_docs) / 3.0), overall_fusion_efficiency=min(1.0, len(retrieved_docs) / 3.0) * 0.3)
            return {}, metrics
            
        property_groups: Dict[str, List[ExtractedProperty]] = defaultdict(list)
        for record in fusion_records:
            for prop in record.extracted_properties:
                key = prop.normalized_name
                if not property_filter or key in property_filter: property_groups[key].append(prop)
                
        fused_properties: Dict[str, FusedPropertyEntry] = {}
        for prop_name, props in property_groups.items():
            fused = self._fuse_property_group(prop_name, props)
            if fused: fused_properties[prop_name] = fused
            
        metrics = self._compute_fusion_metrics(fusion_records, fused_properties, retrieved_docs, query)
        self.fusion_history.append({"timestamp": datetime.now().isoformat(), "query": query, "input_docs": len(retrieved_docs), "extracted_properties": sum(len(r.extracted_properties) for r in fusion_records), "fused_properties": len(fused_properties), "efficiency": metrics.overall_fusion_efficiency})
        return fused_properties, metrics

    def _fuse_property_group(self, prop_name: str, properties: List[ExtractedProperty]) -> Optional[FusedPropertyEntry]:
        if not properties: return None
        numeric_props = [p for p in properties if p.normalized_value is not None and isinstance(p.normalized_value, (int, float))]
        fused = FusedPropertyEntry(property_name=prop_name, unit=properties[0].normalized_unit if properties[0].normalized_unit else properties[0].unit, source_count=len(properties), sources=[{"citation": p.source_citation, "chunk_id": p.source_chunk_id} for p in properties])
        
        if numeric_props and len(numeric_props) >= 1:
            values = [p.normalized_value for p in numeric_props if p.normalized_value is not None]
            if values:
                fused.fused_value = np.mean(values)
                fused.value_range = (min(values), max(values))
                fused.standard_deviation = np.std(values) if len(values) > 1 else 0.0
                cv = fused.standard_deviation / abs(fused.fused_value) if fused.fused_value != 0 else 1.0
                if cv < 0.1 and len(numeric_props) >= 2: fused.fusion_confidence = FusionConfidence.HIGH
                elif cv < 0.3 or len(numeric_props) == 1: fused.fusion_confidence = FusionConfidence.MODERATE
                else: fused.fusion_confidence = FusionConfidence.LOW; fused.conflicts_detected = True; fused.conflict_notes.append(f"High variation: CV={cv:.2f}")
                conditions = defaultdict(set)
                for p in numeric_props:
                    if p.condition: conditions["context"].add(p.condition)
                    if p.experimental_conditions:
                        for k, v in p.experimental_conditions.items(): conditions[k].add(str(v))
                fused.conditions_summary = {k: list(v) for k, v in conditions.items()}
        else:
            value_counts = Counter(str(p.value) for p in properties if p.value is not None)
            if value_counts:
                fused.fused_value = value_counts.most_common(1)[0][0]
                fused.fusion_confidence = FusionConfidence.HIGH if value_counts.most_common(1)[0][1] == len(properties) else FusionConfidence.MODERATE if value_counts.most_common(1)[0][1] > len(properties) / 2 else FusionConfidence.LOW
                if fused.fusion_confidence == FusionConfidence.LOW: fused.conflicts_detected = True; fused.conflict_notes.append(f"Multiple distinct values: {list(value_counts.keys())[:3]}")
        return fused

    def _compute_fusion_metrics(self, fusion_records, fused_properties, retrieved_docs, query) -> FusionEfficiencyMetrics:
        metrics = FusionEfficiencyMetrics()
        unique_sources = set(r.chunk_id for r in fusion_records)
        metrics.unique_sources_used = len(unique_sources); metrics.source_diversity_score = min(1.0, len(unique_sources) / 3.0)
        total_extracted = sum(len(r.extracted_properties) for r in fusion_records); metrics.total_properties_extracted = total_extracted
        metrics.properties_fused_successfully = len(fused_properties); metrics.property_coverage_ratio = len(fused_properties) / total_extracted if total_extracted > 0 else 0.0
        if fused_properties:
            consistent = sum(1 for f in fused_properties.values() if not f.conflicts_detected and f.fusion_confidence != FusionConfidence.LOW)
            conflicting = sum(1 for f in fused_properties.values() if f.conflicts_detected)
            total_evaluated = consistent + conflicting; metrics.consistent_properties = consistent; metrics.conflicting_properties = conflicting
            metrics.consistency_ratio = consistent / total_evaluated if total_evaluated > 0 else 1.0
        else: metrics.consistency_ratio = 1.0
        uncertainties = []
        for f in fused_properties.values():
            if isinstance(f.fused_value, (int, float)) and f.fused_value != 0 and f.standard_deviation is not None: uncertainties.append(f.standard_deviation / abs(f.fused_value))
        metrics.average_uncertainty_magnitude = np.mean(uncertainties) if uncertainties else 0.1
        
        confidence_weights = {FusionConfidence.HIGH: 1.0, FusionConfidence.MODERATE: 0.7, FusionConfidence.LOW: 0.4, FusionConfidence.UNKNOWN: 0.2}
        if fused_properties:
            weighted_sum = sum(confidence_weights.get(f.fusion_confidence, 0.5) for f in fused_properties.values())
            metrics.weighted_confidence_score = weighted_sum / len(fused_properties)
        else: metrics.weighted_confidence_score = 0.5
        
        metrics.answer_specificity_score = self._estimate_answer_specificity(query, fused_properties)
        metrics.citation_density = min(1.0, len(fused_properties) * 2 / 100)
        metrics.compute_overall()
        return metrics

    def _estimate_answer_specificity(self, query: str, fused_props: Dict[str, FusedPropertyEntry]) -> float:
        if not fused_props: return 0.5 if any(kw in query.lower() for kw in ['compare', 'versus', 'vs', 'difference', 'threshold', 'strength', 'hardness']) else 0.3
        query_lower = query.lower(); specificity_indicators = 0
        for prop_name in fused_props.keys():
            if prop_name.replace('_', ' ') in query_lower or prop_name in query_lower: specificity_indicators += 2
        if any(mat in query_lower for mat in ['silicon', 'aluminum', 'titanium', 'steel', 'composite', 'alloy']): specificity_indicators += 1
        if any(param in query_lower for param in ['fluence', 'threshold', 'duration', 'wavelength', 'strength', 'hardness']): specificity_indicators += 1
        if re.search(r'[\d.]+\s*(?:j/cm|mpa|fs|nm|%|percent|hv|hrc)', query_lower): specificity_indicators += 2
        return min(1.0, specificity_indicators / 5.0)

    def generate_comparison_table(self, fused_properties: Dict[str, FusedPropertyEntry], format: str = "markdown") -> str:
        if not fused_properties: return "_No properties available for comparison_"
        lines = ["| Property | Value | Unit | Range | Sources | Confidence |", "|----------|-------|------|-------|---------|------------|"]
        for prop_name, entry in sorted(fused_properties.items(), key=lambda x: x[0]):
            if entry.fused_value is not None and isinstance(entry.fused_value, (int, float)):
                value_str = f"{entry.fused_value:.3f}"
                if entry.standard_deviation is not None: value_str = f"{value_str} ± {entry.standard_deviation:.3f}"
            else: value_str = str(entry.fused_value) if entry.fused_value is not None else "–"
            range_str = f"{entry.value_range[0]:.2f}–{entry.value_range[1]:.2f}" if entry.value_range else "–"
            confidence_icon = {"high": "🟢", "moderate": "🟡", "low": "🔴", "unknown": "⚪"}.get(entry.fusion_confidence.value, "⚪")
            lines.append(f"| {prop_name.replace('_', ' ').title()} | {value_str} | {entry.unit or '–'} | {range_str} | {entry.source_count} | {confidence_icon} {entry.fusion_confidence.value} |")
        return "\n".join(lines)

# =============================================
# SESSION STATE INITIALIZATION
# =============================================
def initialize_session_state():
    defaults = {
        "processed_files": set(), "vectorstore": None, "all_chunks": [], "messages": [],
        "llm_model_choice": None, "llm_tokenizer": None, "llm_model": None, "llm_backend": None,
        "embeddings": None, "processing_complete": False, "laser_domain_boost": True,
        "show_sources": True, "citation_style": "apa", "max_retrieved_chunks": 4,
        "use_4bit_quantization": True, "ollama_host": "http://localhost:11434",
        "metadata_cache": metadata_cache, "enable_multi_doc_fusion": True,
        "fusion_property_filter": None, "fusion_material_filter": None,
        "debug_extraction": False, "evaluation_mode": False,
        "entity_map": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

# =============================================
# UTILITY FUNCTIONS
# =============================================
def is_ollama_model(model_key: str) -> bool: return model_key.startswith("ollama:") or model_key.startswith("[Ollama]")
def extract_ollama_tag(model_key: str) -> str:
    if model_key.startswith("ollama:"): return model_key.replace("ollama:", "", 1)
    elif model_key.startswith("[Ollama]"):
        match = re.search(r'\]\s*([^\s(]+)', model_key)
        if match: return match.group(1)
    return model_key
def get_hf_repo_id(model_key: str) -> str:
    if ":" in model_key and not model_key.startswith("http"):
        parts = model_key.split(":", 1)
        if len(parts) == 2 and "/" in parts[1]: return parts[1].strip()
    return model_key
def get_available_gpu_memory() -> Optional[float]:
    if not torch.cuda.is_available(): return None
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        return total_memory - reserved
    except: return None
def estimate_model_memory(model_key: str, use_4bit: bool = False) -> Dict[str, any]:
    repo_id = get_hf_repo_id(model_key) if not is_ollama_model(model_key) else model_key
    return MODEL_MEMORY_ESTIMATES.get(repo_id, {"params": "Unknown", "vram_fp16": "Unknown", "vram_4bit": "Unknown", "cpu_ok": False})

# =============================================
# LOCAL MODEL LOADING
# =============================================
@st.cache_resource(show_spinner="Loading local embedding model (~80MB)...")
def load_local_embeddings():
    try:
        return HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL, model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    except Exception as e: st.error(f"Failed to load embeddings: {e}"); return None

@st.cache_resource(show_spinner="Loading local LLM (this may take 1-2 minutes on first load)...")
def load_local_llm(model_key: str, use_4bit: bool = True):
    try:
        if is_ollama_model(model_key): return _load_ollama_model(model_key)
        else: return _load_transformers_model(model_key, use_4bit)
    except Exception as e:
        st.error(f"Failed to load LLM '{model_key}': {e}")
        st.warning("Falling back to GPT-2...")
        try:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2"); model = GPT2LMHeadModel.from_pretrained("gpt2")
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            model.eval(); device = "cuda" if torch.cuda.is_available() else "cpu"
            return tokenizer, model, device, "transformers"
        except Exception as e2: st.error(f"Fallback also failed: {e2}"); return None, None, None, None

def _load_ollama_model(model_key: str):
    if not OLLAMA_AVAILABLE: raise ImportError("ollama library not installed. Run: pip install ollama")
    model_tag = extract_ollama_tag(model_key)
    try:
        client = ollama.Client(host=st.session_state.ollama_host)
        response = client.list(); models_list = response.get('models', []) if isinstance(response, dict) else getattr(response, 'models', [])
        model_names = []
        for m in models_list:
            name = m.get('model') if isinstance(m, dict) else getattr(m, 'model', None)
            if name: model_names.append(name)
        if model_tag not in model_names: st.warning(f"⚠️ Model '{model_tag}' not found in Ollama.")
        return None, model_tag, st.session_state.ollama_host, "ollama"
    except Exception as conn_err: st.error(f"❌ Connection Error: {conn_err}"); return None, None, st.session_state.ollama_host, "ollama"

def _load_transformers_model(model_key: str, use_4bit: bool = True):
    repo_id = get_hf_repo_id(model_key); device = "cuda" if torch.cuda.is_available() else "cpu"
    available_vram = get_available_gpu_memory(); mem_info = estimate_model_memory(model_key, use_4bit)
    st.sidebar.info(f"""📊 Model Memory Estimate:
- Parameters: {mem_info['params']}
- VRAM (FP16): {mem_info['vram_fp16']}
- VRAM (4-bit): {mem_info['vram_4bit']}
- CPU OK: {'✅ Yes' if mem_info['cpu_ok'] else '❌ No'}
- Available VRAM: {f'{available_vram:.1f}GB' if available_vram else 'N/A (CPU)'}""")
    
    if "0.5B" in repo_id or "1.1B" in repo_id or "gpt2" in repo_id: use_4bit = False; quantization_config = None
    elif use_4bit and device == "cuda" and available_vram:
        try: quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"); st.sidebar.success("✅ 4-bit quantization enabled")
        except ImportError: st.sidebar.warning("⚠️ bitsandbytes not installed. Install with: pip install bitsandbytes"); use_4bit = False; quantization_config = None
    else: quantization_config = None
    
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True, padding_side="left", use_fast=True)
    model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.float16 if device == "cuda" else torch.float32}
    if quantization_config: model_kwargs["quantization_config"] = quantization_config; model_kwargs["device_map"] = "auto"
    elif device == "cuda": model_kwargs["device_map"] = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(repo_id, **model_kwargs)
    if "device_map" not in model_kwargs and device == "cpu": model = model.to(device)
    model.eval()
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model, device, "transformers"

# =============================================
# DOCUMENT LOADING & CHUNKING WITH FUSION SUPPORT
# =============================================
def extract_laser_metadata(text: str, filename: str) -> Dict[str, any]:
    metadata = {"source": filename, "laser_topics": [], "parameters_found": {},
                "has_equations": bool(re.search(r'[\(=]\s*[\d.]+\s*[×*]\s*10\^', text)),
                "has_figures": bool(re.search(r'Figure\s*\d+|Fig\.\s*\d+', text, re.I))}
    text_lower = text.lower()
    for topic, keywords in LASER_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords): metadata["laser_topics"].append(topic)
    param_patterns = {
        "wavelength_nm": r'(\d+(?:\.\d+)?)\s*(?:nm|nanometers?)\s*(?:wavelength|λ|lambda)',
        "pulse_duration_fs": r'(\d+(?:\.\d+)?)\s*(?:fs|femtoseconds?)\s*(?:pulse|duration)',
        "fluence_Jcm2": r'(\d+(?:\.\d+)?)\s*(?:J/cm²|J/cm2|fluence)',
        "repetition_rate": r'(\d+(?:\.\d+)?)\s*(?:kHz|MHz|Hz)\s*(?:repetition|rate|freq)',
        "spot_size_um": r'(\d+(?:\.\d+)?)\s*(?:µm|um|microns?)\s*(?:spot|diameter)',
    }
    for param, pattern in param_patterns.items():
        match = re.search(pattern, text, re.I)
        if match:
            try: metadata["parameters_found"][param] = float(match.group(1))
            except: pass
    return metadata

def load_and_chunk_laser_documents(uploaded_files: List) -> List[Document]:
    all_chunks = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf" if uploaded_file.name.endswith('.pdf') else ".txt") as tmp:
            tmp.write(uploaded_file.getbuffer()); tmp_path = tmp.name
            try:
                file_hash = compute_file_hash(tmp_path)
                cached_meta = st.session_state.metadata_cache.get(uploaded_file.name, file_hash)
                if cached_meta: bib_meta = cached_meta; st.info(f"📚 Using cached metadata for `{uploaded_file.name}`")
                else:
                    if uploaded_file.name.endswith('.pdf'): bib_meta = extract_metadata_from_pdf_file(tmp_path, uploaded_file.name)
                    else:
                        with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f: text_content = f.read()
                        bib_meta = extract_metadata_from_text_file(text_content, uploaded_file.name)
                    st.session_state.metadata_cache.set(uploaded_file.name, bib_meta, file_hash)
                    st.info(f"📚 Extracted metadata: {bib_meta.format_citation('apa')}")
                
                if uploaded_file.name.endswith('.pdf'): loader = PyPDFLoader(tmp_path)
                else: loader = TextLoader(tmp_path, encoding='utf-8')
                pages = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=LASER_DOMAIN_CONFIG["chunk_size"], chunk_overlap=LASER_DOMAIN_CONFIG["chunk_overlap"], separators=["\n", "Equation", "Parameter:", "Figure", "Table", ""], length_function=len)
                chunks = text_splitter.split_documents(pages)
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({"source": uploaded_file.name, "chunk_index": i, "total_chunks": len(chunks), **extract_laser_metadata(chunk.page_content, uploaded_file.name), "bibliographic": bib_meta.to_dict(), "citation_display": bib_meta.format_citation(st.session_state.get('citation_style', 'apa'))})
                    all_chunks.extend(chunks)
                st.info(f"✅ Loaded {len(chunks)} chunks from `{uploaded_file.name}`")
            except Exception as e: st.error(f"❌ Error processing `{uploaded_file.name}`: {e}")
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)
    return all_chunks

# =============================================
# VECTOR STORE CREATION
# =============================================
@st.cache_resource
def create_local_vector_store(chunks: List[Document], embedding_model_key: str):
    try:
        embeddings = load_local_embeddings()
        if embeddings is None: return None
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.metadata = {"total_chunks": len(chunks), "embedding_model": embedding_model_key, "created_at": datetime.now().isoformat(), "laser_topics": list(set(topic for chunk in chunks for topic in chunk.metadata.get("laser_topics", [])))}
        return vectorstore
    except Exception as e: st.error(f"Failed to create vector store: {e}"); return None

# =============================================
# ENHANCED RAG CHAIN WITH MULTI-DOCUMENT FUSION
# =============================================
def create_laser_rag_prompt(retrieved_chunks: List[Document], query: str) -> str:
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        citation = chunk.metadata.get("citation_display")
        if not citation: citation = f"[Source {i} - {chunk.metadata.get('source', 'unknown')}]"
        content = chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content
        context_parts.append(f"{citation}\n{content}\n")
    context = "\n---\n".join(context_parts)
    laser_system = """You are an expert assistant for laser-microstructure interaction research.
Your role is to answer questions based ONLY on the provided document context.
Be precise, technical, and cite your sources using the provided citation format.
Rules:
1. Use ONLY information from the retrieved context below
2. If the answer isn't in the context, say "Based on the provided documents, I cannot determine..."
3. Never invent parameters, equations, or experimental conditions
4. When citing, use the EXACT citation string provided
5. For numerical values, include units when available
6. Be concise but technically complete
"""
    user_query = f"""Retrieved Context from Laser Microstructure Documents:
{context}
User Question: {query}
Answer (cite sources using provided citation format, be technical and precise):"""
    return laser_system + user_query

def _create_fusion_aware_prompt(retrieved_docs: List[Document], query: str, fused_properties: Dict[str, FusedPropertyEntry], fusion_metrics: FusionEfficiencyMetrics, comparison_table: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        citation = doc.metadata.get('citation_display', f"[Source {i+1}]")
        content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
        context_parts.append(f"[{i+1}] {citation}\n{content}\n")
    context = "\n---\n".join(context_parts)
    properties_summary = ""
    if fused_properties:
        properties_summary = "**Fused Property Summary**:\n"
        for prop_name, entry in list(fused_properties.items())[:8]:
            if entry.fused_value is not None and isinstance(entry.fused_value, (int, float)):
                value_str = f"{entry.fused_value:.3f}"
                if entry.standard_deviation is not None: value_str = f"{value_str} ± {entry.standard_deviation:.3f}"
            else: value_str = str(entry.fused_value) if entry.fused_value is not None else "N/A"
            properties_summary += f"• {prop_name.replace('_', ' ').title()}: {value_str} {entry.unit or ''} [conf: {entry.fusion_confidence.value}, sources: {entry.source_count}]\n"
        properties_summary += "\n"
    table_section = f"**Comparison Table**:\n{comparison_table}\n" if comparison_table else ""
    efficiency_note = ""
    if fusion_metrics.overall_fusion_efficiency >= 0.7: efficiency_note = f"🎯 High-confidence fusion ({fusion_metrics.overall_fusion_efficiency:.2f}/1.0): Properties synthesized from {fusion_metrics.unique_sources_used} sources.\n"
    elif fusion_metrics.overall_fusion_efficiency >= 0.4: efficiency_note = f"⚠️ Moderate-confidence fusion ({fusion_metrics.overall_fusion_efficiency:.2f}/1.0): Some property variations detected across sources.\n"
    else: efficiency_note = f"🔍 Low-confidence fusion ({fusion_metrics.overall_fusion_efficiency:.2f}/1.0): Limited or conflicting data; interpret with caution.\n"
    
    system_prompt = """You are an expert scientific assistant specializing in laser-microstructure and materials research.
YOUR TASK:
1. Answer the user's question using the retrieved document context AND the fused property summary below
2. When property values are available from fusion, PREFER the fused consensus value with its uncertainty range
3. Cite sources precisely using [Author, Year] or [DOI:xxx] format immediately after claims
4. If fused properties show conflicts, acknowledge the variation and note possible causes (different conditions, methods, materials)
5. For comparative questions, reference the comparison table if provided
6. Always include units for numerical values and note experimental conditions when relevant
RESPONSE STRUCTURE:
1. Direct answer (1-2 sentences)
2. Supporting evidence with fused property values and citations
3. Comparison table reference if relevant to query
4. Uncertainty/limitations note if fusion confidence is moderate/low
5. Suggested follow-up if appropriate
"""
    user_prompt = f"""RETRIEVED DOCUMENT CONTEXT:\n{context}\n{efficiency_note}{properties_summary}{table_section}
USER QUESTION: {query}
SCIENTIFIC ANSWER (use fused properties when available, cite sources precisely):"""
    return system_prompt + user_prompt, {"fused_properties_count": len(fused_properties), "fusion_efficiency": fusion_metrics.overall_fusion_efficiency, "comparison_table_available": comparison_table is not None}

def generate_local_response_transformers(tokenizer, model, device: str, prompt: str, backend_name: str) -> str:
    try:
        if "Qwen" in backend_name or "qwen" in backend_name.lower() or "Llama" in backend_name or "llama" in backend_name.lower():
            messages = [{"role": "system", "content": "You are an expert in laser-microstructure interaction."}, {"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        elif "Mistral" in backend_name or "mistral" in backend_name.lower(): formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        else: formatted_prompt = prompt
        inputs = tokenizer.encode(formatted_prompt, return_tensors='pt', truncation=True, max_length=LASER_DOMAIN_CONFIG["max_context_tokens"])
        if device == "cuda" and torch.cuda.is_available(): inputs = inputs.to('cuda')
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=LASER_DOMAIN_CONFIG["max_new_tokens"], temperature=LASER_DOMAIN_CONFIG["temperature"], do_sample=(LASER_DOMAIN_CONFIG["temperature"] > 0), pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=3, early_stopping=True)
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in full_text: answer = full_text.split("[/INST]")[-1].strip()
        elif "Answer (cite sources" in full_text: answer = full_text.split("Answer (cite sources")[-1].strip(); answer = re.split(r'\n(?:Question|User|Context):', answer)[0].strip()
        else: answer = full_text[-LASER_DOMAIN_CONFIG["max_new_tokens"]*2:].strip()
        return answer if answer else "I was unable to generate a response. Please try rephrasing your question."
    except Exception as e: st.error(f"Generation error: {e}"); return f"Error generating response: {str(e)[:200]}..."

def generate_local_response_ollama(model_tag: str, ollama_host: str, prompt: str) -> str:
    try:
        client = ollama.Client(host=ollama_host)
        messages = [{"role": "system", "content": "You are an expert in laser-microstructure interaction research. Answer based ONLY on the provided context."}, {"role": "user", "content": prompt}]
        try:
            response = client.chat(model=model_tag, messages=messages, options={"temperature": LASER_DOMAIN_CONFIG["temperature"], "num_predict": LASER_DOMAIN_CONFIG["max_new_tokens"]}, stream=True)
            full_response = ""
            for chunk in response:
                if isinstance(chunk, dict) and 'message' in chunk and 'content' in chunk['message']: full_response += chunk['message']['content']
                elif isinstance(chunk, dict) and 'content' in chunk: full_response += chunk['content']
                elif hasattr(chunk, 'message') and hasattr(chunk.message, 'content'): full_response += chunk.message.content
        except TypeError:
            response = client.chat(model=model_tag, messages=messages, options={"temperature": LASER_DOMAIN_CONFIG["temperature"], "num_predict": LASER_DOMAIN_CONFIG["max_new_tokens"]})
            if isinstance(response, dict): full_response = response.get('message', {}).get('content', '')
            elif hasattr(response, 'message'): full_response = response.message.content
            else: full_response = str(response)
        return full_response.strip() if full_response.strip() else "I was unable to generate a response. Please try rephrasing your question."
    except Exception as e: st.error(f"Ollama generation error: {e}"); return f"Error generating response via Ollama: {str(e)[:200]}..."

def generate_local_response(tokenizer, model_or_tag, device_or_host: str, prompt: str, backend: str, backend_type: str) -> str:
    if backend_type == "ollama": return generate_local_response_ollama(model_or_tag, device_or_host, prompt)
    else: return generate_local_response_transformers(tokenizer, model_or_tag, device_or_host, prompt, backend)

def retrieve_and_answer(vectorstore, tokenizer, model, device_or_host: str, backend: str, backend_type: str, query: str, k: int = None, score_threshold: float = None) -> Tuple[str, List[Document], float]:
    k = k or LASER_DOMAIN_CONFIG["retrieval_k"]; score_threshold = score_threshold or LASER_DOMAIN_CONFIG["score_threshold"]
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": k, "score_threshold": score_threshold})
    retrieved_docs = retriever.invoke(query)
    if retrieved_docs:
        query_embedding = vectorstore.embedding_function.embed_query(query); scores = []
        for doc in retrieved_docs:
            doc_embedding = vectorstore.embedding_function.embed_query(doc.page_content[:500])
            sim = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-8); scores.append(sim)
        avg_relevance = np.mean(scores) if scores else 0.0
    else: avg_relevance = 0.0
    if not retrieved_docs: return "Based on the uploaded documents, I could not find information relevant to your question. Try rephrasing or checking document content.", [], avg_relevance
    prompt = create_laser_rag_prompt(retrieved_docs, query)
    answer = generate_local_response(tokenizer=tokenizer, model_or_tag=model, device_or_host=device_or_host, prompt=prompt, backend=backend, backend_type=backend_type)
    return answer, retrieved_docs, avg_relevance

def retrieve_and_answer_with_fusion(vectorstore, tokenizer, model, device_or_host: str, backend: str, backend_type: str, query: str, k: int = None, score_threshold: float = None, enable_fusion: bool = True, material_filter: Optional[str] = None, property_filter: Optional[List[str]] = None) -> Tuple[str, List[Document], float, Dict[str, Any]]:
    k = k or LASER_DOMAIN_CONFIG["retrieval_k"]; score_threshold = score_threshold or LASER_DOMAIN_CONFIG["score_threshold"]
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": k, "score_threshold": score_threshold})
    retrieved_docs = retriever.invoke(query)
    if retrieved_docs:
        query_embedding = vectorstore.embedding_function.embed_query(query); scores = []
        for doc in retrieved_docs:
            doc_embedding = vectorstore.embedding_function.embed_query(doc.page_content[:500])
            sim = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-8); scores.append(sim)
        avg_relevance = np.mean(scores) if scores else 0.0
    else: avg_relevance = 0.0
    if not retrieved_docs: return "Based on the uploaded documents, I could not find information relevant to your question. Try rephrasing or checking document content.", [], avg_relevance, {"error": "no_relevant_chunks", "fusion_enabled": enable_fusion}
    
    if enable_fusion:
        property_extractor = MultiDocumentPropertyExtractor(LASER_KEYWORDS)
        fusion_engine = MultiDocumentFusionEngine(property_extractor)
        fused_properties, fusion_metrics = fusion_engine.fuse_documents(retrieved_docs, query, material_filter=material_filter, property_filter=property_filter)
        comparison_table = None
        if fused_properties: comparison_table = fusion_engine.generate_comparison_table(fused_properties, format="markdown")
        prompt, fusion_context = _create_fusion_aware_prompt(retrieved_docs, query, fused_properties, fusion_metrics, comparison_table)
        answer = generate_local_response(tokenizer=tokenizer, model_or_tag=model, device_or_host=device_or_host, prompt=prompt, backend=backend, backend_type=backend_type)
        if fusion_metrics.overall_fusion_efficiency > 0.5 and comparison_table: answer += f"\n---\n**📊 Property Comparison**:\n{comparison_table}"
        metadata = {"fusion_enabled": True, "fusion_metrics": {"efficiency": fusion_metrics.overall_fusion_efficiency, "display": fusion_metrics.to_display_dict()}, "fused_properties": {k: v.to_comparison_row() for k, v in fused_properties.items()}, "comparison_table": comparison_table, "source_citations": [{"citation": doc.metadata.get('citation_display', 'Unknown'), "relevance": scores[i] if i < len(scores) else 0, "topics": doc.metadata.get('laser_topics', [])} for i, doc in enumerate(retrieved_docs)], "retrieval_relevance": avg_relevance}
        
        # Update global entity map
        new_ents = {}
        for doc in retrieved_docs:
            for ent in property_extractor.extract_properties_from_chunk(doc.page_content, doc.metadata).extracted_properties:
                norm = ent.normalized_name
                if norm not in new_ents: new_ents[norm] = ScientificEntity(name=ent.name, normalized=norm, count=0)
                new_ents[norm].count += 1
                new_ents[norm].doc_sources.add(doc.metadata.get('source', 'unknown'))
                
        # Merge into session state
        if 'entity_map' not in st.session_state: st.session_state.entity_map = {}
        for k, v in new_ents.items():
            if k in st.session_state.entity_map:
                st.session_state.entity_map[k].count += v.count
                st.session_state.entity_map[k].doc_sources.update(v.doc_sources)
            else: st.session_state.entity_map[k] = v
        return answer, retrieved_docs, avg_relevance, metadata
    else:
        prompt = create_laser_rag_prompt(retrieved_docs, query)
        answer = generate_local_response(tokenizer=tokenizer, model_or_tag=model, device_or_host=device_or_host, prompt=prompt, backend=backend, backend_type=backend_type)
        return answer, retrieved_docs, avg_relevance, {"fusion_enabled": False}

# =============================================
# STREAMLIT UI COMPONENTS - ENHANCED
# =============================================
def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        backend_option = st.radio("🔧 Inference Backend", options=["Hugging Face Transformers", "Ollama (if installed)"], index=0, help="Transformers: direct HF model loading\nOllama: use local ollama serve (faster switching)")
        st.session_state.inference_backend = backend_option
        
        if backend_option == "Ollama (if installed)":
            if not OLLAMA_AVAILABLE: st.error("❌ ollama library not installed"); st.code("pip install ollama")
            available_ollama_models = [k for k in LOCAL_LLM_OPTIONS.keys() if is_ollama_model(k)]
            model_choice = st.selectbox("🧠 Local LLM Backend (Ollama)", options=available_ollama_models if available_ollama_models else ["No Ollama models available"], index=0 if available_ollama_models else 0, help="Models served via local Ollama instance")
        else:
            hf_models = [k for k in LOCAL_LLM_OPTIONS.keys() if not is_ollama_model(k)]
            model_choice = st.selectbox("🧠 Local LLM Backend (Hugging Face)", options=hf_models, index=2, help="Models loaded directly via transformers library")
        st.session_state.llm_model_choice = model_choice
        
        if backend_option == "Hugging Face Transformers" and not is_ollama_model(model_choice):
            st.session_state.use_4bit_quantization = st.checkbox("🗜️ Use 4-bit quantization (reduces VRAM usage)", value=True, help="Enable for models >3B parameters to reduce memory usage by ~75%")
        if backend_option == "Ollama (if installed)" or is_ollama_model(model_choice):
            st.session_state.ollama_host = st.text_input("🌐 Ollama Host", value=st.session_state.ollama_host, help="URL of your Ollama server (default: http://localhost:11434)")
            
        st.markdown("#### 🔬 Laser Domain Settings")
        st.session_state.laser_domain_boost = st.checkbox("Boost laser-topic relevance", value=True)
        st.session_state.show_sources = st.checkbox("Show source citations", value=True)
        st.session_state.enable_multi_doc_fusion = st.checkbox("🔗 Enable Multi-Document Fusion", value=True)
        st.session_state.debug_extraction = st.checkbox("🐛 Debug Property Extraction", value=False)
        st.session_state.evaluation_mode = st.checkbox("📊 Enable Evaluation Mode", value=False)
        st.markdown("#### 📝 Citation Format")
        st.session_state.citation_style = st.selectbox("Citation display style", options=["apa", "doi", "full", "short"], index=0, format_func=lambda x: {"apa": "APA: FirstAuthor et al., Journal, Year", "doi": "DOI: 10.xxxx/xxxxx", "full": "Full: Author (Year). Title. Journal, Vol(Issue), Pages", "short": "Short: [FirstAuthor Year] or [DOI]"}[x])
        st.session_state.max_retrieved_chunks = st.slider("Chunks to retrieve", min_value=2, max_value=8, value=4)

        st.markdown("---")
        st.markdown("""<div style="background:#f0f9ff;padding:1rem;border-radius:0.5rem;border-left:4px solid #3b82f6">
<strong>💡 Tips for Best Results:</strong><ul style="margin:0.5rem 0 0 1rem;padding:0">
<li>Upload papers about laser ablation, LIPSS, ultrafast processing</li>
<li>Ask specific questions: "What fluence threshold for silicon ablation?"</li>
<li>Small models (≤1.5B) work on CPU; larger need GPU</li>
<li>First load may take 1-2 min (model download)</li>
<li>For Ollama: run <code>ollama pull qwen2.5:7b</code> first</li>
<li>🔗 Fusion works best with comparative queries across multiple studies</li>
<li>🐛 Enable debug mode to see extracted properties</li>
<li>📊 Enable evaluation mode for metrics</li>
</ul></div>""", unsafe_allow_html=True)

        st.markdown("---")
        gpu_info = "CUDA" if torch.cuda.is_available() else "CPU"
        vram_info = f"{get_available_gpu_memory():.1f}GB free" if torch.cuda.is_available() and get_available_gpu_memory() else "N/A"
        st.caption(f"🖥️ Device: {gpu_info}"); st.caption(f"💾 Available VRAM: {vram_info}")
        st.caption(f"📦 Embedding model: ~80MB"); st.caption(f"🤖 LLM: {LOCAL_LLM_OPTIONS.get(model_choice, 'unknown')}")

def render_document_uploader():
    st.markdown("### 📁 Upload Laser Microstructure Documents")
    uploaded_files = st.file_uploader("Select PDF or TXT files about laser processing, ablation, microstructuring, etc.", type=["pdf", "txt"], accept_multiple_files=True, help="Documents will be processed locally - no data leaves your browser session.")
    return uploaded_files

def process_documents(uploaded_files):
    if not uploaded_files: return False
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    if not new_files: st.info("✓ All uploaded files already processed"); return st.session_state.processing_complete
    st.session_state.messages = []; st.session_state.vectorstore = None; st.session_state.all_chunks = []
    with st.spinner(f"Processing {len(new_files)} document(s) and extracting bibliographic metadata..."):
        try:
            chunks = load_and_chunk_laser_documents(new_files)
            if not chunks: st.error("No chunks extracted. Check file format."); return False
            for f in new_files: st.session_state.processed_files.add(f.name)
            st.session_state.all_chunks.extend(chunks)
            with st.spinner("Creating vector index (this may take a minute)..."):
                vectorstore = create_local_vector_store(st.session_state.all_chunks, LOCAL_EMBEDDING_MODEL)
                if vectorstore is None: return False
                st.session_state.vectorstore = vectorstore
                st.success(f"✅ Ready! Indexed {len(st.session_state.all_chunks)} chunks from {len(st.session_state.processed_files)} files")
                st.session_state.processing_complete = True
                return True
        except Exception as e: st.error(f"Processing failed: {e}"); import traceback; st.error(traceback.format_exc()); return False

def render_fusion_metrics_panel(fusion_metadata: Dict[str, Any]):
    if not fusion_metadata.get("fusion_enabled"): return
    metrics_display = fusion_metadata.get("fusion_metrics", {}).get("display", {})
    if not metrics_display: return
    with st.expander("📊 Information Fusion Efficiency", expanded=True):
        overall = fusion_metadata["fusion_metrics"]["efficiency"]
        if overall is not None: st.progress(min(1.0, max(0.0, overall))); st.caption(f"Overall Fusion Efficiency: {overall:.2f}/1.0")
        else: st.caption("Overall Fusion Efficiency: N/A")
        cols = st.columns(2)
        for i, (label, value) in enumerate(list(metrics_display.items())):
            with cols[i % 2]: st.metric(label=label, value=value.split(":")[-1].strip() if ":" in value and value else "N/A")

def render_extracted_properties_debug(extracted_props: List[ExtractedProperty], source_citation: str):
    if not extracted_props: st.info("🔍 No properties extracted from this chunk"); return
    with st.expander(f"🐛 Extracted Properties: {source_citation}", expanded=False):
        for i, prop in enumerate(extracted_props, 1):
            st.markdown(f"**{i}. {prop.normalized_name}**"); st.caption(f"Value: `{prop.value}` {prop.unit or ''} | Type: {prop.property_type}")
            if prop.condition: st.caption(f"Condition: {prop.condition}")
            if prop.context_snippet: st.code(prop.context_snippet[:200] + "..." if len(prop.context_snippet) > 200 else prop.context_snippet, language="text")
            st.divider()

def render_comparison_table_in_chat(comparison_table: Optional[str], fused_properties: Dict, unique_key_suffix: str):
    if not comparison_table: return
    with st.expander("📋 Property Comparison Table", expanded=False):
        st.markdown(comparison_table, unsafe_allow_html=True)
        if fused_properties:
            # FIXED: Use unique key suffix to prevent StreamlitDuplicateElementKey error
            selected_prop = st.selectbox("🔍 Explore property details:", options=["Select a property..."] + list(fused_properties.keys()), key=f"fusion_prop_select_{unique_key_suffix}")
            if selected_prop and selected_prop != "Select a property...":
                prop_data = fused_properties[selected_prop]
                st.json({"property": selected_prop, "fused_value": prop_data["value"], "unit": prop_data["unit"], "range": prop_data["range"], "sources": prop_data["sources"], "confidence": prop_data["confidence"]})

def render_evaluation_dashboard(vectorstore, embed_model):
    st.header("📊 RAG Evaluation Dashboard"); st.caption("Physics-aware quality assessment for laser-microstructure retrieval")
    evaluator = RetrievalEvaluator(embed_model); faithfulness_checker = PhysicsFaithfulnessChecker(embed_model)
    tabs = st.tabs(["🔍 Retrieval Metrics", "🧮 Physics Validation", "🎯 Benchmark Suite", "📈 Trends"])
    with tabs[0]:
        st.subheader("Retrieval Quality Analysis"); test_query = st.text_input("Enter test query:", "What is the Gibbs free energy for FCC Fe-Cr at 843K?", key="eval_query")
        if st.button("Evaluate Retrieval", key="eval_retrieval"):
            with st.spinner("Evaluating..."):
                retriever = vectorstore.as_retriever(search_kwargs={"k": 10}); retrieved = retriever.invoke(test_query)
                metrics = evaluator.evaluate_query(test_query, retrieved, relevant_doc_ids=set(), relevance_scores={})
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Recall@5", f"{metrics.recall_at_k.get(5, 0):.2f}"); col2.metric("Precision@5", f"{metrics.precision_at_k.get(5, 0):.2f}")
                col3.metric("MRR", f"{metrics.mrr:.3f}"); col4.metric("Context Relevance", f"{metrics.context_relevance:.3f}")
    with tabs[1]:
        st.subheader("Physics-Aware Output Validation"); sample_answer = st.text_area("Paste LLM answer to validate:", height=200, key="sample_answer")
        if st.button("Validate Physics", key="validate_physics") and sample_answer:
            result = faithfulness_checker.full_check(sample_answer, []); trust = result["overall_trust_score"]
            st.progress(trust, text=f"Trust Score: {trust*100:.0f}%")
            if result["is_physics_valid"]: st.success("✅ No physical violations detected")
            else: st.error(f"❌ {result['total_violations']} violation(s) found")
            faithfulness_checker.render_violation_report()
    with tabs[2]:
        st.subheader("DECLARMIMA Benchmark Suite")
        if st.button("Run Full Benchmark", key="run_benchmark"):
            with st.spinner("Running benchmark suite (this may take a while)..."):
                benchmark_df = run_benchmark_evaluation(vectorstore, evaluator, k=5)
                if benchmark_df is not None: st.dataframe(benchmark_df, use_container_width=True)
                if "category" in benchmark_df.columns:
                    st.markdown("### Performance by Category"); cat_perf = benchmark_df.groupby("category")[["recall@5", "precision@5", "mrr"]].mean(); st.bar_chart(cat_perf)
    with tabs[3]:
        st.subheader("Performance Trends")
        if evaluator.query_history:
            trend_df = evaluator.get_aggregate_report()
            if len(trend_df) > 0 and "query" in trend_df.columns: st.line_chart(trend_df.set_index("query")[["recall@5", "precision@5", "mrr"]])
            else: st.info("Run evaluations to see trends.")

def render_visualization_dashboard():
    if not st.session_state.get('entity_map') and not st.session_state.get('processed_files'):
        st.info("👆 Process documents first to see knowledge graphs and visualizations.")
        return
    st.header("🔬 Knowledge Graph & Visualizations")
    
    viz = VisualizationEngine(st.session_state.get('entity_map', {}), {f: {} for f in st.session_state.processed_files})
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Top Entities", "Knowledge Graph", "Hierarchies", "Co-occurrence", "Doc Profiles"])
    
    with tab1:
        if st.button("Generate Top-N Entities Bar Chart"):
            fig = viz.top_entities_bar(15)
            if fig: st.plotly_chart(fig, use_container_width=True)
            else: st.warning("No entities extracted yet.")
    with tab2:
        if st.button("Generate Static Network Graph"):
            fig = viz.static_network_graph(15)
            if fig: st.pyplot(fig)
            else: st.warning("No edges found to draw.")
    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Methods Sunburst"):
                fig = viz.methods_sunburst()
                if fig: st.plotly_chart(fig, use_container_width=True)
        with col_b:
            if st.button("Materials Sunburst"):
                fig = viz.materials_sunburst()
                if fig: st.plotly_chart(fig, use_container_width=True)
    with tab4:
        if st.button("Generate Co-occurrence Heatmap"):
            fig = viz.cooccurrence_heatmap(10)
            if fig: st.plotly_chart(fig, use_container_width=True)
    with tab5:
        if st.button("Document Radar Chart"):
            fig = viz.radar_docs()
            if fig: st.plotly_chart(fig, use_container_width=True)

def render_chat_interface():
    if not st.session_state.get('vectorstore'):
        st.info("👆 Upload documents above to start chatting with your laser microstructure knowledge base")
        return
    if st.session_state.llm_tokenizer is None and st.session_state.llm_model_choice:
        backend_type = "ollama" if is_ollama_model(st.session_state.llm_model_choice) else "transformers"
        with st.spinner(f"Loading {st.session_state.llm_model_choice}..."):
            result = load_local_llm(st.session_state.llm_model_choice, use_4bit=st.session_state.get('use_4bit_quantization', True))
            tokenizer, model, device_or_host, loaded_backend = result
            if tokenizer is not None or model is not None:
                st.session_state.llm_tokenizer = tokenizer; st.session_state.llm_model = model; st.session_state.llm_device_or_host = device_or_host; st.session_state.llm_backend_type = loaded_backend
                st.success("✓ Model loaded!")
            else: st.error("Failed to load model. Try selecting a different option."); return

    has_model = (st.session_state.llm_backend_type == "ollama" and st.session_state.llm_model is not None) or (st.session_state.llm_backend_type == "transformers" and st.session_state.llm_tokenizer is not None)
    if not has_model: st.warning("Please select and load a model in the sidebar first"); return
    
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources") and st.session_state.show_sources:
                with st.expander("📚 Retrieved Sources with Citations"):
                    for i, src in enumerate(message["sources"], 1):
                        citation = src.metadata.get("citation_display", "Unknown source")
                        st.markdown(f"**[{i}]** {citation}")
                        st.markdown(f"> {src.page_content[:300]}...")
                        if st.session_state.debug_extraction:
                            extractor = MultiDocumentPropertyExtractor(LASER_KEYWORDS)
                            record = extractor.extract_properties_from_chunk(src.page_content, src.metadata)
                            render_extracted_properties_debug(record.extracted_properties, citation)
            if message.get("fusion_metadata") and st.session_state.enable_multi_doc_fusion:
                render_fusion_metrics_panel(message["fusion_metadata"])
                # FIXED: Pass unique key suffix based on message index
                render_comparison_table_in_chat(message["fusion_metadata"]["comparison_table"], message["fusion_metadata"].get("fused_properties", {}), f"msg_{idx}")
                
    if prompt := st.chat_input("Ask about laser parameters, material properties, or compare studies..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("🔍 Retrieving, fusing multi-document data, and generating..."):
                try:
                    if st.session_state.enable_multi_doc_fusion:
                        answer, retrieved_docs, relevance, metadata = retrieve_and_answer_with_fusion(vectorstore=st.session_state.vectorstore, tokenizer=st.session_state.llm_tokenizer, model=st.session_state.llm_model, device_or_host=st.session_state.llm_device_or_host, backend=st.session_state.llm_model_choice, backend_type=st.session_state.llm_backend_type, query=prompt, k=st.session_state.max_retrieved_chunks, enable_fusion=True)
                    else:
                        answer, retrieved_docs, relevance = retrieve_and_answer(vectorstore=st.session_state.vectorstore, tokenizer=st.session_state.llm_tokenizer, model=st.session_state.llm_model, device_or_host=st.session_state.llm_device_or_host, backend=st.session_state.llm_model_choice, backend_type=st.session_state.llm_backend_type, query=prompt, k=st.session_state.max_retrieved_chunks)
                        metadata = {"fusion_enabled": False}
                    display_text = ""
                    for word in answer.split(): display_text += word + " "; message_placeholder.markdown(display_text + "▌"); time.sleep(0.02)
                    message_placeholder.markdown(answer)
                    message_dict = {"role": "assistant", "content": answer, "sources": retrieved_docs if st.session_state.show_sources else None, "relevance": relevance}
                    if st.session_state.enable_multi_doc_fusion: message_dict["fusion_metadata"] = metadata
                    st.session_state.messages.append(message_dict)
                    fusion_eff = metadata.get("fusion_metrics", {}).get("efficiency", 0) if metadata.get("fusion_enabled") else None
                    st.caption(f"📊 Response relevance: {relevance:.2f}/1.0 | Fusion efficiency: {fusion_eff:.2f}/1.0" if fusion_eff is not None else f"📊 Response relevance: {relevance:.2f}/1.0")
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)[:300]}"; st.error(error_msg); st.session_state.messages.append({"role": "assistant", "content": error_msg})

def render_footer():
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📚 Example Questions:**"); st.caption("• What is the ablation threshold for silicon at 800nm?"); st.caption("• How does pulse duration affect LIPSS formation?")
    with col2:
        st.markdown("**⚡ Performance Tips:**"); st.caption("• Keep questions focused and specific"); st.caption("• CPU mode: allow 10-30s per response")
    with col3:
        st.markdown("**🔐 Privacy & Fusion:**"); st.caption("• All processing happens locally in your session"); st.caption("• Multi-document fusion extracts & compares properties across studies")

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    st.set_page_config(page_title="🔬 Laser Microstructure RAG Assistant", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""<style>.main-header {font-size: 2.5rem; background: linear-gradient(90deg, #1e40af, #7c3aed, #059669); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; text-align: center; padding: 1rem 0;}.stChatMessage {border-radius: 0.5rem; margin: 0.25rem 0;}</style>""", unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🔬 Laser Microstructure RAG Assistant</h1>', unsafe_allow_html=True)
    
    initialize_session_state()
    render_sidebar()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        uploaded_files = render_document_uploader()
        if uploaded_files and st.button("🔄 Process Documents", type="primary", use_container_width=True): process_documents(uploaded_files)
        if st.session_state.processing_complete:
            st.success("✅ Knowledge base ready")
            if st.session_state.vectorstore and hasattr(st.session_state.vectorstore, 'metadata'):
                meta = st.session_state.vectorstore.metadata; st.caption(f"📦 {meta.get('total_chunks', '?')} chunks")
                topics = meta.get('laser_topics', [])
                if topics: st.caption(f"🔬 Topics: {', '.join(topics[:5])}" + ("..." if len(topics)>5 else ""))
        elif uploaded_files: st.warning("⏳ Click 'Process Documents' to begin")
        else: st.info("📁 Upload PDF/TXT files to start")
        
        if st.session_state.processed_files:
            if st.button("🗑️ Clear All", use_container_width=True): st.session_state.clear(); st.rerun()
            
    with col2:
        if st.session_state.processing_complete and st.session_state.vectorstore:
            if st.session_state.evaluation_mode:
                st.subheader("📊 Evaluation Dashboard"); render_evaluation_dashboard(st.session_state.vectorstore, load_local_embeddings()); st.divider()
            else:
                render_visualization_dashboard()
                st.divider()
            render_chat_interface()
        else:
            st.markdown("""<div style="background:#f8fafc;border-left:4px solid #3b82f6;padding:1rem;border-radius:0 0.5rem 0.5rem 0;margin:0.5rem 0;">
<h3>👋 Welcome!</h3>
<p>This assistant helps you query documents about:</p>
<ul><li>🔥 Laser ablation thresholds & mechanisms</li><li>🌊 LIPSS and surface morphology formation</li>
<li>⚡ Ultrafast laser-matter interactions</li><li>🔬 Characterization techniques (SEM, AFM, etc.)</li>
<li>📐 Process parameter optimization</li><li>🔗 <strong>Multi-document property comparison</strong></li>
<li>📊 <strong>Retrieval quality metrics</strong> (Recall@k, Precision@k, MRR)</li></ul>
<p><strong>Getting started:</strong></p>
<ol><li>Upload PDF/TXT files in the left panel</li><li>Click "Process Documents"</li>
<li>Select your preferred local LLM in sidebar</li><li>Enable "Multi-Document Fusion" for comparative queries</li></ol></div>""", unsafe_allow_html=True)
            st.markdown("**Try asking:**")
            demo_qs = ["What factors affect ablation threshold in metals?", "How does pulse duration influence LIPSS periodicity?"]
            for q in demo_qs:
                if st.button(f"💬 {q}", use_container_width=True, key=f"demo_{q[:20]}"): st.session_state.messages.append({"role": "user", "content": q}); st.rerun()
    render_footer()

if __name__ == "__main__":
    main()
