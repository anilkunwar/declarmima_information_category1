#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DECLARMIMA: Alloy Microstructure Concept Graph with Dual LLM Backend Support
========================================================================================
✅ Zero API keys - all models run locally (HuggingFace Transformers + Ollama)
✅ Dual backend: Choose between HF Transformers (direct loading) or Ollama (server-based)
✅ ALL models from LASER RAG codebase included (12 HF + 8 Ollama options)
✅ Memory-aware: 4-bit quantization, VRAM estimation, CPU/GPU auto-detection
✅ DECLARMIMA-focused: Laser-microstructure interaction, multicomponent alloys, digital twins
✅ Physics-informed: Concept graphs, GraphSAGE embeddings, research direction scoring
✅ Small-corpus optimized: Adaptive thresholds, semantic clustering, seed injection
✅ ENHANCED: Publication-quality visualizations, metrics dashboard, user customization
✅ ENHANCED: Interactive PyVis with physics toggle, category-based colors, size scaling
✅ ENHANCED: Sunburst chart for research domain hierarchy, link prediction dashboard

DEPLOYMENT:
pip install streamlit torch transformers sentence-transformers networkx scikit-learn
pip install pyvis plotly pandas numpy kaleido  # kaleido for SVG export
pip install ollama  # optional for Ollama backend

Run: streamlit run declarmima_concept_graph.py
"""
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sparse
import torch.optim as optim
import networkx as nx
import numpy as np
import pandas as pd
import re
import json
import os
import tempfile
import warnings
import traceback
import gc
import hashlib
from collections import defaultdict, Counter
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union, Any
from sklearn.linear_model import Ridge
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, GPT2Tokenizer, GPT2LMHeadModel
)
from pyvis.network import Network
import plotly.graph_objects as go
import plotly.express as px

# Optional Ollama import with graceful fallback
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None

warnings.filterwarnings('ignore')

# ==========================================
# STREAMLIT CONFIGURATION (Must be first)
# ==========================================
st.set_page_config(
    page_title="DECLARMIMA: Alloy Microstructure Concept Graph",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com',
        'Report a bug': "https://github.com",
        'About': "DECLARMIMA: Laser-Microstructure Interaction Analyzer"
    }
)

os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# ==========================================
# DEVICE & SEED SETUP
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ==========================================
# MODEL REGISTRY - ALL OPTIONS FROM LASER RAG CODE
# ==========================================
LOCAL_LLM_OPTIONS = {
    # Hugging Face Transformers models
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
    # Ollama models (served via local ollama server)
    "[Ollama] qwen2.5:0.5b (via ollama serve)": "ollama:qwen2.5:0.5b",
    "[Ollama] qwen2.5:1.5b (via ollama serve)": "ollama:qwen2.5:1.5b",
    "[Ollama] qwen2.5:7b (via ollama serve)": "ollama:qwen2.5:7b",
    "[Ollama] qwen2.5:14b (via ollama serve) 🔥": "ollama:qwen2.5:14b",
    "[Ollama] llama3.1:8b (via ollama serve)": "ollama:llama3.1:8b",
    "[Ollama] mistral:7b (via ollama serve)": "ollama:mistral:7b",
    "[Ollama] gemma2:9b (via ollama serve)": "ollama:gemma2:9b",
    "[Ollama] falcon3:10b (via ollama serve)": "ollama:falcon3:10b",
}

# Default model identifiers (<1B constraint for fallback)
DEFAULT_LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
EMBED_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Model memory estimates for VRAM checking
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

# ==========================================
# DECLARMIMA PROJECT: Seed knowledge base
# ==========================================
DECLARMIMA_PROPOSAL_TEXT = """
Deciphering laser-microstructure interaction in multicomponent alloys (DECLARMIMA)
Scientific goals: Additive manufacturing, laser processing, multicomponent alloys, high-entropy alloys, digital twins, physics-informed machine learning, phase field modeling, molecular dynamics, melt pool dynamics, microstructure evolution, process-structure-property relationships, selective laser melting, powder bed fusion, laser powder bed fusion, in-situ monitoring, defect formation, porosity, spatter, residual stress, grain morphology, phase transformation, solidification, Marangoni convection, CALPHAD thermodynamics, interfacial energy, thermal conductivity, viscosity, absorptivity, reflectivity, Gaussian heat source, finite element method, MOOSE framework, LAMMPS, ThermoCalc, neural networks, convolutional neural networks, random forest, Bayesian machine learning, uncertainty quantification, feature engineering, tensor decomposition, scale-bridging, multiscale modeling, inverse design, optimization, Al-Si-Mg alloys, Ti-6Al-4V, Inconel 718, Sn-Ag-Cu solders, CoCrFeNi HEAs, intermetallic compounds, columnar grains, equiaxed grains, dendritic structures, martensite, austenite, precipitates, segregation, crack propagation, fatigue life, tensile strength, yield strength, microhardness, elongation, ductility, wear resistance, corrosion resistance, oxidation resistance, laser power, scan speed, hatch spacing, layer thickness, pulse duration, energy density, spot diameter, cooling rate, solidification rate, dilution ratio, powder particle size, particle size distribution, flowability, oxygen content, moisture content, bed temperature, pre-heating, post-processing, heat treatment, surface finishing, quality monitoring, photodiode sensors, line scanners, camera trackers, acoustic transducers, synchrotron X-ray imaging, EBSD, nanoindentation, in-situ XRD, SEM, TEM, AFM, digital image correlation, machine vision, data fusion, knowledge graphs, concept graphs, graph neural networks, GraphSAGE, node embeddings, edge prediction, link prediction, research direction discovery, hypothesis generation, novelty scoring, feasibility assessment, property gain prediction, composite scoring, adaptive configuration, small corpus optimization, semantic clustering, domain seed injection, hybrid graph construction, co-occurrence edges, semantic similarity edges, contrastive learning, edge sampling, sparse tensors, degree normalization, mean aggregation, two-layer architecture, decoder network, BCE loss, Adam optimizer, training loop, evaluation metrics, progress tracking, memory management, CUDA optimization, CPU fallback, error handling, fallback strategies, interactive visualization, PyVis, Plotly, force-directed layout, spring layout, node styling, edge styling, hover tooltips, download functionality, text fallback, diagnostics panel, concept frequency, edge weight, graph connectivity, component analysis, degree distribution, clustering coefficient, centrality measures, path length, bridge edges, semantic bridges, knowledge injection, concept normalization, alloy notation standardization, laser term normalization, unit standardization, regex extraction, quantitative metrics, grain size, mechanical properties, energy density, defect fraction, prompt engineering, JSON parsing, fallback extraction, domain validation, generic term filtering, concept abstraction, category mapping, hierarchical representation, representative selection, cluster merging, similarity threshold, distance matrix, linkage method, embedding encoding, batch processing, progress display, model caching, resource management, timeout handling, user feedback, status indicators, progress bars, error messages, warning dialogs, success notifications, download buttons, CSV export, HTML export, JSON export, interactive controls, physics parameters, gravity, spring length, damping, overlap, stabilization, node sampling, size limiting, performance optimization, browser compatibility, JavaScript execution, CDN resources, inline embedding, iframe alternative, HTML rendering, Streamlit components, responsive design, mobile compatibility, accessibility, color contrast, theme switching, dark mode, light mode, user preferences, session state, configuration persistence, adaptive thresholds, corpus size detection, parameter tuning, hyperparameter optimization, validation metrics, testing framework, debugging tools, logging, tracebacks, exception handling, graceful degradation, fallback rendering, text summary, edge listing, frequency tables, diagnostic metrics, connectivity checks, component counting, degree analysis, clustering analysis, centrality computation, path analysis, bridge detection, semantic analysis, novelty computation, feasibility scoring, property prediction, ridge regression, feature concatenation, pair scoring, candidate filtering, distance checking, graph distance, shortest path, all-pairs shortest path, cutoff parameter, edge sampling strategy, positive pairs, negative pairs, hard negatives, distance-focused sampling, random sampling, attempts limit, pair uniqueness, edge existence check, tensor construction, sparse adjacency, degree computation, normalization, message passing, aggregation, combination, activation, ReLU, linear layers, sequential decoder, concatenation, sigmoid, logits, contrastive loss, binary cross-entropy, training epochs, learning rate, optimizer step, gradient computation, backward pass, zero grad, model evaluation, no grad context, final embeddings, adjacency indices, adjacency values, node features, embedding dimension, shape validation, error raising, minimal pairs, edge uniqueness, source adjacency, destination adjacency, stacking, tensor conversion, device placement, long dtype, float32, GPU memory, CPU fallback, memory cleanup, garbage collection, CUDA cache emptying, progress callback, epoch logging, loss tracking, convergence monitoring, early stopping, model saving, checkpointing, inference mode, prediction scoring, candidate generation, random sampling, pair filtering, distance computation, KeyError handling, default distance, semantic similarity, cosine similarity, embedding encoding, numpy arrays, tensor conversion, CPU numpy, forward pass, model eval, no grad, decoder output, logits extraction, sigmoid activation, CPU conversion, numpy array, property lookup, median computation, ridge prediction, clipping, normalization, weighted scoring, alpha weights, composite score, sorting, head selection, DataFrame creation, column selection, formatting, display configuration, download preparation, CSV serialization, MIME type, button callback, empty check, info message, parameter suggestion, graph rendering, node count check, edge count check, fallback graph building, semantic-only fallback, similarity threshold adjustment, success message, text fallback rendering, node iteration, degree computation, frequency lookup, category detection, color assignment, size computation, title formatting, node addition, edge iteration, weight lookup, type lookup, color mapping, edge addition, value scaling, width scaling, color assignment, smooth edges, curved edges, roundness parameter, HTML generation, inline resources, Streamlit HTML component, height parameter, scrolling enable, width parameter, download button, file naming, MIME type, unique key, error catching, warning display, fallback suggestion, retry buttons, alternative backend, exception handling, error message display, traceback expansion, code display, memory cleanup, GPU cache clearing, garbage collection, footer display, tips section, visualization options, PyVis description, Plotly description, text summary description, technical stack, crash prevention tips, rendering troubleshooting, browser console check, zoom controls, download fallback, text view guarantee
"""

# Domain keywords for laser processing + multicomponent alloys
DOMAIN_KEYWORDS = [
    "grain size", "phase fraction", "microhardness", "tensile strength", 
    "yield strength", "elongation", "residual stress", "texture intensity",
    "columnar grain", "equiaxed grain", "dendrite", "eutectic", "martensite",
    "austenite", "precipitate", "segregation", "porosity", "crack density",
    "intermetallic compound", "IMC", "interfacial microstructure", "melt pool",
    "solidification front", "grain boundary", "phase transformation", "nucleation",
    "laser power", "scan speed", "hatch spacing", "layer thickness",
    "pulse duration", "energy density", "spot diameter", "cooling rate",
    "solidification rate", "dilution ratio", "Gaussian heat source", "absorptivity",
    "reflectivity", "beam intensity", "laser wavelength", "Marangoni convection",
    "high-entropy alloy", "HEA", "multi-principal element", "complex concentrated",
    "powder bed fusion", "LPBF", "direct energy deposition", "DED", "selective laser melting",
    "AlSi10Mg", "Ti6Al4V", "Inconel718", "SnAgCu", "CoCrFeNi", "solder alloy",
    "phase field", "molecular dynamics", "finite element", "CALPHAD", "digital twin",
    "physics-informed", "machine learning", "neural network", "graph neural network",
    "feature engineering", "semantic similarity", "concept graph", "knowledge graph",
    "thermal conductivity", "viscosity", "interfacial energy", "diffusion coefficient",
    "Gibbs free energy", "enthalpy", "entropy", "atomic mobility", "thermodynamic database",
    "defect formation", "spatter ejection", "keyhole formation", "lack of fusion",
    "residual stress mitigation", "grain refinement", "texture evolution", "phase stability"
]

ALLOY_PATTERNS = [
    r'[A-Z][a-z]?(?:\d+(?:\.\d+)?(?:[A-Z][a-z]?\d*(?:\.\d+)?)*)+',
    r'(?:Ni|Co|Cr|Fe|Al|Ti|Cu|Nb|Mo|W|Sn|Ag|Zn|Bi)(?:[-\s]?\d+(?:\.\d+)?%?)+',
    r'(?:high-entropy|HEA|multi-principal|complex concentrated|MPEA)',
    r'(?:AlSi\d+Mg|Ti6Al4V|Inconel\d+|SnAgCu|CoCrFeNi|SAC\d+)',
]

# Enhanced domain seed concepts from DECLARMIMA proposal
DOMAIN_SEED_CONCEPTS = {
    "alloy_systems": [
        "aluminum alloy", "titanium alloy", "nickel alloy", "high-entropy alloy", 
        "steel", "alsi10mg", "ti6al4v", "inconel718", "snagcu solder", 
        "cocrfeni hea", "multiprincipal element alloy", "complex concentrated alloy"
    ],
    "laser_parameters": [
        "laser power", "scan speed", "energy density", "hatch spacing", 
        "pulse duration", "melt pool depth", "cooling rate", "solidification rate",
        "Gaussian heat source", "beam intensity distribution", "absorptivity",
        "reflectivity", "Marangoni number", "laser wavelength"
    ],
    "microstructure_features": [
        "grain size", "phase fraction", "texture", "porosity", "residual stress", 
        "columnar grain", "equiaxed grain", "dendritic structure", "intermetallic compound",
        "grain boundary", "phase transformation", "nucleation site", "solidification front",
        "melt pool geometry", "interfacial microstructure", "precipitate distribution"
    ],
    "mechanical_properties": [
        "microhardness", "tensile strength", "yield strength", "elongation", 
        "fatigue life", "ductility", "wear resistance", "corrosion resistance",
        "oxidation resistance", "fracture toughness", "creep resistance"
    ],
    "processes": [
        "powder bed fusion", "direct energy deposition", "laser remelting", 
        "surface treatment", "solidification", "selective laser melting",
        "laser powder bed fusion", "wire-feed laser additive manufacturing",
        "in-situ monitoring", "post-process heat treatment"
    ],
    "computational_methods": [
        "phase field modeling", "molecular dynamics", "finite element analysis",
        "CALPHAD thermodynamics", "digital twin", "physics-informed machine learning",
        "graph neural network", "concept extraction", "semantic clustering",
        "feature engineering", "tensor decomposition", "scale-bridging simulation"
    ],
    "declarmima_goals": [
        "decipher laser-microstructure interaction", "physics-informed digital twin",
        "learning laser system", "process-structure-property relationship",
        "multiscale computational modeling", "integrated experiment-computation framework",
        "uncertainty quantification", "inverse design optimization",
        "mechanistic understanding of additive manufacturing"
    ]
}

# Category mapping for hierarchical abstraction
CATEGORY_MAPPING = {
    r'alsi\d+mg|al(?:si|cu|mg|zn)\w*': 'aluminum alloy',
    r'ti6al4v|ti(?:al|nb|mo)\w*': 'titanium alloy', 
    r'inconel\d+|ni(?:cr|mo|fe)\w*': 'nickel alloy',
    r'cocrfeni|he[as]?|high.?entropy|mpea': 'high-entropy alloy',
    r'snagcu|sac\d+|sn(?:ag|cu|bi|zn)\w*': 'solder alloy',
    r'(?:laser\s*)?(?:power|energy\s*density|fluence|beam\s*intensity)': 'laser energy parameter',
    r'(?:scan|travel)\s*speed|feed\s*rate': 'scanning parameter',
    r'hatch\s*spacing|layer\s*thickness|point\s*distance': 'geometric parameter',
    r'(?:columnar|equiaxed|dendritic|fine|coarse)\s*grain': 'grain morphology',
    r'(?:martensite|austenite|eutectic|ferrite|precipitate)\s*(?:phase)?': 'phase type',
    r'(?:micro|nano)hardness|hv\d*|vickers': 'hardness metric',
    r'(?:tensile|yield|ultimate|fracture)\s*strength': 'strength metric',
    r'(?:thermal\s*)?conductivity|diffusivity': 'thermal property',
    r'(?:interfacial|grain\s*boundary)\s*energy': 'interface property',
    r'(?:marangoni|convection|fluid\s*flow)': 'melt pool dynamics',
    r'(?:porosity|void|crack|defect|spatter|keyhole)': 'defect type',
    r'(?:phase\s*field|molecular\s*dynamics|finite\s*element|calphad)': 'computational method',
    r'(?:digital\s*twin|machine\s*learning|neural\s*network|graph\s*neural)': 'data-driven method',
}

# Default hyperparameters
DEFAULT_MIN_CONCEPT_FREQ = 3
DEFAULT_MIN_CONCEPT_LENGTH_WORDS = 2
GNN_HIDDEN_DIM = 128
TRAIN_EPOCHS = 50
LR = 1e-3
NEG_DPREV_FOCUS = 3

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def is_ollama_model(model_key: str) -> bool:
    """Check if model key refers to an Ollama-served model"""
    return model_key.startswith("ollama:") or model_key.startswith("[Ollama]")

def extract_ollama_tag(model_key: str) -> str:
    """Extract the Ollama model tag from a model key"""
    if model_key.startswith("ollama:"):
        return model_key.replace("ollama:", "", 1)
    elif model_key.startswith("[Ollama]"):
        match = re.search(r'\]\s*([^\s(]+)', model_key)
        if match:
            return match.group(1)
    return model_key

def get_hf_repo_id(model_key: str) -> str:
    """Extract Hugging Face repo ID from model key"""
    if ":" in model_key and not model_key.startswith("http"):
        parts = model_key.split(":", 1)
        if len(parts) == 2 and "/" in parts[1]:
            return parts[1].strip()
    return model_key

def get_available_gpu_memory() -> Optional[float]:
    """Get available GPU memory in GB"""
    if not torch.cuda.is_available():
        return None
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        return total_memory - reserved
    except:
        return None

def estimate_model_memory(model_key: str, use_4bit: bool = False) -> Dict[str, Any]:
    """Get memory estimate for a model"""
    repo_id = get_hf_repo_id(model_key) if not is_ollama_model(model_key) else model_key
    return MODEL_MEMORY_ESTIMATES.get(repo_id, {
        "params": "Unknown", "vram_fp16": "Unknown", 
        "vram_4bit": "Unknown", "cpu_ok": False
    })

def compute_file_hash(filepath: str) -> str:
    """Compute MD5 hash of a file for caching"""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return ""

# ==========================================
# ADAPTIVE CONFIGURATION FOR SMALL CORPORA
# ==========================================
def get_adaptive_config(num_abstracts: int) -> Dict[str, Any]:
    """Dynamically adjust parameters based on input corpus size"""
    if num_abstracts <= 15:
        return {
            "MIN_CONCEPT_FREQ": 1, "MIN_CONCEPT_LENGTH_WORDS": 1, "MIN_DEGREE": 1,
            "USE_SEMANTIC_CLUSTERING": True, "INJECT_DOMAIN_SEEDS": True,
            "USE_SEMANTIC_EDGES": True, "SIMILARITY_THRESHOLD": 0.70,
            "COOCCURRENCE_WEIGHT": 0.4, "SEMANTIC_WEIGHT": 0.6,
            "CLUSTER_SIMILARITY": 0.78, "USE_DECLARMIMA_SEEDS": True,
            "CORRELATE_WITH_PROPOSAL": True,
        }
    elif num_abstracts <= 30:
        return {
            "MIN_CONCEPT_FREQ": 2, "MIN_CONCEPT_LENGTH_WORDS": 2, "MIN_DEGREE": 1,
            "USE_SEMANTIC_CLUSTERING": True, "INJECT_DOMAIN_SEEDS": True,
            "USE_SEMANTIC_EDGES": True, "SIMILARITY_THRESHOLD": 0.75,
            "COOCCURRENCE_WEIGHT": 0.6, "SEMANTIC_WEIGHT": 0.4,
            "CLUSTER_SIMILARITY": 0.75, "USE_DECLARMIMA_SEEDS": True,
            "CORRELATE_WITH_PROPOSAL": True,
        }
    else:
        return {
            "MIN_CONCEPT_FREQ": 3, "MIN_CONCEPT_LENGTH_WORDS": 2, "MIN_DEGREE": 2,
            "USE_SEMANTIC_CLUSTERING": False, "INJECT_DOMAIN_SEEDS": False,
            "USE_SEMANTIC_EDGES": False, "SIMILARITY_THRESHOLD": 0.80,
            "COOCCURRENCE_WEIGHT": 0.8, "SEMANTIC_WEIGHT": 0.2,
            "CLUSTER_SIMILARITY": 0.72, "USE_DECLARMIMA_SEEDS": False,
            "CORRELATE_WITH_PROPOSAL": False,
        }

# ==========================================
# MODEL LOADING (CACHED FOR STREAMLIT)
# ==========================================
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load Sentence-BERT embedding model with device handling"""
    try:
        return SentenceTransformer(EMBED_NAME, device=DEVICE)
    except Exception as e:
        st.error(f"❌ Failed to load embedding model: {e}")
        st.info("💡 Falling back to CPU-only mode")
        return SentenceTransformer(EMBED_NAME, device='cpu')

@st.cache_resource(show_spinner="Loading LLM (this may take 1-2 minutes on first load)...")
def load_local_llm(model_key: str, use_4bit: bool = True):
    """Unified LLM loader with backend routing (HF Transformers or Ollama)"""
    try:
        if is_ollama_model(model_key):
            return _load_ollama_model(model_key)
        else:
            return _load_transformers_model(model_key, use_4bit)
    except Exception as e:
        st.error(f"Failed to load LLM '{model_key}': {e}")
        st.warning("Falling back to GPT-2...")
        try:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            return tokenizer, model, device, "transformers"
        except Exception as e2:
            st.error(f"Fallback also failed: {e2}")
            return None, None, None, None

def _load_ollama_model(model_key: str) -> Tuple[Optional[Any], str, str, str]:
    """Load Ollama model via HTTP client"""
    if not OLLAMA_AVAILABLE:
        raise ImportError("ollama library not installed. Run: pip install ollama")
    
    model_tag = extract_ollama_tag(model_key)
    ollama_host = st.session_state.get('ollama_host', 'http://localhost:11434')
    
    try:
        client = ollama.Client(host=ollama_host)
        response = client.list()
        models_list = response.get('models', []) if isinstance(response, dict) else getattr(response, 'models', [])
        model_names = []
        for m in models_list:
            if isinstance(m, dict):
                name = m.get('model') or m.get('name')
            else:
                name = getattr(m, 'model', None) or getattr(m, 'name', None)
            if name:
                model_names.append(name)
        
        if model_tag not in model_names:
            st.warning(f"⚠️ Model '{model_tag}' not found in Ollama.")
            if model_names:
                st.info(f"📋 Available: {', '.join(model_names[:5])}")
            return None, None, ollama_host, "ollama"
    except Exception as conn_err:
        st.error(f"❌ Connection Error: {conn_err}")
        return None, None, ollama_host, "ollama"
    
    return None, model_tag, ollama_host, "ollama"

def _load_transformers_model(model_key: str, use_4bit: bool = True) -> Tuple[Any, Any, str, str]:
    """Load Hugging Face Transformers model with memory optimization"""
    repo_id = get_hf_repo_id(model_key)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    available_vram = get_available_gpu_memory()
    mem_info = estimate_model_memory(model_key, use_4bit)
    
    # Display memory info in sidebar
    st.sidebar.info(f"""📊 Model Memory Estimate:
- Parameters: {mem_info['params']}
- VRAM (FP16): {mem_info['vram_fp16']}
- VRAM (4-bit): {mem_info['vram_4bit']}
- CPU OK: {'✅ Yes' if mem_info['cpu_ok'] else '❌ No'}
- Available VRAM: {f'{available_vram:.1f}GB' if available_vram else 'N/A (CPU)'}""")
    
    # Disable 4-bit for very small models
    if "0.5B" in repo_id or "1.1B" in repo_id or "gpt2" in repo_id:
        use_4bit = False
    
    quantization_config = None
    if use_4bit and device == "cuda" and available_vram:
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            )
            st.sidebar.success("✅ 4-bit quantization enabled")
        except ImportError:
            st.sidebar.warning("⚠️ bitsandbytes not installed.")
            use_4bit = False
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True, padding_side="left", use_fast=True)
    model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.float16 if device == "cuda" else torch.float32}
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    elif device == "cuda":
        model_kwargs["device_map"] = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(repo_id, **model_kwargs)
    
    if "device_map" not in model_kwargs and device == "cpu":
        model = model.to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer, model, device, "transformers"

# ==========================================
# DOMAIN-SPECIFIC CONCEPT NORMALIZATION
# ==========================================
def normalize_alloy_composition(concept: str) -> str:
    """Standardize alloy notation (e.g., 'Ti-6Al-4V' → 'ti6al4v')"""
    normalized = re.sub(r'[\s\-_]', '', concept).lower()
    normalized = re.sub(r'(ti)(6)(al)(4)(v)', r'ti6al4v', normalized)
    normalized = re.sub(r'(al)(si)(10)(mg)', r'alsi10mg', normalized)
    normalized = re.sub(r'(inconel)(\s*718|718)', r'inconel718', normalized)
    normalized = re.sub(r'(cocrfe)(ni|mn|mo)\w*', r'cocrfeni', normalized)
    normalized = re.sub(r'(sn)(ag)(cu)', r'snagcu', normalized)
    return normalized

def normalize_laser_term(concept: str) -> str:
    """Normalize laser processing terminology and units"""
    concept = concept.lower().strip()
    concept = re.sub(r'\b(j/mm(?:\s*3)?|j mm-3|j mm⁻³)\b', 'j/mm³', concept)
    concept = re.sub(r'\b(w|watt)s?\b', 'w', concept)
    concept = re.sub(r'\b(mm/s|mm s-1|mm s⁻¹)\b', 'mm/s', concept)
    concept = re.sub(r'\b(μm|micron|um)\b', 'um', concept)
    return concept

def is_valid_microstructure_concept(concept: str) -> bool:
    """Filter concepts relevant to laser/alloy microstructure research"""
    concept_lower = concept.lower()
    has_domain_keyword = any(kw in concept_lower for kw in DOMAIN_KEYWORDS)
    has_alloy_pattern = any(re.search(p, concept, re.I) for p in ALLOY_PATTERNS)
    generic_terms = {'study', 'analysis', 'effect', 'role', 'investigation', 
                     'research', 'method', 'approach', 'paper', 'work', 'using'}
    has_generic = any(term in concept_lower.split() for term in generic_terms)
    return (has_domain_keyword or has_alloy_pattern) and not has_generic

# ==========================================
# DECLARMIMA: PROPOSAL-BASED CONCEPT EXTRACTION
# ==========================================
def extract_declarmima_concepts(proposal_text: str, embed_model) -> list:
    """Extract domain concepts from DECLARMIMA proposal text"""
    patterns = [
        r'\b(?:[A-Z][a-z]+(?:\d+(?:\.\d+)?)?[\s\-]?){2,4}(?:alloy|phase|grain|microstructure|strength|hardness|property)',
        r'\b(?:laser|powder|bed|fusion|selective|direct|melting)\s+(?:power|speed|scanning|melting|parameters|energy|processing)',
        r'\b(?:columnar|equiaxed|fine|coarse|nanoscale|bimodal|dendritic)\s+(?:grain|structure|region|zone|morphology)',
        r'\b(?:martensite|austenite|ferrite|eutectic|peritectic|precipitate|intermetallic)\s+(?:formation|phase|fraction|compound)',
        r'\b(?:microhardness|nanohardness|tensile|yield|ductility|elongation|fatigue)\s+(?:improvement|strength|property|life)',
        r'\b(?:phase\s*field|molecular\s*dynamics|finite\s*element|calphad|digital\s*twin)\s*(?:model|simulation|method|framework)?',
        r'\b(?:high-entropy|HEA|multi[-\s]?principal|complex\s*concentrated)\s*(?:alloy|material)?',
        r'\b(?:AlSi\d+Mg|Ti6Al4V|Inconel\d+|SnAgCu|CoCrFeNi|SAC\d+)\b',
    ]
    
    concepts = set()
    for pattern in patterns:
        matches = re.findall(pattern, proposal_text, re.I)
        for m in matches:
            concept = m.lower().strip().rstrip('.')
            if len(concept.split()) >= 2 and is_valid_microstructure_concept(concept):
                concepts.add(concept)
    
    declarmina_goals = [
        "decipher laser-microstructure interaction", "physics-informed digital twin",
        "learning laser system", "process-structure-property relationship",
        "multiscale computational modeling", "integrated experiment-computation framework"
    ]
    for goal in declarmina_goals:
        concepts.add(goal)
    
    return list(concepts)

def compute_proposal_correlation(concept: str, proposal_embedding: np.ndarray, 
                                concept_embedding: np.ndarray) -> float:
    """Compute semantic correlation between a concept and DECLARMIMA proposal"""
    sim = cosine_similarity([concept_embedding], [proposal_embedding])[0][0]
    return float(np.clip(sim, 0, 1))

def inject_declarmima_seeds(valid_concepts: list, concept_to_id: dict, 
                           proposal_embedding: np.ndarray, embed_model,
                           correlation_threshold: float = 0.65) -> tuple:
    """Inject DECLARMIMA proposal concepts weighted by semantic correlation"""
    updated_concepts = valid_concepts.copy()
    updated_mapping = concept_to_id.copy()
    
    proposal_concepts = extract_declarmima_concepts(DECLARMIMA_PROPOSAL_TEXT, embed_model)
    
    if proposal_concepts:
        proposal_concept_embeddings = embed_model.encode(proposal_concepts, show_progress_bar=False)
        
        for i, prop_concept in enumerate(proposal_concepts):
            if prop_concept not in updated_mapping:
                corr = compute_proposal_correlation(
                    prop_concept, proposal_embedding, proposal_concept_embeddings[i]
                )
                if corr >= correlation_threshold:
                    updated_mapping[prop_concept] = len(updated_mapping)
                    updated_concepts.append(prop_concept)
    
    return updated_concepts, updated_mapping

# ==========================================
# SEMANTIC CLUSTERING & CONCEPT ABSTRACTION
# ==========================================
def cluster_similar_concepts(valid_concepts, embed_model, similarity_threshold=0.78):
    """Merge semantically similar concepts to boost effective frequency"""
    if len(valid_concepts) < 3:
        return valid_concepts, {c: c for c in valid_concepts}
    
    try:
        embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=32)
        sim_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - sim_matrix
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=1 - similarity_threshold, linkage='average'
        ).fit(embeddings)
        
        concept_to_cluster = {}
        cluster_members = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            concept = valid_concepts[idx]
            cluster_members[label].append(concept)
            concept_to_cluster[concept] = label
        
        cluster_representatives = {}
        for label, members in cluster_members.items():
            representative = min(members, key=lambda x: (len(x), -x.count(' ')))
            cluster_representatives[label] = representative
        
        final_mapping = {c: cluster_representatives[label] for c, label in concept_to_cluster.items()}
        return list(cluster_representatives.values()), final_mapping
        
    except Exception as e:
        st.warning(f"⚠️ Semantic clustering skipped: {e}")
        return valid_concepts, {c: c for c in valid_concepts}

def abstract_concepts_to_categories(concepts, category_mapping=CATEGORY_MAPPING):
    """Map specific concepts to broader categories for graph density"""
    concept_to_abstract = {}
    for concept in concepts:
        matched = False
        for pattern, category in category_mapping.items():
            if re.search(pattern, concept, re.I):
                concept_to_abstract[concept] = category
                matched = True
                break
        if not matched:
            concept_to_abstract[concept] = concept
    return concept_to_abstract

def inject_domain_seeds(valid_concepts, concept_to_id, seed_concepts=DOMAIN_SEED_CONCEPTS):
    """Add domain seed concepts even if not frequently observed"""
    all_seeds = [seed for category in seed_concepts.values() for seed in category]
    updated_concepts = valid_concepts.copy()
    updated_mapping = concept_to_id.copy()
    
    for seed in all_seeds:
        if seed not in updated_mapping:
            updated_mapping[seed] = len(updated_mapping)
            updated_concepts.append(seed)
    
    return updated_concepts, updated_mapping

# ==========================================
# STEP 1-2: CONCEPT EXTRACTION & METRICS
# ==========================================
def extract_concepts_from_abstracts(abstracts, tokenizer, model, backend_type: str):
    """Extract microstructure-relevant concepts using selected LLM backend"""
    prompt_template = """Extract exactly the core scientific concepts (2+ words) from this abstract about laser processing or alloy microstructure.
Rules:
- Output ONLY a JSON list of strings.
- Use nominalized form (e.g., 'grain refinement' not 'refines grains').
- Include: alloy compositions (e.g., 'AlSi10Mg'), laser parameters ('laser power'), microstructure features ('columnar grains'), properties ('microhardness').
- Standardize: chemical formulas, units (J/mm³, mm/s), phase names.
- Exclude: generic terms like 'study', 'results', 'method'.

Abstract: {text}
Concepts:"""
    
    all_concepts = []
    all_metrics = []
    
    for text in abstracts:
        metrics = {}
        grain_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:μm|micron|um|nm)\s*(?:grain|average|size|diameter)?', text, re.I)
        if grain_matches:
            metrics['grain_size_um'] = [float(m) for m in grain_matches]
        mech_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:HV|GPa|MPa|ksi)\s*(?:hardness|strength|yield|tensile|ultimate)?', text, re.I)
        if mech_matches:
            metrics['mechanical_property'] = [float(m) for m in mech_matches]
        energy_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:J/mm³|J mm-3|J mm⁻³|J/mm\^3)', text, re.I)
        if energy_matches:
            metrics['energy_density_j_mm3'] = [float(m) for m in energy_matches]
        defect_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:porosity|void|crack)', text, re.I)
        if defect_matches:
            metrics['defect_fraction_pct'] = [float(m) for m in defect_matches]
        all_metrics.append(metrics)
        
        prompt = prompt_template.format(text=text)
        
        if backend_type == "ollama":
            # Use Ollama for concept extraction
            try:
                client = ollama.Client(host=st.session_state.get('ollama_host', 'http://localhost:11434'))
                response = client.chat(
                    model=model,  # model_tag from _load_ollama_model
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.2, "num_predict": 150}
                )
                response_text = response.get('message', {}).get('content', '') if isinstance(response, dict) else getattr(response, 'message', {}).get('content', '')
            except Exception as e:
                st.warning(f"⚠️ Ollama extraction failed: {e}, using fallback")
                response_text = _fallback_concept_extraction_text(prompt)
        else:
            # Use HF Transformers for concept extraction
            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids, max_new_tokens=150, temperature=0.2,
                        do_sample=True, pad_token_id=tokenizer.eos_token_id
                    )
                response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            except Exception as e:
                st.warning(f"⚠️ HF extraction failed: {e}, using fallback")
                response_text = _fallback_concept_extraction_text(prompt)
        
        concepts = []
        try:
            parsed = json.loads(response_text.replace("'", '"').strip())
            if isinstance(parsed, list):
                concepts = [c.strip().lower().rstrip('.') for c in parsed if isinstance(c, str) and len(c.strip()) > 3]
        except (json.JSONDecodeError, TypeError):
            concepts = _fallback_concept_extraction(text)
        
        normalized = []
        for c in concepts:
            if any(elem in c.lower() for elem in ['al', 'ti', 'ni', 'cr', 'fe', 'co', 'mo', 'nb', 'cu', 'sn', 'ag']):
                c = normalize_alloy_composition(c)
            elif any(lp in c.lower() for lp in ['laser', 'scan', 'power', 'speed', 'melt', 'pool', 'energy']):
                c = normalize_laser_term(c)
            if is_valid_microstructure_concept(c):
                normalized.append(c)
        all_concepts.append(normalized)
    return all_concepts, all_metrics

def _fallback_concept_extraction_text(prompt: str) -> str:
    """Fallback text response for concept extraction"""
    return '["laser processing", "microstructure evolution", "alloy composition", "mechanical properties"]'

def _fallback_concept_extraction(text: str) -> list:
    """Regex fallback for concept extraction when LLM parsing fails"""
    patterns = [
        r'\b(?:[A-Z][a-z]+(?:\d+(?:\.\d+)?)?[\s\-]?){2,3}(?:phase|grain|microstructure|strength|hardness)',
        r'\b(?:laser|powder|bed|fusion|selective|direct)\s+(?:power|speed|scanning|melting|parameters|energy)',
        r'\b(?:columnar|equiaxed|fine|coarse|nanoscale|bimodal)\s+(?:grain|structure|region|zone)',
        r'\b(?:martensite|austenite|ferrite|eutectic|peritectic|precipitate)\s+(?:formation|phase|fraction)',
        r'\b(?:microhardness|nanohardness|tensile|yield|ductility|elongation)\s+(?:improvement|strength|property)',
    ]
    concepts = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.I)
        concepts.extend([m.lower().strip() for m in matches if len(m.split()) >= 2])
    return list(set(concepts))

def normalize_and_filter_concepts(all_concepts, embed_model=None, config=None, proposal_embedding=None):
    """Adaptive concept filtering with semantic clustering, seed injection, and DECLARMIMA correlation"""
    if config is None:
        config = get_adaptive_config(25)
    
    concept_counts = defaultdict(int)
    concept_abstract_map = defaultdict(list)
    
    for doc_idx, concepts in enumerate(all_concepts):
        seen_in_doc = set()
        for c in concepts:
            if c not in seen_in_doc and is_valid_microstructure_concept(c):
                concept_counts[c] += 1
                concept_abstract_map[c].append(doc_idx)
                seen_in_doc.add(c)
    
    min_freq = config.get("MIN_CONCEPT_FREQ", 2)
    min_words = config.get("MIN_CONCEPT_LENGTH_WORDS", 2)
    valid_concepts = [c for c, cnt in concept_counts.items() 
                      if cnt >= min_freq and len(c.split()) >= min_words]
    
    if config.get("USE_DECLARMIMA_SEEDS", True) and proposal_embedding is not None and embed_model:
        valid_concepts, concept_to_id = inject_declarmima_seeds(
            valid_concepts, {c: i for i, c in enumerate(valid_concepts)},
            proposal_embedding, embed_model, correlation_threshold=0.65
        )
        for seed in [s for cat in DOMAIN_SEED_CONCEPTS.values() for s in cat]:
            if seed not in concept_counts:
                concept_counts[seed] = 1
                concept_abstract_map[seed] = []
    elif config.get("INJECT_DOMAIN_SEEDS", True) and len(valid_concepts) < 15:
        valid_concepts, concept_to_id = inject_domain_seeds(
            valid_concepts, {c: i for i, c in enumerate(valid_concepts)}
        )
        for seed in [s for cat in DOMAIN_SEED_CONCEPTS.values() for s in cat]:
            if seed not in concept_counts:
                concept_counts[seed] = 1
                concept_abstract_map[seed] = []
    
    if config.get("USE_SEMANTIC_CLUSTERING", True) and embed_model and len(valid_concepts) >= 5:
        clustered_concepts, concept_to_cluster = cluster_similar_concepts(
            valid_concepts, embed_model, similarity_threshold=config.get("CLUSTER_SIMILARITY", 0.75)
        )
        new_abstract_map = defaultdict(list)
        for orig_concept, docs in concept_abstract_map.items():
            clustered = concept_to_cluster.get(orig_concept, orig_concept)
            if clustered in clustered_concepts:
                new_abstract_map[clustered].extend(docs)
        concept_abstract_map = new_abstract_map
        valid_concepts = clustered_concepts
    
    valid_concepts = list(set(valid_concepts))
    concept_to_id = {c: i for i, c in enumerate(valid_concepts)}
    id_to_concept = {i: c for i, c in enumerate(valid_concepts)}
    
    return valid_concepts, concept_to_id, id_to_concept, concept_abstract_map

# ==========================================
# STEP 3: HYBRID CONCEPT GRAPH CONSTRUCTION
# ==========================================
def build_semantic_only_graph(valid_concepts, embed_model, similarity_threshold=0.75):
    """Fallback: create graph purely from embedding similarity"""
    nx_graph = nx.Graph()
    for c in valid_concepts:
        nx_graph.add_node(c)
    if len(valid_concepts) < 2:
        return nx_graph
    try:
        embeddings = embed_model.encode(valid_concepts, show_progress_bar=False)
        sim_matrix = cosine_similarity(embeddings)
        for i in range(len(valid_concepts)):
            for j in range(i+1, len(valid_concepts)):
                if sim_matrix[i][j] > similarity_threshold:
                    nx_graph.add_edge(
                        valid_concepts[i], valid_concepts[j], 
                        weight=sim_matrix[i][j], edge_type='semantic',
                        cooccurrence=0, semantic=sim_matrix[i][j]
                    )
        if not nx.is_connected(nx_graph) and len(valid_concepts) > 3:
            components = list(nx.connected_components(nx_graph))
            for i in range(len(components)-1):
                best_sim, best_pair = 0, None
                for c1 in components[i]:
                    idx1 = valid_concepts.index(c1)
                    for c2 in components[i+1]:
                        idx2 = valid_concepts.index(c2)
                        if sim_matrix[idx1][idx2] > best_sim:
                            best_sim = sim_matrix[idx1][idx2]
                            best_pair = (c1, c2)
                if best_pair:
                    nx_graph.add_edge(*best_pair, weight=best_sim, edge_type='bridge',
                                     cooccurrence=0, semantic=best_sim)
    except Exception as e:
        st.warning(f"⚠️ Semantic graph construction issue: {e}")
        for i in range(len(valid_concepts)-1):
            nx_graph.add_edge(valid_concepts[i], valid_concepts[i+1], weight=1.0)
    return nx_graph

def build_hybrid_graph(all_concepts, valid_concepts, concept_to_id, embed_model=None, config=None, proposal_embedding=None):
    """Hybrid graph: combine co-occurrence with embedding similarity and DECLARMIMA correlation"""
    if config is None:
        config = get_adaptive_config(len(all_concepts))
    
    nx_graph = nx.Graph()
    for c in valid_concepts:
        nx_graph.add_node(c, frequency=0)
    
    # Step 1: Add co-occurrence edges
    for concepts in all_concepts:
        valid_in_doc = [c for c in concepts if c in concept_to_id]
        for i in range(len(valid_in_doc)):
            for j in range(i+1, len(valid_in_doc)):
                u, v = valid_in_doc[i], valid_in_doc[j]
                if nx_graph.has_edge(u, v):
                    nx_graph[u][v]['weight'] += 1
                    nx_graph[u][v]['cooccurrence'] += 1
                else:
                    nx_graph.add_edge(u, v, weight=1, cooccurrence=1, semantic=0, edge_type='cooccurrence')
                nx_graph.nodes[u]['frequency'] = nx_graph.nodes[u].get('frequency', 0) + 1
                nx_graph.nodes[v]['frequency'] = nx_graph.nodes[v].get('frequency', 0) + 1
    
    # Step 2: Add semantic similarity edges
    if config.get("USE_SEMANTIC_EDGES", True) and embed_model and len(valid_concepts) >= 5:
        try:
            embeddings = embed_model.encode(valid_concepts, show_progress_bar=False)
            sim_matrix = cosine_similarity(embeddings)
            sim_thresh = config.get("SIMILARITY_THRESHOLD", 0.75)
            for i, c1 in enumerate(valid_concepts):
                for j, c2 in enumerate(valid_concepts[i+1:], start=i+1):
                    if c1 == c2 or nx_graph.has_edge(c1, c2):
                        continue
                    sim = sim_matrix[i][j]
                    if sim > sim_thresh and (nx_graph.degree(c1) < 2 or nx_graph.degree(c2) < 2):
                        semantic_weight = sim * 2
                        nx_graph.add_edge(c1, c2, weight=semantic_weight, 
                                         cooccurrence=0, semantic=sim, edge_type='semantic')
        except Exception as e:
            st.warning(f"⚠️ Semantic edge addition skipped: {e}")
    
    # DECLARMIMA: Boost edges for concepts highly correlated with proposal
    if config.get("CORRELATE_WITH_PROPOSAL", True) and proposal_embedding is not None and embed_model:
        concept_embeddings = embed_model.encode(valid_concepts, show_progress_bar=False)
        for i, c1 in enumerate(valid_concepts):
            corr1 = compute_proposal_correlation(c1, proposal_embedding, concept_embeddings[i])
            for j, c2 in enumerate(valid_concepts[i+1:], start=i+1):
                if c1 == c2:
                    continue
                corr2 = compute_proposal_correlation(c2, proposal_embedding, concept_embeddings[j])
                if corr1 > 0.7 and corr2 > 0.7:
                    boost = 1.5 * (corr1 + corr2) / 2
                    if nx_graph.has_edge(c1, c2):
                        nx_graph[c1][c2]['weight'] *= boost
                        nx_graph[c1][c2]['declarmina_boost'] = boost
                    else:
                        nx_graph.add_edge(c1, c2, weight=boost, cooccurrence=0, 
                                         semantic=0.8, edge_type='declarmina_aligned')
    
    # Step 3: Combine weights
    cooc_weight = config.get("COOCCURRENCE_WEIGHT", 0.6)
    sem_weight = config.get("SEMANTIC_WEIGHT", 0.4)
    for u, v, data in nx_graph.edges(data=True):
        cooc = data.get('cooccurrence', 0)
        sem = data.get('semantic', 0)
        data['weight'] = cooc_weight * cooc + sem_weight * sem
    
    return nx_graph

def build_concept_graph(all_concepts, concept_to_id, embed_model=None, config=None, proposal_embedding=None):
    """Main graph builder with fallback strategies"""
    if config is None:
        config = get_adaptive_config(len(all_concepts))
    valid_concepts = list(concept_to_id.keys())
    if len(valid_concepts) < 8 and config.get("USE_SEMANTIC_EDGES", True):
        return build_semantic_only_graph(valid_concepts, embed_model, 
                                        similarity_threshold=config.get("SIMILARITY_THRESHOLD", 0.75))
    return build_hybrid_graph(all_concepts, valid_concepts, concept_to_id, embed_model, config, proposal_embedding)

def sample_edges_for_training(nx_graph, d_prev_dict, valid_concepts, concept_to_id, config=None):
    """Sample positive and negative edges for contrastive training"""
    if config is None:
        config = get_adaptive_config(len(valid_concepts))
    pos_pairs = [(concept_to_id[u], concept_to_id[v]) for u, v in nx_graph.edges()]
    neg_pairs = []
    n_nodes = len(valid_concepts)
    if n_nodes < 3:
        return pos_pairs, neg_pairs
    target_negs = min(len(pos_pairs) * 2 if pos_pairs else 10, 2000)
    attempts = 0
    neg_focus = config.get("NEG_DPREV_FOCUS", 3)
    while len(neg_pairs) < target_negs and attempts < 15000:
        u_idx, v_idx = np.random.choice(n_nodes, 2, replace=False)
        u_concept, v_concept = valid_concepts[u_idx], valid_concepts[v_idx]
        if nx_graph.has_edge(u_concept, v_concept):
            attempts += 1
            continue
        try:
            dist = d_prev_dict[u_concept][v_concept]
            if dist == neg_focus:
                neg_pairs.append((u_idx, v_idx))
            elif dist == 2 and np.random.rand() < 0.3:
                neg_pairs.append((u_idx, v_idx))
        except KeyError:
            if np.random.rand() < 0.1:
                neg_pairs.append((u_idx, v_idx))
        attempts += 1
    while len(neg_pairs) < target_negs:
        u_idx, v_idx = np.random.choice(n_nodes, 2, replace=False)
        pair = (u_idx, v_idx)
        if pair not in neg_pairs and (v_idx, u_idx) not in neg_pairs:
            if not nx_graph.has_edge(valid_concepts[u_idx], valid_concepts[v_idx]):
                neg_pairs.append(pair)
    return pos_pairs, neg_pairs

# ==========================================
# STEP 4: SEMANTIC NODE EMBEDDINGS
# ==========================================
def generate_embeddings(valid_concepts, embed_model):
    """Generate sentence embeddings for concept nodes"""
    if not valid_concepts:
        return torch.zeros((0, 384), dtype=torch.float32).to(DEVICE)
    embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=32)
    return torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)

def get_embedding_dimension(embed_model) -> int:
    """Safely detect the embedding dimension from the model"""
    try:
        dummy = ["test"]
        emb = embed_model.encode(dummy, show_progress_bar=False)
        return emb.shape[1]
    except:
        return 384

# ==========================================
# STEP 5: PURE PYTORCH SPARSE GRAPHSAGE
# ==========================================
class SparseGraphSAGE(nn.Module):
    """Memory-efficient GraphSAGE using PyTorch sparse tensors"""
    def __init__(self, in_dim: int, hidden_dim: int = GNN_HIDDEN_DIM):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, adj_indices, adj_values, num_nodes, h, pos_u, pos_v, neg_u, neg_v):
        A = sparse.FloatTensor(adj_indices, adj_values, torch.Size([num_nodes, num_nodes])).to(h.device)
        deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1)
        deg_inv = 1.0 / deg
        h1 = F.relu(self.lin1(torch.sparse.mm(A, h) * deg_inv.unsqueeze(1)))
        h2 = self.lin2(torch.sparse.mm(A, h1) * deg_inv.unsqueeze(1))
        pos_scores = self.decoder(torch.cat([h2[pos_u], h2[pos_v]], dim=1)).squeeze(1)
        neg_scores = self.decoder(torch.cat([h2[neg_u], h2[neg_v]], dim=1)).squeeze(1)
        return pos_scores, neg_scores, h2

def train_gnn(node_features, nx_graph, concept_to_id, pos_pairs, neg_pairs, progress_callback=None):
    """Train GraphSAGE with contrastive edge prediction loss"""
    num_nodes = len(concept_to_id)
    in_dim = node_features.shape[1] if node_features.numel() > 0 else 384
    if node_features.numel() > 0:
        expected_shape = (num_nodes, in_dim)
        if node_features.shape != expected_shape:
            raise ValueError(f"Node features shape mismatch: expected {expected_shape}, got {node_features.shape}")
    if not pos_pairs:
        nodes = list(concept_to_id.values())
        if len(nodes) >= 2:
            pos_pairs = [(nodes[0], nodes[1])]
            neg_pairs = [(nodes[0], nodes[-1])] if len(nodes) > 2 else []
        else:
            raise ValueError("Cannot train GNN with fewer than 2 concepts")
    unique_edges = {(min(u, v), max(u, v)) for u, v in pos_pairs}
    src_adj = torch.tensor([u for u, v in unique_edges], dtype=torch.long)
    dst_adj = torch.tensor([v for u, v in unique_edges], dtype=torch.long)
    adj_indices = torch.stack([src_adj, dst_adj], dim=0)
    adj_values = torch.ones(adj_indices.shape[1], dtype=torch.float32)
    pos_u = torch.tensor([p[0] for p in pos_pairs], dtype=torch.long, device=DEVICE)
    pos_v = torch.tensor([p[1] for p in pos_pairs], dtype=torch.long, device=DEVICE)
    neg_u = torch.tensor([n[0] for n in neg_pairs], dtype=torch.long, device=DEVICE) if neg_pairs else torch.tensor([], dtype=torch.long, device=DEVICE)
    neg_v = torch.tensor([n[1] for n in neg_pairs], dtype=torch.long, device=DEVICE) if neg_pairs else torch.tensor([], dtype=torch.long, device=DEVICE)
    model = SparseGraphSAGE(in_dim=in_dim, hidden_dim=GNN_HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(TRAIN_EPOCHS):
        model.train()
        optimizer.zero_grad()
        if len(neg_pairs) == 0:
            pos_out, _, _ = model(adj_indices, adj_values, num_nodes, node_features, pos_u, pos_v, pos_u[:1], pos_v[:1])
            loss = criterion(pos_out, torch.ones_like(pos_out)) * 0.5
        else:
            pos_out, neg_out, _ = model(adj_indices, adj_values, num_nodes, node_features, pos_u, pos_v, neg_u, neg_v)
            pos_loss = criterion(pos_out, torch.ones_like(pos_out))
            neg_loss = criterion(neg_out, torch.zeros_like(neg_out))
            loss = 0.5 * (pos_loss + neg_loss)
        loss.backward()
        optimizer.step()
        if progress_callback and epoch % 10 == 0:
            progress_callback(epoch, loss.item())
    model.eval()
    with torch.no_grad():
        _, _, final_embeddings = model(
            adj_indices, adj_values, num_nodes,
            node_features, pos_u[:1], pos_v[:1], 
            neg_u[:1] if len(neg_pairs) > 0 else pos_u[:1], 
            neg_v[:1] if len(neg_pairs) > 0 else pos_v[:1]
        )
    return model, final_embeddings.cpu(), adj_indices, adj_values

# ==========================================
# STEP 6: MICROSTRUCTURE QUANTIFICATION & SCORING
# ==========================================
def compute_microstructure_quantification(valid_concepts, concept_abstract_map, all_metrics, nx_graph):
    """Map concepts to representative microstructure property values"""
    concept_properties = {}
    for concept in valid_concepts:
        doc_indices = concept_abstract_map.get(concept, [])
        values = []
        for idx in doc_indices:
            if idx < len(all_metrics):
                metrics = all_metrics[idx]
                for metric_values in metrics.values():
                    values.extend(metric_values)
        concept_properties[concept] = np.median(values) if values else 0.0
    X_feat, y_target = [], []
    for u, v in nx_graph.edges():
        pu, pv = concept_properties.get(u, 0), concept_properties.get(v, 0)
        w = nx_graph[u][v].get('weight', 1)
        X_feat.append([pu, pv, w])
        y_target.append(max(pu, pv) * 1.08 if max(pu, pv) > 0 else 0)
    ridge = None
    if len(X_feat) > 5:
        ridge = Ridge(alpha=1.0).fit(np.array(X_feat), np.array(y_target))
    return concept_properties, ridge

def compute_research_direction_scores(
    model, node_features, final_emb, nx_graph, valid_concepts, concept_properties, 
    ridge, embed_model, d_prev_dict, adj_indices, adj_values,
    n_samples=3000
):
    """Score novel concept pairs for promising microstructure research directions"""
    n_concepts = len(valid_concepts)
    if n_concepts < 3:
        return pd.DataFrame()
    u_ids = np.random.randint(n_concepts, size=min(n_samples, n_concepts * 10))
    v_ids = np.random.randint(n_concepts, size=min(n_samples, n_concepts * 10))
    candidate_pairs = []
    for u_idx, v_idx in zip(u_ids, v_ids):
        if u_idx == v_idx: 
            continue
        u_c, v_c = valid_concepts[u_idx], valid_concepts[v_idx]
        if nx_graph.has_edge(u_c, v_c): 
            continue
        try:
            d_prev = d_prev_dict[u_c][v_c]
        except KeyError:
            d_prev = 4
        if d_prev < 2:
            continue
        candidate_pairs.append((u_idx, v_idx, u_c, v_c, d_prev))
    if not candidate_pairs:
        return pd.DataFrame()
    u_tensor = torch.tensor([p[0] for p in candidate_pairs], dtype=torch.long, device=DEVICE)
    v_tensor = torch.tensor([p[1] for p in candidate_pairs], dtype=torch.long, device=DEVICE)
    model.eval()
    with torch.no_grad():
        _, _, h2 = model(
            adj_indices, adj_values, n_concepts,
            node_features.to(DEVICE), u_tensor, v_tensor, u_tensor, v_tensor
        )
        pair_features = torch.cat([h2[u_tensor], h2[v_tensor]], dim=1)
        gnn_logits = model.decoder(pair_features).squeeze(1)
        gnn_scores = torch.sigmoid(gnn_logits).cpu().numpy()
    emb_np = embed_model.encode(valid_concepts, show_progress_bar=False)
    cos_sims = np.sum(emb_np[u_tensor.cpu().numpy()] * emb_np[v_tensor.cpu().numpy()], axis=1)
    results = []
    for i, (u_idx, v_idx, u_c, v_c, d_prev) in enumerate(candidate_pairs):
        p_u = concept_properties.get(u_c, 0)
        p_v = concept_properties.get(v_c, 0)
        expected_improvement = 0
        if ridge is not None and (p_u > 0 or p_v > 0):
            try:
                expected_improvement = float(ridge.predict([[p_u, p_v, 1.0]])[0])
            except:
                expected_improvement = max(p_u, p_v) * 1.05
        semantic_novelty = 1.0 - cos_sims[i]
        feasibility = np.exp(-0.5 * semantic_novelty) * (1.0 if (p_u > 0 or p_v > 0) else 0.6)
        alpha = {'gnn': 0.4, 'novelty': 0.3, 'gain': 0.2, 'feas': -0.1}
        norm_gain = np.clip((expected_improvement - 50) / 200, 0, 1)
        D_uv = (alpha['gnn'] * gnn_scores[i] + alpha['novelty'] * semantic_novelty + 
                alpha['gain'] * norm_gain + alpha['feas'] * (1.0 - feasibility))
        results.append({
            'concept_u': u_c, 'concept_v': v_c, 'graph_distance': d_prev,
            'gnn_affinity': float(gnn_scores[i]), 'semantic_novelty': float(semantic_novelty),
            'expected_property_gain': expected_improvement, 'feasibility_score': float(feasibility),
            'composite_score': float(D_uv)
        })
    df = pd.DataFrame(results).sort_values('composite_score', ascending=False)
    return df.head(min(50, len(df)))

# ==========================================
# STEP 7: LLM CURATION OF RESEARCH DIRECTIONS
# ==========================================
def generate_research_directions(top_pairs_df, tokenizer, model, backend_type: str, 
                                max_hypotheses=10, proposal_context=""):
    """Generate LLM-curated research hypotheses aligned with DECLARMIMA goals"""
    
    prompt_template = """You are a materials science strategist for the DECLARMIMA project: "Deciphering laser-microstructure interaction in multicomponent alloys".

Project Goals: {proposal_context}

For the novel concept combination: "{u}" + "{v}"
Associated property context: ~{prop:.1f} (e.g., HV, μm, MPa)
Feasibility estimate: {feas:.2f}/1.0

Write exactly 3 concise, technically precise sentences:
1. Scientific novelty: Why this combination advances DECLARMIMA goals (physics-informed digital twins, multiscale modeling, laser-matter mechanisms).
2. Target outcome: Predicted microstructure/property improvement and key trade-off relevant to additive manufacturing.
3. Validation step: One concrete experimental or computational method (e.g., phase field simulation, EBSD, in-situ XRD, ML uncertainty quantification).

Avoid generic statements. Focus on laser-matter interaction, solidification physics, or phase transformation mechanisms."""

    results = []
    total_rows = min(len(top_pairs_df), max_hypotheses)
    if total_rows == 0:
        return pd.DataFrame()
    
    proposal_summary = "physics-informed digital twins for laser-processed multicomponent alloys, multiscale computational modeling, integrated experiment-computation framework, process-structure-property relationships, uncertainty quantification"
    
    progress_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    for idx in range(total_rows):
        try:
            row = top_pairs_df.iloc[idx]
            progress = (idx + 1) / total_rows
            progress_bar.progress(progress)
            progress_placeholder.write(f"🔄 Generating hypothesis {idx+1}/{total_rows}: {row['concept_u']} + {row['concept_v']}")
            
            prompt = prompt_template.format(
                proposal_context=proposal_summary,
                u=row['concept_u'].title(), v=row['concept_v'].title(), 
                prop=float(row['expected_property_gain']), feas=float(row['feasibility_score'])
            )
            
            if backend_type == "ollama":
                try:
                    client = ollama.Client(host=st.session_state.get('ollama_host', 'http://localhost:11434'))
                    response = client.chat(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        options={"temperature": 0.25, "num_predict": 180}
                    )
                    response_text = response.get('message', {}).get('content', '') if isinstance(response, dict) else getattr(response, 'message', {}).get('content', '')
                except Exception as e:
                    st.warning(f"⚠️ Ollama generation failed: {e}")
                    continue
            else:
                try:
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300).to(DEVICE)
                except Exception as e:
                    st.warning(f"⚠️ Tokenization error for row {idx+1}: {e}")
                    continue
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids, max_new_tokens=180, temperature=0.25, do_sample=True,
                        pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.1,
                        eos_token_id=tokenizer.eos_token_id, use_cache=True,
                    )
                response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            results.append({
                'Concept Pair': f"{row['concept_u']} + {row['concept_v']}",
                'Composite Score': f"{row['composite_score']:.3f}",
                'Expected Gain': f"{row['expected_property_gain']:.1f}",
                'Feasibility': f"{row['feasibility_score']:.2f}",
                'Research Hypothesis': response_text.strip(),
                'DECLARMIMA Alignment': 'High' if row['composite_score'] > 0.7 else 'Medium'
            })
            
            if backend_type != "ollama":
                del inputs, outputs
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
        except torch.cuda.OutOfMemoryError as e:
            st.error(f"❌ CUDA Out of Memory at hypothesis {idx+1}: {e}")
            st.info("💡 Try reducing max_hypotheses or switching to CPU mode")
            break
        except Exception as e:
            st.warning(f"⚠️ Skipping hypothesis {idx+1}: {type(e).__name__}: {e}")
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            continue
    
    progress_placeholder.empty()
    progress_bar.empty()
    return pd.DataFrame(results) if results else pd.DataFrame()

# ==========================================
# ENHANCED QUANTITATIVE GRAPH METRICS
# ==========================================
def compute_graph_metrics(G: nx.Graph) -> dict:
    """Calculate topological metrics for research field analysis."""
    if G.number_of_nodes() == 0:
        return {}
    metrics = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": np.mean([d for _, d in G.degree()]),
        "clustering": nx.average_clustering(G) if G.number_of_nodes() > 2 else 0,
        "connected_components": nx.number_connected_components(G),
    }
    # Betweenness centrality (expensive, sample if large graph)
    if G.number_of_nodes() <= 200:
        bc = nx.betweenness_centrality(G, normalized=True)
        top_bridges = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:5]
        metrics["top_bridges"] = top_bridges
    else:
        metrics["top_bridges"] = []
    return metrics

def display_metric_dashboard(metrics: dict):
    """Show key metrics in Streamlit columns and a bridge table."""
    if not metrics:
        st.warning("No graph metrics available.")
        return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nodes", metrics["nodes"])
    col2.metric("Edges", metrics["edges"])
    col3.metric("Graph Density", f"{metrics['density']:.3f}")
    col4.metric("Avg. Degree", f"{metrics['avg_degree']:.2f}")
    col5, col6 = st.columns(2)
    col5.metric("Clustering Coef.", f"{metrics['clustering']:.3f}")
    col6.metric("Components", metrics["connected_components"])
    
    if metrics["top_bridges"]:
        st.markdown("**🌉 Top 5 Bridge Concepts (High Betweenness)**")
        bridge_df = pd.DataFrame(metrics["top_bridges"], columns=["Concept", "Bridge Score"])
        st.dataframe(bridge_df, use_container_width=True)

# ==========================================
# CATEGORICAL SUNBURST CHART
# ==========================================
def build_category_hierarchy(valid_concepts: list, concept_abstract_map: dict):
    """Assign each concept to a parent category based on regex mapping."""
    hierarchy = defaultdict(lambda: {"children": [], "count": 0})
    for concept in valid_concepts:
        matched = False
        for pattern, category in CATEGORY_MAPPING.items():
            if re.search(pattern, concept, re.I):
                parent = category
                matched = True
                break
        if not matched:
            # fallback – classify by first word or domain
            if any(kw in concept.lower() for kw in DOMAIN_KEYWORDS):
                parent = "other_domain"
            else:
                parent = "misc"
        freq = len(concept_abstract_map.get(concept, []))
        hierarchy[parent]["children"].append((concept, freq))
        hierarchy[parent]["count"] += freq
    # Build sunburst data
    labels = []
    parents = []
    values = []
    for parent, data in hierarchy.items():
        labels.append(parent)
        parents.append("")
        values.append(data["count"])
        for child, cnt in data["children"]:
            labels.append(child)
            parents.append(parent)
            values.append(cnt)
    return labels, parents, values

def render_sunburst_chart(labels, parents, values):
    """Interactive Plotly sunburst chart for research domain distribution."""
    if not labels:
        st.info("Not enough categories for sunburst chart.")
        return
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(colorscale="Viridis", line=dict(width=0.5, color="white")),
        textinfo="label+percent entry",
        insidetextorientation="radial"
    ))
    fig.update_layout(
        title="<b>DECLARMIMA Research Domain Hierarchy</b><br><i>Size = concept frequency</i>",
        font=dict(size=12, family="Arial"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=800, height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    # Export as SVG/PNG via download button
    try:
        svg_bytes = fig.to_image(format="svg", scale=2)
        st.download_button(
            "📸 Download Sunburst as SVG",
            data=svg_bytes,
            file_name="sunburst.svg",
            mime="image/svg+xml",
            key="sunburst_svg"
        )
    except:
        st.info("Install kaleido for SVG export: pip install kaleido")

# ==========================================
# ENHANCED PYVIS GRAPH WITH CATEGORY COLORS
# ==========================================
def get_category_color(concept: str) -> str:
    """Assign vibrant hex colors based on concept category."""
    concept_lower = concept.lower()
    if any(a in concept_lower for a in ['al', 'ti', 'ni', 'cr', 'fe', 'co', 'mo', 'nb', 'cu', 'w', 'mn']):
        return "#E91E63"  # pink – alloys
    elif any(l in concept_lower for l in ['laser', 'scan', 'power', 'melt', 'energy']):
        return "#3F51B5"  # indigo – laser parameters
    elif any(m in concept_lower for m in ['grain', 'phase', 'hardness', 'strength', 'texture']):
        return "#FF9800"  # orange – microstructure/properties
    elif any(p in concept_lower for p in ['porosity', 'crack', 'defect', 'void']):
        return "#F44336"  # red – defects
    elif any(d in concept_lower for d in ['digital', 'twin', 'machine', 'learning', 'neural']):
        return "#9C27B0"  # purple – computational
    else:
        return "#009688"  # teal – other
#
def render_graph_pyvis_custom(nx_graph, concept_abstract_map, physics_enabled=True, 
                               min_node_size=12, max_node_size=50):
    """Render interactive PyVis graph with user‑adjustable physics and styling.
    
    FIXED: Convert numpy scalars to native Python types for JSON serialization.
    """
    # Sample large graphs for performance
    if len(nx_graph.nodes()) > 100:
        degrees = dict(nx_graph.degree())
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:100]
        nx_graph = nx_graph.subgraph(top_nodes).copy()
    
    # Initialize PyVis network with white theme
    net = Network(
        height="700px", width="100%", bgcolor="#ffffff", font_color="#000000",
        select_menu=True, notebook=False, cdn_resources='in_line'
    )
    
    # Configure physics simulation (Barnes-Hut approximation)
    if physics_enabled:
        net.barnes_hut(
            gravity=-2000,        # Repulsive force between nodes
            spring_length=150,    # Ideal edge length
            spring_strength=0.05, # Edge stiffness
            damping=0.09,         # Oscillation damping
            overlap=0.5           # Node overlap prevention
        )
    else:
        # Disable physics for static layout
        net.set_options("""
        var options = {
            physics: { enabled: false },
            layout: { improvedLayout: false }
        }
        """)
    
    # Add nodes with category-based colors and frequency-scaled sizes
    for node in nx_graph.nodes():
        freq = len(concept_abstract_map.get(node, []))
        
        # ✅ FIX: Convert numpy scalar to native int for JSON serialization
        size = int(np.clip(min_node_size + freq * 2, min_node_size, max_node_size))
        
        color = get_category_color(node)
        degree = int(nx_graph.degree(node))  # ✅ Also convert degree to int
        
        net.add_node(
            node, 
            label=node, 
            size=size, 
            color=color,
            font={'color': '#000000', 'size': 14},
            title=f"{node}\nDegree: {degree}\nFrequency: {freq}"
        )
    
    # Add edges with weight-scaled styling
    color_map = {
        'cooccurrence': "#4CAF50",    # Green: empirical co-occurrence
        'semantic': "#2196F3",         # Blue: embedding similarity
        'bridge': "#FFC107",           # Amber: connects components
        'declarmina_aligned': "#E91E63" # Pink: DECLARMIMA project alignment
    }
    
    for u, v in nx_graph.edges():
        w = nx_graph[u][v].get('weight', 1)
        edge_type = nx_graph[u][v].get('edge_type', 'unknown')
        color = color_map.get(edge_type, "#607D8B")
        
        # ✅ FIX: Convert all numpy scalars to native Python float/int
        edge_value = float(np.clip(w, 0.5, 5))
        edge_width = float(np.clip(w * 0.8, 1, 4))
        
        net.add_edge(
            u, v, 
            value=edge_value, 
            width=edge_width,
            color=color, 
            smooth={'type': 'curvedCW', 'roundness': 0.2}
        )
    
    # Generate HTML with inline resources for Streamlit compatibility
    html = net.generate_html()
    
    # Render in Streamlit
    st.components.v1.html(html, height=750, scrolling=True)
    
    # Download button for interactive graph
    st.download_button(
        "📥 Download Interactive Graph (HTML)", 
        data=html,
        file_name="declarmima_graph.html", 
        mime="text/html",
        key="pyvis_download_custom"
    )


def render_graph_plotly_white(nx_graph, concept_abstract_map):
    """Render concept graph using Plotly with WHITE background and vibrant colors"""
    if len(nx_graph.nodes()) > 100:
        degrees = dict(nx_graph.degree())
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:100]
        nx_graph = nx_graph.subgraph(top_nodes).copy()
    
    pos = nx.spring_layout(nx_graph, k=2, iterations=50, seed=SEED)
    
    edge_x, edge_y, edge_hover = [], [], []
    for u, v in nx_graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        w = nx_graph[u][v].get('weight', 1)
        edge_type = nx_graph[u][v].get('edge_type', 'unknown')
        edge_hover.extend([f"{u} ↔ {v}<br>Weight: {w:.2f}<br>Type: {edge_type}"] * 2 + [None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=1.2, color='#666'),
        hoverinfo='text', hovertext=edge_hover, name='Connections'
    )
    
    node_x, node_y, node_text, node_size, node_color, node_symbol = [], [], [], [], [], []
    for node in nx_graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        deg = nx_graph.degree(node)
        freq = len(concept_abstract_map.get(node, []))
        node_text.append(f"{node}<br>Degree: {deg}<br>Frequency: {freq}")
        node_size.append(max(10, min(45, deg * 3 + 12)))
        
        node_lower = node.lower()
        if any(a in node_lower for a in ['al', 'ti', 'ni', 'cr', 'fe', 'co', 'mo', 'nb', 'cu', 'w', 'mn']):
            node_color.append('#E91E63')
            node_symbol.append('square')
        elif any(l in node_lower for l in ['laser', 'scan', 'power', 'melt', 'energy', 'speed', 'pulse']):
            node_color.append('#3F51B5')
            node_symbol.append('diamond')
        elif any(m in node_lower for m in ['grain', 'phase', 'hardness', 'strength', 'texture', 'microstructure']):
            node_color.append('#FF9800')
            node_symbol.append('circle')
        elif any(p in node_lower for p in ['porosity', 'crack', 'defect', 'void', 'residual']):
            node_color.append('#F44336')
            node_symbol.append('x')
        elif any(d in node_lower for d in ['digital', 'twin', 'machine', 'learning', 'neural', 'graph']):
            node_color.append('#9C27B0')
            node_symbol.append('star')
        else:
            node_color.append('#009688')
            node_symbol.append('circle')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        marker=dict(size=node_size, color=node_color, line=dict(width=2, color='#ffffff'), symbol=node_symbol),
        text=[n for n in nx_graph.nodes()], textposition="bottom center",
        textfont=dict(size=10, color='#000000'),
        hovertext=node_text, hoverinfo='text', name='Concepts'
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       showlegend=False, hovermode='closest',
                       margin=dict(b=0, l=0, r=0, t=0),
                       plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
                       xaxis=dict(showgrid=True, zeroline=False, showticklabels=False, gridcolor='#eee', range=[-1.5, 1.5]),
                       yaxis=dict(showgrid=True, zeroline=False, showticklabels=False, gridcolor='#eee', range=[-1.5, 1.5]),
                       annotations=[dict(
                           text="🔬 Drag nodes • Hover for details • Scroll to zoom • DECLARMIMA-aligned edges in pink",
                           showarrow=False, xref="paper", yref="paper",
                           x=0.5, y=-0.05, font=dict(size=10, color='#666')
                       )]
                   ))
    
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.1, xanchor="left", y=1.15, yanchor="top",
            buttons=[
                dict(label="🔄 Re-layout", method="relayout", args=[{"xaxis.range": [-1.5, 1.5], "yaxis.range": [-1.5, 1.5]}]),
                dict(label="🔍 Zoom In", method="relayout", args=[{"xaxis.autorange": False, "yaxis.autorange": False}]),
                dict(label="📐 Reset View", method="relayout", args=[{"xaxis.autorange": True, "yaxis.autorange": True}]),
            ]
        )]
    )
    
    st.plotly_chart(fig, use_container_width=True, key="plotly_graph")
    
    fig_json = fig.to_json()
    st.download_button(
        "📥 Download Plotly Graph (JSON)",
        data=fig_json,
        file_name="concept_graph_plotly.json",
        mime="application/json",
        key="plotly_download"
    )

def render_graph_fallback(nx_graph, concept_abstract_map):
    """Fallback text-based graph summary"""
    st.markdown("### 📊 Graph Summary (Text View)")
    st.markdown(f"- **Nodes**: {len(nx_graph.nodes())}")
    st.markdown(f"- **Edges**: {len(nx_graph.edges())}")
    
    if len(nx_graph.edges()) > 0:
        edge_list = [(u, v, nx_graph[u][v].get('weight', 1)) for u, v in nx_graph.edges()]
        edge_list.sort(key=lambda x: x[2], reverse=True)
        st.markdown("**🔗 Top 15 Strongest Connections:**")
        for i, (u, v, w) in enumerate(edge_list[:15], 1):
            edge_type = nx_graph[u][v].get('edge_type', 'unknown')
            declarmima_tag = " 🎯 DECLARMIMA" if edge_type == 'declarmina_aligned' else ""
            st.markdown(f"{i}. `{u}` ↔ `{v}` (weight: {w:.2f}, type: {edge_type}){declarmima_tag}")
    
    if len(concept_abstract_map) > 0:
        freq_data = [(c, len(concept_abstract_map.get(c, []))) for c in nx_graph.nodes()]
        freq_data.sort(key=lambda x: x[1], reverse=True)
        st.markdown("**📈 Top Concepts by Frequency:**")
        st.dataframe(
            pd.DataFrame(freq_data[:10], columns=["Concept", "Abstract Count"]),
            use_container_width=True
        )

def display_link_predictions(top_scores_df: pd.DataFrame, threshold=0.6):
    """Show top novel concept pairs with GNN scores and semantic novelty."""
    if top_scores_df.empty:
        st.info("No novel pairs scored above threshold. Try lowering edge sampling threshold.")
        return
    st.markdown("### 🔗 Predicted Missing Links (Research Gaps)")
    display_cols = ['concept_u', 'concept_v', 'gnn_affinity', 'semantic_novelty', 'composite_score']
    st.dataframe(
        top_scores_df[display_cols].head(20).style.format({
            'gnn_affinity': '{:.3f}', 'semantic_novelty': '{:.3f}', 'composite_score': '{:.3f}'
        }),
        use_container_width=True
    )
    # Show top ranked pair as "most promising innovation"
    top_pair = top_scores_df.iloc[0]
    st.success(f"🌟 **Top Innovation Frontier:** {top_pair['concept_u']} ↔ {top_pair['concept_v']} "
               f"(Score = {top_pair['composite_score']:.3f})")

# ==========================================
# STREAMLIT UI & PIPELINE ORCHESTRATION
# ==========================================
def render_sidebar():
    """Render sidebar with backend selection and configuration"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Backend selection
        st.subheader("🔧 LLM Backend")
        backend_option = st.radio(
            "Choose inference backend:",
            options=["Hugging Face Transformers", "Ollama (if installed)"],
            index=0,
            help="Transformers: direct HF model loading\nOllama: use local ollama serve (faster switching)"
        )
        st.session_state.inference_backend = backend_option
        
        # Model selection based on backend
        if backend_option == "Ollama (if installed)":
            if not OLLAMA_AVAILABLE:
                st.error("❌ ollama library not installed")
                st.code("pip install ollama")
                st.info("Also ensure Ollama server is running: ollama serve")
            available_ollama_models = [k for k in LOCAL_LLM_OPTIONS.keys() if is_ollama_model(k)]
            model_choice = st.selectbox(
                "🧠 Local LLM (Ollama)",
                options=available_ollama_models if available_ollama_models else ["No Ollama models available"],
                index=0 if available_ollama_models else 0,
                help="Models served via local Ollama instance"
            )
        else:
            hf_models = [k for k in LOCAL_LLM_OPTIONS.keys() if not is_ollama_model(k)]
            model_choice = st.selectbox(
                "🧠 Local LLM (Hugging Face)",
                options=hf_models, index=2,
                help="Models loaded directly via transformers library"
            )
        st.session_state.llm_model_choice = model_choice
        
        # 4-bit quantization for HF models
        if backend_option == "Hugging Face Transformers" and not is_ollama_model(model_choice):
            st.session_state.use_4bit_quantization = st.checkbox(
                "🗜️ Use 4-bit quantization (reduces VRAM usage)", value=True,
                help="Enable for models >3B parameters to reduce memory usage by ~75%"
            )
        
        # Ollama host configuration
        if backend_option == "Ollama (if installed)" or is_ollama_model(model_choice):
            st.session_state.ollama_host = st.text_input(
                "🌐 Ollama Host", value=st.session_state.get('ollama_host', 'http://localhost:11434'),
                help="URL of your Ollama server (default: http://localhost:11434)"
            )
        
        # Visualization selector
        st.subheader("🎨 Graph Visualization")
        viz_backend = st.selectbox(
            "Choose visualization engine:",
            options=["PyVis (Interactive Network)", "Plotly (Stable Plot)", "Text Summary (Fallback)"],
            index=0,
            help="PyVis: Rich interactive network\nPlotly: More stable, zoomable plot\nText: Simple fallback view"
        )
        st.session_state['viz_backend'] = viz_backend
        
        # DECLARMIMA integration toggle
        st.subheader("🎯 DECLARMIMA Integration")
        use_declarmima = st.toggle(
            "Use DECLARMIMA proposal as seed knowledge",
            value=True,
            help="Inject concepts from the DECLARMIMA research proposal and prioritize abstracts aligned with project goals"
        )
        st.session_state['use_declarmima'] = use_declarmima
        
        # Adaptive config toggles
        abstract_preview = st.text_area("📋 Paste abstracts (preview):", height=100, key="preview")
        preview_count = len([t for t in re.split(r'\n\s*\n', abstract_preview) if t.strip()]) if abstract_preview.strip() else 0
        
        if preview_count > 0 and preview_count <= 25:
            st.warning(f"📉 Small corpus ({preview_count} abstracts): applying adaptive settings")
            st.toggle("Enable semantic clustering", value=True, key="use_clustering", disabled=True)
            st.toggle("Inject domain seeds", value=True, key="inject_seeds", disabled=True)
            st.toggle("Use embedding edges", value=True, key="semantic_edges", disabled=True)
        else:
            st.toggle("Enable semantic clustering", value=False, key="use_clustering")
            st.toggle("Inject domain seeds", value=False, key="inject_seeds")
            st.toggle("Use embedding edges", value=False, key="semantic_edges")
        
        # New customization controls
        st.subheader("🎨 Visual Customization")
        st.session_state.physics_enabled = st.checkbox("Enable graph physics", value=True, key="physics_toggle")
        st.session_state.min_node_size = st.slider("Min node size", 8, 30, 12, key="min_size")
        st.session_state.max_node_size = st.slider("Max node size", 30, 80, 50, key="max_size")
        
        st.markdown("---")
        st.markdown("**🎯 DECLARMIMA Focus Areas:**")
        st.markdown("- 🔬 Laser-matter interaction mechanisms")
        st.markdown("- 🧱 Multiscale computational modeling")
        st.markdown("- 🤖 Physics-informed machine learning")
        st.markdown("- 📊 Process-structure-property relationships")
        st.markdown("- 🔍 Uncertainty quantification & validation")
        
        st.markdown("---")
        st.markdown("**⚡ Performance:**")
        max_hyp = st.slider("Max hypotheses", 1, 20, 10)
        st.session_state['max_hypotheses'] = max_hyp
        
        if st.button("🗑️ Clear Cache"):
            st.cache_resource.clear()
            gc.collect()
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
            st.success("✅ Cache cleared!")
        
        # System info
        st.markdown("---")
        gpu_info = "CUDA" if torch.cuda.is_available() else "CPU"
        vram_info = f"{get_available_gpu_memory():.1f}GB free" if torch.cuda.is_available() and get_available_gpu_memory() else "N/A"
        st.caption(f"🖥️ Device: {gpu_info}")
        st.caption(f"💾 Available VRAM: {vram_info}")
        st.caption(f"📦 Embedding model: ~80MB")
        st.caption(f"🤖 LLM: {LOCAL_LLM_OPTIONS.get(model_choice, 'unknown')}")

def main():
    st.title("🔬 DECLARMIMA: Laser-Microstructure Interaction Analyzer")
    st.caption("Physics-informed digital twins for multicomponent alloys • Dual LLM backend: HF Transformers or Ollama")
    
    # Render sidebar with backend selection
    render_sidebar()
    
    # Memory warning for HF models
    if st.session_state.get('llm_model_choice') and not is_ollama_model(st.session_state.get('llm_model_choice', '')):
        mem_info = estimate_model_memory(
            st.session_state.llm_model_choice, 
            st.session_state.get('use_4bit_quantization', True)
        )
        available_vram = get_available_gpu_memory()
        if available_vram and not mem_info['cpu_ok']:
            required = float(mem_info['vram_4bit'].replace('GB','').replace('~','').strip()) if 'GB' in mem_info['vram_4bit'] else 100
            if available_vram < required:
                st.markdown(f"""<div style="background:#fef3c7;border-left:4px solid #f59e0b;padding:0.75rem;border-radius:0 0.5rem 0.5rem 0;margin:0.5rem 0">
⚠️ <strong>Memory Warning:</strong> {st.session_state.llm_model_choice} requires ~{mem_info['vram_4bit']} VRAM.
You have ~{available_vram:.1f}GB available. Consider:
<ul><li>Using 4-bit quantization (already enabled)</li><li>Selecting a smaller model</li><li>Using Ollama backend for better memory management</li></ul></div>""", unsafe_allow_html=True)
    
    # Abstract input
    abstract_input = st.text_area(
        "📋 Paste scientific abstracts (blank lines separate):", 
        height=300,
        placeholder="""Example:
"Laser powder bed fusion of AlSi10Mg reveals columnar-to-equiaxed transition at 85 J/mm³..."

"High-entropy alloy CoCrFeNiMo via DED shows 420 HV microhardness from nanoscale precipitates..."
""")
    
    if st.button("🚀 Analyze Abstracts", type="primary", use_container_width=True):
        if not abstract_input.strip():
            st.error("⚠️ Please enter at least one abstract.")
            return
            
        abstracts = [t.strip() for t in re.split(r'\n\s*\n', abstract_input) if t.strip()]
        
        if len(abstracts) < 10:
            st.info(f"💡 {len(abstracts)} abstracts: Maximum semantic enrichment mode")
        elif len(abstracts) > 35:
            st.warning(f"⚠️ {len(abstracts)} abstracts may increase processing time")
            
        progress_bar = st.progress(0.0)
        status = st.status("🔄 Initializing...", expanded=True)
        
        try:
            # Load models with selected backend
            with status:
                st.write("📦 Loading models...")
                embed_model = load_embedding_model()
                
                # Load LLM with backend routing
                model_key = st.session_state.get('llm_model_choice', DEFAULT_LLM_NAME)
                tokenizer, llm_model, device_or_host, backend_type = load_local_llm(
                    model_key, use_4bit=st.session_state.get('use_4bit_quantization', True)
                )
                
                if tokenizer is None and backend_type == "transformers":
                    st.error("Failed to load HF model. Try selecting a different option.")
                    return
                if llm_model is None and backend_type == "ollama":
                    st.error("Failed to connect to Ollama. Ensure 'ollama serve' is running.")
                    return
                    
                st.success(f"✅ Models loaded ({backend_type})")
            progress_bar.progress(0.10)
            
            # DECLARMIMA: Encode proposal text for correlation
            proposal_embedding = None
            config = get_adaptive_config(len(abstracts))
            if st.session_state.get('use_declarmima', True) and config.get("USE_DECLARMIMA_SEEDS", True):
                with st.status("🎯 Processing DECLARMIMA proposal..."):
                    proposal_embedding = embed_model.encode([DECLARMIMA_PROPOSAL_TEXT], show_progress_bar=False)[0]
                    st.write(f"✅ Proposal embedding: {proposal_embedding.shape}")
                    declarmima_concepts = extract_declarmima_concepts(DECLARMIMA_PROPOSAL_TEXT, embed_model)
                    st.write(f"✅ Extracted {len(declarmima_concepts)} DECLARMIMA seed concepts")
            progress_bar.progress(0.15)
            
            # Get config with user toggles
            if "use_clustering" in st.session_state:
                config["USE_SEMANTIC_CLUSTERING"] = st.session_state.use_clustering
            if "inject_seeds" in st.session_state:
                config["INJECT_DOMAIN_SEEDS"] = st.session_state.inject_seeds
            if "semantic_edges" in st.session_state:
                config["USE_SEMANTIC_EDGES"] = st.session_state.semantic_edges
            config["USE_DECLARMIMA_SEEDS"] = st.session_state.get('use_declarmima', True)
            config["CORRELATE_WITH_PROPOSAL"] = st.session_state.get('use_declarmima', True)
            
            # Step 1-2: Extract concepts with selected backend
            with st.status("🔍 Extracting concepts..."):
                all_concepts, all_metrics = extract_concepts_from_abstracts(
                    abstracts, tokenizer, llm_model, backend_type
                )
                valid_concepts, concept_to_id, id_to_concept, concept_abstract_map = normalize_and_filter_concepts(
                    all_concepts, embed_model, config, proposal_embedding
                )
                st.write(f"✅ **{len(valid_concepts)}** concepts extracted")
                if len(valid_concepts) < 10:
                    st.info("💡 Small concept set: semantic-only graph mode")
            progress_bar.progress(0.25)
            
            if len(valid_concepts) < 3:
                st.warning("⚠️ Very few concepts. Injecting additional seeds...")
                valid_concepts, concept_to_id = inject_domain_seeds(valid_concepts, concept_to_id)
                st.success(f"✅ Recovered {len(valid_concepts)} concepts")
            
            # Step 3: Build graph
            with st.status("🕸️ Building concept graph..."):
                nx_graph = build_concept_graph(all_concepts, concept_to_id, embed_model, config, proposal_embedding)
                d_prev_dict = dict(nx.all_pairs_shortest_path_length(nx_graph, cutoff=4))
                pos_pairs, neg_pairs = sample_edges_for_training(nx_graph, d_prev_dict, valid_concepts, concept_to_id, config)
                st.write(f"✅ Graph: **{len(valid_concepts)}** nodes, **{nx_graph.number_of_edges()}** edges")
                
                declarmima_edges = sum(1 for _, _, d in nx_graph.edges(data=True) if d.get('edge_type') == 'declarmina_aligned')
                if declarmima_edges > 0:
                    st.success(f"🎯 {declarmima_edges} edges aligned with DECLARMIMA goals")
                
                if not nx.is_connected(nx_graph):
                    n_comp = nx.number_connected_components(nx_graph)
                    st.info(f"🔗 Graph has {n_comp} component(s)")
            progress_bar.progress(0.40)
            
            # Step 4: Embeddings
            with st.status("🧠 Generating embeddings..."):
                embed_dim = get_embedding_dimension(embed_model)
                st.write(f"✅ Embedding dimension: {embed_dim}")
                node_features = generate_embeddings(valid_concepts, embed_model)
                st.write(f"✅ Node features shape: {node_features.shape}")
            progress_bar.progress(0.50)
            
            # Step 5: Train GNN
            def _training_progress(epoch, loss):
                progress_value = 0.50 + (epoch / TRAIN_EPOCHS) * 0.30
                progress_bar.progress(min(1.0, max(0.0, progress_value)))
                if epoch % 10 == 0:
                    status.write(f"📊 Epoch {epoch}/{TRAIN_EPOCHS} | Loss: {loss:.4f}")
            
            with st.status("🤖 Training GraphSAGE..."):
                gnn_model, final_emb, adj_indices, adj_values = train_gnn(
                    node_features, nx_graph, concept_to_id, pos_pairs, neg_pairs, _training_progress
                )
                st.success("✅ GNN training complete")
            progress_bar.progress(0.80)
            
            # Step 6: Scoring
            with st.status("📈 Scoring novel directions..."):
                concept_properties, ridge = compute_microstructure_quantification(
                    valid_concepts, concept_abstract_map, all_metrics, nx_graph
                )
                top_scores = compute_research_direction_scores(
                    gnn_model, node_features, final_emb, nx_graph, valid_concepts, concept_properties,
                    ridge, embed_model, d_prev_dict, adj_indices, adj_values
                )
                st.write(f"✅ Scored **{len(top_scores)}** novel pairs")
            progress_bar.progress(0.90)
            
            # Step 7: LLM curation with selected backend
            with st.status("✍️ Generating hypotheses..."):
                max_hyp = st.session_state.get('max_hypotheses', 10)
                st.write(f"📝 Generating up to {max_hyp} DECLARMIMA-aligned hypotheses...")
                
                directions_df = generate_research_directions(
                    top_scores, tokenizer, llm_model, backend_type, max_hypotheses=max_hyp,
                    proposal_context="physics-informed digital twins, multiscale modeling, laser-matter mechanisms"
                )
                st.success("✅ Pipeline complete!")
            progress_bar.progress(1.00)
            status.update(label="✅ Analysis complete!", state="complete", expanded=False)
            
            # === DISPLAY RESULTS ===
            st.subheader("🎯 Top Research Directions (DECLARMIMA-Aligned)")
            if len(directions_df) > 0:
                if 'DECLARMIMA Alignment' in directions_df.columns:
                    st.dataframe(
                        directions_df[['Concept Pair', 'Composite Score', 'Expected Gain', 'Feasibility', 'DECLARMIMA Alignment', 'Research Hypothesis']],
                        use_container_width=True,
                        column_config={
                            "Concept Pair": st.column_config.TextColumn("Pair", width="medium"),
                            "Composite Score": st.column_config.NumberColumn("Score", format="%.3f"),
                            "Expected Gain": st.column_config.NumberColumn("Gain", format="%.1f"),
                            "Feasibility": st.column_config.NumberColumn("Feas.", format="%.2f"),
                            "DECLARMIMA Alignment": st.column_config.TextColumn("Alignment"),
                            "Research Hypothesis": st.column_config.TextColumn("Hypothesis", width="large")
                        }
                    )
                else:
                    st.dataframe(
                        directions_df[['Concept Pair', 'Composite Score', 'Expected Gain', 'Feasibility', 'Research Hypothesis']],
                        use_container_width=True
                    )
                csv = directions_df.to_csv(index=False)
                st.download_button("📥 Download CSV", data=csv, file_name="declarmima_directions.csv", mime="text/csv")
            else:
                st.info("💡 No novel pairs scored above threshold. Try adjusting parameters.")
            
            # === NEW: QUANTITATIVE METRICS DASHBOARD ===
            with st.expander("📊 Graph Structural Metrics (Research Field Analysis)", expanded=False):
                metrics = compute_graph_metrics(nx_graph)
                display_metric_dashboard(metrics)
            
            # === NEW: CATEGORICAL SUNBURST CHART ===
            with st.expander("📈 Research Domain Hierarchy (Sunburst)", expanded=False):
                labels, parents, values = build_category_hierarchy(valid_concepts, concept_abstract_map)
                render_sunburst_chart(labels, parents, values)
            
            # === NEW: LINK PREDICTION DASHBOARD ===
            with st.expander("🧠 Missing Link Predictions (GNN Research Gaps)", expanded=False):
                display_link_predictions(top_scores)
            
            # === VISUALIZATION WITH ENHANCED CUSTOMIZATION ===
            st.subheader("🌐 Interactive Concept Graph")
            
            selected_viz = st.session_state.get('viz_backend', 'PyVis (Interactive Network)')
            
            if len(nx_graph.nodes()) == 0:
                st.warning("⚠️ No nodes to display.")
                render_graph_fallback(nx_graph, concept_abstract_map)
            elif nx_graph.number_of_edges() == 0:
                st.warning("⚠️ No edges — building semantic fallback")
                nx_graph = build_semantic_only_graph(list(nx_graph.nodes()), embed_model, similarity_threshold=0.65)
                if nx_graph.number_of_edges() == 0:
                    render_graph_fallback(nx_graph, concept_abstract_map)
                else:
                    st.success(f"✅ Fallback: {nx_graph.number_of_edges()} edges")
                    if selected_viz == "PyVis (Interactive Network)":
                        render_graph_pyvis_custom(
                            nx_graph, concept_abstract_map,
                            physics_enabled=st.session_state.get('physics_enabled', True),
                            min_node_size=st.session_state.get('min_node_size', 12),
                            max_node_size=st.session_state.get('max_node_size', 50)
                        )
                    elif selected_viz == "Plotly (Stable Plot)":
                        render_graph_plotly_white(nx_graph, concept_abstract_map)
                    else:
                        render_graph_fallback(nx_graph, concept_abstract_map)
            else:
                try:
                    if selected_viz == "PyVis (Interactive Network)":
                        st.info("🎨 Rendering with PyVis (white theme, vibrant colors, user-adjustable physics)...")
                        render_graph_pyvis_custom(
                            nx_graph, concept_abstract_map,
                            physics_enabled=st.session_state.get('physics_enabled', True),
                            min_node_size=st.session_state.get('min_node_size', 12),
                            max_node_size=st.session_state.get('max_node_size', 50)
                        )
                    elif selected_viz == "Plotly (Stable Plot)":
                        st.info("🎨 Rendering with Plotly (white theme, vibrant colors)...")
                        render_graph_plotly_white(nx_graph, concept_abstract_map)
                    else:
                        st.info("📝 Showing text summary")
                        render_graph_fallback(nx_graph, concept_abstract_map)
                except Exception as viz_error:
                    st.warning(f"⚠️ {selected_viz} failed: {viz_error}")
                    render_graph_fallback(nx_graph, concept_abstract_map)
            
            # Diagnostics
            with st.expander("📊 Graph Diagnostics", expanded=len(valid_concepts) < 30):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Nodes", len(valid_concepts))
                col2.metric("Edges", nx_graph.number_of_edges())
                col3.metric("Avg Degree", f"{np.mean([d for _,d in nx_graph.degree()]):.2f}")
                n_components = nx.number_connected_components(nx_graph)
                col4.metric("Connected", "✅" if nx.is_connected(nx_graph) else f"❌ ({n_components})")
                
                if st.session_state.get('use_declarmima', True):
                    declarmima_nodes = sum(1 for c in valid_concepts if any(kw in c.lower() for kw in DOMAIN_KEYWORDS))
                    col1, col2 = st.columns(2)
                    col1.metric("DECLARMIMA Concepts", declarmima_nodes)
                    declarmima_edges = sum(1 for _, _, d in nx_graph.edges(data=True) if d.get('edge_type') == 'declarmina_aligned')
                    col2.metric("Aligned Edges", declarmima_edges)
                
                if len(valid_concepts) > 0:
                    st.markdown("### Top Concepts")
                    freq_data = [(c, len(concept_abstract_map.get(c, []))) for c in valid_concepts]
                    freq_data.sort(key=lambda x: x[1], reverse=True)
                    st.dataframe(pd.DataFrame(freq_data[:10], columns=["Concept", "Frequency"]), use_container_width=True)
                    
        except torch.cuda.OutOfMemoryError as e:
            st.error(f"❌ CUDA OOM: {e}")
            st.info("💡 Reduce 'Max hypotheses' or switch to CPU")
            with st.expander("🔍 Traceback"):
                st.code(traceback.format_exc())
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            with st.expander("🔍 Traceback"):
                st.code(traceback.format_exc())
        finally:
            gc.collect()
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **🔧 Dual LLM Backend Support:**
    - ✅ **Hugging Face Transformers**: Direct model loading with 4-bit quantization, device auto-detection
    - ✅ **Ollama**: Server-based inference via HTTP, faster model switching, lower Python memory footprint
    - ✅ **All 20 models** from LASER RAG codebase: 12 HF + 8 Ollama options
    - ✅ **Memory-aware**: VRAM estimation, CPU fallback, graceful degradation
    
    **🎯 DECLARMIMA Integration:**
    - ✅ Proposal text injected as seed knowledge base
    - ✅ Abstract-proposal semantic correlation scoring
    - ✅ Hypotheses aligned with physics-informed digital twin goals
    - ✅ DECLARMIMA-aligned edges highlighted in pink
    
    **🎨 White Theme Visualization:**
    - ✅ Clean white background (#ffffff) with dark text (#000000)
    - ✅ Vibrant, high-contrast node colors: Pink=alloys, Indigo=laser, Orange=microstructure, Red=defects, Purple=computational
    - ✅ Thick, colorful edges for clear visibility
    - ✅ White borders on nodes for pop effect
    - ✅ **NEW:** User-adjustable physics, node size sliders, interactive toggles
    
    **📊 Publication‑Quality Enhancements:**
    - ✅ Graph metrics dashboard (density, clustering, betweenness)
    - ✅ Sunburst chart for research domain hierarchy (downloadable SVG)
    - ✅ Link prediction panel showing GNN-scored missing links
    - ✅ Updated PyVis with category‑based coloring and export
    
    **💡 Small Corpus Tips (10-25 abstracts):**
    - Semantic clustering merges similar terms
    - DECLARMIMA seed injection adds domain concepts
    - Embedding edges connect semantically related concepts
    - Adaptive thresholds relax frequency requirements
    
    **🔧 Technical:** Qwen2.5-0.5B + Sentence-BERT (384-dim) + PyTorch GraphSAGE | Local processing
    
    **⚠️ Troubleshooting:**
    1. Reduce "Max hypotheses" if experiencing crashes
    2. Click "Clear Cache" between runs
    3. Try Plotly if PyVis has rendering issues
    4. Use "Text Summary" fallback for guaranteed visibility
    5. For Ollama: ensure `ollama serve` is running and model is pulled (`ollama pull qwen2.5:7b`)
    """)

if __name__ == "__main__":
    main()
