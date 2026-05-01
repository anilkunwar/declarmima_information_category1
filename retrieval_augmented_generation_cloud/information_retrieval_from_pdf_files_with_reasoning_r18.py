#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LASER MICROSTRUCTURE RAG CHATBOT - CROSS-DOCUMENT SCIENTIFIC REASONING & VISUALIZATION
========================================================================================
FULLY UPGRADED VERSION (CODE 17+): CONCEPT UNIFICATION & QUERY EMPHASIS
========================================================================================
ARCHITECTURE: Three-Layer Concept Unification
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Lexical Normalization (Rule-Based)                  │
│ • Alias dictionaries (MATERIAL_ALIASES, METHOD_ALIASES)     │
│ • Regex pattern expansion for hyphenated/spaced variants    │
│ • Stemming + lemmatization fallback                          │
│ • Fast but requires pre-programmed knowledge                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Semantic Clustering (Embedding-Based)              │
│ • Extract candidate n-grams from corpus                      │
│ • Embed candidates using sentence transformer                │
│ • Build similarity graph (cosine > 0.82 threshold)          │
│ • Run connected components / HDBSCAN clustering             │
│ • Auto-discover unanticipated synonyms (e.g., "keyhole" ↔   │
│   "vapor capillary regime")                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Contextual Disambiguation (Query-Time)             │
│ • Extract query concepts with same pipeline                  │
│ • Boost salience of semantically similar document terms     │
│ • Re-rank retrieval using expanded synonym set              │
│ • Highlight query terms AND recognized synonyms in answer   │
│ • Enable cross-document consensus across alias variants     │
└─────────────────────────────────────────────────────────────┘

QUERY EMPHASIS MECHANICS:
1. User asks: "How does laser power affect keyhole stability in Ti-6Al-4V?"
2. System extracts: laser power, keyhole, Ti-6Al-4V
3. Expands via synonym clusters:
   - laser power → [beam intensity, irradiance, fluence, laser irradiation]
   - keyhole → [vapor capillary, deep penetration regime, keyhole mode]
4. Injects temporary salience boost (0.95) for all expanded terms
5. Retrieves chunks matching ANY variant, re-ranked by hybrid score
6. Computes consensus across papers using unified concept families
7. Generates answer with bolded query terms AND recognized synonyms
8. Shows "Also searched for: [synonym list]" for transparency

UNIFIED CONCEPT DATA STRUCTURE:
Instead of fragmented entities:
  entities = {
      "multicomponent alloy": [...],
      "multi-component alloy": [...],
      "MPEA": [...],
      "HEA": [...]
  }

We store unified families:
  unified_concepts = {
      "multicomponent_alloy_family": {
          "canonical": "multicomponent alloy",
          "aliases": {"multi-component alloy", "MPEA", "HEA", "CCA", ...},
          "embedding_centroid": np.array([...]),  # mean of all alias embeddings
          "documents": {"doc1": [Entity(...)], "doc2": [...]},
          "salience": 0.94,
          "consensus_values": [...],
          "source_terminology": {"doc1": "HEA", "doc2": "complex concentrated alloy"}
      }
  }

BENEFITS:
✓ Consensus detection works across all terminology variants automatically
✓ Visualization shows one node with multiple labels, not fragmented nodes
✓ User queries in ANY terminology find the unified concept family
✓ Full traceability: we know which paper used which term
✓ No redaction: all surface forms preserved in raw index, linked to canonical
"""

# =============================================================================
# IMPORTS & DEPENDENCIES
# =============================================================================
import streamlit as st
import os
import tempfile
import time
import re
import json
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
from typing import List, Dict, Optional, Tuple, Union, Any, Set, Callable, FrozenSet
from datetime import datetime
import sys
import subprocess
import platform
from pathlib import Path
from collections import defaultdict, Counter, OrderedDict
import hashlib
from dataclasses import dataclass, field, asdict
import gc
import logging
from functools import lru_cache

# Matplotlib / NetworkX for static publication graphs
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx

# NEW: Advanced visualization libraries (optional)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from bokeh.plotting import figure, output_file, save, show
    from bokeh.models import (Circle, MultiLine, HoverTool, BoxSelectTool, 
                              TapTool, ColumnDataSource, LabelSet, CustomJS)
    from bokeh.palettes import Category20, Viridis256, Plasma256, Inferno256, Magma256, Cividis256
    from bokeh.layouts import column, row
    from bokeh.embed import file_html
    from bokeh.resources import CDN
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

try:
    import holoviews as hv
    from holoviews import opts
    hv.extension('bokeh')
    HOLOVIEWS_AVAILABLE = True
except ImportError:
    HOLOVIEWS_AVAILABLE = False

# LangChain / RAG imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Transformers
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, AutoModelForCausalLM,
    pipeline, set_seed, BitsAndBytesConfig
)

# Optional: Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Bibliographic metadata
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

# Pyvis for network graph
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

# Optional DGL
try:
    import dgl
    import dgl.function as fn
    from dgl.nn import HeteroGraphConv, SAGEConv
    import torch.nn as nn
    import torch.nn.functional as F
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    dgl = None

# Optional scikit-learn
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.cluster import HDBSCAN, AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Optional NLTK for lemmatization
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    try:
        wordnet.ensure_loaded()
        NLTK_AVAILABLE = True
        lemmatizer = WordNetLemmatizer()
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        NLTK_AVAILABLE = True
        lemmatizer = WordNetLemmatizer()
except ImportError:
    NLTK_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
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
    "[Ollama] qwen2.5:14b (via ollama serve)": "ollama:qwen2.5:14b",
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
    "max_context_tokens": 2048,
    "max_new_tokens": 512,
    "temperature": 0.05,
    "synonym_similarity_threshold": 0.82,  # For semantic clustering
    "query_expansion_boost": 0.95,  # Temporary salience boost for query synonyms
    "min_cluster_size": 2,  # Minimum terms for semantic cluster
    "max_ngram_length": 4,  # Max n-gram length for candidate extraction
}

# =============================================================================
# DECLARMIMA-ALIGNED KEYWORDS & ALIASES (EXPANDED)
# =============================================================================

LASER_KEYWORDS = {
    "ablation": ["ablation", "material removal", "threshold fluence", "laser ablation", "ablation threshold"],
    "plasma": ["plasma formation", "ionization", "electron density", "plume", "plasma shielding"],
    "thermal": ["heat affected zone", "melting", "thermal diffusion", "resolidification", "heat-affected zone"],
    "ultrafast": ["femtosecond", "picosecond", "pulse duration", "ultrafast laser", "fs laser"],
    "morphology": ["ripples", "LIPSS", "surface structuring", "periodic structures", "nanostructures", "microstructures"],
    "parameters": ["fluence", "wavelength", "pulse energy", "repetition rate", "spot size", "scan speed", "overlap",
                   "hatch distance", "laser power", "point distance"],
    "materials": ["silicon", "steel", "titanium", "polymer", "glass", "ceramic", "aluminum", "copper", "tungsten",
                  "multicomponent alloy", "high entropy alloy", "solder", "Sn-Ag-Cu", "Al-Cr-Fe-Ni", "Inconel"],
    "characterization": ["SEM", "AFM", "profilometry", "spectroscopy", "microscopy", "Raman", "XRD", "EDX",
                         "EBSD", "Tomography", "X-ray radiography"],
    "additive_manufacturing": ["additive manufacturing", "3D printing", "selective laser melting", "SLM",
                               "laser powder bed fusion", "LPBF", "directed energy deposition"],
    "multicomponent": ["multicomponent alloy", "multi-principal element alloy", "MPEA", "high entropy alloy",
                       "HEA", "multi-component", "complex concentrated alloy"],
    "digital_twin": ["digital twin", "physics-informed digital twin", "PIDT", "in-silico", "virtual qualification"],
    "simulation": ["phase field", "molecular dynamics", "MD simulation", "finite element", "MOOSE",
                   "CALPHAD", "Thermo-Calc", "multi-scale", "mesoscale", "nanoscale"],
    "data_driven": ["machine learning", "neural network", "random forest", "CNN", "data-driven",
                    "physics-informed ML", "feature engineering", "tensor decomposition"],
    "properties": ["interfacial energy", "thermal conductivity", "diffusion coefficient", "viscosity",
                   "gibbs free energy", "enthalpy", "absorptivity", "reflectivity", "spatter", "porosity"],
}

# =============================================================================
# EXPANDED ALIAS DICTIONARIES FOR LAYER 1: LEXICAL NORMALIZATION
# =============================================================================

MATERIAL_ALIASES = {
    # Silicon family
    "silicon": ["silicon", "si", "crystalline silicon", "c-si", "si(100)", "si(111)", 
                "mono-crystalline silicon", "polycrystalline silicon", "poly-si"],
    
    # Titanium family
    "titanium": ["titanium", "ti", "cp-ti", "ti-6al-4v", "ti6al4v", "ti-6-4", 
                 "grade 5 titanium", "titanium alloy"],
    
    # Steel family
    "steel": ["steel", "stainless steel", "ss304", "ss316", "mild steel", "carbon steel",
              "austenitic steel", "martensitic steel", "ferritic steel"],
    
    # Aluminum family
    "aluminum": ["aluminum", "aluminium", "al", "al6061", "al-6061", "al-si-mg",
                 "aluminum alloy", "aluminium alloy"],
    
    # Copper family
    "copper": ["copper", "cu", "electrolytic copper", "ofhc copper"],
    
    # Tungsten family
    "tungsten": ["tungsten", "w", "wolfram"],
    
    # Glass family
    "glass": ["glass", "fused silica", "sio2", "borosilicate", "quartz glass"],
    
    # Polymer family
    "polymer": ["polymer", "pmma", "polyimide", "pei", "pc", "polycarbonate", "ptfe",
                "peek", "abs", "pla", "polyethylene", "pe", "pp", "thermoplastic"],
    
    # Ceramic family
    "ceramic": ["ceramic", "alumina", "al2o3", "zirconia", "zro2", "silicon carbide", "sic"],
    
    # Solder family
    "Sn-Ag-Cu": ["snagcu", "sac", "sn-ag-cu", "sn-3.5ag-0.5cu", "solder", "lead-free solder",
                 "sac305", "sac405", "tin-silver-copper"],
    
    # Inconel family
    "Al-Cr-Fe-Ni": ["alcrfeni", "al-cr-fe-ni", "inconel 718", "in718", "nickel superalloy",
                    "inconel", "ni-based superalloy"],
    
    # HIGH ENTROPY ALLOY / MULTICOMPONENT ALLOY FAMILY (EXPANDED)
    "multicomponent alloy": [
        # Canonical forms
        "multicomponent alloy", "multicomponent alloys",
        "multi-component alloy", "multi-component alloys",
        "multi component alloy", "multi component alloys",
        
        # Abbreviations
        "mca", "mpea", "hea", "cca", "mcea",
        
        # High entropy family
        "high entropy alloy", "high-entropy alloy", "highentropy alloy",
        "hea", "heas", "high entropy alloys",
        "medium entropy alloy", "medium-entropy alloy", "mea",
        "low entropy alloy", "low-entropy alloy", "lea",
        "refractory high entropy alloy", "refractory hea", "rhea",
        
        # Multi-principal element family
        "multi-principal element alloy", "multi principal element alloy",
        "multiprincipal element alloy", "multiprincipal-element alloy",
        "multi-principal-element alloy", "mpea", "mpeas",
        
        # Complex concentrated family
        "complex concentrated alloy", "complex-concentrated alloy",
        "cca", "ccas", "complex concentrated alloys",
        
        # Descriptive variants
        "multielement alloy", "multi-element alloy", "multi element alloy",
        "many element alloy", "many elements alloy", "many-component alloy",
        "multiple elements alloy", "multiple components alloy",
        "several elements alloy", "several components alloy",
        "numerous elements alloy", "various elements alloy",
        "complex alloy", "complicated alloy", "heterogeneous alloy",
        "polycomponent alloy", "polymetallic alloy", "multi-base alloy",
        
        # Composition-based
        "quinary alloy", "quaternary alloy", "ternary alloy",
        "5-component alloy", "4-component alloy", "3-component alloy",
        "five-component", "four-component", "three-component",
        ">2 elements", ">2 components", "more than two elements",
        "more than 2 elements", "more than two components",
        
        # Specific HEA compositions (common in literature)
        "cocrfeni", "cocrfenimn", "alcocrfeni", "crmnfeconi",
        "alcrfeni", "fecomnCrNi", "nicocrfe", "alticocrfeni",
        "co-cr-fe-ni", "al-co-cr-fe-ni", "cr-mn-fe-co-ni",
        
        # System descriptors
        "multicomponent alloy system", "multi-component alloy system",
        "multicomponent metallic", "multi-component metallic",
        "multicomponent system", "multi-component system",
    ],
    
    # High entropy alloy as separate canonical (for backward compatibility)
    "high entropy alloy": ["hea", "high-entropy alloy", "highentropy alloy",
                          "high entropy alloys", "heas"],
}

METHOD_ALIASES = {
    # Microscopy
    "sem": ["sem", "scanning electron microscopy", "scanning electron microscope",
            "field emission sem", "fe-sem", "environmental sem", "e-sem"],
    "afm": ["afm", "atomic force microscopy", "atomic force microscope",
            "tapping mode afm", "contact mode afm"],
    "tem": ["tem", "transmission electron microscopy", "transmission electron microscope",
            "stem", "scanning tem", "high-resolution tem", "hrtem"],
    "ebsd": ["ebsd", "electron backscatter diffraction", "eb sd", "backscatter diffraction"],
    
    # Spectroscopy
    "raman": ["raman", "raman spectroscopy", "micro-raman", "confocal raman",
              "surface enhanced raman", "sers"],
    "xrd": ["xrd", "x-ray diffraction", "x ray diffraction", "powder xrd",
            "glancing angle xrd", "grazing incidence xrd"],
    "edx": ["edx", "eds", "energy dispersive x-ray", "energy-dispersive",
            "edax", "energy dispersive spectroscopy"],
    "xps": ["xps", "x-ray photoelectron spectroscopy", "esca", "photoelectron spectroscopy"],
    
    # Imaging & Tomography
    "x-ray_imaging": ["synchrotron x-ray", "x-ray radiography", "x-ray tomography",
                      "micro-ct", "computed tomography", "ct scan",
                      "in-situ x-ray", "operando x-ray"],
    
    # Profilometry & Surface Analysis
    "profilometry": ["profilometry", "optical profilometry", "white light interferometry",
                     "wli", "confocal profilometry", "stylus profilometry"],
    
    # Thermal Analysis
    "thermal_analysis": ["dsc", "differential scanning calorimetry", "dta", "tga",
                         "thermogravimetric", "thermogravimetric analysis"],
    
    # Computational Methods
    "phase_field": ["phase-field", "phase field", "pf simulation", "moose",
                    "micress", "phasefield", "phase field modeling"],
    "finite_element": ["finite element", "fem", "finite element method", "fea",
                       "abaqus", "ansys", "comsol", "finite element analysis"],
    "calphad": ["calphad", "thermo-calc", "thermocalc", "pandat", "fact sage",
                "thermodynamic modeling", "calphad modeling"],
    "molecular_dynamics": ["md", "molecular dynamics", "molecular dynamics simulation",
                           "lammps", "gromacs", "namd", "atomistic simulation"],
    "dft": ["dft", "density functional theory", "ab initio", "first principles",
            "vasp", "quantum espresso", "castep"],
    
    # Data-Driven Methods
    "machine_learning": ["machine learning", "ml", "deep learning", "cnn", "gnn",
                         "graph neural network", "random forest", "surrogate model",
                         "physics-informed ml", "pinns", "feature engineering"],
}

# =============================================================================
# QUANTITY PATTERNS FOR NUMERIC EXTRACTION
# =============================================================================
QUANTITY_PATTERNS = {
    "wavelength": re.compile(r'(\d+(?:\.\d+)?)\s*(?:nm|nanometers?)\s*(?:wavelength|λ|lambda)', re.I),
    "pulse_duration": re.compile(r'(\d+(?:\.\d+)?)\s*(?:fs|femtoseconds?|ps|picoseconds?|ns|nanoseconds?)\s*(?:pulse|duration)', re.I),
    "fluence": re.compile(r'(\d+(?:\.\d+)?)\s*(?:J/cm²|J/cm2|J\s*cm[-²2]|fluence)', re.I),
    "repetition_rate": re.compile(r'(\d+(?:\.\d+)?)\s*(?:kHz|MHz|Hz)\s*(?:repetition|rate|freq)', re.I),
    "spot_size": re.compile(r'(\d+(?:\.\d+)?)\s*(?:µm|um|microns?)\s*(?:spot|diameter|beam\s*radius|waist)', re.I),
    "periodicity": re.compile(r'(\d+(?:\.\d+)?)\s*(?:nm|µm|um|microns?)\s*(?:period|periodicity|spacing|LSFL|HSFL)', re.I),
    "roughness": re.compile(r'(\d+(?:\.\d+)?)\s*(?:nm|µm|um)\s*(?:roughness|Ra|RMS|Rq)', re.I),
    "threshold": re.compile(r'(?:threshold|ablation\s*threshold)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(?:J/cm²|J/cm2|mJ/cm²|GW/cm²|TW/cm²)', re.I),
    "power": re.compile(r'(\d+(?:\.\d+)?)\s*(?:W|mW|kW|MW)\s*(?:power|average\s*power)', re.I),
    "pulse_energy": re.compile(r'(\d+(?:\.\d+)?)\s*(?:µJ|uJ|mJ|nJ)\s*(?:pulse\s*energy|energy\s*per\s*pulse)', re.I),
    "scan_speed": re.compile(r'(\d+(?:\.\d+)?)\s*(?:mm/s|mm/min|m/s)\s*(?:scan\s*speed|travel\s*speed)', re.I),
    "hatch_distance": re.compile(r'(\d+(?:\.\d+)?)\s*(?:µm|um|mm)\s*(?:hatch\s*distance|hatch\s*spacing)', re.I),
    "laser_power": re.compile(r'(\d+(?:\.\d+)?)\s*(?:W)\s*(?:laser\s*power|nominal\s*power)', re.I),
    "component_fraction": re.compile(r'(\d+(?:\.\d+)?)\s*(?:at\.%|wt\.%|at%|wt%)\s*(?:of\s*)?([A-Za-z]+)', re.I),
    "interfacial_energy": re.compile(r'(\d+(?:\.\d+)?)\s*(?:J/m²|J/m2|mJ/m²|mJ/m2)\s*(?:interfacial\s*energy|surface\s*tension)', re.I),
    "thermal_conductivity": re.compile(r'(\d+(?:\.\d+)?)\s*(?:W/(?:m·?K|mK))\s*(?:thermal\s*conductivity)', re.I),
}

# =============================================================================
# MODEL MEMORY ESTIMATES
# =============================================================================
MODEL_MEMORY_ESTIMATES = {
    "gpt2": {"params": "1.5B", "vram_fp16": "~3GB", "vram_4bit": "~1GB", "cpu_ok": True},
    "Qwen/Qwen2-0.5B-Instruct": {"params": "0.5B", "vram_fp16": "~1GB", "vram_4bit": "~400MB", "cpu_ok": True},
    "Qwen/Qwen2.5-0.5B-Instruct": {"params": "0.5B", "vram_fp16": "~1GB", "vram_4bit": "~400MB", "cpu_ok": True},
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {"params": "1.1B", "vram_fp16": "~2.5GB", "vram_4bit": "~800MB", "cpu_ok": True},
    "Qwen/Qwen2.5-1.5B-Instruct": {"params": "1.5B", "vram_fp16": "~3.5GB", "vram_4bit": "~1.2GB", "cpu_ok": False},
    "Qwen2.5-3B-Instruct": {"params": "3B", "vram_fp16": "~6GB", "vram_4bit": "~2GB", "cpu_ok": False},
    "mistralai/Mistral-7B-Instruct-v0.3": {"params": "7B", "vram_fp16": "~14GB", "vram_4bit": "~4.5GB", "cpu_ok": False},
    "meta-llama/Llama-3.2-3B-Instruct": {"params": "3B", "vram_fp16": "~6GB", "vram_4bit": "~2GB", "cpu_ok": False},
    "Qwen2.5-7B-Instruct": {"params": "7B", "vram_fp16": "~14GB", "vram_4bit": "~4.5GB", "cpu_ok": False},
    "meta-llama/Llama-3.1-8B-Instruct": {"params": "8B", "vram_fp16": "~16GB", "vram_4bit": "~5GB", "cpu_ok": False},
    "google/gemma-2-9b-it": {"params": "9B", "vram_fp16": "~18GB", "vram_4bit": "~6GB", "cpu_ok": False},
    "tiiuae/falcon-7b-instruct": {"params": "7B", "vram_fp16": "~14GB", "vram_4bit": "~4.5GB", "cpu_ok": False},
}

# =============================================================================
# HIERARCHICAL TAXONOMY FOR VISUALIZATION
# =============================================================================
ENTITY_TAXONOMY = {
    "MATERIAL": {
        "Pure Element": {
            "Metal": [
                "titanium", "ti", "cp-ti", "copper", "cu", "aluminum", "al", "al6061", "al-6061",
                "tungsten", "w", "nickel", "ni", "iron", "fe", "chromium", "cr", "cobalt", "co",
                "manganese", "mn", "zinc", "zn", "tin", "sn", "silver", "ag", "gold", "au", "lead", "pb"
            ],
            "Metalloid": ["silicon", "si", "germanium", "ge", "crystalline silicon", "c-si", "si(100)", "si(111)"],
            "Refractory": ["tungsten", "w", "molybdenum", "mo", "tantalum", "ta", "niobium", "nb", "rhenium", "re"]
        },
        "Alloy System": {
            "Binary": ["sn-cu", "cu-ni", "ni-al", "ti-al", "fe-cr", "al-cr", "cu-zn", "brass"],
            "Ternary": ["sn-ag-cu", "sac", "sn-3.5ag-0.5cu", "al-cr-fe", "ni-cr-fe", "ti-al-v", "ti6al4v", "ti-6al-4v"],
            "Quaternary+ / HEA": [
                "alcrfeni", "al-cr-fe-ni", "cocrfeni", "cocrfenimn", "alcocrfeni",
                "hea", "high entropy alloy", "mpea", "multi-principal element alloy",
                "complex concentrated alloy", "refractory hea", "crmnfeconi",
                "multicomponent alloy", "multi-component alloy", "multicomponent"
            ],
            "Superalloy": ["inconel", "in718", "in-718", "nimonic", "rene", "haynes", "nickel superalloy"]
        },
        "Compound / Ceramic": {
            "Oxide": ["sio2", "al2o3", "zro2", "tio2", "zirconia", "alumina", "fused silica", "silica", "borosilicate"],
            "Carbide": ["sic", "wc", "tungsten carbide", "tic", "b4c", "boron carbide"],
            "Nitride": ["si3n4", "tin", "aln", "crn", "gan"]
        },
        "Polymer": {
            "Thermoplastic": ["pmma", "pc", "pei", "peek", "ptfe", "polycarbonate", "polyimide", "abs", "pla", "polyethylene", "pe", "pp"],
            "Thermoset": ["epoxy", "polyurethane", "phenolic", "polyester", "polyimide"]
        },
        "Composite": ["cfrp", "carbon fiber", "metal matrix composite", "mmc", "ceramic matrix composite", "cmc", "glass fiber"]
    },
    "METHOD": {
        "Experimental": {
            "Microscopy": [
                "sem", "scanning electron microscopy", "scanning electron microscope",
                "afm", "atomic force microscopy", "atomic force microscope",
                "tem", "transmission electron microscopy",
                "ebsd", "electron backscatter diffraction",
                "optical microscopy", "confocal microscopy"
            ],
            "Spectroscopy": [
                "raman", "raman spectroscopy", "micro-raman",
                "xrd", "x-ray diffraction",
                "edx", "eds", "energy dispersive x-ray", "energy-dispersive",
                "xps", "ftir", "libs", "spectroscopy"
            ],
            "Tomography & Imaging": [
                "synchrotron x-ray", "x-ray radiography", "x-ray tomography",
                "ct scan", "computed tomography", "ultrasound", "radiography", "tomography"
            ],
            "Profilometry": ["profilometry", "optical profilometry", "white light interferometry", "wli"],
            "Thermal Analysis": ["dsc", "differential scanning calorimetry", "dta", "tga", "thermogravimetric"]
        },
        "Computational": {
            "Atomistic": [
                "md", "molecular dynamics", "molecular dynamics simulation",
                "dft", "density functional theory", "ab initio",
                "lammps", "vasp", "quantum espresso", "atomistic"
            ],
            "Continuum Mechanics": [
                "fem", "finite element", "finite element method", "fea",
                "abaqus", "ansys", "comsol"
            ],
            "Phase-Field": [
                "phase field", "phase-field", "pf simulation", "moose", "micress", "phasefield"
            ],
            "Thermodynamic": [
                "calphad", "thermo-calc", "thermocalc", "pandat", "fact sage", "thermodynamic modeling"
            ],
            "Fluid Dynamics": [
                "cfd", "computational fluid dynamics", "flow3d", "openfoam", "fluent", "flow-3d"
            ],
            "Data-Driven": [
                "machine learning", "ml", "deep learning", "cnn", "gnn", "graph neural network",
                "random forest", "surrogate model", "digital twin", "physics-informed", "pinns",
                "physics-informed ml", "feature engineering", "tensor decomposition"
            ]
        }
    },
    "PHENOMENON": {
        "Laser-Matter Interaction": {
            "Thermal Regime": [
                "melting", "vaporization", "heat affected zone", "haz", "heat-affected zone",
                "thermal diffusion", "resolidification", "recrystallization",
                "solidification", "cooling rate", "thermal gradient"
            ],
            "Optical / Plasma": [
                "ablation", "plasma", "plume", "ionization", "plasma shielding",
                "reflection", "absorptivity", "multiphoton", "avalanche ionization"
            ],
            "Structural Evolution": [
                "ripples", "lipss", "nanostructures", "microstructures",
                "periodic structures", "surface structuring", "self-organization", "hsfl", "lsfl"
            ]
        },
        "Material Response": {
            "Mechanical": [
                "residual stress", "distortion", "cracking", "delamination",
                "spatter", "warping", "deformation", "stress"
            ],
            "Microstructural": [
                "grain growth", "dendrite", "cellular structure", "epitaxial growth",
                "texture", "porosity", "void", "inclusion", "segregation", "grain boundary"
            ],
            "Interfacial": [
                "imc", "intermetallic", "intermetallic compound", "intermetallics",
                "wetting", "spreading", "contact angle",
                "interfacial energy", "surface tension", "marangoni", "buoyancy"
            ]
        }
    },
    "PARAMETER": {
        "Laser Source": {
            "Spatial": ["wavelength", "spot size", "beam radius", "waist", "m2", "beam quality", "focal spot"],
            "Temporal": ["pulse duration", "pulse energy", "repetition rate", "peak power", "duty cycle"],
            "Process Control": ["laser power", "average power", "fluence", "irradiance", "intensity", "focal position", "defocus"]
        },
        "Process Kinematics": {
            "Scanning": ["scan speed", "travel speed", "scan strategy", "raster", "contour", "meander", "island"],
            "Powder Bed": ["hatch distance", "point distance", "exposure time", "layer thickness", "overlap", "stripe width"],
            "Environment": ["atmosphere", "shielding gas", "oxygen level", "substrate temperature", "preheat", "build plate temperature", "chamber pressure"]
        },
        "Outcome Metric": {
            "Geometric": ["roughness", "ra", "rms", "rq", "periodicity", "period", "spacing", "waviness", "flatness"],
            "Performance": ["hardness", "tensile strength", "yield strength", "elongation", "fatigue life", "wear rate", "corrosion resistance", "conductivity"],
            "Defect Metric": ["porosity fraction", "crack density", "spatter rate", "balling", "keyhole depth", "lack of fusion"]
        }
    }
}

# =============================================================================
# DECLARMIMA PROPOSAL TEXT (for salience seeding)
# =============================================================================
DECLARMIMA_PROPOSAL_TEXT = """Deciphering laser-microstructure interaction in multicomponent alloys (DECLARMIMA) Scientific goals: Additive manufacturing, laser processing, multicomponent alloys, high-entropy alloys, digital twins, physics-informed machine learning, phase field modeling, molecular dynamics, melt pool dynamics, microstructure evolution, process-structure-property relationships, selective laser melting, powder bed fusion, laser powder bed fusion, in-situ monitoring, defect formation, porosity, spatter, residual stress, grain morphology, phase transformation, solidification, Marangoni convection, CALPHAD thermodynamics, interfacial energy, thermal conductivity, viscosity, absorptivity, reflectivity, Gaussian heat source, finite element method, MOOSE framework, LAMMPS, ThermoCalc, neural networks, convolutional neural networks, random forest, Bayesian machine learning, uncertainty quantification, feature engineering, tensor decomposition, scale-bridging, multiscale modeling, inverse design, optimization, Al-Si-Mg alloys, Ti-6Al-4V, Inconel 718, Sn-Ag-Cu solders, CoCrFeNi HEAs, intermetallic compounds, columnar grains, equiaxed grains, dendritic structures, martensite, austenite, precipitates, segregation, crack propagation, fatigue life, tensile strength, yield strength, microhardness, elongation, ductility, wear resistance, corrosion resistance, oxidation resistance, laser power, scan speed, hatch spacing, layer thickness, pulse duration, energy density, spot diameter, cooling rate, solidification rate, dilution ratio, powder particle size, particle size distribution, flowability, oxygen content, moisture content, bed temperature, pre-heating, post-processing, heat treatment, surface finishing, quality monitoring, photodiode sensors, line scanners, camera trackers, acoustic transducers, synchrotron X-ray imaging, EBSD, nanoindentation, in-situ XRD, SEM, TEM, AFM, digital image correlation, machine vision, data fusion, knowledge graphs, concept graphs, graph neural networks, GraphSAGE, node embeddings, edge prediction, link prediction, research direction discovery, hypothesis generation, novelty scoring, feasibility assessment, property gain prediction, composite scoring, adaptive configuration, small corpus optimization, semantic clustering, domain seed injection, hybrid graph construction, co-occurrence edges, semantic similarity edges, contrastive learning, edge sampling, sparse tensors, degree normalization, mean aggregation, two-layer architecture, decoder network, BCE loss, Adam optimizer, training loop, evaluation metrics, progress tracking, memory management, CUDA optimization, CPU fallback, error handling, fallback strategies, interactive visualization, PyVis, Plotly, force-directed layout, spring layout, node styling, edge styling, hover tooltips, download functionality, text fallback, diagnostics panel, concept frequency, edge weight, graph connectivity, component analysis, degree distribution, clustering coefficient, centrality measures, path length, bridge edges, semantic bridges, knowledge injection, concept normalization, alloy notation standardization, laser term normalization, unit standardization, regex extraction, quantitative metrics, grain size, mechanical properties, energy density, defect fraction, prompt engineering, JSON parsing, fallback extraction, domain validation, generic term filtering, concept abstraction, category mapping, hierarchical representation, representative selection, cluster merging, similarity threshold, distance matrix, linkage method, embedding encoding, batch processing, progress display, model caching, resource management, timeout handling, user feedback, status indicators, progress bars, error messages, warning dialogs, success notifications, download buttons, CSV export, HTML export, JSON export, interactive controls, physics parameters, gravity, spring length, damping, overlap, stabilization, node sampling, size limiting, performance optimization, browser compatibility, JavaScript execution, CDN resources, inline embedding, iframe alternative, HTML rendering, Streamlit components, responsive design, mobile compatibility, accessibility, color contrast, theme switching, dark mode, light mode, user preferences, session state, configuration persistence, adaptive thresholds, corpus size detection, parameter tuning, hyperparameter optimization, validation metrics, testing framework, debugging tools, logging, tracebacks, exception handling, graceful degradation, fallback rendering, text summary, edge listing, frequency tables, diagnostic metrics, connectivity checks, component counting, degree analysis, clustering analysis, centrality computation, path analysis, bridge detection, semantic analysis, novelty computation, feasibility scoring, property prediction, ridge regression, feature concatenation, pair scoring, candidate filtering, distance checking, graph distance, shortest path, all-pairs shortest path, cutoff parameter, edge sampling strategy, positive pairs, negative pairs, hard negatives, distance-focused sampling, random sampling, attempts limit, pair uniqueness, edge existence check, tensor construction, sparse adjacency, degree computation, normalization, message passing, aggregation, combination, activation, ReLU, linear layers, sequential decoder, concatenation, sigmoid, logits, contrastive loss, binary cross-entropy, training epochs, learning rate, optimizer step, gradient computation, backward pass, zero grad, model evaluation, no grad context, final embeddings, adjacency indices, adjacency values, node features, embedding dimension, shape validation, error raising, minimal pairs, edge uniqueness, source adjacency, destination adjacency, stacking, tensor conversion, device placement, long dtype, float32, GPU memory, CPU fallback, memory cleanup, garbage collection, CUDA cache emptying, progress callback, epoch logging, loss tracking, convergence monitoring, early stopping, model saving, checkpointing, inference mode, prediction scoring, candidate generation, random sampling, pair filtering, distance computation, KeyError handling, default distance, semantic similarity, cosine similarity, embedding encoding, numpy arrays, tensor conversion, CPU numpy, forward pass, model eval, no grad, decoder output, logits extraction, sigmoid activation, CPU conversion, numpy array, property lookup, median computation, ridge prediction, clipping, normalization, weighted scoring, alpha weights, composite score, sorting, head selection, DataFrame creation, column selection, formatting, display configuration, download preparation, CSV serialization, MIME type, button callback, empty check, info message, parameter suggestion, graph rendering, node count check, edge count check, fallback graph building, semantic-only fallback, similarity threshold adjustment, success message, text fallback rendering, node iteration, degree computation, frequency lookup, category detection, color assignment, size computation, title formatting, node addition, edge iteration, weight lookup, type lookup, color mapping, edge addition, value scaling, width scaling, color assignment, smooth edges, curved edges, roundness parameter, HTML generation, inline resources, Streamlit HTML component, height parameter, scrolling enable, width parameter, download button, file naming, MIME type, unique key, error catching, warning display, fallback suggestion, retry buttons, alternative backend, exception handling, error message display, traceback expansion, code display, memory cleanup, GPU cache clearing, garbage collection, footer display, tips section, visualization options, PyVis description, Plotly description, text summary description, technical stack, crash prevention tips, rendering troubleshooting, browser console check, zoom controls, download fallback, text view guarantee"""

# =============================================================================
# LAYER 1: LEXICAL NORMALIZER - RULE-BASED SYNONYM RESOLUTION
# =============================================================================

class ConceptNormalizer:
    """
    Layer 1: Lexical Normalization for concept unification.
    
    Resolves surface-form variations to canonical forms using:
    1. Pre-defined alias dictionaries (MATERIAL_ALIASES, METHOD_ALIASES)
    2. Regex pattern expansion for hyphenated/spaced/concatenated variants
    3. Optional stemming/lemmatization fallback via NLTK
    
    Fast but requires pre-programmed knowledge of domain terminology.
    """
    
    def __init__(self, alias_dicts: Optional[Dict[str, List[str]]] = None,
                 enable_lemmatization: bool = True):
        """
        Initialize the normalizer.
        
        Args:
            alias_dicts: Optional dict of {canonical: [aliases]} to merge with defaults
            enable_lemmatization: Whether to use NLTK lemmatization as fallback
        """
        self._synonym_map: Dict[str, str] = {}
        self._canonical_to_variants: Dict[str, Set[str]] = defaultdict(set)
        self._fuzzy_patterns: Dict[str, re.Pattern] = {}
        self._enable_lemmatization = enable_lemmatization and NLTK_AVAILABLE
        
        # Build master synonym map from all sources
        self._build_master_synonym_map(alias_dicts)
        self._build_reverse_index()
        self._build_fuzzy_patterns()
        
        logger.info(f"ConceptNormalizer initialized with {len(self._synonym_map)} synonym mappings")
    
    def _build_master_synonym_map(self, custom_aliases: Optional[Dict[str, List[str]]] = None):
        """Build comprehensive synonym mapping from all alias dictionaries."""
        
        # Start with MATERIAL_ALIASES and METHOD_ALIASES
        for canonical, aliases in MATERIAL_ALIASES.items():
            self._add_canonical(canonical, aliases)
        
        for canonical, aliases in METHOD_ALIASES.items():
            self._add_canonical(canonical, aliases)
        
        # Add LASER_KEYWORDS as topics
        for topic, keywords in LASER_KEYWORDS.items():
            self._add_canonical(topic, keywords)
        
        # Add core pillars with high priority
        core_pillars = {
            "laser": ["laser", "lasers", "lasing", "laser beam", "laser-beam", "laserbeam",
                      "laser radiation", "laser light", "coherent light", "laser source"],
            "microstructure": ["microstructure", "micro-structure", "micro structure",
                              "microstructural", "grain structure", "grain morphology",
                              "phase structure", "phase morphology", "crystal structure"],
            "interaction": ["interaction", "interactions", "coupling", "coupled",
                           "correlation", "relationship", "interplay", "synergy"],
            "multicomponent alloy": MATERIAL_ALIASES.get("multicomponent alloy", []),
        }
        
        for canonical, aliases in core_pillars.items():
            self._add_canonical(canonical, aliases, priority=True)
        
        # Add domain-specific families
        self._add_family("melt pool", [
            "melt pool", "melt-pool", "meltpool", "molten pool", "molten-pool",
            "fusion zone", "fusion-zone", "melt zone", "liquid pool", "weld pool",
            "keyhole", "key hole", "key-hole", "vapor cavity", "deep penetration"
        ])
        
        self._add_family("marangoni convection", [
            "marangoni convection", "marangoni-convection", "marangoni flow",
            "marangoni effect", "thermocapillary convection", "thermocapillary flow",
            "surface tension gradient", "capillary convection", "capillary flow"
        ])
        
        self._add_family("porosity", [
            "porosity", "porous", "void", "voids", "pore", "pores", "cavity", "cavities",
            "gas pore", "shrinkage pore", "keyhole pore", "microporosity", "nanoporosity"
        ])
        
        self._add_family("residual stress", [
            "residual stress", "residual-stress", "residual stresses", "residual strain",
            "internal stress", "thermal stress", "distortion", "warping", "crack", "cracking"
        ])
        
        self._add_family("intermetallic compound", [
            "intermetallic compound", "intermetallic-compound", "imc", "intermetallic",
            "intermetallics", "imc layer", "intermetallic layer", "intermetallic phase"
        ])
        
        self._add_family("solidification", [
            "solidification", "solidify", "solidifying", "resolidification",
            "crystallization", "nucleation", "grain formation", "phase transformation"
        ])
        
        self._add_family("grain morphology", [
            "grain morphology", "grain structure", "grain size", "grain boundary",
            "grain orientation", "grain texture", "grain growth", "grain refinement",
            "equiaxed grain", "columnar grain", "dendrite", "dendritic structure"
        ])
        
        # Merge custom aliases if provided
        if custom_aliases:
            for canonical, aliases in custom_aliases.items():
                self._add_canonical(canonical, aliases)
    
    def _add_canonical(self, canonical: str, aliases: List[str], priority: bool = False):
        """Add a canonical form with its aliases to the synonym map."""
        canonical_clean = canonical.lower().strip()
        
        # Add all variants
        for alias in aliases:
            alias_clean = alias.lower().strip()
            if alias_clean and alias_clean != canonical_clean:
                # Direct mapping
                self._synonym_map[alias_clean] = canonical_clean
                
                # Handle hyphen/space variations
                no_hyphen = alias_clean.replace("-", " ")
                no_space = alias_clean.replace(" ", "")
                hyphenated = alias_clean.replace(" ", "-")
                
                for variant in [no_hyphen, no_space, hyphenated]:
                    if variant and variant != alias_clean and variant != canonical_clean:
                        self._synonym_map[variant] = canonical_clean
                
                # Track reverse mapping
                self._canonical_to_variants[canonical_clean].add(alias_clean)
        
        # Ensure canonical maps to itself
        self._synonym_map[canonical_clean] = canonical_clean
        self._canonical_to_variants[canonical_clean].add(canonical_clean)
    
    def _add_family(self, canonical: str, aliases: List[str]):
        """Add a concept family with enhanced pattern generation."""
        self._add_canonical(canonical, aliases)
        
        # Generate regex patterns for fuzzy matching
        for alias in aliases:
            if len(alias) > 3:  # Skip very short terms
                # Escape special regex characters but keep word boundaries
                escaped = re.escape(alias)
                # Allow optional hyphens/spaces between words
                pattern_str = r'\b' + escaped.replace(r'\ ', r'[\s\-]?') + r'\b'
                try:
                    self._fuzzy_patterns[canonical] = re.compile(pattern_str, re.IGNORECASE)
                except re.error:
                    pass  # Skip invalid patterns
    
    def _build_reverse_index(self):
        """Build reverse index: canonical -> set of all aliases."""
        # Already built during _add_canonical, but ensure completeness
        for variant, canon in self._synonym_map.items():
            self._canonical_to_variants[canon].add(variant)
    
    def _build_fuzzy_patterns(self):
        """Build regex patterns for fuzzy matching of concept families."""
        # Patterns already built in _add_family, but add global patterns here
        global_patterns = {
            r'multi[\s\-]?component': "multicomponent alloy",
            r'high[\s\-]?entropy[\s\-]?alloy': "high entropy alloy",
            r'multi[\s\-]?principal[\s\-]?element': "multi-principal element alloy",
            r'complex[\s\-]?concentrated[\s\-]?alloy': "complex concentrated alloy",
            r'(?:laser[\s\-]?power|beam[\s\-]?power|irradiance|fluence)': "laser power",
            r'(?:melt[\s\-]?pool|molten[\s\-]?pool|fusion[\s\-]?zone)': "melt pool",
        }
        
        for pattern_str, canonical in global_patterns.items():
            try:
                self._fuzzy_patterns[canonical] = re.compile(pattern_str, re.IGNORECASE)
            except re.error:
                continue
    
    def normalize(self, text: str, context: Optional[str] = None) -> str:
        """
        Normalize a text string to its canonical concept form.
        
        Args:
            text: The text to normalize
            context: Optional surrounding context for disambiguation
            
        Returns:
            The canonical form of the concept, or the original text if no match
        """
        if not text or not isinstance(text, str):
            return text
        
        cleaned = text.lower().strip()
        
        # Layer 1a: Direct lookup
        if cleaned in self._synonym_map:
            return self._synonym_map[cleaned]
        
        # Layer 1b: Normalization variants
        variants = [
            cleaned,
            cleaned.replace(" ", "").replace("-", ""),  # No spaces/hyphens
            cleaned.replace("-", " "),  # Hyphens to spaces
            cleaned.replace(" ", "-"),  # Spaces to hyphens
            re.sub(r'[\s\-]+', ' ', cleaned).strip(),  # Normalize whitespace
        ]
        
        for variant in variants:
            if variant in self._synonym_map:
                return self._synonym_map[variant]
        
        # Layer 1c: Fuzzy pattern matching
        for canonical, pattern in self._fuzzy_patterns.items():
            if pattern.search(cleaned):
                return canonical
        
        # Layer 1d: Lemmatization fallback (if enabled)
        if self._enable_lemmatization:
            lemmatized = self._lemmatize(cleaned)
            if lemmatized in self._synonym_map:
                return self._synonym_map[lemmatized]
        
        # No match found - return original (will be handled by Layer 2)
        return cleaned
    
    def _lemmatize(self, text: str) -> str:
        """Lemmatize text using NLTK (if available)."""
        if not NLTK_AVAILABLE:
            return text
        
        try:
            # Simple lemmatization: split and lemmatize each word
            words = text.split()
            lemmatized = [lemmatizer.lemmatize(w) for w in words]
            return ' '.join(lemmatized)
        except Exception:
            return text  # Fallback to original on error
    
    def get_variants(self, canonical: str) -> FrozenSet[str]:
        """Get all known variants (aliases) for a canonical concept."""
        return frozenset(self._canonical_to_variants.get(canonical.lower(), {canonical}))
    
    def is_canonical(self, text: str) -> bool:
        """Check if text is already in canonical form."""
        return text.lower().strip() in self._canonical_to_variants
    
    def batch_normalize(self, texts: List[str], contexts: Optional[List[str]] = None) -> List[str]:
        """Normalize a batch of texts efficiently."""
        if contexts is None:
            contexts = [None] * len(texts)
        return [self.normalize(t, c) for t, c in zip(texts, contexts)]
    
    def find_matching_canonical(self, text: str) -> Optional[str]:
        """Find which canonical form a text matches, or None if no match."""
        normalized = self.normalize(text)
        if normalized != text.lower().strip() and normalized in self._canonical_to_variants:
            return normalized
        return None


# =============================================================================
# LAYER 2: SEMANTIC CLUSTERER - EMBEDDING-BASED SYNONYM DISCOVERY
# =============================================================================

if SKLEARN_AVAILABLE:
    class SemanticClusterer:
        """
        Layer 2: Semantic Clustering for automatic synonym discovery.
        
        Discovers unanticipated synonyms by:
        1. Extracting candidate n-grams from corpus
        2. Embedding candidates using sentence transformer
        3. Building similarity graph (cosine > threshold)
        4. Running connected components / HDBSCAN clustering
        5. Electing representative canonical from each cluster
        
        This catches domain jargon not in pre-defined alias lists.
        """
        
        def __init__(self, 
                     embedding_model,
                     similarity_threshold: float = 0.82,
                     min_cluster_size: int = 2,
                     max_cluster_size: Optional[int] = None,
                     clustering_method: str = 'connected_components',
                     batch_size: int = 128,
                     max_ngram: int = 4):
            """
            Initialize the semantic clusterer.
            
            Args:
                embedding_model: Sentence transformer or LangChain embedding model
                similarity_threshold: Cosine similarity threshold for edge creation
                min_cluster_size: Minimum terms to form a valid cluster
                max_cluster_size: Maximum cluster size (None = unlimited)
                clustering_method: 'connected_components', 'hdbscan', or 'agglomerative'
                batch_size: Batch size for embedding computation
                max_ngram: Maximum n-gram length for candidate extraction
            """
            self.embed_model = embedding_model
            self.sim_threshold = similarity_threshold
            self.min_cluster_size = min_cluster_size
            self.max_cluster_size = max_cluster_size
            self.method = clustering_method
            self.batch_size = batch_size
            self.max_ngram = max_ngram
            
            self.clusters: List[Set[str]] = []
            self.centroids: Dict[int, np.ndarray] = {}
            self.term_to_cluster: Dict[str, int] = {}
            self._candidate_cache: Dict[str, np.ndarray] = {}
            
            logger.info(f"SemanticClusterer initialized: threshold={similarity_threshold}, "
                       f"method={clustering_method}, min_size={min_cluster_size}")
        
        def extract_candidates(self, 
                             chunks: List[Document], 
                             max_ngram: Optional[int] = None) -> List[str]:
            """
            Extract candidate n-grams from document chunks.
            
            Args:
                chunks: List of Document objects to process
                max_ngram: Override default max n-gram length
                
            Returns:
                List of unique candidate n-grams
            """
            max_n = max_ngram or self.max_ngram
            candidates = set()
            word_pattern = re.compile(r'\b[a-z][a-z\-]{2,}\b')  # Words with at least 3 chars
            
            for chunk in chunks:
                text = chunk.page_content.lower()
                
                # Split into sentences for better n-gram boundaries
                sentences = re.split(r'[.!?;]\s*', text)
                
                for sent in sentences:
                    # Extract words
                    tokens = word_pattern.findall(sent)
                    
                    # Generate n-grams
                    for n in range(1, min(max_n + 1, len(tokens) + 1)):
                        for i in range(len(tokens) - n + 1):
                            ngram = ' '.join(tokens[i:i+n])
                            # Filter: meaningful length, not just numbers
                            if len(ngram) > 3 and not ngram.isdigit():
                                candidates.add(ngram)
            
            # Filter out very common stop words and generic terms
            stop_terms = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                         'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 
                         'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
                         'do', 'does', 'did', 'will', 'would', 'could', 'should',
                         'may', 'might', 'can', 'this', 'that', 'these', 'those',
                         'it', 'its', 'fig', 'figure', 'table', 'eq', 'equation',
                         'ref', 'reference', 'et', 'al', 'al.', 'etc'}
            
            filtered = [c for c in candidates 
                       if not any(stop in c for stop in stop_terms)
                       and len(c.split()) <= max_n]
            
            logger.info(f"Extracted {len(filtered)} candidate n-grams from {len(chunks)} chunks")
            return filtered
        
        def _embed_batch(self, texts: List[str]) -> np.ndarray:
            """Embed a batch of texts using the configured embedding model."""
            embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                if hasattr(self.embed_model, 'embed_documents'):
                    # LangChain HuggingFaceEmbeddings
                    batch_emb = self.embed_model.embed_documents(batch)
                elif hasattr(self.embed_model, 'encode'):
                    # SentenceTransformer
                    batch_emb = self.embed_model.encode(batch, show_progress_bar=False)
                else:
                    # Fallback: embed one by one
                    batch_emb = [self._embed_single(t) for t in batch]
                
                embeddings.extend(batch_emb)
            
            return np.array(embeddings)
        
        def _embed_single(self, text: str) -> np.ndarray:
            """Embed a single text with caching."""
            if text in self._candidate_cache:
                return self._candidate_cache[text]
            
            if hasattr(self.embed_model, 'embed_query'):
                emb = self.embed_model.embed_query(text)
            elif hasattr(self.embed_model, 'encode'):
                emb = self.embed_model.encode(text)
            else:
                raise ValueError("Embedding model has no suitable method")
            
            self._candidate_cache[text] = np.array(emb)
            return np.array(emb)
        
        def cluster_terms(self, terms: List[str]) -> None:
            """
            Cluster terms using embedding similarity.
            
            Args:
                terms: List of candidate terms to cluster
            """
            if len(terms) < self.min_cluster_size:
                logger.warning(f"Too few terms ({len(terms)}) for clustering")
                return
            
            logger.info(f"Clustering {len(terms)} terms using {self.method}...")
            
            # Step 1: Embed all terms
            logger.info("Computing embeddings...")
            embeddings = self._embed_batch(terms)
            
            # Step 2: Compute similarity matrix
            logger.info("Computing similarity matrix...")
            sim_matrix = cosine_similarity(embeddings)
            
            # Step 3: Build similarity graph
            G = nx.Graph()
            G.add_nodes_from(range(len(terms)))
            
            edge_count = 0
            for i in range(len(terms)):
                for j in range(i + 1, len(terms)):
                    if sim_matrix[i, j] >= self.sim_threshold:
                        G.add_edge(i, j, weight=sim_matrix[i, j])
                        edge_count += 1
            
            logger.info(f"Built similarity graph: {G.number_of_nodes()} nodes, {edge_count} edges")
            
            # Step 4: Run clustering algorithm
            if self.method == 'connected_components':
                components = list(nx.connected_components(G))
                clusters = [set(terms[idx] for idx in comp) for comp in components]
                
            elif self.method == 'hdbscan':
                try:
                    clusterer = HDBSCAN(
                        min_cluster_size=self.min_cluster_size,
                        metric='euclidean',
                        cluster_selection_method='eom'
                    )
                    labels = clusterer.fit_predict(embeddings)
                    
                    # Group terms by cluster label
                    cluster_dict = defaultdict(set)
                    for term, label in zip(terms, labels):
                        if label != -1:  # -1 = noise
                            cluster_dict[label].add(term)
                    
                    clusters = list(cluster_dict.values())
                    
                except Exception as e:
                    logger.warning(f"HDBSCAN failed: {e}, falling back to connected components")
                    components = list(nx.connected_components(G))
                    clusters = [set(terms[idx] for idx in comp) for comp in components]
                    
            elif self.method == 'agglomerative':
                try:
                    clusterer = AgglomerativeClustering(
                        n_clusters=None,
                        distance_threshold=1 - self.sim_threshold,
                        linkage='average'
                    )
                    labels = clusterer.fit_predict(embeddings)
                    
                    cluster_dict = defaultdict(set)
                    for term, label in zip(terms, labels):
                        cluster_dict[label].add(term)
                    
                    clusters = list(cluster_dict.values())
                    
                except Exception as e:
                    logger.warning(f"Agglomerative clustering failed: {e}, falling back")
                    components = list(nx.connected_components(G))
                    clusters = [set(terms[idx] for idx in comp) for comp in components]
            else:
                raise ValueError(f"Unknown clustering method: {self.method}")
            
            # Step 5: Filter and store valid clusters
            self.clusters = []
            for cluster in clusters:
                if self.min_cluster_size <= len(cluster):
                    if self.max_cluster_size and len(cluster) > self.max_cluster_size:
                        continue  # Skip overly large clusters
                    self.clusters.append(cluster)
            
            # Step 6: Compute centroids for each cluster
            for i, cluster in enumerate(self.clusters):
                indices = [terms.index(t) for t in cluster if t in terms]
                if indices:
                    centroid = np.mean(embeddings[indices], axis=0)
                    self.centroids[i] = centroid
                    
                    # Map terms to cluster ID
                    for term in cluster:
                        self.term_to_cluster[term] = i
            
            logger.info(f"Discovered {len(self.clusters)} synonym clusters")
        
        def get_cluster_for_term(self, term: str) -> Optional[Set[str]]:
            """Get the synonym cluster containing a term, or None."""
            cluster_id = self.term_to_cluster.get(term.lower())
            if cluster_id is not None and cluster_id < len(self.clusters):
                return self.clusters[cluster_id]
            return None
        
        def get_canonical_for_term(self, term: str) -> str:
            """Get the canonical (representative) form for a term's cluster."""
            cluster = self.get_cluster_for_term(term)
            if cluster:
                # Elect canonical: shortest term, or most frequent in corpus
                return min(cluster, key=lambda x: (len(x), x))
            return term.lower()
        
        def merge_with_existing_families(self, 
                                       existing_families: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
            """
            Merge discovered clusters with pre-defined concept families.
            
            Args:
                existing_families: Dict of {canonical: set(aliases)}
                
            Returns:
                Updated families dict with merged clusters
            """
            merged = {canon: set(aliases) for canon, aliases in existing_families.items()}
            
            for cluster in self.clusters:
                # Find best matching existing family
                best_canon = None
                best_overlap = 0
                
                for canon, aliases in merged.items():
                    overlap = len(cluster.intersection(aliases))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_canon = canon
                
                if best_overlap > 0:
                    # Merge into existing family
                    merged[best_canon].update(cluster)
                    logger.debug(f"Merged cluster into '{best_canon}': +{len(cluster) - best_overlap} new aliases")
                else:
                    # Create new family from cluster
                    # Elect canonical: shortest or first alphabetically
                    candidates = sorted(cluster, key=lambda x: (len(x), x))
                    new_canon = candidates[0]
                    merged[new_canon] = cluster
                    logger.debug(f"Created new family '{new_canon}' with {len(cluster)} aliases")
            
            return merged
        
        def get_cluster_summary(self) -> Dict[str, Any]:
            """Get summary statistics about discovered clusters."""
            return {
                "num_clusters": len(self.clusters),
                "avg_cluster_size": np.mean([len(c) for c in self.clusters]) if self.clusters else 0,
                "max_cluster_size": max(len(c) for c in self.clusters) if self.clusters else 0,
                "min_cluster_size": min(len(c) for c in self.clusters) if self.clusters else 0,
                "total_terms_clustered": sum(len(c) for c in self.clusters),
                "clusters": [
                    {
                        "size": len(c),
                        "representative": min(c, key=len),
                        "sample_aliases": list(c)[:5]
                    }
                    for c in sorted(self.clusters, key=len, reverse=True)[:10]
                ]
            }


# =============================================================================
# LAYER 3: UNIFIED CONCEPT REGISTRY - CENTRAL CONCEPT MANAGEMENT
# =============================================================================

class UnifiedConceptRegistry:
    """
    Central registry for unified concept families.
    
    Manages the mapping between surface forms and canonical concepts,
    integrating both pre-defined aliases (Layer 1) and discovered clusters (Layer 2).
    
    Provides:
    - Concept unification: normalize any surface form to canonical
    - Synonym expansion: get all known variants of a concept
    - Query expansion: expand query terms with synonyms for retrieval
    - Salience management: track importance scores per concept family
    - Embedding centroids: mean embedding for semantic similarity
    """
    
    def __init__(self, 
                 embed_model=None,
                 normalizer: Optional[ConceptNormalizer] = None,
                 similarity_threshold: float = 0.75):
        """
        Initialize the concept registry.
        
        Args:
            embed_model: Optional embedding model for centroid computation
            normalizer: Optional pre-configured ConceptNormalizer
            similarity_threshold: Threshold for semantic similarity matching
        """
        self.embed_model = embed_model
        self.normalizer = normalizer or ConceptNormalizer()
        self.sim_threshold = similarity_threshold
        
        # Core data structures
        self.families: Dict[str, Dict[str, Any]] = {}  # canonical -> family info
        self._alias_to_canonical: Dict[str, str] = {}  # alias -> canonical
        self._canonical_embeddings: Dict[str, np.ndarray] = {}  # cached centroids
        
        # Build initial families from alias dictionaries
        self._build_initial_families()
        
        logger.info(f"UnifiedConceptRegistry initialized with {len(self.families)} concept families")
    
    def _build_initial_families(self):
        """Initialize families from MATERIAL_ALIASES, METHOD_ALIASES, and LASER_KEYWORDS."""
        
        def add_family(canonical: str, aliases: List[str], domain: Optional[str] = None):
            """Add or update a concept family."""
            canon_clean = canonical.lower().strip()
            
            if canon_clean not in self.families:
                self.families[canon_clean] = {
                    "canonical": canon_clean,
                    "aliases": set(),
                    "domain": domain,
                    "centroid": None,
                    "salience": 0.5,  # Default salience
                    "frequency": 0,
                    "sources": defaultdict(int),  # doc_id -> count
                }
            
            # Add all aliases
            for alias in aliases:
                alias_clean = alias.lower().strip()
                if alias_clean:
                    self.families[canon_clean]["aliases"].add(alias_clean)
                    self._alias_to_canonical[alias_clean] = canon_clean
            
            # Ensure canonical is in its own aliases
            self.families[canon_clean]["aliases"].add(canon_clean)
            self._alias_to_canonical[canon_clean] = canon_clean
        
        # Add material families
        for canonical, aliases in MATERIAL_ALIASES.items():
            add_family(canonical, aliases, domain="MATERIAL")
        
        # Add method families
        for canonical, aliases in METHOD_ALIASES.items():
            add_family(canonical, aliases, domain="METHOD")
        
        # Add laser topics
        for topic, keywords in LASER_KEYWORDS.items():
            add_family(topic, keywords, domain="TOPIC")
        
        # Add core pillars with high initial salience
        core_pillars = ["laser", "microstructure", "interaction", "multicomponent alloy"]
        for pillar in core_pillars:
            if pillar in self.families:
                self.families[pillar]["salience"] = 1.0
                self.families[pillar]["is_core_pillar"] = True
    
    def unify(self, term: str, context: Optional[str] = None) -> str:
        """
        Unify a surface form to its canonical concept.
        
        Applies Layer 1 (lexical) then Layer 2 (semantic) normalization.
        
        Args:
            term: Surface form to normalize
            context: Optional surrounding context for disambiguation
            
        Returns:
            Canonical concept form
        """
        if not term:
            return term
        
        term_clean = term.lower().strip()
        
        # Layer 1: Lexical normalization
        lex_normalized = self.normalizer.normalize(term_clean, context)
        
        # Check if we have this in our registry
        if lex_normalized in self._alias_to_canonical:
            return self._alias_to_canonical[lex_normalized]
        
        # Layer 2: Semantic similarity to known centroids
        if self.embed_model and term_clean not in self._alias_to_canonical:
            best_match = self._find_semantic_match(term_clean)
            if best_match:
                return best_match
        
        # No match found - return lexical normalization result
        return lex_normalized
    
    def _find_semantic_match(self, term: str, threshold: Optional[float] = None) -> Optional[str]:
        """Find semantically similar canonical concept using embeddings."""
        if not self.embed_model:
            return None
        
        threshold = threshold or self.sim_threshold
        
        try:
            # Embed the query term
            term_emb = self._get_embedding(term)
            
            # Compare to known centroids
            best_sim = 0
            best_canon = None
            
            for canon, family in self.families.items():
                centroid = self._get_centroid(canon)
                if centroid is not None:
                    sim = cosine_similarity([term_emb], [centroid])[0][0]
                    if sim > best_sim and sim >= threshold:
                        best_sim = sim
                        best_canon = canon
            
            return best_canon
            
        except Exception as e:
            logger.debug(f"Semantic matching failed for '{term}': {e}")
            return None
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text string."""
        if hasattr(self.embed_model, 'embed_query'):
            return np.array(self.embed_model.embed_query(text))
        elif hasattr(self.embed_model, 'encode'):
            return self.embed_model.encode(text)
        else:
            raise ValueError("Embedding model has no suitable method")
    
    def _get_centroid(self, canonical: str) -> Optional[np.ndarray]:
        """Get or compute the embedding centroid for a concept family."""
        if canonical in self._canonical_embeddings:
            return self._canonical_embeddings[canonical]
        
        if canonical not in self.families:
            return None
        
        family = self.families[canonical]
        
        if not self.embed_model or not family["aliases"]:
            return None
        
        try:
            # Compute mean of all alias embeddings
            aliases = list(family["aliases"])
            embeddings = [self._get_embedding(a) for a in aliases]
            centroid = np.mean(embeddings, axis=0)
            
            # Cache and return
            self._canonical_embeddings[canonical] = centroid
            family["centroid"] = centroid
            return centroid
            
        except Exception as e:
            logger.debug(f"Failed to compute centroid for '{canonical}': {e}")
            return None
    
    def get_family(self, canonical: str) -> Optional[Dict[str, Any]]:
        """Get full family info for a canonical concept."""
        return self.families.get(canonical.lower())
    
    def get_all_aliases(self, canonical: str) -> FrozenSet[str]:
        """Get all known surface forms for a canonical concept."""
        family = self.get_family(canonical)
        if family:
            return frozenset(family["aliases"])
        return frozenset({canonical})
    
    def expand_query_terms(self, terms: List[str]) -> Set[str]:
        """
        Expand query terms with all known synonyms for retrieval.
        
        Args:
            terms: List of query terms
            
        Returns:
            Set of all terms including synonyms
        """
        expanded = set()
        
        for term in terms:
            # Get canonical form
            canon = self.unify(term)
            
            # Add canonical and all its aliases
            expanded.add(canon)
            expanded.update(self.get_all_aliases(canon))
        
        return expanded
    
    def set_salience(self, canonical: str, salience: float):
        """Update salience score for a concept family."""
        canon_clean = canonical.lower().strip()
        if canon_clean in self.families:
            self.families[canon_clean]["salience"] = min(1.0, max(0.0, salience))
    
    def update_centroids(self):
        """Recompute centroids for all families (call after adding new aliases)."""
        for canon in self.families:
            self._get_centroid(canon)  # This will recompute if needed
    
    def merge_from_clusterer(self, clusterer: 'SemanticClusterer'):
        """
        Merge discovered synonym clusters into the registry.
        
        Args:
            clusterer: Configured SemanticClusterer with discovered clusters
        """
        if not clusterer.clusters:
            return
        
        logger.info(f"Merging {len(clusterer.clusters)} discovered clusters into registry")
        
        for cluster in clusterer.clusters:
            # Find best matching existing family
            best_canon = None
            best_overlap = 0
            
            for canon, family in self.families.items():
                overlap = len(cluster.intersection(family["aliases"]))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_canon = canon
            
            if best_overlap > 0:
                # Merge into existing family
                self.families[best_canon]["aliases"].update(cluster)
                for alias in cluster:
                    self._alias_to_canonical[alias.lower()] = best_canon
                logger.debug(f"Merged cluster into '{best_canon}'")
            else:
                # Create new family
                candidates = sorted(cluster, key=lambda x: (len(x), x))
                new_canon = candidates[0]
                
                self.families[new_canon] = {
                    "canonical": new_canon,
                    "aliases": set(cluster),
                    "domain": None,
                    "centroid": None,
                    "salience": 0.5,
                    "frequency": 0,
                    "sources": defaultdict(int),
                    "discovered": True,  # Mark as auto-discovered
                }
                
                for alias in cluster:
                    self._alias_to_canonical[alias.lower()] = new_canon
                
                logger.debug(f"Created new family '{new_canon}' from discovered cluster")
        
        # Recompute centroids for updated families
        self.update_centroids()
    
    def get_concept_graph(self) -> nx.Graph:
        """Build a graph of concept relationships for visualization."""
        G = nx.Graph()
        
        # Add nodes for each canonical concept
        for canon, family in self.families.items():
            G.add_node(canon, 
                      canonical=canon,
                      aliases=len(family["aliases"]),
                      salience=family["salience"],
                      domain=family.get("domain"),
                      is_core=family.get("is_core_pillar", False))
        
        # Add edges based on co-occurrence in alias sets
        # (Two concepts are related if they share aliases - indicates potential overlap)
        canonicals = list(self.families.keys())
        for i, canon1 in enumerate(canonicals):
            for canon2 in canonicals[i+1:]:
                aliases1 = self.families[canon1]["aliases"]
                aliases2 = self.families[canon2]["aliases"]
                overlap = len(aliases1.intersection(aliases2))
                
                if overlap > 0:
                    G.add_edge(canon1, canon2, weight=overlap, overlap=overlap)
        
        return G
    
    def to_dict(self) -> Dict[str, Any]:
        """Export registry state for serialization."""
        return {
            "families": {
                canon: {
                    "canonical": info["canonical"],
                    "aliases": list(info["aliases"]),
                    "domain": info.get("domain"),
                    "salience": info["salience"],
                    "frequency": info["frequency"],
                    "is_core_pillar": info.get("is_core_pillar", False),
                    "discovered": info.get("discovered", False),
                }
                for canon, info in self.families.items()
            },
            "alias_count": len(self._alias_to_canonical),
            "total_aliases": sum(len(f["aliases"]) for f in self.families.values()),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], embed_model=None) -> 'UnifiedConceptRegistry':
        """Load registry from serialized dict."""
        registry = cls(embed_model=embed_model)
        
        for canon, info in data.get("families", {}).items():
            registry.families[canon] = {
                "canonical": info["canonical"],
                "aliases": set(info["aliases"]),
                "domain": info.get("domain"),
                "centroid": None,  # Will be recomputed if needed
                "salience": info["salience"],
                "frequency": info.get("frequency", 0),
                "sources": defaultdict(int),
                "is_core_pillar": info.get("is_core_pillar", False),
                "discovered": info.get("discovered", False),
            }
            
            for alias in info["aliases"]:
                registry._alias_to_canonical[alias.lower()] = canon
        
        return registry


# =============================================================================
# ENHANCED SCIENTIFIC ENTITY & CLAIM (using unified concepts)
# =============================================================================

@dataclass
class EnhancedScientificEntity:
    """Scientific entity with unified concept tracking."""
    text: str  # Original surface form
    label: str  # Entity type (MATERIAL, METHOD, etc.)
    value: Optional[float]  # Numeric value if applicable
    unit: Optional[str]  # Unit of measurement
    doc_source: str  # Source document
    chunk_id: int  # Chunk index
    context: str  # Surrounding text
    
    # Computed fields
    confidence: float = 1.0
    canonical: str = field(init=False)  # Unified concept form
    domain: str = field(init=False)
    category: str = field(init=False)
    subcategory: str = field(init=False)
    
    def __post_init__(self):
        """Post-initialization: compute canonical form and taxonomy."""
        self.canonical = normalize_concept(self.text)
        self.domain, self.category, self.subcategory = classify_entity(self.canonical)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "label": self.label,
            "value": self.value,
            "unit": self.unit,
            "doc_source": self.doc_source,
            "chunk_id": self.chunk_id,
            "context": self.context[:200],  # Truncate for storage
            "canonical": self.canonical,
            "confidence": self.confidence,
            "domain": self.domain,
            "category": self.category,
            "subcategory": self.subcategory,
        }


@dataclass
class EnhancedScientificClaim:
    """Scientific claim with unified concept subjects/objects."""
    claim_text: str
    subject: str  # Will be unified
    predicate: str
    object_val: str  # Will be unified
    doc_source: str
    chunk_id: int
    confidence: float
    
    # Tracking
    supporting: List[Tuple[str, int]] = field(default_factory=list)
    contradicting: List[Tuple[str, int]] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization: unify subject and object."""
        self.subject = normalize_concept(self.subject)
        self.object_val = normalize_concept(self.object_val)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "claim": self.claim_text,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object_val,
            "source": self.doc_source,
            "confidence": self.confidence,
            "supporting_count": len(self.supporting),
            "contradicting_count": len(self.contradicting),
        }


# =============================================================================
# GLOBAL CONCEPT NORMALIZER INSTANCE
# =============================================================================

_concept_normalizer: Optional[ConceptNormalizer] = None
_concept_registry: Optional[UnifiedConceptRegistry] = None


def get_concept_normalizer() -> ConceptNormalizer:
    """Get or create the global concept normalizer."""
    global _concept_normalizer
    if _concept_normalizer is None:
        _concept_normalizer = ConceptNormalizer()
    return _concept_normalizer


def get_concept_registry(embed_model=None) -> UnifiedConceptRegistry:
    """Get or create the global concept registry."""
    global _concept_registry
    if _concept_registry is None:
        _concept_registry = UnifiedConceptRegistry(embed_model=embed_model)
    return _concept_registry


def normalize_concept(text: str, context: Optional[str] = None) -> str:
    """Convenience function: normalize text to canonical concept."""
    return get_concept_normalizer().normalize(text, context)


def classify_entity(normalized: str) -> Tuple[str, str, str]:
    """Return (domain, category, subcategory) for a canonical entity."""
    norm = normalized.lower().strip()
    
    def _search_level(node, path):
        """Recursively search taxonomy."""
        if isinstance(node, list):
            if any(alias in norm for alias in node):
                while len(path) < 3:
                    path.append("General")
                return tuple(path[:3])
            return None
        elif isinstance(node, dict):
            for key, child in node.items():
                result = _search_level(child, path + [key])
                if result is not None:
                    return result
            return None
        else:
            return None
    
    for domain, categories in ENTITY_TAXONOMY.items():
        result = _search_level(categories, [domain])
        if result is not None:
            return result
    
    return "UNKNOWN", "UNKNOWN", "UNKNOWN"


# =============================================================================
# FULL-TEXT CONCEPT EXTRACTOR WITH UNIFIED CONCEPTS
# =============================================================================

class FullTextConceptExtractor:
    """
    Extracts scientific concepts from full-text with unified concept handling.
    
    Uses the three-layer architecture:
    1. Lexical normalization via ConceptNormalizer
    2. Semantic clustering via SemanticClusterer (optional)
    3. Registry-based unification via UnifiedConceptRegistry
    
    Computes multi-factor salience scores per concept family.
    """
    
    def __init__(self, 
                 embed_model,
                 registry: UnifiedConceptRegistry,
                 proposal_text: str = None):
        """
        Initialize the extractor.
        
        Args:
            embed_model: Embedding model for semantic operations
            registry: UnifiedConceptRegistry for concept management
            proposal_text: Domain proposal text for salience seeding
        """
        self.embed_model = embed_model
        self.registry = registry
        self.proposal_text = proposal_text or DECLARMIMA_PROPOSAL_TEXT
        
        # Pre-compute proposal embedding
        self.proposal_embedding = self._embed_text(self.proposal_text)
        
        # Core pillars with high salience
        self.core_pillars = {
            "laser": 1.00,
            "microstructure": 1.00,
            "interaction": 1.00,
            "multicomponent alloy": 1.00,
            "multicomponent": 0.98,
            "alloy": 0.95,
            "laser microstructure interaction": 1.00,
            "laser-matter interaction": 1.00,
            "laser alloy interaction": 0.98,
            "laser multicomponent interaction": 1.00,
            "melt pool": 0.96,
            "keyhole": 0.95,
            "marangoni convection": 0.94,
            "additive manufacturing": 0.93,
            "solidification": 0.92,
            "intermetallic compound": 0.91,
            "residual stress": 0.90,
            "porosity": 0.89,
            "spatter": 0.88,
            "grain morphology": 0.87,
        }
        
        # Domain seeds
        self.domain_seeds = {
            "melt pool": 0.95, "keyhole": 0.94, "marangoni convection": 0.92,
            "porosity": 0.90, "spatter": 0.88, "intermetallic compound": 0.90,
            "columnar to equiaxed": 0.87, "residual stress": 0.88,
            "solidification": 0.85, "grain morphology": 0.82,
            "multicomponent alloy": 0.94, "high entropy alloy": 0.94, "hea": 0.94,
            "complex concentrated alloy": 0.92, "cocrfeni": 0.90, "alcocrfeni": 0.90,
            "refractory hea": 0.89, "sn-ag-cu": 0.85, "sac solder": 0.85,
            "inconel 718": 0.85, "ti-6al-4v": 0.85, "scan speed": 0.84,
            "hatch distance": 0.83, "laser power": 0.85, "pulse duration": 0.82,
            "thermal conductivity": 0.81, "interfacial energy": 0.80,
            "viscosity": 0.79, "diffusion coefficient": 0.78, "absorptivity": 0.77,
            "phase field": 0.86, "molecular dynamics": 0.85, "finite element": 0.84,
            "calphad": 0.83, "machine learning": 0.82, "digital twin": 0.81,
        }
        
        self.section_weights = {
            "RESULTS": 1.00, "DISCUSSION": 0.92, "CONCLUSION": 0.88,
            "ABSTRACT": 0.75, "INTRODUCTION": 0.65, "METHODS": 0.40,
            "BODY": 0.55, "UNKNOWN": 0.30
        }
        
        self.custom_priority: Dict[str, float] = {}
        self._pillar_embeddings: Dict[str, np.ndarray] = {}
        self._semantic_boost_threshold = 0.72
        self._semantic_boost_factor = 0.35
    
    def _embed_text(self, text: str) -> np.ndarray:
        """Embed text using configured model."""
        if hasattr(self.embed_model, 'embed_query'):
            return np.array(self.embed_model.embed_query(text))
        elif hasattr(self.embed_model, 'encode'):
            return self.embed_model.encode(text)
        else:
            raise AttributeError("Embedding model has neither 'embed_query' nor 'encode' method")
    
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed batch of texts."""
        if hasattr(self.embed_model, 'embed_documents'):
            return np.array(self.embed_model.embed_documents(texts))
        elif hasattr(self.embed_model, 'encode'):
            return self.embed_model.encode(texts, show_progress_bar=False)
        else:
            return np.array([self._embed_text(t) for t in texts])
    
    def set_custom_priority(self, concepts: List[str]):
        """Set user-defined priority concepts."""
        if concepts:
            self.custom_priority = {c.lower().strip(): 0.88 for c in concepts if c.strip()}
        else:
            self.custom_priority = {}
    
    def _compute_semantic_pillar_embeddings(self):
        """Pre-compute embeddings for core pillars."""
        if not self._pillar_embeddings:
            for pillar in self.core_pillars:
                self._pillar_embeddings[pillar] = self._embed_text(pillar)
    
    def _get_semantic_boost(self, concept: str, concept_embedding: np.ndarray) -> float:
        """Compute semantic similarity boost for pillar relatives."""
        self._compute_semantic_pillar_embeddings()
        max_sim = 0.0
        for pillar, pillar_emb in self._pillar_embeddings.items():
            sim = float(np.dot(concept_embedding, pillar_emb) /
                       (np.linalg.norm(concept_embedding) * np.linalg.norm(pillar_emb) + 1e-8))
            if sim > max_sim:
                max_sim = sim
        if max_sim >= self._semantic_boost_threshold:
            return self._semantic_boost_factor * max_sim
        return 0.0
    
    def _extract_candidates(self, chunks: List[Document]) -> List[str]:
        """Extract candidate concepts from chunks."""
        candidates = set()
        
        for chunk in chunks:
            text = chunk.page_content.lower()
            
            # Match LASER_KEYWORDS
            for topic, keywords in LASER_KEYWORDS.items():
                for kw in keywords:
                    if kw.lower() in text:
                        candidates.add(kw.lower())
            
            # Match MATERIAL_ALIASES and METHOD_ALIASES
            for canonical in list(MATERIAL_ALIASES.keys()) + list(METHOD_ALIASES.keys()):
                if canonical.lower() in text:
                    candidates.add(canonical.lower())
            
            # Extract quantity contexts
            for match in re.finditer(r'(\d+(?:\.\d+)?)\s*(?:μm|um|nm|%|J/mm³|HV|MPa|W|mm/s)', text):
                context = text[max(0, match.start()-60):match.end()+60]
                candidates.add(context.strip()[:80])
            
            # Multicomponent alloy patterns
            mca_patterns = [
                r'\b(?:multicomponent|multi-component|multi\s+component)\s+(?:alloy|system|material|metallic)\b',
                r'\b(?:high\s+entropy|complex\s+concentrated)\s+(?:alloy|alloys)\b',
                r'\b(?:multi-principal|multiprincipal)\s+(?:element|elemental)\s+(?:alloy|alloys)\b',
                r'\b(?:cocrfeni|alcocrfeni|crmnfeconi|alcrfeni|cocrfenimn)\b',
                r'\b(?:hea|mpea|cca)\s+(?:system|alloy|family|composition)\b',
            ]
            for pattern in mca_patterns:
                for match in re.finditer(pattern, text, re.I):
                    candidates.add(match.group(0).lower().strip()[:60])
        
        return list(candidates)
    
    def extract_concepts(self, 
                        chunks: List[Document], 
                        min_salience: float = 0.42) -> Tuple[List[str], Dict[str, Dict]]:
        """
        Main extraction: extract concepts and compute salience.
        
        Args:
            chunks: Document chunks to process
            min_salience: Minimum salience threshold for inclusion
            
        Returns:
            (list of canonical concepts, dict of metadata per concept)
        """
        candidates = self._extract_candidates(chunks)
        
        # Map candidates to canonical forms via registry
        canon_map = {c: self.registry.unify(c) for c in candidates}
        
        # Group by canonical form
        canon_candidates = defaultdict(list)
        for c, canon in canon_map.items():
            canon_candidates[canon].append(c)
        
        salience_scores = {}
        
        for canon, aliases in canon_candidates.items():
            # Frequency: count chunks where any alias appears
            freq = 0
            for ch in chunks:
                ch_text = ch.page_content.lower()
                if any(alias in ch_text for alias in aliases):
                    freq += 1
            
            freq_norm = np.log1p(freq) / np.log1p(len(chunks)) if chunks else 0
            
            # Cross-document presence
            docs_with = set()
            for ch in chunks:
                if any(alias in ch.page_content.lower() for alias in aliases):
                    docs_with.add(ch.metadata.get("source", "unknown"))
            
            total_docs = len(set(ch.metadata.get("source", "") for ch in chunks))
            cross_doc = len(docs_with) / total_docs if total_docs > 0 else 0
            
            # Section importance
            section_scores = []
            for ch in chunks:
                if any(alias in ch.page_content.lower() for alias in aliases):
                    section_scores.append(
                        self.section_weights.get(
                            ch.metadata.get("section", "UNKNOWN").upper(), 0.3
                        )
                    )
            section_imp = np.mean(section_scores) if section_scores else 0.3
            
            # Proposal similarity
            if canon in self.registry.families and self.registry.families[canon]["centroid"] is not None:
                emb = self.registry.families[canon]["centroid"]
            else:
                emb = self._embed_text(canon)
            
            proposal_sim = float(np.dot(emb, self.proposal_embedding) /
                               (np.linalg.norm(emb) * np.linalg.norm(self.proposal_embedding) + 1e-8))
            
            # Base salience computation
            base_salience = (
                0.25 * freq_norm +
                0.20 * cross_doc +
                0.18 * section_imp +
                0.15 * proposal_sim +
                0.12 * 0.6  # Default for quantitative signal
            )
            
            # Boost factors
            boost = max(
                self.core_pillars.get(canon.lower(), 0),
                self.domain_seeds.get(canon.lower(), 0),
                self.custom_priority.get(canon.lower(), 0)
            )
            
            semantic_boost = self._get_semantic_boost(canon, emb)
            
            # Final score
            final_score = base_salience * (1 + 0.65 * boost + semantic_boost)
            
            if final_score >= min_salience or boost >= 0.8:
                salience_scores[canon] = final_score
                self.registry.set_salience(canon, final_score)
        
        # Sort by salience
        final_concepts = sorted(salience_scores.keys(), 
                               key=lambda c: salience_scores[c], 
                               reverse=True)
        
        # Build metadata
        metadata = {}
        for c in final_concepts:
            metadata[c] = {
                "salience": salience_scores[c],
                "is_core_pillar": c.lower() in self.core_pillars,
                "is_domain_seed": c.lower() in self.domain_seeds,
                "is_custom": c.lower() in self.custom_priority,
                "frequency": sum(1 for ch in chunks if c.lower() in ch.page_content.lower()),
                "aliases": list(self.registry.get_all_aliases(c)),
            }
        
        return final_concepts, metadata


# =============================================================================
# ENHANCED CROSS-DOCUMENT KNOWLEDGE GRAPH (with unified concepts)
# =============================================================================

class EnhancedCrossDocumentKnowledgeGraph:
    """
    Knowledge graph with unified concept families.
    
    Stores entities and claims using canonical forms, enabling:
    - Consensus detection across synonym variants
    - Unified visualization of concept families
    - Query expansion for retrieval
    """
    
    def __init__(self, registry: UnifiedConceptRegistry):
        """Initialize with concept registry."""
        self.registry = registry
        self.entities: Dict[str, List[EnhancedScientificEntity]] = defaultdict(list)
        self.claims: List[EnhancedScientificClaim] = []
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)
        self.chunk_index: Dict[str, List[Document]] = defaultdict(list)
        self.concept_metadata: Dict[str, Dict] = {}
        self.dgl_graph = None
        self.dgl_node_maps = {}
        self.entity_embeddings = None
        self._entity_list = []
    
    def add_document(self, 
                    doc_id: str, 
                    chunks: List[Document], 
                    bib_meta: Any,
                    concept_metadata: Optional[Dict[str, Dict]] = None):
        """
        Add a document's chunks to the graph.
        
        Args:
            doc_id: Document identifier
            chunks: List of Document chunks
            bib_meta: Bibliographic metadata
            concept_metadata: Optional salience metadata from extractor
        """
        self.documents[doc_id] = {
            "bib_meta": bib_meta.to_dict() if hasattr(bib_meta, 'to_dict') else {},
            "chunk_count": len(chunks),
            "topics": set(),
            "years": getattr(bib_meta, 'year', None)
        }
        self.chunk_index[doc_id] = chunks
        
        for i, chunk in enumerate(chunks):
            # Extract and unify entities
            raw_entities = self._extract_entities_from_chunk(chunk, i)
            for ent in raw_entities:
                # Unify to canonical form via registry
                canonical = self.registry.unify(ent.canonical)
                ent.canonical = canonical
                
                self.entities[canonical].append(ent)
                self.entity_index[canonical].add(doc_id)
                self.documents[doc_id]["topics"].add(ent.label)
            
            # Extract and unify claims
            claims = self._extract_claims_from_chunk(chunk, i)
            for claim in claims:
                claim.subject = self.registry.unify(claim.subject)
                claim.object_val = self.registry.unify(claim.object_val)
                self.claims.append(claim)
        
        # Merge concept metadata
        if concept_metadata:
            for concept, meta in concept_metadata.items():
                canon = self.registry.unify(concept)
                if canon not in self.concept_metadata:
                    self.concept_metadata[canon] = meta
                else:
                    # Keep higher salience, accumulate frequency
                    if meta.get("salience", 0) > self.concept_metadata[canon].get("salience", 0):
                        self.concept_metadata[canon]["salience"] = meta["salience"]
                    self.concept_metadata[canon]["frequency"] = (
                        self.concept_metadata[canon].get("frequency", 0) + 
                        meta.get("frequency", 0)
                    )
    
    def _extract_entities_from_chunk(self, chunk: Document, chunk_id: int) -> List[EnhancedScientificEntity]:
        """Extract entities from a chunk."""
        text = chunk.page_content
        doc = chunk.metadata.get("source", "unknown")
        entities = []
        
        # Extract quantity patterns
        for param_name, pattern in QUANTITY_PATTERNS.items():
            for match in pattern.finditer(text):
                val_str = match.group(1)
                try:
                    val = float(val_str)
                except:
                    val = None
                
                unit_match = re.search(r'(nm|µm|um|fs|ps|ns|J/cm²|J/cm2|kHz|MHz|W|mW|mJ|µJ|uJ)', 
                                      match.group(0), re.I)
                unit = unit_match.group(1) if unit_match else None
                
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end].replace('\n', ' ')
                
                entities.append(EnhancedScientificEntity(
                    text=match.group(0), label=param_name, value=val, unit=unit,
                    doc_source=doc, chunk_id=chunk_id, context=context, confidence=0.85
                ))
        
        # Extract material/method aliases
        text_lower = text.lower()
        for canonical, aliases in {**MATERIAL_ALIASES, **METHOD_ALIASES}.items():
            for alias in aliases:
                for match in re.finditer(r'\b' + re.escape(alias) + r'\b', text_lower):
                    start = max(0, match.start() - 80)
                    end = min(len(text), match.end() + 80)
                    context = text[start:end]
                    lbl = "MATERIAL" if canonical in MATERIAL_ALIASES else "METHOD"
                    
                    entities.append(EnhancedScientificEntity(
                        text=alias, label=lbl, value=None, unit=None,
                        doc_source=doc, chunk_id=chunk_id, context=context, confidence=0.9
                    ))
        
        # Extract laser topics
        for topic, keywords in LASER_KEYWORDS.items():
            for kw in keywords:
                for match in re.finditer(r'\b' + re.escape(kw.lower()) + r'\b', text_lower):
                    start = max(0, match.start() - 80)
                    end = min(len(text), match.end() + 80)
                    
                    entities.append(EnhancedScientificEntity(
                        text=kw, label=topic, value=None, unit=None,
                        doc_source=doc, chunk_id=chunk_id, context=text[start:end], confidence=0.8
                    ))
        
        return entities
    
    def _extract_claims_from_chunk(self, chunk: Document, chunk_id: int) -> List[EnhancedScientificClaim]:
        """Extract scientific claims from a chunk."""
        text = chunk.page_content
        doc = chunk.metadata.get("source", "unknown")
        claims = []
        
        claim_patterns = [
            (r'(?:ablation\s*threshold|threshold\s*fluence)\s*(?:of|for)\s+([a-z\s\-]+?)\s+(?:is|was|were|are|≈|~|about)\s+(\d+\.?\d*\s*[A-Za-z/²²]+)', 'has_ablation_threshold'),
            (r'([a-z\s\-]+?)\s+(?:exhibits|shows|displays|forms|produces)\s+([a-z\s\-]+?(?:ripples|LIPSS|structures|morphology))', 'exhibits_morphology'),
            (r'(?:periodicity|period|spacing)\s*(?:of|for)\s+([a-z\s\-]+?)\s+(?:is|was|≈|~)\s+(\d+\.?\d*\s*(?:nm|µm|um))', 'has_periodicity'),
            (r'(?:roughness|Ra)\s*(?:of|for)\s+([a-z\s\-]+?)\s+(?:is|was|≈|~)\s+(\d+\.?\d*\s*(?:nm|µm|um))', 'has_roughness'),
            (r'([a-z\s\-]+?)\s+(?:increases|decreases|reduces|enhances|promotes|suppresses)\s+([a-z\s\-]+?(?:growth|formation|porosity|cracking|stress))', 'causes_effect'),
        ]
        
        for pattern, predicate in claim_patterns:
            for match in re.finditer(pattern, text, re.I):
                subject = match.group(1).strip()
                obj = match.group(2).strip()
                start = max(0, match.start() - 120)
                end = min(len(text), match.end() + 120)
                context = text[start:end]
                
                claims.append(EnhancedScientificClaim(
                    claim_text=context, subject=subject, predicate=predicate,
                    object_val=obj, doc_source=doc, chunk_id=chunk_id, confidence=0.7
                ))
        
        return claims
    
    def find_consensus(self, entity_normalized: str) -> Optional[Dict[str, Any]]:
        """Find consensus values for a canonical entity across documents."""
        ents = self.entities.get(entity_normalized, [])
        if len(ents) < 2:
            return None
        
        # Group by document
        by_doc = defaultdict(list)
        for e in ents:
            by_doc[e.doc_source].append(e)
        
        if len(by_doc) < 2:
            return None
        
        # Extract numeric values
        values = [e.value for e in ents if e.value is not None]
        if not values:
            return None
        
        return {
            "entity": entity_normalized,
            "domain": ents[0].domain,
            "category": ents[0].category,
            "subcategory": ents[0].subcategory,
            "doc_count": len(by_doc),
            "value_count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "unit": ents[0].unit,
            "sources": list(by_doc.keys()),
            "values_by_doc": {
                d: [e.value for e in ev if e.value is not None] 
                for d, ev in by_doc.items()
            }
        }
    
    def find_contradictions(self, 
                           entity_normalized: str, 
                           threshold_factor: float = 2.0) -> List[Dict[str, Any]]:
        """Find contradictory values for an entity across documents."""
        ents = self.entities.get(entity_normalized, [])
        by_doc = defaultdict(list)
        
        for e in ents:
            if e.value is not None:
                by_doc[e.doc_source].append(e.value)
        
        contradictions = []
        docs = list(by_doc.keys())
        
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                vals_i, vals_j = by_doc[docs[i]], by_doc[docs[j]]
                mean_i, mean_j = np.mean(vals_i), np.mean(vals_j)
                
                if mean_i > 0 and mean_j > 0:
                    ratio = max(mean_i, mean_j) / min(mean_i, mean_j)
                    if ratio > threshold_factor:
                        contradictions.append({
                            "entity": entity_normalized,
                            "doc_a": docs[i],
                            "mean_a": float(mean_i),
                            "std_a": float(np.std(vals_i)),
                            "doc_b": docs[j],
                            "mean_b": float(mean_j),
                            "std_b": float(np.std(vals_j)),
                            "ratio": float(ratio),
                            "severity": "critical" if ratio > 10 else "high" if ratio > 5 else "moderate"
                        })
        
        return contradictions
    
    def find_all_consensus(self, min_docs: int = 2) -> List[Dict[str, Any]]:
        """Find consensus for all entities with sufficient documentation."""
        results = []
        for ent_norm in self.entities:
            cons = self.find_consensus(ent_norm)
            if cons and cons["doc_count"] >= min_docs:
                results.append(cons)
        return sorted(results, key=lambda x: x["doc_count"], reverse=True)
    
    def find_all_contradictions(self, threshold_factor: float = 2.0) -> List[Dict[str, Any]]:
        """Find all contradictions above threshold."""
        results = []
        seen = set()
        
        for ent_norm in self.entities:
            contrs = self.find_contradictions(ent_norm, threshold_factor)
            for c in contrs:
                key = tuple(sorted([c["doc_a"], c["doc_b"]]) + [c["entity"]])
                if key not in seen:
                    results.append(c)
                    seen.add(key)
        
        return sorted(results, key=lambda x: x["ratio"], reverse=True)
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the knowledge graph."""
        return {
            "total_entities": sum(len(v) for v in self.entities.values()),
            "unique_entities": len(self.entities),
            "total_claims": len(self.claims),
            "document_count": len(self.documents),
            "top_entities": Counter(
                [e.canonical for ents in self.entities.values() for e in ents]
            ).most_common(15),
            "high_salience_concepts": sorted(
                self.concept_metadata.items(),
                key=lambda x: x[1].get("salience", 0),
                reverse=True
            )[:10],
            "consensus_topics": [
                k for k, v in self.entities.items() 
                if len(self.entity_index.get(k, set())) > 1
            ],
            "domains": Counter(
                [e.domain for ents in self.entities.values() for e in ents]
            ).most_common(),
            "categories": Counter(
                [e.category for ents in self.entities.values() for e in ents]
            ).most_common(),
        }


# =============================================================================
# REASONING CHAIN (THINKING TRACE)
# =============================================================================

@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""
    step_type: str
    description: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class ReasoningChain:
    """Explicit chain-of-thought for cross-document synthesis."""
    
    def __init__(self, query: str):
        self.query = query
        self.steps: List[ReasoningStep] = []
        self.thinking_graph: Optional[nx.DiGraph] = None
    
    def add_step(self, step_type: str, description: str, data: Dict[str, Any]):
        """Add a reasoning step."""
        self.steps.append(ReasoningStep(step_type, description, data))
    
    def build_thinking_graph(self) -> nx.DiGraph:
        """Build a directed graph of the reasoning process."""
        G = nx.DiGraph()
        G.add_node("QUERY", node_type="query", text=self.query, layer=0)
        
        prev_node = "QUERY"
        for i, step in enumerate(self.steps):
            node_id = f"STEP_{i}_{step.step_type}"
            G.add_node(
                node_id, 
                node_type=step.step_type, 
                description=step.description,
                layer=i+1, 
                timestamp=step.timestamp.isoformat()
            )
            G.add_edge(prev_node, node_id, relation="leads_to")
            
            # Add entity nodes if present
            if "entities" in step.data:
                for ent in step.data["entities"]:
                    ent_id = f"ENT_{ent}_{i}"
                    G.add_node(ent_id, node_type="entity", name=ent, layer=i+1)
                    G.add_edge(node_id, ent_id, relation="involves")
            
            # Add chunk nodes if present
            if "chunks" in step.data:
                for chunk_idx, chunk_src in enumerate(step.data["chunks"]):
                    chk_id = f"CHK_{chunk_src}_{chunk_idx}_{i}"
                    G.add_node(chk_id, node_type="chunk", source=chunk_src, layer=i+1)
                    G.add_edge(node_id, chk_id, relation="retrieves")
            
            prev_node = node_id
        
        G.add_node("ANSWER", node_type="answer", layer=len(self.steps)+1)
        G.add_edge(prev_node, "ANSWER", relation="synthesizes")
        
        self.thinking_graph = G
        return G
    
    def to_markdown(self) -> str:
        """Export reasoning chain as markdown."""
        lines = [f"### 🧠 Reasoning Trace: *{self.query}*", ""]
        
        for i, step in enumerate(self.steps, 1):
            lines.append(f"**Step {i} — {step.step_type}**")
            lines.append(f"{step.description}")
            if step.data:
                lines.append(f"```json\n{json.dumps(step.data, default=str)[:300]}\n```")
            lines.append("")
        
        return "\n".join(lines)


# =============================================================================
# GRAPH DIFFUSION RETRIEVER
# =============================================================================

class GraphDiffusionRetriever:
    """Retriever that combines vector similarity with graph diffusion."""
    
    def __init__(self, 
                 graph: EnhancedCrossDocumentKnowledgeGraph, 
                 embedding_fn: Optional[Callable] = None):
        self.graph = graph
        self.embedding_fn = embedding_fn
        self.nx_graph: Optional[nx.Graph] = None
        self._build_nx_fallback()
    
    def _build_nx_fallback(self):
        """Build NetworkX fallback graph for diffusion."""
        G = nx.Graph()
        
        # Add document nodes
        for doc_id in self.graph.documents:
            G.add_node(doc_id, node_type="doc", bipartite=0)
        
        # Add entity nodes and edges
        for ent_norm, ents in self.graph.entities.items():
            G.add_node(ent_norm, node_type="entity", bipartite=1,
                      domain=ents[0].domain if ents else "UNKNOWN")
            for e in ents:
                G.add_edge(e.doc_source, ent_norm, weight=e.confidence)
        
        self.nx_graph = G
    
    def retrieve(self, 
                query: str, 
                query_entities: List[str], 
                chunks: List[Document],
                vector_scores: Dict[int, float], 
                top_k: int = 6,
                alpha: float = 0.5) -> List[Tuple[Document, float, str]]:
        """
        Hybrid retrieval: vector + graph diffusion.
        
        Args:
            query: User query
            query_entities: Extracted entities from query
            chunks: Candidate chunks
            vector_scores: Pre-computed vector similarity scores
            top_k: Number of results to return
            alpha: Weight for vector score (1-alpha for graph score)
            
        Returns:
            List of (chunk, score, reason) tuples
        """
        if not query_entities:
            # Fall back to pure vector retrieval
            sorted_chunks = sorted(
                chunks, 
                key=lambda c: vector_scores.get(c.metadata.get("chunk_index", -1), 0), 
                reverse=True
            )
            return [(c, vector_scores.get(c.metadata.get("chunk_index", -1), 0), "vector-only") 
                    for c in sorted_chunks[:top_k]]
        
        # Compute graph diffusion scores
        diffusion_scores = self._nx_diffusion(query_entities, chunks)
        
        # Combine scores
        hybrid = []
        for chunk in chunks:
            cidx = chunk.metadata.get("chunk_index", -1)
            v_score = vector_scores.get(cidx, 0.0)
            g_score = diffusion_scores.get(cidx, 0.0)
            
            final = alpha * v_score + (1 - alpha) * g_score
            reason = "graph-boosted" if g_score > v_score else "hybrid"
            
            hybrid.append((chunk, final, reason))
        
        hybrid.sort(key=lambda x: x[1], reverse=True)
        return hybrid[:top_k]
    
    def _nx_diffusion(self, 
                     query_entities: List[str], 
                     chunks: List[Document]) -> Dict[int, float]:
        """Personalized PageRank diffusion on NetworkX graph."""
        if self.nx_graph is None:
            return {}
        
        # Build personalization vector
        personalization = {n: 0.0 for n in self.nx_graph.nodes()}
        for ent in query_entities:
            if ent in personalization:
                personalization[ent] = 1.0
        
        if sum(personalization.values()) == 0:
            return {}
        
        try:
            pr = nx.pagerank(
                self.nx_graph, 
                personalization=personalization, 
                weight='weight',
                max_iter=100
            )
        except Exception:
            pr = {}
        
        # Score chunks based on connected entities
        chunk_scores = {}
        for chunk in chunks:
            cidx = chunk.metadata.get("chunk_index", -1)
            doc = chunk.metadata.get("source", "unknown")
            
            score = pr.get(doc, 0.0) * 0.3  # Document importance
            for ent in query_entities:
                score += pr.get(ent, 0.0) * 0.7  # Entity importance
            
            chunk_scores[cidx] = score
        
        return chunk_scores


# =============================================================================
# CROSS-DOCUMENT THINKER (with query expansion & term highlighting)
# =============================================================================

class CrossDocumentThinker:
    """
    Orchestrates cross-document reasoning with concept unification.
    
    Features:
    - Query entity extraction with synonym expansion
    - Hybrid vector+graph retrieval with expanded terms
    - Consensus/contradiction detection across unified concepts
    - Answer generation with term highlighting
    """
    
    def __init__(self, 
                 graph: EnhancedCrossDocumentKnowledgeGraph,
                 vectorstore: Any,
                 embedding_fn: Callable,
                 llm_generate_fn: Callable,
                 registry: UnifiedConceptRegistry):
        self.graph = graph
        self.vectorstore = vectorstore
        self.embedding_fn = embedding_fn
        self.llm_generate_fn = llm_generate_fn
        self.registry = registry
        self.retriever = GraphDiffusionRetriever(graph, embedding_fn)
    
    def think_and_answer(self, query: str, k: int = 6) -> Tuple[str, ReasoningChain, List[Document], Dict[str, Any]]:
        """
        Full reasoning pipeline for a query.
        
        Returns:
            (answer, reasoning_chain, retrieved_docs, metadata)
        """
        chain = ReasoningChain(query)
        
        # Step 1: Extract query entities
        raw_entities = self._extract_query_entities(query)
        chain.add_step("entity_extraction", 
                      f"Extracted {len(raw_entities)} raw entities", 
                      {"entities": raw_entities})
        
        # Step 2: Expand with synonyms via registry
        expanded_terms = self.registry.expand_query_terms(raw_entities)
        chain.add_step("query_expansion", 
                      f"Expanded to {len(expanded_terms)} terms including synonyms",
                      {"expanded": list(expanded_terms), "original": raw_entities})
        
        # Step 3: Vector retrieval with query expansion boost
        semantic_docs = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k*3, "score_threshold": 0.2}
        ).invoke(query)
        
        # Boost chunks matching expanded terms
        for idx, doc in enumerate(semantic_docs):
            content_lower = doc.page_content.lower()
            boost = 0
            for term in expanded_terms:
                if term.lower() in content_lower:
                    boost += 0.1
            doc.metadata["temp_boost"] = min(boost, 0.5)  # Cap boost
        
        # Compute vector scores with expansion
        query_emb = self.embedding_fn(query)
        if expanded_terms:
            expanded_embs = [self.embedding_fn(t) for t in expanded_terms]
            expanded_emb = np.mean(expanded_embs, axis=0) if expanded_embs else query_emb
        else:
            expanded_emb = query_emb
        
        vector_scores = {}
        for doc in semantic_docs:
            cidx = doc.metadata.get("chunk_index", -1)
            doc_emb = self.embedding_fn(doc.page_content[:500])
            sim = float(np.dot(expanded_emb, doc_emb) / 
                       (np.linalg.norm(expanded_emb)*np.linalg.norm(doc_emb)+1e-8))
            boost = doc.metadata.get("temp_boost", 0)
            vector_scores[cidx] = sim + 0.2 * boost
        
        chain.add_step("vector_retrieval", 
                      f"Retrieved {len(semantic_docs)} chunks with query expansion",
                      {"num_boosted": sum(1 for d in semantic_docs if d.metadata.get("temp_boost", 0) > 0)})
        
        # Step 4: Graph diffusion re-ranking with expanded terms
        all_chunks = []
        for doc_id in self.graph.chunk_index:
            all_chunks.extend(self.graph.chunk_index[doc_id])
        
        hybrid_results = self.retriever.retrieve(
            query, list(expanded_terms), all_chunks, vector_scores, top_k=k, alpha=0.6
        )
        retrieved_docs = [r[0] for r in hybrid_results]
        
        chain.add_step("graph_diffusion", 
                      f"Re-ranked via graph diffusion with expanded terms",
                      {"reasons": Counter(r[2] for r in hybrid_results)})
        
        # Step 5: Claim analysis with unified concepts
        relevant_claims = []
        for claim in self.graph.claims:
            if any(term in claim.subject.lower() or term in claim.object_val.lower() 
                  for term in expanded_terms):
                relevant_claims.append(claim)
        
        chain.add_step("claim_analysis", 
                      f"Found {len(relevant_claims)} relevant claims",
                      {"predicates": [c.predicate for c in relevant_claims[:5]]})
        
        # Step 6: Cross-document consensus/contradiction
        consensus_data = []
        contradictions = []
        
        for term in expanded_terms:
            cons = self.graph.find_consensus(term)
            if cons:
                consensus_data.append(cons)
            contr = self.graph.find_contradictions(term, threshold_factor=1.5)
            contradictions.extend(contr)
        
        chain.add_step("cross_doc_analysis", 
                      f"Consensus: {len(consensus_data)}, Contradictions: {len(contradictions)}",
                      {"consensus_entities": [c["entity"] for c in consensus_data[:3]],
                       "contradiction_pairs": [(c["doc_a"], c["doc_b"], c["entity"]) 
                                              for c in contradictions[:3]]})
        
        # Step 7: Build prompt with context and highlighting
        prompt = self._build_reasoning_prompt_with_synonyms(
            retrieved_docs, query, consensus_data, contradictions, 
            relevant_claims, expanded_terms
        )
        
        # Step 8: Generate answer
        answer = self.llm_generate_fn(prompt)
        
        # Highlight query terms and synonyms in answer
        for term in expanded_terms:
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            answer = pattern.sub(lambda m: f"**{m.group(0)}**", answer)
        
        chain.add_step("synthesis", 
                      "Generated answer with term highlighting",
                      {"prompt_length": len(prompt), "answer_length": len(answer)})
        
        # Metadata
        meta = {
            "query_entities": raw_entities,
            "expanded_terms": list(expanded_terms),
            "consensus_found": len(consensus_data),
            "contradictions_found": len(contradictions),
            "claim_count": len(relevant_claims),
            "retrieval_method": "hybrid_vector_graph_expanded",
            "reasoning_chain": chain.to_markdown()
        }
        
        return answer, chain, retrieved_docs, meta
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract entities from query using registry."""
        entities = []
        q = query.lower()
        
        # Match against registry families
        for canon, info in self.registry.families.items():
            for alias in info["aliases"]:
                if alias.lower() in q:
                    entities.append(canon)
                    break
        
        # Match quantity patterns
        for param_name in QUANTITY_PATTERNS.keys():
            if param_name.replace("_", " ") in q or param_name in q:
                entities.append(param_name)
        
        return list(set(entities))
    
    def _build_reasoning_prompt_with_synonyms(self, 
                                             retrieved_docs, 
                                             query, 
                                             consensus_data, 
                                             contradictions, 
                                             claims,
                                             expanded_terms) -> str:
        """Build prompt with context, consensus, contradictions, and synonym awareness."""
        
        # Format retrieved chunks with term highlighting
        context_parts = []
        for i, chunk in enumerate(retrieved_docs, 1):
            citation = chunk.metadata.get("citation_display") or f"[Source {i}]"
            section = chunk.metadata.get("section", "UNKNOWN")
            content = chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content
            
            # Highlight expanded terms in content
            for term in expanded_terms:
                pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                content = pattern.sub(lambda m: f"**{m.group(0)}**", content)
            
            context_parts.append(f"---\n[{i}] {citation} | Section: {section}\n{content}\n")
        
        context = "\n".join(context_parts)
        
        # Format consensus
        consensus_text = ""
        if consensus_data:
            consensus_text = "\nCross-Document Consensus (unified across synonyms):\n"
            for cons in consensus_data[:3]:
                consensus_text += (
                    f"- {cons['entity']} ({cons['domain']}): {cons['mean']:.2f} ± {cons['std']:.2f} "
                    f"{cons['unit']} across {cons['doc_count']} papers (n={cons['value_count']})\n"
                )
        
        # Format contradictions
        contradiction_text = ""
        if contradictions:
            contradiction_text = "\nDetected Contradictions:\n"
            for contr in contradictions[:3]:
                contradiction_text += (
                    f"- {contr['entity']}: {Path(contr['doc_a']).stem}={contr['mean_a']:.2f} vs "
                    f"{Path(contr['doc_b']).stem}={contr['mean_b']:.2f} "
                    f"(ratio {contr['ratio']:.1f}x, {contr['severity']})\n"
                )
        
        # Format claims
        claim_text = ""
        if claims:
            claim_text = "\nRelevant Claims from Literature:\n"
            for c in claims[:5]:
                claim_text += f"- [{c.doc_source}] {c.subject} → {c.predicate} → {c.object_val}\n"
        
        # Add synonym expansion note
        synonyms_note = (
            f"\n**Note:** Your query terms were expanded to include synonyms: "
            f"{', '.join(list(expanded_terms)[:10])}. "
            f"Results from all these terms are unified under the same concept families.\n"
        )
        
        # System prompt
        system = """You are an expert scientific research assistant specializing in laser-microstructure interactions, multicomponent alloys, and physics-informed digital twins.

SYNTHESIZE across documents. Identify CONSENSUS and CONTRADICTIONS explicitly.
Report UNCERTAINTY: use ranges, standard deviations, and confidence statements.
Cite using the exact format provided. Distinguish experimental results from theory.
If evidence is insufficient, state so clearly.

OUTPUT STRUCTURE:
1. **Direct Answer**
2. **Evidence Synthesis** (with citations)
3. **Consensus & Variability**
4. **Contradictions & Limitations**
5. **Confidence Assessment** (High/Medium/Low)"""
        
        # User prompt
        user = (
            f"{context}\n"
            f"{consensus_text}\n"
            f"{contradiction_text}\n"
            f"{claim_text}\n"
            f"{synonyms_note}\n\n"
            f"Question: {query}\n\n"
            f"Provide a rigorous scientific answer following the structure above."
        )
        
        return system + "\n\n" + user


# =============================================================================
# EMBEDDING WRAPPER
# =============================================================================

class EmbeddingWrapper:
    """Wrapper to unify different embedding API signatures."""
    
    def __init__(self, embedding_source):
        self.source = embedding_source
    
    def __call__(self, text: str) -> np.ndarray:
        if hasattr(self.source, 'embed_query'):
            return np.array(self.source.embed_query(text))
        elif hasattr(self.source, 'embed_documents'):
            return np.array(self.source.embed_documents([text])[0])
        elif hasattr(self.source, 'encode'):
            return self.source.encode(text)
        else:
            raise ValueError("Embedding source has no suitable method")


# =============================================================================
# SEMANTIC CHUNKING WITH STRUCTURE AWARENESS
# =============================================================================

def detect_scientific_sections(text: str) -> List[Tuple[str, str]]:
    """Detect scientific paper sections using regex patterns."""
    section_patterns = [
        (r'(?:^|\n)\s*Abstract\s*\n', 'ABSTRACT'),
        (r'(?:^|\n)\s*1\.\s*Introduction\s*\n', 'INTRODUCTION'),
        (r'(?:^|\n)\s*(?:2\.)?\s*Experimental\s*(?:Setup|Methods|Details)?\s*\n', 'METHODS'),
        (r'(?:^|\n)\s*(?:3\.)?\s*Results\s*(?:and\s*Discussion)?\s*\n', 'RESULTS'),
        (r'(?:^|\n)\s*(?:4\.)?\s*Discussion\s*\n', 'DISCUSSION'),
        (r'(?:^|\n)\s*Conclusion', 'CONCLUSION'),
    ]
    
    boundaries = []
    for pattern, name in section_patterns:
        for match in re.finditer(pattern, text, re.I):
            boundaries.append((match.start(), name))
    
    if not boundaries:
        return [("BODY", text)]
    
    boundaries.sort()
    sections = []
    
    for i, (pos, name) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        section_text = text[pos:end].strip()
        if len(section_text) > 50:
            sections.append((name, section_text))
    
    return sections if sections else [("BODY", text)]


def semantic_chunk_document(pages: List[Document], filename: str) -> List[Document]:
    """Chunk documents with section-aware sizing."""
    all_text = "\n\n".join([p.page_content for p in pages])
    sections = detect_scientific_sections(all_text)
    
    chunks = []
    for section_name, section_text in sections:
        # Different chunk sizes per section type
        if section_name in ['ABSTRACT', 'CONCLUSION']:
            chunk_size, overlap = 400, 50
        elif section_name == 'METHODS':
            chunk_size, overlap = 600, 100
        else:
            chunk_size, overlap = LASER_DOMAIN_CONFIG["chunk_size"], LASER_DOMAIN_CONFIG["chunk_overlap"]
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", "; ", ", "],
            length_function=len
        )
        
        section_chunks = splitter.create_documents([section_text])
        
        for i, chunk in enumerate(section_chunks):
            chunk.metadata.update({
                "source": filename,
                "section": section_name,
                "chunk_index": len(chunks) + i,
                "section_chunk_index": i,
            })
        
        chunks.extend(section_chunks)
    
    # Re-index chunks globally
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)
    
    return chunks


# =============================================================================
# BIBLIOGRAPHIC METADATA
# =============================================================================

class BibliographicMetadata:
    """Extract and store bibliographic metadata from documents."""
    
    DOI_PATTERN = re.compile(r'\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b', re.IGNORECASE)
    ARXIV_PATTERN = re.compile(r'\barXiv[:\s]+(\d{4}\.\d{4,5}(v\d+)?)\b', re.IGNORECASE)
    YEAR_PATTERN = re.compile(r'\b((?:19|20)\d{2})\b')
    
    def __init__(self, source_filename: str):
        self.source_filename = source_filename
        self.doi: Optional[str] = None
        self.arxiv_id: Optional[str] = None
        self.title: Optional[str] = None
        self.authors: List[str] = []
        self.journal: Optional[str] = None
        self.year: Optional[int] = None
        self.confidence: float = 0.0
        self.extraction_method: str = "none"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_filename,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "journal": self.journal,
            "year": self.year,
            "extraction_method": self.extraction_method,
            "confidence": self.confidence,
        }
    
    def format_citation(self, style: str = "apa") -> str:
        if self.doi and self.confidence > 0.8:
            if style == "doi":
                return f"DOI:{self.doi}"
            elif style == "short":
                return f"[DOI:{self.doi}]"
        
        if self.authors and self.year:
            first_author = self.authors[0].split(',')[0] if ',' in self.authors[0] else self.authors[0]
            et_al = " et al." if len(self.authors) > 1 else ""
            
            if style == "apa":
                journal_part = f", {self.journal}" if self.journal else ""
                return f"{first_author}{et_al}{journal_part}, {self.year}"
            elif style == "short":
                return f"[{first_author.split()[0]} {self.year}]"
        
        base_name = Path(self.source_filename).stem
        if self.year:
            return f"[{base_name}, {self.year}]"
        return f"[{base_name}]"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_ollama_model(model_key: str) -> bool:
    return model_key.startswith("ollama:") or model_key.startswith("[Ollama]")


def extract_ollama_tag(model_key: str) -> str:
    if model_key.startswith("ollama:"):
        return model_key.replace("ollama:", "", 1)
    elif model_key.startswith("[Ollama]"):
        match = re.search(r'\]\s*([^\s(]+)', model_key)
        if match:
            return match.group(1)
    return model_key


def get_hf_repo_id(model_key: str) -> str:
    if ":" in model_key and not model_key.startswith("http"):
        parts = model_key.split(":", 1)
        if len(parts) == 2 and "/" in parts[1]:
            return parts[1].strip()
    return model_key


def get_available_gpu_memory() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        return total_memory - reserved
    except:
        return None


def estimate_model_memory(model_key: str, use_4bit: bool = False) -> Dict[str, Any]:
    repo_id = get_hf_repo_id(model_key) if not is_ollama_model(model_key) else model_key
    return MODEL_MEMORY_ESTIMATES.get(repo_id, {
        "params": "Unknown", 
        "vram_fp16": "Unknown", 
        "vram_4bit": "Unknown", 
        "cpu_ok": False
    })


def compute_text_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource(show_spinner="Loading local embedding model (~80MB)...")
def load_local_embeddings():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    except Exception as e:
        st.error(f"Failed to load embeddings: {e}")
        return None


@st.cache_resource(show_spinner="Loading local LLM (this may take 1-2 minutes on first load)...")
def load_local_llm(model_key: str, use_4bit: bool = True):
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


def _load_ollama_model(model_key: str):
    if not OLLAMA_AVAILABLE:
        raise ImportError("ollama library not installed. Run: pip install ollama")
    
    model_tag = extract_ollama_tag(model_key)
    
    try:
        client = ollama.Client(host=st.session_state.ollama_host)
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
            return None, None, st.session_state.ollama_host, "ollama"
    except Exception as conn_err:
        st.error(f"❌ Connection Error: {conn_err}")
        return None, None, st.session_state.ollama_host, "ollama"
    
    return None, model_tag, st.session_state.ollama_host, "ollama"


def _load_transformers_model(model_key: str, use_4bit: bool = True):
    repo_id = get_hf_repo_id(model_key)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    available_vram = get_available_gpu_memory()
    mem_info = estimate_model_memory(model_key, use_4bit)
    
    st.sidebar.info(f"""
    📊 Model Memory Estimate:
    - Parameters: {mem_info['params']}
    - VRAM (FP16): {mem_info['vram_fp16']}
    - VRAM (4-bit): {mem_info['vram_4bit']}
    - CPU OK: {'✅ Yes' if mem_info['cpu_ok'] else '❌ No'}
    - Available VRAM: {f'{available_vram:.1f}GB' if available_vram else 'N/A (CPU)'}
    - Device: {device.upper()}
    """)
    
    if "0.5B" in repo_id or "1.1B" in repo_id or "gpt2" in repo_id or device == "cpu":
        use_4bit = False
    
    quantization_config = None
    if use_4bit and device == "cuda" and available_vram:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            st.sidebar.success("✅ 4-bit quantization enabled")
        except ImportError:
            st.sidebar.warning("⚠️ bitsandbytes not installed.")
            use_4bit = False
    
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, trust_remote_code=True, padding_side="left", use_fast=True
    )
    
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
    }
    
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


# =============================================================================
# DOCUMENT PROCESSING
# =============================================================================

def load_pdf_chunks(uploaded_files):
    """Load and chunk uploaded PDF files."""
    all_chunks = []
    
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getbuffer())
            loader = PyPDFLoader(tmp.name)
            pages = loader.load()
            chunks = semantic_chunk_document(pages, file.name)
            all_chunks.extend(chunks)
            os.unlink(tmp.name)
    
    return all_chunks


@st.cache_resource
def create_local_vector_store(chunks: List[Document], embedding_model_key: str):
    """Create FAISS vector store from chunks."""
    try:
        embeddings = load_local_embeddings()
        if embeddings is None:
            return None
        
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.metadata = {
            "total_chunks": len(chunks),
            "embedding_model": embedding_model_key,
            "created_at": datetime.now().isoformat(),
        }
        return vectorstore
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None


def process_documents(uploaded_files):
    """Main document processing pipeline."""
    if not uploaded_files:
        return False
    
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    if not new_files:
        st.info("✓ All uploaded files already processed")
        return st.session_state.processing_complete
    
    # Reset state for new processing
    st.session_state.messages = []
    st.session_state.vectorstore = None
    st.session_state.all_chunks = []
    st.session_state.knowledge_graph = None
    st.session_state.visualization_engine = None
    
    with st.spinner(f"Processing {len(new_files)} PDF(s) with concept unification..."):
        try:
            # Load and chunk PDFs
            all_chunks = load_pdf_chunks(new_files)
            if not all_chunks:
                st.error("No chunks extracted. Check file format.")
                return False
            
            for f in new_files:
                st.session_state.processed_files.add(f.name)
            st.session_state.all_chunks.extend(all_chunks)
            
            # Load embedding model
            embed_model = load_local_embeddings()
            if embed_model is None:
                st.error("Failed to load embedding model.")
                return False
            
            # Initialize concept registry
            registry = get_concept_registry(embed_model)
            
            # Optional: Run semantic clustering to discover new synonyms
            if SKLEARN_AVAILABLE:
                with st.spinner("Running semantic clustering to discover synonyms..."):
                    clusterer = SemanticClusterer(
                        embed_model, 
                        similarity_threshold=LASER_DOMAIN_CONFIG["synonym_similarity_threshold"],
                        min_cluster_size=LASER_DOMAIN_CONFIG["min_cluster_size"],
                        max_ngram=LASER_DOMAIN_CONFIG["max_ngram_length"]
                    )
                    
                    # Extract candidates from sample of chunks
                    candidates = clusterer.extract_candidates(
                        all_chunks[:200], 
                        max_ngram=LASER_DOMAIN_CONFIG["max_ngram_length"]
                    )
                    
                    if len(candidates) > 10:
                        clusterer.cluster_terms(candidates[:500])
                        registry.merge_from_clusterer(clusterer)
                        st.success(f"✨ Discovered {len(clusterer.clusters)} synonym clusters")
                    else:
                        st.info("Not enough candidates for semantic clustering")
            else:
                st.info("scikit-learn not installed, skipping semantic clustering")
            
            # Extract concepts with unified registry
            extractor = FullTextConceptExtractor(embed_model, registry)
            custom_list = st.session_state.get('custom_priority_concepts',
                                              ["melt pool dynamics", "keyhole mode", "marangoni convection"])
            extractor.set_custom_priority(custom_list)
            
            valid_concepts, concept_metadata = extractor.extract_concepts(
                all_chunks, min_salience=0.42
            )
            st.info(f"📊 Extracted {len(valid_concepts)} high-salience concept families")
            
            # Build knowledge graph with unified concepts
            graph = EnhancedCrossDocumentKnowledgeGraph(registry)
            dummy_bib = BibliographicMetadata("dummy")
            dummy_bib.title = "Processed documents"
            
            # Group chunks by source
            doc_chunks = {}
            for chunk in all_chunks:
                src = chunk.metadata.get("source", "unknown")
                if src not in doc_chunks:
                    doc_chunks[src] = []
                doc_chunks[src].append(chunk)
            
            # Add each document to graph
            for src, chunks in doc_chunks.items():
                graph.add_document(src, chunks, dummy_bib, concept_metadata=concept_metadata)
            
            st.session_state.knowledge_graph = graph
            st.session_state.concept_registry = registry
            
            # Create vector store
            vectorstore = create_local_vector_store(all_chunks, LOCAL_EMBEDDING_MODEL)
            if vectorstore is None:
                return False
            st.session_state.vectorstore = vectorstore
            
            # Summary
            summary = graph.get_knowledge_summary()
            st.success(
                f"✅ Ready! Indexed {len(all_chunks)} chunks, "
                f"{summary['unique_entities']} unified concepts, "
                f"{summary['total_claims']} claims from {summary['document_count']} papers"
            )
            
            if summary['high_salience_concepts']:
                st.caption(
                    f"⭐ Top concepts: {', '.join([c[:30] for c, _ in summary['high_salience_concepts'][:5]])}"
                )
            
            st.session_state.processing_complete = True
            return True
            
        except Exception as e:
            st.error(f"Processing failed: {e}")
            import traceback
            st.error(traceback.format_exc())
            return False


# =============================================================================
# RESPONSE GENERATION
# =============================================================================

def generate_local_response(tokenizer, model_or_tag, device_or_host: str, 
                           prompt: str, backend: str, backend_type: str) -> str:
    """Route response generation to appropriate backend."""
    if backend_type == "ollama":
        return generate_local_response_ollama(model_or_tag, device_or_host, prompt)
    else:
        return generate_local_response_transformers(tokenizer, model_or_tag, device_or_host, prompt, backend)


def generate_local_response_transformers(tokenizer, model, device: str, 
                                        prompt: str, backend_name: str) -> str:
    """Generate response using HuggingFace transformers."""
    try:
        # Format prompt based on model type
        if "Qwen" in backend_name or "qwen" in backend_name.lower():
            messages = [
                {"role": "system", "content": "You are an expert in laser-microstructure interaction research. Synthesize evidence across multiple papers rigorously."},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        elif "Llama" in backend_name or "llama" in backend_name.lower():
            messages = [
                {"role": "system", "content": "You are an expert in laser-microstructure interaction research. Synthesize evidence across multiple papers rigorously."},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        elif "Mistral" in backend_name or "mistral" in backend_name.lower():
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        else:
            formatted_prompt = prompt
        
        # Tokenize and generate
        inputs = tokenizer.encode(
            formatted_prompt, return_tensors='pt', truncation=True,
            max_length=LASER_DOMAIN_CONFIG["max_context_tokens"]
        )
        
        if device == "cuda" and torch.cuda.is_available():
            inputs = inputs.to('cuda')
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=LASER_DOMAIN_CONFIG["max_new_tokens"],
                temperature=LASER_DOMAIN_CONFIG["temperature"],
                do_sample=(LASER_DOMAIN_CONFIG["temperature"] > 0),
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        
        # Decode and extract answer
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "[/INST]" in full_text:
            answer = full_text.split("[/INST]")[-1].strip()
        elif "Confidence Assessment:" in full_text:
            answer = full_text[full_text.find("Direct Answer:"):].strip() if "Direct Answer:" in full_text else full_text[-1500:].strip()
        else:
            answer = full_text[-LASER_DOMAIN_CONFIG["max_new_tokens"] * 2:].strip()
        
        answer = re.sub(r'\s+', ' ', answer).strip()
        return answer if answer else "I was unable to generate a response. Please try rephrasing your question."
        
    except Exception as e:
        st.error(f"Generation error: {e}")
        return f"Error generating response: {str(e)[:200]}..."


def generate_local_response_ollama(model_tag: str, ollama_host: str, prompt: str) -> str:
    """Generate response using Ollama."""
    try:
        client = ollama.Client(host=ollama_host)
        messages = [
            {"role": "system", "content": "You are an expert in laser-microstructure interaction research. Synthesize evidence across multiple papers rigorously."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Try streaming first
            response = client.chat(
                model=model_tag, messages=messages,
                options={"temperature": LASER_DOMAIN_CONFIG["temperature"], 
                        "num_predict": LASER_DOMAIN_CONFIG["max_new_tokens"]},
                stream=True
            )
            full_response = ""
            for chunk in response:
                if isinstance(chunk, dict):
                    if 'message' in chunk and 'content' in chunk['message']:
                        full_response += chunk['message']['content']
                    elif 'content' in chunk:
                        full_response += chunk['content']
                elif hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
                    full_response += chunk.message.content
        except TypeError:
            # Fall back to non-streaming
            response = client.chat(
                model=model_tag, messages=messages,
                options={"temperature": LASER_DOMAIN_CONFIG["temperature"], 
                        "num_predict": LASER_DOMAIN_CONFIG["max_new_tokens"]}
            )
            if isinstance(response, dict):
                full_response = response.get('message', {}).get('content', '')
            elif hasattr(response, 'message'):
                full_response = response.message.content
            else:
                full_response = str(response)
        
        return full_response.strip() if full_response.strip() else "I was unable to generate a response. Please try rephrasing your question."
        
    except Exception as e:
        st.error(f"Ollama generation error: {e}")
        return f"Error generating response via Ollama: {str(e)[:200]}..."


def retrieve_and_answer(vectorstore, graph, tokenizer, model, device_or_host, 
                       backend, backend_type, query, k=None, score_threshold=None):
    """Main retrieval and answer function."""
    k = k or LASER_DOMAIN_CONFIG["retrieval_k"]
    
    # Get embedding function
    emb_source = getattr(vectorstore, 'embedding_function', 
                        getattr(vectorstore, 'embeddings', vectorstore))
    emb_fn = EmbeddingWrapper(emb_source)
    
    # LLM generation wrapper
    def llm_generate(prompt):
        return generate_local_response(
            tokenizer=tokenizer, model_or_tag=model, device_or_host=device_or_host,
            prompt=prompt, backend=backend, backend_type=backend_type
        )
    
    # Get concept registry from session or create new
    registry = st.session_state.get('concept_registry')
    if registry is None:
        registry = UnifiedConceptRegistry()
    
    # Create thinker and run pipeline
    thinker = CrossDocumentThinker(graph, vectorstore, emb_fn, llm_generate, registry)
    answer, chain, retrieved_docs, meta = thinker.think_and_answer(query, k=k)
    
    # Compute relevance score
    avg_relevance = 0.0
    if retrieved_docs:
        query_emb = emb_fn(query)
        scores = []
        for doc in retrieved_docs:
            doc_emb = emb_fn(doc.page_content[:500])
            sim = float(np.dot(query_emb, doc_emb) / 
                       (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-8))
            scores.append(sim)
        avg_relevance = np.mean(scores) if scores else 0.0
    
    meta["avg_vector_score"] = avg_relevance
    return answer, retrieved_docs, avg_relevance, meta, chain


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def initialize_session_state():
    """Initialize Streamlit session state with defaults."""
    defaults = {
        "processed_files": set(),
        "vectorstore": None,
        "all_chunks": [],
        "messages": [],
        "llm_model_choice": None,
        "llm_tokenizer": None,
        "llm_model": None,
        "llm_backend": None,
        "llm_device_or_host": None,
        "llm_backend_type": None,
        "embeddings": None,
        "processing_complete": False,
        "laser_domain_boost": True,
        "show_sources": True,
        "citation_style": "apa",
        "max_retrieved_chunks": 6,
        "use_4bit_quantization": True,
        "ollama_host": "http://localhost:11434",
        "metadata_cache": {},
        "knowledge_graph": None,
        "reasoning_mode": True,
        "show_reasoning_chain": True,
        "cross_doc_consensus": True,
        "feedback_map": {},
        "precision_recall": None,
        "show_network": False,
        "selected_entity": None,
        "plot_code": "",
        "last_plot_fig": None,
        "reasoning_chain": None,
        "visualization_engine": None,
        "custom_priority_concepts": ["melt pool dynamics", "keyhole mode", "marangoni convection"],
        "concept_unifier": None,
        "viz_font_family": "DejaVu Sans",
        "viz_font_size": 10,
        "viz_title_font_size": 14,
        "viz_label_font_size": 9,
        "viz_colormap": "viridis",
        "viz_figure_dpi": 300,
        "viz_layout": "spring",
        "concept_registry": None,
        "synonym_expansion_enabled": True,
        "show_synonym_hints": True,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# STREAMLIT UI COMPONENTS
# =============================================================================

def render_sidebar():
    """Render sidebar with configuration options."""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Backend selection
        backend_option = st.radio(
            "🔧 Inference Backend", 
            options=["Hugging Face Transformers", "Ollama (if installed)"], 
            index=0
        )
        st.session_state.inference_backend = backend_option
        
        if backend_option == "Ollama (if installed)":
            if not OLLAMA_AVAILABLE:
                st.error("❌ ollama library not installed")
                st.code("pip install ollama")
            
            available_ollama_models = [k for k in LOCAL_LLM_OPTIONS.keys() if is_ollama_model(k)]
            model_choice = st.selectbox(
                "🧠 Local LLM Backend (Ollama)", 
                options=available_ollama_models if available_ollama_models else ["No Ollama models available"], 
                index=0
            )
        else:
            hf_models = [k for k in LOCAL_LLM_OPTIONS.keys() if not is_ollama_model(k)]
            model_choice = st.selectbox(
                "🧠 Local LLM Backend (Hugging Face)", 
                options=hf_models, 
                index=2
            )
        
        st.session_state.llm_model_choice = model_choice
        
        if backend_option == "Hugging Face Transformers" and not is_ollama_model(model_choice):
            st.session_state.use_4bit_quantization = st.checkbox(
                "🗜️ Use 4-bit quantization", value=True
            )
        
        if backend_option == "Ollama (if installed)" or is_ollama_model(model_choice):
            st.session_state.ollama_host = st.text_input(
                "🌐 Ollama Host", value=st.session_state.ollama_host
            )
        
        # Reasoning settings
        st.markdown("#### 🔬 Reasoning Settings")
        st.session_state.reasoning_mode = st.checkbox(
            "🧠 Cross-document reasoning", value=True,
            help="Enable entity extraction, consensus detection, and multi-hop retrieval across papers"
        )
        st.session_state.cross_doc_consensus = st.checkbox(
            "📊 Detect consensus & contradictions", value=True,
            help="Statistically compare reported values across documents"
        )
        st.session_state.show_reasoning_chain = st.checkbox(
            "🔍 Show reasoning chain", value=True,
            help="Display the logical steps and evidence linking"
        )
        
        # Concept unification settings
        st.markdown("#### 🔗 Concept Unification")
        st.session_state.synonym_expansion_enabled = st.checkbox(
            "✨ Enable synonym expansion", value=True,
            help="Expand query terms with known synonyms for broader retrieval"
        )
        st.session_state.show_synonym_hints = st.checkbox(
            "💡 Show synonym hints", value=True,
            help="Display 'Also searched for' hints in responses"
        )
        
        # Laser domain settings
        st.markdown("#### 🔬 Laser Domain Settings")
        st.session_state.laser_domain_boost = st.checkbox(
            "Boost laser-topic relevance", value=True
        )
        st.session_state.show_sources = st.checkbox(
            "Show source citations", value=True
        )
        
        # Priority concepts
        st.markdown("#### ⭐ Core Pillars & Priority Concepts")
        st.markdown("**Always High Salience:** • LASER • MICROSTRUCTURE • INTERACTION • MULTICOMPONENT ALLOY")
        
        default_priority = [
            "melt pool dynamics", "keyhole mode", "marangoni convection",
            "porosity formation", "intermetallic compound", "columnar to equiaxed transition",
            "residual stress", "solidification microstructure", "multicomponent alloy",
            "high entropy alloy", "complex concentrated alloy"
        ]
        
        selected_custom = st.multiselect(
            "Add extra high-priority concepts (boosted salience)",
            options=[
                "melt pool dynamics", "keyhole mode", "marangoni convection",
                "porosity formation", "spatter ejection", "lack of fusion",
                "intermetallic compound", "IMC", "Cu6Sn5",
                "columnar to equiaxed transition", "CET", "epitaxial growth",
                "residual stress", "grain morphology", "solidification",
                "solidification microstructure",
                "multicomponent alloy", "high entropy alloy", "hea", "mpea", "complex concentrated alloy",
                "digital twin", "physics-informed modeling", "process-structure-property"
            ],
            default=default_priority,
            key="custom_priority_concepts",
            help="These concepts will receive strong salience boost in extraction and visualization"
        )
        
        # Visualization customization
        st.markdown("#### 🎨 Visualization Customization")
        st.session_state.viz_font_family = st.selectbox(
            "Font Family",
            ["DejaVu Sans", "Arial", "Helvetica", "Times New Roman", "Computer Modern", "serif", "sans-serif"],
            index=0
        )
        st.session_state.viz_font_size = st.slider("Base Font Size", 8, 16, 10)
        st.session_state.viz_title_font_size = st.slider("Title Font Size", 10, 24, 14)
        st.session_state.viz_label_font_size = st.slider("Label Font Size", 6, 14, 9)
        st.session_state.viz_colormap = st.selectbox(
            "Default Colormap",
            ["viridis", "plasma", "inferno", "magma", "cividis", "Blues", "Greens", "Oranges"],
            index=0
        )
        st.session_state.viz_layout = st.selectbox(
            "Network Layout", ["spring", "kamada_kawai", "circular"], index=0
        )
        st.session_state.viz_figure_dpi = st.slider("Figure DPI", 150, 600, 300, step=50)
        
        # Citation format
        st.markdown("#### 📝 Citation Format")
        st.session_state.citation_style = st.selectbox(
            "Citation display style", 
            options=["apa", "doi", "full", "short"], 
            index=0,
            format_func=lambda x: {
                "apa": "APA: FirstAuthor et al., Journal, Year", 
                "doi": "DOI: 10.xxxx/xxxxx",
                "full": "Full: Author (Year). Title. Journal, Vol(Issue), Pages", 
                "short": "Short: [FirstAuthor Year] or [DOI]"
            }[x]
        )
        
        st.session_state.max_retrieved_chunks = st.slider(
            "Chunks to retrieve", min_value=2, max_value=10, value=6
        )
        
        st.markdown("---")
        
        # Visualizations toggle
        st.markdown("### 🕸️ Visualisations")
        st.session_state.show_network = st.sidebar.checkbox(
            "Show Knowledge Graph", value=False
        )
        
        # Entity explorer
        if st.session_state.knowledge_graph and st.session_state.knowledge_graph.entities:
            entity_options = list(st.session_state.knowledge_graph.entities.keys())
            if entity_options:
                st.session_state.selected_entity = st.sidebar.selectbox(
                    "Explore entity consensus",
                    options=entity_options,
                    format_func=lambda x: x,
                    key="selected_entity_select"
                )
            else:
                st.session_state.selected_entity = None
        
        st.markdown("---")
        
        # System info
        gpu_info = "CUDA" if torch.cuda.is_available() else "CPU"
        vram_info = f"{get_available_gpu_memory():.1f}GB free" if torch.cuda.is_available() and get_available_gpu_memory() else "N/A"
        st.caption(f"🖥️ Device: {gpu_info}")
        st.caption(f"💾 Available VRAM: {vram_info}")
        
        if PDF2DOI_AVAILABLE:
            st.success("✅ pdf2doi: Available")
        else:
            st.info("ℹ️ pdf2doi: Optional for DOI lookup")
        
        if CROSSREF_AVAILABLE:
            st.success("✅ Crossref API: Available")
        else:
            st.info("ℹ️ Crossref: Optional for metadata enrichment")


def render_document_uploader():
    """Render PDF upload component."""
    st.markdown("### 📁 Upload Full-Text PDF Documents")
    uploaded_files = st.file_uploader(
        "Select PDF files about laser processing, multicomponent alloys, additive manufacturing, etc.",
        type=["pdf"], accept_multiple_files=True,
        help="Documents will be processed with full-text extraction, section detection, and salience-based concept ranking."
    )
    return uploaded_files


def render_chat_interface():
    """Render the main chat interface."""
    if not st.session_state.get('vectorstore'):
        st.info("👆 Upload PDF documents above to start chatting with cross-document reasoning")
        return
    
    # Load model if needed
    if st.session_state.llm_tokenizer is None and st.session_state.llm_model_choice:
        backend_type = "ollama" if is_ollama_model(st.session_state.llm_model_choice) else "transformers"
        
        with st.spinner(f"Loading {st.session_state.llm_model_choice}..."):
            result = load_local_llm(
                st.session_state.llm_model_choice, 
                use_4bit=st.session_state.get('use_4bit_quantization', True)
            )
            tokenizer, model, device_or_host, loaded_backend = result
            
            if tokenizer is not None or model is not None:
                st.session_state.llm_tokenizer = tokenizer
                st.session_state.llm_model = model
                st.session_state.llm_device_or_host = device_or_host
                st.session_state.llm_backend_type = loaded_backend
                st.success("✓ Model loaded!")
            else:
                st.error("Failed to load model. Try selecting a different option.")
                return
    
    # Check model availability
    has_model = (
        st.session_state.llm_backend_type == "ollama" and st.session_state.llm_model is not None
    ) or (
        st.session_state.llm_backend_type == "transformers" and st.session_state.llm_tokenizer is not None
    )
    
    if not has_model:
        st.warning("Please select and load a model in the sidebar first")
        return
    
    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if enabled
            if message.get("sources") and st.session_state.show_sources:
                with st.expander("📚 Retrieved Sources with Citations"):
                    for j, src in enumerate(message["sources"], 1):
                        citation = src.metadata.get("citation_display", "Unknown source")
                        section = src.metadata.get("section", "UNKNOWN")
                        st.markdown(f"**[{j}]** {citation} | *{section}*")
                        
                        bib = src.metadata.get("bibliographic", {})
                        if bib and any(bib.get(k) for k in ['doi', 'authors', 'journal', 'year']):
                            with st.expander("🔍 Bibliographic Details"):
                                if bib.get('doi'):
                                    st.markdown(f"**DOI:** `{bib['doi']}`")
                                if bib.get('authors'):
                                    st.markdown(f"**Authors:** {', '.join(bib['authors'][:3])}{'...' if len(bib['authors'])>3 else ''}")
                                if bib.get('journal'):
                                    st.markdown(f"**Journal:** {bib['journal']}")
                                if bib.get('year'):
                                    st.markdown(f"**Year:** {bib['year']}")
                        
                        st.markdown(f"> {src.page_content[:300]}...")
            
            # Show reasoning metadata if enabled
            if message.get("reasoning_meta") and st.session_state.show_reasoning_chain and message["role"] == "assistant":
                meta = message["reasoning_meta"]
                with st.expander("🧠 Reasoning Chain"):
                    st.markdown(f"**Query entities detected:** {', '.join(meta.get('query_entities', [])) or 'None'}")
                    
                    if st.session_state.synonym_expansion_enabled:
                        expanded = meta.get('expanded_terms', [])
                        if expanded:
                            st.markdown(f"**Expanded synonyms:** {', '.join(expanded[:10])}{'...' if len(expanded) > 10 else ''}")
                    
                    st.markdown(f"**Cross-document consensus found:** {meta.get('consensus_found', 0)}")
                    st.markdown(f"**Contradictions detected:** {meta.get('contradictions_found', 0)}")
                    
                    if meta.get('relevance'):
                        st.markdown(f"**Response relevance:** {meta['relevance']:.2f}/1.0")
            
            # Show full reasoning chain if enabled
            if message.get("reasoning_chain") and st.session_state.show_reasoning_chain and message["role"] == "assistant":
                with st.expander("🧠 Full Thinking Trace", expanded=False):
                    st.markdown(message["reasoning_chain"].to_markdown())
                    
                    if st.button("Render Thinking Graph", key=f"think_graph_{i}"):
                        # Visualization would go here
                        st.info("Thinking graph visualization coming soon")
    
    # Chat input
    if prompt := st.chat_input("Ask a cross-document scientific question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking across documents with synonym expansion..."):
                answer, retrieved_docs, avg_relevance, reasoning_meta, chain = retrieve_and_answer(
                    st.session_state.vectorstore, 
                    st.session_state.knowledge_graph,
                    st.session_state.llm_tokenizer, 
                    st.session_state.llm_model,
                    st.session_state.llm_device_or_host, 
                    st.session_state.llm_model_choice,
                    st.session_state.llm_backend_type, 
                    prompt,
                    k=st.session_state.max_retrieved_chunks
                )
                
                reasoning_meta['relevance'] = avg_relevance
                
                # Show synonym hints if enabled
                if st.session_state.show_synonym_hints and reasoning_meta.get('expanded_terms'):
                    original = reasoning_meta.get('query_entities', [])
                    expanded = set(reasoning_meta.get('expanded_terms', [])) - set(original)
                    if expanded:
                        st.caption(f"💡 Also searched for: {', '.join(list(expanded)[:5])}{'...' if len(expanded) > 5 else ''}")
                
                st.markdown(answer)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": retrieved_docs,
                    "reasoning_meta": reasoning_meta,
                    "reasoning_chain": chain
                })


def render_footer():
    """Render footer with tips and examples."""
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📚 Example Questions:**")
        st.caption("• What is the effect of composition on IMC growth in Sn‑Ag‑Cu solders during laser soldering?")
        st.caption("• How do multi‑scale simulations predict grain structure in SLM of Al‑Cr‑Fe‑Ni alloys?")
        st.caption("• What contradictions exist regarding the influence of Marangoni convection on porosity formation?")
    
    with col2:
        st.markdown("**⚡ Reasoning Tips:**")
        st.caption("• Ask comparative questions to trigger consensus detection")
        st.caption("• Query specific alloy families (e.g., 'Sn‑Ag‑Cu', 'AlCrFeNi') to activate entity linking")
        st.caption("• Look for the 🧠 Reasoning Chain expander for transparency")
        st.caption("• Try different terminology: 'HEA' vs 'multicomponent alloy' to test unification")
    
    with col3:
        st.markdown("**🔐 Privacy & Science:**")
        st.caption("• All processing happens locally")
        st.caption("• Cross-document reasoning uses extracted entities only")
        st.caption("• Uncertainty is explicitly reported, never hidden")
        st.caption("• Concept unification preserves source terminology while enabling synthesis")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="🔬 DECLARMIMA: Unified Concepts + Query Emphasis + Publication Viz",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #1e40af, #7c3aed, #059669);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
    }
    .info-card {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin: 0.5rem 0;
    }
    .reasoning-badge {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        padding: 0.2rem 0.6rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
        margin: 0.1rem 0.2rem 0.1rem 0;
    }
    .consensus-badge {
        display: inline-block;
        background: #d1fae5;
        color: #065f46;
        padding: 0.2rem 0.6rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
        margin: 0.1rem 0.2rem 0.1rem 0;
    }
    .contradiction-badge {
        display: inline-block;
        background: #fee2e2;
        color: #991b1b;
        padding: 0.2rem 0.6rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
        margin: 0.1rem 0.2rem 0.1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🔬 DECLARMIMA: Unified Concepts + Query Emphasis</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:#64748b;margin-bottom:1.5rem">
    Upload <strong>full-text PDF papers</strong> on multicomponent alloys and laser processing.   
    This system uses <strong>three-layer concept unification</strong> (lexical, semantic clustering, contextual disambiguation) and <strong>query-term emphasis</strong> with synonym expansion.  
    All variants like "HEA", "multi-principal element alloy", and "complex concentrated alloy" are treated as <strong>one unified concept family</strong>.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Memory warning if needed
    if st.session_state.llm_model_choice and not is_ollama_model(st.session_state.llm_model_choice):
        mem_info = estimate_model_memory(
            st.session_state.llm_model_choice, 
            st.session_state.get('use_4bit_quantization', True)
        )
        available_vram = get_available_gpu_memory()
        
        if available_vram and not mem_info['cpu_ok']:
            required = float(mem_info['vram_4bit'].replace('GB','').replace('~','').strip()) if 'GB' in mem_info['vram_4bit'] else 100
            if available_vram < required:
                st.markdown(f"""
                <div style="background:#fef3c7;border-left:4px solid #f59e0b;padding:0.75rem;border-radius:0 0.5rem 0.5rem 0;margin:0.5rem 0">
                ⚠️ <strong>Memory Warning:</strong> {st.session_state.llm_model_choice} requires ~{mem_info['vram_4bit']} VRAM.
                You have ~{available_vram:.1f}GB available. Consider using 4-bit quantization or a smaller model.
                </div>
                """, unsafe_allow_html=True)
    
    # Main layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Document uploader
        uploaded_files = render_document_uploader()
        
        if uploaded_files and st.button("🔄 Process PDFs", type="primary", use_container_width=True):
            process_documents(uploaded_files)
        
        # Processing status
        if st.session_state.processing_complete:
            st.success("✅ Knowledge base ready")
            
            if st.session_state.knowledge_graph:
                summary = st.session_state.knowledge_graph.get_knowledge_summary()
                st.caption(
                    f"📦 {len(st.session_state.all_chunks)} chunks | "
                    f"{summary['unique_entities']} unified concepts | "
                    f"{summary['total_claims']} claims"
                )
                
                if summary['high_salience_concepts']:
                    st.markdown("**⭐ High-Salience Concepts:**")
                    for ent, meta in summary['high_salience_concepts'][:5]:
                        badge_class = "reasoning-badge" if meta.get('is_core_pillar') else "consensus-badge"
                        aliases = meta.get('aliases', [])
                        alias_str = f" ({len(aliases)} variants)" if len(aliases) > 1 else ""
                        st.markdown(
                            f'<span class="{badge_class}">{ent}{alias_str} (salience {meta["salience"]:.2f})</span>', 
                            unsafe_allow_html=True
                        )
        elif uploaded_files:
            st.warning("⏳ Click 'Process PDFs' to begin")
        else:
            st.info("📁 Upload full-text PDF files to start")
        
        # Clear button
        if st.session_state.processed_files:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.clear()
                st.rerun()
    
    with col2:
        if st.session_state.processing_complete and st.session_state.vectorstore:
            render_chat_interface()
        else:
            # Welcome/info card
            st.markdown("""
            <div class="info-card">
            <h3>👋 Welcome to the Unified Concept & Query Emphasis System</h3>
            <p><strong>Three-Layer Concept Unification:</strong></p>
            <ol>
            <li><strong>Lexical Normalization:</strong> Pre-defined aliases + regex patterns for hyphenated/spaced variants</li>
            <li><strong>Semantic Clustering:</strong> Auto-discover synonyms via embedding similarity (HDBSCAN/connected components)</li>
            <li><strong>Contextual Disambiguation:</strong> Query-time resolution using embedding similarity to concept centroids</li>
            </ol>
            <p><strong>Query Emphasis Features:</strong></p>
            <ul>
            <li>✅ Synonym expansion: "laser power" → [beam intensity, irradiance, fluence]</li>
            <li>✅ Temporary salience boost for query-relevant terms</li>
            <li>✅ Cross-document consensus across all terminology variants</li>
            <li>✅ Bold highlighting of query terms AND recognized synonyms in answers</li>
            <li>✅ "Also searched for" hints for transparency</li>
            </ul>
            <p><strong>Getting started:</strong> Upload PDFs, wait for processing (semantic clustering may take a few minutes), then ask a question using any terminology variant.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Demo questions
            st.markdown("**Try asking:**")
            demo_qs = [
                "What is the effect of laser power on interfacial IMC thickness in Sn‑Ag‑Cu/Cu joints?",
                "Do these papers agree on the optimal hatch distance for defect‑free LPBF of Al‑Cr‑Fe‑Ni alloys?",
                "Summarize the phase‑field models used for simulating selective laser melting of multicomponent alloys.",
                "How does the composition of high entropy alloys affect their thermal conductivity during laser processing?",
                "Compare keyhole stability in Ti-6Al-4V using different laser parameters",
            ]
            
            for q in demo_qs:
                if st.button(f"💬 {q}", use_container_width=True, key=f"demo_{hash(q) % 10000}"):
                    st.session_state.demo_question = q
                    st.rerun()
    
    # Visualization dashboard (if processing complete)
    if st.session_state.knowledge_graph and st.session_state.processing_complete:
        st.markdown("---")
        st.markdown("## 🔬 Unified Concept Visualization Dashboard")
        
        # Initialize visualization engine
        if st.session_state.visualization_engine is None:
            # Placeholder - full visualization engine would be initialized here
            st.session_state.visualization_engine = None
        
        # Concept family explorer
        with st.expander("🔍 Explore Concept Families", expanded=False):
            if st.session_state.concept_registry:
                registry = st.session_state.concept_registry
                
                # Search box
                search_term = st.text_input("Search concepts...", key="concept_search")
                
                if search_term:
                    # Find matching families
                    matches = [
                        (canon, info) for canon, info in registry.families.items()
                        if search_term.lower() in canon or any(search_term.lower() in alias for alias in info["aliases"])
                    ]
                    
                    if matches:
                        st.markdown(f"**Found {len(matches)} matching concept families:**")
                        for canon, info in matches[:10]:
                            with st.container():
                                st.markdown(f"#### {canon}")
                                st.markdown(f"**Aliases:** {', '.join(list(info['aliases'])[:10])}{'...' if len(info['aliases']) > 10 else ''}")
                                st.markdown(f"**Salience:** {info['salience']:.2f} | **Frequency:** {info['frequency']}")
                                if info.get('discovered'):
                                    st.caption("✨ Auto-discovered via semantic clustering")
                                st.markdown("---")
                    else:
                        st.info("No matching concepts found")
                else:
                    # Show top families by salience
                    top_families = sorted(
                        registry.families.items(),
                        key=lambda x: x[1]["salience"],
                        reverse=True
                    )[:10]
                    
                    st.markdown("**Top 10 Concept Families by Salience:**")
                    for canon, info in top_families:
                        with st.container():
                            st.markdown(f"#### {canon}")
                            st.markdown(f"**Aliases:** {', '.join(list(info['aliases'])[:8])}{'...' if len(info['aliases']) > 8 else ''}")
                            st.markdown(f"**Salience:** {info['salience']:.2f} | **Frequency:** {info['frequency']}")
                            if info.get('is_core_pillar'):
                                st.caption("⭐ Core pillar")
                            elif info.get('discovered'):
                                st.caption("✨ Auto-discovered")
                            st.markdown("---")
            else:
                st.info("Concept registry not yet initialized")
        
        # Query expansion demo
        with st.expander("🧪 Test Query Expansion", expanded=False):
            test_query = st.text_input("Enter a test query:", placeholder="e.g., 'laser power effect on melt pool'")
            
            if test_query and st.session_state.concept_registry:
                registry = st.session_state.concept_registry
                
                # Extract and expand
                from CrossDocumentThinker import CrossDocumentThinker  # Would import properly in real code
                thinker = CrossDocumentThinker.__new__(CrossDocumentThinker)
                thinker.registry = registry
                
                raw = thinker._extract_query_entities(test_query)
                expanded = registry.expand_query_terms(raw)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Extracted entities:**")
                    for ent in raw:
                        st.markdown(f"- {ent}")
                
                with col_b:
                    st.markdown("**Expanded with synonyms:**")
                    new_terms = set(expanded) - set(raw)
                    if new_terms:
                        for term in new_terms:
                            st.markdown(f"- {term} ✨")
                    else:
                        st.markdown("No additional synonyms found")
    
    # Render footer
    render_footer()
    
    # Handle demo question
    if hasattr(st.session_state, 'demo_question') and st.session_state.demo_question:
        st.session_state.messages.append({"role": "user", "content": st.session_state.demo_question})
        del st.session_state.demo_question
        st.rerun()


if __name__ == "__main__":
    main()
