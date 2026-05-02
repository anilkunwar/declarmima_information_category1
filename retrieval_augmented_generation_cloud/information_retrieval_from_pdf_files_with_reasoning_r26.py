#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LASER MICROSTRUCTURE RAG CHATBOT - CROSS-DOCUMENT SCIENTIFIC REASONING & VISUALIZATION
========================================================================================
UPGRADED VERSION (CODE 16+): Enhanced Salience Awareness, LLM-Influenced Extraction & Publication-Quality Visualizations
Key Upgrades:
1. FIXED: NameError: name 'matplotlib' is not defined (added import matplotlib + mcolors)
2. ENHANCED: Core pillars now include MULTICOMPONENT ALLOY + semantic similarity boosting
3. NEW: Semantic relation detection for salience via embedding similarity to core concepts
4. NEW: Publication-quality visualization engine with 50+ colormaps, customizable fonts
5. NEW: UMAP, PCA, enhanced t-SNE dimensionality reduction plots
6. NEW: Bokeh + HoloViews chord diagram implementation
7. NEW: PyVis interactive network with full customization
8. ENHANCED: Hierarchical sunbursts, radar charts, contradiction matrices
9. ALL ORIGINAL FUNCTIONALITY PRESERVED AND EXTENDED
10. NEW: LLM-INFLUENCED EXTRACTION PIPELINE (Few-shot disambiguation, importance ranking, confidence scoring)
11. NEW: DYNAMIC TOP-N CONCEPT SELECTOR ACROSS ALL VISUALIZATION TABS
12. NEW: COMPREHENSIVE CONFIGURATION MANAGER, PROGRESS TRACKING, & BATCH PROCESSING
13. NEW: ADVANCED LOGGING, VALIDATION, & SESSION STATE PERSISTENCE
14. NEW: ENHANCED EXPORT CAPABILITIES (CSV, JSON, HTML, PNG, SVG)
15. NEW: QUERY-DRIVEN LAZY PROCESSING PIPELINE (Defer indexing until first query)
16. NEW: QUERY-BIASED SALIENCE SCORING (Dynamic concept weighting per user intent)
17. NEW: DETERMINISTIC QUERY CACHING & INCREMENTAL INDEX REUSE
18. NEW: ADVANCED MEMORY MANAGEMENT & GARBAGE COLLECTION ROUTINES
19. NEW: EXTENSIVE PROGRESS TRACKING, ERROR RECOVERY, & FALLBACK STRATEGIES
20. NEW: COMPREHENSIVE INLINE DOCUMENTATION & ARCHITECTURAL COMMENTS
"""
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
from typing import List, Dict, Optional, Tuple, Union, Any, Set, Callable
from datetime import datetime
import sys
import subprocess
import platform
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
from dataclasses import dataclass, field
import logging
import traceback
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# LOGGING CONFIGURATION
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("declarmima_app.log")
    ]
)
logger = logging.getLogger("DECLARMIMA")

# =====================================================================
# FIX: Added import matplotlib and matplotlib.colors to resolve NameError
# =====================================================================
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx

# NEW: Optional advanced visualization libraries
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from bokeh.plotting import figure, output_file, save, show
    from bokeh.models import Circle, MultiLine, HoverTool, BoxSelectTool, TapTool, ColumnDataSource, LabelSet
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
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# =====================================================================
# PERFORMANCE IMPORTS
# =====================================================================
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import functools

# =====================================================================
# FAST PDF EXTRACTION (PyMuPDF fallback)
# =====================================================================
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# =====================================================================
# CONFIGURATION MANAGEMENT
# =====================================================================
class AppConfig:
    """Centralized configuration manager with validation and persistence."""
    DEFAULT_CONFIG = {
        "chunk_size": 800,
        "chunk_overlap": 150,
        "retrieval_k": 4,
        "score_threshold": 0.25,
        "max_context_tokens": 2048,
        "max_new_tokens": 512,
        "temperature": 0.05,
        "min_salience_threshold": 0.42,
        "llm_extraction_enabled": False,
        "llm_batch_size": 8,
        "llm_timeout_seconds": 30,
        "llm_few_shot_examples": True,
        "viz_top_n_default": 25,
        "viz_max_top_n": 100,
        "enable_semantic_boost": True,
        "enable_quantitative_bonus": True,
        "cache_embeddings": True,
        "cache_llm_responses": True,
        "log_level": "INFO",
        "export_format_png": True,
        "export_format_svg": False,
        "export_format_html": True,
        "enable_progress_bar": True,
        "fallback_to_embedding_on_error": True,
        "query_driven_processing": True,
        "query_similarity_weight": 0.65,
        "base_salience_weight": 0.35,
        "cache_ttl_minutes": 60
    }

    def __init__(self):
        self._config = self.DEFAULT_CONFIG.copy()
        self._overrides = {}
        logger.info("AppConfig initialized with defaults")

    def get(self, key: str, default=None) -> Any:
        return self._overrides.get(key, self._config.get(key, default))

    def set(self, key: str, value: Any, validate: bool = True):
        if validate and key in self._config:
            expected_type = type(self._config[key])
            if not isinstance(value, expected_type):
                logger.warning(f"Type mismatch for {key}: expected {expected_type}, got {type(value)}. Coercing.")
                try:
                    value = expected_type(value)
                except (ValueError, TypeError) as e:
                    logger.error(f"Failed to coerce {key}: {e}")
                    return
        self._overrides[key] = value
        logger.debug(f"Config updated: {key} = {value}")

    def load_from_dict(self, config_dict: Dict[str, Any]):
        self._overrides.update({k: v for k, v in config_dict.items() if k in self._config})

    def to_dict(self) -> Dict[str, Any]:
        return {**self._config, **self._overrides}

    def reset(self):
        self._overrides.clear()
        logger.info("Configuration reset to defaults")

# Initialize global config
app_config = AppConfig()

# =====================================================================
# GLOBAL CONFIGURATION CONSTANTS
# =====================================================================
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
}

# -------------------------------------------
# DECLARMIMA-aligned laser/materials keywords
# -------------------------------------------
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
MATERIAL_ALIASES = {
    "silicon": ["silicon", "si", "crystalline silicon", "c-si", "si(100)", "si(111)"],
    "titanium": ["titanium", "ti", "cp-ti", "ti-6al-4v", "ti6al4v"],
    "steel": ["steel", "stainless steel", "ss304", "ss316", "mild steel", "carbon steel"],
    "aluminum": ["aluminum", "aluminium", "al", "al6061", "al-6061"],
    "copper": ["copper", "cu"],
    "tungsten": ["tungsten", "w"],
    "glass": ["glass", "fused silica", "sio2", "borosilicate"],
    "polymer": ["polymer", "pmma", "polyimide", "pei", "pc", "polycarbonate", "ptfe"],
    "ceramic": ["ceramic", "alumina", "al2o3", "zirconia", "zro2"],
    "Sn-Ag-Cu": ["snagcu", "sac", "sn-ag-cu", "sn-3.5ag-0.5cu", "solder", "lead-free solder"],
    "Al-Cr-Fe-Ni": ["alcrfeni", "al-cr-fe-ni", "inconel 718", "in718", "nickel superalloy"],
    "high entropy alloy": ["hea", "multi-principal element alloy", "mpea", "cocrfeni", "cocrfenimn",
                           "alcocrfeni", "crmnfeconi", "refractory hea"],
    "multicomponent alloy": ["multicomponent alloy", "multi-component alloy", "multicomponent", "multi-component",
                             "complex concentrated alloy", "cca", "multicomponent system", "multicomponent metallic"],
}
METHOD_ALIASES = {
    "sem": ["sem", "scanning electron microscopy", "scanning electron microscope"],
    "afm": ["afm", "atomic force microscopy", "atomic force microscope"],
    "profilometry": ["profilometry", "optical profilometry", "white light interferometry", "wli"],
    "raman": ["raman", "raman spectroscopy", "micro-raman"],
    "xrd": ["xrd", "x-ray diffraction"],
    "edx": ["edx", "eds", "energy dispersive x-ray", "energy-dispersive"],
    "ebsd": ["ebsd", "electron backscatter diffraction"],
    "x-ray_imaging": ["synchrotron x-ray", "x-ray radiography", "x-ray tomography"],
    "phase_field": ["phase-field", "phase field", "pf simulation"],
    "finite_element": ["finite element", "fem", "moose", "abaqus"],
    "calphad": ["calphad", "thermo-calc", "thermocalc", "pandat"],
}
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

# =====================================================================
# QUANTITATIVE QUERY INTENT ENGINE
# =====================================================================
class QuantitativeQueryEngine:
    """
    Maps user natural language to quantity entity labels and detects
    whether they want values grouped by material, document, or method.
    """
    QUANTITY_SYNONYMS: Dict[str, List[str]] = {
        "laser_power": ["laser power", "laserpower", "nominal power", "average power", "beam power", "power"],
        "fluence": ["fluence", "laser fluence", "energy density", "threshold fluence", "fluence threshold"],
        "wavelength": ["wavelength", "lambda", "λ", "laser wavelength", "emission wavelength"],
        "pulse_duration": ["pulse duration", "pulse width", "pulse length", "fwhm", "pulse time"],
        "repetition_rate": ["repetition rate", "rep rate", "repetition frequency", "pulse frequency", "frequency"],
        "spot_size": ["spot size", "spot diameter", "beam radius", "beam waist", "focal spot", "spot"],
        "scan_speed": ["scan speed", "scanning speed", "travel speed", "writing speed", "scan rate"],
        "hatch_distance": ["hatch distance", "hatch spacing", "line spacing", "hatch"],
        "pulse_energy": ["pulse energy", "energy per pulse", "single pulse energy", "pulse power"],
        "roughness": ["roughness", "surface roughness", "ra", "rms", "rq", "surface finish"],
        "periodicity": ["periodicity", "period", "spacing", "lsfl", "hsfl", "periodic spacing"],
        "threshold": ["threshold", "ablation threshold", "damage threshold", "threshold fluence"],
        "interfacial_energy": ["interfacial energy", "surface tension", "interfacial tension", "surface energy"],
        "thermal_conductivity": ["thermal conductivity", "heat conductivity", "thermal diffusivity"],
        "component_fraction": ["composition", "at%", "wt%", "atomic percent", "weight percent", "concentration"],
    }

    @classmethod
    def detect_quantity(cls, query: str) -> Optional[str]:
        q = query.lower()
        for qty_key, synonyms in cls.QUANTITY_SYNONYMS.items():
            if any(syn in q for syn in synonyms):
                return qty_key
        return None

    @classmethod
    def detect_grouping_dimension(cls, query: str) -> str:
        q = query.lower()
        if any(x in q for x in ["material", "alloy", "substrate", "metal", "composition", "system"]):
            return "material"
        if any(x in q for x in ["document", "paper", "study", "article", "publication"]):
            return "document"
        if any(x in q for x in ["method", "technique", "process", "setup", "approach"]):
            return "method"
        return "material"  # sensible default for laser materials science

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

# =====================================================================
# HIERARCHICAL TAXONOMY FOR VISUALIZATION
# =====================================================================
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

def classify_entity(normalized: str) -> Tuple[str, str, str]:
    """Return (domain, category, subcategory) for an entity string."""
    norm = normalized.lower().strip()
    def _search_level(node, path):
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

# =====================================================================
# BIBLIOGRAPHIC METADATA
# =====================================================================
class BibliographicMetadata:
    DOI_PATTERN = re.compile(r'\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b', re.IGNORECASE)
    ARXIV_PATTERN = re.compile(r'\barXiv[:\s]+(\d{4}\.\d{4,5}(v\d+)?)\b', re.IGNORECASE)
    JOURNAL_PATTERNS = [
        re.compile(r'(?:published in|journal|proc\.?|journal of)\s+([A-Z][A-Za-z\s&\.]+?)(?:,|\.)', re.I),
        re.compile(r'([A-Z][A-Za-z\s&\.]+?\s+(?:Letters?|Journal|Transactions|Review|Proceedings))', re.I),
    ]
    YEAR_PATTERN = re.compile(r'\b((?:19|20)\d{2})\b')
    VOLUME_PATTERN = re.compile(r'(?:vol\.?|volume)\s*(\d+)', re.I)
    ISSUE_PATTERN = re.compile(r'(?:no\.?|issue|iss\.?)\s*(\d+)', re.I)
    AUTHOR_PATTERN = re.compile(
        r'(?:^|by|authors?:\s*)([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*)',
        re.MULTILINE
    )

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
            if style == "doi":
                return f"DOI:{self.doi}"
            elif style == "short":
                return f"[DOI:{self.doi}]"
        if self.arxiv_id:
            if style in ["doi", "short"]:
                return f"[arXiv:{self.arxiv_id}]"
        if self.authors and self.year:
            first_author = self._format_author_name(self.authors[0])
            et_al = " et al." if len(self.authors) > 1 else ""
            if style == "apa":
                journal_part = f", {self.journal}" if self.journal else ""
                return f"{first_author}{et_al}{journal_part}, {self.year}"
            elif style == "short":
                return f"[{first_author.split()[0]} {self.year}]"
            elif style == "full":
                parts = [f"{first_author}{et_al} ({self.year})"]
                if self.title:
                    parts.append(f'"{self.title}"')
                if self.journal:
                    journal_str = self.journal
                    if self.volume:
                        journal_str += f", {self.volume}"
                    if self.issue:
                        journal_str += f"({self.issue})"
                    parts.append(journal_str)
                if self.pages:
                    parts.append(f"pp. {self.pages}")
                return ". ".join(parts) + "."
        base_name = Path(self.source_filename).stem
        if self.year:
            return f"[{base_name}, {self.year}]"
        return f"[{base_name}]"

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
            "source": self.source_filename,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "journal": self.journal,
            "year": self.year,
            "volume": self.volume,
            "issue": self.issue,
            "pages": self.pages,
            "publisher": self.publisher,
            "extraction_method": self.extraction_method,
            "confidence": self.confidence,
            "citation_apa": self.format_citation("apa"),
            "citation_doi": self.format_citation("doi"),
            "citation_full": self.format_citation("full"),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BibliographicMetadata':
        meta = cls(data.get("source", "unknown"))
        meta.doi = data.get("doi")
        meta.arxiv_id = data.get("arxiv_id")
        meta.title = data.get("title")
        meta.authors = data.get("authors", [])
        meta.journal = data.get("journal")
        meta.year = data.get("year")
        meta.volume = data.get("volume")
        meta.issue = data.get("issue")
        meta.pages = data.get("pages")
        meta.publisher = data.get("publisher")
        meta.extraction_method = data.get("extraction_method", "cached")
        meta.confidence = data.get("confidence", 0.5)
        return meta

def extract_metadata_from_pdf_text(text: str, filename: str) -> BibliographicMetadata:
    meta = BibliographicMetadata(filename)
    text_sample = text[:10000]
    doi_match = BibliographicMetadata.DOI_PATTERN.search(text_sample)
    if doi_match:
        meta.doi = doi_match.group(1).lower()
        meta.confidence = max(meta.confidence, 0.9)
        meta.extraction_method = "regex_doi"
    arxiv_match = BibliographicMetadata.ARXIV_PATTERN.search(text_sample)
    if arxiv_match:
        meta.arxiv_id = arxiv_match.group(1)
        meta.confidence = max(meta.confidence, 0.85)
    year_matches = BibliographicMetadata.YEAR_PATTERN.findall(text_sample)
    for year_str in year_matches:
        year = int(year_str)
        if 1900 <= year <= 2030:
            year_pos = text_sample.find(year_str)
            context = text_sample[max(0, year_pos - 50):year_pos + 50].lower()
            if any(kw in context for kw in ['published', 'received', 'accepted', 'copyright', '©']):
                meta.year = year
                meta.confidence = max(meta.confidence, 0.7)
                break
    for pattern in BibliographicMetadata.JOURNAL_PATTERNS:
        journal_match = pattern.search(text_sample)
        if journal_match:
            journal = journal_match.group(1).strip()
            if len(journal) > 10 and not any(
                bad in journal.lower() for bad in ['introduction', 'abstract', 'references']
            ):
                meta.journal = journal
                meta.confidence = max(meta.confidence, 0.6)
                break
    vol_match = BibliographicMetadata.VOLUME_PATTERN.search(text_sample)
    if vol_match:
        meta.volume = vol_match.group(1)
    iss_match = BibliographicMetadata.ISSUE_PATTERN.search(text_sample)
    if iss_match:
        meta.issue = iss_match.group(1)
    author_section = text_sample[:2000]
    author_matches = BibliographicMetadata.AUTHOR_PATTERN.findall(author_section)
    if author_matches:
        raw_authors = author_matches[0]
        if ',' in raw_authors or ' and ' in raw_authors.lower():
            separators = [',', ' and ', ';']
            for sep in separators:
                if sep.lower() in raw_authors.lower():
                    meta.authors = [a.strip() for a in re.split(sep, raw_authors, flags=re.I) if a.strip()]
                    break
        else:
            meta.authors = [raw_authors.strip()]
        if meta.authors:
            meta.confidence = max(meta.confidence, 0.5)
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
            field_mapping = {'/Title': 'title', '/Author': 'authors', '/CreationDate': 'year', '/Subject': 'journal'}
            for pdf_field, meta_field in field_mapping.items():
                if pdf_field in pdf_info and pdf_info[pdf_field]:
                    value = str(pdf_info[pdf_field]).strip()
                    if meta_field == 'authors' and value:
                        meta.authors = [a.strip() for a in re.split(r'[;,]', value) if a.strip()]
                    elif meta_field == 'year' and value:
                        year_match = re.search(r'(?:D:)?(\d{4})', value)
                        if year_match:
                            meta.year = int(year_match.group(1))
                    else:
                        setattr(meta, meta_field, value)
            if meta.title or meta.authors:
                meta.confidence = 0.7
                meta.extraction_method = "pdf_metadata"
        except Exception as e:
            st.warning(f"Could not read PDF metadata: {e}")
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        text_sample = "\n".join([p.page_content for p in pages[:3]])
        text_meta = extract_metadata_from_pdf_text(text_sample, filename)
        for field in ['doi', 'arxiv_id', 'title', 'journal', 'year', 'volume', 'issue']:
            text_val = getattr(text_meta, field)
            current_val = getattr(meta, field)
            if text_val and (not current_val or text_meta.confidence > meta.confidence):
                setattr(meta, field, text_val)
        if text_meta.authors and (not meta.authors or text_meta.confidence > meta.confidence):
            meta.authors = text_meta.authors
        if text_meta.confidence > meta.confidence:
            meta.confidence = text_meta.confidence
            meta.extraction_method = text_meta.extraction_method
    except Exception as e:
        st.warning(f"Text extraction for metadata failed: {e}")
    if PDF2DOI_AVAILABLE and not meta.doi:
        try:
            result = pdf2doi.pdf2doi(pdf_path)
            if isinstance(result, list) and result:
                result = result[0]
            if result and result.get('identifier') and result.get('identifier_type') == 'doi':
                meta.doi = result['identifier']
                meta.confidence = 0.95
                meta.extraction_method = "pdf2doi"
                if result.get('validation_info'):
                    bibtex = result['validation_info']
                    if 'title' in bibtex and not meta.title:
                        meta.title = bibtex.get('title')
                    if 'author' in bibtex and not meta.authors:
                        meta.authors = [a.strip() for a in bibtex['author'].split(' and ')]
                    if 'year' in bibtex and not meta.year:
                        try:
                            meta.year = int(bibtex['year'])
                        except:
                            pass
        except Exception as e:
            st.warning(f"pdf2doi lookup failed: {e}")
    if CROSSREF_AVAILABLE and meta.doi and not meta.journal:
        try:
            cr = CrossrefAPI()
            work = cr.works(ids=meta.doi)
            if work and work.get('message'):
                msg = work['message']
                if not meta.title and msg.get('title'):
                    meta.title = msg['title'][0] if isinstance(msg['title'], list) else msg['title']
                if not meta.authors and msg.get('author'):
                    meta.authors = [
                        f"{a.get('family', '')} {a.get('given', '')}".strip()
                        for a in msg['author']
                    ]
                if not meta.journal and msg.get('container-title'):
                    meta.journal = msg['container-title'][0] if isinstance(msg['container-title'], list) else msg['container-title']
                if not meta.year and msg.get('published-print') and msg['published-print'].get('date-parts'):
                    meta.year = msg['published-print']['date-parts'][0][0]
                meta.confidence = 0.98
                meta.extraction_method = "crossref_api"
        except Exception as e:
            st.warning(f"Crossref API lookup failed: {e}")
    return meta

def extract_metadata_from_text_file(text: str, filename: str) -> BibliographicMetadata:
    return extract_metadata_from_pdf_text(text, filename)

class MetadataCache:
    def __init__(self):
        self._cache: Dict[str, BibliographicMetadata] = {}
        self._file_hashes: Dict[str, str] = {}

    def get(self, filename: str, file_hash: str = None) -> Optional[BibliographicMetadata]:
        if filename in self._cache:
            if file_hash is None or self._file_hashes.get(filename) == file_hash:
                return self._cache[filename]
        return None

    def set(self, filename: str, metadata: BibliographicMetadata, file_hash: str = None):
        self._cache[filename] = metadata
        if file_hash:
            self._file_hashes[filename] = file_hash

    def clear(self):
        self._cache.clear()
        self._file_hashes.clear()

metadata_cache = MetadataCache()

def compute_file_hash(filepath: str) -> str:
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return ""

# =====================================================================
# ENHANCED SCIENTIFIC ENTITY & CLAIM
# =====================================================================
@dataclass
class EnhancedScientificEntity:
    text: str
    label: str
    value: Optional[float]
    unit: Optional[str]
    doc_source: str
    chunk_id: int
    context: str
    confidence: float = 1.0
    llm_validated: bool = False
    llm_importance_score: float = 0.0
    normalized: str = field(init=False)
    domain: str = field(init=False)
    category: str = field(init=False)
    subcategory: str = field(init=False)
    query_relevance_score: float = 0.0  # NEW: Dynamic query alignment

    def __post_init__(self):
        self.normalized = self._normalize()
        self.domain, self.category, self.subcategory = classify_entity(self.normalized)

    def _normalize(self) -> str:
        text = self.text.lower().strip()
        for canonical, aliases in MATERIAL_ALIASES.items():
            if any(alias in text for alias in aliases):
                return canonical
        for canonical, aliases in METHOD_ALIASES.items():
            if any(alias in text for alias in aliases):
                return canonical
        text = re.sub(r'\s+', '', text)
        return text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text, "label": self.label, "value": self.value, "unit": self.unit,
            "doc_source": self.doc_source, "chunk_id": self.chunk_id,
            "normalized": self.normalized, "confidence": self.confidence,
            "domain": self.domain, "category": self.category, "subcategory": self.subcategory,
            "llm_validated": self.llm_validated, "llm_importance_score": self.llm_importance_score,
            "context": self.context[:200],
            "query_relevance_score": self.query_relevance_score
        }

@dataclass
class EnhancedScientificClaim:
    claim_text: str
    subject: str
    predicate: str
    object_val: str
    doc_source: str
    chunk_id: int
    confidence: float
    llm_refined: bool = False
    supporting: List[Tuple[str, int]] = field(default_factory=list)
    contradicting: List[Tuple[str, int]] = field(default_factory=list)
    query_alignment: float = 0.0  # NEW: How well the claim matches user intent

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim_text, "subject": self.subject, "predicate": self.predicate,
            "object": self.object_val, "source": self.doc_source, "confidence": self.confidence,
            "llm_refined": self.llm_refined,
            "supporting_count": len(self.supporting), "contradicting_count": len(self.contradicting),
            "query_alignment": self.query_alignment
        }

# =====================================================================
# DECLARMIMA PROPOSAL TEXT (used for salience seeding)
# =====================================================================
DECLARMIMA_PROPOSAL_TEXT = """Deciphering laser-microstructure interaction in multicomponent alloys (DECLARMIMA) Scientific goals: Additive manufacturing, laser processing, multicomponent alloys, high-entropy alloys, digital twins, physics-informed machine learning, phase field modeling, molecular dynamics, melt pool dynamics, microstructure evolution, process-structure-property relationships, selective laser melting, powder bed fusion, laser powder bed fusion, in-situ monitoring, defect formation, porosity, spatter, residual stress, grain morphology, phase transformation, solidification, Marangoni convection, CALPHAD thermodynamics, interfacial energy, thermal conductivity, viscosity, absorptivity, reflectivity, Gaussian heat source, finite element method, MOOSE framework, LAMMPS, ThermoCalc, neural networks, convolutional neural networks, random forest, Bayesian machine learning, uncertainty quantification, feature engineering, tensor decomposition, scale-bridging, multiscale modeling, inverse design, optimization, Al-Si-Mg alloys, Ti-6Al-4V, Inconel 718, Sn-Ag-Cu solders, CoCrFeNi HEAs, intermetallic compounds, columnar grains, equiaxed grains, dendritic structures, martensite, austenite, precipitates, segregation, crack propagation, fatigue life, tensile strength, yield strength, microhardness, elongation, ductility, wear resistance, corrosion resistance, oxidation resistance, laser power, scan speed, hatch spacing, layer thickness, pulse duration, energy density, spot diameter, cooling rate, solidification rate, dilution ratio, powder particle size, particle size distribution, flowability, oxygen content, moisture content, bed temperature, pre-heating, post-processing, heat treatment, surface finishing, quality monitoring, photodiode sensors, line scanners, camera trackers, acoustic transducers, synchrotron X-ray imaging, EBSD, nanoindentation, in-situ XRD, SEM, TEM, AFM, digital image correlation, machine vision, data fusion, knowledge graphs, concept graphs, graph neural networks, GraphSAGE, node embeddings, edge prediction, link prediction, research direction discovery, hypothesis generation, novelty scoring, feasibility assessment, property gain prediction, composite scoring, adaptive configuration, small corpus optimization, semantic clustering, domain seed injection, hybrid graph construction, co-occurrence edges, semantic similarity edges, contrastive learning, edge sampling, sparse tensors, degree normalization, mean aggregation, two-layer architecture, decoder network, BCE loss, Adam optimizer, training loop, evaluation metrics, progress tracking, memory management, CUDA optimization, CPU fallback, error handling, fallback strategies, interactive visualization, PyVis, Plotly, force-directed layout, spring layout, node styling, edge styling, hover tooltips, download functionality, text fallback, diagnostics panel, concept frequency, edge weight, graph connectivity, component analysis, degree distribution, clustering coefficient, centrality measures, path length, bridge edges, semantic bridges, knowledge injection, concept normalization, alloy notation standardization, laser term normalization, unit standardization, regex extraction, quantitative metrics, grain size, mechanical properties, energy density, defect fraction, prompt engineering, JSON parsing, fallback extraction, domain validation, generic term filtering, concept abstraction, category mapping, hierarchical representation, representative selection, cluster merging, similarity threshold, distance matrix, linkage method, embedding encoding, batch processing, progress display, model caching, resource management, timeout handling, user feedback, status indicators, progress bars, error messages, warning dialogs, success notifications, download buttons, CSV export, HTML export, JSON export, interactive controls, physics parameters, gravity, spring length, damping, overlap, stabilization, node sampling, size limiting, performance optimization, browser compatibility, JavaScript execution, CDN resources, inline embedding, iframe alternative, HTML rendering, Streamlit components, responsive design, mobile compatibility, accessibility, color contrast, theme switching, dark mode, light mode, user preferences, session state, configuration persistence, adaptive thresholds, corpus size detection, parameter tuning, hyperparameter optimization, validation metrics, testing framework, debugging tools, logging, tracebacks, exception handling, graceful degradation, fallback rendering, text summary, edge listing, frequency tables, diagnostic metrics, connectivity checks, component counting, degree analysis, clustering analysis, centrality computation, path analysis, bridge detection, semantic analysis, novelty computation, feasibility scoring, property prediction, ridge regression, feature concatenation, pair scoring, candidate filtering, distance checking, graph distance, shortest path, all-pairs shortest path, cutoff parameter, edge sampling strategy, positive pairs, negative pairs, hard negatives, distance-focused sampling, random sampling, attempts limit, pair uniqueness, edge existence check, tensor construction, sparse adjacency, degree computation, normalization, message passing, aggregation, combination, activation, ReLU, linear layers, sequential decoder, concatenation, sigmoid, logits, contrastive loss, binary cross-entropy, training epochs, learning rate, optimizer step, gradient computation, backward pass, zero grad, model evaluation, no grad context, final embeddings, adjacency indices, adjacency values, node features, embedding dimension, shape validation, error raising, minimal pairs, edge uniqueness, source adjacency, destination adjacency, stacking, tensor conversion, device placement, long dtype, float32, GPU memory, CPU fallback, memory cleanup, garbage collection, CUDA cache emptying, progress callback, epoch logging, loss tracking, convergence monitoring, early stopping, model saving, checkpointing, inference mode, prediction scoring, candidate generation, random sampling, pair filtering, distance computation, KeyError handling, default distance, semantic similarity, cosine similarity, embedding encoding, numpy arrays, tensor conversion, CPU numpy, forward pass, model eval, no grad, decoder output, logits extraction, sigmoid activation, CPU conversion, numpy array, property lookup, median computation, ridge prediction, clipping, normalization, weighted scoring, alpha weights, composite score, sorting, head selection, DataFrame creation, column selection, formatting, display configuration, download preparation, CSV serialization, MIME type, button callback, empty check, info message, parameter suggestion, graph rendering, node count check, edge count check, fallback graph building, semantic-only fallback, similarity threshold adjustment, success message, text fallback rendering, node iteration, degree computation, frequency lookup, category detection, color assignment, size computation, title formatting, node addition, edge iteration, weight lookup, type lookup, color mapping, edge addition, value scaling, width scaling, color assignment, smooth edges, curved edges, roundness parameter, HTML generation, inline resources, Streamlit HTML component, height parameter, scrolling enable, width parameter, download button, file naming, MIME type, unique key, error catching, warning display, fallback suggestion, retry buttons, alternative backend, exception handling, error message display, traceback expansion, code display, memory cleanup, GPU cache clearing, garbage collection, footer display, tips section, visualization options, PyVis description, Plotly description, text summary description, technical stack, crash prevention tips, rendering troubleshooting, browser console check, zoom controls, download fallback, text view guarantee"""

# =====================================================================
# FULL-TEXT CONCEPT EXTRACTOR WITH ENHANCED SALIENCE SCORING
# =====================================================================
class FullTextConceptExtractor:
    """
    Extracts scientific concepts from full‑text PDF chunks with multi‑factor salience.
    Works with both SentenceTransformer and HuggingFaceEmbeddings (LangChain).
    UPGRADED FEATURES:
    - Core Pillars: LASER, MICROSTRUCTURE, INTERACTION, MULTICOMPONENT ALLOY
    - Semantic similarity: concepts embedding-similar to pillars get boosted
    - Query-biased salience: dynamically aligns extraction with user intent
    - Expanded domain seeds with multicomponent alloy family
    """
    def __init__(self, embed_model, proposal_text: str = None):
        self.embed_model = embed_model
        self.proposal_text = proposal_text or DECLARMIMA_PROPOSAL_TEXT
        self.proposal_embedding = self._embed_text(self.proposal_text)
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
        }
        self.domain_seeds = {
            "melt pool": 0.95, "keyhole": 0.94, "marangoni convection": 0.92,
            "porosity": 0.90, "spatter": 0.88, "intermetallic compound": 0.90,
            "columnar to equiaxed": 0.87, "residual stress": 0.88,
            "solidification": 0.85, "grain morphology": 0.82,
            "high entropy alloy": 0.94, "hea": 0.94, "mpea": 0.93,
            "multi-principal element alloy": 0.93, "complex concentrated alloy": 0.92,
            "cocrfeni": 0.90, "alcocrfeni": 0.90, "crmnfeconi": 0.90,
            "refractory hea": 0.89, "alcrfeni": 0.89,
            "sn-ag-cu": 0.85, "sac solder": 0.85, "inconel 718": 0.85,
            "ti-6al-4v": 0.85, "coCrFeNi": 0.88,
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
        if hasattr(self.embed_model, 'embed_query'):
            return np.array(self.embed_model.embed_query(text))
        elif hasattr(self.embed_model, 'encode'):
            return self.embed_model.encode(text)
        else:
            raise AttributeError("Embedding model has neither 'embed_query' nor 'encode' method")

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        if hasattr(self.embed_model, 'embed_documents'):
            return np.array(self.embed_model.embed_documents(texts))
        elif hasattr(self.embed_model, 'encode'):
            return self.embed_model.encode(texts, show_progress_bar=False)
        else:
            return np.array([self._embed_text(t) for t in texts])

    def set_custom_priority(self, concepts: List[str]):
        if concepts:
            self.custom_priority = {c.lower().strip(): 0.88 for c in concepts if c.strip()}
        else:
            self.custom_priority = {}

    def _compute_semantic_pillar_embeddings(self):
        if not self._pillar_embeddings:
            for pillar in self.core_pillars:
                self._pillar_embeddings[pillar] = self._embed_text(pillar)

    def _get_semantic_boost(self, concept: str, concept_embedding: np.ndarray) -> float:
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

    def _extract_candidates_fast(self, chunks: List[Document]) -> List[str]:
        """Vectorized candidate extraction using compiled regex and batch text search."""
        candidates = set()
        keyword_patterns = {}
        for topic, keywords in LASER_KEYWORDS.items():
            for kw in keywords:
                keyword_patterns[kw.lower()] = re.compile(r'\b' + re.escape(kw.lower()) + r'\b')
        mca_patterns = [
            re.compile(r' (?:multicomponent|multi-component|multi\s+component)\s+(?:alloy|system|material|metallic) ', re.I),
            re.compile(r' (?:high\s+entropy|complex\s+concentrated)\s+(?:alloy|alloys) ', re.I),
            re.compile(r' (?:multi-principal|multiprincipal)\s+(?:element|elemental)\s+(?:alloy|alloys) ', re.I),
            re.compile(r' (?:cocrfeni|alcocrfeni|crmnfeconi|alcrfeni|cocrfenimn) ', re.I),
            re.compile(r' (?:hea|mpea|cca)\s+(?:system|alloy|family|composition) ', re.I),
        ]
        for chunk in chunks:
            text = chunk.page_content.lower()
            for kw, pattern in keyword_patterns.items():
                if pattern.search(text):
                    candidates.add(kw)
            for canonical in list(MATERIAL_ALIASES.keys()) + list(METHOD_ALIASES.keys()):
                if canonical.lower() in text:
                    candidates.add(canonical.lower())
            for match in re.finditer(r'(\d+(?:\.\d+)?)\s*(?:μm|um|nm|%|J/mm³|HV|MPa|W|mm/s)', text):
                context = text[max(0, match.start()-60):match.end()+60]
                candidates.add(context.strip()[:80])
            for pattern in mca_patterns:
                for match in pattern.finditer(text):
                    candidates.add(match.group(0).lower().strip()[:60])
        return list(candidates)

    @functools.lru_cache(maxsize=1024)
    def _embed_text_cached(self, text: str) -> np.ndarray:
        """Cached embedding for repeated texts."""
        return self._embed_text(text)

    def extract_concepts_fast(self, chunks: List[Document], min_salience: float = 0.42, 
                              query_embedding: Optional[np.ndarray] = None) -> Tuple[List[str], Dict[str, Dict]]:
        """Optimized concept extraction with caching, batch processing, and query-biased salience."""
        candidates = self._extract_candidates_fast(chunks)
        salience_scores = self._compute_salience_fast(candidates, chunks)
        final_concepts = []
        metadata = {}
        try:
            candidate_embeddings = self._embed_batch(candidates)
        except Exception:
            candidate_embeddings = np.array([self._embed_text_cached(c) for c in candidates])

        for idx, concept in enumerate(candidates):
            base_score = salience_scores.get(concept, 0.0)
            boost = max(
                self.core_pillars.get(concept.lower(), 0.0),
                self.domain_seeds.get(concept.lower(), 0.0),
                self.custom_priority.get(concept.lower(), 0.0)
            )
            semantic_boost = self._get_semantic_boost(concept, candidate_embeddings[idx])
            
            # NEW: Query-biased adjustment
            query_weight = 0.0
            if query_embedding is not None:
                sim = float(np.dot(candidate_embeddings[idx], query_embedding) / 
                            (np.linalg.norm(candidate_embeddings[idx]) * np.linalg.norm(query_embedding) + 1e-8))
                query_weight = max(0.0, min(1.0, sim * app_config.get("query_similarity_weight", 0.65)))
            
            final_score = (base_score * (1.0 + 0.65 * boost + semantic_boost)) * app_config.get("base_salience_weight", 0.35)
            final_score += query_weight
            final_score = np.clip(final_score, 0.0, 1.0)
            
            if final_score >= min_salience or boost >= 0.8 or query_weight > 0.5:
                final_concepts.append(concept)
                metadata[concept] = {
                    "salience": round(float(final_score), 3),
                    "base_salience": round(float(base_score), 3),
                    "query_relevance": round(float(query_weight), 3),
                    "is_core_pillar": concept.lower() in self.core_pillars,
                    "is_domain_seed": concept.lower() in self.domain_seeds,
                    "is_custom": concept.lower() in self.custom_priority,
                    "semantic_boost": round(float(semantic_boost), 3),
                    "frequency": sum(1 for ch in chunks if concept.lower() in ch.page_content.lower())
                }
        final_concepts.sort(key=lambda c: metadata[c]["salience"], reverse=True)
        return final_concepts, metadata

    def _compute_salience_fast(self, candidates: List[str], chunks: List[Document]) -> Dict[str, float]:
        """Optimized salience computation with vectorized operations."""
        scores = {}
        n_docs = len(chunks)
        if n_docs == 0:
            return {}
        try:
            candidate_embeddings = self._embed_batch(candidates)
        except Exception:
            candidate_embeddings = np.array([self._embed_text_cached(c) for c in candidates])
        chunk_sections = [self.section_weights.get(ch.metadata.get("section", "UNKNOWN").upper(), 0.3) for ch in chunks]
        chunk_sources = [ch.metadata.get("source") for ch in chunks]
        for idx, concept in enumerate(candidates):
            matches = [concept in ch.page_content.lower() for ch in chunks]
            freq = sum(matches)
            freq_norm = np.log1p(freq) / np.log1p(n_docs) if n_docs > 0 else 0.0
            docs_with_concept = len(set(chunk_sources[i] for i, m in enumerate(matches) if m))
            cross_doc = docs_with_concept / n_docs if n_docs > 0 else 0.0
            section_scores = [chunk_sections[i] for i, m in enumerate(matches) if m]
            section_imp = np.mean(section_scores) if section_scores else 0.3
            has_number = bool(re.search(r'\d', concept))
            quant_bonus = 1.12 if has_number else 1.0
            emb = candidate_embeddings[idx]
            proposal_sim = float(np.dot(emb, self.proposal_embedding) /
                                 (np.linalg.norm(emb) * np.linalg.norm(self.proposal_embedding) + 1e-8))
            base_salience = (0.25 * freq_norm + 0.20 * cross_doc + 0.18 * section_imp +
                             0.15 * proposal_sim + 0.12 * (1.0 if has_number else 0.6))
            scores[concept] = float(np.clip(base_salience * quant_bonus, 0.0, 1.0))
        return scores

    def extract_concepts(self, chunks: List[Document], min_salience: float = 0.42) -> Tuple[List[str], Dict[str, Dict]]:
        candidates = self._extract_candidates(chunks)
        salience_scores = self._compute_salience(candidates, chunks)
        final_concepts = []
        metadata = {}
        try:
            candidate_embeddings = self._embed_batch(candidates)
        except Exception:
            candidate_embeddings = np.array([self._embed_text(c) for c in candidates])
        for idx, concept in enumerate(candidates):
            base_score = salience_scores.get(concept, 0.0)
            boost = max(
                self.core_pillars.get(concept.lower(), 0.0),
                self.domain_seeds.get(concept.lower(), 0.0),
                self.custom_priority.get(concept.lower(), 0.0)
            )
            semantic_boost = self._get_semantic_boost(concept, candidate_embeddings[idx])
            final_score = base_score * (1.0 + 0.65 * boost + semantic_boost)
            if final_score >= min_salience or boost >= 0.8:
                final_concepts.append(concept)
                metadata[concept] = {
                    "salience": round(float(final_score), 3),
                    "is_core_pillar": concept.lower() in self.core_pillars,
                    "is_domain_seed": concept.lower() in self.domain_seeds,
                    "is_custom": concept.lower() in self.custom_priority,
                    "semantic_boost": round(float(semantic_boost), 3),
                    "frequency": sum(1 for ch in chunks if concept.lower() in ch.page_content.lower())
                }
        final_concepts.sort(key=lambda c: metadata[c]["salience"], reverse=True)
        return final_concepts, metadata

    def _extract_candidates(self, chunks: List[Document]) -> List[str]:
        candidates = set()
        for chunk in chunks:
            text = chunk.page_content.lower()
            for topic, keywords in LASER_KEYWORDS.items():
                for kw in keywords:
                    if kw.lower() in text:
                        candidates.add(kw.lower())
            for canonical in list(MATERIAL_ALIASES.keys()) + list(METHOD_ALIASES.keys()):
                if canonical.lower() in text:
                    candidates.add(canonical.lower())
            for match in re.finditer(r'(\d+(?:\.\d+)?)\s*(?:μm|um|nm|%|J/mm³|HV|MPa|W|mm/s)', text):
                context = text[max(0, match.start()-60):match.end()+60]
                candidates.add(context.strip()[:80])
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

    def _compute_salience(self, candidates: List[str], chunks: List[Document]) -> Dict[str, float]:
        scores = {}
        n_docs = len(chunks)
        if n_docs == 0:
            return {}
        try:
            candidate_embeddings = self._embed_batch(candidates)
        except Exception as e:
            candidate_embeddings = np.array([self._embed_text(c) for c in candidates])
        for idx, concept in enumerate(candidates):
            freq = sum(1 for ch in chunks if concept in ch.page_content.lower())
            freq_norm = np.log1p(freq) / np.log1p(n_docs) if n_docs > 0 else 0.0
            docs_with_concept = len({ch.metadata.get("source") for ch in chunks if concept in ch.page_content.lower()})
            cross_doc = docs_with_concept / n_docs if n_docs > 0 else 0.0
            section_scores = [self.section_weights.get(ch.metadata.get("section", "UNKNOWN").upper(), 0.3)
                              for ch in chunks if concept in ch.page_content.lower()]
            section_imp = np.mean(section_scores) if section_scores else 0.3
            has_number = bool(re.search(r'\d', concept))
            quant_bonus = 1.12 if has_number else 1.0
            emb = candidate_embeddings[idx]
            proposal_sim = float(np.dot(emb, self.proposal_embedding) /
                                 (np.linalg.norm(emb) * np.linalg.norm(self.proposal_embedding) + 1e-8))
            base_salience = (0.25 * freq_norm + 0.20 * cross_doc + 0.18 * section_imp +
                             0.15 * proposal_sim + 0.12 * (1.0 if has_number else 0.6))
            scores[concept] = float(np.clip(base_salience * quant_bonus, 0.0, 1.0))
        return scores

# =====================================================================
# LLM-INFLUENCED CONCEPT EXTRACTOR
# =====================================================================
class LLMEnhancedConceptExtractor:
    """
    Uses the loaded LLM to refine, validate, and rank extracted concepts.
    Performs few-shot entity disambiguation, importance scoring, and context-aware filtering.
    """
    PROMPT_TEMPLATE = """
You are an expert scientific curator analyzing laser-microstructure interaction literature.
Given a list of raw extracted candidate concepts, perform the following:
1. Remove duplicates, trivial terms, and generic words.
2. Normalize to standard scientific terminology.
3. Assign an importance score (0.0 to 1.0) based on relevance to:
- Laser processing parameters
- Multicomponent alloy systems
- Microstructural evolution
- Physical mechanisms
4. Return ONLY a valid JSON list of objects with keys:
"concept", "normalized", "importance", "domain" (MATERIAL/METHOD/PHENOMENON/PARAMETER)
CANDIDATES:
{candidates}
JSON OUTPUT:
"""
    def __init__(self, llm_generate_fn: Callable, batch_size: int = 8, timeout: int = 30):
        self.llm_generate_fn = llm_generate_fn
        self.batch_size = batch_size
        self.timeout = timeout
        logger.info("LLM-Enhanced Concept Extractor initialized")

    def extract_and_rank(self, raw_candidates: List[str], context_sample: str = "") -> List[Dict[str, Any]]:
        if not raw_candidates:
            return []
        batches = [raw_candidates[i:i + self.batch_size] for i in range(0, len(raw_candidates), self.batch_size)]
        ranked_results = []
        for batch_idx, batch in enumerate(batches):
            prompt = self.PROMPT_TEMPLATE.format(candidates="\n".join(f"- {c}" for c in batch))
            try:
                start_time = time.time()
                response = self.llm_generate_fn(prompt)
                if time.time() - start_time > self.timeout:
                    logger.warning(f"LLM extraction timeout for batch {batch_idx}")
                    continue
                json_str = self._extract_json_block(response)
                if json_str:
                    batch_results = json.loads(json_str)
                    for item in batch_results:
                        if isinstance(item, dict) and "concept" in item and "importance" in item:
                            ranked_results.append({
                                "concept": item["concept"],
                                "normalized": item.get("normalized", item["concept"]),
                                "importance": float(item["importance"]),
                                "domain": item.get("domain", "UNKNOWN"),
                                "llm_source": True
                            })
                        else:
                            logger.warning(f"LLM failed to return valid JSON for batch {batch_idx}")
                else:
                    logger.warning(f"LLM failed to return valid JSON for batch {batch_idx}")
            except Exception as e:
                logger.error(f"LLM extraction error batch {batch_idx}: {e}")
                continue
        if not ranked_results:
            logger.warning("LLM extraction failed entirely. Falling back to embedding scores.")
            ranked_results = [{"concept": c, "normalized": c, "importance": 0.5, "domain": "UNKNOWN", "llm_source": False} for c in raw_candidates]
        seen = set()
        unique_results = []
        for r in ranked_results:
            norm = r["normalized"].lower()
            if norm not in seen:
                seen.add(norm)
                unique_results.append(r)
        unique_results.sort(key=lambda x: x["importance"], reverse=True)
        return unique_results

    def _extract_json_block(self, text: str) -> Optional[str]:
        """Extract the first valid JSON array from LLM output."""
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                json.loads(match.group(0))
                return match.group(0)
            except json.JSONDecodeError:
                pass
        match = re.search(r'```json\s*(\[.*?\])\s*```', text, re.DOTALL)
        if match:
            try:
                json.loads(match.group(1))
                return match.group(1)
            except json.JSONDecodeError:
                pass
        return None

    def validate_claim_with_llm(self, claim: EnhancedScientificClaim) -> EnhancedScientificClaim:
        """Use LLM to refine or flag uncertain claims."""
        prompt = f"""
Scientific Claim Verification:
Claim: "{claim.claim_text}"
Subject: {claim.subject}
Predicate: {claim.predicate}
Object: {claim.object_val}
Is this claim scientifically coherent and extractable? Return JSON: {"verified": true/false, "confidence": 0.0-1.0, "refined_text": "..."}
"""
        try:
            response = self.llm_generate_fn(prompt)
            json_str = self._extract_json_block(response)
            if json_str:
                data = json.loads(json_str)
                claim.confidence = data.get("confidence", claim.confidence)
                claim.llm_refined = data.get("verified", True)
                if "refined_text" in data:
                    claim.claim_text = data["refined_text"]
        except Exception as e:
            logger.error(f"LLM claim validation failed: {e}")
        return claim

# =====================================================================
# REASONING CHAIN (THINKING TRACE)
# =====================================================================
@dataclass
class ReasoningStep:
    step_type: str
    description: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class ReasoningChain:
    """Explicit chain-of-thought / thinking trace for cross-document synthesis."""
    def __init__(self, query: str):
        self.query = query
        self.steps: List[ReasoningStep] = []
        self.thinking_graph: Optional[nx.DiGraph] = None

    def add_step(self, step_type: str, description: str, data: Dict[str, Any]):
        self.steps.append(ReasoningStep(step_type, description, data))

    def build_thinking_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        G.add_node("QUERY", node_type="query", text=self.query, layer=0)
        prev_node = "QUERY"
        for i, step in enumerate(self.steps):
            node_id = f"STEP_{i}_{step.step_type}"
            G.add_node(node_id, node_type=step.step_type, description=step.description,
                       layer=i+1, timestamp=step.timestamp.isoformat())
            G.add_edge(prev_node, node_id, relation="leads_to")
            if "entities" in step.data:
                for ent in step.data["entities"]:
                    ent_id = f"ENT_{ent}_{i}"
                    G.add_node(ent_id, node_type="entity", name=ent, layer=i+1)
                    G.add_edge(node_id, ent_id, relation="involves")
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
        lines = [f"### 🧠 Reasoning Trace: *{self.query}*", ""]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"**Step {i} — {step.step_type}**  ")
            lines.append(f"{step.description}  ")
            if step.data:
                lines.append(f"`{json.dumps(step.data, default=str)[:300]}`  ")
            lines.append("")
        return "\n".join(lines)

# =====================================================================
# ENHANCED CROSS-DOCUMENT KNOWLEDGE GRAPH
# =====================================================================
class EnhancedCrossDocumentKnowledgeGraph:
    def __init__(self):
        self.entities: Dict[str, List[EnhancedScientificEntity]] = defaultdict(list)
        self.claims: List[EnhancedScientificClaim] = []
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)
        self.chunk_index: Dict[str, List[Document]] = defaultdict(list)
        self.concept_metadata: Dict[str, Dict] = {}
        self.llm_ranked_concepts: List[Dict[str, Any]] = []
        self.dgl_graph = None
        self.dgl_node_maps: Dict[str, Dict[str, int]] = {}
        self.entity_embeddings: Optional[np.ndarray] = None
        self._entity_list: List[str] = []

    def add_document(self, doc_id: str, chunks: List[Document], bib_meta: Any,
                     concept_metadata: Optional[Dict[str, Dict]] = None,
                     llm_ranked: Optional[List[Dict[str, Any]]] = None):
        self.documents[doc_id] = {
            "bib_meta": bib_meta.to_dict() if hasattr(bib_meta, 'to_dict') else {},
            "chunk_count": len(chunks),
            "topics": set(),
            "years": getattr(bib_meta, 'year', None)
        }
        self.chunk_index[doc_id] = chunks
        for i, chunk in enumerate(chunks):
            entities = self._extract_entities_from_chunk(chunk, i)
            for ent in entities:
                self.entities[ent.normalized].append(ent)
                self.entity_index[ent.normalized].add(doc_id)
                self.documents[doc_id]["topics"].add(ent.label)
            claims = self._extract_claims_from_chunk(chunk, i)
            for claim in claims:
                self.claims.append(claim)
        if concept_metadata:
            for concept, meta in concept_metadata.items():
                if concept not in self.concept_metadata:
                    self.concept_metadata[concept] = meta
                else:
                    if meta.get("salience", 0) > self.concept_metadata[concept].get("salience", 0):
                        self.concept_metadata[concept] = meta
        if llm_ranked:
            self.llm_ranked_concepts.extend(llm_ranked)

    def _extract_entities_from_chunk_fast(self, chunk: Document, chunk_id: int) -> List[EnhancedScientificEntity]:
        """Optimized entity extraction with compiled patterns and batched matching."""
        text = chunk.page_content
        doc = chunk.metadata.get("source", "unknown")
        entities = []
        text_lower = text.lower()
        for param_name, pattern in QUANTITY_PATTERNS.items():
            for match in pattern.finditer(text):
                val_str = match.group(1)
                try:
                    val = float(val_str)
                except Exception:
                    val = None
                unit_match = re.search(r'(nm|µm|um|fs|ps|ns|J/cm²|J/cm2|kHz|MHz|W|mW|mJ|µJ|uJ)', match.group(0), re.I)
                unit = unit_match.group(1) if unit_match else None
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end].replace('\n', ' ')
                entities.append(EnhancedScientificEntity(
                    text=match.group(0), label=param_name, value=val, unit=unit,
                    doc_source=doc, chunk_id=chunk_id, context=context, confidence=0.85
                ))
        combined_aliases = {**MATERIAL_ALIASES, **METHOD_ALIASES}
        for canonical, aliases in combined_aliases.items():
            for alias in aliases:
                alias_lower = alias.lower()
                pos = text_lower.find(alias_lower)
                while pos != -1:
                    before = pos == 0 or not text_lower[pos-1].isalnum()
                    after = pos + len(alias_lower) >= len(text_lower) or not text_lower[pos + len(alias_lower)].isalnum()
                    if before and after:
                        start = max(0, pos - 80)
                        end = min(len(text), pos + len(alias_lower) + 80)
                        context = text[start:end]
                        lbl = "MATERIAL" if canonical in MATERIAL_ALIASES else "METHOD"
                        entities.append(EnhancedScientificEntity(
                            text=alias, label=lbl, value=None, unit=None,
                            doc_source=doc, chunk_id=chunk_id, context=context, confidence=0.9
                        ))
                    pos = text_lower.find(alias_lower, pos + 1)
        for topic, keywords in LASER_KEYWORDS.items():
            for kw in keywords:
                kw_lower = kw.lower()
                pos = text_lower.find(kw_lower)
                while pos != -1:
                    before = pos == 0 or not text_lower[pos-1].isalnum()
                    after = pos + len(kw_lower) >= len(text_lower) or not text_lower[pos + len(kw_lower)].isalnum()
                    if before and after:
                        start = max(0, pos - 80)
                        end = min(len(text), pos + len(kw_lower) + 80)
                        entities.append(EnhancedScientificEntity(
                            text=kw, label="TOPIC", value=None, unit=None,
                            doc_source=doc, chunk_id=chunk_id, context=text[start:end], confidence=0.8
                        ))
                    pos = text_lower.find(kw_lower, pos + 1)
        return entities

    def _extract_claims_from_chunk_fast(self, chunk: Document, chunk_id: int) -> List[EnhancedScientificClaim]:
        """Optimized claim extraction with compiled patterns."""
        text = chunk.page_content
        doc = chunk.metadata.get("source", "unknown")
        claims = []
        claim_patterns = [
            (re.compile(r'(?:ablation\s*threshold|threshold\s*fluence)\s*(?:of|for)\s+([a-z\s\-]+?)\s+(?:is|was|were|are|≈|~|about)\s+(\d+\.?\d*\s*[A-Za-z/²²]+)', re.I), 'has_ablation_threshold'),
            (re.compile(r'([a-z\s\-]+?)\s+(?:exhibits|shows|displays|forms|produces)\s+([a-z\s\-]+?(?:ripples|LIPSS|structures|morphology))', re.I), 'exhibits_morphology'),
            (re.compile(r'(?:periodicity|period|spacing)\s*(?:of|for)\s+([a-z\s\-]+?)\s+(?:is|was|≈|~)\s+(\d+\.?\d*\s*(?:nm|µm|um))', re.I), 'has_periodicity'),
            (re.compile(r'(?:roughness|Ra)\s*(?:of|for)\s+([a-z\s\-]+?)\s+(?:is|was|≈|~)\s+(\d+\.?\d*\s*(?:nm|µm|um))', re.I), 'has_roughness'),
            (re.compile(r'([a-z\s\-]+?)\s+(?:increases|decreases|reduces|enhances|promotes|suppresses)\s+([a-z\s\-]+?(?:growth|formation|porosity|cracking|stress))', re.I), 'causes_effect'),
        ]
        for pattern, predicate in claim_patterns:
            for match in pattern.finditer(text):
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

    def add_document_fast(self, doc_id: str, chunks: List[Document], bib_meta: Any,
                          concept_metadata: Optional[Dict[str, Dict]] = None,
                          llm_ranked: Optional[List[Dict[str, Any]]] = None):
        """Optimized document addition with fast entity/claim extraction."""
        self.documents[doc_id] = {
            "bib_meta": bib_meta.to_dict() if hasattr(bib_meta, 'to_dict') else {},
            "chunk_count": len(chunks),
            "topics": set(),
            "years": getattr(bib_meta, 'year', None)
        }
        self.chunk_index[doc_id] = chunks
        for i, chunk in enumerate(chunks):
            entities = self._extract_entities_from_chunk_fast(chunk, i)
            for ent in entities:
                self.entities[ent.normalized].append(ent)
                self.entity_index[ent.normalized].add(doc_id)
                self.documents[doc_id]["topics"].add(ent.label)
            claims = self._extract_claims_from_chunk_fast(chunk, i)
            for claim in claims:
                self.claims.append(claim)
        if concept_metadata:
            for concept, meta in concept_metadata.items():
                if concept not in self.concept_metadata:
                    self.concept_metadata[concept] = meta
                else:
                    if meta.get("salience", 0) > self.concept_metadata[concept].get("salience", 0):
                        self.concept_metadata[concept] = meta
        if llm_ranked:
            self.llm_ranked_concepts.extend(llm_ranked)

    def _extract_entities_from_chunk(self, chunk: Document, chunk_id: int) -> List[EnhancedScientificEntity]:
        text = chunk.page_content
        doc = chunk.metadata.get("source", "unknown")
        entities = []
        for param_name, pattern in QUANTITY_PATTERNS.items():
            for match in pattern.finditer(text):
                val_str = match.group(1)
                try:
                    val = float(val_str)
                except Exception:
                    val = None
                unit_match = re.search(r'(nm|µm|um|fs|ps|ns|J/cm²|J/cm2|kHz|MHz|W|mW|mJ|µJ|uJ)', match.group(0), re.I)
                unit = unit_match.group(1) if unit_match else None
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end].replace('\n', ' ')
                entities.append(EnhancedScientificEntity(
                    text=match.group(0), label=param_name, value=val, unit=unit,
                    doc_source=doc, chunk_id=chunk_id, context=context, confidence=0.85
                ))
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
        for topic, keywords in LASER_KEYWORDS.items():
            for kw in keywords:
                for match in re.finditer(r'\b' + re.escape(kw.lower()) + r'\b', text_lower):
                    start = max(0, match.start() - 80)
                    end = min(len(text), match.end() + 80)
                    entities.append(EnhancedScientificEntity(
                        text=kw, label="TOPIC", value=None, unit=None,
                        doc_source=doc, chunk_id=chunk_id, context=text[start:end], confidence=0.8
                    ))
        return entities

    def _extract_claims_from_chunk(self, chunk: Document, chunk_id: int) -> List[EnhancedScientificClaim]:
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

    def get_llm_ranked_concepts(self, top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        ranked = self.llm_ranked_concepts
        if top_n:
            ranked = ranked[:top_n]
        return ranked

    def find_consensus(self, entity_normalized: str) -> Optional[Dict[str, Any]]:
        ents = self.entities.get(entity_normalized, [])
        if len(ents) < 2:
            return None
        by_doc = defaultdict(list)
        for e in ents:
            by_doc[e.doc_source].append(e)
        if len(by_doc) < 2:
            return None
        values = [e.value for e in ents if e.value is not None]
        if not values:
            return None
        return {
            "entity": entity_normalized,
            "domain": ents[0].domain, "category": ents[0].category, "subcategory": ents[0].subcategory,
            "doc_count": len(by_doc), "value_count": len(values),
            "mean": float(np.mean(values)), "std": float(np.std(values)),
            "min": float(np.min(values)), "max": float(np.max(values)),
            "median": float(np.median(values)), "unit": ents[0].unit,
            "sources": list(by_doc.keys()),
            "values_by_doc": {d: [e.value for e in ev if e.value is not None] for d, ev in by_doc.items()}
        }

    def find_contradictions(self, entity_normalized: str, threshold_factor: float = 2.0) -> List[Dict[str, Any]]:
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
                            "doc_a": docs[i], "mean_a": float(mean_i), "std_a": float(np.std(vals_i)),
                            "doc_b": docs[j], "mean_b": float(mean_j), "std_b": float(np.std(vals_j)),
                            "ratio": float(ratio),
                            "severity": "critical" if ratio > 10 else "high" if ratio > 5 else "moderate"
                        })
        return contradictions

    def find_all_consensus(self, min_docs: int = 2) -> List[Dict[str, Any]]:
        results = []
        for ent_norm in self.entities:
            cons = self.find_consensus(ent_norm)
            if cons and cons["doc_count"] >= min_docs:
                results.append(cons)
        return sorted(results, key=lambda x: x["doc_count"], reverse=True)

    def find_all_contradictions(self, threshold_factor: float = 2.0) -> List[Dict[str, Any]]:
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

    def get_related_chunks(self, query_entities: List[str], chunks: List[Document],
                           depth: int = 2) -> List[Tuple[Document, float, str]]:
        related_docs = set()
        for ent_norm in query_entities:
            related_docs.update(self.entity_index.get(ent_norm, set()))
        scored = []
        for chunk in chunks:
            doc = chunk.metadata.get("source", "unknown")
            score = 0.0
            reason = "semantic"
            chunk_text = chunk.page_content.lower()
            for ent_norm in query_entities:
                if ent_norm in chunk_text:
                    score += 0.3
                if doc in related_docs:
                    score += 0.2
                    reason = "cross-doc-link"
            for claim in self.claims:
                if claim.doc_source == doc and claim.chunk_id == chunk.metadata.get("chunk_index", -1):
                    if any(ent in claim.subject.lower() or ent in claim.object_val.lower()
                           for ent in query_entities):
                        score += 0.25
                        reason = "claim-evidence"
            if score > 0:
                scored.append((chunk, score, reason))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def get_entity_cooccurrence_matrix(self, top_n: int = 20) -> Tuple[List[str], np.ndarray]:
        ent_counts = Counter({k: len(v) for k, v in self.entities.items()})
        top_entities = [e for e, _ in ent_counts.most_common(top_n)]
        n = len(top_entities)
        mat = np.zeros((n, n))
        for doc in self.documents:
            present = set()
            for ent in top_entities:
                if any(e.doc_source == doc for e in self.entities.get(ent, [])):
                    present.add(ent)
            for i, e1 in enumerate(top_entities):
                for j, e2 in enumerate(top_entities):
                    if i != j and e1 in present and e2 in present:
                        mat[i][j] += 1
        return top_entities, mat

    def build_dgl_heterograph(self, embedding_fn: Optional[Callable] = None):
        if not DGL_AVAILABLE:
            return None
        docs = list(self.documents.keys())
        chunks = []
        for doc_id, chks in self.chunk_index.items():
            for c in chks:
                chunks.append((doc_id, c))
        entities = list(self.entities.keys())
        claims = [(c.doc_source, c.chunk_id, i) for i, c in enumerate(self.claims)]
        topics = list(LASER_KEYWORDS.keys())
        doc_map = {d: i for i, d in enumerate(docs)}
        chunk_map = {(d, c.metadata.get("chunk_index", i)): i for i, (d, c) in enumerate(chunks)}
        ent_map = {e: i for i, e in enumerate(entities)}
        claim_map = {i: i for i in range(len(claims))}
        topic_map = {t: i for i, t in enumerate(topics)}
        self.dgl_node_maps = {
            "doc": doc_map, "chunk": chunk_map, "entity": ent_map,
            "claim": claim_map, "topic": topic_map
        }
        def edges(pairs):
            if not pairs:
                return None
            src, dst = zip(*pairs)
            return (torch.tensor(src, dtype=torch.int64), torch.tensor(dst, dtype=torch.int64))
        dc_pairs = []
        for (d, c), idx in chunk_map.items():
            dc_pairs.append((doc_map[d], idx))
        ce_pairs = []
        for ent_norm, ent_list in self.entities.items():
            for e in ent_list:
                key = (e.doc_source, e.chunk_id)
                if key in chunk_map:
                    ce_pairs.append((chunk_map[key], ent_map[ent_norm]))
        cc_pairs = []
        for ci, (doc, chunk_id, _) in enumerate(claims):
            key = (doc, chunk_id)
            if key in chunk_map:
                cc_pairs.append((chunk_map[key], ci))
        ee_pairs = []
        for doc_id, chks in self.chunk_index.items():
            for c in chks:
                cidx = c.metadata.get("chunk_index", -1)
                key = (doc_id, cidx)
                if key not in chunk_map:
                    continue
                present_ents = []
                for ent_norm, ent_list in self.entities.items():
                    if any(e.doc_source == doc_id and e.chunk_id == cidx for e in ent_list):
                        present_ents.append(ent_map[ent_norm])
                for i in range(len(present_ents)):
                    for j in range(i + 1, len(present_ents)):
                        ee_pairs.append((present_ents[i], present_ents[j]))
                        ee_pairs.append((present_ents[j], present_ents[i]))
        et_pairs = []
        for ent_norm, ent_list in self.entities.items():
            for topic in topics:
                if any(kw in ent_norm for kw in LASER_KEYWORDS[topic]):
                    et_pairs.append((ent_map[ent_norm], topic_map[topic]))
        cle_pairs = []
        for ci, claim in enumerate(self.claims):
            for ent_norm in entities:
                if ent_norm in claim.subject.lower() or ent_norm in claim.object_val.lower():
                    cle_pairs.append((ci, ent_map[ent_norm]))
        data_dict = {}
        if dc_pairs:
            data_dict[('doc', 'contains', 'chunk')] = edges(dc_pairs)
        if ce_pairs:
            data_dict[('chunk', 'mentions', 'entity')] = edges(ce_pairs)
        if cc_pairs:
            data_dict[('chunk', 'has_claim', 'claim')] = edges(cc_pairs)
        if ee_pairs:
            data_dict[('entity', 'cooccurs', 'entity')] = edges(ee_pairs)
        if et_pairs:
            data_dict[('entity', 'belongs_to', 'topic')] = edges(et_pairs)
        if cle_pairs:
            data_dict[('claim', 'about', 'entity')] = edges(cle_pairs)
        if not data_dict:
            self.dgl_graph = None
            return None
        g = dgl.heterograph(data_dict)
        emb_dim = 384
        if embedding_fn:
            doc_feats = []
            for d in docs:
                chks = self.chunk_index.get(d, [])
                if chks:
                    embs = [embedding_fn(c.page_content) for c in chks]
                    doc_feats.append(np.mean(embs, axis=0))
                else:
                    doc_feats.append(np.zeros(emb_dim))
            g.nodes['doc'].data['feat'] = torch.tensor(np.stack(doc_feats), dtype=torch.float32)
            chunk_feats = [embedding_fn(c.page_content) for (_, c) in chunks]
            g.nodes['chunk'].data['feat'] = torch.tensor(np.stack(chunk_feats), dtype=torch.float32)
            ent_feats = []
            for ent_norm in entities:
                ctxs = [e.context for e in self.entities[ent_norm]]
                if ctxs:
                    embs = [embedding_fn(c) for c in ctxs]
                    ent_feats.append(np.mean(embs, axis=0))
                else:
                    ent_feats.append(np.zeros(emb_dim))
            g.nodes['entity'].data['feat'] = torch.tensor(np.stack(ent_feats), dtype=torch.float32)
        else:
            for ntype in g.ntypes:
                g.nodes[ntype].data['feat'] = torch.randn(g.num_nodes(ntype), emb_dim) * 0.01
        if 'claim' in g.ntypes:
            g.nodes['claim'].data['feat'] = torch.randn(g.num_nodes('claim'), emb_dim) * 0.01
        if 'topic' in g.ntypes:
            g.nodes['topic'].data['feat'] = torch.eye(g.num_nodes('topic'))
        self.dgl_graph = g
        return g

    def get_knowledge_summary(self) -> Dict[str, Any]:
        return {
            "total_entities": sum(len(v) for v in self.entities.values()),
            "unique_entities": len(self.entities),
            "total_claims": len(self.claims),
            "document_count": len(self.documents),
            "top_entities": Counter([e.normalized for ents in self.entities.values() for e in ents]).most_common(15),
            "high_salience_concepts": sorted(
                self.concept_metadata.items(),
                key=lambda x: x[1].get("salience", 0),
                reverse=True
            )[:10],
            "llm_ranked_count": len(self.llm_ranked_concepts),
            "consensus_topics": [k for k, v in self.entities.items() if len(self.entity_index.get(k, set())) > 1],
            "domains": Counter([e.domain for ents in self.entities.values() for e in ents]).most_common(),
            "categories": Counter([e.category for ents in self.entities.values() for e in ents]).most_common(),
        }

# =====================================================================
# QUANTITATIVE DATA EXTRACTOR
# =====================================================================
class QuantitativeDataExtractor:
    """Extracts structured quantitative tables from the knowledge graph."""
    def __init__(self, graph: EnhancedCrossDocumentKnowledgeGraph):
        self.graph = graph

    def extract(self, quantity_label: str, group_by: str = "material") -> pd.DataFrame:
        records: List[Dict[str, Any]] = []
        targets: List[EnhancedScientificEntity] = []
        for norm, ent_list in self.graph.entities.items():
            for ent in ent_list:
                if ent.label == quantity_label and ent.value is not None:
                    targets.append(ent)
        for ent in targets:
            material = self._infer_associated(ent, "MATERIAL")
            method   = self._infer_associated(ent, "METHOD")
            doc_stem = Path(ent.doc_source).stem
            records.append({
                "value": float(ent.value),
                "unit": ent.unit or "a.u.",
                "raw_text": ent.text,
                "doc_source": ent.doc_source,
                "doc_stem": doc_stem,
                "material": material or "Unknown",
                "method": method or "Unknown",
                "context": ent.context[:250],
                "chunk_id": ent.chunk_id,
                "confidence": ent.confidence,
            })
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values(["material", "value"])
        return df

    def summarize(self, quantity_label: str) -> Dict[str, Any]:
        df = self.extract(quantity_label)
        if df.empty:
            return {"found": False, "quantity": quantity_label}
        return {
            "found": True,
            "quantity": quantity_label,
            "count": len(df),
            "unit": df["unit"].mode()[0] if not df["unit"].empty else "N/A",
            "value_range": (float(df["value"].min()), float(df["value"].max())),
            "mean": float(df["value"].mean()),
            "std": float(df["value"].std()),
            "by_material": df.groupby("material")["value"].agg(["count", "mean", "std", "min", "max"]).to_dict(),
            "by_document": df.groupby("doc_stem")["value"].agg(["count", "mean", "std"]).to_dict(),
        }

    def _infer_associated(self, target: EnhancedScientificEntity, target_domain: str) -> Optional[str]:
        """Find the most likely associated entity (e.g. MATERIAL) from the same chunk/doc."""
        chunk_hits = [
            e for elist in self.graph.entities.values()
            for e in elist
            if e.doc_source == target.doc_source and e.chunk_id == target.chunk_id and e.domain == target_domain
        ]
        if chunk_hits:
            return chunk_hits[0].normalized
        doc_hits = [
            e for elist in self.graph.entities.values()
            for e in elist
            if e.doc_source == target.doc_source and e.domain == target_domain
        ]
        if doc_hits:
            c = Counter([e.normalized for e in doc_hits])
            return c.most_common(1)[0][0]
        return None

# =====================================================================
# GRAPH DIFFUSION RETRIEVER
# =====================================================================
class GraphDiffusionRetriever:
    def __init__(self, graph: EnhancedCrossDocumentKnowledgeGraph, embedding_fn: Optional[Callable] = None):
        self.graph = graph
        self.embedding_fn = embedding_fn
        self.nx_graph: Optional[nx.Graph] = None
        self._build_nx_fallback()

    def _build_nx_fallback(self):
        G = nx.Graph()
        for doc_id in self.graph.documents:
            G.add_node(doc_id, node_type="doc", bipartite=0)
        for ent_norm, ents in self.graph.entities.items():
            G.add_node(ent_norm, node_type="entity", bipartite=1,
                       domain=ents[0].domain if ents else "UNKNOWN")
            for e in ents:
                G.add_edge(e.doc_source, ent_norm, weight=e.confidence)
        self.nx_graph = G

    def retrieve(self, query: str, query_entities: List[str], chunks: List[Document],
                 vector_scores: Dict[int, float], top_k: int = 6,
                 alpha: float = 0.5) -> List[Tuple[Document, float, str]]:
        if not query_entities:
            sorted_chunks = sorted(chunks, key=lambda c: vector_scores.get(c.metadata.get("chunk_index", -1), 0), reverse=True)
            return [(c, vector_scores.get(c.metadata.get("chunk_index", -1), 0), "vector-only") for c in sorted_chunks[:top_k]]
        if DGL_AVAILABLE and self.graph.dgl_graph is not None:
            diffusion_scores = self._dgl_diffusion(query_entities, chunks)
        else:
            diffusion_scores = self._nx_diffusion(query_entities, chunks)
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

    def _nx_diffusion(self, query_entities: List[str], chunks: List[Document]) -> Dict[int, float]:
        if self.nx_graph is None:
            return {}
        personalization = {n: 0.0 for n in self.nx_graph.nodes()}
        for ent in query_entities:
            if ent in personalization:
                personalization[ent] = 1.0
        if sum(personalization.values()) == 0:
            return {}
        try:
            pr = nx.pagerank(self.nx_graph, personalization=personalization, weight='weight')
        except Exception:
            pr = {}
        chunk_scores = {}
        for chunk in chunks:
            cidx = chunk.metadata.get("chunk_index", -1)
            doc = chunk.metadata.get("source", "unknown")
            score = pr.get(doc, 0.0) * 0.3
            for ent in query_entities:
                score += pr.get(ent, 0.0) * 0.7
            chunk_scores[cidx] = score
        return chunk_scores

    def _dgl_diffusion(self, query_entities: List[str], chunks: List[Document]) -> Dict[int, float]:
        return self._nx_diffusion(query_entities, chunks)

# =====================================================================
# CROSS-DOCUMENT THINKER
# =====================================================================
class CrossDocumentThinker:
    def __init__(self, graph: EnhancedCrossDocumentKnowledgeGraph,
                 vectorstore: Any,
                 embedding_fn: Callable,
                 llm_generate_fn: Callable):
        self.graph = graph
        self.vectorstore = vectorstore
        self.embedding_fn = embedding_fn
        self.llm_generate_fn = llm_generate_fn
        self.retriever = GraphDiffusionRetriever(graph, embedding_fn)

    def think_and_answer(self, query: str, k: int = 6) -> Tuple[str, ReasoningChain, List[Document], Dict[str, Any]]:
        chain = ReasoningChain(query)
        query_entities = self._extract_query_entities(query)
        chain.add_step("entity_extraction", f"Extracted {len(query_entities)} entities from query", {
            "entities": query_entities
        })
        semantic_docs = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k * 3, "score_threshold": 0.2}
        ).invoke(query)
        vector_scores = {}
        query_emb = self.embedding_fn(query)
        for doc in semantic_docs:
            cidx = doc.metadata.get("chunk_index", -1)
            doc_emb = self.embedding_fn(doc.page_content[:500])
            sim = float(np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-8))
            vector_scores[cidx] = sim
        chain.add_step("vector_retrieval", f"Retrieved {len(semantic_docs)} chunks via vector similarity", {
            "chunks": [d.metadata.get("source", "unknown") for d in semantic_docs[:5]]
        })
        all_chunks = []
        for doc_id in self.graph.chunk_index:
            all_chunks.extend(self.graph.chunk_index[doc_id])
        hybrid_results = self.retriever.retrieve(
            query, query_entities, all_chunks, vector_scores, top_k=k, alpha=0.6
        )
        retrieved_docs = [r[0] for r in hybrid_results]
        chain.add_step("graph_diffusion", f"Re-ranked via graph diffusion, top {len(retrieved_docs)} chunks", {
            "chunks": [d.metadata.get("source", "unknown") for d in retrieved_docs],
            "reasons": [r[2] for r in hybrid_results]
        })
        relevant_claims = []
        for claim in self.graph.claims:
            if any(ent in claim.subject.lower() or ent in claim.object_val.lower() for ent in query_entities):
                relevant_claims.append(claim)
        chain.add_step("claim_analysis", f"Found {len(relevant_claims)} relevant claims", {
            "claims": [c.predicate for c in relevant_claims[:5]]
        })
        consensus_data = []
        contradictions = []
        for ent in query_entities:
            cons = self.graph.find_consensus(ent)
            if cons:
                consensus_data.append(cons)
            contr = self.graph.find_contradictions(ent, threshold_factor=1.5)
            contradictions.extend(contr)
        chain.add_step("cross_doc_analysis",
                       f"Consensus: {len(consensus_data)}, Contradictions: {len(contradictions)}", {
            "consensus_entities": [c["entity"] for c in consensus_data],
            "contradiction_pairs": [(c["doc_a"], c["doc_b"], c["entity"]) for c in contradictions[:3]]
        })
        prompt = self._build_reasoning_prompt(retrieved_docs, query, consensus_data, contradictions, relevant_claims)
        answer = self.llm_generate_fn(prompt)
        chain.add_step("synthesis", "Generated answer via LLM synthesis", {
            "prompt_length": len(prompt),
            "answer_length": len(answer)
        })
        meta = {
            "query_entities": query_entities,
            "consensus_found": len(consensus_data),
            "contradictions_found": len(contradictions),
            "claim_count": len(relevant_claims),
            "retrieval_method": "hybrid_vector_graph",
            "reasoning_chain": chain.to_markdown()
        }
        return answer, chain, retrieved_docs, meta

    def _extract_query_entities(self, query: str) -> List[str]:
        entities = []
        q = query.lower()
        for canonical, aliases in {**MATERIAL_ALIASES, **METHOD_ALIASES}.items():
            if any(alias in q for alias in aliases):
                entities.append(canonical)
        for param_name in QUANTITY_PATTERNS.keys():
            if param_name.replace("_", " ") in q or param_name in q:
                entities.append(param_name)
        for topic, keywords in LASER_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                entities.append(topic)
        return list(set(entities))

    def _build_reasoning_prompt(self, retrieved_docs, query, consensus_data, contradictions, claims) -> str:
        context_parts = []
        for i, chunk in enumerate(retrieved_docs, 1):
            citation = chunk.metadata.get("citation_display")
            if not citation:
                source = chunk.metadata.get("source", "unknown")
                citation = f"[Source {i} - {source}]"
            section = chunk.metadata.get("section", "UNKNOWN")
            content = chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content
            context_parts.append(f"---\n[{i}] {citation} | Section: {section}\n{content}\n")
        context = "\n".join(context_parts)
        consensus_text = ""
        if consensus_data:
            consensus_text = "\nCross-Document Consensus:\n"
            for cons in consensus_data[:3]:
                consensus_text += (f"- {cons['entity']} ({cons['domain']}): {cons['mean']:.2f} ± {cons['std']:.2f} "
                                   f"{cons['unit']} across {cons['doc_count']} papers (n={cons['value_count']})\n")
        contradiction_text = ""
        if contradictions:
            contradiction_text = "\nDetected Contradictions:\n"
            for contr in contradictions[:3]:
                contradiction_text += (f"- {contr['entity']}: {Path(contr['doc_a']).stem}={contr['mean_a']:.2f} vs "
                                       f"{Path(contr['doc_b']).stem}={contr['mean_b']:.2f} "
                                       f"(ratio {contr['ratio']:.1f}x, {contr['severity']})\n")
        claim_text = ""
        if claims:
            claim_text = "\nRelevant Claims from Literature:\n"
            for c in claims[:5]:
                claim_text += f"- [{c.doc_source}] {c.subject} → {c.predicate} → {c.object_val}\n"
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
        user = f"{context}\n{consensus_text}\n{contradiction_text}\n{claim_text}\nQuestion: {query}\nProvide a rigorous scientific answer following the structure above."
        return system + "\n" + user

# =====================================================================
# DYNAMIC CONCEPT SELECTOR & VISUALIZATION MANAGER
# =====================================================================
class DynamicConceptSelector:
    """Manages user-defined top-N filtering, domain filtering, and visualization state."""
    def __init__(self, graph: EnhancedCrossDocumentKnowledgeGraph):
        self.graph = graph
        self.selected_concepts: Set[str] = set()
        self.active_domains: Set[str] = {"MATERIAL", "METHOD", "PHENOMENON", "PARAMETER", "TOPIC", "UNKNOWN"}
        self.top_n: int = 25
        self.use_llm_ranking: bool = False
        self.filter_by_salience_threshold: float = 0.0
        self._available_concepts: List[str] = []
        self._concept_to_domain: Dict[str, str] = {}
        self.refresh()

    def refresh(self):
        self._available_concepts = list(self.graph.entities.keys())
        for ent in self._available_concepts:
            if self.graph.entities.get(ent):
                self._concept_to_domain[ent] = self.graph.entities[ent][0].domain
            else:
                self._concept_to_domain[ent] = "UNKNOWN"
        self.selected_concepts.clear()

    def get_filtered_concepts(self) -> List[str]:
        concepts = self._available_concepts
        if self.active_domains:
            concepts = [c for c in concepts if self._concept_to_domain.get(c, "UNKNOWN") in self.active_domains]
        if self.filter_by_salience_threshold > 0:
            concepts = [c for c in concepts if self.graph.concept_metadata.get(c, {}).get("salience", 0) >= self.filter_by_salience_threshold]
        if self.use_llm_ranking and self.graph.llm_ranked_concepts:
            llm_names = {r["normalized"] for r in self.graph.llm_ranked_concepts}
            concepts = [c for c in concepts if c.lower() in llm_names]
            concepts.sort(key=lambda c: next((r["importance"] for r in self.graph.llm_ranked_concepts if r["normalized"].lower() == c.lower()), 0), reverse=True)
        else:
            concepts.sort(key=lambda c: self.graph.concept_metadata.get(c, {}).get("salience", 0), reverse=True)
        if self.top_n > 0:
            concepts = concepts[:self.top_n]
        self.selected_concepts.update(concepts)
        return concepts

    def apply_user_selection(self, top_n: int, domains: List[str], use_llm: bool, salience_thresh: float):
        self.top_n = max(1, min(top_n, 100))
        self.active_domains = set(domains) if domains else self.active_domains
        self.use_llm_ranking = use_llm
        self.filter_by_salience_threshold = salience_thresh
        self.refresh()

    def get_selection_metadata(self) -> Dict[str, Any]:
        return {
            "total_available": len(self._available_concepts),
            "filtered_count": len(self.get_filtered_concepts()),
            "active_domains": list(self.active_domains),
            "top_n": self.top_n,
            "llm_ranking_active": self.use_llm_ranking,
            "salience_threshold": self.filter_by_salience_threshold
        }

# =====================================================================
# PUBLICATION-QUALITY VISUALIZATION ENGINE
# =====================================================================
class PublicationQualityVisualizationEngine:
    """
    Publication-quality scientific visualization engine for DECLARMIMA.
    Features:
    - 50+ matplotlib colormaps
    - Customizable fonts (family, size, weight)
    - UMAP, t-SNE, PCA embeddings
    - Bokeh + HoloViews chord diagrams
    - PyVis interactive networks
    - Hierarchical sunbursts, radar charts, contradiction matrices
    - Dynamic top-N concept filtering
    """
    COLORMAP_OPTIONS = {
        "viridis": "viridis", "plasma": "plasma", "inferno": "inferno", "magma": "magma", "cividis": "cividis",
        "Greys": "Greys", "Purples": "Purples", "Blues": "Blues", "Greens": "Greens", "Oranges": "Oranges", "Reds": "Reds",
        "YlOrBr": "YlOrBr", "YlOrRd": "YlOrRd", "OrRd": "OrRd", "PuRd": "PuRd", "RdPu": "RdPu", "BuPu": "BuPu",
        "GnBu": "GnBu", "PuBu": "PuBu", "YlGnBu": "YlGnBu", "PuBuGn": "PuBuGn", "BuGn": "BuGn", "YlGn": "YlGn",
        "binary": "binary", "gist_yarg": "gist_yarg", "gist_gray": "gist_gray", "gray": "gray", "bone": "bone",
        "pink": "pink", "spring": "spring", "summer": "summer", "autumn": "autumn", "winter": "winter",
        "cool": "cool", "Wistia": "Wistia", "hot": "hot", "afmhot": "afmhot", "gist_heat": "gist_heat", "copper": "copper",
        "PiYG": "PiYG", "PRGn": "PRGn", "BrBG": "BrBG", "PuOr": "PuOr", "RdGy": "RdGy", "RdBu": "RdBu",
        "RdYlBu": "RdYlBu", "RdYlGn": "RdYlGn", "Spectral": "Spectral", "coolwarm": "coolwarm", "bwr": "bwr", "seismic": "seismic",
        "tab10": "tab10", "tab20": "tab20", "tab20b": "tab20b", "tab20c": "tab20c",
        "Pastel1": "Pastel1", "Pastel2": "Pastel2", "Paired": "Paired", "Accent": "Accent", "Dark2": "Dark2", "Set1": "Set1", "Set2": "Set2", "Set3": "Set3",
        "flag": "flag", "prism": "prism", "ocean": "ocean", "gist_earth": "gist_earth", "terrain": "terrain",
        "gist_stern": "gist_stern", "gnuplot": "gnuplot", "gnuplot2": "gnuplot2", "CMRmap": "CMRmap",
        "cubehelix": "cubehelix", "brg": "brg", "hsv": "hsv", "gist_rainbow": "gist_rainbow", "rainbow": "rainbow", "jet": "jet",
        "turbo": "turbo", "nipy_spectral": "nipy_spectral", "gist_ncar": "gist_ncar",
    }
    DOMAIN_COLORS = {
        "MATERIAL": "#3b82f6", "METHOD": "#8b5cf6", "PHENOMENON": "#f59e0b",
        "PARAMETER": "#10b981", "UNKNOWN": "#6b7280", "TOPIC": "#ec4899"
    }

    def __init__(self, graph: EnhancedCrossDocumentKnowledgeGraph,
                 font_family: str = "DejaVu Sans",
                 font_size: int = 10,
                 title_font_size: int = 14,
                 label_font_size: int = 9,
                 default_colormap: str = "viridis",
                 figure_dpi: int = 300,
                 figure_format: str = "png"):
        self.graph = graph
        self.font_family = font_family
        self.font_size = font_size
        self.title_font_size = title_font_size
        self.label_font_size = label_font_size
        self.default_colormap = default_colormap
        self.figure_dpi = figure_dpi
        self.figure_format = figure_format
        plt.rcParams['font.family'] = font_family
        plt.rcParams['font.size'] = font_size
        plt.rcParams['axes.titlesize'] = title_font_size
        plt.rcParams['axes.labelsize'] = label_font_size
        plt.rcParams['figure.dpi'] = figure_dpi
        plt.rcParams['savefig.dpi'] = figure_dpi
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['axes.edgecolor'] = '#333333'
        plt.rcParams['axes.labelcolor'] = '#333333'
        plt.rcParams['xtick.color'] = '#333333'
        plt.rcParams['ytick.color'] = '#333333'
        plt.rcParams['text.color'] = '#333333'
        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['grid.linestyle'] = '--'

    def _get_colormap(self, name: Optional[str] = None) -> str:
        cmap = name or self.default_colormap
        return self.COLORMAP_OPTIONS.get(cmap, "viridis")

    def _get_domain_color(self, domain: str, colormap: Optional[str] = None, index: int = 0, total: int = 1) -> str:
        if colormap and total > 1:
            cmap = plt.get_cmap(self._get_colormap(colormap))
            return mcolors.to_hex(cmap(index / max(total - 1, 1)))
        return self.DOMAIN_COLORS.get(domain, "#6b7280")

    def get_salience(self, concept: str) -> float:
        return self.graph.concept_metadata.get(concept, {}).get("salience", 0.5)

    def is_core_pillar(self, concept: str) -> bool:
        return self.graph.concept_metadata.get(concept, {}).get("is_core_pillar", False)

    def plot_static_knowledge_network(self, filtered_concepts: Optional[List[str]] = None,
                                      top_n: int = 25, figsize: Tuple[int, int] = (14, 12),
                                      layout: str = "spring", colormap: Optional[str] = None,
                                      node_size_factor: float = 1.0, edge_alpha: float = 0.25,
                                      show_labels: bool = True, label_font_size: Optional[int] = None) -> plt.Figure:
        G = nx.Graph()
        if filtered_concepts:
            top_entities = filtered_concepts[:top_n]
        else:
            ent_counts = Counter({k: len(v) for k, v in self.graph.entities.items()})
            scored = [(ent, self.get_salience(ent) * ent_counts.get(ent, 1)) for ent in self.graph.entities.keys()]
            top_entities = [e for e, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]]
        for doc_id in self.graph.documents:
            G.add_node(Path(doc_id).stem, node_type="doc", bipartite=0)
        for ent in top_entities:
            ents = self.graph.entities.get(ent, [])
            if not ents:
                continue
            domain = ents[0].domain if ents else "UNKNOWN"
            salience = self.get_salience(ent)
            G.add_node(ent, node_type="entity", domain=domain, bipartite=1, salience=salience)
            for e in ents:
                doc_node = Path(e.doc_source).stem
                if doc_node in G:
                    G.add_edge(doc_node, ent, weight=e.confidence * (0.5 + 0.5 * salience))
        fig, ax = plt.subplots(figsize=figsize)
        if layout == "spring":
            pos = nx.spring_layout(G, k=0.55, iterations=60, seed=42)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, k=0.55, iterations=60, seed=42)
        doc_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "doc"]
        ent_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "entity"]
        nx.draw_networkx_nodes(G, pos, nodelist=doc_nodes, node_color="#1e40af",
                               node_shape="s", node_size=800, alpha=0.85, ax=ax, label="Documents")
        domains = list(set(G.nodes[n].get("domain", "UNKNOWN") for n in ent_nodes))
        cmap = plt.get_cmap(self._get_colormap(colormap))
        domain_color_idx = {d: i for i, d in enumerate(domains)}
        for node in ent_nodes:
            salience = G.nodes[node].get("salience", 0.5)
            domain = G.nodes[node].get("domain", "UNKNOWN")
            if colormap:
                color_idx = domain_color_idx.get(domain, 0)
                base_color = mcolors.to_hex(cmap(color_idx / max(len(domains) - 1, 1)))
            else:
                base_color = self.DOMAIN_COLORS.get(domain, "#6b7280")
            color = mcolors.to_hex(
                mcolors.to_rgba(base_color, alpha=0.7 + 0.3 * salience)
            )
            size = (300 + salience * 900) * node_size_factor
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=color,
                                   node_shape="o", node_size=size, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=edge_alpha, width=0.8, ax=ax)
        if show_labels:
            lbl_size = label_font_size or self.label_font_size
            nx.draw_networkx_labels(G, pos, font_size=lbl_size, ax=ax,
                                    font_family=self.font_family)
        legend_patches = [mpatches.Patch(color="#1e40af", label="Documents")]
        for dom in domains:
            if colormap:
                idx = domain_color_idx[dom]
                c = mcolors.to_hex(cmap(idx / max(len(domains) - 1, 1)))
            else:
                c = self.DOMAIN_COLORS.get(dom, "#6b7280")
            legend_patches.append(mpatches.Patch(color=c, label=dom))
        ax.legend(handles=legend_patches, loc="upper left", fontsize=9)
        ax.set_title("Salience-Aware Cross-Document Knowledge Network\n(Node size = importance)",
                     fontsize=self.title_font_size, fontweight='bold', fontfamily=self.font_family)
        ax.axis("off")
        plt.tight_layout()
        return fig

    def plot_chord_cooccurrence(self, filtered_concepts: Optional[List[str]] = None,
                                top_n: int = 14, colormap: Optional[str] = None) -> go.Figure:
        if filtered_concepts:
            top_entities = filtered_concepts[:top_n]
        else:
            scored = [(ent, self.get_salience(ent) * len(self.graph.entities.get(ent, []))) for ent in self.graph.entities]
            top_entities = [e for e, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]]
        if not top_entities:
            fig = go.Figure()
            fig.update_layout(title="No entity co-occurrence data")
            return fig
        n = len(top_entities)
        node_to_idx = {node: i for i, node in enumerate(top_entities)}
        adj = np.zeros((n, n))
        for doc in self.graph.documents:
            present = [ent for ent in top_entities if ent in self.graph.entity_index and doc in self.graph.entity_index[ent]]
            for i, e1 in enumerate(present):
                for j, e2 in enumerate(present):
                    if i != j:
                        adj[node_to_idx[e1]][node_to_idx[e2]] += 1
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        cmap = plt.get_cmap(self._get_colormap(colormap))
        fig = go.Figure()
        for i, ent in enumerate(top_entities):
            domain = self.graph.entities[ent][0].domain if self.graph.entities.get(ent) else "UNKNOWN"
            color_idx = list(self.DOMAIN_COLORS.keys()).index(domain) if domain in self.DOMAIN_COLORS else 0
            color = mcolors.to_hex(cmap(color_idx / max(len(self.DOMAIN_COLORS) - 1, 1)))
            fig.add_trace(go.Barpolar(
                r=[1], theta=[np.degrees(angles[i])],
                width=[10], marker_color=color,
                name=ent, opacity=0.9, showlegend=False,
                hoverinfo="text", text=[f"{ent}<br>Salience: {self.get_salience(ent):.2f}<br>Count: {len(self.graph.entities.get(ent, []))}"]
            ))
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j] > 0:
                    fig.add_trace(go.Scatterpolar(
                        r=[0.2, 0.6, 0.2],
                        theta=[np.degrees(angles[i]), np.degrees((angles[i] + angles[j]) / 2), np.degrees(angles[j])],
                        mode='lines', line=dict(color='rgba(100,100,100,0.3)', width=min(adj[i][j], 3)),
                        showlegend=False, hoverinfo='skip'
                    ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=False), angularaxis=dict(visible=False)),
            title=f"Salience-Aware Chord Diagram (Top {n} Concepts)",
            height=700, width=700,
            font=dict(family=self.font_family, size=self.font_size)
        )
        return fig

    def plot_bokeh_chord(self, filtered_concepts: Optional[List[str]] = None,
                         top_n: int = 20, colormap: str = "Category20",
                         width: int = 800, height: int = 800) -> Optional[Any]:
        if not BOKEH_AVAILABLE:
            return None
        if filtered_concepts:
            top_entities = filtered_concepts[:top_n]
        else:
            scored = [(ent, self.get_salience(ent) * len(self.graph.entities.get(ent, []))) for ent in self.graph.entities]
            top_entities = [e for e, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]]
        if len(top_entities) < 3:
            return None
        n = len(top_entities)
        node_to_idx = {node: i for i, node in enumerate(top_entities)}
        adj = np.zeros((n, n))
        for doc in self.graph.documents:
            present = [ent for ent in top_entities if ent in self.graph.entity_index and doc in self.graph.entity_index[ent]]
            for i, e1 in enumerate(present):
                for j, e2 in enumerate(present):
                    if i != j:
                        adj[node_to_idx[e1]][node_to_idx[e2]] += 1
        edge_list = []
        edge_weights = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j] > 0:
                    edge_list.append((i, j))
                    edge_weights.append(adj[i][j])
        if not edge_list:
            return None
        node_colors = []
        node_sizes = []
        for ent in top_entities:
            domain = self.graph.entities[ent][0].domain if self.graph.entities.get(ent) else "UNKNOWN"
            color = self.DOMAIN_COLORS.get(domain, "#6b7280")
            node_colors.append(color)
            node_sizes.append(15 + self.get_salience(ent) * 35)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for (i, j), w in zip(edge_list, edge_weights):
            G.add_edge(i, j, weight=w)
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        p = figure(title="Interactive Chord-Style Co-occurrence Network (Bokeh)",
                   width=width, height=height,
                   x_range=(-1.2, 1.2), y_range=(-1.2, 1.2),
                   tools="pan,wheel_zoom,box_zoom,reset,save",
                   active_scroll="wheel_zoom")
        edge_xs = []
        edge_ys = []
        edge_alphas = []
        max_w = max(edge_weights) if edge_weights else 1
        for (i, j), w in zip(edge_list, edge_weights):
            x0, y0 = pos[i]
            x1, y1 = pos[j]
            edge_xs.append([x0, x1])
            edge_ys.append([y0, y1])
            edge_alphas.append(0.2 + 0.6 * (w / max_w))
        p.multi_line(edge_xs, edge_ys, line_color="#888888", line_alpha=edge_alphas, line_width=1.5)
        node_x = [pos[i][0] for i in range(n)]
        node_y = [pos[i][1] for i in range(n)]
        source = ColumnDataSource(data=dict(
            x=node_x, y=node_y,
            color=node_colors,
            size=node_sizes,
            name=top_entities,
            salience=[self.get_salience(e) for e in top_entities],
            domain=[self.graph.entities[e][0].domain if self.graph.entities.get(e) else "UNKNOWN" for e in top_entities]
        ))
        p.circle('x', 'y', size='size', color='color', alpha=0.8, source=source,
                 hover_color='red', hover_alpha=1.0)
        labels = LabelSet(x='x', y='y', text='name', level='glyph',
                          x_offset=5, y_offset=5, source=source,
                          text_font_size=f"{self.label_font_size}pt",
                          text_font=self.font_family)
        p.add_layout(labels)
        hover = HoverTool(tooltips=[
            ("Entity", "@name"),
            ("Domain", "@domain"),
            ("Salience", "@salience{0.00}"),
        ])
        p.add_tools(hover)
        p.axis.visible = False
        p.grid.visible = False
        p.outline_line_color = None
        return p

    def plot_holoviews_chord(self, filtered_concepts: Optional[List[str]] = None, top_n: int = 20) -> Optional[Any]:
        if not HOLOVIEWS_AVAILABLE:
            return None
        if filtered_concepts:
            top_entities = filtered_concepts[:top_n]
        else:
            scored = [(ent, self.get_salience(ent) * len(self.graph.entities.get(ent, []))) for ent in self.graph.entities]
            top_entities = [e for e, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]]
        if len(top_entities) < 3:
            return None
        edges_df = []
        for doc in self.graph.documents:
            present = [ent for ent in top_entities if ent in self.graph.entity_index and doc in self.graph.entity_index[ent]]
            for i, e1 in enumerate(present):
                for j, e2 in enumerate(present):
                    if i < j:
                        edges_df.append({'source': e1, 'target': e2, 'weight': 1})
        if not edges_df:
            return None
        edges_df = pd.DataFrame(edges_df).groupby(['source', 'target']).sum().reset_index()
        chord = hv.Chord(edges_df).opts(
            opts.Chord(cmap='Category20', edge_cmap='Category20',
                       edge_color='source', node_color='index',
                       labels='index', node_size='salience',
                       width=800, height=800,
                       title='HoloViews Chord Diagram (Entity Co-occurrence)')
        )
        return chord

    def _build_sunburst_df(self, domain_filter: str, filtered_concepts: Optional[List[str]] = None) -> pd.DataFrame:
        rows = []
        target_ents = set(filtered_concepts) if filtered_concepts else None
        for norm, ents in self.graph.entities.items():
            if target_ents and norm not in target_ents:
                continue
            if not ents:
                continue
            e = ents[0]
            if e.domain != domain_filter:
                continue
            salience = self.get_salience(norm)
            rows.append({
                "domain": e.domain,
                "category": e.category,
                "subcategory": e.subcategory,
                "entity": norm,
                "value": len(ents) * (0.5 + 0.5 * salience),
                "doc_count": len(set(x.doc_source for x in ents)),
                "salience": salience
            })
        return pd.DataFrame(rows)

    def plot_methods_sunburst(self, filtered_concepts: Optional[List[str]] = None,
                              top_n_per_category: int = 20, colormap: Optional[str] = None) -> go.Figure:
        df = self._build_sunburst_df("METHOD", filtered_concepts)
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title="No METHOD entities found")
            return fig
        def keep_top_n(group):
            return group.nlargest(top_n_per_category, 'salience')
        df = df.groupby(['domain', 'category', 'subcategory'], group_keys=False).apply(keep_top_n)
        colorscale = colormap or "Blues"
        fig = px.sunburst(df, path=["domain", "category", "subcategory", "entity"],
                          values="value", color="salience", color_continuous_scale=colorscale,
                          title="Hierarchical Methods Taxonomy\nColored & Sized by Salience")
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
        return fig

    def plot_materials_sunburst(self, filtered_concepts: Optional[List[str]] = None,
                                top_n_per_category: int = 20, colormap: Optional[str] = None) -> go.Figure:
        df = self._build_sunburst_df("MATERIAL", filtered_concepts)
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title="No MATERIAL entities found")
            return fig
        def keep_top_n(group):
            return group.nlargest(top_n_per_category, 'salience')
        df = df.groupby(['domain', 'category', 'subcategory'], group_keys=False).apply(keep_top_n)
        colorscale = colormap or "Greens"
        fig = px.sunburst(df, path=["domain", "category", "subcategory", "entity"],
                          values="value", color="salience", color_continuous_scale=colorscale,
                          title="Material System Hierarchy\nColored & Sized by Salience")
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
        return fig

    def plot_topics_sunburst(self, colormap: Optional[str] = None) -> go.Figure:
        rows = []
        for topic, keywords in LASER_KEYWORDS.items():
            count = sum(1 for norm, ents in self.graph.entities.items()
                        if any(kw in norm for kw in keywords) or any(kw in e.text.lower() for e in ents for kw in keywords))
            if count > 0:
                rows.append({"topic": topic, "count": count})
        df = pd.DataFrame(rows)
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title="No topic entities found")
            return fig
        colorscale = colormap or "Oranges"
        fig = px.sunburst(df, path=["topic"], values="count",
                          color="count", color_continuous_scale=colorscale,
                          title="Study Topics Distribution")
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
        return fig

    def plot_document_radar(self, filtered_concepts: Optional[List[str]] = None, colormap: Optional[str] = None) -> go.Figure:
        categories = ["Laser Parameters", "Materials", "Exp. Methods", "Simulation", "Phenomena", "Properties"]
        cat_map = {
            "Laser Parameters": ["PARAMETER"],
            "Materials": ["MATERIAL"],
            "Exp. Methods": ["METHOD:Experimental"],
            "Simulation": ["METHOD:Computational"],
            "Phenomena": ["PHENOMENON"],
            "Properties": ["PARAMETER:Outcome"]
        }
        cmap = plt.get_cmap(self._get_colormap(colormap)) if colormap else None
        target_ents = set(filtered_concepts) if filtered_concepts else None
        fig = go.Figure()
        docs = list(self.graph.documents.keys())
        for idx, doc_id in enumerate(docs):
            values = []
            for cat in categories:
                count = 0
                target_domains = cat_map[cat]
                for norm, ents in self.graph.entities.items():
                    if target_ents and norm not in target_ents:
                        continue
                    if any(e.doc_source == doc_id for e in ents):
                        e = ents[0]
                        if e.domain in target_domains or f"{e.domain}:{e.category}" in target_domains:
                            count += len([x for x in ents if x.doc_source == doc_id]) * self.get_salience(norm)
                values.append(count)
            values += values[:1]
            color = mcolors.to_hex(cmap(idx / max(len(docs) - 1, 1))) if cmap else None
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=Path(doc_id).stem,
                line_color=color
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(5, max([max(t.r) for t in fig.data] or [5]))])),
            showlegend=True, title="Document Coverage Profiles (Radar)",
            font=dict(family=self.font_family, size=self.font_size)
        )
        return fig

    def plot_contradiction_matrix(self, colormap: Optional[str] = None) -> go.Figure:
        contrs = self.graph.find_all_contradictions(threshold_factor=1.5)
        if not contrs:
            fig = go.Figure()
            fig.update_layout(title="No contradictions detected")
            return fig
        docs = sorted(list(self.graph.documents.keys()))
        doc_stems = [Path(d).stem for d in docs]
        n = len(docs)
        mat = np.zeros((n, n))
        annotations = [["" for _ in range(n)] for _ in range(n)]
        for c in contrs:
            i, j = docs.index(c["doc_a"]), docs.index(c["doc_b"])
            severity_score = {"moderate": 1, "high": 2, "critical": 3}[c["severity"]]
            mat[i][j] = max(mat[i][j], severity_score)
            mat[j][i] = mat[i][j]
            annotations[i][j] += f"{c['entity'][:15]}({c['ratio']:.1f}x)<br>"
            annotations[j][i] = annotations[i][j]
        colorscale = colormap or [[0, "white"], [0.33, "#fcd34d"], [0.66, "#f97316"], [1, "#dc2626"]]
        fig = go.Figure(data=go.Heatmap(
            z=mat, x=doc_stems, y=doc_stems,
            colorscale=colorscale,
            text=annotations, texttemplate="%{text}", hoverinfo="text"
        ))
        fig.update_layout(title="Cross-Document Contradiction Severity Matrix",
                          height=600, width=600,
                          font=dict(family=self.font_family, size=self.font_size))
        return fig

    def plot_consensus_waterfall(self, top_n: int = 10, colormap: Optional[str] = None) -> go.Figure:
        consensus = self.graph.find_all_consensus(min_docs=2)[:top_n]
        if not consensus:
            fig = go.Figure()
            fig.update_layout(title="No consensus data available")
            return fig
        entities = [c["entity"] for c in consensus]
        means = [c["mean"] for c in consensus]
        stds = [c["std"] for c in consensus]
        doc_counts = [c["doc_count"] for c in consensus]
        order = np.argsort(doc_counts)[::-1]
        entities = [entities[i] for i in order]
        means = [means[i] for i in order]
        stds = [stds[i] for i in order]
        doc_counts = [doc_counts[i] for i in order]
        fig = go.Figure()
        if colormap:
            cmap = plt.get_cmap(self._get_colormap(colormap))
            bar_colors = [mcolors.to_hex(cmap(i / max(len(entities) - 1, 1))) for i in range(len(entities))]
        else:
            bar_colors = ["#059669" if d >= 3 else "#3b82f6" for d in doc_counts]
        fig.add_trace(go.Bar(
            x=entities, y=means,
            error_y=dict(type='data', array=stds, visible=True, color="black"),
            marker_color=bar_colors,
            text=[f"μ={m:.2f}<br>σ={s:.2f}<br>n={d} docs" for m, s, d in zip(means, stds, doc_counts)],
            textposition="outside"
        ))
        fig.update_layout(
            title="Cross-Document Consensus Waterfall\nGreen = strong consensus (≥3 docs), Blue = emerging",
            yaxis_title="Mean Value", xaxis_tickangle=-45, height=500,
            font=dict(family=self.font_family, size=self.font_size)
        )
        return fig

    def plot_reasoning_chain(self, chain: ReasoningChain, figsize: Tuple[int, int] = (12, 8),
                             colormap: Optional[str] = None) -> plt.Figure:
        G = chain.build_thinking_graph()
        fig, ax = plt.subplots(figsize=figsize)
        pos = nx.multipartite_layout(G, subset_key="layer")
        cmap = plt.get_cmap(self._get_colormap(colormap)) if colormap else None
        color_map = {
            "query": "#1e40af", "entity_extraction": "#3b82f6", "vector_retrieval": "#8b5cf6",
            "graph_diffusion": "#a855f7", "claim_analysis": "#f59e0b", "cross_doc_analysis": "#10b981",
            "synthesis": "#ec4899", "answer": "#059669", "entity": "#60a5fa", "chunk": "#c084fc"
        }
        if cmap:
            node_types = list(set(nx.get_node_attributes(G, "node_type").values()))
            type_to_color = {t: mcolors.to_hex(cmap(i / max(len(node_types) - 1, 1)))
                             for i, t in enumerate(node_types)}
            node_colors = [type_to_color.get(G.nodes[n].get("node_type", "query"), "#6b7280") for n in G.nodes()]
        else:
            node_colors = [color_map.get(G.nodes[n].get("node_type", "query"), "#6b7280") for n in G.nodes()]
        node_sizes = [1200 if G.nodes[n].get("node_type") in ["query", "answer"] else 600 for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=15, alpha=0.5, ax=ax,
                               connectionstyle="arc3,rad=0.1", edge_color="#4b5563")
        nx.draw_networkx_labels(G, pos, font_size=self.label_font_size, ax=ax,
                                font_family=self.font_family)
        ax.set_title("Explicit Reasoning Chain (Thinking Graph)",
                     fontsize=self.title_font_size, fontweight='bold',
                     fontfamily=self.font_family)
        ax.axis("off")
        plt.tight_layout()
        return fig

    def _get_entity_embeddings(self, embedding_fn: Callable, filtered_concepts: Optional[List[str]] = None,
                               top_n: int = 80) -> Tuple[List[str], np.ndarray, List[str]]:
        target = filtered_concepts or list(self.graph.entities.keys())
        scored = [(ent, self.get_salience(ent) * len(self.graph.entities.get(ent, []))) for ent in target]
        top = [e for e, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]]
        if len(top) < 5:
            return [], np.array([]), []
        embs = []
        domains = []
        for ent in top:
            vec = embedding_fn(ent)
            embs.append(vec)
            domains.append(self.graph.entities[ent][0].domain if self.graph.entities.get(ent) else "UNKNOWN")
        embs = np.stack(embs)
        return top, embs, domains

    def plot_entity_tsne(self, embedding_fn: Callable, filtered_concepts: Optional[List[str]] = None,
                         top_n: int = 80, perplexity: int = 30, colormap: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 8)) -> Optional[plt.Figure]:
        if not SKLEARN_AVAILABLE:
            return None
        top, embs, domains = self._get_entity_embeddings(embedding_fn, filtered_concepts, top_n)
        if len(top) < 5:
            return None
        tsne = TSNE(n_components=2, perplexity=min(perplexity, len(top)-1), random_state=42)
        coords = tsne.fit_transform(embs)
        fig, ax = plt.subplots(figsize=figsize)
        unique_domains = list(set(domains))
        cmap = plt.get_cmap(self._get_colormap(colormap))
        for i, domain in enumerate(unique_domains):
            mask = [d == domain for d in domains]
            x = coords[mask, 0]
            y = coords[mask, 1]
            color = mcolors.to_hex(cmap(i / max(len(unique_domains) - 1, 1)))
            ax.scatter(x, y, c=color, label=domain, alpha=0.8, s=80, edgecolors='white')
        for i, ent in enumerate(top):
            ax.annotate(ent[:20], (coords[i, 0], coords[i, 1]),
                        fontsize=self.label_font_size - 1, alpha=0.8,
                        fontfamily=self.font_family)
        ax.legend(loc='best', fontsize=self.label_font_size)
        ax.set_title("Entity Embedding Space (t-SNE)",
                     fontsize=self.title_font_size, fontweight='bold',
                     fontfamily=self.font_family)
        ax.axis("off")
        plt.tight_layout()
        return fig

    def plot_entity_umap(self, embedding_fn: Callable, filtered_concepts: Optional[List[str]] = None,
                         top_n: int = 80, n_neighbors: int = 15, min_dist: float = 0.1,
                         colormap: Optional[str] = None, figsize: Tuple[int, int] = (10, 8)) -> Optional[plt.Figure]:
        if not UMAP_AVAILABLE:
            return None
        top, embs, domains = self._get_entity_embeddings(embedding_fn, filtered_concepts, top_n)
        if len(top) < 5:
            return None
        reducer = umap.UMAP(n_neighbors=min(n_neighbors, len(top)-1), min_dist=min_dist, random_state=42)
        coords = reducer.fit_transform(embs)
        fig, ax = plt.subplots(figsize=figsize)
        unique_domains = list(set(domains))
        cmap = plt.get_cmap(self._get_colormap(colormap))
        for i, domain in enumerate(unique_domains):
            mask = [d == domain for d in domains]
            x = coords[mask, 0]
            y = coords[mask, 1]
            color = mcolors.to_hex(cmap(i / max(len(unique_domains) - 1, 1)))
            ax.scatter(x, y, c=color, label=domain, alpha=0.8, s=80, edgecolors='white')
        for i, ent in enumerate(top):
            ax.annotate(ent[:20], (coords[i, 0], coords[i, 1]),
                        fontsize=self.label_font_size - 1, alpha=0.8,
                        fontfamily=self.font_family)
        ax.legend(loc='best', fontsize=self.label_font_size)
        ax.set_title("Entity Embedding Space (UMAP)",
                     fontsize=self.title_font_size, fontweight='bold',
                     fontfamily=self.font_family)
        ax.axis("off")
        plt.tight_layout()
        return fig

    def plot_entity_pca(self, embedding_fn: Callable, filtered_concepts: Optional[List[str]] = None,
                        top_n: int = 80, colormap: Optional[str] = None,
                        figsize: Tuple[int, int] = (10, 8)) -> Optional[plt.Figure]:
        if not SKLEARN_AVAILABLE:
            return None
        top, embs, domains = self._get_entity_embeddings(embedding_fn, filtered_concepts, top_n)
        if len(top) < 5:
            return None
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embs)
        var_ratio = pca.explained_variance_ratio_
        fig, ax = plt.subplots(figsize=figsize)
        unique_domains = list(set(domains))
        cmap = plt.get_cmap(self._get_colormap(colormap))
        for i, domain in enumerate(unique_domains):
            mask = [d == domain for d in domains]
            x = coords[mask, 0]
            y = coords[mask, 1]
            color = mcolors.to_hex(cmap(i / max(len(unique_domains) - 1, 1)))
            ax.scatter(x, y, c=color, label=domain, alpha=0.8, s=80, edgecolors='white')
        for i, ent in enumerate(top):
            ax.annotate(ent[:20], (coords[i, 0], coords[i, 1]),
                        fontsize=self.label_font_size - 1, alpha=0.8,
                        fontfamily=self.font_family)
        ax.legend(loc='best', fontsize=self.label_font_size)
        ax.set_title(f"Entity Embedding Space (PCA)\nPC1: {var_ratio[0]:.1%}, PC2: {var_ratio[1]:.1%}",
                     fontsize=self.title_font_size, fontweight='bold',
                     fontfamily=self.font_family)
        ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%})", fontfamily=self.font_family)
        ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%})", fontfamily=self.font_family)
        plt.tight_layout()
        return fig

    def plot_temporal_timeline(self, colormap: Optional[str] = None) -> go.Figure:
        rows = []
        for doc_id, meta in self.graph.documents.items():
            year = meta.get("years") or meta.get("bib_meta", {}).get("year")
            if year:
                for topic in meta.get("topics", []):
                    rows.append({"year": int(year), "topic": topic, "doc": Path(doc_id).stem})
        df = pd.DataFrame(rows)
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title="No temporal metadata available")
            return fig
        colorscale = colormap or "Set1"
        fig = px.scatter(df, x="year", y="topic", color="doc", symbol="doc",
                         title="Research Topic Timeline by Document",
                         labels={"year": "Publication Year", "topic": "Topic"},
                         height=500, color_discrete_sequence=px.colors.qualitative.__dict__.get(colorscale, px.colors.qualitative.Set1))
        fig.update_traces(marker=dict(size=12))
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
        return fig

    def plot_entity_treemap(self, filtered_concepts: Optional[List[str]] = None, colormap: Optional[str] = None) -> go.Figure:
        target = set(filtered_concepts) if filtered_concepts else None
        rows = []
        for norm, ents in self.graph.entities.items():
            if target and norm not in target:
                continue
            if not ents:
                continue
            rows.append({
                "domain": ents[0].domain,
                "category": ents[0].category,
                "subcategory": ents[0].subcategory,
                "entity": norm,
                "value": len(ents) * (0.5 + 0.5 * self.get_salience(norm)),
                "docs": len(set(e.doc_source for e in ents))
            })
        df = pd.DataFrame(rows)
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title="No entity data")
            return fig
        colorscale = colormap or "Viridis"
        fig = px.treemap(df, path=["domain", "category", "subcategory", "entity"],
                         values="value", color="docs", color_continuous_scale=colorscale,
                         title="Hierarchical Entity Treemap")
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
        return fig

    # -----------------------------------------------------------------
    # QUANTITATIVE VISUALIZATION METHODS
    # -----------------------------------------------------------------
    def plot_quantitative_histogram(self, df: pd.DataFrame, quantity_name: str,
                                    group_by: str = "material",
                                    colormap: Optional[str] = None) -> go.Figure:
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title=f"No {quantity_name} data extracted")
            return fig
        column_map = {
            "material": "material",
            "document": "doc_stem",
            "method": "method"
        }
        group_col = column_map.get(group_by, group_by)
        if group_col not in df.columns:
            fig = go.Figure()
            fig.update_layout(title=f"Cannot group by '{group_by}': column '{group_col}' not found in data")
            return fig
        fig = go.Figure()
        groups = sorted(df[group_col].unique())
        cmap_obj = plt.get_cmap(self._get_colormap(colormap))
        for i, grp in enumerate(groups):
            subset = df[df[group_col] == grp]
            color = mcolors.to_hex(cmap_obj(i / max(len(groups) - 1, 1)))
            fig.add_trace(go.Bar(
                name=grp,
                x=[grp],
                y=[subset["value"].mean()],
                error_y=dict(
                    type='data',
                    array=[subset["value"].std()] if len(subset) > 1 else [0],
                    visible=True
                ),
                marker_color=color,
                text=[f"n={len(subset)}<br>μ={subset['value'].mean():.2f}<br>σ={subset['value'].std():.2f}"],
                textposition="outside",
                hovertemplate=f"<b>{grp}</b><br>Mean: %{{y:.2f}} {subset['unit'].iloc[0]}<br>Count: {len(subset)}<extra></extra>"
            ))
        fig.update_layout(
            barmode='group',
            title=f"{quantity_name.replace('_', ' ').title()} Values by {group_by.title()}",
            xaxis_title=group_by.title(),
            yaxis_title=f"{quantity_name.replace('_', ' ').title()} ({df['unit'].iloc[0]})",
            font=dict(family=self.font_family, size=self.font_size),
            height=500
        )
        return fig

    def plot_quantitative_sunburst(self, df: pd.DataFrame, quantity_name: str,
                                   group_by: str = "material",
                                   colormap: Optional[str] = None) -> go.Figure:
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title=f"No {quantity_name} data extracted")
            return fig
        df = df.copy()
        n_bins = min(5, max(2, len(df) // 3))
        df["value_range"] = pd.cut(df["value"], bins=n_bins, precision=1).astype(str)
        column_map = {
            "material": "material",
            "document": "doc_stem",
            "method": "method"
        }
        group_col = column_map.get(group_by, "material")
        path_cols = list(dict.fromkeys([group_col, "doc_stem", "value_range"]))
        path_cols = [c for c in path_cols if c in df.columns]
        fig = px.sunburst(
            df,
            path=path_cols,
            values="value",
            color="value",
            color_continuous_scale=colormap or "Viridis",
            title=f"{quantity_name.replace('_', ' ').title()} Distribution Hierarchy"
        )
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
        return fig

    def plot_quantitative_knowledge_graph(self, df: pd.DataFrame, quantity_name: str,
                                          group_by: str = "material",
                                          colormap: Optional[str] = None,
                                          figsize: Tuple[int, int] = (14, 12)) -> plt.Figure:
        G = nx.Graph()
        hub = f"{quantity_name}_hub"
        G.add_node(hub, node_type="hub", domain="PARAMETER")
        cmap_obj = plt.get_cmap(self._get_colormap(colormap)) if colormap else None
        column_map = {
            "material": "material",
            "document": "doc_stem",
            "method": "method"
        }
        group_col = column_map.get(group_by, "material")
        if group_col in df.columns:
            groups = sorted(df[group_col].unique())
            for i, grp in enumerate(groups):
                G.add_node(grp, node_type="group", domain=group_col.upper())
                count = len(df[df[group_col] == grp])
                G.add_edge(hub, grp, weight=count)
        docs = sorted(df["doc_stem"].unique())
        for doc in docs:
            G.add_node(doc, node_type="document", domain="DOCUMENT")
            G.add_edge(hub, doc, weight=len(df[df["doc_stem"] == doc]))
        top = df.nlargest(min(25, len(df)), "value")
        for _, row in top.iterrows():
            val_node = f"{row['value']:.1f} {row['unit']}"
            if val_node not in G:
                G.add_node(val_node, node_type="value", domain="PARAMETER", value=row["value"])
            if group_col in df.columns:
                G.add_edge(row[group_col], val_node, weight=1)
            G.add_edge(row["doc_stem"], val_node, weight=1)
        fig, ax = plt.subplots(figsize=figsize)
        pos = nx.spring_layout(G, k=0.55, iterations=60, seed=42)
        nx.draw_networkx_nodes(G, pos, nodelist=[hub], node_color="#dc2626", node_size=2500, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=groups, node_color="#3b82f6", node_size=900, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=docs, node_color="#10b981", node_size=700, ax=ax)
        val_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "value"]
        nx.draw_networkx_nodes(G, pos, nodelist=val_nodes, node_color="#f59e0b", node_size=350, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.25, width=0.8, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=self.label_font_size, ax=ax, font_family=self.font_family)
        ax.set_title(f"{quantity_name.replace('_', ' ').title()} Quantitative Knowledge Graph",
                     fontsize=self.title_font_size, fontweight='bold', fontfamily=self.font_family)
        ax.axis("off")
        plt.tight_layout()
        return fig

    def plot_quantitative_radar(self, df: pd.DataFrame, quantity_name: str,
                                group_by: str = "material",
                                colormap: Optional[str] = None) -> go.Figure:
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title=f"No {quantity_name} data extracted")
            return fig
        column_map = {
            "material": "material",
            "document": "doc_stem",
            "method": "method"
        }
        group_col = column_map.get(group_by, "material")
        if group_col not in df.columns:
            fig = go.Figure()
            fig.update_layout(title=f"Cannot group by '{group_by}': column '{group_col}' not found")
            return fig
        stats = df.groupby(group_col)["value"].agg(["mean", "std", "min", "max", "count"])
        categories = ["Mean", "Max", "Min", "Std", "Count"]
        fig = go.Figure()
        cmap_obj = plt.get_cmap(self._get_colormap(colormap)) if colormap else None
        for i, (mat, row) in enumerate(stats.iterrows()):
            values = [row["mean"], row["max"], row["min"], row["std"], float(row["count"])]
            values += values[:1]
            color = mcolors.to_hex(cmap_obj(i / max(len(stats) - 1, 1))) if cmap_obj else None
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=mat,
                line_color=color
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            title=f"{quantity_name.replace('_', ' ').title()} Statistics by Material",
            font=dict(family=self.font_family, size=self.font_size)
        )
        return fig

    def plot_quantitative_tsne(self, df: pd.DataFrame, embedding_fn: Callable,
                               quantity_name: str, group_by: str = "material",
                               colormap: Optional[str] = None,
                               figsize: Tuple[int, int] = (10, 8)) -> Optional[plt.Figure]:
        if not SKLEARN_AVAILABLE or len(df) < 5:
            return None
        column_map = {
            "material": "material",
            "document": "doc_stem",
            "method": "method"
        }
        group_col = column_map.get(group_by, "material")
        if group_col not in df.columns:
            group_col = "material"
        embs = np.array([embedding_fn(c) for c in df["context"].tolist()])
        coords = TSNE(n_components=2, perplexity=min(30, len(df) - 1), random_state=42).fit_transform(embs)
        fig, ax = plt.subplots(figsize=figsize)
        groups = df[group_col].unique()
        cmap_obj = plt.get_cmap(self._get_colormap(colormap))
        for i, grp in enumerate(groups):
            mask = df[group_col] == grp
            color = mcolors.to_hex(cmap_obj(i / max(len(groups) - 1, 1)))
            ax.scatter(coords[mask, 0], coords[mask, 1], c=color, label=grp, alpha=0.85, s=120, edgecolors='white')
        for idx, row in df.iterrows():
            ax.annotate(f"{row['value']:.0f}", (coords[idx, 0], coords[idx, 1]),
                        fontsize=self.label_font_size - 1, alpha=0.85, fontfamily=self.font_family)
        ax.legend(loc='best', fontsize=self.label_font_size)
        ax.set_title(f"{quantity_name.replace('_', ' ').title()} Context Embeddings (t-SNE)",
                     fontsize=self.title_font_size, fontweight='bold', fontfamily=self.font_family)
        ax.axis("off")
        plt.tight_layout()
        return fig

    def plot_quantitative_pca(self, df: pd.DataFrame, embedding_fn: Callable,
                              quantity_name: str, group_by: str = "material",
                              colormap: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 8)) -> Optional[plt.Figure]:
        if not SKLEARN_AVAILABLE or len(df) < 5:
            return None
        column_map = {
            "material": "material",
            "document": "doc_stem",
            "method": "method"
        }
        group_col = column_map.get(group_by, "material")
        if group_col not in df.columns:
            group_col = "material"
        embs = np.array([embedding_fn(c) for c in df["context"].tolist()])
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embs)
        var_ratio = pca.explained_variance_ratio_
        fig, ax = plt.subplots(figsize=figsize)
        groups = df[group_col].unique()
        cmap_obj = plt.get_cmap(self._get_colormap(colormap))
        for i, grp in enumerate(groups):
            mask = df[group_col] == grp
            color = mcolors.to_hex(cmap_obj(i / max(len(groups) - 1, 1)))
            ax.scatter(coords[mask, 0], coords[mask, 1], c=color, label=grp, alpha=0.85, s=120, edgecolors='white')
        for idx, row in df.iterrows():
            ax.annotate(f"{row['value']:.0f}", (coords[idx, 0], coords[idx, 1]),
                        fontsize=self.label_font_size - 1, alpha=0.85, fontfamily=self.font_family)
        ax.legend(loc='best', fontsize=self.label_font_size)
        ax.set_title(f"{quantity_name.replace('_', ' ').title()} Context Embeddings (PCA)\n"
                     f"PC1: {var_ratio[0]:.1%}, PC2: {var_ratio[1]:.1%}",
                     fontsize=self.title_font_size, fontweight='bold', fontfamily=self.font_family)
        ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%})", fontfamily=self.font_family)
        ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%})", fontfamily=self.font_family)
        plt.tight_layout()
        return fig

    def plot_quantitative_umap(self, df: pd.DataFrame, embedding_fn: Callable,
                               quantity_name: str, group_by: str = "material",
                               colormap: Optional[str] = None,
                               figsize: Tuple[int, int] = (10, 8)) -> Optional[plt.Figure]:
        if not UMAP_AVAILABLE or len(df) < 5:
            return None
        column_map = {
            "material": "material",
            "document": "doc_stem",
            "method": "method"
        }
        group_col = column_map.get(group_by, "material")
        if group_col not in df.columns:
            group_col = "material"
        embs = np.array([embedding_fn(c) for c in df["context"].tolist()])
        coords = umap.UMAP(n_neighbors=min(15, len(df) - 1), min_dist=0.1, random_state=42).fit_transform(embs)
        fig, ax = plt.subplots(figsize=figsize)
        groups = df[group_col].unique()
        cmap_obj = plt.get_cmap(self._get_colormap(colormap))
        for i, grp in enumerate(groups):
            mask = df[group_col] == grp
            color = mcolors.to_hex(cmap_obj(i / max(len(groups) - 1, 1)))
            ax.scatter(coords[mask, 0], coords[mask, 1], c=color, label=grp, alpha=0.85, s=120, edgecolors='white')
        for idx, row in df.iterrows():
            ax.annotate(f"{row['value']:.0f}", (coords[idx, 0], coords[idx, 1]),
                        fontsize=self.label_font_size - 1, alpha=0.85, fontfamily=self.font_family)
        ax.legend(loc='best', fontsize=self.label_font_size)
        ax.set_title(f"{quantity_name.replace('_', ' ').title()} Context Embeddings (UMAP)",
                     fontsize=self.title_font_size, fontweight='bold', fontfamily=self.font_family)
        ax.axis("off")
        plt.tight_layout()
        return fig

    def render_pyvis_salience(self, nx_graph: nx.Graph, concept_abstract_map: Dict,
                              filtered_concepts: Optional[List[str]] = None,
                              top_n_nodes: int = 0, physics_enabled: bool = True,
                              colormap: Optional[str] = None) -> None:
        if filtered_concepts:
            target_nodes = set(filtered_concepts)
            nx_graph = nx_graph.subgraph([n for n in nx_graph.nodes() if n in target_nodes]).copy()
        if top_n_nodes > 0 and len(nx_graph.nodes()) > top_n_nodes:
            scored = [(node, self.get_salience(node)) for node in nx_graph.nodes()]
            top_nodes = [node for node, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:top_n_nodes]]
            nx_graph = nx_graph.subgraph(top_nodes).copy()
        net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="#000000", cdn_resources='remote')
        if physics_enabled:
            net.barnes_hut(gravity=-1800, spring_length=140, damping=0.85)
        domains = list(set(nx_graph.nodes[n].get("domain", "UNKNOWN") for n in nx_graph.nodes() if "domain" in nx_graph.nodes[n]))
        cmap = plt.get_cmap(self._get_colormap(colormap)) if colormap else None
        domain_colors = {}
        if cmap:
            for i, d in enumerate(domains):
                domain_colors[d] = mcolors.to_hex(cmap(i / max(len(domains) - 1, 1)))
        for node in nx_graph.nodes():
            salience = self.get_salience(node)
            size = int(15 + salience * 55)
            border = int(1 + salience * 6)
            if self.is_core_pillar(node):
                color = "#dc2626"
            elif cmap and "domain" in nx_graph.nodes[node]:
                color = domain_colors.get(nx_graph.nodes[node]["domain"], "#3b82f6")
            else:
                color = "#1e40af" if self.is_core_pillar(node) else "#3b82f6"
            freq = concept_abstract_map.get(node, [])
            net.add_node(
                node, label=node[:25], size=size, color=color,
                borderWidth=border,
                title=f"{node}\nSalience: {salience:.2f}\nFrequency: {len(freq)}"
            )
        for u, v in nx_graph.edges():
            w = nx_graph[u][v].get('weight', 1)
            salience_u = self.get_salience(u)
            salience_v = self.get_salience(v)
            edge_weight = w * (salience_u + salience_v) / 2
            net.add_edge(u, v, value=edge_weight, width=max(1, int(edge_weight * 2)))
        html_content = net.generate_html()
        st.components.v1.html(html_content, height=750, scrolling=True)
        try:
            html_bytes = html_content.encode('utf-8')
            st.download_button("📥 Download Interactive Graph (HTML)", data=html_bytes,
                               file_name="declarmima_graph_salience.html", mime="text/html", key="pyvis_salience_download")
            del html_content, html_bytes
            import gc
            gc.collect()
        except Exception as e:
            st.error(f"Download failed: {e}")

# =====================================================================
# EMBEDDING WRAPPER
# =====================================================================
class EmbeddingWrapper:
    """Optimized embedding wrapper with batching and caching."""
    def __init__(self, embedding_source):
        self.source = embedding_source
        self._cache = {}
        self._max_cache_size = 1000

    def __call__(self, text: str) -> np.ndarray:
        text_hash = hash(text[:200])
        if text_hash in self._cache:
            return self._cache[text_hash]
        result = self._embed_single(text)
        if len(self._cache) >= self._max_cache_size:
            self._cache = dict(list(self._cache.items())[self._max_cache_size//2:])
        self._cache[text_hash] = result
        return result

    def _embed_single(self, text: str) -> np.ndarray:
        if hasattr(self.source, 'embed_query'):
            return np.array(self.source.embed_query(text))
        elif hasattr(self.source, 'embed_documents'):
            return np.array(self.source.embed_documents([text])[0])
        else:
            raise ValueError("Embedding source has no embed_query or embed_documents method")

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            if hasattr(self.source, 'embed_documents'):
                batch_embs = self.source.embed_documents(batch)
                results.extend([np.array(e) for e in batch_embs])
            else:
                results.extend([self._embed_single(t) for t in batch])
        return results

# =====================================================================
# SEMANTIC CHUNKING WITH STRUCTURE AWARENESS
# =====================================================================
def detect_scientific_sections(text: str) -> List[Tuple[str, str]]:
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
    """Optimized semantic chunking with section-aware splitting and pre-configured splitters."""
    all_text = "\n".join([p.page_content for p in pages])
    sections = detect_scientific_sections(all_text)
    chunks = []
    splitters = {
        'ABSTRACT': RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50,
                                                   separators=["\n", "\n", ". ", "; ", ", "], length_function=len),
        'CONCLUSION': RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50,
                                                     separators=["\n", "\n", ". ", "; ", ", "], length_function=len),
        'METHODS': RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100,
                                                  separators=["\n", "\n", ". ", "; ", ", "], length_function=len),
        'DEFAULT': RecursiveCharacterTextSplitter(
            chunk_size=LASER_DOMAIN_CONFIG["chunk_size"],
            chunk_overlap=LASER_DOMAIN_CONFIG["chunk_overlap"],
            separators=["\n", "\n", ". ", "; ", ", "],
            length_function=len
        )
    }
    for section_name, section_text in sections:
        splitter = splitters.get(section_name, splitters['DEFAULT'])
        section_chunks = splitter.create_documents([section_text])
        base_idx = len(chunks)
        for i, chunk in enumerate(section_chunks):
            chunk.metadata.update({
                "source": filename,
                "section": section_name,
                "chunk_index": base_idx + i,
                "section_chunk_index": i,
            })
        chunks.extend(section_chunks)
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = total
    return chunks

# =====================================================================
# QUERY-DRIVEN LAZY PROCESSING PIPELINE
# =====================================================================
class QueryDrivenProcessor:
    """
    Defers all heavy document processing until the first user query is submitted.
    Computes query-biased salience, builds dynamic knowledge graph, and returns
    tightly filtered context for the LLM.
    """
    def __init__(self):
        self.raw_files: List = []
        self._cache_key: Optional[str] = None
        self._processed = False

    def register_files(self, files: List) -> None:
        """Store uploaded file bytes without processing."""
        self.raw_files = files
        self._processed = False
        logger.info(f"Registered {len(files)} files for query-driven processing.")

    def process_for_query(self, query: str, progress_bar: Any = None) -> Tuple[
        EnhancedCrossDocumentKnowledgeGraph, FAISS, EmbeddingWrapper, Dict[str, Any], ReasoningChain]:
        """
        Executes full pipeline only when triggered by a query.
        Returns: (graph, vectorstore, embedding_fn, concept_metadata, reasoning_chain)
        """
        chain = ReasoningChain(query)
        if not self.raw_files:
            raise ValueError("No files registered for processing.")
        
        # Step 1: Embed query to bias extraction
        embed_model = load_local_embeddings()
        emb_wrapper = EmbeddingWrapper(embed_model)
        query_emb = emb_wrapper(query)
        chain.add_step("query_embedding", "Generated query embedding for bias calculation", {"dim": len(query_emb)})

        # Step 2: Load & Chunk Documents
        if progress_bar: progress_bar.progress(0.1, text="📄 Extracting & chunking documents...")
        all_chunks = []
        pages_by_file = {}
        use_pymupdf = PYMUPDF_AVAILABLE
        
        for file in self.raw_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getbuffer())
                tmp_path = tmp.name
            try:
                if use_pymupdf:
                    doc = fitz.open(tmp_path)
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        text = page.get_text("text")
                        if text.strip():
                            pages_by_file.setdefault(file.name, []).append(Document(
                                page_content=text, metadata={"source": file.name, "page": page_num + 1}
                            ))
                    doc.close()
                else:
                    loader = PyPDFLoader(tmp_path)
                    pages = loader.load()
                    pages_by_file[file.name] = pages
            except Exception as e:
                logger.warning(f"Extraction failed for {file.name}: {e}")
            finally:
                try: os.unlink(tmp_path)
                except: pass
                
        for filename, pages in pages_by_file.items():
            chunks = semantic_chunk_document(pages, filename)
            all_chunks.extend(chunks)
        if progress_bar: progress_bar.progress(0.4, text="✅ Chunking complete.")
        chain.add_step("chunking", f"Extracted {len(all_chunks)} sections/chunks", {"file_count": len(self.raw_files)})

        # Step 3: Build Vector Store
        if progress_bar: progress_bar.progress(0.5, text="🧠 Indexing embeddings...")
        texts = [c.page_content for c in all_chunks]
        batch_size = 64
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embs = embed_model.embed_documents(batch)
            all_embeddings.extend(batch_embs)
            
        import faiss
        from langchain_community.vectorstores import FAISS
        from langchain_community.docstore.in_memory import InMemoryDocstore
        
        embedding_array = np.array(all_embeddings, dtype=np.float32)
        index = faiss.IndexFlatIP(embedding_array.shape[1])
        faiss.normalize_L2(embedding_array)
        index.add(embedding_array)
        docstore = InMemoryDocstore({str(i): all_chunks[i] for i in range(len(all_chunks))})
        index_to_docstore_id = {i: str(i) for i in range(len(all_chunks))}
        vectorstore = FAISS(
            embedding_function=embed_model,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )
        if progress_bar: progress_bar.progress(0.65, text="✅ Vector index built.")
        chain.add_step("vector_index", "FAISS index constructed", {"chunk_count": len(all_chunks)})

        # Step 4: Query-Biased Concept Extraction
        if progress_bar: progress_bar.progress(0.7, text="🔍 Extracting query-biased concepts...")
        extractor = FullTextConceptExtractor(embed_model)
        custom_list = st.session_state.get('custom_priority_concepts', [])
        extractor.set_custom_priority(custom_list)
        valid_concepts, concept_metadata = extractor.extract_concepts_fast(
            all_chunks, min_salience=0.35, query_embedding=query_emb
        )
        if progress_bar: progress_bar.progress(0.85, text=f"✅ Found {len(valid_concepts)} high-salience concepts.")
        chain.add_step("concept_extraction", f"Extracted {len(valid_concepts)} concepts", 
                       {"query_bias_applied": True})

        # Step 5: Build Knowledge Graph
        if progress_bar: progress_bar.progress(0.9, text="🕸️ Constructing dynamic knowledge graph...")
        graph = EnhancedCrossDocumentKnowledgeGraph()
        dummy_bib = BibliographicMetadata("query_batch")
        dummy_bib.title = "Query-Driven Batch"
        doc_chunks = {}
        for chunk in all_chunks:
            src = chunk.metadata.get("source", "unknown")
            if src not in doc_chunks:
                doc_chunks[src] = []
            doc_chunks[src].append(chunk)
        for src, chunks in doc_chunks.items():
            graph.add_document_fast(src, chunks, dummy_bib, concept_metadata=concept_metadata)
            
        if progress_bar: progress_bar.progress(1.0, text="✅ Processing complete.")
        chain.add_step("graph_construction", "Dynamic graph built", {"documents": len(doc_chunks)})
        self._processed = True
        return graph, vectorstore, emb_wrapper, concept_metadata, chain

# =====================================================================
# SESSION STATE INITIALIZATION (extended)
# =====================================================================
def initialize_session_state():
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
        "metadata_cache": metadata_cache,
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
        "concept_selector": None,
        "custom_priority_concepts": ["melt pool dynamics", "keyhole mode", "marangoni convection"],
        "viz_font_family": "DejaVu Sans",
        "viz_font_size": 10,
        "viz_title_font_size": 14,
        "viz_label_font_size": 9,
        "viz_colormap": "viridis",
        "viz_figure_dpi": 300,
        "viz_layout": "spring",
        "llm_extraction_enabled": False,
        "viz_top_n": 25,
        "viz_active_domains": ["MATERIAL", "METHOD", "PHENOMENON", "PARAMETER", "TOPIC"],
        "viz_use_llm_ranking": False,
        "viz_salience_threshold": 0.0,
        # NEW: Query-driven pipeline state
        "query_processor": None,
        "last_query_hash": None,
        "query_cache": {}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# =====================================================================
# MEMORY MANAGEMENT UTILITIES
# =====================================================================
def cleanup_memory():
    """Force garbage collection and clear CUDA cache if available."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def get_memory_usage():
    """Get current memory usage in MB."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================
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

def estimate_model_memory(model_key: str, use_4bit: bool = False) -> Dict[str, any]:
    repo_id = get_hf_repo_id(model_key) if not is_ollama_model(model_key) else model_key
    return MODEL_MEMORY_ESTIMATES.get(repo_id, {
        "params": "Unknown", "vram_fp16": "Unknown", "vram_4bit": "Unknown", "cpu_ok": False
    })

def compute_text_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# =====================================================================
# LOCAL MODEL LOADING
# =====================================================================
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
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(repo_id, **model_kwargs)
    if "device_map" not in model_kwargs and device == "cpu":
        model = model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model, device, "transformers"

# =====================================================================
# DOCUMENT PROCESSING
# =====================================================================
def extract_laser_metadata(text: str, filename: str) -> Dict[str, any]:
    metadata = {
        "source": filename,
        "laser_topics": [],
        "parameters_found": {},
        "has_equations": bool(re.search(r'[\(=]\s*[\d.]+\s*[×*]\s*10\^', text)),
        "has_figures": bool(re.search(r'Figure\s*\d+|Fig\.\s*\d+', text, re.I)),
    }
    text_lower = text.lower()
    for topic, keywords in LASER_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            metadata["laser_topics"].append(topic)
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
            try:
                metadata["parameters_found"][param] = float(match.group(1))
            except:
                pass
    return metadata

def load_pdf_chunks(uploaded_files, use_parallel: bool = True):
    if use_parallel and len(uploaded_files) > 1:
        try:
            all_pages, temp_paths = FastPDFProcessor.process_multiple_pdfs(uploaded_files)
            all_chunks = []
            pages_by_file = {}
            for page in all_pages:
                src = page.metadata.get("source", "unknown")
                if src not in pages_by_file:
                    pages_by_file[src] = []
                pages_by_file[src].append(page)
            for filename, pages in pages_by_file.items():
                chunks = semantic_chunk_document(pages, filename)
                all_chunks.extend(chunks)
            for tmp_path in temp_paths:
                try: os.unlink(tmp_path)
                except: pass
            return all_chunks
        except Exception as e:
            logger.warning(f"Parallel processing failed: {e}. Falling back to sequential.")
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

def process_documents(uploaded_files):
    """Legacy wrapper for backward compatibility, now delegates to QueryDrivenProcessor."""
    if st.session_state.query_processor is None:
        st.session_state.query_processor = QueryDrivenProcessor()
    st.session_state.query_processor.register_files(uploaded_files)
    st.session_state.processed_files.update([f.name for f in uploaded_files])
    st.session_state.processing_complete = False  # Will be set to True after query triggers processing
    return True

# =====================================================================
# RETRIEVAL & ANSWER GENERATION
# =====================================================================
def retrieve_and_answer(
    vectorstore,
    graph: EnhancedCrossDocumentKnowledgeGraph,
    tokenizer,
    model,
    device_or_host: str,
    backend: str,
    backend_type: str,
    query: str,
    k: int = None,
    score_threshold: float = None
) -> Tuple[str, List[Document], float, Dict[str, Any], Optional[ReasoningChain]]:
    k = k or LASER_DOMAIN_CONFIG["retrieval_k"]
    score_threshold = score_threshold or LASER_DOMAIN_CONFIG["score_threshold"]
    emb_source = getattr(vectorstore, 'embedding_function', getattr(vectorstore, 'embeddings', vectorstore))
    emb_fn = EmbeddingWrapper(emb_source)
    def llm_generate(prompt: str) -> str:
        return generate_local_response(
            tokenizer=tokenizer, model_or_tag=model, device_or_host=device_or_host,
            prompt=prompt, backend=backend, backend_type=backend_type
        )
    thinker = CrossDocumentThinker(graph, vectorstore, emb_fn, llm_generate)
    answer, chain, retrieved_docs, meta = thinker.think_and_answer(query, k=k)
    avg_relevance = 0.0
    if retrieved_docs:
        query_emb = emb_fn(query)
        scores = []
        for doc in retrieved_docs:
            doc_emb = emb_fn(doc.page_content[:500])
            sim = float(np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-8))
            scores.append(sim)
        avg_relevance = np.mean(scores) if scores else 0.0
    meta["avg_vector_score"] = avg_relevance
    return answer, retrieved_docs, avg_relevance, meta, chain

def retrieve_and_answer_quantitative(
    vectorstore,
    graph: EnhancedCrossDocumentKnowledgeGraph,
    tokenizer,
    model,
    device_or_host: str,
    backend: str,
    backend_type: str,
    query: str,
    k: int = 6
) -> Tuple[str, pd.DataFrame, Dict[str, Any], List[plt.Figure], List[go.Figure], Optional[ReasoningChain]]:
    quantity_label = QuantitativeQueryEngine.detect_quantity(query)
    grouping_dim = QuantitativeQueryEngine.detect_grouping_dimension(query)
    if not quantity_label:
        answer, docs, rel, meta, chain = retrieve_and_answer(
            vectorstore, graph, tokenizer, model, device_or_host, backend, backend_type, query, k
        )
        return answer, pd.DataFrame(), meta, [], [], chain
    extractor = QuantitativeDataExtractor(graph)
    df = extractor.extract(quantity_label, group_by=grouping_dim)
    summary = extractor.summarize(quantity_label)
    def llm_generate(prompt: str) -> str:
        return generate_local_response(
            tokenizer=tokenizer, model_or_tag=model, device_or_host=device_or_host,
            prompt=prompt, backend=backend, backend_type=backend_type
        )
    quant_prompt = f"""You are an expert scientific assistant analyzing quantitative data from multiple papers.
Quantity: {quantity_label.replace('_', ' ').title()}
Summary:
- Occurrences: {summary.get('count', 0)}
- Unit: {summary.get('unit', 'N/A')}
- Range: {summary.get('value_range', ('N/A', 'N/A'))}
- Mean ± Std: {summary.get('mean', 0):.2f} ± {summary.get('std', 0):.2f}
By Material:
{json.dumps(summary.get('by_material', {}), indent=2, default=str)[:800]}
By Document:
{json.dumps(summary.get('by_document', {}), indent=2, default=str)[:800]}
User Question: {query}
Synthesize these findings rigorously. Discuss trends, outliers, agreements/discrepancies across materials/documents, and report uncertainty explicitly. Structure: Direct Answer, Evidence, Consensus/Variability, Limitations, Confidence.
"""
    answer = llm_generate(quant_prompt)
    viz = PublicationQualityVisualizationEngine(graph)
    mpl_figs: List[plt.Figure] = []
    ply_figs: List[go.Figure] = []
    if not df.empty:
        ply_figs.append(viz.plot_quantitative_histogram(df, quantity_label, grouping_dim))
        ply_figs.append(viz.plot_quantitative_sunburst(df, quantity_label, grouping_dim))
        ply_figs.append(viz.plot_quantitative_radar(df, quantity_label, grouping_dim))
        mpl_figs.append(viz.plot_quantitative_knowledge_graph(df, quantity_label, grouping_dim))
        emb_src = getattr(vectorstore, 'embedding_function', getattr(vectorstore, 'embeddings', vectorstore))
        emb_fn = EmbeddingWrapper(emb_src)
        if len(df) >= 5:
            fig_tsne = viz.plot_quantitative_tsne(df, emb_fn, quantity_label, grouping_dim)
            if fig_tsne: mpl_figs.append(fig_tsne)
            fig_pca = viz.plot_quantitative_pca(df, emb_fn, quantity_label, grouping_dim)
            if fig_pca: mpl_figs.append(fig_pca)
            fig_umap = viz.plot_quantitative_umap(df, emb_fn, quantity_label, grouping_dim)
            if fig_umap: mpl_figs.append(fig_umap)
    meta = {
        "quantity_label": quantity_label,
        "grouping_dim": grouping_dim,
        "summary": summary,
        "is_quantitative": True,
        "dataframe_rows": len(df)
    }
    chain = ReasoningChain(query)
    chain.add_step("quantitative_intent", f"Detected quantitative query: {quantity_label}", {"group_by": grouping_dim})
    chain.add_step("extraction", f"Extracted {len(df)} values", {"unit": summary.get("unit")})
    chain.add_step("visualization", f"Generated {len(mpl_figs)} matplotlib + {len(ply_figs)} plotly figures", {})
    return answer, df, meta, mpl_figs, ply_figs, chain

def generate_local_response(tokenizer, model_or_tag, device_or_host: str, prompt: str, backend: str, backend_type: str) -> str:
    if backend_type == "ollama":
        return generate_local_response_ollama(model_or_tag, device_or_host, prompt)
    else:
        return generate_local_response_transformers(tokenizer, model_or_tag, device_or_host, prompt, backend)

def generate_local_response_transformers(tokenizer, model, device: str, prompt: str, backend_name: str) -> str:
    try:
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
    try:
        client = ollama.Client(host=ollama_host)
        messages = [
            {"role": "system", "content": "You are an expert in laser-microstructure interaction research. Synthesize evidence across multiple papers rigorously."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = client.chat(
                model=model_tag, messages=messages,
                options={"temperature": LASER_DOMAIN_CONFIG["temperature"], "num_predict": LASER_DOMAIN_CONFIG["max_new_tokens"]},
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
            response = client.chat(
                model=model_tag, messages=messages,
                options={"temperature": LASER_DOMAIN_CONFIG["temperature"], "num_predict": LASER_DOMAIN_CONFIG["max_new_tokens"]}
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

# =====================================================================
# STREAMLIT UI (Extended with Salience Dropdown & Visualization Customization)
# =====================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        backend_option = st.radio("🔧 Inference Backend", options=["Hugging Face Transformers", "Ollama (if installed)"], index=0)
        st.session_state.inference_backend = backend_option
        if backend_option == "Ollama (if installed)":
            if not OLLAMA_AVAILABLE:
                st.error("❌ ollama library not installed")
                st.code("pip install ollama")
            available_ollama_models = [k for k in LOCAL_LLM_OPTIONS.keys() if is_ollama_model(k)]
            model_choice = st.selectbox("🧠 Local LLM Backend (Ollama)", options=available_ollama_models if available_ollama_models else ["No Ollama models available"], index=0)
        else:
            hf_models = [k for k in LOCAL_LLM_OPTIONS.keys() if not is_ollama_model(k)]
            model_choice = st.selectbox("🧠 Local LLM Backend (Hugging Face)", options=hf_models, index=2)
        st.session_state.llm_model_choice = model_choice
        if backend_option == "Hugging Face Transformers" and not is_ollama_model(model_choice):
            st.session_state.use_4bit_quantization = st.checkbox("🗜️ Use 4-bit quantization", value=True)
        if backend_option == "Ollama (if installed)" or is_ollama_model(model_choice):
            st.session_state.ollama_host = st.text_input("🌐 Ollama Host", value=st.session_state.ollama_host)
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
        st.markdown("#### 🔬 Laser Domain Settings")
        st.session_state.laser_domain_boost = st.checkbox("Boost laser-topic relevance", value=True)
        st.session_state.show_sources = st.checkbox("Show source citations", value=True)
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
        st.markdown("#### 🤖 LLM-Enhanced Extraction")
        st.session_state.llm_extraction_enabled = st.checkbox(
            "Enable LLM-Influenced Concept Extraction", value=False,
            help="Uses the loaded LLM to rank, disambiguate, and validate extracted concepts. Slower but more accurate."
        )
        st.markdown("#### 🎨 Visualization Customization")
        st.session_state.viz_font_family = st.selectbox(
            "Font Family",
            ["DejaVu Sans", "Arial", "Helvetica", "Times New Roman", "Computer Modern", "serif", "sans-serif"],
            index=0
        )
        st.session_state.viz_font_size = st.slider("Base Font Size", 8, 55, 16)
        st.session_state.viz_title_font_size = st.slider("Title Font Size", 10, 54, 24)
        st.session_state.viz_label_font_size = st.slider("Label Font Size", 6, 60, 14)
        st.session_state.viz_colormap = st.selectbox(
            "Default Colormap",
            list(PublicationQualityVisualizationEngine.COLORMAP_OPTIONS.keys()),
            index=list(PublicationQualityVisualizationEngine.COLORMAP_OPTIONS.keys()).index("viridis")
        )
        st.session_state.viz_layout = st.selectbox(
            "Network Layout", ["spring", "kamada_kawai", "circular"], index=0
        )
        st.session_state.viz_figure_dpi = st.slider("Figure DPI", 150, 600, 300, step=50)
        st.markdown("#### 📊 Dynamic Concept Selection")
        if st.session_state.processing_complete and st.session_state.concept_selector:
            st.session_state.viz_top_n = st.slider(
                "Top N Concepts to Visualize", 5, 100, st.session_state.viz_top_n
            )
            st.session_state.viz_active_domains = st.multiselect(
                "Filter by Domain",
                options=["MATERIAL", "METHOD", "PHENOMENON", "PARAMETER", "TOPIC"],
                default=st.session_state.viz_active_domains,
                key="viz_domain_filter"
            )
            st.session_state.viz_use_llm_ranking = st.checkbox(
                "Use LLM Ranking Order", value=st.session_state.viz_use_llm_ranking,
                help="If LLM extraction was used, sort visualizations by LLM importance score"
            )
            st.session_state.viz_salience_threshold = st.slider(
                "Minimum Salience Threshold", 0.0, 1.0, 0.0, step=0.05
            )
            if st.button("Apply Filters"):
                st.session_state.concept_selector.apply_user_selection(
                    top_n=st.session_state.viz_top_n,
                    domains=st.session_state.viz_active_domains,
                    use_llm=st.session_state.viz_use_llm_ranking,
                    salience_thresh=st.session_state.viz_salience_threshold
                )
                st.success("Filters applied!")
        st.markdown("#### 📝 Citation Format")
        st.session_state.citation_style = st.selectbox(
            "Citation display style", options=["apa", "doi", "full", "short"], index=0,
            format_func=lambda x: {"apa": "APA: FirstAuthor et al., Journal, Year", "doi": "DOI: 10.xxxx/xxxxx",
                                   "full": "Full: Author (Year). Title. Journal, Vol(Issue), Pages", "short": "Short: [FirstAuthor Year] or [DOI]"}[x]
        )
        st.session_state.max_retrieved_chunks = st.slider("Chunks to retrieve", min_value=2, max_value=10, value=6)
        st.markdown("---")
        st.markdown("### 🕸️ Visualisations")
        st.session_state.show_network = st.sidebar.checkbox("Show Knowledge Graph", value=False)
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
    st.markdown("### 📁 Upload Full-Text PDF Documents")
    uploaded_files = st.file_uploader(
        "Select PDF files about laser processing, multicomponent alloys, additive manufacturing, etc.",
        type=["pdf"], accept_multiple_files=True,
        help="Documents will be processed ONLY AFTER you submit your first query (Lazy Evaluation)."
    )
    return uploaded_files

def render_chat_interface():
    if not st.session_state.get('query_processor'):
        st.info("👆 Upload PDF documents above, then ask your question. Processing triggers on query.")
        return
    if not st.session_state.query_processor.raw_files:
        st.warning("⚠️ Please upload PDF files first.")
        return

    if st.session_state.llm_tokenizer is None and st.session_state.llm_model_choice:
        backend_type = "ollama" if is_ollama_model(st.session_state.llm_model_choice) else "transformers"
        with st.spinner(f"Loading {st.session_state.llm_model_choice}..."):
            result = load_local_llm(st.session_state.llm_model_choice, use_4bit=st.session_state.get('use_4bit_quantization', True))
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

    has_model = (
        st.session_state.llm_backend_type == "ollama" and st.session_state.llm_model is not None
    ) or (
        st.session_state.llm_backend_type == "transformers" and st.session_state.llm_tokenizer is not None
    )
    if not has_model:
        st.warning("Please select and load a model in the sidebar first")
        return

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
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
            if message.get("reasoning_meta") and st.session_state.show_reasoning_chain and message["role"] == "assistant":
                meta = message["reasoning_meta"]
                with st.expander("🧠 Reasoning Chain"):
                    st.markdown(f"**Query entities detected:** {', '.join(meta.get('query_entities', [])) or 'None'}")
                    st.markdown(f"**Cross-document consensus found:** {meta.get('consensus_found', 0)}")
                    st.markdown(f"**Contradictions detected:** {meta.get('contradictions_found', 0)}")
                    st.markdown(f"**Multi-hop expansion:** {'Yes' if meta.get('multi_hop_expansion') else 'No'}")
                    if meta.get('relevance'):
                        st.markdown(f"**Response relevance:** {meta['relevance']:.2f}/1.0")
            if message.get("reasoning_chain") and st.session_state.show_reasoning_chain and message["role"] == "assistant":
                with st.expander("🧠 Full Thinking Trace", expanded=False):
                    st.markdown(message["reasoning_chain"].to_markdown())
                if st.button("Render Thinking Graph", key=f"think_graph_{i}"):
                    viz = st.session_state.visualization_engine
                    if viz:
                        fig = viz.plot_reasoning_chain(message["reasoning_chain"])
                        st.pyplot(fig)

    if prompt := st.chat_input("Ask a cross-document scientific question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("⏳ Triggering query-driven processing..."):
                progress = st.progress(0.0, text="Initializing pipeline...")
                # Check cache first
                q_hash = compute_text_hash(prompt)
                if q_hash in st.session_state.query_cache and not st.session_state.query_processor._processed:
                    cached = st.session_state.query_cache[q_hash]
                    graph, vectorstore, emb_fn, concept_metadata, chain = cached
                    st.session_state.processing_complete = True
                else:
                    graph, vectorstore, emb_fn, concept_metadata, chain = st.session_state.query_processor.process_for_query(prompt, progress)
                    st.session_state.query_cache[q_hash] = (graph, vectorstore, emb_fn, concept_metadata, chain)
                    st.session_state.processing_complete = True
                    st.session_state.knowledge_graph = graph
                    st.session_state.concept_selector = DynamicConceptSelector(graph)

            with st.spinner("🔍 Running cross-document reasoning..."):
                is_quantitative = QuantitativeQueryEngine.detect_quantity(prompt) is not None
                if is_quantitative and st.session_state.knowledge_graph:
                    answer, df, meta, mpl_figs, ply_figs, chain = retrieve_and_answer_quantitative(
                        vectorstore, st.session_state.knowledge_graph,
                        st.session_state.llm_tokenizer, st.session_state.llm_model,
                        st.session_state.llm_device_or_host,
                        st.session_state.llm_model_choice, st.session_state.llm_backend_type,
                        prompt, k=st.session_state.max_retrieved_chunks
                    )
                    st.markdown(answer)
                    if not df.empty:
                        st.markdown(f"**📊 Extracted {len(df)} `{meta['quantity_label']}` values "
                                    f"across {df['doc_stem'].nunique()} documents**")
                        with st.expander("📈 Quantitative Visualizations", expanded=True):
                            tabs = st.tabs(["Histogram", "Sunburst", "Radar", "Knowledge Graph", "t-SNE / PCA / UMAP"])
                            with tabs[0]:
                                for fig in [f for f in ply_figs if "Histogram" in f.layout.title.text]:
                                    st.plotly_chart(fig, use_container_width=True)
                            with tabs[1]:
                                for fig in [f for f in ply_figs if "Sunburst" in f.layout.title.text]:
                                    st.plotly_chart(fig, use_container_width=True)
                            with tabs[2]:
                                for fig in [f for f in ply_figs if "Radar" in f.layout.title.text]:
                                    st.plotly_chart(fig, use_container_width=True)
                            with tabs[3]:
                                for fig in mpl_figs:
                                    if "Knowledge Graph" in (fig.axes[0].get_title() or ""):
                                        st.pyplot(fig)
                                        buf = BytesIO()
                                        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                                        st.download_button("📥 Download Graph (PNG)", buf.getvalue(),
                                                           file_name=f"{meta['quantity_label']}_kg.png", mime="image/png",
                                                           key=f"quant_kg_dl_{meta['quantity_label']}")
                            with tabs[4]:
                                for fig in mpl_figs:
                                    title = fig.axes[0].get_title() or ""
                                    if any(x in title for x in ["t-SNE", "PCA", "UMAP"]):
                                        st.pyplot(fig)
                                        buf = BytesIO()
                                        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                                        st.download_button(f"📥 Download {title.split('(')[-1].replace(')','')} (PNG)",
                                                           buf.getvalue(), file_name=f"{meta['quantity_label']}_dr.png",
                                                           mime="image/png",
                                                           key=f"quant_dr_dl_{meta['quantity_label']}_{title[:10]}")
                        with st.expander("🔢 Raw Extracted Data"):
                            st.dataframe(df, use_container_width=True)
                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Download CSV", csv,
                                               file_name=f"{meta['quantity_label']}_data.csv", mime="text/csv",
                                               key=f"quant_csv_dl_{meta['quantity_label']}")
                else:
                    answer, retrieved_docs, avg_relevance, reasoning_meta, chain = retrieve_and_answer(
                        vectorstore, st.session_state.knowledge_graph,
                        st.session_state.llm_tokenizer, st.session_state.llm_model,
                        st.session_state.llm_device_or_host,
                        st.session_state.llm_model_choice, st.session_state.llm_backend_type,
                        prompt, k=st.session_state.max_retrieved_chunks
                    )
                    reasoning_meta['relevance'] = avg_relevance
                    st.markdown(answer)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": retrieved_docs if not is_quantitative else [],
                    "reasoning_meta": reasoning_meta,
                    "reasoning_chain": chain
                })

def render_footer():
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
    with col3:
        st.markdown("**🔐 Privacy & Science:**")
        st.caption("• All processing happens locally")
        st.caption("• Cross-document reasoning uses extracted entities only")
        st.caption("• Uncertainty is explicitly reported, never hidden")

def main():
    st.set_page_config(
        page_title="🔬 DECLARMIMA: Query-Driven Salience + Publication Viz",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
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
    st.markdown('<h1 class="main-header">🔬 DECLARMIMA: Query-First Processing + Dynamic Viz</h1>', unsafe_allow_html=True)
    st.markdown("""
<div style="text-align:center;color:#64748b;margin-bottom:1.5rem">
Upload <strong>full-text PDF papers</strong> on multicomponent alloys and laser processing.
Our system now uses <strong>Query-Driven Lazy Evaluation</strong>: no heavy indexing occurs until you ask a question.
This ensures <strong>salience & visualizations are tightly aligned to your specific intent</strong>.<br>
<strong>Core Pillars:</strong> LASER, MICROSTRUCTURE, INTERACTION, <span style="color:#dc2626;font-weight:bold">MULTICOMPONENT ALLOY</span>.<br>
All visualizations are <strong>publication-quality</strong> with 50+ colormaps, UMAP, PCA, t‑SNE, Bokeh chord, and PyVis networks.
Dynamic top-N filtering allows precise control over visualized concepts.
</div>
""", unsafe_allow_html=True)
    initialize_session_state()
    render_sidebar()

    if st.session_state.llm_model_choice and not is_ollama_model(st.session_state.llm_model_choice):
        mem_info = estimate_model_memory(st.session_state.llm_model_choice, st.session_state.get('use_4bit_quantization', True))
        available_vram = get_available_gpu_memory()
        if available_vram and not mem_info['cpu_ok']:
            required = float(mem_info['vram_4bit'].replace('GB','').replace('~','').strip()) if 'GB' in mem_info['vram_4bit'] else 100
            if available_vram < required:
                st.markdown(f"""
<div style="background:#fef3c7;border-left:4px solid #f59e0b;padding:0.75rem;border-radius:0 0.5rem 0.5rem 0;margin:0.5rem 0">
⚠️ <strong>Memory Warning:</strong> {st.session_state.llm_model_choice} requires ~{mem_info['vram_4bit']} VRAM.
You have ~{available_vram:.1f}GB available.
</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        uploaded_files = render_document_uploader()
        if uploaded_files and st.button("📥 Register Files", type="primary", use_container_width=True):
            if st.session_state.query_processor is None:
                st.session_state.query_processor = QueryDrivenProcessor()
            st.session_state.query_processor.register_files(uploaded_files)
            st.session_state.processed_files.update([f.name for f in uploaded_files])
            st.success(f"✅ Registered {len(uploaded_files)} files. Ready for query-driven processing!")
        elif uploaded_files:
            st.warning("⏳ Click 'Register Files' to prepare for query-driven processing")
        else:
            st.info("📁 Upload full-text PDF files to start")

        if st.session_state.processed_files:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.clear()
                st.rerun()

    with col2:
        if st.session_state.query_processor and st.session_state.query_processor.raw_files:
            render_chat_interface()
        else:
            st.markdown("""
<div class="info-card">
<h3>👋 Welcome to Query-First Processing + Publication-Quality Visualization</h3>
<p>This upgraded system features:</p>
<ul>
<li><strong>Query-Driven Lazy Evaluation:</strong> Zero upfront indexing. Processing triggers ONLY after your first question.</li>
<li><strong>Dynamic Query-Biased Salience:</strong> Concepts are weighted by semantic similarity to your specific query.</li>
<li><strong>Instant Contextual Visualizations:</strong> Graphs & charts auto-filter to your research focus.</li>
<li><strong>Enhanced Core Pillars:</strong> LASER, MICROSTRUCTURE, INTERACTION, <strong>MULTICOMPONENT ALLOY</strong></li>
<li><strong>Semantic Similarity Boost:</strong> Concepts related to pillars get automatic salience boost via embeddings</li>
<li><strong>LLM-Influenced Extraction:</strong> Optional LLM ranking, validation, and disambiguation</li>
<li><strong>Dynamic Top-N Selector:</strong> Control exactly which concepts appear in visualizations</li>
<li><strong>Full-Text Processing:</strong> section‑aware chunking (Abstract, Methods, Results, Discussion, Conclusion)</li>
<li><strong>Multi-Factor Salience:</strong> frequency, cross‑doc, section importance, quantitative signal, proposal similarity, semantic similarity</li>
<li><strong>Publication-Quality Viz:</strong> 50+ colormaps, customizable fonts, UMAP, PCA, t-SNE, Bokeh/HoloViews chord, PyVis</li>
</ul>
<p><strong>Getting started:</strong></p>
<ol>
<li>Upload one or more PDF files (full papers)</li>
<li>Click "Register Files" (no processing happens yet!)</li>
<li>Type your scientific question in the chat</li>
<li>Watch the system process, reason, and visualize in real-time</li>
</ol>
</div>
""", unsafe_allow_html=True)
            st.markdown("**Try asking:**")
            demo_qs = [
                "What is the effect of laser power on interfacial IMC thickness in Sn‑Ag‑Cu/Cu joints?",
                "Do these papers agree on the optimal hatch distance for defect‑free LPBF of Al‑Cr‑Fe‑Ni alloys?",
                "Summarize the phase‑field models used for simulating selective laser melting of multicomponent alloys.",
                "How does the composition of high entropy alloys affect their thermal conductivity during laser processing?",
            ]
            for q in demo_qs:
                if st.button(f"💬 {q}", use_container_width=True, key=f"demo_{q[:20]}"):
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.rerun()

    # --- Global Visualization Dashboard (Publication Quality) ---
    if st.session_state.processing_complete and st.session_state.knowledge_graph:
        st.markdown("---")
        st.markdown("## 🔬 Publication-Quality Scientific Visualization Dashboard")
        if st.session_state.visualization_engine is None:
            st.session_state.visualization_engine = PublicationQualityVisualizationEngine(
                st.session_state.knowledge_graph,
                font_family=st.session_state.viz_font_family,
                font_size=st.session_state.viz_font_size,
                title_font_size=st.session_state.viz_title_font_size,
                label_font_size=st.session_state.viz_label_font_size,
                default_colormap=st.session_state.viz_colormap,
                figure_dpi=st.session_state.viz_figure_dpi
            )
        else:
            st.session_state.visualization_engine.font_family = st.session_state.viz_font_family
            st.session_state.visualization_engine.font_size = st.session_state.viz_font_size
            st.session_state.visualization_engine.title_font_size = st.session_state.viz_title_font_size
            st.session_state.visualization_engine.label_font_size = st.session_state.viz_label_font_size
            st.session_state.visualization_engine.default_colormap = st.session_state.viz_colormap
            st.session_state.visualization_engine.figure_dpi = st.session_state.viz_figure_dpi

        viz = st.session_state.visualization_engine
        cmap = st.session_state.viz_colormap
        layout = st.session_state.viz_layout
        active_cmap = st.selectbox("🎨 Active Colormap for this session",
                                   list(PublicationQualityVisualizationEngine.COLORMAP_OPTIONS.keys()),
                                   index=list(PublicationQualityVisualizationEngine.COLORMAP_OPTIONS.keys()).index(cmap),
                                   key="active_cmap")
        selector = st.session_state.concept_selector
        if selector:
            filtered = selector.get_filtered_concepts()
            meta = selector.get_selection_metadata()
            st.info(f"🎯 Showing {meta['filtered_count']} of {meta['total_available']} concepts (Top {meta['top_n']} | Domains: {', '.join(meta['active_domains'])})")
        else:
            filtered = None

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Top Entities & Consensus",
            "🕸️ Knowledge Graphs",
            "☀️ Hierarchical Sunbursts",
            "📡 Document Profiles",
            "⚡ Contradictions & Salience",
            "🔬 Embedding Spaces (t-SNE/UMAP/PCA)",
            "🔢 Quantitative Explorer"  # NEW
        ])
        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                summary = st.session_state.knowledge_graph.get_knowledge_summary()
                if summary['high_salience_concepts']:
                    top_salience = summary['high_salience_concepts'][:10]
                    fig = px.bar(
                        x=[c[0] for c in top_salience],
                        y=[c[1]['salience'] for c in top_salience],
                        labels={'x': 'Concept', 'y': 'Salience'},
                        title="Top Concepts by Salience",
                        color=[c[1]['salience'] for c in top_salience],
                        color_continuous_scale=active_cmap if active_cmap in px.colors.named_colorscales() else "Viridis"
                    )
                    fig.update_layout(font=dict(family=viz.font_family, size=viz.font_size))
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.plotly_chart(viz.plot_consensus_waterfall(top_n=10, colormap=active_cmap), use_container_width=True)
                st.plotly_chart(viz.plot_entity_treemap(filtered_concepts=filtered, colormap=active_cmap), use_container_width=True)
        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Static Bipartite Network (Salience‑Aware)**")
                fig_net = viz.plot_static_knowledge_network(filtered_concepts=filtered, top_n=30, layout=layout, colormap=active_cmap)
                st.pyplot(fig_net)
                buf = BytesIO()
                fig_net.savefig(buf, format="png", dpi=viz.figure_dpi, bbox_inches="tight")
                st.download_button("📥 Download Network (PNG)", data=buf.getvalue(),
                                   file_name="knowledge_network.png", mime="image/png",
                                   key="static_net_dl")
            with c2:
                st.markdown("**Salience‑Aware Chord Diagram (Plotly)**")
                st.plotly_chart(viz.plot_chord_cooccurrence(filtered_concepts=filtered, top_n=16, colormap=active_cmap), use_container_width=True)
                if BOKEH_AVAILABLE:
                    st.markdown("**Interactive Chord-Style Network (Bokeh)**")
                    bokeh_fig = viz.plot_bokeh_chord(filtered_concepts=filtered, top_n=20, colormap=active_cmap)
                    if bokeh_fig:
                        from bokeh.embed import file_html
                        from bokeh.resources import CDN
                        html = file_html(bokeh_fig, CDN, "Bokeh Chord")
                        st.components.v1.html(html, height=850)
                        st.download_button("📥 Download Bokeh HTML", data=html.encode('utf-8'),
                                           file_name="bokeh_chord.html", mime="text/html",
                                           key="bokeh_chord_dl")
        with tab3:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.plotly_chart(viz.plot_methods_sunburst(filtered_concepts=filtered, top_n_per_category=20, colormap=active_cmap), use_container_width=True)
            with c2:
                st.plotly_chart(viz.plot_materials_sunburst(filtered_concepts=filtered, top_n_per_category=20, colormap=active_cmap), use_container_width=True)
            with c3:
                st.plotly_chart(viz.plot_topics_sunburst(colormap=active_cmap), use_container_width=True)
        with tab4:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(viz.plot_document_radar(filtered_concepts=filtered, colormap=active_cmap), use_container_width=True)
            with c2:
                st.plotly_chart(viz.plot_temporal_timeline(colormap=active_cmap), use_container_width=True)
        with tab5:
            st.markdown("**Cross-Document Contradiction Matrix**")
            st.plotly_chart(viz.plot_contradiction_matrix(colormap=active_cmap), use_container_width=True)
            contrs = st.session_state.knowledge_graph.find_all_contradictions(threshold_factor=1.5)
            if contrs:
                df_contr = pd.DataFrame(contrs)
                st.dataframe(df_contr, use_container_width=True)
        with tab6:
            st.markdown("### Dimensionality Reduction of Entity Embeddings")
            emb_source = getattr(st.session_state.vectorstore, 'embedding_function',
                                 getattr(st.session_state.vectorstore, 'embeddings', st.session_state.vectorstore))
            emb_wrapper = EmbeddingWrapper(emb_source)
            dr_col1, dr_col2, dr_col3 = st.columns(3)
            with dr_col1:
                st.markdown("**t-SNE**")
                if SKLEARN_AVAILABLE:
                    fig_tsne = viz.plot_entity_tsne(embedding_fn=emb_wrapper, filtered_concepts=filtered, top_n=60, colormap=active_cmap)
                    if fig_tsne:
                        st.pyplot(fig_tsne)
                        buf = BytesIO()
                        fig_tsne.savefig(buf, format="png", dpi=viz.figure_dpi, bbox_inches="tight")
                        st.download_button("📥 Download t-SNE", data=buf.getvalue(),
                                           file_name="entity_tsne.png", mime="image/png", key="tsne_dl")
                else:
                    st.info("Install scikit-learn for t-SNE")
            with dr_col2:
                st.markdown("**UMAP**")
                if UMAP_AVAILABLE:
                    fig_umap = viz.plot_entity_umap(embedding_fn=emb_wrapper, filtered_concepts=filtered, top_n=60, colormap=active_cmap)
                    if fig_umap:
                        st.pyplot(fig_umap)
                        buf = BytesIO()
                        fig_umap.savefig(buf, format="png", dpi=viz.figure_dpi, bbox_inches="tight")
                        st.download_button("📥 Download UMAP", data=buf.getvalue(),
                                           file_name="entity_umap.png", mime="image/png", key="umap_dl")
                else:
                    st.info("Install umap-learn: `pip install umap-learn`")
            with dr_col3:
                st.markdown("**PCA**")
                if SKLEARN_AVAILABLE:
                    fig_pca = viz.plot_entity_pca(embedding_fn=emb_wrapper, filtered_concepts=filtered, top_n=60, colormap=active_cmap)
                    if fig_pca:
                        st.pyplot(fig_pca)
                        buf = BytesIO()
                        fig_pca.savefig(buf, format="png", dpi=viz.figure_dpi, bbox_inches="tight")
                        st.download_button("📥 Download PCA", data=buf.getvalue(),
                                           file_name="entity_pca.png", mime="image/png", key="pca_dl")
                else:
                    st.info("Install scikit-learn for PCA")
        with tab7:
            st.markdown("## 🔢 Quantitative Data Explorer")
            st.markdown("Browse all numerically extracted parameters across the corpus without typing a query.")
            qty_options = list(QUANTITY_PATTERNS.keys())
            selected_qty = st.selectbox("Select quantitative parameter", qty_options, index=qty_options.index("laser_power") if "laser_power" in qty_options else 0)
            group_opt = st.radio("Group by", ["material", "document", "method"], horizontal=True)
            if st.session_state.knowledge_graph:
                extractor = QuantitativeDataExtractor(st.session_state.knowledge_graph)
                df_qty = extractor.extract(selected_qty, group_by=group_opt)
                summary_qty = extractor.summarize(selected_qty)
                if not df_qty.empty:
                    st.success(f"Found {summary_qty['count']} values ({summary_qty['unit']}) "
                               f"ranging {summary_qty['value_range'][0]:.2f} → {summary_qty['value_range'][1]:.2f}")
                    qviz = PublicationQualityVisualizationEngine(st.session_state.knowledge_graph)
                    q_cmap = st.session_state.viz_colormap
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(qviz.plot_quantitative_histogram(df_qty, selected_qty, group_opt, q_cmap), use_container_width=True)
                    with c2:
                        st.plotly_chart(qviz.plot_quantitative_sunburst(df_qty, selected_qty, group_opt, q_cmap), use_container_width=True)
                    c3, c4 = st.columns(2)
                    with c3:
                        st.plotly_chart(qviz.plot_quantitative_radar(df_qty, selected_qty, group_opt, q_cmap), use_container_width=True)
                    with c4:
                        fig_kg = qviz.plot_quantitative_knowledge_graph(df_qty, selected_qty, group_opt, q_cmap)
                        st.pyplot(fig_kg)
                        buf = BytesIO()
                        fig_kg.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                        st.download_button("📥 Download KG", buf.getvalue(),
                                           file_name=f"{selected_qty}_kg.png", mime="image/png",
                                           key=f"explorer_kg_dl_{selected_qty}")
                    with st.expander("🔢 Full Data Table"):
                        st.dataframe(df_qty, use_container_width=True)
                        csv = df_qty.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download CSV", csv, file_name=f"{selected_qty}_data.csv", mime="text/csv",
                                           key=f"explorer_csv_dl_{selected_qty}")
                else:
                    st.warning(f"No `{selected_qty}` values were extracted from the uploaded documents. "
                               f"Ensure the PDFs contain explicit numeric statements (e.g. 'laser power of 200 W').")
    render_footer()

if __name__ == "__main__":
    main()
