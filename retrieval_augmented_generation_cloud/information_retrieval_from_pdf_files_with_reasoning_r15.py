#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LASER MICROSTRUCTURE RAG CHATBOT - CROSS-DOCUMENT SCIENTIFIC REASONING & VISUALIZATION
========================================================================================
Single standalone Streamlit application integrating:
  - Cross-document consensus, contradiction, and gap detection
  - Hierarchical taxonomy (Materials, Methods, Phenomena, Parameters)
  - DGL heterogeneous graph construction (optional) + NetworkX fallback
  - Graph diffusion retrieval (personalized PageRank re-ranking)
  - Explicit reasoning chain / "thinking" trace with visual graph
  - Publication-ready static visualizations:
      * Static bipartite knowledge network (NetworkX + Matplotlib)
      * Chord-style co-occurrence (Plotly)
      * Hierarchical sunbursts (Methods, Materials, Topics)
      * Document radar profiles
      * Contradiction severity matrix
      * Consensus waterfall
      * Entity t-SNE embedding map
      * Temporal timeline
      * Reasoning chain graph
  - All original PyVis interactive graph, retrieval debugger, feedback loop
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

# Matplotlib / NetworkX for static publication graphs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx

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
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# =============================================
# GLOBAL CONFIGURATION
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
# HIERARCHICAL TAXONOMY FOR VISUALIZATION
# =============================================
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
                "complex concentrated alloy", "refractory hea", "crmnfeconi"
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
    for domain, categories in ENTITY_TAXONOMY.items():
        for category, subcategories in categories.items():
            for subcategory, aliases in subcategories.items():
                if any(alias in norm for alias in aliases):
                    return domain, category, subcategory
    return "UNKNOWN", "UNKNOWN", "UNKNOWN"


# =============================================
# BIBLIOGRAPHIC METADATA
# =============================================

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


# =============================================
# ENHANCED SCIENTIFIC ENTITY & CLAIM
# =============================================

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
    normalized: str = field(init=False)
    domain: str = field(init=False)
    category: str = field(init=False)
    subcategory: str = field(init=False)

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
            "context": self.context[:200]
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
    supporting: List[Tuple[str, int]] = field(default_factory=list)
    contradicting: List[Tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim_text, "subject": self.subject, "predicate": self.predicate,
            "object": self.object_val, "source": self.doc_source, "confidence": self.confidence,
            "supporting_count": len(self.supporting), "contradicting_count": len(self.contradicting)
        }


# =============================================
# REASONING CHAIN (THINKING TRACE)
# =============================================

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


# =============================================
# ENHANCED CROSS-DOCUMENT KNOWLEDGE GRAPH
# =============================================

class EnhancedCrossDocumentKnowledgeGraph:
    def __init__(self):
        self.entities: Dict[str, List[EnhancedScientificEntity]] = defaultdict(list)
        self.claims: List[EnhancedScientificClaim] = []
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)
        self.chunk_index: Dict[str, List[Document]] = defaultdict(list)
        self.dgl_graph = None
        self.dgl_node_maps: Dict[str, Dict[str, int]] = {}
        self.entity_embeddings: Optional[np.ndarray] = None
        self._entity_list: List[str] = []

    def add_document(self, doc_id: str, chunks: List[Document], bib_meta: Any):
        self.documents[doc_id] = {
            "bib_meta": bib_meta.to_dict() if hasattr(bib_meta, 'to_dict') else {},
            "chunk_count": len(chunks),
            "topics": set(),
            "years": bib_meta.year if hasattr(bib_meta, 'year') else None
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
            "consensus_topics": [k for k, v in self.entities.items() if len(self.entity_index.get(k, set())) > 1],
            "domains": Counter([e.domain for ents in self.entities.values() for e in ents]).most_common(),
            "categories": Counter([e.category for ents in self.entities.values() for e in ents]).most_common(),
        }


# =============================================
# GRAPH DIFFUSION RETRIEVER
# =============================================

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


# =============================================
# CROSS-DOCUMENT THINKER
# =============================================

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
        user = f"{context}\n{consensus_text}\n{contradiction_text}\n{claim_text}\n\nQuestion: {query}\n\nProvide a rigorous scientific answer following the structure above."
        return system + "\n\n" + user


# =============================================
# VISUALIZATION ENGINE
# =============================================

class VisualizationEngine:
    def __init__(self, graph: EnhancedCrossDocumentKnowledgeGraph):
        self.graph = graph
        self.color_map = {
            "MATERIAL": "#3b82f6", "METHOD": "#8b5cf6", "PHENOMENON": "#f59e0b",
            "PARAMETER": "#10b981", "UNKNOWN": "#6b7280", "TOPIC": "#ec4899"
        }

    def plot_static_knowledge_network(self, top_n: int = 25, figsize: Tuple[int, int] = (14, 12),
                                      layout: str = "spring") -> plt.Figure:
        G = nx.Graph()
        ent_counts = Counter({k: len(v) for k, v in self.graph.entities.items()})
        top_entities = [e for e, _ in ent_counts.most_common(top_n)]

        for doc_id in self.graph.documents:
            G.add_node(Path(doc_id).stem, node_type="doc", bipartite=0)

        for ent in top_entities:
            ents = self.graph.entities[ent]
            domain = ents[0].domain if ents else "UNKNOWN"
            G.add_node(ent, node_type="entity", domain=domain, bipartite=1)
            for e in ents:
                doc_node = Path(e.doc_source).stem
                if doc_node in G:
                    G.add_edge(doc_node, ent, weight=e.confidence)

        fig, ax = plt.subplots(figsize=figsize)
        if layout == "spring":
            pos = nx.spring_layout(G, k=0.6, iterations=50, seed=42)
        elif layout == "kamada":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.shell_layout(G)

        doc_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "doc"]
        ent_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "entity"]

        nx.draw_networkx_nodes(G, pos, nodelist=doc_nodes, node_color="#1e40af",
                               node_shape="s", node_size=900, alpha=0.9, ax=ax, label="Documents")
        ent_colors = [self.color_map.get(G.nodes[n].get("domain", "UNKNOWN"), "#6b7280") for n in ent_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=ent_nodes, node_color=ent_colors,
                               node_shape="o", node_size=450, alpha=0.85, ax=ax, label="Entities")
        nx.draw_networkx_edges(G, pos, alpha=0.25, width=0.8, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)

        legend_patches = [mpatches.Patch(color="#1e40af", label="Document")]
        for dom, col in self.color_map.items():
            if dom != "UNKNOWN":
                legend_patches.append(mpatches.Patch(color=col, label=dom))
        ax.legend(handles=legend_patches, loc="upper left", fontsize=8)
        ax.set_title("Static Cross-Document Knowledge Network", fontsize=14, fontweight='bold')
        ax.axis("off")
        plt.tight_layout()
        return fig

    def plot_chord_cooccurrence(self, top_n: int = 14) -> go.Figure:
        entities, mat = self.graph.get_entity_cooccurrence_matrix(top_n)
        if not entities:
            fig = go.Figure()
            fig.update_layout(title="No entity co-occurrence data")
            return fig
        angles = np.linspace(0, 2 * np.pi, len(entities), endpoint=False)
        fig = go.Figure()
        for i, ent in enumerate(entities):
            fig.add_trace(go.Barpolar(
                r=[1], theta=[np.degrees(angles[i])],
                width=[10], marker_color=self.color_map.get(
                    self.graph.entities[ent][0].domain if self.graph.entities.get(ent) else "UNKNOWN", "gray"),
                name=ent, opacity=0.9, showlegend=False,
                hoverinfo="text", text=[f"{ent}<br>Count: {len(self.graph.entities.get(ent, []))}"]
            ))
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                if mat[i][j] > 0:
                    fig.add_trace(go.Scatterpolar(
                        r=[0.2, 0.6, 0.2],
                        theta=[np.degrees(angles[i]), np.degrees((angles[i] + angles[j]) / 2), np.degrees(angles[j])],
                        mode='lines', line=dict(color='rgba(100,100,100,0.3)', width=min(mat[i][j], 3)),
                        showlegend=False, hoverinfo='skip'
                    ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=False), angularaxis=dict(visible=False)),
            title="Entity Co-occurrence Chord Diagram (Document-level)",
            height=700, width=700
        )
        return fig

    def _build_sunburst_df(self, domain_filter: str) -> pd.DataFrame:
        rows = []
        for norm, ents in self.graph.entities.items():
            if not ents:
                continue
            e = ents[0]
            if e.domain != domain_filter:
                continue
            rows.append({
                "domain": e.domain,
                "category": e.category,
                "subcategory": e.subcategory,
                "entity": norm,
                "value": len(ents),
                "doc_count": len(set(x.doc_source for x in ents))
            })
        return pd.DataFrame(rows)

    def plot_methods_sunburst(self) -> go.Figure:
        df = self._build_sunburst_df("METHOD")
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title="No METHOD entities found")
            return fig
        fig = px.sunburst(df, path=["domain", "category", "subcategory", "entity"],
                          values="value", color="doc_count", color_continuous_scale="Blues",
                          title="Hierarchical Methods Taxonomy<br><sub>Experimental → Microscopy/Spectroscopy | Computational → FEM/MD/DFT/Phase-Field/ML</sub>")
        return fig

    def plot_materials_sunburst(self) -> go.Figure:
        df = self._build_sunburst_df("MATERIAL")
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title="No MATERIAL entities found")
            return fig
        fig = px.sunburst(df, path=["domain", "category", "subcategory", "entity"],
                          values="value", color="doc_count", color_continuous_scale="Greens",
                          title="Material System Hierarchy<br><sub>Pure → Binary → Ternary → HEA → Compound → Polymer</sub>")
        return fig

    def plot_topics_sunburst(self) -> go.Figure:
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
        fig = px.sunburst(df, path=["topic"], values="count",
                          color="count", color_continuous_scale="Oranges",
                          title="Study Topics Distribution<br><sub>Laser, Microstructure, Interface, AM, Simulation...</sub>")
        return fig

    def plot_document_radar(self) -> go.Figure:
        categories = ["Laser Parameters", "Materials", "Exp. Methods", "Simulation", "Phenomena", "Properties"]
        cat_map = {
            "Laser Parameters": ["PARAMETER"],
            "Materials": ["MATERIAL"],
            "Exp. Methods": ["METHOD:Experimental"],
            "Simulation": ["METHOD:Computational"],
            "Phenomena": ["PHENOMENON"],
            "Properties": ["PARAMETER:Outcome"]
        }
        fig = go.Figure()
        for doc_id in self.graph.documents:
            values = []
            for cat in categories:
                count = 0
                target_domains = cat_map[cat]
                for norm, ents in self.graph.entities.items():
                    if any(e.doc_source == doc_id for e in ents):
                        e = ents[0]
                        if e.domain in target_domains or f"{e.domain}:{e.category}" in target_domains:
                            count += len([x for x in ents if x.doc_source == doc_id])
                values.append(count)
            values += values[:1]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=Path(doc_id).stem
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(5, max([max(t.r) for t in fig.data] or [5]))])),
            showlegend=True, title="Document Coverage Profiles (Radar)"
        )
        return fig

    def plot_contradiction_matrix(self) -> go.Figure:
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
        fig = go.Figure(data=go.Heatmap(
            z=mat, x=doc_stems, y=doc_stems,
            colorscale=[[0, "white"], [0.33, "#fcd34d"], [0.66, "#f97316"], [1, "#dc2626"]],
            text=annotations, texttemplate="%{text}", hoverinfo="text"
        ))
        fig.update_layout(title="Cross-Document Contradiction Severity Matrix", height=600, width=600)
        return fig

    def plot_consensus_waterfall(self, top_n: int = 10) -> go.Figure:
        consensus = self.graph.find_all_consensus(min_docs=2)[:top_n]
        if not consensus:
            fig = go.Figure()
            fig.update_layout(title="No consensus data available")
            return fig
        entities = [c["entity"] for c in consensus]
        means = [c["mean"] for c in consensus]
        stds = [c["std"] for c in consensus]
        doc_counts = [c["doc_count"] for c in consensus]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=entities, y=means,
            error_y=dict(type='data', array=stds, visible=True, color="black"),
            marker_color=["#059669" if d >= 3 else "#3b82f6" for d in doc_counts],
            text=[f"μ={m:.2f}<br>σ={s:.2f}<br>n={d} docs" for m, s, d in zip(means, stds, doc_counts)],
            textposition="outside"
        ))
        fig.update_layout(
            title="Cross-Document Consensus Waterfall<br><sub>Green = strong consensus (≥3 docs), Blue = emerging</sub>",
            yaxis_title="Mean Value", xaxis_tickangle=-45, height=500
        )
        return fig

    def plot_reasoning_chain(self, chain: ReasoningChain, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        G = chain.build_thinking_graph()
        fig, ax = plt.subplots(figsize=figsize)
        pos = nx.multipartite_layout(G, subset_key="layer")
        color_map = {
            "query": "#1e40af", "entity_extraction": "#3b82f6", "vector_retrieval": "#8b5cf6",
            "graph_diffusion": "#a855f7", "claim_analysis": "#f59e0b", "cross_doc_analysis": "#10b981",
            "synthesis": "#ec4899", "answer": "#059669", "entity": "#60a5fa", "chunk": "#c084fc"
        }
        node_colors = [color_map.get(G.nodes[n].get("node_type", "query"), "#6b7280") for n in G.nodes()]
        node_sizes = [1200 if G.nodes[n].get("node_type") in ["query", "answer"] else 600 for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=15, alpha=0.5, ax=ax,
                               connectionstyle="arc3,rad=0.1", edge_color="#4b5563")
        nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
        ax.set_title("Explicit Reasoning Chain (Thinking Graph)", fontsize=13, fontweight='bold')
        ax.axis("off")
        plt.tight_layout()
        return fig

    def plot_entity_tsne(self, embedding_fn: Callable, top_n: int = 80) -> Optional[plt.Figure]:
        if not SKLEARN_AVAILABLE:
            return None
        ent_counts = Counter({k: len(v) for k, v in self.graph.entities.items()})
        top = [e for e, _ in ent_counts.most_common(top_n)]
        if len(top) < 5:
            return None
        embs = []
        domains = []
        for ent in top:
            vec = embedding_fn(ent)
            embs.append(vec)
            domains.append(self.graph.entities[ent][0].domain if self.graph.entities.get(ent) else "UNKNOWN")
        embs = np.stack(embs)
        tsne = TSNE(n_components=2, perplexity=min(30, len(top)-1), random_state=42)
        coords = tsne.fit_transform(embs)
        fig, ax = plt.subplots(figsize=(10, 8))
        for domain in set(domains):
            mask = [d == domain for d in domains]
            x = coords[mask, 0]
            y = coords[mask, 1]
            ax.scatter(x, y, c=self.color_map.get(domain, "gray"), label=domain, alpha=0.8, s=80, edgecolors='white')
        for i, ent in enumerate(top):
            ax.annotate(ent[:20], (coords[i, 0], coords[i, 1]), fontsize=6, alpha=0.8)
        ax.legend(loc='best', fontsize=8)
        ax.set_title("Entity Embedding Space (t-SNE)", fontsize=12, fontweight='bold')
        ax.axis("off")
        plt.tight_layout()
        return fig

    def plot_temporal_timeline(self) -> go.Figure:
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
        fig = px.scatter(df, x="year", y="topic", color="doc", symbol="doc",
                         title="Research Topic Timeline by Document",
                         labels={"year": "Publication Year", "topic": "Topic"},
                         height=500)
        fig.update_traces(marker=dict(size=12))
        return fig

    def plot_entity_treemap(self) -> go.Figure:
        rows = []
        for norm, ents in self.graph.entities.items():
            if not ents:
                continue
            rows.append({
                "domain": ents[0].domain,
                "category": ents[0].category,
                "subcategory": ents[0].subcategory,
                "entity": norm,
                "value": len(ents),
                "docs": len(set(e.doc_source for e in ents))
            })
        df = pd.DataFrame(rows)
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title="No entity data")
            return fig
        fig = px.treemap(df, path=["domain", "category", "subcategory", "entity"],
                         values="value", color="docs", color_continuous_scale="Viridis",
                         title="Hierarchical Entity Treemap")
        return fig


# =============================================
# EMBEDDING WRAPPER
# =============================================

class EmbeddingWrapper:
    def __init__(self, embedding_source):
        self.source = embedding_source

    def __call__(self, text: str) -> np.ndarray:
        if hasattr(self.source, 'embed_query'):
            return np.array(self.source.embed_query(text))
        elif hasattr(self.source, 'embed_documents'):
            return np.array(self.source.embed_documents([text])[0])
        else:
            raise ValueError("Embedding source has no embed_query or embed_documents method")


# =============================================
# SEMANTIC CHUNKING WITH STRUCTURE AWARENESS
# =============================================

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
    all_text = "\n\n".join([p.page_content for p in pages])
    sections = detect_scientific_sections(all_text)
    chunks = []
    for section_name, section_text in sections:
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
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)
    return chunks


# =============================================
# SESSION STATE INITIALIZATION (extended)
# =============================================

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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================
# UTILITY FUNCTIONS
# =============================================

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


# =============================================
# LOCAL MODEL LOADING
# =============================================

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
    """)
    if "0.5B" in repo_id or "1.1B" in repo_id or "gpt2" in repo_id:
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


# =============================================
# DOCUMENT PROCESSING
# =============================================

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


def load_and_chunk_laser_documents(uploaded_files: List) -> Tuple[List[Document], EnhancedCrossDocumentKnowledgeGraph]:
    all_chunks = []
    graph = EnhancedCrossDocumentKnowledgeGraph()

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf" if uploaded_file.name.endswith('.pdf') else ".txt") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            file_hash = compute_file_hash(tmp_path)
            cached_meta = st.session_state.metadata_cache.get(uploaded_file.name, file_hash)

            if cached_meta:
                bib_meta = cached_meta
                st.info(f"📚 Using cached metadata for `{uploaded_file.name}`")
            else:
                if uploaded_file.name.endswith('.pdf'):
                    bib_meta = extract_metadata_from_pdf_file(tmp_path, uploaded_file.name)
                else:
                    with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text_content = f.read()
                    bib_meta = extract_metadata_from_text_file(text_content, uploaded_file.name)
                st.session_state.metadata_cache.set(uploaded_file.name, bib_meta, file_hash)
                st.info(f"📚 Extracted metadata: {bib_meta.format_citation('apa')}")

            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, encoding='utf-8')

            pages = loader.load()
            chunks = semantic_chunk_document(pages, uploaded_file.name)

            for chunk in chunks:
                chunk.metadata.update({
                    **extract_laser_metadata(chunk.page_content, uploaded_file.name),
                    "bibliographic": bib_meta.to_dict(),
                    "citation_display": bib_meta.format_citation(st.session_state.get('citation_style', 'apa')),
                })

            graph.add_document(uploaded_file.name, chunks, bib_meta)
            all_chunks.extend(chunks)
            st.info(f"✅ Loaded {len(chunks)} semantic chunks from `{uploaded_file.name}`")

        except Exception as e:
            st.error(f"❌ Error processing `{uploaded_file.name}`: {e}")
            import traceback
            st.error(traceback.format_exc())
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return all_chunks, graph


@st.cache_resource
def create_local_vector_store(chunks: List[Document], embedding_model_key: str):
    try:
        embeddings = load_local_embeddings()
        if embeddings is None:
            return None
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.metadata = {
            "total_chunks": len(chunks),
            "embedding_model": embedding_model_key,
            "created_at": datetime.now().isoformat(),
            "laser_topics": list(set(
                topic for chunk in chunks for topic in chunk.metadata.get("laser_topics", [])
            ))
        }
        return vectorstore
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None


# =============================================
# RAG FUNCTIONS (Enhanced with Thinking)
# =============================================

def extract_query_entities(query: str) -> List[str]:
    entities = []
    query_lower = query.lower()
    for canonical, aliases in MATERIAL_ALIASES.items():
        if any(alias in query_lower for alias in aliases):
            entities.append(canonical)
    for canonical, aliases in METHOD_ALIASES.items():
        if any(alias in query_lower for alias in aliases):
            entities.append(canonical)
    for param_name in QUANTITY_PATTERNS.keys():
        if param_name.replace("_", " ") in query_lower or param_name in query_lower:
            entities.append(param_name)
    for topic, keywords in LASER_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            entities.append(topic)
    return entities


def create_scientific_reasoning_prompt(
    retrieved_chunks: List[Document],
    query: str,
    graph: EnhancedCrossDocumentKnowledgeGraph,
    consensus_data: List[Dict],
    contradictions: List[Dict]
) -> str:
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        citation = chunk.metadata.get("citation_display")
        if not citation:
            source = chunk.metadata.get("source", "unknown")
            citation = f"[Source {i} - {source}]"
        section = chunk.metadata.get("section", "UNKNOWN")
        content = chunk.page_content[:600] + "..." if len(chunk.page_content) > 600 else chunk.page_content
        context_parts.append(f"---\n[{i}] {citation} | Section: {section}\n{content}\n")
    context = "\n".join(context_parts)

    consensus_text = ""
    if consensus_data:
        consensus_text = "\nCross-Document Consensus (statistical agreement across papers):\n"
        for cons in consensus_data[:3]:
            consensus_text += f"- {cons['entity']}: {cons['mean']:.2f} ± {cons['std']:.2f} {cons['unit']} (across {cons['doc_count']} papers, n={cons['value_count']})\n"

    contradiction_text = ""
    if contradictions:
        contradiction_text = "\nDetected Contradictions Across Documents:\n"
        for contr in contradictions[:3]:
            contradiction_text += f"- {contr['entity']}: {Path(contr['doc_a']).stem} reports {contr['mean_a']:.2f} vs {Path(contr['doc_b']).stem} reports {contr['mean_b']:.2f} (ratio: {contr['ratio']:.1f}x, {contr['severity']})\n"

    system_prompt = """You are an expert scientific research assistant specializing in laser-microstructure interactions, with a focus on multicomponent alloys, additive manufacturing, and physics-informed digital twins.
Your task is to synthesize evidence from multiple research papers and provide a scientifically rigorous answer.

REASONING RULES:
1. SYNTHESIZE across documents — do not just summarize one paper at a time
2. Identify CONSENSUS where multiple papers agree, and CONTRADICTIONS where they disagree
3. Report UNCERTAINTY explicitly — use phrases like "reported values range from X to Y", "the consensus mean is Z ± σ"
4. Cite sources using the EXACT citation format provided (Author et al., Journal, Year)
5. If evidence is insufficient or contradictory, state this explicitly rather than fabricating consensus
6. Distinguish between direct experimental results and inferred/theoretical claims
7. For numerical values, include units and note if papers use different measurement conditions

OUTPUT STRUCTURE:
1. **Direct Answer**: Concise answer to the question
2. **Evidence Synthesis**: Integration of findings across papers with citations
3. **Consensus & Variability**: Statistical summary if multiple papers report the same parameter
4. **Contradictions & Limitations**: Note any conflicting results or methodological differences
5. **Confidence Assessment**: State your confidence (High/Medium/Low) and why
"""
    user_prompt = f"""Retrieved Document Context:
{context}
{consensus_text}
{contradiction_text}

User Question: {query}

Provide a scientifically rigorous answer following the structure above. Be precise about uncertainty and cross-document agreement."""
    return system_prompt + user_prompt


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


def generate_local_response(tokenizer, model_or_tag, device_or_host: str, prompt: str, backend: str, backend_type: str) -> str:
    if backend_type == "ollama":
        return generate_local_response_ollama(model_or_tag, device_or_host, prompt)
    else:
        return generate_local_response_transformers(tokenizer, model_or_tag, device_or_host, prompt, backend)


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


# =============================================
# ORIGINAL VISUALIZATION HELPERS (PyVis)
# =============================================

def build_network_html(graph: EnhancedCrossDocumentKnowledgeGraph) -> str:
    if not PYVIS_AVAILABLE:
        return "<p>Install pyvis to see network graph: pip install pyvis</p>"
    net = Network(height="500px", width="100%", directed=False)
    for doc_id in graph.documents:
        net.add_node(doc_id, label=Path(doc_id).stem, shape="box", color="#97C2FC")
    entity_counts = Counter([e.normalized for ents in graph.entities.values() for e in ents])
    added_entities = set()
    for ent, cnt in entity_counts.most_common(15):
        net.add_node(ent, label=f"{ent} ({cnt})", shape="ellipse", color="#FFA500")
        added_entities.add(ent)
    for ent_norm, doc_names in graph.entity_index.items():
        if ent_norm in added_entities and entity_counts[ent_norm] >= 2:
            for doc in doc_names:
                if doc in graph.documents:
                    net.add_edge(ent_norm, doc)
    return net.generate_html()


def plot_consensus_chart(consensus_data: List[Dict]) -> go.Figure:
    if not consensus_data:
        return None
    entities = [d['entity'] for d in consensus_data]
    means = [d['mean'] for d in consensus_data]
    stds = [d['std'] for d in consensus_data]
    units = [d['unit'] for d in consensus_data]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=entities, y=means, error_y=dict(type='data', array=stds, visible=True),
        name='Consensus value', text=[f"{m:.2f} ± {s:.2f} ({u})" for m, s, u in zip(means, stds, units)],
        hoverinfo='text'
    ))
    fig.update_layout(title="Cross-document consensus (mean ± std)", yaxis_title="Value", xaxis_title="Entity")
    return fig


def render_contradiction_table(contradictions: List[Dict]) -> pd.DataFrame:
    if not contradictions:
        return pd.DataFrame()
    rows = []
    for c in contradictions:
        rows.append({
            "Entity": c['entity'],
            "Doc A": Path(c['doc_a']).stem,
            "Mean A": f"{c['mean_a']:.3f}",
            "Doc B": Path(c['doc_b']).stem,
            "Mean B": f"{c['mean_b']:.3f}",
            "Ratio": f"{c['ratio']:.1f}x",
            "Severity": c['severity']
        })
    return pd.DataFrame(rows)


def extract_equations_from_text(text: str) -> List[str]:
    eq_pattern = re.compile(r'(\$[^$]+\$|\\\[.*?\\\]|([A-Za-z_\{\}]+)\s*=\s*[^\n]+)')
    matches = eq_pattern.findall(text)
    return [m[0].strip() for m in matches if len(m[0]) > 10]


# =============================================
# STREAMLIT UI (Fully Extended)
# =============================================

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
    st.markdown("### 📁 Upload Laser Microstructure Documents")
    uploaded_files = st.file_uploader(
        "Select PDF or TXT files about laser processing, multicomponent alloys, additive manufacturing, etc.",
        type=["pdf", "txt"], accept_multiple_files=True,
        help="Documents will be processed with semantic section detection and cross-document entity linking."
    )
    return uploaded_files


def process_documents(uploaded_files):
    if not uploaded_files:
        return False

    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    if not new_files:
        st.info("✓ All uploaded files already processed")
        return st.session_state.processing_complete

    st.session_state.messages = []
    st.session_state.vectorstore = None
    st.session_state.all_chunks = []
    st.session_state.knowledge_graph = None
    st.session_state.visualization_engine = None

    with st.spinner(f"Processing {len(new_files)} document(s) with semantic reasoning..."):
        try:
            chunks, graph = load_and_chunk_laser_documents(new_files)
            if not chunks:
                st.error("No chunks extracted. Check file format.")
                return False

            for f in new_files:
                st.session_state.processed_files.add(f.name)

            st.session_state.all_chunks.extend(chunks)
            st.session_state.knowledge_graph = graph

            with st.spinner("Creating vector index and knowledge graph..."):
                vectorstore = create_local_vector_store(st.session_state.all_chunks, LOCAL_EMBEDDING_MODEL)
                if vectorstore is None:
                    return False
                st.session_state.vectorstore = vectorstore

            if graph:
                summary = graph.get_knowledge_summary()
                st.success(
                    f"✅ Ready! Indexed {len(st.session_state.all_chunks)} chunks, "
                    f"{summary['unique_entities']} unique entities, "
                    f"{summary['total_claims']} claims from {summary['document_count']} papers"
                )
                if summary['consensus_topics']:
                    st.caption(f"🔗 Cross-document consensus available for: {', '.join(summary['consensus_topics'][:5])}")
            else:
                st.success(f"✅ Ready! Indexed {len(st.session_state.all_chunks)} chunks")

            st.session_state.processing_complete = True
            return True

        except Exception as e:
            st.error(f"Processing failed: {e}")
            import traceback
            st.error(traceback.format_exc())
            return False


def render_chat_interface():
    if not st.session_state.get('vectorstore'):
        st.info("👆 Upload documents above to start chatting with cross-document reasoning")
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
            with st.spinner("Thinking across documents..."):
                answer, retrieved_docs, avg_relevance, reasoning_meta, chain = retrieve_and_answer(
                    st.session_state.vectorstore, st.session_state.knowledge_graph,
                    st.session_state.llm_tokenizer, st.session_state.llm_model,
                    st.session_state.llm_device_or_host, st.session_state.llm_model_choice,
                    st.session_state.llm_backend_type, prompt,
                    k=st.session_state.max_retrieved_chunks
                )
                reasoning_meta['relevance'] = avg_relevance
                st.markdown(answer)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": retrieved_docs,
                    "reasoning_meta": reasoning_meta,
                    "reasoning_chain": chain
                })

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        last_answer = st.session_state.messages[-1]["content"]
        if st.button("📈 Generate plot from answer (LLM will write Python code)", key="gen_plot"):
            with st.spinner("Generating plot code..."):
                plot_prompt = f"Based on the following scientific answer, write a short Python script using matplotlib to create a relevant plot. Only output the Python code, no explanation.\n\nAnswer:\n{last_answer}\n\nPython code:"
                if st.session_state.llm_backend_type == "transformers" and st.session_state.llm_tokenizer:
                    code_prompt = plot_prompt
                    inputs = st.session_state.llm_tokenizer.encode(code_prompt, return_tensors='pt', truncation=True, max_length=1024)
                    if torch.cuda.is_available():
                        inputs = inputs.to('cuda')
                    with torch.no_grad():
                        outputs = st.session_state.llm_model.generate(inputs, max_new_tokens=300, temperature=0.1)
                    raw_code = st.session_state.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    if "Python code:" in raw_code:
                        raw_code = raw_code.split("Python code:")[-1].strip()
                    st.session_state.plot_code = raw_code
                else:
                    st.session_state.plot_code = "# Ollama backend: replace with actual LLM call"
                try:
                    local_vars = {}
                    exec(st.session_state.plot_code, {"plt": __import__('matplotlib.pyplot')}, local_vars)
                    fig = local_vars.get('plt').gcf()
                    st.session_state.last_plot_fig = fig
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Plot execution error: {e}")
                    st.code(st.session_state.plot_code)

    if st.session_state.show_network and st.session_state.knowledge_graph:
        st.markdown("### 🕸️ Cross-Document Knowledge Graph")
        if PYVIS_AVAILABLE:
            html = build_network_html(st.session_state.knowledge_graph)
            st.components.v1.html(html, height=550, scrolling=True)
        else:
            st.warning("Install pyvis to display network graph: `pip install pyvis`")

    if st.session_state.knowledge_graph and st.session_state.selected_entity:
        with st.expander(f"📊 Consensus for '{st.session_state.selected_entity}'", expanded=True):
            cons = st.session_state.knowledge_graph.find_consensus(st.session_state.selected_entity)
            if cons:
                chart = plot_consensus_chart([cons])
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                docs_data = []
                for ent in st.session_state.knowledge_graph.entities.get(st.session_state.selected_entity, []):
                    docs_data.append({"Document": Path(ent.doc_source).stem, "Value": ent.value, "Unit": ent.unit})
                if docs_data:
                    df = pd.DataFrame(docs_data).dropna(subset=["Value"])
                    if not df.empty:
                        st.dataframe(df, use_container_width=True)
            else:
                st.info("No numerical consensus (need multiple documents with quantitative data).")
            contr = st.session_state.knowledge_graph.find_contradictions(st.session_state.selected_entity, threshold_factor=1.5)
            if contr:
                st.markdown("**Contradictions**")
                contr_df = render_contradiction_table(contr)
                if not contr_df.empty:
                    st.dataframe(contr_df.style.applymap(lambda x: 'background-color: #ffcccc' if x == 'high' else '', subset=['Severity']))

    with st.sidebar.expander("🔍 Retrieval Debugger", expanded=False):
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            last_meta = st.session_state.messages[-1].get("reasoning_meta", {})
            last_sources = st.session_state.messages[-1].get("sources", [])
            if last_sources:
                st.markdown("**Retrieved chunks (click to view)**")
                for j, doc in enumerate(last_sources):
                    source = doc.metadata.get("source", "unknown")
                    section = doc.metadata.get("section", "")
                    chunk_id = doc.metadata.get("chunk_index", -1)
                    score = last_meta.get('relevance', 0.0) if j == 0 else 0.0
                    with st.expander(f"Chunk {j+1} ({Path(source).stem}, {section}) – score: {score:.3f}"):
                        st.text(doc.page_content[:500])
                        col_fb1, col_fb2 = st.columns(2)
                        with col_fb1:
                            if st.button("👍 Relevant", key=f"rel_{source}_{chunk_id}"):
                                st.session_state.feedback_map[f"{source}_{chunk_id}"] = 1
                        with col_fb2:
                            if st.button("👎 Not relevant", key=f"nrel_{source}_{chunk_id}"):
                                st.session_state.feedback_map[f"{source}_{chunk_id}"] = 0
        if st.button("Compute Precision/Recall"):
            feedbacks = list(st.session_state.feedback_map.values())
            if feedbacks:
                retrieved = len(feedbacks)
                relevant = sum(feedbacks)
                st.session_state.precision_recall = {
                    "precision": relevant / retrieved if retrieved else 0,
                    "recall": relevant / retrieved if retrieved else 0
                }
        if st.session_state.precision_recall:
            st.metric("Precision (user-rated)", f"{st.session_state.precision_recall['precision']:.2%}")
            st.metric("Recall (session)", f"{st.session_state.precision_recall['recall']:.2%}")


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
        page_title="🔬 Laser Microstructure RAG + Cross-Doc Reasoning",
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

    st.markdown('<h1 class="main-header">🔬 Laser Microstructure RAG + Cross-Doc Reasoning</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:#64748b;margin-bottom:1.5rem">
    Upload research papers on multicomponent alloys and laser processing, and get <strong>scientifically rigorous answers</strong> with 
    <span class="consensus-badge">cross-document consensus</span>, 
    <span class="contradiction-badge">contradiction detection</span>, and 
    <span class="reasoning-badge">multi-hop reasoning</span>.
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
        if uploaded_files and st.button("🔄 Process Documents", type="primary", use_container_width=True):
            process_documents(uploaded_files)

        if st.session_state.processing_complete:
            st.success("✅ Knowledge base ready")
            if st.session_state.knowledge_graph:
                summary = st.session_state.knowledge_graph.get_knowledge_summary()
                st.caption(f"📦 {len(st.session_state.all_chunks)} chunks | {summary['unique_entities']} entities | {summary['total_claims']} claims")
                if summary['top_entities']:
                    st.markdown("**Top entities:**")
                    for ent, count in summary['top_entities'][:5]:
                        st.markdown(f'<span class="reasoning-badge">{ent} ({count})</span>', unsafe_allow_html=True)
        elif uploaded_files:
            st.warning("⏳ Click 'Process Documents' to begin")
        else:
            st.info("📁 Upload PDF/TXT files to start")

        if st.session_state.processed_files:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.clear()
                st.rerun()

    with col2:
        if st.session_state.processing_complete and st.session_state.vectorstore:
            render_chat_interface()
        else:
            st.markdown("""
            <div class="info-card">
            <h3>👋 Welcome to Cross-Document Scientific Reasoning!</h3>
            <p>This assistant goes beyond simple retrieval:</p>
            <ul>
            <li><strong>Semantic Chunking:</strong> Preserves Abstract/Methods/Results/Discussion structure</li>
            <li><strong>Entity Extraction:</strong> Identifies materials, parameters, methods automatically</li>
            <li><strong>Cross-Document Alignment:</strong> Links the same entity across different papers</li>
            <li><strong>Consensus Detection:</strong> Statistically aggregates values reported in multiple papers</li>
            <li><strong>Contradiction Flagging:</strong> Highlights when papers disagree significantly</li>
            <li><strong>Multi-Hop Retrieval:</strong> Follows entity links to find related evidence</li>
            <li><strong>Uncertainty Calibration:</strong> Explicit confidence levels in every answer</li>
            </ul>
            <p><strong>Getting started:</strong></p>
            <ol>
            <li>Upload 2+ PDF/TXT papers on multicomponent alloys or laser processing</li>
            <li>Enable "Cross-document reasoning" in sidebar</li>
            <li>Ask comparative or synthesizing questions</li>
            <li>Expand "🧠 Reasoning Chain" to see the logical steps</li>
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
                    st.session_state.demo_question = q
                    st.rerun()

    # ========================================================================
    # GLOBAL VISUALIZATION DASHBOARD (Enhanced)
    # ========================================================================
    if st.session_state.knowledge_graph and st.session_state.processing_complete:
        st.markdown("---")
        st.markdown("## 🔬 Scientific Visualization Dashboard")

        if st.session_state.visualization_engine is None:
            st.session_state.visualization_engine = VisualizationEngine(st.session_state.knowledge_graph)

        viz = st.session_state.visualization_engine

        if DGL_AVAILABLE and st.session_state.knowledge_graph.dgl_graph is None:
            with st.spinner("Building heterogeneous graph for GNN retrieval..."):
                emb_source = getattr(st.session_state.vectorstore, 'embedding_function',
                                     getattr(st.session_state.vectorstore, 'embeddings', st.session_state.vectorstore))
                emb_fn = EmbeddingWrapper(emb_source)
                st.session_state.knowledge_graph.build_dgl_heterograph(embedding_fn=emb_fn)
                if st.session_state.knowledge_graph.dgl_graph:
                    st.success("✅ DGL HeteroGraph built")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Top Entities & Consensus",
            "🕸️ Knowledge Graphs",
            "☀️ Hierarchical Sunbursts",
            "📡 Document Profiles",
            "⚡ Contradictions"
        ])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                summary = st.session_state.knowledge_graph.get_knowledge_summary()
                if summary['top_entities']:
                    fig = px.bar(
                        x=[x[0] for x in summary['top_entities']],
                        y=[x[1] for x in summary['top_entities']],
                        labels={'x': 'Entity', 'y': 'Occurrences'},
                        title="Top Entities by Frequency",
                        color=[x[1] for x in summary['top_entities']],
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.plotly_chart(viz.plot_consensus_waterfall(top_n=10), use_container_width=True)
            st.plotly_chart(viz.plot_entity_treemap(), use_container_width=True)

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Static Bipartite Network (Publication Quality)**")
                fig_net = viz.plot_static_knowledge_network(top_n=30, layout="spring")
                st.pyplot(fig_net)
            with c2:
                st.markdown("**Chord-Style Co-occurrence**")
                st.plotly_chart(viz.plot_chord_cooccurrence(top_n=16), use_container_width=True)
            if SKLEARN_AVAILABLE:
                st.markdown("**Entity Embedding Space (t-SNE)**")
                emb_source = getattr(st.session_state.vectorstore, 'embedding_function',
                                     getattr(st.session_state.vectorstore, 'embeddings', st.session_state.vectorstore))
                fig_tsne = viz.plot_entity_tsne(embedding_fn=EmbeddingWrapper(emb_source), top_n=60)
                if fig_tsne:
                    st.pyplot(fig_tsne)

        with tab3:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.plotly_chart(viz.plot_methods_sunburst(), use_container_width=True)
            with c2:
                st.plotly_chart(viz.plot_materials_sunburst(), use_container_width=True)
            with c3:
                st.plotly_chart(viz.plot_topics_sunburst(), use_container_width=True)

        with tab4:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(viz.plot_document_radar(), use_container_width=True)
            with c2:
                st.plotly_chart(viz.plot_temporal_timeline(), use_container_width=True)

        with tab5:
            st.markdown("**Cross-Document Contradiction Matrix**")
            st.plotly_chart(viz.plot_contradiction_matrix(), use_container_width=True)
            contrs = st.session_state.knowledge_graph.find_all_contradictions(threshold_factor=1.5)
            if contrs:
                df_contr = pd.DataFrame(contrs)
                st.dataframe(df_contr, use_container_width=True)

    render_footer()

    if hasattr(st.session_state, 'demo_question') and st.session_state.demo_question:
        st.session_state.messages.append({"role": "user", "content": st.session_state.demo_question})
        del st.session_state.demo_question
        st.rerun()


if __name__ == "__main__":
    main()
