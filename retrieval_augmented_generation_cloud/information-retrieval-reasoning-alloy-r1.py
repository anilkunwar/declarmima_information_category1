#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ALLOY MICROSTRUCTURE RAG CHATBOT - FUSION & VISUALIZATION EDITION
========================================================================================
✅ Zero API keys - all models run locally (HuggingFace Transformers + Ollama)
✅ Alloy-focused: FCC/BCC/HCP/LIQUID phases, compositions, multicomponent systems
✅ Information fusion: Cross-document property extraction for microstructure ONLY
✅ Chat-driven visualizations: Bar charts, Radar charts, Pie charts with customization
✅ Physics-aware validation: Thermodynamic bounds, phase rule consistency
✅ Human-readable citations: DOI, Author-Year-Journal format
✅ FAISS vector storage with alloy-aware semantic chunking
✅ Memory-efficient: 4-bit quantization, CPU/GPU auto-detection
✅ DECLARMIMA-aligned: Multicomponent alloys, laser-microstructure interaction domain

DEPLOYMENT:
pip install streamlit langchain langchain-community faiss-cpu sentence-transformers
pip install transformers torch plotly pandas numpy scikit-learn
pip install pypdf2 pdf2doi crossrefapi  # optional for enhanced metadata
pip install ollama  # optional for Ollama backend

Run: streamlit run alloy_rag_viz.py
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

# Optional libraries with graceful fallbacks
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
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# =============================================
# ALLOY-SPECIFIC CONFIGURATION
# =============================================

ALLOY_STRUCTURE_TYPES = {
    "FCC": ["fcc", "face-centered cubic", "austenitic", "γ-phase", "gamma phase", "a1"],
    "BCC": ["bcc", "body-centered cubic", "ferritic", "α-phase", "alpha phase", "a2"],
    "HCP": ["hcp", "hexagonal close-packed", "ε-phase", "epsilon phase", "martensitic", "a3"],
    "LIQUID": ["liquid", "molten", "melt pool", "liquidus", "melt", "liquid phase"],
    "SOLID": ["solid", "solidus", "solid solution", "solid phase"],
    "AMORPHOUS": ["amorphous", "glass", "glassy", "metallic glass"],
    "INTERMETALLIC": ["intermetallic", "IMC", "compound phase", "sigma phase", "laves phase", "mu phase"],
    "LAVES": ["laves", "laves phase", "c14", "c15", "c36"],
    "SIGMA": ["sigma", "sigma phase", "σ-phase"],
}

ALLOY_COMPOSITION_PATTERNS = {
    "binary": re.compile(r'([A-Z][a-z]?)\s*[-–]?\s*([A-Z][a-z]?)(?:\s+alloy)?', re.I),
    "ternary": re.compile(r'([A-Z][a-z]?)\s*[-–]?\s*([A-Z][a-z]?)\s*[-–]?\s*([A-Z][a-z]?)(?:\s+alloy)?', re.I),
    "quaternary": re.compile(r'([A-Z][a-z]?)\s*[-–]?\s*([A-Z][a-z]?)\s*[-–]?\s*([A-Z][a-z]?)\s*[-–]?\s*([A-Z][a-z]?)(?:\s+alloy)?', re.I),
    "atomic_percent": re.compile(r'(\d+(?:\.\d+)?)\s*(?:at\.%|at%)\s*([A-Z][a-z]?)', re.I),
    "weight_percent": re.compile(r'(\d+(?:\.\d+)?)\s*(?:wt\.%|wt%)\s*([A-Z][a-z]?)', re.I),
    "nominal_composition": re.compile(r'([A-Z][a-z]?\d*(?:[-–][A-Z][a-z]?\d*)+)', re.I),
    "element_list": re.compile(r'([A-Z][a-z]?)(?:\s*[,-]\s*([A-Z][a-z]?))+', re.I),
}

MICROSTRUCTURE_FEATURES = {
    "grain_size": re.compile(r'(\d+(?:\.\d+)?)\s*(?:µm|um|nm)\s*(?:grain\s*size|average\s*grain|d50|mean\s*grain)', re.I),
    "phase_fraction": re.compile(r'(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:phase\s*fraction|volume\s*fraction|([A-Z]+)\s*phase)', re.I),
    "dendrite_arm": re.compile(r'(\d+(?:\.\d+)?)\s*(?:µm|um)\s*(?:dendrite\s*arm\s*spacing|DAS|secondary\s*arm|primary\s*arm)', re.I),
    "precipitate_size": re.compile(r'(\d+(?:\.\d+)?)\s*(?:nm|µm|um)\s*(?:precipitate|particle|carbide|nitride|intermetallic)', re.I),
    "layer_thickness": re.compile(r'(\d+(?:\.\d+)?)\s*(?:µm|um)\s*(?:layer\s*thickness|deposition\s*layer|coating\s*thickness)', re.I),
    "pore_size": re.compile(r'(\d+(?:\.\d+)?)\s*(?:µm|um)\s*(?:pore|porosity|void|defect)', re.I),
    "columnar_fraction": re.compile(r'(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:columnar|equiaxed)\s*(?:grain|structure)', re.I),
}

ALLOY_SYSTEMS = {
    "Fe-based": ["steel", "iron", "fe", "stainless", "martensitic", "ferritic", "austenitic", "duplex"],
    "Ni-based": ["nickel", "ni", "inconel", "hastelloy", "monel", "nimonic", "waspoloy"],
    "Al-based": ["aluminum", "aluminium", "al", "alsi", "aluminum alloy", "al-si", "al-cu", "al-mg"],
    "Ti-based": ["titanium", "ti", "ti6al4v", "ti-6al-4v", "titanium alloy", "ti-al", "ti-v"],
    "Co-based": ["cobalt", "co", "stellite", "cobalt-chrome", "cocr", "co-cr"],
    "Cu-based": ["copper", "cu", "bronze", "brass", "copper alloy", "cu-zn", "cu-sn"],
    "HEA": ["high entropy", "hea", "multi-principal", "mpea", "cocrfeni", "alcocrfeni", "refractory hea"],
    "superalloy": ["superalloy", "nickel superalloy", "cobalt superalloy", "directionally solidified"],
    "Sn-Ag-Cu": ["snagcu", "sac", "sn-ag-cu", "solder", "lead-free solder", "tin-silver-copper"],
    "Al-Si": ["alsi", "al-si", "alsi10mg", "al-si10-mg", "hypoeutectic", "hypereutectic"],
}

# Physical property bounds for validation
ALLOY_PROPERTY_BOUNDS = {
    "grain_size_um": {"min": 0.01, "max": 10000, "unit": "µm"},
    "phase_fraction_pct": {"min": 0, "max": 100, "unit": "%"},
    "hardness_hv": {"min": 0, "max": 2000, "unit": "HV"},
    "yield_strength_mpa": {"min": 0, "max": 3000, "unit": "MPa"},
    "elastic_modulus_gpa": {"min": 10, "max": 500, "unit": "GPa"},
    "thermal_conductivity_wmk": {"min": 1, "max": 500, "unit": "W/(m·K)"},
    "diffusion_coeff_m2s": {"min": 1e-20, "max": 1e-6, "unit": "m²/s"},
    "melting_point_k": {"min": 300, "max": 4000, "unit": "K"},
    "density_gcc": {"min": 1, "max": 25, "unit": "g/cm³"},
}

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
    "[Ollama] qwen2.5:14b (via ollama serve) 🔥": "ollama:qwen2.5:14b",
    "[Ollama] llama3.1:8b (via ollama serve)": "ollama:llama3.1:8b",
    "[Ollama] mistral:7b (via ollama serve)": "ollama:mistral:7b",
    "[Ollama] gemma2:9b (via ollama serve)": "ollama:gemma2:9b",
    "[Ollama] falcon3:10b (via ollama serve)": "ollama:falcon3:10b",
}

LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ALLOY_DOMAIN_CONFIG = {
    "chunk_size": 800,
    "chunk_overlap": 150,
    "retrieval_k": 4,
    "score_threshold": 0.25,
    "max_context_tokens": 1024,
    "max_new_tokens": 256,
    "temperature": 0.1,
}

# Alloy-specific keywords for retrieval boosting
ALLOY_KEYWORDS = {
    "composition": ["at.%", "wt.%", "atomic percent", "weight percent", "nominal composition", "alloying element"],
    "phase": ["FCC", "BCC", "HCP", "liquid", "solid", "intermetallic", "phase fraction", "phase diagram"],
    "microstructure": ["grain", "dendrite", "precipitate", "columnar", "equiaxed", "morphology", "texture"],
    "mechanical": ["hardness", "yield strength", "tensile strength", "elongation", "ductility", "modulus"],
    "thermal": ["melting point", "solidus", "liquidus", "thermal conductivity", "diffusion coefficient"],
    "processing": ["solidification", "cooling rate", "heat treatment", "aging", "annealing", "quenching"],
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
# DATA STRUCTURES & ENUMS
# =============================================

class FusionConfidence(Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNKNOWN = "unknown"

@dataclass
class AlloyProperty:
    """Alloy-specific property with microstructure context"""
    name: str
    value: Union[float, str, List]
    unit: Optional[str] = None
    uncertainty: Optional[str] = None
    condition: Optional[str] = None
    source_chunk_id: str = ""
    source_citation: str = ""
    extraction_confidence: float = 0.5
    context_snippet: str = ""
    property_type: str = "microstructure"  # microstructure, mechanical, thermal, composition
    alloy_system: Optional[str] = None
    phase: Optional[str] = None
    composition: Dict[str, float] = field(default_factory=dict)
    normalized_name: str = ""
    normalized_value: Optional[float] = None
    normalized_unit: Optional[str] = None
    
    def __post_init__(self):
        if not self.normalized_name:
            self.normalized_name = self._normalize_property_name(self.name)
        if self.normalized_value is None and isinstance(self.value, (int, float)):
            self.normalized_value = self.value
    
    def _normalize_property_name(self, name: str) -> str:
        """Normalize property names for consistent fusion"""
        synonym_map = {
            "grain size": "grain_size",
            "average grain size": "grain_size",
            "mean grain diameter": "grain_size",
            "d50 grain": "grain_size",
            "phase fraction": "phase_fraction",
            "volume fraction": "phase_fraction",
            "fcc fraction": "phase_fraction_fcc",
            "bcc fraction": "phase_fraction_bcc",
            "hcp fraction": "phase_fraction_hcp",
            "yield strength": "yield_strength",
            "ys": "yield_strength",
            "ultimate tensile strength": "ultimate_tensile_strength",
            "uts": "ultimate_tensile_strength",
            "tensile strength": "ultimate_tensile_strength",
            "elongation": "elongation_at_break",
            "elongation at break": "elongation_at_break",
            "hardness": "hardness",
            "microhardness": "hardness",
            "vickers hardness": "hardness",
            "hv": "hardness",
            "young modulus": "elastic_modulus",
            "elastic modulus": "elastic_modulus",
            "shear modulus": "shear_modulus",
            "bulk modulus": "bulk_modulus",
            "thermal conductivity": "thermal_conductivity",
            "diffusion coefficient": "diffusion_coefficient",
            "melting point": "melting_point",
            "solidus temperature": "solidus_temperature",
            "liquidus temperature": "liquidus_temperature",
            "dendrite arm spacing": "dendrite_arm_spacing",
            "das": "dendrite_arm_spacing",
            "precipitate size": "precipitate_size",
            "pore size": "pore_size",
            "porosity": "porosity_fraction",
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
class AlloyFusionRecord:
    """Record of extracted alloy properties from a document chunk"""
    source_filename: str
    chunk_index: int
    chunk_id: str
    bibliographic_citation: str
    extracted_properties: List[AlloyProperty] = field(default_factory=list)
    alloy_topics: List[str] = field(default_factory=list)
    experimental_conditions: Dict[str, Any] = field(default_factory=dict)
    alloy_system: Optional[str] = None
    processing_method: Optional[str] = None
    phases_detected: List[str] = field(default_factory=list)
    
    def add_property(self, prop: AlloyProperty):
        self.extracted_properties.append(prop)
    
    def get_properties_by_name(self, prop_name: str) -> List[AlloyProperty]:
        normalized = AlloyProperty("", "")._normalize_property_name(prop_name)
        return [p for p in self.extracted_properties if p.normalized_name == normalized]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "citation": self.bibliographic_citation,
            "alloy_system": self.alloy_system,
            "phases": self.phases_detected,
            "properties": [p.to_dict() for p in self.extracted_properties],
            "topics": self.alloy_topics,
            "conditions": self.experimental_conditions
        }

@dataclass
class FusedAlloyProperty:
    """Fused property entry with statistical aggregation"""
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
    alloy_system: Optional[str] = None
    phase: Optional[str] = None
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
            "alloy_system": self.alloy_system,
            "phase": self.phase,
            "conditions": self.conditions_summary
        }

@dataclass
class FusionEfficiencyMetrics:
    """Metrics for evaluating fusion quality"""
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
        weights = {
            "source_diversity": 0.15,
            "property_coverage": 0.20,
            "consistency": 0.25,
            "precision": 0.15,
            "confidence": 0.15,
            "specificity": 0.10
        }
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
# BIBLIOGRAPHIC METADATA EXTRACTION
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
            context = text_sample[max(0, year_pos-50):year_pos+50].lower()
            if any(kw in context for kw in ['published', 'received', 'accepted', 'copyright', '©']):
                meta.year = year
                meta.confidence = max(meta.confidence, 0.7)
                break
    
    for pattern in BibliographicMetadata.JOURNAL_PATTERNS:
        journal_match = pattern.search(text_sample)
        if journal_match:
            journal = journal_match.group(1).strip()
            if len(journal) > 10 and not any(bad in journal.lower() for bad in ['introduction', 'abstract', 'references']):
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
                    meta.authors = [f"{a.get('family', '')} {a.get('given', '')}".strip() for a in msg['author']]
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
# ALLOY PROPERTY EXTRACTION ENGINE
# =============================================

class AlloyPropertyExtractor:
    """Extract alloy-specific properties focused on microstructure"""
    
    UNIT_CONVERSIONS = {
        "nm": {"factor": 1e-9, "base": "m"},
        "μm": {"factor": 1e-6, "base": "m"},
        "um": {"factor": 1e-6, "base": "m"},
        "mm": {"factor": 1e-3, "base": "m"},
        "MPa": {"factor": 1e6, "base": "Pa"},
        "GPa": {"factor": 1e9, "base": "Pa"},
        "HV": {"factor": 1, "base": "HV"},
        "HRC": {"factor": 1, "base": "HRC"},
        "HB": {"factor": 1, "base": "HB"},
        "W/(m·K)": {"factor": 1, "base": "W/(m·K)"},
        "m²/s": {"factor": 1, "base": "m²/s"},
    }
    
    MATERIAL_SYNONYMS = {
        "si": "silicon", "aluminum alloy": "aluminum", "alsi10mg": "AlSi10Mg",
        "ti-6al-4v": "Ti6Al4V", "titanium alloy": "Ti6Al4V",
        "stainless steel": "steel", "ss316l": "steel",
        "cocrfeni": "CoCrFeNi", "alcocrfeni": "AlCoCrFeNi",
    }
    
    def __init__(self, alloy_keywords: Dict[str, List[str]]):
        self.alloy_keywords = alloy_keywords
        self._compile_extraction_patterns()
    
    def _compile_extraction_patterns(self):
        numeric_pattern = r'([\d.]+(?:\s*[×x*]\s*10\^?-?\d+)?)(?:\s*([±\+-])\s*([\d.]+))?'
        unit_pattern = r'\s*(' + '|'.join(re.escape(u) for u in self.UNIT_CONVERSIONS.keys()) + r')'
        
        self.property_pattern = re.compile(
            r'([\w\s\-_/]+?)\s*(?:is|was|of|at|:|=|≈|~|yields|results in)\s*' + 
            numeric_pattern + unit_pattern +
            r'(?:\s*[\(\[]([^)\]]+)[\)\]])?', re.I)
        
        self.table_row_pattern = re.compile(r'(?:^|\n)\s*[|│]?\s*([^\n|│]+?)\s*[|│]?\s*(?:\n|$)', re.MULTILINE)
        
        material_list = list(self.MATERIAL_SYNONYMS.keys()) + ['silicon', 'steel', 'titanium', 'aluminum', 'alloy', 'cocrfeni', 'alsi10mg']
        self.material_property_pattern = re.compile(
            r'(' + '|'.join(re.escape(m) for m in material_list) + r').{0,200}?' +
            r'([\w\s]+?\s*(?:is|was|of|at|:|=)\s*[\d.]+)', re.I | re.DOTALL)
    
    def extract_properties_from_chunk(self, chunk_text: str, chunk_metadata: Dict[str, Any]) -> AlloyFusionRecord:
        record = AlloyFusionRecord(
            source_filename=chunk_metadata.get('source', 'unknown'),
            chunk_index=chunk_metadata.get('chunk_index', 0),
            chunk_id=f"{chunk_metadata.get('source', 'unknown')}:{chunk_metadata.get('chunk_index', 0)}",
            bibliographic_citation=chunk_metadata.get('citation_display', 'Unknown'),
            alloy_topics=chunk_metadata.get('alloy_topics', []),
            experimental_conditions=chunk_metadata.get('parameters_found', {}),
            alloy_system=self._detect_alloy_system(chunk_text),
            processing_method=self._detect_processing_method(chunk_text),
            phases_detected=self._detect_phases(chunk_text)
        )
        
        # Extract from tables
        table_properties = self._extract_from_tables(chunk_text)
        for prop in table_properties:
            prop.source_chunk_id = record.chunk_id
            prop.source_citation = record.bibliographic_citation
            prop.alloy_system = record.alloy_system
            prop.phase = record.phases_detected[0] if record.phases_detected else None
            record.add_property(prop)
        
        # Extract inline properties
        inline_properties = self._extract_inline_properties(chunk_text)
        for prop in inline_properties:
            if not any(p.normalized_name == prop.normalized_name and 
                      (abs(p.normalized_value - prop.normalized_value) < 1e-6 if p.normalized_value and prop.normalized_value else False)
                      for p in record.extracted_properties):
                prop.source_chunk_id = record.chunk_id
                prop.source_citation = record.bibliographic_citation
                record.add_property(prop)
        
        # Extract composition data
        composition_props = self._extract_composition_data(chunk_text)
        for prop in composition_props:
            prop.source_chunk_id = record.chunk_id
            prop.source_citation = record.bibliographic_citation
            record.add_property(prop)
        
        return record
    
    def _detect_alloy_system(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        for system, keywords in ALLOY_SYSTEMS.items():
            if any(kw in text_lower for kw in keywords):
                return system
        return None
    
    def _detect_processing_method(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        methods = [
            ("additive manufacturing", "AM"), ("3d printing", "AM"),
            ("selective laser melting", "SLM"), ("laser powder bed fusion", "LPBF"),
            ("directed energy deposition", "DED"), ("casting", "casting"),
            ("forging", "forging"), ("rolling", "rolling"),
            ("heat treatment", "heat_treatment"), ("aging", "aging"),
            ("annealing", "annealing"), ("quenching", "quenching"),
            ("solidification", "solidification"), ("rapid solidification", "rapid_solidification"),
        ]
        for pattern, canonical in methods:
            if pattern in text_lower:
                return canonical
        return None
    
    def _detect_phases(self, text: str) -> List[str]:
        phases = []
        text_lower = text.lower()
        for phase_type, keywords in ALLOY_STRUCTURE_TYPES.items():
            if any(kw in text_lower for kw in keywords):
                phases.append(phase_type)
        return phases
    
    def _extract_from_tables(self, text: str) -> List[AlloyProperty]:
        properties = []
        if r'\begin{tabular}' in text or r'\begin{table}' in text:
            properties.extend(self._parse_latex_table(text))
        elif '|' in text and re.search(r'\|\s*[-:]+\s*\|', text):
            properties.extend(self._parse_markdown_table(text))
        elif self._detect_plain_text_table(text):
            properties.extend(self._parse_plain_text_table(text))
        return properties
    
    def _parse_latex_table(self, latex_text: str) -> List[AlloyProperty]:
        properties = []
        table_match = re.search(r'\\begin\{tabular\}.*?\\end\{tabular\}', latex_text, re.DOTALL)
        if not table_match:
            return properties
        
        table_content = table_match.group(0)
        rows = re.split(r'\\\\', table_content)
        header_row, data_rows = None, []
        
        for row in rows:
            cells = re.findall(r'&\s*([^{&}]+?)\s*(?:&|\\\\|$)', row)
            cells = [c.strip().replace(r'\hline', '').replace(r'\cline', '').strip() for c in cells if c.strip()]
            if not cells:
                continue
            if header_row is None:
                header_row = cells
            else:
                data_rows.append(cells)
        
        if not header_row or len(data_rows) == 0:
            return properties
        
        header_map = {h.lower().strip(): i for i, h in enumerate(header_row)}
        property_cols = [i for i, h in enumerate(header_row) if any(kw in h.lower() for kw in 
            ['grain', 'phase', 'fraction', 'strength', 'hardness', 'modulus', 'temperature', 'size', 'spacing', 'diameter'])]
        descriptor_cols = [i for i in range(len(header_row)) if i not in property_cols]
        
        for row in data_rows:
            if len(row) <= max(property_cols, default=-1):
                continue
            row_conditions = {}
            for col_idx in descriptor_cols:
                if col_idx < len(row) and row[col_idx]:
                    cell = row[col_idx].strip()
                    if any(m in cell.lower() for m in ['as-built', 'aged', 'treated', 'annealed', 'quenched']):
                        row_conditions['treatment'] = cell
                    elif any(m in cell.lower() for m in list(self.MATERIAL_SYNONYMS.keys()) + ['fcc', 'bcc', 'hcp', 'liquid']):
                        row_conditions['phase'] = self.MATERIAL_SYNONYMS.get(cell.lower(), cell)
            
            for prop_col in property_cols:
                if prop_col >= len(row) or not row[prop_col].strip():
                    continue
                prop_name = header_row[prop_col].strip()
                prop_value_raw = row[prop_col].strip()
                parsed = self._parse_property_value(prop_value_raw, prop_name)
                if parsed:
                    prop = AlloyProperty(
                        name=prop_name, value=parsed['value'], unit=parsed['unit'],
                        uncertainty=parsed['uncertainty'], condition=self._format_conditions(row_conditions),
                        extraction_confidence=0.85, context_snippet=prop_value_raw, property_type="microstructure"
                    )
                    self._normalize_property_units(prop)
                    properties.append(prop)
        return properties
    
    def _parse_markdown_table(self, text: str) -> List[AlloyProperty]:
        properties = []
        lines = [l.strip() for l in text.split('\n') if '|' in l and l.strip()]
        if len(lines) < 3:
            return properties
        
        headers = [h.strip() for h in lines[0].split('|') if h.strip()]
        data_start = 2 if re.match(r'^[\s|:-]+$', lines[1]) else 1
        
        for line in lines[data_start:]:
            cells = [c.strip() for c in line.split('|') if c.strip()]
            if len(cells) != len(headers):
                continue
            row_data = dict(zip(headers, cells))
            
            for header, value in row_data.items():
                if not value or value == '-':
                    continue
                if any(kw in header.lower() for kw in ['grain', 'phase', 'fraction', 'strength', 'hardness', 'modulus', 'size']):
                    parsed = self._parse_property_value(value, header)
                    if parsed:
                        prop = AlloyProperty(
                            name=header, value=parsed['value'], unit=parsed['unit'],
                            uncertainty=parsed['uncertainty'], condition=row_data.get('Material') or row_data.get('Condition'),
                            extraction_confidence=0.8, context_snippet=value, property_type="microstructure"
                        )
                        self._normalize_property_units(prop)
                        properties.append(prop)
        return properties
    
    def _parse_plain_text_table(self, text: str) -> List[AlloyProperty]:
        properties = []
        lines = [l for l in text.split('\n') if l.strip() and not l.strip().startswith(('#', '//', '%'))]
        for line in lines:
            numeric_tokens = re.findall(r'[\d.]+(?:\s*[×x*]\s*10\^?-?\d+)?(?:\s*[±\+-]\s*[\d.]+)?\s*(?:[a-zA-Z/²³μ]+)?', line)
            if len(numeric_tokens) >= 2:
                tokens = line.split()
                if len(tokens) >= 3:
                    prop_name = ' '.join(tokens[:2]) if len(tokens[0]) < 15 else tokens[0]
                    value_match = re.match(r'([\d.]+)', numeric_tokens[0])
                    if value_match:
                        try:
                            prop = AlloyProperty(
                                name=prop_name, value=float(value_match.group(1)),
                                extraction_confidence=0.6, context_snippet=line[:100], property_type="observation"
                            )
                            properties.append(prop)
                        except (ValueError, TypeError):
                            continue
        return properties
    
    def _extract_inline_properties(self, text: str) -> List[AlloyProperty]:
        properties = []
        for match in self.property_pattern.finditer(text):
            groups = match.groups()
            if len(groups) >= 5 and groups[1]:
                prop_name = groups[0].strip()
                value_str = groups[1].strip()
                uncertainty = f"{groups[2]}{groups[3]}" if groups[2] and groups[3] else None
                unit = groups[4].strip() if groups[4] else None
                condition = groups[5].strip() if len(groups) > 5 and groups[5] else None
                
                numeric_value = self._safe_parse_numeric(value_str)
                prop = AlloyProperty(
                    name=prop_name, value=numeric_value if numeric_value is not None else value_str,
                    unit=unit, uncertainty=uncertainty, condition=condition,
                    extraction_confidence=0.7, context_snippet=match.group(0)[:150],
                    property_type="microstructure" if any(kw in prop_name.lower() for kw in ['grain', 'phase', 'size', 'fraction']) else "parameter"
                )
                self._normalize_property_units(prop)
                properties.append(prop)
        return properties
    
    def _extract_composition_data(self, text: str) -> List[AlloyProperty]:
        properties = []
        # Extract atomic percentages
        for match in ALLOY_COMPOSITION_PATTERNS["atomic_percent"].finditer(text):
            value = float(match.group(1))
            element = match.group(2)
            context = text[max(0, match.start()-100):min(len(text), match.end()+100)]
            prop = AlloyProperty(
                name=f"composition_{element}",
                value=value,
                unit="at.%",
                context_snippet=context,
                extraction_confidence=0.9,
                property_type="composition"
            )
            self._normalize_property_units(prop)
            properties.append(prop)
        # Extract weight percentages
        for match in ALLOY_COMPOSITION_PATTERNS["weight_percent"].finditer(text):
            value = float(match.group(1))
            element = match.group(2)
            context = text[max(0, match.start()-100):min(len(text), match.end()+100)]
            prop = AlloyProperty(
                name=f"composition_{element}",
                value=value,
                unit="wt.%",
                context_snippet=context,
                extraction_confidence=0.9,
                property_type="composition"
            )
            self._normalize_property_units(prop)
            properties.append(prop)
        return properties
    
    def _parse_property_value(self, raw_value: str, prop_name: str) -> Optional[Dict[str, Any]]:
        if not raw_value or raw_value.strip() in ['-', '.', '', 'N/A', 'n/a', 'NA', 'na', '--', '...']:
            return None
        result = {"value": None, "unit": None, "uncertainty": None}
        
        # Extract uncertainty (± notation)
        uncertainty_match = re.search(r'([±\+-])\s*([\d.]+)', raw_value)
        if uncertainty_match:
            result["uncertainty"] = f"{uncertainty_match.group(1)}{uncertainty_match.group(2)}"
            raw_value = raw_value.replace(uncertainty_match.group(0), '').strip()
        
        # Extract unit (at end of string)
        for unit in sorted(self.UNIT_CONVERSIONS.keys(), key=len, reverse=True):
            if raw_value.lower().endswith(unit.lower()):
                result["unit"] = unit
                raw_value = raw_value[:-len(unit)].strip()
                break
        
        # Safe numeric value extraction
        numeric_value = self._safe_parse_numeric(raw_value)
        if numeric_value is not None:
            result["value"] = numeric_value
        else:
            result["value"] = raw_value.strip() if raw_value.strip() else None
        
        return result if result["value"] is not None else None
    
    def _safe_parse_numeric(self, value_str: str) -> Optional[float]:
        if not value_str:
            return None
        cleaned = value_str.strip()
        if cleaned in ['.', '-', '--', '...', 'N/A', 'n/a', 'NA', 'na', 'null', 'None', '']:
            return None
        if '-' in cleaned and not cleaned.startswith('-'):
            parts = cleaned.split('-')
            if len(parts) == 2:
                cleaned = parts[0].strip()
        cleaned = re.sub(r'\s*[×x*]\s*10\^?', 'e', cleaned)
        cleaned = re.sub(r'\s*[×x*]\s*10', 'e', cleaned)
        match = re.match(r'^\s*([+-]?\s*[\d.]+(?:e[+-]?\d+)?)', cleaned, re.I)
        if not match:
            return None
        num_str = match.group(1).replace(' ', '')
        if num_str in ['.', '+.', '-.']:
            return None
        try:
            return float(num_str)
        except (ValueError, TypeError, OverflowError):
            return None
    
    def _normalize_property_units(self, prop: AlloyProperty):
        if not prop.unit or prop.unit not in self.UNIT_CONVERSIONS:
            prop.normalized_unit = prop.unit
            if isinstance(prop.value, (int, float)):
                prop.normalized_value = prop.value
            elif prop.normalized_value is None and isinstance(prop.value, str):
                prop.normalized_value = self._safe_parse_numeric(prop.value)
            return
        conversion = self.UNIT_CONVERSIONS[prop.unit]
        if isinstance(prop.value, (int, float)):
            prop.normalized_value = prop.value * conversion["factor"]
            prop.normalized_unit = conversion["base"]
        elif prop.normalized_value is not None:
            prop.normalized_value = prop.normalized_value * conversion["factor"]
            prop.normalized_unit = conversion["base"]
        else:
            prop.normalized_unit = prop.unit
    
    def _format_conditions(self, conditions: Dict[str, str]) -> Optional[str]:
        if not conditions:
            return None
        parts = [f"{k}: {v}" for k, v in conditions.items() if v]
        return "; ".join(parts) if parts else None
    
    def _detect_plain_text_table(self, text: str) -> bool:
        lines = [l for l in text.split('\n') if l.strip()]
        if len(lines) < 3:
            return False
        first_line = lines[0].strip()
        first_line_cols = len(first_line.split()) if first_line else 0
        return first_line_cols >= 3 and all(len(l.split()) >= first_line_cols - 1 for l in lines[1:4])

# =============================================
# ALLOY INFORMATION FUSION ENGINE
# =============================================

class AlloyFusionEngine:
    """Fuse alloy microstructure properties across documents"""
    
    def __init__(self, property_extractor: AlloyPropertyExtractor):
        self.extractor = property_extractor
        self.fusion_history: List[Dict] = []
    
    def fuse_alloy_documents(self, retrieved_docs: List[Document], query: str,
                           alloy_filter: Optional[str] = None,
                           phase_filter: Optional[List[str]] = None,
                           property_filter: Optional[List[str]] = None) -> Tuple[Dict[str, FusedAlloyProperty], FusionEfficiencyMetrics]:
        
        fusion_records: List[AlloyFusionRecord] = []
        
        for doc in retrieved_docs:
            record = self.extractor.extract_properties_from_chunk(doc.page_content, doc.metadata)
            
            # Apply filters
            if alloy_filter and record.alloy_system != alloy_filter:
                continue
            if phase_filter and not any(p in phase_filter for p in record.phases_detected):
                continue
            if property_filter:
                record.extracted_properties = [p for p in record.extracted_properties if p.normalized_name in property_filter]
            
            if record.extracted_properties:
                fusion_records.append(record)
        
        if not fusion_records:
            metrics = FusionEfficiencyMetrics(
                unique_sources_used=len(retrieved_docs),
                source_diversity_score=min(1.0, len(retrieved_docs) / 3.0),
                overall_fusion_efficiency=min(1.0, len(retrieved_docs) / 3.0) * 0.3
            )
            return {}, metrics
        
        # Group properties by normalized name
        property_groups: Dict[str, List[AlloyProperty]] = defaultdict(list)
        for record in fusion_records:
            for prop in record.extracted_properties:
                key = prop.normalized_name
                if not property_filter or key in property_filter:
                    property_groups[key].append(prop)
        
        # Fuse each property group
        fused_properties: Dict[str, FusedAlloyProperty] = {}
        for prop_name, props in property_groups.items():
            fused = self._fuse_alloy_property_group(prop_name, props)
            if fused:
                fused_properties[prop_name] = fused
        
        # Compute fusion metrics
        metrics = self._compute_fusion_metrics(fusion_records, fused_properties, retrieved_docs, query)
        
        self.fusion_history.append({
            "timestamp": datetime.now().isoformat(), "query": query,
            "input_docs": len(retrieved_docs),
            "extracted_properties": sum(len(r.extracted_properties) for r in fusion_records),
            "fused_properties": len(fused_properties),
            "efficiency": metrics.overall_fusion_efficiency
        })
        
        return fused_properties, metrics
    
    def _fuse_alloy_property_group(self, prop_name: str, properties: List[AlloyProperty]) -> Optional[FusedAlloyProperty]:
        if not properties:
            return None
        
        numeric_props = [p for p in properties if p.normalized_value is not None and isinstance(p.normalized_value, (int, float))]
        
        fused = FusedAlloyProperty(
            property_name=prop_name,
            fused_value=None,
            unit=properties[0].normalized_unit if properties[0].normalized_unit else properties[0].unit,
            source_count=len(properties),
            sources=[{"citation": p.source_citation, "chunk_id": p.source_chunk_id} for p in properties],
            alloy_system=properties[0].alloy_system if properties else None,
            phase=properties[0].phase if properties else None
        )
        
        if numeric_props and len(numeric_props) >= 1:
            values = [p.normalized_value for p in numeric_props if p.normalized_value is not None]
            if values:
                fused.fused_value = np.mean(values)
                fused.value_range = (min(values), max(values))
                fused.standard_deviation = np.std(values) if len(values) > 1 else 0.0
                
                # Compute coefficient of variation for confidence
                if fused.fused_value != 0:
                    cv = fused.standard_deviation / abs(fused.fused_value)
                else:
                    cv = 1.0
                
                if cv < 0.1 and len(numeric_props) >= 2:
                    fused.fusion_confidence = FusionConfidence.HIGH
                elif cv < 0.3 or len(numeric_props) == 1:
                    fused.fusion_confidence = FusionConfidence.MODERATE
                else:
                    fused.fusion_confidence = FusionConfidence.LOW
                    fused.conflicts_detected = True
                    fused.conflict_notes.append(f"High variation: CV={cv:.2f}")
                
                # Aggregate conditions
                conditions = defaultdict(set)
                for p in numeric_props:
                    if p.condition:
                        conditions["context"].add(p.condition)
                    if p.experimental_conditions:
                        for k, v in p.experimental_conditions.items():
                            conditions[k].add(str(v))
                fused.conditions_summary = {k: list(v) for k, v in conditions.items()}
        else:
            # Non-numeric: use most common value
            value_counts = Counter(str(p.value) for p in properties if p.value is not None)
            if value_counts:
                fused.fused_value = value_counts.most_common(1)[0][0]
                fused.fusion_confidence = (
                    FusionConfidence.HIGH if value_counts.most_common(1)[0][1] == len(properties)
                    else FusionConfidence.MODERATE if value_counts.most_common(1)[0][1] > len(properties) / 2
                    else FusionConfidence.LOW
                )
                if fused.fusion_confidence == FusionConfidence.LOW:
                    fused.conflicts_detected = True
                    fused.conflict_notes.append(f"Multiple distinct values: {list(value_counts.keys())[:3]}")
        
        return fused
    
    def _compute_fusion_metrics(self, fusion_records: List[AlloyFusionRecord],
                               fused_properties: Dict[str, FusedAlloyProperty],
                               retrieved_docs: List[Document], query: str) -> FusionEfficiencyMetrics:
        metrics = FusionEfficiencyMetrics()
        
        unique_sources = set(r.chunk_id for r in fusion_records)
        metrics.unique_sources_used = len(unique_sources)
        metrics.source_diversity_score = min(1.0, len(unique_sources) / 3.0)
        
        total_extracted = sum(len(r.extracted_properties) for r in fusion_records)
        metrics.total_properties_extracted = total_extracted
        metrics.properties_fused_successfully = len(fused_properties)
        metrics.property_coverage_ratio = len(fused_properties) / total_extracted if total_extracted > 0 else 0.0
        
        if fused_properties:
            consistent = sum(1 for f in fused_properties.values() if not f.conflicts_detected and f.fusion_confidence != FusionConfidence.LOW)
            conflicting = sum(1 for f in fused_properties.values() if f.conflicts_detected)
            total_evaluated = consistent + conflicting
            metrics.consistent_properties = consistent
            metrics.conflicting_properties = conflicting
            metrics.consistency_ratio = consistent / total_evaluated if total_evaluated > 0 else 1.0
        else:
            metrics.consistency_ratio = 1.0
        
        # Uncertainty metrics
        numeric_with_uncertainty = [f for f in fused_properties.values() if f.standard_deviation is not None]
        metrics.numeric_properties_with_uncertainty = len(numeric_with_uncertainty)
        
        if fused_properties:
            uncertainties = []
            for f in fused_properties.values():
                if isinstance(f.fused_value, (int, float)) and f.fused_value != 0 and f.standard_deviation is not None:
                    uncertainties.append(f.standard_deviation / abs(f.fused_value))
            if uncertainties:
                metrics.average_uncertainty_magnitude = np.mean(uncertainties)
            else:
                metrics.average_uncertainty_magnitude = 0.1
        else:
            metrics.average_uncertainty_magnitude = 0.1
        
        # Confidence weighting
        confidence_weights = {FusionConfidence.HIGH: 1.0, FusionConfidence.MODERATE: 0.7, FusionConfidence.LOW: 0.4, FusionConfidence.UNKNOWN: 0.2}
        if fused_properties:
            weighted_sum = sum(confidence_weights.get(f.fusion_confidence, 0.5) for f in fused_properties.values())
            metrics.weighted_confidence_score = weighted_sum / len(fused_properties)
            metrics.high_confidence_fusions = sum(1 for f in fused_properties.values() if f.fusion_confidence == FusionConfidence.HIGH)
            metrics.low_confidence_fusions = sum(1 for f in fused_properties.values() if f.fusion_confidence == FusionConfidence.LOW)
        else:
            metrics.weighted_confidence_score = 0.5
        
        # Specificity estimation
        metrics.answer_specificity_score = self._estimate_alloy_specificity(query, fused_properties)
        metrics.citation_density = min(1.0, len(fused_properties) * 2 / 100)
        
        metrics.compute_overall()
        return metrics
    
    def _estimate_alloy_specificity(self, query: str, fused_props: Dict[str, FusedAlloyProperty]) -> float:
        if not fused_props:
            query_lower = query.lower()
            if any(kw in query_lower for kw in ['compare', 'versus', 'vs', 'difference', 'grain', 'phase', 'fraction', 'strength', 'hardness']):
                return 0.5
            return 0.3
        
        query_lower = query.lower()
        specificity_indicators = 0
        
        for prop_name in fused_props.keys():
            if prop_name.replace('_', ' ') in query_lower or prop_name in query_lower:
                specificity_indicators += 2
        
        if any(mat in query_lower for mat in ['fcc', 'bcc', 'hcp', 'liquid', 'grain', 'phase', 'alloy', 'composition']):
            specificity_indicators += 1
        
        if any(param in query_lower for param in ['grain size', 'phase fraction', 'strength', 'hardness', 'modulus']):
            specificity_indicators += 1
        
        if re.search(r'[\d.]+\s*(?:µm|um|nm|mpa|gpa|hv|%|percent)', query_lower):
            specificity_indicators += 2
        
        return min(1.0, specificity_indicators / 5.0)
    
    def generate_comparison_table(self, fused_properties: Dict[str, FusedAlloyProperty], format: str = "markdown") -> str:
        if not fused_properties:
            return "_No properties available for comparison_"
        
        if format == "markdown":
            return self._generate_markdown_table(fused_properties)
        elif format == "latex":
            return self._generate_latex_table(fused_properties)
        elif format == "html":
            return self._generate_html_table(fused_properties)
        else:
            return self._generate_plain_text_table(fused_properties)
    
    def _generate_markdown_table(self, fused_props: Dict[str, FusedAlloyProperty]) -> str:
        lines = []
        lines.append("| Property | Value | Unit | Range | Sources | Confidence | Alloy/Phase |")
        lines.append("|----------|-------|------|-------|---------|------------|-------------|")
        
        for prop_name, entry in sorted(fused_props.items(), key=lambda x: x[0]):
            if entry.fused_value is not None and isinstance(entry.fused_value, (int, float)):
                value_str = f"{entry.fused_value:.3f}"
                if entry.standard_deviation is not None:
                    value_str = f"{value_str} ± {entry.standard_deviation:.3f}"
            else:
                value_str = str(entry.fused_value) if entry.fused_value is not None else "–"
            
            range_str = f"{entry.value_range[0]:.2f}–{entry.value_range[1]:.2f}" if entry.value_range else "–"
            confidence_icon = {"high": "🟢", "moderate": "🟡", "low": "🔴", "unknown": "⚪"}.get(entry.fusion_confidence.value, "⚪")
            alloy_phase = f"{entry.alloy_system or ''} {entry.phase or ''}".strip() or "–"
            
            lines.append(f"| {prop_name.replace('_', ' ').title()} | {value_str} | {entry.unit or '–'} | {range_str} | {entry.source_count} | {confidence_icon} {entry.fusion_confidence.value} | {alloy_phase} |")
        
        return "\n".join(lines)
    
    def _generate_latex_table(self, fused_props: Dict[str, FusedAlloyProperty]) -> str:
        lines = [r"\begin{tabular}{|l|c|c|c|c|c|l|}", r"\hline",
                r"\textbf{Property} & \textbf{Value} & \textbf{Unit} & \textbf{Range} & \textbf{Sources} & \textbf{Confidence} & \textbf{Alloy/Phase} \\", r"\hline"]
        
        for prop_name, entry in sorted(fused_props.items()):
            if entry.fused_value is not None and isinstance(entry.fused_value, (int, float)):
                value_str = f"{entry.fused_value:.3f}"
                if entry.standard_deviation is not None:
                    value_str = f"{value_str} \\pm {entry.standard_deviation:.3f}"
            else:
                value_str = str(entry.fused_value) if entry.fused_value is not None else "--"
            
            range_str = f"{entry.value_range[0]:.2f}--{entry.value_range[1]:.2f}" if entry.value_range else "--"
            conf_symbol = {"high": "high", "moderate": "mod", "low": "low"}.get(entry.fusion_confidence.value, "?")
            alloy_phase = f"{entry.alloy_system or ''} {entry.phase or ''}".strip() or "--"
            
            lines.append(f"{prop_name.replace('_', r'\_').title()} & {value_str} & {entry.unit or '--'} & {range_str} & {entry.source_count} & {conf_symbol} & {alloy_phase} \\\\")
        
        lines.extend([r"\hline", r"\end{tabular}"])
        return "\n".join(lines)
    
    def _generate_html_table(self, fused_props: Dict[str, FusedAlloyProperty]) -> str:
        lines = ['<table class="fusion-table" style="border-collapse: collapse; width: 100%;">']
        lines.append('<thead><tr style="background: #f0f9ff;">')
        for header in ["Property", "Value", "Unit", "Range", "Sources", "Confidence", "Alloy/Phase"]:
            lines.append(f'<th style="border: 1px solid #ccc; padding: 8px; text-align: left;">{header}</th>')
        lines.append('</tr></thead><tbody>')
        
        for prop_name, entry in fused_props.items():
            if entry.fused_value is not None and isinstance(entry.fused_value, (int, float)):
                value_str = f"{entry.fused_value:.3f}"
                if entry.standard_deviation is not None:
                    value_str = f"{value_str} ± {entry.standard_deviation:.3f}"
            else:
                value_str = str(entry.fused_value) if entry.fused_value is not None else "–"
            
            bg_color = {"high": "#dcfce7", "moderate": "#fef3c7", "low": "#fee2e2"}.get(entry.fusion_confidence.value, "#f1f5f9")
            lines.append(f'<tr style="background: {bg_color};">')
            lines.append(f'<td style="border: 1px solid #ccc; padding: 8px;">{prop_name.replace("_", " ").title()}</table>')
            lines.append(f'<td style="border: 1px solid #ccc; padding: 8px;">{value_str}</td>')
            lines.append(f'<td style="border: 1px solid #ccc; padding: 8px;">{entry.unit or "–"}</td>')
            range_display = f"{entry.value_range[0]:.2f}–{entry.value_range[1]:.2f}" if entry.value_range else "–"
            lines.append(f'<td style="border: 1px solid #ccc; padding: 8px;">{range_display}</td>')
            lines.append(f'<td style="border: 1px solid #ccc; padding: 8px; text-align: center;">{entry.source_count}</td>')
            lines.append(f'<td style="border: 1px solid #ccc; padding: 8px;">{entry.fusion_confidence.value.title()}</td>')
            alloy_phase = f"{entry.alloy_system or ''} {entry.phase or ''}".strip() or "–"
            lines.append(f'<td style="border: 1px solid #ccc; padding: 8px;">{alloy_phase}</td>')
            lines.append('</tr>')
        
        lines.append('</tbody></table>')
        return "\n".join(lines)
    
    def _generate_plain_text_table(self, fused_props: Dict[str, FusedAlloyProperty]) -> str:
        lines = []
        lines.append(f"{'Property':<30} {'Value':<15} {'Unit':<10} {'Confidence':<10} {'Alloy/Phase':<20}")
        lines.append("-" * 90)
        
        for prop_name, entry in fused_props.items():
            if entry.fused_value is not None and isinstance(entry.fused_value, (int, float)):
                value_str = f"{entry.fused_value:.3f}"
            else:
                value_str = str(entry.fused_value) if entry.fused_value is not None else "–"
            
            alloy_phase = f"{entry.alloy_system or ''} {entry.phase or ''}".strip() or "–"
            lines.append(f"{prop_name.replace('_', ' ').title():<30} {value_str:<15} {entry.unit or '–':<10} {entry.fusion_confidence.value:<10} {alloy_phase:<20}")
        
        return "\n".join(lines)

# =============================================
# VISUALIZATION ENGINE FOR ALLOY MICROSTRUCTURE
# =============================================

class AlloyVisualizationEngine:
    """Generate customized visualizations for alloy microstructure data"""
    
    @staticmethod
    def create_composition_pie_chart(composition_data: List[Dict], alloy_name: str, title: str = None) -> go.Figure:
        """Create interactive pie chart for alloy composition"""
        if not composition_data:
            return None
        
        elements = [d['element'] for d in composition_data]
        percentages = [d['percentage'] for d in composition_data]
        
        fig = go.Figure(data=[go.Pie(
            labels=elements,
            values=percentages,
            hole=0.3,
            marker=dict(colors=px.colors.qualitative.Set3),
            textinfo='label+percent',
            insidetextorientation='radial',
            hovertemplate='<b>%{label}</b><br>%{value:.1f}%<extra></extra>'
        )])
        
        fig.update_layout(
            title=title or f"Composition of {alloy_name}",
            annotations=[dict(text=alloy_name, x=0.5, y=0.5, font_size=16, showarrow=False)],
            height=400,
            margin=dict(t=50, b=20, l=20, r=20)
        )
        
        return fig
    
    @staticmethod
    def create_microstructure_radar_chart(microstructure_metrics: Dict[str, float], 
                                         title: str = "Microstructure Profile") -> go.Figure:
        """Create radar chart for microstructure characteristics"""
        if not microstructure_metrics:
            return None
        
        categories = list(microstructure_metrics.keys())
        values = list(microstructure_metrics.values())
        
        # Normalize values to 0-1 scale for radar visualization
        max_val = max(values) if values else 1
        normalized_values = [v/max_val for v in values]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_values + [normalized_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Microstructure Profile',
            line=dict(color='rgb(31, 119, 180)', width=2),
            fillcolor='rgba(31, 119, 180, 0.25)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickformat='.0%',
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=11)
                )
            ),
            title=dict(text=title, x=0.5),
            showlegend=False,
            height=450,
            margin=dict(t=50, b=20, l=20, r=20)
        )
        
        return fig
    
    @staticmethod
    def create_phase_fraction_bar_chart(phase_data: List[Dict], 
                                       title: str = "Phase Fraction Distribution",
                                       stacked: bool = True) -> go.Figure:
        """Create bar chart for phase fractions across samples"""
        if not phase_data:
            return None
        
        df = pd.DataFrame(phase_data)
        
        if stacked:
            fig = px.bar(
                df, 
                x='sample', 
                y='fraction', 
                color='phase',
                title=title,
                labels={'fraction': 'Volume Fraction (%)', 'sample': 'Sample', 'phase': 'Phase'},
                color_discrete_sequence=px.colors.qualitative.Pastel,
                text='fraction'
            )
            fig.update_layout(barmode='stack')
        else:
            fig = px.bar(
                df,
                x='sample',
                y='fraction',
                color='phase',
                barmode='group',
                title=title,
                labels={'fraction': 'Volume Fraction (%)', 'sample': 'Sample', 'phase': 'Phase'},
                color_discrete_sequence=px.colors.qualitative.Bold
            )
        
        fig.update_layout(
            height=400,
            margin=dict(t=50, b=40, l=40, r=20),
            xaxis_title="Sample",
            yaxis_title="Volume Fraction (%)"
        )
        
        return fig
    
    @staticmethod
    def create_alloy_property_comparison_chart(alloy_properties: List[Dict], 
                                               property_name: str,
                                               title: str = None) -> go.Figure:
        """Create comparison chart for alloy properties"""
        if not alloy_properties:
            return None
        
        df = pd.DataFrame(alloy_properties)
        
        fig = px.bar(
            df,
            x='alloy_name',
            y=property_name,
            color='alloy_system',
            title=title or f"{property_name.replace('_', ' ').title()} Across Alloys",
            labels={'alloy_name': 'Alloy', property_name: property_name.replace('_', ' ').title()},
            error_y='uncertainty' if 'uncertainty' in df.columns else None,
            color_discrete_sequence=px.colors.qualitative.Bold,
            hover_data=['phase', 'condition'] if 'phase' in df.columns else None
        )
        
        fig.update_layout(
            xaxis_title="Alloy Composition",
            yaxis_title=property_name.replace('_', ' ').title(),
            showlegend=True,
            height=400,
            margin=dict(t=50, b=60, l=40, r=20)
        )
        
        return fig
    
    @staticmethod
    def create_grain_size_distribution(grain_data: List[Dict], 
                                      title: str = "Grain Size Distribution") -> go.Figure:
        """Create histogram/box plot for grain size distribution"""
        if not grain_data:
            return None
        
        df = pd.DataFrame(grain_data)
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Distribution", "By Alloy System"))
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=df['grain_size'], name='Grain Size', marker_color='rgb(31, 119, 180)', nbinsx=20),
            row=1, col=1
        )
        
        # Box plot by alloy system
        if 'alloy_system' in df.columns:
            for system in df['alloy_system'].unique():
                system_data = df[df['alloy_system'] == system]['grain_size']
                fig.add_trace(
                    go.Box(y=system_data, name=system),
                    row=1, col=2
                )
        
        fig.update_layout(
            title=title,
            height=400,
            margin=dict(t=50, b=40, l=40, r=20),
            showlegend=False
        )
        fig.update_xaxes(title_text="Grain Size (µm)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Grain Size (µm)", row=1, col=2)
        
        return fig
    
    @staticmethod
    def create_phase_evolution_chart(phase_data: List[Dict], 
                                     temperature_col: str = 'temperature',
                                     title: str = "Phase Evolution with Temperature") -> go.Figure:
        """Create line chart showing phase fraction evolution with temperature"""
        if not phase_data:
            return None
        
        df = pd.DataFrame(phase_data)
        
        fig = px.line(
            df,
            x=temperature_col,
            y='fraction',
            color='phase',
            title=title,
            labels={temperature_col: 'Temperature (K)', 'fraction': 'Phase Fraction (%)', 'phase': 'Phase'},
            color_discrete_sequence=px.colors.qualitative.Set2,
            markers=True
        )
        
        fig.update_layout(
            height=400,
            margin=dict(t=50, b=40, l=40, r=20),
            xaxis_title="Temperature (K)",
            yaxis_title="Phase Fraction (%)"
        )
        
        return fig

# =============================================
# PHYSICS-AWARE VALIDATION FOR ALLOY PROPERTIES
# =============================================

class AlloyPhysicsValidator:
    """Validate alloy properties against physical constraints"""
    
    def __init__(self):
        self.violation_log: List[Dict] = []
    
    def check_property_bounds(self, property_name: str, value: float, unit: str) -> Dict:
        """Check if a property value is within physical bounds"""
        bounds_key = f"{property_name}_{unit.lower().replace('/', '_').replace('·', '_')}"
        
        if bounds_key not in ALLOY_PROPERTY_BOUNDS:
            # Try to match by property name only
            for key, bounds in ALLOY_PROPERTY_BOUNDS.items():
                if property_name in key:
                    bounds_key = key
                    break
        
        if bounds_key not in ALLOY_PROPERTY_BOUNDS:
            return {"valid": True, "message": "No bounds defined for this property"}
        
        bounds = ALLOY_PROPERTY_BOUNDS[bounds_key]
        
        if value < bounds["min"] or value > bounds["max"]:
            violation = {
                "type": "bound_violation",
                "property": property_name,
                "value": value,
                "unit": unit,
                "bounds": bounds,
                "severity": "HIGH" if value < 0 or value > bounds["max"] * 2 else "MEDIUM"
            }
            self.violation_log.append(violation)
            return {"valid": False, "violation": violation}
        
        return {"valid": True, "message": "Value within physical bounds"}
    
    def check_phase_rule_consistency(self, phases: List[str], components: int, 
                                     conditions: Dict[str, Any]) -> Dict:
        """Check Gibbs phase rule: F = C - P + 2"""
        if not phases or components < 1:
            return {"valid": True, "message": "Insufficient data for phase rule check"}
        
        P = len(phases)  # Number of phases
        C = components    # Number of components
        F = C - P + 2     # Degrees of freedom
        
        # Check if conditions are consistent with degrees of freedom
        fixed_params = len([v for v in conditions.values() if v is not None])
        
        if fixed_params > F and F >= 0:
            return {
                "valid": False,
                "message": f"Phase rule violation: F={F} but {fixed_params} conditions fixed",
                "phase_rule": f"F = C - P + 2 = {C} - {P} + 2 = {F}",
                "severity": "MEDIUM"
            }
        
        return {"valid": True, "phase_rule": f"F = {C} - {P} + 2 = {F}", "degrees_of_freedom": F}
    
    def check_composition_sum(self, composition: Dict[str, float], unit: str = "at.%") -> Dict:
        """Check if composition percentages sum to ~100%"""
        if not composition:
            return {"valid": True, "message": "No composition data"}
        
        total = sum(composition.values())
        
        if abs(total - 100) > 5:  # Allow 5% tolerance
            return {
                "valid": False,
                "message": f"Composition sum {total:.1f}{unit} deviates from 100%",
                "total": total,
                "severity": "HIGH" if abs(total - 100) > 10 else "MEDIUM"
            }
        
        return {"valid": True, "total": total, "message": "Composition sums to ~100%"}
    
    def full_validation(self, fused_properties: Dict[str, FusedAlloyProperty]) -> Dict:
        """Run all validation checks on fused properties"""
        results = {
            "total_checks": 0,
            "passed": 0,
            "failed": 0,
            "violations": [],
            "warnings": []
        }
        
        for prop_name, prop in fused_properties.items():
            if isinstance(prop.fused_value, (int, float)) and prop.unit:
                results["total_checks"] += 1
                check = self.check_property_bounds(prop_name, prop.fused_value, prop.unit)
                if check["valid"]:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["violations"].append(check.get("violation", {}))
            
            # Check composition sums if applicable
            if "composition" in prop_name.lower() and isinstance(prop.fused_value, dict):
                results["total_checks"] += 1
                check = self.check_composition_sum(prop.fused_value)
                if check["valid"]:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["warnings"].append(check)
        
        results["validation_score"] = results["passed"] / results["total_checks"] if results["total_checks"] > 0 else 1.0
        return results

# =============================================
# SESSION STATE & UTILITIES
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
        "embeddings": None,
        "processing_complete": False,
        "alloy_domain_boost": True,
        "show_sources": True,
        "citation_style": "apa",
        "max_retrieved_chunks": 4,
        "use_4bit_quantization": True,
        "ollama_host": "http://localhost:11434",
        "metadata_cache": metadata_cache,
        "enable_alloy_fusion": True,
        "fusion_alloy_filter": None,
        "fusion_phase_filter": None,
        "fusion_property_filter": None,
        "debug_extraction": False,
        "viz_chart_type": "bar",
        "viz_property_focus": None,
        "viz_alloy_focus": None,
        "viz_phase_focus": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

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
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        return total_memory - reserved
    except:
        return None

def estimate_model_memory(model_key: str, use_4bit: bool = False) -> Dict[str, any]:
    repo_id = get_hf_repo_id(model_key) if not is_ollama_model(model_key) else model_key
    return MODEL_MEMORY_ESTIMATES.get(repo_id, {"params": "Unknown", "vram_fp16": "Unknown", "vram_4bit": "Unknown", "cpu_ok": False})

# =============================================
# MODEL LOADING
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
    
    st.sidebar.info(f"""📊 Model Memory Estimate:
- Parameters: {mem_info['params']}
- VRAM (FP16): {mem_info['vram_fp16']}
- VRAM (4-bit): {mem_info['vram_4bit']}
- CPU OK: {'✅ Yes' if mem_info['cpu_ok'] else '❌ No'}
- Available VRAM: {f'{available_vram:.1f}GB' if available_vram else 'N/A (CPU)'}""")
    
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

# =============================================
# DOCUMENT PROCESSING
# =============================================

def extract_alloy_metadata(text: str, filename: str) -> Dict[str, any]:
    metadata = {
        "source": filename,
        "alloy_topics": [],
        "parameters_found": {},
        "has_equations": bool(re.search(r'[\(=]\s*[\d.]+\s*[×*]\s*10\^', text)),
        "has_figures": bool(re.search(r'Figure\s*\d+|Fig\.\s*\d+', text, re.I)),
    }
    
    text_lower = text.lower()
    for topic, keywords in ALLOY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            metadata["alloy_topics"].append(topic)
    
    # Extract alloy-specific parameters
    param_patterns = {
        "grain_size_um": r'(\d+(?:\.\d+)?)\s*(?:µm|um)\s*(?:grain\s*size|average\s*grain)',
        "phase_fraction_pct": r'(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:phase\s*fraction|volume\s*fraction)',
        "hardness_hv": r'(\d+(?:\.\d+)?)\s*(?:HV|Vickers)\s*(?:hardness)?',
        "yield_strength_mpa": r'(\d+(?:\.\d+)?)\s*(?:MPa)\s*(?:yield\s*strength|YS)?',
    }
    
    for param, pattern in param_patterns.items():
        match = re.search(pattern, text, re.I)
        if match:
            try:
                metadata["parameters_found"][param] = float(match.group(1))
            except:
                pass
    
    return metadata

def load_and_chunk_alloy_documents(uploaded_files: List) -> List[Document]:
    all_chunks = []
    
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
            
            # Alloy-aware text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=ALLOY_DOMAIN_CONFIG["chunk_size"],
                chunk_overlap=ALLOY_DOMAIN_CONFIG["chunk_overlap"],
                separators=["\n\n", "\n", "Phase:", "Composition:", "Microstructure:", "Grain:", "Table", ""],
                length_function=len
            )
            
            chunks = text_splitter.split_documents(pages)
            
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "source": uploaded_file.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    **extract_alloy_metadata(chunk.page_content, uploaded_file.name),
                    "bibliographic": bib_meta.to_dict(),
                    "citation_display": bib_meta.format_citation(st.session_state.get('citation_style', 'apa')),
                })
            
            all_chunks.extend(chunks)
            st.info(f"✅ Loaded {len(chunks)} chunks from `{uploaded_file.name}`")
            
        except Exception as e:
            st.error(f"❌ Error processing `{uploaded_file.name}`: {e}")
            import traceback
            st.error(traceback.format_exc())
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    return all_chunks

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
            "alloy_topics": list(set(topic for chunk in chunks for topic in chunk.metadata.get("alloy_topics", [])))
        }
        return vectorstore
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None

# =============================================
# RAG WITH ALLOY FUSION & VISUALIZATION
# =============================================

def create_alloy_rag_prompt(retrieved_chunks: List[Document], query: str) -> str:
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        citation = chunk.metadata.get("citation_display")
        if not citation:
            source = chunk.metadata.get("source", "unknown")
            topics = chunk.metadata.get("alloy_topics", [])
            topic_str = f" [{', '.join(topics)}]" if topics else ""
            citation = f"[Source {i}{topic_str} - {source}]"
        content = chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content
        context_parts.append(f"{citation}\n{content}\n")
    
    context = "\n---\n".join(context_parts)
    
    alloy_system_prompt = """You are an expert assistant for alloy microstructure research.
Your role is to answer questions based ONLY on the provided document context.
Focus on: alloy composition, phase structure (FCC/BCC/HCP/LIQUID), microstructure features, and mechanical properties.

Rules:
1. Use ONLY information from the retrieved context below
2. If the answer isn't in the context, say "Based on the provided documents, I cannot determine..."
3. Never invent compositions, phase fractions, or microstructure parameters
4. When citing, use the EXACT citation string provided
5. For numerical values, include units when available
6. Be concise but technically complete
7. Focus on microstructure, NOT laser parameters or numerical modeling unless explicitly asked
"""
    
    user_query = f"""Retrieved Context from Alloy Microstructure Documents:
{context}

User Question: {query}

Answer (cite sources using provided citation format, focus on alloy microstructure):"""
    
    return alloy_system_prompt + user_query

def _create_fusion_aware_alloy_prompt(retrieved_docs: List[Document], query: str,
                                      fused_properties: Dict[str, FusedAlloyProperty],
                                      fusion_metrics: FusionEfficiencyMetrics,
                                      comparison_table: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        citation = doc.metadata.get('citation_display', f"[Source {i+1}]")
        content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
        context_parts.append(f"[{i+1}] {citation}\n{content}\n")
    context = "\n---\n".join(context_parts)
    
    properties_summary = ""
    if fused_properties:
        properties_summary = "**Fused Microstructure Properties**:\n"
        for prop_name, entry in list(fused_properties.items())[:8]:
            if entry.fused_value is not None and isinstance(entry.fused_value, (int, float)):
                value_str = f"{entry.fused_value:.3f}"
                if entry.standard_deviation is not None:
                    value_str = f"{value_str} ± {entry.standard_deviation:.3f}"
                else:
                    value_str = str(entry.fused_value) if entry.fused_value is not None else "N/A"
                phase_info = f" ({entry.phase})" if entry.phase else ""
                properties_summary += f"• {prop_name.replace('_', ' ').title()}: {value_str} {entry.unit or ''}{phase_info} [conf: {entry.fusion_confidence.value}, sources: {entry.source_count}]\n"
        properties_summary += "\n"
    
    table_section = f"**Comparison Table**:\n{comparison_table}\n" if comparison_table else ""
    
    efficiency_note = ""
    if fusion_metrics.overall_fusion_efficiency >= 0.7:
        efficiency_note = f"🎯 High-confidence fusion ({fusion_metrics.overall_fusion_efficiency:.2f}/1.0): Properties synthesized from {fusion_metrics.unique_sources_used} sources.\n"
    elif fusion_metrics.overall_fusion_efficiency >= 0.4:
        efficiency_note = f"⚠️ Moderate-confidence fusion ({fusion_metrics.overall_fusion_efficiency:.2f}/1.0): Some property variations detected.\n"
    else:
        efficiency_note = f"🔍 Low-confidence fusion ({fusion_metrics.overall_fusion_efficiency:.2f}/1.0): Limited or conflicting data.\n"
    
    system_prompt = """You are an expert scientific assistant specializing in alloy microstructure research.
YOUR TASK:
1. Answer the user's question using the retrieved document context AND the fused property summary below
2. When microstructure property values are available from fusion, PREFER the fused consensus value with its uncertainty range
3. Cite sources precisely using [Author, Year] or [DOI:xxx] format immediately after claims
4. If fused properties show conflicts, acknowledge the variation and note possible causes (different compositions, heat treatments, phases)
5. For comparative questions, reference the comparison table if provided
6. Always include units for numerical values and note alloy system/phase when relevant
7. FOCUS on microstructure: grain size, phase fractions, precipitates, NOT laser parameters unless asked

RESPONSE STRUCTURE:
1. Direct answer (1-2 sentences)
2. Supporting evidence with fused property values and citations
3. Comparison table reference if relevant to query
4. Uncertainty/limitations note if fusion confidence is moderate/low
5. Suggested follow-up if appropriate
"""
    
    user_prompt = f"""RETRIEVED DOCUMENT CONTEXT:
{context}

{efficiency_note}{properties_summary}{table_section}

USER QUESTION: {query}

SCIENTIFIC ANSWER (use fused properties when available, cite sources precisely, focus on alloy microstructure):"""
    
    full_prompt = system_prompt + user_prompt
    context_metadata = {
        "fused_properties_count": len(fused_properties),
        "fusion_efficiency": fusion_metrics.overall_fusion_efficiency,
        "comparison_table_available": comparison_table is not None
    }
    
    return full_prompt, context_metadata

def generate_local_response_transformers(tokenizer, model, device: str, prompt: str, backend_name: str) -> str:
    try:
        if "Qwen" in backend_name or "qwen" in backend_name.lower():
            messages = [
                {"role": "system", "content": "You are an expert in alloy microstructure interaction. Focus on composition, phases, microstructure."},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        elif "Llama" in backend_name or "llama" in backend_name.lower():
            messages = [
                {"role": "system", "content": "You are an expert in alloy microstructure interaction. Focus on composition, phases, microstructure."},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        elif "Mistral" in backend_name or "mistral" in backend_name.lower():
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        else:
            formatted_prompt = prompt
        
        inputs = tokenizer.encode(
            formatted_prompt, return_tensors='pt', truncation=True,
            max_length=ALLOY_DOMAIN_CONFIG["max_context_tokens"]
        )
        if device == "cuda" and torch.cuda.is_available():
            inputs = inputs.to('cuda')
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=ALLOY_DOMAIN_CONFIG["max_new_tokens"],
                temperature=ALLOY_DOMAIN_CONFIG["temperature"],
                do_sample=(ALLOY_DOMAIN_CONFIG["temperature"] > 0),
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "[/INST]" in full_text:
            answer = full_text.split("[/INST]")[-1].strip()
        elif "Answer (cite sources" in full_text:
            answer = full_text.split("Answer (cite sources")[-1].strip()
            answer = re.split(r'\n(?:Question|User|Context):', answer)[0].strip()
        else:
            answer = full_text[-ALLOY_DOMAIN_CONFIG["max_new_tokens"]*2:].strip()
        
        answer = re.sub(r'\s+', ' ', answer).strip()
        return answer if answer else "I was unable to generate a response. Please try rephrasing your question."
    
    except Exception as e:
        st.error(f"Generation error: {e}")
        return f"Error generating response: {str(e)[:200]}..."

def generate_local_response_ollama(model_tag: str, ollama_host: str, prompt: str) -> str:
    try:
        client = ollama.Client(host=ollama_host)
        messages = [
            {"role": "system", "content": "You are an expert in alloy microstructure research. Answer based ONLY on the provided context."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = client.chat(
                model=model_tag, messages=messages,
                options={"temperature": ALLOY_DOMAIN_CONFIG["temperature"], "num_predict": ALLOY_DOMAIN_CONFIG["max_new_tokens"]},
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
                options={"temperature": ALLOY_DOMAIN_CONFIG["temperature"], "num_predict": ALLOY_DOMAIN_CONFIG["max_new_tokens"]}
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

def retrieve_and_answer_with_alloy_fusion(vectorstore, tokenizer, model, device_or_host: str, backend: str, backend_type: str,
                                          query: str, k: int = None, score_threshold: float = None,
                                          enable_fusion: bool = True, alloy_filter: Optional[str] = None,
                                          phase_filter: Optional[List[str]] = None,
                                          property_filter: Optional[List[str]] = None) -> Tuple[str, List[Document], float, Dict[str, Any]]:
    
    k = k or ALLOY_DOMAIN_CONFIG["retrieval_k"]
    score_threshold = score_threshold or ALLOY_DOMAIN_CONFIG["score_threshold"]
    
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold}
    )
    retrieved_docs = retriever.invoke(query)
    
    if retrieved_docs:
        query_embedding = vectorstore.embedding_function.embed_query(query)
        scores = []
        for doc in retrieved_docs:
            doc_embedding = vectorstore.embedding_function.embed_query(doc.page_content[:500])
            sim = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-8)
            scores.append(sim)
        avg_relevance = np.mean(scores) if scores else 0.0
    else:
        avg_relevance = 0.0
    
    if not retrieved_docs:
        return ("Based on the uploaded documents, I could not find information relevant to your question about alloy microstructure. Try rephrasing or checking document content.",
                [], avg_relevance, {"error": "no_relevant_chunks", "fusion_enabled": enable_fusion})
    
    if enable_fusion:
        property_extractor = AlloyPropertyExtractor(ALLOY_KEYWORDS)
        fusion_engine = AlloyFusionEngine(property_extractor)
        
        fused_properties, fusion_metrics = fusion_engine.fuse_alloy_documents(
            retrieved_docs, query,
            alloy_filter=alloy_filter,
            phase_filter=phase_filter,
            property_filter=property_filter
        )
        
        comparison_table = None
        if fused_properties:
            comparison_table = fusion_engine.generate_comparison_table(fused_properties, format="markdown")
        
        prompt, fusion_context = _create_fusion_aware_alloy_prompt(
            retrieved_docs, query, fused_properties, fusion_metrics, comparison_table
        )
        
        answer = generate_local_response(
            tokenizer=tokenizer, model_or_tag=model, device_or_host=device_or_host,
            prompt=prompt, backend=backend, backend_type=backend_type
        )
        
        if fusion_metrics.overall_fusion_efficiency > 0.5 and comparison_table:
            answer += f"\n---\n**📊 Microstructure Property Comparison**:\n{comparison_table}"
        
        metadata = {
            "fusion_enabled": True,
            "fusion_metrics": {"efficiency": fusion_metrics.overall_fusion_efficiency, "display": fusion_metrics.to_display_dict()},
            "fused_properties": {k: v.to_comparison_row() for k, v in fused_properties.items()},
            "comparison_table": comparison_table,
            "source_citations": [
                {"citation": doc.metadata.get('citation_display', 'Unknown'), "relevance": scores[i] if i < len(scores) else 0, "topics": doc.metadata.get('alloy_topics', [])}
                for i, doc in enumerate(retrieved_docs)
            ],
            "retrieval_relevance": avg_relevance
        }
        
        return answer, retrieved_docs, avg_relevance, metadata
    
    else:
        prompt = create_alloy_rag_prompt(retrieved_docs, query)
        answer = generate_local_response(
            tokenizer=tokenizer, model_or_tag=model, device_or_host=device_or_host,
            prompt=prompt, backend=backend, backend_type=backend_type
        )
        return answer, retrieved_docs, avg_relevance, {"fusion_enabled": False}

# =============================================
# VISUALIZATION UI COMPONENTS
# =============================================

def render_viz_control_panel(fused_properties: Dict[str, FusedAlloyProperty], retrieved_docs: List[Document]):
    """Render interactive visualization controls"""
    
    st.markdown("### 📊 Alloy Microstructure Visualizations")
    
    # Visualization type selector
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Composition Pie Chart", "Microstructure Radar Chart", 
         "Phase Fraction Bar Chart", "Property Comparison", "Grain Size Distribution", "Phase Evolution"],
        key="viz_type_select"
    )
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        alloy_options = ["All Alloys"] + list(set(ALLOY_SYSTEMS.keys()))
        st.session_state.viz_alloy_focus = st.selectbox("Alloy System Filter", options=alloy_options, key="viz_alloy")
    
    with col2:
        phase_options = ["All Phases"] + list(ALLOY_STRUCTURE_TYPES.keys())
        st.session_state.viz_phase_focus = st.selectbox("Phase Filter", options=phase_options, key="viz_phase")
    
    with col3:
        property_options = ["All Properties"] + [p.replace('_', ' ').title() for p in fused_properties.keys()]
        st.session_state.viz_property_focus = st.selectbox("Property Focus", options=property_options, key="viz_prop")
    
    # Generate visualization based on selection
    viz_engine = AlloyVisualizationEngine()
    
    if viz_type == "Composition Pie Chart":
        # Extract composition data from fused properties
        composition_data = []
        for prop_name, prop in fused_properties.items():
            if "composition" in prop_name.lower() and isinstance(prop.fused_value, (float, int)):
                composition_data.append({
                    "element": prop_name.replace("composition_", "").replace("_", "").upper(),
                    "percentage": prop.fused_value
                })
        
        if composition_data:
            alloy_name = st.session_state.viz_alloy_focus if st.session_state.viz_alloy_focus != "All Alloys" else "Multicomponent Alloy"
            fig = viz_engine.create_composition_pie_chart(composition_data, alloy_name)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No composition data available. Upload documents with alloy composition information.")
    
    elif viz_type == "Microstructure Radar Chart":
        # Build microstructure metrics dictionary
        microstructure_metrics = {}
        for prop_name, prop in fused_properties.items():
            if prop.fused_value is not None and isinstance(prop.fused_value, (int, float)):
                display_name = prop_name.replace('_', ' ').title()
                # Filter by selected property if set
                if st.session_state.viz_property_focus == "All Properties" or st.session_state.viz_property_focus == display_name:
                    microstructure_metrics[display_name] = prop.fused_value
        
        if microstructure_metrics:
            title = f"Microstructure Profile"
            if st.session_state.viz_alloy_focus != "All Alloys":
                title += f" - {st.session_state.viz_alloy_focus}"
            if st.session_state.viz_phase_focus != "All Phases":
                title += f" ({st.session_state.viz_phase_focus})"
            
            fig = viz_engine.create_microstructure_radar_chart(microstructure_metrics, title=title)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient microstructure data for radar chart. Ensure documents contain quantitative microstructure properties.")
    
    elif viz_type == "Phase Fraction Bar Chart":
        # Extract phase fraction data
        phase_data = []
        for prop_name, prop in fused_properties.items():
            if "phase_fraction" in prop_name.lower() or "fraction" in prop_name.lower():
                if prop.fused_value is not None and isinstance(prop.fused_value, (int, float)):
                    phase_data.append({
                        "sample": prop.sources[0]["citation"][:30] + "..." if prop.sources else "Unknown",
                        "phase": prop.phase or prop_name.replace("phase_fraction_", "").replace("_", " ").title(),
                        "fraction": prop.fused_value
                    })
        
        if phase_data:
            stacked = st.checkbox("Stacked bars", value=True, key="phase_stacked")
            fig = viz_engine.create_phase_fraction_bar_chart(phase_data, stacked=stacked)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No phase fraction data available.")
    
    elif viz_type == "Property Comparison":
        # Extract property data for comparison
        alloy_properties = []
        for prop_name, prop in fused_properties.items():
            if prop.fused_value is not None and isinstance(prop.fused_value, (int, float)):
                # Filter by selected property
                if st.session_state.viz_property_focus == "All Properties" or st.session_state.viz_property_focus == prop_name.replace('_', ' ').title():
                    alloy_properties.append({
                        "alloy_name": prop.alloy_system or "Unknown",
                        "alloy_system": prop.alloy_system or "Other",
                        "property_name": prop_name,
                        prop_name: prop.fused_value,
                        "uncertainty": prop.standard_deviation,
                        "phase": prop.phase,
                        "condition": "; ".join(prop.conditions_summary.get("context", [])) if prop.conditions_summary else None
                    })
        
        if alloy_properties:
            property_name = st.session_state.viz_property_focus if st.session_state.viz_property_focus != "All Properties" else list(fused_properties.keys())[0]
            fig = viz_engine.create_alloy_property_comparison_chart(alloy_properties, property_name)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No property comparison data available.")
    
    elif viz_type == "Grain Size Distribution":
        # Extract grain size data
        grain_data = []
        for prop_name, prop in fused_properties.items():
            if "grain" in prop_name.lower() and "size" in prop_name.lower():
                if prop.fused_value is not None and isinstance(prop.fused_value, (int, float)):
                    grain_data.append({
                        "grain_size": prop.fused_value,
                        "alloy_system": prop.alloy_system or "Unknown",
                        "phase": prop.phase or "Unknown",
                        "source": prop.sources[0]["citation"][:30] if prop.sources else "Unknown"
                    })
        
        if grain_data:
            fig = viz_engine.create_grain_size_distribution(grain_data)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No grain size data available.")
    
    elif viz_type == "Phase Evolution":
        st.info("Phase evolution chart requires temperature-dependent data. Ensure documents contain phase fraction vs. temperature information.")

def render_fusion_metrics_panel(fusion_metadata: Dict[str, Any]):
    """Display fusion efficiency metrics"""
    if not fusion_metadata.get("fusion_enabled"):
        return
    
    metrics_display = fusion_metadata.get("fusion_metrics", {}).get("display", {})
    if not metrics_display:
        return
    
    with st.expander("📊 Information Fusion Efficiency", expanded=True):
        overall = fusion_metadata["fusion_metrics"]["efficiency"]
        if overall is not None:
            st.progress(min(1.0, max(0.0, overall)))
            st.caption(f"Overall Fusion Efficiency: {overall:.2f}/1.0")
        else:
            st.caption("Overall Fusion Efficiency: N/A")
        
        cols = st.columns(2)
        metric_items = list(metrics_display.items())
        for i, (label, value) in enumerate(metric_items):
            with cols[i % 2]:
                display_value = value.split(":")[-1].strip() if ":" in value and value else "N/A"
                st.metric(label=label, value=display_value)
        
        sources = fusion_metadata.get("source_citations", [])
        if sources:
            st.markdown("**📚 Sources Contributing to Fusion**:")
            for src in sources[:4]:
                relevance = src.get("relevance", 0)
                relevance_bar = "🟢" if relevance > 0.6 else "🟡" if relevance > 0.3 else "🔴"
                topics = src.get("topics", [])
                topics_str = ", ".join(topics[:2]) if topics else "none"
                st.caption(f"{relevance_bar} {src['citation']} (topics: {topics_str})")
        
        fused_props = fusion_metadata.get("fused_properties", {})
        if fused_props:
            conflicts = [k for k, v in fused_props.items() if v.get("confidence") == "low"]
            if conflicts:
                st.warning(f"⚠️ {len(conflicts)} property(ies) have low-confidence fusion: {', '.join(conflicts[:3])}")

def render_extracted_properties_debug(extracted_props: List[AlloyProperty], source_citation: str):
    """Debug view for extracted properties"""
    if not extracted_props:
        st.info("🔍 No alloy properties extracted from this chunk")
        return
    
    with st.expander(f"🐛 Extracted Alloy Properties: {source_citation}", expanded=False):
        for i, prop in enumerate(extracted_props, 1):
            st.markdown(f"**{i}. {prop.normalized_name}**")
            st.caption(f"Value: `{prop.value}` {prop.unit or ''} | Type: {prop.property_type}")
            if prop.phase:
                st.caption(f"Phase: {prop.phase}")
            if prop.alloy_system:
                st.caption(f"Alloy System: {prop.alloy_system}")
            if prop.condition:
                st.caption(f"Condition: {prop.condition}")
            if prop.context_snippet:
                st.code(prop.context_snippet[:200] + "..." if len(prop.context_snippet) > 200 else prop.context_snippet, language="text")
            st.divider()

def render_comparison_table_in_chat(comparison_table: Optional[str], fused_properties: Dict):
    """Display comparison table in chat"""
    if not comparison_table:
        return
    
    with st.expander("📋 Microstructure Property Comparison Table", expanded=False):
        st.markdown(comparison_table, unsafe_allow_html=True)
    
    if fused_properties:
        selected_prop = st.selectbox(
            "🔍 Explore property details:",
            options=["Select a property..."] + list(fused_properties.keys()),
            key="fusion_prop_select"
        )
        if selected_prop and selected_prop != "Select a property...":
            prop_data = fused_properties[selected_prop]
            st.json({
                "property": selected_prop,
                "fused_value": prop_data["value"],
                "unit": prop_data["unit"],
                "range": prop_data["range"],
                "sources": prop_data["sources"],
                "confidence": prop_data["confidence"],
                "alloy_system": prop_data.get("alloy_system"),
                "phase": prop_data.get("phase")
            })

def render_physics_validation_panel(fused_properties: Dict[str, FusedAlloyProperty]):
    """Display physics validation results"""
    if not fused_properties:
        return
    
    validator = AlloyPhysicsValidator()
    validation_results = validator.full_validation(fused_properties)
    
    with st.expander("🧮 Physics Validation", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Validation Score", f"{validation_results['validation_score']:.0%}")
        col2.metric("Checks Passed", validation_results["passed"])
        col3.metric("Checks Failed", validation_results["failed"])
        
        if validation_results["violations"]:
            st.warning("⚠️ Physical bound violations detected:")
            for v in validation_results["violations"]:
                st.error(f"**{v.get('property')}**: {v.get('value')} {v.get('unit')} outside bounds")
                st.caption(f"Severity: {v.get('severity')}")
        
        if validation_results["warnings"]:
            st.info("ℹ️ Validation warnings:")
            for w in validation_results["warnings"]:
                st.caption(w.get("message", "Unknown warning"))

# =============================================
# MAIN UI COMPONENTS
# =============================================

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        backend_option = st.radio(
            "🔧 Inference Backend",
            options=["Hugging Face Transformers", "Ollama (if installed)"],
            index=0,
            help="Transformers: direct HF model loading\nOllama: use local ollama serve (faster switching)"
        )
        st.session_state.inference_backend = backend_option
        
        if backend_option == "Ollama (if installed)":
            if not OLLAMA_AVAILABLE:
                st.error("❌ ollama library not installed")
                st.code("pip install ollama")
                st.info("Also ensure Ollama server is running: ollama serve")
            available_ollama_models = [k for k in LOCAL_LLM_OPTIONS.keys() if is_ollama_model(k)]
            model_choice = st.selectbox(
                "🧠 Local LLM Backend (Ollama)",
                options=available_ollama_models if available_ollama_models else ["No Ollama models available"],
                index=0 if available_ollama_models else 0,
                help="Models served via local Ollama instance"
            )
        else:
            hf_models = [k for k in LOCAL_LLM_OPTIONS.keys() if not is_ollama_model(k)]
            model_choice = st.selectbox(
                "🧠 Local LLM Backend (Hugging Face)",
                options=hf_models, index=2,
                help="Models loaded directly via transformers library"
            )
        
        st.session_state.llm_model_choice = model_choice
        
        if backend_option == "Hugging Face Transformers" and not is_ollama_model(model_choice):
            st.session_state.use_4bit_quantization = st.checkbox(
                "🗜️ Use 4-bit quantization (reduces VRAM usage)", value=True,
                help="Enable for models >3B parameters to reduce memory usage by ~75%"
            )
        
        if backend_option == "Ollama (if installed)" or is_ollama_model(model_choice):
            st.session_state.ollama_host = st.text_input(
                "🌐 Ollama Host", value=st.session_state.ollama_host,
                help="URL of your Ollama server (default: http://localhost:11434)"
            )
        
        st.markdown("#### 🔬 Alloy Domain Settings")
        st.session_state.alloy_domain_boost = st.checkbox(
            "Boost alloy-topic relevance", value=True,
            help="Prioritize chunks containing alloy-specific keywords (FCC, BCC, grain, phase, etc.)"
        )
        st.session_state.show_sources = st.checkbox(
            "Show source citations", value=True,
            help="Display which documents chunks came from"
        )
        st.session_state.enable_alloy_fusion = st.checkbox(
            "🔗 Enable Alloy Microstructure Fusion", value=True,
            help="Enable cross-document property extraction focused on alloy microstructure ONLY"
        )
        st.session_state.debug_extraction = st.checkbox(
            "🐛 Debug Property Extraction", value=False,
            help="Show extracted alloy properties in UI for diagnosis"
        )
        
        st.markdown("#### 📊 Visualization Settings")
        st.session_state.viz_chart_type = st.selectbox(
            "Default Chart Type",
            options=["bar", "radar", "pie", "line"],
            index=0,
            help="Default visualization type for alloy properties"
        )
        
        st.markdown("#### 📝 Citation Format")
        st.session_state.citation_style = st.selectbox(
            "Citation display style",
            options=["apa", "doi", "full", "short"], index=0,
            format_func=lambda x: {
                "apa": "APA: FirstAuthor et al., Journal, Year",
                "doi": "DOI: 10.xxxx/xxxxx",
                "full": "Full: Author (Year). Title. Journal, Vol(Issue), Pages",
                "short": "Short: [FirstAuthor Year] or [DOI]"
            }[x],
            help="How citations appear in responses and source lists"
        )
        
        st.session_state.max_retrieved_chunks = st.slider(
            "Chunks to retrieve", min_value=2, max_value=8, value=4,
            help="More chunks = more context but slower responses"
        )
        
        st.markdown("---")
        st.markdown("""<div style="background:#f0f9ff;padding:1rem;border-radius:0.5rem;border-left:4px solid #3b82f6">
        <strong>💡 Tips for Best Results:</strong><ul style="margin:0.5rem 0 0 1rem;padding:0">
        <li>Upload papers about alloy microstructure, phase diagrams, solidification</li>
        <li>Ask specific questions: "What is the grain size of FCC AlSi10Mg?"</li>
        <li>Small models (≤1.5B) work on CPU; larger need GPU</li>
        <li>First load may take 1-2 min (model download)</li>
        <li>For Ollama: run <code>ollama pull qwen2.5:7b</code> first</li>
        <li>🔗 Fusion works best with comparative queries across multiple alloy studies</li>
        <li>🐛 Enable debug mode to see extracted microstructure properties</li>
        <li>📊 Use visualization panel for interactive charts</li>
        </ul></div>""", unsafe_allow_html=True)
        
        st.markdown("---")
        gpu_info = "CUDA" if torch.cuda.is_available() else "CPU"
        vram_info = f"{get_available_gpu_memory():.1f}GB free" if torch.cuda.is_available() and get_available_gpu_memory() else "N/A"
        st.caption(f"🖥️ Device: {gpu_info}")
        st.caption(f"💾 Available VRAM: {vram_info}")
        st.caption(f"📦 Embedding model: ~80MB")
        st.caption(f"🤖 LLM: {LOCAL_LLM_OPTIONS.get(model_choice, 'unknown')}")
        
        st.markdown("#### 📚 Metadata Extraction")
        if PDF2DOI_AVAILABLE:
            st.success("✅ pdf2doi: Available for DOI lookup")
        else:
            st.info("ℹ️ pdf2doi: Install with `pip install pdf2doi` for enhanced DOI extraction")
        if CROSSREF_AVAILABLE:
            st.success("✅ Crossref API: Available for metadata enrichment")
        else:
            st.info("ℹ️ Crossref: Install with `pip install crossrefapi` for journal/author lookup")
        
        if backend_option == "Ollama (if installed)" and OLLAMA_AVAILABLE:
            try:
                client = ollama.Client(host=st.session_state.ollama_host)
                models = client.list()
                st.success(f"✅ Ollama connected: {len(models.get('models', []))} models available")
            except:
                st.error("❌ Cannot connect to Ollama")

def render_document_uploader():
    st.markdown("### 📁 Upload Alloy Microstructure Documents")
    uploaded_files = st.file_uploader(
        "Select PDF or TXT files about alloy composition, phase structure, microstructure, solidification, etc.",
        type=["pdf", "txt"], accept_multiple_files=True,
        help="Documents will be processed locally - no data leaves your browser. Bibliographic metadata (DOI, authors, journal, year) will be extracted for human-readable citations. Focus: FCC/BCC/HCP phases, grain size, phase fractions, mechanical properties."
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
    
    with st.spinner(f"Processing {len(new_files)} document(s) and extracting alloy microstructure data..."):
        try:
            chunks = load_and_chunk_alloy_documents(new_files)
            if not chunks:
                st.error("No chunks extracted. Check file format.")
                return False
            
            for f in new_files:
                st.session_state.processed_files.add(f.name)
            
            st.session_state.all_chunks.extend(chunks)
            
            with st.spinner("Creating vector index (this may take a minute)..."):
                vectorstore = create_local_vector_store(st.session_state.all_chunks, LOCAL_EMBEDDING_MODEL)
                if vectorstore is None:
                    return False
                st.session_state.vectorstore = vectorstore
            
            st.success(f"✅ Ready! Indexed {len(st.session_state.all_chunks)} chunks from {len(st.session_state.processed_files)} files")
            st.session_state.processing_complete = True
            return True
            
        except Exception as e:
            st.error(f"Processing failed: {e}")
            import traceback
            st.error(traceback.format_exc())
            return False

def render_chat_interface():
    if not st.session_state.get('vectorstore'):
        st.info("👆 Upload alloy microstructure documents above to start chatting")
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
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message.get("sources") and st.session_state.show_sources:
                with st.expander("📚 Retrieved Sources with Citations"):
                    for i, src in enumerate(message["sources"], 1):
                        citation = src.metadata.get("citation_display", "Unknown source")
                        source_name = src.metadata.get("source", "unknown")
                        topics = src.metadata.get("alloy_topics", [])
                        st.markdown(f"**[{i}]** {citation}")
                        
                        bib = src.metadata.get("bibliographic", {})
                        if bib and any(bib.get(k) for k in ['doi', 'authors', 'journal', 'year']):
                            with st.expander("🔍 Bibliographic Details"):
                                if bib.get('doi'):
                                    st.markdown(f"**DOI:** `{bib['doi']}`")
                                if bib.get('arxiv_id'):
                                    st.markdown(f"**arXiv:** `{bib['arxiv_id']}`")
                                if bib.get('authors'):
                                    st.markdown(f"**Authors:** {', '.join(bib['authors'][:3])}{'...' if len(bib['authors'])>3 else ''}")
                                if bib.get('journal'):
                                    st.markdown(f"**Journal:** {bib['journal']}")
                                if bib.get('year'):
                                    st.markdown(f"**Year:** {bib['year']}")
                                st.caption(f"Extraction method: {bib.get('extraction_method', 'unknown')} (confidence: {bib.get('confidence', 0):.2f})")
                        
                        st.markdown(f"> {src.page_content[:300]}...")
                        
                        if st.session_state.debug_extraction:
                            extractor = AlloyPropertyExtractor(ALLOY_KEYWORDS)
                            record = extractor.extract_properties_from_chunk(src.page_content, src.metadata)
                            render_extracted_properties_debug(record.extracted_properties, citation)
            
            if message.get("fusion_metadata") and st.session_state.enable_alloy_fusion:
                render_fusion_metrics_panel(message["fusion_metadata"])
                if message["fusion_metadata"].get("comparison_table"):
                    render_comparison_table_in_chat(
                        message["fusion_metadata"]["comparison_table"],
                        message["fusion_metadata"].get("fused_properties", {})
                    )
                # Physics validation
                fused_props = message["fusion_metadata"].get("fused_properties", {})
                if fused_props:
                    # Reconstruct FusedAlloyProperty objects for validation
                    reconstructed = {}
                    for name, data in fused_props.items():
                        prop = FusedAlloyProperty(property_name=name)
                        prop.fused_value = data.get("value")
                        prop.unit = data.get("unit")
                        prop.fusion_confidence = FusionConfidence(data.get("confidence", "unknown"))
                        prop.source_count = data.get("sources", 0)
                        prop.standard_deviation = float(data["std"]) if data.get("std") else None
                        if data.get("range"):
                            parts = data["range"].split("–")
                            if len(parts) == 2:
                                prop.value_range = (float(parts[0]), float(parts[1]))
                        prop.alloy_system = data.get("alloy_system")
                        prop.phase = data.get("phase")
                        reconstructed[name] = prop
                    render_physics_validation_panel(reconstructed)
    
    # Chat input
    if prompt := st.chat_input("Ask about alloy composition, phase structure, grain size, or compare microstructures..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("🔍 Retrieving, fusing alloy microstructure data, and generating..."):
                try:
                    if st.session_state.enable_alloy_fusion:
                        answer, retrieved_docs, relevance, metadata = retrieve_and_answer_with_alloy_fusion(
                            vectorstore=st.session_state.vectorstore,
                            tokenizer=st.session_state.llm_tokenizer,
                            model=st.session_state.llm_model,
                            device_or_host=st.session_state.llm_device_or_host,
                            backend=st.session_state.llm_model_choice,
                            backend_type=st.session_state.llm_backend_type,
                            query=prompt,
                            k=st.session_state.max_retrieved_chunks,
                            enable_fusion=True,
                            alloy_filter=st.session_state.viz_alloy_focus if st.session_state.viz_alloy_focus != "All Alloys" else None,
                            phase_filter=[st.session_state.viz_phase_focus] if st.session_state.viz_phase_focus != "All Phases" else None,
                            property_filter=[st.session_state.viz_property_focus.lower().replace(' ', '_')] if st.session_state.viz_property_focus != "All Properties" else None
                        )
                    else:
                        answer, retrieved_docs, relevance, metadata = retrieve_and_answer_with_alloy_fusion(
                            vectorstore=st.session_state.vectorstore,
                            tokenizer=st.session_state.llm_tokenizer,
                            model=st.session_state.llm_model,
                            device_or_host=st.session_state.llm_device_or_host,
                            backend=st.session_state.llm_model_choice,
                            backend_type=st.session_state.llm_backend_type,
                            query=prompt,
                            k=st.session_state.max_retrieved_chunks,
                            enable_fusion=False
                        )
                    
                    # Stream the response
                    display_text = ""
                    for word in answer.split():
                        display_text += word + " "
                        message_placeholder.markdown(display_text + "▌")
                        time.sleep(0.02)
                    message_placeholder.markdown(answer)
                    
                    # Save message with metadata
                    message_dict = {
                        "role": "assistant",
                        "content": answer,
                        "sources": retrieved_docs if st.session_state.show_sources else None,
                        "relevance": relevance
                    }
                    if st.session_state.enable_alloy_fusion:
                        message_dict["fusion_metadata"] = metadata
                    st.session_state.messages.append(message_dict)
                    
                    # Show relevance and fusion metrics
                    if relevance > 0:
                        fusion_eff = metadata.get("fusion_metrics", {}).get("efficiency", 0) if metadata.get("fusion_enabled") else None
                        if fusion_eff is not None:
                            st.caption(f"📊 Response relevance: {relevance:.2f}/1.0 | Fusion efficiency: {fusion_eff:.2f}/1.0")
                        else:
                            st.caption(f"📊 Response relevance: {relevance:.2f}/1.0")
                    
                    # Show visualization panel if fusion was enabled
                    if st.session_state.enable_alloy_fusion and metadata.get("fusion_enabled") and metadata.get("fused_properties"):
                        render_viz_control_panel(metadata["fused_properties"], retrieved_docs)
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)[:300]}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

def render_footer():
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📚 Example Questions:**")
        st.caption("• What is the grain size of FCC AlSi10Mg after SLM?")
        st.caption("• Compare phase fractions in BCC vs FCC CoCrFeNi HEA")
        st.caption("• How does aging affect precipitate size in Al-Cu alloys?")
        st.caption("• What is the yield strength of Ti6Al4V with equiaxed grains?")
    
    with col2:
        st.markdown("**⚡ Performance Tips:**")
        st.caption("• Keep questions focused on alloy microstructure")
        st.caption("• Specify alloy system (Fe-based, Ni-based, HEA) for better retrieval")
        st.caption("• CPU mode: allow 10-30s per response; GPU: 2-10s")
        st.caption("• Enable fusion for comparative queries across studies")
    
    with col3:
        st.markdown("**🔐 Privacy & Fusion:**")
        st.caption("• All processing happens locally in your session")
        st.caption("• Multi-document fusion extracts & compares microstructure properties")
        st.caption("• Fusion efficiency metrics quantify synthesis quality")
        st.caption("• Citations display as 'FirstAuthor et al., Journal, Year' or DOI")
        st.caption("• Physics validation checks thermodynamic consistency")

# =============================================
# MAIN APPLICATION
# =============================================

def main():
    st.set_page_config(
        page_title="🔬 Alloy Microstructure RAG + Fusion + Viz",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""<style>
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
    .stChatMessage {
        border-radius: 0.5rem;
        margin: 0.25rem 0;
    }
    .model-warning {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 0.75rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin: 0.5rem 0;
    }
    .citation-badge {
        display: inline-block;
        background: #e0e7ff;
        color: #3730a3;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
        margin: 0.1rem 0;
    }
    .fusion-badge {
        display: inline-block;
        background: #dcfce7;
        color: #166534;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
        margin: 0.1rem 0;
    }
    </style>""", unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🔬 Alloy Microstructure RAG + Fusion + Visualization</h1>', unsafe_allow_html=True)
    st.markdown("""<div style="text-align:center;color:#64748b;margin-bottom:1.5rem">
    Upload research papers on multicomponent alloys and microstructure. Get answers with 
    <strong>human-readable citations</strong> and <strong>interactive visualizations</strong>.
    <br><span class="fusion-badge">🔗 Multi-document fusion: FCC/BCC/HCP/LIQUID phases, grain size, phase fractions</span>
    <br><span class="fusion-badge">📊 Customizable charts: Pie, Radar, Bar with alloy/phase filters</span>
    <br><span class="fusion-badge">🧮 Physics validation: Thermodynamic bounds, composition checks</span>
    </div>""", unsafe_allow_html=True)
    
    # Initialize
    initialize_session_state()
    render_sidebar()
    
    # Memory warning if needed
    if st.session_state.llm_model_choice and not is_ollama_model(st.session_state.llm_model_choice):
        mem_info = estimate_model_memory(st.session_state.llm_model_choice, st.session_state.get('use_4bit_quantization', True))
        available_vram = get_available_gpu_memory()
        if available_vram and not mem_info['cpu_ok']:
            required = float(mem_info['vram_4bit'].replace('GB','').replace('~','').strip()) if 'GB' in mem_info['vram_4bit'] else 100
            if available_vram < required:
                st.markdown(f"""<div class="model-warning">⚠️ <strong>Memory Warning:</strong> {st.session_state.llm_model_choice} requires ~{mem_info['vram_4bit']} VRAM.
                You have ~{available_vram:.1f}GB available. Consider:
                <ul><li>Using 4-bit quantization (already enabled)</li><li>Selecting a smaller model</li><li>Using Ollama backend for better memory management</li></ul></div>""", unsafe_allow_html=True)
    
    # Main layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        uploaded_files = render_document_uploader()
        if uploaded_files and st.button("🔄 Process Documents", type="primary", use_container_width=True):
            process_documents(uploaded_files)
        
        if st.session_state.processing_complete:
            st.success("✅ Knowledge base ready")
            if st.session_state.vectorstore and hasattr(st.session_state.vectorstore, 'metadata'):
                meta = st.session_state.vectorstore.metadata
                st.caption(f"📦 {meta.get('total_chunks', '?')} chunks")
                topics = meta.get('alloy_topics', [])
                if topics:
                    st.caption(f"🔬 Topics: {', '.join(topics[:5])}" + ("..." if len(topics)>5 else ""))
            if st.session_state.all_chunks:
                sample_chunk = st.session_state.all_chunks[0]
                citation = sample_chunk.metadata.get("citation_display")
                if citation:
                    st.markdown(f'<span class="citation-badge">📝 Sample citation: {citation}</span>', unsafe_allow_html=True)
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
            st.markdown("""<div class="info-card"><h3>👋 Welcome to Alloy Microstructure RAG!</h3>
            <p>This assistant helps you query documents about:</p>
            <ul>
            <li>🔬 Alloy composition (at.%, wt.%) and multicomponent systems</li>
            <li>🧱 Phase structure: FCC, BCC, HCP, LIQUID, intermetallics</li>
            <li>🌾 Microstructure: grain size, dendrite spacing, precipitates</li>
            <li>📊 Phase fractions and evolution with temperature</li>
            <li>💪 Mechanical properties: hardness, yield strength, modulus</li>
            <li>🔗 <strong>Multi-document fusion</strong> with efficiency metrics</li>
            <li>📈 <strong>Interactive visualizations</strong>: Pie, Radar, Bar charts</li>
            <li>🧮 <strong>Physics validation</strong>: Thermodynamic consistency</li>
            </ul>
            <p><strong>🎯 Enhanced Features:</strong></p>
            <ul>
            <li>Citations display as "Smith et al., Acta Mater., 2023" or DOI</li>
            <li>🔗 Cross-document property extraction focused on microstructure ONLY</li>
            <li>📊 Fusion efficiency metrics per answer</li>
            <li>📋 Automatic comparison table generation</li>
            <li>🎨 Customizable visualizations with alloy/phase filters</li>
            <li>🐛 Debug mode for property extraction diagnosis</li>
            <li>🧮 Physics-aware validation for numerical properties</li>
            </ul>
            <p><strong>Getting started:</strong></p>
            <ol>
            <li>Upload PDF/TXT files about alloy microstructure in the left panel</li>
            <li>Click "Process Documents"</li>
            <li>Select your preferred local LLM in sidebar</li>
            <li>Enable "Alloy Microstructure Fusion" for comparative queries</li>
            <li>Use visualization panel for interactive charts</li>
            <li>Start asking technical questions about alloy microstructure!</li>
            </ol></div>""", unsafe_allow_html=True)
            
            st.markdown("**Try asking:**")
            demo_qs = [
                "What is the typical grain size of FCC AlSi10Mg after selective laser melting?",
                "Compare phase fractions in BCC vs FCC CoCrFeNi high-entropy alloys",
                "How does aging treatment affect precipitate size in Al-Cu alloys?",
                "What is the yield strength of Ti6Al4V with equiaxed vs columnar grains?",
                "What composition range stabilizes the liquid phase in Sn-Ag-Cu solders?"
            ]
            for q in demo_qs:
                if st.button(f"💬 {q}", use_container_width=True, key=f"demo_{q[:20]}"):
                    st.session_state.demo_question = q
                    st.rerun()
    
    render_footer()
    
    # Handle demo question if set
    if hasattr(st.session_state, 'demo_question') and st.session_state.demo_question:
        st.session_state.messages.append({"role": "user", "content": st.session_state.demo_question})
        del st.session_state.demo_question
        st.rerun()

if __name__ == "__main__":
    main()
