#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LASER HEAT SOURCE RAG CHATBOT - FOCUSED RETRIEVAL & FUSION EDITION
========================================================================================
✅ Zero API keys - all models run locally (HuggingFace Transformers + Ollama)
✅ LASER-ONLY FOCUS: Heat source types, power instruments, processing parameters
✅ Narrowed scope: Maximum retrieval efficiency for laser parameter queries
✅ Information fusion: Cross-document laser parameter extraction & consensus
✅ Physics-aware validation: Energy density, power density, thermal constraints
✅ Chat-driven visualizations: Power charts, parameter space maps, pulse diagrams
✅ FAISS vector storage with laser-boosted semantic chunking
✅ Memory-efficient: 4-bit quantization, CPU/GPU auto-detection
✅ Human-readable citations: DOI, Author-Year-Journal format

DEPLOYMENT:
pip install streamlit langchain langchain-community faiss-cpu sentence-transformers
pip install transformers torch plotly pandas numpy scikit-learn
pip install pypdf2 pdf2doi crossrefapi  # optional for enhanced metadata
pip install ollama  # optional for Ollama backend

Run: streamlit run laser_source_rag.py
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
# LASER HEAT SOURCE CONFIGURATION - NARROWED SCOPE
# =============================================
LASER_SOURCE_TYPES = {
    "CW": ["continuous wave", "cw laser", "continuous-wave", "steady-state laser"],
    "PULSED": ["pulsed laser", "pulse laser", "q-switched", "mode-locked"],
    "FEMTOSECOND": ["femtosecond", "fs laser", "ultrafast", "10^-15 s", "ti:sapphire"],
    "PICOSECOND": ["picosecond", "ps laser", "10^-12 s"],
    "NANOSECOND": ["nanosecond", "ns laser", "10^-9 s", "nd:yag"],
    "MICROSECOND": ["microsecond", "μs laser", "10^-6 s"],
    "FIBER": ["fiber laser", "ytterbium fiber", "erbium fiber"],
    "CO2": ["co2 laser", "carbon dioxide laser", "10.6 μm"],
    "DIODE": ["diode laser", "semiconductor laser", "laser diode"],
    "EXCIMER": ["excimer laser", "arf", "krf", "xecl"],
    "DISK": ["disk laser", "thin-disk"],
    "SLAB": ["slab laser"],
}

POWER_INSTRUMENTS = {
    "power_meter": ["power meter", "optical power meter", "laser power meter", "wattmeter"],
    "energy_meter": ["energy meter", "joulemeter", "pulse energy meter"],
    "beam_profiler": ["beam profiler", "beam analyzer", "m² measurement"],
    "pyrometer": ["pyrometer", "infrared thermometer", "temperature sensor"],
    "thermocouple": ["thermocouple", "type-k", "type-s", "temperature probe"],
    "photodiode": ["photodiode", "fast photodiode", "detector"],
    "oscilloscope": ["oscilloscope", "digital scope", "waveform capture"],
    "spectrometer": ["spectrometer", "optical spectrum analyzer", "osa"],
    "calorimeter": ["calorimeter", "thermal power sensor"],
}

LASER_PARAMETERS = {
    "power": {
        "patterns": [
            r'(\d+(?:\.\d+)?)\s*(?:W|mW|kW|MW)\s*(?:laser\s*power|output\s*power|average\s*power|nominal\s*power)?',
            r'(?:power|P)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:W|mW|kW)',
        ],
        "unit_conversions": {"mW": 1e-3, "W": 1, "kW": 1e3, "MW": 1e6},
        "base_unit": "W",
        "physical_bounds": {"min": 1e-6, "max": 1e6},
    },
    "pulse_energy": {
        "patterns": [
            r'(\d+(?:\.\d+)?)\s*(?:J|mJ|μJ|uJ|nJ)\s*(?:pulse\s*energy|energy\s*per\s*pulse)?',
            r'(?:energy|E)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:J|mJ|μJ)',
        ],
        "unit_conversions": {"nJ": 1e-9, "μJ": 1e-6, "uJ": 1e-6, "mJ": 1e-3, "J": 1},
        "base_unit": "J",
        "physical_bounds": {"min": 1e-12, "max": 1e3},
    },
    "pulse_duration": {
        "patterns": [
            r'(\d+(?:\.\d+)?)\s*(?:fs|ps|ns|μs|us|ms)\s*(?:pulse\s*duration|pulse\s*width|pulse\s*length)?',
            r'(?:duration|τ|pulse)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:fs|ps|ns)',
        ],
        "unit_conversions": {"fs": 1e-15, "ps": 1e-12, "ns": 1e-9, "μs": 1e-6, "us": 1e-6, "ms": 1e-3},
        "base_unit": "s",
        "physical_bounds": {"min": 1e-18, "max": 1},
    },
    "repetition_rate": {
        "patterns": [
            r'(\d+(?:\.\d+)?)\s*(?:Hz|kHz|MHz|GHz)\s*(?:repetition\s*rate|pulse\s*frequency|rep\s*rate)?',
            r'(?:frequency|frep|rep)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:kHz|MHz)',
        ],
        "unit_conversions": {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9},
        "base_unit": "Hz",
        "physical_bounds": {"min": 0.1, "max": 1e11},
    },
    "wavelength": {
        "patterns": [
            r'(\d+(?:\.\d+)?)\s*(?:nm|μm|um|mm)\s*(?:wavelength|λ|lambda)?',
            r'(?:wavelength|λ)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:nm|μm)',
        ],
        "unit_conversions": {"nm": 1e-9, "μm": 1e-6, "um": 1e-6, "mm": 1e-3},
        "base_unit": "m",
        "physical_bounds": {"min": 1e-10, "max": 1e-2},
    },
    "spot_size": {
        "patterns": [
            r'(\d+(?:\.\d+)?)\s*(?:μm|um|nm|mm)\s*(?:spot\s*size|beam\s*diameter|waist|1/e²\s*diameter)?',
            r'(?:spot|diameter|waist)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:μm|mm)',
        ],
        "unit_conversions": {"nm": 1e-9, "μm": 1e-6, "um": 1e-6, "mm": 1e-3},
        "base_unit": "m",
        "physical_bounds": {"min": 1e-9, "max": 1e-1},
    },
    "fluence": {
        "patterns": [
            r'(\d+(?:\.\d+)?)\s*(?:J/cm²|J/cm2|mJ/cm²|mJ/cm2|J/m²)\s*(?:fluence|fluence\s*threshold|energy\s*density)?',
            r'(?:fluence|F)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:J/cm²|mJ/cm²)',
        ],
        "unit_conversions": {"J/cm²": 1e4, "J/cm2": 1e4, "mJ/cm²": 10, "mJ/cm2": 10, "J/m²": 1},
        "base_unit": "J/m²",
        "physical_bounds": {"min": 1e-3, "max": 1e6},
    },
    "power_density": {
        "patterns": [
            r'(\d+(?:\.\d+)?)\s*(?:W/cm²|W/cm2|kW/cm²|MW/cm²|W/m²)\s*(?:power\s*density|intensity|irradiance)?',
            r'(?:intensity|I)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:W/cm²|MW/cm²)',
        ],
        "unit_conversions": {"W/cm²": 1e4, "W/cm2": 1e4, "kW/cm²": 1e7, "MW/cm²": 1e10, "W/m²": 1},
        "base_unit": "W/m²",
        "physical_bounds": {"min": 1, "max": 1e15},
    },
    "scan_speed": {
        "patterns": [
            r'(\d+(?:\.\d+)?)\s*(?:mm/s|mm/min|m/s|cm/s)\s*(?:scan\s*speed|travel\s*speed|writing\s*speed)?',
            r'(?:speed|v)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:mm/s|m/s)',
        ],
        "unit_conversions": {"mm/s": 1e-3, "mm/min": 1.667e-5, "m/s": 1, "cm/s": 1e-2},
        "base_unit": "m/s",
        "physical_bounds": {"min": 1e-6, "max": 1e3},
    },
    "hatch_distance": {
        "patterns": [
            r'(\d+(?:\.\d+)?)\s*(?:μm|um|mm)\s*(?:hatch\s*distance|hatch\s*spacing|line\s*spacing)?',
            r'(?:hatch|spacing)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:μm|mm)',
        ],
        "unit_conversions": {"μm": 1e-6, "um": 1e-6, "mm": 1e-3},
        "base_unit": "m",
        "physical_bounds": {"min": 1e-9, "max": 1e-2},
    },
    "overlap": {
        "patterns": [
            r'(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:overlap|pulse\s*overlap|spatial\s*overlap)?',
            r'(?:overlap)\s*[=:]\s*(\d+(?:\.\d+)?)\s*%',
        ],
        "unit_conversions": {"%": 1, "percent": 1},
        "base_unit": "%",
        "physical_bounds": {"min": 0, "max": 100},
    },
    "dwell_time": {
        "patterns": [
            r'(\d+(?:\.\d+)?)\s*(?:μs|us|ms|ns|s)\s*(?:dwell\s*time|exposure\s*time|interaction\s*time)?',
            r'(?:dwell|exposure)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:μs|ms)',
        ],
        "unit_conversions": {"ns": 1e-9, "μs": 1e-6, "us": 1e-6, "ms": 1e-3, "s": 1},
        "base_unit": "s",
        "physical_bounds": {"min": 1e-15, "max": 1e3},
    },
}

LASER_PROCESSING_MODES = {
    "ablation": ["ablation", "material removal", "laser ablation", "vaporization"],
    "melting": ["melting", "fusion", "melt pool", "liquid phase"],
    "sintering": ["sintering", "partial melting", "powder consolidation"],
    "annealing": ["annealing", "heat treatment", "thermal treatment"],
    "hardening": ["hardening", "surface hardening", "laser hardening"],
    "cladding": ["cladding", "laser cladding", "additive deposition"],
    "marking": ["marking", "engraving", "surface texturing"],
    "cutting": ["cutting", "laser cutting", "kerf"],
    "welding": ["welding", "laser welding", "fusion welding"],
    "surface_modification": ["surface modification", "lipss", "ripples", "periodic structures"],
}

# Laser-specific keywords for retrieval boosting (SCOPE: LASER ONLY)
LASER_KEYWORDS = {
    "source_type": list(LASER_SOURCE_TYPES.keys()) + [kw for v in LASER_SOURCE_TYPES.values() for kw in v],
    "power_instrument": list(POWER_INSTRUMENTS.keys()) + [kw for v in POWER_INSTRUMENTS.values() for kw in v],
    "parameter": list(LASER_PARAMETERS.keys()),
    "processing_mode": list(LASER_PROCESSING_MODES.keys()) + [kw for v in LASER_PROCESSING_MODES.values() for kw in v],
    "beam_quality": ["m²", "m2", "beam quality", "m-squared", "beam parameter product"],
    "thermal_effects": ["heat affected zone", "haz", "thermal diffusion", "heat conduction", "thermal lensing"],
    "measurement": ["calibration", "measurement uncertainty", "traceability", "standard"],
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

LASER_DOMAIN_CONFIG = {
    "chunk_size": 600,
    "chunk_overlap": 100,
    "retrieval_k": 4,
    "score_threshold": 0.30,
    "max_context_tokens": 1024,
    "max_new_tokens": 256,
    "temperature": 0.1,
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
class LaserParameter:
    """Laser-specific parameter with source context"""
    name: str
    value: Union[float, str, List]
    unit: Optional[str] = None
    uncertainty: Optional[str] = None
    condition: Optional[str] = None
    source_chunk_id: str = ""
    source_citation: str = ""
    extraction_confidence: float = 0.5
    context_snippet: str = ""
    parameter_type: str = "operational"
    laser_source_type: Optional[str] = None
    processing_mode: Optional[str] = None
    normalized_name: str = ""
    normalized_value: Optional[float] = None
    normalized_unit: Optional[str] = None

    def __post_init__(self):
        if not self.normalized_name:
            self.normalized_name = self._normalize_parameter_name(self.name)
        if self.normalized_value is None and isinstance(self.value, (int, float)):
            self.normalized_value = self.value

    def _normalize_parameter_name(self, name: str) -> str:
        synonym_map = {
            "laser power": "power", "output power": "power", "average power": "power",
            "peak power": "peak_power", "pulse energy": "pulse_energy",
            "energy per pulse": "pulse_energy", "pulse duration": "pulse_duration",
            "pulse width": "pulse_duration", "pulse length": "pulse_duration",
            "repetition rate": "repetition_rate", "pulse frequency": "repetition_rate",
            "rep rate": "repetition_rate", "wavelength": "wavelength",
            "laser wavelength": "wavelength", "spot size": "spot_size",
            "beam diameter": "spot_size", "beam waist": "spot_size",
            "1/e2 diameter": "spot_size", "fluence": "fluence",
            "energy density": "fluence", "fluence threshold": "fluence_threshold",
            "power density": "power_density", "intensity": "power_density",
            "irradiance": "power_density", "scan speed": "scan_speed",
            "travel speed": "scan_speed", "writing speed": "scan_speed",
            "hatch distance": "hatch_distance", "hatch spacing": "hatch_distance",
            "line spacing": "hatch_distance", "overlap": "overlap",
            "pulse overlap": "overlap", "dwell time": "dwell_time",
            "exposure time": "dwell_time", "interaction time": "dwell_time",
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
class LaserDocumentRecord:
    """Record of extracted laser parameters from a document chunk"""
    source_filename: str
    chunk_index: int
    chunk_id: str
    bibliographic_citation: str
    extracted_parameters: List[LaserParameter] = field(default_factory=list)
    laser_topics: List[str] = field(default_factory=list)
    experimental_setup: Dict[str, Any] = field(default_factory=dict)
    laser_source_type: Optional[str] = None
    processing_mode: Optional[str] = None
    power_instruments_used: List[str] = field(default_factory=list)

    def add_parameter(self, param: LaserParameter):
        self.extracted_parameters.append(param)

    def get_parameters_by_name(self, param_name: str) -> List[LaserParameter]:
        normalized = LaserParameter("", "")._normalize_parameter_name(param_name)
        return [p for p in self.extracted_parameters if p.normalized_name == normalized]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "citation": self.bibliographic_citation,
            "laser_source_type": self.laser_source_type,
            "processing_mode": self.processing_mode,
            "parameters": [p.to_dict() for p in self.extracted_parameters],
            "topics": self.laser_topics,
            "instruments": self.power_instruments_used,
            "setup": self.experimental_setup
        }

@dataclass
class FusedLaserParameter:
    """Fused laser parameter entry with statistical aggregation"""
    parameter_name: str
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
    laser_source_type: Optional[str] = None
    processing_mode: Optional[str] = None
    fusion_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_comparison_row(self) -> Dict[str, Any]:
        return {
            "parameter": self.parameter_name,
            "value": self.fused_value,
            "unit": self.unit,
            "range": f"{self.value_range[0]:.2f}–{self.value_range[1]:.2f}" if self.value_range else None,
            "std": f"{self.standard_deviation:.3f}" if self.standard_deviation else None,
            "sources": len(self.sources),
            "confidence": self.fusion_confidence.value,
            "laser_type": self.laser_source_type,
            "processing_mode": self.processing_mode,
            "conditions": self.conditions_summary
        }

@dataclass
class FusionEfficiencyMetrics:
    """Metrics for evaluating laser parameter fusion quality"""
    unique_sources_used: int = 0
    source_diversity_score: float = 0.0
    total_parameters_extracted: int = 0
    parameters_fused_successfully: int = 0
    parameter_coverage_ratio: float = 0.0
    consistent_parameters: int = 0
    conflicting_parameters: int = 0
    consistency_ratio: float = 0.0
    numeric_parameters_with_uncertainty: int = 0
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
            "parameter_coverage": 0.20,
            "consistency": 0.25,
            "precision": 0.15,
            "confidence": 0.15,
            "specificity": 0.10
        }
        if self.total_parameters_extracted == 0:
            self.overall_fusion_efficiency = self.source_diversity_score * 0.3
            return self.overall_fusion_efficiency
        components = [
            self.source_diversity_score * weights["source_diversity"],
            self.parameter_coverage_ratio * weights["parameter_coverage"],
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
            "🔍 Parameters": f"{self.parameters_fused_successfully}/{max(self.total_parameters_extracted, 1)}",
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
# LASER PARAMETER EXTRACTION ENGINE
# =============================================
class LaserParameterExtractor:
    """Extract laser heat source parameters focused on power, instruments, processing"""

    def __init__(self, laser_keywords: Dict[str, List[str]]):
        self.laser_keywords = laser_keywords
        self._compile_extraction_patterns()

    def _compile_extraction_patterns(self):
        numeric_pattern = r'([\d.]+(?:\s*[×x*]\s*10\^?-?\d+)?)(?:\s*([±\+-])\s*([\d.]+))?'
        all_units = []
        for param_config in LASER_PARAMETERS.values():
            all_units.extend(param_config["unit_conversions"].keys())
        unit_pattern = r'\s*(' + '|'.join(re.escape(u) for u in all_units) + r')'
        self.property_pattern = re.compile(
            r'([\w\s\-_/]+?)\s*(?:is|was|of|at|:|=|≈|~|yields|results in|measured as|set to|configured at)\s*' +
            numeric_pattern + unit_pattern +
            r'(?:\s*[\(\[]([^)\]]+)[\)\]])?', re.I)
        self.table_row_pattern = re.compile(r'(?:^|\n)\s*[|│]?\s*([^|\n│]+?)\s*[|│]?\s*(?:\n|$)', re.MULTILINE)
        source_keywords = [kw for v in LASER_SOURCE_TYPES.values() for kw in v]
        self.source_type_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(kw) for kw in source_keywords) + r')\b', re.I)
        mode_keywords = [kw for v in LASER_PROCESSING_MODES.values() for kw in v]
        self.processing_mode_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(kw) for kw in mode_keywords) + r')\b', re.I)
        instrument_keywords = [kw for v in POWER_INSTRUMENTS.values() for kw in v]
        self.instrument_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(kw) for kw in instrument_keywords) + r')\b', re.I)

    def extract_parameters_from_chunk(self, chunk_text: str, chunk_metadata: Dict[str, Any]) -> LaserDocumentRecord:
        record = LaserDocumentRecord(
            source_filename=chunk_metadata.get('source', 'unknown'),
            chunk_index=chunk_metadata.get('chunk_index', 0),
            chunk_id=f"{chunk_metadata.get('source', 'unknown')}:{chunk_metadata.get('chunk_index', 0)}",
            bibliographic_citation=chunk_metadata.get('citation_display', 'Unknown'),
            laser_topics=chunk_metadata.get('laser_topics', []),
            experimental_setup=chunk_metadata.get('parameters_found', {}),
            laser_source_type=self._detect_laser_source_type(chunk_text),
            processing_mode=self._detect_processing_mode(chunk_text),
            power_instruments_used=self._detect_power_instruments(chunk_text)
        )
        table_params = self._extract_from_tables(chunk_text)
        for param in table_params:
            param.source_chunk_id = record.chunk_id
            param.source_citation = record.bibliographic_citation
            param.laser_source_type = record.laser_source_type
            param.processing_mode = record.processing_mode
            record.add_parameter(param)
        inline_params = self._extract_inline_parameters(chunk_text)
        for param in inline_params:
            if not any(p.normalized_name == param.normalized_name and
                      (abs(p.normalized_value - param.normalized_value) < 1e-6 if p.normalized_value and param.normalized_value else False)
                      for p in record.extracted_parameters):
                param.source_chunk_id = record.chunk_id
                param.source_citation = record.bibliographic_citation
                record.add_parameter(param)
        return record

    def _detect_laser_source_type(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        for source_type, keywords in LASER_SOURCE_TYPES.items():
            if any(kw.lower() in text_lower for kw in keywords):
                return source_type
        return None

    def _detect_processing_mode(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        for mode, keywords in LASER_PROCESSING_MODES.items():
            if any(kw.lower() in text_lower for kw in keywords):
                return mode
        return None

    def _detect_power_instruments(self, text: str) -> List[str]:
        instruments = []
        text_lower = text.lower()
        for instrument_type, keywords in POWER_INSTRUMENTS.items():
            if any(kw.lower() in text_lower for kw in keywords):
                instruments.append(instrument_type)
        return instruments

    def _extract_from_tables(self, text: str) -> List[LaserParameter]:
        parameters = []
        if r'\begin{tabular}' in text or r'\begin{table}' in text:
            parameters.extend(self._parse_latex_table(text))
        elif '|' in text and re.search(r'\|\s*[-:]+\s*\|', text):
            parameters.extend(self._parse_markdown_table(text))
        elif self._detect_plain_text_table(text):
            parameters.extend(self._parse_plain_text_table(text))
        return parameters

    def _parse_latex_table(self, latex_text: str) -> List[LaserParameter]:
        parameters = []
        table_match = re.search(r'\\begin\{tabular\}.*?\\end\{tabular\}', latex_text, re.DOTALL)
        if not table_match:
            return parameters
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
            return parameters
        header_map = {h.lower().strip(): i for i, h in enumerate(header_row)}
        param_keywords = list(LASER_PARAMETERS.keys()) + ['power', 'energy', 'duration', 'fluence', 'wavelength', 'spot']
        param_cols = [i for i, h in enumerate(header_row) if any(kw in h.lower() for kw in param_keywords)]
        descriptor_cols = [i for i in range(len(header_row)) if i not in param_cols]
        for row in data_rows:
            if len(row) <= max(param_cols, default=-1):
                continue
            row_conditions = {}
            for col_idx in descriptor_cols:
                if col_idx < len(row) and row[col_idx]:
                    cell = row[col_idx].strip()
                    for source_type, keywords in LASER_SOURCE_TYPES.items():
                        if any(kw.lower() in cell.lower() for kw in keywords):
                            row_conditions['laser_type'] = source_type
                            break
                    for mode, keywords in LASER_PROCESSING_MODES.items():
                        if any(kw.lower() in cell.lower() for kw in keywords):
                            row_conditions['processing_mode'] = mode
                            break
            for prop_col in param_cols:
                if prop_col >= len(row) or not row[prop_col].strip():
                    continue
                prop_name = header_row[prop_col].strip()
                prop_value_raw = row[prop_col].strip()
                parsed = self._parse_parameter_value(prop_value_raw, prop_name)
                if parsed:
                    param = LaserParameter(
                        name=prop_name, value=parsed['value'], unit=parsed['unit'],
                        uncertainty=parsed['uncertainty'], condition=self._format_conditions(row_conditions),
                        extraction_confidence=0.85, context_snippet=prop_value_raw, parameter_type="measurement"
                    )
                    self._normalize_parameter_units(param)
                    parameters.append(param)
        return parameters

    def _parse_markdown_table(self, text: str) -> List[LaserParameter]:
        parameters = []
        lines = [l.strip() for l in text.split('\n') if '|' in l and l.strip()]
        if len(lines) < 3:
            return parameters
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
                if any(kw in header.lower() for kw in list(LASER_PARAMETERS.keys()) + ['power', 'energy', 'fluence', 'wavelength']):
                    parsed = self._parse_parameter_value(value, header)
                    if parsed:
                        param = LaserParameter(
                            name=header, value=parsed['value'], unit=parsed['unit'],
                            uncertainty=parsed['uncertainty'], condition=row_data.get('Laser') or row_data.get('Mode'),
                            extraction_confidence=0.8, context_snippet=value, parameter_type="measurement"
                        )
                        self._normalize_parameter_units(param)
                        parameters.append(param)
        return parameters

    def _parse_plain_text_table(self, text: str) -> List[LaserParameter]:
        parameters = []
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
                            param = LaserParameter(
                                name=prop_name, value=float(value_match.group(1)),
                                extraction_confidence=0.6, context_snippet=line[:100], parameter_type="observation"
                            )
                            parameters.append(param)
                        except (ValueError, TypeError):
                            continue
        return parameters

    def _extract_inline_parameters(self, text: str) -> List[LaserParameter]:
        parameters = []
        for match in self.property_pattern.finditer(text):
            groups = match.groups()
            if len(groups) >= 5 and groups[1]:
                prop_name = groups[0].strip()
                value_str = groups[1].strip()
                uncertainty = f"{groups[2]}{groups[3]}" if groups[2] and groups[3] else None
                unit = groups[4].strip() if groups[4] else None
                condition = groups[5].strip() if len(groups) > 5 and groups[5] else None
                numeric_value = self._safe_parse_numeric(value_str)
                param = LaserParameter(
                    name=prop_name, value=numeric_value if numeric_value is not None else value_str,
                    unit=unit, uncertainty=uncertainty, condition=condition,
                    extraction_confidence=0.7, context_snippet=match.group(0)[:150],
                    parameter_type="parameter" if any(kw in prop_name.lower() for kw in LASER_PARAMETERS.keys()) else "observation"
                )
                self._normalize_parameter_units(param)
                parameters.append(param)
        return parameters

    def _parse_parameter_value(self, raw_value: str, param_name: str) -> Optional[Dict[str, Any]]:
        if not raw_value or raw_value.strip() in ['-', '.', '', 'N/A', 'n/a', 'NA', 'na', '--', '...']:
            return None
        result = {"value": None, "unit": None, "uncertainty": None}
        uncertainty_match = re.search(r'([±\+-])\s*([\d.]+)', raw_value)
        if uncertainty_match:
            result["uncertainty"] = f"{uncertainty_match.group(1)}{uncertainty_match.group(2)}"
            raw_value = raw_value.replace(uncertainty_match.group(0), '').strip()
        for unit in sorted(LASER_PARAMETERS.get(param_name, {}).get("unit_conversions", {}).keys(), key=len, reverse=True):
            if raw_value.lower().endswith(unit.lower()):
                result["unit"] = unit
                raw_value = raw_value[:-len(unit)].strip()
                break
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

    def _normalize_parameter_units(self, param: LaserParameter):
        param_config = LASER_PARAMETERS.get(param.normalized_name)
        if not param_config or not param.unit or param.unit not in param_config["unit_conversions"]:
            param.normalized_unit = param.unit
            if isinstance(param.value, (int, float)):
                param.normalized_value = param.value
            elif param.normalized_value is None and isinstance(param.value, str):
                param.normalized_value = self._safe_parse_numeric(param.value)
            return
        conversion = param_config["unit_conversions"][param.unit]
        if isinstance(param.value, (int, float)):
            param.normalized_value = param.value * conversion
            param.normalized_unit = param_config["base_unit"]
        elif param.normalized_value is not None:
            param.normalized_value = param.normalized_value * conversion
            param.normalized_unit = param_config["base_unit"]
        else:
            param.normalized_unit = param.unit

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
# LASER PARAMETER FUSION ENGINE
# =============================================
class LaserFusionEngine:
    """Fuse laser heat source parameters across documents"""

    def __init__(self, parameter_extractor: LaserParameterExtractor):
        self.extractor = parameter_extractor
        self.fusion_history: List[Dict] = []

    def fuse_laser_documents(self, retrieved_docs: List[Document], query: str,
                          source_type_filter: Optional[str] = None,
                          processing_mode_filter: Optional[str] = None,
                          parameter_filter: Optional[List[str]] = None) -> Tuple[Dict[str, FusedLaserParameter], FusionEfficiencyMetrics]:
        fusion_records: List[LaserDocumentRecord] = []
        for doc in retrieved_docs:
            record = self.extractor.extract_parameters_from_chunk(doc.page_content, doc.metadata)
            if source_type_filter and record.laser_source_type != source_type_filter:
                continue
            if processing_mode_filter and record.processing_mode != processing_mode_filter:
                continue
            if parameter_filter:
                record.extracted_parameters = [p for p in record.extracted_parameters if p.normalized_name in parameter_filter]
            if record.extracted_parameters:
                fusion_records.append(record)
        if not fusion_records:
            metrics = FusionEfficiencyMetrics(
                unique_sources_used=len(retrieved_docs),
                source_diversity_score=min(1.0, len(retrieved_docs) / 3.0),
                overall_fusion_efficiency=min(1.0, len(retrieved_docs) / 3.0) * 0.3
            )
            return {}, metrics
        parameter_groups: Dict[str, List[LaserParameter]] = defaultdict(list)
        for record in fusion_records:
            for param in record.extracted_parameters:
                key = param.normalized_name
                if not parameter_filter or key in parameter_filter:
                    parameter_groups[key].append(param)
        fused_parameters: Dict[str, FusedLaserParameter] = {}
        for param_name, params in parameter_groups.items():
            fused = self._fuse_parameter_group(param_name, params)
            if fused:
                fused_parameters[param_name] = fused
        metrics = self._compute_fusion_metrics(fusion_records, fused_parameters, retrieved_docs, query)
        self.fusion_history.append({
            "timestamp": datetime.now().isoformat(), "query": query,
            "input_docs": len(retrieved_docs),
            "extracted_parameters": sum(len(r.extracted_parameters) for r in fusion_records),
            "fused_parameters": len(fused_parameters),
            "efficiency": metrics.overall_fusion_efficiency
        })
        return fused_parameters, metrics

    def _fuse_parameter_group(self, param_name: str, parameters: List[LaserParameter]) -> Optional[FusedLaserParameter]:
        if not parameters:
            return None
        numeric_params = [p for p in parameters if p.normalized_value is not None and isinstance(p.normalized_value, (int, float))]
        fused = FusedLaserParameter(
            parameter_name=param_name,
            fused_value=None,
            unit=parameters[0].normalized_unit if parameters[0].normalized_unit else parameters[0].unit,
            source_count=len(parameters),
            sources=[{"citation": p.source_citation, "chunk_id": p.source_chunk_id} for p in parameters],
            laser_source_type=parameters[0].laser_source_type if parameters else None,
            processing_mode=parameters[0].processing_mode if parameters else None
        )
        if numeric_params and len(numeric_params) >= 1:
            values = [p.normalized_value for p in numeric_params if p.normalized_value is not None]
            if values:
                fused.fused_value = np.mean(values)
                fused.value_range = (min(values), max(values))
                fused.standard_deviation = np.std(values) if len(values) > 1 else 0.0
                if fused.fused_value != 0:
                    cv = fused.standard_deviation / abs(fused.fused_value)
                else:
                    cv = 1.0
                if cv < 0.1 and len(numeric_params) >= 2:
                    fused.fusion_confidence = FusionConfidence.HIGH
                elif cv < 0.3 or len(numeric_params) == 1:
                    fused.fusion_confidence = FusionConfidence.MODERATE
                else:
                    fused.fusion_confidence = FusionConfidence.LOW
                    fused.conflicts_detected = True
                    fused.conflict_notes.append(f"High variation: CV={cv:.2f}")
                conditions = defaultdict(set)
                for p in numeric_params:
                    if p.condition:
                        conditions["context"].add(p.condition)
                    if p.experimental_setup:
                        for k, v in p.experimental_setup.items():
                            conditions[k].add(str(v))
                fused.conditions_summary = {k: list(v) for k, v in conditions.items()}
        else:
            value_counts = Counter(str(p.value) for p in parameters if p.value is not None)
            if value_counts:
                fused.fused_value = value_counts.most_common(1)[0][0]
                fused.fusion_confidence = (
                    FusionConfidence.HIGH if value_counts.most_common(1)[0][1] == len(parameters)
                    else FusionConfidence.MODERATE if value_counts.most_common(1)[0][1] > len(parameters) / 2
                    else FusionConfidence.LOW
                )
                if fused.fusion_confidence == FusionConfidence.LOW:
                    fused.conflicts_detected = True
                    fused.conflict_notes.append(f"Multiple distinct values: {list(value_counts.keys())[:3]}")
        return fused

    def _compute_fusion_metrics(self, fusion_records: List[LaserDocumentRecord],
                               fused_parameters: Dict[str, FusedLaserParameter],
                               retrieved_docs: List[Document], query: str) -> FusionEfficiencyMetrics:
        metrics = FusionEfficiencyMetrics()
        unique_sources = set(r.chunk_id for r in fusion_records)
        metrics.unique_sources_used = len(unique_sources)
        metrics.source_diversity_score = min(1.0, len(unique_sources) / 3.0)
        total_extracted = sum(len(r.extracted_parameters) for r in fusion_records)
        metrics.total_parameters_extracted = total_extracted
        metrics.parameters_fused_successfully = len(fused_parameters)
        metrics.parameter_coverage_ratio = len(fused_parameters) / total_extracted if total_extracted > 0 else 0.0
        if fused_parameters:
            consistent = sum(1 for f in fused_parameters.values() if not f.conflicts_detected and f.fusion_confidence != FusionConfidence.LOW)
            conflicting = sum(1 for f in fused_parameters.values() if f.conflicts_detected)
            total_evaluated = consistent + conflicting
            metrics.consistent_parameters = consistent
            metrics.conflicting_parameters = conflicting
            metrics.consistency_ratio = consistent / total_evaluated if total_evaluated > 0 else 1.0
        else:
            metrics.consistency_ratio = 1.0
        numeric_with_uncertainty = [f for f in fused_parameters.values() if f.standard_deviation is not None or any("±" in str(s.get("citation", "")) for s in f.sources)]
        metrics.numeric_parameters_with_uncertainty = len(numeric_with_uncertainty)
        if fused_parameters:
            uncertainties = []
            for f in fused_parameters.values():
                if isinstance(f.fused_value, (int, float)) and f.fused_value != 0 and f.standard_deviation is not None:
                    uncertainties.append(f.standard_deviation / abs(f.fused_value))
            if uncertainties:
                metrics.average_uncertainty_magnitude = np.mean(uncertainties)
            else:
                metrics.average_uncertainty_magnitude = 0.1
        else:
            metrics.average_uncertainty_magnitude = 0.1
        confidence_weights = {FusionConfidence.HIGH: 1.0, FusionConfidence.MODERATE: 0.7, FusionConfidence.LOW: 0.4, FusionConfidence.UNKNOWN: 0.2}
        if fused_parameters:
            weighted_sum = sum(confidence_weights.get(f.fusion_confidence, 0.5) for f in fused_parameters.values())
            metrics.weighted_confidence_score = weighted_sum / len(fused_parameters)
            metrics.high_confidence_fusions = sum(1 for f in fused_parameters.values() if f.fusion_confidence == FusionConfidence.HIGH)
            metrics.low_confidence_fusions = sum(1 for f in fused_parameters.values() if f.fusion_confidence == FusionConfidence.LOW)
        else:
            metrics.weighted_confidence_score = 0.5
        metrics.answer_specificity_score = self._estimate_answer_specificity(query, fused_parameters)
        metrics.citation_density = min(1.0, len(fused_parameters) * 2 / 100)
        metrics.compute_overall()
        return metrics

    def _estimate_answer_specificity(self, query: str, fused_params: Dict[str, FusedLaserParameter]) -> float:
        if not fused_params:
            query_lower = query.lower()
            if any(kw in query_lower for kw in ['compare', 'versus', 'vs', 'difference', 'power', 'fluence', 'pulse', 'wavelength']):
                return 0.5
            return 0.3
        query_lower = query.lower()
        specificity_indicators = 0
        for param_name in fused_params.keys():
            if param_name.replace('_', ' ') in query_lower or param_name in query_lower:
                specificity_indicators += 2
        if any(src in query_lower for src in list(LASER_SOURCE_TYPES.keys())):
            specificity_indicators += 1
        if any(param in query_lower for param in ['power', 'fluence', 'pulse_duration', 'wavelength', 'spot_size']):
            specificity_indicators += 1
        if re.search(r'[\d.]+\s*(?:W|mW|J|mJ|fs|ps|ns|nm|μm|J/cm²|W/cm²)', query_lower):
            specificity_indicators += 2
        return min(1.0, specificity_indicators / 5.0)

    def generate_comparison_table(self, fused_parameters: Dict[str, FusedLaserParameter], format: str = "markdown") -> str:
        if not fused_parameters:
            return "_No laser parameters available for comparison_"
        if format == "markdown":
            return self._generate_markdown_table(fused_parameters)
        elif format == "latex":
            return self._generate_latex_table(fused_parameters)
        elif format == "html":
            return self._generate_html_table(fused_parameters)
        else:
            return self._generate_plain_text_table(fused_parameters)

    def _generate_markdown_table(self, fused_params: Dict[str, FusedLaserParameter]) -> str:
        lines = []
        lines.append("| Parameter | Value | Unit | Range | Sources | Confidence | Laser Type |")
        lines.append("|-----------|-------|------|-------|---------|------------|------------|")
        for param_name, entry in sorted(fused_params.items(), key=lambda x: x[0]):
            if entry.fused_value is not None and isinstance(entry.fused_value, (int, float)):
                value_str = f"{entry.fused_value:.3f}"
                if entry.standard_deviation is not None:
                    value_str = f"{value_str} ± {entry.standard_deviation:.3f}"
            else:
                value_str = str(entry.fused_value) if entry.fused_value is not None else "–"
            range_str = f"{entry.value_range[0]:.2f}–{entry.value_range[1]:.2f}" if entry.value_range else "–"
            confidence_icon = {"high": "🟢", "moderate": "🟡", "low": "🔴", "unknown": "⚪"}.get(entry.fusion_confidence.value, "⚪")
            laser_type = entry.laser_source_type or "–"
            lines.append(f"| {param_name.replace('_', ' ').title()} | {value_str} | {entry.unit or '–'} | {range_str} | {entry.source_count} | {confidence_icon} {entry.fusion_confidence.value} | {laser_type} |")
        return "\n".join(lines)

    def _generate_latex_table(self, fused_params: Dict[str, FusedLaserParameter]) -> str:
        lines = [r"\begin{tabular}{|l|c|c|c|c|c|l|}", r"\hline",
                r"\textbf{Parameter} & \textbf{Value} & \textbf{Unit} & \textbf{Range} & \textbf{Sources} & \textbf{Confidence} & \textbf{Laser Type} \\", r"\hline"]
        for param_name, entry in fused_params.items():
            if entry.fused_value is not None and isinstance(entry.fused_value, (int, float)):
                value_str = f"{entry.fused_value:.3f}"
                if entry.standard_deviation is not None:
                    value_str = f"{value_str} \\pm {entry.standard_deviation:.3f}"
            else:
                value_str = str(entry.fused_value) if entry.fused_value is not None else "--"
            range_str = f"{entry.value_range[0]:.2f}--{entry.value_range[1]:.2f}" if entry.value_range else "--"
            conf_symbol = {"high": "high", "moderate": "mod", "low": "low"}.get(entry.fusion_confidence.value, "?")
            laser_type = entry.laser_source_type or "--"
            lines.append(f"{param_name.replace('_', r'\_').title()} & {value_str} & {entry.unit or '--'} & {range_str} & {entry.source_count} & {conf_symbol} & {laser_type} \\\\")
        lines.extend([r"\hline", r"\end{tabular}"])
        return "\n".join(lines)

    def _generate_html_table(self, fused_params: Dict[str, FusedLaserParameter]) -> str:
        lines = ['<table class="fusion-table" style="border-collapse: collapse; width: 100%;">']
        lines.append('<thead><tr style="background: #f0f9ff;">')
        for header in ["Parameter", "Value", "Unit", "Range", "Sources", "Confidence", "Laser Type"]:
            lines.append(f'<th style="border: 1px solid #ccc; padding: 8px; text-align: left;">{header}</th>')
        lines.append('</tr></thead><tbody>')
        for param_name, entry in fused_params.items():
            if entry.fused_value is not None and isinstance(entry.fused_value, (int, float)):
                value_str = f"{entry.fused_value:.3f}"
                if entry.standard_deviation is not None:
                    value_str = f"{value_str} ± {entry.standard_deviation:.3f}"
            else:
                value_str = str(entry.fused_value) if entry.fused_value is not None else "–"
            bg_color = {"high": "#dcfce7", "moderate": "#fef3c7", "low": "#fee2e2"}.get(entry.fusion_confidence.value, "#f1f5f9")
            lines.append(f'<tr style="background: {bg_color};">')
            lines.append(f'<td style="border: 1px solid #ccc; padding: 8px;">{param_name.replace("_", " ").title()}</td>')
            lines.append(f'<td style="border: 1px solid #ccc; padding: 8px;">{value_str}</td>')
            lines.append(f'<td style="border: 1px solid #ccc; padding: 8px;">{entry.unit or "–"}</td>')
            range_display = f"{entry.value_range[0]:.2f}–{entry.value_range[1]:.2f}" if entry.value_range else "–"
            lines.append(f'<td style="border: 1px solid #ccc; padding: 8px;">{range_display}</td>')
            lines.append(f'<td style="border: 1px solid #ccc; padding: 8px; text-align: center;">{entry.source_count}</td>')
            lines.append(f'<td style="border: 1px solid #ccc; padding: 8px;">{entry.fusion_confidence.value.title()}</td>')
            laser_type = entry.laser_source_type or "–"
            lines.append(f'<td style="border: 1px solid #ccc; padding: 8px;">{laser_type}</td>')
            lines.append('</tr>')
        lines.append('</tbody></table>')
        return "\n".join(lines)

    def _generate_plain_text_table(self, fused_params: Dict[str, FusedLaserParameter]) -> str:
        lines = []
        lines.append(f"{'Parameter':<30} {'Value':<15} {'Unit':<10} {'Confidence':<10} {'Laser Type':<15}")
        lines.append("-" * 85)
        for param_name, entry in fused_params.items():
            if entry.fused_value is not None and isinstance(entry.fused_value, (int, float)):
                value_str = f"{entry.fused_value:.3f}"
            else:
                value_str = str(entry.fused_value) if entry.fused_value is not None else "–"
            laser_type = entry.laser_source_type or "–"
            lines.append(f"{param_name.replace('_', ' ').title():<30} {value_str:<15} {entry.unit or '–':<10} {entry.fusion_confidence.value:<10} {laser_type:<15}")
        return "\n".join(lines)

# =============================================
# LASER PHYSICS VALIDATOR
# =============================================
class LaserPhysicsValidator:
    """Validate laser parameters against physical constraints"""

    def __init__(self):
        self.violation_log: List[Dict] = []

    def check_parameter_bounds(self, param_name: str, value: float, unit: str) -> Dict:
        param_config = LASER_PARAMETERS.get(param_name)
        if not param_config:
            return {"valid": True, "message": "No bounds defined for this parameter"}
        bounds = param_config["physical_bounds"]
        if value < bounds["min"] or value > bounds["max"]:
            violation = {
                "type": "physical_bound_violation",
                "parameter": param_name,
                "value": value,
                "unit": unit,
                "bounds": bounds,
                "severity": "HIGH" if value < 0 or value > bounds["max"] * 2 else "MEDIUM"
            }
            self.violation_log.append(violation)
            return {"valid": False, "violation": violation}
        return {"valid": True, "message": "Value within physical bounds"}

    def check_energy_consistency(self, power: Optional[float], pulse_energy: Optional[float],
                                pulse_duration: Optional[float], repetition_rate: Optional[float]) -> Dict:
        violations = []
        if power is not None and pulse_energy is not None and repetition_rate is not None:
            calculated_power = pulse_energy * repetition_rate
            if abs(calculated_power - power) / power > 0.1 and power > 0:
                violations.append({
                    "type": "energy_consistency",
                    "rule": "P_avg = E_pulse × f_rep",
                    "calculated": calculated_power,
                    "reported": power,
                    "severity": "MEDIUM"
                })
        if pulse_energy is not None and pulse_duration is not None:
            peak_power = pulse_energy / pulse_duration if pulse_duration > 0 else None
            if peak_power is not None and peak_power > 1e15:
                violations.append({
                    "type": "extreme_peak_power",
                    "parameter": "peak_power_estimate",
                    "value": peak_power,
                    "severity": "LOW"
                })
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "message": "Energy consistency check passed" if not violations else f"{len(violations)} inconsistency(ies) found"
        }

    def check_fluence_power_density_consistency(self, fluence: Optional[float], power_density: Optional[float],
                                               pulse_duration: Optional[float]) -> Dict:
        if fluence is None or power_density is None or pulse_duration is None or pulse_duration <= 0:
            return {"valid": True, "message": "Insufficient data for consistency check"}
        calculated_intensity = fluence / pulse_duration
        if abs(calculated_intensity - power_density) / power_density > 0.2 and power_density > 0:
            return {
                "valid": False,
                "message": f"Intensity mismatch: I=F/τ gives {calculated_intensity:.2e} W/m² vs reported {power_density:.2e} W/m²",
                "severity": "MEDIUM"
            }
        return {"valid": True, "message": "Fluence-intensity consistency verified"}

    def full_validation(self, fused_parameters: Dict[str, FusedLaserParameter]) -> Dict:
        results = {
            "total_checks": 0,
            "passed": 0,
            "failed": 0,
            "violations": [],
            "warnings": []
        }
        param_values = {}
        for param_name, param in fused_parameters.items():
            if isinstance(param.fused_value, (int, float)):
                param_values[param_name] = param.fused_value
        for param_name, param in fused_parameters.items():
            if isinstance(param.fused_value, (int, float)) and param.unit:
                results["total_checks"] += 1
                check = self.check_parameter_bounds(param_name, param.fused_value, param.unit)
                if check["valid"]:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["violations"].append(check.get("violation", {}))
        energy_check = self.check_energy_consistency(
            param_values.get("power"),
            param_values.get("pulse_energy"),
            param_values.get("pulse_duration"),
            param_values.get("repetition_rate")
        )
        if not energy_check["valid"]:
            results["total_checks"] += 1
            results["failed"] += 1
            results["warnings"].extend(energy_check.get("violations", []))
        else:
            results["total_checks"] += 1
            results["passed"] += 1
        fluence_check = self.check_fluence_power_density_consistency(
            param_values.get("fluence"),
            param_values.get("power_density"),
            param_values.get("pulse_duration")
        )
        if not fluence_check["valid"]:
            results["total_checks"] += 1
            results["failed"] += 1
            results["warnings"].append({"type": "fluence_consistency", "message": fluence_check["message"]})
        else:
            results["total_checks"] += 1
            results["passed"] += 1
        results["validation_score"] = results["passed"] / results["total_checks"] if results["total_checks"] > 0 else 1.0
        return results

# =============================================
# LASER VISUALIZATION ENGINE
# =============================================
class LaserVisualizationEngine:
    """Generate visualizations focused on laser heat source parameters"""

    @staticmethod
    def create_power_parameter_chart(parameter_data: List[Dict],
                                    x_param: str = "pulse_duration",
                                    y_param: str = "fluence",
                                    title: str = None) -> go.Figure:
        if not parameter_data:
            return None
        df = pd.DataFrame(parameter_data)
        if x_param not in df.columns or y_param not in df.columns:
            return None
        fig = px.scatter(
            df,
            x=x_param,
            y=y_param,
            color='laser_type' if 'laser_type' in df.columns else None,
            size='sources' if 'sources' in df.columns else None,
            hover_data=['unit', 'confidence', 'processing_mode'],
            title=title or f"{y_param.replace('_', ' ').title()} vs {x_param.replace('_', ' ').title()}",
            labels={x_param: x_param.replace('_', ' ').title(), y_param: y_param.replace('_', ' ').title()},
            log_x=True if any(v < 0.01 for v in df[x_param] if isinstance(v, (int, float))) else False,
            log_y=True if any(v < 0.01 for v in df[y_param] if isinstance(v, (int, float))) else False
        )
        fig.update_layout(
            height=400,
            margin=dict(t=50, b=40, l=40, r=20),
            showlegend=True
        )
        return fig

    @staticmethod
    def create_parameter_distribution_chart(parameter_data: List[Dict],
                                           param_name: str,
                                           title: str = None) -> go.Figure:
        if not parameter_data:
            return None
        values = [d['value'] for d in parameter_data if d.get('value') is not None and isinstance(d['value'], (int, float))]
        if not values:
            return None
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Distribution", "By Laser Type"))
        fig.add_trace(
            go.Histogram(x=values, name=param_name, marker_color='rgb(31, 119, 180)', nbinsx=20),
            row=1, col=1
        )
        if 'laser_type' in parameter_data[0]:
            for lt in set(d.get('laser_type') for d in parameter_data if d.get('laser_type')):
                lt_values = [d['value'] for d in parameter_data if d.get('laser_type') == lt and d.get('value') is not None]
                if lt_values:
                    fig.add_trace(
                        go.Box(y=lt_values, name=lt),
                        row=1, col=2
                    )
        fig.update_layout(
            title=title or f"{param_name.replace('_', ' ').title()} Distribution",
            height=400,
            margin=dict(t=50, b=40, l=40, r=20),
            showlegend=False
        )
        fig.update_xaxes(title_text=param_name.replace('_', ' ').title(), row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text=param_name.replace('_', ' ').title(), row=1, col=2)
        return fig

    @staticmethod
    def create_laser_type_comparison_chart(parameter_data: List[Dict],
                                          param_name: str,
                                          title: str = None) -> go.Figure:
        if not parameter_data:
            return None
        df = pd.DataFrame(parameter_data)
        if 'laser_type' not in df.columns:
            return None
        agg_df = df.groupby('laser_type')[param_name].agg(['mean', 'std', 'count']).reset_index()
        agg_df = agg_df[agg_df['count'] >= 1]
        if agg_df.empty:
            return None
        fig = px.bar(
            agg_df,
            x='laser_type',
            y='mean',
            error_y='std',
            title=title or f"{param_name.replace('_', ' ').title()} by Laser Source Type",
            labels={'laser_type': 'Laser Source Type', 'mean': param_name.replace('_', ' ').title()},
            color='laser_type',
            text=agg_df['count'].apply(lambda x: f"n={int(x)}")
        )
        fig.update_layout(
            height=400,
            margin=dict(t=50, b=60, l=40, r=20),
            xaxis_title="Laser Source Type",
            yaxis_title=param_name.replace('_', ' ').title(),
            showlegend=False
        )
        return fig

    @staticmethod
    def create_processing_mode_radar_chart(mode_data: List[Dict],
                                          title: str = "Processing Mode Parameter Profile") -> go.Figure:
        if not mode_data:
            return None
        modes = list(set(d.get('processing_mode') for d in mode_data if d.get('processing_mode')))
        if not modes:
            return None
        param_counts = Counter(d.get('parameter') for d in mode_data if d.get('parameter'))
        top_params = [p for p, c in param_counts.most_common(5) if p]
        if not top_params:
            return None
        categories = [p.replace('_', ' ').title() for p in top_params]
        fig = go.Figure()
        for mode in modes:
            mode_params = [d for d in mode_data if d.get('processing_mode') == mode]
            values = []
            for param in top_params:
                param_vals = [d.get('normalized_value') for d in mode_params if d.get('parameter') == param and d.get('normalized_value') is not None]
                if param_vals:
                    values.append(np.mean(param_vals) / max(1, max(abs(v) for v in param_vals)))
                else:
                    values.append(0)
            values.append(values[0])
            cats_closed = categories + [categories[0]]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=cats_closed,
                fill='toself',
                name=mode,
                line=dict(width=2),
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickformat='.0%'),
                angularaxis=dict(tickfont=dict(size=10))
            ),
            title=dict(text=title, x=0.5),
            showlegend=True,
            height=450,
            margin=dict(t=50, b=20, l=20, r=20)
        )
        return fig

    @staticmethod
    def create_parameter_space_heatmap(parameter_data: List[Dict],
                                      x_param: str,
                                      y_param: str,
                                      z_param: str,
                                      title: str = None) -> go.Figure:
        if not parameter_data:
            return None
        df = pd.DataFrame(parameter_data)
        if x_param not in df.columns or y_param not in df.columns or z_param not in df.columns:
            return None
        df_clean = df.dropna(subset=[x_param, y_param, z_param])
        if df_clean.empty:
            return None
        fig = px.density_heatmap(
            df_clean,
            x=x_param,
            y=y_param,
            z=z_param,
            title=title or f"Parameter Space: {z_param.replace('_', ' ').title()} over {x_param}×{y_param}",
            labels={x_param: x_param.replace('_', ' ').title(), y_param: y_param.replace('_', ' ').title(), z_param: z_param.replace('_', ' ').title()},
            log_x=True if df_clean[x_param].min() < 0.01 else False,
            log_y=True if df_clean[y_param].min() < 0.01 else False
        )
        fig.update_layout(
            height=400,
            margin=dict(t=50, b=40, l=40, r=20)
        )
        return fig

# =============================================
# SESSION STATE & UTILITIES - FIXED VERSION
# =============================================
def safe_get_session_state(key: str, default: Any = None) -> Any:
    """Safely get a session state value with fallback to default"""
    return st.session_state.get(key, default)

def safe_set_session_state(key: str, value: Any) -> None:
    """Safely set a session state value"""
    st.session_state[key] = value

def initialize_session_state():
    """
    Initialize ALL session state variables with defaults.
    This prevents AttributeError when accessing st.session_state attributes.
    """
    defaults = {
        # === File Processing ===
        "processed_files": set(),
        "vectorstore": None,
        "all_chunks": [],
        "processing_complete": False,
        
        # === Chat System ===
        "messages": [],
        
        # === LLM Configuration ===
        "llm_model_choice": None,
        "llm_tokenizer": None,
        "llm_model": None,
        "llm_backend": None,
        "llm_backend_type": None,
        "llm_device_or_host": None,
        "inference_backend": "Hugging Face Transformers",
        
        # === Embeddings ===
        "embeddings": None,
        
        # === Domain Settings ===
        "laser_domain_boost": True,
        "show_sources": True,
        "citation_style": "apa",
        "max_retrieved_chunks": 4,
        "use_4bit_quantization": True,
        "ollama_host": "http://localhost:11434",
        "metadata_cache": metadata_cache,
        
        # === Fusion Engine ===
        "enable_laser_fusion": True,
        "fusion_source_type_filter": None,
        "fusion_processing_mode_filter": None,
        "fusion_parameter_filter": None,
        
        # === Debug & Evaluation ===
        "debug_extraction": False,
        "evaluation_mode": False,
        
        # === Visualization Controls - ALL REQUIRED ATTRIBUTES ===
        "viz_chart_type": "scatter",
        "viz_param_focus": None,
        "viz_laser_type_focus": None,
        "viz_mode": "All Modes",  # ← THE KEY FIX: Always initialized
        "viz_x_param": "pulse_duration",
        "viz_y_param": "fluence",
        "viz_z_param": "power_density",
        
        # === Demo & Utility ===
        "demo_question": None,
        "last_query": None,
        "response_timestamp": None,
    }
    
    # Initialize each key defensively
    for key, value in defaults.items():
        if key not in st.session_state:
            if isinstance(value, (set, list, dict)):
                # Create new instance for mutable defaults
                st.session_state[key] = type(value)(value) if value else type(value)()
            else:
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL,
            #model_kwargs={'device': 'cpu'},
            model_kwargs={'device': device},
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
# DOCUMENT PROCESSING - LASER FOCUSED
# =============================================
def extract_laser_source_metadata(text: str, filename: str) -> Dict[str, any]:
    """Extract laser-specific metadata from text"""
    metadata = {
        "source": filename,
        "laser_topics": [],
        "parameters_found": {},
        "has_equations": bool(re.search(r'[\(=]\s*[\d.]+\s*[×*]\s*10\^', text)),
        "has_figures": bool(re.search(r'Figure\s*\d+|Fig\.\s*\d+', text, re.I)),
    }
    text_lower = text.lower()
    for topic, keywords in LASER_KEYWORDS.items():
        if any(kw.lower() in text_lower for kw in keywords):
            metadata["laser_topics"].append(topic)
    for param_name, param_config in LASER_PARAMETERS.items():
        for pattern in param_config["patterns"]:
            match = re.search(pattern, text, re.I)
            if match:
                try:
                    metadata["parameters_found"][param_name] = float(match.group(1))
                except:
                    pass
    return metadata

def load_and_chunk_laser_documents(uploaded_files: List) -> List[Document]:
    """Load and chunk documents with laser-aware splitting"""
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
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=LASER_DOMAIN_CONFIG["chunk_size"],
                chunk_overlap=LASER_DOMAIN_CONFIG["chunk_overlap"],
                separators=["\n\n", "\n", "Power:", "Fluence:", "Wavelength:", "Pulse:", "Table", "Parameter:", "Laser:", ""],
                length_function=len
            )
            chunks = text_splitter.split_documents(pages)
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "source": uploaded_file.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    **extract_laser_source_metadata(chunk.page_content, uploaded_file.name),
                    "bibliographic": bib_meta.to_dict(),
                    "citation_display": bib_meta.format_citation(st.session_state.get('citation_style', 'apa')),
                })
                all_chunks.extend(chunks)
            st.info(f"✅ Loaded {len(chunks)} laser-focused chunks from `{uploaded_file.name}`")
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
    """Create FAISS vector store with laser-boosted embeddings"""
    try:
        embeddings = load_local_embeddings()
        if embeddings is None:
            return None
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.metadata = {
            "total_chunks": len(chunks),
            "embedding_model": embedding_model_key,
            "created_at": datetime.now().isoformat(),
            "laser_topics": list(set(topic for chunk in chunks for topic in chunk.metadata.get("laser_topics", [])))
        }
        return vectorstore
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None

# =============================================
# RAG WITH LASER FUSION & VISUALIZATION
# =============================================
def create_laser_rag_prompt(retrieved_chunks: List[Document], query: str) -> str:
    """Create prompt for laser-focused RAG"""
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        citation = chunk.metadata.get("citation_display")
        if not citation:
            source = chunk.metadata.get("source", "unknown")
            topics = chunk.metadata.get("laser_topics", [])
            topic_str = f" [{', '.join(topics)}]" if topics else ""
            citation = f"[Source {i}{topic_str} - {source}]"
        content = chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content
        context_parts.append(f"{citation}\n{content}\n")
    context = "\n---\n".join(context_parts)
    laser_system_prompt = """You are an expert assistant for laser heat source research.
Your role is to answer questions about laser source types, power instruments, and processing parameters.
Focus ONLY on: laser power, pulse characteristics, wavelength, fluence, spot size, scan parameters.
Do NOT discuss material properties, microstructure, or non-laser topics unless explicitly asked.
Rules:
1. Use ONLY information from the retrieved context below
2. If the answer isn't in the context, say "Based on the provided documents, I cannot determine..."
3. Never invent laser parameters or experimental conditions
4. When citing, use the EXACT citation string provided (e.g., "Smith et al., Opt. Express, 2023" or "DOI:10.1364/OE.123456")
5. For numerical values, ALWAYS include units
6. Be precise about laser source type (CW, pulsed, femtosecond, fiber, etc.)
7. Note measurement instruments when mentioned (power meter, energy meter, etc.)
"""
    user_query = f"""Retrieved Context from Laser Heat Source Documents:
{context}
User Question: {query}
Answer (cite sources using provided citation format, focus on laser parameters only):"""
    return laser_system_prompt + user_query

def _create_fusion_aware_laser_prompt(retrieved_docs: List[Document], query: str,
                                     fused_parameters: Dict[str, FusedLaserParameter],
                                     fusion_metrics: FusionEfficiencyMetrics,
                                     comparison_table: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """Create prompt that incorporates fused laser parameter data"""
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        citation = doc.metadata.get('citation_display', f"[Source {i+1}]")
        content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
        context_parts.append(f"[{i+1}] {citation}\n{content}\n")
    context = "\n---\n".join(context_parts)
    parameters_summary = ""
    if fused_parameters:
        parameters_summary = "**Fused Laser Parameter Summary**:\n"
        for param_name, entry in list(fused_parameters.items())[:8]:
            if entry.fused_value is not None and isinstance(entry.fused_value, (int, float)):
                value_str = f"{entry.fused_value:.3f}"
                if entry.standard_deviation is not None:
                    value_str = f"{value_str} ± {entry.standard_deviation:.3f}"
            else:
                value_str = str(entry.fused_value) if entry.fused_value is not None else "N/A"
            laser_info = f" ({entry.laser_source_type})" if entry.laser_source_type else ""
            parameters_summary += f"• {param_name.replace('_', ' ').title()}: {value_str} {entry.unit or ''}{laser_info} [conf: {entry.fusion_confidence.value}, sources: {entry.source_count}]\n"
        parameters_summary += "\n"
    table_section = f"**Parameter Comparison Table**:\n{comparison_table}\n" if comparison_table else ""
    efficiency_note = ""
    if fusion_metrics.overall_fusion_efficiency >= 0.7:
        efficiency_note = f"🎯 High-confidence fusion ({fusion_metrics.overall_fusion_efficiency:.2f}/1.0): Parameters synthesized from {fusion_metrics.unique_sources_used} laser studies.\n"
    elif fusion_metrics.overall_fusion_efficiency >= 0.4:
        efficiency_note = f"⚠️ Moderate-confidence fusion ({fusion_metrics.overall_fusion_efficiency:.2f}/1.0): Some parameter variations detected across sources.\n"
    else:
        efficiency_note = f"🔍 Low-confidence fusion ({fusion_metrics.overall_fusion_efficiency:.2f}/1.0): Limited or conflicting laser parameter data.\n"
    system_prompt = """You are an expert scientific assistant specializing in laser heat source research.
YOUR TASK:
1. Answer the user's question using the retrieved document context AND the fused parameter summary below
2. When laser parameter values are available from fusion, PREFER the fused consensus value with its uncertainty range
3. Cite sources precisely using [Author, Year] or [DOI:xxx] format immediately after claims
4. If fused parameters show conflicts, acknowledge the variation and note possible causes (different laser types, measurement methods, conditions)
5. For comparative questions, reference the comparison table if provided
6. ALWAYS include units for numerical values and specify laser source type when relevant
7. FOCUS on laser heat source parameters: power, pulse, wavelength, fluence, spot size, scan parameters
RESPONSE STRUCTURE:
1. Direct answer (1-2 sentences)
2. Supporting evidence with fused parameter values and citations
3. Comparison table reference if relevant to query
4. Uncertainty/limitations note if fusion confidence is moderate/low
5. Suggested follow-up if appropriate
"""
    user_prompt = f"""RETRIEVED DOCUMENT CONTEXT:
{context}
{efficiency_note}{parameters_summary}{table_section}
USER QUESTION: {query}
SCIENTIFIC ANSWER (use fused parameters when available, cite sources precisely, focus on laser heat source parameters):"""
    full_prompt = system_prompt + user_prompt
    context_metadata = {
        "fused_parameters_count": len(fused_parameters),
        "fusion_efficiency": fusion_metrics.overall_fusion_efficiency,
        "comparison_table_available": comparison_table is not None
    }
    return full_prompt, context_metadata

def generate_local_response_transformers(tokenizer, model, device: str, prompt: str, backend_name: str) -> str:
    """Generate response using HuggingFace Transformers"""
    try:
        if "Qwen" in backend_name or "qwen" in backend_name.lower():
            messages = [
                {"role": "system", "content": "You are an expert in laser heat source research. Focus on power, pulse, wavelength, fluence."},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        elif "Llama" in backend_name or "llama" in backend_name.lower():
            messages = [
                {"role": "system", "content": "You are an expert in laser heat source research. Focus on power, pulse, wavelength, fluence."},
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
        elif "Answer (cite sources" in full_text:
            answer = full_text.split("Answer (cite sources")[-1].strip()
            answer = re.split(r'\n(?:Question|User|Context):', answer)[0].strip()
        else:
            answer = full_text[-LASER_DOMAIN_CONFIG["max_new_tokens"]*2:].strip()
        answer = re.sub(r'\s+', ' ', answer).strip()
        return answer if answer else "I was unable to generate a response. Please try rephrasing your question."
    except Exception as e:
        st.error(f"Generation error: {e}")
        return f"Error generating response: {str(e)[:200]}..."

def generate_local_response_ollama(model_tag: str, ollama_host: str, prompt: str) -> str:
    """Generate response using Ollama"""
    try:
        client = ollama.Client(host=ollama_host)
        messages = [
            {"role": "system", "content": "You are an expert in laser heat source research. Answer based ONLY on the provided context."},
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
    """Unified response generation"""
    if backend_type == "ollama":
        return generate_local_response_ollama(model_or_tag, device_or_host, prompt)
    else:
        return generate_local_response_transformers(tokenizer, model_or_tag, device_or_host, prompt, backend)

def retrieve_and_answer_with_laser_fusion(vectorstore, tokenizer, model, device_or_host: str, backend: str, backend_type: str,
                                         query: str, k: int = None, score_threshold: float = None,
                                         enable_fusion: bool = True, source_type_filter: Optional[str] = None,
                                         processing_mode_filter: Optional[str] = None,
                                         parameter_filter: Optional[List[str]] = None) -> Tuple[str, List[Document], float, Dict[str, Any]]:
    """Main retrieval and answer function with laser parameter fusion"""
    k = k or LASER_DOMAIN_CONFIG["retrieval_k"]
    score_threshold = score_threshold or LASER_DOMAIN_CONFIG["score_threshold"]
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
        return ("Based on the uploaded documents, I could not find information relevant to your laser heat source question. Try rephrasing or checking document content.",
                [], avg_relevance, {"error": "no_relevant_chunks", "fusion_enabled": enable_fusion})
    if enable_fusion:
        parameter_extractor = LaserParameterExtractor(LASER_KEYWORDS)
        fusion_engine = LaserFusionEngine(parameter_extractor)
        fused_parameters, fusion_metrics = fusion_engine.fuse_laser_documents(
            retrieved_docs, query,
            source_type_filter=source_type_filter,
            processing_mode_filter=processing_mode_filter,
            parameter_filter=parameter_filter
        )
        comparison_table = None
        if fused_parameters:
            comparison_table = fusion_engine.generate_comparison_table(fused_parameters, format="markdown")
        prompt, fusion_context = _create_fusion_aware_laser_prompt(
            retrieved_docs, query, fused_parameters, fusion_metrics, comparison_table
        )
        answer = generate_local_response(
            tokenizer=tokenizer, model_or_tag=model, device_or_host=device_or_host,
            prompt=prompt, backend=backend, backend_type=backend_type
        )
        if fusion_metrics.overall_fusion_efficiency > 0.5 and comparison_table:
            answer += f"\n---\n**📊 Laser Parameter Comparison**:\n{comparison_table}"
        metadata = {
            "fusion_enabled": True,
            "fusion_metrics": {"efficiency": fusion_metrics.overall_fusion_efficiency, "display": fusion_metrics.to_display_dict()},
            "fused_parameters": {k: v.to_comparison_row() for k, v in fused_parameters.items()},
            "comparison_table": comparison_table,
            "source_citations": [
                {"citation": doc.metadata.get('citation_display', 'Unknown'), "relevance": scores[i] if i < len(scores) else 0, "topics": doc.metadata.get('laser_topics', [])}
                for i, doc in enumerate(retrieved_docs)
            ],
            "retrieval_relevance": avg_relevance
        }
        return answer, retrieved_docs, avg_relevance, metadata
    else:
        prompt = create_laser_rag_prompt(retrieved_docs, query)
        answer = generate_local_response(
            tokenizer=tokenizer, model_or_tag=model, device_or_host=device_or_host,
            prompt=prompt, backend=backend, backend_type=backend_type
        )
        return answer, retrieved_docs, avg_relevance, {"fusion_enabled": False}

# =============================================
# VISUALIZATION UI COMPONENTS
# =============================================
def render_viz_control_panel(fused_parameters: Dict[str, FusedLaserParameter], retrieved_docs: List[Document]):
    """Render interactive visualization controls for laser parameters"""
    st.markdown("### 📊 Laser Parameter Visualizations")
    
    # Ensure viz_mode and other viz attributes exist before use (defensive initialization)
    if "viz_mode" not in st.session_state:
        st.session_state.viz_mode = "All Modes"
    if "viz_laser_type_focus" not in st.session_state:
        st.session_state.viz_laser_type_focus = "All Laser Types"
    if "viz_param_focus" not in st.session_state:
        st.session_state.viz_param_focus = "All Parameters"
    if "viz_chart_type" not in st.session_state:
        st.session_state.viz_chart_type = "scatter"
    
    # Visualization type selector
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Parameter Scatter Plot", "Parameter Distribution", "Laser Type Comparison",
         "Processing Mode Radar", "Parameter Space Heatmap"],
        key="viz_type_select"
    )
    
    # Filters with safe session state access
    col1, col2, col3 = st.columns(3)
    with col1:
        laser_options = ["All Laser Types"] + list(LASER_SOURCE_TYPES.keys())
        current_laser = safe_get_session_state("viz_laser_type_focus", "All Laser Types")
        st.session_state.viz_laser_type_focus = st.selectbox(
            "Laser Source Type Filter", 
            options=laser_options, 
            key="viz_laser",
            index=0 if current_laser == "All Laser Types" else (laser_options.index(current_laser) if current_laser in laser_options else 0)
        )
    with col2:
        param_options = ["All Parameters"] + [p.replace('_', ' ').title() for p in LASER_PARAMETERS.keys()]
        current_param = safe_get_session_state("viz_param_focus", "All Parameters")
        st.session_state.viz_param_focus = st.selectbox(
            "Parameter Focus", 
            options=param_options, 
            key="viz_param",
            index=0 if current_param == "All Parameters" else (param_options.index(current_param) if current_param in param_options else 0)
        )
    with col3:
        mode_options = ["All Modes"] + list(LASER_PROCESSING_MODES.keys())
        current_mode = safe_get_session_state("viz_mode", "All Modes")
        st.session_state.viz_mode = st.selectbox(
            "Processing Mode Filter", 
            options=mode_options, 
            key="viz_mode",
            index=0 if current_mode == "All Modes" else (mode_options.index(current_mode) if current_mode in mode_options else 0)
        )
    
    # Generate visualization based on selection
    viz_engine = LaserVisualizationEngine()
    if viz_type == "Parameter Scatter Plot":
        parameter_data = []
        for param_name, param in fused_parameters.items():
            if param.fused_value is not None and isinstance(param.fused_value, (int, float)):
                if st.session_state.viz_param_focus == "All Parameters" or st.session_state.viz_param_focus == param_name.replace('_', ' ').title():
                    parameter_data.append({
                        "parameter": param_name,
                        "value": param.fused_value,
                        "unit": param.unit,
                        "laser_type": param.laser_source_type,
                        "processing_mode": param.processing_mode,
                        "sources": param.source_count,
                        "confidence": param.fusion_confidence.value,
                        "normalized_value": param.fused_value
                    })
        if parameter_data:
            x_param = st.selectbox("X-axis Parameter", options=["pulse_duration", "wavelength", "power", "fluence"], key="viz_x")
            y_param = st.selectbox("Y-axis Parameter", options=["fluence", "power_density", "spot_size", "pulse_energy"], key="viz_y")
            title = f"Laser Parameter Relationship"
            if st.session_state.viz_laser_type_focus != "All Laser Types":
                title += f" - {st.session_state.viz_laser_type_focus}"
            fig = viz_engine.create_power_parameter_chart(parameter_data, x_param=x_param, y_param=y_param, title=title)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No parameter data available for scatter plot. Upload documents with quantitative laser parameters.")
    elif viz_type == "Parameter Distribution":
        param_name = st.session_state.viz_param_focus if st.session_state.viz_param_focus != "All Parameters" else list(fused_parameters.keys())[0] if fused_parameters else None
        if param_name:
            param_data = []
            for p_name, param in fused_parameters.items():
                if p_name == param_name and param.fused_value is not None and isinstance(param.fused_value, (int, float)):
                    param_data.append({
                        "value": param.fused_value,
                        "laser_type": param.laser_source_type,
                        "unit": param.unit
                    })
            if param_data:
                title = f"{param_name.replace('_', ' ').title()} Distribution"
                fig = viz_engine.create_parameter_distribution_chart(param_data, param_name, title=title)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No numeric values found for parameter: {param_name}")
        else:
            st.info("Select a parameter to visualize its distribution")
    elif viz_type == "Laser Type Comparison":
        param_name = st.session_state.viz_param_focus if st.session_state.viz_param_focus != "All Parameters" else list(fused_parameters.keys())[0] if fused_parameters else None
        if param_name:
            param_data = []
            for p_name, param in fused_parameters.items():
                if p_name == param_name and param.fused_value is not None and isinstance(param.fused_value, (int, float)):
                    param_data.append({
                        "parameter": p_name,
                        "value": param.fused_value,
                        "laser_type": param.laser_source_type,
                        "unit": param.unit
                    })
            if param_data:
                title = f"{param_name.replace('_', ' ').title()} by Laser Source Type"
                fig = viz_engine.create_laser_type_comparison_chart(param_data, param_name, title=title)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need multiple laser types with the same parameter for comparison")
            else:
                st.info(f"No numeric values found for parameter: {param_name}")
        else:
            st.info("Select a parameter to compare across laser types")
    elif viz_type == "Processing Mode Radar":
        mode_data = []
        for param_name, param in fused_parameters.items():
            if param.fused_value is not None and isinstance(param.fused_value, (int, float)):
                mode_data.append({
                    "parameter": param_name,
                    "normalized_value": param.fused_value,
                    "processing_mode": param.processing_mode,
                    "laser_type": param.laser_source_type
                })
        if mode_data:
            title = "Processing Mode Parameter Profile"
            if st.session_state.viz_laser_type_focus != "All Laser Types":
                title += f" - {st.session_state.viz_laser_type_focus}"
            fig = viz_engine.create_processing_mode_radar_chart(mode_data, title=title)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient processing mode data for radar chart")
    elif viz_type == "Parameter Space Heatmap":
        param_data = []
        for param_name, param in fused_parameters.items():
            if param.fused_value is not None and isinstance(param.fused_value, (int, float)):
                param_data.append({
                    param_name: param.fused_value,
                    "laser_type": param.laser_source_type,
                    "processing_mode": param.processing_mode
                })
        if len(param_data) >= 3:
            x_param = st.selectbox("X-axis", options=list(LASER_PARAMETERS.keys()), index=2, key="heatmap_x")
            y_param = st.selectbox("Y-axis", options=list(LASER_PARAMETERS.keys()), index=5, key="heatmap_y")
            z_param = st.selectbox("Color/Size", options=list(LASER_PARAMETERS.keys()), index=4, key="heatmap_z")
            title = f"Parameter Space: {z_param} over {x_param}×{y_param}"
            fig = viz_engine.create_parameter_space_heatmap(param_data, x_param, y_param, z_param, title=title)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 3 parameters with numeric values for heatmap visualization")

def render_fusion_metrics_panel(fusion_metadata: Dict[str, Any]):
    """Display fusion efficiency metrics for laser parameters"""
    if not fusion_metadata.get("fusion_enabled"):
        return
    metrics_display = fusion_metadata.get("fusion_metrics", {}).get("display", {})
    if not metrics_display:
        return
    with st.expander("📊 Laser Parameter Fusion Efficiency", expanded=True):
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
                st.caption(f"{relevance_bar} {src['citation']} (laser topics: {topics_str})")
        fused_params = fusion_metadata.get("fused_parameters", {})
        if fused_params:
            conflicts = [k for k, v in fused_params.items() if v.get("confidence") == "low"]
            if conflicts:
                st.warning(f"⚠️ {len(conflicts)} parameter(s) have low-confidence fusion: {', '.join(conflicts[:3])}")

def render_extracted_parameters_debug(extracted_params: List[LaserParameter], source_citation: str):
    """Debug view for extracted laser parameters"""
    if not extracted_params:
        st.info("🔍 No laser parameters extracted from this chunk")
        return
    with st.expander(f"🐛 Extracted Laser Parameters: {source_citation}", expanded=False):
        for i, param in enumerate(extracted_params, 1):
            st.markdown(f"**{i}. {param.normalized_name}**")
            st.caption(f"Value: `{param.value}` {param.unit or ''} | Type: {param.parameter_type}")
            if param.laser_source_type:
                st.caption(f"Laser Type: {param.laser_source_type}")
            if param.processing_mode:
                st.caption(f"Processing Mode: {param.processing_mode}")
            if param.condition:
                st.caption(f"Condition: {param.condition}")
            if param.context_snippet:
                st.code(param.context_snippet[:200] + "..." if len(param.context_snippet) > 200 else param.context_snippet, language="text")
            st.divider()

def render_comparison_table_in_chat(comparison_table: Optional[str], fused_parameters: Dict):
    """Display comparison table in chat interface"""
    if not comparison_table:
        return
    with st.expander("📋 Laser Parameter Comparison Table", expanded=False):
        st.markdown(comparison_table, unsafe_allow_html=True)
        if fused_parameters:
            selected_param = st.selectbox(
                "🔍 Explore parameter details:",
                options=["Select a parameter..."] + list(fused_parameters.keys()),
                key="fusion_param_select"
            )
            if selected_param and selected_param != "Select a parameter...":
                param_data = fused_parameters[selected_param]
                st.json({
                    "parameter": selected_param,
                    "fused_value": param_data["value"],
                    "unit": param_data["unit"],
                    "range": param_data["range"],
                    "sources": param_data["sources"],
                    "confidence": param_data["confidence"],
                    "laser_type": param_data.get("laser_type"),
                    "processing_mode": param_data.get("processing_mode")
                })

def render_physics_validation_panel(fused_parameters: Dict[str, FusedLaserParameter]):
    """Display physics validation results for laser parameters"""
    if not fused_parameters:
        return
    validator = LaserPhysicsValidator()
    validation_results = validator.full_validation(fused_parameters)
    with st.expander("🧮 Laser Physics Validation", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Validation Score", f"{validation_results['validation_score']:.0%}")
        col2.metric("Checks Passed", validation_results["passed"])
        col3.metric("Checks Failed", validation_results["failed"])
        if validation_results["violations"]:
            st.warning("⚠️ Physical bound violations detected:")
            for v in validation_results["violations"]:
                st.error(f"**{v.get('parameter')}**: {v.get('value')} {v.get('unit')} outside bounds")
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
        st.markdown("#### 🔬 Laser Domain Settings")
        st.session_state.laser_domain_boost = st.checkbox(
            "Boost laser-topic relevance", value=True,
            help="Prioritize chunks containing laser-specific keywords (power, fluence, pulse, wavelength)"
        )
        st.session_state.show_sources = st.checkbox(
            "Show source citations", value=True,
            help="Display which documents chunks came from"
        )
        st.session_state.enable_laser_fusion = st.checkbox(
            "🔗 Enable Laser Parameter Fusion", value=True,
            help="Enable cross-document laser parameter extraction and consensus (power, pulse, fluence, etc.)"
        )
        st.session_state.debug_extraction = st.checkbox(
            "🐛 Debug Parameter Extraction", value=False,
            help="Show extracted laser parameters in UI for diagnosis"
        )
        st.session_state.evaluation_mode = st.checkbox(
            "📊 Enable Evaluation Mode", value=False,
            help="Show retrieval metrics and physics validation"
        )
        st.markdown("#### 📊 Visualization Settings")
        st.session_state.viz_chart_type = st.selectbox(
            "Default Chart Type",
            options=["scatter", "bar", "radar", "heatmap"],
            index=0,
            help="Default visualization type for laser parameters"
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
<li>Upload papers about laser sources, power measurement, processing parameters</li>
<li>Ask specific questions: "What pulse energy for femtosecond ablation?"</li>
<li>Small models (≤1.5B) work on CPU; larger need GPU</li>
<li>First load may take 1-2 min (model download)</li>
<li>For Ollama: run <code>ollama pull qwen2.5:7b</code> first</li>
<li>🔗 Fusion works best with comparative queries across multiple laser studies</li>
<li>🐛 Enable debug mode to see extracted laser parameters</li>
<li>📊 Use visualization panel for interactive parameter charts</li>
<li>🎯 SCOPE: This system focuses ONLY on laser heat source parameters</li>
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
    st.markdown("### 📁 Upload Laser Heat Source Documents")
    uploaded_files = st.file_uploader(
        "Select PDF or TXT files about laser sources, power instruments, processing parameters, etc.",
        type=["pdf", "txt"], accept_multiple_files=True,
        help="Documents will be processed locally - no data leaves your browser. Bibliographic metadata (DOI, authors, journal, year) will be extracted for human-readable citations. FOCUS: laser power, pulse characteristics, wavelength, fluence, spot size, scan parameters."
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
    with st.spinner(f"Processing {len(new_files)} document(s) and extracting laser parameter metadata..."):
        try:
            chunks = load_and_chunk_laser_documents(new_files)
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
            st.success(f"✅ Ready! Indexed {len(st.session_state.all_chunks)} laser-focused chunks from {len(st.session_state.processed_files)} files")
            st.session_state.processing_complete = True
            return True
        except Exception as e:
            st.error(f"Processing failed: {e}")
            import traceback
            st.error(traceback.format_exc())
            return False

def render_chat_interface():
    if not st.session_state.get('vectorstore'):
        st.info("👆 Upload laser heat source documents above to start chatting")
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
                        topics = src.metadata.get("laser_topics", [])
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
                            extractor = LaserParameterExtractor(LASER_KEYWORDS)
                            record = extractor.extract_parameters_from_chunk(src.page_content, src.metadata)
                            render_extracted_parameters_debug(record.extracted_parameters, citation)
            if message.get("fusion_metadata") and st.session_state.enable_laser_fusion:
                render_fusion_metrics_panel(message["fusion_metadata"])
                if message["fusion_metadata"].get("comparison_table"):
                    render_comparison_table_in_chat(
                        message["fusion_metadata"]["comparison_table"],
                        message["fusion_metadata"].get("fused_parameters", {})
                    )
            fused_params = message["fusion_metadata"].get("fused_parameters", {}) if message.get("fusion_metadata") else {}
            if fused_params:
                render_physics_validation_panel(fused_params)
    # Chat input
    if prompt := st.chat_input("Ask about laser power, pulse parameters, wavelength, fluence, or compare laser sources..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("🔍 Retrieving, fusing laser parameter data, and generating..."):
                try:
                    # Safe access for viz filters with defaults
                    viz_laser = safe_get_session_state("viz_laser_type_focus", "All Laser Types")
                    viz_mode = safe_get_session_state("viz_mode", "All Modes")
                    viz_param = safe_get_session_state("viz_param_focus", "All Parameters")
                    
                    source_type_filter = viz_laser if viz_laser != "All Laser Types" else None
                    processing_mode_filter = viz_mode if viz_mode != "All Modes" else None
                    parameter_filter = [viz_param.lower().replace(' ', '_')] if viz_param != "All Parameters" else None
                    
                    # Always capture 4 return values
                    answer, retrieved_docs, relevance, metadata = retrieve_and_answer_with_laser_fusion(
                        vectorstore=st.session_state.vectorstore,
                        tokenizer=st.session_state.llm_tokenizer,
                        model=st.session_state.llm_model,
                        device_or_host=st.session_state.llm_device_or_host,
                        backend=st.session_state.llm_model_choice,
                        backend_type=st.session_state.llm_backend_type,
                        query=prompt,
                        k=st.session_state.max_retrieved_chunks,
                        enable_fusion=st.session_state.enable_laser_fusion,
                        source_type_filter=source_type_filter,
                        processing_mode_filter=processing_mode_filter,
                        parameter_filter=parameter_filter
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
                    if st.session_state.enable_laser_fusion and metadata.get("fusion_enabled"):
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
                    if st.session_state.enable_laser_fusion and metadata.get("fusion_enabled") and metadata.get("fused_parameters"):
                        render_viz_control_panel(metadata["fused_parameters"], retrieved_docs)
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)[:300]}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

def render_footer():
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📚 Example Questions:**")
        st.caption("• What pulse energy is typical for femtosecond laser ablation?")
        st.caption("• Compare fluence thresholds for CW vs pulsed lasers")
        st.caption("• How does wavelength affect power density for a given spot size?")
        st.caption("• What scan speeds are used for laser marking with fiber lasers?")
    with col2:
        st.markdown("**⚡ Performance Tips:**")
        st.caption("• Keep questions focused on laser parameters (power, pulse, fluence)")
        st.caption("• Specify laser source type (femtosecond, fiber, CO2) for better retrieval")
        st.caption("• CPU mode: allow 10-30s per response; GPU: 2-10s")
        st.caption("• Enable fusion for comparative queries across laser studies")
    with col3:
        st.markdown("**🔐 Privacy & Fusion:**")
        st.caption("• All processing happens locally in your session")
        st.caption("• Multi-document fusion extracts & compares laser parameters ONLY")
        st.caption("• Fusion efficiency metrics quantify synthesis quality")
        st.caption("• Citations display as 'FirstAuthor et al., Journal, Year' or DOI")
        st.caption("• Physics validation checks energy consistency, parameter bounds")

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    st.set_page_config(
        page_title="🔬 Laser Heat Source RAG + Fusion + Viz",
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
    st.markdown('<h1 class="main-header">🔬 Laser Heat Source RAG + Fusion + Visualization</h1>', unsafe_allow_html=True)
    st.markdown("""<div style="text-align:center;color:#64748b;margin-bottom:1.5rem">
Upload research papers on laser sources, power instruments, and processing parameters. Get answers with
<strong>human-readable citations</strong> and <strong>interactive visualizations</strong>.
<br><span class="fusion-badge">🔗 Multi-document fusion: power, pulse, wavelength, fluence, spot size</span>
<br><span class="fusion-badge">📊 Customizable charts: Scatter, Distribution, Comparison with laser-type filters</span>
<br><span class="fusion-badge">🧮 Physics validation: Energy consistency, parameter bounds</span>
<br><span class="fusion-badge">🎯 SCOPE: LASER HEAT SOURCE PARAMETERS ONLY</span>
</div>""", unsafe_allow_html=True)
    
    # ⚠️ CRITICAL: Initialize session state BEFORE any access
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
                st.caption(f"📦 {meta.get('total_chunks', '?')} laser-focused chunks")
                topics = meta.get('laser_topics', [])
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
            st.markdown("""<div class="info-card"><h3>👋 Welcome to Laser Heat Source RAG!</h3>
<p>This assistant helps you query documents about:</p>
<ul>
<li>🔥 Laser source types: CW, pulsed, femtosecond, fiber, CO2, diode</li>
<li>⚡ Power & energy: average power, pulse energy, peak power</li>
<li>⏱️ Pulse characteristics: duration, repetition rate, duty cycle</li>
<li>🌈 Wavelength & beam: wavelength, spot size, beam quality (M²)</li>
<li>💡 Fluence & intensity: energy density, power density, irradiance</li>
<li>🔄 Processing parameters: scan speed, hatch distance, overlap, dwell time</li>
<li>🔧 Power instruments: power meters, energy meters, beam profilers</li>
<li>🔗 <strong>Multi-document fusion</strong> with efficiency metrics</li>
<li>📈 <strong>Interactive visualizations</strong>: Scatter, Distribution, Comparison charts</li>
<li>🧮 <strong>Physics validation</strong>: Energy consistency, parameter bounds</li>
</ul>
<p><strong>🎯 Enhanced Features:</strong></p>
<ul>
<li>Citations display as "Smith et al., Opt. Express, 2023" or DOI</li>
<li>🔗 Cross-document laser parameter extraction focused on heat source ONLY</li>
<li>📊 Fusion efficiency metrics per answer</li>
<li>📋 Automatic comparison table generation</li>
<li>🎨 Customizable visualizations with laser-type/processing-mode filters</li>
<li>🐛 Debug mode for parameter extraction diagnosis</li>
<li>🧮 Physics-aware validation for laser parameter consistency</li>
<li>🚫 SCOPE: This system EXCLUDES material properties, microstructure, non-laser topics</li>
</ul>
<p><strong>Getting started:</strong></p>
<ol>
<li>Upload PDF/TXT files about laser sources and processing parameters in the left panel</li>
<li>Click "Process Documents"</li>
<li>Select your preferred local LLM in sidebar</li>
<li>Enable "Laser Parameter Fusion" for comparative queries</li>
<li>Use visualization panel for interactive parameter charts</li>
<li>Start asking technical questions about laser heat sources!</li>
</ol></div>""", unsafe_allow_html=True)
            st.markdown("**Try asking:**")
            demo_qs = [
                "What is the typical pulse energy for femtosecond laser ablation of metals?",
                "Compare fluence thresholds for CW vs pulsed laser processing",
                "How does wavelength affect power density for a given spot size?",
                "What scan speeds are used for laser marking with fiber lasers?",
                "What power meter accuracy is required for low-power diode laser calibration?"
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
