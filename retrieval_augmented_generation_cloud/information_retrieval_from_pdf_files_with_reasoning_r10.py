#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LASER MICROSTRUCTURE RAG CHATBOT - FULLY API-FREE VERSION WITH ENHANCED MULTI-DOCUMENT FUSION
========================================================================================
✅ Zero API keys required - all models run locally (optional Crossref/pdf2doi for metadata)
✅ Supports Hugging Face transformers models AND Ollama local models
✅ Optimized for Streamlit Cloud and local deployment
✅ Laser-microstructure domain specialization
✅ PDF/text document ingestion with FAISS vector storage
✅ 🎯 SOURCE CITATION WITH HUMAN-READABLE IDs: DOI, FirstAuthor et al., Journal, Year, Volume
✅ 🔗 MULTI-DOCUMENT REASONING: Cross-document property extraction, fusion, and comparison
✅ 📊 ENHANCED FUSION EFFICIENCY: Robust metrics computation with fallbacks and debugging
✅ 📋 TABULAR OUTPUT: Automatic generation of comparison tables from multiple studies
✅ Confidence scoring, relevance filtering, and uncertainty quantification
✅ Responsive UI with streaming-like output simulation
✅ Memory-efficient loading with quantization support for large models
✅ Automatic fallback to smaller models if GPU memory is limited

ENHANCEMENTS APPLIED:
• Fixed zero-efficiency metrics: proper fallback computation when no properties extracted
• Enhanced property extraction: patterns for equations, standardized values, statistical features
• Fixed UI display: null-safe metric rendering, proper source citation formatting
• Added extraction debugging: log extracted properties for diagnosis
• Improved bibliographic parsing: handle complex publisher strings like "Elsevier Ltd. All rights reserved"

Deploy to Streamlit Cloud with requirements.txt below.
For local use with Ollama: install ollama Python library and run `ollama pull <model>`
For enhanced metadata extraction: pip install pdf2doi crossrefapi (optional)
For advanced table parsing: pip install pandas tabulate lxml (recommended)
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

# Configure logging for debugging extraction issues
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LangChain / RAG imports (local-only, no API calls)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Transformers for local LLM inference via Hugging Face
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, AutoModelForCausalLM,
    pipeline, set_seed, BitsAndBytesConfig
)

# Optional: Ollama support for local model serving
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Optional: Bibliographic metadata extraction libraries
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

# Optional: PyPDF2 for reading PDF metadata
try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

# Optional: Advanced table parsing
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


# =============================================
# GLOBAL CONFIGURATION - LASER MICROSTRUCTURE DOMAIN
# =============================================

# Local model choices - Using proper Hugging Face repo IDs
LOCAL_LLM_OPTIONS = {
    # === TINY MODELS (Good for low-latency testing, CPU-friendly) ===
    "GPT-2 (1.5B, fastest startup, CPU OK)": "gpt2",
    "Qwen2-0.5B-Instruct (best JSON, recommended)": "Qwen/Qwen2-0.5B-Instruct",
    "Qwen2.5-0.5B-Instruct (newest, best reasoning)": "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama-1.1B-Chat (balanced small model)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    
    # === MEDIUM MODELS (Require GPU or good CPU, 4-8GB VRAM) ===
    "Qwen2.5-1.5B-Instruct (efficient mid-size)": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct (strong reasoning)": "Qwen/Qwen2.5-3B-Instruct",
    "Mistral-7B-Instruct-v0.3 (reliable & efficient)": "mistralai/Mistral-7B-Instruct-v0.3",
    "Llama-3.2-3B-Instruct (Meta's latest small)": "meta-llama/Llama-3.2-3B-Instruct",
    
    # === LARGE MODELS (Require GPU with 12-24GB VRAM, use 4-bit quantization) ===
    "Qwen2.5-7B-Instruct (excellent all-rounder)": "Qwen/Qwen2.5-7B-Instruct",
    "Llama-3.1-8B-Instruct (most popular balanced)": "meta-llama/Llama-3.1-8B-Instruct",
    "Gemma-2-9B-it (Google's latest, great logic)": "google/gemma-2-9b-it",
    "Falcon-7B-Instruct (lightweight & modern)": "tiiuae/falcon-7b-instruct",
    
    # === OLLAMA BACKEND MODELS (if ollama library installed) ===
    "[Ollama] qwen2.5:0.5b (via ollama serve)": "ollama:qwen2.5:0.5b",
    "[Ollama] qwen2.5:1.5b (via ollama serve)": "ollama:qwen2.5:1.5b",
    "[Ollama] qwen2.5:7b (via ollama serve)": "ollama:qwen2.5:7b",
    "[Ollama] qwen2.5:14b (via ollama serve) 🔥": "ollama:qwen2.5:14b",
    "[Ollama] llama3.1:8b (via ollama serve)": "ollama:llama3.1:8b",
    "[Ollama] mistral:7b (via ollama serve)": "ollama:mistral:7b",
    "[Ollama] gemma2:9b (via ollama serve)": "ollama:gemma2:9b",
    "[Ollama] falcon3:10b (via ollama serve)": "ollama:falcon3:10b",
}

# Local embedding model (~80MB, CPU-friendly)
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Laser-microstructure domain settings
LASER_DOMAIN_CONFIG = {
    "chunk_size": 800,
    "chunk_overlap": 150,
    "retrieval_k": 4,
    "score_threshold": 0.25,
    "max_context_tokens": 1024,
    "max_new_tokens": 256,
    "temperature": 0.1,
}

# Laser-specific keywords for domain filtering and boosting
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

# Memory estimation for model loading
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
# FUSION DATA STRUCTURES AND ENUMS
# =============================================

class FusionConfidence(Enum):
    """Confidence levels for fused property entries."""
    HIGH = "high"          # Multiple consistent sources, high extraction confidence
    MODERATE = "moderate"  # Single source or minor variations across sources
    LOW = "low"            # Conflicting sources or low-confidence extraction
    UNKNOWN = "unknown"    # Insufficient data


@dataclass
class ExtractedProperty:
    """
    Represents a single scientific property extracted from document text.
    Supports normalization, uncertainty tracking, and source attribution.
    """
    name: str                          # e.g., "ablation_threshold", "yield_strength"
    value: Union[float, str, List]    # Numeric value, range, or categorical
    unit: Optional[str] = None         # e.g., "J/cm²", "MPa", "fs"
    uncertainty: Optional[str] = None  # e.g., "±0.2", "0.1-0.5", "approx."
    condition: Optional[str] = None    # e.g., "at 800nm", "for silicon", "aged 24h@150°C"
    source_chunk_id: str = ""          # Unique identifier: filename:chunk_index
    source_citation: str = ""          # Human-readable: "Smith et al., 2023"
    extraction_confidence: float = 0.5 # From metadata extraction
    context_snippet: str = ""          # Original text snippet for verification
    property_type: str = "parameter"   # parameter, measurement, observation, comparison
    material_system: Optional[str] = None  # e.g., "silicon", "AlSi10Mg"
    experimental_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Normalization fields
    normalized_name: str = ""          # Standardized name after synonym mapping
    normalized_value: Optional[float] = None  # Numeric value after unit conversion
    normalized_unit: Optional[str] = None     # Standard unit after conversion
    
    def __post_init__(self):
        if not self.normalized_name:
            self.normalized_name = self._normalize_property_name(self.name)
        # Ensure normalized_value is properly set
        if self.normalized_value is None and isinstance(self.value, (int, float)):
            self.normalized_value = self.value
    
    def _normalize_property_name(self, name: str) -> str:
        """Map property name synonyms to canonical form."""
        synonym_map = {
            # Laser domain
            "ablation threshold": "ablation_threshold",
            "threshold fluence": "ablation_threshold",
            "fluence threshold": "ablation_threshold",
            "pulse duration": "pulse_duration",
            "pulse width": "pulse_duration",
            "pulse length": "pulse_duration",
            "wavelength": "wavelength",
            "laser wavelength": "wavelength",
            "repetition rate": "repetition_rate",
            "pulse frequency": "repetition_rate",
            "spot size": "spot_size",
            "beam diameter": "spot_size",
            "fluence": "fluence",
            "laser fluence": "fluence",
            # Materials/mechanical domain
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
            "rockwell hardness": "hardness",
            "brinell hardness": "hardness",
            "young modulus": "elastic_modulus",
            "elastic modulus": "elastic_modulus",
            "shear modulus": "shear_modulus",
            "bulk modulus": "bulk_modulus",
            "fracture toughness": "fracture_toughness",
            "fatigue strength": "fatigue_strength",
            "creep resistance": "creep_resistance",
        }
        name_lower = name.lower().strip()
        return synonym_map.get(name_lower, name_lower.replace(" ", "_"))
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def format_for_display(self) -> str:
        """Format property for human-readable display."""
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
    """
    Aggregates extracted properties from a single document chunk.
    Serves as intermediate representation before cross-document fusion.
    """
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
        """Get all properties matching (normalized) name."""
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
    """
    Result of fusing multiple ExtractedProperty instances across documents.
    Contains consensus value, variation metrics, and source attribution.
    """
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
        """Format for tabular comparison output."""
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
    """
    Quantitative metrics for evaluating information fusion quality.
    Computed per-answer to assess multi-document reasoning effectiveness.
    """
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
        """Compute composite efficiency score (0-1) with fallback for edge cases."""
        weights = {
            "source_diversity": 0.15,
            "property_coverage": 0.20,
            "consistency": 0.25,
            "precision": 0.15,
            "confidence": 0.15,
            "specificity": 0.10
        }
        
        # Handle edge case: no properties extracted
        if self.total_properties_extracted == 0:
            # Return baseline efficiency based on source diversity only
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
        """Format metrics for UI display with null-safety."""
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
    """
    Container for bibliographic metadata extracted from academic documents.
    Supports human-readable citation formatting with multiple fallback strategies.
    """
    
    DOI_PATTERN = re.compile(r'\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b', re.IGNORECASE)
    ARXIV_PATTERN = re.compile(r'\barXiv[:\s]+(\d{4}\.\d{4,5}(v\d+)?)\b', re.IGNORECASE)
    JOURNAL_PATTERNS = [
        re.compile(r'(?:published in|journal|proc\.?|journal of)\s+([A-Z][A-Za-z\s&\.]+?)(?:,|\.)', re.I),
        re.compile(r'([A-Z][A-Za-z\s&\.]+?\s+(?:Letters?|Journal|Transactions|Review|Proceedings))', re.I),
        # ENHANCED: Handle publisher strings like "Elsevier Ltd. All rights reserved"
        re.compile(r'([A-Z][A-Za-z\s&\.]+?(?:Journal|Transactions|Review|Proceedings|Applications|Engineering|Science|Materials|Physics|Chemistry))', re.I),
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
        self.raw_meta: Dict[str, Any] = {}
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
        
        # ENHANCED: Better fallback for complex publisher strings
        base_name = Path(self.source_filename).stem
        # Clean up publisher noise from filename
        clean_name = re.sub(r'\s*(Elsevier|Ltd|All rights reserved|ScienceDirect|Contents lists available).*$', '', base_name, flags=re.I)
        if self.year:
            return f"[{clean_name}, {self.year}]"
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

#
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
            if len(journal) > 10 and not any(bad in journal.lower() for bad in [
                'introduction', 'abstract', 'references', 'elsevier', 'all rights', 'contents lists'
            ]):
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
            field_mapping = {
                '/Title': 'title',
                '/Author': 'authors',
                '/CreationDate': 'year',
                '/Subject': 'journal',
                '/Keywords': 'keywords',
            }
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
            st.warning(f"Could not read PDF meta {e}")
    
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


def extract_metadata_from_text_file(text: str, filename: str) -> BibliographicMeta:
    return extract_metadata_from_pdf_text(text, filename)


# =============================================
# GLOBAL METADATA CACHE
# =============================================

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
    import hashlib
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return ""


# =============================================
# MULTI-DOCUMENT PROPERTY EXTRACTION ENGINE - ENHANCED
# =============================================

class MultiDocumentPropertyExtractor:
    UNIT_CONVERSIONS = {
        "nm": {"factor": 1e-9, "base": "m"},
        "μm": {"factor": 1e-6, "base": "m"},
        "um": {"factor": 1e-6, "base": "m"},
        "mm": {"factor": 1e-3, "base": "m"},
        "fs": {"factor": 1e-15, "base": "s"},
        "ps": {"factor": 1e-12, "base": "s"},
        "ns": {"factor": 1e-9, "base": "s"},
        "J/cm²": {"factor": 1e4, "base": "J/m²"},
        "J/cm2": {"factor": 1e4, "base": "J/m²"},
        "mJ/cm²": {"factor": 10, "base": "J/m²"},
        "MPa": {"factor": 1e6, "base": "Pa"},
        "GPa": {"factor": 1e9, "base": "Pa"},
        "kN": {"factor": 1e3, "base": "N"},
        "HV": {"factor": 1, "base": "HV"},  # Vickers hardness
        "HRC": {"factor": 1, "base": "HRC"},  # Rockwell C
        "HB": {"factor": 1, "base": "HB"},  # Brinell
    }
    
    MATERIAL_SYNONYMS = {
        "si": "silicon", "si substrate": "silicon", "crystalline silicon": "silicon", "c-si": "silicon",
        "aluminum alloy": "aluminum", "alsi10mg": "AlSi10Mg", "al-si10-mg": "AlSi10Mg",
        "ti-6al-4v": "Ti6Al4V", "titanium alloy": "Ti6Al4V",
        "stainless steel": "steel", "ss316l": "steel",
    }
    
    def __init__(self, laser_keywords: Dict[str, List[str]]):
        self.laser_keywords = laser_keywords
        self._compile_extraction_patterns()
    
    def _compile_extraction_patterns(self):
        numeric_pattern = r'([\d.]+(?:\s*[×x*]\s*10\^?-?\d+)?)(?:\s*([±\+-])\s*([\d.]+))?'
        unit_pattern = r'\s*(' + '|'.join(re.escape(u) for u in self.UNIT_CONVERSIONS.keys()) + r')'
        self.property_pattern = re.compile(
            r'([\w\s\-_/]+?)\s*(?:is|was|of|at|:|=|≈|~|yields|results in|produces)\s*' + numeric_pattern + unit_pattern +
            r'(?:\s*[\(\[]([^)\]]+)[\)\]])?', re.I)
        
        # ENHANCED: Pattern for equation-based properties like "Eq. (3) can yield positive, negative, or zero values"
        self.equation_property_pattern = re.compile(
            r'(?:Eq\.?\s*\(?(\d+)\)?|Equation\s+(\d+)).{0,200}?' +
            r'([\w\s]+?)\s*(?:falls below|exceeds|is|yields|produces|results in)\s*' +
            r'([\w\s]+?(?:below|above|zero|positive|negative|standardized))', re.I)
        
        # ENHANCED: Pattern for statistical/standardized values
        self.statistical_pattern = re.compile(
            r'(?:standardized|normalized|scaled)\s+([\w\s]+?)\s+(?:feature|value|variable)\s+' +
            r'(?:below|above|equals)?\s*(zero|0|one|1|mean|average)', re.I)
        
        self.table_row_pattern = re.compile(r'(?:^|\n)\s*[|│]?\s*([^\n|│]+?)\s*[|│]?\s*(?:\n|$)', re.MULTILINE)
        self.latex_cell_pattern = re.compile(r'&\s*([^{&}]+)\s*(?:&|\\\\)')
        material_list = list(self.MATERIAL_SYNONYMS.keys()) + ['silicon', 'steel', 'titanium', 'polymer', 'glass', 'ceramic', 'aluminum', 'composite', 'alloy']
        self.material_property_pattern = re.compile(
            r'(' + '|'.join(re.escape(m) for m in material_list) + r').{0,200}?' +
            r'([\w\s]+?\s*(?:is|was|of|at|:|=)\s*[\d.]+)', re.I | re.DOTALL)
    
    def extract_properties_from_chunk(self, chunk_text: str, chunk_metadata: Dict[str, Any]) -> DocumentFusionRecord:
        record = DocumentFusionRecord(
            source_filename=chunk_metadata.get('source', 'unknown'),
            chunk_index=chunk_metadata.get('chunk_index', 0),
            chunk_id=f"{chunk_metadata.get('source', 'unknown')}:{chunk_metadata.get('chunk_index', 0)}",
            bibliographic_citation=chunk_metadata.get('citation_display', 'Unknown'),
            laser_topics=chunk_metadata.get('laser_topics', []),
            experimental_conditions=chunk_metadata.get('parameters_found', {}),
            material_system=self._detect_material_system(chunk_text),
            processing_method=self._detect_processing_method(chunk_text)
        )
        
        # Log extraction start for debugging
        logger.info(f"Extracting properties from chunk {record.chunk_id}")
        
        table_properties = self._extract_from_tables(chunk_text)
        logger.info(f"  Found {len(table_properties)} table properties")
        for prop in table_properties:
            prop.source_chunk_id = record.chunk_id
            prop.source_citation = record.bibliographic_citation
            prop.material_system = record.material_system
            record.add_property(prop)
        
        inline_properties = self._extract_inline_properties(chunk_text)
        logger.info(f"  Found {len(inline_properties)} inline properties")
        for prop in inline_properties:
            if not any(p.normalized_name == prop.normalized_name and 
                      (abs(p.normalized_value - prop.normalized_value) < 1e-6 if p.normalized_value and prop.normalized_value else False)
                      for p in record.extracted_properties):
                prop.source_chunk_id = record.chunk_id
                prop.source_citation = record.bibliographic_citation
                record.add_property(prop)
        
        # ENHANCED: Extract equation-based and statistical properties
        equation_props = self._extract_equation_properties(chunk_text)
        logger.info(f"  Found {len(equation_props)} equation-based properties")
        for prop in equation_props:
            record.add_property(prop)
        
        statistical_props = self._extract_statistical_properties(chunk_text)
        logger.info(f"  Found {len(statistical_props)} statistical properties")
        for prop in statistical_props:
            record.add_property(prop)
        
        comparative_props = self._extract_comparative_properties(chunk_text)
        for prop in comparative_props:
            record.add_property(prop)
        
        logger.info(f"  Total properties for chunk: {len(record.extracted_properties)}")
        return record
    
    def _detect_material_system(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        for canonical, synonyms in self.MATERIAL_SYNONYMS.items():
            if isinstance(synonyms, list):
                if any(s.lower() in text_lower for s in synonyms):
                    return canonical
            elif synonyms.lower() in text_lower:
                return canonical
        material_match = re.search(r'\b([A-Z][a-z]+(?:[-\s]?[A-Z]?[a-z0-9]+)*)\b', text)
        if material_match:
            candidate = material_match.group(1)
            if any(kw in candidate.lower() for kw in ['silicon', 'titanium', 'aluminum', 'steel', 'polymer', 'glass', 'ceramic', 'composite', 'alloy']):
                return candidate
        return None
    
    def _detect_processing_method(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        methods = [
            ("femtosecond laser", "femtosecond_ablation"), ("picosecond laser", "picosecond_ablation"),
            ("nanosecond laser", "nanosecond_ablation"), ("ultrafast laser", "ultrafast_processing"),
            ("laser ablation", "laser_ablation"), ("aging", "aging_treatment"),
            ("annealing", "annealing"), ("heat treatment", "heat_treatment"),
            ("surface texturing", "surface_texturing"), ("lipss", "lipss_formation"),
            ("additive manufacturing", "additive_manufacturing"), ("3d printing", "additive_manufacturing"),
        ]
        for pattern, canonical in methods:
            if pattern in text_lower:
                return canonical
        return None
    
    def _extract_from_tables(self, text: str) -> List[ExtractedProperty]:
        properties = []
        if r'\begin{tabular}' in text or r'\begin{table}' in text:
            properties.extend(self._parse_latex_table(text))
        elif '|' in text and re.search(r'\|\s*[-:]+\s*\|', text):
            properties.extend(self._parse_markdown_table(text))
        elif self._detect_plain_text_table(text):
            properties.extend(self._parse_plain_text_table(text))
        return properties
    
    def _parse_latex_table(self, latex_text: str) -> List[ExtractedProperty]:
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
        property_cols = [i for i, h in enumerate(header_row) if any(kw in h.lower() for kw in ['strength', 'threshold', 'duration', 'fluence', 'wavelength', 'elongation', 'hardness', 'modulus', 'temperature', 'yield', 'tensile', 'hv', 'hrc', 'hb'])]
        descriptor_cols = [i for i in range(len(header_row)) if i not in property_cols]
        for row in data_rows:
            if len(row) <= max(property_cols, default=-1):
                continue
            row_conditions = {}
            for col_idx in descriptor_cols:
                if col_idx < len(row) and row[col_idx]:
                    cell = row[col_idx].strip()
                    if any(m in cell.lower() for m in ['as-built', 'aged', 'treated', 'composite', 'alloy', 'annealed', 'quenched', 'tempered']):
                        row_conditions['treatment'] = cell
                    elif any(m in cell.lower() for m in list(self.MATERIAL_SYNONYMS.keys()) + ['silicon', 'steel', 'titanium', 'aluminum', 'alloy']):
                        row_conditions['material'] = self.MATERIAL_SYNONYMS.get(cell.lower(), cell)
            for prop_col in property_cols:
                if prop_col >= len(row) or not row[prop_col].strip():
                    continue
                prop_name = header_row[prop_col].strip()
                prop_value_raw = row[prop_col].strip()
                parsed = self._parse_property_value(prop_value_raw, prop_name)
                if parsed:
                    prop = ExtractedProperty(
                        name=prop_name, value=parsed['value'], unit=parsed['unit'],
                        uncertainty=parsed['uncertainty'], condition=self._format_conditions(row_conditions),
                        extraction_confidence=0.85, context_snippet=prop_value_raw, property_type="measurement"
                    )
                    self._normalize_property_units(prop)
                    properties.append(prop)
        return properties
    
    def _parse_markdown_table(self, text: str) -> List[ExtractedProperty]:
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
                if any(kw in header.lower() for kw in ['strength', 'threshold', 'duration', 'fluence', 'elongation', 'hardness', 'modulus', 'yield', 'tensile', 'hv', 'hrc', 'hb']):
                    parsed = self._parse_property_value(value, header)
                    if parsed:
                        prop = ExtractedProperty(
                            name=header, value=parsed['value'], unit=parsed['unit'],
                            uncertainty=parsed['uncertainty'], condition=row_data.get('Material') or row_data.get('Condition'),
                            extraction_confidence=0.8, context_snippet=value, property_type="measurement"
                        )
                        self._normalize_property_units(prop)
                        properties.append(prop)
        return properties
    
    def _parse_plain_text_table(self, text: str) -> List[ExtractedProperty]:
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
                            prop = ExtractedProperty(
                                name=prop_name, value=float(value_match.group(1)),
                                extraction_confidence=0.6, context_snippet=line[:100], property_type="observation"
                            )
                            properties.append(prop)
                        except (ValueError, TypeError):
                            continue
        return properties
    
    def _extract_inline_properties(self, text: str) -> List[ExtractedProperty]:
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
                prop = ExtractedProperty(
                    name=prop_name, value=numeric_value if numeric_value is not None else value_str,
                    unit=unit, uncertainty=uncertainty, condition=condition,
                    extraction_confidence=0.7, context_snippet=match.group(0)[:150],
                    property_type="parameter" if any(kw in prop_name.lower() for kw in ['fluence', 'duration', 'wavelength', 'threshold']) else "measurement"
                )
                self._normalize_property_units(prop)
                properties.append(prop)
        return properties
    
    def _extract_equation_properties(self, text: str) -> List[ExtractedProperty]:
        """ENHANCED: Extract properties from equation-based statements like the user's example."""
        properties = []
        for match in self.equation_property_pattern.finditer(text):
            groups = match.groups()
            if len(groups) >= 4:
                eq_num = groups[0] or groups[1]
                prop_context = groups[2].strip()
                outcome = groups[3].strip()
                
                # Map outcome descriptions to categorical values
                outcome_map = {
                    "positive": 1.0, "negative": -1.0, "zero": 0.0,
                    "below zero": -0.5, "above zero": 0.5,
                    "standardized below zero": -0.5, "standardized above zero": 0.5,
                }
                numeric_outcome = outcome_map.get(outcome.lower(), None)
                
                prop = ExtractedProperty(
                    name=f"eq_{eq_num}_{prop_context.replace(' ', '_')}",
                    value=numeric_outcome if numeric_outcome is not None else outcome,
                    unit="dimensionless" if numeric_outcome is not None else None,
                    condition=f"Eq. ({eq_num})",
                    extraction_confidence=0.6,
                    context_snippet=match.group(0)[:200],
                    property_type="equation_outcome"
                )
                properties.append(prop)
                logger.info(f"  Extracted equation property: {prop.name} = {prop.value}")
        return properties
    
    def _extract_statistical_properties(self, text: str) -> List[ExtractedProperty]:
        """ENHANCED: Extract standardized/normalized value properties."""
        properties = []
        for match in self.statistical_pattern.finditer(text):
            groups = match.groups()
            if len(groups) >= 3:
                feature_name = groups[0].strip()
                reference = groups[2].strip()
                
                # Map reference to numeric value
                ref_map = {"zero": 0.0, "0": 0.0, "one": 1.0, "1": 1.0, "mean": 0.0, "average": 0.0}
                numeric_ref = ref_map.get(reference.lower(), None)
                
                prop = ExtractedProperty(
                    name=f"standardized_{feature_name.replace(' ', '_')}",
                    value=numeric_ref if numeric_ref is not None else reference,
                    unit="std_dev" if numeric_ref is not None else None,
                    condition="statistical normalization",
                    extraction_confidence=0.65,
                    context_snippet=match.group(0)[:150],
                    property_type="statistical_feature"
                )
                properties.append(prop)
                logger.info(f"  Extracted statistical property: {prop.name} = {prop.value}")
        return properties
    
    def _extract_comparative_properties(self, text: str) -> List[ExtractedProperty]:
        properties = []
        comparative_pattern = re.compile(
            r'([\w\s]+?)\s+(?:is|was|shows|exhibits)\s+(?:approximately\s+)?'
            r'([+-]?\s*[\d.]+(?:\s*%|percent)?)\s+(?:higher|lower|greater|less|increased|decreased)'
            r'(?:\s+than|\s+compared to|\s+vs\.?\s+)([\w\s]+)', re.I)
        for match in comparative_pattern.finditer(text):
            prop_name, change_value, reference = match.groups()
            properties.append(ExtractedProperty(
                name=f"{prop_name.strip()}_vs_{reference.strip()}", value=change_value.strip(),
                extraction_confidence=0.65, context_snippet=match.group(0)[:150], property_type="comparison"
            ))
        return properties
    
    def _parse_property_value(self, raw_value: str, prop_name: str) -> Optional[Dict[str, Any]]:
        """FIXED: Robust parsing with error handling for edge cases"""
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
        
        # FIXED: Safe numeric value extraction
        numeric_value = self._safe_parse_numeric(raw_value)
        if numeric_value is not None:
            result["value"] = numeric_value
        else:
            # Keep as string if parsing fails
            result["value"] = raw_value.strip() if raw_value.strip() else None
        
        return result if result["value"] is not None else None
    
    def _safe_parse_numeric(self, value_str: str) -> Optional[float]:
        """
        FIXED: Safely parse numeric value with comprehensive error handling.
        Handles: '.', '-', empty strings, malformed scientific notation, ranges.
        """
        if not value_str:
            return None
        
        # Clean the string
        cleaned = value_str.strip()
        
        # Handle common non-numeric placeholders
        if cleaned in ['.', '-', '--', '...', 'N/A', 'n/a', 'NA', 'na', 'null', 'None', '']:
            return None
        
        # Handle ranges (take first value)
        if '-' in cleaned and not cleaned.startswith('-'):
            parts = cleaned.split('-')
            if len(parts) == 2:
                cleaned = parts[0].strip()
        
        # Handle scientific notation variations
        cleaned = re.sub(r'\s*[×x*]\s*10\^?', 'e', cleaned)
        cleaned = re.sub(r'\s*[×x*]\s*10', 'e', cleaned)
        
        # Extract numeric portion (handle trailing text)
        match = re.match(r'^\s*([+-]?\s*[\d.]+(?:e[+-]?\d+)?)', cleaned, re.I)
        if not match:
            return None
        
        num_str = match.group(1).replace(' ', '')
        
        # Handle edge case: just a decimal point
        if num_str in ['.', '+.', '-.']:
            return None
        
        try:
            return float(num_str)
        except (ValueError, TypeError, OverflowError):
            return None
    
    def _normalize_property_units(self, prop: ExtractedProperty):
        if not prop.unit or prop.unit not in self.UNIT_CONVERSIONS:
            prop.normalized_unit = prop.unit
            if isinstance(prop.value, (int, float)):
                prop.normalized_value = prop.value
            elif prop.normalized_value is None and isinstance(prop.value, str):
                # Try to parse string value as numeric
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
# INFORMATION FUSION ENGINE WITH ENHANCED METRICS
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
            if material_filter and record.material_system != material_filter:
                continue
            if property_filter:
                record.extracted_properties = [p for p in record.extracted_properties if p.normalized_name in property_filter]
            if record.extracted_properties:
                fusion_records.append(record)
        
        # ENHANCED: Handle case where no properties extracted - return baseline metrics
        if not fusion_records:
            logger.warning("No fusion records created - returning baseline metrics")
            metrics = FusionEfficiencyMetrics(
                unique_sources_used=len(retrieved_docs),
                source_diversity_score=min(1.0, len(retrieved_docs) / 3.0),
                overall_fusion_efficiency=min(1.0, len(retrieved_docs) / 3.0) * 0.3  # Baseline
            )
            return {}, metrics
            
        property_groups: Dict[str, List[ExtractedProperty]] = defaultdict(list)
        for record in fusion_records:
            for prop in record.extracted_properties:
                key = prop.normalized_name
                if not property_filter or key in property_filter:
                    property_groups[key].append(prop)
        
        fused_properties: Dict[str, FusedPropertyEntry] = {}
        for prop_name, props in property_groups.items():
            fused = self._fuse_property_group(prop_name, props)
            if fused:
                fused_properties[prop_name] = fused
        
        metrics = self._compute_fusion_metrics(fusion_records, fused_properties, retrieved_docs, query)
        self.fusion_history.append({
            "timestamp": datetime.now().isoformat(), "query": query,
            "input_docs": len(retrieved_docs),
            "extracted_properties": sum(len(r.extracted_properties) for r in fusion_records),
            "fused_properties": len(fused_properties),
            "efficiency": metrics.overall_fusion_efficiency
        })
        return fused_properties, metrics
    
    def _fuse_property_group(self, prop_name: str, properties: List[ExtractedProperty]) -> Optional[FusedPropertyEntry]:
        if not properties:
            return None
        
        # Filter to numeric values for statistical fusion (if applicable)
        numeric_props = [p for p in properties if p.normalized_value is not None and isinstance(p.normalized_value, (int, float))]
        
        # FIXED: Initialize with fused_value=None as default
        fused = FusedPropertyEntry(
            property_name=prop_name,
            fused_value=None,
            unit=properties[0].normalized_unit if properties[0].normalized_unit else properties[0].unit,
            source_count=len(properties),
            sources=[{"citation": p.source_citation, "chunk_id": p.source_chunk_id} for p in properties]
        )
        
        if numeric_props and len(numeric_props) >= 1:
            # Statistical fusion for numeric properties
            values = [p.normalized_value for p in numeric_props if p.normalized_value is not None]
            if values:
                fused.fused_value = np.mean(values)
                fused.value_range = (min(values), max(values))
                fused.standard_deviation = np.std(values) if len(values) > 1 else 0.0
                
                # Determine fusion confidence based on variation
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
            # Non-numeric or categorical fusion: use most frequent value
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
    
    def _compute_fusion_metrics(self, fusion_records: List[DocumentFusionRecord],
                               fused_properties: Dict[str, FusedPropertyEntry],
                               retrieved_docs: List[Document], query: str) -> FusionEfficiencyMetrics:
        metrics = FusionEfficiencyMetrics()
        
        # Source diversity
        unique_sources = set(r.chunk_id for r in fusion_records)
        metrics.unique_sources_used = len(unique_sources)
        metrics.source_diversity_score = min(1.0, len(unique_sources) / 3.0)
        
        # Property coverage
        total_extracted = sum(len(r.extracted_properties) for r in fusion_records)
        metrics.total_properties_extracted = total_extracted
        metrics.properties_fused_successfully = len(fused_properties)
        metrics.property_coverage_ratio = len(fused_properties) / total_extracted if total_extracted > 0 else 0.0
        
        # Consistency analysis
        if fused_properties:
            consistent = sum(1 for f in fused_properties.values() if not f.conflicts_detected and f.fusion_confidence != FusionConfidence.LOW)
            conflicting = sum(1 for f in fused_properties.values() if f.conflicts_detected)
            total_evaluated = consistent + conflicting
            metrics.consistent_properties = consistent
            metrics.conflicting_properties = conflicting
            metrics.consistency_ratio = consistent / total_evaluated if total_evaluated > 0 else 1.0
        else:
            metrics.consistency_ratio = 1.0  # No conflicts if no properties
        
        # Precision metrics (uncertainty analysis)
        numeric_with_uncertainty = [f for f in fused_properties.values() if f.standard_deviation is not None or any("±" in str(s.get("citation", "")) for s in f.sources)]
        metrics.numeric_properties_with_uncertainty = len(numeric_with_uncertainty)
        if fused_properties:
            uncertainties = []
            for f in fused_properties.values():
                if isinstance(f.fused_value, (int, float)) and f.fused_value != 0 and f.standard_deviation is not None:
                    uncertainties.append(f.standard_deviation / abs(f.fused_value))
            if uncertainties:
                metrics.average_uncertainty_magnitude = np.mean(uncertainties)
            else:
                metrics.average_uncertainty_magnitude = 0.1  # Default baseline
        else:
            metrics.average_uncertainty_magnitude = 0.1
        
        # Confidence aggregation
        confidence_weights = {FusionConfidence.HIGH: 1.0, FusionConfidence.MODERATE: 0.7, FusionConfidence.LOW: 0.4, FusionConfidence.UNKNOWN: 0.2}
        if fused_properties:
            weighted_sum = sum(confidence_weights.get(f.fusion_confidence, 0.5) for f in fused_properties.values())
            metrics.weighted_confidence_score = weighted_sum / len(fused_properties)
            metrics.high_confidence_fusions = sum(1 for f in fused_properties.values() if f.fusion_confidence == FusionConfidence.HIGH)
            metrics.low_confidence_fusions = sum(1 for f in fused_properties.values() if f.fusion_confidence == FusionConfidence.LOW)
        else:
            metrics.weighted_confidence_score = 0.5  # Default baseline
        
        # Answer quality proxies
        metrics.answer_specificity_score = self._estimate_answer_specificity(query, fused_properties)
        metrics.citation_density = min(1.0, len(fused_properties) * 2 / 100)
        
        # Compute composite with fallback
        metrics.compute_overall()
        
        return metrics
    
    def _estimate_answer_specificity(self, query: str, fused_props: Dict[str, FusedPropertyEntry]) -> float:
        if not fused_props:
            # ENHANCED: Return baseline specificity based on query content
            query_lower = query.lower()
            if any(kw in query_lower for kw in ['compare', 'versus', 'vs', 'difference', 'threshold', 'strength', 'hardness']):
                return 0.5
            return 0.3
        
        query_lower = query.lower()
        specificity_indicators = 0
        for prop_name in fused_props.keys():
            if prop_name.replace('_', ' ') in query_lower or prop_name in query_lower:
                specificity_indicators += 2
        if any(mat in query_lower for mat in ['silicon', 'aluminum', 'titanium', 'steel', 'composite', 'alloy']):
            specificity_indicators += 1
        if any(param in query_lower for param in ['fluence', 'threshold', 'duration', 'wavelength', 'strength', 'hardness']):
            specificity_indicators += 1
        if re.search(r'[\d.]+\s*(?:j/cm|mpa|fs|nm|%|percent|hv|hrc)', query_lower):
            specificity_indicators += 2
        return min(1.0, specificity_indicators / 5.0)
    
    def generate_comparison_table(self, fused_properties: Dict[str, FusedPropertyEntry], format: str = "markdown") -> str:
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
    
    def _generate_markdown_table(self, fused_props: Dict[str, FusedPropertyEntry]) -> str:
        lines = []
        lines.append("| Property | Value | Unit | Range | Sources | Confidence |")
        lines.append("|----------|-------|------|-------|---------|------------|")
        for prop_name, entry in sorted(fused_props.items(), key=lambda x: x[0]):
            if entry.fused_value is not None and isinstance(entry.fused_value, (int, float)):
                value_str = f"{entry.fused_value:.3f}"
            else:
                value_str = str(entry.fused_value) if entry.fused_value is not None else "–"
            if entry.uncertainty and entry.uncertainty not in value_str:
                value_str = f"{value_str} {entry.uncertainty}"
            range_str = f"{entry.value_range[0]:.2f}–{entry.value_range[1]:.2f}" if entry.value_range else "–"
            confidence_icon = {"high": "🟢", "moderate": "🟡", "low": "🔴", "unknown": "⚪"}.get(entry.fusion_confidence.value, "⚪")
            lines.append(f"| {prop_name.replace('_', ' ').title()} | {value_str} | {entry.unit or '–'} | {range_str} | {entry.source_count} | {confidence_icon} {entry.fusion_confidence.value} |")
        return "\n".join(lines)
    
    def _generate_latex_table(self, fused_props: Dict[str, FusedPropertyEntry]) -> str:
        lines = [r"\begin{tabular}{|l|c|c|c|c|c|}", r"\hline",
                r"\textbf{Property} & \textbf{Value} & \textbf{Unit} & \textbf{Range} & \textbf{Sources} & \textbf{Confidence} \\", r"\hline"]
        for prop_name, entry in sorted(fused_props.items()):
            if entry.fused_value is not None and isinstance(entry.fused_value, (int, float)):
                value_str = f"{entry.fused_value:.3f}"
            else:
                value_str = str(entry.fused_value) if entry.fused_value is not None else "--"
            range_str = f"{entry.value_range[0]:.2f}--{entry.value_range[1]:.2f}" if entry.value_range else "--"
            conf_symbol = {"high": "high", "moderate": "mod", "low": "low"}.get(entry.fusion_confidence.value, "?")
            lines.append(f"{prop_name.replace('_', r'\_').title()} & {value_str} & {entry.unit or '--'} & {range_str} & {entry.source_count} & {conf_symbol} \\\\")
        lines.extend([r"\hline", r"\end{tabular}"])
        return "\n".join(lines)
    
    def _generate_html_table(self, fused_props: Dict[str, FusedPropertyEntry]) -> str:
        lines = ['<table class="fusion-table" style="border-collapse: collapse; width: 100%;">']
        lines.append('<thead><tr style="background: #f0f9ff;">')
        for header in ["Property", "Value", "Unit", "Range", "Sources", "Confidence"]:
            lines.append(f'<th style="border: 1px solid #ccc; padding: 8px; text-align: left;">{header}</th>')
        lines.append('</tr></thead><tbody>')
        for prop_name, entry in fused_props.items():
            if entry.fused_value is not None and isinstance(entry.fused_value, (int, float)):
                value_str = f"{entry.fused_value:.3f}"
            else:
                value_str = str(entry.fused_value) if entry.fused_value is not None else "–"
            bg_color = {"high": "#dcfce7", "moderate": "#fef3c7", "low": "#fee2e2"}.get(entry.fusion_confidence.value, "#f1f5f9")
            lines.append(f'<tr style="background: {bg_color};">')
            lines.append(f'<td style="border: 1px solid #ccc; padding: 8px;">{prop_name.replace("_", " ").title()}</td>')
            lines.append(f'<td style="border: 1px solid #ccc; padding: 8px;">{value_str}</td>')
            lines.append(f'<td style="border: 1px solid #ccc; padding: 8px;">{entry.unit or "–"}</td>')
            range_display = f"{entry.value_range[0]:.2f}–{entry.value_range[1]:.2f}" if entry.value_range else "–"
            lines.append(f'<td style="border: 1px solid #ccc; padding: 8px;">{range_display}</td>')
            lines.append(f'<td style="border: 1px solid #ccc; padding: 8px; text-align: center;">{entry.source_count}</td>')
            lines.append(f'<td style="border: 1px solid #ccc; padding: 8px;">{entry.fusion_confidence.value.title()}</td>')
            lines.append('</tr>')
        lines.append('</tbody></table>')
        return "\n".join(lines)
    
    def _generate_plain_text_table(self, fused_props: Dict[str, FusedPropertyEntry]) -> str:
        lines = []
        lines.append(f"{'Property':<30} {'Value':<15} {'Unit':<10} {'Confidence':<10}")
        lines.append("-" * 70)
        for prop_name, entry in fused_props.items():
            if entry.fused_value is not None and isinstance(entry.fused_value, (int, float)):
                value_str = f"{entry.fused_value:.3f}"
            else:
                value_str = str(entry.fused_value) if entry.fused_value is not None else "–"
            lines.append(f"{prop_name.replace('_', ' ').title():<30} {value_str:<15} {entry.unit or '–':<10} {entry.fusion_confidence.value:<10}")
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
        "debug_extraction": False,  # ENHANCED: Toggle for extraction debugging
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
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        return total_memory - reserved
    except:
        return None

def estimate_model_memory(model_key: str, use_4bit: bool = False) -> Dict[str, any]:
    repo_id = get_hf_repo_id(model_key) if not is_ollama_model(model_key) else model_key
    return MODEL_MEMORY_ESTIMATES.get(repo_id, {"params": "Unknown", "vram_fp16": "Unknown", "vram_4bit": "Unknown", "cpu_ok": False})


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
            st.sidebar.warning("⚠️ bitsandbytes not installed. Install with: pip install bitsandbytes")
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
# DOCUMENT LOADING & CHUNKING WITH FUSION SUPPORT
# =============================================

def extract_laser_metadata(text: str, filename: str) -> Dict[str, any]:
    metadata = {"source": filename, "laser_topics": [], "parameters_found": {},
                "has_equations": bool(re.search(r'[\(=]\s*[\d.]+\s*[×*]\s*10\^', text)),
                "has_figures": bool(re.search(r'Figure\s*\d+|Fig\.\s*\d+', text, re.I))}
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

def load_and_chunk_laser_documents(uploaded_files: List) -> List[Document]:
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
                st.info(f"📚 Extracted meta {bib_meta.format_citation('apa')}")
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, encoding='utf-8')
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=LASER_DOMAIN_CONFIG["chunk_size"],
                chunk_overlap=LASER_DOMAIN_CONFIG["chunk_overlap"],
                separators=["\n\n", "\n", "Equation", "Parameter:", "Figure", "Table", ""],
                length_function=len
            )
            chunks = text_splitter.split_documents(pages)
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "source": uploaded_file.name, "chunk_index": i, "total_chunks": len(chunks),
                    **extract_laser_metadata(chunk.page_content, uploaded_file.name),
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


# =============================================
# VECTOR STORE CREATION
# =============================================

@st.cache_resource
def create_local_vector_store(chunks: List[Document], embedding_model_key: str):
    try:
        embeddings = load_local_embeddings()
        if embeddings is None:
            return None
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.metadata = {
            "total_chunks": len(chunks), "embedding_model": embedding_model_key,
            "created_at": datetime.now().isoformat(),
            "laser_topics": list(set(topic for chunk in chunks for topic in chunk.metadata.get("laser_topics", [])))
        }
        return vectorstore
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None


# =============================================
# ENHANCED RAG CHAIN WITH MULTI-DOCUMENT FUSION
# =============================================

def create_laser_rag_prompt(retrieved_chunks: List[Document], query: str) -> str:
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
    laser_system = """You are an expert assistant for laser-microstructure interaction research.
Your role is to answer questions based ONLY on the provided document context.
Be precise, technical, and cite your sources using the provided citation format.
Rules:
1. Use ONLY information from the retrieved context below
2. If the answer isn't in the context, say "Based on the provided documents, I cannot determine..."
3. Never invent parameters, equations, or experimental conditions
4. When citing, use the EXACT citation string provided (e.g., "Smith et al., J. Appl. Phys., 2023" or "DOI:10.1063/1.234567")
5. For numerical values, include units when available
6. Be concise but technically complete
"""
    user_query = f"""Retrieved Context from Laser Microstructure Documents:
{context}
User Question: {query}
Answer (cite sources using provided citation format, be technical and precise):"""
    return laser_system + user_query

def _create_fusion_aware_prompt(retrieved_docs: List[Document], query: str,
                               fused_properties: Dict[str, FusedPropertyEntry],
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
        properties_summary = "**Fused Property Summary**:\n"
        for prop_name, entry in list(fused_properties.items())[:8]:
            if entry.fused_value is not None and isinstance(entry.fused_value, (int, float)):
                value_str = f"{entry.fused_value:.3f}"
            else:
                value_str = str(entry.fused_value) if entry.fused_value is not None else "N/A"
            properties_summary += f"• {prop_name.replace('_', ' ').title()}: {value_str} {entry.unit or ''} [conf: {entry.fusion_confidence.value}, sources: {entry.source_count}]\n"
        properties_summary += "\n"
    table_section = f"**Comparison Table**:\n{comparison_table}\n\n" if comparison_table else ""
    efficiency_note = ""
    if fusion_metrics.overall_fusion_efficiency >= 0.7:
        efficiency_note = f"🎯 High-confidence fusion ({fusion_metrics.overall_fusion_efficiency:.2f}/1.0): Properties synthesized from {fusion_metrics.unique_sources_used} sources.\n\n"
    elif fusion_metrics.overall_fusion_efficiency >= 0.4:
        efficiency_note = f"⚠️ Moderate-confidence fusion ({fusion_metrics.overall_fusion_efficiency:.2f}/1.0): Some property variations detected across sources.\n\n"
    else:
        efficiency_note = f"🔍 Low-confidence fusion ({fusion_metrics.overall_fusion_efficiency:.2f}/1.0): Limited or conflicting data; interpret with caution.\n\n"
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
    user_prompt = f"""RETRIEVED DOCUMENT CONTEXT:
{context}
{efficiency_note}{properties_summary}{table_section}
USER QUESTION: {query}
SCIENTIFIC ANSWER (use fused properties when available, cite sources precisely):"""
    full_prompt = system_prompt + user_prompt
    context_metadata = {"fused_properties_count": len(fused_properties), "fusion_efficiency": fusion_metrics.overall_fusion_efficiency, "comparison_table_available": comparison_table is not None}
    return full_prompt, context_metadata

def generate_local_response_transformers(tokenizer, model, device: str, prompt: str, backend_name: str) -> str:
    try:
        if "Qwen" in backend_name or "qwen" in backend_name.lower():
            messages = [{"role": "system", "content": "You are an expert in laser-microstructure interaction."}, {"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        elif "Llama" in backend_name or "llama" in backend_name.lower():
            messages = [{"role": "system", "content": "You are an expert in laser-microstructure interaction."}, {"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        elif "Mistral" in backend_name or "mistral" in backend_name.lower():
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        else:
            formatted_prompt = prompt
        inputs = tokenizer.encode(formatted_prompt, return_tensors='pt', truncation=True, max_length=LASER_DOMAIN_CONFIG["max_context_tokens"])
        if device == "cuda" and torch.cuda.is_available():
            inputs = inputs.to('cuda')
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=LASER_DOMAIN_CONFIG["max_new_tokens"],
                                    temperature=LASER_DOMAIN_CONFIG["temperature"],
                                    do_sample=(LASER_DOMAIN_CONFIG["temperature"] > 0),
                                    pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
                                    no_repeat_ngram_size=3, early_stopping=True)
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in full_text:
            answer = full_text.split("[/INST]")[-1].strip()
        elif "Answer (cite sources" in full_text:
            answer = full_text.split("Answer (cite sources")[-1].strip()
            answer = re.split(r'\n\n(?:Question|User|Context):', answer)[0].strip()
        else:
            answer = full_text[-LASER_DOMAIN_CONFIG["max_new_tokens"]*2:].strip()
        answer = re.sub(r'\s+', ' ', answer).strip()
        return answer if answer else "I was unable to generate a response. Please try rephrasing your question."
    except Exception as e:
        st.error(f"Generation error: {e}")
        return f"Error generating response: {str(e)[:200]}..."

def generate_local_response_ollama(model_tag: str, ollama_host: str, prompt: str) -> str:
    try:
        client = ollama.Client(host=ollama_host)
        messages = [{"role": "system", "content": "You are an expert in laser-microstructure interaction research. Answer based ONLY on the provided context."}, {"role": "user", "content": prompt}]
        try:
            response = client.chat(model=model_tag, messages=messages,
                                  options={"temperature": LASER_DOMAIN_CONFIG["temperature"], "num_predict": LASER_DOMAIN_CONFIG["max_new_tokens"]}, stream=True)
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
            response = client.chat(model=model_tag, messages=messages,
                                  options={"temperature": LASER_DOMAIN_CONFIG["temperature"], "num_predict": LASER_DOMAIN_CONFIG["max_new_tokens"]})
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

def retrieve_and_answer(vectorstore, tokenizer, model, device_or_host: str, backend: str, backend_type: str,
                       query: str, k: int = None, score_threshold: float = None) -> Tuple[str, List[Document], float]:
    k = k or LASER_DOMAIN_CONFIG["retrieval_k"]
    score_threshold = score_threshold or LASER_DOMAIN_CONFIG["score_threshold"]
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": k, "score_threshold": score_threshold})
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
        return "Based on the uploaded documents, I could not find information relevant to your question. Try rephrasing or checking document content.", [], avg_relevance
    prompt = create_laser_rag_prompt(retrieved_docs, query)
    answer = generate_local_response(tokenizer=tokenizer, model_or_tag=model, device_or_host=device_or_host,
                                    prompt=prompt, backend=backend, backend_type=backend_type)
    return answer, retrieved_docs, avg_relevance

def retrieve_and_answer_with_fusion(vectorstore, tokenizer, model, device_or_host: str, backend: str, backend_type: str,
                                   query: str, k: int = None, score_threshold: float = None,
                                   enable_fusion: bool = True, material_filter: Optional[str] = None,
                                   property_filter: Optional[List[str]] = None) -> Tuple[str, List[Document], float, Dict[str, Any]]:
    k = k or LASER_DOMAIN_CONFIG["retrieval_k"]
    score_threshold = score_threshold or LASER_DOMAIN_CONFIG["score_threshold"]
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": k, "score_threshold": score_threshold})
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
        return ("Based on the uploaded documents, I could not find information relevant to your question. Try rephrasing or checking document content.",
                [], avg_relevance, {"error": "no_relevant_chunks", "fusion_enabled": enable_fusion})
    if enable_fusion:
        property_extractor = MultiDocumentPropertyExtractor(LASER_KEYWORDS)
        fusion_engine = MultiDocumentFusionEngine(property_extractor)
        fused_properties, fusion_metrics = fusion_engine.fuse_documents(retrieved_docs, query, material_filter=material_filter, property_filter=property_filter)
        comparison_table = None
        if fused_properties:
            comparison_table = fusion_engine.generate_comparison_table(fused_properties, format="markdown")
        prompt, fusion_context = _create_fusion_aware_prompt(retrieved_docs, query, fused_properties, fusion_metrics, comparison_table)
        answer = generate_local_response(tokenizer=tokenizer, model_or_tag=model, device_or_host=device_or_host,
                                        prompt=prompt, backend=backend, backend_type=backend_type)
        if fusion_metrics.overall_fusion_efficiency > 0.5 and comparison_table:
            answer += f"\n\n---\n**📊 Property Comparison**:\n{comparison_table}"
        metadata = {
            "fusion_enabled": True,
            "fusion_metrics": {"efficiency": fusion_metrics.overall_fusion_efficiency, "display": fusion_metrics.to_display_dict()},
            "fused_properties": {k: v.to_comparison_row() for k, v in fused_properties.items()},
            "comparison_table": comparison_table,
            "source_citations": [{"citation": doc.metadata.get('citation_display', 'Unknown'), "relevance": scores[i] if i < len(scores) else 0, "topics": doc.metadata.get('laser_topics', [])} for i, doc in enumerate(retrieved_docs)],
            "retrieval_relevance": avg_relevance
        }
        return answer, retrieved_docs, avg_relevance, metadata
    else:
        prompt = create_laser_rag_prompt(retrieved_docs, query)
        answer = generate_local_response(tokenizer=tokenizer, model_or_tag=model, device_or_host=device_or_host,
                                        prompt=prompt, backend=backend, backend_type=backend_type)
        return answer, retrieved_docs, avg_relevance, {"fusion_enabled": False}


# =============================================
# STREAMLIT UI COMPONENTS - ENHANCED
# =============================================

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        backend_option = st.radio("🔧 Inference Backend", options=["Hugging Face Transformers", "Ollama (if installed)"], index=0,
                                 help="Transformers: direct HF model loading\nOllama: use local ollama serve (faster switching)")
        st.session_state.inference_backend = backend_option
        if backend_option == "Ollama (if installed)":
            if not OLLAMA_AVAILABLE:
                st.error("❌ ollama library not installed")
                st.code("pip install ollama")
                st.info("Also ensure Ollama server is running: ollama serve")
            available_ollama_models = [k for k in LOCAL_LLM_OPTIONS.keys() if is_ollama_model(k)]
            model_choice = st.selectbox("🧠 Local LLM Backend (Ollama)",
                                       options=available_ollama_models if available_ollama_models else ["No Ollama models available"],
                                       index=0 if available_ollama_models else 0, help="Models served via local Ollama instance")
        else:
            hf_models = [k for k in LOCAL_LLM_OPTIONS.keys() if not is_ollama_model(k)]
            model_choice = st.selectbox("🧠 Local LLM Backend (Hugging Face)", options=hf_models, index=2,
                                       help="Models loaded directly via transformers library")
        st.session_state.llm_model_choice = model_choice
        if backend_option == "Hugging Face Transformers" and not is_ollama_model(model_choice):
            st.session_state.use_4bit_quantization = st.checkbox("🗜️ Use 4-bit quantization (reduces VRAM usage)", value=True,
                                                                help="Enable for models >3B parameters to reduce memory usage by ~75%")
        if backend_option == "Ollama (if installed)" or is_ollama_model(model_choice):
            st.session_state.ollama_host = st.text_input("🌐 Ollama Host", value=st.session_state.ollama_host,
                                                        help="URL of your Ollama server (default: http://localhost:11434)")
        st.markdown("#### 🔬 Laser Domain Settings")
        st.session_state.laser_domain_boost = st.checkbox("Boost laser-topic relevance", value=True,
                                                         help="Prioritize chunks containing laser-specific keywords")
        st.session_state.show_sources = st.checkbox("Show source citations", value=True,
                                                   help="Display which documents chunks came from")
        st.session_state.enable_multi_doc_fusion = st.checkbox("🔗 Enable Multi-Document Fusion", value=True,
                                                              help="Enable cross-document property extraction and comparison")
        # ENHANCED: Debug toggle for extraction diagnostics
        st.session_state.debug_extraction = st.checkbox("🐛 Debug Property Extraction", value=False,
                                                       help="Show extracted properties in UI for diagnosis")
        st.markdown("#### 📝 Citation Format")
        st.session_state.citation_style = st.selectbox("Citation display style", options=["apa", "doi", "full", "short"], index=0,
                                                      format_func=lambda x: {"apa": "APA: FirstAuthor et al., Journal, Year", "doi": "DOI: 10.xxxx/xxxxx",
                                                                            "full": "Full: Author (Year). Title. Journal, Vol(Issue), Pages",
                                                                            "short": "Short: [FirstAuthor Year] or [DOI]"}[x],
                                                      help="How citations appear in responses and source lists")
        st.session_state.max_retrieved_chunks = st.slider("Chunks to retrieve", min_value=2, max_value=8, value=4,
                                                         help="More chunks = more context but slower responses")
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
    st.markdown("### 📁 Upload Laser Microstructure Documents")
    uploaded_files = st.file_uploader("Select PDF or TXT files about laser processing, ablation, microstructuring, etc.",
                                     type=["pdf", "txt"], accept_multiple_files=True,
                                     help="Documents will be processed locally - no data leaves your browser session. Bibliographic metadata (DOI, authors, journal, year) will be extracted for human-readable citations.")
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
    with st.spinner(f"Processing {len(new_files)} document(s) and extracting bibliographic metadata..."):
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
            st.success(f"✅ Ready! Indexed {len(st.session_state.all_chunks)} chunks from {len(st.session_state.processed_files)} files")
            st.session_state.processing_complete = True
            return True
        except Exception as e:
            st.error(f"Processing failed: {e}")
            import traceback
            st.error(traceback.format_exc())
            return False

def render_fusion_metrics_panel(fusion_meta: Dict[str, Any]):
    if not fusion_metadata.get("fusion_enabled"):
        return
    metrics_display = fusion_metadata.get("fusion_metrics", {}).get("display", {})
    if not metrics_display:
        return
    with st.expander("📊 Information Fusion Efficiency", expanded=True):
        overall = fusion_metadata["fusion_metrics"]["efficiency"]
        # ENHANCED: Handle None/zero values gracefully
        if overall is not None:
            st.progress(min(1.0, max(0.0, overall)))
            st.caption(f"Overall Fusion Efficiency: {overall:.2f}/1.0")
        else:
            st.caption("Overall Fusion Efficiency: N/A")
        
        cols = st.columns(2)
        metric_items = list(metrics_display.items())
        for i, (label, value) in enumerate(metric_items):
            with cols[i % 2]:
                # ENHANCED: Null-safe value display
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

def render_extracted_properties_debug(extracted_props: List[ExtractedProperty], source_citation: str):
    """ENHANCED: Debug panel showing extracted properties for diagnosis."""
    if not extracted_props:
        st.info("🔍 No properties extracted from this chunk")
        return
    
    with st.expander(f"🐛 Extracted Properties: {source_citation}", expanded=False):
        for i, prop in enumerate(extracted_props, 1):
            st.markdown(f"**{i}. {prop.normalized_name}**")
            st.caption(f"Value: `{prop.value}` {prop.unit or ''} | Type: {prop.property_type}")
            if prop.condition:
                st.caption(f"Condition: {prop.condition}")
            if prop.context_snippet:
                st.code(prop.context_snippet[:200] + "..." if len(prop.context_snippet) > 200 else prop.context_snippet, language="text")
            st.divider()

def render_comparison_table_in_chat(comparison_table: Optional[str], fused_properties: Dict):
    if not comparison_table:
        return
    with st.expander("📋 Property Comparison Table", expanded=False):
        st.markdown(comparison_table, unsafe_allow_html=True)
        if fused_properties:
            selected_prop = st.selectbox("🔍 Explore property details:", options=["Select a property..."] + list(fused_properties.keys()), key="fusion_prop_select")
            if selected_prop and selected_prop != "Select a property...":
                prop_data = fused_properties[selected_prop]
                st.json({"property": selected_prop, "fused_value": prop_data["value"], "unit": prop_data["unit"],
                        "range": prop_data["range"], "sources": prop_data["sources"], "confidence": prop_data["confidence"]})

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
                st.session_state.llm_tokenizer = tokenizer
                st.session_state.llm_model = model
                st.session_state.llm_device_or_host = device_or_host
                st.session_state.llm_backend_type = loaded_backend
                st.success("✓ Model loaded!")
            else:
                st.error("Failed to load model. Try selecting a different option.")
                return
    has_model = (st.session_state.llm_backend_type == "ollama" and st.session_state.llm_model is not None) or (st.session_state.llm_backend_type == "transformers" and st.session_state.llm_tokenizer is not None)
    if not has_model:
        st.warning("Please select and load a model in the sidebar first")
        return
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
                                if bib.get('volume'):
                                    vol_str = f"Vol. {bib['volume']}"
                                    if bib.get('issue'):
                                        vol_str += f"({bib['issue']})"
                                    st.markdown(f"**Volume/Issue:** {vol_str}")
                                st.caption(f"Extraction method: {bib.get('extraction_method', 'unknown')} (confidence: {bib.get('confidence', 0):.2f})")
                        st.markdown(f"> {src.page_content[:300]}...")
                        
                        # ENHANCED: Show extracted properties if debug mode enabled
                        if st.session_state.debug_extraction:
                            # Re-extract properties for display (in production, cache this)
                            extractor = MultiDocumentPropertyExtractor(LASER_KEYWORDS)
                            record = extractor.extract_properties_from_chunk(src.page_content, src.metadata)
                            render_extracted_properties_debug(record.extracted_properties, citation)
            
            # Display fusion metadata if present
            if message.get("fusion_metadata") and st.session_state.enable_multi_doc_fusion:
                render_fusion_metrics_panel(message["fusion_metadata"])
                if message["fusion_metadata"].get("comparison_table"):
                    render_comparison_table_in_chat(message["fusion_metadata"]["comparison_table"], message["fusion_metadata"].get("fused_properties", {}))
    
    if prompt := st.chat_input("Ask about laser parameters, material properties, or compare studies..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("🔍 Retrieving, fusing multi-document data, and generating..."):
                try:
                    if st.session_state.enable_multi_doc_fusion:
                        answer, retrieved_docs, relevance, metadata = retrieve_and_answer_with_fusion(
                            vectorstore=st.session_state.vectorstore, tokenizer=st.session_state.llm_tokenizer,
                            model=st.session_state.llm_model, device_or_host=st.session_state.llm_device_or_host,
                            backend=st.session_state.llm_model_choice, backend_type=st.session_state.llm_backend_type,
                            query=prompt, k=st.session_state.max_retrieved_chunks, enable_fusion=True)
                    else:
                        answer, retrieved_docs, relevance = retrieve_and_answer(
                            vectorstore=st.session_state.vectorstore, tokenizer=st.session_state.llm_tokenizer,
                            model=st.session_state.llm_model, device_or_host=st.session_state.llm_device_or_host,
                            backend=st.session_state.llm_model_choice, backend_type=st.session_state.llm_backend_type,
                            query=prompt, k=st.session_state.max_retrieved_chunks)
                        metadata = {"fusion_enabled": False}
                    display_text = ""
                    for word in answer.split():
                        display_text += word + " "
                        message_placeholder.markdown(display_text + "▌")
                        time.sleep(0.02)
                    message_placeholder.markdown(answer)
                    message_dict = {"role": "assistant", "content": answer, "sources": retrieved_docs if st.session_state.show_sources else None, "relevance": relevance}
                    if st.session_state.enable_multi_doc_fusion:
                        message_dict["fusion_metadata"] = metadata
                    st.session_state.messages.append(message_dict)
                    if relevance > 0:
                        fusion_eff = metadata.get("fusion_metrics", {}).get("efficiency", 0) if metadata.get("fusion_enabled") else None
                        if fusion_eff is not None:
                            st.caption(f"📊 Response relevance: {relevance:.2f}/1.0 | Fusion efficiency: {fusion_eff:.2f}/1.0")
                        else:
                            st.caption(f"📊 Response relevance: {relevance:.2f}/1.0")
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)[:300]}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

def render_footer():
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📚 Example Questions:**")
        st.caption("• What is the ablation threshold for silicon at 800nm?")
        st.caption("• How does pulse duration affect LIPSS formation?")
        st.caption("• Compare yield strength of AlSi10Mg under different treatments")
    with col2:
        st.markdown("**⚡ Performance Tips:**")
        st.caption("• Keep questions focused and specific")
        st.caption("• Smaller chunks = more precise retrieval")
        st.caption("• CPU mode: allow 10-30s per response; GPU: 2-10s")
    with col3:
        st.markdown("**🔐 Privacy & Fusion:**")
        st.caption("• All processing happens locally in your session")
        st.caption("• Multi-document fusion extracts & compares properties across studies")
        st.caption("• Fusion efficiency metrics quantify synthesis quality")
        st.caption("• Citations display as 'FirstAuthor et al., Journal, Year' or DOI")
        if st.session_state.debug_extraction:
            st.caption("🐛 Debug mode: Property extraction details visible")

# =============================================
# MAIN APPLICATION
# =============================================

def main():
    st.set_page_config(page_title="🔬 Laser Microstructure RAG Assistant", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""<style>
    .main-header {font-size: 2.5rem; background: linear-gradient(90deg, #1e40af, #7c3aed, #059669); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; text-align: center; padding: 1rem 0;}
    .info-card {background: #f8fafc; border-left: 4px solid #3b82f6; padding: 1rem; border-radius: 0 0.5rem 0.5rem 0; margin: 0.5rem 0;}
    .stChatMessage {border-radius: 0.5rem; margin: 0.25rem 0;}
    .model-warning {background: #fef3c7; border-left: 4px solid #f59e0b; padding: 0.75rem; border-radius: 0 0.5rem 0.5rem 0; margin: 0.5rem 0;}
    .citation-badge {display: inline-block; background: #e0e7ff; color: #3730a3; padding: 0.2rem 0.5rem; border-radius: 0.25rem; font-size: 0.85rem; margin: 0.1rem 0;}
    .fusion-badge {display: inline-block; background: #dcfce7; color: #166534; padding: 0.2rem 0.5rem; border-radius: 0.25rem; font-size: 0.85rem; margin: 0.1rem 0;}
    .debug-badge {display: inline-block; background: #fef3c7; color: #92400e; padding: 0.2rem 0.5rem; border-radius: 0.25rem; font-size: 0.85rem; margin: 0.1rem 0;}
    </style>""", unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🔬 Laser Microstructure RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown("""<div style="text-align:center;color:#64748b;margin-bottom:1.5rem">
    Upload research papers, experimental reports, or simulation data about laser-matter interaction.
    Ask questions and get answers with <strong>human-readable citations</strong> (DOI, Author-Year-Journal) - all running locally, no API keys required.
    <br><span class="fusion-badge">🔗 Multi-document property fusion with efficiency metrics</span>
    {"🐛 Debug extraction" if st.session_state.debug_extraction else ""}
    </div>""", unsafe_allow_html=True)
    initialize_session_state()
    render_sidebar()
    if st.session_state.llm_model_choice and not is_ollama_model(st.session_state.llm_model_choice):
        mem_info = estimate_model_memory(st.session_state.llm_model_choice, st.session_state.get('use_4bit_quantization', True))
        available_vram = get_available_gpu_memory()
        if available_vram and not mem_info['cpu_ok']:
            required = float(mem_info['vram_4bit'].replace('GB','').replace('~','').strip()) if 'GB' in mem_info['vram_4bit'] else 100
            if available_vram < required:
                st.markdown(f"""<div class="model-warning">⚠️ <strong>Memory Warning:</strong> {st.session_state.llm_model_choice} requires ~{mem_info['vram_4bit']} VRAM.
                You have ~{available_vram:.1f}GB available. Consider:
                <ul><li>Using 4-bit quantization (already enabled)</li><li>Selecting a smaller model</li><li>Using Ollama backend for better memory management</li></ul></div>""", unsafe_allow_html=True)
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
            st.markdown("""<div class="info-card"><h3>👋 Welcome!</h3>
            <p>This assistant helps you query documents about:</p>
            <ul><li>🔥 Laser ablation thresholds & mechanisms</li><li>🌊 LIPSS and surface morphology formation</li>
            <li>⚡ Ultrafast laser-matter interactions</li><li>🔬 Characterization techniques (SEM, AFM, etc.)</li>
            <li>📐 Process parameter optimization</li><li>🔗 <strong>Multi-document property comparison</strong></li></ul>
            <p><strong>🎯 Enhanced Features:</strong></p>
            <ul><li>Citations display as "Smith et al., J. Appl. Phys., 2023" or DOI</li>
            <li>🔗 Cross-document property extraction & fusion</li>
            <li>📊 Fusion efficiency metrics per answer</li>
            <li>📋 Automatic comparison table generation</li>
            <li>🐛 Debug mode for extraction diagnostics</li></ul>
            <p><strong>Getting started:</strong></p>
            <ol><li>Upload PDF/TXT files in the left panel</li><li>Click "Process Documents"</li>
            <li>Select your preferred local LLM in sidebar</li><li>Enable "Multi-Document Fusion" for comparative queries</li>
            <li>Start asking technical questions!</li></ol></div>""", unsafe_allow_html=True)
            st.markdown("**Try asking:**")
            demo_qs = ["What factors affect ablation threshold in metals?", "How does pulse duration influence LIPSS periodicity?",
                      "Compare yield strength of AlSi10Mg under different heat treatments", "What is the typical fluence range for femtosecond laser processing?"]
            for q in demo_qs:
                if st.button(f"💬 {q}", use_container_width=True, key=f"demo_{q[:20]}"):
                    st.session_state.demo_question = q
                    st.rerun()
    render_footer()
    if hasattr(st.session_state, 'demo_question') and st.session_state.demo_question:
        st.session_state.messages.append({"role": "user", "content": st.session_state.demo_question})
        del st.session_state.demo_question
        st.rerun()

if __name__ == "__main__":
    main()
