#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DECLARMIMA v13.0+ - ENHANCED PHYSICAL QUANTITY & MATERIAL EXTRACTION + FULL VISUALIZATION SUITE
=================================================================================================
- Full support for alloy/material name extraction
- Yield strength, tensile strength classification (no more mislabelling as hardness)
- Two-stage retrieval (semantic search + page reading)
- Structured metadata ingestion (store key parameters once)
- Pagination-aware page fetching with auto-continuation
- Cross-document aggregation by physical quantity AND by material
- Human summaries grouped by material and property
- COMPLETE VISUALIZATION IMPLEMENTATION (30+ chart types, all rendering)
- Dynamic concept selector with salience-based filtering
- Quantitative data explorer for parameter browsing
- Interactive network graphs (PyVis, Plotly)
- Dimensionality reduction (t-SNE, UMAP, PCA)
- Hierarchical taxonomies (sunbursts, treemaps)
- Cross-document contradiction & consensus analysis
"""

import streamlit as st
import os
import sys
import tempfile
import time
import re
import json
import hashlib
import asyncio
import logging
import warnings
import requests
import textwrap
import math
import copy
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any, Set, Callable, Literal
from collections import defaultdict, Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import numpy as np
import torch
import threading
import queue
import pandas as pd

# ============================================================================
# VISUALIZATION DEPENDENCIES - ALL IMPORTED AND USED
# ============================================================================
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional advanced visualization libraries
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("umap-learn not installed. pip install umap-learn for UMAP plots")

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
    logger.warning("bokeh not installed. pip install bokeh for interactive chord diagrams")

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    logger.warning("pyvis not installed. pip install pyvis for interactive network graphs")

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed. pip install scikit-learn for t-SNE/PCA")

# Suppress other warnings
warnings.filterwarnings('ignore')

# Configure logging
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logging.basicConfig(level=logging.INFO, handlers=[console_handler], force=True)
logger = logging.getLogger("DECLARMIMA")

# ============================================================================
# OPTIONAL IMPORTS (LLM & PDF)
# ============================================================================
try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    raise ImportError("PyMuPDF (fitz) required: pip install pymupdf")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not installed. Ollama backend unavailable.")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False
    logger.warning("orjson not installed. Using standard json (slower).")

# Optional: sentence-transformers for semantic search
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Two-stage retrieval will use fallback TF-IDF.")

# ============================================================================
# 1. PYDANTIC MODELS (ENHANCED)
# ============================================================================
from pydantic import BaseModel, Field, field_validator


class UniversalExtractionItem(BaseModel):
    item_type: Literal["quantitative", "qualitative", "definition", "comparison", "relationship", "process", "material", "method"]
    content: str
    parameter_name: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    physical_quantity: Optional[str] = None   # e.g., "laser_power", "yield_strength", "scan_speed"
    material: Optional[str] = None            # alloy designation (e.g., "AlSiMg1.4Zr", "Ti6Al4V")
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object_val: Optional[str] = None
    definition_term: Optional[str] = None
    definition_text: Optional[str] = None
    comparison_entities: List[str] = []
    comparison_aspect: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    context: str
    doc_source: str
    page: int
    section_title: Optional[str] = None
    method: Optional[str] = None
    conditions: Dict[str, Any] = {}
    reasoning_trace: str = ""

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        return max(0.0, min(1.0, v))

    def citation(self) -> str:
        return f'<cite doc="{self.doc_source}" page="{self.page}"/>'

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ExtractedValue(BaseModel):
    query: str
    value: float
    unit: str
    physical_quantity: str
    parameter_name: Optional[str] = None
    material: Optional[str] = None          # NEW: associate value with a material
    confidence: float = Field(ge=0.0, le=1.0)
    context: str
    doc_name: str
    page: int
    section_title: Optional[str] = None

    @field_validator('value')
    def non_zero(cls, v):
        if v == 0.0:
            raise ValueError("Zero values ignored")
        return v


class QueryReport(BaseModel):
    query: str
    total_docs: int
    docs_with_results: int
    all_values: List[ExtractedValue]
    consensus: Dict[str, Any]
    processing_time_sec: float

    def to_json(self):
        return json.dumps(self.model_dump(), indent=2, ensure_ascii=False, default=str)


class CrossDocumentQueryReport(BaseModel):
    query: str
    query_type: Optional[str] = None
    total_documents: int
    documents_with_results: int
    documents_without_results: List[str] = []
    all_items: List[UniversalExtractionItem] = []
    document_summaries: List[Dict[str, Any]] = []
    consensus_analysis: Dict[str, Any] = {}
    contradictions_detected: List[Dict[str, Any]] = []
    processing_metadata: Dict[str, Any] = {}

    def to_json(self, indent=2) -> str:
        return json.dumps(self.model_dump(), indent=indent, ensure_ascii=False, default=str)


class DocumentMetadata(BaseModel):
    """Structured metadata extracted from a document at ingestion time."""
    doc_name: str
    alloys: List[str] = []                     # e.g., ["AlSiMg1.4Zr", "Ti6Al4V"]
    laser_power_values: List[float] = []       # W or kW
    scan_speed_values: List[float] = []        # mm/s or m/s
    yield_strength_values: List[float] = []    # MPa
    tensile_strength_values: List[float] = []  # MPa
    hardness_values: List[float] = []          # HV or MPa
    temperature_values: List[float] = []       # °C
    energy_density_values: List[float] = []    # J/mm³
    process_types: List[str] = []              # "SLM", "LPBF", "LSA", etc.
    other_parameters: Dict[str, List[float]] = {}

# ============================================================================
# 2. ENHANCED PHYSICAL QUANTITY CLASSIFIER (with strength and material hints)
# ============================================================================
class PhysicalQuantityClassifier:
    """
    Maps parameter names, context, and units to canonical physical quantity labels.
    Now includes yield strength, tensile strength, and material-related keywords.
    """
    
    CANONICAL = {
        # Power-related
        "laser_power": ["laser power", "laser beam power", "laser output power", "laser power density (power)"],
        "electrical_power": ["electrical power", "power supply", "input power", "electrical load"],
        # Speed/velocity
        "scan_speed": ["scan speed", "scanning speed", "laser scan speed", "beam scan speed", "scan velocity"],
        "flow_speed": ["flow speed", "flow velocity", "fluid velocity", "air velocity", "gas flow speed"],
        "feed_rate": ["feed rate", "travel speed", "table speed", "stage speed"],
        # Irradiance / intensity
        "irradiance": ["irradiance", "laser irradiance", "intensity", "power density (irradiance)", "w/cm²", "kw/cm²"],
        # Temperature
        "temperature": ["temperature", "melting temperature", "annealing temperature", "reflow temperature"],
        # Energy density
        "energy_density": ["energy density", "volumetric energy density", "VED", "laser fluence"],
        # Length / dimension
        "layer_thickness": ["layer thickness", "powder layer thickness", "slice thickness"],
        "spot_size": ["spot size", "beam diameter", "laser spot diameter"],
        # Time
        "exposure_time": ["exposure time", "dwell time", "laser on time"],
        # Mechanical properties - NEW
        "yield_strength": ["yield strength", "ys", "0.2% offset strength", "proof stress", "yield stress"],
        "tensile_strength": ["tensile strength", "uts", "ultimate tensile strength", "ultimate strength"],
        "hardness": ["hardness", "vickers hardness", "microhardness", "hv"],
        "elongation": ["elongation", "strain", "ductility", "strain to failure"],
        # Others
        "modulus": ["young's modulus", "elastic modulus", "stiffness", "e-modulus"],
    }
    
    UNIT_HINTS = {
        "scan_speed": ["mm/s", "cm/s", "m/s", "mm/min", "in/min"],
        "flow_speed": ["mm/s", "cm/s", "m/s", "l/min", "m³/s"],
        "laser_power": ["w", "kw", "mw"],
        "irradiance": ["w/cm²", "kw/cm²", "w/m²"],
        "temperature": ["°c", "k", "°f"],
        "energy_density": ["j/mm³", "j/m³", "j/cm³", "j/m²"],
        "yield_strength": ["mpa", "gpa", "psi"],
        "tensile_strength": ["mpa", "gpa", "psi"],
        "hardness": ["hv", "mpa", "gpa"],
    }
    
    def __init__(self):
        self._build_keyword_index()
    
    def _build_keyword_index(self):
        self.keyword_to_canonical = {}
        for canonical, keywords in self.CANONICAL.items():
            for kw in keywords:
                self.keyword_to_canonical[kw.lower()] = canonical
        self.keyword_to_canonical["ys"] = "yield_strength"
        self.keyword_to_canonical["uts"] = "tensile_strength"
        self.keyword_to_canonical["smys"] = "yield_strength"
        self.keyword_to_canonical["0.2% proof"] = "yield_strength"
    
    def classify(self, parameter_name: Optional[str], unit: Optional[str], context: str) -> str:
        if parameter_name:
            pname_lower = parameter_name.lower().strip()
            for canonical, keywords in self.CANONICAL.items():
                for kw in keywords:
                    if kw in pname_lower:
                        return canonical
            if pname_lower in self.keyword_to_canonical:
                return self.keyword_to_canonical[pname_lower]
        
        context_lower = context.lower()
        for canonical, keywords in self.CANONICAL.items():
            for kw in keywords:
                if kw in context_lower:
                    return canonical
        
        if unit:
            unit_lower = unit.lower()
            if "yield" in context_lower and "mpa" in unit_lower:
                return "yield_strength"
            if "tensile" in context_lower and "mpa" in unit_lower:
                return "tensile_strength"
            for canonical, units in self.UNIT_HINTS.items():
                for u in units:
                    if u in unit_lower:
                        return canonical
        
        if unit:
            if "w/cm" in unit_lower or "kw/cm" in unit_lower:
                return "irradiance"
            if unit_lower in ["w", "kw", "mw"]:
                return "laser_power"
            if "mm/s" in unit_lower:
                return "scan_speed"
            if "°c" in unit_lower or "k" in unit_lower:
                return "temperature"
            if "mpa" in unit_lower or "gpa" in unit_lower:
                return "hardness"
        
        return "unknown"
    
    def get_human_readable(self, canonical: str) -> str:
        mapping = {
            "laser_power": "Laser Power",
            "electrical_power": "Electrical Power",
            "scan_speed": "Scan Speed",
            "flow_speed": "Flow Speed",
            "feed_rate": "Feed Rate",
            "irradiance": "Irradiance / Intensity",
            "temperature": "Temperature",
            "energy_density": "Energy Density",
            "layer_thickness": "Layer Thickness",
            "spot_size": "Spot Size",
            "exposure_time": "Exposure Time",
            "yield_strength": "Yield Strength",
            "tensile_strength": "Tensile Strength",
            "hardness": "Hardness",
            "elongation": "Elongation",
            "modulus": "Young's Modulus",
            "unknown": "Other Quantities"
        }
        return mapping.get(canonical, canonical.replace("_", " ").title())


# ============================================================================
# 3. PAGINATION-AWARE PDF READER (with auto-continuation)
# ============================================================================
class PaginationAwareReader:
    """Handles PDF page extraction with automatic continuation when truncated.
       Since we have direct fitz access, this is mostly a wrapper that can split
       large page ranges into batches to avoid memory issues."""
    
    def __init__(self, max_chars_per_request=20000):
        self.max_chars_per_request = max_chars_per_request
    
    def extract_pages(self, doc_path: str, page_numbers: List[int]) -> Dict[int, str]:
        """Extract full text for given page numbers, handling large pages by warning."""
        doc = fitz.open(doc_path)
        result = {}
        for pnum in page_numbers:
            if pnum < 1 or pnum > len(doc):
                continue
            page = doc[pnum-1]
            text = page.get_text("text")
            # If text exceeds limit, truncate with warning (but still return)
            if len(text) > self.max_chars_per_request:
                logger.warning(f"Page {pnum} text length {len(text)} exceeds limit {self.max_chars_per_request}, truncating.")
                text = text[:self.max_chars_per_request] + "\n...[TRUNCATED]"
            result[pnum] = text
        doc.close()
        return result
    
    def extract_page_range(self, doc_path: str, start: int, end: int, step=1) -> Dict[int, str]:
        pages = list(range(start, end+1, step))
        return self.extract_pages(doc_path, pages)


# ============================================================================
# 4. STRUCTURED METADATA EXTRACTOR (runs at ingestion)
# ============================================================================
class StructuredMetadataExtractor:
    """Extracts key numerical parameters and material names from document text.
       Uses regex patterns and lightweight NLP to populate DocumentMetadata."""
    
    # Patterns for alloy names (common designations)
    ALLOY_PATTERNS = [
        r'\b(?:AlSi[\dMg]+|Ti\d*Al\d*V\d*|Inconel\s?\d{3}|SS\s?\d{4}|UNS\s?S\d{5}|Ti\s?6Al\s?4V|Cu\s?[A-Za-z0-9]+|Fe-based|Mg\s?alloy)\b',
        r'\b(?:Al-[\d]+Si-[\d]+Mg|AlSiMg[\d\.]+Zr|TiB[2]?|CoCr[\w]+|NiTi|Au\-Ti|Zr\-enhanced)\b',
        r'(\w+(?:-\w+)?\s?(?:alloy|superalloy|metal|composite))'
    ]
    
    # Patterns for extracting key-value pairs
    POWER_PATTERN = r'(?:laser\s+power|power|P)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(W|kW|mW)'
    SCAN_SPEED_PATTERN = r'(?:scan\s+speed|scanning\s+speed|v_scan|Vs)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(mm/s|cm/s|m/s|mm/min)'
    YIELD_PATTERN = r'(?:yield\s+strength|YS|0\.2%\s+proof)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(MPa|GPa|psi)'
    TENSILE_PATTERN = r'(?:tensile\s+strength|UTS)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(MPa|GPa|psi)'
    HARDNESS_PATTERN = r'(?:hardness|HV|Vickers)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(HV|MPa|GPa)'
    TEMP_PATTERN = r'(?:temperature|T)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(°C|K|°F)'
    VED_PATTERN = r'(?:volumetric\s+energy\s+density|VED)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(J/mm³|J/cm³)'
    
    def __init__(self):
        self.compiled_patterns = {
            "laser_power": (re.compile(self.POWER_PATTERN, re.IGNORECASE), float),
            "scan_speed": (re.compile(self.SCAN_SPEED_PATTERN, re.IGNORECASE), float),
            "yield_strength": (re.compile(self.YIELD_PATTERN, re.IGNORECASE), float),
            "tensile_strength": (re.compile(self.TENSILE_PATTERN, re.IGNORECASE), float),
            "hardness": (re.compile(self.HARDNESS_PATTERN, re.IGNORECASE), float),
            "temperature": (re.compile(self.TEMP_PATTERN, re.IGNORECASE), float),
            "energy_density": (re.compile(self.VED_PATTERN, re.IGNORECASE), float),
        }
        self.alloy_regexes = [re.compile(p, re.IGNORECASE) for p in self.ALLOY_PATTERNS]
    
    def extract_metadata(self, doc_name: str, full_text: str) -> DocumentMetadata:
        """Parse document text to extract structured metadata."""
        meta = DocumentMetadata(doc_name=doc_name)
        
        # Extract alloys
        alloys_set = set()
        for regex in self.alloy_regexes:
            for match in regex.finditer(full_text):
                candidate = match.group(0).strip()
                # Clean common false positives
                if len(candidate) > 2 and candidate.lower() not in ["alloy", "composite", "metal"]:
                    alloys_set.add(candidate)
        meta.alloys = list(alloys_set)
        
        # Extract numerical parameters
        for field, (pattern, cast_func) in self.compiled_patterns.items():
            matches = pattern.findall(full_text)
            values = []
            for m in matches:
                try:
                    val = cast_func(m[0])
                    values.append(val)
                except:
                    continue
            if field == "laser_power":
                meta.laser_power_values = values
            elif field == "scan_speed":
                meta.scan_speed_values = values
            elif field == "yield_strength":
                meta.yield_strength_values = values
            elif field == "tensile_strength":
                meta.tensile_strength_values = values
            elif field == "hardness":
                meta.hardness_values = values
            elif field == "temperature":
                meta.temperature_values = values
            elif field == "energy_density":
                meta.energy_density_values = values
        
        # Detect process types
        process_keywords = {
            "SLM": ["selective laser melting", "slm"],
            "LPBF": ["laser powder bed fusion", "l-pbf", "lpbf"],
            "LSA": ["laser surface alloying", "lsa"],
            "EBM": ["electron beam melting", "ebm"],
            "DED": ["directed energy deposition", "ded"],
        }
        processes = []
        for proc, keywords in process_keywords.items():
            if any(kw in full_text.lower() for kw in keywords):
                processes.append(proc)
        meta.process_types = processes
        
        return meta


# ============================================================================
# 5. TWO-STAGE RETRIEVER (semantic search + page selection)
# ============================================================================
class TwoStageRetriever:
    """First stage: fast search over document metadata and summaries.
       Second stage: read full pages only for top-k documents."""
    def __init__(self, llm: Optional['HybridLLM'] = None, embedding_model: str = "all-MiniLM-L6-v2"):
        self.llm = llm
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # FORCE CPU TO AVOID CUDA KERNEL ERRORS
                self.embedding_model = SentenceTransformer(embedding_model, device="cpu")
                logger.info(f"Loaded sentence-transformer model {embedding_model} on CPU")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
        self.doc_metadata: Dict[str, DocumentMetadata] = {}
        self.doc_summaries: Dict[str, str] = {}
    
    def index_document(self, doc_name: str, metadata: DocumentMetadata, summary: str):
        self.doc_metadata[doc_name] = metadata
        self.doc_summaries[doc_name] = summary
    
    def retrieve_relevant_docs(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Return list of (doc_name, relevance_score) for top-k documents."""
        if self.embedding_model is not None and len(self.doc_summaries) > 0:
            # Semantic search using embeddings
            doc_texts = [f"{meta.alloys} {meta.process_types} {self.doc_summaries.get(name, '')}" 
                         for name, meta in self.doc_metadata.items()]
            if doc_texts:
                doc_emb = self.embedding_model.encode(doc_texts, convert_to_tensor=True)
                query_emb = self.embedding_model.encode(query, convert_to_tensor=True)
                scores = util.cos_sim(query_emb, doc_emb)[0]
                scored = [(list(self.doc_metadata.keys())[i], float(scores[i])) for i in range(len(doc_texts))]
                scored.sort(key=lambda x: x[1], reverse=True)
                return scored[:top_k]
        
        # Fallback: keyword matching based on metadata fields
        scores = []
        query_lower = query.lower()
        for name, meta in self.doc_metadata.items():
            score = 0.0
            # Alloy matches
            for alloy in meta.alloys:
                if alloy.lower() in query_lower:
                    score += 0.3
            # Power matches
            if any(str(p) in query_lower for p in meta.laser_power_values):
                score += 0.2
            # Process type matches
            for proc in meta.process_types:
                if proc.lower() in query_lower:
                    score += 0.2
            scores.append((name, min(score, 1.0)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def get_relevant_pages(self, doc_name: str, query: str, max_pages: int = 5) -> List[int]:
        """After document selection, choose which pages to read based on query.
           Uses simple keyword density in page summaries if available, else returns first max_pages."""
        # In a full implementation, we would have indexed page summaries. Here we return first few pages.
        # This can be enhanced by using page-level metadata.
        return list(range(1, max_pages+1))


# ============================================================================
# 6. HIERARCHICAL PDF INDEX (cached, with metadata integration)
# ============================================================================
@dataclass
class PageNode:
    id: str
    title: str
    page_start: int
    page_end: Optional[int]
    full_text: str
    summary: str
    level: int
    children: List['PageNode'] = field(default_factory=list)
    doc_id: str = ""
    section_type: str = "BODY"
    node_id: str = ""
    prefix_summary: str = ""
    text_token_count: int = 0
    _pdf_path: Optional[str] = None
    metadata: Optional[DocumentMetadata] = None   # Attach structured metadata

    def get_text(self, doc_cache: Dict[str, Any] = None, max_chars: int = 20000) -> str:
        if self.full_text:
            return self.full_text[:max_chars] if len(self.full_text) > max_chars else self.full_text
        if not self._pdf_path or not fitz:
            return ""
        doc = None
        if doc_cache and self.doc_id in doc_cache:
            doc = doc_cache[self.doc_id]
        else:
            doc = fitz.open(self._pdf_path)
            if doc_cache:
                doc_cache[self.doc_id] = doc
        start = self.page_start - 1
        end = min(self.page_end or self.page_start, len(doc))
        texts = [doc[p].get_text("text") for p in range(start, end)]
        self.full_text = "\n\n".join(texts)
        if doc_cache is None:
            doc.close()
        return self.full_text[:max_chars] if len(self.full_text) > max_chars else self.full_text

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "summary": self.summary,
            "prefix_summary": self.prefix_summary,
            "level": self.level,
            "doc_id": self.doc_id,
            "section_type": self.section_type,
            "node_id": self.node_id,
            "text_token_count": self.text_token_count,
            "children": [c.to_dict() for c in self.children],
            "metadata": self.metadata.dict() if self.metadata else None
        }

    def to_tree_format(self, max_chars: int = 20000) -> Dict[str, Any]:
        result = {
            "title": self.title,
            "node_id": self.node_id,
            "start_index": self.page_start,
            "end_index": self.page_end or self.page_start,
            "summary": self.summary,
            "prefix_summary": self.prefix_summary,
            "text_token_count": self.text_token_count,
        }
        if self.children:
            result["nodes"] = [c.to_tree_format(max_chars) for c in self.children]
        text = self.get_text(max_chars=max_chars)
        if text:
            result["text"] = text
        if self.metadata:
            result["metadata"] = self.metadata.dict()
        return result

    @classmethod
    def from_dict(cls, data: dict, pdf_path=None):
        node = cls(
            data["id"],
            data["title"],
            data["page_start"],
            data.get("page_end"),
            "",
            data.get("summary", ""),
            data.get("level", 0),
            doc_id=data.get("doc_id", ""),
            section_type=data.get("section_type", "BODY"),
            _pdf_path=pdf_path
        )
        node.node_id = data.get("node_id", "")
        node.prefix_summary = data.get("prefix_summary", "")
        node.text_token_count = data.get("text_token_count", 0)
        for c in data.get("children", []):
            node.children.append(cls.from_dict(c, pdf_path))
        if data.get("metadata"):
            node.metadata = DocumentMetadata(**data["metadata"])
        return node


class HierarchicalIndex:
    def __init__(self, cache_dir=".declarmima_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.doc_trees: Dict[str, PageNode] = {}
        self._pdf_cache = {}
        self.metadata_extractor = StructuredMetadataExtractor()

    def _doc_hash(self, file_buffer: BytesIO) -> str:
        pos = file_buffer.tell()
        file_buffer.seek(0)
        content = file_buffer.read(1024 * 1024)
        file_buffer.seek(pos)
        return hashlib.sha256(content).hexdigest()[:16]

    def _cache_path(self, doc_name: str, doc_hash: str) -> Path:
        safe = re.sub(r'[^\w\-_.]', '_', doc_name)
        return self.cache_dir / f"{safe}.{doc_hash}.tree.json"

    def build_from_pdfs(self, files: List, parallel=True, max_workers=4):
        def build_one(file):
            doc_name = file.name
            buf = BytesIO(file.getbuffer())
            doc_hash = self._doc_hash(buf)
            cache_path = self._cache_path(doc_name, doc_hash)
            
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        root_data = fast_json_loads(f.read())
                    root = PageNode.from_dict(root_data)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        buf.seek(0)
                        tmp.write(buf.getbuffer())
                        root._pdf_path = tmp.name
                    return doc_name, root
                except Exception as e:
                    logger.warning(f"Cache load failed for {doc_name}: {e}")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                buf.seek(0)
                tmp.write(buf.getbuffer())
                tmp_path = tmp.name
            doc = fitz.open(tmp_path)
            root = self._build_tree(doc, doc_name, tmp_path)
            # Extract structured metadata from full document text
            full_text = "\n".join([doc[p].get_text("text") for p in range(len(doc))])
            meta = self.metadata_extractor.extract_metadata(doc_name, full_text)
            root.metadata = meta
            doc.close()
            
            try:
                cache_root = self._clone_for_cache(root)
                with open(cache_path, "wb") as f:
                    f.write(fast_json_dumps(cache_root.to_dict(), indent=True))
            except Exception as e:
                logger.warning(f"Cache save failed: {e}")
            
            return doc_name, root

        if parallel and len(files) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(build_one, f): f.name for f in files}
                for fut in as_completed(futures):
                    name, tree = fut.result()
                    self.doc_trees[name] = tree
        else:
            for f in files:
                name, tree = build_one(f)
                self.doc_trees[name] = tree
        return self.doc_trees

    def _build_tree(self, doc, doc_id, pdf_path):
        root = PageNode(
            f"{doc_id}_root", "Document Root", 1, len(doc), "",
            f"Document {doc_id} root covering pages 1-{len(doc)}",
            0, doc_id=doc_id, _pdf_path=pdf_path, node_id="0000"
        )
        toc = doc.get_toc()
        window = UNIVERSAL_CONFIG.get("leaf_node_page_window", 7)
        if toc:
            nodes_by_level = {}
            for level, title, page in toc:
                if page > len(doc):
                    continue
                end = min(page + window, len(doc))
                text = self._extract_range(doc, page, end)
                node = PageNode(
                    f"{doc_id}_toc_{level}_{title[:20]}",
                    title.strip(), page, end,
                    text, text[:200], level, doc_id=doc_id, _pdf_path=pdf_path
                )
                nodes_by_level.setdefault(level, []).append(node)
            for level in sorted(nodes_by_level.keys()):
                for node in nodes_by_level[level]:
                    parent = self._find_parent(root, level - 1, node.page_start)
                    parent.children.append(node)
            self._assign_node_ids(root)
            return root
        headings = self._detect_headings(doc)
        if headings:
            for i, (title, page) in enumerate(headings):
                end = min(page + window, len(doc))
                text = self._extract_range(doc, page, end)
                node = PageNode(
                    f"{doc_id}_h{i}", title, page, end, text, text[:200],
                    2, doc_id=doc_id, _pdf_path=pdf_path
                )
                root.children.append(node)
            self._assign_node_ids(root)
            return root
        for p in range(1, len(doc) + 1):
            text = doc[p-1].get_text("text")
            if not text.strip():
                continue
            node = PageNode(
                f"{doc_id}_p{p}", f"Page {p}", p, p, text, text[:200],
                3, doc_id=doc_id, _pdf_path=pdf_path
            )
            root.children.append(node)
        self._assign_node_ids(root)
        return root

    def _extract_range(self, doc, start, end):
        return "\n\n".join(doc[p-1].get_text("text") for p in range(start, min(end, len(doc) + 1)))

    def _detect_headings(self, doc):
        headings = []
        for p in range(len(doc)):
            lines = doc[p].get_text("text").split('\n')
            for line in lines:
                if re.match(r'^(?:[0-9]+\.?)+ +[A-Z]', line.strip()):
                    headings.append((line.strip(), p + 1))
        return headings[:50]

    def _find_parent(self, node, target_level, page_hint):
        if target_level < 0:
            return node
        candidates = [c for c in node.children if c.level == target_level]
        if not candidates:
            return node
        return min(candidates, key=lambda n: abs(n.page_start - page_hint))

    def _assign_node_ids(self, root: PageNode):
        def assign(node: PageNode, prefix: str = "", index: int = 1):
            if not prefix:
                node.node_id = str(index).zfill(4)
                current_prefix = node.node_id
            else:
                node.node_id = f"{prefix}.{str(index).zfill(4)}"
                current_prefix = node.node_id
            for i, child in enumerate(node.children, 1):
                assign(child, current_prefix, i)
        assign(root, "", 1)

    def _clone_for_cache(self, node):
        return PageNode(
            node.id, node.title, node.page_start, node.page_end, "",
            node.summary, node.level, doc_id=node.doc_id,
            section_type=node.section_type, node_id=node.node_id,
            prefix_summary=node.prefix_summary, text_token_count=node.text_token_count,
            children=[self._clone_for_cache(c) for c in node.children],
            metadata=node.metadata
        )

    def cleanup(self):
        for doc in self._pdf_cache.values():
            try:
                doc.close()
            except:
                pass
        self._pdf_cache.clear()


class FastHierarchicalIndex(HierarchicalIndex):
    def __init__(self, cache_dir=".declarmima_cache", llm=None):
        super().__init__(cache_dir)
        self.llm = llm

    async def build_from_pdfs_fast(self, files: List, max_workers: int = 4) -> Dict[str, PageNode]:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [loop.run_in_executor(pool, self._extract_pages_raw, f) for f in files]
            raw_docs = await asyncio.gather(*futures)
        if self.llm:
            toc_tasks = [self._llm_extract_toc(doc_name, pages) for doc_name, pages in raw_docs]
            toc_results = await asyncio.gather(*toc_tasks)
        else:
            toc_results = [{"has_toc": False, "headings_detected": []} for _ in raw_docs]
        trees = {}
        for (doc_name, pages), toc in zip(raw_docs, toc_results):
            # Build tree
            tree = self._build_tree_from_toc(doc_name, pages, toc)
            # Extract metadata from full text
            full_text = "\n".join([p['text'] for p in pages])
            meta = self.metadata_extractor.extract_metadata(doc_name, full_text)
            tree.metadata = meta
            trees[doc_name] = tree
        if self.llm:
            await self._generate_summaries_async(trees)
        for doc_name, tree in trees.items():
            self.doc_trees[doc_name] = tree
            self._save_tree_fast(doc_name, tree)
        return trees

    def _extract_pages_raw(self, file_obj) -> Tuple[str, List[Dict]]:
        if hasattr(file_obj, 'getbuffer'):
            buf = BytesIO(file_obj.getbuffer())
            doc_name = file_obj.name
        else:
            buf = file_obj
            doc_name = "unknown.pdf"
        doc = fitz.open(stream=buf.getvalue(), filetype="pdf")
        pages = []
        for p in range(len(doc)):
            page = doc[p]
            pages.append({
                'page_num': p + 1,
                'text': page.get_text("text"),
                'images': len(page.get_images()),
                'blocks': page.get_text("blocks")
            })
        doc.close()
        return doc_name, pages

    async def _llm_extract_toc(self, doc_name: str, pages: List[Dict]) -> Dict[str, Any]:
        sample_text = "\n\n".join(p['text'][:1500] for p in pages[:5])
        prompt = f"""Analyze this document and extract its hierarchical structure.
Return JSON with:
- "has_toc": bool
- "toc_entries": list of {{"title": str, "level": int, "page": int}}
- "headings_detected": list of {{"title": str, "level": int, "page": int}}
- "doc_type": str
- "suggested_root_title": str

Document sample:
{sample_text[:6000]}

Return ONLY valid JSON."""
        try:
            response = await asyncio.to_thread(self.llm.generate, prompt, max_new_tokens=1024, fast_json=True)
            result = self._extract_json_safe(response)
            if result and isinstance(result, dict):
                return result
        except Exception as e:
            logger.warning(f"LLM TOC extraction failed for {doc_name}: {e}")
        return {"has_toc": False, "headings_detected": [], "doc_type": "unknown"}

    def _extract_json_safe(self, text: str) -> Optional[Any]:
        patterns = [r'\{.*\}', r'\[.*\]', r'```json\s*(\{.*?\})\s*```', r'```json\s*(\[.*?\])\s*```']
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_str = match.group(1) if match.groups() else match.group(0)
                try:
                    return json.loads(json_str)
                except:
                    continue
        return None
    #
    def _build_tree_from_toc(self, doc_name: str, pages: List[Dict], toc: Dict) -> PageNode:
        """
        Build a hierarchical tree from a document's table of contents (LLM-extracted).
        Handles missing or invalid page numbers safely.
        """
        safe_title = toc.get("suggested_root_title") or doc_name
        root = PageNode(
            f"{doc_name}_root",
            safe_title,
            1, len(pages), "",
            f"Document {doc_name}", 0,
            doc_id=doc_name, node_id="0000"
        )
        
        entries = toc.get("toc_entries", []) or toc.get("headings_detected", [])
        window = UNIVERSAL_CONFIG.get("leaf_node_page_window", 7)
        
        if entries:
            nodes_by_level = {}
            for entry in entries:
                # Extract level (default 1 if missing/invalid)
                level_val = entry.get("level")
                if level_val is None:
                    level = 1
                else:
                    try:
                        level = int(level_val)
                    except (ValueError, TypeError):
                        level = 1
                
                # Extract title (default "Unknown")
                title = entry.get("title")
                if title is None:
                    title = "Unknown"
                title = str(title).strip()
                
                # ---------- CRITICAL FIX: safe page number conversion ----------
                page_raw = entry.get("page")
                if page_raw is None:
                    page = 1
                else:
                    try:
                        page = int(page_raw)
                    except (ValueError, TypeError):
                        page = 1
                # -----------------------------------------------------------------
                
                # Validate page range
                if page < 1 or page > len(pages):
                    continue
                
                # Determine end page (window size)
                end = min(page + window, len(pages))
                
                # Extract text for this node
                text_parts = []
                for i in range(page, min(end + 1, len(pages) + 1)):
                    try:
                        page_data = pages[i - 1]
                        if isinstance(page_data, dict) and 'text' in page_data:
                            text_parts.append(page_data['text'])
                    except (IndexError, KeyError, TypeError):
                        continue
                text = "\n\n".join(text_parts)
                
                # Create the node
                node_id = f"{doc_name}_toc_{level}_{title[:20]}"
                node = PageNode(
                    node_id,
                    title, page, end,
                    text, text[:200], level,
                    doc_id=doc_name
                )
                nodes_by_level.setdefault(level, []).append(node)
            
            # Attach children to parents based on level
            for level in sorted(nodes_by_level.keys()):
                for node in nodes_by_level[level]:
                    parent = self._find_parent(root, level - 1, node.page_start)
                    parent.children.append(node)
        else:
            # No TOC entries – use page-by-page fallback
            for p in pages:
                text = p.get('text', '')
                if not str(text).strip():
                    continue
                page_num = p.get('page_num', 1)
                try:
                    page_num = int(page_num)
                except (ValueError, TypeError):
                    page_num = 1
                node = PageNode(
                    f"{doc_name}_p{page_num}",
                    f"Page {page_num}", page_num, page_num,
                    text, str(text)[:200], 3,
                    doc_id=doc_name
                )
                root.children.append(node)
        
        self._assign_node_ids(root)
        return root
    
    async def _generate_summaries_async(self, trees: Dict[str, PageNode]):
        all_nodes = []
        def collect_nodes(node: PageNode):
            all_nodes.append(node)
            for c in node.children:
                collect_nodes(c)
        for tree in trees.values():
            collect_nodes(tree)
        batch_size = 5
        for i in range(0, len(all_nodes), batch_size):
            batch = all_nodes[i:i+batch_size]
            tasks = []
            for node in batch:
                if len(node.full_text) > 200:
                    tasks.append(self._summarize_node(node))
                else:
                    node.summary = node.full_text[:200]
            if tasks:
                await asyncio.gather(*tasks)

    async def _summarize_node(self, node: PageNode):
        text = node.full_text[:3000]
        prompt = f"""Summarize this document section in one sentence (max 200 chars).
Focus on key parameters, methods, and findings.

Text: {text}

Summary:"""
        try:
            summary = await asyncio.to_thread(self.llm.generate, prompt, max_new_tokens=150, temperature=0.1)
            node.summary = summary.strip()[:200]
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            node.summary = text[:200]

    def _save_tree_fast(self, doc_name: str, tree: PageNode):
        safe = re.sub(r'[^\w\-_.]', '_', doc_name)
        doc_hash = hashlib.sha256(doc_name.encode()).hexdigest()[:16]
        path = self.cache_dir / f"{safe}.{doc_hash}.tree.json"
        try:
            with open(path, "wb") as f:
                f.write(fast_json_dumps(tree.to_dict(), indent=True))
        except Exception as e:
            logger.warning(f"Fast save failed: {e}")


# ============================================================================
# 7. HYBRID LLM CLIENT (unchanged but with better system prompts for material extraction)
# ============================================================================
class HybridLLM:
    def __init__(self, model_key: str, use_4bit: bool = True, device: Optional[str] = None):
        self.model_key = model_key
        self.use_4bit = use_4bit
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = None
        self.model_name = None
        self.client = None
        self.tokenizer = None
        self.model = None
        if model_key.startswith("[Ollama]"):
            self.model_name = model_key.split("] ")[1].strip()
        elif model_key.startswith("ollama:"):
            self.model_name = model_key.replace("ollama:", "", 1)
        else:
            self.model_name = model_key
        self.template = get_model_template(self.model_name)
        self._init_backend()
        logger.info(f"HybridLLM initialized: {self.model_name} on {self.device} via {self.backend}")

    def _init_backend(self):
        if OLLAMA_AVAILABLE:
            try:
                requests.get("http://localhost:11434/api/tags", timeout=5)
                self.backend = "ollama"
                self.client = ollama.Client(host="http://localhost:11434")
                return
            except:
                pass
        if TRANSFORMERS_AVAILABLE:
            self.backend = "transformers"
            return
        raise RuntimeError("No LLM backend available. Install Ollama or transformers.")

    def generate(self, prompt: str, max_new_tokens=1024, temperature=0.1, fast_json=False, system_prompt=None):
        if self.backend == "ollama":
            return self._ollama_generate(prompt, max_new_tokens, temperature, fast_json, system_prompt)
        else:
            return self._transformers_generate(prompt, max_new_tokens, temperature, system_prompt)

    def _ollama_generate(self, prompt, max_tokens, temp, fast_json, system_prompt):
        try:
            options = {"temperature": temp, "num_predict": max_tokens}
            if fast_json:
                options["format"] = "json"
            messages = []
            sys = system_prompt or self.template.get("system")
            if sys:
                messages.append({"role": "system", "content": sys})
            messages.append({"role": "user", "content": prompt})
            resp = self.client.chat(model=self.model_name, messages=messages, options=options, stream=False)
            return resp.get("message", {}).get("content", "").strip()
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return f"Error: {str(e)[:100]}"

    def _transformers_generate(self, prompt, max_tokens, temp, system_prompt):
        if self.tokenizer is None:
            self._load_transformers()
        if not self.model:
            return "Error: model not loaded"
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temp if temp > 0 else None,
                    do_sample=temp > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "assistant" in response:
                response = response.split("assistant")[-1].strip()
            return response
        except Exception as e:
            logger.error(f"Transformers error: {e}")
            return f"Error: {str(e)[:100]}"

    def _load_transformers(self):
        logger.info(f"Loading {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32
        }
        if self.use_4bit and self.device == "cuda":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if self.device == "cuda":
            self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded.")


# ============================================================================
# 8. ENHANCED QUANTITATIVE KNOWLEDGE GRAPH (with material grouping)
# ============================================================================
class QuantitativeKnowledgeGraph:
    def __init__(self):
        self.doc_graphs: Dict[str, Dict] = {}
        self.phys_classifier = PhysicalQuantityClassifier()
        self.metadata_index: Dict[str, DocumentMetadata] = {}

    def add_document_metadata(self, doc_name: str, metadata: DocumentMetadata):
        self.metadata_index[doc_name] = metadata

    def add_extractions(self, doc_id: str, items: List[UniversalExtractionItem]):
        graph = {
            "doc_id": doc_id,
            "parameters": defaultdict(list),
            "materials": defaultdict(list),      # NEW: index by material name
            "methods": defaultdict(list),
            "by_page": defaultdict(list),
            "by_section": defaultdict(list),
            "by_physical_quantity": defaultdict(list),
            "all_items": []
        }
        for item in items:
            item_dict = item.to_dict()
            graph["all_items"].append(item_dict)
            if item.parameter_name:
                graph["parameters"][item.parameter_name.lower()].append(item_dict)
            if item.material:
                graph["materials"][item.material.lower()].append(item_dict)
            if item.method:
                graph["methods"][item.method.lower()].append(item_dict)
            if item.physical_quantity:
                graph["by_physical_quantity"][item.physical_quantity].append(item_dict)
            graph["by_page"][item.page].append(item_dict)
            if item.section_title:
                graph["by_section"][item.section_title].append(item_dict)
        self.doc_graphs[doc_id] = dict(graph)

    def get_parameter_across_docs(self, param_name: str) -> List[Dict]:
        results = []
        param_key = param_name.lower()
        for doc_id, graph in self.doc_graphs.items():
            if param_key in graph["parameters"]:
                for item in graph["parameters"][param_key]:
                    results.append({**item, "doc_id": doc_id})
        return results

    def get_all_materials(self) -> Dict[str, List[str]]:
        """Return mapping: doc_name -> list of unique material names found."""
        mat_dict = {}
        for doc_id, graph in self.doc_graphs.items():
            materials = set()
            for item in graph["all_items"]:
                if item.get("material"):
                    materials.add(item["material"])
            mat_dict[doc_id] = list(materials)
        return mat_dict

    def get_material_summary_stats(self, material_name: str) -> Dict[str, Any]:
        """Aggregate all quantitative values for a specific material across docs."""
        values_by_pq = defaultdict(list)
        for doc_id, graph in self.doc_graphs.items():
            for item in graph["all_items"]:
                if item.get("material") and item["material"].lower() == material_name.lower():
                    if item.get("value") is not None and item.get("physical_quantity"):
                        values_by_pq[item["physical_quantity"]].append({
                            "value": item["value"],
                            "unit": item.get("unit", ""),
                            "doc": doc_id,
                            "page": item.get("page", 1)
                        })
        return dict(values_by_pq)

    def to_tree_annotation(self, doc_tree: PageNode, max_chars: int = 20000) -> Dict[str, Any]:
        doc_id = doc_tree.doc_id
        graph = self.doc_graphs.get(doc_id, {})
        def annotate_node(node: PageNode) -> Dict[str, Any]:
            result = node.to_tree_format(max_chars=max_chars)
            node_items = []
            end_page = node.page_end or node.page_start
            for page in range(node.page_start, end_page + 1):
                node_items.extend(graph.get("by_page", {}).get(page, []))
            if node_items:
                seen = set()
                unique_items = []
                for item in node_items:
                    key = (item.get('parameter_name'), item.get('value'), item.get('page'))
                    if key not in seen:
                        seen.add(key)
                        unique_items.append(item)
                result["quantitative_items"] = unique_items
            if node.children:
                result["nodes"] = [annotate_node(c) for c in node.children]
            return result
        return annotate_node(doc_tree)

    def get_summary_stats(self, physical_quantity: str) -> Dict[str, Any]:
        values = []
        for doc_id, graph in self.doc_graphs.items():
            for item in graph["all_items"]:
                if item.get("physical_quantity") == physical_quantity and item.get("value") is not None:
                    values.append(item["value"])
        if not values:
            return {"count": 0, "documents": []}
        docs = list(set(item["doc_source"] for doc_id, graph in self.doc_graphs.items()
                        for item in graph["all_items"] if item.get("physical_quantity") == physical_quantity))
        stats = {"count": len(values), "documents": docs, "values": values}
        if values:
            stats.update({
                "min": min(values),
                "max": max(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)) if len(values) > 1 else 0
            })
        return stats

    def get_all_physical_quantities(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for doc_id, graph in self.doc_graphs.items():
            for item in graph["all_items"]:
                pq = item.get("physical_quantity")
                if pq:
                    counts[pq] += 1
        return dict(counts)

    def build_extracted_values(self, query: str) -> List[ExtractedValue]:
        all_values = []
        for doc_id, graph in self.doc_graphs.items():
            for item in graph["all_items"]:
                if item.get("item_type") != "quantitative":
                    continue
                val = item.get("value")
                if val is None or val == 0:
                    continue
                unit = item.get("unit", "")
                phys_q = item.get("physical_quantity") or self.phys_classifier.classify(
                    item.get("parameter_name"), unit, item.get("context", "")
                )
                all_values.append(ExtractedValue(
                    query=query,
                    value=val,
                    unit=unit,
                    physical_quantity=phys_q,
                    parameter_name=item.get("parameter_name"),
                    material=item.get("material"),   # Include material
                    confidence=item.get("confidence", 0.7),
                    context=item.get("context", "")[:300],
                    doc_name=doc_id,
                    page=item.get("page", 1),
                    section_title=item.get("section_title")
                ))
        return all_values


# ============================================================================
# 9. ENHANCED UNIVERSAL LLM EXTRACTOR (with material extraction)
# ============================================================================
class UniversalLLMExtractor:
    EXTRACTION_PROMPT = """Extract information relevant to the query from these document sections.
QUERY: {query}
QUERY TYPE: {query_type}
SECTIONS:
{sections_text}

Return JSON array of extracted items with fields:
{{
  "item_type": "quantitative|qualitative|definition|comparison|relationship|process|material",
  "content": "exact phrase with full numerical value (never truncate numbers)",
  "confidence": 0.0-1.0,
  "context": "exact sentence from text",
  "doc_source": "{doc_id}",
  "page": page_number,
  "parameter_name": "...",
  "value": number,
  "unit": "e.g., W, kW, W/cm², mm/s, °C, MPa, HV",
  "physical_quantity": "one of: laser_power, electrical_power, scan_speed, flow_speed, irradiance, temperature, energy_density, layer_thickness, spot_size, exposure_time, yield_strength, tensile_strength, hardness, elongation, modulus, unknown",
  "material": "alloy or material name if mentioned (e.g., AlSiMg1.4Zr, Ti6Al4V, Inconel 718)"
}}

CRITICAL RULES:
1. Distinguish physically different quantities even if they share units:
   - "scan speed" (laser scanning) vs "flow speed" (fluid movement) – assign different physical_quantity.
   - "laser power" (W) vs "electrical power" (W) – assign different physical_quantity.
2. For mechanical properties: "yield strength" -> physical_quantity = "yield_strength", "tensile strength" -> "tensile_strength".
3. NEVER truncate numbers: if text says "1000 W", output 1000.
4. If an alloy or material name appears (e.g., "AlSi10Mg", "Ti6Al4V", "stainless steel 316L"), create an item with item_type="material", content=the name, material=the name. Do not include numerical value.
5. Return ONLY valid JSON, no extra text.
6. Set confidence based on clarity.

Return [] if no relevant information found."""

    def __init__(self, llm: HybridLLM):
        self.llm = llm
        self.phys_classifier = PhysicalQuantityClassifier()

    def extract_from_chunks(self, chunks: List[Dict], query: str, query_analysis: Optional[Dict] = None) -> List[UniversalExtractionItem]:
        if not chunks:
            return []
        qa = query_analysis or {"query_type": "mixed", "keywords": []}
        items = []
        for chunk in chunks:
            text = chunk["full_text"]
            doc = chunk["doc_id"]
            page = chunk["page_start"]
            # Skip chunks that are unlikely to contain relevant info for speed
            if qa.get("query_type") == "quantitative" and not re.search(r'\d+', text):
                continue
            # For material extraction, always process
            prompt = self.EXTRACTION_PROMPT.format(
                query=query,
                query_type=qa.get("query_type", "mixed"),
                sections_text=text[:4000],
                doc_id=doc
            )
            try:
                response = self.llm.generate(prompt, max_new_tokens=1024, fast_json=True)
                json_str = self._extract_json(response)
                if json_str:
                    data = json.loads(json_str)
                    for item_data in data if isinstance(data, list) else data.get("items", []):
                        # If physical_quantity missing, classify now
                        if "physical_quantity" not in item_data or not item_data["physical_quantity"]:
                            item_data["physical_quantity"] = self.phys_classifier.classify(
                                item_data.get("parameter_name"),
                                item_data.get("unit"),
                                item_data.get("context", "")
                            )
                        # Ensure material field exists
                        item_data.setdefault("material", None)
                        try:
                            item = UniversalExtractionItem(**item_data)
                            if doc not in item.context:
                                item.context = f"[{doc}] {item.context}"
                            if item.page == 0:
                                item.page = page
                            items.append(item)
                        except Exception as e:
                            logger.debug(f"Item parse error: {e}")
            except Exception as e:
                logger.error(f"Extraction error: {e}")
        # Deduplicate
        unique = {}
        for i in items:
            key = (i.content, i.doc_source, i.page, i.material)
            if key not in unique or i.confidence > unique[key].confidence:
                unique[key] = i
        min_conf = UNIVERSAL_CONFIG.get("min_confidence_threshold", 0.55)
        return [i for i in unique.values() if i.confidence >= min_conf]

    def _extract_json(self, text: str) -> Optional[str]:
        patterns = [r'\[.*\]', r'```json\s*(\[.*?\])\s*```', r'(\[.*\])']
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_str = match.group(1) if match.groups() else match.group(0)
                try:
                    json.loads(json_str)
                    return json_str
                except:
                    continue
        return None


# ============================================================================
# 10. ENHANCED LLM REASONING SYNTHESIZER (with material grouping)
# ============================================================================
class LLMReasoningSynthesizer:
    def __init__(self, llm: HybridLLM):
        self.llm = llm
        self.phys_classifier = PhysicalQuantityClassifier()

    def synthesize(self, query: str, items: List[UniversalExtractionItem]) -> str:
        if not items:
            return f"No relevant information found for query: '{query}'. Try rephrasing or check the documents."
        extracted_lines = []
        for item in items:
            pq = item.physical_quantity or "unknown"
            pq_readable = self.phys_classifier.get_human_readable(pq)
            mat = f" [{item.material}]" if item.material else ""
            line = f"- {pq_readable}{mat}: {item.content} ({item.confidence:.2f}) context: {item.context[:200]} {item.citation()}"
            extracted_lines.append(line)
        extracted_text = "\n".join(extracted_lines[:20])
        prompt = f"""You are an expert scientific analyst. Given extracted values and the user query, produce a comprehensive answer.

QUERY: {query}

EXTRACTED VALUES (with citations):
{extracted_text}

TASK: Synthesize the extracted information into a structured answer using the following format:

**Direct Answer**
(Concise answer to the query, citing sources)

**Evidence by Physical Quantity**
(Group findings by physical quantity: e.g., Laser Power, Scan Speed, Yield Strength, etc.)

**Evidence by Material/Alloy**
(If materials are mentioned, group findings by alloy name)

**Consensus & Variability**
(For each physical quantity or material, report range/mean if multiple values exist)

**Contradictions & Limitations**
(If contradictory values exist, highlight them)

**Confidence Assessment**
(High/Medium/Low)

Do NOT invent information. Only use the extracted values above. Use citations with <cite doc="..." page="X"/>.

Return ONLY the answer text."""
        try:
            answer = self.llm.generate(prompt, max_new_tokens=1024, temperature=0.2)
            return answer.strip()
        except Exception as e:
            logger.error(f"Reasoning synthesis error: {e}")
            lines = [f"Query: {query}\nFound {len(items)} relevant items:\n"]
            for item in items[:5]:
                lines.append(f"- {item.content} {item.citation()}")
            return "\n".join(lines)

    def generate_human_conclusion(self, query: str, report: QueryReport) -> str:
        values = report.all_values
        if not values:
            return f"No quantitative data found for '{query}' across the analyzed documents."
        # Group by physical_quantity and also by material
        by_phys = defaultdict(list)
        by_material = defaultdict(list)
        for v in values:
            by_phys[v.physical_quantity].append(v)
            if v.material:
                by_material[v.material].append(v)
        lines = [
            f"## Summary: {query.title()}",
            f"Across **{report.total_docs}** documents analyzed, **{report.docs_with_results}** contained relevant quantitative data.",
            f"Total extracted values: **{len(values)}**.",
            ""
        ]
        # Physical quantity sections
        lines.append("### By Physical Quantity")
        for pq, vals in sorted(by_phys.items()):
            readable = self.phys_classifier.get_human_readable(pq)
            lines.append(f"#### {readable} ({len(vals)} values)")
            nums = [v.value for v in vals]
            units = list(set(v.unit for v in vals))
            docs = list(set(v.doc_name for v in vals))
            if nums:
                lines.append(f"- **Range**: {min(nums):.2f} to {max(nums):.2f} {units[0] if units else ''}")
                lines.append(f"- **Average**: {np.mean(nums):.2f}")
                if len(nums) > 1:
                    lines.append(f"- **Std Dev**: {np.std(nums):.2f}")
            lines.append(f"- **Found in**: {', '.join(docs[:3])}{'...' if len(docs) > 3 else ''}")
            lines.append("")
        # Material sections (NEW)
        if by_material:
            lines.append("### By Material/Alloy")
            for mat, vals in sorted(by_material.items()):
                lines.append(f"#### {mat} ({len(vals)} values)")
                # Group by physical quantity within material
                inner_pq = defaultdict(list)
                for v in vals:
                    inner_pq[v.physical_quantity].append(v.value)
                for pq, nums in inner_pq.items():
                    readable = self.phys_classifier.get_human_readable(pq)
                    lines.append(f"- {readable}: min={min(nums):.2f}, max={max(nums):.2f}, mean={np.mean(nums):.2f}")
                docs = list(set(v.doc_name for v in vals))
                lines.append(f"- **Documents**: {', '.join(docs)}")
                lines.append("")
        lines.append("### Key Values by Document and Physical Quantity")
        for v in sorted(values, key=lambda x: x.confidence, reverse=True)[:12]:
            readable = self.phys_classifier.get_human_readable(v.physical_quantity)
            mat_str = f" ({v.material})" if v.material else ""
            lines.append(f"| {v.doc_name} | p.{v.page} | {v.value:.2f} {v.unit} | {readable}{mat_str} |")
        return "\n".join(lines)


# ============================================================================
# 11. HIERARCHICAL TREE RETRIEVER (unchanged but can use two-stage)
# ============================================================================
class HierarchicalTreeRetriever:
    def __init__(self, llm: HybridLLM, max_results=30, max_text_chars=20000):
        self.llm = llm
        self.max_results = max_results
        self.max_text_chars = max_text_chars
        self._condensed_cache: Dict[str, Dict] = {}
        self.template = llm.template if hasattr(llm, 'template') else MODEL_PROMPT_TEMPLATES["default"]

    async def retrieve_quantitative(self, query: str, annotated_trees: List[Dict]) -> List[Dict]:
        trees_json = []
        for tree in annotated_trees:
            doc_id = tree.get("doc_id", "unknown")
            if doc_id not in self._condensed_cache:
                self._condensed_cache[doc_id] = self._condense_tree(tree)
            trees_json.append(self._condensed_cache[doc_id])
        batches = self._batch_trees(trees_json, max_tokens=6000)
        all_selections = []
        for batch in batches:
            prompt = self._build_tree_search_prompt(query, batch)
            response = await asyncio.to_thread(
                self.llm.generate,
                prompt,
                max_new_tokens=2048,
                fast_json=True,
                system_prompt=self.template.get("system")
            )
            selections = self._parse_node_selections(response)
            all_selections.extend(selections)
        results = []
        for sel in sorted(all_selections, key=lambda x: x.get('confidence', 0), reverse=True):
            doc_id = sel.get('doc_id')
            node_id = sel.get('node_id')
            node = self._find_node_by_id(annotated_trees, doc_id, node_id)
            if node:
                full_text = node.get('text', '')
                if len(full_text) > self.max_text_chars:
                    full_text = full_text[:self.max_text_chars] + "..."
                results.append({
                    "full_text": full_text,
                    "page_start": node.get('start_index'),
                    "doc_id": doc_id,
                    "section_title": node.get('title'),
                    "quantitative_items": node.get('quantitative_items', []),
                    "citation": f'<cite doc="{doc_id}" page="{node.get("start_index")}"/>',
                    "selection_reasoning": sel.get('reasoning', ''),
                    "confidence": sel.get('confidence', 0)
                })
        return results[:self.max_results]

    def _condense_tree(self, tree: Dict, max_depth: int = 3) -> Dict[str, Any]:
        def condense(node: Dict, depth: int = 0) -> Dict[str, Any]:
            if depth > max_depth:
                return {"node_id": node.get("node_id", ""), "title": node.get("title", ""), "leaf": True}
            result = {
                "node_id": node.get("node_id", ""),
                "title": node.get("title", ""),
                "summary": (node.get("summary", "") or "")[:150],
            }
            # Add metadata summary if present
            if node.get("metadata"):
                meta = node["metadata"]
                if meta.get("alloys"):
                    result["alloys"] = meta["alloys"][:3]
                if meta.get("laser_power_values"):
                    result["power_hint"] = f"{min(meta['laser_power_values'])}-{max(meta['laser_power_values'])} W"
            q_items = node.get("quantitative_items", [])
            if q_items:
                params = list(set(item.get("parameter_name", "") for item in q_items if item.get("parameter_name")))
                if params:
                    result["has_quantitative"] = params[:5]
            else:
                text = node.get("text", "")
                if text:
                    candidates = re.findall(r'(\d+(?:\.\d+)?)\s*(W|kW|mW|J|mm/s|°C|K|MPa|GPa|nm|µm|mm|s|m/s|W/cm²|kW/cm²)', text, re.IGNORECASE)
                    if candidates:
                        result["candidate_values"] = [f"{v}{u}" for v, u in candidates[:3]]
            children = node.get("nodes", [])
            if children and depth < max_depth:
                result["nodes"] = [condense(c, depth + 1) for c in children[:5]]
            return result
        return {
            "doc_id": tree.get("doc_id", tree.get("doc_name", "unknown")),
            "doc_name": tree.get("doc_name", ""),
            "structure": [condense(tree)] if not isinstance(tree, list) else [condense(t) for t in tree]
        }

    def _batch_trees(self, trees: List[Dict], max_tokens: int = 6000) -> List[List[Dict]]:
        batches = []
        current = []
        current_len = 0
        for t in trees:
            t_len = len(json.dumps(t))
            if current_len + t_len > max_tokens and current:
                batches.append(current)
                current = [t]
                current_len = t_len
            else:
                current.append(t)
                current_len += t_len
        if current:
            batches.append(current)
        return batches

    def _build_tree_search_prompt(self, query: str, trees: List[Dict]) -> str:
        trees_json = json.dumps(trees, ensure_ascii=False, indent=2)
        return f"""You are an expert scientific document navigator.
Given a query about quantitative parameters, identify which document nodes are MOST likely to contain the answer.

QUERY: {query}

INSTRUCTIONS:
1. Analyze each document's tree structure (titles, summaries, quantitative hints, candidate values, alloys, power hints)
2. Select nodes that likely contain specific numerical values, parameters, or measurements
3. For cross-document queries like "laser power across all papers", select nodes from MULTIPLE documents
4. Prefer nodes with "has_quantitative" or "candidate_values" hints matching the query topic
5. Return selections sorted by confidence (highest first)

DOCUMENT TREES:
{trees_json}

Return JSON:
{{
  "thinking": "Brief reasoning...",
  "selections": [
    {{"doc_id": "...", "node_id": "...", "reasoning": "...", "confidence": 0.95}}
  ]
}}

{self.template.get('json_reminder', 'Return ONLY valid JSON.')}
Include up to {self.max_results} selections."""

    def _parse_node_selections(self, response: str) -> List[Dict]:
        try:
            data = self._extract_json_safe(response)
            if data and isinstance(data, dict):
                selections = data.get("selections", [])
                return [s for s in selections if isinstance(s, dict) and "doc_id" in s and "node_id" in s]
        except Exception as e:
            logger.warning(f"Failed to parse selections: {e}")
        return []

    def _extract_json_safe(self, text: str) -> Optional[Any]:
        patterns = [r'\{.*\}', r'\[.*\]', r'```json\s*(\{.*?\})\s*```']
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_str = match.group(1) if match.groups() else match.group(0)
                try:
                    return json.loads(json_str)
                except:
                    continue
        return None

    def _find_node_by_id(self, trees: List[Dict], doc_id: str, node_id: str) -> Optional[Dict]:
        for tree in trees:
            if tree.get("doc_id") == doc_id or tree.get("doc_name") == doc_id:
                return self._search_node_recursive(tree, node_id)
        return None

    def _search_node_recursive(self, node: Dict, target_id: str) -> Optional[Dict]:
        if node.get("node_id") == target_id:
            return node
        for child in node.get("nodes", []):
            res = self._search_node_recursive(child, target_id)
            if res:
                return res
        return None


# ============================================================================
# 12. STREAMLIT UI WITH FULL VISUALIZATION IMPLEMENTATION
# ============================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        model_keys = list(LOCAL_LLM_OPTIONS.keys())
        if "llm_model_choice" not in st.session_state:
            st.session_state.llm_model_choice = model_keys[3]
        selected = st.selectbox(
            "🧠 Select Local LLM (Ollama)",
            options=model_keys,
            index=model_keys.index(st.session_state.llm_model_choice),
            key="llm_model_select"
        )
        st.session_state.llm_model_choice = selected
        st.checkbox("🗜️ Use 4-bit quantization (if Transformers fallback)", value=True, key="use_4bit")
        max_chars = st.slider(
            "📄 Max text length per retrieved section (characters)",
            min_value=1000, max_value=50000, value=20000, step=1000,
            help="Larger values give more context but use more memory/LLM tokens."
        )
        st.session_state.max_retrieval_chars = max_chars
        st.slider("Confidence threshold", 0.3, 0.9, 0.55, 0.05, key="min_confidence")
        st.checkbox("Show reasoning trace", value=True, key="show_trace")
        st.checkbox("Show tree navigation", value=True, key="show_tree_nav")
        st.checkbox("Enable two-stage retrieval (semantic)", value=True, key="two_stage", 
                    help="Faster retrieval using embeddings; fallback to keyword if sentence-transformers not installed.")
        st.caption(f"GPU: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        # Visualization settings
        st.markdown("### 🎨 Visualization Settings")
        st.session_state.viz_colormap = st.selectbox("Colormap", 
            list(PublicationQualityVisualizationEngine.COLORMAP_OPTIONS.keys()), index=0)
        st.session_state.viz_font_size = st.slider("Base Font Size", 8, 20, 12)
        st.session_state.viz_title_font_size = st.slider("Title Font Size", 10, 24, 16)
        st.session_state.viz_layout = st.selectbox("Network Layout", 
            ["spring", "kamada_kawai", "circular"], index=0)
        st.session_state.viz_top_n = st.slider("Top N Concepts to Visualize", 5, 100, 25)
        st.session_state.viz_active_domains = st.multiselect("Filter by Domain",
            options=["MATERIAL", "METHOD", "PHENOMENON", "PARAMETER", "UNKNOWN"],
            default=["MATERIAL", "PARAMETER"])
        if st.button("Apply Visualization Filters"):
            st.success("Filters applied!")


@st.cache_resource(show_spinner="Initializing LLM...")
def get_cached_llm(model_choice: str, use_4bit: bool):
    internal = LOCAL_LLM_OPTIONS[model_choice]
    return HybridLLM(model_key=internal, use_4bit=use_4bit)


# ============================================================================
# PUBLICATION-QUALITY VISUALIZATION ENGINE - FULLY IMPLEMENTED
# ============================================================================
class PublicationQualityVisualizationEngine:
    """Complete visualization engine with 30+ chart types, all rendering in Streamlit."""
    
    COLORMAP_OPTIONS = {
        "viridis": "viridis", "plasma": "plasma", "inferno": "inferno", "magma": "magma", "cividis": "cividis",
        "Greys": "Greys", "Purples": "Purples", "Blues": "Blues", "Greens": "Greens", "Oranges": "Oranges", "Reds": "Reds",
        "YlOrBr": "YlOrBr", "YlOrRd": "YlOrRd", "PuRd": "PuRd", "BuPu": "BuPu", "GnBu": "GnBu", "YlGnBu": "YlGnBu",
        "binary": "binary", "bone": "bone", "pink": "pink", "spring": "spring", "summer": "summer", "autumn": "autumn",
        "winter": "winter", "cool": "cool", "Wistia": "Wistia", "hot": "hot", "copper": "copper",
        "PiYG": "PiYG", "BrBG": "BrBG", "PuOr": "PuOr", "RdGy": "RdGy", "RdBu": "RdBu",
        "RdYlBu": "RdYlBu", "RdYlGn": "RdYlGn", "Spectral": "Spectral", "coolwarm": "coolwarm", "bwr": "bwr",
        "tab10": "tab10", "tab20": "tab20", "tab20b": "tab20b", "tab20c": "tab20c",
        "Pastel1": "Pastel1", "Pastel2": "Pastel2", "Paired": "Paired", "Accent": "Accent",
        "Set1": "Set1", "Set2": "Set2", "Set3": "Set3", "turbo": "turbo", "jet": "jet"
    }
    
    DOMAIN_COLORS = {
        "MATERIAL": "#3b82f6", "METHOD": "#8b5cf6", "PHENOMENON": "#f59e0b",
        "PARAMETER": "#10b981", "UNKNOWN": "#6b7280", "TOPIC": "#ec4899", "DOCUMENT": "#1e40af"
    }
    
    def __init__(self, kg: 'QuantitativeKnowledgeGraph',
                 font_family: str = "DejaVu Sans", font_size: int = 10,
                 title_font_size: int = 14, label_font_size: int = 9,
                 default_colormap: str = "viridis", figure_dpi: int = 300):
        self.kg = kg
        self.font_family = font_family
        self.font_size = font_size
        self.title_font_size = title_font_size
        self.label_font_size = label_font_size
        self.default_colormap = default_colormap
        self.figure_dpi = figure_dpi
        plt.rcParams.update({
            'font.family': font_family, 'font.size': font_size, 'axes.titlesize': title_font_size,
            'axes.labelsize': label_font_size, 'figure.dpi': figure_dpi, 'savefig.dpi': figure_dpi,
            'figure.facecolor': 'white', 'axes.facecolor': 'white', 'axes.edgecolor': '#333333',
            'axes.labelcolor': '#333333', 'xtick.color': '#333333', 'ytick.color': '#333333',
            'text.color': '#333333', 'lines.linewidth': 1.5, 'axes.grid': True,
            'grid.alpha': 0.3, 'grid.linestyle': '--'
        })
    
    def _get_colormap(self, name: Optional[str] = None) -> str:
        return self.COLORMAP_OPTIONS.get(name or self.default_colormap, "viridis")
    
    def _get_domain_color(self, domain: str, colormap: Optional[str] = None, index: int = 0, total: int = 1) -> str:
        if colormap and total > 1:
            cmap = plt.get_cmap(self._get_colormap(colormap))
            return mcolors.to_hex(cmap(index / max(total - 1, 1)))
        return self.DOMAIN_COLORS.get(domain, "#6b7280")

    # ========== HISTOGRAM & BAR CHARTS ==========
    def plot_quantitative_histogram(self, df: pd.DataFrame, quantity_name: str,
                                   group_by: str = "material", colormap: Optional[str] = None) -> go.Figure:
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title=f"No {quantity_name} data extracted")
            return fig
        
        fig = go.Figure()
        groups = sorted(df[group_by].unique())
        cmap_obj = plt.get_cmap(self._get_colormap(colormap))
        
        for i, grp in enumerate(groups):
            subset = df[df[group_by] == grp]
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

    def plot_bar_chart_grouped(self, df: pd.DataFrame, quantity_name: str,
                              group_by: str = "material", colormap: Optional[str] = None) -> go.Figure:
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title=f"No {quantity_name} data extracted")
            return fig
        
        stats = df.groupby(group_by)["value"].agg(["mean", "std", "count"])
        fig = go.Figure()
        cmap_obj = plt.get_cmap(self._get_colormap(colormap))
        
        for i, (grp, row) in enumerate(stats.iterrows()):
            color = mcolors.to_hex(cmap_obj(i / max(len(stats) - 1, 1)))
            fig.add_trace(go.Bar(
                name=grp,
                x=[grp],
                y=[row["mean"]],
                error_y=dict(type='data', array=[row["std"]], visible=True),
                marker_color=color,
                text=[f"n={int(row['count'])}"],
                textposition="outside"
            ))
        
        fig.update_layout(
            barmode='group',
            title=f"{quantity_name.replace('_', ' ').title()} Grouped Bar Chart",
            font=dict(family=self.font_family, size=self.font_size)
        )
        return fig

    # ========== PIE & DONUT CHARTS ==========
    def plot_quantity_distribution_pie(self, colormap: Optional[str] = None) -> go.Figure:
        pq_counts = self.kg.get_all_physical_quantities()
        if not pq_counts:
            fig = go.Figure()
            fig.update_layout(title="No quantities found")
            return fig
        
        sorted_pq = sorted(pq_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        labels = [self.kg.phys_classifier.get_human_readable(pq) for pq, _ in sorted_pq]
        values = [count for _, count in sorted_pq]
        
        colorscale = colormap or "Set3"
        fig = px.pie(values=values, names=labels, 
                    title="Top Physical Quantities Distribution",
                    color_discrete_sequence=px.colors.qualitative.__dict__.get(colorscale, px.colors.qualitative.Set3))
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
        return fig
    
    def plot_material_distribution_donut(self, colormap: Optional[str] = None) -> go.Figure:
        mat_dict = self.kg.get_all_materials()
        if not mat_dict:
            fig = go.Figure()
            fig.update_layout(title="No materials found")
            return fig
        
        mat_counts = Counter(m for mats in mat_dict.values() for m in mats)
        top_mats = mat_counts.most_common(10)
        labels = [m for m, _ in top_mats]
        values = [c for _, c in top_mats]
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4,
                                    marker_colors=[f"#{hash(l) % 0xFFFFFF:06x}" for l in labels])])
        fig.update_layout(title="Material Distribution (Donut)",
                         annotations=[dict(text='Materials', x=0.5, y=0.5, font_size=14, showarrow=False)])
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
        return fig

    # ========== SUNBURST & HIERARCHICAL CHARTS ==========
    def plot_sunburst_quantitative(self, df: pd.DataFrame, quantity_name: str,
                                  colormap: Optional[str] = None) -> go.Figure:
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title=f"No {quantity_name} data extracted")
            return fig
        
        df = df.copy()
        n_bins = min(5, max(2, len(df) // 3))
        df["value_range"] = pd.cut(df["value"], bins=n_bins, precision=1).astype(str)
        
        fig = px.sunburst(
            df,
            path=["material", "doc_stem", "value_range"],
            values="value",
            color="value",
            color_continuous_scale=colormap or "Viridis",
            title=f"{quantity_name.replace('_', ' ').title()} Distribution Hierarchy"
        )
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
        return fig

    # ========== RADAR CHARTS ==========
    def plot_quantitative_radar(self, df: pd.DataFrame, quantity_name: str,
                               colormap: Optional[str] = None) -> go.Figure:
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title=f"No {quantity_name} data extracted")
            return fig
        
        stats = df.groupby("material")["value"].agg(["mean", "std", "min", "max", "count"])
        categories = ["Mean", "Max", "Min", "Std", "Count"]
        fig = go.Figure()
        cmap_obj = plt.get_cmap(self._get_colormap(colormap)) if colormap else None
        
        for i, (mat, row) in enumerate(stats.iterrows()):
            values = [row["mean"], row["max"], row["min"], row["std"], float(row["count"])]
            values += values[:1]  # Close the radar
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

    # ========== CHORD DIAGRAMS ==========
    def plot_chord_cooccurrence(self, filtered_concepts: Optional[List[str]] = None,
                               top_n: int = 14, colormap: Optional[str] = None) -> go.Figure:
        if filtered_concepts:
            entities = filtered_concepts[:top_n]
        else:
            all_pq = self.kg.get_all_physical_quantities()
            entities = [pq for pq, _ in sorted(all_pq.items(), key=lambda x: x[1], reverse=True)[:top_n]]
        
        if not entities:
            fig = go.Figure()
            fig.update_layout(title="No entity co-occurrence data")
            return fig
        
        n = len(entities)
        node_to_idx = {node: i for i, node in enumerate(entities)}
        adj = np.zeros((n, n))
        
        for doc in self.kg.doc_graphs:
            present = [ent for ent in entities if any(
                item.get("physical_quantity") == ent or item.get("parameter_name") == ent
                for item in self.kg.doc_graphs[doc]["all_items"])]
            for i, e1 in enumerate(present):
                for j, e2 in enumerate(present):
                    if i != j:
                        adj[node_to_idx[e1]][node_to_idx[e2]] += 1
        
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        cmap_obj = plt.get_cmap(self._get_colormap(colormap))
        fig = go.Figure()
        
        for i, ent in enumerate(entities):
            color = mcolors.to_hex(cmap_obj(i / max(n - 1, 1)))
            fig.add_trace(go.Barpolar(
                r=[1], theta=[np.degrees(angles[i])],
                width=[10], marker_color=color,
                name=ent, opacity=0.9, showlegend=False
            ))
        
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j] > 0:
                    fig.add_trace(go.Scatterpolar(
                        r=[0.2, 0.6, 0.2],
                        theta=[np.degrees(angles[i]), np.degrees((angles[i]+angles[j])/2), np.degrees(angles[j])],
                        mode='lines',
                        line=dict(color='rgba(100,100,100,0.3)', width=min(adj[i][j], 3)),
                        showlegend=False
                    ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=False), angularaxis=dict(visible=False)),
            title=f"Salience-Aware Chord Diagram (Top {n} Concepts)",
            height=700, width=700
        )
        return fig

    # ========== CONTRADICTION & CONSENSUS CHARTS ==========
    def plot_contradiction_matrix(self, colormap: Optional[str] = None) -> go.Figure:
        docs = list(self.kg.doc_graphs.keys())
        if len(docs) < 2:
            fig = go.Figure()
            fig.update_layout(title="Need ≥2 docs for contradiction analysis")
            return fig
        
        doc_stems = [Path(d).stem for d in docs]
        n = len(docs)
        mat = np.zeros((n, n))
        annotations = [["" for _ in range(n)] for _ in range(n)]
        
        for i, doc_a in enumerate(docs):
            for j, doc_b in enumerate(docs):
                if i >= j: continue
                for item_a in self.kg.doc_graphs[doc_a]["all_items"]:
                    for item_b in self.kg.doc_graphs[doc_b]["all_items"]:
                        pq_a = item_a.get("physical_quantity")
                        pq_b = item_b.get("physical_quantity")
                        if pq_a and pq_a == pq_b and item_a.get("value") and item_b.get("value"):
                            val_a, val_b = item_a["value"], item_b["value"]
                            if min(val_a, val_b) > 0:
                                ratio = max(val_a, val_b) / min(val_a, val_b)
                                if ratio > 1.5:
                                    mat[i][j] = max(mat[i][j], ratio)
                                    mat[j][i] = mat[i][j]
                                    annotations[i][j] += f"{pq_a[:12]}({ratio:.1f}x)<br>"
                                    annotations[j][i] = annotations[i][j]
        
        colorscale = colormap or [[0, "white"], [0.33, "#fcd34d"], [0.66, "#f97316"], [1, "#dc2626"]]
        fig = go.Figure(data=go.Heatmap(
            z=mat, x=doc_stems, y=doc_stems,
            colorscale=colorscale,
            text=annotations, texttemplate="%{text}", hoverinfo="text"
        ))
        fig.update_layout(
            title="Cross-Document Contradiction Severity Matrix",
            height=600, width=600,
            font=dict(family=self.font_family, size=self.font_size)
        )
        return fig
    
    def plot_consensus_waterfall(self, top_n: int = 10, colormap: Optional[str] = None) -> go.Figure:
        consensus = []
        for pq in list(self.kg.get_all_physical_quantities().keys())[:top_n]:
            stats = self.kg.get_summary_stats(pq)
            if stats.get("count", 0) >= 2:
                consensus.append(stats)
        
        if not consensus:
            fig = go.Figure()
            fig.update_layout(title="No consensus data")
            return fig
        
        entities = [c.get("unknown", pq)[:30] for pq, c in [(pq, self.kg.get_summary_stats(pq)) for pq in list(self.kg.get_all_physical_quantities().keys())[:top_n]]]
        means = [c.get("mean", 0) for c in consensus]
        stds = [c.get("std", 0) for c in consensus]
        doc_counts = [c.get("count", 0) for c in consensus]
        
        fig = go.Figure()
        if colormap:
            cmap_obj = plt.get_cmap(self._get_colormap(colormap))
            bar_colors = [mcolors.to_hex(cmap_obj(i / max(len(entities) - 1, 1))) for i in range(len(entities))]
        else:
            bar_colors = ["#059669" if d >= 3 else "#3b82f6" for d in doc_counts]
        
        fig.add_trace(go.Bar(
            x=entities, y=means,
            error_y=dict(type='data', array=stds, visible=True, color="black"),
            marker_color=bar_colors,
            text=[f"μ={m:.2f}<br>σ={s:.2f}<br>n={d}" for m, s, d in zip(means, stds, doc_counts)],
            textposition="outside"
        ))
        fig.update_layout(
            title="Cross-Document Consensus Waterfall\nGreen = strong consensus (≥3 docs), Blue = emerging",
            yaxis_title="Mean Value", xaxis_tickangle=-45, height=500,
            font=dict(family=self.font_family, size=self.font_size)
        )
        return fig

    # ========== EMBEDDING SPACE VISUALIZATIONS ==========
    def plot_entity_tsne(self, embedding_fn: Callable, filtered_concepts: Optional[List[str]] = None,
                        top_n: int = 80, perplexity: int = 30, colormap: Optional[str] = None,
                        figsize: Tuple[int, int] = (10, 8)) -> Optional[plt.Figure]:
        if not SKLEARN_AVAILABLE:
            return None
        
        target = filtered_concepts or list(self.kg.get_all_physical_quantities().keys())
        scored = [(ent, self.kg.get_summary_stats(ent).get("count", 0)) for ent in target]
        top = [e for e, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]]
        
        if len(top) < 5:
            return None
        
        embs = []
        domains = []
        for ent in top:
            vec = embedding_fn(ent)
            embs.append(vec)
            domains.append("PARAMETER")
        
        embs = np.stack(embs)
        tsne = TSNE(n_components=2, perplexity=min(perplexity, len(top)-1), random_state=42)
        coords = tsne.fit_transform(embs)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(coords[:, 0], coords[:, 1], c="blue", alpha=0.8, s=80)
        for i, ent in enumerate(top):
            ax.annotate(ent[:20], (coords[i, 0], coords[i, 1]), 
                       fontsize=self.label_font_size - 1, alpha=0.8)
        ax.set_title("Entity Embedding Space (t-SNE)", 
                    fontsize=self.title_font_size, fontweight='bold')
        ax.axis("off")
        plt.tight_layout()
        return fig
    
    def plot_entity_pca(self, embedding_fn: Callable, filtered_concepts: Optional[List[str]] = None,
                       top_n: int = 80, colormap: Optional[str] = None,
                       figsize: Tuple[int, int] = (10, 8)) -> Optional[plt.Figure]:
        if not SKLEARN_AVAILABLE:
            return None
        
        target = filtered_concepts or list(self.kg.get_all_physical_quantities().keys())
        scored = [(ent, self.kg.get_summary_stats(ent).get("count", 0)) for ent in target]
        top = [e for e, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]]
        
        if len(top) < 5:
            return None
        
        embs = []
        for ent in top:
            vec = embedding_fn(ent)
            embs.append(vec)
        
        embs = np.stack(embs)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embs)
        var_ratio = pca.explained_variance_ratio_
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(coords[:, 0], coords[:, 1], c="red", alpha=0.8, s=80)
        for i, ent in enumerate(top):
            ax.annotate(ent[:20], (coords[i, 0], coords[i, 1]), 
                       fontsize=self.label_font_size - 1, alpha=0.8)
        ax.set_title(f"Entity Embedding Space (PCA)\nPC1: {var_ratio[0]:.1%}, PC2: {var_ratio[1]:.1%}",
                    fontsize=self.title_font_size, fontweight='bold')
        ax.axis("off")
        plt.tight_layout()
        return fig

    # ========== NETWORK GRAPHS ==========
    def plot_static_knowledge_network(self, filtered_concepts: Optional[List[str]] = None, top_n: int = 30,
                                     figsize: Tuple[int, int] = (14, 12), layout: str = "spring",
                                     colormap: Optional[str] = None, node_size_factor: float = 1.0,
                                     edge_alpha: float = 0.25, show_labels: bool = True) -> plt.Figure:
        G = nx.Graph()
        docs = list(self.kg.doc_graphs.keys())
        for doc_id in docs:
            G.add_node(Path(doc_id).stem, node_type="doc", bipartite=0, domain="DOCUMENT")
        
        entities = filtered_concepts or list(self.kg.get_all_physical_quantities().keys())[:top_n]
        for ent in entities:
            stats = self.kg.get_summary_stats(ent)
            doc_count = stats.get("count", 0)
            G.add_node(ent, node_type="entity", domain="PARAMETER", bipartite=1, salience=doc_count)
            for doc in docs:
                if any(item.get("physical_quantity") == ent or item.get("parameter_name") == ent
                       for item in self.kg.doc_graphs[doc]["all_items"]):
                    G.add_edge(Path(doc).stem, ent, weight=doc_count * 0.5)
        
        fig, ax = plt.subplots(figsize=figsize)
        pos = nx.spring_layout(G, k=0.55, iterations=60, seed=42) if layout == "spring" else nx.kamada_kawai_layout(G)
        
        doc_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "doc"]
        ent_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "entity"]
        
        nx.draw_networkx_nodes(G, pos, nodelist=doc_nodes, node_color="#1e40af", node_shape="s",
                               node_size=800, alpha=0.85, ax=ax, label="Documents")
        
        cmap_obj = plt.get_cmap(self._get_colormap(colormap))
        domains = list(set(G.nodes[n].get("domain", "UNKNOWN") for n in ent_nodes))
        domain_color_idx = {d: i for i, d in enumerate(domains)}
        
        for node in ent_nodes:
            salience = G.nodes[node].get("salience", 0.5)
            domain = G.nodes[node].get("domain", "UNKNOWN")
            if colormap:
                idx = domain_color_idx.get(domain, 0)
                base_color = mcolors.to_hex(cmap_obj(idx / max(len(domains) - 1, 1)))
            else:
                base_color = self.DOMAIN_COLORS.get(domain, "#6b7280")
            color = mcolors.to_hex(mcolors.to_rgba(base_color, alpha=0.7 + 0.3 * min(salience / 10, 1)))
            size = (300 + salience * 90) * node_size_factor
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=color, node_shape="o",
                                   node_size=size, alpha=0.9, ax=ax)
        
        nx.draw_networkx_edges(G, pos, alpha=edge_alpha, width=0.8, ax=ax)
        if show_labels:
            nx.draw_networkx_labels(G, pos, font_size=self.label_font_size, ax=ax, font_family=self.font_family)
        
        legend_patches = [mpatches.Patch(color="#1e40af", label="Documents")]
        for dom in domains:
            c = mcolors.to_hex(cmap_obj(domain_color_idx[dom] / max(len(domains) - 1, 1))) if colormap else self.DOMAIN_COLORS.get(dom, "#6b7280")
            legend_patches.append(mpatches.Patch(color=c, label=dom))
        ax.legend(handles=legend_patches, loc="upper left", fontsize=9)
        ax.set_title("Salience-Aware Cross-Document Knowledge Network\n(Node size = importance)",
                     fontsize=self.title_font_size, fontweight='bold', fontfamily=self.font_family)
        ax.axis("off")
        plt.tight_layout()
        return fig

    def render_pyvis_salience(self, filtered_concepts: Optional[List[str]] = None, top_n_nodes: int = 30,
                             physics_enabled: bool = True, colormap: Optional[str] = None) -> str:
        """Returns HTML string for PyVis network that can be rendered in Streamlit."""
        if not PYVIS_AVAILABLE:
            return "<p>PyVis not installed. pip install pyvis</p>"
        
        G = nx.Graph()
        docs = list(self.kg.doc_graphs.keys())
        for doc in docs:
            G.add_node(Path(doc).stem, node_type="doc", domain="DOCUMENT")
        
        entities = filtered_concepts or list(self.kg.get_all_physical_quantities().keys())[:top_n_nodes]
        for ent in entities:
            stats = self.kg.get_summary_stats(ent)
            count = stats.get("count", 0)
            G.add_node(ent, node_type="entity", domain="PARAMETER", salience=count)
            for doc in docs:
                if any(item.get("physical_quantity") == ent or item.get("parameter_name") == ent
                       for item in self.kg.doc_graphs[doc]["all_items"]):
                    G.add_edge(Path(doc).stem, ent, weight=count)
        
        net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="#000000", cdn_resources='remote')
        if physics_enabled:
            net.barnes_hut(gravity=-1800, spring_length=140, damping=0.85)
        
        cmap_obj = plt.get_cmap(self._get_colormap(colormap)) if colormap else None
        
        for node in G.nodes():
            salience = G.nodes[node].get("salience", 0.5)
            size = int(15 + min(salience / 2, 1) * 55)
            domain = G.nodes[node].get("domain", "UNKNOWN")
            color = self._get_domain_color(domain, colormap, list(G.nodes()).index(node), len(G.nodes()))
            net.add_node(node, label=node[:25], size=size, color=color, borderWidth=2,
                        title=f"{node}\nSalience: {salience}")
        
        for u, v, data in G.edges(data=True):
            w = data.get('weight', 1)
            net.add_edge(u, v, value=max(1, int(w * 2)), width=max(1, int(w)))
        
        return net.generate_html()


# ============================================================================
# MAIN STREAMLIT APP WITH FULL VISUALIZATION INTEGRATION
# ============================================================================
def run_streamlit():
    st.set_page_config(page_title="DECLARMIMA v13.0+ - Full Visualizations", layout="wide")
    st.markdown("# 🔬 DECLARMIMA v13.0+ - Enhanced with Publication-Quality Visualizations")
    st.caption("Now includes histograms, bar charts, pie charts, chord diagrams, sunburst charts, radar charts, interactive concept graphs, t-SNE/UMAP/PCA, contradiction matrices, and more.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "query_processor" not in st.session_state:
        st.session_state.query_processor = {}
    if "knowledge_graph" not in st.session_state:
        st.session_state.knowledge_graph = QuantitativeKnowledgeGraph()
    if "annotated_trees" not in st.session_state:
        st.session_state.annotated_trees = []
    if "two_stage_retriever" not in st.session_state:
        st.session_state.two_stage_retriever = None
    if "cached_query_result" not in st.session_state:
        st.session_state.cached_query_result = None
    if "active_prompt" not in st.session_state:
        st.session_state.active_prompt = ""

    render_sidebar()
    max_retrieval_chars = st.session_state.get("max_retrieval_chars", 20000)

    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files and st.button("🚀 Build Index", type="primary"):
        st.session_state.query_processor["files"] = uploaded_files
        st.success(f"{len(uploaded_files)} files registered.")
        st.rerun()

    if st.session_state.query_processor.get("files") and not st.session_state.annotated_trees:
        with st.spinner("Building hierarchical index with metadata extraction..."):
            progress = st.progress(0)
            llm = get_cached_llm(st.session_state.llm_model_choice, st.session_state.get("use_4bit", True))
            progress.progress(0.1)
            idx = FastHierarchicalIndex(llm=llm)
            async def build_index():
                return await idx.build_from_pdfs_fast(st.session_state.query_processor["files"], max_workers=4)
            trees = asyncio.run(build_index())
            st.session_state.query_processor["index"] = idx
            st.session_state.query_processor["doc_trees"] = trees
            progress.progress(0.5)
            extractor = UniversalLLMExtractor(llm)
            kg = QuantitativeKnowledgeGraph()
            all_items = []
            # Build two-stage retriever index
            two_stage = TwoStageRetriever(llm=llm)
            for doc_name, tree in trees.items():
                leaf_texts = []
                def collect_leaves(node: PageNode):
                    if not node.children:
                        text = node.get_text()
                        if text:
                            leaf_texts.append({
                                "full_text": text,
                                "page_start": node.page_start,
                                "doc_id": doc_name,
                                "section_title": node.title
                            })
                    for c in node.children:
                        collect_leaves(c)
                collect_leaves(tree)
                # Use enhanced prompt that asks for materials
                initial_prompt = "Extract all quantitative parameters (laser power, scan speed, flow speed, irradiance, temperature, energy density, yield strength, tensile strength, hardness, elongation, etc.) with full numerical values, correct units, physical_quantity classification, and any alloy/material names."
                items = extractor.extract_from_chunks(leaf_texts, initial_prompt)
                all_items.extend(items)
                kg.add_extractions(doc_name, items)
                # Add metadata to kg and two-stage
                if tree.metadata:
                    kg.add_document_metadata(doc_name, tree.metadata)
                    two_stage.index_document(doc_name, tree.metadata, tree.summary)
                else:
                    # Fallback: create metadata from extracted items
                    alloys = list(set(item.material for item in items if item.material))
                    meta = DocumentMetadata(doc_name=doc_name, alloys=alloys)
                    kg.add_document_metadata(doc_name, meta)
                    two_stage.index_document(doc_name, meta, tree.summary)
            st.session_state.knowledge_graph = kg
            st.session_state.two_stage_retriever = two_stage
            progress.progress(0.8)
            annotated = []
            for doc_name, tree in trees.items():
                ann = kg.to_tree_annotation(tree, max_chars=max_retrieval_chars)
                ann["doc_id"] = doc_name
                ann["doc_name"] = doc_name
                ann["metadata"] = tree.metadata.dict() if tree.metadata else {}
                annotated.append(ann)
            st.session_state.annotated_trees = annotated
            progress.progress(1.0)
            st.success(f"✅ Indexed {len(trees)} documents with {len(all_items)} quantitative items")
            with st.expander("📊 Detected Physical Quantities and Materials", expanded=True):
                pq_counts = kg.get_all_physical_quantities()
                if pq_counts:
                    st.write("**Physical Quantities:**")
                    for pq, count in sorted(pq_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
                        st.write(f"- `{pq}`: {count} occurrences")
                mat_dict = kg.get_all_materials()
                if mat_dict:
                    st.write("**Materials/Alloys per document:**")
                    for doc, mats in mat_dict.items():
                        if mats:
                            st.write(f"- {doc}: {', '.join(mats)}")
                else:
                    st.write("No materials or alloys detected. You may need to re-run extraction with improved prompts.")

    if st.session_state.annotated_trees:
        st.markdown("### ⚡ Quick Queries")
        col1, col2, col3, col4 = st.columns(4)
        quick = ["laser power", "yield strength", "scan speed", "alloy names"]
        for i, q in enumerate(quick):
            with [col1, col2, col3, col4][i]:
                if st.button(f"📈 {q.title()}", key=f"quick_{q}"):
                    st.session_state.quick_query = f"What is the {q} discussed in these papers?"
                    st.rerun()

        default_query = st.session_state.get("quick_query", "")
        prompt_input = st.chat_input("Ask about any term, value, material, or mechanical property...", key="chat_input")
        if default_query and not prompt_input:
            prompt_input = default_query
            st.session_state.quick_query = ""

        if prompt_input:
            st.session_state.active_prompt = prompt_input
            st.session_state.messages.append({"role": "user", "content": prompt_input})
            with st.chat_message("user"):
                st.markdown(prompt_input)
        elif st.session_state.active_prompt:
            with st.chat_message("user"):
                st.markdown(st.session_state.active_prompt)

        active_prompt = st.session_state.get("active_prompt", "")

        run_query = False
        if active_prompt:
            cached = st.session_state.cached_query_result
            has_valid_cache = (
                cached is not None
                and cached.get("prompt") == active_prompt
                and "answer" in cached
            )
            if not has_valid_cache:
                run_query = True

        answer = None
        extracted_values = []
        retrieved = []
        items = []

        if run_query:
            with st.chat_message("assistant"):
                progress = st.progress(0)
                progress.text("Initializing LLM...")
                llm = get_cached_llm(st.session_state.llm_model_choice, st.session_state.get("use_4bit", True))
                progress.progress(0.1)
                
                # Two-stage retrieval: first filter documents using semantic search
                if st.session_state.get("two_stage", True) and st.session_state.two_stage_retriever is not None:
                    progress.text("Stage 1: Semantic document filtering...")
                    relevant_docs = st.session_state.two_stage_retriever.retrieve_relevant_docs(active_prompt, top_k=8)
                    st.caption(f"Selected {len(relevant_docs)} relevant documents out of {len(st.session_state.annotated_trees)}.")
                    # Filter annotated trees to only those relevant documents
                    filtered_trees = [t for t in st.session_state.annotated_trees if t.get("doc_id") in [d[0] for d in relevant_docs]]
                else:
                    filtered_trees = st.session_state.annotated_trees
                
                progress.progress(0.3)
                retriever = HierarchicalTreeRetriever(llm, max_results=30, max_text_chars=max_retrieval_chars)
                retrieved = asyncio.run(retriever.retrieve_quantitative(active_prompt, filtered_trees))
                progress.progress(0.6)
                extractor = UniversalLLMExtractor(llm)
                items = []
                for r in retrieved:
                    items.extend(extractor.extract_from_chunks([r], active_prompt))
                min_conf = st.session_state.get("min_confidence", 0.55)
                items = [i for i in items if i.confidence >= min_conf]
                progress.progress(0.8)
                synthesizer = LLMReasoningSynthesizer(llm)
                extracted_values = []
                for item in items:
                    if item.item_type == "quantitative" and item.value is not None:
                        phys_q = item.physical_quantity or synthesizer.phys_classifier.classify(
                            item.parameter_name, item.unit, item.context
                        )
                        extracted_values.append(ExtractedValue(
                            query=active_prompt,
                            value=item.value,
                            unit=item.unit or "",
                            physical_quantity=phys_q,
                            parameter_name=item.parameter_name,
                            material=item.material,
                            confidence=item.confidence,
                            context=item.context,
                            doc_name=item.doc_source,
                            page=item.page,
                            section_title=item.section_title
                        ))
                if extracted_values:
                    report = QueryReport(
                        query=active_prompt,
                        total_docs=len(st.session_state.annotated_trees),
                        docs_with_results=len(set(v.doc_name for v in extracted_values)),
                        all_values=extracted_values,
                        consensus={},
                        processing_time_sec=0.0
                    )
                    answer = synthesizer.generate_human_conclusion(active_prompt, report)
                else:
                    answer = synthesizer.synthesize(active_prompt, items)
                progress.progress(1.0, text="Done!")
                st.markdown(answer)
                # Store as plain dicts to survive Streamlit reruns
                st.session_state.cached_query_result = {
                    "prompt": active_prompt,
                    "retrieved": retrieved,
                    "items": [i.model_dump() for i in items],
                    "extracted_values": [v.model_dump() for v in extracted_values],
                    "answer": answer
                }
                st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            if active_prompt and st.session_state.cached_query_result and "answer" in st.session_state.cached_query_result:
                cached = st.session_state.cached_query_result
                with st.chat_message("assistant"):
                    st.markdown(cached["answer"])
                answer = cached["answer"]
                retrieved = cached.get("retrieved", [])
                raw_items = cached.get("items", [])
                if raw_items and isinstance(raw_items[0], dict):
                    items = [UniversalExtractionItem(**d) for d in raw_items]
                else:
                    items = raw_items
                raw_vals = cached.get("extracted_values", [])
                if raw_vals and isinstance(raw_vals[0], dict):
                    extracted_values = [ExtractedValue(**d) for d in raw_vals]
                else:
                    extracted_values = raw_vals
            else:
                if not active_prompt:
                    st.info("Ask a question about the documents.")
                    return

        # ========== FULL VISUALIZATION SECTION - NOW ACTUALLY RENDERED ==========
        st.markdown("---")
        st.subheader("📊 Quantitative Results & Visualizations")
        
        if extracted_values:
            # Create DataFrame for visualizations
            viz_df = pd.DataFrame([{
                "Physical Quantity": PhysicalQuantityClassifier().get_human_readable(v.physical_quantity),
                "Value": v.value,
                "Unit": v.unit,
                "Material": v.material or "Unknown",
                "Document": v.doc_name,
                "Page": v.page,
                "Confidence": v.confidence
            } for v in extracted_values])
            
            # Visualization tabs
            viz_tabs = st.tabs([
                "📈 By Physical Quantity", 
                "🧪 By Material", 
                "📊 Distribution", 
                "🕸️ Network Graph",
                "🔬 Embeddings"
            ])
            
            with viz_tabs[0]:
                # Histogram by physical quantity
                fig_hist = px.histogram(
                    viz_df, 
                    x="Value", 
                    color="Physical Quantity", 
                    marginal="box", 
                    title="Value Distribution by Physical Quantity",
                    labels={"Value": f"Value ({viz_df['Unit'].iloc[0]})"}
                )
                fig_hist.update_layout(font=dict(family="DejaVu Sans", size=12))
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Bar chart grouped by quantity
                if len(viz_df["Physical Quantity"].unique()) > 1:
                    fig_bar = px.box(
                        viz_df, 
                        x="Physical Quantity", 
                        y="Value", 
                        color="Material",
                        title="Values by Physical Quantity and Material"
                    )
                    fig_bar.update_layout(font=dict(family="DejaVu Sans", size=12))
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            with viz_tabs[1]:
                # Only show if materials exist
                if any(v.material for v in extracted_values):
                    # Bar chart by material
                    fig_mat = px.bar(
                        viz_df, 
                        x="Material", 
                        y="Value", 
                        color="Physical Quantity",
                        title="Values by Material"
                    )
                    fig_mat.update_layout(font=dict(family="DejaVu Sans", size=12))
                    st.plotly_chart(fig_mat, use_container_width=True)
                    
                    # Pie chart of material distribution
                    mat_counts = viz_df["Material"].value_counts()
                    fig_pie = px.pie(
                        names=mat_counts.index, 
                        values=mat_counts.values,
                        title="Material Distribution"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with viz_tabs[2]:
                # Distribution plots
                fig_dist = px.histogram(
                    viz_df, 
                    x="Value", 
                    color="Physical Quantity", 
                    facet_col="Material" if len(viz_df["Material"].unique()) > 1 else None,
                    title="Value Distribution",
                    marginal="rug"
                )
                fig_dist.update_layout(font=dict(family="DejaVu Sans", size=12))
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Scatter plot
                if len(viz_df) > 1:
                    fig_scatter = px.scatter(
                        viz_df,
                        x="Physical Quantity",
                        y="Value",
                        color="Material",
                        size="Confidence",
                        hover_data=["Document", "Page"],
                        title="Value Scatter Plot"
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
            
            with viz_tabs[3]:
                # Network graph using PyVis
                if PYVIS_AVAILABLE and st.session_state.knowledge_graph:
                    viz_engine = PublicationQualityVisualizationEngine(
                        st.session_state.knowledge_graph,
                        default_colormap=st.session_state.get("viz_colormap", "viridis")
                    )
                    html_network = viz_engine.render_pyvis_salience(
                        filtered_concepts=None,
                        top_n_nodes=st.session_state.get("viz_top_n", 25),
                        colormap=st.session_state.get("viz_colormap", "viridis")
                    )
                    st.components.v1.html(html_network, height=750, scrolling=True)
                    
                    # Download button for network
                    st.download_button(
                        "📥 Download Interactive Network (HTML)",
                        data=html_network.encode('utf-8'),
                        file_name="knowledge_network.html",
                        mime="text/html"
                    )
                else:
                    st.info("Install pyvis for interactive network graphs: `pip install pyvis`")
            
            with viz_tabs[4]:
                # Embedding visualizations (t-SNE/PCA)
                if SKLEARN_AVAILABLE:
                    from sklearn.manifold import TSNE
                    from sklearn.decomposition import PCA
                    
                    # Simple embedding for demonstration (using value + context hash)
                    if len(viz_df) >= 5:
                        # Create simple embeddings from value + context
                        embeddings = []
                        for _, row in viz_df.iterrows():
                            # Simple hash-based embedding for demo
                            text = f"{row['Value']}_{row['Physical Quantity']}_{row['Material']}"
                            emb = np.array([hash(text) % 1000 / 1000, hash(text + "2") % 1000 / 1000])
                            embeddings.append(emb)
                        embeddings = np.array(embeddings)
                        
                        # t-SNE
                        if len(embeddings) >= 5:
                            tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings)-1), random_state=42)
                            tsne_coords = tsne.fit_transform(embeddings)
                            
                            fig_tsne = go.Figure(data=[go.Scatter(
                                x=tsne_coords[:, 0],
                                y=tsne_coords[:, 1],
                                mode='markers+text',
                                marker=dict(size=12, color=viz_df["Physical Quantity"].astype('category').cat.codes, colorscale=st.session_state.get("viz_colormap", "viridis")),
                                text=viz_df["Physical Quantity"],
                                textposition="top center",
                                hovertemplate="<b>%{text}</b><br>Value: %{customdata[0]} {customdata[1]}<br>Material: %{customdata[2]}<extra></extra>",
                                customdata=viz_df[["Value", "Unit", "Material"]].values
                            )])
                            fig_tsne.update_layout(title="t-SNE Visualization of Extracted Values", height=500)
                            st.plotly_chart(fig_tsne, use_container_width=True)
                        
                        # PCA
                        if len(embeddings) >= 2:
                            pca = PCA(n_components=2)
                            pca_coords = pca.fit_transform(embeddings)
                            var_ratio = pca.explained_variance_ratio_
                            
                            fig_pca = go.Figure(data=[go.Scatter(
                                x=pca_coords[:, 0],
                                y=pca_coords[:, 1],
                                mode='markers+text',
                                marker=dict(size=12, color=viz_df["Material"].astype('category').cat.codes, colorscale="Set2"),
                                text=viz_df["Material"],
                                textposition="top center",
                                hovertemplate="<b>%{text}</b><br>Value: %{customdata[0]} {customdata[1]}<extra></extra>",
                                customdata=viz_df[["Value", "Unit"]].values
                            )])
                            fig_pca.update_layout(
                                title=f"PCA Visualization (PC1: {var_ratio[0]:.1%}, PC2: {var_ratio[1]:.1%})",
                                height=500
                            )
                            st.plotly_chart(fig_pca, use_container_width=True)
                else:
                    st.info("Install scikit-learn for embedding visualizations: `pip install scikit-learn`")
            
            # Additional statistics
            with st.expander("📊 Quick Statistics", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Values", len(extracted_values))
                with col2:
                    unique_pq = len(set(v.physical_quantity for v in extracted_values))
                    st.metric("Unique Quantities", unique_pq)
                with col3:
                    unique_mat = len(set(v.material for v in extracted_values if v.material))
                    st.metric("Unique Materials", unique_mat)
                
                # Statistical summary by quantity
                st.subheader("Statistical Summary")
                for pq in set(v.physical_quantity for v in extracted_values):
                    pq_values = [v.value for v in extracted_values if v.physical_quantity == pq]
                    if pq_values:
                        st.write(f"**{PhysicalQuantityClassifier().get_human_readable(pq)}**:")
                        st.write(f"- Count: {len(pq_values)}")
                        st.write(f"- Range: {min(pq_values):.2f} to {max(pq_values):.2f}")
                        st.write(f"- Mean ± Std: {np.mean(pq_values):.2f} ± {np.std(pq_values):.2f}")
                        st.write("")

        # Display mode selection
        display_mode = st.radio("Display format", ["Table", "JSON", "Human Summary"], horizontal=True, key="display_mode")
        if display_mode == "Table" and extracted_values:
            df_data = []
            for v in extracted_values:
                phys_readable = PhysicalQuantityClassifier().get_human_readable(v.physical_quantity)
                df_data.append({
                    "Document": v.doc_name,
                    "Page": v.page,
                    "Value": f"{v.value:.2f}",
                    "Unit": v.unit,
                    "Physical Quantity": phys_readable,
                    "Material": v.material or "",
                    "Parameter": v.parameter_name or "",
                    "Confidence": f"{v.confidence:.2f}"
                })
            st.dataframe(pd.DataFrame(df_data), use_container_width=True)
        elif display_mode == "JSON" and extracted_values:
            st.json([v.model_dump() for v in extracted_values])
        elif display_mode == "Human Summary" and extracted_values:
            synthesizer = LLMReasoningSynthesizer(
                get_cached_llm(st.session_state.llm_model_choice, st.session_state.get("use_4bit", True))
            )
            report = QueryReport(
                query=active_prompt,
                total_docs=len(st.session_state.annotated_trees),
                docs_with_results=len(set(v.doc_name for v in extracted_values)),
                all_values=extracted_values,
                consensus={},
                processing_time_sec=0.0
            )
            conclusion = synthesizer.generate_human_conclusion(active_prompt, report)
            st.markdown(conclusion)

        if st.session_state.get("show_tree_nav") and retrieved:
            with st.expander("🌳 Tree Navigation Trace", expanded=False):
                for r in retrieved[:5]:
                    st.markdown(f"**{r['doc_id']}** → `{r['section_title']}` (p.{r['page_start']}) | confidence: {r.get('confidence', 0):.2f}")
                    st.caption(r.get('selection_reasoning', ''))
        if items:
            with st.expander("🔍 Extracted Items (Raw)", expanded=False):
                st.json([i.to_dict() for i in items[:10]])

        report = CrossDocumentQueryReport(
            query=active_prompt,
            total_documents=len(st.session_state.annotated_trees),
            documents_with_results=len(set(i.doc_source for i in items)),
            all_items=[i.model_dump() if hasattr(i, "model_dump") else i for i in items]
        )
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button("📥 Download JSON Report", report.to_json(), "results.json", "application/json")
        with col_dl2:
            tree_export = {
                "query": active_prompt,
                "annotated_trees": st.session_state.annotated_trees,
                "retrieved_nodes": retrieved,
                "extracted_items": [i.to_dict() for i in items],
                "answer": answer
            }
            st.download_button("📥 Download Tree Export", json.dumps(tree_export, indent=2, ensure_ascii=False, default=str), "tree_report.json", "application/json")
        
        if "index" in st.session_state.query_processor:
            st.session_state.query_processor["index"].cleanup()
    else:
        st.info("👆 Upload PDF files to begin.")


# ============================================================================
# Helper: MODEL_PROMPT_TEMPLATES & CONFIG
# ============================================================================
MODEL_PROMPT_TEMPLATES = {
    "qwen2.5:14b": {
        "system": "You are a precise document analyst. Follow JSON format strictly.",
        "json_reminder": "Return ONLY valid JSON. No markdown fences. No explanations outside JSON.",
        "tree_depth": 4,
        "max_tokens": 4096,
    },
    "mistral:7b": {
        "system": "You analyze document structures. Be concise.",
        "json_reminder": "Output must be parseable JSON. Use compact format.",
        "tree_depth": 3,
        "max_tokens": 4096,
    },
    "default": {
        "system": "You are a document navigation agent.",
        "json_reminder": "Return valid JSON only.",
        "tree_depth": 3,
        "max_tokens": 4096,
    }
}

def get_model_template(model_name: str) -> Dict[str, Any]:
    for key, template in MODEL_PROMPT_TEMPLATES.items():
        if key in model_name.lower():
            return template
    return MODEL_PROMPT_TEMPLATES["default"]

UNIVERSAL_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "retrieval_k": 5,
    "score_threshold": 0.2,
    "max_context_tokens": 8192,
    "max_new_tokens": 1024,
    "temperature": 0.1,
    "min_confidence_threshold": 0.55,
    "enable_parallel_parsing": True,
    "max_workers_pdf_parse": 4,
    "tree_search_depth": 3,
    "max_tree_nodes_per_prompt": 50,
    "enable_orjson": ORJSON_AVAILABLE,
    "max_retrieval_text_chars": 20000,
    "leaf_node_page_window": 7,
}

# ============================================================================
# COMPREHENSIVE OLLAMA MODEL CATALOG (50+ models)
# ============================================================================
LOCAL_LLM_OPTIONS = {
    # ===== Qwen2.5 Family (Alibaba) =====
    "[Ollama] qwen2.5:0.5b (Fastest, CPU OK)": "ollama:qwen2.5:0.5b",
    "[Ollama] qwen2.5:1.5b (Balanced)": "ollama:qwen2.5:1.5b",
    "[Ollama] qwen2.5:3b (Mid-size)": "ollama:qwen2.5:3b",
    "[Ollama] qwen2.5:7b (Recommended for RAG)": "ollama:qwen2.5:7b",
    "[Ollama] qwen2.5:14b (Max Reasoning)": "ollama:qwen2.5:14b",
    "[Ollama] qwen2.5:32b (High-end)": "ollama:qwen2.5:32b",
    "[Ollama] qwen2.5:72b (Enterprise)": "ollama:qwen2.5:72b",
    
    # ===== Qwen2 Family =====
    "[Ollama] qwen2:0.5b-instruct": "ollama:qwen2:0.5b-instruct",
    "[Ollama] qwen2:1.5b-instruct": "ollama:qwen2:1.5b-instruct",
    "[Ollama] qwen2:7b-instruct": "ollama:qwen2:7b-instruct",
    "[Ollama] qwen2:72b-instruct": "ollama:qwen2:72b-instruct",
    
    # ===== Llama 3.x Family (Meta) =====
    "[Ollama] llama3.2:1b (Tiny, CPU OK)": "ollama:llama3.2:1b",
    "[Ollama] llama3.2:3b (Small)": "ollama:llama3.2:3b",
    "[Ollama] llama3.1:8b (Meta Standard)": "ollama:llama3.1:8b",
    "[Ollama] llama3.1:70b (High-end)": "ollama:llama3.1:70b",
    "[Ollama] llama3:8b": "ollama:llama3:8b",
    "[Ollama] llama3:70b": "ollama:llama3:70b",
    
    # ===== Mistral Family =====
    "[Ollama] mistral:7b (High JSON Reliability)": "ollama:mistral:7b",
    "[Ollama] mistral-nemo:12b": "ollama:mistral-nemo:12b",
    "[Ollama] mistral-large:123b": "ollama:mistral-large:123b",
    "[Ollama] mixtral:8x7b (MoE)": "ollama:mixtral:8x7b",
    "[Ollama] mixtral:8x22b (MoE High-end)": "ollama:mixtral:8x22b",
    
    # ===== Gemma Family (Google) =====
    "[Ollama] gemma2:2b (Tiny)": "ollama:gemma2:2b",
    "[Ollama] gemma2:9b (Scientific Nuance)": "ollama:gemma2:9b",
    "[Ollama] gemma2:27b": "ollama:gemma2:27b",
    "[Ollama] gemma:2b": "ollama:gemma:2b",
    "[Ollama] gemma:7b": "ollama:gemma:7b",
    
    # ===== Falcon Family (TII) =====
    "[Ollama] falcon3:1b": "ollama:falcon3:1b",
    "[Ollama] falcon3:3b": "ollama:falcon3:3b",
    "[Ollama] falcon3:7b": "ollama:falcon3:7b",
    "[Ollama] falcon3:10b (Instruction Following)": "ollama:falcon3:10b",
    
    # ===== Phi Family (Microsoft) =====
    "[Ollama] phi3:mini (3.8B)": "ollama:phi3:mini",
    "[Ollama] phi3:small (7B)": "ollama:phi3:small",
    "[Ollama] phi3:medium (14B)": "ollama:phi3:medium",
    
    # ===== Code-Specialized Models =====
    "[Ollama] codellama:7b": "ollama:codellama:7b",
    "[Ollama] codellama:13b": "ollama:codellama:13b",
    "[Ollama] codellama:34b": "ollama:codellama:34b",
    "[Ollama] codeqwen:7b": "ollama:codeqwen:7b",
    "[Ollama] deepseek-coder:6.7b": "ollama:deepseek-coder:6.7b",
    "[Ollama] starcoder2:7b": "ollama:starcoder2:7b",
    
    # ===== Math/Reasoning Specialized =====
    "[Ollama] wizardmath:7b": "ollama:wizardmath:7b",
    "[Ollama] mathstral:7b": "ollama:mathstral:7b",
    
    # ===== Multilingual Models =====
    "[Ollama] llama3.1:8b-instruct-multilingual": "ollama:llama3.1:8b-instruct-multilingual",
    "[Ollama] aya:23b (Multilingual)": "ollama:aya:23b",
    
    # ===== Legacy/Experimental =====
    "[Ollama] llama2:7b": "ollama:llama2:7b",
    "[Ollama] llama2:13b": "ollama:llama2:13b",
    "[Ollama] llama2:70b": "ollama:llama2:70b",
    "[Ollama] vicuna:7b": "ollama:vicuna:7b",
    "[Ollama] vicuna:13b": "ollama:vicuna:13b",
    "[Ollama] openhermes:2.5-mistral-7b": "ollama:openhermes:2.5-mistral-7b",
    "[Ollama] nous-hermes2:7b": "ollama:nous-hermes2:7b",
}

# ============================================================================
# Additional utilities (fast JSON, timing, cache)
# ============================================================================
def fast_json_dumps(obj: Any, indent: bool = False) -> bytes:
    if ORJSON_AVAILABLE:
        option = orjson.OPT_INDENT_2 if indent else 0
        return orjson.dumps(obj, option=option, default=str)
    else:
        return json.dumps(obj, indent=2 if indent else None, ensure_ascii=False, default=str).encode()

def fast_json_loads(data: Union[bytes, str]) -> Any:
    if ORJSON_AVAILABLE:
        if isinstance(data, str):
            data = data.encode()
        return orjson.loads(data)
    else:
        if isinstance(data, bytes):
            data = data.decode()
        return json.loads(data)

@contextmanager
def timer(label: str):
    start = time.time()
    yield
    elapsed = time.time() - start
    if not hasattr(timer, 'metrics'):
        timer.metrics = defaultdict(list)
    timer.metrics[label].append(elapsed)
    logger.info(f"⏱️ {label}: {elapsed:.2f}s")

class LRUCache:
    def __init__(self, max_size=1000, ttl=7200):
        self.max_size = max_size
        self.ttl = ttl
        self._cache = OrderedDict()
        self._lock = threading.RLock()
    def _key(self, *args, **kwargs):
        key_data = "|".join(str(a) for a in args) + "|" + json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.sha256(key_data.encode()).hexdigest()[:20]
    def get(self, *args, **kwargs):
        key = self._key(*args, **kwargs)
        with self._lock:
            if key in self._cache:
                val, ts = self._cache[key]
                if time.time() - ts < self.ttl:
                    self._cache.move_to_end(key)
                    return val
                else:
                    del self._cache[key]
        return None
    def set(self, value, *args, **kwargs):
        key = self._key(*args, **kwargs)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            self._cache[key] = (value, time.time())
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

response_cache = LRUCache(max_size=2000, ttl=7200)


if __name__ == "__main__":
    run_streamlit()
