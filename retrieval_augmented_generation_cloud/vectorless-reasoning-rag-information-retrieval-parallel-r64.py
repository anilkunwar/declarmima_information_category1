#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DECLARMIMA v13.1 - ENHANCED PHYSICAL QUANTITY & MATERIAL EXTRACTION
                       + PUBLICATION-QUALITY VISUALIZATIONS
=======================================================================
- Full support for alloy/material name extraction
- Yield strength, tensile strength classification
- Two-stage retrieval (semantic search + page reading)
- Structured metadata ingestion
- Pagination-aware page fetching
- Cross-document aggregation by physical quantity AND material
- NEW: Publication-quality visualization engine (matplotlib, plotly, networkx)
- NEW: Interactive dashboards – histograms, knowledge graphs, sunbursts, consensus charts
- NEW: Customizable colormaps, font sizes, layout algorithms
- NEW: Export plots as PNG, SVG, HTML, CSV
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

# NEW visualization imports
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logging.basicConfig(level=logging.INFO, handlers=[console_handler], force=True)
logger = logging.getLogger("DECLARMIMA")

# ============================================================================
# OPTIONAL IMPORTS (keep original)
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

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Two-stage retrieval will use fallback TF-IDF.")

# ============================================================================
# 1. PYDANTIC MODELS (unchanged from v13.0)
# ============================================================================
from pydantic import BaseModel, Field, field_validator

class UniversalExtractionItem(BaseModel):
    item_type: Literal["quantitative", "qualitative", "definition", "comparison", "relationship", "process", "material", "method"]
    content: str
    parameter_name: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    physical_quantity: Optional[str] = None
    material: Optional[str] = None
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
    material: Optional[str] = None
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
    doc_name: str
    alloys: List[str] = []
    laser_power_values: List[float] = []
    scan_speed_values: List[float] = []
    yield_strength_values: List[float] = []
    tensile_strength_values: List[float] = []
    hardness_values: List[float] = []
    temperature_values: List[float] = []
    energy_density_values: List[float] = []
    process_types: List[str] = []
    other_parameters: Dict[str, List[float]] = {}

# ============================================================================
# 2. PHYSICAL QUANTITY CLASSIFIER (unchanged)
# ============================================================================
class PhysicalQuantityClassifier:
    CANONICAL = {
        "laser_power": ["laser power", "laser beam power", "laser output power"],
        "electrical_power": ["electrical power", "power supply"],
        "scan_speed": ["scan speed", "scanning speed", "laser scan speed"],
        "flow_speed": ["flow speed", "flow velocity"],
        "irradiance": ["irradiance", "laser irradiance", "intensity", "power density"],
        "temperature": ["temperature", "melting temperature", "annealing temperature"],
        "energy_density": ["energy density", "volumetric energy density", "VED"],
        "layer_thickness": ["layer thickness", "powder layer thickness"],
        "spot_size": ["spot size", "beam diameter"],
        "exposure_time": ["exposure time", "dwell time"],
        "yield_strength": ["yield strength", "ys", "0.2% offset strength", "proof stress"],
        "tensile_strength": ["tensile strength", "uts", "ultimate tensile strength"],
        "hardness": ["hardness", "vickers hardness", "microhardness", "hv"],
        "elongation": ["elongation", "strain", "ductility"],
        "modulus": ["young's modulus", "elastic modulus", "stiffness"],
    }
    UNIT_HINTS = {
        "scan_speed": ["mm/s", "cm/s", "m/s"],
        "laser_power": ["w", "kw", "mw"],
        "irradiance": ["w/cm²", "kw/cm²"],
        "temperature": ["°c", "k"],
        "energy_density": ["j/mm³", "j/m³"],
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
            "laser_power": "Laser Power", "scan_speed": "Scan Speed", "yield_strength": "Yield Strength",
            "tensile_strength": "Tensile Strength", "hardness": "Hardness", "temperature": "Temperature",
            "energy_density": "Energy Density", "unknown": "Other Quantities"
        }
        return mapping.get(canonical, canonical.replace("_", " ").title())

# ============================================================================
# 3. PAGINATION-AWARE PDF READER (unchanged)
# ============================================================================
class PaginationAwareReader:
    def __init__(self, max_chars_per_request=20000):
        self.max_chars_per_request = max_chars_per_request
    def extract_pages(self, doc_path: str, page_numbers: List[int]) -> Dict[int, str]:
        doc = fitz.open(doc_path)
        result = {}
        for pnum in page_numbers:
            if pnum < 1 or pnum > len(doc):
                continue
            page = doc[pnum-1]
            text = page.get_text("text")
            if len(text) > self.max_chars_per_request:
                text = text[:self.max_chars_per_request] + "\n...[TRUNCATED]"
            result[pnum] = text
        doc.close()
        return result
    def extract_page_range(self, doc_path: str, start: int, end: int, step=1) -> Dict[int, str]:
        pages = list(range(start, end+1, step))
        return self.extract_pages(doc_path, pages)

# ============================================================================
# 4. STRUCTURED METADATA EXTRACTOR (unchanged)
# ============================================================================
class StructuredMetadataExtractor:
    ALLOY_PATTERNS = [
        r'\b(?:AlSi[\dMg]+|Ti\d*Al\d*V\d*|Inconel\s?\d{3}|SS\s?\d{4}|UNS\s?S\d{5})\b',
        r'\b(?:Al-[\d]+Si-[\d]+Mg|AlSiMg[\d\.]+Zr|TiB[2]?|CoCr[\w]+|NiTi)\b',
        r'(\w+(?:-\w+)?\s?(?:alloy|superalloy|metal|composite))'
    ]
    POWER_PATTERN = r'(?:laser\s+power|power|P)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(W|kW|mW)'
    SCAN_SPEED_PATTERN = r'(?:scan\s+speed|scanning\s+speed|v_scan)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(mm/s|cm/s|m/s)'
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
        meta = DocumentMetadata(doc_name=doc_name)
        alloys_set = set()
        for regex in self.alloy_regexes:
            for match in regex.finditer(full_text):
                candidate = match.group(0).strip()
                if len(candidate) > 2 and candidate.lower() not in ["alloy", "composite", "metal"]:
                    alloys_set.add(candidate)
        meta.alloys = list(alloys_set)
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
        process_keywords = {
            "SLM": ["selective laser melting", "slm"],
            "LPBF": ["laser powder bed fusion", "l-pbf", "lpbf"],
            "LSA": ["laser surface alloying", "lsa"],
        }
        processes = []
        for proc, keywords in process_keywords.items():
            if any(kw in full_text.lower() for kw in keywords):
                processes.append(proc)
        meta.process_types = processes
        return meta

# ============================================================================
# 5. TWO-STAGE RETRIEVER (unchanged)
# ============================================================================
class TwoStageRetriever:
    def __init__(self, llm: Optional['HybridLLM'] = None, embedding_model: str = "all-MiniLM-L6-v2"):
        self.llm = llm
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
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
        if self.embedding_model is not None and len(self.doc_summaries) > 0:
            doc_texts = [f"{meta.alloys} {meta.process_types} {self.doc_summaries.get(name, '')}" 
                         for name, meta in self.doc_metadata.items()]
            if doc_texts:
                doc_emb = self.embedding_model.encode(doc_texts, convert_to_tensor=True)
                query_emb = self.embedding_model.encode(query, convert_to_tensor=True)
                scores = util.cos_sim(query_emb, doc_emb)[0]
                scored = [(list(self.doc_metadata.keys())[i], float(scores[i])) for i in range(len(doc_texts))]
                scored.sort(key=lambda x: x[1], reverse=True)
                return scored[:top_k]
        scores = []
        query_lower = query.lower()
        for name, meta in self.doc_metadata.items():
            score = 0.0
            for alloy in meta.alloys:
                if alloy.lower() in query_lower:
                    score += 0.3
            if any(str(p) in query_lower for p in meta.laser_power_values):
                score += 0.2
            for proc in meta.process_types:
                if proc.lower() in query_lower:
                    score += 0.2
            scores.append((name, min(score, 1.0)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    def get_relevant_pages(self, doc_name: str, query: str, max_pages: int = 5) -> List[int]:
        return list(range(1, max_pages+1))

# ============================================================================
# 6. HIERARCHICAL PDF INDEX (unchanged, keep all)
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
    metadata: Optional[DocumentMetadata] = None
    def get_text(self, doc_cache: Dict[str, Any] = None, max_chars: int = 20000) -> str:
        if self.full_text:
            return self.full_text[:max_chars]
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
        return self.full_text[:max_chars]
    def to_dict(self):
        return {
            "id": self.id, "title": self.title, "page_start": self.page_start,
            "page_end": self.page_end, "summary": self.summary,
            "prefix_summary": self.prefix_summary, "level": self.level,
            "doc_id": self.doc_id, "section_type": self.section_type,
            "node_id": self.node_id, "text_token_count": self.text_token_count,
            "children": [c.to_dict() for c in self.children],
            "metadata": self.metadata.dict() if self.metadata else None
        }
    def to_tree_format(self, max_chars: int = 20000) -> Dict[str, Any]:
        result = {
            "title": self.title, "node_id": self.node_id, "start_index": self.page_start,
            "end_index": self.page_end or self.page_start, "summary": self.summary,
            "prefix_summary": self.prefix_summary, "text_token_count": self.text_token_count,
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
            data["id"], data["title"], data["page_start"], data.get("page_end"),
            "", data.get("summary", ""), data.get("level", 0),
            doc_id=data.get("doc_id", ""), section_type=data.get("section_type", "BODY"),
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
                    f"{doc_id}_toc_{level}_{title[:20]}", title.strip(), page, end,
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
            tree = self._build_tree_from_toc(doc_name, pages, toc)
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
    def _build_tree_from_toc(self, doc_name: str, pages: List[Dict], toc: Dict) -> PageNode:
        safe_title = toc.get("suggested_root_title") or doc_name
        root = PageNode(
            f"{doc_name}_root", safe_title, 1, len(pages), "",
            f"Document {doc_name}", 0, doc_id=doc_name, node_id="0000"
        )
        entries = toc.get("toc_entries", []) or toc.get("headings_detected", [])
        window = UNIVERSAL_CONFIG.get("leaf_node_page_window", 7)
        if entries:
            nodes_by_level = {}
            for entry in entries:
                level_val = entry.get("level")
                level = int(level_val) if level_val is not None else 1
                title = str(entry.get("title", "Unknown")).strip()
                page_raw = entry.get("page")
                page = int(page_raw) if page_raw is not None else 1
                if page < 1 or page > len(pages):
                    continue
                end = min(page + window, len(pages))
                text_parts = []
                for i in range(page, min(end + 1, len(pages) + 1)):
                    try:
                        page_data = pages[i - 1]
                        if isinstance(page_data, dict) and 'text' in page_data:
                            text_parts.append(page_data['text'])
                    except (IndexError, KeyError, TypeError):
                        continue
                text = "\n\n".join(text_parts)
                node = PageNode(
                    f"{doc_name}_toc_{level}_{title[:20]}", title, page, end,
                    text, text[:200], level, doc_id=doc_name
                )
                nodes_by_level.setdefault(level, []).append(node)
            for level in sorted(nodes_by_level.keys()):
                for node in nodes_by_level[level]:
                    parent = self._find_parent(root, level - 1, node.page_start)
                    parent.children.append(node)
        else:
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
                    f"{doc_name}_p{page_num}", f"Page {page_num}", page_num, page_num,
                    text, str(text)[:200], 3, doc_id=doc_name
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
# 7. HYBRID LLM CLIENT (unchanged)
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
# 8. QUANTITATIVE KNOWLEDGE GRAPH (unchanged)
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
            "materials": defaultdict(list),
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
        mat_dict = {}
        for doc_id, graph in self.doc_graphs.items():
            materials = set()
            for item in graph["all_items"]:
                if item.get("material"):
                    materials.add(item["material"])
            mat_dict[doc_id] = list(materials)
        return mat_dict
    def get_material_summary_stats(self, material_name: str) -> Dict[str, Any]:
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
                    material=item.get("material"),
                    confidence=item.get("confidence", 0.7),
                    context=item.get("context", "")[:300],
                    doc_name=doc_id,
                    page=item.get("page", 1),
                    section_title=item.get("section_title")
                ))
        return all_values

# ============================================================================
# 9. UNIVERSAL LLM EXTRACTOR (unchanged)
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
1. Distinguish physically different quantities even if they share units.
2. For mechanical properties: "yield strength" -> physical_quantity = "yield_strength", "tensile strength" -> "tensile_strength".
3. NEVER truncate numbers: if text says "1000 W", output 1000.
4. If an alloy or material name appears, create an item with item_type="material", content=the name, material=the name.
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
            if qa.get("query_type") == "quantitative" and not re.search(r'\d+', text):
                continue
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
                        if "physical_quantity" not in item_data or not item_data["physical_quantity"]:
                            item_data["physical_quantity"] = self.phys_classifier.classify(
                                item_data.get("parameter_name"),
                                item_data.get("unit"),
                                item_data.get("context", "")
                            )
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
# 10. LLM REASONING SYNTHESIZER (unchanged)
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
        if by_material:
            lines.append("### By Material/Alloy")
            for mat, vals in sorted(by_material.items()):
                lines.append(f"#### {mat} ({len(vals)} values)")
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
# 11. HIERARCHICAL TREE RETRIEVER (unchanged)
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
# 12. NEW: PUBLICATION VISUALIZATION ENGINE (adapted from second code)
# ============================================================================
class PublicationVisualizationEngine:
    """Publication-quality visualizations from the QuantitativeKnowledgeGraph."""
    DOMAIN_COLORS = {
        "laser_power": "#3b82f6", "scan_speed": "#8b5cf6", "yield_strength": "#f59e0b",
        "tensile_strength": "#10b981", "hardness": "#ec4899", "temperature": "#ef4444",
        "energy_density": "#06b6d4", "unknown": "#6b7280"
    }
    COLORMAP_OPTIONS = {
        "viridis": "viridis", "plasma": "plasma", "inferno": "inferno", "magma": "magma", "cividis": "cividis",
        "Blues": "Blues", "Greens": "Greens", "Reds": "Reds", "OrRd": "OrRd", "PuBu": "PuBu",
        "RdYlBu": "RdYlBu", "Spectral": "Spectral", "coolwarm": "coolwarm", "Set1": "Set1", "Set2": "Set2"
    }
    def __init__(self, kgraph: QuantitativeKnowledgeGraph, config: Dict[str, Any] = None):
        self.kgraph = kgraph
        self.config = config or {
            "font_family": "DejaVu Sans",
            "font_size": 10,
            "title_font_size": 14,
            "label_font_size": 9,
            "default_colormap": "viridis",
            "figure_dpi": 300,
        }
        plt.rcParams['font.family'] = self.config["font_family"]
        plt.rcParams['font.size'] = self.config["font_size"]
        plt.rcParams['axes.titlesize'] = self.config["title_font_size"]
        plt.rcParams['axes.labelsize'] = self.config["label_font_size"]
        plt.rcParams['figure.dpi'] = self.config["figure_dpi"]
    def _get_dataframe(self) -> pd.DataFrame:
        rows = []
        for doc_id, graph in self.kgraph.doc_graphs.items():
            for item in graph["all_items"]:
                phys = item.get("physical_quantity", "unknown")
                mat = item.get("material", "Unknown")
                val = item.get("value")
                unit = item.get("unit", "")
                if val is not None:
                    rows.append({
                        "doc": doc_id,
                        "doc_stem": doc_id.split("/")[-1].replace(".pdf", ""),
                        "physical_quantity": phys,
                        "material": mat,
                        "value": val,
                        "unit": unit,
                        "confidence": item.get("confidence", 0.5),
                        "page": item.get("page", 0)
                    })
        return pd.DataFrame(rows)
    def plot_histogram(self, quantity: Optional[str] = None, group_by: str = "material") -> go.Figure:
        df = self._get_dataframe()
        if df.empty:
            return go.Figure()
        if quantity:
            df = df[df["physical_quantity"] == quantity]
        if df.empty:
            return go.Figure()
        fig = go.Figure()
        groups = sorted(df[group_by].unique())
        for grp in groups:
            subset = df[df[group_by] == grp]
            mean_val = subset["value"].mean()
            std_val = subset["value"].std()
            fig.add_trace(go.Bar(
                name=str(grp), x=[grp], y=[mean_val],
                error_y=dict(type='data', array=[std_val] if len(subset)>1 else [0]),
                text=[f"n={len(subset)}<br>μ={mean_val:.2f}<br>σ={std_val:.2f}"],
                textposition="outside"
            ))
        yunit = df["unit"].iloc[0] if not df.empty else ""
        title = f"{quantity if quantity else 'All Quantities'} by {group_by}"
        fig.update_layout(title=title, yaxis_title=yunit, barmode='group')
        return fig
    def plot_consensus_waterfall(self, top_n: int = 10) -> go.Figure:
        df = self._get_dataframe()
        if df.empty:
            return go.Figure()
        grouped = df.groupby(["physical_quantity", "material"])["value"].agg(["mean", "std", "count"]).reset_index()
        grouped = grouped.sort_values("count", ascending=False).head(top_n)
        labels = grouped["physical_quantity"] + " (" + grouped["material"] + ")"
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=grouped["mean"], error_y=dict(type='data', array=grouped["std"]),
            marker_color="#059669", text=[f"n={c}" for c in grouped["count"]]
        ))
        fig.update_layout(title="Cross‑Document Consensus (mean ± std)", yaxis_title="Value")
        return fig
    def plot_knowledge_graph(self, quantity: Optional[str] = None) -> plt.Figure:
        df = self._get_dataframe()
        if df.empty:
            return plt.figure()
        if quantity:
            df = df[df["physical_quantity"] == quantity]
        if df.empty:
            return plt.figure()
        G = nx.Graph()
        hub = f"{quantity if quantity else 'kg'}_hub"
        G.add_node(hub, node_type="hub", color="#dc2626")
        for mat in df["material"].unique():
            G.add_node(mat, node_type="material", color="#3b82f6")
            G.add_edge(hub, mat, weight=len(df[df["material"] == mat]))
        for doc in df["doc_stem"].unique():
            G.add_node(doc, node_type="doc", color="#10b981")
            G.add_edge(hub, doc, weight=len(df[df["doc_stem"] == doc]))
        for _, row in df.nlargest(25, "value").iterrows():
            leaf = f"{row['value']:.1f} {row['unit']}"
            G.add_node(leaf, node_type="value", color="#f59e0b")
            G.add_edge(row["material"], leaf, weight=1)
            G.add_edge(row["doc_stem"], leaf, weight=1)
        pos = nx.spring_layout(G, k=0.6, seed=42)
        fig, ax = plt.subplots(figsize=(14,12))
        for ntype, color in [("hub","#dc2626"), ("material","#3b82f6"), ("doc","#10b981"), ("value","#f59e0b")]:
            nodes = [n for n,d in G.nodes(data=True) if d.get("node_type")==ntype]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, node_size=800 if ntype!="value" else 300, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.8, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
        ax.set_title(f"Quantitative Knowledge Graph – {quantity if quantity else 'All'}")
        ax.axis("off")
        return fig
    def plot_sunburst(self) -> go.Figure:
        df = self._get_dataframe()
        if df.empty:
            return go.Figure()
        df["count"] = 1
        fig = px.sunburst(df, path=["physical_quantity", "material"], values="count",
                          title="Material & Quantity Hierarchy", color="value", color_continuous_scale="Viridis")
        return fig
    def plot_contradiction_matrix(self, quantity: Optional[str] = None) -> go.Figure:
        df = self._get_dataframe()
        if df.empty:
            return go.Figure()
        if quantity:
            df = df[df["physical_quantity"] == quantity]
        if df.empty:
            return go.Figure()
        docs = df["doc_stem"].unique()
        if len(docs) < 2:
            return go.Figure()
        mat = np.zeros((len(docs), len(docs)))
        for i, d1 in enumerate(docs):
            v1 = df[df["doc_stem"] == d1]["value"].mean()
            for j, d2 in enumerate(docs):
                if i == j: continue
                v2 = df[df["doc_stem"] == d2]["value"].mean()
                if v2 != 0:
                    mat[i,j] = abs(v1 - v2) / v2
        fig = go.Figure(data=go.Heatmap(z=mat, x=docs, y=docs, colorscale="RdBu_r"))
        fig.update_layout(title=f"Contradiction Matrix for {quantity if quantity else 'All Quantities'}")
        return fig
    def plot_radar(self, quantity: Optional[str] = None) -> go.Figure:
        df = self._get_dataframe()
        if df.empty:
            return go.Figure()
        if quantity:
            df = df[df["physical_quantity"] == quantity]
        if df.empty:
            return go.Figure()
        categories = ["Mean", "Max", "Min", "Std", "Count"]
        fig = go.Figure()
        for mat, group in df.groupby("material"):
            values = [group["value"].mean(), group["value"].max(), group["value"].min(),
                      group["value"].std(), len(group)]
            values += values[:1]
            fig.add_trace(go.Scatterpolar(r=values, theta=categories + [categories[0]], fill='toself', name=mat))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title=f"Radar Chart – {quantity if quantity else 'All'}")
        return fig

# ============================================================================
# 13. STREAMLIT UI WITH VISUALIZATION DASHBOARD
# ============================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        model_keys = list(LOCAL_LLM_OPTIONS.keys())
        if "llm_model_choice" not in st.session_state:
            st.session_state.llm_model_choice = model_keys[3]
        selected = st.selectbox("🧠 Select Local LLM (Ollama)", options=model_keys, index=model_keys.index(st.session_state.llm_model_choice), key="llm_model_select")
        st.session_state.llm_model_choice = selected
        st.checkbox("🗜️ Use 4-bit quantization (if Transformers fallback)", value=True, key="use_4bit")
        max_chars = st.slider("📄 Max text length per retrieved section (characters)", min_value=1000, max_value=50000, value=20000, step=1000)
        st.session_state.max_retrieval_chars = max_chars
        st.slider("Confidence threshold", 0.3, 0.9, 0.55, 0.05, key="min_confidence")
        st.checkbox("Show reasoning trace", value=True, key="show_trace")
        st.checkbox("Show tree navigation", value=True, key="show_tree_nav")
        st.checkbox("Enable two-stage retrieval (semantic)", value=True, key="two_stage")
        st.markdown("### 🎨 Visualization Settings")
        st.selectbox("Colormap", list(PublicationVisualizationEngine.COLORMAP_OPTIONS.keys()), index=0, key="viz_cmap")
        st.slider("Figure DPI", 100, 600, 300, step=50, key="viz_dpi")
        st.caption(f"GPU: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

@st.cache_resource(show_spinner="Initializing LLM...")
def get_cached_llm(model_choice: str, use_4bit: bool):
    internal = LOCAL_LLM_OPTIONS[model_choice]
    return HybridLLM(model_key=internal, use_4bit=use_4bit)

def run_streamlit():
    st.set_page_config(page_title="DECLARMIMA v13.1 - Enhanced Material Extraction + Visualization", layout="wide")
    st.markdown("# 🔬 DECLARMIMA v13.1 - Enhanced Material & Property Extraction + Publication Visualizations")
    st.caption("Now with customizable, publication‑quality charts: histograms, knowledge graphs, sunbursts, contradictions, radar, and more.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "query_processor" not in st.session_state:
        st.session_state.query_processor = {}
    if "knowledge_graph" not in st.session_state:
        st.session_state.knowledge_graph = QuantitativeKnowledgeGraph()
    if "annotated_trees" not in st.session_state:
        st.session_state.annotated_trees = []
    if "cached_query_result" not in st.session_state:
        st.session_state.cached_query_result = None
    if "active_prompt" not in st.session_state:
        st.session_state.active_prompt = ""
    if "two_stage_retriever" not in st.session_state:
        st.session_state.two_stage_retriever = None

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
                initial_prompt = "Extract all quantitative parameters (laser power, scan speed, yield strength, tensile strength, hardness, energy density, etc.) with full numerical values, units, physical_quantity classification, and any alloy/material names."
                items = extractor.extract_from_chunks(leaf_texts, initial_prompt)
                all_items.extend(items)
                kg.add_extractions(doc_name, items)
                if tree.metadata:
                    kg.add_document_metadata(doc_name, tree.metadata)
                    two_stage.index_document(doc_name, tree.metadata, tree.summary)
                else:
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
            # Show summary of extracted quantities
            pq_counts = kg.get_all_physical_quantities()
            if pq_counts:
                st.write("**Detected Physical Quantities:**")
                for pq, count in sorted(pq_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                    st.write(f"- `{pq}`: {count} occurrences")
            mat_dict = kg.get_all_materials()
            if mat_dict:
                st.write("**Materials/Alloys per document:**")
                for doc, mats in mat_dict.items():
                    if mats:
                        st.write(f"- {doc}: {', '.join(mats)}")

    if st.session_state.annotated_trees:
        # Quick query buttons
        st.markdown("### ⚡ Quick Queries")
        cols = st.columns(4)
        quick = ["laser power", "yield strength", "scan speed", "alloy names"]
        for i, q in enumerate(quick):
            with cols[i]:
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
            if not (cached and cached.get("prompt") == active_prompt and "answer" in cached):
                run_query = True

        answer = None
        extracted_values = []
        items = []

        if run_query:
            with st.chat_message("assistant"):
                progress = st.progress(0)
                progress.text("Initializing LLM...")
                llm = get_cached_llm(st.session_state.llm_model_choice, st.session_state.get("use_4bit", True))
                progress.progress(0.1)
                if st.session_state.get("two_stage", True) and st.session_state.two_stage_retriever is not None:
                    progress.text("Stage 1: Semantic document filtering...")
                    relevant_docs = st.session_state.two_stage_retriever.retrieve_relevant_docs(active_prompt, top_k=8)
                    st.caption(f"Selected {len(relevant_docs)} relevant documents out of {len(st.session_state.annotated_trees)}.")
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

        # Display quantitative results in table/JSON/human summary
        if extracted_values:
            st.markdown("---")
            st.subheader("📊 Quantitative Results")
            display_mode = st.radio("Display format", ["Table", "JSON", "Human Summary"], horizontal=True, key="display_mode")
            if display_mode == "Table":
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
                st.dataframe(df_data, use_container_width=True)
            elif display_mode == "JSON":
                st.json([v.model_dump() for v in extracted_values])
            elif display_mode == "Human Summary":
                synthesizer = LLMReasoningSynthesizer(get_cached_llm(st.session_state.llm_model_choice, st.session_state.get("use_4bit", True)))
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

        # ========== NEW VISUALIZATION DASHBOARD ==========
        if st.session_state.knowledge_graph and len(st.session_state.knowledge_graph.doc_graphs) > 0:
            st.markdown("---")
            st.subheader("📈 Publication-Quality Visualization Dashboard")
            viz_config = {
                "font_family": "DejaVu Sans",
                "font_size": 10,
                "title_font_size": 14,
                "label_font_size": 9,
                "default_colormap": st.session_state.get("viz_cmap", "viridis"),
                "figure_dpi": st.session_state.get("viz_dpi", 300),
            }
            viz_engine = PublicationVisualizationEngine(st.session_state.knowledge_graph, viz_config)
            df_viz = viz_engine._get_dataframe()
            if not df_viz.empty:
                all_qties = sorted(df_viz["physical_quantity"].unique())
                selected_qty = st.selectbox("Filter by physical quantity (optional)", [""] + all_qties, index=0)
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Histogram / Consensus", "🕸️ Knowledge Graph", "☀️ Sunburst", "⚠️ Contradictions", "📡 Radar"])
                with tab1:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_hist = viz_engine.plot_histogram(quantity=selected_qty if selected_qty else None, group_by="material")
                        st.plotly_chart(fig_hist, use_container_width=True)
                    with col2:
                        fig_wf = viz_engine.plot_consensus_waterfall(top_n=10)
                        st.plotly_chart(fig_wf, use_container_width=True)
                with tab2:
                    if selected_qty:
                        fig_kg = viz_engine.plot_knowledge_graph(quantity=selected_qty)
                        st.pyplot(fig_kg)
                        buf = BytesIO()
                        fig_kg.savefig(buf, format="png", dpi=viz_config["figure_dpi"], bbox_inches="tight")
                        st.download_button("📥 Download KG as PNG", buf.getvalue(), f"{selected_qty}_kg.png")
                    else:
                        st.info("Select a specific quantity to display the knowledge graph.")
                with tab3:
                    fig_sun = viz_engine.plot_sunburst()
                    st.plotly_chart(fig_sun, use_container_width=True)
                with tab4:
                    fig_contra = viz_engine.plot_contradiction_matrix(quantity=selected_qty if selected_qty else None)
                    st.plotly_chart(fig_contra, use_container_width=True)
                with tab5:
                    fig_radar = viz_engine.plot_radar(quantity=selected_qty if selected_qty else None)
                    st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.info("No quantitative data available for visualizations. Run a query or re-index with better extraction.")

        # Download reports
        if items:
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
                    "retrieved_nodes": retrieved if 'retrieved' in locals() else [],
                    "extracted_items": [i.to_dict() for i in items],
                    "answer": answer
                }
                st.download_button("📥 Download Tree Export", json.dumps(tree_export, indent=2, ensure_ascii=False, default=str), "tree_report.json", "application/json")

        if "index" in st.session_state.query_processor:
            st.session_state.query_processor["index"].cleanup()
    else:
        st.info("👆 Upload PDF files to begin.")

# ============================================================================
# Helper: MODEL_PROMPT_TEMPLATES (unchanged)
# ============================================================================
MODEL_PROMPT_TEMPLATES = {
    "qwen2.5:14b": {"system": "You are a precise document analyst. Follow JSON format strictly.", "json_reminder": "Return ONLY valid JSON."},
    "mistral:7b": {"system": "You analyze document structures. Be concise.", "json_reminder": "Output must be parseable JSON."},
    "default": {"system": "You are a document navigation agent.", "json_reminder": "Return valid JSON only."}
}
def get_model_template(model_name: str) -> Dict[str, Any]:
    for key, template in MODEL_PROMPT_TEMPLATES.items():
        if key in model_name.lower():
            return template
    return MODEL_PROMPT_TEMPLATES["default"]

UNIVERSAL_CONFIG = {
    "chunk_size": 1000, "chunk_overlap": 200, "retrieval_k": 5, "score_threshold": 0.2,
    "max_context_tokens": 8192, "max_new_tokens": 1024, "temperature": 0.1,
    "min_confidence_threshold": 0.55, "enable_parallel_parsing": True, "max_workers_pdf_parse": 4,
    "tree_search_depth": 3, "max_tree_nodes_per_prompt": 50, "enable_orjson": ORJSON_AVAILABLE,
    "max_retrieval_text_chars": 20000, "leaf_node_page_window": 7,
}
LOCAL_LLM_OPTIONS = {
    "[Ollama] qwen2.5:0.5b (Fastest, CPU OK)": "ollama:qwen2.5:0.5b",
    "[Ollama] qwen2.5:1.5b (Balanced)": "ollama:qwen2.5:1.5b",
    "[Ollama] qwen2.5:7b (Recommended for RAG)": "ollama:qwen2.5:7b",
    "[Ollama] qwen2.5:14b (Max Reasoning)": "ollama:qwen2.5:14b",
    "[Ollama] llama3.1:8b (Meta Standard)": "ollama:llama3.1:8b",
    "[Ollama] mistral:7b (High JSON Reliability)": "ollama:mistral:7b",
    "[Ollama] gemma2:9b (Scientific Nuance)": "ollama:gemma2:9b",
    "[Ollama] falcon3:10b (Instruction Following)": "ollama:falcon3:10b",
}

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

if __name__ == "__main__":
    run_streamlit()
