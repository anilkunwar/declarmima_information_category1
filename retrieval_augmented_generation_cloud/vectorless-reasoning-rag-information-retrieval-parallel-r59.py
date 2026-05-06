#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DECLARMIMA v12.3-VECTORLESS - PHYSICAL QUANTITY DISTINCTION
=======================================================================
- Distinguishes quantities with same unit (scan speed vs flow speed, laser power vs electrical power)
- Configurable retrieval text length (sidebar slider)
- Cached query results to avoid reruns on display toggle
- Physical quantity classifier using parameter names, context, and units
- Human summary grouped by physical quantity (e.g., "Scan Speed", "Laser Power", "Irradiance", "Flow Speed")
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
from dataclasses import dataclass, field
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
# OPTIONAL IMPORTS
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

# ============================================================================
# 1. PYDANTIC MODELS WITH PHYSICAL QUANTITY FIELD
# ============================================================================
from pydantic import BaseModel, Field, field_validator


class UniversalExtractionItem(BaseModel):
    item_type: Literal["quantitative", "qualitative", "definition", "comparison", "relationship", "process", "material", "method"]
    content: str
    parameter_name: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    physical_quantity: Optional[str] = None   # e.g., "laser_power", "scan_speed", "flow_speed", "temperature"
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
    material: Optional[str] = None
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
    physical_quantity: str   # Unified physical category (e.g., "scan_speed")
    parameter_name: Optional[str] = None
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


# ============================================================================
# 2. PHYSICAL QUANTITY CLASSIFIER (DISTINGUISHES SCAN SPEED vs FLOW SPEED)
# ============================================================================
class PhysicalQuantityClassifier:
    """
    Maps parameter names, context, and units to canonical physical quantity labels.
    This allows distinguishing quantities that share units (e.g., scan speed vs flow speed).
    """
    
    # Canonical physical quantity categories
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
        # Other
        "hardness": ["hardness", "vickers hardness", "microhardness", "hv"],
        "elongation": ["elongation", "strain", "ductility"],
    }
    
    # Unit hints for disambiguation (when parameter name is ambiguous)
    UNIT_HINTS = {
        "scan_speed": ["mm/s", "cm/s", "m/s", "mm/min", "in/min"],
        "flow_speed": ["mm/s", "cm/s", "m/s", "l/min", "m³/s"],
        "laser_power": ["w", "kw", "mw"],
        "irradiance": ["w/cm²", "kw/cm²", "w/m²"],
        "temperature": ["°c", "k", "°f"],
        "energy_density": ["j/mm³", "j/m³", "j/cm³", "j/m²"],
    }
    
    def __init__(self):
        self._build_keyword_index()
    
    def _build_keyword_index(self):
        """Build a reverse mapping from keywords to canonical labels."""
        self.keyword_to_canonical = {}
        for canonical, keywords in self.CANONICAL.items():
            for kw in keywords:
                self.keyword_to_canonical[kw.lower()] = canonical
        # Add common variants
        self.keyword_to_canonical["scanning velocity"] = "scan_speed"
        self.keyword_to_canonical["scan velocity"] = "scan_speed"
        self.keyword_to_canonical["flow rate"] = "flow_speed"
        self.keyword_to_canonical["power"] = "laser_power"  # default, but will be overridden by context
        self.keyword_to_canonical["speed"] = "scan_speed"
    
    def classify(self, parameter_name: Optional[str], unit: Optional[str], context: str) -> str:
        """
        Determine the physical quantity based on parameter name, unit, and context.
        Returns a canonical string like "scan_speed", "laser_power", etc.
        """
        # 1. Priority: explicit parameter name keywords
        if parameter_name:
            pname_lower = parameter_name.lower().strip()
            # Direct match
            for canonical, keywords in self.CANONICAL.items():
                for kw in keywords:
                    if kw in pname_lower:
                        return canonical
            # Check keyword index
            for kw, canonical in self.keyword_to_canonical.items():
                if kw in pname_lower:
                    return canonical
        
        # 2. Use context (first 200 chars) to find keywords
        context_lower = context.lower()
        for canonical, keywords in self.CANONICAL.items():
            for kw in keywords:
                if kw in context_lower:
                    return canonical
        
        # 3. Fallback: use unit if available
        if unit:
            unit_lower = unit.lower()
            for canonical, units in self.UNIT_HINTS.items():
                for u in units:
                    if u in unit_lower:
                        return canonical
        
        # 4. Default mapping based on unit patterns
        if unit:
            if "w/cm" in unit_lower or "kw/cm" in unit_lower:
                return "irradiance"
            if unit_lower in ["w", "kw", "mw"]:
                return "laser_power"  # conservative default
            if "mm/s" in unit_lower or "cm/s" in unit_lower:
                return "scan_speed"   # default speed -> scan
            if "°c" in unit_lower or "k" in unit_lower:
                return "temperature"
        
        # 5. Last resort: unknown
        return "unknown"
    
    def get_human_readable(self, canonical: str) -> str:
        """Convert canonical label to human-readable title."""
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
            "hardness": "Hardness",
            "elongation": "Elongation",
            "unknown": "Other Quantities"
        }
        return mapping.get(canonical, canonical.replace("_", " ").title())


class QuantityClassifierLegacy:
    """Legacy classifier for unit-based classification – kept for compatibility."""
    @staticmethod
    def classify(value, unit, context):
        unit = unit.lower()
        if "w/cm²" in unit or "kw/cm²" in unit:
            return "irradiance"
        if unit in ["w", "kw", "mw"]:
            return "laser_power"
        if "mm/s" in unit:
            if "flow" in context.lower():
                return "flow_speed"
            else:
                return "scan_speed"
        return "unknown"


# ============================================================================
# 3. GLOBAL CONFIGURATION
# ============================================================================
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


# ============================================================================
# 4. TIMING & CACHING (unchanged)
# ============================================================================
@contextmanager
def timer(label: str):
    start = time.time()
    yield
    elapsed = time.time() - start
    if not hasattr(timer, 'metrics'):
        timer.metrics = defaultdict(list)
    timer.metrics[label].append(elapsed)
    logger.info(f"⏱️ {label}: {elapsed:.2f}s")

def get_timer_metrics():
    if not hasattr(timer, 'metrics'):
        return {}
    return {k: {"mean": np.mean(v), "std": np.std(v), "count": len(v)} for k, v in timer.metrics.items()}

def reset_timer_metrics():
    if hasattr(timer, 'metrics'):
        timer.metrics.clear()


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


# ============================================================================
# 5. FAST JSON UTILITIES
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


# ============================================================================
# 6. HIERARCHICAL PDF INDEX (with configurable text length)
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

    def get_text(self, doc_cache: Dict[str, Any] = None) -> str:
        if self.full_text:
            return self.full_text
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
        return self.full_text

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
            "children": [c.to_dict() for c in self.children]
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
        if self.full_text:
            if len(self.full_text) > max_chars:
                result["text"] = self.full_text[:max_chars] + "..."
            else:
                result["text"] = self.full_text
        elif self._pdf_path and self.get_text():
            full = self.get_text()
            if len(full) > max_chars:
                result["text"] = full[:max_chars] + "..."
            else:
                result["text"] = full
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
        return node


class HierarchicalIndex:
    def __init__(self, cache_dir=".declarmima_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.doc_trees: Dict[str, PageNode] = {}
        self._pdf_cache = {}

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
            text = doc[p - 1].get_text("text")
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
        return "\n\n".join(doc[p - 1].get_text("text") for p in range(start, min(end, len(doc) + 1)))

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
            children=[self._clone_for_cache(c) for c in node.children]
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
        # ROBUST: toc title may be explicit None
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
                # ---- ROBUST NULL/INVALID HANDLING ----
                level = entry.get("level")
                try:
                    level = int(level) if level is not None else 1
                except (ValueError, TypeError):
                    level = 1
                
                title = entry.get("title")
                if title is None:
                    title = "Unknown"
                title = str(title).strip()
                
                page = entry.get("page")
                try:
                    page = int(page) if page is not None else 1
                except (ValueError, TypeError):
                    page = 1
                # ---------------------------------------
                
                if page < 1 or page > len(pages):
                    continue
                
                end = min(page + window, len(pages))
                # Safe text extraction with bounds checking
                text_parts = []
                for i in range(page, min(end + 1, len(pages) + 1)):
                    try:
                        page_data = pages[i - 1]
                        if isinstance(page_data, dict) and 'text' in page_data:
                            text_parts.append(page_data['text'])
                    except (IndexError, KeyError, TypeError):
                        continue
                text = "\n\n".join(text_parts)
                
                node_id = f"{doc_name}_toc_{level}_{title[:20]}"
                node = PageNode(
                    node_id,
                    title, page, end,
                    text, text[:200], level,
                    doc_id=doc_name
                )
                nodes_by_level.setdefault(level, []).append(node)
            
            for level in sorted(nodes_by_level.keys()):
                for node in nodes_by_level[level]:
                    parent = self._find_parent(root, level - 1, node.page_start)
                    parent.children.append(node)
        else:
            for p in pages:
                if not isinstance(p, dict):
                    continue
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
# 7. HYBRID LLM CLIENT (unchanged from earlier)
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
# 8. QUANTITATIVE KNOWLEDGE GRAPH (with physical quantity)
# ============================================================================
class QuantitativeKnowledgeGraph:
    def __init__(self):
        self.doc_graphs: Dict[str, Dict] = {}
        self.phys_classifier = PhysicalQuantityClassifier()

    def add_extractions(self, doc_id: str, items: List[UniversalExtractionItem]):
        graph = {
            "doc_id": doc_id,
            "parameters": defaultdict(list),
            "materials": defaultdict(list),
            "methods": defaultdict(list),
            "by_page": defaultdict(list),
            "by_section": defaultdict(list),
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
        """Get stats for a specific physical quantity across all documents."""
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
        """Return mapping physical_quantity -> count of occurrences."""
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
                    confidence=item.get("confidence", 0.7),
                    context=item.get("context", "")[:300],
                    doc_name=doc_id,
                    page=item.get("page", 1),
                    section_title=item.get("section_title")
                ))
        return all_values


# ============================================================================
# 9. HIERARCHICAL TREE RETRIEVER (passes max_chars)
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
1. Analyze each document's tree structure (titles, summaries, quantitative hints, candidate values)
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
# 10. UNIVERSAL LLM EXTRACTOR (ENHANCED WITH PHYSICAL QUANTITY)
# ============================================================================
class UniversalLLMExtractor:
    EXTRACTION_PROMPT = """Extract information relevant to the query from these document sections.
QUERY: {query}
QUERY TYPE: {query_type}
SECTIONS:
{sections_text}

Return JSON array of extracted items with fields:
{{"item_type": "quantitative|qualitative|definition...",
  "content": "exact phrase with full numerical value (never truncate numbers)",
  "confidence": 0.0-1.0,
  "context": "exact sentence from text",
  "doc_source": "{doc_id}",
  "page": page_number,
  "parameter_name": "...",
  "value": number,
  "unit": "e.g., W, kW, W/cm², mm/s, °C",
  "physical_quantity": "one of: laser_power, electrical_power, scan_speed, flow_speed, irradiance, temperature, energy_density, layer_thickness, spot_size, exposure_time, hardness, elongation, unknown"}}

For quantitative items, ALWAYS include:
- "parameter_name" (e.g., "laser power", "scan speed", "flow velocity")
- "value" (full number, never truncated)
- "unit" (with correct formatting)
- "physical_quantity" (choose from the list above, based on the physical meaning, not just the unit)

CRITICAL RULES:
1. Distinguish physically different quantities even if they share units:
   - "scan speed" (laser scanning) vs "flow speed" (fluid movement) – both may use mm/s, but assign different physical_quantity.
   - "laser power" (W) vs "electrical power" (W) – assign different physical_quantity.
2. NEVER truncate numbers: if text says "1000 W", output 1000.
3. Return ONLY valid JSON, no extra text.
4. Set confidence based on clarity.

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
                        # If physical_quantity missing, classify now
                        if "physical_quantity" not in item_data or not item_data["physical_quantity"]:
                            item_data["physical_quantity"] = self.phys_classifier.classify(
                                item_data.get("parameter_name"),
                                item_data.get("unit"),
                                item_data.get("context", "")
                            )
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
            key = (i.content, i.doc_source, i.page)
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
# 11. LLM REASONING SYNTHESIZER (GROUPED BY PHYSICAL QUANTITY)
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
            line = f"- {pq_readable}: {item.content} ({item.confidence:.2f}) context: {item.context[:200]} {item.citation()}"
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
(Group findings by physical quantity: e.g., Laser Power, Scan Speed, Irradiance, etc.)

**Consensus & Variability**
(For each physical quantity, report range/mean if multiple values exist)

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
        # Group by physical_quantity
        by_phys = defaultdict(list)
        for v in values:
            by_phys[v.physical_quantity].append(v)
        lines = [
            f"## Summary: {query.title()}",
            f"Across **{report.total_docs}** documents analyzed, **{report.docs_with_results}** contained relevant quantitative data.",
            f"Total extracted values: **{len(values)}**.",
            ""
        ]
        for pq, vals in sorted(by_phys.items()):
            readable = self.phys_classifier.get_human_readable(pq)
            lines.append(f"### {readable} ({len(vals)} values)")
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
        lines.append("### Key Values by Document and Physical Quantity")
        for v in sorted(values, key=lambda x: x.confidence, reverse=True)[:12]:
            readable = self.phys_classifier.get_human_readable(v.physical_quantity)
            lines.append(f"| {v.doc_name} | p.{v.page} | {v.value:.2f} {v.unit} | {readable} |")
        return "\n".join(lines)


# ============================================================================
# 12. STREAMLIT UI (with physical quantity grouping)
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
        st.caption(f"GPU: {'CUDA' if torch.cuda.is_available() else 'CPU'}")


@st.cache_resource(show_spinner="Initializing LLM...")
def get_cached_llm(model_choice: str, use_4bit: bool):
    internal = LOCAL_LLM_OPTIONS[model_choice]
    return HybridLLM(model_key=internal, use_4bit=use_4bit)


def run_streamlit():
    st.set_page_config(page_title="DECLARMIMA v12.3 - Physical Quantity Distinction", layout="wide")
    st.markdown("# 🔬 DECLARMIMA v12.3 - Physical Quantity Distinction")
    st.caption("Hierarchical RAG that distinguishes scan speed vs flow speed, laser power vs electrical power, etc.")

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

    render_sidebar()
    max_retrieval_chars = st.session_state.get("max_retrieval_chars", 20000)

    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files and st.button("🚀 Build Index", type="primary"):
        st.session_state.query_processor["files"] = uploaded_files
        st.success(f"{len(uploaded_files)} files registered.")
        st.rerun()

    if st.session_state.query_processor.get("files") and not st.session_state.annotated_trees:
        with st.spinner("Building hierarchical index..."):
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
                initial_prompt = "Extract all quantitative parameters (laser power, scan speed, flow speed, irradiance, temperature, energy density, etc.) with full numerical values, correct units, and physical_quantity classification."
                items = extractor.extract_from_chunks(leaf_texts, initial_prompt)
                all_items.extend(items)
                kg.add_extractions(doc_name, items)
            st.session_state.knowledge_graph = kg
            progress.progress(0.8)
            annotated = []
            for doc_name, tree in trees.items():
                ann = kg.to_tree_annotation(tree, max_chars=max_retrieval_chars)
                ann["doc_id"] = doc_name
                ann["doc_name"] = doc_name
                annotated.append(ann)
            st.session_state.annotated_trees = annotated
            progress.progress(1.0)
            st.success(f"✅ Indexed {len(trees)} documents with {len(all_items)} quantitative items")
            with st.expander("📊 Detected Physical Quantities", expanded=True):
                pq_counts = kg.get_all_physical_quantities()
                if pq_counts:
                    for pq, count in sorted(pq_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
                        st.write(f"- `{pq}`: {count} occurrences")
                else:
                    st.write("No physical quantities detected yet.")

    if st.session_state.annotated_trees:
        st.markdown("### ⚡ Quick Queries")
        col1, col2, col3, col4 = st.columns(4)
        quick = ["laser power", "scan speed", "flow speed", "temperature"]
        for i, q in enumerate(quick):
            with [col1, col2, col3, col4][i]:
                if st.button(f"📈 {q.title()}", key=f"quick_{q}"):
                    st.session_state.quick_query = f"What is the {q} discussed in these papers?"
                    st.rerun()

        default_query = st.session_state.get("quick_query", "")
        prompt = st.chat_input("Ask about any term, value, or concept...", key="chat_input")
        if default_query and not prompt:
            prompt = default_query
            st.session_state.quick_query = ""

        run_query = False
        if prompt:
            if st.session_state.cached_query_result is None or st.session_state.cached_query_result.get("prompt") != prompt:
                run_query = True
                st.session_state.cached_query_result = {"prompt": prompt}
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

        if run_query:
            with st.chat_message("assistant"):
                progress = st.progress(0)
                progress.text("Initializing LLM...")
                llm = get_cached_llm(st.session_state.llm_model_choice, st.session_state.get("use_4bit", True))
                progress.progress(0.15)
                retriever = HierarchicalTreeRetriever(llm, max_results=30, max_text_chars=max_retrieval_chars)
                retrieved = asyncio.run(retriever.retrieve_quantitative(prompt, st.session_state.annotated_trees))
                progress.progress(0.4)
                extractor = UniversalLLMExtractor(llm)
                items = []
                for r in retrieved:
                    items.extend(extractor.extract_from_chunks([r], prompt))
                min_conf = st.session_state.get("min_confidence", 0.55)
                items = [i for i in items if i.confidence >= min_conf]
                progress.progress(0.6)
                synthesizer = LLMReasoningSynthesizer(llm)
                extracted_values = []
                for item in items:
                    if item.item_type == "quantitative" and item.value is not None:
                        phys_q = item.physical_quantity or synthesizer.phys_classifier.classify(
                            item.parameter_name, item.unit, item.context
                        )
                        extracted_values.append(ExtractedValue(
                            query=prompt,
                            value=item.value,
                            unit=item.unit or "",
                            physical_quantity=phys_q,
                            parameter_name=item.parameter_name,
                            confidence=item.confidence,
                            context=item.context,
                            doc_name=item.doc_source,
                            page=item.page,
                            section_title=item.section_title
                        ))
                if extracted_values:
                    report = QueryReport(
                        query=prompt,
                        total_docs=len(st.session_state.annotated_trees),
                        docs_with_results=len(set(v.doc_name for v in extracted_values)),
                        all_values=extracted_values,
                        consensus={},
                        processing_time_sec=0.0
                    )
                    answer = synthesizer.generate_human_conclusion(prompt, report)
                else:
                    answer = synthesizer.synthesize(prompt, items)
                progress.progress(1.0, text="Done!")
                st.markdown(answer)
                st.session_state.cached_query_result = {
                    "prompt": prompt,
                    "retrieved": retrieved,
                    "items": items,
                    "extracted_values": extracted_values,
                    "answer": answer
                }
        else:
            if st.session_state.cached_query_result and st.session_state.cached_query_result.get("prompt") == prompt:
                cached = st.session_state.cached_query_result
                with st.chat_message("assistant"):
                    st.markdown(cached["answer"])
                    retrieved = cached["retrieved"]
                    items = cached["items"]
                    extracted_values = cached["extracted_values"]
            else:
                if not prompt:
                    st.info("Ask a question about the documents.")
                return

        st.markdown("---")
        st.subheader("📊 Quantitative Results")
        display_mode = st.radio("Display format", ["Table", "JSON", "Human Summary"], horizontal=True, key="display_mode")
        if display_mode == "Table" and extracted_values:
            import pandas as pd
            df_data = []
            for v in extracted_values:
                phys_readable = PhysicalQuantityClassifier().get_human_readable(v.physical_quantity)
                df_data.append({
                    "Document": v.doc_name,
                    "Page": v.page,
                    "Value": f"{v.value:.2f}",
                    "Unit": v.unit,
                    "Physical Quantity": phys_readable,
                    "Parameter": v.parameter_name or "",
                    "Confidence": f"{v.confidence:.2f}"
                })
            st.dataframe(df_data, use_container_width=True)
        elif display_mode == "JSON" and extracted_values:
            st.json([v.model_dump() for v in extracted_values])
        elif display_mode == "Human Summary" and extracted_values:
            report = QueryReport(
                query=prompt,
                total_docs=len(st.session_state.annotated_trees),
                docs_with_results=len(set(v.doc_name for v in extracted_values)),
                all_values=extracted_values,
                consensus={},
                processing_time_sec=0.0
            )
            conclusion = LLMReasoningSynthesizer(None).generate_human_conclusion(prompt, report)
            st.markdown(conclusion)

        if st.session_state.get("show_tree_nav") and retrieved:
            with st.expander("🌳 Tree Navigation Trace", expanded=False):
                for r in retrieved[:5]:
                    st.markdown(f"**{r['doc_id']}** → `{r['section_title']}` (p.{r['page_start']}) | confidence: {r.get('confidence', 0):.2f}")
                    st.caption(r.get('selection_reasoning', ''))
        if items:
            with st.expander("🔍 Extracted Items (Raw)", expanded=False):
                st.json([i.to_dict() for i in items[:10]])
        if extracted_values:
            with st.expander("📊 Quick Statistics by Physical Quantity", expanded=False):
                by_phys = defaultdict(list)
                for v in extracted_values:
                    by_phys[v.physical_quantity].append(v.value)
                for phys, vals in by_phys.items():
                    readable = PhysicalQuantityClassifier().get_human_readable(phys)
                    st.metric(f"{readable} count", len(vals))
                    if vals:
                        st.metric(f"{readable} min", f"{min(vals):.2f}")
                        st.metric(f"{readable} max", f"{max(vals):.2f}")
                        st.metric(f"{readable} mean", f"{np.mean(vals):.2f}")

        report = CrossDocumentQueryReport(
            query=prompt,
            total_documents=len(st.session_state.annotated_trees),
            documents_with_results=len(set(i.doc_source for i in items)),
            all_items=items
        )
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button("📥 Download JSON Report", report.to_json(), "results.json", "application/json")
        with col_dl2:
            tree_export = {
                "query": prompt,
                "annotated_trees": st.session_state.annotated_trees,
                "retrieved_nodes": retrieved,
                "extracted_items": [i.to_dict() for i in items],
                "answer": answer
            }
            st.download_button("📥 Download Tree Export", json.dumps(tree_export, indent=2, ensure_ascii=False, default=str), "tree_report.json", "application/json")
        st.session_state.messages.append({"role": "assistant", "content": answer})
        if "index" in st.session_state.query_processor:
            st.session_state.query_processor["index"].cleanup()
    else:
        st.info("👆 Upload PDF files to begin.")


if __name__ == "__main__":
    run_streamlit()
