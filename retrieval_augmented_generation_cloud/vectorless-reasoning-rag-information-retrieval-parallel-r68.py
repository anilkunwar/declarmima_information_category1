#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DECLARMIMA v14.0 - VECTORLESS RETRIEVAL + LASER POWER / SCAN SPEED VISUALIZATION SUITE
========================================================================================
- Focused extraction: laser power (W, kW), scan speed (mm/s, m/s, mm/min), materials/alloys/compounds
- Two-stage retrieval using ONLY metadata & regex (no sentence-transformers required)
- LLM reasoning explicitly separates quantifiable features
- Rich visualizations: counts per material, sunburst, network, chord diagram, radar, contradiction matrix
- Knowledge graph linking documents → materials → parameters
- Full PyVis interactive network with salience
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
# VISUALIZATION DEPENDENCIES - ALL IMPORTED
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

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from bokeh.plotting import figure
    from bokeh.models import HoverTool, ColumnDataSource, LabelSet
    from bokeh.embed import file_html
    from bokeh.resources import CDN
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ============================================================================
# OPTIONAL IMPORTS (LLM & PDF only)
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
    logging.warning("Ollama not installed. Ollama backend unavailable.")

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

# We deliberately do NOT import sentence_transformers – vectorless retrieval

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("DECLARMIMA")

# ============================================================================
# PYDANTIC MODELS
# ============================================================================
from pydantic import BaseModel, Field, field_validator

class UniversalExtractionItem(BaseModel):
    item_type: Literal["quantitative", "qualitative", "definition", "comparison", "relationship", "process", "material", "method"]
    content: str
    parameter_name: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    physical_quantity: Optional[str] = None   # "laser_power", "scan_speed", "material"
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
# PHYSICAL QUANTITY CLASSIFIER (focused on laser power & scan speed)
# ============================================================================
class PhysicalQuantityClassifier:
    CANONICAL = {
        "laser_power": ["laser power", "laser beam power", "laser output power", "power", "p (w)"],
        "scan_speed": ["scan speed", "scanning speed", "laser scan speed", "scan velocity", "v_scan", "vs"],
        "material": ["alloy", "material", "compound", "composition", "aluminum", "titanium", "steel", "inconel", "ni", "ti", "al", "mg", "cu", "fe"],
    }
    UNIT_HINTS = {
        "laser_power": ["w", "kw", "mw"],
        "scan_speed": ["mm/s", "cm/s", "m/s", "mm/min", "in/min"],
    }
    def __init__(self):
        self._build_keyword_index()
    def _build_keyword_index(self):
        self.keyword_to_canonical = {}
        for canonical, keywords in self.CANONICAL.items():
            for kw in keywords:
                self.keyword_to_canonical[kw.lower()] = canonical
    def classify(self, parameter_name: Optional[str], unit: Optional[str], context: str) -> str:
        if parameter_name:
            pname_lower = parameter_name.lower()
            if "laser power" in pname_lower or "power" in pname_lower and ("w" in pname_lower or "kw" in pname_lower):
                return "laser_power"
            if "scan speed" in pname_lower or "scanning speed" in pname_lower or "scan velocity" in pname_lower:
                return "scan_speed"
            for kw in ["alloy", "material", "compound"]:
                if kw in pname_lower:
                    return "material"
        context_lower = context.lower()
        if "laser power" in context_lower or ("power" in context_lower and ("w" in context_lower or "kw" in context_lower)):
            return "laser_power"
        if "scan speed" in context_lower or "scanning speed" in context_lower:
            return "scan_speed"
        if any(kw in context_lower for kw in ["alloy", "material", "compound", "ti6al4v", "alsi10mg", "inconel"]):
            return "material"
        if unit:
            unit_lower = unit.lower()
            if any(u in unit_lower for u in ["w", "kw", "mw"]):
                return "laser_power"
            if any(u in unit_lower for u in ["mm/s", "cm/s", "m/s", "mm/min"]):
                return "scan_speed"
        return "unknown"
    def get_human_readable(self, canonical: str) -> str:
        mapping = {"laser_power": "Laser Power", "scan_speed": "Scan Speed", "material": "Material/Alloy", "unknown": "Other"}
        return mapping.get(canonical, canonical.replace("_", " ").title())

# ============================================================================
# PAGINATION-AWARE PDF READER (unchanged)
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
                logger.warning(f"Page {pnum} text length {len(text)} exceeds limit, truncating.")
                text = text[:self.max_chars_per_request] + "\n...[TRUNCATED]"
            result[pnum] = text
        doc.close()
        return result
    def extract_page_range(self, doc_path: str, start: int, end: int, step=1) -> Dict[int, str]:
        pages = list(range(start, end+1, step))
        return self.extract_pages(doc_path, pages)

# ============================================================================
# STRUCTURED METADATA EXTRACTOR (enhanced for laser power, scan speed, materials)
# ============================================================================
class StructuredMetadataExtractor:
    ALLOY_PATTERNS = [
        r'\b(?:AlSi[\dMg]+|Ti\d*Al\d*V\d*|Inconel\s?\d{3}|SS\s?\d{4}|UNS\s?S\d{5}|Ti\s?6Al\s?4V|Cu\s?[A-Za-z0-9]+|Fe-based|Mg\s?alloy)\b',
        r'\b(?:Al-[\d]+Si-[\d]+Mg|AlSiMg[\d\.]+Zr|TiB[2]?|CoCr[\w]+|NiTi|Au\-Ti|Zr\-enhanced)\b',
        r'(\w+(?:-\w+)?\s?(?:alloy|superalloy|metal|composite))'
    ]
    POWER_PATTERN = r'(?:laser\s+power|power|P)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(W|kW|mW)'
    SCAN_SPEED_PATTERN = r'(?:scan\s+speed|scanning\s+speed|v_scan|Vs)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(mm/s|cm/s|m/s|mm/min)'
    def __init__(self):
        self.power_re = re.compile(self.POWER_PATTERN, re.IGNORECASE)
        self.speed_re = re.compile(self.SCAN_SPEED_PATTERN, re.IGNORECASE)
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
        # Laser power
        power_matches = self.power_re.findall(full_text)
        for val_str, unit in power_matches:
            try:
                val = float(val_str)
                if unit.lower() == "kw":
                    val *= 1000
                elif unit.lower() == "mw":
                    val /= 1000
                meta.laser_power_values.append(val)
            except:
                pass
        # Scan speed
        speed_matches = self.speed_re.findall(full_text)
        for val_str, unit in speed_matches:
            try:
                val = float(val_str)
                if unit.lower() == "cm/s":
                    val *= 10
                elif unit.lower() == "m/s":
                    val *= 1000
                elif unit.lower() == "mm/min":
                    val /= 60
                meta.scan_speed_values.append(val)
            except:
                pass
        return meta

# ============================================================================
# TWO-STAGE RETRIEVER (VECTORLESS - pure keyword + metadata scoring)
# ============================================================================
class TwoStageRetriever:
    def __init__(self, llm: Optional['HybridLLM'] = None):
        self.llm = llm
        self.doc_metadata: Dict[str, DocumentMetadata] = {}
        self.doc_summaries: Dict[str, str] = {}
    def index_document(self, doc_name: str, metadata: DocumentMetadata, summary: str):
        self.doc_metadata[doc_name] = metadata
        self.doc_summaries[doc_name] = summary
    def retrieve_relevant_docs(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Pure keyword/field matching – no embeddings."""
        query_lower = query.lower()
        scores = []
        for name, meta in self.doc_metadata.items():
            score = 0.0
            # laser power mentioned in query?
            if "laser power" in query_lower or "power" in query_lower:
                if meta.laser_power_values:
                    score += 0.5
            # scan speed
            if "scan speed" in query_lower or "scanning speed" in query_lower:
                if meta.scan_speed_values:
                    score += 0.5
            # materials
            if any(mat.lower() in query_lower for mat in meta.alloys):
                score += 0.3
            # process types
            if any(proc.lower() in query_lower for proc in meta.process_types):
                score += 0.2
            # if query contains numeric hints like "100 W"
            numeric_match = re.search(r'\d+\s*(W|kW|mm/s)', query_lower)
            if numeric_match:
                score += 0.2
            scores.append((name, min(score, 1.0)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    def get_relevant_pages(self, doc_name: str, query: str, max_pages: int = 5) -> List[int]:
        # Simple: return first max_pages (can be enhanced with keyword density)
        return list(range(1, max_pages+1))

# ============================================================================
# HIERARCHICAL PDF INDEX (unchanged structure, abbreviated for length)
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
            "id": self.id, "title": self.title, "page_start": self.page_start, "page_end": self.page_end,
            "summary": self.summary, "prefix_summary": self.prefix_summary, "level": self.level,
            "doc_id": self.doc_id, "section_type": self.section_type, "node_id": self.node_id,
            "text_token_count": self.text_token_count, "children": [c.to_dict() for c in self.children],
            "metadata": self.metadata.dict() if self.metadata else None
        }
    def to_tree_format(self, max_chars: int = 20000) -> Dict[str, Any]:
        result = {"title": self.title, "node_id": self.node_id, "start_index": self.page_start,
                  "end_index": self.page_end or self.page_start, "summary": self.summary,
                  "prefix_summary": self.prefix_summary, "text_token_count": self.text_token_count}
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
        node = cls(data["id"], data["title"], data["page_start"], data.get("page_end"), "",
                   data.get("summary", ""), data.get("level", 0), doc_id=data.get("doc_id", ""),
                   section_type=data.get("section_type", "BODY"), _pdf_path=pdf_path)
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
                        root_data = orjson.loads(f.read()) if ORJSON_AVAILABLE else json.load(f)
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
                    data = orjson.dumps(cache_root.to_dict(), option=orjson.OPT_INDENT_2) if ORJSON_AVAILABLE else json.dumps(cache_root.to_dict(), indent=2).encode()
                    f.write(data)
            except:
                pass
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
        root = PageNode(f"{doc_id}_root", "Document Root", 1, len(doc), "",
                        f"Document {doc_id} root covering pages 1-{len(doc)}", 0, doc_id=doc_id, _pdf_path=pdf_path, node_id="0000")
        window = 7
        # Fallback: page-by-page
        for p in range(1, len(doc)+1):
            text = doc[p-1].get_text("text")
            if not text.strip():
                continue
            node = PageNode(f"{doc_id}_p{p}", f"Page {p}", p, p, text, text[:200], 3, doc_id=doc_id, _pdf_path=pdf_path)
            root.children.append(node)
        self._assign_node_ids(root)
        return root
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
        return PageNode(node.id, node.title, node.page_start, node.page_end, "",
                        node.summary, node.level, doc_id=node.doc_id, section_type=node.section_type,
                        node_id=node.node_id, prefix_summary=node.prefix_summary, text_token_count=node.text_token_count,
                        children=[self._clone_for_cache(c) for c in node.children], metadata=node.metadata)
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
        trees = {}
        for (doc_name, pages) in raw_docs:
            tree = self._build_tree_from_pages(doc_name, pages)
            full_text = "\n".join([p['text'] for p in pages])
            meta = self.metadata_extractor.extract_metadata(doc_name, full_text)
            tree.metadata = meta
            trees[doc_name] = tree
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
            pages.append({'page_num': p+1, 'text': page.get_text("text"), 'images': len(page.get_images()), 'blocks': page.get_text("blocks")})
        doc.close()
        return doc_name, pages
    def _build_tree_from_pages(self, doc_name: str, pages: List[Dict]) -> PageNode:
        root = PageNode(f"{doc_name}_root", doc_name, 1, len(pages), "", f"Document {doc_name}", 0, doc_id=doc_name, node_id="0000")
        for p in pages:
            text = p.get('text', '')
            if not str(text).strip():
                continue
            page_num = int(p.get('page_num', 1))
            node = PageNode(f"{doc_name}_p{page_num}", f"Page {page_num}", page_num, page_num, text, str(text)[:200], 3, doc_id=doc_name)
            root.children.append(node)
        self._assign_node_ids(root)
        return root
    def _save_tree_fast(self, doc_name: str, tree: PageNode):
        safe = re.sub(r'[^\w\-_.]', '_', doc_name)
        doc_hash = hashlib.sha256(doc_name.encode()).hexdigest()[:16]
        path = self.cache_dir / f"{safe}.{doc_hash}.tree.json"
        try:
            with open(path, "wb") as f:
                data = orjson.dumps(tree.to_dict(), option=orjson.OPT_INDENT_2) if ORJSON_AVAILABLE else json.dumps(tree.to_dict(), indent=2).encode()
                f.write(data)
        except:
            pass

# ============================================================================
# HYBRID LLM CLIENT (unchanged)
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
                outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=temp if temp > 0 else None, do_sample=temp > 0, pad_token_id=self.tokenizer.eos_token_id)
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
        model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32}
        if self.use_4bit and self.device == "cuda":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if self.device == "cuda":
            self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded.")

# ============================================================================
# QUANTITATIVE KNOWLEDGE GRAPH (enhanced for laser power, scan speed, materials)
# ============================================================================
class QuantitativeKnowledgeGraph:
    def __init__(self):
        self.doc_graphs: Dict[str, Dict] = {}
        self.phys_classifier = PhysicalQuantityClassifier()
        self.metadata_index: Dict[str, DocumentMetadata] = {}
    def add_document_metadata(self, doc_name: str, metadata: DocumentMetadata):
        self.metadata_index[doc_name] = metadata
    def add_extractions(self, doc_id: str, items: List[UniversalExtractionItem]):
        graph = {"doc_id": doc_id, "parameters": defaultdict(list), "materials": defaultdict(list),
                 "methods": defaultdict(list), "by_page": defaultdict(list), "by_section": defaultdict(list),
                 "by_physical_quantity": defaultdict(list), "all_items": []}
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
                        values_by_pq[item["physical_quantity"]].append({"value": item["value"], "unit": item.get("unit",""), "doc": doc_id, "page": item.get("page",1)})
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
        docs = list(set(item["doc_source"] for doc_id, graph in self.doc_graphs.items() for item in graph["all_items"] if item.get("physical_quantity") == physical_quantity))
        stats = {"count": len(values), "documents": docs, "values": values}
        if values:
            stats.update({"min": min(values), "max": max(values), "mean": float(np.mean(values)), "std": float(np.std(values)) if len(values) > 1 else 0})
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
                phys_q = item.get("physical_quantity") or self.phys_classifier.classify(item.get("parameter_name"), unit, item.get("context", ""))
                all_values.append(ExtractedValue(query=query, value=val, unit=unit, physical_quantity=phys_q, parameter_name=item.get("parameter_name"), material=item.get("material"), confidence=item.get("confidence", 0.7), context=item.get("context", "")[:300], doc_name=doc_id, page=item.get("page", 1), section_title=item.get("section_title")))
        return all_values
    # New methods for material/parameter network
    def get_parameter_values_per_material(self, parameter: str) -> Dict[str, List[float]]:
        result = defaultdict(list)
        for doc_id, graph in self.doc_graphs.items():
            for item in graph["all_items"]:
                if item.get("physical_quantity") == parameter and item.get("value") is not None:
                    mat = item.get("material", "Unknown")
                    result[mat].append(item["value"])
        return result

# ============================================================================
# UNIVERSAL LLM EXTRACTOR (focused on laser power, scan speed, materials)
# ============================================================================
class UniversalLLMExtractor:
    EXTRACTION_PROMPT = """Extract information relevant to the query from these document sections.
QUERY: {query}
QUERY TYPE: {query_type}
SECTIONS:
{sections_text}

Return JSON array of extracted items with fields:
{{
  "item_type": "quantitative|material",
  "content": "exact phrase with full numerical value or material name",
  "confidence": 0.0-1.0,
  "context": "exact sentence from text",
  "doc_source": "{doc_id}",
  "page": page_number,
  "parameter_name": "laser power or scan speed",
  "value": number (only for quantitative),
  "unit": "W, kW, mm/s, m/s, mm/min",
  "physical_quantity": "laser_power or scan_speed or material",
  "material": "alloy or material name if mentioned"
}}

CRITICAL RULES:
1. For laser power: use physical_quantity="laser_power", unit in W (convert kW to W)
2. For scan speed: use physical_quantity="scan_speed", unit in mm/s (convert cm/s, m/s, mm/min to mm/s)
3. For materials: create item with item_type="material", physical_quantity="material", material=name
4. Always extract the numerical value exactly as in text
5. Return ONLY valid JSON, no extra text.

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
            # Skip if no numeric and no alloy pattern
            if not re.search(r'\d+', text) and not re.search(r'(Al|Ti|Ni|Fe|Mg|Cu|Inconel|alloy|material)', text, re.I):
                continue
            prompt = self.EXTRACTION_PROMPT.format(query=query, query_type=qa.get("query_type","mixed"), sections_text=text[:4000], doc_id=doc)
            try:
                response = self.llm.generate(prompt, max_new_tokens=1024, fast_json=True)
                json_str = self._extract_json(response)
                if json_str:
                    data = json.loads(json_str)
                    for item_data in data if isinstance(data, list) else data.get("items", []):
                        if "physical_quantity" not in item_data or not item_data["physical_quantity"]:
                            item_data["physical_quantity"] = self.phys_classifier.classify(item_data.get("parameter_name"), item_data.get("unit"), item_data.get("context", ""))
                        item_data.setdefault("material", None)
                        # Unit conversion
                        if item_data.get("physical_quantity") == "laser_power" and item_data.get("unit"):
                            unit = item_data["unit"].lower()
                            if unit == "kw":
                                item_data["value"] *= 1000
                                item_data["unit"] = "W"
                            elif unit == "mw":
                                item_data["value"] /= 1000
                                item_data["unit"] = "W"
                        if item_data.get("physical_quantity") == "scan_speed" and item_data.get("unit"):
                            unit = item_data["unit"].lower()
                            if unit == "cm/s":
                                item_data["value"] *= 10
                                item_data["unit"] = "mm/s"
                            elif unit == "m/s":
                                item_data["value"] *= 1000
                                item_data["unit"] = "mm/s"
                            elif unit == "mm/min":
                                item_data["value"] /= 60
                                item_data["unit"] = "mm/s"
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
        min_conf = 0.55
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
# LLM REASONING SYNTHESIZER (separates laser power, scan speed, materials)
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
        prompt = f"""You are an expert scientific analyst. The user wants to find laser power and/or scan speed for materials/alloys/compounds.

QUERY: {query}

EXTRACTED VALUES (with citations):
{extracted_text}

TASK: Synthesize the extracted information into a structured answer that SEPARATES the three features:

**Direct Answer**
(Concise answer listing found laser power values, scan speed values, and associated materials)

**Laser Power by Material**
(For each material, list laser power values with units and citations)

**Scan Speed by Material**
(For each material, list scan speed values with units and citations)

**Materials / Alloys / Compounds Found**
(List all unique materials mentioned, with document sources)

**Consensus & Variability**
(If multiple values exist for same material/parameter, report range/mean)

**Confidence Assessment**
(High/Medium/Low)

Do NOT invent information. Only use the extracted values above. Use citations with <cite doc="..." page="X"/>.

Return ONLY the answer text."""
        try:
            answer = self.llm.generate(prompt, max_new_tokens=1024, temperature=0.2)
            return answer.strip()
        except Exception as e:
            logger.error(f"Reasoning synthesis error: {e}")
            lines = [f"Query: {query}\nFound {len(items)} relevant items:\n"] + [f"- {item.content} {item.citation()}" for item in items[:5]]
            return "\n".join(lines)
    def generate_human_conclusion(self, query: str, report: QueryReport) -> str:
        values = report.all_values
        if not values:
            return f"No quantitative data found for '{query}' across the analyzed documents."
        by_material = defaultdict(lambda: {"laser_power": [], "scan_speed": []})
        for v in values:
            if v.material:
                if v.physical_quantity == "laser_power":
                    by_material[v.material]["laser_power"].append(v.value)
                elif v.physical_quantity == "scan_speed":
                    by_material[v.material]["scan_speed"].append(v.value)
        lines = [f"## Summary: {query.title()}", f"Across **{report.total_docs}** documents, **{report.docs_with_results}** contained relevant data.", f"Total extracted values: **{len(values)}**.", ""]
        lines.append("### Laser Power per Material")
        for mat, data in by_material.items():
            if data["laser_power"]:
                vals = data["laser_power"]
                lines.append(f"- **{mat}**: {min(vals):.2f} to {max(vals):.2f} W (mean {np.mean(vals):.2f} W)")
        lines.append("\n### Scan Speed per Material")
        for mat, data in by_material.items():
            if data["scan_speed"]:
                vals = data["scan_speed"]
                lines.append(f"- **{mat}**: {min(vals):.2f} to {max(vals):.2f} mm/s (mean {np.mean(vals):.2f} mm/s)")
        lines.append("\n### Materials Found")
        all_mats = set(v.material for v in values if v.material)
        for mat in sorted(all_mats):
            lines.append(f"- {mat}")
        return "\n".join(lines)

# ============================================================================
# HIERARCHICAL TREE RETRIEVER (simplified, no embeddings)
# ============================================================================
class HierarchicalTreeRetriever:
    def __init__(self, llm: HybridLLM, max_results=30, max_text_chars=20000):
        self.llm = llm
        self.max_results = max_results
        self.max_text_chars = max_text_chars
        self.template = llm.template if hasattr(llm, 'template') else {"system": "", "json_reminder": ""}
    async def retrieve_quantitative(self, query: str, annotated_trees: List[Dict]) -> List[Dict]:
        # Simplified: return all leaf nodes that contain numeric values or material mentions
        results = []
        for tree in annotated_trees:
            doc_id = tree.get("doc_id")
            self._collect_leaf_nodes(tree, doc_id, results)
        # Remove duplicates by page and doc
        unique = {}
        for r in results:
            key = (r["doc_id"], r["page_start"])
            if key not in unique:
                unique[key] = r
        return list(unique.values())[:self.max_results]
    def _collect_leaf_nodes(self, node: Dict, doc_id: str, results: List):
        if "nodes" in node and node["nodes"]:
            for child in node["nodes"]:
                self._collect_leaf_nodes(child, doc_id, results)
        else:
            text = node.get("text", "")
            if text and (re.search(r'\d+', text) or re.search(r'(Al|Ti|Ni|Fe|Mg|Cu|Inconel|alloy)', text, re.I)):
                results.append({
                    "full_text": text,
                    "page_start": node.get("start_index"),
                    "doc_id": doc_id,
                    "section_title": node.get("title"),
                    "quantitative_items": node.get("quantitative_items", [])
                })

# ============================================================================
# PUBLICATION-QUALITY VISUALIZATION ENGINE (focused on laser power, scan speed, materials)
# ============================================================================
class PublicationVisualizationEngine:
    COLORMAP_OPTIONS = {
        "viridis": "viridis", "plasma": "plasma", "inferno": "inferno", "magma": "magma",
        "cividis": "cividis", "Blues": "Blues", "Greens": "Greens", "Oranges": "Oranges",
        "Reds": "Reds", "coolwarm": "coolwarm", "Set1": "Set1", "tab10": "tab10"
    }
    def __init__(self, kgraph: QuantitativeKnowledgeGraph,
                 font_family: str = "DejaVu Sans", font_size: int = 10,
                 title_font_size: int = 14, label_font_size: int = 9,
                 default_colormap: str = "viridis", figure_dpi: int = 300):
        self.kgraph = kgraph
        self.font_family = font_family
        self.font_size = font_size
        self.title_font_size = title_font_size
        self.label_font_size = label_font_size
        self.default_colormap = default_colormap
        self.figure_dpi = figure_dpi
        plt.rcParams['font.family'] = font_family
        plt.rcParams['font.size'] = font_size
    def _get_colormap(self, name: Optional[str] = None) -> str:
        return self.COLORMAP_OPTIONS.get(name or self.default_colormap, "viridis")
    def plot_laser_power_by_material(self, df: pd.DataFrame, colormap: Optional[str] = None) -> go.Figure:
        """Bar chart of laser power values grouped by material."""
        if df.empty:
            return go.Figure().update_layout(title="No laser power data")
        subset = df[df["physical_quantity"] == "laser_power"]
        if subset.empty:
            return go.Figure().update_layout(title="No laser power extracted")
        fig = px.bar(subset, x="material", y="value", color="material", title="Laser Power by Material",
                     labels={"value": "Laser Power (W)", "material": "Material/Alloy"},
                     color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size), height=500)
        return fig
    def plot_scan_speed_by_material(self, df: pd.DataFrame, colormap: Optional[str] = None) -> go.Figure:
        subset = df[df["physical_quantity"] == "scan_speed"]
        if subset.empty:
            return go.Figure().update_layout(title="No scan speed data")
        fig = px.bar(subset, x="material", y="value", color="material", title="Scan Speed by Material",
                     labels={"value": "Scan Speed (mm/s)", "material": "Material/Alloy"},
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size), height=500)
        return fig
    def plot_material_count_sunburst(self, df: pd.DataFrame, colormap: Optional[str] = None) -> go.Figure:
        """Sunburst showing hierarchy: Document -> Material -> Parameter (laser_power/scan_speed)"""
        if df.empty:
            return go.Figure()
        # Build aggregated data
        records = []
        for _, row in df.iterrows():
            records.append({"doc": row["doc_name"], "material": row["material"], "parameter": row["physical_quantity"]})
        agg = pd.DataFrame(records).groupby(["doc", "material", "parameter"]).size().reset_index(name="count")
        fig = px.sunburst(agg, path=["doc", "material", "parameter"], values="count", title="Document–Material–Parameter Hierarchy",
                          color="parameter", color_discrete_map={"laser_power": "#3b82f6", "scan_speed": "#f59e0b"})
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
        return fig
    def plot_knowledge_network(self, df: pd.DataFrame, colormap: Optional[str] = None,
                               figsize=(12,10)) -> plt.Figure:
        """Network graph connecting Documents -> Materials -> Parameters (laser_power, scan_speed)"""
        G = nx.Graph()
        # Add nodes and edges
        docs = set(df["doc_name"])
        materials = set(df["material"].fillna("Unknown"))
        parameters = set(df["physical_quantity"])
        for d in docs:
            G.add_node(d, node_type="doc", group="doc")
        for m in materials:
            G.add_node(m, node_type="material", group="material")
        for p in parameters:
            G.add_node(p, node_type="parameter", group="parameter")
        for _, row in df.iterrows():
            d = row["doc_name"]
            m = row["material"] if pd.notna(row["material"]) else "Unknown"
            p = row["physical_quantity"]
            G.add_edge(d, m)
            G.add_edge(m, p)
        pos = nx.spring_layout(G, k=0.8, seed=42)
        fig, ax = plt.subplots(figsize=figsize)
        node_colors = []
        for node, data in G.nodes(data=True):
            group = data.get("group", "unknown")
            if group == "doc":
                node_colors.append("#1e40af")
            elif group == "material":
                node_colors.append("#10b981")
            else:
                node_colors.append("#f59e0b")
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.8, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=self.label_font_size, ax=ax, font_family=self.font_family)
        ax.set_title("Knowledge Graph: Documents → Materials → Parameters", fontsize=self.title_font_size, fontweight='bold')
        ax.axis("off")
        plt.tight_layout()
        return fig
    def plot_contradiction_heatmap(self, df: pd.DataFrame, parameter: str, colormap: Optional[str] = None) -> go.Figure:
        """Heatmap of value differences between documents for a given parameter."""
        subset = df[df["physical_quantity"] == parameter]
        if subset.empty:
            return go.Figure()
        docs = subset["doc_name"].unique()
        if len(docs) < 2:
            return go.Figure()
        # Build average per doc
        avg_per_doc = subset.groupby("doc_name")["value"].mean()
        mat = np.zeros((len(docs), len(docs)))
        for i, d1 in enumerate(docs):
            for j, d2 in enumerate(docs):
                if i == j:
                    continue
                v1 = avg_per_doc[d1]
                v2 = avg_per_doc[d2]
                if v2 != 0:
                    mat[i,j] = abs(v1 - v2) / v2
        fig = go.Figure(data=go.Heatmap(z=mat, x=docs, y=docs, colorscale="RdBu", hoverongaps=False))
        fig.update_layout(title=f"Contradiction Heatmap for {parameter.replace('_',' ').title()}", height=500)
        return fig
    def plot_radar_by_material(self, df: pd.DataFrame, colormap: Optional[str] = None) -> go.Figure:
        """Radar chart showing average laser power and scan speed per material."""
        pivot = df.pivot_table(index="material", columns="physical_quantity", values="value", aggfunc="mean").fillna(0)
        if pivot.empty or len(pivot.columns) < 2:
            return go.Figure()
        categories = ["laser_power", "scan_speed"]
        fig = go.Figure()
        for mat in pivot.index:
            values = [pivot.loc[mat, c] for c in categories]
            values += values[:1]
            fig.add_trace(go.Scatterpolar(r=values, theta=categories + [categories[0]], fill='toself', name=mat))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Material Performance Radar (Laser Power & Scan Speed)")
        return fig

# ============================================================================
# STREAMLIT UI (focus on vectorless retrieval and visualization)
# ============================================================================
LOCAL_LLM_OPTIONS = {
    "[Ollama] qwen2.5:0.5b (Fastest)": "ollama:qwen2.5:0.5b",
    "[Ollama] qwen2.5:1.5b (Balanced)": "ollama:qwen2.5:1.5b",
    "[Ollama] qwen2.5:7b (Recommended)": "ollama:qwen2.5:7b",
    "[Ollama] mistral:7b": "ollama:mistral:7b",
}
MODEL_PROMPT_TEMPLATES = {
    "qwen2.5": {"system": "You are a document analyst. Return JSON only.", "json_reminder": "Return ONLY valid JSON."},
    "default": {"system": "", "json_reminder": "Return valid JSON only."}
}
def get_model_template(model_name: str) -> Dict[str, Any]:
    for key in MODEL_PROMPT_TEMPLATES:
        if key in model_name.lower():
            return MODEL_PROMPT_TEMPLATES[key]
    return MODEL_PROMPT_TEMPLATES["default"]
UNIVERSAL_CONFIG = {"leaf_node_page_window": 7, "min_confidence_threshold": 0.55}

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        model_keys = list(LOCAL_LLM_OPTIONS.keys())
        if "llm_model_choice" not in st.session_state:
            st.session_state.llm_model_choice = model_keys[2]
        selected = st.selectbox("🧠 Select Local LLM", options=model_keys, index=model_keys.index(st.session_state.llm_model_choice), key="llm_model_select")
        st.session_state.llm_model_choice = selected
        st.checkbox("🗜️ Use 4-bit quantization (if Transformers)", value=True, key="use_4bit")
        st.slider("Confidence threshold", 0.3, 0.9, 0.55, 0.05, key="min_confidence")
        st.checkbox("Show reasoning trace", value=True, key="show_trace")
        st.checkbox("Show tree navigation", value=True, key="show_tree_nav")
        st.caption(f"GPU: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        st.markdown("#### 🎨 Visualization Settings")
        st.selectbox("Default colormap", list(PublicationVisualizationEngine.COLORMAP_OPTIONS.keys()), index=0, key="viz_colormap")
        st.caption("Vectorless retrieval active (no sentence-transformers)")

@st.cache_resource(show_spinner="Initializing LLM...")
def get_cached_llm(model_choice: str, use_4bit: bool):
    internal = LOCAL_LLM_OPTIONS[model_choice]
    return HybridLLM(model_key=internal, use_4bit=use_4bit)

def run_streamlit():
    st.set_page_config(page_title="DECLARMIMA v14.0 - Laser Power & Scan Speed Explorer", layout="wide")
    st.markdown("# 🔬 DECLARMIMA v14.0 - Vectorless Retrieval + Material/Parameter Visualization")
    st.caption("Focused on extracting **laser power**, **scan speed**, and **materials/alloys/compounds** from PDFs. Pure keyword/metadata retrieval – no embeddings required.")

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
    max_retrieval_chars = 20000

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
                            leaf_texts.append({"full_text": text, "page_start": node.page_start, "doc_id": doc_name, "section_title": node.title})
                    for c in node.children:
                        collect_leaves(c)
                collect_leaves(tree)
                initial_prompt = "Extract laser power (W), scan speed (mm/s), and material/alloy/compound names from these document sections."
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
            st.success(f"✅ Indexed {len(trees)} documents with {len(all_items)} extracted items")
            with st.expander("📊 Detected Materials and Parameters", expanded=True):
                pq_counts = kg.get_all_physical_quantities()
                if pq_counts:
                    st.write("**Physical Quantities:**")
                    for pq, count in sorted(pq_counts.items(), key=lambda x: x[1], reverse=True):
                        st.write(f"- `{pq}`: {count} occurrences")
                mat_dict = kg.get_all_materials()
                if mat_dict:
                    st.write("**Materials/Alloys per document:**")
                    for doc, mats in mat_dict.items():
                        if mats:
                            st.write(f"- {doc}: {', '.join(mats)}")

    if st.session_state.annotated_trees:
        st.markdown("### ⚡ Default Query (Laser Power & Scan Speed)")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Run Default Query", use_container_width=True):
                st.session_state.default_query = "Find out the laser power and/or scan speed for materials / alloys / compounds in the documents."
                st.rerun()
        with col2:
            prompt_input = st.chat_input("Or ask a custom query...", key="chat_input")
        default_query = st.session_state.get("default_query", "")
        if default_query:
            prompt_input = default_query
            st.session_state.default_query = ""

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
            has_valid_cache = cached and cached.get("prompt") == active_prompt and "answer" in cached
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
                # Two-stage retrieval (vectorless)
                if st.session_state.two_stage_retriever is not None:
                    progress.text("Stage 1: Metadata filtering...")
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
                        phys_q = item.physical_quantity or synthesizer.phys_classifier.classify(item.parameter_name, item.unit, item.context)
                        extracted_values.append(ExtractedValue(query=active_prompt, value=item.value, unit=item.unit or "", physical_quantity=phys_q, parameter_name=item.parameter_name, material=item.material, confidence=item.confidence, context=item.context, doc_name=item.doc_source, page=item.page, section_title=item.section_title))
                if extracted_values:
                    report = QueryReport(query=active_prompt, total_docs=len(st.session_state.annotated_trees), docs_with_results=len(set(v.doc_name for v in extracted_values)), all_values=extracted_values, consensus={}, processing_time_sec=0.0)
                    answer = synthesizer.generate_human_conclusion(active_prompt, report)
                else:
                    answer = synthesizer.synthesize(active_prompt, items)
                progress.progress(1.0, text="Done!")
                st.markdown(answer)
                st.session_state.cached_query_result = {"prompt": active_prompt, "retrieved": retrieved, "items": [i.model_dump() for i in items], "extracted_values": [v.model_dump() for v in extracted_values], "answer": answer}
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
                    st.info("Click 'Run Default Query' or ask a question about laser power and scan speed.")
                    return

        # ========== VISUALIZATION DASHBOARD (focused on laser power, scan speed, materials) ==========
        st.markdown("---")
        st.subheader("📊 Visualization Dashboard: Laser Power & Scan Speed per Material")

        if extracted_values:
            df_viz = pd.DataFrame([{
                "doc_name": v.doc_name,
                "material": v.material or "Unknown",
                "physical_quantity": v.physical_quantity,
                "value": v.value,
                "unit": v.unit,
                "confidence": v.confidence
            } for v in extracted_values])
            viz_engine = PublicationVisualizationEngine(st.session_state.knowledge_graph,
                default_colormap=st.session_state.get("viz_colormap", "viridis"))

            tabs = st.tabs(["📊 Bar Charts", "🕸️ Network Graph", "☀️ Sunburst", "🔍 Radar & Heatmap"])
            with tabs[0]:
                col1, col2 = st.columns(2)
                with col1:
                    fig_power = viz_engine.plot_laser_power_by_material(df_viz)
                    st.plotly_chart(fig_power, use_container_width=True)
                with col2:
                    fig_speed = viz_engine.plot_scan_speed_by_material(df_viz)
                    st.plotly_chart(fig_speed, use_container_width=True)
            with tabs[1]:
                fig_network = viz_engine.plot_knowledge_network(df_viz)
                st.pyplot(fig_network)
                if PYVIS_AVAILABLE:
                    st.markdown("### Interactive PyVis Network")
                    # Build network for PyVis
                    G = nx.Graph()
                    for _, row in df_viz.iterrows():
                        doc = row["doc_name"]
                        mat = row["material"] if pd.notna(row["material"]) else "Unknown"
                        param = row["physical_quantity"]
                        G.add_node(doc, group="doc")
                        G.add_node(mat, group="material")
                        G.add_node(param, group="param")
                        G.add_edge(doc, mat)
                        G.add_edge(mat, param)
                    net = Network(height="600px", width="100%", bgcolor="#ffffff")
                    for node in G.nodes():
                        grp = G.nodes[node].get("group", "unknown")
                        color = {"doc": "#1e40af", "material": "#10b981", "param": "#f59e0b"}.get(grp, "#6b7280")
                        net.add_node(node, label=node[:25], color=color)
                    for u, v in G.edges():
                        net.add_edge(u, v)
                    html = net.generate_html()
                    st.components.v1.html(html, height=650, scrolling=True)
                    st.download_button("📥 Download Network HTML", html.encode(), "network.html", "text/html")
            with tabs[2]:
                fig_sun = viz_engine.plot_material_count_sunburst(df_viz)
                st.plotly_chart(fig_sun, use_container_width=True)
            with tabs[3]:
                col1, col2 = st.columns(2)
                with col1:
                    fig_radar = viz_engine.plot_radar_by_material(df_viz)
                    st.plotly_chart(fig_radar, use_container_width=True)
                with col2:
                    param_choice = st.selectbox("Select parameter for contradiction heatmap", ["laser_power", "scan_speed"])
                    fig_contra = viz_engine.plot_contradiction_heatmap(df_viz, param_choice)
                    st.plotly_chart(fig_contra, use_container_width=True)

        # Display raw extracted values table
        st.markdown("### 📋 Extracted Values (Table)")
        if extracted_values:
            df_table = pd.DataFrame([{"Document": v.doc_name, "Page": v.page, "Material": v.material or "", "Parameter": v.physical_quantity, "Value": f"{v.value:.2f}", "Unit": v.unit} for v in extracted_values])
            st.dataframe(df_table, use_container_width=True)
            csv = df_table.to_csv(index=False).encode()
            st.download_button("📥 Download CSV", csv, "extracted_data.csv", "text/csv")
        else:
            st.info("No quantitative values extracted yet. Run a query that asks for laser power or scan speed.")

        # Tree navigation and raw items
        if st.session_state.get("show_tree_nav") and retrieved:
            with st.expander("🌳 Tree Navigation Trace", expanded=False):
                for r in retrieved[:5]:
                    st.markdown(f"**{r['doc_id']}** → p.{r['page_start']}")
        if items:
            with st.expander("🔍 Raw Extraction Items", expanded=False):
                st.json([i.to_dict() for i in items[:10]])

        # Cleanup
        if "index" in st.session_state.query_processor:
            st.session_state.query_processor["index"].cleanup()
    else:
        st.info("👆 Upload PDF files and click 'Build Index' to start.")

def fast_json_dumps(obj: Any, indent: bool = False) -> bytes:
    if ORJSON_AVAILABLE:
        option = orjson.OPT_INDENT_2 if indent else 0
        return orjson.dumps(obj, option=option, default=str)
    return json.dumps(obj, indent=2 if indent else None, ensure_ascii=False, default=str).encode()

def fast_json_loads(data: Union[bytes, str]) -> Any:
    if ORJSON_AVAILABLE:
        if isinstance(data, str):
            data = data.encode()
        return orjson.loads(data)
    if isinstance(data, bytes):
        data = data.decode()
    return json.loads(data)

if __name__ == "__main__":
    run_streamlit()
