#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DECLARMIMA v7.3 – Persistent Index + Reliable Extraction
=========================================================
- Stores uploaded files and index in session state.
- Reuses index for multiple queries (no re-upload, no rebuild).
- Extracts laser power, etc., with high precision.
"""

import streamlit as st
import asyncio
import json
import re
import os
import tempfile
import time
import hashlib
import pickle
import logging
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# PDF processing
try:
    import fitz
except ImportError:
    raise ImportError("PyMuPDF (fitz) required: pip install pymupdf")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("DECLARMIMA")

# ============================================================================
# Pydantic Models
# ============================================================================
from pydantic import BaseModel, Field, field_validator

class ExtractedValue(BaseModel):
    query: str
    value: float
    unit: str
    confidence: float = Field(ge=0.0, le=1.0)
    context: str
    doc_name: str
    page: int
    section_title: Optional[str] = None
    material: Optional[str] = None

    @field_validator('value')
    def non_zero(cls, v):
        if v == 0.0:
            raise ValueError("Zero values are always false positives")
        return v

    @field_validator('unit')
    def valid_unit(cls, v):
        allowed = {"W", "kW", "mW", "MW", "W/cm", "kW/cm", "W/cm²", "kW/cm²",
                   "W/cm2", "kW/cm2", "W/m²", "kW/m²", "W/m2", "kW/m2"}
        if not any(v.startswith(u) for u in allowed):
            raise ValueError(f"Invalid unit: {v}")
        return v

    def citation(self) -> str:
        return f'<cite doc="{self.doc_name}" page="{self.page}"/>'

class DocumentResult(BaseModel):
    doc_name: str
    values: List[ExtractedValue] = []

class QueryReport(BaseModel):
    query: str
    total_docs: int
    docs_with_results: int
    docs_without_results: List[str]
    all_values: List[ExtractedValue]
    document_results: List[DocumentResult]
    consensus: Dict[str, Any]
    processing_time_sec: float

    def to_json(self, indent=2) -> str:
        return json.dumps(self.model_dump(), indent=indent, ensure_ascii=False)

# ============================================================================
# Hierarchical Index (Vectorless) – same as before
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
        return {"id": self.id, "title": self.title, "page_start": self.page_start,
                "page_end": self.page_end, "summary": self.summary, "level": self.level,
                "doc_id": self.doc_id, "section_type": self.section_type,
                "children": [c.to_dict() for c in self.children]}

    @classmethod
    def from_dict(cls, data: dict, pdf_path=None):
        node = cls(data["id"], data["title"], data["page_start"], data["page_end"],
                   "", data["summary"], data["level"], doc_id=data["doc_id"],
                   section_type=data["section_type"], _pdf_path=pdf_path)
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
        content = file_buffer.read(1024*1024)
        file_buffer.seek(pos)
        return hashlib.sha256(content).hexdigest()[:16]

    def _cache_path(self, doc_name: str, doc_hash: str) -> Path:
        safe = re.sub(r'[^\w\-_.]', '_', doc_name)
        return self.cache_dir / f"{safe}.{doc_hash}.tree.pkl"

    def build_from_pdfs(self, files: List, parallel=True, max_workers=4):
        def build_one(file):
            doc_name = file.name
            buf = BytesIO(file.getbuffer())
            doc_hash = self._doc_hash(buf)
            cache_path = self._cache_path(doc_name, doc_hash)
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        root_data = pickle.load(f)
                    root = PageNode.from_dict(root_data)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        buf.seek(0)
                        tmp.write(buf.getbuffer())
                        root._pdf_path = tmp.name
                    return doc_name, root
                except:
                    pass
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
                    pickle.dump(cache_root.to_dict(), f)
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
        root = PageNode(f"{doc_id}_root", "Document Root", 1, len(doc), "", doc_id, 0, doc_id=doc_id, _pdf_path=pdf_path)
        toc = doc.get_toc()
        if toc:
            nodes_by_level = {}
            for level, title, page in toc:
                if page > len(doc): continue
                end = min(page+3, len(doc))
                text = self._extract_range(doc, page, end)
                node = PageNode(f"{doc_id}_toc_{level}_{title[:20]}", title.strip(), page, end,
                                text, text[:200], level, doc_id=doc_id, _pdf_path=pdf_path)
                nodes_by_level.setdefault(level, []).append(node)
            for level in sorted(nodes_by_level.keys()):
                for node in nodes_by_level[level]:
                    parent = self._find_parent(root, level-1, node.page_start)
                    parent.children.append(node)
            return root
        headings = self._detect_headings(doc)
        if headings:
            for i, (title, page) in enumerate(headings):
                end = min(page+3, len(doc))
                text = self._extract_range(doc, page, end)
                node = PageNode(f"{doc_id}_h{i}", title, page, end, text, text[:200], 2, doc_id=doc_id, _pdf_path=pdf_path)
                root.children.append(node)
            return root
        for p in range(1, len(doc)+1):
            text = doc[p-1].get_text("text")
            if not text.strip(): continue
            node = PageNode(f"{doc_id}_p{p}", f"Page {p}", p, p, text, text[:200], 3, doc_id=doc_id, _pdf_path=pdf_path)
            root.children.append(node)
        return root

    def _extract_range(self, doc, start, end):
        return "\n\n".join(doc[p-1].get_text("text") for p in range(start, min(end, len(doc))))

    def _detect_headings(self, doc):
        headings = []
        for p in range(len(doc)):
            lines = doc[p].get_text("text").split('\n')
            for line in lines:
                if re.match(r'^(?:[0-9]+\.?)+ +[A-Z]', line.strip()):
                    headings.append((line.strip(), p+1))
        return headings[:50]

    def _find_parent(self, node, target_level, page_hint):
        if target_level < 0:
            return node
        candidates = [c for c in node.children if c.level == target_level]
        if not candidates:
            return node
        return min(candidates, key=lambda n: abs(n.page_start - page_hint))

    def _clone_for_cache(self, node):
        return PageNode(node.id, node.title, node.page_start, node.page_end, "",
                        node.summary, node.level, doc_id=node.doc_id,
                        section_type=node.section_type,
                        children=[self._clone_for_cache(c) for c in node.children])

    def cleanup(self):
        for doc in self._pdf_cache.values():
            try: doc.close()
            except: pass
        self._pdf_cache.clear()

# ============================================================================
# Robust Extractor (handles real-world variations)
# ============================================================================
class PreciseExtractor:
    def __init__(self, query: str):
        self.query = query.lower()
        self.allowed_units = {"W", "kW", "mW", "MW", "W/cm", "kW/cm", "W/cm²", "kW/cm²",
                              "W/cm2", "kW/cm2", "W/m²", "kW/m²", "W/m2", "kW/m2"}
        # Expanded patterns – captures "250W", "250 W", etc.
        self.patterns = [
            rf'(?:{self.query})\s*[=:]\s*(\d+(?:\.\d+)?)\s*([a-zA-Z²0-9/]+)',
            rf'(?:{self.query})\s+of\s+(\d+(?:\.\d+)?)\s*([a-zA-Z²0-9/]+)',
            rf'(\d+(?:\.\d+)?)\s*([a-zA-Z²0-9/]+)\s+(?:{self.query})',
            r'power\s*[=:]\s*(\d+(?:\.\d+)?)\s*([WkWMm]{1,3})',
            r'P\s*[=:]\s*(\d+(?:\.\d+)?)\s*([WkWMm]{1,3})',
            # direct "250W" (no space) or "250 W" (non-breaking space)
            r'(\d+(?:\.\d+)?)([WkWMm]{1,3})\b',
            r'(\d+(?:\.\d+)?)\s*[  ]?\s*([WkWMm]{1,3})\b',
        ]

    def _is_valid(self, value: float, unit: str, context: str) -> bool:
        if value == 0.0:
            return False
        if not any(unit.startswith(u) for u in self.allowed_units):
            return False
        # Context must contain query OR "laser" (for power queries)
        lower_ctx = context.lower()
        if self.query not in lower_ctx and "laser" not in lower_ctx:
            return False
        # Also require a power-related phrase (avoids false positives like "250 µm")
        power_phrases = ["power", "input power", "laser power", "beam power", "P =", "power =", "irradiance"]
        if not any(phrase in lower_ctx for phrase in power_phrases):
            return False
        return True

    def extract_from_text(self, text: str, doc_name: str, page: int, section_title: str) -> List[ExtractedValue]:
        results = []
        for pattern in self.patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    groups = match.groups()
                    if len(groups) == 2:
                        val_str, unit = groups
                    else:
                        val_str = groups[0]
                        unit = ""
                    value = float(val_str)
                    unit = unit.strip().upper().replace("²", "2")
                    # Context window (±80 characters)
                    start = max(0, match.start() - 80)
                    end = min(len(text), match.end() + 80)
                    context = text[start:end].strip()
                    if not self._is_valid(value, unit, context):
                        continue
                    # Extract exact sentence
                    sentences = re.split(r'(?<=[.!?])\s+', text)
                    exact = ""
                    for sent in sentences:
                        if match.group(0) in sent:
                            exact = sent.strip()
                            break
                    if not exact:
                        exact = context[:200]
                    # Simple material heuristic
                    material = None
                    for mat in ["AlSiMg", "SDSS", "Ti6Al4V", "Ti-Cr", "Cu6Sn5", "Al-Cu-Ni", "Ti3Au", "Inconel"]:
                        if mat in exact:
                            material = mat
                            break
                    results.append(ExtractedValue(
                        query=self.query,
                        value=value,
                        unit=unit,
                        confidence=0.95,
                        context=exact,
                        doc_name=doc_name,
                        page=page,
                        section_title=section_title,
                        material=material
                    ))
                except (ValueError, IndexError):
                    continue
        # Deduplicate by (value, unit, page, doc)
        unique = {}
        for v in results:
            key = (v.value, v.unit, v.page, v.doc_name)
            if key not in unique or v.confidence > unique[key].confidence:
                unique[key] = v
        return list(unique.values())

# ============================================================================
# Parallel Query Engine
# ============================================================================
class ParallelQueryEngine:
    def __init__(self, index: HierarchicalIndex, max_workers=8):
        self.index = index
        self.max_workers = max_workers

    async def run_query(self, query: str) -> QueryReport:
        start = time.time()
        extractor = PreciseExtractor(query)
        tasks = []
        for doc_name, root in self.index.doc_trees.items():
            tasks.append(self._process_doc(doc_name, root, extractor))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_vals = []
        doc_res = []
        docs_with = 0
        docs_without = []
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Document error: {res}")
                continue
            doc_name, values = res
            if values:
                docs_with += 1
                all_vals.extend(values)
                doc_res.append(DocumentResult(doc_name=doc_name, values=values))
            else:
                docs_without.append(doc_name)
                doc_res.append(DocumentResult(doc_name=doc_name, values=[]))
        # Consensus (only for power-like units)
        power_vals = [v for v in all_vals if any(v.unit.startswith(u) for u in ["W", "kW", "mW"])]
        consensus = {}
        if power_vals:
            nums = [v.value for v in power_vals]
            unit_counts = Counter(v.unit for v in power_vals)
            most_common_unit = unit_counts.most_common(1)[0][0]
            consensus = {
                "parameter": query,
                "count": len(power_vals),
                "mean": float(np.mean(nums)),
                "std": float(np.std(nums)),
                "min": float(np.min(nums)),
                "max": float(np.max(nums)),
                "unit": most_common_unit,
                "sources": list(set(v.doc_name for v in power_vals))
            }
        elapsed = time.time() - start
        return QueryReport(
            query=query,
            total_docs=len(self.index.doc_trees),
            docs_with_results=docs_with,
            docs_without_results=docs_without,
            all_values=all_vals,
            document_results=doc_res,
            consensus=consensus,
            processing_time_sec=elapsed
        )

    async def _process_doc(self, doc_name: str, root: PageNode, extractor: PreciseExtractor):
        def traverse(node: PageNode):
            vals = []
            if not node.children:
                text = node.get_text(self.index._pdf_cache)
                if text:
                    vals.extend(extractor.extract_from_text(text, doc_name, node.page_start, node.title))
            else:
                for c in node.children:
                    vals.extend(traverse(c))
            return vals
        loop = asyncio.get_event_loop()
        values = await loop.run_in_executor(None, traverse, root)
        return doc_name, values

# ============================================================================
# Streamlit UI with Session State (persistent files & index)
# ============================================================================
def run_streamlit():
    st.set_page_config(page_title="DECLARMIMA v7.3 – Persistent Extraction", layout="wide")
    st.title("🔬 DECLARMIMA v7.3 – Multi‑Query Extraction")
    st.caption("Upload your PDFs once. Ask multiple questions (e.g., 'laser power', 'scan speed', 'irradiance').")

    # ----- Initialise session state -----
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = None
    if "index_built" not in st.session_state:
        st.session_state.index_built = False
    if "index" not in st.session_state:
        st.session_state.index = None

    # ----- File uploader (stores files in session) -----
    uploaded = st.file_uploader("Upload PDF files (once)", type="pdf", accept_multiple_files=True, key="file_uploader")
    if uploaded:
        st.session_state.uploaded_files = uploaded
        # Reset index so it will be rebuilt with new files
        st.session_state.index_built = False
        st.session_state.index = None
        st.rerun()   # force rerun to show updated state

    # ----- Clear files button -----
    if st.button("🗑️ Clear all uploaded files"):
        st.session_state.uploaded_files = None
        st.session_state.index_built = False
        st.session_state.index = None
        st.rerun()

    # ----- Build index only once after files are uploaded -----
    if st.session_state.uploaded_files and not st.session_state.index_built:
        with st.spinner("Building hierarchical index (this takes a few seconds, only once)..."):
            idx = HierarchicalIndex()
            # Use parallel processing
            idx.build_from_pdfs(st.session_state.uploaded_files, parallel=True)
            st.session_state.index = idx
            st.session_state.index_built = True
        st.success("✅ Index built! You can now ask multiple questions.")
        st.rerun()

    # ----- Query input (only visible when index is ready) -----
    if st.session_state.index_built:
        query = st.text_input("Enter query (e.g., 'laser power', 'power', 'irradiance', 'scan speed')",
                              "laser power")
        if st.button("🔍 Extract", type="primary"):
            with st.spinner("Extracting values..."):
                report = asyncio.run(ParallelQueryEngine(st.session_state.index).run_query(query))
            if report.docs_with_results == 0:
                st.warning(f"No values found for '{query}'. Try 'power' or check that PDFs contain relevant numbers.")
            else:
                st.success(f"✅ {report.docs_with_results} documents contain relevant '{query}' values")
            st.json(report.model_dump())
            if report.all_values:
                st.download_button("📥 Download JSON", report.to_json(),
                                   f"{query.replace(' ', '_')}_report.json", "application/json")
    else:
        if not st.session_state.uploaded_files:
            st.info("👆 Upload your PDF files above (one time). Then the index will be built automatically.")
        else:
            st.info("Index is being built. Please wait a moment...")

# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    run_streamlit()
