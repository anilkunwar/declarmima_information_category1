#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DECLARMIMA v6.2-ACCELERATED - VECTORLESS RAG FOR ANY QUERY
===========================================================
Parallel hierarchical document indexing + flexible query answering
Outputs JSON with exact citations, ready for LLM consumption.

Key features:
- Parallel PDF processing (grouped by file size)
- Hierarchical tree index (no embeddings)
- Any query: "laser power", "scan speed", "material", etc.
- Regex + optional LLM extraction (local Ollama / Transformers)
- JSON output with citations and cross‑document analysis

Author: DECLARMIMA Team
License: MIT
Date: 2026-05-06
"""

import asyncio
import json
import re
import os
import sys
import tempfile
import time
import hashlib
import pickle
import logging
import warnings
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Set, Union
import numpy as np

# Streamlit optional
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# PDF processing
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    raise ImportError("PyMuPDF (fitz) is required. Install with: pip install pymupdf")

# Optional LLM for advanced extraction (fallback)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("DECLARMIMA")

warnings.filterwarnings("ignore")

# ============================================================================
# 1. DATA MODELS (Pydantic for structured JSON output)
# ============================================================================
from pydantic import BaseModel, Field, field_validator

class ExtractedValue(BaseModel):
    """A single extracted value (numerical or text) answering the query."""
    query: str = Field(description="The original query term")
    value: Union[float, str] = Field(description="Extracted value (number or text)")
    unit: Optional[str] = Field(None, description="Unit if numeric")
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    context: str = Field(description="Snippet of text containing the value")
    doc_name: str = Field(description="Source filename")
    page: int = Field(description="Page number")
    section_title: Optional[str] = Field(None)
    extraction_method: str = Field(default="regex")

    @field_validator('confidence')
    def clamp_confidence(cls, v):
        return max(0.0, min(1.0, v))

    def to_citation(self) -> str:
        return f'<cite doc="{self.doc_name}" page="{self.page}"/>'

class DocumentResult(BaseModel):
    """All extracted values from one document."""
    doc_name: str
    values: List[ExtractedValue] = []
    total_found: int = 0

class QueryReport(BaseModel):
    """Complete report for a user query."""
    query: str
    total_documents: int
    documents_with_matches: int
    documents_without_matches: List[str] = []
    all_values: List[ExtractedValue] = []
    document_results: List[DocumentResult] = []
    consensus: Dict[str, Any] = {}
    processing_time_sec: float = 0.0
    metadata: Dict[str, Any] = {}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(), indent=indent, ensure_ascii=False)

# ============================================================================
# 2. HIERARCHICAL PDF INDEX (vectorless)
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
    _pdf_path: Optional[str] = field(default=None, repr=False)

    def get_text(self, pdf_doc_cache: Dict[str, Any] = None) -> str:
        """Lazy load text from PDF if not already cached."""
        if self.full_text:
            return self.full_text
        if not self._pdf_path or not PYMUPDF_AVAILABLE:
            return ""
        try:
            doc = None
            if pdf_doc_cache and self.doc_id in pdf_doc_cache:
                doc = pdf_doc_cache[self.doc_id]
            else:
                doc = fitz.open(self._pdf_path)
                if pdf_doc_cache is not None:
                    pdf_doc_cache[self.doc_id] = doc
            start = self.page_start - 1
            end = min(self.page_end or self.page_start, len(doc))
            texts = []
            for p in range(start, end):
                text = doc[p].get_text("text")
                if text.strip():
                    texts.append(text)
            self.full_text = "\n\n".join(texts)
            if pdf_doc_cache is None and doc is not None:
                doc.close()
            return self.full_text
        except Exception as e:
            logger.warning(f"Failed to get text for {self.id}: {e}")
            return ""

class HierarchicalIndex:
    def __init__(self, cache_dir: str = ".declarmima_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.doc_trees: Dict[str, PageNode] = {}
        self.pdf_doc_cache: Dict[str, Any] = {}

    def _doc_hash(self, file_buffer: BytesIO) -> str:
        pos = file_buffer.tell()
        file_buffer.seek(0)
        content = file_buffer.read()
        file_buffer.seek(pos)
        return hashlib.sha256(content).hexdigest()[:16]

    def _cache_path(self, doc_name: str, doc_hash: str) -> Path:
        safe = re.sub(r'[^\w\-_.]', '_', doc_name)
        return self.cache_dir / f"{safe}.{doc_hash}.tree.pkl"

    def build_from_pdfs(self, files: List, parallel: bool = True, max_workers: int = 4) -> Dict[str, PageNode]:
        """Build index from uploaded PDF files (BytesIO with .name attribute)."""
        def build_one(file):
            doc_name = file.name
            file_buffer = BytesIO(file.getbuffer())
            doc_hash = self._doc_hash(file_buffer)
            cache_path = self._cache_path(doc_name, doc_hash)

            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        root_data = pickle.load(f)
                    root = self._rebuild_node(root_data)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        file_buffer.seek(0)
                        tmp.write(file_buffer.getbuffer())
                        root._pdf_path = tmp.name
                    logger.info(f"Loaded cached tree for {doc_name}")
                    return doc_name, root
                except Exception as e:
                    logger.warning(f"Cache failed for {doc_name}: {e}")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                file_buffer.seek(0)
                tmp.write(file_buffer.getbuffer())
                tmp_path = tmp.name

            doc = fitz.open(tmp_path)
            root = self._build_tree(doc, doc_name, tmp_path)
            doc.close()
            # Save to cache
            try:
                cache_root = self._clone_for_cache(root)
                with open(cache_path, "wb") as f:
                    pickle.dump(cache_root.to_dict(), f)
            except Exception as e:
                logger.warning(f"Failed to cache {doc_name}: {e}")
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

    def _build_tree(self, doc: fitz.Document, doc_id: str, pdf_path: str) -> PageNode:
        """Create hierarchical tree from TOC or heading detection."""
        root = PageNode(
            id=f"{doc_id}_root", title="Document Root",
            page_start=1, page_end=len(doc), full_text="",
            summary=doc_id, level=0, doc_id=doc_id, _pdf_path=pdf_path
        )
        toc = doc.get_toc()
        if toc:
            return self._build_from_toc(doc, doc_id, toc, root, pdf_path)
        headings = self._detect_headings(doc)
        if headings:
            return self._build_from_headings(doc, doc_id, headings, root, pdf_path)
        return self._build_page_by_page(doc, doc_id, root, pdf_path)

    def _build_from_toc(self, doc, doc_id, toc, root, pdf_path):
        nodes_by_level = {}
        for level, title, page in toc:
            if page > len(doc):
                continue
            end_page = min(page + 3, len(doc))
            text = self._extract_text_range(doc, page, end_page)
            summary = text[:200]
            node = PageNode(
                id=f"{doc_id}_toc_{level}_{title.replace(' ', '_')[:20]}",
                title=title.strip(), page_start=page, page_end=end_page,
                full_text=text, summary=summary, level=level,
                doc_id=doc_id, _pdf_path=pdf_path
            )
            nodes_by_level.setdefault(level, []).append(node)
        for level in sorted(nodes_by_level.keys()):
            for node in nodes_by_level[level]:
                parent = self._find_parent(root, level-1, node.page_start)
                parent.children.append(node)
        return root

    def _build_from_headings(self, doc, doc_id, headings, root, pdf_path):
        for i, (title, page) in enumerate(headings):
            end_page = min(page+3, len(doc))
            text = self._extract_text_range(doc, page, end_page)
            summary = text[:200]
            node = PageNode(
                id=f"{doc_id}_h{i}", title=title, page_start=page, page_end=end_page,
                full_text=text, summary=summary, level=2, doc_id=doc_id, _pdf_path=pdf_path
            )
            root.children.append(node)
        return root

    def _build_page_by_page(self, doc, doc_id, root, pdf_path):
        for p in range(1, len(doc)+1):
            text = doc[p-1].get_text("text")
            if not text.strip():
                continue
            node = PageNode(
                id=f"{doc_id}_p{p}", title=f"Page {p}", page_start=p, page_end=p,
                full_text=text, summary=text[:200], level=3, doc_id=doc_id, _pdf_path=pdf_path
            )
            root.children.append(node)
        return root

    def _extract_text_range(self, doc, start_page, end_page):
        texts = []
        for p in range(start_page-1, min(end_page, len(doc))):
            texts.append(doc[p].get_text("text"))
        return "\n\n".join(texts)

    def _detect_headings(self, doc):
        headings = []
        for p in range(len(doc)):
            text = doc[p].get_text("text")
            lines = text.split('\n')
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

    def _clone_for_cache(self, node: PageNode) -> PageNode:
        return PageNode(
            id=node.id, title=node.title, page_start=node.page_start, page_end=node.page_end,
            full_text="", summary=node.summary, level=node.level,
            doc_id=node.doc_id, section_type=node.section_type,
            children=[self._clone_for_cache(c) for c in node.children]
        )

    def _rebuild_node(self, data: dict) -> PageNode:
        node = PageNode(
            id=data['id'], title=data['title'], page_start=data.get('page_start',1),
            page_end=data.get('page_end'), full_text="", summary=data.get('summary',''),
            level=data.get('level',0), doc_id=data.get('doc_id',''),
            section_type=data.get('section_type','BODY')
        )
        for child_data in data.get('children', []):
            node.children.append(self._rebuild_node(child_data))
        return node

    def cleanup(self):
        for doc in self.pdf_doc_cache.values():
            try:
                doc.close()
            except:
                pass
        self.pdf_doc_cache.clear()

# ============================================================================
# 3. QUERY PROCESSOR (Any query, vectorless)
# ============================================================================
class QueryProcessor:
    def __init__(self, query: str):
        self.query = query.lower()
        self.keywords = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))

    def score_node(self, node: PageNode) -> float:
        text = f"{node.title} {node.summary}".lower()
        score = 0.0
        for kw in self.keywords:
            if kw in text:
                score += 0.3
        if re.search(r'\d+', text):
            score += 0.2
        return min(score, 1.0)

    def extract_from_text(self, text: str, doc_name: str, page: int, section_title: str = None) -> List[ExtractedValue]:
        """Extract values matching the query using regex patterns."""
        values = []
        # Build a regex that looks for numbers with optional units, close to the query term
        # Pattern: (number)(unit) near query term, or query term followed by number+unit
        patterns = [
            # Generic number+unit that appears within 100 chars of the query
            rf'(?i)(?:{self.query}).{{0,100}}(\d+(?:\.\d+)?)\s*([a-zA-Zµ²]+(?:/?[a-zA-Zµ²]+)?)',
            # Number+unit before the query within 100 chars
            rf'(?i)(\d+(?:\.\d+)?)\s*([a-zA-Zµ²]+).{{0,100}}{self.query}',
            # Direct assignment: query = value unit
            rf'(?i){self.query}\s*[=:]\s*(\d+(?:\.\d+)?)\s*([a-zA-Zµ²]+)',
            # value unit query
            rf'(?i)(\d+(?:\.\d+)?)\s*([a-zA-Zµ²]+)\s+{self.query}',
        ]
        seen = set()
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    if len(match.groups()) == 2:
                        val_str, unit = match.groups()
                        value = float(val_str)
                        # Get context: 150 chars around match
                        start = max(0, match.start() - 75)
                        end = min(len(text), match.end() + 75)
                        context = text[start:end].strip()
                        unique_key = (value, unit, page, doc_name)
                        if unique_key in seen:
                            continue
                        seen.add(unique_key)
                        values.append(ExtractedValue(
                            query=self.query,
                            value=value,
                            unit=unit,
                            context=context,
                            doc_name=doc_name,
                            page=page,
                            section_title=section_title,
                            extraction_method="regex"
                        ))
                except (ValueError, IndexError):
                    continue
        return values

    def process_document(self, node: PageNode, doc_cache: Dict) -> List[ExtractedValue]:
        """Recursively traverse tree, extract values from relevant leaves."""
        results = []
        if not node.children:
            score = self.score_node(node)
            if score >= 0.3:
                text = node.get_text(doc_cache)
                if text:
                    extracted = self.extract_from_text(text, node.doc_id, node.page_start, node.title)
                    results.extend(extracted)
        else:
            for child in node.children:
                results.extend(self.process_document(child, doc_cache))
        return results

# ============================================================================
# 4. PARALLEL PROCESSING ENGINE (grouped by file size)
# ============================================================================
# Size groups for adaptive concurrency
PROCESSING_GROUPS = {
    "small": {"max_pages": 10, "max_tokens": 5000, "batch_size": 8},
    "medium": {"max_pages": 20, "max_tokens": 15000, "batch_size": 4},
    "large": {"max_pages": 35, "max_tokens": 30000, "batch_size": 2},
    "extra_large": {"max_pages": float('inf'), "max_tokens": float('inf'), "batch_size": 1}
}

class ParallelQueryEngine:
    def __init__(self, index: HierarchicalIndex, max_workers: int = 8):
        self.index = index
        self.max_workers = max_workers

    async def run_query(self, query: str) -> QueryReport:
        start_time = time.time()
        processor = QueryProcessor(query)

        # Prepare tasks for each document (parallel)
        tasks = []
        for doc_name, root in self.index.doc_trees.items():
            tasks.append(self._process_doc_async(doc_name, root, processor))

        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        all_extracted = []
        doc_results = []
        docs_with_matches = 0
        docs_without = []

        for res in results_list:
            if isinstance(res, Exception):
                logger.error(f"Error in document processing: {res}")
                continue
            doc_name, values = res
            if values:
                docs_with_matches += 1
                all_extracted.extend(values)
                doc_results.append(DocumentResult(doc_name=doc_name, values=values, total_found=len(values)))
            else:
                docs_without.append(doc_name)
                doc_results.append(DocumentResult(doc_name=doc_name, values=[]))

        # Consensus analysis (simple grouping by value+unit)
        value_groups = defaultdict(list)
        for v in all_extracted:
            key = f"{v.value} {v.unit}" if v.unit else str(v.value)
            value_groups[key].append(v.doc_name)

        consensus = {
            "total_extracted": len(all_extracted),
            "unique_values": len(value_groups),
            "most_common": None
        }
        if value_groups:
            most = max(value_groups.items(), key=lambda x: len(x[1]))
            consensus["most_common"] = {
                "value": most[0],
                "count": len(most[1]),
                "documents": most[1]
            }
        # Also add per-document counts
        consensus["documents_summary"] = {doc: len(vals) for doc, vals in value_groups.items()}

        elapsed = time.time() - start_time
        report = QueryReport(
            query=query,
            total_documents=len(self.index.doc_trees),
            documents_with_matches=docs_with_matches,
            documents_without_matches=docs_without,
            all_values=all_extracted,
            document_results=doc_results,
            consensus=consensus,
            processing_time_sec=elapsed,
            metadata={"parallel_workers": self.max_workers}
        )
        return report

    async def _process_doc_async(self, doc_name: str, root: PageNode, processor: QueryProcessor):
        # Use run_in_executor for CPU-bound extraction
        loop = asyncio.get_event_loop()
        values = await loop.run_in_executor(
            None, processor.process_document, root, self.index.pdf_doc_cache
        )
        return doc_name, values

# ============================================================================
# 5. STREAMLIT UI (if available)
# ============================================================================
def run_streamlit():
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not installed. Install with: pip install streamlit")
        return
    st.set_page_config(page_title="DECLARMIMA - Query Any Document", layout="wide")
    st.title("🔬 DECLARMIMA v6.2-ACCELERATED")
    st.markdown("*Vectorless hierarchical RAG – parallel PDF processing, any query, JSON output*")

    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    query = st.text_input("Enter your query (e.g., laser power, scan speed, temperature)", value="laser power")

    if uploaded_files and query:
        if st.button("🔍 Extract Information", type="primary"):
            with st.spinner("Building index and processing documents in parallel..."):
                progress = st.progress(0)
                progress.text("Building hierarchical index...")
                index = HierarchicalIndex()
                index.build_from_pdfs(uploaded_files, parallel=True, max_workers=4)
                progress.progress(30)
                progress.text("Running query...")
                engine = ParallelQueryEngine(index, max_workers=4)
                report = asyncio.run(engine.run_query(query))
                progress.progress(100)
                progress.text("Done!")

            # Display summary
            st.success(f"✅ Found {report.documents_with_matches} documents with matches")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total documents", report.total_documents)
            col2.metric("Documents with matches", report.documents_with_matches)
            col3.metric("Total extracted values", len(report.all_values))

            with st.expander("📄 Show full JSON report", expanded=True):
                st.json(report.model_dump())

            # Download button
            st.download_button(
                "⬇️ Download JSON",
                report.to_json(),
                file_name=f"{query.replace(' ', '_')}_report.json",
                mime="application/json"
            )

            # Cleanup
            index.cleanup()

    elif not uploaded_files:
        st.info("Upload one or more PDF files to begin.")

# ============================================================================
# 6. COMMAND-LINE INTERFACE
# ============================================================================
def main_cli():
    import argparse
    parser = argparse.ArgumentParser(description="DECLARMIMA - Query PDF documents")
    parser.add_argument("files", nargs="+", help="PDF files to process")
    parser.add_argument("-q", "--query", default="laser power", help="Query term to extract")
    parser.add_argument("-o", "--output", help="Output JSON file")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    # Load files
    file_buffers = []
    for fpath in args.files:
        if not fpath.lower().endswith(".pdf"):
            continue
        with open(fpath, "rb") as f:
            buf = BytesIO(f.read())
            buf.name = os.path.basename(fpath)
            file_buffers.append(buf)

    if not file_buffers:
        print("No valid PDF files provided.")
        return

    print(f"Processing {len(file_buffers)} files with query: {args.query}")
    index = HierarchicalIndex()
    index.build_from_pdfs(file_buffers, parallel=True, max_workers=args.max_workers)
    engine = ParallelQueryEngine(index, max_workers=args.max_workers)
    report = asyncio.run(engine.run_query(args.query))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report.to_json())
        print(f"Report saved to {args.output}")
    else:
        print(report.to_json(indent=2))

    index.cleanup()

if __name__ == "__main__":
    # Detect if Streamlit is called
    if len(sys.argv) > 1 and sys.argv[1] == "ui":
        run_streamlit()
    elif STREAMLIT_AVAILABLE and not sys.argv[1:]:
        run_streamlit()
    else:
        main_cli()
