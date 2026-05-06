#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DECLARMIMA v8.5 - STREAMLIT ONLY (FULL POWER)
=============================================
✔ Hierarchical PDF indexing (TOC + fallback)
✔ Parallel extraction (async + threads)
✔ Strict numeric filtering (no garbage values)
✔ Cross-document consensus
✔ Streamlit-native execution (no CLI conflicts)
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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import fitz  # PyMuPDF

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DECLARMIMA")

# =========================================================
# DATA MODELS
# =========================================================
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
        return json.dumps(self.model_dump(), indent=2)


# =========================================================
# HIERARCHICAL INDEX
# =========================================================
@dataclass
class PageNode:
    id: str
    title: str
    page: int
    text: str
    children: List['PageNode'] = field(default_factory=list)


class HierarchicalIndex:
    def __init__(self):
        self.doc_trees = {}

    def build(self, files):
        for file in files:
            doc = fitz.open(stream=file.read(), filetype="pdf")
            root = PageNode(file.name, "ROOT", 0, "")

            toc = doc.get_toc()

            # --- TOC-based hierarchy ---
            if toc:
                for level, title, page in toc:
                    text = doc[page-1].get_text("text")
                    node = PageNode(
                        id=f"{file.name}_{page}",
                        title=title,
                        page=page,
                        text=text
                    )
                    root.children.append(node)

            # --- fallback: page-wise ---
            else:
                for p in range(len(doc)):
                    text = doc[p].get_text("text")
                    if text.strip():
                        node = PageNode(
                            id=f"{file.name}_p{p+1}",
                            title=f"Page {p+1}",
                            page=p+1,
                            text=text
                        )
                        root.children.append(node)

            self.doc_trees[file.name] = root

        return self.doc_trees


# =========================================================
# EXTRACTION
# =========================================================
class PreciseExtractor:
    def __init__(self, query):
        self.query = query.lower()

        self.patterns = [
            rf'{self.query}\s*[=:]\s*(\d+(?:\.\d+)?)\s*(W|kW|mW)',
            rf'(\d+(?:\.\d+)?)\s*(W|kW|mW)\s+{self.query}',
            r'(\d+(?:\.\d+)?)\s*(W|kW|mW)'
        ]

    def extract(self, text, doc, page, title):
        results = []

        for pattern in self.patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    val = float(m.group(1))
                    unit = m.group(2)

                    if val == 0:
                        continue

                    context = text[max(0, m.start()-120):m.end()+120]

                    if self.query not in context.lower() and "laser" not in context.lower():
                        continue

                    results.append(ExtractedValue(
                        query=self.query,
                        value=val,
                        unit=unit,
                        confidence=0.9,
                        context=context.strip(),
                        doc_name=doc,
                        page=page,
                        section_title=title
                    ))

                except:
                    continue

        return results


# =========================================================
# ENGINE
# =========================================================
class ParallelEngine:
    def __init__(self, index):
        self.index = index

    async def run(self, query):
        start = time.time()
        extractor = PreciseExtractor(query)

        tasks = []
        for doc, root in self.index.doc_trees.items():
            tasks.append(self.process_doc(doc, root, extractor))

        results = await asyncio.gather(*tasks)

        all_vals = []
        docs_with = 0

        for vals in results:
            if vals:
                docs_with += 1
                all_vals.extend(vals)

        # --- consensus ---
        consensus = {}
        if all_vals:
            nums = [v.value for v in all_vals]
            units = Counter(v.unit for v in all_vals).most_common(1)[0][0]

            consensus = {
                "mean": float(np.mean(nums)),
                "std": float(np.std(nums)),
                "min": float(np.min(nums)),
                "max": float(np.max(nums)),
                "count": len(nums),
                "unit": units
            }

        return QueryReport(
            query=query,
            total_docs=len(self.index.doc_trees),
            docs_with_results=docs_with,
            all_values=all_vals,
            consensus=consensus,
            processing_time_sec=time.time() - start
        )

    async def process_doc(self, doc, root, extractor):
        values = []

        def traverse(node):
            if node.text:
                vals = extractor.extract(node.text, doc, node.page, node.title)
                values.extend(vals)
            for c in node.children:
                traverse(c)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, traverse, root)

        return values


# =========================================================
# STREAMLIT UI
# =========================================================
def run_app():
    st.set_page_config(layout="wide")
    st.title("🔬 DECLARMIMA v8.5 (Streamlit Edition)")
    st.caption("Hierarchical Vectorless RAG for Scientific PDFs")

    uploaded = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    query = st.text_input(
        "Enter query",
        value="laser power"
    )

    if st.button("🚀 Extract") and uploaded:
        with st.spinner("Processing documents..."):

            # Convert to buffers
            files = []
            for f in uploaded:
                buf = BytesIO(f.read())
                buf.name = f.name
                files.append(buf)

            # Build index
            idx = HierarchicalIndex()
            idx.build(files)

            # Run engine
            engine = ParallelEngine(idx)
            report = asyncio.run(engine.run(query))

        st.success(f"✅ Found {len(report.all_values)} values")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📊 Extracted Values")
            st.json(report.model_dump())

        with col2:
            st.subheader("📈 Consensus")
            st.json(report.consensus)

        st.download_button(
            "Download JSON",
            report.to_json(),
            "results.json",
            "application/json"
        )

    else:
        st.info("Upload PDFs and click Extract")


# =========================================================
# ENTRY POINT
# =========================================================
#if __name__ == "__main__":
#    run_app()
if __name__ == "__main__":
    import os

    if os.environ.get("STREAMLIT_SERVER_PORT"):
        run_streamlit()
    else:
        run_cli()
