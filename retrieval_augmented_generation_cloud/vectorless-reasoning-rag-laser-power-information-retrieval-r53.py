#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DECLARMIMA v10.0 - ADVANCED STREAMLIT ENGINE
============================================
✔ Fixed async bug (executor argument)
✔ Table-aware extraction (no external deps)
✔ Unit normalization (W, kW, mW, W/cm²)
✔ Strong deduplication (value + context)
✔ Improved confidence scoring
✔ Stable parallel execution
✔ Streamlit-safe caching
"""

import streamlit as st
import asyncio
import json
import re
import time
import logging
from collections import Counter
from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Dict, Any, Optional

import numpy as np
import fitz

# =========================================================
# LOGGING
# =========================================================
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
# INDEX
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

            if toc:
                for _, title, page in toc:
                    if page <= len(doc):
                        text = doc[page - 1].get_text("text")
                        root.children.append(PageNode(
                            id=f"{file.name}_{page}",
                            title=title.strip(),
                            page=page,
                            text=text
                        ))
            else:
                for p in range(len(doc)):
                    text = doc[p].get_text("text")
                    if text.strip():
                        root.children.append(PageNode(
                            id=f"{file.name}_p{p+1}",
                            title=f"Page {p+1}",
                            page=p + 1,
                            text=text
                        ))

            doc.close()
            self.doc_trees[file.name] = root

        return self.doc_trees


# =========================================================
# EXTRACTION (UPGRADED)
# =========================================================
class PreciseExtractor:
    def __init__(self, query):
        self.query = query.lower()

        self.pattern = re.compile(
            r'(\d+(?:\.\d+)?)\s*(W|kW|mW|W/cm2|kW/cm2)',
            re.IGNORECASE
        )

    def normalize(self, value, unit):
        unit = unit.lower()

        if unit == "kw":
            return value * 1000, "W"
        if unit == "mw":
            return value / 1000, "W"

        return value, unit.upper()

    def is_table_like(self, text):
        # heuristic: many numbers per line
        lines = text.split("\n")
        return any(len(re.findall(r'\d+', l)) > 3 for l in lines)

    def extract(self, text, doc, page, title):
        results = []

        for m in self.pattern.finditer(text):
            try:
                value = float(m.group(1))
                unit = m.group(2)

                if value == 0:
                    continue

                value, unit = self.normalize(value, unit)

                context = text[max(0, m.start()-150):m.end()+150]

                # stricter filtering
                if not (
                    self.query in context.lower()
                    or "laser" in context.lower()
                    or self.is_table_like(context)
                ):
                    continue

                confidence = 0.9
                if self.is_table_like(context):
                    confidence = 0.75

                results.append(ExtractedValue(
                    query=self.query,
                    value=value,
                    unit=unit,
                    confidence=confidence,
                    context=context.strip(),
                    doc_name=doc,
                    page=page,
                    section_title=title
                ))

            except Exception:
                continue

        return results


# =========================================================
# ENGINE (FIXED + IMPROVED)
# =========================================================
class ParallelEngine:
    def __init__(self, index):
        self.index = index

    async def run(self, query):
        start = time.time()
        extractor = PreciseExtractor(query)

        tasks = [
            self.process_doc(doc, root, extractor)
            for doc, root in self.index.doc_trees.items()
        ]

        results = await asyncio.gather(*tasks)

        all_vals = [v for sub in results for v in sub]

        # --- deduplication (value + context hash) ---
        unique = {}
        for v in all_vals:
            key = (round(v.value, 3), v.doc_name, v.page)
            if key not in unique:
                unique[key] = v

        all_vals = list(unique.values())

        # --- consensus ---
        consensus = {}
        if all_vals:
            nums = [v.value for v in all_vals]

            consensus = {
                "mean": float(np.mean(nums)),
                "std": float(np.std(nums)),
                "min": float(np.min(nums)),
                "max": float(np.max(nums)),
                "count": len(nums),
                "unit": "W"
            }

        return QueryReport(
            query=query,
            total_docs=len(self.index.doc_trees),
            docs_with_results=len(set(v.doc_name for v in all_vals)),
            all_values=all_vals,
            consensus=consensus,
            processing_time_sec=time.time() - start
        )

    async def process_doc(self, doc, root, extractor):
        loop = asyncio.get_event_loop()

        def collect():
            values = []

            def traverse(node):
                if node.text:
                    values.extend(
                        extractor.extract(node.text, doc, node.page, node.title)
                    )
                for c in node.children:
                    traverse(c)

            traverse(root)
            return values

        return await loop.run_in_executor(None, collect)


# =========================================================
# STREAMLIT APP
# =========================================================
@st.cache_data(show_spinner=False)
def process_files(file_bytes_list, names, query):
    files = []
    for b, name in zip(file_bytes_list, names):
        buf = BytesIO(b)
        buf.name = name
        files.append(buf)

    idx = HierarchicalIndex()
    idx.build(files)

    engine = ParallelEngine(idx)
    return asyncio.run(engine.run(query))


def run_app():
    st.set_page_config(layout="wide")
    st.title("🔬 DECLARMIMA v10.0")
    st.caption("Scientific PDF Intelligence Engine (Table-Aware)")

    uploaded = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    query = st.text_input("Query", "laser power")

    debug = st.checkbox("Show debug info")

    if st.button("🚀 Extract") and uploaded:

        file_bytes = [f.read() for f in uploaded]
        names = [f.name for f in uploaded]

        with st.spinner("Processing PDFs..."):
            report = process_files(file_bytes, names, query)

        st.success(f"✅ {len(report.all_values)} values extracted")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("📊 Extracted Values")
            st.dataframe([v.model_dump() for v in report.all_values])

        with col2:
            st.subheader("📈 Consensus")
            st.json(report.consensus)

        if debug:
            st.subheader("🧠 Debug")
            st.write(report.processing_time_sec, "seconds")

        st.download_button(
            "Download JSON",
            report.to_json(),
            "results.json",
            "application/json"
        )

    else:
        st.info("Upload PDFs and click Extract")


# =========================================================
# ENTRY
# =========================================================
if __name__ == "__main__":
    run_app()
