#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DECLARMIMA v11.0 - LLM-AWARE SCIENTIFIC ENGINE
==============================================
✔ Table + JSON display toggle
✔ LLM-assisted classification (power vs irradiance)
✔ Unit-aware physics distinction
✔ Clean UI + better outputs
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

# Optional LLM (Ollama)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except:
    OLLAMA_AVAILABLE = False

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
    quantity_type: str  # NEW
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
                        root.children.append(PageNode(
                            id=f"{file.name}_{page}",
                            title=title,
                            page=page,
                            text=doc[page-1].get_text("text")
                        ))
            else:
                for p in range(len(doc)):
                    text = doc[p].get_text("text")
                    if text.strip():
                        root.children.append(PageNode(
                            id=f"{file.name}_p{p+1}",
                            title=f"Page {p+1}",
                            page=p+1,
                            text=text
                        ))

            doc.close()
            self.doc_trees[file.name] = root

        return self.doc_trees


# =========================================================
# CLASSIFIER (RULE + LLM)
# =========================================================
class QuantityClassifier:

    def classify(self, value, unit, context):
        unit = unit.lower()

        # --- deterministic physics rules ---
        if "cm" in unit:
            return "irradiance"

        if unit in ["w", "kw", "mw"]:
            return "laser_power"

        # --- fallback: LLM ---
        if OLLAMA_AVAILABLE:
            try:
                prompt = f"""
Classify the quantity:

Value: {value} {unit}
Context: {context}

Return ONLY one:
laser_power OR irradiance OR unknown
"""
                resp = ollama.chat(
                    model="llama3",
                    messages=[{"role": "user", "content": prompt}]
                )
                return resp['message']['content'].strip().lower()
            except:
                return "unknown"

        return "unknown"


# =========================================================
# EXTRACTION
# =========================================================
class PreciseExtractor:
    def __init__(self, query):
        self.query = query.lower()
        self.classifier = QuantityClassifier()

        self.pattern = re.compile(
            r'(\d+(?:\.\d+)?)\s*(W|kW|mW|W/cm2|kW/cm2)',
            re.IGNORECASE
        )

    def normalize(self, value, unit):
        u = unit.lower()
        if u == "kw":
            return value * 1000, "W"
        if u == "mw":
            return value / 1000, "W"
        return value, unit.upper()

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

                if self.query not in context.lower() and "laser" not in context.lower():
                    continue

                qtype = self.classifier.classify(value, unit, context)

                results.append(ExtractedValue(
                    query=self.query,
                    value=value,
                    unit=unit,
                    quantity_type=qtype,
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

        tasks = [
            self.process_doc(doc, root, extractor)
            for doc, root in self.index.doc_trees.items()
        ]

        results = await asyncio.gather(*tasks)
        all_vals = [v for sub in results for v in sub]

        # --- separate by type ---
        power_vals = [v.value for v in all_vals if v.quantity_type == "laser_power"]

        consensus = {}
        if power_vals:
            consensus = {
                "laser_power_mean": float(np.mean(power_vals)),
                "count": len(power_vals),
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
            vals = []

            def traverse(node):
                if node.text:
                    vals.extend(extractor.extract(node.text, doc, node.page, node.title))
                for c in node.children:
                    traverse(c)

            traverse(root)
            return vals

        return await loop.run_in_executor(None, collect)


# =========================================================
# STREAMLIT UI
# =========================================================
@st.cache_data
def process_files(file_bytes_list, names, query):
    files = []
    for b, n in zip(file_bytes_list, names):
        buf = BytesIO(b)
        buf.name = n
        files.append(buf)

    idx = HierarchicalIndex()
    idx.build(files)

    engine = ParallelEngine(idx)
    return asyncio.run(engine.run(query))


def run_app():
    st.set_page_config(layout="wide")
    st.title("🔬 DECLARMIMA v11.0")
    st.caption("LLM-aware Scientific Extraction Engine")

    uploaded = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    query = st.text_input("Query", "laser power")

    display_mode = st.radio("Display mode", ["Table", "JSON"])

    if st.button("🚀 Extract") and uploaded:

        file_bytes = [f.read() for f in uploaded]
        names = [f.name for f in uploaded]

        report = process_files(file_bytes, names, query)

        st.success(f"Extracted {len(report.all_values)} values")

        if display_mode == "Table":
            st.dataframe([v.model_dump() for v in report.all_values])
        else:
            st.json(report.model_dump())

        st.subheader("Consensus")
        st.json(report.consensus)

        st.download_button(
            "Download JSON",
            report.to_json(),
            "results.json"
        )


if __name__ == "__main__":
    run_app()
