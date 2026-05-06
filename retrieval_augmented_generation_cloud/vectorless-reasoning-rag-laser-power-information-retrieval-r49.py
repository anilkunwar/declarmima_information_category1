#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DECLARMIMA v8.0 - CLEAN PRODUCTION VERSION
==========================================
- Streamlit UI (default when launched via streamlit)
- CLI mode (when run via python)
- Fixed argument handling
- Improved robustness
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

# PDF processing
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

    @field_validator('value')
    def non_zero_value(cls, v):
        if v == 0.0:
            raise ValueError("Zero values are ignored")
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
# PDF INDEX
# =========================================================
@dataclass
class PageNode:
    page: int
    text: str

class SimpleIndex:
    def __init__(self):
        self.docs = {}

    def build(self, files):
        for f in files:
            doc = fitz.open(stream=f.read(), filetype="pdf")
            pages = []
            for i in range(len(doc)):
                txt = doc[i].get_text("text")
                if txt.strip():
                    pages.append(PageNode(i + 1, txt))
            self.docs[f.name] = pages
        return self.docs


# =========================================================
# EXTRACTION
# =========================================================
class Extractor:
    def __init__(self, query):
        self.query = query.lower()
        self.pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(W|kW|mW)', re.IGNORECASE)

    def extract(self, text, doc, page):
        results = []
        for m in self.pattern.finditer(text):
            value = float(m.group(1))
            unit = m.group(2)

            context = text[max(0, m.start()-80):m.end()+80]

            if self.query not in context.lower() and "laser" not in context.lower():
                continue

            results.append(ExtractedValue(
                query=self.query,
                value=value,
                unit=unit,
                confidence=0.9,
                context=context.strip(),
                doc_name=doc,
                page=page
            ))
        return results


# =========================================================
# ENGINE
# =========================================================
class Engine:
    def __init__(self, index):
        self.index = index

    async def run(self, query):
        start = time.time()
        extractor = Extractor(query)

        tasks = []
        for doc, pages in self.index.docs.items():
            tasks.append(self.process_doc(doc, pages, extractor))

        results = await asyncio.gather(*tasks)

        all_vals = []
        docs_with = 0

        for vals in results:
            if vals:
                docs_with += 1
                all_vals.extend(vals)

        consensus = {}
        if all_vals:
            nums = [v.value for v in all_vals]
            consensus = {
                "mean": float(np.mean(nums)),
                "min": float(np.min(nums)),
                "max": float(np.max(nums)),
                "count": len(nums)
            }

        return QueryReport(
            query=query,
            total_docs=len(self.index.docs),
            docs_with_results=docs_with,
            all_values=all_vals,
            consensus=consensus,
            processing_time_sec=time.time() - start
        )

    async def process_doc(self, doc, pages, extractor):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: [
                v
                for p in pages
                for v in extractor.extract(p.text, doc, p.page)
            ]
        )


# =========================================================
# STREAMLIT UI
# =========================================================
def run_streamlit():
    st.set_page_config(layout="wide")
    st.title("🔬 DECLARMIMA v8.0")

    uploaded = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    query = st.text_input("Query", "laser power")

    if st.button("Run") and uploaded:
        files = [BytesIO(f.read()) for f in uploaded]
        for i, f in enumerate(files):
            f.name = uploaded[i].name

        index = SimpleIndex()
        index.build(files)

        engine = Engine(index)
        report = asyncio.run(engine.run(query))

        st.success(f"Found {len(report.all_values)} values")
        st.json(report.model_dump())

        st.download_button("Download JSON", report.to_json(), "results.json")


# =========================================================
# CLI
# =========================================================
def run_cli():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("-q", "--query", default="laser power")
    parser.add_argument("-o", "--output")

    args = parser.parse_args()

    files = []
    for path in args.files:
        with open(path, "rb") as f:
            buf = BytesIO(f.read())
            buf.name = os.path.basename(path)
            files.append(buf)

    index = SimpleIndex()
    index.build(files)

    engine = Engine(index)
    report = asyncio.run(engine.run(args.query))

    if args.output:
        with open(args.output, "w") as f:
            f.write(report.to_json())
    else:
        print(report.to_json())


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    import sys

    # Detect if running via Streamlit
    if any("streamlit" in arg for arg in sys.argv):
        run_streamlit()
    else:
        run_cli()
