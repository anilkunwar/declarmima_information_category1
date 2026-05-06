#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DECLARMIMA v8.0 - PRECISION LASER POWER EXTRACTOR
==================================================
Vectorless hierarchical RAG for accurate extraction of laser power values
from scientific PDFs. Returns a structured table with exact citations.

FEATURES:
- Strict regex patterns for laser power (W, kW, mW, W/cm²)
- Filters out zero values and invalid units
- Prioritises "Materials and Methods", "Experimental", "Results" sections
- Parallel processing, caching, Streamlit UI
- Output: JSON + Markdown table with page citations

AUTHOR: DECLARMIMA Team
LICENSE: MIT
VERSION: 8.0-PRECISION
"""

import streamlit as st
import os
import re
import json
import tempfile
import time
import hashlib
import pickle
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd

# PDF processing
try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    raise ImportError("PyMuPDF (fitz) required. Install: pip install pymupdf")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("DECLARMIMA")

# ============================================================================
# 1. Data structures
# ============================================================================
@dataclass
class LaserPowerEntry:
    """A single laser power extraction."""
    value: float
    unit: str
    page: int
    section: str
    context: str
    is_irradiance: bool = False

    def citation(self, doc_name: str) -> str:
        return f'<cite doc="{doc_name}" page="{self.page}"/>'

    def __str__(self):
        return f"{self.value} {self.unit}"

@dataclass
class DocumentResult:
    """Results for one document."""
    name: str
    entries: List[LaserPowerEntry]
    primary_power: Optional[LaserPowerEntry] = None

    def to_dict(self):
        return {
            "doc_name": self.name,
            "laser_power": str(self.primary_power) if self.primary_power else None,
            "irradiance": str([e for e in self.entries if e.is_irradiance][0]) if any(e.is_irradiance for e in self.entries) else None,
            "all_values": [(e.value, e.unit, e.page) for e in self.entries],
            "citations": [e.citation(self.name) for e in self.entries]
        }

# ============================================================================
# 2. Hierarchical document tree (simplified, only needed for lazy loading)
# ============================================================================
class PDFTree:
    def __init__(self):
        self.doc_trees = {}
        self.pdf_cache = {}

    def build(self, files: List, max_workers: int = 4):
        def build_one(file):
            doc_name = file.name
            file_buffer = BytesIO(file.getbuffer())
            # Try cache
            cache_path = Path(".declarmima_cache") / f"{hashlib.sha256(file_buffer.getbuffer()).hexdigest()[:16]}.pkl"
            cache_path.parent.mkdir(exist_ok=True)
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        tree = pickle.load(f)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            file_buffer.seek(0)
                            tmp.write(file_buffer.getbuffer())
                            tree._pdf_path = tmp.name
                        return doc_name, tree
                except:
                    pass
            # Build new
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                file_buffer.seek(0)
                tmp.write(file_buffer.getbuffer())
                tmp_path = tmp.name
            doc = fitz.open(tmp_path)
            root = self._build_tree(doc, doc_name, tmp_path)
            doc.close()
            with open(cache_path, "wb") as f:
                pickle.dump(root, f)
            return doc_name, root

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(build_one, f) for f in files]
            for fut in as_completed(futures):
                name, tree = fut.result()
                self.doc_trees[name] = tree

    def _build_tree(self, doc, doc_id, pdf_path):
        """Minimal tree: one node per page (simplest, reliable)."""
        root = PageNode(f"{doc_id}_root", "Document Root", 1, len(doc), "", doc_id, _pdf_path=pdf_path)
        for p in range(1, len(doc)+1):
            node = PageNode(f"{doc_id}_p{p}", f"Page {p}", p, p, "", doc_id, _pdf_path=pdf_path, level=3)
            root.children.append(node)
        return root

    def cleanup(self):
        for doc in self.pdf_cache.values():
            try:
                doc.close()
            except:
                pass
        self.pdf_cache.clear()

# Simple PageNode (partial)
@dataclass
class PageNode:
    id: str
    title: str
    page_start: int
    page_end: int
    full_text: str
    doc_id: str
    level: int = 3
    children: List = field(default_factory=list)
    _pdf_path: str = None

    def get_text(self, cache=None):
        if self.full_text:
            return self.full_text
        if not self._pdf_path:
            return ""
        doc = fitz.open(self._pdf_path)
        text = doc[self.page_start-1].get_text("text")
        doc.close()
        self.full_text = text
        return text

# ============================================================================
# 3. Precision Laser Power Extraction Engine
# ============================================================================
class LaserPowerExtractor:
    # Strict patterns – only capture explicit laser power statements
    PATTERNS = [
        r'laser\s+power\s*[=:]\s*(\d+(?:\.\d+)?)\s*(W|kW|mW)',
        r'power\s*[=:]\s*(\d+(?:\.\d+)?)\s*(W|kW|mW)\s+(?:laser|beam)',
        r'(\d+(?:\.\d+)?)\s*(W|kW|mW)\s+laser\s+power',
        r'laser\s+power\s+of\s*(\d+(?:\.\d+)?)\s*(W|kW|mW)',
        r'P\s*=\s*(\d+(?:\.\d+)?)\s*(W|kW|mW)',          # Common shorthand
        r'laser\s+power\s*\(\s*P\s*\)\s*=\s*(\d+(?:\.\d+)?)\s*(W|kW|mW)',
        # Irradiance (optional)
        r'irradiance\s*[=:]\s*(\d+(?:\.\d+)?)\s*(kW/cm²|kW/cm2|W/cm²|W/cm2)',
        r'power\s+density\s*[=:]\s*(\d+(?:\.\d+)?)\s*(kW/cm²|W/cm²)',
    ]

    # Sections where laser power is most likely to appear
    TARGET_SECTIONS = ["METHODS", "MATERIALS", "EXPERIMENTAL", "RESULTS", "TABLE", "FIGURE"]

    @staticmethod
    def extract_from_document(root: PageNode) -> List[LaserPowerEntry]:
        entries = []
        # Traverse only leaf nodes (pages)
        stack = [root]
        while stack:
            node = stack.pop()
            if not node.children:
                text = node.get_text()
                if not text:
                    continue
                # Quick section type detection (heuristic from first 200 chars)
                first_200 = text[:200].lower()
                section_type = "OTHER"
                if any(kw in first_200 for kw in ["methods", "experimental", "materials"]):
                    section_type = "METHODS"
                elif any(kw in first_200 for kw in ["results", "findings", "data"]):
                    section_type = "RESULTS"
                elif any(kw in first_200 for kw in ["table", "figure", "graph"]):
                    section_type = "TABLE"
                if section_type not in LaserPowerExtractor.TARGET_SECTIONS:
                    continue
                # Scan with patterns
                for pattern in LaserPowerExtractor.PATTERNS:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        value = float(match.group(1))
                        unit = match.group(2)
                        # Filter out zero values (rarely true laser power)
                        if value == 0.0:
                            continue
                        # Filter out invalid units (e.g., lm, mm/s) – our patterns already restrict to W/kW/mW or irradiance units
                        # Get context (surrounding 100 chars)
                        start = max(0, match.start() - 50)
                        end = min(len(text), match.end() + 50)
                        context = text[start:end].strip()
                        is_irradiance = "cm²" in unit or "cm2" in unit
                        entries.append(LaserPowerEntry(
                            value=value,
                            unit=unit,
                            page=node.page_start,
                            section=section_type,
                            context=context,
                            is_irradiance=is_irradiance
                        ))
            else:
                stack.extend(node.children)
        # Deduplicate by (value, unit, page)
        unique = {}
        for e in entries:
            key = (e.value, e.unit, e.page)
            if key not in unique:
                unique[key] = e
        entries = list(unique.values())
        # Heuristic: prefer values from METHODS over RESULTS, and non-zero
        entries.sort(key=lambda x: (x.value == 0, x.section != "METHODS", x.section != "RESULTS"))
        return entries

# ============================================================================
# 4. Report Generation (Structured Table)
# ============================================================================
def generate_report(results: Dict[str, DocumentResult]) -> Dict:
    """Produce a dictionary structured like pageindex output."""
    docs_with_power = []
    docs_without = []
    for doc_name, res in results.items():
        if res.primary_power:
            docs_with_power.append({
                "paper": doc_name,
                "laser power": f"{res.primary_power.value} {res.primary_power.unit}",
                "irradiance": str([e for e in res.entries if e.is_irradiance][0]) if any(e.is_irradiance for e in res.entries) else None,
                "notes": f"Page {res.primary_power.page}, section: {res.primary_power.section}",
                "citation": res.primary_power.citation(doc_name)
            })
        else:
            docs_without.append(doc_name)
    return {
        "total_documents": len(results),
        "documents_with_laser_power": len(docs_with_power),
        "documents_without_laser_power": docs_without,
        "results": docs_with_power,
        "summary_table": pd.DataFrame(docs_with_power) if docs_with_power else pd.DataFrame()
    }

# ============================================================================
# 5. Streamlit UI
# ============================================================================
def main():
    st.set_page_config(page_title="DECLARMIMA v8.0 – Laser Power Extractor", layout="wide")
    st.title("🔬 DECLARMIMA v8.0 – Precision Laser Power Extractor")
    st.markdown("Upload PDFs → Extracts **real laser power values** (W, kW, mW) with exact page citations.")

    uploaded_files = st.file_uploader("Upload PDF papers", type="pdf", accept_multiple_files=True)
    if not uploaded_files:
        st.info("👆 Upload one or more PDF files to begin.")
        return

    if st.button("🔍 Extract Laser Power", type="primary"):
        with st.spinner("Building index and extracting values..."):
            progress = st.progress(0)
            progress.text("Indexing PDFs...")
            tree = PDFTree()
            tree.build(uploaded_files, max_workers=4)
            progress.progress(30)
            progress.text("Scanning for laser power...")
            all_results = {}
            for doc_name, root in tree.doc_trees.items():
                entries = LaserPowerExtractor.extract_from_document(root)
                # Choose primary power: prefer from METHODS, then highest value (most likely real)
                primary = None
                if entries:
                    # Sort: METHODS first, then RESULTS, then by value (larger possibly more important)
                    entries.sort(key=lambda e: (0 if e.section == "METHODS" else 1 if e.section == "RESULTS" else 2, -e.value))
                    primary = entries[0]
                all_results[doc_name] = DocumentResult(doc_name, entries, primary)
                progress.progress(30 + 60 * (len(all_results) / len(tree.doc_trees)))
            progress.progress(100)
            progress.empty()

        report = generate_report(all_results)

        # Display results as a table
        if report["documents_with_laser_power"] == 0:
            st.warning("No laser power values found. Try papers that explicitly mention laser power in Methods/Results.")
        else:
            st.success(f"✅ Found laser power in {report['documents_with_laser_power']} out of {report['total_documents']} documents.")
            st.dataframe(report["summary_table"], use_container_width=True)

            # Markdown table with citations
            md_lines = ["| Paper | Laser Power | Irradiance | Citation |", "|-------|-------------|------------|----------|"]
            for r in report["results"]:
                irrad = r["irradiance"] or "-"
                md_lines.append(f"| {r['paper']} | {r['laser power']} | {irrad} | {r['citation']} |")
            st.markdown("\n".join(md_lines))

            # JSON download
            json_output = json.dumps(report, indent=2, default=str)
            st.download_button("📥 Download JSON Report", json_output, "laser_power_report.json", "application/json")

        tree.cleanup()

if __name__ == "__main__":
    main()
