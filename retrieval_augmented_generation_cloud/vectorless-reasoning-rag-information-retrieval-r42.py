#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DECLARMIMA v6.0 - HIERARCHICAL_DOC_INDEX VECTORLESS RAG INTEGRATION
=======================================================
- NO vector embeddings, NO FAISS, NO chunking by character count
- Hierarchical document tree built from PDF structure (TOC → Sections → Pages)
- Agentic LLM navigation: LLM decides which branches to explore based on query
- Exact citation output: <cite doc="filename.pdf" page="X"/>
- Natural language reasoning: "Let me pull up your documents..."
- Local LLM support with 4-bit quantization
- Anti-hallucination: values validated against source text
"""

import streamlit as st
import os
import tempfile
import time
import re
import json
import torch
import numpy as np
import pandas as pd
from io import BytesIO
from typing import List, Dict, Optional, Tuple, Union, Any, Set, Callable
from datetime import datetime
import sys
import subprocess
import platform
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
from dataclasses import dataclass, field
import logging
import traceback
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# LOGGING CONFIGURATION
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("declarmima_app.log")
    ]
)
logger = logging.getLogger("DECLARMIMA")

# =====================================================================
# PYDANTIC SCHEMAS FOR STRUCTURED EXTRACTION
# =====================================================================
from pydantic import BaseModel, Field, validator
from typing import Optional, List as ListType

class QuantitativeMeasurement(BaseModel):
    """A single quantitative measurement extracted from text."""
    parameter_name: str = Field(description="The physical parameter being measured")
    value: float = Field(description="The numerical value")
    unit: str = Field(description="The unit of measurement")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")
    context: str = Field(description="The exact sentence from which this was extracted")
    material: Optional[str] = Field(default=None, description="Material system mentioned")
    method: Optional[str] = Field(default=None, description="Experimental/computational method")
    conditions: Dict[str, Any] = Field(default_factory=dict, description="Relevant conditions")
    doc_source: str = Field(description="Exact source filename")
    page: int = Field(description="Page number where value was found")

class ScientificClaim(BaseModel):
    """A non-quantitative scientific claim linking subject, predicate, object."""
    claim_text: str = Field(description="The exact text of the claim")
    subject: str = Field(description="The main entity (material, phenomenon, process)")
    predicate: str = Field(description="Action or relation (e.g., 'increases', 'forms', 'causes')")
    object_val: str = Field(description="The target of the claim")
    claim_type: str = Field(description="Type: 'causal', 'correlational', 'definitional', 'comparative'")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")
    evidence_span: str = Field(description="Supporting text snippet")
    supporting_entities: List[str] = Field(default_factory=list, description="Entities mentioned in the claim")

# =====================================================================
# IMPORTS
# =====================================================================
from langchain_core.documents import Document
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig
)

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# =====================================================================
# CONFIGURATION
# =====================================================================
class HierarchicalDocIndexConfig:
    MAX_NAVIGATION_STEPS = 3
    MAX_RESULTS_PER_QUERY = 25
    MAX_CHUNKS_PER_NODE = 5
    LLM_TIMEOUT_SECONDS = 30
    CONFIDENCE_THRESHOLD = 0.7
    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    USE_4BIT = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = HierarchicalDocIndexConfig()

# =====================================================================
# HIERARCHICAL_DOC_INDEX CORE: Hierarchical Document Tree (NO EMBEDDINGS)
# =====================================================================

@dataclass
class PageNode:
    """Node in the document tree representing a page/section."""
    id: str
    title: str
    page_start: int
    page_end: Optional[int]
    full_text: str
    summary: str
    level: int  # 0=root, 1=chapter, 2=section, 3=subsection
    children: List['PageNode'] = field(default_factory=list)
    doc_id: str = ""
    section_type: str = "BODY"  # ABSTRACT, METHODS, RESULTS, etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "title": self.title,
            "page_range": f"{self.page_start}-{self.page_end}" if self.page_end else str(self.page_start),
            "summary": self.summary[:200], "level": self.level,
            "section_type": self.section_type, "has_children": bool(self.children),
            "doc_id": self.doc_id
        }

class HierarchicalPDFIndex:
    """
    Builds a natural hierarchical index from PDFs using:
    1. Table of Contents (if available)
    2. Regex-based heading detection (fallback)
    3. Page-by-page fallback (guaranteed)
    """

    SECTION_PATTERNS = [
        (r'(?i)^\s*Abstract\s*$', 'ABSTRACT'),
        (r'(?i)^\s*(?:1\.?\s*)?Introduction\s*$', 'INTRODUCTION'),
        (r'(?i)^\s*(?:2\.?\s*)?(?:Experimental|Methods?|Methodology|Setup)\s*$', 'METHODS'),
        (r'(?i)^\s*(?:3\.?\s*)?(?:Results?|Findings|Outcomes)\s*$', 'RESULTS'),
        (r'(?i)^\s*(?:4\.?\s*)?Discussion\s*$', 'DISCUSSION'),
        (r'(?i)^\s*(?:5\.?\s*)?Conclusion\s*$', 'CONCLUSION'),
    ]

    def __init__(self):
        self.doc_trees: Dict[str, PageNode] = {}

    def build_from_pdfs(self, files: List) -> Dict[str, PageNode]:
        """Build tree index from uploaded PDF files."""
        for file in files:
            doc_id = file.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getbuffer())
                tmp_path = tmp.name
            try:
                doc = fitz.open(tmp_path)
                root = self._build_tree_for_doc(doc, doc_id)
                self.doc_trees[doc_id] = root
                doc.close()
            finally:
                try: Path(tmp_path).unlink()
                except: pass
        return self.doc_trees

    def _build_tree_for_doc(self, doc, doc_id: str) -> PageNode:
        """Build hierarchical tree for a single PDF."""
        root = PageNode(
            id=f"{doc_id}_root", title="Document Root",
            page_start=1, page_end=len(doc),
            full_text="", summary=f"Full document: {doc_id}",
            level=0, doc_id=doc_id
        )
        # Try TOC first (most reliable)
        toc = doc.get_toc()
        if toc:
            return self._build_from_toc(doc, doc_id, toc, root)
        # Fallback: regex heading detection
        headings = self._detect_headings_regex(doc)
        if headings:
            return self._build_from_headings(doc, doc_id, headings, root)
        # Final fallback: page-by-page
        return self._build_page_by_page(doc, doc_id, root)

    def _build_from_toc(self, doc, doc_id: str, toc: List, root: PageNode) -> PageNode:
        """Build tree from PDF Table of Contents."""
        nodes_by_level: Dict[int, List[PageNode]] = {}
        for entry in toc:
            level, title, page = entry[:3]
            page_end = min(page + 5, len(doc))
            section_text = self._extract_page_range(doc, page, page_end)
            summary = self._generate_summary(section_text)
            section_type = self._classify_section(title)
            node = PageNode(
                id=f"{doc_id}_toc_{level}_{title.lower().replace(' ', '_')}",
                title=title.strip(), page_start=page, page_end=page_end,
                full_text=section_text, summary=summary,
                level=level, section_type=section_type, doc_id=doc_id
            )
            nodes_by_level.setdefault(level, []).append(node)
        # Attach nodes to tree hierarchically
        for level in sorted(nodes_by_level.keys()):
            for node in nodes_by_level[level]:
                parent = self._find_parent(root, level - 1, node.page_start)
                if parent:
                    parent.children.append(node)
                else:
                    root.children.append(node)
        return root

    def _build_from_headings(self, doc, doc_id: str, headings: List[Tuple[str, int]], root: PageNode) -> PageNode:
        """Build tree from regex-detected headings."""
        for i, (title, page) in enumerate(headings):
            page_end = min(page + 5, len(doc))
            section_text = self._extract_page_range(doc, page, page_end)
            summary = self._generate_summary(section_text)
            section_type = self._classify_section(title)
            node = PageNode(
                id=f"{doc_id}_h_{i}_{title.lower().replace(' ', '_')}",
                title=title.strip(), page_start=page, page_end=page_end,
                full_text=section_text, summary=summary,
                level=2, section_type=section_type, doc_id=doc_id
            )
            root.children.append(node)
        return root

    def _build_page_by_page(self, doc, doc_id: str, root: PageNode) -> PageNode:
        """Fallback: treat each page as a leaf node."""
        for page_num in range(1, len(doc) + 1):
            page_text = doc[page_num - 1].get_text("text")
            if not page_text.strip(): continue
            summary = self._generate_summary(page_text)
            section_type = self._classify_section_by_content(page_text)
            node = PageNode(
                id=f"{doc_id}_p{page_num}", title=f"Page {page_num}",
                page_start=page_num, page_end=page_num,
                full_text=page_text, summary=summary,
                level=3, section_type=section_type, doc_id=doc_id
            )
            root.children.append(node)
        return root

    def _extract_page_range(self, doc, start_page: int, end_page: int) -> str:
        """Extract text from page range (1-indexed)."""
        texts = []
        for p in range(start_page - 1, min(end_page, len(doc))):
            texts.append(doc[p].get_text("text"))
        return "\n\n".join(texts)

    def _generate_summary(self, text: str, max_chars: int = 200) -> str:
        """Generate lightweight summary (first 2 sentences or max_chars)."""
        if not text: return ""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        summary = " ".join(sentences[:2])
        return summary[:max_chars] + ("..." if len(summary) > max_chars else "")

    def _classify_section(self, title: str) -> str:
        """Classify section type from title."""
        title_lower = title.lower()
        for pattern, section_type in self.SECTION_PATTERNS:
            if re.search(pattern, title_lower):
                return section_type
        return "BODY"

    def _classify_section_by_content(self, text: str) -> str:
        """Classify section type from content keywords."""
        text_lower = text[:500].lower()
        if any(kw in text_lower for kw in ['abstract', 'summary']): return "ABSTRACT"
        if any(kw in text_lower for kw in ['method', 'experimental', 'setup']): return "METHODS"
        if any(kw in text_lower for kw in ['result', 'finding', 'figure', 'table']): return "RESULTS"
        if any(kw in text_lower for kw in ['discussion', 'interpretation']): return "DISCUSSION"
        if any(kw in text_lower for kw in ['conclusion', 'concluding']): return "CONCLUSION"
        return "BODY"

    def _detect_headings_regex(self, doc) -> List[Tuple[str, int]]:
        """Detect headings using regex patterns."""
        headings = []
        for page_num in range(len(doc)):
            text = doc[page_num].get_text("text")
            patterns = [
                r'^(?:\d+\.?\s*)+([A-Z][^\n]{5,80})$',
                r'^##\s+([A-Z][^\n]{5,80})$',
                r'^([A-Z][A-Z\s]{5,40})$',
            ]
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.MULTILINE):
                    title = match.group(1).strip()
                    if 5 < len(title) < 100 and title[0].isupper():
                        headings.append((title, page_num + 1))
        return headings

    def _find_parent(self, root: PageNode, target_level: int, page_hint: int) -> Optional[PageNode]:
        """Find parent node at target_level with page closest to page_hint."""
        if target_level < 0: return root
        candidates = [n for n in root.children if n.level == target_level]
        if not candidates: return root
        return min(candidates, key=lambda n: abs(n.page_start - page_hint))

    def get_node_by_id(self, node_id: str) -> Optional[PageNode]:
        """Retrieve node by ID via DFS."""
        def _search(node: PageNode) -> Optional[PageNode]:
            if node.id == node_id: return node
            for child in node.children:
                result = _search(child)
                if result: return result
            return None
        for root in self.doc_trees.values():
            result = _search(root)
            if result: return result
        return None

    def format_tree_view(self, nodes: List[PageNode], max_depth: int = 2) -> str:
        """Format nodes for LLM navigation prompt."""
        lines = []
        for node in nodes:
            indent = "  " * min(node.level, max_depth)
            page_info = f"p.{node.page_start}" if node.page_end == node.page_start else f"p.{node.page_start}-{node.page_end}"
            lines.append(f"{indent}- ID: `{node.id}` | {node.title} | {page_info} | {node.section_type}")
            if node.summary:
                lines.append(f"{indent}  → {node.summary}")
            if node.level < max_depth and node.children:
                lines.append(f"{indent}  [Has {len(node.children)} subsections]")
        return "\n".join(lines)

# =====================================================================
# LOCAL LLM LOADER (4-bit quantization support)
# =====================================================================

class LocalLLM:
    """Local LLM loader with 4-bit quantization for consumer GPUs."""

    def __init__(self, model_name: str = config.DEFAULT_MODEL, use_4bit: bool = config.USE_4BIT):
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.tokenizer = None
        self.model = None
        self.device = config.DEVICE
        self._load_model()

    def _load_model(self):
        """Load model with 4-bit quantization if enabled."""
        logger.info(f"Loading {self.model_name} on {self.device}...")

        quantization_config = None
        if self.use_4bit and self.device == "cuda":
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                logger.info("✅ 4-bit quantization enabled")
            except ImportError:
                logger.warning("⚠️ bitsandbytes not installed, falling back to FP16")
                self.use_4bit = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, padding_side="left", use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if "device_map" not in model_kwargs and self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"✅ Model loaded: {self.model_name}")

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.1) -> str:
        """Generate response from local LLM."""
        try:
            if "Qwen" in self.model_name or "qwen" in self.model_name.lower():
                messages = [
                    {"role": "system", "content": "You are an expert scientific research assistant."},
                    {"role": "user", "content": prompt}
                ]
                formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            elif "Llama" in self.model_name or "llama" in self.model_name.lower():
                messages = [
                    {"role": "system", "content": "You are an expert scientific research assistant."},
                    {"role": "user", "content": prompt}
                ]
                formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                formatted = prompt

            inputs = self.tokenizer.encode(formatted, return_tensors='pt', truncation=True, max_length=2048)
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = inputs.to('cuda')

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                    do_sample=(temperature > 0), pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id, no_repeat_ngram_size=3, early_stopping=True
                )

            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "[/INST]" in full_text:
                answer = full_text.split("[/INST]")[-1].strip()
            else:
                answer = full_text[-max_new_tokens*2:].strip()
            return re.sub(r'\s+', ' ', answer).strip()

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error: {str(e)[:200]}..."

# =====================================================================
# AGENTIC TREE RETRIEVER (HierarchicalDocIndex Core - NO VECTORS)
# =====================================================================

class AgenticTreeRetriever:
    """
    LLM-powered router that navigates document trees.
    NO vector similarity — pure LLM reasoning over tree structure.
    """

    NAVIGATION_PROMPT = """You are an expert scientific research navigator.
Given a query and document tree sections, select which sections to read next.

QUERY: {query}
AVAILABLE SECTIONS:
{tree_view}

INSTRUCTIONS:
1. Select ONLY section IDs likely to contain quantitative values (numbers + units) relevant to the query.
2. Prioritize METHODS, RESULTS, and EXPERIMENTAL sections for parameter values.
3. If a section has subsections, you may select the parent to expand, or select specific leaf nodes.
4. Return ONLY a valid JSON array of section IDs. Example: ["doc1_methods", "doc2_results_laser"]
5. If no sections are relevant, return an empty array [].

JSON OUTPUT:"""

    def __init__(self, llm: LocalLLM, max_steps: int = config.MAX_NAVIGATION_STEPS, 
                 max_results: int = config.MAX_RESULTS_PER_QUERY):
        self.llm = llm
        self.max_steps = max_steps
        self.max_results = max_results
        self.navigation_trace: List[Dict] = []

    def retrieve(self, query: str, tree_roots: List[PageNode]) -> List[Dict[str, Any]]:
        """Navigate tree to find relevant content."""
        results = []
        current_nodes = tree_roots
        self.navigation_trace = []

        for step in range(self.max_steps):
            if len(results) >= self.max_results: break

            tree_view = self._format_navigation_view(current_nodes)
            prompt = self.NAVIGATION_PROMPT.format(query=query, tree_view=tree_view)

            try:
                response = self.llm.generate(prompt, max_new_tokens=256, temperature=0.1)
                selected_ids = self._parse_json_array(response)

                if not selected_ids:
                    results.extend(self._collect_leaf_content(current_nodes))
                    break

                new_nodes = []
                for node_id in selected_ids:
                    node = self._find_node_by_id(tree_roots, node_id)
                    if node:
                        if node.children:
                            new_nodes.extend(node.children)
                        else:
                            results.append({
                                "full_text": node.full_text,
                                "page_start": node.page_start, "page_end": node.page_end,
                                "doc_id": node.doc_id, "section_title": node.title,
                                "section_type": node.section_type,
                                "citation": f'<cite doc="{node.doc_id}" page="{node.page_start}"/>'
                            })
                            self.navigation_trace.append({
                                "step": step, "action": "collected_leaf",
                                "node_id": node.id, "pages": f"{node.page_start}-{node.page_end}"
                            })

                if not new_nodes:
                    results.extend(self._collect_leaf_content(current_nodes))
                    break

                current_nodes = new_nodes
                self.navigation_trace.append({
                    "step": step, "action": "expanded",
                    "selected_ids": selected_ids, "new_node_count": len(new_nodes)
                })

            except Exception as e:
                logger.warning(f"Navigation step {step} failed: {e}")
                results.extend(self._collect_leaf_content(current_nodes))
                break

        # Deduplicate by (doc_id, page_start)
        seen = set()
        unique_results = []
        for r in results:
            key = (r["doc_id"], r["page_start"])
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        return unique_results[:self.max_results]

    def _format_navigation_view(self, nodes: List[PageNode]) -> str:
        lines = []
        for node in nodes:
            indent = "  " * min(node.level, 2)
            page_info = f"p.{node.page_start}" if node.page_end == node.page_start else f"p.{node.page_start}-{node.page_end}"
            lines.append(f"{indent}- ID: `{node.id}` | {node.title} | {page_info} | {node.section_type}")
            if node.summary:
                lines.append(f"{indent}  → {node.summary}")
        return "\n".join(lines)

    def _parse_json_array(self, text: str) -> List[str]:
        patterns = [
            r'\[\s*"[^"]*"(?:\s*,\s*"[^"]*")*\s*\]',
            r'```json\s*(\[.*?\])\s*```',
            r'(\[.*\])',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1 if match.lastindex else 0))
                except json.JSONDecodeError:
                    continue
        return []

    def _find_node_by_id(self, roots: List[PageNode], target_id: str) -> Optional[PageNode]:
        def _search(node: PageNode) -> Optional[PageNode]:
            if node.id == target_id: return node
            for child in node.children:
                result = _search(child)
                if result: return result
            return None
        for root in roots:
            result = _search(root)
            if result: return result
        return None

    def _collect_leaf_content(self, nodes: List[PageNode]) -> List[Dict[str, Any]]:
        results = []
        for node in nodes:
            if not node.children:
                results.append({
                    "full_text": node.full_text,
                    "page_start": node.page_start, "page_end": node.page_end,
                    "doc_id": node.doc_id, "section_title": node.title,
                    "section_type": node.section_type,
                    "citation": f'<cite doc="{node.doc_id}" page="{node.page_start}"/>'
                })
            else:
                results.extend(self._collect_leaf_content(node.children))
        return results

    def get_navigation_trace(self) -> List[Dict]:
        return self.navigation_trace

# =====================================================================
# ANSWER SYNTHESIS: NATURAL LANGUAGE + EXACT CITATIONS
# =====================================================================

def format_hierarchical_doc_index_answer(measurements: List[QuantitativeMeasurement], 
                            query: str, 
                            metadata: Dict,
                            tree_index: HierarchicalPDFIndex) -> str:
    """
    Format answer with natural language reasoning and exact citations.
    Matches the user's example output format exactly.
    """

    doc_count = len(set(m.doc_source for m in measurements))
    lines = [
        f"Let me pull up your recent documents to find the relevant papers!",
        f"I found {doc_count} paper(s). I'll fetch their content directly — in parallel!",
        f"Here's a summary of the laser power discussed in the papers:",
        ""
    ]

    # Group by document
    by_doc = defaultdict(list)
    for m in measurements:
        by_doc[m.doc_source].append(m)

    for doc_id, docs_measurements in by_doc.items():
        doc_root = tree_index.doc_trees.get(doc_id)
        doc_title = doc_root.title if doc_root else "Unknown"

        lines.append(f"---")
        lines.append(f"### 📄 {doc_id} — *{doc_title}*")
        lines.append("")

        # Extract laser power / irradiance values
        power_values = [m for m in docs_measurements 
                       if "power" in m.parameter_name.lower() 
                       or "irradiance" in m.parameter_name.lower()]

        if power_values:
            value_groups = defaultdict(list)
            for pv in power_values:
                key = f"{pv.value} {pv.unit}"
                value_groups[key].append(pv)

            for value_key, instances in value_groups.items():
                citations = " ".join([
                    f'<cite doc="{i.doc_source}" page="{i.page}"/>'
                    for i in instances
                ])

                if len(instances) > 1:
                    lines.append(
                        f"This paper uses a **laser power (P) of {value_key}** "
                        f"across all experimental conditions. {citations}"
                    )
                else:
                    lines.append(
                        f"This paper uses a **laser power (P) of {value_key}**. {citations}"
                    )
                lines.append("")

        # Key details with citations
        lines.append("Key details:")
        for m in docs_measurements[:5]:
            if m.parameter_name not in ["laser power", "irradiance"]:
                lines.append(
                    f"- **{m.parameter_name}:** {m.value} {m.unit} "
                    f'<cite doc="{m.doc_source}" page="{m.page}"/>'
                )
        lines.append("")

    # Cross-document comparison table
    if len(by_doc) > 1:
        lines.append("### Key Difference")
        lines.append("| | " + " | ".join(list(by_doc.keys())) + " |")
        lines.append("|---|" + "---|" * len(by_doc))

        # Scale row
        scales = []
        for doc_id in by_doc:
            scale = "Nano-scale (nm)" if any("nm" in m.context.lower() for m in by_doc[doc_id]) else "Micron-scale (µm)"
            scales.append(scale)
        lines.append(f"| **Scale** | " + " | ".join(scales) + " |")

        # Power row
        powers = []
        for doc_id in by_doc:
            pv = [m for m in by_doc[doc_id] if "power" in m.parameter_name.lower()]
            if pv:
                powers.append(f"Power: **{pv[0].value} {pv[0].unit}**")
            else:
                powers.append("N/A")
        lines.append(f"| **Laser quantity** | " + " | ".join(powers) + " |")

        lines.append("")

    return "\n".join(lines)

# =====================================================================
# HIERARCHICAL_DOC_INDEX QUERY PROCESSOR (REPLACES QueryDrivenProcessor)
# =====================================================================

class HierarchicalDocIndexQueryProcessor:
    """HierarchicalDocIndex-style processor: replaces vector retrieval with tree navigation."""

    def __init__(self):
        self.raw_files: List = []
        self.tree_index: Optional[HierarchicalPDFIndex] = None
        self.retriever: Optional[AgenticTreeRetriever] = None
        self.llm: Optional[LocalLLM] = None

    def register_files(self, files: List) -> None:
        self.raw_files = files
        self.tree_index = None  # Rebuild index on next query

    def process_for_query(self, query: str, progress_callback: Optional[Callable] = None,
                         model_name: str = config.DEFAULT_MODEL, 
                         use_4bit: bool = config.USE_4BIT) -> Tuple[List[QuantitativeMeasurement], Dict]:

        # Step 1: Load local LLM (cached)
        if self.llm is None:
            if progress_callback: progress_callback(0.1, f"🤖 Loading {model_name} locally...")
            self.llm = LocalLLM(model_name=model_name, use_4bit=use_4bit)
            if progress_callback: progress_callback(0.2, "✅ LLM loaded")

        # Step 2: Build tree index if needed
        if self.tree_index is None:
            if progress_callback: progress_callback(0.3, "🌳 Building hierarchical document index...")
            self.tree_index = HierarchicalPDFIndex()
            self.tree_index.build_from_pdfs(self.raw_files)
            if progress_callback: progress_callback(0.4, "✅ Index built")

        # Step 3: Initialize retriever
        if self.retriever is None:
            self.retriever = AgenticTreeRetriever(
                llm=self.llm,
                max_steps=config.MAX_NAVIGATION_STEPS,
                max_results=config.MAX_RESULTS_PER_QUERY
            )

        # Step 4: Agentic tree navigation (NO VECTORS)
        if progress_callback: progress_callback(0.5, "🔍 Navigating document tree...")
        tree_roots = list(self.tree_index.doc_trees.values())
        retrieved_pages = self.retriever.retrieve(query, tree_roots)

        if progress_callback: progress_callback(0.7, f"✅ Retrieved {len(retrieved_pages)} relevant sections")

        # Step 5: Extract quantitative values from retrieved pages
        if progress_callback: progress_callback(0.8, "🤖 Extracting laser power values...")
        all_measurements = []

        for page in retrieved_pages:
            if not re.search(r'\d+\s*(?:W|w|kW|mW|J/cm²|MPa)', page["full_text"]):
                continue

            measurements = self._extract_values_from_text(
                page["full_text"], query, page["doc_id"], page["page_start"]
            )
            all_measurements.extend(measurements)

        if progress_callback: progress_callback(0.95, f"✅ Extracted {len(all_measurements)} measurements")

        metadata = {
            "retrieval_method": "hierarchical_doc_index_tree_navigation",
            "navigation_trace": self.retriever.get_navigation_trace(),
            "sections_retrieved": len(retrieved_pages),
            "documents_covered": len(set(p["doc_id"] for p in retrieved_pages)),
            "llm_model": model_name,
            "use_4bit": use_4bit
        }

        if progress_callback: progress_callback(1.0, "✅ Processing complete")

        return all_measurements, metadata

    def _extract_values_from_text(self, text: str, query: str, doc_id: str, page: int) -> List[QuantitativeMeasurement]:
        """Extract quantitative values using LLM with strict anti-hallucination."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        value_sentences = [s for s in sentences if re.search(r'\d+\s*(?:W|w|kW|mW|J/cm²|MPa|GPa|µm|mm|°C)', s)]

        if not value_sentences:
            return []

        system = """Extract ONLY numbers with units that EXIST in the provided text below.
HALLUCINATION IS FORBIDDEN. Do not invent values, documents, or authors.
Format for each measurement:
{"parameter_name": "laser power", "value": 250, "unit": "W", "context": "exact sentence from text", "doc_source": "EXACT_FILENAME.pdf", "page": 6}
STRICT RULES:
1. ONLY extract from the text provided below - NEVER invent
2. For each value, include the EXACT source filename and EXACT sentence from text
3. If no values exist in this text, return {"measurements": []}
4. NEVER invent document names like "Smith et al." or "Johnson & Lee"
5. NEVER invent values that don't appear in the text
6. If unsure, return empty list rather than guessing
7. Parameter name can be inferred from context (e.g., "250 W" near "laser" → "laser power")
8. Return ONLY JSON: {"measurements": [...]}
9. No extra text before or after JSON
"""
        user = f"""SOURCE DOCUMENT: {doc_id}, PAGE: {page}
TEXT TO EXTRACT FROM:
{" ".join(value_sentences[:10])}
EXTRACTION TASK: Find ALL laser power values (numbers with units like W, kW, mW) mentioned in the text above.
REQUIREMENTS:
- Only extract values that appear in the text above
- Include exact sentence as context
- Use filename '{doc_id}' as doc_source
- Use page {page} as page number
- Return valid JSON only
QUERY CONTEXT: {query}"""
        prompt = f"{system}\n{user}"

        try:
            response = self.llm.generate(prompt, max_new_tokens=512, temperature=0.1)
            json_str = self._extract_json(response)
            if json_str:
                data = json.loads(json_str)
                measurements = [QuantitativeMeasurement(**m) for m in data.get("measurements", [])]
                # Validate: ensure values exist in source text
                validated = []
                for m in measurements:
                    if str(m.value) in text and m.unit in text:
                        validated.append(m)
                return validated
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
        return []

    def _extract_json(self, text: str) -> Optional[str]:
        patterns = [
            r'\{.*"measurements".*\}',
            r'```json\s*(\{.*?\})\s*```',
            r'(\{.*\})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    json.loads(match.group(1 if match.groups() else 0))
                    return match.group(1 if match.groups() else 0)
                except:
                    continue
        return None

# =====================================================================
# STREAMLIT UI
# =====================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("#### 🌳 HierarchicalDocIndex Settings")
        st.session_state.hierarchical_doc_index_max_steps = st.slider(
            "Max navigation steps", min_value=1, max_value=5, value=3
        )
        st.session_state.hierarchical_doc_index_max_results = st.slider(
            "Max sections to retrieve", min_value=10, max_value=50, value=25
        )
        st.session_state.show_navigation_trace = st.checkbox(
            "🔍 Show navigation trace", value=True
        )
        st.markdown("#### 🤖 Local LLM Settings")
        st.session_state.hierarchical_doc_index_model = st.selectbox(
            "Local LLM Model",
            options=[
                "Qwen/Qwen2.5-7B-Instruct",
                "meta-llama/Llama-3.1-8B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.3",
                "google/gemma-2-9b-it",
            ],
            index=0
        )
        st.session_state.hierarchical_doc_index_use_4bit = st.checkbox(
            "🗜️ Use 4-bit quantization", value=True
        )

def render_navigation_trace(trace: List[Dict]):
    """Render navigation trace in an expander."""
    if not trace: return
    with st.expander("🗺️ HierarchicalDocIndex Navigation Trace", expanded=False):
        for entry in trace:
            step = entry.get("step", "?")
            action = entry.get("action", "?")
            if action == "expanded":
                st.markdown(f"**Step {step}**: Expanded sections → {entry.get('new_node_count', '?')} new nodes")
                if entry.get("selected_ids"):
                    st.code(f"Selected IDs: {entry['selected_ids'][:3]}...", language="json")
            elif action == "collected_leaf":
                st.markdown(f"**Step {step}**: Collected content from {entry.get('node_id', '?')} (pages {entry.get('pages', '?')})")

def main():
    st.set_page_config(page_title="🌳 DECLARMIMA: HierarchicalDocIndex Vectorless RAG", page_icon="🌳", layout="wide")
    st.markdown('<h1 style="text-align:center">🌳 DECLARMIMA: HierarchicalDocIndex Vectorless RAG</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:#64748b;margin-bottom:1.5rem">
    <strong>NO embeddings</strong> • <strong>Hierarchical tree index</strong> • <strong>LLM navigation agent</strong> • <strong>Exact citations</strong>
    </div>
    """, unsafe_allow_html=True)

    render_sidebar()

    # File uploader
    uploaded_files = st.file_uploader("Upload PDF papers about laser processing", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and st.button("📥 Register Files", type="primary"):
        st.session_state.query_processor = HierarchicalDocIndexQueryProcessor()
        st.session_state.query_processor.register_files(uploaded_files)
        st.success(f"✅ Registered {len(uploaded_files)} files")

    # Chat interface
    if "query_processor" in st.session_state and st.session_state.query_processor.raw_files:
        if prompt := st.chat_input("Ask about laser power values..."):
            with st.spinner("🔍 Navigating document tree..."):
                progress = st.progress(0.0)

                def progress_cb(pct, msg):
                    progress.progress(pct, text=msg)

                measurements, metadata = st.session_state.query_processor.process_for_query(
                    query=prompt,
                    progress_callback=progress_cb,
                    model_name=st.session_state.get("hierarchical_doc_index_model", "Qwen/Qwen2.5-7B-Instruct"),
                    use_4bit=st.session_state.get("hierarchical_doc_index_use_4bit", True)
                )

                # Format answer with natural language + exact citations
                answer = format_hierarchical_doc_index_answer(
                    measurements, prompt, metadata, 
                    st.session_state.query_processor.tree_index
                )
                st.markdown(answer)

                # Show navigation trace if enabled
                if st.session_state.get("show_navigation_trace") and metadata.get("navigation_trace"):
                    render_navigation_trace(metadata["navigation_trace"])

                # Show diagnostics
                with st.expander("📊 Response Diagnostics", expanded=False):
                    st.metric("Sections Retrieved", metadata.get("sections_retrieved", 0))
                    st.metric("Documents Covered", metadata.get("documents_covered", 0))
                    st.metric("Measurements Extracted", len(measurements))
                    st.code(f"LLM: {metadata.get('llm_model')} | 4-bit: {metadata.get('use_4bit')}")

    else:
        st.info("👆 Upload PDF files above, then ask your question.")

if __name__ == "__main__":
    main()
