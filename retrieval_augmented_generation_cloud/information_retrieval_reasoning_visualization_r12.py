#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LASER MICROSTRUCTURE RAG CHATBOT - CROSS-DOCUMENT SCIENTIFIC REASONING VERSION
========================================================================================
✅ Zero API keys required - all models run locally
✅ Cross-document reasoning: consensus, contradiction, and gap detection
✅ Scientific entity extraction and alignment across papers
✅ Multi-hop retrieval via knowledge graph traversal
✅ Uncertainty-calibrated responses with structured provenance
✅ Enhanced citations with bibliographic metadata
✅ Interactive visualizations:
   - Consensus bar charts (plotly)
   - Contradiction highlight table
   - Interactive knowledge graph (pyvis)
   - Retrieval debugger with relevance scores
   - LaTeX formula preview
   - User feedback loop with precision/recall tracking
   - On‑demand plot generation from LLM answers
✅ DECLARMIMA-aligned domain: multicomponent alloys, laser-microstructure interaction
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
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from typing import List, Dict, Optional, Tuple, Union, Any, Set
from datetime import datetime
import sys
import subprocess
import platform
from pathlib import Path
from collections import defaultdict, Counter
import hashlib

# LangChain / RAG imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Transformers
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, AutoModelForCausalLM,
    pipeline, set_seed, BitsAndBytesConfig
)

# Optional: Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Bibliographic metadata
try:
    import pdf2doi
    PDF2DOI_AVAILABLE = True
except (ImportError, PermissionError, Exception):
    PDF2DOI_AVAILABLE = False

try:
    from crossrefapi import CrossrefAPI
    CROSSREF_AVAILABLE = True
except ImportError:
    CROSSREF_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

# Pyvis for network graph
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

# =============================================
# GLOBAL CONFIGURATION
# =============================================

LOCAL_LLM_OPTIONS = {
    "GPT-2 (1.5B, fastest startup, CPU OK)": "gpt2",
    "Qwen2-0.5B-Instruct (best JSON, recommended)": "Qwen/Qwen2-0.5B-Instruct",
    "Qwen2.5-0.5B-Instruct (newest, best reasoning)": "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama-1.1B-Chat (balanced small model)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen2.5-1.5B-Instruct (efficient mid-size)": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct (strong reasoning)": "Qwen/Qwen2.5-3B-Instruct",
    "Mistral-7B-Instruct-v0.3 (reliable & efficient)": "mistralai/Mistral-7B-Instruct-v0.3",
    "Llama-3.2-3B-Instruct (Meta's latest small)": "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen2.5-7B-Instruct (excellent all-rounder)": "Qwen/Qwen2.5-7B-Instruct",
    "Llama-3.1-8B-Instruct (most popular balanced)": "meta-llama/Llama-3.1-8B-Instruct",
    "Gemma-2-9B-it (Google's latest, great logic)": "google/gemma-2-9b-it",
    "Falcon-7B-Instruct (lightweight & modern)": "tiiuae/falcon-7b-instruct",
    "[Ollama] qwen2.5:0.5b (via ollama serve)": "ollama:qwen2.5:0.5b",
    "[Ollama] qwen2.5:1.5b (via ollama serve)": "ollama:qwen2.5:1.5b",
    "[Ollama] qwen2.5:7b (via ollama serve)": "ollama:qwen2.5:7b",
    "[Ollama] qwen2.5:14b (via ollama serve) 🔥": "ollama:qwen2.5:14b",
    "[Ollama] llama3.1:8b (via ollama serve)": "ollama:llama3.1:8b",
    "[Ollama] mistral:7b (via ollama serve)": "ollama:mistral:7b",
    "[Ollama] gemma2:9b (via ollama serve)": "ollama:gemma2:9b",
    "[Ollama] falcon3:10b (via ollama serve)": "ollama:falcon3:10b",
}

LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LASER_DOMAIN_CONFIG = {
    "chunk_size": 800,
    "chunk_overlap": 150,
    "retrieval_k": 4,
    "score_threshold": 0.25,
    "max_context_tokens": 2048,
    "max_new_tokens": 512,
    "temperature": 0.05,
}

# -------------------------------------------
# DECLARMIMA-aligned laser/materials keywords
# -------------------------------------------
LASER_KEYWORDS = {
    "ablation": ["ablation", "material removal", "threshold fluence", "laser ablation", "ablation threshold"],
    "plasma": ["plasma formation", "ionization", "electron density", "plume", "plasma shielding"],
    "thermal": ["heat affected zone", "melting", "thermal diffusion", "resolidification", "heat-affected zone"],
    "ultrafast": ["femtosecond", "picosecond", "pulse duration", "ultrafast laser", "fs laser"],
    "morphology": ["ripples", "LIPSS", "surface structuring", "periodic structures", "nanostructures", "microstructures"],
    "parameters": ["fluence", "wavelength", "pulse energy", "repetition rate", "spot size", "scan speed", "overlap",
                   "hatch distance", "laser power", "point distance"],
    "materials": ["silicon", "steel", "titanium", "polymer", "glass", "ceramic", "aluminum", "copper", "tungsten",
                  "multicomponent alloy", "high entropy alloy", "solder", "Sn-Ag-Cu", "Al-Cr-Fe-Ni", "Inconel"],
    "characterization": ["SEM", "AFM", "profilometry", "spectroscopy", "microscopy", "Raman", "XRD", "EDX",
                         "EBSD", "Tomography", "X-ray radiography"],
    "additive_manufacturing": ["additive manufacturing", "3D printing", "selective laser melting", "SLM",
                               "laser powder bed fusion", "LPBF", "directed energy deposition"],
    "multicomponent": ["multicomponent alloy", "multi-principal element alloy", "MPEA", "high entropy alloy",
                       "HEA", "multi-component", "complex concentrated alloy"],
    "digital_twin": ["digital twin", "physics-informed digital twin", "PIDT", "in-silico", "virtual qualification"],
    "simulation": ["phase field", "molecular dynamics", "MD simulation", "finite element", "MOOSE",
                   "CALPHAD", "Thermo-Calc", "multi-scale", "mesoscale", "nanoscale"],
    "data_driven": ["machine learning", "neural network", "random forest", "CNN", "data-driven",
                    "physics-informed ML", "feature engineering", "tensor decomposition"],
    "properties": ["interfacial energy", "thermal conductivity", "diffusion coefficient", "viscosity",
                   "gibbs free energy", "enthalpy", "absorptivity", "reflectivity", "spatter", "porosity"],
}

MATERIAL_ALIASES = {
    "silicon": ["silicon", "si", "crystalline silicon", "c-si", "si(100)", "si(111)"],
    "titanium": ["titanium", "ti", "cp-ti", "ti-6al-4v", "ti6al4v"],
    "steel": ["steel", "stainless steel", "ss304", "ss316", "mild steel", "carbon steel"],
    "aluminum": ["aluminum", "aluminium", "al", "al6061", "al-6061"],
    "copper": ["copper", "cu"],
    "tungsten": ["tungsten", "w"],
    "glass": ["glass", "fused silica", "sio2", "borosilicate"],
    "polymer": ["polymer", "pmma", "polyimide", "pei", "pc", "polycarbonate", "ptfe"],
    "ceramic": ["ceramic", "alumina", "al2o3", "zirconia", "zro2"],
    "Sn-Ag-Cu": ["snagcu", "sac", "sn-ag-cu", "sn-3.5ag-0.5cu", "solder", "lead-free solder"],
    "Al-Cr-Fe-Ni": ["alcrfeni", "al-cr-fe-ni", "inconel 718", "in718", "nickel superalloy"],
    "high entropy alloy": ["hea", "multi-principal element alloy", "mpea", "cocrfeni", "cocrfenimn",
                           "alcocrfeni", "crmnfeconi", "refractory hea"],
}

QUANTITY_PATTERNS = {
    "wavelength": re.compile(r'(\d+(?:\.\d+)?)\s*(?:nm|nanometers?)\s*(?:wavelength|λ|lambda)', re.I),
    "pulse_duration": re.compile(r'(\d+(?:\.\d+)?)\s*(?:fs|femtoseconds?|ps|picoseconds?|ns|nanoseconds?)\s*(?:pulse|duration)', re.I),
    "fluence": re.compile(r'(\d+(?:\.\d+)?)\s*(?:J/cm²|J/cm2|J\s*cm[-²2]|fluence)', re.I),
    "repetition_rate": re.compile(r'(\d+(?:\.\d+)?)\s*(?:kHz|MHz|Hz)\s*(?:repetition|rate|freq)', re.I),
    "spot_size": re.compile(r'(\d+(?:\.\d+)?)\s*(?:µm|um|microns?)\s*(?:spot|diameter|beam\s*radius|waist)', re.I),
    "periodicity": re.compile(r'(\d+(?:\.\d+)?)\s*(?:nm|µm|um|microns?)\s*(?:period|periodicity|spacing|LSFL|HSFL)', re.I),
    "roughness": re.compile(r'(\d+(?:\.\d+)?)\s*(?:nm|µm|um)\s*(?:roughness|Ra|RMS|Rq)', re.I),
    "threshold": re.compile(r'(?:threshold|ablation\s*threshold)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(?:J/cm²|J/cm2|mJ/cm²|GW/cm²|TW/cm²)', re.I),
    "power": re.compile(r'(\d+(?:\.\d+)?)\s*(?:W|mW|kW|MW)\s*(?:power|average\s*power)', re.I),
    "pulse_energy": re.compile(r'(\d+(?:\.\d+)?)\s*(?:µJ|uJ|mJ|nJ)\s*(?:pulse\s*energy|energy\s*per\s*pulse)', re.I),
    "scan_speed": re.compile(r'(\d+(?:\.\d+)?)\s*(?:mm/s|mm/min|m/s)\s*(?:scan\s*speed|travel\s*speed)', re.I),
    "hatch_distance": re.compile(r'(\d+(?:\.\d+)?)\s*(?:µm|um|mm)\s*(?:hatch\s*distance|hatch\s*spacing)', re.I),
    "laser_power": re.compile(r'(\d+(?:\.\d+)?)\s*(?:W)\s*(?:laser\s*power|nominal\s*power)', re.I),
    "component_fraction": re.compile(r'(\d+(?:\.\d+)?)\s*(?:at\.%|wt\.%|at%|wt%)\s*(?:of\s*)?([A-Za-z]+)', re.I),
    "interfacial_energy": re.compile(r'(\d+(?:\.\d+)?)\s*(?:J/m²|J/m2|mJ/m²|mJ/m2)\s*(?:interfacial\s*energy|surface\s*tension)', re.I),
    "thermal_conductivity": re.compile(r'(\d+(?:\.\d+)?)\s*(?:W/(?:m·?K|mK))\s*(?:thermal\s*conductivity)', re.I),
}

METHOD_ALIASES = {
    "sem": ["sem", "scanning electron microscopy", "scanning electron microscope"],
    "afm": ["afm", "atomic force microscopy", "atomic force microscope"],
    "profilometry": ["profilometry", "optical profilometry", "white light interferometry", "wli"],
    "raman": ["raman", "raman spectroscopy", "micro-raman"],
    "xrd": ["xrd", "x-ray diffraction"],
    "edx": ["edx", "eds", "energy dispersive x-ray", "energy-dispersive"],
    "ebsd": ["ebsd", "electron backscatter diffraction"],
    "x-ray_imaging": ["synchrotron x-ray", "x-ray radiography", "x-ray tomography"],
    "phase_field": ["phase-field", "phase field", "pf simulation"],
    "finite_element": ["finite element", "fem", "moose", "abaqus"],
    "calphad": ["calphad", "thermo-calc", "thermocalc", "pandat"],
}

MODEL_MEMORY_ESTIMATES = {
    "gpt2": {"params": "1.5B", "vram_fp16": "~3GB", "vram_4bit": "~1GB", "cpu_ok": True},
    "Qwen/Qwen2-0.5B-Instruct": {"params": "0.5B", "vram_fp16": "~1GB", "vram_4bit": "~400MB", "cpu_ok": True},
    "Qwen/Qwen2.5-0.5B-Instruct": {"params": "0.5B", "vram_fp16": "~1GB", "vram_4bit": "~400MB", "cpu_ok": True},
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {"params": "1.1B", "vram_fp16": "~2.5GB", "vram_4bit": "~800MB", "cpu_ok": True},
    "Qwen/Qwen2.5-1.5B-Instruct": {"params": "1.5B", "vram_fp16": "~3.5GB", "vram_4bit": "~1.2GB", "cpu_ok": False},
    "Qwen/Qwen2.5-3B-Instruct": {"params": "3B", "vram_fp16": "~6GB", "vram_4bit": "~2GB", "cpu_ok": False},
    "mistralai/Mistral-7B-Instruct-v0.3": {"params": "7B", "vram_fp16": "~14GB", "vram_4bit": "~4.5GB", "cpu_ok": False},
    "meta-llama/Llama-3.2-3B-Instruct": {"params": "3B", "vram_fp16": "~6GB", "vram_4bit": "~2GB", "cpu_ok": False},
    "Qwen/Qwen2.5-7B-Instruct": {"params": "7B", "vram_fp16": "~14GB", "vram_4bit": "~4.5GB", "cpu_ok": False},
    "Qwen/Qwen2.5-14B-Instruct": {"params": "14B", "vram_fp16": "~28GB", "vram_4bit": "~9GB", "cpu_ok": False},
    "meta-llama/Llama-3.1-8B-Instruct": {"params": "8B", "vram_fp16": "~16GB", "vram_4bit": "~5GB", "cpu_ok": False},
    "google/gemma-2-9b-it": {"params": "9B", "vram_fp16": "~18GB", "vram_4bit": "~6GB", "cpu_ok": False},
    "tiiuae/falcon-7b-instruct": {"params": "7B", "vram_fp16": "~14GB", "vram_4bit": "~4.5GB", "cpu_ok": False},
}

# =============================================
# BIBLIOGRAPHIC METADATA
# =============================================

class BibliographicMetadata:
    DOI_PATTERN = re.compile(r'\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b', re.IGNORECASE)
    ARXIV_PATTERN = re.compile(r'\barXiv[:\s]+(\d{4}\.\d{4,5}(v\d+)?)\b', re.IGNORECASE)
    JOURNAL_PATTERNS = [
        re.compile(r'(?:published in|journal|proc\.?|journal of)\s+([A-Z][A-Za-z\s&\.]+?)(?:,|\.)', re.I),
        re.compile(r'([A-Z][A-Za-z\s&\.]+?\s+(?:Letters?|Journal|Transactions|Review|Proceedings))', re.I),
    ]
    YEAR_PATTERN = re.compile(r'\b((?:19|20)\d{2})\b')
    VOLUME_PATTERN = re.compile(r'(?:vol\.?|volume)\s*(\d+)', re.I)
    ISSUE_PATTERN = re.compile(r'(?:no\.?|issue|iss\.?)\s*(\d+)', re.I)
    AUTHOR_PATTERN = re.compile(
        r'(?:^|by|authors?:\s*)([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*)',
        re.MULTILINE
    )

    def __init__(self, source_filename: str):
        self.source_filename = source_filename
        self.doi: Optional[str] = None
        self.arxiv_id: Optional[str] = None
        self.title: Optional[str] = None
        self.authors: List[str] = []
        self.journal: Optional[str] = None
        self.year: Optional[int] = None
        self.volume: Optional[str] = None
        self.issue: Optional[str] = None
        self.pages: Optional[str] = None
        self.publisher: Optional[str] = None
        self.raw_metadata: Dict[str, Any] = {}
        self.extraction_method: str = "none"
        self.confidence: float = 0.0

    def format_citation(self, style: str = "apa") -> str:
        if self.doi and self.confidence > 0.8:
            if style == "doi":
                return f"DOI:{self.doi}"
            elif style == "short":
                return f"[DOI:{self.doi}]"
        if self.arxiv_id:
            if style in ["doi", "short"]:
                return f"[arXiv:{self.arxiv_id}]"
        if self.authors and self.year:
            first_author = self._format_author_name(self.authors[0])
            et_al = " et al." if len(self.authors) > 1 else ""
            if style == "apa":
                journal_part = f", {self.journal}" if self.journal else ""
                return f"{first_author}{et_al}{journal_part}, {self.year}"
            elif style == "short":
                return f"[{first_author.split()[0]} {self.year}]"
            elif style == "full":
                parts = [f"{first_author}{et_al} ({self.year})"]
                if self.title:
                    parts.append(f'"{self.title}"')
                if self.journal:
                    journal_str = self.journal
                    if self.volume:
                        journal_str += f", {self.volume}"
                        if self.issue:
                            journal_str += f"({self.issue})"
                    parts.append(journal_str)
                if self.pages:
                    parts.append(f"pp. {self.pages}")
                return ". ".join(parts) + "."
        base_name = Path(self.source_filename).stem
        if self.year:
            return f"[{base_name}, {self.year}]"
        return f"[{base_name}]"

    def _format_author_name(self, author_str: str) -> str:
        if "," in author_str:
            parts = [p.strip() for p in author_str.split(",", 1)]
            if len(parts) == 2:
                last, first = parts
                first_initial = first[0] + "." if first else ""
                return f"{last}, {first_initial}"
        return author_str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_filename,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "journal": self.journal,
            "year": self.year,
            "volume": self.volume,
            "issue": self.issue,
            "pages": self.pages,
            "publisher": self.publisher,
            "extraction_method": self.extraction_method,
            "confidence": self.confidence,
            "citation_apa": self.format_citation("apa"),
            "citation_doi": self.format_citation("doi"),
            "citation_full": self.format_citation("full"),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BibliographicMetadata':
        meta = cls(data.get("source", "unknown"))
        meta.doi = data.get("doi")
        meta.arxiv_id = data.get("arxiv_id")
        meta.title = data.get("title")
        meta.authors = data.get("authors", [])
        meta.journal = data.get("journal")
        meta.year = data.get("year")
        meta.volume = data.get("volume")
        meta.issue = data.get("issue")
        meta.pages = data.get("pages")
        meta.publisher = data.get("publisher")
        meta.extraction_method = data.get("extraction_method", "cached")
        meta.confidence = data.get("confidence", 0.5)
        return meta


def extract_metadata_from_pdf_text(text: str, filename: str) -> BibliographicMetadata:
    meta = BibliographicMetadata(filename)
    text_sample = text[:10000]

    doi_match = BibliographicMetadata.DOI_PATTERN.search(text_sample)
    if doi_match:
        meta.doi = doi_match.group(1).lower()
        meta.confidence = max(meta.confidence, 0.9)
        meta.extraction_method = "regex_doi"

    arxiv_match = BibliographicMetadata.ARXIV_PATTERN.search(text_sample)
    if arxiv_match:
        meta.arxiv_id = arxiv_match.group(1)
        meta.confidence = max(meta.confidence, 0.85)

    year_matches = BibliographicMetadata.YEAR_PATTERN.findall(text_sample)
    for year_str in year_matches:
        year = int(year_str)
        if 1900 <= year <= 2030:
            year_pos = text_sample.find(year_str)
            context = text_sample[max(0, year_pos - 50):year_pos + 50].lower()
            if any(kw in context for kw in ['published', 'received', 'accepted', 'copyright', '©']):
                meta.year = year
                meta.confidence = max(meta.confidence, 0.7)
                break

    for pattern in BibliographicMetadata.JOURNAL_PATTERNS:
        journal_match = pattern.search(text_sample)
        if journal_match:
            journal = journal_match.group(1).strip()
            if len(journal) > 10 and not any(
                bad in journal.lower() for bad in ['introduction', 'abstract', 'references']
            ):
                meta.journal = journal
                meta.confidence = max(meta.confidence, 0.6)
                break

    vol_match = BibliographicMetadata.VOLUME_PATTERN.search(text_sample)
    if vol_match:
        meta.volume = vol_match.group(1)
    iss_match = BibliographicMetadata.ISSUE_PATTERN.search(text_sample)
    if iss_match:
        meta.issue = iss_match.group(1)

    author_section = text_sample[:2000]
    author_matches = BibliographicMetadata.AUTHOR_PATTERN.findall(author_section)
    if author_matches:
        raw_authors = author_matches[0]
        if ',' in raw_authors or ' and ' in raw_authors.lower():
            separators = [',', ' and ', ';']
            for sep in separators:
                if sep.lower() in raw_authors.lower():
                    meta.authors = [a.strip() for a in re.split(sep, raw_authors, flags=re.I) if a.strip()]
                    break
        else:
            meta.authors = [raw_authors.strip()]
        if meta.authors:
            meta.confidence = max(meta.confidence, 0.5)

    title_patterns = [
        re.compile(r'(?:^|\n)([A-Z][^.\n]{20,150}(?:\.[^A-Z]|$))'),
        re.compile(r'(?:title:?\s*)([A-Z][^.\n]{20,200}?)\.?(?:\n|$)', re.I),
    ]
    for pattern in title_patterns:
        title_match = pattern.search(text_sample)
        if title_match:
            title = title_match.group(1).strip()
            if 30 < len(title) < 200 and not title.isupper():
                meta.title = title
                meta.confidence = max(meta.confidence, 0.55)
                break

    return meta


def extract_metadata_from_pdf_file(pdf_path: str, filename: str) -> BibliographicMetadata:
    meta = BibliographicMetadata(filename)

    if PYPDF2_AVAILABLE:
        try:
            reader = PdfReader(pdf_path)
            pdf_info = reader.metadata or {}
            field_mapping = {'/Title': 'title', '/Author': 'authors', '/CreationDate': 'year', '/Subject': 'journal'}
            for pdf_field, meta_field in field_mapping.items():
                if pdf_field in pdf_info and pdf_info[pdf_field]:
                    value = str(pdf_info[pdf_field]).strip()
                    if meta_field == 'authors' and value:
                        meta.authors = [a.strip() for a in re.split(r'[;,]', value) if a.strip()]
                    elif meta_field == 'year' and value:
                        year_match = re.search(r'(?:D:)?(\d{4})', value)
                        if year_match:
                            meta.year = int(year_match.group(1))
                    else:
                        setattr(meta, meta_field, value)
            if meta.title or meta.authors:
                meta.confidence = 0.7
                meta.extraction_method = "pdf_metadata"
        except Exception as e:
            st.warning(f"Could not read PDF metadata: {e}")

    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        text_sample = "\n".join([p.page_content for p in pages[:3]])
        text_meta = extract_metadata_from_pdf_text(text_sample, filename)

        for field in ['doi', 'arxiv_id', 'title', 'journal', 'year', 'volume', 'issue']:
            text_val = getattr(text_meta, field)
            current_val = getattr(meta, field)
            if text_val and (not current_val or text_meta.confidence > meta.confidence):
                setattr(meta, field, text_val)
        if text_meta.authors and (not meta.authors or text_meta.confidence > meta.confidence):
            meta.authors = text_meta.authors
        if text_meta.confidence > meta.confidence:
            meta.confidence = text_meta.confidence
            meta.extraction_method = text_meta.extraction_method
    except Exception as e:
        st.warning(f"Text extraction for metadata failed: {e}")

    if PDF2DOI_AVAILABLE and not meta.doi:
        try:
            result = pdf2doi.pdf2doi(pdf_path)
            if isinstance(result, list) and result:
                result = result[0]
            if result and result.get('identifier') and result.get('identifier_type') == 'doi':
                meta.doi = result['identifier']
                meta.confidence = 0.95
                meta.extraction_method = "pdf2doi"
                if result.get('validation_info'):
                    bibtex = result['validation_info']
                    if 'title' in bibtex and not meta.title:
                        meta.title = bibtex.get('title')
                    if 'author' in bibtex and not meta.authors:
                        meta.authors = [a.strip() for a in bibtex['author'].split(' and ')]
                    if 'year' in bibtex and not meta.year:
                        try:
                            meta.year = int(bibtex['year'])
                        except:
                            pass
        except Exception as e:
            st.warning(f"pdf2doi lookup failed: {e}")

    if CROSSREF_AVAILABLE and meta.doi and not meta.journal:
        try:
            cr = CrossrefAPI()
            work = cr.works(ids=meta.doi)
            if work and work.get('message'):
                msg = work['message']
                if not meta.title and msg.get('title'):
                    meta.title = msg['title'][0] if isinstance(msg['title'], list) else msg['title']
                if not meta.authors and msg.get('author'):
                    meta.authors = [
                        f"{a.get('family', '')} {a.get('given', '')}".strip()
                        for a in msg['author']
                    ]
                if not meta.journal and msg.get('container-title'):
                    meta.journal = msg['container-title'][0] if isinstance(msg['container-title'], list) else msg['container-title']
                if not meta.year and msg.get('published-print') and msg['published-print'].get('date-parts'):
                    meta.year = msg['published-print']['date-parts'][0][0]
                meta.confidence = 0.98
                meta.extraction_method = "crossref_api"
        except Exception as e:
            st.warning(f"Crossref API lookup failed: {e}")

    return meta


def extract_metadata_from_text_file(text: str, filename: str) -> BibliographicMetadata:
    return extract_metadata_from_pdf_text(text, filename)


class MetadataCache:
    def __init__(self):
        self._cache: Dict[str, BibliographicMetadata] = {}
        self._file_hashes: Dict[str, str] = {}

    def get(self, filename: str, file_hash: str = None) -> Optional[BibliographicMetadata]:
        if filename in self._cache:
            if file_hash is None or self._file_hashes.get(filename) == file_hash:
                return self._cache[filename]
        return None

    def set(self, filename: str, metadata: BibliographicMetadata, file_hash: str = None):
        self._cache[filename] = metadata
        if file_hash:
            self._file_hashes[filename] = file_hash

    def clear(self):
        self._cache.clear()
        self._file_hashes.clear()


metadata_cache = MetadataCache()


def compute_file_hash(filepath: str) -> str:
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return ""


# =============================================
# REASONING: SCIENTIFIC ENTITY EXTRACTION
# =============================================

class ScientificEntity:
    def __init__(self, text: str, label: str, value: Optional[float], unit: Optional[str],
                 doc_source: str, chunk_id: int, context: str, confidence: float = 1.0):
        self.text = text
        self.label = label
        self.value = value
        self.unit = unit
        self.doc_source = doc_source
        self.chunk_id = chunk_id
        self.context = context
        self.confidence = confidence
        self.normalized = self._normalize()

    def _normalize(self) -> str:
        text = self.text.lower().strip()
        for canonical, aliases in MATERIAL_ALIASES.items():
            if any(alias in text for alias in aliases):
                return canonical
        for canonical, aliases in METHOD_ALIASES.items():
            if any(alias in text for alias in aliases):
                return canonical
        text = re.sub(r'\s+', '', text)
        return text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text, "label": self.label, "value": self.value, "unit": self.unit,
            "doc_source": self.doc_source, "chunk_id": self.chunk_id,
            "normalized": self.normalized, "confidence": self.confidence
        }


class ScientificClaim:
    def __init__(self, claim_text: str, subject: str, predicate: str, object_val: str,
                 doc_source: str, chunk_id: int, confidence: float):
        self.claim_text = claim_text
        self.subject = subject
        self.predicate = predicate
        self.object_val = object_val
        self.doc_source = doc_source
        self.chunk_id = chunk_id
        self.confidence = confidence
        self.supporting: List[Tuple[str, int]] = []
        self.contradicting: List[Tuple[str, int]] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim_text, "subject": self.subject, "predicate": self.predicate,
            "object": self.object_val, "source": self.doc_source, "confidence": self.confidence,
            "supporting_count": len(self.supporting), "contradicting_count": len(self.contradicting)
        }


class CrossDocumentKnowledgeGraph:
    def __init__(self):
        self.entities: Dict[str, List[ScientificEntity]] = defaultdict(list)
        self.claims: List[ScientificClaim] = []
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)

    def add_document(self, doc_id: str, chunks: List[Document], bib_meta: BibliographicMetadata):
        self.documents[doc_id] = {
            "bib_meta": bib_meta.to_dict(),
            "chunk_count": len(chunks),
            "topics": set()
        }

        for i, chunk in enumerate(chunks):
            entities = self._extract_entities_from_chunk(chunk, i)
            for ent in entities:
                self.entities[ent.normalized].append(ent)
                self.entity_index[ent.normalized].add(doc_id)
                self.documents[doc_id]["topics"].add(ent.label)

            claims = self._extract_claims_from_chunk(chunk, i)
            for claim in claims:
                self.claims.append(claim)

    def _extract_entities_from_chunk(self, chunk: Document, chunk_id: int) -> List[ScientificEntity]:
        text = chunk.page_content
        doc = chunk.metadata.get("source", "unknown")
        entities = []

        for param_name, pattern in QUANTITY_PATTERNS.items():
            for match in pattern.finditer(text):
                val_str = match.group(1)
                try:
                    val = float(val_str)
                except:
                    val = None
                unit_match = re.search(r'(nm|µm|um|fs|ps|ns|J/cm²|J/cm2|kHz|MHz|W|mW|mJ|µJ|uJ)', match.group(0), re.I)
                unit = unit_match.group(1) if unit_match else None

                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end].replace('\n', ' ')

                ent = ScientificEntity(
                    text=match.group(0), label=param_name, value=val, unit=unit,
                    doc_source=doc, chunk_id=chunk_id, context=context,
                    confidence=0.85
                )
                entities.append(ent)

        text_lower = text.lower()
        for canonical, aliases in MATERIAL_ALIASES.items():
            for alias in aliases:
                for match in re.finditer(r'\b' + re.escape(alias) + r'\b', text_lower):
                    start = max(0, match.start() - 80)
                    end = min(len(text), match.end() + 80)
                    context = text[start:end]
                    ent = ScientificEntity(
                        text=alias, label="MATERIAL", value=None, unit=None,
                        doc_source=doc, chunk_id=chunk_id, context=context,
                        confidence=0.9
                    )
                    entities.append(ent)

        for canonical, aliases in METHOD_ALIASES.items():
            for alias in aliases:
                for match in re.finditer(r'\b' + re.escape(alias) + r'\b', text_lower):
                    start = max(0, match.start() - 80)
                    end = min(len(text), match.end() + 80)
                    context = text[start:end]
                    ent = ScientificEntity(
                        text=alias, label="METHOD", value=None, unit=None,
                        doc_source=doc, chunk_id=chunk_id, context=context,
                        confidence=0.9
                    )
                    entities.append(ent)

        return entities

    def _extract_claims_from_chunk(self, chunk: Document, chunk_id: int) -> List[ScientificClaim]:
        text = chunk.page_content
        doc = chunk.metadata.get("source", "unknown")
        claims = []

        claim_patterns = [
            (r'(?:ablation\s*threshold|threshold\s*fluence)\s*(?:of|for)\s+([a-z\s]+?)\s+(?:is|was|were|are|≈|~|about)\s+(\d+\.?\d*\s*[A-Za-z/²]+)', 'has_ablation_threshold'),
            (r'([a-z\s]+?)\s+(?:exhibits|shows|displays|forms|produces)\s+([a-z\s]+?(?:ripples|LIPSS|structures|morphology))', 'exhibits_morphology'),
            (r'(?:periodicity|period|spacing)\s*(?:of|for)\s+([a-z\s]+?)\s+(?:is|was|≈|~)\s+(\d+\.?\d*\s*(?:nm|µm|um))', 'has_periodicity'),
            (r'(?:roughness|Ra)\s*(?:of|for)\s+([a-z\s]+?)\s+(?:is|was|≈|~)\s+(\d+\.?\d*\s*(?:nm|µm|um))', 'has_roughness'),
        ]

        for pattern, predicate in claim_patterns:
            for match in re.finditer(pattern, text, re.I):
                subject = match.group(1).strip()
                obj = match.group(2).strip()
                start = max(0, match.start() - 120)
                end = min(len(text), match.end() + 120)
                context = text[start:end]

                claim = ScientificClaim(
                    claim_text=context, subject=subject, predicate=predicate,
                    object_val=obj, doc_source=doc, chunk_id=chunk_id,
                    confidence=0.7
                )
                claims.append(claim)

        return claims

    def find_consensus(self, entity_normalized: str) -> Optional[Dict[str, Any]]:
        ents = self.entities.get(entity_normalized, [])
        if len(ents) < 2:
            return None

        by_doc = defaultdict(list)
        for e in ents:
            by_doc[e.doc_source].append(e)

        if len(by_doc) < 2:
            return None

        values = [e.value for e in ents if e.value is not None]
        if not values:
            return None

        return {
            "entity": entity_normalized,
            "doc_count": len(by_doc),
            "value_count": len(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "unit": ents[0].unit,
            "sources": list(by_doc.keys())
        }

    def find_contradictions(self, entity_normalized: str, threshold_factor: float = 2.0) -> List[Dict[str, Any]]:
        ents = self.entities.get(entity_normalized, [])
        by_doc = defaultdict(list)
        for e in ents:
            if e.value is not None:
                by_doc[e.doc_source].append(e.value)

        contradictions = []
        docs = list(by_doc.keys())
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                vals_i = by_doc[docs[i]]
                vals_j = by_doc[docs[j]]
                mean_i, mean_j = np.mean(vals_i), np.mean(vals_j)
                if mean_i > 0 and mean_j > 0:
                    ratio = max(mean_i, mean_j) / min(mean_i, mean_j)
                    if ratio > threshold_factor:
                        contradictions.append({
                            "entity": entity_normalized,
                            "doc_a": docs[i], "mean_a": mean_i,
                            "doc_b": docs[j], "mean_b": mean_j,
                            "ratio": ratio,
                            "severity": "high" if ratio > 5 else "moderate"
                        })
        return contradictions

    def get_related_chunks(self, query_entities: List[str], chunks: List[Document],
                          depth: int = 2) -> List[Tuple[Document, float, str]]:
        related_docs = set()
        for ent_norm in query_entities:
            related_docs.update(self.entity_index.get(ent_norm, set()))

        scored = []
        for chunk in chunks:
            doc = chunk.metadata.get("source", "unknown")
            score = 0.0
            reason = "semantic"

            chunk_text = chunk.page_content.lower()
            for ent_norm in query_entities:
                if ent_norm in chunk_text:
                    score += 0.3

            if doc in related_docs:
                score += 0.2
                reason = "cross-doc-link"

            for claim in self.claims:
                if claim.doc_source == doc and claim.chunk_id == chunk.metadata.get("chunk_index", -1):
                    if any(ent in claim.subject.lower() or ent in claim.object_val.lower()
                           for ent in query_entities):
                        score += 0.25
                        reason = "claim-evidence"

            if score > 0:
                scored.append((chunk, score, reason))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def get_knowledge_summary(self) -> Dict[str, Any]:
        return {
            "total_entities": sum(len(v) for v in self.entities.values()),
            "unique_entities": len(self.entities),
            "total_claims": len(self.claims),
            "document_count": len(self.documents),
            "top_entities": Counter([e.normalized for ents in self.entities.values() for e in ents]).most_common(10),
            "consensus_topics": [k for k, v in self.entities.items() if len(self.entity_index.get(k, set())) > 1]
        }


# =============================================
# SEMANTIC CHUNKING WITH STRUCTURE AWARENESS
# =============================================

def detect_scientific_sections(text: str) -> List[Tuple[str, str]]:
    section_patterns = [
        (r'(?:^|\n)\s*Abstract\s*\n', 'ABSTRACT'),
        (r'(?:^|\n)\s*1\.\s*Introduction\s*\n', 'INTRODUCTION'),
        (r'(?:^|\n)\s*(?:2\.)?\s*Experimental\s*(?:Setup|Methods|Details)?\s*\n', 'METHODS'),
        (r'(?:^|\n)\s*(?:3\.)?\s*Results\s*(?:and\s*Discussion)?\s*\n', 'RESULTS'),
        (r'(?:^|\n)\s*(?:4\.)?\s*Discussion\s*\n', 'DISCUSSION'),
        (r'(?:^|\n)\s*Conclusion', 'CONCLUSION'),
    ]

    boundaries = []
    for pattern, name in section_patterns:
        for match in re.finditer(pattern, text, re.I):
            boundaries.append((match.start(), name))

    if not boundaries:
        return [("BODY", text)]

    boundaries.sort()

    sections = []
    for i, (pos, name) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        section_text = text[pos:end].strip()
        if len(section_text) > 50:
            sections.append((name, section_text))

    return sections if sections else [("BODY", text)]


def semantic_chunk_document(pages: List[Document], filename: str) -> List[Document]:
    all_text = "\n\n".join([p.page_content for p in pages])
    sections = detect_scientific_sections(all_text)

    chunks = []
    for section_name, section_text in sections:
        if section_name in ['ABSTRACT', 'CONCLUSION']:
            chunk_size, overlap = 400, 50
        elif section_name == 'METHODS':
            chunk_size, overlap = 600, 100
        else:
            chunk_size, overlap = LASER_DOMAIN_CONFIG["chunk_size"], LASER_DOMAIN_CONFIG["chunk_overlap"]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", "; ", ", "],
            length_function=len
        )

        section_chunks = splitter.create_documents([section_text])
        for i, chunk in enumerate(section_chunks):
            chunk.metadata.update({
                "source": filename,
                "section": section_name,
                "chunk_index": len(chunks) + i,
                "section_chunk_index": i,
            })
        chunks.extend(section_chunks)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)

    return chunks


# =============================================
# SESSION STATE INITIALIZATION (extended)
# =============================================

def initialize_session_state():
    defaults = {
        "processed_files": set(),
        "vectorstore": None,
        "all_chunks": [],
        "messages": [],
        "llm_model_choice": None,
        "llm_tokenizer": None,
        "llm_model": None,
        "llm_backend": None,
        "llm_device_or_host": None,
        "llm_backend_type": None,
        "embeddings": None,
        "processing_complete": False,
        "laser_domain_boost": True,
        "show_sources": True,
        "citation_style": "apa",
        "max_retrieved_chunks": 6,
        "use_4bit_quantization": True,
        "ollama_host": "http://localhost:11434",
        "metadata_cache": metadata_cache,
        "knowledge_graph": None,
        "reasoning_mode": True,
        "show_reasoning_chain": True,
        "cross_doc_consensus": True,
        "feedback_map": {},          # chunk_id -> relevance (1/0)
        "precision_recall": None,
        "show_network": False,
        "selected_entity": None,
        "plot_code": "",
        "last_plot_fig": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================
# UTILITY FUNCTIONS
# =============================================

def is_ollama_model(model_key: str) -> bool:
    return model_key.startswith("ollama:") or model_key.startswith("[Ollama]")

def extract_ollama_tag(model_key: str) -> str:
    if model_key.startswith("ollama:"):
        return model_key.replace("ollama:", "", 1)
    elif model_key.startswith("[Ollama]"):
        match = re.search(r'\]\s*([^\s(]+)', model_key)
        if match:
            return match.group(1)
    return model_key

def get_hf_repo_id(model_key: str) -> str:
    if ":" in model_key and not model_key.startswith("http"):
        parts = model_key.split(":", 1)
        if len(parts) == 2 and "/" in parts[1]:
            return parts[1].strip()
    return model_key

def get_available_gpu_memory() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        return total_memory - reserved
    except:
        return None

def estimate_model_memory(model_key: str, use_4bit: bool = False) -> Dict[str, any]:
    repo_id = get_hf_repo_id(model_key) if not is_ollama_model(model_key) else model_key
    return MODEL_MEMORY_ESTIMATES.get(repo_id, {
        "params": "Unknown", "vram_fp16": "Unknown", "vram_4bit": "Unknown", "cpu_ok": False
    })


# =============================================
# LOCAL MODEL LOADING
# =============================================

@st.cache_resource(show_spinner="Loading local embedding model (~80MB)...")
def load_local_embeddings():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    except Exception as e:
        st.error(f"Failed to load embeddings: {e}")
        return None

@st.cache_resource(show_spinner="Loading local LLM (this may take 1-2 minutes on first load)...")
def load_local_llm(model_key: str, use_4bit: bool = True):
    try:
        if is_ollama_model(model_key):
            return _load_ollama_model(model_key)
        else:
            return _load_transformers_model(model_key, use_4bit)
    except Exception as e:
        st.error(f"Failed to load LLM '{model_key}': {e}")
        st.warning("Falling back to GPT-2...")
        try:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            return tokenizer, model, device, "transformers"
        except Exception as e2:
            st.error(f"Fallback also failed: {e2}")
            return None, None, None, None

def _load_ollama_model(model_key: str):
    if not OLLAMA_AVAILABLE:
        raise ImportError("ollama library not installed. Run: pip install ollama")
    model_tag = extract_ollama_tag(model_key)
    try:
        client = ollama.Client(host=st.session_state.ollama_host)
        response = client.list()
        models_list = response.get('models', []) if isinstance(response, dict) else getattr(response, 'models', [])
        model_names = []
        for m in models_list:
            if isinstance(m, dict):
                name = m.get('model') or m.get('name')
            else:
                name = getattr(m, 'model', None) or getattr(m, 'name', None)
            if name:
                model_names.append(name)
        if model_tag not in model_names:
            st.warning(f"⚠️ Model '{model_tag}' not found in Ollama.")
            if model_names:
                st.info(f"📋 Available: {', '.join(model_names[:5])}")
            return None, None, st.session_state.ollama_host, "ollama"
    except Exception as conn_err:
        st.error(f"❌ Connection Error: {conn_err}")
        return None, None, st.session_state.ollama_host, "ollama"
    return None, model_tag, st.session_state.ollama_host, "ollama"

def _load_transformers_model(model_key: str, use_4bit: bool = True):
    repo_id = get_hf_repo_id(model_key)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    available_vram = get_available_gpu_memory()
    mem_info = estimate_model_memory(model_key, use_4bit)
    st.sidebar.info(f"""
    📊 Model Memory Estimate:
    - Parameters: {mem_info['params']}
    - VRAM (FP16): {mem_info['vram_fp16']}
    - VRAM (4-bit): {mem_info['vram_4bit']}
    - CPU OK: {'✅ Yes' if mem_info['cpu_ok'] else '❌ No'}
    - Available VRAM: {f'{available_vram:.1f}GB' if available_vram else 'N/A (CPU)'}
    """)
    if "0.5B" in repo_id or "1.1B" in repo_id or "gpt2" in repo_id:
        use_4bit = False
    quantization_config = None
    if use_4bit and device == "cuda" and available_vram:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            st.sidebar.success("✅ 4-bit quantization enabled")
        except ImportError:
            st.sidebar.warning("⚠️ bitsandbytes not installed.")
            use_4bit = False
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, trust_remote_code=True, padding_side="left", use_fast=True
    )
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
    }
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    elif device == "cuda":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(repo_id, **model_kwargs)
    if "device_map" not in model_kwargs and device == "cpu":
        model = model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model, device, "transformers"


# =============================================
# DOCUMENT PROCESSING
# =============================================

def extract_laser_metadata(text: str, filename: str) -> Dict[str, any]:
    metadata = {
        "source": filename,
        "laser_topics": [],
        "parameters_found": {},
        "has_equations": bool(re.search(r'[\(=]\s*[\d.]+\s*[×*]\s*10\^', text)),
        "has_figures": bool(re.search(r'Figure\s*\d+|Fig\.\s*\d+', text, re.I)),
    }
    text_lower = text.lower()
    for topic, keywords in LASER_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            metadata["laser_topics"].append(topic)
    param_patterns = {
        "wavelength_nm": r'(\d+(?:\.\d+)?)\s*(?:nm|nanometers?)\s*(?:wavelength|λ|lambda)',
        "pulse_duration_fs": r'(\d+(?:\.\d+)?)\s*(?:fs|femtoseconds?)\s*(?:pulse|duration)',
        "fluence_Jcm2": r'(\d+(?:\.\d+)?)\s*(?:J/cm²|J/cm2|fluence)',
        "repetition_rate": r'(\d+(?:\.\d+)?)\s*(?:kHz|MHz|Hz)\s*(?:repetition|rate|freq)',
        "spot_size_um": r'(\d+(?:\.\d+)?)\s*(?:µm|um|microns?)\s*(?:spot|diameter)',
    }
    for param, pattern in param_patterns.items():
        match = re.search(pattern, text, re.I)
        if match:
            try:
                metadata["parameters_found"][param] = float(match.group(1))
            except:
                pass
    return metadata


def load_and_chunk_laser_documents(uploaded_files: List) -> Tuple[List[Document], CrossDocumentKnowledgeGraph]:
    all_chunks = []
    graph = CrossDocumentKnowledgeGraph()

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf" if uploaded_file.name.endswith('.pdf') else ".txt") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            file_hash = compute_file_hash(tmp_path)
            cached_meta = st.session_state.metadata_cache.get(uploaded_file.name, file_hash)

            if cached_meta:
                bib_meta = cached_meta
                st.info(f"📚 Using cached metadata for `{uploaded_file.name}`")
            else:
                if uploaded_file.name.endswith('.pdf'):
                    bib_meta = extract_metadata_from_pdf_file(tmp_path, uploaded_file.name)
                else:
                    with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text_content = f.read()
                    bib_meta = extract_metadata_from_text_file(text_content, uploaded_file.name)
                st.session_state.metadata_cache.set(uploaded_file.name, bib_meta, file_hash)
                st.info(f"📚 Extracted metadata: {bib_meta.format_citation('apa')}")

            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, encoding='utf-8')

            pages = loader.load()
            chunks = semantic_chunk_document(pages, uploaded_file.name)

            for chunk in chunks:
                chunk.metadata.update({
                    **extract_laser_metadata(chunk.page_content, uploaded_file.name),
                    "bibliographic": bib_meta.to_dict(),
                    "citation_display": bib_meta.format_citation(st.session_state.get('citation_style', 'apa')),
                })

            graph.add_document(uploaded_file.name, chunks, bib_meta)
            all_chunks.extend(chunks)
            st.info(f"✅ Loaded {len(chunks)} semantic chunks from `{uploaded_file.name}`")

        except Exception as e:
            st.error(f"❌ Error processing `{uploaded_file.name}`: {e}")
            import traceback
            st.error(traceback.format_exc())
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return all_chunks, graph


@st.cache_resource
def create_local_vector_store(chunks: List[Document], embedding_model_key: str):
    try:
        embeddings = load_local_embeddings()
        if embeddings is None:
            return None
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.metadata = {
            "total_chunks": len(chunks),
            "embedding_model": embedding_model_key,
            "created_at": datetime.now().isoformat(),
            "laser_topics": list(set(
                topic for chunk in chunks for topic in chunk.metadata.get("laser_topics", [])
            ))
        }
        return vectorstore
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None


# =============================================
# RAG FUNCTIONS (with plot generation)
# =============================================

def extract_query_entities(query: str) -> List[str]:
    entities = []
    query_lower = query.lower()

    for canonical, aliases in MATERIAL_ALIASES.items():
        if any(alias in query_lower for alias in aliases):
            entities.append(canonical)

    for canonical, aliases in METHOD_ALIASES.items():
        if any(alias in query_lower for alias in aliases):
            entities.append(canonical)

    for param_name in QUANTITY_PATTERNS.keys():
        if param_name.replace("_", " ") in query_lower or param_name in query_lower:
            entities.append(param_name)

    for topic, keywords in LASER_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            entities.append(topic)

    return entities


def create_scientific_reasoning_prompt(
    retrieved_chunks: List[Document],
    query: str,
    graph: CrossDocumentKnowledgeGraph,
    consensus_data: List[Dict],
    contradictions: List[Dict]
) -> str:
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        citation = chunk.metadata.get("citation_display")
        if not citation:
            source = chunk.metadata.get("source", "unknown")
            citation = f"[Source {i} - {source}]"
        section = chunk.metadata.get("section", "UNKNOWN")
        content = chunk.page_content[:600] + "..." if len(chunk.page_content) > 600 else chunk.page_content
        context_parts.append(f"---\n[{i}] {citation} | Section: {section}\n{content}\n")
    context = "\n".join(context_parts)

    consensus_text = ""
    if consensus_data:
        consensus_text = "\nCross-Document Consensus (statistical agreement across papers):\n"
        for cons in consensus_data[:3]:
            consensus_text += f"- {cons['entity']}: {cons['mean']:.2f} ± {cons['std']:.2f} {cons['unit']} (across {cons['doc_count']} papers, n={cons['value_count']})\n"

    contradiction_text = ""
    if contradictions:
        contradiction_text = "\nDetected Contradictions Across Documents:\n"
        for contr in contradictions[:3]:
            contradiction_text += f"- {contr['entity']}: {contr['doc_a']} reports {contr['mean_a']:.2f} vs {contr['doc_b']} reports {contr['mean_b']:.2f} (ratio: {contr['ratio']:.1f}x, {contr['severity']})\n"

    system_prompt = """You are an expert scientific research assistant specializing in laser-microstructure interactions, with a focus on multicomponent alloys, additive manufacturing, and physics-informed digital twins.
Your task is to synthesize evidence from multiple research papers and provide a scientifically rigorous answer.

REASONING RULES:
1. SYNTHESIZE across documents — do not just summarize one paper at a time
2. Identify CONSENSUS where multiple papers agree, and CONTRADICTIONS where they disagree
3. Report UNCERTAINTY explicitly — use phrases like "reported values range from X to Y", "the consensus mean is Z ± σ"
4. Cite sources using the EXACT citation format provided (Author et al., Journal, Year)
5. If evidence is insufficient or contradictory, state this explicitly rather than fabricating consensus
6. Distinguish between direct experimental results and inferred/theoretical claims
7. For numerical values, include units and note if papers use different measurement conditions

OUTPUT STRUCTURE:
1. **Direct Answer**: Concise answer to the question
2. **Evidence Synthesis**: Integration of findings across papers with citations
3. **Consensus & Variability**: Statistical summary if multiple papers report the same parameter
4. **Contradictions & Limitations**: Note any conflicting results or methodological differences
5. **Confidence Assessment**: State your confidence (High/Medium/Low) and why
"""
    user_prompt = f"""Retrieved Document Context:
{context}
{consensus_text}
{contradiction_text}

User Question: {query}

Provide a scientifically rigorous answer following the structure above. Be precise about uncertainty and cross-document agreement."""
    return system_prompt + user_prompt


def generate_local_response_transformers(tokenizer, model, device: str, prompt: str, backend_name: str) -> str:
    try:
        if "Qwen" in backend_name or "qwen" in backend_name.lower():
            messages = [
                {"role": "system", "content": "You are an expert in laser-microstructure interaction research. Synthesize evidence across multiple papers rigorously."},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        elif "Llama" in backend_name or "llama" in backend_name.lower():
            messages = [
                {"role": "system", "content": "You are an expert in laser-microstructure interaction research. Synthesize evidence across multiple papers rigorously."},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        elif "Mistral" in backend_name or "mistral" in backend_name.lower():
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        else:
            formatted_prompt = prompt

        inputs = tokenizer.encode(
            formatted_prompt, return_tensors='pt', truncation=True,
            max_length=LASER_DOMAIN_CONFIG["max_context_tokens"]
        )
        if device == "cuda" and torch.cuda.is_available():
            inputs = inputs.to('cuda')

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=LASER_DOMAIN_CONFIG["max_new_tokens"],
                temperature=LASER_DOMAIN_CONFIG["temperature"],
                do_sample=(LASER_DOMAIN_CONFIG["temperature"] > 0),
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "[/INST]" in full_text:
            answer = full_text.split("[/INST]")[-1].strip()
        elif "Confidence Assessment:" in full_text:
            answer = full_text[full_text.find("Direct Answer:"):].strip() if "Direct Answer:" in full_text else full_text[-1500:].strip()
        else:
            answer = full_text[-LASER_DOMAIN_CONFIG["max_new_tokens"] * 2:].strip()

        answer = re.sub(r'\s+', ' ', answer).strip()
        return answer if answer else "I was unable to generate a response. Please try rephrasing your question."

    except Exception as e:
        st.error(f"Generation error: {e}")
        return f"Error generating response: {str(e)[:200]}..."


def generate_local_response_ollama(model_tag: str, ollama_host: str, prompt: str) -> str:
    try:
        client = ollama.Client(host=ollama_host)
        messages = [
            {"role": "system", "content": "You are an expert in laser-microstructure interaction research. Synthesize evidence across multiple papers rigorously."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = client.chat(
                model=model_tag, messages=messages,
                options={"temperature": LASER_DOMAIN_CONFIG["temperature"], "num_predict": LASER_DOMAIN_CONFIG["max_new_tokens"]},
                stream=True
            )
            full_response = ""
            for chunk in response:
                if isinstance(chunk, dict):
                    if 'message' in chunk and 'content' in chunk['message']:
                        full_response += chunk['message']['content']
                    elif 'content' in chunk:
                        full_response += chunk['content']
                elif hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
                    full_response += chunk.message.content
        except TypeError:
            response = client.chat(
                model=model_tag, messages=messages,
                options={"temperature": LASER_DOMAIN_CONFIG["temperature"], "num_predict": LASER_DOMAIN_CONFIG["max_new_tokens"]}
            )
            if isinstance(response, dict):
                full_response = response.get('message', {}).get('content', '')
            elif hasattr(response, 'message'):
                full_response = response.message.content
            else:
                full_response = str(response)

        return full_response.strip() if full_response.strip() else "I was unable to generate a response. Please try rephrasing your question."
    except Exception as e:
        st.error(f"Ollama generation error: {e}")
        return f"Error generating response via Ollama: {str(e)[:200]}..."


def generate_local_response(tokenizer, model_or_tag, device_or_host: str, prompt: str, backend: str, backend_type: str) -> str:
    if backend_type == "ollama":
        return generate_local_response_ollama(model_or_tag, device_or_host, prompt)
    else:
        return generate_local_response_transformers(tokenizer, model_or_tag, device_or_host, prompt, backend)


def retrieve_and_answer(
    vectorstore,
    graph: CrossDocumentKnowledgeGraph,
    tokenizer,
    model,
    device_or_host: str,
    backend: str,
    backend_type: str,
    query: str,
    k: int = None,
    score_threshold: float = None
) -> Tuple[str, List[Document], float, Dict[str, Any]]:
    k = k or LASER_DOMAIN_CONFIG["retrieval_k"]
    score_threshold = score_threshold or LASER_DOMAIN_CONFIG["score_threshold"]

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k * 2, "score_threshold": score_threshold}
    )
    semantic_docs = retriever.invoke(query)

    query_entities = extract_query_entities(query)

    if graph and query_entities and st.session_state.get("reasoning_mode", True):
        graph_results = graph.get_related_chunks(query_entities, st.session_state.all_chunks, depth=2)
        seen = {(d.metadata.get("source"), d.metadata.get("chunk_index")) for d in semantic_docs}
        for chunk, score, reason in graph_results:
            key = (chunk.metadata.get("source"), chunk.metadata.get("chunk_index"))
            if key not in seen and len(semantic_docs) < k * 2:
                semantic_docs.append(chunk)
                seen.add(key)

    if semantic_docs:
        query_embedding = vectorstore.embedding_function.embed_query(query)
        scored_docs = []
        for doc in semantic_docs:
            doc_embedding = vectorstore.embedding_function.embed_query(doc.page_content[:500])
            sim = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-8
            )
            section_boost = 0.05 if doc.metadata.get("section") in ["RESULTS", "DISCUSSION"] else 0
            scored_docs.append((doc, sim + section_boost))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        retrieved_docs = [d for d, s in scored_docs[:k]]
        avg_relevance = np.mean([s for d, s in scored_docs[:k]])
    else:
        retrieved_docs = []
        avg_relevance = 0.0

    if not retrieved_docs:
        return "Based on the uploaded documents, I could not find information relevant to your question.", [], avg_relevance, {}

    consensus_data = []
    contradictions = []
    if graph and st.session_state.get("cross_doc_consensus", True):
        for ent in query_entities:
            cons = graph.find_consensus(ent)
            if cons:
                consensus_data.append(cons)
            contr = graph.find_contradictions(ent, threshold_factor=1.5)
            contradictions.extend(contr)

    prompt = create_scientific_reasoning_prompt(retrieved_docs, query, graph, consensus_data, contradictions)
    answer = generate_local_response(
        tokenizer=tokenizer, model_or_tag=model, device_or_host=device_or_host,
        prompt=prompt, backend=backend, backend_type=backend_type
    )

    reasoning_meta = {
        "query_entities": query_entities,
        "consensus_found": len(consensus_data),
        "contradictions_found": len(contradictions),
        "multi_hop_expansion": len(semantic_docs) > k,
    }

    return answer, retrieved_docs, avg_relevance, reasoning_meta


# =============================================
# NEW: VISUALIZATION HELPERS
# =============================================

def build_network_html(graph: CrossDocumentKnowledgeGraph) -> str:
    if not PYVIS_AVAILABLE:
        return "<p>Install pyvis to see network graph: pip install pyvis</p>"
    net = Network(height="500px", width="100%", directed=False)
    for doc_id in graph.documents:
        net.add_node(doc_id, label=Path(doc_id).stem, shape="box", color="#97C2FC")
    entity_counts = Counter([e.normalized for ents in graph.entities.values() for e in ents])
    for ent, cnt in entity_counts.most_common(15):
        net.add_node(ent, label=f"{ent} ({cnt})", shape="ellipse", color="#FFA500")
    for ent_norm, doc_names in graph.entity_index.items():
        if entity_counts[ent_norm] >= 2:
            for doc in doc_names:
                net.add_edge(ent_norm, doc)
    return net.generate_html()

def plot_consensus_chart(consensus_data: List[Dict]) -> go.Figure:
    if not consensus_data:
        return None
    entities = [d['entity'] for d in consensus_data]
    means = [d['mean'] for d in consensus_data]
    stds = [d['std'] for d in consensus_data]
    units = [d['unit'] for d in consensus_data]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=entities, y=means, error_y=dict(type='data', array=stds, visible=True),
        name='Consensus value', text=[f"{m:.2f} ± {s:.2f} ({u})" for m, s, u in zip(means, stds, units)],
        hoverinfo='text'
    ))
    fig.update_layout(title="Cross-document consensus (mean ± std)", yaxis_title="Value", xaxis_title="Entity")
    return fig

def render_contradiction_table(contradictions: List[Dict]) -> pd.DataFrame:
    if not contradictions:
        return pd.DataFrame()
    rows = []
    for c in contradictions:
        rows.append({
            "Entity": c['entity'],
            "Doc A": Path(c['doc_a']).stem,
            "Mean A": f"{c['mean_a']:.3f}",
            "Doc B": Path(c['doc_b']).stem,
            "Mean B": f"{c['mean_b']:.3f}",
            "Ratio": f"{c['ratio']:.1f}x",
            "Severity": c['severity']
        })
    return pd.DataFrame(rows)

def extract_equations_from_text(text: str) -> List[str]:
    eq_pattern = re.compile(r'(\$[^$]+\$|\\\[.*?\\\]|([A-Za-z_\{\}]+)\s*=\s*[^\n]+)')
    matches = eq_pattern.findall(text)
    return [m[0].strip() for m in matches if len(m[0]) > 10]


# =============================================
# STREAMLIT UI (extended with new panels)
# =============================================

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        backend_option = st.radio("🔧 Inference Backend", options=["Hugging Face Transformers", "Ollama (if installed)"], index=0)
        st.session_state.inference_backend = backend_option

        if backend_option == "Ollama (if installed)":
            if not OLLAMA_AVAILABLE:
                st.error("❌ ollama library not installed")
                st.code("pip install ollama")
            available_ollama_models = [k for k in LOCAL_LLM_OPTIONS.keys() if is_ollama_model(k)]
            model_choice = st.selectbox("🧠 Local LLM Backend (Ollama)", options=available_ollama_models if available_ollama_models else ["No Ollama models available"], index=0)
        else:
            hf_models = [k for k in LOCAL_LLM_OPTIONS.keys() if not is_ollama_model(k)]
            model_choice = st.selectbox("🧠 Local LLM Backend (Hugging Face)", options=hf_models, index=2)

        st.session_state.llm_model_choice = model_choice

        if backend_option == "Hugging Face Transformers" and not is_ollama_model(model_choice):
            st.session_state.use_4bit_quantization = st.checkbox("🗜️ Use 4-bit quantization", value=True)

        if backend_option == "Ollama (if installed)" or is_ollama_model(model_choice):
            st.session_state.ollama_host = st.text_input("🌐 Ollama Host", value=st.session_state.ollama_host)

        st.markdown("#### 🔬 Reasoning Settings")
        st.session_state.reasoning_mode = st.checkbox(
            "🧠 Cross-document reasoning", value=True,
            help="Enable entity extraction, consensus detection, and multi-hop retrieval across papers"
        )
        st.session_state.cross_doc_consensus = st.checkbox(
            "📊 Detect consensus & contradictions", value=True,
            help="Statistically compare reported values across documents"
        )
        st.session_state.show_reasoning_chain = st.checkbox(
            "🔍 Show reasoning chain", value=True,
            help="Display the logical steps and evidence linking"
        )

        st.markdown("#### 🔬 Laser Domain Settings")
        st.session_state.laser_domain_boost = st.checkbox("Boost laser-topic relevance", value=True)
        st.session_state.show_sources = st.checkbox("Show source citations", value=True)

        st.markdown("#### 📝 Citation Format")
        st.session_state.citation_style = st.selectbox(
            "Citation display style", options=["apa", "doi", "full", "short"], index=0,
            format_func=lambda x: {"apa": "APA: FirstAuthor et al., Journal, Year", "doi": "DOI: 10.xxxx/xxxxx",
                                   "full": "Full: Author (Year). Title. Journal, Vol(Issue), Pages", "short": "Short: [FirstAuthor Year] or [DOI]"}[x]
        )

        st.session_state.max_retrieved_chunks = st.slider("Chunks to retrieve", min_value=2, max_value=10, value=6)

        st.markdown("---")
        st.markdown("### 🕸️ Visualisations")
        st.session_state.show_network = st.sidebar.checkbox("Show Knowledge Graph", value=False)
        if st.session_state.knowledge_graph and st.session_state.knowledge_graph.entities:
            entity_options = list(st.session_state.knowledge_graph.entities.keys())
            if entity_options:
                st.session_state.selected_entity = st.sidebar.selectbox(
                    "Explore entity consensus",
                    options=entity_options,
                    format_func=lambda x: x,
                    key="selected_entity_select"
                )
        else:
            st.session_state.selected_entity = None

        st.markdown("---")
        gpu_info = "CUDA" if torch.cuda.is_available() else "CPU"
        vram_info = f"{get_available_gpu_memory():.1f}GB free" if torch.cuda.is_available() and get_available_gpu_memory() else "N/A"
        st.caption(f"🖥️ Device: {gpu_info}")
        st.caption(f"💾 Available VRAM: {vram_info}")

        if PDF2DOI_AVAILABLE:
            st.success("✅ pdf2doi: Available")
        else:
            st.info("ℹ️ pdf2doi: Optional for DOI lookup")
        if CROSSREF_AVAILABLE:
            st.success("✅ Crossref API: Available")
        else:
            st.info("ℹ️ Crossref: Optional for metadata enrichment")


def render_document_uploader():
    st.markdown("### 📁 Upload Laser Microstructure Documents")
    uploaded_files = st.file_uploader(
        "Select PDF or TXT files about laser processing, multicomponent alloys, additive manufacturing, etc.",
        type=["pdf", "txt"], accept_multiple_files=True,
        help="Documents will be processed with semantic section detection and cross-document entity linking."
    )
    return uploaded_files


def process_documents(uploaded_files):
    if not uploaded_files:
        return False

    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    if not new_files:
        st.info("✓ All uploaded files already processed")
        return st.session_state.processing_complete

    st.session_state.messages = []
    st.session_state.vectorstore = None
    st.session_state.all_chunks = []
    st.session_state.knowledge_graph = None

    with st.spinner(f"Processing {len(new_files)} document(s) with semantic reasoning..."):
        try:
            chunks, graph = load_and_chunk_laser_documents(new_files)
            if not chunks:
                st.error("No chunks extracted. Check file format.")
                return False

            for f in new_files:
                st.session_state.processed_files.add(f.name)

            st.session_state.all_chunks.extend(chunks)
            st.session_state.knowledge_graph = graph

            with st.spinner("Creating vector index and knowledge graph..."):
                vectorstore = create_local_vector_store(st.session_state.all_chunks, LOCAL_EMBEDDING_MODEL)
                if vectorstore is None:
                    return False
                st.session_state.vectorstore = vectorstore

            if graph:
                summary = graph.get_knowledge_summary()
                st.success(
                    f"✅ Ready! Indexed {len(st.session_state.all_chunks)} chunks, "
                    f"{summary['unique_entities']} unique entities, "
                    f"{summary['total_claims']} claims from {summary['document_count']} papers"
                )
                if summary['consensus_topics']:
                    st.caption(f"🔗 Cross-document consensus available for: {', '.join(summary['consensus_topics'][:5])}")
            else:
                st.success(f"✅ Ready! Indexed {len(st.session_state.all_chunks)} chunks")

            st.session_state.processing_complete = True
            return True

        except Exception as e:
            st.error(f"Processing failed: {e}")
            import traceback
            st.error(traceback.format_exc())
            return False


def render_chat_interface():
    if not st.session_state.get('vectorstore'):
        st.info("👆 Upload documents above to start chatting with cross-document reasoning")
        return

    if st.session_state.llm_tokenizer is None and st.session_state.llm_model_choice:
        backend_type = "ollama" if is_ollama_model(st.session_state.llm_model_choice) else "transformers"
        with st.spinner(f"Loading {st.session_state.llm_model_choice}..."):
            result = load_local_llm(st.session_state.llm_model_choice, use_4bit=st.session_state.get('use_4bit_quantization', True))
            tokenizer, model, device_or_host, loaded_backend = result
            if tokenizer is not None or model is not None:
                st.session_state.llm_tokenizer = tokenizer
                st.session_state.llm_model = model
                st.session_state.llm_device_or_host = device_or_host
                st.session_state.llm_backend_type = loaded_backend
                st.success("✓ Model loaded!")
            else:
                st.error("Failed to load model. Try selecting a different option.")
                return

    has_model = (
        st.session_state.llm_backend_type == "ollama" and st.session_state.llm_model is not None
    ) or (
        st.session_state.llm_backend_type == "transformers" and st.session_state.llm_tokenizer is not None
    )

    if not has_model:
        st.warning("Please select and load a model in the sidebar first")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources") and st.session_state.show_sources:
                with st.expander("📚 Retrieved Sources with Citations"):
                    for i, src in enumerate(message["sources"], 1):
                        citation = src.metadata.get("citation_display", "Unknown source")
                        section = src.metadata.get("section", "UNKNOWN")
                        st.markdown(f"**[{i}]** {citation} | *{section}*")
                        bib = src.metadata.get("bibliographic", {})
                        if bib and any(bib.get(k) for k in ['doi', 'authors', 'journal', 'year']):
                            with st.expander("🔍 Bibliographic Details"):
                                if bib.get('doi'):
                                    st.markdown(f"**DOI:** `{bib['doi']}`")
                                if bib.get('authors'):
                                    st.markdown(f"**Authors:** {', '.join(bib['authors'][:3])}{'...' if len(bib['authors'])>3 else ''}")
                                if bib.get('journal'):
                                    st.markdown(f"**Journal:** {bib['journal']}")
                                if bib.get('year'):
                                    st.markdown(f"**Year:** {bib['year']}")
                        st.markdown(f"> {src.page_content[:300]}...")

            if message.get("reasoning_meta") and st.session_state.show_reasoning_chain and message["role"] == "assistant":
                meta = message["reasoning_meta"]
                with st.expander("🧠 Reasoning Chain"):
                    st.markdown(f"**Query entities detected:** {', '.join(meta.get('query_entities', [])) or 'None'}")
                    st.markdown(f"**Cross-document consensus found:** {meta.get('consensus_found', 0)}")
                    st.markdown(f"**Contradictions detected:** {meta.get('contradictions_found', 0)}")
                    st.markdown(f"**Multi-hop expansion:** {'Yes' if meta.get('multi_hop_expansion') else 'No'}")
                    if meta.get('relevance'):
                        st.markdown(f"**Response relevance:** {meta['relevance']:.2f}/1.0")

    # Plot from answer for the last assistant message
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        last_answer = st.session_state.messages[-1]["content"]
        if st.button("📈 Generate plot from answer (LLM will write Python code)", key="gen_plot"):
            with st.spinner("Generating plot code..."):
                plot_prompt = f"Based on the following scientific answer, write a short Python script using matplotlib to create a relevant plot. Only output the Python code, no explanation.\n\nAnswer:\n{last_answer}\n\nPython code:"
                if st.session_state.llm_backend_type == "transformers" and st.session_state.llm_tokenizer:
                    code_prompt = plot_prompt
                    inputs = st.session_state.llm_tokenizer.encode(code_prompt, return_tensors='pt', truncation=True, max_length=1024)
                    if torch.cuda.is_available():
                        inputs = inputs.to('cuda')
                    with torch.no_grad():
                        outputs = st.session_state.llm_model.generate(inputs, max_new_tokens=300, temperature=0.1)
                    raw_code = st.session_state.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    if "Python code:" in raw_code:
                        raw_code = raw_code.split("Python code:")[-1].strip()
                    st.session_state.plot_code = raw_code
                else:
                    st.session_state.plot_code = "# Ollama backend: replace with actual LLM call"
                try:
                    local_vars = {}
                    exec(st.session_state.plot_code, {"plt": __import__('matplotlib.pyplot')}, local_vars)
                    fig = local_vars.get('plt').gcf()
                    st.session_state.last_plot_fig = fig
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Plot execution error: {e}")
                    st.code(st.session_state.plot_code)

    # Knowledge Graph panel
    if st.session_state.show_network and st.session_state.knowledge_graph:
        st.markdown("### 🕸️ Cross-Document Knowledge Graph")
        if PYVIS_AVAILABLE:
            html = build_network_html(st.session_state.knowledge_graph)
            st.components.v1.html(html, height=550, scrolling=True)
        else:
            st.warning("Install pyvis to display network graph: `pip install pyvis`")

    # Consensus & Contradiction Explorer for selected entity
    if st.session_state.knowledge_graph and st.session_state.selected_entity:
        with st.expander(f"📊 Consensus for '{st.session_state.selected_entity}'", expanded=True):
            cons = st.session_state.knowledge_graph.find_consensus(st.session_state.selected_entity)
            if cons:
                chart = plot_consensus_chart([cons])
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                docs_data = []
                for ent in st.session_state.knowledge_graph.entities.get(st.session_state.selected_entity, []):
                    docs_data.append({"Document": Path(ent.doc_source).stem, "Value": ent.value, "Unit": ent.unit})
                if docs_data:
                    df = pd.DataFrame(docs_data).dropna(subset=["Value"])
                    if not df.empty:
                        st.dataframe(df, use_container_width=True)
            else:
                st.info("No numerical consensus (need multiple documents with quantitative data).")
            contr = st.session_state.knowledge_graph.find_contradictions(st.session_state.selected_entity, threshold_factor=1.5)
            if contr:
                st.markdown("**Contradictions**")
                contr_df = render_contradiction_table(contr)
                if not contr_df.empty:
                    st.dataframe(contr_df.style.applymap(lambda x: 'background-color: #ffcccc' if x == 'high' else '', subset=['Severity']))

    # Retrieval Debugger (sidebar expander)
    with st.sidebar.expander("🔍 Retrieval Debugger", expanded=False):
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            last_meta = st.session_state.messages[-1].get("reasoning_meta", {})
            last_sources = st.session_state.messages[-1].get("sources", [])
            if last_sources:
                st.markdown("**Retrieved chunks (click to view)**")
                for i, doc in enumerate(last_sources):
                    source = doc.metadata.get("source", "unknown")
                    section = doc.metadata.get("section", "")
                    chunk_id = doc.metadata.get("chunk_index", -1)
                    score = last_meta.get('relevance', 0.0) if i == 0 else 0.0
                    with st.expander(f"Chunk {i+1} ({Path(source).stem}, {section}) – score: {score:.3f}"):
                        st.text(doc.page_content[:500])
                        col_fb1, col_fb2 = st.columns(2)
                        with col_fb1:
                            if st.button("👍 Relevant", key=f"rel_{source}_{chunk_id}"):
                                st.session_state.feedback_map[f"{source}_{chunk_id}"] = 1
                        with col_fb2:
                            if st.button("👎 Not relevant", key=f"nrel_{source}_{chunk_id}"):
                                st.session_state.feedback_map[f"{source}_{chunk_id}"] = 0
        if st.button("Compute Precision/Recall"):
            feedbacks = list(st.session_state.feedback_map.values())
            if feedbacks:
                retrieved = len(feedbacks)
                relevant = sum(feedbacks)
                st.session_state.precision_recall = {
                    "precision": relevant / retrieved if retrieved else 0,
                    "recall": relevant / retrieved if retrieved else 0
                }
        if st.session_state.precision_recall:
            st.metric("Precision (user-rated)", f"{st.session_state.precision_recall['precision']:.2%}")
            st.metric("Recall (session)", f"{st.session_state.precision_recall['recall']:.2%}")


def render_footer():
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📚 Example Questions:**")
        st.caption("• What is the effect of composition on IMC growth in Sn‑Ag‑Cu solders during laser soldering?")
        st.caption("• How do multi‑scale simulations predict grain structure in SLM of Al‑Cr‑Fe‑Ni alloys?")
        st.caption("• What contradictions exist regarding the influence of Marangoni convection on porosity formation?")
    with col2:
        st.markdown("**⚡ Reasoning Tips:**")
        st.caption("• Ask comparative questions to trigger consensus detection")
        st.caption("• Query specific alloy families (e.g., 'Sn‑Ag‑Cu', 'AlCrFeNi') to activate entity linking")
        st.caption("• Look for the 🧠 Reasoning Chain expander for transparency")
    with col3:
        st.markdown("**🔐 Privacy & Science:**")
        st.caption("• All processing happens locally")
        st.caption("• Cross-document reasoning uses extracted entities only")
        st.caption("• Uncertainty is explicitly reported, never hidden")



# =============================================
# CHAT-DRIVEN SCIENTIFIC VISUALIZATION SYSTEM
# =============================================
# Integrates with CODE 10's CrossDocumentKnowledgeGraph
# Generates plots dynamically from natural language queries

import streamlit as st
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

# =============================================
# VISUALIZATION QUERY PARSER
# =============================================

class VizQueryParser:
    """Parses natural language queries into structured visualization requests."""

    # Chart type patterns
    CHART_PATTERNS = {
        'bar': re.compile(r'\b(bar chart|bar plot|histogram|count|frequency|distribution)\b', re.I),
        'pie': re.compile(r'\b(pie chart|proportion|percentage|share|composition)\b', re.I),
        'line': re.compile(r'\b(line chart|trend|over time|temporal|evolution|progression)\b', re.I),
        'scatter': re.compile(r'\b(scatter plot|correlation|relationship|vs\.|versus|against)\b', re.I),
        'heatmap': re.compile(r'\b(heatmap|heat map|matrix|correlation matrix|co-occurrence)\b', re.I),
        'radar': re.compile(r'\b(radar chart|spider chart|polar chart|profile|multi-parameter)\b', re.I),
        'box': re.compile(r'\b(box plot|boxplot|distribution|spread|variability|outlier)\b', re.I),
        'bubble': re.compile(r'\b(bubble chart|bubble plot|3d|three dimensional)\b', re.I),
    }

    # Entity category patterns
    ENTITY_CATEGORIES = {
        'multicomponent_alloy': re.compile(r'\b(multicomponent alloy|multi-component alloy|high entropy alloy|hea|mpea|complex concentrated alloy|multi-principal element)\b', re.I),
        'solder': re.compile(r'\b(solder|sn-ag-cu|sac|sn-3\.5ag|lead-free solder|solder joint|soldering)\b', re.I),
        'superalloy': re.compile(r'\b(superalloy|inconel|in718|nimonic|haynes|waspaloy|nickel superalloy)\b', re.I),
        'steel': re.compile(r'\b(steel|stainless steel|ss304|ss316|carbon steel|tool steel|maraging steel)\b', re.I),
        'titanium': re.compile(r'\b(titanium|ti-6al-4v|ti6al4v|cp-ti|ti alloy)\b', re.I),
        'aluminum': re.compile(r'\b(aluminum|aluminium|al-6061|al6061|al alloy|al-mg|al-si)\b', re.I),
        'copper': re.compile(r'\b(copper|cu|cu alloy|brass|bronze)\b', re.I),
        'ceramic': re.compile(r'\b(ceramic|alumina|al2o3|zirconia|zro2|silicon carbide|sic)\b', re.I),
        'polymer': re.compile(r'\b(polymer|pmma|polyimide|pei|pc|polycarbonate|ptfe|peek)\b', re.I),
        'silicon': re.compile(r'\b(silicon|si|c-si|si\(100\)|si\(111\)|crystalline silicon)\b', re.I),
    }

    # Laser parameter patterns
    LASER_PARAM_PATTERNS = {
        'power': re.compile(r'\b(power|laser power|nominal power|average power|wattage)\b', re.I),
        'wavelength': re.compile(r'\b(wavelength|lambda|λ|nm wavelength)\b', re.I),
        'pulse_duration': re.compile(r'\b(pulse duration|pulse width|femtosecond|picosecond|nanosecond|fs laser|ps laser)\b', re.I),
        'repetition_rate': re.compile(r'\b(repetition rate|rep rate|frequency|khz|mhz|hz)\b', re.I),
        'fluence': re.compile(r'\b(fluence|energy density|j/cm²|j/cm2|threshold fluence)\b', re.I),
        'scan_speed': re.compile(r'\b(scan speed|scanning speed|travel speed|mm/s|m/s)\b', re.I),
        'hatch_distance': re.compile(r'\b(hatch distance|hatch spacing|layer thickness|stripe width)\b', re.I),
        'spot_size': re.compile(r'\b(spot size|beam diameter|beam radius|beam waist|focus spot)\b', re.I),
        'pulse_energy': re.compile(r'\b(pulse energy|energy per pulse|μj|mj|nj per pulse)\b', re.I),
    }

    # Method/technique patterns
    METHOD_PATTERNS = {
        'slm': re.compile(r'\b(slm|selective laser melting|laser powder bed fusion|lpbf|powder bed)\b', re.I),
        'ded': re.compile(r'\b(ded|directed energy deposition|laser cladding|laser metal deposition)\b', re.I),
        'soldering': re.compile(r'\b(laser soldering|soldering|reflow|wave soldering)\b', re.I),
        'ablation': re.compile(r'\b(laser ablation|ablation|material removal|micromachining)\b', re.I),
        'annealing': re.compile(r'\b(laser annealing|annealing|heat treatment|thermal treatment)\b', re.I),
        'welding': re.compile(r'\b(laser welding|welding|weld seam|fusion welding)\b', re.I),
        'surface_structuring': re.compile(r'\b(surface structuring|texturing|lipss|ripples|nanostructuring)\b', re.I),
    }

    # Property patterns
    PROPERTY_PATTERNS = {
        'interfacial_energy': re.compile(r'\b(interfacial energy|surface energy|surface tension|γ|gamma)\b', re.I),
        'thermal_conductivity': re.compile(r'\b(thermal conductivity|heat conductivity|k value|w/mk)\b', re.I),
        'diffusion_coefficient': re.compile(r'\b(diffusion coefficient|diffusivity|d value|atomic mobility)\b', re.I),
        'viscosity': re.compile(r'\b(viscosity|dynamic viscosity|kinematic viscosity|η|eta)\b', re.I),
        'hardness': re.compile(r'\b(hardness|microhardness|vickers|hv|nanoindentation)\b', re.I),
        'roughness': re.compile(r'\b(roughness|surface roughness|ra|rq|rms|surface finish)\b', re.I),
        'porosity': re.compile(r'\b(porosity|void fraction|pore density|lack of fusion)\b', re.I),
        'residual_stress': re.compile(r'\b(residual stress|internal stress|stress state|σ|sigma)\b', re.I),
        'grain_size': re.compile(r'\b(grain size|crystallite size|dendrite spacing|cell size|λ1|lambda1)\b', re.I),
        'absorptivity': re.compile(r'\b(absorptivity|absorption|reflectivity|emissivity|a value)\b', re.I),
    }

    # Aggregation patterns
    AGG_PATTERNS = {
        'count': re.compile(r'\b(count|number of|how many|frequency|occurrence)\b', re.I),
        'mean': re.compile(r'\b(average|mean|typical|typical value|central)\b', re.I),
        'range': re.compile(r'\b(range|spread|min and max|minimum maximum|variability)\b', re.I),
        'compare': re.compile(r'\b(compare|comparison|versus|vs|difference between|among|between)\b', re.I),
        'top': re.compile(r'\b(top|most frequent|most common|highest|lowest|ranking)\b', re.I),
    }

    @classmethod
    def parse(cls, query: str) -> Dict[str, Any]:
        """Parse a natural language query into visualization parameters."""
        query_lower = query.lower()
        result = {
            'chart_type': 'bar',  # default
            'entity_filter': None,
            'parameter_focus': None,
            'method_focus': None,
            'property_focus': None,
            'aggregation': 'count',
            'group_by': None,
            'filter_docs': None,
            'x_axis': None,
            'y_axis': None,
            'color_by': None,
            'title': None,
            'confidence': 0.0
        }

        # Detect chart type
        for chart_type, pattern in cls.CHART_PATTERNS.items():
            if pattern.search(query):
                result['chart_type'] = chart_type
                result['confidence'] += 0.3
                break

        # Detect entity filter
        for entity, pattern in cls.ENTITY_CATEGORIES.items():
            if pattern.search(query):
                result['entity_filter'] = entity
                result['confidence'] += 0.25
                break

        # Detect laser parameter focus
        for param, pattern in cls.LASER_PARAM_PATTERNS.items():
            if pattern.search(query):
                result['parameter_focus'] = param
                result['confidence'] += 0.2
                break

        # Detect method focus
        for method, pattern in cls.METHOD_PATTERNS.items():
            if pattern.search(query):
                result['method_focus'] = method
                result['confidence'] += 0.2
                break

        # Detect property focus
        for prop, pattern in cls.PROPERTY_PATTERNS.items():
            if pattern.search(query):
                result['property_focus'] = prop
                result['confidence'] += 0.2
                break

        # Detect aggregation
        for agg, pattern in cls.AGG_PATTERNS.items():
            if pattern.search(query):
                result['aggregation'] = agg
                result['confidence'] += 0.15
                break

        # Detect "only multicomponent" or "among all materials"
        if re.search(r'\b(only|just|among|compared to|relative to|fraction of|proportion of|percentage of)\b', query, re.I):
            result['group_by'] = 'material_category'
            result['confidence'] += 0.2

        # Detect "by document" or "per paper"
        if re.search(r'\b(per paper|by document|by paper|across papers|by source)\b', query, re.I):
            result['group_by'] = 'document'
            result['confidence'] += 0.15

        result['confidence'] = min(result['confidence'], 1.0)
        return result


# =============================================
# DATA EXTRACTOR FROM KNOWLEDGE GRAPH
# =============================================

class KnowledgeGraphDataExtractor:
    """Extracts structured data from CrossDocumentKnowledgeGraph for visualization."""

    def __init__(self, graph):
        self.graph = graph

    def get_material_distribution(self, filter_multicomponent_only: bool = False) -> pd.DataFrame:
        """Get material distribution across documents."""
        material_counts = defaultdict(lambda: defaultdict(int))

        for ent_norm, entities in self.graph.entities.items():
            # Check if it's a material entity
            is_material = any(e.label == "MATERIAL" for e in entities)
            if not is_material:
                continue

            # Categorize material
            category = self._categorize_material(ent_norm)

            if filter_multicomponent_only and category != 'multicomponent_alloy':
                continue

            for e in entities:
                material_counts[category][e.doc_source] += 1

        # Convert to DataFrame
        rows = []
        for category, docs in material_counts.items():
            total = sum(docs.values())
            unique_docs = len(docs)
            rows.append({
                'category': category,
                'total_mentions': total,
                'document_count': unique_docs,
                'avg_per_doc': total / unique_docs if unique_docs > 0 else 0
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('total_mentions', ascending=False)
        return df

    def get_laser_parameter_distribution(self, param_type: str = None) -> pd.DataFrame:
        """Get distribution of laser parameters."""
        param_data = []

        for ent_norm, entities in self.graph.entities.items():
            # Check if entity matches parameter type
            if param_type and ent_norm != param_type:
                continue

            # Check if it's a quantity with value
            for e in entities:
                if e.value is not None and e.label in QUANTITY_PATTERNS:
                    param_data.append({
                        'parameter': e.label,
                        'value': e.value,
                        'unit': e.unit,
                        'document': e.doc_source,
                        'context': e.context[:100]
                    })

        df = pd.DataFrame(param_data)
        if not df.empty:
            df = df.sort_values('value')
        return df

    def get_method_distribution(self) -> pd.DataFrame:
        """Get distribution of processing methods."""
        method_counts = defaultdict(lambda: defaultdict(int))

        for ent_norm, entities in self.graph.entities.items():
            # Check if it's a method entity
            is_method = any(e.label == "METHOD" for e in entities)
            if not is_method:
                continue

            category = self._categorize_method(ent_norm)

            for e in entities:
                method_counts[category][e.doc_source] += 1

        rows = []
        for category, docs in method_counts.items():
            total = sum(docs.values())
            unique_docs = len(docs)
            rows.append({
                'method': category,
                'total_mentions': total,
                'document_count': unique_docs
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('total_mentions', ascending=False)
        return df

    def get_property_values(self, property_type: str = None) -> pd.DataFrame:
        """Get property values across documents."""
        prop_data = []

        for ent_norm, entities in self.graph.entities.items():
            if property_type and ent_norm != property_type:
                continue

            for e in entities:
                if e.value is not None:
                    prop_data.append({
                        'property': e.label,
                        'value': e.value,
                        'unit': e.unit,
                        'document': e.doc_source,
                        'entity': ent_norm
                    })

        df = pd.DataFrame(prop_data)
        return df

    def get_document_comparison(self, entity_type: str = None) -> pd.DataFrame:
        """Get entity counts per document for comparison."""
        doc_data = defaultdict(lambda: defaultdict(int))

        for ent_norm, entities in self.graph.entities.items():
            if entity_type:
                category = self._categorize_entity(ent_norm)
                if category != entity_type:
                    continue

            for e in entities:
                doc_data[e.doc_source][ent_norm] += 1

        rows = []
        for doc, entities in doc_data.items():
            for ent, count in entities.items():
                rows.append({
                    'document': Path(doc).stem,
                    'entity': ent,
                    'count': count
                })

        df = pd.DataFrame(rows)
        return df

    def get_entity_cooccurrence(self, entity_types: List[str] = None) -> pd.DataFrame:
        """Get co-occurrence matrix of entities."""
        # Get documents and their entities
        doc_entities = defaultdict(set)

        for ent_norm, entities in self.graph.entities.items():
            category = self._categorize_entity(ent_norm)
            if entity_types and category not in entity_types:
                continue

            for e in entities:
                doc_entities[e.doc_source].add(ent_norm)

        # Build co-occurrence matrix
        all_entities = set()
        for ents in doc_entities.values():
            all_entities.update(ents)

        all_entities = sorted(list(all_entities))
        matrix = np.zeros((len(all_entities), len(all_entities)))

        for doc, ents in doc_entities.items():
            ents = list(ents)
            for i, e1 in enumerate(ents):
                for e2 in ents[i:]:
                    idx1 = all_entities.index(e1)
                    idx2 = all_entities.index(e2)
                    matrix[idx1][idx2] += 1
                    if idx1 != idx2:
                        matrix[idx2][idx1] += 1

        df = pd.DataFrame(matrix, index=all_entities, columns=all_entities)
        return df

    def _categorize_material(self, ent_norm: str) -> str:
        """Categorize material entity."""
        ent_lower = ent_norm.lower()

        multicomponent_keywords = ['hea', 'mpea', 'multicomponent', 'multi-component', 'high entropy', 
                                   'cocrfeni', 'alcocrfeni', 'crmnfeconi', 'refractory', 'alcrfeni']
        if any(kw in ent_lower for kw in multicomponent_keywords):
            return 'multicomponent_alloy'

        solder_keywords = ['solder', 'snagcu', 'sac', 'sn-ag-cu', 'sn-3.5ag']
        if any(kw in ent_lower for kw in solder_keywords):
            return 'solder'

        superalloy_keywords = ['inconel', 'in718', 'nimonic', 'haynes', 'waspaloy', 'superalloy']
        if any(kw in ent_lower for kw in superalloy_keywords):
            return 'superalloy'

        steel_keywords = ['steel', 'ss304', 'ss316', 'stainless']
        if any(kw in ent_lower for kw in steel_keywords):
            return 'steel'

        ti_keywords = ['titanium', 'ti-6al-4v', 'ti6al4v', 'cp-ti']
        if any(kw in ent_lower for kw in ti_keywords):
            return 'titanium'

        al_keywords = ['aluminum', 'aluminium', 'al-6061', 'al6061']
        if any(kw in ent_lower for kw in al_keywords):
            return 'aluminum'

        cu_keywords = ['copper', 'cu', 'brass', 'bronze']
        if any(kw in ent_lower for kw in cu_keywords):
            return 'copper'

        ceramic_keywords = ['ceramic', 'alumina', 'al2o3', 'zirconia', 'zro2', 'sic']
        if any(kw in ent_lower for kw in ceramic_keywords):
            return 'ceramic'

        polymer_keywords = ['polymer', 'pmma', 'polyimide', 'pei', 'pc', 'polycarbonate', 'ptfe', 'peek']
        if any(kw in ent_lower for kw in polymer_keywords):
            return 'polymer'

        si_keywords = ['silicon', 'si', 'c-si']
        if any(kw in ent_lower for kw in si_keywords):
            return 'silicon'

        return 'other_material'

    def _categorize_method(self, ent_norm: str) -> str:
        """Categorize method entity."""
        ent_lower = ent_norm.lower()

        slm_keywords = ['slm', 'selective laser melting', 'lpbf', 'laser powder bed']
        if any(kw in ent_lower for kw in slm_keywords):
            return 'SLM/LPBF'

        ded_keywords = ['ded', 'directed energy deposition', 'laser cladding', 'laser metal deposition']
        if any(kw in ent_lower for kw in ded_keywords):
            return 'DED/Cladding'

        soldering_keywords = ['soldering', 'reflow', 'wave soldering']
        if any(kw in ent_lower for kw in soldering_keywords):
            return 'Laser Soldering'

        ablation_keywords = ['ablation', 'micromachining', 'material removal']
        if any(kw in ent_lower for kw in ablation_keywords):
            return 'Laser Ablation'

        welding_keywords = ['welding', 'weld seam', 'fusion welding']
        if any(kw in ent_lower for kw in welding_keywords):
            return 'Laser Welding'

        structuring_keywords = ['structuring', 'texturing', 'lipss', 'ripples', 'nanostructuring']
        if any(kw in ent_lower for kw in structuring_keywords):
            return 'Surface Structuring'

        annealing_keywords = ['annealing', 'heat treatment', 'thermal treatment']
        if any(kw in ent_lower for kw in annealing_keywords):
            return 'Laser Annealing'

        return 'other_method'

    def _categorize_entity(self, ent_norm: str) -> str:
        """General entity categorization."""
        material_cat = self._categorize_material(ent_norm)
        if material_cat != 'other_material':
            return 'material'

        method_cat = self._categorize_method(ent_norm)
        if method_cat != 'other_method':
            return 'method'

        # Check if it's a quantity/parameter
        if ent_norm in QUANTITY_PATTERNS:
            return 'parameter'

        return 'other'


# =============================================
# CHART GENERATOR
# =============================================

class ChartGenerator:
    """Generates Plotly charts from extracted data."""

    COLOR_PALETTE = px.colors.qualitative.Bold + px.colors.qualitative.Vivid

    @classmethod
    def generate(cls, chart_type: str, data: pd.DataFrame, **kwargs) -> go.Figure:
        """Generate chart based on type and data."""
        generators = {
            'bar': cls._bar_chart,
            'pie': cls._pie_chart,
            'line': cls._line_chart,
            'scatter': cls._scatter_chart,
            'heatmap': cls._heatmap_chart,
            'radar': cls._radar_chart,
            'box': cls._box_chart,
            'bubble': cls._bubble_chart,
        }

        generator = generators.get(chart_type, cls._bar_chart)
        return generator(data, **kwargs)

    @classmethod
    def _bar_chart(cls, data: pd.DataFrame, x: str = None, y: str = None, 
                   color: str = None, title: str = "", orientation: str = 'v',
                   show_values: bool = True, **kwargs) -> go.Figure:
        """Create bar chart."""
        if data.empty:
            return cls._empty_chart("No data available")

        # Auto-detect columns if not specified
        if x is None:
            x = data.columns[0]
        if y is None:
            y = data.columns[1] if len(data.columns) > 1 else data.columns[0]

        if orientation == 'h':
            fig = px.bar(data, y=x, x=y, color=color, title=title,
                        color_discrete_sequence=cls.COLOR_PALETTE,
                        orientation='h', **kwargs)
        else:
            fig = px.bar(data, x=x, y=y, color=color, title=title,
                        color_discrete_sequence=cls.COLOR_PALETTE,
                        **kwargs)

        if show_values:
            fig.update_traces(texttemplate='%{y:.2f}' if orientation == 'v' else '%{x:.2f}',
                            textposition='outside')

        fig.update_layout(
            xaxis_title=x.replace('_', ' ').title(),
            yaxis_title=y.replace('_', ' ').title(),
            template='plotly_white',
            showlegend=True if color else False
        )
        return fig

    @classmethod
    def _pie_chart(cls, data: pd.DataFrame, names: str = None, values: str = None,
                   title: str = "", hole: float = 0.0, **kwargs) -> go.Figure:
        """Create pie/donut chart."""
        if data.empty:
            return cls._empty_chart("No data available")

        if names is None:
            names = data.columns[0]
        if values is None:
            values = data.columns[1] if len(data.columns) > 1 else data.columns[0]

        fig = px.pie(data, names=names, values=values, title=title,
                    color_discrete_sequence=cls.COLOR_PALETTE,
                    hole=hole, **kwargs)

        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(template='plotly_white')
        return fig

    @classmethod
    def _line_chart(cls, data: pd.DataFrame, x: str = None, y: str = None,
                    color: str = None, title: str = "", **kwargs) -> go.Figure:
        """Create line chart."""
        if data.empty:
            return cls._empty_chart("No data available")

        if x is None:
            x = data.columns[0]
        if y is None:
            y = data.columns[1] if len(data.columns) > 1 else data.columns[0]

        fig = px.line(data, x=x, y=y, color=color, title=title,
                     color_discrete_sequence=cls.COLOR_PALETTE,
                     markers=True, **kwargs)

        fig.update_layout(template='plotly_white')
        return fig

    @classmethod
    def _scatter_chart(cls, data: pd.DataFrame, x: str = None, y: str = None,
                       color: str = None, size: str = None, title: str = "",
                       **kwargs) -> go.Figure:
        """Create scatter plot."""
        if data.empty:
            return cls._empty_chart("No data available")

        if x is None:
            x = data.columns[0]
        if y is None:
            y = data.columns[1] if len(data.columns) > 1 else data.columns[0]

        fig = px.scatter(data, x=x, y=y, color=color, size=size, title=title,
                        color_discrete_sequence=cls.COLOR_PALETTE,
                        **kwargs)

        fig.update_layout(template='plotly_white')
        return fig

    @classmethod
    def _heatmap_chart(cls, data: pd.DataFrame, title: str = "", **kwargs) -> go.Figure:
        """Create heatmap."""
        if data.empty:
            return cls._empty_chart("No data available")

        fig = px.imshow(data, text_auto=True, aspect="auto", title=title,
                       color_continuous_scale='RdYlBu_r', **kwargs)
        fig.update_layout(template='plotly_white')
        return fig

    @classmethod
    def _radar_chart(cls, data: pd.DataFrame, categories: str = None, 
                     values: str = None, color: str = None, title: str = "",
                     **kwargs) -> go.Figure:
        """Create radar/spider chart."""
        if data.empty:
            return cls._empty_chart("No data available")

        if categories is None:
            categories = data.columns[0]
        if values is None:
            values = data.columns[1] if len(data.columns) > 1 else data.columns[0]

        fig = go.Figure()

        if color and color in data.columns:
            for group in data[color].unique():
                group_data = data[data[color] == group]
                fig.add_trace(go.Scatterpolar(
                    r=group_data[values].tolist(),
                    theta=group_data[categories].tolist(),
                    fill='toself',
                    name=str(group)
                ))
        else:
            fig.add_trace(go.Scatterpolar(
                r=data[values].tolist(),
                theta=data[categories].tolist(),
                fill='toself',
                name='Value'
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            title=title,
            template='plotly_white'
        )
        return fig

    @classmethod
    def _box_chart(cls, data: pd.DataFrame, x: str = None, y: str = None,
                   color: str = None, title: str = "", **kwargs) -> go.Figure:
        """Create box plot."""
        if data.empty:
            return cls._empty_chart("No data available")

        if x is None:
            x = data.columns[0]
        if y is None:
            y = data.columns[1] if len(data.columns) > 1 else data.columns[0]

        fig = px.box(data, x=x, y=y, color=color, title=title,
                    color_discrete_sequence=cls.COLOR_PALETTE,
                    **kwargs)

        fig.update_layout(template='plotly_white')
        return fig

    @classmethod
    def _bubble_chart(cls, data: pd.DataFrame, x: str = None, y: str = None,
                      size: str = None, color: str = None, title: str = "",
                      **kwargs) -> go.Figure:
        """Create bubble chart."""
        if data.empty:
            return cls._empty_chart("No data available")

        if x is None:
            x = data.columns[0]
        if y is None:
            y = data.columns[1] if len(data.columns) > 1 else data.columns[0]

        fig = px.scatter(data, x=x, y=y, size=size, color=color, title=title,
                        color_discrete_sequence=cls.COLOR_PALETTE,
                        **kwargs)

        fig.update_layout(template='plotly_white')
        return fig

    @classmethod
    def _empty_chart(cls, message: str) -> go.Figure:
        """Create empty chart with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(template='plotly_white')
        return fig


# =============================================
# CHAT-DRIVEN VISUALIZATION ENGINE
# =============================================

class ChatDrivenVisualization:
    """Main engine that processes chat queries and generates visualizations."""

    def __init__(self, graph):
        self.graph = graph
        self.parser = VizQueryParser()
        self.extractor = KnowledgeGraphDataExtractor(graph)
        self.generator = ChartGenerator()

    def process_query(self, query: str) -> Tuple[go.Figure, str, Dict[str, Any]]:
        """Process a natural language query and return visualization + explanation."""

        # Parse the query
        parsed = self.parser.parse(query)

        # Extract data based on parsed intent
        data, explanation = self._extract_data(parsed)

        # Generate chart
        fig = self.generator.generate(
            chart_type=parsed['chart_type'],
            data=data,
            title=self._generate_title(parsed, query)
        )

        # Build explanation
        explanation = self._build_explanation(parsed, data, explanation)

        return fig, explanation, parsed

    def _extract_data(self, parsed: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
        """Extract appropriate data based on parsed query."""

        # Case 1: Material distribution (e.g., "plot multicomponent alloys")
        if parsed['entity_filter'] == 'multicomponent_alloy' or 'multicomponent' in str(parsed.get('group_by', '')):
            if 'only' in str(parsed).lower() or 'among' in str(parsed).lower():
                # Compare multicomponent vs others
                df_all = self.extractor.get_material_distribution(filter_multicomponent_only=False)
                if not df_all.empty:
                    df_all['is_multicomponent'] = df_all['category'] == 'multicomponent_alloy'
                    return df_all, "Material distribution across all documents"
            else:
                df = self.extractor.get_material_distribution(filter_multicomponent_only=True)
                return df, "Multicomponent alloy distribution"

        # Case 2: General material distribution
        if parsed['entity_filter'] in ['solder', 'superalloy', 'steel', 'titanium', 'aluminum']:
            df = self.extractor.get_material_distribution()
            df_filtered = df[df['category'] == parsed['entity_filter']] if not df.empty else df
            return df_filtered, f"{parsed['entity_filter'].replace('_', ' ').title()} distribution"

        # Case 3: Laser parameter distribution
        if parsed['parameter_focus']:
            df = self.extractor.get_laser_parameter_distribution(parsed['parameter_focus'])
            return df, f"Laser {parsed['parameter_focus'].replace('_', ' ')} distribution"

        # Case 4: Method distribution
        if parsed['method_focus'] or 'method' in str(parsed).lower():
            df = self.extractor.get_method_distribution()
            return df, "Laser processing method distribution"

        # Case 5: Property values
        if parsed['property_focus']:
            df = self.extractor.get_property_values(parsed['property_focus'])
            return df, f"{parsed['property_focus'].replace('_', ' ').title()} values"

        # Default: Show material distribution
        df = self.extractor.get_material_distribution()
        return df, "Material distribution across documents"

    def _generate_title(self, parsed: Dict[str, Any], query: str) -> str:
        """Generate chart title from parsed query."""
        parts = []

        if parsed['aggregation'] == 'count':
            parts.append("Count of")
        elif parsed['aggregation'] == 'mean':
            parts.append("Average")

        if parsed['entity_filter']:
            parts.append(parsed['entity_filter'].replace('_', ' ').title())
        elif parsed['parameter_focus']:
            parts.append(parsed['parameter_focus'].replace('_', ' ').title())
        elif parsed['method_focus']:
            parts.append(parsed['method_focus'].replace('_', ' ').title())
        elif parsed['property_focus']:
            parts.append(parsed['property_focus'].replace('_', ' ').title())
        else:
            parts.append("Entities")

        if parsed['group_by'] == 'document':
            parts.append("by Document")

        return " ".join(parts)

    def _build_explanation(self, parsed: Dict[str, Any], data: pd.DataFrame, base_explanation: str) -> str:
        """Build human-readable explanation of the visualization."""
        lines = [f"**Visualization Type:** {parsed['chart_type'].title()} Chart"]
        lines.append(f"**Data Source:** {base_explanation}")
        lines.append(f"**Query Confidence:** {parsed['confidence']:.0%}")

        if not data.empty:
            lines.append(f"**Records Found:** {len(data)}")
            if 'total_mentions' in data.columns:
                total = data['total_mentions'].sum()
                lines.append(f"**Total Mentions:** {total}")
            if 'document_count' in data.columns:
                docs = data['document_count'].sum()
                lines.append(f"**Documents Covered:** {docs}")

        return "
".join(lines)

    def get_suggested_queries(self) -> List[str]:
        """Return suggested visualization queries."""
        return [
            "Plot a bar chart of multicomponent alloys among all materials",
            "Show pie chart of laser processing methods used across papers",
            "Create a radar chart comparing laser parameters",
            "Plot distribution of laser power values",
            "Show heatmap of material-method co-occurrence",
            "Compare solder alloys vs other materials",
            "Plot scan speed vs power scatter plot",
            "Show box plot of grain sizes across documents",
            "Create bubble chart of properties by material",
            "Plot line chart of thermal conductivity trends",
        ]


# =============================================
# STREAMLIT UI COMPONENT
# =============================================

def render_chat_visualization_panel():
    """Render the chat-driven visualization panel in Streamlit."""

    st.markdown("---")
    st.markdown("## 📊 Chat-Driven Scientific Visualizations")

    if not st.session_state.get('knowledge_graph') or not st.session_state.knowledge_graph.entities:
        st.info("📁 Upload and process documents first to enable chat-driven visualizations")
        return

    # Initialize visualization engine
    if 'viz_engine' not in st.session_state:
        st.session_state.viz_engine = ChatDrivenVisualization(st.session_state.knowledge_graph)

    viz_engine = st.session_state.viz_engine

    # Query input
    col1, col2 = st.columns([3, 1])
    with col1:
        viz_query = st.text_input(
            "💬 Ask for a visualization (e.g., 'Plot multicomponent alloys among all materials')",
            key="viz_query_input",
            placeholder="Type your visualization request..."
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("📈 Generate Plot", type="primary", use_container_width=True)

    # Suggested queries
    with st.expander("💡 Suggested Visualization Queries", expanded=False):
        suggestions = viz_engine.get_suggested_queries()
        cols = st.columns(2)
        for i, query in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(f"▶ {query}", key=f"suggest_{i}", use_container_width=True):
                    st.session_state.viz_query_input = query
                    st.rerun()

    # Generate visualization
    if generate_btn and viz_query:
        with st.spinner("🔍 Parsing query and extracting data..."):
            try:
                fig, explanation, parsed = viz_engine.process_query(viz_query)

                # Store in session state
                st.session_state.last_viz_fig = fig
                st.session_state.last_viz_explanation = explanation
                st.session_state.last_viz_parsed = parsed

            except Exception as e:
                st.error(f"Error generating visualization: {e}")
                return

    # Display visualization
    if st.session_state.get('last_viz_fig'):
        col_viz, col_info = st.columns([2, 1])

        with col_viz:
            st.plotly_chart(st.session_state.last_viz_fig, use_container_width=True)

            # Download button
            buf = BytesIO()
            st.session_state.last_viz_fig.write_image(buf, format="png", scale=2)
            buf.seek(0)
            st.download_button(
                label="📥 Download PNG",
                data=buf,
                file_name="scientific_visualization.png",
                mime="image/png",
                use_container_width=True
            )

        with col_info:
            st.markdown("### 📋 Visualization Info")
            st.markdown(st.session_state.last_viz_explanation)

            # Show parsed parameters
            with st.expander("🔍 Parsed Query Parameters"):
                parsed = st.session_state.get('last_viz_parsed', {})
                for key, value in parsed.items():
                    if value is not None:
                        st.markdown(f"**{key.replace('_', ' ').title()}:** `{value}`")

        # Data table
        with st.expander("📊 View Underlying Data"):
            # Re-extract data for display
            _, data_df, _ = viz_engine._extract_data(st.session_state.last_viz_parsed)
            if not data_df.empty:
                st.dataframe(data_df, use_container_width=True)
            else:
                st.info("No tabular data available for this visualization")


# =============================================
# INTEGRATION: Add to main() function
# =============================================

# Add this line at the end of main(), before render_footer():
# render_chat_visualization_panel()


def main():
    st.set_page_config(
        page_title="🔬 Laser Microstructure RAG + Cross-Doc Reasoning",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #1e40af, #7c3aed, #059669);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
    }
    .info-card {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin: 0.5rem 0;
    }
    .reasoning-badge {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        padding: 0.2rem 0.6rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
        margin: 0.1rem 0.2rem 0.1rem 0;
    }
    .consensus-badge {
        display: inline-block;
        background: #d1fae5;
        color: #065f46;
        padding: 0.2rem 0.6rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
        margin: 0.1rem 0.2rem 0.1rem 0;
    }
    .contradiction-badge {
        display: inline-block;
        background: #fee2e2;
        color: #991b1b;
        padding: 0.2rem 0.6rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
        margin: 0.1rem 0.2rem 0.1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🔬 Laser Microstructure RAG + Cross-Doc Reasoning</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:#64748b;margin-bottom:1.5rem">
    Upload research papers on multicomponent alloys and laser processing, and get <strong>scientifically rigorous answers</strong> with 
    <span class="consensus-badge">cross-document consensus</span>, 
    <span class="contradiction-badge">contradiction detection</span>, and 
    <span class="reasoning-badge">multi-hop reasoning</span>.
    </div>
    """, unsafe_allow_html=True)

    initialize_session_state()
    render_sidebar()

    if st.session_state.llm_model_choice and not is_ollama_model(st.session_state.llm_model_choice):
        mem_info = estimate_model_memory(st.session_state.llm_model_choice, st.session_state.get('use_4bit_quantization', True))
        available_vram = get_available_gpu_memory()
        if available_vram and not mem_info['cpu_ok']:
            required = float(mem_info['vram_4bit'].replace('GB','').replace('~','').strip()) if 'GB' in mem_info['vram_4bit'] else 100
            if available_vram < required:
                st.markdown(f"""
                <div style="background:#fef3c7;border-left:4px solid #f59e0b;padding:0.75rem;border-radius:0 0.5rem 0.5rem 0;margin:0.5rem 0">
                ⚠️ <strong>Memory Warning:</strong> {st.session_state.llm_model_choice} requires ~{mem_info['vram_4bit']} VRAM.
                You have ~{available_vram:.1f}GB available.
                </div>
                """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded_files = render_document_uploader()
        if uploaded_files and st.button("🔄 Process Documents", type="primary", use_container_width=True):
            process_documents(uploaded_files)

        if st.session_state.processing_complete:
            st.success("✅ Knowledge base ready")
            if st.session_state.knowledge_graph:
                summary = st.session_state.knowledge_graph.get_knowledge_summary()
                st.caption(f"📦 {len(st.session_state.all_chunks)} chunks | {summary['unique_entities']} entities | {summary['total_claims']} claims")
                if summary['top_entities']:
                    st.markdown("**Top entities:**")
                    for ent, count in summary['top_entities'][:5]:
                        st.markdown(f'<span class="reasoning-badge">{ent} ({count})</span>', unsafe_allow_html=True)
        elif uploaded_files:
            st.warning("⏳ Click 'Process Documents' to begin")
        else:
            st.info("📁 Upload PDF/TXT files to start")

        if st.session_state.processed_files:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.clear()
                st.rerun()

    with col2:
        if st.session_state.processing_complete and st.session_state.vectorstore:
            render_chat_interface()
        else:
            st.markdown("""
            <div class="info-card">
            <h3>👋 Welcome to Cross-Document Scientific Reasoning!</h3>
            <p>This assistant goes beyond simple retrieval:</p>
            <ul>
            <li><strong>Semantic Chunking:</strong> Preserves Abstract/Methods/Results/Discussion structure</li>
            <li><strong>Entity Extraction:</strong> Identifies materials, parameters, methods automatically</li>
            <li><strong>Cross-Document Alignment:</strong> Links the same entity across different papers</li>
            <li><strong>Consensus Detection:</strong> Statistically aggregates values reported in multiple papers</li>
            <li><strong>Contradiction Flagging:</strong> Highlights when papers disagree significantly</li>
            <li><strong>Multi-Hop Retrieval:</strong> Follows entity links to find related evidence</li>
            <li><strong>Uncertainty Calibration:</strong> Explicit confidence levels in every answer</li>
            </ul>
            <p><strong>Getting started:</strong></p>
            <ol>
            <li>Upload 2+ PDF/TXT papers on multicomponent alloys or laser processing</li>
            <li>Enable "Cross-document reasoning" in sidebar</li>
            <li>Ask comparative or synthesizing questions</li>
            <li>Expand "🧠 Reasoning Chain" to see the logical steps</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Try asking:**")
            demo_qs = [
                "What is the effect of laser power on interfacial IMC thickness in Sn‑Ag‑Cu/Cu joints?",
                "Do these papers agree on the optimal hatch distance for defect‑free LPBF of Al‑Cr‑Fe‑Ni alloys?",
                "Summarize the phase‑field models used for simulating selective laser melting of multicomponent alloys.",
                "How does the composition of high entropy alloys affect their thermal conductivity during laser processing?",
            ]
            for q in demo_qs:
                if st.button(f"💬 {q}", use_container_width=True, key=f"demo_{q[:20]}"):
                    st.session_state.demo_question = q
                    st.rerun()

    # Global visualisation section
    st.markdown("---")
    st.markdown("## 🔬 Scientific Visualizations")
    vis_col1, vis_col2 = st.columns(2)
    with vis_col1:
        if st.session_state.knowledge_graph:
            entity_counts = Counter([e.normalized for ents in st.session_state.knowledge_graph.entities.values() for e in ents]).most_common(15)
            if entity_counts:
                fig = px.bar(x=[x[0] for x in entity_counts], y=[x[1] for x in entity_counts],
                             labels={'x':'Entity', 'y':'Occurrences'}, title="Top Entities in Knowledge Base")
                st.plotly_chart(fig, use_container_width=True)
    with vis_col2:
        if st.session_state.knowledge_graph:
            docs = list(st.session_state.knowledge_graph.documents.keys())
            top_entities = [x[0] for x in entity_counts[:8]] if entity_counts else []
            data = []
            for doc in docs:
                row = []
                for ent in top_entities:
                    count = sum(1 for e in st.session_state.knowledge_graph.entities.get(ent, []) if e.doc_source == doc)
                    row.append(count)
                data.append(row)
            if data and top_entities:
                df_heat = pd.DataFrame(data, index=[Path(d).stem for d in docs], columns=top_entities)
                fig2 = px.imshow(df_heat, text_auto=True, aspect="auto", title="Entity × Document co‑occurrence")
                st.plotly_chart(fig2, use_container_width=True)

    render_chat_visualization_panel()

    render_footer()

    if hasattr(st.session_state, 'demo_question') and st.session_state.demo_question:
        st.session_state.messages.append({"role": "user", "content": st.session_state.demo_question})
        del st.session_state.demo_question
        st.rerun()


if __name__ == "__main__":
    main()
