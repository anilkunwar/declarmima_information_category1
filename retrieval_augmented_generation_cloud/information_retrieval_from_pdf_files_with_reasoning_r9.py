#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LASER MICROSTRUCTURE RAG CHATBOT - CROSS-DOCUMENT SCIENTIFIC REASONING VERSION
========================================================================================
DECLARMIMA-ENHANCED: Physics-informed digital twin for laser-multicomponent alloy interaction
✅ Zero API keys required - all models run locally
✅ Cross-document reasoning: consensus, contradiction, and gap detection
✅ Scientific entity extraction and alignment across papers
✅ Multi-hop retrieval via knowledge graph traversal
✅ Uncertainty-calibrated responses with structured provenance
✅ Enhanced citations with bibliographic metadata
✅ DOMAIN: Additive Manufacturing, SLM/LPBF, HEAs, Sn/Al-based multicomponent alloys
"""
import streamlit as st
import os
import tempfile
import time
import re
import json
import torch
import numpy as np
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

# Optional: Bibliographic metadata
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

# DECLARMIMA-ENHANCED: Expanded laser keywords for AM/SLM/HEA domain
LASER_KEYWORDS = {
    "ablation": ["ablation", "material removal", "threshold fluence", "laser ablation", "ablation threshold"],
    "plasma": ["plasma formation", "ionization", "electron density", "plume", "plasma shielding"],
    "thermal": ["heat affected zone", "melting", "thermal diffusion", "resolidification", "heat-affected zone", "cooling rate"],
    "ultrafast": ["femtosecond", "picosecond", "pulse duration", "ultrafast laser", "fs laser"],
    "morphology": ["ripples", "LIPSS", "surface structuring", "periodic structures", "nanostructures", "microstructures"],
    "parameters": ["fluence", "wavelength", "pulse energy", "repetition rate", "spot size", "scan speed", "overlap"],
    "materials": ["silicon", "steel", "titanium", "polymer", "glass", "ceramic", "aluminum", "copper", "tungsten"],
    "characterization": ["SEM", "AFM", "profilometry", "spectroscopy", "microscopy", "Raman", "XRD", "EDX"],
    # DECLARMIMA additions
    "additive_manufacturing": ["additive manufacturing", "3d printing", "selective laser melting", "slm", "laser powder bed fusion", "lpbf", "wire-feed laser additive manufacturing", "wflam", "direct energy deposition"],
    "melt_pool": ["melt pool", "meltpool", "molten pool", "melt track", "melt-track", "keyhole", "vapor channel", "melt pool geometry", "melt pool dynamics"],
    "defects": ["porosity", "pore", "spatter", "spatter ejection", "defect", "crack", "lack of fusion", "depression", "denuded zone", "balling"],
    "high_entropy_alloys": ["high entropy alloy", "hea", "multi-principal component alloy", "mpea", "multi-principal element alloy", "multi-component alloy"],
    "intermetallic": ["intermetallic", "imc", "intermetallic compound", "cu6sn5", "interfacial intermetallic"],
    "marangoni": ["marangoni", "marangoni convection", "thermocapillary", "surface tension driven flow"],
    "powder": ["powder", "powdered alloy", "particle size", "powder size", "d50", "d10", "d90", "packing density", "flowability", "powder layer", "powder bed"],
    "solidification": ["solidification", "grain growth", "grain boundary", "microstructure evolution", "phase evolution", "dendrite", "epitaxial growth"],
    "residual_stress": ["residual stress", "thermal stress", "stress distribution", "distortion", "warpage"],
    "digital_twin": ["digital twin", "physics-informed", "physics informed", "machine learning", "data-driven", "computational model"],
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
# REASONING: SCIENTIFIC ENTITY & CLAIM PATTERNS
# =============================================

# DECLARMIMA-ENHANCED: Quantitative patterns for AM/SLM scientific findings
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
    # DECLARMIMA AM-specific parameters
    "scan_speed": re.compile(r'(\d+(?:\.\d+)?)\s*(?:mm/s|mm/min|m/s)\s*(?:scan\s*speed|scanning\s*speed|speed)', re.I),
    "hatch_distance": re.compile(r'(\d+(?:\.\d+)?)\s*(?:µm|um|mm)\s*(?:hatch\s*distance|hatch\s*spacing)', re.I),
    "layer_thickness": re.compile(r'(\d+(?:\.\d+)?)\s*(?:µm|um|mm)\s*(?:layer\s*thickness|layer\s*height)', re.I),
    "bed_temperature": re.compile(r'(\d+(?:\.\d+)?)\s*(?:°?C|K)\s*(?:bed\s*temperature|preheat|substrate\s*temperature|build\s*plate)', re.I),
    "laser_power": re.compile(r'(\d+(?:\.\d+)?)\s*(?:W|kW|mW)\s*(?:laser\s*power|power)', re.I),
    "absorptivity": re.compile(r'(\d+(?:\.\d+)?)\s*(?:absorptivity|absorptance|absorption)', re.I),
    "powder_size": re.compile(r'(\d+(?:\.\d+)?)\s*(?:µm|um|mm)\s*(?:powder\s*size|particle\s*size|d50|d10|d90)', re.I),
}

# DECLARMIMA-ENHANCED: Material normalizations including AM alloys
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
    # DECLARMIMA additions
    "ti6al4v": ["ti6al4v", "ti-6al-4v", "ti 6al 4v", "titanium alloy"],
    "inconel_718": ["inconel 718", "inconel-718", "nickel superalloy"],
    "invar_36": ["invar 36", "invar-36"],
    "sac": ["sac", "sn-ag-cu", "sn-ag-cu-x", "sn-3.5ag-0.5cu", "sac305", "sn-ag-cu-bi", "sn-ag-cu-zn", "sn-ag-cu-ni"],
    "alcrfeni": ["al-cr-fe-ni", "alcrfeni", "al-ni-cr", "al-ni-fe-cr", "al-ni", "al-ni-cu"],
    "hea": ["high entropy alloy", "hea", "co-cr-fe-mn-ni", "al-co-cr-fe-ni"],
}

# DECLARMIMA-ENHANCED: Method normalizations including AM characterization
METHOD_ALIASES = {
    "sem": ["sem", "scanning electron microscopy", "scanning electron microscope"],
    "afm": ["afm", "atomic force microscopy", "atomic force microscope"],
    "profilometry": ["profilometry", "optical profilometry", "white light interferometry", "wli"],
    "raman": ["raman", "raman spectroscopy", "micro-raman"],
    "xrd": ["xrd", "x-ray diffraction"],
    "edx": ["edx", "eds", "energy dispersive x-ray", "energy-dispersive"],
    # DECLARMIMA additions
    "x_ray_imaging": ["x-ray imaging", "x-ray radiography", "x-ray radiographic", "synchrotron x-ray", "computed tomography", "ct scan", "tomography"],
    "high_speed_camera": ["high speed camera", "high-speed camera", "photron", "fastcam", "in-situ imaging"],
    "phase_field": ["phase field", "phase-field", "pfm", "phase field model", "phase-field model", "phase field simulation"],
    "molecular_dynamics": ["molecular dynamics", "md simulation", "lammps", "ase", "atomic simulation"],
    "finite_element": ["finite element", "fem", "finite element method", "moose framework", "multiphysics simulation", "finite element analysis"],
    "calphad": ["calphad", "thermocalc", "thermodynamic database", "tcni8", "tchea2", "mobni5", "mobhea2"],
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
            context = text_sample[max(0, year_pos-50):year_pos+50].lower()
            if any(kw in context for kw in ['published', 'received', 'accepted', 'copyright', '©']):
                meta.year = year
                meta.confidence = max(meta.confidence, 0.7)
                break

    for pattern in BibliographicMetadata.JOURNAL_PATTERNS:
        journal_match = pattern.search(text_sample)
        if journal_match:
            journal = journal_match.group(1).strip()
            if len(journal) > 10 and not any(bad in journal.lower() for bad in ['introduction', 'abstract', 'references']):
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
                    meta.authors = [f"{a.get('family', '')} {a.get('given', '')}".strip() for a in msg['author']]
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
                unit_match = re.search(r'(nm|µm|um|fs|ps|ns|J/cm²|J/cm2|kHz|MHz|W|mW|mJ|µJ|uJ|mm/s|mm/min|°?C|K)', match.group(0), re.I)
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
            # DECLARMIMA-specific claim patterns
            (r'(?:melt\s*pool\s*(?:depth|width|length))\s*(?:of|for)\s+([a-z\s]+?)\s+(?:is|was|≈|~)\s+(\d+\.?\d*\s*(?:µm|um|mm|nm))', 'has_melt_pool_dimension'),
            (r'(?:porosity|pore\s*(?:fraction|density))\s*(?:of|for|in)\s+([a-z\s]+?)\s+(?:is|was|≈|~)\s+(\d+\.?\d*\s*(?:%|pct|percent|vol\.?%))', 'has_porosity'),
            (r'(?:scan\s*speed|scanning\s*speed)\s*(?:of|for)\s+([a-z\s]+?)\s+(?:is|was|≈|~)\s+(\d+\.?\d*\s*(?:mm/s|mm/min))', 'has_scan_speed'),
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
            for j in range(i+1, len(docs)):
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
        """FIXED: Now includes total_chunks to prevent KeyError."""
        total_chunks = sum(d.get("chunk_count", 0) for d in self.documents.values())
        return {
            "total_chunks": total_chunks,
            "total_entities": sum(len(v) for v in self.entities.values()),
            "unique_entities": len(self.entities),
            "total_claims": len(self.claims),
            "document_count": len(self.documents),
            "top_entities": Counter([e.normalized for ents in self.entities.values() for e in ents]).most_common(10),
            "consensus_topics": [k for k, v in self.entities.items() if len(self.entity_index.get(k, set())) > 1]
        }


# =============================================
# REASONING: SEMANTIC CHUNKING WITH STRUCTURE AWARENESS
# =============================================

def detect_scientific_sections(text: str) -> List[Tuple[str, str]]:
    section_patterns = [
        (r'(?:^|\n)\s*Abstract\s*\n', 'ABSTRACT'),
        (r'(?:^|\n)\s*1\.\s*Introduction\s*\n', 'INTRODUCTION'),
        (r'(?:^|\n)\s*(?:2\.)?\s*Experimental\s*(?:Setup|Methods|Details)?\s*\n', 'METHODS'),
        (r'(?:^|\n)\s*(?:3\.)?\s*Results\s*(?:and\s*Discussion)?\s*\n', 'RESULTS'),
        (r'(?:^|\n)\s*(?:4\.)?\s*Discussion\s*\n', 'DISCUSSION'),
        (r'(?:^|\n)\s*Conclusion', 'CONCLUSION'),
        # DECLARMIMA-specific sections
        (r'(?:^|\n)\s*(?:5\.)?\s*Research\s*Methodology\s*\n', 'METHODOLOGY'),
        (r'(?:^|\n)\s*(?:6\.)?\s*References\s*\n', 'REFERENCES'),
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
        end = boundaries[i+1][0] if i+1 < len(boundaries) else len(text)
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
        elif section_name == 'REFERENCES':
            continue  # Skip references
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
# SESSION STATE & UTILITIES
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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


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
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
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
# REASONING: DOCUMENT PROCESSING WITH KNOWLEDGE GRAPH
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
        # DECLARMIMA-specific
        "scan_speed": r'(\d+(?:\.\d+)?)\s*(?:mm/s|mm/min)\s*(?:scan\s*speed|scanning\s*speed)',
        "laser_power_W": r'(\d+(?:\.\d+)?)\s*(?:W|kW)\s*(?:laser\s*power|power)',
        "hatch_distance_um": r'(\d+(?:\.\d+)?)\s*(?:µm|um|mm)\s*(?:hatch\s*distance|hatch\s*spacing)',
        "layer_thickness_um": r'(\d+(?:\.\d+)?)\s*(?:µm|um|mm)\s*(?:layer\s*thickness|layer\s*height)',
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
# REASONING: ENHANCED RAG WITH CROSS-DOCUMENT SYNTHESIS
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

    system_prompt = """You are an expert scientific research assistant specializing in laser-microstructure interactions and additive manufacturing.
Your task is to synthesize evidence from multiple research papers and provide a scientifically rigorous answer.

REASONING RULES:
1. SYNTHESIZE across documents — do not just summarize one paper at a time
2. Identify CONSENSUS where multiple papers agree, and CONTRADICTIONS where they disagree
3. Report UNCERTAINTY explicitly — use phrases like "reported values range from X to Y", "the consensus mean is Z ± σ"
4. Cite sources using the EXACT citation format provided (Author et al., Journal, Year)
5. If evidence is insufficient or contradictory, state this explicitly rather than fabricating consensus
6. Distinguish between direct experimental results and inferred/theoretical claims
7. For numerical values, include units and note if papers use different measurement conditions
8. For DECLARMIMA-related queries, emphasize physics-informed digital twin concepts, multi-scale modeling, and process-structure-property relationships

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
                {"role": "system", "content": "You are an expert in laser-microstructure interaction and additive manufacturing research. Synthesize evidence across multiple papers rigorously."},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        elif "Llama" in backend_name or "llama" in backend_name.lower():
            messages = [
                {"role": "system", "content": "You are an expert in laser-microstructure interaction and additive manufacturing research. Synthesize evidence across multiple papers rigorously."},
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
            answer = full_text[-LASER_DOMAIN_CONFIG["max_new_tokens"]*2:].strip()

        answer = re.sub(r'\s+', ' ', answer).strip()
        return answer if answer else "I was unable to generate a response. Please try rephrasing your question."

    except Exception as e:
        st.error(f"Generation error: {e}")
        return f"Error generating response: {str(e)[:200]}..."


def generate_local_response_ollama(model_tag: str, ollama_host: str, prompt: str) -> str:
    try:
        client = ollama.Client(host=ollama_host)
        messages = [
            {"role": "system", "content": "You are an expert in laser-microstructure interaction and additive manufacturing research. Synthesize evidence across multiple papers rigorously."},
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
        search_kwargs={"k": k*2, "score_threshold": score_threshold}
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
        return "Based on the uploaded documents, I could not find information relevant to your question. Try rephrasing or checking document content.", [], avg_relevance, {}

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
# STREAMLIT UI
# =============================================

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        backend_option = st.radio(
            "🔧 Inference Backend",
            options=["Hugging Face Transformers", "Ollama (if installed)"],
            index=0
        )
        st.session_state.inference_backend = backend_option

        if backend_option == "Ollama (if installed)":
            if not OLLAMA_AVAILABLE:
                st.error("❌ ollama library not installed")
                st.code("pip install ollama")
            available_ollama_models = [k for k in LOCAL_LLM_OPTIONS.keys() if is_ollama_model(k)]
            model_choice = st.selectbox(
                "🧠 Local LLM Backend (Ollama)",
                options=available_ollama_models if available_ollama_models else ["No Ollama models available"],
                index=0
            )
        else:
            hf_models = [k for k in LOCAL_LLM_OPTIONS.keys() if not is_ollama_model(k)]
            model_choice = st.selectbox(
                "🧠 Local LLM Backend (Hugging Face)",
                options=hf_models,
                index=2
            )

        st.session_state.llm_model_choice = model_choice

        if backend_option == "Hugging Face Transformers" and not is_ollama_model(model_choice):
            st.session_state.use_4bit_quantization = st.checkbox(
                "🗜️ Use 4-bit quantization", value=True
            )

        if backend_option == "Ollama (if installed)" or is_ollama_model(model_choice):
            st.session_state.ollama_host = st.text_input(
                "🌐 Ollama Host", value=st.session_state.ollama_host
            )

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
            "Citation display style",
            options=["apa", "doi", "full", "short"],
            index=0,
            format_func=lambda x: {
                "apa": "APA: FirstAuthor et al., Journal, Year",
                "doi": "DOI: 10.xxxx/xxxxx",
                "full": "Full: Author (Year). Title. Journal, Vol(Issue), Pages",
                "short": "Short: [FirstAuthor Year] or [DOI]"
            }[x]
        )

        st.session_state.max_retrieved_chunks = st.slider(
            "Chunks to retrieve", min_value=2, max_value=10, value=6
        )

        st.markdown("---")
        st.markdown("""
        <div style="background:#f0f9ff;padding:1rem;border-radius:0.5rem;border-left:4px solid #3b82f6">
        <strong>💡 DECLARMIMA Domain Features:</strong>
        <ul style="margin:0.5rem 0 0 1rem;padding:0">
        <li><b>Materials:</b> Ti6Al4V, Inconel 718, SAC, Al-Cr-Fe-Ni, HEAs</li>
        <li><b>Processes:</b> SLM, LPBF, WFLAM, DED</li>
        <li><b>Parameters:</b> Laser power, scan speed, hatch distance, layer thickness</li>
        <li><b>Methods:</b> Phase field, MD, FEM, CALPHAD, X-ray imaging</li>
        <li><b>Cross-doc consensus</b>: Statistical agreement across papers</li>
        <li><b>Contradiction detection</b>: Flags conflicting results</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

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
        "Select PDF or TXT files about laser processing, ablation, microstructuring, additive manufacturing, etc.",
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
                st.success(f"✅ Ready! Indexed {summary['total_chunks']} chunks, {summary['unique_entities']} unique entities, {summary['total_claims']} claims from {summary['document_count']} papers")
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

    if prompt := st.chat_input("Ask about laser parameters, ablation thresholds, LIPSS formation, SLM process, HEAs, etc."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            with st.spinner("🔍 Performing cross-document reasoning..."):
                try:
                    answer, retrieved_docs, relevance, reasoning_meta = retrieve_and_answer(
                        vectorstore=st.session_state.vectorstore,
                        graph=st.session_state.knowledge_graph,
                        tokenizer=st.session_state.llm_tokenizer,
                        model=st.session_state.llm_model,
                        device_or_host=st.session_state.llm_device_or_host,
                        backend=st.session_state.llm_model_choice,
                        backend_type=st.session_state.llm_backend_type,
                        query=prompt,
                        k=st.session_state.max_retrieved_chunks
                    )

                    reasoning_meta["relevance"] = relevance

                    display_text = ""
                    for word in answer.split():
                        display_text += word + " "
                        message_placeholder.markdown(display_text + "▌")
                        time.sleep(0.015)
                    message_placeholder.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": retrieved_docs if st.session_state.show_sources else None,
                        "relevance": relevance,
                        "reasoning_meta": reasoning_meta
                    })

                except Exception as e:
                    error_msg = f"❌ Error: {str(e)[:300]}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


def render_footer():
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📚 Example Questions:**")
        st.caption("• What is the consensus ablation threshold for silicon across papers?")
        st.caption("• How does scan speed affect melt pool geometry in Ti6Al4V SLM?")
        st.caption("• What contradictions exist regarding optimal laser power for Inconel 718?")

    with col2:
        st.markdown("**⚡ Reasoning Tips:**")
        st.caption("• Ask comparative questions to trigger consensus detection")
        st.caption("• Query specific materials (Ti6Al4V, SAC, HEA) to activate entity linking")
        st.caption("• Look for the 🧠 Reasoning Chain expander for transparency")

    with col3:
        st.markdown("**🔐 Privacy & Science:**")
        st.caption("• All processing happens locally")
        st.caption("• Cross-document reasoning uses extracted entities only")
        st.caption("• Uncertainty is explicitly reported, never hidden")


def main():
    st.set_page_config(
        page_title="🔬 DECLARMIMA RAG + Cross-Doc Reasoning",
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

    st.markdown('<h1 class="main-header">🔬 DECLARMIMA RAG + Cross-Doc Reasoning</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:#64748b;margin-bottom:1.5rem">
    Upload research papers and get <strong>scientifically rigorous answers</strong> with 
    <span class="consensus-badge">cross-document consensus</span>, 
    <span class="contradiction-badge">contradiction detection</span>, and 
    <span class="reasoning-badge">multi-hop reasoning</span>.
    <br><em>Specialized for Additive Manufacturing, SLM/LPBF, High Entropy Alloys, and Laser-Microstructure Interaction.</em>
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
                st.caption(f"📦 {summary['total_chunks']} chunks | {summary['unique_entities']} entities | {summary['total_claims']} claims")
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
            <li><strong>Entity Extraction:</strong> Identifies materials (Ti6Al4V, Inconel, SAC, HEA), parameters, methods</li>
            <li><strong>Cross-Document Alignment:</strong> Links the same entity across different papers</li>
            <li><strong>Consensus Detection:</strong> Statistically aggregates values reported in multiple papers</li>
            <li><strong>Contradiction Flagging:</strong> Highlights when papers disagree significantly</li>
            <li><strong>Multi-Hop Retrieval:</strong> Follows entity links to find related evidence</li>
            <li><strong>Uncertainty Calibration:</strong> Explicit confidence levels in every answer</li>
            </ul>
            <p><strong>Getting started:</strong></p>
            <ol>
            <li>Upload 2+ PDF/TXT papers on the same topic</li>
            <li>Enable "Cross-document reasoning" in sidebar</li>
            <li>Ask comparative or synthesizing questions</li>
            <li>Expand "🧠 Reasoning Chain" to see the logical steps</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Try asking:**")
            demo_qs = [
                "What is the consensus ablation threshold for silicon across all papers?",
                "Do these papers agree on the effect of scan speed on melt pool depth in Ti6Al4V?",
                "What contradictions exist regarding optimal laser power for Inconel 718 LPBF?",
                "Summarize the characterization methods used across all uploaded papers.",
            ]
            for q in demo_qs:
                if st.button(f"💬 {q}", use_container_width=True, key=f"demo_{q[:20]}"):
                    st.session_state.demo_question = q
                    st.rerun()

    render_footer()

    if hasattr(st.session_state, 'demo_question') and st.session_state.demo_question:
        st.session_state.messages.append({"role": "user", "content": st.session_state.demo_question})
        del st.session_state.demo_question
        st.rerun()


if __name__ == "__main__":
    main()
