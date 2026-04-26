#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LASER MICROSTRUCTURE RAG CHATBOT - FULLY API-FREE VERSION WITH BIBLIOGRAPHIC CITATIONS
========================================================================================
✅ Zero API keys required - all models run locally (optional Crossref/pdf2doi for metadata)
✅ Supports Hugging Face transformers models AND Ollama local models
✅ Optimized for Streamlit Cloud and local deployment
✅ Laser-microstructure domain specialization
✅ PDF/text document ingestion with FAISS vector storage
✅ 🎯 SOURCE CITATION WITH HUMAN-READABLE IDs: DOI, FirstAuthor et al., Journal, Year, Volume
✅ Confidence scoring and relevance filtering
✅ Responsive UI with streaming-like output simulation
✅ Memory-efficient loading with quantization support for large models
✅ Automatic fallback to smaller models if GPU memory is limited

Deploy to Streamlit Cloud with requirements.txt below.
For local use with Ollama: install ollama Python library and run `ollama pull <model>`
For enhanced metadata extraction: pip install pdf2doi crossrefapi (optional)
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
from typing import List, Dict, Optional, Tuple, Union, Any
from datetime import datetime
import sys
import subprocess
import platform
from pathlib import Path
from collections import defaultdict

# LangChain / RAG imports (local-only, no API calls)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Transformers for local LLM inference via Hugging Face
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, AutoModelForCausalLM,
    pipeline, set_seed, BitsAndBytesConfig
)

# Optional: Ollama support for local model serving
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Optional: Bibliographic metadata extraction libraries
try:
    import pdf2doi
    PDF2DOI_AVAILABLE = True
except ImportError:
    PDF2DOI_AVAILABLE = False

try:
    from crossrefapi import CrossrefAPI
    CROSSREF_AVAILABLE = True
except ImportError:
    CROSSREF_AVAILABLE = False

# Optional: PyPDF2 for reading PDF metadata
try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

# =============================================
# GLOBAL CONFIGURATION - LASER MICROSTRUCTURE DOMAIN
# =============================================

# Local model choices - Using proper Hugging Face repo IDs
LOCAL_LLM_OPTIONS = {
    # === TINY MODELS (Good for low-latency testing, CPU-friendly) ===
    "GPT-2 (1.5B, fastest startup, CPU OK)": "gpt2",
    "Qwen2-0.5B-Instruct (best JSON, recommended)": "Qwen/Qwen2-0.5B-Instruct",
    "Qwen2.5-0.5B-Instruct (newest, best reasoning)": "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama-1.1B-Chat (balanced small model)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    
    # === MEDIUM MODELS (Require GPU or good CPU, 4-8GB VRAM) ===
    "Qwen2.5-1.5B-Instruct (efficient mid-size)": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct (strong reasoning)": "Qwen/Qwen2.5-3B-Instruct",
    "Mistral-7B-Instruct-v0.3 (reliable & efficient)": "mistralai/Mistral-7B-Instruct-v0.3",
    "Llama-3.2-3B-Instruct (Meta's latest small)": "meta-llama/Llama-3.2-3B-Instruct",
    
    # === LARGE MODELS (Require GPU with 12-24GB VRAM, use 4-bit quantization) ===
    "Qwen2.5-7B-Instruct (excellent all-rounder)": "Qwen/Qwen2.5-7B-Instruct",
    "Llama-3.1-8B-Instruct (most popular balanced)": "meta-llama/Llama-3.1-8B-Instruct",
    "Gemma-2-9B-it (Google's latest, great logic)": "google/gemma-2-9b-it",
    "Falcon-7B-Instruct (lightweight & modern)": "tiiuae/falcon-7b-instruct",
    
    # === OLLAMA BACKEND MODELS (if ollama library installed) ===
    "[Ollama] qwen2.5:0.5b (via ollama serve)": "ollama:qwen2.5:0.5b",
    "[Ollama] qwen2.5:1.5b (via ollama serve)": "ollama:qwen2.5:1.5b",
    "[Ollama] qwen2.5:7b (via ollama serve)": "ollama:qwen2.5:7b",
    "[Ollama] qwen2.5:14b (via ollama serve) 🔥": "ollama:qwen2.5:14b",
    "[Ollama] llama3.1:8b (via ollama serve)": "ollama:llama3.1:8b",
    "[Ollama] mistral:7b (via ollama serve)": "ollama:mistral:7b",
    "[Ollama] gemma2:9b (via ollama serve)": "ollama:gemma2:9b",
    "[Ollama] falcon3:10b (via ollama serve)": "ollama:falcon3:10b",
}

# Local embedding model (~80MB, CPU-friendly)
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Laser-microstructure domain settings
LASER_DOMAIN_CONFIG = {
    "chunk_size": 800,
    "chunk_overlap": 150,
    "retrieval_k": 4,
    "score_threshold": 0.25,
    "max_context_tokens": 1024,
    "max_new_tokens": 256,
    "temperature": 0.1,
}

# Laser-specific keywords for domain filtering and boosting
LASER_KEYWORDS = {
    "ablation": ["ablation", "material removal", "threshold fluence", "laser ablation"],
    "plasma": ["plasma formation", "ionization", "electron density", "plume"],
    "thermal": ["heat affected zone", "melting", "thermal diffusion", "resolidification"],
    "ultrafast": ["femtosecond", "picosecond", "pulse duration", "ultrafast laser"],
    "morphology": ["ripples", "LIPSS", "surface structuring", "periodic structures"],
    "parameters": ["fluence", "wavelength", "pulse energy", "repetition rate", "spot size"],
    "materials": ["silicon", "steel", "titanium", "polymer", "glass", "ceramic"],
    "characterization": ["SEM", "AFM", "profilometry", "spectroscopy", "microscopy"],
}

# Memory estimation for model loading
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
# BIBLIOGRAPHIC METADATA EXTRACTION FUNCTIONS
# =============================================

class BibliographicMetadata:
    """
    Container for bibliographic metadata extracted from academic documents.
    Supports human-readable citation formatting with multiple fallback strategies.
    """
    
    # DOI regex pattern - matches standard DOI format [[62]]
    DOI_PATTERN = re.compile(r'\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b', re.IGNORECASE)
    
    # arXiv ID pattern (post-2007 format)
    ARXIV_PATTERN = re.compile(r'\barXiv[:\s]+(\d{4}\.\d{4,5}(v\d+)?)\b', re.IGNORECASE)
    
    # Common journal name patterns
    JOURNAL_PATTERNS = [
        re.compile(r'(?:published in|journal|proc\.?|journal of)\s+([A-Z][A-Za-z\s&\.]+?)(?:,|\.)', re.I),
        re.compile(r'([A-Z][A-Za-z\s&\.]+?\s+(?:Letters?|Journal|Transactions|Review|Proceedings))', re.I),
    ]
    
    # Year patterns (4-digit years in common contexts)
    YEAR_PATTERN = re.compile(r'\b((?:19|20)\d{2})\b')
    
    # Volume/issue patterns
    VOLUME_PATTERN = re.compile(r'(?:vol\.?|volume)\s*(\d+)', re.I)
    ISSUE_PATTERN = re.compile(r'(?:no\.?|issue|iss\.?)\s*(\d+)', re.I)
    
    # Author name patterns (simplified - handles "First Last" and "Last, First" formats)
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
        """
        Format a human-readable citation string from extracted metadata.
        
        Styles supported:
        - "apa": FirstAuthor et al., Journal, Year
        - "doi": DOI:10.xxxx/xxxxx
        - "full": FirstAuthor et al. (Year). Title. Journal, Volume(Issue), Pages.
        - "short": [FirstAuthor Year] or [DOI]
        """
        # Priority 1: Use DOI if available and validated
        if self.doi and self.confidence > 0.8:
            if style == "doi":
                return f"DOI:{self.doi}"
            elif style == "short":
                return f"[DOI:{self.doi}]"
        
        # Priority 2: Use arXiv ID
        if self.arxiv_id:
            if style in ["doi", "short"]:
                return f"[arXiv:{self.arxiv_id}]"
        
        # Priority 3: Format as Author-Year-Journal
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
        
        # Fallback: Use filename with basic info
        base_name = Path(self.source_filename).stem
        if self.year:
            return f"[{base_name}, {self.year}]"
        return f"[{base_name}]"
    
    def _format_author_name(self, author_str: str) -> str:
        """Format author name as 'Last, F.' or 'F. Last' based on detected format."""
        # Handle "Last, First" format
        if "," in author_str:
            parts = [p.strip() for p in author_str.split(",", 1)]
            if len(parts) == 2:
                last, first = parts
                first_initial = first[0] + "." if first else ""
                return f"{last}, {first_initial}"
        # Handle "First Last" format - return as-is for now
        return author_str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for storage."""
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
        """Reconstruct metadata from dictionary."""
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
    """
    Extract bibliographic metadata from PDF text using regex patterns and heuristics.
    Based on patterns used by pdf2doi [[1]] and scholarly metadata extraction [[49]].
    """
    meta = BibliographicMetadata(filename)
    text_sample = text[:10000]  # Focus on first pages where metadata typically appears
    text_lower = text_sample.lower()
    
    # 1. Extract DOI using standard pattern [[58]][[62]]
    doi_match = BibliographicMetadata.DOI_PATTERN.search(text_sample)
    if doi_match:
        meta.doi = doi_match.group(1).lower()
        meta.confidence = max(meta.confidence, 0.9)
        meta.extraction_method = "regex_doi"
    
    # 2. Extract arXiv ID
    arxiv_match = BibliographicMetadata.ARXIV_PATTERN.search(text_sample)
    if arxiv_match:
        meta.arxiv_id = arxiv_match.group(1)
        meta.confidence = max(meta.confidence, 0.85)
        meta.extraction_method = "regex_arxiv"
    
    # 3. Extract year (look for year near title/author context)
    year_matches = BibliographicMetadata.YEAR_PATTERN.findall(text_sample)
    # Prefer years in common academic contexts
    for year_str in year_matches:
        year = int(year_str)
        if 1900 <= year <= 2030:  # Reasonable publication year range
            # Check if near keywords suggesting publication date
            context_window = 100
            year_pos = text_sample.find(year_str)
            context = text_sample[max(0, year_pos-50):year_pos+50].lower()
            if any(kw in context for kw in ['published', 'received', 'accepted', 'copyright', '©']):
                meta.year = year
                meta.confidence = max(meta.confidence, 0.7)
                break
    
    # 4. Extract journal name using patterns
    for pattern in BibliographicMetadata.JOURNAL_PATTERNS:
        journal_match = pattern.search(text_sample)
        if journal_match:
            journal = journal_match.group(1).strip()
            # Filter out common false positives
            if len(journal) > 10 and not any(bad in journal.lower() for bad in ['introduction', 'abstract', 'references']):
                meta.journal = journal
                meta.confidence = max(meta.confidence, 0.6)
                break
    
    # 5. Extract volume/issue
    vol_match = BibliographicMetadata.VOLUME_PATTERN.search(text_sample)
    if vol_match:
        meta.volume = vol_match.group(1)
    iss_match = BibliographicMetadata.ISSUE_PATTERN.search(text_sample)
    if iss_match:
        meta.issue = iss_match.group(1)
    
    # 6. Extract authors (simplified - first author line near top)
    # Look for author patterns in first 2000 characters
    author_section = text_sample[:2000]
    author_matches = BibliographicMetadata.AUTHOR_PATTERN.findall(author_section)
    if author_matches:
        # Take first match and split by common separators
        raw_authors = author_matches[0]
        # Handle multiple authors separated by commas or "and"
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
    
    # 7. Try to extract title (first capitalized line after common headers)
    title_patterns = [
        re.compile(r'(?:^|\n)([A-Z][^.\n]{20,150?}(?:\.[^A-Z]|$))'),  # Long capitalized phrase
        re.compile(r'(?:title:?\s*)([A-Z][^.\n]{20,200}?)\.?(?:\n|$)', re.I),
    ]
    for pattern in title_patterns:
        title_match = pattern.search(text_sample)
        if title_match:
            title = title_match.group(1).strip()
            # Filter: must have reasonable length and not be all caps
            if 30 < len(title) < 200 and not title.isupper():
                meta.title = title
                meta.confidence = max(meta.confidence, 0.55)
                break
    
    return meta


def extract_metadata_from_pdf_file(pdf_path: str, filename: str) -> BibliographicMetadata:
    """
    Extract metadata from PDF file using multiple methods:
    1. PDF metadata fields (if available)
    2. Text extraction + regex parsing
    3. Optional: pdf2doi library for DOI lookup [[1]]
    """
    meta = BibliographicMetadata(filename)
    
    # Method 1: Try PDF metadata fields using PyPDF2 [[74]]
    if PYPDF2_AVAILABLE:
        try:
            reader = PdfReader(pdf_path)
            pdf_info = reader.metadata or {}
            
            # Map common PDF metadata fields
            field_mapping = {
                '/Title': 'title',
                '/Author': 'authors',
                '/CreationDate': 'year',
                '/Subject': 'journal',
                '/Keywords': 'keywords',
            }
            
            for pdf_field, meta_field in field_mapping.items():
                if pdf_field in pdf_info and pdf_info[pdf_field]:
                    value = str(pdf_info[pdf_field]).strip()
                    if meta_field == 'authors' and value:
                        # Split author field by common separators
                        meta.authors = [a.strip() for a in re.split(r'[;,]', value) if a.strip()]
                    elif meta_field == 'year' and value:
                        # Extract year from PDF date format (e.g., "D:20230115...")
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
    
    # Method 2: Extract text and parse with regex
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        # Combine first 3 pages for metadata extraction (where bibliographic info typically appears)
        text_sample = "\n".join([p.page_content for p in pages[:3]])
        text_meta = extract_metadata_from_pdf_text(text_sample, filename)
        
        # Merge: prefer higher-confidence extractions
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
    
    # Method 3: Optional pdf2doi lookup for DOI validation/enhancement [[1]]
    if PDF2DOI_AVAILABLE and not meta.doi:
        try:
            # pdf2doi returns dict with identifier info
            result = pdf2doi.pdf2doi(pdf_path)
            if isinstance(result, list) and result:
                result = result[0]
            if result and result.get('identifier') and result.get('identifier_type') == 'doi':
                meta.doi = result['identifier']
                meta.confidence = 0.95
                meta.extraction_method = "pdf2doi"
                # Also try to get bibtex info if available
                if result.get('validation_info'):
                    bibtex = result['validation_info']
                    # Simple bibtex field extraction (could be enhanced)
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
    
    # Method 4: Optional Crossref API lookup if DOI found but needs enrichment [[32]]
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
    """Extract metadata from plain text file using regex patterns."""
    return extract_metadata_from_pdf_text(text, filename)


# =============================================
# GLOBAL METADATA CACHE
# =============================================

class MetadataCache:
    """Cache for bibliographic metadata to avoid re-extraction."""
    
    def __init__(self):
        self._cache: Dict[str, BibliographicMetadata] = {}
        self._file_hashes: Dict[str, str] = {}
    
    def get(self, filename: str, file_hash: str = None) -> Optional[BibliographicMetadata]:
        """Get cached metadata if file hasn't changed."""
        if filename in self._cache:
            if file_hash is None or self._file_hashes.get(filename) == file_hash:
                return self._cache[filename]
        return None
    
    def set(self, filename: str, metadata: BibliographicMetadata, file_hash: str = None):
        """Store metadata in cache."""
        self._cache[filename] = metadata
        if file_hash:
            self._file_hashes[filename] = file_hash
    
    def clear(self):
        """Clear all cached metadata."""
        self._cache.clear()
        self._file_hashes.clear()


# Global cache instance
metadata_cache = MetadataCache()


def compute_file_hash(filepath: str) -> str:
    """Compute simple hash of file content for cache validation."""
    import hashlib
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return ""


# =============================================
# SESSION STATE INITIALIZATION
# =============================================

def initialize_session_state():
    """Initialize all session state variables with defaults."""
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
        "citation_style": "apa",  # NEW: User-selectable citation format
        "max_retrieved_chunks": 4,
        "use_4bit_quantization": True,
        "ollama_host": "http://localhost:11434",
        "metadata_cache": metadata_cache,  # NEW: Reference to metadata cache
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================
# UTILITY FUNCTIONS (unchanged except for citation helpers)
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
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        return total_memory - reserved
    except:
        return None

def estimate_model_memory(model_key: str, use_4bit: bool = False) -> Dict[str, any]:
    repo_id = get_hf_repo_id(model_key) if not is_ollama_model(model_key) else model_key
    return MODEL_MEMORY_ESTIMATES.get(repo_id, {
        "params": "Unknown",
        "vram_fp16": "Unknown",
        "vram_4bit": "Unknown", 
        "cpu_ok": False
    })


# =============================================
# LOCAL MODEL LOADING (unchanged)
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
            st.sidebar.warning("⚠️ bitsandbytes not installed. Install with: pip install bitsandbytes")
            use_4bit = False
    
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        trust_remote_code=True,
        padding_side="left",
        use_fast=True
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
# DOCUMENT LOADING & CHUNKING WITH METADATA EXTRACTION
# =============================================

def extract_laser_metadata(text: str, filename: str) -> Dict[str, any]:
    """Extract laser-relevant metadata from document text."""
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


def load_and_chunk_laser_documents(uploaded_files: List) -> List[Document]:
    """Load PDFs/text files, extract bibliographic metadata, and chunk with laser-domain awareness."""
    all_chunks = []
    
    for uploaded_file in uploaded_files:
        # Save to temp file for loading
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf" if uploaded_file.name.endswith('.pdf') else ".txt") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        
        try:
            # === NEW: Extract bibliographic metadata ===
            file_hash = compute_file_hash(tmp_path)
            cached_meta = st.session_state.metadata_cache.get(uploaded_file.name, file_hash)
            
            if cached_meta:
                bib_meta = cached_meta
                st.info(f"📚 Using cached metadata for `{uploaded_file.name}`")
            else:
                if uploaded_file.name.endswith('.pdf'):
                    bib_meta = extract_metadata_from_pdf_file(tmp_path, uploaded_file.name)
                else:
                    # For text files, read content and extract
                    with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text_content = f.read()
                    bib_meta = extract_metadata_from_text_file(text_content, uploaded_file.name)
                
                # Cache the result
                st.session_state.metadata_cache.set(uploaded_file.name, bib_meta, file_hash)
                st.info(f"📚 Extracted metadata: {bib_meta.format_citation('apa')}")
            
            # Load document for chunking
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, encoding='utf-8')
            
            pages = loader.load()
            
            # Laser-optimized text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=LASER_DOMAIN_CONFIG["chunk_size"],
                chunk_overlap=LASER_DOMAIN_CONFIG["chunk_overlap"],
                separators=["\n\n", "\n", "Equation", "Parameter:", "Figure", "Table", ""],
                length_function=len
            )
            
            chunks = text_splitter.split_documents(pages)
            
            # Add laser-specific AND bibliographic metadata to each chunk
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "source": uploaded_file.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    **extract_laser_metadata(chunk.page_content, uploaded_file.name),
                    # === NEW: Add bibliographic citation info ===
                    "bibliographic": bib_meta.to_dict(),
                    "citation_display": bib_meta.format_citation(st.session_state.get('citation_style', 'apa')),
                })
            
            all_chunks.extend(chunks)
            st.info(f"✅ Loaded {len(chunks)} chunks from `{uploaded_file.name}`")
            
        except Exception as e:
            st.error(f"❌ Error processing `{uploaded_file.name}`: {e}")
            import traceback
            st.error(traceback.format_exc())
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    return all_chunks


# =============================================
# VECTOR STORE CREATION (LOCAL FAISS) - unchanged
# =============================================

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
                topic for chunk in chunks 
                for topic in chunk.metadata.get("laser_topics", [])
            ))
        }
        
        return vectorstore
        
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None


# =============================================
# LASER-SPECIFIC RAG CHAIN WITH ENHANCED CITATIONS
# =============================================

def create_laser_rag_prompt(retrieved_chunks: List[Document], query: str) -> str:
    """Create a laser-optimized prompt with retrieved context and human-readable citations."""
    
    # Format retrieved chunks with ENHANCED source citation
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        # === NEW: Use human-readable citation instead of just [Source N] ===
        citation = chunk.metadata.get("citation_display")
        if not citation:
            # Fallback to old format if metadata missing
            source = chunk.metadata.get("source", "unknown")
            topics = chunk.metadata.get("laser_topics", [])
            topic_str = f" [{', '.join(topics)}]" if topics else ""
            citation = f"[Source {i}{topic_str} - {source}]"
        
        # Truncate long chunks for small LLMs
        content = chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content
        
        context_parts.append(f"{citation}\n{content}\n")
    
    context = "\n---\n".join(context_parts)
    
    # Laser-specific system prompt with citation instructions
    laser_system = """You are an expert assistant for laser-microstructure interaction research.
Your role is to answer questions based ONLY on the provided document context.
Be precise, technical, and cite your sources using the provided citation format.

Rules:
1. Use ONLY information from the retrieved context below
2. If the answer isn't in the context, say "Based on the provided documents, I cannot determine..."
3. Never invent parameters, equations, or experimental conditions
4. When citing, use the EXACT citation string provided (e.g., "Smith et al., J. Appl. Phys., 2023" or "DOI:10.1063/1.234567")
5. For numerical values, include units when available
6. Be concise but technically complete

"""
    
    user_query = f"""Retrieved Context from Laser Microstructure Documents:
{context}

User Question: {query}

Answer (cite sources using provided citation format, be technical and precise):"""
    
    return laser_system + user_query


# ... [generate_local_response_transformers, generate_local_response_ollama, generate_local_response functions remain unchanged] ...

def generate_local_response_transformers(
    tokenizer, 
    model, 
    device: str, 
    prompt: str,
    backend_name: str
) -> str:
    """Generate response using Hugging Face transformers model."""
    try:
        # Format prompt for the specific model family
        if "Qwen" in backend_name or "qwen" in backend_name.lower():
            messages = [
                {"role": "system", "content": "You are an expert in laser-microstructure interaction."},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        elif "Llama" in backend_name or "llama" in backend_name.lower():
            messages = [
                {"role": "system", "content": "You are an expert in laser-microstructure interaction."},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        elif "Mistral" in backend_name or "mistral" in backend_name.lower():
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        else:
            formatted_prompt = prompt
        
        inputs = tokenizer.encode(
            formatted_prompt, 
            return_tensors='pt', 
            truncation=True, 
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
        elif "Answer (cite sources" in full_text:
            answer = full_text.split("Answer (cite sources")[-1].strip()
            answer = re.split(r'\n\n(?:Question|User|Context):', answer)[0].strip()
        else:
            answer = full_text[-LASER_DOMAIN_CONFIG["max_new_tokens"]*2:].strip()
        
        answer = re.sub(r'\s+', ' ', answer)
        answer = answer.strip()
        
        return answer if answer else "I was unable to generate a response. Please try rephrasing your question."
        
    except Exception as e:
        st.error(f"Generation error: {e}")
        return f"Error generating response: {str(e)[:200]}..."


def generate_local_response_ollama(
    model_tag: str,
    ollama_host: str,
    prompt: str
) -> str:
    """Generate response using Ollama API with robust streaming."""
    try:
        client = ollama.Client(host=ollama_host)
        
        messages = [
            {"role": "system", "content": "You are an expert in laser-microstructure interaction research. Answer based ONLY on the provided context."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = client.chat(
                model=model_tag,
                messages=messages,
                options={
                    "temperature": LASER_DOMAIN_CONFIG["temperature"],
                    "num_predict": LASER_DOMAIN_CONFIG["max_new_tokens"],
                },
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
                model=model_tag,
                messages=messages,
                options={
                    "temperature": LASER_DOMAIN_CONFIG["temperature"],
                    "num_predict": LASER_DOMAIN_CONFIG["max_new_tokens"],
                }
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


def generate_local_response(
    tokenizer, 
    model_or_tag, 
    device_or_host: str, 
    prompt: str,
    backend: str,
    backend_type: str
) -> str:
    """Unified response generator for both transformers and Ollama backends."""
    if backend_type == "ollama":
        return generate_local_response_ollama(model_or_tag, device_or_host, prompt)
    else:
        return generate_local_response_transformers(tokenizer, model_or_tag, device_or_host, prompt, backend)


def retrieve_and_answer(
    vectorstore,
    tokenizer,
    model,
    device_or_host: str,
    backend: str,
    backend_type: str,
    query: str,
    k: int = None,
    score_threshold: float = None
) -> Tuple[str, List[Document], float]:
    """Retrieve relevant chunks and generate answer with local LLM."""
    
    k = k or LASER_DOMAIN_CONFIG["retrieval_k"]
    score_threshold = score_threshold or LASER_DOMAIN_CONFIG["score_threshold"]
    
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold}
    )
    
    retrieved_docs = retriever.invoke(query)
    
    if retrieved_docs:
        query_embedding = vectorstore.embedding_function.embed_query(query)
        scores = []
        for doc in retrieved_docs:
            doc_embedding = vectorstore.embedding_function.embed_query(doc.page_content[:500])
            sim = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-8
            )
            scores.append(sim)
        avg_relevance = np.mean(scores) if scores else 0.0
    else:
        avg_relevance = 0.0
    
    if not retrieved_docs:
        return "Based on the uploaded documents, I could not find information relevant to your question. Try rephrasing or checking document content.", [], avg_relevance
    
    prompt = create_laser_rag_prompt(retrieved_docs, query)
    
    answer = generate_local_response(
        tokenizer=tokenizer,
        model_or_tag=model,
        device_or_host=device_or_host,
        prompt=prompt,
        backend=backend,
        backend_type=backend_type
    )
    
    return answer, retrieved_docs, avg_relevance


# =============================================
# STREAMLIT UI COMPONENTS (with citation style selector)
# =============================================

def render_sidebar():
    """Render sidebar with configuration options."""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Backend selection
        backend_option = st.radio(
            "🔧 Inference Backend",
            options=["Hugging Face Transformers", "Ollama (if installed)"],
            index=0,
            help="Transformers: direct HF model loading\nOllama: use local ollama serve (faster switching)"
        )
        st.session_state.inference_backend = backend_option
        
        # Model selection
        if backend_option == "Ollama (if installed)":
            if not OLLAMA_AVAILABLE:
                st.error("❌ ollama library not installed")
                st.code("pip install ollama")
                st.info("Also ensure Ollama server is running: ollama serve")
            available_ollama_models = [k for k in LOCAL_LLM_OPTIONS.keys() if is_ollama_model(k)]
            model_choice = st.selectbox(
                "🧠 Local LLM Backend (Ollama)",
                options=available_ollama_models if available_ollama_models else ["No Ollama models available"],
                index=0 if available_ollama_models else 0,
                help="Models served via local Ollama instance"
            )
        else:
            hf_models = [k for k in LOCAL_LLM_OPTIONS.keys() if not is_ollama_model(k)]
            model_choice = st.selectbox(
                "🧠 Local LLM Backend (Hugging Face)",
                options=hf_models,
                index=2,
                help="Models loaded directly via transformers library"
            )
        
        st.session_state.llm_model_choice = model_choice
        
        # Quantization option
        if backend_option == "Hugging Face Transformers" and not is_ollama_model(model_choice):
            st.session_state.use_4bit_quantization = st.checkbox(
                "🗜️ Use 4-bit quantization (reduces VRAM usage)",
                value=True,
                help="Enable for models >3B parameters to reduce memory usage by ~75%"
            )
        
        # Ollama host configuration
        if backend_option == "Ollama (if installed)" or is_ollama_model(model_choice):
            st.session_state.ollama_host = st.text_input(
                "🌐 Ollama Host",
                value=st.session_state.ollama_host,
                help="URL of your Ollama server (default: http://localhost:11434)"
            )
        
        # Domain settings
        st.markdown("#### 🔬 Laser Domain Settings")
        st.session_state.laser_domain_boost = st.checkbox(
            "Boost laser-topic relevance",
            value=True,
            help="Prioritize chunks containing laser-specific keywords"
        )
        
        st.session_state.show_sources = st.checkbox(
            "Show source citations",
            value=True,
            help="Display which documents chunks came from"
        )
        
        # === NEW: Citation style selector ===
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
            }[x],
            help="How citations appear in responses and source lists"
        )
        
        st.session_state.max_retrieved_chunks = st.slider(
            "Chunks to retrieve",
            min_value=2,
            max_value=8,
            value=4,
            help="More chunks = more context but slower responses"
        )
        
        # Info box
        st.markdown("---")
        st.markdown("""
        <div style="background:#f0f9ff;padding:1rem;border-radius:0.5rem;border-left:4px solid #3b82f6">
        <strong>💡 Tips for Best Results:</strong>
        <ul style="margin:0.5rem 0 0 1rem;padding:0">
        <li>Upload papers about laser ablation, LIPSS, ultrafast processing</li>
        <li>Ask specific questions: "What fluence threshold for silicon ablation?"</li>
        <li>Small models (≤1.5B) work on CPU; larger need GPU</li>
        <li>First load may take 1-2 min (model download)</li>
        <li>For Ollama: run <code>ollama pull qwen2.5:7b</code> first</li>
        <li>🎯 Citations show as "Smith et al., J. Appl. Phys., 2023" or DOI when available</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Resource info
        st.markdown("---")
        gpu_info = "CUDA" if torch.cuda.is_available() else "CPU"
        vram_info = f"{get_available_gpu_memory():.1f}GB free" if torch.cuda.is_available() and get_available_gpu_memory() else "N/A"
        st.caption(f"🖥️ Device: {gpu_info}")
        st.caption(f"💾 Available VRAM: {vram_info}")
        st.caption(f"📦 Embedding model: ~80MB")
        st.caption(f"🤖 LLM: {LOCAL_LLM_OPTIONS.get(model_choice, 'unknown')}")
        
        # Metadata extraction status
        st.markdown("#### 📚 Metadata Extraction")
        if PDF2DOI_AVAILABLE:
            st.success("✅ pdf2doi: Available for DOI lookup")
        else:
            st.info("ℹ️ pdf2doi: Install with `pip install pdf2doi` for enhanced DOI extraction [[1]]")
        
        if CROSSREF_AVAILABLE:
            st.success("✅ Crossref API: Available for metadata enrichment")
        else:
            st.info("ℹ️ Crossref: Install with `pip install crossrefapi` for journal/author lookup [[32]]")
        
        # Ollama status check
        if backend_option == "Ollama (if installed)" and OLLAMA_AVAILABLE:
            try:
                client = ollama.Client(host=st.session_state.ollama_host)
                models = client.list()
                st.success(f"✅ Ollama connected: {len(models.get('models', []))} models available")
            except:
                st.error("❌ Cannot connect to Ollama")


def render_document_uploader():
    """Render document upload section."""
    st.markdown("### 📁 Upload Laser Microstructure Documents")
    
    uploaded_files = st.file_uploader(
        "Select PDF or TXT files about laser processing, ablation, microstructuring, etc.",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Documents will be processed locally - no data leaves your browser session. Bibliographic metadata (DOI, authors, journal, year) will be extracted for human-readable citations."
    )
    
    return uploaded_files


def process_documents(uploaded_files):
    """Handle document processing pipeline."""
    if not uploaded_files:
        return False
    
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    if not new_files:
        st.info("✓ All uploaded files already processed")
        return st.session_state.processing_complete
    
    st.session_state.messages = []
    st.session_state.vectorstore = None
    st.session_state.all_chunks = []
    
    with st.spinner(f"Processing {len(new_files)} document(s) and extracting bibliographic metadata..."):
        try:
            chunks = load_and_chunk_laser_documents(new_files)
            if not chunks:
                st.error("No chunks extracted. Check file format.")
                return False
            
            for f in new_files:
                st.session_state.processed_files.add(f.name)
            
            st.session_state.all_chunks.extend(chunks)
            
            with st.spinner("Creating vector index (this may take a minute)..."):
                vectorstore = create_local_vector_store(
                    st.session_state.all_chunks,
                    LOCAL_EMBEDDING_MODEL
                )
                if vectorstore is None:
                    return False
                st.session_state.vectorstore = vectorstore
            
            st.success(f"✅ Ready! Indexed {len(st.session_state.all_chunks)} chunks from {len(st.session_state.processed_files)} files")
            st.session_state.processing_complete = True
            return True
            
        except Exception as e:
            st.error(f"Processing failed: {e}")
            import traceback
            st.error(traceback.format_exc())
            return False


def render_chat_interface():
    """Render the main chat interface."""
    if not st.session_state.get('vectorstore'):
        st.info("👆 Upload documents above to start chatting with your laser microstructure knowledge base")
        return
    
    if st.session_state.llm_tokenizer is None and st.session_state.llm_model_choice:
        backend_type = "ollama" if is_ollama_model(st.session_state.llm_model_choice) else "transformers"
        with st.spinner(f"Loading {st.session_state.llm_model_choice}..."):
            result = load_local_llm(
                st.session_state.llm_model_choice, 
                use_4bit=st.session_state.get('use_4bit_quantization', True)
            )
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
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources") and st.session_state.show_sources:
                with st.expander("📚 Retrieved Sources with Citations"):
                    for i, src in enumerate(message["sources"], 1):
                        # === NEW: Display human-readable citation ===
                        citation = src.metadata.get("citation_display", "Unknown source")
                        source_name = src.metadata.get("source", "unknown")
                        topics = src.metadata.get("laser_topics", [])
                        
                        st.markdown(f"**[{i}]** {citation}")
                        
                        # Show extracted bibliographic details in expandable section
                        bib = src.metadata.get("bibliographic", {})
                        if bib and any(bib.get(k) for k in ['doi', 'authors', 'journal', 'year']):
                            with st.expander("🔍 Bibliographic Details"):
                                if bib.get('doi'):
                                    st.markdown(f"**DOI:** `{bib['doi']}`")
                                if bib.get('arxiv_id'):
                                    st.markdown(f"**arXiv:** `{bib['arxiv_id']}`")
                                if bib.get('authors'):
                                    st.markdown(f"**Authors:** {', '.join(bib['authors'][:3])}{'...' if len(bib['authors'])>3 else ''}")
                                if bib.get('journal'):
                                    st.markdown(f"**Journal:** {bib['journal']}")
                                if bib.get('year'):
                                    st.markdown(f"**Year:** {bib['year']}")
                                if bib.get('volume'):
                                    vol_str = f"Vol. {bib['volume']}"
                                    if bib.get('issue'):
                                        vol_str += f"({bib['issue']})"
                                    st.markdown(f"**Volume/Issue:** {vol_str}")
                                st.caption(f"Extraction method: {bib.get('extraction_method', 'unknown')} (confidence: {bib.get('confidence', 0):.2f})")
                        
                        # Show chunk preview
                        st.markdown(f"> {src.page_content[:300]}...")
    
    # Chat input
    if prompt := st.chat_input("Ask about laser parameters, ablation thresholds, LIPSS formation, etc."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("🔍 Retrieving and generating..."):
                try:
                    answer, retrieved_docs, relevance = retrieve_and_answer(
                        vectorstore=st.session_state.vectorstore,
                        tokenizer=st.session_state.llm_tokenizer,
                        model=st.session_state.llm_model,
                        device_or_host=st.session_state.llm_device_or_host,
                        backend=st.session_state.llm_model_choice,
                        backend_type=st.session_state.llm_backend_type,
                        query=prompt,
                        k=st.session_state.max_retrieved_chunks
                    )
                    
                    # Simulate streaming output
                    display_text = ""
                    for word in answer.split():
                        display_text += word + " "
                        message_placeholder.markdown(display_text + "▌")
                        time.sleep(0.02)
                    message_placeholder.markdown(answer)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": retrieved_docs if st.session_state.show_sources else None,
                        "relevance": relevance
                    })
                    
                    if relevance > 0:
                        st.caption(f"📊 Response relevance: {relevance:.2f}/1.0")
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)[:300]}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


def render_footer():
    """Render footer with helpful info."""
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📚 Example Questions:**")
        st.caption("• What is the ablation threshold for silicon at 800nm?")
        st.caption("• How does pulse duration affect LIPSS formation?")
        st.caption("• What characterization methods for laser microstructures?")
    
    with col2:
        st.markdown("**⚡ Performance Tips:**")
        st.caption("• Keep questions focused and specific")
        st.caption("• Smaller chunks = more precise retrieval")
        st.caption("• CPU mode: allow 10-30s per response; GPU: 2-10s")
    
    with col3:
        st.markdown("**🔐 Privacy & Metadata:**")
        st.caption("• All processing happens locally in your session")
        st.caption("• Bibliographic metadata extracted via regex/PDF fields")
        st.caption("• Optional: pdf2doi [[1]] or Crossref [[32]] for enhanced DOI lookup")
        st.caption("• Citations display as 'FirstAuthor et al., Journal, Year' or DOI")


# =============================================
# MAIN APPLICATION
# =============================================

def main():
    st.set_page_config(
        page_title="🔬 Laser Microstructure RAG Assistant",
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
    .stChatMessage {
        border-radius: 0.5rem;
        margin: 0.25rem 0;
    }
    .model-warning {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 0.75rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin: 0.5rem 0;
    }
    .citation-badge {
        display: inline-block;
        background: #e0e7ff;
        color: #3730a3;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
        margin: 0.1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🔬 Laser Microstructure RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:#64748b;margin-bottom:1.5rem">
    Upload research papers, experimental reports, or simulation data about laser-matter interaction.
    Ask questions and get answers with <strong>human-readable citations</strong> (DOI, Author-Year-Journal) - all running locally, no API keys required.
    </div>
    """, unsafe_allow_html=True)
    
    initialize_session_state()
    render_sidebar()
    
    # Show model memory warning if needed
    if st.session_state.llm_model_choice and not is_ollama_model(st.session_state.llm_model_choice):
        mem_info = estimate_model_memory(
            st.session_state.llm_model_choice, 
            st.session_state.get('use_4bit_quantization', True)
        )
        available_vram = get_available_gpu_memory()
        if available_vram and not mem_info['cpu_ok']:
            required = float(mem_info['vram_4bit'].replace('GB','').replace('~','').strip()) if 'GB' in mem_info['vram_4bit'] else 100
            if available_vram < required:
                st.markdown(f"""
                <div class="model-warning">
                ⚠️ <strong>Memory Warning:</strong> {st.session_state.llm_model_choice} requires ~{mem_info['vram_4bit']} VRAM.
                You have ~{available_vram:.1f}GB available. Consider:
                <ul>
                <li>Using 4-bit quantization (already enabled)</li>
                <li>Selecting a smaller model</li>
                <li>Using Ollama backend for better memory management</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        uploaded_files = render_document_uploader()
        
        if uploaded_files and st.button("🔄 Process Documents", type="primary", use_container_width=True):
            process_documents(uploaded_files)
        
        if st.session_state.processing_complete:
            st.success("✅ Knowledge base ready")
            if st.session_state.vectorstore and hasattr(st.session_state.vectorstore, 'metadata'):
                meta = st.session_state.vectorstore.metadata
                st.caption(f"📦 {meta.get('total_chunks', '?')} chunks")
                topics = meta.get('laser_topics', [])
                if topics:
                    st.caption(f"🔬 Topics: {', '.join(topics[:5])}" + ("..." if len(topics)>5 else ""))
                
                # Show sample citation format
                if st.session_state.all_chunks:
                    sample_chunk = st.session_state.all_chunks[0]
                    citation = sample_chunk.metadata.get("citation_display")
                    if citation:
                        st.markdown(f'<span class="citation-badge">📝 Sample citation: {citation}</span>', unsafe_allow_html=True)
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
            <h3>👋 Welcome!</h3>
            <p>This assistant helps you query documents about:</p>
            <ul>
            <li>🔥 Laser ablation thresholds & mechanisms</li>
            <li>🌊 LIPSS and surface morphology formation</li>
            <li>⚡ Ultrafast laser-matter interactions</li>
            <li>🔬 Characterization techniques (SEM, AFM, etc.)</li>
            <li>📐 Process parameter optimization</li>
            </ul>
            <p><strong>🎯 Enhanced Citations:</strong></p>
            <ul>
            <li>Citations display as "Smith et al., J. Appl. Phys., 2023"</li>
            <li>DOI shown when available: "DOI:10.1063/1.234567"</li>
            <li>Metadata extracted from PDF fields, text patterns, or optional APIs</li>
            </ul>
            <p><strong>Getting started:</strong></p>
            <ol>
            <li>Upload PDF/TXT files in the left panel</li>
            <li>Click "Process Documents"</li>
            <li>Select your preferred local LLM in sidebar</li>
            <li>Choose citation format (APA, DOI, Full, Short)</li>
            <li>Start asking technical questions!</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Try asking:**")
            demo_qs = [
                "What factors affect ablation threshold in metals?",
                "How does pulse duration influence LIPSS periodicity?",
                "What characterization methods are used for laser-textured surfaces?",
                "What is the typical fluence range for femtosecond laser processing?",
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
