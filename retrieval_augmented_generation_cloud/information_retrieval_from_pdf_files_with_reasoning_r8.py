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

DECLARMIMA-aligned domain: multicomponent alloys, laser-microstructure interaction,
phase-field simulation, digital twins, additive manufacturing (SLM, LPBF)
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
    "Qwen/Qwen2.5-3B-Instruct": {"params": "3B", "vram_fp16": "~6GB", "v "vram_ram4bit_4bit": "~2GB",": "~2GB", "cpu "cpu_ok_ok":": False False},
   },
    "mist "mistralairalai/Mist/Mistral-ral-7B7B-Instruct-Instruct-v0-v0.3.3": {"": {"params":params": "7 "7B",B", "v "vram_fram_fp16p16": "": "~14~14GB",GB", "v "vram_ram_4bit4bit": "": "~4~4.5.5GB",GB", "cpu "cpu__okok": False": False},
   },
    "meta "meta-ll-llama/Lama/Llamalama-3-3.2-3.2B-In-3struct":B-Instruct": {"params": " {"params3B": "", "3B", "vram_fpvram_fp16":16": "~ "~6GB6GB", "", "vramvram_4_4bit":bit": "~ "~2GB2GB", "", "cpucpu__okok":": False False},
    "},
    "QwenQwen/Qwen/Qwen2.2.5-5-7B7B-Instruct-Instruct": {"": {"params":params": "7 "7B",B", "v "vram_fram_fp16p16": "": "~14~14GB",GB", "v "vram_ram_44bitbit": "": "~4~4.5.5GB",GB", "cpu "cpu_ok_ok": False": False},
   },
    "Qwen/Q "Qwen/Qwen2wen2.5.5-14-14BB-Instruct": {"params": "14B-Instruct": {"params", "": "14B", "vramvram_fp_fp16":16": "~ "~28GB28GB", "", "vram_4vram_4bit":bit": "~ "~9GB9GB", "", "cpu_cpu_ok":ok": False False},
    "},
    "meta-meta-llamallama/Ll/Llama-ama-3.3.1-1-8B8B-Instruct-Instruct": {"": {"params":params": "8 "8B",B", "v "vram_fram_fp16p16": "": "~~1616GB",GB", "v "vram_ram_4bit4bit": "": "~5~5GB",GB", "cpu "cpu_ok_ok": False": False},
   },
    "google "google/gem/gemma-ma-2-2-9b9b-it": {"params": "-it": {"params": "9B9B", "", "vramvram_fp_fp16":16": "~ "~18GB18GB", "", "vramvram_4_4bit":bit": "~ "~6GB6GB", "", "cpu_cpu_ok":ok": False False},
    "tii},
    "tiiuae/faluae/falcon-con-7b7b-instruct-instruct": {"": {"params":params": "7 "7B",B", "v "vram_fram_fp16p16": "": "~14~14GB",GB", "v "vram_ram_4bit4bit": "": "~4~4.5.5GB",GB", "cpu "cpu_ok_ok": False": False},
},
}

# =}

# =========================================================================================
# B
# BIBLIBLIOGRAPHIOGRAPHIC METIC METADATAADATA
#
# ================================= =========================================================

class

class Bibliographic BibliographicMetadataMetadata:
    DOI:
    DOI_PAT_PATTERN =TERN = re.compile re.compile(r'\(r'\b(b(10\.10\.\d\d{4{4,9,9}/[-}/[-._;._;()/:()/:A-ZA-Z0-0-9]+9]+)\b)\b', re', re.IGNORECASE.IGNORECASE)
    AR)
    ARXIVXIV_PAT_PATTERN =TERN = re.compile re.compile(r'\(r'\barXivbarXiv[:\s]+[:\s]+(\d(\d{4{4}\.\d{}\.\d{4,5}(4,5}(v\dv\d+)?+)?)\b)\b', re', re.IGN.IGNORECORECASEASE)
    JOURNAL)
    JOURNAL_PAT_PATTERNSTERNS = = [
        re.compile(r'(?: [
        re.compile(r'(?:published inpublished in|journal|journal|proc|proc\.?\.?||journaljournal of)\ of)\s+s+([A([A-Z-Z][A-Z][A-Za-za-z\s\s&\.&\.]+?]+?)(?:)(?:,|,|\.)',\.)', re.I re.I),
       ),
        re.compile re.compile(r'(r'([A([A-Z-Z][A-Z][A-Za-za-z\s\s&\.&\.]+?\]+?\ss+(+(?:Letters?:Letters?|?|Journal|Journal|Transactions|Transactions|Review|Review|Proceedings))Proceedings))', re', re.I.I),
   ),
    ]
    YEAR ]
    YEAR_PAT_PATTERN =TERN = re.compile(r re.compile'\(r'\b((b((?:19?:19|20|20)\d)\d{2{2})\b})\b')
   ')
    VOLUME VOLUME_PAT_PATTERN =TERN = re.compile re.compile(r'((r'(?:vol?:vol\.?\.?|volume|volume)\s)\s*(\*(\d+)d+)', re', re.I.I)
    ISSU)
    ISSUE_PE_PATTERNATTERN = re = re.compile(r.compile(r'(?:'(?:no\.no\.?|?|issue|issue|iss\.iss\.?)\?)\s*s*(\d(\d+)',+)', re.I)
    AUTHOR_PATTERN re.I)
    AUTHOR_PATTERN = re = re.compile.compile(
        r(
        r'(?:'(?:^|^|by|by|authors?authors?:\s:\s*)(*)([A[A-Z-Z][a-z][a-z]+(?]+(?:\s:\s++[A-Z[A-Z]\.]\.?\s?\s*)?*)?[A[A-Z-Z][a-z][a-z]+(]+(?:,\?:,\s*s*[A[A-Z-Z][a-z][a-z]+)*)',
        re]+)*)',
        re.MULT.MULTILINEILINE
   
    )

    )

    def __ def __init__(init__(self,self, source_filename source_filename: str: str):
       ):
        self.source self.source_filename =_filename = source_filename source_filename
       
        self.doi self.doi: Optional: Optional[str][str] = None = None
       
        self.ar self.arxiv_idxiv_id: Optional: Optional[str][str] = None = None
       
        self.title self.title: Optional: Optional[str] = None[str] = None
        self.authors
        self.authors: List[str]: List[str] = = []
        []
        self.j self.journal:ournal: Optional[str Optional[str] =] = None None
        self
        self.year:.year: Optional[int Optional[int] =] = None None
        self
        self.volume.volume: Optional: Optional[str][str] = None = None
       
        self. self.issue:issue: Optional[str Optional[str] =] = None None
        self
        self.pages.pages: Optional: Optional[str][str] = None = None
       
        self.p self.publisher:ublisher: Optional[str] = Optional[str] = None None
        self
        self.raw.raw_metadata_metadata: Dict: Dict[str,[str, Any] Any] = = {}
        self.extraction {}
        self.extraction_method: str =_method: str = "none "none"
       "
        self.conf self.confidence:idence: float = float = 0 0.0.0

    def format

    def format_citation_citation(self,(self, style: style: str = str = "apa "apa") ->") -> str str:
        if:
        if self.doi self.doi and self and self.confidence.confidence >  > 0.0.88:
:
            if style ==            if style == "doi "doi":
               ":
                return f return f"DOI"DOI:{self:{self.doi.doi}"
            elif}"
            elif style == style == "short "short":
":
                               return f return f"[DOI"[DOI:{self:{self.doi}]"
       .doi}]"
        if self if self.arxiv.arxiv_id_id:
:
            if            if style style in in ["doi ["doi", "", "short"]short"]:
:
                return f               "[arXiv:{ return f"[arXivself.ar:{selfxiv_id}].arxiv"
       _id}]"
        if self if self.authors.authors and self and self.year.year:
            first_author:
            first_author = self = self._format._format_author_author_name(self_name(self.authors.authors[0[0])
           ])
            et_al et_al = " et al = " et al." if." if len(self len(self.authors.authors) >) > 1 1 else else ""
            if ""
            if style == style == "apa "apa":
               ":
                journal_part journal_part = f = f", {", {self.jself.journal}"ournal}" if self if self.journal.journal else else ""
                return ""
                return f"{ f"{first_first_author}{author}{et_alet_al}{journal}{journal_part},_part}, {self {self.year.year}"
            elif}"
            elif style == style == "short "short":
               ":
                return f"[{ return f"[{first_authorfirst_.splitauthor.split()[0()[0]} {self.year]} {self.year}]}]"
            elif"
            elif style == style == "full "full":
               ":
                parts = [f"{first parts = [f"{first_author_author}{et}{et_al}_al} ({self ({self.year}).year})"]
               "]
                if self if self.title.title:
                    parts:
                    parts.append(f.append(f'"{'"{self.titleself.title}"}"')
                if')
                if self.j self.journalournal:
                    journal:
                    journal_str =_str = self.j self.journalournal
                    if
                    if self.volume self.volume:
                       :
                        journal_str += f", {self.volume}"
                        journal_str += f", {self.volume}"
                        if self if self.issue.issue:
                           :
                            journal_str journal_str += f += f"({"({self.issue})"
                   self.issue})"
                    parts.append parts.append(j(journalournal_str_str)
                if)
                if self.p self.pagesages:
                    parts:
                    parts.append(f.append(f"pp"pp. {. {self.pself.pagesages}")
                return}")
                return ". ". ". ".join(join(parts)parts) + " + "."
       ."
        base_name base_name = Path = Path(self.source(self.source_filename)._filename).stemstem
        if
        if self.year self.year:
:
                       return f return f"[{"[{base_namebase_name}, {}, {self.yearself.year}]}]"
        return"
        return f f"[{base"[{base_name}]_name}]"

   "

    def _ def _format_format_author_nameauthor_name(self,(self, author_str author_str: str: str) ->) -> str str:
        if:
        if "," in "," in author_str author_str:
           :
            parts = parts = [p [p.strip().strip() for p for p in author_str in author.split(",_str.split",(",", 1 1)]
           )]
            if len if len(parts(parts) ==) == 2 2:
               :
                last, first = last, first = parts parts
                first
                first_initial_initial = first = first[0[0] +] + "." if "." if first else first else ""
                ""
                return f return f"{last"{last}, {}, {first_first_initialinitial}"
        return}"
        return author_str author_str

   

    def to_dict def to(self_dict(self) ->) -> Dict[str Dict[str, Any, Any]:
       ]:
        return return {
            " {
            "source":source": self.source self.source_filename_filename,
,
            "            "doidoi": self.doi": self.doi,
           ,
            "arxiv "arxiv_id":_id": self.arxiv_id,
            "title": self.title self.arxiv_id,
            "title": self.title,
            ",
            "authors":authors": self.a self.authorsuthors,
            ",
            "journaljournal":": self.j self.journalournal,
            ",
            "year":year": self.year self.year,
           ,
            "volume "volume": self": self.volume.volume,
           ,
            "issue "issue": self": self.issue.issue,
           ,
            "pages "pages": self": self.pages.pages,
           ,
            "p "publisher":ublisher": self.p self.publisherublisher,
            ",
            "extractionextraction_method":_method": self.ext self.extraction_methodraction_method,
           ,
            "confidence "confidence": self": self.confidence.confidence,
           ,
            "citation "citation_apa_apa": self": self.format_c.format_citation("itation("apaapa"),
            ""),
            "citation_citation_doi":doi": self.format self.format_citation_citation("doi("doi"),
           "),
            "citation "citation_full":_full": self.format self.format_citation_citation("full("full"),
        }

   "),
        }

    @class @classmethodmethod
    def
    def from_dict from_dict(cls,(cls, data: data: Dict[str Dict[str, Any, Any]) ->]) -> 'BibliographicMetadata 'BibliographicMetadata':
       ':
        meta = meta = cls(data cls(data.get(".get("source",source", "unknown "unknown"))
       "))
        meta.doi = data.get("doi")
        meta.doi = data.get("doi")
        meta meta.arxiv.arxiv_id =_id = data.get data.get("arxiv("arxiv_id_id")
        meta")
        meta.title =.title = data.get data.get("title")
("title")
               meta.a meta.authors =uthors = data.get data.get("authors("authors",", [])
        [])
        meta.journal = data.get("journal")
        meta.journal = data.get("journal")
        meta.year meta.year = data = data.get(".get("yearyear")
        meta")
        meta.volume.volume = data = data.get("volume.get("volume")
        meta")
        meta.issue = data.get(".issue = data.get("issue")
        meta.pagesissue")
        meta.pages = data = data.get(".get("pagespages")
        meta")
        meta.publisher.publisher = data = data.get("p.get("publisherublisher")
       ")
        meta.ext meta.extraction_methodraction_method = data = data.get("ext.get("extraction_methodraction_method", "", "ccached")
        metaached")
        meta.confidence.confidence = data = data.get(".get("confidence",confidence", 0 0.5)
       .5)
        return meta return meta


def


def extract_ extract_metadata_frommetadata_from_pdf_pdf_text(text_text(text: str: str, filename, filename: str: str) -> BibliographicMetadata) -> BibliographicMetadata:
    meta:
    meta = = Bibli BibliographicMetadataographicMetadata(filename(filename)
    text)
    text_sample =_sample = text[: text[:1000010000]

   ]

    doi_match doi_match = Bibli = BibliographicMetadataographicMetadata.DOI.DOI_PAT_PATTERN.searchTERN.search(text_sample(text_sample)
   )
    if doi if doi_match_match:
        meta:
        meta.doi = doi_match.doi = doi_match.group(.group(1).1).lowerlower()
        meta()
        meta.confidence.confidence = max = max(meta(meta.confidence.confidence, , 0.0.99)
        meta)
        meta.extraction.extraction_method =_method = "regex "regex_doi_doi"

   "

    arxiv arxiv_match =_match = Bibliographic BibliographicMetadata.Metadata.ARXARXIVIV_P_PATTERNATTERN.search(text.search(text_sample_sample)
    if)
    if arxiv arxiv_match_match:
        meta:
        meta.arxiv.arxiv_id =_id = arxiv arxiv_match.group_match.group(1(1)
        meta.confidence = max(meta.confidence,)
        meta.confidence = max(meta.confidence,  00.85.85)

    year)

    year_m_matches =atches = Bibliographic BibliographicMetadata.YMetadata.YEAR_PEAR_PATTERNATTERN.findall.findall(text(text_sample)
   _sample for year)
   _str in for year year_m_str in year_matchesatches:
        year:
        year = int = int(year_str(year_str)
       )
        if  if 19001900 <= year <= year <=  <= 20302030:
           :
            year_pos year_pos = text = text_sample.find_sample.find(year_str(year_str)
           )
            context = context = text_sample text_sample[max([max(0,0, year_pos year_pos -  - 50):50):year_posyear_pos +  + 50].50].lowerlower()
            if any()
            if any(k(kw in context forw in context for kw in ['published kw in ['published', '', 'received',received', 'ac 'accepted',cepted', 'copyright 'copyright', '', '©']):
               ©']):
                meta.year meta.year = year = year
               
                meta.conf meta.confidence =idence = max( max(meta.confmeta.confidence,idence, 0 0.7.7)
               )
                break break

    for

 pattern in    for pattern in Bibliographic BibliographicMetadata.JMetadata.JOURNAL_POURNAL_PATTERATTERNSNS:
        journal:
        journal_match =_match = pattern.search pattern.search(text_sample(text_sample)
       )
        if journal if journal_match_match:
            journal:
            journal = journal_match.group = journal(1_match.group).strip(1).strip()
           ()
            if len if len(journal(journal) > 10 and not any) > 10 and not any(
                bad(
                bad in journal in journal.lower().lower() for bad for bad in [' in ['introductionintroduction', '', 'abstract',abstract', 'references 'references']
           ']
            ):
                ):
                meta meta.j.journal =ournal = journal journal
                meta
                meta.confidence.confidence = max = max(meta(meta.confidence, .confidence, 0.0.66)
)
                break                break

   

    vol_match vol_match = Bibli = BibliographicMetadataographicMetadata.VOL.VOLUME_PUME_PATTERNATTERN.search(text.search(text_sample_sample)
    if)
    if vol_match vol_match:
       :
        meta. meta.volume =volume = vol_match.group( vol_match.group(11)
    iss)
    iss_match =_match = BibliographicMetadata.ISSUE_PATTERN.search(text_sample)
    if BibliographicMetadata.ISSUE_PATTERN.search(text_sample)
    if iss_match:
        meta. iss_match:
        meta.issue =issue = iss_match iss_match.group(.group(11)

    author)

    author_section_section = text = text_sample[:_sample[:20002000]
   ]
    author_m author_matches =atches = Bibliographic BibliographicMetadata.AMetadata.AUTHOR_PUTHOR_PATTERNATTERN.findall(author_section.findall(author_section)
   )
    if author if author_matches_matches:
       :
        raw_a raw_authors =uthors = author_m author_matches[0atches[0]
        if]
        if ',' in ',' in raw_a raw_authorsuthors or or ' and ' and ' in ' in raw_a raw_authors.loweruthors.lower():
           ():
            separators separators = = [',', [',', ' and ' and ', '; ', '']
           ; for']
            for sep in sep in separators separators:
               :
                if sep if sep.lower().lower() in raw in raw_authors.lower_authors.lower():
                    meta():
                    meta.authors.authors = = [a.strip [a.strip() for() for a in a in re.split re.split(sep(sep, raw, raw_a_authorsuthors, flags, flags=re.I)=re.I) if a if a.strip.strip()]
                    break()]
                    break
       
        else else:
            meta:
            meta.authors.authors = = [raw_a [raw_authors.striputhors.strip()]
       ()]
        if meta if meta.authors.authors:
           :
            meta.conf meta.confidence =idence = max( max(meta.confidencemeta.conf,idence, 0.5 0.5)

   )

    title_pattern title_patterns =s = [
 [
               re.compile re.compile(r'((r'(?:^|\n?:^|\n)()([A-Z[A-Z][^][^.\n.\n]{20]{20,150,150}(?}(?:\:\.[^A.[^A-Z]-Z]|$|$))))'),
'),
        re        re.compile(r.compile(r'(?:'(?:title:title:?\s?\s*)(*)([A[A-Z-Z][^.\][^.\n]{n]{20,20,200}200}?)\.?)\.?(??(?:\n:\n|$|$)', re)', re.I),
   .I),
    ]
    for ]
    for pattern in pattern in title_pattern title_patterns:
        titles:
        title_match = pattern.search_match = pattern.search(text_sample(text_sample)
       )
        if title if title_match_match:
            title:
            title = title = title_match.group_match.group(1(1).strip).strip()
           ()
            if  if 3030 < len(title < len(title)) < 200 < 200 and not title.is and not title.isupperupper():
                meta():
                meta.title = title.title = title

                meta.confidence = max                meta.confidence = max(meta(meta.confidence.confidence, , 0.0.5555)
                break)
                break

   

    return meta return meta


def


def extract_ extract_metadata_frommetadata_from_pdf_pdf_file(p_file(pdf_pathdf_path: str, filename: str: str, filename: str)) -> -> Bibliographic BibliographicMetadataMetadata:
    meta:
    meta = Bibli = BibliographicMetadataographicMetadata(filename(filename)

    if)

    if PYPDF PYPDF2_2_AVAILABLE:
       AVAILABLE:
        try try:
            reader:
            reader = Pdf = PdfReader(pReader(pdf_pathdf_path)
           )
            pdf_info pdf_info = reader.metadata or = reader.metadata or {}
            field {}
            field_mapping_mapping = {' = {'/Title/Title': '': 'title',title', '/Author '/Author': 'authors',': 'authors', '/Creation '/CreationDate':Date': 'year 'year', '/', '/Subject':Subject': 'journal 'journal'}
           '}
            for pdf for pdf_field,_field, meta_field meta_field in field in field_mapping_mapping.items.items():
                if():
                if pdf_field pdf_field in pdf in pdf_info and_info and pdf_info pdf_info[pdf[pdf_field_field]:
                    value = str]:
                    value(pdf = str_info(pdf_info[pdf_field]).strip[pdf_field()
                   ]).strip if meta()
                   _field == if meta 'authors_field == 'authors' and' and value value:
                        meta:
                        meta.authors.authors = = [a.strip [a.strip() for() for a in a in re.split re.split(r'(r'[;[;,]',,]', value) value) if a if a.strip.strip()]
                    elif()]
 meta_field                    elif meta_field == ' == 'year'year' and value and value:
                        year_match = re:
                        year_match = re.search(r.search(r'(?:'(?:D:D:)?(\)?(\d{d{4})4})', value', value)
                       )
                        if year if year_match_match:
                            meta:
                            meta.year =.year = int(year int(year_match.group_match.group(1(1))
                   ))
                    else else:
                        set:
                        setattr(attr(metameta,, meta_field meta_field, value, value)
           )
            if meta if meta.title or.title or meta.a meta.authorsuthors:
                meta:
                meta.confidence.confidence =  = 0.0.77
                meta
                meta.extraction.extraction_method =_method = "pdf_metadata "pdf"
       _metadata except Exception"
        as e except Exception as e:
            st.w:
            st.warning(farning(f"Could"Could not read not read PDF metadata PDF metadata: {: {e}e}")

    try")

    try:
        loader:
        loader = Py = PyPDFLoaderPDFLoader(pdf(pdf_path_path)
        pages)
        pages = loader = loader.load.load()
        text()
        text_sample =_sample = "\n "\n".join".join([p([p.page_content.page_content for p for p in pages in pages[:3[:3]])
       ]])
        text_ text_meta =meta = extract_ extract_metadata_frommetadata_from_pdf_pdf_text(text_text(text_sample,_sample, filename filename)

        for)

        for field in field ['doi in ['doi', '', 'arxiv_idarxiv_id', '', 'title',title', 'journal', ' 'journal', 'year',year', 'volume 'volume', '', 'issue']issue']:
           :
            text_val text_val = get = getattr(textattr(text_meta_meta, field, field)
           )
            current_val current_val = get = getattr(attr(meta,meta, field field)
            if)
            if text_val text_val and (not current and (_val ornot current text__val ormeta.conf text_meta.confidence >idence > meta.conf meta.confidenceidence):
                set):
                setattr(attr(meta, fieldmeta,, text field,_val text_val)
       )
        if text if text_meta_meta.authors.authors and (not meta and (.authorsnot meta or text.authors_meta or text.confidence_meta > meta.confidence.confidence > meta.confidence):
           ):
            meta.a meta.authors =uthors = text_ text_meta.ameta.authorsuthors
        if
        if text_ text_meta.confmeta.confidence >idence > meta.conf meta.confidenceidence:
            meta:
            meta.confidence.confidence = text = text_meta_meta.confidence.confidence
           
            meta.ext meta.extraction_methodraction_method = text = text_meta_meta.extraction.extraction_method_method
    except Exception as e
    except Exception as e:
        st:
        st.warning.warning(f(f""Text extractionText extraction for metadata for metadata failed: failed: {e {e}}")

    if")

    if PDF2 PDF2DOI_DOI_AVAILABLEAVAILABLE and not and not meta.doi meta.doi:
       :
        try try:
            result:
            result = pdf = pdf2doi2doi.pdf2.pdf2doi(pdoi(pdf_path)
            if isinstance(result,df_path)
            if isinstance(result, list) and result:
                list) and result result =:
                result = result result[0[0]
            if]
            if result and result and result.get result.get('identifier('identifier') and') and result.get result.get('identifier('identifier_type')_type') == ' == 'doi':
doi':
                meta.doi =                meta result['.doi = result['identifieridentifier']
                meta']
                meta.confidence.confidence =  = 0.0.9595
                meta.ext
                metaraction.extraction_method =_method = "pdf "pdf2doi2doi"
               "
                if result.get(' if result.get('validation_infovalidation_info'):
'):
                    bibtex                    = result bibtex['validation = result_info['validation']
                    if_info 'title']
                    if 'title' in' in bibtex bibtex and not and not meta.title meta.title:
                       :
                        meta.title meta.title = bib = bibtex.gettex.get('title')
('title')
                                       if ' if 'author'author' in bib in bibtex andtex and not meta.authors not meta.authors:
                       :
                        meta.a meta.authors =uthors = [a [a.strip().strip() for a for a in bib in bibtex['tex['author'].author'].split('split(' and ' and ')]
                   )]
                    if ' if 'year'year' in bib in bibtex andtex and not meta not meta.year.year:
                       :
                        try try:
                           :
                            meta.year meta.year = int = int(bib(bibtex['tex['yearyear'])
                        except'])
                        except:
                           :
                            pass pass
        except
        except Exception as Exception as e e:
            st:
            st.warning.warning(f"(f"pdf2pdf2doi lookup failed: {e}")

    if CROSSREF_AVAILABLE and meta.doi and not meta.journal:
        try:
            crdoi lookup failed: {e}")

    if CROSSREF_AVAILABLE and meta.doi and not meta.journal:
        try:
            cr = CrossrefAPI()
            work = cr = CrossrefAPI()
            work = cr.works(.ids=metaworks(ids=meta.doi.doi)
            if)
            if work and work and work.get work.get('message('message'):
                msg = work['message'):
                msg = work['message']
                if']
                if not meta not meta.title and.title and msg.get msg.get('title('title'):
                   '):
                    meta.title meta.title = = msg msg['title['title'][0'][0] if] if isinstance isinstance(msg(msg['title['title'], list'], list) else) else msg[' msg['titletitle']
                if']
                if not meta not meta.authors.authors and msg and msg.get('.get('authorauthor'):
                    meta'):
                    meta.authors.authors = = [
                        f [
                        f"{a"{a.get('.get('family', 'family',') '')} {} {a.get('givena.get('given', '')}".', '')}".stripstrip()
()
                        for                        for a in a in msg[' msg['authorauthor']
                   ']
                    ]
                if ]
                if not meta not meta.journal.journal and msg and msg.get('.get('container-titlecontainer-title'):
'):
                                       meta.j meta.journal =ournal = msg[' msg['container-titlecontainer-title'][0'][0] if] if isinstance(msg isinstance(msg['container['container-title'],-title'], list) else list) msg else msg['container-title['container-title']
                if not meta']
                if.year and not meta.year and msg.get msg.get('published('published-print-print') and') and msg[' msg['published-published-print'].print'].get('get('date-pdate-partsarts'):
                    meta.year ='):
                    meta.year = msg[' msg['published-published-printprint']['']['date-pdate-partsarts'][0'][0][0][0]
                meta]
                meta.confidence.confidence =  = 0.0.98
                meta98
                meta.extraction.extraction_method =_method = "cross "crossref_apiref_api"
       "
        except Exception except Exception as e as e:
           :
            st.w st.warning(farning(f"Crossref"Crossref API lookup API lookup failed: failed: {e {e}}")

    return")

    return meta meta


def extract


def extract_metadata_metadata_from_text_from_text_file(text_file(text: str: str, filename, filename: str) -> Bibliographic: str) -> BibliographicMetadataMetadata:
    return:
    return extract_ extract_metadata_frommetadata_from_pdf_pdf_text(text_text(text, filename, filename)


class)


class MetadataCache MetadataCache:
   :
    def __ def __init__(init__(selfself):
        self):
        self._cache._cache: Dict: Dict[str,[str, Bibliographic BibliographicMetadata]Metadata] = = {}
        self {}
        self._file._file_has_hashes:hes: Dict[str Dict[str, str, str] =] = {}

    {}

    def get def get(self,(self, filename: filename: str, str, file_hash file_hash: str: str = None = None) ->) -> Optional Optional[Bibliographic[BibliographicMetadata]:
        ifMetadata]:
        if filename in filename in self._ self._cachecache:
            if:
            if file_hash file_hash is None is None or self or self._file._file_has_hashes.gethes.get(filename)(filename) == file == file_hash_hash:
:
                return                self._ return self._cache[cache[filenamefilename]
       ]
        return return None None

    def

    def set(self set(self, filename, filename: str: str, metadata, metadata: Bibli: BibliographicMetadataographicMetadata, file, file_hash:_hash: str = str = None None):
        self):
        self._cache._cache[filename[filename] =] = metadata
        if metadata file_hash
        if:
            file_hash self._:
            self._file_file_hasheshashes[filename[filename] =] = file_hash file_hash

   

    def clear def clear(self(self):
        self):
        self._cache._cache.clear.clear()
        self()
        self._file._file_has_hashes.clearhes.clear()


metadata()


metadata_cache =_cache = MetadataCache MetadataCache()


()


defdef compute_file compute_file_hash(file_hash(filepath:path: str) str) -> str -> str:
   :
    try try:
        with:
        with open(file open(filepath,path, 'rb 'rb') as') as f f:
            return:
            return has hashlhlib.mdib.md5(f5(f.read())..read()).hexdighexdigestest()
    except()
    except:
       :
        return " return ""


#"


# ================================= =========================================================
#
# REASON REASONING:ING: SCIENT SCIENTIFIC ENTIFIC ENTITY EXITY EXTRACTION
#TRACTION =================================
# =============================================

class============ ScientificEntity

class:
    ScientificEntity:
 def    def __init __init__(self, text__(self, text: str: str, label, label: str: str, value, value: Optional: Optional[float[float], unit], unit: Optional: Optional[str[str],
                ],
                 doc doc_source:_source: str, str, chunk chunk_id: int_id, context: int: str, context: str, confidence, confidence: float: float =  = 1.1.00):
        self.text = text
        self):
        self.text = text
        self.label =.label = label label
        self
        self.value.value = = value
        self value
        self.unit.unit = = unit unit
       
        self.doc self.doc_source =_source = doc_source doc_source
       
        self.ch self.chunk_id = chunk_id
        self.context = context
        self.confidence = confidenceunk_id = chunk_id
        self.context = context
        self.confidence = confidence
        self.n
        self.normalized = self._normalormalized = self._normalizeize()

    def()

    def _normal _normalize(selfize(self) ->) -> str str:
        text:
        text = self = self.text.lower.text.lower().strip().strip()
       ()
        for canonical for canonical, ali, aliases inases in MATERIAL_ MATERIAL_ALIASALIASES.itemsES.items():
           ():
            if any if any(alias(alias in text in text for alias for alias in ali in aliasesases):
                return):
                canonical return
        for canonical canonical,
        for canonical, aliases aliases in METHOD_ALIAS in METHOD_ALIASESES.items.items():
            if():
            if any( any(alias in textalias in for text for alias in alias in aliases):
                aliases return canonical):
               
        return canonical text =
        re.sub text =(r'\ re.subs+(r'\', '',s+', '', text text)
        return)
        return text text

    def

    def to_dict to_dict(self)(self) -> Dict -> Dict[str,[str, Any Any]:
        return]:
        return {
            {
            "text "text": self": self.text,.text, "label "label": self": self.label,.label, "value "value": self": self.value,.value, "unit "unit": self": self.unit.unit,
           ,
            "doc "doc_source":_source": self.doc self.doc_source,_source, "ch "chunk_idunk_id": self": self.ch.chunkunk_id_id,
            "normalized,
            "normalized": self": self.normal.normalized,ized, "confidence "confidence": self": self.confidence.confidence
       
        }


class }


class ScientificClaim ScientificClaim:
   :
    def __ def __init__(init__(self,self, claim_text claim_text: str: str, subject:, subject: str str, predicate, predicate: str: str, object, object_val:_val: str str,
                 doc,
                 doc_source:_source: str, str, chunk_id chunk_id: int: int, confidence, confidence: float: float):
       ):
        self. self.claim_textclaim_text = = claim claim_text_text
        self
        self.subject =.subject = subject subject
        self
        self.pred.predicate =icate = predicate predicate
        self
        self.object_val.object_val = object = object_val_val
        self
        self.doc_source.doc_source = doc = doc_source_source
        self
        self.chunk.chunk_id =_id = chunk_id chunk_id
       
        self.conf self.confidence =idence = confidence confidence
        self
        self.supporting.supporting: List: List[Tuple[Tuple[str,[str, int]] int]] = []
        self = []
        self.cont.contradictradicting:ing: List List[Tuple[str[Tuple[str,, int int]] =]] = []

    []

    def to def to_dict(self_dict(self) ->) -> Dict[str Dict[str, Any, Any]:
]:
        return        {
            " returnclaim": {
            " self.claim":claim_text self., "claim_textsubject":, " self.subjectsubject": self.subject, ", "predicatepredicate": self": self.pred.predicateicate,
            ",
            "object":object": self.object self.object_val,_val, "source "source": self": self.doc_source, ".doc_source, "confidence":confidence": self.conf self.confidenceidence,
            ",
            "supportingsupporting_count":_count": len(self len(self.support.supporting), "ing), "contradcontradicting_counticting":_count": len(self.cont len(self.contradictradictinging)
        }


class Cross)
        }


class CrossDocumentKnowledgeDocumentKnowledgeGraphGraph:
    def:
    def __init __init__(__(selfself):
       ):
        self. self.entities:entities: Dict[str Dict[str, List, List[Scientific[ScientificEntity]]Entity]] = defaultdict = defaultdict(list(list)
        self)
        self.claims.claims: List: List[Scientific[ScientificClaim]Claim] = = []
        self []
        self.doc.documentsuments: Dict: Dict[str,[str, Dict[str Dict[str, Any, Any]] =]] = {}
        {}
        self.entity_index: self.entity_index: Dict[str Dict[str, Set, Set[str]][str]] = defaultdict = defaultdict(set(set)

    def)

    def add_d add_document(selfocument(self, doc, doc_id:_id: str, str, chunks: chunks: List List[Document],[Document], bib_ bib_meta:meta: Bibliographic BibliographicMetadataMetadata):
        self):
        self.documents.documents[doc[doc_id]_id] = = {
            " {
            "bib_bib_meta": bib_meta.tometa": bib_meta.to_dict_dict(),
            "(),
            "chunkchunk_count":_count": len(ch len(chunksunks),
            "),
            "topicstopics": set": set()
()
        }

               }

        for i for i, chunk, chunk in enumerate in enumerate(chunks(chunks):
):
                       entities = entities = self._ self._extractextract_entities_entities_from_ch_from_chunk(chunk(chunk,unk, i i)
            for)
            for ent in ent in entities entities:
                self:
                self.entities.entities[ent[ent.normal.normalized].ized].append(append(entent)
                self)
                self.entity_index.entity_index[ent.n[entormal.normalized].add(dized].oc_idadd(d)
               oc_id)
                self.doc self.documentsuments[doc_id[doc_id]["top]["topics"].ics"].add(add(ent.labelent.label)

           )

            claims = claims = self._extract self._extract_claims_claims_from_ch_from_chunk(chunk(chunk,unk, i i)
            for)
            for claim in claim in claims claims:
                self:
                self.claims.claims.append(.append(claimclaim)

    def)

    def _ext _extractract__entities_fromentities_from_chunk(self,_chunk(self, chunk: chunk: Document, Document, chunk_id chunk_id: int: int) ->) -> List List[ScientificEntity]:
       [ScientificEntity]:
        text text = = chunk.page chunk.page_content_content
        doc
        doc = chunk = chunk.metadata.metadata.get(".get("source", "unknownsource",")
        "unknown entities =")
        entities = []

        []

        for param for param_name,_name, pattern in pattern in QUANT QUANTITY_PITY_PATTERATTERNS.itemsNS.items():
           ():
            for match for match in pattern in pattern.finditer.finditer(text(text):
                val):
                val_str = match_str =.group match.group(1(1)
               )
                try try:
                    val:
                    val = float = float(val_str(val_str)
               )
                except except:
                    val:
                    val = None = None
                unit_match
                unit_match = re.search(r = re.search(r'(nm'(nm|µm|µm|um|um|fs|fs|ps|ps||nsns|J|J/cm²/cm²|J|J/cm2/cm2|kHz|kHz|MHz|MHz|W|W|m|mW|W|mJmJ|µ|µJ|J|uJuJ)', match)', match.group(0.group(),0), re.I re.I)
               )
                unit = unit = unit_match unit_match.group(.group(11) if unit) if unit_match else_match else None None

                start

                start = max = max(0(0, match, match.start().start() - 100 - )
                end100)
                end = min = min(len(text(len(text), match), match.end().end() +  + 100100)
                context)
                context = text = text[start:[start:end].end].replace('\replace('\n', ' 'n',)

                ' ')

                ent = ent = ScientificEntity ScientificEntity(
                   (
                    text=m text=match.groupatch.group(0(0), label), label=param=param_name,_name, value= value=val,val, unit= unit=unitunit,
                    doc,
                    doc_source_source==doc,doc, chunk_id=chunk_id, context chunk_id=chunk=context_id, context=context,
                   ,
                    confidence= confidence=0.0.8585
               
                )
                entities )
                entities.append(.append(entent)

        text)

        text_lower_lower = text = text.lower.lower()
        for canonical()
        for, canonical, aliases aliases in MATERIAL in MATERIAL_AL_ALIASESIASES.items.items():
            for():
            for alias in alias in aliases aliases:
               :
                for match for match in re in re.finditer.finditer(r'\(r'\b'b' + re + re.escape(alias) + r'\.escape(alias) + r'\b',b', text_l text_lowerower):
                    start):
                    start = max = max((00, match, match.start().start() -  - 8080)
                    end)
                    end = min(len(text = min(len(text), match), match.end() + 80.end() + 80)
                    context)
                    context = text = text[start:[start:endend]
                    ent]
                    ent = Scientific = ScientificEntityEntity(
                        text(
                        text=alias=alias, label, label="MAT="MATERIAL",ERIAL", value=None value=None, unit, unit=None=None,
                        doc,
                        doc_source=_source=doc,doc, chunk_id chunk_id=ch=chunk_idunk_id, context, context=context=context,
                       ,
                        confidence= confidence=0.0.99
                   
                    )
                    entities )
                    entities.append(.append(entent)

        for)

        for canonical, canonical, aliases aliases in METHOD in METHOD__ALALIASESIASES.items.items():
            for():
            for alias in alias in aliases:
 aliases               :
                for match for match in re in re.finditer.finditer(r'\(r'\b'b' + re + re.escape.escape(alias(alias) +) + r'\ r'\b',b', text_l text_lowerower):
                    start = max(0):
                    start = max(0, match, match.start().start() - 80 - 80)
                    end)
                    end = min = min(len(text(len(text), match), match.end().end() +  + 8080)
                    context)
                    context = text = text[start[start::endend]
                    ent]
                    ent = Scientific = ScientificEntityEntity(
                        text(
                        text=alias=alias, label, label="METHOD="METHOD", value", value=None,=None, unit=None unit=None,
                       ,
                        doc_source doc_source=doc, chunk=doc, chunk_id=_id=chunkchunk_id,_id, context= context=contextcontext,
                        confidence,
                        confidence=0=0.9.9
                   
                    )
                    )
                    entities.append entities.append(ent(ent)

       )

        return entities return entities

   

    def _ def _extractextract_claims_claims_from_ch_from_chunk(selfunk(self, chunk, chunk: Document: Document, chunk, chunk_id:_id: int) int) -> List -> List[Scientific[ScientificClaimClaim]:
        text]:
        text = chunk = chunk.page_content.page_content
       
        doc = doc = chunk.m chunk.metadata.getetadata.get("source("source", "unknown", "unknown")
        claims =")
        claims = []

        claim []

        claim_patterns_patterns = [
            = [
            ( (r'(r'(?:ablation\s*?:ablation\s*threshold|threshold|threshold\threshold\s*s*fluence)\fluence)\s*(s*(?:of?:of|for|for)\s)\s+([a-z\s+([a-z\s]+?]+?)\s)\s+(?:+(?:is|is|was|was|were|were|are|are|≈|≈|~|~|about)\about)\s+s+(\d(\d+\.+\.?\d?\d*\s*\s**[A-Z[A-Za-za-z/²/²]+)',]+)', 'has 'has_ab_ablation_thlation_thresholdreshold'),
            ('),
            (r'r'([a([a-z\-z\s]+s]+?)\?)\s+(s+(?:ex?:exhibitshibits|sh|shows|ows|displaysdisplays|forms|forms|produ|producesces)\)\s+s+([a([a-z\-z\s]+s]+?(?:?(?:ripplesripples|L|LIPSSIPSS|structures|structures|m|morphologyorphology))',))', 'ex 'exhibitshibits_morph_morphologyology'),
            ('),
            (r'(r'(?:period?:periodicity|icity|period|period|spacingspacing)\s)\s*(?:*(?:of|of|for)\for)\s+s+([a([a-z\-z\ss]+]+?)\?)\s+(s+(?:is?:is|was||was≈|≈|~|~)\s)\s+(\+(\d+d+\.?\\.?\d*\d*\s*(s*(?:nm?:nm|µm|µm|um|um))',))', 'has 'has_period_periodicityicity'),
            ('),
            (r'(r'(?:rough?:roughness|ness|Ra)\Ra)\s*(s*(?:?:ofof|for|for)\s)\s+([a-z\s]+?)\s+([a-z\s]+?)\s+(?:+(?:is|is|was|was|≈|≈|~)\~)\s+s+(\d(\d+\.+\.?\d?\d*\s*\s*(?:*(?:nm|nm|µm|µm|um))um))', '', 'has_has_roughnessroughness'),
       '),
        ]

        for ]

        pattern, predicate for pattern in claim, predicate_patterns in claim:
           _patterns:
            for match in re for match.finditer in re.finditer(pattern,(pattern, text, re.I text, re.I):
               ):
                subject = subject = match.group match.group(1(1).strip).strip()
               ()
                obj = obj = match.group match.group(2(2).strip).strip()
               ()
                start = start = max( max(0,0, match.start match.start() -() - 120 120)
               )
                end = end = min(len min(len(text),(text), match.end match.end() +() + 120 120)
               )
                context = context = text[start text[start:end:end]

               ]

                claim = claim = ScientificClaim ScientificClaim(
                   (
                    claim_text claim_text=context=context, subject, subject=subject=subject, predicate, predicate=pred=predicateicate,
                    object,
                    object_val=_val=obj,obj, doc_source doc_source=doc=doc, chunk_id, chunk=_id=chchunkunk_id_id,
                    confidence,
                    confidence=0=0.7.7
               
                )
                )
                claims.append claims.append(claim(claim)

        return claims)

        return claims

   

    def find def find_cons_consensus(selfensus(self, entity, entity_normalized_normalized: str: str) ->) -> Optional Optional[Dict[str[Dict[str,, Any Any]]:
       ]]:
        ents = self ents = self.entities.entities.get(entity.get(entity_normalized_normalized,, [])
        [])
        if len if len(ents) < 2(ents) < 2:
           :
            return None return None

       

        by_d by_doc =oc = defaultdict(list defaultdict(list)
       )
        for for e e in ents in ents:
           :
            by_doc[e by_d.doc_sourceoc[e].append.doc_source(e].append)

        if(e)

        len( if len(by_dby_doc)oc) <  < 22:
            return:
            return None None

        values

        values = = [e.value [e.value for e for e in ents in ents if e if e.value is.value is not None not None]
       ]
        if not if not values values:
            return None:
            return None

        return

        return {
            {
            "entity "entity": entity":_normalized entity_normalized,
           ,
            "doc "doc_count":_count": len( len(by_dby_dococ),
            "),
            "value_countvalue_count": len": len(values(values),
            "),
            "mean":mean": np.mean np.mean(values(values),
            "std),
            "": np.stdstd":(values np.std),
            "(valuesmin":),
            " np.minmin": np.min(values),
            "(values),
            "max":max": np.max np.max(values(values),
            "),
            "unit":unit": ents ents[0].[0].unitunit,
            ",
            "sourcessources": list": list(by(by_d_dococ.keys())
       .keys }

    def())
        }

    def find_contrad find_contradictions(selfictions(self, entity, entity_normalized_normalized: str: str, threshold, threshold_factor:_factor: float = float = 2 2.0.0) ->) -> List List[Dict[str[Dict[str, Any, Any]]]]:
        ents:
        ents = self = self.entities.entities.get(entity.get(entity_normalized_normalized,, [])
        by_doc = [])
        by_doc = defaultdict(list defaultdict(list)
       )
        for e for e in ents in ents:
           :
            if e if e.value is.value is not None not None:
               :
                by_doc[e.doc by_doc[e.doc_source_source].append].append(e.value(e.value)

       )

        contradictions = contradictions = []
        docs []
        = docs = list(by_d list(by_doc.keys())
       oc.keys())
        for i for i in in range range(len(len(docs(docs)):
            for)):
            for j in j in range(i range(i + 1, + 1, len(d len(docs)):
                valsocs)):
                vals_i = by_d_i = by_doc[doc[docs[iocs[i]]
               ]]
                vals_j vals_j = by = by_doc_doc[docs[docs[j[j]]
                mean]]
                mean_i,_i, mean_j mean_j = np = np.mean(v.mean(vals_ials_i), np), np.mean(v.mean(vals_jals_j)
               )
                if mean if mean_i >_i > 0 0 and mean and mean_j >_j > 0 0:
                   :
                    ratio = ratio = max( max(mean_imean_i, mean, mean_j)_j) / min / min(mean(mean_i,_i, mean_j mean_j)
                   )
                    if ratio if ratio > threshold > threshold_factor_factor:
                        contradictions:
                        contradictions.append.append({
                            "({
                            "entity":entity": entity_normal entity_normalizedized,
                            ",
                            "doc_a":doc_a": docs[i], "mean docs[i], "mean_a": mean_i_a": mean_i,
                           ,
                            "doc "doc_b":_b": docs[j docs[j], "], "mean_bmean_b": mean": mean_j_j,
                            ",
                            "ratio": ratioratio": ratio,
                            ",
                            "severityseverity": "": "high"high" if if ratio ratio >  > 55 else else "mod "moderateerate"
                       "
                        })
        return })
        return contradictions contradictions

    def

    def get_ get_related_chrelated_chunks(selfunks(self, query, query_entities_entities: List[str],: List[str], chunks: chunks: List List[Document[Document],
                          depth],
                          depth: int: int =  = 2)2) -> List -> List[Tuple[Tuple[Document[Document, float, float, str, str]]]]:
:
        related        related_docs =_docs set = set()
       ()
        for ent for ent_norm_norm in query in query_entities_entities:
           :
            related_d related_docs.updateocs.update(self.entity(self.entity_index.get_index.get(ent(ent_norm, set()))

        scored = []
        for chunk in chunks_norm, set()))

        scored = []
        for chunk in chunks:
            doc:
            doc = chunk = chunk.metadata.metadata.get(".get("source",source", "unknown "unknown")
           ")
            score = score = 0 0.0.0
           
            reason = reason = "sem "semantic"

            chunkantic"

            chunk_text =_text = chunk chunk.page.page_content.lower_content.lower()
           ()
            for ent_norm for ent_norm in query in query_entities_entities:
               :
                if ent if ent_norm_norm in chunk in chunk_text_text:
                    score:
                    score += += 0. 0.33

            if

            if doc in doc in related_docs:
 related_docs                score:
                score += 0 += 0..22
                reason
                reason = " = "cross-dcross-doc-linkoc-link"

           "

            for claim for claim in self in self.claims.claims:
               :
                if claim if claim.doc_source == doc.doc_source == doc and claim and claim.chunk.chunk_id ==_id == chunk.metadata chunk.metadata.get("chunk.get_index",("chunk_index", -1 -1):
                    if):
                    if any( any(ent inent in claim.subject claim.subject.lower().lower() or ent or ent in claim in claim.object_val.object_val.lower.lower()
                           for()
                           for ent in ent in query_ query_entitiesentities):
                        score):
                        score +=  += 0.0.25
                        reason = "25
                        reason = "claim-evclaim-evidenceidence"

            if"

            if score > score > 0 0:
               :
                scored.append scored.append((ch((chunk,unk, score, score, reason reason))

))

        scored       .sort(key scored.sort(key=lambda x=lambda x: x: x[1[1], reverse], reverse=True=True)
        return)
        return scored scored

    def

    def get get_k_knowledgenowledge_summary_summary(self)(self) -> Dict -> Dict[str,[str, Any Any]:
        return]:
        return {
            "total_entities {
            "total_entities": sum": sum(len(v(len(v) for) for v in v in self. self.entities.valuesentities.values()),
            "unique_entities": len(self.entities),
            "total_claims": len(self.claims),
            "document()),
            "unique_entities": len(self.entities),
            "total_claims": len(self.claims),
            "document_count":_count": len(self len(self.documents.documents),
           ),
            "top "top_entities": Counter_entities": Counter([e([e.normal.normalized forized for ents in ents in self. self.entities.valuesentities.values() for() for e in e in ents]). ents]).most_most_common(common(10),
            "10),
            "consensusconsensus_topics_topics":": [k for k, [k for k, v in v in self. self.entities.itemsentities.items() if len(self.entity_index.get(k() if len(self.entity_index.get(k, set, set())) >())) > 1 1]
        }


#]
        }


# ================================= =========================================================
#
# SEMANT SEMANTIC CHIC CHUNKINGUNKING WITH WITH STRUCTURE STRUCTURE AWAR AWARENESSENESS
#
# ================================= =========================================================

def

def detect_s detect_scientific_scientific_sections(textections(text: str: str) ->) -> List List[Tuple[str[Tuple[str, str, str]]]]:
    section:
    section_patterns_patterns = = [
        ( [
        (r'(?:^r'(?:^|\n|\n)\s)\s*Abstract*Abstract\s\s*\n*\n', '', 'ABSTRACTABSTRACT'),
        ('),
        (r'(r'(?:^?:^|\n)\s|\n)\s*1*1\.\\.\s*s*Introduction\Introduction\s*\s*\n',n', 'INTRODUCTION 'INTRODUCTION'),
       '),
        (r (r'(?:'(?:^|\^|\n)\n)\s*(s*(?:2?:2\.)?\.)?\s\s*Experimental*Experimental\s\s*(?:*(?:Setup|Setup|Methods|Methods|Details)?Details)?\s\s*\n',*\n 'METHODS', ''),
       METHODS (r'),
        (r'(?:^|\n)\s*(?:3'(?:^|\n)\s*(?:3\.)?\.)?\s\s*Results*Results\s\s*(?:*(?:and\s*and\s*Discussion)?Discussion)?\s\s*\n*\n', '', 'RESULTSRESULTS'),
        ('),
        (r'(r'(?:^?:^|\n|\n)\s*()\s*(?:?:4\.4\.)?\)?\s*s*Discussion\Discussion\s*\s*\n', 'Dn', 'DISCUSSION'),
        (r'(?:ISCUSSION'),
        (r'(?:^|\^|\n)\n)\s*s*Conclusion',Conclusion', 'CONCLUS 'CONCLUSIONION'),
   '),
    ]

    boundaries ]

    boundaries = = []
    for []
    for pattern, pattern, name in name in section_pattern section_patternss:
        for:
        for match in match in re.finditer(pattern, text re.finditer(pattern, text, re, re.I.I):
            boundaries):
            boundaries.append((.append((match.startmatch.start(), name(), name))

   ))

    if not if not boundaries boundaries:
        return:
        return [(" [("BODYBODY", text", text)]

   )]

    boundaries.sort boundaries.sort()

   ()

    sections = sections = []
    []
    for for i i, (, (pos,pos, name) name) in enumerate(bound in enumerate(boundariesaries):
        end):
        end = boundaries = boundaries[i +[i + 1 1][0][0] if i +] if i + 1 1 < len < len(bound(boundaries)aries) else len else len(text(text)
        section)
        section_text =_text = text[pos text[pos:end:end].strip].strip()
       ()
        if len if len(section(section_text)_text) >  > 5050:
            sections:
            sections.append((.append((name,name, section_text))

    return sections section_text))

    return sections if sections if sections else else [("B [("BODY",ODY", text text)]


def)]


def semantic_ch semantic_chunk_dunk_document(pocument(pages:ages: List List[Document],[Document], filename: filename: str str)) -> List[Document -> List[Document]:
   ]:
    all_text all_text = "\ = "\n\nn\n".join".join([p([p.page_content.page_content for p for p in pages in pages])
    sections])
    sections = detect = detect_scientific_scientific_sections_sections(all_text(all_text)

   )

    chunks = chunks = []
    []
    for section for section_name,_name, section_text section_text in sections in sections:
       :
        if section if section_name in_name in ['ABSTRACT ['ABSTRACT', '', 'CONCLUSION']CONCLUSION:
            chunk_size, overlap = 400']:
            chunk_size, overlap = 400, , 5050
        elif
        elif section_name section_name == ' == 'METHODSMETHODS':
           ':
            chunk_size chunk_size, overlap, overlap =  = 600,600, 100 100
       
        else else:
            chunk:
            chunk_size,_size, overlap = overlap = LASER LASER_DOM_DOMAIN_CONFIGAIN_CONFIG["ch["chunk_sizeunk_size"], LAS"], LASER_DOMER_DOMAINAIN_CONFIG["_CONFIG["chunk_overlapchunk_overlap"]

       "]

        splitter splitter = Rec = RecursiveCharacterursiveCharacterTextSplitTextSplitterter(
            chunk(
            chunk_size=_size=chunkchunk_size_size,
            chunk,
            chunk_overlap_overlap=over=overlaplap,
            separ,
            separators=["ators=["\n\n\n",\n", "\n "\n", ".", ". ", "; ", "; ", ", ", ", " "],
            length],
            length_function=_function=lenlen
       
        )

        section )

        section_chunks_chunks = split = splitter.createter.create_documents_documents([section([section_text_text])
        for])
        for i, i, chunk in chunk in enumerate(s enumerate(section_chection_chunksunks):
            chunk):
            chunk.metadata.metadata.update.update({
                "({
                "source":source": filename,
                " filename,
                "section":section": section_name section_name,
               ,
                "ch "chunk_indexunk_index": len": len(chunks(chunks) +) + i i,
                ",
                "section_chsection_chunk_indexunk_index": i": i,
           ,
            })
        })
        chunks.extend(section chunks.extend(section_chunks_chunks)

   )

    for i for i, chunk, chunk in enumerate in enumerate(chunks(chunks):
       ):
        chunk.m chunk.metadata["etadata["chunkchunk_index"]_index"] = i = i
       
        chunk chunk.m.metadata["etadata["total_chtotal_chunks"]unks"] = len = len(chunks(chunks)

   )

    return return chunks chunks


#


# ================================= =========================================================
#
# SESSION SESSION STATE INIT STATE INITIALIZATIONIALIZATION
#
# ================================= =========================================================

def

def initialize_session initialize_session_state():
_state():
    defaults = {
        "processed_files    defaults = {
        "processed_files": set": set(),
       (),
        "vector "vectorstore":store": None None,
        ",
        "all_chall_chunks":unks": [],
        [],
        "messages "messages":": [],
        " [],
        "llm_model_choice": None,
llm_model_choice": None,
        "ll        "llmm_tokenizer_tokenizer": None": None,
       ,
        "ll "llm_modelm_model": None": None,
       ,
        "llm "ll_backm_backend":end": None None,
        ",
        "llmllm_device_or_device_or_host":_host": None,
        " Nonellm,
        "_backendllm_backend_type":_type": None None,
        ",
        "embeddembeddings":ings": None None,
        ",
        "processing_complete":processing_complete": False False,
        ",
        "laserlaser_domain_domain_boost_boost": True": True,
       ,
        "show "show_sources_sources": True": True,
        "citation,
        "citation_style_style": "": "apaapa",
        "",
        "max_max_retrievedretrieved_chunks_chunks": ": 6,
        "6,
        "useuse__4bit4bit_quant_quantization":ization": True True,
        ",
        "ollama_host": "httpollama_host": "http://localhost://localhost:114:1143434",
        "",
        "metadata_cachemetadata_cache": metadata": metadata_cache_cache,
        ",
        "knowledge_graphknowledge_graph": None": None,
       ,
        " "reasoning_mode": True,
       reasoning_mode": True,
        "show "show_reason_reasoning_ing_chain":chain": True True,
        ",
        "cross_dcross_doc_oc_consconsensusensus": True": True,
   ,
    }
    }
    for key for key, value, value in defaults in defaults.items.items():
        if():
        if key not key not in st in st.session_state.session_state:
           :
            st.session st.session_state[key_state[key] =] = value value


# =========================================


# =================================================
# UT
# UTILITY FUNILITY FUNCTIONSCTIONS
#
# ================================= =============================================

def============

def is_ is_ollamaollama_model(model_model(model_key:_key: str) str) -> bool -> bool:
   :
    return model return model_key.startswith("_key.startswith("ollamaollama:"):") or model or model_key.start_key.startswithswith("[Oll("[Ollama]ama]")

def")

def extract_ extract_ollamaollama_tag(model_tag(model_key:_key: str) str) -> str -> str:
   :
    if model_key.start if modelswith("_key.startollamaswith(":"ollama):
        return:" model_key):
        return model_key.replace(".replace("ollama:", "",ollama 1:", "", 1)
   )
    elif model elif model_key.start_key.startswithswith("[Oll("[Ollama]"ama]"):
        match = re.search):
        match = re.search(r'(r'\]\\]\s*s*([^\([^\s(s(]+)',]+)', model_key)
        model_key)
        if match if match:
           :
            return match return match.group(.group(11)
    return)
    return model_key model_key

def get_h

def get_hf_ref_repo_idpo_id(model_key(model_key: str: str) ->) -> str str:
    if:
    if ":" ":" in model in model_key and_key and not model not model_key.start_key.startswith("swith("httphttp"):
        parts"):
        parts = model = model_key.split_key.split(":",(":", 1 1)
       )
        if len if len(parts(parts) ==) == 2 2 and "/ and "/" in" in parts parts[1[1]:
            return]:
            return parts parts[1].[1].stripstrip()
()
    return    return model_key model_key

def

def get_ get_available_gavailable_gpu_mpu_memory()emory() -> Optional -> Optional[float[float]:
   ]:
    if not if not torch.c torch.cuda.isuda.is_available_available():
       ():
        return None return None
   
    try try:
        total:
        total_memory_memory = torch = torch.cuda.cuda.get_device_properties.get_device_properties(0(0).total).total_memory_memory / ( / (10241024 **  ** 33)
        reserved)
        reserved = torch = torch.cuda.cuda.memory.memory_reserved_reserved(0(0) /) / (102 (1024 **4 ** 3 3)
       )
        return total return total_memory_memory - reserved - reserved
   
    except except:
        return:
        return None None

def estimate

def estimate_model_m_model_memory(modelemory(model_key:_key: str, str, use_ use_4bit4bit: bool: bool = False = False) ->) -> Dict[str Dict[str, any, any]:
   ]:
    repo_id = repo_id = get_hf_repo_id(model get_hf_repo_id(model_key)_key) if not if not is_ is_ollamaollama_model(model_model(model_key)_key) else model else model_key_key
    return
    return MODEL_M MODEL_MEMORYEMORY_ESTIMATES_ESTIMATES.get(re.get(repo_idpo_id,, {
        " {
        "params":params": "Unknown "Unknown", "", "vram_fpvram_fp16":16": "Unknown "Unknown", "", "vramvram_4_4bit":bit": "Unknown "Unknown", "", "cpu_cpu_ok":ok": False False
    }
    })


#)


# ================================= =========================================================
#
# LOCAL MODEL LOCAL MODEL LOAD LOADINGING
# =
# =========================================================================================

@st

@st.cache.cache_resource(_resource(show_spshow_spinner="inner="Loading localLoading local embedding model embedding model (~80 (~80MB)...")
defMB)... load_local")
def_emb load_localeddings_embeddings():
    try():
    try:
        embeddings:
        embeddings = Hug = HuggingFacegingFaceEmbeddingsEmbeddings(
           (
            model_name model_name=LOC=LOCAL_AL_EMBEMBEDDEDDING_MODING_MODELEL,
,
            model           _kw model_kwargs={'deviceargs={'device': 'cpu'},
           ': ' encode_kwcpu'},
            encode_kwargs={'normalargs={'normalize_ize_embeddembeddings':ings': True True}
       }
        )
        return )
        return embeddings embeddings
    except
    except Exception as Exception as e e:
        st:
        st.error(f.error(f"Failed"Failed to load to load embeddings: {e embeddings:}")
        {e return None}")
        return None

@st.c

@ache_resourcest.c(showache_resource_spinner(show_sp="Loadinginner="Loading local LL local LLM (M (this maythis may take  take 1-1-2 minutes2 minutes on first on first load)... load)...")
def")
def load_local load_local_ll_llm(modelm(model_key:_key: str, str, use_ use_4bit4bit: bool: bool = True = True):
   ):
    try try:
        if:
        if is_ is_ollamaollama_model(model_model(model_key_key):
            return):
            return _load _load_oll_ollamaama_model_model(model_key(model_key)
       )
        else else:
            return:
            return _load _load_transformers_transformers_model(model_model(model_key, use__key, use_4bit4bit)
   )
    except Exception except Exception as e as e:
       :
        st.error st.error(f"(f"Failed toFailed to load LL load LLM '{model_key}':M '{model_key}': {e {e}")
}")
               st.w st.warning("arning("FallingFalling back to back to GPT- GPT-2...2...")
       ")
        try try:
            token:
            tokenizer =izer = GPT2 GPT2TokenizerTokenizer.from.from_pret_pretrained("rained("gptgpt2")
            model2")
            model = GPT = GPT2LM2LMHeadModelHeadModel.from_p.from_pretrainedretrained("g("gpt2pt2")
")
                       if token if tokenizer.pizer.pad_tokenad_token is None is None:
                tokenizer:
                tokenizer.pad.pad_token =_token = tokenizer tokenizer.eos_token.eos_token
           
            model model.eval.eval()
           ()
            device = device = "c "cuda"uda" if torch if torch.cuda.cuda.is_available().is_available() else " else "cpucpu"
            return"
            return tokenizer, tokenizer, model model, device, device, ", "transformerstransformers"
       "
        except Exception except Exception as e2 as e2:
:
            st            st.error(f.error(f"Fall"Fallback alsoback also failed: failed: {e {e22}")
            return}")
            return None, None, None, None, None, None, None None

def _

def _load_load_ollamaollama_model(model_model(model_key: str_key: str):
    if):
    if not O not OLLAMALLAMA_AVA_AVAILABLEILABLE:
        raise:
        raise ImportError ImportError("("ollamaollama library library not installed not installed. Run. Run: pip: pip install oll install ollama")
ama    model")
    model_tag =_tag = extract_ extract_ollamaollama_tag(model_tag(model_key_key)
    try)
    try:
       :
        client = client = ollama ollama.Client(.Client(host=host=st.sessionst.session_state._state.ollamaollama_host_host)
        response)
        response = client = client.list.list()
        models()
        models_list =_list = response.get('models', response.get('models', []) if []) if isinstance(response isinstance(response, dict, dict) else) else getattr(response getattr,(response, 'models 'models', [])
        model_names =', [])
        model_names = []
        for []
        for m in m in models_list models_list:
           :
            if isinstance if isinstance(m,(m, dict dict):
                name):
                name = m = m.get('.get('model')model') or m or m.get('.get('namename')
            else')
            else:
               :
                name = getattr name = getattr(m,(m, 'model 'model',', None None) or) or getattr getattr(m,(m, 'name 'name', None', None)
           )
            if name if name:
                model_names:
                model_names.append(name.append(name)
       )
        if model if model_tag not_tag not in model in model_names_names:
            st:
            st.warning.warning(f"(f"⚠️⚠️ Model '{ Model '{model_tagmodel_tag}' not}' not found found in in Ollama Ollama.")
           .")
            if model if model_names_names:
               :
                st.info(f st.info(f""📋 Available: {📋 Available: {', '.', '.join(modeljoin(model_names[:_names[:5])5])}")
           }")
            return None return None,, None, st None, st.session_state.session_state.oll.ollama_hostama_host,, " "ollama"
   ollama"
    except Exception except Exception as conn as conn_err:
        st_err:
        st.error(f.error(f""❌ Connection❌ Connection Error: Error: {conn {conn_err_err}")
        return}")
        return None, None, None, None, st.session st.session_state._state.ollamaollama_host,_host, "oll "ollamaama"
    return"
    return None, model_tag None, model_tag, st.session_state, st.session.oll_state.ollama_hostama_host, ", "ollama"

def _load_transformersoll_model(modelama"

def _load_transformers_key:_model(model str,_key: use_ str,4bit use_4bit: bool = True: bool = True):
   ):
    repo_id repo_id = get = get_hf_hf_repo_repo_id(model_id(model_key_key)
    device)
    device = " = "cudacuda" if" if torch.c torch.cuda.isuda.is__availableavailable() else() else "cpu "cpu"
   "
    available_v available_vram =ram = get_ get_available_gavailable_gpu_mpu_memoryemory()
   ()
    mem mem_info =_info = estimate_model estimate_model_memory_memory(model_key(model_key, use, use_4bit_4)
    stbit.side)
    stbar.info.sidebar.info(f"""
   (f"""
    📊 Model 📊 Model Memory Estimate Memory Estimate:
   :
    - Parameters - Parameters: {: {mem_infomem_info['params['params']']}
    -}
    - VRAM VRAM (FP (FP16):16): {mem {mem_info['_info['vramvram_fp_fp16']16']}
}
       - VR - VRAM (AM (4-bit4-bit): {): {mem_infomem_info['v['vram_ram_4bit']4bit']}
    - CPU OK: {'✅ Yes' if mem_info['cpu_ok'] else '❌ No'}
    - Available}
    - CPU OK: {'✅ Yes' if mem_info['cpu_ok'] else '❌ No'}
    VRAM - Available VRAM: {: {f'{available_vf'{available_vram:.ram:.1f1f}GB}GB' if' if available_v available_vram elseram else 'N 'N/A (/A (CPU)'CPU)'}
   }
    "" """)
    if")
    if " "00.5.5B"B" in repo in repo_id or_id or "1.1 "1.1B"B" in repo in repo_id or "_id or "ggpt2pt2" in" in repo_id repo_id:
       :
        use_ use_4bit = False4bit = False
   
    quantization_config quantization_config = None = None
   
    if use if use_4_4bit and device == "cbit and device == "cuda" anduda" and available available_vram_vram:
       :
        try try:
            from:
            from transformers import transformers import BitsAnd BitsAndBytesConfigBytesConfig
           
            quantization_config quantization_config = Bits = BitsAndBytesAndBytesConfigConfig(
                load(
                load_in__in_4bit=True4bit=True,
                bnb_,
                bnb_4bit4bit_compute_compute_dtype_dtype=tor=torch.floatch.float1616,
                b,
                bnb_4nb_bit4bit_use_use_double_double_quant_quant=True=True,
                b,
                bnb_nb_4bit4bit_quant_quant_type="_type="nf4nf4",
           ",
            )
            )
            st st.s.sidebaridebar.success("✅ .success("4-bit✅  quantization enabled4-bit quantization enabled")
       ")
        except Import except ImportErrorError:
            st.side:
            st.sidebar.wbar.warning("arning("⚠️ bitsand⚠️ bitsandbytes notbytes not installed installed.")
            use.")
            use_4_4bit =bit = False False
    token
    tokenizer =izer = AutoTokenizer.from_p AutoTokenizer.from_pretrainedretrained(
       (
        repo_id repo_id, trust, trust_remote_remote_code_code=True=True, padding, padding_side_side="left="left", use", use_fast_fast=True=True
   
    )
    model )
    model_kw_kwargs =args = {
        {
        "trust "trust_remote_remote_code":_code": True True,
        ",
        "torchtorch_dtype_dtype": torch.float16": torch.float16 if device if device == " == "cudacuda" else" else torch.float torch.float3232,
   ,
    }
    if }
    if quantization_config quantization_config:
       :
        model model_k_kwargswargs["quant["quantization_configization_config"] = quantization_config
"] = quantization_config
        model_k       wargs model_k["devicewargs_map"]["device = "_map"]auto = ""
    elifauto"
    elif device == device == "c "cudauda":
       ":
        model model_k_kwwargs["args["device_mapdevice_map"] ="] = "auto "auto"
   "
    model = model = AutoModel AutoModelForCForCausalLMausalLM.from_p.from_pretrainedretrained(repo(repo_id,_id, **model **model_kwargs_kwargs)
    if "device_map" not in model_k)
    if "device_map" not in model_kwargswargs and device and device == " == "cpucpu":
":
        model        model = model = model.to(.to(device)
   device)
    model.eval model.eval()
   ()
    if token if tokenizer.pizer.pad_tokenad_token is None is None:
:
               tokenizer tokenizer.pad.pad_token =_token = tokenizer tokenizer.eos.eos_token_token
    return
    return tokenizer tokenizer, model, model, device,, device, "transformers "transformers"


#"


# ================================= =========================================================
#
# DOCUMENT DOCUMENT PROCESSING
 PROCESSING#
# ================================= =========================================================

def

def extract_l extract_laser_aser_metadata(textmetadata(text: str: str, filename, filename: str) ->: str Dict[str) ->, any Dict[str]:
   , any]:
    metadata = metadata = {
        {
        "source "source": filename,
        "laser_topics": [],
        "parameters_found":": filename,
        "laser_topics": [],
        "parameters_found": {},
 {},
        "        "has_equhas_equationsations": bool": bool(re.search(re.search(r'(r'[\([\(=]\=]\s*[\d.]+\s*[\d.]+\s*s*[×[×*]\*]\s*s*10\10\^',^', text text)),
        ")),
        "has_fhas_figigures": boolures": bool(re.search(re.search(r'(r'Figure\Figure\s*\s*\d+|d+|Fig\.Fig\.\s\s*\d*\d+',+', text, text, re.I re.I)),
   )),
    }
    }
    text_lower = text_lower = text.lower text.lower()
    for()
    topic for topic, keywords, keywords in LAS in LASER_KEYER_KEYWORDSWORDS.items.items():
        if():
        if any(kw in text_lower for kw in any(kw in text_lower for kw in keywords keywords):
            metadata):
            metadata["las["laser_toper_topics"].ics"].append(tappend(topicopic)
   )
    param param_patterns_patterns = {
        "wavelength_nm = {
        "wavelength_nm": r": r'(\'(\d+(d+(?:\.?:\.\d\d+)?+)?)\s)\s*(?:*(?:nm|nm|nanometersnanometers?)\?)\s*(s*(?:w?:wavelength|avelength|λ|λ|lambda)lambda)',
       ',
        "p "pulse_dulse_duration_furation_fs": r's": r'(\d(\d+(?:+(?:\.\\.\d+d+)?)?)\)\s*(s*(?:fs?:fs|f|femtosecondsemtoseconds?)\?)\s*(s*(?:p?:pulse|ulse|duration)',
duration)       ',
        "fluence_Jcm "fluence2":_Jcm r'2":(\d r'+(?:(\d\.\+(?:d+\.\)?)\d+s*()?)\?:Js*(/cm²?:J|J/cm²/cm2|J/cm2|fluence|fluence))',
       ',
        "repet "ition_raterepetition_rate": r'(\": r'(\d+(d+(?:\.?:\.\\d+)?d+)?)\s)\s*(?:kHz|*(?:kHz|MHz|MHz|Hz)\Hz)\s*(s*(?:re?:repetitionpetition|rate|rate|f|freq)req)',
        "',
       spot "spot_size_um":_size_um": r' r'(\d+(?:(\d+(?:\.\\.\d+d+)?)\)?)\s*(s*(?:µm?:µm|um|um|mic|microns?rons?)\s)\s*(?:*(?:spot|spot|diameterdiameter))',
   ',
    }
    for }
    for param, param, pattern pattern in in param param_pattern_patterns.itemss.items():
       ():
        match = match = re.search re.search(pattern,(pattern, text, text, re.I re.I)
       )
        if match if match:
           :
            try try:
                metadata:
                metadata["parameters["parameters_found_found"][param]"][param] = float = float(match(match.group(1.group(1))
            except))
            except:
               :
                pass pass

    return    return metadata metadata


def load


def load_and_chunk_laser_documents(uploaded_files: List)_and_chunk_laser_documents(uploaded_files: List) -> Tuple -> Tuple[List[List[Document],[Document], CrossDocument CrossDocumentKnowledgeGraphKnowledgeGraph]:
   ]:
    all_ch all_chunks =unks = []
    []
    graph = graph = CrossDocument CrossDocumentKnowledgeGraphKnowledgeGraph()

   ()

    for uploaded for uploaded_file in_file in uploaded_files uploaded_files:
        with:
        with temp tempfile.Nfile.NamedTamedTemporaryFileemporaryFile(delete(delete=False,=False, suffix=".pdf suffix=".pdf" if" if uploaded_file uploaded_file.name.end.name.endswith('.swith('.pdf')pdf') else ". else ".txt")txt") as tmp as tmp:
           :
            tmp.write tmp.write(upload(uploaded_fileed_file.getbuffer())
           .getbuffer())
            tmp tmp_path_path = tmp = tmp.name.name

        try

        try:
           :
            file_hash file_hash = compute = compute_file_hash_file_hash(tmp_path(tmp_path)
           )
            cached_ cached_meta =meta = st.session st.session_state.m_state.metadata_cacheetadata_cache.get(u.get(uploadploadeded_file.name_file.name, file, file_hash_hash)

            if)

            if cached_ cached_metameta:
                bib:
                bib_meta_meta = cached = cached_meta_meta
               
                st.info st.info(f"📚(f" Using cached📚 Using cached metadata for metadata for `{upload `{eduploaded_file.name_file.name}`}`")
            else")
            else:
               :
                if uploaded if uploaded_file.name_file.name.endswith.endswith('.pdf'):
                   ('.pdf'):
                    bib_ bib_meta =meta = extract_ extract_metadata_frommetadata_from_pdf_pdf_file(tmp_file(tmp_path,_path, uploaded_file uploaded_file.name)
                else:
                   .name)
                else:
                    with open with open(tmp_path(tmp_path, ', 'r',r', encoding=' encoding='utf-utf-8',8', errors=' errors='ignore')ignore') as f as f:
                       :
                        text_content text_content = = f f.read.read()
                    bib()
                    bib_meta_meta = extract = extract_metadata_metadata_from_text_from_text_file(text_file(text_content,_content, uploaded_file uploaded_file.name.name)
                st)
                st.session_state.session_state.metadata.metadata_cache.set_cache.set(uploaded_file.name,(uploaded_file.name, bib_ bib_meta,meta, file_hash file_hash)
)
                               st.info st.info(f"(f"📚📚 Extracted Extracted metadata: metadata: {bib {bib_meta_meta.format_c.format_citation('apaitation('')apa')}}")

            if")

            if uploaded_file uploaded_file.name.end.name.endswith('.swith('.pdfpdf'):
                loader'):
                loader = Py = PyPDFLoaderPDFLoader(tmp_path(tmp_path)
           )
            else else:
                loader:
                loader = Text = TextLoader(tmpLoader(tmp_path,_path, encoding=' encoding='utf-8utf-8')

            pages')

            pages = loader = loader.load.load()
()
            chunks            chunks = semantic = semantic_chunk_chunk_document_document(p(pages, uploadedages, uploaded_file.name_file.name)

           )

            for chunk for chunk in chunks in chunks:
                chunk:
               .metadata chunk.m.updateetadata.update({
                   ({
                    **extract_l **extaser_ract_lmetadata(chaser_unkmetadata(ch.pageunk.page_content,_content, uploaded_file.name),
                    "bibliographic uploaded_file.name),
                    "bibliographic": bib": bib_meta_meta.to_dict.to_dict(),
                   (),
                    "citation "citation_display_display": bib": bib_meta_meta.format_c.format_citation(stitation(st.session_state.session_state.get('.get('citation_citation_style',style', 'apa 'apa')),
               ')),
                })

            })

            graph.add graph.add_document_document(upload(uploaded_fileed_file.name,.name, chunks, chunks, bib_ bib_metameta)
            all)
            all_chunks_chunks.extend(ch.extend(chunksunks)
            st)
            st.info(f.info(f"✅"✅ Loaded {len Loaded {len(chunks)} semantic(chunks)} semantic chunks from chunks from `{ `{uploadeduploaded_file.name_file.name}`}`")

        except")

        except Exception as Exception as e e:
            st:
            st.error(f.error(f""❌❌ Error Error processing ` processing `{upload{uploaded_fileed_file.name}`.name}`: {: {ee}")
}")
            import            import traceback traceback
           
            st.error st.error(trace(traceback.formatback.format_exc())
       _exc())
        finally finally:
            if:
            if os.path os.path.exists(tmp.exists(tmp_path_path):
                os):
                os.remove(tmp.remove(tmp_path_path)

    return)

    return all_ch all_chunks,unks, graph graph


@st


@st.cache.cache_resource_resource
def create
def create_local_vector_local_vector_store(ch_store(chunks:unks: List List[Document],[Document], embedding_model embedding_model_key:_key: str str):
    try):
    try:
       :
        embeddings = embeddings = load load_local_local_emb_embeddingseddings()
       ()
        if embeddings if embeddings is None is None:
           :
            return None return None
       
        vectorstore vectorstore = FA = FAISS.fromISS.from_documents_documents(chunks(chunks, embeddings)
       , embeddings)
        vectorstore.metadata vectorstore.metadata = = {
            " {
            "total_chtotal_chunks":unks": len(ch len(chunksunks),
            "),
            "embeddingembedding_model":_model": embedding_model embedding_model_key_key,
            "created_at,
            "created_at": datetime": datetime.now()..now().isoformatisoformat(),
           (),
            "las "laserer_top_topics":ics": list(set list(set(
               (
                topic for topic for chunk in chunk in chunks for topic in chunk.m chunks for topic in chunk.metadata.get("lasetadata.get("laser_toper_topics",ics", [ [])
           ])
            ))
        ))
        }
        return }
        return vectorstore vectorstore
   
    except Exception except Exception as e as e:
       :
        st.error st.error(f"(f"Failed toFailed to create vector create vector store: store: {e {e}")
       }")
        return None return None


# =================================


#============ =================================
#============ RAG
# RAG FUN FUNCTIONSCTIONS
#
# ================================= =========================================================

def extract_query

def extract_query_entities_entities(query:(query: str) str) -> List -> List[str[str]:
    entities]:
    entities = = []
 []
    query   _lower query_lower = query.lower = query.lower()

    for()

    for canonical, canonical, aliases aliases in MATERIAL in MATERIAL_AL_ALIASESIASES.items.items():
        if any(():
        ifalias in any( query_lalias in query_lower forower for alias in alias in aliases aliases):
           ):
            entities.append entities.append(canon(canonical)

    for canonical, aliasesical)

    for canonical, aliases in METHOD_AL in METHOD_ALIASESIASES.items.items():
        if():
        if any( any(alias inalias in query_l query_lowerower for for alias in alias in aliases aliases):
           ):
            entities.append entities.append(canonical(canonical)

    for)

    for param_name param_name in QU in QUANTITYANTITY_PAT_PATTERNSTERNS.keys.keys():
        if():
        if param_name param_name.replace(".replace("__",", " ") " ") in query in query_lower_lower or param or param_name in_name in query_l query_lowerower:
            entities:
            entities.append(param.append(param_name_name)

    for topic,)

    for keywords in topic, LASER keywords in LASER_KEYWOR_KEYWORDSDS.items.items():
       ():
        if any if any(kw(kw in query in query_lower_lower for kw for kw in keywords in keywords):
           ):
            entities.append entities.append(topic(topic)

   )

    return entities return entities


def


def create_s create_scientific_recientific_reasoningasoning_prompt_prompt(
   (
    retrieved_ch retrieved_chunks:unks: List List[Document[Document],
    query],
    query: str: str,
   ,
    graph: graph: CrossDocument CrossDocumentKnowledgeGraphKnowledgeGraph,
    consensus_data,
    consensus_data: List: List[Dict[Dict],
   ],
    contradictions: contradictions: List List[Dict[Dict]
) ->]
) -> str str:
    context:
    context_parts_parts = = []
    for []
    for i, i, chunk chunk in in enumerate(retrieved_ch enumerateunks,(retrieved_ch 1unks, 1):
        citation =):
        citation = chunk.metadata.get chunk.metadata.get("citation("citation_display_display")
       ")
        if not citation:
            source if not citation:
            source = chunk = chunk.metadata.metadata.get("source.get("source",", "unknown "unknown")
           ")
            citation = citation = f f"[Source {"[Source {i}i} - { - {source}]source}]"
        section"
        = chunk.m section =etadata.get chunk.metadata.get("section("section", "", "UNKNOWNUNKNOWN")
       ")
        content = content = chunk.page chunk.page_content[:_content[:600600]] + " + "..." if..." if len(ch len(chunk.pageunk.page_content)_content) >  > 600 else600 else chunk.page chunk.page_content_content
        context
        context_parts_parts.append(f"---.append(f"---\n\n[{i[{i}] {}] {citation}citation} | Section | Section: {: {section}\section}\n{n{content}\content}\nn")
    context")
    context = "\ = "\n".join(context_parts)

   n".join(context_parts)

    consensus_text consensus_text = = ""
    if ""
    if consensus_data consensus_data:
       :
        consensus_text consensus_text = "\ = "\nCrossnCross-Document-Document Consensus ( Consensus (statisticalstatistical agreement across agreement across papers): papers):\n\n"
       "
        for cons for cons in consensus in consensus_data[:_data[:33]:
            consensus]:
            consensus_text +=_text += f"- f"- {cons {cons['['entityentity']}:']}: {cons {cons['mean['mean']:.']:.2f2f} ±} ± {cons {cons['std['std']:.']:.2f2f} {} {cons['cons['unit']unit']} (across {cons} (across {cons['doc['doc_count']_count']} papers, n={cons} papers, n={cons['value['value_count']_count']})\})\nn"

   "

    contradiction_text contradiction_text = = ""
    if ""
    if contradictions contradictions:
        contradiction:
        contradiction_text =_text = "\n "\nDetectedDetected Contrad Contradictions Acrossictions Across Documents:\n Documents:\n"
        for"
        for contr in contr in contradictions[: contradictions[:33]:
            contradiction]:
            contradiction_text +=_text += f"- f"- {contr {contr['entity['entity']}:']}: {contr {contr['doc['doc_a_a']} reports {contr['mean']} reports_a'] {contr['mean_a']:.2:.2f}f} vs { vs {contr['contr['doc_bdoc_b']}']} reports { reports {contr['contr['mean_bmean_b']:.']:.2f2f} (} (ratio:ratio: {contr {contr['ratio['ratio']:.']:.1f1f}x, {contr}x, {contr['sever['severity']ity']})\n})\n"

"

       system_p system_prompt =rompt = """You are an """You are an expert scientific expert scientific research assistant research assistant specializing in specializing in laser-microstructure interactions laser-microstructure interactions, with a focus on multic, with a focus on multicomponentomponent alloys, alloys, additive manufacturing additive manufacturing, and, and physics physics-informed digital-informed digital twins twins.
Your task.
Your task is to is to synthesize evidence synthesize evidence from multiple from multiple research papers research papers and and provide a provide scientifically a scientifically rigorous answer rigorous answer.

RE.

REASONINGASONING RULES RULES:
1:
1. SYN. SYNTHESTHESIZE acrossIZE across documents — documents — do not do not just summarize just summarize one paper one paper at a at a time time
2.
2. Identify CONS Identify CONSENSUSENSUS where multiple where multiple papers agree papers agree, and CON, and CONTRTRADICTADICTIONS whereIONS where they disagree they disagree
3. Report
3. Report UNCERT UNCERTAINTAINTY explicitlyY explicitly — use — use phrases like phrases like "re "reported valuesported values range from range from X to X to Y", Y", "the "the consensus mean consensus mean is Z is Z ± σ ± σ"
"
44. Cite sources using. Cite sources using the EX the EXACT citationACT citation format provided format provided (Author (Author et al et al., Journal., Journal, Year, Year)
5)
5. If evidence is. If evidence is insufficient or insufficient or contradictory, contradictory, state this state this explicitly rather explicitly rather than fabric than fabricating consensusating consensus
6
6. Dist. Distinguish betweeninguish between direct experimental direct experimental results and results and inferred/the inferred/theoreticaloretical claims claims
7
7. For. For numerical values numerical values, include, include units and units and note if papers use note if papers use different measurement different measurement conditions conditions

OUTPUT

OUTPUT STRUCTURE STRUCTURE:
1:
1. **. **Direct AnswerDirect Answer**: Conc**: Concise answerise answer to the to the question question
2
2.. **Evidence **Evidence Synthesis**: Synthesis**: Integration of Integration of findings across papers with citations findings across papers with citations
3.
3. **Cons **Consensus &ensus & Variability Variability**: Statistical**: Statistical summary if summary if multiple papers multiple papers report the report the same parameter same parameter
4
4.. ** **ContradContradictions &ictions & Limitations**: Note Limitations**: any Note any conflicting results or methodological conflicting results or methodological differences differences
5.
5. **Conf **Confidence Assessmentidence Assessment**: State**: State your confidence your confidence (High (High/Medium/Medium/Low/Low) and) and why why
"""
   
"""
    user_p user_prompt =rompt = f""" f"""RetrievedRetrieved Document Context Document Context:
{:
{contextcontext}
{cons}
{consensus_textensus_text}
{}
{contradcontradiction_textiction_text}

User}

User Question: Question: {query {query}

Provide}

Provide a scientifically a scientifically rigorous answer following the rigorous answer following the structure above structure above. Be. Be precise about precise about uncertainty and uncertainty and cross-d cross-document agreementocument agreement."""
   ."""
    return system return system_prompt_prompt + user + user_prompt_prompt


def


def generate_local generate_local_response_transform_response_transformersers(token(tokenizer,izer, model, model, device: device: str, str, prompt: prompt: str, str, backend_name backend_name: str: str) ->) -> str str:
    try:
    try:
       :
        if " if "QwenQwen" in backend_name or "" in backend_name or "qwenqwen" in" in backend_name backend_name.lower.lower():
            messages():
            messages = [
                {" = [
                {"role":role": "system "system", "", "content": "Youcontent": "You are an are an expert in expert in laser-micro laser-microstructure interaction research.structure interaction research. Synthes Synthesize evidenceize evidence across multiple across multiple papers rigorously papers rigorously."."},
                {"role":},
                {"role": "user "user", "", "content":content": prompt prompt}
           }
            ]
            formatted ]
            formatted_prompt_prompt = token = tokenizer.applyizer.apply_chat_chat_template(m_template(messages,essages, tokenize tokenize=False,=False, add_g add_generation_peneration_prompt=True)
        elif "rompt=True)
        elif "LlamaLlama" in" in backend_name backend_name or " or "llllama"ama in" in backend_name backend_name.lower.lower():
            messages():
            messages = = [
                {"role": [
                {"role": "system "system", "", "content":content": "You "You are an are an expert in expert in laser-micro laser-microstructure interactionstructure interaction research. research. Synthes Synthesize evidenceize evidence across multiple across multiple papers rigorously papers rigorously."."},
                {"},
                {"role":role": "user "user", "", "content":content": prompt prompt}
           }
            ]
            formatted ]
            formatted_prompt_prompt = token = tokenizer.applyizer.apply_chat_chat_template(m_template(messages,essages, tokenize tokenize=False,=False, add_g add_generation_prompt=Trueeneration_prompt=True)
       )
        elif " elif "MistMistral"ral" in backend in backend_name or_name or "mist "mistral"ral" in backend in backend_name.lower_name.lower():
           ():
            formatted_p formatted_prompt =rompt = f" f"<s<s>[INST>[INST] {] {prompromptpt} [/INST]"
       } [/INST]"
        else else:
            formatted:
            formatted_prompt_prompt = prompt = prompt

       

        inputs = inputs = tokenizer tokenizer.encode.encode(
            formatted(
            formatted_prompt_prompt,, return_tensors return='pt_tensors='pt', trunc', truncation=Trueation=True,
           ,
            max_length max_length=LAS=LASERER_D_DOMAIN_CONFIG["max_contextOMAIN_CONFIG["max_context_tokens_tokens"]
       "]
        )
        )
        if device if device == " == "cudacuda" and" and torch.c torch.cuda.is_available():
           uda.is_available():
            inputs = inputs = inputs.to inputs.to('c('cudauda')

')

        with        with torch.no torch.no_grad_grad():
           ():
            outputs = outputs = model.generate model.generate(
                inputs,
                max_new_tokens=LASER(
                inputs,
                max_new_tokens=LASER_DOMAIN_CONFIG_DOMAIN_CONFIG["max["max_new_t_new_tokensokens"],
                temperature"],
                temperature=LAS=LASER_DER_DOMAINOMAIN_CONFIG["_CONFIG["temperaturetemperature"],
                do"],
                do_sample=(_sample=(LASERLASER_DOM_DOMAIN_CONFIGAIN_CONFIG["temperature["temperature"] >"] >  00),
               ),
                pad_token pad_token_id=_id=tokenizertokenizer.eos.eos_token_id_token_id,
               ,
                eos eos_token_id_token_id=token=tokenizer.eos_token_id,
                noizer.eos_token_id,
                no_repeat_repeat_ng_ngram_sizeram_size=3=3,
               ,
                early_st early_stopping=True,
            )

        full_textopping=True,
            )

        full_text = token = tokenizer.decodeizer.decode(outputs(outputs[0[0], skip], skip_special_special_t_tokensokens=True=True)

        if)

        if "[/ "[/INST]"INST]" in full in full_text_text:
            answer:
            answer = full = full_text.split_text.split("[/("[/INST]INST]")[-")[-1].1].stripstrip()
        elif()
        elif "Conf "Confidence Assessmentidence Assessment:" in:" in full_text full_text:
           :
            answer = answer = full_text full_text[full[full_text.find_text.find("Direct("Direct Answer:" Answer:"):].):].strip()strip() if " if "Direct Answer:" inDirect Answer:" in full_text full_text else full else full_text[-1500:]._text[-1500:].stripstrip()
        else()
        else:
           :
            answer = answer = full full_text_text[-[-LASLASER_DER_DOMAINOMAIN_CONFIG["_CONFIG["max_newmax_new_tokens_tokens"] *"] * 2 2:].:].stripstrip()

        answer = re()

        answer = re.sub(r.sub(r'\s'\s+',+', ' ', ' ', answer). answer).stripstrip()
        return()
        return answer if answer if answer else answer else " "I wasI was unable unable to generate to generate a response a response. Please. Please try re try rephrphrasing yourasing your question question."

    except."

    except Exception as Exception as e e:
        st:
        st.error(f.error(f"Generation"Generation error: error: {e {e}")
       }")
        return f return f"Error"Error generating response generating response: {: {str(estr(e)[:)[:200]}200]}......"


def generate"


def generate_local_response_local_response_oll_ollama(modelama(model_tag:_tag: str, str, ollama ollama_host_host:: str str, prompt:, prompt: str) str) -> str -> str:
   :
    try try:
        client:
        client = oll = ollama.Clientama.Client(host(host=oll=ollama_hostama_host)
       )
        messages messages = = [
            [
            {"role {"role": "": "system",system", "content "content": "": "You are an expert in laserYou are an expert in laser-microstructure-microstructure interaction research interaction research. Synt. Synthesizehesize evidence evidence across across multiple papers multiple papers rigorously." rigorously."},
           },
            {"role {"role": "": "user",user", "content "content": prompt": prompt}
       }
        ]
 ]
               try try:
            response:
            response = client = client.chat(
                model=.chat(
                model=model_tagmodel_tag, messages, messages=messages=messages,
               ,
                options={" options={"temperature": LASERtemperature": LASER_DOM_DOMAIN_CONFIGAIN_CONFIG["temperature["temperature"], "num_predict": LASER_DOM"], "num_predict": LASER_DOMAIN_CONFIGAIN_CONFIG["max["max_new_tokens"]},
               _new_tokens"]},
                stream=True stream=True
           
            )
            )
            full_response full_response = = ""
            for ""
            for chunk in response chunk in response:
                if:
                if isinstance(ch isinstance(chunk,unk, dict dict):
                    if):
                    if 'message 'message'' in in chunk and chunk and 'content' in chunk[' 'content' in chunk['message']message']:
                       :
                        full_response full_response += chunk += chunk['message['message']['content']['content']
                   ']
                    elif ' elif 'content'content' in chunk in chunk:
                       :
                        full_response full_response += chunk += chunk['content['content']
               ']
                elif has elif hasattr(chattr(chunkunk,, ' 'messagemessage') and') and hasattr hasattr(chunk(chunk.message,.message, 'content 'content'):
                   '):
                    full_response full_response += += chunk chunk.message.content.message.content
       
        except TypeError except TypeError:
            response = client:
            response = client.chat.chat(
                model(
                model=model=model_tag,_tag, messages messages=m=messagesessages,
                options,
                options={"temperature={"temperature": LAS": LASER_DER_DOMAINOMAIN_CONFIG["_CONFIG["temperature"],temperature"], "num "num_predict_predict": LAS": LASER_DER_DOMAINOMAIN_CONFIG["_CONFIG["max_newmax_new_tokens_tokens"]"]}
           }
            )
            if )
            if isinstance(response isinstance(response, dict, dict):
):
                full_response = response                full_response = response.get('.get('message',message', {}). {}).get('get('content',content', ' '')
            elif')
            elif hasattr hasattr(response,(response, 'message 'message'):
                full_response ='):
                full_response = response.message.content response
           .message.content
            else:
 else:
                full_response                full =_response = str(response str(response)

        return full)

        return full_response.strip() if_response.strip full_response() if.strip() full_response else ".strip()I was else " unable toI was unable generate a to generate a response. response. Please try Please try reph rephrasingrasing your question your question."
   ."
    except Exception except Exception as e as e:
       :
        st.error st.error(f"(f"OllOllama generationama generation error: error: {e {e}")
       }")
        return f return f"Error"Error generating response generating response via Oll via Ollama: {strama: {str(e(e)[:200)[:200]}...]}..."


def"


def generate_local generate_local_response(token_response(tokenizer,izer, model_or model_or_tag,_tag, device_or device_or_host:_host: str, str, prompt: prompt: str, str, backend: backend: str, str, backend_type: str backend_type: str) ->) -> str str:
    if:
    if backend_type backend_type == " == "ollamaollama":
       ":
        return generate return generate_local_response_local_response_oll_ollama(modelama(model_or_tag_or_tag, device, device_or_host_or_host, prompt, prompt)
   )
    else else:
        return:
        return generate_local generate_local_response_response_transform_transformers(tokeners(tokenizer,izer, model_or model_or_tag,_tag, device_or device_or_host,_host, prompt, prompt, backend backend)


def retrieve)


def retrieve_and_answer_and_answer(
   (
    vectorstore vectorstore,
    graph:,
    graph: CrossDocument CrossDocumentKnowledgeKnowledgeGraphGraph,
   ,
    tokenizer tokenizer,
   ,
    model,
    device model,
    device_or_host: str,
   _or_host: str,
    backend: backend: str str,
    backend,
    backend_type:_type: str str,
    query,
    query: str: str,
   ,
    k: k: int = int = None,
    None,
 score_threshold    score: float_threshold: float = None = None
)
) -> Tuple[str, -> Tuple[str, List List[Document],[Document], float, Dict[str float, Dict[str, Any, Any]]]]:
    k:
    k = k = k or LAS or LASER_DER_DOMAINOMAIN_CONFIG["_CONFIG["retrieretrieval_kval_k"]
   "]
    score_th score_threshold =reshold = score_th score_threshold orreshold or LAS LASERER_DOMAIN_CONFIG_DOMAIN_CONFIG["score["score_threshold_threshold"]

   "]

    retri retriever =ever = vectorstore vectorstore.as_.as_retriretrieverever(
        search(
        search_type="_type="similaritysimilarity_score_th_score_thresholdreshold",
        search",
        search_kw_kwargs={"k": k *args={"k": k * 2, " 2, "score_thscore_thresholdreshold":": score_th score_thresholdreshold}
   }
    )
    semantic )
    semantic_docs_docs = ret = retrieverriever.invoke.invoke(query(query)

    query)

    query_entities_entities = extract = extract_query_entities_query_entities(query)

    if graph and query_entities and st.session_state(query.get(")

    if graph and query_entities and st.session_state.get("reasoningreasoning_mode",_mode", True True):
        graph):
        graph_results =_results = graph.get graph.get_related_related_chunks_chunks(query_(query_entities,entities, st.session_state.all st.session_state.all_chunks, depth_chunks, depth=2=2)
       )
        seen = seen = {(d {(d.metadata.metadata.get(".get("source"),source"), d.m d.metadata.getetadata.get("ch("chunk_indexunk_index")) for d in")) for d in semantic_d semantic_docsocs}
        for}
        for chunk, chunk, score, score, reason in reason in graph_results graph_results:
           :
            key = key = (ch (chunk.munk.metadata.getetadata.get("source("source"), chunk"), chunk.metadata.metadata.get(".get("chunkchunk_index_index"))
            if"))
            if key not key not in seen in seen and len and len(sem(semantic_dantic_docs)ocs) < k < k *  * 22:
                semantic:
                semantic_docs_docs.append(ch.append(chunkunk)
                seen)
                seen.add(key.add(key)

   )

    if semantic_docs if semantic_docs:
       :
        query_ query_embeddingembedding = vector = vectorstore.store.embeddingembedding_function._function.embed_queryembed_query(query(query)
        scored)
        scored_docs_docs = = []
        for []
        for doc in doc in semantic_d semantic_docsocs:
            doc:
            doc_embed_ding =embedding = vectorstore vectorstore.embed.embedding_functionding_function.embed.embed_query(d_query(doc.pageoc.page_content[:_content[:500500])
            sim])
            sim = np.dot(query = np.dot(query_embed_embedding,ding, doc_ doc_embeddingembedding) /) / (
                (
                np.l np.linalginalg.norm.norm(query_(query_embeddingembedding) *) * np.l np.linalinalgg.norm.norm(doc(doc_embed_embedding)ding) +  + 1e1e-8-8
           
            )
            )
            section_ section_boostboost = = 0 0.05.05 if doc if doc.metadata.metadata.get(".get("section")section") in [" in ["RESULTS",RESULTS", "D "DISCUSSIONISCUSSION"] else"] else 0 0
           
            scored_d scored_docs.appendocs.append((doc((doc, sim, sim + section + section_boost_boost))

       ))

        scored_d scored_docs.sortocs.sort(key=lambda x:(key=lambda x: x x[1],[1], reverse=True reverse=True)
       )
        retrieved_d retrieved_docs =ocs = [d [d for d for d, s, s in scored in scored_docs_docs[:k[:k]]
       ]]
        avg_re avg_relevance =levance = np.mean np.mean([s([s for d for d,, s s in scored in scored_docs_docs[:k[:k]])
   ]])
    else else:
        retrieved:
        retrieved_docs_docs = = []
        avg []
        avg_relevance_relevance =  = 00.0.

   0 if

    if not retrieved not retrieved_docs_docs:
       :
        return " return "Based onBased on the uploaded the uploaded documents, documents, I could I could not find not find information relevant information relevant to your to your question.", question.", [], avg [], avg_relevance_relevance,, {}

    consensus {}

    consensus_data =_data = []
    contradictions []
    = contradictions = []
    if graph and []
    if graph and st st.session_state.session_state.get(".get("cross_dcross_doc_oc_consensusconsensus", True", True):
       ):
        for ent for ent in query in query_entities_entities:
           :
            cons = cons = graph.find graph.find_cons_consensus(ensus(entent)
            if)
            if cons:
                consensus_data.append(cons)
            contr = graph.find_contradictions(ent, threshold_factor= cons:
                consensus_data.append(cons)
            contr = graph.find_contradictions(ent, threshold_factor=1.1.55)
            contradictions.extend()
            contradictions.extend(contr)

    prompt = create_scientificcontr)

    prompt = create_scientific_reason_reasoning_ping_prompt(retrompt(retrieved_drieved_docs,ocs, query, query, graph, graph, consensus_data consensus_data, contradictions, contradictions)
   )
    answer = answer = generate_local generate_local_response_response(
        tokenizer=(
        tokentokenizerizer=, modeltokenizer, model_or_tag=model_or_tag=model, device, device_or_host_or_host=device_or=device_or_host_host,
        prompt,
        prompt=prom=prompt,pt, backend= backend=backend,backend, backend_type backend_type=backend=backend_type_type
   
    )

    reasoning )

    reasoning_meta_meta = = {
        " {
        "query_query_entities":entities": query_ query_entitiesentities,
        ",
        "consensusconsensus_found_found": len": len(cons(consensus_dataensus_data),
       ),
        "cont "contradictionsradictions_found_found": len": len(cont(contradictionsradictions),
       ),
        "multi "multi_hop_hop_expansion_expansion": len": len(sem(semantic_dantic_docs)ocs) > k > k,
   ,
    }

 }

       return answer return answer, retrieved, retrieved_docs_docs, avg, avg_relevance_relevance, reasoning, reasoning_meta_meta


#


# ================================= =========================================================
#
# STRE STREAMLITAMLIT UI
 UI
# =============================================# =============================================

def render_sidebar

def render_sidebar():
   ():
    with st with st.side.sidebar:
        stbar:
        st.markdown.markdown("###("### ⚙️ ⚙️ Configuration Configuration")
        backend")
        backend_option =_option = st. st.radio("radio("🔧🔧 Inference Backend", Inference Backend", options=[" options=["HugHugging Faceging Face Transformers Transformers", "",Oll "Ollama (ama (if installed)"],if installed)"], index=0 index=0)
       )
        st st.session_state.session_state.inference.inference_backend_backend = backend = backend_option_option

        if

        if backend_option backend_option == " == "OllOllama (ama (if installed)if installed":
           ) if not O":
            ifLLAMA not OLLAMA_AVA_AVAILABLEILABLE:
                st:
                st.error(".error("❌❌ ollama ollama library not library not installed installed")
                st")
                st.code(".code("pip installpip install ollama ollama")
           ")
            available_ available_ollamaollama_models_models = = [k for [k for k in k in LOCAL_ LOCAL_LLMLLM_OPT_OPTIONS.keysIONS.keys() if() if is_ is_ollamaollama_model(k_model(k)]
           )]
            model_ model_choice =choice = st.select st.selectbox("box("🧠🧠 Local LL Local LLM BackM Backend (end (OllOllama)",ama)", options options==available_available_ollamaollama_models_models if available if available_oll_ollama_ama_models elsemodels else ["No ["No Ollama Ollama models available models available"], index"], index=0=0)
       )
        else else:
            h:
            hf_f_models =models = [k [k for k for k in LOCAL in LOCAL_LL_LLM_M_OPTIONSOPTIONS.keys().keys() if not if not is_ is_ollollamaama_model(k_model(k)]
           )]
            model_ model_choice =choice = st.select st.selectbox("box("🧠🧠 Local LL Local LLM BackM Backend (end (HugHugging Faceging Face)", options)", options=hf=hf_models_models, index, index=2=2)

       )

        st.session_state.llm_model_ st.session_state.llm_model_choice =choice = model model__choicechoice

        if

        if backend_option == " backend_optionHug == "ging FaceHug Transformersging Face" and Transformers" and not is not is_oll_ollama_modelama_model(model_(model_choicechoice):
            st):
            st.session_state.session_state.use_.use_4bit4bit_quantization_quantization = st.checkbox(" =🗜 st.checkbox("🗜️ Use️ Use 4 4-bit quantization-bit quantization", value", value=True=True)

        if)

        if backend_option == " backend_option == "OllOllama (ama (if installedif installed)" or)" or is_ is_ollamaollama_model(model_choice_model(model_choice):
           ):
            st.session st.session_state._state.ollamaollama_host =_host = st.text st.text_input("_input("🌐🌐 Ollama Ollama Host Host",", value= value=st.sessionst.session_state._state.ollamaollama_host_host)

       )

        st st.markdown.markdown("####("#### 🔬 🔬 Reasoning Settings Reasoning Settings")
       ")
        st.session st.session_state.re_state.reasoningasoning_mode =_mode = st.check st.checkboxbox(
            "(
            "🧠🧠 Cross-d Cross-document reasoningocument reasoning", value", value=True=True,
            help="Enable,
            help="Enable entity extraction entity extraction, consensus, consensus detection, detection, and multi and multi-hop retrieval-hop retrieval across papers across papers"
       "
        )
        st.session_state.c )
        st.session_state.cross_dross_doc_oc_consensusconsensus = st = st.checkbox.checkbox(
           (
            " "📊 Detect📊 Detect consensus & consensus & contradictions", contradictions", value=True value=True,
           ,
            help=" help="StatisticallyStatistically compare reported compare reported values across values across documents documents"
       "
        )
        st )
        st.session_state.session_state.show_reasoning.show_reasoning_chain_chain = st = st.check.checkboxbox(
           (
            " "🔍 Show🔍 Show reasoning reasoning chain chain", value", value=True=True,
            help="Display the logical steps,
            help="Display the logical steps and evidence linking and evidence linking"
       "
        )

        )

        st.mark st.markdown("down("######## 🔬 Laser 🔬 Laser Domain Settings Domain Settings")
       ")
        st.session st.session_state.laser_d_state.laser_domain_omain_boost =boost = st.check st.checkbox("box("Boost laserBoost laser-topic-topic relevance", value=True)
        relevance", value=True)
        st.session st.session_state.show_state.show_s_sources = stources = st.checkbox("Show source citations.checkbox("Show source citations", value", value=True=True)

        st)

        st.markdown.markdown("####("#### 📝 📝 Citation Format Citation Format")
       ")
        st.session st.session_state.c_state.citation_itation_style =style = st.select st.selectboxbox(
            "(
            "Citation displayCitation display style", style", options=[" options=["apa",apa", "doi "doi", "", "full",full", "short "short"], index=0"], index=0,
            format,
            format_func=lambda_func=lambda x: x: {"apa {"apa": "": "APA:APA: FirstAuthor FirstAuthor et al et al., Journal., Journal, Year, Year", "", "doi":doi": "DOI "DOI: : 10.10.xxxx/xxxx/xxxxxxxxxx",
                                  ",
                                   "full "full": "": "Full:Full: Author ( Author (Year).Year). Title. Title. Journal, Journal, Vol(I Vol(Issuessue), Pages), Pages",", "short "short": "": "Short: [FirstShort: [FirstAuthor YearAuthor Year] or] or [DOI [DOI]"]"}[x}[x]
       ]
        )

 )

        st        st.session_state.session_state.max_.max_retrievedretrieved_chunks = st_chunks = st.sl.slider("ider("ChunksChunks to retrieve to retrieve", min", min_value=_value=2,2, max_value max_value=10=10, value, value=6=6)

       )

        st.mark st.markdown("down("------")
        st")
        st.markdown.markdown("("""
       ""
        <div <div style=" style="background:#background:#ff00f9f9ff;ff;padding:padding:1rem1rem;border;border-radius-radius::0.5rem0.5rem;border-left:;border-left:4px4px solid # solid #3b3b82f82f66">
       ">
        <strong> <strong>💡💡 New Reasoning New Reasoning Features:</ Features:</strong>
strong>
        <ul        <ul style style="margin="margin:0:0.5.5rem 0 rem 0 0 0 1rem1rem;padding;padding:0:0">
       ">
        <li <li><b><b>Cross-doc consensus</>Cross-doc consensus</b>:b>: Statistical agreement Statistical agreement across papers across papers</li</li>
        <li>
        <li><b><b>Cont>Contradictionradiction detection</ detection</b>:b>: Flags conflicting Flags conflicting results</ results</lili>
       >
        <li <li><b>><b>Multi-hopMulti-hop retrieval</ retrieval</b>:b>: Follows Follows entity links entity links across documents across documents</li</li>
       >
        <li <li><b><b>Section>Section-aware chunk-aware chunking</ing</b>:b>: Preserves Preserves Abstract/M Abstract/Methods/ethods/Results structureResults structure</li</li>
       >
        <li <li><b><b>Un>Uncertainty calibration</certainty calibration</b>:b>: Explicit confidence Explicit confidence in answers in answers</li</li>
       >
        </ul </ul>
       >
        </div </div>
       >
        """, """, unsafe_ unsafe_allow_htmlallow_html=True=True)

)

        st        st.markdown.markdown("---("---")
       ")
        gpu gpu_info =_info = "CU "CUDA"DA" if torch if torch.cuda.cuda.is_.is_available()available() else "CPU"
        vram_info = f"{get_available_gpu_memory():.1f}GB free" if torch.cuda.is_available() else "CPU"
        vram_info = f"{get_available_gpu_memory():.1f}GB free" if torch.cuda.is_available() and get_available_gpu_m and get_available_gpu_memory() else "Nemory() else "N/A"
        st/A.caption"
        st(f".caption(f"🖥️🖥️ Device: Device: { {gpu_infog}")
       pu_info st.c}")
       aption(f st.c"aption(f💾 Available" VRAM💾 Available: { VRAMvram: {vram_info}_info}")

       ")

        if PDF if PDF2DOI2DOI_AVA_AVAILABLEILABLE:
            st:
           .success(" st.success("✅ pdf✅ pdf2doi2doi: Available: Available")
       ")
        else else:
            st:
            st.info(".info("ℹℹ️ pdf️ pdf2doi2doi: Optional: Optional for DOI for DOI lookup lookup")
        if")
        if CROSSREF_ CROSSREF_AVAILABLEAVAILABLE:
           :
            st.success st.success("✅("✅ Crossref API Crossref API: Available: Available")
       ")
        else else:
            st:
            st.info(".info("ℹℹ️ Crossref️ Crossref: Optional: Optional for metadata for metadata enrichment enrichment")


def render")


def render_document_upload_document_uploaderer():
    st():
    st.markdown.markdown("###("### 📁 📁 Upload Laser Upload Laser Microstructure Microstructure Documents Documents")
    uploaded")
    uploaded_files =_files = st.file st.file_upload_uploaderer(
        "(
        "Select PDFSelect PDF or TX or TXT filesT files about laser about laser processing, processing, multicom multicomponent alloys, additiveponent alloys, additive manufacturing, manufacturing, etc etc.",
        type.",
        type=["pdf=["pdf", "", "txt"],txt"], accept_mult accept_multiple_filesiple_files=True=True,
        help,
        help="Documents="Documents will be will be processed with processed with semantic section semantic section detection and detection and cross-d cross-document entityocument entity linking linking."
   ."
    )
    return )
    return uploaded_files uploaded_files


def


def process_d process_documents(uocuments(uploadedploaded_files):
    if_files):
    if not not uploaded uploaded_files_files:
       :
        return return False False

   

    new_files new =_files = [f [f for f for f in uploaded in uploaded_files if_files if f.name f.name not in not in st.session st.session_state.process_state.processed_filesed_files]
   ]
    if not if not new_files new_files:
       :
        st.info st.info("✓("✓ All uploaded All uploaded files already files already processed processed")
        return")
        return st.session_state st.session.pro_state.processing_completecessing_com

    stplete.session_state

    st.messages.session_state.messages = []
    st.session_state.vector = []
    st.session_state.vectorstore =store = None None
    st
    st.session_state.session_state.all_chunks =.all_chunks = []
    []
    st.session st.session_state.k_state.knowledgenowledge_graph =_graph = None None

    with

    with st.sp st.spinner(finner(f"Processing"Processing {len {len(new(new_files_files)} document)} document(s)(s) with semantic with semantic reasoning..." reasoning..."):
       ):
        try try:
            chunks:
            chunks, graph, graph = load = load_and_ch_and_chunk_lunk_laser_daser_documents(newocuments(new_files_files)
            if)
            if not chunks not chunks:
               :
                st.error st.error("No("No chunks extracted chunks extracted. Check. Check file format.")
 file format               .")
                return False return False

           

            for f in new for f in new_files_files:
                st:
                st.session_state.session_state.processed.processed_files.add(f.name_files.add(f.name)

           )

            st.session st.session_state.all_state.all_chunks_chunks.extend(ch.extend(chunksunks)
            st)
            st.session_state.session_state.know.knowledge_graphledge_graph = graph = graph

           

            with st with st.spinner.spinner("Creating("Creating vector index vector index and knowledge graph..." and knowledge):
                graph..." vector):
               store = create vectorstore = create_local_vector_local_vector_store(st_store(st.session_state.session_state.all_ch.all_chunks,unks, LOCAL LOCAL__EMBEMBEDDEDDING_MODING_MODELEL)
                if)
                if vectorstore vectorstore is None is None:
                   :
                    return False return False
               
                st.session st.session_state._state.vectorstorevectorstore = vector = vectorstorestore

            if graph

            if graph:
                summary:
                summary = graph = graph.get_k.get_knowledge_summarynowledge_summary()
               ()
                # F # FIX:IX: use len use len(st.session(st.session_state.all_state.all_chunks_chunks) instead) instead of summary of summary['total['total_chunks_chunks']
               ']
                st.success(
                    st.success(
                    f f""✅ Ready✅ Ready! Index! Indexed {ed {len(stlen(st.session_state.session_state.all_ch.all_chunks)}unks)} chunks, chunks, "
                    "
                    f"{ f"{summary['summary['unique_unique_entities']} uniqueentities']} unique entities, entities, "
                    "
                    f"{ f"{summary['summary['total_total_claims']claims']} claims} claims from { from {summary['summary['document_countdocument_count']}']} papers papers"
               "
                )
                if )
                if summary[' summary['consensusconsensus_topics_topics']']:
                    st:
                    st.caption.caption(f"(f"🔗🔗 Cross-d Cross-document consensusocument consensus available for available for: {: {', '.', '.join(sumjoin(summary['mary['consensusconsensus_topics_topics'][:'][:55])])}")
           }")
            else else:
                st:
                st.success(f.success(f"✅ Ready! Indexed"✅ Ready! Indexed {len {len(st.session(st.session_state.all_state.all_chunks_chunks)} chunks)} chunks")

")

                       st st.session.session_state.pro_state.processing_comcessing_completeplete = = True True

            return            True

        except return True

        except Exception as Exception as e e:
            st:
            st.error(f.error(f"Processing"Processing failed: failed: {e {e}")
           }")
            import trace import traceback
            stback
            st.error(t.error(tracebackraceback.format_ex.format_excc())
            return())
            return False


def render_chat_interface():
    False


def render_chat_interface():
    if not if not st.session st.session_state.get_state.get('vector('vectorstorestore'):
        st'):
        st.info(".info("👆👆 Upload documents Upload documents above to above to start chatting start chatting with cross with cross-document-document reasoning reasoning")
        return")
        return

   

    if st if st.session_state.session_state.ll.llm_tokenm_tokenizer isizer is None and None and st.session st.session_state._state.llmllm_model__model_choicechoice:
        backend:
        backend_type =_type = "oll "ollama"ama" if is if is_oll_ollama_modelama_model(st.session(st.session_state._state.llmllm_model__model_choice)choice) else " else "transformerstransformers"
       "
        with st with st.spinner.spinner(f"(f"LoadingLoading { {st.sessionst.session_state._state.llm_modelllm_model__choice}choice}..."):
..."            result):
            result = load = load_local__local_llmllm(st.session(st.session_state._state.llmllm_model__model_choice,choice, use_ use_4bit4bit=st=st.session_state.session_state.get('.get('use_use_4bit4bit_quant_quantization',ization', True True))
            token))
            tokenizer,izer, model, model, device_or device_or_host,_host, loaded_back loaded_backend =end = result result
            if
            if tokenizer tokenizer is not is not None or None or model is not None model is not None:
               :
                st.session st.session_state._state.llmllm_tokenizer_tokenizer = token = tokenizer
izer                st
                st.session_state.ll.session_state.llm_model = modelm_model = model
               
                st.session_state. st.session_state.llmllm_device_or_device_or_host =_host = device_or device_or_host_host

                st                st.session_state.session_state.ll.llm_backm_backend_typeend_type = loaded = loaded_backend_backend
               
                st.success st.success("✓("✓ Model loaded Model loaded!")
           !")
            else:
                st else.error("Failed to load model:
                st.error(". TryFailed to load model. Try selecting a selecting a different option different option.")
                return.")
               

    has return_model =

    has (
       _model = st.session (
       _state. st.sessionllm_state.llm_backend_type ==_backend "oll_type ==ama" "ollama" and st and st.session_state.session_state.ll.llm_modelm_model is not is not None None
    )
    ) or or (
        st (
        st.session_state.session_state.ll.llm_backm_backend_typeend_type == " == "transformerstransformers" and" and st.session st.session_state._state.llm_tokenllmizer_tokenizer is not is not None None
   
    )

    if )

    if not has not has_model_model:
        st.warning("Please:
        st.warning("Please select and select and load a load a model in model in the sidebar the sidebar first first")
        return")
        return

   

    for message for message in st in st.session_state.session_state.messages.messages:
       :
        with st with st.chat.chat_message(message["_message(messagerole"]["role"]):
            st.markdown):
            st.markdown(message["content(message["content"])
            if"])
            if message.get message.get("s("sources") and stources") and st.session_state.session_state.show_s.show_sourcesources:
                with:
                with st.exp st.expander("ander("📚📚 Retrieved Sources Retrieved Sources with Citations with Citations"):
                   "):
                    for i, src for i, src in enumerate in enumerate(message["(message["sources"], sources"], 11):
                        citation):
                        citation = src = src.metadata.metadata.get(".get("citation_dcitation_display",isplay", " "UnknownUnknown source source")
                        section =")
                        section = src src.metadata.metadata.get(".get("section",section", "UNKNOWN "UNKNOWN")
                        st")
                        st.markdown.markdown(f"(f"**[{**[{i}]i}]** {** {citation}citation} | * | *{section{section}*}*")
                       ")
                        bib = bib = src.m src.metadata.getetadata.get("bibli("bibliographicographic",", { {})
                        if})
                        if bib and bib and any(b any(bib.getib.get(k)(k) for for k k in [' in ['doi',doi', 'authors 'authors', '', 'journal',journal', 'year 'year']']):
                            with):
                            with st.exp st.expander("ander("🔍🔍 Bibliographic Bibliographic Details Details"):
                                if"):
                                if bib.get bib.get('doi('doi'):
                                   '):
                                    st.mark st.markdown(f"**down(fDOI:**"** `{DOI:**bib[' `{doi']bib['}`doi']")
                                if}`")
 bib.get                                if bib.get('authors'):
                                   ('authors'):
                                    st.mark st.markdown(fdown(f"**"**Authors:**Authors:** {', {', '.join '.join(bib(bib['authors['authors'][:'][:3])3])}{'...' if len}{'...' if len(bib(bib['authors['authors'])>'])>3 else3 else '' ''}")
                                if}")
                                if bib.get bib.get('journal('journal'):
                                   '):
                                    st.mark st.markdown(fdown(f"**"**Journal:**Journal:** {bib['journal'] {bib['journal']}")
                                if}")
                                if bib bib.get.get('year('year'):
                                   '):
                                    st.mark st.markdown(fdown(f"**"**Year:**Year:** {bib {bib['year['year']']}")
                        st}")
                        st.markdown.markdown(f">(f"> {src {src.page_content.page[:300_content]}...[:300]}...")

           ")

            if message if message.get(".get("reasoningreasoning_meta_meta") and") and st.session st.session_state.show_state.show_reason_reasoning_ing_chainchain and and message[" message["role"]role"] == " == "assistant":
               assistant":
                meta = meta = message[" message["reasoningreasoning_meta_meta"]
               "]
                with st with st.expander.expander("("🧠 Reasoning🧠 Reasoning Chain Chain"):
                    st"):
                    st.markdown.markdown(f"(f"**Query**Query entities detected entities detected:** {', '.:** {', '.join(join(meta.getmeta.get('query('query_entities_entities', []', [])) or)) or ' 'NoneNone''}")
                    st}")
                    st.markdown.markdown(f"**(f"Cross**Cross-document consensus found-document:** { consensus foundmeta.get:** {('consmeta.getensus_f('consound',ensus_f 0ound', 0)}")
                    st)}")
                    st.markdown.markdown(f"(f"**Cont**Contradictionsradictions detected:** detected:** {meta {meta.get('.get('contradcontradictions_fictions_found',ound', 0 0)}")
                    st)}")
                    st.markdown.markdown(f"(f"**Multi**Multi-hop expansion-hop expansion:** {':** {'Yes'Yes' if meta if meta.get('.get('multi_multi_hop_exphop_expansion')ansion else '') else 'No'No'}")
                   }")
                    if meta if meta.get('.get('relevancerelevance'):
                       '):
                        st.mark st.markdown(fdown(f"**"**Response relevanceResponse relevance:** {:** {meta['meta['relevancerelevance']:.']:.2f2f}/1}/1.0.0")

   ")

    if prompt if prompt := st := st.chat.chat_input("_input("Ask aboutAsk about laser parameters laser parameters, multic, multicomponentomponent alloys, alloys, digital twins digital twins, etc, etc."."):
        st):
        st.session_state.session_state.messages.messages.append({"role":.append({"role": "user "user", "content":", " promptcontent": prompt})
        with st.ch})
        withat_message st.ch("userat_message"):
           ("user st.mark"):
           down(p st.markromptdown(p)

        withrompt st.ch)

        with st.chat_messageat_message("ass("assistantistant"):
"):
            message            message_placeholder_placeholder = st = st.empty.empty()

            with()

            with st.sp st.spinner("inner("🔍🔍 Performing cross Performing cross-document-document reasoning..." reasoning..."):
                try):
                try:
                    answer:
                    answer, retrieved, retrieved_docs_docs, relevance, relevance, reasoning, reasoning_meta_meta = retrieve = retrieve_and_answer_and_answer(
                       (
                        vectorstore vectorstore=st=st.session_state.session_state.vector.vectorstorestore,
                        graph,
                        graph=st=st.session_state.session_state.know.knowledge_graphledge_graph,
                       ,
                        tokenizer tokenizer=st=st.session_state.session_state.ll.llm_tokenm_tokenizerizer,
                        model,
                        model=st=st.session_state.llm_model.session_state.llm_model,
                       ,
                        device_or device_or_host=_host=st.sessionst.session_state.llm_state.llm_device_or_device_or_host_host,
                        backend,
                        backend=st=st.session_state.session_state.ll.llm_modelm_model_choice_choice,
                       ,
                        backend_type backend_type=st=st.session_state.session_state.ll.llm_backm_backend_typeend_type,
                       ,
                        query= query=promptprompt,
                       ,
                        k= k=st.sessionst.session_state.max_state.max_ret_retrieved_chrieved_chunksunks
                   
                    )

                    reasoning )

                    reasoning_meta_meta["relevance"] = relevance["relevance"] = relevance

                   

                    display_text display_text = = ""
                    for ""
                    for word in word in answer.split answer.split():
                       ():
                        display_text display_text += word += word + " + " "
                        "
                        message_ message_placeholder.markplaceholder.markdown(down(display_textdisplay_text + " + "▌▌")
                       ")
                        time.sleep time.sleep(0.015(0)
                   .015 message_)
                   placeholder.mark message_down(answerplaceholder.mark)

                   down(answer st.session)

                   _state.m st.session_state.messages.append({
                        "roleessages.append({
                        "role": "": "assistantassistant",
                       ",
                        "content "content": answer": answer,
                       ,
                        "s "sources":ources": retrieved_d retrieved_docs ifocs if st.session st.session_state.show_state.show_sources else_sources None,
 else None                        "re,
                       levance": "relevance": relevance relevance,
                        ",
                        "reasoningreasoning_meta_meta": reasoning": reasoning_meta_meta
                   
                    })

                })

                except Exception except Exception as e as e:
                   :
                    error_msg error_msg = f = f""❌ Error❌ Error: {: {str(estr(e)[:)[:300]}300]}"
                   "
                    st.error st.error(error_msg(error_msg)
                   )
                    st.session st.session_state_state.m.messages.appendessages.append({"role({"role": "": "assistantassistant", "", "content":content": error_msg error_msg}})


def render)


def render_footer_footer():
   ():
    st.mark st.markdown("down("------")
    col")
    col1,1, col2 col2, col, col3 =3 = st.columns st.columns(3(3)

   )

    with col with col11:
        st:
        st.markdown.markdown("**("**📚📚 Example Questions:** Example Questions:**")
        st")
        st.caption.caption("•("• What is What is the effect the effect of composition of composition on I on IMC growthMC growth in Sn in Sn‑Ag‑Ag‑Cu‑Cu solders solders during laser during laser soldering soldering??")
        st.caption")
        st.caption("•("• How do How do multi‑ multi‑scale simulationsscale simulations predict grain predict grain structure in structure in SLM SLM of Al of Al‑Cr‑Cr‑Fe‑Fe‑Ni‑Ni alloys? alloys?")
        st.caption("• What")
        st.caption contradictions exist("• What contradictions exist regarding the regarding the influence influence of Marangoni of Marangoni convection on convection on porosity formation porosity formation?"?")

    with)

    with col2 col2:
       :
        st.mark st.markdown("down("****⚡ Reasoning⚡ Reasoning Tips:** Tips:**")
       ")
        st.caption(" st.caption("• Ask• Ask comparative questions comparative questions to trigger to trigger consensus detection consensus detection")
       ")
        st.c st.caption("aption("• Query• Query specific alloy specific alloy families (e.g., 'Sn‑ families (e.g., 'Sn‑Ag‑Ag‑Cu',Cu', 'Al 'AlCrFeCrFeNi')Ni') to activate to activate entity entity linking linking")
       ")
        st.c st.caption("aption("• Look for the• Look for the 🧠 Reasoning 🧠 Reasoning Chain expand Chain expander forer for transparency transparency")

    with")

    with col3 col3:
:
               st.mark st.markdown("down("****🔐 Privacy🔐 Privacy & Science & Science:**")
        st.caption("• All:**")
        st.caption("• All processing processing happens locally happens locally")
       ")
        st.c st.caption("aption("• Cross• Cross-document-document reasoning uses reasoning uses extracted entities extracted entities only only")
        st")
        st.caption.caption("•("• Uncertainty is Uncertainty is explicitly reported, explicitly reported never, never hidden hidden")


def main")


def main():
   ():
    st.set st.set_page_config_page_config(
(
               page_title page_title="="🔬 Laser🔬 Laser Microstructure RAG Microstructure RAG + Cross-Doc + Cross-Doc Reasoning Reasoning",
        page",
        page_icon_icon="="🔬🔬",
        layout",
        layout="wide="wide",
       ",
        initial_ initial_sidebar_statesidebar_state="exp="expanded"
anded"
       )

    st.markdown("""
    )

    st.markdown("""
    <style <style>
   >
    .main .main-header-header {
        font {
        font-size:-size: 2 2.5.5remrem;
        background;
        background: linear: linear-gradient(-gradient(90deg90deg, #, #1e1e40af40af, #, #7c7c3a3aed,ed, #059669 #059669);
        -);
        -webkit-backwebkit-background-clground-clip: text;
        -ip: text;
        -webkit-textwebkit-text-fill-fill-color:-color: transparent transparent;
        font;
        font-weight:-weight: 800 800;
       ;
        text-align text-align: center: center;
       ;
        padding: padding: 1 1rem rem 00;
   ;
    }
    .info-card }
    . {
       info-card background: {
        #f background:8f #f8fafafc;
        border-left: 4px solid #3b82fc6;
        border-left: 4px solid #3b82f;
        padding6: ;
        padding1rem: ;
       1rem;
        border-radius border-radius: : 0 0 0.0.5rem5rem 0 0.5.5rem rem 00;
        margin;
        margin:: 0. 0.5rem5rem 0 0;
   ;
    }
    }
    .reason .reasoning-bing-badgeadge {
        {
        display display: inline: inline-block-block;
        background: #dbe;
        background: #dbeafeafe;
        color;
        color: #: #1e1e40af40af;
       ;
        padding: padding: 0.2rem 0.6rem;
        border-radius 0.2rem 0.6rem;
        border-radius: : 0.25rem0.25rem;
       ;
        font-size font-size: : 0.85rem;
       0.85rem;
        margin: margin: 0 0.1.1rem rem 0.0.2rem2rem 0 0.1.1rem rem 00;
   ;
    }
    . }
    .consensusconsensus-badge-badge {
        {
        display: display: inline-block inline-block;
        background: #d1fae5;
        color:;
        background: #d1fae5;
        color: #065 #065f46f46;
       ;
        padding: padding: 0 0.2.2rem rem 0.0.6rem6rem;
       ;
        border-radius border-radius: : 0.0.25rem25rem;
;
               font-size font-size: : 0.0.8585remrem;
;
               margin: margin: 0 0.1.1rem rem 0.0.2rem2rem 0 0.1.1rem rem 00;
   ;
    }
    . }
    .contradcontradiction-biction-badgeadge {
        display {
        display: inline: inline-block-block;
        background;
        background: #: #fee2fee2e2e2;
       ;
        color color:: #991 #991b1b1bb;
        padding: 0.2rem 0;
        padding: 0.2rem 0.6.6remrem;
        border;
        border-radius:-radius: 0 0.25.25remrem;
        font;
        font-size:-size: 0 0.85.85remrem;
        margin;
        margin: : 0.0.1rem1rem 0 0.2.2rem rem 00.1rem.1rem 0 0;
   ;
    }
    }
    </style </style>
   >
    """, """, unsafe_ unsafe_allow_htmlallow_html=True=True)

)

    st.markdown    st.markdown('<h('<h1 class1 class="main="main-header">-header">🔬🔬 Laser Micro Laser Microstructure Rstructure RAG +AG + Cross-D Cross-Doc Reasoningoc Reasoning</h</h1>1>', unsafe', unsafe_allow_allow_html=True)
   _html=True)
    st.mark st.markdown("down(""""
   "
    <div style <div style="text="text-align:-align:center;center;color:#color:#6474864748b;margin-bottomb;:1margin-bottom.5:1rem.5rem">
">
    Upload    Upload research papers research papers on on multicom multicomponent alloysponent alloys and and laser processing laser processing, and, and get get <strong> <strong>scientscientifically rigorousifically rigorous answers</ answers</strong>strong> with with 
    
    <span class <span class="cons="consensus-bensus-badge">cross-dadge">ocument consensuscross-document consensus</span</span>,>, 
    
    < <span class="contspan class="contradictionradiction-badge-badge">cont">contradictionradiction detection</ detection</span>,span>, and and 
    
    <span class <span class="reason="reasoning-bing-badge">adge">multi-hopmulti-hop reasoning</ reasoning</span>span>.
   .
    </div </div>
   >
    """, """, unsafe_ unsafe_allow_htmlallow_html=True=True)

    initialize)

    initialize_session_state()
    render_session_state()
    render_sidebar_sidebar()

   ()

    if if st.session st.session_state._statellm.llm_model__model_choice andchoice and not is not is_oll_ollama_model(st.sessionama_model(st.session_state._state.llmllm_model__model_choicechoice):
        mem):
        mem_info_info = = estimate_model estimate_model_memory_memory(st.session(st.session_state._state.llmllm_model__model_choicechoice,, st st.session.session_state_state.get('use_.get('use_4bit4bit_quantization', True))
       _quantization', True))
        available_v available_vram =ram = get_available_g get_available_gpu_mpu_memory()
        ifemory()
        if available_v available_vram andram and not mem not mem_info['_info['cpu_cpu_ok']ok']:
           :
            required = required = float(m float(mem_infoem_info['vram_4bit['vram_4bit'].replace'].replace('GB('GB','').','').replace('replace('~','~','').strip').strip()) if()) if 'GB 'GB' in' in mem_info mem_info['v['vram_ram_4bit'] else 1004bit'] else 100
           
            if available if available_vram_vram < required < required:
               :
                st.mark st.markdown(fdown(f"""
               """
                <div <div style=" style="background:#background:#feffef3c3c7;7;border-leftborder-left:4:4px solid #fpx solid #f59e59e0b0b;padding;padding:0:0.75.75rem;rem;border-radiusborder-radius:0:0 0 0.5.5rem rem 0.0.55remrem 0 0;margin;margin::00.5.5rem 0rem 0">
               ">
                ⚠ ⚠️️ <strong <strong>>Memory WarningMemory Warning:</strong:</strong> {st.session> {st.session_state._state.llmllm_model__model_choice} requires ~choice} requires ~{mem{mem_info['_info['vramvram_4_4bit']bit']} VR} VRAMAM.
                You.
                You have ~{available_vram:.1f} have ~{available_vram:.1f}GB availableGB available.
               .
                </div </div>
                """,>
                """, unsafe_ unsafe_allow_htmlallow_html=True=True)

    col1,)

    col1, col2 col2 = st = st.columns.columns([1,([1, 2 2])

   ])

    with col with col11:
        uploaded:
        uploaded_files = render_d_files = render_document_ocument_uploaderuploader()

       ()

        if uploaded if uploaded_files and_files and st.button st.button("("🔄 Process🔄 Process Documents", Documents", type=" type="primary", use_container_width=True):
           primary", use_container_width=True):
            process_d process_documents(uocuments(uploadedploaded_files_files)

        if)

        if st.session st.session_state.pro_state.processing_comcessing_completeplete:
            st:
            st.success("✅ Knowledge base ready.success("✅ Knowledge base ready")
           ")
            if st if st.session_state.session_state.know.knowledge_graphledge_graph:
               :
                summary = summary = st.session st.session_state.k_state.knowledgenowledge_graph.get_graph.get_know_knowledge_sumledge_summarymary()
                #()
                # FIX FIX: use: use len(st len(st.session_state.all_ch.session_state.all_chunks)unks) instead of instead of summary summary['['totaltotal_chunks_chunks']
                st']
                st.caption.caption(f"(f"📦📦 {len {len(st.session(st.session_state.all_state.all_chunks_chunks)} chunks)} chunks | { | {summary['summary['unique_unique_entities']entities']} entities} entities | { | {summary['summary['total_total_claims']claims']} claims")
               } claims")
                if summary['top if summary['top_entities_entities']']:
                    st:
                    st.markdown.markdown("**("**Top entitiesTop entities:**:**")
                    for")
                    for ent, ent, count in count in summary[' summary['top_top_entitiesentities'][:5]:
                       '][:5]:
                        st.mark st.markdown(fdown(f''<span class="reason<span class="reasoning-bing-badge">{adge">{ent}ent} ({count ({count})</})</span>span>', unsafe', unsafe_allow_allow_html=True_html=True)
       )
        elif uploaded elif uploaded_files_files:
            st:
            st.warning.warning("⏳("⏳ Click ' Click 'Process DocumentsProcess Documents' to' to begin")
        else begin")
        else:
           :
            st.info st.info("📁 Upload("📁 Upload PDF/T PDF/TXTXT files files to start to start")

       ")

        if st if st.session_state.session_state.processed.processed_files_files:
            if:
            if st.button st.button("🗑️ Clear All("🗑️ Clear All", use", use_container_width_container_width=True=True):
                st):
                st.session_state.session_state.clear.clear()
                st.rer()
                st.rerunun()

    with()

    with col2 col2:
       :
        if st if st.session_state.session_state.processing.processing_complete_complete and st and st.session_state.vectorstore.session_state.vectorstore:
            render:
            render_chat_chat_interface_interface()
       ()
        else else:
            st:
            st.markdown.markdown(""("""
           "
            <div <div class="info class="info-card-card">
           ">
            <h <h3>3>👋 Welcome to👋 Welcome to Cross-D Cross-Document Scientificocument Scientific Reasoning!</h3 Reasoning!</h3>
           >
            < <pp>This>This assistant goes beyond assistant goes simple beyond simple retrieval:</ retrieval:</pp>
           >
            <ul <ul>
           >
            <li <li><strong>><strong>SemanticSemantic Chunking:</ Chunking:</strong>strong> Preserves Preserves Abstract/Methods/ Abstract/Methods/Results/DResults/Discussioniscussion structure</ structure</li>
           li>
            <li <li><><strong>strong>Entity ExtractionEntity Extraction:</strong> Ident:</strong> Identifies materialsifies materials, parameters, parameters, methods, methods automatically</ automatically</lili>
           >
            <li <li><strong>Cross-D><strong>ocument AlignmentCross-D:</strongocument Alignment> Links:</strong> Links the same the same entity across entity across different papers different papers</li</li>
           >
            <li <li><strong>Cons><strong>Consensus Detectionensus Detection:</strong> Statistically aggregates:</strong> Statistically aggregates values reported values reported in multiple in multiple papers</ papers</lili>
           >
            <li <li><strong>><strong>ContradContradiction Flagiction Flagging:</ging:</strong>strong> Highlights when Highlights when papers disagree papers disagree significantly</li significantly</li>
           >
            <li <li><strong>><strong>Multi-HMulti-Hop Retrieval:</op Retrieval:</strong>strong> Follows Follows entity links entity links to find to find related evidence related evidence</li</li>
           >
            <li><strong <li><strong>Un>Uncertaintycertainty Calibration Calibration:</strong:</strong> Explicit> Explicit confidence levels confidence levels in every in every answer</ answer</lili>
            </ul>
            </ul>
           >
            <p <p><strong>Getting started><strong>:</strongGetting started:</strong></p></p>
           >
            <ol <ol>
           >
            <li <li>Upload>Upload 2 2+ PDF/TXT+ PDF/TXT papers on papers on multicom multicomponent alloysponent alloys or laser or laser processing</ processing</lili>
           >
            <li> <li>Enable "Enable "Cross-dCross-document reasoningocument reasoning" in sidebar</li" in sidebar</li>
           >
            <li> <li>Ask comparativeAsk comparative or synthes or synthesizing questionsizing questions</li</li>
           >
            <li <li>Expand>Expand " "🧠 Reasoning🧠 Reasoning Chain" Chain" to see to see the logical steps</ the logical steps</lili>
            </>
            </olol>
            </div>
            </div>
            "">
            """, unsafe", unsafe_allow_allow_html=True_html=True)

           )

            st.mark st.markdown("down("**Try**Try asking:** asking:**")
           ")
            demo_q demo_qs =s = [
                [
                "What "What is the is the effect effect of laser of laser power on power on interfacial interfacial IMC IMC thickness in thickness in Sn‑ Sn‑Ag‑Cu/CAg‑Cu/Cu jointsu joints??",
                "",
                "Do theseDo these papers agree papers agree on the on the optimal hatch optimal hatch distance distance for defect for defect‑‑free LPfree LPBF ofBF of Al‑ Al‑Cr‑Fe‑Cr‑Fe‑Ni alloysNi alloys??",
                "",
                "SummarizeSummarize the phase the phase‑field‑field models used models used for simulating for simulating selective selective laser melting laser melting of multic of multicomponentomponent alloys alloys.",
                ".",
                "How doesHow does the the composition of composition of high entropy high entropy alloys affect alloys affect their their thermal thermal conductivity during conductivity during laser processing? laser processing?",
           ",
            ]
            for ]
            for q in q in demo_q demo_qss:
                if:
                if st.button st.button(f"(f"💬💬 {q {q}", use}", use_container_width_container_width=True,=True, key=f key=f"demo"demo_{q_{q[:20[:20]}"]}"):
                   ):
                    st.session_state.demo st.session__state.demoquestion_question = q = q

                    st.r                    st.rerunerun()

    render_f()

    render_footerooter()

    if()

    if hasattr hasattr(st.session(st.session_state,_state, 'demo 'demo_question_question') and') and st.session st.session_state.demo_state.demo_question_question:
       :
        st.session st.session_state.m_state.messages.appendessages.append({"({"rolerole": "user": "user", "content", "content": st": st.session_state.session_state.demo_.demo_questionquestion})
        del})
        del st.session st.session_state.demo_state.demo_question_question
       
        st.r st.rerunerun()


if()


if __name __name__ == "__main__ == "__main__":
   __":
    main main()
