#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
DECLARMIMA v20.0 - PURE PAGEINDEX AGENT (Agentic Navigation & Structural Reasoning)
================================================================================
v20.0 PURE PAGEINDEX ARCHITECTURE (Agentic Navigation & Structural Reasoning):
- IterativeTreeNavigator replaces single-shot retrieve_quantitative
- Agentic MCTS: LLM navigates document trees iteratively (drill_down / extract_text)
- Cross-Document Meta-Tree: stitches equivalent sections across all docs
- Layout-aware parsing: pymupdf4llm preserves Markdown #/##/### hierarchies and tables
- Table & Figure node types: tabular_data and figure_caption as first-class structural citizens
- Expanded schema: sketch_description, figure_caption, mermaid_diagram, ascii_sketch, figure_page
- 100% VECTORLESS: all sentence-transformers, t-SNE, UMAP, PCA purged
- Agentic Navigation replaces Search & Extract paradigm
- Unified Cross-Document Meta-Tree for simultaneous multi-doc comparison
- Layout-aware Markdown parsing with pymupdf4llm
- Table/figure nodes treated as structural citizens (no fragmentation)
"""
import streamlit as st
import json
import re
import torch
import requests
import os
import asyncio
import logging
import time
import hashlib
import textwrap
import math
import copy
import threading
import queue
import numpy as np
import pandas as pd
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any, Set, Callable, Literal
from collections import defaultdict, Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logging.basicConfig(level=logging.INFO, handlers=[console_handler], force=True)
logger = logging.getLogger("DECLARMIMA_EXTENDED")

# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

# ============================================================================
# PYMUPDF IMPORT (FIXED)
# ============================================================================
try:
    import pymupdf as fitz  # PyMuPDF >= 1.24
except ImportError:
    try:
        import fitz  # Older PyMuPDF versions
    except ImportError:
        st.error(
            "PyMuPDF is not installed.\n"
            "Install with:\n"
            "pip install pymupdf"
        )
        st.stop()

# ============================================================================
# PDF EXTRACTION (FIXED)
# ============================================================================
def extract_text_from_pdf(file_bytes: bytes, max_pages: int = None) -> list[dict]:
    """
    Extract text page-by-page using PyMuPDF.
    Compatible with modern PyMuPDF versions.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {e}")

    pages_data = []
    total_pages = len(doc)

    if max_pages is not None:
        total_pages = min(total_pages, max_pages)

    for i in range(total_pages):
        try:
            page = doc[i]
            text = page.get_text("text").strip()
            if text:
                pages_data.append({
                    "page_num": i + 1,
                    "text": text
                })
        except Exception:
            continue

    doc.close()
    return pages_data

# ============================================================================
# MODEL OPTIONS (Expanded with HF Small Models)
# ============================================================================
MODEL_OPTIONS = {
    # Hugging Face Models (Runs directly via Transformers, no API key needed)
    "🤗 HF: Qwen2.5-0.5B-Instruct (Ultra Fast, CPU OK)": "Qwen/Qwen2.5-0.5B-Instruct",
    "🤗 HF: Qwen2.5-1.5B-Instruct (Fast, Balanced)": "Qwen/Qwen2.5-1.5B-Instruct",
    "🤗 HF: SmolLM2-1.7B-Instruct (Lightweight)": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "🤗 HF: Llama-3.2-3B-Instruct (Compact Meta)": "meta-llama/Llama-3.2-3B-Instruct",
    "🤗 HF: Qwen2.5-7B-Instruct (Recommended, needs GPU)": "Qwen/Qwen2.5-7B-Instruct",
    # Ollama Models (Requires local Ollama service running)
    "🦙 Ollama: qwen2.5:7b (Recommended for RAG)": "ollama:qwen2.5:7b",
    "🦙 Ollama: qwen2.5:14b (Max Reasoning)": "ollama:qwen2.5:14b",
    "🦙 Ollama: llama3.1:8b (Meta Standard)": "ollama:llama3.1:8b",
    "🦙 Ollama: mistral:7b (High JSON Reliability)": "ollama:mistral:7b",
    "🦙 Ollama: falcon3:10b (Instruction Following)": "ollama:falcon3:10b",
}

MODEL_PROMPT_TEMPLATES = {
    "qwen2.5": {"system": "You are a precise document analyst. Follow JSON format strictly.", "json_reminder": "Return ONLY valid JSON."},
    "smollm": {"system": "You are a precise document analyst. Follow JSON format strictly.", "json_reminder": "Return ONLY valid JSON."},
    "llama": {"system": "You are a precise document analyst. Follow JSON format strictly.", "json_reminder": "Return ONLY valid JSON."},
    "mistral": {"system": "You are a precise document analyst. Follow JSON format strictly.", "json_reminder": "Return ONLY valid JSON."},
    "falcon": {"system": "You are a precise document analyst. Follow JSON format strictly.", "json_reminder": "Return ONLY valid JSON."},
    "default": {"system": "You are a document navigation agent.", "json_reminder": "Return valid JSON only."}
}

def get_model_template(model_name: str):
    model_name_lower = model_name.lower()
    for key, template in MODEL_PROMPT_TEMPLATES.items():
        if key in model_name_lower:
            return template
    return MODEL_PROMPT_TEMPLATES["default"]

# ============================================================================
# HYBRID LLM ENGINE
# ============================================================================
class HybridLLM:
    def __init__(self, model_key: str, use_4bit: bool = True, device: Optional[str] = None):
        self.model_key = model_key
        self.use_4bit = use_4bit
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = None
        self.model_name = None
        self.client = None
        self.tokenizer = None
        self.model = None
        
        if model_key.startswith("ollama:"):
            self.model_name = model_key.replace("ollama:", "", 1)
            self._init_ollama()
        else:
            self.model_name = model_key
            self.backend = "transformers"
            
        self.template = get_model_template(self.model_name)
        logger.info(f"HybridLLM initialized: {self.model_name} on {self.device} via {self.backend}")

    def _init_ollama(self):
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama python package not installed. Run: pip install ollama")
        try:
            requests.get("http://localhost:11434/api/tags", timeout=5)
            self.backend = "ollama"
            self.client = ollama.Client(host="http://localhost:11434")
        except Exception as e:
            raise RuntimeError(f"Cannot connect to Ollama at localhost:11434. Is it running? Error: {e}")

    def generate(self, prompt: str, max_new_tokens=1024, temperature=0.1, fast_json=False, system_prompt=None):
        if self.backend == "ollama":
            return self._ollama_generate(prompt, max_new_tokens, temperature, fast_json, system_prompt)
        else:
            return self._transformers_generate(prompt, max_new_tokens, temperature, system_prompt)

    def _ollama_generate(self, prompt, max_tokens, temp, fast_json, system_prompt):
        try:
            options = {"temperature": temp, "num_predict": max_tokens}
            if fast_json:
                options["format"] = "json"
            messages = []
            sys = system_prompt or self.template.get("system")
            if sys:
                messages.append({"role": "system", "content": sys})
            messages.append({"role": "user", "content": prompt})
            resp = self.client.chat(model=self.model_name, messages=messages, options=options, stream=False)
            return resp.get("message", {}).get("content", "").strip()
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return f"Error: {str(e)[:100]}"

    def _transformers_generate(self, prompt, max_tokens, temp, system_prompt):
        if not TRANSFORMERS_AVAILABLE:
            return "Error: transformers not installed. Please run: pip install transformers bitsandbytes accelerate"
        if self.tokenizer is None:
            self._load_transformers()
        if not self.model:
            return "Error: model not loaded"
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_tokens, 
                    temperature=temp if temp > 0 else None, 
                    do_sample=temp > 0, 
                    pad_token_id=self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "assistant" in response:
                response = response.split("assistant")[-1].strip()
            return response
        except Exception as e:
            logger.error(f"Transformers error: {e}")
            return f"Error: {str(e)[:100]}"

    def _load_transformers(self):
        st.info(f"📥 Loading {self.model_name} on {self.device}... (This may take a minute)")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32}
        if self.use_4bit and self.device == "cuda":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if self.device == "cuda" and not self.use_4bit:
            self.model.to(self.device)
        self.model.eval()
        st.success("✅ Model loaded successfully!")

# ============================================================================
# PYDANTIC MODELS & SCHEMAS
# ============================================================================
from pydantic import BaseModel, Field, field_validator

class UniversalExtractionItem(BaseModel):
    item_type: Literal["quantitative", "qualitative", "definition", "comparison", "relationship",
                       "process", "material", "method", "equation", "tabular_data", "figure_caption",
                       "phase_field", "molecular_dynamics", "plasticity", "thermal", "mechanical", 
                       "microstructural", "electrochemical", "multiphysics", "ai_ml", "digital_twin", "informatics"]
    content: str
    parameter_name: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    physical_quantity: Optional[str] = None
    material: Optional[str] = None
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object_val: Optional[str] = None
    definition_term: Optional[str] = None
    definition_text: Optional[str] = None
    comparison_entities: List[str] = []
    comparison_aspect: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    context: str
    equation_latex: Optional[str] = None
    model_name: Optional[str] = None
    variables_defined: Optional[Dict[str, str]] = {}
    doc_source: str
    page: int
    section_title: Optional[str] = None
    method: Optional[str] = None
    conditions: Dict[str, Any] = {}
    reasoning_trace: str = ""
    multiphysics_context: Optional[str] = None
    simulation_type: Optional[str] = None
    mesh_size: Optional[float] = None
    timestep: Optional[float] = None
    boundary_conditions: Optional[str] = None
    heat_source_type: Optional[str] = None
    alloy_system: Optional[str] = None
    phase_fraction: Optional[float] = None
    sfe_value: Optional[float] = None
    usfe_value: Optional[float] = None
    lewis_number: Optional[float] = None
    jackson_parameter: Optional[float] = None

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        return max(0.0, min(1.0, v))

    def citation(self) -> str:
        return f'<cite doc="{self.doc_source}" page="{self.page}"/>'

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

class ExtractedValue(BaseModel):
    query: str
    value: float
    unit: str
    physical_quantity: str
    parameter_name: Optional[str] = None
    material: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    context: str
    equation_latex: Optional[str] = None
    model_name: Optional[str] = None
    variables_defined: Optional[Dict[str, str]] = {}
    doc_name: str
    page: int
    section_title: Optional[str] = None
    simulation_context: Optional[str] = None
    method_origin: Optional[str] = None
    temperature_dependent: bool = False
    phase: Optional[str] = None

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
    uncertainty_metrics: Optional[Dict[str, float]] = None
    xai_attributions: Optional[Dict[str, Any]] = None
    microstructural_summary: Optional[Dict[str, Any]] = None
    electrochemical_summary: Optional[Dict[str, Any]] = None

    def to_json(self):
        return json.dumps(self.model_dump(), indent=2, ensure_ascii=False, default=str)

class DocumentMetadata(BaseModel):
    doc_name: str
    alloys: List[str] = []
    laser_power_values: List[float] = []
    scan_speed_values: List[float] = []
    energy_density_values: List[float] = []
    layer_thickness_values: List[float] = []
    yield_strength_values: List[float] = []
    tensile_strength_values: List[float] = []
    hardness_values: List[float] = []
    elongation_values: List[float] = []
    temperature_values: List[float] = []
    melting_temperature_values: List[float] = []
    phase_fraction_values: List[float] = []
    grain_size_values: List[float] = []
    porosity_values: List[float] = []
    relative_density_values: List[float] = []
    corrosion_potential_values: List[float] = []
    corrosion_current_density_values: List[float] = []
    polarization_resistance_values: List[float] = []
    phase_field_iterations: Optional[int] = None
    md_steps: Optional[int] = None
    calphad_database: Optional[str] = None
    heat_source_type: Optional[str] = None
    ai_model_used: Optional[str] = None
    digital_twin_active: bool = False
    process_types: List[str] = []
    other_parameters: Dict[str, List[float]] = {}

@dataclass
class QueryContext:
    query: str
    relevant_doc_ids: Set[str]
    physical_quantities: List[str]
    materials: List[str]
    extracted_values: List[ExtractedValue]
    retrieved_nodes: List[Dict]
    multiphysics_flags: List[str] = field(default_factory=list)
    electrochemical_flags: List[str] = field(default_factory=list)
    ai_ml_flags: List[str] = field(default_factory=list)
    microstructural_features: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_cache(cls, cache: Dict) -> 'QueryContext':
        raw_vals = cache.get("extracted_values", [])
        extracted_vals = [ExtractedValue(**v) for v in raw_vals] if raw_vals and isinstance(raw_vals[0], dict) else []
        return cls(
            query=cache.get("prompt", ""),
            relevant_doc_ids={v.doc_name for v in extracted_vals},
            physical_quantities=list({v.physical_quantity for v in extracted_vals if v.physical_quantity}),
            materials=list({v.material for v in extracted_vals if v.material}),
            extracted_values=extracted_vals,
            retrieved_nodes=cache.get("retrieved", []),
            multiphysics_flags=cache.get("multiphysics_flags", []),
            electrochemical_flags=cache.get("electrochemical_flags", []),
            ai_ml_flags=cache.get("ai_ml_flags", []),
            microstructural_features=cache.get("microstructural_features", [])
        )

    def has_data(self) -> bool:
        return len(self.extracted_values) > 0

# ============================================================================
# PHYSICAL QUANTITY CLASSIFIER & CONCEPT NORMALIZER
# ============================================================================
class PhysicalQuantityClassifier:
    CANONICAL = {
        "laser_power": ["laser power", "laser beam power", "laser output power", "power", "p", "beam power"],
        "scan_speed": ["scan speed", "scanning speed", "laser scan speed", "beam scan speed", "scan velocity", "v_scan", "vs", "travel speed"],
        "temperature": ["temperature", "melting temperature", "annealing temperature", "reflow temperature", "solution annealing", "stress relief"],
        "energy_density": ["energy density", "volumetric energy density", "ved", "laser fluence", "J/mm3"],
        "yield_strength": ["yield strength", "ys", "0.2% offset strength", "proof stress", "yield stress", "Rp0.2"],
        "tensile_strength": ["tensile strength", "uts", "ultimate tensile strength", "ultimate strength", "Rm"],
        "hardness": ["hardness", "vickers hardness", "microhardness", "hv", "nano hardness", "HRC"],
        "elongation": ["elongation", "strain", "ductility", "strain to failure", "reduction of area"],
        "modulus": ["young's modulus", "elastic modulus", "stiffness", "e-modulus", "E"],
        "phase_fraction": ["phase fraction", "volume fraction", "austenite fraction", "ferrite fraction", "martensite fraction"],
        "grain_size": ["grain size", "average grain size", "cell size", "subgrain size", "dendrite arm spacing", "d_g"],
        "porosity": ["porosity", "pore fraction", "void fraction", "relative porosity"],
        "relative_density": ["relative density", "density ratio", "packing density", "compactness"],
        "stacking_fault_energy": ["stacking fault energy", "sfe", "gsfe", "generalized stacking fault energy", "γ_SFE"],
        "corrosion_potential": ["corrosion potential", "e_corr", "ecorr", "corrosion potential ecorr", "open circuit potential", "e_ocp", "eocp", "OCP"],
        "corrosion_current_density": ["corrosion current density", "j_corr", "jcorr", "corrosion current", "i_corr", "Icorr"],
        "polarization_resistance": ["polarization resistance", "r_p", "rp", "apparent polarization resistance", "rp_app", "R_p"],
        "phase_field_method": ["phase field method", "pfm", "phase-field", "cahn-hilliard", "allen-cahn", "PFM"],
        "molecular_dynamics": ["molecular dynamics", "md", "lammps", "atomistic simulation", "gsfe"],
        "digital_twin": ["digital twin", "vdt", "virtual twin", "real-time twin", "conditional automation"],
        "pinn": ["physics-informed neural network", "pinn", "physics-informed ml", "pinns"],
        "unet": ["u-net", "unet", "convolutional unet", "segmentation unet", "U-Net"],
        "convlstm": ["convlstm", "conv-lstm", "spatiotemporal lstm", "sequence prediction"],
        "calphad": ["calphad", "thermocalc", "pycalphad", "thermodynamic database", "tdb", "CALPHAD"],
        "xai": ["explainable ai", "xai", "shap", "lime", "feature attribution"],
        "uncertainty_quantification": ["uncertainty quantification", "uq", "confidence calibration", "error propagation"],
        "unknown": ["unknown", "other", "miscellaneous", "not classified"]
    }

    def __init__(self):
        self._build_keyword_index()

    def _build_keyword_index(self):
        self.keyword_to_canonical = {}
        for canonical, keywords in self.CANONICAL.items():
            for kw in keywords:
                self.keyword_to_canonical[kw.lower()] = canonical

    def classify(self, parameter_name: Optional[str], unit: Optional[str], context: str) -> str:
        if parameter_name:
            pname_lower = parameter_name.lower().strip()
            if pname_lower in self.keyword_to_canonical:
                return self.keyword_to_canonical[pname_lower]
        context_lower = context.lower()
        for canonical, keywords in self.CANONICAL.items():
            for kw in keywords:
                if kw in context_lower:
                    return canonical
        return "unknown"

    def get_human_readable(self, canonical: str) -> str:
        mapping = {
            "laser_power": "Laser Power", "scan_speed": "Scan Speed", "temperature": "Temperature",
            "energy_density": "Energy Density (VED)", "yield_strength": "Yield Strength", "tensile_strength": "Tensile Strength",
            "hardness": "Hardness", "elongation": "Elongation", "modulus": "Young's Modulus",
            "phase_fraction": "Phase Fraction", "grain_size": "Grain Size", "porosity": "Porosity",
            "relative_density": "Relative Density", "stacking_fault_energy": "Stacking Fault Energy",
            "corrosion_potential": "Corrosion Potential", "corrosion_current_density": "Corrosion Current Density",
            "polarization_resistance": "Polarization Resistance", "phase_field_method": "Phase Field Method (PFM)",
            "molecular_dynamics": "Molecular Dynamics (MD)", "digital_twin": "Digital Twin Framework",
            "pinn": "Physics-Informed Neural Network (PINN)", "unet": "U-Net Architecture",
            "convlstm": "ConvLSTM Sequence Model", "calphad": "CALPHAD Thermodynamic Database",
            "xai": "Explainable AI (XAI)", "uncertainty_quantification": "Uncertainty Quantification (UQ)",
            "unknown": "Other Quantities"
        }
        return mapping.get(canonical, canonical.replace("_", " ").title())

class ConceptNormalizer:
    ALIAS_DICTIONARIES = {
        "yield_strength": ["yield strength", "ys", "0.2% proof", "proof stress", "yield stress", "0.2% offset strength", "σ_y", "Rp0.2"],
        "tensile_strength": ["tensile strength", "uts", "ultimate tensile strength", "ultimate strength", "tensile stress", "σ_uts", "Rm"],
        "laser_power": ["laser power", "laser beam power", "laser output power", "beam power", "P"],
        "scan_speed": ["scan speed", "scanning speed", "laser scan speed", "beam scan speed", "scan velocity", "v_scan", "vs"],
        "hardness": ["hardness", "vickers hardness", "microhardness", "hv", "nano hardness", "HRC"],
        "lpbf": ["lpbf", "l-pbf", "laser powder bed fusion", "selective laser melting", "slm", "laser powder-bed fusion", "pbf-lb"],
        "ded": ["ded", "directed energy deposition", "direct energy deposition", "laser metal deposition", "lmd"],
        "fem": ["fem", "finite element method", "finite element analysis", "fea", "finite element"],
        "md": ["md", "molecular dynamics", "molecular dynamics simulation", "molecular dynamics (md)", "atomistic", "LAMMPS"],
        "phase_field": ["phase field method", "pfm", "phase-field", "cahn-hilliard", "allen-cahn", "phase field", "PFM"],
        "calphad": ["calphad", "thermocalc", "pycalphad", "thermodynamic database", "tdb", "CALPHAD", "Thermo-Calc"],
        "pinn": ["physics-informed neural network", "pinn", "physics-informed ml", "pinns", "PINN"],
        "unet": ["u-net", "unet", "convolutional unet", "segmentation unet", "U-Net"],
        "convlstm": ["convlstm", "conv-lstm", "spatiotemporal lstm", "sequence prediction", "ConvLSTM"],
        "digital_twin": ["digital twin", "vdt", "virtual twin", "real-time twin", "conditional automation", "DT"],
        "xai": ["explainable ai", "xai", "shap", "lime", "feature attribution", "XAI"],
        "uncertainty_quantification": ["uncertainty quantification", "uq", "confidence calibration", "error propagation", "UQ"],
        "unknown": ["unknown", "other", "miscellaneous", "not classified"]
    }

    def __init__(self):
        self._build_reverse_index()

    def _build_reverse_index(self):
        self.alias_to_canonical: Dict[str, str] = {}
        for canonical, aliases in self.ALIAS_DICTIONARIES.items():
            for alias in aliases:
                self.alias_to_canonical[alias.lower()] = canonical

    def normalize(self, term: str) -> str:
        if not term or not str(term).strip():
            return "unknown"
        term_lower = str(term).lower().strip()
        if term_lower in self.alias_to_canonical:
            return self.alias_to_canonical[term_lower]
        for alias, canonical in sorted(self.alias_to_canonical.items(), key=lambda x: -len(x[0])):
            if alias in term_lower:
                return canonical
        return term_lower

# ============================================================================
# ITERATIVE TREE NAVIGATOR (v20.0)
# ============================================================================
class IterativeTreeNavigator:
    def __init__(self, llm: "HybridLLM", max_steps: int = 4):
        self.llm = llm
        self.max_steps = max_steps

    async def navigate_and_retrieve(self, query: str, trees: List[Dict]) -> List[Dict]:
        current_nodes = self._get_nodes_at_level(trees, level=1)
        final_chunks = []
        for step in range(self.max_steps):
            if not current_nodes:
                break
            if len(current_nodes) > 25:
                logger.warning(f"MCTS Context Overflow: {len(current_nodes)} nodes. Truncating to top 25.")
                scored_nodes = []
                for n in current_nodes:
                    summary_lower = n.get('summary', '').lower()
                    score = sum(1 for kw in [r'\b\d*\s*w\b', r'\b\d*\s*kw\b', r'\bmm/s\b', r'\bmpa\b', r'\bj/mm\b', r'\b\u00b0c\b', r'\b\d*\s*k\b', r'\b%\b'] if re.search(kw, summary_lower))
                    scored_nodes.append((score, n))
                scored_nodes.sort(key=lambda x: x[0], reverse=True)
                current_nodes = [n for _, n in scored_nodes[:25]]

            prompt = self._build_mcts_prompt(query, current_nodes, step)
            response = await asyncio.to_thread(self.llm.generate, prompt, fast_json=True)
            actions = self._parse_actions(response)

            if not actions and step == 0:
                logger.warning("⚠️ MCTS JSON failed on Step 1. Falling back to top 3 summary scores.")
                top_nodes = sorted(current_nodes, key=lambda x: len(x.get('summary', '')), reverse=True)[:3]
                for n in top_nodes:
                    chunk = self._get_full_text(trees, n['node_id'])
                    if chunk:
                        final_chunks.append(chunk)
                        break
                break

            if not actions:
                break

            next_level_nodes = []
            for action in actions:
                node_id = action.get('node_id')
                action_type = action.get('action')
                if action_type == 'drill_down':
                    children = self._get_children(trees, node_id)
                    next_level_nodes.extend(children)
                elif action_type == 'extract_text':
                    chunk = self._get_full_text(trees, node_id)
                    if chunk:
                        final_chunks.append(chunk)
            current_nodes = next_level_nodes
        return final_chunks

    def _get_nodes_at_level(self, trees: List[Dict], level: int) -> List[Dict]:
        nodes = []
        for tree in trees:
            doc_id = tree.get('doc_id', tree.get('doc_name', 'unknown'))
            self._collect_at_level(tree, level, doc_id, nodes)
        return nodes

    def _collect_at_level(self, node: Dict, target_level: int, doc_id: str, result: List[Dict]):
        node_level = self._infer_level(node)
        if node_level == target_level:
            result.append({
                'node_id': node.get('node_id', ''),
                'doc_id': doc_id,
                'title': node.get('title', ''),
                'summary': node.get('summary', '')[:200],
                'level': node_level
            })
        for child in node.get('nodes', []):
            self._collect_at_level(child, target_level, doc_id, result)

    def _infer_level(self, node: Dict) -> int:
        node_id = node.get('node_id', '')
        if '.' not in node_id:
            return 0
        return node_id.count('.')

    def _get_children(self, trees: List[Dict], node_id: str) -> List[Dict]:
        children = []
        for tree in trees:
            doc_id = tree.get('doc_id', tree.get('doc_name', 'unknown'))
            node = self._find_node_by_id(tree, node_id)
            if node:
                for child in node.get('nodes', []):
                    children.append({
                        'node_id': child.get('node_id', ''),
                        'doc_id': doc_id,
                        'title': child.get('title', ''),
                        'summary': child.get('summary', '')[:200],
                        'level': self._infer_level(child)
                    })
        return children

    def _get_full_text(self, trees: List[Dict], node_id: str) -> Optional[Dict]:
        for tree in trees:
            doc_id = tree.get('doc_id', tree.get('doc_name', 'unknown'))
            node = self._find_node_by_id(tree, node_id)
            if node:
                text = node.get('text', '')
                if not text:
                    text = node.get('summary', '')
                return {
                    'full_text': text[:20000],
                    'page_start': node.get('start_index', 1),
                    'doc_id': doc_id,
                    'section_title': node.get('title', ''),
                    'quantitative_items': node.get('quantitative_items', []),
                    'citation': f'<cite doc="{doc_id}" page="{node.get("start_index", 1)}"/>',
                    'node_id': node_id
                }
        return None

    def _find_node_by_id(self, tree: Dict, node_id: str) -> Optional[Dict]:
        if tree.get('node_id') == node_id:
            return tree
        for child in tree.get('nodes', []):
            result = self._find_node_by_id(child, node_id)
            if result:
                return result
        return None

    def _build_mcts_prompt(self, query, nodes, step):
        nodes_summary = json.dumps([{
            "node_id": n['node_id'],
            "doc_id": n['doc_id'],
            "title": n['title'],
            "summary": n.get('summary', '')[:150]
        } for n in nodes], indent=2)
        return f"""You are an expert scientific navigator. You are at Step {step+1} of searching a document tree.
QUERY: "{query}"
CURRENT AVAILABLE NODES:
{nodes_summary}
INSTRUCTIONS:
1. Analyze the titles and summaries.
2. If a node's summary indicates it contains the answer or specific data, choose "extract_text".
3. If a node is a parent section (e.g., "3. Results") and you need more detail, choose "drill_down".
4. Ignore irrelevant nodes.
Return JSON:
{{
"reasoning": "Brief thought process",
"actions": [
{{"node_id": "...", "action": "drill_down"}},
{{"node_id": "...", "action": "extract_text"}}
]
}}"""

    def _parse_actions(self, response: str) -> List[Dict]:
        try:
            md_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if md_match:
                data = json.loads(md_match.group(1))
            else:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    clean_json = re.sub(r',\s*([\]}])', r'\1', json_match.group())
                    data = json.loads(clean_json)
                else:
                    logger.error(f"MCTS returned NO JSON. Raw response: {response[:300]}")
                    return []
            actions = data.get('actions', [])
            return [a for a in actions if isinstance(a, dict) and 'node_id' in a and 'action' in a]
        except json.JSONDecodeError as e:
            logger.error(f"MCTS JSON Parse Error: {e} | Raw response: {response[:300]}")
            return []
        except Exception as e:
            logger.error(f"MCTS Unexpected Error: {e}")
            return []

# ============================================================================
# SCIENTIFIC INTENT ROUTER
# ============================================================================
class ScientificIntentRouter:
    VALUE_TRIGGERS = [
        r"\bvalue\b", r"\bnumber\b", r"\bamount\b", r"\bquantity\b",
        r"\bhow much\b", r"\bhow many\b", r"\bwhat is the \w+ of\b",
        r"\bfind the\b.*\b(power|speed|strength|temperature|pressure|density|modulus|hardness)\b",
        r"\b\d+\s*(?:W|kW|mW|mm/s|MPa|GPa|K|°C|J/mm|µm|nm|mJ/m²)",
        r"\btable\b.*\b(value|data|parameter)\b",
        r"\blist all\b.*\b(values|numbers|parameters)\b",
        r"\bparameter\b.*\b(value|list|table)\b",
        r"\bextract\b.*\b(number|value|data)\b",
        r"\bwhat\b.*\b(value|number|parameter)\b.*\bis\b",
        r"\bcompare\b.*\b(value|parameter|data)\b.*\bacross\b",
        r"\bwatts\b", r"\bkilowatts\b", r"\bkW\b", r"\bmm/s\b", r"\bmpa\b",
        r"\bdescribe\b.*\b(power|speed|strength|temperature|pressure|density|modulus|hardness|current|voltage)\b",
        r"\bwhat\s+(?:is|are)\s+the\b.*\b(power|speed|strength|temperature|parameters?)\b",
        r"\bhollomon\b", r"\bramberg[- ]osgood\b", r"\bcahn[- ]hilliard\b",
        r"\bbutler[- ]volmer\b", r"\bnavier[- ]stokes\b",
        r"\bparameters?\b.*\b(of|for)\b", r"\bcoefficient\b", r"\bexponent\b", r"\bconstant\b",
        r"\bflow\b.*\b(stress|curve)\b", r"\bhardening\b.*\b(exponent|rate|coefficient)\b",
        r"\bstrength\b.*\b(coefficient|parameter)\b",
    ]
    INTENT_PATTERNS = {
        "equation": [r"\bequation\b", r"\bformula\b", r"\bgoverning\b", r"\bconstitutive\b", r"\bnavier[- ]stokes\b", r"\bboussinesq\b", r"\bmarangoni\b", r"\bmomentum\s+equation\b", r"\bcontinuity\s+equation\b", r"\benergy\s+equation\b", r"\bcahn[- ]hilliard\b", r"\bbutler[- ]volmer\b", r"\bhollomon\b", r"\bramberg[- ]osgood\b", r"\bderive\b", r"\bmathematical model\b", r"\bpde\b", r"\bpartial\s+differential\b", r"\bgoverning\s+equation\b", r"\bconstitutive\s+model\b", r"\bfluid\s+flow\s+equation\b", r"\bheat\s+equation\b", r"\bdiffusion\s+equation\b", r"\bschrodinger\b", r"\bmaxwell\b", r"\bfick\b", r"\bfourier\b", r"\barrhenius\b", r"\bjohnson[- ]cook\b", r"\bvoce\b", r"\bludwik\b", r"\bswift\b"],
        "constitutive": [r"\bhollomon\b", r"\bramberg[- ]osgood\b", r"\bludwik\b", r"\bswift\b", r"\bvoce\b", r"\bjohnson[- ]cook\b", r"\bconstitutive\s+(model|law|relation|equation|parameters?)\b", r"\bflow\s+stress\b", r"\bhardening\s+(law|model|rule)\b", r"\byield\s+(criterion|surface|function)\b", r"\bparameters?\b.*\b(hollomon|ramberg|osgood|ludwik|swift|voce|johnson)\b", r"\b(strength\s+coefficient|hardening\s+exponent|strain\s+hardening)\b"],
        "mechanism": [r"\bwhy\b", r"\bhow does\b", r"\bmechanism\b", r"\bexplain\b", r"\bcause\b", r"\bdriving force\b", r"\bphysical process\b", r"\bfluid flow\b", r"\bmelt[- ]?pool\s+(dynamics|behavior|regime|formation)\b", r"\bporosity\s+formation\b", r"\bgrain\s+growth\b", r"\bsolidification\b", r"\bphase\s+transformation\b", r"\brecrystallization\b", r"\bnucleation\b", r"\bgrowth\b", r"\bsegregation\b", r"\bdiffusion\s+mechanism\b", r"\bcorrosion\s+mechanism\b", r"\bfatigue\s+mechanism\b", r"\bfracture\s+mechanism\b", r"\bwear\s+mechanism\b", r"\bthermal\s+cycle\b", r"\bresidual\s+stress\b", r"\bwhat\s+happens\b", r"\bwhat\s+is\s+the\s+reason\b", r"\bdescribe\s+the\s+process\b"],
        "comparison": [r"\bcompare\b", r"\bvs\.?\b", r"\bversus\b", r"\bdifference\b", r"\bacross all\b", r"\bwhich is better\b", r"\bcontrast\b", r"\bwhich\b.*\b(higher|lower|stronger|weaker|better|worse)\b", r"\brelative\b.*\b(performance|strength|behavior)\b", r"\bsimilarities\b", r"\bdifferences\b"],
        "sketch_diagram": [r"\bsketch\b", r"\bdiagram\b", r"\bdraw\b", r"\bvisualize\b", r"\bschematic\b", r"\bflowchart\b", r"\billustrate\b", r"\bfigure\b.*\b(show|depict|display|present)\b", r"\bshow\b.*\b(process|setup|structure|microstructure)\b", r"\bprocess\s+flow\b", r"\bsetup\s+diagram\b", r"\bmicrostructure\s+image\b", r"\bschematic\s+diagram\b", r"\bblock\s+diagram\b", r"\bphase\s+diagram\b"],
        "boolean": [r"\bis it\b", r"\bdoes\b", r"\bdo\b", r"\bare there\b", r"\bcan\b", r"\bwas\b", r"\bwere\b", r"\bhas\b", r"\bhave\b", r"\bis there\b", r"\bare they\b", r"\bmentioned\b"],
        "definition": [r"\bwhat is\b", r"\bwhat are\b", r"\bdefine\b", r"\bmeaning of\b", r"\bdefinition of\b", r"\bdescribe\b", r"\btell me about\b", r"\bexplain\b.*\bwhat\b"],
    }

    def route(self, query: str) -> Dict[str, Any]:
        q_lower = query.lower()
        has_value_trigger = any(re.search(p, q_lower) for p in self.VALUE_TRIGGERS)
        intent = "open_query"
        output_format = "prose"
        for i_type, patterns in self.INTENT_PATTERNS.items():
            if any(re.search(p, q_lower) for p in patterns):
                intent = i_type
                break
        if has_value_trigger:
            intent = "value_extraction"
            output_format = "table"
        elif intent == "constitutive":
            output_format = "constitutive_hybrid"
        elif intent == "equation":
            output_format = "latex"
        elif intent == "sketch_diagram":
            output_format = "mermaid_or_ascii"
        elif intent == "comparison":
            output_format = "contrastive_table"
        elif intent == "mechanism":
            output_format = "causal_chain"
        elif intent == "boolean":
            output_format = "yes_no_evidence"
        elif intent == "definition":
            output_format = "prose"

        weights = {
            "metadata": 0.8 if intent in ("value_extraction", "constitutive") else 0.15,
            "keyword": 0.5 if intent in ("value_extraction", "constitutive") else 0.25,
            "semantic": 0.2 if intent in ("value_extraction", "constitutive") else 0.7,
            "tree_search": 0.4 if intent in ("value_extraction", "constitutive") else 0.95,
            "equation_boost": 0.6 if intent in ("equation", "constitutive") else 0.0,
            "figure_boost": 0.5 if intent == "sketch_diagram" else 0.0,
            "mechanism_boost": 0.5 if intent == "mechanism" else 0.0,
        }
        return {
            "intent": intent,
            "output_format": output_format,
            "weights": weights,
            "value_extraction_gated": has_value_trigger,
            "focus_instruction": self._build_focus(intent, query, has_value_trigger)
        }

    def _build_focus(self, intent, query, has_value_trigger):
        if intent == "value_extraction" and has_value_trigger:
            return "EXPLICIT VALUE REQUEST: Extract exact numerical values with units. Build a structured table. Prioritize quantitative data."
        elif intent == "constitutive":
            return "CONSTITUTIVE MODEL REQUEST: Extract BOTH governing equations (LaTeX) AND numerical parameters (values with units). For temperature-dependent parameters, extract polynomial expressions. Define all variables. Build a hybrid table with equations and parameter values."
        elif intent == "equation":
            return "EQUATION REQUEST: Extract governing equations in LaTeX. Define all variables. Do NOT extract unrelated numbers. Focus on mathematical models and constitutive relations."
        elif intent == "mechanism":
            return "MECHANISM REQUEST: Explain the physical process qualitatively. Cite causal relationships. Numbers are secondary. Focus on 'why' and 'how'."
        elif intent == "sketch_diagram":
            return "VISUAL REQUEST: Extract figure descriptions, schematics, and process flows. Generate Mermaid diagram or ASCII art. Describe visual elements."
        elif intent == "comparison":
            return "COMPARISON REQUEST: Compare entities across documents. Highlight differences and similarities. Use contrastive format."
        elif intent == "boolean":
            return "BOOLEAN REQUEST: Verify presence/absence of information. Provide yes/no with evidence."
        elif intent == "definition":
            return "DEFINITION REQUEST: Provide conceptual explanation. Include definitions, context, and related concepts."
        else:
            return f"OPEN QUERY: Provide a comprehensive answer combining concepts, equations, and data as relevant to: {query}"

# ============================================================================
# UNIVERSAL LLM EXTRACTOR
# ============================================================================
class UniversalLLMExtractor:
    EXTRACTION_PROMPT = """Extract ALL quantitative information, mathematical models, qualitative mechanisms, figure descriptions, and visual elements relevant to the query from these document sections.
QUERY: {query}
QUERY TYPE: {query_type}
{dynamic_focus}
SECTIONS:
{sections_text}
Return JSON array of extracted items with fields:
{{
"item_type": "quantitative|qualitative|definition|comparison|relationship|process|material|method|equation|phase_field|molecular_dynamics|plasticity|thermal|mechanical|microstructural|electrochemical|multiphysics|ai_ml|digital_twin|informatics|sketch_description|figure_caption|tabular_data",
"content": "exact phrase with full numerical value (never truncate numbers)",
"confidence": 0.0-1.0,
"context": "exact sentence from text",
"doc_source": "{doc_id}",
"page": page_number,
"parameter_name": "...",
"value": number,
"unit": "e.g., W, kW, mW, mm/s, MPa, GPa, HV, mV, V, µA/cm², A/cm², J/mm³, J/mm², J/m, mJ/m², nm, µm, mm, K, °C, wt%, at%, vol%, g/cm³, kg/m³, W/m·K, Pa·s, mPa·s, kΩ·cm², ppm, unitless, iterations, steps, fs, ps, ns, ms, s, μm, mm, cm, m",
"physical_quantity": "one of: laser_power, electrical_power, scan_speed, flow_speed, feed_rate, irradiance, temperature, melting_temperature, energy_density, areal_energy_density, linear_energy_density, layer_thickness, spot_size, exposure_time, enthalpy, viscosity, thermal_conductivity, density, yield_strength, tensile_strength, ultimate_tensile_strength, hardness, elongation, modulus, stacking_fault_energy, unstable_stacking_fault_energy, ideal_shear_strength, corrosion_potential, pitting_potential, breakdown_potential, repassivation_potential, open_circuit_potential, corrosion_current_density, polarization_resistance, apparent_polarization_resistance, current_density, PREN, phase_fraction, austenite_fraction, ferrite_fraction, grain_size, cell_size, porosity, relative_density, surface_roughness, sauter_mean_diameter, spray_penetration, plume_height, film_thickness, absorption_coefficient, youngs_modulus, poisson_ratio, coefficient_thermal_expansion, lewis_number, jackson_parameter, meltpool_depth, meltpool_width, hatch_distance, rotation_angle, work_hardening_rate, hollomon_strength, hollomon_exponent, ramberg_osgood_k, ramberg_osgood_n, plasticity_model, phase_field_method, molecular_dynamics, digital_twin, pinn, unet, convlstm, calphad, xai, uncertainty_quantification, bimodal_microstructure, martensitic_transformation, eigenstrain, marangoni_effect, boussinesq_approximation, lead_lag_dynamics, positional_time_lag, solute_clustering, grain_boundary_energy, diffuse_interface_width, common_tangent, phase_stability, unknown",
"material": "alloy or material name if mentioned (e.g., Ti3Au, CP Ti, Grade II Ti, SDSS 2507, UNS S32750, AlSiMgZr, Al-Si-Mg-Zr, TiB2/Al-Si-Mg-Zr, Fe-based metallic glass, Au-Ti, 316L, 2205, Inconel 718, Ti6Al4V, CoCrNi, nt-Cu, HEA/MPEA)",
"method": "e.g., LPBF, L-PBF, DED, SLM, PFI, GDI, FEM, MD, nanoindentation, EIS, CPP, XRD, SEM, TEM, EBSD, EDS, DTA, CALPHAD, PINN, U-Net, ConvLSTM, Digital Twin, Phase Field, Tucker Decomposition, TF-IDF, PMI, NER",
"simulation_type": "type of simulation if mentioned (e.g., phase-field, MD, FEM, PINN, U-Net, ConvLSTM, CALPHAD, digital twin)",
"multiphysics_context": "context describing coupled physics if mentioned (e.g., thermal-mechanical, electrochemical-thermal, marangoni-boussinesq)",
"mesh_size": "mesh or grid size if specified",
"timestep": "simulation timestep if specified",
"boundary_conditions": "boundary conditions if specified",
"equation_latex": "LaTeX string of any governing equation (e.g., r\"\\sigma = K \\varepsilon^n\")",
"model_name": "Name of constitutive model or PDE (e.g., Hollomon, Cahn-Hilliard, Butler-Volmer, Navier-Stokes)",
"variables_defined": {{}} JSON object mapping variable symbols to descriptions,
"figure_description": "Raw caption or descriptive text about any figure, diagram, or sketch mentioned in the section",
"mermaid_diagram": "Generated Mermaid syntax representing the process, structure, or flow described (if applicable)",
"ascii_sketch": "ASCII art representation of the figure/diagram (if Mermaid is insufficient)",
"figure_page": "Page number where the original figure appears"
}}
CRITICAL RULES:
1. Capture ALL numbers with units.
2. For electrochemical: map Ecorr/Erp/Epit/Ebr to corrosion_potential/pitting_potential/etc.
3. For LPBF/DED: capture VED, AED, LED, hatch distance, layer thickness, laser power, scan speed.
4. NEVER truncate numbers.
5. If an alloy or material name appears, create an item with item_type="material", content=the name, material=the name.
6. Return ONLY valid JSON, no extra text.
7. Set confidence based on clarity.
8. If a mathematical equation, constitutive model, or governing PDE is present, set item_type="equation".
9. Extract the equation into the "equation_latex" field using standard LaTeX syntax.
10. Identify the "model_name".
11. Define the variables in "variables_defined" as a JSON object.
12. For qualitative mechanism queries, extract causal paragraphs verbatim into the "context" field and set item_type="qualitative".
13. If the text contains a Markdown table, set item_type="tabular_data" and paste the EXACT raw Markdown table in "content".
14. If the text contains a figure caption, set item_type="figure_caption" and paste the full caption text in "content".
Return [] if no relevant information found."""

    def __init__(self, llm: "HybridLLM"):
        self.llm = llm
        self.phys_classifier = PhysicalQuantityClassifier()
        self.concept_normalizer = ConceptNormalizer()

    def extract_from_chunks(self, chunks: List[Dict], query: str, query_analysis: Optional[Dict] = None) -> List[UniversalExtractionItem]:
        if not chunks:
            return []
        qa = query_analysis or {"query_type": "mixed", "keywords": [], "focus_instruction": ""}
        items = []
        for chunk in chunks:
            text = chunk["full_text"]
            doc = chunk["doc_id"]
            page = chunk["page_start"]
            if qa.get("query_type") == "quantitative" and not re.search(r'\d+', text):
                continue
            prompt = self.EXTRACTION_PROMPT.format(
                query=query,
                query_type=qa.get("intent", qa.get("query_type", "mixed")),
                sections_text=text[:4000],
                doc_id=doc,
                dynamic_focus=qa.get("focus_instruction", "")
            )
            try:
                response = self.llm.generate(prompt, max_new_tokens=2048, fast_json=True)
                json_str = self._extract_json(response)
                if json_str:
                    data = json.loads(json_str)
                    if isinstance(data, list):
                        items_data = data
                    elif isinstance(data, dict):
                        if "items" in data and isinstance(data["items"], list):
                            items_data = data["items"]
                        elif "item_type" in data:
                            items_data = [data]
                        else:
                            items_data = []
                    else:
                        items_data = []
                    for item_data in items_data:
                        if "content" not in item_data or not item_data["content"]:
                            item_data["content"] = item_data.get("context", "No content extracted")
                        if "context" not in item_data or not item_data["context"]:
                            item_data["context"] = item_data.get("content", "")[:300]
                        if "doc_source" not in item_data:
                            item_data["doc_source"] = doc
                        if "page" not in item_data:
                            item_data["page"] = page
                        else:
                            try:
                                item_data["page"] = int(str(item_data["page"]).replace("page", "").replace("p", "").strip())
                            except:
                                item_data["page"] = page
                        conf = item_data.get("confidence", 0.5)
                        if isinstance(conf, str):
                            conf = 0.8 if "high" in conf.lower() else 0.5 if "med" in conf.lower() else 0.2
                        try:
                            item_data["confidence"] = float(max(0.0, min(1.0, conf)))
                        except:
                            item_data["confidence"] = 0.5
                        valid_types = ["quantitative", "qualitative", "definition", "comparison", "relationship",
                                       "process", "material", "method", "equation", "tabular_data", "figure_caption",
                                       "phase_field", "molecular_dynamics", "plasticity", "thermal", "mechanical",
                                       "microstructural", "electrochemical", "multiphysics", "ai_ml", "digital_twin", "informatics"]
                        if item_data.get("item_type") not in valid_types:
                            item_data["item_type"] = "quantitative" if item_data.get("value") else "qualitative"
                        if "physical_quantity" not in item_data or not item_data["physical_quantity"]:
                            item_data["physical_quantity"] = self.phys_classifier.classify(item_data.get("parameter_name"), item_data.get("unit"), item_data.get("context", ""))
                        item_data.setdefault("material", None)
                        item_data.setdefault("multiphysics_context", None)
                        item_data.setdefault("simulation_type", None)
                        item_data.setdefault("mesh_size", None)
                        item_data.setdefault("timestep", None)
                        item_data.setdefault("boundary_conditions", None)
                        if item_data.get("physical_quantity"):
                            item_data["physical_quantity"] = self.concept_normalizer.normalize(item_data["physical_quantity"])
                        if item_data.get("material"):
                            item_data["material"] = self.concept_normalizer.normalize(item_data["material"])
                        try:
                            item = UniversalExtractionItem(**item_data)
                            if doc not in item.context:
                                item.context = f"[{doc}] {item.context}"
                            if item.page == 0:
                                item.page = page
                            items.append(item)
                        except Exception as e:
                            logger.debug(f"Item parse error: {e} | Data: {item_data}")
            except Exception as e:
                logger.error(f"Extraction error: {e}")
        unique = {}
        for i in items:
            key = (i.content, i.doc_source, i.page, i.material)
            if key not in unique or i.confidence > unique[key].confidence:
                unique[key] = i
        return [i for i in unique.values() if i.confidence >= 0.55]

    def _extract_json(self, text: str) -> Optional[str]:
        md_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if md_match:
            try:
                json.loads(md_match.group(1))
                return md_match.group(1)
            except:
                pass
        for pattern in [r'\{.*\}', r'\[.*\]']:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_str = match.group(0)
                json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
                try:
                    json.loads(json_str)
                    return json_str
                except:
                    continue
        return None

# ============================================================================
# HIERARCHICAL TREE RETRIEVER
# ============================================================================
class HierarchicalTreeRetriever:
    MATH_PATTERN = re.compile(r'\\[a-zA-Z]+|[$][$].*?[$][$]|[$][^$]+[$]|[=^_{}]|(?:sigma|epsilon|delta|theta|lambda|alpha|beta|gamma|omega)\s*[=<>]', re.IGNORECASE)

    def __init__(self, llm: "HybridLLM", max_results=30, max_text_chars=20000):
        self.llm = llm
        self.max_results = max_results
        self.max_text_chars = max_text_chars
        self._condensed_cache: Dict[str, Dict] = {}
        self.template = llm.template if hasattr(llm, 'template') else MODEL_PROMPT_TEMPLATES["default"]

    async def retrieve_quantitative(self, query: str, annotated_trees: List[Dict]) -> List[Dict]:
        qa = {"intent": "open_query"}
        if any(kw in query.lower() for kw in ["equation", "formula", "pde", "navier", "stokes", "momentum", "continuity", "governing"]):
            qa["intent"] = "equation"
        self._current_query_intent = qa.get("intent", "open_query")
        trees_json = []
        for tree in annotated_trees:
            doc_id = tree.get("doc_id", "unknown")
            if doc_id not in self._condensed_cache:
                self._condensed_cache[doc_id] = self._condense_tree(tree)
            trees_json.append(self._condensed_cache[doc_id])
        batches = self._batch_trees(trees_json, max_tokens=6000)
        all_selections = []
        for batch in batches:
            prompt = self._build_tree_search_prompt(query, batch)
            response = await asyncio.to_thread(self.llm.generate, prompt, max_new_tokens=2048, fast_json=True, system_prompt=self.template.get("system"))
            selections = self._parse_node_selections(response)
            all_selections.extend(selections)
        results = []
        for sel in sorted(all_selections, key=lambda x: x.get('confidence', 0), reverse=True):
            doc_id = sel.get('doc_id')
            node_id = sel.get('node_id')
            node = self._find_node_by_id(annotated_trees, doc_id, node_id)
            if node:
                full_text = node.get('text', '')
                max_chars_for_node = self.max_text_chars
                if self.MATH_PATTERN.search(full_text):
                    max_chars_for_node = self.max_text_chars * 2
                parent = self._find_parent_node(annotated_trees, doc_id, node_id)
                if parent:
                    sibling_texts = []
                    for sibling in parent.get('nodes', []):
                        if sibling.get('node_id') != node_id:
                            s_text = sibling.get('text', '')
                            if s_text:
                                sibling_texts.append(s_text[:2000])
                    if sibling_texts:
                        full_text = full_text + "\n[EXPANDED CONTEXT]:\n" + "\n".join(sibling_texts)
                if len(full_text) > max_chars_for_node:
                    full_text = full_text[:max_chars_for_node] + "..."
                results.append({
                    "full_text": full_text,
                    "page_start": node.get('start_index'),
                    "doc_id": doc_id,
                    "section_title": node.get('title'),
                    "quantitative_items": node.get('quantitative_items', []),
                    "citation": f'<cite doc="{doc_id}" page="{node.get("start_index")}"/>',
                    "selection_reasoning": sel.get('reasoning', ''),
                    "confidence": sel.get('confidence', 0)
                })
        return results[:self.max_results]

    def _condense_tree(self, tree: Dict, max_depth: int = 3) -> Dict[str, Any]:
        def condense(node: Dict, depth: int = 0) -> Dict[str, Any]:
            if depth > max_depth:
                return {"node_id": node.get("node_id", ""), "title": node.get("title", ""), "leaf": True}
            raw_summary = node.get("summary", "") or ""
            if hasattr(self, '_current_query_intent') and self._current_query_intent == "equation":
                math_preserved = re.sub(r'(Eq\.\s*\(\d+\)|\$\$.*?\$\$|\*[^*]+_\{[^}]+\})', r' [MATH:\1] ', raw_summary)
                result = {"node_id": node.get("node_id", ""), "title": node.get("title", ""), "summary": math_preserved[:250]}
            else:
                result = {"node_id": node.get("node_id", ""), "title": node.get("title", ""), "summary": raw_summary[:150]}
            if node.get("metadata"):
                meta = node["metadata"]
                if meta.get("alloys"): result["alloys"] = meta["alloys"][:3]
                if meta.get("laser_power_values"): result["power_hint"] = f"{min(meta['laser_power_values'])}-{max(meta['laser_power_values'])} W"
                if meta.get("scan_speed_values"): result["speed_hint"] = f"{min(meta['scan_speed_values'])}-{max(meta['scan_speed_values'])} mm/s"
                if meta.get("lewis_number_values"): result["le_hint"] = f"{min(meta['lewis_number_values'])}-{max(meta['lewis_number_values'])}"
                if meta.get("meltpool_depth_values"): result["melt_depth"] = f"{min(meta['meltpool_depth_values'])}-{max(meta['meltpool_depth_values'])} μm"
                if meta.get("phase_field_iterations"): result["pfm_iters"] = meta["phase_field_iterations"]
                if meta.get("md_steps"): result["md_steps"] = meta["md_steps"]
            q_items = node.get("quantitative_items", [])
            if q_items:
                params = list(set(item.get("parameter_name", "") for item in q_items if item.get("parameter_name")))
                if params: result["has_quantitative"] = params[:5]
            else:
                text = node.get("text", "")
                if text:
                    candidates = re.findall(r'(\d+(?:\.\d+)?)\s*(W|kW|mW|J|mm/s|C|K|MPa|GPa|nm|um|mm|s|m/s|W/cm2|kW/cm2|mJ/m2|unitless|iterations|steps|fs|ps|ns|ms)', text, re.IGNORECASE)
                    if candidates: result["candidate_values"] = [f"{v}{u}" for v, u in candidates[:3]]
            children = node.get("nodes", [])
            if children and depth < max_depth:
                result["nodes"] = [condense(c, depth+1) for c in children[:5]]
            return result
        return {"doc_id": tree.get("doc_id", tree.get("doc_name", "unknown")), "doc_name": tree.get("doc_name", ""), "structure": [condense(tree)] if not isinstance(tree, list) else [condense(t) for t in tree]}

    def _batch_trees(self, trees: List[Dict], max_tokens: int = 6000) -> List[List[Dict]]:
        batches = []
        current = []
        current_len = 0
        for t in trees:
            t_len = len(json.dumps(t))
            if current_len + t_len > max_tokens and current:
                batches.append(current)
                current = [t]
                current_len = t_len
            else:
                current.append(t)
                current_len += t_len
        if current:
            batches.append(current)
        return batches

    def _build_tree_search_prompt(self, query: str, trees: List[Dict]) -> str:
        trees_json = json.dumps(trees, ensure_ascii=False, indent=2)
        return f"""You are an expert scientific document navigator.
Given a query about quantitative parameters, identify which document nodes are MOST likely to contain the answer.
QUERY: {query}
INSTRUCTIONS:
1. Analyze each document's tree structure (titles, summaries, quantitative hints, candidate values, alloys, power hints, speed hints, lewis number hints, meltpool depth hints, PFM/MD simulation hints)
2. Select nodes that likely contain specific numerical values, parameters, or measurements
3. For cross-document queries like "laser power across all papers", select nodes from MULTIPLE documents
4. Prefer nodes with "has_quantitative" or "candidate_values" hints matching the query topic
5. Return selections sorted by confidence (highest first)
DOCUMENT TREES:
{trees_json}
Return JSON:
{{
"thinking": "Brief reasoning...",
"selections": [
{{"doc_id": "...", "node_id": "...", "reasoning": "...", "confidence": 0.95}}
]
}}
{self.template.get('json_reminder', 'Return ONLY valid JSON.')}
Include up to {self.max_results} selections."""

    def _parse_node_selections(self, response: str) -> List[Dict]:
        try:
            data = self._extract_json_safe(response)
            if data and isinstance(data, dict):
                selections = data.get("selections", [])
                return [s for s in selections if isinstance(s, dict) and "doc_id" in s and "node_id" in s]
        except Exception as e:
            logger.warning(f"Failed to parse selections: {e}")
        return []

    def _extract_json_safe(self, text: str) -> Optional[Any]:
        patterns = [r'\{.*\}', r'\[.*\]', r'```json\s*(\{.*?\})\s*```']
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_str = match.group(1) if match.groups() else match.group(0)
                try:
                    return json.loads(json_str)
                except:
                    continue
        return None

    def _find_node_by_id(self, trees: List[Dict], doc_id: str, node_id: str) -> Optional[Dict]:
        for tree in trees:
            if tree.get("doc_id") == doc_id or tree.get("doc_name") == doc_id:
                return self._search_node_recursive(tree, node_id)
        return None

    def _search_node_recursive(self, node: Dict, target_id: str) -> Optional[Dict]:
        if node.get("node_id") == target_id:
            return node
        for child in node.get("nodes", []):
            res = self._search_node_recursive(child, target_id)
            if res: return res
        return None

    def _find_parent_node(self, trees: List[Dict], doc_id: str, child_node_id: str) -> Optional[Dict]:
        for tree in trees:
            if tree.get("doc_id") == doc_id or tree.get("doc_name") == doc_id:
                result = self._search_parent_recursive(tree, child_node_id)
                if result:
                    return result
        return None

    def _search_parent_recursive(self, node: Dict, target_id: str) -> Optional[Dict]:
        for child in node.get("nodes", []):
            if child.get("node_id") == target_id:
                return node
            res = self._search_parent_recursive(child, target_id)
            if res:
                return res
        return None

    def build_cross_document_meta_tree(self, query: str, trees: List[Dict]) -> Dict:
        intent_prompt = f'''To answer "{query}", which structural section of a scientific paper is most relevant?
(e.g., 'Experimental Setup', 'Results', 'Methodology', 'Introduction').
Return ONLY the section name as a string.'''
        try:
            target_section = self.llm.generate(intent_prompt, max_new_tokens=20).strip().strip('"')
        except Exception:
            target_section = "Results"
        meta_root = {
            "node_id": "meta_root",
            "title": f"Cross-Document Meta-Tree: {target_section}",
            "doc_id": "META",
            "nodes": []
        }
        for tree in trees:
            doc_id = tree.get('doc_id', tree.get('doc_name', 'unknown'))
            matching_node = self._find_section_by_keyword(tree, target_section)
            if matching_node:
                meta_root["nodes"].append({
                    "node_id": f"meta_{doc_id}_{matching_node.get('node_id', '')}",
                    "title": f"[{doc_id}] {matching_node.get('title', '')}",
                    "summary": matching_node.get('summary', ''),
                    "doc_id": doc_id,
                    "original_node_id": matching_node.get('node_id', ''),
                    "text": matching_node.get('text', ''),
                    "nodes": matching_node.get('nodes', [])
                })
        return meta_root

    def _find_section_by_keyword(self, tree: Dict, keyword: str) -> Optional[Dict]:
        core_keywords = [k for k in keyword.lower().split() if len(k) > 3]
        if not core_keywords:
            core_keywords = [keyword.lower()]
        synonym_map = {
            "experimental": ["experiment", "methodology", "methods", "setup", "procedure"],
            "results": ["result", "finding", "outcome", "data", "measurement"],
            "methodology": ["method", "methods", "experimental", "procedure", "setup"],
            "introduction": ["intro", "background", "overview"],
            "discussion": ["discuss", "analysis", "interpretation"],
            "conclusion": ["conclude", "summary", "final"],
            "abstract": ["summary", "overview"],
            "materials": ["material", "sample", "specimen", "alloy"],
            "properties": ["property", "mechanical", "thermal", "electrochemical"],
        }
        expanded_keywords = set(core_keywords)
        for kw in core_keywords:
            for canonical, syns in synonym_map.items():
                if kw in syns or kw == canonical:
                    expanded_keywords.update(syns)
                    expanded_keywords.add(canonical)
        expanded_keywords = list(expanded_keywords)
        def search(node):
            title_lower = node.get('title', '').lower()
            if any(kw in title_lower for kw in expanded_keywords):
                return node
            for child in node.get('nodes', []):
                res = search(child)
                if res:
                    return res
            return None
        return search(tree)

# ============================================================================
# ADAPTIVE RESPONSE GENERATOR
# ============================================================================
class AdaptiveResponseGenerator:
    FORMATTERS = {
        "table": "_format_as_value_comparison",
        "latex": "_format_as_equations",
        "constitutive_hybrid": "_format_as_constitutive_hybrid",
        "causal_chain": "_format_as_mechanism",
        "contrastive_table": "_format_as_value_comparison",
        "mermaid_or_ascii": "_format_as_visual",
        "prose": "_format_as_prose",
        "yes_no_evidence": "_format_as_boolean",
        "standard": "_format_as_prose",
        "tabular_data": "_format_as_table_data",
    }

    def __init__(self, llm: "HybridLLM"):
        self.llm = llm
        self.phys_classifier = PhysicalQuantityClassifier()

    def generate(self, query, verified_items, query_analysis):
        fmt = query_analysis.get("output_format", "prose")
        formatter_name = self.FORMATTERS.get(fmt, "_format_as_prose")
        formatter = getattr(self, formatter_name)
        return formatter(query, verified_items, query_analysis)

    def _format_as_equations(self, query, items, qa):
        eq_items = [i for i in items if i.item_type == "equation" and i.equation_latex]
        if not eq_items:
            eq_items = [i for i in items if i.equation_latex]
        if not eq_items:
            return f"## Governing Equations\nNo governing equations found for '{query}'. The documents may not contain explicit mathematical formulations. Try rephrasing or check if the equations are embedded in figures/tables."
        lines = ["## Governing Equations\n"]
        seen_models = set()
        for item in eq_items:
            model_key = (item.model_name or "Unknown", item.equation_latex)
            if model_key in seen_models:
                continue
            seen_models.add(model_key)
            lines.append(f"### {item.model_name or 'Governing Equation'}")
            lines.append(f"$$ {item.equation_latex} $$")
            if item.variables_defined:
                vars_str = ", ".join(f"${k}$: {v}" for k, v in item.variables_defined.items())
                lines.append(f"*Where: {vars_str}*")
            if item.context:
                lines.append(f"\n**Context:** {item.context[:300]}")
            lines.append(f"\n@@CITE:doc={item.doc_source};page={item.page}@@\n")
        return "\n".join(lines)

    def _format_as_constitutive_hybrid(self, query, items, qa):
        eq_items = [i for i in items if i.item_type == "equation" and i.equation_latex]
        param_items = [i for i in items if i.value is not None and i.parameter_name and any(kw in (i.parameter_name or "").lower() for kw in ["hollomon", "ramberg", "osgood", "strength_coefficient", "hardening_exponent", "strain_hardening", "ludwik", "swift", "voce", "johnson", "cook", "flow_stress"])]
        content_items = [i for i in items if i.value is not None and any(kw in i.content.lower() for kw in ["hollomon", "ramberg", "osgood", "strength coefficient", "hardening exponent", "strain hardening", "ludwik", "swift", "voce", "johnson-cook", "flow stress"])]
        seen_keys = set()
        merged_params = []
        for item in param_items + content_items:
            key = (item.content, item.doc_source, item.page)
            if key not in seen_keys:
                seen_keys.add(key)
                merged_params.append(item)
        if not eq_items and not merged_params:
            fallback_items = [i for i in items if any(kw in (i.content or "").lower() for kw in ["hollomon", "ramberg", "osgood", "strength", "hardening"])]
            if fallback_items:
                return self._format_as_prose(query, fallback_items, qa) + "\n*[Note: Constitutive parameters found but not structured as equations or values.]*"
            return self._format_as_prose(query, items, qa) + "\n*[Note: No constitutive model equations or parameters found for this query.]*"
        eq_block = []
        if eq_items:
            for item in eq_items:
                eq_block.append(f"Model: {item.model_name or 'Unknown'}")
                eq_block.append(f"LaTeX: {item.equation_latex}")
                if item.variables_defined:
                    vars_str = ", ".join(f"{k}: {v}" for k, v in item.variables_defined.items())
                    eq_block.append(f"Variables: {vars_str}")
                eq_block.append(f"Source: @@CITE:doc={item.doc_source};page={item.page}@@")
                eq_block.append("")
        param_block = []
        if merged_params:
            for item in sorted(merged_params, key=lambda x: x.confidence, reverse=True)[:20]:
                pq = self.phys_classifier.get_human_readable(item.physical_quantity or "unknown")
                val = f"{item.value:.4g}" if item.value is not None else (item.content[:60] if item.content else "N/A")
                param_block.append(f"- {pq}: {val} {item.unit or ''} (conf={item.confidence:.2f}) @@CITE:doc={item.doc_source};page={item.page}@@")
        eq_text = "\n".join(eq_block) if eq_block else "No equations extracted."
        param_text = "\n".join(param_block) if param_block else "No parameters extracted."
        prompt = f"""You are a scientific analyst. You MUST follow this EXACT output structure. Do NOT deviate.
QUERY: {query}
EXTRACTED EQUATIONS:
{eq_text}
EXTRACTED PARAMETERS:
{param_text}
REQUIRED OUTPUT STRUCTURE (follow exactly):
## Constitutive Model: {query}
### Governing Equations
[If equations exist, restate them in $$ ... $$ format with all variables defined]
[If no equations, state "No explicit governing equations found in the corpus."]
### Parameter Values
[Create a markdown table with columns: Parameter | Value | Unit | Source]
[EVERY row MUST end with @@CITE:doc=filename;page=N@@]
### Physical Interpretation
[2-3 sentences explaining what these parameters mean physically]
### Confidence & Limitations
[1 sentence on data quality, temperature dependence, or gaps]
RULES:
- EVERY parameter row MUST have an @@CITE:doc=filename;page=N@@ tag
- Do NOT output any text before "## Constitutive Model"
- Do NOT add sections not listed above
- If parameters are temperature-dependent, note the temperature range"""
        return self.llm.generate(prompt, max_new_tokens=2048, temperature=0.05)

    def _format_as_visual(self, query, items, qa):
        sketch_items = [i for i in items if i.item_type in ["sketch_description", "figure_caption", "process", "material"]]
        lines = ["## Visual Representation\n"]
        for item in sketch_items:
            if item.mermaid_diagram:
                lines.append(f"### Process Diagram\n```mermaid\n{item.mermaid_diagram}\n```")
            if item.ascii_sketch:
                lines.append(f"```\n{item.ascii_sketch}\n```")
            if item.figure_description:
                lines.append(f"**Description:** {item.figure_description}")
            if item.content and len(item.content) > 20:
                lines.append(f"**Content:** {item.content[:400]}")
            lines.append(f"@@CITE:doc={item.doc_source};page={item.page}@@\n")
        if len(lines) == 1:
            prompt = f"""The user asked: '{query}'
Based on the following evidence from documents, generate a Mermaid diagram or ASCII sketch that visualizes the key concepts, processes, or structures described.
EVIDENCE:
{chr(10).join([f"- {i.content[:300]} (from {i.doc_source}, p.{i.page})" for i in items[:10]])}
Generate either:
1. A Mermaid flowchart/diagram in ```mermaid blocks
2. An ASCII art representation
3. A detailed verbal description of what the figure would show
Include citations in @@CITE:doc=X;page=Y@@ format."""
            response = self.llm.generate(prompt, max_new_tokens=2048, temperature=0.15)
            lines.append(response)
        return "\n".join(lines)

    def _format_as_prose(self, query, items, qa):
        if not items:
            return f"No relevant information found for '{query}'. Try rephrasing your question or uploading additional documents."
        evidence_summary = []
        for i in items[:25]:
            cite = f"@@CITE:doc={i.doc_source};page={i.page}@@"
            if i.item_type == "equation" and i.equation_latex:
                evidence_summary.append(f"[EQUATION] {i.model_name or 'Equation'}: {i.equation_latex} {cite}")
            elif i.value is not None:
                pq = i.physical_quantity or "value"
                evidence_summary.append(f"[DATA] {pq}: {i.value} {i.unit or ''} | {i.content[:150]} {cite}")
            else:
                evidence_summary.append(f"[TEXT] {i.item_type}: {i.content[:200]} {cite}")
        evidence = "\n".join(evidence_summary)
        prompt = f"""You are a scientific analyst. You MUST follow this EXACT output structure. Do NOT deviate.
QUERY: {query}
INTENT: {qa.get('intent', 'open_query')}
EVIDENCE:
{evidence}
REQUIRED OUTPUT STRUCTURE (follow exactly):
## Direct Answer
[2-3 sentence comprehensive answer synthesizing the evidence above]
## Key Findings
- [Finding 1 with @@CITE:doc=filename;page=N@@ citation]
- [Finding 2 with @@CITE:doc=filename;page=N@@ citation]
- [Finding 3 with @@CITE:doc=filename;page=N@@ citation]
## Confidence & Limitations
[1 sentence on data quality or gaps]
RULES:
- EVERY factual claim MUST have an @@CITE:doc=filename;page=N@@ tag
- Do NOT output any text before "## Direct Answer"
- Do NOT add sections not listed above
- If equations exist in evidence, include them inline as $$ ... $$
- If information is missing, explicitly state the limitation"""
        return self.llm.generate(prompt, max_new_tokens=2048, temperature=0.05)

    def _format_as_table(self, query, items, qa):
        value_items = [i for i in items if i.value is not None]
        if not value_items:
            return self._format_as_prose(query, items, qa) + "\n*[Note: No explicit numerical values were found for this query. The response above provides conceptual information instead.]*"
        param_block = []
        for item in sorted(value_items, key=lambda x: x.confidence, reverse=True)[:30]:
            pq = self.phys_classifier.get_human_readable(item.physical_quantity or "unknown")
            val = f"{item.value:.4g}" if item.value is not None else "N/A"
            unit = item.unit or ""
            mat = item.material or ""
            doc = item.doc_source
            page = item.page
            conf = f"{item.confidence:.2f}"
            param_block.append(f"- {pq}: {val} {unit} | Material: {mat} | Doc: {doc} | Page: {page} | Conf: {conf} | @@CITE:doc={doc};page={page}@@")
        param_text = "\n".join(param_block)
        by_pq = {}
        for item in value_items:
            pq = item.physical_quantity or "unknown"
            if pq not in by_pq:
                by_pq[pq] = []
            by_pq[pq].append(item.value)
        stats_block = []
        for pq, vals in by_pq.items():
            readable = self.phys_classifier.get_human_readable(pq)
            stats_block.append(f"- {readable}: n={len(vals)}, range={min(vals):.4g} to {max(vals):.4g}, mean={sum(vals)/len(vals):.4g}")
        stats_text = "\n".join(stats_block)
        prompt = f"""You are a scientific analyst. You MUST follow this EXACT output structure. Do NOT deviate.
QUERY: {query}
EXTRACTED PARAMETERS:
{param_text}
SUMMARY STATISTICS:
{stats_text}
REQUIRED OUTPUT STRUCTURE (follow exactly):
## Extracted Values: {query}
### Parameter Table
| Parameter | Value | Unit | Material | Document | Page | Confidence |
|-----------|-------|------|----------|----------|------|------------|
[RECREATE the table above with ALL parameters. EVERY row MUST end with @@CITE:doc=filename;page=N@@]
### Summary Statistics
[Restate the summary statistics from above]
### Physical Interpretation
[2-3 sentences explaining what these values mean in context]
RULES:
- EVERY table row MUST have an @@CITE:doc=filename;page=N@@ tag
- Do NOT output any text before "## Extracted Values"
- Do NOT add sections not listed above
- Include ALL extracted parameters, not just a subset"""
        return self.llm.generate(prompt, max_new_tokens=2048, temperature=0.05)

    def _format_as_value_comparison(self, query, items, qa):
        value_items = [i for i in items if i.value is not None or i.item_type == "tabular_data"]
        if not value_items:
            return self._format_as_prose(query, items, qa)
        evidence_blocks = []
        for item in sorted(value_items, key=lambda x: x.confidence, reverse=True):
            pq = self.phys_classifier.get_human_readable(item.physical_quantity or "unknown")
            val = f"{item.value:.4g}" if item.value is not None else "N/A"
            unit = item.unit or ""
            mat = item.material or "Unspecified Alloy"
            cite = f'<cite doc="{item.doc_source}" page="{item.page}"/>'
            evidence_blocks.append(f"- [{mat}] {pq}: {val} {unit} | Context: {item.content[:300]} {cite}")
        evidence_text = "\n".join(evidence_blocks)
        prompt = f"""You are an expert scientific analyst. You MUST follow this EXACT output structure. Do NOT deviate.
QUERY: {query}
EXTRACTED EVIDENCE:
{evidence_text}
REQUIRED OUTPUT STRUCTURE (follow exactly):
Here is a comprehensive summary of laser power settings used across the different materials and alloys in your library:
### Laser Power in Watts — by Material/Alloy
[For each distinct material/alloy found in the evidence, create a numbered section:]
1. [Material/Alloy Name] ([Process, e.g., LPBF, SLM, Laser Microalloying])
- **Laser Powers Tested:** [List the specific Wattages, e.g., 250 W and 350 W]
- **Outcomes & Context:** [Describe what happened at these powers. Be specific with numbers, e.g., "Relative densities ranged from 99.44% to 99.85%", "caused vaporization", "optimal processability".]
- **Key Finding:** [State the optimal power, critical threshold, or main conclusion for this material]
- **Citation:** <cite doc="filename.pdf" page="N"/>
### Summary Table
[Create a Markdown table with exactly these columns: Material / Alloy | Process | Laser Power (W) | Optimal Power]
[Fill the table with the extracted data. Keep it concise.]
### Key Takeaway
[Write a 2-3 sentence paragraph synthesizing the macro-trend across all materials. For example, note the typical power ranges for Al-based vs. Ti-based alloys, or how specific additions affect absorption.]
RULES:
- You MUST use the exact heading names provided above ("### Laser Power in Watts...", "### Summary Table", "### Key Takeaway").
- EVERY factual claim and table row MUST be backed by a <cite doc="filename.pdf" page="N"/> tag.
- Do NOT invent data. If a detail is missing, state "Not specified".
- Keep the tone professional, precise, and scientific."""
        return self.llm.generate(prompt, max_new_tokens=2048, temperature=0.1)

    def _format_as_mechanism(self, query, items, qa):
        if not items:
            return f"No mechanism information found for '{query}'."
        evidence = []
        for i in items[:20]:
            cite = f"@@CITE:doc={i.doc_source};page={i.page}@@"
            evidence.append(f"- {i.content[:300]} {cite}")
        evidence_text = "\n".join(evidence)
        prompt = f"""You are a scientific analyst. You MUST follow this EXACT output structure. Do NOT deviate.
QUERY: {query}
EVIDENCE FROM DOCUMENTS:
{evidence_text}
REQUIRED OUTPUT STRUCTURE (follow exactly):
## Mechanism: {query}
### Step 1: [Initial Condition / Trigger]
[Explain what starts the process. MUST cite: @@CITE:doc=filename;page=N@@]
### Step 2: [Primary Process]
[Explain the main physical process. MUST cite: @@CITE:doc=filename;page=N@@]
### Step 3: [Secondary Effects / Consequences]
[Explain downstream effects. MUST cite: @@CITE:doc=filename;page=N@@]
### Governing Equations (if any)
[Include equations in $$ ... $$ format with variable definitions]
### Summary
[1-2 sentence summary of the overall mechanism]
### Uncertainties
[Note any gaps or alternative explanations]
RULES:
- EVERY step MUST have an @@CITE:doc=filename;page=N@@ tag
- Do NOT output any text before "## Mechanism"
- Do NOT add sections not listed above
- Number steps sequentially"""
        return self.llm.generate(prompt, max_new_tokens=2048, temperature=0.05)

    def _format_as_comparison(self, query, items, qa):
        if not items:
            return f"No comparison data found for '{query}'."
        evidence = []
        for i in items[:20]:
            cite = f"@@CITE:doc={i.doc_source};page={i.page}@@"
            if i.value is not None:
                pq = self.phys_classifier.get_human_readable(i.physical_quantity or "unknown")
                evidence.append(f"- {pq}: {i.value} {i.unit or ''} (material={i.material or 'N/A'}) {cite}")
            else:
                evidence.append(f"- {i.content[:200]} {cite}")
        evidence_text = "\n".join(evidence)
        prompt = f"""You are a scientific analyst. You MUST follow this EXACT output structure. Do NOT deviate.
QUERY: {query}
EVIDENCE:
{evidence_text}
REQUIRED OUTPUT STRUCTURE (follow exactly):
## Comparative Analysis: {query}
### Similarities
- [Similarity 1 with @@CITE:doc=filename;page=N@@]
- [Similarity 2 with @@CITE:doc=filename;page=N@@]
### Differences
| Aspect | Entity A | Entity B | Source |
|--------|----------|----------|--------|
| [Aspect] | [Value A] | [Value B] | @@CITE:doc=filename;page=N@@ |
### Key Insight
[1-2 sentence synthesis of the most important comparison finding]
### Confidence
[Note any limitations in the comparison]
RULES:
- EVERY claim MUST have an @@CITE:doc=filename;page=N@@ tag
- Do NOT output any text before "## Comparative Analysis"
- Do NOT add sections not listed above
- Use markdown tables for quantitative comparisons"""
        return self.llm.generate(prompt, max_new_tokens=2048, temperature=0.05)

    def _format_as_boolean(self, query, items, qa):
        if not items:
            return f"## Verification Result\n**Answer:** The documents do not contain sufficient information to confirm or deny: '{query}'\n**Note:** Try rephrasing or uploading additional documents."
        evidence = []
        for i in items[:15]:
            cite = f"@@CITE:doc={i.doc_source};page={i.page}@@"
            evidence.append(f"- {i.content[:300]} {cite}")
        evidence_text = "\n".join(evidence)
        prompt = f"""You are a scientific analyst. You MUST follow this EXACT output structure. Do NOT deviate.
QUESTION: {query}
EVIDENCE:
{evidence_text}
REQUIRED OUTPUT STRUCTURE (follow exactly):
## Verification: {query}
### Answer
**[Yes / No / Partially / Uncertain]**
### Confidence
**[High / Medium / Low]** — [1 sentence justification]
### Explanation
[2-3 sentences explaining the reasoning. EVERY sentence MUST cite: @@CITE:doc=filename;page=N@@]
### Key Evidence
- [Most relevant evidence with @@CITE:doc=filename;page=N@@]
- [Second piece with @@CITE:doc=filename;page=N@@]
RULES:
- EVERY claim MUST have an @@CITE:doc=filename;page=N@@ tag
- Do NOT output any text before "## Verification"
- Do NOT add sections not listed above
- Choose ONLY ONE answer: Yes, No, Partially, or Uncertain"""
        return self.llm.generate(prompt, max_new_tokens=1024, temperature=0.05)

    def _format_as_table_data(self, query, items, qa):
        table_items = [i for i in items if i.item_type == "tabular_data"]
        if not table_items:
            return self._format_as_prose(query, items, qa)
        lines = ["## Extracted Tabular Data\n"]
        for item in table_items:
            lines.append(f"### Source: {item.doc_source} (Page {item.page})")
            lines.append(f"**Columns:** {item.parameter_name}")
            lines.append(f"```markdown\n{item.content}\n```")
            lines.append(f"@@CITE:doc={item.doc_source};page={item.page}@@\n")
        return "\n".join(lines)

    def generate_human_conclusion(self, query: str, report: QueryReport) -> str:
        values = report.all_values
        if not values:
            return f"No quantitative data found for '{query}' across the analyzed documents."
        by_phys = defaultdict(list)
        by_material = defaultdict(list)
        for v in values:
            by_phys[v.physical_quantity].append(v)
            if v.material:
                by_material[v.material].append(v)
        lines = [f"## Summary: {query.title()}", f"Across **{report.total_docs}** documents analyzed, **{report.docs_with_results}** contained relevant quantitative data.", f"Total extracted values: **{len(values)}**.", ""]
        lines.append("### By Physical Quantity")
        for pq, vals in sorted(by_phys.items()):
            readable = self.phys_classifier.get_human_readable(pq)
            lines.append(f"#### {readable} ({len(vals)} values)")
            nums = [v.value for v in vals]
            units = list(set(v.unit for v in vals))
            docs = list(set(v.doc_name for v in vals))
            if nums:
                lines.append(f"- **Range**: {min(nums):.2f} to {max(nums):.2f} {units[0] if units else ''}")
                lines.append(f"- **Average**: {np.mean(nums):.2f}")
                if len(nums) > 1:
                    lines.append(f"- **Std Dev**: {np.std(nums):.2f}")
                lines.append(f"- **Found in**: {', '.join(docs[:3])}{'...' if len(docs) > 3 else ''}")
            lines.append("")
        if by_material:
            lines.append("### By Material/Alloy")
            for mat, vals in sorted(by_material.items()):
                lines.append(f"#### {mat} ({len(vals)} values)")
                inner_pq = defaultdict(list)
                for v in vals:
                    inner_pq[v.physical_quantity].append(v.value)
                for pq, nums in inner_pq.items():
                    readable = self.phys_classifier.get_human_readable(pq)
                    lines.append(f"- {readable}: min={min(nums):.2f}, max={max(nums):.2f}, mean={np.mean(nums):.2f}")
                docs = list(set(v.doc_name for v in vals))
                lines.append(f"- **Documents**: {', '.join(docs)}")
                lines.append("")
        lines.append("### Key Values by Document and Physical Quantity")
        for v in sorted(values, key=lambda x: x.confidence, reverse=True)[:12]:
            readable = self.phys_classifier.get_human_readable(v.physical_quantity)
            mat_str = f" ({v.material})" if v.material else ""
            lines.append(f"| {v.doc_name} | p.{v.page} | {v.value:.2f} {v.unit} | {readable}{mat_str} |")
        if report.uncertainty_metrics:
            lines.append("### Uncertainty Quantification (UQ)")
            for k, v in report.uncertainty_metrics.items():
                lines.append(f"- {k}: {v}")
        if report.xai_attributions:
            lines.append("### Explainable AI (XAI) Feature Attributions")
            for k, v in report.xai_attributions.items():
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)

# ============================================================================
# KNOWLEDGE GRAPH
# ============================================================================
class QuantitativeKnowledgeGraph:
    def __init__(self):
        self.doc_graphs: Dict[str, Dict] = {}
        self.phys_classifier = PhysicalQuantityClassifier()
        self.metadata_index: Dict[str, DocumentMetadata] = {}
        self.concept_normalizer = ConceptNormalizer()

    def add_document_metadata(self, doc_name: str, metadata: DocumentMetadata):
        self.metadata_index[doc_name] = metadata

    def add_extractions(self, doc_id: str, items: List[UniversalExtractionItem]):
        graph = {
            "doc_id": doc_id,
            "parameters": defaultdict(list),
            "materials": defaultdict(list),
            "methods": defaultdict(list),
            "by_page": defaultdict(list),
            "by_section": defaultdict(list),
            "by_physical_quantity": defaultdict(list),
            "by_multiphysics": defaultdict(list),
            "by_electrochemical": defaultdict(list),
            "by_ai_ml": defaultdict(list),
            "by_equation_model": defaultdict(list),
            "all_items": []
        }
        for item in items:
            item_dict = item.to_dict()
            if item.physical_quantity:
                item_dict["physical_quantity"] = self.concept_normalizer.normalize(item.physical_quantity)
            if item.material:
                item_dict["material"] = self.concept_normalizer.normalize(item.material)
            if item.simulation_type:
                item_dict["simulation_type"] = self.concept_normalizer.normalize(item.simulation_type)
            if item.item_type == "equation" and item.model_name:
                normalized_model = item.model_name.lower().replace(" ", "_")
                item_dict["normalized_model"] = normalized_model
                item_dict["original_model_name"] = item.model_name
                graph["by_equation_model"][normalized_model].append(item_dict)
            graph["all_items"].append(item_dict)
            if item.parameter_name:
                graph["parameters"][item.parameter_name.lower()].append(item_dict)
            if item.material:
                graph["materials"][item.material.lower()].append(item_dict)
            if item.method:
                graph["methods"][item.method.lower()].append(item_dict)
            if item.physical_quantity:
                graph["by_physical_quantity"][item.physical_quantity].append(item_dict)
            if item.item_type in ["phase_field", "molecular_dynamics", "plasticity", "thermal", "mechanical", "microstructural", "multiphysics"]:
                graph["by_multiphysics"][item.item_type].append(item_dict)
            if item.item_type == "electrochemical":
                graph["by_electrochemical"]["eis_cpp"].append(item_dict)
            if item.item_type in ["ai_ml", "digital_twin"]:
                graph["by_ai_ml"][item.item_type].append(item_dict)
            graph["by_page"][item.page].append(item_dict)
            if item.section_title:
                graph["by_section"][item.section_title].append(item_dict)
        self.doc_graphs[doc_id] = dict(graph)

    def get_all_physical_quantities(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for doc_id, graph in self.doc_graphs.items():
            for item in graph["all_items"]:
                pq = item.get("physical_quantity")
                if pq:
                    counts[pq] += 1
        return dict(counts)

    def get_all_materials(self) -> Dict[str, List[str]]:
        mat_dict = {}
        for doc_id, graph in self.doc_graphs.items():
            materials = set()
            for item in graph["all_items"]:
                if item.get("material"):
                    materials.add(item["material"])
            mat_dict[doc_id] = list(materials)
        return mat_dict

    def get_all_entity_names(self) -> List[str]:
        entities = set()
        for doc_id, graph in self.doc_graphs.items():
            for item in graph["all_items"]:
                if item.get("material"):
                    entities.add(item["material"])
                if item.get("physical_quantity"):
                    entities.add(item["physical_quantity"])
                if item.get("parameter_name"):
                    entities.add(item["parameter_name"])
                if item.get("method"):
                    entities.add(item["method"])
        return sorted(entities)

    def get_entity_consensus(self, entity_name: str) -> Dict[str, Any]:
        values = []
        units = set()
        docs = set()
        for doc_id, graph in self.doc_graphs.items():
            for item in graph["all_items"]:
                if (item.get("material") == entity_name or item.get("physical_quantity") == entity_name):
                    if item.get("value") is not None:
                        values.append(item["value"])
                        units.add(item.get("unit", ""))
                        docs.add(doc_id)
        if not values:
            return {"found": False, "entity": entity_name}
        return {
            "found": True, "entity": entity_name, "count": len(values),
            "unit": list(units)[0] if units else "unknown",
            "range": (min(values), max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)) if len(values) > 1 else 0.0,
            "documents": list(docs), "values": values
        }

    def get_entity_contradictions(self, entity_name: str, threshold_factor: float = 2.0) -> List[Dict[str, Any]]:
        by_doc = defaultdict(list)
        for doc_id, graph in self.doc_graphs.items():
            for item in graph["all_items"]:
                if (item.get("material") == entity_name or item.get("physical_quantity") == entity_name):
                    if item.get("value") is not None:
                        by_doc[doc_id].append(item["value"])
        contradictions = []
        docs = list(by_doc.keys())
        for i in range(len(docs)):
            for j in range(i+1, len(docs)):
                if by_doc[docs[i]] and by_doc[docs[j]]:
                    mean_i = np.mean(by_doc[docs[i]])
                    mean_j = np.mean(by_doc[docs[j]])
                    if mean_i > 0 and mean_j > 0:
                        ratio = max(mean_i, mean_j) / min(mean_i, mean_j)
                        if ratio > threshold_factor:
                            contradictions.append({
                                "entity": entity_name, "doc_a": docs[i], "value_a": mean_i,
                                "doc_b": docs[j], "value_b": mean_j, "ratio": ratio,
                                "severity": "high" if ratio > 5 else "moderate"
                            })
        return contradictions

    def to_tree_annotation(self, doc_tree: Any, max_chars: int = 20000) -> Dict[str, Any]:
        doc_id = getattr(doc_tree, 'doc_id', 'unknown')
        graph = self.doc_graphs.get(doc_id, {})
        def annotate_node(node: Any) -> Dict[str, Any]:
            result = {
                "title": getattr(node, 'title', ''),
                "node_id": getattr(node, 'node_id', ''),
                "start_index": getattr(node, 'page_start', 1),
                "end_index": getattr(node, 'page_end', getattr(node, 'page_start', 1)),
                "summary": getattr(node, 'summary', ''),
                "prefix_summary": getattr(node, 'prefix_summary', ''),
                "text_token_count": getattr(node, 'text_token_count', 0)
            }
            node_items = []
            end_page = getattr(node, 'page_end', getattr(node, 'page_start', 1))
            for page in range(getattr(node, 'page_start', 1), end_page + 1):
                node_items.extend(graph.get("by_page", {}).get(page, []))
            if node_items:
                seen = set()
                unique_items = []
                for item in node_items:
                    key = (item.get('parameter_name'), item.get('value'), item.get('page'))
                    if key not in seen:
                        seen.add(key)
                        unique_items.append(item)
                result["quantitative_items"] = unique_items
            children = getattr(node, 'children', [])
            if children:
                result["nodes"] = [annotate_node(c) for c in children]
            text = getattr(node, 'full_text', '')
            if text:
                result["text"] = text[:max_chars]
            metadata = getattr(node, 'metadata', None)
            if metadata:
                result["metadata"] = metadata.dict() if hasattr(metadata, 'dict') else metadata
            return result
        return annotate_node(doc_tree)

# ============================================================================
# STREAMLIT UI & UTILITIES
# ============================================================================
UNIVERSAL_CONFIG = {"leaf_node_page_window": 7, "min_confidence_threshold": 0.55}

def render_sidebar():
    with st.sidebar:
        st.markdown("### Configuration")
        model_keys = list(MODEL_OPTIONS.keys())
        if "llm_model_choice" not in st.session_state:
            st.session_state.llm_model_choice = model_keys[0]
        selected = st.selectbox("Select LLM Backend", options=model_keys, index=model_keys.index(st.session_state.llm_model_choice), key="llm_model_select")
        st.session_state.llm_model_choice = selected
        st.checkbox("Use 4-bit quantization (if Transformers)", value=True, key="use_4bit")
        st.slider("Confidence threshold", 0.3, 0.9, 0.55, 0.05, key="min_confidence")
        max_chars = st.slider("Max text length per retrieved section (characters)", min_value=1000, max_value=50000, value=20000, step=1000, help="Larger values give more context but use more memory/LLM tokens.")
        st.session_state.max_retrieval_chars = max_chars
        st.checkbox("Show reasoning trace", value=True, key="show_trace")
        st.checkbox("Show tree navigation", value=True, key="show_tree_nav")
        st.checkbox("Enable two-stage retrieval", value=True, key="two_stage")
        st.markdown("#### Visualization Settings")
        st.selectbox("Default colormap", ["viridis", "plasma", "inferno", "magma", "cividis", "Blues", "Greens", "Oranges", "Reds", "Purples"], index=0, key="viz_colormap")
        st.selectbox("Document label style", ["doi", "number", "alias", "short"], index=0, key="viz_label_style")
        st.slider("Top N concepts", 5, 100, 25, key="viz_top_n")
        st.multiselect("Filter domains", options=["laser_power","scan_speed","yield_strength","tensile_strength","hardness","temperature","energy_density","phase_field_method","molecular_dynamics","pinn","unet","convlstm","calphad","digital_twin","xai","uncertainty_quantification"], default=["laser_power","scan_speed","yield_strength"], key="viz_domains")
        st.caption(f"GPU: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        if st.button("Clear Cache & Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

@st.cache_resource(show_spinner="Initializing LLM...")
def get_cached_llm(model_choice: str, use_4bit: bool):
    internal = MODEL_OPTIONS[model_choice]
    return HybridLLM(model_key=internal, use_4bit=use_4bit)

def run_streamlit():
    st.set_page_config(page_title="DECLARMIMA v20.0 - Agentic Navigation & Structural Reasoning", layout="wide")
    st.markdown("# DECLARMIMA v20.0 - Pure PageIndex Agent (Agentic Navigation & Structural Reasoning)")
    st.caption("v20.0: Iterative MCTS navigation, Cross-Document Meta-Trees, layout-aware Markdown parsing, table/figure structural nodes. 100% vectorless.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "query_processor" not in st.session_state:
        st.session_state.query_processor = {}
    if "knowledge_graph" not in st.session_state:
        st.session_state.knowledge_graph = QuantitativeKnowledgeGraph()
    if "annotated_trees" not in st.session_state:
        st.session_state.annotated_trees = []
    if "cached_query_result" not in st.session_state:
        st.session_state.cached_query_result = None
    if "active_prompt" not in st.session_state:
        st.session_state.active_prompt = ""
    if "two_stage_retriever" not in st.session_state:
        st.session_state.two_stage_retriever = None
    if "doc_aliases" not in st.session_state:
        st.session_state.doc_aliases = {}

    render_sidebar()
    max_retrieval_chars = st.session_state.get("max_retrieval_chars", 20000)
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files and st.button("Build Index", type="primary"):
        st.session_state.query_processor["files"] = uploaded_files
        st.success(f"{len(uploaded_files)} files registered.")
        st.rerun()

    if st.session_state.query_processor.get("files") and not st.session_state.annotated_trees:
        with st.spinner("Building hierarchical index with metadata extraction..."):
            progress = st.progress(0)
            llm = get_cached_llm(st.session_state.llm_model_choice, st.session_state.get("use_4bit", True))
            progress.progress(0.1)
            
            # Simple tree building for demo purposes
            trees = {}
            extractor = UniversalLLMExtractor(llm)
            kg = QuantitativeKnowledgeGraph()
            all_items = []
            
            for file in st.session_state.query_processor["files"]:
                doc_name = file.name
                pages_data = extract_text_from_pdf(file.read(), max_pages=50)
                
                # Build a simple flat tree for the document
                tree = {
                    "doc_id": doc_name,
                    "doc_name": doc_name,
                    "title": doc_name,
                    "node_id": "0000",
                    "start_index": 1,
                    "end_index": len(pages_data),
                    "summary": f"Document {doc_name}",
                    "nodes": []
                }
                
                # Chunk pages for extraction
                chunk_size = 5
                for i in range(0, len(pages_data), chunk_size):
                    chunk = pages_data[i:i+chunk_size]
                    text = "\n".join([f"<page_{p['page_num']}>\n{p['text']}\n</page_{p['page_num']}>" for p in chunk])
                    
                    node = {
                        "doc_id": doc_name,
                        "title": f"Pages {chunk[0]['page_num']}-{chunk[-1]['page_num']}",
                        "node_id": f"0000.{(i//chunk_size)+1:04d}",
                        "start_index": chunk[0]['page_num'],
                        "end_index": chunk[-1]['page_num'],
                        "summary": text[:200] + "...",
                        "text": text,
                        "nodes": []
                    }
                    tree["nodes"].append(node)
                    
                    # Extract items
                    prompt = "Extract ALL quantitative parameters, materials, methods, and equations. Include units, material names, and page numbers."
                    items = extractor.extract_from_chunks([{"full_text": text, "doc_id": doc_name, "page_start": chunk[0]['page_num']}], prompt)
                    all_items.extend(items)
                
                trees[doc_name] = tree
                kg.add_extractions(doc_name, items)
                kg.add_document_metadata(doc_name, DocumentMetadata(doc_name=doc_name))
                
            st.session_state.query_processor["doc_trees"] = trees
            st.session_state.knowledge_graph = kg
            st.session_state.annotated_trees = list(trees.values())
            progress.progress(1.0)
            st.success(f"Indexed {len(trees)} documents with {len(all_items)} quantitative items")
            st.rerun()

    if st.session_state.annotated_trees:
        st.markdown("### Quick Queries")
        col1, col2, col3, col4 = st.columns(4)
        quick = ["laser power", "yield strength", "scan speed", "alloy names", "temperature", "digital twin"]
        for i, q in enumerate(quick):
            with [col1, col2, col3, col4][i % 4]:
                if st.button(f"{q.title()}", key=f"quick_{q}"):
                    st.session_state.quick_query = f"What is the {q} discussed in these papers?"
                    st.rerun()

        default_query = st.session_state.get("quick_query", "")
        prompt_input = st.chat_input("Ask about any term, value, material, or mechanical property...", key="chat_input")
        
        if default_query and not prompt_input:
            prompt_input = default_query
            st.session_state.quick_query = ""

        if prompt_input:
            st.session_state.active_prompt = prompt_input
            st.session_state.messages.append({"role": "user", "content": prompt_input})
            with st.chat_message("user"):
                st.markdown(prompt_input)
        elif st.session_state.active_prompt:
            with st.chat_message("user"):
                st.markdown(st.session_state.active_prompt)

        active_prompt = st.session_state.get("active_prompt", "")
        run_query = False
        if active_prompt:
            cached = st.session_state.cached_query_result
            has_valid_cache = cached and cached.get("prompt") == active_prompt and "answer" in cached
            if not has_valid_cache:
                run_query = True

        if run_query:
            with st.chat_message("assistant"):
                progress = st.progress(0)
                progress.text("Initializing LLM...")
                llm = get_cached_llm(st.session_state.llm_model_choice, st.session_state.get("use_4bit", True))
                
                router = ScientificIntentRouter()
                query_analysis = router.route(active_prompt)
                
                st.warning(f"🔍 ROUTING CHECK | Intent: `{query_analysis['intent']}` | Format: `{query_analysis['output_format']}` | Gated: `{'✅ Explicit' if query_analysis['value_extraction_gated'] else '❌ Implicit'}`")
                progress.progress(0.1)
                
                retriever = HierarchicalTreeRetriever(llm, max_results=30, max_text_chars=max_retrieval_chars)
                meta_tree = retriever.build_cross_document_meta_tree(active_prompt, st.session_state.annotated_trees)
                navigator = IterativeTreeNavigator(llm, max_steps=4)
                all_trees = [meta_tree] + st.session_state.annotated_trees if meta_tree.get("nodes") else st.session_state.annotated_trees
                
                progress.text("Navigating document trees...")
                retrieved = asyncio.run(navigator.navigate_and_retrieve(active_prompt, all_trees))
                progress.progress(0.5)
                
                progress.text("Extracting information...")
                extractor = UniversalLLMExtractor(llm)
                items = []
                for r in retrieved:
                    items.extend(extractor.extract_from_chunks([r], active_prompt, query_analysis=query_analysis))
                items = [i for i in items if i.confidence >= st.session_state.get("min_confidence", 0.55)]
                progress.progress(0.8)
                
                progress.text("Generating response...")
                extracted_values = []
                for item in items:
                    if item.item_type == "quantitative" and item.value is not None:
                        phys_q = item.physical_quantity or PhysicalQuantityClassifier().classify(item.parameter_name, item.unit, item.context)
                        extracted_values.append(ExtractedValue(
                            query=active_prompt, value=item.value, unit=item.unit or "", physical_quantity=phys_q,
                            parameter_name=item.parameter_name, material=item.material, confidence=item.confidence,
                            context=item.context, doc_name=item.doc_source, page=item.page, section_title=item.section_title,
                            simulation_context=item.simulation_type, temperature_dependent="temperature" in item.context.lower()
                        ))
                
                generator = AdaptiveResponseGenerator(llm)
                answer = generator.generate(active_prompt, items, query_analysis)
                progress.progress(1.0, text="Done!")
                st.markdown(answer)
                
                st.session_state.cached_query_result = {
                    "prompt": active_prompt,
                    "relevant_docs": [(t.get("doc_id", t.get("doc_name", "unknown")), 1.0) for t in st.session_state.annotated_trees],
                    "retrieved": retrieved,
                    "items": [i.model_dump() for i in items],
                    "extracted_values": [v.model_dump() for v in extracted_values],
                    "answer": answer,
                    "query_analysis": query_analysis,
                    "multiphysics_flags": [],
                    "electrochemical_flags": [],
                    "ai_ml_flags": [],
                    "microstructural_features": []
                }
                st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            if active_prompt and st.session_state.cached_query_result and "answer" in st.session_state.cached_query_result:
                cached = st.session_state.cached_query_result
                with st.chat_message("assistant"):
                    st.markdown(cached["answer"])
            elif not active_prompt:
                st.info("Ask a question about the documents.")

if __name__ == "__main__":
    run_streamlit()
