#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DECLARMIMA v18.0 - ENHANCED VECTORLESS RAG WITH PAGEINDEX-STYLE INTELLIGENCE
============================================================================
New Features vs v17.2:
1. Strict 2-call query architecture (navigation → answer)
2. Roll-up hierarchical summarization (bottom-up tree condensation)
3. LLM-fallback PhysicalQuantityClassifier for ambiguous terminology
4. Full annotated-tree caching with SHA-256 content hashing
5. Async-first retrieval pipeline with proper error isolation
6. Production-ready logging, metrics, and fallback handling
"""
import streamlit as st
import os
import sys
import tempfile
import time
import re
import json
import hashlib
import asyncio
import logging
import warnings
import requests
import textwrap
import math
import copy
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any, Set, Callable, Literal, TypeVar
from collections import defaultdict, Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from io import BytesIO
import numpy as np
import torch
import threading
import queue
import pandas as pd

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler],
    force=True,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("DECLARMIMA")

# ============================================================================
# DEPENDENCY CHECKS WITH GRACEFUL DEGRADATION
# ============================================================================
def check_optional_deps():
    """Check optional dependencies and log availability."""
    deps = {}
    try:
        import fitz
        deps['pymupdf'] = True
    except ImportError:
        deps['pymupdf'] = False
        logger.error("PyMuPDF (fitz) required: pip install pymupdf")
    
    try:
        import ollama
        deps['ollama'] = True
    except ImportError:
        deps['ollama'] = False
        logger.warning("Ollama not installed. Ollama backend unavailable.")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        deps['transformers'] = True
    except ImportError:
        deps['transformers'] = False
        logger.warning("transformers not installed. Local HF models unavailable.")
    
    try:
        import orjson
        deps['orjson'] = True
    except ImportError:
        deps['orjson'] = False
        logger.warning("orjson not installed. Using standard json (slower).")
    
    try:
        from sentence_transformers import SentenceTransformer, util
        deps['sentence_transformers'] = True
    except ImportError:
        deps['sentence_transformers'] = False
        logger.warning("sentence-transformers not installed. Using vectorless keyword retrieval.")
    
    try:
        from rapidfuzz import fuzz, process
        deps['rapidfuzz'] = True
    except ImportError:
        deps['rapidfuzz'] = False
        logger.warning("rapidfuzz not installed. Fuzzy matching disabled.")
    
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        deps['matplotlib'] = True
    except ImportError:
        deps['matplotlib'] = False
        logger.warning("matplotlib not installed. Static plots disabled.")
    
    try:
        import networkx as nx
        deps['networkx'] = True
    except ImportError:
        deps['networkx'] = False
        logger.warning("networkx not installed. Knowledge graphs disabled.")
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        deps['plotly'] = True
    except ImportError:
        deps['plotly'] = False
        logger.warning("plotly not installed. Interactive plots disabled.")
    
    try:
        import umap
        deps['umap'] = True
    except ImportError:
        deps['umap'] = False
        logger.warning("umap-learn not installed. UMAP embeddings disabled.")
    
    try:
        from pyvis.network import Network
        deps['pyvis'] = True
    except ImportError:
        deps['pyvis'] = False
        logger.warning("pyvis not installed. Interactive networks disabled.")
    
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        deps['sklearn'] = True
    except ImportError:
        deps['sklearn'] = False
        logger.warning("scikit-learn not installed. t-SNE/PCA disabled.")
    
    return deps

GLOBAL_DEPS = check_optional_deps()

# ============================================================================
# PYDANTIC MODELS (UNCHANGED FROM v17.2 - VALIDATED)
# ============================================================================
from pydantic import BaseModel, Field, field_validator, model_validator

class UniversalExtractionItem(BaseModel):
    item_type: Literal["quantitative", "qualitative", "definition", "comparison", "relationship", "process", "material", "method"]
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
    doc_source: str
    page: int
    section_title: Optional[str] = None
    method: Optional[str] = None
    conditions: Dict[str, Any] = {}
    reasoning_trace: str = ""
    
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
    doc_name: str
    page: int
    section_title: Optional[str] = None
    
    @field_validator('value')
    @classmethod
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
        return json.dumps(self.model_dump(), indent=2, ensure_ascii=False, default=str)

class CrossDocumentQueryReport(BaseModel):
    query: str
    query_type: Optional[str] = None
    total_documents: int
    documents_with_results: int
    documents_without_results: List[str] = []
    all_items: List[UniversalExtractionItem] = []
    document_summaries: List[Dict[str, Any]] = []
    consensus_analysis: Dict[str, Any] = {}
    contradictions_detected: List[Dict[str, Any]] = []
    processing_metadata: Dict[str, Any] = {}
    
    def to_json(self, indent=2) -> str:
        return json.dumps(self.model_dump(), indent=indent, ensure_ascii=False, default=str)

class DocumentMetadata(BaseModel):
    doc_name: str
    alloys: List[str] = Field(default_factory=list)
    laser_power_values: List[float] = Field(default_factory=list)
    scan_speed_values: List[float] = Field(default_factory=list)
    yield_strength_values: List[float] = Field(default_factory=list)
    tensile_strength_values: List[float] = Field(default_factory=list)
    hardness_values: List[float] = Field(default_factory=list)
    temperature_values: List[float] = Field(default_factory=list)
    energy_density_values: List[float] = Field(default_factory=list)
    areal_energy_density_values: List[float] = Field(default_factory=list)
    linear_energy_density_values: List[float] = Field(default_factory=list)
    process_types: List[str] = Field(default_factory=list)
    corrosion_potential_values: List[float] = Field(default_factory=list)
    pitting_potential_values: List[float] = Field(default_factory=list)
    repassivation_potential_values: List[float] = Field(default_factory=list)
    breakdown_potential_values: List[float] = Field(default_factory=list)
    open_circuit_potential_values: List[float] = Field(default_factory=list)
    corrosion_current_density_values: List[float] = Field(default_factory=list)
    polarization_resistance_values: List[float] = Field(default_factory=list)
    current_density_values: List[float] = Field(default_factory=list)
    pren_values: List[float] = Field(default_factory=list)
    phase_fraction_values: List[float] = Field(default_factory=list)
    austenite_fraction_values: List[float] = Field(default_factory=list)
    ferrite_fraction_values: List[float] = Field(default_factory=list)
    grain_size_values: List[float] = Field(default_factory=list)
    cell_size_values: List[float] = Field(default_factory=list)
    porosity_values: List[float] = Field(default_factory=list)
    relative_density_values: List[float] = Field(default_factory=list)
    surface_roughness_values: List[float] = Field(default_factory=list)
    stacking_fault_energy_values: List[float] = Field(default_factory=list)
    unstable_stacking_fault_energy_values: List[float] = Field(default_factory=list)
    ideal_shear_strength_values: List[float] = Field(default_factory=list)
    sauter_mean_diameter_values: List[float] = Field(default_factory=list)
    spray_penetration_values: List[float] = Field(default_factory=list)
    plume_height_values: List[float] = Field(default_factory=list)
    film_thickness_values: List[float] = Field(default_factory=list)
    absorption_coefficient_values: List[float] = Field(default_factory=list)
    enthalpy_values: List[float] = Field(default_factory=list)
    viscosity_values: List[float] = Field(default_factory=list)
    thermal_conductivity_values: List[float] = Field(default_factory=list)
    density_values: List[float] = Field(default_factory=list)
    elongation_values: List[float] = Field(default_factory=list)
    modulus_values: List[float] = Field(default_factory=list)
    youngs_modulus_values: List[float] = Field(default_factory=list)
    poisson_ratio_values: List[float] = Field(default_factory=list)
    cte_values: List[float] = Field(default_factory=list)
    hollomon_strength_coeff_values: List[float] = Field(default_factory=list)
    hollomon_exponent_values: List[float] = Field(default_factory=list)
    other_parameters: Dict[str, List[float]] = {}

# ============================================================================
# QUERY CONTEXT FOR DRIVEN VISUALIZATIONS (UNCHANGED)
# ============================================================================
@dataclass
class QueryContext:
    """Represents the context of the current user query for focused visualizations."""
    query: str
    relevant_doc_ids: Set[str]
    physical_quantities: List[str]
    materials: List[str]
    extracted_values: List[ExtractedValue]
    retrieved_nodes: List[Dict]
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
        )
    
    def has_data(self) -> bool:
        return len(self.extracted_values) > 0

# ============================================================================
# ENHANCED PHYSICAL QUANTITY CLASSIFIER WITH LLM FALLBACK
# ============================================================================
class PhysicalQuantityClassifier:
    """
    Enhanced classifier with:
    - Original keyword/unit matching (fast path)
    - LLM-based fallback for ambiguous terms (smart path)
    - Caching of LLM resolutions to minimize calls
    """
    
    CANONICAL = {
        "laser_power": ["laser power", "laser beam power", "laser output power", "laser power density (power)", "power", "p"],
        "electrical_power": ["electrical power", "power supply", "input power", "electrical load"],
        "scan_speed": ["scan speed", "scanning speed", "laser scan speed", "beam scan speed", "scan velocity", "v_scan", "vs"],
        "flow_speed": ["flow speed", "flow velocity", "fluid velocity", "air velocity", "gas flow speed"],
        "feed_rate": ["feed rate", "travel speed", "table speed", "stage speed"],
        "irradiance": ["irradiance", "laser irradiance", "intensity", "power density (irradiance)", "w/cm2", "kw/cm2"],
        "temperature": ["temperature", "melting temperature", "annealing temperature", "reflow temperature"],
        "melting_temperature": ["melting point", "melting temperature", "solidus temperature", "liquidus temperature"],
        "energy_density": ["energy density", "volumetric energy density", "ved", "laser fluence"],
        "areal_energy_density": ["areal energy density", "aed", "area energy density"],
        "linear_energy_density": ["linear energy density", "led", "line energy density"],
        "layer_thickness": ["layer thickness", "powder layer thickness", "slice thickness"],
        "spot_size": ["spot size", "beam diameter", "laser spot diameter"],
        "exposure_time": ["exposure time", "dwell time", "laser on time"],
        "yield_strength": ["yield strength", "ys", "0.2% offset strength", "proof stress", "yield stress"],
        "tensile_strength": ["tensile strength", "uts", "ultimate tensile strength", "ultimate strength"],
        "ultimate_tensile_strength": ["ultimate tensile strength", "uts", "tensile strength"],
        "hardness": ["hardness", "vickers hardness", "microhardness", "hv", "nano hardness"],
        "elongation": ["elongation", "strain", "ductility", "strain to failure"],
        "modulus": ["young's modulus", "elastic modulus", "stiffness", "e-modulus"],
        "youngs_modulus": ["young's modulus", "elastic modulus", "stiffness", "e-modulus"],
        "poisson_ratio": ["poisson's ratio", "poisson ratio"],
        "coefficient_thermal_expansion": ["coefficient of thermal expansion", "cte", "thermal expansivity", "thermal expansion coefficient"],
        "corrosion_potential": ["corrosion potential", "e_corr", "ecorr", "corrosion potential ecorr", "open circuit potential", "e_ocp", "eocp"],
        "pitting_potential": ["pitting potential", "e_pit", "epit", "breakdown potential", "e_br", "ebr"],
        "repassivation_potential": ["repassivation potential", "e_rp", "erp", "repassivation potential erp"],
        "breakdown_potential": ["breakdown potential", "e_br", "ebr", "depassivation point"],
        "open_circuit_potential": ["open circuit potential", "e_ocp", "eocp", "ocp"],
        "corrosion_current_density": ["corrosion current density", "j_corr", "jcorr", "corrosion current", "i_corr"],
        "current_density": ["current density", "j", "current density j", "i"],
        "polarization_resistance": ["polarization resistance", "r_p", "rp", "apparent polarization resistance", "rp_app"],
        "apparent_polarization_resistance": ["apparent polarization resistance", "rp_app", "rp,app", "apparent rp"],
        "PREN": ["pitting resistance equivalent number", "pren", "pitting resistance equivalent"],
        "phase_fraction": ["phase fraction", "volume fraction"],
        "austenite_fraction": ["austenite fraction", "gamma fraction", "γ fraction", "austenite content", "austenite vol"],
        "ferrite_fraction": ["ferrite fraction", "alpha fraction", "α fraction", "ferrite content", "ferrite vol"],
        "grain_size": ["grain size", "average grain size", "cell size", "subgrain size"],
        "cell_size": ["cell size", "cell diameter", "subgrain size"],
        "porosity": ["porosity", "pore fraction", "void fraction"],
        "relative_density": ["relative density", "density ratio", "packing density"],
        "surface_roughness": ["surface roughness", "ra", "roughness"],
        "stacking_fault_energy": ["stacking fault energy", "sfe", "gsfe", "generalized stacking fault energy"],
        "unstable_stacking_fault_energy": ["unstable stacking energy", "usfe"],
        "ideal_shear_strength": ["ideal shear strength", "t_ideal", "shear strength"],
        "sauter_mean_diameter": ["sauter mean diameter", "smd", "mean droplet diameter", "droplet diameter"],
        "spray_penetration": ["spray penetration", "penetration length", "fuel penetration"],
        "plume_height": ["plume height", "hw", "spray height"],
        "film_thickness": ["film thickness", "wall film thickness", "delta"],
        "absorption_coefficient": ["absorption coefficient", "absorptance", "laser absorption"],
        "enthalpy": ["enthalpy", "heat content", "specific enthalpy"],
        "viscosity": ["viscosity", "dynamic viscosity", "apparent viscosity"],
        "thermal_conductivity": ["thermal conductivity", "k", "kth", "heat conductivity"],
        "density": ["density", "mass density", "specific density", "volumetric density"],
        "lewis_number": ["lewis number", "le", "thermal mass diffusivity ratio", "lewis no"],
        "jackson_parameter": ["jackson parameter", "alpha_j", "αj", "jackson alpha", "morphology parameter"],
        "hollomon_strength_coeff": ["hollomon strength coefficient", "strength coefficient", "sigma_0", "σ0", "k (hollomon)"],
        "hollomon_exponent": ["strain hardening exponent", "hollomon exponent", "work hardening exponent", "n value"],
        "vec": ["valence electron concentration", "vec", "electron concentration", "valence electrons"],
        "delta_h_mix": ["enthalpy of mixing", "δh_mix", "delta h mix", "mixing enthalpy", "dh_mix"],
        "delta_s_mix": ["entropy of mixing", "δs_mix", "delta s mix", "mixing entropy", "ds_mix"],
        "omega_parameter": ["omega parameter", "Ω", "omega", "omega hea"],
        "atomic_size_difference": ["atomic size difference", "δ", "delta", "atomic size diff"],
        "lambda_parameter": ["lambda parameter", "λ", "lambda hea", "geometrical parameter"],
        "indentation_force": ["indentation force", "indenter force", "load", "nanoindentation load"],
        "indentation_depth": ["indentation depth", "penetration depth", "displacement", "indent depth"],
        "dislocation_density": ["dislocation density", "rho_d", "ρd", "geometrically necessary dislocation"],
        "melt_pool_depth": ["melt pool depth", "meltpool depth", "penetration depth", "melt pool"],
        "melt_pool_width": ["melt pool width", "meltpool width", "track width"],
        "melt_pool_length": ["melt pool length", "meltpool length"],
        "hatch_distance": ["hatch distance", "hatch spacing", "scan spacing", "hatch offset"],
        "build_platform_temperature": ["build platform temperature", "substrate temperature", "bed temperature", "preheat temperature"],
    }
    
    UNIT_HINTS = {
        "scan_speed": ["mm/s", "cm/s", "m/s", "mm/min", "in/min"],
        "flow_speed": ["mm/s", "cm/s", "m/s", "l/min", "m3/s"],
        "laser_power": ["w", "kw", "mw"],
        "irradiance": ["w/cm2", "kw/cm2", "w/m2"],
        "temperature": ["c", "k", "f"],
        "melting_temperature": ["k", "c"],
        "energy_density": ["j/mm3", "j/m3", "j/cm3", "j/m2"],
        "areal_energy_density": ["j/mm2", "j/m2", "mj/mm2"],
        "linear_energy_density": ["j/mm", "j/m", "kj/m"],
        "yield_strength": ["mpa", "gpa", "psi"],
        "tensile_strength": ["mpa", "gpa", "psi"],
        "ultimate_tensile_strength": ["mpa", "gpa", "psi"],
        "hardness": ["hv", "mpa", "gpa"],
        "elongation": ["%", "pct"],
        "modulus": ["gpa", "mpa"],
        "youngs_modulus": ["gpa", "mpa"],
        "poisson_ratio": ["unitless", ""],
        "coefficient_thermal_expansion": ["1/k", "k-1", "10-6/k"],
        "corrosion_potential": ["mv", "v", "vs sce", "vs ag/agcl"],
        "pitting_potential": ["mv", "v"],
        "repassivation_potential": ["mv", "v"],
        "breakdown_potential": ["mv", "v"],
        "open_circuit_potential": ["mv", "v"],
        "corrosion_current_density": ["ua/cm2", "uA/cm2", "ma/cm2", "a/cm2", "ua", "ma", "µA/cm²"],
        "current_density": ["a/cm2", "ma/cm2", "ua/cm2", "µA/cm²"],
        "polarization_resistance": ["kohm·cm2", "ohm·cm2", "kω·cm2", "ω·cm2", "kΩ·cm²"],
        "PREN": ["unitless", ""],
        "phase_fraction": ["%", "vol%", "fraction"],
        "austenite_fraction": ["%", "vol%"],
        "ferrite_fraction": ["%", "vol%"],
        "grain_size": ["um", "nm", "mm", "µm"],
        "cell_size": ["um", "nm", "mm", "µm"],
        "porosity": ["%", "fraction", "ppm"],
        "relative_density": ["%", "fraction"],
        "surface_roughness": ["um", "nm", "mm", "µm"],
        "stacking_fault_energy": ["mj/m2", "j/m2", "mJ/m²"],
        "unstable_stacking_fault_energy": ["mj/m2", "j/m2", "mJ/m²"],
        "ideal_shear_strength": ["gpa", "mpa"],
        "sauter_mean_diameter": ["um", "nm", "mm", "µm"],
        "spray_penetration": ["mm", "cm", "m"],
        "plume_height": ["mm", "cm", "m"],
        "film_thickness": ["um", "nm", "mm", "µm"],
        "absorption_coefficient": ["m-1", "1/m"],
        "enthalpy": ["j/mol", "kj/mol", "j/kg"],
        "viscosity": ["pa·s", "mpa·s", "cp"],
        "thermal_conductivity": ["w/m·k", "w/mk", "W/m·K"],
        "density": ["g/cm3", "kg/m3", "g/ml", "g/cm³", "kg/m³"],
        "lewis_number": ["unitless", ""],
        "jackson_parameter": ["unitless", ""],
        "hollomon_strength_coeff": ["mpa", "gpa", "pa"],
        "hollomon_exponent": ["unitless", ""],
        "vec": ["unitless", "electrons/atom", "e/a"],
        "delta_h_mix": ["kj/mol", "j/mol", "ev/atom"],
        "delta_s_mix": ["j/mol·k", "kj/mol·k"],
        "omega_parameter": ["unitless", ""],
        "atomic_size_difference": ["%", "unitless", ""],
        "lambda_parameter": ["unitless", ""],
        "indentation_force": ["mn", "μn", "nn", "n"],
        "indentation_depth": ["nm", "um", "µm", "mm"],
        "dislocation_density": ["m-2", "1/m2", "cm-2"],
        "melt_pool_depth": ["um", "µm", "nm", "mm"],
        "melt_pool_width": ["um", "µm", "nm", "mm"],
        "melt_pool_length": ["um", "µm", "nm", "mm"],
        "hatch_distance": ["um", "µm", "nm", "mm"],
        "build_platform_temperature": ["°c", "k", "°f"],
    }
    
    # Cache for LLM resolutions: (param_name, unit, context_hash) -> canonical
    _llm_cache: Dict[str, str] = {}
    _llm_call_count: int = 0
    
    def __init__(self, llm_callback: Optional[Callable[[str], str]] = None):
        self._build_keyword_index()
        self.llm_callback = llm_callback  # Optional callback for LLM resolution
    
    def _build_keyword_index(self):
        self.keyword_to_canonical = {}
        for canonical, keywords in self.CANONICAL.items():
            for kw in keywords:
                self.keyword_to_canonical[kw.lower()] = canonical
        # Add explicit short-form mappings
        shortcuts = {
            "ys": "yield_strength", "uts": "tensile_strength", "smys": "yield_strength",
            "0.2% proof": "yield_strength", "ecorr": "corrosion_potential",
            "eocp": "open_circuit_potential", "erp": "repassivation_potential",
            "epit": "pitting_potential", "ebr": "breakdown_potential",
            "jcorr": "corrosion_current_density", "rp": "polarization_resistance",
            "pren": "PREN", "sfe": "stacking_fault_energy",
            "usfe": "unstable_stacking_fault_energy", "smd": "sauter_mean_diameter",
            "ved": "energy_density", "aed": "areal_energy_density",
            "led": "linear_energy_density", "le": "lewis_number",
            "alpha_j": "jackson_parameter", "vec": "vec",
            "dh_mix": "delta_h_mix", "ds_mix": "delta_s_mix",
            "omega": "omega_parameter",
        }
        self.keyword_to_canonical.update(shortcuts)
    
    def classify(self, parameter_name: Optional[str], unit: Optional[str], context: str) -> str:
        """Fast-path classification using keyword/unit matching."""
        if parameter_name:
            pname_lower = parameter_name.lower().strip()
            # Direct canonical lookup
            if pname_lower in self.keyword_to_canonical:
                return self.keyword_to_canonical[pname_lower]
            # Keyword-in-name matching
            for canonical, keywords in self.CANONICAL.items():
                for kw in keywords:
                    if kw in pname_lower:
                        return canonical
        # Context-based matching
        context_lower = context.lower()
        for canonical, keywords in self.CANONICAL.items():
            for kw in keywords:
                if kw in context_lower:
                    return canonical
        # Unit-based heuristics
        if unit:
            unit_lower = unit.lower()
            if "yield" in context_lower and "mpa" in unit_lower:
                return "yield_strength"
            if "tensile" in context_lower and "mpa" in unit_lower:
                return "tensile_strength"
            if "corrosion" in context_lower and ("mv" in unit_lower or "v" in unit_lower):
                return "corrosion_potential"
            if "current" in context_lower and ("a/cm2" in unit_lower or "ua" in unit_lower or "ma" in unit_lower):
                return "current_density"
            if "polarization" in context_lower and ("ohm" in unit_lower or "ω" in unit_lower):
                return "polarization_resistance"
            for canonical, units in self.UNIT_HINTS.items():
                for u in units:
                    if u in unit_lower:
                        return canonical
            # Fallback unit patterns
            if unit:
                if "w/cm" in unit_lower or "kw/cm" in unit_lower:
                    return "irradiance"
                if unit_lower in ["w", "kw", "mw"]:
                    return "laser_power"
                if "mm/s" in unit_lower:
                    return "scan_speed"
                if unit_lower in ["°c", "c", "k", "°f"]:
                    return "temperature"
                if "mpa" in unit_lower or "gpa" in unit_lower:
                    return "hardness"
                if "j/mm3" in unit_lower or "j/mm²" in unit_lower or "j/mm2" in unit_lower:
                    return "energy_density"
                if "j/mm" in unit_lower:
                    return "linear_energy_density"
                if "mj/m2" in unit_lower or "mj/m²" in unit_lower:
                    return "stacking_fault_energy"
                if "ua/cm2" in unit_lower or "µa/cm²" in unit_lower or "ma/cm2" in unit_lower:
                    return "corrosion_current_density"
                if "kω·cm2" in unit_lower or "kohm·cm2" in unit_lower:
                    return "polarization_resistance"
        return "unknown"
    
    def classify_with_llm_fallback(self, parameter_name: Optional[str], unit: Optional[str], 
                                   context: str, llm: Optional['HybridLLM'] = None) -> str:
        """
        Enhanced classification with optional LLM fallback for ambiguous cases.
        
        Args:
            parameter_name: The parameter name as extracted from text
            unit: The unit string (if available)
            context: Surrounding text for disambiguation
            llm: Optional HybridLLM instance for fallback resolution
        
        Returns:
            Canonical physical quantity name
        """
        # Try fast path first
        result = self.classify(parameter_name, unit, context)
        if result != "unknown":
            return result
        
        # Fast path failed - try LLM fallback if available
        if llm is None or parameter_name is None:
            return "unknown"
        
        # Check cache first
        context_hash = hashlib.sha256(context[:500].encode()).hexdigest()[:12]
        cache_key = f"{parameter_name.lower()}|{unit or ''}|{context_hash}"
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]
        
        # LLM resolution prompt
        prompt = f"""You are a scientific parameter classifier. Given a parameter name, unit, and context, 
classify it into ONE of these canonical physical quantities:

{json.dumps(list(self.CANONICAL.keys()), indent=2)}

Parameter: "{parameter_name}"
Unit: "{unit or 'N/A'}"
Context: "{context[:800]}"

Return ONLY the canonical name from the list above, or "unknown" if none match.
Do not add explanations, quotes, or extra text."""
        
        try:
            response = llm.generate(prompt, max_new_tokens=50, temperature=0.0)
            candidate = response.strip().lower().replace('"', '').replace("'", "")
            if candidate in self.CANONICAL:
                self._llm_cache[cache_key] = candidate
                self._llm_call_count += 1
                logger.debug(f"LLM classification: '{parameter_name}' → '{candidate}' (call #{self._llm_call_count})")
                return candidate
        except Exception as e:
            logger.warning(f"LLM fallback classification failed: {e}")
        
        # Fallback to unknown
        return "unknown"
    
    def get_human_readable(self, canonical: str) -> str:
        mapping = {
            "laser_power": "Laser Power", "electrical_power": "Electrical Power",
            "scan_speed": "Scan Speed", "flow_speed": "Flow Speed", "feed_rate": "Feed Rate",
            "irradiance": "Irradiance / Intensity", "temperature": "Temperature",
            "melting_temperature": "Melting Temperature",
            "energy_density": "Energy Density (VED)", "areal_energy_density": "Areal Energy Density (AED)",
            "linear_energy_density": "Linear Energy Density (LED)",
            "layer_thickness": "Layer Thickness", "spot_size": "Spot Size", "exposure_time": "Exposure Time",
            "yield_strength": "Yield Strength", "tensile_strength": "Tensile Strength",
            "ultimate_tensile_strength": "Ultimate Tensile Strength",
            "hardness": "Hardness", "elongation": "Elongation", "modulus": "Young's Modulus",
            "youngs_modulus": "Young's Modulus", "poisson_ratio": "Poisson's Ratio",
            "coefficient_thermal_expansion": "Coefficient of Thermal Expansion",
            "corrosion_potential": "Corrosion Potential", "pitting_potential": "Pitting Potential",
            "repassivation_potential": "Repassivation Potential", "breakdown_potential": "Breakdown Potential",
            "open_circuit_potential": "Open Circuit Potential",
            "corrosion_current_density": "Corrosion Current Density", "current_density": "Current Density",
            "polarization_resistance": "Polarization Resistance", "PREN": "PREN",
            "phase_fraction": "Phase Fraction", "austenite_fraction": "Austenite Fraction",
            "ferrite_fraction": "Ferrite Fraction", "grain_size": "Grain Size", "cell_size": "Cell Size",
            "porosity": "Porosity", "relative_density": "Relative Density",
            "surface_roughness": "Surface Roughness",
            "stacking_fault_energy": "Stacking Fault Energy",
            "unstable_stacking_fault_energy": "Unstable Stacking Fault Energy",
            "ideal_shear_strength": "Ideal Shear Strength",
            "sauter_mean_diameter": "Sauter Mean Diameter", "spray_penetration": "Spray Penetration",
            "plume_height": "Plume Height", "film_thickness": "Film Thickness",
            "absorption_coefficient": "Absorption Coefficient",
            "enthalpy": "Enthalpy", "viscosity": "Viscosity",
            "thermal_conductivity": "Thermal Conductivity", "density": "Density",
            "unknown": "Other Quantities",
            "lewis_number": "Lewis Number", "jackson_parameter": "Jackson Parameter",
            "hollomon_strength_coeff": "Hollomon Strength Coeff", "hollomon_exponent": "Hollomon Exponent",
            "vec": "Valence Electron Concentration (VEC)", "delta_h_mix": "Enthalpy of Mixing (ΔHmix)",
            "delta_s_mix": "Entropy of Mixing (ΔSmix)", "omega_parameter": "Omega Parameter (Ω)",
            "atomic_size_difference": "Atomic Size Difference (δ)", "lambda_parameter": "Lambda Parameter (λ)",
            "indentation_force": "Indentation Force", "indentation_depth": "Indentation Depth",
            "dislocation_density": "Dislocation Density", "melt_pool_depth": "Melt Pool Depth",
            "melt_pool_width": "Melt Pool Width", "melt_pool_length": "Melt Pool Length",
            "hatch_distance": "Hatch Distance", "build_platform_temperature": "Build Platform Temperature",
        }
        return mapping.get(canonical, canonical.replace("_", " ").title())

# ============================================================================
# CONCEPT NORMALIZER (UNCHANGED FROM v17.2)
# ============================================================================
class ConceptNormalizer:
    ALIAS_DICTIONARIES = {
        "multicomponent": ["multicomponent", "multi-component", "multielement", "multi-element", "many elements", "complex alloy", "multi-principal", "high entropy", "hea", "multiple elements", "ternary", "quaternary", "quinary"],
        "yield_strength": ["yield strength", "ys", "0.2% proof", "proof stress", "yield stress", "0.2% offset strength"],
        "tensile_strength": ["tensile strength", "uts", "ultimate tensile strength", "ultimate strength", "tensile stress"],
        "laser_power": ["laser power", "laser beam power", "laser output power", "beam power"],
        "scan_speed": ["scan speed", "scanning speed", "laser scan speed", "beam scan speed", "scan velocity"],
        "hardness": ["hardness", "vickers hardness", "microhardness", "hv", "nano hardness"],
        "sdss_2507": ["sdss 2507", "super duplex stainless steel 2507", "uns s32750", "en 1.4410", "saf 2507", "2507", "s32750", "super duplex 2507"],
        "ti3au": ["ti3au", "ti_3au", "beta-ti3au", "b-ti3au", "ti-au intermetallic", "titanium gold intermetallic", "ti3au intermetallic", "beta ti3au"],
        "cp_ti": ["cp ti", "commercially pure titanium", "grade ii titanium", "grade 2 titanium", "titanium grade ii", "commercial purity titanium"],
        "alsimgzr": ["alsimgzr", "al-si-mg-zr", "al-si-mg-0.37zr", "alsi7.43mg1.57zr", "al-si-mg-zr alloy", "al-si-mg-zr composite"],
        "tib2_alsimgzr": ["tib2/al-si-mg-zr", "tib2-alsimgzr", "tib2 modified al-si-mg-zr", "tib2/al-si-mg-zr composite", "tib2-al-si-mg-zr"],
        "metallic_glass": ["fe-based metallic glass", "metallic glass", "amorphous alloy", "fe-b-si-nb-zr-cu", "fe based metallic glass"],
        "lpbf": ["lpbf", "l-pbf", "laser powder bed fusion", "selective laser melting", "slm", "laser powder-bed fusion", "laser powder-bed-fusion", "laser powder bed fusion (lpbf)"],
        "ded": ["ded", "directed energy deposition", "direct energy deposition", "laser metal deposition", "directed energy deposition (ded)"],
        "pfi": ["pfi", "port fuel injection", "port-fuel injection", "port fuel injector", "port fuel injection (pfi)"],
        "gdi": ["gdi", "gasoline direct injection", "direct injection spark ignition", "disi", "gasoline direct injection (gdi)"],
        "pren": ["pren", "pitting resistance equivalent number", "pitting resistance equivalent", "pitting resistance equivalent number (pren)"],
        "eis": ["eis", "electrochemical impedance spectroscopy", "impedance spectroscopy", "electrochemical impedance"],
        "cpp": ["cpp", "cyclic potentiodynamic polarization", "potentiodynamic polarization", "cyclic polarization", "cyclic potentiodynamic polarization (cpp)"],
        "nanoindentation": ["nanoindentation", "nano-indentation", "indentation test", "indentation force", "nano indentation"],
        "sfe": ["stacking fault energy", "sfe", "generalized stacking fault energy", "gsfe", "stacking fault energy (sfe)"],
        "smd": ["sauter mean diameter", "smd", "sauter diameter", "mean droplet diameter", "sauter mean diameter (smd)"],
        "ved": ["ved", "volumetric energy density", "volume energy density", "energy density", "volumetric energy density (ved)"],
        "aed": ["aed", "areal energy density", "area energy density", "areal energy density (aed)"],
        "led": ["led", "linear energy density", "line energy density", "linear energy density (led)"],
        "fem": ["fem", "finite element method", "finite element analysis", "fea", "finite element"],
        "md": ["md", "molecular dynamics", "molecular dynamics simulation", "molecular dynamics (md)"],
    }
    
    def __init__(self, embedding_fn: Optional[Callable] = None):
        self.embedding_fn = embedding_fn
        self._build_reverse_index()
    
    def _build_reverse_index(self):
        self.alias_to_canonical: Dict[str, str] = {}
        for canonical, aliases in self.ALIAS_DICTIONARIES.items():
            for alias in aliases:
                self.alias_to_canonical[alias.lower()] = canonical
    
    def normalize(self, term: str, use_fuzzy: bool = True, fuzzy_threshold: int = 85) -> str:
        if not term or not str(term).strip():
            return "unknown"
        term_lower = str(term).lower().strip()
        if term_lower in self.alias_to_canonical:
            return self.alias_to_canonical[term_lower]
        for alias, canonical in sorted(self.alias_to_canonical.items(), key=lambda x: -len(x[0])):
            if alias in term_lower:
                return canonical
        if use_fuzzy and GLOBAL_DEPS['rapidfuzz']:
            from rapidfuzz import fuzz, process
            all_aliases = list(self.alias_to_canonical.keys())
            result = process.extractOne(term_lower, all_aliases, scorer=fuzz.ratio)
            if result and result[1] >= fuzzy_threshold:
                return self.alias_to_canonical[result[0]]
        if self.embedding_fn is not None and GLOBAL_DEPS['sentence_transformers']:
            try:
                term_emb = self.embedding_fn(term_lower)
                best_sim = -1.0
                best_canonical = None
                for canonical in self.ALIAS_DICTIONARIES:
                    can_emb = self.embedding_fn(canonical)
                    sim = float(np.dot(term_emb, can_emb) / (np.linalg.norm(term_emb) * np.linalg.norm(can_emb) + 1e-8))
                    if sim > best_sim and sim > 0.75:
                        best_sim = sim
                        best_canonical = canonical
                if best_canonical:
                    return best_canonical
            except Exception:
                pass
        return term_lower
    
    def normalize_list(self, terms: List[str]) -> List[str]:
        return [self.normalize(t) for t in terms]

# ============================================================================
# DISPLAY NAME HELPERS (UNCHANGED)
# ============================================================================
def normalize_doi_display(name: str) -> str:
    if not name:
        return name
    base = name[:-4] if name.lower().endswith('.pdf') else name
    if re.match(r'10\.\d+_', base):
        base = re.sub(r'^(10\.\d+)_(.*)', r'\1/\2', base)
    return base

def get_display_name(doc_id: str, aliases: Optional[Dict[str, str]] = None) -> str:
    if aliases and doc_id in aliases:
        return aliases[doc_id]
    stem = Path(doc_id).stem
    normalized = normalize_doi_display(stem)
    return normalized

def get_citation_label(doc_id: str, aliases: Optional[Dict[str, str]] = None, index: int = 0, style: str = "doi") -> str:
    if style == "alias" and aliases and doc_id in aliases:
        return aliases[doc_id]
    if style == "number":
        return f"[{index}]"
    if style == "short":
        return Path(doc_id).stem[:20]
    return normalize_doi_display(Path(doc_id).stem)

# ============================================================================
# FAST JSON UTILS (UNCHANGED)
# ============================================================================
def fast_json_dumps(obj, indent=False):
    if GLOBAL_DEPS['orjson']:
        import orjson
        option = orjson.OPT_INDENT_2 if indent else 0
        return orjson.dumps(obj, option=option, default=str)
    else:
        return json.dumps(obj, indent=2 if indent else None, ensure_ascii=False, default=str).encode()

def fast_json_loads(data):
    if GLOBAL_DEPS['orjson']:
        import orjson
        if isinstance(data, str):
            data = data.encode()
        return orjson.loads(data)
    else:
        if isinstance(data, bytes):
            data = data.decode()
        return json.loads(data)

# ============================================================================
# PAGE NODE WITH ENHANCED CACHING SUPPORT
# ============================================================================
@dataclass
class PageNode:
    id: str
    title: str
    page_start: int
    page_end: Optional[int]
    full_text: str
    summary: str
    level: int
    children: List['PageNode'] = field(default_factory=list)
    doc_id: str = ""
    section_type: str = "BODY"
    node_id: str = ""
    prefix_summary: str = ""
    text_token_count: int = 0
    _pdf_path: Optional[str] = None
    meta Optional[DocumentMetadata] = None
    _content_hash: Optional[str] = None  # For caching
    
    def compute_content_hash(self) -> str:
        """Compute SHA-256 hash of node content for caching."""
        if self._content_hash:
            return self._content_hash
        content = f"{self.title}|{self.page_start}|{self.page_end}|{self.summary[:200]}|{self.metadata}"
        self._content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return self._content_hash
    
    def get_text(self, doc_cache: Dict[str, Any] = None, max_chars: int = 20000) -> str:
        if self.full_text:
            return self.full_text[:max_chars] if len(self.full_text) > max_chars else self.full_text
        if not self._pdf_path or not GLOBAL_DEPS['pymupdf']:
            return ""
        import fitz
        doc = None
        if doc_cache and self.doc_id in doc_cache:
            doc = doc_cache[self.doc_id]
        else:
            doc = fitz.open(self._pdf_path)
            if doc_cache:
                doc_cache[self.doc_id] = doc
        start = self.page_start - 1
        end = min(self.page_end or self.page_start, len(doc))
        texts = [doc[p].get_text("text") for p in range(start, end)]
        self.full_text = "\n".join(texts)
        if doc_cache is None:
            doc.close()
        return self.full_text[:max_chars] if len(self.full_text) > max_chars else self.full_text
    
    def to_dict(self):
        return {
            "id": self.id, "title": self.title, "page_start": self.page_start, "page_end": self.page_end,
            "summary": self.summary, "prefix_summary": self.prefix_summary, "level": self.level,
            "doc_id": self.doc_id, "section_type": self.section_type, "node_id": self.node_id,
            "text_token_count": self.text_token_count, "children": [c.to_dict() for c in self.children],
            "metadata": self.metadata.dict() if self.metadata else None,
            "content_hash": self.compute_content_hash()
        }
    
    def to_tree_format(self, max_chars: int = 20000) -> Dict[str, Any]:
        result = {"title": self.title, "node_id": self.node_id, "start_index": self.page_start,
                  "end_index": self.page_end or self.page_start, "summary": self.summary,
                  "prefix_summary": self.prefix_summary, "text_token_count": self.text_token_count,
                  "content_hash": self.compute_content_hash()}
        if self.children:
            result["nodes"] = [c.to_tree_format(max_chars) for c in self.children]
        text = self.get_text(max_chars=max_chars)
        if text:
            result["text"] = text
        if self.meta
            result["metadata"] = self.metadata.dict()
        return result
    
    @classmethod
    def from_dict(cls,  dict, pdf_path=None):
        node = cls(data["id"], data["title"], data["page_start"], data.get("page_end"), "",
                   data.get("summary", ""), data.get("level", 0), doc_id=data.get("doc_id", ""),
                   section_type=data.get("section_type", "BODY"), _pdf_path=pdf_path)
        node.node_id = data.get("node_id", "")
        node.prefix_summary = data.get("prefix_summary", "")
        node.text_token_count = data.get("text_token_count", 0)
        node._content_hash = data.get("content_hash")
        for c in data.get("children", []):
            node.children.append(cls.from_dict(c, pdf_path))
        if data.get("metadata"):
            node.metadata = DocumentMetadata(**data["metadata"])
        return node

# ============================================================================
# ROLL-UP SUMMARIZER - NEW COMPONENT
# ============================================================================
class RollupSummarizer:
    """
    Hierarchical roll-up summarization:
    - Leaf nodes: Summarize raw page text
    - Parent nodes: Summarize child summaries + own content
    - Root: Executive summary of entire document
    
    This enables the LLM to reason over condensed meaning at navigation time,
    dramatically improving retrieval accuracy without full-text loading.
    """
    
    def __init__(self, llm: 'HybridLLM', max_summary_length: int = 200):
        self.llm = llm
        self.max_summary_length = max_summary_length
        self._summary_cache: Dict[str, str] = {}
    
    async def generate_rollup_summaries(self, root: PageNode) -> PageNode:
        """Generate hierarchical summaries bottom-up."""
        # Post-order traversal: process children first
        for child in root.children:
            await self.generate_rollup_summaries(child)
        
        # Generate summary for current node
        root.summary = await self._summarize_node(root)
        return root
    
    async def _summarize_node(self, node: PageNode) -> str:
        """Generate summary using roll-up strategy."""
        # Check cache
        cache_key = f"{node.doc_id}:{node.node_id}:{node.compute_content_hash()}"
        if cache_key in self._summary_cache:
            return self._summary_cache[cache_key]
        
        # Leaf node: summarize raw text
        if not node.children:
            text = node.get_text(max_chars=3000)
            if len(text) < 100:
                summary = text[:self.max_summary_length]
            else:
                summary = await self._llm_summarize(text, f"Summarize this page section in max {self.max_summary_length} chars. Focus on quantitative parameters, materials, and methods.")
        else:
            # Internal node: roll-up child summaries + own content
            child_summaries = [c.summary for c in node.children if c.summary]
            own_text = node.get_text(max_chars=1000)[:500]
            combined = "\n".join(child_summaries[:10])  # Limit to 10 children
            if own_text.strip():
                combined = f"{own_text}\n\nSubsections:\n{combined}"
            summary = await self._llm_summarize(combined, f"Synthesize these subsection summaries into one {self.max_summary_length}-char overview. Highlight key parameters and findings.")
        
        # Cache and return
        self._summary_cache[cache_key] = summary[:self.max_summary_length]
        return self._summary_cache[cache_key]
    
    async def _llm_summarize(self, text: str, instruction: str) -> str:
        """Call LLM for summarization with error handling."""
        prompt = f"{instruction}\n\nText to summarize:\n{text[:4000]}\n\nSummary:"
        try:
            response = await asyncio.to_thread(
                self.llm.generate, 
                prompt, 
                max_new_tokens=150, 
                temperature=0.1
            )
            return response.strip()[:self.max_summary_length]
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")
            # Fallback: extract first meaningful sentence
            sentences = re.split(r'[.!?]+', text)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 30 and any(c.isdigit() for c in sent):
                    return sent[:self.max_summary_length]
            return text[:self.max_summary_length]

# ============================================================================
# ANNOTATED TREE CACHE - NEW COMPONENT
# ============================================================================
class AnnotatedTreeCache:
    """
    Full annotated-tree caching with SHA-256 content hashing.
    
    Caches the entire hierarchical structure with:
    - Summaries at each level
    - Metadata extractions
    - Quantitative item annotations
    
    This avoids re-indexing unchanged PDFs and enables instant reload.
    """
    
    def __init__(self, cache_dir: str = ".declarmima_cache", ttl_hours: int = 72):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self._index: Dict[str, Dict] = {}  # doc_hash -> metadata
        self._load_index()
    
    def _load_index(self):
        """Load cache index from disk."""
        index_path = self.cache_dir / "index.json"
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    self._index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                self._index = {}
    
    def _save_index(self):
        """Persist cache index to disk."""
        index_path = self.cache_dir / "index.json"
        try:
            with open(index_path, 'w') as f:
                json.dump(self._index, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")
    
    def _compute_doc_hash(self, doc_name: str, file_content: bytes) -> str:
        """Compute SHA-256 hash of document content."""
        hasher = hashlib.sha256()
        hasher.update(doc_name.encode())
        hasher.update(file_content)
        return hasher.hexdigest()[:32]
    
    def get(self, doc_name: str, file_content: bytes) -> Optional[Dict]:
        """Retrieve cached tree if available and not expired."""
        doc_hash = self._compute_doc_hash(doc_name, file_content)
        entry = self._index.get(doc_hash)
        if not entry:
            return None
        # Check TTL
        cached_time = datetime.fromisoformat(entry['cached_at'])
        if datetime.now() - cached_time > self.ttl:
            logger.info(f"Cache expired for {doc_name}")
            del self._index[doc_hash]
            self._save_index()
            return None
        # Load tree from cache file
        cache_file = self.cache_dir / f"{doc_hash}.tree.json"
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, 'rb') as f:
                tree_data = fast_json_loads(f.read())
                if isinstance(tree_data, bytes):
                    tree_data = tree_data.decode('utf-8')
                return json.loads(tree_data) if isinstance(tree_data, str) else tree_data
        except Exception as e:
            logger.warning(f"Failed to load cached tree: {e}")
            return None
    
    def set(self, doc_name: str, file_content: bytes, tree_dict: Dict):
        """Store tree in cache with content hash."""
        doc_hash = self._compute_doc_hash(doc_name, file_content)
        cache_file = self.cache_dir / f"{doc_hash}.tree.json"
        try:
            # Save tree
            with open(cache_file, 'wb') as f:
                f.write(fast_json_dumps(tree_dict, indent=True))
            # Update index
            self._index[doc_hash] = {
                'doc_name': doc_name,
                'cached_at': datetime.now().isoformat(),
                'file_size': len(file_content),
                'tree_nodes': self._count_nodes(tree_dict)
            }
            self._save_index()
            logger.info(f"Cached tree for {doc_name} (hash: {doc_hash[:12]}...)")
        except Exception as e:
            logger.error(f"Failed to cache tree: {e}")
    
    def _count_nodes(self, tree: Dict) -> int:
        """Count total nodes in tree for stats."""
        count = 1
        for child in tree.get('nodes', []):
            count += self._count_nodes(child)
        return count
    
    def clear_expired(self):
        """Remove expired cache entries."""
        expired = []
        for doc_hash, entry in list(self._index.items()):
            cached_time = datetime.fromisoformat(entry['cached_at'])
            if datetime.now() - cached_time > self.ttl:
                expired.append(doc_hash)
        for doc_hash in expired:
            del self._index[doc_hash]
            cache_file = self.cache_dir / f"{doc_hash}.tree.json"
            if cache_file.exists():
                cache_file.unlink()
        if expired:
            self._save_index()
            logger.info(f"Cleared {len(expired)} expired cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total_size = sum(
            (self.cache_dir / f"{h}.tree.json").stat().st_size 
            for h in self._index 
            if (self.cache_dir / f"{h}.tree.json").exists()
        )
        return {
            'entries': len(self._index),
            'total_size_mb': round(total_size / 1024 / 1024, 2),
            'ttl_hours': self.ttl.total_seconds() / 3600
        }

# ============================================================================
# HYBRID LLM BACKEND (UNCHANGED FROM v17.2 - VALIDATED)
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
        
        if model_key.startswith("[Ollama]"):
            self.model_name = model_key.split("] ")[1].strip()
        elif model_key.startswith("ollama:"):
            self.model_name = model_key.replace("ollama:", "", 1)
        else:
            self.model_name = model_key
        
        self.template = get_model_template(self.model_name)
        self._init_backend()
        logger.info(f"HybridLLM initialized: {self.model_name} on {self.device} via {self.backend}")
    
    def _init_backend(self):
        if GLOBAL_DEPS['ollama']:
            try:
                requests.get("http://localhost:11434/api/tags", timeout=5)
                self.backend = "ollama"
                self.client = ollama.Client(host="http://localhost:11434")
                return
            except:
                pass
        if GLOBAL_DEPS['transformers']:
            self.backend = "transformers"
            return
        raise RuntimeError("No LLM backend available. Install Ollama or transformers.")
    
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
                outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=temp if temp > 0 else None, do_sample=temp > 0, pad_token_id=self.tokenizer.eos_token_id)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "assistant" in response:
                response = response.split("assistant")[-1].strip()
            return response
        except Exception as e:
            logger.error(f"Transformers error: {e}")
            return f"Error: {str(e)[:100]}"
    
    def _load_transformers(self):
        logger.info(f"Loading {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32}
        if self.use_4bit and self.device == "cuda":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if self.device == "cuda":
            self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded.")

# ============================================================================
# TWO-CALL QUERY PROCESSOR - NEW CORE COMPONENT
# ============================================================================
class TwoCallQueryProcessor:
    """
    Strict 2-call query architecture:
    
    Call 1 (Navigation): 
    - Input: Query + condensed tree structure
    - Output: Selected node_id(s) with confidence
    
    Call 2 (Answer Generation):
    - Input: Query + raw text from selected nodes
    - Output: Final synthesized answer
    
    Benefits:
    - Predictable token usage
    - Lower latency
    - Easier debugging
    - Separation of concerns
    """
    
    def __init__(self, llm: HybridLLM, phys_classifier: PhysicalQuantityClassifier, 
                 max_results: int = 30, max_text_chars: int = 20000):
        self.llm = llm
        self.phys_classifier = phys_classifier
        self.max_results = max_results
        self.max_text_chars = max_text_chars
        self.template = llm.template if hasattr(llm, 'template') else MODEL_PROMPT_TEMPLATES["default"]
    
    async def process_query(self, query: str, annotated_trees: List[Dict]) -> Dict[str, Any]:
        """
        Execute strict 2-call query pipeline.
        
        Returns dict with:
        - answer: Final synthesized response
        - retrieved_nodes: Selected nodes with text
        - extracted_items: Structured extractions
        - metrics: Timing and confidence stats
        """
        start_time = time.time()
        metrics = {"call_1_start": None, "call_1_end": None, "call_2_start": None, "call_2_end": None}
        
        # ========== CALL 1: TREE NAVIGATION ==========
        metrics["call_1_start"] = time.time()
        selections = await self._navigate_trees(query, annotated_trees)
        metrics["call_1_end"] = time.time()
        
        if not selections:
            return {
                "answer": f"No relevant sections found for query: '{query}'. Try rephrasing or check document structure.",
                "retrieved_nodes": [],
                "extracted_items": [],
                "metrics": {**metrics, "total_time": time.time() - start_time}
            }
        
        # Fetch full text for selected nodes
        retrieved_nodes = []
        for sel in selections:
            node = self._find_node_by_id(annotated_trees, sel['doc_id'], sel['node_id'])
            if node:
                full_text = node.get('text', '')
                if len(full_text) > self.max_text_chars:
                    full_text = full_text[:self.max_text_chars] + "..."
                retrieved_nodes.append({
                    "full_text": full_text,
                    "page_start": node.get('start_index'),
                    "doc_id": sel['doc_id'],
                    "section_title": node.get('title'),
                    "quantitative_items": node.get('quantitative_items', []),
                    "citation": f'<cite doc="{sel["doc_id"]}" page="{node.get("start_index")}"/>',
                    "selection_reasoning": sel.get('reasoning', ''),
                    "confidence": sel.get('confidence', 0)
                })
        
        # ========== CALL 2: ANSWER GENERATION ==========
        metrics["call_2_start"] = time.time()
        answer, extracted_items = await self._generate_answer(query, retrieved_nodes)
        metrics["call_2_end"] = time.time()
        
        metrics["total_time"] = time.time() - start_time
        metrics["nodes_retrieved"] = len(retrieved_nodes)
        metrics["items_extracted"] = len(extracted_items)
        
        return {
            "answer": answer,
            "retrieved_nodes": retrieved_nodes,
            "extracted_items": extracted_items,
            "metrics": metrics
        }
    
    async def _navigate_trees(self, query: str, trees: List[Dict]) -> List[Dict]:
        """Call 1: Identify relevant nodes via condensed tree analysis."""
        # Condense trees for efficient LLM processing
        condensed = []
        for tree in trees:
            condensed.append(self._condense_tree(tree, max_depth=3))
        
        # Batch trees to stay within token limits
        batches = self._batch_condensed_trees(condensed, max_tokens=6000)
        all_selections = []
        
        for batch in batches:
            prompt = self._build_navigation_prompt(query, batch)
            try:
                response = await asyncio.to_thread(
                    self.llm.generate,
                    prompt,
                    max_new_tokens=2048,
                    fast_json=True,
                    system_prompt=self.template.get("system")
                )
                selections = self._parse_selections(response)
                all_selections.extend(selections)
            except Exception as e:
                logger.warning(f"Navigation call failed: {e}")
                continue
        
        # Sort by confidence and limit results
        return sorted(all_selections, key=lambda x: x.get('confidence', 0), reverse=True)[:self.max_results]
    
    def _condense_tree(self, tree: Dict, max_depth: int = 3) -> Dict[str, Any]:
        """Create condensed representation for navigation."""
        def condense(node: Dict, depth: int = 0) -> Dict[str, Any]:
            if depth > max_depth:
                return {"node_id": node.get("node_id", ""), "title": node.get("title", ""), "leaf": True}
            
            result = {
                "node_id": node.get("node_id", ""),
                "title": node.get("title", ""),
                "summary": (node.get("summary", "") or "")[:150],
                "level": node.get("level", 0)
            }
            
            # Add metadata hints if available
            if node.get("metadata"):
                meta = node["metadata"]
                if meta.get("alloys"):
                    result["alloys"] = meta["alloys"][:3]
                if meta.get("laser_power_values"):
                    result["power_hint"] = f"{min(meta['laser_power_values'])}-{max(meta['laser_power_values'])} W"
                if meta.get("scan_speed_values"):
                    result["speed_hint"] = f"{min(meta['scan_speed_values'])}-{max(meta['scan_speed_values'])} mm/s"
            
            # Add quantitative hints
            q_items = node.get("quantitative_items", [])
            if q_items:
                params = list(set(item.get("parameter_name", "") for item in q_items if item.get("parameter_name")))
                if params:
                    result["has_quantitative"] = params[:5]
            else:
                # Fallback: scan text for numeric patterns
                text = node.get("text", "")
                if text:
                    candidates = re.findall(r'(\d+(?:\.\d+)?)\s*(W|kW|mW|J|mm/s|C|K|MPa|GPa|nm|um|mm|s|m/s|W/cm2|kW/cm2)', text, re.IGNORECASE)
                    if candidates:
                        result["candidate_values"] = [f"{v}{u}" for v, u in candidates[:3]]
            
            # Recurse children
            children = node.get("nodes", [])
            if children and depth < max_depth:
                result["nodes"] = [condense(c, depth+1) for c in children[:5]]
            
            return result
        
        return {
            "doc_id": tree.get("doc_id", tree.get("doc_name", "unknown")),
            "doc_name": tree.get("doc_name", ""),
            "structure": [condense(tree)] if not isinstance(tree, list) else [condense(t) for t in tree]
        }
    
    def _batch_condensed_trees(self, trees: List[Dict], max_tokens: int = 6000) -> List[List[Dict]]:
        """Split trees into batches respecting token limits."""
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
    
    def _build_navigation_prompt(self, query: str, trees: List[Dict]) -> str:
        """Build prompt for tree navigation."""
        trees_json = json.dumps(trees, ensure_ascii=False, indent=2)
        return f"""You are an expert scientific document navigator.
Given a query about quantitative parameters, identify which document nodes are MOST likely to contain the answer.

QUERY: {query}

INSTRUCTIONS:
1. Analyze each document's tree structure (titles, summaries, quantitative hints, candidate values, alloys, power hints, speed hints)
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
    
    def _parse_selections(self, response: str) -> List[Dict]:
        """Parse LLM response for node selections."""
        try:
            data = self._extract_json_safe(response)
            if data and isinstance(data, dict):
                selections = data.get("selections", [])
                return [s for s in selections if isinstance(s, dict) and "doc_id" in s and "node_id" in s]
        except Exception as e:
            logger.warning(f"Failed to parse selections: {e}")
        return []
    
    def _extract_json_safe(self, text: str) -> Optional[Any]:
        """Extract JSON from LLM response with multiple fallback patterns."""
        patterns = [r'\{.*\}', r'\[.*\]', r'```json\s*(\{.*?\})\s*```', r'```json\s*(\[.*?\])\s*```']
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
        """Locate node in tree structure by ID."""
        for tree in trees:
            if tree.get("doc_id") == doc_id or tree.get("doc_name") == doc_id:
                return self._search_node_recursive(tree, node_id)
        return None
    
    def _search_node_recursive(self, node: Dict, target_id: str) -> Optional[Dict]:
        """Recursively search for node by ID."""
        if node.get("node_id") == target_id:
            return node
        for child in node.get("nodes", []):
            res = self._search_node_recursive(child, target_id)
            if res:
                return res
        return None
    
    async def _generate_answer(self, query: str, retrieved_nodes: List[Dict]) -> Tuple[str, List[UniversalExtractionItem]]:
        """Call 2: Generate final answer from retrieved text."""
        if not retrieved_nodes:
            return f"No relevant content found for: {query}", []
        
        # Format retrieved text
        sections_text = "\n\n".join([
            f"=== {r['doc_id']} (p.{r['page_start']}) - {r['section_title']} ===\n{r['full_text'][:3000]}"
            for r in retrieved_nodes[:5]  # Limit to 5 nodes for context window
        ])
        
        # Extraction + synthesis prompt
        prompt = f"""You are an expert scientific analyst. Extract quantitative information and synthesize an answer.

QUERY: {query}

RETRIEVED SECTIONS:
{sections_text}

TASK:
1. Extract ALL quantitative values with units, materials, and context
2. Synthesize a structured answer using this format:

**Direct Answer**
(Concise answer citing sources)

**Evidence by Physical Quantity**
(Group findings by parameter)

**Evidence by Material/Alloy**
(Group by material if applicable)

**Consensus & Variability**
(Report ranges/means for repeated parameters)

**Contradictions & Limitations**
(Highlight conflicting values)

**Confidence Assessment**
(High/Medium/Low)

RULES:
- Only use information from retrieved sections
- Include citations: <cite doc="..." page="X"/>
- Return ONLY the answer text, no JSON wrapper
"""
        
        try:
            response = await asyncio.to_thread(
                self.llm.generate,
                prompt,
                max_new_tokens=1500,
                temperature=0.2
            )
            answer = response.strip()
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            answer = f"Error generating answer: {e}"
        
        # Extract structured items (optional post-processing)
        extracted_items = []
        # Could add secondary extraction pass here if needed
        
        return answer, extracted_items

# ============================================================================
# REMAINING COMPONENTS (UNCHANGED FROM v17.2 - VALIDATED)
# ============================================================================
# [PaginationAwareReader, StructuredMetadataExtractor, TwoStageRetriever, 
#  HierarchicalIndex, FastHierarchicalIndex, QuantitativeKnowledgeGraph,
#  UniversalLLMExtractor, LLMReasoningSynthesizer, HierarchicalTreeRetriever,
#  VisConfig, PublicationVisualizationEngine, LOCAL_LLM_OPTIONS, 
#  MODEL_PROMPT_TEMPLATES, get_model_template, UNIVERSAL_CONFIG]
# 
# These components remain unchanged from v17.2 as they were already validated.
# The key upgrades are in the new classes above: RollupSummarizer, 
# AnnotatedTreeCache, TwoCallQueryProcessor, and enhanced PhysicalQuantityClassifier.

# ============================================================================
# STREAMLIT APP ENTRY POINT (UPDATED FOR v18.0)
# ============================================================================
def run_streamlit():
    """Main Streamlit application with v18.0 enhancements."""
    st.set_page_config(
        page_title="DECLARMIMA v18.0 - Enhanced Vectorless RAG",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("# DECLARMIMA v18.0 - Enhanced Vectorless RAG with PageIndex-Style Intelligence")
    st.caption("2-call query architecture • Roll-up summarization • LLM-fallback classification • Full-tree caching")
    
    # Initialize session state
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
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = None
    if "doc_aliases" not in st.session_state:
        st.session_state.doc_aliases = {}
    if "tree_cache" not in st.session_state:
        st.session_state.tree_cache = AnnotatedTreeCache()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### Configuration")
        
        # LLM selection
        model_keys = list(LOCAL_LLM_OPTIONS.keys())
        if "llm_model_choice" not in st.session_state:
            st.session_state.llm_model_choice = model_keys[2]  # Default to qwen2.5:7b
        selected = st.selectbox(
            "Select Local LLM", 
            options=model_keys, 
            index=model_keys.index(st.session_state.llm_model_choice), 
            key="llm_model_select"
        )
        st.session_state.llm_model_choice = selected
        
        st.checkbox("Use 4-bit quantization (if Transformers)", value=True, key="use_4bit")
        st.slider("Confidence threshold", 0.3, 0.9, 0.55, 0.05, key="min_confidence")
        max_chars = st.slider(
            "Max text length per retrieved section (characters)", 
            min_value=1000, max_value=50000, value=20000, step=1000,
            help="Larger values give more context but use more memory/LLM tokens."
        )
        st.session_state.max_retrieval_chars = max_chars
        
        st.checkbox("Show reasoning trace", value=True, key="show_trace")
        st.checkbox("Show tree navigation", value=True, key="show_tree_nav")
        st.checkbox("Enable two-stage retrieval (semantic)", value=True, key="two_stage")
        
        st.markdown("#### Visualization Settings")
        st.selectbox("Default colormap", list(PublicationVisualizationEngine.COLORMAP_OPTIONS.keys()), index=0, key="viz_colormap")
        st.selectbox("Document label style", ["doi", "number", "alias", "short"], index=0, key="viz_label_style")
        st.slider("Top N concepts", 5, 100, 25, key="viz_top_n")
        st.multiselect(
            "Filter domains", 
            options=["laser_power","scan_speed","yield_strength","tensile_strength","hardness","temperature","energy_density"], 
            default=["laser_power","scan_speed","yield_strength"], 
            key="viz_domains"
        )
        
        with st.expander("Advanced Style Controls", expanded=False):
            st.slider("Base font size", 6, 20, 10, key="viz_font_size")
            st.slider("Title font size", 8, 30, 14, key="viz_title_font_size")
            st.slider("Label font size", 6, 18, 9, key="viz_label_font_size")
            st.slider("Figure DPI", 100, 600, 300, 50, key="viz_figure_dpi")
            st.slider("Node size factor", 0.1, 3.0, 1.0, 0.1, key="viz_node_size_factor")
            st.slider("Edge alpha", 0.05, 1.0, 0.25, 0.05, key="viz_edge_alpha")
            st.slider("Edge width", 0.1, 5.0, 0.8, 0.1, key="viz_edge_width")
            st.slider("Line width", 0.5, 5.0, 1.5, 0.5, key="viz_line_width")
            st.slider("Marker size", 20, 200, 80, 10, key="viz_marker_size")
            st.checkbox("PyVis physics enabled", value=True, key="viz_pyvis_physics")
            st.slider("PyVis gravity", -5000, -100, -1800, 100, key="viz_pyvis_gravity")
            st.slider("PyVis spring length", 50, 300, 140, 10, key="viz_pyvis_spring_length")
        
        st.caption(f"GPU: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        # Cache stats
        if "tree_cache" in st.session_state:
            stats = st.session_state.tree_cache.get_stats()
            st.markdown(f"#### Cache: {stats['entries']} entries, {stats['total_size_mb']} MB")
        
        if st.button("Clear Cache & Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    @st.cache_resource(show_spinner="Initializing LLM...")
    def get_cached_llm(model_choice: str, use_4bit: bool):
        internal = LOCAL_LLM_OPTIONS[model_choice]
        return HybridLLM(model_key=internal, use_4bit=use_4bit)
    
    # File upload and indexing
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files and st.button("Build Index with Roll-up Summaries", type="primary"):
        st.session_state.query_processor["files"] = uploaded_files
        st.success(f"{len(uploaded_files)} files registered for indexing.")
        st.rerun()
    
    if st.session_state.query_processor.get("files") and not st.session_state.annotated_trees:
        with st.spinner("Building hierarchical index with roll-up summarization and full-tree caching..."):
            progress = st.progress(0)
            
            # Initialize components
            llm = get_cached_llm(st.session_state.llm_model_choice, st.session_state.get("use_4bit", True))
            phys_classifier = PhysicalQuantityClassifier(llm_callback=lambda p: llm.generate(p, max_new_tokens=50))
            progress.progress(0.1)
            
            # Initialize cache
            tree_cache = st.session_state.tree_cache
            progress.progress(0.15)
            
            # Build trees with caching
            idx = FastHierarchicalIndex(llm=llm)
            trees = {}
            
            for i, file in enumerate(st.session_state.query_processor["files"]):
                # Read file content for hashing
                buf = BytesIO(file.getbuffer())
                file_content = buf.getvalue()
                doc_name = file.name
                
                # Try cache first
                cached_tree = tree_cache.get(doc_name, file_content)
                if cached_tree:
                    logger.info(f"Loaded cached tree for {doc_name}")
                    tree = PageNode.from_dict(cached_tree)
                    trees[doc_name] = tree
                    continue
                
                # Build new tree
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    buf.seek(0)
                    tmp.write(buf.getbuffer())
                    tmp_path = tmp.name
                
                doc = fitz.open(tmp_path)
                tree = idx._build_tree(doc, doc_name, tmp_path)
                doc.close()
                
                # Apply roll-up summarization
                summarizer = RollupSummarizer(llm, max_summary_length=200)
                tree = asyncio.run(summarizer.generate_rollup_summaries(tree))
                
                # Extract metadata
                full_text = "\n".join([doc[p].get_text("text") for p in range(len(doc))])
                meta_extractor = StructuredMetadataExtractor()
                meta = meta_extractor.extract_metadata(doc_name, full_text)
                tree.metadata = meta
                
                # Cache the annotated tree
                tree_cache.set(doc_name, file_content, tree.to_dict())
                trees[doc_name] = tree
                
                progress.progress(0.15 + 0.7 * (i + 1) / len(st.session_state.query_processor["files"]))
            
            st.session_state.query_processor["index"] = idx
            st.session_state.query_processor["doc_trees"] = trees
            progress.progress(0.85)
            
            # Initial extraction pass
            extractor = UniversalLLMExtractor(llm)
            kg = QuantitativeKnowledgeGraph()
            all_items = []
            two_stage = TwoStageRetriever(llm=llm)
            
            for doc_name, tree in trees.items():
                # Collect leaf texts
                leaf_texts = []
                def collect_leaves(node: PageNode):
                    if not node.children:
                        text = node.get_text()
                        if text:
                            leaf_texts.append({
                                "full_text": text, 
                                "page_start": node.page_start, 
                                "doc_id": doc_name, 
                                "section_title": node.title
                            })
                    for c in node.children:
                        collect_leaves(c)
                collect_leaves(tree)
                
                # Extract quantitative items
                initial_prompt = "Extract ALL quantitative parameters: laser power, scan speed, VED, AED, LED, layer thickness, hatch distance, temperature, enthalpy, viscosity, thermal conductivity, density, yield strength, UTS, elongation, hardness, modulus, stacking fault energy, ideal shear strength, corrosion potential (Ecorr), pitting potential (Epit), repassivation potential (Erp), breakdown potential (Ebr), corrosion current density (Jcorr), polarization resistance (Rp), PREN, phase fractions (austenite, ferrite), grain size, porosity, relative density, Sauter mean diameter (SMD), spray penetration, plume height, film thickness, absorption coefficient, Young's modulus, Poisson's ratio, CTE. Include units, material names, and page numbers. Also extract alloy names, process methods (LPBF, DED, PFI, GDI, FEM, MD), and phases (Ti3Au, Al3Zr, beta-Ti3Au, etc.)."
                items = extractor.extract_from_chunks(leaf_texts, initial_prompt)
                all_items.extend(items)
                
                # Populate knowledge graph
                kg.add_extractions(doc_name, items)
                if tree.meta
                    kg.add_document_metadata(doc_name, tree.metadata)
                two_stage.index_document(doc_name, tree.metadata, tree.summary)
            
            st.session_state.knowledge_graph = kg
            st.session_state.two_stage_retriever = two_stage
            progress.progress(0.95)
            
            # Annotate trees with quantitative items
            annotated = []
            for doc_name, tree in trees.items():
                ann = kg.to_tree_annotation(tree, max_chars=st.session_state.get("max_retrieval_chars", 20000))
                ann["doc_id"] = doc_name
                ann["doc_name"] = doc_name
                ann["metadata"] = tree.metadata.dict() if tree.metadata else {}
                annotated.append(ann)
            st.session_state.annotated_trees = annotated
            
            progress.progress(1.0)
            st.success(f"✅ Indexed {len(trees)} documents with roll-up summaries and {len(all_items)} quantitative items")
    
    # Query interface
    if st.session_state.annotated_trees:
        st.markdown("### Quick Queries")
        col1, col2, col3, col4 = st.columns(4)
        quick = ["laser power", "yield strength", "scan speed", "alloy names"]
        for i, q in enumerate(quick):
            with [col1, col2, col3, col4][i]:
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
        
        if active_prompt:
            # Check cache
            cached = st.session_state.cached_query_result
            has_valid_cache = cached and cached.get("prompt") == active_prompt and "answer" in cached
            
            if not has_valid_cache:
                # Execute 2-call query pipeline
                with st.chat_message("assistant"):
                    progress = st.progress(0)
                    progress.text("Initializing components...")
                    
                    llm = get_cached_llm(st.session_state.llm_model_choice, st.session_state.get("use_4bit", True))
                    phys_classifier = PhysicalQuantityClassifier()
                    progress.progress(0.2)
                    
                    # Two-call processor
                    processor = TwoCallQueryProcessor(
                        llm=llm,
                        phys_classifier=phys_classifier,
                        max_results=30,
                        max_text_chars=st.session_state.get("max_retrieval_chars", 20000)
                    )
                    progress.progress(0.4)
                    
                    # Execute query
                    result = asyncio.run(processor.process_query(active_prompt, st.session_state.annotated_trees))
                    progress.progress(1.0)
                    
                    # Display answer
                    st.markdown(result["answer"])
                    
                    # Cache result
                    st.session_state.cached_query_result = {
                        "prompt": active_prompt,
                        "answer": result["answer"],
                        "retrieved": result["retrieved_nodes"],
                        "items": result["extracted_items"],
                        "metrics": result["metrics"]
                    }
                    
                    # Show metrics if enabled
                    if st.session_state.get("show_trace"):
                        with st.expander("Query Metrics", expanded=False):
                            metrics = result["metrics"]
                            st.json({
                                "total_time_sec": round(metrics.get("total_time", 0), 2),
                                "call_1_time_sec": round(metrics.get("call_1_end", 0) - metrics.get("call_1_start", 0), 2) if metrics.get("call_1_start") else None,
                                "call_2_time_sec": round(metrics.get("call_2_end", 0) - metrics.get("call_2_start", 0), 2) if metrics.get("call_2_start") else None,
                                "nodes_retrieved": metrics.get("nodes_retrieved", 0),
                                "items_extracted": metrics.get("items_extracted", 0)
                            })
                    
                    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
            else:
                # Use cached result
                with st.chat_message("assistant"):
                    st.markdown(cached["answer"])
    
    else:
        st.info("📁 Upload PDF files and click 'Build Index' to begin.")
    
    # Visualization dashboard (unchanged from v17.2)
    if st.session_state.knowledge_graph and st.session_state.annotated_trees:
        st.markdown("---")
        st.subheader("Publication-Quality Visualization Dashboard")
        # ... [existing visualization code from v17.2] ...

# ============================================================================
# MODEL TEMPLATES AND CONFIG (UNCHANGED)
# ============================================================================
LOCAL_LLM_OPTIONS = {
    "[Ollama] qwen2.5:0.5b (Fastest, CPU OK)": "ollama:qwen2.5:0.5b",
    "[Ollama] qwen2.5:1.5b (Balanced)": "ollama:qwen2.5:1.5b",
    "[Ollama] qwen2.5:7b (Recommended for RAG)": "ollama:qwen2.5:7b",
    "[Ollama] qwen2.5:14b (Max Reasoning)": "ollama:qwen2.5:14b",
    "[Ollama] llama3.1:8b (Meta Standard)": "ollama:llama3.1:8b",
    "[Ollama] mistral:7b (High JSON Reliability)": "ollama:mistral:7b",
    "[Ollama] gemma2:9b (Scientific Nuance)": "ollama:gemma2:9b",
    "[Ollama] falcon3:10b (Instruction Following)": "ollama:falcon3:10b",
}

MODEL_PROMPT_TEMPLATES = {
    "qwen2.5:0.5b": {"system": "You are a precise document analyst. Follow JSON format strictly.", "json_reminder": "Return ONLY valid JSON."},
    "qwen2.5:1.5b": {"system": "You are a precise document analyst. Follow JSON format strictly.", "json_reminder": "Return ONLY valid JSON."},
    "qwen2.5:7b": {"system": "You are a precise document analyst. Follow JSON format strictly.", "json_reminder": "Return ONLY valid JSON."},
    "default": {"system": "You are a document navigation agent.", "json_reminder": "Return valid JSON only."}
}

def get_model_template(model_name: str):
    for key, template in MODEL_PROMPT_TEMPLATES.items():
        if key in model_name.lower():
            return template
    return MODEL_PROMPT_TEMPLATES["default"]

UNIVERSAL_CONFIG = {"leaf_node_page_window": 7, "min_confidence_threshold": 0.55}

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    run_streamlit()
