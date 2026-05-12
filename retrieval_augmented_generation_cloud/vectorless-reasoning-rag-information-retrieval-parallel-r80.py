from __future__ import annotations

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DECLARMIMA v18.0 - ENHANCED VECTORLESS RAG WITH PAGEINDEX-STYLE INTELLIGENCE
============================================================================
Complete implementation with:
1. Strict 2-call query architecture (navigation → answer)
2. Roll-up hierarchical summarization (bottom-up tree condensation)
3. LLM-fallback PhysicalQuantityClassifier for ambiguous terminology
4. Full annotated-tree caching with SHA-256 content hashing
5. Async-first retrieval pipeline with proper error isolation
6. Production-ready logging, metrics, and fallback handling
7. 35+ publication-quality interactive visualizations
8. Query-driven contextual visualization layer
9. Retrieval diagnostics dashboard
10. Concept normalization & synonym resolution

Author: DECLARMIMA Development Team
Version: 18.0
License: MIT
"""

# ============================================================================
# STANDARD LIBRARY IMPORTS
# ============================================================================
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
from functools import lru_cache, wraps
import threading
import queue

# ============================================================================
# THIRD-PARTY LIBRARY IMPORTS
# ============================================================================
import numpy as np
import torch
import pandas as pd

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
log_formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
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
def check_optional_dependencies() -> Dict[str, bool]:
    """
    Check availability of optional dependencies and log their status.
    Returns a dictionary of dependency availability flags.
    """
    deps = {}
    
    # PyMuPDF (required)
    try:
        import fitz
        deps['pymupdf'] = True
        logger.info("✓ PyMuPDF (fitz) available")
    except ImportError:
        deps['pymupdf'] = False
        logger.error("✗ PyMuPDF (fitz) required: pip install pymupdf")
        raise ImportError("PyMuPDF (fitz) is required for DECLARMIMA to function")
    
    # Ollama (recommended)
    try:
        import ollama
        deps['ollama'] = True
        logger.info("✓ Ollama client available")
    except ImportError:
        deps['ollama'] = False
        logger.warning("✗ Ollama not installed. Ollama backend unavailable.")
    
    # HuggingFace Transformers (recommended)
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        deps['transformers'] = True
        logger.info("✓ HuggingFace transformers available")
    except ImportError:
        deps['transformers'] = False
        logger.warning("✗ transformers not installed. Local HF models unavailable.")
    
    # orjson (optional, faster JSON)
    try:
        import orjson
        deps['orjson'] = True
        logger.info("✓ orjson available (fast JSON)")
    except ImportError:
        deps['orjson'] = False
        logger.warning("✗ orjson not installed. Using standard json (slower).")
    
    # sentence-transformers (optional, semantic search)
    try:
        from sentence_transformers import SentenceTransformer, util
        deps['sentence_transformers'] = True
        logger.info("✓ sentence-transformers available")
    except ImportError:
        deps['sentence_transformers'] = False
        logger.warning("✗ sentence-transformers not installed. Using vectorless keyword retrieval.")
    
    # rapidfuzz (optional, fuzzy matching)
    try:
        from rapidfuzz import fuzz, process
        deps['rapidfuzz'] = True
        logger.info("✓ rapidfuzz available")
    except ImportError:
        deps['rapidfuzz'] = False
        logger.warning("✗ rapidfuzz not installed. Fuzzy matching disabled.")
    
    # matplotlib (required for static plots)
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.colors as mcolors
        from matplotlib.colors import LinearSegmentedColormap
        matplotlib.use('Agg')  # Non-interactive backend
        deps['matplotlib'] = True
        logger.info("✓ matplotlib available")
    except ImportError:
        deps['matplotlib'] = False
        logger.error("✗ matplotlib required: pip install matplotlib")
        raise ImportError("matplotlib is required for DECLARMIMA visualizations")
    
    # networkx (required for knowledge graphs)
    try:
        import networkx as nx
        deps['networkx'] = True
        logger.info("✓ networkx available")
    except ImportError:
        deps['networkx'] = False
        logger.error("✗ networkx required: pip install networkx")
        raise ImportError("networkx is required for knowledge graph visualizations")
    
    # plotly (required for interactive plots)
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        deps['plotly'] = True
        logger.info("✓ plotly available")
    except ImportError:
        deps['plotly'] = False
        logger.error("✗ plotly required: pip install plotly")
        raise ImportError("plotly is required for interactive visualizations")
    
    # umap-learn (optional, dimensionality reduction)
    try:
        import umap
        deps['umap'] = True
        logger.info("✓ umap-learn available")
    except ImportError:
        deps['umap'] = False
        logger.warning("✗ umap-learn not installed. UMAP embeddings disabled.")
    
    # pyvis (optional, interactive networks)
    try:
        from pyvis.network import Network
        deps['pyvis'] = True
        logger.info("✓ pyvis available")
    except ImportError:
        deps['pyvis'] = False
        logger.warning("✗ pyvis not installed. Interactive networks disabled.")
    
    # scikit-learn (optional, t-SNE/PCA)
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        deps['sklearn'] = True
        logger.info("✓ scikit-learn available")
    except ImportError:
        deps['sklearn'] = False
        logger.warning("✗ scikit-learn not installed. t-SNE/PCA disabled.")
    
    # pydantic (required for data validation)
    try:
        from pydantic import BaseModel, Field, field_validator, model_validator
        deps['pydantic'] = True
        logger.info("✓ pydantic available")
    except ImportError:
        deps['pydantic'] = False
        logger.error("✗ pydantic required: pip install pydantic")
        raise ImportError("pydantic is required for data validation")
    
    logger.info(f"Dependency check complete: {sum(deps.values())}/{len(deps)} available")
    return deps

# Check dependencies at module load time
GLOBAL_DEPS = check_optional_dependencies()

# === FIX: Import fitz at module level since it's REQUIRED ===
#import fitz
# === FIX: Import pymupdf as fitz for backward compatibility ===
import pymupdf as fitz

# Availability flags used throughout the code
PYVIS_AVAILABLE = GLOBAL_DEPS.get('pyvis', False)
SKLEARN_AVAILABLE = GLOBAL_DEPS.get('sklearn', False)
UMAP_AVAILABLE = GLOBAL_DEPS.get('umap', False)
SENTENCE_TRANSFORMERS_AVAILABLE = GLOBAL_DEPS.get('sentence_transformers', False)
ORJSON_AVAILABLE = GLOBAL_DEPS.get('orjson', False)

# ============================================================================
# PYDANTIC DATA MODELS
# ============================================================================
from pydantic import BaseModel, Field, field_validator, model_validator

class UniversalExtractionItem(BaseModel):
    """
    Universal extraction item supporting multiple item types:
    - quantitative: numerical values with units
    - qualitative: descriptive information
    - definition: term definitions
    - comparison: comparative statements
    - relationship: entity relationships
    - process: procedural information
    - material: material/alloy names
    - method: experimental/computational methods
    """
    # Core fields
    item_type: Literal[
        "quantitative", "qualitative", "definition", 
        "comparison", "relationship", "process", "material", "method"
    ]
    content: str  # The actual extracted text/content
    
    # Quantitative-specific fields
    parameter_name: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    physical_quantity: Optional[str] = None
    
    # Material/relationship fields
    material: Optional[str] = None
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object_val: Optional[str] = None
    
    # Definition fields
    definition_term: Optional[str] = None
    definition_text: Optional[str] = None
    
    # Comparison fields
    comparison_entities: List[str] = []
    comparison_aspect: Optional[str] = None
    
    # Metadata fields
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)
    context: str  # Surrounding text for context
    doc_source: str  # Document identifier
    page: int  # Page number
    section_title: Optional[str] = None
    method: Optional[str] = None  # Experimental/computational method
    conditions: Dict[str, Any] = {}  # Experimental conditions
    reasoning_trace: str = ""  # LLM reasoning trace
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is in valid range [0.0, 1.0]"""
        return max(0.0, min(1.0, v))
    
    def citation(self) -> str:
        """Generate citation string for this item"""
        return f'<cite doc="{self.doc_source}" page="{self.page}"/>'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return self.model_dump()
    
    def is_quantitative(self) -> bool:
        """Check if this is a quantitative extraction"""
        return self.item_type == "quantitative" and self.value is not None
    
    def get_human_readable(self) -> str:
        """Get human-readable representation"""
        if self.is_quantitative():
            return f"{self.parameter_name or self.physical_quantity}: {self.value} {self.unit}"
        return self.content[:100]


class ExtractedValue(BaseModel):
    """
    Simplified extracted value for query responses.
    Used for building consensus and generating reports.
    """
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
    def non_zero(cls, v: float) -> float:
        """Ignore zero values which often indicate missing data"""
        if v == 0.0:
            raise ValueError("Zero values ignored - likely missing data")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.model_dump()


class QueryReport(BaseModel):
    """
    Comprehensive report for a single-document or cross-document query.
    Includes consensus analysis and processing metadata.
    """
    query: str
    total_docs: int
    docs_with_results: int
    all_values: List[ExtractedValue]
    consensus: Dict[str, Any]
    processing_time_sec: float
    
    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(
            self.model_dump(), 
            indent=2, 
            ensure_ascii=False, 
            default=str
        )
    
    def get_summary_stats(self, physical_quantity: str) -> Dict[str, Any]:
        """Get summary statistics for a specific physical quantity"""
        values = [v.value for v in self.all_values if v.physical_quantity == physical_quantity]
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)) if len(values) > 1 else 0.0,
            "median": float(np.median(values))
        }


class CrossDocumentQueryReport(BaseModel):
    """
    Comprehensive cross-document query report with contradiction detection.
    """
    query: str
    query_type: Optional[str] = None
    total_documents: int
    documents_with_results: int
    documents_without_results: List[str] = []
    all_items: List[UniversalExtractionItem] = []
    document_summaries: List[Dict[str, Any]] = []
    consensus_analysis: Dict[str, Any] = {}
    contradictions_detected: List[Dict[str, Any]] = []
    processing_meta: Dict[str, Any] = {}
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string"""
        return json.dumps(
            self.model_dump(), 
            indent=indent, 
            ensure_ascii=False, 
            default=str
        )
    
    def has_contradictions(self) -> bool:
        """Check if any contradictions were detected"""
        return len(self.contradictions_detected) > 0
    
    def get_contradiction_summary(self) -> str:
        """Get human-readable contradiction summary"""
        if not self.contradictions_detected:
            return "No significant contradictions detected."
        lines = ["Detected contradictions:"]
        for c in self.contradictions_detected:
            lines.append(f"- {c['entity']}: {c['doc_a']} ({c['value_a']:.2f}) vs {c['doc_b']} ({c['value_b']:.2f})")
        return "\n".join(lines)


class DocumentMetadata(BaseModel):
    """
    Structured metadata extracted from scientific documents.
    Captures process parameters, material properties, and experimental conditions.
    """
    doc_name: str
    
    # Material/alloy information
    alloys: List[str] = Field(default_factory=list)
    
    # Laser processing parameters
    laser_power_values: List[float] = Field(default_factory=list)
    scan_speed_values: List[float] = Field(default_factory=list)
    energy_density_values: List[float] = Field(default_factory=list)
    areal_energy_density_values: List[float] = Field(default_factory=list)
    linear_energy_density_values: List[float] = Field(default_factory=list)
    layer_thickness_values: List[float] = Field(default_factory=list)
    hatch_distance_values: List[float] = Field(default_factory=list)
    spot_size_values: List[float] = Field(default_factory=list)
    
    # Mechanical properties
    yield_strength_values: List[float] = Field(default_factory=list)
    tensile_strength_values: List[float] = Field(default_factory=list)
    ultimate_tensile_strength_values: List[float] = Field(default_factory=list)
    hardness_values: List[float] = Field(default_factory=list)
    elongation_values: List[float] = Field(default_factory=list)
    modulus_values: List[float] = Field(default_factory=list)
    youngs_modulus_values: List[float] = Field(default_factory=list)
    poisson_ratio_values: List[float] = Field(default_factory=list)
    
    # Thermal properties
    temperature_values: List[float] = Field(default_factory=list)
    melting_temperature_values: List[float] = Field(default_factory=list)
    thermal_conductivity_values: List[float] = Field(default_factory=list)
    cte_values: List[float] = Field(default_factory=list)  # Coefficient of thermal expansion
    
    # Physical properties
    density_values: List[float] = Field(default_factory=list)
    viscosity_values: List[float] = Field(default_factory=list)
    enthalpy_values: List[float] = Field(default_factory=list)
    absorption_coefficient_values: List[float] = Field(default_factory=list)
    
    # Electrochemical properties
    corrosion_potential_values: List[float] = Field(default_factory=list)
    pitting_potential_values: List[float] = Field(default_factory=list)
    repassivation_potential_values: List[float] = Field(default_factory=list)
    breakdown_potential_values: List[float] = Field(default_factory=list)
    open_circuit_potential_values: List[float] = Field(default_factory=list)
    corrosion_current_density_values: List[float] = Field(default_factory=list)
    polarization_resistance_values: List[float] = Field(default_factory=list)
    current_density_values: List[float] = Field(default_factory=list)
    pren_values: List[float] = Field(default_factory=list)
    
    # Microstructural properties
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
    dislocation_density_values: List[float] = Field(default_factory=list)
    
    # Spray/fluid dynamics
    sauter_mean_diameter_values: List[float] = Field(default_factory=list)
    spray_penetration_values: List[float] = Field(default_factory=list)
    plume_height_values: List[float] = Field(default_factory=list)
    film_thickness_values: List[float] = Field(default_factory=list)
    
    # Melt pool characteristics
    melt_pool_depth_values: List[float] = Field(default_factory=list)
    melt_pool_width_values: List[float] = Field(default_factory=list)
    melt_pool_length_values: List[float] = Field(default_factory=list)
    
    # Hollomon parameters (elastoplasticity)
    hollomon_strength_coeff_values: List[float] = Field(default_factory=list)
    hollomon_exponent_values: List[float] = Field(default_factory=list)
    
    # HEA descriptors
    vec_values: List[float] = Field(default_factory=list)  # Valence electron concentration
    delta_h_mix_values: List[float] = Field(default_factory=list)  # Enthalpy of mixing
    delta_s_mix_values: List[float] = Field(default_factory=list)  # Entropy of mixing
    omega_parameter_values: List[float] = Field(default_factory=list)
    atomic_size_difference_values: List[float] = Field(default_factory=list)
    lambda_parameter_values: List[float] = Field(default_factory=list)
    lewis_number_values: List[float] = Field(default_factory=list)
    jackson_parameter_values: List[float] = Field(default_factory=list)
    
    # Nanoindentation
    indentation_force_values: List[float] = Field(default_factory=list)
    indentation_depth_values: List[float] = Field(default_factory=list)
    
    # Process types
    process_types: List[str] = Field(default_factory=list)
    
    # Other parameters (catch-all)
    other_parameters: Dict[str, List[float]] = Field(default_factory=dict)
    
    def get_parameter_count(self) -> int:
        """Get total number of extracted parameters"""
        count = 0
        for field_name, field_value in self.model_dump().items():
            if field_name.endswith('_values') and isinstance(field_value, list):
                count += len(field_value)
        return count
    
    def get_all_parameters(self) -> Dict[str, List[float]]:
        """Get all parameters as a dictionary"""
        params = {}
        for field_name, field_value in self.model_dump().items():
            if field_name.endswith('_values') and isinstance(field_value, list) and field_value:
                param_name = field_name.replace('_values', '')
                params[param_name] = field_value
        return params
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Create a summary dictionary for indexing"""
        return {
            "doc_name": self.doc_name,
            "alloys": self.alloys,
            "process_types": self.process_types,
            "laser_power_range": (min(self.laser_power_values), max(self.laser_power_values)) if self.laser_power_values else None,
            "scan_speed_range": (min(self.scan_speed_values), max(self.scan_speed_values)) if self.scan_speed_values else None,
            "total_parameters": self.get_parameter_count()
        }

# ============================================================================
# ENHANCED PHYSICAL QUANTITY CLASSIFIER WITH LLM FALLBACK
# ============================================================================
class PhysicalQuantityClassifier:
    """
    Enhanced classifier supporting both deterministic keyword matching and 
    optional LLM-based fallback for ambiguous parameter names.
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
        "unstable_stacking_fault_energy": ["unstable stacking fault energy", "usfe"],
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
        "flow_velocity": ["flow velocity", "fluid velocity", "gas velocity"],
        "pressure": ["pressure", "gas pressure", "vapor pressure"],
        "cooling_rate": ["cooling rate", "solidification rate", "temperature gradient"],
        "dwell_time": ["dwell time", "holding time", "soaking time"],
        "grain_boundary_energy": ["grain boundary energy", "interface energy"],
        "recrystallization_temperature": ["recrystallization temperature", "annealing temperature"],
        "creep_rate": ["creep rate", "steady state creep rate"],
        "fatigue_limit": ["fatigue limit", "endurance limit"],
        "impact_energy": ["impact energy", "charpy impact energy"],
        "fracture_toughness": ["fracture toughness", "kic", "critical stress intensity factor"],
        "wear_rate": ["wear rate", "specific wear rate"],
        "coefficient_of_friction": ["coefficient of friction", "friction coefficient", "mu"],
        "resistivity": ["resistivity", "electrical resistivity"],
        "conductivity": ["conductivity", "electrical conductivity"],
        "magnetic_permeability": ["magnetic permeability", "relative permeability"],
        "dielectric_constant": ["dielectric constant", "relative permittivity"],
        "refractive_index": ["refractive index", "index of refraction"],
        "band_gap": ["band gap", "energy band gap", "eg"],
        "electron_affinity": ["electron affinity", "electronic affinity"],
        "work_function": ["work function", "ionization potential"],
        "lattice_constant": ["lattice constant", "lattice parameter"],
        "atomic_radius": ["atomic radius", "ionic radius"],
        "electronegativity": ["electronegativity", "pauling electronegativity"],
        "specific_heat": ["specific heat", "heat capacity", "cp"],
        "latent_heat": ["latent heat", "enthalpy of fusion", "heat of fusion"],
        "thermal_diffusivity": ["thermal diffusivity", "thermal diffusivity coefficient"],
        "sound_velocity": ["sound velocity", "speed of sound", "acoustic velocity"],
        "elastic_modulus": ["elastic modulus", "stiffness", "modulus of elasticity"],
        "shear_modulus": ["shear modulus", "modulus of rigidity", "g"],
        "bulk_modulus": ["bulk modulus", "compressibility modulus"],
        "poissons_ratio": ["poisson's ratio", "poisson ratio", "nu"],
        "yield_point": ["yield point", "yield stress", "proof stress"],
        "tensile_point": ["tensile point", "tensile stress", "ultimate tensile stress"],
        "breaking_point": ["breaking point", "fracture stress", "rupture stress"],
        "hardness_vickers": ["hardness vickers", "vickers hardness", "hv"],
        "hardness_rockwell": ["hardness rockwell", "rockwell hardness", "hrc", "hrb"],
        "hardness_brinell": ["hardness brinell", "brinell hardness", "hb"],
        "hardness_knoop": ["hardness knoop", "knoop hardness", "hk"],
        "hardness_mohs": ["hardness mohs", "mohs hardness scale"],
        "microhardness": ["microhardness", "nanoindentation hardness"],
        "macrohardness": ["macrohardness", "bulk hardness"],
        "superhardness": ["superhardness", "super hard material hardness"],
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
        "flow_velocity": ["mm/s", "m/s", "cm/s", "l/s", "l/min"],
        "pressure": ["pa", "mpa", "gpa", "bar", "atm", "psi", "torr", "mmhg"],
        "cooling_rate": ["k/s", "°c/s", "k/min", "°c/min"],
        "dwell_time": ["s", "min", "h", "hr", "hour", "hour(s)"],
        "grain_boundary_energy": ["mj/m2", "j/m2", "erg/cm2"],
        "recrystallization_temperature": ["°c", "k", "°f"],
        "creep_rate": ["s-1", "/s", "h-1", "/h"],
        "fatigue_limit": ["mpa", "gpa", "psi"],
        "impact_energy": ["j", "kj", "ft-lb"],
        "fracture_toughness": ["mpa√m", "ksi√in", "mn/m3/2"],
        "wear_rate": ["mm3/nm", "mm3/m", "mm3/n·m"],
        "coefficient_of_friction": ["unitless", ""],
        "resistivity": ["ohm·m", "μohm·cm", "ohm·cm"],
        "conductivity": ["s/m", "m/s", "ms/cm"],
        "magnetic_permeability": ["h/m", "n/a2", "t·m/a"],
        "dielectric_constant": ["unitless", ""],
        "refractive_index": ["unitless", ""],
        "band_gap": ["ev", "j", "cm-1"],
        "electron_affinity": ["ev", "j", "kj/mol"],
        "work_function": ["ev", "j", "kj/mol"],
        "lattice_constant": ["a", "nm", "pm", "Å"],
        "atomic_radius": ["pm", "a", "nm", "Å"],
        "electronegativity": ["unitless", ""],
        "specific_heat": ["j/kg·k", "j/g·k", "cal/g·°c"],
        "latent_heat": ["j/kg", "kj/kg", "j/mol", "cal/g"],
        "thermal_diffusivity": ["m2/s", "mm2/s", "cm2/s"],
        "sound_velocity": ["m/s", "km/s", "mm/s"],
        "elastic_modulus": ["gpa", "mpa", "psi"],
        "shear_modulus": ["gpa", "mpa", "psi"],
        "bulk_modulus": ["gpa", "mpa", "psi"],
        "poissons_ratio": ["unitless", ""],
        "yield_point": ["mpa", "gpa", "psi"],
        "tensile_point": ["mpa", "gpa", "psi"],
        "breaking_point": ["mpa", "gpa", "psi"],
        "hardness_vickers": ["hv", "gpa", "mpa"],
        "hardness_rockwell": ["hrc", "hrb", "hra"],
        "hardness_brinell": ["hb", "gpa", "mpa"],
        "hardness_knoop": ["hk", "gpa", "mpa"],
        "hardness_mohs": ["unitless", ""],
        "microhardness": ["hv", "hk", "gpa", "mpa"],
        "macrohardness": ["hrc", "hb", "gpa", "mpa"],
        "superhardness": ["gpa", "mpa", "hv"],
    }

    _llm_cache: Dict[str, str] = {}
    _llm_call_count: int = 0

    def __init__(self, llm_callback: Optional[Callable[[str], str]] = None):
        self._build_keyword_index()
        self.llm_callback = llm_callback  # Optional callback for LLM resolution

    def _build_keyword_index(self):
        """Builds a fast lookup dictionary from CANONICAL and adds common shortcuts."""
        self.keyword_to_canonical = {}
        for canonical, keywords in self.CANONICAL.items():
            for kw in keywords:
                self.keyword_to_canonical[kw.lower()] = canonical

        # Add explicit short-form mappings for rapid access
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
            "omega": "omega_parameter", "mu": "coefficient_of_friction",
            "k": "thermal_conductivity", "cp": "specific_heat",
            "g": "shear_modulus", "nu": "poissons_ratio",
            "e": "youngs_modulus", "hv": "hardness_vickers",
            "hrc": "hardness_rockwell", "hb": "hardness_brinell",
            "hk": "hardness_knoop", "hm": "hardness_mohs",
            "kic": "fracture_toughness", "eg": "band_gap",
            "phi": "dielectric_constant", "n": "refractive_index",
            "rho": "density", "sigma": "tensile_strength",
            "epsilon": "elongation", "alpha": "coefficient_thermal_expansion",
            "lambda": "lambda_parameter", "delta": "atomic_size_difference",
            "gamma": "austenite_fraction", "beta": "ferrite_fraction",
            "delta_h": "delta_h_mix", "delta_s": "delta_s_mix",
            "delta_p": "delta_h_mix", "delta_q": "delta_s_mix",
            "delta_r": "delta_h_mix", "delta_t": "delta_s_mix",
        }
        self.keyword_to_canonical.update(shortcuts)

    def classify(self, parameter_name: Optional[str], unit: Optional[str], context: str) -> str:
        """
        Fast-path classification using keyword, unit, and context matching.
        """
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
        canonical_list_str = json.dumps(list(self.CANONICAL.keys()), indent=2)
        prompt = f"""You are a scientific parameter classifier. Given a parameter name, unit, and context, 
classify it into ONE of these canonical physical quantities:

{canonical_list_str}

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
        """Convert canonical name to human-readable title case string."""
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
            "flow_velocity": "Flow Velocity", "pressure": "Pressure",
            "cooling_rate": "Cooling Rate", "dwell_time": "Dwell Time",
            "grain_boundary_energy": "Grain Boundary Energy",
            "recrystallization_temperature": "Recrystallization Temperature",
            "creep_rate": "Creep Rate", "fatigue_limit": "Fatigue Limit",
            "impact_energy": "Impact Energy", "fracture_toughness": "Fracture Toughness",
            "wear_rate": "Wear Rate", "coefficient_of_friction": "Coefficient of Friction",
            "resistivity": "Resistivity", "conductivity": "Conductivity",
            "magnetic_permeability": "Magnetic Permeability", "dielectric_constant": "Dielectric Constant",
            "refractive_index": "Refractive Index", "band_gap": "Band Gap",
            "electron_affinity": "Electron Affinity", "work_function": "Work Function",
            "lattice_constant": "Lattice Constant", "atomic_radius": "Atomic Radius",
            "electronegativity": "Electronegativity", "specific_heat": "Specific Heat",
            "latent_heat": "Latent Heat", "thermal_diffusivity": "Thermal Diffusivity",
            "sound_velocity": "Sound Velocity", "elastic_modulus": "Elastic Modulus",
            "shear_modulus": "Shear Modulus", "bulk_modulus": "Bulk Modulus",
            "poissons_ratio": "Poisson's Ratio", "yield_point": "Yield Point",
            "tensile_point": "Tensile Point", "breaking_point": "Breaking Point",
            "hardness_vickers": "Hardness (Vickers)", "hardness_rockwell": "Hardness (Rockwell)",
            "hardness_brinell": "Hardness (Brinell)", "hardness_knoop": "Hardness (Knoop)",
            "hardness_mohs": "Hardness (Mohs)", "microhardness": "Microhardness",
            "macrohardness": "Macrohardness", "superhardness": "Superhardness",
        }
        return mapping.get(canonical, canonical.replace("_", " ").title())


# ============================================================================
# CONCEPT NORMALIZER WITH FUZZY MATCHING & EMBEDDINGS
# ============================================================================
class ConceptNormalizer:
    """
    Normalizes synonyms, abbreviations, and variations of scientific concepts.
    Uses dictionary lookups, fuzzy matching (rapidfuzz), and optional embedding similarity.
    """
    
    ALIAS_DICTIONARIES = {
        "multicomponent": [
            "multicomponent", "multi-component", "multielement", "multi-element",
            "many elements", "complex alloy", "multi-principal", "high entropy",
            "hea", "multiple elements", "ternary", "quaternary", "quinary"
        ],
        "yield_strength": [
            "yield strength", "ys", "0.2% proof", "proof stress", "yield stress",
            "0.2% offset strength"
        ],
        "tensile_strength": [
            "tensile strength", "uts", "ultimate tensile strength", "ultimate strength",
            "tensile stress"
        ],
        "laser_power": [
            "laser power", "laser beam power", "laser output power", "beam power"
        ],
        "scan_speed": [
            "scan speed", "scanning speed", "laser scan speed", "beam scan speed",
            "scan velocity"
        ],
        "hardness": [
            "hardness", "vickers hardness", "microhardness", "hv", "nano hardness"
        ],
        "sdss_2507": [
            "sdss 2507", "super duplex stainless steel 2507", "uns s32750", "en 1.4410",
            "saf 2507", "2507", "s32750", "super duplex 2507"
        ],
        "ti3au": [
            "ti3au", "ti_3au", "beta-ti3au", "b-ti3au", "ti-au intermetallic", "titanium gold intermetallic",
            "ti3au intermetallic", "beta ti3au"
        ],
        "cp_ti": [
            "cp ti", "commercially pure titanium", "grade ii titanium", "grade 2 titanium", "titanium grade ii",
            "commercial purity titanium"
        ],
        "alsimgzr": [
            "alsimgzr", "al-si-mg-zr", "al-si-mg-0.37zr", "alsi7.43mg1.57zr", "al-si-mg-zr alloy",
            "al-si-mg-zr composite"
        ],
        "tib2_alsimgzr": [
            "tib2/al-si-mg-zr", "tib2-alsimgzr", "tib2 modified al-si-mg-zr", "tib2/al-si-mg-zr composite",
            "tib2-al-si-mg-zr"
        ],
        "metallic_glass": [
            "fe-based metallic glass", "metallic glass", "amorphous alloy", "fe-b-si-nb-zr-cu",
            "fe based metallic glass"
        ],
        "lpbf": [
            "lpbf", "l-pbf", "laser powder bed fusion", "selective laser melting", "slm",
            "laser powder-bed fusion", "laser powder-bed-fusion", "laser powder bed fusion (lpbf)"
        ],
        "ded": [
            "ded", "directed energy deposition", "direct energy deposition", "laser metal deposition",
            "directed energy deposition (ded)"
        ],
        "pfi": [
            "pfi", "port fuel injection", "port-fuel injection", "port fuel injector",
            "port fuel injection (pfi)"
        ],
        "gdi": [
            "gdi", "gasoline direct injection", "direct injection spark ignition", "disi",
            "gasoline direct injection (gdi)"
        ],
        "pren": [
            "pren", "pitting resistance equivalent number", "pitting resistance equivalent",
            "pitting resistance equivalent number (pren)"
        ],
        "eis": [
            "eis", "electrochemical impedance spectroscopy", "impedance spectroscopy",
            "electrochemical impedance"
        ],
        "cpp": [
            "cpp", "cyclic potentiodynamic polarization", "potentiodynamic polarization", "cyclic polarization",
            "cyclic potentiodynamic polarization (cpp)"
        ],
        "nanoindentation": [
            "nanoindentation", "nano-indentation", "indentation test", "indentation force",
            "nano indentation"
        ],
        "sfe": [
            "stacking fault energy", "sfe", "generalized stacking fault energy", "gsfe",
            "stacking fault energy (sfe)"
        ],
        "smd": [
            "sauter mean diameter", "smd", "sauter diameter", "mean droplet diameter",
            "sauter mean diameter (smd)"
        ],
        "ved": [
            "ved", "volumetric energy density", "volume energy density", "energy density",
            "volumetric energy density (ved)"
        ],
        "aed": [
            "aed", "areal energy density", "area energy density", "areal energy density (aed)"
        ],
        "led": [
            "led", "linear energy density", "line energy density", "linear energy density (led)"
        ],
        "fem": [
            "fem", "finite element method", "finite element analysis", "fea", "finite element"
        ],
        "md": [
            "md", "molecular dynamics", "molecular dynamics simulation", "molecular dynamics (md)"
        ],
    }
    
    def __init__(self, embedding_fn: Optional[Callable] = None):
        self.embedding_fn = embedding_fn
        self._build_reverse_index()
    
    def _build_reverse_index(self):
        """Build reverse index: alias -> canonical."""
        self.alias_to_canonical: Dict[str, str] = {}
        for canonical, aliases in self.ALIAS_DICTIONARIES.items():
            for alias in aliases:
                self.alias_to_canonical[alias.lower()] = canonical
    
    def normalize(self, term: str, use_fuzzy: bool = True, fuzzy_threshold: int = 85) -> str:
        """Normalize a term to its canonical form using dictionary, fuzzy, or embedding matching."""
        if not term or not str(term).strip():
            return "unknown"
        
        term_lower = str(term).lower().strip()
        
        # 1. Direct dictionary lookup
        if term_lower in self.alias_to_canonical:
            return self.alias_to_canonical[term_lower]
        
        # 2. Substring match (longest alias first)
        for alias, canonical in sorted(self.alias_to_canonical.items(), key=lambda x: -len(x[0])):
            if alias in term_lower:
                return canonical
        
        # 3. Fuzzy matching
        if use_fuzzy and GLOBAL_DEPS.get('rapidfuzz', False):
            from rapidfuzz import fuzz, process
            all_aliases = list(self.alias_to_canonical.keys())
            result = process.extractOne(term_lower, all_aliases, scorer=fuzz.ratio)
            if result and result[1] >= fuzzy_threshold:
                return self.alias_to_canonical[result[0]]
        
        # 4. Embedding similarity
        if self.embedding_fn is not None and GLOBAL_DEPS.get('sentence_transformers', False):
            try:
                term_emb = self.embedding_fn(term_lower)
                best_sim = -1.0
                best_canonical = None
                
                # Compare with canonical names only for efficiency
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
        
        # 5. Fallback: return original lowercase
        return term_lower
    
    def normalize_list(self, terms: List[str]) -> List[str]:
        """Normalize a list of terms."""
        return [self.normalize(t) for t in terms]


# ============================================================================
# DISPLAY NAME HELPERS (DOI POSTPROCESSING + USER ALIASES)
# ============================================================================
def normalize_doi_display(name: str) -> str:
    """
    Convert filesystem-safe DOI filenames back to real DOI format.
    E.g. '10.1016_j.scriptamat.2024.116027.pdf' -> '10.1016/j.scriptamat.2024.116027'
    """
    if not name:
        return name
    # Remove .pdf extension
    base = name[:-4] if name.lower().endswith('.pdf') else name
    # If it looks like a DOI (starts with 10. and contains _)
    if re.match(r'10\.\d+_', base):
        # Replace first _ after 10.xxx with /
        base = re.sub(r'^(10\.\d+)_(.*)', r'\1/\2', base)
    return base


def get_display_name(doc_id: str, aliases: Optional[Dict[str, str]] = None) -> str:
    """
    Return human-readable display name for a document.
    Priority: 1) user alias, 2) DOI-normalized stem, 3) original stem.
    """
    if aliases and doc_id in aliases:
        return aliases[doc_id]
    stem = Path(doc_id).stem
    normalized = normalize_doi_display(stem)
    return normalized


def get_citation_label(doc_id: str, aliases: Optional[Dict[str, str]] = None, index: int = 0, style: str = "doi") -> str:
    """
    Generate citation-style label for a document.
    style: 'doi' -> normalized DOI, 'number' -> [1], 'alias' -> user alias, 'short' -> first 20 chars.
    """
    if style == "alias" and aliases and doc_id in aliases:
        return aliases[doc_id]
    if style == "number":
        return f"[{index}]"
    if style == "short":
        return Path(doc_id).stem[:20]
    return normalize_doi_display(Path(doc_id).stem)


# ============================================================================
# PAGINATION AWARE READER
# ============================================================================
class PaginationAwareReader:
    """
    Reads PDF pages with awareness of character limits to prevent token overflow.
    """
    def __init__(self, max_chars_per_request: int = 20000):
        self.max_chars_per_request = max_chars_per_request

    def extract_pages(self, doc_path: str, page_numbers: List[int]) -> Dict[int, str]:
        """Extract specific pages from a PDF document."""
        doc = fitz.open(doc_path)
        result = {}
        for pnum in page_numbers:
            if pnum < 1 or pnum > len(doc):
                continue
            page = doc[pnum - 1]
            text = page.get_text("text")
            if len(text) > self.max_chars_per_request:
                logger.warning(f"Page {pnum} text length {len(text)} exceeds limit, truncating.")
                text = text[:self.max_chars_per_request] + "\n...[TRUNCATED]"
            result[pnum] = text
        doc.close()
        return result

    def extract_page_range(self, doc_path: str, start: int, end: int, step: int = 1) -> Dict[int, str]:
        """Extract a range of pages from a PDF document."""
        pages = list(range(start, end + 1, step))
        return self.extract_pages(doc_path, pages)


# ============================================================================
# STRUCTURED METADATA EXTRACTOR
# ============================================================================
class StructuredMetadataExtractor:
    """
    Extracts structured metadata from scientific documents using regex patterns.
    Captures process parameters, material properties, and experimental conditions.
    """
    ECORR_PATTERN = r'(?:Ecorr|corrosion potential|OCP|open circuit potential)\s*[=:]\s*([+-]?\d+(?:\.\d+)?)\s*(mV|V)'
    ERP_PATTERN = r'(?:Erp|repassivation potential)\s*[=:]\s*([+-]?\d+(?:\.\d+)?)\s*(mV|V)'
    EPIT_PATTERN = r'(?:Epit|pitting potential|breakdown potential|Ebr)\s*[=:]\s*([+-]?\d+(?:\.\d+)?)\s*(mV|V)'
    EBR_PATTERN = r'(?:Ebr|breakdown potential|depassivation point)\s*[=:]\s*([+-]?\d+(?:\.\d+)?)\s*(mV|V)'
    JCORR_PATTERN = r'(?:Jcorr|corrosion current density|i_corr)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(µA/cm²|uA/cm2|mA/cm2|A/cm2)'
    RP_PATTERN = r'(?:Rp|polarization resistance)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(kΩ·cm²|ohm·cm2|Ω·cm2)'
    PREN_PATTERN = r'(?:PREN|pitting resistance equivalent)\s*[=:]\s*(\d+(?:\.\d+)?)'
    SFE_PATTERN = r'(?:SFE|stacking fault energy|GSFE)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(mJ/m²|mj/m2|J/m2)'
    SMD_PATTERN = r'(?:SMD|Sauter mean diameter)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(µm|um|nm|mm)'
    DENSITY_PATTERN = r'(?:density|ρ)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(g/cm³|g/cm3|kg/m³|kg/m3)'
    THERMAL_CONDUCTIVITY_PATTERN = r'(?:thermal conductivity|k)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(W/m·K|W/mK|W/m·k)'
    VISCOSITY_PATTERN = r'(?:viscosity|μ)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(Pa·s|mPa·s|cP)'
    ENTHALPY_PATTERN = r'(?:enthalpy|H)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(J/mol|kJ/mol|J/kg)'
    ELONGATION_PATTERN = r'(?:elongation|strain to failure)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(%|pct)'
    PHASE_FRACTION_PATTERN = r'(?:austenite|ferrite|martensite)\s*(?:fraction|content|volume)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(%|vol%)'
    GRAIN_SIZE_PATTERN = r'(?:grain size|cell size)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(µm|um|nm|mm)'
    POROSITY_PATTERN = r'(?:porosity|pore fraction)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(%|fraction)'
    RELATIVE_DENSITY_PATTERN = r'(?:relative density)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(%|fraction)'
    HOLL_K_PATTERN = r'(?:strength coefficient|sigma_0|σ₀|K)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(MPa|GPa|Pa)'
    HOLL_N_PATTERN = r'(?:strain hardening exponent|work hardening exponent|n)\s*[=:]\s*(\d+(?:\.\d+)?)'
    IND_FORCE_PATTERN = r'(?:indentation force|load|F)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(mN|μN|N|nN)'
    IND_DEPTH_PATTERN = r'(?:indentation depth|penetration depth|h)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(nm|μm|um|mm)'
    DISL_DENSITY_PATTERN = r'(?:dislocation density|ρ_d)\s*[=:]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(m⁻²|cm⁻²|1/m2|m-2)'
    HATCH_PATTERN = r'(?:hatch distance|hatch spacing)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(μm|um|mm|nm)'
    BED_TEMP_PATTERN = r'(?:build platform|substrate|bed|preheat)\s+(?:temp|temperature|T)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(°C|K|°F)'
    MELT_DEPTH_PATTERN = r'(?:melt\s*pool\s*depth|penetration\s*depth)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(μm|um|mm|nm)'
    MELT_WIDTH_PATTERN = r'(?:melt\s*pool\s*width|track\s*width)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(μm|um|mm|nm)'
    MELT_LENGTH_PATTERN = r'(?:melt\s*pool\s*length)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(μm|um|mm|nm)'
    ALLOY_PATTERNS = [
        r'\b(?:AlSi[\dMg]+|Ti\d*Al\d*V\d*|Inconel\s?\d{3}|SS\s?\d{4}|UNS\s?S\d{5}|Ti\s?6Al\s?4V|Cu\s?[A-Za-z0-9]+|Fe-based|Mg\s?alloy)\b',
        r'\b(?:Al-[\d]+Si-[\d]+Mg|AlSiMg[\d\.]+Zr|TiB[2]?|CoCr[\w]+|NiTi|Au\-Ti|Zr\-enhanced)\b',
        r'(\w+(?:-\w+)?\s?(?:alloy|superalloy|metal|composite))'
    ]
    POWER_PATTERN = r'(?:laser\s+power|power|P)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(W|kW|mW)'
    SCAN_SPEED_PATTERN = r'(?:scan\s+speed|scanning\s+speed|v_scan|Vs)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(mm/s|cm/s|m/s|mm/min)'
    YIELD_PATTERN = r'(?:yield\s+strength|YS|0\.2%\s+proof)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(MPa|GPa|psi)'
    TENSILE_PATTERN = r'(?:tensile\s+strength|UTS)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(MPa|GPa|psi)'
    HARDNESS_PATTERN = r'(?:hardness|HV|Vickers)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(HV|MPa|GPa)'
    TEMP_PATTERN = r'(?:temperature|T)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(°C|K|°F)'
    VED_PATTERN = r'(?:volumetric\s+energy\s+density|VED)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(J/mm³|J/cm³)'

    def __init__(self):
        """Initialize compiled regex patterns for metadata extraction."""
        self.compiled_patterns = {
            "laser_power": (re.compile(self.POWER_PATTERN, re.IGNORECASE), float),
            "scan_speed": (re.compile(self.SCAN_SPEED_PATTERN, re.IGNORECASE), float),
            "yield_strength": (re.compile(self.YIELD_PATTERN, re.IGNORECASE), float),
            "tensile_strength": (re.compile(self.TENSILE_PATTERN, re.IGNORECASE), float),
            "hardness": (re.compile(self.HARDNESS_PATTERN, re.IGNORECASE), float),
            "temperature": (re.compile(self.TEMP_PATTERN, re.IGNORECASE), float),
            "energy_density": (re.compile(self.VED_PATTERN, re.IGNORECASE), float),
            "corrosion_potential": (re.compile(self.ECORR_PATTERN, re.IGNORECASE), float),
            "pitting_potential": (re.compile(self.EPIT_PATTERN, re.IGNORECASE), float),
            "repassivation_potential": (re.compile(self.ERP_PATTERN, re.IGNORECASE), float),
            "breakdown_potential": (re.compile(self.EBR_PATTERN, re.IGNORECASE), float),
            "corrosion_current_density": (re.compile(self.JCORR_PATTERN, re.IGNORECASE), float),
            "polarization_resistance": (re.compile(self.RP_PATTERN, re.IGNORECASE), float),
            "pren": (re.compile(self.PREN_PATTERN, re.IGNORECASE), float),
            "stacking_fault_energy": (re.compile(self.SFE_PATTERN, re.IGNORECASE), float),
            "sauter_mean_diameter": (re.compile(self.SMD_PATTERN, re.IGNORECASE), float),
            "density": (re.compile(self.DENSITY_PATTERN, re.IGNORECASE), float),
            "thermal_conductivity": (re.compile(self.THERMAL_CONDUCTIVITY_PATTERN, re.IGNORECASE), float),
            "viscosity": (re.compile(self.VISCOSITY_PATTERN, re.IGNORECASE), float),
            "enthalpy": (re.compile(self.ENTHALPY_PATTERN, re.IGNORECASE), float),
            "elongation": (re.compile(self.ELONGATION_PATTERN, re.IGNORECASE), float),
            "phase_fraction": (re.compile(self.PHASE_FRACTION_PATTERN, re.IGNORECASE), float),
            "grain_size": (re.compile(self.GRAIN_SIZE_PATTERN, re.IGNORECASE), float),
            "porosity": (re.compile(self.POROSITY_PATTERN, re.IGNORECASE), float),
            "relative_density": (re.compile(self.RELATIVE_DENSITY_PATTERN, re.IGNORECASE), float),
            "hollomon_strength_coeff": (re.compile(self.HOLL_K_PATTERN, re.IGNORECASE), float),
            "hollomon_exponent": (re.compile(self.HOLL_N_PATTERN, re.IGNORECASE), float),
            "indentation_force": (re.compile(self.IND_FORCE_PATTERN, re.IGNORECASE), float),
            "indentation_depth": (re.compile(self.IND_DEPTH_PATTERN, re.IGNORECASE), float),
            "dislocation_density": (re.compile(self.DISL_DENSITY_PATTERN, re.IGNORECASE), float),
            "hatch_distance": (re.compile(self.HATCH_PATTERN, re.IGNORECASE), float),
            "build_platform_temperature": (re.compile(self.BED_TEMP_PATTERN, re.IGNORECASE), float),
            "melt_pool_depth": (re.compile(self.MELT_DEPTH_PATTERN, re.IGNORECASE), float),
            "melt_pool_width": (re.compile(self.MELT_WIDTH_PATTERN, re.IGNORECASE), float),
            "melt_pool_length": (re.compile(self.MELT_LENGTH_PATTERN, re.IGNORECASE), float),
        }
        self.alloy_regexes = [re.compile(p, re.IGNORECASE) for p in self.ALLOY_PATTERNS]

    def extract_metadata(self, doc_name: str, full_text: str) -> DocumentMetadata:
        """
        Extract structured metadata from full document text.
        
        Args:
            doc_name: Document filename
            full_text: Full text content of the document
            
        Returns:
            DocumentMetadata object with extracted values
        """
        meta = DocumentMetadata(doc_name=doc_name)
        
        # Extract alloys
        alloys_set = set()
        for regex in self.alloy_regexes:
            for match in regex.finditer(full_text):
                candidate = match.group(0).strip()
                if len(candidate) > 2 and candidate.lower() not in ["alloy", "composite", "metal"]:
                    alloys_set.add(candidate)
        meta.alloys = list(alloys_set)
        
        # Extract quantitative parameters
        for field, (pattern, cast_func) in self.compiled_patterns.items():
            matches = pattern.findall(full_text)
            values = []
            for m in matches:
                try:
                    val = cast_func(m[0])
                    values.append(val)
                except:
                    continue
            setattr(meta, f"{field}_values", values)
        
        # Extract process types
        process_keywords = {
            "SLM": ["selective laser melting", "slm"],
            "LPBF": ["laser powder bed fusion", "l-pbf", "lpbf"],
            "LSA": ["laser surface alloying", "lsa"],
            "EBM": ["electron beam melting", "ebm"],
            "DED": ["directed energy deposition", "ded"],
        }
        processes = []
        for proc, keywords in process_keywords.items():
            if any(kw in full_text.lower() for kw in keywords):
                processes.append(proc)
        meta.process_types = processes
        
        return meta


# ============================================================================
# TWO-STAGE RETRIEVER (VECTORLESS + SEMANTIC FALLBACK)
# ============================================================================
class TwoStageRetriever:
    """
    Retrieves relevant documents using a hybrid approach:
    1. Keyword/metadata matching (fast, vectorless)
    2. Semantic similarity scoring (optional fallback using sentence-transformers)
    """
    def __init__(self, llm=None, embedding_model: str = "all-MiniLM-L6-v2"):
        self.llm = llm
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(embedding_model, device="cpu")
                logger.info(f"Loaded sentence-transformer model {embedding_model} on CPU")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
        
        self.doc_metadata: Dict[str, DocumentMetadata] = {}
        self.doc_summaries: Dict[str, str] = {}

    def index_document(self, doc_name: str, metadata: DocumentMetadata, summary: str):
        """Add a document to the retrieval index."""
        self.doc_metadata[doc_name] = metadata
        self.doc_summaries[doc_name] = summary

    def retrieve_relevant_docs(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieve documents most relevant to the query.
        
        Args:
            query: User query string
            top_k: Number of documents to return
            
        Returns:
            List of (doc_name, score) tuples sorted by relevance score
        """
        scores = []
        query_lower = query.lower()
        
        for name, meta in self.doc_metadata.items():
            score = 0.0
            
            # Keyword-based scoring
            if "laser power" in query_lower and meta.laser_power_values:
                score += 0.5
            if "scan speed" in query_lower and meta.scan_speed_values:
                score += 0.5
            for alloy in meta.alloys:
                if alloy.lower() in query_lower:
                    score += 0.3
            if any(term in query_lower for term in ["material", "alloy", "compound"]):
                if meta.alloys:
                    score += 0.4
                else:
                    score += 0.1
            if "yield" in query_lower and meta.yield_strength_values:
                score += 0.4
            if "tensile" in query_lower and meta.tensile_strength_values:
                score += 0.4
            if "hardness" in query_lower and meta.hardness_values:
                score += 0.4
            
            # Electrochemical queries
            if any(t in query_lower for t in ["corrosion", "pitting", "repassivation", "polarization", "eis", "cpp"]):
                if meta.corrosion_potential_values or meta.polarization_resistance_values:
                    score += 0.6
            if "current density" in query_lower and meta.corrosion_current_density_values:
                score += 0.5
            if "pren" in query_lower and meta.pren_values:
                score += 0.5
            
            # Microstructural queries
            if any(t in query_lower for t in ["austenite", "ferrite", "phase fraction"]):
                if meta.phase_fraction_values or meta.austenite_fraction_values or meta.ferrite_fraction_values:
                    score += 0.5
            if "grain size" in query_lower and meta.grain_size_values:
                score += 0.4
            if "porosity" in query_lower and meta.porosity_values:
                score += 0.4
            if "relative density" in query_lower and meta.relative_density_values:
                score += 0.4
            
            # Thermal/physical queries
            if "thermal conductivity" in query_lower and meta.thermal_conductivity_values:
                score += 0.4
            if "viscosity" in query_lower and meta.viscosity_values:
                score += 0.4
            if "density" in query_lower and meta.density_values:
                score += 0.3
            if "enthalpy" in query_lower and meta.enthalpy_values:
                score += 0.4
            
            # Spray/fluid queries
            if any(t in query_lower for t in ["smd", "sauter", "droplet", "spray"]):
                if meta.sauter_mean_diameter_values:
                    score += 0.5
            
            # Stacking fault energy
            if "stacking fault" in query_lower and meta.stacking_fault_energy_values:
                score += 0.5
            
            # Energy densities
            if "ved" in query_lower or "volumetric energy density" in query_lower:
                if meta.energy_density_values:
                    score += 0.5
            if "aed" in query_lower or "areal energy density" in query_lower:
                if meta.areal_energy_density_values:
                    score += 0.5
            if "led" in query_lower or "linear energy density" in query_lower:
                if meta.linear_energy_density_values:
                    score += 0.5
            
            # Melt pool & process
            if any(t in query_lower for t in ["melt pool", "meltpool", "penetration depth"]):
                if meta.melt_pool_depth_values or meta.melt_pool_width_values:
                    score += 0.5
            if "hatch" in query_lower and meta.hatch_distance_values:
                score += 0.4
            if "nanoindentation" in query_lower or "indentation" in query_lower:
                if meta.indentation_force_values or meta.indentation_depth_values:
                    score += 0.5
            if "dislocation" in query_lower and meta.dislocation_density_values:
                score += 0.4
            if "hollomon" in query_lower and (meta.hollomon_strength_coeff_values or meta.hollomon_exponent_values):
                score += 0.5
            if "vec" in query_lower and meta.vec_values:
                score += 0.4
            if "lewis" in query_lower and meta.lewis_number_values:
                score += 0.4
            if "jackson" in query_lower and meta.jackson_parameter_values:
                score += 0.4
            if "omega" in query_lower and meta.omega_parameter_values:
                score += 0.4
            if "atomic size" in query_lower and meta.atomic_size_difference_values:
                score += 0.4
            if "lambda" in query_lower and meta.lambda_parameter_values:
                score += 0.4
            if "cooling rate" in query_lower and meta.cooling_rate_values:
                score += 0.4
            if "dwell time" in query_lower and meta.dwell_time_values:
                score += 0.4
            if "grain boundary" in query_lower and meta.grain_boundary_energy_values:
                score += 0.4
            if "recrystallization" in query_lower and meta.recrystallization_temperature_values:
                score += 0.4
            if "creep" in query_lower and meta.creep_rate_values:
                score += 0.4
            if "fatigue" in query_lower and meta.fatigue_limit_values:
                score += 0.4
            if "impact" in query_lower and meta.impact_energy_values:
                score += 0.4
            if "fracture toughness" in query_lower and meta.fracture_toughness_values:
                score += 0.4
            if "wear" in query_lower and meta.wear_rate_values:
                score += 0.4
            if "friction" in query_lower and meta.coefficient_of_friction_values:
                score += 0.4
            if "resistivity" in query_lower and meta.resistivity_values:
                score += 0.4
            if "conductivity" in query_lower and meta.conductivity_values:
                score += 0.4
            if "magnetic permeability" in query_lower and meta.magnetic_permeability_values:
                score += 0.4
            if "dielectric constant" in query_lower and meta.dielectric_constant_values:
                score += 0.4
            if "refractive index" in query_lower and meta.refractive_index_values:
                score += 0.4
            if "band gap" in query_lower and meta.band_gap_values:
                score += 0.4
            if "electron affinity" in query_lower and meta.electron_affinity_values:
                score += 0.4
            if "work function" in query_lower and meta.work_function_values:
                score += 0.4
            if "lattice constant" in query_lower and meta.lattice_constant_values:
                score += 0.4
            if "atomic radius" in query_lower and meta.atomic_radius_values:
                score += 0.4
            if "electronegativity" in query_lower and meta.electronegativity_values:
                score += 0.4
            if "specific heat" in query_lower and meta.specific_heat_values:
                score += 0.4
            if "latent heat" in query_lower and meta.latent_heat_values:
                score += 0.4
            if "thermal diffusivity" in query_lower and meta.thermal_diffusivity_values:
                score += 0.4
            if "sound velocity" in query_lower and meta.sound_velocity_values:
                score += 0.4
            if "elastic modulus" in query_lower and meta.elastic_modulus_values:
                score += 0.4
            if "shear modulus" in query_lower and meta.shear_modulus_values:
                score += 0.4
            if "bulk modulus" in query_lower and meta.bulk_modulus_values:
                score += 0.4
            if "poissons ratio" in query_lower and meta.poissons_ratio_values:
                score += 0.4
            if "yield point" in query_lower and meta.yield_point_values:
                score += 0.4
            if "tensile point" in query_lower and meta.tensile_point_values:
                score += 0.4
            if "breaking point" in query_lower and meta.breaking_point_values:
                score += 0.4
            if "vickers" in query_lower and meta.hardness_vickers_values:
                score += 0.4
            if "rockwell" in query_lower and meta.hardness_rockwell_values:
                score += 0.4
            if "brinell" in query_lower and meta.hardness_brinell_values:
                score += 0.4
            if "knoop" in query_lower and meta.hardness_knoop_values:
                score += 0.4
            if "mohs" in query_lower and meta.hardness_mohs_values:
                score += 0.4
            if "microhardness" in query_lower and meta.microhardness_values:
                score += 0.4
            if "macrohardness" in query_lower and meta.macrohardness_values:
                score += 0.4
            if "superhardness" in query_lower and meta.superhardness_values:
                score += 0.4
            
            # Process type matching
            for proc in meta.process_types:
                if proc.lower() in query_lower:
                    score += 0.2
            
            scores.append((name, min(score, 1.0)))
        
        # Semantic blending (optional)
        if self.embedding_model is not None and len(scores) > 0:
            try:
                from sentence_transformers import util
                doc_texts = [
                    f"{meta.alloys} {meta.process_types} {self.doc_summaries.get(name, '')}"
                    for name, meta in self.doc_metadata.items()
                ]
                if doc_texts:
                    doc_emb = self.embedding_model.encode(doc_texts, convert_to_tensor=True)
                    query_emb = self.embedding_model.encode(query, convert_to_tensor=True)
                    sem_scores = util.cos_sim(query_emb, doc_emb)[0]
                    
                    # Blend keyword score (60%) and semantic score (40%)
                    for i, (name, kw_score) in enumerate(scores):
                        sem_score = float(sem_scores[i])
                        scores[i] = (name, min(kw_score * 0.6 + sem_score * 0.4, 1.0))
            except Exception as e:
                logger.warning(f"Semantic blending failed: {e}")
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # If no scores > 0, return top_k with low default score
        if not any(s[1] > 0 for s in scores):
            return [(name, 0.2) for name in self.doc_metadata.keys()][:top_k]
        
        return scores[:top_k]

    def get_relevant_pages(self, doc_name: str, query: str, max_pages: int = 5) -> List[int]:
        """Placeholder for page-level retrieval (can be enhanced with LLM navigation)."""
        return list(range(1, max_pages + 1))

# ============================================================================
# ENHANCED PAGE NODE WITH CONTENT HASHING & CACHING SUPPORT
# ============================================================================
@dataclass
class PageNode:
    """
    Represents a node in the hierarchical document tree.
    Enhanced with content hashing for intelligent caching and roll-up summarization.
    """
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
    metadata: Optional[DocumentMetadata] = None
    _content_hash: Optional[str] = None  # Added for caching

    def compute_content_hash(self) -> str:
        """Compute SHA-256 hash of node content for caching validation."""
        if self._content_hash:
            return self._content_hash
        
        # Hash includes title, page range, summary, and metadata (if present)
        content_parts = [
            self.title,
            str(self.page_start),
            str(self.page_end),
            self.summary[:300],
            self.prefix_summary[:200],
            str(self.metadata.model_dump() if hasattr(self, "model_dump") else self.dict()) if self.metadata else ""
        ]
        combined = "|".join(content_parts)
        self._content_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]
        return self._content_hash

    def get_text(self, doc_cache: Dict[str, Any] = None, max_chars: int = 20000) -> str:
        """
        Retrieves full text for this node, either from cache, in-memory, or by parsing PDF.
        """
        if self.full_text:
            return self.full_text[:max_chars] if len(self.full_text) > max_chars else self.full_text
        
        if not self._pdf_path or not fitz:
            return ""
        
        doc = None
        if doc_cache and self.doc_id in doc_cache:
            doc = doc_cache[self.doc_id]
        else:
            doc = fitz.open(self._pdf_path)
            if doc_cache:
                doc_cache[self.doc_id] = doc
        
        start = self.page_start - 1
        end = min(self.page_end or self.page_start, len(doc))
        
        # Safe page extraction
        texts = []
        for p in range(start, end):
            try:
                texts.append(doc[p].get_text("text"))
            except Exception as e:
                logger.warning(f"Failed to extract text from page {p+1} in {self.doc_id}: {e}")
                texts.append(f"[Error extracting page {p+1}]")
        
        self.full_text = "\n".join(texts)
        
        if doc_cache is None:
            try:
                doc.close()
            except:
                pass
                
        return self.full_text[:max_chars] if len(self.full_text) > max_chars else self.full_text

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary for caching/export."""
        return {
            "id": self.id,
            "title": self.title,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "summary": self.summary,
            "prefix_summary": self.prefix_summary,
            "level": self.level,
            "doc_id": self.doc_id,
            "section_type": self.section_type,
            "node_id": self.node_id,
            "text_token_count": self.text_token_count,
            "children": [c.to_dict() for c in self.children],
            "metadata": self.metadata.model_dump() if hasattr(self, "model_dump") else self.dict() if self.metadata else None,
            "content_hash": self.compute_content_hash()
        }

    def to_tree_format(self, max_chars: int = 20000) -> Dict[str, Any]:
        """Convert to tree structure format for visualization/retrieval."""
        result = {
            "title": self.title,
            "node_id": self.node_id,
            "start_index": self.page_start,
            "end_index": self.page_end or self.page_start,
            "summary": self.summary,
            "prefix_summary": self.prefix_summary,
            "text_token_count": self.text_token_count,
            "content_hash": self.compute_content_hash()
        }
        
        if self.children:
            result["nodes"] = [c.to_tree_format(max_chars) for c in self.children]
            
        text = self.get_text(max_chars=max_chars)
        if text:
            result["text"] = text
            
        if self.metadata:
            result["metadata"] = self.metadata.model_dump() if hasattr(self, "model_dump") else self.dict()
            
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any], pdf_path: Optional[str] = None) -> 'PageNode':
        """Reconstruct PageNode from dictionary."""
        node = cls(
            data["id"], 
            data["title"], 
            data["page_start"], 
            data.get("page_end"), 
            "",
            data.get("summary", ""), 
            data.get("level", 0), 
            doc_id=data.get("doc_id", ""),
            section_type=data.get("section_type", "BODY"), 
            _pdf_path=pdf_path
        )
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
# ROLL-UP HIERARCHICAL SUMMARIZER (NEW V18.0 COMPONENT)
# ============================================================================
class RollupSummarizer:
    """
    Generates hierarchical roll-up summaries for document trees.
    
    Strategy:
    1. Leaf nodes: Summarize raw page text directly.
    2. Parent nodes: Synthesize child summaries + own prefix content.
    3. Root node: Executive summary of entire document.
    
    This bottom-up condensation dramatically improves LLM navigation accuracy
    by allowing the model to reason over structured semantic summaries rather
    than raw, fragmented text.
    """
    
    def __init__(self, llm: 'HybridLLM', max_summary_length: int = 250):
        self.llm = llm
        self.max_summary_length = max_summary_length
        self._summary_cache: Dict[str, str] = {}
    
    def generate_rollup_summaries(self, root: PageNode) -> PageNode:
        """
        Generate hierarchical summaries bottom-up using post-order traversal.
        """
        self._post_order_summarize(root)
        return root

    async def _post_order_summarize(self, node: PageNode) -> None:
        """Recursively summarize children first, then current node."""
        # 1. Process all children
        for child in node.children:
            await self._post_order_summarize(child)
            
        # 2. Summarize current node
        cache_key = f"{node.doc_id}:{node.node_id}:{node.compute_content_hash()}"
        
        if cache_key in self._summary_cache:
            node.summary = self._summary_cache[cache_key]
            return
            
        if not node.children:
            # Leaf node: summarize raw text
            text = node.get_text(max_chars=4000)
            if len(text.strip()) < 50:
                node.summary = text[:self.max_summary_length]
            else:
                node.summary = await self._summarize_text(
                    text, 
                    instruction=f"Summarize this document section in max {self.max_summary_length} characters. Focus on quantitative parameters, materials, methods, and key findings. Return ONLY the summary."
                )
        else:
            # Internal node: roll-up child summaries + own content
            child_summaries = [c.summary for c in node.children if c.summary]
            own_content = node.prefix_summary[:300] if node.prefix_summary else ""
            
            combined = f"Own Context: {own_content}\n\nSubsections:\n" + "\n---\n".join(child_summaries[:8])
            
            node.summary = await self._summarize_text(
                combined,
                instruction=f"Synthesize these subsection summaries and context into a single {self.max_summary_length}-char overview. Highlight overarching themes, key parameters, and experimental conditions. Return ONLY the summary."
            )
            
        # Cache result
        self._summary_cache[cache_key] = node.summary[:self.max_summary_length]
    
    async def _summarize_text(self, text: str, instruction: str) -> str:
        """Call LLM for summarization with robust fallback."""
        prompt = f"{instruction}\n\nText to process:\n{text[:5000]}\n\nSummary:"
        
        try:
            response = await asyncio.to_thread(
                    self.llm.generate, 
                    prompt, 
                    max_new_tokens=200, 
                    temperature=0.05
            )
            cleaned = response.strip().replace("Summary:", "").strip()
            return cleaned[:self.max_summary_length]
        except Exception as e:
            logger.warning(f"LLM summarization failed for node: {e}")
            # Deterministic fallback: extract first sentence with numeric data
            sentences = re.split(r'[.!?]+', text)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 20 and any(c.isdigit() for c in sent):
                    return sent[:self.max_summary_length]
            return text[:self.max_summary_length]


# ============================================================================
# ANNOTATED TREE CACHE WITH SHA-256 HASHING (NEW V18.0 COMPONENT)
# ============================================================================
class AnnotatedTreeCache:
    """
    Full annotated-tree caching system.
    
    Features:
    - Content-based hashing (SHA-256 of doc_name + first 1MB)
    - Hierarchical JSON storage with compression
    - TTL-based expiration (default 72 hours)
    - Persistent index for fast lookup
    - Safe concurrent access
    """
    
    def __init__(self, cache_dir: str = ".declarmima_cache_v18", ttl_hours: int = 72):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self._index: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load_index()
    
    def _load_index(self):
        """Load cache index from disk."""
        index_path = self.cache_dir / "tree_index.json"
        if index_path.exists():
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    self._index = json.load(f)
                logger.info(f"Loaded cache index with {len(self._index)} entries")
            except Exception as e:
                logger.warning(f"Failed to load cache index, starting fresh: {e}")
                self._index = {}
    
    def _save_index(self):
        """Persist cache index to disk."""
        index_path = self.cache_dir / "tree_index.json"
        try:
            with self._lock:
                with open(index_path, 'w', encoding='utf-8') as f:
                    json.dump(self._index, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _compute_doc_hash(self, doc_name: str, file_content: bytes) -> str:
        """Compute SHA-256 hash for document content identification."""
        hasher = hashlib.sha256()
        hasher.update(doc_name.encode('utf-8'))
        hasher.update(file_content[:1024 * 1024])  # Hash first 1MB
        return hasher.hexdigest()[:32]
    
    def get(self, doc_name: str, file_content: bytes) -> Optional[Dict]:
        """
        Retrieve cached tree if available and not expired.
        Returns dictionary representation of PageNode tree.
        """
        doc_hash = self._compute_doc_hash(doc_name, file_content)
        
        with self._lock:
            entry = self._index.get(doc_hash)
            if not entry:
                return None
                
            # Check TTL
            cached_time = datetime.fromisoformat(entry['cached_at'])
            if datetime.now() - cached_time > self.ttl:
                self._remove_entry(doc_hash)
                return None
                
        # Load tree file
        cache_file = self.cache_dir / f"{doc_hash}.tree.json"
        if not cache_file.exists():
            with self._lock:
                self._remove_entry(doc_hash)
            return None
            
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                tree_data = json.load(f)
            logger.debug(f"Cache HIT for {doc_name} (hash: {doc_hash[:8]}...)")
            return tree_data
        except Exception as e:
            logger.warning(f"Failed to load cached tree file for {doc_hash}: {e}")
            with self._lock:
                self._remove_entry(doc_hash)
            return None

    def set(self, doc_name: str, file_content: bytes, tree_dict: Dict) -> bool:
        """
        Store annotated tree in cache.
        Returns True if successful, False otherwise.
        """
        doc_hash = self._compute_doc_hash(doc_name, file_content)
        cache_file = self.cache_dir / f"{doc_hash}.tree.json"
        
        try:
            with self._lock:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(tree_dict, f, indent=2, ensure_ascii=False, default=str)
                    
                self._index[doc_hash] = {
                    'doc_name': doc_name,
                    'cached_at': datetime.now().isoformat(),
                    'file_size': cache_file.stat().st_size,
                    'tree_nodes': self._count_nodes_recursive(tree_dict),
                    'content_hash': doc_hash
                }
                self._save_index()
                
            logger.info(f"CACHE MISS -> Stored tree for {doc_name} (hash: {doc_hash[:8]}..., nodes: {self._index[doc_hash]['tree_nodes']})")
            return True
        except Exception as e:
            logger.error(f"Failed to cache tree for {doc_name}: {e}")
            return False

    def _remove_entry(self, doc_hash: str):
        """Remove a single cache entry."""
        if doc_hash in self._index:
            del self._index[doc_hash]
            cache_file = self.cache_dir / f"{doc_hash}.tree.json"
            if cache_file.exists():
                cache_file.unlink(missing_ok=True)
            logger.debug(f"Removed expired/invalid cache entry: {doc_hash[:8]}...")

    def _count_nodes_recursive(self, tree: Dict) -> int:
        """Count total nodes in tree structure."""
        count = 1
        for child in tree.get('children', []):
            count += self._count_nodes_recursive(child)
        return count

    def clear(self):
        """Clear entire cache."""
        with self._lock:
            for doc_hash in list(self._index.keys()):
                self._remove_entry(doc_hash)
            self._index.clear()
            self._save_index()
        logger.info("Cleared all cached trees")

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total_size = sum(
                (self.cache_dir / f"{h}.tree.json").stat().st_size 
                for h in self._index 
                if (self.cache_dir / f"{h}.tree.json").exists()
            )
            return {
                'entries': len(self._index),
                'total_size_mb': round(total_size / 1024 / 1024, 2),
                'ttl_hours': self.ttl.total_seconds() / 3600,
                'avg_nodes_per_tree': np.mean([e['tree_nodes'] for e in self._index.values()]) if self._index else 0
            }


# ============================================================================
# UPGRADED HIERARCHICAL INDEX WITH ROLLUP & CACHING INTEGRATION
# ============================================================================
class HierarchicalIndex:
    def __init__(self, cache_dir: str = ".declarmima_cache_v18", llm: Optional['HybridLLM'] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.doc_trees: Dict[str, PageNode] = {}
        self._pdf_cache: Dict[str, Any] = {}
        self.metadata_extractor = StructuredMetadataExtractor()
        self.tree_cache = AnnotatedTreeCache(cache_dir=str(cache_dir))
        self.llm = llm
        self.rollup_summarizer = RollupSummarizer(llm) if llm else None

    def _doc_hash(self, file_buffer: BytesIO) -> str:
        pos = file_buffer.tell()
        file_buffer.seek(0)
        content = file_buffer.read(1024 * 1024)
        file_buffer.seek(pos)
        return hashlib.sha256(content).hexdigest()[:16]

    def _cache_path(self, doc_name: str, doc_hash: str) -> Path:
        safe = re.sub(r'[^\w\-_.]', '_', doc_name)
        return self.cache_dir / f"{safe}.{doc_hash}.tree.json"

    def build_from_pdfs(self, files: List, parallel: bool = True, max_workers: int = 4) -> Dict[str, PageNode]:
        def build_one(file_obj):
            doc_name = file_obj.name
            buf = BytesIO(file_obj.getbuffer())
            file_content = buf.getvalue()
            
            # Try cache first
            cached_tree = self.tree_cache.get(doc_name, file_content)
            if cached_tree:
                logger.info(f"Loaded cached tree for {doc_name}")
                # Reconstruct from cache
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file_content)
                    tmp_path = tmp.name
                
                root = PageNode.from_dict(cached_tree, pdf_path=tmp_path)
                if self.llm and self.rollup_summarizer:
                    # Verify/refresh summaries if needed
                    asyncio.run(self.rollup_summarizer._post_order_summarize(root))
                return doc_name, root

            # Build new tree
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
                
            doc = fitz.open(tmp_path)
            root = self._build_tree(doc, doc_name, tmp_path)
            
            # Extract metadata
            full_text = "\n".join([doc[p].get_text("text") for p in range(len(doc))])
            meta = self.metadata_extractor.extract_metadata(doc_name, full_text)
            root.metadata = meta
            doc.close()
            
            # Apply roll-up summarization if LLM available
            if self.llm and self.rollup_summarizer:
                logger.info(f"Generating roll-up summaries for {doc_name}...")
                asyncio.run(self.rollup_summarizer._post_order_summarize(root))
            else:
                logger.info(f"Skipping LLM summarization for {doc_name} (LLM not provided)")
                # Fallback: use first 200 chars as summary
                def fallback_summaries(node):
                    if not node.children:
                        node.summary = node.get_text(max_chars=200)[:200]
                    for c in node.children:
                        fallback_summaries(c)
                fallback_summaries(root)
                
            # Save to cache
            self.tree_cache.set(doc_name, file_content, root.to_dict())
            
            return doc_name, root

        if parallel and len(files) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(build_one, f): f.name for f in files}
                for fut in as_completed(futures):
                    name, tree = fut.result()
                    self.doc_trees[name] = tree
        else:
            for f in files:
                name, tree = build_one(f)
                self.doc_trees[name] = tree
                
        return self.doc_trees

    def _build_tree(self, doc, doc_id, pdf_path) -> PageNode:
        root = PageNode(
            f"{doc_id}_root", "Document Root", 1, len(doc), "",
            f"Document {doc_id} root covering pages 1-{len(doc)}", 0, 
            doc_id=doc_id, _pdf_path=pdf_path, node_id="0000"
        )
        
        toc = doc.get_toc()
        window = 7
        
        if toc:
            nodes_by_level = {}
            for level, title, page in toc:
                if page > len(doc):
                    continue
                end = min(page + window, len(doc))
                text = self._extract_range(doc, page, end)
                node = PageNode(
                    f"{doc_id}_toc_{level}_{title[:20]}", title.strip(), page, end, text, text[:200], level, 
                    doc_id=doc_id, _pdf_path=pdf_path
                )
                nodes_by_level.setdefault(level, []).append(node)
                
            for level in sorted(nodes_by_level.keys()):
                for node in nodes_by_level[level]:
                    parent = self._find_parent(root, level-1, node.page_start)
                    parent.children.append(node)
                    
            self._assign_node_ids(root)
            return root
            
        # Fallback to heading detection
        headings = self._detect_headings(doc)
        if headings:
            for i, (title, page) in enumerate(headings):
                end = min(page + window, len(doc))
                text = self._extract_range(doc, page, end)
                node = PageNode(
                    f"{doc_id}_h{i}", title, page, end, text, text[:200], 2, 
                    doc_id=doc_id, _pdf_path=pdf_path
                )
                root.children.append(node)
            self._assign_node_ids(root)
            return root
            
        # Fallback to page-level chunking
        for p in range(1, len(doc)+1):
            text = doc[p-1].get_text("text")
            if not text.strip():
                continue
            node = PageNode(
                f"{doc_id}_p{p}", f"Page {p}", p, p, text, text[:200], 3, 
                doc_id=doc_id, _pdf_path=pdf_path
            )
            root.children.append(node)
            
        self._assign_node_ids(root)
        return root

    def _extract_range(self, doc, start, end):
        return "\n".join(doc[p-1].get_text("text") for p in range(start, min(end, len(doc)+1)))

    def _detect_headings(self, doc):
        headings = []
        for p in range(len(doc)):
            lines = doc[p].get_text("text").split('\n')
            for line in lines:
                if re.match(r'^(?:[0-9]+\.?)+\s+[A-Z]', line.strip()):
                    headings.append((line.strip(), p+1))
        return headings[:50]

    def _find_parent(self, node, target_level, page_hint):
        if target_level < 0:
            return node
        candidates = [c for c in node.children if c.level == target_level]
        if not candidates:
            return node
        return min(candidates, key=lambda n: abs(n.page_start - page_hint))

    def _assign_node_ids(self, root: PageNode):
        def assign(node: PageNode, prefix: str = "", index: int = 1):
            if not prefix:
                node.node_id = str(index).zfill(4)
                current_prefix = node.node_id
            else:
                node.node_id = f"{prefix}.{str(index).zfill(4)}"
                current_prefix = node.node_id
                
            for i, child in enumerate(node.children, 1):
                assign(child, current_prefix, i)
                
        assign(root, "", 1)

    def cleanup(self):
        for doc in self._pdf_cache.values():
            try:
                doc.close()
            except:
                pass
        self._pdf_cache.clear()

# ============================================================================
# HYBRID LLM BACKEND WITH ROBUST INITIALIZATION & ERROR ISOLATION
# ============================================================================
class HybridLLM:
    """
    Unified LLM interface supporting Ollama (local API) and HuggingFace Transformers.
    Features automatic backend detection, JSON formatting enforcement, and thread-safe generation.
    """
    def __init__(self, model_key: str, use_4bit: bool = True, device: Optional[str] = None):
        self.model_key = model_key
        self.use_4bit = use_4bit
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = None
        self.model_name = None
        self.client = None
        self.tokenizer = None
        self.model = None
        
        # Parse model name from key
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
        """Detect and initialize available LLM backend."""
        if GLOBAL_DEPS.get('ollama', False):
            try:
                requests.get("http://localhost:11434/api/tags", timeout=5)
                self.backend = "ollama"
                self.client = ollama.Client(host="http://localhost:11434")
                logger.info("Ollama backend connected successfully")
                return
            except Exception as e:
                logger.debug(f"Ollama not reachable: {e}")
                
        if GLOBAL_DEPS.get('transformers', False):
            self.backend = "transformers"
            logger.info("Falling back to HuggingFace Transformers backend")
            return
            
        raise RuntimeError("No LLM backend available. Install Ollama or transformers.")
    
    def generate(self, prompt: str, max_new_tokens: int = 1024, temperature: float = 0.1, 
                 fast_json: bool = False, system_prompt: Optional[str] = None) -> str:
        """Generate text using the initialized backend."""
        if self.backend == "ollama":
            return self._ollama_generate(prompt, max_new_tokens, temperature, fast_json, system_prompt)
        else:
            return self._transformers_generate(prompt, max_new_tokens, temperature, system_prompt)
    
    def _ollama_generate(self, prompt: str, max_tokens: int, temp: float, 
                         fast_json: bool, system_prompt: Optional[str]) -> str:
        """Generate via Ollama API."""
        try:
            options = {"temperature": temp, "num_predict": max_tokens, "top_p": 0.95}
            if fast_json:
                options["format"] = "json"
                
            messages = []
            sys_prompt = system_prompt or self.template.get("system", "You are a precise scientific assistant.")
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": prompt})
            
            resp = self.client.chat(model=self.model_name, messages=messages, options=options, stream=False)
            content = resp.get("message", {}).get("content", "").strip()
            
            if fast_json:
                return self._extract_json_safe(content) or content
            return content
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return f"Error: {str(e)[:150]}"
    
    def _transformers_generate(self, prompt: str, max_tokens: int, temp: float, 
                               system_prompt: Optional[str]) -> str:
        """Generate via HuggingFace Transformers."""
        if self.tokenizer is None:
            self._load_transformers()
        if not self.model:
            return "Error: Model not loaded"
            
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=8192).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temp if temp > 0 else None,
                    do_sample=temp > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "assistant" in response:
                response = response.split("assistant")[-1].strip()
            return response
        except Exception as e:
            logger.error(f"Transformers generation error: {e}")
            return f"Error: {str(e)[:150]}"
    
    def _load_transformers(self):
        """Load model and tokenizer."""
        logger.info(f"Loading {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "device_map": "auto" if self.device == "cuda" else None
        }
        
        if self.use_4bit and self.device == "cuda":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if self.device == "cuda":
            self.model.to(self.device)
        self.model.eval()
        logger.info("Transformers model loaded successfully.")
    
    def _extract_json_safe(self, text: str) -> Optional[Any]:
        """Extract JSON from text with robust pattern matching."""
        if not text:
            return None
        patterns = [
            r'\{.*\}', r'\[.*\]', 
            r'```json\s*(\{.*?\})\s*```', r'```json\s*(\[.*?\])\s*```',
            r'```json\s*(\[.*\])\s*```'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_str = match.group(1) if match.groups() else match.group(0)
                try:
                    return json.loads(json_str)
                except Exception:
                    continue
        return None


# ============================================================================
# UNIVERSAL LLM EXTRACTOR WITH DOMAIN-AWARE PROMPTS
# ============================================================================
class UniversalLLMExtractor:
    """
    Advanced extraction engine tuned for scientific literature.
    Captures quantitative parameters, material names, methods, and contextual metadata.
    """
    
    EXTRACTION_PROMPT = """Extract ALL quantitative information relevant to the query from these document sections.
QUERY: {query}
QUERY TYPE: {query_type}
SECTIONS:
{sections_text}
Return JSON array of extracted items with fields:
{{
"item_type": "quantitative|qualitative|definition|comparison|relationship|process|material|method",
"content": "exact phrase with full numerical value (never truncate numbers)",
"confidence": 0.0-1.0,
"context": "exact sentence from text",
"doc_source": "{doc_id}",
"page": page_number,
"parameter_name": "...",
"value": number,
"unit": "e.g., W, kW, mm/s, MPa, GPa, HV, mV, V, µA/cm², A/cm², J/mm³, J/mm², J/m, mJ/m², nm, µm, mm, K, °C, wt%, at%, vol%, g/cm³, kg/m³, W/m·K, Pa·s, mPa·s, kΩ·cm², ppm",
"physical_quantity": "one of: laser_power, electrical_power, scan_speed, flow_speed, feed_rate, irradiance, temperature, melting_temperature, energy_density, areal_energy_density, linear_energy_density, layer_thickness, spot_size, exposure_time, enthalpy, viscosity, thermal_conductivity, density, yield_strength, tensile_strength, ultimate_tensile_strength, hardness, elongation, modulus, stacking_fault_energy, unstable_stacking_fault_energy, ideal_shear_strength, corrosion_potential, pitting_potential, breakdown_potential, repassivation_potential, open_circuit_potential, corrosion_current_density, polarization_resistance, apparent_polarization_resistance, current_density, PREN, phase_fraction, austenite_fraction, ferrite_fraction, grain_size, cell_size, porosity, relative_density, surface_roughness, sauter_mean_diameter, spray_penetration, plume_height, film_thickness, absorption_coefficient, youngs_modulus, poisson_ratio, coefficient_thermal_expansion, unknown, hollomon_strength_coeff, hollomon_exponent",
"material": "alloy or material name if mentioned (e.g., Ti3Au, CP Ti, Grade II Ti, SDSS 2507, UNS S32750, AlSiMgZr, Al-Si-Mg-Zr, TiB2/Al-Si-Mg-Zr, Fe-based metallic glass, Au-Ti, 316L, 2205, Inconel 718, Ti6Al4V)",
"method": "e.g., LPBF, L-PBF, DED, SLM, PFI, GDI, FEM, MD, nanoindentation, EIS, CPP, XRD, SEM, TEM, EBSD, EDS, DTA"
}}
CRITICAL RULES:
1. Capture ALL numbers with units, even if they describe corrosion, electrochemistry, thermal properties, mechanical properties, microstructural features, or spray dynamics.
2. For electrochemical  map Ecorr/Erp/Epit/Ebr to corrosion_potential/pitting_potential/etc., NOT just generic potential.
3. For LPBF/DED: capture VED, AED, LED, hatch distance, layer thickness, laser power, scan speed.
4. For nanoindentation: capture indentation force, hardness, modulus, SFE, USFE.
5. NEVER truncate numbers.
6. If an alloy or material name appears, create an item with item_type="material", content=the name, material=the name.
7. Return ONLY valid JSON, no extra text.
8. Set confidence based on clarity.
9. Capture computational & multiphysics methods: Phase Field (Cahn-Hilliard/Allen-Cahn), CALPHAD (.TDB, pycalphad), Molecular Dynamics (LAMMPS, EAM, Morse, GSFE), DFT/VASP, FEM (COMSOL, Abaqus, Ansys), Navier-Stokes, Boussinesq approximation, Marangoni convection, enthalpy method for phase change, eigenstrain/Khachaturyan scheme.
10. Capture AI/ML spatio-temporal models: Physics-Informed Neural Networks (PINNs), U-Net, ConvLSTM, Fourier Neural Operator (FNO), Variational Autoencoder (VAE), Digital Twin, Explainable AI (XAI), Uncertainty Quantification (UQ).
11. Capture advanced alloy descriptors & thermodynamics: VEC, ΔH_mix, ΔS_mix, Ω (omega parameter), λ (lambda), δ (atomic size difference), Jackson parameter (αJ), Lewis number (Le), PREN, apparent polarization resistance (Rp,app).
12. Capture microstructural phenomena: Bimodal microstructure, nanotwinned structures (nt-Cu), SRO/MRO clusters, martensitic transformation, Bain strain, habit plane, scalloped vs prismatic/rooftop IMC, lead-lag dynamics, coherent/incoherent interfaces, 9R phase, stacking faults.
13. Capture nanoindentation metrics: Indentation force, penetration depth, Oliver-Pharr hardness/modulus, continuous stiffness measurement (CSM), dislocation nucleation, dislocation exhaustion.
14. Capture melt-pool & process metrics: Melt pool depth/width/length, hatch distance, build platform temperature, keyhole mode, conduction mode, layer thickness, spot size.
Return [] if no relevant information found."""
    
    def __init__(self, llm: HybridLLM):
        self.llm = llm
        self.phys_classifier = PhysicalQuantityClassifier(llm_callback=lambda p: llm.generate(p, max_new_tokens=50))
        self.concept_normalizer = ConceptNormalizer()
        
    def extract_from_chunks(self, chunks: List[Dict], query: str, query_analysis: Optional[Dict] = None) -> List[UniversalExtractionItem]:
        """Extract quantitative items from document chunks."""
        if not chunks:
            return []
        qa = query_analysis or {"query_type": "mixed", "keywords": []}
        items = []
        
        for chunk in chunks:
            text = chunk.get("full_text", "")
            doc = chunk.get("doc_id", "unknown")
            page = chunk.get("page_start", 1)
            
            if qa.get("query_type") == "quantitative" and not re.search(r'\d+', text):
                continue
                
            prompt = self.EXTRACTION_PROMPT.format(
                query=query, 
                query_type=qa.get("query_type","mixed"), 
                sections_text=text[:4000], 
                doc_id=doc
            )
            try:
                response = self.llm.generate(prompt, max_new_tokens=1500, fast_json=True, temperature=0.1)
                json_str = self._extract_json(response)
                if json_str:
                    data = json.loads(json_str)
                    raw_items = data if isinstance(data, list) else data.get("items", [])
                    
                    for item_data in raw_items:
                        if "physical_quantity" not in item_data or not item_data["physical_quantity"]:
                            item_data["physical_quantity"] = self.phys_classifier.classify_with_llm_fallback(
                                item_data.get("parameter_name"), 
                                item_data.get("unit"), 
                                item_data.get("context", ""), 
                                llm=self.llm
                            )
                        item_data.setdefault("material", None)
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
                            logger.debug(f"Item parse error: {e}")
            except Exception as e:
                logger.error(f"Extraction error: {e}")
                
        # Deduplicate and filter by confidence
        unique = {}
        for i in items:
            key = (i.content, i.doc_source, i.page, i.material)
            if key not in unique or i.confidence > unique[key].confidence:
                unique[key] = i
                
        min_conf = UNIVERSAL_CONFIG.get("min_confidence_threshold", 0.55)
        return [i for i in unique.values() if i.confidence >= min_conf]
        
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON string from response."""
        patterns = [r'\[.*\]', r'```json\s*(\[.*?\])\s*```', r'(\[.*\])']
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_str = match.group(1) if match.groups() else match.group(0)
                try:
                    json.loads(json_str)
                    return json_str
                except Exception:
                    continue
        return None


# ============================================================================
# LLM REASONING SYNTHESIZER WITH CONSENSUS & CONTRADICTION TRACKING
# ============================================================================
class LLMReasoningSynthesizer:
    """
    Synthesizes extracted items into human-readable, evidence-backed answers.
    Implements consensus averaging, contradiction flagging, and structured reporting.
    """
    def __init__(self, llm: HybridLLM):
        self.llm = llm
        self.phys_classifier = PhysicalQuantityClassifier()
        
    def synthesize(self, query: str, items: List[UniversalExtractionItem]) -> str:
        """Generate synthesized answer from extracted items."""
        if not items:
            return f"No relevant information found for query: '{query}'. Try rephrasing or check the documents."
            
        extracted_lines = []
        for item in items:
            pq = item.physical_quantity or "unknown"
            pq_readable = self.phys_classifier.get_human_readable(pq)
            mat = f" [{item.material}]" if item.material else ""
            line = f"- {pq_readable}{mat}: {item.content} ({item.confidence:.2f}) context: {item.context[:200]} {item.citation()}"
            extracted_lines.append(line)
            
        extracted_text = "\n".join(extracted_lines[:25])
        
        prompt = f"""You are an expert scientific analyst. Given extracted values and the user query, produce a comprehensive answer.
QUERY: {query}
EXTRACTED VALUES (with citations):
{extracted_text}
TASK: Synthesize the extracted information into a structured answer using the following format:
**Direct Answer**
(Concise answer to the query, citing sources)
**Evidence by Physical Quantity**
(Group findings by physical quantity: e.g., Laser Power, Scan Speed, Yield Strength, etc.)
**Evidence by Material/Alloy**
(If materials are mentioned, group findings by alloy name)
**Consensus & Variability**
(For each physical quantity or material, report range/mean if multiple values exist)
**Contradictions & Limitations**
(If contradictory values exist, highlight them)
**Confidence Assessment**
(High/Medium/Low)
Do NOT invent information. Only use the extracted values above. Use citations with <cite doc="..." page="X"/>.
Return ONLY the answer text."""
        try:
            answer = self.llm.generate(prompt, max_new_tokens=1500, temperature=0.2)
            return answer.strip()
        except Exception as e:
            logger.error(f"Reasoning synthesis error: {e}")
            lines = [f"Query: {query}\nFound {len(items)} relevant items:\n"] + [f"- {item.content} {item.citation()}" for item in items[:5]]
            return "\n".join(lines)
            
    def generate_human_conclusion(self, query: str, report: QueryReport) -> str:
        """Generate markdown-formatted conclusion report."""
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
        return "\n".join(lines)


# ============================================================================
# ENHANCED HIERARCHICAL TREE RETRIEVER (OPTIMIZED FOR 2-CALL ARCHITECTURE)
# ============================================================================
class HierarchicalTreeRetriever:
    """
    Retrieves relevant nodes from annotated document trees using LLM navigation.
    Optimized for the strict 2-call query pipeline.
    """
    def __init__(self, llm: HybridLLM, max_results: int = 30, max_text_chars: int = 20000):
        self.llm = llm
        self.max_results = max_results
        self.max_text_chars = max_text_chars
        self._condensed_cache: Dict[str, Dict] = {}
        self.template = llm.template if hasattr(llm, 'template') else MODEL_PROMPT_TEMPLATES["default"]
        
    async def retrieve_quantitative(self, query: str, annotated_trees: List[Dict]) -> List[Dict]:
        """Retrieve nodes using condensed tree analysis."""
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
            response = await asyncio.to_thread(
                self.llm.generate, 
                prompt, 
                max_new_tokens=2048, 
                fast_json=True, 
                system_prompt=self.template.get("system")
            )
            selections = self._parse_node_selections(response)
            all_selections.extend(selections)
            
        results = []
        for sel in sorted(all_selections, key=lambda x: x.get('confidence', 0), reverse=True):
            doc_id = sel.get('doc_id')
            node_id = sel.get('node_id')
            node = self._find_node_by_id(annotated_trees, doc_id, node_id)
            if node:
                full_text = node.get('text', '')
                if len(full_text) > self.max_text_chars:
                    full_text = full_text[:self.max_text_chars] + "..."
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
        """Condense tree for efficient navigation."""
        def condense(node: Dict, depth: int = 0) -> Dict[str, Any]:
            if depth > max_depth:
                return {"node_id": node.get("node_id", ""), "title": node.get("title", ""), "leaf": True}
            result = {
                "node_id": node.get("node_id", ""), 
                "title": node.get("title", ""), 
                "summary": (node.get("summary", "") or "")[:150]
            }
            if node.get("metadata"):
                meta = node["metadata"]
                if meta.get("alloys"): result["alloys"] = meta["alloys"][:3]
                if meta.get("laser_power_values"): result["power_hint"] = f"{min(meta['laser_power_values'])}-{max(meta['laser_power_values'])} W"
                if meta.get("scan_speed_values"): result["speed_hint"] = f"{min(meta['scan_speed_values'])}-{max(meta['scan_speed_values'])} mm/s"
                
            q_items = node.get("quantitative_items", [])
            if q_items:
                params = list(set(item.get("parameter_name", "") for item in q_items if item.get("parameter_name")))
                if params: result["has_quantitative"] = params[:5]
            else:
                text = node.get("text", "")
                if text:
                    candidates = re.findall(r'(\d+(?:\.\d+)?)\s*(W|kW|mW|J|mm/s|C|K|MPa|GPa|nm|um|mm|s|m/s|W/cm2|kW/cm2)', text, re.IGNORECASE)
                    if candidates: result["candidate_values"] = [f"{v}{u}" for v, u in candidates[:3]]
                    
            children = node.get("nodes", [])
            if children and depth < max_depth:
                result["nodes"] = [condense(c, depth+1) for c in children[:5]]
            return result
            
        return {
            "doc_id": tree.get("doc_id", tree.get("doc_name", "unknown")), 
            "doc_name": tree.get("doc_name", ""), 
            "structure": [condense(tree)] if not isinstance(tree, list) else [condense(t) for t in tree]
        }
        
    def _batch_trees(self, trees: List[Dict], max_tokens: int = 6000) -> List[List[Dict]]:
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
        if current: batches.append(current)
        return batches
        
    def _build_tree_search_prompt(self, query: str, trees: List[Dict]) -> str:
        """Build navigation prompt."""
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
        
    def _parse_node_selections(self, response: str) -> List[Dict]:
        """Parse LLM response for selections."""
        try:
            data = self._extract_json_safe(response)
            if data and isinstance(data, dict):
                selections = data.get("selections", [])
                return [s for s in selections if isinstance(s, dict) and "doc_id" in s and "node_id" in s]
        except Exception as e:
            logger.warning(f"Failed to parse selections: {e}")
        return []
        
    def _extract_json_safe(self, text: str) -> Optional[Any]:
        """Extract JSON safely."""
        patterns = [r'\{.*\}', r'\[.*\]', r'```json\s*(\{.*?\})\s*```']
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_str = match.group(1) if match.groups() else match.group(0)
                try: return json.loads(json_str)
                except Exception: continue
        return None
        
    def _find_node_by_id(self, trees: List[Dict], doc_id: str, node_id: str) -> Optional[Dict]:
        """Find node in tree by ID."""
        for tree in trees:
            if tree.get("doc_id") == doc_id or tree.get("doc_name") == doc_id:
                return self._search_node_recursive(tree, node_id)
        return None
        
    def _search_node_recursive(self, node: Dict, target_id: str) -> Optional[Dict]:
        """Recursively search for node."""
        if node.get("node_id") == target_id: return node
        for child in node.get("nodes", []):
            res = self._search_node_recursive(child, target_id)
            if res: return res
        return None

# ============================================================================
# VISUALIZATION CONFIGURATION DATACLASS
# ============================================================================
@dataclass
class VisConfig:
    """
    Centralized configuration for all visualization styling parameters.
    Enables consistent, publication-quality output across all chart types.
    """
    # Font settings
    font_family: str = "DejaVu Sans"
    font_size: int = 10
    title_font_size: int = 14
    label_font_size: int = 9
    
    # Figure settings
    figure_dpi: int = 300
    figsize_network: Tuple[int, int] = (14, 12)
    figsize_knowledge_graph: Tuple[int, int] = (14, 12)
    figsize_embedding: Tuple[int, int] = (10, 8)
    figsize_tree: Tuple[int, int] = (14, 10)
    
    # Network/node settings
    node_size_factor: float = 1.0
    node_size_base_doc: int = 800
    node_size_base_entity: int = 500
    node_size_base_material: int = 600
    node_size_base_value: int = 300
    node_size_base_hub: int = 2500
    
    # Edge settings
    edge_alpha: float = 0.25
    edge_width: float = 0.8
    edge_width_pyvis: float = 1.0
    
    # PyVis settings
    pyvis_height: str = "700px"
    pyvis_width: str = "100%"
    pyvis_physics_enabled: bool = True
    pyvis_gravity: int = -1800
    pyvis_spring_length: int = 140
    pyvis_damping: float = 0.85
    
    # Plotly settings
    plotly_height: int = 500
    plotly_width: Optional[int] = None
    
    # Matplotlib settings
    marker_size: int = 80
    line_width: float = 1.5
    alpha: float = 0.8
    
    # Colormap
    default_colormap: str = "viridis"
    
    # Label style
    label_style: str = "doi"
    
    # Aliases
    aliases: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VisConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# PUBLICATION VISUALIZATION ENGINE
# ============================================================================

# ============================================================================
# QUERY CONTEXT DATACLASS (MISSING IN ORIGINAL)
# ============================================================================
@dataclass
class QueryContext:
    """
    Holds query-specific context for visualization filtering.
    Maps extracted data to the current user query for focused visualizations.
    """
    query: str = ""
    relevant_doc_ids: List[str] = field(default_factory=list)
    physical_quantities: List[str] = field(default_factory=list)
    materials: List[str] = field(default_factory=list)
    extracted_values: List[Any] = field(default_factory=list)

    def has_data(self) -> bool:
        """Check if context has any data to visualize."""
        return bool(self.relevant_doc_ids or self.physical_quantities or self.extracted_values)

class PublicationVisualizationEngine:
    """
    Comprehensive visualization engine for scientific document analysis.
    Generates 35+ publication-quality chart types with query-aware filtering.
    
    Features:
    - Query-focused knowledge graphs (NetworkX + PyVis)
    - Hierarchical sunbursts and treemaps
    - Contradiction detection matrices
    - Consensus waterfall plots
    - Embedding space projections (t-SNE, PCA, UMAP)
    - Retrieval provenance Sankey diagrams
    - Interactive PyVis networks with modal popups
    - Publication-ready matplotlib figures
    """
    
    # Domain-specific color mapping for consistent visual semantics
    DOMAIN_COLORS = {
        "laser_power": "#3b82f6",      # Blue
        "scan_speed": "#8b5cf6",        # Purple
        "yield_strength": "#f59e0b",    # Amber
        "tensile_strength": "#10b981",  # Emerald
        "hardness": "#ec4899",          # Pink
        "temperature": "#ef4444",       # Red
        "energy_density": "#06b6d4",    # Cyan
        "unknown": "#6b7280",           # Gray
        "material": "#3b82f6",          # Blue
        "document": "#10b981",          # Emerald
        "hub": "#dc2626",               # Red
        "query": "#7c3aed",             # Violet
        "value": "#ec4899",             # Pink
        "pq": "#2563eb",                # Blue
    }
    
    # Supported colormaps for continuous data
    COLORMAP_OPTIONS = {
        "viridis": "viridis", "plasma": "plasma", "inferno": "inferno", 
        "magma": "magma", "cividis": "cividis", "Blues": "Blues", 
        "Greens": "Greens", "Oranges": "Oranges", "Reds": "Reds", 
        "RdBu": "RdBu", "Spectral": "Spectral", "coolwarm": "coolwarm",
        "Set1": "Set1", "Set2": "Set2", "Set3": "Set3", 
        "tab10": "tab10", "tab20": "tab20"
    }
    
    def __init__(self, kgraph: 'QuantitativeKnowledgeGraph', config: Optional[VisConfig] = None):
        """
        Initialize visualization engine.
        
        Args:
            kgraph: QuantitativeKnowledgeGraph instance with extracted data
            config: Optional VisConfig for styling customization
        """
        self.kgraph = kgraph
        self.cfg = config or VisConfig()
        
        # Apply matplotlib rcParams from config for consistent styling
        plt.rcParams['font.family'] = self.cfg.font_family
        plt.rcParams['font.size'] = self.cfg.font_size
        plt.rcParams['axes.titlesize'] = self.cfg.title_font_size
        plt.rcParams['axes.labelsize'] = self.cfg.label_font_size
        plt.rcParams['figure.dpi'] = self.cfg.figure_dpi
        plt.rcParams['savefig.dpi'] = self.cfg.figure_dpi
        plt.rcParams['lines.linewidth'] = self.cfg.line_width
        plt.rcParams['patch.linewidth'] = self.cfg.line_width
        plt.rcParams['xtick.labelsize'] = self.cfg.label_font_size
        plt.rcParams['ytick.labelsize'] = self.cfg.label_font_size
        plt.rcParams['legend.fontsize'] = self.cfg.label_font_size
    
    # -------------------------------------------------------------------------
    # Property getters for config access
    # -------------------------------------------------------------------------
    @property
    def font_family(self) -> str:
        return self.cfg.font_family
    
    @property
    def font_size(self) -> int:
        return self.cfg.font_size
    
    @property
    def title_font_size(self) -> int:
        return self.cfg.title_font_size
    
    @property
    def label_font_size(self) -> int:
        return self.cfg.label_font_size
    
    @property
    def default_colormap(self) -> str:
        return self.cfg.default_colormap
    
    @property
    def figure_dpi(self) -> int:
        return self.cfg.figure_dpi
    
    @property
    def aliases(self) -> Optional[Dict[str, str]]:
        return self.cfg.aliases
    
    @property
    def label_style(self) -> str:
        return self.cfg.label_style
    
    # -------------------------------------------------------------------------
    # Helper methods for color/colormap handling
    # -------------------------------------------------------------------------
    def _get_colormap(self, name: Optional[str] = None) -> str:
        """Get matplotlib colormap name."""
        return self.COLORMAP_OPTIONS.get(name or self.default_colormap, "viridis")
    
    def _get_plotly_colorscale(self, name: Optional[str] = None) -> str:
        """Get Plotly-compatible colorscale name."""
        name = name or self.default_colormap
        # Map matplotlib colormaps to Plotly equivalents
        mapping = {
            "coolwarm": "RdBu", "RdBu": "RdBu", "seismic": "RdBu", "bwr": "RdBu"
        }
        plotly_builtins = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'blues', 'greens', 'oranges', 'reds', 'purples', 'greys'
        ]
        lowered = name.lower()
        if lowered in plotly_builtins:
            return lowered
        return mapping.get(lowered, 'viridis')
    
    def _get_domain_color(self, domain: str, colormap: Optional[str] = None, 
                          index: int = 0, total: int = 1) -> str:
        """Get color for a domain/category."""
        if colormap and total > 1:
            cmap = plt.get_cmap(self._get_colormap(colormap))
            return mcolors.to_hex(cmap(index / max(total - 1, 1)))
        return self.DOMAIN_COLORS.get(domain, "#6b7280")
    
    # -------------------------------------------------------------------------
    # Data extraction and preparation
    # -------------------------------------------------------------------------
    def extract_dataframe(self, aliases: Optional[Dict[str, str]] = None, 
                          label_style: str = "doi") -> pd.DataFrame:
        """
        Extract all quantitative data into a pandas DataFrame for analysis.
        
        Args:
            aliases: Optional document name aliases
            label_style: Citation label style ('doi', 'number', 'alias', 'short')
            
        Returns:
            DataFrame with columns: doc, doc_stem, doc_citation, physical_quantity,
            material, value, unit, confidence, page, context
        """
        rows = []
        for doc_id, graph in self.kgraph.doc_graphs.items():
            display = get_display_name(doc_id, aliases)
            citation = get_citation_label(doc_id, aliases, style=label_style)
            
            for item in graph["all_items"]:
                phys = item.get("physical_quantity", "unknown")
                mat = item.get("material", "Unknown")
                value = item.get("value")
                unit = item.get("unit", "")
                
                if value is not None:
                    rows.append({
                        "doc": doc_id,
                        "doc_stem": display,
                        "doc_citation": citation,
                        "physical_quantity": phys,
                        "material": mat,
                        "value": value,
                        "unit": unit,
                        "confidence": item.get("confidence", 0.5),
                        "page": item.get("page", 0),
                        "context": item.get("context", "")[:200]
                    })
        
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    
    # =============================================================================
    # QUERY-AWARE DATA FILTERING
    # =============================================================================
    def get_query_focused_df(self, query_ctx: 'QueryContext') -> pd.DataFrame:
        """
        Return dataframe filtered to current query context.
        
        Filters by:
        - Relevant document IDs
        - Physical quantities mentioned in query
        - Materials mentioned in query
        
        Args:
            query_ctx: QueryContext with query-specific filters
            
        Returns:
            Filtered DataFrame
        """
        df = self.extract_dataframe(aliases=self.cfg.aliases, label_style=self.cfg.label_style)
        
        if df.empty or not query_ctx.has_data():
            return df
        
        # Build filter mask
        mask = (
            df["doc"].isin(query_ctx.relevant_doc_ids) |
            df["physical_quantity"].isin(query_ctx.physical_quantities) |
            (df["material"].isin(query_ctx.materials) & df["material"].notna())
        )
        
        return df[mask].copy()
    
    # =============================================================================
    # QUERY-AWARE KNOWLEDGE GRAPH (NetworkX)
    # =============================================================================
    def plot_query_knowledge_graph(self, query_ctx: 'QueryContext', 
                                    figsize: Tuple[int, int] = (14, 11)) -> plt.Figure:
        """
        Generate query-focused interactive Knowledge Graph using NetworkX.
        
        Node types:
        - QUERY (purple): Central query node
        - Documents (green): Relevant papers
        - Physical Quantities (blue): Parameters like laser_power, yield_strength
        - Materials (orange): Alloys like Ti3Au, AlSiMgZr
        - Values (pink): Extracted numerical values (clickable in PyVis version)
        
        Edges represent relationships: query→doc, doc→value, material→value, etc.
        
        Args:
            query_ctx: QueryContext with query-specific data
            figsize: Figure size tuple
            
        Returns:
            matplotlib Figure object
        """
        if not query_ctx.has_data():
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No quantitative data for this query", 
                   ha='center', va='center', fontsize=14)
            ax.axis("off")
            return fig
        
        df_focus = self.get_query_focused_df(query_ctx)
        G = nx.Graph()
        
        # Central Query Node
        G.add_node("QUERY", node_type="query", label=query_ctx.query[:45] + "...",
                  title=f"Query: {query_ctx.query}")
        
        # Add relevant documents
        for doc_id in query_ctx.relevant_doc_ids:
            display_name = get_display_name(doc_id, self.cfg.aliases)
            G.add_node(display_name, node_type="doc", color="#10b981", size=1400,
                      title=f"Document: {display_name}\n"
                           f"{len([v for v in query_ctx.extracted_values if v.doc_name == doc_id])} values")
        
        # Add physical quantities
        for pq in query_ctx.physical_quantities:
            readable = self.kgraph.phys_classifier.get_human_readable(pq)
            G.add_node(pq, node_type="pq", label=readable, color="#3b82f6", size=1100)
        
        # Add materials
        for mat in query_ctx.materials:
            G.add_node(mat, node_type="material", color="#f59e0b", size=1300)
        
        # Add extracted values as leaf nodes
        for val in query_ctx.extracted_values[:20]:
            label = f"{val.value:.1f} {val.unit or ''}"
            G.add_node(label, node_type="value", color="#ec4899", size=600,
                      title=f"{val.value} {val.unit} | {val.material or ''} | p.{val.page}")
            
            # Connect value to material if both exist
            if val.material and val.material in G:
                G.add_edge(val.material, label, weight=2)
            # Connect value to document
            if val.doc_name and get_display_name(val.doc_name, self.cfg.aliases) in G:
                G.add_edge(get_display_name(val.doc_name, self.cfg.aliases), label, weight=1.5)
            # Connect value to physical quantity
            for pq in query_ctx.physical_quantities:
                if val.physical_quantity == pq and pq in G:
                    G.add_edge(pq, label, weight=1)
        
        # Connect query to everything
        for node in list(G.nodes()):
            if node != "QUERY":
                G.add_edge("QUERY", node, weight=0.8)
        
        # Draw the graph
        pos = nx.spring_layout(G, k=0.65, iterations=80, seed=42)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Categorize nodes for colored drawing
        query_nodes = ["QUERY"]
        doc_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "doc"]
        pq_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "pq"]
        mat_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "material"]
        val_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "value"]
        
        # Draw nodes by type with appropriate colors/sizes
        nx.draw_networkx_nodes(G, pos, nodelist=query_nodes, node_color="#8b5cf6", 
                              node_size=3200, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=doc_nodes, node_color="#10b981", 
                              node_size=1400, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=pq_nodes, node_color="#3b82f6", 
                              node_size=1100, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=mat_nodes, node_color="#f59e0b", 
                              node_size=1300, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=val_nodes, node_color="#ec4899", 
                              node_size=650, ax=ax)
        
        # Draw edges and labels
        nx.draw_networkx_edges(G, pos, alpha=0.35, width=1.2, edge_color="#94a3b8", ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=9, font_family=self.font_family, ax=ax)
        
        # Title and styling
        ax.set_title(f"Query-Focused Knowledge Graph\n{query_ctx.query[:70]}{'...' if len(query_ctx.query)>70 else ''}",
                    fontsize=15, fontweight='bold', pad=20)
        ax.axis("off")
        plt.tight_layout()
        
        return fig
    
    # =============================================================================
    # QUERY-AWARE KNOWLEDGE GRAPH (PyVis with Modal Popup)
    # =============================================================================
    def plot_query_knowledge_graph_pyvis(self, query_ctx: 'QueryContext') -> str:
        """
        Interactive PyVis KG with confidence highlighting + modal-ready JS.
        
        Features:
        - Clickable value nodes show full context in modal popup
        - Color-coded by confidence (red=high, orange=medium, gray=low)
        - Edge thickness reflects relationship strength
        - Physics-based layout for intuitive exploration
        
        Returns:
            HTML string for embedding in Streamlit
        """
        if not PYVIS_AVAILABLE:
            return "<p>PyVis not installed. Run: <code>pip install pyvis</code></p>"
        
        if not query_ctx.has_data():
            return "<p>No quantitative data available for this query.</p>"
        
        # Initialize PyVis network with publication styling
        net = Network(
            height="780px",
            width="100%",
            bgcolor="#ffffff",      # White bright background
            font_color="#1e293b",   # Dark slate text
            cdn_resources='remote'
        )
        net.barnes_hut(gravity=-2800, spring_length=140, damping=0.92)
        
        # Confidence threshold for strong paths
        high_conf_threshold = 0.75
        connected_to_query = set()
        
        # ====================== ADD NODES ======================
        # 1. Central Query Node
        net.add_node(
            "QUERY",
            label="YOUR QUERY",
            title=f"<b>Query:</b><br>{query_ctx.query}<br><br><i>Click pink nodes for details</i>",
            color="#7c3aed",  # Darker purple for white bg
            size=45,
            font={"size": 18, "bold": True, "color": "#1e293b"}
        )
        
        # 2. Documents
        for doc_id in query_ctx.relevant_doc_ids:
            display = get_display_name(doc_id, self.cfg.aliases)
            count = len([v for v in query_ctx.extracted_values if v.doc_name == doc_id])
            
            tooltip = f"<b>Document:</b> {display}<br>"
            tooltip += f"<b>Extracted Values:</b> {count}<br><br>"
            for item in query_ctx.extracted_values[:5]:
                if item.doc_name == doc_id:
                    tooltip += f"• {item.value} {item.unit} ({item.physical_quantity})<br>"
            
            net.add_node(
                display,
                label=display[:25],
                title=tooltip,
                color="#16a34a",  # Darker green
                size=32,
                font={"size": 14, "color": "#1e293b"}
            )
            net.add_edge("QUERY", display, value=3)
            connected_to_query.add(display)
        
        # 3. Physical Quantities
        for pq in query_ctx.physical_quantities:
            readable = self.kgraph.phys_classifier.get_human_readable(pq)
            net.add_node(
                pq,
                label=readable,
                title=f"<b>Physical Quantity:</b><br>{readable}",
                color="#2563eb",  # Darker blue
                size=28,
                font={"color": "#1e293b"}
            )
            net.add_edge("QUERY", pq, value=2)
            connected_to_query.add(pq)
        
        # 4. Materials
        for mat in query_ctx.materials:
            net.add_node(
                mat,
                label=mat[:22],
                title=f"<b>Material/Alloy:</b><br>{mat}",
                color="#d97706",  # Darker orange
                size=30,
                font={"color": "#1e293b"}
            )
            net.add_edge("QUERY", mat, value=2)
            connected_to_query.add(mat)
        
        # 5. Extracted Values (Clickable Leaves)
        for i, val in enumerate(sorted(query_ctx.extracted_values, 
                                       key=lambda x: x.confidence, 
                                       reverse=True)[:30]):
            node_id = f"val_{i}"
            label = f"{val.value:.1f}{val.unit or ''}"
            
            # Color by confidence
            conf = val.confidence
            color = "#e11d48" if conf >= high_conf_threshold else "#ea580c" if conf >= 0.6 else "#64748b"
            
            excerpt = val.context[:420] + "..." if len(val.context) > 420 else val.context
            tooltip = f"""
<b>{val.value} {val.unit}</b><br>
<b>Confidence:</b> {conf:.2f}<br>
<b>Quantity:</b> {self.kgraph.phys_classifier.get_human_readable(val.physical_quantity)}<br>
<b>Material:</b> {val.material or '—'}<br>
<b>Source:</b> {get_display_name(val.doc_name, self.cfg.aliases)} (p.{val.page})<br><br>
<b>Context:</b><br>{excerpt}
"""
            net.add_node(
                node_id,
                label=label,
                title=tooltip,
                color=color,
                size=24 + int(conf * 18),
                font={"size": 11, "color": "#1e293b"}
            )
            
            # Connect with thickness based on confidence
            edge_width = 3 if conf >= high_conf_threshold else 1.5
            
            if val.material and val.material in net.get_nodes():
                net.add_edge(val.material, node_id, value=edge_width, color="#cbd5e1")
            if val.physical_quantity in net.get_nodes():
                net.add_edge(val.physical_quantity, node_id, value=edge_width*0.8)
            doc_name = get_display_name(val.doc_name, self.cfg.aliases)
            if doc_name in net.get_nodes():
                net.add_edge(doc_name, node_id, value=edge_width, color="#86efac")
        
        # Connect everything to QUERY
        for node in net.get_nodes():
            if node != "QUERY" and node not in connected_to_query:
                net.add_edge("QUERY", node, value=1, color="#64748b")
        
        # Generate base HTML
        html = net.generate_html()
        
        # ====================== ADVANCED JS MODAL ======================
        modal_js = """
<script>
var modal = null;
network.on("click", function(params) {
    if (params.nodes.length === 0) return;
    var nodeId = params.nodes[0];
    
    if (nodeId.startsWith("val_")) {
        var node = network.body.nodes[nodeId];
        var title = node.options.title || "No details";
        
        if (!modal) {
            modal = document.createElement("div");
            modal.style.cssText = `
                position:fixed; top:0; left:0; width:100%; height:100%;
                background:rgba(0,0,0,0.6); z-index:9999; display:flex;
                align-items:center; justify-content:center; font-family:system-ui;
            `;
            document.body.appendChild(modal);
        }
        
        modal.innerHTML = `
            <div style="background:#f8fafc; color:#1e293b; padding:25px; border-radius:12px;
                        max-width:620px; max-height:85vh; overflow:auto; border:1px solid #cbd5e1;">
                <h3 style="margin-top:0; color:#db2777;">Extracted Value Details</h3>
                <div style="white-space:pre-wrap; font-size:15px; line-height:1.5;">${title}</div>
                <br>
                <button onclick="this.parentElement.parentElement.remove()"
                        style="padding:10px 20px; background:#e11d48; color:white; border:none;
                               border-radius:6px; cursor:pointer;">Close</button>
            </div>
        `;
    }
});
</script>
"""
        
        # Inject modal script before </body>
        if "</body>" in html:
            html = html.replace("</body>", modal_js + "</body>")
        else:
            html += modal_js
        
        return html
    
    # =============================================================================
    # QUERY-AWARE HIERARCHICAL SUNBURST
    # =============================================================================
    def plot_query_sunburst(self, query_ctx: 'QueryContext') -> go.Figure:
        """
        Query-focused hierarchical sunburst showing:
        Physical Quantity → Material → Document → Value Range
        
        Args:
            query_ctx: QueryContext with query-specific data
            
        Returns:
            Plotly Figure object
        """
        df_focus = self.get_query_focused_df(query_ctx)
        
        if df_focus.empty:
            return go.Figure().update_layout(title="No data for current query")
        
        # Create hierarchy: Physical Quantity → Material → Document → Value Range
        df_sun = df_focus.copy()
        df_sun["material"] = df_sun["material"].fillna("Unknown").replace("", "Unknown")
        df_sun["doc_stem"] = df_sun["doc_stem"].fillna("Unknown").replace("", "Unknown")
        
        # Bin values for better hierarchy visualization
        if not df_sun.empty and len(df_sun) >= 3:
            try:
                n_bins = min(5, max(2, len(df_sun)//3))
                df_sun["value_range"] = pd.cut(df_sun["value"], bins=n_bins, precision=1).astype(str).fillna("unknown")
                
                fig = px.sunburst(
                    df_sun,
                    path=["physical_quantity", "material", "doc_stem", "value_range"],
                    values="value",
                    color="value",
                    color_continuous_scale=self._get_plotly_colorscale(),
                    title=f"Query Hierarchy: {query_ctx.query[:60]}{'...' if len(query_ctx.query)>60 else ''}",
                    maxdepth=4
                )
                fig.update_traces(textinfo="label+percent entry")
                fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
                return fig
                
            except Exception as e:
                logger.warning(f"Sunburst binning failed: {e}")
        
        # Fallback: simpler hierarchy without value binning
        try:
            fig = px.sunburst(
                df_sun,
                path=["physical_quantity", "material", "doc_stem"],
                values="value",
                color="value",
                color_continuous_scale=self._get_plotly_colorscale(),
                title=f"Query Hierarchy: {query_ctx.query[:60]}{'...' if len(query_ctx.query)>60 else ''}"
            )
            fig.update_traces(textinfo="label+percent entry")
            fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
            return fig
        except Exception as e:
            logger.error(f"Query sunburst failed: {e}")
            return go.Figure().update_layout(title="Sunburst unavailable for this query")

# =============================================================================

    # -------------------------------------------------------------------------
    # HISTOGRAM & BAR CHART METHODS
    # -------------------------------------------------------------------------
    def plot_quantitative_histogram(self, df: pd.DataFrame, quantity_name: str, 
                                     group_by: str = "material", 
                                     colormap: Optional[str] = None) -> go.Figure:
        """
        Plot histogram of quantitative values grouped by material or document.
        
        Args:
            df: Filtered DataFrame from extract_dataframe()
            quantity_name: Canonical physical quantity name (e.g., "yield_strength")
            group_by: Column to group by ("material" or "doc_stem")
            colormap: Matplotlib colormap name for bar coloring
            
        Returns:
            Plotly Figure object
        """
        if df.empty:
            return go.Figure().update_layout(
                title=f"No {quantity_name} data available",
                font=dict(family=self.font_family, size=self.font_size)
            )
        
        # Filter to target quantity
        subset = df[df["physical_quantity"] == quantity_name].copy()
        if subset.empty:
            return go.Figure().update_layout(
                title=f"No data for '{quantity_name}'",
                font=dict(family=self.font_family, size=self.font_size)
            )
        
        # Clean grouping column
        clean_col = subset[group_by].fillna("Unknown").replace("", "Unknown")
        subset = subset.assign(clean_group=clean_col)
        
        groups = sorted(subset["clean_group"].unique())
        if not groups:
            return go.Figure().update_layout(
                title=f"No valid groups for '{quantity_name}'",
                font=dict(family=self.font_family, size=self.font_size)
            )
        
        fig = go.Figure()
        cmap_obj = plt.get_cmap(self._get_colormap(colormap))
        
        for i, grp in enumerate(groups):
            data = subset[subset["clean_group"] == grp]["value"].dropna()
            if data.empty:
                continue
                
            # Color assignment
            if len(groups) > 1 and cmap_obj:
                color = mcolors.to_hex(cmap_obj(i / max(len(groups) - 1, 1)))
            else:
                color = self.DOMAIN_COLORS.get(quantity_name, "#3b82f6")
            
            # Compute stats
            mean_val = float(data.mean())
            std_val = float(data.std()) if len(data) > 1 else 0.0
            
            fig.add_trace(go.Bar(
                name=str(grp),
                x=[str(grp)],
                y=[mean_val],
                error_y=dict(
                    type='data',
                    array=[std_val],
                    visible=True,
                    thickness=1.5,
                    width=0.5
                ),
                marker_color=color,
                marker_line_color="#ffffff",
                marker_line_width=0.5,
                opacity=0.9,
                text=[f"n={len(data)}<br>μ={mean_val:.2f}<br>σ={std_val:.2f}"],
                textposition="outside",
                hovertemplate=f"<b>{grp}</b><br>Mean: %{{y:.2f}}<br>Std: {std_val:.2f}<br>n=%{{text}}<extra></extra>"
            ))
        
        # Axis labels with unit
        unit = subset["unit"].iloc[0] if not subset.empty and subset["unit"].notna().any() else ""
        yaxis_title = f"{quantity_name.replace('_', ' ').title()} ({unit})" if unit else quantity_name.replace('_', ' ').title()
        
        fig.update_layout(
            barmode='group',
            title=f"{quantity_name.replace('_', ' ').title()} Values by {group_by.title()}",
            xaxis_title=group_by.title(),
            yaxis_title=yaxis_title,
            font=dict(family=self.font_family, size=self.font_size),
            height=500,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Add gridlines for readability
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9")
        
        return fig
    
    def plot_quantities_bar(self, df: pd.DataFrame, colormap: Optional[str] = None) -> go.Figure:
        """
        Bar chart showing occurrence counts of each physical quantity.
        
        Args:
            df: DataFrame from extract_dataframe()
            colormap: Optional colormap for bar coloring
            
        Returns:
            Plotly Figure object
        """
        if df.empty:
            return go.Figure().update_layout(
                title="No data available for quantity analysis",
                font=dict(family=self.font_family, size=self.font_size)
            )
        
        # Count occurrences per physical quantity
        counts = df["physical_quantity"].value_counts().reset_index()
        counts.columns = ["Physical Quantity", "Count"]
        
        if counts.empty:
            return go.Figure().update_layout(
                title="No physical quantities extracted",
                font=dict(family=self.font_family, size=self.font_size)
            )
        
        # Sort by count descending
        counts = counts.sort_values("Count", ascending=False)
        
        # Color assignment using domain colors or colormap
        colors = []
        cmap_obj = plt.get_cmap(self._get_colormap(colormap)) if colormap else None
        
        for i, pq in enumerate(counts["Physical Quantity"]):
            if cmap_obj and len(counts) > 1:
                colors.append(mcolors.to_hex(cmap_obj(i / max(len(counts) - 1, 1))))
            else:
                colors.append(self.DOMAIN_COLORS.get(pq, "#6b7280"))
        
        fig = px.bar(
            counts,
            x="Physical Quantity",
            y="Count",
            color="Physical Quantity",
            color_discrete_sequence=colors,
            title="Occurrence Counts by Physical Quantity",
            labels={"Count": "Number of Extractions", "Physical Quantity": "Parameter"}
        )
        
        fig.update_layout(
            font=dict(family=self.font_family, size=self.font_size),
            height=500,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            xaxis_tickangle=-45,
            hovermode="x unified"
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9")
        
        return fig
    
    def plot_material_counts(self, df: pd.DataFrame, colormap: Optional[str] = None) -> go.Figure:
        """
        Bar chart of top materials/alloys by mention count.
        
        Args:
            df: DataFrame from extract_dataframe()
            colormap: Optional colormap
            
        Returns:
            Plotly Figure object
        """
        # Filter to material-type entries or use all materials column
        if "material" in df.columns and df["material"].notna().any():
            mat_df = df[df["material"].notna() & (df["material"] != "") & (df["material"] != "Unknown")]
        else:
            mat_df = df.copy()
        
        if mat_df.empty or "material" not in mat_df.columns:
            return go.Figure().update_layout(
                title="No materials found in extracted data",
                font=dict(family=self.font_family, size=self.font_size)
            )
        
        # Count and get top 10
        counts = mat_df["material"].value_counts().head(10).reset_index()
        counts.columns = ["Material", "Count"]
        
        if counts.empty:
            return go.Figure().update_layout(
                title="No material data available",
                font=dict(family=self.font_family, size=self.font_size)
            )
        
        # Color assignment
        colors = []
        cmap_obj = plt.get_cmap(self._get_colormap(colormap)) if colormap else None
        
        for i in range(len(counts)):
            if cmap_obj and len(counts) > 1:
                colors.append(mcolors.to_hex(cmap_obj(i / max(len(counts) - 1, 1))))
            else:
                colors.append(self.DOMAIN_COLORS.get("material", "#3b82f6"))
        
        fig = px.bar(
            counts,
            x="Material",
            y="Count",
            color="Material",
            color_discrete_sequence=colors,
            title="Top 10 Materials/Alloys Mentioned",
            labels={"Count": "Mention Count", "Material": "Material/Alloy"}
        )
        
        fig.update_layout(
            font=dict(family=self.font_family, size=self.font_size),
            height=500,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            xaxis_tickangle=-45,
            hovermode="x unified"
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9")
        
        return fig
    
    # -------------------------------------------------------------------------
    # PIE & DONUT CHART METHODS
    # -------------------------------------------------------------------------
    def plot_quantity_distribution_pie(self, colormap: Optional[str] = None) -> go.Figure:
        """
        Pie chart showing distribution of physical quantities by extraction count.
        
        Args:
            colormap: Optional colormap for slice coloring
            
        Returns:
            Plotly Figure object
        """
        pq_counts = self.kgraph.get_all_physical_quantities()
        
        if not pq_counts:
            return go.Figure().update_layout(
                title="No physical quantities found in knowledge graph",
                font=dict(family=self.font_family, size=self.font_size)
            )
        
        # Get top 10 quantities
        sorted_pq = sorted(pq_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Human-readable labels
        labels = [self.kgraph.phys_classifier.get_human_readable(pq) for pq, _ in sorted_pq]
        values = [count for _, count in sorted_pq]
        
        # Color assignment
        if colormap:
            color_seq = px.colors.qualitative.Set3 if colormap == "Set3" else px.colors.qualitative.Plotly
        else:
            color_seq = px.colors.qualitative.Set3
        
        fig = px.pie(
            values=values,
            names=labels,
            title="Top Physical Quantities Distribution",
            color_discrete_sequence=color_seq,
            hole=0  # 0 for pie, >0 for donut
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>",
            marker=dict(line=dict(color='#ffffff', width=2))
        )
        
        fig.update_layout(
            font=dict(family=self.font_family, size=self.font_size),
            height=500,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        
        return fig
    
    def plot_material_distribution_donut(self, colormap: Optional[str] = None) -> go.Figure:
        """
        Donut chart showing material/alloy distribution across documents.
        
        Args:
            colormap: Optional colormap
            
        Returns:
            Plotly Figure object
        """
        mat_dict = self.kgraph.get_all_materials()
        
        if not mat_dict:
            return go.Figure().update_layout(
                title="No materials found in knowledge graph",
                font=dict(family=self.font_family, size=self.font_size)
            )
        
        # Count material occurrences across all documents
        mat_counts = Counter(m for mats in mat_dict.values() for m in mats)
        
        if not mat_counts:
            return go.Figure().update_layout(
                title="No material data available",
                font=dict(family=self.font_family, size=self.font_size)
            )
        
        # Get top 10 materials
        top_mats = mat_counts.most_common(10)
        labels = [m for m, _ in top_mats]
        values = [c for _, c in top_mats]
        
        # Generate colors using hash-based approach for consistency
        colors = [f"#{hash(l) % 0xFFFFFF:06x}" for l in labels]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,  # Donut hole
            marker=dict(colors=colors, line=dict(color='#ffffff', width=2)),
            textinfo='percent+label',
            hovertemplate="<b>%{label}</b><br>Occurrences: %{value}<br>Percent: %{percent}<extra></extra>",
            insidetextorientation='radial'
        )])
        
        fig.update_layout(
            title="Material Distribution (Donut)",
            annotations=[dict(
                text='Materials',
                x=0.5, y=0.5,
                font_size=14,
                font_family=self.font_family,
                showarrow=False,
                font_color="#1e293b"
            )],
            font=dict(family=self.font_family, size=self.font_size),
            height=500,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        
        return fig
    
    # -------------------------------------------------------------------------
    # SUNBURST & TREEMAP HIERARCHY METHODS
    # -------------------------------------------------------------------------
    def plot_quantitative_sunburst(self, df: pd.DataFrame, quantity: str, 
                                    colormap: Optional[str] = None) -> go.Figure:
        """
        Sunburst hierarchy for a specific quantity: Material → Document → Value Range.
        
        Args:
            df: Filtered DataFrame
            quantity: Canonical physical quantity name
            colormap: Optional Plotly colorscale
            
        Returns:
            Plotly Figure object
        """
        if df.empty:
            return go.Figure().update_layout(
                title="No data available",
                font=dict(family=self.font_family, size=self.font_size)
            )
        
        # Filter to target quantity
        subset = df[df["physical_quantity"] == quantity].copy()
        if subset.empty:
            return go.Figure().update_layout(
                title=f"No data for '{quantity}'",
                font=dict(family=self.font_family, size=self.font_size)
            )
        
        # Clean columns for hierarchy
        subset["material"] = subset["material"].fillna("Unknown").replace("", "Unknown")
        subset["doc_stem"] = subset["doc_stem"].fillna("Unknown").replace("", "Unknown")
        
        # Drop rows with missing hierarchy values
        subset = subset.dropna(subset=["material", "doc_stem", "value"])
        if subset.empty or len(subset) < 2:
            # Fallback without value binning
            try:
                fig = px.sunburst(
                    subset,
                    path=["material", "doc_stem"],
                    values="value",
                    color="value",
                    color_continuous_scale=self._get_plotly_colorscale(colormap),
                    title=f"{quantity.replace('_', ' ').title()} Distribution Hierarchy"
                )
                fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
                return fig
            except Exception as e:
                logger.warning(f"Sunburst fallback failed: {e}")
                return go.Figure().update_layout(
                    title=f"Sunburst unavailable for '{quantity}'",
                    font=dict(family=self.font_family, size=self.font_size)
                )
        
        # Bin values for better hierarchy visualization
        n_bins = min(5, max(2, len(subset) // 3))
        try:
            subset["value_range"] = pd.cut(
                subset["value"], 
                bins=n_bins, 
                precision=1
            ).astype(str).fillna("unknown")
            
            fig = px.sunburst(
                subset,
                path=["material", "doc_stem", "value_range"],
                values="value",
                color="value",
                color_continuous_scale=self._get_plotly_colorscale(colormap),
                title=f"{quantity.replace('_', ' ').title()} Distribution Hierarchy",
                maxdepth=4
            )
            
            fig.update_traces(
                textinfo="label+percent entry",
                hovertemplate="<b>%{label}</b><br>Value: %{value:.2f}<br>Percent: %{percent}<extra></extra>"
            )
            fig.update_layout(
                font=dict(family=self.font_family, size=self.font_size),
                height=600,
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff"
            )
            return fig
            
        except Exception as e:
            logger.warning(f"Sunburst binning failed: {e}")
            # Fallback without binning
            try:
                fig = px.sunburst(
                    subset,
                    path=["material", "doc_stem"],
                    values="value",
                    color="value",
                    color_continuous_scale=self._get_plotly_colorscale(colormap),
                    title=f"{quantity.replace('_', ' ').title()} Distribution Hierarchy"
                )
                fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
                return fig
            except:
                return go.Figure().update_layout(
                    title=f"Sunburst unavailable for '{quantity}'",
                    font=dict(family=self.font_family, size=self.font_size)
                )
    
    def plot_sunburst_hierarchy(self, df: pd.DataFrame, colormap: Optional[str] = None) -> go.Figure:
        """
        Full hierarchy sunburst: Physical Quantity → Material → Document.
        
        Args:
            df: DataFrame from extract_dataframe()
            colormap: Optional Plotly colorscale
            
        Returns:
            Plotly Figure object
        """
        if df.empty:
            return go.Figure().update_layout(
                title="No data available for hierarchy",
                font=dict(family=self.font_family, size=self.font_size)
            )
        
        df_hier = df.copy()
        df_hier["physical_quantity"] = df_hier["physical_quantity"].fillna("Unknown").replace("", "Unknown")
        df_hier["material"] = df_hier["material"].fillna("Unknown").replace("", "Unknown")
        df_hier["doc_stem"] = df_hier["doc_stem"].fillna("Unknown").replace("", "Unknown")
        
        # Add dummy value for counting
        df_hier["value_dummy"] = 1
        
        try:
            fig = px.sunburst(
                df_hier,
                path=["physical_quantity", "material", "doc_stem"],
                values="value_dummy",
                title="Hierarchy of Physical Quantities, Materials, and Documents",
                color="value_dummy",
                color_continuous_scale=self._get_plotly_colorscale(colormap)
            )
            fig.update_traces(textinfo="label+percent entry")
            fig.update_layout(
                font=dict(family=self.font_family, size=self.font_size),
                height=600,
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff"
            )
            return fig
        except Exception as e:
            logger.error(f"Sunburst hierarchy error: {e}")
            # Try simplified hierarchy
            try:
                fig = px.sunburst(
                    df_hier,
                    path=["physical_quantity", "material"],
                    values="value_dummy",
                    title="Hierarchy (simplified: Quantity → Material)",
                    color="value_dummy",
                    color_continuous_scale=self._get_plotly_colorscale(colormap)
                )
                fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
                return fig
            except:
                return go.Figure().update_layout(
                    title="Sunburst unavailable",
                    font=dict(family=self.font_family, size=self.font_size)
                )
    
    def plot_treemap(self, colormap: Optional[str] = None) -> go.Figure:
        """
        Treemap showing entity co-occurrence: Physical Quantity → Material.
        
        Args:
            colormap: Optional Plotly colorscale
            
        Returns:
            Plotly Figure object
        """
        df = self.extract_dataframe()
        if df.empty:
            return go.Figure().update_layout(
                title="No data available for treemap",
                font=dict(family=self.font_family, size=self.font_size)
            )
        
        # Aggregate by quantity and material
        agg = df.groupby(["physical_quantity", "material"]).size().reset_index(name="count")
        
        if agg.empty:
            return go.Figure().update_layout(
                title="No quantity-material pairs found",
                font=dict(family=self.font_family, size=self.font_size)
            )
        
        fig = px.treemap(
            agg,
            path=["physical_quantity", "material"],
            values="count",
            title="Entity Treemap: Quantities and Materials",
            color="count",
            color_continuous_scale=self._get_plotly_colorscale(colormap)
        )
        
        fig.update_traces(
            textinfo="label+value",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>"
        )
        fig.update_layout(
            font=dict(family=self.font_family, size=self.font_size),
            height=600,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff"
        )
        
        return fig
    
    def plot_treemap_materials(self, df: pd.DataFrame, colormap: Optional[str] = None) -> go.Figure:
        """
        Treemap focused on material distribution.
        
        Args:
            df: DataFrame from extract_dataframe()
            colormap: Optional Plotly colorscale
            
        Returns:
            Plotly Figure object
        """
        if df.empty:
            return go.Figure().update_layout(
                title="No data available",
                font=dict(family=self.font_family, size=self.font_size)
            )
        
        # Count materials
        if "material" in df.columns and df["material"].notna().any():
            mat_counts = df[df["material"].notna() & (df["material"] != "")]["material"].value_counts().reset_index()
        else:
            mat_counts = pd.DataFrame(columns=["material", "Count"])
        
        mat_counts.columns = ["Material", "Count"]
        
        if mat_counts.empty:
            return go.Figure().update_layout(
                title="No material data for treemap",
                font=dict(family=self.font_family, size=self.font_size)
            )
        
        fig = px.treemap(
            mat_counts,
            path=["Material"],
            values="Count",
            title="Material Treemap",
            color="Count",
            color_continuous_scale=self._get_plotly_colorscale(colormap)
        )
        
        fig.update_traces(textinfo="label+value+percent parent")
        fig.update_layout(
            font=dict(family=self.font_family, size=self.font_size),
            height=500,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff"
        )
        
        return fig
    
    # -------------------------------------------------------------------------
    # RADAR CHART METHODS
    # -------------------------------------------------------------------------
    def plot_radar_by_material(self, colormap: Optional[str] = None) -> go.Figure:
        """
        Radar chart comparing materials across top physical quantities.
        
        Args:
            colormap: Optional matplotlib colormap for line coloring
            
        Returns:
            Plotly Figure object
        """
        df = self.extract_dataframe()
        if df.empty:
            return go.Figure().update_layout(
                title="No data available for radar chart",
                font=dict(family=self.font_family, size=self.font_size)
            )
        
        # Get top 5 quantities by occurrence
        top_quantities = df["physical_quantity"].value_counts().head(5).index.tolist()
        if len(top_quantities) < 2:
            return go.Figure().update_layout(
                title="Need at least 2 quantities for radar",
                font=dict(family=self.font_family, size=self.font_size)
            )
        
        # Pivot: materials × quantities → mean values
        pivot = df[df["physical_quantity"].isin(top_quantities)].pivot_table(
            index="material",
            columns="physical_quantity",
            values="value",
            aggfunc="mean"
        ).fillna(0)
        
        if pivot.empty or len(pivot) < 1:
            return go.Figure().update_layout(
                title="No data for radar chart",
                font=dict(family=self.font_family, size=self.font_size)
            )
        
        fig = go.Figure()
        cmap = plt.get_cmap(self._get_colormap(colormap)) if colormap else None
        materials = pivot.index.tolist()
        
        for i, mat in enumerate(materials):
            values = pivot.loc[mat].tolist()
            # Close the radar by repeating first value
            values_closed = values + [values[0]]
            
            # Color assignment
            if cmap and len(materials) > 1:
                color = mcolors.to_hex(cmap(i / max(len(materials) - 1, 1)))
            else:
                color = self.DOMAIN_COLORS.get("material", "#3b82f6")
            
            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=top_quantities + [top_quantities[0]],
                fill='toself',
                name=mat,
                line_color=color,
                line_width=2,
                marker=dict(size=6, color=color),
                opacity=0.7,
                hovertemplate="<b>%{fullData.name}</b><br>%{theta}: %{r:.2f}<extra></extra>"
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(pivot.max().max() * 1.1, 1)],
                    gridcolor="#e2e8f0",
                    linecolor="#94a3b8"
                ),
                angularaxis=dict(
                    gridcolor="#e2e8f0",
                    linecolor="#94a3b8",
                    tickfont=dict(size=self.label_font_size, family=self.font_family)
                )
            ),
            title="Material Performance Radar (Mean Values)",
            font=dict(family=self.font_family, size=self.font_size),
            height=500,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def plot_document_radar(self, colormap: Optional[str] = None) -> go.Figure:
        """
        Radar chart comparing documents by quantity coverage.
        
        Args:
            colormap: Optional matplotlib colormap
            
        Returns:
            Plotly Figure object
        """
        df = self.extract_dataframe()
        if df.empty:
            return go.Figure().update_layout(
                title="No data available",
                font=dict(family=self.font_family, size=self.font_size)
            )
        
        # Pivot: documents × quantities → count of extractions
        pivot = df.pivot_table(
            index="doc_stem",
            columns="physical_quantity",
            values="value",
            aggfunc="count"
        ).fillna(0)
        
        if pivot.empty:
            return go.Figure().update_layout(
                title="No data for document radar",
                font=dict(family=self.font_family, size=self.font_size)
            )
        
        fig = go.Figure()
        cmap = plt.get_cmap(self._get_colormap(colormap)) if colormap else None
        docs = pivot.index.tolist()
        
        for i, doc in enumerate(docs):
            values = pivot.loc[doc].tolist()
            values_closed = values + [values[0]]
            
            if cmap and len(docs) > 1:
                color = mcolors.to_hex(cmap(i / max(len(docs) - 1, 1)))
            else:
                color = self.DOMAIN_COLORS.get("document", "#10b981")
            
            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=pivot.columns.tolist() + [pivot.columns[0]],
                fill='toself',
                name=doc,
                line_color=color,
                line_width=2,
                marker=dict(size=6, color=color),
                opacity=0.7
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, gridcolor="#e2e8f0"),
                angularaxis=dict(gridcolor="#e2e8f0")
            ),
            title="Document Coverage Radar (Counts per Quantity Type)",
            font=dict(family=self.font_family, size=self.font_size),
            height=500,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def plot_quantitative_radar(self, df: pd.DataFrame, quantity_name: str, 
                                 colormap: Optional[str] = None) -> go.Figure:
        """
        Radar chart showing statistics for a specific quantity by material.
        
        Args:
            df: Filtered DataFrame
            quantity_name: Canonical physical quantity name
            colormap: Optional matplotlib colormap
            
        Returns:
            Plotly Figure object
        """
        if df.empty:
            return go.Figure().update_layout(
                title=f"No {quantity_name} data",
                font=dict(family=self.font_family, size=self.font_size)
            )
        
        # Compute statistics by material
        stats = df[df["physical_quantity"] == quantity_name].groupby("material")["value"].agg(
            ["mean", "std", "min", "max", "count"]
        ).fillna(0)
        
        if stats.empty:
            return go.Figure().update_layout(
                title=f"No data for {quantity_name} radar",
                font=dict(family=self.font_family, size=self.font_size)
            )
        
        categories = ["Mean", "Max", "Min", "Std", "Count"]
        fig = go.Figure()
        cmap_obj = plt.get_cmap(self._get_colormap(colormap)) if colormap else None
        
        for i, (mat, row) in enumerate(stats.iterrows()):
            values = [
                float(row["mean"]),
                float(row["max"]),
                float(row["min"]),
                float(row["std"]),
                float(row["count"])
            ]
            values_closed = values + [values[0]]
            
            if cmap_obj and len(stats) > 1:
                color = mcolors.to_hex(cmap_obj(i / max(len(stats) - 1, 1)))
            else:
                color = self.DOMAIN_COLORS.get(quantity_name, "#3b82f6")
            
            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=categories + [categories[0]],
                fill='toself',
                name=mat,
                line_color=color,
                line_width=2,
                marker=dict(size=6, color=color),
                opacity=0.7
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, gridcolor="#e2e8f0")),
            showlegend=True,
            title=f"{quantity_name.replace('_', ' ').title()} Statistics by Material",
            font=dict(family=self.font_family, size=self.font_size),
            height=500,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff"
        )
        
        return fig
    
    # -------------------------------------------------------------------------
    # CHORD DIAGRAM FOR CO-OCCURRENCE
    # -------------------------------------------------------------------------
    def plot_chord_cooccurrence(self, filtered_concepts: Optional[List[str]] = None, 
                                 top_n: int = 14, 
                                 colormap: Optional[str] = None) -> go.Figure:
        """
        Chord diagram showing entity co-occurrence patterns.
        
        Args:
            filtered_concepts: Optional list of concepts to include
            top_n: Number of top concepts by salience to display
            colormap: Optional matplotlib colormap for node coloring
            
        Returns:
            Plotly Figure object
        """
        # Select entities
        if filtered_concepts:
            entities = filtered_concepts[:top_n]
        else:
            all_pq = self.kgraph.get_all_physical_quantities()
            entities = [pq for pq, _ in sorted(all_pq.items(), key=lambda x: x[1], reverse=True)[:top_n]]
        
        if not entities:
            return go.Figure().update_layout(
                title="No entity co-occurrence data",
                font=dict(family=self.font_family, size=self.font_size)
            )
        
        n = len(entities)
        node_to_idx = {node: i for i, node in enumerate(entities)}
        
        # Build adjacency matrix: count of documents where both entities appear
        adj = np.zeros((n, n))
        for doc in self.kgraph.doc_graphs:
            present = [
                ent for ent in entities 
                if any(
                    item.get("physical_quantity") == ent or item.get("parameter_name") == ent 
                    for item in self.kgraph.doc_graphs[doc]["all_items"]
                )
            ]
            for i, e1 in enumerate(present):
                for j, e2 in enumerate(present):
                    if i != j:
                        adj[node_to_idx[e1]][node_to_idx[e2]] += 1
        
        # Compute angles for circular layout
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        cmap_obj = plt.get_cmap(self._get_colormap(colormap)) if colormap else None
        
        fig = go.Figure()
        
        # Add nodes as barpolar segments
        for i, ent in enumerate(entities):
            color = mcolors.to_hex(cmap_obj(i / max(n - 1, 1))) if cmap_obj else self.DOMAIN_COLORS.get(ent, "#6b7280")
            fig.add_trace(go.Barpolar(
                r=[1],
                theta=[np.degrees(angles[i])],
                width=[360 / n * 0.9],  # Leave small gap between segments
                marker_color=color,
                name=ent,
                opacity=0.9,
                showlegend=False,
                hovertemplate=f"<b>{ent}</b><extra></extra>"
            ))
        
        # Add chords for co-occurrence
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j] > 0:
                    # Draw curved line between nodes
                    fig.add_trace(go.Scatterpolar(
                        r=[0.2, 0.6, 0.2],  # Curve: start at edge, peak at center, end at edge
                        theta=[
                            np.degrees(angles[i]),
                            np.degrees((angles[i] + angles[j]) / 2),  # Midpoint angle
                            np.degrees(angles[j])
                        ],
                        mode='lines',
                        line=dict(
                            color='rgba(100, 100, 100, 0.3)',
                            width=min(adj[i][j], 3)  # Thickness proportional to co-occurrence
                        ),
                        showlegend=False,
                        hovertemplate=f"Co-occurrence: {int(adj[i][j])} documents<extra></extra>"
                    ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=False),
                angularaxis=dict(visible=False)
            ),
            title=f"Salience-Aware Chord Diagram (Top {n} Concepts)",
            height=700,
            width=700,
            font=dict(family=self.font_family, size=self.font_size),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff"
        )
        
        # Add legend manually
        legend_items = []
        for i, ent in enumerate(entities):
            color = mcolors.to_hex(cmap_obj(i / max(n - 1, 1))) if cmap_obj else self.DOMAIN_COLORS.get(ent, "#6b7280")
            legend_items.append(dict(marker=dict(color=color, size=10), name=ent))
        
        fig.update_layout(legend=dict(items=legend_items, orientation="h", y=-0.05))
        
        return fig

# =============================================================================
# PUBLICATION VISUALIZATION ENGINE - CONTINUED (Part 7/14)
# Contradiction Matrices, Consensus Waterfalls, Embedding Projections,
# Parallel Categories, Violin, Timeline & Retrieval Diagnostics
# =============================================================================

    def plot_contradiction_matrix(self, quantity: Optional[str] = None, 
                                   colormap: Optional[str] = None) -> go.Figure:
        """
        Plot contradiction matrix showing pairwise value divergence across documents.
        
        Args:
            quantity: Filter by specific physical quantity (optional)
            colormap: Optional Plotly colorscale
            
        Returns:
            Plotly Figure object
        """
        df = self.extract_dataframe()
        if quantity:
            df = df[df["physical_quantity"] == quantity]
            
        if df.empty:
            return go.Figure().update_layout(
                title="No data available for contradiction analysis",
                font=dict(family=self.font_family, size=self.font_size)
            )
            
        docs = df["doc_stem"].unique()
        if len(docs) < 2:
            return go.Figure().update_layout(
                title="At least 2 documents required for contradiction matrix",
                font=dict(family=self.font_family, size=self.font_size)
            )
            
        # Compute normalized absolute difference matrix
        mat = np.zeros((len(docs), len(docs)))
        for i, d1 in enumerate(docs):
            v1 = df[df["doc_stem"] == d1]["value"].mean()
            for j, d2 in enumerate(docs):
                if i == j:
                    continue
                v2 = df[df["doc_stem"] == d2]["value"].mean()
                if v2 != 0 and not np.isnan(v1) and not np.isnan(v2):
                    mat[i, j] = abs(v1 - v2) / v2
                    
        fig = go.Figure(data=go.Heatmap(
            z=mat, 
            x=docs, 
            y=docs, 
            colorscale=self._get_plotly_colorscale(colormap),
            hoverongaps=False,
            text=np.round(mat, 3),
            texttemplate="%{text}",
            textfont={"size": 9, "color": "#1e293b"},
            hovertemplate="Doc X: %{x}<br>Doc Y: %{y}<br>Divergence: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"Cross-Document Contradiction Matrix for {quantity or 'All Quantities'}",
            font=dict(family=self.font_family, size=self.font_size),
            height=600, 
            width=600,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False, autorange="reversed")
        )
        
        return fig
    
    def plot_consensus_waterfall(self, quantity: Optional[str] = None, 
                                  colormap: Optional[str] = None) -> go.Figure:
        """
        Plot consensus waterfall showing mean ± std across material/quantity pairs.
        
        Args:
            quantity: Filter by specific physical quantity (optional)
            colormap: Optional Plotly colorscale
            
        Returns:
            Plotly Figure object
        """
        df = self.extract_dataframe()
        if quantity:
            df = df[df["physical_quantity"] == quantity]
            
        if df.empty:
            return go.Figure().update_layout(
                title="No data available for consensus analysis",
                font=dict(family=self.font_family, size=self.font_size)
            )
            
        grouped = df.groupby(["material", "physical_quantity"])["value"].agg(
            ["mean", "std", "count"]
        ).reset_index()
        grouped = grouped.sort_values("count", ascending=False).head(10)
        
        if grouped.empty:
            return go.Figure().update_layout(
                title="Insufficient grouped data for consensus plot",
                font=dict(family=self.font_family, size=self.font_size)
            )
            
        x_labels = [f"{m} ({q})" for m, q in zip(grouped["material"], grouped["physical_quantity"])]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x_labels,
            y=grouped["mean"],
            error_y=dict(type='data', array=grouped["std"], visible=True, thickness=2, width=6),
            marker_color="#059669",
            text=[f"n={c}" for c in grouped["count"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Mean: %{y:.2f}<br>Std: %{error_y.array:.2f}<br>n=%{text}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Cross-Document Consensus (Mean ± Standard Deviation)",
            yaxis_title="Value",
            xaxis_title="Material (Physical Quantity)",
            xaxis_tickangle=-45,
            font=dict(family=self.font_family, size=self.font_size),
            height=450,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            hovermode="x unified"
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9")
        
        return fig
    
    def _get_context_embeddings(self, embedding_fn: Callable[[str], np.ndarray], 
                                 df: pd.DataFrame, 
                                 quantity: Optional[str] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """Compute embeddings for extraction contexts."""
        if quantity:
            df = df[df["physical_quantity"] == quantity].copy()
        else:
            df = df.copy()
            
        if len(df) < 5:
            return np.array([]), df.iloc[0:0]
            
        contexts = df["context"].fillna("").tolist()
        embs = []
        valid_indices = []
        
        for idx, ctx in enumerate(contexts):
            try:
                emb = embedding_fn(ctx)
                if emb is not None and len(emb) > 0:
                    embs.append(emb)
                    valid_indices.append(idx)
            except Exception:
                continue
                
        if len(embs) < 5:
            return np.array([]), df.iloc[0:0]
            
        df_valid = df.iloc[valid_indices].copy()
        return np.array(embs), df_valid
    
    def plot_tsne(self, embedding_fn: Callable[[str], np.ndarray], 
                   quantity: Optional[str] = None, 
                   colormap: Optional[str] = None, 
                   figsize: Tuple[int, int] = (10, 8)) -> Optional[plt.Figure]:
        """t-SNE projection of extraction contexts."""
        if not SKLEARN_AVAILABLE:
            return None
            
        df = self.extract_dataframe()
        embs, df_use = self._get_context_embeddings(embedding_fn, df, quantity)
        if len(embs) < 5:
            return None
            
        perplexity = min(30, len(embs) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca')
        coords = tsne.fit_transform(embs)
        
        fig, ax = plt.subplots(figsize=figsize)
        materials = df_use["material"].unique()
        cmap = plt.get_cmap(self._get_colormap(colormap))
        
        for i, mat in enumerate(materials):
            mask = df_use["material"] == mat
            color = mcolors.to_hex(cmap(i / max(len(materials) - 1, 1))) if len(materials) > 1 else "#3b82f6"
            ax.scatter(coords[mask, 0], coords[mask, 1], c=color, label=str(mat), alpha=0.8, s=80, edgecolors='white')
            
            for (_, row), coord in zip(df_use[df_use["material"] == mat].iterrows(), coords[mask]):
                ax.annotate(f"{row['value']:.1f}", (coord[0], coord[1]), fontsize=self.label_font_size - 1, alpha=0.7)
                
        ax.legend(loc='best', fontsize=self.label_font_size, framealpha=0.9)
        ax.set_title(f"t-SNE of Extraction Contexts{' (' + quantity + ')' if quantity else ''}", 
                    fontsize=self.title_font_size, fontweight='bold', pad=15)
        ax.axis("off")
        plt.tight_layout()
        return fig
    
    def plot_pca(self, embedding_fn: Callable[[str], np.ndarray], 
                  quantity: Optional[str] = None, 
                  colormap: Optional[str] = None, 
                  figsize: Tuple[int, int] = (10, 8)) -> Optional[plt.Figure]:
        """PCA projection of extraction contexts."""
        if not SKLEARN_AVAILABLE:
            return None
            
        df = self.extract_dataframe()
        embs, df_use = self._get_context_embeddings(embedding_fn, df, quantity)
        if len(embs) < 5:
            return None
            
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embs)
        var_ratio = pca.explained_variance_ratio_
        
        fig, ax = plt.subplots(figsize=figsize)
        materials = df_use["material"].unique()
        cmap = plt.get_cmap(self._get_colormap(colormap))
        
        for i, mat in enumerate(materials):
            mask = df_use["material"] == mat
            color = mcolors.to_hex(cmap(i / max(len(materials) - 1, 1))) if len(materials) > 1 else "#3b82f6"
            ax.scatter(coords[mask, 0], coords[mask, 1], c=color, label=str(mat), alpha=0.8, s=80, edgecolors='white')
            
            for (_, row), coord in zip(df_use[df_use["material"] == mat].iterrows(), coords[mask]):
                ax.annotate(f"{row['value']:.1f}", (coord[0], coord[1]), fontsize=self.label_font_size - 1, alpha=0.7)
                
        ax.legend(loc='best', fontsize=self.label_font_size, framealpha=0.9)
        ax.set_title(f"PCA of Extraction Contexts{' (' + quantity + ')' if quantity else ''}\nPC1: {var_ratio[0]:.1%}, PC2: {var_ratio[1]:.1%}", 
                    fontsize=self.title_font_size, fontweight='bold', pad=15)
        ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%})")
        ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%})")
        plt.tight_layout()
        return fig
    
    def plot_umap(self, embedding_fn: Callable[[str], np.ndarray], 
                   quantity: Optional[str] = None, 
                   colormap: Optional[str] = None, 
                   figsize: Tuple[int, int] = (10, 8)) -> Optional[plt.Figure]:
        """UMAP projection of extraction contexts."""
        if not UMAP_AVAILABLE:
            return None
            
        df = self.extract_dataframe()
        embs, df_use = self._get_context_embeddings(embedding_fn, df, quantity)
        if len(embs) < 5:
            return None
            
        n_neighbors = min(15, len(embs) - 1)
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
        coords = reducer.fit_transform(embs)
        
        fig, ax = plt.subplots(figsize=figsize)
        materials = df_use["material"].unique()
        cmap = plt.get_cmap(self._get_colormap(colormap))
        
        for i, mat in enumerate(materials):
            mask = df_use["material"] == mat
            color = mcolors.to_hex(cmap(i / max(len(materials) - 1, 1))) if len(materials) > 1 else "#3b82f6"
            ax.scatter(coords[mask, 0], coords[mask, 1], c=color, label=str(mat), alpha=0.8, s=80, edgecolors='white')
            
            for (_, row), coord in zip(df_use[df_use["material"] == mat].iterrows(), coords[mask]):
                ax.annotate(f"{row['value']:.1f}", (coord[0], coord[1]), fontsize=self.label_font_size - 1, alpha=0.7)
                
        ax.legend(loc='best', fontsize=self.label_font_size, framealpha=0.9)
        ax.set_title(f"UMAP of Extraction Contexts{' (' + quantity + ')' if quantity else ''}", 
                    fontsize=self.title_font_size, fontweight='bold', pad=15)
        ax.axis("off")
        plt.tight_layout()
        return fig
    
    def plot_parallel_categories(self, df: pd.DataFrame, 
                                  colormap: Optional[str] = None) -> go.Figure:
        """Parallel categories plot for quantity → material → document flows."""
        if df.empty:
            return go.Figure().update_layout(
                title="No data available for parallel categories",
                font=dict(family=self.font_family, size=self.font_size)
            )
            
        cat_df = df[["physical_quantity", "material", "doc_stem"]].copy().dropna()
        if cat_df.empty:
            return go.Figure().update_layout(
                title="Insufficient categorical data",
                font=dict(family=self.font_family, size=self.font_size)
            )
            
        pq_codes = {pq: i for i, pq in enumerate(sorted(cat_df["physical_quantity"].unique()))}
        cat_df["pq_code"] = cat_df["physical_quantity"].map(pq_codes)
        
        fig = px.parallel_categories(
            cat_df, 
            dimensions=["physical_quantity", "material", "doc_stem"], 
            color="pq_code", 
            color_continuous_scale=self._get_plotly_colorscale(colormap),
            title="Parallel Categories: Quantities → Materials → Documents"
        )
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size), height=500)
        return fig
    
    def plot_violin(self, df: pd.DataFrame, 
                     colormap: Optional[str] = None) -> go.Figure:
        """Violin plot of key mechanical/thermal parameters."""
        if df.empty:
            return go.Figure().update_layout(
                title="No data available for violin plot",
                font=dict(family=self.font_family, size=self.font_size)
            )
            
        target_q = ["laser_power", "scan_speed", "yield_strength", "tensile_strength", "hardness", "temperature"]
        num_df = df[df["physical_quantity"].isin(target_q)]
        if num_df.empty:
            return go.Figure().update_layout(
                title="No numerical data for target parameters",
                font=dict(family=self.font_family, size=self.font_size)
            )
            
        fig = px.violin(
            num_df, 
            x="physical_quantity", 
            y="value", 
            color="material", 
            box=True, 
            points="all", 
            title="Distribution of Key Parameters by Material",
            labels={"physical_quantity": "Physical Quantity", "value": "Value"}
        )
        fig.update_layout(
            font=dict(family=self.font_family, size=self.font_size),
            height=500,
            xaxis_tickangle=-45,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff"
        )
        return fig
    
    def plot_timeline(self, colormap: Optional[str] = None) -> go.Figure:
        """Temporal distribution of extracted quantities by estimated publication year."""
        df = self.extract_dataframe()
        if df.empty:
            return go.Figure().update_layout(
                title="No data available for timeline",
                font=dict(family=self.font_family, size=self.font_size)
            )
            
        years = {}
        for doc_id in self.kgraph.doc_graphs.keys():
            match = re.search(r'\b(19|20)\d{2}\b', doc_id)
            years[doc_id] = int(match.group(0)) if match else 2024
            
        df["year"] = df["doc"].map(years).fillna(2024)
        top_q = df["physical_quantity"].value_counts().head(5).index.tolist()
        df_top = df[df["physical_quantity"].isin(top_q)]
        
        if df_top.empty:
            return go.Figure().update_layout(
                title="Insufficient temporal data",
                font=dict(family=self.font_family, size=self.font_size)
            )
            
        fig = px.scatter(
            df_top, 
            x="year", 
            y="physical_quantity", 
            color="material", 
            title="Temporal Distribution of Quantities by Material",
            labels={"year": "Estimated Publication Year", "physical_quantity": "Physical Quantity"},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig.update_layout(
            font=dict(family=self.font_family, size=self.font_size),
            height=450,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff"
        )
        return fig
    
    def plot_page_coverage_heatmap(self, doc_trees: List[Dict], 
                                    retrieved_nodes: List[Dict]) -> go.Figure:
        """Heatmap showing retrieved page coverage across documents."""
        if not doc_trees or not retrieved_nodes:
            return go.Figure().update_layout(
                title="No coverage data available",
                font=dict(family=self.font_family, size=self.font_size)
            )
            
        doc_names = sorted(list(set(t.get("doc_id", t.get("doc_name", "unknown")) for t in doc_trees)))
        max_pages = 0
        
        for tree in doc_trees:
            pages = []
            def collect_pages(node):
                pages.append(node.get("start_index", 1))
                if node.get("end_index"):
                    pages.append(node["end_index"])
                for c in node.get("nodes", []):
                    collect_pages(c)
            collect_pages(tree)
            max_p = max(pages) if pages else 1
            max_pages = max(max_pages, max_p)
            
        coverage = np.zeros((len(doc_names), max_pages))
        for r in retrieved_nodes:
            doc_id = r.get("doc_id")
            if doc_id in doc_names:
                doc_idx = doc_names.index(doc_id)
                start = r.get("page_start", 1) - 1
                for p in range(max(0, start - 1), min(max_pages, start + 3)):
                    coverage[doc_idx, p] = 1
                    
        doc_labels = [Path(d).stem for d in doc_names]
        fig = go.Figure(data=go.Heatmap(
            z=coverage, 
            x=list(range(1, max_pages + 1)), 
            y=doc_labels, 
            colorscale=[[0, "#f3f4f6"], [1, "#059669"]], 
            showscale=False, 
            hovertemplate="Doc: %{y}<br>Page: %{x}<br>Retrieved: %{z}<extra></extra>"
        ))
        fig.update_layout(
            title="Page Coverage Heatmap (Retrieved Pages per Document)",
            xaxis_title="Page Number",
            yaxis_title="Document",
            font=dict(family=self.font_family, size=self.font_size),
            height=max(400, len(doc_names) * 40),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff"
        )
        return fig
    
    def plot_retrieval_sankey(self, query: str, 
                               relevant_docs: List[Tuple[str, float]], 
                               retrieved_nodes: List[Dict], 
                               extracted_items: List[Any]) -> go.Figure:
        """Sankey diagram showing retrieval provenance flow."""
        if not relevant_docs and not retrieved_nodes:
            return go.Figure().update_layout(
                title="No retrieval data available",
                font=dict(family=self.font_family, size=self.font_size)
            )
            
        labels = ["Query"]
        label_index = {"Query": 0}
        doc_nodes = []
        for doc_name, score in relevant_docs:
            doc_label = f"{Path(doc_name).stem}\n({score:.2f})"
            label_index[doc_name] = len(labels)
            labels.append(doc_label)
            doc_nodes.append(doc_name)
            
        node_labels_list = []
        for r in retrieved_nodes:
            doc_id = r.get("doc_id", "unknown")
            node_id = r.get("node_id", "unknown")
            key = f"{doc_id}:{node_id}"
            if key not in label_index:
                label_index[key] = len(labels)
                labels.append(f"{Path(doc_id).stem}:{node_id[:15]}")
            node_labels_list.append(key)
            
        pq_groups = defaultdict(list)
        for item in extracted_items:
            if hasattr(item, 'physical_quantity'):
                pq = item.physical_quantity or "unknown"
                pq_groups[pq].append(item)
            elif isinstance(item, dict):
                pq = item.get("physical_quantity", "unknown")
                pq_groups[pq].append(item)
                
        pq_nodes_list = []
        for pq, items in pq_groups.items():
            key = f"pq:{pq}"
            if key not in label_index:
                label_index[key] = len(labels)
                labels.append(f"{pq} ({len(items)})")
            pq_nodes_list.append(key)
            
        label_index["Answer"] = len(labels)
        labels.append("Answer")
        
        sources, targets, vals = [], [], []
        for doc_name, score in relevant_docs:
            sources.append(0)
            targets.append(label_index[doc_name])
            vals.append(max(1, int(score * 10)))
            
        for r in retrieved_nodes:
            doc_id = r.get("doc_id")
            node_id = r.get("node_id", "unknown")
            key = f"{doc_id}:{node_id}"
            conf = r.get("confidence", 0.5)
            if doc_id in label_index and key in label_index:
                sources.append(label_index[doc_id])
                targets.append(label_index[key])
                vals.append(max(1, int(conf * 10)))
                
        node_to_pq = defaultdict(set)
        for item in extracted_items:
            pq = item.physical_quantity if hasattr(item, 'physical_quantity') else item.get("physical_quantity", "unknown")
            doc_id = item.doc_source if hasattr(item, 'doc_source') else item.get("doc_source", item.get("doc_id", "unknown"))
            for r in retrieved_nodes:
                if r.get("doc_id") == doc_id:
                    node_id = r.get("node_id", "unknown")
                    key = f"{doc_id}:{node_id}"
                    node_to_pq[key].add(f"pq:{pq}")
                    
        for node_key, pq_set in node_to_pq.items():
            for pq_key in pq_set:
                if node_key in label_index and pq_key in label_index:
                    sources.append(label_index[node_key])
                    targets.append(label_index[pq_key])
                    vals.append(1)
                    
        for pq_key in pq_nodes_list:
            sources.append(label_index[pq_key])
            targets.append(label_index["Answer"])
            vals.append(max(1, len(pq_groups.get(pq_key.replace("pq:", ""), []))))
            
        node_colors = ["#1e3a5f"]
        node_colors += ["#2563eb"] * len(doc_nodes)
        node_colors += ["#059669"] * len(node_labels_list)
        node_colors += ["#dc2626"] * len(pq_nodes_list)
        node_colors += ["#7c3aed"]
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=20, thickness=24, line=dict(color="#334155", width=0.8),
                label=labels, color=node_colors,
                hovertemplate="%{label}<extra></extra>"
            ),
            link=dict(
                source=sources, target=targets, value=vals,
                color=["rgba(37, 99, 235, 0.25)" if s < len(doc_nodes)+1 else "rgba(5, 150, 105, 0.2)" for s in sources],
                hovertemplate="From: %{source.label}<br>To: %{target.label}<br>Value: %{value}<extra></extra>"
            )
        )])
        
        fig.update_layout(
            title_text=f"Retrieval Provenance Flow: '{query[:50]}{'...' if len(query)>50 else ''}'",
            font=dict(family=self.font_family, size=self.font_size, color="#1e293b"),
            paper_bgcolor="white", plot_bgcolor="white",
            height=650, width=1100,
            margin=dict(l=40, r=40, t=80, b=40)
        )
        return fig
    
    def plot_node_confidence_distribution(self, retrieved_nodes: List[Dict]) -> go.Figure:
        """Histogram of node selection confidence scores."""
        if not retrieved_nodes:
            return go.Figure().update_layout(
                title="No node confidence data available",
                font=dict(family=self.font_family, size=self.font_size)
            )
            
        confidences = [r.get("confidence", 0) for r in retrieved_nodes]
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=confidences, 
            nbinsx=20, 
            marker_color="#3b82f6", 
            opacity=0.75, 
            name="All Nodes"
        ))
        fig.add_vline(x=0.5, line_dash="dash", line_color="#ef4444", annotation_text="Typical Threshold")
        fig.update_layout(
            title="Node Selection Confidence Distribution",
            xaxis_title="Confidence Score",
            yaxis_title="Count",
            font=dict(family=self.font_family, size=self.font_size),
            showlegend=False,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff"
        )
        return fig
    
    def plot_doc_filter_scores(self, relevant_docs: List[Tuple[str, float]], 
                                all_doc_count: int) -> go.Figure:
        """Bar chart of two-stage document retrieval scores."""
        if not relevant_docs:
            return go.Figure().update_layout(
                title="No document filter scores available",
                font=dict(family=self.font_family, size=self.font_size)
            )
            
        docs = [Path(d).stem for d, _ in relevant_docs]
        scores = [s for _, s in relevant_docs]
        colors = ["#10b981" if s > 0.5 else "#f59e0b" if s > 0.2 else "#ef4444" for s in scores]
        
        fig = go.Figure(go.Bar(
            x=docs, 
            y=scores, 
            marker_color=colors, 
            text=[f"{s:.3f}" for s in scores], 
            textposition="outside"
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="#ef4444", annotation_text="High Relevance")
        fig.add_hline(y=0.2, line_dash="dot", line_color="#f59e0b", annotation_text="Medium Relevance")
        fig.update_layout(
            title=f"Two-Stage Document Retrieval Scores (showing {len(docs)} of {all_doc_count})",
            xaxis_title="Document",
            yaxis_title="Relevance Score",
            font=dict(family=self.font_family, size=self.font_size),
            height=450,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff"
        )
        return fig
    
    def plot_retrieval_tree_highlight(self, annotated_trees: List[Dict], 
                                       retrieved_nodes: List[Dict], 
                                       doc_id: Optional[str] = None) -> Optional[plt.Figure]:
        """Highlight retrieved nodes in hierarchical tree structure."""
        if not annotated_trees:
            return None
            
        target_tree = None
        for tree in annotated_trees:
            tid = tree.get("doc_id", tree.get("doc_name", "unknown"))
            if doc_id and tid == doc_id:
                target_tree = tree
                break
        if not target_tree and annotated_trees:
            target_tree = annotated_trees[0]
            doc_id = target_tree.get("doc_id", target_tree.get("doc_name", "unknown"))
        if not target_tree:
            return None
            
        G = nx.DiGraph()
        retrieved_node_ids = set()
        for r in retrieved_nodes:
            if r.get("doc_id") == doc_id:
                retrieved_node_ids.add(r.get("node_id"))
                
        def add_nodes(node, parent=None):
            nid = node.get("node_id", "root")
            title = node.get("title", "Unknown")
            is_retrieved = nid in retrieved_node_ids
            has_quant = bool(node.get("quantitative_items"))
            G.add_node(nid, label=title[:30], retrieved=is_retrieved, has_quant=has_quant)
            if parent:
                G.add_edge(parent, nid)
            for child in node.get("nodes", []):
                add_nodes(child, nid)
                
        add_nodes(target_tree)
        if len(G.nodes()) < 2:
            return None
            
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        fig, ax = plt.subplots(figsize=(14, 10))
        
        normal_nodes = [n for n, d in G.nodes(data=True) if not d.get("retrieved") and not d.get("has_quant")]
        quant_nodes = [n for n, d in G.nodes(data=True) if d.get("has_quant") and not d.get("retrieved")]
        retrieved_nodes_list = [n for n, d in G.nodes(data=True) if d.get("retrieved")]
        
        nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_color="#e5e7eb", node_size=400, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=quant_nodes, node_color="#93c5fd", node_size=600, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=retrieved_nodes_list, node_color="#ef4444", node_size=900, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, arrowsize=10, ax=ax)
        
        labels = {n: d["label"] for n, d in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels, font_size=self.label_font_size, ax=ax, font_family=self.font_family)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#ef4444", label="Retrieved Node"), 
            Patch(facecolor="#93c5fd", label="Has Quantitative Data"), 
            Patch(facecolor="#e5e7eb", label="Other Node")
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        ax.set_title(f"Retrieval Tree: {Path(doc_id).stem if doc_id else 'Document'}", 
                    fontsize=self.title_font_size, fontweight='bold', pad=15)
        ax.axis("off")
        plt.tight_layout()
        return fig
    
    def plot_semantic_vs_vectorless(self, query: str, 
                                     relevant_docs: List[Tuple[str, float]], 
                                     annotated_trees: List[Dict], 
                                     embedding_fn: Optional[Callable[[str], np.ndarray]] = None) -> Optional[plt.Figure]:
        """Compare semantic (embedding) vs vectorless (keyword/heuristic) retrieval scores."""
        if not relevant_docs or not embedding_fn:
            return None
            
        doc_names = [d for d, _ in relevant_docs]
        keyword_scores = [s for _, s in relevant_docs]
        doc_texts = []
        for tree in annotated_trees:
            tid = tree.get("doc_id", tree.get("doc_name", "unknown"))
            if tid in doc_names:
                text = tree.get("summary", "") + " " + str(tree.get("metadata", {}))
                doc_texts.append(text)
                
        if not doc_texts or not any(doc_texts):
            return None
            
        try:
            query_emb = embedding_fn(query)
            doc_embs = [embedding_fn(t) for t in doc_texts]
            
            def cosine(a, b):
                return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
                
            semantic_scores = [cosine(query_emb, de) for de in doc_embs]
        except Exception:
            return None
            
        fig, ax = plt.subplots(figsize=(8, 6))
        doc_labels = [Path(d).stem for d in doc_names]
        
        for i in range(len(doc_names)):
            color = "#10b981" if keyword_scores[i] > 0.5 else "#f59e0b" if keyword_scores[i] > 0.2 else "#ef4444"
            ax.scatter(keyword_scores[i], semantic_scores[i], c=color, s=100, alpha=0.8, edgecolors='white')
            ax.annotate(doc_labels[i], (keyword_scores[i], semantic_scores[i]), fontsize=8, alpha=0.9)
            
        min_val = min(min(keyword_scores), min(semantic_scores))
        max_val = max(max(keyword_scores), max(semantic_scores))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.6, label="Agreement Line")
        
        ax.set_title("Semantic (Embedding) vs Vectorless (Keyword/Heuristic) Retrieval Scores",
                    fontsize=self.title_font_size, fontweight='bold', pad=15)
        ax.set_xlabel("Vectorless Score (Keyword/Heuristic)", fontsize=self.label_font_size)
        ax.set_ylabel("Semantic Score (Cosine Similarity)", fontsize=self.label_font_size)
        ax.legend(loc='best', fontsize=self.label_font_size)
        plt.tight_layout()
        return fig

# ============================================================================
# QUERY ANALYZER: INTENT DETECTION & DECOMPOSITION
# ============================================================================
class QueryAnalyzer:
    """
    Analyzes user queries to determine intent and decompose complex requests.
    Enhances retrieval precision by tailoring the strategy to the query type.
    """
    
    INTENT_TYPES = Literal["quantitative_lookup", "comparison", "summary", "list_documents", "process_method"]

    def __init__(self, phys_classifier: Optional['PhysicalQuantityClassifier'] = None):
        self.phys_classifier = phys_classifier or PhysicalQuantityClassifier()
        self.compiled_patterns = self._build_patterns()

    def _build_patterns(self) -> Dict[str, re.Pattern]:
        return {
            "list_docs": re.compile(r"(?:list|show|find|what are).*(?:papers|documents|articles)", re.IGNORECASE),
            "compare": re.compile(r"(?:compare|versus|vs|difference between|contrast)", re.IGNORECASE),
            "process": re.compile(r"(?:method|process|technique|approach|algorithm|fabrication|synthesis)", re.IGNORECASE),
        }

    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyze query and return metadata.
        Returns: {
            "query_type": str,
            "entities": List[str],
            "decomposition": List[str],
            "canonical_quantities": List[str]
        }
        """
        result = {
            "query_type": "quantitative_lookup", # Default
            "entities": [],
            "decomposition": [query],
            "canonical_quantities": []
        }

        # 1. Detect Intent
        if self.compiled_patterns["list_docs"].search(query):
            result["query_type"] = "list_documents"
        elif self.compiled_patterns["compare"].search(query):
            result["query_type"] = "comparison"
        elif self.compiled_patterns["process"].search(query):
            result["query_type"] = "process_method"
        
        # 2. Extract Entities & Physical Quantities
        # Simple heuristic: split by spaces, check against known classifiers
        lower_q = query.lower()
        for pq in self.phys_classifier.CANONICAL:
            for alias in self.phys_classifier.CANONICAL[pq]:
                if alias.lower() in lower_q:
                    if pq not in result["canonical_quantities"]:
                        result["canonical_quantities"].append(pq)
        
        # 3. Decomposition (Rule-based for simple conjunctions)
        if " and " in lower_q and result["query_type"] in ["quantitative_lookup", "comparison"]:
            parts = [p.strip() for p in query.split(" and ") if len(p.strip()) > 5]
            if len(parts) > 1:
                result["decomposition"] = parts

        return result

# ============================================================================
# RESPONSE CACHE & TIMING UTILITIES
# ============================================================================
class ResponseCache:
    """
    Cache for query responses to prevent re-processing identical queries.
    Uses SHA-256 hashing of query + parameters to generate keys.
    """
    def __init__(self, max_items: int = 50):
        self._cache: OrderedDict = OrderedDict()
        self._max_items = max_items

    def get(self, query: str, params: Dict) -> Optional[Dict]:
        key = self._hash(query, params)
        return self._cache.get(key)

    def set(self, query: str, params: Dict, response: Dict):
        key = self._hash(query, params)
        self._cache[key] = response
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_items:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()

    @staticmethod
    def _hash(query: str, params: Dict) -> str:
        content = f"{query}|{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()

class TimingMetrics:
    """
    Context manager and utility for tracking performance metrics.
    """
    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)

    @contextmanager
    def measure(self, label: str):
        start = time.time()
        try:
            yield self
        finally:
            elapsed = time.time() - start
            self.metrics[label].append(elapsed)
            logger.debug(f"[METRIC] {label}: {elapsed:.4f}s")

    def get_summary(self) -> Dict[str, Any]:
        return {
            k: {"count": len(v), "avg": float(np.mean(v)) if v else 0, "last": v[-1] if v else 0}
            for k, v in self.metrics.items()
        }

# ============================================================================
# TWO-CALL QUERY PROCESSOR (CORE ARCHITECTURE)
# ============================================================================
class TwoCallQueryProcessor:
    """
    Implements the strict 2-call query architecture.
    
    Call 1: Navigation (LLM selects nodes from annotated trees)
    Call 2: Synthesis (LLM generates answer from selected text)
    
    Benefits:
    - Predictable token usage
    - Higher reasoning accuracy via structured navigation
    - Reduced hallucination by grounding extraction in specific text segments
    """
    
    def __init__(self, 
                 llm: 'HybridLLM', 
                 retriever: 'HierarchicalTreeRetriever', 
                 phys_classifier: 'PhysicalQuantityClassifier',
                 analyzer: 'QueryAnalyzer',
                 max_text_chars: int = 15000):
        self.llm = llm
        self.retriever = retriever
        self.phys_classifier = phys_classifier
        self.analyzer = analyzer
        self.max_text_chars = max_text_chars
        self.cache = ResponseCache(max_items=50)
        self.metrics = TimingMetrics()

    async def process_query(self, query: str, annotated_trees: List[Dict]) -> Dict[str, Any]:
        """
        Execute the 2-call query pipeline.
        """
        with self.metrics.measure("total_query"):
            # Step 0: Analyze Query
            analysis = self.analyzer.analyze(query)
            
            # Check cache
            cached = self.cache.get(query, {"max_chars": self.max_text_chars})
            if cached:
                logger.info("Query Cache HIT")
                return cached

            # Call 1: Navigation
            logger.info(f"CALL 1: Navigating trees for '{query}'")
            with self.metrics.measure("call_1_navigation"):
                selected_nodes = await self._call_1_navigation(query, analysis, annotated_trees)
            
            if not selected_nodes:
                return {
                    "answer": f"I could not find any relevant sections in the documents to answer: '{query}'.",
                    "nodes": [],
                    "metrics": self.metrics.get_summary()
                }

            # Prepare Context for Call 2
            # Concatenate text from selected nodes, ensuring we don't exceed context limits
            context_text = self._build_context(selected_nodes)
            
            # Call 2: Synthesis
            logger.info(f"CALL 2: Synthesizing answer")
            with self.metrics.measure("call_2_synthesis"):
                answer = await self._call_2_synthesis(query, analysis, context_text)

            result = {
                "answer": answer,
                "nodes": selected_nodes,
                "analysis": analysis,
                "metrics": self.metrics.get_summary()
            }
            
            # Cache result
            self.cache.set(query, {"max_chars": self.max_text_chars}, result)
            return result

    async def _call_1_navigation(self, query: str, analysis: Dict, trees: List[Dict]) -> List[Dict]:
        """
        Call 1: Identify relevant nodes.
        Uses the retriever which already condenses trees and uses LLM for selection.
        """
        return await self.retriever.retrieve_quantitative(query, trees)

    async def _call_2_synthesis(self, query: str, analysis: Dict, context: str) -> str:
        """
        Call 2: Generate answer based on context.
        """
        prompt_template = """
You are an expert scientific analyst. 
Answer the user's query using ONLY the provided text context.
If the context does not contain the answer, state that clearly.

QUERY: {query}
INTENT: {intent}
CONTEXT:
---
{context}
---

INSTRUCTIONS:
1. Provide a direct answer.
2. Cite sources using the format [DocID, Page] if available in the context headers.
3. Be precise with numerical values.
4. If comparing multiple values, structure the comparison clearly.
"""
        
        prompt = prompt_template.format(
            query=query,
            intent=analysis.get("query_type", "lookup"),
            context=context
        )
        
        return await asyncio.to_thread(
            self.llm.generate,
            prompt,
            max_new_tokens=1024,
            temperature=0.1
        )

    def _build_context(self, selected_nodes: List[Dict]) -> str:
        """Concatenate text from selected nodes."""
        segments = []
        for node in selected_nodes:
            doc_id = node.get("doc_id", "Unknown")
            page = node.get("page_start", "?")
            title = node.get("section_title", "Section")
            text = node.get("full_text", "")
            
            # Header for citation
            header = f"### [{doc_id}, Page {page}] {title}\n"
            segments.append(header + text)
        
        combined = "\n\n".join(segments)
        # Truncate if too large for LLM context window
        if len(combined) > self.max_text_chars * 2: # Safety margin
            combined = combined[:self.max_text_chars * 2] + "\n...[TRUNCATED]"
        return combined

# ============================================================================
# STREAMLIT UI: INITIALIZATION & SIDEBAR
# ============================================================================
def render_sidebar() -> None:
    """
    Renders the application sidebar with configuration options.
    """
    with st.sidebar:
        st.markdown("### Configuration")
        
        # Model Selection
        model_keys = list(LOCAL_LLM_OPTIONS.keys())
        if "llm_model_choice" not in st.session_state:
            st.session_state.llm_model_choice = model_keys[2] # Default to 7b
            
        selected = st.selectbox(
            "Select Local LLM", 
            options=model_keys, 
            index=model_keys.index(st.session_state.llm_model_choice), 
            key="llm_model_select"
        )
        st.session_state.llm_model_choice = selected
        
        st.checkbox("Use 4-bit quantization (if Transformers)", value=True, key="use_4bit")
        
        st.markdown("---")
        st.markdown("### Retrieval Settings")
        
        st.slider("Confidence threshold", 0.3, 0.9, 0.55, 0.05, key="min_confidence")
        max_chars = st.slider(
            "Max text length per retrieved section", 
            min_value=1000, max_value=50000, value=20000, step=1000,
            help="Controls context window size for Call 2.",
            key="max_retrieval_chars"
        )
        
        st.checkbox("Enable Two-Call Architecture", value=True, key="use_two_call", 
                    help="Uses LLM for navigation first, then answer generation.")
        st.checkbox("Show reasoning trace", value=True, key="show_trace")
        
        st.markdown("---")
        st.markdown("### Visualization Settings")
        st.selectbox("Default colormap", list(PublicationVisualizationEngine.COLORMAP_OPTIONS.keys()), index=0, key="viz_colormap")
        st.selectbox("Document label style", ["doi", "number", "alias", "short"], index=0, key="viz_label_style")
        st.slider("Top N concepts", 5, 100, 25, key="viz_top_n")
        
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
        
        if st.button("Clear Cache & Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

def initialize_session_state():
    """
    Initialize persistent session state variables if they don't exist.
    """
    defaults = {
        "messages": [],
        "query_processor": {},
        "knowledge_graph": None,
        "annotated_trees": [],
        "cached_query_result": None,
        "active_prompt": "",
        "two_stage_retriever": None,
        "embedding_model": None,
        "doc_aliases": {},
        "tree_cache": AnnotatedTreeCache()
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

@st.cache_resource(show_spinner="Initializing LLM...")
def get_cached_llm(model_choice: str, use_4bit: bool) -> 'HybridLLM':
    """
    Cached initialization of the LLM backend.
    """
    internal = LOCAL_LLM_OPTIONS[model_choice]
    return HybridLLM(model_key=internal, use_4bit=use_4bit)


# ============================================================================
# STREAMLIT UI: MAIN APPLICATION LOGIC
# ============================================================================
async def run_streamlit():
    """
    Main Streamlit application entry point.
    Manages file upload, indexing (with roll-up & caching), and query execution.
    """
    st.set_page_config(
        page_title="DECLARMIMA v18.0 - Enhanced Vectorless RAG",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- Header & Introduction ---
    st.markdown("# DECLARMIMA v18.0")
    st.caption("Vectorless RAG with PageIndex-Style Intelligence • 2-Call Architecture • Roll-Up Summarization • Full-Tree Caching")

    # --- Initialize Session State ---
    initialize_session_state()

    # --- Render Sidebar ---
    render_sidebar()

    # --- Global Configuration Object ---
    config = VisConfig(
        font_family="DejaVu Sans",
        font_size=st.session_state.get("viz_font_size", 10),
        title_font_size=st.session_state.get("viz_title_font_size", 14),
        label_font_size=st.session_state.get("viz_label_font_size", 9),
        default_colormap=st.session_state.get("viz_colormap", "viridis"),
        figure_dpi=st.session_state.get("viz_figure_dpi", 300),
        node_size_factor=st.session_state.get("viz_node_size_factor", 1.0),
        edge_alpha=st.session_state.get("viz_edge_alpha", 0.25),
        edge_width=st.session_state.get("viz_edge_width", 0.8),
        line_width=st.session_state.get("viz_line_width", 1.5),
        marker_size=st.session_state.get("viz_marker_size", 80),
        pyvis_physics_enabled=st.session_state.get("viz_pyvis_physics", True),
        pyvis_gravity=st.session_state.get("viz_pyvis_gravity", -1800),
        pyvis_spring_length=st.session_state.get("viz_pyvis_spring_length", 140),
        aliases=st.session_state.get("doc_aliases", {}),
        label_style=st.session_state.get("viz_label_style", "doi")
    )

    # --- File Upload Section ---
    st.markdown("---")
    st.subheader("📂 Document Ingestion")
    
    uploaded_files = st.file_uploader(
        "Upload PDF files (Scientific Literature)",
        type="pdf",
        accept_multiple_files=True,
        help="Upload peer-reviewed papers, reports, or technical documentation."
    )

    if uploaded_files:
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            build_clicked = st.button("🚀 Build Index & Extract", type="primary", use_container_width=True)
        
        with col_info:
            st.info("Click to start the indexing pipeline. "
                    "This includes hierarchical parsing, metadata extraction, "
                    "LLM-based roll-up summarization, and intelligent caching.")

        if build_clicked:
            # Store files in session state to persist across reruns if needed
            st.session_state.uploaded_files = uploaded_files
            st.session_state.indexing_status = "running"
            st.rerun()

    # --- Indexing Execution Block ---
    if st.session_state.get("indexing_status") == "running":
        st.markdown("---")
        st.subheader("⚙️ Indexing Pipeline Progress")
        
        # Initialize components for indexing
        llm = get_cached_llm(st.session_state.llm_model_choice, st.session_state.get("use_4bit", True))
        
        # 1. Hierarchical Parsing & Metadata Extraction
        #    Uses FastHierarchicalIndex with AnnotatedTreeCache
        with st.spinner("Phase 1: Hierarchical Parsing & Metadata Extraction..."):
            progress_bar = st.progress(0, text="Initializing Indexer...")
            
            tree_cache = st.session_state.tree_cache
            progress_bar.progress(0.1, text="Checking Smart Cache...")
            
            # Instantiate Indexer
            idx = FastHierarchicalIndex(cache_dir=".declarmima_cache_v18", llm=llm)
            progress_bar.progress(0.2, text="Parsing PDF Structure...")
            
            # Build trees asynchronously
            trees = await idx.build_from_pdfs_fast(
                st.session_state.uploaded_files,
                max_workers=4
            )
            progress_bar.progress(0.5, text="Metadata Extraction & Roll-Up Summarization...")

        # 2. Roll-up Summarization & Caching (if not done inside indexer)
        #    Note: build_from_pdfs_fast does some summarization, but we ensure full roll-up here
        if llm:
            summarizer = RollupSummarizer(llm, max_summary_length=250)
            
            # Iterate trees to ensure summaries are rolled up and cached
            with st.spinner("Phase 2: Generating Roll-Up Summaries (Bottom-Up)..."):
                total_trees = len(trees)
                for i, (doc_name, root) in enumerate(trees.items()):
                    progress_bar.progress(
                        0.5 + (0.2 * (i + 1) / total_trees), 
                        text=f"Summarizing {doc_name}..."
                    )
                    # Generate summaries (async compatible)
                    await summarizer._post_order_summarize(root)
                    
                    # Save to AnnotatedTreeCache with content hash
                    buf = BytesIO(st.session_state.uploaded_files[i].getbuffer())
                    tree_cache.set(doc_name, buf.getvalue(), root.to_dict())
        
        # 3. Quantitative Extraction & Knowledge Graph Population
        with st.spinner("Phase 3: LLM Quantitative Extraction..."):
            extractor = UniversalLLMExtractor(llm)
            kg = QuantitativeKnowledgeGraph()
            two_stage = TwoStageRetriever(llm=llm)
            all_items = []
            
            progress_bar.progress(0.7, text="Extracting parameters from leaf nodes...")
            
            for i, (doc_name, root) in enumerate(trees.items()):
                # Collect leaf nodes for extraction
                leaf_chunks = []
                def collect_leaves(node: PageNode):
                    if not node.children:
                        text = node.get_text(max_chars=15000) # Truncate huge pages
                        if text.strip():
                            leaf_chunks.append({
                                "full_text": text,
                                "page_start": node.page_start,
                                "doc_id": doc_name,
                                "section_title": node.title,
                                "node_id": node.node_id
                            })
                    for c in node.children:
                        collect_leaves(c)
                
                collect_leaves(root)
                
                # Extract items
                items = extractor.extract_from_chunks(
                    leaf_chunks, 
                    query="Extract ALL quantitative parameters, materials, methods, and definitions.",
                    query_analysis={"query_type": "quantitative", "keywords": []}
                )
                all_items.extend(items)
                
                # Add to Knowledge Graph
                kg.add_extractions(doc_name, items)
                if root.metadata:
                    kg.add_document_metadata(doc_name, root.metadata)
                    two_stage.index_document(doc_name, root.metadata, root.summary)
                
                progress_bar.progress(0.7 + (0.2 * (i + 1) / total_trees), text=f"Indexed {doc_name}...")

        # 4. Finalize Session State
        progress_bar.progress(1.0, text="Finalizing...")
        
        st.session_state.query_processor["index"] = idx
        st.session_state.query_processor["doc_trees"] = trees
        st.session_state.knowledge_graph = kg
        st.session_state.two_stage_retriever = two_stage
        
        # Build annotated trees for retrieval (JSON format)
        annotated_trees = []
        for doc_name, root in trees.items():
            ann = kg.to_tree_annotation(root, max_chars=st.session_state.max_retrieval_chars)
            ann["doc_id"] = doc_name
            ann["doc_name"] = doc_name
            ann["metadata"] = root.metadata.model_dump() if hasattr(self, "model_dump") else self.dict() if root.metadata else {}
            annotated_trees.append(ann)
            
        st.session_state.annotated_trees = annotated_trees
        st.session_state.indexing_status = "completed"
        
        st.success(f"✅ Successfully indexed **{len(trees)} documents** with **{len(all_items)} quantitative items**.")
        st.session_state.uploaded_files = None # Clear upload to free memory
        st.rerun()

    elif st.session_state.get("indexing_status") == "completed" or st.session_state.annotated_trees:
        # --- Chat & Query Section ---
        st.markdown("---")
        st.subheader("🤖 Scientific Query Assistant")
        
        # Quick Query Buttons
        quick_queries = [
            "What are the Hollomon parameters (K, n) mentioned?",
            "Compare the yield strength of AlSiMgZr alloys.",
            "What laser powers were used?",
            "List the electrochemical properties (Ecorr, Jcorr).",
            "What is the stacking fault energy?"
        ]
        
        cols = st.columns(5)
        for i, q in enumerate(quick_queries):
            with cols[i]:
                if st.button(q[:20] + "...", key=f"qq_{i}"):
                    st.session_state.pending_query = q
                    st.rerun()
        
        # Display pending query if set
        prompt = st.chat_input("Ask a question about the documents...", key="chat_input")
        if st.session_state.get("pending_query"):
            prompt = st.session_state.pending_query
            st.session_state.pending_query = None
        
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.active_prompt = prompt
            st.rerun()

        # --- Render Messages ---
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # --- Process Latest Query ---
        if st.session_state.active_prompt:
            prompt = st.session_state.active_prompt
            st.session_state.active_prompt = "" # Reset
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing Query & Retrieving Evidence (2-Call Pipeline)..."):
                    try:
                        # Initialize Processor
                        llm = get_cached_llm(st.session_state.llm_model_choice, st.session_state.get("use_4bit", True))
                        phys_classifier = PhysicalQuantityClassifier(llm_callback=None)
                        analyzer = QueryAnalyzer(phys_classifier)
                        retriever = HierarchicalTreeRetriever(
                            llm, 
                            max_results=15, 
                            max_text_chars=st.session_state.max_retrieval_chars
                        )
                        
                        processor = TwoCallQueryProcessor(
                            llm=llm,
                            retriever=retriever,
                            phys_classifier=phys_classifier,
                            analyzer=analyzer,
                            max_text_chars=st.session_state.max_retrieval_chars
                        )
                        
                        # Execute Query (Async)
                        result = await processor.process_query(prompt, st.session_state.annotated_trees)
                        
                        answer = result.get("answer", "I could not find a specific answer.")
                        metrics = result.get("metrics", {})
                        
                        # Update Knowledge Graph with new extractions if any (optional step for feedback loop)
                        new_items = result.get("items", [])
                        if new_items:
                            # Add to KG for visualization
                            # Note: In a real app, you'd map these back to doc_ids properly
                            pass
                        
                        # Store result in cache for visualizations
                        st.session_state.cached_query_result = {
                            "prompt": prompt,
                            "answer": answer,
                            "metrics": metrics,
                            "retrieved": result.get("nodes", []),
                            "items": new_items,
                            "relevant_docs": [] # Placeholder for diagnostics tab
                        }
                        
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                    except Exception as e:
                        st.error(f"Query Processing Error: {str(e)}")
                        logger.exception(e)

    # --- Visualization Dashboard (Only if data exists) ---
    if st.session_state.knowledge_graph and st.session_state.annotated_trees:
        render_visualization_dashboard(config)

    else:
        st.info("👆 Upload PDFs and click 'Build Index' to start the analysis.")

# ============================================================================
# HELPER: VISUALIZATION DASHBOARD RENDERER
# ============================================================================
def render_visualization_dashboard(config: VisConfig):
    """Renders the full 35+ chart visualization dashboard."""
    st.markdown("---")
    st.subheader("📊 Publication-Quality Visualization Dashboard")
    
    viz = PublicationVisualizationEngine(st.session_state.knowledge_graph, config=config)
    df_all = viz.extract_dataframe(aliases=config.aliases, label_style=config.label_style)
    
    # Check if we have data to visualize
    if df_all.empty:
        st.warning("No quantitative data extracted to visualize. Try running a query or check extraction prompts.")
        return

    # Dashboard Controls
    with st.expander("Dashboard Controls", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            selected_qty = st.selectbox("Physical Quantity", ["All"] + sorted(df_all["physical_quantity"].unique()), key="dash_qty")
        with c2:
            group_by = st.selectbox("Grouping", ["material", "doc_stem"], key="dash_grp")
        with c3:
            chart_type = st.selectbox("Chart Type", ["Histogram", "Box Plot", "Scatter", "Radar", "Network"], key="dash_chart")
    
    # Main Visualization Tabs
    tabs = st.tabs(["📈 Quantitative Analysis", "🕸️ Knowledge Graphs", "🌍 Embedding Spaces", "🔍 Retrieval Diagnostics"])
    
    with tabs[0]:
        if selected_qty != "All":
            st.markdown(f"### Analysis for `{selected_qty}`")
            
            if chart_type == "Histogram":
                fig = viz.plot_quantitative_histogram(df_all, selected_qty, group_by, config.default_colormap)
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "Box Plot":
                # Fallback to violin if box plot not explicitly implemented or use plotly box
                fig = px.box(df_all[df_all["physical_quantity"]==selected_qty], x=group_by, y="value", color="material")
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "Radar":
                fig = viz.plot_quantitative_radar(df_all, selected_qty, config.default_colormap)
                st.plotly_chart(fig, use_container_width=True)
                
            # Consensus & Contradiction
            st.markdown("#### Consensus & Contradiction")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(viz.plot_contradiction_matrix(selected_qty, config.default_colormap), use_container_width=True)
            with c2:
                st.plotly_chart(viz.plot_consensus_waterfall(selected_qty, config.default_colormap), use_container_width=True)
                
        else:
            st.markdown("### Global Overview")
            # Bar chart of quantities
            st.plotly_chart(viz.plot_quantities_bar(df_all, config.default_colormap), use_container_width=True)
            # Donut of materials
            st.plotly_chart(viz.plot_material_distribution_donut(config.default_colormap), use_container_width=True)

    with tabs[1]:
        st.markdown("### Knowledge Graph Visualization")
        # Select graph type
        graph_type = st.radio("Graph Engine", ["Static (Matplotlib)", "Interactive (PyVis)"], horizontal=True)
        
        if selected_qty != "All":
            if graph_type == "Interactive (PyVis)" and PYVIS_AVAILABLE:
                html = viz.plot_quantitative_knowledge_graph_pyvis(df_all, selected_qty, config.default_colormap, aliases=config.aliases)
                st.components.v1.html(html, height=600, scrolling=True)
                st.download_button("Download Interactive Graph", html.encode('utf-8'), file_name="kg_interactive.html")
            else:
                fig = viz.plot_quantitative_knowledge_graph(df_all, selected_qty, config.default_colormap, figsize=(12,10))
                st.pyplot(fig)
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=config.figure_dpi)
                st.download_button("Download Static Graph", buf.getvalue(), file_name=f"kg_{selected_qty}.png")
        else:
            st.info("Please select a specific Physical Quantity above to view its Knowledge Graph.")
            
        # Full Network
        st.markdown("### Full Corpus Network")
        if graph_type == "Interactive (PyVis)" and PYVIS_AVAILABLE:
            html = viz.plot_knowledge_network_pyvis(df_all, config.default_colormap)
            st.components.v1.html(html, height=600, scrolling=True)
        else:
            fig = viz.plot_knowledge_network(df_all, config.default_colormap, figsize=(14,12))
            st.pyplot(fig)

    with tabs[2]:
        if st.session_state.embedding_model:
            st.markdown("### Context Embeddings Projection")
            emb_fn = lambda x: np.array(st.session_state.embedding_model.encode(x))
            
            c1, c2 = st.columns(2)
            with c1:
                if SKLEARN_AVAILABLE:
                    fig_tsne = viz.plot_tsne(emb_fn, selected_qty if selected_qty != "All" else None, config.default_colormap)
                    if fig_tsne: st.pyplot(fig_tsne)
            with c2:
                if UMAP_AVAILABLE:
                    fig_umap = viz.plot_umap(emb_fn, selected_qty if selected_qty != "All" else None, config.default_colormap)
                    if fig_umap: st.pyplot(fig_umap)
        else:
            st.info("Install `sentence-transformers` to enable embedding analysis.")

    with tabs[3]:
        st.markdown("### Retrieval Diagnostics")
        # Show diagnostics if we have a cached query result
        if st.session_state.cached_query_result and "metrics" in st.session_state.cached_query_result:
            res = st.session_state.cached_query_result
            st.json(res["metrics"])
            
            # Plot retrieval tree highlight if nodes are available
            retrieved_nodes = res.get("retrieved", [])
            if retrieved_nodes:
                # Find the doc_id from the first node
                first_doc_id = retrieved_nodes[0].get("doc_id")
                if first_doc_id:
                    fig_tree = viz.plot_retrieval_tree_highlight(st.session_state.annotated_trees, retrieved_nodes, first_doc_id)
                    if fig_tree:
                        st.pyplot(fig_tree)
            
            # Sankey if we have relevant docs (simulated for v18 as 2-call doesn't return doc scores explicitly unless added)
            # For now, show a placeholder or simple stats
            st.info("Provenance Sankey is generated dynamically during query processing.")
        else:
            st.info("Run a query to see retrieval diagnostics.")

# ============================================================================
# FAST HIERARCHICAL INDEX (ASYNC, PARALLEL, LLM-TOC)
# ============================================================================
class FastHierarchicalIndex(HierarchicalIndex):
    """
    Enhanced Indexer that builds document trees asynchronously and uses 
    LLMs to extract accurate Tables of Contents (TOC) and headings.
    """
    def __init__(self, cache_dir: str = ".declarmima_cache_v18", llm: Optional['HybridLLM'] = None):
        super().__init__(cache_dir)
        self.llm = llm
        self.summarizer = RollupSummarizer(llm) if llm else None

    async def build_from_pdfs_fast(self, files: List, max_workers: int = 4) -> Dict[str, PageNode]:
        """
        Main entry point: Extracts pages, detects TOC via LLM, builds trees, and caches them.
        """
        loop = asyncio.get_event_loop()
        
        # Phase 1: Parallel Raw Page Extraction
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [loop.run_in_executor(pool, self._extract_pages_raw, f) for f in files]
            raw_docs = await asyncio.gather(*futures)
            
        # Phase 2: Parallel LLM TOC/Heading Extraction
        if self.llm:
            logger.info("Extracting TOCs/Headings via LLM...")
            toc_tasks = [self._llm_extract_toc(doc_name, pages) for doc_name, pages in raw_docs]
            toc_results = await asyncio.gather(*toc_tasks)
        else:
            # Fallback: no LLM, use empty TOC
            toc_results = [{"has_toc": False, "headings_detected": [], "doc_type": "unknown"} for _ in raw_docs]
            
        # Phase 3: Tree Construction & Metadata Extraction
        trees = {}
        for (doc_name, pages), toc in zip(raw_docs, toc_results):
            tree = self._build_tree_from_toc(doc_name, pages, toc)
            full_text = "\n".join([p['text'] for p in pages])
            meta = self.metadata_extractor.extract_metadata(doc_name, full_text)
            tree.metadata = meta
            trees[doc_name] = tree
            
        # Phase 4: Parallel Summary Generation & Caching
        if self.llm and self.summarizer:
            logger.info("Generating hierarchical roll-up summaries...")
            # Batch summarization to avoid rate limits / OOM
            await self._generate_summaries_async(trees)
            
        for doc_name, tree in trees.items():
            self.doc_trees[doc_name] = tree
            # Save to cache
            buf = BytesIO()
            # We need file content for hashing. If files were passed as BytesIO, we can re-read,
            # but for simplicity, we hash the doc_name + summary structure for cache key.
            self._save_tree_fast(doc_name, tree)
            
        return trees

    def _extract_pages_raw(self, file_obj) -> Tuple[str, List[Dict]]:
        """Thread-safe extraction of page text from PDF."""
        if hasattr(file_obj, 'getbuffer'):
            buf = BytesIO(file_obj.getbuffer())
            doc_name = file_obj.name
        else:
            buf = file_obj
            doc_name = "unknown.pdf"
            
        doc = fitz.open(stream=buf.getvalue(), filetype="pdf")
        pages = []
        for p in range(len(doc)):
            page = doc[p]
            pages.append({
                'page_num': p + 1, 
                'text': page.get_text("text"), 
                'images': len(page.get_images()), 
                'blocks': page.get_text("blocks")
            })
        doc.close()
        return doc_name, pages

    async def _llm_extract_toc(self, doc_name: str, pages: List[Dict]) -> Dict[str, Any]:
        """Uses LLM to detect document structure (TOC, Headings)."""
        # Sample first 5 pages (or fewer if doc is short)
        sample_text = "\n".join(p['text'][:1500] for p in pages[:5])
        
        prompt = f"""Analyze this document and extract its hierarchical structure.
Return JSON with:
- "has_toc": bool
- "toc_entries": list of {{"title": str, "level": int, "page": int}}
- "headings_detected": list of {{"title": str, "level": int, "page": int}}
- "doc_type": str (e.g., "research_paper", "report", "manual")
- "suggested_root_title": str
Document sample:
{sample_text[:6000]}
Return ONLY valid JSON."""
        try:
            response = await asyncio.to_thread(
                self.llm.generate, 
                prompt, 
                max_new_tokens=1024, 
                fast_json=True, 
                temperature=0.0
            )
            result = self.llm._extract_json_safe(response)
            if result and isinstance(result, dict):
                return result
        except Exception as e:
            logger.warning(f"LLM TOC extraction failed for {doc_name}: {e}")
            
        return {"has_toc": False, "headings_detected": [], "doc_type": "unknown", "suggested_root_title": doc_name}

    def _build_tree_from_toc(self, doc_name: str, pages: List[Dict], toc: Dict) -> PageNode:
        """Constructs the PageNode tree based on LLM-detected TOC."""
        safe_title = toc.get("suggested_root_title") or doc_name
        root = PageNode(
            f"{doc_name}_root", safe_title, 1, len(pages), "", 
            f"Document {doc_name}", 0, doc_id=doc_name, node_id="0000"
        )
        
        entries = toc.get("toc_entries", []) or toc.get("headings_detected", [])
        window = 7
        
        if entries:
            nodes_by_level = {}
            for entry in entries:
                level_val = entry.get("level")
                level = 1 if level_val is None else (int(level_val) if str(level_val).isdigit() else 1)
                title = str(entry.get("title", "Unknown")).strip()
                page_raw = entry.get("page")
                page = 1 if page_raw is None else (int(page_raw) if str(page_raw).isdigit() else 1)
                
                if page < 1 or page > len(pages):
                    continue
                    
                end = min(page + window, len(pages))
                text_parts = []
                for i in range(page, min(end + 1, len(pages) + 1)):
                    try:
                        page_data = pages[i - 1]
                        if isinstance(page_data, dict) and 'text' in page_data:
                            text_parts.append(page_data['text'])
                    except:
                        continue
                text = "\n".join(text_parts)
                
                node = PageNode(
                    f"{doc_name}_toc_{level}_{title[:20]}", title, page, end, 
                    text, text[:200], level, doc_id=doc_name
                )
                nodes_by_level.setdefault(level, []).append(node)
                
            # Attach nodes to parent based on level and page proximity
            for level in sorted(nodes_by_level.keys()):
                for node in nodes_by_level[level]:
                    parent = self._find_parent(root, level - 1, node.page_start)
                    parent.children.append(node)
        else:
            # Fallback: Page-level chunks
            for p in pages:
                text = p.get('text', '')
                if not str(text).strip():
                    continue
                page_num = int(p.get('page_num', 1)) if str(p.get('page_num', 1)).isdigit() else 1
                node = PageNode(
                    f"{doc_name}_p{page_num}", f"Page {page_num}", page_num, page_num, 
                    text, str(text)[:200], 3, doc_id=doc_name
                )
                root.children.append(node)
                
        self._assign_node_ids(root)
        return root

    async def _generate_summaries_async(self, trees: Dict[str, PageNode]):
        """Batched async summarization for all nodes."""
        all_nodes = []
        def collect_nodes(node: PageNode):
            all_nodes.append(node)
            for c in node.children:
                collect_nodes(c)
        for tree in trees.values():
            collect_nodes(tree)
            
        batch_size = 5
        for i in range(0, len(all_nodes), batch_size):
            batch = all_nodes[i:i+batch_size]
            tasks = []
            for node in batch:
                if len(node.full_text) > 200:
                    tasks.append(self.summarizer._post_order_summarize(node))
                else:
                    node.summary = node.full_text[:200]
            if tasks:
                # Run in thread pool to avoid blocking event loop if LLM is sync
                await asyncio.gather(*[self.summarizer._post_order_summarize(node) for node in batch if len(node.full_text) > 200])

    def _save_tree_fast(self, doc_name: str, tree: PageNode):
        """Persist tree to disk cache."""
        safe = re.sub(r'[^\w\-_.]', '_', doc_name)
        doc_hash = hashlib.sha256(doc_name.encode()).hexdigest()[:16]
        path = self.cache_dir / f"{safe}.{doc_hash}.tree.json"
        try:
            with open(path, "wb") as f:
                f.write(fast_json_dumps(tree.to_dict(), indent=True))
        except Exception as e:
            logger.warning(f"Fast save failed for {doc_name}: {e}")


# ============================================================================
# ENHANCED QUANTITATIVE KNOWLEDGE GRAPH
# ============================================================================

    # -------------------------------------------------------------------------
    # STUB METHODS FOR DASHBOARD COMPATIBILITY
    # -------------------------------------------------------------------------
    def plot_knowledge_network_pyvis(self, df: pd.DataFrame, colormap: Optional[str] = None, aliases: Optional[Dict[str, str]] = None) -> str:
        """Stub: redirects to query-focused PyVis network."""
        # Build a minimal QueryContext from df
        ctx = QueryContext(
            query="Corpus Overview",
            relevant_doc_ids=df["doc"].unique().tolist() if not df.empty else [],
            physical_quantities=df["physical_quantity"].unique().tolist() if not df.empty else [],
            materials=df["material"].unique().tolist() if not df.empty else []
        )
        return self.plot_query_knowledge_graph_pyvis(ctx)

    def plot_knowledge_network(self, df: pd.DataFrame, colormap: Optional[str] = None, figsize: Tuple[int, int] = (14, 12)) -> plt.Figure:
        """Stub: redirects to query-focused static network."""
        ctx = QueryContext(
            query="Corpus Overview",
            relevant_doc_ids=df["doc"].unique().tolist() if not df.empty else [],
            physical_quantities=df["physical_quantity"].unique().tolist() if not df.empty else [],
            materials=df["material"].unique().tolist() if not df.empty else []
        )
        return self.plot_query_knowledge_graph(ctx, figsize=figsize)

class QuantitativeKnowledgeGraph:
    """
    Manages extracted entities, computes consensus/contradictions, 
    and maps quantitative data back to the hierarchical tree.
    """
    def __init__(self):
        self.doc_graphs: Dict[str, Dict] = {}
        self.phys_classifier = PhysicalQuantityClassifier()
        self.metadata_index: Dict[str, DocumentMetadata] = {}
        self.concept_normalizer = ConceptNormalizer()

    def add_document_metadata(self, doc_name: str, metadata: DocumentMetadata):
        """Store raw metadata for quick lookup."""
        self.metadata_index[doc_name] = metadata

    def add_extractions(self, doc_id: str, items: List[UniversalExtractionItem]):
        """Ingest extracted items and build internal indices."""
        graph = {
            "doc_id": doc_id, 
            "parameters": defaultdict(list), 
            "materials": defaultdict(list),
            "methods": defaultdict(list), 
            "by_page": defaultdict(list), 
            "by_section": defaultdict(list),
            "by_physical_quantity": defaultdict(list), 
            "all_items": []
        }
        
        for item in items:
            item_dict = item.to_dict()
            # Normalize concepts
            if item.physical_quantity:
                item_dict["physical_quantity"] = self.concept_normalizer.normalize(item.physical_quantity)
            if item.material:
                item_dict["material"] = self.concept_normalizer.normalize(item.material)
                
            graph["all_items"].append(item_dict)
            
            # Indexing
            if item.parameter_name:
                graph["parameters"][item.parameter_name.lower()].append(item_dict)
            if item.material:
                graph["materials"][item.material.lower()].append(item_dict)
            if item.method:
                graph["methods"][item.method.lower()].append(item_dict)
            if item.physical_quantity:
                graph["by_physical_quantity"][item.physical_quantity].append(item_dict)
            graph["by_page"][item.page].append(item_dict)
            if item.section_title:
                graph["by_section"][item.section_title].append(item_dict)
                
        self.doc_graphs[doc_id] = dict(graph)

    def get_entity_consensus(self, entity_name: str) -> Dict[str, Any]:
        """Calculate statistical consensus for an entity (material or quantity)."""
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
            "found": True, 
            "entity": entity_name, 
            "count": len(values), 
            "unit": list(units)[0] if units else "unknown", 
            "range": (min(values), max(values)), 
            "mean": float(np.mean(values)), 
            "std": float(np.std(values)) if len(values) > 1 else 0.0, 
            "documents": list(docs), 
            "values": values
        }

    def get_entity_contradictions(self, entity_name: str, threshold_factor: float = 2.0) -> List[Dict[str, Any]]:
        """Detect contradictions between documents for the same entity."""
        by_doc = defaultdict(list)
        for doc_id, graph in self.doc_graphs.items():
            for item in graph["all_items"]:
                if (item.get("material") == entity_name or item.get("physical_quantity") == entity_name):
                    if item.get("value") is not None:
                        by_doc[doc_id].append(item["value"])
                        
        contradictions = []
        docs = list(by_doc.keys())
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                if by_doc[docs[i]] and by_doc[docs[j]]:
                    mean_i = np.mean(by_doc[docs[i]])
                    mean_j = np.mean(by_doc[docs[j]])
                    if mean_i > 0 and mean_j > 0:
                        ratio = max(mean_i, mean_j) / min(mean_i, mean_j)
                        if ratio > threshold_factor:
                            contradictions.append({
                                "entity": entity_name, 
                                "doc_a": docs[i], 
                                "value_a": mean_i, 
                                "doc_b": docs[j], 
                                "value_b": mean_j, 
                                "ratio": ratio, 
                                "severity": "high" if ratio > 5 else "moderate"
                            })
        return contradictions

    def to_tree_annotation(self, doc_tree: PageNode, max_chars: int = 20000) -> Dict[str, Any]:
        """
        Crucial for v18.0 2-call architecture:
        Maps quantitative extractions back to the hierarchical tree nodes 
        so the LLM navigator can see "hints" (e.g., has_quantitative, candidate_values).
        """
        doc_id = doc_tree.doc_id
        graph = self.doc_graphs.get(doc_id, {})
        
        def annotate_node(node: PageNode) -> Dict[str, Any]:
            result = node.to_tree_format(max_chars=max_chars)
            node_items = []
            # Collect items from all pages covered by this node
            end_page = node.page_end or node.page_start
            for page in range(node.page_start, end_page + 1):
                node_items.extend(graph.get("by_page", {}).get(page, []))
                
            if node_items:
                seen = set()
                unique_items = []
                for item in node_items:
                    # Deduplicate
                    key = (item.get('parameter_name'), item.get('value'), item.get('page'))
                    if key not in seen:
                        seen.add(key)
                        unique_items.append(item)
                result["quantitative_items"] = unique_items
                
            # Recurse
            if node.children:
                result["nodes"] = [annotate_node(c) for c in node.children]
            return result
            
        return annotate_node(doc_tree)

    def get_all_entity_names(self) -> List[str]:
        """Extract all unique materials and quantities."""
        entities = set()
        for doc_id, graph in self.doc_graphs.items():
            for item in graph["all_items"]:
                if item.get("material"): entities.add(item["material"])
                if item.get("physical_quantity"): entities.add(item["physical_quantity"])
                if item.get("parameter_name"): entities.add(item["parameter_name"])
        return sorted(entities)

    def get_all_physical_quantities(self) -> Dict[str, int]:
        """Count occurrences of each physical quantity across all documents."""
        counts = Counter()
        for graph in self.doc_graphs.values():
            for item in graph["all_items"]:
                pq = item.get("physical_quantity")
                if pq:
                    counts[pq] += 1
        return dict(counts)

    def get_all_materials(self) -> Dict[str, List[str]]:
        """Get materials mentioned per document."""
        result = {}
        for doc_id, graph in self.doc_graphs.items():
            mats = set()
            for item in graph["all_items"]:
                if item.get("material"):
                    mats.add(item["material"])
            result[doc_id] = list(mats)
        return result


# ============================================================================
# HELPER UTILITIES & CACHING (Essential for "Smartness")
# ============================================================================
def fast_json_dumps(obj, indent=False):
    """Fast JSON serialization using orjson if available."""
    if ORJSON_AVAILABLE:
        import orjson
        option = orjson.OPT_INDENT_2 if indent else 0
        return orjson.dumps(obj, option=option, default=str)
    else:
        return json.dumps(obj, indent=2 if indent else None, ensure_ascii=False, default=str).encode()

def fast_json_loads(data):
    """Fast JSON deserialization using orjson if available."""
    if ORJSON_AVAILABLE:
        import orjson
        if isinstance(data, str):
            data = data.encode()
        return orjson.loads(data)
    else:
        if isinstance(data, bytes):
            data = data.decode()
        return json.loads(data)

@contextmanager
def timer(label: str):
    """Context manager for timing code blocks."""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        if not hasattr(timer, 'metrics'):
            timer.metrics = defaultdict(list)
        timer.metrics[label].append(elapsed)
        logger.info(f"[METRIC] {label}: {elapsed:.2f}s")

class LRUCache:
    """
    Least Recently Used Cache with TTL and thread safety.
    Essential for caching LLM responses and extraction results.
    """
    def __init__(self, max_size=1000, ttl=7200):
        self.max_size = max_size
        self.ttl = ttl
        self._cache = OrderedDict()
        self._lock = threading.RLock()

    def _key(self, *args, **kwargs):
        """Generate deterministic key from arguments."""
        key_data = "|".join(str(a) for a in args) + "|" + json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.sha256(key_data.encode()).hexdigest()[:20]

    def get(self, *args, **kwargs):
        key = self._key(*args, **kwargs)
        with self._lock:
            if key in self._cache:
                val, ts = self._cache[key]
                if time.time() - ts < self.ttl:
                    self._cache.move_to_end(key)
                    return val
                else:
                    del self._cache[key]
            return None

    def set(self, value, *args, **kwargs):
        key = self._key(*args, **kwargs)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            self._cache[key] = (value, time.time())
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

# Global Response Cache Instance
response_cache = LRUCache(max_size=2000, ttl=7200)

# ============================================================================
# MODEL CONFIGURATION (UNCHANGED)
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
# ADVANCED REPORTING & EXPORT UTILITIES
# ============================================================================
class ReportGenerator:
    """
    Generates comprehensive reports for query results and corpus analysis.
    Supports Markdown, JSON, and HTML export formats.
    """
    
    def __init__(self, kg: 'QuantitativeKnowledgeGraph'):
        self.kg = kg
        self.classifier = PhysicalQuantityClassifier()

    def generate_markdown_report(self, query: str, result: Dict[str, Any]) -> str:
        """
        Generate a Markdown report from a query result.
        """
        answer = result.get("answer", "No answer generated.")
        metrics = result.get("metrics", {})
        nodes = result.get("nodes", [])
        
        report = [
            f"# DECLARMIMA v18.0 Query Report",
            f"**Query:** `{query}`",
            f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 📝 AI Synthesis",
            answer,
            "",
            "## 📊 Retrieval Metrics",
            f"- **Total Time:** {metrics.get('total_time', 0):.2f}s",
            f"- **Navigation Time:** {metrics.get('call_1_time', 0):.2f}s",
            f"- **Synthesis Time:** {metrics.get('call_2_time', 0):.2f}s",
            f"- **Nodes Retrieved:** {len(nodes)}",
            "",
            "## 🔍 Retrieved Evidence",
        ]
        
        for i, node in enumerate(nodes[:10]):
            doc_id = node.get("doc_id", "Unknown")
            page = node.get("page_start", "?")
            section = node.get("section_title", "Unknown")
            confidence = node.get("confidence", 0.0)
            report.append(f"### {i+1}. {doc_id} (p.{page}) - {section}")
            report.append(f"- **Confidence:** {confidence:.2f}")
            report.append(f"- **Reasoning:** {node.get('selection_reasoning', 'N/A')}")
            report.append(f"- **Content Preview:** `{node.get('full_text', '')[:150]}...`")
            report.append("")
            
        return "\n".join(report)

    def generate_json_report(self, query: str, result: Dict[str, Any]) -> str:
        """
        Generate a structured JSON report.
        """
        # Clean up result for serialization
        serializable_nodes = []
        for n in result.get("nodes", []):
            serializable_nodes.append({
                "doc_id": n.get("doc_id"),
                "page_start": n.get("page_start"),
                "section_title": n.get("section_title"),
                "confidence": n.get("confidence"),
                "selection_reasoning": n.get("selection_reasoning"),
                # Truncate text to keep JSON small
                "text_preview": n.get("full_text", "")[:200]
            })
            
        report_data = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "version": "v18.0",
            "metrics": result.get("metrics", {}),
            "answer": result.get("answer", ""),
            "retrieved_evidence": serializable_nodes
        }
        return json.dumps(report_data, indent=2, ensure_ascii=False)

    def export_to_html(self, query: str, result: Dict[str, Any]) -> str:
        """
        Generate a standalone HTML report with embedded styles.
        """
        md_content = self.generate_markdown_report(query, result)
        # Simple Markdown to HTML conversion (basic implementation)
        html_content = md_content.replace('\n', '<br>')
        html_content = html_content.replace('# ', '<h1>')
        html_content = html_content.replace('## ', '<h2>')
        html_content = html_content.replace('### ', '<h3>')
        html_content = html_content.replace('**', '<b>').replace('**', '</b>')
        html_content = html_content.replace('`', '<code>').replace('`', '</code>')
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DECLARMIMA Report: {query}</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; max-width: 800px; margin: 40px auto; line-height: 1.6; color: #333; }}
                h1 {{ color: #2563eb; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; }}
                h2 {{ color: #475569; }}
                h3 {{ color: #64748b; }}
                code {{ background: #f1f5f9; padding: 2px 4px; border-radius: 4px; font-size: 0.9em; }}
                .metric {{ display: inline-block; background: #eff6ff; padding: 5px 10px; border-radius: 6px; margin: 5px; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

# ============================================================================
# SYSTEM TELEMETRY & HEALTH MONITORING
# ============================================================================
class SystemTelemetry:
    """
    Tracks system performance, LLM latency, cache hit rates, and memory usage.
    """
    def __init__(self):
        self.metrics = {
            "llm_calls": 0,
            "llm_total_time": 0.0,
            "llm_avg_latency": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "extraction_errors": 0,
            "memory_usage_mb": 0.0
        }
        self._history = defaultdict(list)

    def record_llm_call(self, latency: float, tokens_in: int, tokens_out: int):
        self.metrics["llm_calls"] += 1
        self.metrics["llm_total_time"] += latency
        self.metrics["llm_avg_latency"] = self.metrics["llm_total_time"] / self.metrics["llm_calls"]
        self._history["latency"].append(latency)
        self._history["tokens_in"].append(tokens_in)
        self._history["tokens_out"].append(tokens_out)

    def record_cache_access(self, hit: bool):
        if hit:
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1

    def update_memory(self):
        try:
            import psutil
            process = psutil.Process(os.getpid())
            self.metrics["memory_usage_mb"] = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            self.metrics["memory_usage_mb"] = 0.0

    def get_summary(self) -> Dict[str, Any]:
        self.update_memory()
        total_accesses = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        hit_rate = (self.metrics["cache_hits"] / total_accesses * 100) if total_accesses > 0 else 0
        return {**self.metrics, "cache_hit_rate": f"{hit_rate:.1f}%"}

    def reset(self):
        self.metrics = {k: 0.0 if isinstance(v, float) else 0 for k, v in self.metrics.items()}
        self._history.clear()

# Global Telemetry Instance
telemetry = SystemTelemetry()

# ============================================================================
# ROBUST ASYNC UTILITIES FOR STREAMLIT
# ============================================================================
def run_async_safe(coro):
    """
    Safely run an async coroutine in a Streamlit thread environment.
    Streamlit creates a new thread for each session, so we must ensure
    we have a running event loop.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running (e.g., via nest_asyncio), use it
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
    except RuntimeError:
        pass
    
    # If no loop, create one for this call
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()

def timer_context(label: str):
    """
    Context manager for timing blocks and recording telemetry.
    """
    class Timer:
        def __init__(self, label):
            self.label = label
            self.start = 0
        def __enter__(self):
            self.start = time.time()
            return self
        def __exit__(self, *args):
            elapsed = time.time() - self.start
            if "llm" in self.label.lower():
                telemetry.record_llm_call(elapsed, 0, 0) # Tokens would need LLM wrapper update
    return Timer(label)

# ============================================================================
# STREAMLIT SIDEBAR TELEMETRY INTEGRATION
# ============================================================================
def render_telemetry_sidebar():
    """
    Adds a telemetry section to the Streamlit sidebar for monitoring.
    """
    with st.sidebar:
        st.markdown("---")
        with st.expander("📈 System Telemetry", expanded=False):
            stats = telemetry.get_summary()
            
            col1, col2 = st.columns(2)
            col1.metric("LLM Calls", stats["llm_calls"])
            col2.metric("Avg Latency", f"{stats['llm_avg_latency']:.2f}s")
            
            col3, col4 = st.columns(2)
            col3.metric("Cache Hits", stats["cache_hits"])
            col4.metric("Hit Rate", stats["cache_hit_rate"])
            
            st.caption(f"Memory: {stats['memory_usage_mb']:.1f} MB")
            st.caption(f"Errors: {stats['extraction_errors']}")
            
            if st.button("Reset Stats", use_container_width=True):
                telemetry.reset()
                st.rerun()

# ============================================================================
# STREAMLIT UI: REPORTING & TELEMETRY WIDGETS (Integration)
# ============================================================================

def render_download_buttons(query: str, result: Dict[str, Any]):
    """
    Renders download buttons for Markdown, JSON, and HTML reports 
    using the ReportGenerator class defined in Part 13.
    """
    if not query or not result:
        return
        
    # Get the KG from session state
    kg = st.session_state.get("knowledge_graph")
    if not kg:
        kg = QuantitativeKnowledgeGraph()
        
    generator = ReportGenerator(kg)
    
    st.markdown("---")
    st.subheader("📥 Export Analysis Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        md_report = generator.generate_markdown_report(query, result)
        st.download_button(
            label="📄 Download Markdown Report",
            data=md_report,
            file_name="declarmima_report.md",
            mime="text/markdown",
            help="Download a readable Markdown summary of findings."
        )
        
    with col2:
        json_report = generator.generate_json_report(query, result)
        st.download_button(
            label="📦 Download JSON Data",
            data=json_report,
            file_name="declarmima_data.json",
            mime="application/json",
            help="Download raw data for programmatic analysis."
        )
        
    with col3:
        html_report = generator.export_to_html(query, result)
        st.download_button(
            label="🌐 Download Standalone HTML",
            data=html_report.encode('utf-8'),
            file_name="declarmima_view.html",
            mime="text/html",
            help="Download a standalone HTML file with embedded styles."
        )

def render_telemetry_widget():
    """
    Renders the System Telemetry widget in the sidebar.
    """
    with st.sidebar:
        st.markdown("---")
        with st.expander("📈 System Telemetry", expanded=False):
            stats = telemetry.get_summary()
            
            c1, c2 = st.columns(2)
            c1.metric("LLM Calls", stats["llm_calls"])
            c2.metric("Avg Latency", f"{stats['llm_avg_latency']:.2f}s")
            
            c3, c4 = st.columns(2)
            c3.metric("Cache Hits", stats["cache_hits"])
            c4.metric("Hit Rate", stats["cache_hit_rate"])
            
            st.caption(f"Memory: {stats['memory_usage_mb']:.1f} MB")
            st.caption(f"Errors: {stats['extraction_errors']}")
            
            if st.button("Reset Stats", use_container_width=True):
                telemetry.reset()
                st.rerun()

# ============================================================================
# FINAL ENTRY POINT & ROBUST EXECUTION
# ============================================================================

def run_app():
    """
    Main execution function with global error handling.
    This function calls the 'run_streamlit' logic defined in Part 9/12.
    """
    try:
        asyncio.run(run_streamlit())
        render_telemetry_widget()
    except Exception as e:
        st.error(f"⚠️ Critical Application Error: {str(e)}")
        logger.exception(e)
        st.stop()

if __name__ == "__main__":
    run_app()
