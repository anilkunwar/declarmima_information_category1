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
    processing_meta Dict[str, Any] = {}
    
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

    def extract_metadata(self, doc_name: str, full_text: str) -> DocumentMeta
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
                self.embedding_model = SentenceTransformer(embedding_model, device="cpu")
                logger.info(f"Loaded sentence-transformer model {embedding_model} on CPU")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
        
        self.doc_metadata: Dict[str, DocumentMetadata] = {}
        self.doc_summaries: Dict[str, str] = {}

    def index_document(self, doc_name: str, meta DocumentMetadata, summary: str):
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
