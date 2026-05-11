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
