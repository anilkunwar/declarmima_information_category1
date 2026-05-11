#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DECLARMIMA v17.2 - UNIFIED ROBUST VECTORLESS RAG WITH QUERY-DRIVEN VISUALIZATION
================================================================================
Merged strengths from v13.0+, v16.0, and v13.3 without weaknesses:
- Comprehensive physical quantity classifier
- Vectorless retrieval with robust semantic fallback
- Fixed networkx "None cannot be a node" error
- Fixed sunburst hierarchy error
- Full visualization suite: 35+ chart types
- Query-driven contextual visualization layer
- Retrieval diagnostics dashboard
- Concept normalization & synonym resolution
- Enhanced structured metadata extraction
- Comprehensive Ollama model catalog
- Safe hierarchical tree building
- LRU cache, timing utilities, fast JSON serialization
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
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any, Set, Callable, Literal
from collections import defaultdict, Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import numpy as np
import torch
import threading
import queue
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logging.basicConfig(level=logging.INFO, handlers=[console_handler], force=True)
logger = logging.getLogger("DECLARMIMA")

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    raise ImportError("PyMuPDF (fitz) required: pip install pymupdf")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not installed. Ollama backend unavailable.")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False
    logger.warning("orjson not installed. Using standard json (slower).")

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Using vectorless keyword retrieval.")

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


from pydantic import BaseModel, Field, field_validator

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
    model_config = ConfigDict(extra='allow')

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
    indentation_force_values: List[float] = Field(default_factory=list)
    indentation_depth_values: List[float] = Field(default_factory=list)
    dislocation_density_values: List[float] = Field(default_factory=list)
    hatch_distance_values: List[float] = Field(default_factory=list)
    build_platform_temperature_values: List[float] = Field(default_factory=list)
    melt_pool_depth_values: List[float] = Field(default_factory=list)
    melt_pool_width_values: List[float] = Field(default_factory=list)
    melt_pool_length_values: List[float] = Field(default_factory=list)
    vec_values: List[float] = Field(default_factory=list)
    lewis_number_values: List[float] = Field(default_factory=list)
    omega_parameter_values: List[float] = Field(default_factory=list)
    delta_h_mix_values: List[float] = Field(default_factory=list)
    delta_s_mix_values: List[float] = Field(default_factory=list)
    atomic_size_difference_values: List[float] = Field(default_factory=list)
    lambda_parameter_values: List[float] = Field(default_factory=list)
    jackson_parameter_values: List[float] = Field(default_factory=list)
    other_parameters: Dict[str, List[float]] = {}


# =============================================================================
# QUERY CONTEXT FOR DRIVEN VISUALIZATIONS
# =============================================================================
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


class PhysicalQuantityClassifier:
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

    def __init__(self):
        self._build_keyword_index()

    def _build_keyword_index(self):
        self.keyword_to_canonical = {}
        for canonical, keywords in self.CANONICAL.items():
            for kw in keywords:
                self.keyword_to_canonical[kw.lower()] = canonical
        self.keyword_to_canonical["ys"] = "yield_strength"
        self.keyword_to_canonical["uts"] = "tensile_strength"
        self.keyword_to_canonical["smys"] = "yield_strength"
        self.keyword_to_canonical["0.2% proof"] = "yield_strength"
        self.keyword_to_canonical["ecorr"] = "corrosion_potential"
        self.keyword_to_canonical["eocp"] = "open_circuit_potential"
        self.keyword_to_canonical["erp"] = "repassivation_potential"
        self.keyword_to_canonical["epit"] = "pitting_potential"
        self.keyword_to_canonical["ebr"] = "breakdown_potential"
        self.keyword_to_canonical["jcorr"] = "corrosion_current_density"
        self.keyword_to_canonical["rp"] = "polarization_resistance"
        self.keyword_to_canonical["pren"] = "PREN"
        self.keyword_to_canonical["sfe"] = "stacking_fault_energy"
        self.keyword_to_canonical["usfe"] = "unstable_stacking_fault_energy"
        self.keyword_to_canonical["smd"] = "sauter_mean_diameter"
        self.keyword_to_canonical["ved"] = "energy_density"
        self.keyword_to_canonical["aed"] = "areal_energy_density"
        self.keyword_to_canonical["led"] = "linear_energy_density"
        self.keyword_to_canonical["le"] = "lewis_number"

        self.keyword_to_canonical["alpha_j"] = "jackson_parameter"

        self.keyword_to_canonical["vec"] = "vec"

        self.keyword_to_canonical["dh_mix"] = "delta_h_mix"

        self.keyword_to_canonical["ds_mix"] = "delta_s_mix"

        self.keyword_to_canonical["omega"] = "omega_parameter"


    def classify(self, parameter_name: Optional[str], unit: Optional[str], context: str) -> str:
        if parameter_name:
            pname_lower = parameter_name.lower().strip()
            for canonical, keywords in self.CANONICAL.items():
                for kw in keywords:
                    if kw in pname_lower:
                        return canonical
            if pname_lower in self.keyword_to_canonical:
                return self.keyword_to_canonical[pname_lower]
        context_lower = context.lower()
        for canonical, keywords in self.CANONICAL.items():
            for kw in keywords:
                if kw in context_lower:
                    return canonical
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


class ConceptNormalizer:
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
        if use_fuzzy and RAPIDFUZZ_AVAILABLE:
            all_aliases = list(self.alias_to_canonical.keys())
            result = process.extractOne(term_lower, all_aliases, scorer=fuzz.ratio)
            if result and result[1] >= fuzzy_threshold:
                return self.alias_to_canonical[result[0]]
        if self.embedding_fn is not None:
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
# DISPLAY NAME HELPERS (DOI postprocessing + user aliases)
# ============================================================================
def normalize_doi_display(name: str) -> str:
    """Convert filesystem-safe DOI filenames back to real DOI format.
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
    """Return human-readable display name for a document.
    Priority: 1) user alias, 2) DOI-normalized stem, 3) original stem.
    """
    if aliases and doc_id in aliases:
        return aliases[doc_id]
    stem = Path(doc_id).stem
    normalized = normalize_doi_display(stem)
    return normalized


def get_citation_label(doc_id: str, aliases: Optional[Dict[str, str]] = None, index: int = 0, style: str = "doi") -> str:
    """Generate citation-style label for a document.
    style: 'doi' -> normalized DOI, 'number' -> [1], 'alias' -> user alias, 'short' -> first 20 chars.
    """
    if style == "alias" and aliases and doc_id in aliases:
        return aliases[doc_id]
    if style == "number":
        return f"[{index}]"
    if style == "short":
        return Path(doc_id).stem[:20]
    return normalize_doi_display(Path(doc_id).stem)

class PaginationAwareReader:
    def __init__(self, max_chars_per_request=20000):
        self.max_chars_per_request = max_chars_per_request

    def extract_pages(self, doc_path: str, page_numbers: List[int]) -> Dict[int, str]:
        doc = fitz.open(doc_path)
        result = {}
        for pnum in page_numbers:
            if pnum < 1 or pnum > len(doc):
                continue
            page = doc[pnum-1]
            text = page.get_text("text")
            if len(text) > self.max_chars_per_request:
                logger.warning(f"Page {pnum} text length {len(text)} exceeds limit, truncating.")
                text = text[:self.max_chars_per_request] + "\n...[TRUNCATED]"
            result[pnum] = text
        doc.close()
        return result

    def extract_page_range(self, doc_path: str, start: int, end: int, step=1) -> Dict[int, str]:
        pages = list(range(start, end+1, step))
        return self.extract_pages(doc_path, pages)


class StructuredMetadataExtractor:
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
        meta = DocumentMetadata(doc_name=doc_name)
        alloys_set = set()
        for regex in self.alloy_regexes:
            for match in regex.finditer(full_text):
                candidate = match.group(0).strip()
                if len(candidate) > 2 and candidate.lower() not in ["alloy", "composite", "metal"]:
                    alloys_set.add(candidate)
        meta.alloys = list(alloys_set)
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


class TwoStageRetriever:
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

    def index_document(self, doc_name: str, metadata: DocumentMetadata, summary: str):
        self.doc_metadata[doc_name] = metadata
        self.doc_summaries[doc_name] = summary

    def retrieve_relevant_docs(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        scores = []
        query_lower = query.lower()
        for name, meta in self.doc_metadata.items():
            score = 0.0
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
            # Electrochemical
            if any(t in query_lower for t in ["corrosion", "pitting", "repassivation", "polarization", "eis", "cpp"]):
                if meta.corrosion_potential_values or meta.polarization_resistance_values:
                    score += 0.6
            if "current density" in query_lower and meta.corrosion_current_density_values:
                score += 0.5
            if "pren" in query_lower and meta.pren_values:
                score += 0.5
            # Microstructural
            if any(t in query_lower for t in ["austenite", "ferrite", "phase fraction"]):
                if meta.phase_fraction_values or meta.austenite_fraction_values or meta.ferrite_fraction_values:
                    score += 0.5
            if "grain size" in query_lower and meta.grain_size_values:
                score += 0.4
            if "porosity" in query_lower and meta.porosity_values:
                score += 0.4
            if "relative density" in query_lower and meta.relative_density_values:
                score += 0.4
            # Thermal / physical
            if "thermal conductivity" in query_lower and meta.thermal_conductivity_values:
                score += 0.4
            if "viscosity" in query_lower and meta.viscosity_values:
                score += 0.4
            if "density" in query_lower and meta.density_values:
                score += 0.3
            if "enthalpy" in query_lower and meta.enthalpy_values:
                score += 0.4
            # Spray / fluid
            if any(t in query_lower for t in ["smd", "sauter", "droplet", "spray"]):
                if meta.sauter_mean_diameter_values:
                    score += 0.5
            # SFE
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
            # NEW v17.2 scoring
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

                if meta.linear_energy_density_values:
                    score += 0.5
            for proc in meta.process_types:
                if proc.lower() in query_lower:
                    score += 0.2
            scores.append((name, min(score, 1.0)))



            try:
                doc_texts = [f"{meta.alloys} {meta.process_types} {self.doc_summaries.get(name, '')}"
                             for name, meta in self.doc_metadata.items()]
                if doc_texts:
                    doc_emb = self.embedding_model.encode(doc_texts, convert_to_tensor=True)
                    query_emb = self.embedding_model.encode(query, convert_to_tensor=True)
                    sem_scores = util.cos_sim(query_emb, doc_emb)[0]
                    for i, (name, kw_score) in enumerate(scores):
                        sem_score = float(sem_scores[i])
                        scores[i] = (name, min(kw_score * 0.6 + sem_score * 0.4, 1.0))
            except Exception as e:
                logger.warning(f"Semantic blending failed: {e}")
        scores.sort(key=lambda x: x[1], reverse=True)
        if not any(s[1] > 0 for s in scores):
            return [(name, 0.2) for name in self.doc_metadata.keys()][:top_k]
        return scores[:top_k]

    def get_relevant_pages(self, doc_name: str, query: str, max_pages: int = 5) -> List[int]:
        return list(range(1, max_pages+1))


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
    metadata: Optional[DocumentMetadata] = None

    def get_text(self, doc_cache: Dict[str, Any] = None, max_chars: int = 20000) -> str:
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
        texts = [doc[p].get_text("text") for p in range(start, end)]
        self.full_text = "\n\n".join(texts)
        if doc_cache is None:
            doc.close()
        return self.full_text[:max_chars] if len(self.full_text) > max_chars else self.full_text

    def to_dict(self):
        return {
            "id": self.id, "title": self.title, "page_start": self.page_start, "page_end": self.page_end,
            "summary": self.summary, "prefix_summary": self.prefix_summary, "level": self.level,
            "doc_id": self.doc_id, "section_type": self.section_type, "node_id": self.node_id,
            "text_token_count": self.text_token_count, "children": [c.to_dict() for c in self.children],
            "metadata": self.metadata.dict() if self.metadata else None
        }

    def to_tree_format(self, max_chars: int = 20000) -> Dict[str, Any]:
        result = {"title": self.title, "node_id": self.node_id, "start_index": self.page_start,
                  "end_index": self.page_end or self.page_start, "summary": self.summary,
                  "prefix_summary": self.prefix_summary, "text_token_count": self.text_token_count}
        if self.children:
            result["nodes"] = [c.to_tree_format(max_chars) for c in self.children]
        text = self.get_text(max_chars=max_chars)
        if text:
            result["text"] = text
        if self.metadata:
            result["metadata"] = self.metadata.dict()
        return result

    @classmethod
    def from_dict(cls, data: dict, pdf_path=None):
        node = cls(data["id"], data["title"], data["page_start"], data.get("page_end"), "",
                   data.get("summary", ""), data.get("level", 0), doc_id=data.get("doc_id", ""),
                   section_type=data.get("section_type", "BODY"), _pdf_path=pdf_path)
        node.node_id = data.get("node_id", "")
        node.prefix_summary = data.get("prefix_summary", "")
        node.text_token_count = data.get("text_token_count", 0)
        for c in data.get("children", []):
            node.children.append(cls.from_dict(c, pdf_path))
        if data.get("metadata"):
            node.metadata = DocumentMetadata(**data["metadata"])
        return node


class HierarchicalIndex:
    def __init__(self, cache_dir=".declarmima_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.doc_trees: Dict[str, PageNode] = {}
        self._pdf_cache = {}
        self.metadata_extractor = StructuredMetadataExtractor()

    def _doc_hash(self, file_buffer: BytesIO) -> str:
        pos = file_buffer.tell()
        file_buffer.seek(0)
        content = file_buffer.read(1024 * 1024)
        file_buffer.seek(pos)
        return hashlib.sha256(content).hexdigest()[:16]

    def _cache_path(self, doc_name: str, doc_hash: str) -> Path:
        safe = re.sub(r'[^\w\-_.]', '_', doc_name)
        return self.cache_dir / f"{safe}.{doc_hash}.tree.json"

    def build_from_pdfs(self, files: List, parallel=True, max_workers=4):
        def build_one(file):
            doc_name = file.name
            buf = BytesIO(file.getbuffer())
            doc_hash = self._doc_hash(buf)
            cache_path = self._cache_path(doc_name, doc_hash)
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        root_data = fast_json_loads(f.read())
                    root = PageNode.from_dict(root_data)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        buf.seek(0)
                        tmp.write(buf.getbuffer())
                        root._pdf_path = tmp.name
                    return doc_name, root
                except Exception as e:
                    logger.warning(f"Cache load failed for {doc_name}: {e}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                buf.seek(0)
                tmp.write(buf.getbuffer())
                tmp_path = tmp.name
            doc = fitz.open(tmp_path)
            root = self._build_tree(doc, doc_name, tmp_path)
            full_text = "\n".join([doc[p].get_text("text") for p in range(len(doc))])
            meta = self.metadata_extractor.extract_metadata(doc_name, full_text)
            root.metadata = meta
            doc.close()
            try:
                cache_root = self._clone_for_cache(root)
                with open(cache_path, "wb") as f:
                    f.write(fast_json_dumps(cache_root.to_dict(), indent=True))
            except Exception as e:
                logger.warning(f"Cache save failed: {e}")
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

    def _build_tree(self, doc, doc_id, pdf_path):
        root = PageNode(f"{doc_id}_root", "Document Root", 1, len(doc), "",
                        f"Document {doc_id} root covering pages 1-{len(doc)}", 0, doc_id=doc_id, _pdf_path=pdf_path, node_id="0000")
        toc = doc.get_toc()
        window = 7
        if toc:
            nodes_by_level = {}
            for level, title, page in toc:
                if page > len(doc):
                    continue
                end = min(page + window, len(doc))
                text = self._extract_range(doc, page, end)
                node = PageNode(f"{doc_id}_toc_{level}_{title[:20]}", title.strip(), page, end, text, text[:200], level, doc_id=doc_id, _pdf_path=pdf_path)
                nodes_by_level.setdefault(level, []).append(node)
            for level in sorted(nodes_by_level.keys()):
                for node in nodes_by_level[level]:
                    parent = self._find_parent(root, level-1, node.page_start)
                    parent.children.append(node)
            self._assign_node_ids(root)
            return root
        headings = self._detect_headings(doc)
        if headings:
            for i, (title, page) in enumerate(headings):
                end = min(page + window, len(doc))
                text = self._extract_range(doc, page, end)
                node = PageNode(f"{doc_id}_h{i}", title, page, end, text, text[:200], 2, doc_id=doc_id, _pdf_path=pdf_path)
                root.children.append(node)
            self._assign_node_ids(root)
            return root
        for p in range(1, len(doc)+1):
            text = doc[p-1].get_text("text")
            if not text.strip():
                continue
            node = PageNode(f"{doc_id}_p{p}", f"Page {p}", p, p, text, text[:200], 3, doc_id=doc_id, _pdf_path=pdf_path)
            root.children.append(node)
        self._assign_node_ids(root)
        return root

    def _extract_range(self, doc, start, end):
        return "\n\n".join(doc[p-1].get_text("text") for p in range(start, min(end, len(doc)+1)))

    def _detect_headings(self, doc):
        headings = []
        for p in range(len(doc)):
            lines = doc[p].get_text("text").split('\n')
            for line in lines:
                if re.match(r'^(?:[0-9]+\.?)+ +[A-Z]', line.strip()):
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

    def _clone_for_cache(self, node):
        return PageNode(node.id, node.title, node.page_start, node.page_end, "",
                        node.summary, node.level, doc_id=node.doc_id, section_type=node.section_type,
                        node_id=node.node_id, prefix_summary=node.prefix_summary, text_token_count=node.text_token_count,
                        children=[self._clone_for_cache(c) for c in node.children], metadata=node.metadata)

    def cleanup(self):
        for doc in self._pdf_cache.values():
            try:
                doc.close()
            except:
                pass
        self._pdf_cache.clear()


class FastHierarchicalIndex(HierarchicalIndex):
    def __init__(self, cache_dir=".declarmima_cache", llm=None):
        super().__init__(cache_dir)
        self.llm = llm

    async def build_from_pdfs_fast(self, files: List, max_workers: int = 4) -> Dict[str, PageNode]:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [loop.run_in_executor(pool, self._extract_pages_raw, f) for f in files]
            raw_docs = await asyncio.gather(*futures)
        if self.llm:
            toc_tasks = [self._llm_extract_toc(doc_name, pages) for doc_name, pages in raw_docs]
            toc_results = await asyncio.gather(*toc_tasks)
        else:
            toc_results = [{"has_toc": False, "headings_detected": []} for _ in raw_docs]
        trees = {}
        for (doc_name, pages), toc in zip(raw_docs, toc_results):
            tree = self._build_tree_from_toc(doc_name, pages, toc)
            full_text = "\n".join([p['text'] for p in pages])
            meta = self.metadata_extractor.extract_metadata(doc_name, full_text)
            tree.metadata = meta
            trees[doc_name] = tree
        if self.llm:
            await self._generate_summaries_async(trees)
        for doc_name, tree in trees.items():
            self.doc_trees[doc_name] = tree
            self._save_tree_fast(doc_name, tree)
        return trees

    def _extract_pages_raw(self, file_obj) -> Tuple[str, List[Dict]]:
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
            pages.append({'page_num': p+1, 'text': page.get_text("text"), 'images': len(page.get_images()), 'blocks': page.get_text("blocks")})
        doc.close()
        return doc_name, pages

    async def _llm_extract_toc(self, doc_name: str, pages: List[Dict]) -> Dict[str, Any]:
        sample_text = "\n\n".join(p['text'][:1500] for p in pages[:5])
        prompt = f"""Analyze this document and extract its hierarchical structure.
Return JSON with:
- "has_toc": bool
- "toc_entries": list of {{"title": str, "level": int, "page": int}}
- "headings_detected": list of {{"title": str, "level": int, "page": int}}
- "doc_type": str
- "suggested_root_title": str
Document sample:
{sample_text[:6000]}
Return ONLY valid JSON."""
        try:
            response = await asyncio.to_thread(self.llm.generate, prompt, max_new_tokens=1024, fast_json=True)
            result = self._extract_json_safe(response)
            if result and isinstance(result, dict):
                return result
        except Exception as e:
            logger.warning(f"LLM TOC extraction failed for {doc_name}: {e}")
        return {"has_toc": False, "headings_detected": [], "doc_type": "unknown"}

    def _extract_json_safe(self, text: str) -> Optional[Any]:
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

    def _build_tree_from_toc(self, doc_name: str, pages: List[Dict], toc: Dict) -> PageNode:
        safe_title = toc.get("suggested_root_title") or doc_name
        root = PageNode(f"{doc_name}_root", safe_title, 1, len(pages), "", f"Document {doc_name}", 0, doc_id=doc_name, node_id="0000")
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
                for i in range(page, min(end+1, len(pages)+1)):
                    try:
                        page_data = pages[i-1]
                        if isinstance(page_data, dict) and 'text' in page_data:
                            text_parts.append(page_data['text'])
                    except:
                        continue
                text = "\n\n".join(text_parts)
                node = PageNode(f"{doc_name}_toc_{level}_{title[:20]}", title, page, end, text, text[:200], level, doc_id=doc_name)
                nodes_by_level.setdefault(level, []).append(node)
            for level in sorted(nodes_by_level.keys()):
                for node in nodes_by_level[level]:
                    parent = self._find_parent(root, level-1, node.page_start)
                    parent.children.append(node)
        else:
            for p in pages:
                text = p.get('text', '')
                if not str(text).strip():
                    continue
                page_num = int(p.get('page_num', 1)) if str(p.get('page_num', 1)).isdigit() else 1
                node = PageNode(f"{doc_name}_p{page_num}", f"Page {page_num}", page_num, page_num, text, str(text)[:200], 3, doc_id=doc_name)
                root.children.append(node)
        self._assign_node_ids(root)
        return root

    async def _generate_summaries_async(self, trees: Dict[str, PageNode]):
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
                    tasks.append(self._summarize_node(node))
                else:
                    node.summary = node.full_text[:200]
            if tasks:
                await asyncio.gather(*tasks)

    async def _summarize_node(self, node: PageNode):
        text = node.full_text[:3000]
        prompt = f"""Summarize this document section in one sentence (max 200 chars).
Focus on key parameters, methods, and findings.
Text: {text}
Summary:"""
        try:
            summary = await asyncio.to_thread(self.llm.generate, prompt, max_new_tokens=150, temperature=0.1)
            node.summary = summary.strip()[:200]
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            node.summary = text[:200]

    def _save_tree_fast(self, doc_name: str, tree: PageNode):
        safe = re.sub(r'[^\w\-_.]', '_', doc_name)
        doc_hash = hashlib.sha256(doc_name.encode()).hexdigest()[:16]
        path = self.cache_dir / f"{safe}.{doc_hash}.tree.json"
        try:
            with open(path, "wb") as f:
                f.write(fast_json_dumps(tree.to_dict(), indent=True))
        except Exception as e:
            logger.warning(f"Fast save failed: {e}")


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
        if OLLAMA_AVAILABLE:
            try:
                requests.get("http://localhost:11434/api/tags", timeout=5)
                self.backend = "ollama"
                self.client = ollama.Client(host="http://localhost:11434")
                return
            except:
                pass
        if TRANSFORMERS_AVAILABLE:
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


class QuantitativeKnowledgeGraph:
    def __init__(self):
        self.doc_graphs: Dict[str, Dict] = {}
        self.phys_classifier = PhysicalQuantityClassifier()
        self.metadata_index: Dict[str, DocumentMetadata] = {}
        self.concept_normalizer = ConceptNormalizer()

    def add_document_metadata(self, doc_name: str, metadata: DocumentMetadata):
        self.metadata_index[doc_name] = metadata

    def add_extractions(self, doc_id: str, items: List[UniversalExtractionItem]):
        graph = {"doc_id": doc_id, "parameters": defaultdict(list), "materials": defaultdict(list),
                 "methods": defaultdict(list), "by_page": defaultdict(list), "by_section": defaultdict(list),
                 "by_physical_quantity": defaultdict(list), "all_items": []}
        for item in items:
            item_dict = item.to_dict()
            if item.physical_quantity:
                item_dict["physical_quantity"] = self.concept_normalizer.normalize(item.physical_quantity)
            if item.material:
                item_dict["material"] = self.concept_normalizer.normalize(item.material)
            graph["all_items"].append(item_dict)
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

    def get_parameter_across_docs(self, param_name: str) -> List[Dict]:
        results = []
        param_key = param_name.lower()
        for doc_id, graph in self.doc_graphs.items():
            if param_key in graph["parameters"]:
                for item in graph["parameters"][param_key]:
                    results.append({**item, "doc_id": doc_id})
        return results

    def get_all_materials(self) -> Dict[str, List[str]]:
        mat_dict = {}
        for doc_id, graph in self.doc_graphs.items():
            materials = set()
            for item in graph["all_items"]:
                if item.get("material"):
                    materials.add(item["material"])
            mat_dict[doc_id] = list(materials)
        return mat_dict

    def get_material_summary_stats(self, material_name: str) -> Dict[str, Any]:
        values_by_pq = defaultdict(list)
        for doc_id, graph in self.doc_graphs.items():
            for item in graph["all_items"]:
                if item.get("material") and item["material"].lower() == material_name.lower():
                    if item.get("value") is not None and item.get("physical_quantity"):
                        values_by_pq[item["physical_quantity"]].append({"value": item["value"], "unit": item.get("unit",""), "doc": doc_id, "page": item.get("page",1)})
        return dict(values_by_pq)

    def to_tree_annotation(self, doc_tree: PageNode, max_chars: int = 20000) -> Dict[str, Any]:
        doc_id = doc_tree.doc_id
        graph = self.doc_graphs.get(doc_id, {})
        def annotate_node(node: PageNode) -> Dict[str, Any]:
            result = node.to_tree_format(max_chars=max_chars)
            node_items = []
            end_page = node.page_end or node.page_start
            for page in range(node.page_start, end_page + 1):
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
            if node.children:
                result["nodes"] = [annotate_node(c) for c in node.children]
            return result
        return annotate_node(doc_tree)

    def get_summary_stats(self, physical_quantity: str) -> Dict[str, Any]:
        values = []
        for doc_id, graph in self.doc_graphs.items():
            for item in graph["all_items"]:
                if item.get("physical_quantity") == physical_quantity and item.get("value") is not None:
                    values.append(item["value"])
        if not values:
            return {"count": 0, "documents": []}
        docs = list(set(item["doc_source"] for doc_id, graph in self.doc_graphs.items() for item in graph["all_items"] if item.get("physical_quantity") == physical_quantity))
        stats = {"count": len(values), "documents": docs, "values": values}
        if values:
            stats.update({"min": min(values), "max": max(values), "mean": float(np.mean(values)), "std": float(np.std(values)) if len(values) > 1 else 0})
        return stats

    def get_all_physical_quantities(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for doc_id, graph in self.doc_graphs.items():
            for item in graph["all_items"]:
                pq = item.get("physical_quantity")
                if pq:
                    counts[pq] += 1
        return dict(counts)

    def build_extracted_values(self, query: str) -> List[ExtractedValue]:
        all_values = []
        for doc_id, graph in self.doc_graphs.items():
            for item in graph["all_items"]:
                if item.get("item_type") != "quantitative":
                    continue
                val = item.get("value")
                if val is None or val == 0:
                    continue
                unit = item.get("unit", "")
                phys_q = item.get("physical_quantity") or self.phys_classifier.classify(item.get("parameter_name"), unit, item.get("context", ""))
                # Do NOT skip "unknown" — keep them so the user can see what was missed
                all_values.append(ExtractedValue(
                    query=query, value=val, unit=unit, physical_quantity=phys_q or "unknown",
                    parameter_name=item.get("parameter_name"), material=item.get("material"),
                    confidence=item.get("confidence", 0.7), context=item.get("context", "")[:300],
                    doc_name=doc_id, page=item.get("page", 1), section_title=item.get("section_title")
                ))
        return all_values

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
        return {"found": True, "entity": entity_name, "count": len(values), "unit": list(units)[0] if units else "unknown", "range": (min(values), max(values)), "mean": float(np.mean(values)), "std": float(np.std(values)) if len(values) > 1 else 0.0, "documents": list(docs), "values": values}

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
                            contradictions.append({"entity": entity_name, "doc_a": docs[i], "value_a": mean_i, "doc_b": docs[j], "value_b": mean_j, "ratio": ratio, "severity": "high" if ratio > 5 else "moderate"})
        return contradictions

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
        return sorted(entities)


class UniversalLLMExtractor:
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
  "physical_quantity": "one of: laser_power, electrical_power, scan_speed, flow_speed, feed_rate, irradiance, temperature, melting_temperature, energy_density, areal_energy_density, linear_energy_density, layer_thickness, spot_size, exposure_time, enthalpy, viscosity, thermal_conductivity, density, yield_strength, tensile_strength, ultimate_tensile_strength, hardness, elongation, modulus, stacking_fault_energy, unstable_stacking_fault_energy, ideal_shear_strength, corrosion_potential, pitting_potential, breakdown_potential, repassivation_potential, open_circuit_potential, corrosion_current_density, polarization_resistance, apparent_polarization_resistance, current_density, PREN, phase_fraction, austenite_fraction, ferrite_fraction, grain_size, cell_size, porosity, relative_density, surface_roughness, sauter_mean_diameter, spray_penetration, plume_height, film_thickness, absorption_coefficient, youngs_modulus, poisson_ratio, coefficient_thermal_expansion, unknown",
  "material": "alloy or material name if mentioned (e.g., Ti3Au, CP Ti, Grade II Ti, SDSS 2507, UNS S32750, AlSiMgZr, Al-Si-Mg-Zr, TiB2/Al-Si-Mg-Zr, Fe-based metallic glass, Au-Ti, 316L, 2205, Inconel 718, Ti6Al4V)",
  "method": "e.g., LPBF, L-PBF, DED, SLM, PFI, GDI, FEM, MD, nanoindentation, EIS, CPP, XRD, SEM, TEM, EBSD, EDS, DTA"
}}

CRITICAL RULES:
1. Capture ALL numbers with units, even if they describe corrosion, electrochemistry, thermal properties, mechanical properties, microstructural features, or spray dynamics.
2. For electrochemical data: map Ecorr/Erp/Epit/Ebr to corrosion_potential/pitting_potential/etc., NOT just generic potential.
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
        self.phys_classifier = PhysicalQuantityClassifier()
        self.concept_normalizer = ConceptNormalizer()

    def extract_from_chunks(self, chunks: List[Dict], query: str, query_analysis: Optional[Dict] = None) -> List[UniversalExtractionItem]:
        if not chunks:
            return []
        qa = query_analysis or {"query_type": "mixed", "keywords": []}
        items = []
        for chunk in chunks:
            text = chunk["full_text"]
            doc = chunk["doc_id"]
            page = chunk["page_start"]
            if qa.get("query_type") == "quantitative" and not re.search(r'\d+', text):
                continue
            prompt = self.EXTRACTION_PROMPT.format(query=query, query_type=qa.get("query_type","mixed"), sections_text=text[:4000], doc_id=doc)
            try:
                response = self.llm.generate(prompt, max_new_tokens=1024, fast_json=True)
                json_str = self._extract_json(response)
                if json_str:
                    data = json.loads(json_str)
                    for item_data in data if isinstance(data, list) else data.get("items", []):
                        if "physical_quantity" not in item_data or not item_data["physical_quantity"]:
                            item_data["physical_quantity"] = self.phys_classifier.classify(item_data.get("parameter_name"), item_data.get("unit"), item_data.get("context", ""))
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
        unique = {}
        for i in items:
            key = (i.content, i.doc_source, i.page, i.material)
            if key not in unique or i.confidence > unique[key].confidence:
                unique[key] = i
        min_conf = UNIVERSAL_CONFIG.get("min_confidence_threshold", 0.55)
        return [i for i in unique.values() if i.confidence >= min_conf]

    def _extract_json(self, text: str) -> Optional[str]:
        patterns = [r'\[.*\]', r'```json\s*(\[.*?\])\s*```', r'(\[.*\])']
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_str = match.group(1) if match.groups() else match.group(0)
                try:
                    json.loads(json_str)
                    return json_str
                except:
                    continue
        return None


class LLMReasoningSynthesizer:
    def __init__(self, llm: HybridLLM):
        self.llm = llm
        self.phys_classifier = PhysicalQuantityClassifier()

    def synthesize(self, query: str, items: List[UniversalExtractionItem]) -> str:
        if not items:
            return f"No relevant information found for query: '{query}'. Try rephrasing or check the documents."
        extracted_lines = []
        for item in items:
            pq = item.physical_quantity or "unknown"
            pq_readable = self.phys_classifier.get_human_readable(pq)
            mat = f" [{item.material}]" if item.material else ""
            line = f"- {pq_readable}{mat}: {item.content} ({item.confidence:.2f}) context: {item.context[:200]} {item.citation()}"
            extracted_lines.append(line)
        extracted_text = "\n".join(extracted_lines[:20])
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
            answer = self.llm.generate(prompt, max_new_tokens=1024, temperature=0.2)
            return answer.strip()
        except Exception as e:
            logger.error(f"Reasoning synthesis error: {e}")
            lines = [f"Query: {query}\nFound {len(items)} relevant items:\n"] + [f"- {item.content} {item.citation()}" for item in items[:5]]
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
        return "\n".join(lines)


class HierarchicalTreeRetriever:
    def __init__(self, llm: HybridLLM, max_results=30, max_text_chars=20000):
        self.llm = llm
        self.max_results = max_results
        self.max_text_chars = max_text_chars
        self._condensed_cache: Dict[str, Dict] = {}
        self.template = llm.template if hasattr(llm, 'template') else MODEL_PROMPT_TEMPLATES["default"]

    async def retrieve_quantitative(self, query: str, annotated_trees: List[Dict]) -> List[Dict]:
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
                if len(full_text) > self.max_text_chars:
                    full_text = full_text[:self.max_text_chars] + "..."
                results.append({"full_text": full_text, "page_start": node.get('start_index'), "doc_id": doc_id, "section_title": node.get('title'), "quantitative_items": node.get('quantitative_items', []), "citation": f'<cite doc="{doc_id}" page="{node.get("start_index")}"/>', "selection_reasoning": sel.get('reasoning', ''), "confidence": sel.get('confidence', 0)})
        return results[:self.max_results]

    def _condense_tree(self, tree: Dict, max_depth: int = 3) -> Dict[str, Any]:
        def condense(node: Dict, depth: int = 0) -> Dict[str, Any]:
            if depth > max_depth:
                return {"node_id": node.get("node_id", ""), "title": node.get("title", ""), "leaf": True}
            result = {"node_id": node.get("node_id", ""), "title": node.get("title", ""), "summary": (node.get("summary", "") or "")[:150]}
            if node.get("metadata"):
                meta = node["metadata"]
                if meta.get("alloys"):
                    result["alloys"] = meta["alloys"][:3]
                if meta.get("laser_power_values"):
                    result["power_hint"] = f"{min(meta['laser_power_values'])}-{max(meta['laser_power_values'])} W"
                if meta.get("scan_speed_values"):
                    result["speed_hint"] = f"{min(meta['scan_speed_values'])}-{max(meta['scan_speed_values'])} mm/s"
            q_items = node.get("quantitative_items", [])
            if q_items:
                params = list(set(item.get("parameter_name", "") for item in q_items if item.get("parameter_name")))
                if params:
                    result["has_quantitative"] = params[:5]
            else:
                text = node.get("text", "")
                if text:
                    candidates = re.findall(r'(\d+(?:\.\d+)?)\s*(W|kW|mW|J|mm/s|C|K|MPa|GPa|nm|um|mm|s|m/s|W/cm2|kW/cm2)', text, re.IGNORECASE)
                    if candidates:
                        result["candidate_values"] = [f"{v}{u}" for v, u in candidates[:3]]
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
            if res:
                return res
        return None



# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================
@dataclass
class VisConfig:
    """Centralized configuration for all visualization styling parameters."""
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
    plotly_width: int = None
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

class PublicationVisualizationEngine:
    DOMAIN_COLORS = {
        "laser_power": "#3b82f6", "scan_speed": "#8b5cf6", "yield_strength": "#f59e0b",
        "tensile_strength": "#10b981", "hardness": "#ec4899", "temperature": "#ef4444",
        "energy_density": "#06b6d4", "unknown": "#6b7280", "material": "#3b82f6",
        "document": "#10b981", "hub": "#dc2626"
    }
    COLORMAP_OPTIONS = {
        "viridis": "viridis", "plasma": "plasma", "inferno": "inferno", "magma": "magma",
        "cividis": "cividis", "Blues": "Blues", "Greens": "Greens", "Oranges": "Oranges",
        "Reds": "Reds", "RdBu": "RdBu", "Spectral": "Spectral", "coolwarm": "coolwarm",
        "Set1": "Set1", "Set2": "Set2", "Set3": "Set3", "tab10": "tab10", "tab20": "tab20"
    }

    def __init__(self, kgraph: QuantitativeKnowledgeGraph, config: Optional[VisConfig] = None):
        self.kgraph = kgraph
        self.cfg = config or VisConfig()
        # Apply matplotlib rcParams from config
        plt.rcParams['font.family'] = self.cfg.font_family
        plt.rcParams['font.size'] = self.cfg.font_size
        plt.rcParams['axes.titlesize'] = self.cfg.title_font_size
        plt.rcParams['axes.labelsize'] = self.cfg.label_font_size
        plt.rcParams['figure.dpi'] = self.cfg.figure_dpi
        plt.rcParams['savefig.dpi'] = self.cfg.figure_dpi
        plt.rcParams['lines.linewidth'] = self.cfg.line_width

    @property
    def font_family(self): return self.cfg.font_family
    @property
    def font_size(self): return self.cfg.font_size
    @property
    def title_font_size(self): return self.cfg.title_font_size
    @property
    def label_font_size(self): return self.cfg.label_font_size
    @property
    def default_colormap(self): return self.cfg.default_colormap
    @property
    def figure_dpi(self): return self.cfg.figure_dpi
    @property
    def aliases(self): return self.cfg.aliases
    @property
    def label_style(self): return self.cfg.label_style

    def _get_colormap(self, name: Optional[str] = None) -> str:
        return self.COLORMAP_OPTIONS.get(name or self.default_colormap, "viridis")

    def _get_plotly_colorscale(self, name: Optional[str] = None) -> str:
        name = name or self.default_colormap
        mapping = {"coolwarm": "RdBu", "RdBu": "RdBu", "seismic": "RdBu", "bwr": "RdBu"}
        plotly_builtins = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'blues', 'greens', 'oranges', 'reds']
        lowered = name.lower()
        if lowered in plotly_builtins:
            return lowered
        return mapping.get(lowered, 'viridis')

    def extract_dataframe(self, aliases: Optional[Dict[str, str]] = None, label_style: str = "doi") -> pd.DataFrame:
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
                    rows.append({"doc": doc_id, "doc_stem": display, "doc_citation": citation, "physical_quantity": phys, "material": mat, "value": value, "unit": unit, "confidence": item.get("confidence", 0.5), "page": item.get("page", 0), "context": item.get("context", "")[:200]})
        return pd.DataFrame(rows)

    # =============================================================================
    # QUERY-AWARE DATA FILTERING
    # =============================================================================
    def get_query_focused_df(self, query_ctx: QueryContext) -> pd.DataFrame:
        """Return dataframe filtered to current query context."""
        df = self.extract_dataframe(aliases=self.cfg.aliases, label_style=self.cfg.label_style)
        if df.empty or not query_ctx.has_data():
            return df
        mask = (
            df["doc"].isin(query_ctx.relevant_doc_ids) |
            df["physical_quantity"].isin(query_ctx.physical_quantities) |
            (df["material"].isin(query_ctx.materials) & df["material"].notna())
        )
        return df[mask].copy()

    # =============================================================================
    # QUERY-AWARE KNOWLEDGE GRAPH (NetworkX)
    # =============================================================================
    def plot_query_knowledge_graph(self, query_ctx: QueryContext, figsize=(14, 11)) -> plt.Figure:
        """Query-focused interactive Knowledge Graph"""
        if not query_ctx.has_data():
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No quantitative data for this query", ha='center', va='center', fontsize=14)
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
                      title=f"Document: {display_name}\n{len([v for v in query_ctx.extracted_values if v.doc_name == doc_id])} values")

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

            if val.material and val.material in G:
                G.add_edge(val.material, label, weight=2)
            if val.doc_name and get_display_name(val.doc_name, self.cfg.aliases) in G:
                G.add_edge(get_display_name(val.doc_name, self.cfg.aliases), label, weight=1.5)
            for pq in query_ctx.physical_quantities:
                if val.physical_quantity == pq and pq in G:
                    G.add_edge(pq, label, weight=1)

        # Connect query to everything
        for node in list(G.nodes()):
            if node != "QUERY":
                G.add_edge("QUERY", node, weight=0.8)

        # Draw
        pos = nx.spring_layout(G, k=0.65, iterations=80, seed=42)
        fig, ax = plt.subplots(figsize=figsize)
        
        query_nodes = ["QUERY"]
        doc_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "doc"]
        pq_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "pq"]
        mat_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "material"]
        val_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "value"]

        nx.draw_networkx_nodes(G, pos, nodelist=query_nodes, node_color="#8b5cf6", node_size=3200, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=doc_nodes, node_color="#10b981", node_size=1400, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=pq_nodes, node_color="#3b82f6", node_size=1100, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=mat_nodes, node_color="#f59e0b", node_size=1300, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=val_nodes, node_color="#ec4899", node_size=650, ax=ax)

        nx.draw_networkx_edges(G, pos, alpha=0.35, width=1.2, edge_color="#94a3b8", ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=9, font_family=self.font_family, ax=ax)

        ax.set_title(f"Query-Focused Knowledge Graph\n{query_ctx.query[:70]}...", 
                    fontsize=15, fontweight='bold', pad=20)
        ax.axis("off")
        plt.tight_layout()
        return fig

    # =============================================================================
    # QUERY-AWARE KNOWLEDGE GRAPH (PyVis with Modal)
    # =============================================================================

    def plot_query_sunburst(self, query_ctx: QueryContext) -> go.Figure:


        """Query-focused hierarchical sunburst"""
        df_focus = self.get_query_focused_df(query_ctx)
        if df_focus.empty:
            return go.Figure().update_layout(title="No data for current query")

        # Create hierarchy: Physical Quantity → Material → Document → Value Range
        df_sun = df_focus.copy()
        df_sun["material"] = df_sun["material"].fillna("Unknown").replace("", "Unknown")
        df_sun["doc_stem"] = df_sun["doc_stem"].fillna("Unknown").replace("", "Unknown")
        
        # Bin values for better hierarchy
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
                    title=f"Query Hierarchy: {query_ctx.query[:60]}...",
                    maxdepth=4
                )
                fig.update_traces(textinfo="label+percent entry")
                fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
                return fig
            except Exception as e:
                logger.warning(f"Sunburst binning failed: {e}")
        
        # Fallback: simpler hierarchy
        try:
            fig = px.sunburst(
                df_sun, 
                path=["physical_quantity", "material", "doc_stem"], 
                values="value",
                color="value",
                color_continuous_scale=self._get_plotly_colorscale(),
                title=f"Query Hierarchy: {query_ctx.query[:60]}..."
            )
            fig.update_traces(textinfo="label+percent entry")
            fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
            return fig
        except Exception as e:
            logger.error(f"Query sunburst failed: {e}")
            return go.Figure().update_layout(title="Sunburst unavailable for this query")

    # =============================================================================
    # EXISTING VISUALIZATION METHODS (REPAIRED)
    # =============================================================================
    def plot_query_knowledge_graph_pyvis(self, query_ctx: QueryContext) -> str:
        """Interactive PyVis KG with confidence highlighting + modal-ready JS"""
        if not PYVIS_AVAILABLE:
            return "<p>PyVis not installed. Run: <code>pip install pyvis</code></p>"

        if not query_ctx.has_data():
            return "<p>No quantitative data available for this query.</p>"

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
        for i, val in enumerate(sorted(query_ctx.extracted_values, key=lambda x: x.confidence, reverse=True)[:30]):
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

        if "</body>" in html:
            html = html.replace("</body>", modal_js + "</body>")
        else:
            html += modal_js

        return html
    def plot_quantitative_histogram(self, df: pd.DataFrame, quantity_name: str, group_by: str = "material", colormap: Optional[str] = None) -> go.Figure:
        if df.empty:
            return go.Figure().update_layout(title=f"No {quantity_name} data")
        subset = df[df["physical_quantity"] == quantity_name]
        if subset.empty:
            return go.Figure()
        clean_col = subset[group_by].fillna("Unknown").replace("", "Unknown")
        subset = subset.assign(clean_group=clean_col)
        groups = sorted(subset["clean_group"].unique())
        fig = go.Figure()
        cmap_obj = plt.get_cmap(self._get_colormap(colormap))
        for i, grp in enumerate(groups):
            data = subset[subset["clean_group"] == grp]["value"]
            color = mcolors.to_hex(cmap_obj(i / max(len(groups)-1, 1))) if len(groups) > 1 else "#3b82f6"
            fig.add_trace(go.Bar(name=str(grp), x=[grp], y=[data.mean()], error_y=dict(type='data', array=[data.std()] if len(data)>1 else [0], visible=True), marker_color=color, text=[f"n={len(data)}<br>u={data.mean():.2f}<br>s={data.std():.2f}"], textposition="outside"))
        unit = subset["unit"].iloc[0] if not subset.empty else ""
        fig.update_layout(barmode='group', title=f"{quantity_name.replace('_',' ').title()} Values by {group_by.title()}", xaxis_title=group_by.title(), yaxis_title=f"{quantity_name.replace('_',' ').title()} ({unit})", font=dict(family=self.font_family, size=self.font_size), height=500)
        return fig

    def plot_quantities_bar(self, df: pd.DataFrame, colormap: Optional[str] = None) -> go.Figure:
        if df.empty:
            return go.Figure().update_layout(title="No data")
        counts = df["physical_quantity"].value_counts().reset_index()
        counts.columns = ["Physical Quantity", "Count"]
        fig = px.bar(counts, x="Physical Quantity", y="Count", color="Physical Quantity", title="Occurrence Counts by Physical Quantity", color_discrete_sequence=[self.DOMAIN_COLORS.get(q, "#6b7280") for q in counts["Physical Quantity"]])
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
        return fig

    def plot_material_counts(self, df: pd.DataFrame, colormap: Optional[str] = None) -> go.Figure:
        mat_df = df[df["physical_quantity"] == "material"] if "material" in df["physical_quantity"].values else df
        if mat_df.empty:
            return go.Figure().update_layout(title="No materials found")
        counts = mat_df["material"].value_counts().head(10).reset_index()
        counts.columns = ["Material", "Count"]
        fig = px.bar(counts, x="Material", y="Count", color="Material", title="Top 10 Materials/Alloys Mentioned")
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
        return fig

    def plot_quantity_distribution_pie(self, colormap: Optional[str] = None) -> go.Figure:
        pq_counts = self.kgraph.get_all_physical_quantities()
        if not pq_counts:
            return go.Figure().update_layout(title="No quantities found")
        sorted_pq = sorted(pq_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        labels = [self.kgraph.phys_classifier.get_human_readable(pq) for pq, _ in sorted_pq]
        values = [count for _, count in sorted_pq]
        fig = px.pie(values=values, names=labels, title="Top Physical Quantities Distribution", color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
        return fig

    def plot_material_distribution_donut(self, colormap: Optional[str] = None) -> go.Figure:
        mat_dict = self.kgraph.get_all_materials()
        if not mat_dict:
            return go.Figure().update_layout(title="No materials found")
        mat_counts = Counter(m for mats in mat_dict.values() for m in mats)
        top_mats = mat_counts.most_common(10)
        labels = [m for m, _ in top_mats]
        values = [c for _, c in top_mats]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4, marker_colors=[f"#{hash(l) % 0xFFFFFF:06x}" for l in labels])])
        fig.update_layout(title="Material Distribution (Donut)", annotations=[dict(text='Materials', x=0.5, y=0.5, font_size=14, showarrow=False)])
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
        return fig

    def plot_quantitative_sunburst(self, df: pd.DataFrame, quantity: str, colormap: Optional[str] = None) -> go.Figure:
        if df.empty:
            return go.Figure()
        subset = df[df["physical_quantity"] == quantity].copy()
        if subset.empty:
            return go.Figure()
        subset["material"] = subset["material"].fillna("Unknown").replace("", "Unknown")
        subset["doc_stem"] = subset["doc_stem"].fillna("Unknown").replace("", "Unknown")
        subset = subset.dropna(subset=["material", "doc_stem"])
        subset = subset.dropna(subset=["value"])
        if subset.empty or len(subset) < 2:
            # Fallback without value_range binning
            try:
                fig = px.sunburst(subset, path=["material", "doc_stem"], values="value", color="value",
                                  color_continuous_scale=self._get_plotly_colorscale(colormap),
                                  title=f"{quantity.replace('_',' ').title()} Distribution Hierarchy")
                fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
                return fig
            except:
                return go.Figure()
        n_bins = min(5, max(2, len(subset)//3))
        try:
            subset["value_range"] = pd.cut(subset["value"], bins=n_bins, precision=1).astype(str).fillna("unknown")
        except Exception:
            subset["value_range"] = "single"
        fig = px.sunburst(subset, path=["material", "doc_stem", "value_range"], values="value", color="value", color_continuous_scale=self._get_plotly_colorscale(colormap), title=f"{quantity.replace('_',' ').title()} Distribution Hierarchy")
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
        return fig

    def plot_sunburst_hierarchy(self, df: pd.DataFrame, colormap: Optional[str] = None) -> go.Figure:
        if df.empty:
            return go.Figure().update_layout(title="No data")
        df_hier = df.copy()
        df_hier["physical_quantity"] = df_hier["physical_quantity"].fillna("Unknown").replace("", "Unknown")
        df_hier["material"] = df_hier["material"].fillna("Unknown").replace("", "Unknown")
        df_hier["doc_stem"] = df_hier["doc_stem"].fillna("Unknown").replace("", "Unknown")
        df_hier["value_dummy"] = 1
        try:
            fig = px.sunburst(df_hier, path=["physical_quantity", "material", "doc_stem"], values="value_dummy", title="Hierarchy of Physical Quantities, Materials, and Documents")
        except Exception as e:
            logger.error(f"Sunburst error: {e}")
            try:
                fig = px.sunburst(df_hier, path=["physical_quantity", "material"], values="value_dummy", title="Hierarchy (simplified)")
            except:
                fig = go.Figure().update_layout(title="Sunburst unavailable")
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
        return fig

    def plot_treemap(self, colormap: Optional[str] = None) -> go.Figure:
        df = self.extract_dataframe()
        if df.empty:
            return go.Figure().update_layout(title="No data")
        agg = df.groupby(["physical_quantity", "material"]).size().reset_index(name="count")
        fig = px.treemap(agg, path=["physical_quantity", "material"], values="count", title="Entity Treemap: Quantities and Materials", color="count", color_continuous_scale=self._get_plotly_colorscale(colormap))
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
        return fig

    def plot_treemap_materials(self, df: pd.DataFrame, colormap: Optional[str] = None) -> go.Figure:
        if df.empty:
            return go.Figure().update_layout(title="No data")
        mat_counts = df[df["physical_quantity"] == "material"]["material"].value_counts().reset_index() if "material" in df["physical_quantity"].values else df["material"].value_counts().reset_index()
        mat_counts.columns = ["Material", "Count"]
        if mat_counts.empty:
            return go.Figure().update_layout(title="No material data for treemap")
        fig = px.treemap(mat_counts, path=["Material"], values="Count", title="Material Treemap")
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
        return fig

    def plot_scatter_power_vs_speed(self, df: pd.DataFrame, colormap: Optional[str] = None) -> go.Figure:
        power_df = df[df["physical_quantity"] == "laser_power"][["doc", "material", "value"]].rename(columns={"value": "laser_power"})
        speed_df = df[df["physical_quantity"] == "scan_speed"][["doc", "material", "value"]].rename(columns={"value": "scan_speed"})
        merged = pd.merge(power_df, speed_df, on=["doc", "material"], how="inner")
        if merged.empty:
            return go.Figure().update_layout(title="No paired laser power and scan speed data")
        fig = px.scatter(merged, x="laser_power", y="scan_speed", color="material", title="Laser Power vs Scan Speed by Material", labels={"laser_power": "Laser Power (W)", "scan_speed": "Scan Speed (mm/s)"})
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
        return fig

    def plot_radar_by_material(self, colormap: Optional[str] = None) -> go.Figure:
        df = self.extract_dataframe()
        if df.empty:
            return go.Figure().update_layout(title="No data")
        top_quantities = df["physical_quantity"].value_counts().head(5).index.tolist()
        pivot = df[df["physical_quantity"].isin(top_quantities)].pivot_table(index="material", columns="physical_quantity", values="value", aggfunc="mean").fillna(0)
        if pivot.empty:
            return go.Figure().update_layout(title="No data for radar")
        fig = go.Figure()
        cmap = plt.get_cmap(self._get_colormap(colormap))
        materials = pivot.index.tolist()
        for i, mat in enumerate(materials):
            values = pivot.loc[mat].tolist()
            values += values[:1]
            color = mcolors.to_hex(cmap(i / max(len(materials)-1, 1))) if len(materials)>1 else "#3b82f6"
            fig.add_trace(go.Scatterpolar(r=values, theta=top_quantities + [top_quantities[0]], fill='toself', name=mat, line_color=color))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Material Performance Radar (Mean Values)", font=dict(family=self.font_family, size=self.font_size))
        return fig

    def plot_document_radar(self, colormap: Optional[str] = None) -> go.Figure:
        df = self.extract_dataframe()
        if df.empty:
            return go.Figure().update_layout(title="No data")
        pivot = df.pivot_table(index="doc_stem", columns="physical_quantity", values="value", aggfunc="count").fillna(0)
        if pivot.empty:
            return go.Figure().update_layout(title="No data")
        fig = go.Figure()
        cmap = plt.get_cmap(self._get_colormap(colormap))
        docs = pivot.index.tolist()
        for i, doc in enumerate(docs):
            values = pivot.loc[doc].tolist()
            values += values[:1]
            color = mcolors.to_hex(cmap(i / max(len(docs)-1, 1))) if len(docs)>1 else "#3b82f6"
            fig.add_trace(go.Scatterpolar(r=values, theta=pivot.columns.tolist() + [pivot.columns[0]], fill='toself', name=doc, line_color=color))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Document Coverage Radar (Counts per Quantity Type)", font=dict(family=self.font_family, size=self.font_size))
        return fig

    def plot_quantitative_radar(self, df: pd.DataFrame, quantity_name: str, colormap: Optional[str] = None) -> go.Figure:
        if df.empty:
            return go.Figure().update_layout(title=f"No {quantity_name} data")
        stats = df[df["physical_quantity"] == quantity_name].groupby("material")["value"].agg(["mean", "std", "min", "max", "count"])
        if stats.empty:
            return go.Figure().update_layout(title="No data for radar")
        categories = ["Mean", "Max", "Min", "Std", "Count"]
        fig = go.Figure()
        cmap_obj = plt.get_cmap(self._get_colormap(colormap)) if colormap else None
        for i, (mat, row) in enumerate(stats.iterrows()):
            values = [row["mean"], row["max"], row["min"], row["std"], float(row["count"])]
            values += values[:1]
            color = mcolors.to_hex(cmap_obj(i / max(len(stats)-1, 1))) if cmap_obj else None
            fig.add_trace(go.Scatterpolar(r=values, theta=categories + [categories[0]], fill='toself', name=mat, line_color=color))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, title=f"{quantity_name.replace('_',' ').title()} Statistics by Material", font=dict(family=self.font_family, size=self.font_size))
        return fig

    def plot_quantitative_knowledge_graph(self, df: pd.DataFrame, quantity: str, colormap: Optional[str] = None, figsize: Tuple[int,int] = (14,12), aliases: Optional[Dict[str,str]] = None, label_style: str = "doi") -> plt.Figure:
        G = nx.Graph()
        hub = f"{quantity}_hub"
        G.add_node(hub, node_type="hub")
        subset = df[df["physical_quantity"] == quantity]
        if subset.empty:
            return plt.figure()
        for mat in subset["material"].unique():
            if pd.notna(mat) and str(mat).strip() and mat != "Unknown":
                mat_vals = subset[subset["material"] == mat]["value"].tolist()
                tooltip = f"Material: {mat}\nCount: {len(mat_vals)}\nRange: {min(mat_vals):.2f} - {max(mat_vals):.2f}\nMean: {np.mean(mat_vals):.2f}"
                G.add_node(mat, node_type="material", title=tooltip)
                G.add_edge(hub, mat, weight=len(subset[subset["material"] == mat]))
        for doc in subset["doc_stem"].unique():
            if pd.notna(doc) and str(doc).strip():
                orig_doc = None
                for d in self.kgraph.doc_graphs:
                    if get_display_name(d, aliases) == doc:
                        orig_doc = d
                        break
                doc_label = get_citation_label(orig_doc, aliases, style=label_style) if orig_doc else doc
                doc_vals = subset[subset["doc_stem"] == doc]["value"].tolist()
                tooltip = f"Document: {doc_label}\nCount: {len(doc_vals)}\nRange: {min(doc_vals):.2f} - {max(doc_vals):.2f}\nMean: {np.mean(doc_vals):.2f}"
                G.add_node(doc, node_type="doc", title=tooltip)
                G.add_edge(hub, doc, weight=len(subset[subset["doc_stem"] == doc]))
        top = subset.nlargest(min(25, len(subset)), "value")
        for _, row in top.iterrows():
            leaf = f"{row['value']:.1f} {row['unit']}"
            tooltip = f"Value: {row['value']:.2f} {row['unit']}\nMaterial: {row['material']}\nDocument: {row['doc_stem']}"
            G.add_node(leaf, node_type="value", value=row["value"], title=tooltip)
            if pd.notna(row["material"]) and str(row["material"]).strip() and row["material"] != "Unknown":
                G.add_edge(row["material"], leaf, weight=1)
            if pd.notna(row["doc_stem"]) and str(row["doc_stem"]).strip():
                G.add_edge(row["doc_stem"], leaf, weight=1)
        pos = nx.spring_layout(G, k=0.6, seed=42)
        fig, ax = plt.subplots(figsize=figsize)
        nx.draw_networkx_nodes(G, pos, nodelist=[hub], node_color="#dc2626", node_size=2500, ax=ax)
        materials = [n for n,d in G.nodes(data=True) if d.get("node_type")=="material"]
        docs = [n for n,d in G.nodes(data=True) if d.get("node_type")=="doc"]
        vals = [n for n,d in G.nodes(data=True) if d.get("node_type")=="value"]
        nx.draw_networkx_nodes(G, pos, nodelist=materials, node_color="#3b82f6", node_size=800, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=docs, node_color="#10b981", node_size=600, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=vals, node_color="#f59e0b", node_size=300, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.25, width=0.8, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=self.label_font_size, ax=ax, font_family=self.font_family)
        ax.set_title(f"Quantitative Knowledge Graph - {quantity.replace('_',' ').title()}", fontsize=self.title_font_size, fontweight='bold')
        ax.axis("off")
        plt.tight_layout()
        return fig

    def plot_knowledge_network(self, df: pd.DataFrame, colormap: Optional[str] = None, figsize: Tuple[int,int] = (12,10), aliases: Optional[Dict[str,str]] = None, label_style: str = "doi") -> plt.Figure:
        G = nx.Graph()
        docs = [d for d in df["doc_stem"].unique() if pd.notna(d) and str(d).strip() != "" and str(d).lower() != "nan"]
        for doc in docs:
            orig_doc = None
            for d in self.kgraph.doc_graphs:
                if get_display_name(d, aliases) == doc:
                    orig_doc = d
                    break
            if not orig_doc:
                continue
            tooltip = f"Document: {get_citation_label(orig_doc, aliases, style=label_style)}\n"
            doc_items = [it for it in self.kgraph.doc_graphs[orig_doc]["all_items"] if it.get("value") is not None]
            top_vals = sorted(doc_items, key=lambda x: x.get("confidence", 0), reverse=True)[:5]
            for it in top_vals:
                tooltip += f"- {it.get('physical_quantity', 'unknown')}: {it.get('value')} {it.get('unit', '')}\n"
            G.add_node(doc, node_type="doc", color="#1e40af", title=tooltip)
        pqs = [p for p in df["physical_quantity"].unique() if pd.notna(p) and str(p).strip() != "" and str(p).lower() != "nan"]
        for pq in pqs:
            stats = self.kgraph.get_summary_stats(pq)
            tooltip = f"Quantity: {pq}\n"
            if stats.get("count", 0) > 0:
                tooltip += f"Count: {stats['count']}\nRange: {stats.get('min', 0):.2f} - {stats.get('max', 0):.2f}\nMean: {stats.get('mean', 0):.2f}"
            G.add_node(pq, node_type="pq", color=self.DOMAIN_COLORS.get(pq, "#6b7280"), title=tooltip)
        mats = [m for m in df["material"].unique() if pd.notna(m) and str(m).strip() != "" and m != "Unknown" and str(m).lower() != "nan"]
        for mat in mats:
            stats = self.kgraph.get_material_summary_stats(mat)
            tooltip = f"Material: {mat}\n"
            for pq, vals in list(stats.items())[:3]:
                if vals:
                    nums = [v["value"] for v in vals]
                    tooltip += f"- {pq}: {min(nums):.2f} to {max(nums):.2f} ({len(nums)} values)\n"
            G.add_node(mat, node_type="material", color="#f59e0b", title=tooltip)
        for _, row in df.iterrows():
            doc = row["doc_stem"]
            pq = row["physical_quantity"]
            mat = row["material"]
            if doc is None or str(doc).strip() == "" or str(doc).lower() == "nan":
                continue
            if pq is None or str(pq).strip() == "" or str(pq).lower() == "nan":
                continue
            if mat and mat != "Unknown" and str(mat).strip() != "" and str(mat).lower() != "nan":
                if doc in G and mat in G:
                    G.add_edge(doc, mat)
                if pq in G and mat in G:
                    G.add_edge(pq, mat)
            if doc in G and pq in G:
                G.add_edge(doc, pq)
        if len(G.nodes()) == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No valid nodes to display", ha='center', va='center')
            ax.axis("off")
            return fig
        pos = nx.spring_layout(G, k=0.5, seed=42)
        fig, ax = plt.subplots(figsize=figsize)
        node_colors = [G.nodes[n].get("color", "#6b7280") for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        ax.set_title("Knowledge Network: Documents <-> Quantities <-> Materials")
        ax.axis("off")
        plt.tight_layout()
        return fig

    def plot_static_knowledge_network(self, filtered_concepts: Optional[List[str]] = None, top_n: int = 30, figsize: Tuple[int,int] = (14, 12), layout: str = "spring", colormap: Optional[str] = None, node_size_factor: float = 1.0, edge_alpha: float = 0.25, show_labels: bool = True, aliases: Optional[Dict[str,str]] = None, label_style: str = "doi") -> plt.Figure:
        G = nx.Graph()
        docs = list(self.kgraph.doc_graphs.keys())
        for doc_id in docs:
            display = get_display_name(doc_id, aliases)
            label = get_citation_label(doc_id, aliases, style=label_style)
            tooltip = f"Document: {label}\n"
            doc_items = [it for it in self.kgraph.doc_graphs[doc_id]["all_items"] if it.get("value") is not None]
            if doc_items:
                tooltip += "Top values:\n"
                top_vals = sorted(doc_items, key=lambda x: x.get("confidence", 0), reverse=True)[:5]
                for it in top_vals:
                    tooltip += f"- {it.get('physical_quantity', 'unknown')}: {it.get('value')} {it.get('unit', '')}\n"
            G.add_node(display, node_type="doc", bipartite=0, domain="DOCUMENT", title=tooltip, orig_doc=doc_id)
        entities = filtered_concepts or list(self.kgraph.get_all_physical_quantities().keys())[:top_n]
        for ent in entities:
            stats = self.kgraph.get_summary_stats(ent)
            doc_count = stats.get("count", 0)
            G.add_node(ent, node_type="entity", domain="PARAMETER", bipartite=1, salience=doc_count)
            for doc in docs:
                if any(item.get("physical_quantity") == ent or item.get("parameter_name") == ent for item in self.kgraph.doc_graphs[doc]["all_items"]):
                    doc_display = get_display_name(doc, aliases)
                    G.add_edge(doc_display, ent, weight=doc_count * 0.5)
        fig, ax = plt.subplots(figsize=figsize)
        pos = nx.spring_layout(G, k=0.55, iterations=60, seed=42) if layout == "spring" else nx.kamada_kawai_layout(G)
        doc_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "doc"]
        ent_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "entity"]
        nx.draw_networkx_nodes(G, pos, nodelist=doc_nodes, node_color="#1e40af", node_shape="s", node_size=800, alpha=0.85, ax=ax, label="Documents")
        cmap_obj = plt.get_cmap(self._get_colormap(colormap)) if colormap else None
        domains = list(set(G.nodes[n].get("domain", "UNKNOWN") for n in ent_nodes))
        domain_color_idx = {d: i for i, d in enumerate(domains)}
        for node in ent_nodes:
            salience = G.nodes[node].get("salience", 0.5)
            domain = G.nodes[node].get("domain", "UNKNOWN")
            if colormap and cmap_obj:
                idx = domain_color_idx.get(domain, 0)
                base_color = mcolors.to_hex(cmap_obj(idx / max(len(domains) - 1, 1)))
            else:
                base_color = self.DOMAIN_COLORS.get(domain, "#6b7280")
            color = mcolors.to_hex(mcolors.to_rgba(base_color, alpha=0.7 + 0.3 * min(salience / 10, 1)))
            size = (300 + salience * 90) * node_size_factor
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=color, node_shape="o", node_size=size, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=edge_alpha, width=0.8, ax=ax)
        if show_labels:
            nx.draw_networkx_labels(G, pos, font_size=self.label_font_size, ax=ax, font_family=self.font_family)
        legend_patches = [mpatches.Patch(color="#1e40af", label="Documents")]
        for dom in domains:
            if colormap and cmap_obj:
                c = mcolors.to_hex(cmap_obj(domain_color_idx[dom] / max(len(domains) - 1, 1)))
            else:
                c = self.DOMAIN_COLORS.get(dom, "#6b7280")
            legend_patches.append(mpatches.Patch(color=c, label=dom))
        ax.legend(handles=legend_patches, loc="upper left", fontsize=9)
        ax.set_title("Salience-Aware Cross-Document Knowledge Network\n(Node size = importance | Labels: {} format)".format(label_style), fontsize=self.title_font_size, fontweight='bold', fontfamily=self.font_family)
        ax.axis("off")
        plt.tight_layout()
        return fig

    def render_pyvis_salience(self, filtered_concepts: Optional[List[str]] = None, top_n_nodes: int = 30, physics_enabled: bool = True, colormap: Optional[str] = None, aliases: Optional[Dict[str,str]] = None, label_style: str = "doi") -> str:
        if not PYVIS_AVAILABLE:
            return "<p>PyVis not installed. pip install pyvis</p>"
        G = nx.Graph()
        docs = list(self.kgraph.doc_graphs.keys())
        for doc in docs:
            display = get_display_name(doc, aliases)
            label = get_citation_label(doc, aliases, style=label_style)
            tooltip = f"<b>{label}</b><br>"
            tooltip += f"File: {Path(doc).name}<br>"
            doc_items = [it for it in self.kgraph.doc_graphs[doc]["all_items"] if it.get("value") is not None]
            if doc_items:
                tooltip += "<b>Top values:</b><br>"
                top_vals = sorted(doc_items, key=lambda x: x.get("confidence", 0), reverse=True)[:5]
                for it in top_vals:
                    tooltip += f"- {it.get('physical_quantity', 'unknown')}: {it.get('value')} {it.get('unit', '')} (p.{it.get('page', '?')})<br>"
            G.add_node(display, node_type="doc", domain="DOCUMENT", title=tooltip, orig_doc=doc)
        entities = filtered_concepts or list(self.kgraph.get_all_physical_quantities().keys())[:top_n_nodes]
        for ent in entities:
            stats = self.kgraph.get_summary_stats(ent)
            count = stats.get("count", 0)
            tooltip = f"<b>{ent}</b><br>"
            tooltip += f"Occurrences: {count}<br>"
            if stats.get("count", 0) > 0:
                tooltip += f"Range: {stats.get('min', 0):.2f} - {stats.get('max', 0):.2f}<br>"
                tooltip += f"Mean: {stats.get('mean', 0):.2f}<br>"
                tooltip += f"Std: {stats.get('std', 0):.2f}<br>"
                tooltip += "<b>By document:</b><br>"
                for doc in docs:
                    doc_vals = [it["value"] for it in self.kgraph.doc_graphs[doc]["all_items"] if it.get("physical_quantity") == ent and it.get("value") is not None]
                    if doc_vals:
                        doc_label = get_citation_label(doc, aliases, style=label_style)
                        tooltip += f"- {doc_label}: {min(doc_vals):.2f} to {max(doc_vals):.2f}<br>"
            G.add_node(ent, node_type="entity", domain="PARAMETER", salience=count, title=tooltip)
            for doc in docs:
                if any(item.get("physical_quantity") == ent or item.get("parameter_name") == ent for item in self.kgraph.doc_graphs[doc]["all_items"]):
                    doc_display = get_display_name(doc, aliases)
                    G.add_edge(doc_display, ent, weight=count)
        net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="#000000", cdn_resources='remote')
        if physics_enabled:
            net.barnes_hut(gravity=-1800, spring_length=140, damping=0.85)
        cmap_obj = plt.get_cmap(self._get_colormap(colormap)) if colormap else None
        for i, node in enumerate(G.nodes()):
            salience = G.nodes[node].get("salience", 0.5)
            size = int(15 + min(salience / 2, 1) * 55)
            domain = G.nodes[node].get("domain", "UNKNOWN")
            color = self._get_domain_color(domain, colormap, i, len(G.nodes()))
            title = G.nodes[node].get("title", f"{node}\nSalience: {salience}")
            net.add_node(node, label=node[:25], size=size, color=color, borderWidth=2, title=title)
        for u, v, data in G.edges(data=True):
            w = data.get('weight', 1)
            net.add_edge(u, v, value=max(1, int(w * 2)), width=max(1, int(w)))
        return net.generate_html()

    def plot_quantitative_knowledge_graph_pyvis(self, df: pd.DataFrame, quantity: str, colormap: Optional[str] = None, aliases: Optional[Dict[str,str]] = None, label_style: str = "doi") -> str:
        """PyVis interactive version of quantitative knowledge graph."""
        if not PYVIS_AVAILABLE:
            return "<p>PyVis not installed. pip install pyvis</p>"
        subset = df[df["physical_quantity"] == quantity]
        if subset.empty:
            return "<p>No data for this quantity</p>"
        net = Network(height=self.cfg.pyvis_height, width=self.cfg.pyvis_width, bgcolor="#ffffff", font_color="#000000", cdn_resources='remote')
        if self.cfg.pyvis_physics_enabled:
            net.barnes_hut(gravity=self.cfg.pyvis_gravity, spring_length=self.cfg.pyvis_spring_length, damping=self.cfg.pyvis_damping)
        hub = f"{quantity}_hub"
        net.add_node(hub, label=quantity.replace("_", " ").title(), size=self.cfg.node_size_base_hub, color="#dc2626", borderWidth=3, title=f"Hub: {quantity}")
        cmap_obj = plt.get_cmap(self._get_colormap(colormap))
        for mat in subset["material"].unique():
            if pd.notna(mat) and str(mat).strip() and mat != "Unknown":
                mat_vals = subset[subset["material"] == mat]["value"].tolist()
                tooltip = f"<b>{mat}</b><br>Count: {len(mat_vals)}<br>Range: {min(mat_vals):.2f} - {max(mat_vals):.2f}<br>Mean: {np.mean(mat_vals):.2f}"
                net.add_node(mat, label=mat[:25], size=int(self.cfg.node_size_base_material * self.cfg.node_size_factor), color="#3b82f6", title=tooltip)
                net.add_edge(hub, mat, width=int(len(mat_vals) * self.cfg.edge_width_pyvis))
        for doc in subset["doc_stem"].unique():
            if pd.notna(doc) and str(doc).strip():
                doc_vals = subset[subset["doc_stem"] == doc]["value"].tolist()
                tooltip = f"<b>{doc}</b><br>Count: {len(doc_vals)}<br>Range: {min(doc_vals):.2f} - {max(doc_vals):.2f}<br>Mean: {np.mean(doc_vals):.2f}"
                net.add_node(doc, label=doc[:25], size=int(self.cfg.node_size_base_doc * self.cfg.node_size_factor), color="#10b981", title=tooltip)
                net.add_edge(hub, doc, width=int(len(doc_vals) * self.cfg.edge_width_pyvis))
        top = subset.nlargest(min(25, len(subset)), "value")
        for _, row in top.iterrows():
            leaf = f"{row['value']:.1f} {row['unit']}"
            tooltip = f"<b>{row['value']:.2f} {row['unit']}</b><br>Material: {row['material']}<br>Document: {row['doc_stem']}"
            net.add_node(leaf, label=leaf, size=self.cfg.node_size_base_value, color="#f59e0b", title=tooltip)
            if pd.notna(row["material"]) and str(row["material"]).strip() and row["material"] != "Unknown":
                net.add_edge(row["material"], leaf, width=self.cfg.edge_width_pyvis)
            if pd.notna(row["doc_stem"]) and str(row["doc_stem"]).strip():
                net.add_edge(row["doc_stem"], leaf, width=self.cfg.edge_width_pyvis)
        return net.generate_html()

    def plot_knowledge_network_pyvis(self, df: pd.DataFrame, colormap: Optional[str] = None, aliases: Optional[Dict[str,str]] = None, label_style: str = "doi") -> str:
        """PyVis interactive version of full knowledge network."""
        if not PYVIS_AVAILABLE:
            return "<p>PyVis not installed. pip install pyvis</p>"
        net = Network(height=self.cfg.pyvis_height, width=self.cfg.pyvis_width, bgcolor="#ffffff", font_color="#000000", cdn_resources='remote')
        if self.cfg.pyvis_physics_enabled:
            net.barnes_hut(gravity=self.cfg.pyvis_gravity, spring_length=self.cfg.pyvis_spring_length, damping=self.cfg.pyvis_damping)
        docs = [d for d in df["doc_stem"].unique() if pd.notna(d) and str(d).strip() != "" and str(d).lower() != "nan"]
        for doc in docs:
            net.add_node(doc, label=doc[:25], size=int(self.cfg.node_size_base_doc * self.cfg.node_size_factor), color="#1e40af", title=f"Document: {doc}")
        pqs = [p for p in df["physical_quantity"].unique() if pd.notna(p) and str(p).strip() != "" and str(p).lower() != "nan"]
        for pq in pqs:
            stats = self.kgraph.get_summary_stats(pq)
            tooltip = f"<b>{pq}</b><br>Count: {stats.get('count', 0)}"
            net.add_node(pq, label=pq[:25], size=int(self.cfg.node_size_base_entity * self.cfg.node_size_factor), color=self.DOMAIN_COLORS.get(pq, "#6b7280"), title=tooltip)
        mats = [m for m in df["material"].unique() if pd.notna(m) and str(m).strip() != "" and m != "Unknown" and str(m).lower() != "nan"]
        for mat in mats:
            net.add_node(mat, label=mat[:25], size=int(self.cfg.node_size_base_material * self.cfg.node_size_factor), color="#f59e0b", title=f"Material: {mat}")
        for _, row in df.iterrows():
            doc = row["doc_stem"]; pq = row["physical_quantity"]; mat = row["material"]
            if doc and pq and doc in net.get_nodes() and pq in net.get_nodes():
                net.add_edge(doc, pq, width=self.cfg.edge_width_pyvis)
            if mat and mat != "Unknown" and doc and mat in net.get_nodes() and doc in net.get_nodes():
                net.add_edge(doc, mat, width=self.cfg.edge_width_pyvis)
            if mat and mat != "Unknown" and pq and mat in net.get_nodes() and pq in net.get_nodes():
                net.add_edge(pq, mat, width=self.cfg.edge_width_pyvis)
        return net.generate_html()

    def _get_domain_color(self, domain: str, colormap: Optional[str] = None, index: int = 0, total: int = 1) -> str:
        if colormap and total > 1:
            cmap = plt.get_cmap(self._get_colormap(colormap))
            return mcolors.to_hex(cmap(index / max(total - 1, 1)))
        return self.DOMAIN_COLORS.get(domain, "#6b7280")

    def plot_contradiction_matrix(self, quantity: Optional[str] = None, colormap: Optional[str] = None) -> go.Figure:
        df = self.extract_dataframe()
        if quantity:
            df = df[df["physical_quantity"] == quantity]
        if df.empty:
            return go.Figure().update_layout(title="No data")
        docs = df["doc_stem"].unique()
        if len(docs) < 2:
            return go.Figure().update_layout(title="Need at least 2 documents")
        mat = np.zeros((len(docs), len(docs)))
        for i, d1 in enumerate(docs):
            v1 = df[df["doc_stem"] == d1]["value"].mean()
            for j, d2 in enumerate(docs):
                if i == j:
                    continue
                v2 = df[df["doc_stem"] == d2]["value"].mean()
                if v2 != 0 and not np.isnan(v1) and not np.isnan(v2):
                    mat[i,j] = abs(v1 - v2) / v2
        fig = go.Figure(data=go.Heatmap(z=mat, x=docs, y=docs, colorscale=self._get_plotly_colorscale(colormap), hoverongaps=False))
        fig.update_layout(title=f"Contradiction Matrix for {quantity if quantity else 'All Quantities'}", font=dict(family=self.font_family, size=self.font_size), height=600, width=600)
        return fig

    def plot_consensus_waterfall(self, quantity: Optional[str] = None, colormap: Optional[str] = None) -> go.Figure:
        df = self.extract_dataframe()
        if quantity:
            df = df[df["physical_quantity"] == quantity]
        if df.empty:
            return go.Figure().update_layout(title="No data")
        grouped = df.groupby(["material", "physical_quantity"])["value"].agg(["mean", "std", "count"]).reset_index()
        grouped = grouped.sort_values("count", ascending=False).head(10)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=grouped["material"] + " (" + grouped["physical_quantity"] + ")", y=grouped["mean"], error_y=dict(type='data', array=grouped["std"]), marker_color="#059669", text=[f"n={c}" for c in grouped["count"]], textposition="outside"))
        fig.update_layout(title="Cross-Document Consensus (mean +- std)", yaxis_title="Value", xaxis_title="Material (Quantity)", font=dict(family=self.font_family, size=self.font_size))
        return fig

    def _get_context_embeddings(self, embedding_fn: Callable, df: pd.DataFrame, quantity: Optional[str] = None) -> Tuple[np.ndarray, pd.DataFrame]:
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

    def plot_tsne(self, embedding_fn: Callable, quantity: Optional[str] = None, colormap: Optional[str] = None, figsize: Tuple[int,int] = (10,8)) -> Optional[plt.Figure]:
        if not SKLEARN_AVAILABLE:
            return None
        df = self.extract_dataframe()
        embs, df_use = self._get_context_embeddings(embedding_fn, df, quantity)
        if len(embs) < 5:
            return None
        tsne = TSNE(n_components=2, perplexity=min(30, len(embs)-1), random_state=42)
        coords = tsne.fit_transform(embs)
        fig, ax = plt.subplots(figsize=figsize)
        materials = df_use["material"].unique()
        cmap = plt.get_cmap(self._get_colormap(colormap))
        for i, mat in enumerate(materials):
            mask = df_use["material"] == mat
            color = mcolors.to_hex(cmap(i / max(len(materials)-1, 1))) if len(materials)>1 else "#3b82f6"
            ax.scatter(coords[mask,0], coords[mask,1], c=color, label=mat, alpha=0.8, s=80, edgecolors='white')
        for (_, row), coord in zip(df_use.iterrows(), coords):
            ax.annotate(f"{row['value']:.0f}", (coord[0], coord[1]), fontsize=self.label_font_size-1, alpha=0.8)
        ax.legend(loc='best', fontsize=self.label_font_size)
        ax.set_title(f"t-SNE of Extraction Contexts{' ('+quantity+')' if quantity else ''}", fontsize=self.title_font_size, fontweight='bold')
        ax.axis("off")
        plt.tight_layout()
        return fig

    def plot_pca(self, embedding_fn: Callable, quantity: Optional[str] = None, colormap: Optional[str] = None, figsize: Tuple[int,int] = (10,8)) -> Optional[plt.Figure]:
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
            color = mcolors.to_hex(cmap(i / max(len(materials)-1, 1))) if len(materials)>1 else "#3b82f6"
            ax.scatter(coords[mask,0], coords[mask,1], c=color, label=mat, alpha=0.8, s=80, edgecolors='white')
        for (_, row), coord in zip(df_use.iterrows(), coords):
            ax.annotate(f"{row['value']:.0f}", (coord[0], coord[1]), fontsize=self.label_font_size-1, alpha=0.8)
        ax.legend(loc='best', fontsize=self.label_font_size)
        ax.set_title(f"PCA of Extraction Contexts{' ('+quantity+')' if quantity else ''}\nPC1: {var_ratio[0]:.1%}, PC2: {var_ratio[1]:.1%}", fontsize=self.title_font_size, fontweight='bold')
        ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%})")
        ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%})")
        plt.tight_layout()
        return fig

    def plot_umap(self, embedding_fn: Callable, quantity: Optional[str] = None, colormap: Optional[str] = None, figsize: Tuple[int,int] = (10,8)) -> Optional[plt.Figure]:
        if not UMAP_AVAILABLE:
            return None
        df = self.extract_dataframe()
        embs, df_use = self._get_context_embeddings(embedding_fn, df, quantity)
        if len(embs) < 5:
            return None
        reducer = umap.UMAP(n_neighbors=min(15, len(embs)-1), min_dist=0.1, random_state=42)
        coords = reducer.fit_transform(embs)
        fig, ax = plt.subplots(figsize=figsize)
        materials = df_use["material"].unique()
        cmap = plt.get_cmap(self._get_colormap(colormap))
        for i, mat in enumerate(materials):
            mask = df_use["material"] == mat
            color = mcolors.to_hex(cmap(i / max(len(materials)-1, 1))) if len(materials)>1 else "#3b82f6"
            ax.scatter(coords[mask,0], coords[mask,1], c=color, label=mat, alpha=0.8, s=80, edgecolors='white')
        for (_, row), coord in zip(df_use.iterrows(), coords):
            ax.annotate(f"{row['value']:.0f}", (coord[0], coord[1]), fontsize=self.label_font_size-1, alpha=0.8)
        ax.legend(loc='best', fontsize=self.label_font_size)
        ax.set_title(f"UMAP of Extraction Contexts{' ('+quantity+')' if quantity else ''}", fontsize=self.title_font_size, fontweight='bold')
        ax.axis("off")
        plt.tight_layout()
        return fig

    def plot_parallel_categories(self, df: pd.DataFrame, colormap: Optional[str] = None) -> go.Figure:
        if df.empty:
            return go.Figure().update_layout(title="No data")
        cat_df = df[["physical_quantity", "material", "doc_stem"]].copy()
        cat_df = cat_df.dropna()
        if cat_df.empty:
            return go.Figure().update_layout(title="Insufficient categorical data")
        pq_codes = {pq: i for i, pq in enumerate(sorted(cat_df["physical_quantity"].unique()))}
        cat_df["pq_code"] = cat_df["physical_quantity"].map(pq_codes)
        fig = px.parallel_categories(cat_df, dimensions=["physical_quantity", "material"], color="pq_code", color_continuous_scale=self._get_plotly_colorscale(colormap), title="Parallel Categories: Quantities and Materials")
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
        return fig

    def plot_violin(self, df: pd.DataFrame, colormap: Optional[str] = None) -> go.Figure:
        if df.empty:
            return go.Figure().update_layout(title="No data")
        num_df = df[df["physical_quantity"].isin(["laser_power", "scan_speed", "yield_strength", "tensile_strength", "hardness"])]
        if num_df.empty:
            return go.Figure().update_layout(title="No numerical data for violin plot")
        fig = px.violin(num_df, x="physical_quantity", y="value", color="material", box=True, points="all", title="Violin Plot of Values by Material")
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
        return fig

    def plot_chord_cooccurrence(self, filtered_concepts: Optional[List[str]] = None, top_n: int = 14, colormap: Optional[str] = None) -> go.Figure:
        if filtered_concepts:
            entities = filtered_concepts[:top_n]
        else:
            all_pq = self.kgraph.get_all_physical_quantities()
            entities = [pq for pq, _ in sorted(all_pq.items(), key=lambda x: x[1], reverse=True)[:top_n]]
        if not entities:
            return go.Figure().update_layout(title="No entity co-occurrence data")
        n = len(entities)
        node_to_idx = {node: i for i, node in enumerate(entities)}
        adj = np.zeros((n, n))
        for doc in self.kgraph.doc_graphs:
            present = [ent for ent in entities if any(item.get("physical_quantity") == ent or item.get("parameter_name") == ent for item in self.kgraph.doc_graphs[doc]["all_items"])]
            for i, e1 in enumerate(present):
                for j, e2 in enumerate(present):
                    if i != j:
                        adj[node_to_idx[e1]][node_to_idx[e2]] += 1
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        cmap_obj = plt.get_cmap(self._get_colormap(colormap))
        fig = go.Figure()
        for i, ent in enumerate(entities):
            color = mcolors.to_hex(cmap_obj(i / max(n - 1, 1)))
            fig.add_trace(go.Barpolar(r=[1], theta=[np.degrees(angles[i])], width=[10], marker_color=color, name=ent, opacity=0.9, showlegend=False))
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j] > 0:
                    fig.add_trace(go.Scatterpolar(r=[0.2, 0.6, 0.2], theta=[np.degrees(angles[i]), np.degrees((angles[i]+angles[j])/2), np.degrees(angles[j])], mode='lines', line=dict(color='rgba(100,100,100,0.3)', width=min(adj[i][j], 3)), showlegend=False))
        fig.update_layout(polar=dict(radialaxis=dict(visible=False), angularaxis=dict(visible=False)), title=f"Salience-Aware Chord Diagram (Top {n} Concepts)", height=700, width=700)
        return fig

    def plot_timeline(self, colormap: Optional[str] = None) -> go.Figure:
        df = self.extract_dataframe()
        if df.empty:
            return go.Figure().update_layout(title="No data")
        years = {}
        for doc_id in self.kgraph.doc_graphs.keys():
            match = re.search(r'\b(19|20)\d{2}\b', doc_id)
            if match:
                years[doc_id] = int(match.group(0))
            else:
                years[doc_id] = 2023
        df["year"] = df["doc"].map(years)
        top_q = df["physical_quantity"].value_counts().head(5).index.tolist()
        df_top = df[df["physical_quantity"].isin(top_q)]
        fig = px.scatter(df_top, x="year", y="physical_quantity", color="material", title="Temporal Distribution of Quantities by Material", labels={"year": "Estimated Year", "physical_quantity": "Physical Quantity"}, color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
        return fig


    def plot_page_coverage_heatmap(self, doc_trees, retrieved_nodes):


        if not doc_trees or not retrieved_nodes:
            return go.Figure().update_layout(title="No coverage data")
        doc_names = sorted(list(set(t.get("doc_id", t.get("doc_name", "unknown")) for t in doc_trees)))
        max_pages = 0
        for tree in doc_trees:
            doc_id = tree.get("doc_id", tree.get("doc_name", "unknown"))
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
        fig = go.Figure(data=go.Heatmap(z=coverage, x=list(range(1, max_pages + 1)), y=doc_labels, colorscale=[[0, "#f3f4f6"], [1, "#059669"]], showscale=False, hovertemplate="Doc: %{y}<br>Page: %{x}<br>Retrieved: %{z}<extra></extra>"))
        fig.update_layout(title="Page Coverage Heatmap (Retrieved Pages per Document)", xaxis_title="Page Number", yaxis_title="Document", font=dict(family=self.font_family, size=self.font_size), height=max(400, len(doc_names) * 40))
        return fig
    def plot_retrieval_sankey(self, query: str, relevant_docs, retrieved_nodes, extracted_items):
        if not relevant_docs and not retrieved_nodes:
            return go.Figure().update_layout(title="No retrieval data available")
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
            sources.append(0); targets.append(label_index[doc_name]); vals.append(max(1, int(score * 10)))
        for r in retrieved_nodes:
            doc_id = r.get("doc_id"); node_id = r.get("node_id", "unknown"); key = f"{doc_id}:{node_id}"
            conf = r.get("confidence", 0.5)
            if doc_id in label_index and key in label_index:
                sources.append(label_index[doc_id]); targets.append(label_index[key]); vals.append(max(1, int(conf * 10)))
        node_to_pq = defaultdict(set)
        for item in extracted_items:
            pq = item.get("physical_quantity", "unknown")
            doc_id = item.get("doc_source", item.get("doc_id", "unknown"))
            for r in retrieved_nodes:
                if r.get("doc_id") == doc_id:
                    node_id = r.get("node_id", "unknown"); key = f"{doc_id}:{node_id}"
                    node_to_pq[key].add(f"pq:{pq}")
        for node_key, pq_set in node_to_pq.items():
            for pq_key in pq_set:
                if node_key in label_index and pq_key in label_index:
                    sources.append(label_index[node_key]); targets.append(label_index[pq_key]); vals.append(1)
        for pq_key in pq_nodes_list:
            sources.append(label_index[pq_key]); targets.append(label_index["Answer"]); vals.append(max(1, len(pq_groups.get(pq_key.replace("pq:", ""), []))))
        # Publication-quality color scheme
        node_colors = ["#1e3a5f"]  # Query: deep navy
        node_colors += ["#2563eb"] * len(doc_nodes)  # Docs: blue
        node_colors += ["#059669"] * len(node_labels_list)  # Nodes: emerald
        node_colors += ["#dc2626"] * len(pq_nodes_list)  # PQ: red
        node_colors += ["#7c3aed"]  # Answer: purple
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

    def plot_node_confidence_distribution(self, retrieved_nodes):
        if not retrieved_nodes:
            return go.Figure().update_layout(title="No node confidence data")
        confidences = [r.get("confidence", 0) for r in retrieved_nodes]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=confidences, nbinsx=20, marker_color="#3b82f6", opacity=0.75, name="All Nodes"))
        fig.add_vline(x=0.5, line_dash="dash", line_color="#ef4444", annotation_text="Typical Threshold")
        fig.update_layout(title="Node Selection Confidence Distribution", xaxis_title="Confidence Score", yaxis_title="Count", font=dict(family=self.font_family, size=self.font_size), showlegend=False)
        return fig

    def plot_doc_filter_scores(self, relevant_docs, all_doc_count):
        if not relevant_docs:
            return go.Figure().update_layout(title="No document filter scores")
        docs = [Path(d).stem for d, _ in relevant_docs]
        scores = [s for _, s in relevant_docs]
        colors = ["#10b981" if s > 0.5 else "#f59e0b" if s > 0.2 else "#ef4444" for s in scores]
        fig = go.Figure(go.Bar(x=docs, y=scores, marker_color=colors, text=[f"{s:.3f}" for s in scores], textposition="outside"))
        fig.add_hline(y=0.5, line_dash="dash", line_color="#ef4444", annotation_text="High Relevance")
        fig.add_hline(y=0.2, line_dash="dot", line_color="#f59e0b", annotation_text="Medium Relevance")
        fig.update_layout(title=f"Two-Stage Document Retrieval Scores (showing {len(docs)} of {all_doc_count})", xaxis_title="Document", yaxis_title="Relevance Score", font=dict(family=self.font_family, size=self.font_size), height=450)
        return fig

    def plot_retrieval_tree_highlight(self, annotated_trees, retrieved_nodes, doc_id=None):
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
        legend_elements = [Patch(facecolor="#ef4444", label="Retrieved Node"), Patch(facecolor="#93c5fd", label="Has Quantitative Data"), Patch(facecolor="#e5e7eb", label="Other Node")]
        ax.legend(handles=legend_elements, loc='upper right')
        ax.set_title(f"Retrieval Tree: {Path(doc_id).stem if doc_id else 'Document'}", fontsize=self.title_font_size, fontweight='bold')
        ax.axis("off")
        plt.tight_layout()
        return fig

    def plot_semantic_vs_vectorless(self, query, relevant_docs, annotated_trees, embedding_fn=None):
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
        fig = go.Figure()
        doc_labels = [Path(d).stem for d in doc_names]
        fig.add_trace(go.Scatter(x=keyword_scores, y=semantic_scores, mode='markers+text', text=doc_labels, textposition="top center", marker=dict(size=14, color="#3b82f6"), name="Documents"))
        min_val = min(min(keyword_scores), min(semantic_scores))
        max_val = max(max(keyword_scores), max(semantic_scores))
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', line=dict(dash='dash', color='#ef4444'), name="Agreement Line"))
        fig.update_layout(title="Semantic (Embedding) vs Vectorless (Keyword/Heuristic) Retrieval Scores", xaxis_title="Vectorless Score (Keyword/Heuristic)", yaxis_title="Semantic Score (Cosine Similarity)", font=dict(family=self.font_family, size=self.font_size), height=500)
        return fig


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

def render_sidebar():
    with st.sidebar:
        st.markdown("### Configuration")
        model_keys = list(LOCAL_LLM_OPTIONS.keys())
        if "llm_model_choice" not in st.session_state:
            st.session_state.llm_model_choice = model_keys[2]
        selected = st.selectbox("Select Local LLM", options=model_keys, index=model_keys.index(st.session_state.llm_model_choice), key="llm_model_select")
        st.session_state.llm_model_choice = selected
        st.checkbox("Use 4-bit quantization (if Transformers)", value=True, key="use_4bit")
        st.slider("Confidence threshold", 0.3, 0.9, 0.55, 0.05, key="min_confidence")
        max_chars = st.slider("Max text length per retrieved section (characters)", min_value=1000, max_value=50000, value=20000, step=1000, help="Larger values give more context but use more memory/LLM tokens.")
        st.session_state.max_retrieval_chars = max_chars
        st.checkbox("Show reasoning trace", value=True, key="show_trace")
        st.checkbox("Show tree navigation", value=True, key="show_tree_nav")
        st.checkbox("Enable two-stage retrieval (semantic)", value=True, key="two_stage")

        st.markdown("#### Visualization Settings")
        st.selectbox("Default colormap", list(PublicationVisualizationEngine.COLORMAP_OPTIONS.keys()), index=0, key="viz_colormap")
        st.selectbox("Document label style", ["doi", "number", "alias", "short"], index=0, key="viz_label_style")
        st.slider("Top N concepts", 5, 100, 25, key="viz_top_n")
        st.multiselect("Filter domains", options=["laser_power","scan_speed","yield_strength","tensile_strength","hardness","temperature","energy_density"], default=["laser_power","scan_speed","yield_strength"], key="viz_domains")

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

@st.cache_resource(show_spinner="Initializing LLM...")
def get_cached_llm(model_choice: str, use_4bit: bool):
    internal = LOCAL_LLM_OPTIONS[model_choice]
    return HybridLLM(model_key=internal, use_4bit=use_4bit)


def run_streamlit():
    st.set_page_config(page_title="DECLARMIMA v17.2 - Unified Robust RAG + Query Viz", layout="wide")
    st.markdown("# DECLARMIMA v17.2 - Unified Robust Vectorless RAG + Query-Driven Visualizations")
    st.caption("Vectorless retrieval with semantic fallback. 35+ chart types. Query-aware contextual visualization. Concept normalization. Retrieval diagnostics.")

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
            idx = FastHierarchicalIndex(llm=llm)
            async def build_index():
                return await idx.build_from_pdfs_fast(st.session_state.query_processor["files"], max_workers=4)
            trees = asyncio.run(build_index())
            st.session_state.query_processor["index"] = idx
            st.session_state.query_processor["doc_trees"] = trees
            progress.progress(0.5)
            extractor = UniversalLLMExtractor(llm)
            kg = QuantitativeKnowledgeGraph()
            all_items = []
            two_stage = TwoStageRetriever(llm=llm)
            for doc_name, tree in trees.items():
                leaf_texts = []
                def collect_leaves(node: PageNode):
                    if not node.children:
                        text = node.get_text()
                        if text:
                            leaf_texts.append({"full_text": text, "page_start": node.page_start, "doc_id": doc_name, "section_title": node.title})
                    for c in node.children:
                        collect_leaves(c)
                collect_leaves(tree)
                initial_prompt = "Extract ALL quantitative parameters: laser power, scan speed, VED, AED, LED, layer thickness, hatch distance, temperature, enthalpy, viscosity, thermal conductivity, density, yield strength, UTS, elongation, hardness, modulus, stacking fault energy, ideal shear strength, corrosion potential (Ecorr), pitting potential (Epit), repassivation potential (Erp), breakdown potential (Ebr), corrosion current density (Jcorr), polarization resistance (Rp), PREN, phase fractions (austenite, ferrite), grain size, porosity, relative density, Sauter mean diameter (SMD), spray penetration, plume height, film thickness, absorption coefficient, Young's modulus, Poisson's ratio, CTE. Include units, material names, and page numbers. Also extract alloy names, process methods (LPBF, DED, PFI, GDI, FEM, MD), and phases (Ti3Au, Al3Zr, beta-Ti3Au, etc.)."
                items = extractor.extract_from_chunks(leaf_texts, initial_prompt)
                all_items.extend(items)
                kg.add_extractions(doc_name, items)
                if tree.metadata:
                    kg.add_document_metadata(doc_name, tree.metadata)
                    two_stage.index_document(doc_name, tree.metadata, tree.summary)
                else:
                    alloys = list(set(item.material for item in items if item.material))
                    meta = DocumentMetadata(doc_name=doc_name, alloys=alloys)
                    kg.add_document_metadata(doc_name, meta)
                    two_stage.index_document(doc_name, meta, tree.summary)
            st.session_state.knowledge_graph = kg
            st.session_state.two_stage_retriever = two_stage
            progress.progress(0.8)
            annotated = []
            for doc_name, tree in trees.items():
                ann = kg.to_tree_annotation(tree, max_chars=max_retrieval_chars)
                ann["doc_id"] = doc_name
                ann["doc_name"] = doc_name
                ann["metadata"] = tree.metadata.dict() if tree.metadata else {}
                annotated.append(ann)
            st.session_state.annotated_trees = annotated
            progress.progress(1.0)
            st.success(f"Indexed {len(trees)} documents with {len(all_items)} quantitative items")
            if "doc_aliases" not in st.session_state:
                st.session_state.doc_aliases = {}
            with st.expander("Detected Physical Quantities and Materials", expanded=True):
                pq_counts = kg.get_all_physical_quantities()
                if pq_counts:
                    st.write("**Physical Quantities:**")
                    for pq, count in sorted(pq_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
                        st.write(f"- `{pq}`: {count} occurrences")
                mat_dict = kg.get_all_materials()
                if mat_dict:
                    st.write("**Materials/Alloys per document:**")
                    for doc, mats in mat_dict.items():
                        if mats:
                            st.write(f"- {doc}: {', '.join(mats)}")
            if SENTENCE_TRANSFORMERS_AVAILABLE and st.session_state.embedding_model is None:
                st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")


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
        run_query = False
        if active_prompt:
            cached = st.session_state.cached_query_result
            has_valid_cache = cached and cached.get("prompt") == active_prompt and "answer" in cached
            if not has_valid_cache:
                run_query = True

        answer = None
        extracted_values = []
        retrieved = []
        items = []
        relevant_docs = []

        if run_query:
            with st.chat_message("assistant"):
                progress = st.progress(0)
                progress.text("Initializing LLM...")
                llm = get_cached_llm(st.session_state.llm_model_choice, st.session_state.get("use_4bit", True))
                progress.progress(0.1)
                if st.session_state.get("two_stage", True) and st.session_state.two_stage_retriever is not None:
                    progress.text("Stage 1: Document filtering (vectorless + semantic)...")
                    relevant_docs = st.session_state.two_stage_retriever.retrieve_relevant_docs(active_prompt, top_k=8)
                    st.caption(f"Selected {len(relevant_docs)} relevant documents out of {len(st.session_state.annotated_trees)}.")
                    filtered_trees = [t for t in st.session_state.annotated_trees if t.get("doc_id") in [d[0] for d in relevant_docs]]
                else:
                    filtered_trees = st.session_state.annotated_trees
                    relevant_docs = [(t.get("doc_id", t.get("doc_name", "unknown")), 1.0) for t in filtered_trees]
                progress.progress(0.3)
                retriever = HierarchicalTreeRetriever(llm, max_results=30, max_text_chars=max_retrieval_chars)
                retrieved = asyncio.run(retriever.retrieve_quantitative(active_prompt, filtered_trees))
                progress.progress(0.6)
                extractor = UniversalLLMExtractor(llm)
                items = []
                for r in retrieved:
                    items.extend(extractor.extract_from_chunks([r], active_prompt))
                min_conf = st.session_state.get("min_confidence", 0.55)
                items = [i for i in items if i.confidence >= min_conf]
                progress.progress(0.8)
                synthesizer = LLMReasoningSynthesizer(llm)
                extracted_values = []
                for item in items:
                    if item.item_type == "quantitative" and item.value is not None:
                        phys_q = item.physical_quantity or synthesizer.phys_classifier.classify(item.parameter_name, item.unit, item.context)
                        extracted_values.append(ExtractedValue(query=active_prompt, value=item.value, unit=item.unit or "", physical_quantity=phys_q, parameter_name=item.parameter_name, material=item.material, confidence=item.confidence, context=item.context, doc_name=item.doc_source, page=item.page, section_title=item.section_title))
                if extracted_values:
                    report = QueryReport(query=active_prompt, total_docs=len(st.session_state.annotated_trees), docs_with_results=len(set(v.doc_name for v in extracted_values)), all_values=extracted_values, consensus={}, processing_time_sec=0.0)
                    answer = synthesizer.generate_human_conclusion(active_prompt, report)
                else:
                    answer = synthesizer.synthesize(active_prompt, items)
                progress.progress(1.0, text="Done!")
                st.markdown(answer)
                st.session_state.cached_query_result = {"prompt": active_prompt, "relevant_docs": relevant_docs, "retrieved": retrieved, "items": [i.model_dump() for i in items], "extracted_values": [v.model_dump() for v in extracted_values], "answer": answer}
                st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            if active_prompt and st.session_state.cached_query_result and "answer" in st.session_state.cached_query_result:
                cached = st.session_state.cached_query_result
                with st.chat_message("assistant"):
                    st.markdown(cached["answer"])
                answer = cached["answer"]
                relevant_docs = cached.get("relevant_docs", [])
                retrieved = cached.get("retrieved", [])
                raw_items = cached.get("items", [])
                if raw_items and isinstance(raw_items[0], dict):
                    items = [UniversalExtractionItem(**d) for d in raw_items]
                else:
                    items = raw_items
                raw_vals = cached.get("extracted_values", [])
                if raw_vals and isinstance(raw_vals[0], dict):
                    extracted_values = [ExtractedValue(**d) for d in raw_vals]
                else:
                    extracted_values = raw_vals
            else:
                if not active_prompt:
                    st.info("Ask a question about the documents.")
                    return


        st.markdown("---")
        st.subheader("Quantitative Results")
        display_mode = st.radio("Display format", ["Table", "JSON", "Human Summary"], horizontal=True, key="display_mode")
        if display_mode == "Table" and extracted_values:
            df_disp = pd.DataFrame([{"Document": v.doc_name, "Page": v.page, "Value": f"{v.value:.2f}", "Unit": v.unit, "Physical Quantity": PhysicalQuantityClassifier().get_human_readable(v.physical_quantity), "Material": v.material or "", "Parameter": v.parameter_name or "", "Confidence": f"{v.confidence:.2f}"} for v in extracted_values])
            st.dataframe(df_disp, use_container_width=True, key="df_df_disp")

        elif display_mode == "JSON" and extracted_values:
            st.json([v.model_dump() for v in extracted_values])
        elif display_mode == "Human Summary" and extracted_values:
            synthesizer = LLMReasoningSynthesizer(get_cached_llm(st.session_state.llm_model_choice, st.session_state.get("use_4bit", True)))
            report = QueryReport(query=active_prompt, total_docs=len(st.session_state.annotated_trees), docs_with_results=len(set(v.doc_name for v in extracted_values)), all_values=extracted_values, consensus={}, processing_time_sec=0.0)
            conclusion = synthesizer.generate_human_conclusion(active_prompt, report)
            st.markdown(conclusion)

        # =============================================================================
        # QUERY-AWARE VISUALIZATIONS
        # =============================================================================
        if active_prompt and st.session_state.get("cached_query_result"):
            query_ctx = QueryContext.from_cache(st.session_state.cached_query_result)
            
            if query_ctx.has_data():
                st.markdown("---")
                st.subheader("🎯 Query-Focused Visualizations")
                st.caption(f"**Focused on:** {active_prompt[:90]}{'...' if len(active_prompt)>90 else ''}")

                viz_tabs = st.tabs([
                    "🌐 Interactive Knowledge Graph", 
                    "☀️ Sunburst Hierarchy", 
                    "🔄 Provenance Flow",
                    "📊 Quick Charts",
                    "🌍 Global Dashboard"
                ])

                aliases = st.session_state.get("doc_aliases", {})
                label_style = st.session_state.get("viz_label_style", "doi")
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
                    aliases=aliases,
                    label_style=label_style
                )
                viz = PublicationVisualizationEngine(st.session_state.knowledge_graph, config=config)
                df_all = viz.extract_dataframe(aliases=aliases, label_style=label_style)

                with viz_tabs[0]:
                    st.markdown("**Interactive Query Knowledge Graph** (Click pink value nodes for full context modal)")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if PYVIS_AVAILABLE:
                            html_graph = viz.plot_query_knowledge_graph_pyvis(query_ctx)
                            st.components.v1.html(html_graph, height=820, scrolling=True, key="html_html_graph")

                            st.download_button(
                                "Download Interactive Graph HTML", 
                                html_graph.encode('utf-8'), 
                                "query_knowledge_graph.html", 
                                mime="text/html",
                                key="dl_pyvis_query"
                            )

                        else:
                            fig_kg = viz.plot_query_knowledge_graph(query_ctx)
                            st.pyplot(fig_kg, key="py_fig_kg")

                            buf = BytesIO()
                            fig_kg.savefig(buf, format="png", dpi=config.figure_dpi, bbox_inches='tight')
                            st.download_button("Download Query KG (PNG)", buf.getvalue(), "query_knowledge_graph.png", mime="image/png", key="dl_kg")

                    with col2:
                        st.markdown("### Legend")
                        st.markdown("""
                        - **Purple** → Your Query (Center)
                        - **Green** → Relevant Documents
                        - **Blue** → Physical Quantities  
                        - **Orange** → Materials/Alloys
                        - **Pink** → Extracted Values (clickable)
                        """)
                        st.caption("**Tip:** Hover for tooltips • Click pink nodes for context")

                with viz_tabs[1]:
                    fig_sun = viz.plot_query_sunburst(query_ctx)
                    st.plotly_chart(fig_sun, use_container_width=True, key="plc_fig_sun")

                    st.caption("This sunburst shows the hierarchy of quantities → materials → documents for your specific query.")

                with viz_tabs[2]:
                    st.subheader("🔄 Retrieval Provenance Flow")
                    cached = st.session_state.cached_query_result
                    fig_sankey = viz.plot_retrieval_sankey(
                        active_prompt, 
                        cached.get("relevant_docs", []), 
                        cached.get("retrieved", []), 
                        cached.get("items", [])
                    )
                    st.plotly_chart(fig_sankey, use_container_width=True, key="plc_fig_sankey")


                with viz_tabs[3]:
                    st.markdown("### Quick Relevant Charts")
                    for pq in query_ctx.physical_quantities[:3]:
                        fig = viz.plot_quantitative_histogram(df_all, pq)
                        st.plotly_chart(fig, use_container_width=True, key="plc_fig")


                with viz_tabs[4]:
                    st.info("Full corpus visualizations are available in the dashboard below ↓")
            else:
                st.info("No quantitative data extracted for this query yet. Run a query to see query-focused visualizations.")

        # =============================================================================
        # GLOBAL DASHBOARD (Existing, now below query-focused section)
        # =============================================================================
        if st.session_state.knowledge_graph and st.session_state.annotated_trees:
            st.markdown("---")
            with st.expander("Document Aliases & Label Editor", expanded=False):
                st.markdown("Rename documents for cleaner visualization labels. DOI underscores are auto-converted to slashes.")
                doc_list = sorted(list(st.session_state.knowledge_graph.doc_graphs.keys()))
                alias_style = st.session_state.get("viz_label_style", "doi")
                for i, doc_id in enumerate(doc_list):
                    cols = st.columns([3, 2, 1])
                    original = normalize_doi_display(Path(doc_id).stem)
                    current_alias = st.session_state.doc_aliases.get(doc_id, "")
                    with cols[0]:
                        st.caption(f"Original: {original}")
                    with cols[1]:
                        new_alias = st.text_input(f"Alias {i}", value=current_alias, placeholder="e.g. Smith et al. 2024", label_visibility="collapsed", key=f"alias_{doc_id}")
                        if new_alias:
                            st.session_state.doc_aliases[doc_id] = new_alias
                        elif doc_id in st.session_state.doc_aliases:
                            del st.session_state.doc_aliases[doc_id]
                    with cols[2]:
                        preview = get_display_name(doc_id, st.session_state.doc_aliases)
                        st.caption(f"Preview: {preview}")
                if st.button("Reset all aliases"):
                    st.session_state.doc_aliases = {}
                    st.rerun()
            st.markdown("---")
            st.subheader("Publication-Quality Visualisation Dashboard")
            aliases = st.session_state.get("doc_aliases", {})
            label_style = st.session_state.get("viz_label_style", "doi")
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
                aliases=aliases,
                label_style=label_style
            )
            viz = PublicationVisualizationEngine(st.session_state.knowledge_graph, config=config)
            df_all = viz.extract_dataframe(aliases=aliases, label_style=label_style)
            if not df_all.empty:
                selected_qty = st.selectbox("Filter by physical quantity", options=["All"] + sorted(df_all["physical_quantity"].unique()), key="viz_qty_filter")
                group_by = st.selectbox("Group by", ["material", "doc_stem"], key="viz_group_by")
                colormap = st.session_state.get("viz_colormap", "viridis")
                tabs = st.tabs(["Histograms & Bars", "Pie & Donut", "Sunburst & Treemap", "Radar & Chord", "Contradiction & Consensus", "Networks", "Embedding Spaces", "Scatter & Violin", "Entity Explorer", "Retrieval Diagnostics"])

                with tabs[0]:
                    if selected_qty != "All":
                        fig_hist = viz.plot_quantitative_histogram(df_all, selected_qty, group_by, colormap)
                        st.plotly_chart(fig_hist, use_container_width=True, key="plc_fig_hist")

                    fig_bar = viz.plot_quantities_bar(df_all, colormap)
                    st.plotly_chart(fig_bar, use_container_width=True, key="plc_fig_bar")

                    fig_mat = viz.plot_material_counts(df_all, colormap)
                    st.plotly_chart(fig_mat, use_container_width=True, key="plc_fig_mat")


                with tabs[1]:
                    fig_pie = viz.plot_quantity_distribution_pie(colormap)
                    st.plotly_chart(fig_pie, use_container_width=True, key="plc_fig_pie")

                    fig_donut = viz.plot_material_distribution_donut(colormap)
                    st.plotly_chart(fig_donut, use_container_width=True, key="plc_fig_donut")


                with tabs[2]:
                    if selected_qty != "All":
                        fig_sun = viz.plot_quantitative_sunburst(df_all, selected_qty, colormap)
                        st.plotly_chart(fig_sun, use_container_width=True, key="plc_fig_sun_2")

                    fig_sun_all = viz.plot_sunburst_hierarchy(df_all, colormap)
                    st.plotly_chart(fig_sun_all, use_container_width=True, key="plc_fig_sun_all")

                    fig_treemap = viz.plot_treemap(colormap)
                    st.plotly_chart(fig_treemap, use_container_width=True, key="plc_fig_treemap")

                    fig_treemap_mat = viz.plot_treemap_materials(df_all, colormap)
                    st.plotly_chart(fig_treemap_mat, use_container_width=True, key="plc_fig_treemap_mat")


                with tabs[3]:
                    if selected_qty != "All":
                        fig_radar_qty = viz.plot_quantitative_radar(df_all, selected_qty, colormap)
                        st.plotly_chart(fig_radar_qty, use_container_width=True, key="plc_fig_radar_qty")

                    fig_radar_mat = viz.plot_radar_by_material(colormap)
                    st.plotly_chart(fig_radar_mat, use_container_width=True, key="plc_fig_radar_mat")

                    fig_radar_doc = viz.plot_document_radar(colormap)
                    st.plotly_chart(fig_radar_doc, use_container_width=True, key="plc_fig_radar_doc")

                    fig_chord = viz.plot_chord_cooccurrence(None, st.session_state.get("viz_top_n", 25), colormap)
                    st.plotly_chart(fig_chord, use_container_width=True, key="plc_fig_chord")


                with tabs[4]:
                    fig_contra = viz.plot_contradiction_matrix(None if selected_qty=="All" else selected_qty, colormap)
                    st.plotly_chart(fig_contra, use_container_width=True, key="plc_fig_contra")

                    fig_cons = viz.plot_consensus_waterfall(None if selected_qty=="All" else selected_qty, colormap)
                    st.plotly_chart(fig_cons, use_container_width=True, key="plc_fig_cons")


                with tabs[5]:
                    st.markdown("### Network Visualizations")
                    net_subtabs = st.tabs(["Quantitative KG (NetworkX)", "Quantitative KG (PyVis)", "Full Network (NetworkX)", "Full Network (PyVis)", "Salience Network (NetworkX)", "Salience Network (PyVis)"])

                    with net_subtabs[0]:
                        if selected_qty != "All":
                            fig_kg = viz.plot_quantitative_knowledge_graph(df_all, selected_qty, colormap, aliases=aliases, label_style=label_style)
                            st.pyplot(fig_kg, key="py_fig_kg_2")

                            buf = BytesIO()
                            fig_kg.savefig(buf, format="png", dpi=config.figure_dpi)
                            st.download_button("Download KG as PNG", buf.getvalue(), f"{selected_qty}_kg.png", mime="image/png", key="dl_Download_KG_as_PNG")

                        else:
                            st.info("Select a specific quantity to see its knowledge graph.")

                    with net_subtabs[1]:
                        if PYVIS_AVAILABLE and selected_qty != "All":
                            html_kg = viz.plot_quantitative_knowledge_graph_pyvis(df_all, selected_qty, colormap, aliases=aliases, label_style=label_style)
                            st.components.v1.html(html_kg, height=750, scrolling=True, key="html_html_kg")

                            st.download_button("Download PyVis KG HTML", html_kg.encode('utf-8'), f"{selected_qty}_kg_pyvis.html", mime="text/html", key="dl_Download_PyVis_KG_HTML")

                        else:
                            st.info("Select a specific quantity and install pyvis for interactive graph.")

                    with net_subtabs[2]:
                        fig_net = viz.plot_knowledge_network(df_all, colormap, aliases=aliases, label_style=label_style)
                        st.pyplot(fig_net, key="py_fig_net")

                        buf = BytesIO()
                        fig_net.savefig(buf, format="png", dpi=config.figure_dpi)
                        st.download_button("Download Network PNG", buf.getvalue(), "knowledge_network.png", mime="image/png", key="dl_Download_Network_PNG")


                    with net_subtabs[3]:
                        if PYVIS_AVAILABLE:
                            html_full = viz.plot_knowledge_network_pyvis(df_all, colormap, aliases=aliases, label_style=label_style)
                            st.components.v1.html(html_full, height=750, scrolling=True, key="html_html_full")

                            st.download_button("Download PyVis Network HTML", html_full.encode('utf-8'), "knowledge_network_pyvis.html", mime="text/html", key="dl_Download_PyVis_Network_HTML")

                        else:
                            st.info("Install pyvis for interactive network: pip install pyvis")

                    with net_subtabs[4]:
                        fig_static = viz.plot_static_knowledge_network(None, st.session_state.get("viz_top_n", 25), colormap=colormap, aliases=aliases, label_style=label_style)
                        st.pyplot(fig_static, key="py_fig_static")

                        buf = BytesIO()
                        fig_static.savefig(buf, format="png", dpi=config.figure_dpi)
                        st.download_button("Download Salience Network PNG", buf.getvalue(), "salience_network.png", mime="image/png", key="dl_Download_Salience_Network_PNG")


                    with net_subtabs[5]:
                        if PYVIS_AVAILABLE:
                            html_salience = viz.render_pyvis_salience(None, st.session_state.get("viz_top_n", 25), True, colormap, aliases=aliases, label_style=label_style)
                            st.components.v1.html(html_salience, height=750, scrolling=True, key="html_html_salience")

                            st.download_button("Download PyVis Salience HTML", html_salience.encode('utf-8'), "salience_network_pyvis.html", mime="text/html", key="dl_Download_PyVis_Salience_HTML")

                        else:
                            st.info("Install pyvis for interactive network: pip install pyvis")

                with tabs[6]:
                    if st.session_state.embedding_model is not None:
                        emb_fn = lambda x: np.array(st.session_state.embedding_model.encode(x))
                        if SKLEARN_AVAILABLE:
                            fig_tsne = viz.plot_tsne(emb_fn, None if selected_qty=="All" else selected_qty, colormap, figsize=config.figsize_embedding)
                            if fig_tsne:
                                st.pyplot(fig_tsne, key="py_fig_tsne")

                                buf = BytesIO()
                                fig_tsne.savefig(buf, format="png", dpi=config.figure_dpi)
                                st.download_button("Download t-SNE PNG", buf.getvalue(), "tsne.png", mime="image/png", key="dl_Download_t-SNE_PNG")

                            fig_pca = viz.plot_pca(emb_fn, None if selected_qty=="All" else selected_qty, colormap, figsize=config.figsize_embedding)
                            if fig_pca:
                                st.pyplot(fig_pca, key="py_fig_pca")

                                buf = BytesIO()
                                fig_pca.savefig(buf, format="png", dpi=config.figure_dpi)
                                st.download_button("Download PCA PNG", buf.getvalue(), "pca.png", mime="image/png", key="dl_Download_PCA_PNG")

                        if UMAP_AVAILABLE:
                            fig_umap = viz.plot_umap(emb_fn, None if selected_qty=="All" else selected_qty, colormap, figsize=config.figsize_embedding)
                            if fig_umap:
                                st.pyplot(fig_umap, key="py_fig_umap")

                                buf = BytesIO()
                                fig_umap.savefig(buf, format="png", dpi=config.figure_dpi)
                                st.download_button("Download UMAP PNG", buf.getvalue(), "umap.png", mime="image/png", key="dl_Download_UMAP_PNG")

                    else:
                        st.warning("Install sentence-transformers and re-index to enable t-SNE/PCA/UMAP.")

                with tabs[7]:
                    fig_scatter = viz.plot_scatter_power_vs_speed(df_all, colormap)
                    st.plotly_chart(fig_scatter, use_container_width=True, key="plc_fig_scatter")

                    fig_parallel = viz.plot_parallel_categories(df_all, colormap)
                    st.plotly_chart(fig_parallel, use_container_width=True, key="plc_fig_parallel")

                    fig_violin = viz.plot_violin(df_all, colormap)
                    st.plotly_chart(fig_violin, use_container_width=True, key="plc_fig_violin")

                    fig_timeline = viz.plot_timeline(colormap)
                    st.plotly_chart(fig_timeline, use_container_width=True, key="plc_fig_timeline")


                with tabs[8]:
                    st.markdown("### Interactive Knowledge Graph Explorer")
                    entities = st.session_state.knowledge_graph.get_all_entity_names()
                    if entities:
                        selected_entity = st.selectbox("Choose entity", entities, key="kg_entity_select")
                        if selected_entity:
                            consensus = st.session_state.knowledge_graph.get_entity_consensus(selected_entity)
                            if consensus["found"]:
                                st.markdown(f"#### Consensus for **{selected_entity}**")
                                col1, col2, col3, col4, col5 = st.columns(5)
                                col1.metric("Count", consensus["count"])
                                col2.metric("Mean", f"{consensus['mean']:.2f} {consensus['unit']}")
                                col3.metric("Std Dev", f"{consensus['std']:.2f}")
                                col4.metric("Min", f"{consensus['range'][0]:.2f}")
                                col5.metric("Max", f"{consensus['range'][1]:.2f}")
                                st.markdown(f"**Documents:** {', '.join(consensus['documents'])}")
                            else:
                                st.info(f"No quantitative values found for '{selected_entity}'.")
                            contradictions = st.session_state.knowledge_graph.get_entity_contradictions(selected_entity, threshold_factor=1.5)
                            if contradictions:
                                st.markdown("#### Detected Contradictions")
                                for c in contradictions:
                                    st.warning(f"**{c['entity']}**: {c['doc_a']} ({c['value_a']:.2f}) vs {c['doc_b']} ({c['value_b']:.2f}) - ratio {c['ratio']:.1f}x ({c['severity']})")
                            else:
                                st.success("No significant contradictions detected for this entity.")
                            items_for_entity = []
                            for doc_id, graph in st.session_state.knowledge_graph.doc_graphs.items():
                                for item in graph["all_items"]:
                                    if (item.get("material") == selected_entity or item.get("physical_quantity") == selected_entity or item.get("method") == selected_entity or item.get("parameter_name") == selected_entity):
                                        items_for_entity.append(item)
                            if items_for_entity:
                                df_entity = pd.DataFrame([{"Doc": i["doc_source"], "Page": i.get("page",0), "Type": i.get("item_type",""), "Content": i.get("content","")[:150], "Value": i.get("value",""), "Unit": i.get("unit",""), "Confidence": i.get("confidence",0)} for i in items_for_entity])
                                st.dataframe(df_entity, use_container_width=True, key="df_df_entity")

                            else:
                                st.info("No extracted items found for this entity.")
                    else:
                        st.info("No entities extracted yet. Run a query or re-index.")

                with tabs[9]:
                    st.markdown("### Retrieval Diagnostics & Provenance")
                    cached = st.session_state.get("cached_query_result", {})
                    rel_docs = cached.get("relevant_docs", [])
                    retrieved_nodes = cached.get("retrieved", [])
                    raw_items = cached.get("items", [])

                    st.markdown("#### Retrieval Hyperparameters")
                    col_w, col_c, col_r, col_conf = st.columns(4)
                    with col_w:
                        window_size = st.slider("Page window", 1, 20, 7, key="window_size")
                    with col_c:
                        max_chars_viz = st.slider("Max chars", 5000, 50000, 20000, 5000, key="max_chars_slider")
                    with col_r:
                        max_results_viz = st.slider("Max results", 5, 100, 30, 5, key="max_results_slider")
                    with col_conf:
                        conf_thresh_viz = st.slider("Conf threshold", 0.3, 0.9, 0.55, 0.05, key="conf_thresh_slider")
                    st.caption(f"Current config: window={window_size}, max_chars={max_chars_viz}, max_results={max_results_viz}, conf>={conf_thresh_viz}")

                    st.markdown("#### Retrieval Provenance Flow")
                    fig_sankey = viz.plot_retrieval_sankey(active_prompt, rel_docs, retrieved_nodes, raw_items)
                    st.plotly_chart(fig_sankey, use_container_width=True, key="plc_fig_sankey_2")


                    st.markdown("#### Document Filter Scores")
                    fig_doc_scores = viz.plot_doc_filter_scores(rel_docs, len(st.session_state.annotated_trees))
                    st.plotly_chart(fig_doc_scores, use_container_width=True, key="plc_fig_doc_scores")


                    st.markdown("#### Page Coverage Heatmap")
                    fig_coverage = viz.plot_page_coverage_heatmap(st.session_state.annotated_trees, retrieved_nodes)
                    st.plotly_chart(fig_coverage, use_container_width=True, key="plc_fig_coverage")


                    st.markdown("#### Node Selection Confidence")
                    fig_conf = viz.plot_node_confidence_distribution(retrieved_nodes)
                    st.plotly_chart(fig_conf, use_container_width=True, key="plc_fig_conf")


                    st.markdown("#### Hierarchical Tree Explorer")
                    tree_doc_options = sorted(list(set(t.get("doc_id", t.get("doc_name", "unknown")) for t in st.session_state.annotated_trees)))
                    if tree_doc_options:
                        selected_tree_doc = st.selectbox("Select document to visualize", tree_doc_options, key="tree_doc_select")
                        fig_tree = viz.plot_retrieval_tree_highlight(st.session_state.annotated_trees, retrieved_nodes, selected_tree_doc)
                        if fig_tree:
                            st.pyplot(fig_tree, key="py_fig_tree")

                            buf = BytesIO()
                            fig_tree.savefig(buf, format="png", dpi=config.figure_dpi)
                            st.download_button("Download Tree PNG", buf.getvalue(), f"{selected_tree_doc}_tree.png", mime="image/png", key="dl_Download_Tree_PNG")

                        else:
                            st.info("No tree data available for this document.")
                    else:
                        st.info("No tree data available.")

                    if st.session_state.embedding_model is not None:
                        st.markdown("#### Semantic vs Vectorless Score Comparison")
                        emb_fn = lambda x: np.array(st.session_state.embedding_model.encode(x))
                        fig_comp = viz.plot_semantic_vs_vectorless(active_prompt, rel_docs, st.session_state.annotated_trees, emb_fn)
                        if fig_comp:
                            st.plotly_chart(fig_comp, use_container_width=True, key="plc_fig_comp")

                        else:
                            st.info("Could not compute semantic scores for comparison.")

                    st.markdown("#### Raw Retrieval Metadata")
                    if retrieved_nodes:
                        df_ret = pd.DataFrame([{"Document": r.get("doc_id", ""), "Node ID": r.get("node_id", ""), "Section": r.get("section_title", ""), "Page": r.get("page_start", 0), "Confidence": r.get("confidence", 0), "Reasoning": r.get("selection_reasoning", "")[:100]} for r in retrieved_nodes])
                        st.dataframe(df_ret, use_container_width=True, key="df_df_ret")

                        csv_ret = df_ret.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Retrieval Metadata CSV", csv_ret, "retrieval_metadata.csv", mime="text/csv", key="dl_Download_Retrieval_Metadata_CS")

                    else:
                        st.info("No retrieved node metadata available.")
            else:
                st.info("No quantitative data extracted yet. Run a query to populate the knowledge graph.")

        if st.session_state.get("show_tree_nav") and retrieved:
            with st.expander("Tree Navigation Trace", expanded=False):
                for r in retrieved[:5]:
                    st.markdown(f"**{r['doc_id']}** -> `{r['section_title']}` (p.{r['page_start']}) | confidence: {r.get('confidence', 0):.2f}")
                    st.caption(r.get('selection_reasoning', ''))
        if items:
            with st.expander("Extracted Items (Raw)", expanded=False):
                st.json([i.to_dict() for i in items[:10]])

        report = CrossDocumentQueryReport(query=active_prompt, total_documents=len(st.session_state.annotated_trees), documents_with_results=len(set(i.doc_source for i in items)), all_items=[i.model_dump() for i in items])
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button("Download JSON Report", report.to_json(), "results.json", "application/json", key="dl_Download_JSON_Report")

        with col_dl2:
            tree_export = {"query": active_prompt, "annotated_trees": st.session_state.annotated_trees, "retrieved_nodes": retrieved, "extracted_items": [i.to_dict() for i in items], "answer": answer}
            st.download_button("Download Tree Export", json.dumps(tree_export, indent=2, ensure_ascii=False, default=str), "tree_report.json", "application/json", key="dl_Download_Tree_Export")


        if "index" in st.session_state.query_processor:
            st.session_state.query_processor["index"].cleanup()
    else:
        st.info("Upload PDF files to begin.")


def fast_json_dumps(obj, indent=False):
    if ORJSON_AVAILABLE:
        option = orjson.OPT_INDENT_2 if indent else 0
        return orjson.dumps(obj, option=option, default=str)
    else:
        return json.dumps(obj, indent=2 if indent else None, ensure_ascii=False, default=str).encode()

def fast_json_loads(data):
    if ORJSON_AVAILABLE:
        if isinstance(data, str):
            data = data.encode()
        return orjson.loads(data)
    else:
        if isinstance(data, bytes):
            data = data.decode()
        return json.loads(data)

@contextmanager
def timer(label: str):
    start = time.time()
    yield
    elapsed = time.time() - start
    if not hasattr(timer, 'metrics'):
        timer.metrics = defaultdict(list)
    timer.metrics[label].append(elapsed)
    logger.info(f"{label}: {elapsed:.2f}s")

class LRUCache:
    def __init__(self, max_size=1000, ttl=7200):
        self.max_size = max_size
        self.ttl = ttl
        self._cache = OrderedDict()
        self._lock = threading.RLock()
    def _key(self, *args, **kwargs):
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

response_cache = LRUCache(max_size=2000, ttl=7200)

if __name__ == "__main__":
    run_streamlit()
