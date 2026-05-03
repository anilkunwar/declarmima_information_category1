
# =====================================================================
# QUERY-BIASED QUANTITATIVE EXTRACTION THEORY (QBQE)
# =====================================================================
# Mathematical framework for enhanced quantitative information extraction
# from scientific documents with query-biased salience scoring.
#
# This module replaces the rigid regex-based extraction with a soft,
# context-aware, and query-biased approach that finds ALL relevant values
# across documents, not just exact keyword matches.
# =====================================================================

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
import re
from collections import defaultdict

# =====================================================================
# ENHANCED PATTERN DEFINITIONS
# =====================================================================

ENHANCED_QUANTITY_PATTERNS = {
    "laser_power": {
        "patterns": [
            r'(\d+(?:\.\d+)?)\s*(?:W|watts?)\s*(?:laser\s*)?(?:power|output|source)',
            r'(\d+(?:\.\d+)?)\s*(?:W|watts?)\s*(?:continuous|cw|pulsed|average|peak)',
            r'(\d+(?:\.\d+)?)\s*(?:W|watts?)\s*(?:at\s*(?:the\s*)?laser)',
            r'(?:power|output)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(?:W|watts?)',
            r'(?:laser|beam)\s*(?:power|output)\s*(?:of\s*|[:=]\s*)(\d+(?:\.\d+)?)\s*(?:W|watts?)',
            r'P\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:W|watts?)',
            r'(?:operating|working|applied)\s*(?:at|with|under)\s*(\d+(?:\.\d+)?)\s*(?:W|watts?)',
            r'(\d+(?:\.\d+)?)\s*(?:W|watts?)\s*(?:laser|beam|fiber|diode|CO2|Nd:YAG|fiber)',
            r'(?:using|with|at)\s+a\s*(\d+(?:\.\d+)?)\s*(?:W|watts?)\s*(?:laser|source)',
            r'(?:Laser\s+power|Power)\s*[:\|]\s*(\d+(?:\.\d+)?)\s*(?:W|watts?)',
        ],
        "context_keywords": ["laser", "beam", "processing", "ablation", "melting", 
                            "sintering", "welding", "cutting", "drilling", "surface"],
        "unit": "W",
        "typical_range": (1, 10000),
        "synonyms": ["power", "output power", "laser output", "beam power", 
                    "incident power", "applied power", "nominal power"]
    },

    "scan_speed": {
        "patterns": [
            r'(\d+(?:\.\d+)?)\s*(?:mm/s|mm/min|m/s)\s*(?:scan\s*speed|scanning\s*speed|travel\s*speed|feed\s*rate)',
            r'(?:scan\s*speed|scanning\s*speed)\s*(?:of\s*|[:=]\s*)(\d+(?:\.\d+)?)\s*(?:mm/s|mm/min|m/s)',
            r'v\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:mm/s|mm/min|m/s)',
            r'(\d+(?:\.\d+)?)\s*(?:mm/s|mm/min|m/s)\s*(?:scan|scanning|hatch|raster)',
        ],
        "context_keywords": ["scan", "scanning", "hatch", "raster", "speed", "velocity", 
                            "feed", "traversal", "beam path"],
        "unit": "mm/s",
        "typical_range": (0.1, 5000),
        "synonyms": ["scan speed", "scanning speed", "travel speed", "feed rate", 
                    "scanning velocity", "hatch speed"]
    },

    "fluence": {
        "patterns": [
            r'(\d+(?:\.\d+)?)\s*(?:J/cm²|J/cm2|J\s*cm[-²2])\s*(?:fluence|energy\s*density|threshold)',
            r'(?:fluence|energy\s*density)\s*(?:of\s*|[:=]\s*)(\d+(?:\.\d+)?)\s*(?:J/cm²|J/cm2)',
            r'F\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:J/cm²|J/cm2)',
            r'(\d+(?:\.\d+)?)\s*(?:J/cm²|J/cm2)\s*(?:pulse|laser|beam)',
        ],
        "context_keywords": ["fluence", "threshold", "ablation", "energy density", 
                            "laser", "pulse", "damage"],
        "unit": "J/cm²",
        "typical_range": (0.01, 100),
        "synonyms": ["fluence", "energy density", "laser fluence", "pulse fluence"]
    },

    "wavelength": {
        "patterns": [
            r'(\d+(?:\.\d+)?)\s*(?:nm|nanometers?)\s*(?:wavelength|λ|lambda|emission)',
            r'(?:wavelength|λ|lambda)\s*(?:of\s*|[:=]\s*)(\d+(?:\.\d+)?)\s*(?:nm|nanometers?)',
            r'λ\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:nm|nanometers?)',
            r'(\d+(?:\.\d+)?)\s*(?:nm|nanometers?)\s*(?:laser|beam|source|fiber|diode)',
        ],
        "context_keywords": ["wavelength", "laser", "emission", "spectral", "IR", 
                            "UV", "visible", "infrared", "ultraviolet"],
        "unit": "nm",
        "typical_range": (100, 11000),
        "synonyms": ["wavelength", "emission wavelength", "laser wavelength", "operating wavelength"]
    },

    "pulse_duration": {
        "patterns": [
            r'(\d+(?:\.\d+)?)\s*(?:fs|femtoseconds?|ps|picoseconds?|ns|nanoseconds?|μs|microseconds?|ms|milliseconds?)\s*(?:pulse|duration|width|length|fwhm)',
            r'(?:pulse\s*duration|pulse\s*width|fwhm)\s*(?:of\s*|[:=]\s*)(\d+(?:\.\d+)?)\s*(?:fs|ps|ns|μs|ms)',
            r'τ\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:fs|ps|ns|μs|ms)',
        ],
        "context_keywords": ["pulse", "duration", "width", "fwhm", "temporal", 
                            "femtosecond", "picosecond", "nanosecond"],
        "unit": "fs",
        "typical_range": (1, 1e9),
        "synonyms": ["pulse duration", "pulse width", "pulse length", "fwhm", "temporal width"]
    },

    "repetition_rate": {
        "patterns": [
            r'(\d+(?:\.\d+)?)\s*(?:kHz|MHz|Hz)\s*(?:repetition|rep\s*rate|frequency|rate|pulse\s*rate)',
            r'(?:repetition\s*rate|rep\s*rate|frequency)\s*(?:of\s*|[:=]\s*)(\d+(?:\.\d+)?)\s*(?:kHz|MHz|Hz)',
            r'f\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:kHz|MHz|Hz)',
            r'(\d+(?:\.\d+)?)\s*(?:kHz|MHz|Hz)\s*(?:laser|pulse|beam)',
        ],
        "context_keywords": ["repetition", "frequency", "rate", "pulse rate", "kHz", "MHz"],
        "unit": "kHz",
        "typical_range": (1, 1e6),
        "synonyms": ["repetition rate", "pulse repetition rate", "rep rate", "frequency", "pulse frequency"]
    },

    "spot_size": {
        "patterns": [
            r'(\d+(?:\.\d+)?)\s*(?:µm|um|microns?|mm|nm)\s*(?:spot|beam\s*radius|waist|diameter|focus|focal\s*spot)',
            r'(?:spot\s*size|beam\s*radius|waist|diameter)\s*(?:of\s*|[:=]\s*)(\d+(?:\.\d+)?)\s*(?:µm|um|mm|nm)',
            r'w₀\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:µm|um|mm|nm)',
            r'(\d+(?:\.\d+)?)\s*(?:µm|um|mm|nm)\s*(?:focal\s*spot|beam\s*waist)',
        ],
        "context_keywords": ["spot", "beam", "waist", "focus", "focal", "diameter", "radius"],
        "unit": "µm",
        "typical_range": (0.1, 1000),
        "synonyms": ["spot size", "beam radius", "beam waist", "focal spot", "spot diameter"]
    },

    "hatch_distance": {
        "patterns": [
            r'(\d+(?:\.\d+)?)\s*(?:µm|um|mm)\s*(?:hatch\s*distance|hatch\s*spacing|line\s*spacing)',
            r'(?:hatch\s*distance|hatch\s*spacing)\s*(?:of\s*|[:=]\s*)(\d+(?:\.\d+)?)\s*(?:µm|um|mm)',
            r'h\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:µm|um|mm)',
        ],
        "context_keywords": ["hatch", "spacing", "distance", "overlap", "scan strategy"],
        "unit": "µm",
        "typical_range": (10, 500),
        "synonyms": ["hatch distance", "hatch spacing", "line spacing", "scan spacing"]
    },

    "layer_thickness": {
        "patterns": [
            r'(\d+(?:\.\d+)?)\s*(?:µm|um|mm|nm)\s*(?:layer\s*thickness|layer\s*height|slice\s*thickness)',
            r'(?:layer\s*thickness|layer\s*height)\s*(?:of\s*|[:=]\s*)(\d+(?:\.\d+)?)\s*(?:µm|um|mm|nm)',
            r't\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:µm|um|mm|nm)\s*(?:layer|slice)',
        ],
        "context_keywords": ["layer", "thickness", "height", "slice", "build", "deposition"],
        "unit": "µm",
        "typical_range": (10, 200),
        "synonyms": ["layer thickness", "layer height", "slice thickness", "build thickness"]
    },

    "pulse_energy": {
        "patterns": [
            r'(\d+(?:\.\d+)?)\s*(?:µJ|uJ|mJ|nJ|J)\s*(?:pulse\s*energy|energy\s*per\s*pulse|single\s*pulse)',
            r'(?:pulse\s*energy|energy\s*per\s*pulse)\s*(?:of\s*|[:=]\s*)(\d+(?:\.\d+)?)\s*(?:µJ|mJ|nJ|J)',
            r'E_p\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:µJ|mJ|nJ|J)',
        ],
        "context_keywords": ["pulse", "energy", "per pulse", "single pulse", "pulse energy"],
        "unit": "mJ",
        "typical_range": (0.001, 100),
        "synonyms": ["pulse energy", "energy per pulse", "single pulse energy", "pulse fluence"]
    },

    "roughness": {
        "patterns": [
            r'(\d+(?:\.\d+)?)\s*(?:nm|µm|um)\s*(?:roughness|Ra|RMS|Rq|surface\s*finish)',
            r'(?:roughness|Ra|RMS)\s*(?:of\s*|[:=]\s*)(\d+(?:\.\d+)?)\s*(?:nm|µm|um)',
            r'Ra\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:nm|µm|um)',
        ],
        "context_keywords": ["roughness", "surface", "finish", "Ra", "RMS", "quality", "smoothness"],
        "unit": "nm",
        "typical_range": (0.1, 100),
        "synonyms": ["roughness", "surface roughness", "Ra", "RMS roughness", "surface finish"]
    },

    "porosity": {
        "patterns": [
            r'(\d+(?:\.\d+)?)\s*(?:\%|percent)\s*(?:porosity|pore\s*fraction|void\s*fraction)',
            r'(?:porosity|pore\s*fraction)\s*(?:of\s*|[:=]\s*)(\d+(?:\.\d+)?)\s*(?:\%|percent)',
            r'(\d+(?:\.\d+)?)\s*(?:\%|percent)\s*(?:porous|voids|pores)',
        ],
        "context_keywords": ["porosity", "pore", "void", "defect", "density", "fraction"],
        "unit": "%",
        "typical_range": (0, 50),
        "synonyms": ["porosity", "pore fraction", "void fraction", "porous fraction"]
    },
}

# =====================================================================
# QUERY-BIASED QUANTITATIVE EXTRACTOR CLASS
# =====================================================================

class QueryBiasedQuantitativeExtractor:
    """
    Extracts quantitative values with query-biased salience scoring.

    Mathematical Model:
    -------------------
    For each candidate value v at position p in document d:

    Φ(v,p,d) = [v_norm; E(C(p)); S(C(p)); D(d)]  (Contextual embedding)

    σ_q(v,p) = α·cos_sim(e_q, E(C(p)))            (Query similarity)
              + β·keyword_overlap(q, C(p))         (Keyword overlap)
              + γ·entity_proximity(q_entities, p)  (Entity proximity)
              + δ·section_relevance(q, S(C(p)))    (Section relevance)

    conf_P(v,p) = max_pattern [λ_pattern · match_strength(pattern, C(p))]

    consensus(v) = 1 + η·log(1+|D_v|)·(1 - std(v)/mean(v))

    Salience(v,p|q) = σ_q(v,p) · conf_P(v,p) · consensus(v) · exp(-λ·dist_to_entities)

    Values with Salience > threshold are extracted and grouped by material/method.
    """

    def __init__(self, embed_model=None):
        self.patterns = ENHANCED_QUANTITY_PATTERNS
        self.embed_model = embed_model
        self._value_cache = {}

    def extract_all_values(self, chunks: List, query: str = "", 
                          min_confidence: float = 0.3) -> pd.DataFrame:
        """
        Main extraction pipeline with query biasing.

        Args:
            chunks: List of Document objects
            query: User query for biasing
            min_confidence: Minimum pattern confidence threshold

        Returns:
            DataFrame with all extracted values and salience scores
        """
        records = []
        query_embedding = self._get_query_embedding(query) if query else None
        query_entities = self._extract_query_entities(query)

        for chunk in chunks:
            text = chunk.page_content
            doc_source = chunk.metadata.get("source", "unknown")
            section = chunk.metadata.get("section", "BODY")

            # Step 1: Find all candidate numeric values with units
            candidates = self._find_numeric_candidates(text)

            for candidate in candidates:
                value = candidate['value']
                unit = candidate['unit']
                position = candidate['position']
                context_window = self._get_context_window(text, position, window=200)

                # Step 2: Determine quantity type with confidence
                qty_type, confidence = self._classify_quantity(
                    value, unit, context_window, text, position
                )

                if confidence < min_confidence:
                    continue

                # Step 3: Compute query-biased salience
                salience = self._compute_salience(
                    value, unit, qty_type, context_window, 
                    query, query_embedding, query_entities, 
                    confidence, section
                )

                # Step 4: Extract associated entities (material, method, etc.)
                associated = self._extract_associated_entities(context_window)

                records.append({
                    'value': value,
                    'unit': unit,
                    'quantity_type': qty_type,
                    'confidence': confidence,
                    'salience': salience,
                    'doc_source': doc_source,
                    'section': section,
                    'context': context_window[:300],
                    'material': associated.get('material', 'Unknown'),
                    'method': associated.get('method', 'Unknown'),
                    'position': position,
                    'raw_text': candidate['raw']
                })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values('salience', ascending=False)
        return df

    def _find_numeric_candidates(self, text: str) -> List[Dict]:
        """Find all numeric values with units in text."""
        candidates = []

        # Comprehensive pattern for numbers with scientific notation and units
        number_unit_pattern = re.compile(
            r'(\d+(?:\.\d+)?(?:\s*[×x]\s*10\^?\(?[-+]?\d+\)?)?)'  # Number with optional sci notation
            r'\s*'
            r'(W|watts?|mW|kW|MW|'  # Power units
            r'mm/s|mm/min|m/s|'     # Speed units
            r'J/cm²|J/cm2|J\s*cm[-²2]|mJ/cm²|µJ/cm²|'  # Fluence units
            r'nm|µm|um|microns?|mm|cm|'  # Length units
            r'fs|ps|ns|μs|microseconds?|ms|'  # Time units
            r'kHz|MHz|Hz|'  # Frequency units
            r'µJ|uJ|mJ|nJ|J|'  # Energy units
            r'\%|percent|'  # Percentage
            r'K|°C|°F|'  # Temperature
            r'GPa|MPa|Pa|'  # Pressure
            r'g/cm³|kg/m³)',  # Density
            re.IGNORECASE
        )

        for match in number_unit_pattern.finditer(text):
            value_str = match.group(1)
            unit_str = match.group(2)

            # Parse value (handle scientific notation)
            try:
                if '×' in value_str or 'x' in value_str.lower() or '10^' in value_str:
                    value_str = value_str.replace('×', 'e').replace('x', 'e').replace('^', '')
                    value_str = re.sub(r'10e\(?([-+]?\d+)\)?', r'1e\1', value_str)
                value = float(value_str)
            except:
                continue

            candidates.append({
                'value': value,
                'unit': unit_str,
                'position': match.start(),
                'raw': match.group(0)
            })

        return candidates

    def _classify_quantity(self, value: float, unit: str, context: str, 
                          full_text: str, position: int) -> Tuple[str, float]:
        """
        Classify which quantity type this value represents.
        Uses multi-factor confidence scoring.
        """
        best_type = "unknown"
        best_conf = 0.0

        broader_context = full_text[max(0, position-300):min(len(full_text), position+300)]

        for qty_name, config in self.patterns.items():
            # Check 1: Unit match
            unit_conf = self._unit_match_confidence(unit, config['unit'])

            # Check 2: Pattern match in context
            pattern_conf = 0.0
            for pattern in config['patterns']:
                if re.search(pattern, broader_context, re.IGNORECASE):
                    pattern_conf = max(pattern_conf, 0.9)
                    break

            # Check 3: Keyword co-occurrence
            keyword_conf = self._keyword_cooccurrence_confidence(
                broader_context, config['context_keywords']
            )

            # Check 4: Value range plausibility
            range_conf = self._range_plausibility(value, config['typical_range'])

            # Combine confidences (weighted)
            conf = (0.35 * unit_conf + 
                   0.30 * pattern_conf + 
                   0.20 * keyword_conf + 
                   0.15 * range_conf)

            if conf > best_conf:
                best_conf = conf
                best_type = qty_name

        return best_type, best_conf

    def _unit_match_confidence(self, found_unit: str, expected_unit: str) -> float:
        """Score how well the found unit matches the expected unit."""
        found = found_unit.lower().strip()
        expected = expected_unit.lower().strip()

        if found == expected:
            return 1.0

        aliases = {
            'w': ['w', 'watt', 'watts'],
            'um': ['um', 'µm', 'micron', 'microns', 'micrometer'],
            'nm': ['nm', 'nanometer', 'nanometers'],
            'mm': ['mm', 'millimeter', 'millimeters'],
            'fs': ['fs', 'femtosecond', 'femtoseconds'],
            'ps': ['ps', 'picosecond', 'picoseconds'],
            'ns': ['ns', 'nanosecond', 'nanoseconds'],
            'khz': ['khz', 'kilohertz'],
            'mhz': ['mhz', 'megahertz'],
            'j/cm²': ['j/cm²', 'j/cm2', 'j/cm^2'],
            '%': ['%', 'percent', 'percentage']
        }

        for base, alias_list in aliases.items():
            if expected in alias_list or expected == base:
                if found in alias_list:
                    return 1.0

        # Same dimension but different scale
        power_units = ['w', 'mw', 'kw', 'mw']
        if expected in power_units and found in power_units:
            return 0.7

        length_units = ['nm', 'um', 'µm', 'mm', 'cm', 'm']
        if expected in length_units and found in length_units:
            return 0.7

        return 0.0

    def _keyword_cooccurrence_confidence(self, context: str, keywords: List[str]) -> float:
        """Score based on presence of domain keywords in context."""
        context_lower = context.lower()
        matches = sum(1 for kw in keywords if kw.lower() in context_lower)
        return min(matches / max(len(keywords) * 0.3, 1), 1.0)

    def _range_plausibility(self, value: float, typical_range: Tuple[float, float]) -> float:
        """Score based on whether value is in typical range."""
        min_val, max_val = typical_range
        if min_val <= value <= max_val:
            return 1.0
        mean = (min_val + max_val) / 2
        std = (max_val - min_val) / 2
        distance = abs(value - mean)
        return max(0, np.exp(-0.5 * (distance / std) ** 2))

    def _compute_salience(self, value: float, unit: str, qty_type: str,
                         context: str, query: str, query_emb, 
                         query_entities: List[str], base_conf: float,
                         section: str) -> float:
        """
        Compute query-biased salience score.

        σ_q(v,p) = α·cos_sim(e_q, E(C(p))) + β·keyword_overlap(q, C(p)) 
                  + γ·entity_proximity(q_entities, p) + δ·section_relevance(q, S(C(p)))
        """
        # Component 1: Query embedding similarity
        query_sim = 0.5
        if query_emb is not None and self.embed_model is not None:
            try:
                ctx_emb = self.embed_model.encode(context[:500])
                query_sim = float(np.dot(query_emb, ctx_emb) / 
                                 (np.linalg.norm(query_emb) * np.linalg.norm(ctx_emb) + 1e-8))
                query_sim = (query_sim + 1) / 2
            except:
                pass

        # Component 2: Keyword overlap
        keyword_score = self._keyword_overlap_score(query, context)

        # Component 3: Entity proximity
        entity_score = self._entity_proximity_score(query_entities, context)

        # Component 4: Section relevance
        section_weights = {
            'ABSTRACT': 0.9, 'INTRODUCTION': 0.7, 'METHODS': 0.8,
            'RESULTS': 1.0, 'DISCUSSION': 0.95, 'CONCLUSION': 0.85,
            'BODY': 0.6, 'UNKNOWN': 0.5
        }
        section_score = section_weights.get(section.upper(), 0.5)

        # Combine: α=0.4, β=0.25, γ=0.2, δ=0.15
        salience = (0.40 * query_sim + 
                   0.25 * keyword_score + 
                   0.20 * entity_score + 
                   0.15 * section_score)

        # Modulate by base confidence
        salience *= base_conf

        return salience

    def _keyword_overlap_score(self, query: str, context: str) -> float:
        """Jaccard-like overlap between query keywords and context."""
        if not query:
            return 0.5
        query_words = set(query.lower().split())
        context_words = set(context.lower().split())
        overlap = len(query_words & context_words)
        return min(overlap / max(len(query_words) * 0.5, 1), 1.0)

    def _entity_proximity_score(self, query_entities: List[str], context: str) -> float:
        """Score based on presence of query entities in context."""
        if not query_entities:
            return 0.5
        context_lower = context.lower()
        matches = sum(1 for ent in query_entities if ent.lower() in context_lower)
        return min(matches / max(len(query_entities) * 0.5, 1), 1.0)

    def _extract_associated_entities(self, context: str) -> Dict[str, str]:
        """Extract material and method entities from context."""
        associated = {}
        context_lower = context.lower()

        # Material detection
        for canonical, aliases in MATERIAL_ALIASES.items():
            if any(alias in context_lower for alias in aliases):
                associated['material'] = canonical
                break

        # Method detection
        for canonical, aliases in METHOD_ALIASES.items():
            if any(alias in context_lower for alias in aliases):
                associated['method'] = canonical
                break

        return associated

    def _get_query_embedding(self, query: str):
        """Get embedding for query."""
        if self.embed_model is None:
            return None
        try:
            return self.embed_model.encode(query)
        except:
            return None

    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract entities from query."""
        entities = []
        q_lower = query.lower()
        for canonical, aliases in {**MATERIAL_ALIASES, **METHOD_ALIASES}.items():
            if any(alias in q_lower for alias in aliases):
                entities.append(canonical)
        return entities

    def _get_context_window(self, text: str, position: int, window: int = 200) -> str:
        """Extract context window around position."""
        start = max(0, position - window)
        end = min(len(text), position + window)
        return text[start:end]

    def aggregate_by_material(self, df: pd.DataFrame, qty_type: str) -> pd.DataFrame:
        """Aggregate values by material with statistics."""
        if df.empty:
            return df

        df_filtered = df[df['quantity_type'] == qty_type]
        if df_filtered.empty:
            return df_filtered

        grouped = df_filtered.groupby('material').agg({
            'value': ['count', 'mean', 'std', 'min', 'max'],
            'salience': 'mean',
            'confidence': 'mean',
            'doc_source': lambda x: list(set(x))
        }).reset_index()

        grouped.columns = ['material', 'count', 'mean', 'std', 'min', 'max', 
                          'avg_salience', 'avg_confidence', 'sources']
        grouped['std'] = grouped['std'].fillna(0)
        grouped = grouped.sort_values('avg_salience', ascending=False)

        return grouped

    def find_value_clusters(self, df: pd.DataFrame, qty_type: str, 
                           eps: float = 0.1) -> pd.DataFrame:
        """Cluster similar values to find distinct operating regimes."""
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            return df[df['quantity_type'] == qty_type]

        df_filtered = df[df['quantity_type'] == qty_type].copy()
        if len(df_filtered) < 3:
            return df_filtered

        values = df_filtered['value'].values.reshape(-1, 1)
        v_min, v_max = values.min(), values.max()
        if v_max > v_min:
            normalized = (values - v_min) / (v_max - v_min)
        else:
            normalized = np.zeros_like(values)

        clustering = DBSCAN(eps=eps, min_samples=2).fit(normalized)
        df_filtered['cluster'] = clustering.labels_

        return df_filtered

    def compute_consensus_boost(self, df: pd.DataFrame, qty_type: str) -> Dict:
        """
        Compute cross-document consensus boosting.

        consensus_boost(v) = 1 + η·log(1+|D_v|)·(1 - std(v.values)/mean(v.values))
        """
        df_filtered = df[df['quantity_type'] == qty_type]
        if df_filtered.empty:
            return {}

        by_value = df_filtered.groupby('value').agg({
            'doc_source': lambda x: len(set(x)),
            'salience': 'mean'
        }).reset_index()

        boosts = {}
        eta = 0.3
        for _, row in by_value.iterrows():
            doc_count = row['doc_source']
            # Simplified: use doc count for boost
            boost = 1 + eta * np.log(1 + doc_count)
            boosts[row['value']] = boost

        return boosts


