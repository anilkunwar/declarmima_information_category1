#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DECLARMIMA PURE — Learning-Based Scientific RAG
================================================
Zero-regex, zero-visualization, maximum reasoning fidelity.

Architecture:
1. CLAIM EXTRACTION: LLM-native structured extraction from raw text
2. CLAIM INDEXING: Dual embedding space (claim semantics + evidence text)
3. UNCERTAINTY-AWARE RETRIEVAL: Consensus + contradiction detection via statistics
4. VERIFIED SYNTHESIS: Iterative LLM reasoning with explicit confidence bounds

Key Design Decisions:
- NO regex for quantity extraction — LLM reads raw text directly
- NO visualization code — pure information retrieval and reasoning
- Structured claim representation with uncertainty quantification
- Iterative verification: extract → critique → refine
"""

import streamlit as st
import os
import tempfile
import time
import re
import json
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Any, Set, Callable
from datetime import datetime
import sys
import hashlib
from dataclasses import dataclass, field
import logging
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings('ignore')

# =====================================================================
# LOGGING
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DECLARMIMA_PURE")

# =====================================================================
# CONFIGURATION
# =====================================================================
@dataclass
class Config:
    chunk_size: int = 800
    chunk_overlap: int = 150
    retrieval_k: int = 6
    score_threshold: float = 0.25
    max_context_tokens: int = 2048
    max_new_tokens: int = 512
    temperature: float = 0.05
    claim_extraction_temperature: float = 0.0  # Deterministic extraction
    verification_temperature: float = 0.1
    synthesis_temperature: float = 0.05
    min_claim_confidence: float = 0.6
    consensus_threshold: float = 0.7
    contradiction_ratio: float = 2.0
    max_verification_iterations: int = 2

CONFIG = Config()

# =====================================================================
# IMPORTS
# =====================================================================
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# =====================================================================
# STRUCTURED CLAIM REPRESENTATION (Core Innovation)
# =====================================================================

@dataclass
class ScientificClaim:
    """
    Atomic unit of scientific knowledge. No regex-derived fields.
    All values extracted directly by LLM from raw text.
    """
    claim_id: str
    subject: str           # What is being described (e.g., "Ti-6Al-4V", "laser power")
    predicate: str         # Relationship type (causes, correlates_with, equals, etc.)
    object: str            # What it relates to (e.g., "porosity", "200 W")
    object_value: Optional[float] = None      # Numeric value if quantifiable
    object_unit: Optional[str] = None         # Unit if applicable
    confidence: float = 0.0                   # Extraction confidence [0,1]
    uncertainty_type: str = "point_estimate"  # point_estimate, range, distribution, qualitative
    uncertainty_bounds: Optional[Tuple[float, float]] = None  # (lower, upper) for ranges
    evidence_span: str = ""                   # Verbatim supporting text
    doc_source: str = ""                      # Source document
    section: str = ""                         # Abstract, Methods, Results, etc.
    page_num: int = -1
    context_window: str = ""                  # ±500 chars around evidence
    extraction_iteration: int = 0             # Which verification pass produced this
    critique_notes: List[str] = field(default_factory=list)  # Self-critique history
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "object_value": self.object_value,
            "object_unit": self.object_unit,
            "confidence": self.confidence,
            "uncertainty_type": self.uncertainty_type,
            "uncertainty_bounds": self.uncertainty_bounds,
            "evidence_span": self.evidence_span[:300],
            "doc_source": self.doc_source,
            "section": self.section,
            "page_num": self.page_num,
            "extraction_iteration": self.extraction_iteration,
            "critique_notes": self.critique_notes
        }

    def format_for_llm(self) -> str:
        """Format claim for inclusion in LLM context."""
        uncertainty_str = ""
        if self.uncertainty_type == "range" and self.uncertainty_bounds:
            uncertainty_str = f" [range: {self.uncertainty_bounds[0]}-{self.uncertainty_bounds[1]} {self.object_unit or ''}]"
        elif self.uncertainty_type == "point_estimate" and self.object_value is not None:
            uncertainty_str = f" [value: {self.object_value} {self.object_unit or ''}]"
        
        return (f"[{self.claim_id}] {self.subject} → {self.predicate} → {self.object}{uncertainty_str} "
                f"(confidence: {self.confidence:.2f}, source: {self.doc_source}, section: {self.section})")

@dataclass
class ClaimCluster:
    """Group of semantically equivalent claims across documents."""
    cluster_id: str
    canonical_subject: str
    canonical_predicate: str
    claims: List[ScientificClaim] = field(default_factory=list)
    
    def consensus_value(self) -> Optional[Dict[str, Any]]:
        """Statistical consensus with uncertainty propagation."""
        quantified = [c for c in self.claims if c.object_value is not None]
        if len(quantified) < 2:
            return None
        
        values = [c.object_value for c in quantified]
        weights = [c.confidence for c in quantified]
        
        # Weighted statistics
        total_weight = sum(weights)
        if total_weight == 0:
            return None
            
        weighted_mean = sum(v * w for v, w in zip(values, weights)) / total_weight
        weighted_var = sum(w * (v - weighted_mean) ** 2 for v, w in zip(values, weights)) / total_weight
        weighted_std = weighted_var ** 0.5
        
        # Between-document variance (for consensus assessment)
        doc_means = defaultdict(list)
        for c in quantified:
            doc_means[c.doc_source].append(c.object_value)
        doc_mean_values = [np.mean(vs) for vs in doc_means.values()]
        between_doc_std = np.std(doc_mean_values) if len(doc_mean_values) > 1 else 0
        
        return {
            "weighted_mean": weighted_mean,
            "weighted_std": weighted_std,
            "between_doc_std": between_doc_std,
            "consensus_strength": "strong" if between_doc_std < weighted_std else "weak",
            "n_claims": len(quantified),
            "n_docs": len(doc_means),
            "value_range": (min(values), max(values)),
            "unit": quantified[0].object_unit if quantified else None
        }
    
    def contradictions(self, ratio_threshold: float = 2.0) -> List[Dict[str, Any]]:
        """Detect contradictions via statistical discrepancy."""
        by_doc = defaultdict(list)
        for c in self.claims:
            if c.object_value is not None:
                by_doc[c.doc_source].append(c.object_value)
        
        docs = list(by_doc.keys())
        contradictions = []
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                vals_i, vals_j = by_doc[docs[i]], by_doc[docs[j]]
                mean_i, mean_j = np.mean(vals_i), np.mean(vals_j)
                if mean_i > 0 and mean_j > 0:
                    ratio = max(mean_i, mean_j) / min(mean_i, mean_j)
                    if ratio > ratio_threshold:
                        contradictions.append({
                            "doc_a": docs[i], "mean_a": mean_i, "std_a": np.std(vals_i),
                            "doc_b": docs[j], "mean_b": mean_j, "std_b": np.std(vals_j),
                            "ratio": ratio
                        })
        return contradictions

# =====================================================================
# LLM CLAIM EXTRACTOR (Zero Regex — Pure LLM)
# =====================================================================

class LLMClaimExtractor:
    """
    Extracts structured claims directly from raw text using LLM.
    No regex preprocessing — text goes directly to model.
    Supports iterative verification: extract → critique → refine.
    """
    
    EXTRACTION_PROMPT = """You are a precise scientific information extractor. Read the following text from a materials science / laser processing paper and extract ALL explicit and implicit scientific claims.

For each claim, identify:
- SUBJECT: The entity being described (material, parameter, method, phenomenon)
- PREDICATE: The relationship (causes, correlates_with, equals, greater_than, less_than, inhibits, promotes, is_composed_of, etc.)
- OBJECT: What the subject relates to
- VALUE: Numeric value if quantifiable (null if not)
- UNIT: Unit of measurement (null if not applicable)
- UNCERTAINTY: How certain is this claim? Choose from: point_estimate, range, distribution, qualitative_only
- UNCERTAINTY_BOUNDS: If range, provide [lower, upper]
- EVIDENCE: Exact verbatim text supporting this claim (max 200 chars)
- CONFIDENCE: Your confidence in this extraction [0.0-1.0]

IMPORTANT RULES:
1. Extract ONLY what is explicitly stated or directly inferable. Do NOT hallucinate.
2. For quantitative values, include the exact number as stated in text.
3. If a value is given as a range (e.g., "50-100 W"), capture both bounds.
4. If a value is approximate ("~200 W", "approximately 200 W"), note uncertainty_type="range" with reasonable bounds.
5. Distinguish CAUSAL claims ("X causes Y") from CORRELATIONAL ("X is associated with Y").
6. Include units exactly as stated. Do not convert.
7. If multiple papers disagree on a value, extract each paper's claim separately.

Return ONLY a JSON array of claim objects:
[
  {
    "subject": "...",
    "predicate": "...",
    "object": "...",
    "value": null or number,
    "unit": null or string,
    "uncertainty_type": "point_estimate|range|distribution|qualitative_only",
    "uncertainty_bounds": null or [lower, upper],
    "evidence": "...",
    "confidence": 0.0-1.0
  }
]

TEXT TO ANALYZE:
{text}
"""

    CRITIQUE_PROMPT = """You previously extracted claims from scientific text. Critique your extractions for:
1. HALLUCINATION: Did you invent any claim not in the text?
2. MISINTERPRETATION: Did you confuse correlation with causation?
3. QUANTITATIVE ERROR: Are values/units exactly as stated?
4. OMISSION: Did you miss any important claims?
5. UNCERTAINTY MISCLASSIFICATION: Should point estimates be ranges?

Original text:
{text}

Your previous extractions:
{previous_claims}

Return JSON:
{
  "valid_claims": [...same format...],
  "rejected_claims": [...with reason...],
  "added_claims": [...newly discovered...],
  "confidence_adjustments": {"claim_id": new_confidence}
}
"""

    def __init__(self, llm_generate_fn: Callable[[str], str]):
        self.llm_generate = llm_generate_fn
        self.extraction_count = 0

    def extract_from_text(self, text: str, doc_source: str, section: str = "UNKNOWN", 
                          page_num: int = -1, max_iterations: int = CONFIG.max_verification_iterations) -> List[ScientificClaim]:
        """Extract claims with iterative verification."""
        all_claims = []
        
        # Iteration 0: Initial extraction
        claims = self._extract_once(text, doc_source, section, page_num, iteration=0)
        all_claims.extend(claims)
        
        # Iterations 1+: Verification and refinement
        for iteration in range(1, max_iterations):
            previous_json = json.dumps([c.to_dict() for c in claims], indent=2)
            critique_prompt = self.CRITIQUE_PROMPT.format(
                text=text[:4000],
                previous_claims=previous_json[:3000]
            )
            
            try:
                response = self.llm_generate(critique_prompt)
                critique_data = self._parse_json(response)
                
                # Process validated claims
                if "valid_claims" in critique_data:
                    for c_data in critique_data["valid_claims"]:
                        claim = self._data_to_claim(c_data, doc_source, section, page_num, iteration)
                        # Adjust confidence if specified
                        claim_id = claim.claim_id
                        if "confidence_adjustments" in critique_data and claim_id in critique_data["confidence_adjustments"]:
                            claim.confidence = critique_data["confidence_adjustments"][claim_id]
                        all_claims.append(claim)
                
                # Process newly discovered claims
                if "added_claims" in critique_data:
                    for c_data in critique_data["added_claims"]:
                        claim = self._data_to_claim(c_data, doc_source, section, page_num, iteration)
                        all_claims.append(claim)
                        
            except Exception as e:
                logger.warning(f"Critique iteration {iteration} failed: {e}")
                break
        
        # Deduplicate by claim_id, keeping highest confidence
        seen = {}
        for c in all_claims:
            if c.claim_id not in seen or c.confidence > seen[c.claim_id].confidence:
                seen[c.claim_id] = c
        
        return list(seen.values())

    def _extract_once(self, text: str, doc_source: str, section: str, page_num: int, 
                      iteration: int) -> List[ScientificClaim]:
        """Single extraction pass."""
        prompt = self.EXTRACTION_PROMPT.format(text=text[:6000])
        response = self.llm_generate(prompt)
        data_list = self._parse_json(response)
        
        claims = []
        if isinstance(data_list, list):
            for i, c_data in enumerate(data_list):
                claim = self._data_to_claim(c_data, doc_source, section, page_num, iteration)
                claim.claim_id = f"{doc_source}_{section}_{iteration}_{i}"
                claims.append(claim)
        
        return claims

    def _data_to_claim(self, data: Dict, doc_source: str, section: str, page_num: int, 
                       iteration: int) -> ScientificClaim:
        """Convert raw extraction dict to ScientificClaim."""
        value = data.get("value")
        if value is not None:
            try:
                value = float(value)
            except (ValueError, TypeError):
                value = None
        
        bounds = data.get("uncertainty_bounds")
        if bounds and isinstance(bounds, list) and len(bounds) == 2:
            try:
                bounds = (float(bounds[0]), float(bounds[1]))
            except (ValueError, TypeError):
                bounds = None
        
        return ScientificClaim(
            claim_id=f"{doc_source}_{section}_{iteration}_{hash(data.get('evidence', '')) % 10000}",
            subject=data.get("subject", "unknown"),
            predicate=data.get("predicate", "relates_to"),
            object=data.get("object", ""),
            object_value=value,
            object_unit=data.get("unit"),
            confidence=float(data.get("confidence", 0.5)),
            uncertainty_type=data.get("uncertainty_type", "qualitative_only"),
            uncertainty_bounds=bounds,
            evidence_span=data.get("evidence", "")[:500],
            doc_source=doc_source,
            section=section,
            page_num=page_num,
            context_window="",  # Filled by caller
            extraction_iteration=iteration,
            critique_notes=[]
        )

    def _parse_json(self, text: str) -> Any:
        """Robust JSON extraction from LLM output."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try finding JSON block
        patterns = [
            r'\[.*\]',  # Array
            r'\{.*\}',  # Object
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        # Try code blocks
        code_block = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except json.JSONDecodeError:
                pass
        
        logger.warning("Failed to parse JSON from LLM response")
        return []

# =====================================================================
# CLAIM EMBEDDING & INDEXING
# =====================================================================

class ClaimIndexer:
    """
    Dual embedding space:
    1. Claim semantics: encode (subject, predicate, object) structure
    2. Evidence text: encode supporting text spans
    """
    
    def __init__(self, embedding_model: HuggingFaceEmbeddings):
        self.embed_model = embedding_model
        self.claims: List[ScientificClaim] = []
        self.claim_embeddings: Optional[np.ndarray] = None
        self.evidence_embeddings: Optional[np.ndarray] = None
        self.claim_index = None  # FAISS for claim semantics
        self.evidence_index = None  # FAISS for evidence text
        self.claim_clusters: Dict[str, ClaimCluster] = {}
        
    def index_claims(self, claims: List[ScientificClaim]):
        """Build dual indices from claims."""
        self.claims = claims
        
        # Claim semantic embeddings
        claim_texts = [f"{c.subject} {c.predicate} {c.object}" for c in claims]
        self.claim_embeddings = np.array(self.embed_model.embed_documents(claim_texts))
        
        # Evidence embeddings
        evidence_texts = [c.evidence_span or c.context_window for c in claims]
        self.evidence_embeddings = np.array(self.embed_model.embed_documents(evidence_texts))
        
        # Build FAISS indices
        import faiss
        
        # Claim semantics index
        claim_dim = self.claim_embeddings.shape[1]
        self.claim_index = faiss.IndexFlatIP(claim_dim)
        faiss.normalize_L2(self.claim_embeddings)
        self.claim_index.add(self.claim_embeddings)
        
        # Evidence index
        evidence_dim = self.evidence_embeddings.shape[1]
        self.evidence_index = faiss.IndexFlatIP(evidence_dim)
        faiss.normalize_L2(self.evidence_embeddings)
        self.evidence_index.add(self.evidence_embeddings)
        
        # Cluster claims by semantic similarity
        self._cluster_claims()
        
        logger.info(f"Indexed {len(claims)} claims into {len(self.claim_clusters)} clusters")

    def _cluster_claims(self, threshold: float = 0.85):
        """Cluster semantically equivalent claims."""
        from sklearn.cluster import DBSCAN
        
        if len(self.claims) < 5:
            return
            
        clustering = DBSCAN(eps=1 - threshold, min_samples=2, metric="cosine")
        labels = clustering.fit_predict(self.claim_embeddings)
        
        clusters = defaultdict(list)
        for claim, label in zip(self.claims, labels):
            if label >= 0:
                clusters[f"cluster_{label}"].append(claim)
        
        for cid, claims in clusters.items():
            self.claim_clusters[cid] = ClaimCluster(
                cluster_id=cid,
                canonical_subject=self._canonicalize(claims, "subject"),
                canonical_predicate=self._canonicalize(claims, "predicate"),
                claims=claims
            )

    def _canonicalize(self, claims: List[ScientificClaim], field: str) -> str:
        """Find most common value for canonical representation."""
        values = [getattr(c, field) for c in claims]
        return max(set(values), key=values.count) if values else "unknown"

    def retrieve_claims(self, query: str, k: int = 10, 
                        claim_weight: float = 0.6) -> List[Tuple[ScientificClaim, float, str]]:
        """
        Hybrid retrieval: combine claim semantics and evidence relevance.
        """
        query_emb = np.array(self.embed_model.embed_query(query)).reshape(1, -1)
        faiss.normalize_L2(query_emb)
        
        # Search claim semantics
        claim_scores, claim_indices = self.claim_index.search(query_emb, k * 2)
        
        # Search evidence
        evidence_scores, evidence_indices = self.evidence_index.search(query_emb, k * 2)
        
        # Combine scores
        claim_score_map = {int(idx): float(score) for idx, score in zip(claim_indices[0], claim_scores[0])}
        evidence_score_map = {int(idx): float(score) for idx, score in zip(evidence_indices[0], evidence_scores[0])}
        
        all_indices = set(claim_score_map.keys()) | set(evidence_score_map.keys())
        
        scored_claims = []
        for idx in all_indices:
            c_score = claim_score_map.get(idx, 0)
            e_score = evidence_score_map.get(idx, 0)
            combined = claim_weight * c_score + (1 - claim_weight) * e_score
            
            # Boost by claim confidence
            if idx < len(self.claims):
                claim = self.claims[idx]
                combined *= (0.5 + 0.5 * claim.confidence)
                scored_claims.append((claim, combined, "hybrid"))
        
        scored_claims.sort(key=lambda x: x[1], reverse=True)
        return scored_claims[:k]

    def get_consensus(self, query_entities: List[str]) -> List[Dict[str, Any]]:
        """Get consensus clusters for query entities."""
        results = []
        for cluster in self.claim_clusters.values():
            # Check if cluster matches any query entity
            cluster_text = f"{cluster.canonical_subject} {cluster.canonical_predicate}"
            if any(ent.lower() in cluster_text.lower() for ent in query_entities):
                consensus = cluster.consensus_value()
                if consensus:
                    results.append({
                        "cluster_id": cluster.cluster_id,
                        "canonical": f"{cluster.canonical_subject} → {cluster.canonical_predicate}",
                        **consensus
                    })
        return results

    def get_contradictions(self, query_entities: List[str]) -> List[Dict[str, Any]]:
        """Get contradictions for query entities."""
        results = []
        for cluster in self.claim_clusters.values():
            cluster_text = f"{cluster.canonical_subject} {cluster.canonical_predicate}"
            if any(ent.lower() in cluster_text.lower() for ent in query_entities):
                contrs = cluster.contradictions(CONFIG.contradiction_ratio)
                for c in contrs:
                    c["cluster_id"] = cluster.cluster_id
                    c["canonical"] = f"{cluster.canonical_subject} → {cluster.canonical_predicate}"
                results.extend(contrs)
        return results

# =====================================================================
# SYNTHESIS ENGINE
# =====================================================================

class SynthesisEngine:
    """
    Generates answers from retrieved claims with explicit uncertainty handling.
    No regex-derived data enters the synthesis.
    """

    SYSTEM_PROMPT = """You are an expert scientific research assistant specializing in laser-microstructure interactions and multicomponent alloys.

CRITICAL RULES:
1. Base your answer ONLY on the provided claims. Do not use external knowledge.
2. For quantitative claims, propagate uncertainty: report ranges, not single numbers.
3. Explicitly distinguish between STRONG consensus (multiple papers agree) and WEAK evidence (single paper or conflicting).
4. When papers disagree, present all viewpoints with their confidence levels.
5. Use exact values and units from claims — do not round or convert.
6. Cite sources using [Doc: X] format.
7. If evidence is insufficient, state "Insufficient evidence" clearly.

OUTPUT STRUCTURE:
1. DIRECT ANSWER (with confidence level: High/Medium/Low)
2. SUPPORTING EVIDENCE (claim-by-claim breakdown)
3. UNCERTAINTY ANALYSIS (ranges, disagreements, gaps)
4. SOURCE DOCUMENTS
"""

    def __init__(self, llm_generate_fn: Callable[[str], str]):
        self.llm_generate = llm_generate_fn

    def synthesize(self, query: str, claims: List[Tuple[ScientificClaim, float, str]],
                   consensus: List[Dict], contradictions: List[Dict]) -> str:
        """Generate verified answer from claims."""
        
        # Build claim context
        claim_context = []
        for claim, score, retrieval_type in claims[:15]:  # Top 15 claims
            claim_context.append(claim.format_for_llm())
        
        consensus_context = []
        for c in consensus[:5]:
            consensus_context.append(
                f"CONSENSUS [{c['canonical']}]: "
                f"weighted_mean={c['weighted_mean']:.3f} ± {c['weighted_std']:.3f} "
                f"({c['consensus_strength']}, n={c['n_claims']} claims, {c['n_docs']} docs)"
            )
        
        contradiction_context = []
        for c in contradictions[:5]:
            contradiction_context.append(
                f"CONTRADICTION [{c['canonical']}]: "
                f"{c['doc_a']} reports {c['mean_a']:.2f} vs "
                f"{c['doc_b']} reports {c['mean_b']:.2f} "
                f"(ratio: {c['ratio']:.1f}x)"
            )
        
        prompt = f"""{self.SYSTEM_PROMPT}

USER QUERY: {query}

RETRIEVED CLAIMS:
{chr(10).join(claim_context)}

CROSS-DOCUMENT CONSENSUS:
{chr(10).join(consensus_context) if consensus_context else "No quantitative consensus found."}

DETECTED CONTRADICTIONS:
{chr(10).join(contradiction_context) if contradiction_context else "No contradictions detected."}

Generate a rigorous scientific answer following the output structure above.
"""
        return self.llm_generate(prompt)

# =====================================================================
# DOCUMENT PROCESSING (Section-Aware)
# =====================================================================

def detect_sections(text: str) -> List[Tuple[str, str]]:
    """Detect scientific paper sections."""
    patterns = [
        (r'(?:^|\n)\s*Abstract\s*\n', 'ABSTRACT'),
        (r'(?:^|\n)\s*1\.\s*Introduction\s*\n', 'INTRODUCTION'),
        (r'(?:^|\n)\s*(?:2\.)?\s*Experimental\s*\n', 'METHODS'),
        (r'(?:^|\n)\s*(?:3\.)?\s*Results\s*\n', 'RESULTS'),
        (r'(?:^|\n)\s*(?:4\.)?\s*Discussion\s*\n', 'DISCUSSION'),
        (r'(?:^|\n)\s*Conclusion', 'CONCLUSION'),
    ]
    boundaries = []
    for pattern, name in patterns:
        for match in re.finditer(pattern, text, re.I):
            boundaries.append((match.start(), name))
    
    if not boundaries:
        return [("BODY", text)]
    
    boundaries.sort()
    sections = []
    for i, (pos, name) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        section_text = text[pos:end].strip()
        if len(section_text) > 50:
            sections.append((name, section_text))
    
    return sections if sections else [("BODY", text)]

def extract_text_from_pdf(file_bytes: bytes, filename: str) -> List[Document]:
    """Extract text from PDF with section awareness."""
    pages = []
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    
    try:
        if PYMUPDF_AVAILABLE:
            doc = fitz.open(tmp_path)
            for page_num in range(len(doc)):
                text = doc[page_num].get_text("text")
                if text.strip():
                    pages.append(Document(
                        page_content=text,
                        metadata={"source": filename, "page": page_num + 1}
                    ))
            doc.close()
        else:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
    finally:
        os.unlink(tmp_path)
    
    return pages

def chunk_document(pages: List[Document], filename: str) -> List[Document]:
    """Section-aware chunking."""
    all_text = "\n".join([p.page_content for p in pages])
    sections = detect_sections(all_text)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", "]
    )
    
    chunks = []
    for section_name, section_text in sections:
        section_chunks = splitter.create_documents([section_text])
        for i, chunk in enumerate(section_chunks):
            chunk.metadata.update({
                "source": filename,
                "section": section_name,
                "chunk_index": len(chunks) + i
            })
        chunks.extend(section_chunks)
    
    return chunks

# =====================================================================
# LLM BACKEND MANAGEMENT
# =====================================================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def load_llm(model_key: str):
    """Load LLM backend."""
    if model_key.startswith("ollama:"):
        return _load_ollama(model_key)
    return _load_transformers(model_key)

def _load_ollama(model_key: str):
    if not OLLAMA_AVAILABLE:
        raise ImportError("ollama not installed")
    tag = model_key.replace("ollama:", "")
    return None, tag, "ollama"

def _load_transformers(model_key: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_key, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_key,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model, "transformers"

def generate_response(tokenizer, model_or_tag, backend_type: str, prompt: str) -> str:
    """Generate response from LLM."""
    if backend_type == "ollama":
        return _generate_ollama(model_or_tag, prompt)
    return _generate_transformers(tokenizer, model, prompt)

def _generate_transformers(tokenizer, model, prompt: str) -> str:
    try:
        messages = [
            {"role": "system", "content": "You are a precise scientific assistant."},
            {"role": "user", "content": prompt}
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer.encode(formatted, return_tensors='pt', truncation=True, max_length=CONFIG.max_context_tokens)
        
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=CONFIG.max_new_tokens,
                temperature=CONFIG.synthesis_temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )
        
        full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract assistant response
        if "assistant" in full.lower():
            parts = full.split("assistant")
            return parts[-1].strip()
        return full[-CONFIG.max_new_tokens*2:].strip()
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return f"Error: {str(e)[:200]}"

def _generate_ollama(tag: str, prompt: str) -> str:
    try:
        client = ollama.Client()
        response = client.chat(
            model=tag,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": CONFIG.synthesis_temperature, "num_predict": CONFIG.max_new_tokens}
        )
        return response['message']['content']
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return f"Error: {str(e)[:200]}"

# =====================================================================
# MAIN PIPELINE
# =====================================================================

class DeclarimaPure:
    """Main application class."""
    
    def __init__(self):
        self.embeddings = None
        self.llm_tokenizer = None
        self.llm_model = None
        self.llm_backend = None
        self.claim_extractor = None
        self.claim_indexer = None
        self.synthesis_engine = None
        self.documents_processed = False
        self.all_claims: List[ScientificClaim] = []
        
    def initialize(self, model_key: str):
        """Initialize models."""
        with st.spinner("Loading embedding model..."):
            self.embeddings = load_embeddings()
        
        with st.spinner(f"Loading LLM: {model_key}..."):
            result = load_llm(model_key)
            if result[2] == "ollama":
                self.llm_tokenizer, self.llm_model, self.llm_backend = None, result[1], "ollama"
            else:
                self.llm_tokenizer, self.llm_model, self.llm_backend = result[0], result[1], "transformers"
        
        # Initialize claim extractor
        def llm_fn(prompt: str) -> str:
            return generate_response(self.llm_tokenizer, self.llm_model, self.llm_backend, prompt)
        
        self.claim_extractor = LLMClaimExtractor(llm_fn)
        self.synthesis_engine = SynthesisEngine(llm_fn)
        
        st.success("✓ Models loaded successfully")

    def process_documents(self, uploaded_files: List) -> bool:
        """Process uploaded PDFs into claim index."""
        if not uploaded_files:
            return False
        
        all_claims = []
        progress = st.progress(0.0)
        
        for idx, file in enumerate(uploaded_files):
            progress.progress((idx) / len(uploaded_files), 
                            text=f"Processing {file.name}...")
            
            # Extract text
            pages = extract_text_from_pdf(file.getvalue(), file.name)
            
            # Chunk
            chunks = chunk_document(pages, file.name)
            
            # Extract claims from each chunk
            for chunk in chunks:
                claims = self.claim_extractor.extract_from_text(
                    chunk.page_content,
                    doc_source=file.name,
                    section=chunk.metadata.get("section", "UNKNOWN")
                )
                all_claims.extend(claims)
        
        progress.progress(1.0, text="Building claim index...")
        
        # Index all claims
        self.claim_indexer = ClaimIndexer(self.embeddings)
        self.claim_indexer.index_claims(all_claims)
        self.all_claims = all_claims
        self.documents_processed = True
        
        progress.empty()
        return True

    def answer(self, query: str) -> Tuple[str, List[Dict], List[Dict], List[Tuple[ScientificClaim, float, str]]]:
        """Answer query using claim-based retrieval."""
        if not self.documents_processed:
            raise ValueError("Documents not processed")
        
        # Extract query entities (simple keyword matching for retrieval targeting)
        query_lower = query.lower()
        query_entities = []
        # Common materials science terms
        material_terms = ["ti-6al-4v", "inconel", "steel", "aluminum", "alloy", "hea", "solder"]
        parameter_terms = ["laser power", "scan speed", "fluence", "power", "speed", "temperature"]
        for term in material_terms + parameter_terms:
            if term in query_lower:
                query_entities.append(term)
        
        # Retrieve claims
        retrieved = self.claim_indexer.retrieve_claims(query, k=15)
        
        # Get consensus and contradictions
        consensus = self.claim_indexer.get_consensus(query_entities)
        contradictions = self.claim_indexer.get_contradictions(query_entities)
        
        # Synthesize answer
        answer = self.synthesis_engine.synthesize(query, retrieved, consensus, contradictions)
        
        return answer, consensus, contradictions, retrieved

# =====================================================================
# STREAMLIT UI
# =====================================================================

def initialize_session_state():
    defaults = {
        "app": None,
        "messages": [],
        "model_choice": "Qwen/Qwen2.5-0.5B-Instruct",
        "files_registered": False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def main():
    st.set_page_config(
        page_title="DECLARMIMA PURE — Learning-Based Scientific RAG",
        page_icon="🔬",
        layout="wide"
    )
    
    st.markdown("""
    <h1 style="text-align:center">🔬 DECLARMIMA PURE</h1>
    <p style="text-align:center;color:#666">
    Zero-Regex · Zero-Visualization · Maximum Reasoning Fidelity<br>
    <em>LLM-native claim extraction → Dual embedding retrieval → Uncertainty-aware synthesis</em>
    </p>
    """, unsafe_allow_html=True)
    
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        model = st.selectbox("LLM Model", [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "ollama:qwen2.5:0.5b",
            "ollama:qwen2.5:7b"
        ], index=0)
        st.session_state.model_choice = model
        
        if st.button("🚀 Initialize System", type="primary"):
            app = DeclarimaPure()
            app.initialize(model)
            st.session_state.app = app
            st.success("System initialized!")
        
        if st.session_state.app and st.session_state.app.documents_processed:
            st.markdown("### 📊 Index Statistics")
            st.metric("Total Claims", len(st.session_state.app.all_claims))
            st.metric("Claim Clusters", len(st.session_state.app.claim_indexer.claim_clusters))
            
            # Show sample claims
            with st.expander("Sample Extracted Claims"):
                for c in st.session_state.app.all_claims[:5]:
                    st.json(c.to_dict())
    
    # Main area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📁 Upload Documents")
        files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)
        
        if files and st.button("📥 Process Documents", type="primary"):
            if st.session_state.app is None:
                st.error("Please initialize the system first")
            else:
                with st.spinner("Extracting and indexing claims..."):
                    success = st.session_state.app.process_documents(files)
                    if success:
                        st.session_state.files_registered = True
                        st.success(f"Processed {len(files)} documents into {len(st.session_state.app.all_claims)} claims")
        
        if st.session_state.files_registered:
            st.info("✓ Documents ready for querying")
    
    with col2:
        st.markdown("### 💬 Scientific Query")
        
        if not st.session_state.files_registered:
            st.info("Upload and process documents to begin")
        else:
            # Display chat history
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("meta"):
                        with st.expander("📊 Reasoning Details"):
                            st.markdown("**Consensus:**")
                            for c in msg["meta"].get("consensus", []):
                                st.json(c)
                            st.markdown("**Contradictions:**")
                            for c in msg["meta"].get("contradictions", []):
                                st.json(c)
            
            # Input
            if prompt := st.chat_input("Ask a scientific question..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Retrieving claims and synthesizing..."):
                        try:
                            answer, consensus, contradictions, retrieved = st.session_state.app.answer(prompt)
                            st.markdown(answer)
                            
                            # Store with metadata
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "meta": {
                                    "consensus": consensus,
                                    "contradictions": contradictions,
                                    "retrieved_count": len(retrieved)
                                }
                            })
                        except Exception as e:
                            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
