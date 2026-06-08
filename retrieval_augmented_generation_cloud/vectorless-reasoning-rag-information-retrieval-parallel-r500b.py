#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
DECLARMIMA v20.1 — Architecturally Refactored Hybrid Vectorless RAG
====================================================================
Core improvements over v20.0:
1. Pipeline Architecture: Intent → Navigate → Extract → Validate → Synthesize
2. Dataclass-based structured data (Document, Section, ExtractedItem)
3. Protocol-based LLM abstraction (OllamaBackend, TransformersBackend)
4. Iterative MCTS Navigator with actual drill-down capability
5. CitationValidator with fuzzy matching fallback
6. AdaptiveResponseGenerator with strict template enforcement
7. Clean separation: Core logic is 100% independent of Streamlit
"""

import streamlit as st
import os
import sys
import json
import re
import hashlib
import logging
import warnings
import requests
import textwrap
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any, Protocol
from dataclasses import dataclass, field
from collections import defaultdict
from io import BytesIO
from enum import Enum

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============================================================================
# LOGGING
# ============================================================================
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logging.basicConfig(level=logging.INFO, handlers=[console_handler], force=True)
logger = logging.getLogger("DECLARMIMA_v21")

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Section:
    """A document section with hierarchical metadata."""
    id: str
    title: str
    start_page: int
    end_page: int
    summary: str = ""
    doc_name: str = ""

    @property
    def page_range(self) -> str:
        return f"{self.start_page}-{self.end_page}"


@dataclass
class ExtractedItem:
    """A structured data item extracted from document text."""
    item_type: str  # "quantitative", "equation", "mechanism", "prose"
    parameter_name: str = ""
    value: Optional[Union[float, str]] = None
    unit: str = ""
    equation_latex: str = ""
    context: str = ""
    doc_name: str = ""
    page: int = 0
    confidence: float = 0.0
    section_title: str = ""


@dataclass
class Document:
    """Container for a processed PDF document."""
    name: str
    filename: str
    pages: List[Dict[str, Any]] = field(default_factory=list)
    sections: List[Section] = field(default_factory=list)
    pdf_bytes: bytes = b""

    def get_page_text(self, page_num: int) -> str:
        """1-based page access."""
        if 0 < page_num <= len(self.pages):
            return self.pages[page_num - 1].get("text", "")
        return ""


class QueryIntent(Enum):
    """Classification of user query intent."""
    VALUE_EXTRACTION = "value_extraction"
    EQUATION = "equation"
    MECHANISM = "mechanism"
    COMPARISON = "comparison"
    OPEN_QUERY = "open_query"


# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================

class DependencyManager:
    """Centralized dependency checking with graceful degradation."""

    _instance = None
    _deps: Dict[str, bool] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._check_all()
        return cls._instance

    def _check_all(self):
        # PyMuPDF (required)
        try:
            import pymupdf as fitz
            self._deps["pymupdf"] = True
        except ImportError:
            try:
                import fitz
                self._deps["pymupdf"] = True
            except ImportError:
                self._deps["pymupdf"] = False
                logger.error("PyMuPDF required: pip install pymupdf")

        # Ollama
        try:
            import ollama
            self._deps["ollama"] = True
        except ImportError:
            self._deps["ollama"] = False

        # Transformers
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            self._deps["transformers"] = True
        except ImportError:
            self._deps["transformers"] = False

        # PyTorch
        try:
            import torch
            self._deps["torch"] = True
        except ImportError:
            self._deps["torch"] = False

        # rapidfuzz
        try:
            from rapidfuzz import fuzz
            self._deps["rapidfuzz"] = True
        except ImportError:
            self._deps["rapidfuzz"] = False

        # orjson
        try:
            import orjson
            self._deps["orjson"] = True
        except ImportError:
            self._deps["orjson"] = False

        logger.info(f"Dependencies: {sum(self._deps.values())}/{len(self._deps)} available")

    @property
    def available(self) -> Dict[str, bool]:
        return self._deps.copy()

    def check(self, name: str) -> bool:
        return self._deps.get(name, False)


DEPS = DependencyManager()

# ============================================================================
# PDF PROCESSING
# ============================================================================

class PDFProcessor:
    """Robust PDF text extraction with multiple fallback strategies."""

    def __init__(self, max_chars_per_page: int = 20000):
        self.max_chars = max_chars_per_page
        if not DEPS.check("pymupdf"):
            raise RuntimeError("PyMuPDF is required but not installed.")
        try:
            import pymupdf as fitz
            self.fitz = fitz
        except ImportError:
            import fitz
            self.fitz = fitz

    def extract_pages(self, file_bytes: bytes, max_pages: Optional[int] = None) -> List[Dict]:
        """Extract text page-by-page with error resilience."""
        try:
            doc = self.fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF: {e}")

        pages_data = []
        total = min(len(doc), max_pages) if max_pages else len(doc)

        for i in range(total):
            try:
                page = doc[i]
                text = page.get_text("text").strip()
                if text:
                    # Truncate if excessively long
                    if len(text) > self.max_chars:
                        text = text[:self.max_chars] + "\n...[TRUNCATED]"
                    pages_data.append({"page_num": i + 1, "text": text})
            except Exception as e:
                logger.warning(f"Error extracting page {i+1}: {e}")
                continue

        doc.close()
        return pages_data

    def extract_pages_with_latex(self, doc_path: str, page_numbers: List[int]) -> Dict[int, str]:
        """LaTeX-aware extraction using pymupdf4llm if available."""
        result = {}

        # Try pymupdf4llm first
        if DEPS.check("pymupdf"):
            try:
                import pymupdf4llm
                chunks = pymupdf4llm.to_markdown(doc_path, page_chunks=True)
                chunk_map = {}
                for chunk in chunks:
                    meta = chunk.get("metadata", {})
                    p_num = meta.get("page_number", 0)
                    if p_num > 0:
                        chunk_map[p_num] = chunk.get("text", "")

                for pnum in page_numbers:
                    if pnum in chunk_map:
                        result[pnum] = chunk_map[pnum]

                if len(result) == len(page_numbers):
                    return result
            except Exception as e:
                logger.warning(f"pymupdf4llm failed, falling back: {e}")

        # Standard fallback
        doc = self.fitz.open(doc_path)
        for pnum in page_numbers:
            if 1 <= pnum <= len(doc) and pnum not in result:
                text = doc[pnum - 1].get_text("text")
                if len(text) > self.max_chars:
                    text = text[:self.max_chars] + "\n...[TRUNCATED]"
                result[pnum] = text
        doc.close()
        return result


# ============================================================================
# LLM ABSTRACTION LAYER
# ============================================================================

class LLMBackend(Protocol):
    """Protocol defining the LLM interface."""

    def generate(self, prompt: str, max_new_tokens: int = 1024, 
                 temperature: float = 0.1, system_prompt: Optional[str] = None,
                 fast_json: bool = False) -> str: ...

    @property
    def model_name(self) -> str: ...

    @property
    def backend_type(self) -> str: ...


class OllamaBackend:
    """Ollama API backend."""

    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        import ollama
        self.client = ollama.Client(host=host)
        self._model_name = model_name
        self._host = host

        # Verify server is up
        try:
            requests.get(f"{host}/api/tags", timeout=5)
            logger.info(f"Ollama backend connected: {model_name}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Ollama server not running at {host}")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def backend_type(self) -> str:
        return "ollama"

    def generate(self, prompt: str, max_new_tokens: int = 1024,
                 temperature: float = 0.1, system_prompt: Optional[str] = None,
                 fast_json: bool = False) -> str:
        import ollama
        options = {"temperature": temperature, "num_predict": max_new_tokens}
        if fast_json:
            options["format"] = "json"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = self.client.chat(
                model=self._model_name,
                messages=messages,
                options=options,
                stream=False
            )
            return resp.get("message", {}).get("content", "").strip()
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return f"Error: {str(e)[:100]}"


class TransformersBackend:
    """HuggingFace Transformers local backend."""

    def __init__(self, model_name: str, use_4bit: bool = True, device: Optional[str] = None):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch

        self._model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_4bit = use_4bit and self.device == "cuda"

        logger.info(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32
        }

        if self.use_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if self.device == "cuda":
            self.model.to(self.device)
        self.model.eval()
        logger.info("Transformers model loaded successfully.")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def backend_type(self) -> str:
        return "transformers"

    def generate(self, prompt: str, max_new_tokens: int = 1024,
                 temperature: float = 0.1, system_prompt: Optional[str] = None,
                 fast_json: bool = False) -> str:
        import torch

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            if hasattr(self.tokenizer, "apply_chat_template"):
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract assistant response
            if "assistant" in response:
                parts = response.split("assistant")
                response = parts[-1].strip()
            return response
        except Exception as e:
            logger.error(f"Transformers generation error: {e}")
            return f"Error: {str(e)[:100]}"


class HybridLLM:
    """Unified LLM interface with automatic backend selection."""

    MODEL_OPTIONS = {
        "[Ollama] qwen2.5:0.5b (Fastest, CPU OK)": "ollama:qwen2.5:0.5b",
        "[Ollama] qwen2.5:1.5b (Balanced)": "ollama:qwen2.5:1.5b",
        "[Ollama] qwen2.5:7b (Recommended for RAG)": "ollama:qwen2.5:7b",
        "[Ollama] qwen2.5:14b (Max Reasoning)": "ollama:qwen2.5:14b",
        "[Ollama] llama3.1:8b (Meta Standard)": "ollama:llama3.1:8b",
        "[Ollama] mistral:7b (High JSON Reliability)": "ollama:mistral:7b",
        "[Ollama] gemma2:9b (Scientific Nuance)": "ollama:gemma2:9b",
        "[HF] Qwen2.5-0.5B-Instruct (Tiny, CPU OK)": "Qwen/Qwen2.5-0.5B-Instruct",
        "[HF] Qwen2.5-1.5B-Instruct (Small, Fast)": "Qwen/Qwen2.5-1.5B-Instruct",
        "[HF] Qwen2.5-3B-Instruct (Compact)": "Qwen/Qwen2.5-3B-Instruct",
        "[HF] Qwen2.5-7B-Instruct (Local)": "Qwen/Qwen2.5-7B-Instruct",
        "[HF] Mistral-7B-Instruct-v0.3 (Local)": "mistralai/Mistral-7B-Instruct-v0.3",
        "[HF] Llama-3.2-1B-Instruct (Tiny)": "meta-llama/Llama-3.2-1B-Instruct",
        "[HF] Llama-3.2-3B-Instruct (Small)": "meta-llama/Llama-3.2-3B-Instruct",
        "[HF] Llama-3.1-8B-Instruct (Local)": "meta-llama/Llama-3.1-8B-Instruct",
        "[HF] SmolLM2-1.7B-Instruct (Ultra Small)": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    }

    def __init__(self, model_choice: str, use_4bit: bool = True):
        internal_name = self.MODEL_OPTIONS.get(model_choice, model_choice)

        if internal_name.startswith("ollama:"):
            model_name = internal_name.replace("ollama:", "")
            self._backend: LLMBackend = OllamaBackend(model_name)
        else:
            self._backend = TransformersBackend(internal_name, use_4bit=use_4bit)

    @property
    def model_name(self) -> str:
        return self._backend.model_name

    @property
    def backend_type(self) -> str:
        return self._backend.backend_type

    def generate(self, prompt: str, max_new_tokens: int = 1024,
                 temperature: float = 0.1, system_prompt: Optional[str] = None,
                 fast_json: bool = False) -> str:
        return self._backend.generate(
            prompt, max_new_tokens, temperature, system_prompt, fast_json
        )


# ============================================================================
# JSON UTILITIES
# ============================================================================

class JSONExtractor:
    """Bulletproof JSON extraction from LLM responses."""

    @staticmethod
    def extract(text: str) -> Optional[Any]:
        if not text:
            return None

        # Direct parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass

        # Markdown code block
        md_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if md_match:
            try:
                return json.loads(md_match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        # Find JSON object or array
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                candidate = text[start:end+1]
                # Fix common LLM JSON errors
                candidate = re.sub(r",\s*([\]}])", r"\1", candidate)  # Trailing commas
                candidate = re.sub(r"\s+", " ", candidate)  # Normalize whitespace
                try:
                    return json.loads(candidate)
                except (json.JSONDecodeError, ValueError):
                    pass

        return None


# ============================================================================
# ARCHITECTURAL PILLAR 1: SCIENTIFIC INTENT ROUTER
# ============================================================================

class IntentRouter:
    """Classifies query intent to drive downstream pipeline behavior."""

    PATTERNS = {
        QueryIntent.VALUE_EXTRACTION: [
            r"\bvalue\b", r"\bhow much\b", r"\b\d+\s*(?:W|kW|MPa|mm/s|GPa|°C|K|m/s)\b",
            r"\btable\b", r"\blist\b", r"\bwhat is the\b.*\bof\b", r"\bparameter\b",
            r"\bproperty\b", r"\bmeasurement\b", r"\bdimension\b"
        ],
        QueryIntent.EQUATION: [
            r"\bequation\b", r"\bformula\b", r"\bconstitutive\b", r"\bnavier[- ]stokes\b",
            r"\bcahn[- ]hilliard\b", r"\bgoverning\b", r"\bmathematical model\b",
            r"\bexpression\b", r"\bderive\b"
        ],
        QueryIntent.MECHANISM: [
            r"\bwhy\b", r"\bhow does\b", r"\bmechanism\b", r"\bcause\b",
            r"\bdriving force\b", r"\bexplain\b", r"\bprocess\b", r"\bphenomenon\b",
            r"\bwhat happens\b"
        ],
        QueryIntent.COMPARISON: [
            r"\bcompare\b", r"\bdifference\b", r"\bversus\b", r"\bvs\b",
            r"\bhigher than\b", r"\blower than\b", r"\bbetter\b", r"\bworse\b"
        ],
    }

    def route(self, query: str) -> Tuple[QueryIntent, Dict[str, Any]]:
        """Returns intent and metadata for pipeline configuration."""
        q_lower = query.lower()

        for intent, patterns in self.PATTERNS.items():
            if any(re.search(p, q_lower) for p in patterns):
                return intent, {"output_format": intent.value, "requires_table": intent == QueryIntent.VALUE_EXTRACTION}

        return QueryIntent.OPEN_QUERY, {"output_format": "prose", "requires_table": False}


# ============================================================================
# ARCHITECTURAL PILLAR 2: ITERATIVE TREE NAVIGATOR (MCTS-Style)
# ============================================================================

class NavigationTrace:
    """Records the reasoning trace of the navigator."""

    def __init__(self):
        self.steps: List[Dict[str, Any]] = []

    def add(self, step_num: int, reasoning: str, actions: List[Dict], 
            selected_sections: List[Section]):
        self.steps.append({
            "step": step_num,
            "reasoning": reasoning,
            "actions": actions,
            "selected_count": len(selected_sections)
        })

    def to_text(self) -> str:
        lines = []
        for step in self.steps:
            lines.append(f"**Step {step['step']}:** {step['reasoning']}")
            lines.append(f"→ Selected {step['selected_count']} sections")
        return "\n\n".join(lines)


class TreeNavigator:
    """Agentic drill-down navigator with iterative section refinement."""

    def __init__(self, llm: HybridLLM, max_steps: int = 2, max_sections_per_step: int = 3):
        self.llm = llm
        self.max_steps = max_steps
        self.max_sections = max_sections_per_step
        self.trace = NavigationTrace()

    def _build_forest(self, documents: Dict[str, Document]) -> str:
        """Build a compact forest description from document sections."""
        lines = []
        for doc_name, doc in documents.items():
            lines.append(f"--- Document: {doc_name} ---")
            for sec in doc.sections:
                ns_id = f"{doc_name}:::{sec.id}"
                summary = sec.summary[:200] if sec.summary else "No summary"
                lines.append(
                    f"ID: {ns_id} | Title: {sec.title} | "
                    f"Pages: {sec.page_range} | Summary: {summary}"
                )
        return "\n".join(lines)

    def navigate(self, query: str, documents: Dict[str, Document]) -> Tuple[List[Dict], NavigationTrace]:
        """
        Iteratively navigate document trees to find relevant sections.
        Returns list of retrieved chunks and the navigation trace.
        """
        forest = self._build_forest(documents)
        current_forest = forest
        all_chunks: List[Dict] = []

        for step in range(self.max_steps):
            prompt = f"""You are a scientific document navigator. Your task is to find sections containing information relevant to the user's query.

Query: "{query}"

Available Document Sections:
{current_forest}

Instructions:
1. Analyze which sections are MOST likely to contain the answer.
2. Return ONLY a JSON object in this exact format:
{{
  "reasoning": "Step-by-step analysis of why these sections were selected...",
  "actions": [
    {{"id": "doc_name:::sec_id", "action": "extract_text", "rationale": "why this section"}}
  ]
}}
3. Select at most {self.max_sections} sections.
4. If no sections seem relevant, return an empty actions list."""

            system = "You are a precise navigation agent. Return ONLY valid JSON with no markdown formatting."
            resp = self.llm.generate(prompt, max_new_tokens=1024, system_prompt=system, fast_json=True)
            actions_data = JSONExtractor.extract(resp)

            if not actions_data or not isinstance(actions_data, dict):
                logger.warning(f"Navigator step {step+1}: Invalid response format")
                break

            reasoning = actions_data.get("reasoning", "No reasoning provided.")
            actions = actions_data.get("actions", [])

            if not actions:
                self.trace.add(step + 1, reasoning, [], [])
                break

            selected_sections: List[Section] = []
            step_chunks: List[Dict] = []

            for action in actions[:self.max_sections]:
                ns_id = action.get("id", "")
                if ":::" not in ns_id:
                    continue

                doc_name, sec_id = ns_id.split(":::", 1)
                if doc_name not in documents:
                    continue

                doc = documents[doc_name]
                sec = next((s for s in doc.sections if s.id == sec_id), None)
                if not sec:
                    continue

                selected_sections.append(sec)

                # Fetch raw text
                text_parts = []
                for p_num in range(sec.start_page, sec.end_page + 1):
                    text = doc.get_page_text(p_num)
                    if text:
                        text_parts.append(text)

                full_text = "\n".join(text_parts)
                step_chunks.append({
                    "doc_name": doc_name,
                    "section": sec,
                    "full_text": full_text,
                    "rationale": action.get("rationale", "")
                })

            self.trace.add(step + 1, reasoning, actions, selected_sections)
            all_chunks.extend(step_chunks)

            # If we found good chunks, we can stop early
            if step_chunks and step < self.max_steps - 1:
                # Build a focused forest from just the selected document for next step
                # This simulates drilling down
                focused_lines = []
                for chunk in step_chunks:
                    doc_name = chunk["doc_name"]
                    sec = chunk["section"]
                    focused_lines.append(
                        f"ID: {doc_name}:::{sec.id} | Title: {sec.title} | "
                        f"Pages: {sec.page_range} | Text preview: {chunk['full_text'][:300]}..."
                    )
                current_forest = "\n".join(focused_lines)
            else:
                break

        # Deduplicate chunks by section ID
        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            key = f"{chunk['doc_name']}:::{chunk['section'].id}"
            if key not in seen:
                seen.add(key)
                unique_chunks.append(chunk)

        return unique_chunks, self.trace


# ============================================================================
# ARCHITECTURAL PILLAR 3: STRUCTURED EXTRACTOR & CITATION VALIDATOR
# ============================================================================

class StructuredExtractor:
    """Extracts structured data items from raw text chunks."""

    EXTRACTION_PROMPT = """Extract all quantitative data, equations, mechanisms, and key facts from the following scientific text.

Return a JSON list of objects. Each object MUST have these fields:
- "item_type": "quantitative" | "equation" | "mechanism" | "fact"
- "parameter_name": name of the parameter or concept (string)
- "value": numeric value if applicable, otherwise null
- "unit": unit of measurement (string, empty if none)
- "equation_latex": LaTeX string if item_type is "equation", otherwise empty
- "context": the exact sentence or phrase from the text containing this information
- "page": page number (integer)

Text to analyze:
{text}

Return ONLY a valid JSON list. No markdown, no explanations."""

    def __init__(self, llm: HybridLLM):
        self.llm = llm

    def extract(self, chunks: List[Dict]) -> List[ExtractedItem]:
        """Extract structured items from text chunks."""
        items: List[ExtractedItem] = []

        for chunk in chunks:
            doc_name = chunk["doc_name"]
            sec = chunk["section"]
            text = chunk["full_text"]

            # Truncate to avoid context overflow
            truncated = text[:4000] if len(text) > 4000 else text

            prompt = self.EXTRACTION_PROMPT.format(text=truncated)
            resp = self.llm.generate(prompt, max_new_tokens=1024, 
                                    system_prompt="Return ONLY valid JSON list.",
                                    fast_json=True)
            parsed = JSONExtractor.extract(resp)

            if not isinstance(parsed, list):
                logger.warning(f"Extractor returned non-list for {doc_name}::{sec.id}")
                continue

            for raw_item in parsed:
                if not isinstance(raw_item, dict):
                    continue

                item = ExtractedItem(
                    item_type=raw_item.get("item_type", "fact"),
                    parameter_name=raw_item.get("parameter_name", ""),
                    value=raw_item.get("value"),
                    unit=raw_item.get("unit", ""),
                    equation_latex=raw_item.get("equation_latex", ""),
                    context=raw_item.get("context", ""),
                    doc_name=doc_name,
                    page=raw_item.get("page", sec.start_page),
                    section_title=sec.title
                )
                items.append(item)

        return items


class CitationValidator:
    """Cross-checks extracted items against raw PDF text to prevent hallucinations."""

    def __init__(self, use_fuzzy: bool = True):
        self.use_fuzzy = use_fuzzy and DEPS.check("rapidfuzz")
        if self.use_fuzzy:
            from rapidfuzz import fuzz
            self.fuzz = fuzz

    def verify(self, items: List[ExtractedItem], documents: Dict[str, Document]) -> List[ExtractedItem]:
        """Verify each item against raw text and assign confidence scores."""
        verified: List[ExtractedItem] = []

        for item in items:
            doc_name = item.doc_name
            page_num = item.page

            if doc_name not in documents:
                item.confidence = 0.1
                verified.append(item)
                continue

            doc = documents[doc_name]
            raw_text = doc.get_page_text(page_num).lower()

            if not raw_text:
                item.confidence = 0.1
                verified.append(item)
                continue

            # Verification strategies
            val_str = str(item.value).lower() if item.value is not None else ""
            param = item.parameter_name.lower()
            context = item.context.lower()

            confidence = 0.0

            # Strategy 1: Exact value match in raw text
            if val_str and val_str in raw_text:
                confidence = max(confidence, 0.95)

            # Strategy 2: Parameter name in raw text
            if param and param in raw_text:
                confidence = max(confidence, 0.75)

            # Strategy 3: Context substring in raw text
            if context and len(context) > 10:
                if context in raw_text:
                    confidence = max(confidence, 0.9)
                elif self.use_fuzzy:
                    # Fuzzy match for partial context
                    score = self.fuzz.partial_ratio(context, raw_text) / 100.0
                    if score > 0.8:
                        confidence = max(confidence, score * 0.85)

            # Strategy 4: Equations are harder to verify; trust LLM more
            if item.item_type == "equation" and item.equation_latex:
                # Check if key terms from LaTeX appear in text
                latex_terms = re.findall(r"\\[a-zA-Z]+", item.equation_latex)
                if latex_terms:
                    matches = sum(1 for term in latex_terms if term.lower() in raw_text)
                    if matches > 0:
                        confidence = max(confidence, 0.6 + 0.1 * matches)
                    else:
                        confidence = max(confidence, 0.5)
                else:
                    confidence = max(confidence, 0.5)

            # Strategy 5: Unit match
            if item.unit and item.unit.lower() in raw_text:
                confidence = max(confidence, 0.7)

            # Default minimum confidence
            if confidence == 0.0:
                confidence = 0.3

            item.confidence = min(confidence, 1.0)
            verified.append(item)

        # Sort by confidence descending
        verified.sort(key=lambda x: x.confidence, reverse=True)
        return verified


# ============================================================================
# ARCHITECTURAL PILLAR 4: ADAPTIVE RESPONSE SYNTHESIZER
# ============================================================================

class ResponseSynthesizer:
    """Generates format-enforced responses based on detected intent."""

    def __init__(self, llm: HybridLLM):
        self.llm = llm

    def _build_evidence_block(self, items: List[ExtractedItem], chunks: List[Dict]) -> str:
        """Build a structured evidence block from verified items and chunks."""
        lines = []

        # Add chunk context
        for chunk in chunks:
            lines.append(
                f"[Source: {chunk['doc_name']}, Section: {chunk['section'].title}, "
                f"Pages {chunk['section'].page_range}]"
            )
            preview = chunk["full_text"][:500].replace("\n", " ")
            lines.append(f"Text: {preview}...")
            lines.append("")

        # Add extracted items
        for item in items[:15]:
            if item.item_type == "equation":
                lines.append(
                    f"- [{item.doc_name}, p.{item.page}] Equation: {item.equation_latex}"
                )
            elif item.item_type == "quantitative":
                lines.append(
                    f"- [{item.doc_name}, p.{item.page}] {item.parameter_name}: "
                    f"{item.value} {item.unit} (confidence: {item.confidence:.2f})"
                )
            else:
                lines.append(
                    f"- [{item.doc_name}, p.{item.page}] {item.parameter_name}: "
                    f"{item.context[:100]}..."
                )

        return "\n".join(lines)

    def synthesize(self, query: str, items: List[ExtractedItem], 
                   chunks: List[Dict], intent: QueryIntent) -> str:
        """Generate a response strictly formatted according to intent."""
        evidence = self._build_evidence_block(items, chunks)

        if intent == QueryIntent.VALUE_EXTRACTION:
            prompt = f"""You are a scientific data analyst. Create a comprehensive Markdown table answering the user's query.

Query: {query}

Evidence from documents:
{evidence}

Requirements:
1. Output ONLY a Markdown table with columns: | Parameter | Value | Unit | Source Document | Page | Notes |
2. Every row must be directly supported by the evidence above.
3. If a value has low confidence (<0.6), mark it with ⚠️ in the Notes column.
4. Do NOT invent data not present in the evidence.
5. After the table, add a brief summary paragraph (2-3 sentences)."""
            system = "You create precise Markdown tables. Never invent data."

        elif intent == QueryIntent.EQUATION:
            prompt = f"""You are a scientific editor. Present the governing equations from the evidence.

Query: {query}

Evidence from documents:
{evidence}

Requirements:
1. State each equation in LaTeX format using $$ ... $$ blocks.
2. Immediately below each equation, define ALL variables used.
3. Cite the source document and page number for each equation like [DocName, p.X].
4. If the derivation or assumptions are mentioned in the evidence, include them.
5. Do NOT invent equations not present in the evidence."""
            system = "You format scientific equations in LaTeX. Be precise and cite sources."

        elif intent == QueryIntent.MECHANISM:
            prompt = f"""You are a scientific explainer. Explain the physical mechanism answering the user's query.

Query: {query}

Evidence from documents:
{evidence}

Requirements:
1. Explain the mechanism step-by-step using causal language ("because", "leads to", "results in").
2. Every factual claim must have an inline citation like [DocName, p.X].
3. Use bold for key terms and concepts.
4. If the evidence mentions numerical thresholds or conditions, include them.
5. If evidence is insufficient, state what is known and what is uncertain."""
            system = "You explain physical mechanisms clearly with inline citations."

        elif intent == QueryIntent.COMPARISON:
            prompt = f"""You are a comparative analyst. Compare the items mentioned in the query.

Query: {query}

Evidence from documents:
{evidence}

Requirements:
1. Create a comparison table if applicable (Markdown format).
2. Highlight similarities and differences explicitly.
3. Use inline citations [DocName, p.X] for every comparative claim.
4. Note if the evidence is insufficient for a complete comparison."""
            system = "You compare scientific data objectively with proper citations."

        else:  # OPEN_QUERY
            prompt = f"""You are a research assistant. Answer the user's query comprehensively based ONLY on the provided evidence.

Query: {query}

Evidence from documents:
{evidence}

Requirements:
1. Be thorough but concise.
2. Use inline citations [DocName, p.X] for every factual claim.
3. If the answer is not in the evidence, say "I don't know based on the provided documents."
4. Structure the answer with clear headings if multiple aspects are covered."""
            system = "You answer questions accurately with inline citations."

        return self.llm.generate(prompt, max_new_tokens=2048, temperature=0.1, system_prompt=system)


# ============================================================================
# DOCUMENT INDEXER
# ============================================================================

class DocumentIndexer:
    """Builds hierarchical section trees from PDF documents."""

    def __init__(self, llm: HybridLLM, chunk_size: int = 5):
        self.llm = llm
        self.chunk_size = chunk_size
        self.pdf_processor = PDFProcessor()

    def index(self, uploaded_files: List[Any]) -> Dict[str, Document]:
        """Index multiple PDFs into structured Document objects."""
        documents: Dict[str, Document] = {}
        total = len(uploaded_files)

        for idx, uploaded_file in enumerate(uploaded_files):
            doc_name = os.path.splitext(uploaded_file.name)[0]
            if doc_name in documents:
                doc_name = f"{doc_name}_{idx+1}"

            logger.info(f"Indexing ({idx+1}/{total}): {uploaded_file.name}")

            pdf_bytes = uploaded_file.read()
            pages_data = self.pdf_processor.extract_pages(pdf_bytes)

            # Extract sections using LLM
            sections = self._extract_sections(pages_data)

            # Generate summaries for each section
            sections = self._summarize_sections(sections, pages_data)

            documents[doc_name] = Document(
                name=doc_name,
                filename=uploaded_file.name,
                pages=pages_data,
                sections=sections,
                pdf_bytes=pdf_bytes
            )

        return documents

    def _extract_sections(self, pages_data: List[Dict]) -> List[Section]:
        """Use LLM to identify document sections and their page ranges."""
        sections: List[Section] = []

        for i in range(0, len(pages_data), self.chunk_size):
            chunk = pages_data[i:i+self.chunk_size]
            text_with_tags = "\n".join([
                f"<page_{p['page_num']}>\n{p['text']}\n</page_{p['page_num']}>"
                for p in chunk
            ])

            prompt = f"""Analyze the following document text and extract the main section headings with their starting page numbers.

Return a JSON list of objects:
[{{"title": "Section Title", "start_page": 1}}]

Rules:
- Only include actual content sections (Introduction, Methods, Results, etc.)
- Ignore headers, footers, page numbers, and running headers.
- Use the page numbers indicated in the XML tags.

Text:
{text_with_tags}"""

            system = "You are an expert document analyzer. Return ONLY a valid JSON list."
            resp = self.llm.generate(prompt, max_new_tokens=1024, system_prompt=system, fast_json=True)
            parsed = JSONExtractor.extract(resp)

            if parsed and isinstance(parsed, list):
                for item in parsed:
                    if "title" in item and "start_page" in item:
                        sections.append(Section(
                            id=f"sec_{len(sections)+1:03d}",
                            title=item["title"],
                            start_page=item["start_page"],
                            end_page=len(pages_data)  # Temporary, will fix below
                        ))

        # Sort and fix end pages
        sections.sort(key=lambda s: s.start_page)
        for i in range(len(sections)):
            if i + 1 < len(sections):
                sections[i].end_page = sections[i + 1].start_page - 1
            else:
                sections[i].end_page = len(pages_data)

        return sections

    def _summarize_sections(self, sections: List[Section], pages_data: List[Dict]) -> List[Section]:
        """Generate LLM summaries for each section."""
        for sec in sections:
            start_idx = sec.start_page - 1
            end_idx = sec.end_page
            section_text = "\n".join([
                p["text"] for p in pages_data[start_idx:end_idx]
            ])

            # Truncate for summary
            truncated = section_text[:4000] + "..." if len(section_text) > 4000 else section_text

            prompt = f"""Summarize the following text in 2-3 concise sentences.
Focus on key facts, numbers, findings, and the section's purpose.

Text:
{truncated}"""

            system = "You are a scientific summarizer. Be concise and factual."
            summary = self.llm.generate(prompt, max_new_tokens=256, system_prompt=system)
            sec.summary = summary.strip()

        return sections


# ============================================================================
# UNIFIED PIPELINE
# ============================================================================

class RAGPipeline:
    """Orchestrates the full retrieval and generation pipeline."""

    def __init__(self, llm: HybridLLM, max_context_chars: int = 15000):
        self.llm = llm
        self.max_context = max_context_chars
        self.router = IntentRouter()
        self.navigator = TreeNavigator(llm)
        self.extractor = StructuredExtractor(llm)
        self.validator = CitationValidator()
        self.synthesizer = ResponseSynthesizer(llm)

    def run(self, query: str, documents: Dict[str, Document]) -> Dict[str, Any]:
        """Execute the full pipeline and return structured results."""
        # Phase 1: Intent Classification
        intent, intent_meta = self.router.route(query)
        logger.info(f"Query intent detected: {intent.value}")

        # Phase 2: Tree Navigation
        chunks, trace = self.navigator.navigate(query, documents)

        if not chunks:
            return {
                "answer": "I could not find any relevant sections in the selected documents to answer your query.",
                "thinking": trace.to_text(),
                "items": [],
                "intent": intent.value,
                "confidence": 0.0
            }

        # Phase 3: Structured Extraction
        raw_items = self.extractor.extract(chunks)

        # Phase 4: Citation Validation
        verified_items = self.validator.verify(raw_items, documents)

        # Filter low-confidence hallucinations
        high_conf_items = [i for i in verified_items if i.confidence > 0.5]

        # Phase 5: Adaptive Synthesis
        answer = self.synthesizer.synthesize(query, high_conf_items, chunks, intent)

        # Calculate overall confidence
        if high_conf_items:
            avg_conf = sum(i.confidence for i in high_conf_items) / len(high_conf_items)
        else:
            avg_conf = 0.0

        return {
            "answer": answer,
            "thinking": trace.to_text(),
            "items": high_conf_items,
            "intent": intent.value,
            "confidence": avg_conf,
            "retrieved_chunks": chunks
        }


# ============================================================================
# STREAMLIT UI
# ============================================================================

@st.cache_resource(show_spinner="Initializing LLM...")
def get_cached_llm(model_choice: str, use_4bit: bool = True) -> HybridLLM:
    """Cached LLM initialization."""
    return HybridLLM(model_choice, use_4bit=use_4bit)


def render_sidebar():
    """Render configuration sidebar."""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        model_keys = list(HybridLLM.MODEL_OPTIONS.keys())
        if "llm_model_choice" not in st.session_state:
            st.session_state.llm_model_choice = model_keys[2]

        selected = st.selectbox(
            "Select LLM Model",
            options=model_keys,
            index=model_keys.index(st.session_state.llm_model_choice),
            key="llm_model_select",
            help="Ollama (fast API) or HuggingFace Transformers (local loading)"
        )
        st.session_state.llm_model_choice = selected

        model_key = HybridLLM.MODEL_OPTIONS[selected]
        if model_key.startswith("ollama:"):
            st.caption("🟢 Backend: Ollama (API)")
            st.caption(f"Model: `{model_key.replace('ollama:', '')}`")
        else:
            st.caption("🔵 Backend: Transformers (Local)")
            st.caption(f"Model: `{model_key}`")

        if not model_key.startswith("ollama:"):
            st.checkbox("Use 4-bit quantization (saves VRAM)", value=True, key="use_4bit")
            if DEPS.check("torch"):
                import torch
                if torch.cuda.is_available():
                    st.caption(f"GPU: {torch.cuda.get_device_name(0)}")
                else:
                    st.warning("⚠️ No GPU detected. Model will run on CPU (slow).")
        else:
            st.session_state.use_4bit = False

        st.markdown("---")
        st.markdown("#### 📊 System Status")

        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"{'✅' if DEPS.check('ollama') else '❌'} Ollama")
        with cols[1]:
            st.markdown(f"{'✅' if DEPS.check('transformers') else '❌'} Transformers")
        cols2 = st.columns(2)
        with cols2[0]:
            st.markdown(f"{'✅' if DEPS.check('torch') else '❌'} PyTorch")
        with cols2[1]:
            st.markdown(f"{'✅' if DEPS.check('pymupdf') else '❌'} PyMuPDF")

        st.markdown("---")

        with st.expander("Advanced Settings", expanded=False):
            st.slider("Max context chars", 5000, 30000, 15000, 1000,
                     key="max_context_chars",
                     help="Maximum characters to send to LLM as context")
            st.slider("Chunk size (pages)", 1, 10, 5, 1,
                     key="chunk_size",
                     help="Pages per chunk for section extraction")
            st.slider("Nav max steps", 1, 5, 2, 1,
                     key="nav_max_steps",
                     help="Maximum navigator drill-down steps")

        st.markdown("---")

        if st.button("🗑️ Clear Cache & Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def run_streamlit():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="DECLARMIMA v21 — Architected RAG",
        layout="wide"
    )
    st.title("🌲 DECLARMIMA v21 — Architected Vectorless RAG")
    st.markdown(
        "**Pipeline:** Intent Router → Tree Navigator → Structured Extractor → "
        "Citation Validator → Adaptive Synthesizer  
"
        "**No Vector DBs. No Embeddings. Agentic. Verified.**"
    )

    # Session state initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_docs_for_query" not in st.session_state:
        st.session_state.selected_docs_for_query = []
    if "documents" not in st.session_state:
        st.session_state.documents = {}
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None

    render_sidebar()

    # Document upload section
    with st.sidebar:
        st.markdown("---")
        st.header("📄 Documents")
        uploaded_files = st.file_uploader(
            "Upload one or more PDFs",
            type=["pdf"],
            accept_multiple_files=True
        )

        if uploaded_files and st.button("🚀 Build Document Trees", type="primary", use_container_width=True):
            with st.spinner("Initializing LLM and indexing documents..."):
                try:
                    # Initialize LLM
                    llm = get_cached_llm(
                        st.session_state.llm_model_choice,
                        st.session_state.get("use_4bit", True)
                    )

                    # Initialize pipeline
                    pipeline = RAGPipeline(
                        llm,
                        max_context_chars=st.session_state.get("max_context_chars", 15000)
                    )
                    st.session_state.pipeline = pipeline
                    st.success(f"✅ LLM loaded: {llm.model_name} via {llm.backend_type}")

                    # Index documents
                    indexer = DocumentIndexer(
                        llm,
                        chunk_size=st.session_state.get("chunk_size", 5)
                    )

                    progress_bar = st.progress(0)
                    for idx, _ in enumerate(uploaded_files):
                        progress_bar.progress((idx + 1) / len(uploaded_files))

                    documents = indexer.index(uploaded_files)
                    st.session_state.documents = documents
                    st.session_state.messages = []
                    progress_bar.empty()

                    st.success(f"✅ Indexed {len(uploaded_files)} document(s) with {sum(len(d.sections) for d in documents.values())} sections")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.error(f"Initialization error: {e}", exc_info=True)

        # Document selector and tree viewer
        if st.session_state.documents:
            st.markdown("---")
            st.header("🎯 Query Scope")
            doc_names = list(st.session_state.documents.keys())
            st.session_state.selected_docs_for_query = st.multiselect(
                "Select documents to search:",
                doc_names,
                default=doc_names
            )

            with st.expander("🌳 View Document Trees", expanded=False):
                for doc_name in doc_names:
                    st.subheader(f"📄 {doc_name}")
                    doc = st.session_state.documents[doc_name]
                    for sec in doc.sections:
                        st.markdown(f"**{sec.id}: {sec.title}** *(pp. {sec.page_range})*")
                        st.caption(sec.summary)
                        st.divider()

    # Main chat area
    if not st.session_state.documents:
        st.info("👈 Please upload PDF(s) in the sidebar and click 'Build Document Trees' to start.")

        with st.expander("📖 Quick Start Guide", expanded=True):
            st.markdown("""
            ### Getting Started

            1. **Choose your LLM backend** in the sidebar:
               - **Ollama** (recommended): Fast API backend. Install from [ollama.com](https://ollama.com)
               - **HuggingFace**: Local model loading. Requires more RAM/VRAM.

            2. **Upload PDFs** using the sidebar uploader

            3. **Click 'Build Document Trees'** to index your documents

            4. **Ask questions** in the chat below

            ### Architecture
            The system uses a 5-phase pipeline:
            1. **Intent Router** — classifies what you want (values, equations, mechanisms)
            2. **Tree Navigator** — agentically drills down to relevant sections
            3. **Structured Extractor** — pulls quantitative data into strict JSON
            4. **Citation Validator** — cross-checks against raw PDF text to kill hallucinations
            5. **Adaptive Synthesizer** — formats output as tables, LaTeX, or causal explanations

            ### Requirements
            - **PyMuPDF**: `pip install pymupdf`
            - **For Ollama**: `pip install ollama` + [Ollama app](https://ollama.com)
            - **For HF**: `pip install transformers torch`
            - **Optional**: `pip install rapidfuzz` (better hallucination detection)
            """)
    else:
        st.subheader("💬 Chat with your Documents")

        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "pipeline_result" in msg:
                    result = msg["pipeline_result"]
                    with st.expander("🧠 Pipeline Trace & Verified Data"):
                        st.markdown(f"**Detected Intent:** `{result.get('intent', 'unknown')}`")
                        st.markdown(f"**Overall Confidence:** {result.get('confidence', 0):.2f}")
                        st.markdown(f"**Navigation Trace:**")
                        st.markdown(result.get("thinking", "No trace available."))

                        items = result.get("items", [])
                        if items:
                            st.markdown("**Verified Extractions (Hallucination-Checked):**")
                            for item in items[:10]:
                                conf = item.confidence
                                color = "🟢" if conf > 0.8 else "🟡" if conf > 0.6 else "🔴"
                                if item.item_type == "equation":
                                    st.caption(
                                        f"{color} **Equation**: `{item.equation_latex}` | "
                                        f"Conf: {conf:.2f} | [{item.doc_name}, p.{item.page}]"
                                    )
                                else:
                                    st.caption(
                                        f"{color} **{item.parameter_name}**: {item.value} {item.unit} | "
                                        f"Conf: {conf:.2f} | [{item.doc_name}, p.{item.page}]"
                                    )

        # Chat input
        if prompt := st.chat_input("Ask a question about the documents..."):
            if not st.session_state.selected_docs_for_query:
                st.error("Please select at least one document in the sidebar to search.")
            else:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Running pipeline: Intent → Navigate → Extract → Validate → Synthesize..."):
                        try:
                            # Filter documents
                            docs_to_search = {
                                k: st.session_state.documents[k]
                                for k in st.session_state.selected_docs_for_query
                            }

                            # Get pipeline
                            pipeline = st.session_state.pipeline
                            if not pipeline:
                                llm = get_cached_llm(
                                    st.session_state.llm_model_choice,
                                    st.session_state.get("use_4bit", True)
                                )
                                pipeline = RAGPipeline(
                                    llm,
                                    max_context_chars=st.session_state.get("max_context_chars", 15000)
                                )
                                st.session_state.pipeline = pipeline

                            # Run pipeline
                            result = pipeline.run(prompt, docs_to_search)

                            st.markdown(result["answer"])

                            with st.expander("🧠 Pipeline Trace & Verified Data"):
                                st.markdown(f"**Detected Intent:** `{result['intent']}`")
                                st.markdown(f"**Overall Confidence:** {result['confidence']:.2f}")
                                st.markdown(f"**Navigation Trace:**")
                                st.markdown(result["thinking"])

                                items = result.get("items", [])
                                if items:
                                    st.markdown("**Verified Extractions (Hallucination-Checked):**")
                                    for item in items[:10]:
                                        conf = item.confidence
                                        color = "🟢" if conf > 0.8 else "🟡" if conf > 0.6 else "🔴"
                                        if item.item_type == "equation":
                                            st.caption(
                                                f"{color} **Equation**: `{item.equation_latex}` | "
                                                f"Conf: {conf:.2f} | [{item.doc_name}, p.{item.page}]"
                                            )
                                        else:
                                            st.caption(
                                                f"{color} **{item.parameter_name}**: {item.value} {item.unit} | "
                                                f"Conf: {conf:.2f} | [{item.doc_name}, p.{item.page}]"
                                            )

                            # Save to history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": result["answer"],
                                "pipeline_result": result
                            })

                        except Exception as e:
                            error_msg = f"❌ Pipeline error: {str(e)}"
                            st.error(error_msg)
                            logger.error(f"Query error: {e}", exc_info=True)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": error_msg
                            })


if __name__ == "__main__":
    run_streamlit()
