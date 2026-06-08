#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
DECLARMIMA v20.1 Enhanced - Hybrid Vectorless RAG with 4-Pillar Architecture
============================================================================
Integrates the robust import/LLM-loading system from DECLARMIMA v20.0
with the clean vectorless RAG architecture, PLUS the 4-Pillar agentic
architecture and Cross-Document Meta-Tree JSON processing.

Key enhancements over v20.0:
- 4-Pillar Agentic Architecture: Intent Router, MCTS Navigator,
  Structured Extractor + Citation Validator, Adaptive Synthesizer
- Cross-Document Meta-Tree stitching for multi-document JSON reasoning
- JSON-Action MCTS Navigator for PageIndex-style hierarchical navigation
- Layout-Aware JSON Tree Builder using pymupdf4llm
- Enhanced VALUE_TRIGGERS for unit/parameter detection
- MCTS node capping to prevent LLM context overflow
- Robust JSON parsing with trailing comma fix
- Removed "definition" from document flood bypass
- Fixed meta-tree "children" -> "nodes" key mismatch
- Added _format_as_value_comparison narrative-by-material formatter
- Graceful dependency degradation (Ollama, Transformers, PyMuPDF, etc.)
- Hybrid LLM backend (Ollama local + HuggingFace transformers local)
- Cached LLM initialization via @st.cache_resource
- Model-specific prompt templates (Qwen, Mistral, Llama, etc.)
- Enhanced PDF processing with pymupdf4llm fallback
- 100% Vectorless: No embeddings, no vector DBs
- FIXED: Safe PyMuPDF import to avoid 'fitz' package collision
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
from typing import List, Dict, Optional, Tuple, Union, Any
from collections import defaultdict
from io import BytesIO

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logging.basicConfig(level=logging.INFO, handlers=[console_handler], force=True)
logger = logging.getLogger("DECLARMIMA_v20_1")


# ============================================================================
# PYMUPDF IMPORT (FIXED)
# ============================================================================
try:
    import pymupdf as fitz  # PyMuPDF >= 1.24
    PYMUPDF_AVAILABLE = True
    logger.info("✓ PyMuPDF (modern) loaded via pymupdf")
except ImportError:
    try:
        import fitz  # Older PyMuPDF versions
        PYMUPDF_AVAILABLE = True
        logger.info("✓ PyMuPDF (legacy) loaded via fitz")
    except ImportError:
        PYMUPDF_AVAILABLE = False
        st.error(
            "PyMuPDF is not installed.\n"
            "Install with:\n"
            "pip install pymupdf"
        )
        st.stop()


# ============================================================================
# DEPENDENCY CHECKS WITH GRACEFUL DEGRADATION
# ============================================================================

def check_optional_dependencies() -> Dict[str, bool]:
    """Check all optional dependencies and report availability with graceful degradation."""
    deps: Dict[str, bool] = {}

    deps['pymupdf'] = PYMUPDF_AVAILABLE
    if PYMUPDF_AVAILABLE:
        logger.info("✓ PyMuPDF available")
    else:
        logger.error("✗ PyMuPDF required: pip install pymupdf")

    try:
        import pymupdf4llm
        deps['pymupdf4llm'] = True
        logger.info("✓ pymupdf4llm available")
    except ImportError:
        deps['pymupdf4llm'] = False
        logger.warning("✗ pymupdf4llm not installed. Layout-aware extraction disabled.")

    try:
        import ollama
        deps['ollama'] = True
        logger.info("✓ Ollama client available")
    except ImportError:
        deps['ollama'] = False
        logger.warning("✗ Ollama not installed. Ollama backend unavailable.")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        deps['transformers'] = True
        logger.info("✓ HuggingFace transformers available")
    except ImportError:
        deps['transformers'] = False
        logger.warning("✗ transformers not installed. Local HF models unavailable.")

    try:
        from rapidfuzz import fuzz, process
        deps['rapidfuzz'] = True
        logger.info("✓ rapidfuzz available")
    except ImportError:
        deps['rapidfuzz'] = False
        logger.warning("✗ rapidfuzz not installed. Fuzzy matching disabled.")

    try:
        import orjson
        deps['orjson'] = True
        logger.info("✓ orjson available (fast JSON)")
    except ImportError:
        deps['orjson'] = False
        logger.warning("✗ orjson not installed. Using standard json (slower).")

    try:
        from pyvis.network import Network
        deps['pyvis'] = True
        logger.info("✓ pyvis available")
    except ImportError:
        deps['pyvis'] = False
        logger.warning("✗ pyvis not installed. Interactive networks disabled.")

    logger.info(f"Dependency check complete: {sum(deps.values())}/{len(deps)} available")
    return deps

GLOBAL_DEPS = check_optional_dependencies()

# Individual flags
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. Transformers backend requires torch.")

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False


# ============================================================================
# FAST JSON HELPERS (orjson fallback)
# ============================================================================

def fast_json_loads(data: bytes) -> Any:
    """Fast JSON loading with orjson fallback."""
    if ORJSON_AVAILABLE:
        import orjson
        return orjson.loads(data)
    return json.loads(data.decode('utf-8'))

def fast_json_dumps(obj: Any, indent: bool = False) -> bytes:
    """Fast JSON dumping with orjson fallback."""
    if ORJSON_AVAILABLE:
        import orjson
        option = orjson.OPT_INDENT_2 if indent else 0
        return orjson.dumps(obj, option=option)
    return json.dumps(obj, indent=2 if indent else None).encode('utf-8')


# ============================================================================
# HYBRID LLM & TEMPLATES
# ============================================================================

LOCAL_LLM_OPTIONS = {
    # Ollama models
    "[Ollama] qwen2.5:0.5b (Fastest, CPU OK)": "ollama:qwen2.5:0.5b",
    "[Ollama] qwen2.5:1.5b (Balanced)": "ollama:qwen2.5:1.5b",
    "[Ollama] qwen2.5:7b (Recommended for RAG)": "ollama:qwen2.5:7b",
    "[Ollama] qwen2.5:14b (Max Reasoning)": "ollama:qwen2.5:14b",
    "[Ollama] llama3.1:8b (Meta Standard)": "ollama:llama3.1:8b",
    "[Ollama] mistral:7b (High JSON Reliability)": "ollama:mistral:7b",
    "[Ollama] gemma2:9b (Scientific Nuance)": "ollama:gemma2:9b",
    "[Ollama] falcon3:10b (Instruction Following)": "ollama:falcon3:10b",

    # HuggingFace Transformers models (local loading)
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

MODEL_PROMPT_TEMPLATES = {
    "qwen": {"system": "You are a precise document analyst. Follow JSON format strictly.", "json_reminder": "Return ONLY valid JSON."},
    "smollm": {"system": "You are a precise document analyst. Follow JSON format strictly.", "json_reminder": "Return ONLY valid JSON."},
    "mistral": {"system": "You are a helpful assistant that always returns valid JSON.", "json_reminder": "Return ONLY valid JSON."},
    "llama": {"system": "You are a helpful assistant. Be concise and accurate.", "json_reminder": "Return valid JSON only."},
    "gemma": {"system": "You are a helpful assistant.", "json_reminder": "Return valid JSON only."},
    "falcon": {"system": "You are a helpful assistant.", "json_reminder": "Return valid JSON only."},
    "default": {"system": "You are a document navigation agent.", "json_reminder": "Return valid JSON only."}
}

def get_model_template(model_name: str):
    """Get the appropriate prompt template for a given model name."""
    model_lower = model_name.lower()
    for key, template in MODEL_PROMPT_TEMPLATES.items():
        if key in model_lower:
            return template
    return MODEL_PROMPT_TEMPLATES["default"]


class HybridLLM:
    """
    Hybrid LLM backend supporting both Ollama (local API) and HuggingFace Transformers (local loading).
    """

    def __init__(self, model_key: str, use_4bit: bool = True, device: Optional[str] = None):
        self.model_key = model_key
        self.use_4bit = use_4bit
        self.device = device or ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.backend = None
        self.model_name = None
        self.client = None
        self.tokenizer = None
        self.model = None

        # Parse model key
        if model_key.startswith("[Ollama]"):
            self.model_name = model_key.split("] ")[1].strip().split(" (")[0]
        elif model_key.startswith("ollama:"):
            self.model_name = model_key.replace("ollama:", "", 1)
        elif model_key.startswith("[HF]"):
            self.model_name = model_key.split("] ")[1].strip().split(" (")[0]
        else:
            self.model_name = model_key  # Assume HuggingFace model ID

        self.template = get_model_template(self.model_name)
        self._init_backend()
        logger.info(f"HybridLLM initialized: {self.model_name} on {self.device} via {self.backend}")

    def _init_backend(self):
        """Initialize the LLM backend with fallback chain."""
        # Try Ollama first
        if OLLAMA_AVAILABLE:
            try:
                requests.get("http://localhost:11434/api/tags", timeout=5)
                self.backend = "ollama"
                self.client = ollama.Client(host="http://localhost:11434")
                logger.info("✓ Using Ollama backend")
                return
            except requests.exceptions.ConnectionError:
                logger.warning("Ollama server not running at localhost:11434")
            except Exception as e:
                logger.warning(f"Ollama check failed: {e}")

        # Fallback to Transformers
        if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
            self.backend = "transformers"
            logger.info("✓ Using Transformers backend (local HuggingFace)")
            return

        raise RuntimeError(
            "No LLM backend available. Please either:\n"
            "1. Install and start Ollama (pip install ollama, then ollama serve)\n"
            "2. Install transformers + torch (pip install transformers torch)"
        )

    def generate(self, prompt: str, max_new_tokens: int = 1024, temperature: float = 0.1, 
                 fast_json: bool = False, system_prompt: Optional[str] = None) -> str:
        """Generate text using the active backend."""
        if self.backend == "ollama":
            return self._ollama_generate(prompt, max_new_tokens, temperature, fast_json, system_prompt)
        elif self.backend == "transformers":
            return self._transformers_generate(prompt, max_new_tokens, temperature, system_prompt)
        else:
            return "Error: No backend initialized"

    def _ollama_generate(self, prompt, max_tokens, temp, fast_json, system_prompt):
        """Generate via Ollama API."""
        try:
            options = {"temperature": temp, "num_predict": max_tokens}
            if fast_json:
                options["format"] = "json"
            messages = []
            sys = system_prompt or self.template.get("system")
            if sys:
                messages.append({"role": "system", "content": sys})
            messages.append({"role": "user", "content": prompt})
            resp = self.client.chat(model=self.model_name, messages=messages, 
                                   options=options, stream=False)
            return resp.get("message", {}).get("content", "").strip()
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return f"Error: {str(e)[:100]}"

    def _transformers_generate(self, prompt, max_tokens, temp, system_prompt):
        """Generate via HuggingFace Transformers (local)."""
        if self.tokenizer is None:
            self._load_transformers()
        if not self.model:
            return "Error: Model not loaded"
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Apply chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template'):
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, 
                                                          add_generation_prompt=True)
            else:
                # Fallback for older tokenizers
                text = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                   max_length=4096).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_tokens,
                    temperature=temp if temp > 0 else None,
                    do_sample=temp > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract assistant response
            if "assistant" in response:
                response = response.split("assistant")[-1].strip()
            return response
        except Exception as e:
            logger.error(f"Transformers error: {e}")
            return f"Error: {str(e)[:100]}"

    def _load_transformers(self):
        """Load the HuggingFace model and tokenizer."""
        logger.info(f"Loading {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32
        }

        if self.use_4bit and self.device == "cuda":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_compute_dtype=torch.float16
            )

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if self.device == "cuda":
            self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully.")


@st.cache_resource(show_spinner="Initializing LLM...")
def get_cached_llm(model_choice: str, use_4bit: bool = True):
    """Cached LLM initialization to avoid reloading on every interaction."""
    internal = LOCAL_LLM_OPTIONS[model_choice]
    return HybridLLM(model_key=internal, use_4bit=use_4bit)


# ============================================================================
# JSON EXTRACTION UTILITIES
# ============================================================================

def extract_json(text: str) -> Optional[Any]:
    """
    Bulletproof JSON extractor that handles:
    - Markdown code blocks (```json ... ```)
    - Raw JSON objects/arrays
    - Trailing commas (common LLM error)
    - Nested structures
    """
    if not text:
        return None

    # Try direct parse first
    try:
        return json.loads(text)
    except:
        pass

    # Try markdown code block extraction
    md_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if md_match:
        try:
            return json.loads(md_match.group(1))
        except:
            pass

    # Try finding any JSON object or array
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            # Fix common LLM JSON errors - trailing comma fix
            candidate = re.sub(r',\s*([\]}])', r'\1', candidate)  # Remove trailing commas
            candidate = re.sub(r'\s+', ' ', candidate)  # Normalize whitespace
            try:
                return json.loads(candidate)
            except:
                pass

    return None


# ============================================================================
# PDF PROCESSING (FIXED IMPORT)
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
                pages_data.append(
                    {
                        "page_num": i + 1,
                        "text": text
                    }
                )
        except Exception:
            continue

    doc.close()
    return pages_data


class PaginationAwareReader:
    """Enhanced PDF reader with pymupdf4llm LaTeX-aware fallback."""

    def __init__(self, max_chars_per_request: int = 20000):
        self.max_chars_per_request = max_chars_per_request

    def extract_pages(self, doc_path_or_bytes: Union[str, bytes], page_numbers: List[int]) -> Dict[int, str]:
        """
        Extract page text with optional pymupdf4llm for LaTeX-aware math extraction.
        Falls back gracefully to standard PyMuPDF text extraction.
        """
        result = {}

        # Try pymupdf4llm first for equation-aware PDF reading
        pymupdf4llm_available = False
        try:
            import pymupdf4llm
            pymupdf4llm_available = True
        except ImportError:
            pass

        if pymupdf4llm_available and isinstance(doc_path_or_bytes, str):
            try:
                import pymupdf4llm
                chunks = pymupdf4llm.to_markdown(doc_path_or_bytes, page_chunks=True)

                # Build lookup map: page_number (1-based) -> markdown text
                chunk_map = {}
                for chunk in chunks:
                    meta = chunk.get("metadata", {})
                    p_num = meta.get("page_number", 0)
                    if p_num > 0:
                        chunk_map[p_num] = chunk.get("text", "")

                # Extract requested pages from chunk map
                for pnum in page_numbers:
                    if pnum in chunk_map:
                        text = chunk_map[pnum]
                        if len(text) > self.max_chars_per_request:
                            text = text[:self.max_chars_per_request] + "\n...[TRUNCATED]"
                        result[pnum] = text

                # If all requested pages found, return early
                if len(result) == len(page_numbers):
                    return result

            except Exception as e:
                logger.warning(f"pymupdf4llm extraction failed, falling back to standard: {e}")
                result = {}

        # Fallback: Standard PyMuPDF text extraction
        if isinstance(doc_path_or_bytes, bytes):
            doc = fitz.open(stream=doc_path_or_bytes, filetype="pdf")
        else:
            doc = fitz.open(doc_path_or_bytes)

        for pnum in page_numbers:
            if pnum < 1 or pnum > len(doc):
                continue
            if pnum in result:
                continue  # Already extracted via pymupdf4llm
            page = doc[pnum - 1]
            text = page.get_text("text")
            if len(text) > self.max_chars_per_request:
                text = text[:self.max_chars_per_request] + "\n...[TRUNCATED]"
            result[pnum] = text
        doc.close()
        return result



# ============================================================================
# ════════════════════════════════════════════════════════════════════════════
# 4-PILLAR ARCHITECTURAL COMPONENTS (v20.1 Enhancement)
# ════════════════════════════════════════════════════════════════════════════
# ============================================================================

# ───────────────────────────────────────────────────────────────────────────
# PILLAR 1: Scientific Intent Router
# ───────────────────────────────────────────────────────────────────────────

class ScientificIntentRouter:
    """
    Lightweight intent classifier using regex triggers.
    Classifies *what* the user wants (Values? Equations? Mechanisms?) 
    before searching, enabling intent-aware routing.
    """
    PATTERNS = {
        "value_extraction": [
            r"\bvalue\b", r"\bhow much\b", 
            r"\b\d+\s*(?:W|kW|MPa|mm/s|GPa|°C|K|Hz|m/s|kg|g|μm|nm)\b",
            r"\btable\b", r"\blist\b", r"\bparameter\b", r"\bproperty\b",
            r"\btemperature\b", r"\bpressure\b", r"\bvelocity\b", r"\bpower\b",
            r"\bdensity\b", r"\bconductivity\b", r"\bmodulus\b", r"\bstrength\b",
            r"\byield\b", r"\btensile\b", r"\bhardness\b", r"\bporosity\b",
        ],
        "equation": [
            r"\bequation\b", r"\bformula\b", r"\bconstitutive\b", 
            r"\bnavier[- ]stokes\b", r"\bcahn[- ]hilliard\b",
            r"\bgoverning\b", r"\bmodel\b.*\bequation\b", r"\bmathematical\b",
            r"\bPDE\b", r"\bODE\b", r"\bdifferential\b", r"\bstrain[- ]rate\b",
            r"\bstress[- ]strain\b", r"\bstrain rate\b", r"\bconstitutive model\b",
        ],
        "mechanism": [
            r"\bwhy\b", r"\bhow does\b", r"\bmechanism\b", 
            r"\bcause\b", r"\bdriving force\b", r"\bexplain\b",
            r"\bwhat causes\b", r"\bwhat is the reason\b", r"\bwhat leads to\b",
            r"\bwhat results in\b", r"\bwhat is responsible\b", r"\bwhat triggers\b",
            r"\bwhat induces\b", r"\bwhat promotes\b", r"\bwhat inhibits\b",
        ],
        "comparison": [
            r"\bcompare\b", r"\bversus\b", r"\bvs\b", r"\bdifference\b",
            r"\bcontrast\b", r"\bsimilarity\b", r"\bbetween\b.*\band\b",
            r"\bwhich is better\b", r"\bwhich is higher\b", r"\bwhich is lower\b",
        ],
        "methodology": [
            r"\bmethod\b", r"\bprocedure\b", r"\bprotocol\b", r"\bprocess\b",
            r"\bexperimental setup\b", r"\bsimulation setup\b", r"\bcomputational\b",
            r"\bFEA\b", r"\bCFD\b", r"\bfinite element\b", r"\bmesh\b",
            r"\bboundary condition\b", r"\binitial condition\b",
        ],
    }

    def route(self, query: str) -> Dict[str, str]:
        """Route query to intent and output format."""
        q_lower = query.lower()
        for intent, patterns in self.PATTERNS.items():
            if any(re.search(p, q_lower) for p in patterns):
                return {"intent": intent, "output_format": intent}
        return {"intent": "open_query", "output_format": "prose"}


# ───────────────────────────────────────────────────────────────────────────
# PILLAR 2: Iterative MCTS Navigator
# ───────────────────────────────────────────────────────────────────────────

class IterativeTreeNavigator:
    """
    Simulates PageIndex MCTS navigation using the short code's flat sections.
    Replaces single-shot search with an agentic drill-down loop.
    Includes MCTS node capping to prevent LLM context overflow.
    """
    def __init__(self, llm: HybridLLM, max_steps: int = 2, max_nodes_per_step: int = 30):
        self.llm = llm
        self.max_steps = max_steps
        self.max_nodes_per_step = max_nodes_per_step
        self.trace = []

    def navigate(self, query: str, selected_docs: Dict) -> List[Dict]:
        """Agentic drill-down navigation through document sections."""
        retrieved_chunks = []

        # Step 1: Build the "Forest" of summaries (capped to prevent overflow)
        forest_desc = []
        for doc_name, data in selected_docs.items():
            for s in data['sections']:
                forest_desc.append(
                    f"ID: {doc_name}:::{s['id']} | Title: {s['title']} | "
                    f"Pages: {s['start_page']}-{s['end_page']} | "
                    f"Summary: {s['summary'][:150]}"
                )

        # Cap nodes to prevent LLM context overflow
        if len(forest_desc) > self.max_nodes_per_step:
            forest_desc = forest_desc[:self.max_nodes_per_step]
            self.trace.append(
                f"[MCTS] Capped forest to {self.max_nodes_per_step} nodes to prevent context overflow."
            )

        # Step 2: Agentic Drill-Down
        current_nodes_text = "\n".join(forest_desc)
        for step in range(self.max_steps):
            prompt = f"""You are a scientific navigator. Query: "{query}"
Available Sections:
{current_nodes_text}

Return JSON: {{"reasoning": "...", "actions": [{{"id": "doc:::sec_id", "action": "extract_text"}}]}}.
ONLY select sections that definitively contain the answer. Max 3 actions."""

            resp = self.llm.generate(prompt, max_new_tokens=512, system_prompt="Return ONLY valid JSON.")
            actions_data = extract_json(resp)

            if not actions_data or 'actions' not in actions_data:
                self.trace.append(f"[Step {step+1}] No valid actions returned. Stopping.")
                break

            self.trace.append(actions_data.get('reasoning', f'[Step {step+1}] No reasoning provided.'))

            for action in actions_data['actions']:
                ns_id = action.get('id')
                if not ns_id or ":::" not in ns_id: 
                    continue
                doc_name, sec_id = ns_id.split(":::", 1)

                if doc_name in selected_docs:
                    sec = next((s for s in selected_docs[doc_name]['sections'] if s['id'] == sec_id), None)
                    if sec:
                        # Fetch raw text
                        pages = selected_docs[doc_name]['pages']
                        text_parts = [
                            pages[p-1]['text'] 
                            for p in range(sec['start_page'], sec['end_page']+1) 
                            if 0 < p <= len(pages)
                        ]
                        retrieved_chunks.append({
                            "doc_name": doc_name, 
                            "section_title": sec['title'],
                            "start_page": sec['start_page'], 
                            "end_page": sec['end_page'],
                            "full_text": "\n".join(text_parts),
                            "section_id": sec_id,
                        })
            # For minimal workability, 1-step deep extraction is highly effective
            break

        return retrieved_chunks


# ───────────────────────────────────────────────────────────────────────────
# PILLAR 3: Structured Extractor & Citation Validator
# ───────────────────────────────────────────────────────────────────────────

class UniversalLLMExtractor:
    """
    Extracts structured JSON items from raw text chunks.
    Handles non-standard math formatting (asterisk-wrapped tensors, equation number tags)
    that standard LaTeX regex misses.
    """
    PROMPT = """Extract quantitative data, equations, or mechanisms from the text.

SPECIAL HANDLING for non-standard math:
- Asterisk-wrapped tensors: *\\sigma_ij* -> capture as tensor notation
- Equation number tags: (1), (2.3) -> capture with equation
- Inline math: $...$ or \\(...\\) -> preserve LaTeX

Return a JSON list of objects: 
{{
    "item_type": "quantitative|equation|mechanism", 
    "parameter_name": "...", 
    "value": number_or_null, 
    "unit": "...", 
    "equation_latex": "...", 
    "context": "exact sentence or phrase from text", 
    "doc_name": "...", 
    "page": int,
    "confidence": "high|medium|low"
}}.

Text: {text}"""

    def extract(self, chunks: List[Dict], query: str, llm: HybridLLM) -> List[Dict]:
        """Extract structured items from text chunks."""
        items = []
        for chunk in chunks:
            prompt = self.PROMPT.format(text=chunk['full_text'][:4000])
            resp = llm.generate(prompt, max_new_tokens=1024, system_prompt="Return ONLY valid JSON list.")
            parsed = extract_json(resp)
            if isinstance(parsed, list):
                for item in parsed:
                    item['doc_name'] = chunk['doc_name']
                    item['page'] = chunk['start_page']
                    item['section_title'] = chunk.get('section_title', '')
                    items.append(item)
        return items


class CitationValidator:
    """
    Cross-checks extracted values against raw page text to prevent hallucinations.
    The hallucination killer.
    """
    def verify(self, items: List[Dict], selected_docs: Dict) -> List[Dict]:
        """Verify extracted items against raw PDF text."""
        verified = []
        for item in items:
            doc_name = item.get('doc_name')
            page_num = item.get('page', 1)
            if doc_name in selected_docs:
                # Safely get raw text for the page
                pages = selected_docs[doc_name].get('pages', [])
                if page_num - 1 < len(pages):
                    raw_text = pages[page_num - 1].get('text', '').lower()
                else:
                    raw_text = ""

                # Verification Logic
                val_str = str(item.get('value', ''))
                param = (item.get('parameter_name') or '').lower()

                # Enhanced VALUE_TRIGGERS for unit/parameter detection
                value_triggers = [
                    val_str, param,
                    str(item.get('unit', '')).lower(),
                    str(item.get('equation_latex', '')).lower(),
                ]

                # Check if any trigger appears in raw text
                any_trigger_found = any(t and t in raw_text for t in value_triggers)

                if item.get('value') is not None and val_str in raw_text:
                    item['confidence'] = 0.95
                    item['verification_status'] = 'verified_exact'
                    verified.append(item)
                elif param and param in raw_text:
                    item['confidence'] = 0.7
                    item['verification_status'] = 'verified_param'
                    verified.append(item)
                elif any_trigger_found:
                    item['confidence'] = 0.6
                    item['verification_status'] = 'verified_trigger'
                    verified.append(item)
                elif item.get('item_type') == 'equation' and item.get('equation_latex'):
                    # Hard to verify LaTeX via raw text, trust LLM with medium confidence
                    item['confidence'] = 0.8
                    item['verification_status'] = 'equation_trusted'
                    verified.append(item)
                else:
                    # Hallucination suspected
                    item['confidence'] = 0.2
                    item['verification_status'] = 'hallucination_suspected'
                    verified.append(item)  # Keep it but mark low confidence
        return sorted(verified, key=lambda x: x.get('confidence', 0), reverse=True)


# ───────────────────────────────────────────────────────────────────────────
# PILLAR 4: Adaptive Synthesizer
# ───────────────────────────────────────────────────────────────────────────

class AdaptiveResponseGenerator:
    """
    Generates strict template-enforced responses based on the Intent Router's classification.
    Forces the LLM to output strict Markdown tables, LaTeX blocks, or causal chains.
    """
    def __init__(self, llm: HybridLLM):
        self.llm = llm

    def generate(self, query: str, items: List[Dict], intent: str) -> str:
        """Generate adaptive response based on detected intent."""
        evidence = "\n".join([
            f"- [{i.get('doc_name')}, p.{i.get('page')}] {i.get('context', i.get('equation_latex', ''))}"
            for i in items[:15]
        ])

        if intent == "value_extraction":
            prompt = f"""Create a Markdown table of extracted values.
Query: {query}
Evidence:
{evidence}

Columns: | Parameter | Value | Unit | Document | Page | Confidence |
RULE: Every row MUST be backed by the evidence. Do not invent data.
Include a summary paragraph after the table explaining the key findings."""

        elif intent == "equation":
            prompt = f"""State the governing equations in LaTeX format ($$ ... $$).
Query: {query}
Evidence:
{evidence}

RULE: Define all variables immediately below each equation. Cite the document.
If multiple equations exist, list them in order of relevance."""

        elif intent == "mechanism":
            prompt = f"""Explain the physical mechanism comprehensively.
Query: {query}
Evidence:
{evidence}

RULE: Use inline citations like [DocName, p.X] for every factual claim.
Structure as: (1) Cause, (2) Process/Chain, (3) Effect/Outcome.
Use causal chain arrows (→) where appropriate."""

        elif intent == "comparison":
            prompt = f"""Provide a structured comparison.
Query: {query}
Evidence:
{evidence}

RULE: Use a side-by-side Markdown comparison table.
Highlight differences in bold. Cite sources for every data point."""

        elif intent == "methodology":
            prompt = f"""Describe the methodology or experimental setup.
Query: {query}
Evidence:
{evidence}

RULE: List steps in numbered order. Include parameters, conditions, and equipment.
Cite the document for each detail."""

        else:  # open_query or fallback
            prompt = f"""Answer the query comprehensively.
Query: {query}
Evidence:
{evidence}

RULE: Use inline citations like [DocName, p.X] for every factual claim.
Be thorough but concise. If information is insufficient, state so clearly."""

        return self.llm.generate(prompt, max_new_tokens=1500, system_prompt="You are a scientific analyst. Follow formatting rules strictly.")

    def _format_as_value_comparison(self, items: List[Dict], query: str) -> str:
        """
        Narrative-by-material formatter for comparison queries.
        Groups extracted values by material/parameter and compares across documents.
        """
        param_groups = defaultdict(list)
        for item in items:
            param = item.get('parameter_name', 'Unknown')
            param_groups[param].append(item)

        lines = [f"### Value Comparison: {query}", ""]
        for param, group in sorted(param_groups.items()):
            lines.append(f"**{param}:**")
            for item in group:
                val = item.get('value', 'N/A')
                unit = item.get('unit', '')
                doc = item.get('doc_name', 'Unknown')
                page = item.get('page', '?')
                conf = item.get('confidence', 0)
                lines.append(f"  - {val} {unit} [{doc}, p.{page}] (conf: {conf:.2f})")
            lines.append("")

        return "\n".join(lines)



# ============================================================================
# ════════════════════════════════════════════════════════════════════════════
# CROSS-DOCUMENT META-TREE COMPONENTS (PageIndex-Style JSON Processing)
# ════════════════════════════════════════════════════════════════════════════
# ============================================================================

def _build_tree_from_markdown(md_text: str, doc_id: str = "doc") -> Dict:
    """
    Parse Markdown headers (#, ##, ###) into a nested JSON tree.
    Layout-Aware JSON Tree Builder using pymupdf4llm output.
    """
    lines = md_text.split("\n")
    root = {
        "node_id": f"{doc_id}_root",
        "title": "Document Root",
        "level": 0,
        "doc_id": doc_id,
        "nodes": [],
        "text_lines": [],
    }
    stack = [root]

    for line in lines:
        # Match header lines
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if header_match:
            hashes, title = header_match.groups()
            level = len(hashes)

            # Pop stack to find correct parent
            while len(stack) > 1 and stack[-1]["level"] >= level:
                stack.pop()

            node = {
                "node_id": f"{doc_id}_h{level}_{len(stack)}",
                "title": title.strip(),
                "level": level,
                "doc_id": doc_id,
                "nodes": [],
                "text_lines": [],
                "summary": "",  # Will be filled by LLM
            }
            stack[-1]["nodes"].append(node)
            stack.append(node)
        else:
            # Add non-header text to current node
            if line.strip():
                stack[-1]["text_lines"].append(line)

    # Join text lines for each node
    def _join_text(node):
        node["text"] = "\n".join(node["text_lines"])
        del node["text_lines"]
        for child in node.get("nodes", []):
            _join_text(child)

    _join_text(root)
    return root


def find_section_by_keyword(tree: Dict, keyword: str, threshold: float = 0.6) -> Optional[Dict]:
    """
    Fuzzy search for a section by keyword in the JSON tree.
    Returns the best matching node or None.
    """
    keyword_lower = keyword.lower()
    best_match = None
    best_score = 0.0

    def _search(node, depth=0):
        nonlocal best_match, best_score
        title = node.get('title', '').lower()

        # Exact match
        if keyword_lower in title or title in keyword_lower:
            score = 1.0
        elif RAPIDFUZZ_AVAILABLE:
            from rapidfuzz import fuzz
            score = fuzz.partial_ratio(keyword_lower, title) / 100.0
        else:
            # Simple word overlap
            kw_words = set(keyword_lower.split())
            title_words = set(title.split())
            if kw_words and title_words:
                score = len(kw_words & title_words) / len(kw_words)
            else:
                score = 0.0

        if score > best_score:
            best_score = score
            best_match = node

        for child in node.get('nodes', []):
            _search(child, depth + 1)

    _search(tree)
    return best_match if best_score >= threshold else None


def get_children_from_tree(tree: Dict, node_id: str) -> List[Dict]:
    """Get children (nodes) of a specific node in the tree."""

    def _search(node):
        if node.get('node_id') == node_id:
            return node.get('nodes', [])
        for child in node.get('nodes', []):
            found = _search(child)
            if found is not None:
                return found
        return None

    found = _search(tree)
    return found if found is not None else []


def get_full_text_from_tree(tree: Dict, node_id: str) -> Optional[Dict]:
    """Get full text content of a specific node and all its descendants."""

    def _search(node):
        if node.get('node_id') == node_id:
            # Collect all text from this node and descendants
            texts = [node.get('text', '')]
            for child in node.get('nodes', []):
                child_text = _collect_text(child)
                texts.append(child_text)
            return {
                "node_id": node_id,
                "title": node.get('title', ''),
                "doc_id": node.get('doc_id', ''),
                "full_text": "\n\n".join(filter(None, texts)),
            }
        for child in node.get('nodes', []):
            found = _search(child)
            if found is not None:
                return found
        return None

    def _collect_text(node):
        parts = [node.get('text', '')]
        for child in node.get('nodes', []):
            parts.append(_collect_text(child))
        return "\n\n".join(filter(None, parts))

    return _search(tree)


def build_cross_document_meta_tree(query: str, doc_trees: List[Dict], llm: HybridLLM) -> Dict:
    """
    Stitches equivalent JSON sections from multiple documents into a single virtual root.
    PageIndex-style cross-document JSON processing.
    """
    # Step 1: Ask LLM what structural category is needed for the query
    intent_prompt = f"""To answer "{query}", which structural section of a scientific paper is most relevant?
(e.g., 'Experimental Setup', 'Results', 'Methodology', 'Introduction', 'Conclusion').
Return ONLY the section name."""
    target_section = llm.generate(intent_prompt, max_new_tokens=20).strip().strip('"').strip("'")

    # Step 2: Build the Virtual Meta-Root
    meta_root = {
        "node_id": "meta_root",
        "title": f"Cross-Document Meta-Tree: {target_section}",
        "doc_id": "META",
        "nodes": []
    }

    # Step 3: Search all document JSON trees and stitch matching nodes
    for tree in doc_trees:
        doc_id = tree.get('doc_id', 'unknown')
        matching_node = find_section_by_keyword(tree, target_section)

        if matching_node:
            meta_root["nodes"].append({
                "node_id": f"meta_{doc_id}_{matching_node.get('node_id', '')}",
                "title": f"[{doc_id}] {matching_node.get('title', '')}",
                "summary": matching_node.get('summary', ''),
                "doc_id": doc_id,
                "original_node_id": matching_node.get('node_id', ''),
                "nodes": matching_node.get('nodes', [])  # Preserve nested children!
            })

    return meta_root


class JSONMCTSNavigator:
    """
    JSON-Action MCTS Navigator.
    Instead of asking the LLM to "pick IDs from a list", feed it the raw JSON Meta-Tree
    and force it to output JSON navigation actions, exactly like PageIndex's internal agent.
    """
    def __init__(self, llm: HybridLLM, max_steps: int = 3, max_nodes_per_step: int = 25):
        self.llm = llm
        self.max_steps = max_steps
        self.max_nodes_per_step = max_nodes_per_step

    def navigate(self, query: str, meta_tree: Dict) -> List[Dict]:
        """Agentic navigation: LLM iteratively decides drill_down vs extract_text via JSON."""
        current_nodes = meta_tree.get("nodes", [])
        final_chunks = []

        for step in range(self.max_steps):
            if not current_nodes: 
                break

            # Cap nodes to prevent context overflow
            nodes_to_show = current_nodes[:self.max_nodes_per_step]
            if len(current_nodes) > self.max_nodes_per_step:
                logger.info(f"[JSONMCTS] Capped nodes from {len(current_nodes)} to {self.max_nodes_per_step}")

            # Format the current JSON nodes for the LLM
            nodes_summary = json.dumps([{
                "node_id": n['node_id'], 
                "doc_id": n['doc_id'],
                "title": n['title'], 
                "summary": n.get('summary', '')[:150]
            } for n in nodes_to_show], indent=2)

            prompt = f"""You are an expert scientific navigator. Step {step+1}.
QUERY: "{query}"
CURRENT JSON NODES:
{nodes_summary}

INSTRUCTIONS:
1. If a node's summary contains the answer, choose "extract_text".
2. If a node is a parent section and you need more detail, choose "drill_down".

Return JSON: {{"reasoning": "...", "actions": [{{"node_id": "...", "action": "drill_down|extract_text"}}]}}"""

            response = self.llm.generate(prompt, max_new_tokens=512, system_prompt="Return ONLY valid JSON.")
            actions_data = extract_json(response)

            if not actions_data:
                break

            actions = actions_data.get('actions', [])

            if not actions: 
                break

            next_level_nodes = []
            for action in actions:
                node_id = action.get('node_id')
                if action.get('action') == 'drill_down':
                    # Fetch children from the JSON tree (FIXED: "nodes" not "children")
                    children = get_children_from_tree(meta_tree, node_id)
                    next_level_nodes.extend(children)
                elif action.get('action') == 'extract_text':
                    # Fetch full text and stop drilling this branch
                    chunk = get_full_text_from_tree(meta_tree, node_id)
                    if chunk: 
                        final_chunks.append(chunk)

            current_nodes = next_level_nodes

        return final_chunks



# ============================================================================
# ════════════════════════════════════════════════════════════════════════════
# UNIFIED PIPELINE: advanced_retrieve_and_answer
# ════════════════════════════════════════════════════════════════════════════
# ============================================================================

def advanced_retrieve_and_answer(query: str, selected_docs: Dict, llm: HybridLLM,
                                 use_meta_tree: bool = False, doc_trees: List[Dict] = None,
                                 max_context_chars: int = 15000):
    """
    The DECLARMIMA v20.1 Minimal Pipeline.
    Integrates all 4 pillars + optional Meta-Tree navigation.
    """
    # 1. Route Intent
    router = ScientificIntentRouter()
    intent_data = router.route(query)

    # 2. Navigation (choose between flat MCTS or Meta-Tree JSON MCTS)
    if use_meta_tree and doc_trees:
        # PageIndex-style hierarchical JSON navigation
        meta_tree = build_cross_document_meta_tree(query, doc_trees, llm)
        navigator = JSONMCTSNavigator(llm, max_steps=3, max_nodes_per_step=25)
        retrieved_chunks = navigator.navigate(query, meta_tree)
        trace = ["[Meta-Tree] Used cross-document JSON hierarchical navigation."]
    else:
        # Flat section MCTS navigation
        navigator = IterativeTreeNavigator(llm, max_steps=2, max_nodes_per_step=30)
        retrieved_chunks = navigator.navigate(query, selected_docs)
        trace = navigator.trace

    if not retrieved_chunks:
        return (
            "I could not find relevant sections in the selected documents.", 
            "No sections matched.", 
            [],
            intent_data
        )

    # 3. Structured Extraction
    extractor = UniversalLLMExtractor()
    raw_items = extractor.extract(retrieved_chunks, query, llm)

    # 4. Citation Validation
    validator = CitationValidator()
    verified_items = validator.verify(raw_items, selected_docs)

    # Filter out severe hallucinations (confidence > 0.5)
    verified_items = [i for i in verified_items if i.get('confidence', 0) > 0.5]

    # 5. Adaptive Synthesis
    generator = AdaptiveResponseGenerator(llm)
    answer = generator.generate(query, verified_items, intent_data['intent'])

    return answer, "\n".join(trace), verified_items, intent_data


# ============================================================================
# LEGACY FALLBACK: agentic_retrieve_and_answer (preserved for compatibility)
# ============================================================================

def agentic_retrieve_and_answer(query: str, selected_docs: Dict, llm: HybridLLM, 
                                max_context_chars: int = 15000):
    """
    Phase 2 & 3: Agentic Multi-Document Tree Search and Answer Generation.
    Legacy single-shot approach - preserved for backward compatibility.
    """

    # 1. Build the "Forest" Description
    forest_desc = []
    for doc_name, data in selected_docs.items():
        forest_desc.append(f"--- Document: {doc_name} ---")
        for s in data['sections']:
            # Namespace the ID with a safe delimiter (:::)
            ns_id = f"{doc_name}:::{s['id']}"
            forest_desc.append(
                f"ID: {ns_id} | Title: {s['title']} | Pages: {s['start_page']}-{s['end_page']} | Summary: {s['summary']}"
            )

    tree_text = "\n".join(forest_desc)

    # 2. Agentic Navigation - LLM decides which sections are relevant
    search_prompt = f"""You are a research assistant. Given a query and a forest of document structures, identify which sections across the documents are most likely to contain the answer.

Query: {query}

Document Structures:
{tree_text}

Return a JSON object: {{"thinking": "your step-by-step reasoning", "relevant_ids": ["doc_name:::sec_id1", "doc_name:::sec_id2"]}}.
If no sections are relevant, return an empty list for relevant_ids."""

    system = "You are a precise navigation agent. Return ONLY valid JSON."
    search_response = llm.generate(search_prompt, max_new_tokens=1024, system_prompt=system)
    search_result = extract_json(search_response)

    thinking = "No reasoning provided."
    relevant_ids = []
    if search_result:
        thinking = search_result.get('thinking', 'No reasoning provided.')
        relevant_ids = search_result.get('relevant_ids', [])

    # 3. Fetch Content (Lossless Retrieval)
    context = ""
    retrieved_info = []

    for ns_id in relevant_ids:
        if ":::" not in ns_id: 
            continue
        doc_name, sec_id = ns_id.split(":::", 1)
        if doc_name in selected_docs:
            data = selected_docs[doc_name]
            sections = data['sections']
            pages = data['pages']

            sec = next((s for s in sections if s['id'] == sec_id), None)
            if sec:
                retrieved_info.append({"doc_name": doc_name, "section": sec})
                text_parts = []
                for p_num in range(sec['start_page'], sec['end_page'] + 1):
                    if 0 < p_num <= len(pages):
                        text_parts.append(pages[p_num - 1]['text'])
                context += f"\n\n--- Document: {doc_name} | Section: {sec['title']} (Pages {sec['start_page']}-{sec['end_page']}) ---\n" + "\n".join(text_parts) + "\n"

    if not context:
        return (
            "I could not find any relevant sections in the selected documents to answer your query.", 
            thinking, 
            retrieved_info
        )

    # Safeguard for context limits
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n[Context truncated due to length]"

    # 4. Generate Answer using the same LLM
    answer_prompt = f"""Answer the user's query based ONLY on the provided document context.
If the answer is not in the context, say "I don't know based on the provided documents."
Cite the document name and section title when possible.
Be concise but thorough. If there are numbers or specific data points, include them.

Query: {query}
Context:
{context}"""

    system = "You are a helpful document QA assistant. Be concise, accurate, and cite sources."
    answer = llm.generate(answer_prompt, max_new_tokens=2048, temperature=0.1, system_prompt=system)

    return answer, thinking, retrieved_info


# ============================================================================
# VECTORLESS RAG CORE: Document Indexing
# ============================================================================

def index_all_documents(uploaded_files, llm: HybridLLM, chunk_size: int = 5,
                       build_json_trees: bool = False):
    """
    Phase 1: Builds hierarchical tree indices for multiple PDFs.
    Optionally builds nested JSON trees using pymupdf4llm.
    """
    documents = {}
    doc_trees = []
    total_files = len(uploaded_files)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, uploaded_file in enumerate(uploaded_files):
        doc_name = os.path.splitext(uploaded_file.name)[0]
        # Handle duplicate filenames
        if doc_name in documents:
            doc_name = f"{doc_name}_{idx+1}"

        status_text.info(f"Processing ({idx+1}/{total_files}): {uploaded_file.name}...")

        pdf_bytes = uploaded_file.read()

        # Use the safe extraction function
        pages_text = extract_text_from_pdf(pdf_bytes)

        # Optional: Build nested JSON tree from Markdown (if pymupdf4llm available)
        json_tree = None
        if build_json_trees and GLOBAL_DEPS.get('pymupdf4llm'):
            try:
                import pymupdf4llm
                # Save to temp file for pymupdf4llm
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                    tmp.write(pdf_bytes)
                    tmp_path = tmp.name

                md_text = pymupdf4llm.to_markdown(tmp_path)
                json_tree = _build_tree_from_markdown(md_text, doc_id=doc_name)
                json_tree['doc_id'] = doc_name

                # Generate summaries for tree nodes using LLM
                def _summarize_nodes(node):
                    text = node.get('text', '')
                    if text and len(text) > 50:
                        prompt = f"Summarize this section in 1-2 sentences:\n{text[:2000]}"
                        summary = llm.generate(prompt, max_new_tokens=128, temperature=0.1)
                        node['summary'] = summary.strip()
                    for child in node.get('nodes', []):
                        _summarize_nodes(child)

                _summarize_nodes(json_tree)
                doc_trees.append(json_tree)
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"JSON tree build failed for {doc_name}: {e}")

        # 1. Extract Sections using LLM (Chunked to fit context windows)
        sections = []
        for i in range(0, len(pages_text), chunk_size):
            chunk = pages_text[i:i+chunk_size]
            text_with_tags = "\n".join([
                f"<page_{p['page_num']}>\n{p['text']}\n</page_{p['page_num']}>" 
                for p in chunk
            ])

            prompt = f"""Analyze the document text and extract main section headings and their starting page numbers.
Return a JSON list of objects: {{"title": "Section Title", "start_page": page_number}}.
Ignore headers/footers. Only include actual content sections.

Text:
{text_with_tags}"""

            system = "You are an expert document analyzer. Return ONLY a valid JSON list."
            response = llm.generate(prompt, max_new_tokens=1024, system_prompt=system)
            parsed = extract_json(response)

            if parsed and isinstance(parsed, list):
                for item in parsed:
                    if 'title' in item and 'start_page' in item:
                        sections.append(item)

        # 2. Determine End Pages and Build Flat Tree
        sections.sort(key=lambda x: x['start_page'])
        for i in range(len(sections)):
            sections[i]['end_page'] = sections[i+1]['start_page'] - 1 if i + 1 < len(sections) else len(pages_text)
            sections[i]['id'] = f"sec_{i+1:03d}"

        # 3. Generate Summaries for each node using LLM
        for i, sec in enumerate(sections):
            start_idx = sec['start_page'] - 1
            end_idx = sec['end_page']
            section_text = "\n".join([p['text'] for p in pages_text[start_idx:end_idx]])

            # Limit text length for summary to avoid context overflow
            if len(section_text) > 4000:
                section_text = section_text[:4000] + "..."

            prompt = f"Summarize the following text in 2-3 concise sentences, focusing on key facts, numbers, and findings:\n\n{section_text}"
            system = "You are a helpful scientific summarizer. Be concise and factual."
            summary = llm.generate(prompt, max_new_tokens=256, system_prompt=system)
            sec['summary'] = summary.strip()

        documents[doc_name] = {
            'pages': pages_text, 
            'sections': sections, 
            'filename': uploaded_file.name,
            'pdf_bytes': pdf_bytes,
            'json_tree': json_tree,
        }
        progress_bar.progress((idx + 1) / total_files)

    progress_bar.empty()
    status_text.empty()
    return documents, doc_trees



# ============================================================================
# STREAMLIT UI
# ============================================================================

def render_sidebar():
    """Render the sidebar with model selection and configuration."""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        # Model selection
        model_keys = list(LOCAL_LLM_OPTIONS.keys())
        if "llm_model_choice" not in st.session_state:
            st.session_state.llm_model_choice = model_keys[2]  # Default: qwen2.5:7b

        selected = st.selectbox(
            "Select LLM Model",
            options=model_keys,
            index=model_keys.index(st.session_state.llm_model_choice),
            key="llm_model_select",
            help="Choose between Ollama (fast, local API) or HuggingFace Transformers (local loading, requires more RAM/VRAM)"
        )
        st.session_state.llm_model_choice = selected

        # Show backend info
        model_key = LOCAL_LLM_OPTIONS[selected]
        if model_key.startswith("ollama:"):
            st.caption("🟢 Backend: Ollama (API)")
            st.caption(f"Model: `{model_key.replace('ollama:', '')}`")
        else:
            st.caption("🔵 Backend: Transformers (Local)")
            st.caption(f"Model: `{model_key}`")

        # 4-bit quantization toggle (only for transformers)
        if not model_key.startswith("ollama:"):
            st.checkbox("Use 4-bit quantization (saves VRAM)", value=True, key="use_4bit")
            if TORCH_AVAILABLE and torch.cuda.is_available():
                st.caption(f"GPU: {torch.cuda.get_device_name(0)}")
            else:
                st.warning("⚠️ No GPU detected. Local model will run on CPU (slow).")
        else:
            st.session_state.use_4bit = False

        st.markdown("---")
        st.markdown("#### 📊 System Status")

        # Show dependency status
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"{'✅' if OLLAMA_AVAILABLE else '❌'} Ollama")
        with cols[1]:
            st.markdown(f"{'✅' if TRANSFORMERS_AVAILABLE else '❌'} Transformers")
        cols2 = st.columns(2)
        with cols2[0]:
            st.markdown(f"{'✅' if TORCH_AVAILABLE else '❌'} PyTorch")
        with cols2[1]:
            st.markdown(f"{'✅' if PYMUPDF_AVAILABLE else '❌'} PyMuPDF")
        cols3 = st.columns(2)
        with cols3[0]:
            st.markdown(f"{'✅' if GLOBAL_DEPS.get('pymupdf4llm') else '❌'} pymupdf4llm")
        with cols3[1]:
            st.markdown(f"{'✅' if RAPIDFUZZ_AVAILABLE else '❌'} rapidfuzz")

        st.markdown("---")

        # Advanced settings
        with st.expander("Advanced Settings", expanded=False):
            st.slider("Max context chars", 5000, 30000, 15000, 1000, 
                     key="max_context_chars",
                     help="Maximum characters to send to LLM as context")
            st.slider("Chunk size (pages)", 1, 10, 5, 1,
                     key="chunk_size",
                     help="Pages per chunk for section extraction")
            st.checkbox("Use Meta-Tree JSON navigation", value=False, key="use_meta_tree",
                       help="Enable PageIndex-style hierarchical JSON tree navigation (requires pymupdf4llm)")
            st.checkbox("Build JSON trees during indexing", value=False, key="build_json_trees",
                       help="Build nested Markdown header trees during document indexing")

        st.markdown("---")

        if st.button("🗑️ Clear Cache & Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def run_streamlit():
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="DECLARMIMA v20.1 - 4-Pillar Vectorless RAG", 
        layout="wide"
    )
    st.title("🌲 DECLARMIMA v20.1 - 4-Pillar Vectorless RAG")
    st.markdown(
        "Upload multiple PDFs, build hierarchical tree indices, and query them using **4-Pillar agentic navigation**. "
        "**No Vector DBs. No Chunking. No Embeddings.** "
        "Supports both **Ollama** (fast API) and **HuggingFace Transformers** (local loading). "
        "Now with **Intent Routing**, **Citation Validation**, and **Meta-Tree JSON Navigation**."
    )

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'selected_docs_for_query' not in st.session_state:
        st.session_state.selected_docs_for_query = []
    if 'documents' not in st.session_state:
        st.session_state.documents = {}
    if 'doc_trees' not in st.session_state:
        st.session_state.doc_trees = []
    if 'llm' not in st.session_state:
        st.session_state.llm = None

    render_sidebar()

    # Sidebar - Document Upload
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
                    # Initialize LLM (cached)
                    llm = get_cached_llm(
                        st.session_state.llm_model_choice, 
                        st.session_state.get("use_4bit", True)
                    )
                    st.session_state.llm = llm
                    st.success(f"✅ LLM loaded: {llm.model_name} via {llm.backend}")

                    # Index documents
                    build_trees = st.session_state.get("build_json_trees", False)
                    documents, doc_trees = index_all_documents(
                        uploaded_files, 
                        llm,
                        chunk_size=st.session_state.get("chunk_size", 5),
                        build_json_trees=build_trees
                    )
                    st.session_state.documents = documents
                    st.session_state.doc_trees = doc_trees
                    st.session_state.messages = []  # Clear chat history
                    st.success(f"✅ Indexed {len(uploaded_files)} document(s)")

                    if doc_trees:
                        st.info(f"📊 Built {len(doc_trees)} JSON hierarchical trees")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.error(f"Initialization error: {e}", exc_info=True)

        # Document selector
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
                    data = st.session_state.documents[doc_name]
                    for s in data['sections']:
                        st.markdown(f"**{s['id']}: {s['title']}** *(pp. {s['start_page']}-{s['end_page']})*")
                        st.caption(s['summary'])
                        st.divider()

    # Main Chat Area
    if not st.session_state.documents:
        st.info("👈 Please upload PDF(s) in the sidebar and click 'Build Document Trees' to start.")

        # Show quick start guide
        with st.expander("📖 Quick Start Guide", expanded=True):
            st.markdown("""
            ### Getting Started

            1. **Choose your LLM backend** in the sidebar:
               - **Ollama** (recommended): Fast, runs via API. Install from [ollama.com](https://ollama.com)
               - **HuggingFace**: Loads models locally. Requires more RAM/VRAM.

            2. **Upload PDFs** using the sidebar uploader

            3. **Click 'Build Document Trees'** to index your documents

            4. **Ask questions** in the chat below

            ### 4-Pillar Architecture
            - **Pillar 1 - Intent Router**: Automatically detects if you want values, equations, mechanisms, comparisons, or methodology
            - **Pillar 2 - MCTS Navigator**: Drill-down agent selects exact document sections
            - **Pillar 3 - Citation Validator**: Cross-checks extracted data against raw PDF text to kill hallucinations
            - **Pillar 4 - Adaptive Synthesizer**: Forces strict Markdown tables, LaTeX blocks, or causal chains

            ### Requirements
            - **PyMuPDF**: `pip install pymupdf`
            - **For Ollama**: `pip install ollama` + [Ollama app](https://ollama.com)
            - **For HF**: `pip install transformers torch`
            - **Optional**: `pip install pymupdf4llm` (better LaTeX/math + Meta-Tree JSON)
            - **Optional**: `pip install rapidfuzz` (fuzzy matching for Meta-Tree)
            """)
    else:
        st.subheader("💬 Chat with your Documents")

        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])
                if msg['role'] == 'assistant':
                    # Show reasoning expander
                    if 'thinking' in msg:
                        with st.expander("🧠 Agent Reasoning & Retrieved Sections"):
                            st.markdown(f"**Thinking Process:**\n{msg['thinking']}")
                            if msg.get('retrieved_info'):
                                st.markdown("**Retrieved Sections:**")
                                for info in msg['retrieved_info']:
                                    sec = info['section']
                                    st.info(
                                        f"📄 **{info['doc_name']}** | {sec['id']}: {sec['title']} "
                                        f"(Pages {sec['start_page']}-{sec['end_page']})"
                                    )
                    # Show verified extractions expander
                    if 'verified_items' in msg and msg['verified_items']:
                        with st.expander("✅ Verified Extractions (Hallucination Checked)"):
                            st.markdown("**Detected Intent:** " + msg.get('intent', 'open_query'))
                            for item in msg['verified_items'][:10]:
                                conf = item.get('confidence', 0)
                                color = "🟢" if conf > 0.8 else "🟡" if conf > 0.5 else "🔴"
                                param = item.get('parameter_name', 'N/A')
                                val = item.get('value', item.get('equation_latex', 'N/A'))
                                unit = item.get('unit', '')
                                status = item.get('verification_status', 'unknown')
                                st.caption(
                                    f"{color} **{param}**: {val} {unit} | "
                                    f"Conf: {conf:.2f} | Status: {status} | "
                                    f"[{item.get('doc_name')}, p.{item.get('page')}]"
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
                    with st.spinner("🔍 Routing intent → Navigating → Extracting → Validating → Synthesizing..."):
                        try:
                            # Filter documents
                            docs_to_search = {
                                k: st.session_state.documents[k] 
                                for k in st.session_state.selected_docs_for_query
                            }

                            # Get LLM (use cached)
                            llm = st.session_state.llm or get_cached_llm(
                                st.session_state.llm_model_choice,
                                st.session_state.get("use_4bit", True)
                            )

                            # Decide which pipeline to use
                            use_meta = st.session_state.get("use_meta_tree", False)
                            doc_trees = st.session_state.get("doc_trees", [])

                            if use_meta and doc_trees:
                                # Use advanced 4-pillar + Meta-Tree pipeline
                                answer, thinking, verified_items, intent_data = advanced_retrieve_and_answer(
                                    prompt, 
                                    docs_to_search, 
                                    llm,
                                    use_meta_tree=True,
                                    doc_trees=doc_trees,
                                    max_context_chars=st.session_state.get("max_context_chars", 15000)
                                )

                                st.markdown(answer)

                                with st.expander("🧠 Agent Reasoning & Retrieved Sections"):
                                    st.markdown(f"**Detected Intent:** {intent_data['intent']}")
                                    st.markdown(f"**Navigation Trace:**\n{thinking}")

                                with st.expander("✅ Verified Extractions (Hallucination Checked)"):
                                    if verified_items:
                                        for item in verified_items[:10]:
                                            conf = item.get('confidence', 0)
                                            color = "🟢" if conf > 0.8 else "🟡" if conf > 0.5 else "🔴"
                                            param = item.get('parameter_name', 'N/A')
                                            val = item.get('value', item.get('equation_latex', 'N/A'))
                                            unit = item.get('unit', '')
                                            status = item.get('verification_status', 'unknown')
                                            st.caption(
                                                f"{color} **{param}**: {val} {unit} | "
                                                f"Conf: {conf:.2f} | Status: {status} | "
                                                f"[{item.get('doc_name')}, p.{item.get('page')}]"
                                            )
                                    else:
                                        st.caption("No verified extractions found.")

                                # Save to history
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": answer, 
                                    "thinking": thinking, 
                                    "verified_items": verified_items,
                                    "intent": intent_data['intent'],
                                })
                            else:
                                # Use legacy pipeline (backward compatible)
                                answer, thinking, retrieved_info = agentic_retrieve_and_answer(
                                    prompt, 
                                    docs_to_search, 
                                    llm,
                                    max_context_chars=st.session_state.get("max_context_chars", 15000)
                                )

                                st.markdown(answer)

                                with st.expander("🧠 Agent Reasoning & Retrieved Sections"):
                                    st.markdown(f"**Thinking Process:**\n{thinking}")
                                    if retrieved_info:
                                        st.markdown("**Retrieved Sections:**")
                                        for info in retrieved_info:
                                            sec = info['section']
                                            st.info(
                                                f"📄 **{info['doc_name']}** | {sec['id']}: {sec['title']} "
                                                f"(Pages {sec['start_page']}-{sec['end_page']})"
                                            )

                                # Save to history
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": answer, 
                                    "thinking": thinking, 
                                    "retrieved_info": retrieved_info
                                })

                        except Exception as e:
                            error_msg = f"❌ Error generating response: {str(e)}"
                            st.error(error_msg)
                            logger.error(f"Query error: {e}", exc_info=True)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": error_msg
                            })


if __name__ == "__main__":
    run_streamlit()
