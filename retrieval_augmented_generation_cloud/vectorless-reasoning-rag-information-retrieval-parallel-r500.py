#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
DECLARMIMA v20.0 Enhanced - Hybrid Vectorless RAG
===================================================
Combines the robust import/LLM-loading system from DECLARMIMA v20.0
with the clean vectorless RAG architecture from the shorter codebase.

Key enhancements:
- Graceful dependency degradation (Ollama, Transformers, PyMuPDF, etc.)
- Hybrid LLM backend (Ollama local + HuggingFace transformers local)
- Cached LLM initialization via @st.cache_resource
- Model-specific prompt templates (Qwen, Mistral, Llama, etc.)
- Enhanced PDF processing with pymupdf4llm fallback
- 100% Vectorless: No embeddings, no vector DBs
- Agentic tree navigation for multi-document retrieval
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
logger = logging.getLogger("DECLARMIMA_ENHANCED")

# ============================================================================
# DEPENDENCY CHECKS WITH GRACEFUL DEGRADATION (from DECLARMIMA v20.0)
# ============================================================================

def check_optional_dependencies() -> Dict[str, bool]:
    """Check all optional dependencies and report availability with graceful degradation."""
    deps: Dict[str, bool] = {}

    # PyMuPDF (required)
    try:
        try:
            import pymupdf
            deps['pymupdf'] = True
            import sys
            if 'fitz' not in sys.modules:
                sys.modules['fitz'] = pymupdf
        except ImportError:
            import fitz
            deps['pymupdf'] = True
        logger.info("✓ PyMuPDF available")
    except ImportError:
        deps['pymupdf'] = False
        logger.error("✗ PyMuPDF required: pip install pymupdf")
        raise ImportError("PyMuPDF is required for this application to function")

    # Ollama (recommended for local LLM)
    try:
        import ollama
        deps['ollama'] = True
        logger.info("✓ Ollama client available")
    except ImportError:
        deps['ollama'] = False
        logger.warning("✗ Ollama not installed. Ollama backend unavailable.")

    # HuggingFace Transformers (recommended for local LLM)
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        deps['transformers'] = True
        logger.info("✓ HuggingFace transformers available")
    except ImportError:
        deps['transformers'] = False
        logger.warning("✗ transformers not installed. Local HF models unavailable.")

    # rapidfuzz (optional - for fuzzy matching)
    try:
        from rapidfuzz import fuzz, process
        deps['rapidfuzz'] = True
        logger.info("✓ rapidfuzz available")
    except ImportError:
        deps['rapidfuzz'] = False
        logger.warning("✗ rapidfuzz not installed. Fuzzy matching disabled.")

    # orjson (optional - faster JSON)
    try:
        import orjson
        deps['orjson'] = True
        logger.info("✓ orjson available (fast JSON)")
    except ImportError:
        deps['orjson'] = False
        logger.warning("✗ orjson not installed. Using standard json (slower).")

    # pyvis (optional - interactive networks)
    try:
        from pyvis.network import Network
        deps['pyvis'] = True
        logger.info("✓ pyvis available")
    except ImportError:
        deps['pyvis'] = False
        logger.warning("✗ pyvis not installed. Interactive networks disabled.")

    logger.info(f"Dependency check complete: {sum(deps.values())}/{len(deps)} available")
    return deps

# Check dependencies at module load time
GLOBAL_DEPS = check_optional_dependencies()

# ============================================================================
# UNIFIED PDF IMPORT (handles both old and new PyMuPDF)
# ============================================================================
try:
    # Try modern import first (PyMuPDF >= 1.23)
    try:
        import pymupdf
        PYMUPDF_AVAILABLE = True
        # Make fitz available as alias for backward compatibility
        import sys
        if 'fitz' not in sys.modules:
            sys.modules['fitz'] = pymupdf
        import fitz  # Now available as alias
        logger.info("✓ PyMuPDF (modern) loaded via pymupdf")
    except ImportError:
        # Fallback to legacy import (PyMuPDF < 1.23)
        import fitz
        PYMUPDF_AVAILABLE = True
        logger.info("✓ PyMuPDF (legacy) loaded via fitz")
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.error("✗ PyMuPDF required: pip install pymupdf")
    raise ImportError("PyMuPDF is required. Run: pip install pymupdf")

# Ollama client
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not installed. Ollama backend unavailable.")

# Transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# PyTorch (needed for transformers)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. Transformers backend requires torch.")

# orjson for fast JSON
try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False
    logger.warning("orjson not installed. Using standard json (slower).")

# rapidfuzz for fuzzy matching
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

# pyvis for interactive networks
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
# HYBRID LLM & TEMPLATES (from DECLARMIMA v20.0)
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
    # HuggingFace Transformers models (local loading)
    "[HF] Qwen2.5-7B-Instruct (Local)": "Qwen/Qwen2.5-7B-Instruct",
    "[HF] Mistral-7B-Instruct-v0.3 (Local)": "mistralai/Mistral-7B-Instruct-v0.3",
    "[HF] Llama-3.1-8B-Instruct (Local)": "meta-llama/Llama-3.1-8B-Instruct",
}

MODEL_PROMPT_TEMPLATES = {
    "qwen2.5:0.5b": {"system": "You are a precise document analyst. Follow JSON format strictly.", "json_reminder": "Return ONLY valid JSON."},
    "qwen2.5:1.5b": {"system": "You are a precise document analyst. Follow JSON format strictly.", "json_reminder": "Return ONLY valid JSON."},
    "qwen2.5:7b": {"system": "You are a precise document analyst. Follow JSON format strictly.", "json_reminder": "Return ONLY valid JSON."},
    "qwen2.5:14b": {"system": "You are a precise document analyst. Follow JSON format strictly.", "json_reminder": "Return ONLY valid JSON."},
    "mistral": {"system": "You are a helpful assistant that always returns valid JSON.", "json_reminder": "Return ONLY valid JSON."},
    "llama3.1": {"system": "You are a helpful assistant. Be concise and accurate.", "json_reminder": "Return valid JSON only."},
    "gemma2": {"system": "You are a helpful assistant.", "json_reminder": "Return valid JSON only."},
    "falcon3": {"system": "You are a helpful assistant.", "json_reminder": "Return valid JSON only."},
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

    Backend selection priority:
    1. Ollama (if available and running at localhost:11434)
    2. Transformers (if transformers + torch installed)

    Model key formats:
    - "ollama:model_name" -> Ollama backend
    - "Qwen/Qwen2.5-7B-Instruct" -> Transformers backend (HuggingFace model ID)
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
            "No LLM backend available. Please either:
"
            "1. Install and start Ollama (pip install ollama, then ollama serve)
"
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
# JSON EXTRACTION UTILITIES (enhanced from both codes)
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
            # Fix common LLM JSON errors
            candidate = re.sub(r',\s*([\]}])', r'\1', candidate)  # Remove trailing commas
            candidate = re.sub(r'\s+', ' ', candidate)  # Normalize whitespace
            try:
                return json.loads(candidate)
            except:
                pass

    return None


# ============================================================================
# PDF PROCESSING (enhanced with pymupdf4llm fallback from DECLARMIMA)
# ============================================================================

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

    def extract_all_pages(self, pdf_bytes: bytes) -> List[Dict[str, Any]]:
        """Extract all pages from PDF bytes with metadata."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text("text")
            pages.append({
                'page': i + 1,
                'text': text,
                'images': len(page.get_images()),
            })
        doc.close()
        return pages


# ============================================================================
# VECTORLESS RAG CORE (from shorter code, enhanced)
# ============================================================================

def index_all_documents(uploaded_files, llm: HybridLLM, chunk_size: int = 5):
    """
    Phase 1: Builds hierarchical tree indices for multiple PDFs.
    Enhanced with better error handling and progress tracking.
    """
    documents = {}
    total_files = len(uploaded_files)
    progress_bar = st.progress(0)
    status_text = st.empty()
    reader = PaginationAwareReader()

    for idx, uploaded_file in enumerate(uploaded_files):
        doc_name = os.path.splitext(uploaded_file.name)[0]
        # Handle duplicate filenames
        if doc_name in documents:
            doc_name = f"{doc_name}_{idx+1}"

        status_text.info(f"Processing ({idx+1}/{total_files}): {uploaded_file.name}...")

        pdf_bytes = uploaded_file.read()

        # Use enhanced reader for page extraction
        pages_text = reader.extract_all_pages(pdf_bytes)

        # 1. Extract Sections using LLM (Chunked to fit context windows)
        sections = []
        for i in range(0, len(pages_text), chunk_size):
            chunk = pages_text[i:i+chunk_size]
            text_with_tags = "\n".join([
                f"<page_{p['page']}>\n{p['text']}\n</page_{p['page']}>" 
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
            'pdf_bytes': pdf_bytes
        }
        progress_bar.progress((idx + 1) / total_files)

    progress_bar.empty()
    status_text.empty()
    return documents


def agentic_retrieve_and_answer(query: str, selected_docs: Dict, llm: HybridLLM, 
                                max_context_chars: int = 15000):
    """
    Phase 2 & 3: Agentic Multi-Document Tree Search and Answer Generation.
    Enhanced with the HybridLLM integration.
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
    reader = PaginationAwareReader()

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
# STREAMLIT UI (enhanced with model selection and better UX)
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
            if torch.cuda.is_available():
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

        st.markdown("---")

        # Advanced settings
        with st.expander("Advanced Settings", expanded=False):
            st.slider("Max context chars", 5000, 30000, 15000, 1000, 
                     key="max_context_chars",
                     help="Maximum characters to send to LLM as context")
            st.slider("Chunk size (pages)", 1, 10, 5, 1,
                     key="chunk_size",
                     help="Pages per chunk for section extraction")

        st.markdown("---")

        if st.button("🗑️ Clear Cache & Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def run_streamlit():
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="DECLARMIMA Enhanced - Hybrid Vectorless RAG", 
        layout="wide"
    )
    st.title("🌲 DECLARMIMA Enhanced - Hybrid Vectorless RAG")
    st.markdown(
        "Upload multiple PDFs, build hierarchical tree indices, and query them using **agentic navigation**. "
        "**No Vector DBs. No Chunking. No Embeddings.** "
        "Supports both **Ollama** (fast API) and **HuggingFace Transformers** (local loading)."
    )

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'selected_docs_for_query' not in st.session_state:
        st.session_state.selected_docs_for_query = []
    if 'documents' not in st.session_state:
        st.session_state.documents = {}
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
                    st.session_state.documents = index_all_documents(
                        uploaded_files, 
                        llm,
                        chunk_size=st.session_state.get("chunk_size", 5)
                    )
                    st.session_state.messages = []  # Clear chat history
                    st.success(f"✅ Indexed {len(uploaded_files)} document(s)")

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

            ### Requirements
            - **PyMuPDF**: `pip install pymupdf`
            - **For Ollama**: `pip install ollama` + [Ollama app](https://ollama.com)
            - **For HF**: `pip install transformers torch`
            - **Optional**: `pip install pymupdf4llm` (better LaTeX/math extraction)
            """)
    else:
        st.subheader("💬 Chat with your Documents")

        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])
                if msg['role'] == 'assistant' and 'thinking' in msg:
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
                    with st.spinner("Searching document trees and generating answer..."):
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

                            # Agentic retrieval and answer
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
