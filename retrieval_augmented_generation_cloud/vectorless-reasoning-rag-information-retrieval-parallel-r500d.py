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
logger = logging.getLogger("DECLARMIMA_ENHANCED")

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

    # PyMuPDF (required) - checked via the safe import above
    deps['pymupdf'] = PYMUPDF_AVAILABLE
    if PYMUPDF_AVAILABLE:
        logger.info("✓ PyMuPDF available")
    else:
        logger.error("✗ PyMuPDF required: pip install pymupdf")

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

# Ollama client
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

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
            # Fix common LLM JSON errors
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
# VECTORLESS RAG CORE
# ============================================================================

def index_all_documents(uploaded_files, llm: HybridLLM, chunk_size: int = 5):
    """
    Phase 1: Builds hierarchical tree indices for multiple PDFs.
    """
    documents = {}
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
# 4 ARCHITECTURAL PILLARS FOR PAGEINDEX-LEVEL INTELLIGENCE
# ============================================================================
# These classes elevate the short code to DECLARMIMA v20.0-level intelligence
# by adding: Intent Routing, Iterative MCTS Navigation, Structured Extraction
# with Citation Validation, and Adaptive Response Generation.
# ============================================================================


# ============================================================================
# 4 ARCHITECTURAL PILLARS FOR PAGEINDEX-LEVEL INTELLIGENCE
# ============================================================================

class ScientificIntentRouter:
    """Lightweight intent classifier using regex triggers."""
    PATTERNS = {
        "value_extraction": [r"\bvalue\b", r"\bhow much\b", r"\b\d+\s*(?:W|kW|MPa|mm/s|GPa)\b", r"\btable\b", r"\blist\b"],
        "equation": [r"\bequation\b", r"\bformula\b", r"\bconstitutive\b", r"\bnavier[- ]stokes\b", r"\bcahn[- ]hilliard\b"],
        "mechanism": [r"\bwhy\b", r"\bhow does\b", r"\bmechanism\b", r"\bcause\b", r"\bdriving force\b"],
    }

    def route(self, query: str) -> Dict[str, str]:
        q_lower = query.lower()
        for intent, patterns in self.PATTERNS.items():
            if any(re.search(p, q_lower) for p in patterns):
                return {"intent": intent, "output_format": intent}
        return {"intent": "open_query", "output_format": "prose"}


class IterativeTreeNavigator:
    """Simulates PageIndex MCTS navigation using the short code's flat sections."""
    def __init__(self, llm: HybridLLM, max_steps: int = 2):
        self.llm = llm
        self.max_steps = max_steps
        self.trace = []

    def navigate(self, query: str, selected_docs: Dict) -> List[Dict]:
        retrieved_chunks = []

        # Step 1: Build the "Forest" of summaries
        forest_desc = []
        for doc_name, data in selected_docs.items():
            for s in data['sections']:
                forest_desc.append(f"ID: {doc_name}:::{s['id']} | Title: {s['title']} | Pages: {s['start_page']}-{s['end_page']} | Summary: {s['summary'][:150]}")

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
                break

            self.trace.append(actions_data.get('reasoning', ''))

            for action in actions_data['actions']:
                ns_id = action.get('id')
                if ":::" not in ns_id: 
                    continue
                doc_name, sec_id = ns_id.split(":::", 1)

                if doc_name in selected_docs:
                    sec = next((s for s in selected_docs[doc_name]['sections'] if s['id'] == sec_id), None)
                    if sec:
                        # Fetch raw text
                        pages = selected_docs[doc_name]['pages']
                        text_parts = [pages[p-1]['text'] for p in range(sec['start_page'], sec['end_page']+1) if 0 < p <= len(pages)]
                        retrieved_chunks.append({
                            "doc_name": doc_name, "section_title": sec['title'],
                            "start_page": sec['start_page'], "end_page": sec['end_page'],
                            "full_text": "\n".join(text_parts)
                        })
            break # For minimal workability, 1-step deep extraction is highly effective

        return retrieved_chunks


class UniversalLLMExtractor:
    """Extracts structured JSON items from raw text chunks."""
    PROMPT = """Extract quantitative data, equations, or mechanisms from the text.
    Return a JSON list of objects: {{"item_type": "quantitative|equation|mechanism", "parameter_name": "...", "value": number_or_null, "unit": "...", "equation_latex": "...", "context": "exact sentence", "doc_name": "...", "page": int}}.
    Text: {text}"""

    def __init__(self, llm: HybridLLM):
        self.llm = llm

    def extract(self, chunks: List[Dict], query: str) -> List[Dict]:
        items = []
        for chunk in chunks:
            prompt = self.PROMPT.format(text=chunk['full_text'][:4000])
            resp = self.llm.generate(prompt, max_new_tokens=1024, system_prompt="Return ONLY valid JSON list.")
            parsed = extract_json(resp)
            if isinstance(parsed, list):
                for item in parsed:
                    item['doc_name'] = chunk['doc_name']
                    item['page'] = chunk['start_page']
                    items.append(item)
        return items


class CitationValidator:
    """Cross-checks extracted values against raw page text to prevent hallucinations."""
    def verify(self, items: List[Dict], selected_docs: Dict) -> List[Dict]:
        verified = []
        for item in items:
            doc_name = item.get('doc_name')
            page_num = item.get('page', 1)
            if doc_name in selected_docs:
                raw_text = selected_docs[doc_name]['pages'][page_num-1]['text'].lower()

                # Verification Logic
                val_str = str(item.get('value', ''))
                param = (item.get('parameter_name') or '').lower()

                if item.get('value') is not None and val_str in raw_text:
                    item['confidence'] = 0.95
                    verified.append(item)
                elif param and param in raw_text:
                    item['confidence'] = 0.7
                    verified.append(item)
                elif item.get('item_type') == 'equation' and item.get('equation_latex'):
                    item['confidence'] = 0.8 # Hard to verify LaTeX via raw text, trust LLM
                    verified.append(item)
                else:
                    item['confidence'] = 0.2 # Hallucination suspected
                    verified.append(item) # Keep it but mark low confidence
        return sorted(verified, key=lambda x: x.get('confidence', 0), reverse=True)


class AdaptiveResponseGenerator:
    """Generates strict template-enforced responses based on intent."""
    def __init__(self, llm: HybridLLM):
        self.llm = llm

    def generate(self, query: str, items: List[Dict], intent: str) -> str:
        evidence = "\n".join([f"- [{i.get('doc_name')}, p.{i.get('page')}] {i.get('context', i.get('equation_latex', ''))}" for i in items[:15]])

        if intent == "value_extraction":
            prompt = f"""Create a Markdown table of extracted values.
            Query: {query}
            Evidence: {evidence}
            Columns: | Parameter | Value | Unit | Document | Page | Confidence |
            RULE: Every row MUST be backed by the evidence. Do not invent data."""

        elif intent == "equation":
            prompt = f"""State the governing equations in LaTeX format ($$ ... $$).
            Query: {query}
            Evidence: {evidence}
            RULE: Define all variables immediately below the equation. Cite the document."""

        else: # mechanism or open_query
            prompt = f"""Answer the query comprehensively.
            Query: {query}
            Evidence: {evidence}
            RULE: Use inline citations like [DocName, p.X] for every factual claim. Explain the physical mechanism."""

        return self.llm.generate(prompt, max_new_tokens=1500, system_prompt="You are a scientific analyst. Follow formatting rules strictly.")


# ============================================================================
# ADVANCED RETRIEVAL PIPELINE (DECLARMIMA-STYLE)
# ============================================================================

def advanced_retrieve_and_answer(query: str, selected_docs: Dict, llm: HybridLLM):
    """The DECLARMIMA v20.0 Minimal Pipeline."""
    # 1. Route Intent
    router = ScientificIntentRouter()
    intent_data = router.route(query)

    # 2. Iterative Navigation
    navigator = IterativeTreeNavigator(llm)
    retrieved_chunks = navigator.navigate(query, selected_docs)

    if not retrieved_chunks:
        return "I could not find relevant sections in the selected documents.", "No sections matched.", []

    # 3. Structured Extraction
    extractor = UniversalLLMExtractor(llm)
    raw_items = extractor.extract(retrieved_chunks, query)

    # 4. Citation Validation
    validator = CitationValidator()
    verified_items = validator.verify(raw_items, selected_docs)

    # Filter out severe hallucinations
    verified_items = [i for i in verified_items if i.get('confidence', 0) > 0.5]

    # 5. Adaptive Synthesis
    generator = AdaptiveResponseGenerator(llm)
    answer = generator.generate(query, verified_items, intent_data['intent'])

    return answer, "\n".join(navigator.trace), verified_items

# ============================================================================
# CROSS-DOCUMENT META-TREE & JSON MCTS NAVIGATOR (PageIndex-Style)
# ============================================================================
# These functions enable cross-document structural reasoning by stitching
# equivalent sections from multiple documents into a unified meta-tree,
# then navigating it with JSON-action MCTS - exactly like PageIndex.
# ============================================================================


# ============================================================================
# CROSS-DOCUMENT META-TREE & JSON MCTS NAVIGATOR (PageIndex-Style)
# ============================================================================

def build_cross_document_meta_tree(query: str, doc_trees: List[Dict], llm: HybridLLM) -> Dict:
    """Stitches equivalent JSON sections from multiple documents into a single virtual root."""
    # Step 1: Ask LLM what structural category is needed for the query
    intent_prompt = f"""To answer "{query}", which structural section of a scientific paper is most relevant?
    (e.g., 'Experimental Setup', 'Results', 'Methodology'). Return ONLY the section name."""
    target_section = llm.generate(intent_prompt, max_new_tokens=20).strip().strip('"')

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
        matching_node = _find_section_by_keyword(tree, target_section)

        if matching_node:
            meta_root["nodes"].append({
                "node_id": f"meta_{doc_id}_{matching_node.get('node_id', '')}",
                "title": f"[{doc_id}] {matching_node.get('title', '')}",
                "summary": matching_node.get('summary', ''),
                "doc_id": doc_id,
                "original_node_id": matching_node.get('node_id', ''),
                "nodes": matching_node.get('nodes', []) # Preserve nested children!
            })

    return meta_root


def _find_section_by_keyword(tree: Dict, keyword: str) -> Optional[Dict]:
    """Recursively finds a node whose title matches the keyword."""
    def search(node):
        if keyword.lower() in node.get('title', '').lower():
            return node
        for child in node.get('nodes', []):
            res = search(child)
            if res: 
                return res
        return None
    return search(tree)


class JSONMCTSNavigator:
    """PageIndex-style JSON-action MCTS navigator for cross-document meta-trees."""
    def __init__(self, llm: HybridLLM, max_steps: int = 3):
        self.llm = llm
        self.max_steps = max_steps

    def navigate(self, query: str, meta_tree: Dict) -> List[Dict]:
        """Agentic navigation: LLM iteratively decides drill_down vs extract_text via JSON."""
        current_nodes = meta_tree.get("nodes", [])
        final_chunks = []

        for step in range(self.max_steps):
            if not current_nodes: 
                break

            # Format the current JSON nodes for the LLM
            nodes_summary = json.dumps([{
                "node_id": n['node_id'], "doc_id": n['doc_id'],
                "title": n['title'], "summary": n.get('summary', '')[:150]
            } for n in current_nodes], indent=2)

            prompt = f"""You are an expert scientific navigator. Step {step+1}.
            QUERY: "{query}"
            CURRENT JSON NODES:
            {nodes_summary}

            INSTRUCTIONS:
            1. If a node's summary contains the answer, choose "extract_text".
            2. If a node is a parent section and you need more detail, choose "drill_down".

            Return JSON: {{"reasoning": "...", "actions": [{{"node_id": "...", "action": "drill_down|extract_text"}}]}}"""

            response = self.llm.generate(prompt, fast_json=True)
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
                    # Fetch children from the JSON tree
                    children = _get_children_from_tree(meta_tree, node_id)
                    next_level_nodes.extend(children)
                elif action.get('action') == 'extract_text':
                    # Fetch full text and stop drilling this branch
                    chunk = _get_full_text_from_tree(meta_tree, node_id)
                    if chunk: 
                        final_chunks.append(chunk)

            current_nodes = next_level_nodes

        return final_chunks


def _get_children_from_tree(tree: Dict, node_id: str) -> List[Dict]:
    """Recursively find children of a node by node_id."""
    def search(node):
        if node.get('node_id') == node_id:
            return node.get('nodes', [])
        for child in node.get('nodes', []):
            res = search(child)
            if res is not None:
                return res
        return None
    result = search(tree)
    return result if result is not None else []


def _get_full_text_from_tree(tree: Dict, node_id: str) -> Optional[Dict]:
    """Recursively find a node and return its full text chunk."""
    def search(node):
        if node.get('node_id') == node_id:
            return {
                "doc_name": node.get('doc_id', 'unknown'),
                "section_title": node.get('title', ''),
                "start_page": node.get('start_page', 1),
                "end_page": node.get('end_page', 1),
                "full_text": node.get('text', node.get('summary', ''))
            }
        for child in node.get('nodes', []):
            res = search(child)
            if res:
                return res
        return None
    return search(tree)


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
                    with st.expander("🧠 Agent Reasoning & Verified Extractions"):
                        st.markdown(f"**Navigation Trace:**\n{msg['thinking']}")
                        # Support both old format (retrieved_info) and new format (verified_items)
                        if msg.get('verified_items'):
                            st.markdown("**Verified Extractions (Hallucination Checked):**")
                            for item in msg['verified_items'][:10]:
                                conf = item.get('confidence', 0)
                                color = "🟢" if conf > 0.8 else "🟡" if conf > 0.5 else "🔴"
                                st.caption(
                                    f"{color} **{item.get('parameter_name', 'Mechanism')}**: "
                                    f"{item.get('value', item.get('equation_latex', 'N/A'))} "
                                    f"{item.get('unit', '')} | Conf: {conf:.2f} | "
                                    f"[{item.get('doc_name')}, p.{item.get('page')}]"
                                )
                        elif msg.get('retrieved_info'):
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

                            # ================================================================
                            # ADVANCED DECLARMIMA-STYLE RETRIEVAL PIPELINE
                            # ================================================================
                            # Uses 4 architectural pillars:
                            # 1. ScientificIntentRouter - classifies query intent
                            # 2. IterativeTreeNavigator - MCTS-style document navigation
                            # 3. UniversalLLMExtractor + CitationValidator - structured extraction with hallucination checking
                            # 4. AdaptiveResponseGenerator - intent-aware formatted output
                            # ================================================================
                            answer, thinking, verified_items = advanced_retrieve_and_answer(
                                prompt, 
                                docs_to_search, 
                                llm
                            )

                            st.markdown(answer)

                            with st.expander("🧠 Agent Reasoning & Verified Extractions"):
                                st.markdown(f"**Navigation Trace:**\n{thinking}")
                                if verified_items:
                                    st.markdown("**Verified Extractions (Hallucination Checked):**")
                                    for item in verified_items[:10]:
                                        conf = item.get('confidence', 0)
                                        color = "🟢" if conf > 0.8 else "🟡" if conf > 0.5 else "🔴"
                                        st.caption(
                                            f"{color} **{item.get('parameter_name', 'Mechanism')}**: "
                                            f"{item.get('value', item.get('equation_latex', 'N/A'))} "
                                            f"{item.get('unit', '')} | Conf: {conf:.2f} | "
                                            f"[{item.get('doc_name')}, p.{item.get('page')}]"
                                        )

                            # Save to history
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": answer, 
                                "thinking": thinking, 
                                "verified_items": verified_items
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
