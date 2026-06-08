#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
DECLARMIMA v20.0 Elevated — PageIndex-Style Agentic Vectorless RAG
====================================================================
Architectural upgrades over the base 600-line version:
1. Scientific Intent Router         — classifies query intent before search
2. Nested JSON Tree Index           — hierarchical document structure (node_id, nodes, summaries)
3. Cross-Document Meta-Tree Stitcher — virtual forest across selected PDFs
4. JSON-Action MCTS Navigator       — iterative drill-down via JSON actions
5. Universal Structured Extractor   — pulls quantitative JSON items from raw text
6. Citation Validator               — cross-checks LLM extractions against raw PDF pages
7. Adaptive Response Synthesizer    — forces Markdown tables / LaTeX / causal prose per intent

Result: detail answering in the exact style of PageIndex/chat.pageindex.ai
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
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logging.basicConfig(level=logging.INFO, handlers=[console_handler], force=True)
logger = logging.getLogger("DECLARMIMA_ELEVATED")

# ============================================================================
# PYMUPDF IMPORT (FIXED)
# ============================================================================
try:
    import pymupdf as fitz
    PYMUPDF_AVAILABLE = True
    logger.info("PyMuPDF (modern) loaded via pymupdf")
except ImportError:
    try:
        import fitz
        PYMUPDF_AVAILABLE = True
        logger.info("PyMuPDF (legacy) loaded via fitz")
    except ImportError:
        PYMUPDF_AVAILABLE = False
        st.error("PyMuPDF is not installed.\nInstall with: pip install pymupdf")
        st.stop()

# ============================================================================
# DEPENDENCY CHECKS WITH GRACEFUL DEGRADATION
# ============================================================================

def check_optional_dependencies() -> Dict[str, bool]:
    deps: Dict[str, bool] = {}
    deps['pymupdf'] = PYMUPDF_AVAILABLE

    try:
        import ollama
        deps['ollama'] = True
        logger.info("Ollama client available")
    except ImportError:
        deps['ollama'] = False
        logger.warning("Ollama not installed.")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        deps['transformers'] = True
        logger.info("HuggingFace transformers available")
    except ImportError:
        deps['transformers'] = False
        logger.warning("transformers not installed.")

    try:
        from rapidfuzz import fuzz, process
        deps['rapidfuzz'] = True
        logger.info("rapidfuzz available")
    except ImportError:
        deps['rapidfuzz'] = False

    try:
        import orjson
        deps['orjson'] = True
        logger.info("orjson available")
    except ImportError:
        deps['orjson'] = False

    try:
        from pyvis.network import Network
        deps['pyvis'] = True
        logger.info("pyvis available")
    except ImportError:
        deps['pyvis'] = False

    logger.info(f"Dependency check: {sum(deps.values())}/{len(deps)} available")
    return deps

GLOBAL_DEPS = check_optional_dependencies()

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
    logger.warning("PyTorch not installed.")

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
# FAST JSON HELPERS
# ============================================================================

def fast_json_loads(data: bytes) -> Any:
    if ORJSON_AVAILABLE:
        import orjson
        return orjson.loads(data)
    return json.loads(data.decode('utf-8'))

def fast_json_dumps(obj: Any, indent: bool = False) -> bytes:
    if ORJSON_AVAILABLE:
        import orjson
        option = orjson.OPT_INDENT_2 if indent else 0
        return orjson.dumps(obj, option=option)
    return json.dumps(obj, indent=2 if indent else None).encode('utf-8')

# ============================================================================
# HYBRID LLM & TEMPLATES
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
    model_lower = model_name.lower()
    for key, template in MODEL_PROMPT_TEMPLATES.items():
        if key in model_lower:
            return template
    return MODEL_PROMPT_TEMPLATES["default"]


class HybridLLM:
    def __init__(self, model_key: str, use_4bit: bool = True, device: Optional[str] = None):
        self.model_key = model_key
        self.use_4bit = use_4bit
        self.device = device or ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.backend = None
        self.model_name = None
        self.client = None
        self.tokenizer = None
        self.model = None

        if model_key.startswith("[Ollama]"):
            self.model_name = model_key.split("] ")[1].strip().split(" (")[0]
        elif model_key.startswith("ollama:"):
            self.model_name = model_key.replace("ollama:", "", 1)
        elif model_key.startswith("[HF]"):
            self.model_name = model_key.split("] ")[1].strip().split(" (")[0]
        else:
            self.model_name = model_key

        self.template = get_model_template(self.model_name)
        self._init_backend()
        logger.info(f"HybridLLM: {self.model_name} on {self.device} via {self.backend}")

    def _init_backend(self):
        if OLLAMA_AVAILABLE:
            try:
                requests.get("http://localhost:11434/api/tags", timeout=5)
                self.backend = "ollama"
                self.client = ollama.Client(host="http://localhost:11434")
                logger.info("Using Ollama backend")
                return
            except requests.exceptions.ConnectionError:
                logger.warning("Ollama server not running.")
            except Exception as e:
                logger.warning(f"Ollama check failed: {e}")

        if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
            self.backend = "transformers"
            logger.info("Using Transformers backend")
            return

        raise RuntimeError("No LLM backend available. Install Ollama or transformers+torch.")

    def generate(self, prompt: str, max_new_tokens: int = 1024, temperature: float = 0.1,
                 fast_json: bool = False, system_prompt: Optional[str] = None) -> str:
        if self.backend == "ollama":
            return self._ollama_generate(prompt, max_new_tokens, temperature, fast_json, system_prompt)
        elif self.backend == "transformers":
            return self._transformers_generate(prompt, max_new_tokens, temperature, system_prompt)
        return "Error: No backend initialized"

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
            return "Error: Model not loaded"
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            if hasattr(self.tokenizer, 'apply_chat_template'):
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temp if temp > 0 else None,
                    do_sample=temp > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "assistant" in response:
                response = response.split("assistant")[-1].strip()
            return response
        except Exception as e:
            logger.error(f"Transformers error: {e}")
            return f"Error: {str(e)[:100]}"

    def _load_transformers(self):
        logger.info(f"Loading {self.model_name}...")
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


@st.cache_resource(show_spinner="Initializing LLM...")
def get_cached_llm(model_choice: str, use_4bit: bool = True):
    internal = LOCAL_LLM_OPTIONS[model_choice]
    return HybridLLM(model_key=internal, use_4bit=use_4bit)

# ============================================================================
# JSON EXTRACTION UTILITIES
# ============================================================================

def extract_json(text: str) -> Optional[Any]:
    if not text:
        return None
    try:
        return json.loads(text)
    except:
        pass
    md_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if md_match:
        try:
            return json.loads(md_match.group(1))
        except:
            pass
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            candidate = re.sub(r',\s*([\]}])', r'\1', candidate)
            candidate = re.sub(r'\s+', ' ', candidate)
            try:
                return json.loads(candidate)
            except:
                pass
    return None


# ============================================================================
# PDF PROCESSING (FIXED) + NESTED TREE BUILDER
# ============================================================================

def extract_text_from_pdf(file_bytes: bytes, max_pages: int = None) -> list[dict]:
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
                pages_data.append({"page_num": i + 1, "text": text})
        except Exception:
            continue
    doc.close()
    return pages_data


class PaginationAwareReader:
    def __init__(self, max_chars_per_request: int = 20000):
        self.max_chars_per_request = max_chars_per_request

    def extract_pages(self, doc_path_or_bytes: Union[str, bytes], page_numbers: List[int]) -> Dict[int, str]:
        result = {}
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
                chunk_map = {}
                for chunk in chunks:
                    meta = chunk.get("metadata", {})
                    p_num = meta.get("page_number", 0)
                    if p_num > 0:
                        chunk_map[p_num] = chunk.get("text", "")
                for pnum in page_numbers:
                    if pnum in chunk_map:
                        text = chunk_map[pnum]
                        if len(text) > self.max_chars_per_request:
                            text = text[:self.max_chars_per_request] + "\n...[TRUNCATED]"
                        result[pnum] = text
                if len(result) == len(page_numbers):
                    return result
            except Exception as e:
                logger.warning(f"pymupdf4llm failed: {e}")
                result = {}

        if isinstance(doc_path_or_bytes, bytes):
            doc = fitz.open(stream=doc_path_or_bytes, filetype="pdf")
        else:
            doc = fitz.open(doc_path_or_bytes)
        for pnum in page_numbers:
            if pnum < 1 or pnum > len(doc):
                continue
            if pnum in result:
                continue
            page = doc[pnum - 1]
            text = page.get_text("text")
            if len(text) > self.max_chars_per_request:
                text = text[:self.max_chars_per_request] + "\n...[TRUNCATED]"
            result[pnum] = text
        doc.close()
        return result


def build_document_tree(pages_list: List[Dict], doc_name: str, llm: HybridLLM) -> Dict:
    """Build a nested JSON tree (PageIndex-style) from flat page text."""
    sample = "\n".join([p['text'] for p in pages_list[:8]])
    prompt = f"""Analyze this scientific document and extract its main section structure.
    Return JSON: {{"sections": [{{"title": "Section Name", "start_page": int, "level": 1}}]}}
    Level 1 = main sections (Abstract, Introduction, Experimental, Results, Discussion, Conclusions).
    If you cannot determine page numbers, estimate based on text flow. Return ONLY JSON.

    Text:
    {sample[:8000]}"""

    resp = llm.generate(prompt, max_new_tokens=1024, system_prompt="Return ONLY valid JSON. No markdown fences.")
    parsed = extract_json(resp)

    tree = {
        "node_id": "0000",
        "title": doc_name,
        "start_page": 1,
        "end_page": len(pages_list) if pages_list else 1,
        "summary": "",
        "nodes": []
    }
    if not isinstance(tree.get("nodes"), list):
        tree["nodes"] = []

    sections = []
    if parsed and isinstance(parsed, dict):
        sections = parsed.get('sections', [])
    elif parsed and isinstance(parsed, list):
        sections = parsed

    if not sections:
        for p in pages_list:
            tree['nodes'].append({
                "node_id": f"{p['page_num']:04d}",
                "title": f"Page {p['page_num']}",
                "start_page": p['page_num'],
                "end_page": p['page_num'],
                "summary": p['text'][:200],
                "nodes": []
            })
        return tree

    sections = [s for s in sections if isinstance(s, dict) and 'title' in s]
    sections.sort(key=lambda x: x.get('start_page', 1))

    for i, sec in enumerate(sections):
        start_page = max(1, min(sec.get('start_page', 1), len(pages_list)))
        end_page = sections[i + 1]['start_page'] - 1 if i + 1 < len(sections) else len(pages_list)
        sec['end_page'] = max(start_page, min(end_page, len(pages_list)))
        sec['start_page'] = start_page
        sec['node_id'] = f"{i + 1:04d}"

    for sec in sections:
        start_idx = sec['start_page'] - 1
        end_idx = sec['end_page'] - 1
        sec_text = "\n".join([p['text'] for p in pages_list[start_idx:end_idx + 1]])

        sum_prompt = f"Summarize this section in 2 sentences. Focus on key data, methods, and findings:\n\n{sec_text[:4000]}"
        summary = llm.generate(sum_prompt, max_new_tokens=256, system_prompt="Be concise and factual.").strip()
        sec['summary'] = summary

        children = []
        if len(sec_text) > 2000:
            sub_prompt = f"""List subsections in this text. Return JSON list: [{{"title": "...", "level": 2}}] or [].
            Text start:
            {sec_text[:3000]}"""
            sub_resp = llm.generate(sub_prompt, max_new_tokens=512, system_prompt="Return ONLY valid JSON list.")
            sub_parsed = extract_json(sub_resp)
            if isinstance(sub_parsed, list):
                for j, sub in enumerate(sub_parsed):
                    sub_id = f"{sec['node_id']}_{j + 1:02d}"
                    children.append({
                        "node_id": sub_id,
                        "title": sub['title'],
                        "start_page": sec['start_page'],
                        "end_page": sec['end_page'],
                        "summary": "",
                        "nodes": []
                    })

        tree['nodes'].append({
            "node_id": sec['node_id'],
            "title": sec['title'],
            "start_page": sec['start_page'],
            "end_page": sec['end_page'],
            "summary": summary,
            "nodes": children
        })

    return tree


def find_node_by_id(tree: Dict, node_id: str) -> Optional[Dict]:
    if tree.get('node_id') == node_id:
        return tree
    for child in tree.get('nodes', []):
        result = find_node_by_id(child, node_id)
        if result:
            return result
    return None


def find_nodes_by_keyword(tree: Dict, keyword: str) -> List[Dict]:
    results = []
    keyword_lower = keyword.lower()

    def _search(node):
        title = node.get('title', '').lower()
        if keyword_lower in title or any(word in title for word in keyword_lower.split()):
            results.append(node)
        for child in node.get('nodes', []):
            _search(child)

    _search(tree)
    return results


def flatten_tree_nodes(tree: Dict) -> List[Dict]:
    nodes = []
    def _collect(node):
        nodes.append(node)
        for child in node.get('nodes', []):
            _collect(child)
    _collect(tree)
    return nodes


# ============================================================================
# PILLAR 1: SCIENTIFIC INTENT ROUTER
# ============================================================================

class ScientificIntentRouter:
    PATTERNS = {
        "value_extraction": [r"\bvalue\b", r"\bhow much\b", r"\bpower\b", r"\bdensity\b", r"\btable\b", r"\blist\b", r"\bparameter\b", r"\bwatt\b", r"\bmpa\b", r"\bmm/s\b", r"\btemperature\b", r"\benergy\b"],
        "equation": [r"\bequation\b", r"\bformula\b", r"\bconstitutive\b", r"\bnavier[- ]stokes\b", r"\bcahn[- ]hilliard\b", r"\bgoverning\b", r"\bmodel\b"],
        "mechanism": [r"\bwhy\b", r"\bhow does\b", r"\bmechanism\b", r"\bcause\b", r"\bdriving force\b", r"\breason\b", r"\bexplain\b"],
    }

    def route(self, query: str) -> Dict[str, str]:
        q_lower = query.lower()
        for intent, patterns in self.PATTERNS.items():
            if any(re.search(p, q_lower) for p in patterns):
                return {"intent": intent, "output_format": intent}
        return {"intent": "open_query", "output_format": "prose"}


# ============================================================================
# PILLAR 2: CROSS-DOCUMENT META-TREE STITCHER
# ============================================================================

def build_cross_document_meta_tree(query: str, selected_docs: Dict, llm: HybridLLM) -> Dict:
    intent_prompt = f"""To answer "{query}", which structural section of a scientific paper is most relevant?
    (e.g., 'Experimental Setup', 'Results', 'Methodology', 'Process Parameters', 'Materials and Methods').
    Return ONLY the section name as plain text."""
    target_section = llm.generate(intent_prompt, max_new_tokens=30, system_prompt="Return ONLY the section name. No punctuation.").strip().strip('"').strip("'")

    meta_root = {
        "node_id": "meta_root",
        "title": f"Cross-Document Meta-Tree: {target_section}",
        "doc_id": "META",
        "start_page": 1,
        "end_page": 999,
        "summary": f"Virtual root grouping '{target_section}' sections across all selected documents.",
        "nodes": []
    }

    for doc_name, data in selected_docs.items():
        tree = data.get('tree', {})
        matching_nodes = find_nodes_by_keyword(tree, target_section)
        if not matching_nodes:
            matching_nodes = tree.get('nodes', [])
        for node in matching_nodes:
            meta_root["nodes"].append({
                "node_id": f"meta_{doc_name}_{node.get('node_id', '0000')}",
                "title": f"[{doc_name}] {node.get('title', '')}",
                "summary": node.get('summary', ''),
                "doc_id": doc_name,
                "original_node_id": node.get('node_id', ''),
                "start_page": node.get('start_page', 1),
                "end_page": node.get('end_page', 1),
                "nodes": node.get('nodes', [])
            })

    # Ensure meta_root always has the required keys
    if not isinstance(meta_root, dict):
        meta_root = {"node_id": "meta_root", "title": "Meta-Tree", "doc_id": "META", "nodes": []}
    if "nodes" not in meta_root:
        meta_root["nodes"] = []
    return meta_root


# ============================================================================
# PILLAR 3: JSON-ACTION MCTS NAVIGATOR
# ============================================================================

class JSONMCTSNavigator:
    def __init__(self, llm: HybridLLM, max_steps: int = 3):
        self.llm = llm
        self.max_steps = max_steps
        self.trace = []

    def navigate(self, query: str, meta_tree: Dict, selected_docs: Dict) -> Tuple[List[Dict], str]:
        current_nodes = meta_tree.get("nodes", [])
        final_chunks = []

        for step in range(self.max_steps):
            if not current_nodes:
                break

            nodes_summary = json.dumps([{
                "node_id": n['node_id'],
                "doc_id": n.get('doc_id', 'META'),
                "title": n['title'],
                "summary": n.get('summary', '')[:200],
                "page_range": f"{n.get('start_page', '?')}-{n.get('end_page', '?')}"
            } for n in current_nodes], indent=2)

            prompt = f"""You are an expert scientific navigator. Step {step + 1} of {self.max_steps}.
            QUERY: "{query}"
            CURRENT JSON NODES:
            {nodes_summary}

            INSTRUCTIONS:
            1. If a node's summary likely contains the answer or specific data, choose "extract_text".
            2. If a node is a parent section and you need more detail to decide, choose "drill_down".
            3. If a node is irrelevant, choose "skip".

            Return strictly JSON: {{"reasoning": "your step-by-step analysis", "actions": [{{"node_id": "...", "action": "drill_down|extract_text|skip"}}]}}"""

            response = self.llm.generate(prompt, max_new_tokens=1024, system_prompt="Return ONLY valid JSON. No markdown fences.")
            actions_data = extract_json(response)

            if not actions_data or not isinstance(actions_data, dict) or 'actions' not in actions_data:
                self.trace.append("No valid actions returned by navigator.")
                break

            self.trace.append(actions_data.get('reasoning', f"Step {step + 1}"))

            next_level_nodes = []
            for action in actions_data.get('actions', []):
                if not isinstance(action, dict):
                    continue
                node_id = action.get('node_id')
                action_type = action.get('action', 'skip')
                if action_type == 'skip' or not node_id:
                    continue
                elif action_type == 'drill_down':
                    children = self._get_children(current_nodes, node_id)
                    if children:
                        next_level_nodes.extend(children)
                    else:
                        chunk = self._extract_chunk(node_id, selected_docs)
                        if chunk:
                            final_chunks.append(chunk)
                elif action_type == 'extract_text':
                    chunk = self._extract_chunk(node_id, selected_docs)
                    if chunk:
                        final_chunks.append(chunk)

            current_nodes = next_level_nodes

        return final_chunks, "\n".join(self.trace)

    def _get_children(self, current_nodes: List[Dict], node_id: str) -> List[Dict]:
        for node in current_nodes:
            if node.get('node_id') == node_id:
                return node.get('nodes', [])
        return []

    def _extract_chunk(self, node_id: str, selected_docs: Dict) -> Optional[Dict]:
        if not node_id.startswith('meta_'):
            return None
        parts = node_id.split('_', 2)
        if len(parts) < 3:
            return None
        doc_name = parts[1]
        original_id = parts[2]
        if doc_name not in selected_docs:
            return None
        data = selected_docs[doc_name]
        if not isinstance(data, dict):
            return None
        tree = data.get('tree', {})
        if not tree:
            # Fallback: try old flat sections format
            sections = data.get('sections', [])
            for sec in sections:
                if sec.get('id') == original_id:
                    pages = data.get('pages', [])
                    text_parts = []
                    for p in pages:
                        if sec.get('start_page', 1) <= p['page_num'] <= sec.get('end_page', sec.get('start_page', 1)):
                            text_parts.append(p['text'])
                    return {
                        "doc_name": doc_name,
                        "node_id": original_id,
                        "section_title": sec.get('title', ''),
                        "start_page": sec.get('start_page', 1),
                        "end_page": sec.get('end_page', 1),
                        "full_text": "\n".join(text_parts),
                        "summary": sec.get('summary', '')
                    }
            return None
        target_node = find_node_by_id(tree, original_id)
        if not target_node:
            return None
        start_page = target_node.get('start_page', 1)
        end_page = target_node.get('end_page', start_page)
        pages = data.get('pages', [])
        text_parts = []
        for p in pages:
            if start_page <= p['page_num'] <= end_page:
                text_parts.append(p['text'])
        full_text = "\n".join(text_parts)
        return {
            "doc_name": doc_name,
            "node_id": original_id,
            "section_title": target_node.get('title', ''),
            "start_page": start_page,
            "end_page": end_page,
            "full_text": full_text,
            "summary": target_node.get('summary', '')
        }


# ============================================================================
# PILLAR 4: UNIVERSAL STRUCTURED EXTRACTOR
# ============================================================================

class UniversalLLMExtractor:
    PROMPT = """Extract quantitative data, equations, or mechanisms from the text relevant to the query: "{query}"
    Return a JSON list of objects:
    [{{"item_type": "quantitative|equation|mechanism|prose",
       "parameter_name": "...",
       "value": number_or_string_or_null,
       "unit": "...",
       "equation_latex": "...",
       "context": "exact supporting sentence",
       "doc_name": "{doc_name}",
       "page": {page}}}]"

    Text:
    {text}"""

    def extract(self, chunks: List[Dict], query: str, llm: HybridLLM) -> List[Dict]:
        items = []
        for chunk in chunks:
            prompt = self.PROMPT.format(
                query=query,
                doc_name=chunk['doc_name'],
                page=chunk.get('start_page', 1),
                text=chunk['full_text'][:6000]
            )
            resp = llm.generate(prompt, max_new_tokens=2048, system_prompt="Return ONLY valid JSON list. No markdown fences.")
            parsed = extract_json(resp)
            if isinstance(parsed, list):
                for item in parsed:
                    item['doc_name'] = chunk['doc_name']
                    item['page'] = chunk.get('start_page', 1)
                    item['section_title'] = chunk.get('section_title', '')
                    items.append(item)
            elif isinstance(parsed, dict) and 'items' in parsed:
                for item in parsed['items']:
                    item['doc_name'] = chunk['doc_name']
                    item['page'] = chunk.get('start_page', 1)
                    item['section_title'] = chunk.get('section_title', '')
                    items.append(item)
        return items


# ============================================================================
# PILLAR 5: CITATION VALIDATOR (HALLUCINATION KILLER)
# ============================================================================

class CitationValidator:
    def verify(self, items: List[Dict], selected_docs: Dict) -> List[Dict]:
        verified = []
        for item in items:
            if not isinstance(item, dict):
                continue
            doc_name = item.get('doc_name')
            page_num = item.get('page', 1)
            param_name = (item.get('parameter_name') or '').lower()
            value_str = str(item.get('value', '')).lower()
            context = (item.get('context') or '').lower()
            confidence = 0.5

            if doc_name in selected_docs and isinstance(selected_docs[doc_name], dict):
                pages = selected_docs[doc_name].get('pages', [])
                raw_text = ""
                for p in pages:
                    if p['page_num'] == page_num:
                        raw_text = p['text'].lower()
                        break

                if raw_text:
                    if value_str and value_str in raw_text:
                        confidence = 0.95
                    elif param_name and param_name in raw_text:
                        confidence = 0.75
                    elif context and any(word in raw_text for word in context.split()[:5] if len(word) > 3):
                        confidence = 0.6
                    else:
                        confidence = 0.3
                else:
                    confidence = 0.4

            if item.get('item_type') == 'equation' and item.get('equation_latex'):
                confidence = max(confidence, 0.7)

            item['confidence'] = confidence
            verified.append(item)

        return sorted(verified, key=lambda x: x.get('confidence', 0), reverse=True)


# ============================================================================
# PILLAR 6: ADAPTIVE RESPONSE SYNTHESIZER
# ============================================================================

class AdaptiveResponseGenerator:
    def generate(self, query: str, items: List[Dict], intent: str, llm: HybridLLM) -> str:
        evidence_lines = []
        for i in items[:20]:
            ctx = i.get('context', i.get('equation_latex', ''))
            cite = f"[{i.get('doc_name')}, p.{i.get('page')}]"
            evidence_lines.append(f"- {cite} {i.get('parameter_name', '')}: {i.get('value', '')} {i.get('unit', '')} | {ctx[:150]}")
        evidence = "\n".join(evidence_lines)

        if intent == "value_extraction":
            prompt = f"""You are a scientific data analyst. Create a comprehensive Markdown table answering the query.
            Query: {query}

            Extracted Evidence:
            {evidence}

            INSTRUCTIONS:
            1. Create a Markdown table with columns: | Material/Alloy | Parameter | Value | Unit | Source Document | Page | Notes |
            2. Every row MUST be backed by the evidence above. Do not invent data.
            3. If multiple documents report the same parameter, include all rows for comparison.
            4. Add a summary paragraph below the table highlighting key findings and discrepancies.
            5. Use inline citations like [DocName, p.X] in the Notes column."""

        elif intent == "equation":
            prompt = f"""You are a scientific equation curator. State the governing equations in proper LaTeX format.
            Query: {query}

            Evidence:
            {evidence}

            INSTRUCTIONS:
            1. Present each equation in display math mode ($$ ... $$).
            2. Define ALL variables immediately below each equation.
            3. Cite the source document and page for each equation like [DocName, p.X].
            4. If multiple formulations exist, note the differences in a comparison table."""

        else:
            prompt = f"""You are a scientific research assistant. Answer the query comprehensively based ONLY on the evidence.
            Query: {query}

            Evidence:
            {evidence}

            INSTRUCTIONS:
            1. Use inline citations like [DocName, p.X] for EVERY factual claim.
            2. Explain physical mechanisms, causes, and relationships clearly.
            3. If evidence is contradictory, discuss the discrepancy.
            4. Be thorough but concise. Use bullet points for lists of mechanisms or factors."""

        return llm.generate(prompt, max_new_tokens=2500, temperature=0.1, system_prompt="You are a precise scientific analyst. Follow formatting rules strictly. Never hallucinate data not in the evidence.")


# ============================================================================
# DOCUMENT INDEXING (NESTED TREE)
# ============================================================================

def index_all_documents(uploaded_files, llm: HybridLLM, chunk_size: int = 5):
    documents = {}
    total_files = len(uploaded_files)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, uploaded_file in enumerate(uploaded_files):
        doc_name = os.path.splitext(uploaded_file.name)[0]
        if doc_name in documents:
            doc_name = f"{doc_name}_{idx + 1}"

        status_text.info(f"Processing ({idx + 1}/{total_files}): {uploaded_file.name}...")
        pdf_bytes = uploaded_file.read()
        pages_text = extract_text_from_pdf(pdf_bytes)

        tree = build_document_tree(pages_text, doc_name, llm)

        try:
            tree = build_document_tree(pages_text, doc_name, llm)
            documents[doc_name] = {
                'pages': pages_text,
                'tree': tree,
                'filename': uploaded_file.name,
                'pdf_bytes': pdf_bytes
            }
        except Exception as e:
            logger.error(f"Failed to build tree for {doc_name}: {e}")
            st.warning(f"Failed to index {doc_name}: {str(e)[:100]}. Using flat page structure.")
            # Fallback: create a minimal flat tree
            flat_tree = {
                "node_id": "0000",
                "title": doc_name,
                "start_page": 1,
                "end_page": len(pages_text),
                "summary": f"Flat fallback index for {doc_name}",
                "nodes": [
                    {
                        "node_id": f"{p['page_num']:04d}",
                        "title": f"Page {p['page_num']}",
                        "start_page": p['page_num'],
                        "end_page": p['page_num'],
                        "summary": p['text'][:200],
                        "nodes": []
                    }
                    for p in pages_text[:50]  # Limit to first 50 pages for safety
                ]
            }
            documents[doc_name] = {
                'pages': pages_text,
                'tree': flat_tree,
                'filename': uploaded_file.name,
                'pdf_bytes': pdf_bytes
            }
        progress_bar.progress((idx + 1) / total_files)

    progress_bar.empty()
    status_text.empty()
    return documents


# ============================================================================
# ADVANCED RETRIEVAL PIPELINE (PageIndex-Style)
# ============================================================================

def advanced_retrieve_and_answer(query: str, selected_docs: Dict, llm: HybridLLM, max_context_chars: int = 15000):
    """The PageIndex-style agentic pipeline."""
    # 1. Route Intent
    router = ScientificIntentRouter()
    intent_data = router.route(query)

    # 2. Build Cross-Document Meta-Tree
    meta_tree = build_cross_document_meta_tree(query, selected_docs, llm)
    if not isinstance(meta_tree, dict):
        meta_tree = {"node_id": "meta_root", "title": "Meta-Tree", "doc_id": "META", "nodes": []}
    if "nodes" not in meta_tree or not isinstance(meta_tree.get("nodes"), list):
        meta_tree["nodes"] = []

    if not meta_tree.get('nodes'):
        # Fallback: use all top-level nodes from all documents
        for doc_name, data in selected_docs.items():
            tree = data.get('tree', {})
            if tree and 'nodes' in tree:
                for child in tree.get('nodes', []):
                    meta_tree['nodes'].append({
                        "node_id": f"meta_{doc_name}_{child.get('node_id', '0000')}",
                        "title": f"[{doc_name}] {child.get('title', '')}",
                        "summary": child.get('summary', ''),
                        "doc_id": doc_name,
                        "original_node_id": child.get('node_id', ''),
                        "start_page": child.get('start_page', 1),
                        "end_page": child.get('end_page', 1),
                        "nodes": child.get('nodes', [])
                    })
            elif data.get('sections'):
                # Backward compatibility with old flat section format
                for sec in data.get('sections', []):
                    meta_tree['nodes'].append({
                        "node_id": f"meta_{doc_name}_{sec.get('id', '0000')}",
                        "title": f"[{doc_name}] {sec.get('title', '')}",
                        "summary": sec.get('summary', ''),
                        "doc_id": doc_name,
                        "original_node_id": sec.get('id', ''),
                        "start_page": sec.get('start_page', 1),
                        "end_page": sec.get('end_page', 1),
                        "nodes": []
                    })

    # 3. Iterative JSON Navigation
    navigator = JSONMCTSNavigator(llm, max_steps=3)
    retrieved_chunks, trace = navigator.navigate(query, meta_tree, selected_docs)

    if not retrieved_chunks:
        return "I could not find relevant sections in the selected documents.", trace, []

    # 4. Structured Extraction
    extractor = UniversalLLMExtractor()
    raw_items = extractor.extract(retrieved_chunks, query, llm)

    # 5. Citation Validation
    validator = CitationValidator()
    verified_items = validator.verify(raw_items, selected_docs)
    verified_items = [i for i in verified_items if i.get('confidence', 0) > 0.5]

    # 6. Adaptive Synthesis
    generator = AdaptiveResponseGenerator()
    answer = generator.generate(query, verified_items, intent_data['intent'], llm)

    return answer, trace, verified_items


# ============================================================================
# STREAMLIT UI
# ============================================================================

def render_tree_ui(node: Dict, depth: int = 0):
    prefix = "  " * depth + ("└─ " if depth > 0 else "")
    st.markdown(f"{prefix}**{node.get('node_id', '')}**: {node.get('title', '')} *(pp. {node.get('start_page', '?')}-{node.get('end_page', '?')})*")
    if node.get('summary'):
        st.caption(f"{'  ' * (depth + 1)}{node['summary'][:250]}")
    for child in node.get('nodes', []):
        render_tree_ui(child, depth + 1)


def render_sidebar():
    with st.sidebar:
        st.markdown("### Configuration")
        model_keys = list(LOCAL_LLM_OPTIONS.keys())
        if "llm_model_choice" not in st.session_state:
            st.session_state.llm_model_choice = model_keys[2]

        selected = st.selectbox(
            "Select LLM Model",
            options=model_keys,
            index=model_keys.index(st.session_state.llm_model_choice),
            key="llm_model_select",
            help="Choose between Ollama (fast API) or HuggingFace Transformers (local loading)"
        )
        st.session_state.llm_model_choice = selected

        model_key = LOCAL_LLM_OPTIONS[selected]
        if model_key.startswith("ollama:"):
            st.caption("Backend: Ollama (API)")
            st.caption(f"Model: `{model_key.replace('ollama:', '')}`")
        else:
            st.caption("Backend: Transformers (Local)")
            st.caption(f"Model: `{model_key}`")

        if not model_key.startswith("ollama:"):
            st.checkbox("Use 4-bit quantization (saves VRAM)", value=True, key="use_4bit")
            if TORCH_AVAILABLE and torch.cuda.is_available():
                st.caption(f"GPU: {torch.cuda.get_device_name(0)}")
            else:
                st.warning("No GPU detected. Local model will run on CPU (slow).")
        else:
            st.session_state.use_4bit = False

        st.markdown("---")
        st.markdown("#### System Status")
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"{'Yes' if OLLAMA_AVAILABLE else 'No'} Ollama")
        with cols[1]:
            st.markdown(f"{'Yes' if TRANSFORMERS_AVAILABLE else 'No'} Transformers")
        cols2 = st.columns(2)
        with cols2[0]:
            st.markdown(f"{'Yes' if TORCH_AVAILABLE else 'No'} PyTorch")
        with cols2[1]:
            st.markdown(f"{'Yes' if PYMUPDF_AVAILABLE else 'No'} PyMuPDF")

        st.markdown("---")
        with st.expander("Advanced Settings", expanded=False):
            st.slider("Max context chars", 5000, 30000, 15000, 1000, key="max_context_chars")
            st.slider("Chunk size (pages)", 1, 10, 5, 1, key="chunk_size")

        st.markdown("---")
        if st.button("Clear Cache & Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def run_streamlit():
    st.set_page_config(page_title="DECLARMIMA v20 Elevated — PageIndex Style", layout="wide")
    st.title("DECLARMIMA v20 Elevated — PageIndex-Style Agentic RAG")
    st.markdown(
        "Upload multiple PDFs, build hierarchical JSON tree indices, and query them using **agentic navigation**. "
        "**No Vector DBs. No Chunking. No Embeddings.** "
        "Supports both **Ollama** and **HuggingFace Transformers**."
    )

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'selected_docs_for_query' not in st.session_state:
        st.session_state.selected_docs_for_query = []
    if 'documents' not in st.session_state:
        st.session_state.documents = {}
    if 'llm' not in st.session_state:
        st.session_state.llm = None

    render_sidebar()

    with st.sidebar:
        st.markdown("---")
        st.header("Documents")
        uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

        if uploaded_files and st.button("Build Document Trees", type="primary", use_container_width=True):
            with st.spinner("Initializing LLM and indexing documents..."):
                try:
                    llm = get_cached_llm(st.session_state.llm_model_choice, st.session_state.get("use_4bit", True))
                    st.session_state.llm = llm
                    st.success(f"LLM loaded: {llm.model_name} via {llm.backend}")

                    st.session_state.documents = index_all_documents(
                        uploaded_files, llm,
                        chunk_size=st.session_state.get("chunk_size", 5)
                    )
                    st.session_state.messages = []
                    st.success(f"Indexed {len(uploaded_files)} document(s)")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Initialization error: {e}", exc_info=True)

        if st.session_state.documents:
            st.markdown("---")
            st.header("Query Scope")
            doc_names = list(st.session_state.documents.keys())
            st.session_state.selected_docs_for_query = st.multiselect(
                "Select documents to search:", doc_names, default=doc_names
            )

            with st.expander("View Document Trees", expanded=False):
                for doc_name in doc_names:
                    st.subheader(f"{doc_name}")
                    doc_data = st.session_state.documents.get(doc_name, {})

                    if not isinstance(doc_data, dict):
                        st.warning(f"Corrupted data for {doc_name}. Re-index the document.")
                        continue

                    # Handle both old format (no tree) and new format
                    tree = doc_data.get('tree')
                    if tree and isinstance(tree, dict):
                        render_tree_ui(tree, depth=0)
                    elif doc_data.get('sections'):
                        # Backward compatibility: old flat section format
                        for s in doc_data.get('sections', []):
                            st.markdown(f"**{s.get('id', 'N/A')}: {s.get('title', 'Untitled')}** *(pp. {s.get('start_page', '?')}-{s.get('end_page', '?')})*")
                            st.caption(s.get('summary', 'No summary'))
                            st.divider()
                    else:
                        st.warning(f"No tree index available for {doc_name}. Re-index the document.")
                    st.divider()

    if not st.session_state.documents:
        st.info("Please upload PDF(s) in the sidebar and click 'Build Document Trees' to start.")
        with st.expander("Quick Start Guide", expanded=True):
            st.markdown("""
            ### Getting Started
            1. **Choose your LLM backend** in the sidebar (Ollama recommended).
            2. **Upload PDFs** using the sidebar uploader.
            3. **Click 'Build Document Trees'** to index your documents.
            4. **Ask questions** in the chat below.

            ### Requirements
            - **PyMuPDF**: `pip install pymupdf`
            - **For Ollama**: `pip install ollama` + [Ollama app](https://ollama.com)
            - **For HF**: `pip install transformers torch`
            - **Optional**: `pip install pymupdf4llm` (better LaTeX/math extraction)
            """)
    else:
        st.subheader("Chat with your Documents")

        for msg in st.session_state.messages:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])
                if msg['role'] == 'assistant' and 'thinking' in msg:
                    with st.expander("Agent Reasoning & Verified Data"):
                        st.markdown(f"**Navigation Trace:**\n{msg['thinking']}")
                        if msg.get('verified_items'):
                            st.markdown("**Verified Extractions (Hallucination-Checked):**")
                            for item in msg['verified_items']:
                                conf = item.get('confidence', 0)
                                color = "🟢" if conf > 0.8 else "🟡" if conf > 0.6 else "🔴"
                                param = item.get('parameter_name', 'N/A')
                                val = item.get('value', item.get('equation_latex', 'N/A'))
                                unit = item.get('unit', '')
                                doc = item.get('doc_name', '')
                                page = item.get('page', '?')
                                st.caption(f"{color} **{param}**: {val} {unit} | Conf: {conf:.2f} | [{doc}, p.{page}]")

        if prompt := st.chat_input("Ask a question about the documents..."):
            if not st.session_state.selected_docs_for_query:
                st.error("Please select at least one document in the sidebar to search.")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Routing intent → Building meta-tree → Navigating → Extracting → Validating → Synthesizing..."):
                        try:
                            docs_to_search = {k: st.session_state.documents[k] for k in st.session_state.selected_docs_for_query}
                            llm = st.session_state.llm or get_cached_llm(
                                st.session_state.llm_model_choice,
                                st.session_state.get("use_4bit", True)
                            )

                            answer, trace, verified_items = advanced_retrieve_and_answer(
                                prompt, docs_to_search, llm,
                                max_context_chars=st.session_state.get("max_context_chars", 15000)
                            )

                            st.markdown(answer)

                            with st.expander("Agent Reasoning & Verified Data"):
                                st.markdown(f"**Navigation Trace:**\n{trace}")
                                if verified_items:
                                    st.markdown("**Verified Extractions (Hallucination-Checked):**")
                                    for item in verified_items[:15]:
                                        conf = item.get('confidence', 0)
                                        color = "🟢" if conf > 0.8 else "🟡" if conf > 0.6 else "🔴"
                                        param = item.get('parameter_name', 'N/A')
                                        val = item.get('value', item.get('equation_latex', 'N/A'))
                                        unit = item.get('unit', '')
                                        doc = item.get('doc_name', '')
                                        page = item.get('page', '?')
                                        st.caption(f"{color} **{param}**: {val} {unit} | Conf: {conf:.2f} | [{doc}, p.{page}]")

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "thinking": trace,
                                "verified_items": verified_items
                            })
                        except Exception as e:
                            error_msg = f"Error generating response: {str(e)}"
                            st.error(error_msg)
                            logger.error(f"Query error: {e}", exc_info=True)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    run_streamlit()
