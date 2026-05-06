#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DECLARMIMA v8.0 – Vectorless RAG + LLM Reasoning
=================================================
- Hierarchical PDF index (no vectors)
- Dropdown for local LLMs (Ollama / Transformers)
- Extracts values via strict regex, then uses LLM to explain
- Outputs JSON and Markdown with citations
"""

import streamlit as st
import asyncio
import json
import re
import os
import tempfile
import time
import hashlib
import pickle
import logging
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch

# ----------------------------------------------------------------------
# PDF processing (PyMuPDF required)
try:
    import fitz
except ImportError:
    raise ImportError("PyMuPDF (fitz) required: pip install pymupdf")

# Optional LLM backends
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("DECLARMIMA")

# ============================================================================
# 1. Pydantic Models for Structured Output
# ============================================================================
from pydantic import BaseModel, Field, field_validator

class ExtractedValue(BaseModel):
    """Single extracted measurement (non‑zero, valid unit, context)."""
    query: str
    value: float
    unit: str
    confidence: float = Field(ge=0.0, le=1.0)
    context: str
    doc_name: str
    page: int
    section_title: Optional[str] = None
    material: Optional[str] = None

    @field_validator('value')
    def non_zero(cls, v):
        if v == 0.0:
            raise ValueError("Zero values are always false positives")
        return v

    @field_validator('unit')
    def valid_unit(cls, v):
        allowed = {"W", "kW", "mW", "MW", "W/cm", "kW/cm", "W/cm²", "kW/cm²",
                   "W/cm2", "kW/cm2", "W/m²", "kW/m²", "W/m2", "kW/m2"}
        if not any(v.startswith(u) for u in allowed):
            raise ValueError(f"Invalid unit: {v}")
        return v

    def citation(self) -> str:
        return f'<cite doc="{self.doc_name}" page="{self.page}"/>'

    def to_dict(self):
        return self.model_dump()

class QueryReport(BaseModel):
    query: str
    total_docs: int
    docs_with_results: int
    docs_without_results: List[str]
    all_values: List[ExtractedValue]
    consensus: Dict[str, Any]
    processing_time_sec: float

    def to_json(self, indent=2) -> str:
        return json.dumps(self.model_dump(), indent=indent, ensure_ascii=False)

# ============================================================================
# 2. Hierarchical PDF Index (Vectorless)
# ============================================================================
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
    _pdf_path: Optional[str] = None

    def get_text(self, doc_cache: Dict[str, Any] = None) -> str:
        if self.full_text:
            return self.full_text
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
        return self.full_text

    def to_dict(self):
        return {"id": self.id, "title": self.title, "page_start": self.page_start,
                "page_end": self.page_end, "summary": self.summary, "level": self.level,
                "doc_id": self.doc_id, "section_type": self.section_type,
                "children": [c.to_dict() for c in self.children]}

    @classmethod
    def from_dict(cls, data: dict, pdf_path=None):
        node = cls(data["id"], data["title"], data["page_start"], data["page_end"], "",
                   data["summary"], data["level"], doc_id=data["doc_id"],
                   section_type=data["section_type"], _pdf_path=pdf_path)
        for c in data.get("children", []):
            node.children.append(cls.from_dict(c, pdf_path))
        return node

class HierarchicalIndex:
    def __init__(self, cache_dir=".declarmima_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.doc_trees: Dict[str, PageNode] = {}
        self._pdf_cache = {}

    def _doc_hash(self, file_buffer: BytesIO) -> str:
        pos = file_buffer.tell()
        file_buffer.seek(0)
        content = file_buffer.read(1024*1024)
        file_buffer.seek(pos)
        return hashlib.sha256(content).hexdigest()[:16]

    def _cache_path(self, doc_name: str, doc_hash: str) -> Path:
        safe = re.sub(r'[^\w\-_.]', '_', doc_name)
        return self.cache_dir / f"{safe}.{doc_hash}.tree.pkl"

    def build_from_pdfs(self, files: List, parallel=True, max_workers=4):
        def build_one(file):
            doc_name = file.name
            buf = BytesIO(file.getbuffer())
            doc_hash = self._doc_hash(buf)
            cache_path = self._cache_path(doc_name, doc_hash)
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        root_data = pickle.load(f)
                    root = PageNode.from_dict(root_data)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        buf.seek(0)
                        tmp.write(buf.getbuffer())
                        root._pdf_path = tmp.name
                    return doc_name, root
                except:
                    pass
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                buf.seek(0)
                tmp.write(buf.getbuffer())
                tmp_path = tmp.name
            doc = fitz.open(tmp_path)
            root = self._build_tree(doc, doc_name, tmp_path)
            doc.close()
            try:
                cache_root = self._clone_for_cache(root)
                with open(cache_path, "wb") as f:
                    pickle.dump(cache_root.to_dict(), f)
            except:
                pass
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
        root = PageNode(f"{doc_id}_root", "Document Root", 1, len(doc), "", doc_id, 0, doc_id=doc_id, _pdf_path=pdf_path)
        toc = doc.get_toc()
        if toc:
            nodes_by_level = {}
            for level, title, page in toc:
                if page > len(doc): continue
                end = min(page+3, len(doc))
                text = self._extract_range(doc, page, end)
                node = PageNode(f"{doc_id}_toc_{level}_{title[:20]}", title.strip(), page, end,
                                text, text[:200], level, doc_id=doc_id, _pdf_path=pdf_path)
                nodes_by_level.setdefault(level, []).append(node)
            for level in sorted(nodes_by_level.keys()):
                for node in nodes_by_level[level]:
                    parent = self._find_parent(root, level-1, node.page_start)
                    parent.children.append(node)
            return root
        headings = self._detect_headings(doc)
        if headings:
            for i, (title, page) in enumerate(headings):
                end = min(page+3, len(doc))
                text = self._extract_range(doc, page, end)
                node = PageNode(f"{doc_id}_h{i}", title, page, end, text, text[:200], 2, doc_id=doc_id, _pdf_path=pdf_path)
                root.children.append(node)
            return root
        # fallback: each page as leaf
        for p in range(1, len(doc)+1):
            text = doc[p-1].get_text("text")
            if not text.strip(): continue
            node = PageNode(f"{doc_id}_p{p}", f"Page {p}", p, p, text, text[:200], 3, doc_id=doc_id, _pdf_path=pdf_path)
            root.children.append(node)
        return root

    def _extract_range(self, doc, start, end):
        return "\n\n".join(doc[p-1].get_text("text") for p in range(start, min(end, len(doc))))

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

    def _clone_for_cache(self, node):
        return PageNode(node.id, node.title, node.page_start, node.page_end, "",
                        node.summary, node.level, doc_id=node.doc_id,
                        section_type=node.section_type,
                        children=[self._clone_for_cache(c) for c in node.children])

    def cleanup(self):
        for doc in self._pdf_cache.values():
            try: doc.close()
            except: pass
        self._pdf_cache.clear()

# ============================================================================
# 3. Strict Regex Extractor (no hallucinations)
# ============================================================================
class StrictExtractor:
    def __init__(self, query: str):
        self.query = query.lower()
        self.allowed_units = {"W", "kW", "mW", "MW", "W/cm", "kW/cm", "W/cm²", "kW/cm²",
                              "W/cm2", "kW/cm2", "W/m²", "kW/m²", "W/m2", "kW/m2"}
        self.patterns = [
            rf'(?:{self.query})\s*[=:]\s*(\d+(?:\.\d+)?)\s*([a-zA-Z²0-9/]+)',
            rf'(?:{self.query})\s+of\s+(\d+(?:\.\d+)?)\s*([a-zA-Z²0-9/]+)',
            rf'(\d+(?:\.\d+)?)\s*([a-zA-Z²0-9/]+)\s+(?:{self.query})',
            r'power\s*[=:]\s*(\d+(?:\.\d+)?)\s*([WkWMm]{1,3})',
            r'P\s*[=:]\s*(\d+(?:\.\d+)?)\s*([WkWMm]{1,3})',
            r'(\d+(?:\.\d+)?)([WkWMm]{1,3})\b',
        ]

    def _is_valid(self, value: float, unit: str, context: str) -> bool:
        if value == 0.0:
            return False
        if not any(unit.startswith(u) for u in self.allowed_units):
            return False
        lower_ctx = context.lower()
        if self.query not in lower_ctx and "laser" not in lower_ctx:
            return False
        # additionally require a power phrase
        power_phrases = ["power", "laser power", "beam power", "irradiance"]
        if not any(p in lower_ctx for p in power_phrases):
            return False
        return True

    def extract_from_text(self, text: str, doc_name: str, page: int, section_title: str) -> List[ExtractedValue]:
        results = []
        for pattern in self.patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    groups = match.groups()
                    if len(groups) == 2:
                        val_str, unit = groups
                    else:
                        val_str = groups[0]
                        unit = ""
                    value = float(val_str)
                    unit = unit.strip().upper().replace("²", "2")
                    start = max(0, match.start() - 80)
                    end = min(len(text), match.end() + 80)
                    context = text[start:end].strip()
                    if not self._is_valid(value, unit, context):
                        continue
                    # exact sentence
                    sentences = re.split(r'(?<=[.!?])\s+', text)
                    exact = ""
                    for sent in sentences:
                        if match.group(0) in sent:
                            exact = sent.strip()
                            break
                    if not exact:
                        exact = context[:200]
                    material = None
                    for mat in ["AlSiMg", "SDSS", "Ti6Al4V", "Ti-Cr", "Cu6Sn5", "Al-Cu-Ni", "Ti3Au", "Inconel"]:
                        if mat in exact:
                            material = mat
                            break
                    results.append(ExtractedValue(
                        query=self.query,
                        value=value,
                        unit=unit,
                        confidence=0.95,
                        context=exact,
                        doc_name=doc_name,
                        page=page,
                        section_title=section_title,
                        material=material
                    ))
                except (ValueError, IndexError):
                    continue
        # deduplicate
        unique = {}
        for v in results:
            key = (v.value, v.unit, v.page, v.doc_name)
            if key not in unique or v.confidence > unique[key].confidence:
                unique[key] = v
        return list(unique.values())

# ============================================================================
# 4. LLM Reasoning Engine (supports Ollama & Transformers)
# ============================================================================
class LLMReasoner:
    def __init__(self, model_choice: str, use_4bit: bool = True):
        self.model_choice = model_choice
        self.use_4bit = use_4bit
        self.backend = None
        self.model_name = None
        self.tokenizer = None
        self.model = None
        self.client = None
        self._init()

    def _init(self):
        # determine if Ollama or HF
        if "[Ollama]" in self.model_choice:
            self.backend = "ollama"
            self.model_name = self.model_choice.split("] ")[1].strip()
            if OLLAMA_AVAILABLE:
                self.client = ollama.Client(host="http://localhost:11434")
            else:
                raise RuntimeError("Ollama not available")
        else:
            self.backend = "transformers"
            self.model_name = self.model_choice
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError("Transformers not available")
            # we lazy load later

    def _load_transformers(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading {self.model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.float16 if device=="cuda" else torch.float32}
        if self.use_4bit and device=="cuda":
            try:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            except:
                pass
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if device == "cuda":
            self.model.to(device)
        self.model.eval()
        logger.info("Model loaded.")

    def generate(self, prompt: str, max_tokens=1024, temperature=0.1) -> str:
        if self.backend == "ollama":
            try:
                resp = self.client.generate(model=self.model_name, prompt=prompt,
                                            options={"num_predict": max_tokens, "temperature": temperature})
                return resp.get("response", "").strip()
            except Exception as e:
                return f"Ollama error: {e}"
        else:
            if self.tokenizer is None:
                self._load_transformers()
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature,
                                             do_sample=temperature>0, pad_token_id=self.tokenizer.eos_token_id)
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # remove prompt from answer
            if "assistant" in answer:
                answer = answer.split("assistant")[-1].strip()
            return answer

# ============================================================================
# 5. Parallel Extraction (no LLM for values, just regex)
# ============================================================================
class ParallelExtractor:
    def __init__(self, index: HierarchicalIndex):
        self.index = index

    async def extract(self, query: str) -> QueryReport:
        start = time.time()
        extractor = StrictExtractor(query)
        all_vals = []
        docs_with = 0
        docs_without = []
        for doc_name, root in self.index.doc_trees.items():
            values = await self._extract_doc(doc_name, root, extractor)
            if values:
                docs_with += 1
                all_vals.extend(values)
            else:
                docs_without.append(doc_name)
        # consensus on laser power
        power_vals = [v for v in all_vals if any(v.unit.startswith(u) for u in ["W","kW","mW"])]
        consensus = {}
        if power_vals:
            nums = [v.value for v in power_vals]
            unit_counts = Counter(v.unit for v in power_vals)
            most_common_unit = unit_counts.most_common(1)[0][0]
            consensus = {
                "count": len(power_vals),
                "mean": float(np.mean(nums)),
                "std": float(np.std(nums)),
                "min": float(np.min(nums)),
                "max": float(np.max(nums)),
                "unit": most_common_unit,
                "sources": list(set(v.doc_name for v in power_vals))
            }
        elapsed = time.time() - start
        return QueryReport(
            query=query,
            total_docs=len(self.index.doc_trees),
            docs_with_results=docs_with,
            docs_without_results=docs_without,
            all_values=all_vals,
            consensus=consensus,
            processing_time_sec=elapsed
        )

    async def _extract_doc(self, doc_name: str, root: PageNode, extractor: StrictExtractor):
        def traverse(node: PageNode):
            vals = []
            if not node.children:
                text = node.get_text(self.index._pdf_cache)
                if text:
                    vals.extend(extractor.extract_from_text(text, doc_name, node.page_start, node.title))
            else:
                for c in node.children:
                    vals.extend(traverse(c))
            return vals
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, traverse, root)

# ============================================================================
# 6. Streamlit UI with LLM Reasoning
# ============================================================================
def run_streamlit():
    st.set_page_config(page_title="DECLARMIMA v8 – LLM Reasoning", layout="wide")
    st.title("🔬 DECLARMIMA v8 – Vectorless RAG + LLM Reasoning")
    st.caption("Extract values (laser power, etc.) then ask the LLM to explain the results across documents.")

    # Session state for files and index
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = None
    if "index_built" not in st.session_state:
        st.session_state.index_built = False
    if "index" not in st.session_state:
        st.session_state.index = None
    if "llm_reasoner" not in st.session_state:
        st.session_state.llm_reasoner = None

    # Sidebar: model selection (dropdown)
    with st.sidebar:
        st.markdown("### 🧠 LLM Model")
        # Available models – both Ollama and HF names
        model_options = [
            "[Ollama] qwen2.5:0.5b (Fastest)",
            "[Ollama] qwen2.5:7b (Recommended)",
            "[Ollama] qwen2.5:14b (High VRAM)",
            "[Ollama] llama3.1:8b",
            "[Ollama] mistral:7b",
            "[Ollama] gemma2:9b",
            "[Ollama] falcon3:10b",
            "Qwen/Qwen2.5-7B-Instruct (HF)",
            "meta-llama/Llama-3.1-8B-Instruct (HF)",
            "mistralai/Mistral-7B-Instruct-v0.3 (HF)"
        ]
        selected_model = st.selectbox("Choose LLM for reasoning", model_options, index=1)
        use_4bit = st.checkbox("Use 4‑bit quantization (HF only)", value=True)

        # File uploader
        uploaded = st.file_uploader("Upload PDF files (once)", type="pdf", accept_multiple_files=True)

        if uploaded:
            st.session_state.uploaded_files = uploaded
            st.session_state.index_built = False
            st.session_state.index = None
            st.rerun()

        if st.button("🗑️ Clear all files"):
            st.session_state.uploaded_files = None
            st.session_state.index_built = False
            st.session_state.index = None
            st.rerun()

        # Build index button (only when files are present)
        if st.session_state.uploaded_files and not st.session_state.index_built:
            if st.button("📚 Build Index (one time)", type="primary"):
                with st.spinner("Building hierarchical index..."):
                    idx = HierarchicalIndex()
                    idx.build_from_pdfs(st.session_state.uploaded_files, parallel=True)
                    st.session_state.index = idx
                    st.session_state.index_built = True
                st.success("Index built! You can now ask questions.")
                st.rerun()

    # Main area
    if st.session_state.index_built:
        # Query input
        query = st.text_input("Enter your query (e.g., 'laser power', 'scan speed')", "laser power")
        col1, col2 = st.columns([1, 3])
        with col1:
            extract_btn = st.button("🔍 Extract Values", type="primary")
        with col2:
            explain_btn = st.button("🧠 Ask LLM to Explain Results", disabled=not st.session_state.get("last_report", None))

        if extract_btn:
            with st.spinner("Extracting values from documents..."):
                extractor = ParallelExtractor(st.session_state.index)
                report = asyncio.run(extractor.extract(query))
                st.session_state.last_report = report
            st.success(f"✅ Found {report.docs_with_results} documents with relevant values")
            st.json(report.to_json())
            # download JSON
            st.download_button("📥 Download JSON", report.to_json(), "extracted_values.json", "application/json")

        if explain_btn and st.session_state.get("last_report"):
            report = st.session_state.last_report
            # build a prompt for the LLM
            if report.docs_with_results == 0:
                st.info("No values extracted. Ask a different query or upload other PDFs.")
            else:
                power_vals = [v for v in report.all_values if any(v.unit.startswith(u) for u in ["W","kW","mW"])]
                if not power_vals:
                    st.warning("No power‑related values found for LLM analysis.")
                else:
                    # Prepare context
                    context = f"Found {len(power_vals)} laser power measurements across {report.docs_with_results} documents.\n\n"
                    for v in power_vals[:10]:  # limit
                        context += f"- {v.value} {v.unit} in {v.doc_name} (page {v.page}): {v.context}\n"
                    context += f"\nConsensus: mean {report.consensus.get('mean',0):.1f} ± {report.consensus.get('std',0):.1f} {report.consensus.get('unit','W')} over {report.consensus.get('count',0)} values.\n"
                    prompt = f"""You are a scientific assistant. The user asked: "{query}".

Here are the extracted values from the uploaded PDFs:

{context}

Please write a clear, concise answer that:
1. Lists the key laser power values with citations.
2. Explains any variation or consensus.
3. Mentions which documents had no data (if any).
4. Uses exact citations like <cite doc="filename.pdf" page="X"/>.

Answer:
"""
                    # Load LLM if not already loaded
                    if st.session_state.llm_reasoner is None:
                        with st.spinner(f"Loading {selected_model}..."):
                            st.session_state.llm_reasoner = LLMReasoner(selected_model, use_4bit)
                    with st.spinner("Generating reasoning..."):
                        answer = st.session_state.llm_reasoner.generate(prompt, max_tokens=1024, temperature=0.2)
                    st.markdown("### 🧠 LLM Explanation")
                    st.markdown(answer)
                    # Also show the raw values
                    with st.expander("📊 Raw extracted values"):
                        st.json([v.to_dict() for v in power_vals])

    else:
        st.info("👆 Upload PDF files and click 'Build Index' first.")

if __name__ == "__main__":
    run_streamlit()
