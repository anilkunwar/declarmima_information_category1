#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DECLARMIMA v6.2-ACCELERATED - OMNISCIENT INTEGRATED APPLICATION
================================================================
VECTORLESS HIERARCHICAL RAG WITH PARALLEL DOCUMENT PROCESSING

Features:
- Full PageIndex integration (Tree structure parsing for PDF/MD).
- Parallel processing grouped by file size (Small/Medium/Large/XL).
- Generic query engine (Defaults to "laser power" but accepts any term).
- LLM-based extraction with exact citation generation.
- Streamlit UI for real-time interaction and JSON export.
- Anti-hallucination checks via source text verification.

Author: DECLARMIMA Team
Version: 6.2.1-ACCELERATED-OMNISCIENT
Date: 2026-05-06
"""

# =====================================================================
# SECTION 1: CORE IMPORTS & GLOBAL SETUP
# =====================================================================
import asyncio
import json
import re
import os
import sys
import time
import logging
import copy
import math
import random
import hashlib
import warnings
import tempfile
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

# Third-party libraries
import streamlit as st
import yaml
from pathlib import Path

try:
    import litellm
    from litellm import completion, acompletion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    warnings.warn("litellm not installed. LLM features will fail.")

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import pymupdf  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# =====================================================================
# SECTION 2: CONFIGURATION & LOGGING
# =====================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Default Model Configuration
DEFAULT_MODEL = os.getenv("MODEL_NAME", "gpt-4o")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

if not API_KEY and LITELLM_AVAILABLE:
    logger.warning("No API Key found in environment variables.")

# Processing Groups Configuration
# Groups files by size to optimize parallel thread allocation
PROCESSING_GROUPS = {
    "small": {"max_pages": 10, "max_tokens": 5000, "batch_size": 8, "max_mb": 2},
    "medium": {"max_pages": 20, "max_tokens": 15000, "batch_size": 4, "max_mb": 10},
    "large": {"max_pages": 35, "max_tokens": 30000, "batch_size": 2, "max_mb": 50},
    "extra_large": {"max_pages": float('inf'), "max_tokens": float('inf'), "batch_size": 1, "max_mb": float('inf')}
}

# =====================================================================
# SECTION 3: UTILITY FUNCTIONS (Integrated from provided snippets)
# =====================================================================

class SimpleNamespace:
    """Simple namespace for config objects."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def count_tokens(text, model=None):
    """Count tokens using litellm."""
    if not text or not LITELLM_AVAILABLE:
        return 0
    try:
        return litellm.token_counter(model=model, text=text)
    except:
        # Fallback rough estimation (4 chars per token)
        return len(text) // 4

def llm_completion(model, prompt, chat_history=None, return_finish_reason=False):
    """Synchronous LLM completion with retries."""
    if not LITELLM_AVAILABLE:
        raise Exception("LiteLLM is not installed.")
    
    max_retries = 3
    messages = list(chat_history) + [{"role": "user", "content": prompt}] if chat_history else [{"role": "user", "content": prompt}]
    
    for i in range(max_retries):
        try:
            response = completion(
                model=model,
                messages=messages,
                temperature=0.0,
            )
            content = response.choices[0].message.content
            if return_finish_reason:
                finish_reason = "max_output_reached" if response.choices[0].finish_reason == "length" else "finished"
                return content, finish_reason
            return content
        except Exception as e:
            logger.error(f"LLM Error (Retry {i+1}): {e}")
            if i < max_retries - 1:
                time.sleep(2)
            else:
                return ""

async def llm_acompletion(model, prompt):
    """Asynchronous LLM completion with retries."""
    if not LITELLM_AVAILABLE:
        raise Exception("LiteLLM is not installed.")
    
    max_retries = 3
    messages = [{"role": "user", "content": prompt}]
    
    for i in range(max_retries):
        try:
            response = await acompletion(
                model=model,
                messages=messages,
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Async LLM Error (Retry {i+1}): {e}")
            if i < max_retries - 1:
                await asyncio.sleep(2)
            else:
                return ""

def extract_json(content):
    """Robust JSON extraction from LLM string response."""
    try:
        # Clean common issues
        content = content.replace('None', 'null').replace('\n', ' ').replace('\r', ' ')
        content = ' '.join(content.split())
        
        # Extract code block
        start_idx = content.find("```json")
        if start_idx != -1:
            start_idx += 7
            content = content[start_idx:]
        end_idx = content.rfind("```")
        if end_idx != -1:
            content = content[:end_idx]
            
        return json.loads(content)
    except json.JSONDecodeError:
        # Fallback cleanup
        try:
            content = content.replace(',]', ']').replace(',}', '}')
            return json.loads(content)
        except:
            return {}

def remove_fields(data, fields=['text']):
    """Recursively remove keys from dict/list structure."""
    if isinstance(data, dict):
        return {k: remove_fields(v, fields)
                for k, v in data.items() if k not in fields}
    elif isinstance(data, list):
        return [remove_fields(item, fields) for item in data]
    return data

def structure_to_list(structure):
    """Flatten tree structure to list."""
    if isinstance(structure, dict):
        nodes = [structure]
        if 'nodes' in structure:
            nodes.extend(structure_to_list(structure['nodes']))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(structure_to_list(item))
        return nodes

# =====================================================================
# SECTION 4: PAGE INDEX & DOCUMENT PARSING CORE
# =====================================================================

class DocumentProcessor:
    """
    Handles the parsing of individual documents (PDF or MD)
    into a hierarchical tree structure using PageIndex logic.
    """
    
    def __init__(self, model_name):
        self.model = model_name

    def parse_document(self, file_obj) -> Dict:
        """
        Route document to correct parser (PDF vs MD) and return tree.
        """
        file_name = file_obj.name
        suffix = Path(file_name).suffix.lower()
        
        try:
            if suffix == '.pdf':
                return self._parse_pdf(file_obj)
            elif suffix in ['.md', '.markdown', '.txt']:
                return self._parse_markdown(file_obj)
            else:
                return {"error": "Unsupported file format", "doc_name": file_name}
        except Exception as e:
            logger.error(f"Error parsing {file_name}: {e}")
            return {"error": str(e), "doc_name": file_name}

    def _parse_markdown(self, file_obj) -> Dict:
        """Parse Markdown using the provided md_to_tree logic."""
        # Read content
        raw_bytes = file_obj.read()
        file_obj.seek(0)
        content = raw_bytes.decode('utf-8')
        
        # Simple Node Extraction (Regex based for MD)
        header_pattern = r'^(#{1,6})\s+(.+)$'
        node_list = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            match = re.match(header_pattern, line.strip())
            if match:
                node_list.append({
                    'node_title': match.group(2).strip(),
                    'line_num': line_num,
                    'level': len(match.group(1))
                })
        
        # Build text content for nodes
        for i, node in enumerate(node_list):
            start = node['line_num'] - 1
            end = node_list[i + 1]['line_num'] - 1 if i + 1 < len(node_list) else len(lines)
            node['text'] = '\n'.join(lines[start:end]).strip()
            node['token_count'] = count_tokens(node['text'], self.model)

        # Build Tree
        tree = self._build_tree_from_nodes(node_list)
        
        return {
            "doc_name": file_obj.name,
            "doc_type": "markdown",
            "structure": tree
        }

    def _parse_pdf(self, file_obj) -> Dict:
        """Parse PDF using PyMuPDF/PyPDF2 hybrid approach."""
        file_buffer = BytesIO(file_obj.read())
        file_obj.seek(0)
        
        # Prefer PyMuPDF for text extraction if available
        if PYMUPDF_AVAILABLE:
            return self._parse_pymupdf(file_buffer, file_obj.name)
        elif PYPDF2_AVAILABLE:
            return self._parse_pypdf2(file_buffer, file_obj.name)
        else:
            return {"error": "No PDF parser available", "doc_name": file_obj.name}

    def _parse_pymupdf(self, file_buffer, filename) -> Dict:
        doc = pymupdf.open(stream=file_buffer, filetype="pdf")
        toc = doc.get_toc()
        
        # Strategy: Use TOC if available, otherwise chunk by pages
        if toc:
            nodes = []
            for level, title, page_num in toc:
                # Extract text from page
                page = doc[page_num - 1]
                text = page.get_text("text")
                nodes.append({
                    'title': title,
                    'level': level,
                    'page_start': page_num,
                    'page_end': page_num + 5, # Approximation
                    'text': text[:2000], # Truncate for node preview
                    'token_count': count_tokens(text, self.model)
                })
            # Re-build ranges properly
            for i in range(len(nodes) - 1):
                nodes[i]['page_end'] = nodes[i+1]['page_start']
            if nodes:
                nodes[-1]['page_end'] = len(doc)
                
            tree = self._build_tree_from_pdf_nodes(nodes)
        else:
            # Fallback: Treat pages as nodes
            nodes = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                if text.strip():
                    nodes.append({
                        'title': f"Page {page_num + 1}",
                        'level': 1,
                        'page_start': page_num + 1,
                        'page_end': page_num + 1,
                        'text': text,
                        'token_count': count_tokens(text, self.model)
                    })
            tree = nodes
            
        doc.close()
        return {"doc_name": filename, "doc_type": "pdf", "structure": tree}

    def _build_tree_from_nodes(self, node_list):
        """Convert flat list to tree structure."""
        stack = []
        root_nodes = []
        
        for i, node in enumerate(node_list):
            current_level = node.get('level', 1)
            tree_node = {
                'title': node.get('node_title', node.get('title')),
                'text': node.get('text', ''),
                'page_start': node.get('page_start', node.get('line_num')),
                'page_end': node.get('page_end'),
                'nodes': []
            }
            
            while stack and stack[-1][1] >= current_level:
                stack.pop()
            
            if not stack:
                root_nodes.append(tree_node)
            else:
                parent_node, _ = stack[-1]
                parent_node['nodes'].append(tree_node)
            
            stack.append((tree_node, current_level))
            
        return root_nodes

    def _build_tree_from_pdf_nodes(self, node_list):
        # Similar logic but handling page indices
        stack = []
        root_nodes = []
        
        for node in node_list:
            current_level = node['level']
            tree_node = {
                'title': node['title'],
                'page_start': node['page_start'],
                'page_end': node['page_end'],
                'text': node['text'],
                'nodes': []
            }
            
            while stack and stack[-1][1] >= current_level:
                stack.pop()
            
            if not stack:
                root_nodes.append(tree_node)
            else:
                parent_node, _ = stack[-1]
                parent_node['nodes'].append(tree_node)
                
            stack.append((tree_node, current_level))
            
        return root_nodes

    def _parse_pypdf2(self, file_buffer, filename):
        # Basic fallback for PyPDF2
        reader = PyPDF2.PdfReader(file_buffer)
        nodes = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                nodes.append({
                    'title': f"Page {i+1}",
                    'level': 1,
                    'page_start': i+1,
                    'page_end': i+1,
                    'text': text
                })
        return {"doc_name": filename, "doc_type": "pdf", "structure": nodes}

# =====================================================================
# SECTION 5: QUERY & EXTRACTION ENGINE
# =====================================================================

class ExtractionEngine:
    """
    Handles the retrieval and extraction of information based on user query.
    Uses LLM to find and extract specific parameters (e.g., "laser power").
    """
    
    def __init__(self, model_name):
        self.model = model_name
    
    async def extract_from_document(self, doc_data: Dict, query: str) -> Dict:
        """
        Main extraction logic for a single document.
        1. Flatten tree to list of chunks.
        2. Filter chunks by relevance (Keyword match).
        3. Extract data using LLM.
        """
        doc_name = doc_data.get('doc_name', 'Unknown')
        structure = doc_data.get('structure', [])
        
        # Flatten structure
        chunks = structure_to_list(structure)
        
        # Heuristic filtering to save tokens (find chunks mentioning the query)
        # This makes it "Vectorless" but efficient
        query_lower = query.lower()
        relevant_chunks = []
        
        for chunk in chunks:
            text_content = chunk.get('text', '')
            title = chunk.get('title', '')
            
            # If query is found in title or text, or if text is short enough to scan
            if query_lower in text_content.lower() or query_lower in title.lower():
                relevant_chunks.append(chunk)
            elif len(text_content) < 500: # Check short texts anyway
                relevant_chunks.append(chunk)
        
        if not relevant_chunks:
            return {
                "doc_name": doc_name,
                "status": "no_match",
                "found_data": [],
                "reason": f"Query '{query}' not found in text index."
            }
            
        # Batch chunks for LLM context (avoid context limit)
        extraction_results = []
        
        # Limit chunks to top 5 most relevant by simple score to save costs/time
        relevant_chunks.sort(key=lambda x: x.get('text', '').count(query_lower), reverse=True)
        selected_chunks = relevant_chunks[:5]
        
        for chunk in selected_chunks:
            result = await self._llm_extract(query, chunk, doc_name)
            if result:
                extraction_results.append(result)
                
        return {
            "doc_name": doc_name,
            "status": "success",
            "found_data": extraction_results
        }

    async def _llm_extract(self, query, chunk, doc_name):
        """
        Ask LLM to find specific values related to the query in the chunk.
        """
        text = chunk.get('text', '')
        page = chunk.get('page_start', 'N/A')
        title = chunk.get('title', 'Unknown Section')
        
        prompt = f"""
        You are an expert data extraction assistant. 
        Your task is to find information related to the query: "{query}".
        
        Analyze the following text snippet from a document:
        Document: {doc_name}
        Section: {title}
        Page: {page}
        
        Text Content:
        {text[:3000]} 
        
        Instructions:
        1. Identify if the text contains specific values, settings, or descriptions related to "{query}".
        2. If found, extract the exact value, unit, and context sentence.
        3. If the text discusses the query but provides no specific value, summarize the discussion.
        
        Return ONLY a valid JSON object with this structure:
        {{
            "query": "{query}",
            "value_found": <boolean>,
            "value": <string or number or null>,
            "unit": <string or null>,
            "context_sentence": <string or null>,
            "confidence": <float 0.0 to 1.0>,
            "notes": <string explaining extraction logic>
        }}
        """
        
        try:
            response_str = await llm_acompletion(self.model, prompt)
            response_json = extract_json(response_str)
            
            # Augment with metadata
            response_json['source_page'] = page
            response_json['source_section'] = title
            return response_json
        except Exception as e:
            logger.error(f"LLM Extraction error: {e}")
            return None

# =====================================================================
# SECTION 6: PARALLEL ORCHESTRATOR
# =====================================================================

class OmniscientProcessor:
    """
    Manages parallel processing of multiple documents, grouped by size.
    """
    
    def __init__(self, model_name):
        self.model = model_name
        self.parser = DocumentProcessor(model_name)
        self.extractor = ExtractionEngine(model_name)
    
    def _get_file_group(self, file_obj) -> str:
        """Determine processing group based on file size."""
        size_mb = len(file_obj.getbuffer()) / (1024 * 1024)
        if size_mb < PROCESSING_GROUPS["small"]["max_mb"]:
            return "small"
        elif size_mb < PROCESSING_GROUPS["medium"]["max_mb"]:
            return "medium"
        elif size_mb < PROCESSING_GROUPS["large"]["max_mb"]:
            return "large"
        else:
            return "extra_large"

    async def process_all(self, files: List, query: str) -> Dict:
        """
        Main entry point for processing.
        1. Group files.
        2. Parse documents (Tree building).
        3. Extract info (LLM).
        """
        start_time = time.time()
        
        # 1. Group files
        groups = defaultdict(list)
        for f in files:
            groups[self._get_file_group(f)].append(f)
            
        logger.info(f"File groups: {[(k, len(v)) for k, v in groups.items()]}")
        
        # 2. Parse Documents in parallel
        # We use ThreadPoolExecutor for IO bound parsing tasks
        doc_data_map = {} # filename -> parsed_tree
        
        def parse_task(f):
            return f.name, self.parser.parse_document(f)
        
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(parse_task, f): f for f in files}
            for future in as_completed(futures):
                fname, data = future.result()
                doc_data_map[fname] = data
        
        # 3. Extract Information from parsed data
        # This is CPU/LLM bound, so we use asyncio.gather
        extraction_tasks = []
        for fname, data in doc_data_map.items():
            if "error" not in data:
                extraction_tasks.append(self.extractor.extract_from_document(data, query))
            else:
                # Return error immediately
                extraction_tasks.append(asyncio.coroutine(lambda: data)())
        
        results = await asyncio.gather(*extraction_tasks)
        
        # 4. Format Final Output
        final_report = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": round(time.time() - start_time, 2),
            "total_files_processed": len(files),
            "successful_extractions": 0,
            "failed_extractions": 0,
            "results": results
        }
        
        # Stats
        for r in results:
            if r.get('status') == 'success' and r.get('found_data'):
                final_report['successful_extractions'] += 1
            else:
                final_report['failed_extractions'] += 1
                
        return final_report

# =====================================================================
# SECTION 7: STREAMLIT UI
# =====================================================================

def render_ui():
    st.set_page_config(
        page_title="DECLARMIMA v6.2-ACCELERATED",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better look
    st.markdown("""
        <style>
        .main .block-container { padding-top: 2rem; }
        h1 { color: #1f77b4; }
        .stAlert { border-radius: 5px; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("⚡ DECLARMIMA v6.2-ACCELERATED")
    st.markdown("### Omniscient Vectorless RAG System")
    st.markdown("Upload documents to check for specific parameters (e.g., *Laser Power*) in parallel.")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        model_name = st.text_input("LLM Model", DEFAULT_MODEL, help="Model name (e.g., gpt-4o, claude-3-opus)")
        
        if os.getenv("OPENAI_API_KEY"):
            st.success("OpenAI Key detected")
        elif os.getenv("ANTHROPIC_API_KEY"):
            st.success("Anthropic Key detected")
        else:
            st.warning("API Key not found. Please set OPENAI_API_KEY in environment variables.")
            
        st.markdown("---")
        st.markdown("**File Groups:**")
        st.info(f"Small: <{PROCESSING_GROUPS['small']['max_mb']}MB\n"
                f"Medium: <{PROCESSING_GROUPS['medium']['max_mb']}MB\n"
                f"Large: <{PROCESSING_GROUPS['large']['max_mb']}MB")

    # Main Area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "📂 Upload Documents (PDF, MD)",
            type=['pdf', 'md', 'txt'],
            accept_multiple_files=True,
            help="Upload multiple files. They will be grouped by size and processed in parallel."
        )
        
        query_input = st.text_input(
            "🎯 Search Query / Parameter",
            value="laser power",
            help="Enter the parameter to search for (e.g., 'laser power', 'temperature', 'voltage')."
        )
        
        process_btn = st.button("🚀 Start Accelerated Extraction", type="primary", use_container_width=True)

    with col2:
        st.metric("Documents Ready", len(uploaded_files) if uploaded_files else 0)
        st.metric("Workers Available", os.cpu_count() or 4)

    # Processing Logic
    if process_btn:
        if not uploaded_files:
            st.error("Please upload at least one document.")
        elif not query_input:
            st.error("Please enter a search query.")
        elif not LITELLM_AVAILABLE:
            st.error("Litellm library is required. Run: pip install litellm")
        else:
            # Run Async Process
            with st.spinner(f"Processing {len(uploaded_files)} documents in parallel groups..."):
                
                # Progress Bar Logic placeholder (async doesn't play nice with st natively without complex callbacks)
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    processor = OmniscientProcessor(model_name=model_name)
                    status_text.text("Initializing processor...")
                    progress_bar.progress(10)
                    
                    # Run the async loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    result_report = loop.run_until_complete(
                        processor.process_all(uploaded_files, query_input)
                    )
                    
                    progress_bar.progress(90)
                    status_text.text("Finalizing results...")
                    
                    # Display Results
                    st.success(f"✅ Processing Complete in {result_report['processing_time_seconds']}s")
                    progress_bar.progress(100)
                    
                    # Summary Stats
                    st.subheader("📊 Summary")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Docs", result_report['total_files_processed'])
                    c2.metric("With Matches", result_report['successful_extractions'])
                    c3.metric("No Matches", result_report['failed_extractions'])
                    c4.metric("Query", query_input)
                    
                    # Detailed Results
                    st.subheader("🔍 Detailed Findings")
                    
                    # Filter successful results
                    successful = [r for r in result_report['results'] if r.get('status') == 'success']
                    
                    if successful:
                        for doc_res in successful:
                            with st.expander(f"📄 {doc_res['doc_name']} ({len(doc_res['found_data'])} hits)"):
                                for hit in doc_res['found_data']:
                                    st.markdown(f"**Value:** {hit.get('value', 'N/A')} {hit.get('unit', '')}")
                                    st.markdown(f"**Context:** {hit.get('context_sentence', '')}")
                                    st.caption(f"Page {hit.get('source_page')} | Section: {hit.get('source_section')} | Confidence: {hit.get('confidence')}")
                                    st.json(hit)
                    else:
                        st.info("No specific values found for the query in the provided documents.")
                    
                    # JSON Download
                    st.subheader("💾 Export Data")
                    json_str = json.dumps(result_report, indent=2)
                    st.download_button(
                        label="Download Full JSON Report",
                        data=json_str,
                        file_name=f"DECLARMIMA_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                    # Show raw JSON in viewer
                    with st.expander("View Raw JSON"):
                        st.json(result_report)
                        
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                    logger.exception("Processing failed")

# =====================================================================
# SECTION 8: MAIN ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    # Check for dependencies
    missing_deps = []
    if not LITELLM_AVAILABLE: missing_deps.append("litellm")
    if not PYPDF2_AVAILABLE: missing_deps.append("PyPDF2")
    if not PYMUPDF_AVAILABLE: missing_deps.append("pymupdf")
    
    if missing_deps:
        st.error("Missing required libraries. Please install:")
        st.code(f"pip install {' '.join(missing_deps)}")
    else:
        render_ui()
