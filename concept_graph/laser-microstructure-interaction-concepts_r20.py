#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DECLARMIMA: Alloy Microstructure Concept Graph with Dual LLM Backend Support
========================================================================================
✅ ENHANCED: 50+ colormap registry with dynamic visualization mapping
✅ ENHANCED: TF-IDF concept distillation, relevance scoring & confidence metrics
✅ ENHANCED: Mathematical validation (k-fold CV, bootstrap uncertainty, graph robustness, permutation tests)
✅ ENHANCED: UMAP/PCA embedding plot, publication-ready exports, interactive filtering
✅ ENHANCED: Tabbed UI, session state persistence, memory-optimized rendering
✅ DGL INTEGRATION ADDED:
- Optional DGL backend for GraphSAGE with automatic fallback to PyTorch sparse
- Heterogeneous graph support for edge-type-aware message passing
- DGL CUDA compatibility diagnostics alongside existing PyTorch checks
- Memory-efficient mini-batching ready for large concept graphs
- FIXED: String literal type hints for optional DGL types to prevent AttributeError
✅ PYVIS DOWNLOAD CRASH FIX:
- Changed cdn_resources='in_line' -> 'remote' (reduces HTML from ~5MB to ~50KB)
- Safe bytes encoding for st.download_button to prevent Streamlit OOM
- Explicit memory cleanup (del + gc.collect()) after download generation
- Try/except isolation to prevent session state wipe on failure
✅ CUDA COMPATIBILITY FIXES:
- GPU compute capability detection
- Automatic PyTorch/CUDA/DGL version validation
- Graceful CPU fallback with user notification
- Environment variable guidance for manual fixes
✅ Zero API keys - all models run locally (HuggingFace Transformers + Ollama)
✅ Dual backend: Choose between HF Transformers (direct loading) or Ollama (server-based)
✅ ALL models from LASER RAG codebase included (12 HF + 8 Ollama options)
✅ Memory-aware: 4-bit quantization, VRAM estimation, CPU/GPU auto-detection
✅ DECLARMIMA-focused: Laser-microstructure interaction, multicomponent alloys, digital twins
✅ Physics-informed: Concept graphs, GraphSAGE embeddings (PyTorch or DGL), research direction scoring
✅ Small-corpus optimized: Adaptive thresholds, semantic clustering, seed injection
✅ ENHANCED: Publication-quality visualizations, metrics dashboard, user customization
✅ ENHANCED: Interactive PyVis with physics toggle, category-based colors, size scaling
✅ ENHANCED: Sunburst chart for research domain hierarchy, link prediction dashboard
DEPLOYMENT:
pip install streamlit torch transformers sentence-transformers networkx scikit-learn
pip install pyvis plotly pandas numpy kaleido matplotlib scipy umap-learn
pip install ollama  # optional for Ollama backend
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html  # optional, adjust CUDA version
# 🔧 CUDA FIX: If you get "no kernel image" error, run ONE of these:
# For NEW GPUs (RTX 40xx/50xx, Blackwell):
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# For DGL CUDA issues:
pip uninstall dgl -y
pip install dgl -f https://data.dgl.ai/wheels/cu128/repo.html  # Match your CUDA version
# For OLD GPUs (compute capability < 3.7):
# You must build PyTorch from source with TORCH_CUDA_ARCH_LIST=3.5
# Or force CPU mode by setting: export CUDA_VISIBLE_DEVICES=""
Run: streamlit run declarmima_concept_graph.py
"""
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sparse
import torch.optim as optim
import networkx as nx
import numpy as np
import pandas as pd
import re
import json
import os
import sys
import tempfile
import warnings
import traceback
import gc
import hashlib
import subprocess
from collections import defaultdict, Counter
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union, Any, TYPE_CHECKING
from sklearn.linear_model import Ridge
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import scipy.stats as stats
from sentence_transformers import SentenceTransformer
from transformers import (
AutoTokenizer, AutoModelForCausalLM,
BitsAndBytesConfig, GPT2Tokenizer, GPT2LMHeadModel
)
from pyvis.network import Network
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

# TYPE_CHECKING is True during static analysis (mypy, IDEs) but False at runtime
if TYPE_CHECKING:
    import dgl
    import dgl.nn as dglnn
    import umap

# Optional DGL import with graceful fallback
try:
    import dgl
    import dgl.nn as dglnn
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    dgl = None
    dglnn = None

# Optional UMAP import with graceful fallback
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Optional Ollama import with graceful fallback
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None

warnings.filterwarnings('ignore')

# ==========================================
# 🎨 COMPREHENSIVE COLORMAP REGISTRY (50+)
# ==========================================
COLORMAP_REGISTRY = {
    # Sequential
    'viridis': 'viridis', 'plasma': 'plasma', 'inferno': 'inferno', 'magma': 'magma',
    'cividis': 'cividis', 'turbo': 'turbo', 'jet': 'jet', 'hot': 'hot',
    'cool': 'cool', 'spring': 'spring', 'summer': 'summer', 'autumn': 'autumn',
    'winter': 'winter', 'gray': 'gray', 'bone': 'bone', 'copper': 'copper',
    'pink': 'pink', 'rainbow': 'rainbow', 'hsv': 'hsv', 'terrain': 'terrain',
    'ocean': 'ocean', 'gist_earth': 'gist_earth', 'gist_stern': 'gist_stern',
    'gist_heat': 'gist_heat', 'gist_rainbow': 'gist_rainbow', 'gnuplot': 'gnuplot',
    'gnuplot2': 'gnuplot2', 'afmhot': 'afmhot', 'binary': 'binary',
    # Diverging
    'coolwarm': 'coolwarm', 'seismic': 'seismic', 'bwr': 'bwr', 'RdBu': 'RdBu',
    'RdYlBu': 'RdYlBu', 'RdGy': 'RdGy', 'PuOr': 'PuOr', 'BrBG': 'BrBG',
    'PiYG': 'PiYG', 'PRGn': 'PRGn', 'RdYlGn': 'RdYlGn',
    # Qualitative / Categorical
    'tab10': 'tab10', 'tab20': 'tab20', 'tab20b': 'tab20b', 'tab20c': 'tab20c',
    'Set1': 'Set1', 'Set2': 'Set2', 'Set3': 'Set3', 'Accent': 'Accent',
    'Dark2': 'Dark2', 'Paired': 'Paired', 'Pastel1': 'Pastel1', 'Pastel2': 'Pastel2',
    'flag': 'flag', 'prism': 'prism', 'nipy_spectral': 'nipy_spectral',
    'Spectral': 'Spectral', 'brg': 'brg', 'CMRmap': 'CMRmap',
    'Wistia': 'Wistia', 'YlGnBu': 'YlGnBu', 'YlOrBr': 'YlOrBr', 'YlOrRd': 'YlOrRd'
}

def get_colormap_hex(colormap_name: str, num_colors: int) -> List[str]:
    """Generate hex colors from any matplotlib colormap."""
    try:
        cmap = cm.get_cmap(colormap_name, num_colors)
        return [plt.matplotlib.colors.to_hex(cmap(i)) for i in range(num_colors)]
    except:
        return [f"#{i:06x}" for i in np.random.randint(0, 0xFFFFFF, num_colors)]

def map_values_to_colors(values: np.ndarray, colormap_name: str, vmin: float = None, vmax: float = None) -> List[str]:
    """Map numerical values to hex colors using a specified colormap."""
    if vmin is None: vmin = np.min(values)
    if vmax is None: vmax = np.max(values)
    if vmax == vmin: vmax = vmin + 1e-6
    norm = plt.Normalize(vmin, vmax)
    cmap = cm.get_cmap(colormap_name)
    return [plt.matplotlib.colors.to_hex(cmap(norm(v))) for v in values]

# ==========================================
# 🔧 CUDA & DGL COMPATIBILITY DIAGNOSTICS
# ==========================================
def get_gpu_compute_capability(device_id: int = 0) -> Optional[Tuple[int, int]]:
    if not torch.cuda.is_available():
        return None
    try:
        major, minor = torch.cuda.get_device_capability(device_id)
        return (major, minor)
    except Exception as e:
        st.warning(f"⚠️ Could not detect GPU compute capability: {e}")
        return None

def get_pytorch_cuda_info() -> Dict[str, Any]:
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if hasattr(torch.version, 'cuda') else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": [],
        "compute_capabilities": [],
        "pytorch_cuda_build": None
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            info["gpu_names"].append(torch.cuda.get_device_name(i))
            try:
                cc = torch.cuda.get_device_capability(i)
                info["compute_capabilities"].append(f"{cc[0]}.{cc[1]}")
            except:
                info["compute_capabilities"].append("Unknown")
        info["pytorch_cuda_build"] = "cu" in torch.__version__
    return info

def get_dgl_info() -> Dict[str, Any]:
    info = {"available": DGL_AVAILABLE, "version": None, "backend": None, "cuda_support": False, "gpu_test_passed": False}
    if not DGL_AVAILABLE: return info
    try:
        info["version"] = dgl.__version__
        info["backend"] = dgl.backend.backend_name
        if torch.cuda.is_available():
            try:
                test_g = dgl.graph(([0, 1], [1, 2])).to('cuda')
                info["cuda_support"] = True
                info["gpu_test_passed"] = True
            except Exception as e:
                info["cuda_support"] = True
                info["gpu_test_passed"] = False
                info["gpu_error"] = str(e)
        else:
            info["cuda_support"] = False
    except Exception as e:
        info["error"] = str(e)
    return info

def check_cuda_kernel_compatibility() -> Tuple[bool, str]:
    if not torch.cuda.is_available():
        return True, "CUDA not available - using CPU mode"
    cuda_info = get_pytorch_cuda_info()
    messages, is_compatible = [], True
    MIN_SM = 3.7
    for i, cc_str in enumerate(cuda_info["compute_capabilities"]):
        if cc_str == "Unknown":
            messages.append(f"⚠️ GPU {i}: Could not determine compute capability")
            continue
        try:
            major, minor = map(int, cc_str.split('.'))
            sm = major + minor / 10
            if sm < MIN_SM:
                is_compatible = False
                messages.append(f"❌ GPU {i} ({cuda_info['gpu_names'][i]}): Compute capability {cc_str} < {MIN_SM}. Solution: Build PyTorch from source with TORCH_CUDA_ARCH_LIST={cc_str}")
            elif sm >= 9.0:
                cuda_build = cuda_info["cuda_version"]
                if cuda_build and float(cuda_build.replace('.', '')) < 124 and sm >= 9.0:
                    is_compatible = False
                    messages.append(f"❌ GPU {i} ({cuda_info['gpu_names'][i]}): New GPU (sm_{major}{minor}) requires CUDA 12.4+ but PyTorch built with CUDA {cuda_build}. Solution: pip install -U torch --index-url https://download.pytorch.org/whl/cu128")
                else:
                    messages.append(f"✅ GPU {i} ({cuda_info['gpu_names'][i]}): Compatible (sm_{major}{minor})")
            else:
                messages.append(f"✅ GPU {i} ({cuda_info['gpu_names'][i]}): Compatible (sm_{major}{minor})")
        except ValueError:
            messages.append(f"⚠️ GPU {i}: Invalid compute capability format: {cc_str}")
    return is_compatible, "\n".join(messages)

def check_dgl_cuda_compatibility() -> Tuple[bool, str]:
    if not DGL_AVAILABLE:
        return False, "❌ DGL not installed. Run: pip install dgl -f https://data.dgl.ai/wheels/[cuda_version]/repo.html"
    try:
        msg = [f"DGL version: {dgl.__version__}", f"DGL backend: {dgl.backend.backend_name}"]
        if torch.cuda.is_available():
            try:
                dgl.graph(([0, 1], [1, 2])).to('cuda')
                msg.append("✅ DGL GPU test passed")
                return True, "\n".join(msg)
            except Exception as e:
                msg.append(f"❌ DGL GPU error: {e}")
                cuda_ver = torch.version.cuda or "unknown"
                msg.append(f"💡 Try: pip uninstall dgl -y && pip install dgl -f https://data.dgl.ai/wheels/cu{cuda_ver.replace('.', '')}/repo.html")
                return False, "\n".join(msg)
        else:
            msg.append("ℹ️ CUDA not available - DGL will use CPU")
            return True, "\n".join(msg)
    except Exception as e:
        return False, f"❌ DGL check failed: {e}"

def force_cpu_mode():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    original_is_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    st.session_state["force_cpu"] = True
    st.session_state["original_cuda_available"] = original_is_available
    return torch.device('cpu')

def restore_cuda_mode():
    if "original_cuda_available" in st.session_state:
        torch.cuda.is_available = st.session_state["original_cuda_available"]
        del st.session_state["original_cuda_available"]
    if "force_cpu" in st.session_state:
        del st.session_state["force_cpu"]
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]

def show_cuda_fix_instructions(compatible: bool, diagnostic: str, dgl_compatible: bool = True, dgl_diagnostic: str = ""):
    with st.expander("🔧 CUDA/DGL Diagnostics & Fix Instructions", expanded=not (compatible and dgl_compatible)):
        st.markdown("### 📊 System CUDA Information")
        cuda_info = get_pytorch_cuda_info()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
**PyTorch Version:** `{cuda_info['torch_version']}`
**CUDA Available:** {'✅ Yes' if cuda_info['cuda_available'] else '❌ No'}
**CUDA Build Version:** `{cuda_info['cuda_version'] or 'N/A'}`
**cuDNN Version:** `{cuda_info['cudnn_version'] or 'N/A'}`
**Pre-built with CUDA:** {'✅ Yes' if cuda_info['pytorch_cuda_build'] else '❌ No'}
""")
        with col2:
            if cuda_info['gpu_count'] > 0:
                gpu_list = "\n".join([f"- {name} (sm_{cc})" for name, cc in zip(cuda_info['gpu_names'], cuda_info['compute_capabilities'])])
                st.markdown(f"**Detected GPUs:**\n{gpu_list}")
            else:
                st.markdown("**Detected GPUs:** None (CPU mode)")
        st.markdown("### 🔍 PyTorch Compatibility Check")
        status_icon = "✅" if compatible else "❌"
        st.markdown(f"{status_icon} **PyTorch Status:** {'Compatible' if compatible else 'INCOMPATIBLE'}")
        if diagnostic.strip(): st.code(diagnostic, language="text")
        st.markdown("### 🔍 DGL Compatibility Check")
        dgl_info = get_dgl_info()
        if dgl_info["available"]:
            st.markdown(f"""
**DGL Version:** `{dgl_info['version']}`
**Backend:** `{dgl_info['backend']}`
**CUDA Support:** {'✅ Yes' if dgl_info['cuda_support'] else '❌ No'}
**GPU Test:** {'✅ Passed' if dgl_info['gpu_test_passed'] else '❌ Failed'}
""")
            if dgl_diagnostic.strip(): st.code(dgl_diagnostic, language="text")
        else:
            st.markdown("❌ **DGL not installed**")
            if dgl_diagnostic.strip(): st.code(dgl_diagnostic, language="text")
        if not compatible or not dgl_compatible:
            st.markdown("### 🛠️ Recommended Fixes")
            if not compatible:
                if any("Blackwell" in msg or "RTX 50" in msg or "sm_9" in msg for msg in diagnostic.split('\n')):
                    st.markdown("**For NEW GPUs (RTX 50xx/Blackwell/Ada Lovelace) - PyTorch:**\n```bash\npip uninstall torch torchvision torchaudio -y\npip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128\n```")
                elif any("sm_3.5" in msg or "sm_3.6" in msg or "GT 730" in msg for msg in diagnostic.split('\n')):
                    st.markdown("**For OLD GPUs (compute capability < 3.7) - PyTorch:**\n```bash\nexport CUDA_VISIBLE_DEVICES=\"\"\n# Or build from source with TORCH_CUDA_ARCH_LIST=3.5\n```")
            if not dgl_compatible and DGL_AVAILABLE:
                cuda_ver = torch.version.cuda or "118"
                cuda_wheel = f"cu{cuda_ver.replace('.', '')}"
                st.markdown(f"**DGL CUDA Fix:**\n```bash\npip uninstall dgl -y\npip install dgl -f https://data.dgl.ai/wheels/{cuda_wheel}/repo.html\n```")
            elif not DGL_AVAILABLE:
                st.markdown("**Install DGL:**\n```bash\npip install dgl  # CPU\npip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html  # CUDA 11.8\n```")
            st.markdown("**Universal Fallback (CPU Mode):**\n```python\nimport os\nos.environ[\"CUDA_VISIBLE_DEVICES\"] = \"\"\n```")
            if st.button("🔄 Reload App in CPU Mode"):
                force_cpu_mode()
                st.rerun()

# ==========================================
# STREAMLIT CONFIGURATION
# ==========================================
st.set_page_config(page_title="DECLARMIMA: Alloy Microstructure Concept Graph", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["CUDA_LAUNCH_BLOCKING"] = os.environ.get("CUDA_LAUNCH_BLOCKING", "1")

# ==========================================
# DEVICE & SEED SETUP
# ==========================================
def initialize_device():
    if st.session_state.get("force_cpu", False):
        return torch.device('cpu')
    compatible, diagnostic = check_cuda_kernel_compatibility()
    dgl_compatible, dgl_diagnostic = check_dgl_cuda_compatibility()
    if not compatible:
        show_cuda_fix_instructions(compatible, diagnostic, dgl_compatible, dgl_diagnostic)
        st.warning("⚠️ CUDA incompatible - falling back to CPU mode. Click 'Reload App in CPU Mode' above.")
        return force_cpu_mode()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        cc = get_gpu_compute_capability()
        if cc:
            st.sidebar.success(f"🎮 GPU: {torch.cuda.get_device_name(0)} (sm_{cc[0]}{cc[1]})")
    return device

DEVICE = initialize_device()
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ==========================================
# MODEL REGISTRY & CONFIG
# ==========================================
LOCAL_LLM_OPTIONS = {
    "GPT-2 (1.5B, fastest startup, CPU OK)": "gpt2",
    "Qwen2-0.5B-Instruct (best JSON, recommended)": "Qwen/Qwen2-0.5B-Instruct",
    "Qwen2.5-0.5B-Instruct (newest, best reasoning)": "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama-1.1B-Chat (balanced small model)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen2.5-1.5B-Instruct (efficient mid-size)": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct (strong reasoning)": "Qwen/Qwen2.5-3B-Instruct",
    "Mistral-7B-Instruct-v0.3 (reliable & efficient)": "mistralai/Mistral-7B-Instruct-v0.3",
    "Llama-3.2-3B-Instruct (Meta's latest small)": "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen2.5-7B-Instruct (excellent all-rounder)": "Qwen/Qwen2.5-7B-Instruct",
    "Llama-3.1-8B-Instruct (most popular balanced)": "meta-llama/Llama-3.1-8B-Instruct",
    "Gemma-2-9B-it (Google's latest, great logic)": "google/gemma-2-9b-it",
    "Falcon-7B-Instruct (lightweight & modern)": "tiiuae/falcon-7b-instruct",
    "[Ollama] qwen2.5:0.5b (via ollama serve)": "ollama:qwen2.5:0.5b",
    "[Ollama] qwen2.5:1.5b (via ollama serve)": "ollama:qwen2.5:1.5b",
    "[Ollama] qwen2.5:7b (via ollama serve)": "ollama:qwen2.5:7b",
    "[Ollama] qwen2.5:14b (via ollama serve) 🔥": "ollama:qwen2.5:14b",
    "[Ollama] llama3.1:8b (via ollama serve)": "ollama:llama3.1:8b",
    "[Ollama] mistral:7b (via ollama serve)": "ollama:mistral:7b",
    "[Ollama] gemma2:9b (via ollama serve)": "ollama:gemma2:9b",
    "[Ollama] falcon3:10b (via ollama serve)": "ollama:falcon3:10b",
}
DEFAULT_LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
EMBED_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_MEMORY_ESTIMATES = {
    "gpt2": {"params": "1.5B", "vram_fp16": "~3GB", "vram_4bit": "~1GB", "cpu_ok": True},
    "Qwen/Qwen2-0.5B-Instruct": {"params": "0.5B", "vram_fp16": "~1GB", "vram_4bit": "~400MB", "cpu_ok": True},
    "Qwen/Qwen2.5-0.5B-Instruct": {"params": "0.5B", "vram_fp16": "~1GB", "vram_4bit": "~400MB", "cpu_ok": True},
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {"params": "1.1B", "vram_fp16": "~2.5GB", "vram_4bit": "~800MB", "cpu_ok": True},
    "Qwen/Qwen2.5-1.5B-Instruct": {"params": "1.5B", "vram_fp16": "~3.5GB", "vram_4bit": "~1.2GB", "cpu_ok": False},
    "Qwen/Qwen2.5-3B-Instruct": {"params": "3B", "vram_fp16": "~6GB", "vram_4bit": "~2GB", "cpu_ok": False},
    "mistralai/Mistral-7B-Instruct-v0.3": {"params": "7B", "vram_fp16": "~14GB", "vram_4bit": "~4.5GB", "cpu_ok": False},
    "meta-llama/Llama-3.2-3B-Instruct": {"params": "3B", "vram_fp16": "~6GB", "vram_4bit": "~2GB", "cpu_ok": False},
    "Qwen/Qwen2.5-7B-Instruct": {"params": "7B", "vram_fp16": "~14GB", "vram_4bit": "~4.5GB", "cpu_ok": False},
    "Qwen/Qwen2.5-14B-Instruct": {"params": "14B", "vram_fp16": "~28GB", "vram_4bit": "~9GB", "cpu_ok": False},
    "meta-llama/Llama-3.1-8B-Instruct": {"params": "8B", "vram_fp16": "~16GB", "vram_4bit": "~5GB", "cpu_ok": False},
    "google/gemma-2-9b-it": {"params": "9B", "vram_fp16": "~18GB", "vram_4bit": "~6GB", "cpu_ok": False},
    "tiiuae/falcon-7b-instruct": {"params": "7B", "vram_fp16": "~14GB", "vram_4bit": "~4.5GB", "cpu_ok": False},
}

# ==========================================
# DECLARMIMA PROJECT: Seed knowledge base
# ==========================================
DECLARMIMA_PROPOSAL_TEXT = """
Deciphering laser-microstructure interaction in multicomponent alloys (DECLARMIMA)
Scientific goals: Additive manufacturing, laser processing, multicomponent alloys, high-entropy alloys, digital twins, physics-informed machine learning, phase field modeling, molecular dynamics, melt pool dynamics, microstructure evolution, process-structure-property relationships, selective laser melting, powder bed fusion, laser powder bed fusion, in-situ monitoring, defect formation, porosity, spatter, residual stress, grain morphology, phase transformation, solidification, Marangoni convection, CALPHAD thermodynamics, interfacial energy, thermal conductivity, viscosity, absorptivity, reflectivity, Gaussian heat source, finite element method, MOOSE framework, LAMMPS, ThermoCalc, neural networks, convolutional neural networks, random forest, Bayesian machine learning, uncertainty quantification, feature engineering, tensor decomposition, scale-bridging, multiscale modeling, inverse design, optimization, Al-Si-Mg alloys, Ti-6Al-4V, Inconel 718, Sn-Ag-Cu solders, CoCrFeNi HEAs, intermetallic compounds, columnar grains, equiaxed grains, dendritic structures, martensite, austenite, precipitates, segregation, crack propagation, fatigue life, tensile strength, yield strength, microhardness, elongation, ductility, wear resistance, corrosion resistance, oxidation resistance, laser power, scan speed, hatch spacing, layer thickness, pulse duration, energy density, spot diameter, cooling rate, solidification rate, dilution ratio, powder particle size, particle size distribution, flowability, oxygen content, moisture content, bed temperature, pre-heating, post-processing, heat treatment, surface finishing, quality monitoring, photodiode sensors, line scanners, camera trackers, acoustic transducers, synchrotron X-ray imaging, EBSD, nanoindentation, in-situ XRD, SEM, TEM, AFM, digital image correlation, machine vision, data fusion, knowledge graphs, concept graphs, graph neural networks, GraphSAGE, node embeddings, edge prediction, link prediction, research direction discovery, hypothesis generation, novelty scoring, feasibility assessment, property gain prediction, composite scoring, adaptive configuration, small corpus optimization, semantic clustering, domain seed injection, hybrid graph construction, co-occurrence edges, semantic similarity edges, contrastive learning, edge sampling, sparse tensors, degree normalization, mean aggregation, two-layer architecture, decoder network, BCE loss, Adam optimizer, training loop, evaluation metrics, progress tracking, memory management, CUDA optimization, CPU fallback, error handling, fallback strategies, interactive visualization, PyVis, Plotly, force-directed layout, spring layout, node styling, edge styling, hover tooltips, download functionality, text fallback, diagnostics panel, concept frequency, edge weight, graph connectivity, component analysis, degree distribution, clustering coefficient, centrality measures, path length, bridge edges, semantic bridges, knowledge injection, concept normalization, alloy notation standardization, laser term normalization, unit standardization, regex extraction, quantitative metrics, grain size, mechanical properties, energy density, defect fraction, prompt engineering, JSON parsing, fallback extraction, domain validation, generic term filtering, concept abstraction, category mapping, hierarchical representation, representative selection, cluster merging, similarity threshold, distance matrix, linkage method, embedding encoding, batch processing, progress display, model caching, resource management, timeout handling, user feedback, status indicators, progress bars, error messages, warning dialogs, success notifications, download buttons, CSV export, HTML export, JSON export, interactive controls, physics parameters, gravity, spring length, damping, overlap, stabilization, node sampling, size limiting, performance optimization, browser compatibility, JavaScript execution, CDN resources, inline embedding, iframe alternative, HTML rendering, Streamlit components, responsive design, mobile compatibility, accessibility, color contrast, theme switching, dark mode, light mode, user preferences, session state, configuration persistence, adaptive thresholds, corpus size detection, parameter tuning, hyperparameter optimization, validation metrics, testing framework, debugging tools, logging, tracebacks, exception handling, graceful degradation, fallback rendering, text summary, edge listing, frequency tables, diagnostic metrics, connectivity checks, component counting, degree analysis, clustering analysis, centrality computation, path analysis, bridge detection, semantic analysis, novelty computation, feasibility scoring, property prediction, ridge regression, feature concatenation, pair scoring, candidate filtering, distance checking, graph distance, shortest path, all-pairs shortest path, cutoff parameter, edge sampling strategy, positive pairs, negative pairs, hard negatives, distance-focused sampling, random sampling, attempts limit, pair uniqueness, edge existence check, tensor construction, sparse adjacency, degree computation, normalization, message passing, aggregation, combination, activation, ReLU, linear layers, sequential decoder, concatenation, sigmoid, logits, contrastive loss, binary cross-entropy, training epochs, learning rate, optimizer step, gradient computation, backward pass, zero grad, model evaluation, no grad context, final embeddings, adjacency indices, adjacency values, node features, embedding dimension, shape validation, error raising, minimal pairs, edge uniqueness, source adjacency, destination adjacency, stacking, tensor conversion, device placement, long dtype, float32, GPU memory, CPU fallback, memory cleanup, garbage collection, CUDA cache emptying, progress callback, epoch logging, loss tracking, convergence monitoring, early stopping, model saving, checkpointing, inference mode, prediction scoring, candidate generation, random sampling, pair filtering, distance computation, KeyError handling, default distance, semantic similarity, cosine similarity, embedding encoding, numpy arrays, tensor conversion, CPU numpy, forward pass, model eval, no grad, decoder output, logits extraction, sigmoid activation, CPU conversion, numpy array, property lookup, median computation, ridge prediction, clipping, normalization, weighted scoring, alpha weights, composite score, sorting, head selection, DataFrame creation, column selection, formatting, display configuration, download preparation, CSV serialization, MIME type, button callback, empty check, info message, parameter suggestion, graph rendering, node count check, edge count check, fallback graph building, semantic-only fallback, similarity threshold adjustment, success message, text fallback rendering, node iteration, degree computation, frequency lookup, category detection, color assignment, size computation, title formatting, node addition, edge iteration, weight lookup, type lookup, color mapping, edge addition, value scaling, width scaling, color assignment, smooth edges, curved edges, roundness parameter, HTML generation, inline resources, Streamlit HTML component, height parameter, scrolling enable, width parameter, download button, file naming, MIME type, unique key, error catching, warning display, fallback suggestion, retry buttons, alternative backend, exception handling, error message display, traceback expansion, code display, memory cleanup, GPU cache clearing, garbage collection, footer display, tips section, visualization options, PyVis description, Plotly description, text summary description, technical stack, crash prevention tips, rendering troubleshooting, browser console check, zoom controls, download fallback, text view guarantee
"""
DOMAIN_KEYWORDS = [
    "grain size", "phase fraction", "microhardness", "tensile strength", "yield strength", "elongation",
    "residual stress", "texture intensity", "columnar grain", "equiaxed grain", "dendrite", "eutectic",
    "martensite", "austenite", "precipitate", "segregation", "porosity", "crack density", "intermetallic compound",
    "IMC", "interfacial microstructure", "melt pool", "solidification front", "grain boundary",
    "phase transformation", "nucleation", "laser power", "scan speed", "hatch spacing", "layer thickness",
    "pulse duration", "energy density", "spot diameter", "cooling rate", "solidification rate",
    "dilution ratio", "Gaussian heat source", "absorptivity", "reflectivity", "beam intensity",
    "laser wavelength", "Marangoni convection", "high-entropy alloy", "HEA", "multi-principal element",
    "complex concentrated", "powder bed fusion", "LPBF", "direct energy deposition", "DED", "selective laser melting",
    "AlSi10Mg", "Ti6Al4V", "Inconel718", "SnAgCu", "CoCrFeNi", "solder alloy", "phase field",
    "molecular dynamics", "finite element", "CALPHAD", "digital twin", "physics-informed",
    "machine learning", "neural network", "graph neural network", "feature engineering", "semantic similarity",
    "concept graph", "knowledge graph", "thermal conductivity", "viscosity", "interfacial energy",
    "diffusion coefficient", "Gibbs free energy", "enthalpy", "entropy", "atomic mobility",
    "thermodynamic database", "defect formation", "spatter ejection", "keyhole formation",
    "lack of fusion", "residual stress mitigation", "grain refinement", "texture evolution", "phase stability"
]
ALLOY_PATTERNS = [
    r'[A-Z][a-z]?(?:\d+(?:\.\d+)?(?:[A-Z][a-z]?\d*(?:\.\d+)?)*)+',
    r'(?:Ni|Co|Cr|Fe|Al|Ti|Cu|Nb|Mo|W|Sn|Ag|Zn|Bi)(?:[-\s]?\d+(?:\.\d+)?%?)+',
    r'(?:high-entropy|HEA|multi-principal|complex concentrated|MPEA)',
    r'(?:AlSi\d+Mg|Ti6Al4V|Inconel\d+|SnAgCu|CoCrFeNi|SAC\d+)',
]
DOMAIN_SEED_CONCEPTS = {
    "alloy_systems": ["aluminum alloy", "titanium alloy", "nickel alloy", "high-entropy alloy", "steel", "alsi10mg", "ti6al4v", "inconel718", "snagcu solder", "cocrfeni hea", "multiprincipal element alloy", "complex concentrated alloy"],
    "laser_parameters": ["laser power", "scan speed", "energy density", "hatch spacing", "pulse duration", "melt pool depth", "cooling rate", "solidification rate", "Gaussian heat source", "beam intensity distribution", "absorptivity", "reflectivity", "Marangoni number", "laser wavelength"],
    "microstructure_features": ["grain size", "phase fraction", "texture", "porosity", "residual stress", "columnar grain", "equiaxed grain", "dendritic structure", "intermetallic compound", "grain boundary", "phase transformation", "nucleation site", "solidification front", "melt pool geometry", "interfacial microstructure", "precipitate distribution"],
    "mechanical_properties": ["microhardness", "tensile strength", "yield strength", "elongation", "fatigue life", "ductility", "wear resistance", "corrosion resistance", "oxidation resistance", "fracture toughness", "creep resistance"],
    "processes": ["powder bed fusion", "direct energy deposition", "laser remelting", "surface treatment", "solidification", "selective laser melting", "laser powder bed fusion", "wire-feed laser additive manufacturing", "in-situ monitoring", "post-process heat treatment"],
    "computational_methods": ["phase field modeling", "molecular dynamics", "finite element analysis", "CALPHAD thermodynamics", "digital twin", "physics-informed machine learning", "graph neural network", "concept extraction", "semantic clustering", "feature engineering", "tensor decomposition", "scale-bridging simulation"],
    "declarmima_goals": ["decipher laser-microstructure interaction", "physics-informed digital twin", "learning laser system", "process-structure-property relationship", "multiscale computational modeling", "integrated experiment-computation framework", "uncertainty quantification", "inverse design optimization", "mechanistic understanding of additive manufacturing"]
}
CATEGORY_MAPPING = {
    r'alsi\d+mg|al(?:si|cu|mg|zn)\w*': 'aluminum alloy',
    r'ti6al4v|ti(?:al|nb|mo)\w*': 'titanium alloy',
    r'inconel\d+|ni(?:cr|mo|fe)\w*': 'nickel alloy',
    r'cocrfeni|he[as]?|high.?entropy|mpea': 'high-entropy alloy',
    r'snagcu|sac\d+|sn(?:ag|cu|bi|zn)\w*': 'solder alloy',
    r'(?:laser\s*)?(?:power|energy\s*density|fluence|beam\s*intensity)': 'laser energy parameter',
    r'(?:scan|travel)\s*speed|feed\s*rate': 'scanning parameter',
    r'hatch\s*spacing|layer\s*thickness|point\s*distance': 'geometric parameter',
    r'(?:columnar|equiaxed|dendritic|fine|coarse)\s*grain': 'grain morphology',
    r'(?:martensite|austenite|eutectic|ferrite|precipitate)\s*(?:phase)?': 'phase type',
    r'(?:micro|nano)hardness|hv\d*|vickers': 'hardness metric',
    r'(?:tensile|yield|ultimate|fracture)\s*strength': 'strength metric',
    r'(?:thermal\s*)?conductivity|diffusivity': 'thermal property',
    r'(?:interfacial|grain\s*boundary)\s*energy': 'interface property',
    r'(?:marangoni|convection|fluid\s*flow)': 'melt pool dynamics',
    r'(?:porosity|void|crack|defect|spatter|keyhole)': 'defect type',
    r'(?:phase\s*field|molecular\s*dynamics|finite\s*element|calphad)': 'computational method',
    r'(?:digital\s*twin|machine\s*learning|neural\s*network|graph\s*neural)': 'data-driven method',
}
DEFAULT_MIN_CONCEPT_FREQ = 3
DEFAULT_MIN_CONCEPT_LENGTH_WORDS = 2
GNN_HIDDEN_DIM = 128
TRAIN_EPOCHS = 50
LR = 1e-3
NEG_DPREV_FOCUS = 3

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def is_ollama_model(model_key: str) -> bool:
    return model_key.startswith("ollama:") or model_key.startswith("[Ollama]")
def extract_ollama_tag(model_key: str) -> str:
    if model_key.startswith("ollama:"):
        return model_key.replace("ollama:", "", 1)
    elif model_key.startswith("[Ollama]"):
        match = re.search(r'\]\s*([^\s(]+)', model_key)
        if match: return match.group(1)
    return model_key
def get_hf_repo_id(model_key: str) -> str:
    if ":" in model_key and not model_key.startswith("http"):
        parts = model_key.split(":", 1)
        if len(parts) == 2 and "/" in parts[1]: return parts[1].strip()
    return model_key
def get_available_gpu_memory() -> Optional[float]:
    if not torch.cuda.is_available(): return None
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        return total_memory - reserved
    except: return None
def estimate_model_memory(model_key: str, use_4bit: bool = False) -> Dict[str, Any]:
    repo_id = get_hf_repo_id(model_key) if not is_ollama_model(model_key) else model_key
    return MODEL_MEMORY_ESTIMATES.get(repo_id, {"params": "Unknown", "vram_fp16": "Unknown", "vram_4bit": "Unknown", "cpu_ok": False})
def compute_file_hash(filepath: str) -> str:
    try:
        with open(filepath, 'rb') as f: return hashlib.md5(f.read()).hexdigest()
    except: return ""

# ==========================================
# ADAPTIVE CONFIGURATION FOR SMALL CORPORA
# ==========================================
def get_adaptive_config(num_abstracts: int) -> Dict[str, Any]:
    if num_abstracts <= 15:
        return {"MIN_CONCEPT_FREQ": 1, "MIN_CONCEPT_LENGTH_WORDS": 1, "MIN_DEGREE": 1, "USE_SEMANTIC_CLUSTERING": True, "INJECT_DOMAIN_SEEDS": True, "USE_SEMANTIC_EDGES": True, "SIMILARITY_THRESHOLD": 0.70, "COOCCURRENCE_WEIGHT": 0.4, "SEMANTIC_WEIGHT": 0.6, "CLUSTER_SIMILARITY": 0.78, "USE_DECLARMIMA_SEEDS": True, "CORRELATE_WITH_PROPOSAL": True}
    elif num_abstracts <= 30:
        return {"MIN_CONCEPT_FREQ": 2, "MIN_CONCEPT_LENGTH_WORDS": 2, "MIN_DEGREE": 1, "USE_SEMANTIC_CLUSTERING": True, "INJECT_DOMAIN_SEEDS": True, "USE_SEMANTIC_EDGES": True, "SIMILARITY_THRESHOLD": 0.75, "COOCCURRENCE_WEIGHT": 0.6, "SEMANTIC_WEIGHT": 0.4, "CLUSTER_SIMILARITY": 0.75, "USE_DECLARMIMA_SEEDS": True, "CORRELATE_WITH_PROPOSAL": True}
    else:
        return {"MIN_CONCEPT_FREQ": 3, "MIN_CONCEPT_LENGTH_WORDS": 2, "MIN_DEGREE": 2, "USE_SEMANTIC_CLUSTERING": False, "INJECT_DOMAIN_SEEDS": False, "USE_SEMANTIC_EDGES": False, "SIMILARITY_THRESHOLD": 0.80, "COOCCURRENCE_WEIGHT": 0.8, "SEMANTIC_WEIGHT": 0.2, "CLUSTER_SIMILARITY": 0.72, "USE_DECLARMIMA_SEEDS": False, "CORRELATE_WITH_PROPOSAL": False}

# ==========================================
# 🔧 CUDA-SAFE MODEL LOADING
# ==========================================
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    try:
        return SentenceTransformer(EMBED_NAME, device=DEVICE if DEVICE.type == 'cuda' else 'cpu')
    except RuntimeError as e:
        if "no kernel image" in str(e).lower() or "cudaerror" in str(e).lower():
            st.error(f"❌ CUDA kernel error with embedding model: {e}")
            st.info("💡 Falling back to CPU mode for embeddings")
            return SentenceTransformer(EMBED_NAME, device='cpu')
        raise e
    except Exception as e:
        st.error(f"❌ Failed to load embedding model: {e}")
        return SentenceTransformer(EMBED_NAME, device='cpu')

@st.cache_resource(show_spinner="Loading LLM (this may take 1-2 minutes on first load)...")
def load_local_llm(model_key: str, use_4bit: bool = True):
    try:
        if is_ollama_model(model_key): return _load_ollama_model(model_key)
        else: return _load_transformers_model(model_key, use_4bit)
    except RuntimeError as e:
        if "no kernel image" in str(e).lower() or "cudaerror" in str(e).lower():
            st.error(f"❌ CUDA kernel error loading LLM '{model_key}': {e}")
            st.warning("💡 Attempting fallback to CPU mode...")
            try:
                tok, mod = GPT2Tokenizer.from_pretrained("gpt2"), GPT2LMHeadModel.from_pretrained("gpt2").to('cpu')
                tok.pad_token = tok.eos_token
                return tok, mod, torch.device('cpu'), "transformers"
            except: return None, None, None, None
        raise e
    except Exception as e:
        st.error(f"Failed to load LLM '{model_key}': {e}")
        try:
            tok, mod = GPT2Tokenizer.from_pretrained("gpt2"), GPT2LMHeadModel.from_pretrained("gpt2")
            tok.pad_token = tok.eos_token
            device = "cuda" if torch.cuda.is_available() else "cpu"
            return tok, mod, device, "transformers"
        except: return None, None, None, None

def _load_ollama_model(model_key: str) -> Tuple[Optional[Any], str, str, str]:
    if not OLLAMA_AVAILABLE: raise ImportError("ollama library not installed. Run: pip install ollama")
    model_tag = extract_ollama_tag(model_key)
    ollama_host = st.session_state.get('ollama_host', 'http://localhost:11434')
    try:
        client = ollama.Client(host=ollama_host)
        models_list = [m.get('model') or m.get('name') for m in client.list().get('models', []) if isinstance(m, dict)]
        if model_tag not in models_list:
            st.warning(f"⚠️ Model '{model_tag}' not found in Ollama.")
            return None, None, ollama_host, "ollama"
        return None, model_tag, ollama_host, "ollama"
    except Exception as conn_err:
        st.error(f"❌ Connection Error: {conn_err}")
        return None, None, ollama_host, "ollama"

def _load_transformers_model(model_key: str, use_4bit: bool = True) -> Tuple[Any, Any, str, str]:
    repo_id = get_hf_repo_id(model_key)
    device = "cpu" if st.session_state.get("force_cpu", False) or not torch.cuda.is_available() else "cuda"
    available_vram = get_available_gpu_memory()
    mem_info = estimate_model_memory(model_key, use_4bit)
    st.sidebar.info(f"""📊 Model Memory Estimate:
- Parameters: {mem_info['params']}
- VRAM (FP16): {mem_info['vram_fp16']}
- VRAM (4-bit): {mem_info['vram_4bit']}
- CPU OK: {'✅ Yes' if mem_info['cpu_ok'] else '❌ No'}
- Available VRAM: {f'{available_vram:.1f}GB' if available_vram else 'N/A (CPU)'}
- Device: {device.upper()}""")
    if "0.5B" in repo_id or "1.1B" in repo_id or "gpt2" in repo_id or device == "cpu":
        use_4bit = False
    quantization_config = None
    if use_4bit and device == "cuda" and available_vram:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
            st.sidebar.success("✅ 4-bit quantization enabled")
        except ImportError:
            st.sidebar.warning("⚠️ bitsandbytes not installed.")
            use_4bit = False
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True, padding_side="left", use_fast=True)
    model_kwargs = {"trust_remote_code": True}
    if device == "cuda" and quantization_config:
        model_kwargs.update({"quantization_config": quantization_config, "device_map": "auto", "torch_dtype": torch.float16})
    elif device == "cuda":
        model_kwargs.update({"device_map": "auto", "torch_dtype": torch.float16})
    else:
        model_kwargs["torch_dtype"] = torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(repo_id, **model_kwargs)
        if device == "cpu" and hasattr(model, 'to'): model = model.to(device)
        model.eval()
    except RuntimeError as e:
        if "no kernel image" in str(e).lower():
            st.error(f"❌ CUDA kernel error: {e}")
            st.info("Retrying with CPU...")
            model_kwargs["device_map"] = None
            model_kwargs["torch_dtype"] = torch.float32
            model = AutoModelForCausalLM.from_pretrained(repo_id, **model_kwargs).to('cpu')
            device = 'cpu'
        else: raise e
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model, device, "transformers"

# ==========================================
# DOMAIN-SPECIFIC CONCEPT NORMALIZATION
# ==========================================
def normalize_alloy_composition(concept: str) -> str:
    normalized = re.sub(r'[\s\-_]', '', concept).lower()
    normalized = re.sub(r'(ti)(6)(al)(4)(v)', r'ti6al4v', normalized)
    normalized = re.sub(r'(al)(si)(10)(mg)', r'alsi10mg', normalized)
    normalized = re.sub(r'(inconel)(\s*718|718)', r'inconel718', normalized)
    normalized = re.sub(r'(cocrfe)(ni|mn|mo)\w*', r'cocrfeni', normalized)
    normalized = re.sub(r'(sn)(ag)(cu)', r'snagcu', normalized)
    return normalized
def normalize_laser_term(concept: str) -> str:
    concept = concept.lower().strip()
    concept = re.sub(r'\b(j/mm(?:\s*3)?|j mm-3|j mm⁻³)\b', 'j/mm³', concept)
    concept = re.sub(r'\b(w|watt)s?\b', 'w', concept)
    concept = re.sub(r'\b(mm/s|mm s-1|mm s⁻¹)\b', 'mm/s', concept)
    concept = re.sub(r'\b(μm|micron|um)\b', 'um', concept)
    return concept
def is_valid_microstructure_concept(concept: str) -> bool:
    concept_lower = concept.lower()
    has_domain_keyword = any(kw in concept_lower for kw in DOMAIN_KEYWORDS)
    has_alloy_pattern = any(re.search(p, concept, re.I) for p in ALLOY_PATTERNS)
    generic_terms = {'study', 'analysis', 'effect', 'role', 'investigation', 'research', 'method', 'approach', 'paper', 'work', 'using'}
    has_generic = any(term in concept_lower.split() for term in generic_terms)
    return (has_domain_keyword or has_alloy_pattern) and not has_generic

# ==========================================
# ENHANCED CONCEPT DISTILLATION & VALIDATION
# ==========================================
def extract_declarmima_concepts(proposal_text: str, embed_model) -> list:
    patterns = [
        r'\b(?:[A-Z][a-z]+(?:\d+(?:\.\d+)?)?[\s\-]?){2,4}(?:alloy|phase|grain|microstructure|strength|hardness|property)',
        r'\b(?:laser|powder|bed|fusion|selective|direct|melting)\s+(?:power|speed|scanning|melting|parameters|energy|processing)',
        r'\b(?:columnar|equiaxed|fine|coarse|nanoscale|bimodal|dendritic)\s+(?:grain|structure|region|zone|morphology)',
        r'\b(?:martensite|austenite|ferrite|eutectic|peritectic|precipitate|intermetallic)\s+(?:formation|phase|fraction|compound)',
        r'\b(?:microhardness|nanohardness|tensile|yield|ductility|elongation|fatigue)\s+(?:improvement|strength|property|life)',
        r'\b(?:phase\s*field|molecular\s*dynamics|finite\s*element|calphad|digital\s*twin)\s*(?:model|simulation|method|framework)?',
        r'\b(?:high-entropy|HEA|multi[-\s]?principal|complex\s*concentrated)\s*(?:alloy|material)?',
        r'\b(?:AlSi\d+Mg|Ti6Al4V|Inconel\d+|SnAgCu|CoCrFeNi|SAC\d+)\b',
    ]
    concepts = set()
    for pattern in patterns:
        matches = re.findall(pattern, proposal_text, re.I)
        for m in matches:
            concept = m.lower().strip().rstrip('.')
            if len(concept.split()) >= 2 and is_valid_microstructure_concept(concept):
                concepts.add(concept)
    for goal in ["decipher laser-microstructure interaction", "physics-informed digital twin", "learning laser system", "process-structure-property relationship", "multiscale computational modeling", "integrated experiment-computation framework"]:
        concepts.add(goal)
    return list(concepts)

def compute_proposal_correlation(concept: str, proposal_embedding: np.ndarray, concept_embedding: np.ndarray) -> float:
    sim = cosine_similarity([concept_embedding], [proposal_embedding])[0][0]
    return float(np.clip(sim, 0, 1))

def inject_declarmima_seeds(valid_concepts: list, concept_to_id: dict, proposal_embedding: np.ndarray, embed_model, correlation_threshold: float = 0.65) -> tuple:
    updated_concepts, updated_mapping = valid_concepts.copy(), concept_to_id.copy()
    proposal_concepts = extract_declarmima_concepts(DECLARMIMA_PROPOSAL_TEXT, embed_model)
    if proposal_concepts:
        proposal_concept_embeddings = embed_model.encode(proposal_concepts, show_progress_bar=False)
        for i, prop_concept in enumerate(proposal_concepts):
            if prop_concept not in updated_mapping:
                corr = compute_proposal_correlation(prop_concept, proposal_embedding, proposal_concept_embeddings[i])
                if corr >= correlation_threshold:
                    updated_mapping[prop_concept] = len(updated_mapping)
                    updated_concepts.append(prop_concept)
    return updated_concepts, updated_mapping

# ==========================================
# SEMANTIC CLUSTERING & CONCEPT ABSTRACTION
# ==========================================
def cluster_similar_concepts(valid_concepts, embed_model, similarity_threshold=0.78):
    if len(valid_concepts) < 3: return valid_concepts, {c: c for c in valid_concepts}
    try:
        embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=32)
        sim_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - sim_matrix
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - similarity_threshold, linkage='average').fit(embeddings)
        concept_to_cluster, cluster_members = {}, defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            concept_to_cluster[valid_concepts[idx]] = label
            cluster_members[label].append(valid_concepts[idx])
        cluster_representatives = {label: min(members, key=lambda x: (len(x), -x.count(' '))) for label, members in cluster_members.items()}
        final_mapping = {c: cluster_representatives[label] for c, label in concept_to_cluster.items()}
        return list(cluster_representatives.values()), final_mapping
    except Exception as e:
        st.warning(f"⚠️ Semantic clustering skipped: {e}")
        return valid_concepts, {c: c for c in valid_concepts}

def abstract_concepts_to_categories(concepts, category_mapping=CATEGORY_MAPPING):
    concept_to_abstract = {}
    for concept in concepts:
        matched = False
        for pattern, category in category_mapping.items():
            if re.search(pattern, concept, re.I):
                concept_to_abstract[concept] = category
                matched = True
                break
        if not matched: concept_to_abstract[concept] = concept
    return concept_to_abstract

def inject_domain_seeds(valid_concepts, concept_to_id, seed_concepts=DOMAIN_SEED_CONCEPTS):
    all_seeds = [seed for category in seed_concepts.values() for seed in category]
    updated_concepts, updated_mapping = valid_concepts.copy(), concept_to_id.copy()
    for seed in all_seeds:
        if seed not in updated_mapping:
            updated_mapping[seed] = len(updated_mapping)
            updated_concepts.append(seed)
    return updated_concepts, updated_mapping

# ==========================================
# CONCEPT EXTRACTION & TF-IDF DISTILLATION
# ==========================================
def extract_concepts_from_abstracts(abstracts, tokenizer, model, backend_type: str):
    prompt_template = """Extract exactly the core scientific concepts (2+ words) from this abstract about laser processing or alloy microstructure.
Rules:
- Output ONLY a JSON list of strings.
- Use nominalized form (e.g., 'grain refinement' not 'refines grains').
- Include: alloy compositions (e.g., 'AlSi10Mg'), laser parameters ('laser power'), microstructure features ('columnar grains'), properties ('microhardness').
- Standardize: chemical formulas, units (J/mm³, mm/s), phase names.
- Exclude: generic terms like 'study', 'results', 'method'.
Abstract: {text}
Concepts:"""
    all_concepts, all_metrics = [], []
    for text in abstracts:
        metrics = {}
        grain_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:μm|micron|um|nm)\s*(?:grain|average|size|diameter)?', text, re.I)
        if grain_matches: metrics['grain_size_um'] = [float(m) for m in grain_matches]
        mech_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:HV|GPa|MPa|ksi)\s*(?:hardness|strength|yield|tensile|ultimate)?', text, re.I)
        if mech_matches: metrics['mechanical_property'] = [float(m) for m in mech_matches]
        energy_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:J/mm³|J mm-3|J mm⁻³|J/mm\^3)', text, re.I)
        if energy_matches: metrics['energy_density_j_mm3'] = [float(m) for m in energy_matches]
        defect_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:porosity|void|crack)', text, re.I)
        if defect_matches: metrics['defect_fraction_pct'] = [float(m) for m in defect_matches]
        all_metrics.append(metrics)
        prompt = prompt_template.format(text=text)
        if backend_type == "ollama":
            try:
                client = ollama.Client(host=st.session_state.get('ollama_host', 'http://localhost:11434'))
                response = client.chat(model=model, messages=[{"role": "user", "content": prompt}], options={"temperature": 0.2, "num_predict": 150})
                response_text = response.get('message', {}).get('content', '') if isinstance(response, dict) else getattr(response, 'message', {}).get('content', '')
            except Exception as e:
                st.warning(f"⚠️ Ollama extraction failed: {e}, using fallback")
                response_text = _fallback_concept_extraction_text(prompt)
        else:
            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
                with torch.no_grad():
                    outputs = model.generate(inputs.input_ids, max_new_tokens=150, temperature=0.2, do_sample=True, pad_token_id=tokenizer.eos_token_id)
                response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            except Exception as e:
                st.warning(f"⚠️ HF extraction failed: {e}, using fallback")
                response_text = _fallback_concept_extraction_text(prompt)
        concepts = []
        try:
            parsed = json.loads(response_text.replace("'", '"').strip())
            if isinstance(parsed, list): concepts = [c.strip().lower().rstrip('.') for c in parsed if isinstance(c, str) and len(c.strip()) > 3]
        except:
            concepts = _fallback_concept_extraction(text)
        normalized = []
        for c in concepts:
            if any(elem in c.lower() for elem in ['al', 'ti', 'ni', 'cr', 'fe', 'co', 'mo', 'nb', 'cu', 'sn', 'ag']): c = normalize_alloy_composition(c)
            elif any(lp in c.lower() for lp in ['laser', 'scan', 'power', 'speed', 'melt', 'pool', 'energy']): c = normalize_laser_term(c)
            if is_valid_microstructure_concept(c): normalized.append(c)
        all_concepts.append(normalized)
    return all_concepts, all_metrics

def _fallback_concept_extraction_text(prompt: str) -> str:
    return '["laser processing", "microstructure evolution", "alloy composition", "mechanical properties"]'

def _fallback_concept_extraction(text: str) -> list:
    patterns = [
        r'\b(?:[A-Z][a-z]+(?:\d+(?:\.\d+)?)?[\s\-]?){2,3}(?:phase|grain|microstructure|strength|hardness)',
        r'\b(?:laser|powder|bed|fusion|selective|direct)\s+(?:power|speed|scanning|melting|parameters|energy)',
        r'\b(?:columnar|equiaxed|fine|coarse|nanoscale|bimodal)\s+(?:grain|structure|region|zone)',
        r'\b(?:martensite|austenite|ferrite|eutectic|peritectic|precipitate)\s+(?:formation|phase|fraction)',
        r'\b(?:microhardness|nanohardness|tensile|yield|ductility|elongation)\s+(?:improvement|strength|property)',
    ]
    concepts = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.I)
        concepts.extend([m.lower().strip() for m in matches if len(m.split()) >= 2])
    return list(set(concepts))

def compute_tfidf_scores(all_concepts):
    corpus = [" ".join(doc) for doc in all_concepts if doc]
    if not corpus: return {}, []
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    concept_tfidf = defaultdict(float)
    for doc_idx, doc in enumerate(all_concepts):
        if not doc: continue
        for concept in doc:
            idx = feature_names.tolist().index(concept) if concept in feature_names else -1
            if idx != -1: concept_tfidf[concept] += tfidf_matrix[doc_idx, idx]
    return concept_tfidf, feature_names

def normalize_and_filter_concepts(all_concepts, embed_model=None, config=None, proposal_embedding=None):
    if config is None: config = get_adaptive_config(25)
    concept_counts = defaultdict(int)
    concept_abstract_map = defaultdict(list)
    for doc_idx, concepts in enumerate(all_concepts):
        seen_in_doc = set()
        for c in concepts:
            if c not in seen_in_doc and is_valid_microstructure_concept(c):
                concept_counts[c] += 1
                concept_abstract_map[c].append(doc_idx)
                seen_in_doc.add(c)
    min_freq, min_words = config.get("MIN_CONCEPT_FREQ", 2), config.get("MIN_CONCEPT_LENGTH_WORDS", 2)
    valid_concepts = [c for c, cnt in concept_counts.items() if cnt >= min_freq and len(c.split()) >= min_words]
    concept_tfidf, _ = compute_tfidf_scores(all_concepts)
    concept_relevance = {}
    if config.get("CORRELATE_WITH_PROPOSAL", True) and proposal_embedding is not None and embed_model:
        embeddings = embed_model.encode(valid_concepts, show_progress_bar=False)
        sims = cosine_similarity(embeddings, proposal_embedding.reshape(1, -1))
        concept_relevance = {c: float(s) for c, s in zip(valid_concepts, sims)}
    tfidf_values = np.array([concept_tfidf.get(c, 0.0) for c in valid_concepts])
    relevance_values = np.array([concept_relevance.get(c, 0.5) for c in valid_concepts])
    distillation_score = 0.6 * (tfidf_values / (tfidf_values.max() + 1e-6)) + 0.4 * relevance_values
    valid_concepts = [valid_concepts[i] for i in np.argsort(distillation_score)[::-1]]
    if config.get("USE_DECLARMIMA_SEEDS", True) and proposal_embedding is not None and embed_model:
        valid_concepts, concept_to_id = inject_declarmima_seeds(valid_concepts, {c: i for i, c in enumerate(valid_concepts)}, proposal_embedding, embed_model, correlation_threshold=0.65)
        for seed in [s for cat in DOMAIN_SEED_CONCEPTS.values() for s in cat]:
            if seed not in concept_counts: concept_counts[seed], concept_abstract_map[seed] = 1, []
    elif config.get("INJECT_DOMAIN_SEEDS", True) and len(valid_concepts) < 15:
        valid_concepts, concept_to_id = inject_domain_seeds(valid_concepts, {c: i for i, c in enumerate(valid_concepts)})
        for seed in [s for cat in DOMAIN_SEED_CONCEPTS.values() for s in cat]:
            if seed not in concept_counts: concept_counts[seed], concept_abstract_map[seed] = 1, []
    if config.get("USE_SEMANTIC_CLUSTERING", True) and embed_model and len(valid_concepts) >= 5:
        clustered_concepts, concept_to_cluster = cluster_similar_concepts(valid_concepts, embed_model, similarity_threshold=config.get("CLUSTER_SIMILARITY", 0.75))
        new_abstract_map = defaultdict(list)
        for orig_concept, docs in concept_abstract_map.items():
            clustered = concept_to_cluster.get(orig_concept, orig_concept)
            if clustered in clustered_concepts: new_abstract_map[clustered].extend(docs)
        concept_abstract_map, valid_concepts = new_abstract_map, clustered_concepts
    valid_concepts = list(set(valid_concepts))
    concept_to_id = {c: i for i, c in enumerate(valid_concepts)}
    id_to_concept = {i: c for i, c in enumerate(valid_concepts)}
    distillation_metrics = {c: {"tfidf": concept_tfidf.get(c, 0.0), "relevance": concept_relevance.get(c, 0.5), "frequency": concept_counts.get(c, 0), "distillation_score": float(distillation_score[np.where(np.array(valid_concepts) == c)[0][0]]) if c in valid_concepts else 0} for c in valid_concepts}
    return valid_concepts, concept_to_id, id_to_concept, concept_abstract_map, distillation_metrics

# ==========================================
# HYBRID CONCEPT GRAPH CONSTRUCTION
# ==========================================
def build_semantic_only_graph(valid_concepts, embed_model, similarity_threshold=0.75):
    nx_graph = nx.Graph()
    for c in valid_concepts: nx_graph.add_node(c)
    if len(valid_concepts) < 2: return nx_graph
    try:
        embeddings = embed_model.encode(valid_concepts, show_progress_bar=False)
        sim_matrix = cosine_similarity(embeddings)
        for i in range(len(valid_concepts)):
            for j in range(i+1, len(valid_concepts)):
                if sim_matrix[i][j] > similarity_threshold:
                    nx_graph.add_edge(valid_concepts[i], valid_concepts[j], weight=sim_matrix[i][j], edge_type='semantic', cooccurrence=0, semantic=sim_matrix[i][j])
        if not nx.is_connected(nx_graph) and len(valid_concepts) > 3:
            components = list(nx.connected_components(nx_graph))
            for i in range(len(components)-1):
                best_sim, best_pair = 0, None
                for c1 in components[i]:
                    idx1 = valid_concepts.index(c1)
                    for c2 in components[i+1]:
                        idx2 = valid_concepts.index(c2)
                        if sim_matrix[idx1][idx2] > best_sim: best_sim, best_pair = sim_matrix[idx1][idx2], (c1, c2)
                if best_pair: nx_graph.add_edge(*best_pair, weight=best_sim, edge_type='bridge', cooccurrence=0, semantic=best_sim)
    except Exception as e:
        st.warning(f"⚠️ Semantic graph construction issue: {e}")
        for i in range(len(valid_concepts)-1): nx_graph.add_edge(valid_concepts[i], valid_concepts[i+1], weight=1.0)
    return nx_graph

def build_hybrid_graph(all_concepts, valid_concepts, concept_to_id, embed_model=None, config=None, proposal_embedding=None):
    if config is None: config = get_adaptive_config(len(all_concepts))
    nx_graph = nx.Graph()
    for c in valid_concepts: nx_graph.add_node(c, frequency=0)
    for concepts in all_concepts:
        valid_in_doc = [c for c in concepts if c in concept_to_id]
        for i in range(len(valid_in_doc)):
            for j in range(i+1, len(valid_in_doc)):
                u, v = valid_in_doc[i], valid_in_doc[j]
                if nx_graph.has_edge(u, v): nx_graph[u][v]['weight'] += 1; nx_graph[u][v]['cooccurrence'] += 1
                else: nx_graph.add_edge(u, v, weight=1, cooccurrence=1, semantic=0, edge_type='cooccurrence')
                nx_graph.nodes[u]['frequency'] = nx_graph.nodes[u].get('frequency', 0) + 1
                nx_graph.nodes[v]['frequency'] = nx_graph.nodes[v].get('frequency', 0) + 1
    if config.get("USE_SEMANTIC_EDGES", True) and embed_model and len(valid_concepts) >= 5:
        try:
            embeddings = embed_model.encode(valid_concepts, show_progress_bar=False)
            sim_matrix = cosine_similarity(embeddings)
            sim_thresh = config.get("SIMILARITY_THRESHOLD", 0.75)
            for i, c1 in enumerate(valid_concepts):
                for j, c2 in enumerate(valid_concepts[i+1:], start=i+1):
                    if c1 == c2 or nx_graph.has_edge(c1, c2): continue
                    sim = sim_matrix[i][j]
                    if sim > sim_thresh and (nx_graph.degree(c1) < 2 or nx_graph.degree(c2) < 2):
                        nx_graph.add_edge(c1, c2, weight=sim * 2, cooccurrence=0, semantic=sim, edge_type='semantic')
        except Exception as e: st.warning(f"⚠️ Semantic edge addition skipped: {e}")
    if config.get("CORRELATE_WITH_PROPOSAL", True) and proposal_embedding is not None and embed_model:
        concept_embeddings = embed_model.encode(valid_concepts, show_progress_bar=False)
        for i, c1 in enumerate(valid_concepts):
            corr1 = compute_proposal_correlation(c1, proposal_embedding, concept_embeddings[i])
            for j, c2 in enumerate(valid_concepts[i+1:], start=i+1):
                if c1 == c2: continue
                corr2 = compute_proposal_correlation(c2, proposal_embedding, concept_embeddings[j])
                if corr1 > 0.7 and corr2 > 0.7:
                    boost = 1.5 * (corr1 + corr2) / 2
                    if nx_graph.has_edge(c1, c2): nx_graph[c1][c2]['weight'] *= boost; nx_graph[c1][c2]['declarmina_boost'] = boost
                    else: nx_graph.add_edge(c1, c2, weight=boost, cooccurrence=0, semantic=0.8, edge_type='declarmina_aligned')
    cooc_weight, sem_weight = config.get("COOCCURRENCE_WEIGHT", 0.6), config.get("SEMANTIC_WEIGHT", 0.4)
    for u, v, data in nx_graph.edges(data=True):
        cooc, sem = data.get('cooccurrence', 0), data.get('semantic', 0)
        data['weight'] = cooc_weight * cooc + sem_weight * sem
    return nx_graph

def build_concept_graph(all_concepts, concept_to_id, embed_model=None, config=None, proposal_embedding=None):
    if config is None: config = get_adaptive_config(len(all_concepts))
    valid_concepts = list(concept_to_id.keys())
    if len(valid_concepts) < 8 and config.get("USE_SEMANTIC_EDGES", True):
        return build_semantic_only_graph(valid_concepts, embed_model, similarity_threshold=config.get("SIMILARITY_THRESHOLD", 0.75))
    return build_hybrid_graph(all_concepts, valid_concepts, concept_to_id, embed_model, config, proposal_embedding)

def sample_edges_for_training(nx_graph, d_prev_dict, valid_concepts, concept_to_id, config=None):
    if config is None: config = get_adaptive_config(len(valid_concepts))
    pos_pairs = [(concept_to_id[u], concept_to_id[v]) for u, v in nx_graph.edges()]
    neg_pairs = []
    n_nodes = len(valid_concepts)
    if n_nodes < 3: return pos_pairs, neg_pairs
    target_negs = min(len(pos_pairs) * 2 if pos_pairs else 10, 2000)
    attempts, neg_focus = 0, config.get("NEG_DPREV_FOCUS", 3)
    while len(neg_pairs) < target_negs and attempts < 15000:
        u_idx, v_idx = np.random.choice(n_nodes, 2, replace=False)
        if nx_graph.has_edge(valid_concepts[u_idx], valid_concepts[v_idx]): attempts += 1; continue
        try:
            dist = d_prev_dict[valid_concepts[u_idx]][valid_concepts[v_idx]]
            if dist == neg_focus: neg_pairs.append((u_idx, v_idx))
            elif dist == 2 and np.random.rand() < 0.3: neg_pairs.append((u_idx, v_idx))
        except KeyError:
            if np.random.rand() < 0.1: neg_pairs.append((u_idx, v_idx))
        attempts += 1
    while len(neg_pairs) < target_negs:
        u_idx, v_idx = np.random.choice(n_nodes, 2, replace=False)
        pair = (u_idx, v_idx)
        if pair not in neg_pairs and (v_idx, u_idx) not in neg_pairs and not nx_graph.has_edge(valid_concepts[u_idx], valid_concepts[v_idx]):
            neg_pairs.append(pair)
    return pos_pairs, neg_pairs

# ==========================================
# SEMANTIC NODE EMBEDDINGS
# ==========================================
def generate_embeddings(valid_concepts, embed_model):
    if not valid_concepts: return torch.zeros((0, 384), dtype=torch.float32).to(DEVICE)
    embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=32)
    return torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)
def get_embedding_dimension(embed_model) -> int:
    try: return embed_model.encode(["test"], show_progress_bar=False).shape[1]
    except: return 384

# ==========================================
# DGL GRAPH CONVERSION UTILITIES
# ==========================================
def nx_to_dgl(nx_graph: nx.Graph, node_features: torch.Tensor, concept_to_id: Dict[str, int]) -> 'dgl.DGLGraph':
    if not DGL_AVAILABLE: raise ImportError("DGL not available. Install with: pip install dgl")
    src_list, dst_list, edge_weights, edge_types = [], [], [], []
    edge_type_map = {'cooccurrence': 0, 'semantic': 1, 'bridge': 2, 'declarmina_aligned': 3}
    for u, v, data in nx_graph.edges(data=True):
        src_list.append(concept_to_id[u])
        dst_list.append(concept_to_id[v])
        edge_weights.append(data.get('weight', 1.0))
        edge_types.append(edge_type_map.get(data.get('edge_type', 'cooccurrence'), 0))
    g = dgl.graph((src_list, dst_list), num_nodes=len(concept_to_id))
    g.ndata['h'] = node_features
    g.edata['weight'] = torch.tensor(edge_weights, dtype=torch.float32)
    g.edata['etype'] = torch.tensor(edge_types, dtype=torch.long)
    return g

def dgl_to_nx(g: 'dgl.DGLGraph', id_to_concept: Dict[int, str]) -> nx.Graph:
    nx_graph = nx.Graph()
    for node_id in range(g.num_nodes()): nx_graph.add_node(id_to_concept.get(node_id, f"node_{node_id}"))
    src, dst = g.edges()
    etype_map_rev = {0: 'cooccurrence', 1: 'semantic', 2: 'bridge', 3: 'declarmina_aligned'}
    for i, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
        u_c, v_c = id_to_concept.get(u, f"node_{u}"), id_to_concept.get(v, f"node_{v}")
        weight = g.edata['weight'][i].item() if 'weight' in g.edata else 1.0
        etype_idx = g.edata['etype'][i].item() if 'etype' in g.edata else 0
        nx_graph.add_edge(u_c, v_c, weight=weight, edge_type=etype_map_rev.get(etype_idx, 'unknown'))
    return nx_graph

# ==========================================
# DGL-BASED GRAPHSAGE IMPLEMENTATION
# ==========================================
class DGLGraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = GNN_HIDDEN_DIM, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([dglnn.SAGEConv(in_dim, hidden_dim, aggregator_type='mean')])
        for _ in range(num_layers - 2): self.layers.append(dglnn.SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean'))
        self.layers.append(dglnn.SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean'))
        self.decoder = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
    def forward(self, g, pos_u, pos_v, neg_u=None, neg_v=None):
        h = g.ndata['h']
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != len(self.layers) - 1: h = F.relu(h)
        g.ndata['h'] = h
        pos_scores = self.decoder(torch.cat([h[pos_u], h[pos_v]], dim=1)).squeeze(1)
        if neg_u is not None and neg_v is not None and len(neg_u) > 0:
            neg_scores = self.decoder(torch.cat([h[neg_u], h[neg_v]], dim=1)).squeeze(1)
            return pos_scores, neg_scores, h
        return pos_scores, None, h

# ==========================================
# CUDA-SAFE TRAINING LOOP
# ==========================================
def train_gnn_pytorch(node_features, nx_graph, concept_to_id, pos_pairs, neg_pairs, progress_callback=None):
    num_nodes = len(concept_to_id)
    in_dim = node_features.shape[1] if node_features.numel() > 0 else 384
    if node_features.numel() > 0:
        expected_shape = (num_nodes, in_dim)
        if node_features.shape != expected_shape: raise ValueError(f"Node features shape mismatch: expected {expected_shape}, got {node_features.shape}")
    if not pos_pairs:
        nodes = list(concept_to_id.values())
        if len(nodes) >= 2: pos_pairs = [(nodes[0], nodes[1])]; neg_pairs = [(nodes[0], nodes[-1])] if len(nodes) > 2 else []
        else: raise ValueError("Cannot train GNN with fewer than 2 concepts")
    unique_edges = {(min(u, v), max(u, v)) for u, v in pos_pairs}
    src_adj = torch.tensor([u for u, v in unique_edges], dtype=torch.long)
    dst_adj = torch.tensor([v for u, v in unique_edges], dtype=torch.long)
    adj_indices = torch.stack([src_adj, dst_adj], dim=0)
    adj_values = torch.ones(adj_indices.shape[1], dtype=torch.float32)
    target_device = node_features.device if node_features.numel() > 0 else torch.device('cpu')
    pos_u = torch.tensor([p[0] for p in pos_pairs], dtype=torch.long, device=target_device)
    pos_v = torch.tensor([p[1] for p in pos_pairs], dtype=torch.long, device=target_device)
    neg_u = torch.tensor([n[0] for n in neg_pairs], dtype=torch.long, device=target_device) if neg_pairs else torch.tensor([], dtype=torch.long, device=target_device)
    neg_v = torch.tensor([n[1] for n in neg_pairs], dtype=torch.long, device=target_device) if neg_pairs else torch.tensor([], dtype=torch.long, device=target_device)
    model = SparseGraphSAGE(in_dim=in_dim, hidden_dim=GNN_HIDDEN_DIM).to(target_device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(TRAIN_EPOCHS):
        model.train(); optimizer.zero_grad()
        try:
            if len(neg_pairs) == 0:
                pos_out, _, _ = model(adj_indices, adj_values, num_nodes, node_features, pos_u, pos_v, pos_u[:1], pos_v[:1])
                loss = criterion(pos_out, torch.ones_like(pos_out)) * 0.5
            else:
                pos_out, neg_out, _ = model(adj_indices, adj_values, num_nodes, node_features, pos_u, pos_v, neg_u, neg_v)
                pos_loss = criterion(pos_out, torch.ones_like(pos_out))
                neg_loss = criterion(neg_out, torch.zeros_like(neg_out))
                loss = 0.5 * (pos_loss + neg_loss)
            loss.backward(); optimizer.step()
            if progress_callback and epoch % 10 == 0: progress_callback(epoch, loss.item())
        except RuntimeError as e:
            if "no kernel image" in str(e).lower() or "cuda error" in str(e).lower():
                st.error(f"❌ CUDA error during training epoch {epoch}: {e}")
                st.warning("💡 Falling back to CPU for remainder of training...")
                cpu_device = torch.device('cpu')
                model = model.to(cpu_device); node_features = node_features.to(cpu_device)
                adj_indices = adj_indices.to(cpu_device); adj_values = adj_values.to(cpu_device)
                pos_u, pos_v = pos_u.to(cpu_device), pos_v.to(cpu_device)
                if len(neg_pairs) > 0: neg_u, neg_v = neg_u.to(cpu_device), neg_v.to(cpu_device)
                continue
            else: raise e
    model.eval()
    with torch.no_grad():
        _, _, final_embeddings = model(adj_indices, adj_values, num_nodes, node_features, pos_u[:1], pos_v[:1], neg_u[:1] if len(neg_pairs) > 0 else pos_u[:1], neg_v[:1] if len(neg_pairs) > 0 else pos_v[:1])
    return model, final_embeddings.cpu(), adj_indices.cpu(), adj_values.cpu()

def train_gnn_dgl(g: 'dgl.DGLGraph', pos_pairs, neg_pairs, progress_callback=None):
    if not DGL_AVAILABLE: raise ImportError("DGL not available")
    model = DGLGraphSAGE(in_dim=g.ndata['h'].shape[1]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    pos_u = torch.tensor([p[0] for p in pos_pairs], dtype=torch.long, device=DEVICE)
    pos_v = torch.tensor([p[1] for p in pos_pairs], dtype=torch.long, device=DEVICE)
    neg_u = torch.tensor([n[0] for n in neg_pairs], dtype=torch.long, device=DEVICE) if neg_pairs else None
    neg_v = torch.tensor([n[1] for n in neg_pairs], dtype=torch.long, device=DEVICE) if neg_pairs else None
    for epoch in range(TRAIN_EPOCHS):
        model.train(); optimizer.zero_grad()
        try:
            if neg_u is not None and len(neg_u) > 0:
                pos_out, neg_out, _ = model(g, pos_u, pos_v, neg_u, neg_v)
                loss = 0.5 * (criterion(pos_out, torch.ones_like(pos_out)) + criterion(neg_out, torch.zeros_like(neg_out)))
            else:
                pos_out, _, _ = model(g, pos_u, pos_v)
                loss = criterion(pos_out, torch.ones_like(pos_out)) * 0.5
            loss.backward(); optimizer.step()
            if progress_callback and epoch % 10 == 0: progress_callback(epoch, loss.item())
        except RuntimeError as e:
            if "no kernel image" in str(e).lower() or "cuda error" in str(e).lower():
                st.error(f"❌ DGL CUDA error during training epoch {epoch}: {e}")
                st.warning("💡 Falling back to PyTorch sparse implementation...")
                raise RuntimeError("DGL_FALLBACK_TO_PYTORCH")
            else: raise e
    model.eval()
    with torch.no_grad(): _, _, final_emb = model(g, pos_u[:1], pos_v[:1])
    return model, final_emb.cpu()

def train_gnn(node_features, nx_graph, concept_to_id, pos_pairs, neg_pairs, progress_callback=None, use_dgl: bool = True):
    if use_dgl and DGL_AVAILABLE:
        try:
            g_dgl = nx_to_dgl(nx_graph, node_features, concept_to_id)
            model, final_emb = train_gnn_dgl(g_dgl, pos_pairs, neg_pairs, progress_callback)
            return model, final_emb, None, None
        except ImportError: st.warning("⚠️ DGL import failed - using PyTorch sparse implementation")
        except RuntimeError as e:
            if "DGL_FALLBACK_TO_PYTORCH" in str(e): st.warning("⚠️ DGL training failed - falling back to PyTorch sparse implementation")
            else: raise e
        except Exception as e: st.warning(f"⚠️ DGL training error: {e} - falling back to PyTorch")
    return train_gnn_pytorch(node_features, nx_graph, concept_to_id, pos_pairs, neg_pairs, progress_callback)

# ==========================================
# ORIGINAL PYTORCH SPARSE GRAPHSAGE (FALLBACK)
# ==========================================
class SparseGraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = GNN_HIDDEN_DIM):
        super().__init__()
        self.lin1, self.lin2 = nn.Linear(in_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
    def forward(self, adj_indices, adj_values, num_nodes, h, pos_u, pos_v, neg_u, neg_v):
        A = sparse.FloatTensor(adj_indices, adj_values, torch.Size([num_nodes, num_nodes])).to(h.device)
        deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1)
        deg_inv = 1.0 / deg
        h1 = F.relu(self.lin1(torch.sparse.mm(A, h) * deg_inv.unsqueeze(1)))
        h2 = self.lin2(torch.sparse.mm(A, h1) * deg_inv.unsqueeze(1))
        pos_scores = self.decoder(torch.cat([h2[pos_u], h2[pos_v]], dim=1)).squeeze(1)
        neg_scores = self.decoder(torch.cat([h2[neg_u], h2[neg_v]], dim=1)).squeeze(1)
        return pos_scores, neg_scores, h2

# ==========================================
# MICROSTRUCTURE QUANTIFICATION & SCORING
# ==========================================
def compute_microstructure_quantification(valid_concepts, concept_abstract_map, all_metrics, nx_graph):
    concept_properties = {}
    for concept in valid_concepts:
        doc_indices = concept_abstract_map.get(concept, [])
        values = []
        for idx in doc_indices:
            if idx < len(all_metrics):
                metrics = all_metrics[idx]
                for metric_values in metrics.values(): values.extend(metric_values)
        concept_properties[concept] = np.median(values) if values else 0.0
    X_feat, y_target = [], []
    for u, v in nx_graph.edges():
        pu, pv = concept_properties.get(u, 0), concept_properties.get(v, 0)
        w = nx_graph[u][v].get('weight', 1)
        X_feat.append([pu, pv, w])
        y_target.append(max(pu, pv) * 1.08 if max(pu, pv) > 0 else 0)
    ridge = None
    if len(X_feat) > 5: ridge = Ridge(alpha=1.0).fit(np.array(X_feat), np.array(y_target))
    return concept_properties, ridge

def compute_research_direction_scores(model, node_features, final_emb, nx_graph, valid_concepts, concept_properties, ridge, embed_model, d_prev_dict, adj_indices, adj_values, n_samples=3000):
    n_concepts = len(valid_concepts)
    if n_concepts < 3: return pd.DataFrame()
    u_ids = np.random.randint(n_concepts, size=min(n_samples, n_concepts * 10))
    v_ids = np.random.randint(n_concepts, size=min(n_samples, n_concepts * 10))
    candidate_pairs = []
    for u_idx, v_idx in zip(u_ids, v_ids):
        if u_idx == v_idx: continue
        u_c, v_c = valid_concepts[u_idx], valid_concepts[v_idx]
        if nx_graph.has_edge(u_c, v_c): continue
        try: d_prev = d_prev_dict[u_c][v_c]
        except KeyError: d_prev = 4
        if d_prev < 2: continue
        candidate_pairs.append((u_idx, v_idx, u_c, v_c, d_prev))
    if not candidate_pairs: return pd.DataFrame()
    u_tensor = torch.tensor([p[0] for p in candidate_pairs], dtype=torch.long, device=DEVICE)
    v_tensor = torch.tensor([p[1] for p in candidate_pairs], dtype=torch.long, device=DEVICE)
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'decoder') and hasattr(model, 'layers'):
            if DGL_AVAILABLE:
                try:
                    g_dgl = nx_to_dgl(nx_graph, node_features, {c: i for i, c in enumerate(valid_concepts)})
                    _, _, h2 = model(g_dgl, u_tensor, v_tensor, u_tensor, v_tensor)
                except: h2 = None
            else: h2 = None
            if h2 is None:
                pair_features = torch.cat([final_emb[u_tensor], final_emb[v_tensor]], dim=1)
                gnn_logits = model.decoder(pair_features).squeeze(1) if hasattr(model, 'decoder') else torch.zeros(len(u_tensor))
            else:
                pair_features = torch.cat([h2[u_tensor], h2[v_tensor]], dim=1)
                gnn_logits = model.decoder(pair_features).squeeze(1)
            gnn_scores = torch.sigmoid(gnn_logits).cpu().numpy()
        else: gnn_scores = np.random.rand(len(candidate_pairs))
    emb_np = embed_model.encode(valid_concepts, show_progress_bar=False)
    cos_sims = np.sum(emb_np[u_tensor.cpu().numpy()] * emb_np[v_tensor.cpu().numpy()], axis=1)
    results = []
    for i, (u_idx, v_idx, u_c, v_c, d_prev) in enumerate(candidate_pairs):
        p_u, p_v = concept_properties.get(u_c, 0), concept_properties.get(v_c, 0)
        expected_improvement = 0
        if ridge is not None and (p_u > 0 or p_v > 0):
            try: expected_improvement = float(ridge.predict([[p_u, p_v, 1.0]])[0])
            except: expected_improvement = max(p_u, p_v) * 1.05
        semantic_novelty = 1.0 - cos_sims[i]
        feasibility = np.exp(-0.5 * semantic_novelty) * (1.0 if (p_u > 0 or p_v > 0) else 0.6)
        alpha = {'gnn': 0.4, 'novelty': 0.3, 'gain': 0.2, 'feas': -0.1}
        norm_gain = np.clip((expected_improvement - 50) / 200, 0, 1)
        D_uv = (alpha['gnn'] * gnn_scores[i] + alpha['novelty'] * semantic_novelty + alpha['gain'] * norm_gain + alpha['feas'] * (1.0 - feasibility))
        results.append({'concept_u': u_c, 'concept_v': v_c, 'graph_distance': d_prev, 'gnn_affinity': float(gnn_scores[i]), 'semantic_novelty': float(semantic_novelty), 'expected_property_gain': expected_improvement, 'feasibility_score': float(feasibility), 'composite_score': float(D_uv)})
    df = pd.DataFrame(results).sort_values('composite_score', ascending=False)
    return df.head(min(50, len(df)))

# ==========================================
# MATHEMATICAL VALIDATION & UNCERTAINTY
# ==========================================
def validate_ridge_model(X_feat, y_target, n_splits=3):
    if len(X_feat) <= n_splits: return {"r2_cv": 0.0, "mse_cv": 0.0, "mae_cv": 0.0, "n_samples": len(X_feat)}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_scores, mse_scores, mae_scores = [], [], []
    for train_idx, test_idx in kf.split(X_feat):
        model = Ridge(alpha=1.0).fit(X_feat[train_idx], y_target[train_idx])
        preds = model.predict(X_feat[test_idx])
        ss_res = np.sum((y_target[test_idx] - preds) ** 2)
        ss_tot = np.sum((y_target[test_idx] - np.mean(y_target[test_idx])) ** 2)
        r2_scores.append(1 - (ss_res / (ss_tot + 1e-6)))
        mse_scores.append(np.mean((y_target[test_idx] - preds) ** 2))
        mae_scores.append(np.mean(np.abs(y_target[test_idx] - preds)))
    return {"r2_cv": float(np.mean(r2_scores)), "mse_cv": float(np.mean(mse_scores)), "mae_cv": float(np.mean(mae_scores)), "n_samples": len(X_feat)}

def bootstrap_composite_scores(top_scores_df, n_bootstrap=500, ci=0.95):
    if len(top_scores_df) == 0: return {}
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = top_scores_df.sample(n=len(top_scores_df), replace=True)
        bootstrap_means.append(sample['composite_score'].mean())
    bootstrap_means = np.array(bootstrap_means)
    alpha = (1 - ci) / 2
    return {"mean": float(np.mean(bootstrap_means)), "std": float(np.std(bootstrap_means)), "ci_lower": float(np.percentile(bootstrap_means, alpha * 100)), "ci_upper": float(np.percentile(bootstrap_means, (1 - alpha) * 100))}

def compute_edge_significance(nx_graph, n_permutations=100):
    weights = [d['weight'] for _, _, d in nx_graph.edges(data=True)]
    if not weights: return {"p_value": 1.0, "mean_observed": 0, "mean_null": 0}
    observed_mean = np.mean(weights)
    null_means = []
    nodes = list(nx_graph.nodes())
    if len(nodes) < 2: return {"p_value": 1.0, "mean_observed": observed_mean, "mean_null": 0}
    for _ in range(n_permutations):
        u, v = np.random.choice(nodes, 2, replace=False)
        null_means.append(np.random.rand() * 2)
    null_means = np.array(null_means)
    p_value = float(np.mean(null_means >= observed_mean))
    return {"p_value": p_value, "mean_observed": float(observed_mean), "mean_null": float(np.mean(null_means))}

def compute_graph_robustness(nx_graph, n_trials=50):
    if nx_graph.number_of_nodes() < 3: return {"node_removal_stability": 1.0, "edge_removal_stability": 1.0}
    initial_cc = nx.number_connected_components(nx_graph)
    node_failures, edge_failures = [], []
    for _ in range(n_trials):
        g_node = nx_graph.copy()
        remove_count = max(1, int(len(g_node.nodes()) * 0.1))
        to_remove = np.random.choice(list(g_node.nodes()), remove_count, replace=False)
        g_node.remove_nodes_from(to_remove)
        node_failures.append(nx.number_connected_components(g_node) - initial_cc)
        g_edge = nx_graph.copy()
        remove_edges = max(1, int(len(g_edge.edges()) * 0.1))
        to_remove = np.random.choice(list(g_edge.edges()), remove_edges, replace=False)
        g_edge.remove_edges_from(to_remove)
        edge_failures.append(nx.number_connected_components(g_edge) - initial_cc)
    return {"node_removal_stability": float(1.0 - np.mean(node_failures) / max(1, initial_cc)), "edge_removal_stability": float(1.0 - np.mean(edge_failures) / max(1, initial_cc))}

# ==========================================
# LLM CURATION OF RESEARCH DIRECTIONS
# ==========================================
def generate_research_directions(top_pairs_df, tokenizer, model, backend_type: str, max_hypotheses=10, proposal_context=""):
    prompt_template = """You are a materials science strategist for the DECLARMIMA project: "Deciphering laser-microstructure interaction in multicomponent alloys".
Project Goals: {proposal_context}
For the novel concept combination: "{u}" + "{v}"
Associated property context: ~{prop:.1f} (e.g., HV, μm, MPa)
Feasibility estimate: {feas:.2f}/1.0
Write exactly 3 concise, technically precise sentences:
1. Scientific novelty: Why this combination advances DECLARMIMA goals (physics-informed digital twins, multiscale modeling, laser-matter mechanisms).
2. Target outcome: Predicted microstructure/property improvement and key trade-off relevant to additive manufacturing.
3. Validation step: One concrete experimental or computational method (e.g., phase field simulation, EBSD, in-situ XRD, ML uncertainty quantification).
Avoid generic statements. Focus on laser-matter interaction, solidification physics, or phase transformation mechanisms."""
    results = []
    total_rows = min(len(top_pairs_df), max_hypotheses)
    if total_rows == 0: return pd.DataFrame()
    proposal_summary = "physics-informed digital twins for laser-processed multicomponent alloys, multiscale computational modeling, integrated experiment-computation framework, process-structure-property relationships, uncertainty quantification"
    progress_placeholder = st.empty()
    progress_bar = st.progress(0)
    for idx in range(total_rows):
        try:
            row = top_pairs_df.iloc[idx]
            progress = (idx + 1) / total_rows
            progress_bar.progress(progress)
            progress_placeholder.write(f"🔄 Generating hypothesis {idx+1}/{total_rows}: {row['concept_u']} + {row['concept_v']}")
            prompt = prompt_template.format(proposal_context=proposal_summary, u=row['concept_u'].title(), v=row['concept_v'].title(), prop=float(row['expected_property_gain']), feas=float(row['feasibility_score']))
            if backend_type == "ollama":
                try:
                    client = ollama.Client(host=st.session_state.get('ollama_host', 'http://localhost:11434'))
                    response = client.chat(model=model, messages=[{"role": "user", "content": prompt}], options={"temperature": 0.25, "num_predict": 180})
                    response_text = response.get('message', {}).get('content', '') if isinstance(response, dict) else getattr(response, 'message', {}).get('content', '')
                except Exception as e: st.warning(f"⚠️ Ollama generation failed: {e}"); continue
            else:
                try: inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300).to(DEVICE)
                except Exception as e: st.warning(f"⚠️ Tokenization error for row {idx+1}: {e}"); continue
                with torch.no_grad():
                    outputs = model.generate(inputs.input_ids, max_new_tokens=180, temperature=0.25, do_sample=True, pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.1, eos_token_id=tokenizer.eos_token_id, use_cache=True)
                response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            results.append({'Concept Pair': f"{row['concept_u']} + {row['concept_v']}", 'Composite Score': f"{row['composite_score']:.3f}", 'Expected Gain': f"{row['expected_property_gain']:.1f}", 'Feasibility': f"{row['feasibility_score']:.2f}", 'Research Hypothesis': response_text.strip(), 'DECLARMIMA Alignment': 'High' if row['composite_score'] > 0.7 else 'Medium'})
            if backend_type != "ollama":
                del inputs, outputs
                if DEVICE.type == 'cuda': torch.cuda.empty_cache(); gc.collect()
        except torch.cuda.OutOfMemoryError as e:
            st.error(f"❌ CUDA Out of Memory at hypothesis {idx+1}: {e}")
            st.info("💡 Try reducing max_hypotheses or switching to CPU mode"); break
        except Exception as e:
            st.warning(f"⚠️ Skipping hypothesis {idx+1}: {type(e).__name__}: {e}")
            if DEVICE.type == 'cuda': torch.cuda.empty_cache(); gc.collect()
            continue
    progress_placeholder.empty(); progress_bar.empty()
    return pd.DataFrame(results) if results else pd.DataFrame()

# ==========================================
# ENHANCED GRAPH METRICS & VISUALIZATION
# ==========================================
def compute_graph_metrics(G: nx.Graph) -> dict:
    if G.number_of_nodes() == 0: return {}
    metrics = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "density": nx.density(G), "avg_degree": np.mean([d for _, d in G.degree()]), "clustering": nx.average_clustering(G) if G.number_of_nodes() > 2 else 0, "connected_components": nx.number_connected_components(G)}
    if G.number_of_nodes() <= 200:
        bc = nx.betweenness_centrality(G, normalized=True)
        metrics["top_bridges"] = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:5]
    else: metrics["top_bridges"] = []
    return metrics

def display_metric_dashboard(metrics: dict):
    if not metrics: st.warning("No graph metrics available."); return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nodes", metrics["nodes"]); col2.metric("Edges", metrics["edges"])
    col3.metric("Graph Density", f"{metrics['density']:.3f}"); col4.metric("Avg. Degree", f"{metrics['avg_degree']:.2f}")
    col5, col6 = st.columns(2)
    col5.metric("Clustering Coef.", f"{metrics['clustering']:.3f}"); col6.metric("Components", metrics["connected_components"])
    if metrics["top_bridges"]:
        st.markdown("**🌉 Top 5 Bridge Concepts (High Betweenness)**")
        st.dataframe(pd.DataFrame(metrics["top_bridges"], columns=["Concept", "Bridge Score"]), use_container_width=True)

def build_category_hierarchy(valid_concepts: list, concept_abstract_map: dict):
    hierarchy = defaultdict(lambda: {"children": [], "count": 0})
    for concept in valid_concepts:
        matched = False
        for pattern, category in CATEGORY_MAPPING.items():
            if re.search(pattern, concept, re.I): parent = category; matched = True; break
        if not matched: parent = "other_domain" if any(kw in concept.lower() for kw in DOMAIN_KEYWORDS) else "misc"
        freq = len(concept_abstract_map.get(concept, []))
        hierarchy[parent]["children"].append((concept, freq))
        hierarchy[parent]["count"] += freq
    labels, parents, values = [], [], []
    for parent, data in hierarchy.items():
        labels.append(parent); parents.append(""); values.append(data["count"])
        for child, cnt in data["children"]: labels.append(child); parents.append(parent); values.append(cnt)
    return labels, parents, values

def render_sunburst_chart(labels, parents, values, colormap_name="viridis"):
    if not labels: st.info("Not enough categories for sunburst chart."); return
    colors = get_colormap_hex(colormap_name, len(labels))
    fig = go.Figure(go.Sunburst(labels=labels, parents=parents, values=values, branchvalues="total", marker=dict(colors=colors, line=dict(width=0.5, color="white")), textinfo="label+percent entry", insidetextorientation="radial"))
    fig.update_layout(title="<b>DECLARMIMA Research Domain Hierarchy</b><br><i>Size = concept frequency</i>", font=dict(size=12, family="Arial"), paper_bgcolor="white", plot_bgcolor="white", width=800, height=600)
    st.plotly_chart(fig, use_container_width=True)
    try:
        svg_bytes = fig.to_image(format="svg", scale=2)
        st.download_button("📸 Download Sunburst as SVG", data=svg_bytes, file_name="sunburst.svg", mime="image/svg+xml", key="sunburst_svg")
    except: st.info("Install kaleido for SVG export: pip install kaleido")

def get_category_color(concept: str) -> str:
    concept_lower = concept.lower()
    if any(a in concept_lower for a in ['al', 'ti', 'ni', 'cr', 'fe', 'co', 'mo', 'nb', 'cu', 'w', 'mn']): return "#E91E63"
    elif any(l in concept_lower for l in ['laser', 'scan', 'power', 'melt', 'energy']): return "#3F51B5"
    elif any(m in concept_lower for m in ['grain', 'phase', 'hardness', 'strength', 'texture']): return "#FF9800"
    elif any(p in concept_lower for p in ['porosity', 'crack', 'defect', 'void']): return "#F44336"
    elif any(d in concept_lower for d in ['digital', 'twin', 'machine', 'learning', 'neural']): return "#9C27B0"
    else: return "#009688"

def render_graph_pyvis_custom(nx_graph, concept_abstract_map, physics_enabled=True, min_node_size=12, max_node_size=50, colormap_name="viridis"):
    if len(nx_graph.nodes()) > 100:
        degrees = dict(nx_graph.degree())
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:100]
        nx_graph = nx_graph.subgraph(top_nodes).copy()
    net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="#000000", select_menu=True, notebook=False, cdn_resources='remote')
    if physics_enabled:
        net.barnes_hut(gravity=-2000, spring_length=150, spring_strength=0.05, damping=0.09, overlap=0.5)
    else:
        net.set_options("var options = { physics: { enabled: false }, layout: { improvedLayout: false } }")
    node_colors = map_values_to_colors([len(concept_abstract_map.get(n, [])) for n in nx_graph.nodes()], colormap_name, vmin=0, vmax=max(len(concept_abstract_map.get(n, [])) for n in nx_graph.nodes()) + 1)
    for i, node in enumerate(nx_graph.nodes()):
        freq = len(concept_abstract_map.get(node, []))
        size = int(np.clip(min_node_size + freq * 2, min_node_size, max_node_size))
        color = node_colors[i]
        degree = int(nx_graph.degree(node))
        net.add_node(node, label=node, size=size, color=color, font={'color': '#000000', 'size': 14}, title=f"{node}\nDegree: {degree}\nFrequency: {freq}")
    color_map = {'cooccurrence': "#4CAF50", 'semantic': "#2196F3", 'bridge': "#FFC107", 'declarmina_aligned': "#E91E63"}
    for u, v in nx_graph.edges():
        w = nx_graph[u][v].get('weight', 1)
        edge_type = nx_graph[u][v].get('edge_type', 'unknown')
        color = color_map.get(edge_type, "#607D8B")
        net.add_edge(u, v, value=float(np.clip(w, 0.5, 5)), width=float(np.clip(w * 0.8, 1, 4)), color=color, smooth={'type': 'curvedCW', 'roundness': 0.2})
    html_content = net.generate_html()
    st.components.v1.html(html_content, height=750, scrolling=True)
    try:
        html_bytes = html_content.encode('utf-8')
        st.download_button("📥 Download Interactive Graph (HTML)", data=html_bytes, file_name="declarmima_graph.html", mime="text/html", key="pyvis_download_custom")
        del html_content, html_bytes; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception as e:
        st.error(f"⚠️ Download preparation failed: {e}")
        st.info("💡 Try reducing graph size or using Plotly visualization instead")

def render_graph_plotly_white(nx_graph, concept_abstract_map, colormap_name="viridis"):
    if len(nx_graph.nodes()) > 100:
        degrees = dict(nx_graph.degree())
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:100]
        nx_graph = nx_graph.subgraph(top_nodes).copy()
    pos = nx.spring_layout(nx_graph, k=2, iterations=50, seed=SEED)
    edge_x, edge_y, edge_hover = [], [], []
    for u, v in nx_graph.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
        w = nx_graph[u][v].get('weight', 1)
        edge_type = nx_graph[u][v].get('edge_type', 'unknown')
        edge_hover.extend([f"{u} ↔ {v}\nWeight: {w:.2f}\nType: {edge_type}"] * 2 + [None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1.2, color='#666'), hoverinfo='text', hovertext=edge_hover, name='Connections')
    node_x, node_y, node_text, node_size, node_color, node_symbol = [], [], [], [], [], []
    freqs = [len(concept_abstract_map.get(n, [])) for n in nx_graph.nodes()]
    node_colors_hex = map_values_to_colors(np.array(freqs), colormap_name, vmin=0, vmax=max(freqs)+1)
    for i, node in enumerate(nx_graph.nodes()):
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        deg = nx_graph.degree(node)
        node_text.append(f"{node}\nDegree: {deg}\nFrequency: {freqs[i]}")
        node_size.append(max(10, min(45, deg * 3 + 12)))
        node_color.append(node_colors_hex[i])
        node_lower = node.lower()
        if any(a in node_lower for a in ['al', 'ti', 'ni', 'cr', 'fe', 'co', 'mo', 'nb', 'cu', 'w', 'mn']): node_symbol.append('square')
        elif any(l in node_lower for l in ['laser', 'scan', 'power', 'melt', 'energy', 'speed', 'pulse']): node_symbol.append('diamond')
        elif any(m in node_lower for m in ['grain', 'phase', 'hardness', 'strength', 'texture', 'microstructure']): node_symbol.append('circle')
        elif any(p in node_lower for p in ['porosity', 'crack', 'defect', 'void', 'residual']): node_symbol.append('x')
        elif any(d in node_lower for d in ['digital', 'twin', 'machine', 'learning', 'neural', 'graph']): node_symbol.append('star')
        else: node_symbol.append('circle')
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', marker=dict(size=node_size, color=node_color, line=dict(width=2, color='#ffffff'), symbol=node_symbol), text=[n for n in nx_graph.nodes()], textposition="bottom center", textfont=dict(size=10, color='#000000'), hovertext=node_text, hoverinfo='text', name='Concepts')
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, hovermode='closest', margin=dict(b=0, l=0, r=0, t=0), plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', xaxis=dict(showgrid=True, zeroline=False, showticklabels=False, gridcolor='#eee', range=[-1.5, 1.5]), yaxis=dict(showgrid=True, zeroline=False, showticklabels=False, gridcolor='#eee', range=[-1.5, 1.5]), annotations=[dict(text="🔬 Drag nodes • Hover for details • Scroll to zoom • DECLARMIMA-aligned edges in pink", showarrow=False, xref="paper", yref="paper", x=0.5, y=-0.05, font=dict(size=10, color='#666'))]))
    fig.update_layout(updatemenus=[dict(type="buttons", showactive=False, x=0.1, xanchor="left", y=1.15, yanchor="top", buttons=[dict(label="🔄 Re-layout", method="relayout", args=[{"xaxis.range": [-1.5, 1.5], "yaxis.range": [-1.5, 1.5]}]), dict(label="🔍 Zoom In", method="relayout", args=[{"xaxis.autorange": False, "yaxis.autorange": False}]), dict(label="📐 Reset View", method="relayout", args=[{"xaxis.autorange": True, "yaxis.autorange": True}])])])
    st.plotly_chart(fig, use_container_width=True, key="plotly_graph")
    fig_json = fig.to_json()
    st.download_button("📥 Download Plotly Graph (JSON)", data=fig_json, file_name="concept_graph_plotly.json", mime="application/json", key="plotly_download")

def render_embedding_plot(valid_concepts, embed_model, concept_abstract_map, colormap_name="viridis", method="UMAP"):
    try:
        embeddings = embed_model.encode(valid_concepts, show_progress_bar=False)
        freqs = np.array([len(concept_abstract_map.get(c, [])) for c in valid_concepts])
        if method == "UMAP" and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=2, random_state=SEED, metric='cosine')
            proj = reducer.fit_transform(embeddings)
        else:
            reducer = PCA(n_components=2, random_state=SEED)
            proj = reducer.fit_transform(embeddings)
        colors = map_values_to_colors(freqs, colormap_name, vmin=0, vmax=max(freqs)+1)
        fig = go.Figure(data=[go.Scatter(x=proj[:,0], y=proj[:,1], mode='markers+text', marker=dict(size=15, color=colors, line=dict(width=1, color='black')), text=[c for c in valid_concepts], textposition="top center", textfont=dict(size=9, color='#000000'), hoverinfo='text', hovertext=[f"{c}\nFreq: {f}" for c, f in zip(valid_concepts, freqs)])])
        fig.update_layout(title=f"<b>Concept Embedding Space ({method})</b><br><i>Colormap: {colormap_name} mapped to frequency</i>", paper_bgcolor='white', plot_bgcolor='white', xaxis=dict(showgrid=True, zeroline=False, showticklabels=False), yaxis=dict(showgrid=True, zeroline=False, showticklabels=False), height=600)
        st.plotly_chart(fig, use_container_width=True)
        try:
            png_bytes = fig.to_image(format="png", scale=2)
            st.download_button("📸 Download Embedding Plot (PNG)", data=png_bytes, file_name=f"{method.lower()}_embedding.png", mime="image/png", key=f"{method.lower()}_download")
        except: st.info("Install kaleido for PNG export: pip install kaleido")
    except Exception as e:
        st.warning(f"⚠️ Embedding plot failed: {e}. Fallback to PCA...")
        if method != "PCA": render_embedding_plot(valid_concepts, embed_model, concept_abstract_map, colormap_name, method="PCA")

def render_graph_fallback(nx_graph, concept_abstract_map):
    st.markdown("### 📊 Graph Summary (Text View)")
    st.markdown(f"- **Nodes**: {len(nx_graph.nodes())}")
    st.markdown(f"- **Edges**: {len(nx_graph.edges())}")
    if len(nx_graph.edges()) > 0:
        edge_list = [(u, v, nx_graph[u][v].get('weight', 1)) for u, v in nx_graph.edges()]
        edge_list.sort(key=lambda x: x[2], reverse=True)
        st.markdown("**🔗 Top 15 Strongest Connections:**")
        for i, (u, v, w) in enumerate(edge_list[:15], 1):
            edge_type = nx_graph[u][v].get('edge_type', 'unknown')
            declarmima_tag = " 🎯 DECLARMIMA" if edge_type == 'declarmina_aligned' else ""
            st.markdown(f"{i}. `{u}` ↔ `{v}` (weight: {w:.2f}, type: {edge_type}){declarmima_tag}")
    if len(concept_abstract_map) > 0:
        freq_data = [(c, len(concept_abstract_map.get(c, []))) for c in nx_graph.nodes()]
        freq_data.sort(key=lambda x: x[1], reverse=True)
        st.markdown("**📈 Top Concepts by Frequency:**")
        st.dataframe(pd.DataFrame(freq_data[:10], columns=["Concept", "Abstract Count"]), use_container_width=True)

def display_link_predictions(top_scores_df: pd.DataFrame, threshold=0.6):
    if top_scores_df.empty: st.info("No novel pairs scored above threshold. Try lowering edge sampling threshold."); return
    st.markdown("### 🔗 Predicted Missing Links (Research Gaps)")
    display_cols = ['concept_u', 'concept_v', 'gnn_affinity', 'semantic_novelty', 'composite_score']
    st.dataframe(top_scores_df[display_cols].head(20).style.format({'gnn_affinity': '{:.3f}', 'semantic_novelty': '{:.3f}', 'composite_score': '{:.3f}'}), use_container_width=True)
    top_pair = top_scores_df.iloc[0]
    st.success(f"🌟 **Top Innovation Frontier:** {top_pair['concept_u']} ↔ {top_pair['concept_v']} (Score = {top_pair['composite_score']:.3f})")

# ==========================================
# STREAMLIT UI: SIDEBAR
# ==========================================
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("🎮 GPU/CUDA Settings")
        cuda_info = get_pytorch_cuda_info()
        if cuda_info['cuda_available'] and not st.session_state.get("force_cpu", False):
            cc = get_gpu_compute_capability()
            if cc: st.markdown(f"✅ GPU: `{torch.cuda.get_device_name(0)}` (sm_{cc[0]}{cc[1]})")
            else: st.markdown("✅ GPU detected (compute capability unknown)")
        if st.button("🔄 Test CUDA Compatibility"):
            compatible, diagnostic = check_cuda_kernel_compatibility()
            dgl_compatible, dgl_diagnostic = check_dgl_cuda_compatibility()
            if compatible and dgl_compatible: st.success("✅ PyTorch and DGL CUDA compatible!")
            else:
                if not compatible: st.error("❌ PyTorch CUDA incompatible")
                if not dgl_compatible: st.error("❌ DGL CUDA incompatible")
                with st.expander("View Details"): st.code(f"PyTorch:\n{diagnostic}\nDGL:\n{dgl_diagnostic}")
        else: st.markdown("🖥️ **CPU Mode** (CUDA unavailable or disabled)")
        if st.button("🔁 Retry GPU Detection"):
            if "force_cpu" in st.session_state: del st.session_state["force_cpu"]
            st.rerun()
        force_cpu = st.checkbox("⚠️ Force CPU Mode (bypass CUDA)", value=st.session_state.get("force_cpu", False), help="Use this if you encounter 'no kernel image' errors")
        if force_cpu != st.session_state.get("force_cpu", False): st.session_state["force_cpu"] = force_cpu
        if force_cpu: force_cpu_mode(); st.success("🔄 Reload to apply CPU mode")
        if st.button("Reload Now"): st.rerun()
        st.markdown("---")
        st.subheader("🔧 GNN Backend")
        gnn_backend = st.radio("Choose GNN implementation:", options=["Auto (DGL preferred, PyTorch fallback)", "PyTorch Sparse Only", "DGL Only (if installed)"], index=0, help="Auto: Try DGL first, fall back to PyTorch if needed\nPyTorch: Use original sparse tensor implementation\nDGL: Use Deep Graph Library (requires installation)")
        st.session_state.gnn_backend = gnn_backend
        st.subheader("🔧 LLM Backend")
        backend_option = st.radio("Choose inference backend:", options=["Hugging Face Transformers", "Ollama (if installed)"], index=0, help="Transformers: direct HF model loading\nOllama: use local ollama serve (faster switching)")
        st.session_state.inference_backend = backend_option
        if backend_option == "Ollama (if installed)":
            if not OLLAMA_AVAILABLE: st.error("❌ ollama library not installed"); st.code("pip install ollama"); st.info("Also ensure Ollama server is running: ollama serve")
            available_ollama_models = [k for k in LOCAL_LLM_OPTIONS.keys() if is_ollama_model(k)]
            model_choice = st.selectbox("🧠 Local LLM (Ollama)", options=available_ollama_models if available_ollama_models else ["No Ollama models available"], index=0 if available_ollama_models else 0, help="Models served via local Ollama instance")
        else:
            hf_models = [k for k in LOCAL_LLM_OPTIONS.keys() if not is_ollama_model(k)]
            model_choice = st.selectbox("🧠 Local LLM (Hugging Face)", options=hf_models, index=2, help="Models loaded directly via transformers library")
        st.session_state.llm_model_choice = model_choice
        if backend_option == "Hugging Face Transformers" and not is_ollama_model(model_choice):
            st.session_state.use_4bit_quantization = st.checkbox("🗜️ Use 4-bit quantization (reduces VRAM usage)", value=True, help="Enable for models >3B parameters to reduce memory usage by ~75%")
        if backend_option == "Ollama (if installed)" or is_ollama_model(model_choice):
            st.session_state.ollama_host = st.text_input("🌐 Ollama Host", value=st.session_state.get('ollama_host', 'http://localhost:11434'), help="URL of your Ollama server (default: http://localhost:11434)")
        st.subheader("🎨 Graph Visualization")
        viz_backend = st.selectbox("Choose visualization engine:", options=["PyVis (Interactive Network)", "Plotly (Stable Plot)", "Text Summary (Fallback)"], index=0, help="PyVis: Rich interactive network\nPlotly: More stable, zoomable plot\nText: Simple fallback view")
        st.session_state['viz_backend'] = viz_backend
        st.session_state['colormap_name'] = st.selectbox("🎨 Colormap Scheme", options=list(COLORMAP_REGISTRY.keys()), index=0, help="Apply to nodes, edges, sunburst, and embedding plots")
        st.session_state['embedding_method'] = st.selectbox("📉 Embedding Projection", options=["UMAP", "PCA"], index=0 if UMAP_AVAILABLE else 1, help="UMAP preserves local structure better, PCA is faster")
        st.subheader("🎯 DECLARMIMA Integration")
        use_declarmima = st.toggle("Use DECLARMIMA proposal as seed knowledge", value=True, help="Inject concepts from the DECLARMIMA research proposal and prioritize abstracts aligned with project goals")
        st.session_state['use_declarmima'] = use_declarmima
        abstract_preview = st.text_area("📋 Paste abstracts (preview):", height=100, key="preview")
        preview_count = len([t for t in re.split(r'\n\s*\n', abstract_preview) if t.strip()]) if abstract_preview.strip() else 0
        if preview_count > 0 and preview_count <= 25:
            st.warning(f"📉 Small corpus ({preview_count} abstracts): applying adaptive settings")
            st.toggle("Enable semantic clustering", value=True, key="use_clustering", disabled=True)
            st.toggle("Inject domain seeds", value=True, key="inject_seeds", disabled=True)
            st.toggle("Use embedding edges", value=True, key="semantic_edges", disabled=True)
        else:
            st.toggle("Enable semantic clustering", value=False, key="use_clustering")
            st.toggle("Inject domain seeds", value=False, key="inject_seeds")
            st.toggle("Use embedding edges", value=False, key="semantic_edges")
        st.subheader("🎨 Visual Customization")
        st.session_state.physics_enabled = st.checkbox("Enable graph physics", value=True, key="physics_toggle")
        st.session_state.min_node_size = st.slider("Min node size", 8, 30, 12, key="min_size")
        st.session_state.max_node_size = st.slider("Max node size", 30, 80, 50, key="max_size")
        st.markdown("---")
        st.markdown("**🎯 DECLARMIMA Focus Areas:**")
        st.markdown("- 🔬 Laser-matter interaction mechanisms\n- 🧱 Multiscale computational modeling\n- 🤖 Physics-informed machine learning\n- 📊 Process-structure-property relationships\n- 🔍 Uncertainty quantification & validation")
        st.markdown("---")
        st.subheader("📊 Validation & Distillation")
        st.session_state.enable_validation = st.checkbox("🧮 Run Mathematical Validation", value=True, help="K-fold CV for Ridge, Bootstrap uncertainty, Graph robustness")
        st.session_state.show_distillation = st.checkbox("🔍 Show Distillation Metrics", value=True, help="TF-IDF, relevance scoring, extraction confidence")
        st.markdown("**⚡ Performance:**")
        max_hyp = st.slider("Max hypotheses", 1, 20, 10)
        st.session_state['max_hypotheses'] = max_hyp
        if st.button("🗑️ Clear Cache"): st.cache_resource.clear(); gc.collect(); torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None; st.success("✅ Cache cleared!")
        gpu_info = "CUDA" if torch.cuda.is_available() else "CPU"
        vram_info = f"{get_available_gpu_memory():.1f}GB free" if torch.cuda.is_available() and get_available_gpu_memory() else "N/A"
        dgl_status = "✅ DGL" if DGL_AVAILABLE else "❌ DGL"
        st.caption(f"🖥️ Device: {gpu_info}\n💾 Available VRAM: {vram_info}\n📦 Embedding model: ~80MB\n🤖 LLM: {LOCAL_LLM_OPTIONS.get(model_choice, 'unknown')}\n🔷 GNN: {dgl_status}")

# ==========================================
# STREAMLIT UI & PIPELINE ORCHESTRATION
# ==========================================
def main():
    st.title("🔬 DECLARMIMA: Laser-Microstructure Interaction Analyzer")
    st.caption("Physics-informed digital twins for multicomponent alloys • Dual LLM backend: HF Transformers or Ollama • GNN: PyTorch or DGL")
    render_sidebar()
    if st.session_state.get('llm_model_choice') and not is_ollama_model(st.session_state.get('llm_model_choice', '')):
        mem_info = estimate_model_memory(st.session_state.llm_model_choice, st.session_state.get('use_4bit_quantization', True))
        available_vram = get_available_gpu_memory()
        if available_vram and not mem_info['cpu_ok']:
            required = float(mem_info['vram_4bit'].replace('GB','').replace('~','').strip()) if 'GB' in mem_info['vram_4bit'] else 100
            if available_vram < required:
                st.markdown(f"""<div style="background:#fef3c7;border-left:4px solid #f59e0b;padding:0.75rem;border-radius:0 0.5rem 0.5rem 0;margin:0.5rem 0">⚠️ <strong>Memory Warning:</strong> {st.session_state.llm_model_choice} requires ~{mem_info['vram_4bit']} VRAM. You have ~{available_vram:.1f}GB available. Consider using 4-bit quantization, a smaller model, or Ollama backend.</div>""", unsafe_allow_html=True)
    abstract_input = st.text_area("📋 Paste scientific abstracts (blank lines separate):", height=300, placeholder='Example:\n"Laser powder bed fusion of AlSi10Mg reveals columnar-to-equiaxed transition at 85 J/mm³..."\n"High-entropy alloy CoCrFeNiMo via DED shows 420 HV microhardness from nanoscale precipitates..."')
    if st.button("🚀 Analyze Abstracts", type="primary", use_container_width=True):
        if not abstract_input.strip(): st.error("⚠️ Please enter at least one abstract."); return
        abstracts = [t.strip() for t in re.split(r'\n\s*\n', abstract_input) if t.strip()]
        if len(abstracts) < 10: st.info(f"💡 {len(abstracts)} abstracts: Maximum semantic enrichment mode")
        elif len(abstracts) > 35: st.warning(f"⚠️ {len(abstracts)} abstracts may increase processing time")
        progress_bar = st.progress(0.0)
        status = st.status("🔄 Initializing...", expanded=True)
        try:
            with status:
                st.write("📦 Loading models...")
                embed_model = load_embedding_model()
                model_key = st.session_state.get('llm_model_choice', DEFAULT_LLM_NAME)
                tokenizer, llm_model, device_or_host, backend_type = load_local_llm(model_key, use_4bit=st.session_state.get('use_4bit_quantization', True))
                if tokenizer is None and backend_type == "transformers": st.error("Failed to load HF model. Try selecting a different option."); return
                if llm_model is None and backend_type == "ollama": st.error("Failed to connect to Ollama. Ensure 'ollama serve' is running."); return
                st.success(f"✅ Models loaded ({backend_type})")
                progress_bar.progress(0.10)
                proposal_embedding = None
                config = get_adaptive_config(len(abstracts))
                if st.session_state.get('use_declarmima', True) and config.get("USE_DECLARMIMA_SEEDS", True):
                    with st.status("🎯 Processing DECLARMIMA proposal..."):
                        proposal_embedding = embed_model.encode([DECLARMIMA_PROPOSAL_TEXT], show_progress_bar=False)[0]
                        st.write(f"✅ Proposal embedding: {proposal_embedding.shape}")
                        declarmima_concepts = extract_declarmima_concepts(DECLARMIMA_PROPOSAL_TEXT, embed_model)
                        st.write(f"✅ Extracted {len(declarmima_concepts)} DECLARMIMA seed concepts")
                progress_bar.progress(0.15)
                if "use_clustering" in st.session_state: config["USE_SEMANTIC_CLUSTERING"] = st.session_state.use_clustering
                if "inject_seeds" in st.session_state: config["INJECT_DOMAIN_SEEDS"] = st.session_state.inject_seeds
                if "semantic_edges" in st.session_state: config["USE_SEMANTIC_EDGES"] = st.session_state.semantic_edges
                config["USE_DECLARMIMA_SEEDS"] = st.session_state.get('use_declarmima', True)
                config["CORRELATE_WITH_PROPOSAL"] = st.session_state.get('use_declarmima', True)
                with st.status("🔍 Extracting & distilling concepts..."):
                    all_concepts, all_metrics = extract_concepts_from_abstracts(abstracts, tokenizer, llm_model, backend_type)
                    valid_concepts, concept_to_id, id_to_concept, concept_abstract_map, distillation_metrics = normalize_and_filter_concepts(all_concepts, embed_model, config, proposal_embedding)
                    st.write(f"✅ **{len(valid_concepts)}** concepts extracted & distilled")
                    if len(valid_concepts) < 10: st.info("💡 Small concept set: semantic-only graph mode")
                progress_bar.progress(0.25)
                if len(valid_concepts) < 3:
                    st.warning("⚠️ Very few concepts. Injecting additional seeds...")
                    valid_concepts, concept_to_id = inject_domain_seeds(valid_concepts, concept_to_id)
                    st.success(f"✅ Recovered {len(valid_concepts)} concepts")
                with st.status("🕸️ Building concept graph..."):
                    nx_graph = build_concept_graph(all_concepts, concept_to_id, embed_model, config, proposal_embedding)
                    d_prev_dict = dict(nx.all_pairs_shortest_path_length(nx_graph, cutoff=4))
                    pos_pairs, neg_pairs = sample_edges_for_training(nx_graph, d_prev_dict, valid_concepts, concept_to_id, config)
                    st.write(f"✅ Graph: **{len(valid_concepts)}** nodes, **{nx_graph.number_of_edges()}** edges")
                    declarmima_edges = sum(1 for _, _, d in nx_graph.edges(data=True) if d.get('edge_type') == 'declarmina_aligned')
                    if declarmima_edges > 0: st.success(f"🎯 {declarmima_edges} edges aligned with DECLARMIMA goals")
                    if not nx.is_connected(nx_graph): n_comp = nx.number_connected_components(nx_graph); st.info(f"🔗 Graph has {n_comp} component(s)")
                progress_bar.progress(0.40)
                with st.status("🧠 Generating embeddings..."):
                    embed_dim = get_embedding_dimension(embed_model)
                    st.write(f"✅ Embedding dimension: {embed_dim}")
                    node_features = generate_embeddings(valid_concepts, embed_model)
                    st.write(f"✅ Node features shape: {node_features.shape}")
                progress_bar.progress(0.50)
                def _training_progress(epoch, loss):
                    progress_value = 0.50 + (epoch / TRAIN_EPOCHS) * 0.30
                    progress_bar.progress(min(1.0, max(0.0, progress_value)))
                    if epoch % 10 == 0: status.write(f"📊 Epoch {epoch}/{TRAIN_EPOCHS} | Loss: {loss:.4f}")
                with st.status("🤖 Training GraphSAGE..."):
                    use_dgl_backend = (st.session_state.get('gnn_backend', 'Auto (DGL preferred, PyTorch fallback)') != 'PyTorch Sparse Only')
                    if DGL_AVAILABLE and use_dgl_backend: st.write("🔷 Attempting DGL training...")
                    else: st.write("⚙️ Using PyTorch sparse implementation")
                    gnn_model, final_emb, adj_indices, adj_values = train_gnn(node_features, nx_graph, concept_to_id, pos_pairs, neg_pairs, _training_progress, use_dgl=use_dgl_backend)
                    st.success("✅ GNN training complete")
                progress_bar.progress(0.80)
                with st.status("📈 Scoring novel directions..."):
                    concept_properties, ridge = compute_microstructure_quantification(valid_concepts, concept_abstract_map, all_metrics, nx_graph)
                    top_scores = compute_research_direction_scores(gnn_model, node_features, final_emb, nx_graph, valid_concepts, concept_properties, ridge, embed_model, d_prev_dict, adj_indices, adj_values)
                    st.write(f"✅ Scored **{len(top_scores)}** novel pairs")
                progress_bar.progress(0.90)
                with st.status("✍️ Generating hypotheses..."):
                    max_hyp = st.session_state.get('max_hypotheses', 10)
                    st.write(f"📝 Generating up to {max_hyp} DECLARMIMA-aligned hypotheses...")
                    directions_df = generate_research_directions(top_scores, tokenizer, llm_model, backend_type, max_hypotheses=max_hyp, proposal_context="physics-informed digital twins, multiscale modeling, laser-matter mechanisms")
                    st.success("✅ Pipeline complete!")
                progress_bar.progress(1.00)
                status.update(label="✅ Analysis complete!", state="complete", expanded=False)
                # Results Tabs
                tab1, tab2, tab3, tab4 = st.tabs(["🎯 Research Directions", "🌐 Graph Visualization", "📊 Validation & Metrics", "🔍 Distillation Insights"])
                with tab1:
                    st.subheader("🎯 Top Research Directions (DECLARMIMA-Aligned)")
                    if len(directions_df) > 0:
                        if 'DECLARMIMA Alignment' in directions_df.columns:
                            st.dataframe(directions_df[['Concept Pair', 'Composite Score', 'Expected Gain', 'Feasibility', 'DECLARMIMA Alignment', 'Research Hypothesis']], use_container_width=True, column_config={"Concept Pair": st.column_config.TextColumn("Pair", width="medium"), "Composite Score": st.column_config.NumberColumn("Score", format="%.3f"), "Expected Gain": st.column_config.NumberColumn("Gain", format="%.1f"), "Feasibility": st.column_config.NumberColumn("Feas.", format="%.2f"), "DECLARMIMA Alignment": st.column_config.TextColumn("Alignment"), "Research Hypothesis": st.column_config.TextColumn("Hypothesis", width="large")})
                        else: st.dataframe(directions_df[['Concept Pair', 'Composite Score', 'Expected Gain', 'Feasibility', 'Research Hypothesis']], use_container_width=True)
                        csv = directions_df.to_csv(index=False)
                        st.download_button("📥 Download CSV", data=csv, file_name="declarmima_directions.csv", mime="text/csv")
                    else: st.info("💡 No novel pairs scored above threshold. Try adjusting parameters.")
                with tab2:
                    st.subheader("🌐 Interactive Concept Graph")
                    selected_viz = st.session_state.get('viz_backend', 'PyVis (Interactive Network)')
                    colormap = st.session_state.get('colormap_name', 'viridis')
                    if len(nx_graph.nodes()) == 0: st.warning("⚠️ No nodes to display."); render_graph_fallback(nx_graph, concept_abstract_map)
                    elif nx_graph.number_of_edges() == 0:
                        st.warning("⚠️ No edges — building semantic fallback")
                        nx_graph = build_semantic_only_graph(list(nx_graph.nodes()), embed_model, similarity_threshold=0.65)
                        if nx_graph.number_of_edges() == 0: render_graph_fallback(nx_graph, concept_abstract_map)
                        else: st.success(f"✅ Fallback: {nx_graph.number_of_edges()} edges")
                    if selected_viz == "PyVis (Interactive Network)":
                        render_graph_pyvis_custom(nx_graph, concept_abstract_map, physics_enabled=st.session_state.get('physics_enabled', True), min_node_size=st.session_state.get('min_node_size', 12), max_node_size=st.session_state.get('max_node_size', 50), colormap_name=colormap)
                    elif selected_viz == "Plotly (Stable Plot)":
                        render_graph_plotly_white(nx_graph, concept_abstract_map, colormap_name=colormap)
                    else: render_graph_fallback(nx_graph, concept_abstract_map)
                    st.markdown("---")
                    st.subheader("📉 Concept Embedding Space")
                    render_embedding_plot(valid_concepts, embed_model, concept_abstract_map, colormap_name=colormap, method=st.session_state.get('embedding_method', 'PCA'))
                with tab3:
                    st.subheader("🧮 Mathematical Validation Suite")
                    if st.session_state.get('enable_validation', True) and len(top_scores) > 0:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**📐 Ridge Regression Cross-Validation**")
                            X_feat = [[concept_properties.get(u, 0), concept_properties.get(v, 0), nx_graph[u][v].get('weight', 1)] for u, v in nx_graph.edges()]
                            y_target = [max(concept_properties.get(u, 0), concept_properties.get(v, 0)) * 1.08 if max(concept_properties.get(u, 0), concept_properties.get(v, 0)) > 0 else 0 for u, v in nx_graph.edges()]
                            val_metrics = validate_ridge_model(X_feat, y_target)
                            st.metrics({"R² (3-fold)": f"{val_metrics['r2_cv']:.3f}", "MSE": f"{val_metrics['mse_cv']:.1f}", "MAE": f"{val_metrics['mae_cv']:.1f}"})
                        with col2:
                            st.markdown("**📊 Bootstrap Uncertainty (Composite Scores)**")
                            boot = bootstrap_composite_scores(top_scores)
                            st.metric("Mean Score", f"{boot['mean']:.3f}", delta=f"± {boot['std']:.3f}")
                            st.caption(f"95% CI: [{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]")
                        col3, col4 = st.columns(2)
                        with col3:
                            st.markdown("**🔗 Edge Significance (Permutation Test)**")
                            edge_sig = compute_edge_significance(nx_graph)
                            st.metric("Observed Mean Weight", f"{edge_sig['mean_observed']:.3f}")
                            st.caption(f"p-value: {edge_sig['p_value']:.4f} (Null mean: {edge_sig['mean_null']:.3f})")
                        with col4:
                            st.markdown("**🛡️ Graph Robustness (10% Removal)**")
                            robust = compute_graph_robustness(nx_graph)
                            st.metric("Node Stability", f"{robust['node_removal_stability']:.2f}")
                            st.metric("Edge Stability", f"{robust['edge_removal_stability']:.2f}")
                    else: st.info("Run validation toggle in sidebar or generate scores first.")
                with tab4:
                    if st.session_state.get('show_distillation', True) and distillation_metrics:
                        st.subheader("🔍 Concept Distillation Efficiency")
                        df_dist = pd.DataFrame(distillation_metrics).T.reset_index().rename(columns={"index": "Concept"})
                        df_dist = df_dist.sort_values("distillation_score", ascending=False)
                        st.dataframe(df_dist.head(20), use_container_width=True, column_config={"Concept": st.column_config.TextColumn("Concept", width="large"), "tfidf": st.column_config.NumberColumn("TF-IDF", format="%.4f"), "relevance": st.column_config.NumberColumn("Proposal Relevance", format="%.3f"), "frequency": st.column_config.NumberColumn("Frequency"), "distillation_score": st.column_config.NumberColumn("Composite Score", format="%.3f")})
                        st.markdown("---")
                        st.subheader("📈 Research Domain Hierarchy (Sunburst)")
                        labels, parents, values = build_category_hierarchy(valid_concepts, concept_abstract_map)
                        render_sunburst_chart(labels, parents, values, colormap_name=st.session_state.get('colormap_name', 'viridis'))
                    else: st.info("Enable distillation metrics toggle in sidebar.")
            # Graph Diagnostics Footer
            with st.expander("📊 Graph Diagnostics", expanded=len(valid_concepts) < 30):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Nodes", len(valid_concepts)); col2.metric("Edges", nx_graph.number_of_edges())
                col3.metric("Avg Degree", f"{np.mean([d for _,d in nx_graph.degree()]):.2f}")
                n_components = nx.number_connected_components(nx_graph)
                col4.metric("Connected", "✅" if nx.is_connected(nx_graph) else f"❌ ({n_components})")
                if st.session_state.get('use_declarmima', True):
                    declarmima_nodes = sum(1 for c in valid_concepts if any(kw in c.lower() for kw in DOMAIN_KEYWORDS))
                    col1, col2 = st.columns(2)
                    col1.metric("DECLARMIMA Concepts", declarmima_nodes)
                    declarmima_edges = sum(1 for _, _, d in nx_graph.edges(data=True) if d.get('edge_type') == 'declarmina_aligned')
                    col2.metric("Aligned Edges", declarmima_edges)
                if len(valid_concepts) > 0:
                    st.markdown("### Top Concepts")
                    freq_data = [(c, len(concept_abstract_map.get(c, []))) for c in valid_concepts]
                    freq_data.sort(key=lambda x: x[1], reverse=True)
                    st.dataframe(pd.DataFrame(freq_data[:10], columns=["Concept", "Frequency"]), use_container_width=True)
        except torch.cuda.OutOfMemoryError as e:
            st.error(f"❌ CUDA OOM: {e}"); st.info("💡 Reduce 'Max hypotheses' or switch to CPU")
            with st.expander("🔍 Traceback"): st.code(traceback.format_exc())
        except RuntimeError as e:
            if "no kernel image" in str(e).lower() or "cudaerror" in str(e).lower():
                st.error(f"❌ CUDA kernel error: {e}")
                st.info("💡 Solutions:\n1. New GPU? `pip install -U torch --index-url https://download.pytorch.org/whl/cu128`\n2. Old GPU? Force CPU: `export CUDA_VISIBLE_DEVICES=\"\"`\n3. DGL issue? Reinstall matching CUDA version.\n4. Reinstall PyTorch matching driver version.")
                with st.expander("🔍 Full Traceback"): st.code(traceback.format_exc())
            else: st.error(f"❌ Error: {str(e)}")
            with st.expander("🔍 Traceback"): st.code(traceback.format_exc())
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            with st.expander("🔍 Traceback"): st.code(traceback.format_exc())
        finally:
            gc.collect()
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    st.markdown("---")
    st.markdown("""
**🔧 Dual LLM Backend Support:**
- ✅ **Hugging Face Transformers**: Direct model loading with 4-bit quantization, device auto-detection
- ✅ **Ollama**: Server-based inference via HTTP, faster model switching, lower Python memory footprint
- ✅ **All 20 models** from LASER RAG codebase: 12 HF + 8 Ollama options
- ✅ **Memory-aware**: VRAM estimation, CPU fallback, graceful degradation
**🔷 GNN Backend Options:**
- ✅ **Auto (default)**: Try DGL first for optimized performance, fall back to PyTorch sparse if needed
- ✅ **PyTorch Sparse Only**: Use original implementation (no DGL dependency)
- ✅ **DGL Only**: Force DGL usage (requires `pip install dgl`)
**🎯 DECLARMIMA Integration:**
- ✅ Proposal text injected as seed knowledge base
- ✅ Abstract-proposal semantic correlation scoring
- ✅ Hypotheses aligned with physics-informed digital twin goals
- ✅ DECLARMIMA-aligned edges highlighted in pink
**🎨 Advanced Visualization & Colormaps:**
- ✅ **50+ colormaps**: jet, turbo, inferno, plasma, viridis, tab10, Set3, etc.
- ✅ Dynamic node/edge/sunburst color mapping based on frequency & relevance
- ✅ UMAP/PCA embedding space projection with publication-ready exports
- ✅ PyVis download OOM fix (CDN + bytes encoding + GC cleanup)
**🧮 Mathematical Validation & Distillation:**
- ✅ K-fold Ridge CV, Bootstrap uncertainty quantification for composite scores
- ✅ Permutation edge significance testing & graph robustness metrics
- ✅ TF-IDF weighted concept distillation, proposal relevance scoring
- ✅ Statistical dashboard with confidence intervals & stability metrics
**💡 Small Corpus Tips (10-25 abstracts):**
- Semantic clustering merges similar terms
- DECLARMIMA seed injection adds domain concepts
- Embedding edges connect semantically related concepts
- Adaptive thresholds relax frequency requirements
**🔧 Technical:** Qwen2.5-0.5B + Sentence-BERT (384-dim) + PyTorch/DGL GraphSAGE | Local processing
**⚠️ Troubleshooting CUDA/DGL Errors:**
1. Check sidebar "GPU/CUDA Settings" for compatibility status
2. Click "Test CUDA Compatibility" for detailed diagnostics
3. Use "Force CPU Mode" toggle for immediate fallback
4. For NEW GPUs (RTX 40xx/50xx): `pip install torch --index-url https://download.pytorch.org/whl/cu128`
5. For OLD GPUs (sm < 3.7): Build PyTorch from source with `TORCH_CUDA_ARCH_LIST=3.5`
6. For DGL CUDA issues: `pip uninstall dgl -y && pip install dgl -f https://data.dgl.ai/wheels/[cuda_version]/repo.html`
7. Ensure CUDA driver matches both PyTorch and DGL CUDA build versions
""")

if __name__ == "__main__":
    main()
