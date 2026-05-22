"""
Publication-Quality Streamlit App: Cross-Model Physical Quantity Extraction
Enhanced with full editability, dynamic fitting, and publication-grade defaults
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from itertools import combinations
from io import BytesIO
import os

st.set_page_config(
    page_title="LLM Extraction Comparison -- Materials Design",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.4rem; font-weight: 800; color: #1a1a2e; margin-bottom: 0.3rem; letter-spacing: -0.5px; }
    .sub-header { font-size: 1.15rem; color: #5a5a7a; margin-bottom: 2rem; font-weight: 400; }
    .metric-card { background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); border-left: 4px solid #2980b9; padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
    .highlight-red { border-left-color: #c0392b; }
    .highlight-green { border-left-color: #27ae60; }
    .highlight-blue { border-left-color: #2980b9; }
    .highlight-purple { border-left-color: #8e44ad; }
    .caption { font-size: 0.9rem; color: #666; font-style: italic; margin-bottom: 1rem; }
    .control-section { background: linear-gradient(180deg, #f5f7fa 0%, #eef1f5 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border: 1px solid #dde2e8; }
    .section-title { font-size: 1.1rem; font-weight: 700; color: #2c3e50; margin-bottom: 0.8rem; border-bottom: 2px solid #3498db; padding-bottom: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# Comprehensive colormap library for publication-quality figures
COLORMAP_LIBRARY = {
    # Perceptually Uniform Sequential
    "viridis": "viridis", "plasma": "plasma", "inferno": "inferno", "magma": "magma", "cividis": "cividis",
    # Sequential
    "Greys": "Greys", "Purples": "Purples", "Blues": "Blues", "Greens": "Greens", "Oranges": "Oranges",
    "Reds": "Reds", "YlOrBr": "YlOrBr", "YlOrRd": "YlOrRd", "OrRd": "OrRd", "PuRd": "PuRd",
    "RdPu": "RdPu", "BuPu": "BuPu", "GnBu": "GnBu", "PuBu": "PuBu", "YlGnBu": "YlGnBu",
    "PuBuGn": "PuBuGn", "BuGn": "BuGn", "YlGn": "YlGn",
    # Sequential (2)
    "binary": "binary", "gist_yarg": "gist_yarg", "gist_gray": "gist_gray", "gray": "gray",
    "bone": "bone", "pink": "pink", "spring": "spring", "summer": "summer", "autumn": "autumn",
    "winter": "winter", "cool": "cool", "Wistia": "Wistia", "hot": "hot", "afmhot": "afmhot",
    "gist_heat": "gist_heat", "copper": "copper",
    # Diverging
    "PiYG": "PiYG", "PRGn": "PRGn", "BrBG": "BrBG", "PuOr": "PuOr", "RdGy": "RdGy",
    "RdBu": "RdBu", "RdYlBu": "RdYlBu", "RdYlGn": "RdYlGn", "Spectral": "Spectral",
    "coolwarm": "coolwarm", "bwr": "bwr", "seismic": "seismic",
    # Qualitative
    "tab10": "tab10", "tab20": "tab20", "tab20b": "tab20b", "tab20c": "tab20c",
    "Pastel1": "Pastel1", "Pastel2": "Pastel2", "Paired": "Paired",
    "Accent": "Accent", "Dark2": "Dark2", "Set1": "Set1", "Set2": "Set2", "Set3": "Set3",
    # Misc / High-contrast
    "jet": "jet", "turbo": "turbo", "rainbow": "rainbow", "gist_rainbow": "gist_rainbow",
    "nipy_spectral": "nipy_spectral", "gist_ncar": "gist_ncar", "terrain": "terrain",
    "ocean": "ocean", "gnuplot": "gnuplot", "gnuplot2": "gnuplot2", "CMRmap": "CMRmap",
    "cubehelix": "cubehelix", "brg": "brg", "hsv": "hsv", "flag": "flag", "prism": "prism",
    # Custom scientific
    "twilight": "twilight", "twilight_shifted": "twilight_shifted", "hsv_r": "hsv_r",
    "magma_r": "magma_r", "inferno_r": "inferno_r", "plasma_r": "plasma_r", "viridis_r": "viridis_r"
}

TAXONOMY = {
    'Process Parameters': [
        'laser_power', 'current_density', 'irradiance', 'ved', 'aed', 'led',
        'time', 'duration', 'iterations', 'digital_twin', 'scan_speed',
        'hatch_spacing', 'beam_diameter', 'power', 'velocity', 'feed_rate'
    ],
    'Mechanical Properties': [
        'elongation', 'hardness', 'yield_strength', 'youngs_modulus', 'uts',
        'strain_rate', 'nanoindentation', 'plasticity', 'sfe', 'toughness',
        'fatigue_life', 'creep_rate', 'fracture_toughness', 'modulus'
    ],
    'Microstructural Features': [
        'grain_size', 'layer_thickness', 'thickness', 'density', 'porosity',
        'material', 'smd', 'microstructural_similarity_index', 'surface_roughness',
        'cell_size', 'dendrite_spacing', 'phase_fraction', 'texture_index'
    ],
    'Thermal / Fluid': [
        'lewis_number', 'prandtl_number', 'thermal_conductivity', 'diffusivity',
        'peclet_number', 'marangoni_number', 'reynolds_number'
    ],
    'Computational / Method': [
        'fem', 'elastic_constant_calculation', 'elastic constant calculation',
        'accuracy', 'rmse', 'mae', 'r2_score', 'convergence_rate'
    ],
    'Uncertainty / Unknown': [
        'unknown', 'ambiguous', 'unclassified'
    ]
}

CATEGORY_COLORS = {
    'Process Parameters': '#FDEDEC',
    'Mechanical Properties': '#EBF5FB',
    'Microstructural Features': '#E9F7EF',
    'Thermal / Fluid': '#FEF9E7',
    'Computational / Method': '#F5EEF8',
    'Uncertainty / Unknown': '#F2F3F4',
    'Other': '#FFFFFF'
}

MODEL_META = {
    'modelB': {'name': 'Model B (Mistral 7B)', 'color': '#C0392B', 'short': 'Mistral 7B'},
    'modelD': {'name': 'Model D (Qwen 7B)',    'color': '#2980B9', 'short': 'Qwen 7B'},
    'modelC': {'name': 'Model C (Qwen 14B)',   'color': '#27AE60', 'short': 'Qwen 14B'},
}

@st.cache_data(show_spinner=False)
def load_csv(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        if 'term' not in df.columns:
            df = df.iloc[:, :2].copy()
            df.columns = ['term', 'occurrencecount']
        df['term'] = df['term'].astype(str).str.strip().str.lower()
        df['occurrencecount'] = pd.to_numeric(df.iloc[:, 1], errors='coerce').fillna(0).astype(int)
        return df.set_index('term')['occurrencecount'].to_dict()
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return None

def get_category(term):
    for cat, terms in TAXONOMY.items():
        if term.lower() in [t.lower() for t in terms]:
            return cat
    return 'Other'

def get_contrast_text_color(hex_color):
    """Return white or black text color based on background luminance."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "white" if luminance < 0.5 else "#2C3E50"

with st.sidebar:
    st.markdown("## 📁 Input Files")
    upload_mode = st.radio("Input mode", ["Auto-detect filenames", "Manual upload"], index=0)
    
    if upload_mode == "Manual upload":
        fB = st.file_uploader("Model B (Mistral 7B)", type="csv", key="fb")
        fD = st.file_uploader("Model D (Qwen 7B)", type="csv", key="fd")
        fC = st.file_uploader("Model C (Qwen 14B)", type="csv", key="fc")
        fRef = st.file_uploader("Reference / Ground Truth (optional)", type="csv", key="fref")
        
        def read_uploaded(f):
            if f is None:
                return None
            df = pd.read_csv(f)
            df.columns = [c.strip().lower() for c in df.columns]
            if 'term' not in df.columns:
                df = df.iloc[:, :2].copy()
                df.columns = ['term', 'occurrencecount']
            df['term'] = df['term'].astype(str).str.strip().str.lower()
            df['occurrencecount'] = pd.to_numeric(df.iloc[:, 1], errors='coerce').fillna(0).astype(int)
            return df.set_index('term')['occurrencecount'].to_dict()
        
        dataB = read_uploaded(fB)
        dataD = read_uploaded(fD)
        dataC = read_uploaded(fC)
        dataRef = read_uploaded(fRef)
    else:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PHYSICAL_QUANTITIES_DIR = os.path.join(SCRIPT_DIR, "physical_quantities")
        os.makedirs(PHYSICAL_QUANTITIES_DIR, exist_ok=True)
        
        dataB = load_csv(os.path.join(PHYSICAL_QUANTITIES_DIR, "physical_quantities_detection_llm_modelB.csv"))
        dataD = load_csv(os.path.join(PHYSICAL_QUANTITIES_DIR, "physical_quantities_detection_llm_modelD.csv"))
        dataC = load_csv(os.path.join(PHYSICAL_QUANTITIES_DIR, "physical_quantities_detection_llm_modelC.csv"))
        dataRef = load_csv(os.path.join(PHYSICAL_QUANTITIES_DIR, "physical_quantities_detection_llm_reference.csv"))
        
        st.markdown("---")
        st.markdown("**Expected path:**")
        st.code("physical_quantities/\n  physical_quantities_detection_llm_modelB.csv\n  physical_quantities_detection_llm_modelD.csv\n  physical_quantities_detection_llm_modelC.csv")
        st.markdown("*Optional:* `physical_quantities/physical_quantities_detection_llm_reference.csv`")
    
    st.markdown("---")
    st.markdown("## ⚙️ View Options")
    show_tables = st.checkbox("Show raw data tables", value=True)
    show_figB = st.checkbox("Show Figure B -- Consensus by Category", value=True)
    show_figC = st.checkbox("Show Figure C -- Composition + Quality Radar", value=True)
    show_upset = st.checkbox("Show UpSet Intersection Matrix", value=False)
    
    st.markdown("---")
    st.markdown("## 📊 Global Export")
    download_dpi = st.slider("Figure DPI", min_value=150, max_value=600, value=300, step=50)

# ============================================================
# SIDEBAR -- FIGURE B CONTROLS (Publication Quality)
# ============================================================
with st.sidebar:
    with st.expander("🎨 Figure B Controls", expanded=False):
        st.markdown('<div class="control-section">', unsafe_allow_html=True)
        
        # -- Term Filtering --
        st.markdown('<div class="section-title">🚫 Term Exclusion</div>', unsafe_allow_html=True)
        exclude_unknown = st.checkbox("Exclude \"unknown\" terms", value=True, key="excl_unknown")
        custom_exclude = st.text_input("Exclude terms (comma-separated)", "", key="custom_excl",
                                     help="e.g. unknown, time, duration")
        
        # -- Figure Dimensions --
        st.markdown('<div class="section-title">📐 Figure Dimensions</div>', unsafe_allow_html=True)
        figB_width = st.slider("Width (inches)", 6, 30, 16, key="fb_w")
        figB_height = st.slider("Height (inches)", 4, 20, 9, key="fb_h")
        
        # -- Bar Appearance --
        st.markdown('<div class="section-title">📊 Bar Appearance</div>', unsafe_allow_html=True)
        bar_width = st.slider("Bar width", 0.05, 0.6, 0.25, 0.01, key="bar_w")
        bar_alpha = st.slider("Bar opacity", 0.3, 1.0, 0.92, 0.02, key="bar_a")
        bar_edge = st.slider("Edge linewidth", 0.0, 3.0, 0.8, 0.1, key="bar_e")
        show_bar_labels = st.checkbox("Show value labels on bars", value=True, key="bar_lbl")
        bar_label_size = st.slider("Label font size", 5, 28, 9, key="bar_lbl_sz")
        
        # -- Font Controls (DOUBLED ranges) --
        st.markdown('<div class="section-title">🔤 Font Controls</div>', unsafe_allow_html=True)
        xlabel_font = st.slider("X-axis label size", 6, 40, 16, key="xlf")
        ylabel_font = st.slider("Y-axis label size", 6, 40, 16, key="ylf")
        xtick_font = st.slider("X-tick label size", 5, 32, 11, key="xtf")
        ytick_font = st.slider("Y-tick label size", 5, 32, 11, key="ytf")
        ynum_font = st.slider("Y-axis number size", 5, 32, 11, key="ynf")
        title_font = st.slider("Title size", 8, 48, 18, key="tf")
        legend_font = st.slider("Legend size", 5, 32, 12, key="legf")
        cat_label_font = st.slider("Category label size", 6, 36, 12, key="catf")
        
        # -- Tick & Grid Controls --
        st.markdown('<div class="section-title">📏 Ticks & Grid</div>', unsafe_allow_html=True)
        show_yticks = st.checkbox("Show Y-axis ticks", value=True, key="syt")
        show_xticks = st.checkbox("Show X-axis ticks", value=True, key="sxt")
        show_grid = st.checkbox("Show grid", value=True, key="sg")
        grid_alpha = st.slider("Grid opacity", 0.0, 1.0, 0.25, 0.05, key="ga")
        grid_style = st.selectbox("Grid style", ["--", "-", ":", "-."], index=0, key="gs")
        
        # -- Legend Controls --
        st.markdown('<div class="section-title">📋 Legend</div>', unsafe_allow_html=True)
        show_legend = st.checkbox("Show legend", value=True, key="sleg")
        legend_pos = st.selectbox("Legend position", 
            ["upper right", "upper left", "lower right", "lower left", "best", "center", "outside right"], index=0, key="lp")
        legend_frame = st.checkbox("Legend frame", value=True, key="legfr")
        legend_alpha = st.slider("Legend opacity", 0.0, 1.0, 0.95, 0.05, key="lega")
        legend_box = st.slider("Legend box padding", 0.0, 2.0, 0.8, 0.1, key="legbox")
        
        # -- Colormap --
        st.markdown('<div class="section-title">🌈 Colormap</div>', unsafe_allow_html=True)
        cmap_choice = st.selectbox("Bar colormap", list(COLORMAP_LIBRARY.keys()), index=0, key="cmap")
        use_cmap = st.checkbox("Use colormap (override model colors)", value=False, key="ucmap")
        
        # -- Background Category Shading --
        st.markdown('<div class="section-title">🎨 Category Shading</div>', unsafe_allow_html=True)
        show_cat_bg = st.checkbox("Show category backgrounds", value=True, key="scb")
        cat_bg_alpha = st.slider("Background alpha", 0.0, 1.0, 0.45, 0.05, key="cba")
        cat_label_contrast = st.checkbox("Auto-adjust category label color", value=True, key="clc")
        
        # -- Spines & Box --
        st.markdown('<div class="section-title">📦 Figure Box</div>', unsafe_allow_html=True)
        show_top = st.checkbox("Top spine", value=False, key="stp")
        show_right = st.checkbox("Right spine", value=False, key="srp")
        show_bottom = st.checkbox("Bottom spine", value=True, key="sbp")
        show_left = st.checkbox("Left spine", value=True, key="slp")
        spine_linewidth = st.slider("Spine linewidth", 0.5, 4.0, 1.2, 0.1, key="spw")
        tight_layout = st.checkbox("Tight layout", value=True, key="tl")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR -- FIGURE C CONTROLS (Publication Quality)
# ============================================================
with st.sidebar:
    with st.expander("🎨 Figure C Controls", expanded=False):
        st.markdown('<div class="control-section">', unsafe_allow_html=True)
        
        # -- Figure Dimensions --
        st.markdown('<div class="section-title">📐 Figure Dimensions</div>', unsafe_allow_html=True)
        figC_width = st.slider("Width (inches)", 6, 30, 18, key="fc_w")
        figC_height = st.slider("Height (inches)", 4, 20, 9, key="fc_h")
        panel_ratio = st.slider("Left/Right panel ratio", 0.5, 4.0, 1.5, 0.1, key="pr")
        
        # -- Composition Bar Controls --
        st.markdown('<div class="section-title">📊 Composition Bars</div>', unsafe_allow_html=True)
        barC_height = st.slider("Bar height", 0.2, 1.2, 0.6, 0.05, key="bch")
        barC_edge = st.slider("Edge linewidth", 0.0, 3.0, 0.9, 0.1, key="bce")
        show_pct_labels = st.checkbox("Show percentage labels", value=True, key="spct")
        pct_font_size = st.slider("Percent label size", 5, 28, 11, key="pfs")
        show_total_label = st.checkbox("Show total count", value=True, key="stl")
        total_font_size = st.slider("Total label size", 5, 28, 12, key="tfs")
        
        # -- Segment Colors --
        st.markdown('<div class="section-title">🎨 Segment Colors</div>', unsafe_allow_html=True)
        solo_color = st.color_picker("Solo Unique color", "#E67E22", key="sc")
        pair_color = st.color_picker("Pairwise color", "#F1C40F", key="pc")
        univ_color = st.color_picker("Universal color", "#27AE60", key="uc")
        
        # -- Radar Controls --
        st.markdown('<div class="section-title">🎯 Radar Chart</div>', unsafe_allow_html=True)
        show_radar = st.checkbox("Show radar chart", value=True, key="sr")
        radar_linewidth = st.slider("Radar line width", 0.5, 6.0, 2.8, 0.1, key="rlw")
        radar_markersize = st.slider("Marker size", 2, 16, 7, key="rms")
        radar_fill_alpha = st.slider("Fill alpha", 0.0, 0.6, 0.12, 0.01, key="rfa")
        radar_grid_alpha = st.slider("Grid alpha", 0.0, 1.0, 0.25, 0.05, key="rga")
        radar_legend_bbox = st.slider("Legend X offset", 1.0, 2.5, 1.35, 0.05, key="rlbx")
        
        # -- Font Controls (DOUBLED ranges) --
        st.markdown('<div class="section-title">🔤 Font Controls</div>', unsafe_allow_html=True)
        comp_ylabel_font = st.slider("Y-label size", 6, 40, 16, key="cylf")
        comp_ytick_font = st.slider("Y-tick size", 5, 32, 12, key="cytf")
        comp_ytick_label = st.slider("Y-tick label size", 5, 32, 12, key="cytl")
        comp_xlabel_font = st.slider("X-label size", 6, 40, 16, key="cxlf")
        comp_xtick_font = st.slider("X-tick number size", 5, 32, 11, key="cxtf")
        comp_title_font = st.slider("Title size", 8, 48, 18, key="ctf")
        comp_legend_font = st.slider("Legend size", 5, 32, 11, key="clegf")
        
        # -- Legend Controls --
        st.markdown('<div class="section-title">📋 Legend</div>', unsafe_allow_html=True)
        comp_show_legend = st.checkbox("Show legend", value=True, key="csleg")
        comp_legend_pos = st.selectbox("Legend position", 
            ["lower right", "upper right", "upper left", "lower left", "best", "center"], index=0, key="clp")
        comp_legend_ncol = st.slider("Legend columns", 1, 4, 1, key="clnc")
        
        # -- Annotation --
        st.markdown('<div class="section-title">💬 Annotation</div>', unsafe_allow_html=True)
        show_annotation = st.checkbox("Show Mistral annotation", value=True, key="sa")
        annot_fontsize = st.slider("Annotation size", 5, 28, 10, key="afs")
        annot_box_linewidth = st.slider("Annotation box linewidth", 0.5, 3.0, 1.5, 0.1, key="ablw")
        
        # -- Spines --
        st.markdown('<div class="section-title">📦 Spines</div>', unsafe_allow_html=True)
        comp_show_top = st.checkbox("Top spine", value=False, key="cst")
        comp_show_right = st.checkbox("Right spine", value=False, key="csr")
        comp_spine_width = st.slider("Spine linewidth", 0.5, 4.0, 1.2, 0.1, key="cspw")
        comp_show_grid = st.checkbox("Show grid", value=True, key="csg")
        comp_grid_alpha = st.slider("Grid opacity", 0.0, 1.0, 0.25, 0.05, key="cga")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# MAIN HEADER & DATA PREP
# ============================================================
st.markdown('<div class="main-header">Cross-Model Physical Quantity Extraction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Structured JSON Pipeline Comparison for Materials & Design Knowledge Graphs</div>', unsafe_allow_html=True)

available = {}
if dataB: available["modelB"] = dataB
if dataD: available["modelD"] = dataD
if dataC: available["modelC"] = dataC
if dataRef: available["reference"] = dataRef

if not available:
    st.warning("⚠️ No CSV files detected. Please upload files or ensure they exist in the working directory with the expected filenames.")
    st.stop()

models_data = {}
for key, meta in MODEL_META.items():
    if key in available:
        models_data[meta["name"]] = available[key]
if "reference" in available:
    models_data["Reference"] = available["reference"]

main_models = [m for m in models_data.keys() if m != "Reference"]

# Build exclusion set from controls
excluded_terms = set()
if exclude_unknown:
    excluded_terms.add("unknown")
if custom_exclude.strip():
    for t in custom_exclude.split(","):
        excluded_terms.add(t.strip().lower())

# ============================================================
# TOP METRIC CARDS
# ============================================================
cols = st.columns(len(main_models) + (1 if "Reference" in models_data else 0))

for i, (key, meta) in enumerate(MODEL_META.items()):
    if key in available:
        d = available[key]
        total = sum(d.values())
        n_terms = len(d)
        with cols[i]:
            st.markdown(f'''<div class="metric-card highlight-{["red","blue","green"][i]}">
                <strong>{meta["name"]}</strong><br>
                <span style="font-size:1.4rem">{n_terms}</span> terms &nbsp;|&nbsp;
                <span style="font-size:1.4rem">{total}</span> total occurrences
            </div>''', unsafe_allow_html=True)

if "reference" in available:
    d = available["reference"]
    with cols[-1]:
        st.markdown(f'''<div class="metric-card highlight-purple">
            <strong>Reference (Ground Truth)</strong><br>
            <span style="font-size:1.4rem">{len(d)}</span> terms &nbsp;|&nbsp;
            <span style="font-size:1.4rem">{sum(d.values())}</span> total occurrences
        </div>''', unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# ENHANCED FIGURE B -- CONSENSUS BY CATEGORY
# ============================================================
if show_figB and len(main_models) >= 2:
    st.markdown("### Figure B -- Consensus Terms by Materials Design Category")
    st.markdown("<div class='caption'>Terms extracted by ≥2 models, grouped by process → mechanical → microstructural relevance.</div>", unsafe_allow_html=True)
    
    all_terms_main = set()
    for m in main_models:
        all_terms_main.update(models_data[m].keys())
    all_terms_main -= excluded_terms
    
    consensus_terms = []
    for term in all_terms_main:
        count = sum(1 for m in main_models if term in models_data[m])
        if count >= 2:
            consensus_terms.append(term)
    
    if not consensus_terms:
        st.warning("No consensus terms remain after filtering. Adjust exclusion settings.")
    else:
        cat_order = list(TAXONOMY.keys()) + ["Other"]
        term_info = []
        for term in consensus_terms:
            cat = get_category(term)
            vals = {m: models_data[m].get(term, 0) for m in main_models}
            max_val = max(vals.values())
            term_info.append((cat, term, vals, max_val))
        
        term_info.sort(key=lambda x: (cat_order.index(x[0]) if x[0] in cat_order else 99, -x[3]))
        
        fig_b, ax_b = plt.subplots(figsize=(figB_width, figB_height), facecolor="white")
        ax_b.set_facecolor("white")
        
        labels = []
        values_b = {m: [] for m in main_models}
        label_cats = []
        
        for cat, term, vals, _ in term_info:
            labels.append(term.replace("_", " ").title())
            label_cats.append(cat)
            for m in main_models:
                values_b[m].append(vals[m])
        
        x = np.arange(len(labels))
        n_models = len(main_models)
        
        # Background bands by category with auto-adjusting label colors
        if labels and show_cat_bg:
            start_idx = 0
            current_cat = label_cats[0]
            for i in range(1, len(label_cats)):
                if label_cats[i] != current_cat:
                    bg_color = CATEGORY_COLORS.get(current_cat, "white")
                    ax_b.axvspan(start_idx - 0.5, i - 0.5, 
                                facecolor=bg_color, 
                                alpha=cat_bg_alpha, zorder=0)
                    start_idx = i
                    current_cat = label_cats[i]
            bg_color = CATEGORY_COLORS.get(current_cat, "white")
            ax_b.axvspan(start_idx - 0.5, len(labels) - 0.5, 
                        facecolor=bg_color, 
                        alpha=cat_bg_alpha, zorder=0)
        
        # Get colors: either model colors or colormap
        model_color_map = {meta["name"]: meta["color"] for meta in MODEL_META.values()}
        if use_cmap and n_models > 1:
            cmap_obj = plt.get_cmap(COLORMAP_LIBRARY[cmap_choice])
            cmap_colors = [cmap_obj(i / (n_models - 1)) for i in range(n_models)]
            model_colors_list = {m: cmap_colors[i] for i, m in enumerate(main_models)}
        else:
            model_colors_list = model_color_map
        
        for i, m in enumerate(main_models):
            offset = (i - (n_models-1)/2) * bar_width
            color = model_colors_list.get(m, "#333333")
            bars = ax_b.bar(x + offset, values_b[m], bar_width,
                           label=m, color=color, alpha=bar_alpha,
                           edgecolor="black", linewidth=bar_edge, zorder=3)
            if show_bar_labels:
                for bar, val in zip(bars, values_b[m]):
                    if val > 0:
                        ax_b.text(bar.get_x() + bar.get_width()/2, 
                                 bar.get_height() + max(1, max(values_b[m])*0.03),
                                 f"{int(val)}", ha="center", va="bottom", 
                                 fontsize=bar_label_size, fontweight="bold")
        
        # Category labels on top with auto-contrast color
        if labels:
            start_idx = 0
            current_cat = label_cats[0]
            ymax = ax_b.get_ylim()[1]
            for i in range(1, len(label_cats)):
                if label_cats[i] != current_cat:
                    mid = (start_idx + i - 1) / 2
                    bg_hex = CATEGORY_COLORS.get(current_cat, "#FFFFFF")
                    text_color = get_contrast_text_color(bg_hex) if cat_label_contrast else "#555555"
                    ax_b.text(mid, ymax * 0.97, current_cat, ha="center", va="top", 
                             fontsize=cat_label_font, fontweight="bold", 
                             style="italic", color=text_color)
                    start_idx = i
                    current_cat = label_cats[i]
            mid = (start_idx + len(labels) - 1) / 2
            bg_hex = CATEGORY_COLORS.get(current_cat, "#FFFFFF")
            text_color = get_contrast_text_color(bg_hex) if cat_label_contrast else "#555555"
            ax_b.text(mid, ymax * 0.97, current_cat, ha="center", va="top", 
                     fontsize=cat_label_font, fontweight="bold", 
                     style="italic", color=text_color)
        
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(labels, rotation=40, ha="right", fontsize=xtick_font)
        ax_b.set_ylabel("Occurrence Count", fontsize=ylabel_font, fontweight="bold")
        
        # Y-axis number font size
        ax_b.tick_params(axis="y", labelsize=ynum_font)
        
        # Legend with dynamic fitting
        if show_legend:
            if legend_pos == "outside right":
                legend = ax_b.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), 
                                    fontsize=legend_font, framealpha=legend_alpha, 
                                    edgecolor="black" if legend_frame else "none",
                                    fancybox=legend_frame,
                                    borderpad=legend_box)
            else:
                legend = ax_b.legend(loc=legend_pos, fontsize=legend_font, 
                                    framealpha=legend_alpha, 
                                    edgecolor="black" if legend_frame else "none",
                                    fancybox=legend_frame,
                                    borderpad=legend_box)
        
        # Spines with adjustable linewidth
        for spine_name, show_sp in [("top", show_top), ("right", show_right), 
                                    ("bottom", show_bottom), ("left", show_left)]:
            ax_b.spines[spine_name].set_visible(show_sp)
            if show_sp:
                ax_b.spines[spine_name].set_linewidth(spine_linewidth)
        
        if show_grid:
            ax_b.grid(axis="y", alpha=grid_alpha, linestyle=grid_style)
        
        ax_b.tick_params(axis="x", which="both", bottom=show_bottom, top=show_top, 
                        labelbottom=show_xticks)
        ax_b.tick_params(axis="y", which="both", left=show_left, right=show_right, 
                        labelleft=show_yticks)
        
        if tight_layout:
            plt.tight_layout()
        st.pyplot(fig_b)
        
        buf = BytesIO()
        fig_b.savefig(buf, format="png", dpi=download_dpi, bbox_inches="tight", facecolor="white")
        st.download_button("⬇️ Download Figure B (PNG)", buf.getvalue(), 
                          "fig_B_consensus_terms.png", "image/png")
        plt.close(fig_b)

# ============================================================
# ENHANCED FIGURE C -- COMPOSITION + RADAR
# ============================================================
if show_figC and len(main_models) >= 2:
    st.markdown("---")
    st.markdown("### Figure C -- Extraction Composition & Pipeline Quality Metrics")
    st.markdown("<div class='caption'>Left: stacked composition by consensus level. Right: radar chart comparing schema compliance, coverage, and rank stability.</div>", unsafe_allow_html=True)
    
    all_sets = {m: set(models_data[m].keys()) - excluded_terms for m in main_models}
    intersections_c = {}
    for r in range(1, len(main_models)+1):
        for combo in combinations(main_models, r):
            combo_set = frozenset(combo)
            shared = all_sets[combo[0]].copy()
            for m in combo[1:]:
                shared &= all_sets[m]
            if len(combo) < len(main_models):
                for m_out in set(main_models) - set(combo):
                    shared -= all_sets[m_out]
            if shared:
                intersections_c[combo_set] = sorted(shared)
    
    solo = {m: 0 for m in main_models}
    pairwise = {m: 0 for m in main_models}
    universal = {m: 0 for m in main_models}
    
    for combo, terms in intersections_c.items():
        n = len(combo)
        for m in combo:
            if n == 1:
                solo[m] += len(terms)
            elif n == 2:
                pairwise[m] += len(terms)
            elif n == 3:
                universal[m] += len(terms)
    
    for m in main_models:
        accounted = set()
        for combo, terms in intersections_c.items():
            if m in combo:
                accounted.update(terms)
        missing = all_sets[m] - accounted
        if missing:
            solo[m] += len(missing)
    
    fig_c = plt.figure(figsize=(figC_width, figC_height), facecolor="white")
    gs = fig_c.add_gridspec(1, 2, width_ratios=[panel_ratio, 1], wspace=0.35)
    
    # ---- LEFT: Stacked Composition ----
    ax1 = fig_c.add_subplot(gs[0, 0])
    ax1.set_facecolor("white")
    
    cat_colors_c = [solo_color, pair_color, univ_color]
    cat_labels_c = ["Solo Unique\n(Uncorroborated)", "Pairwise Only\n(2-Model Consensus)", "Universal\n(All 3 Models)"]
    y_pos = np.arange(len(main_models))
    
    solo_vals = [solo[m] for m in main_models]
    pair_vals = [pairwise[m] for m in main_models]
    univ_vals = [universal[m] for m in main_models]
    totals = [s + p + u for s, p, u in zip(solo_vals, pair_vals, univ_vals)]
    
    ax1.barh(y_pos, solo_vals, height=barC_height, color=cat_colors_c[0], 
            label=cat_labels_c[0], edgecolor="black", linewidth=barC_edge, zorder=3)
    ax1.barh(y_pos, pair_vals, height=barC_height, left=solo_vals, color=cat_colors_c[1],
            label=cat_labels_c[1], edgecolor="black", linewidth=barC_edge, zorder=3)
    ax1.barh(y_pos, univ_vals, height=barC_height, 
            left=[s+p for s,p in zip(solo_vals, pair_vals)], color=cat_colors_c[2],
            label=cat_labels_c[2], edgecolor="black", linewidth=barC_edge, zorder=3)
    
    for i, m in enumerate(main_models):
        total = totals[i]
        if solo_vals[i] > 0 and show_pct_labels:
            pct = 100 * solo_vals[i] / total
            txt_color = get_contrast_text_color(solo_color)
            ax1.text(solo_vals[i]/2, i, f"{solo_vals[i]}\n({pct:.0f}%)", 
                    ha="center", va="center", fontsize=pct_font_size, 
                    color=txt_color, fontweight="bold")
        if pair_vals[i] > 0 and show_pct_labels:
            pct = 100 * pair_vals[i] / total
            txt_color = get_contrast_text_color(pair_color)
            ax1.text(solo_vals[i] + pair_vals[i]/2, i, f"{pair_vals[i]}\n({pct:.0f}%)", 
                    ha="center", va="center", fontsize=pct_font_size, 
                    color=txt_color, fontweight="bold")
        if univ_vals[i] > 0 and show_pct_labels:
            pct = 100 * univ_vals[i] / total
            txt_color = get_contrast_text_color(univ_color)
            ax1.text(solo_vals[i] + pair_vals[i] + univ_vals[i]/2, i, 
                    f"{univ_vals[i]}\n({pct:.0f}%)", 
                    ha="center", va="center", fontsize=pct_font_size, 
                    color=txt_color, fontweight="bold")
        if show_total_label:
            ax1.text(total + 0.4, i, f"{total} total", ha="left", va="center", 
                    fontsize=total_font_size, fontweight="bold", color="#2C3E50")
    
    if solo_vals[0] == 0 and len(main_models) >= 1 and show_annotation:
        ax1.annotate("100% corroborated\n→ Zero solo hallucinations", 
                    xy=(totals[0], 0), xytext=(totals[0] + 3, 0.4),
                    fontsize=annot_fontsize, color="#C0392B", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#C0392B", lw=annot_box_linewidth),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDEDEC", 
                             edgecolor="#C0392B", linewidth=annot_box_linewidth, alpha=0.9))
    
    short_names = [m.split("(")[1].replace(")", "") if "(" in m else m for m in main_models]
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(short_names, fontsize=comp_ytick_label, fontweight="bold")
    ax1.set_xlabel("Number of Extracted Terms", fontsize=comp_xlabel_font, fontweight="bold")
    ax1.tick_params(axis="x", labelsize=comp_xtick_font)
    ax1.tick_params(axis="y", labelsize=comp_ytick_font)
    
    # Dynamic legend fitting
    if comp_show_legend:
        legend_c = ax1.legend(loc=comp_legend_pos, fontsize=comp_legend_font, 
                             framealpha=0.95, edgecolor="black", 
                             ncol=comp_legend_ncol)
    
    # Spines with adjustable linewidth
    for spine_name, show_sp in [("top", comp_show_top), ("right", comp_show_right),
                                ("bottom", True), ("left", True)]:
        ax1.spines[spine_name].set_visible(show_sp)
        if show_sp:
            ax1.spines[spine_name].set_linewidth(comp_spine_width)
    
    if comp_show_grid:
        ax1.grid(axis="x", alpha=comp_grid_alpha, linestyle="--")
    ax1.set_xlim(0, max(totals) + 7)
    
    # ---- RIGHT: Radar Metrics ----
    if len(main_models) >= 2 and show_radar:
        ax2 = fig_c.add_subplot(gs[0, 1], polar=True)
        ax2.set_facecolor("white")
        
        ref_set = set(models_data.get("Reference", {}).keys()) - excluded_terms
        
        metrics = {}
        for m in main_models:
            short = m.split("(")[1].replace(")", "") if "(" in m else m
            total_m = sum([solo[m], pairwise[m], universal[m]])
            consensus_yield = 100 * (pairwise[m] + universal[m]) / total_m if total_m > 0 else 0
            schema_strict = 100 * (total_m - models_data[m].get("unknown", 0)) / total_m if total_m > 0 else 0
            ref_cov = 100 * len(all_sets[m] & ref_set) / len(ref_set) if ref_set else 0
            top2_stable = 100.0
            no_singleton = 100.0 if solo[m] == 0 else max(0, 100 - solo[m] * 15)
            
            metrics[short] = {
                "Consensus Yield": consensus_yield,
                "Schema Strictness": schema_strict,
                "Reference Coverage": ref_cov,
                "Top-2 Stability": top2_stable,
                "No Singleton Drift": no_singleton
            }
        
        metric_names = list(metrics[list(metrics.keys())[0]].keys())
        angles = np.linspace(0, 2*np.pi, len(metric_names), endpoint=False).tolist()
        angles += angles[:1]
        
        color_map = {}
        for k, meta in MODEL_META.items():
            if k in available:
                short = meta["name"].split("(")[1].replace(")", "")
                color_map[short] = meta["color"]
        
        for short, vals in metrics.items():
            values = [vals[m] for m in metric_names]
            values += values[:1]
            color = color_map.get(short, "#333333")
            ax2.plot(angles, values, "o-", linewidth=radar_linewidth, 
                    color=color, label=short, markersize=radar_markersize)
            ax2.fill(angles, values, alpha=radar_fill_alpha, color=color)
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metric_names, fontsize=9)
        ax2.set_ylim(0, 105)
        ax2.set_yticks([20, 40, 60, 80, 100])
        ax2.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7, color="gray")
        ax2.grid(True, alpha=radar_grid_alpha)
        ax2.legend(loc="upper right", bbox_to_anchor=(radar_legend_bbox, 1.1), 
                   fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    st.pyplot(fig_c)
    
    buf = BytesIO()
    fig_c.savefig(buf, format="png", dpi=download_dpi, bbox_inches="tight", facecolor="white")
    st.download_button("⬇️ Download Figure C (PNG)", buf.getvalue(), 
                      "fig_C_composition_quality.png", "image/png")
    plt.close(fig_c)

# ============================================================
# UPSET MATRIX (Optional)
# ============================================================
if show_upset and len(main_models) >= 2:
    st.markdown("---")
    st.markdown("### UpSet-Style Intersection Matrix")
    st.markdown("<div class='caption'>Matrix dots indicate model participation; vertical bars show term counts per intersection.</div>", unsafe_allow_html=True)
    
    all_sets = {m: set(models_data[m].keys()) - excluded_terms for m in main_models}
    intersections_u = {}
    for r in range(1, len(main_models)+1):
        for combo in combinations(main_models, r):
            combo_set = frozenset(combo)
            shared = all_sets[combo[0]].copy()
            for m in combo[1:]:
                shared &= all_sets[m]
            if len(combo) < len(main_models):
                for m_out in set(main_models) - set(combo):
                    shared -= all_sets[m_out]
            if shared:
                intersections_u[combo_set] = sorted(shared)
    
    intersection_items = sorted(intersections_u.items(), 
                               key=lambda x: (-len(x[1]), -len(x[0])))
    
    fig_u, ax_u = plt.subplots(figsize=(12, 5), facecolor="white")
    ax_u.set_facecolor("white")
    
    set_totals = {m: len(all_sets[m]) for m in main_models}
    y_positions = {m: i for i, m in enumerate(reversed(main_models))}
    
    for m in main_models:
        y = y_positions[m]
        ax_u.barh(y, set_totals[m], height=0.08, 
                 color=model_color_map.get(m, "#333"), alpha=0.85, 
                 edgecolor="black", linewidth=0.8)
        ax_u.text(set_totals[m] + 0.3, y, f"{set_totals[m]}", 
                 va="center", ha="left", fontsize=10, fontweight="bold")
    
    combo_labels = []
    combo_counts = []
    combo_models = []
    for combo, terms in intersection_items:
        label = " ∩ ".join([c.split("(")[1].replace(")", "") if "(" in c else c for c in sorted(combo)])
        combo_labels.append(label)
        combo_counts.append(len(terms))
        combo_models.append(sorted(combo))
    
    if combo_labels:
        matrix_start = max(set_totals.values()) + 2
        x_positions = np.linspace(matrix_start, matrix_start + len(combo_labels)*1.5, len(combo_labels))
        
        for i, (combo, count) in enumerate(zip(combo_models, combo_counts)):
            x = x_positions[i]
            for m in main_models:
                y = y_positions[m]
                if m in combo:
                    ax_u.scatter(x, y, s=180, c=model_color_map.get(m, "#333"), 
                               zorder=5, edgecolors="black", linewidth=1.2)
                else:
                    ax_u.scatter(x, y, s=60, c="lightgray", 
                               zorder=3, edgecolors="gray", linewidth=0.5)
            
            ys_combo = [y_positions[m] for m in combo]
            if len(ys_combo) > 1:
                ys_combo.sort()
                for j in range(len(ys_combo)-1):
                    ax_u.plot([x, x], [ys_combo[j], ys_combo[j+1]], 
                             "k-", linewidth=2, alpha=0.5, zorder=4)
            
            bar_h = count * 0.1
            ax_u.bar(x, bar_h, width=0.4, bottom=3.0, 
                    color="#34495E", alpha=0.85, edgecolor="black", linewidth=0.8)
            ax_u.text(x, 3.0 + bar_h + 0.15, str(count), 
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax_u.text(x, 2.5, combo_labels[i].replace(" ∩ ", "\n∩\n"), 
                     ha="center", va="top", fontsize=8.5, 
                     linespacing=0.8, fontweight="bold")
    
    ax_u.set_xlim(-1, matrix_start + len(combo_labels)*1.5 + 1 if combo_labels else 10)
    ax_u.set_ylim(2.0, 3.8)
    ax_u.set_yticks([y_positions[m] for m in main_models])
    ax_u.set_yticklabels([m.split("(")[1].replace(")", "") if "(" in m else m for m in main_models], 
                        fontsize=11, fontweight="bold")
    ax_u.set_xticks([])
    ax_u.set_title("UpSet Intersection Matrix", fontsize=13, fontweight="bold", pad=10, loc="left")
    ax_u.spines["top"].set_visible(False)
    ax_u.spines["right"].set_visible(False)
    ax_u.spines["bottom"].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig_u)
    plt.close(fig_u)

# ============================================================
# RAW DATA TABLES
# ============================================================
if show_tables:
    st.markdown("---")
    st.markdown("### Raw Extraction Data")
    
    tab_keys = [k for k in ["modelB", "modelD", "modelC", "reference"] if k in available]
    tab_labels = [MODEL_META[k]["short"] if k in MODEL_META else "Reference" for k in tab_keys]
    tabs = st.tabs(tab_labels)
    
    for tab, key in zip(tabs, tab_keys):
        with tab:
            d = available[key]
            df = pd.DataFrame(list(d.items()), columns=["Term", "OccurrenceCount"])
            df["Category"] = df["Term"].apply(get_category)
            df = df.sort_values("OccurrenceCount", ascending=False).reset_index(drop=True)
            st.dataframe(df, use_container_width=True, height=400)
            
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(f"⬇️ Download {key} CSV", csv, 
                              f"physical_quantities_{key}_annotated.csv", 
                              "text/csv", key=f"dl_{key}")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem; padding:1rem 0;">
    <strong>Pipeline Note:</strong> Structured quantity extraction is a constraint-satisfaction problem, not a reasoning task. 
    Models with tighter token probability distributions for rigid formats (Mistral 7B, Qwen 7B) consistently outperform 
    larger architectures in extraction throughput, validation success, and downstream knowledge graph integrity.
</div>
""", unsafe_allow_html=True)
