"""
AlloyExtraction Nexus — Memory-Hardened, Publication-Grade Cross-Model Visualization v2.4
Unified Chord · Sankey · Network · Heatmap · Circos · 3D UMAP · Animation
Hybrid Material Validator · Editable Aliases (Data Editor) · Aggressive Caching & GC
+ Diagnostic Panel · Limited Cache Entries · Capped Figure Sizes
+ 50+ Color Colormaps (Turbo/Jet/Inferno/Rainbow/etc.) · Interactive Category Toggling (Morphology Control)
+ v2.3: Fixed Chord & Circos spatial arrangement, label padding, and text orientation
+ v2.4: Configurable label modes (parallel/vertical/inclined/radial), Chord legend/model toggles, enhanced Circos padding
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge, PathPatch
from matplotlib.path import Path
from io import BytesIO
import os, json, re, gc, tempfile, psutil
from collections import defaultdict
from functools import lru_cache
from itertools import combinations

# ------------------------------------------------------------------
# Optional backends
# ------------------------------------------------------------------
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except Exception:
    HAS_PYVIS = False

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# ------------------------------------------------------------------
# GLOBAL PUBLICATION DEFAULTS + MEMORY GUARD
# ------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#2C3E50',
    'text.color': '#1a1a2e',
    'axes.labelcolor': '#1a1a2e',
    'xtick.color': '#2C3E50',
    'ytick.color': '#2C3E50',
    'figure.max_open_warning': 0,
})

plt.switch_backend('Agg')

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="AlloyExtraction Nexus — Memory-Hardened v2.4",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.6rem; font-weight: 800; color: #1a1a2e; margin-bottom: 0.3rem; letter-spacing: -0.5px; }
    .sub-header { font-size: 1.2rem; color: #5a5a7a; margin-bottom: 2rem; font-weight: 400; }
    .metric-card { background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); border-left: 4px solid #2980b9; padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
    .highlight-red { border-left-color: #c0392b; }
    .highlight-green { border-left-color: #27ae60; }
    .highlight-blue { border-left-color: #2980b9; }
    .highlight-gold { border-left-color: #f39c12; }
    .highlight-purple { border-left-color: #8e44ad; }
    .caption { font-size: 0.95rem; color: #555; font-style: italic; margin-bottom: 1rem; }
    .control-section { background: linear-gradient(180deg, #f5f7fa 0%, #eef1f5 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border: 1px solid #dde2e8; }
    .section-title { font-size: 1.15rem; font-weight: 700; color: #2c3e50; margin-bottom: 0.8rem; border-bottom: 2px solid #3498db; padding-bottom: 0.3rem; }
    .memory-btn { background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); color: white; font-weight: bold; border: none; padding: 0.6rem 1.2rem; border-radius: 6px; cursor: pointer; }
    .memory-btn:hover { background: linear-gradient(135deg, #c0392b 0%, #a93226 100%); }
    .diag-metric { display: inline-block; margin: 0.3rem 0.5rem 0.3rem 0; padding: 0.4rem 0.8rem; background: #f8f9fa; border-radius: 4px; font-size: 0.9rem; }
    .diag-warn { color: #c0392b; font-weight: 600; }
    .diag-ok { color: #27ae60; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# MEMORY EMERGENCY BUTTON (Sidebar top)
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🚨 Memory Management")
    if st.button("🧹 Clear All Memory & Cache", type="primary", key="clear_mem"):
        plt.close('all')
        gc.collect()
        st.cache_data.clear()
        st.cache_resource.clear()
        for key in list(st.session_state.keys()):
            if key not in ['doi_aliases', 'material_aliases', 'validation_cache', 'alias_temp_doi', 'alias_temp_mat']:
                del st.session_state[key]
        st.success("Memory cleared! Rerunning...")
        st.rerun()

# ------------------------------------------------------------------
# PERIODIC TABLE & METADATA
# ------------------------------------------------------------------
PERIODIC_TABLE = {
    'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar',
    'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br',
    'Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb',
    'Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho',
    'Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi',
    'Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es',
    'Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc',
    'Lv','Ts','Og'
}

MODEL_META = {
    'modelB': {'name': 'Model B (Mistral 7B)', 'color': '#C0392B', 'short': 'Mistral 7B'},
    'modelC': {'name': 'Model C (Qwen 14B)',   'color': '#27AE60', 'short': 'Qwen 14B'},
    'modelD': {'name': 'Model D (Qwen 7B)',    'color': '#2980B9', 'short': 'Qwen 7B'},
}

TAXONOMY = {
    'Process Parameters': [
        'laser_power', 'current_density', 'irradiance', 'ved', 'aed', 'led',
        'time', 'duration', 'iterations', 'digital_twin', 'scan_speed',
        'hatch_spacing', 'beam_diameter', 'power', 'velocity', 'feed_rate', 'lpbf'
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
    'Alloy / Composition': [
        'alsimgzr', 'ti-au', 'ti3au', 'heas_mpeas', 'al', 'al2cu', 'nt_cu', 'cu6sn5',
        'au-ti', 'ti-cr', 'metallic_glass', 'b0.3er0.5al0.2n', 'aln', 'au', 'ti',
        'cu', 'fe', 'ti6al4v', 'ti2cu', 'cu-mn', 'cr0.4w0.5(zrhfnb)0.1',
        'cr0.5w0.3(vnbta)0.2', 'mo0.1ti0.8(vnbta)0.1', 'ti0.7(zrhfnb)0.3',
        'sdss_2507', 'sdss', 'strontium_titanate', 'steel_sheet', 'solder_interconnects',
        'tib2/alsimgzr', 'al-si-mg-zr', 'gold', 'tib2_al-si-mg-zr'
    ],
    'Uncertainty / Unknown': ['unknown', 'ambiguous', 'unclassified'],
}

CATEGORY_COLORS = {
    'Process Parameters':    '#FDEDEC',
    'Mechanical Properties': '#EBF5FB',
    'Microstructural Features':'#E9F7EF',
    'Thermal / Fluid':       '#FEF9E7',
    'Computational / Method':'#F5EEF8',
    'Alloy / Composition':   '#E8F6F3',
    'Uncertainty / Unknown': '#F2F3F4',
    'Other':                 '#FFFFFF',
}

SYNONYM_MAP = {
    'al–si–mg–zr alloy': 'alsimgzr',
    'al-si-mg-zr alloy': 'alsimgzr',
    'tib2/al–si–mg–zr alloy': 'tib2_alsimgzr',
    'ti–au': 'ti-au',
    'au-ti': 'ti-au',
    'aln ceramics': 'aln',
    'cu6sn5 imc': 'cu6sn5',
    'ti6al4v matrix': 'ti6al4v',
    'ti6al4v alloy': 'ti6al4v',
    'ti2cu imc': 'ti2cu',
    'ti2cu imc/ti6al4v matrix': 'ti2cu_ti6al4v',
    'cu–mn': 'cu-mn',
    'strontium titanate': 'strontium_titanate',
    'steel sheet': 'steel_sheet',
    'solder interconnects': 'solder_interconnects',
    'b0.3er0.5al0.2n': 'b0.3er0.5al0.2n',
    'heas_mpeas': 'heas_mpeas',
    'metallic_glass': 'metallic_glass',
    'lpbf': 'lpbf',
    'sdss_2507': 'sdss_2507',
    'sdss': 'sdss',
    'nt_cu': 'nt_cu',
    'yield_strength': 'yield_strength',
    'nanoindentation': 'nanoindentation',
    'lewis_number': 'lewis_number',
    'laser_power': 'laser_power',
    'ti-cr alloy': 'ti-cr',
    'gold': 'au',
    'tib2_al-si-mg-zr': 'tib2_alsimgzr',
}

# ------------------------------------------------------------------
# 50+ COLORMAP CATALOG (v2.3+)
# ------------------------------------------------------------------
COLORMAP_CATALOG = {
    "Perceptually Uniform": [
        "viridis", "plasma", "inferno", "magma", "cividis"
    ],
    "Sequential": [
        "turbo", "jet", "rainbow", "gist_rainbow", "gist_ncar", "nipy_spectral",
        "gnuplot", "gnuplot2", "CMRmap", "cubehelix", "brg", "afmhot",
        "hot", "cool", "spring", "summer", "autumn", "winter", "copper",
        "bone", "pink", "terrain", "ocean"
    ],
    "Diverging": [
        "coolwarm", "bwr", "seismic", "RdYlBu", "RdYlGn", "Spectral"
    ],
    "Cyclic": [
        "twilight", "twilight_shifted", "hsv", "flag", "prism"
    ],
    "Qualitative (≤20)": [
        "tab10", "tab20", "Set1", "Set2", "Set3", "Paired", "Dark2", "Accent", "Pastel1", "Pastel2"
    ]
}
ALL_COLORMAPS = []
for group, cmaps in COLORMAP_CATALOG.items():
    ALL_COLORMAPS.extend(cmaps)
ALL_COLORMAPS = sorted(list(set(ALL_COLORMAPS)))

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def normalize_term(term: str) -> str:
    t = term.strip().lower()
    t = t.replace('–', '-').replace('—', '-')
    return SYNONYM_MAP.get(t, t)

def get_category(term: str) -> str:
    t = term.lower()
    for cat, terms in TAXONOMY.items():
        if t in [x.lower() for x in terms]:
            return cat
    return 'Other'

def get_contrast_text_color(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "white" if luminance < 0.5 else "#2C3E50"

def prettify_material(name: str) -> str:
    overrides = {
        'heas_mpeas': 'HEAs / MPEAs',
        'metallic_glass': 'Metallic Glass',
        'strontium_titanate': 'SrTiO₃',
        'steel_sheet': 'Steel Sheet',
        'solder_interconnects': 'Solder Interconnects',
        'nanoindentation': 'Nanoindentation',
        'yield_strength': 'Yield Strength',
        'laser_power': 'Laser Power',
        'lewis_number': 'Lewis Number',
        'lpbf': 'LPBF',
        'sdss_2507': 'SDSS 2507',
        'sdss': 'SDSS',
        'nt_cu': 'NT Cu',
        'tib2_alsimgzr': 'TiB₂/Al-Si-Mg-Zr',
        'ti2cu_ti6al4v': 'Ti₂Cu / Ti-6Al-4V',
        'b0.3er0.5al0.2n': 'B₀.₃Er₀.₅Al₀.₂N',
    }
    if name in overrides:
        return overrides[name]
    s = name.lower()
    parts = []
    i = 0
    while i < len(s):
        matched = False
        if i + 1 < len(s):
            two_check = s[i].upper() + s[i+1].lower()
            if two_check in PERIODIC_TABLE:
                parts.append(two_check)
                i += 2
                matched = True
        if not matched:
            one = s[i].upper()
            if one in PERIODIC_TABLE and s[i].isalpha():
                parts.append(one)
                i += 1
                matched = True
        if not matched:
            parts.append(s[i])
            i += 1
    result = []
    for j, p in enumerate(parts):
        if j > 0 and p in PERIODIC_TABLE and parts[j-1] in PERIODIC_TABLE:
            result.append('-')
        result.append(p)
    pretty = ''.join(result)
    if pretty and pretty[0].islower():
        pretty = pretty[0].upper() + pretty[1:]
    return pretty if pretty != name else name.replace('_', ' ').title()

# ------------------------------------------------------------------
# HYBRID MATERIAL VALIDATOR
# ------------------------------------------------------------------
def llm_validate_material(term: str) -> dict:
    return {'is_material': None, 'method': 'llm_stub', 'reason': 'No LLM loaded'}

def regex_validate_material(term: str) -> dict:
    t = term.lower()
    if t in [x.lower() for x in TAXONOMY['Alloy / Composition']]:
        return {'is_material': True, 'method': 'taxonomy', 'reason': 'Listed in Alloy/Composition taxonomy'}
    for cat, terms in TAXONOMY.items():
        if cat != 'Alloy / Composition' and t in [x.lower() for x in terms]:
            return {'is_material': False, 'method': 'taxonomy', 'reason': f'Listed in {cat} taxonomy'}
    found_elements = set()
    s = t
    i = 0
    while i < len(s):
        matched = False
        if i + 1 < len(s):
            two = s[i].upper() + s[i+1].lower()
            if two in PERIODIC_TABLE:
                found_elements.add(two)
                i += 2
                matched = True
        if not matched:
            one = s[i].upper()
            if one in PERIODIC_TABLE and s[i].isalpha():
                found_elements.add(one)
            i += 1
    indicators = ['alloy', 'imc', 'matrix', 'ceramic', 'intermetallic', 'metal', 'glass', 'solder']
    has_indicator = any(ind in t for ind in indicators)
    if len(found_elements) >= 2 or has_indicator:
        return {'is_material': True, 'method': 'regex+periodic_table',
                'reason': f'Elements: {", ".join(sorted(found_elements))}' + ('; indicator' if has_indicator else '')}
    if len(found_elements) == 1 and len(t) <= 2:
        return {'is_material': True, 'method': 'regex+periodic_table', 'reason': f'Element symbol: {list(found_elements)[0]}'}
    return {'is_material': False, 'method': 'regex', 'reason': 'No material signatures detected'}

def validate_material(term: str, use_llm: bool = False) -> dict:
    if use_llm:
        llm_res = llm_validate_material(term)
        if llm_res['is_material'] is not None:
            return llm_res
    return regex_validate_material(term)

# ------------------------------------------------------------------
# CACHED DATA LOADING
# ------------------------------------------------------------------
DEFAULT_DATA_B = """DOI,Materials_Alloys
10.1007/s40195-025-01825-1,"al–si–mg–zr alloy, lpbf, alsimgzr, tib2/al–si–mg–zr alloy"
10.1016/j.apenergy.2024.122901,"b0.3er0.5al0.2n"
10.1016/j.commatsci.2025.113875,"ti-au"
10.1016/j.engappai.2024.107902,"heas_mpeas"
10.1016/j.ijsolstr.2024.112894,"al, ti3au, nanoindentation, al2cu"
10.1016/j.jallcom.2024.174876,"nt_cu, cu6sn5"
10.1016/j.jestch.2023.101413,"au-ti"
10.1016/j.measurement.2024.114123,"lewis_number"
10.1016/j.msea.2025.148865,"yield_strength, alsimgzr"
10.1016/j.scriptamat.2024.116027,"ti-cr alloy"
10.1016/j.surfin.2023.102728,"lewis_number, cu6sn5, solder interconnects"
10.1080/17452759.2024.2416518,"lpbf, alsimgzr, metallic_glass"
10.1109/ICEPT56209.2022.9873310,"cu6sn5 imc"
10.3390/met12060964,"heas_mpeas"
10.3390/met12111884,"strontium titanate, steel sheet, nanoindentation"
10.26434/chemrxiv-2025-sk6h5,"alsimgzr, sdss_2507, nanoindentation, laser_power, fe"""

DEFAULT_DATA_C = """DOI,Materials_Alloys
10.1007/s00366-025-02117-z,"laser_power, yield_strength, cu–mn"
10.1016/j.commatsci.2025.113875,"ti3au, yield_strength"
10.1016/j.engappai.2024.107902,"cr0.4w0.5(zrhfnb)0.1, heas_mpeas, cr0.5w0.3(vnbta)0.2, mo0.1ti0.8(vnbta)0.1, ti0.7(zrhfnb)0.3"
10.1016/j.ijsolstr.2024.112894,"nanoindentation, al, al2cu, yield_strength"
10.1016/j.jallcom.2024.174876,"cu6sn5"
10.1016/j.jestch.2023.101413,"ti3au"
10.1016/j.matdes.2024.113312,"ti6al4v, ti6al4v matrix, lewis_number, ti6al4v alloy, ti2cu imc, ti2cu imc/ti6al4v matrix, yield_strength"
10.1016/j.surfin.2023.102728,"cu6sn5"
10.1080/17452759.2024.2416518,"alsimgzr, metallic_glass"""

DEFAULT_DATA_D = """DOI,Materials_Alloys
10.1016/j.apenergy.2024.122901,"b0.3er0.5al0.2n, aln ceramics, b0.3er0.5al0.2n, aln"
10.1016/j.commatsci.2025.113875,"ti-au, alsimgzr, ti–au"
10.1016/j.commatsci.2025.114456,"au, alsimgzr"
10.1016/j.jallcom.2024.174876,"cu, nt_cu, cu6sn5"
10.1016/j.jestch.2023.101413,"ti, alsimgzr, ti-au, gold, ti3au, lewis_number"
10.1016/j.matdes.2024.113312,"ti6al4v, cu, cu, ti6al4v"
10.1016/j.msea.2025.148865,"alsimgzr, yield_strength"
10.1080/17452759.2024.2416518,"metallic_glass, alsimgzr"
10.26434/chemrxiv-2025-sk6h5,"lewis_number, nanoindentation, sdss_2507, laser_power, sdss"""

@st.cache_data(ttl=600, max_entries=3, show_spinner="Parsing CSV data...")
def parse_alloys_csv_cached(filepath_or_content, model_name):
    try:
        if os.path.exists(filepath_or_content):
            with open(filepath_or_content, 'r', encoding='utf-8') as f:
                raw = f.read()
        else:
            raw = filepath_or_content
    except Exception:
        raw = filepath_or_content
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        return {}
    records = []
    if lines[0].startswith('DOI,') or lines[0].startswith('doi,'):
        df = pd.read_csv(pd.io.common.StringIO(raw))
        df.columns = [c.strip() for c in df.columns]
        for _, row in df.iterrows():
            doi = str(row.iloc[0]).strip()
            mats = str(row.iloc[1]).strip() if len(row) > 1 else ""
            records.append((doi, mats))
    else:
        for line in lines:
            if ':' in line and '.pdf' in line.split(':')[0]:
                left, right = line.split(':', 1)
                doi = left.replace('.pdf', '').replace('_', '/').strip()
                records.append((doi, right.strip()))
            elif ',' in line:
                parts = line.split(',', 1)
                records.append((parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""))
    data = defaultdict(lambda: defaultdict(int))
    for doi, mat_str in records:
        if not doi or doi.lower() == 'doi':
            continue
        mats = [normalize_term(m) for m in mat_str.split(',') if m.strip()]
        seen = set()
        for m in mats:
            if m and m not in seen:
                seen.add(m)
                data[doi][m] += 1
    return dict(data)

@st.cache_data(ttl=600, max_entries=1, show_spinner="Loading data...")
def load_data_from_disk_or_default():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ALLOYS_DIR = os.path.join(SCRIPT_DIR, "alloys_materials")
    os.makedirs(ALLOYS_DIR, exist_ok=True)
    paths = {
        'modelB': os.path.join(ALLOYS_DIR, "alloys_materials_in_documents_searched_via_LLM_modelB.csv"),
        'modelC': os.path.join(ALLOYS_DIR, "alloys_materials_in_documents_searched_via_LLM_modelC.csv"),
        'modelD': os.path.join(ALLOYS_DIR, "alloys_materials_in_documents_searched_via_LLM_modelD.csv"),
    }
    loaded = {}
    for key, path in paths.items():
        if os.path.exists(path):
            loaded[key] = parse_alloys_csv_cached(path, MODEL_META[key]['name'])
        else:
            default = {'modelB': DEFAULT_DATA_B, 'modelC': DEFAULT_DATA_C, 'modelD': DEFAULT_DATA_D}[key]
            loaded[key] = parse_alloys_csv_cached(default, MODEL_META[key]['name'])
    return loaded

# ------------------------------------------------------------------
# ALIAS MANAGEMENT
# ------------------------------------------------------------------
def init_aliases(all_dois, all_materials):
    if 'doi_aliases' not in st.session_state:
        st.session_state.doi_aliases = {}
    for i, doi in enumerate(sorted(all_dois)):
        if doi not in st.session_state.doi_aliases:
            st.session_state.doi_aliases[doi] = f"[{chr(65 + i)}]" if i < 26 else f"[{i+1}]"

    if 'material_aliases' not in st.session_state:
        st.session_state.material_aliases = {}
    for mat in sorted(all_materials):
        if mat not in st.session_state.material_aliases:
            st.session_state.material_aliases[mat] = prettify_material(mat)

def get_alias(doi):
    return st.session_state.doi_aliases.get(doi, doi[:20])

def get_mat_label(mat):
    return st.session_state.material_aliases.get(mat, mat)

# ------------------------------------------------------------------
# CACHED FILTERING
# ------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner="Validating materials...")
def filter_data_by_validation_cached(data_dict_tuple, materials_only, enable_validator):
    data_dict = {k: dict(v) for k, v in data_dict_tuple}
    if not enable_validator or not materials_only:
        return data_dict
    filtered = {}
    for mk, data in data_dict.items():
        fdata = {}
        for doi, mats in data.items():
            fmats = {}
            for mat, cnt in mats.items():
                res = validate_material(mat, use_llm=False)
                if res['is_material']:
                    fmats[mat] = cnt
            if fmats:
                fdata[doi] = fmats
        filtered[mk] = fdata
    return filtered

# ------------------------------------------------------------------
# CACHED UMAP EMBEDDING
# ------------------------------------------------------------------
@st.cache_resource(ttl=600, max_entries=3, show_spinner="Computing UMAP embedding...")
def compute_umap_embedding_cached(features_tuple, n_neighbors, min_dist, metric):
    features = np.array(features_tuple)
    n_samples = len(features)
    nn = min(n_neighbors, n_samples - 1)
    if nn < 2 or n_samples < 3:
        return None
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    reducer = umap.UMAP(n_components=3, n_neighbors=nn, min_dist=min_dist,
                        metric=metric, random_state=42)
    emb = reducer.fit_transform(X)
    return emb.tolist()

# ------------------------------------------------------------------
# CACHED PYVIS NETWORK
# ------------------------------------------------------------------
@st.cache_resource(ttl=300, max_entries=3, show_spinner="Building network graph...")
def get_network_html_cached(display_data_json):
    if not HAS_PYVIS:
        return None, ""
    import json
    display_data = json.loads(display_data_json)

    net = Network(height='720px', width='100%', bgcolor='white', font_color='black', heading='')
    net.barnes_hut(gravity=-9000, central_gravity=0.35, spring_length=140,
                   spring_strength=0.04, damping=0.09)

    all_papers = set()
    all_materials = set()
    for data in display_data.values():
        all_papers.update(data.keys())
        for mats in data.values():
            all_materials.update(mats.keys())

    for doi in sorted(all_papers):
        alias = get_alias(doi)
        net.add_node(doi, label=alias, title=f"DOI: {doi}", shape='box',
                     color={'background': '#E8F4FD', 'border': '#2980B9'},
                     borderWidth=2, font={'size': 14, 'face': 'arial', 'color': '#1a1a2e'},
                     size=18)

    for mat in sorted(all_materials):
        cat = get_category(mat)
        fill = CATEGORY_COLORS.get(cat, '#FFFFFF')
        net.add_node(mat, label=get_mat_label(mat), shape='dot',
                     color={'background': fill, 'border': '#555555'},
                     borderWidth=2, font={'size': 12, 'color': '#2c3e50'}, size=14,
                     title=f"Category: {cat}")

    for model_key, data in display_data.items():
        meta = MODEL_META[model_key]
        for doi, mats in data.items():
            for mat, cnt in mats.items():
                net.add_edge(doi, mat, width=min(cnt, 4), color=meta['color'],
                             title=f"{meta['name']}: {cnt}x", smooth={'type': 'continuous'})

    net.set_options('{"physics": {"stabilization": {"iterations": 200}}, "interaction": {"hover": true, "tooltipDelay": 100}}')

    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
        net.save_graph(f.name)
        with open(f.name, 'r', encoding='utf-8') as hf:
            html = hf.read()
        os.remove(f.name)

    legend_html = """
    <div style="margin-top:8px; padding:10px; background:#f8f9fa; border:1px solid #dde2e8; border-radius:6px; font-family:serif;">
    <strong style="font-size:1.05rem;">Model Edge Legend</strong><br>
    <span style="display:inline-block; width:14px; height:14px; background:#C0392B; margin-right:6px; border:1px solid black;"></span> Mistral 7B<br>
    <span style="display:inline-block; width:14px; height:14px; background:#27AE60; margin-right:6px; border:1px solid black;"></span> Qwen 14B<br>
    <span style="display:inline-block; width:14px; height:14px; background:#2980B9; margin-right:6px; border:1px solid black;"></span> Qwen 7B<br>
    <span style="display:inline-block; width:14px; height:14px; background:#F39C12; margin-right:6px; border:1px solid black;"></span> Consensus (≥2 models)
    </div>
    """
    return html, legend_html

# ------------------------------------------------------------------
# FIGURE RENDERING WITH GUARANTEED CLEANUP
# ------------------------------------------------------------------
def render_matplotlib_figure(fig, filename, dpi):
    if fig is None:
        return
    buf = None
    try:
        st.pyplot(fig)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        st.download_button(f"⬇️ Download {filename}", buf,
                           file_name=filename, mime="image/png", key=f"dl_{filename.replace('.', '_')}")
    except Exception as e:
        st.error(f"Figure render failed: {e}")
    finally:
        if fig:
            plt.close(fig)

# ------------------------------------------------------------------
# v2.4 UNIFIED CHORD — Label Modes + Legend/Model Toggles
# ------------------------------------------------------------------
def draw_unified_chord(all_models_data, figsize=(16, 16), radius=1.0,
                       node_width=0.08, ribbon_alpha=0.55, font_size=10,
                       paper_span=np.pi * 0.80, material_span=np.pi * 0.80,
                       min_ribbon_width=0.8, max_ribbon_width=6.0,
                       cmap_name='turbo', gap_padding=0.015,
                       label_offset=0.20, curve_tension=0.35,
                       label_mode='radial',
                       show_models=None):
    """
    v2.4 Chord diagram with:
    - 4 label modes: radial, parallel, vertical, inclined
    - Per-model & consensus/DOI visibility toggles
    - Dynamic label padding & gap separation
    - Auto colormap fallback for >20 materials
    - FIXED: Label alignment logic to prevent bunching in center.
    """
    if show_models is None:
        show_models = {'modelB': True, 'modelC': True, 'modelD': True,
                       'consensus': True, 'doi_nodes': True}

    figsize = (min(figsize[0], 24), min(figsize[1], 24))

    # Filter data by selected models
    filtered_data = {}
    for mk, data in all_models_data.items():
        if show_models.get(mk, True):
            filtered_data[mk] = data
    if not filtered_data:
        return None

    all_dois = sorted({d for data in filtered_data.values() for d in data})
    all_mats = sorted({m for data in filtered_data.values() for mats in data.values() for m in mats})
    if not all_dois or not all_mats:
        return None

    n_mats = len(all_mats)
    actual_cmap = cmap_name
    qual_cmaps = ['tab20', 'tab10', 'Set1', 'Set2', 'Set3', 'Paired', 'Dark2', 'Accent', 'Pastel1', 'Pastel2']
    if actual_cmap in qual_cmaps and n_mats > 20:
        actual_cmap = 'turbo'
    elif actual_cmap in qual_cmaps and n_mats > 10:
        actual_cmap = 'tab20'

    cmap = plt.get_cmap(actual_cmap, max(n_mats, 1))
    mat_colors = {mat: mcolors.to_hex(cmap(i)) for i, mat in enumerate(all_mats)}

    doi_counts = defaultdict(int)
    mat_counts = defaultdict(int)
    connections = []
    for mk, data in filtered_data.items():
        for doi, mats in data.items():
            for mat, cnt in mats.items():
                doi_counts[doi] += cnt
                mat_counts[mat] += cnt
                connections.append((doi, mat, cnt, mk))

    pair_models = defaultdict(list)
    for doi, mat, cnt, mk in connections:
        pair_models[(doi, mat)].append(mk)

    unique_connections = []
    for (doi, mat), models in pair_models.items():
        total_cnt = sum(c for d, m, c, k in connections if d == doi and m == mat)
        consensus = len(models)
        if consensus >= 2 and not show_models.get('consensus', True):
            pass
        unique_connections.append((doi, mat, total_cnt, consensus, models))

    # --- PAPER ARC (top half) ---
    p_total = sum(doi_counts.values())
    n_dois = len(all_dois)
    total_gap = gap_padding * n_dois
    available_paper = paper_span - total_gap
    p_start = np.pi / 2 + available_paper / 2 + total_gap / 2
    p_angles = {}
    cur = p_start
    for doi in all_dois:
        cnt = doi_counts[doi]
        w = (cnt / p_total) * available_paper if p_total > 0 else 0
        p_angles[doi] = {'start': cur, 'end': cur - w, 'mid': cur - w / 2, 'width': w}
        cur -= (w + gap_padding)

    # --- MATERIAL ARC (bottom half) ---
    m_total = sum(mat_counts.values())
    n_mats = len(all_mats)
    total_mgap = gap_padding * n_mats
    available_mat = material_span - total_mgap
    m_start = -np.pi / 2 - available_mat / 2 - total_mgap / 2
    m_angles = {}
    cur = m_start
    for mat in all_mats:
        cnt = mat_counts[mat]
        w = (cnt / m_total) * available_mat if m_total > 0 else 0
        m_angles[mat] = {'start': cur, 'end': cur + w, 'mid': cur + w / 2, 'width': w}
        cur += (w + gap_padding)

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    # --- Helper: compute label transform for a given angle ---
    def compute_label_transform(angle_rad, mode, is_paper=True):
        """Returns (rotation_deg, ha, va, offset_x, offset_y)"""
        deg = np.degrees(angle_rad)
        deg_norm = deg % 360  # Normalize to 0-360 range
        
        # Default offsets
        offset_x, offset_y = 0, 0
        
        if mode == 'radial':
            # Text points away from center
            if 90 < deg_norm < 270:
                # Left side: Text should go Left (ha='right')
                text_rot = deg_norm - 180
                ha = 'right'
            else:
                # Right side: Text should go Right (ha='left')
                text_rot = deg_norm
                ha = 'left'
                
            # Vertical alignment based on top/bottom arc
            if is_paper:
                # Top arc: Text should be Above (va='bottom')
                va = 'bottom'
                if 80 < deg_norm < 100: ha = 'center' 
            else:
                # Bottom arc: Text should be Below (va='top')
                va = 'top'
                if 260 < deg_norm < 280: ha = 'center'

        elif mode == 'parallel':
            # Tangent to the circle
            text_rot = deg_norm + 90
            if text_rot > 180: text_rot -= 360
            
            # Ensure text is upright
            if text_rot > 90 or text_rot < -90:
                 text_rot += 180
                 if text_rot > 180: text_rot -= 360
            
            ha = 'center'
            va = 'center'
            # Push out slightly
            if is_paper:
                 va = 'bottom'
            else:
                 va = 'top'

        elif mode == 'vertical':
            text_rot = 0
            ha = 'center'
            va = 'bottom' if is_paper else 'top'
            # To prevent crossing center for long labels on sides:
            if 90 < deg_norm < 270:
                ha = 'right' # Push left
            else:
                ha = 'left' # Push right

        elif mode == 'inclined':
            # 45 degree slant
            if is_paper:
                if deg_norm < 90: # Top Right
                    text_rot = 30
                    ha = 'left'
                    va = 'bottom'
                elif deg_norm > 90: # Top Left
                    text_rot = -30
                    ha = 'right'
                    va = 'bottom'
                else:
                    text_rot = 0
                    ha = 'center'
                    va = 'bottom'
            else:
                # Bottom
                if deg_norm < 270: # Bottom Left
                     text_rot = 30
                     ha = 'right'
                     va = 'top'
                else: # Bottom Right
                     text_rot = -30
                     ha = 'left'
                     va = 'top'

        else:
            text_rot = 0
            ha = 'center'
            va = 'center'

        return text_rot, ha, va, offset_x, offset_y

    # --- DRAW PAPER NODES (top arc) ---
    if show_models.get('doi_nodes', True):
        doi_color = '#34495E'
        for doi, ang in p_angles.items():
            wdg = Wedge((0, 0), radius,
                        np.degrees(ang['end']), np.degrees(ang['start']),
                        width=node_width, facecolor=doi_color, edgecolor='black',
                        linewidth=0.8, alpha=0.9, zorder=5)
            ax.add_patch(wdg)

            lr = radius + node_width + label_offset
            x = lr * np.cos(ang['mid'])
            y = lr * np.sin(ang['mid'])
            alias = get_alias(doi)

            text_rot, ha, va, _, _ = compute_label_transform(ang['mid'], label_mode, is_paper=True)

            ax.text(x, y, alias, ha=ha, va=va, fontsize=font_size,
                    fontweight='bold', color='#1a1a2e', zorder=6,
                    clip_on=False, rotation=text_rot, rotation_mode='anchor')

    # --- DRAW MATERIAL NODES (bottom arc) ---
    for mat, ang in m_angles.items():
        fill = mat_colors[mat]
        wdg = Wedge((0, 0), radius,
                    np.degrees(ang['start']), np.degrees(ang['end']),
                    width=node_width, facecolor=fill, edgecolor='black',
                    linewidth=0.8, alpha=0.9, zorder=5)
        ax.add_patch(wdg)

        lr = radius + node_width + label_offset
        x = lr * np.cos(ang['mid'])
        y = lr * np.sin(ang['mid'])

        text_rot, ha, va, _, _ = compute_label_transform(ang['mid'], label_mode, is_paper=False)

        txt_color = get_contrast_text_color(fill)
        label = get_mat_label(mat)
        ax.text(x, y, label, ha=ha, va=va, fontsize=font_size - 1,
                fontweight='bold', color=txt_color, zorder=6,
                clip_on=False, rotation=text_rot, rotation_mode='anchor')

    # --- DRAW RIBBONS ---
    max_count = max([c for _, _, c, _, _ in unique_connections]) if unique_connections else 1
    for doi, mat, count, consensus, models in unique_connections:
        if doi not in p_angles or mat not in m_angles:
            continue
        if not show_models.get('doi_nodes', True):
            continue
        visible_models = [m for m in models if show_models.get(m, True)]
        if not visible_models:
            continue

        pa = p_angles[doi]
        ma = m_angles[mat]
        r_rib = radius - node_width / 2
        x1 = r_rib * np.cos(pa['mid'])
        y1 = r_rib * np.sin(pa['mid'])
        x2 = r_rib * np.cos(ma['mid'])
        y2 = r_rib * np.sin(ma['mid'])
        a1, a2 = pa['mid'], ma['mid']

        if abs(a1 - a2) > np.pi:
            if a1 > a2:
                a2 += 2 * np.pi
            else:
                a1 += 2 * np.pi

        cp_r = r_rib * curve_tension
        cp1_a = a1 * 0.70 + a2 * 0.30
        cp2_a = a1 * 0.30 + a2 * 0.70
        cp1x, cp1y = cp_r * np.cos(cp1_a), cp_r * np.sin(cp1_a)
        cp2x, cp2y = cp_r * np.cos(cp2_a), cp_r * np.sin(cp2_a)
        verts = [(x1, y1), (cp1x, cp1y), (cp2x, cp2y), (x2, y2)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        path = Path(verts, codes)

        if consensus >= 2 and show_models.get('consensus', True):
            color = '#F39C12'
            alpha = 0.9
        else:
            color = MODEL_META[visible_models[0]]['color']
            alpha = ribbon_alpha

        lw = min_ribbon_width + (max_ribbon_width - min_ribbon_width) * (count / max_count)
        patch = PathPatch(path, facecolor='none', edgecolor=color,
                          linewidth=lw, alpha=alpha, zorder=2, capstyle='round')
        ax.add_patch(patch)

    # --- AXIS & LEGEND ---
    pad = radius + node_width + label_offset + 0.30
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Unified Cross-Model Extraction Chord", fontsize=18, fontweight='bold', pad=20, color='#1a1a2e')

    # Build legend only for visible items
    legend_elements = []
    if show_models.get('modelB', True):
        legend_elements.append(mpatches.Patch(facecolor=MODEL_META['modelB']['color'], edgecolor='black', label=MODEL_META['modelB']['name']))
    if show_models.get('modelC', True):
        legend_elements.append(mpatches.Patch(facecolor=MODEL_META['modelC']['color'], edgecolor='black', label=MODEL_META['modelC']['name']))
    if show_models.get('modelD', True):
        legend_elements.append(mpatches.Patch(facecolor=MODEL_META['modelD']['color'], edgecolor='black', label=MODEL_META['modelD']['name']))
    if show_models.get('consensus', True):
        legend_elements.append(mpatches.Patch(facecolor='#F39C12', edgecolor='black', label='Consensus (≥2 models)'))
    if show_models.get('doi_nodes', True):
        legend_elements.append(mpatches.Patch(facecolor='#34495E', edgecolor='black', label='Document (DOI)'))

    if legend_elements:
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10,
                  framealpha=0.95, title='Legend', title_fontsize=11,
                  edgecolor='black', fancybox=True, bbox_to_anchor=(1.02, -0.02))
    return fig

# ------------------------------------------------------------------
# SANKEY
# ------------------------------------------------------------------
def hex_to_rgba(hex_color, alpha=0.25):
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'

def draw_sankey(paper_materials, model_name, model_color):
    if not HAS_PLOTLY:
        return None
    papers = list(paper_materials.keys())
    materials = sorted({m for mats in paper_materials.values() for m in mats})
    if not papers or not materials:
        return None
    node_labels = [get_alias(p) for p in papers] + materials
    node_colors = [model_color] * len(papers) + ['#2C3E50'] * len(materials)

    sources, targets, values, link_colors = [], [], [], []
    for i, doi in enumerate(papers):
        for mat, cnt in paper_materials[doi].items():
            sources.append(i)
            targets.append(len(papers) + materials.index(mat))
            values.append(cnt)
            link_colors.append(model_color)

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=node_labels, color=node_colors, pad=18, thickness=22,
                  line=dict(color='black', width=0.6)),
        link=dict(source=sources, target=targets, value=values,
                  color=[hex_to_rgba(c, 0.25) for c in link_colors])
    )])
    fig.update_layout(
        title_text=f"{model_name} — Paper → Material Flow",
        font_size=12, paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# ------------------------------------------------------------------
# HEATMAP
# ------------------------------------------------------------------
def draw_heatmap(all_models_data):
    all_papers = sorted({d for data in all_models_data.values() for d in data})
    all_materials = sorted({m for data in all_models_data.values() for mats in data.values() for m in mats})
    if not all_papers or not all_materials:
        return None
    matrix = np.zeros((len(all_papers), len(all_materials)))
    for data in all_models_data.values():
        for doi, mats in data.items():
            i = all_papers.index(doi)
            for mat in mats:
                j = all_materials.index(mat)
                matrix[i, j] += 1

    width = min(max(14, len(all_materials) * 0.5), 24)
    height = min(max(8, len(all_papers) * 0.4), 18)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=3)

    ax.set_xticks(np.arange(len(all_materials)))
    ax.set_xticklabels([get_mat_label(m) for m in all_materials], rotation=55, ha='right', fontsize=10)
    ax.set_yticks(np.arange(len(all_papers)))
    ax.set_yticklabels([get_alias(d) for d in all_papers], fontsize=10)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Model Agreement Count', fontsize=12, fontweight='bold')
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['0', '1', '2', '3'])

    for i in range(len(all_papers)):
        for j in range(len(all_materials)):
            if matrix[i, j] > 0:
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > 1.5 else "black",
                        fontweight='bold', fontsize=9)

    ax.set_title("Model Agreement Heatmap: Document × Material", fontsize=16, fontweight='bold', pad=12)
    ax.set_xlabel("Material / Concept", fontsize=13, fontweight='bold')
    ax.set_ylabel("Document Alias", fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig

# ------------------------------------------------------------------
# v2.4 CIRCOS — Enhanced Padding + Smart Label Rotation
# ------------------------------------------------------------------
def draw_circos_clean(all_models_data, figsize=(14, 14), inner_r=1.0,
                      track_h=0.5, gap_deg=1.5, fs=9, show_labels=True,
                      max_bar_scale=1.0, cmap_name='viridis',
                      label_padding=0.40, label_buffer=0.15,
                      label_rotation_mode='auto'):
    """
    v2.4 Circos with:
    - Enhanced label padding + buffer to prevent overlap
    - Smart rotation: text always readable (never upside down)
    - Better gap distribution
    - Auto colormap fallback
    """
    figsize = (min(figsize[0], 24), min(figsize[1], 24))

    mat_model = defaultdict(lambda: defaultdict(int))
    for mk, data in all_models_data.items():
        for doi, mats in data.items():
            for mat, cnt in mats.items():
                mat_model[mat][mk] += cnt

    mats_sorted = sorted(mat_model.keys())
    n_mats = len(mats_sorted)
    if n_mats == 0:
        return None

    actual_cmap = cmap_name
    qual_cmaps = ['tab20', 'tab10', 'Set1', 'Set2', 'Set3', 'Paired', 'Dark2', 'Accent', 'Pastel1', 'Pastel2']
    if actual_cmap in qual_cmaps and n_mats > 20:
        actual_cmap = 'turbo'
    elif actual_cmap in qual_cmaps and n_mats > 10:
        actual_cmap = 'tab20'

    cmap = plt.get_cmap(actual_cmap, max(n_mats, 1))
    rim_colors = {mat: mcolors.to_hex(cmap(i)) for i, mat in enumerate(mats_sorted)}

    avail = 2 * np.pi - np.radians(gap_deg)
    step = avail / n_mats if n_mats > 0 else 0

    n_tracks = len(MODEL_META)
    outer_extent = inner_r + n_tracks * track_h + label_padding + label_buffer + 0.15

    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor('white')

    for i, mat in enumerate(mats_sorted):
        ma = i * step + step / 2
        max_cnt = max(mat_model[mat].values()) if mat_model[mat] else 1

        for ti, (mk, meta) in enumerate(MODEL_META.items()):
            cnt = mat_model[mat].get(mk, 0)
            if cnt == 0:
                continue
            bottom = inner_r + ti * track_h
            h = track_h * (0.15 + 0.85 * cnt / max(1, max_cnt)) * max_bar_scale
            ax.bar(ma, h, width=step * 0.85, bottom=bottom,
                   color=meta['color'], alpha=0.92,
                   edgecolor='black', linewidth=0.4, zorder=3)
            if cnt >= 2 and h > track_h * 0.35:
                ax.text(ma, bottom + h / 2, str(cnt),
                        ha='center', va='center', fontsize=max(fs - 2, 7),
                        color='white', fontweight='bold', zorder=4)

        # Outer rim indicator
        rim_bottom = inner_r + n_tracks * track_h + 0.04
        ax.bar(ma, 0.08, width=step * 0.88, bottom=rim_bottom,
               color=rim_colors[mat], alpha=0.9, edgecolor='black', linewidth=0.3, zorder=5)

        # Improved label positioning with enhanced padding
        if show_labels:
            lbl = get_mat_label(mat)
            label_r = inner_r + n_tracks * track_h + label_padding

            rot_deg = np.degrees(ma)
            rot_deg_norm = rot_deg % 360

            # Smart rotation: always keep text readable
            if label_rotation_mode == 'auto':
                if 90 < rot_deg_norm < 270:
                    text_rot = rot_deg_norm - 180
                    ha = 'right' # Push text outward to the left
                    va = 'center'
                else:
                    text_rot = rot_deg_norm
                    ha = 'left' # Push text outward to the right
                    va = 'center'
            else:
                text_rot = 0
                ha = 'center'
                va = 'center'

            ax.text(ma, label_r, lbl,
                    ha=ha, va=va, fontsize=fs - 1,
                    fontweight='bold', color='#2c3e50', zorder=6,
                    rotation=text_rot,
                    rotation_mode='anchor')

    ax.set_ylim(0, outer_extent)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)

    handles = [mpatches.Patch(facecolor=meta['color'], edgecolor='black', label=meta['short'])
               for meta in MODEL_META.values()]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.22, 1.08),
              fontsize=fs, framealpha=0.95, title='Model Track', title_fontsize=fs + 1,
              edgecolor='black', fancybox=True)
    return fig

# ------------------------------------------------------------------
# UMAP FEATURE BUILDER
# ------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner="Building UMAP features...")
def build_umap_features_cached(all_models_data_json):
    import json
    all_models_data = json.loads(all_models_data_json)
    all_mats = sorted({m for data in all_models_data.values() for mats in data.values() for m in mats})
    cat_list = list(TAXONOMY.keys())
    features, labels, categories, presence_list, totals, doi_counts = [], [], [], [], [], []
    for mat in all_mats:
        vec = []
        total = 0
        presence = []
        for mk in MODEL_META.keys():
            cnt = sum(data.get(doi, {}).get(mat, 0) for doi, data in all_models_data[mk].items())
            vec.append(np.log1p(cnt))
            total += cnt
            presence.append(1 if cnt > 0 else 0)
        cat = get_category(mat)
        for c in cat_list:
            vec.append(1 if c == cat else 0)
        n_dois = len({doi for mk, data in all_models_data.items()
                      for doi, mats in data.items() if mat in mats})
        vec.append(np.log1p(n_dois))
        features.append(vec)
        labels.append(mat)
        categories.append(cat)
        presence_list.append(presence)
        totals.append(total)
        doi_counts.append(n_dois)
    return {
        'features': features,
        'labels': labels,
        'categories': categories,
        'presence': presence_list,
        'totals': totals,
        'doi_counts': doi_counts
    }

def draw_umap_3d(all_models_data, n_neighbors=5, min_dist=0.3, metric='euclidean'):
    if not HAS_UMAP or not HAS_SKLEARN:
        return None, "UMAP or scikit-learn not installed."

    import json
    data_json = json.dumps(all_models_data)
    cached = build_umap_features_cached(data_json)
    features = np.array(cached['features'])
    labels = cached['labels']
    categories = cached['categories']
    presence = cached['presence']
    totals = cached['totals']
    doi_counts = cached['doi_counts']

    n_samples = len(labels)
    if n_samples < 3:
        return None, "Need ≥3 materials for UMAP."
    nn = min(n_neighbors, n_samples - 1)
    if nn < 2:
        return None, "Not enough samples for UMAP neighbors."

    emb_list = compute_umap_embedding_cached(tuple(map(tuple, features)), nn, min_dist, metric)
    if emb_list is None:
        return None, "UMAP embedding failed."
    emb = np.array(emb_list)

    df = pd.DataFrame({
        'UMAP1': emb[:, 0], 'UMAP2': emb[:, 1], 'UMAP3': emb[:, 2],
        'Material': [get_mat_label(m) for m in labels],
        'RawMaterial': labels,
        'Category': categories, 'TotalCount': totals,
        'DOIs': doi_counts,
        'InB': [p[0] for p in presence], 'InC': [p[1] for p in presence],
        'InD': [p[2] for p in presence],
    })
    color_map = {cat: CATEGORY_COLORS.get(cat, '#cccccc') for cat in set(categories)}
    fig = px.scatter_3d(df, x='UMAP1', y='UMAP2', z='UMAP3',
                        color='Category', size='TotalCount',
                        hover_data=['RawMaterial', 'TotalCount', 'DOIs', 'InB', 'InC', 'InD'],
                        color_discrete_map=color_map,
                        title='3D UMAP Material Embedding',
                        opacity=0.9)
    fig.update_traces(marker=dict(line=dict(width=1.2, color='black'), sizemin=6))
    fig.update_layout(
        scene=dict(
            xaxis_title='UMAP 1', yaxis_title='UMAP 2', zaxis_title='UMAP 3',
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.3)),
            aspectmode='cube'
        ),
        paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(l=0, r=0, t=50, b=0),
        title=dict(font=dict(size=18, family='serif'))
    )
    return fig, None

# ------------------------------------------------------------------
# ANIMATION DATA BUILDERS
# ------------------------------------------------------------------
def build_animation_data(all_models_data):
    rows = []
    mat_totals = defaultdict(int)
    for mk, meta in MODEL_META.items():
        data = all_models_data.get(mk, {})
        for doi, mats in data.items():
            for mat, cnt in mats.items():
                mat_totals[mat] += cnt

    top_mats = sorted(mat_totals.keys(), key=lambda x: mat_totals[x], reverse=True)[:50]

    for mk, meta in MODEL_META.items():
        data = all_models_data.get(mk, {})
        mat_counts = defaultdict(int)
        mat_cats = {}
        for doi, mats in data.items():
            for mat, cnt in mats.items():
                if mat in top_mats:
                    mat_counts[mat] += cnt
                    mat_cats[mat] = get_category(mat)
        for mat, cnt in mat_counts.items():
            rows.append({
                'Frame': meta['short'],
                'Model': meta['name'],
                'Material': get_mat_label(mat),
                'Count': cnt,
                'Category': mat_cats[mat],
                'LogCount': np.log1p(cnt)
            })
    return pd.DataFrame(rows)

def build_consensus_animation_data(all_models_data):
    all_mats = sorted({m for data in all_models_data.values() for mats in data.values() for m in mats})

    mat_totals = defaultdict(int)
    for mk, data in all_models_data.items():
        for doi, mats in data.items():
            for mat, cnt in mats.items():
                mat_totals[mat] += cnt
    top_mats = sorted(mat_totals.keys(), key=lambda x: mat_totals[x], reverse=True)[:50]
    all_mats = [m for m in all_mats if m in top_mats]

    mat_agreement = {}
    mat_maxcnt = {}
    for mat in all_mats:
        models_found = []
        max_cnt = 0
        for mk in MODEL_META.keys():
            cnt = sum(data.get(doi, {}).get(mat, 0) for doi, data in all_models_data[mk].items())
            if cnt > 0:
                models_found.append(mk)
            max_cnt = max(max_cnt, cnt)
        mat_agreement[mat] = len(models_found)
        mat_maxcnt[mat] = max_cnt

    rows = []
    for level in [1, 2, 3]:
        for mat, ag in mat_agreement.items():
            if ag >= level:
                rows.append({
                    'Frame': f'≥{level} Model{"s" if level > 1 else ""}',
                    'Material': get_mat_label(mat),
                    'Category': get_category(mat),
                    'MaxCount': mat_maxcnt[mat],
                    'Agreement': ag
                })
    return pd.DataFrame(rows)

# ------------------------------------------------------------------
# LOAD & INITIALIZE
# ------------------------------------------------------------------
raw_data = load_data_from_disk_or_default()
all_dois = sorted({d for data in raw_data.values() for d in data})
all_materials = sorted({m for data in raw_data.values() for mats in data.values() for m in mats})
init_aliases(all_dois, all_materials)

if 'validation_cache' not in st.session_state:
    st.session_state.validation_cache = {}

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📁 Data Source")
    upload_mode = st.radio("Input mode", ["Auto-detect / Embedded defaults", "Manual upload"], index=0)
    if upload_mode == "Manual upload":
        fB = st.file_uploader("Model B CSV", type="csv", key="fb")
        fC = st.file_uploader("Model C CSV", type="csv", key="fc")
        fD = st.file_uploader("Model D CSV", type="csv", key="fd")
        if fB:
            raw_data['modelB'] = parse_alloys_csv_cached(fB.getvalue().decode('utf-8'), 'modelB')
            st.cache_data.clear()
        if fC:
            raw_data['modelC'] = parse_alloys_csv_cached(fC.getvalue().decode('utf-8'), 'modelC')
            st.cache_data.clear()
        if fD:
            raw_data['modelD'] = parse_alloys_csv_cached(fD.getvalue().decode('utf-8'), 'modelD')
            st.cache_data.clear()
        all_dois = sorted({d for data in raw_data.values() for d in data})
        all_materials = sorted({m for data in raw_data.values() for mats in data.values() for m in mats})
        init_aliases(all_dois, all_materials)

    st.markdown("---")
    st.markdown("## 🧬 Material Validator")
    st.markdown("<small>Regex + Periodic-Table hybrid. Toggle to filter non-materials.</small>", unsafe_allow_html=True)
    enable_validator = st.checkbox("Enable validation", value=True, key="en_val")
    materials_only = st.checkbox("Show ONLY validated materials", value=False, key="mat_only")
    st.markdown("<small>When enabled, non-material nodes/edges are erased, dynamically altering diagram morphology.</small>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("## 🏷️ Alias Editors")

    with st.expander("✏️ DOI Aliases", expanded=False):
        alias_search = st.text_input("Filter DOIs", "", key="alias_filter")
        filtered_dois = [d for d in all_dois if not alias_search or alias_search.lower() in d.lower()]
        if filtered_dois:
            doi_df = pd.DataFrame({
                "DOI": filtered_dois,
                "Alias": [st.session_state.doi_aliases.get(d, f"[{chr(65 + all_dois.index(d))}]" if all_dois.index(d) < 26 else f"[{all_dois.index(d) + 1}]") for d in filtered_dois]
            })
            edited_doi_df = st.data_editor(doi_df, hide_index=True, use_container_width=True, key="doi_editor")
            if st.button("💾 Save DOI Alias Changes", key="save_doi_aliases", type="primary"):
                for _, row in edited_doi_df.iterrows():
                    st.session_state.doi_aliases[row["DOI"]] = row["Alias"]
                st.success("✅ DOI Alias changes saved!")
                st.rerun()

    with st.expander("✏️ Material Labels", expanded=False):
        mat_search = st.text_input("Filter materials", "", key="mat_filter")
        filtered_mats = [m for m in all_materials if not mat_search or mat_search.lower() in m.lower()]
        if filtered_mats:
            mat_df = pd.DataFrame({
                "Raw Material": filtered_mats,
                "Display Label": [st.session_state.material_aliases.get(m, prettify_material(m)) for m in filtered_mats]
            })
            edited_mat_df = st.data_editor(mat_df, hide_index=True, use_container_width=True, key="mat_editor")
            if st.button("💾 Save Material Label Changes", key="save_mat_aliases", type="primary"):
                for _, row in edited_mat_df.iterrows():
                    st.session_state.material_aliases[row["Raw Material"]] = row["Display Label"]
                st.success("✅ Material Label changes saved!")
                st.rerun()

    st.markdown("---")
    st.markdown("## ⚙️ Diagram Toggles")
    show_chord = st.checkbox("Unified Chord", value=True)
    show_sankey = st.checkbox("Sankey Flow", value=True)
    show_network = st.checkbox("Network Graph", value=HAS_PYVIS)
    show_heatmap = st.checkbox("Agreement Heatmap", value=True)
    show_circos = st.checkbox("Circos Radial", value=True)
    show_umap = st.checkbox("3D UMAP", value=HAS_UMAP)
    show_animation = st.checkbox("Animated Transitions", value=HAS_PLOTLY)
    show_validation = st.checkbox("Validation Table", value=True)
    show_legend_panel = st.checkbox("DOI Legend Panel", value=True)

    st.markdown("---")
    st.markdown("## 🎨 Unified Chord Style")
    with st.expander("Controls", expanded=False):
        chord_figsize = st.slider("Figure size", 8, 24, 16, key="chord_fs")
        chord_radius = st.slider("Radius", 0.6, 2.8, 1.0, 0.05, key="chord_r")
        chord_node_w = st.slider("Node arc width", 0.02, 0.32, 0.08, 0.01, key="chord_nw")
        chord_rib_a = st.slider("Ribbon opacity", 0.1, 1.0, 0.55, 0.05, key="chord_ra")
        chord_font = st.slider("Font size", 5, 36, 10, key="chord_fnt")
        chord_min_lw = st.slider("Min ribbon width", 0.2, 6.0, 0.8, 0.1, key="chord_mlw")
        chord_max_lw = st.slider("Max ribbon width", 1.0, 20.0, 6.0, 0.2, key="chord_Mlw")

        # v2.4: Label alignment mode
        chord_label_mode = st.selectbox(
            "📐 Label alignment",
            ["radial", "parallel", "vertical", "inclined"],
            index=0,
            key="chord_lbl_mode",
            help="Radial = points outward. Parallel = follows arc tangent. Vertical = always upright. Inclined = 30° slant."
        )

        chord_gap = st.slider("Arc gap padding", 0.0, 0.08, 0.015, 0.005, key="chord_gap",
                              help="Gap between adjacent arc segments.")
        chord_lbl_off = st.slider("Label offset", 0.10, 0.60, 0.20, 0.01, key="chord_lo",
                                  help="Radial distance of labels from arc edge. Increase if labels overlap.")
        chord_tension = st.slider("Curve tension", 0.15, 0.60, 0.35, 0.05, key="chord_ten",
                                  help="Bezier control-point radius factor.")

        # v2.4: Model / Consensus / DOI legend toggles
        st.markdown("<small><b>Chord Legend & Visibility</b></small>", unsafe_allow_html=True)
        show_chord_modelB = st.checkbox("Show Model B", value=True, key="chord_show_b")
        show_chord_modelC = st.checkbox("Show Model C", value=True, key="chord_show_c")
        show_chord_modelD = st.checkbox("Show Model D", value=True, key="chord_show_d")
        show_chord_consensus = st.checkbox("Show Consensus", value=True, key="chord_show_cons")
        show_chord_doi = st.checkbox("Show DOI Nodes", value=True, key="chord_show_doi")

        # Category Filter
        all_cats = sorted(list(TAXONOMY.keys()) + ['Other'])
        selected_cats = st.multiselect(
            "📂 Toggle Categories",
            options=all_cats,
            default=all_cats,
            key="chord_cat_filter",
            help="Deselecting erases nodes and links."
        )

        # 50+ Colormap
        chord_cmap = st.selectbox(
            "🎨 Material Colormap", 
            ALL_COLORMAPS,
            index=ALL_COLORMAPS.index('turbo') if 'turbo' in ALL_COLORMAPS else 0,
            key="chord_cmap",
            help="Continuous maps support unlimited colors. Qualitative maps auto-fallback to turbo if N>20."
        )

    st.markdown("## 🧬 Circos Style")
    with st.expander("Controls", expanded=False):
        circos_size = st.slider("Figure size", 8, 24, 14, key="circos_fs")
        circos_inner = st.slider("Inner radius", 0.5, 4.0, 1.0, key="circos_ir")
        circos_track = st.slider("Track height", 0.2, 3.0, 0.5, key="circos_th")
        circos_gap = st.slider("Gap (°)", 0.0, 20.0, 1.5, key="circos_gap")
        circos_fs = st.slider("Font size", 5, 32, 9, key="circos_font")
        circos_scale = st.slider("Bar scale", 0.5, 4.0, 1.0, key="circos_scale")

        # v2.4: Enhanced padding controls
        circos_lbl_pad = st.slider("Label padding", 0.15, 1.0, 0.40, 0.05, key="circos_lp",
                                   help="Radial distance from outer rim to label baseline.")
        circos_lbl_buf = st.slider("Label buffer", 0.05, 0.50, 0.15, 0.05, key="circos_lb",
                                   help="Extra clearance added to figure bounds to prevent text clipping.")
        circos_rot_mode = st.selectbox(
            "Label rotation",
            ["auto", "horizontal"],
            index=0,
            key="circos_rot",
            help="Auto = radial rotation (readable from outside). Horizontal = always upright."
        )

        # 50+ Colormap
        circos_cmap = st.selectbox(
            "🎨 Material Colormap", 
            ALL_COLORMAPS,
            index=ALL_COLORMAPS.index('viridis') if 'viridis' in ALL_COLORMAPS else 0,
            key="circos_cmap",
            help="Outer rim colormap."
        )

    st.markdown("## 🔮 UMAP Style")
    with st.expander("Controls", expanded=False):
        umap_neighbors = st.slider("Neighbors", 2, 30, 5, step=2, key="umap_nn")
        umap_dist = st.slider("Min distance", 0.05, 1.8, 0.3, step=0.05, key="umap_md")
        umap_metric = st.selectbox("Metric", ['euclidean', 'cosine', 'correlation'], index=0, key="umap_met")

    st.markdown("## 🎬 Animation Style")
    with st.expander("Controls", expanded=False):
        anim_type = st.selectbox("Mode", ["Model Sweep", "Consensus Buildup"], index=0, key="anim_type")
        anim_speed = st.slider("Frame duration (ms)", 400, 6000, 1000, key="anim_spd")

    st.markdown("## 📊 Global Export")
    download_dpi = st.slider("Figure DPI", 150, 600, 300, 50)

    st.markdown("---")
    with st.expander("🩺 System Diagnostics", expanded=False):
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        mem_warn = mem_mb > 1500

        st.markdown(f"""
        <div style="font-family:monospace; font-size:0.85rem;">
            <span class="diag-metric">🧠 Memory: <span class="{'diag-warn' if mem_warn else 'diag-ok'}">{mem_mb:.1f} MB</span></span>
            <span class="diag-metric">🔑 Session Keys: {len(st.session_state)}</span>
            <span class="diag-metric">💾 Cache: Auto-managed (TTL + max_entries)</span>
        </div>
        """, unsafe_allow_html=True)

        if mem_warn:
            st.warning("⚠️ Memory usage high! Consider clearing cache or reducing dataset size.")

        if st.button("🗑️ Force GC + Clear Caches", key="diag_clear"):
            plt.close('all')
            gc.collect()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Caches cleared! Rerunning...")
            st.rerun()

# ------------------------------------------------------------------
# FILTER DATA
# ------------------------------------------------------------------
import json
raw_data_plain = {k: dict(v) for k, v in raw_data.items()}
display_data = filter_data_by_validation_cached(
    tuple((k, tuple(sorted(v.items()))) for k, v in raw_data_plain.items()),
    materials_only,
    enable_validator
)

# ------------------------------------------------------------------
# MAIN HEADER
# ------------------------------------------------------------------
st.markdown('<div class="main-header">AlloyExtraction Nexus v2.4</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Memory-Hardened • 50+ Colormaps • Configurable Label Modes • Enhanced Padding</div>', unsafe_allow_html=True)

cols = st.columns(len(MODEL_META))
for i, (key, meta) in enumerate(MODEL_META.items()):
    data = display_data.get(key, {})
    n_doi = len(data)
    n_mat = len({m for mats in data.values() for m in mats})
    n_conn = sum(len(mats) for mats in data.values())
    with cols[i]:
        st.markdown(f'<div class="metric-card highlight-{["red","green","blue"][i]}">'
            f'<strong>{meta["name"]}</strong><br>'
            f'<span style="font-size:1.3rem">{n_doi}</span> docs &nbsp;|&nbsp;'
            f'<span style="font-size:1.3rem">{n_mat}</span> unique concepts &nbsp;|&nbsp;'
            f'<span style="font-size:1.3rem">{n_conn}</span> extractions'
            f'</div>', unsafe_allow_html=True)

st.markdown("---")

# ------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------
tab_labels = []
if show_chord:      tab_labels.append("🔵 Unified Chord")
if show_sankey:     tab_labels.append("🌊 Sankey")
if show_network:    tab_labels.append("🕸️ Network")
if show_heatmap:    tab_labels.append("🔥 Heatmap")
if show_circos:     tab_labels.append("🧬 Circos")
if show_umap:       tab_labels.append("🔮 3D UMAP")
if show_animation:  tab_labels.append("🎬 Animation")
if show_validation: tab_labels.append("🧪 Validation")
if show_legend_panel: tab_labels.append("📋 Legend")

if not tab_labels:
    st.warning("Enable at least one diagram in the sidebar.")
    st.stop()

tabs = st.tabs(tab_labels)
t_idx = 0

# ------------------------------------------------------------------
# TAB 1: UNIFIED CHORD (v2.4 — Label Modes + Legend Toggles)
# ------------------------------------------------------------------
if show_chord:
    with tabs[t_idx]:
        st.markdown("### Unified Multi-Model Chord Diagram")
        st.markdown("<div class='caption'>v2.4: 4 label modes (radial/parallel/vertical/inclined) + per-model/consensus/DOI visibility toggles.</div>", unsafe_allow_html=True)

        # Category Filter
        chord_data = {}
        for mk, data in display_data.items():
            fdata = {}
            for doi, mats in data.items():
                fmats = {m: c for m, c in mats.items() if get_category(m) in selected_cats}
                if fmats:
                    fdata[doi] = fmats
            if fdata:
                chord_data[mk] = fdata

        # Build show_models dict from toggles
        show_models_dict = {
            'modelB': show_chord_modelB,
            'modelC': show_chord_modelC,
            'modelD': show_chord_modelD,
            'consensus': show_chord_consensus,
            'doi_nodes': show_chord_doi,
        }

        fig = draw_unified_chord(
            chord_data, figsize=(chord_figsize, chord_figsize), radius=chord_radius,
            node_width=chord_node_w, ribbon_alpha=chord_rib_a, font_size=chord_font,
            min_ribbon_width=chord_min_lw, max_ribbon_width=chord_max_lw,
            cmap_name=chord_cmap,
            gap_padding=chord_gap,
            label_offset=chord_lbl_off,
            curve_tension=chord_tension,
            label_mode=chord_label_mode,
            show_models=show_models_dict
        )
        render_matplotlib_figure(fig, "unified_chord.png", download_dpi)
        del fig
        gc.collect()
    t_idx += 1

# ------------------------------------------------------------------
# TAB 2: SANKEY
# ------------------------------------------------------------------
if show_sankey:
    with tabs[t_idx]:
        st.markdown("### Hierarchical Sankey Flow")
        st.markdown("<div class='caption'>Per-model paper-to-material flow. Useful for spotting coverage gaps.</div>", unsafe_allow_html=True)
        if not HAS_PLOTLY:
            st.warning("Plotly not installed. `pip install plotly` to enable.")
        else:
            for key, meta in MODEL_META.items():
                if key not in display_data:
                    continue
                fig = draw_sankey(display_data[key], meta['name'], meta['color'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    del fig
                    gc.collect()
                else:
                    st.info(f"No Sankey data for {meta['short']}.")
    t_idx += 1

# ------------------------------------------------------------------
# TAB 3: NETWORK (CACHED)
# ------------------------------------------------------------------
if show_network:
    with tabs[t_idx]:
        st.markdown("### Interactive Bipartite Network")
        st.markdown("<div class='caption'>Force-directed physics. ▭ = document; ● = material. Edge color = model. Drag, zoom, hover.</div>", unsafe_allow_html=True)
        if not HAS_PYVIS:
            st.warning("PyVis not installed. `pip install pyvis` to enable.")
        else:
            display_data_json = json.dumps(display_data)
            html, legend_html = get_network_html_cached(display_data_json)
            if html:
                import streamlit.components.v1 as components
                components.html(html, height=740, scrolling=False)
                st.markdown(legend_html, unsafe_allow_html=True)
                st.download_button("⬇️ Download Network HTML", html.encode('utf-8'),
                                   "alloy_network.html", "text/html")
                del html
                gc.collect()
    t_idx += 1

# ------------------------------------------------------------------
# TAB 4: HEATMAP
# ------------------------------------------------------------------
if show_heatmap:
    with tabs[t_idx]:
        st.markdown("### Model Agreement Heatmap")
        st.markdown("<div class='caption'>Cell value = number of models extracting that material for that DOI. Darker = higher agreement.</div>", unsafe_allow_html=True)
        fig_h = draw_heatmap(display_data)
        render_matplotlib_figure(fig_h, "agreement_heatmap.png", download_dpi)
        del fig_h
        gc.collect()
    t_idx += 1

# ------------------------------------------------------------------
# TAB 5: CIRCOS (v2.4 — Enhanced Padding)
# ------------------------------------------------------------------
if show_circos:
    with tabs[t_idx]:
        st.markdown("### Circos-Style Radial Histogram")
        st.markdown("<div class='caption'>v2.4: enhanced label padding + buffer prevents text-image overlap. Smart rotation keeps labels readable.</div>", unsafe_allow_html=True)
        fig_circ = draw_circos_clean(
            display_data, figsize=(circos_size, circos_size), inner_r=circos_inner,
            track_h=circos_track, gap_deg=circos_gap, fs=circos_fs,
            max_bar_scale=circos_scale, cmap_name=circos_cmap,
            label_padding=circos_lbl_pad,
            label_buffer=circos_lbl_buf,
            label_rotation_mode=circos_rot_mode
        )
        render_matplotlib_figure(fig_circ, "circos_radial.png", download_dpi)
        del fig_circ
        gc.collect()
    t_idx += 1

# ------------------------------------------------------------------
# TAB 6: 3D UMAP (CACHED)
# ------------------------------------------------------------------
if show_umap:
    with tabs[t_idx]:
        st.markdown("### 3D UMAP Clustering of Materials")
        st.markdown("<div class='caption'>UMAP on occurrence vectors. Color = taxonomy; size = total count. Black edges for visibility.</div>", unsafe_allow_html=True)
        if not HAS_UMAP or not HAS_SKLEARN:
            st.warning("UMAP or scikit-learn missing. `pip install umap-learn scikit-learn` to enable.")
        else:
            fig_umap, err = draw_umap_3d(display_data, n_neighbors=umap_neighbors,
                                         min_dist=umap_dist, metric=umap_metric)
            if err:
                st.error(err)
            elif fig_umap:
                st.plotly_chart(fig_umap, use_container_width=True)
                html_str = fig_umap.to_html(include_plotlyjs='cdn')
                st.download_button("⬇️ Download UMAP HTML", html_str.encode('utf-8'),
                                   "umap_3d.html", "text/html")
                del fig_umap, html_str
                gc.collect()
    t_idx += 1

# ------------------------------------------------------------------
# TAB 7: ANIMATION
# ------------------------------------------------------------------
if show_animation:
    with tabs[t_idx]:
        st.markdown("### Animated Transitions")
        st.markdown("<div class='caption'>Press play to sweep across models or watch consensus build up. (Limited to Top 50 materials for performance).</div>", unsafe_allow_html=True)
        if not HAS_PLOTLY:
            st.warning("Plotly not installed. `pip install plotly` to enable.")
        else:
            cat_color_map = {k: CATEGORY_COLORS.get(k, '#cccccc') for k in list(TAXONOMY.keys()) + ['Other']}
            if anim_type == "Model Sweep":
                df_anim = build_animation_data(display_data)
                if df_anim.empty:
                    st.info("No animation data.")
                else:
                    ymax = max(1, df_anim['Count'].max() * 1.15)
                    anim_key = f"anim_model_{hash(str(display_data)) % 10000}"
                    fig_anim = px.scatter(df_anim, x='Material', y='Count', animation_frame='Frame',
                                          color='Category', size='Count',
                                          color_discrete_map=cat_color_map,
                                          range_y=[0, ymax],
                                          title='Model Extraction Landscape',
                                          height=650)
                    fig_anim.update_traces(marker=dict(line=dict(width=1, color='black')))
                    if fig_anim.layout.updatemenus:
                        fig_anim.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = anim_speed
                        fig_anim.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = anim_speed // 2
                    st.plotly_chart(fig_anim, use_container_width=True, key=anim_key)
                    del df_anim, fig_anim
                    gc.collect()
            else:
                df_cons = build_consensus_animation_data(display_data)
                if df_cons.empty:
                    st.info("No consensus animation data.")
                else:
                    ymax = max(1, df_cons['MaxCount'].max() * 1.15)
                    anim_key = f"anim_consensus_{hash(str(display_data)) % 10000}"
                    fig_cons = px.scatter(df_cons, x='Material', y='MaxCount', animation_frame='Frame',
                                          color='Category', size='MaxCount',
                                          color_discrete_map=cat_color_map,
                                          range_y=[0, ymax],
                                          title='Consensus Buildup',
                                          height=650)
                    fig_cons.update_traces(marker=dict(line=dict(width=1, color='black')))
                    if fig_cons.layout.updatemenus:
                        fig_cons.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = anim_speed
                        fig_cons.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = anim_speed // 2
                    st.plotly_chart(fig_cons, use_container_width=True, key=anim_key)
                    del df_cons, fig_cons
                    gc.collect()
    t_idx += 1

# ------------------------------------------------------------------
# TAB 8: VALIDATION TABLE
# ------------------------------------------------------------------
if show_validation:
    with tabs[t_idx]:
        st.markdown("### Hybrid Material Validation Results")
        st.markdown("<div class='caption'>Regex + periodic-table classification. Toggle 'Show ONLY validated materials' to filter diagrams.</div>", unsafe_allow_html=True)
        val_rows = []
        for mat in sorted(all_materials):
            if mat not in st.session_state.validation_cache:
                st.session_state.validation_cache[mat] = validate_material(mat, use_llm=False)
            res = st.session_state.validation_cache[mat]
            val_rows.append({
                'Term': mat,
                'Display Label': get_mat_label(mat),
                'Is Material': '✅ Yes' if res['is_material'] else '❌ No',
                'Method': res['method'],
                'Reason': res['reason']
            })
        df_val = pd.DataFrame(val_rows)
        st.dataframe(df_val, use_container_width=True, height=500)
        csv_val = df_val.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Validation CSV", csv_val, "material_validation.csv", "text/csv")
        del df_val, val_rows, csv_val
        gc.collect()
    t_idx += 1

# ------------------------------------------------------------------
# TAB 9: LEGEND
# ------------------------------------------------------------------
if show_legend_panel:
    with tabs[t_idx]:
        st.markdown("### DOI Alias Legend")
        st.markdown("<div class='caption'>Each alias maps to exactly one DOI, globally across all models. Export/import JSON for reproducibility.</div>", unsafe_allow_html=True)
        legend_df = pd.DataFrame([
            {"Alias": get_alias(doi), "DOI": doi, "Models Found In": ", ".join([
                MODEL_META[k]['short'] for k in display_data if doi in display_data[k]
            ])}
            for doi in all_dois
        ]).drop_duplicates(subset=['Alias'])
        st.dataframe(legend_df, use_container_width=True, height=500)

        alias_json = json.dumps(st.session_state.doi_aliases, indent=2)
        st.download_button("⬇️ Export Aliases JSON", alias_json.encode('utf-8'),
                           "doi_aliases.json", "application/json")

        uploaded_aliases = st.file_uploader("Import Aliases JSON", type="json", key="alias_import")
        if uploaded_aliases is not None:
            imported = json.load(uploaded_aliases)
            st.session_state.doi_aliases.update(imported)
            st.success("Aliases imported! Click Rerun to refresh all diagrams.")
            st.button("🔄 Rerun", on_click=lambda: st.rerun())
        del legend_df, alias_json
        gc.collect()
    t_idx += 1

# ------------------------------------------------------------------
# FINAL MEMORY CLEANUP
# ------------------------------------------------------------------
plt.close('all')
gc.collect()

# ------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#666; font-size:0.88rem; padding:1rem 0; font-family:serif;">
    <strong>v2.4 Upgrades:</strong> 
    ✅ 4 Label Modes (radial / parallel / vertical / inclined) with <code>rotation_mode='anchor'</code> | 
    ✅ Chord Legend Toggles (per-model + consensus + DOI nodes, all default ON) | 
    ✅ Enhanced Circos Padding (label_padding + label_buffer prevents overlap) | 
    ✅ 50+ Colormaps (turbo, jet, inferno, rainbow, plasma, magma, cividis, etc.)<br><br>
    <strong>Memory Hardening:</strong> All expensive operations remain wrapped in <code>@st.cache_data</code> or 
    <code>@st.cache_resource(max_entries=3)</code>. Matplotlib figures are force-closed via 
    <code>plt.close()</code> + <code>gc.collect()</code> after every render.
</div>
""", unsafe_allow_html=True)
