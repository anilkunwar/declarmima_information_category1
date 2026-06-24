"""
AlloyExtraction Nexus -- Model A (Falcon 10B) Standalone Deep-Dive v1.0
Single-model analysis with 10,000-character context limit awareness
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
import os, json, re, gc
from collections import defaultdict, Counter
from itertools import combinations

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

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

st.set_page_config(
    page_title="Falcon 10B Alloy Extraction -- Standalone Analysis",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.6rem; font-weight: 800; color: #1a1a2e; margin-bottom: 0.3rem; letter-spacing: -0.5px; }
    .sub-header { font-size: 1.2rem; color: #5a5a7a; margin-bottom: 2rem; font-weight: 400; }
    .metric-card { background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); border-left: 4px solid #f39c12; padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
    .metric-card-blue { border-left-color: #2980b9; }
    .metric-card-green { border-left-color: #27ae60; }
    .metric-card-red { border-left-color: #c0392b; }
    .caption { font-size: 0.95rem; color: #555; font-style: italic; margin-bottom: 1rem; }
    .control-section { background: linear-gradient(180deg, #f5f7fa 0%, #eef1f5 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border: 1px solid #dde2e8; }
    .section-title { font-size: 1.15rem; font-weight: 700; color: #2c3e50; margin-bottom: 0.8rem; border-bottom: 2px solid #f39c12; padding-bottom: 0.3rem; }
    .context-banner { background: linear-gradient(135deg, #fef9e7 0%, #fdebd0 100%); border: 2px solid #f39c12; border-left: 5px solid #f39c12; padding: 1.2rem; border-radius: 8px; margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(243,156,18,0.15); }
    .insight-box { background: linear-gradient(135deg, #ebf5fb 0%, #d4e6f1 100%); border-left: 4px solid #2980b9; padding: 1rem; border-radius: 6px; margin: 0.5rem 0; }
    .warning-box { background: linear-gradient(135deg, #fdedec 0%, #f5b7b1 100%); border-left: 4px solid #c0392b; padding: 1rem; border-radius: 6px; margin: 0.5rem 0; }
    .success-box { background: linear-gradient(135deg, #e9f7ef 0%, #a9dfbf 100%); border-left: 4px solid #27ae60; padding: 1rem; border-radius: 6px; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

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
    'modelA': {
        'name': 'Model A (Falcon 10B)',
        'short': 'Falcon 10B',
        'color': '#F39C12',
        'context_limit': 10000,
        'context_note': '10k chars',
        'is_limited': True
    }
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

CATEGORY_BAR_COLORS = {
    'Process Parameters':    '#E74C3C',
    'Mechanical Properties': '#2980B9',
    'Microstructural Features':'#27AE60',
    'Thermal / Fluid':       '#F39C12',
    'Computational / Method':'#8E44AD',
    'Alloy / Composition':   '#1ABC9C',
    'Uncertainty / Unknown': '#7F8C8D',
    'Other':                 '#BDC3C7',
}

SYNONYM_MAP = {
    'al-si-mg-zr alloy': 'alsimgzr',
    'tib2/al-si-mg-zr alloy': 'tib2_alsimgzr',
    'ti-au': 'ti-au',
    'au-ti': 'ti-au',
    'aln ceramics': 'aln',
    'cu6sn5 imc': 'cu6sn5',
    'ti6al4v matrix': 'ti6al4v',
    'ti6al4v alloy': 'ti6al4v',
    'ti2cu imc': 'ti2cu',
    'ti2cu imc/ti6al4v matrix': 'ti2cu_ti6al4v',
    'cu-mn': 'cu-mn',
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

COLORMAP_CATALOG = {
    "Perceptually Uniform": ["viridis", "plasma", "inferno", "magma", "cividis"],
    "Sequential": ["turbo", "jet", "rainbow", "gist_rainbow", "gist_ncar", "nipy_spectral",
        "gnuplot", "gnuplot2", "CMRmap", "cubehelix", "brg", "afmhot",
        "hot", "cool", "spring", "summer", "autumn", "winter", "copper",
        "bone", "pink", "terrain", "ocean"],
    "Diverging": ["coolwarm", "bwr", "seismic", "RdYlBu", "RdYlGn", "Spectral"],
    "Cyclic": ["twilight", "twilight_shifted", "hsv", "flag", "prism"],
    "Qualitative (<=20)": ["tab10", "tab20", "Set1", "Set2", "Set3", "Paired", "Dark2", "Accent", "Pastel1", "Pastel2"]
}
ALL_COLORMAPS = []
for group, cmaps in COLORMAP_CATALOG.items():
    ALL_COLORMAPS.extend(cmaps)
ALL_COLORMAPS = sorted(list(set(ALL_COLORMAPS)))

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
        'strontium_titanate': 'SrTiO3',
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
        'tib2_alsimgzr': 'TiB2/Al-Si-Mg-Zr',
        'ti2cu_ti6al4v': 'Ti2Cu / Ti-6Al-4V',
        'b0.3er0.5al0.2n': 'B0.3Er0.5Al0.2N',
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

def validate_material(term: str) -> dict:
    return regex_validate_material(term)

DEFAULT_DATA_A = """DOI,Materials_Alloys
10.1007/s40195-025-01825-1,"alsimgzr, lpbf"
10.1016/j.apenergy.2024.122901,"b0.3er0.5al0.2n, aln"
10.1016/j.commatsci.2025.113875,"ti-au, alsimgzr"
10.1016/j.engappai.2024.107902,"heas_mpeas"
10.1016/j.ijsolstr.2024.112894,"al, al2cu"
10.1016/j.jallcom.2024.174876,"nt_cu, cu6sn5"
10.1016/j.jestch.2023.101413,"au-ti"
10.1016/j.measurement.2024.114123,"lewis_number"
10.1016/j.msea.2025.148865,"yield_strength, alsimgzr"
10.1016/j.scriptamat.2024.116027,"ti-cr"
10.1016/j.surfin.2023.102728,"cu6sn5"
10.1080/17452759.2024.2416518,"alsimgzr, metallic_glass"
10.1109/ICEPT56209.2022.9873310,"cu6sn5"
10.3390/met12060964,"heas_mpeas"
10.3390/met12111884,"strontium_titanate, steel_sheet"
10.26434/chemrxiv-2025-sk6h5,"alsimgzr, sdss_2507, laser_power, fe"""

@st.cache_data(ttl=600, max_entries=3, show_spinner="Parsing CSV data...")
def parse_alloys_csv_cached(filepath_or_content):
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
    path = os.path.join(ALLOYS_DIR, "alloys_materials_in_documents_searched_via_LLM_modelA.csv")
    if os.path.exists(path):
        return parse_alloys_csv_cached(path)
    else:
        return parse_alloys_csv_cached(DEFAULT_DATA_A)

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

@st.cache_data(ttl=300, show_spinner="Validating materials...")
def filter_data_by_validation_cached(data_tuple, materials_only):
    data = {k: dict(v) for k, v in data_tuple}
    if not materials_only:
        return data
    filtered = {}
    for doi, mats in data.items():
        fmats = {}
        for mat, cnt in mats.items():
            res = validate_material(mat)
            if res['is_material']:
                fmats[mat] = cnt
        if fmats:
            filtered[doi] = fmats
    return filtered

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

def draw_single_chord(data, figsize=(14, 14), radius=1.0,
                      node_width=0.08, ribbon_alpha=0.55, font_size=10,
                      paper_span=np.pi * 0.80, material_span=np.pi * 0.80,
                      min_ribbon_width=0.8, max_ribbon_width=6.0,
                      cmap_name='turbo', gap_padding=0.015,
                      label_offset=0.20, curve_tension=0.35,
                      label_mode='radial'):

    all_dois = sorted(data.keys())
    all_mats = sorted({m for mats in data.values() for m in mats})
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
    for doi, mats in data.items():
        for mat, cnt in mats.items():
            doi_counts[doi] += cnt
            mat_counts[mat] += cnt
            connections.append((doi, mat, cnt))

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

        deg = np.degrees(ang['mid'])
        deg_norm = deg % 360
        if label_mode == 'radial':
            if 90 < deg_norm < 270:
                text_rot = deg_norm - 180
                ha = 'right'
            else:
                text_rot = deg_norm
                ha = 'left'
            va = 'bottom'
            if 80 < deg_norm < 100: ha = 'center'
        else:
            text_rot = 0
            ha = 'center'
            va = 'bottom'

        ax.text(x, y, alias, ha=ha, va=va, fontsize=font_size,
                fontweight='bold', color='#1a1a2e', zorder=6,
                clip_on=False, rotation=text_rot, rotation_mode='anchor')

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

        deg = np.degrees(ang['mid'])
        deg_norm = deg % 360
        if label_mode == 'radial':
            if 90 < deg_norm < 270:
                text_rot = deg_norm - 180
                ha = 'right'
            else:
                text_rot = deg_norm
                ha = 'left'
            va = 'top'
            if 260 < deg_norm < 280: ha = 'center'
        else:
            text_rot = 0
            ha = 'center'
            va = 'top'

        label = get_mat_label(mat)
        ax.text(x, y, label, ha=ha, va=va, fontsize=font_size - 1,
                fontweight='bold', color='#1a1a2e', zorder=6,
                clip_on=False, rotation=text_rot, rotation_mode='anchor')

    max_count = max([c for _, _, c in connections]) if connections else 1
    for doi, mat, count in connections:
        if doi not in p_angles or mat not in m_angles:
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
            if a1 > a2: a2 += 2 * np.pi
            else: a1 += 2 * np.pi
        cp_r = r_rib * curve_tension
        cp1_a = a1 * 0.70 + a2 * 0.30
        cp2_a = a1 * 0.30 + a2 * 0.70
        cp1x, cp1y = cp_r * np.cos(cp1_a), cp_r * np.sin(cp1_a)
        cp2x, cp2y = cp_r * np.cos(cp2_a), cp_r * np.sin(cp2_a)
        verts = [(x1, y1), (cp1x, cp1y), (cp2x, cp2y), (x2, y2)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        path = Path(verts, codes)
        color = MODEL_META['modelA']['color']
        lw = min_ribbon_width + (max_ribbon_width - min_ribbon_width) * (count / max_count)
        patch = PathPatch(path, facecolor='none', edgecolor=color,
                          linewidth=lw, alpha=ribbon_alpha, zorder=2, capstyle='round')
        ax.add_patch(patch)

    pad = radius + node_width + label_offset + 0.30
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Falcon 10B -- Paper-Material Extraction Chord", fontsize=18, fontweight='bold', pad=20, color='#1a1a2e')

    legend_elements = [
        mpatches.Patch(facecolor=MODEL_META['modelA']['color'], edgecolor='black', label='Falcon 10B (10k chars)'),
        mpatches.Patch(facecolor='#34495E', edgecolor='black', label='Document (DOI)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10,
              framealpha=0.95, title='Legend', title_fontsize=11,
              edgecolor='black', fancybox=True)
    return fig

def draw_sankey_single(data, model_name, model_color):
    if not HAS_PLOTLY:
        return None
    papers = list(data.keys())
    materials = sorted({m for mats in data.values() for m in mats})
    if not papers or not materials:
        return None
    node_labels = [get_alias(p) for p in papers] + [get_mat_label(m) for m in materials]
    node_colors = [model_color] * len(papers) + ['#2C3E50'] * len(materials)

    def hex_to_rgba(hex_color, alpha=0.25):
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'

    sources, targets, values, link_colors = [], [], [], []
    for i, doi in enumerate(papers):
        for mat, cnt in data[doi].items():
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
        title_text=f"{model_name} -- Paper to Material Flow (10k-char context)",
        font_size=12, paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def draw_category_breakdown(data, figsize=(14, 8), bar_alpha=0.92, bar_edge=0.8,
                           font_size=13, show_bar_labels=True, bar_label_size=10,
                           show_grid=True, grid_alpha=0.25, cmap_name='turbo'):

    cat_counts = defaultdict(int)
    cat_n_terms = defaultdict(int)
    cat_terms = defaultdict(list)

    for doi, mats in data.items():
        for mat, cnt in mats.items():
            cat = get_category(mat)
            cat_counts[cat] += cnt
            cat_n_terms[cat] += 1
            cat_terms[cat].append((mat, cnt))

    sorted_cats = sorted(cat_counts.keys(), key=lambda c: -cat_counts[c])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor='white')

    colors_pie = [CATEGORY_BAR_COLORS.get(c, '#BDC3C7') for c in sorted_cats]
    wedges, texts, autotexts = ax1.pie(
        [cat_counts[c] for c in sorted_cats],
        labels=[c.replace(' / ', '\n') for c in sorted_cats],
        colors=colors_pie,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': font_size - 2, 'fontweight': 'bold'},
        pctdistance=0.75,
        labeldistance=1.15
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(font_size - 3)
    ax1.set_title('Occurrence Distribution by Category', fontsize=font_size + 2, fontweight='bold', pad=15)

    y_pos = np.arange(len(sorted_cats))
    bar_colors = [CATEGORY_BAR_COLORS.get(c, '#BDC3C7') for c in sorted_cats]
    bars = ax2.barh(y_pos, [cat_n_terms[c] for c in sorted_cats], color=bar_colors,
                    alpha=bar_alpha, edgecolor='black', linewidth=bar_edge, height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([c.replace(' / ', '\n') for c in sorted_cats], fontsize=font_size - 1)
    ax2.set_xlabel('Number of Unique Terms', fontsize=font_size, fontweight='bold')
    ax2.set_title('Unique Terms per Category', fontsize=font_size + 2, fontweight='bold', pad=15)

    if show_bar_labels:
        for bar, cat in zip(bars, sorted_cats):
            width = bar.get_width()
            ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{int(width)} terms\n({cat_counts[cat]} occ.)',
                    ha='left', va='center', fontsize=bar_label_size, fontweight='bold')

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    if show_grid:
        ax2.grid(axis='x', alpha=grid_alpha, linestyle='--')

    plt.tight_layout()
    return fig

def draw_ranking_distribution(data, figsize=(14, 8), top_n=20,
                                bar_alpha=0.92, bar_edge=0.8, font_size=13,
                                show_bar_labels=True, bar_label_size=10,
                                show_grid=True, grid_alpha=0.25,
                                use_category_colors=True, custom_color='#F39C12'):

    all_terms = defaultdict(int)
    for doi, mats in data.items():
        for mat, cnt in mats.items():
            all_terms[mat] += cnt

    sorted_terms = sorted(all_terms.items(), key=lambda x: -x[1])
    ranks = np.arange(1, len(sorted_terms) + 1)
    frequencies = [c for _, c in sorted_terms]

    fig, (ax_rank, ax_top) = plt.subplots(1, 2, figsize=figsize, facecolor='white')

    ax_rank.loglog(ranks, frequencies, 'o-', color=MODEL_META['modelA']['color'], markersize=6, linewidth=2,
                   markeredgecolor='black', markeredgewidth=0.8)
    ax_rank.set_xlabel('Rank (log scale)', fontsize=font_size, fontweight='bold')
    ax_rank.set_ylabel('Occurrence Count (log scale)', fontsize=font_size, fontweight='bold')
    ax_rank.set_title('Rank-Frequency Distribution', fontsize=font_size + 2, fontweight='bold', pad=15)
    ax_rank.grid(True, alpha=grid_alpha, linestyle='--', which='both')
    ax_rank.spines['top'].set_visible(False)
    ax_rank.spines['right'].set_visible(False)

    if len(frequencies) > 5:
        drop_ratio = frequencies[0] / frequencies[4] if frequencies[4] > 0 else 0
        ax_rank.annotate(f'Top-5 concentration:\n{drop_ratio:.1f}x drop',
                        xy=(3, frequencies[2]), xytext=(len(frequencies)*0.3, frequencies[0]*0.5),
                        fontsize=9, color='#C0392B', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.2),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEDEC',
                                 edgecolor='#C0392B', linewidth=1.2, alpha=0.9))

    top_terms = sorted_terms[:top_n]
    top_labels = [get_mat_label(t) for t, _ in top_terms]
    top_vals = [c for _, c in top_terms]
    top_cats = [get_category(t) for t, _ in top_terms]

    y_pos = np.arange(len(top_labels))
    if use_category_colors:
        bar_colors = [CATEGORY_BAR_COLORS.get(c, custom_color) for c in top_cats]
    else:
        bar_colors = [custom_color] * len(top_labels)

    bars = ax_top.barh(y_pos, top_vals, color=bar_colors, alpha=bar_alpha,
                       edgecolor='black', linewidth=bar_edge, height=0.7)
    ax_top.set_yticks(y_pos)
    ax_top.set_yticklabels(top_labels, fontsize=font_size - 2)
    ax_top.invert_yaxis()
    ax_top.set_xlabel('Occurrence Count', fontsize=font_size, fontweight='bold')
    ax_top.set_title(f'Top {top_n} Extracted Terms', fontsize=font_size + 2, fontweight='bold', pad=15)

    if show_bar_labels:
        for bar, val in zip(bars, top_vals):
            ax_top.text(val + max(top_vals)*0.01, bar.get_y() + bar.get_height()/2,
                       f'{val}', ha='left', va='center', fontsize=bar_label_size, fontweight='bold')

    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    if show_grid:
        ax_top.grid(axis='x', alpha=grid_alpha, linestyle='--')

    plt.tight_layout()
    return fig

def draw_radial_histogram(data, figsize=(12, 12), inner_r=1.0,
                          track_h=0.6, gap_deg=2.0, fs=10, show_labels=True,
                          max_bar_scale=1.0, cmap_name='turbo',
                          label_padding=0.45, label_buffer=0.15):

    mat_counts = defaultdict(int)
    for doi, mats in data.items():
        for mat, cnt in mats.items():
            mat_counts[mat] += cnt

    mats_sorted = sorted(mat_counts.keys(), key=lambda x: -mat_counts[x])
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
    outer_extent = inner_r + track_h + label_padding + label_buffer + 0.15

    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor('white')

    for i, mat in enumerate(mats_sorted):
        ma = i * step + step / 2
        cnt = mat_counts[mat]
        max_cnt = max(mat_counts.values()) if mat_counts else 1
        h = track_h * (0.15 + 0.85 * cnt / max(1, max_cnt)) * max_bar_scale
        ax.bar(ma, h, width=step * 0.85, bottom=inner_r,
               color=MODEL_META['modelA']['color'], alpha=0.92,
               edgecolor='black', linewidth=0.4, zorder=3)

        rim_bottom = inner_r + track_h + 0.04
        ax.bar(ma, 0.08, width=step * 0.88, bottom=rim_bottom,
               color=rim_colors[mat], alpha=0.9, edgecolor='black', linewidth=0.3, zorder=5)

        if show_labels:
            lbl = get_mat_label(mat)
            label_r = inner_r + track_h + label_padding
            rot_deg = np.degrees(ma)
            rot_deg_norm = rot_deg % 360
            if 90 < rot_deg_norm < 270:
                text_rot = rot_deg_norm - 180
                ha = 'right'
            else:
                text_rot = rot_deg_norm
                ha = 'left'
            va = 'center'
            ax.text(ma, label_r, lbl, ha=ha, va=va, fontsize=fs - 1,
                    fontweight='bold', color='#2c3e50', zorder=6,
                    rotation=text_rot, rotation_mode='anchor')

    ax.set_ylim(0, outer_extent)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    ax.set_title('Falcon 10B -- Radial Term Histogram', fontsize=16, fontweight='bold', pad=20, color='#1a1a2e')

    handles = [mpatches.Patch(facecolor=MODEL_META['modelA']['color'], edgecolor='black', label='Falcon 10B (10k chars)')]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.22, 1.08),
              fontsize=fs, framealpha=0.95, title='Model', title_fontsize=fs + 1,
              edgecolor='black', fancybox=True)
    return fig

# ------------------------------------------------------------------
# LOAD & INITIALIZE
# ------------------------------------------------------------------
raw_data = load_data_from_disk_or_default()
all_dois = sorted(raw_data.keys())
all_materials = sorted({m for mats in raw_data.values() for m in mats})
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
        fA = st.file_uploader("Model A CSV (Falcon 10B)", type="csv", key="fa")
        if fA:
            raw_data = parse_alloys_csv_cached(fA.getvalue().decode('utf-8'))
            all_dois = sorted(raw_data.keys())
            all_materials = sorted({m for mats in raw_data.values() for m in mats})
            init_aliases(all_dois, all_materials)
            st.cache_data.clear()

    st.markdown("---")
    st.markdown("## 🧬 Material Validator")
    st.markdown("<small>Regex + Periodic-Table hybrid. Toggle to filter non-materials.</small>", unsafe_allow_html=True)
    enable_validator = st.checkbox("Enable validation", value=True, key="en_val")
    materials_only = st.checkbox("Show ONLY validated materials", value=False, key="mat_only")

    st.markdown("---")
    st.markdown("## 🏷️ Alias Editors")
    with st.expander("✏️ DOI Aliases", expanded=False):
        alias_search = st.text_input("Filter DOIs", "", key="alias_filter")
        filtered_dois = [d for d in all_dois if not alias_search or alias_search.lower() in d.lower()]
        if filtered_dois:
            doi_df = pd.DataFrame({
                "DOI": filtered_dois,
                "Alias": [st.session_state.doi_aliases.get(d, f"[{chr(65 + all_dois.index(d))}]") for d in filtered_dois]
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
    st.markdown("## ⚙️ View Options")
    show_overview = st.checkbox("Show Overview Dashboard", value=True)
    show_chord = st.checkbox("Show Paper-Material Chord", value=True)
    show_sankey = st.checkbox("Show Sankey Flow", value=HAS_PLOTLY)
    show_category = st.checkbox("Show Category Breakdown", value=True)
    show_ranking = st.checkbox("Show Term Ranking", value=True)
    show_radial = st.checkbox("Show Radial Histogram", value=True)
    show_validation = st.checkbox("Show Validation Table", value=True)
    show_raw = st.checkbox("Show Raw Data Table", value=True)

    st.markdown("---")
    st.markdown("## 🎨 Figure Controls")
    with st.expander("Global Settings", expanded=False):
        fig_width = st.slider("Width (inches)", 6, 24, 14, key="fw")
        fig_height = st.slider("Height (inches)", 4, 16, 8, key="fh")
        bar_alpha = st.slider("Bar opacity", 0.3, 1.0, 0.92, 0.02, key="ba")
        bar_edge = st.slider("Edge linewidth", 0.0, 3.0, 0.8, 0.1, key="be")
        show_bar_labels = st.checkbox("Show value labels", value=True, key="bl")
        bar_label_size = st.slider("Label font size", 5, 24, 10, key="bls")
        font_size = st.slider("Font size", 6, 24, 13, key="fs")
        show_grid = st.checkbox("Show grid", value=True, key="sg")
        grid_alpha = st.slider("Grid opacity", 0.0, 1.0, 0.25, 0.05, key="ga")
        cmap_choice = st.selectbox("Colormap", ALL_COLORMAPS, index=ALL_COLORMAPS.index('turbo'), key="cmap")
        use_cat_colors = st.checkbox("Use category colors", value=True, key="ucc")

    st.markdown("---")
    st.markdown("## 📊 Export")
    download_dpi = st.slider("Figure DPI", 150, 600, 300, 50)

# ------------------------------------------------------------------
# FILTER DATA
# ------------------------------------------------------------------
raw_data_tuple = tuple(sorted(raw_data.items()))
display_data = filter_data_by_validation_cached(raw_data_tuple, materials_only)

# ------------------------------------------------------------------
# MAIN HEADER
# ------------------------------------------------------------------
st.markdown('<div class="main-header">🦅 Falcon 10B Alloy Extraction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Standalone Deep-Dive -- Single-Model Analysis under 10,000-Character Constraint</div>', unsafe_allow_html=True)

st.markdown("""
<div class="context-banner">
    <strong>⚠️ Context Limit Constraint</strong><br>
    Model A (Falcon 10B) was evaluated with a <strong>10,000-character input limit</strong> -- 
    one-fifth of the 50,000-character window used for Models B, C, and D in the cross-model comparison. 
    This shorter context forces stricter summarization, potentially favoring <em>precision over recall</em>. 
    Extraction counts and coverage metrics should be interpreted accordingly.
</div>
""", unsafe_allow_html=True)

n_doi = len(display_data)
n_mat = len({m for mats in display_data.values() for m in mats})
n_conn = sum(len(mats) for mats in display_data.values())

cols = st.columns(4)
with cols[0]:
    st.markdown(f'<div class="metric-card"><strong>Documents</strong><br><span style="font-size:2rem; color:#f39c12; font-weight:800">{n_doi}</span></div>', unsafe_allow_html=True)
with cols[1]:
    st.markdown(f'<div class="metric-card metric-card-blue"><strong>Unique Materials</strong><br><span style="font-size:2rem; color:#2980b9; font-weight:800">{n_mat}</span></div>', unsafe_allow_html=True)
with cols[2]:
    st.markdown(f'<div class="metric-card metric-card-green"><strong>Total Extractions</strong><br><span style="font-size:2rem; color:#27ae60; font-weight:800">{n_conn}</span></div>', unsafe_allow_html=True)
with cols[3]:
    avg_per_doc = n_conn / n_doi if n_doi > 0 else 0
    st.markdown(f'<div class="metric-card metric-card-red"><strong>Avg. per Document</strong><br><span style="font-size:2rem; color:#c0392b; font-weight:800">{avg_per_doc:.1f}</span></div>', unsafe_allow_html=True)

st.markdown("""
<div class="insight-box">
    <strong>💡 Context-Normalized Interpretation:</strong> With only <strong>10,000 characters</strong> of input context 
    (vs. 50,000 for other models), each extracted material represents a <strong>higher "information density"</strong> -- 
    the model had less surrounding text to draw from, so each hit is more selective. Lower absolute counts may reflect 
    the reduced window rather than inferior extraction capability.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------
tab_labels = []
if show_chord:      tab_labels.append("🔵 Paper-Material Chord")
if show_sankey:     tab_labels.append("🌊 Sankey Flow")
if show_category:   tab_labels.append("📂 Category Breakdown")
if show_ranking:    tab_labels.append("📈 Term Ranking")
if show_radial:     tab_labels.append("🧬 Radial Histogram")
if show_validation: tab_labels.append("🧪 Validation")
if show_raw:        tab_labels.append("📋 Raw Data")

if not tab_labels:
    st.warning("Enable at least one view in the sidebar.")
    st.stop()

tabs = st.tabs(tab_labels)
t_idx = 0

if show_chord:
    with tabs[t_idx]:
        st.markdown("### Paper-Material Extraction Chord")
        st.markdown("<div class='caption'>Bipartite chord: top arc = documents (DOIs), bottom arc = extracted materials. Ribbon width = occurrence count.</div>", unsafe_allow_html=True)
        fig = draw_single_chord(
            display_data, figsize=(fig_width, fig_height), radius=1.0,
            node_width=0.08, ribbon_alpha=0.55, font_size=font_size,
            min_ribbon_width=0.8, max_ribbon_width=6.0,
            cmap_name=cmap_choice, gap_padding=0.015,
            label_offset=0.20, curve_tension=0.35,
            label_mode='radial'
        )
        render_matplotlib_figure(fig, "falcon10b_chord.png", download_dpi)
        del fig
        gc.collect()
    t_idx += 1

if show_sankey:
    with tabs[t_idx]:
        st.markdown("### Hierarchical Sankey Flow")
        st.markdown("<div class='caption'>Paper to Material flow for Falcon 10B. Node color = document (amber) vs. material (dark).</div>", unsafe_allow_html=True)
        if not HAS_PLOTLY:
            st.warning("Plotly not installed. `pip install plotly` to enable.")
        else:
            fig = draw_sankey_single(display_data, MODEL_META['modelA']['name'], MODEL_META['modelA']['color'])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                del fig
                gc.collect()
    t_idx += 1

if show_category:
    with tabs[t_idx]:
        st.markdown("### Category Breakdown")
        st.markdown("<div class='caption'>Distribution of extracted materials across taxonomy categories.</div>", unsafe_allow_html=True)
        fig = draw_category_breakdown(
            display_data, figsize=(fig_width, fig_height),
            bar_alpha=bar_alpha, bar_edge=bar_edge,
            font_size=font_size, show_bar_labels=show_bar_labels,
            bar_label_size=bar_label_size, show_grid=show_grid,
            grid_alpha=grid_alpha, cmap_name=cmap_choice
        )
        render_matplotlib_figure(fig, "falcon10b_categories.png", download_dpi)
        del fig
        gc.collect()

        st.markdown("#### Category Details")
        cat_counts = defaultdict(int)
        cat_terms = defaultdict(list)
        for doi, mats in display_data.items():
            for mat, cnt in mats.items():
                cat = get_category(mat)
                cat_counts[cat] += cnt
                cat_terms[cat].append((mat, cnt))

        sorted_cats = sorted(cat_counts.keys(), key=lambda c: -cat_counts[c])
        cat_cols = st.columns(min(3, len(sorted_cats)))
        for i, cat in enumerate(sorted_cats):
            with cat_cols[i % 3]:
                n_terms = len(cat_terms[cat])
                st.markdown(f"**{cat}** ({n_terms} terms, {cat_counts[cat]} occ.)")
                df_cat = pd.DataFrame(cat_terms[cat], columns=["Term", "Count"])
                df_cat["Term"] = df_cat["Term"].apply(get_mat_label)
                st.dataframe(df_cat.sort_values("Count", ascending=False), use_container_width=True, height=200, hide_index=True)
    t_idx += 1

if show_ranking:
    with tabs[t_idx]:
        st.markdown("### Term Ranking & Distribution")
        st.markdown("<div class='caption'>Rank-frequency analysis and top-N material bar chart.</div>", unsafe_allow_html=True)
        fig = draw_ranking_distribution(
            display_data, figsize=(fig_width, fig_height), top_n=20,
            bar_alpha=bar_alpha, bar_edge=bar_edge, font_size=font_size,
            show_bar_labels=show_bar_labels, bar_label_size=bar_label_size,
            show_grid=show_grid, grid_alpha=grid_alpha,
            use_category_colors=use_cat_colors, custom_color=MODEL_META['modelA']['color']
        )
        render_matplotlib_figure(fig, "falcon10b_ranking.png", download_dpi)
        del fig
        gc.collect()

        all_terms = defaultdict(int)
        for doi, mats in display_data.items():
            for mat, cnt in mats.items():
                all_terms[mat] += cnt
        frequencies = list(all_terms.values())

        st.markdown("#### Distribution Statistics")
        stats_cols = st.columns(4)
        with stats_cols[0]:
            st.metric("Mean Occurrences", f"{np.mean(frequencies):.1f}")
        with stats_cols[1]:
            st.metric("Median Occurrences", f"{np.median(frequencies):.1f}")
        with stats_cols[2]:
            st.metric("Std. Deviation", f"{np.std(frequencies):.1f}")
        with stats_cols[3]:
            gini = sum(abs(xi - xj) for xi in frequencies for xj in frequencies) / (2 * len(frequencies) * sum(frequencies)) if sum(frequencies) > 0 else 0
            st.metric("Gini Coefficient", f"{gini:.3f}", help="0 = perfectly even; 1 = maximally concentrated")

        if gini < 0.3:
            st.markdown('<div class="success-box"><strong>✅ Balanced Distribution:</strong> Terms are relatively evenly distributed despite the 10k-character constraint.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box"><strong>⚠️ Concentrated Distribution:</strong> High Gini indicates the model is heavily focused on a few key materials. Typical under limited context -- precision over recall.</div>', unsafe_allow_html=True)
    t_idx += 1

if show_radial:
    with tabs[t_idx]:
        st.markdown("### Radial Term Histogram")
        st.markdown("<div class='caption'>Circos-style radial bar chart. Bar height = occurrence count; outer rim color = material identity.</div>", unsafe_allow_html=True)
        fig = draw_radial_histogram(
            display_data, figsize=(fig_width, fig_height), inner_r=1.0,
            track_h=0.6, gap_deg=2.0, fs=font_size, show_labels=True,
            max_bar_scale=1.0, cmap_name=cmap_choice,
            label_padding=0.45, label_buffer=0.15
        )
        render_matplotlib_figure(fig, "falcon10b_radial.png", download_dpi)
        del fig
        gc.collect()
    t_idx += 1

if show_validation:
    with tabs[t_idx]:
        st.markdown("### Material Validation Results")
        st.markdown("<div class='caption'>Regex + periodic-table classification of all extracted terms.</div>", unsafe_allow_html=True)
        val_rows = []
        for mat in sorted(all_materials):
            if mat not in st.session_state.validation_cache:
                st.session_state.validation_cache[mat] = validate_material(mat)
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
        st.download_button("⬇️ Download Validation CSV", csv_val, "falcon10b_validation.csv", "text/csv")
        del df_val, val_rows, csv_val
        gc.collect()
    t_idx += 1

if show_raw:
    with tabs[t_idx]:
        st.markdown("### Raw Extraction Data")
        rows = []
        for doi, mats in display_data.items():
            for mat, cnt in mats.items():
                rows.append({
                    'DOI': doi,
                    'DOI Alias': get_alias(doi),
                    'Material': get_mat_label(mat),
                    'Raw Term': mat,
                    'Category': get_category(mat),
                    'Count': cnt
                })
        df_raw = pd.DataFrame(rows).sort_values(['DOI', 'Count'], ascending=[True, False])
        st.info("📌 **Context Limit Reminder:** This model was evaluated with a **10,000-character** input limit. Lower counts may reflect the reduced context window rather than inferior extraction capability.")
        st.dataframe(df_raw, use_container_width=True, height=500, hide_index=True)
        csv_raw = df_raw.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Raw Data CSV", csv_raw, "falcon10b_raw_data.csv", "text/csv")
        del df_raw, rows, csv_raw
        gc.collect()
    t_idx += 1

plt.close('all')
gc.collect()

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#666; font-size:0.88rem; padding:1rem 0; font-family:serif;">
    <strong>Falcon 10B -- Standalone Alloy Extraction Analysis</strong><br>
    Results obtained under a <strong>10,000-character context limit</strong>. For cross-model comparison, 
    see the full 4-model AlloyExtraction Nexus dashboard where Models B, C, and D use 50,000 characters.<br><br>
    <em>Shorter contexts favor precision over recall. Interpret coverage metrics in light of this constraint.</em>
</div>
""", unsafe_allow_html=True)
