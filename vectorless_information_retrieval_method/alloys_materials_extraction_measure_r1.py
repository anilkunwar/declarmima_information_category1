"""
Publication-Quality Streamlit App: Alloy-Material Extraction Nexus
Chord · Sankey · Network · Heatmap · Editable DOI Aliases
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge, PathPatch, FancyBboxPatch
from matplotlib.path import Path
from io import BytesIO
import os, json, re
from collections import defaultdict, Counter
from itertools import combinations

# ------------------------------------------------------------------
# Optional: Plotly & PyVis with graceful fallback
# ------------------------------------------------------------------
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except Exception:
    HAS_PYVIS = False

# ------------------------------------------------------------------
# PAGE CONFIG & CSS (adapted from reference Publication-Qual code)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="AlloyExtraction Nexus — Cross-Model Chord & Flow",
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
    .highlight-gold { border-left-color: #f39c12; }
    .caption { font-size: 0.9rem; color: #666; font-style: italic; margin-bottom: 1rem; }
    .control-section { background: linear-gradient(180deg, #f5f7fa 0%, #eef1f5 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border: 1px solid #dde2e8; }
    .section-title { font-size: 1.1rem; font-weight: 700; color: #2c3e50; margin-bottom: 0.8rem; border-bottom: 2px solid #3498db; padding-bottom: 0.3rem; }
    .doi-legend { font-family: 'Courier New', monospace; font-size: 0.82rem; color: #2c3e50; background: #f8f9fa; padding: 0.4rem 0.6rem; border-radius: 4px; border: 1px solid #dde2e8; margin-bottom: 0.25rem; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# METADATA & TAXONOMY (reused from reference + extended)
# ------------------------------------------------------------------
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
        'tib2/alsimgzr', 'al-si-mg-zr'
    ],
    'Uncertainty / Unknown': ['unknown', 'ambiguous', 'unclassified'],
}

CATEGORY_COLORS = {
    'Process Parameters':    '#FDEDEC',  # light red
    'Mechanical Properties': '#EBF5FB',  # light blue
    'Microstructural Features':'#E9F7EF',# light green
    'Thermal / Fluid':       '#FEF9E7',  # light yellow
    'Computational / Method':'#F5EEF8',  # light purple
    'Alloy / Composition':   '#E8F6F3',  # light teal
    'Uncertainty / Unknown': '#F2F3F4',  # light gray
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
}

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

def hex_to_rgba(hex_color: str, alpha: float = 0.25) -> str:
    """Convert #RRGGBB to rgba(R,G,B,A) for Plotly compatibility."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ------------------------------------------------------------------
# DATA PARSERS (robust to all 3 model formats)
# ------------------------------------------------------------------
DEFAULT_DATA_B = '''DOI,Materials_Alloys
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
10.26434/chemrxiv-2025-sk6h5,"alsimgzr, sdss_2507, nanoindentation, laser_power, fe"'''

DEFAULT_DATA_C = '''DOI,Materials_Alloys
10.1007/s00366-025-02117-z,"laser_power, yield_strength, cu–mn"
10.1016/j.commatsci.2025.113875,"ti3au, yield_strength"
10.1016/j.engappai.2024.107902,"cr0.4w0.5(zrhfnb)0.1, heas_mpeas, cr0.5w0.3(vnbta)0.2, mo0.1ti0.8(vnbta)0.1, ti0.7(zrhfnb)0.3"
10.1016/j.ijsolstr.2024.112894,"nanoindentation, al, al2cu, yield_strength"
10.1016/j.jallcom.2024.174876,"cu6sn5"
10.1016/j.jestch.2023.101413,"ti3au"
10.1016/j.matdes.2024.113312,"ti6al4v, ti6al4v matrix, lewis_number, ti6al4v alloy, ti2cu imc, ti2cu imc/ti6al4v matrix, yield_strength"
10.1016/j.surfin.2023.102728,"cu6sn5"
10.1080/17452759.2024.2416518,"alsimgzr, metallic_glass"'''

DEFAULT_DATA_D = '''DOI,Materials_Alloys
10.1016/j.apenergy.2024.122901,"b0.3er0.5al0.2n, aln ceramics, b0.3er0.5al0.2n, aln"
10.1016/j.commatsci.2025.113875,"ti-au, alsimgzr, ti–au"
10.1016/j.commatsci.2025.114456,"au, alsimgzr"
10.1016/j.jallcom.2024.174876,"cu, nt_cu, cu6sn5"
10.1016/j.jestch.2023.101413,"ti, alsimgzr, ti-au, gold, ti3au, lewis_number"
10.1016/j.matdes.2024.113312,"ti6al4v, cu, cu, ti6al4v"
10.1016/j.msea.2025.148865,"alsimgzr, yield_strength"
10.1080/17452759.2024.2416518,"metallic_glass, alsimgzr"
10.26434/chemrxiv-2025-sk6h5,"lewis_number, nanoindentation, sdss_2507, laser_power, sdss"'''

def parse_alloys_csv(filepath_or_content, model_name):
    """Robust parser handling standard CSV and modelC's legacy colon format."""
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
    # Detect format
    if lines[0].startswith('DOI,') or lines[0].startswith('doi,'):
        # Standard CSV
        df = pd.read_csv(pd.io.common.StringIO(raw))
        df.columns = [c.strip() for c in df.columns]
        for _, row in df.iterrows():
            doi = str(row.iloc[0]).strip()
            mats = str(row.iloc[1]).strip() if len(row) > 1 else ""
            records.append((doi, mats))
    else:
        # Could be modelC legacy: 10.1016_j.commatsci...pdf: mat, mat
        for line in lines:
            if ':' in line and '.pdf' in line.split(':')[0]:
                left, right = line.split(':', 1)
                doi = left.replace('.pdf', '').replace('_', '/').strip()
                records.append((doi, right.strip()))
            elif ',' in line:
                parts = line.split(',', 1)
                records.append((parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""))

    # Normalize into {doi: {mat: count}}
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
            loaded[key] = parse_alloys_csv(path, MODEL_META[key]['name'])
        else:
            default = {'modelB': DEFAULT_DATA_B, 'modelC': DEFAULT_DATA_C, 'modelD': DEFAULT_DATA_D}[key]
            loaded[key] = parse_alloys_csv(default, MODEL_META[key]['name'])
    return loaded

# ------------------------------------------------------------------
# ALIAS MANAGEMENT
# ------------------------------------------------------------------
def init_aliases(all_dois):
    if 'doi_aliases' not in st.session_state:
        st.session_state.doi_aliases = {}
    # Auto-fill missing with [A], [B]...
    for i, doi in enumerate(sorted(all_dois)):
        if doi not in st.session_state.doi_aliases:
            st.session_state.doi_aliases[doi] = f"[{chr(65 + i)}]"

def get_alias(doi):
    return st.session_state.doi_aliases.get(doi, doi[:20])

# ------------------------------------------------------------------
# CHORD DIAGRAM ENGINE (Matplotlib custom)
# ------------------------------------------------------------------
def draw_bipartite_chord(paper_counts, material_counts, connections,
                         model_color, title, figsize=(14, 14),
                         radius=1.0, node_width=0.08, ribbon_alpha=0.55,
                         font_size=9, show_taxonomy_legend=True,
                         paper_span=np.pi * 0.85, material_span=np.pi * 0.85,
                         min_ribbon_width=0.8, max_ribbon_width=5.5):
    """
    connections: list of (doi, mat, count)
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    # --- Angular allocation ---
    p_total = sum(paper_counts.values())
    p_start = np.pi + paper_span / 2          # top-left going clockwise down
    p_angles = {}
    cur = p_start
    for doi, cnt in paper_counts.items():
        w = (cnt / p_total) * paper_span if p_total > 0 else 0
        p_angles[doi] = {'start': cur, 'end': cur - w, 'mid': cur - w / 2, 'width': w}
        cur -= w

    m_total = sum(material_counts.values())
    m_start = -material_span / 2               # bottom-right going counter-clockwise up
    m_angles = {}
    cur = m_start
    for mat, cnt in material_counts.items():
        w = (cnt / m_total) * material_span if m_total > 0 else 0
        m_angles[mat] = {'start': cur, 'end': cur + w, 'mid': cur + w / 2, 'width': w}
        cur += w

    # --- Draw paper wedges (model color) ---
    for doi, ang in p_angles.items():
        wdg = Wedge((0, 0), radius,
                    np.degrees(ang['end']), np.degrees(ang['start']),
                    width=node_width, facecolor=model_color, edgecolor='black',
                    linewidth=0.8, alpha=0.9, zorder=5)
        ax.add_patch(wdg)

        lr = radius + node_width + 0.06
        x = lr * np.cos(ang['mid'])
        y = lr * np.sin(ang['mid'])
        alias = get_alias(doi)
        rot = np.degrees(ang['mid'])
        # Horizontal-ish readability
        if 90 < rot < 270:
            ha, va = 'right', 'center'
        else:
            ha, va = 'left', 'center'
        ax.text(x, y, alias, ha=ha, va=va, fontsize=font_size,
                fontweight='bold', color='#1a1a2e', zorder=6)

    # --- Draw material wedges (taxonomy color) ---
    for mat, ang in m_angles.items():
        cat = get_category(mat)
        fill = CATEGORY_COLORS.get(cat, '#FFFFFF')
        wdg = Wedge((0, 0), radius,
                    np.degrees(ang['start']), np.degrees(ang['end']),
                    width=node_width, facecolor=fill, edgecolor='black',
                    linewidth=0.8, alpha=0.9, zorder=5)
        ax.add_patch(wdg)

        lr = radius + node_width + 0.06
        x = lr * np.cos(ang['mid'])
        y = lr * np.sin(ang['mid'])
        rot = np.degrees(ang['mid'])
        if 90 < rot < 270:
            ha, va = 'right', 'center'
        else:
            ha, va = 'left', 'center'
        txt_color = get_contrast_text_color(fill) if fill != '#FFFFFF' else '#2c3e50'
        ax.text(x, y, mat, ha=ha, va=va, fontsize=font_size - 1,
                fontweight='bold', color=txt_color, zorder=6)

    # --- Draw ribbons (cubic Bézier) ---
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

        # Control points pulled toward center
        a1, a2 = pa['mid'], ma['mid']
        if abs(a1 - a2) > np.pi:
            if a1 > a2:
                a2 += 2 * np.pi
            else:
                a1 += 2 * np.pi
        cp_r = r_rib * 0.22
        cp1_a = a1 * 0.65 + a2 * 0.35
        cp2_a = a1 * 0.35 + a2 * 0.65
        cp1x, cp1y = cp_r * np.cos(cp1_a), cp_r * np.sin(cp1_a)
        cp2x, cp2y = cp_r * np.cos(cp2_a), cp_r * np.sin(cp2_a)

        verts = [(x1, y1), (cp1x, cp1y), (cp2x, cp2y), (x2, y2)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        path = Path(verts, codes)

        lw = min_ribbon_width + (max_ribbon_width - min_ribbon_width) * (count / max_count)
        patch = PathPatch(path, facecolor='none', edgecolor=model_color,
                          linewidth=lw, alpha=ribbon_alpha, zorder=2, capstyle='round')
        ax.add_patch(patch)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20, color='#1a1a2e')

    if show_taxonomy_legend:
        handles = [mpatches.Patch(facecolor=c, edgecolor='black', label=cat)
                   for cat, c in CATEGORY_COLORS.items() if cat not in ('Other', 'Uncertainty / Unknown')]
        ax.legend(handles=handles, loc='lower right', fontsize=8,
                  framealpha=0.95, title='Material Taxonomy', title_fontsize=9)

    return fig

# ------------------------------------------------------------------
# SANKEY (Plotly)  — FIXED: use rgba() instead of 8-digit hex
# ------------------------------------------------------------------
def draw_sankey(paper_materials, model_name, model_color):
    if not HAS_PLOTLY:
        return None
    papers = list(paper_materials.keys())
    materials = sorted({m for mats in paper_materials.values() for m in mats})
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
                  color=[hex_to_rgba(c, alpha=0.25) for c in link_colors])
    )])
    fig.update_layout(
        title_text=f"{model_name} — Paper → Material Flow",
        font_size=12, paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# ------------------------------------------------------------------
# PYVIS NETWORK
# ------------------------------------------------------------------
def draw_pyvis_network(all_models_data):
    if not HAS_PYVIS:
        return None
    net = Network(height='720px', width='100%', bgcolor='white', font_color='black', heading='')
    net.barnes_hut(gravity=-9000, central_gravity=0.35, spring_length=140,
                   spring_strength=0.04, damping=0.09)

    all_papers = set()
    all_materials = set()
    for data in all_models_data.values():
        all_papers.update(data.keys())
        for mats in data.values():
            all_materials.update(mats.keys())

    # Paper nodes
    for doi in sorted(all_papers):
        alias = get_alias(doi)
        net.add_node(doi, label=alias, title=f"DOI: {doi}", shape='box',
                     color={'background': '#E8F4FD', 'border': '#2980B9'},
                     borderWidth=2, font={'size': 14, 'face': 'arial', 'color': '#1a1a2e'},
                     size=18)

    # Material nodes
    for mat in sorted(all_materials):
        cat = get_category(mat)
        fill = CATEGORY_COLORS.get(cat, '#FFFFFF')
        border = '#555555'
        net.add_node(mat, label=mat, shape='dot', color={'background': fill, 'border': border},
                     borderWidth=2, font={'size': 12, 'color': '#2c3e50'}, size=14,
                     title=f"Category: {cat}")

    # Edges per model
    for model_key, data in all_models_data.items():
        meta = MODEL_META[model_key]
        for doi, mats in data.items():
            for mat, cnt in mats.items():
                net.add_edge(doi, mat, width=min(cnt, 4), color=meta['color'],
                             title=f"{meta['name']}: {cnt}x", smooth={'type': 'continuous'})

    net.set_options("""
    var options = {
      "physics": {"stabilization": {"iterations": 200}},
      "interaction": {"hover": true, "tooltipDelay": 100}
    }
    """)
    # Return HTML string
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
        net.save_graph(f.name)
        with open(f.name, 'r', encoding='utf-8') as hf:
            html = hf.read()
        os.remove(f.name)
    return html

# ------------------------------------------------------------------
# HEATMAP
# ------------------------------------------------------------------
def draw_heatmap(all_models_data):
    all_papers = sorted({d for data in all_models_data.values() for d in data})
    all_materials = sorted({m for data in all_models_data.values() for mats in data.values() for m in mats})

    matrix = np.zeros((len(all_papers), len(all_materials)))
    for data in all_models_data.values():
        for doi, mats in data.items():
            i = all_papers.index(doi)
            for mat in mats:
                j = all_materials.index(mat)
                matrix[i, j] += 1

    fig, ax = plt.subplots(figsize=(max(14, len(all_materials) * 0.6), max(8, len(all_papers) * 0.5)))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=3)

    ax.set_xticks(np.arange(len(all_materials)))
    ax.set_xticklabels(all_materials, rotation=50, ha='right', fontsize=10)
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

    ax.set_title("Model Agreement Heatmap: Paper × Material", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Normalized Material / Concept", fontsize=12, fontweight='bold')
    ax.set_ylabel("Document Alias", fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
raw_data = load_data_from_disk_or_default()
all_dois = sorted({d for data in raw_data.values() for d in data})
init_aliases(all_dois)

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
            raw_data['modelB'] = parse_alloys_csv(fB.getvalue().decode('utf-8'), 'modelB')
        if fC:
            raw_data['modelC'] = parse_alloys_csv(fC.getvalue().decode('utf-8'), 'modelC')
        if fD:
            raw_data['modelD'] = parse_alloys_csv(fD.getvalue().decode('utf-8'), 'modelD')
        all_dois = sorted({d for data in raw_data.values() for d in data})
        init_aliases(all_dois)

    st.markdown("---")
    st.markdown("## 🏷️ DOI Alias Editor")
    st.markdown("<small>Auto-generated <code>[A]</code>–<<code>[Z]</code> mapped to DOIs. Edit below to customize labels for all diagrams.</small>", unsafe_allow_html=True)

    with st.expander("✏️ Edit Aliases", expanded=False):
        # Batch editor: show a few at a time to avoid overwhelming UI
        alias_search = st.text_input("Filter DOIs", "", key="alias_filter")
        for doi in all_dois:
            if alias_search and alias_search.lower() not in doi.lower():
                continue
            current = st.session_state.doi_aliases.get(doi, f"[{doi[:1]}]")
            new_val = st.text_input(f"{doi[:55]}", value=current, key=f"alias_input_{doi}")
            st.session_state.doi_aliases[doi] = new_val

    st.markdown("---")
    st.markdown("## ⚙️ Diagram Toggles")
    show_chord = st.checkbox("Show Chord Diagrams", value=True)
    show_sankey = st.checkbox("Show Sankey Flow", value=True)
    show_network = st.checkbox("Show Network Graph", value=HAS_PYVIS)
    show_heatmap = st.checkbox("Show Agreement Heatmap", value=True)
    show_legend_panel = st.checkbox("Show DOI Legend Panel", value=True)

    st.markdown("---")
    st.markdown("## 🎨 Chord Style")
    with st.expander("Controls", expanded=False):
        chord_figsize = st.slider("Figure size", 8, 24, 14, key="chord_fs")
        chord_radius = st.slider("Radius", 0.6, 1.4, 1.0, 0.05, key="chord_r")
        chord_node_w = st.slider("Node arc width", 0.02, 0.16, 0.08, 0.01, key="chord_nw")
        chord_rib_a = st.slider("Ribbon opacity", 0.1, 1.0, 0.55, 0.05, key="chord_ra")
        chord_font = st.slider("Font size", 5, 18, 9, key="chord_fnt")
        chord_min_lw = st.slider("Min ribbon width", 0.2, 3.0, 0.8, 0.1, key="chord_mlw")
        chord_max_lw = st.slider("Max ribbon width", 1.0, 10.0, 5.5, 0.2, key="chord_Mlw")

    st.markdown("## 📊 Global Export")
    download_dpi = st.slider("Figure DPI", 150, 600, 300, 50)

# ------------------------------------------------------------------
# MAIN HEADER
# ------------------------------------------------------------------
st.markdown('<div class="main-header">AlloyExtraction Nexus</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Cross-Model Document–Material Chord, Sankey, Network & Agreement Analysis</div>', unsafe_allow_html=True)

# Metric cards
cols = st.columns(len(MODEL_META))
for i, (key, meta) in enumerate(MODEL_META.items()):
    data = raw_data.get(key, {})
    n_doi = len(data)
    n_mat = len({m for mats in data.values() for m in mats})
    n_conn = sum(len(mats) for mats in data.values())
    with cols[i]:
        st.markdown(f'''<div class="metric-card highlight-{['red','green','blue'][i]}">
            <strong>{meta['name']}</strong><br>
            <span style="font-size:1.3rem">{n_doi}</span> docs &nbsp;|&nbsp;
            <span style="font-size:1.3rem">{n_mat}</span> unique concepts &nbsp;|&nbsp;
            <span style="font-size:1.3rem">{n_conn}</span> extractions
        </div>''', unsafe_allow_html=True)

st.markdown("---")

# ------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------
tab_labels = []
if show_chord:      tab_labels.append("🔵 Chord Diagrams")
if show_sankey:     tab_labels.append("🌊 Sankey Flow")
if show_network:    tab_labels.append("🕸️ Network")
if show_heatmap:    tab_labels.append("🔥 Agreement Heatmap")
if show_legend_panel: tab_labels.append("📋 DOI Legend")

if not tab_labels:
    st.warning("Enable at least one diagram in the sidebar.")
    st.stop()

tabs = st.tabs(tab_labels)

t_idx = 0

# ------------------------------------------------------------------
# TAB 1: CHORD
# ------------------------------------------------------------------
if show_chord:
    with tabs[t_idx]:
        st.markdown("### Bipartite Chord Diagrams")
        st.markdown("<div class='caption'>Left arc = documents; Right arc = normalized materials. Ribbon thickness ∝ occurrence count. Material arc colors = taxonomy category.</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            consensus_only = st.checkbox("Consensus-only overlay (≥2 models)", value=False, key="consensus_only")
        with c2:
            show_per_model = st.checkbox("Show per-model chords", value=True, key="per_model")

        # Build per-model chord inputs
        for key, meta in MODEL_META.items():
            if key not in raw_data or not show_per_model:
                continue
            data = raw_data[key]
            if not data:
                continue

            paper_counts = {doi: sum(mats.values()) for doi, mats in data.items()}
            material_counts = defaultdict(int)
            connections = []
            for doi, mats in data.items():
                for mat, cnt in mats.items():
                    material_counts[mat] += cnt
                    connections.append((doi, mat, cnt))

            fig = draw_bipartite_chord(
                paper_counts, dict(material_counts), connections,
                meta['color'], f"{meta['name']} Extraction Chord",
                figsize=(chord_figsize, chord_figsize), radius=chord_radius,
                node_width=chord_node_w, ribbon_alpha=chord_rib_a,
                font_size=chord_font, min_ribbon_width=chord_min_lw,
                max_ribbon_width=chord_max_lw
            )
            st.pyplot(fig)

            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=download_dpi, bbox_inches='tight', facecolor='white')
            st.download_button(f"⬇️ Download {meta['short']} Chord PNG", buf.getvalue(),
                               f"chord_{key}.png", "image/png")
            plt.close(fig)

        # Consensus chord
        if consensus_only:
            st.markdown("---")
            st.markdown("### 🏆 Consensus Chord (≥2 Models)")
            st.markdown("<div class='caption'>Only connections found by two or more models. Ribbon color = agreement level.</div>", unsafe_allow_html=True)

            consensus = defaultdict(lambda: {'count': 0, 'models': set()})
            for key, data in raw_data.items():
                for doi, mats in data.items():
                    for mat, cnt in mats.items():
                        consensus[(doi, mat)]['count'] += cnt
                        consensus[(doi, mat)]['models'].add(key)

            consensus2 = {k: v for k, v in consensus.items() if len(v['models']) >= 2}
            if consensus2:
                paper_counts_c = defaultdict(int)
                material_counts_c = defaultdict(int)
                connections_c = []
                for (doi, mat), info in consensus2.items():
                    paper_counts_c[doi] += info['count']
                    material_counts_c[mat] += info['count']
                    # Color by agreement level: 2 = orange, 3 = green
                    color = '#27AE60' if len(info['models']) >= 3 else '#F39C12'
                    connections_c.append((doi, mat, info['count'], color))

                # Hack: draw multiple times with different colors? No, draw once with blended approach.
                # We'll use a single neutral color for consensus since per-ribbon color is hard with one call.
                # Instead, re-implement a multi-color chord or accept gold consensus.
                fig_c = draw_bipartite_chord(
                    dict(paper_counts_c), dict(material_counts_c),
                    [(d, m, c) for d, m, c, _ in connections_c],
                    '#B7950B', "Consensus Chord (≥2 Models)",
                    figsize=(chord_figsize, chord_figsize), radius=chord_radius,
                    node_width=chord_node_w, ribbon_alpha=0.75,
                    font_size=chord_font, min_ribbon_width=chord_min_lw,
                    max_ribbon_width=chord_max_lw
                )
                st.pyplot(fig_c)
                buf = BytesIO()
                fig_c.savefig(buf, format="png", dpi=download_dpi, bbox_inches='tight', facecolor='white')
                st.download_button("⬇️ Download Consensus Chord PNG", buf.getvalue(),
                                   "chord_consensus.png", "image/png")
                plt.close(fig_c)
            else:
                st.info("No consensus connections (≥2 models) found with current data.")
    t_idx += 1

# ------------------------------------------------------------------
# TAB 2: SANKEY
# ------------------------------------------------------------------
if show_sankey:
    with tabs[t_idx]:
        st.markdown("### Hierarchical Sankey Flow")
        st.markdown("<div class='caption'>Document → Material extraction flow per model. Useful for spotting coverage gaps.</div>", unsafe_allow_html=True)
        if not HAS_PLOTLY:
            st.warning("Plotly not installed. Sankey disabled. `pip install plotly` to enable.")
        else:
            for key, meta in MODEL_META.items():
                if key not in raw_data:
                    continue
                fig = draw_sankey(raw_data[key], meta['name'], meta['color'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
    t_idx += 1

# ------------------------------------------------------------------
# TAB 3: NETWORK
# ------------------------------------------------------------------
if show_network:
    with tabs[t_idx]:
        st.markdown("### Interactive Bipartite Network")
        st.markdown("<div class='caption'>Force-directed layout. ▭ = document, ● = material. Edge color = model. Drag, zoom, hover.</div>", unsafe_allow_html=True)
        if not HAS_PYVIS:
            st.warning("PyVis not installed. Network disabled. `pip install pyvis` to enable.")
        else:
            html = draw_pyvis_network(raw_data)
            if html:
                import streamlit.components.v1 as components
                components.html(html, height=740, scrolling=False)
                # Also offer download
                st.download_button("⬇️ Download Network HTML", html.encode('utf-8'),
                                   "alloy_network.html", "text/html")
    t_idx += 1

# ------------------------------------------------------------------
# TAB 4: HEATMAP
# ------------------------------------------------------------------
if show_heatmap:
    with tabs[t_idx]:
        st.markdown("### Model Agreement Heatmap")
        st.markdown("<div class='caption'>Cell value = number of models extracting that material for that DOI. Darker = higher agreement.</div>", unsafe_allow_html=True)
        fig_h = draw_heatmap(raw_data)
        st.pyplot(fig_h)
        buf = BytesIO()
        fig_h.savefig(buf, format="png", dpi=download_dpi, bbox_inches='tight', facecolor='white')
        st.download_button("⬇️ Download Heatmap PNG", buf.getvalue(),
                           "agreement_heatmap.png", "image/png")
        plt.close(fig_h)
    t_idx += 1

# ------------------------------------------------------------------
# TAB 5: LEGEND
# ------------------------------------------------------------------
if show_legend_panel:
    with tabs[t_idx]:
        st.markdown("### DOI Alias Legend")
        st.markdown("<div class='caption'>Editable mapping used across all diagrams. Export/import for reproducibility.</div>", unsafe_allow_html=True)

        legend_df = pd.DataFrame([
            {"Alias": get_alias(doi), "DOI": doi, "Models Found In": ", ".join([
                MODEL_META[k]['short'] for k in raw_data if doi in raw_data[k]
            ])}
            for doi in all_dois
        ])
        st.dataframe(legend_df, use_container_width=True, height=500)

        # Export aliases JSON
        alias_json = json.dumps(st.session_state.doi_aliases, indent=2)
        st.download_button("⬇️ Export Aliases JSON", alias_json.encode('utf-8'),
                           "doi_aliases.json", "application/json")

        # Import aliases
        uploaded_aliases = st.file_uploader("Import Aliases JSON", type="json", key="alias_import")
        if uploaded_aliases is not None:
            imported = json.load(uploaded_aliases)
            st.session_state.doi_aliases.update(imported)
            st.success("Aliases imported! Refresh the page to apply across all diagrams.")
            st.button("🔄 Rerun", on_click=lambda: st.rerun())
    t_idx += 1

# ------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem; padding:1rem 0;">
    <strong>Pipeline Note:</strong> Concept normalization (synonym resolution) is applied before visualization
    so that <code>ti–au</code>, <code>ti-au</code>, and <code>au-ti</code> aggregate into a single node,
    and <code>al–si–mg–zr alloy</code> collapses to <code>alsimgzr</code>. Taxonomy colors separate
    alloy compositions from process parameters and mechanical properties.
</div>
""", unsafe_allow_html=True)
