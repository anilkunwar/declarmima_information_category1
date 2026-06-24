"""
Standalone Streamlit App: Model A (Falcon 10B) Physical Quantity Extraction
Single-model deep-dive with 10,000-character context limit awareness
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
import os
import json
from collections import Counter

st.set_page_config(
    page_title="Falcon 10B Extraction Analysis",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.4rem; font-weight: 800; color: #1a1a2e; margin-bottom: 0.3rem; letter-spacing: -0.5px; }
    .sub-header { font-size: 1.15rem; color: #5a5a7a; margin-bottom: 2rem; font-weight: 400; }
    .metric-card { background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); border-left: 4px solid #f39c12; padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
    .metric-card-blue { border-left-color: #2980b9; }
    .metric-card-green { border-left-color: #27ae60; }
    .metric-card-red { border-left-color: #c0392b; }
    .caption { font-size: 0.9rem; color: #666; font-style: italic; margin-bottom: 1rem; }
    .control-section { background: linear-gradient(180deg, #f5f7fa 0%, #eef1f5 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border: 1px solid #dde2e8; }
    .section-title { font-size: 1.1rem; font-weight: 700; color: #2c3e50; margin-bottom: 0.8rem; border-bottom: 2px solid #f39c12; padding-bottom: 0.3rem; }
    .context-banner { background: linear-gradient(135deg, #fef9e7 0%, #fdebd0 100%); border: 2px solid #f39c12; border-left: 5px solid #f39c12; padding: 1.2rem; border-radius: 8px; margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(243,156,18,0.15); }
    .insight-box { background: linear-gradient(135deg, #ebf5fb 0%, #d4e6f1 100%); border-left: 4px solid #2980b9; padding: 1rem; border-radius: 6px; margin: 0.5rem 0; }
    .warning-box { background: linear-gradient(135deg, #fdedec 0%, #f5b7b1 100%); border-left: 4px solid #c0392b; padding: 1rem; border-radius: 6px; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TAXONOMY
# ─────────────────────────────────────────────────────────────
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
    'Process Parameters': '#E74C3C',
    'Mechanical Properties': '#2980B9',
    'Microstructural Features': '#27AE60',
    'Thermal / Fluid': '#F39C12',
    'Computational / Method': '#8E44AD',
    'Uncertainty / Unknown': '#7F8C8D',
    'Other': '#BDC3C7'
}

CATEGORY_BG_COLORS = {
    'Process Parameters': '#FDEDEC',
    'Mechanical Properties': '#EBF5FB',
    'Microstructural Features': '#E9F7EF',
    'Thermal / Fluid': '#FEF9E7',
    'Computational / Method': '#F5EEF8',
    'Uncertainty / Unknown': '#F2F3F4',
    'Other': '#FFFFFF'
}

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
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
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "white" if luminance < 0.5 else "#2C3E50"

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📁 Input File")
    upload_mode = st.radio("Input mode", ["Auto-detect filename", "Manual upload"], index=0)

    if upload_mode == "Manual upload":
        fA = st.file_uploader("Model A (Falcon 10B) CSV", type="csv", key="fa")
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

        dataA = read_uploaded(fA)
        dataRef = read_uploaded(fRef)
    else:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PHYSICAL_QUANTITIES_DIR = os.path.join(SCRIPT_DIR, "physical_quantities")
        os.makedirs(PHYSICAL_QUANTITIES_DIR, exist_ok=True)

        dataA = load_csv(os.path.join(PHYSICAL_QUANTITIES_DIR, "physical_quantities_detection_llm_modelA.csv"))
        dataRef = load_csv(os.path.join(PHYSICAL_QUANTITIES_DIR, "physical_quantities_detection_llm_reference.csv"))

        st.markdown("---")
        st.markdown("**Expected path:**")
        st.code("physical_quantities/\n  physical_quantities_detection_llm_modelA.csv")
        st.markdown("*Optional:* `physical_quantities/physical_quantities_detection_llm_reference.csv`")

    st.markdown("---")
    st.markdown("## ⚙️ View Options")
    show_overview = st.checkbox("Show Overview Dashboard", value=True)
    show_category_breakdown = st.checkbox("Show Category Breakdown", value=True)
    show_ranking = st.checkbox("Show Term Ranking & Distribution", value=True)
    show_reference_comparison = st.checkbox("Show Reference Comparison", value=True if dataRef else False, disabled=not dataRef)
    show_quality_metrics = st.checkbox("Show Quality Metrics", value=True)
    show_tables = st.checkbox("Show Raw Data Table", value=True)

    st.markdown("---")
    st.markdown("## 📊 Export Settings")
    download_dpi = st.slider("Figure DPI", min_value=150, max_value=600, value=300, step=50)

    # ── Figure Controls ──
    with st.expander("🎨 Figure Controls", expanded=False):
        st.markdown('<div class="control-section">', unsafe_allow_html=True)

        st.markdown('<div class="section-title">📐 Dimensions</div>', unsafe_allow_html=True)
        fig_width = st.slider("Width (inches)", 6, 24, 14, key="fw")
        fig_height = st.slider("Height (inches)", 4, 16, 8, key="fh")

        st.markdown('<div class="section-title">📊 Bar Appearance</div>', unsafe_allow_html=True)
        bar_alpha = st.slider("Bar opacity", 0.3, 1.0, 0.92, 0.02, key="ba")
        bar_edge = st.slider("Edge linewidth", 0.0, 3.0, 0.8, 0.1, key="be")
        show_bar_labels = st.checkbox("Show value labels", value=True, key="bl")
        bar_label_size = st.slider("Label font size", 5, 24, 10, key="bls")

        st.markdown('<div class="section-title">🔤 Fonts</div>', unsafe_allow_html=True)
        title_font = st.slider("Title size", 8, 36, 16, key="tf")
        xlabel_font = st.slider("X-label size", 6, 32, 13, key="xlf")
        ylabel_font = st.slider("Y-label size", 6, 32, 13, key="ylf")
        tick_font = st.slider("Tick size", 5, 24, 10, key="tkf")

        st.markdown('<div class="section-title">📏 Grid & Spines</div>', unsafe_allow_html=True)
        show_grid = st.checkbox("Show grid", value=True, key="sg")
        grid_alpha = st.slider("Grid opacity", 0.0, 1.0, 0.25, 0.05, key="ga")
        show_top = st.checkbox("Top spine", value=False, key="stp")
        show_right = st.checkbox("Right spine", value=False, key="srp")
        spine_width = st.slider("Spine linewidth", 0.5, 3.0, 1.0, 0.1, key="spw")

        st.markdown('<div class="section-title">🌈 Colors</div>', unsafe_allow_html=True)
        use_category_colors = st.checkbox("Use category colors", value=True, key="ucc")
        custom_bar_color = st.color_picker("Custom bar color", "#F39C12", key="cbc")

        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🦅 Falcon 10B Extraction Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Single-Model Deep Dive — Physical Quantity Extraction under 10,000-Character Constraint</div>', unsafe_allow_html=True)

if not dataA:
    st.warning("⚠️ No Model A CSV file detected. Please upload `physical_quantities_detection_llm_modelA.csv` or use manual upload.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# CONTEXT LIMIT BANNER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="context-banner">
    <strong>⚠️ Context Limit Constraint</strong><br>
    Model A (Falcon 10B) was evaluated with a <strong>10,000-character input limit</strong> — 
    one-fifth of the 50,000-character window used for Models B, C, and D in the cross-model comparison. 
    This shorter context forces stricter summarization, potentially favoring <em>precision over recall</em>. 
    Term counts and coverage metrics should be interpreted accordingly.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA PREP
# ─────────────────────────────────────────────────────────────
terms = list(dataA.keys())
counts = list(dataA.values())
total_occurrences = sum(counts)
n_terms = len(terms)

# Categorize
categorized = {}
for term, count in dataA.items():
    cat = get_category(term)
    if cat not in categorized:
        categorized[cat] = []
    categorized[cat].append((term, count))

# Sort each category by count descending
for cat in categorized:
    categorized[cat].sort(key=lambda x: -x[1])

# ─────────────────────────────────────────────────────────────
# OVERVIEW DASHBOARD
# ─────────────────────────────────────────────────────────────
if show_overview:
    st.markdown("---")
    st.markdown("### 📊 Overview Dashboard")

    cols = st.columns(4)
    with cols[0]:
        st.markdown(f'''<div class="metric-card">
            <strong>Total Terms</strong><br>
            <span style="font-size:2rem; color:#f39c12; font-weight:800">{n_terms}</span>
        </div>''', unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f'''<div class="metric-card metric-card-blue">
            <strong>Total Occurrences</strong><br>
            <span style="font-size:2rem; color:#2980b9; font-weight:800">{total_occurrences}</span>
        </div>''', unsafe_allow_html=True)
    with cols[2]:
        avg_count = total_occurrences / n_terms if n_terms > 0 else 0
        st.markdown(f'''<div class="metric-card metric-card-green">
            <strong>Avg. Occurrences/Term</strong><br>
            <span style="font-size:2rem; color:#27ae60; font-weight:800">{avg_count:.1f}</span>
        </div>''', unsafe_allow_html=True)
    with cols[3]:
        max_term = max(dataA.items(), key=lambda x: x[1])
        st.markdown(f'''<div class="metric-card metric-card-red">
            <strong>Top Term</strong><br>
            <span style="font-size:1.3rem; color:#c0392b; font-weight:700">{max_term[0].replace("_", " ").title()}</span><br>
            <span style="font-size:1.1rem">{max_term[1]} occurrences</span>
        </div>''', unsafe_allow_html=True)

    # Context-normalized insight
    st.markdown("""
    <div class="insight-box">
        <strong>💡 Context-Normalized Interpretation:</strong> With only <strong>10,000 characters</strong> of input context 
        (vs. 50,000 for other models), each extracted term represents a <strong>higher "information density"</strong> — 
        the model had less surrounding text to draw from, so each hit is more selective.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CATEGORY BREAKDOWN
# ─────────────────────────────────────────────────────────────
if show_category_breakdown:
    st.markdown("---")
    st.markdown("### 📂 Category Breakdown")
    st.markdown("<div class='caption'>Distribution of extracted terms across materials design categories.</div>", unsafe_allow_html=True)

    cat_names = list(CATEGORY_COLORS.keys())
    cat_counts = {cat: sum(c for _, c in categorized.get(cat, [])) for cat in cat_names}
    cat_n_terms = {cat: len(categorized.get(cat, [])) for cat in cat_names}

    # Remove empty categories
    cat_names = [c for c in cat_names if cat_counts[c] > 0]

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height), facecolor="white")

    # Left: Pie chart of occurrences
    colors_pie = [CATEGORY_COLORS[c] for c in cat_names]
    wedges, texts, autotexts = ax1.pie(
        [cat_counts[c] for c in cat_names],
        labels=[c.replace(" / ", "\n") for c in cat_names],
        colors=colors_pie,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": tick_font, "fontweight": "bold"},
        pctdistance=0.75,
        labeldistance=1.15
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(tick_font - 1)
    ax1.set_title("Occurrence Distribution by Category", fontsize=title_font, fontweight="bold", pad=15)

    # Right: Horizontal bar of term counts per category
    y_pos = np.arange(len(cat_names))
    bar_colors = [CATEGORY_COLORS[c] for c in cat_names]
    bars = ax2.barh(y_pos, [cat_n_terms[c] for c in cat_names], color=bar_colors, 
                    alpha=bar_alpha, edgecolor="black", linewidth=bar_edge, height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([c.replace(" / ", "\n") for c in cat_names], fontsize=tick_font)
    ax2.set_xlabel("Number of Unique Terms", fontsize=xlabel_font, fontweight="bold")
    ax2.set_title("Unique Terms per Category", fontsize=title_font, fontweight="bold", pad=15)

    if show_bar_labels:
        for bar, cat in zip(bars, cat_names):
            width = bar.get_width()
            ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f"{int(width)} terms\n({cat_counts[cat]} occ.)", 
                    ha="left", va="center", fontsize=bar_label_size, fontweight="bold")

    ax2.spines["top"].set_visible(show_top)
    ax2.spines["right"].set_visible(show_right)
    for spine in ["top", "right", "bottom", "left"]:
        ax2.spines[spine].set_linewidth(spine_width)
    if show_grid:
        ax2.grid(axis="x", alpha=grid_alpha, linestyle="--")

    plt.tight_layout()
    st.pyplot(fig1)

    buf = BytesIO()
    fig1.savefig(buf, format="png", dpi=download_dpi, bbox_inches="tight", facecolor="white")
    st.download_button("⬇️ Download Category Breakdown (PNG)", buf.getvalue(), 
                      "fig_A_category_breakdown.png", "image/png")
    plt.close(fig1)

    # Category detail tables
    st.markdown("#### Category Details")
    cat_cols = st.columns(min(3, len(cat_names)))
    for i, cat in enumerate(cat_names):
        with cat_cols[i % 3]:
            st.markdown(f"**{cat}** ({cat_n_terms[cat]} terms, {cat_counts[cat]} occ.)")
            df_cat = pd.DataFrame(categorized[cat], columns=["Term", "Count"])
            df_cat["Term"] = df_cat["Term"].str.replace("_", " ").str.title()
            st.dataframe(df_cat, use_container_width=True, height=200, hide_index=True)

# ─────────────────────────────────────────────────────────────
# TERM RANKING & DISTRIBUTION
# ─────────────────────────────────────────────────────────────
if show_ranking:
    st.markdown("---")
    st.markdown("### 📈 Term Ranking & Occurrence Distribution")
    st.markdown("<div class='caption'>Rank-frequency plot and top-N term bar chart.</div>", unsafe_allow_html=True)

    sorted_terms = sorted(dataA.items(), key=lambda x: -x[1])
    ranks = np.arange(1, len(sorted_terms) + 1)
    frequencies = [c for _, c in sorted_terms]

    fig2, (ax_rank, ax_top) = plt.subplots(1, 2, figsize=(fig_width, fig_height), facecolor="white")

    # Left: Rank-frequency (log-log)
    ax_rank.loglog(ranks, frequencies, "o-", color="#F39C12", markersize=6, linewidth=2, 
                   markeredgecolor="black", markeredgewidth=0.8)
    ax_rank.set_xlabel("Rank (log scale)", fontsize=xlabel_font, fontweight="bold")
    ax_rank.set_ylabel("Occurrence Count (log scale)", fontsize=ylabel_font, fontweight="bold")
    ax_rank.set_title("Rank-Frequency Distribution", fontsize=title_font, fontweight="bold", pad=15)
    ax_rank.grid(True, alpha=grid_alpha, linestyle="--", which="both")
    ax_rank.spines["top"].set_visible(show_top)
    ax_rank.spines["right"].set_visible(show_right)
    for spine in ["top", "right", "bottom", "left"]:
        ax_rank.spines[spine].set_linewidth(spine_width)

    # Add annotation for steep drop-off
    if len(frequencies) > 5:
        drop_ratio = frequencies[0] / frequencies[4] if frequencies[4] > 0 else 0
        ax_rank.annotate(f"Top-5 concentration:\n{drop_ratio:.1f}× drop", 
                        xy=(3, frequencies[2]), xytext=(len(frequencies)*0.3, frequencies[0]*0.5),
                        fontsize=9, color="#C0392B", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.2),
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDEDEC", 
                                 edgecolor="#C0392B", linewidth=1.2, alpha=0.9))

    # Right: Top-N bar chart
    top_n = min(20, len(sorted_terms))
    top_terms = sorted_terms[:top_n]
    top_labels = [t.replace("_", " ").title() for t, _ in top_terms]
    top_vals = [c for _, c in top_terms]
    top_cats = [get_category(t) for t, _ in top_terms]

    y_pos = np.arange(len(top_labels))
    if use_category_colors:
        bar_colors_top = [CATEGORY_COLORS.get(c, custom_bar_color) for c in top_cats]
    else:
        bar_colors_top = [custom_bar_color] * len(top_labels)

    bars = ax_top.barh(y_pos, top_vals, color=bar_colors_top, alpha=bar_alpha, 
                       edgecolor="black", linewidth=bar_edge, height=0.7)
    ax_top.set_yticks(y_pos)
    ax_top.set_yticklabels(top_labels, fontsize=tick_font)
    ax_top.invert_yaxis()
    ax_top.set_xlabel("Occurrence Count", fontsize=xlabel_font, fontweight="bold")
    ax_top.set_title(f"Top {top_n} Extracted Terms", fontsize=title_font, fontweight="bold", pad=15)

    if show_bar_labels:
        for bar, val in zip(bars, top_vals):
            ax_top.text(val + max(top_vals)*0.01, bar.get_y() + bar.get_height()/2, 
                       f"{val}", ha="left", va="center", fontsize=bar_label_size, fontweight="bold")

    ax_top.spines["top"].set_visible(show_top)
    ax_top.spines["right"].set_visible(show_right)
    for spine in ["top", "right", "bottom", "left"]:
        ax_top.spines[spine].set_linewidth(spine_width)
    if show_grid:
        ax_top.grid(axis="x", alpha=grid_alpha, linestyle="--")

    plt.tight_layout()
    st.pyplot(fig2)

    buf = BytesIO()
    fig2.savefig(buf, format="png", dpi=download_dpi, bbox_inches="tight", facecolor="white")
    st.download_button("⬇️ Download Ranking & Distribution (PNG)", buf.getvalue(), 
                      "fig_A_ranking_distribution.png", "image/png")
    plt.close(fig2)

    # Distribution statistics
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
        st.metric("Gini Coefficient", f"{gini:.3f}", help="0 = perfectly even distribution; 1 = maximally concentrated")

# ─────────────────────────────────────────────────────────────
# REFERENCE COMPARISON
# ─────────────────────────────────────────────────────────────
if show_reference_comparison and dataRef:
    st.markdown("---")
    st.markdown("### 🎯 Reference Comparison")
    st.markdown("<div class='caption'>Comparison against ground-truth reference terms. Note: reference may have been compiled from full-text, not limited to 10k chars.</div>", unsafe_allow_html=True)

    ref_terms = set(dataRef.keys()) - excluded_terms
    model_terms = set(dataA.keys()) - excluded_terms

    true_positives = model_terms & ref_terms
    false_positives = model_terms - ref_terms
    false_negatives = ref_terms - model_terms

    precision = len(true_positives) / len(model_terms) * 100 if model_terms else 0
    recall = len(true_positives) / len(ref_terms) * 100 if ref_terms else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Metrics
    comp_cols = st.columns(4)
    with comp_cols[0]:
        st.markdown(f'''<div class="metric-card metric-card-green">
            <strong>Precision</strong><br>
            <span style="font-size:2rem; color:#27ae60; font-weight:800">{precision:.1f}%</span><br>
            <span style="font-size:0.85rem">{len(true_positives)} / {len(model_terms)} terms</span>
        </div>''', unsafe_allow_html=True)
    with comp_cols[1]:
        st.markdown(f'''<div class="metric-card metric-card-blue">
            <strong>Recall</strong><br>
            <span style="font-size:2rem; color:#2980b9; font-weight:800">{recall:.1f}%</span><br>
            <span style="font-size:0.85rem">{len(true_positives)} / {len(ref_terms)} terms</span>
        </div>''', unsafe_allow_html=True)
    with comp_cols[2]:
        st.markdown(f'''<div class="metric-card">
            <strong>F1 Score</strong><br>
            <span style="font-size:2rem; color:#f39c12; font-weight:800">{f1:.1f}%</span>
        </div>''', unsafe_allow_html=True)
    with comp_cols[3]:
        coverage = len(true_positives) / len(ref_terms) * 100 if ref_terms else 0
        st.markdown(f'''<div class="metric-card metric-card-red">
            <strong>Reference Coverage</strong><br>
            <span style="font-size:2rem; color:#c0392b; font-weight:800">{coverage:.1f}%</span><br>
            <span style="font-size:0.85rem">{len(true_positives)} of {len(ref_terms)} ref. terms</span>
        </div>''', unsafe_allow_html=True)

    # Venn-style comparison figure
    fig3, (ax_venn, ax_fn) = plt.subplots(1, 2, figsize=(fig_width, fig_height * 0.8), facecolor="white")

    # Left: Overlap bar chart
    categories = ["True Positives\n(Extracted +\nIn Reference)", 
                  "False Positives\n(Extracted +\nNot in Ref)", 
                  "False Negatives\n(Missed from\nReference)"]
    counts = [len(true_positives), len(false_positives), len(false_negatives)]
    colors_venn = ["#27AE60", "#E67E22", "#C0392B"]

    bars = ax_venn.bar(range(len(categories)), counts, color=colors_venn, alpha=bar_alpha,
                       edgecolor="black", linewidth=bar_edge, width=0.6)
    ax_venn.set_xticks(range(len(categories)))
    ax_venn.set_xticklabels(categories, fontsize=tick_font, fontweight="bold")
    ax_venn.set_ylabel("Term Count", fontsize=ylabel_font, fontweight="bold")
    ax_venn.set_title("Extraction vs. Reference Overlap", fontsize=title_font, fontweight="bold", pad=15)

    if show_bar_labels:
        for bar, val in zip(bars, counts):
            ax_venn.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.02,
                        f"{val}", ha="center", va="bottom", fontsize=bar_label_size, fontweight="bold")

    ax_venn.spines["top"].set_visible(show_top)
    ax_venn.spines["right"].set_visible(show_right)
    for spine in ["top", "right", "bottom", "left"]:
        ax_venn.spines[spine].set_linewidth(spine_width)
    if show_grid:
        ax_venn.grid(axis="y", alpha=grid_alpha, linestyle="--")

    # Right: False negatives (missed terms) — most important for limited context
    if false_negatives:
        fn_list = sorted([(t, dataRef.get(t, 0)) for t in false_negatives], key=lambda x: -x[1])[:15]
        fn_labels = [t.replace("_", " ").title() for t, _ in fn_list]
        fn_vals = [c for _, c in fn_list]
        y_pos = np.arange(len(fn_labels))

        ax_fn.barh(y_pos, fn_vals, color="#C0392B", alpha=bar_alpha, 
                   edgecolor="black", linewidth=bar_edge, height=0.6)
        ax_fn.set_yticks(y_pos)
        ax_fn.set_yticklabels(fn_labels, fontsize=tick_font)
        ax_fn.invert_yaxis()
        ax_fn.set_xlabel("Reference Occurrence Count", fontsize=xlabel_font, fontweight="bold")
        ax_fn.set_title("Top Missed Terms (False Negatives)", fontsize=title_font, fontweight="bold", pad=15)

        ax_fn.spines["top"].set_visible(show_top)
        ax_fn.spines["right"].set_visible(show_right)
        for spine in ["top", "right", "bottom", "left"]:
            ax_fn.spines[spine].set_linewidth(spine_width)
        if show_grid:
            ax_fn.grid(axis="x", alpha=grid_alpha, linestyle="--")
    else:
        ax_fn.text(0.5, 0.5, "No False Negatives!\nPerfect Recall", ha="center", va="center",
                  fontsize=16, fontweight="bold", color="#27AE60", transform=ax_fn.transAxes)
        ax_fn.set_xticks([])
        ax_fn.set_yticks([])
        ax_fn.set_title("Top Missed Terms (False Negatives)", fontsize=title_font, fontweight="bold", pad=15)

    plt.tight_layout()
    st.pyplot(fig3)

    buf = BytesIO()
    fig3.savefig(buf, format="png", dpi=download_dpi, bbox_inches="tight", facecolor="white")
    st.download_button("⬇️ Download Reference Comparison (PNG)", buf.getvalue(), 
                      "fig_A_reference_comparison.png", "image/png")
    plt.close(fig3)

    # Context-limit impact note
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Context Limit Impact on Recall:</strong> False negatives may include terms that appeared 
        <em>beyond the 10,000-character window</em> in the source documents. With a 50,000-character limit, 
        these terms might have been captured. Consider this when evaluating recall — the model may be 
        <strong>capable</strong> but <strong>constrained</strong>.
    </div>
    """, unsafe_allow_html=True)

    # Detailed tables
    st.markdown("#### Detailed Comparison Tables")
    tab_tp, tab_fp, tab_fn = st.tabs(["✅ True Positives", "⚠️ False Positives", "❌ False Negatives"])
    with tab_tp:
        df_tp = pd.DataFrame([(t.replace("_", " ").title(), dataA.get(t, 0), dataRef.get(t, 0), get_category(t)) 
                              for t in sorted(true_positives)], 
                             columns=["Term", "Model A Count", "Reference Count", "Category"])
        st.dataframe(df_tp, use_container_width=True, height=350, hide_index=True)
    with tab_fp:
        df_fp = pd.DataFrame([(t.replace("_", " ").title(), dataA.get(t, 0), get_category(t)) 
                              for t in sorted(false_positives)], 
                             columns=["Term", "Model A Count", "Category"])
        st.dataframe(df_fp, use_container_width=True, height=350, hide_index=True)
    with tab_fn:
        df_fn = pd.DataFrame([(t.replace("_", " ").title(), dataRef.get(t, 0), get_category(t)) 
                              for t in sorted(false_negatives)], 
                             columns=["Term", "Reference Count", "Category"])
        st.dataframe(df_fn, use_container_width=True, height=350, hide_index=True)

# ─────────────────────────────────────────────────────────────
# QUALITY METRICS
# ─────────────────────────────────────────────────────────────
if show_quality_metrics:
    st.markdown("---")
    st.markdown("### 🔍 Quality Metrics")
    st.markdown("<div class='caption'>Schema compliance, concentration, and extraction confidence indicators.</div>", unsafe_allow_html=True)

    # Compute metrics
    unknown_count = dataA.get("unknown", 0)
    unknown_terms = sum(1 for t in dataA if t == "unknown" or "unknown" in t)
    schema_compliance = (total_occurrences - unknown_count) / total_occurrences * 100 if total_occurrences > 0 else 100

    # Concentration: how much do top terms dominate?
    sorted_vals = sorted(dataA.values(), reverse=True)
    top5_share = sum(sorted_vals[:5]) / total_occurrences * 100 if total_occurrences > 0 else 0
    top10_share = sum(sorted_vals[:10]) / total_occurrences * 100 if total_occurrences > 0 else 0

    # Effective vocabulary size (Simpson-like)
    probs = np.array([c / total_occurrences for c in dataA.values()]) if total_occurrences > 0 else np.array([0])
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(len(probs)) if len(probs) > 1 else 1
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    qual_cols = st.columns(4)
    with qual_cols[0]:
        st.markdown(f'''<div class="metric-card metric-card-green">
            <strong>Schema Compliance</strong><br>
            <span style="font-size:2rem; color:#27ae60; font-weight:800">{schema_compliance:.1f}%</span><br>
            <span style="font-size:0.85rem">{unknown_count} unknown occurrences</span>
        </div>''', unsafe_allow_html=True)
    with qual_cols[1]:
        st.markdown(f'''<div class="metric-card metric-card-blue">
            <strong>Top-5 Concentration</strong><br>
            <span style="font-size:2rem; color:#2980b9; font-weight:800">{top5_share:.1f}%</span><br>
            <span style="font-size:0.85rem">of total occurrences</span>
        </div>''', unsafe_allow_html=True)
    with qual_cols[2]:
        st.markdown(f'''<div class="metric-card">
            <strong>Normalized Entropy</strong><br>
            <span style="font-size:2rem; color:#f39c12; font-weight:800">{normalized_entropy:.3f}</span><br>
            <span style="font-size:0.85rem">0=concentrated, 1=uniform</span>
        </div>''', unsafe_allow_html=True)
    with qual_cols[3]:
        singletons = sum(1 for c in dataA.values() if c == 1)
        st.markdown(f'''<div class="metric-card metric-card-red">
            <strong>Singleton Terms</strong><br>
            <span style="font-size:2rem; color:#c0392b; font-weight:800">{singletons}</span><br>
            <span style="font-size:0.85rem">{singletons/n_terms*100:.1f}% of all terms</span>
        </div>''', unsafe_allow_html=True)

    # Quality insight
    if normalized_entropy < 0.5:
        st.markdown("""
        <div class="insight-box">
            <strong>💡 Low Entropy Insight:</strong> The extraction is <strong>highly concentrated</strong> on a few key terms. 
            This is typical under a <strong>10,000-character limit</strong> — the model prioritizes the most salient quantities 
            and may miss peripheral ones. This suggests <strong>high precision</strong> but potentially <strong>lower recall breadth</strong>.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="insight-box">
            <strong>💡 Balanced Distribution:</strong> The extraction shows a <strong>relatively even distribution</strong> 
            across terms despite the 10k-character constraint. This suggests the model is effectively identifying 
            a <strong>diverse set of quantities</strong> even with limited context.
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# RAW DATA TABLE
# ─────────────────────────────────────────────────────────────
if show_tables:
    st.markdown("---")
    st.markdown("### 📋 Raw Extraction Data")

    df = pd.DataFrame(list(dataA.items()), columns=["Term", "OccurrenceCount"])
    df["Category"] = df["Term"].apply(get_category)
    df = df.sort_values("OccurrenceCount", ascending=False).reset_index(drop=True)
    df["Term_Display"] = df["Term"].str.replace("_", " ").str.title()
    df_display = df[["Term_Display", "OccurrenceCount", "Category"]].rename(columns={"Term_Display": "Term"})

    st.info("📌 **Context Limit Reminder:** This model was evaluated with a **10,000-character** input limit. Lower counts may reflect the reduced context window rather than inferior extraction capability.")
    st.dataframe(df_display, use_container_width=True, height=500, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Model A Annotated CSV", csv, 
                      "physical_quantities_modelA_annotated.csv", "text/csv")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem; padding:1rem 0;">
    <strong>Model A (Falcon 10B) — Single-Model Analysis</strong><br>
    Results obtained under a <strong>10,000-character context limit</strong>. For cross-model comparison, 
    see the full 4-model dashboard where Models B, C, and D use 50,000 characters.<br><br>
    <em>Shorter contexts favor precision over recall. Interpret coverage metrics in light of this constraint.</em>
</div>
""", unsafe_allow_html=True)
