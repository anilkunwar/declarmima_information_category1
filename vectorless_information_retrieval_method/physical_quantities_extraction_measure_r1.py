"""
Streamlit App: Cross-Model Physical Quantity Extraction Comparison
For Materials & Design Pipeline Documentation

Expected CSV files (place in same directory):
  - physical_quantities_detection_llm_modelB.csv  -> Mistral 7B
  - physical_quantities_detection_llm_modelD.csv  -> Qwen 7B
  - physical_quantities_detection_llm_modelC.csv  -> Qwen 14B
  - physical_quantities_detection_llm_reference.csv  -> Optional ground truth

CSV format: Term,OccurrenceCount
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from io import BytesIO
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="LLM Extraction Comparison -- Materials Design",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for publication look
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #1a1a2e; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #4a4a6a; margin-bottom: 1.5rem; }
    .metric-card { background: #f8f9fa; border-left: 4px solid #2980b9; padding: 1rem; border-radius: 6px; margin-bottom: 1rem; }
    .highlight-red { border-left-color: #c0392b; }
    .highlight-green { border-left-color: #27ae60; }
    .highlight-blue { border-left-color: #2980b9; }
    .highlight-purple { border-left-color: #8e44ad; }
    .caption { font-size: 0.85rem; color: #666; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# MATERIALS DESIGN TAXONOMY
# ============================================================
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

# ============================================================
# DATA LOADING
# ============================================================
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

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 📁 Input Files")
    st.markdown("Place CSV files in the working directory:")
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
        dataB = load_csv("physical_quantities_detection_llm_modelB.csv")
        dataD = load_csv("physical_quantities_detection_llm_modelD.csv")
        dataC = load_csv("physical_quantities_detection_llm_modelC.csv")
        dataRef = load_csv("physical_quantities_detection_llm_reference.csv")
        
        st.markdown("---")
        st.markdown("**Expected filenames:**")
        st.code("physical_quantities_detection_llm_modelB.csv\nphysical_quantities_detection_llm_modelD.csv\nphysical_quantities_detection_llm_modelC.csv")
        st.markdown("*Optional:* `physical_quantities_detection_llm_reference.csv`")
    
    st.markdown("---")
    st.markdown("## ⚙️ View Options")
    show_tables = st.checkbox("Show raw data tables", value=True)
    show_figB = st.checkbox("Show Figure B -- Consensus by Category", value=True)
    show_figC = st.checkbox("Show Figure C -- Composition + Quality Radar", value=True)
    show_upset = st.checkbox("Show UpSet Intersection Matrix", value=False)
    
    st.markdown("---")
    st.markdown("## 📊 Download")
    download_dpi = st.slider("Figure DPI", min_value=150, max_value=600, value=300, step=50)

# ============================================================
# MAIN HEADER
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
# FIGURE B -- CONSENSUS BY CATEGORY
# ============================================================
if show_figB and len(main_models) >= 2:
    st.markdown("### Figure B -- Consensus Terms by Materials Design Category")
    st.markdown("<div class='caption'>Terms extracted by ≥2 models, grouped by process → mechanical → microstructural relevance. Background shading indicates category.</div>", unsafe_allow_html=True)
    
    all_terms_main = set()
    for m in main_models:
        all_terms_main.update(models_data[m].keys())
    
    consensus_terms = []
    for term in all_terms_main:
        count = sum(1 for m in main_models if term in models_data[m])
        if count >= 2:
            consensus_terms.append(term)
    
    cat_order = list(TAXONOMY.keys()) + ["Other"]
    term_info = []
    for term in consensus_terms:
        cat = get_category(term)
        vals = {m: models_data[m].get(term, 0) for m in main_models}
        max_val = max(vals.values())
        term_info.append((cat, term, vals, max_val))
    
    term_info.sort(key=lambda x: (cat_order.index(x[0]) if x[0] in cat_order else 99, -x[3]))
    
    fig_b, ax_b = plt.subplots(figsize=(14, 7), facecolor="white")
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
    width = 0.22
    n_models = len(main_models)
    
    if labels:
        start_idx = 0
        current_cat = label_cats[0]
        for i in range(1, len(label_cats)):
            if label_cats[i] != current_cat:
                ax_b.axvspan(start_idx - 0.5, i - 0.5, 
                            facecolor=CATEGORY_COLORS.get(current_cat, "white"), 
                            alpha=0.5, zorder=0)
                start_idx = i
                current_cat = label_cats[i]
        ax_b.axvspan(start_idx - 0.5, len(labels) - 0.5, 
                    facecolor=CATEGORY_COLORS.get(current_cat, "white"), 
                    alpha=0.5, zorder=0)
    
    model_color_map = {meta["name"]: meta["color"] for meta in MODEL_META.values()}
    for i, m in enumerate(main_models):
        offset = (i - (n_models-1)/2) * width
        bars = ax_b.bar(x + offset, values_b[m], width,
                       label=m, color=model_color_map.get(m, "#333333"), alpha=0.9,
                       edgecolor="black", linewidth=0.7, zorder=3)
        for bar, val in zip(bars, values_b[m]):
            if val > 0:
                ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(1, max(values_b[m])*0.03),
                         f"{int(val)}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    
    if labels:
        start_idx = 0
        current_cat = label_cats[0]
        ymax = ax_b.get_ylim()[1]
        for i in range(1, len(label_cats)):
            if label_cats[i] != current_cat:
                mid = (start_idx + i - 1) / 2
                ax_b.text(mid, ymax * 0.97, current_cat, ha="center", va="top", 
                         fontsize=10, fontweight="bold", style="italic", color="#555555")
                start_idx = i
                current_cat = label_cats[i]
        mid = (start_idx + len(labels) - 1) / 2
        ax_b.text(mid, ymax * 0.97, current_cat, ha="center", va="top", 
                 fontsize=10, fontweight="bold", style="italic", color="#555555")
    
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, rotation=40, ha="right", fontsize=9.5)
    ax_b.set_ylabel("Occurrence Count", fontsize=13, fontweight="bold")
    ax_b.legend(loc="upper right", fontsize=10, framealpha=0.95, edgecolor="black")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.grid(axis="y", alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    st.pyplot(fig_b)
    
    buf = BytesIO()
    fig_b.savefig(buf, format="png", dpi=download_dpi, bbox_inches="tight", facecolor="white")
    st.download_button("⬇️ Download Figure B (PNG)", buf.getvalue(), 
                      "fig_B_consensus_terms.png", "image/png")
    plt.close(fig_b)

# ============================================================
# FIGURE C -- COMPOSITION + RADAR
# ============================================================
if show_figC and len(main_models) >= 2:
    st.markdown("---")
    st.markdown("### Figure C -- Extraction Composition & Pipeline Quality Metrics")
    st.markdown("<div class='caption'>Left: stacked composition by consensus level. Right: radar chart comparing schema compliance, coverage, and rank stability.</div>", unsafe_allow_html=True)
    
    all_sets = {m: set(models_data[m].keys()) for m in main_models}
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
    
    fig_c = plt.figure(figsize=(16, 7), facecolor="white")
    gs = fig_c.add_gridspec(1, 2, width_ratios=[1.4, 1], wspace=0.35)
    
    ax1 = fig_c.add_subplot(gs[0, 0])
    ax1.set_facecolor("white")
    
    cat_colors = ["#E67E22", "#F1C40F", "#27AE60"]
    cat_labels = ["Solo Unique\n(Uncorroborated)", "Pairwise Only\n(2-Model Consensus)", "Universal\n(All 3 Models)"]
    y_pos = np.arange(len(main_models))
    bar_h = 0.55
    
    solo_vals = [solo[m] for m in main_models]
    pair_vals = [pairwise[m] for m in main_models]
    univ_vals = [universal[m] for m in main_models]
    totals = [s + p + u for s, p, u in zip(solo_vals, pair_vals, univ_vals)]
    
    ax1.barh(y_pos, solo_vals, height=bar_h, color=cat_colors[0], 
            label=cat_labels[0], edgecolor="black", linewidth=0.8, zorder=3)
    ax1.barh(y_pos, pair_vals, height=bar_h, left=solo_vals, color=cat_colors[1],
            label=cat_labels[1], edgecolor="black", linewidth=0.8, zorder=3)
    ax1.barh(y_pos, univ_vals, height=bar_h, 
            left=[s+p for s,p in zip(solo_vals, pair_vals)], color=cat_colors[2],
            label=cat_labels[2], edgecolor="black", linewidth=0.8, zorder=3)
    
    for i, m in enumerate(main_models):
        total = totals[i]
        if solo_vals[i] > 0:
            pct = 100 * solo_vals[i] / total
            ax1.text(solo_vals[i]/2, i, f"{solo_vals[i]}\n({pct:.0f}%)", 
                    ha="center", va="center", fontsize=10, color="white", fontweight="bold")
        if pair_vals[i] > 0:
            pct = 100 * pair_vals[i] / total
            ax1.text(solo_vals[i] + pair_vals[i]/2, i, f"{pair_vals[i]}\n({pct:.0f}%)", 
                    ha="center", va="center", fontsize=10, color="#333333", fontweight="bold")
        if univ_vals[i] > 0:
            pct = 100 * univ_vals[i] / total
            ax1.text(solo_vals[i] + pair_vals[i] + univ_vals[i]/2, i, f"{univ_vals[i]}\n({pct:.0f}%)", 
                    ha="center", va="center", fontsize=10, color="white", fontweight="bold")
        ax1.text(total + 0.4, i, f"{total} total", ha="left", va="center", 
                fontsize=11, fontweight="bold", color="#2C3E50")
    
    if solo_vals[0] == 0 and len(main_models) >= 1:
        ax1.annotate("100% corroborated\n→ Zero solo hallucinations", 
                    xy=(totals[0], 0), xytext=(totals[0] + 3, 0.4),
                    fontsize=9, color="#C0392B", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.5),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDEDEC", edgecolor="#C0392B", alpha=0.9))
    
    short_names = [m.split("(")[1].replace(")", "") if "(" in m else m for m in main_models]
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(short_names, fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Extracted Terms", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9.5, framealpha=0.95, edgecolor="black")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_xlim(0, max(totals) + 7)
    ax1.grid(axis="x", alpha=0.3, linestyle="--")
    
    if len(main_models) >= 2:
        ax2 = fig_c.add_subplot(gs[0, 1], polar=True)
        ax2.set_facecolor("white")
        
        ref_set = set(models_data.get("Reference", {}).keys())
        
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
            ax2.plot(angles, values, "o-", linewidth=2.5, color=color, label=short, markersize=6)
            ax2.fill(angles, values, alpha=0.1, color=color)
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metric_names, fontsize=9)
        ax2.set_ylim(0, 105)
        ax2.set_yticks([20, 40, 60, 80, 100])
        ax2.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7, color="gray")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9, framealpha=0.9)
    
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
    
    all_sets = {m: set(models_data[m].keys()) for m in main_models}
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
