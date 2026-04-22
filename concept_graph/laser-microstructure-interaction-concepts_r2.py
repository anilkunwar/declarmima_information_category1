# app.py - Single-file Laser & Multicomponent Alloy Microstructure Concept Graph Analyzer
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
import tempfile
import warnings
from collections import defaultdict
from sklearn.linear_model import Ridge
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyvis.network import Network
import traceback

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION (inline)
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
EMBED_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Domain: Laser Processing + Multicomponent Alloys
DOMAIN_CONFIG = {
    "target_metrics": [
        "grain size", "phase fraction", "microhardness", "tensile strength",
        "yield strength", "elongation", "residual stress", "texture intensity",
        "cooling rate", "melt pool depth", "dilution ratio", "crack density"
    ],
    "alloy_patterns": [
        r'[A-Z][a-z]?(?:\d+(?:\.\d+)?(?:[A-Z][a-z]?\d*(?:\.\d+)?)*)+',
        r'(?:Ni|Co|Cr|Fe|Al|Ti|Cu|Nb|Mo|W)(?:-\d+(?:\.\d+)?%?)+',
        r'(?:high-entropy|HEA|multi-principal|complex concentrated)',
    ],
    "laser_params": [
        "laser power", "scan speed", "hatch spacing", "layer thickness",
        "pulse duration", "energy density", "spot diameter", "beam mode"
    ],
}

MIN_CONCEPT_FREQ = 3
MIN_CONCEPT_LENGTH_WORDS = 2
GNN_HIDDEN_DIM = 128
TRAIN_EPOCHS = 50
LR = 1e-3
NEG_DPREV_FOCUS = 3

# ==========================================
# UTILITY FUNCTIONS (normalization, extraction, etc.)
# ==========================================
def normalize_alloy_composition(concept: str) -> str:
    """Standardize alloy notation (e.g., 'Ti-6Al-4V' → 'ti6al4v')"""
    normalized = re.sub(r'[\s\-_]', '', concept).lower()
    normalized = re.sub(r'(ti)(6)(al)(4)(v)', r'ti6al4v', normalized)
    normalized = re.sub(r'(al)(si)(10)(mg)', r'alsi10mg', normalized)
    return normalized

def normalize_laser_term(concept: str) -> str:
    """Normalize laser processing terminology and units"""
    concept = concept.lower().strip()
    concept = re.sub(r'\b(j/mm(?:\s*3)?|j mm-3|j mm⁻³)\b', 'j/mm³', concept)
    concept = re.sub(r'\b(w|watt)s?\b', 'w', concept)
    concept = re.sub(r'\b(mm/s|mm s-1|mm s⁻¹)\b', 'mm/s', concept)
    return concept

def is_valid_microstructure_concept(concept: str) -> bool:
    """Filter concepts relevant to laser/alloy microstructure research"""
    concept_lower = concept.lower()
    domain_keywords = (
        DOMAIN_CONFIG["target_metrics"] +
        DOMAIN_CONFIG["laser_params"] +
        ["microstructure", "phase", "grain", "precipitate", "solidification",
         "eutectic", "martensite", "austenite", "segregation", "texture"]
    )
    has_domain_keyword = any(kw in concept_lower for kw in domain_keywords)
    generic_terms = {'study', 'analysis', 'effect', 'role', 'investigation', 'research'}
    has_generic = any(term in concept_lower.split() for term in generic_terms)
    return has_domain_keyword and not has_generic and len(concept.split()) >= 2

def generate_embeddings(valid_concepts, embed_model):
    embeddings = embed_model.encode(valid_concepts, show_progress_bar=False)
    return torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)

# ==========================================
# STEP 1-2: CONCEPT EXTRACTION & NORMALIZATION
# ==========================================
def extract_concepts_from_abstracts(abstracts, tokenizer, model):
    prompt_template = """Extract exactly the core scientific concepts (2+ words) from this abstract about laser processing or alloy microstructure.
Rules:
- Output ONLY a JSON list of strings.
- Use nominalized form (e.g., 'grain refinement' not 'refines grains').
- Include: alloy compositions (e.g., 'AlSi10Mg'), laser parameters ('laser power'), microstructure features ('columnar grains'), properties ('microhardness').
- Standardize: chemical formulas, units (J/mm³, mm/s), phase names.
- Exclude: generic terms like 'study', 'results', 'method'.

Abstract: {text}
Concepts:"""

    all_concepts = []
    all_metrics = []

    for text in abstracts:
        metrics = {}
        # Grain size patterns
        grain_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:μm|micron|nm)\s*(?:grain|average|size)', text, re.I)
        if grain_matches:
            metrics['grain_size_um'] = [float(m) for m in grain_matches]
        # Hardness/strength patterns
        mech_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:HV|GPa|MPa|ksi)\s*(?:hardness|strength|yield)', text, re.I)
        if mech_matches:
            metrics['mechanical_property'] = [float(m) for m in mech_matches]
        # Laser energy density
        energy_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:J/mm³|J mm-3|J mm⁻³)', text, re.I)
        if energy_matches:
            metrics['energy_density'] = [float(m) for m in energy_matches]
        all_metrics.append(metrics)

        prompt = prompt_template.format(text=text)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=150,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        concepts = []
        try:
            parsed = json.loads(response.replace("'", '"').strip())
            if isinstance(parsed, list):
                concepts = [c.strip().lower().rstrip('.') for c in parsed if isinstance(c, str)]
        except:
            # Fallback regex
            patterns = [
                r'\b(?:[A-Z][a-z]+(?:\d+(?:\.\d+)?)?[\s\-]?){2,3}(?:phase|grain|microstructure|strength)',
                r'\b(?:laser|powder|bed|fusion|selective)\s+(?:power|speed|scanning|melting|parameters)',
                r'\b(?:columnar|equiaxed|fine|coarse|nanoscale)\s+(?:grain|structure|region)',
            ]
            for pat in patterns:
                concepts.extend(re.findall(pat, text, re.I))
            concepts = list(set(c.lower() for c in concepts))

        # Normalize and filter
        normalized = []
        for c in concepts:
            if any(x in c for x in ['al', 'ti', 'ni', 'fe', 'co', 'cr']):
                c = normalize_alloy_composition(c)
            elif any(lp in c for lp in DOMAIN_CONFIG["laser_params"]):
                c = normalize_laser_term(c)
            if is_valid_microstructure_concept(c):
                normalized.append(c)
        all_concepts.append(normalized)

    return all_concepts, all_metrics

# ==========================================
# STEP 3: GRAPH BUILDING & SAMPLING
# ==========================================
def build_concept_graph(all_concepts, concept_to_id):
    nx_graph = nx.Graph()
    for c in concept_to_id:
        nx_graph.add_node(c)
    for concepts in all_concepts:
        valid_in_doc = [c for c in concepts if c in concept_to_id]
        for i in range(len(valid_in_doc)):
            for j in range(i+1, len(valid_in_doc)):
                u, v = valid_in_doc[i], valid_in_doc[j]
                if nx_graph.has_edge(u, v):
                    nx_graph[u][v]['weight'] += 1
                else:
                    nx_graph.add_edge(u, v, weight=1)
    valid_nodes = [n for n, d in nx_graph.degree() if d >= MIN_CONCEPT_FREQ]
    graph_filtered = nx_graph.subgraph(valid_nodes).copy()
    d_prev_dict = dict(nx.all_pairs_shortest_path_length(graph_filtered, cutoff=4))
    return graph_filtered, d_prev_dict

def sample_edges_for_training(nx_graph, d_prev_dict, valid_concepts, concept_to_id):
    pos_pairs = [(concept_to_id[u], concept_to_id[v]) for u, v in nx_graph.edges()]
    neg_pairs = []
    valid_ids = list(range(len(valid_concepts)))
    n_nodes = len(valid_ids)
    target_negs = min(len(pos_pairs) * 2, 2000)
    attempts = 0
    while len(neg_pairs) < target_negs and attempts < 15000:
        u_idx, v_idx = np.random.choice(n_nodes, 2, replace=False)
        u_concept, v_concept = valid_concepts[u_idx], valid_concepts[v_idx]
        if nx_graph.has_edge(u_concept, v_concept):
            attempts += 1
            continue
        try:
            dist = d_prev_dict[u_concept][v_concept]
            if dist == NEG_DPREV_FOCUS:
                neg_pairs.append((u_idx, v_idx))
            elif dist == 2 and np.random.rand() < 0.3:
                neg_pairs.append((u_idx, v_idx))
        except KeyError:
            if np.random.rand() < 0.05:
                neg_pairs.append((u_idx, v_idx))
        attempts += 1
    while len(neg_pairs) < target_negs:
        u_idx, v_idx = np.random.choice(n_nodes, 2, replace=False)
        if not nx_graph.has_edge(valid_concepts[u_idx], valid_concepts[v_idx]):
            neg_pairs.append((u_idx, v_idx))
    return pos_pairs, neg_pairs

# ==========================================
# STEP 4: PURE PYTORCH SPARSE GRAPHSAGE
# ==========================================
class SparseGraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim=GNN_HIDDEN_DIM):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, adj_indices, adj_values, num_nodes, h, pos_u, pos_v, neg_u, neg_v):
        A = sparse.FloatTensor(adj_indices, adj_values, torch.Size([num_nodes, num_nodes])).to(h.device)
        deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1)
        deg_inv = 1.0 / deg
        h1 = F.relu(self.lin1(torch.sparse.mm(A, h) * deg_inv.unsqueeze(1)))
        h2 = self.lin2(torch.sparse.mm(A, h1) * deg_inv.unsqueeze(1))
        pos_scores = self.decoder(torch.cat([h2[pos_u], h2[pos_v]], dim=1)).squeeze(1)
        neg_scores = self.decoder(torch.cat([h2[neg_u], h2[neg_v]], dim=1)).squeeze(1)
        return pos_scores, neg_scores, h2

def train_gnn(node_features, nx_graph, concept_to_id, pos_pairs, neg_pairs, progress_callback=None):
    unique_edges = {(min(u, v), max(u, v)) for u, v in pos_pairs}
    src_adj = torch.tensor([u for u, v in unique_edges], dtype=torch.long)
    dst_adj = torch.tensor([v for u, v in unique_edges], dtype=torch.long)
    adj_indices = torch.stack([src_adj, dst_adj], dim=0)
    adj_values = torch.ones(adj_indices.shape[1], dtype=torch.float32)

    pos_u = torch.tensor([p[0] for p in pos_pairs], dtype=torch.long, device=DEVICE)
    pos_v = torch.tensor([p[1] for p in pos_pairs], dtype=torch.long, device=DEVICE)
    neg_u = torch.tensor([n[0] for n in neg_pairs], dtype=torch.long, device=DEVICE)
    neg_v = torch.tensor([n[1] for n in neg_pairs], dtype=torch.long, device=DEVICE)

    model = SparseGraphSAGE(node_features.shape[1]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(TRAIN_EPOCHS):
        model.train()
        optimizer.zero_grad()
        pos_out, neg_out, _ = model(adj_indices, adj_values, len(concept_to_id), node_features,
                                    pos_u, pos_v, neg_u, neg_v)
        loss = 0.5 * (criterion(pos_out, torch.ones_like(pos_out)) + criterion(neg_out, torch.zeros_like(neg_out)))
        loss.backward()
        optimizer.step()
        if progress_callback and epoch % 10 == 0:
            progress_callback(epoch, loss.item())
    model.eval()
    with torch.no_grad():
        _, _, final_embeddings = model(adj_indices, adj_values, len(concept_to_id), node_features,
                                       pos_u[:1], pos_v[:1], neg_u[:1], neg_v[:1])
    return model, final_embeddings.cpu(), adj_indices, adj_values

# ==========================================
# STEP 5: QUANTIFICATION & SCORING
# ==========================================
def compute_microstructure_quantification(valid_concepts, concept_abstract_map, all_metrics, nx_graph):
    concept_properties = {}
    for c in valid_concepts:
        doc_indices = concept_abstract_map[c]
        values = []
        for idx in doc_indices:
            for metric_values in all_metrics[idx].values():
                values.extend(metric_values)
        concept_properties[c] = np.median(values) if values else 0.0

    X_feat, y_target = [], []
    for u, v in nx_graph.edges():
        pu, pv = concept_properties.get(u, 0), concept_properties.get(v, 0)
        w = nx_graph[u][v]['weight']
        X_feat.append([pu, pv, w])
        y_target.append(max(pu, pv) * 1.08 if max(pu, pv) > 0 else 0)
    ridge = Ridge(alpha=1.0).fit(np.array(X_feat), np.array(y_target)) if len(X_feat) > 10 else None
    return concept_properties, ridge

def compute_research_direction_scores(model, final_emb, nx_graph, valid_concepts, concept_properties,
                                      ridge, embed_model, d_prev_dict, adj_indices, adj_values, n_samples=3000):
    n_concepts = len(valid_concepts)
    u_ids = np.random.randint(n_concepts, size=n_samples)
    v_ids = np.random.randint(n_concepts, size=n_samples)
    candidate_pairs = []
    for u_idx, v_idx in zip(u_ids, v_ids):
        if u_idx == v_idx: continue
        u_c, v_c = valid_concepts[u_idx], valid_concepts[v_idx]
        if nx_graph.has_edge(u_c, v_c): continue
        candidate_pairs.append((u_idx, v_idx, u_c, v_c))
    if not candidate_pairs:
        return pd.DataFrame()

    u_tensor = torch.tensor([p[0] for p in candidate_pairs], dtype=torch.long, device=DEVICE)
    v_tensor = torch.tensor([p[1] for p in candidate_pairs], dtype=torch.long, device=DEVICE)
    model.eval()
    with torch.no_grad():
        _, _, h2 = model(adj_indices, adj_values, n_concepts, final_emb.to(DEVICE),
                         u_tensor, v_tensor, u_tensor, v_tensor)
        pair_feat = torch.cat([h2, h2], dim=1)
        gnn_logits = model.decoder(pair_feat).squeeze(1)
        gnn_scores = torch.sigmoid(gnn_logits).cpu().numpy()

    emb_np = embed_model.encode(valid_concepts, show_progress_bar=False)
    cos_sims = np.sum(emb_np[u_tensor.cpu().numpy()] * emb_np[v_tensor.cpu().numpy()], axis=1)

    results = []
    for i, (u_idx, v_idx, u_c, v_c) in enumerate(candidate_pairs):
        try:
            d_prev = d_prev_dict[u_c][v_c]
        except KeyError:
            d_prev = 4
        if d_prev < 2: continue
        p_u = concept_properties.get(u_c, 0)
        p_v = concept_properties.get(v_c, 0)
        expected_gain = 0
        if ridge is not None and (p_u > 0 or p_v > 0):
            expected_gain = float(ridge.predict([[p_u, p_v, 1.0]])[0])
        semantic_novelty = 1.0 - cos_sims[i]
        feasibility = np.exp(-0.5 * semantic_novelty) * (1.0 if (p_u > 0 or p_v > 0) else 0.6)
        norm_gain = np.clip((expected_gain - 50) / 200, 0, 1)
        D_uv = (0.4 * gnn_scores[i] + 0.3 * semantic_novelty + 0.2 * norm_gain - 0.1 * (1.0 - feasibility))
        results.append({
            'concept_u': u_c, 'concept_v': v_c,
            'graph_distance': d_prev,
            'gnn_affinity': float(gnn_scores[i]),
            'semantic_novelty': float(semantic_novelty),
            'expected_property_gain': expected_gain,
            'feasibility_score': float(feasibility),
            'composite_score': float(D_uv)
        })
    df = pd.DataFrame(results).sort_values('composite_score', ascending=False)
    return df.head(50)

# ==========================================
# STEP 6: LLM CURATION
# ==========================================
def generate_research_directions(top_pairs_df, tokenizer, model):
    prompt_template = """You are a materials science strategist specializing in laser processing of multicomponent alloys.
For the novel concept combination: "{u}" + "{v}"
Associated property context: ~{prop:.1f} (e.g., HV, μm, MPa)
Feasibility estimate: {feas:.2f}/1.0

Write exactly 3 concise, technically precise sentences:
1. Scientific novelty: Why this combination is underexplored in laser alloy processing.
2. Target outcome: Predicted microstructure/property improvement and key trade-off (e.g., strength vs. ductility).
3. Validation step: One concrete experimental method (e.g., EBSD for texture, nanoindentation, in-situ synchrotron XRD).

Avoid generic statements. Focus on laser-matter interaction, solidification, or phase transformation mechanisms."""
    results = []
    for _, row in top_pairs_df.iterrows():
        prompt = prompt_template.format(
            u=row['concept_u'].title(),
            v=row['concept_v'].title(),
            prop=float(row['expected_property_gain']),
            feas=float(row['feasibility_score'])
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=180,
                temperature=0.25,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        results.append({
            'Concept Pair': f"{row['concept_u']} + {row['concept_v']}",
            'Composite Score': f"{row['composite_score']:.3f}",
            'Expected Gain': f"{row['expected_property_gain']:.1f}",
            'Feasibility': f"{row['feasibility_score']:.2f}",
            'Research Hypothesis': response
        })
    return pd.DataFrame(results)

# ==========================================
# STREAMLIT UI & PIPELINE ORCHESTRATION
# ==========================================
@st.cache_resource(show_spinner=False)
def load_models():
    embed_model = SentenceTransformer(EMBED_NAME, device=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    llm.eval()
    return embed_model, tokenizer, llm

def run_pipeline(abstracts, progress_callback=None):
    embed_model, tokenizer, llm = load_models()
    if progress_callback: progress_callback(10, "Models loaded")

    all_concepts, all_metrics = extract_concepts_from_abstracts(abstracts, tokenizer, llm)
    if progress_callback: progress_callback(25, "Concepts extracted")

    concept_counts = defaultdict(int)
    concept_abstract_map = defaultdict(list)
    for doc_idx, concepts in enumerate(all_concepts):
        seen = set()
        for c in concepts:
            if c not in seen and is_valid_microstructure_concept(c):
                concept_counts[c] += 1
                concept_abstract_map[c].append(doc_idx)
                seen.add(c)
    valid_concepts = [c for c, cnt in concept_counts.items() if cnt >= MIN_CONCEPT_FREQ and len(c.split()) >= MIN_CONCEPT_LENGTH_WORDS]
    concept_to_id = {c: i for i, c in enumerate(valid_concepts)}
    if len(valid_concepts) < 5:
        raise ValueError("Too few valid concepts. Add more abstracts or reduce frequency filter.")
    if progress_callback: progress_callback(40, f"{len(valid_concepts)} concepts validated")

    nx_graph, d_prev_dict = build_concept_graph(all_concepts, concept_to_id)
    pos_pairs, neg_pairs = sample_edges_for_training(nx_graph, d_prev_dict, valid_concepts, concept_to_id)
    if progress_callback: progress_callback(55, f"Graph: {len(valid_concepts)} nodes, {nx_graph.number_of_edges()} edges")

    node_features = generate_embeddings(valid_concepts, embed_model)
    if progress_callback: progress_callback(65, "Embeddings generated")

    def _progress(epoch, loss):
        if progress_callback:
            progress_callback(65 + epoch * 0.2, f"Training GNN | Loss: {loss:.4f}")
    gnn_model, final_emb, adj_indices, adj_values = train_gnn(
        node_features, nx_graph, concept_to_id, pos_pairs, neg_pairs, _progress
    )
    if progress_callback: progress_callback(80, "GNN training complete")

    concept_properties, ridge = compute_microstructure_quantification(valid_concepts, concept_abstract_map, all_metrics, nx_graph)
    top_scores = compute_research_direction_scores(
        gnn_model, final_emb, nx_graph, valid_concepts, concept_properties,
        ridge, embed_model, d_prev_dict, adj_indices, adj_values
    )
    if progress_callback: progress_callback(90, "Scoring complete")

    directions_df = generate_research_directions(top_scores, tokenizer, llm)
    if progress_callback: progress_callback(100, "Pipeline complete")
    return {'graph': nx_graph, 'valid_concepts': valid_concepts, 'top_directions': directions_df}

def main():
    st.set_page_config(page_title="Laser & Alloy Microstructure Concept Graph", layout="wide")
    st.title("🔬 Laser & Multicomponent Alloy Microstructure Analyzer")
    st.caption("Discover novel research directions via concept graph + lightweight LLM + GraphSAGE")

    abstract_input = st.text_area(
        "Paste scientific abstracts (separate by blank lines):",
        height=250,
        placeholder="Example:\n\nLaser powder bed fusion of AlSi10Mg reveals columnar-to-equiaxed transition at 80 J/mm³...\n\nSelective laser melting of Ti6Al4V results in martensitic α' phase with microhardness 420 HV..."
    )
    if st.button("🚀 Analyze Abstracts", type="primary"):
        if not abstract_input.strip():
            st.error("Please enter at least one abstract.")
            return
        abstracts = [t.strip() for t in re.split(r'\n\s*\n', abstract_input) if t.strip()]
        if len(abstracts) < 5:
            st.warning("For best results use 10-25 abstracts. Proceeding with available.")
        progress_bar = st.progress(0)
        status = st.status("Running pipeline...", expanded=True)

        def update_progress(percent, msg):
            progress_bar.progress(percent)
            status.write(f"✓ {msg}")

        try:
            results = run_pipeline(abstracts, update_progress)
            status.update(label="✅ Analysis complete!", state="complete", expanded=False)
            st.subheader("🎯 Top Research Hypotheses")
            st.dataframe(results['top_directions'], use_container_width=True)

            st.subheader("🕸️ Concept Relationship Graph")
            net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
            net.barnes_hut(gravity=-80000, spring_length=200)
            for node in results['graph'].nodes():
                deg = results['graph'].degree(node)
                size = max(10, min(40, deg * 3))
                net.add_node(node, label=node, size=size, title=f"Degree: {deg}")
            for u, v in results['graph'].edges():
                w = results['graph'][u][v]['weight']
                net.add_edge(u, v, value=w, width=min(3, w * 0.5))
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                net.save_graph(tmp.name)
                with open(tmp.name, "r", encoding="utf-8") as f:
                    st.components.v1.html(f.read(), height=650, scrolling=True)
        except Exception as e:
            st.error(f"Pipeline failed: {str(e)}")
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
