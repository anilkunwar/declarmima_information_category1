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
import tempfile
import warnings
import traceback
from collections import defaultdict
from sklearn.linear_model import Ridge
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyvis.network import Network

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION & DEVICE SETUP
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Model identifiers (<1B constraint)
LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
EMBED_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Domain: Laser Processing + Multicomponent Alloys
DOMAIN_KEYWORDS = [
    # Microstructure features
    "grain size", "phase fraction", "microhardness", "tensile strength", 
    "yield strength", "elongation", "residual stress", "texture intensity",
    "columnar grain", "equiaxed grain", "dendrite", "eutectic", "martensite",
    "austenite", "precipitate", "segregation", "porosity", "crack density",
    # Laser parameters
    "laser power", "scan speed", "hatch spacing", "layer thickness",
    "pulse duration", "energy density", "spot diameter", "melt pool",
    "cooling rate", "solidification rate", "dilution ratio",
    # Alloy terminology
    "high-entropy alloy", "HEA", "multi-principal element", "complex concentrated",
    "powder bed fusion", "LPBF", "direct energy deposition", "DED"
]

ALLOY_PATTERNS = [
    r'[A-Z][a-z]?(?:\d+(?:\.\d+)?(?:[A-Z][a-z]?\d*(?:\.\d+)?)*)+',  # AlSi10Mg, Ti6Al4V
    r'(?:Ni|Co|Cr|Fe|Al|Ti|Cu|Nb|Mo|W)(?:[-\s]?\d+(?:\.\d+)?%?)+',  # Ni-20Cr-10Mo
    r'(?:high-entropy|HEA|multi-principal|complex concentrated)',
]

# Pipeline hyperparameters
MIN_CONCEPT_FREQ = 3
MIN_CONCEPT_LENGTH_WORDS = 2
GNN_HIDDEN_DIM = 128
TRAIN_EPOCHS = 50
LR = 1e-3
NEG_DPREV_FOCUS = 3

# ==========================================
# MODEL LOADING (CACHED FOR STREAMLIT)
# ==========================================
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBED_NAME, device=DEVICE)

@st.cache_resource(show_spinner=False)
def load_lightweight_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_NAME, 
        torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return tokenizer, model

# ==========================================
# DOMAIN-SPECIFIC CONCEPT NORMALIZATION
# ==========================================
def normalize_alloy_composition(concept: str) -> str:
    """Standardize alloy notation (e.g., 'Ti-6Al-4V' → 'ti6al4v')"""
    normalized = re.sub(r'[\s\-_]', '', concept).lower()
    normalized = re.sub(r'(ti)(6)(al)(4)(v)', r'ti6al4v', normalized)
    normalized = re.sub(r'(al)(si)(10)(mg)', r'alsi10mg', normalized)
    normalized = re.sub(r'(inconel)(\s*718|718)', r'inconel718', normalized)
    return normalized

def normalize_laser_term(concept: str) -> str:
    """Normalize laser processing terminology and units"""
    concept = concept.lower().strip()
    concept = re.sub(r'\b(j/mm(?:\s*3)?|j mm-3|j mm⁻³)\b', 'j/mm³', concept)
    concept = re.sub(r'\b(w|watt)s?\b', 'w', concept)
    concept = re.sub(r'\b(mm/s|mm s-1|mm s⁻¹)\b', 'mm/s', concept)
    concept = re.sub(r'\b(μm|micron|um)\b', 'um', concept)
    return concept

def is_valid_microstructure_concept(concept: str) -> bool:
    """Filter concepts relevant to laser/alloy microstructure research"""
    concept_lower = concept.lower()
    
    # Must contain at least one domain keyword
    has_domain_keyword = any(kw in concept_lower for kw in DOMAIN_KEYWORDS)
    
    # Check for alloy patterns
    has_alloy_pattern = any(re.search(p, concept, re.I) for p in ALLOY_PATTERNS)
    
    # Exclude generic terms
    generic_terms = {'study', 'analysis', 'effect', 'role', 'investigation', 
                     'research', 'method', 'approach', 'paper', 'work'}
    has_generic = any(term in concept_lower.split() for term in generic_terms)
    
    return (has_domain_keyword or has_alloy_pattern) and not has_generic and len(concept.split()) >= 2

# ==========================================
# STEP 1-2: CONCEPT EXTRACTION & METRICS
# ==========================================
def extract_concepts_from_abstracts(abstracts, tokenizer, model):
    """Extract microstructure-relevant concepts and quantitative metrics"""
    
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
        # Extract quantitative microstructure metrics via regex
        metrics = {}
        
        # Grain size: "15.3 μm grain size" or "average grain: 20 micron"
        grain_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:μm|micron|um|nm)\s*(?:grain|average|size|diameter)?', text, re.I)
        if grain_matches:
            metrics['grain_size_um'] = [float(m) for m in grain_matches]
        
        # Mechanical properties: "450 HV hardness" or "850 MPa strength"
        mech_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:HV|GPa|MPa|ksi|GPa)\s*(?:hardness|strength|yield|tensile|ultimate)?', text, re.I)
        if mech_matches:
            metrics['mechanical_property'] = [float(m) for m in mech_matches]
        
        # Laser energy density: "85 J/mm³" or "120 J mm-3"
        energy_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:J/mm³|J mm-3|J mm⁻³|J/mm\^3)', text, re.I)
        if energy_matches:
            metrics['energy_density_j_mm3'] = [float(m) for m in energy_matches]
        
        # Porosity/crack density: "2.3% porosity" or "crack density: 0.15 mm/mm²"
        defect_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:porosity|void|crack)', text, re.I)
        if defect_matches:
            metrics['defect_fraction_pct'] = [float(m) for m in defect_matches]
        
        all_metrics.append(metrics)
        
        # LLM-based concept extraction
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
        
        # Parse concepts from JSON response
        concepts = []
        try:
            parsed = json.loads(response.replace("'", '"').strip())
            if isinstance(parsed, list):
                concepts = [c.strip().lower().rstrip('.') for c in parsed if isinstance(c, str) and len(c.strip()) > 3]
        except (json.JSONDecodeError, TypeError):
            # Fallback: regex extraction for domain terms
            concepts = _fallback_concept_extraction(text)
        
        # Normalize and filter concepts
        normalized = []
        for c in concepts:
            # Normalize alloy compositions
            if any(elem in c.lower() for elem in ['al', 'ti', 'ni', 'cr', 'fe', 'co', 'mo', 'nb', 'cu']):
                c = normalize_alloy_composition(c)
            # Normalize laser terms
            elif any(lp in c.lower() for lp in ['laser', 'scan', 'power', 'speed', 'melt', 'pool', 'energy']):
                c = normalize_laser_term(c)
            
            if is_valid_microstructure_concept(c):
                normalized.append(c)
        
        all_concepts.append(normalized)
    
    return all_concepts, all_metrics


def _fallback_concept_extraction(text: str) -> list:
    """Regex fallback for concept extraction when LLM parsing fails"""
    patterns = [
        # Alloy + microstructure: "AlSi10Mg columnar grains"
        r'\b(?:[A-Z][a-z]+(?:\d+(?:\.\d+)?)?[\s\-]?){2,3}(?:phase|grain|microstructure|strength|hardness)',
        # Laser parameters: "laser power", "scan speed", "energy density"
        r'\b(?:laser|powder|bed|fusion|selective|direct)\s+(?:power|speed|scanning|melting|parameters|energy)',
        # Microstructure descriptors: "columnar grains", "equiaxed structure"
        r'\b(?:columnar|equiaxed|fine|coarse|nanoscale|bimodal)\s+(?:grain|structure|region|zone)',
        # Phase names: "martensite formation", "eutectic phase"
        r'\b(?:martensite|austenite|ferrite|eutectic|peritectic|precipitate)\s+(?:formation|phase|fraction)',
        # Property terms: "microhardness improvement", "tensile strength"
        r'\b(?:microhardness|nanohardness|tensile|yield|ductility|elongation)\s+(?:improvement|strength|property)',
    ]
    concepts = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.I)
        concepts.extend([m.lower().strip() for m in matches if len(m.split()) >= 2])
    return list(set(concepts))


def normalize_and_filter_concepts(all_concepts):
    """Aggregate concepts across abstracts and filter by frequency"""
    concept_counts = defaultdict(int)
    concept_abstract_map = defaultdict(list)
    
    for doc_idx, concepts in enumerate(all_concepts):
        seen_in_doc = set()
        for c in concepts:
            if c not in seen_in_doc:
                concept_counts[c] += 1
                concept_abstract_map[c].append(doc_idx)
                seen_in_doc.add(c)
                
    valid_concepts = [c for c, cnt in concept_counts.items() 
                      if cnt >= MIN_CONCEPT_FREQ and len(c.split()) >= MIN_CONCEPT_LENGTH_WORDS]
    
    concept_to_id = {c: i for i, c in enumerate(valid_concepts)}
    id_to_concept = {i: c for i, c in enumerate(valid_concepts)}
    
    return valid_concepts, concept_to_id, id_to_concept, concept_abstract_map

# ==========================================
# STEP 3: TEMPORAL CONCEPT GRAPH & DISTANCE
# ==========================================
def build_concept_graph(all_concepts, concept_to_id):
    """Build co-occurrence graph with edge weights = concept co-appearance frequency"""
    nx_graph = nx.Graph()
    
    # Add nodes
    for c in concept_to_id:
        nx_graph.add_node(c)
        
    # Add weighted edges based on co-occurrence in abstracts
    for concepts in all_concepts:
        valid_in_doc = [c for c in concepts if c in concept_to_id]
        for i in range(len(valid_in_doc)):
            for j in range(i + 1, len(valid_in_doc)):
                u, v = valid_in_doc[i], valid_in_doc[j]
                if nx_graph.has_edge(u, v):
                    nx_graph[u][v]['weight'] += 1
                else:
                    nx_graph.add_edge(u, v, weight=1)
                    
    # Filter low-degree nodes for graph sparsity
    valid_nodes = [n for n, d in nx_graph.degree() if d >= MIN_CONCEPT_FREQ]
    graph_filtered = nx_graph.subgraph(valid_nodes).copy()
    
    # Precompute shortest path distances (for novelty scoring)
    d_prev_dict = dict(nx.all_pairs_shortest_path_length(graph_filtered, cutoff=4))
    
    return graph_filtered, d_prev_dict


def sample_edges_for_training(nx_graph, d_prev_dict, valid_concepts, concept_to_id):
    """Sample positive (existing) and negative (non-existing) edges for contrastive training"""
    pos_pairs = [(concept_to_id[u], concept_to_id[v]) for u, v in nx_graph.edges()]
    neg_pairs = []
    
    valid_ids = list(range(len(valid_concepts)))
    n_nodes = len(valid_ids)
    target_negs = min(len(pos_pairs) * 2, 2000)
    attempts = 0
    
    # Strategic negative sampling: focus on moderately distant concepts (novel but plausible)
    while len(neg_pairs) < target_negs and attempts < 15000:
        u_idx, v_idx = np.random.choice(n_nodes, 2, replace=False)
        u_concept, v_concept = valid_concepts[u_idx], valid_concepts[v_idx]
        
        # Skip if edge already exists
        if nx_graph.has_edge(u_concept, v_concept):
            attempts += 1
            continue
        
        # Prefer concepts at target graph distance (novel combinations)
        try:
            dist = d_prev_dict[u_concept][v_concept]
            if dist == NEG_DPREV_FOCUS:
                neg_pairs.append((u_idx, v_idx))
            elif dist == 2 and np.random.rand() < 0.3:  # Occasionally sample closer pairs
                neg_pairs.append((u_idx, v_idx))
        except KeyError:
            # Unconnected: low probability inclusion for exploration
            if np.random.rand() < 0.05:
                neg_pairs.append((u_idx, v_idx))
        
        attempts += 1
    
    # Fill remaining with random negatives
    while len(neg_pairs) < target_negs:
        u_idx, v_idx = np.random.choice(n_nodes, 2, replace=False)
        pair = (u_idx, v_idx)
        if pair not in neg_pairs and (v_idx, u_idx) not in neg_pairs:
            if not nx_graph.has_edge(valid_concepts[u_idx], valid_concepts[v_idx]):
                neg_pairs.append(pair)
    
    return pos_pairs, neg_pairs

# ==========================================
# STEP 4: SEMANTIC NODE EMBEDDINGS
# ==========================================
def generate_embeddings(valid_concepts, embed_model):
    """Generate sentence embeddings for concept nodes"""
    embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=32)
    return torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)

# ==========================================
# STEP 5: PURE PYTORCH SPARSE GRAPHSAGE
# ==========================================
class SparseGraphSAGE(nn.Module):
    """Memory-efficient GraphSAGE using PyTorch sparse tensors (no DGL/PyG)"""
    
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
        # Build sparse adjacency matrix: A
        A = sparse.FloatTensor(adj_indices, adj_values, torch.Size([num_nodes, num_nodes])).to(h.device)
        
        # Mean aggregation with degree normalization: h_neighbor = D^-1 @ A @ h
        deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1)
        deg_inv = 1.0 / deg
        
        # Two-layer GraphSAGE with ReLU activation
        h1 = F.relu(self.lin1(torch.sparse.mm(A, h) * deg_inv.unsqueeze(1)))
        h2 = self.lin2(torch.sparse.mm(A, h1) * deg_inv.unsqueeze(1))
        
        # Edge scoring: concatenate embeddings of endpoint nodes
        pos_scores = self.decoder(torch.cat([h2[pos_u], h2[pos_v]], dim=1)).squeeze(1)
        neg_scores = self.decoder(torch.cat([h2[neg_u], h2[neg_v]], dim=1)).squeeze(1)
        
        return pos_scores, neg_scores, h2


def train_gnn(node_features, nx_graph, concept_to_id, pos_pairs, neg_pairs, progress_callback=None):
    """Train GraphSAGE with contrastive edge prediction loss"""
    
    # Prepare adjacency for sparse matrix (symmetric, no self-loops)
    unique_edges = {(min(u, v), max(u, v)) for u, v in pos_pairs}
    src_adj = torch.tensor([u for u, v in unique_edges], dtype=torch.long)
    dst_adj = torch.tensor([v for u, v in unique_edges], dtype=torch.long)
    adj_indices = torch.stack([src_adj, dst_adj], dim=0)
    adj_values = torch.ones(adj_indices.shape[1], dtype=torch.float32)
    
    # Move training pairs to device
    pos_u = torch.tensor([p[0] for p in pos_pairs], dtype=torch.long, device=DEVICE)
    pos_v = torch.tensor([p[1] for p in pos_pairs], dtype=torch.long, device=DEVICE)
    neg_u = torch.tensor([n[0] for n in neg_pairs], dtype=torch.long, device=DEVICE)
    neg_v = torch.tensor([n[1] for n in neg_pairs], dtype=torch.long, device=DEVICE)
    
    # Initialize model
    model = SparseGraphSAGE(node_features.shape[1]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    for epoch in range(TRAIN_EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        pos_out, neg_out, _ = model(
            adj_indices, adj_values, len(concept_to_id), 
            node_features, pos_u, pos_v, neg_u, neg_v
        )
        
        pos_loss = criterion(pos_out, torch.ones_like(pos_out))
        neg_loss = criterion(neg_out, torch.zeros_like(neg_out))
        loss = 0.5 * (pos_loss + neg_loss)
        
        loss.backward()
        optimizer.step()
        
        if progress_callback and epoch % 10 == 0:
            progress_callback(epoch, loss.item())
    
    # Generate final embeddings (inference mode)
    model.eval()
    with torch.no_grad():
        _, _, final_embeddings = model(
            adj_indices, adj_values, len(concept_to_id),
            node_features, pos_u[:1], pos_v[:1], neg_u[:1], neg_v[:1]
        )
    
    return model, final_embeddings.cpu(), adj_indices, adj_values

# ==========================================
# STEP 6: MICROSTRUCTURE QUANTIFICATION & SCORING
# ==========================================
def compute_microstructure_quantification(valid_concepts, concept_abstract_map, all_metrics, nx_graph):
    """Map concepts to representative microstructure property values"""
    
    concept_properties = {}
    
    for concept in valid_concepts:
        doc_indices = concept_abstract_map[concept]
        values = []
        
        for idx in doc_indices:
            metrics = all_metrics[idx]
            # Aggregate all numeric metrics associated with this concept
            for metric_name, metric_values in metrics.items():
                values.extend(metric_values)
        
        # Use median for robustness to outliers
        concept_properties[concept] = np.median(values) if values else 0.0
    
    # Prepare training data for property prediction model
    X_feat, y_target = [], []
    for u, v in nx_graph.edges():
        pu, pv = concept_properties.get(u, 0), concept_properties.get(v, 0)
        w = nx_graph[u][v]['weight']  # Co-occurrence strength
        
        X_feat.append([pu, pv, w])
        # Target: optimistic but plausible property enhancement
        y_target.append(max(pu, pv) * 1.08 if max(pu, pv) > 0 else 0)
    
    # Train lightweight regressor if sufficient data
    ridge = None
    if len(X_feat) > 10:
        ridge = Ridge(alpha=1.0).fit(np.array(X_feat), np.array(y_target))
    
    return concept_properties, ridge


def compute_research_direction_scores(
    model, final_emb, nx_graph, valid_concepts, concept_properties, 
    ridge, embed_model, d_prev_dict, adj_indices, adj_values,
    n_samples=3000
):
    """Score novel concept pairs for promising microstructure research directions"""
    
    # Sample candidate concept pairs (unconnected in graph)
    n_concepts = len(valid_concepts)
    u_ids = np.random.randint(n_concepts, size=n_samples)
    v_ids = np.random.randint(n_concepts, size=n_samples)
    
    candidate_pairs = []
    for u_idx, v_idx in zip(u_ids, v_ids):
        if u_idx == v_idx: continue
        u_c, v_c = valid_concepts[u_idx], valid_concepts[v_idx]
        if nx_graph.has_edge(u_c, v_c): continue  # Skip existing connections
        candidate_pairs.append((u_idx, v_idx, u_c, v_c))
    
    if not candidate_pairs:
        return pd.DataFrame()
    
    # GNN scoring for novel pairs
    u_tensor = torch.tensor([p[0] for p in candidate_pairs], dtype=torch.long, device=DEVICE)
    v_tensor = torch.tensor([p[1] for p in candidate_pairs], dtype=torch.long, device=DEVICE)
    
    model.eval()
    with torch.no_grad():
        _, _, h2 = model(
            adj_indices, adj_values, n_concepts,
            final_emb.to(DEVICE), u_tensor, v_tensor, u_tensor, v_tensor
        )
        pair_features = torch.cat([h2, h2], dim=1)
        gnn_logits = model.decoder(pair_features).squeeze(1)
        gnn_scores = torch.sigmoid(gnn_logits).cpu().numpy()
    
    # Semantic novelty via embedding cosine similarity
    emb_np = embed_model.encode(valid_concepts, show_progress_bar=False)
    cos_sims = np.sum(
        emb_np[u_tensor.cpu().numpy()] * emb_np[v_tensor.cpu().numpy()], 
        axis=1
    )
    
    # Composite scoring function
    results = []
    for i, (u_idx, v_idx, u_c, v_c) in enumerate(candidate_pairs):
        # Graph distance (novelty proxy): prefer moderately distant concepts
        try:
            d_prev = d_prev_dict[u_c][v_c]
        except KeyError:
            d_prev = 4
        if d_prev < 2: continue  # Too closely related, skip
        
        # Property-based feasibility
        p_u = concept_properties.get(u_c, 0)
        p_v = concept_properties.get(v_c, 0)
        
        expected_improvement = 0
        if ridge is not None and (p_u > 0 or p_v > 0):
            expected_improvement = float(ridge.predict([[p_u, p_v, 1.0]])[0])
        
        semantic_novelty = 1.0 - cos_sims[i]
        feasibility = np.exp(-0.5 * semantic_novelty) * (1.0 if (p_u > 0 or p_v > 0) else 0.6)
        
        # Weighted composite score (tunable weights)
        alpha = {'gnn': 0.4, 'novelty': 0.3, 'gain': 0.2, 'feas': -0.1}
        norm_gain = np.clip((expected_improvement - 50) / 200, 0, 1)  # Normalize to [0,1]
        
        D_uv = (
            alpha['gnn'] * gnn_scores[i] +
            alpha['novelty'] * semantic_novelty + 
            alpha['gain'] * norm_gain + 
            alpha['feas'] * (1.0 - feasibility)
        )
        
        results.append({
            'concept_u': u_c,
            'concept_v': v_c,
            'graph_distance': d_prev,
            'gnn_affinity': float(gnn_scores[i]),
            'semantic_novelty': float(semantic_novelty),
            'expected_property_gain': expected_improvement,
            'feasibility_score': float(feasibility),
            'composite_score': float(D_uv)
        })
    
    df = pd.DataFrame(results).sort_values('composite_score', ascending=False)
    return df.head(50)

# ==========================================
# STEP 7: LLM CURATION OF RESEARCH DIRECTIONS
# ==========================================
def generate_research_directions(top_pairs_df, tokenizer, model):
    """Generate LLM-curated research hypotheses for top-scoring concept pairs"""
    
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
        
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
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
def main():
    st.set_page_config(page_title="Alloy Microstructure Concept Graph", layout="wide")
    st.title("🔬 Laser & Multicomponent Alloy Microstructure Analyzer")
    st.caption("Discover novel research directions via concept graph + lightweight LLM + Pure PyTorch GraphSAGE")
    
    with st.sidebar:
        st.header("⚙️ Pipeline Parameters")
        min_freq = st.slider("Min concept frequency", 2, 10, MIN_CONCEPT_FREQ, key="min_freq")
        st.info("💡 Tip: Paste 20-25 recent abstracts on laser processing or alloy microstructure for best results.")
        st.markdown("---")
        st.markdown("**Domain Focus:**")
        st.markdown("- ✅ Alloy compositions (AlSi10Mg, Ti6Al4V, HEAs)")
        st.markdown("- ✅ Laser parameters (power, speed, energy density)")
        st.markdown("- ✅ Microstructure features (grains, phases, defects)")
        st.markdown("- ✅ Mechanical properties (hardness, strength, ductility)")
    
    # Abstract input area
    abstract_input = st.text_area(
        "📋 Paste scientific abstracts (one per block, separated by blank lines):", 
        height=300,
        placeholder="""Example abstracts:
"Laser powder bed fusion of AlSi10Mg reveals columnar-to-equiaxed transition at 85 J/mm³ energy density, with grain refinement from 45 μm to 12 μm..."

"High-entropy alloy CoCrFeNiMo processed by direct energy deposition shows enhanced microhardness (420 HV) due to nanoscale precipitate formation..."

"Residual stress mitigation in Ti6Al4V via laser remelting: EBSD analysis reveals texture evolution from <001> to <111> fiber..."
""")
    
    if st.button("🚀 Analyze Abstracts", type="primary", use_container_width=True):
        if not abstract_input.strip():
            st.error("⚠️ Please enter at least one scientific abstract.")
            return
            
        # Parse abstracts (split by blank lines)
        abstracts = [t.strip() for t in re.split(r'\n\s*\n', abstract_input) if t.strip()]
        
        if len(abstracts) < 10:
            st.warning(f"⚠️ Only {len(abstracts)} abstracts detected. Pipeline works best with 20-25 abstracts for robust concept graph construction.")
        elif len(abstracts) > 35:
            st.warning(f"⚠️ {len(abstracts)} abstracts may increase processing time. Consider using 20-25 for optimal performance.")
            
        progress_bar = st.progress(0)
        status = st.status("🔄 Initializing pipeline...", expanded=True)
        
        try:
            # Load models
            with status:
                st.write("📦 Loading embedding model & lightweight LLM (<0.5B params)...")
                embed_model = load_embedding_model()
                tokenizer, llm_model = load_lightweight_llm()
                st.success("✅ Models loaded successfully")
            progress_bar.progress(10)
            
            # Step 1-2: Extract concepts & metrics
            with st.status("🔍 Extracting domain concepts & quantitative metrics..."):
                all_concepts, all_metrics = extract_concepts_from_abstracts(abstracts, tokenizer, llm_model)
                valid_concepts, concept_to_id, id_to_concept, concept_abstract_map = normalize_and_filter_concepts(all_concepts)
                st.write(f"✅ Extracted **{len(valid_concepts)}** unique microstructure-relevant concepts")
                if len(valid_concepts) < 10:
                    st.warning("⚠️ Few concepts extracted. Try adding more abstracts or reducing frequency filter.")
            progress_bar.progress(25)
            
            if len(valid_concepts) < 5:
                st.error("❌ Too few valid concepts for graph construction. Add more abstracts or adjust parameters.")
                return
                
            # Step 3: Build concept graph
            with st.status("🕸️ Building concept co-occurrence graph..."):
                nx_graph, d_prev_dict = build_concept_graph(all_concepts, concept_to_id)
                pos_pairs, neg_pairs = sample_edges_for_training(nx_graph, d_prev_dict, valid_concepts, concept_to_id)
                st.write(f"✅ Graph: **{len(valid_concepts)}** nodes, **{nx_graph.number_of_edges()}** edges")
            progress_bar.progress(40)
            
            # Step 4: Generate semantic embeddings
            with st.status("🧠 Generating semantic embeddings for concepts..."):
                node_features = generate_embeddings(valid_concepts, embed_model)
                st.write(f"✅ Embedding dimension: {node_features.shape[1]}")
            progress_bar.progress(50)
            
            # Step 5: Train GraphSAGE
            def _training_progress(epoch, loss):
                progress_bar.progress(50 + epoch * 0.3)
                status.write(f"📊 Epoch {epoch}/{TRAIN_EPOCHS} | Loss: {loss:.4f}")
            
            with st.status("🤖 Training Pure PyTorch GraphSAGE (contrastive learning)..."):
                gnn_model, final_emb, adj_indices, adj_values = train_gnn(
                    node_features, nx_graph, concept_to_id, pos_pairs, neg_pairs, _training_progress
                )
                st.success("✅ GNN training complete")
            progress_bar.progress(80)
            
            # Step 6: Quantification & scoring
            with st.status("📈 Computing property proxies & scoring novel directions..."):
                concept_properties, ridge = compute_microstructure_quantification(
                    valid_concepts, concept_abstract_map, all_metrics, nx_graph
                )
                top_scores = compute_research_direction_scores(
                    gnn_model, final_emb, nx_graph, valid_concepts, concept_properties,
                    ridge, embed_model, d_prev_dict, adj_indices, adj_values
                )
                st.write(f"✅ Scored **{len(top_scores)}** novel concept pairs")
            progress_bar.progress(90)
            
            # Step 7: LLM curation
            with st.status("✍️ Generating LLM-curated research hypotheses..."):
                directions_df = generate_research_directions(top_scores, tokenizer, llm_model)
                st.success("✅ Pipeline complete!")
            progress_bar.progress(100)
            status.update(label="✅ Analysis complete!", state="complete", expanded=False)
            
            # === DISPLAY RESULTS ===
            
            # Top research directions table
            st.subheader("🎯 Top Predicted Research Directions")
            st.markdown("Concept pairs ranked by composite novelty + feasibility score")
            
            # Format dataframe for display
            display_df = directions_df.copy()
            display_df['Composite Score'] = display_df['Composite Score'].astype(float)
            display_df['Expected Gain'] = display_df['Expected Gain'].astype(float)
            
            st.dataframe(
                display_df[['Concept Pair', 'Composite Score', 'Expected Gain', 'Feasibility', 'Research Hypothesis']],
                use_container_width=True,
                column_config={
                    "Concept Pair": st.column_config.TextColumn("Concept Pair", width="medium"),
                    "Composite Score": st.column_config.NumberColumn("Score", format="%.3f"),
                    "Expected Gain": st.column_config.NumberColumn("Expected Gain", format="%.1f"),
                    "Feasibility": st.column_config.NumberColumn("Feasibility", format="%.2f"),
                    "Research Hypothesis": st.column_config.TextColumn("Hypothesis", width="large")
                }
            )
            
            # Download button for results
            csv = directions_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="alloy_research_directions.csv",
                mime="text/csv"
            )
            
            # Interactive concept graph visualization
            st.subheader("🌐 Interactive Concept Graph")
            st.markdown("Nodes = concepts | Edges = co-occurrence in abstracts | Size = concept frequency")
            
            net = Network(
                height="650px", 
                width="100%", 
                bgcolor="#1e1e1e", 
                font_color="white",
                select_menu=True,
                filter_menu=True
            )
            net.barnes_hut(gravity=-80000, spring_length=200, spring_strength=0.05)
            
            # Add nodes with size proportional to degree
            for node in nx_graph.nodes():
                deg = nx_graph.degree(node)
                size = max(12, min(50, deg * 4))
                # Color by concept type
                if any(alloy in node.lower() for alloy in ['al', 'ti', 'ni', 'cr', 'fe', 'co', 'mo']):
                    color = "#4CAF50"  # Green for alloys
                elif any(laser in node.lower() for laser in ['laser', 'scan', 'power', 'melt', 'energy']):
                    color = "#2196F3"  # Blue for laser params
                elif any(micro in node.lower() for micro in ['grain', 'phase', 'microstructure', 'hardness']):
                    color = "#FF9800"  # Orange for microstructure
                else:
                    color = "#9E9E9E"  # Gray for others
                    
                net.add_node(
                    node, 
                    label=node, 
                    size=size, 
                    color=color,
                    title=f"Concept: {node}\nDegree: {deg}\nFrequency: {concept_abstract_map[node].__len__()}"
                )
                
            # Add edges with width proportional to weight
            for u, v in nx_graph.edges():
                w = nx_graph[u][v]['weight']
                net.add_edge(u, v, value=w, width=min(4, w * 0.8), color="#666666")
            
            # Save and display interactive graph
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                net.save_graph(tmp.name)
                with open(tmp.name, "r", encoding="utf-8") as f:
                    st.components.v1.html(f.read(), height=700, scrolling=True)
            
            # Additional insights
            with st.expander("📊 Additional Insights", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Concepts", len(valid_concepts))
                with col2:
                    st.metric("Graph Edges", nx_graph.number_of_edges())
                with col3:
                    st.metric("Avg. Concept Frequency", f"{np.mean([len(concept_abstract_map[c]) for c in valid_concepts]):.1f}")
                
                st.markdown("### Top Concepts by Frequency")
                concept_freq = [(c, len(concept_abstract_map[c])) for c in valid_concepts]
                concept_freq.sort(key=lambda x: x[1], reverse=True)
                freq_df = pd.DataFrame(concept_freq[:15], columns=["Concept", "Frequency"])
                st.dataframe(freq_df, use_container_width=True)
                    
        except Exception as e:
            st.error(f"❌ Pipeline failed: {str(e)}")
            with st.expander("🔍 View Error Details"):
                st.code(traceback.format_exc())
    
    # Footer with usage tips
    st.markdown("---")
    st.markdown("""
    **💡 Usage Tips:**
    - Paste 20-25 abstracts from recent papers on laser additive manufacturing or alloy microstructure
    - Abstracts should mention quantitative metrics (grain size, hardness, energy density) for best quantification
    - The concept graph highlights relationships between alloy compositions, laser parameters, and microstructure outcomes
    - Top-scoring pairs represent novel but feasible research directions worthy of experimental validation
    
    **🔧 Technical Notes:**
    - Uses Qwen2.5-0.5B-Instruct (<1B params) for concept extraction & hypothesis generation
    - Pure PyTorch Sparse GraphSAGE for memory-efficient graph learning
    - All processing runs locally; no external API calls required
    """)

if __name__ == "__main__":
    main()
