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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

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

# Domain keywords (unchanged)
DOMAIN_KEYWORDS = [
    "grain size", "phase fraction", "microhardness", "tensile strength", 
    "yield strength", "elongation", "residual stress", "texture intensity",
    "columnar grain", "equiaxed grain", "dendrite", "eutectic", "martensite",
    "austenite", "precipitate", "segregation", "porosity", "crack density",
    "laser power", "scan speed", "hatch spacing", "layer thickness",
    "pulse duration", "energy density", "spot diameter", "melt pool",
    "cooling rate", "solidification rate", "dilution ratio",
    "high-entropy alloy", "HEA", "multi-principal element", "complex concentrated",
    "powder bed fusion", "LPBF", "direct energy deposition", "DED"
]

ALLOY_PATTERNS = [
    r'[A-Z][a-z]?(?:\d+(?:\.\d+)?(?:[A-Z][a-z]?\d*(?:\.\d+)?)*)+',
    r'(?:Ni|Co|Cr|Fe|Al|Ti|Cu|Nb|Mo|W)(?:[-\s]?\d+(?:\.\d+)?%?)+',
    r'(?:high-entropy|HEA|multi-principal|complex concentrated)',
]

# Pipeline hyperparameters (will be adapted dynamically)
DEFAULT_MIN_CONCEPT_FREQ = 3
DEFAULT_MIN_CONCEPT_LENGTH_WORDS = 2
GNN_HIDDEN_DIM = 128
TRAIN_EPOCHS = 50
LR = 1e-3
NEG_DPREV_FOCUS = 3

# Domain seed concepts (injected when corpus is small)
DOMAIN_SEED_CONCEPTS = [
    # Alloys
    "aluminum alloy", "titanium alloy", "nickel alloy", "high-entropy alloy", "steel",
    # Laser parameters
    "laser power", "scan speed", "energy density", "hatch spacing", "layer thickness",
    # Microstructure
    "grain size", "phase fraction", "porosity", "residual stress", "texture",
    # Mechanical properties
    "microhardness", "tensile strength", "yield strength", "elongation"
]

# ==========================================
# ADAPTIVE THRESHOLDS (NEW)
# ==========================================
def get_adaptive_thresholds(num_abstracts: int):
    """Dynamically adjust frequency and length thresholds based on corpus size."""
    if num_abstracts <= 15:
        return {"min_freq": 1, "min_words": 1, "min_degree": 1}
    elif num_abstracts <= 30:
        return {"min_freq": 2, "min_words": 1, "min_degree": 1}
    else:
        return {"min_freq": 3, "min_words": 2, "min_degree": 2}

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
# DOMAIN-SPECIFIC CONCEPT NORMALIZATION (unchanged)
# ==========================================
def normalize_alloy_composition(concept: str) -> str:
    normalized = re.sub(r'[\s\-_]', '', concept).lower()
    normalized = re.sub(r'(ti)(6)(al)(4)(v)', r'ti6al4v', normalized)
    normalized = re.sub(r'(al)(si)(10)(mg)', r'alsi10mg', normalized)
    normalized = re.sub(r'(inconel)(\s*718|718)', r'inconel718', normalized)
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
    generic_terms = {'study', 'analysis', 'effect', 'role', 'investigation', 
                     'research', 'method', 'approach', 'paper', 'work'}
    has_generic = any(term in concept_lower.split() for term in generic_terms)
    return (has_domain_keyword or has_alloy_pattern) and not has_generic and len(concept.split()) >= 1  # relaxed

# ==========================================
# STEP 1-2: CONCEPT EXTRACTION (unchanged)
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
        grain_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:μm|micron|um|nm)\s*(?:grain|average|size|diameter)?', text, re.I)
        if grain_matches:
            metrics['grain_size_um'] = [float(m) for m in grain_matches]
        mech_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:HV|GPa|MPa|ksi|GPa)\s*(?:hardness|strength|yield|tensile|ultimate)?', text, re.I)
        if mech_matches:
            metrics['mechanical_property'] = [float(m) for m in mech_matches]
        energy_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:J/mm³|J mm-3|J mm⁻³|J/mm\^3)', text, re.I)
        if energy_matches:
            metrics['energy_density_j_mm3'] = [float(m) for m in energy_matches]
        defect_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:porosity|void|crack)', text, re.I)
        if defect_matches:
            metrics['defect_fraction_pct'] = [float(m) for m in defect_matches]
        all_metrics.append(metrics)
        
        prompt = prompt_template.format(text=text)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=150,
                temperature=0.2,
                do_sample=False,  # deterministic for extraction
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        concepts = []
        try:
            parsed = json.loads(response.replace("'", '"').strip())
            if isinstance(parsed, list):
                concepts = [c.strip().lower().rstrip('.') for c in parsed if isinstance(c, str) and len(c.strip()) > 3]
        except (json.JSONDecodeError, TypeError):
            concepts = _fallback_concept_extraction(text)
        
        normalized = []
        for c in concepts:
            if any(elem in c.lower() for elem in ['al', 'ti', 'ni', 'cr', 'fe', 'co', 'mo', 'nb', 'cu']):
                c = normalize_alloy_composition(c)
            elif any(lp in c.lower() for lp in ['laser', 'scan', 'power', 'speed', 'melt', 'pool', 'energy']):
                c = normalize_laser_term(c)
            if is_valid_microstructure_concept(c):
                normalized.append(c)
        all_concepts.append(normalized)
    
    return all_concepts, all_metrics

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

# ==========================================
# NEW: SEMANTIC CLUSTERING & SEED INJECTION
# ==========================================
def cluster_similar_concepts(concepts, embed_model, similarity_threshold=0.75):
    """Merge semantically similar concepts into cluster representatives."""
    if len(concepts) < 3:
        return concepts, {c: c for c in concepts}
    embeddings = embed_model.encode(concepts, show_progress_bar=False)
    sim_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - sim_matrix
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - similarity_threshold,
        linkage='average'
    ).fit(embeddings)
    concept_to_cluster = {}
    cluster_repr = {}
    for idx, label in enumerate(clustering.labels_):
        concept = concepts[idx]
        if label not in cluster_repr:
            # choose the shortest concept as representative (often the canonical form)
            cluster_repr[label] = concept
        concept_to_cluster[concept] = cluster_repr[label]
    unique_concepts = list(cluster_repr.values())
    return unique_concepts, concept_to_cluster

def inject_domain_seeds(valid_concepts, concept_to_id, concept_counts, concept_abstract_map, num_abstracts):
    """Add predefined domain seeds to boost graph connectivity."""
    added = 0
    for seed in DOMAIN_SEED_CONCEPTS:
        if seed not in concept_to_id:
            concept_to_id[seed] = len(concept_to_id)
            valid_concepts.append(seed)
            concept_counts[seed] = max(1, num_abstracts // 10)  # artificial frequency
            concept_abstract_map[seed] = list(range(num_abstracts))  # assume appears everywhere
            added += 1
    return valid_concepts, concept_to_id, concept_counts, concept_abstract_map, added

def normalize_and_filter_concepts(all_concepts, embed_model, num_abstracts, use_clustering=True, inject_seeds=True):
    """Adaptive concept filtering with clustering and seed injection."""
    concept_counts = defaultdict(int)
    concept_abstract_map = defaultdict(list)
    
    for doc_idx, concepts in enumerate(all_concepts):
        seen = set()
        for c in concepts:
            if c not in seen and is_valid_microstructure_concept(c):
                concept_counts[c] += 1
                concept_abstract_map[c].append(doc_idx)
                seen.add(c)
    
    # Adaptive thresholds
    thresholds = get_adaptive_thresholds(num_abstracts)
    min_freq = thresholds["min_freq"]
    min_words = thresholds["min_words"]
    
    valid_concepts = [c for c, cnt in concept_counts.items() if cnt >= min_freq and len(c.split()) >= min_words]
    
    # Inject domain seeds if corpus is small and user allows
    if inject_seeds and len(valid_concepts) < 10:
        valid_concepts, concept_to_id, concept_counts, concept_abstract_map, added = inject_domain_seeds(
            valid_concepts, {c:i for i,c in enumerate(valid_concepts)}, concept_counts, concept_abstract_map, num_abstracts
        )
        if added > 0:
            st.info(f"Injected {added} domain seed concepts to enrich the graph.")
    
    # Semantic clustering to merge near-synonyms
    if use_clustering and embed_model and len(valid_concepts) >= 5:
        clustered_concepts, concept_to_cluster = cluster_similar_concepts(valid_concepts, embed_model, similarity_threshold=0.75)
        # Remap abstract map to clusters
        new_abstract_map = defaultdict(list)
        for orig, docs in concept_abstract_map.items():
            cluster_rep = concept_to_cluster.get(orig, orig)
            if cluster_rep in clustered_concepts:
                new_abstract_map[cluster_rep].extend(docs)
        concept_abstract_map = new_abstract_map
        valid_concepts = clustered_concepts
    
    # Final deduplication
    valid_concepts = list(set(valid_concepts))
    concept_to_id = {c: i for i, c in enumerate(valid_concepts)}
    id_to_concept = {i: c for i, c in enumerate(valid_concepts)}
    
    return valid_concepts, concept_to_id, id_to_concept, concept_abstract_map

# ==========================================
# STEP 3: HYBRID CONCEPT GRAPH (NEW)
# ==========================================
def build_semantic_only_graph(valid_concepts, embed_model, similarity_threshold=0.75):
    """Fallback graph from pure embedding similarity."""
    nx_graph = nx.Graph()
    for c in valid_concepts:
        nx_graph.add_node(c)
    if len(valid_concepts) < 2:
        return nx_graph
    embeddings = embed_model.encode(valid_concepts, show_progress_bar=False)
    sim = cosine_similarity(embeddings)
    for i in range(len(valid_concepts)):
        for j in range(i+1, len(valid_concepts)):
            if sim[i][j] > similarity_threshold:
                nx_graph.add_edge(valid_concepts[i], valid_concepts[j], weight=sim[i][j], edge_type='semantic')
    # Ensure connectivity via minimum spanning tree
    if not nx.is_connected(nx_graph) and len(valid_concepts) > 3:
        components = list(nx.connected_components(nx_graph))
        for k in range(len(components)-1):
            best_sim, best_pair = 0, None
            for u in components[k]:
                idx_u = valid_concepts.index(u)
                for v in components[k+1]:
                    idx_v = valid_concepts.index(v)
                    if sim[idx_u][idx_v] > best_sim:
                        best_sim = sim[idx_u][idx_v]
                        best_pair = (u, v)
            if best_pair:
                nx_graph.add_edge(*best_pair, weight=best_sim, edge_type='bridge')
    return nx_graph

def build_hybrid_graph(all_concepts, concept_to_id, embed_model, cooccurrence_weight=0.5, semantic_weight=0.5):
    """Combine observed co-occurrence with embedding similarity edges."""
    valid_concepts = list(concept_to_id.keys())
    nx_graph = nx.Graph()
    for c in valid_concepts:
        nx_graph.add_node(c)
    
    # Co-occurrence edges
    for concepts in all_concepts:
        valid_in_doc = [c for c in concepts if c in concept_to_id]
        for i in range(len(valid_in_doc)):
            for j in range(i+1, len(valid_in_doc)):
                u, v = valid_in_doc[i], valid_in_doc[j]
                if nx_graph.has_edge(u, v):
                    nx_graph[u][v]['cooccurrence'] = nx_graph[u][v].get('cooccurrence', 0) + 1
                else:
                    nx_graph.add_edge(u, v, cooccurrence=1, semantic=0)
    
    # Semantic edges for missing or low-degree pairs
    if embed_model and len(valid_concepts) >= 5:
        embeddings = embed_model.encode(valid_concepts, show_progress_bar=False)
        sim = cosine_similarity(embeddings)
        for i, u in enumerate(valid_concepts):
            for j, v in enumerate(valid_concepts[i+1:], start=i+1):
                if nx_graph.has_edge(u, v):
                    continue
                if sim[i][j] > 0.72:   # high similarity threshold
                    nx_graph.add_edge(u, v, cooccurrence=0, semantic=sim[i][j])
    
    # Combine weights
    for u, v, data in nx_graph.edges(data=True):
        cooc = data.get('cooccurrence', 0)
        sem = data.get('semantic', 0)
        data['weight'] = cooccurrence_weight * cooc + semantic_weight * sem
    
    return nx_graph

def build_concept_graph(all_concepts, concept_to_id, embed_model=None, use_semantic_edges=True, num_abstracts=25):
    """Main graph construction with fallback to semantic-only if too sparse."""
    if use_semantic_edges and embed_model and len(concept_to_id) < 8:
        return build_semantic_only_graph(list(concept_to_id.keys()), embed_model, similarity_threshold=0.70)
    
    # Determine hybrid weights based on corpus size
    cooc_w = 0.7 if num_abstracts >= 25 else 0.4
    sem_w = 0.3 if num_abstracts >= 25 else 0.6
    return build_hybrid_graph(all_concepts, concept_to_id, embed_model, cooc_w, sem_w)

def sample_edges_for_training(nx_graph, d_prev_dict, valid_concepts, concept_to_id):
    """Same as original, but adapted to possibly smaller graph."""
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
        pair = (u_idx, v_idx)
        if pair not in neg_pairs and (v_idx, u_idx) not in neg_pairs:
            if not nx_graph.has_edge(valid_concepts[u_idx], valid_concepts[v_idx]):
                neg_pairs.append(pair)
    return pos_pairs, neg_pairs

# ==========================================
# STEP 4-7 (unchanged from original except using new graph)
# ==========================================
def generate_embeddings(valid_concepts, embed_model):
    embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=32)
    return torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)

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
    model.eval()
    with torch.no_grad():
        _, _, final_embeddings = model(
            adj_indices, adj_values, len(concept_to_id),
            node_features, pos_u[:1], pos_v[:1], neg_u[:1], neg_v[:1]
        )
    return model, final_embeddings.cpu(), adj_indices, adj_values

def compute_microstructure_quantification(valid_concepts, concept_abstract_map, all_metrics, nx_graph):
    concept_properties = {}
    for concept in valid_concepts:
        doc_indices = concept_abstract_map.get(concept, [])
        values = []
        for idx in doc_indices:
            metrics = all_metrics[idx]
            for metric_name, metric_values in metrics.items():
                values.extend(metric_values)
        concept_properties[concept] = np.median(values) if values else 0.0
    X_feat, y_target = [], []
    for u, v in nx_graph.edges():
        pu, pv = concept_properties.get(u, 0), concept_properties.get(v, 0)
        w = nx_graph[u][v]['weight']
        X_feat.append([pu, pv, w])
        y_target.append(max(pu, pv) * 1.08 if max(pu, pv) > 0 else 0)
    ridge = None
    if len(X_feat) > 10:
        ridge = Ridge(alpha=1.0).fit(np.array(X_feat), np.array(y_target))
    return concept_properties, ridge

def compute_research_direction_scores(
    model, final_emb, nx_graph, valid_concepts, concept_properties, 
    ridge, embed_model, d_prev_dict, adj_indices, adj_values,
    n_samples=3000
):
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
        _, _, h2 = model(
            adj_indices, adj_values, n_concepts,
            final_emb.to(DEVICE), u_tensor, v_tensor, u_tensor, v_tensor
        )
        pair_features = torch.cat([h2, h2], dim=1)
        gnn_logits = model.decoder(pair_features).squeeze(1)
        gnn_scores = torch.sigmoid(gnn_logits).cpu().numpy()
    emb_np = embed_model.encode(valid_concepts, show_progress_bar=False)
    cos_sims = np.sum(
        emb_np[u_tensor.cpu().numpy()] * emb_np[v_tensor.cpu().numpy()], 
        axis=1
    )
    results = []
    for i, (u_idx, v_idx, u_c, v_c) in enumerate(candidate_pairs):
        try:
            d_prev = d_prev_dict[u_c][v_c]
        except KeyError:
            d_prev = 4
        if d_prev < 2: continue
        p_u = concept_properties.get(u_c, 0)
        p_v = concept_properties.get(v_c, 0)
        expected_improvement = 0
        if ridge is not None and (p_u > 0 or p_v > 0):
            expected_improvement = float(ridge.predict([[p_u, p_v, 1.0]])[0])
        semantic_novelty = 1.0 - cos_sims[i]
        feasibility = np.exp(-0.5 * semantic_novelty) * (1.0 if (p_u > 0 or p_v > 0) else 0.6)
        alpha = {'gnn': 0.4, 'novelty': 0.3, 'gain': 0.2, 'feas': -0.1}
        norm_gain = np.clip((expected_improvement - 50) / 200, 0, 1)
        D_uv = (alpha['gnn'] * gnn_scores[i] + alpha['novelty'] * semantic_novelty + 
                alpha['gain'] * norm_gain + alpha['feas'] * (1.0 - feasibility))
        results.append({
            'concept_u': u_c, 'concept_v': v_c, 'graph_distance': d_prev,
            'gnn_affinity': float(gnn_scores[i]), 'semantic_novelty': float(semantic_novelty),
            'expected_property_gain': expected_improvement, 'feasibility_score': float(feasibility),
            'composite_score': float(D_uv)
        })
    df = pd.DataFrame(results).sort_values('composite_score', ascending=False)
    return df.head(50)

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
            u=row['concept_u'].title(), v=row['concept_v'].title(), 
            prop=float(row['expected_property_gain']), feas=float(row['feasibility_score'])
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids, max_new_tokens=180, temperature=0.25,
                do_sample=True, pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.1
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
# STREAMLIT UI (modified with small-corpus controls)
# ==========================================
def main():
    st.set_page_config(page_title="Alloy Microstructure Concept Graph", layout="wide")
    st.title("🔬 Laser & Multicomponent Alloy Microstructure Analyzer")
    st.caption("Discover novel research directions via concept graph + lightweight LLM + Pure PyTorch GraphSAGE")
    
    with st.sidebar:
        st.header("⚙️ Pipeline Parameters")
        # Adaptive settings for small corpus
        st.checkbox("Enable semantic clustering (recommended for <30 abstracts)", value=True, key="use_clustering")
        st.checkbox("Inject domain seed concepts", value=True, key="inject_seeds")
        st.checkbox("Use semantic edges for graph building", value=True, key="semantic_edges")
        st.info("💡 Tip: With 20-25 abstracts, these adaptive strategies ensure a connected concept graph.")
        st.markdown("---")
        st.markdown("**Domain Focus:**")
        st.markdown("- ✅ Alloy compositions (AlSi10Mg, Ti6Al4V, HEAs)")
        st.markdown("- ✅ Laser parameters (power, speed, energy density)")
        st.markdown("- ✅ Microstructure features (grains, phases, defects)")
        st.markdown("- ✅ Mechanical properties (hardness, strength, ductility)")
    
    abstract_input = st.text_area(
        "📋 Paste scientific abstracts (one per block, separated by blank lines):", 
        height=300,
        placeholder="""Example abstracts:
"Laser powder bed fusion of AlSi10Mg reveals columnar-to-equiaxed transition at 85 J/mm³ energy density, with grain refinement from 45 μm to 12 μm..."

"High-entropy alloy CoCrFeNiMo processed by direct energy deposition shows enhanced microhardness (420 HV) due to nanoscale precipitate formation..."

"Residual stress mitigation in Ti6Al4V via laser remelting: EBSD analysis reveals texture evolution from <001> to <111> fiber..."
"""
    )
    
    if st.button("🚀 Analyze Abstracts", type="primary", use_container_width=True):
        if not abstract_input.strip():
            st.error("⚠️ Please enter at least one scientific abstract.")
            return
            
        abstracts = [t.strip() for t in re.split(r'\n\s*\n', abstract_input) if t.strip()]
        st.info(f"Processing {len(abstracts)} abstracts with small‑corpus adaptations.")
        
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
            
            # Step 1-2: Extract concepts
            with st.status("🔍 Extracting domain concepts & quantitative metrics..."):
                all_concepts, all_metrics = extract_concepts_from_abstracts(abstracts, tokenizer, llm_model)
                valid_concepts, concept_to_id, id_to_concept, concept_abstract_map = normalize_and_filter_concepts(
                    all_concepts, embed_model, len(abstracts),
                    use_clustering=st.session_state.get("use_clustering", True),
                    inject_seeds=st.session_state.get("inject_seeds", True)
                )
                st.write(f"✅ Extracted **{len(valid_concepts)}** unique microstructure-relevant concepts")
                if len(valid_concepts) < 5:
                    st.warning("Still very few concepts. Consider adding more abstracts or lowering similarity threshold.")
            progress_bar.progress(25)
            
            if len(valid_concepts) < 3:
                st.error("❌ Too few concepts even after adaptation. Please add at least 10-15 more abstracts.")
                return
                
            # Step 3: Build concept graph (hybrid or semantic-only)
            with st.status("🕸️ Building concept graph (hybrid co-occurrence + semantic)..."):
                nx_graph = build_concept_graph(
                    all_concepts, concept_to_id, embed_model,
                    use_semantic_edges=st.session_state.get("semantic_edges", True),
                    num_abstracts=len(abstracts)
                )
                # Precompute shortest paths for novelty scoring
                if nx_graph.number_of_nodes() > 0 and nx_graph.number_of_edges() > 0:
                    d_prev_dict = dict(nx.all_pairs_shortest_path_length(nx_graph, cutoff=4))
                else:
                    d_prev_dict = {}
                pos_pairs, neg_pairs = sample_edges_for_training(nx_graph, d_prev_dict, valid_concepts, concept_to_id)
                st.write(f"✅ Graph: **{len(valid_concepts)}** nodes, **{nx_graph.number_of_edges()}** edges")
                if not nx.is_connected(nx_graph):
                    st.warning("Graph is disconnected. Consider enabling semantic edges or injecting more seeds.")
            progress_bar.progress(40)
            
            # Step 4: Generate embeddings
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
            
            # Display results (same as original)
            st.subheader("🎯 Top Predicted Research Directions")
            st.dataframe(directions_df[['Concept Pair', 'Composite Score', 'Expected Gain', 'Feasibility', 'Research Hypothesis']],
                         use_container_width=True)
            csv = directions_df.to_csv(index=False)
            st.download_button("📥 Download Results as CSV", data=csv, file_name="alloy_research_directions.csv", mime="text/csv")
            
            # Interactive graph (only if graph is not huge)
            if nx_graph.number_of_nodes() <= 200:
                st.subheader("🌐 Interactive Concept Graph")
                net = Network(height="650px", width="100%", bgcolor="#1e1e1e", font_color="white")
                net.barnes_hut(gravity=-80000, spring_length=200, spring_strength=0.05)
                for node in nx_graph.nodes():
                    deg = nx_graph.degree(node)
                    size = max(12, min(50, deg * 4))
                    if any(alloy in node.lower() for alloy in ['al', 'ti', 'ni', 'cr', 'fe', 'co', 'mo']):
                        color = "#4CAF50"
                    elif any(laser in node.lower() for laser in ['laser', 'scan', 'power', 'melt', 'energy']):
                        color = "#2196F3"
                    elif any(micro in node.lower() for micro in ['grain', 'phase', 'microstructure', 'hardness']):
                        color = "#FF9800"
                    else:
                        color = "#9E9E9E"
                    net.add_node(node, label=node, size=size, color=color, title=f"Degree: {deg}")
                for u, v in nx_graph.edges():
                    w = nx_graph[u][v]['weight']
                    net.add_edge(u, v, value=w, width=min(4, w * 0.8), color="#666666")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                    net.save_graph(tmp.name)
                    with open(tmp.name, "r", encoding="utf-8") as f:
                        st.components.v1.html(f.read(), height=700, scrolling=True)
            else:
                st.info("Graph too large for interactive display; download results instead.")
                
        except Exception as e:
            st.error(f"❌ Pipeline failed: {str(e)}")
            with st.expander("🔍 View Error Details"):
                st.code(traceback.format_exc())
    
    st.markdown("---")
    st.markdown("""
    **💡 Usage Tips for 20‑25 Abstracts:**
    - The pipeline now automatically lowers frequency thresholds and injects domain seeds.
    - Enable **semantic clustering** to merge near‑synonyms (e.g., "grain size" and "grain diameter").
    - Enable **semantic edges** to connect concepts that never co‑occur but are semantically related.
    - If you still see a disconnected graph, add a few more abstracts or increase `similarity_threshold` in the code (0.75 default).
    
    **🔧 Technical Notes:**
    - Adaptive thresholds: freq≥2 for 20‑30 abstracts, freq≥1 for ≤15 abstracts.
    - Domain seeds guarantee coverage of canonical laser/alloy terms.
    - Hybrid graph uses 40% co‑occurrence + 60% semantic weight for small corpora.
    """)

if __name__ == "__main__":
    main()
