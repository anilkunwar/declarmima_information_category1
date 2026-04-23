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
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
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
    r'[A-Z][a-z]?(?:\d+(?:\.\d+)?(?:[A-Z][a-z]?\d*(?:\.\d+)?)*)+',
    r'(?:Ni|Co|Cr|Fe|Al|Ti|Cu|Nb|Mo|W)(?:[-\s]?\d+(?:\.\d+)?%?)+',
    r'(?:high-entropy|HEA|multi-principal|complex concentrated)',
]

# Domain seed concepts for knowledge injection
DOMAIN_SEED_CONCEPTS = {
    "alloy_systems": ["aluminum alloy", "titanium alloy", "nickel alloy", "high-entropy alloy", "steel", "alsi10mg", "ti6al4v", "inconel718"],
    "laser_parameters": ["laser power", "scan speed", "energy density", "hatch spacing", "pulse duration", "melt pool depth"],
    "microstructure_features": ["grain size", "phase fraction", "texture", "porosity", "residual stress", "columnar grain", "equiaxed grain"],
    "mechanical_properties": ["microhardness", "tensile strength", "yield strength", "elongation", "fatigue life"],
    "processes": ["powder bed fusion", "direct energy deposition", "laser remelting", "surface treatment", "solidification"]
}

# Category mapping for hierarchical abstraction
CATEGORY_MAPPING = {
    r'alsi\d+mg|al(?:si|cu|mg|zn)\w*': 'aluminum alloy',
    r'ti6al4v|ti(?:al|nb|mo)\w*': 'titanium alloy', 
    r'inconel\d+|ni(?:cr|mo|fe)\w*': 'nickel alloy',
    r'cocrfeni|he[as]?|high.?entropy': 'high-entropy alloy',
    r'(?:laser\s*)?(?:power|energy\s*density|fluence)': 'laser energy parameter',
    r'(?:scan|travel)\s*speed|feed\s*rate': 'scanning parameter',
    r'hatch\s*spacing|layer\s*thickness': 'geometric parameter',
    r'(?:columnar|equiaxed|dendritic)\s*grain': 'grain morphology',
    r'(?:martensite|austenite|eutectic)\s*(?:phase)?': 'phase type',
    r'(?:micro|nano)hardness|hv\d*': 'hardness metric',
    r'(?:tensile|yield|ultimate)\s*strength': 'strength metric',
}

# Default hyperparameters (will be adapted dynamically)
DEFAULT_MIN_CONCEPT_FREQ = 3
DEFAULT_MIN_CONCEPT_LENGTH_WORDS = 2
GNN_HIDDEN_DIM = 128
TRAIN_EPOCHS = 50
LR = 1e-3
NEG_DPREV_FOCUS = 3

# ==========================================
# ADAPTIVE CONFIGURATION FOR SMALL CORPORA
# ==========================================
def get_adaptive_config(num_abstracts: int):
    """Dynamically adjust parameters based on input corpus size"""
    if num_abstracts <= 15:
        return {
            "MIN_CONCEPT_FREQ": 1,
            "MIN_CONCEPT_LENGTH_WORDS": 1,
            "MIN_DEGREE": 1,
            "USE_SEMANTIC_CLUSTERING": True,
            "INJECT_DOMAIN_SEEDS": True,
            "USE_SEMANTIC_EDGES": True,
            "SIMILARITY_THRESHOLD": 0.70,
            "COOCCURRENCE_WEIGHT": 0.4,
            "SEMANTIC_WEIGHT": 0.6,
            "CLUSTER_SIMILARITY": 0.78,
        }
    elif num_abstracts <= 30:
        return {
            "MIN_CONCEPT_FREQ": 2,
            "MIN_CONCEPT_LENGTH_WORDS": 2,
            "MIN_DEGREE": 1,
            "USE_SEMANTIC_CLUSTERING": True,
            "INJECT_DOMAIN_SEEDS": True,
            "USE_SEMANTIC_EDGES": True,
            "SIMILARITY_THRESHOLD": 0.75,
            "COOCCURRENCE_WEIGHT": 0.6,
            "SEMANTIC_WEIGHT": 0.4,
            "CLUSTER_SIMILARITY": 0.75,
        }
    else:
        return {
            "MIN_CONCEPT_FREQ": 3,
            "MIN_CONCEPT_LENGTH_WORDS": 2,
            "MIN_DEGREE": 2,
            "USE_SEMANTIC_CLUSTERING": False,
            "INJECT_DOMAIN_SEEDS": False,
            "USE_SEMANTIC_EDGES": False,
            "SIMILARITY_THRESHOLD": 0.80,
            "COOCCURRENCE_WEIGHT": 0.8,
            "SEMANTIC_WEIGHT": 0.2,
            "CLUSTER_SIMILARITY": 0.72,
        }

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
    normalized = re.sub(r'(cocrfe)(ni|mn|mo)\w*', r'cocrfeni', normalized)
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
    
    has_domain_keyword = any(kw in concept_lower for kw in DOMAIN_KEYWORDS)
    has_alloy_pattern = any(re.search(p, concept, re.I) for p in ALLOY_PATTERNS)
    
    generic_terms = {'study', 'analysis', 'effect', 'role', 'investigation', 
                     'research', 'method', 'approach', 'paper', 'work', 'using'}
    has_generic = any(term in concept_lower.split() for term in generic_terms)
    
    return (has_domain_keyword or has_alloy_pattern) and not has_generic

# ==========================================
# SEMANTIC CLUSTERING & CONCEPT ABSTRACTION
# ==========================================
def cluster_similar_concepts(valid_concepts, embed_model, similarity_threshold=0.78):
    """Merge semantically similar concepts to boost effective frequency"""
    if len(valid_concepts) < 3:
        return valid_concepts, {c: c for c in valid_concepts}
    
    try:
        embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=32)
        sim_matrix = cosine_similarity(embeddings)
        
        # Hierarchical clustering with distance threshold
        distance_matrix = 1 - sim_matrix
        clustering = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=1 - similarity_threshold,
            linkage='average'
        ).fit(embeddings)
        
        # Map to cluster representatives (shortest concept wins)
        concept_to_cluster = {}
        cluster_members = defaultdict(list)
        
        for idx, label in enumerate(clustering.labels_):
            concept = valid_concepts[idx]
            cluster_members[label].append(concept)
            concept_to_cluster[concept] = label
        
        # Select representative for each cluster
        cluster_representatives = {}
        for label, members in cluster_members.items():
            # Prefer shorter, more frequent-looking concepts
            representative = min(members, key=lambda x: (len(x), -x.count(' ')))
            cluster_representatives[label] = representative
        
        # Create final mapping
        final_mapping = {c: cluster_representatives[label] for c, label in concept_to_cluster.items()}
        return list(cluster_representatives.values()), final_mapping
        
    except Exception as e:
        st.warning(f"⚠️ Semantic clustering skipped: {e}")
        return valid_concepts, {c: c for c in valid_concepts}


def abstract_concepts_to_categories(concepts, category_mapping=CATEGORY_MAPPING):
    """Map specific concepts to broader categories for graph density"""
    concept_to_abstract = {}
    for concept in concepts:
        matched = False
        for pattern, category in category_mapping.items():
            if re.search(pattern, concept, re.I):
                concept_to_abstract[concept] = category
                matched = True
                break
        if not matched:
            concept_to_abstract[concept] = concept
    return concept_to_abstract


def inject_domain_seeds(valid_concepts, concept_to_id, seed_concepts=DOMAIN_SEED_CONCEPTS):
    """Add domain seed concepts even if not frequently observed"""
    all_seeds = [seed for category in seed_concepts.values() for seed in category]
    updated_concepts = valid_concepts.copy()
    updated_mapping = concept_to_id.copy()
    
    for seed in all_seeds:
        if seed not in updated_mapping:
            updated_mapping[seed] = len(updated_mapping)
            updated_concepts.append(seed)
    
    return updated_concepts, updated_mapping

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
        metrics = {}
        
        # Extract quantitative metrics via regex
        grain_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:μm|micron|um|nm)\s*(?:grain|average|size|diameter)?', text, re.I)
        if grain_matches:
            metrics['grain_size_um'] = [float(m) for m in grain_matches]
        
        mech_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:HV|GPa|MPa|ksi)\s*(?:hardness|strength|yield|tensile|ultimate)?', text, re.I)
        if mech_matches:
            metrics['mechanical_property'] = [float(m) for m in mech_matches]
        
        energy_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:J/mm³|J mm-3|J mm⁻³|J/mm\^3)', text, re.I)
        if energy_matches:
            metrics['energy_density_j_mm3'] = [float(m) for m in energy_matches]
        
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
        
        concepts = []
        try:
            parsed = json.loads(response.replace("'", '"').strip())
            if isinstance(parsed, list):
                concepts = [c.strip().lower().rstrip('.') for c in parsed if isinstance(c, str) and len(c.strip()) > 3]
        except (json.JSONDecodeError, TypeError):
            concepts = _fallback_concept_extraction(text)
        
        # Normalize and filter
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
    """Regex fallback for concept extraction when LLM parsing fails"""
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


def normalize_and_filter_concepts(all_concepts, embed_model=None, config=None):
    """Adaptive concept filtering with semantic clustering and seed injection"""
    if config is None:
        config = get_adaptive_config(25)  # Default fallback
    
    concept_counts = defaultdict(int)
    concept_abstract_map = defaultdict(list)
    
    for doc_idx, concepts in enumerate(all_concepts):
        seen_in_doc = set()
        for c in concepts:
            if c not in seen_in_doc and is_valid_microstructure_concept(c):
                concept_counts[c] += 1
                concept_abstract_map[c].append(doc_idx)
                seen_in_doc.add(c)
    
    # Initial filter with adaptive thresholds
    min_freq = config.get("MIN_CONCEPT_FREQ", 2)
    min_words = config.get("MIN_CONCEPT_LENGTH_WORDS", 2)
    
    valid_concepts = [c for c, cnt in concept_counts.items() 
                      if cnt >= min_freq and len(c.split()) >= min_words]
    
    # Inject domain seeds if sparse
    if config.get("INJECT_DOMAIN_SEEDS", True) and len(valid_concepts) < 15:
        valid_concepts, concept_to_id = inject_domain_seeds(
            valid_concepts, {c: i for i, c in enumerate(valid_concepts)}
        )
        # Update counts for injected seeds
        for seed in [s for cat in DOMAIN_SEED_CONCEPTS.values() for s in cat]:
            if seed not in concept_counts:
                concept_counts[seed] = 1
                concept_abstract_map[seed] = []
    
    # Semantic clustering to merge similar concepts
    if config.get("USE_SEMANTIC_CLUSTERING", True) and embed_model and len(valid_concepts) >= 5:
        clustered_concepts, concept_to_cluster = cluster_similar_concepts(
            valid_concepts, embed_model, 
            similarity_threshold=config.get("CLUSTER_SIMILARITY", 0.75)
        )
        # Remap abstract map to clustered concepts
        new_abstract_map = defaultdict(list)
        for orig_concept, docs in concept_abstract_map.items():
            clustered = concept_to_cluster.get(orig_concept, orig_concept)
            if clustered in clustered_concepts:
                new_abstract_map[clustered].extend(docs)
        concept_abstract_map = new_abstract_map
        valid_concepts = clustered_concepts
    
    # Final deduplication and ID mapping
    valid_concepts = list(set(valid_concepts))
    concept_to_id = {c: i for i, c in enumerate(valid_concepts)}
    id_to_concept = {i: c for i, c in enumerate(valid_concepts)}
    
    return valid_concepts, concept_to_id, id_to_concept, concept_abstract_map

# ==========================================
# STEP 3: HYBRID CONCEPT GRAPH CONSTRUCTION
# ==========================================
def build_semantic_only_graph(valid_concepts, embed_model, similarity_threshold=0.75):
    """Fallback: create graph purely from embedding similarity"""
    nx_graph = nx.Graph()
    for c in valid_concepts:
        nx_graph.add_node(c)
    
    if len(valid_concepts) < 2:
        return nx_graph
        
    try:
        embeddings = embed_model.encode(valid_concepts, show_progress_bar=False)
        sim_matrix = cosine_similarity(embeddings)
        
        for i in range(len(valid_concepts)):
            for j in range(i+1, len(valid_concepts)):
                if sim_matrix[i][j] > similarity_threshold:
                    nx_graph.add_edge(
                        valid_concepts[i], valid_concepts[j], 
                        weight=sim_matrix[i][j], 
                        edge_type='semantic',
                        cooccurrence=0,
                        semantic=sim_matrix[i][j]
                    )
        
        # Ensure connectivity via minimum spanning tree if fragmented
        if not nx.is_connected(nx_graph) and len(valid_concepts) > 3:
            components = list(nx.connected_components(nx_graph))
            for i in range(len(components)-1):
                best_sim, best_pair = 0, None
                for c1 in components[i]:
                    idx1 = valid_concepts.index(c1)
                    for c2 in components[i+1]:
                        idx2 = valid_concepts.index(c2)
                        if sim_matrix[idx1][idx2] > best_sim:
                            best_sim = sim_matrix[idx1][idx2]
                            best_pair = (c1, c2)
                if best_pair:
                    nx_graph.add_edge(*best_pair, weight=best_sim, edge_type='bridge',
                                     cooccurrence=0, semantic=best_sim)
    except Exception as e:
        st.warning(f"⚠️ Semantic graph construction issue: {e}")
        # Return minimal connected graph
        for i in range(len(valid_concepts)-1):
            nx_graph.add_edge(valid_concepts[i], valid_concepts[i+1], weight=1.0)
    
    return nx_graph


def build_hybrid_graph(all_concepts, valid_concepts, concept_to_id, embed_model=None, config=None):
    """Hybrid graph: combine co-occurrence with embedding similarity"""
    if config is None:
        config = get_adaptive_config(len(all_concepts))
    
    nx_graph = nx.Graph()
    for c in valid_concepts:
        nx_graph.add_node(c, frequency=0)
    
    # Step 1: Add co-occurrence edges
    for concepts in all_concepts:
        valid_in_doc = [c for c in concepts if c in concept_to_id]
        for i in range(len(valid_in_doc)):
            for j in range(i+1, len(valid_in_doc)):
                u, v = valid_in_doc[i], valid_in_doc[j]
                if nx_graph.has_edge(u, v):
                    nx_graph[u][v]['weight'] += 1
                    nx_graph[u][v]['cooccurrence'] += 1
                else:
                    nx_graph.add_edge(u, v, weight=1, cooccurrence=1, semantic=0, edge_type='cooccurrence')
                nx_graph.nodes[u]['frequency'] = nx_graph.nodes[u].get('frequency', 0) + 1
                nx_graph.nodes[v]['frequency'] = nx_graph.nodes[v].get('frequency', 0) + 1
    
    # Step 2: Add semantic similarity edges for low-connectivity nodes
    if config.get("USE_SEMANTIC_EDGES", True) and embed_model and len(valid_concepts) >= 5:
        try:
            embeddings = embed_model.encode(valid_concepts, show_progress_bar=False)
            sim_matrix = cosine_similarity(embeddings)
            sim_thresh = config.get("SIMILARITY_THRESHOLD", 0.75)
            
            for i, c1 in enumerate(valid_concepts):
                for j, c2 in enumerate(valid_concepts[i+1:], start=i+1):
                    if c1 == c2 or nx_graph.has_edge(c1, c2):
                        continue
                    sim = sim_matrix[i][j]
                    # Add edge if semantically similar AND at least one has low degree
                    if sim > sim_thresh and (nx_graph.degree(c1) < 2 or nx_graph.degree(c2) < 2):
                        semantic_weight = sim * 2
                        nx_graph.add_edge(c1, c2, weight=semantic_weight, 
                                         cooccurrence=0, semantic=sim, edge_type='semantic')
        except Exception as e:
            st.warning(f"⚠️ Semantic edge addition skipped: {e}")
    
    # Step 3: Combine weights using configured ratios
    cooc_weight = config.get("COOCCURRENCE_WEIGHT", 0.6)
    sem_weight = config.get("SEMANTIC_WEIGHT", 0.4)
    
    for u, v, data in nx_graph.edges(data=True):
        cooc = data.get('cooccurrence', 0)
        sem = data.get('semantic', 0)
        data['weight'] = cooc_weight * cooc + sem_weight * sem
    
    return nx_graph


def build_concept_graph(all_concepts, concept_to_id, embed_model=None, config=None):
    """Main graph builder with fallback strategies"""
    if config is None:
        config = get_adaptive_config(len(all_concepts))
    
    valid_concepts = list(concept_to_id.keys())
    
    # If very few concepts, use semantic-only graph
    if len(valid_concepts) < 8 and config.get("USE_SEMANTIC_EDGES", True):
        return build_semantic_only_graph(valid_concepts, embed_model, 
                                        similarity_threshold=config.get("SIMILARITY_THRESHOLD", 0.75))
    
    # Otherwise use hybrid approach
    return build_hybrid_graph(all_concepts, valid_concepts, concept_to_id, embed_model, config)


def sample_edges_for_training(nx_graph, d_prev_dict, valid_concepts, concept_to_id, config=None):
    """Sample positive and negative edges for contrastive training"""
    if config is None:
        config = get_adaptive_config(len(valid_concepts))
    
    pos_pairs = [(concept_to_id[u], concept_to_id[v]) for u, v in nx_graph.edges()]
    neg_pairs = []
    
    n_nodes = len(valid_concepts)
    if n_nodes < 3:
        return pos_pairs, neg_pairs
    
    target_negs = min(len(pos_pairs) * 2 if pos_pairs else 10, 2000)
    attempts = 0
    neg_focus = config.get("NEG_DPREV_FOCUS", 3)
    
    while len(neg_pairs) < target_negs and attempts < 15000:
        u_idx, v_idx = np.random.choice(n_nodes, 2, replace=False)
        u_concept, v_concept = valid_concepts[u_idx], valid_concepts[v_idx]
        
        if nx_graph.has_edge(u_concept, v_concept):
            attempts += 1
            continue
        
        try:
            dist = d_prev_dict[u_concept][v_concept]
            if dist == neg_focus:
                neg_pairs.append((u_idx, v_idx))
            elif dist == 2 and np.random.rand() < 0.3:
                neg_pairs.append((u_idx, v_idx))
        except KeyError:
            if np.random.rand() < 0.1:  # Higher probability for sparse graphs
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
    if not valid_concepts:
        return torch.zeros((0, 384), dtype=torch.float32).to(DEVICE)
    embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=32)
    return torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)

# ==========================================
# STEP 5: PURE PYTORCH SPARSE GRAPHSAGE
# ==========================================
class SparseGraphSAGE(nn.Module):
    """Memory-efficient GraphSAGE using PyTorch sparse tensors"""
    
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
    """Train GraphSAGE with contrastive edge prediction loss"""
    
    if not pos_pairs:
        # Create minimal training pairs if graph is empty
        nodes = list(concept_to_id.values())
        if len(nodes) >= 2:
            pos_pairs = [(nodes[0], nodes[1])]
            neg_pairs = [(nodes[0], nodes[-1])] if len(nodes) > 2 else []
        else:
            raise ValueError("Cannot train GNN with fewer than 2 concepts")
    
    unique_edges = {(min(u, v), max(u, v)) for u, v in pos_pairs}
    src_adj = torch.tensor([u for u, v in unique_edges], dtype=torch.long)
    dst_adj = torch.tensor([v for u, v in unique_edges], dtype=torch.long)
    adj_indices = torch.stack([src_adj, dst_adj], dim=0)
    adj_values = torch.ones(adj_indices.shape[1], dtype=torch.float32)
    
    pos_u = torch.tensor([p[0] for p in pos_pairs], dtype=torch.long, device=DEVICE)
    pos_v = torch.tensor([p[1] for p in pos_pairs], dtype=torch.long, device=DEVICE)
    neg_u = torch.tensor([n[0] for n in neg_pairs], dtype=torch.long, device=DEVICE) if neg_pairs else torch.tensor([], dtype=torch.long, device=DEVICE)
    neg_v = torch.tensor([n[1] for n in neg_pairs], dtype=torch.long, device=DEVICE) if neg_pairs else torch.tensor([], dtype=torch.long, device=DEVICE)
    
    model = SparseGraphSAGE(node_features.shape[1]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(TRAIN_EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # Handle case with no negative samples
        if len(neg_pairs) == 0:
            pos_out, _, _ = model(adj_indices, adj_values, len(concept_to_id), node_features, pos_u, pos_v, pos_u[:1], pos_v[:1])
            loss = criterion(pos_out, torch.ones_like(pos_out)) * 0.5
        else:
            pos_out, neg_out, _ = model(adj_indices, adj_values, len(concept_to_id), node_features, pos_u, pos_v, neg_u, neg_v)
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
            node_features, pos_u[:1], pos_v[:1], 
            neg_u[:1] if len(neg_pairs) > 0 else pos_u[:1], 
            neg_v[:1] if len(neg_pairs) > 0 else pos_v[:1]
        )
    
    return model, final_embeddings.cpu(), adj_indices, adj_values

# ==========================================
# STEP 6: MICROSTRUCTURE QUANTIFICATION & SCORING
# ==========================================
def compute_microstructure_quantification(valid_concepts, concept_abstract_map, all_metrics, nx_graph):
    """Map concepts to representative microstructure property values"""
    concept_properties = {}
    
    for concept in valid_concepts:
        doc_indices = concept_abstract_map.get(concept, [])
        values = []
        for idx in doc_indices:
            if idx < len(all_metrics):
                metrics = all_metrics[idx]
                for metric_values in metrics.values():
                    values.extend(metric_values)
        concept_properties[concept] = np.median(values) if values else 0.0
    
    X_feat, y_target = [], []
    for u, v in nx_graph.edges():
        pu, pv = concept_properties.get(u, 0), concept_properties.get(v, 0)
        w = nx_graph[u][v].get('weight', 1)
        X_feat.append([pu, pv, w])
        y_target.append(max(pu, pv) * 1.08 if max(pu, pv) > 0 else 0)
    
    ridge = None
    if len(X_feat) > 5:
        ridge = Ridge(alpha=1.0).fit(np.array(X_feat), np.array(y_target))
    
    return concept_properties, ridge


def compute_research_direction_scores(
    model, final_emb, nx_graph, valid_concepts, concept_properties, 
    ridge, embed_model, d_prev_dict, adj_indices, adj_values,
    n_samples=3000
):
    """
    Score novel concept pairs for promising microstructure research directions.
    
    FIXED: Uses pre-computed GNN embeddings directly instead of re-running forward pass.
    final_emb has shape [n_concepts, hidden_dim=128] - already aggregated by GraphSAGE.
    """
    n_concepts = len(valid_concepts)
    if n_concepts < 3:
        return pd.DataFrame()
    
    u_ids = np.random.randint(n_concepts, size=min(n_samples, n_concepts * 10))
    v_ids = np.random.randint(n_concepts, size=min(n_samples, n_concepts * 10))
    
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
    
    # ✅ FIX: Use pre-computed GNN embeddings directly (already on CPU, move to device)
    h2 = final_emb.to(DEVICE)  # shape: [n_concepts, GNN_HIDDEN_DIM=128]
    
    # Look up embeddings for candidate pairs
    u_emb = h2[u_tensor]  # [num_pairs, 128]
    v_emb = h2[v_tensor]  # [num_pairs, 128]
    
    # Concatenate and score through decoder only (decoder expects [batch, 256])
    pair_features = torch.cat([u_emb, v_emb], dim=1)  # [num_pairs, 256]
    gnn_logits = model.decoder(pair_features).squeeze(1)  # [num_pairs]
    gnn_scores = torch.sigmoid(gnn_logits).cpu().numpy()
    
    # Semantic novelty using original embeddings
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
        
        expected_improvement = 0
        if ridge is not None and (p_u > 0 or p_v > 0):
            try:
                expected_improvement = float(ridge.predict([[p_u, p_v, 1.0]])[0])
            except:
                expected_improvement = max(p_u, p_v) * 1.05
        
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
    return df.head(min(50, len(df)))

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
2. Target outcome: Predicted microstructure/property improvement and key trade-off.
3. Validation step: One concrete experimental method (e.g., EBSD, nanoindentation, in-situ XRD).

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
                inputs.input_ids, max_new_tokens=180, temperature=0.25, do_sample=True,
                pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.1
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
def main():
    st.set_page_config(page_title="Alloy Microstructure Concept Graph", layout="wide")
    st.title("🔬 Laser & Multicomponent Alloy Microstructure Analyzer")
    st.caption("Small-corpus optimized: Works with 10-30 abstracts via semantic enrichment + knowledge injection")
    
    with st.sidebar:
        st.header("⚙️ Small Corpus Mode")
        
        # Auto-detect corpus size and suggest settings
        abstract_preview = st.text_area("📋 Paste abstracts here (preview):", height=100, key="preview")
        preview_count = len([t for t in re.split(r'\n\s*\n', abstract_preview) if t.strip()]) if abstract_preview.strip() else 0
        
        if preview_count > 0 and preview_count <= 25:
            st.warning(f"📉 Small corpus detected ({preview_count} abstracts): applying adaptive settings")
            st.toggle("Enable semantic clustering", value=True, key="use_clustering", disabled=True)
            st.toggle("Inject domain seed concepts", value=True, key="inject_seeds", disabled=True)
            st.toggle("Use embedding-based edges", value=True, key="semantic_edges", disabled=True)
        else:
            st.toggle("Enable semantic clustering", value=False, key="use_clustering")
            st.toggle("Inject domain seed concepts", value=False, key="inject_seeds")
            st.toggle("Use embedding-based edges", value=False, key="semantic_edges")
        
        st.markdown("---")
        st.markdown("**Domain Focus:**")
        st.markdown("- ✅ Alloys: AlSi10Mg, Ti6Al4V, HEAs, IN718")
        st.markdown("- ✅ Laser: power, speed, energy density, melt pool")
        st.markdown("- ✅ Microstructure: grains, phases, texture, defects")
        st.markdown("- ✅ Properties: hardness, strength, ductility")
    
    abstract_input = st.text_area(
        "📋 Paste scientific abstracts (one per block, separated by blank lines):", 
        height=300,
        placeholder="""Example:
"Laser powder bed fusion of AlSi10Mg reveals columnar-to-equiaxed transition at 85 J/mm³, grain refinement 45→12 μm..."

"High-entropy alloy CoCrFeNiMo via DED shows 420 HV microhardness from nanoscale precipitates..."

"Residual stress mitigation in Ti6Al4V via laser remelting: EBSD reveals <001>→<111> texture evolution..."
""")
    
    if st.button("🚀 Analyze Abstracts", type="primary", use_container_width=True):
        if not abstract_input.strip():
            st.error("⚠️ Please enter at least one scientific abstract.")
            return
            
        abstracts = [t.strip() for t in re.split(r'\n\s*\n', abstract_input) if t.strip()]
        
        if len(abstracts) < 10:
            st.info(f"💡 {len(abstracts)} abstracts: Using maximum semantic enrichment mode")
        elif len(abstracts) > 35:
            st.warning(f"⚠️ {len(abstracts)} abstracts may increase processing time")
            
        progress_bar = st.progress(0.0)
        status = st.status("🔄 Initializing pipeline...", expanded=True)
        
        try:
            # Load models
            with status:
                st.write("📦 Loading models...")
                embed_model = load_embedding_model()
                tokenizer, llm_model = load_lightweight_llm()
                st.success("✅ Models loaded")
            progress_bar.progress(0.10)
            
            # Get adaptive config
            config = get_adaptive_config(len(abstracts))
            # Override with user toggles if available
            if "use_clustering" in st.session_state:
                config["USE_SEMANTIC_CLUSTERING"] = st.session_state.use_clustering
            if "inject_seeds" in st.session_state:
                config["INJECT_DOMAIN_SEEDS"] = st.session_state.inject_seeds
            if "semantic_edges" in st.session_state:
                config["USE_SEMANTIC_EDGES"] = st.session_state.semantic_edges
            
            # Step 1-2: Extract concepts
            with st.status("🔍 Extracting concepts & metrics..."):
                all_concepts, all_metrics = extract_concepts_from_abstracts(abstracts, tokenizer, llm_model)
                valid_concepts, concept_to_id, id_to_concept, concept_abstract_map = normalize_and_filter_concepts(
                    all_concepts, embed_model, config
                )
                st.write(f"✅ **{len(valid_concepts)}** concepts extracted")
                if len(valid_concepts) < 10:
                    st.info("💡 Small concept set: using semantic-only graph mode")
            progress_bar.progress(0.25)
            
            # Fallback if still too few concepts
            if len(valid_concepts) < 3:
                st.warning("⚠️ Very few concepts. Injecting additional domain seeds...")
                valid_concepts, concept_to_id = inject_domain_seeds(
                    valid_concepts, concept_to_id
                )
                st.success(f"✅ Recovered {len(valid_concepts)} concepts via seed injection")
            
            # Step 3: Build graph
            with st.status("🕸️ Building concept graph..."):
                nx_graph = build_concept_graph(all_concepts, concept_to_id, embed_model, config)
                d_prev_dict = dict(nx.all_pairs_shortest_path_length(nx_graph, cutoff=4))
                pos_pairs, neg_pairs = sample_edges_for_training(nx_graph, d_prev_dict, valid_concepts, concept_to_id, config)
                st.write(f"✅ Graph: **{len(valid_concepts)}** nodes, **{nx_graph.number_of_edges()}** edges")
                
                # Connectivity check
                if not nx.is_connected(nx_graph):
                    n_comp = nx.number_connected_components(nx_graph)
                    st.info(f"🔗 Graph has {n_comp} component(s) - using bridge edges for connectivity")
            progress_bar.progress(0.40)
            
            # Step 4: Embeddings
            with st.status("🧠 Generating embeddings..."):
                node_features = generate_embeddings(valid_concepts, embed_model)
                st.write(f"✅ Dimension: {node_features.shape[1]}")
            progress_bar.progress(0.50)
            
            # Step 5: Train GNN
            def _training_progress(epoch, loss):
                progress_value = 0.50 + (epoch / TRAIN_EPOCHS) * 0.30
                progress_bar.progress(min(1.0, max(0.0, progress_value)))
                if epoch % 10 == 0:
                    status.write(f"📊 Epoch {epoch}/{TRAIN_EPOCHS} | Loss: {loss:.4f}")
            
            with st.status("🤖 Training GraphSAGE..."):
                gnn_model, final_emb, adj_indices, adj_values = train_gnn(
                    node_features, nx_graph, concept_to_id, pos_pairs, neg_pairs, _training_progress
                )
                st.success("✅ GNN training complete")
            progress_bar.progress(0.80)
            
            # Step 6: Scoring
            with st.status("📈 Scoring novel directions..."):
                concept_properties, ridge = compute_microstructure_quantification(
                    valid_concepts, concept_abstract_map, all_metrics, nx_graph
                )
                top_scores = compute_research_direction_scores(
                    gnn_model, final_emb, nx_graph, valid_concepts, concept_properties,
                    ridge, embed_model, d_prev_dict, adj_indices, adj_values
                )
                st.write(f"✅ Scored **{len(top_scores)}** novel pairs")
            progress_bar.progress(0.90)
            
            # Step 7: LLM curation
            with st.status("✍️ Generating hypotheses..."):
                directions_df = generate_research_directions(top_scores, tokenizer, llm_model)
                st.success("✅ Pipeline complete!")
            progress_bar.progress(1.00)
            status.update(label="✅ Analysis complete!", state="complete", expanded=False)
            
            # === DISPLAY RESULTS ===
            st.subheader("🎯 Top Research Directions")
            if len(directions_df) > 0:
                st.dataframe(
                    directions_df[['Concept Pair', 'Composite Score', 'Expected Gain', 'Feasibility', 'Research Hypothesis']],
                    use_container_width=True,
                    column_config={
                        "Concept Pair": st.column_config.TextColumn("Pair", width="medium"),
                        "Composite Score": st.column_config.NumberColumn("Score", format="%.3f"),
                        "Expected Gain": st.column_config.NumberColumn("Gain", format="%.1f"),
                        "Feasibility": st.column_config.NumberColumn("Feas.", format="%.2f"),
                        "Research Hypothesis": st.column_config.TextColumn("Hypothesis", width="large")
                    }
                )
                csv = directions_df.to_csv(index=False)
                st.download_button("📥 Download CSV", data=csv, file_name="alloy_directions.csv", mime="text/csv")
            else:
                st.info("💡 No novel pairs scored above threshold. Try adjusting parameters or adding more abstracts.")
            
            # Interactive graph
            st.subheader("🌐 Concept Graph")
            net = Network(height="650px", width="100%", bgcolor="#1e1e1e", font_color="white", select_menu=True)
            net.barnes_hut(gravity=-80000, spring_length=200)
            
            for node in nx_graph.nodes():
                deg = nx_graph.degree(node)
                size = max(12, min(50, deg * 4 + 10))
                freq = len(concept_abstract_map.get(node, []))
                
                # Color coding
                if any(a in node.lower() for a in ['al', 'ti', 'ni', 'cr', 'fe', 'co', 'mo']):
                    color = "#4CAF50"
                elif any(l in node.lower() for l in ['laser', 'scan', 'power', 'melt', 'energy']):
                    color = "#2196F3"
                elif any(m in node.lower() for m in ['grain', 'phase', 'hardness', 'strength']):
                    color = "#FF9800"
                else:
                    color = "#9E9E9E"
                
                net.add_node(node, label=node, size=size, color=color,
                           title=f"{node}\nDegree: {deg}\nFreq: {freq}")
            
            for u, v in nx_graph.edges():
                w = nx_graph[u][v].get('weight', 1)
                edge_type = nx_graph[u][v].get('edge_type', 'unknown')
                color = "#66cc66" if edge_type == 'cooccurrence' else "#6699ff" if edge_type == 'semantic' else "#cccc66"
                net.add_edge(u, v, value=w, width=min(4, w * 0.8), color=color)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                net.save_graph(tmp.name)
                with open(tmp.name, "r", encoding="utf-8") as f:
                    st.components.v1.html(f.read(), height=700, scrolling=True)
            
            # Diagnostics
            with st.expander("📊 Graph Diagnostics", expanded=len(valid_concepts) < 30):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Nodes", len(valid_concepts))
                col2.metric("Edges", nx_graph.number_of_edges())
                col3.metric("Avg Degree", f"{np.mean([d for _,d in nx_graph.degree()]):.2f}")
                col4.metric("Connected", "✅" if nx.is_connected(nx_graph) else f"❌ ({nx.number_connected_components()} comps)")
                
                if len(valid_concepts) > 0:
                    st.markdown("### Top Concepts")
                    freq_data = [(c, len(concept_abstract_map.get(c, []))) for c in valid_concepts]
                    freq_data.sort(key=lambda x: x[1], reverse=True)
                    st.dataframe(pd.DataFrame(freq_data[:10], columns=["Concept", "Frequency"]), use_container_width=True)
                    
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            with st.expander("🔍 Traceback"):
                st.code(traceback.format_exc())
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **💡 Tips for Small Corpora (10-25 abstracts):**
    - ✅ Semantic clustering merges similar terms (e.g., "grain size" + "grain diameter")
    - ✅ Domain seed injection adds known alloy/laser terms even if not in your abstracts
    - ✅ Embedding-based edges connect semantically related concepts without co-occurrence
    - ✅ Adaptive thresholds relax frequency requirements for sparse data
    
    **🔧 Technical:** Qwen2.5-0.5B + Sentence-BERT + Pure PyTorch GraphSAGE | All local processing
    """)

if __name__ == "__main__":
    main()
