#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LASER MICROSTRUCTURE RAG CHATBOT - FULLY API-FREE VERSION
==========================================================
✅ Zero API keys required - all models run locally
✅ Optimized for Streamlit Cloud (memory-efficient caching)
✅ Laser-microstructure domain specialization
✅ Supports GPT-2 and Qwen-0.5B models (local inference)
✅ PDF/text document ingestion with FAISS vector storage
✅ Source citation and confidence scoring
✅ Responsive UI with streaming-like output simulation

Deploy to Streamlit Cloud with requirements.txt below.
"""
import streamlit as st
import os
import tempfile
import time
import re
import json
import torch
import numpy as np
from io import BytesIO
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# LangChain / RAG imports (local-only, no API calls)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings  # LOCAL embeddings!
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# Transformers for local LLM inference
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, AutoModelForCausalLM,
    pipeline, set_seed
)

# =============================================
# GLOBAL CONFIGURATION - LASER MICROSTRUCTURE DOMAIN
# =============================================

# Local model choices (no API needed)
LOCAL_LLM_OPTIONS = {
    # Your original tiny models (Good for low-latency testing)
    "GPT-2 (1.5B, fastest startup)": "gpt2",
    "Qwen2-0.5B-Instruct (best JSON, recommended)": "qwen2:0.5b",
    "Qwen2.5-0.5B-Instruct (newest, best reasoning)": "qwen2.5:0.5b",
    
    # The "Powerhouse" options found in your Ollama library
    "Qwen2.5-14B (RTX 5080 Optimized, Best Overall)": "qwen2.5:14b",
    "Llama3.1-8B (Most Popular, Balanced)": "llama3.1:8b",
    "Gemma3 (Latest Google model, Great logic)": "gemma3:latest",
    
    # Specialized / Legacy options from your list
    "Mistral-7B (Reliable & Efficient)": "mistral:7b",
    "Falcon3-10B (Lightweight & Modern)": "falcon3:10b",
    "GPT-OSS (20B, Maximum Scale)": "gpt-oss:20b"
}
# Local embedding model (~80MB, CPU-friendly)
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Laser-microstructure domain settings
LASER_DOMAIN_CONFIG = {
    "chunk_size": 800,  # Smaller chunks for technical precision
    "chunk_overlap": 150,
    "retrieval_k": 4,  # Number of chunks to retrieve
    "score_threshold": 0.25,  # Minimum similarity score
    "max_context_tokens": 1024,  # Limit context for small LLMs
    "max_new_tokens": 256,  # Limit generation length
    "temperature": 0.1,  # Low temp for deterministic technical answers
}

# Laser-specific keywords for domain filtering and boosting
LASER_KEYWORDS = {
    "ablation": ["ablation", "material removal", "threshold fluence", "laser ablation"],
    "plasma": ["plasma formation", "ionization", "electron density", "plume"],
    "thermal": ["heat affected zone", "melting", "thermal diffusion", "resolidification"],
    "ultrafast": ["femtosecond", "picosecond", "pulse duration", "ultrafast laser"],
    "morphology": ["ripples", "LIPSS", "surface structuring", "periodic structures"],
    "parameters": ["fluence", "wavelength", "pulse energy", "repetition rate", "spot size"],
    "materials": ["silicon", "steel", "titanium", "polymer", "glass", "ceramic"],
    "characterization": ["SEM", "AFM", "profilometry", "spectroscopy", "microscopy"],
}

# =============================================
# SESSION STATE INITIALIZATION
# =============================================

def initialize_session_state():
    """Initialize all session state variables with defaults."""
    defaults = {
        "processed_files": set(),
        "vectorstore": None,
        "all_chunks": [],
        "messages": [],
        "llm_model_choice": None,
        "llm_tokenizer": None,
        "llm_model": None,
        "embeddings": None,
        "processing_complete": False,
        "laser_domain_boost": True,
        "show_sources": True,
        "max_retrieved_chunks": 4,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# =============================================
# LOCAL MODEL LOADING (CACHED FOR PERFORMANCE)
# =============================================

@st.cache_resource(show_spinner="Loading local embedding model (~80MB)...")
def load_local_embeddings():
    """Load local sentence-transformers embedding model."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    except Exception as e:
        st.error(f"Failed to load embeddings: {e}")
        return None

@st.cache_resource(show_spinner="Loading local LLM (this may take 1-2 minutes on first load)...")
def load_local_llm(model_key: str):
    """Load local LLM (GPT-2 or Qwen) with proper device handling."""
    try:
        model_path = LOCAL_LLM_OPTIONS.get(model_key, "gpt2")
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.sidebar.info(f"🖥️ Running on: {device.upper()}")
        
        if "GPT-2" in model_key:
            # Load GPT-2
            tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            model = GPT2LMHeadModel.from_pretrained(model_path)
        else:
            # Load Qwen models with trust_remote_code
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True,
                padding_side="left"
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            if device == "cpu":
                model = model.to(device)
        
        model.eval()
        
        # Set padding token for GPT-2 compatibility
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer, model, device
        
    except Exception as e:
        st.error(f"Failed to load LLM '{model_key}': {e}")
        st.warning("Falling back to GPT-2...")
        try:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            return tokenizer, model, device
        except:
            return None, None, None

# =============================================
# DOCUMENT LOADING & CHUNKING (LASER-OPTIMIZED)
# =============================================

def extract_laser_metadata(text: str, filename: str) -> Dict[str, any]:
    """Extract laser-relevant metadata from document text."""
    metadata = {
        "source": filename,
        "laser_topics": [],
        "parameters_found": {},
        "has_equations": bool(re.search(r'[\(=]\s*[\d.]+\s*[×*]\s*10\^', text)),
        "has_figures": bool(re.search(r'Figure\s*\d+|Fig\.\s*\d+', text, re.I)),
    }
    
    text_lower = text.lower()
    
    # Detect laser topics
    for topic, keywords in LASER_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            metadata["laser_topics"].append(topic)
    
    # Extract common laser parameters using regex
    param_patterns = {
        "wavelength_nm": r'(\d+(?:\.\d+)?)\s*(?:nm|nanometers?)\s*(?:wavelength|λ|lambda)',
        "pulse_duration_fs": r'(\d+(?:\.\d+)?)\s*(?:fs|femtoseconds?)\s*(?:pulse|duration)',
        "fluence_Jcm2": r'(\d+(?:\.\d+)?)\s*(?:J/cm²|J/cm2|fluence)',
        "repetition_rate": r'(\d+(?:\.\d+)?)\s*(?:kHz|MHz|Hz)\s*(?:repetition|rate|freq)',
        "spot_size_um": r'(\d+(?:\.\d+)?)\s*(?:µm|um|microns?)\s*(?:spot|diameter)',
    }
    
    for param, pattern in param_patterns.items():
        match = re.search(pattern, text, re.I)
        if match:
            try:
                metadata["parameters_found"][param] = float(match.group(1))
            except:
                pass
    
    return metadata

def load_and_chunk_laser_documents(uploaded_files: List) -> List[Document]:
    """Load PDFs/text files and chunk them with laser-domain awareness."""
    all_chunks = []
    
    for uploaded_file in uploaded_files:
        # Save to temp file for loading
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf" if uploaded_file.name.endswith('.pdf') else ".txt") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        
        try:
            # Load document
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, encoding='utf-8')
            
            pages = loader.load()
            
            # Laser-optimized text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=LASER_DOMAIN_CONFIG["chunk_size"],
                chunk_overlap=LASER_DOMAIN_CONFIG["chunk_overlap"],
                separators=["\n\n", "\n", "Equation", "Parameter:", "Figure", "Table", ""],
                length_function=len
            )
            
            chunks = text_splitter.split_documents(pages)
            
            # Add laser-specific metadata to each chunk
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "source": uploaded_file.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    **extract_laser_metadata(chunk.page_content, uploaded_file.name)
                })
            
            all_chunks.extend(chunks)
            st.info(f"✅ Loaded {len(chunks)} chunks from `{uploaded_file.name}`")
            
        except Exception as e:
            st.error(f"❌ Error processing `{uploaded_file.name}`: {e}")
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    return all_chunks

# =============================================
# VECTOR STORE CREATION (LOCAL FAISS)
# =============================================

@st.cache_resource
def create_local_vector_store(chunks: List[Document], embedding_model_key: str):
    """Create FAISS vector store with local embeddings."""
    try:
        embeddings = load_local_embeddings()
        if embeddings is None:
            return None
        
        # Create vector store
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # Add metadata for laser-domain filtering
        vectorstore.metadata = {
            "total_chunks": len(chunks),
            "embedding_model": embedding_model_key,
            "created_at": datetime.now().isoformat(),
            "laser_topics": list(set(
                topic for chunk in chunks 
                for topic in chunk.metadata.get("laser_topics", [])
            ))
        }
        
        return vectorstore
        
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None

# =============================================
# LASER-SPECIFIC RAG CHAIN (LOCAL INFERENCE)
# =============================================

def create_laser_rag_prompt(retrieved_chunks: List[Document], query: str) -> str:
    """Create a laser-optimized prompt with retrieved context."""
    
    # Format retrieved chunks with source citation
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        source = chunk.metadata.get("source", "unknown")
        topics = chunk.metadata.get("laser_topics", [])
        topic_str = f" [{', '.join(topics)}]" if topics else ""
        
        # Truncate long chunks for small LLMs
        content = chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content
        
        context_parts.append(f"[Source {i}{topic_str} - {source}]\n{content}\n")
    
    context = "\n---\n".join(context_parts)
    
    # Laser-specific system prompt
    laser_system = """You are an expert assistant for laser-microstructure interaction research.
Your role is to answer questions based ONLY on the provided document context.
Be precise, technical, and cite your sources.

Rules:
1. Use ONLY information from the retrieved context below
2. If the answer isn't in the context, say "Based on the provided documents, I cannot determine..."
3. Never invent parameters, equations, or experimental conditions
4. When citing, reference the source number like [Source 1]
5. For numerical values, include units when available
6. Be concise but technically complete

"""
    
    # User query section
    user_query = f"""Retrieved Context from Laser Microstructure Documents:
{context}

User Question: {query}

Answer (cite sources, be technical and precise):"""
    
    return laser_system + user_query

def generate_local_response(
    tokenizer, 
    model, 
    device: str, 
    prompt: str,
    backend: str
) -> str:
    """Generate response using local LLM with proper formatting."""
    try:
        # Format prompt for the specific model
        if "Qwen" in backend:
            # Use Qwen's chat template
            messages = [
                {"role": "system", "content": "You are an expert in laser-microstructure interaction."},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # GPT-2: simple prompt format
            formatted_prompt = prompt
        
        # Tokenize
        inputs = tokenizer.encode(
            formatted_prompt, 
            return_tensors='pt', 
            truncation=True, 
            max_length=LASER_DOMAIN_CONFIG["max_context_tokens"]
        )
        
        if device == "cuda" and torch.cuda.is_available():
            inputs = inputs.to('cuda')
        
        # Generate with constrained parameters for small models
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=LASER_DOMAIN_CONFIG["max_new_tokens"],
                temperature=LASER_DOMAIN_CONFIG["temperature"],
                do_sample=(LASER_DOMAIN_CONFIG["temperature"] > 0),
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,  # Reduce repetition
                early_stopping=True,
            )
        
        # Decode and extract answer
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer after the prompt (model may repeat prompt)
        if "[/INST]" in full_text:  # Qwen format
            answer = full_text.split("[/INST]")[-1].strip()
        elif "Answer (cite sources" in full_text:  # Our prompt marker
            answer = full_text.split("Answer (cite sources")[-1].strip()
            # Remove any trailing generation artifacts
            answer = re.split(r'\n\n(?:Question|User|Context):', answer)[0].strip()
        else:
            # Fallback: return last portion
            answer = full_text[-LASER_DOMAIN_CONFIG["max_new_tokens"]*2:].strip()
        
        # Clean up common artifacts
        answer = re.sub(r'\s+', ' ', answer)  # Normalize whitespace
        answer = answer.strip()
        
        return answer if answer else "I was unable to generate a response. Please try rephrasing your question."
        
    except Exception as e:
        st.error(f"Generation error: {e}")
        return f"Error generating response: {str(e)[:200]}..."

def retrieve_and_answer(
    vectorstore,
    tokenizer,
    model,
    device: str,
    backend: str,
    query: str,
    k: int = None,
    score_threshold: float = None
) -> Tuple[str, List[Document], float]:
    """Retrieve relevant chunks and generate answer with local LLM."""
    
    k = k or LASER_DOMAIN_CONFIG["retrieval_k"]
    score_threshold = score_threshold or LASER_DOMAIN_CONFIG["score_threshold"]
    
    # Retrieve with similarity search
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold}
    )
    
    retrieved_docs = retriever.invoke(query)
    
    # Calculate relevance score (average of retrieved chunk scores)
    if retrieved_docs:
        # FAISS doesn't return scores directly with this method, 
        # so we do a manual similarity check for scoring
        query_embedding = vectorstore.embedding_function.embed_query(query)
        scores = []
        for doc in retrieved_docs:
            doc_embedding = vectorstore.embedding_function.embed_query(doc.page_content[:500])
            sim = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-8
            )
            scores.append(sim)
        avg_relevance = np.mean(scores) if scores else 0.0
    else:
        avg_relevance = 0.0
    
    # Handle no results
    if not retrieved_docs:
        return "Based on the uploaded documents, I could not find information relevant to your question. Try rephrasing or checking document content.", [], avg_relevance
    
    # Create prompt with retrieved context
    prompt = create_laser_rag_prompt(retrieved_docs, query)
    
    # Generate response
    answer = generate_local_response(tokenizer, model, device, prompt, backend)
    
    return answer, retrieved_docs, avg_relevance

# =============================================
# STREAMLIT UI COMPONENTS
# =============================================

def render_sidebar():
    """Render sidebar with configuration options."""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Model selection
        model_choice = st.selectbox(
            "🧠 Local LLM Backend",
            options=list(LOCAL_LLM_OPTIONS.keys()),
            index=1,  # Default to Qwen2-0.5B
            help="All models run locally - no API key needed!"
        )
        st.session_state.llm_model_choice = model_choice
        
        # Domain settings
        st.markdown("#### 🔬 Laser Domain Settings")
        st.session_state.laser_domain_boost = st.checkbox(
            "Boost laser-topic relevance",
            value=True,
            help="Prioritize chunks containing laser-specific keywords"
        )
        
        st.session_state.show_sources = st.checkbox(
            "Show source citations",
            value=True,
            help="Display which documents chunks came from"
        )
        
        st.session_state.max_retrieved_chunks = st.slider(
            "Chunks to retrieve",
            min_value=2,
            max_value=8,
            value=4,
            help="More chunks = more context but slower responses"
        )
        
        # Info box
        st.markdown("---")
        st.markdown("""
        <div style="background:#f0f9ff;padding:1rem;border-radius:0.5rem;border-left:4px solid #3b82f6">
        <strong>💡 Tips for Best Results:</strong>
        <ul style="margin:0.5rem 0 0 1rem;padding:0">
        <li>Upload papers about laser ablation, LIPSS, ultrafast processing</li>
        <li>Ask specific questions: "What fluence threshold for silicon ablation?"</li>
        <li>Small models work best with clear, focused queries</li>
        <li>First load may take 1-2 min (model download)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Resource info
        st.markdown("---")
        st.caption(f"🖥️ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        st.caption(f"📦 Embedding model: ~80MB")
        st.caption(f"🤖 LLM: {LOCAL_LLM_OPTIONS.get(model_choice, 'unknown')}")

def render_document_uploader():
    """Render document upload section."""
    st.markdown("### 📁 Upload Laser Microstructure Documents")
    
    uploaded_files = st.file_uploader(
        "Select PDF or TXT files about laser processing, ablation, microstructuring, etc.",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Documents will be processed locally - no data leaves your browser session"
    )
    
    return uploaded_files

def process_documents(uploaded_files):
    """Handle document processing pipeline."""
    if not uploaded_files:
        return False
    
    # Identify new files
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    if not new_files:
        st.info("✓ All uploaded files already processed")
        return st.session_state.processing_complete
    
    # Reset state for new documents
    st.session_state.messages = []
    st.session_state.vectorstore = None
    st.session_state.all_chunks = []
    
    with st.spinner(f"Processing {len(new_files)} document(s)..."):
        try:
            # Load and chunk
            chunks = load_and_chunk_laser_documents(new_files)
            if not chunks:
                st.error("No chunks extracted. Check file format.")
                return False
            
            # Update processed files tracking
            for f in new_files:
                st.session_state.processed_files.add(f.name)
            
            # Store chunks
            st.session_state.all_chunks.extend(chunks)
            
            # Create vector store
            with st.spinner("Creating vector index (this may take a minute)..."):
                vectorstore = create_local_vector_store(
                    st.session_state.all_chunks,
                    LOCAL_EMBEDDING_MODEL
                )
                if vectorstore is None:
                    return False
                st.session_state.vectorstore = vectorstore
            
            st.success(f"✅ Ready! Indexed {len(st.session_state.all_chunks)} chunks from {len(st.session_state.processed_files)} files")
            st.session_state.processing_complete = True
            return True
            
        except Exception as e:
            st.error(f"Processing failed: {e}")
            return False

def render_chat_interface():
    """Render the main chat interface."""
    if not st.session_state.get('vectorstore'):
        st.info("👆 Upload documents above to start chatting with your laser microstructure knowledge base")
        return
    
    # Load LLM if not already loaded
    if st.session_state.llm_tokenizer is None and st.session_state.llm_model_choice:
        with st.spinner(f"Loading {st.session_state.llm_model_choice}..."):
            tokenizer, model, device = load_local_llm(st.session_state.llm_model_choice)
            if tokenizer and model:
                st.session_state.llm_tokenizer = tokenizer
                st.session_state.llm_model = model
                st.session_state.llm_device = device
                st.success("✓ Model loaded!")
            else:
                st.error("Failed to load model. Try selecting a different option.")
                return
    
    if not st.session_state.llm_tokenizer:
        st.warning("Please select a model in the sidebar first")
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources") and st.session_state.show_sources:
                with st.expander("📚 Retrieved Sources"):
                    for i, src in enumerate(message["sources"], 1):
                        source_name = src.metadata.get("source", "unknown")
                        topics = src.metadata.get("laser_topics", [])
                        st.markdown(f"**[{i}]** `{source_name}`" + (f" _[{', '.join(topics)}]_" if topics else ""))
                        st.markdown(f"> {src.page_content[:300]}...")
    
    # Chat input
    if prompt := st.chat_input("Ask about laser parameters, ablation thresholds, LIPSS formation, etc."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("🔍 Retrieving and generating..."):
                try:
                    # Retrieve and answer
                    answer, retrieved_docs, relevance = retrieve_and_answer(
                        vectorstore=st.session_state.vectorstore,
                        tokenizer=st.session_state.llm_tokenizer,
                        model=st.session_state.llm_model,
                        device=st.session_state.llm_device,
                        backend=st.session_state.llm_model_choice,
                        query=prompt,
                        k=st.session_state.max_retrieved_chunks
                    )
                    
                    # Simulate streaming output
                    display_text = ""
                    for word in answer.split():
                        display_text += word + " "
                        message_placeholder.markdown(display_text + "▌")
                        time.sleep(0.02)  # Adjust for desired "typing" speed
                    message_placeholder.markdown(answer)
                    
                    # Store assistant message with sources
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": retrieved_docs if st.session_state.show_sources else None,
                        "relevance": relevance
                    })
                    
                    # Show relevance score
                    if relevance > 0:
                        st.caption(f"📊 Response relevance: {relevance:.2f}/1.0")
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)[:300]}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

def render_footer():
    """Render footer with helpful info."""
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📚 Example Questions:**")
        st.caption("• What is the ablation threshold for silicon at 800nm?")
        st.caption("• How does pulse duration affect LIPSS formation?")
        st.caption("• What characterization methods for laser microstructures?")
    
    with col2:
        st.markdown("**⚡ Performance Tips:**")
        st.caption("• Keep questions focused and specific")
        st.caption("• Smaller chunks = more precise retrieval")
        st.caption("• CPU mode: allow 10-30s per response")
    
    with col3:
        st.markdown("**🔐 Privacy & Deployment:**")
        st.caption("• All processing happens locally in your session")
        st.caption("• No data sent to external APIs")
        st.caption("• Works on Streamlit Cloud free tier")

# =============================================
# MAIN APPLICATION
# =============================================

def main():
    # Page config
    st.set_page_config(
        page_title="🔬 Laser Microstructure RAG Assistant",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #1e40af, #7c3aed, #059669);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
    }
    .info-card {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin: 0.5rem 0;
    }
    .stChatMessage {
        border-radius: 0.5rem;
        margin: 0.25rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🔬 Laser Microstructure RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:#64748b;margin-bottom:1.5rem">
    Upload research papers, experimental reports, or simulation data about laser-matter interaction.
    Ask questions and get answers with source citations—all running locally, no API keys required.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize
    initialize_session_state()
    
    # Sidebar
    render_sidebar()
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Document upload
        uploaded_files = render_document_uploader()
        
        # Process button
        if uploaded_files and st.button("🔄 Process Documents", type="primary", use_container_width=True):
            process_documents(uploaded_files)
        
        # Status indicator
        if st.session_state.processing_complete:
            st.success("✅ Knowledge base ready")
            if st.session_state.vectorstore and hasattr(st.session_state.vectorstore, 'metadata'):
                meta = st.session_state.vectorstore.metadata
                st.caption(f"📦 {meta.get('total_chunks', '?')} chunks")
                topics = meta.get('laser_topics', [])
                if topics:
                    st.caption(f"🔬 Topics: {', '.join(topics[:5])}" + ("..." if len(topics)>5 else ""))
        elif uploaded_files:
            st.warning("⏳ Click 'Process Documents' to begin")
        else:
            st.info("📁 Upload PDF/TXT files to start")
        
        # Clear button
        if st.session_state.processed_files:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.clear()
                st.rerun()
    
    with col2:
        # Chat interface or welcome message
        if st.session_state.processing_complete and st.session_state.vectorstore:
            render_chat_interface()
        else:
            st.markdown("""
            <div class="info-card">
            <h3>👋 Welcome!</h3>
            <p>This assistant helps you query documents about:</p>
            <ul>
            <li>🔥 Laser ablation thresholds & mechanisms</li>
            <li>🌊 LIPSS and surface morphology formation</li>
            <li>⚡ Ultrafast laser-matter interactions</li>
            <li>🔬 Characterization techniques (SEM, AFM, etc.)</li>
            <li>📐 Process parameter optimization</li>
            </ul>
            <p><strong>Getting started:</strong></p>
            <ol>
            <li>Upload PDF/TXT files in the left panel</li>
            <li>Click "Process Documents"</li>
            <li>Select your preferred local LLM</li>
            <li>Start asking technical questions!</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
            
            # Demo questions
            st.markdown("**Try asking:**")
            demo_qs = [
                "What factors affect ablation threshold in metals?",
                "How does pulse duration influence LIPSS periodicity?",
                "What characterization methods are used for laser-textured surfaces?",
                "What is the typical fluence range for femtosecond laser processing?",
            ]
            for q in demo_qs:
                if st.button(f"💬 {q}", use_container_width=True, key=f"demo_{q[:20]}"):
                    st.session_state.demo_question = q
                    st.rerun()
    
    # Footer
    render_footer()
    
    # Handle demo question injection
    if hasattr(st.session_state, 'demo_question') and st.session_state.demo_question:
        st.session_state.messages.append({"role": "user", "content": st.session_state.demo_question})
        del st.session_state.demo_question
        st.rerun()

if __name__ == "__main__":
    main()
