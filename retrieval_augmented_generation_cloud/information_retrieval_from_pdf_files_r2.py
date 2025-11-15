import streamlit as st
import os
import tempfile
import time
from io import BytesIO

# LangChain / RAG imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores import FAISS

# --- LIGHTWEIGHT MODEL IMPORTS ---
try:
    # Try to use smaller models that might fit in Streamlit Cloud memory
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import Ollama
    CLOUD_FRIENDLY = True
except ImportError as e:
    st.error(f"Import error: {e}")
    CLOUD_FRIENDLY = False

from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Configuration for Cloud-Optimized Models ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # ~80MB
CHUNK_SIZE = 800  # Smaller chunks for memory efficiency
CHUNK_OVERLAP = 100

# --- Fallback to Hugging Face Hub if local models fail ---
def setup_models(fallback_to_api=True):
    """Setup models with fallback to API if local models fail"""
    models = {}
    
    try:
        # Try local embeddings first
        if CLOUD_FRIENDLY:
            models['embeddings'] = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )
        else:
            raise Exception("Local models not available")
            
    except Exception as e:
        st.warning(f"Local embeddings failed: {e}")
        if fallback_to_api:
            from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
            if st.session_state.get('hf_token'):
                models['embeddings'] = HuggingFaceInferenceAPIEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    api_key=st.session_state.hf_token
                )
            else:
                st.error("HF token required for API fallback")
        else:
            st.error("Could not initialize embeddings")
    
    return models

def setup_llm(fallback_to_api=True):
    """Setup LLM with multiple fallback options"""
    llm = None
    
    # Option 1: Try Ollama with tiny model (if available)
    try:
        if CLOUD_FRIENDLY:
            # Note: This requires Ollama to be running, which isn't possible on Streamlit Cloud
            # But keeping for local development context
            llm = Ollama(model="tinyllama")
    except:
        pass
    
    # Option 2: Fallback to Hugging Face Hub
    if not llm and fallback_to_api and st.session_state.get('hf_token'):
        try:
            from langchain_community.llms import HuggingFaceHub
            llm = HuggingFaceHub(
                repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                huggingfacehub_api_token=st.session_state.hf_token,
                model_kwargs={"temperature": 0.5, "max_new_tokens": 256}  # Reduced tokens
            )
        except Exception as e:
            st.error(f"HF Hub setup failed: {e}")
    
    # Option 3: Ultimate fallback - use a very simple local model
    if not llm and not fallback_to_api:
        st.warning("Using extremely limited local fallback - consider using API option")
        # You could implement a very simple rule-based system here
        # but it won't be very useful for RAG
    
    return llm

# --- Core RAG Functions (Optimized for Memory) ---

def load_and_chunk_pdf(uploaded_file):
    """Loads PDF and splits into smaller chunks for memory efficiency"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()
        
        # More aggressive text splitting for memory conservation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        chunks = text_splitter.split_documents(pages)
        return chunks
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return []
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

@st.cache_resource(show_spinner=False)
def create_vector_store(_chunks, embedding_model):
    """Create vector store with memory optimizations"""
    try:
        # Use FAISS with minimal configuration
        vectorstore = FAISS.from_documents(_chunks, embedding_model)
        return vectorstore
    except Exception as e:
        st.error(f"Vector store creation failed: {e}")
        return None

def create_rag_chain(vectorstore, llm):
    """Create RAG chain with optimized prompt"""
    if not llm:
        return None
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # Fewer chunks
    
    # More efficient prompt
    template = """Use the context below to answer the question. If unsure, say so.

Context: {context}

Question: {input}
Answer: """
    
    RAG_PROMPT = PromptTemplate(
        template=template, input_variables=["context", "input"]
    )

    combine_docs_chain = create_stuff_documents_chain(llm, RAG_PROMPT)
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return qa_chain

# --- Memory Monitoring ---
def get_memory_usage():
    """Simple memory usage monitoring"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    except:
        return None

# --- Streamlit UI ---

def main():
    st.set_page_config(
        page_title="PDF RAG Chatbot - Cloud Optimized", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("📄 PDF Q&A Chatbot (Cloud Optimized)")
    
    # Memory warning
    with st.sidebar:
        st.warning("""
        **Streamlit Cloud Limitations:**
        - ~1GB RAM available
        - CPU only, no GPU
        - Local models very limited
        """)
        
        st.header("Configuration")
        use_api = st.radio(
            "Model Source",
            ["Hugging Face API (Recommended)", "Local Models (Experimental)"],
            help="Local models may fail due to memory constraints"
        )
        
        if use_api == "Hugging Face API (Recommended)":
            st.session_state.hf_token = st.text_input(
                "Hugging Face API Token",
                type="password",
                help="Required for API access. Get from https://huggingface.co/settings/tokens"
            )
        else:
            if st.session_state.get('hf_token'):
                st.session_state.hf_token = None
            st.info("Local models will attempt to run but may fail due to memory limits")

    # Initialize session state
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # File upload
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type="pdf",
        accept_multiple_files=True,
        help="For best results, use small PDFs (<10 pages)"
    )

    # Process documents
    if uploaded_files:
        all_uploaded_names = {f.name for f in uploaded_files}
        new_or_missing_files = (all_uploaded_names - st.session_state.processed_files) 
        
        if new_or_missing_files:
            # Reset on new files
            st.session_state.messages = []
            st.session_state.vectorstore = None
            
            with st.spinner("Processing documents..."):
                try:
                    # Setup models based on user choice
                    fallback_to_api = use_api == "Hugging Face API (Recommended)"
                    models = setup_models(fallback_to_api)
                    
                    if not models.get('embeddings'):
                        st.error("Failed to initialize embeddings. Check configuration.")
                        return
                    
                    # Process all files
                    all_chunks = []
                    for uploaded_file in uploaded_files:
                        chunks = load_and_chunk_pdf(uploaded_file)
                        all_chunks.extend(chunks)
                        st.info(f"Processed {uploaded_file.name}: {len(chunks)} chunks")
                    
                    # Create vector store
                    if all_chunks:
                        st.session_state.vectorstore = create_vector_store(
                            all_chunks, 
                            models['embeddings']
                        )
                        st.session_state.processed_files = all_uploaded_names
                        st.success("Documents processed successfully!")
                    else:
                        st.error("No text could be extracted from the PDFs")
                        
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")
                    memory_usage = get_memory_usage()
                    if memory_usage and memory_usage > 800:
                        st.error("Memory limit exceeded. Try smaller documents or use API option.")

    # Chat interface
    if st.session_state.get('vectorstore'):
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Initialize LLM and RAG chain
        if 'qa_chain' not in st.session_state:
            with st.spinner("Initializing language model..."):
                fallback_to_api = use_api == "Hugging Face API (Recommended)"
                llm = setup_llm(fallback_to_api)
                
                if llm:
                    st.session_state.qa_chain = create_rag_chain(
                        st.session_state.vectorstore, 
                        llm
                    )
                else:
                    st.error("Failed to initialize language model")
                    st.session_state.qa_chain = None

        # Chat input
        if st.session_state.get('qa_chain') and (prompt := st.chat_input("Ask about your documents...")):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Thinking..."):
                        response = st.session_state.qa_chain.invoke({"input": prompt})
                        answer = response.get('answer', "I couldn't generate a response.")
                    
                    # Display response
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    elif uploaded_files:
        st.info("Documents uploaded. Processing may take a moment...")
    else:
        st.info("""
        **Welcome! To get started:**
        1. Choose model source in sidebar (API recommended)
        2. Upload PDF documents
        3. Ask questions about your documents
        
        💡 **Tip**: For best results on Streamlit Cloud:
        - Use Hugging Face API option
        - Upload small PDFs (<10 pages)
        - Ask specific, focused questions
        """)

if __name__ == "__main__":
    main()
