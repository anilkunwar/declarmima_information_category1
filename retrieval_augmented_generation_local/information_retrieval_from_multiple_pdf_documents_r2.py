import streamlit as st
import os
import tempfile
import time
from io import BytesIO
from typing import List, Dict, Any
import numpy as np

# Imports for RAG and LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever

# --- Configuration ---
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Enhanced RAG Functions with Information Fusion ---

def load_and_chunk_pdf(uploaded_file):
    """Loads a PDF from an uploaded file object and splits it into chunks with enhanced metadata."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()

        # Enhanced metadata for better fusion tracking
        for i, page in enumerate(pages):
            page.metadata["source_document"] = uploaded_file.name
            page.metadata["document_id"] = f"{uploaded_file.name}_{hash(uploaded_file.name)}"
            page.metadata["global_page_id"] = f"{uploaded_file.name}_page_{i+1}"

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        chunks = text_splitter.split_documents(pages)
        return chunks
    finally:
        os.remove(tmp_file_path)

def create_cross_document_retriever(vectorstore, llm_model, base_url, search_kwargs={"k": 6}):
    """Creates an enhanced retriever that performs multi-query expansion for better fusion."""
    
    # Standard retriever
    base_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    # Multi-query retriever for diverse perspectives
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=Ollama(model=llm_model, base_url=base_url)
    )
    
    return multi_query_retriever

def create_fusion_aware_rag_chain(vectorstore, llm_model, base_url):
    """Creates a RAG chain with enhanced information fusion capabilities."""
    llm = Ollama(model=llm_model, base_url=base_url)
    
    # Enhanced retriever with multi-query capability
    retriever = create_cross_document_retriever(vectorstore, llm_model, base_url)

    # Advanced prompt for information fusion
    fusion_template = """
    You are an expert research assistant tasked with synthesizing information from multiple documents.
    
    DOCUMENT CONTEXT:
    {context}
    
    USER QUESTION: {question}
    
    ANALYSIS INSTRUCTIONS:
    1. Identify key information from each relevant document source
    2. Compare and contrast different perspectives or information across documents
    3. Synthesize a comprehensive answer that integrates the most relevant information
    4. Note any contradictions or complementary information between sources
    5. If information conflicts, note the sources and provide the most likely accurate information
    6. Cite which documents contributed to which parts of your answer
    
    Please provide a well-structured response that demonstrates information fusion:
    
    SYNTHESIZED ANSWER:
    """
    
    FUSION_PROMPT = PromptTemplate(
        template=fusion_template, 
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": FUSION_PROMPT,
            "document_variable_name": "context"
        }
    )
    return qa_chain

def analyze_document_relationships(chunks):
    """Analyzes relationships between documents based on content overlap."""
    if not chunks:
        return {}
    
    # Simple analysis: document co-occurrence and topic distribution
    doc_sources = {}
    for chunk in chunks:
        source = chunk.metadata.get("source_document", "unknown")
        if source not in doc_sources:
            doc_sources[source] = []
        doc_sources[source].append(chunk.page_content[:200])  # First 200 chars as sample
    
    return {
        "total_documents": len(doc_sources),
        "document_names": list(doc_sources.keys()),
        "chunks_per_document": {k: len(v) for k, v in doc_sources.items()}
    }

def format_response_with_sources(response, question):
    """Formats the response to highlight information fusion and source attribution."""
    answer = response.get('result', 'No answer generated.')
    source_docs = response.get('source_documents', [])
    
    # Analyze source distribution
    source_analysis = {}
    for doc in source_docs:
        source = doc.metadata.get('source_document', 'Unknown')
        if source not in source_analysis:
            source_analysis[source] = 0
        source_analysis[source] += 1
    
    # Create formatted response
    formatted_response = f"**Answer:** {answer}\n\n"
    
    if source_analysis:
        formatted_response += "**Source Analysis:**\n"
        for source, count in source_analysis.items():
            formatted_response += f"- 📄 {source}: {count} relevant chunks\n"
    
    # Add fusion insights
    if len(source_analysis) > 1:
        formatted_response += f"\n**Information Fusion:** 🔄 Integrated insights from {len(source_analysis)} different documents\n"
    
    return formatted_response

@st.cache_resource
def create_vector_store(chunks, embedding_model, base_url):
    """Generates embeddings and creates a FAISS vector store."""
    try:
        embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {e}. Ensure Ollama is running and '{embedding_model}' is pulled.")
        return None

# --- Enhanced Streamlit UI ---

def main():
    st.set_page_config(page_title="Multi-Document RAG with Information Fusion", layout="wide")
    st.title("🔗 Multi-Document RAG with Information Fusion")
    st.markdown("Upload multiple PDF documents and get AI-powered answers that synthesize information across all documents.")
    
    # --- Sidebar for Enhanced Configuration ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.session_state.ollama_base_url = st.text_input(
            "Ollama Base URL",
            value=OLLAMA_BASE_URL,
            help="The URL where your local Ollama server is running."
        )
        st.session_state.llm_model = st.text_input(
            "LLM Model for Generation",
            value=OLLAMA_MODEL,
            help="The model to use for generating answers."
        )
        st.session_state.embed_model = st.text_input(
            "Embedding Model",
            value=OLLAMA_EMBEDDING_MODEL,
            help="The model to use for creating vector embeddings."
        )
        
        st.markdown("---")
        st.header("📊 Fusion Settings")
        st.session_state.retrieval_k = st.slider(
            "Number of chunks to retrieve",
            min_value=3,
            max_value=15,
            value=6,
            help="More chunks can provide better fusion but may be slower."
        )
        
        st.markdown("---")
        st.markdown("### ⚠️ Requirements")
        st.markdown(f"Ensure Ollama is running and you have pulled:\n- `{st.session_state.llm_model}`\n- `{st.session_state.embed_model}`")

    # --- Document Processing with Enhanced Analytics ---
    uploaded_files = st.file_uploader(
        "Upload your PDF documents for multi-document analysis",
        type="pdf",
        accept_multiple_files=True,
        help="Upload multiple related PDFs to enable cross-document information fusion."
    )

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "document_relationships" not in st.session_state:
        st.session_state.document_relationships = {}

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]

        if new_files:
            st.session_state.messages = []
            st.session_state.vectorstore = None

            with st.spinner(f"Processing {len(new_files)} new PDF(s) and analyzing document relationships..."):
                try:
                    all_chunks = []
                    for uploaded_file in new_files:
                        chunks = load_and_chunk_pdf(uploaded_file)
                        all_chunks.extend(chunks)
                        st.info(f"📑 {uploaded_file.name}: Loaded {len(chunks)} text chunks")
                        st.session_state.processed_files.add(uploaded_file.name)

                    if "existing_chunks" not in st.session_state:
                        st.session_state.existing_chunks = []
                    st.session_state.existing_chunks.extend(all_chunks)

                    # Analyze document relationships
                    st.session_state.document_relationships = analyze_document_relationships(
                        st.session_state.existing_chunks
                    )

                    # Create vector store
                    st.session_state.vectorstore = create_vector_store(
                        st.session_state.existing_chunks,
                        st.session_state.embed_model,
                        st.session_state.ollama_base_url
                    )

                    st.success("✅ Document ingestion complete! Information fusion is now enabled.")
                    
                    # Display document analytics
                    with st.expander("📈 Document Relationship Analysis"):
                        st.write(st.session_state.document_relationships)
                        
                except Exception as e:
                    st.error(f"Failed to process PDFs: {e}")
                    for f in new_files:
                        st.session_state.processed_files.discard(f.name)
                    return
        else:
            st.info("All uploaded files have been processed. You can ask questions that span across all documents.")

    # --- Enhanced Chat Interface ---
    if st.session_state.get('vectorstore'):
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Create enhanced RAG Chain
        if 'qa_chain' not in st.session_state:
            st.session_state.qa_chain = create_fusion_aware_rag_chain(
                st.session_state.vectorstore,
                st.session_state.llm_model,
                st.session_state.ollama_base_url
            )

        # Handle user input
        if prompt := st.chat_input("Ask a question that requires information from multiple documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                with st.spinner("🔍 Retrieving and synthesizing information across documents..."):
                    try:
                        response = st.session_state.qa_chain.invoke({"query": prompt})
                        
                        # Format response with fusion insights
                        formatted_response = format_response_with_sources(response, prompt)
                        
                        # Simulate streaming
                        for chunk in formatted_response.split():
                            full_response += chunk + " "
                            time.sleep(0.03)
                            message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)

                    except Exception as e:
                        error_message = f"Error: {e}. Please ensure your Ollama server is running and models are available."
                        st.error(error_message)
                        full_response = error_message

            st.session_state.messages.append({"role": "assistant", "content": full_response})

    elif uploaded_files:
        st.info("⏳ Processing documents... This may take a moment for multi-document analysis.")
    else:
        st.info("👆 Upload multiple PDF documents to enable cross-document information fusion and synthesis.")

if __name__ == "__main__":
    main()
