import streamlit as st
import os
import tempfile
import time
from io import BytesIO

# LangChain / RAG imports (2024–2025 compatible)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores import FAISS

# --- 1. CLOUD-READY MODEL IMPORTS (API-BASED) ---
# We use HuggingFaceHub and HuggingFaceInferenceAPIEmbeddings to offload computation
# from the Streamlit Cloud container, which requires your HF API Token.
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub

from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Configuration (Defaults for Hugging Face models) ---
# Note: These are IDs for models hosted on Hugging Face's Inference Endpoints.
HF_LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Core RAG Functions ---

def load_and_chunk_pdf(uploaded_file):
    """Loads a PDF from an uploaded file object and splits it into chunks."""
    # Use temporary file context to handle the Streamlit UploadedFile object
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file_path = tmp_file.name

    try:
        # 1. Load the PDF
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()

        # 2. Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        chunks = text_splitter.split_documents(pages)
        return chunks
    finally:
        # Clean up the temporary file
        os.remove(tmp_file_path)

@st.cache_resource
def create_vector_store(chunks, embedding_model, hf_token):
    """Generates embeddings and creates a FAISS vector store using Hugging Face Inference API."""
    try:
        # 3. Create Hugging Face Embeddings (remote inference)
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            model_name=embedding_model,
            api_key=hf_token # Requires HF API Token
        )

        # 4. Create Vector Store (using FAISS for simplicity)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {e}. Ensure your Hugging Face API token is valid and the model '{embedding_model}' is accessible.")
        return None

def create_rag_chain(vectorstore, llm_model, hf_token):
    """Creates the LangChain retrieval chain using Hugging Face Hub (remote LLM)."""
    llm = HuggingFaceHub(
        repo_id=llm_model,
        huggingfacehub_api_token=hf_token, # Requires HF API Token
        model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """
    <s>[INST] You are an expert document assistant. Use the following context to answer the user's question concisely and accurately.
    If the answer is not found in the context, state that clearly and do not try to make up an answer.

    Context: {context}

    Question: {input} [/INST]
    """
    RAG_PROMPT = PromptTemplate(
        template=template, input_variables=["context", "input"]
    )

    # Create the combine documents chain
    combine_docs_chain = create_stuff_documents_chain(llm, RAG_PROMPT)

    # Create the retrieval chain
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return qa_chain

# --- Streamlit UI and Logic ---

def main():
    st.set_page_config(page_title="Cloud-Ready PDF RAG Chatbot with TinyLlama", layout="wide")
    st.title("📄 Cloud-Ready PDF Q&A with Hugging Face and TinyLlama")
    st.markdown("Upload documents and chat with them using Hugging Face's cloud-hosted models. This configuration is optimized for Streamlit Cloud.")

    # --- Sidebar for Model Configuration ---
    with st.sidebar:
        st.header("Hugging Face Configuration")
        # Ensure HF token is the input mechanism
        st.session_state.hf_token = st.text_input(
            "Hugging Face API Token",
            type="password",
            help="Get your free API token from https://huggingface.co/settings/tokens"
        )
        st.session_state.llm_model = st.text_input(
            "LLM Model (Repo ID)",
            value=HF_LLM_MODEL,
            help="Hugging Face repo ID for the generation model (e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0)."
        )
        st.session_state.embed_model = st.text_input(
            "Embedding Model (Repo ID)",
            value=HF_EMBEDDING_MODEL,
            help="Hugging Face repo ID for the embedding model (e.g., sentence-transformers/all-MiniLM-L6-v2)."
        )
        st.markdown("---")
        st.markdown("### ☁️ Cloud Optimization")
        st.markdown("This version uses remote Hugging Face APIs, offloading model computation to external servers to conserve Streamlit Cloud resources.")


    # --- Document Processing ---
    uploaded_files = st.file_uploader(
        "Upload your PDF documents",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files to use as the knowledge source."
    )

    # Initialize session state for tracking processed files
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if uploaded_files and st.session_state.hf_token:
        # Identify files that need processing or re-processing
        all_uploaded_names = {f.name for f in uploaded_files}
        new_or_missing_files = (all_uploaded_names - st.session_state.processed_files) or (len(all_uploaded_names) != len(st.session_state.processed_files))
        
        if new_or_missing_files:
            # Reset chat history when new files are added or files are removed
            st.session_state.messages = []
            st.session_state.vectorstore = None
            st.session_state.qa_chain = None

            with st.spinner(f"Processing {len(uploaded_files)} PDF(s)..."):
                try:
                    all_chunks = []
                    # 1. Re-process ALL uploaded files to ensure combined knowledge base
                    for uploaded_file in uploaded_files:
                        chunks = load_and_chunk_pdf(uploaded_file)
                        all_chunks.extend(chunks)
                        if uploaded_file.name not in st.session_state.processed_files:
                             st.info(f"Loaded {len(chunks)} text chunks from {uploaded_file.name}.")
                            
                    # 2. Create Vector Store (Cached for performance)
                    st.session_state.vectorstore = create_vector_store(
                        all_chunks,
                        st.session_state.embed_model,
                        st.session_state.hf_token
                    )

                    st.success("Document ingestion complete! You can now ask questions.")
                    # Update processed files list to only contain successfully processed files
                    st.session_state.processed_files = all_uploaded_names
                except Exception as e:
                    st.error(f"Failed to process PDFs: {e}")
                    st.session_state.vectorstore = None # Ensure chat is blocked
                    return
        
    elif uploaded_files and not st.session_state.hf_token:
        st.warning("Please enter your Hugging Face API token in the sidebar to process documents.")
    else:
        st.info("Upload one or more PDFs and enter your HF API token to start chatting with your knowledge base.")

    # --- Chat Interface ---

    if st.session_state.get('vectorstore'):
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Create RAG Chain if it doesn't exist
        if 'qa_chain' not in st.session_state or st.session_state.qa_chain is None:
            st.session_state.qa_chain = create_rag_chain(
                st.session_state.vectorstore,
                st.session_state.llm_model,
                st.session_state.hf_token
            )
        
        # Guard against chain creation failure
        if st.session_state.qa_chain is None:
            return

        # Handle user input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Display user message in chat message container
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                # Note: The LCEL chain is not inherently streaming, so we use a placeholder and then update.
                try:
                    with st.spinner("Thinking..."):
                        # Invoke the RAG chain
                        response = st.session_state.qa_chain.invoke(
                            {"input": prompt}
                        )
                        full_response = response.get('answer', "Sorry, I couldn't find an answer in the provided document context.")

                        # Simulate streaming display
                        display_text = ""
                        for chunk in full_response.split():
                            display_text += chunk + " "
                            time.sleep(0.01) # Reduced sleep time for faster display
                            message_placeholder.markdown(display_text + "▌")
                        message_placeholder.markdown(full_response)

                except Exception as e:
                    error_message = f"Hugging Face Error: {e}. Please ensure your API token is valid and the model '{st.session_state.llm_model}' is available."
                    st.error(error_message)
                    full_response = error_message

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    elif uploaded_files:
        st.info("Please wait while the documents are being processed or enter your HF token.")
    else:
        st.info("Upload one or more PDFs and enter your HF API token to start chatting with your knowledge base.")

if __name__ == "__main__":
    main()
