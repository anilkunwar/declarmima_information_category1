import streamlit as st
import os
import tempfile
import time
from io import BytesIO

# Imports for RAG and LangChain (need to install: langchain, langchain-community, pypdf, faiss-cpu, langchain-huggingface)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain_community.chains import RetrievalQA


# --- Configuration ---
# Defaults for Hugging Face models
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
            api_key=hf_token
        )

        # 4. Create Vector Store (using FAISS for simplicity)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {e}. Ensure your Hugging Face API token is valid and the model '{embedding_model}' is accessible.")
        return None

def create_rag_chain(vectorstore, llm_model, hf_token):
    """Creates the LangChain RetrievalQA chain using Hugging Face Hub."""
    llm = HuggingFaceHub(
        repo_id=llm_model,
        huggingfacehub_api_token=hf_token,
        model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Define the custom prompt for RAG (adjusted for TinyLlama chat format if needed)
    template = """
    <s>[INST] You are an expert document assistant. Use the following context to answer the user's question concisely and accurately.
    If the answer is not found in the context, state that clearly and do not try to make up an answer.

    Context: {context}

    Question: {question} [/INST]
    """
    RAG_PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )

    # Create the RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": RAG_PROMPT}
    )
    return qa_chain

# --- Streamlit UI and Logic ---

def main():
    st.set_page_config(page_title="Cloud-Ready PDF RAG Chatbot with TinyLlama", layout="wide")
    st.title("📄 Cloud-Ready PDF Q&A with Hugging Face and TinyLlama")
    st.markdown("Upload one or more PDF documents and chat with them using Hugging Face's cloud-hosted models (e.g., TinyLlama). This app can be deployed to Streamlit Cloud.")

    # --- Sidebar for Model Configuration ---
    with st.sidebar:
        st.header("Hugging Face Configuration")
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
        st.markdown("### ⚠️ IMPORTANT")
        st.markdown("Ensure you have a valid Hugging Face API token. Models are hosted remotely—no local setup needed.")
        st.markdown("Note: TinyLlama is a small model; responses may be less accurate than larger ones. API usage may incur costs for heavy use.")


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

    if uploaded_files and st.session_state.hf_token:
        # Identify new files that haven't been processed yet
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]

        if new_files:
            # Reset chat history when new files are added
            st.session_state.messages = []
            st.session_state.vectorstore = None

            with st.spinner(f"Processing {len(new_files)} new PDF(s): {', '.join([f.name for f in new_files])}..."):
                try:
                    all_chunks = []
                    for uploaded_file in new_files:
                        # 1. Load and Chunk each new file
                        chunks = load_and_chunk_pdf(uploaded_file)
                        all_chunks.extend(chunks)
                        st.info(f"Loaded {len(chunks)} text chunks from {uploaded_file.name}.")
                        # Mark as processed
                        st.session_state.processed_files.add(uploaded_file.name)

                    # If there are existing chunks (from previous uploads), merge them
                    if "existing_chunks" not in st.session_state:
                        st.session_state.existing_chunks = []
                    st.session_state.existing_chunks.extend(all_chunks)

                    # 2. Create or Update Vector Store (Cached for performance)
                    st.session_state.vectorstore = create_vector_store(
                        st.session_state.existing_chunks,
                        st.session_state.embed_model,
                        st.session_state.hf_token
                    )

                    st.success("Document ingestion complete! You can now ask questions.")
                except Exception as e:
                    st.error(f"Failed to process PDFs: {e}")
                    # Remove failed files from processed set to allow retry
                    for f in new_files:
                        st.session_state.processed_files.discard(f.name)
                    return
        else:
            st.info("All uploaded files have already been processed.")
    elif uploaded_files and not st.session_state.hf_token:
        st.warning("Please enter your Hugging Face API token in the sidebar to process documents.")

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
        if 'qa_chain' not in st.session_state:
            st.session_state.qa_chain = create_rag_chain(
                st.session_state.vectorstore,
                st.session_state.llm_model,
                st.session_state.hf_token
            )

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

                # Note: RetrievalQA is not inherently streaming, so we use a placeholder and then update.
                with st.spinner("Thinking..."):
                    try:
                        # Invoke the RAG chain
                        response = st.session_state.qa_chain.invoke(
                            {"query": prompt}
                        )
                        full_response = response.get('result', "Sorry, I couldn't find an answer in the provided document context.")

                        # Simulate streaming display
                        for chunk in full_response.split():
                            full_response += chunk + " "
                            time.sleep(0.05)
                            message_placeholder.markdown(full_response + "▌")
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
