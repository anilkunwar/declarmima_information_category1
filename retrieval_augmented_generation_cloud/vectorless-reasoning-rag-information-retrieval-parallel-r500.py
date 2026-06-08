import streamlit as st
import fitz  # PyMuPDF
import json
import re
import os
from huggingface_hub import InferenceClient

# ============================================================================
# CONFIGURATION & MODEL OPTIONS
# ============================================================================
HF_MODEL_OPTIONS = {
    "[HF] Qwen2.5-7B-Instruct (Recommended)": "Qwen/Qwen2.5-7B-Instruct",
    "[HF] Mistral-7B-Instruct-v0.3 (Great for JSON)": "mistralai/Mistral-7B-Instruct-v0.3",
    "[HF] Llama-3.1-8B-Instruct (Meta Standard)": "meta-llama/Llama-3.1-8B-Instruct",
    "[HF] Meta-Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
@st.cache_resource
def get_hf_client():
    """Initialize the Hugging Face client using Streamlit secrets."""
    if "HF_TOKEN" not in st.secrets:
        st.error("⚠️ Please add your HF_TOKEN to Streamlit secrets!")
        st.stop()
    return InferenceClient(api_key=st.secrets["HF_TOKEN"])

def call_llm(model_id, prompt, system="You are a helpful assistant."):
    """Unified wrapper for Hugging Face Inference API."""
    client = get_hf_client()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat_completion(
            messages=messages,
            model=model_id,
            max_tokens=1024,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Hugging Face API Error: {e}")
        return ""

def extract_json(text):
    """Robust JSON extractor that handles markdown wrappers and malformed outputs."""
    if not text: return None
    try: return json.loads(text)
    except: pass
    
    # Try extracting from markdown code block
    match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL | re.IGNORECASE)
    if match:
        try: return json.loads(match.group(1))
        except: pass
        
    # Fallback: find the first { or [ and the last } or ]
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try: return json.loads(text[start:end+1])
            except: pass
    return None

# ============================================================================
# PAGEINDEX CORE LOGIC (VECTORLESS)
# ============================================================================
def index_all_documents(uploaded_files, model_id):
    """Phase 1: Builds hierarchical tree indices for multiple PDFs."""
    documents = {}
    total_files = len(uploaded_files)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, uploaded_file in enumerate(uploaded_files):
        doc_name = os.path.splitext(uploaded_file.name)[0]
        # Handle duplicate filenames
        if doc_name in documents:
            doc_name = f"{doc_name}_{idx+1}"
            
        status_text.info(f"Processing ({idx+1}/{total_files}): {uploaded_file.name}...")
        
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_text = [{"page": i + 1, "text": page.get_text()} for i, page in enumerate(doc)]
        
        # 1. Extract Sections (Chunked to fit context windows)
        chunk_size = 5
        sections = []
        for i in range(0, len(pages_text), chunk_size):
            chunk = pages_text[i:i+chunk_size]
            text_with_tags = "\n".join([f"<page_{p['page']}>\n{p['text']}\n</page_{p['page']}>" for p in chunk])
            
            prompt = f"""Analyze the document text and extract main section headings and their starting page numbers.
            Return a JSON list of objects: {{"title": "Section Title", "start_page": page_number}}.
            Ignore headers/footers. Only include actual content sections.
            
            Text:
            {text_with_tags}"""
            
            system = "You are an expert document analyzer. Return ONLY a valid JSON list."
            response = call_llm(model_id, prompt, system=system)
            parsed = extract_json(response)
            
            if parsed and isinstance(parsed, list):
                for item in parsed:
                    if 'title' in item and 'start_page' in item:
                        sections.append(item)
                        
        # 2. Determine End Pages and Build Flat Tree
        sections.sort(key=lambda x: x['start_page'])
        for i in range(len(sections)):
            sections[i]['end_page'] = sections[i+1]['start_page'] - 1 if i + 1 < len(sections) else len(pages_text)
            sections[i]['id'] = f"sec_{i+1:03d}"
            
        # 3. Generate Summaries for each node
        for i, sec in enumerate(sections):
            start_idx = sec['start_page'] - 1
            end_idx = sec['end_page']
            section_text = "\n".join([p['text'] for p in pages_text[start_idx:end_idx]])
            
            # Limit text length for summary to avoid context overflow
            if len(section_text) > 4000:
                section_text = section_text[:4000] + "..."
                
            prompt = f"Summarize the following text in 2-3 concise sentences:\n\n{section_text}"
            system = "You are a helpful summarizer."
            summary = call_llm(model_id, prompt, system=system)
            sec['summary'] = summary.strip()
            
        documents[doc_name] = {'doc': doc, 'sections': sections, 'filename': uploaded_file.name}
        progress_bar.progress((idx + 1) / total_files)
        
    progress_bar.empty()
    status_text.empty()
    return documents

def agentic_retrieve_and_answer(query, selected_docs, model_id):
    """Phase 2 & 3: Agentic Multi-Document Tree Search and Answer Generation."""
    
    # 1. Build the "Forest" Description
    forest_desc = []
    for doc_name, data in selected_docs.items():
        forest_desc.append(f"--- Document: {doc_name} ---")
        for s in data['sections']:
            # Namespace the ID with a safe delimiter (:::)
            ns_id = f"{doc_name}:::{s['id']}"
            forest_desc.append(f"ID: {ns_id} | Title: {s['title']} | Pages: {s['start_page']}-{s['end_page']} | Summary: {s['summary']}")
    
    tree_text = "\n".join(forest_desc)
    
    search_prompt = f"""You are a research assistant. Given a query and a forest of document structures, identify which sections across the documents are most likely to contain the answer.
    Query: {query}
    
    Document Structures:
    {tree_text}
    
    Return a JSON object: {{"thinking": "your step-by-step reasoning", "relevant_ids": ["doc_name:::sec_id1", "doc_name:::sec_id2"]}}.
    If no sections are relevant, return an empty list for relevant_ids."""
    
    system = "You are a precise navigation agent. Return ONLY valid JSON."
    search_response = call_llm(model_id, search_prompt, system=system)
    search_result = extract_json(search_response)
    
    thinking = "No reasoning provided."
    relevant_ids = []
    if search_result:
        thinking = search_result.get('thinking', 'No reasoning provided.')
        relevant_ids = search_result.get('relevant_ids', [])
        
    # 2. Fetch Content (Lossless Retrieval)
    context = ""
    retrieved_info = []
    
    for ns_id in relevant_ids:
        if ":::" not in ns_id: continue
        doc_name, sec_id = ns_id.split(":::", 1)
        if doc_name in selected_docs:
            data = selected_docs[doc_name]
            doc = data['doc']
            sections = data['sections']
            
            sec = next((s for s in sections if s['id'] == sec_id), None)
            if sec:
                retrieved_info.append({"doc_name": doc_name, "section": sec})
                text_parts = []
                for p_num in range(sec['start_page'], sec['end_page'] + 1):
                    if 0 < p_num <= len(doc):
                        text_parts.append(doc[p_num - 1].get_text())
                context += f"\n\n--- Document: {doc_name} | Section: {sec['title']} (Pages {sec['start_page']}-{sec['end_page']}) ---\n" + "\n".join(text_parts) + "\n"
                
    if not context:
        return "I could not find any relevant sections in the selected documents to answer your query.", thinking, retrieved_info
        
    # Safeguard for context limits
    if len(context) > 15000:
        context = context[:15000] + "\n[Context truncated due to length]"
        
    # 3. Generate Answer
    answer_prompt = f"""Answer the user's query based ONLY on the provided document context.
    If the answer is not in the context, say "I don't know based on the provided documents."
    Cite the document name and section title when possible.
    
    Query: {query}
    Context:
    {context}"""
    
    system = "You are a helpful document QA assistant. Be concise and accurate."
    answer = call_llm(model_id, answer_prompt, system=system)
    
    return answer, thinking, retrieved_info

# ============================================================================
# STREAMLIT UI
# ============================================================================
st.set_page_config(page_title="Multi-Doc Vectorless RAG", layout="wide")
st.title("🌲 Multi-Document Vectorless RAG (Hugging Face)")
st.markdown("Upload multiple PDFs, build hierarchical tree indices, and query them using LLMs via Hugging Face. **No Vector DBs. No Chunking.**")

# Initialize session state variables safely
if 'selected_docs_for_query' not in st.session_state:
    st.session_state['selected_docs_for_query'] = []

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model_name = st.selectbox("Select Hugging Face Model", list(HF_MODEL_OPTIONS.keys()))
    model_id = HF_MODEL_OPTIONS[selected_model_name]
    
    st.markdown("---")
    st.header("📄 Documents")
    uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files and st.button("🚀 Build Document Trees"):
        with st.spinner("Indexing documents... This may take a few minutes."):
            st.session_state['documents'] = index_all_documents(uploaded_files, model_id)
            st.session_state['messages'] = []
            st.success("All documents indexed successfully!")
            
    # Display the Tree Structure and Document Selector
    if 'documents' in st.session_state:
        st.markdown("---")
        st.header("🎯 Query Scope")
        
        doc_names = list(st.session_state['documents'].keys())
        st.session_state['selected_docs_for_query'] = st.multiselect(
            "Select documents to search:",
            doc_names,
            default=doc_names
        )
        
        with st.expander("🌳 View Document Trees", expanded=False):
            for doc_name in doc_names:
                st.subheader(f"📄 {doc_name}")
                data = st.session_state['documents'][doc_name]
                for s in data['sections']:
                    st.markdown(f"**{s['id']}: {s['title']}** *(pp. {s['start_page']}-{s['end_page']})*")
                    st.caption(s['summary'])
                    st.divider()

# Main Chat Area
if 'documents' not in st.session_state:
    st.warning("👈 Please upload PDF(s) and build the document trees to start querying.")
else:
    st.subheader("💬 Chat with your Documents")
    
    # Display chat history
    for msg in st.session_state['messages']:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
            if msg['role'] == 'assistant' and 'thinking' in msg:
                with st.expander("🧠 Agent Reasoning & Retrieved Sections"):
                    st.markdown(f"**Thinking Process:**\n{msg['thinking']}")
                    if msg.get('retrieved_info'):
                        st.markdown("**Retrieved Sections:**")
                        for info in msg['retrieved_info']:
                            sec = info['section']
                            st.info(f"📄 **{info['doc_name']}** | {sec['id']}: {sec['title']} (Pages {sec['start_page']}-{sec['end_page']})")
                            
    # Chat input
    if prompt := st.chat_input("Ask a question about the documents..."):
        if not st.session_state['selected_docs_for_query']:
            st.error("Please select at least one document in the sidebar to search.")
        else:
            st.session_state['messages'].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Searching document trees and generating answer..."):
                    # Filter the documents dict based on user selection
                    docs_to_search = {k: st.session_state['documents'][k] for k in st.session_state['selected_docs_for_query']}
                    
                    answer, thinking, retrieved_info = agentic_retrieve_and_answer(
                        prompt, docs_to_search, model_id
                    )
                    st.markdown(answer)
                    with st.expander("🧠 Agent Reasoning & Retrieved Sections"):
                        st.markdown(f"**Thinking Process:**\n{thinking}")
                        if retrieved_info:
                            st.markdown("**Retrieved Sections:**")
                            for info in retrieved_info:
                                sec = info['section']
                                st.info(f"📄 **{info['doc_name']}** | {sec['id']}: {sec['title']} (Pages {sec['start_page']}-{sec['end_page']})")
                                
            st.session_state['messages'].append({
                "role": "assistant", 
                "content": answer, 
                "thinking": thinking, 
                "retrieved_info": retrieved_info
            })
