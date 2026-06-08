import streamlit as st
import fitz  # PyMuPDF
import ollama
import json
import re
import time

# ============================================================================
# CONFIGURATION & MODEL OPTIONS
# ============================================================================
LOCAL_LLM_OPTIONS = {
    "[Ollama] qwen2.5:0.5b (Fastest, CPU Ok)": "qwen2.5:0.5b",
    "[Ollama] qwen2.5:1.5b (Balanced)": "qwen2.5:1.5b",
    "[Ollama] qwen2.5:7b (Recommended for RAG)": "qwen2.5:7b",
    "[Ollama] qwen2.5:14b (Max Reasoning)": "qwen2.5:14b",
    "[Ollama] llama3.1:8b (Meta Standard)": "llama3.1:8b",
    "[Ollama] mistral:7b (High JSON Reliability)": "mistral:7b",
    "[Ollama] gemma2:9b (Scientific Nuance)": "gemma2:9b",
    "[Ollama] falcon3:10b (Instruction Following)": "falcon3:10b",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def call_ollama(model, prompt, system="You are a helpful assistant.", json_mode=False):
    """Unified wrapper for Ollama chat completions."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]
    try:
        # format='json' enforces JSON output for models that support grammar
        response = ollama.chat(model=model, messages=messages, format='json' if json_mode else '')
        return response['message']['content']
    except Exception as e:
        st.error(f"Ollama Error: {e}")
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
def build_document_tree(pdf_bytes, model):
    """Phase 1: Builds a hierarchical tree index with summaries."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text = [{"page": i + 1, "text": page.get_text()} for i, page in enumerate(doc)]
    
    st.info(f"Extracted text from {len(pages_text)} pages. Building tree structure...")
    progress_bar = st.progress(0)
    
    # 1. Extract Sections (Chunked to fit local model context windows)
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
        response = call_ollama(model, prompt, system=system, json_mode=True)
        parsed = extract_json(response)
        
        if parsed and isinstance(parsed, list):
            for item in parsed:
                if 'title' in item and 'start_page' in item:
                    sections.append(item)
        progress_bar.progress((i + chunk_size) / len(pages_text))
        
    # 2. Determine End Pages and Build Flat Tree
    sections.sort(key=lambda x: x['start_page'])
    for i in range(len(sections)):
        sections[i]['end_page'] = sections[i+1]['start_page'] - 1 if i + 1 < len(sections) else len(pages_text)
        sections[i]['id'] = f"sec_{i+1:03d}"
        
    # 3. Generate Summaries for each node
    st.info("Generating summaries for each section...")
    for i, sec in enumerate(sections):
        start_idx = sec['start_page'] - 1
        end_idx = sec['end_page']
        section_text = "\n".join([p['text'] for p in pages_text[start_idx:end_idx]])
        
        # Limit text length for summary to avoid context overflow on smaller models
        if len(section_text) > 4000:
            section_text = section_text[:4000] + "..."
            
        prompt = f"Summarize the following text in 2-3 concise sentences:\n\n{section_text}"
        system = "You are a helpful summarizer."
        summary = call_ollama(model, prompt, system=system)
        sec['summary'] = summary.strip()
        progress_bar.progress((i + 1) / len(sections))
        
    progress_bar.empty()
    return doc, sections

def agentic_retrieve_and_answer(query, doc, sections, model):
    """Phase 2 & 3: Agentic Tree Search and Answer Generation."""
    
    # 1. Tree Search (Navigation Agent)
    tree_desc = "\n".join([
        f"ID: {s['id']} | Title: {s['title']} | Pages: {s['start_page']}-{s['end_page']} | Summary: {s['summary']}" 
        for s in sections
    ])
    
    search_prompt = f"""You are a research assistant. Given a query and a document structure, identify which sections are most likely to contain the answer.
    Query: {query}
    
    Document Structure:
    {tree_desc}
    
    Return a JSON object: {{"thinking": "your step-by-step reasoning", "relevant_ids": ["id1", "id2"]}}"""
    
    system = "You are a precise navigation agent. Return ONLY valid JSON."
    search_response = call_ollama(model, search_prompt, system=system, json_mode=True)
    search_result = extract_json(search_response)
    
    thinking = "No reasoning provided."
    relevant_ids = []
    if search_result:
        thinking = search_result.get('thinking', 'No reasoning provided.')
        relevant_ids = search_result.get('relevant_ids', [])
        
    # 2. Fetch Content (Lossless Retrieval)
    context = ""
    retrieved_sections = []
    for s in sections:
        if s['id'] in relevant_ids:
            retrieved_sections.append(s)
            text_parts = []
            for p_num in range(s['start_page'], s['end_page'] + 1):
                if 0 < p_num <= len(doc):
                    text_parts.append(doc[p_num - 1].get_text())
            context += f"\n\n--- Section: {s['title']} (Pages {s['start_page']}-{s['end_page']}) ---\n" + "\n".join(text_parts) + "\n"
            
    if not context:
        return "I could not find any relevant sections in the document to answer your query.", thinking, retrieved_sections
        
    # Safeguard for local model context limits
    if len(context) > 15000:
        context = context[:15000] + "\n[Context truncated due to length]"
        
    # 3. Generate Answer
    answer_prompt = f"""Answer the user's query based ONLY on the provided document context.
    If the answer is not in the context, say "I don't know based on the provided document."
    
    Query: {query}
    Context:
    {context}"""
    
    system = "You are a helpful document QA assistant. Be concise and accurate."
    answer = call_ollama(model, answer_prompt, system=system)
    
    return answer, thinking, retrieved_sections

# ============================================================================
# STREAMLIT UI
# ============================================================================
st.set_page_config(page_title="Vectorless RAG with Ollama", layout="wide")
st.title("🌲 PageIndex-Style Vectorless RAG (Local Ollama)")
st.markdown("Upload a PDF, build a hierarchical tree index, and query it using local LLMs (Qwen, Mistral, Falcon). **No Vector DBs. No Chunking.**")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model_name = st.selectbox("Select Ollama Model", list(LOCAL_LLM_OPTIONS.keys()))
    model_id = LOCAL_LLM_OPTIONS[selected_model_name]
    
    if "0.5b" in model_id or "1.5b" in model_id:
        st.warning("⚠️ Small models (<7B) may struggle with strict JSON formatting and complex reasoning.")
        
    st.markdown("---")
    st.header("📄 Document")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file and st.button("🚀 Build Document Tree"):
        with st.spinner("Indexing document... This may take a few minutes."):
            pdf_bytes = uploaded_file.read()
            doc, sections = build_document_tree(pdf_bytes, model_id)
            st.session_state['doc'] = doc
            st.session_state['sections'] = sections
            st.session_state['messages'] = []
            st.success("Tree built successfully!")
            
    # Display the Tree Structure
    if 'sections' in st.session_state:
        with st.expander("🌳 View Document Tree Structure", expanded=False):
            for s in st.session_state['sections']:
                st.markdown(f"**{s['id']}: {s['title']}** *(pp. {s['start_page']}-{s['end_page']})*")
                st.caption(s['summary'])
                st.divider()

# Main Chat Area
if 'doc' not in st.session_state:
    st.warning("👈 Please upload a PDF and build the document tree to start querying.")
else:
    st.subheader("💬 Chat with your Document")
    
    # Display chat history
    for msg in st.session_state['messages']:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
            if msg['role'] == 'assistant' and 'thinking' in msg:
                with st.expander("🧠 Agent Reasoning & Retrieved Sections"):
                    st.markdown(f"**Thinking Process:**\n{msg['thinking']}")
                    if msg.get('retrieved_sections'):
                        st.markdown("**Retrieved Sections:**")
                        for sec in msg['retrieved_sections']:
                            st.info(f"{sec['id']}: {sec['title']} (Pages {sec['start_page']}-{sec['end_page']})")
                            
    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        st.session_state['messages'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Searching tree and generating answer..."):
                answer, thinking, retrieved_secs = agentic_retrieve_and_answer(
                    prompt, st.session_state['doc'], st.session_state['sections'], model_id
                )
                st.markdown(answer)
                with st.expander("🧠 Agent Reasoning & Retrieved Sections"):
                    st.markdown(f"**Thinking Process:**\n{thinking}")
                    if retrieved_secs:
                        st.markdown("**Retrieved Sections:**")
                        for sec in retrieved_secs:
                            st.info(f"{sec['id']}: {sec['title']} (Pages {sec['start_page']}-{sec['end_page']})")
                            
        st.session_state['messages'].append({
            "role": "assistant", 
            "content": answer, 
            "thinking": thinking, 
            "retrieved_sections": retrieved_secs
        })
