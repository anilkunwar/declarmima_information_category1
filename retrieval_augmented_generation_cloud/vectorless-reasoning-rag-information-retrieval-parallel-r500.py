import streamlit as st
import fitz  # PyMuPDF
import json
import re
import torch
import requests
from typing import Optional, Any

# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# ============================================================================
# HYBRID LLM ENGINE (From Provided Code)
# ============================================================================
class HybridLLM:
    def __init__(self, model_key: str, use_4bit: bool = True):
        self.model_key = model_key
        self.use_4bit = use_4bit
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.backend = None
        self.model_name = None
        self.client = None
        self.tokenizer = None
        self.model = None
        
        if model_key.startswith("ollama:"):
            self.model_name = model_key.replace("ollama:", "", 1)
            self._init_ollama()
        else:
            self.model_name = model_key
            self.backend = "transformers"

    def _init_ollama(self):
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama python package not installed. Run: pip install ollama")
        try:
            requests.get("http://localhost:11434/api/tags", timeout=5)
            self.backend = "ollama"
            self.client = ollama.Client(host="http://localhost:11434")
        except Exception as e:
            raise RuntimeError(f"Cannot connect to Ollama at localhost:11434. Is it running? Error: {e}")

    def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant.", max_new_tokens=1024, temperature=0.1):
        if self.backend == "ollama":
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            resp = self.client.chat(
                model=self.model_name, 
                messages=messages, 
                options={"temperature": temperature, "num_predict": max_new_tokens}
            )
            return resp.get("message", {}).get("content", "").strip()
            
        elif self.backend == "transformers":
            if self.model is None:
                self._load_transformers()
            
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens, 
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "assistant" in response:
                response = response.split("assistant")[-1].strip()
            return response

    def _load_transformers(self):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers not installed. Run: pip install transformers accelerate bitsandbytes")
            
        st.info(f"📥 Loading {self.model_name} on {self.device}... (This may take a minute)")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32}
        if self.use_4bit and self.device == "cuda":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if self.device == "cuda" and not self.use_4bit:
            self.model.to(self.device)
        self.model.eval()
        st.success("✅ Model loaded successfully!")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def extract_json(text: str) -> Optional[Any]:
    """Robust JSON extractor that handles markdown wrappers and malformed outputs."""
    if not text: return None
    try: return json.loads(text)
    except: pass
    
    match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL | re.IGNORECASE)
    if match:
        try: return json.loads(match.group(1))
        except: pass
        
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try: return json.loads(text[start:end+1])
            except: pass
    return None

def extract_text_from_pdf(file_bytes: bytes, max_pages: int = None) -> list[dict]:
    """Extract text page-by-page using PyMuPDF."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages_data = []
    total_pages = len(doc)
    if max_pages:
        total_pages = min(total_pages, max_pages)
        
    for i in range(total_pages):
        page = doc[i]
        text = page.get_text("text").strip()
        if text:
            pages_data.append({
                "page_num": i + 1,
                "text": text
            })
    doc.close()
    return pages_data

# ============================================================================
# VECTORLESS RAG PIPELINE (Preserved from Previous Code)
# ============================================================================
def build_document_tree(pages_data: list[dict], model: HybridLLM, chunk_size: int = 5) -> list[dict]:
    """Phase 1: Build hierarchical tree index with summaries."""
    sections = []
    total_chunks = (len(pages_data) + chunk_size - 1) // chunk_size
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, len(pages_data), chunk_size):
        chunk = pages_data[i:i+chunk_size]
        chunk_num = (i // chunk_size) + 1
        status_text.info(f"Analyzing document structure... Chunk {chunk_num}/{total_chunks}")
        
        text_with_tags = "\n".join([f"<page_{p['page_num']}>\n{p['text']}\n</page_{p['page_num']}>" for p in chunk])
        
        prompt = f"""Analyze the document text and extract main section headings and their starting page numbers.
Return a JSON list of objects: {{"title": "Section Title", "start_page": page_number}}.
Ignore headers/footers. Only include actual content sections.

Text:
{text_with_tags}"""
        
        system = "You are a precise document analyst. Return ONLY a valid JSON list."
        response = model.generate(prompt, system_prompt=system, max_new_tokens=512, temperature=0.1)
        
        parsed = extract_json(response)
        if parsed and isinstance(parsed, list):
            for item in parsed:
                if 'title' in item and 'start_page' in item:
                    sections.append(item)
                    
        progress_bar.progress(min(1.0, chunk_num / total_chunks))
        
    status_text.info("Finalizing sections and generating summaries...")
    sections.sort(key=lambda x: x['start_page'])
    for i in range(len(sections)):
        sections[i]['end_page'] = sections[i+1]['start_page'] - 1 if i + 1 < len(sections) else pages_data[-1]['page_num']
        sections[i]['id'] = f"sec_{i+1:03d}"
        
    for i, sec in enumerate(sections):
        start_idx = next((j for j, p in enumerate(pages_data) if p['page_num'] == sec['start_page']), 0)
        end_idx = next((j for j, p in enumerate(pages_data) if p['page_num'] == sec['end_page']), len(pages_data)-1)
        
        section_text = "\n".join([p['text'] for p in pages_data[start_idx:end_idx+1]])
        if len(section_text) > 4000:
            section_text = section_text[:4000] + "..."
            
        prompt = f"Summarize the following text in 2-3 concise sentences:\n\n{section_text}"
        system = "You are a helpful summarizer."
        summary = model.generate(prompt, system_prompt=system, max_new_tokens=256, temperature=0.1)
        sec['summary'] = summary.strip()
        
    progress_bar.empty()
    status_text.empty()
    return sections

def agentic_retrieve_and_answer(query: str, pages_data: list[dict], sections: list[dict], model: HybridLLM) -> tuple[str, str, list]:
    """Phase 2 & 3: Agentic Tree Search and Answer Generation."""
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
    search_response = model.generate(search_prompt, system_prompt=system, max_new_tokens=512, temperature=0.1)
    search_result = extract_json(search_response)
    
    thinking = "No reasoning provided."
    relevant_ids = []
    if search_result:
        thinking = search_result.get('thinking', 'No reasoning provided.')
        relevant_ids = search_result.get('relevant_ids', [])
        
    context = ""
    retrieved_sections = []
    for s in sections:
        if s['id'] in relevant_ids:
            retrieved_sections.append(s)
            text_parts = []
            for p_num in range(s['start_page'], s['end_page'] + 1):
                page_data = next((p for p in pages_data if p['page_num'] == p_num), None)
                if page_data:
                    text_parts.append(page_data['text'])
            
            context += f"\n\n--- Section: {s['title']} (Pages {s['start_page']}-{s['end_page']}) ---\n" + "\n".join(text_parts) + "\n"
            
    if not context:
        return "I could not find any relevant sections in the document to answer your query.", thinking, retrieved_sections
        
    if len(context) > 12000:
        context = context[:12000] + "\n[Context truncated due to length]"
        
    answer_prompt = f"""Answer the user's query based ONLY on the provided document context.
If the answer is not in the context, say "I don't know based on the provided document."

Query: {query}
Context:
{context}"""
    
    system = "You are a helpful document QA assistant. Be concise and accurate."
    answer = model.generate(answer_prompt, system_prompt=system, max_new_tokens=1024, temperature=0.1)
    
    return answer, thinking, retrieved_sections

# ============================================================================
# STREAMLIT UI
# ============================================================================
st.set_page_config(page_title="Hybrid Vectorless RAG", layout="wide")
st.title("🌲 Hybrid Vectorless RAG (No API Required)")
st.caption("Run small models directly via Transformers (Cloud/Local) or large models via Ollama (Local). No chunking, no vector DBs.")

MODEL_OPTIONS = {
    "🚀 Cloud/Local Small (No API, Transformers)": "Qwen/Qwen2.5-1.5B-Instruct",
    "🚀 Cloud/Local Tiny (Fast, CPU OK)": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "🖥️ Local Ollama: Qwen 2.5 7B": "ollama:qwen2.5:7b",
    "🖥️ Local Ollama: Mistral 7B": "ollama:mistral:7b",
    "🖥️ Local Ollama: Llama 3.1 8B": "ollama:llama3.1:8b",
}

with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model_name = st.selectbox("Select LLM Backend", list(MODEL_OPTIONS.keys()))
    model_id = MODEL_OPTIONS[selected_model_name]
    
    use_4bit = st.checkbox("Use 4-bit quantization (Saves VRAM, requires CUDA)", value=True)
    max_pages = st.slider("Max pages to process", 1, 50, 10, help="Limit this to avoid overwhelming local LLM context windows.")
    
    st.markdown("---")
    st.header("📄 Document")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

# Initialize session state
if "pages_data" not in st.session_state:
    st.session_state.pages_data = []
if "sections" not in st.session_state:
    st.session_state.sections = []
if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file and st.button("🚀 Build Document Tree"):
    with st.spinner("Extracting text from PDF..."):
        st.session_state.pages_data = extract_text_from_pdf(uploaded_file.read(), max_pages=max_pages)
    
    if not st.session_state.pages_data:
        st.warning("No text could be extracted. The PDF might be scanned images.")
    else:
        st.success(f"Extracted text from {len(st.session_state.pages_data)} pages.")
        
        try:
            llm = HybridLLM(model_key=model_id, use_4bit=use_4bit)
            with st.spinner("Building hierarchical tree index (this may take a few minutes)..."):
                st.session_state.sections = build_document_tree(st.session_state.pages_data, llm, chunk_size=5)
            st.success("Document tree built successfully!")
            st.session_state.messages = [] # Clear chat history on new doc
        except Exception as e:
            st.error(f"Failed to initialize LLM: {e}")

if st.session_state.sections:
    with st.expander("🌳 View Document Tree Structure", expanded=False):
        for s in st.session_state.sections:
            st.markdown(f"**{s['id']}: {s['title']}** *(pp. {s['start_page']}-{s['end_page']})*")
            st.caption(s['summary'])
            st.divider()

    st.subheader("💬 Chat with your Document")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
            if msg['role'] == 'assistant' and 'thinking' in msg:
                with st.expander("🧠 Agent Reasoning & Retrieved Sections"):
                    st.markdown(f"**Thinking Process:**\n{msg['thinking']}")
                    if msg.get('retrieved_sections'):
                        st.markdown("**Retrieved Sections:**")
                        for sec in msg['retrieved_sections']:
                            st.info(f"{sec['id']}: {sec['title']} (Pages {sec['start_page']}-{sec['end_page']})")
                            
    if prompt := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Searching tree and generating answer..."):
                llm = HybridLLM(model_key=model_id, use_4bit=use_4bit)
                answer, thinking, retrieved_secs = agentic_retrieve_and_answer(
                    prompt, st.session_state.pages_data, st.session_state.sections, llm
                )
                st.markdown(answer)
                with st.expander("🧠 Agent Reasoning & Retrieved Sections"):
                    st.markdown(f"**Thinking Process:**\n{thinking}")
                    if retrieved_secs:
                        st.markdown("**Retrieved Sections:**")
                        for sec in retrieved_secs:
                            st.info(f"{sec['id']}: {sec['title']} (Pages {sec['start_page']}-{sec['end_page']})")
                            
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer, 
            "thinking": thinking, 
            "retrieved_sections": retrieved_secs
        })
else:
    st.info("👈 Please upload a PDF file and click 'Build Document Tree' to begin.")
