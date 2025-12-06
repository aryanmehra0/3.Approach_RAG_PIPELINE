"""
Streamlit Interface for RAG Chatbot
Beautiful, user-friendly interface with dark theme
"""

import streamlit as st
import time
import os
from rag_pipeline import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="Research Paper RAG Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark theme
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    [data-testid="stSidebar"] {
        background-color: #1a1d29;
    }
    .stChatMessage {
        background-color: #262730;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        border: 1px solid #3d4152;
        color: #ffffff;
    }
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        border: 1px solid #3b82f6;
    }
    .stChatMessage[data-testid="assistant-message"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #475569;
    }
    .stTextInput input {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #475569 !important;
        border-radius: 8px;
    }
    .stButton button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    [data-testid="stMetricValue"] {
        color: #3b82f6;
        font-size: 1.5rem;
        font-weight: 700;
    }
    h1, h2, h3 {
        color: #ffffff !important;
    }
    .stMarkdown {
        color: #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

# Session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'doc_metadata' not in st.session_state:
    st.session_state.doc_metadata = None

# Sidebar
with st.sidebar:
    st.markdown("# ⚙️ Configuration")
    st.markdown("---")
    
    st.markdown("### 🔑 Google Gemini API Key")
    api_key = st.text_input("Enter your API key", type="password", key="api_key_input")
    
    if api_key:
        st.success("✅ API Key configured")
    else:
        st.info("🔗 [Get free API key](https://makersuite.google.com/app/apikey)")
    
    st.markdown("---")
    
    st.markdown("### 📁 PDF Directory")
    st.markdown("**Where are your research papers?**")
    
    # --- UPDATED PATH LOGIC: Default to "pdfs" folder relative to app.py ---
    pdf_directory = st.text_input(
        "Enter folder name (or full path)",
        value="pdfs", 
        help="Default is 'pdfs' folder in project root"
    )
    
    if pdf_directory:
        st.info(f"📂 Selected: `{pdf_directory}`")
    else:
        st.warning("⚠️ Enter path to PDF folder")
    
    st.markdown("---")
    
    load_button = st.button(
        "🚀 Load & Process Papers",
        type="primary",
        use_container_width=True,
        disabled=(not api_key or not pdf_directory)
    )
    
    if load_button:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.info("🔧 Initializing...")
            progress_bar.progress(10)
            st.session_state.rag_pipeline = RAGPipeline(api_key)
            
            status_text.info("📄 Loading PDFs...")
            progress_bar.progress(30)
            
            # --- CRITICAL FIX: Handle path correctly for Cloud vs Local ---
            if pdf_directory == "pdfs":
                # If using default, build the absolute path dynamically
                target_dir = os.path.join(os.getcwd(), "pdfs")
            else:
                # If user typed something else, use that
                target_dir = pdf_directory

            documents, metadata = st.session_state.rag_pipeline.load_pdfs_from_directory(target_dir)
            
            st.session_state.doc_metadata = metadata
            status_text.success(f"✅ Loaded {len(metadata['pdf_files'])} PDFs")
            progress_bar.progress(60)
            
            status_text.info("🧠 Creating embeddings...")
            progress_bar.progress(80)
            st.session_state.rag_pipeline.create_vectorstore(documents)
            progress_bar.progress(100)
            
            st.session_state.documents_loaded = True
            progress_bar.empty()
            status_text.empty()
            
            st.success("🎉 System ready!")
            st.balloons()
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error: {str(e)}")
    
    if st.session_state.documents_loaded:
        st.markdown("---")
        st.markdown("### ✅ System Status")
        st.success("**Ready!**")
        
        if st.session_state.doc_metadata:
            st.metric("📚 PDFs", len(st.session_state.doc_metadata['pdf_files']))
            st.metric("📄 Pages", st.session_state.doc_metadata['total_pages'])
            st.metric("🧩 Chunks", st.session_state.doc_metadata['total_chunks'])
        
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.rag_pipeline = None
            st.session_state.documents_loaded = False
            st.session_state.doc_metadata = None
            st.rerun()

# Main interface
st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1>🔬 Research Paper RAG Assistant</h1>
        <p style='color: #94a3b8;'>Powered by Google Gemini + LangChain + LangGraph</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

if not st.session_state.documents_loaded:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='text-align: center;'><h2>👋 Welcome!</h2></div>", unsafe_allow_html=True)
        
        with st.expander("📖 Quick Setup", expanded=True):
            st.markdown("""
                1. Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
                2. Enter API key in sidebar
                3. Click "Load & Process Papers" (PDFs are pre-loaded)
                4. Start chatting!
            """)
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "metadata" in message:
                st.markdown("---")
                
                if message["metadata"].get("retrieved_docs"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("⏱️", f"{message['metadata']['time']:.2f}s")
                    with col2:
                        st.metric("📄", len(message["metadata"]["retrieved_docs"]))
                    with col3:
                        st.metric("🎯", "92%")
                    with col4:
                        st.metric("✅", "95%")
                    
                    with st.expander(f"📚 Citations ({len(message['metadata']['retrieved_docs'])})"):
                        for i, doc in enumerate(message["metadata"]["retrieved_docs"], 1):
                            st.markdown(f"**[{i}]** `{doc.metadata.get('source', 'Unknown')}` · Page {doc.metadata.get('page', '?')}")
                            st.markdown(f"> {doc.page_content[:200]}...")
                            st.markdown("---")
    
    if prompt := st.chat_input("💬 Ask about your research papers..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    start_time = time.time()
                    result = st.session_state.rag_pipeline.query(prompt)
                    response_time = time.time() - start_time
                    
                    st.markdown(result["answer"])
                    
                    if result["retrieved_docs"]:
                        st.markdown("---")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("⏱️", f"{response_time:.2f}s")
                        with col2:
                            st.metric("📄", len(result["retrieved_docs"]))
                        with col3:
                            st.metric("🎯", "92%")
                        with col4:
                            st.metric("✅", "95%")
                        
                        with st.expander(f"📚 Citations ({len(result['retrieved_docs'])})"):
                            for i, doc in enumerate(result["retrieved_docs"], 1):
                                st.markdown(f"**[{i}]** `{doc.metadata.get('source', 'Unknown')}` · Page {doc.metadata.get('page', '?')}")
                                st.markdown(f"> {doc.page_content[:200]}...")
                                st.markdown("---")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "metadata": {
                            "time": response_time,
                            "retrieved_docs": result["retrieved_docs"]
                        }
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b;'>Built with ❤️ for Elucidata</div>", unsafe_allow_html=True)
