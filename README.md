# 🔬 Research Paper RAG Assistant (Hosted link -  https://3approachragpipeline-eeo3knonhbejjbfkrmxyt9.streamlit.app/ )


A production-grade RAG system for querying research papers with intelligent retrieval and citation.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your PDFs:**
   - Place all research papers in `pdfs/` folder

3. **Get API key:**
   - Visit: https://makersuite.google.com/app/apikey
   - Create free API key

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

5. **Configure in UI:**
   - Enter API key
   - Enter PDF folder path
   - Click "Load & Process Papers"
   - Start chatting!

## Features

- ✅ Intelligent document retrieval
- ✅ Detailed answers with citations
- ✅ Beautiful dark theme UI
- ✅ Performance metrics
- ✅ Source tracking

## Architecture

```
PDFs → Chunking → Embeddings → Vector DB → Retrieval → LLM → Answer
```

## Requirements

- Python 3.8+
- Google Gemini API key (free)
- 7 research papers (PDF format)

## Example Questions

- "What are the main findings?"
- "Summarize the methodology"
- "What are the limitations?"
- "Compare results across papers"

## Built With

- Streamlit (UI)
- LangChain (Document processing)
- LangGraph (Workflow)
- Google Gemini (LLM)
- HuggingFace (Embeddings)
- FAISS (Vector search)
