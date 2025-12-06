"""
RAG Pipeline Core Logic
Handles document processing, retrieval, and answer generation
"""

from typing import List, Dict, TypedDict
import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END


class GraphState(TypedDict):
    """State definition for LangGraph workflow"""
    question: str
    context: str
    answer: str
    needs_retrieval: bool
    retrieved_docs: List[Document]


class RAGPipeline:
    """Main RAG Pipeline class"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.vectorstore = None
        self.embeddings_model = None
        self._load_embeddings_model()
    
    def _load_embeddings_model(self):
        """Load HuggingFace embeddings model"""
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def load_pdfs_from_directory(self, directory_path: str) -> tuple[List[Document], Dict]:
        """
        Load all PDFs from directory and split into chunks
        
        Returns:
            Tuple of (document_chunks, metadata_dict)
        """
        all_documents = []
        metadata = {
            'pdf_files': [],
            'total_pages': 0,
            'total_chunks': 0
        }
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Get PDF files
        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in: {directory_path}")
        
        # Load each PDF
        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory_path, pdf_file)
            
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                
                # Add metadata
                for doc in documents:
                    doc.metadata['source'] = pdf_file
                
                all_documents.extend(documents)
                metadata['pdf_files'].append(pdf_file)
                metadata['total_pages'] += len(documents)
                
            except Exception as e:
                print(f"Warning: Could not load {pdf_file}: {e}")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        splits = text_splitter.split_documents(all_documents)
        metadata['total_chunks'] = len(splits)
        
        return splits, metadata
    
    def create_vectorstore(self, documents: List[Document]):
        """Create FAISS vectorstore from documents"""
        self.vectorstore = FAISS.from_documents(documents, self.embeddings_model)
    
    def _assess_retrieval_need(self, state: GraphState) -> GraphState:
        """Determine if retrieval is needed"""
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=self.api_key
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at determining if a question requires looking up information 
            from research papers. Respond with ONLY 'yes' or 'no'.
            
            Answer 'yes' if the question asks about specific research findings, methodologies, results,
            technical concepts, or requires citations.
            
            Answer 'no' for general greetings, casual conversation, or simple factual questions."""),
            ("human", "Does this question require retrieving information from research papers?\n\nQuestion: {question}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"question": state["question"]})
        
        state["needs_retrieval"] = "yes" in response.content.lower()
        return state
    
    def _retrieve_documents(self, state: GraphState) -> GraphState:
        """Retrieve relevant documents"""
        if not state["needs_retrieval"]:
            state["retrieved_docs"] = []
            state["context"] = ""
            return state
        
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        docs = retriever.invoke(state["question"])
        state["retrieved_docs"] = docs
        
        # Format context
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            context_parts.append(
                f"[Document {i} - {source}, Page {page}]\n{doc.page_content}\n"
            )
        
        state["context"] = "\n".join(context_parts)
        return state
    
    def _generate_answer(self, state: GraphState) -> GraphState:
        """Generate final answer"""
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=self.api_key
        )
        
        if state["needs_retrieval"] and state["context"]:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a research assistant analyzing scientific papers. 

Provide comprehensive, well-structured answers with:
1. **Clear Overview**: Start with a concise summary
2. **Key Findings**: Use bullet points with important details
3. **Supporting Evidence**: Include specific data, methodologies, or results
4. **Citations**: Reference sources as [Source: Document_name, Page X]

Format your response with markdown for readability:
- Use **bold** for emphasis
- Use bullet points for lists
- Use numbered lists for sequential information
- Keep paragraphs concise and focused"""),
                ("human", """Context from research papers:
{context}

Question: {question}

Provide a detailed, well-structured answer with clear formatting and citations.""")
            ])
            
            chain = prompt | llm
            response = chain.invoke({
                "context": state["context"],
                "question": state["question"]
            })
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant. Provide clear, friendly responses."),
                ("human", "{question}")
            ])
            
            chain = prompt | llm
            response = chain.invoke({"question": state["question"]})
        
        state["answer"] = response.content
        return state
    
    def create_graph(self):
        """Create LangGraph workflow"""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("assess_need", self._assess_retrieval_need)
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("generate", self._generate_answer)
        
        # Define edges
        workflow.set_entry_point("assess_need")
        workflow.add_edge("assess_need", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def query(self, question: str) -> Dict:
        """
        Main query method
        
        Returns:
            Dict with answer, retrieved_docs, and metadata
        """
        graph = self.create_graph()
        
        initial_state = {
            "question": question,
            "context": "",
            "answer": "",
            "needs_retrieval": False,
            "retrieved_docs": []
        }
        
        result = graph.invoke(initial_state)
        
        return {
            "answer": result["answer"],
            "retrieved_docs": result["retrieved_docs"],
            "needs_retrieval": result["needs_retrieval"]
        }
