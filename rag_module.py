"""
Optional RAG (Retrieval Augmented Generation) module
Only loaded when needed to reduce deployment size
"""

import os

class RAG:
    def __init__(self):
        self.ready = False
        self._initialized = False
        
    def _lazy_init(self):
        """Lazy initialization - only import heavy dependencies when needed"""
        if self._initialized:
            return
            
        try:
            # Import heavy dependencies only when needed
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
            from langchain_community.vectorstores import FAISS
            from langchain.chains import RetrievalQA
            
            self.PyPDFLoader = PyPDFLoader
            self.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
            self.GoogleGenerativeAIEmbeddings = GoogleGenerativeAIEmbeddings
            self.ChatGoogleGenerativeAI = ChatGoogleGenerativeAI
            self.FAISS = FAISS
            self.RetrievalQA = RetrievalQA
            
            self._initialized = True
        except ImportError as e:
            print(f"Warning: RAG dependencies not available: {e}")
            print("Install with: pip install langchain langchain-community langchain-google-genai faiss-cpu")
            self._initialized = False
    
    def build_from_pdf(self, pdf_path):
        """Build knowledge base from PDF"""
        import urllib.request
        import tempfile
        
        is_url = pdf_path.startswith("http://") or pdf_path.startswith("https://")
        local_pdf_path = pdf_path
        temp_file = None
        
        if is_url:
            try:
                temp_dir = tempfile.gettempdir()
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=temp_dir)
                print(f"Downloading PDF from remote URL to local temp file: {temp_file.name}")
                urllib.request.urlretrieve(pdf_path, temp_file.name)
                local_pdf_path = temp_file.name
            except Exception as e:
                print(f"Error downloading PDF from remote URL: {e}")
                return
                
        if not os.path.exists(local_pdf_path):
            print(f"PDF not found: {local_pdf_path}")
            return
            
        self._lazy_init()
        if not self._initialized:
            print("RAG dependencies not available")
            return
            
        try:
            loader = self.PyPDFLoader(local_pdf_path)
            documents = loader.load()
            
            text_splitter = self.RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            embeddings = self.GoogleGenerativeAIEmbeddings(
                model="models/embedding-001"
            )
            
            self.vectorstore = self.FAISS.from_documents(chunks, embeddings)
            
            llm = self.ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=0.3
            )
            
            self.qa_chain = self.RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
            )
            
            self.ready = True
            print("RAG knowledge base built successfully")
        except Exception as e:
            print(f"Error building RAG: {e}")
            self.ready = False
        finally:
            if temp_file:
                try:
                    os.unlink(local_pdf_path)
                    print(f"Cleaned up local temp file: {local_pdf_path}")
                except Exception as e:
                    print(f"Error cleaning up temp file: {e}")
    
    def answer(self, question):
        """Answer a question using the knowledge base"""
        if not self.ready:
            return "Knowledge base not ready. Please upload a PDF first."
            
        if not self._initialized:
            return "RAG system not available."
            
        try:
            result = self.qa_chain.invoke({"query": question})
            return result.get("result", "Sorry, I couldn't find an answer.")
        except Exception as e:
            print(f"Error answering question: {e}")
            return "Sorry, an error occurred while processing your question."

# Global instance
rag = RAG()
