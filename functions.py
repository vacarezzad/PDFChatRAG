import logging
import os
import pathlib
import tempfile
from typing import Any, List

import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import (PyPDFLoader, TextLoader,
                                        UnstructuredEPubLoader,
                                        UnstructuredWordDocumentLoader)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.schema import BaseRetriever, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain_core.prompts.prompt import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# ================================================================
# Environment Setup
# ================================================================

def set_environment():
    """Set environment variables for API keys and IDs."""
    for key, value in globals().items():
        if "API" in key or "ID" in key:
            os.environ[key] = value
            
GOOGLE_API_KEY= 'your_google_api_key'
set_environment()

# ================================================================
# Custom EPub Loader
# ================================================================

class EpubReader(UnstructuredEPubLoader):
    def __init__(self, file_path: str | list[str], **kwargs: Any):
        super().__init__(file_path, **kwargs, mode="elements", strategy="fast")

# ================================================================
# Document Loader and Exception Handling
# ================================================================

class DocumentLoaderException(Exception):
    """Exception raised for errors in the document loading process."""
    pass

class DocumentLoader:
    """Loads a document with a supported extension."""
    supported_extensions = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".epub": EpubReader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader
    }

    @staticmethod
    def load_document(temp_filepath: str) -> List[Document]:
        """Load a file and return it as a list of documents."""
        ext = pathlib.Path(temp_filepath).suffix
        loader_class = DocumentLoader.supported_extensions.get(ext)
        if not loader_class:
            raise DocumentLoaderException(f"Invalid extension type {ext}, cannot load this type of file")
        
        loader = loader_class(temp_filepath)
        docs = loader.load()
        logging.info(docs)
        return docs

# ================================================================
# Retriever Configuration
# ================================================================

def configure_retriever(docs: List[Document], use_compression=False) -> BaseRetriever:
    """Configure a retriever with optional compression."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})
    
    if use_compression:
        embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
        retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=retriever)
    
    return retriever

# ================================================================
# Conversational Chain Configuration
# ================================================================

def configure_chain(retriever: BaseRetriever) -> ConversationalRetrievalChain:
    """Configure the conversational chain with a retriever."""
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

    return ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True, max_tokens_limit=4000
    )

def configure_qa_chain(uploaded_files) -> ConversationalRetrievalChain:
    """Read documents, configure retriever, and set up the QA chain."""
    docs = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            temp_filepath = os.path.join(temp_dir, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
            docs.extend(DocumentLoader.load_document(temp_filepath))
    
    retriever = configure_retriever(docs)
    return configure_chain(retriever)

# ================================================================
# Chat History Management
# ================================================================

def initialize_chat_history():
    """Initialize chat history in the Streamlit session if not exists."""
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

def add_message_to_history(role, message):
    """Add a message to the chat history."""
    st.session_state['chat_history'].append((role, message))

def display_chat_history():
    """Display chat history in the Streamlit interface."""
    for role, message in st.session_state['chat_history']:
        if role == "user":
            st.chat_message("user").markdown(message)
        else:
            st.chat_message("assistant").markdown(message)

# ================================================================
# User Query Processing
# ================================================================

def process_user_query(qa_chain, user_query):
    """Process the user's query and return the assistant's response."""
    assistant = st.chat_message("assistant")
    template = PromptTemplate(
        input_variables=["pregunta", "lenguaje"],
        template="Contesta a la siguiente pregunta: {pregunta} en {lenguaje}"
    )

    prompt = template.format(pregunta=user_query, lenguaje="español")

    stream_handler = StreamlitCallbackHandler(assistant)
    response = qa_chain.run(prompt, callbacks=[stream_handler])
    return response
