# MarxBot: Ask Marx Anything

# Project Overview
# A RAG (Retrieval-Augmented Generation) chatbot using TinyLlama LLM,
# backed by a vector database created from classic Marxist texts (public domain).

# Requirements
# pip install langchain chromadb sentence-transformers langchain-community

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import os

# Step 1: Load Documents (.txt, .pdf, .docx)
doc_dir = "marxtexts"
texts = []
for filename in os.listdir(doc_dir):
    filepath = os.path.join(doc_dir, filename)

    if filename.endswith(".txt"):
        loader = TextLoader(filepath, encoding='utf-8')
        texts.extend(loader.load())

    elif filename.endswith(".pdf"):
        try:
            loader = PyPDFLoader(filepath)
            texts.extend(loader.load())
        except Exception:
            loader = UnstructuredPDFLoader(filepath)
            texts.extend(loader.load())

    elif filename.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(filepath)
        texts.extend(loader.load())

# Step 2: Split & Embed
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(texts)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embedding=embedding_model, persist_directory="marx_index")
vectorstore.persist()
retriever = vectorstore.as_retriever()

# Step 3: Load TinyLlama using Ollama
llm = Ollama(model="tinyllama")

# Step 4: Create QA Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# Step 5: Ask MarxBot
while True:
    query = input("\nAsk MarxBot: ")
    if query.lower() in ["exit", "quit"]:
        break
    result = qa_chain.run(query)
    print("\n", result)
