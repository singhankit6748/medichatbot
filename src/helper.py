from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

# Extract data from PDFs in a directory
def load_pdf_file(data_path: str) -> List[Document]:
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()

# Filter documents to keep only page_content and source metadata
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(page_content=doc.page_content, metadata={"source": src})
        )
    return minimal_docs

# Split the documents into smaller chunks
def text_split(minimal_docs: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    return text_splitter.split_documents(minimal_docs)

# Download HuggingFace embeddings (PyTorch only)
def download_hugging_face_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"}  # Force PyTorch CPU mode
    )
    return embeddings
