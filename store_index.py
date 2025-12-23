from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY is not set in .env")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

print("✅ Environment variables loaded.")

# Step 1: Load and process documents
print("📂 Loading PDF files from data/ ...")
extracted_data = load_pdf_file("data/")
print(f"📄 Loaded {len(extracted_data)} documents.")

filter_data = filter_to_minimal_docs(extracted_data)
texts_chunk = text_split(filter_data)
print(f"✂️ Split into {len(texts_chunk)} text chunks.")

# Step 2: Initialize embeddings (PyTorch only — avoids TensorFlow/Keras)
print("🔍 Initializing HuggingFace embeddings (PyTorch)...")
embeddings = download_hugging_face_embeddings()

# Step 3: Connect to Pinecone
print("🗄 Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

# Step 4: Create index if not exists (new Pinecone SDK method)
if index_name not in pc.list_indexes().names():
    print(f"📦 Creating new Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print(f"✅ Pinecone index '{index_name}' already exists.")

# Step 5: Connect to the index
index = pc.Index(index_name)

# Step 6: Store documents
print("📤 Uploading documents to Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    index_name=index_name,
    embedding=embeddings,
)
print("✅ Documents successfully uploaded to Pinecone!")
