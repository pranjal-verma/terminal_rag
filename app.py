import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from llm_helper import add_documents_batched, get_embeddings

PERSIST_DIR = "./chroma_db"
SOURCE_URL = "https://en.wikipedia.org/wiki/Artificial_intelligence"

print("📥 Loading webpage...")
loader = WebBaseLoader(SOURCE_URL)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(f"✅ Split into {len(splits)} chunks!")

embeddings_model = get_embeddings()

if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    print("📂 Loading existing Chroma database (skipping re-embedding)...")
    vector_store = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings_model,
    )
else:
    print("🧠 Generating embeddings and saving to Chroma...")
    print(f"   ({len(splits)} chunks — batched to avoid rate limits)")

    vector_store = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings_model,
    )
    add_documents_batched(vector_store, splits)
    print("✅ Vector database saved!")

# Test search
query = "What did John McCarthy do for AI?"
similar_docs = vector_store.similarity_search(query, k=3)

print("\n🔍 --- TEST SEARCH RESULTS ---")
for i, doc in enumerate(similar_docs):
    print(f"Chunk {i+1}:\n{doc.page_content[:200]}...")
    print("-" * 30)
