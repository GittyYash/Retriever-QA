from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def create_vector_store(documents):
    # Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)

    # Create vector store
    vector_store = FAISS.from_documents(documents, embeddings)

    # Save vector store
    vector_store.save_local("faiss_index")