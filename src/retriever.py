from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def retrieve_relevant_documents(query):
    # Load vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Retrieve relevant documents
    relevant_documents = vector_store.similarity_search(query, k=5)

    return relevant_documents