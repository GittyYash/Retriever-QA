from src.loader import load_documents
from src.vectorstore import create_vector_store
from src.retriever import retrieve_relevant_documents
from src.qachain import generate_response
from dotenv import load_dotenv

load_dotenv()

def main():
    # Load Documents
    print("Loading documents...")
    docs = load_documents()

    # Create vector store
    print("Creating vector store...")
    create_vector_store(docs)

    # User Query
    print("Enter your query:")
    query = input()

    # Retrieve relevant documents
    print("Retrieving relevant documents...")
    relevant_documents = retrieve_relevant_documents(query)

    # Generate response
    print("Generating response...")
    response = generate_response('gpt-3.5-turbo', query, relevant_documents)
    print("Response:")
    print(response)


if __name__ == "__main__":
    main()
 