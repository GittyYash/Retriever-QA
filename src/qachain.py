from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

system_prompt = """You are a helpful assistant that answers questions based on the provided documents. 
Use only the information from the documents to answer the question. If the answer is not in the documents, say you don't know.

relevant_documents:
{relevant_documents}

user_query:
{query}
"""

def generate_response(model: str, query: str, relevant_documents: list) -> str:
    chat_model = init_chat_model(model)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{query}"),
        ]
    )

    chain = prompt | chat_model

    response = chain.invoke({
        "relevant_documents": relevant_documents,
        "query": query
    })

    return response.content