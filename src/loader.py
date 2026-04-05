# Imports
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    DirectoryLoader, 
    TextLoader,
    Docx2txtLoader,
    WebBaseLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Tuple
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

# loader
def load_documents():

    # Load PDF documents
    pdf_documents = []
    try:
        pdf_loader = PyPDFDirectoryLoader(os.getenv("PDF_DIRECTORY"))
        pdf_documents = pdf_loader.load()
    except Exception as e:
        print(f"Error loading PDF documents: {e}")

    # Load Text documents
    text_documents = []
    try:
        text_loader = DirectoryLoader(os.getenv("TEXT_DIRECTORY"), glob="**/*.txt", show_progress=True)
        text_documents = text_loader.load()
    except Exception as e:
        print(f"Error loading text documents: {e}")

    # Load DOCX documents
    docx_documents = []
    try:
        docx_loader = DirectoryLoader(os.getenv("DOCX_DIRECTORY"), glob="**/*.docx", show_progress=True)
        docx_documents = docx_loader.load()
    except Exception as e:
        print(f"Error loading DOCX documents: {e}")

    # Load Web documents
    web_documents = []
    url = []
    while True:
        print("Enter a URL-if you have any (and/or Enter to finish):")
        url_input = input()
        if url_input.lower() == "":
            break
        url.append(url_input)

    if url:
        try:
            web_loader = WebBaseLoader(url)
            web_documents = web_loader.load()
        except Exception as e:
            print(f"Error loading web documents: {e}")


    # Combine all documents
    all_documents = pdf_documents + text_documents + docx_documents + web_documents

    # Save combined documents to a file (optional)
    with open("combined_documents.txt", "w", encoding="utf-8") as f:
        for doc in all_documents:
            f.write(doc.page_content + "\n\n")

    #split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitted_documents = text_splitter.split_documents(all_documents)

    return splitted_documents
