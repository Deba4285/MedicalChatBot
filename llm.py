from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import os


DATA_PATH="data/"
def load_pdf_file(data):
    loader=DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents=loader.load()
    return documents

documents=load_pdf_file(data=DATA_PATH)
#print(len(documents))

def create_chunk(data):
    txt_split=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    txt=txt_split.split_documents(data)
    return txt

chunks=create_chunk(data=documents)
#print(f"txt size:", len(chunks))
def create_emd():
    embd=HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embd
embeding_model=create_emd()

db_path="vectorstore/db_faiss"
db=FAISS.from_documents(chunks,embeding_model)
db.save_local(db_path)