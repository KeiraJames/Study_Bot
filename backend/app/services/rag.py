from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
import google.generativeai as genai
import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest', google_api_key=GOOGLE_API_KEY, temperature=0.2)

vector_store = None
document_text = None

def process_document(file_bytes):
    global vector_store, document_text
    
    try:
        reader = PdfReader(file_bytes)
        document_text = "".join([page.extract_text() for page in reader.pages])
    except Exception:
        document_text = file_bytes.read().decode("utf-8")
    
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents([Document(page_content=document_text)])
    vector_store = FAISS.from_documents(chunks, embeddings_model)
    
    return document_text


def answer_question(question):
    if not vector_store:
        return "No document loaded."
    retriever = vector_store.as_retriever()
    
   
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)
    response = qa_chain.invoke({"query": question})
    
    return response["result"]
