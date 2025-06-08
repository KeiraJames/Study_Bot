import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document # To wrap raw text
import google.generativeai as genai # Import the base Google library
import os 

# --- Configuration ---
st.set_page_config(page_title="Simple RAG with Gemini", layout="wide")
st.title("📄 Study Bot")

# --- Helper function to initialize models ---
@st.cache_resource
def get_models(api_key):
    try:
        # Configure the base client for listing models
        genai.configure(api_key=api_key)
      
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest', google_api_key=api_key, convert_system_message_to_human=True)

        return embeddings, llm

    except Exception as e:
        st.error("This could mean the model selected from the list is still not compatible with Langchain's wrappers or there's another configuration issue. Check the exact error message.")
        return None, None




# --- Main Application ---
google_api_key_env = os.getenv("GOOGLE_API_KEY")
google_api_key = "AIzaSyDMYArQqF4gjHTVXAVmcwEGwMG4iZDKRh4"


embeddings_model, llm = get_models(google_api_key)


st.subheader("1. Provide Document Context")
document_text = st.text_area("Paste your document text here:", height=200, key="doc_text")

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'doc_processed_text' not in st.session_state:
    st.session_state.doc_processed_text = ""

if document_text:
    if document_text != st.session_state.doc_processed_text:
        with st.spinner("Processing document..."):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs_for_splitting = [Document(page_content=document_text)]
            chunks = text_splitter.split_documents(docs_for_splitting)

            if chunks:
                try:
                    st.session_state.vector_store = FAISS.from_documents(chunks, embeddings_model)
                    st.session_state.doc_processed_text = document_text
                    st.success("Document processed and indexed in memory!")
                except Exception as e:
                    st.error(f"Error creating vector store: {e}")
                    st.session_state.vector_store = None
            else:
                st.warning("No text chunks found to process.")
                st.session_state.vector_store = None
elif st.session_state.vector_store: # If text area is cleared, clear the vector store
    st.session_state.vector_store = None
    st.session_state.doc_processed_text = ""
    st.info("Document context cleared.")

st.subheader("2. Ask a Question")
user_question = st.text_input("Enter your question about the document:", key="user_question")

if st.button("Get Answer"):
    if not st.session_state.vector_store:
        st.warning("Please provide document text and process it first.")
    elif not user_question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching for answers..."):
            try:
                retriever = st.session_state.vector_store.as_retriever()
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
                )
                response = qa_chain.invoke({"query": user_question})
                st.subheader("💡 Answer:")
                st.write(response["result"])
                with st.expander("Show Retrieved Context (Source Chunks)"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.caption(doc.page_content)
            except Exception as e:
                st.error(f"Error during Q&A: {e}")
                st.info("This could be an issue with the LLM, the retriever, or the API.")

st.markdown("---")
st.caption("A simple RAG implementation using Gemini and FAISS .")
