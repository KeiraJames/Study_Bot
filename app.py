import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document # To wrap raw text

# --- Configuration ---
st.set_page_config(page_title="Simple RAG with Gemini", layout="wide")
st.title("📄 Simple RAG with Google Gemini")

# --- Helper function to initialize models (cached) ---
@st.cache_resource
def get_models(api_key):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, convert_system_message_to_human=True)
        return embeddings, llm
    except Exception as e:
        st.error(f"Error initializing Google models: {e}")
        return None, None

# --- Main Application ---

# 1. Get Google API Key
google_api_key = st.text_input("Enter your Google API Key:", type="password", help="Required for Gemini LLM and Embeddings.")

if not google_api_key:
    st.warning("Please enter your Google API Key to proceed.")
    st.stop()

# Initialize models
embeddings_model, llm = get_models(google_api_key)

if not embeddings_model or not llm:
    st.stop()

# 2. Input Document Context
st.subheader("1. Provide Document Context")
document_text = st.text_area("Paste your document text here:", height=200, key="doc_text")

# Session state to store the vector store
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'doc_processed_text' not in st.session_state:
    st.session_state.doc_processed_text = ""


if document_text:
    # Only re-process if the document text has changed
    if document_text != st.session_state.doc_processed_text:
        with st.spinner("Processing document..."):
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            # Langchain's FAISS.from_texts expects list of texts, not Document objects directly for this method.
            # So we first split, then create Document objects if needed, or just pass texts.
            # For simplicity here, we'll wrap the single document_text into a list for splitting.
            # If you have multiple docs, you'd extend this.
            docs_for_splitting = [Document(page_content=document_text)]
            chunks = text_splitter.split_documents(docs_for_splitting)

            if chunks:
                try:
                    # Create FAISS vector store from chunks (in-memory)
                    st.session_state.vector_store = FAISS.from_documents(chunks, embeddings_model)
                    st.session_state.doc_processed_text = document_text # Store processed text
                    st.success("Document processed and indexed in memory!")
                except Exception as e:
                    st.error(f"Error creating vector store: {e}")
                    st.session_state.vector_store = None # Reset on error
            else:
                st.warning("No text chunks found to process.")
                st.session_state.vector_store = None # Reset if no chunks

# 3. Ask a Question
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
                    chain_type="stuff", # "stuff" puts all retrieved docs into the prompt
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

st.markdown("---")
st.caption("A simple RAG implementation using Gemini and FAISS (in-memory).")
