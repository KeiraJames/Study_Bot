import streamlit as st
import os
import weaviate
from dotenv import load_dotenv # For local testing with .env file

# Langchain components
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# --- CONFIGURATION ---
# Load .env file if it exists (for local development)
load_dotenv()

# For deployment (e.g., Streamlit Community Cloud), set these as Secrets
GOOGLE_API_KEY = "2b10X3YLMd8PNAuKOCVPt7MeUe"
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080") # Default to local if not set
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY") # If your Weaviate instance needs an API key

WEAVIATE_CLASS_NAME = "StreamlitPublicDocs"

# --- INITIALIZE MODELS AND CLIENTS (CACHED) ---
@st.cache_resource
def get_weaviate_client():
    try:
        auth_config = None
        if WEAVIATE_API_KEY:
            auth_config = weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY)

        client = weaviate.Client(
            url=WEAVIATE_URL,
            auth_client_secret=auth_config
        )
        if not client.is_ready():
            st.error(f"🔴 Weaviate not ready at {WEAVIATE_URL}. Check URL and API key if used.")
            return None
        return client
    except Exception as e:
        st.error(f"🔴 Failed to connect to Weaviate: {e}")
        st.info(f"Using Weaviate URL: {WEAVIATE_URL}")
        return None

@st.cache_resource
def get_embeddings_model():
    if not GOOGLE_API_KEY:
        st.error("🔴 GOOGLE_API_KEY not found. Please set it as a secret/environment variable.")
        return None
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

@st.cache_resource
def get_llm():
    if not GOOGLE_API_KEY:
        st.error("🔴 GOOGLE_API_KEY not found.")
        return None
    return ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)

# --- Initialize core components ---
# These will only run once per session or if the function inputs change (due to @st.cache_resource)
client = get_weaviate_client()
embeddings_model = get_embeddings_model()
llm = get_llm()

# --- STREAMLIT UI ---
st.set_page_config(page_title="Doc Q&A with Weaviate & Gemini", layout="wide")
st.title("📄 Ask Questions About Your Documents")
st.markdown("Upload PDF or TXT files, index them, and then ask questions.")

if not GOOGLE_API_KEY:
    st.error("🔴 **Setup Required:** `GOOGLE_API_KEY` is not set. Please configure it in your environment or secrets.")
if not client:
    st.error("🔴 **Setup Required:** Could not connect to Weaviate. Please ensure `WEAVIATE_URL` (and `WEAVIATE_API_KEY` if needed) are correctly set and Weaviate is accessible.")

# Initialize session state
if 'docs_indexed_successfully' not in st.session_state:
    st.session_state.docs_indexed_successfully = False
if 'error_indexing' not in st.session_state:
    st.session_state.error_indexing = None


# --- Part 1: Document Upload and Indexing ---
with st.sidebar:
    st.header("1. Document Processing")
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Files are processed in memory and temporarily stored for indexing."
    )

    if st.button("Process and Index Documents", disabled=(not client or not embeddings_model)):
        if uploaded_files:
            st.session_state.docs_indexed_successfully = False # Reset status
            st.session_state.error_indexing = None
            all_docs_content = []
            temp_dir = "temp_uploaded_files_streamlit" # Temporary directory for processing
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            with st.spinner("Extracting text from files..."):
                for uploaded_file in uploaded_files:
                    # Save uploaded file temporarily to disk for Langchain loaders
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    try:
                        if uploaded_file.name.endswith(".pdf"):
                            loader = PyPDFLoader(temp_file_path)
                        elif uploaded_file.name.endswith(".txt"):
                            loader = TextLoader(temp_file_path, encoding='utf-8') # Specify encoding
                        else:
                            continue # Should not happen due to type filter
                        all_docs_content.extend(loader.load())
                    except Exception as e:
                        st.warning(f"Could not process {uploaded_file.name}: {e}")
                    finally:
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path) # Clean up

                if os.path.exists(temp_dir) and not os.listdir(temp_dir): # Cleanup dir if empty
                    os.rmdir(temp_dir)


            if all_docs_content:
                with st.spinner("Splitting documents into chunks..."):
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    chunks = text_splitter.split_documents(all_docs_content)

                if chunks:
                    with st.spinner(f"Embedding {len(chunks)} chunks and storing in Weaviate... (This may take a moment)"):
                        try:
                            # WeaviateVectorStore will create the class if it doesn't exist
                            # with a schema based on text2vec-transformers (if Weaviate is configured that way)
                            # OR it will use the provided `embeddings_model` to generate vectors.
                            # For this setup, we rely on Langchain providing the vectors from Google.
                            # Weaviate class should be configured to accept custom vectors.
                            # This is typically done by NOT specifying a vectorizer for the class in Weaviate.

                            # For simplicity, if class exists, we assume it's compatible.
                            # More robust: check schema or delete/recreate class.
                            # if client.schema.exists(WEAVIATE_CLASS_NAME):
                            # client.schema.delete_class(WEAVIATE_CLASS_NAME) # Uncomment to always re-index fresh

                            WeaviateVectorStore.from_documents(
                                client=client,
                                documents=chunks,
                                embedding=embeddings_model,
                                index_name=WEAVIATE_CLASS_NAME,
                                text_key="text" # default key
                            )
                            st.session_state.docs_indexed_successfully = True
                            st.success(f"✅ Successfully indexed {len(chunks)} chunks into Weaviate class '{WEAVIATE_CLASS_NAME}'.")
                        except Exception as e:
                            st.session_state.error_indexing = f"🔴 Error indexing documents in Weaviate: {e}"
                            st.error(st.session_state.error_indexing)
                            st.info("Hints: Ensure Weaviate is running and accessible. If the class already exists, its schema might be incompatible (e.g., different vectorizer). You might need to delete the class in Weaviate manually if re-indexing with different settings.")
                else:
                    st.warning("⚠️ No text chunks to index after splitting.")
            else:
                st.warning("⚠️ No content extracted from uploaded files to index.")
        else:
            st.warning("⚠️ Please upload files first.")

    if st.session_state.docs_indexed_successfully:
        st.sidebar.success(f"📚 Knowledge base '{WEAVIATE_CLASS_NAME}' is ready.")
    elif st.session_state.error_indexing:
        st.sidebar.error("Indexing failed. See error message above.")
    else:
        st.sidebar.info("Upload documents and click 'Process and Index Documents'.")


# --- Part 2: Question Answering ---
st.header("2. Ask a Question")

if not client or not llm or not embeddings_model:
    st.warning("Core components (Weaviate, LLM, or Embeddings) are not initialized. Please check configuration.")
elif not st.session_state.get('docs_indexed_successfully', False):
    st.info("ℹ️ Please upload and successfully index documents using the sidebar before asking questions.")
else:
    user_query = st.text_input("Enter your question about the indexed documents:", key="query_input")

    if user_query:
        with st.spinner("Searching for answers in your documents..."):
            try:
                # Initialize vector store to connect to existing Weaviate class for retrieval
                vector_store = WeaviateVectorStore(
                    client=client,
                    index_name=WEAVIATE_CLASS_NAME,
                    text_key="text",
                    embedding=embeddings_model # Ensure same embedding model is used
                )
                retriever = vector_store.as_retriever(search_kwargs={'k': 3}) # Retrieve top 3 relevant chunks

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True # Set to True to see which chunks were used
                )
                response = qa_chain.invoke({"query": user_query})

                st.subheader("💡 Answer:")
                st.write(response["result"])

                with st.expander("Show Sources (Relevant Chunks)"):
                    for i, source_doc in enumerate(response["source_documents"]):
                        st.markdown(f"**Source {i+1} (from `{source_doc.metadata.get('source', 'N/A')}`)**")
                        st.caption(source_doc.page_content[:500] + "...") # Show partial content

            except Exception as e:
                st.error(f"🔴 Error during Q&A: {e}")
                st.info("This could be an issue with the LLM, Weaviate search, or API limits.")

st.markdown("---")
st.caption("Basic RAG System by an AI")
