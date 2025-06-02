import streamlit as st
import os
import weaviate # Main weaviate import
from dotenv import load_dotenv # For local testing with .env file

# Langchain components
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# --- PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Doc Q&A with Weaviate & Gemini", layout="wide")

# --- CONFIGURATION ---
# Load .env file if it exists (for local development)
load_dotenv()

GOOGLE_API_KEY = "2b10X3YLMd8PNAuKOCVPt7MeUe"
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080") # Default to local if not set
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY") # If your Weaviate instance needs an API key

WEAVIATE_CLASS_NAME = "StreamlitPublicDocsV4" # Changed name to avoid conflict with old schemas

# --- INITIALIZE MODELS AND CLIENTS (CACHED) ---
@st.cache_resource # Caches the Weaviate client connection
def get_weaviate_client():
    try:
        auth_config = None
        if WEAVIATE_API_KEY:
            from weaviate.auth import AuthApiKey # Specific import for v4
            auth_config = AuthApiKey(api_key=WEAVIATE_API_KEY)

        from weaviate.classes.init import ConnectionParams # Specific import for v4

        # Construct connection parameters
        # For local Docker: WEAVIATE_URL="http://localhost:8080"
        # For WCS: WEAVIATE_URL="https://your-cluster-name.weaviate.network"
        # gRPC port is often 50051 by default. from_url tries to infer.
        conn_params = ConnectionParams.from_url(
            url=WEAVIATE_URL,
            # grpc_port=50051 # You might need to explicitly set this if inference fails
        )
        
        client_instance = weaviate.WeaviateClient(
            connection_params=conn_params,
            auth_client_secret=auth_config,
            # additional_headers={"X-Palm-Api-Key": GOOGLE_API_KEY} # Only if Weaviate itself uses Google models
        )
        
        client_instance.connect() # Explicitly connect
        if not client_instance.is_connected():
            st.error(f"🔴 Weaviate client is not connected at {WEAVIATE_URL}. Check URL and API key if used.")
            # client_instance.close() # Close if not connected and returning None
            return None
        # st.sidebar.success("✅ Weaviate client connected.") # For debugging
        return client_instance # Return the connected client
    except Exception as e:
        st.error(f"🔴 Failed to initialize or connect Weaviate client: {e}")
        st.info(f"Attempted to connect to Weaviate URL: {WEAVIATE_URL}")
        return None

@st.cache_resource # Caches the embedding model
def get_embeddings_model():
    if not GOOGLE_API_KEY:
        # This error will be shown in the main app area if key is missing
        return None
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

@st.cache_resource # Caches the LLM
def get_llm():
    if not GOOGLE_API_KEY:
        # This error will be shown in the main app area if key is missing
        return None
    return ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)

# --- Initialize core components ---
# These will run once and be cached, or rerun if their underlying functions change (not expected here)
client = get_weaviate_client()
embeddings_model = get_embeddings_model()
llm = get_llm()

# --- STREAMLIT UI ---
st.title("📄 Ask Questions About Your Documents")
st.markdown("Upload PDF or TXT files, index them, and then ask questions using Google Gemini.")

# --- Display Setup Status / Errors ---
setup_ok = True
if not GOOGLE_API_KEY:
    st.error("🔴 **Setup Required:** `GOOGLE_API_KEY` is not set. Please configure it in your environment or Streamlit secrets.")
    setup_ok = False
if not client:
    st.error("🔴 **Setup Required:** Could not connect to Weaviate. Please ensure `WEAVIATE_URL` (and `WEAVIATE_API_KEY` if needed) are correctly set and Weaviate is accessible.")
    setup_ok = False
if not embeddings_model and GOOGLE_API_KEY: # Check if model init failed despite key
    st.error("🔴 **Error:** Failed to initialize Google Embeddings model. Check API key and model name.")
    setup_ok = False
if not llm and GOOGLE_API_KEY: # Check if LLM init failed despite key
    st.error("🔴 **Error:** Failed to initialize Google Gemini LLM. Check API key and model name.")
    setup_ok = False


# Initialize session state for tracking indexing
if 'docs_indexed_successfully' not in st.session_state:
    st.session_state.docs_indexed_successfully = False
if 'error_indexing' not in st.session_state:
    st.session_state.error_indexing = None


# --- Part 1: Document Upload and Indexing (Sidebar) ---
with st.sidebar:
    st.header("1. Document Processing")
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Files are processed in memory and temporarily stored for indexing."
    )

    # Disable button if core components are not ready
    process_button_disabled = not setup_ok

    if st.button("Process and Index Documents", disabled=process_button_disabled):
        if uploaded_files:
            st.session_state.docs_indexed_successfully = False # Reset status
            st.session_state.error_indexing = None
            all_docs_content = []
            # Using a temporary directory for robust file handling with Langchain loaders
            temp_dir = "temp_uploaded_files_streamlit"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            with st.spinner("Extracting text from files..."):
                for uploaded_file in uploaded_files:
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    try:
                        if uploaded_file.name.endswith(".pdf"):
                            loader = PyPDFLoader(temp_file_path)
                        elif uploaded_file.name.endswith(".txt"):
                            loader = TextLoader(temp_file_path, encoding='utf-8')
                        all_docs_content.extend(loader.load())
                    except Exception as e:
                        st.warning(f"Could not process {uploaded_file.name}: {e}")
                    finally:
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path) # Clean up temp file
                # Clean up temp directory if it's empty
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)

            if all_docs_content:
                with st.spinner("Splitting documents into chunks..."):
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    chunks = text_splitter.split_documents(all_docs_content)

                if chunks:
                    with st.spinner(f"Embedding {len(chunks)} chunks and storing in Weaviate..."):
                        try:
                            # For Weaviate v4, Langchain's WeaviateVectorStore should handle schema creation
                            # if the class doesn't exist, or append if it does.
                            # Ensure your Weaviate instance is configured for custom vectors if not using its internal vectorizers.
                            WeaviateVectorStore.from_documents(
                                client=client, # Pass the WeaviateClient (v4) object
                                documents=chunks,
                                embedding=embeddings_model,
                                index_name=WEAVIATE_CLASS_NAME, # This is the Weaviate Class name
                                text_key="text" # Default property name for text
                            )
                            st.session_state.docs_indexed_successfully = True
                            st.success(f"✅ Successfully indexed {len(chunks)} chunks into Weaviate class '{WEAVIATE_CLASS_NAME}'.")
                        except Exception as e:
                            st.session_state.error_indexing = f"🔴 Error indexing documents in Weaviate: {e}"
                            st.error(st.session_state.error_indexing)
                            st.info("Hints: Schema mismatch if class exists with different config? Weaviate running? API limits?")
                else:
                    st.warning("⚠️ No text chunks to index after splitting.")
            else:
                st.warning("⚠️ No content extracted from uploaded files to index.")
        else:
            st.warning("⚠️ Please upload files first.")

    # Display indexing status in sidebar
    if st.session_state.docs_indexed_successfully:
        st.sidebar.success(f"📚 Knowledge base '{WEAVIATE_CLASS_NAME}' is ready.")
    elif st.session_state.error_indexing:
        st.sidebar.error("Indexing failed.") # Detailed error shown above button
    elif setup_ok : # Only show this info if setup is ok but not indexed yet
        st.sidebar.info("Upload documents and click 'Process and Index Documents'.")


# --- Part 2: Question Answering (Main Area) ---
st.header("2. Ask a Question")

# Disable Q&A if setup is not ok or documents not indexed
qa_disabled = not setup_ok or not st.session_state.get('docs_indexed_successfully', False)

if qa_disabled:
    if not setup_ok:
        st.warning("Please resolve setup issues (API keys, Weaviate connection) shown above.")
    else: # Setup is OK, but docs not indexed
        st.info("ℹ️ Please upload and successfully index documents using the sidebar before asking questions.")
else:
    user_query = st.text_input("Enter your question about the indexed documents:", key="query_input", disabled=qa_disabled)

    if user_query:
        with st.spinner("Searching for answers in your documents..."):
            try:
                # Initialize vector store to connect to existing Weaviate class for retrieval
                vector_store = WeaviateVectorStore(
                    client=client, # Pass the WeaviateClient (v4) object
                    index_name=WEAVIATE_CLASS_NAME,
                    text_key="text",
                    embedding=embeddings_model # Crucial: use the same embedding model for retrieval
                )
                retriever = vector_store.as_retriever(search_kwargs={'k': 3})

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff", # Simplest method
                    retriever=retriever,
                    return_source_documents=True # Good for debugging and transparency
                )
                response = qa_chain.invoke({"query": user_query})

                st.subheader("💡 Answer:")
                st.write(response["result"])

                with st.expander("Show Sources (Relevant Chunks Used)"):
                    for i, source_doc in enumerate(response["source_documents"]):
                        st.markdown(f"**Source {i+1} (from `{source_doc.metadata.get('source', 'N/A')}`)**")
                        st.caption(source_doc.page_content[:500] + "...") # Show partial content
            except Exception as e:
                st.error(f"🔴 Error during Q&A: {e}")
                st.info("This could be an issue with the LLM, Weaviate search, or API limits.")

st.markdown("---")
st.caption("Basic RAG System")
