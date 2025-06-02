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
load_dotenv()

GOOGLE_API_KEY = "2b10X3YLMd8PNAuKOCVPt7MeUe"
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
# WEAVIATE_API_KEY related lines are now removed

WEAVIATE_CLASS_NAME = "StreamlitPublicDocsV4NoKey" # Changed class name slightly

# --- INITIALIZE MODELS AND CLIENTS (CACHED) ---
@st.cache_resource
def get_weaviate_client():
    try:
        # No WEAVIATE_API_KEY logic needed, auth_config will effectively be None
        auth_config = None # Explicitly None as no API key is used

        try:
            from weaviate.connect.helpers import ConnectionParams
        except ImportError:
            st.error("Failed to import ConnectionParams from weaviate.connect.helpers. Check weaviate-client version.")
            return None
        
        conn_params = ConnectionParams.from_url(
            url=WEAVIATE_URL,
            # grpc_port=50051 # You might need to explicitly set if not inferred correctly
        )
        
        client_instance = weaviate.WeaviateClient(
            connection_params=conn_params,
            auth_client_secret=auth_config, # This will be None
        )
        
        client_instance.connect()
        if not client_instance.is_connected():
            st.error(f"🔴 Weaviate client is not connected at {WEAVIATE_URL}.")
            st.info("Check if Weaviate is running and accessible.")
            return None
        return client_instance
    except Exception as e:
        st.error(f"🔴 Failed to initialize or connect Weaviate client: {e}")
        st.info(f"Attempted to connect to Weaviate URL: {WEAVIATE_URL}")
        return None

@st.cache_resource
def get_embeddings_model():
    if not GOOGLE_API_KEY:
        return None
    try:
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize Google Embeddings model: {e}")
        return None

@st.cache_resource
def get_llm():
    if not GOOGLE_API_KEY:
        return None
    try:
        return ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)
    except Exception as e:
        st.error(f"Failed to initialize Google Gemini LLM: {e}")
        return None

# --- Initialize core components ---
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
    # Updated error message to reflect no API key is expected for Weaviate
    st.error("🔴 **Setup Required:** Could not connect to Weaviate. Ensure `WEAVIATE_URL` is correct and Weaviate is running and accessible.")
    setup_ok = False
if GOOGLE_API_KEY and not embeddings_model :
    st.error("🔴 **Error:** Failed to initialize Google Embeddings model. Check API key validity and model name.")
    setup_ok = False
if GOOGLE_API_KEY and not llm:
    st.error("🔴 **Error:** Failed to initialize Google Gemini LLM. Check API key validity and model name.")
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
    process_button_disabled = not setup_ok

    if st.button("Process and Index Documents", disabled=process_button_disabled):
        if uploaded_files:
            st.session_state.docs_indexed_successfully = False
            st.session_state.error_indexing = None
            all_docs_content = []
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
                            os.remove(temp_file_path)
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)

            if all_docs_content:
                with st.spinner("Splitting documents into chunks..."):
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    chunks = text_splitter.split_documents(all_docs_content)

                if chunks:
                    with st.spinner(f"Embedding {len(chunks)} chunks and storing in Weaviate..."):
                        try:
                            WeaviateVectorStore.from_documents(
                                client=client,
                                documents=chunks,
                                embedding=embeddings_model,
                                index_name=WEAVIATE_CLASS_NAME,
                                text_key="text"
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

    if st.session_state.docs_indexed_successfully:
        st.sidebar.success(f"📚 Knowledge base '{WEAVIATE_CLASS_NAME}' is ready.")
    elif st.session_state.error_indexing:
        st.sidebar.error("Indexing failed.")
    elif setup_ok :
        st.sidebar.info("Upload documents and click 'Process and Index Documents'.")

# --- Part 2: Question Answering (Main Area) ---
st.header("2. Ask a Question")
qa_disabled = not setup_ok or not st.session_state.get('docs_indexed_successfully', False)

if qa_disabled:
    if not setup_ok:
        st.warning("Please resolve setup issues (API keys, Weaviate connection) shown above.")
    else:
        st.info("ℹ️ Please upload and successfully index documents using the sidebar before asking questions.")
else:
    user_query = st.text_input("Enter your question about the indexed documents:", key="query_input", disabled=qa_disabled)
    if user_query:
        with st.spinner("Searching for answers in your documents..."):
            try:
                vector_store = WeaviateVectorStore(
                    client=client,
                    index_name=WEAVIATE_CLASS_NAME,
                    text_key="text",
                    embedding=embeddings_model
                )
                retriever = vector_store.as_retriever(search_kwargs={'k': 3})
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
                )
                response = qa_chain.invoke({"query": user_query})
                st.subheader("💡 Answer:")
                st.write(response["result"])
                with st.expander("Show Sources (Relevant Chunks Used)"):
                    for i, source_doc in enumerate(response["source_documents"]):
                        st.markdown(f"**Source {i+1} (from `{source_doc.metadata.get('source', 'N/A')}`)**")
                        st.caption(source_doc.page_content[:500] + "...")
            except Exception as e:
                st.error(f"🔴 Error during Q&A: {e}")
                st.info("This could be an issue with the LLM, Weaviate search, or API limits.")

st.markdown("---")
st.caption("Basic RAG System")
