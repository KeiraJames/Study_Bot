import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document # To wrap raw text
import google.generativeai as genai # Import the base Google library
import os # For a potential environment variable for API key

# --- Configuration ---
st.set_page_config(page_title="Simple RAG with Gemini", layout="wide")
st.title("📄 Simple RAG with Google Gemini")

# --- Helper function to initialize models (cached) ---
@st.cache_resource
def get_models(api_key):
    st.write("Attempting to initialize Google models...")
    try:
        # 1. Configure the base client for listing models
        genai.configure(api_key=api_key)
        st.write("Google API client configured. Listing available models that support 'generateContent':")

        available_chat_models_from_api = []
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_chat_models_from_api.append(m.name) # This will include "models/" prefix
            if not available_chat_models_from_api:
                st.error("No models supporting 'generateContent' found for your API key via genai.list_models(). Please check your API key permissions and Google Cloud project setup (ensure Generative Language API is enabled).")
                return None, None
            st.info(f"Available models for chat (from genai.list_models()): {available_chat_models_from_api}")
        except Exception as list_e:
            st.error(f"Could not list models using genai.list_models(): {list_e}. This might indicate a more fundamental API key or project setup issue.")
            return None, None

        # 2. Determine which chat model to use for Langchain
        chat_model_name_to_use = None
        preferred_models_for_langchain = [
            "gemini-1.5-pro-latest", # This should be found given your list
            "gemini-pro",
            "gemini-1.0-pro",
            "gemini-1.0-pro-latest",
            "gemini-1.0-pro-001",
        ]

        for preferred_lc_model in preferred_models_for_langchain:
            if f"models/{preferred_lc_model}" in available_chat_models_from_api:
                chat_model_name_to_use = preferred_lc_model
                st.write(f"Found preferred model for Langchain: '{chat_model_name_to_use}'")
                break
        
        if not chat_model_name_to_use and available_chat_models_from_api:
            first_available_from_api = available_chat_models_from_api[0]
            chat_model_name_to_use = first_available_from_api.replace("models/", "")
            st.warning(f"Preferred Gemini models for Langchain not found. Using first available from API list: '{chat_model_name_to_use}' (derived from '{first_available_from_api}')")
        
        if not chat_model_name_to_use:
            st.error("Could not determine a suitable chat model to use. No compatible models found or preferred models are not available.")
            return None, None

        st.write(f"Attempting to use chat model for Langchain: '{chat_model_name_to_use}'")

        # 3. Initialize Langchain components
        embeddings_model_name = "models/embedding-001"
        # Check available embedding models - your list didn't explicitly show embedding models,
        # but genai.list_models() might list them separately or they might just work.
        # 'models/text-embedding-004' is a newer one.
        # If genai.list_models() only shows chat models, we assume embedding models are accessible.
        # A more robust check would list models for 'embedContent' method.
        if "models/text-embedding-004" in available_chat_models_from_api: # Checking against chat models list is not ideal but a proxy
             embeddings_model_name = "models/text-embedding-004"
             st.info(f"Found 'models/text-embedding-004' in general list, attempting to use for embeddings.")
        elif "models/embedding-001" not in available_chat_models_from_api and "models/text-embedding-004" not in available_chat_models_from_api:
             st.warning(f"Neither 'models/embedding-001' nor 'models/text-embedding-004' explicitly found in listed models. Using default '{embeddings_model_name}'. Embeddings might fail if not accessible.")


        embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model_name, google_api_key=api_key)
        llm = ChatGoogleGenerativeAI(model=chat_model_name_to_use, google_api_key=api_key, convert_system_message_to_human=True)

        st.success(f"Embeddings ('{embeddings_model_name}') and LLM ('{chat_model_name_to_use}') initialized successfully!")
        return embeddings, llm

    except Exception as e:
        st.error(f"Error during Google model initialization with Langchain (LLM model tried: '{chat_model_name_to_use if 'chat_model_name_to_use' in locals() else 'unknown'}', Embedding model tried: '{embeddings_model_name if 'embeddings_model_name' in locals() else 'unknown'}'): {e}")
        st.error("This could mean the model selected from the list is still not compatible with Langchain's wrappers or there's another configuration issue. Check the exact error message.")
        return None, None

# --- Main Application ---
google_api_key_env = os.getenv("GOOGLE_API_KEY")
google_api_key = st.text_input(
    "Enter your Google API Key:",
    type="password",
    value=google_api_key_env if google_api_key_env else "",
    help="Required for Gemini LLM and Embeddings. You can also set it as an environment variable GOOGLE_API_KEY."
)

if not google_api_key:
    st.warning("Please enter your Google API Key to proceed.")
    st.stop()

embeddings_model, llm = get_models(google_api_key)

if not embeddings_model or not llm:
    st.error("Model initialization failed. Please check the messages above and your API key.")
    st.stop()

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
elif st.session_state.vector_store:
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
st.caption("A simple RAG implementation using Gemini and FAISS (in-memory).")
