import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import google.generativeai as genai # Import the base Google library

# --- Configuration ---
# (Keep your st.set_page_config and st.title here)

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
        # Langchain often expects model names *without* the "models/" prefix for ChatGoogleGenerativeAI
        chat_model_name_to_use = None
        preferred_models_for_langchain = [
            "gemini-1.5-pro-latest", # Try newest first
            "gemini-pro",            # Common default
            "gemini-1.0-pro",        # Specific version
            "gemini-1.0-pro-latest",
            "gemini-1.0-pro-001",
        ]

        # Check if any of our preferred models (without prefix) are available (by checking against API list with prefix)
        for preferred_lc_model in preferred_models_for_langchain:
            if f"models/{preferred_lc_model}" in available_chat_models_from_api:
                chat_model_name_to_use = preferred_lc_model
                st.write(f"Found preferred model for Langchain: '{chat_model_name_to_use}'")
                break
        
        if not chat_model_name_to_use and available_chat_models_from_api:
            # If none of the preferred are found, take the first from the API list and adapt it
            first_available_from_api = available_chat_models_from_api[0]
            chat_model_name_to_use = first_available_from_api.replace("models/", "")
            st.warning(f"Preferred Gemini models not found for Langchain. Using first available from API list: '{chat_model_name_to_use}' (derived from '{first_available_from_api}')")
        
        if not chat_model_name_to_use:
            st.error("Could not determine a suitable chat model to use. No compatible models found.")
            return None, None

        st.write(f"Attempting to use chat model for Langchain: '{chat_model_name_to_use}'")

        # 3. Initialize Langchain components
        embeddings_model_name = "models/embedding-001" # Usually stable
        embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model_name, google_api_key=api_key)
        llm = ChatGoogleGenerativeAI(model=chat_model_name_to_use, google_api_key=api_key, convert_system_message_to_human=True)

        st.success(f"Embeddings ({embeddings_model_name}) and LLM ('{chat_model_name_to_use}') initialized successfully!")
        return embeddings, llm

    except Exception as e:
        # Catching if Langchain itself fails with the selected model
        st.error(f"Error during Google model initialization with Langchain (model: '{chat_model_name_to_use if 'chat_model_name_to_use' in locals() else 'unknown'}'): {e}")
        st.error("This could mean the model selected from the list is still not compatible with Langchain's ChatGoogleGenerativeAI or there's another configuration issue.")
        return None, None

# --- Main Application ---
# (The rest of your Streamlit app code from your previous version goes here)
# Make sure to include:
# st.set_page_config(...)
# st.title(...)
# google_api_key = st.text_input(...)
# etc.
