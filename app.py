import streamlit as st
import os
import pypdf
from io import BytesIO
import json
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# --- Page Configuration ---
st.set_page_config(page_title="Study Bot with Quizzer", layout="wide")
st.title("📚 Study Bot: Q&A and Quizzes from Your Docs")

# --- Google API Key Setup ---
# Use Streamlit's secrets management for the API key
try:
    GOOGLE_API_KEY = "AIzaSyDMYArQqF4gjHTVXAVmcwEGwMG4iZDKRh4"
except KeyError:
    st.error("Google API Key not found! Please add it to your Streamlit secrets.", icon="🚨")
    st.stop()

# --- Helper Functions ---

@st.cache_resource
def get_models(api_key):
    """Initializes and caches the LangChain models."""
    try:
        genai.configure(api_key=api_key)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest', google_api_key=api_key,
                                     convert_system_message_to_human=True, temperature=0.4)
        return embeddings, llm
    except Exception as e:
        st.error(f"Error initializing Google models: {e}", icon="🔥")
        st.info("This could be due to an invalid API key, network issues, or model access restrictions.")
        return None, None

def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    try:
        pdf_reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def generate_quiz_with_gemini(context_text, num_questions, llm):
    """Generates a quiz using the provided Gemini LLM."""
    if not llm:
        st.error("LLM not initialized. Cannot generate quiz.")
        return None

    # A robust prompt telling the model to generate JSON
    prompt = f"""
    You are an expert quiz maker. Based on the following text, create a multiple-choice quiz with {num_questions} questions.
    The goal of the quiz is to test understanding of the key concepts in the text.

    Please format your entire response as a single, valid JSON object.
    The object should have a single key "questions" which is a list of question objects.
    Each question object must have the following keys: "question", "options", "answer".
    The "options" value should be a dictionary with four keys: "A", "B", "C", "D".
    The "answer" value should be the letter of the correct option (e.g., "A").

    Example Format:
    {{
      "questions": [
        {{
          "question": "What is the primary function of a mitochondria?",
          "options": {{
            "A": "Protein synthesis",
            "B": "Energy production",
            "C": "Waste disposal",
            "D": "Cellular movement"
          }},
          "answer": "B"
        }}
      ]
    }}

    ---
    TEXT TO ANALYZE:
    {context_text}
    ---
    """
    try:
        # Use the LangChain LLM wrapper to invoke the model
        response = llm.invoke(prompt)
        # Clean up the response content in case of markdown formatting
        quiz_json_string = response.content.replace("```json", "").replace("```", "").strip()
        quiz_data = json.loads(quiz_json_string)
        return quiz_data.get("questions", [])
    except json.JSONDecodeError:
        st.error("Failed to parse the quiz from the AI's response. The format might be incorrect. Please try again.", icon="🧩")
        st.write("Raw AI response for debugging:", quiz_json_string)
        return None
    except Exception as e:
        st.error(f"An error occurred while generating the quiz: {e}", icon="💥")
        return None

# --- Initialize Models ---
embeddings_model, llm = get_models(GOOGLE_API_KEY)

# --- Session State Initialization ---
if 'text_content' not in st.session_state:
    st.session_state.text_content = ""
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'quiz_data' not in st.session_state:
    st.session_state.quiz_data = None
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False

# --- Main Page UI ---

# --- Part 1: Document Input ---
st.subheader("1. Provide Your Study Material", divider='rainbow')

input_method = st.radio(
    "Choose your input method:",
    ("Upload a File (.pdf or .txt)", "Manually Type/Paste Text"),
    horizontal=True,
    label_visibility="collapsed"
)

document_text = ""
if input_method == "Upload a File (.pdf or .txt)":
    uploaded_file = st.file_uploader("Upload your document", type=['pdf', 'txt'], label_visibility="collapsed")
    if uploaded_file:
        with st.spinner(f"Reading {uploaded_file.name}..."):
            file_bytes = BytesIO(uploaded_file.getvalue())
            if uploaded_file.type == "application/pdf":
                document_text = extract_text_from_pdf(file_bytes)
            else:
                document_text = file_bytes.read().decode('utf-8')
else:
    document_text = st.text_area("Paste your document text here:", height=250, key="manual_text")

# --- Part 2: Text Processing and Vector Store Creation ---
# This block runs if new text is provided and creates the vector store
if document_text and (document_text != st.session_state.text_content):
    st.session_state.text_content = document_text # Update session state with the new text
    with st.spinner("🧠 Processing document and building knowledge base..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs_for_splitting = [Document(page_content=st.session_state.text_content)]
        chunks = text_splitter.split_documents(docs_for_splitting)

        if chunks and embeddings_model:
            try:
                # Create and store the FAISS vector store in session state
                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings_model)
                st.success("Document processed! You can now ask questions or generate a quiz.", icon="✅")
                with st.expander("View Processed Text"):
                    st.write(st.session_state.text_content)
            except Exception as e:
                st.error(f"Error creating vector store: {e}", icon="❌")
                st.session_state.vector_store = None
        elif not embeddings_model:
            st.warning("Embeddings model not available. Cannot process document.", icon="⚠️")
        else:
            st.warning("No text chunks found to process.", icon="⚠️")
            st.session_state.vector_store = None

# --- Part 3: Q&A Section (Your Original RAG Logic) ---
st.subheader("2. Ask Questions About Your Document", divider='rainbow')

if not st.session_state.vector_store:
    st.info("Please provide a document above to enable the Q&A feature.")
else:
    user_question = st.text_input("Enter your question:", key="user_question")
    if st.button("Get Answer", type="primary"):
        if not user_question:
            st.warning("Please enter a question.", icon="❓")
        else:
            with st.spinner("🔍 Searching for answers..."):
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
                    st.error(f"Error during Q&A: {e}", icon="🔥")

# --- Sidebar for Quiz Generation ---
st.sidebar.title("🧠 Quiz Me!")

if not st.session_state.text_content:
    st.sidebar.info("Upload or paste a document in the main window to get started.")
else:
    st.sidebar.success("Content loaded! Ready for a quiz.")
    num_questions = st.sidebar.slider("Number of questions:", min_value=1, max_value=10, value=5)

    if st.sidebar.button("Generate Quiz", use_container_width=True):
        with st.spinner("🤖 Generating your quiz..."):
            quiz_data = generate_quiz_with_gemini(st.session_state.text_content, num_questions, llm)
            st.session_state.quiz_data = quiz_data
            st.session_state.quiz_submitted = False # Reset submission state

        if st.session_state.quiz_data:
            st.sidebar.success("Quiz generated!")
        else:
            st.sidebar.error("Could not generate quiz.")

# Display and handle the quiz if it exists
if st.session_state.quiz_data:
    with st.sidebar.form("quiz_form"):
        st.sidebar.subheader("Answer the Questions:", divider="gray")
        user_answers = {}
        for i, q in enumerate(st.session_state.quiz_data):
            st.sidebar.write(f"**{i+1}. {q['question']}**")
            options = list(q['options'].values())
            user_answers[i] = st.radio(
                "Choose your answer:", options, key=f"q_{i}", label_visibility="collapsed"
            )

        submitted = st.form_submit_button("Submit Answers")
        if submitted:
            st.session_state.quiz_submitted = True
            st.session_state.user_answers = user_answers

# Grade the quiz after submission
if st.session_state.get('quiz_submitted', False):
    st.sidebar.subheader("📝 Results", divider="gray")
    score = 0
    total = len(st.session_state.quiz_data)

    for i, q in enumerate(st.session_state.quiz_data):
        correct_option_key = q['answer']
        correct_answer_text = q['options'][correct_option_key]
        user_answer_text = st.session_state.user_answers[i]

        if user_answer_text == correct_answer_text:
            score += 1
            st.sidebar.success(f"**Q{i+1}: Correct!**", icon="✔️")
        else:
            st.sidebar.error(f"**Q{i+1}: Incorrect.** Your answer: '{user_answer_text}'. Correct: '{correct_answer_text}'.", icon="❌")

    st.sidebar.header(f"Your Final Score: {score} / {total}")
    if st.sidebar.button("Clear Quiz & Try Again", use_container_width=True):
        st.session_state.quiz_data = None
        st.session_state.quiz_submitted = False
        st.rerun()
