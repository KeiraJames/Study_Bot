import streamlit as st
import pypdf
from io import BytesIO
import json
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# --- Page & Theme Configuration ---
st.set_page_config(
    page_title="Study Bot Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_custom_theme():
    """Applies a custom blue and yellow theme to the app."""
    # Using st.markdown to inject custom CSS for finer control
    custom_css = """
    <style>
        /* Define theme variables */
        :root {
            --primary-color: #4A90E2;      /* Professional Blue */
            --background-color: #F0F8FF;   /* AliceBlue (very light) */
            --secondary-background-color: #EBF5FF; /* Slightly darker blue */
            --text-color: #0F172A;         /* Slate-900 (dark text) */
            --accent-color: #F7B500;       /* Amber/Gold */
        }

        /* Apply theme to the body */
        body {
            background-color: var(--background-color);
            color: var(--text-color);
        }

        /* Style the main content area */
        .main .block-container {
            background-color: var(--background-color);
        }

        /* Style the sidebar */
        .st-emotion-cache-16txtl3 {
            background-color: var(--secondary-background-color);
        }

        /* Style Streamlit's buttons */
        .stButton>button {
            border-color: var(--primary-color);
            background-color: var(--primary-color);
            color: white;
        }
        .stButton>button:hover {
            border-color: #357ABD;
            background-color: #357ABD;
        }

        /* Style the header dividers with the accent color */
        hr {
            background: linear-gradient(to right, var(--accent-color), var(--primary-color));
            height: 3px !important;
            border: none;
        }

        /* Style headers */
        h1, h2 {
            color: #1E3A8A; /* Darker blue for headers */
        }
        h3 {
            color: #2563EB; /* Primary blue for sub-headers */
        }

        /* Style success and info boxes */
        [data-testid="stSuccess"], [data-testid="stInfo"] {
             border-left: 5px solid var(--accent-color) !important;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Apply the theme at the very top of the script
apply_custom_theme()

st.title("📚 Study Bot Pro")

# --- Google API Key Setup ---
try:
    GOOGLE_API_KEY = "AIzaSyDMYArQqF4gjHTVXAVmcwEGwMG4iZDKRh4"
except KeyError:
    st.error("Google API Key not found! Please add it to your Streamlit secrets.", icon="🚨")
    st.stop()

# --- Helper Functions & Model Initialization (No changes in logic) ---

@st.cache_resource
def get_models(api_key):
    """Initializes and caches the LangChain models."""
    try:
        genai.configure(api_key=api_key)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest', google_api_key=api_key,
                                     convert_system_message_to_human=True, temperature=0.2)
        return embeddings, llm
    except Exception as e:
        st.error(f"Error initializing Google models: {e}", icon="🔥")
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

def generate_quiz_with_explanations(context_text, num_questions, difficulty, llm):
    """Generates a quiz with detailed explanations using the Gemini LLM."""
    if not llm:
        st.error("LLM not initialized. Cannot generate quiz.")
        return None
    prompt = f"""
    You are an expert educator and quiz designer. Based on the following text, create a multiple-choice quiz.
    **Instructions:**
    1. Generate exactly {num_questions} questions.
    2. The difficulty of the questions should be '{difficulty}'.
    3. For each question, provide 4 options (A, B, C, D).
    4. **Crucially, for each question, provide a detailed 'explanation' that explains why the correct answer is right, directly referencing concepts from the provided text.**
    5. Your entire response must be a single, valid JSON object. The object must have a single key "questions" which is a list of question objects.
    6. Each question object must have these exact keys: "question", "options", "answer", "explanation".
    7. The "options" value must be a dictionary with keys "A", "B", "C", "D".
    8. The "answer" value must be the letter of the correct option (e.g., "B").
    ---
    **TEXT TO ANALYZE:**
    {context_text}
    ---
    """
    try:
        response = llm.invoke(prompt)
        quiz_json_string = response.content.replace("```json", "").replace("```", "").strip()
        quiz_data = json.loads(quiz_json_string)
        return quiz_data.get("questions", [])
    except json.JSONDecodeError:
        st.error("AI returned an invalid format. Please try generating the quiz again.", icon="🧩")
        st.write("Raw AI response for debugging:", quiz_json_string)
        return None
    except Exception as e:
        st.error(f"An error occurred while generating the quiz: {e}", icon="💥")
        return None

# --- Initialize Models ---
embeddings_model, llm = get_models(GOOGLE_API_KEY)

# --- Session State Initialization ---
if 'page' not in st.session_state:
    st.session_state.page = "RAG Q&A"
if 'text_content' not in st.session_state:
    st.session_state.text_content = ""
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'quiz_data' not in st.session_state:
    st.session_state.quiz_data = None
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}

# =================================================================================================
# --- SIDEBAR: Navigation and Status ---
# =================================================================================================
st.sidebar.title("Navigation")
st.sidebar.divider()
if st.sidebar.button("RAG Q&A", use_container_width=True, type="secondary" if st.session_state.page != "RAG Q&A" else "primary"):
    st.session_state.page = "RAG Q&A"
if st.sidebar.button("Quiz Me", use_container_width=True, type="secondary" if st.session_state.page != "Quiz Me" else "primary"):
    st.session_state.page = "Quiz Me"

st.sidebar.divider()
st.sidebar.header("Document Status")
if st.session_state.text_content:
    st.sidebar.success("Document Loaded & Ready!")
else:
    st.sidebar.info("No document loaded.")

# =================================================================================================
# --- MAIN PAGE CONTENT ---
# =================================================================================================

# --- RAG Q&A PAGE ---
if st.session_state.page == "RAG Q&A":
    st.header("💬 Ask Questions About Your Document")
    st.write("Start by providing your study material below. Once processed, you can ask questions or switch to the 'Quiz Me' page.")

    st.subheader("1. Provide Your Document", divider="blue")
    input_method = st.radio("Choose input method:", ("Upload a File", "Paste Text"), horizontal=True)
    document_text = ""
    if input_method == "Upload a File":
        uploaded_file = st.file_uploader("Upload a .pdf or .txt file", type=['pdf', 'txt'])
        if uploaded_file:
            with st.spinner(f"Reading {uploaded_file.name}..."):
                file_bytes = BytesIO(uploaded_file.getvalue())
                document_text = extract_text_from_pdf(file_bytes) if uploaded_file.type == "application/pdf" else file_bytes.read().decode('utf-8')
    else:
        document_text = st.text_area("Paste your document text here:", height=300, key="manual_text")

    if document_text and (document_text != st.session_state.text_content):
        st.session_state.text_content = document_text
        st.session_state.vector_store = None
        st.session_state.quiz_data = None
        st.session_state.quiz_submitted = False
        with st.spinner("🧠 Processing document..."):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_documents([Document(page_content=st.session_state.text_content)])
            if chunks and embeddings_model:
                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings_model)
                st.success("Document processed successfully!", icon="✅")
            else:
                st.warning("Could not process document.", icon="⚠️")
        st.rerun()

    st.subheader("2. Ask a Question", divider="blue")
    if not st.session_state.vector_store:
        st.info("Please provide a document above to enable the Q&A feature.")
    else:
        user_question = st.text_input("Enter your question:", key="user_question", placeholder="e.g., What is the main theme of the document?")
        if st.button("Get Answer", type="primary"):
            if not user_question:
                st.warning("Please enter a question.", icon="❓")
            else:
                with st.spinner("🔍 Searching for answers..."):
                    retriever = st.session_state.vector_store.as_retriever()
                    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
                    response = qa_chain.invoke({"query": user_question})
                    st.subheader("💡 Answer:")
                    st.write(response["result"])
                    with st.expander("Show Retrieved Context (Source Chunks)"):
                        for i, doc in enumerate(response["source_documents"]):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.caption(doc.page_content)

# --- QUIZ ME PAGE ---
elif st.session_state.page == "Quiz Me":
    st.header("🧠 Quiz Yourself on the Content")
    st.write("Generate a quiz to test your understanding of the key concepts in your document.")

    if not st.session_state.text_content:
        st.info("Please go to the 'RAG Q&A' page to upload a document first.")
    else:
        st.subheader("Create Your Quiz", divider="blue")
        with st.container(border=True):
            col1, col2 = st.columns(2)
            num_questions = col1.number_input("Number of Questions:", min_value=1, max_value=20, value=5)
            difficulty = col2.selectbox("Difficulty:", ["Easy", "Medium", "Hard"])
            if st.button("Generate Quiz", type="primary", use_container_width=True):
                with st.spinner("🤖 Generating your quiz... This might take a moment."):
                    quiz_data = generate_quiz_with_explanations(st.session_state.text_content, num_questions, difficulty, llm)
                    st.session_state.quiz_data = quiz_data
                    st.session_state.quiz_submitted = False
                st.rerun()

        if st.session_state.quiz_data:
            st.subheader("Your Custom Quiz", divider="blue")
            if not st.session_state.quiz_submitted:
                with st.form("quiz_form"):
                    user_answers = {}
                    for i, q in enumerate(st.session_state.quiz_data):
                        st.markdown(f"**{i+1}. {q['question']}**")
                        options = list(q['options'].values())
                        user_answers[i] = st.radio("Select an answer:", options, key=f"q_{i}", label_visibility="collapsed")
                    if st.form_submit_button("Submit Answers"):
                        st.session_state.quiz_submitted = True
                        st.session_state.user_answers = user_answers
                        st.rerun()
            else:
                st.subheader("📝 Quiz Results")
                score = 0
                for i, q in enumerate(st.session_state.quiz_data):
                    with st.container(border=True):
                        st.markdown(f"**Question {i+1}:** {q['question']}")
                        correct_answer_text = q['options'][q['answer']]
                        user_answer_text = st.session_state.user_answers[i]
                        if user_answer_text == correct_answer_text:
                            score += 1
                            st.success(f"✔️ You answered: **{user_answer_text}** (Correct!)")
                        else:
                            st.error(f"❌ You answered: **{user_answer_text}** (Incorrect)")
                            st.info(f"Correct answer: **{correct_answer_text}**")
                        st.info(f"**Explanation:** {q['explanation']}")
                
                st.header(f"Your Final Score: {score}/{len(st.session_state.quiz_data)}", divider="blue")
                if st.button("Take a New Quiz"):
                    st.session_state.quiz_data = None
                    st.session_state.quiz_submitted = False
                    st.rerun()
