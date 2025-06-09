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

# --- Page Configuration ---
st.set_page_config(page_title="Advanced Study Bot", layout="wide")
st.title("📚 Advanced Study Bot")

# --- Google API Key Setup ---
try:
    GOOGLE_API_KEY = "AIzaSyDMYArQqF4gjHTVXAVmcwEGwMG4iZDKRh4"
except KeyError:
    st.error("Google API Key not found! Please add it to your Streamlit secrets.", icon="🚨")
    st.stop()

# --- Helper Functions & Model Initialization ---

@st.cache_resource
def get_models(api_key):
    """Initializes and caches the LangChain models."""
    try:
        genai.configure(api_key=api_key)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        # Use a slightly lower temperature for more deterministic quiz/answer generation
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
    1.  Generate exactly {num_questions} questions.
    2.  The difficulty of the questions should be '{difficulty}'.
    3.  For each question, provide 4 options (A, B, C, D).
    4.  **Crucially, for each question, provide a detailed 'explanation' that explains why the correct answer is right, directly referencing concepts from the provided text.**
    5.  Your entire response must be a single, valid JSON object. The object must have a single key "questions" which is a list of question objects.
    6.  Each question object must have these exact keys: "question", "options", "answer", "explanation".
    7.  The "options" value must be a dictionary with keys "A", "B", "C", "D".
    8.  The "answer" value must be the letter of the correct option (e.g., "B").

    **Example JSON Format:**
    {{
      "questions": [
        {{
          "question": "What is the primary energy currency of the cell?",
          "options": {{
            "A": "Glucose",
            "B": "ATP",
            "C": "DNA",
            "D": "RNA"
          }},
          "answer": "B",
          "explanation": "The text states that ATP (adenosine triphosphate) is used to store and transfer energy within cells, making it the primary energy currency."
        }}
      ]
    }}

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


# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
st.sidebar.divider()
if st.sidebar.button("RAG Q&A", use_container_width=True, type="primary" if st.session_state.page == "RAG Q&A" else "secondary"):
    st.session_state.page = "RAG Q&A"
if st.sidebar.button("Quiz Me", use_container_width=True, type="primary" if st.session_state.page == "Quiz Me" else "secondary"):
    st.session_state.page = "Quiz Me"

st.sidebar.divider()
st.sidebar.header("Upload Document")
# --- Document Input (remains in sidebar for global access) ---
input_method = st.sidebar.radio(
    "Choose input method:",
    ("Upload a File", "Paste Text"),
    label_visibility="collapsed"
)

document_text = ""
if input_method == "Upload a File":
    uploaded_file = st.sidebar.file_uploader("Upload .pdf or .txt", type=['pdf', 'txt'], label_visibility="collapsed")
    if uploaded_file:
        with st.spinner(f"Reading {uploaded_file.name}..."):
            file_bytes = BytesIO(uploaded_file.getvalue())
            if uploaded_file.type == "application/pdf":
                document_text = extract_text_from_pdf(file_bytes)
            else:
                document_text = file_bytes.read().decode('utf-8')
else:
    document_text = st.sidebar.text_area("Paste your text here:", height=200, key="manual_text")

# --- Process document text if new text is provided ---
if document_text and (document_text != st.session_state.text_content):
    st.session_state.text_content = document_text
    st.session_state.vector_store = None # Clear old vector store
    st.session_state.quiz_data = None # Clear old quiz
    st.session_state.quiz_submitted = False
    with st.spinner("🧠 Processing document..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents([Document(page_content=st.session_state.text_content)])
        if chunks and embeddings_model:
            try:
                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings_model)
                st.sidebar.success("Document processed!", icon="✅")
            except Exception as e:
                st.sidebar.error(f"Error creating vector store: {e}", icon="❌")
        else:
            st.sidebar.warning("Could not process document.", icon="⚠️")

# =================================================================================================
# --- MAIN PAGE CONTENT ---
# =================================================================================================

# --- RAG Q&A PAGE ---
if st.session_state.page == "RAG Q&A":
    st.header("💬 Ask Questions About Your Document")
    st.write("Use this page to ask specific questions and get answers directly from your text.")

    if not st.session_state.vector_store:
        st.info("Please upload or paste a document in the sidebar to get started.")
    else:
        user_question = st.text_input("Enter your question:", key="user_question")
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
        st.info("Please upload or paste a document in the sidebar to get started.")
    else:
        # --- Quiz Generation Controls ---
        st.subheader("Create Your Quiz", divider="rainbow")
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                num_questions = st.number_input("Number of Questions:", min_value=1, max_value=20, value=5)
            with col2:
                difficulty = st.selectbox("Difficulty:", ["Easy", "Medium", "Hard"])

            if st.button("Generate Quiz", type="primary", use_container_width=True):
                with st.spinner("🤖 Generating your quiz... This might take a moment."):
                    quiz_data = generate_quiz_with_explanations(st.session_state.text_content, num_questions, difficulty, llm)
                    st.session_state.quiz_data = quiz_data
                    st.session_state.quiz_submitted = False

        # --- Display Quiz or Results ---
        if st.session_state.quiz_data:
            st.subheader("Your Custom Quiz", divider="rainbow")
            if not st.session_state.quiz_submitted:
                # --- Display the quiz form ---
                with st.form("quiz_form"):
                    user_answers = {}
                    for i, q in enumerate(st.session_state.quiz_data):
                        st.markdown(f"**{i+1}. {q['question']}**")
                        # The options are now the values of the options dictionary
                        options = list(q['options'].values())
                        # The key must be unique for each radio button group
                        user_answers[i] = st.radio("Select an answer:", options, key=f"q_{i}", label_visibility="collapsed")

                    submitted = st.form_submit_button("Submit Answers")
                    if submitted:
                        st.session_state.quiz_submitted = True
                        st.session_state.user_answers = user_answers
                        st.rerun() # Rerun the script to show the results
            else:
                # --- Display the results ---
                st.subheader("📝 Quiz Results")
                score = 0
                total = len(st.session_state.quiz_data)
                for i, q in enumerate(st.session_state.quiz_data):
                    correct_option_key = q['answer']
                    correct_answer_text = q['options'][correct_option_key]
                    user_answer_text = st.session_state.user_answers[i]

                    with st.container(border=True):
                        st.markdown(f"**Question {i+1}:** {q['question']}")
                        if user_answer_text == correct_answer_text:
                            score += 1
                            st.success(f"✔️ You answered: **{user_answer_text}** (Correct!)")
                        else:
                            st.error(f"❌ You answered: **{user_answer_text}** (Incorrect)")
                            st.info(f"Correct answer: **{correct_answer_text}**")
                        
                        st.info(f"**Explanation:** {q['explanation']}")
                
                st.header(f"Your Final Score: {score}/{total}", divider="rainbow")

                if st.button("Take a New Quiz"):
                    st.session_state.quiz_data = None
                    st.session_state.quiz_submitted = False
                    st.rerun()
