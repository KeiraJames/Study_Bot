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
    page_title="Study Bot",
    layout="wide",
    initial_sidebar_state="expanded"
)


def apply_custom_styles():
 

    custom_css = """
    <style>
        /* Import the 'Lato' font from Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Lato:wght@400;700&display=swap');

        /* Set the sidebar background to the accent color */
        .st-emotion-cache-16txtl3 {
            background-color: #f7b500; /* Accent Yellow/Gold */
        }

        /* Style for the custom gradient dividers */
        hr {
            background: linear-gradient(to right, #4a90e2, #f7b500); /* Blue to Yellow */
            height: 3px !important;
            border: none;
            margin-top: 5px;
            margin-bottom: 25px;
        }

        /* Adjust header colors for better contrast and style on the new background */
        h1, h2 {
            color: #FFFFFF; /* White for high contrast on the dusky blue background */
            text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        }
        h3 {
            color: #F0F8FF; /* AliceBlue (off-white) for sub-headers */
        }

        /* Ensure sidebar buttons are readable on the yellow background */
        .st-emotion-cache-16txtl3 .stButton>button {
            background-color: #1E3A8A; /* Dark Blue */
            color: white;
            border-color: #1E3A8A;
        }
        .st-emotion-cache-16txtl3 .stButton>button:hover {
            background-color: #2563EB;
            border-color: #2563EB;
        }

    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Apply the theme enhancements
apply_custom_styles()

#st.title("📚 Study Bot")


GOOGLE_API_KEY = "AIzaSyDMYArQqF4gjHTVXAVmcwEGwMG4iZDKRh4"

# --- Helper Functions & Model Initialization (No changes in logic) ---
@st.cache_resource
def get_models(api_key):
    
    genai.configure(api_key=api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    llm = ChatGoogleGenerativeAI(
            model='gemini-1.5-flash-latest', 
            google_api_key=api_key,
            convert_system_message_to_human=True, 
            temperature=0.2
    )
    return embeddings, llm
   

def extract_text_from_pdf(pdf_file): 
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
   
    prompt = f"""
    You are an expert educator and quiz designer. Based on the following text, create a multiple-choice quiz.
    **Instructions:**
    1. Generate exactly {num_questions} questions.
    2. The difficulty of the questions should be '{difficulty}'.
    3. For each question, provide 4 options (A, B, C, D).
    4. **Crucially, for each question, provide a detailed 'explanation' that explains why the correct answer is right, directly referencing concepts from the provided text.**
    5. Your entire response must be a single, valid JSON object. The object must have a single key "questions" which is a list of question objects.
    6. Each question object must have these exact keys: "question", "options", "answer", "explanation".
    ---
    **TEXT TO ANALYZE:**
    {context_text}
    ---
    """ 

    response = llm.invoke(prompt)
    quiz_json_string = response.content.replace("```json", "").replace("```", "").strip()
    quiz_data = json.loads(quiz_json_string)
    return quiz_data.get("questions", [])
    

# --- Initialize Models ---
embeddings_model, llm = get_models(GOOGLE_API_KEY)

if 'page' not in st.session_state: st.session_state.page = "RAG Q&A"
if 'text_content' not in st.session_state: st.session_state.text_content = ""
if 'vector_store' not in st.session_state: st.session_state.vector_store = None
if 'quiz_data' not in st.session_state: st.session_state.quiz_data = None
if 'quiz_submitted' not in st.session_state: st.session_state.quiz_submitted = False
if 'user_answers' not in st.session_state: st.session_state.user_answers = {}

# =================================================================================================
# --- SIDEBAR: Navigation and Status ---
# =================================================================================================
st.sidebar.title("Navigation")
st.sidebar.divider()
if st.sidebar.button("RAG Q&A", use_container_width=True):
    st.session_state.page = "RAG Q&A"
if st.sidebar.button("Quiz Me", use_container_width=True):
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

  
    st.subheader("1. Provide Your Document")
    st.divider()
    
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
            else:
                st.warning("Could not process document.", icon="⚠️")
        st.rerun()

    
    st.subheader("2. Ask a Question")
    st.divider()

    if not st.session_state.vector_store:
        st.info("Please provide a document above to enable the Q&A feature.")
    else:
        user_question = st.text_input("Enter your question:", key="user_question", placeholder="e.g., What is the main theme of the document?")
        if st.button("Get Answer", type="primary"):
            if not user_question:
                st.warning("Please enter a question.", icon="❓")
            else:
                with st.spinner(" Searching for answers..."):
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
        st.subheader("Create Your Quiz")
        st.divider()
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
            
            st.subheader("Your Custom Quiz")
            st.divider()

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
                st.divider()
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
                
                st.header(f"Your Final Score: {score}/{len(st.session_state.quiz_data)}")
                st.divider()
                if st.button("Take a New Quiz"):
                    st.session_state.quiz_data = None
                    st.session_state.quiz_submitted = False
                    st.rerun()
