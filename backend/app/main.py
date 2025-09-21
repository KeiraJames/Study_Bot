from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from io import BytesIO
from backend.app.services.rag import process_document, answer_question
from backend.app.services.quiz import generate_quiz_with_explanations
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = FastAPI(title="Study Bot API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class QuestionRequest(BaseModel):
    question: str

class QuizRequest(BaseModel):
    difficulty: str
    num_questions: int

# --- Endpoints ---
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode('utf-8') if file.content_type == "text/plain" else process_document(BytesIO(content))
    # Save or cache text in session/db as needed
    return {"text": text}

@app.post("/ask")
def ask_question(req: QuestionRequest):
    answer = answer_question(req.question)
    return {"answer": answer}

@app.post("/quiz")
def create_quiz(req: QuizRequest):
    # Get text content from stored session/cache
    context_text = "..."  # Replace with actual document text
    quiz = generate_quiz_with_explanations(context_text, req.num_questions, req.difficulty)
    return {"quiz": quiz}
