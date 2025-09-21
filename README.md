# 📚 Study Bot – AI-Powered Study Assistant

**Study Bot** is an intelligent study companion that helps students learn efficiently using **RAG (Retrieval-Augmented Generation)** powered by **Google Gemini**. Upload your notes, ask questions, or test your knowledge with custom quizzes tailored to your difficulty level.  

---

## 🚀 Features

- **Upload Notes** – Upload your personal study materials in text or PDF format. Study Bot processes and indexes them for instant retrieval.  
- **Ask Questions** – Query your notes naturally. Study Bot uses RAG to provide accurate, context-aware answers.  
- **Custom Quizzes** – Generate quizzes from your notes based on difficulty:
  - `easy`
  - `medium`
  - `hard`  
- **Gemini-Powered LLM** – Leverages Google Gemini for accurate, context-aware responses.  
- **Interactive Interface** – Communicate with Study Bot through CLI or web interface.  

---

## 🛠️ How It Works

1. **Upload & Chunk Notes**  
   Notes are split into smaller sections (chunks) for easier embedding and retrieval.

2. **Embed Chunks**  
   Each chunk is embedded into a vector store for fast semantic search.

3. **Question Answering**  
   When you ask a question, Study Bot:
   - Embeds your query
   - Retrieves the most relevant chunks using **cosine similarity**
   - Passes the retrieved context to Gemini to generate an accurate response

4. **Quiz Generation**  
   You can request quizzes by difficulty level. Study Bot generates multiple-choice or short-answer questions from your uploaded notes.

---

## 📥 Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/study-bot.git
cd study-bot
