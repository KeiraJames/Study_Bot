import json

def generate_quiz_with_explanations(context_text, num_questions, difficulty, llm=None):
   
    prompt = f""" You are an expert educator and quiz designer. Generate exactly {num_questions} multiple-choice questions
    with difficulty {difficulty} from the following text. Include options A-D and explanations.
    Return as a single JSON object with key "questions". TEXT: {context_text} """
   
    
    quiz_json_string = llm.invoke(prompt).content.replace("```json","").replace("```","").strip()
    return json.loads(quiz_json_string).get("questions", [])
