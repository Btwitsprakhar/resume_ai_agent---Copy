import os
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# For actual PDF/DOCX parsing, you'd need libraries like PyPDF2, python-docx, or Tesseract for OCR.
# These are not included here as they require installation and specific file handling.
# For demonstration, we'll assume text is already extracted and sent from frontend.

# Example of installing libraries if running locally:
# pip install Flask Flask-Cors requests spacy chromadb
# python -m spacy download en_core_web_sm
# pip install sentence-transformers # For real embeddings

# --- Configuration ---
# IMPORTANT: Replace with your actual Gemini API key if running outside the Canvas environment
# In a production environment, use environment variables for keys: os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = "AIzaSyBSGPyE2ddGGVVkylH_dIItzloP-H9ydA8" 
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allowing frontend to access

# --- 1. Text Extraction (Conceptual/Placeholder) ---
# In a real backend, you'd integrate libraries here to process uploaded files.
# For this Flask app, we expect the frontend to send the extracted text directly.
def extract_text_from_file_conceptual(file_content_bytes: bytes) -> str:
    """
    Conceptual function to extract text from file content.
    In a real application, this would parse PDF/DOCX bytes.
    For this demo, we'll just return a placeholder or process simple text.
    """
    # This function would be more complex to handle actual file types.
    # For now, we assume the frontend sends the plain text.
    return file_content_bytes.decode('utf-8') # Simple decode if sending text as bytes

# --- 2. Preprocessing and Embedding ---
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except ImportError:
    print("spaCy not found. Please install it: pip install spacy && python -m spacy download en_core_web_sm")
    nlp = None
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Downloading...")
    try:
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"Error downloading spaCy model: {e}. Preprocessing will be basic.")
        nlp = None

def preprocess_text(text: str) -> str:
    """
    Performs tokenization, stopword removal, and lemmatization.
    """
    if nlp is None:
        return text.lower() # Fallback to simple lowercasing if spaCy is not available

    doc = nlp(text)
    processed_tokens = [
        token.lemma_.lower() for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    return " ".join(processed_tokens)

def get_text_embedding(text: str) -> list[float]:
    """
    Conceptual function to get embeddings for text.
    In a real application, this would call an embedding model API (e.g., Google's embedding models, OpenAI, Cohere)
    or use a local model (e.g., Sentence-Transformers).
    For demonstration, returns a dummy embedding.
    """
    # For a real embedding, you'd integrate with a model. Example with Sentence-Transformers:
    # from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # embedding = model.encode(text).tolist()
    # return embedding
    # print(f"Generating dummy embedding for: '{text[:50]}...'")
    return [0.0] * 768 # Placeholder: replace with actual embedding logic

# --- 3. Vector Store Creation (Conceptual ChromaDB) ---
try:
    import chromadb
    client = chromadb.Client() # In-memory client for demo. For persistence: chromadb.PersistentClient(path="/path/to/db")
    collection_name = "resume_embeddings"
    try:
        collection = client.get_or_create_collection(name=collection_name)
    except Exception as e:
        print(f"Error getting/creating ChromaDB collection: {e}")
        collection = None
except ImportError:
    print("ChromaDB not found. Please install it: pip install chromadb")
    client = None
    collection = None

def add_resume_to_vector_store(resume_id: str, resume_text: str, embedding: list[float]):
    """Adds a resume and its embedding to the vector store."""
    if collection:
        try:
            collection.add(
                documents=[resume_text],
                embeddings=[embedding],
                metadatas=[{"source": "resume", "id": resume_id}],
                ids=[resume_id]
            )
            print(f"Resume {resume_id} added to vector store.")
        except Exception as e:
            print(f"Error adding resume to vector store: {e}")
    else:
        print("ChromaDB collection not initialized. Cannot add resume.")

def search_similar_resumes(query_embedding: list[float], n_results: int = 1):
    """Searches for similar resumes in the vector store."""
    if collection:
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return None
    else:
        print("ChromaDB collection not initialized. Cannot search.")
        return None

# --- 4. LLM Service (Core AI Logic) ---
def call_gemini_api(prompt: str, response_schema: dict = None) -> str | dict | None:
    """
    Makes a call to the Gemini API with the given prompt and optional response schema.
    """
    headers = {
        'Content-Type': 'application/json',
    }
    params = {'key': GEMINI_API_KEY} if GEMINI_API_KEY else {}

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }
    if response_schema:
        payload["generationConfig"] = {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=payload)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        result = response.json()

        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            text_response = result["candidates"][0]["content"]["parts"][0]["text"]
            if response_schema:
                return json.loads(text_response)
            return text_response
        else:
            print("Unexpected API response structure or no content:", result)
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from Gemini API: {e}")
        print(f"Raw response: {response.text}")
        return None

# --- Flask API Endpoints ---
def analyze_resume_backend(resume_text, job_description):
    prompt = (
        f"Analyze the following resume for the given job description. "
        f"Resume:\n{resume_text}\n\nJob Description:\n{job_description}\n\n"
        f"Provide a detailed analysis of how well the resume matches the job, "
        f"including strengths, weaknesses, and suggestions for improvement."
    )
    result = call_gemini_api(prompt)
    return result or "Could not get analysis from AI."

def chat_with_resume_backend(resume_text, question):
    prompt = (
        f"You are an AI career assistant. Given the following resume:\n{resume_text}\n\n"
        f"Answer this question about the resume: {question}\n"
        f"Be concise and helpful."
    )
    result = call_gemini_api(prompt)
    return result or "Could not get response from AI."

def generate_interview_questions_backend(resume_text, question_type, question_category):
    prompt = (
        f"Given the following resume:\n{resume_text}\n\n"
        f"Generate 5 {question_type} {question_category} interview questions relevant to this candidate's background. "
        f"Return the questions as a numbered list."
    )
    result = call_gemini_api(prompt)
    # Try to split the result into a list if it's a string
    if isinstance(result, str):
        questions = [q.strip(" .") for q in result.split('\n') if q.strip()]
        return questions
    return result or ["Could not get questions from AI."]

def improve_resume_backend(resume_text):
    prompt = (
        f"Review the following resume and provide 3 specific suggestions to improve it:\n{resume_text}\n\n"
        f"Return the suggestions as a list."
    )
    result = call_gemini_api(prompt)
    if isinstance(result, str):
        suggestions = [s.strip(" .") for s in result.split('\n') if s.strip()]
        return suggestions
    return result or ["Could not get suggestions from AI."]

def generate_tailored_resume_backend(original_resume, job_description):
    prompt = (
        f"Given the following resume:\n{original_resume}\n\n"
        f"And this job description:\n{job_description}\n\n"
        f"Rewrite the resume to better match the job description. "
        f"Keep the format professional and highlight relevant skills and experience."
    )
    result = call_gemini_api(prompt)
    return result or "Could not generate tailored resume from AI."

@app.route('/analyze_resume', methods=['POST'])
def analyze_resume_route():
    data = request.get_json()
    resume_text = data.get('resumeText')
    job_description = data.get('jobDescription')

    if not resume_text or not job_description:
        return jsonify({"error": "Resume text and job description are required."}), 400

    analysis = analyze_resume_backend(resume_text, job_description)
    if analysis:
        return jsonify({"result": analysis})
    return jsonify({"error": "Failed to analyze resume."}), 500

@app.route('/chat_with_resume', methods=['POST'])
def chat_with_resume_route():
    data = request.get_json()
    resume_text = data.get('resumeText')
    question = data.get('question')

    if not resume_text or not question:
        return jsonify({"error": "Resume text and question are required."}), 400

    chat_response = chat_with_resume_backend(resume_text, question)
    if chat_response:
        return jsonify({"response": chat_response})
    return jsonify({"error": "Failed to chat with resume."}), 500

@app.route('/generate_interview_questions', methods=['POST'])
def generate_interview_questions_route():
    data = request.get_json()
    resume_text = data.get('resumeText')
    question_type = data.get('questionType', 'easy')
    question_category = data.get('questionCategory', 'technical')

    if not resume_text:
        return jsonify({"error": "Resume text is required."}), 400

    questions = generate_interview_questions_backend(resume_text, question_type, question_category)
    if questions is not None:
        return jsonify({"questions": questions})
    return jsonify({"error": "Failed to generate interview questions."}), 500

@app.route('/improve_resume', methods=['POST'])
def improve_resume_route():
    data = request.get_json()
    resume_text = data.get('resumeText')

    if not resume_text:
        return jsonify({"error": "Resume text is required."}), 400

    improvements = improve_resume_backend(resume_text)
    if improvements:
        return jsonify({"suggestions": improvements})
    return jsonify({"error": "Failed to get improvement suggestions."}), 500

@app.route('/generate_tailored_resume', methods=['POST'])
def generate_tailored_resume_route():
    data = request.get_json()
    original_resume = data.get('originalResume')
    job_description = data.get('jobDescription')

    if not original_resume or not job_description:
        return jsonify({"error": "Original resume and job description are required."}), 400

    tailored_resume = generate_tailored_resume_backend(original_resume, job_description)
    if tailored_resume:
        return jsonify({"tailoredResume": tailored_resume})
    return jsonify({"error": "Failed to generate tailored resume."}), 500

# You can also add routes for adding/searching embeddings if needed
# @app.route('/add_embedding', methods=['POST'])
# def add_embedding_route():
#     data = request.get_json()
#     resume_id = data.get('resumeId')
#     resume_text = data.get('resumeText')
#     if not resume_id or not resume_text:
#         return jsonify({"error": "Resume ID and text are required."}), 400
#     preprocessed_text = preprocess_text(resume_text)
#     embedding = get_text_embedding(preprocessed_text)
#     add_resume_to_vector_store(resume_id, resume_text, embedding)
#     return jsonify({"message": "Resume added to vector store."}), 200

# @app.route('/search_embeddings', methods=['POST'])
# def search_embeddings_route():
#     data = request.get_json()
#     query_text = data.get('queryText')
#     if not query_text:
#         return jsonify({"error": "Query text is required."}), 400
#     query_embedding = get_text_embedding(preprocess_text(query_text))
#     results = search_similar_resumes(query_embedding)
#     return jsonify({"results": results}), 200


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5000) # Run on port 5000