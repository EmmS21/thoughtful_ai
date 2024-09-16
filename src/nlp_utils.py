import json
import os
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()
question_vectors = None
questions = None

def load_predefined_questions(file_path):
    global questions, question_vectors
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    questions = data.get("questions", [])
    question_texts = [qa['question'] for qa in questions]
    question_vectors = vectorizer.fit_transform(question_texts)
    return questions

def preprocess_input(user_input):
    return user_input.lower().strip()

def find_exact_match(user_input, questions):
    for qa in questions:
        if qa['question'].lower() == user_input:
            return qa['answer']
    return None

import json
import os
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

question_embeddings = None
questions = None

def load_predefined_questions(file_path):
    global questions, question_embeddings
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    questions = data.get("questions", [])
    question_texts = [qa['question'] for qa in questions]
    question_embeddings = model.encode(question_texts)
    return questions

def preprocess_input(user_input):
    return user_input.lower().strip()

def find_exact_match(user_input, questions):
    for qa in questions:
        if qa['question'].lower() == user_input:
            return qa['answer']
    return None

def find_semantic_match(user_input, questions, threshold=0.7):
    global question_embeddings
    if question_embeddings is None:
        raise ValueError("Questions have not been loaded and embedded yet.")

    user_embedding = model.encode([user_input])[0]
    similarities = cosine_similarity([user_embedding], question_embeddings)[0]
    max_similarity = np.max(similarities)
    max_index = np.argmax(similarities)

    if max_similarity >= threshold:
        return questions[max_index]['answer']
    else:
        return None