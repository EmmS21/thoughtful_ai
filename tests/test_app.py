import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
from nlp_utils import load_predefined_questions, preprocess_input, find_exact_match, find_semantic_match
from llm_integration import get_llm_response
from config import SIMILARITY_THRESHOLD

# Load predefined questions
with open(os.path.join(os.path.dirname(__file__), '../data/predefined_questions.json'), 'r') as f:
    predefined_data = json.load(f)
    PREDEFINED_QUESTIONS = predefined_data['questions']

# Load questions
questions = load_predefined_questions(os.path.join(os.path.dirname(__file__), '../data/predefined_questions.json'))

@pytest.mark.parametrize("question,expected_answer", [
    (q['question'], q['answer']) for q in PREDEFINED_QUESTIONS
])
def test_predefined_questions(question, expected_answer):
    processed_input = preprocess_input(question)
    answer = find_exact_match(processed_input, questions)
    assert answer == expected_answer

# Generate semantically similar questions
def generate_similar_questions():
    similar_questions = [
        (q['question'].replace("does", "is the purpose of"), q['answer'])
        for q in PREDEFINED_QUESTIONS if "does" in q['question']
    ]
    similar_questions.extend([
        (q['question'].replace("How does", "Explain how"), q['answer'])
        for q in PREDEFINED_QUESTIONS if q['question'].startswith("How does")
    ])
    similar_questions.extend([
        (q['question'].replace("Tell me about", "What are"), q['answer'])
        for q in PREDEFINED_QUESTIONS if q['question'].startswith("Tell me about")
    ])
    similar_questions.extend([
        (q['question'].replace("What are", "Can you list"), q['answer'])
        for q in PREDEFINED_QUESTIONS if q['question'].startswith("What are")
    ])
    return similar_questions

@pytest.mark.parametrize("similar_question,expected_answer", generate_similar_questions())
def test_semantically_similar_questions(similar_question, expected_answer):
    processed_input = preprocess_input(similar_question)
    answer = find_semantic_match(processed_input, questions, threshold=SIMILARITY_THRESHOLD)
    assert answer == expected_answer

@pytest.mark.parametrize("other_question", [
    "What's the weather like today?",
    "How does machine learning work?",
    "Tell me about healthcare technology",
])
def test_other_questions(other_question):
    processed_input = preprocess_input(other_question)
    exact_match = find_exact_match(processed_input, questions)
    semantic_match = find_semantic_match(processed_input, questions, threshold=SIMILARITY_THRESHOLD)
    
    assert exact_match is None
    assert semantic_match is None
    
    llm_answer = get_llm_response(other_question)
    assert isinstance(llm_answer, str)
    assert len(llm_answer) > 0