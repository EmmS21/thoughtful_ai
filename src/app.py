import os
import warnings
import streamlit as st
from nlp_utils import load_predefined_questions, preprocess_input, find_exact_match, find_semantic_match
from config import SIMILARITY_THRESHOLD
from llm_integration import get_llm_response

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def streamlit_app():
    questions = load_predefined_questions(os.path.join(os.path.dirname(__file__), '../data/predefined_questions.json'))
    st.sidebar.write(f"Loaded {len(questions)} predefined questions.")
    st.title("Welcome to Thoughtful AI Chat Assistant")

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    user_input = st.chat_input("You: ")

    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        processed_input = preprocess_input(user_input)
        answer = find_exact_match(processed_input, questions)
        
        if answer:
            response = f"Agent: {answer}"
        else:
            answer = find_semantic_match(processed_input, questions, threshold=SIMILARITY_THRESHOLD)
            if answer:
                response = f"Agent: {answer}"
            else:
                llm_answer = get_llm_response(user_input)
                response = f"Agent (LLM): {llm_answer}"

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def cli_app():
    questions = load_predefined_questions(os.path.join(os.path.dirname(__file__), '../data/predefined_questions.json'))
    print("Welcome! Type 'exit' to end the chat.")

    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            print("Thank you for chatting. Goodbye!")
            break

        processed_input = preprocess_input(user_input)
        answer = find_exact_match(processed_input, questions)
        
        if answer:
            print(f"Agent: {answer}")
        else:
            answer = find_semantic_match(processed_input, questions, threshold=SIMILARITY_THRESHOLD)
            if answer:
                print(f"Agent: {answer}")
            else:
                llm_answer = get_llm_response(user_input)
                print(f"Agent (LLM): {llm_answer}")
        
        print()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        streamlit_app()
    else:
        cli_app()