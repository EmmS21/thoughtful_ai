# Thoughtful AI Chat Assistant

## Setup

1. Add your OpenAI API key to the `.env.test` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Running the App

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the CLI version:
   ```
   python src/app.py
   ```

3. Run the Streamlit version:
   ```
   streamlit run src/app.py
   ```

## Running Tests

Execute tests using pytest:
pytest tests/test_app.py

## Test Details

The tests in `test_app.py` cover:

1. **Predefined Questions**: Checks if exact matches for predefined questions return the correct answers.

2. **Semantically Similar Questions**: Tests if slightly modified versions of predefined questions still return the expected answers using semantic matching.

3. **Other Questions**: Ensures that unrelated questions:
   - Don't match any predefined or semantic answers
   - Get a non-empty response from the LLM integration

These tests validate the core functionality of the chat assistant, including exact matching, semantic matching, and LLM fallback for handling various types of user inputs.
