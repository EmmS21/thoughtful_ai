import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_llm_response(user_input):
    """
    Get a response from the OpenAI GPT model.

    Args:
        user_input (str): The user's question or input.

    Returns:
        str: The LLM's response.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  
            messages=[
                {"role": "system", "content": "You are a helpful assistant for Thoughtful AI, a company that provides AI-powered automation agents for healthcare processes."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=300,
            n=1,
            stop=None,
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in LLM response: {e}")
        return "I'm sorry, I couldn't generate a response at this time. Please try again later."