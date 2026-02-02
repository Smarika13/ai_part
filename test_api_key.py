import os
from google import genai
from google.genai.errors import APIError # type: ignore

# Ensure your API key is set as an environment variable (recommended)
# or replace os.getenv('GEMINI_API_KEY') with your actual key as a string
api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    print("Error: API key not found. Please set the GEMINI_API_KEY environment variable.")
else:
    try:
        client = genai.Client(api_key=api_key)
        models = client.models.list()
        print("Success! Your API key is working. Available models:")
        for model in models:
            print(f"- {model.name}")
    except APIError as e:
        print(f"Error: Your API key is not working.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")