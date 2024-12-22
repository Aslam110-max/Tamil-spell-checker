import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class GemmaChecker:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.client = Groq(api_key=api_key)

    def get_suggestions(self, tamil_text):
        prompt = f"""
        As a Tamil language expert, analyze the following text for spelling and grammatical errors.
        Provide detailed corrections and suggestions in Tamil:

        Text to analyze: {tamil_text}

        Please provide:
        1. Corrected version of the text
        2. Specific errors found
        3. Explanation of corrections in Tamil
        """
        
        try:
            completion = self.client.chat.completions.create(
                # Using Mixtral model which is currently supported by Groq
                model="gemma2-9b-it",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a Tamil language expert who provides detailed corrections and suggestions for Tamil text."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2048
            )
            return completion.choices[0].message.content
        except Exception as e:
            if hasattr(e, 'code') and e.code == 'model_decommissioned':
                return "Error: The model is no longer supported. Please contact the administrator to update the model."
            return f"Error getting suggestions: {str(e)}"

    def check_text(self, text):
        try:
            suggestions = self.get_suggestions(text)
            if suggestions.startswith("Error"):
                return [("error", suggestions, text)]
            return [("info", suggestions, text)]
        except Exception as e:
            return [("error", f"Error checking text: {str(e)}", text)]